#!/usr/bin/env python3
"""INF-59 E0/E1 runner: YaRN context extension on Qwen3.8-27B (operator-approved 2026-10-04).

Talks ONLY to a dedicated single-slot scratch llama-server (default :18083). It starts, stops and
reloads nothing; launching and killing the arms is RUNBOOK.md's job (launch_arm.sh / kill_arm.sh).

Modes
  --build-inputs   CPU only. Freezes every input under inputs/ with a sha256 manifest:
                   the neutral haystack (GovReport documents, ZeroSCROLLS test split), 20 key->value
                   needles, the stage cut points, and the short-text regression suite (24-prompt
                   production mix + 30 GSM8K + 30 MMLU from question_pool.jsonl, with ground truth).
  --dry-run        Validates the frozen inputs WITHOUT inference and without contacting any server:
                   manifest hashes, needle uniqueness/absence in the haystack, needle depths, per-arm
                   context budgets, truth rows vs the coherence_gate graders, a coherence_gate
                   self-test, and the time estimate. Exit 0 = ready.
  --arm A0|A1|A2 --i-own-the-window
                   RUN against the scratch server: preflight (props, slots, log proof lines, KFD),
                   smoke chain (prefix reuse must work before any long prefill), short suite,
                   chained needle stages. Every call is persisted as it completes (calls.jsonl).
  --gate --a0 DIR --a1 DIR [--a2 DIR]
                   Offline. Applies the PRE-REGISTERED criteria (RUNBOOK.md section 6) with the
                   shared coherence_gate (tier 0 identity, tier 1 needle/gsm8k/mmlu graders +
                   degeneracy.v2 on real token ids). Writes gate/verdict.json and gate/VERDICT.md.

Haystack design (why it is not the Q38-T7 one): Q38-T7's 80k needle was planted in AutoKernel agent
contexts, so the model read the planted fact as an injected instruction and abstained. Here the
haystack is plain expository prose (US government reports), each needle is a declarative sentence
in the same register about a fictional entity, and the question is asked as an ordinary reading-
comprehension request in its OWN user turn.

Prefix reuse on a hybrid (GDN) model: llama-server can only roll the recurrent state back to a
context checkpoint, and v10 creates one at the start of the LAST user message
(tools/server/server-context.cpp:3640-3700). Q38-T7 put document+question in one user message, so
its follow-up questions re-prefilled from token 154. Here every document segment is its own user
turn followed by a fixed assistant acknowledgement, and the question is the last user turn, so:
  * the 5 questions of a stage each re-process only the question (~60 tokens);
  * the next stage reuses the whole previous context and prefills only its new segment
    (128k -> 240k -> 400k -> 500k is ONE 500k prefill in total, not four).
The runner VERIFIES both from timings.prompt_n/cache_n and aborts the chain instead of silently
paying a full re-prefill.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from yarn_proof import check_proof

HERE = Path(__file__).resolve().parent
INPUTS = HERE / "inputs"
RESULTS = Path(os.environ.get("YARN_E1_RESULTS", HERE / "results"))   # selftests point these elsewhere
GATE_DIR = Path(os.environ.get("YARN_E1_GATE", HERE / "gate"))
CG_DEFAULT = HERE / "vendor" / "scripts" / "lib"          # coherence_gate @ 95157ad7 (feat/coherence-gate-ec)
CG_COMMIT_FILE = HERE / "vendor" / "COMMIT"
GOVREPORT = Path("/mnt/raid0/llm/data/eval/zeroscrolls/gov_report/gov_report/test.jsonl")
POOL = Path("/mnt/raid0/llm/epyc-inference-research/benchmarks/prompts/question_pool.jsonl")
PROD_MIX = Path("/mnt/raid0/llm/tmp/inf70/agents/e3-alpha/prompts.json")
HF_TOKENIZER = Path("/mnt/raid0/llm/models/turboderp/Qwen3.8-27B-exl3-4.00bpw/tokenizer.json")
MODEL = "/mnt/raid0/llm/models/Qwen3.8-27B-Q8_0.gguf"
KFD_PROC = Path("/sys/class/kfd/kfd/proc")
VRAM_CARD = Path("/sys/class/drm/card2/device/mem_info_vram_used")
GIB = 1024 ** 3
SEED = "inf59-yarn-e1-20261004"
PRODUCTION_PORTS = {8000, 8070, 8074, 8080, 8083, 8100, 8180}

# Stage = cumulative haystack CONTENT tokens (HF tokenizer). Totals land ~0.3-0.8k above these after
# template, needles, acknowledgements and the question; A0's last stage must stay < 262,144.
STAGES = {"128k": 127_000, "240k": 238_500, "400k": 398_000, "500k": 498_000}
STAGE_ORDER = ["128k", "240k", "400k", "500k"]
DEPTHS = [0.10, 0.30, 0.50, 0.70, 0.90]
ARMS = {
    # n_ctx expected from /props and /slots; stages each arm runs; YaRN proof expected in the log.
    "A0": {"n_ctx": 262_144, "stages": ["128k", "240k"], "yarn": False,
           "what": "native RoPE, -c 262144 (production geometry, lean single-slot shape)"},
    "A1": {"n_ctx": 524_288, "stages": ["128k", "240k", "400k", "500k"], "yarn": True,
           "what": "YaRN f2: --rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144 + context_length override 524288"},
    "A0Q4": {"n_ctx": 262_144, "stages": ["128k", "240k"], "yarn": False,
             "what": "1M option (a) pre-test: A0 with -ctk q4_0 -ctv q4_0 (native RoPE), paired against A0"},
    "A2": {"n_ctx": 524_288, "stages": ["128k", "240k", "400k"], "yarn": False,
           "what": "negative control: context_length override 524288, NO rope flags (raw extrapolation past 262k)"},
}
SMOKE = {"s1": 3_000, "s2": 6_000}     # smoke chain: 2 tiny stages, 2 needles each, before any long prefill
ACK = "Noted. I have read this part of the collection."
DOC_HEAD = ("The following is part {k} of a collection of public reports, provided as background reading. "
            "Read it; questions about the collection will follow.\n\n")
Q_HEAD = "Using only the reports provided above, answer the question. Reply with the answer only, no explanation.\n\nQuestion: "
SHORT_MAX_TOKENS = {"prod": 200, "gsm8k": 640, "mmlu": 32}
NEEDLE_MAX_TOKENS = 48

# 20 neutral facts about fictional entities: 10 code-valued, 10 name-valued. {v} is the value.
NEEDLE_FACTS = [
    ("code", "The survey vessel Marlowe Tern was entered in the coastal register under hull number {v}.",
     "Under what hull number was the survey vessel Marlowe Tern entered in the coastal register?"),
    ("name", "The 1994 orchid survey of the Varanne Valley was led by the botanist {v}.",
     "Which botanist led the 1994 orchid survey of the Varanne Valley?"),
    ("code", "The weather station on Brenlow Island reports under the station identifier {v}.",
     "What station identifier does the weather station on Brenlow Island report under?"),
    ("name", "The first harbour master appointed at Quellhaven was {v}.",
     "Who was the first harbour master appointed at Quellhaven?"),
    ("code", "The restored Ostwick tram line uses the catalogue reference {v} in the county transport archive.",
     "What catalogue reference does the restored Ostwick tram line use in the county transport archive?"),
    ("name", "The glass mosaic in the Telford Reading Room was designed by {v}.",
     "Who designed the glass mosaic in the Telford Reading Room?"),
    ("code", "The seed bank vault at Harrowmere stores the Dalvik barley collection under accession code {v}.",
     "Under what accession code does the seed bank vault at Harrowmere store the Dalvik barley collection?"),
    ("name", "The Corrin Ridge observatory was founded by the astronomer {v}.",
     "Which astronomer founded the Corrin Ridge observatory?"),
    ("code", "The ferry route between Pellwick and Saltmarsh Point is listed in the regional timetable as route {v}.",
     "What route is the ferry between Pellwick and Saltmarsh Point listed as in the regional timetable?"),
    ("name", "The Ambleford choir festival was directed for its first decade by {v}.",
     "Who directed the Ambleford choir festival for its first decade?"),
    ("code", "The Kesteren aqueduct appears in the heritage inventory with the record number {v}.",
     "What record number does the Kesteren aqueduct have in the heritage inventory?"),
    ("name", "The lighthouse at Gannet Head was kept for thirty years by {v}.",
     "Who kept the lighthouse at Gannet Head for thirty years?"),
    ("code", "The Morrow Fen drainage pump is tagged in the maintenance ledger as asset {v}.",
     "What asset tag does the Morrow Fen drainage pump have in the maintenance ledger?"),
    ("name", "The Linnet Street community library was opened in 1962 by {v}.",
     "Who opened the Linnet Street community library in 1962?"),
    ("code", "The Thornbury hill fort excavation archive is shelved under box code {v}.",
     "Under what box code is the Thornbury hill fort excavation archive shelved?"),
    ("name", "The cartographer who completed the Orlin coastal atlas was {v}.",
     "Which cartographer completed the Orlin coastal atlas?"),
    ("code", "The Wexmoor wind farm is connected to the grid through substation {v}.",
     "Through which substation is the Wexmoor wind farm connected to the grid?"),
    ("name", "The Halden river ferry was operated for most of the 1950s by {v}.",
     "Who operated the Halden river ferry for most of the 1950s?"),
    ("code", "The Sorrel Bay oyster beds are licensed under fishery permit {v}.",
     "Under what fishery permit are the Sorrel Bay oyster beds licensed?"),
    ("name", "The stained glass of St Brannoc's chapel was restored in 1988 by {v}.",
     "Who restored the stained glass of St Brannoc's chapel in 1988?"),
]
FIRST = ["Ottilie", "Bramwell", "Isaura", "Teodor", "Wilhelmina", "Casimir", "Philippa", "Leopold", "Rosalind",
         "Evander", "Marisol", "Aurelio", "Henrike", "Lucian", "Gwendolyn", "Ignatius", "Seraphine", "Florian",
         "Annelise", "Tobiah"]
LAST = ["Varga-Lindqvist", "Okonjo-Hale", "Brekkestad", "Marchetti-Rowe", "Dunmore-Achebe", "Szalai-Pryce",
        "Quennell-Ibsen", "Rautavaara-Lee", "Castellane-Oduya", "Wrexford-Nakamura", "Albrecht-Mensah",
        "Halloran-Sato", "Penhaligon-Ruiz", "Ostrowski-Grey", "Fairweather-Kolb", "Iversen-Aldana",
        "Tremaine-Oyelaran", "Vasquez-Holm", "Delacroix-Byrne", "Ashdown-Kaminski"]


# ------------------------------------------------------------------------------ small utils
def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str) + "\n")
    tmp.replace(p)


def write_jsonl(p: Path, rows) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").split("\n") if x.strip()]


def hrank(s: str) -> str:
    return hashlib.sha256(f"{SEED}:{s}".encode()).hexdigest()


class Sink:
    """Append-and-flush per record: persist per call, never only at the end."""

    def __init__(self, p: Path):
        p.parent.mkdir(parents=True, exist_ok=True)
        self.p = p

    def add(self, rec: dict) -> None:
        with open(self.p, "a") as fh:
            fh.write(json.dumps({"at": now_iso(), **rec}, ensure_ascii=False, default=str) + "\n")
            fh.flush()


def load_cg(path: Path):
    sys.path.insert(0, str(path))
    import coherence_gate as CG  # noqa: E402
    return CG


def hf_tokenizer():
    try:
        from tokenizers import Tokenizer
    except ImportError:
        return None
    return Tokenizer.from_file(str(HF_TOKENIZER))


# ------------------------------------------------------------------------------ build inputs
def build_inputs(args) -> int:
    tok = hf_tokenizer()
    if tok is None:
        print("build-inputs needs the `tokenizers` package: run with the orchestrator venv python", file=sys.stderr)
        return 2
    INPUTS.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    # --- haystack: GovReport documents, deterministic order, instruction header stripped
    docs = []
    for line in GOVREPORT.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        d = json.loads(line)
        body = d["input"][d["document_start_index"]:d["document_end_index"]].strip()
        body = re.sub(r"\n{3,}", "\n\n", body)
        if len(body) > 2000:
            docs.append((hrank(d["id"]), d["id"], body))
    docs.sort()
    target_tokens = STAGES["500k"] + 8_000
    parts, ids, total = [], [], 0
    for _, did, body in docs:
        txt = f"Report {len(parts) + 1}\n\n{body}\n\n"
        n = len(tok.encode(txt, add_special_tokens=False).ids)
        parts.append(txt)
        ids.append(did)
        total += n
        if total >= target_tokens:
            break
    corpus = "".join(parts)
    enc = tok.encode(corpus, add_special_tokens=False)
    offs = enc.offsets
    n_corpus = len(enc.ids)
    print(f"corpus: {len(parts)} GovReport documents, {len(corpus):,} chars, {n_corpus:,} tokens (HF)")

    def char_at_token(t: int) -> int:
        return offs[min(t, n_corpus - 1)][0]

    def para_boundary_after(c: int) -> int:
        j = corpus.find("\n\n", c)
        return (j + 2) if j >= 0 else c

    # --- needle values: codes and names, none of which may occur in the corpus
    values = []
    names = [f"{f} {l}" for f, l in zip(FIRST, LAST)]
    rng.shuffle(names)
    ni = 0
    for kind, _, _ in NEEDLE_FACTS:
        if kind == "name":
            v = names[ni]
            ni += 1
        else:
            while True:
                v = f"{rng.choice('BCDFGHJKLMNPRSTVWXZ')}{rng.choice('BCDFGHJKLMNPRSTVWXZ')}-{rng.randint(1000, 9999)}-{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}"
                if v not in corpus:
                    break
        values.append(v)

    # --- needle placement: one needle per (stage, depth), at d x stage content tokens
    plan = []
    k = 0
    for st in STAGE_ORDER:
        for d in DEPTHS:
            t = int(d * STAGES[st])
            plan.append({"needle": f"N{k:02d}", "stage": st, "depth_target": d, "corpus_token": t})
            k += 1
    # cut points (char) of the stage segments at paragraph boundaries
    cuts = {st: para_boundary_after(char_at_token(STAGES[st])) for st in STAGE_ORDER}
    insert_at = []
    for p, (kind, stmt, q), v in zip(plan, NEEDLE_FACTS, values):
        c = para_boundary_after(char_at_token(p["corpus_token"]))
        while c in [x[0] for x in insert_at] or c in cuts.values():
            c = para_boundary_after(c + 1)
        insert_at.append((c, p["needle"]))
        p.update({"kind": kind, "statement": stmt.format(v=v), "question": q, "expected": [v], "char": c})

    # --- segments with the needles spliced in (each needle its own paragraph)
    bounds = [0] + [cuts[s] for s in STAGE_ORDER]
    by_char = {p["char"]: p for p in plan}
    segments = {}
    for i, st in enumerate(STAGE_ORDER):
        lo, hi = bounds[i], bounds[i + 1]
        pieces, last = [], lo
        for c in sorted(x for x in by_char if lo <= x < hi):
            pieces.append(corpus[last:c])
            pieces.append(by_char[c]["statement"] + "\n\n")
            by_char[c]["segment"] = st
            last = c
        pieces.append(corpus[last:hi])
        seg = "".join(pieces)
        segments[st] = seg
        (INPUTS / f"segment_{st}.txt").write_text(seg, encoding="utf-8")
    # smoke chain: tiny, from documents NOT in the main corpus
    rest = "".join(f"Report S{j + 1}\n\n{b}\n\n" for j, (_, _, b) in enumerate(docs[len(parts):len(parts) + 4]))
    renc = tok.encode(rest, add_special_tokens=False)
    smoke_plan, smoke_bounds = [], [0]
    for st in ("s1", "s2"):
        smoke_bounds.append(rest.find("\n\n", renc.offsets[SMOKE[st]][0]) + 2)
    for j, st in enumerate(("s1", "s2")):
        seg = rest[smoke_bounds[j]:smoke_bounds[j + 1]]
        for m, d in enumerate((0.3, 0.7)):
            code = f"QX-{rng.randint(1000, 9999)}-{'S' if m == 0 else 'T'}"
            stmt = f"The smoke-test depot at {['Arnhill', 'Bexcombe', 'Cradley', 'Dunsop'][2 * j + m]} is filed under depot code {code}."
            q = f"Under what depot code is the smoke-test depot at {['Arnhill', 'Bexcombe', 'Cradley', 'Dunsop'][2 * j + m]} filed?"
            c = seg.find("\n\n", int(d * len(seg))) + 2
            seg = seg[:c] + stmt + "\n\n" + seg[c:]
            smoke_plan.append({"needle": f"S{2 * j + m}", "stage": st, "depth_target": d, "kind": "code",
                               "statement": stmt, "question": q, "expected": [code]})
        (INPUTS / f"segment_{st}.txt").write_text(seg, encoding="utf-8")

    write_jsonl(INPUTS / "needles.jsonl", plan + smoke_plan)
    meta = {"corpus_documents": ids, "corpus_source": str(GOVREPORT), "corpus_sha256": sha256_bytes(corpus.encode()),
            "hf_tokenizer": str(HF_TOKENIZER), "hf_tokenizer_sha256": sha256_file(HF_TOKENIZER),
            "stages_content_tokens": STAGES, "depths": DEPTHS, "seed": SEED,
            "doc_head": DOC_HEAD, "ack": ACK, "q_head": Q_HEAD, "needle_max_tokens": NEEDLE_MAX_TOKENS}
    write_json(INPUTS / "haystack_meta.json", meta)

    # --- short suite: production mix (24) + GSM8K (30) + MMLU (30), ground truth from question_pool
    prod = json.loads(PROD_MIX.read_text())
    prod_ids = {p["id"] for p in prod}
    pool_rows: dict[str, dict] = {}
    gsm, mmlu = [], {}
    for line in POOL.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        d = json.loads(line)
        i = d.get("id")
        if not i:
            continue
        if i in prod_ids:
            pool_rows[i] = d
        elif i.startswith("gsm8k_") and d.get("suite") == "math":
            gsm.append(d)
        elif d.get("suite") == "general" and i.startswith("mmlu_") and d.get("scoring_method") == "multiple_choice":
            subj = i.rsplit("_", 1)[0]
            if subj not in mmlu or hrank(i) < hrank(mmlu[subj]["id"]):
                mmlu[subj] = d
    gsm = sorted(gsm, key=lambda d: hrank(d["id"]))[:30]
    mm = sorted(mmlu.values(), key=lambda d: hrank(d["id"]))[:30]
    items, truth = [], []
    for p in prod:
        items.append({"id": p["id"], "set": "prod", "suite": p["suite"], "prompt": p["prompt"],
                      "max_tokens": SHORT_MAX_TOKENS["prod"]})
        g = pool_rows.get(p["id"])
        if g and g.get("expected") not in (None, ""):
            truth.append({"id": p["id"], "expected": g["expected"], "scoring_method": g.get("scoring_method"),
                          "scoring_config": g.get("scoring_config")})
    for d in gsm:
        items.append({"id": d["id"], "set": "gsm8k", "suite": "math", "prompt": d["prompt"],
                      "max_tokens": SHORT_MAX_TOKENS["gsm8k"]})
        truth.append({"id": d["id"], "expected": d["expected"], "grader": "gsm8k"})
    for d in mm:
        items.append({"id": d["id"], "set": "mmlu", "suite": "general", "prompt": d["prompt"],
                      "max_tokens": SHORT_MAX_TOKENS["mmlu"]})
        truth.append({"id": d["id"], "expected": d["expected"], "grader": "mmlu"})
    write_jsonl(INPUTS / "short_items.jsonl", items)
    write_jsonl(INPUTS / "short_truth.jsonl", truth)

    manifest = {"built_at": now_iso(), "builder": str(Path(__file__).resolve()),
                "builder_sha256": sha256_file(Path(__file__)),
                "files": {p.name: sha256_file(p) for p in sorted(INPUTS.glob("*")) if p.name != "manifest.json"}}
    write_json(INPUTS / "manifest.json", manifest)
    print(f"short suite: {len(items)} items ({sum(1 for i in items if i['set'] == 'prod')} prod, {len(gsm)} gsm8k, "
          f"{len(mm)} mmlu); truth rows {len(truth)}")
    print(f"needles: {len(plan)} main + {len(smoke_plan)} smoke; manifest -> {INPUTS / 'manifest.json'}")
    return 0


# ------------------------------------------------------------------------------ inputs (shared)
def load_inputs(verify: bool = True) -> dict:
    man = json.loads((INPUTS / "manifest.json").read_text())
    bad = []
    if verify:
        for name, h in man["files"].items():
            p = INPUTS / name
            if not p.exists() or sha256_file(p) != h:
                bad.append(name)
    needles = read_jsonl(INPUTS / "needles.jsonl")
    segs = {st: (INPUTS / f"segment_{st}.txt").read_text(encoding="utf-8") for st in STAGE_ORDER + ["s1", "s2"]}
    return {"manifest": man, "bad": bad, "needles": needles, "segments": segs,
            "meta": json.loads((INPUTS / "haystack_meta.json").read_text()),
            "short_items": read_jsonl(INPUTS / "short_items.jsonl"),
            "short_truth": read_jsonl(INPUTS / "short_truth.jsonl")}


def chain_messages(segs: dict, stages: list[str], question: str) -> list[dict]:
    msgs = []
    for k, st in enumerate(stages, 1):
        msgs.append({"role": "user", "content": DOC_HEAD.format(k=k) + segs[st]})
        msgs.append({"role": "assistant", "content": ACK})
    msgs.append({"role": "user", "content": Q_HEAD + question})
    return msgs


def stage_chain(stage: str) -> list[str]:
    order = ["s1", "s2"] if stage.startswith("s") else STAGE_ORDER
    return order[: order.index(stage) + 1]


# ------------------------------------------------------------------------------ time model
def prefill_s(L: float, model: str) -> float:
    """Cold prefill seconds for L tokens on the MI210 27B Q8_0 (no drafter).
    F: ASSESSMENT.md section 6 fit to Q38-T7 (554 t/s @16k, 439 t/s @80k): 1.7 ms*L + 7.5e-9 s*L^2.
    M: KVU-19a micro-bench quadratic (+0.052 s per 1k occupied cells per 2048-token chunk ->
       1.27e-8 s*L^2) on a 1.6 ms/token linear term; ~15% above measured at 80k: an upper bound."""
    return 1.7e-3 * L + 7.5e-9 * L * L if model == "F" else 1.6e-3 * L + 1.27e-8 * L * L


def decode_ms(n: float) -> float:
    """No-draft decode ms/token at n occupied cells, single sequence (X0 probe fit 2k..80k:
    29.61/25.78/21.56/18.49 tok/s): 33.25 ms + 0.260 us/cell (FA vec micro-bench: 0.241 us/cell/16 layers)."""
    return 33.25 + 0.260e-3 * n


def estimate(arm: str, need_tokens: dict, short_n: int) -> dict:
    out = {}
    for model in ("F", "M"):
        tot = 180.0 + 60.0  # load (27 GB --no-mmap) + smoke
        tot += short_n * 8.0  # short suite ~ 8 s/item average at ~30 tok/s
        last = 0
        for st in ARMS[arm]["stages"]:
            L = need_tokens[st]
            tot += prefill_s(L, model) - prefill_s(last, model)
            tot += 5 * (2.0 + NEEDLE_MAX_TOKENS * decode_ms(L) / 1000)  # 5 questions, worst case full answer
            tot += 2.0 + LOCAL_MAX_TOKENS * decode_ms(L) / 1000  # local-coherence probe
            last = L
        out[model] = round(tot / 60, 1)
    return out


# ------------------------------------------------------------------------------ dry run
def dry_run(args) -> int:
    problems, notes = [], []
    for p in (GOVREPORT, POOL, PROD_MIX, HF_TOKENIZER, Path(MODEL), INPUTS / "manifest.json"):
        if not p.exists():
            problems.append(f"missing: {p}")
    if problems:
        return report_dry(problems, notes, {})
    inp = load_inputs()
    if inp["bad"]:
        problems.append(f"inputs changed since build (sha256 mismatch): {inp['bad']}")
    segs, needles = inp["segments"], inp["needles"]
    full = "".join(segs[s] for s in STAGE_ORDER)
    # needle sanity: value appears exactly once in the whole haystack, key phrase only in its needle
    for n in needles:
        hay = full if not n["stage"].startswith("s") else segs["s1"] + segs["s2"]
        v = n["expected"][0]
        if hay.count(v) != 1:
            problems.append(f"{n['needle']}: value {v!r} occurs {hay.count(v)}x in the haystack (want 1)")
        if n["statement"] not in segs[n["stage"] if n["stage"].startswith("s") else n.get("segment", "")]:
            problems.append(f"{n['needle']}: statement not found in its segment {n.get('segment')}")
        if any(w in n["statement"].lower() for w in ("secret", "password", "vault code", "ignore", "instruction")):
            problems.append(f"{n['needle']}: statement uses injection-shaped vocabulary")
    tok = hf_tokenizer()
    counts, depths = {}, []
    if tok is None:
        notes.append("no `tokenizers` module: token counts estimated at 4.2 chars/token (use the orchestrator venv for exact)")
    cg = load_cg(Path(args.cg_path))
    for chain_end in STAGE_ORDER + ["s2"]:
        stages = stage_chain(chain_end)
        msgs = chain_messages(segs, stages, "x")
        text = "".join(m["content"] for m in msgs)
        counts[chain_end] = (len(tok.encode(text, add_special_tokens=False).ids) if tok else int(len(text) / 4.2)) + 40 * len(stages)
    if tok:
        # exact token position of every needle within its chain (content-only; template adds ~20 tok/turn)
        for n in needles:
            if n["stage"].startswith("s"):
                continue
            st = n["stage"]
            stages = stage_chain(st)
            pre = "".join(DOC_HEAD.format(k=k) + segs[s] for k, s in enumerate(stages, 1))
            pos = pre.find(n["statement"])
            tpos = len(tok.encode(pre[:pos], add_special_tokens=False).ids) + 40 * (stages.index(n["segment"]) + 1)
            depths.append({"needle": n["needle"], "stage": st, "target": n["depth_target"],
                           "token_pos": tpos, "depth": round(tpos / counts[st], 3)})
            if abs(tpos / counts[st] - n["depth_target"]) > 0.03:
                problems.append(f"{n['needle']}: depth {tpos / counts[st]:.3f} vs target {n['depth_target']}")
    for arm, a in ARMS.items():
        for st in a["stages"]:
            if counts[st] + NEEDLE_MAX_TOKENS + 64 > a["n_ctx"]:
                problems.append(f"{arm} stage {st}: ~{counts[st]} tokens does not fit n_ctx {a['n_ctx']}")
    # truth rows vs graders, and a coherence_gate self-test on synthetic rows (real-looking ids)
    tmap = {t["id"]: t for t in inp["short_truth"]}
    graded = collections.Counter()
    for it in inp["short_items"]:
        g = cg.grader_for(tmap.get(it["id"]))
        graded[f"{it['set']}:{g}"] += 1
        if it["set"] in ("gsm8k", "mmlu") and g is None:
            problems.append(f"{it['id']}: no grader for its truth row")
    t1 = {"id": "x", "expected": ["KT-4821-M"], "grader": "needle", "case_sensitive": False}
    ok_rows = [{"id": "x", "text": "KT-4821-M", "token_ids": [42, 7, 1999, 31]}]
    bad_rows = [{"id": "x", "text": "I cannot find that in the reports.", "token_ids": [40, 4157, 1477, 429, 304, 279, 6682, 13]}]
    rep = cg.evaluate(ok_rows, bad_rows, [t1])
    if rep["aggregate"]["gate"] != "FAIL" or rep["items"][0]["verdict"] != "REGRESSION":
        problems.append(f"coherence_gate self-test: correct->wrong needle not a REGRESSION: {rep['aggregate']}")
    rep2 = cg.evaluate(ok_rows, ok_rows, [t1])
    if rep2["aggregate"]["gate"] != "PASS":
        problems.append("coherence_gate self-test: identical rows not PASS")
    for arm in ARMS:
        notes.append(f"estimate {arm}: {estimate(arm, counts, len(inp['short_items']))} min (F = assessment fit, M = micro-bench upper bound)")
    plan = {"counts_tokens_hf": counts, "needle_depths": depths, "short_grader_map": dict(graded),
            "coherence_gate": {"path": str(args.cg_path), "library": cg.LIBRARY_VERSION,
                               "commit": CG_COMMIT_FILE.read_text().strip() if CG_COMMIT_FILE.exists() else None}}
    return report_dry(problems, notes, plan)


def report_dry(problems: list, notes: list, plan: dict) -> int:
    print(json.dumps(plan, indent=1)[:6000])
    for n in notes:
        print("NOTE", n)
    for p in problems:
        print("PROBLEM", p)
    print("DRY-RUN", "OK" if not problems else f"FAILED ({len(problems)} problems)")
    return 0 if not problems else 1


# ------------------------------------------------------------------------------ server I/O
def http_json(url: str, body: dict | None = None, timeout: float = 30.0) -> tuple[int, object]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, {"raw": raw[:500].decode(errors="replace")}
    except Exception as e:  # noqa: BLE001 - recorded, never raised mid-run
        return 0, {"error": f"{type(e).__name__}: {e}"}


def tokenize(base: str, text: str) -> list[int]:
    st, body = http_json(f"{base}/tokenize", {"content": text}, timeout=600)
    return list((body or {}).get("tokens") or []) if st == 200 else []


def port_pid(port: int) -> int | None:
    try:
        out = subprocess.run(["ss", "-ltnp", f"sport = :{port}"], capture_output=True, text=True, timeout=5).stdout
    except Exception:  # noqa: BLE001
        return None
    m = re.search(r"pid=(\d+)", out)
    return int(m.group(1)) if m else None


def kfd_gib(pid: int | None) -> float | None:
    if pid is None or not (KFD_PROC / str(pid)).exists():
        return None
    tot = 0
    for f in (KFD_PROC / str(pid)).glob("vram_*"):
        try:
            tot += int(f.read_text())
        except (OSError, ValueError):
            pass
    return round(tot / GIB, 3)


def chat(base: str, messages: list[dict], max_tokens: int, cache_prompt: bool, timeout: float) -> dict:
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0.0, "top_k": 1,
            "cache_prompt": cache_prompt, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    st, resp = http_json(f"{base}/v1/chat/completions", body, timeout=timeout)
    wall = round(time.time() - t0, 3)
    ok = st == 200 and isinstance(resp, dict) and bool(resp.get("choices"))
    ch = resp["choices"][0] if ok else {}
    text = ((ch.get("message") or {}).get("content") or "") if ok else ""
    reasoning = ((ch.get("message") or {}).get("reasoning_content") or "") if ok else ""
    t = (resp or {}).get("timings") or {} if isinstance(resp, dict) else {}
    return {"http": st, "http_ok": bool(ok), "wall_s": wall, "text": text, "reasoning": reasoning,
            "finish": ch.get("finish_reason") if ok else None,
            "error": None if ok else str(resp)[:800],
            "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"), "prompt_ms": t.get("prompt_ms"),
            "predicted_n": t.get("predicted_n"), "predicted_ms": t.get("predicted_ms"),
            "decode_tps": round(t["predicted_per_second"], 3) if t.get("predicted_per_second") else None,
            "prefill_tps": round(t["prompt_per_second"], 1) if t.get("prompt_per_second") else None}


# Proof lines (exact v10 strings; see RUNBOOK.md section 3). Each arm must show all of its patterns.
PROOF = {
    # print_info (model hparams; YaRN shows here ONLY via the rope.scaling.* metadata overrides, source H-7),
    # llama_context (effective cparams), llama_kv_cache (16 layers, q8_0, cells), server slot init.
    "A0": [r"print_info: n_ctx_train\s*=\s*262144\b", r"print_info: freq_scale_train\s*=\s*1\s*$",
           r"llama_context: freq_scale\s*=\s*1\s*$", r"llama_context: n_ctx_seq\s*=\s*262144\b",
           r"llama_kv_cache: size =\s*8704\.00 MiB \(\s*262144 cells,\s*16 layers",
           r"n_slots = 1, n_ctx_slot = 262144\b"],
    "A1": [r"Using metadata override \(\s*int\) 'qwen35\.context_length' = 524288",
           r"Using metadata override \(\s*str\) 'qwen35\.rope\.scaling\.type' = yarn",
           r"print_info: n_ctx_train\s*=\s*524288\b", r"print_info: rope scaling\s*=\s*yarn",
           r"print_info: freq_scale_train\s*=\s*0\.5\s*$", r"print_info: n_ctx_orig_yarn\s*=\s*262144\b",
           r"llama_context: freq_scale\s*=\s*0\.5\s*$", r"llama_context: n_ctx_seq\s*=\s*524288\b",
           r"llama_kv_cache: size =\s*17408\.00 MiB \(\s*524288 cells,\s*16 layers",
           r"n_slots = 1, n_ctx_slot = 524288\b"],
    "A0Q4": [r"print_info: n_ctx_train\s*=\s*262144\b", r"llama_context: freq_scale\s*=\s*1\s*$",
             r"llama_kv_cache: size =\s*4608\.00 MiB \(\s*262144 cells,\s*16 layers.*K \(q4_0\)",
             r"n_slots = 1, n_ctx_slot = 262144\b"],
    "A2": [r"Using metadata override \(\s*int\) 'qwen35\.context_length' = 524288",
           r"print_info: n_ctx_train\s*=\s*524288\b", r"print_info: freq_scale_train\s*=\s*1\s*$",
           r"llama_context: freq_scale\s*=\s*1\s*$", r"llama_context: n_ctx_seq\s*=\s*524288\b",
           r"llama_kv_cache: size =\s*17408\.00 MiB \(\s*524288 cells,\s*16 layers",
           r"n_slots = 1, n_ctx_slot = 524288\b"],
}
FORBID = [r"exceeds the training context of the model .* capping", r"failed to allocate", r"out of memory",
          r"CUDA error", r"HIP error", r"Bad metadata override type", r"possible training context overflow"]
# A local-coherence probe per stage (research note section 3 item 4): state collapse / numerics damage
# LOCAL prediction; RoPE OOD damages only long-range retrieval. Graded by degeneracy.v2 and paired.
LOCAL_PROBE = ("Without quoting it, describe in two or three plain sentences what the last report above is about.")
LOCAL_MAX_TOKENS = 160


def preflight(args, base: str) -> dict:
    pf: dict = {"at": now_iso(), "port": args.port, "arm": args.arm}
    pid = port_pid(args.port)
    pf["pid"] = pid
    pf["prod_8083_listening"] = port_pid(8083) is not None
    st, props = http_json(f"{base}/props", timeout=10)
    pf["props_http"] = st
    props = props if isinstance(props, dict) else {}
    dgs = props.get("default_generation_settings") or {}
    pf["n_ctx_props"] = dgs.get("n_ctx") or props.get("n_ctx")
    pf["model_path"] = props.get("model_path")
    pf["build_info"] = props.get("build_info")
    pf["speculative"] = {k: v for k, v in (dgs.get("params") or {}).items() if "specul" in k}
    st, slots = http_json(f"{base}/slots", timeout=10)
    pf["n_slots"] = len(slots) if isinstance(slots, list) else None
    pf["n_ctx_slot"] = (slots[0].get("n_ctx") if isinstance(slots, list) and slots else None)
    pf["kfd_gib"] = kfd_gib(pid)
    if pid:
        try:
            env = dict(kv.split("=", 1) for kv in Path(f"/proc/{pid}/environ").read_bytes().decode(errors="replace").split("\0") if "=" in kv)
            pf["env"] = {k: env.get(k) for k in ("LD_LIBRARY_PATH", "GGML_CUDA_GDN_STATE_BF16", "LLAMA_SERVER_SLOTS_DEBUG",
                                                 "GGML_CUDA_FA_MASK_SKIP", "HIP_VISIBLE_DEVICES")}
        except OSError:
            pf["env"] = None
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
            pf["argv"] = [a for a in cmd if a]
            maps = Path(f"/proc/{pid}/maps").read_text()
            pf["ggml_libs"] = sorted(set(re.findall(r"(/\S*libggml[^ /]*\.so\S*)", maps)))
        except OSError:
            pass
    probs = []
    if pf["prod_8083_listening"]:
        probs.append(":8083 is listening: production must be STOPPED for this arm (VRAM)")
    if pf["n_slots"] != 1:
        probs.append(f"expected exactly 1 slot, got {pf['n_slots']}")
    want = ARMS[args.arm]["n_ctx"]
    if pf["n_ctx_slot"] != want:
        probs.append(f"slot n_ctx {pf['n_ctx_slot']} != {want} for {args.arm}")
    argv = " ".join(pf.get("argv") or [])
    if "-md " in argv or "--spec-type" in argv:
        probs.append("a drafter is attached: E1 arms run with no drafter")
    if args.arm == "A1" and "--rope-scaling yarn" not in argv:
        probs.append("A1 argv lacks --rope-scaling yarn")
    if args.arm != "A1" and "--rope-scaling" in argv:
        probs.append(f"{args.arm} argv carries --rope-scaling")
    if "kv_slots/architect_critic" in argv:
        probs.append("argv points at the PRODUCTION slot-save path; use the scratch path")
    if pf["kfd_gib"] is None:
        probs.append("server PID is not a KFD process: no GPU residency")
    if (pf.get("env") or {}).get("GGML_CUDA_GDN_STATE_BF16") not in (None, "0"):
        probs.append("GGML_CUDA_GDN_STATE_BF16 is set: the GDN state must stay f32 (production precision)")
    if (pf.get("env") or {}).get("LLAMA_SERVER_SLOTS_DEBUG"):
        probs.append("LLAMA_SERVER_SLOTS_DEBUG is set")
    if not any("gpu-20260921-ffc1bac82" in x for x in pf.get("ggml_libs") or []):
        probs.append(f"libggml not from the v10 GPU store: {pf.get('ggml_libs')}")
    if args.server_log:
        log = Path(args.server_log).read_text(errors="replace")
        checked = check_proof(log, PROOF[args.arm], FORBID)
        pf.update({k: checked[k] for k in ("proof", "forbidden")})
        probs += [f"proof line missing: /{p}/" for p, ok in pf["proof"].items() if not ok]
        probs += [f"forbidden log line /{p}/ x{n}" for p, n in pf["forbidden"].items() if n]
    else:
        probs.append("--server-log not given: YaRN proof lines cannot be checked")
    pf["problems"] = probs
    return pf


# ------------------------------------------------------------------------------ run
def answer_row(base: str, rid: str, r: dict, extra: dict) -> dict:
    ids = tokenize(base, r["text"]) if (r["http_ok"] and r["text"]) else []
    return {"id": rid, "text": r["text"], "token_ids": ids, "finish": r["finish"], "http_ok": r["http_ok"],
            "sha256": hashlib.sha256(r["text"].encode()).hexdigest(), **extra,
            "timings": {k: r[k] for k in ("wall_s", "prompt_n", "cache_n", "prompt_ms", "predicted_n",
                                          "decode_tps", "prefill_tps")},
            "reasoning": r["reasoning"], "error": r["error"]}


def run_chain(base: str, out: Path, sink: Sink, segs: dict, needles: list[dict], stages: list[str],
              arm: str, abort_on_reuse_fail: bool, pid: int | None) -> dict:
    """Stages in order; per stage the 5 needle questions on the cached prefix. Verifies reuse."""
    summary = {}
    seg_tok = {st: len(tokenize(base, DOC_HEAD.format(k=1) + segs[st])) for st in stages}
    for si, st in enumerate(stages):
        chain = stage_chain(st)
        qs = [n for n in needles if n["stage"] == st]
        rows = []
        for qi, n in enumerate(qs):
            msgs = chain_messages(segs, chain, n["question"])
            r = chat(base, msgs, NEEDLE_MAX_TOKENS, cache_prompt=True, timeout=6 * 3600)
            row = answer_row(base, n["needle"], r, {"stage": st, "arm": arm, "question": n["question"],
                                                     "expected": n["expected"], "depth_target": n["depth_target"],
                                                     "q_index": qi, "kfd_gib": kfd_gib(pid),
                                                     "prompt_sha256": hashlib.sha256(json.dumps(msgs).encode()).hexdigest()})
            sink.add({"kind": "needle", **row})
            rows.append(row)
            # reuse checks: the first question of a stage may prefill this stage's new segment only;
            # every later question only the question itself.
            allow = (seg_tok[st] + 4096) if (qi == 0 and si > 0) else (10 ** 9 if qi == 0 else 4096)
            pn = r.get("prompt_n") or 0
            print(json.dumps({"arm": arm, "stage": st, "q": qi, "needle": n["needle"], "prompt_n": pn,
                              "cache_n": r.get("cache_n"), "prompt_ms": r.get("prompt_ms"),
                              "answer": r["text"][:80], "expected": n["expected"], "kfd_gib": row["kfd_gib"]}), flush=True)
            if not r["http_ok"]:
                summary[st] = {"aborted": f"HTTP {r['http']}: {r['error']}"}
                write_jsonl(out / f"needle_{st}.jsonl", rows)
                return summary
            if pn > allow:
                msg = (f"prefix reuse FAILED at {st} q{qi}: prompt_n {pn} > allowed {allow} "
                       f"(cache_n {r.get('cache_n')}); a full re-prefill per question is not affordable")
                print("ABORT" if abort_on_reuse_fail else "WARN", msg, flush=True)
                if abort_on_reuse_fail:
                    summary[st] = {"aborted": msg}
                    write_jsonl(out / f"needle_{st}.jsonl", rows)
                    return summary
        write_jsonl(out / f"needle_{st}.jsonl", rows)
        ctx = (rows[0]["timings"]["prompt_n"] or 0) + (rows[0]["timings"]["cache_n"] or 0)
        # local-coherence probe on the same cached prefix (state collapse / numerics vs RoPE reach)
        msgs = chain_messages(segs, chain, LOCAL_PROBE)
        r = chat(base, msgs, LOCAL_MAX_TOKENS, cache_prompt=True, timeout=6 * 3600)
        lrow = answer_row(base, f"L_{st}", r, {"stage": st, "arm": arm, "question": LOCAL_PROBE,
                                                "kfd_gib": kfd_gib(pid)})
        sink.add({"kind": "local", **lrow})
        write_jsonl(out / f"local_{st}.jsonl", [lrow])
        print(json.dumps({"arm": arm, "stage": st, "local_probe": r["text"][:120], "prompt_n": r.get("prompt_n"),
                          "finish": r["finish"]}), flush=True)
        summary[st] = {"n": len(rows), "context_tokens": ctx,
                       "empty_answers": sum(1 for x in rows if not x["text"].strip()),
                       "first_prefill_s": round((rows[0]["timings"]["prompt_ms"] or 0) / 1000, 1)}
        if summary[st]["empty_answers"] >= 3:
            print(f"WARN {arm} {st}: {summary[st]['empty_answers']}/5 empty answers: the llama.cpp #27756 "
                  f"instant-EOS signature (GDN numerics, not RoPE). See RUNBOOK.md section 7.", flush=True)
    return summary


def run(args) -> int:
    if not args.i_own_the_window:
        print("RUN mode needs --i-own-the-window (the :8083 window is granted to the main session)", file=sys.stderr)
        return 2
    if args.port in PRODUCTION_PORTS:
        print(f"REFUSING: port {args.port} is a production port; E1 runs on a scratch instance", file=sys.stderr)
        return 2
    base = f"http://127.0.0.1:{args.port}"
    inp = load_inputs()
    if inp["bad"]:
        print(f"REFUSING: inputs changed since build: {inp['bad']}", file=sys.stderr)
        return 2
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = RESULTS / args.arm / ts
    out.mkdir(parents=True, exist_ok=True)
    sink = Sink(out / "calls.jsonl")
    pf = preflight(args, base)
    write_json(out / "preflight.json", pf)
    print(json.dumps({k: pf.get(k) for k in ("pid", "n_ctx_slot", "n_slots", "kfd_gib", "proof", "problems")}, indent=1))
    if pf["problems"] and not args.force_preflight:
        print("REFUSING: preflight problems (see preflight.json); --force-preflight overrides", file=sys.stderr)
        return 3
    report: dict = {"arm": args.arm, "what": ARMS[args.arm]["what"], "started_at": now_iso(), "out": str(out),
                    "inputs_manifest_sha256": sha256_file(INPUTS / "manifest.json"), "preflight": pf}
    suites = args.suites.split(",")
    # 1) smoke chain: prefix reuse MUST work before any long prefill
    if "smoke" in suites:
        sm = run_chain(base, out, sink, inp["segments"], inp["needles"], ["s1", "s2"], args.arm, True, pf["pid"])
        report["smoke"] = sm
        write_json(out / "report.partial.json", report)
        if any("aborted" in v for v in sm.values()):
            print("SMOKE FAILED: not starting long stages", file=sys.stderr)
            report["finished_at"] = now_iso()
            write_json(out / "report.json", report)
            return 4
    # 2) short-text regression suite (paired across arms offline by --gate)
    if "short" in suites:
        rows = []
        for it in inp["short_items"]:
            r = chat(base, [{"role": "user", "content": it["prompt"]}], it["max_tokens"], cache_prompt=False, timeout=1800)
            row = answer_row(base, it["id"], r, {"set": it["set"], "suite": it["suite"], "arm": args.arm,
                                                 "prompt": it["prompt"], "max_tokens": it["max_tokens"]})
            sink.add({"kind": "short", **row})
            rows.append(row)
            print(json.dumps({"arm": args.arm, "id": it["id"], "finish": r["finish"], "n": r.get("predicted_n"),
                              "decode_tps": r.get("decode_tps"), "head": r["text"][:60]}), flush=True)
        write_jsonl(out / "short.jsonl", rows)
        report["short"] = {"n": len(rows), "http_errors": sum(1 for r in rows if not r["http_ok"])}
        write_json(out / "report.partial.json", report)
    # 3) chained needle stages
    if "needle" in suites:
        stages = [s for s in ARMS[args.arm]["stages"] if not args.stages or s in args.stages.split(",")]
        report["needle"] = run_chain(base, out, sink, inp["segments"], inp["needles"], stages, args.arm,
                                     not args.no_abort_on_reuse_fail, pf["pid"])
    report["kfd_gib_end"] = kfd_gib(pf["pid"])
    report["finished_at"] = now_iso()
    write_json(out / "report.json", report)
    print(f"done -> {out}")
    return 0


# ------------------------------------------------------------------------------ gate (offline)
def gate(args) -> int:
    CG = load_cg(Path(args.cg_path))
    inp = load_inputs(verify=False)
    truth_n = {n["needle"]: {"id": n["needle"], "expected": n["expected"], "grader": "needle", "case_sensitive": False}
               for n in inp["needles"]}
    dirs = {"A0": Path(args.a0) if args.a0 else None, "A1": Path(args.a1) if args.a1 else None,
            "A2": Path(args.a2) if args.a2 else None, "A0Q4": Path(args.a0q4) if args.a0q4 else None}

    def rows(arm: str, name: str) -> list[dict] | None:
        d = dirs.get(arm)
        p = d / name if d else None
        return read_jsonl(p) if p and p.exists() else None

    v: dict = {"schema": "inf59.yarn_e1.gate.v1", "at": now_iso(), "dirs": {k: str(x) for k, x in dirs.items()},
               "coherence_gate": CG.LIBRARY_VERSION, "criteria": {}}
    md = ["# INF-59 E1 verdict (pre-registered criteria, RUNBOOK.md section 6)", ""]
    # C1 short context
    b, c = rows("A0", "short.jsonl"), rows("A1", "short.jsonl")
    if b and c:
        rep = CG.evaluate(b, c, inp["short_truth"], anchor={"arm": "A0 native -c 262144", "binary_version": "10303"})
        write_json(GATE_DIR / "short_A1_vs_A0.json", rep)
        items = rep["items"]
        imp = sum(1 for i in items if (i.get("answer_check") or {}).get("base") and (i.get("answer_check") or {}).get("cand")
                  and i["answer_check"]["base"]["status"] != "correct" and i["answer_check"]["cand"]["status"] == "correct")
        acc = {arm: sum(1 for i in items if (i.get("answer_check") or {}).get(k) and i["answer_check"][k]["status"] == "correct")
               for arm, k in (("A0", "base"), ("A1", "cand"))}
        ngr = sum(1 for i in items if i.get("answer_check"))
        agg = rep["aggregate"]
        v["criteria"]["C1_short"] = {"coherence_gate": agg["gate"], "regressions": agg["regressions"],
                                     "improvements": imp, "graded_correct": acc, "graded_n": ngr,
                                     "identical": agg["identical"], "n": agg["n"], "needs_review": agg["needs_review"],
                                     "pass": agg["regressions"] == 0,
                                     "rule": "PASS iff 0 REGRESSION (tier 0/1). Informational: YaRN is a separate mode either way."}
        md += [f"## C1 short context (A1 vs A0, n={agg['n']})", "",
               f"- coherence_gate **{agg['gate']}**: {agg['regressions']} REGRESSION, {imp} improvement(s), "
               f"{agg['identical']} byte-identical, {agg['both_bad']} BOTH_BAD, {agg['needs_review']} NEEDS_REVIEW (no judge).",
               f"- graded accuracy: A0 {acc['A0']}/{ngr}, A1 {acc['A1']}/{ngr}."]
        md += [f"  - {x}" for x in agg.get("gate_reasons", [])[:20]]
    # C2 within native: paired needles at 128k and 240k
    tot = {"A0": 0, "A1": 0}
    paired_ok = True
    for st in ("128k", "240k"):
        b, c = rows("A0", f"needle_{st}.jsonl"), rows("A1", f"needle_{st}.jsonl")
        if not (b and c):
            paired_ok = False
            continue
        rep = CG.evaluate(b, c, list(truth_n.values()))
        write_json(GATE_DIR / f"needle_{st}_A1_vs_A0.json", rep)
        for arm, rr in (("A0", b), ("A1", c)):
            tot[arm] += sum(1 for r in rr if CG.grade(r["text"], truth_n[r["id"]])["status"] == "correct")
        degen = [i["id"] for i in rep["items"] if CG.severity(i["degeneracy"]["cand"]) > 0]
        v["criteria"].setdefault("C2_native", {})[st] = {"coherence_gate": rep["aggregate"]["gate"],
                                                        "regressions": rep["aggregate"]["regressions"],
                                                        "A1_degenerate": degen}
    if paired_ok:
        degen_any = any(x["A1_degenerate"] for x in v["criteria"]["C2_native"].values() if isinstance(x, dict))
        ok = tot["A1"] >= tot["A0"] - 1 and not degen_any
        v["criteria"]["C2_native"].update({"correct": tot, "pass": ok,
                                           "rule": "A1 correct >= A0 correct - 1 over the 10 paired needles, and no A1 degeneracy"})
        md += ["", f"## C2 within native (128k + 240k, paired): A0 {tot['A0']}/10, A1 {tot['A1']}/10 -> **{'PASS' if ok else 'FAIL'}**"]
    # C3 beyond native: A1 >= 4/5 at 400k and 500k; A2 must be clearly worse if run
    c3 = {}
    for st in ("400k", "500k"):
        c = rows("A1", f"needle_{st}.jsonl")
        if c:
            k = sum(1 for r in c if CG.grade(r["text"], truth_n[r["id"]])["status"] == "correct")
            ab = sum(1 for r in c if CG.grade(r["text"], truth_n[r["id"]]).get("abstained"))
            deg = [r["id"] for r in c if CG.severity(CG.classify(r["text"], r["token_ids"], r.get("finish"))) > 0]
            c3[st] = {"A1_correct": k, "n": len(c), "abstained": ab, "A1_degenerate": deg,
                      # relative distance from the query is the RoPE-vs-state discriminator (research note s3):
                      # far needles (> 262k back) failing alone = attention reach; near ones failing too = prefix/state
                      "depth_pattern": {r["id"]: {"depth": r.get("depth_target"),
                                                  "distance_k": round((1 - r.get("depth_target", 0)) * STAGES[st] / 1000),
                                                  "status": CG.grade(r["text"], truth_n[r["id"]])["status"]} for r in c},
                      "pass": k >= 4 and not deg}
    a2 = rows("A2", "needle_400k.jsonl")
    if a2:
        k2 = sum(1 for r in a2 if CG.grade(r["text"], truth_n[r["id"]])["status"] == "correct")
        c3["A2_400k_correct"] = k2
        if "400k" in c3:
            c3["discriminating"] = k2 <= c3["400k"]["A1_correct"] - 2
    if c3:
        c3["pass"] = all(c3[s]["pass"] for s in ("400k", "500k") if s in c3) and c3.get("discriminating", True)
        c3["rule"] = "A1 >= 4/5 at 400k AND at 500k, no degeneracy; if A2 ran, A2 <= A1 - 2 at 400k (else not discriminating)"
        v["criteria"]["C3_beyond"] = c3
        md += ["", "## C3 beyond native (A1)"] + [f"- {s}: {c3[s]['A1_correct']}/{c3[s]['n']} (abstained {c3[s]['abstained']}) "
                                                   f"depths {c3[s]['depth_pattern']}" for s in ("400k", "500k") if s in c3]
        if "A2_400k_correct" in c3:
            md.append(f"- A2 (no YaRN) 400k: {c3['A2_400k_correct']}/5; discriminating: {c3.get('discriminating')}")
    # Diagnostics (not pass/fail): local-coherence probes and instant-EOS (#27756) counts per arm and stage
    diag: dict = {"local_probe": {}, "empty_answers": {}}
    for arm in ("A0", "A1", "A2"):
        for st in STAGE_ORDER:
            lr = rows(arm, f"local_{st}.jsonl")
            if lr:
                cls = CG.classify(lr[0]["text"], lr[0]["token_ids"], lr[0].get("finish"))
                diag["local_probe"][f"{arm}:{st}"] = {"cls": cls["cls"], "reasons": cls.get("reasons"),
                                                      "text": lr[0]["text"][:400]}
            nr = rows(arm, f"needle_{st}.jsonl")
            if nr:
                diag["empty_answers"][f"{arm}:{st}"] = sum(1 for r in nr if not r["text"].strip())
    for st in ("128k", "240k"):
        b, c = rows("A0", f"local_{st}.jsonl"), rows("A1", f"local_{st}.jsonl")
        if b and c:
            rep = CG.evaluate(b, c, None)
            diag["local_probe"][f"paired:{st}"] = {"verdict": rep["items"][0]["verdict"], "reasons": rep["items"][0]["reasons"]}
    v["diagnostics"] = diag
    md += ["", "## Diagnostics", f"- local-coherence probes: { {k: x.get('cls', x.get('verdict')) for k, x in diag['local_probe'].items()} }",
           f"- empty needle answers (instant-EOS signature, llama.cpp #27756): {diag['empty_answers']}"]
    # C5 (1M option a pre-test, informational): q4_0 KV vs q8_0 KV at native length, paired
    c5 = {}
    b, c = rows("A0", "short.jsonl"), rows("A0Q4", "short.jsonl")
    if b and c:
        r5 = CG.evaluate(b, c, inp["short_truth"])
        write_json(GATE_DIR / "short_A0Q4_vs_A0.json", r5)
        c5["short"] = {k: r5["aggregate"][k] for k in ("gate", "regressions", "identical", "both_bad", "needs_review", "n")}
    nq = {}
    for st in ("128k", "240k"):
        b, c = rows("A0", f"needle_{st}.jsonl"), rows("A0Q4", f"needle_{st}.jsonl")
        if b and c:
            nq[st] = {arm: sum(1 for r in rr if CG.grade(r["text"], truth_n[r["id"]])["status"] == "correct")
                      for arm, rr in (("A0", b), ("A0Q4", c))}
    if nq:
        c5["needles"] = nq
    if c5:
        tq = sum(x["A0Q4"] for x in nq.values()) if nq else None
        t8 = sum(x["A0"] for x in nq.values()) if nq else None
        c5["pass"] = (c5.get("short", {}).get("regressions", 0) == 0) and (tq is None or tq >= t8 - 1)
        c5["rule"] = "q4_0 KV: 0 short-suite REGRESSION vs q8_0 KV and needles >= q8_0 - 1 (10 paired); gates option (a) only"
        v["criteria"]["C5_q4kv_native"] = c5
        md += ["", f"## C5 q4_0 KV vs q8_0 KV at native (option a pre-test): {c5}"]
    # A2 inertness: A2 vs A0 at 128k/240k should be tier-0 identical (override alone changes nothing)
    for st in ("128k", "240k"):
        b, c = rows("A0", f"needle_{st}.jsonl"), rows("A2", f"needle_{st}.jsonl")
        if b and c:
            rep = CG.evaluate(b, c, list(truth_n.values()))
            v["criteria"].setdefault("A2_inert_vs_A0", {})[st] = {"identical": rep["aggregate"]["identical"],
                                                                 "n": rep["aggregate"]["n"],
                                                                 "regressions": rep["aggregate"]["regressions"]}
    # C4 memory: KFD peak from the sampler summaries
    peaks = {}
    for arm, d in dirs.items():
        if d:
            for s in sorted(d.glob("../../kfd_*.summary.json")) + sorted(d.glob("kfd*.summary.json")):
                try:
                    peaks[f"{arm}:{s.name}"] = json.loads(s.read_text()).get("kfd_peak_gib")
                except (OSError, ValueError):
                    pass
    if args.kfd:
        for s in args.kfd:
            peaks[Path(s).name] = json.loads(Path(s).read_text()).get("kfd_peak_gib")
    if peaks:
        v["criteria"]["C4_memory"] = {"peaks_gib": peaks, "pass": all(p is not None and p <= 55.0 for p in peaks.values()),
                                      "rule": "KFD own-PID peak <= 55 GiB for every arm"}
        md += ["", f"## C4 memory: {peaks}"]
    crit = v["criteria"]
    core = [crit.get(k, {}).get("pass") for k in ("C2_native", "C3_beyond", "C4_memory")]
    v["overall"] = ("PASS" if all(x is True for x in core) else "FAIL" if any(x is False for x in core) else "INCOMPLETE")
    md.insert(2, f"**Overall: {v['overall']}** (C2 + C3 + C4 decide; C1 is recorded, informational).\n")
    write_json(GATE_DIR / "verdict.json", v)
    (GATE_DIR / "VERDICT.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0 if v["overall"] == "PASS" else 1


# ------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build-inputs", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--arm", choices=sorted(ARMS))
    mode.add_argument("--gate", action="store_true")
    ap.add_argument("--port", type=int, default=18083)
    ap.add_argument("--i-own-the-window", action="store_true")
    ap.add_argument("--server-log", help="the scratch server's log (launch_arm.sh writes results/<arm>.server.log)")
    ap.add_argument("--suites", default="smoke,short,needle")
    ap.add_argument("--stages", help="subset of the arm's stages, comma-separated (chain order is kept)")
    ap.add_argument("--no-abort-on-reuse-fail", action="store_true")
    ap.add_argument("--force-preflight", action="store_true")
    ap.add_argument("--cg-path", default=str(CG_DEFAULT), help="dir containing the coherence_gate package")
    ap.add_argument("--a0")
    ap.add_argument("--a1")
    ap.add_argument("--a2")
    ap.add_argument("--a0q4")
    ap.add_argument("--kfd", nargs="*", help="kfd_sampler summary json files for C4")
    a = ap.parse_args()
    if a.build_inputs:
        return build_inputs(a)
    if a.dry_run:
        return dry_run(a)
    if a.gate:
        return gate(a)
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
