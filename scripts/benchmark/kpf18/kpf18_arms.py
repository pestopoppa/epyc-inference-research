#!/usr/bin/env python3
"""KPF-18.1 / 18.2 / 18.3 arms for the KPF-18 gate runner (kpf18_gate.py).

handoffs/active/kv-prefix-fork-and-paged-attention.md, rows KPF-18.1 (hit-length rows A-D), KPF-18.2 (drafter
parity, per-component attribution, anti-vacuity, cold control, policy control) and KPF-18.3 (chunk-split cost,
end-checkpoint position, request lane, >=50k warm-vs-cold logprob).

This module holds everything that does not need a server:
  - the server's prefix-hash ladder and the KPF-11 fork-position PREDICTOR (what n_fork_tokens the rule says);
  - drafter-configuration derivation (`strip_speculative`), sweep/position planning, cell + component records;
  - the PREDECLARED decision rules (RULES, written into run_meta.json before any server starts) and the pure
    evaluators that turn per-sibling cells into counts;
  - argparse additions and the registry the gate wires into (`BUILDERS`, `PREFLIGHTS`).
The per-arm step builders (server launches, requests) are in the same file, below, and take the gate module as an
argument (`gate`) instead of importing it: `python3 kpf18_gate.py` runs as `__main__`, and importing it again by
name would create a second copy of its module state (CPU_MODE, TRUNKS, ...).

Record schema is unchanged: `epyc.kpf.run.v1`, one record per server launch x arm, with the per-sibling raw cells
under `cells` (n_fork_tokens as returned in the request's timings AND as read back from /slots, the predicted
value, the checkpoint set that existed before the request, errors). Decision-rule results are COUNTS; no pass word.
"""
from __future__ import annotations

import hashlib
import json
import statistics
from typing import Callable

ARM_ITEMS = (181, 182, 183)
ARM_CMDS = {"kpf181": [181], "kpf182": [182], "kpf183": [183], "kpf18arms": [181, 182, 183]}
TASK_OF = {181: "KPF-18.1", 182: "KPF-18.2", 183: "KPF-18.3"}

BUILDERS: dict[int, Callable] = {}      # item -> fn(ctx, gate) -> list[Step]   (filled by the arm sections below)
PREFLIGHTS: dict[int, Callable] = {}    # item -> fn(ctx, gate) -> (problems, notes)
RULES: dict[int, dict] = {}             # item -> {arm key: predeclared rule text}  (filled by the arm sections)

# server/prompt constants the predictor and the planners share with the runner
FNV_BASIS = 0x14650FB0739D0383         # tools/server/server-context.cpp, the /slots prefix_hash ladder
FNV_PRIME = 0x100000001B3
_M64 = (1 << 64) - 1
N_CKPT_AT_MAX = 8                       # `checkpoint_at` entries a request may pin
FLOOR_EPS = 1e-4                        # probability / logprob resolution below which two floats are "equal"


# ---------------------------------------------------------------- server prefix-hash ladder
def prefix_hashes(ids: list[int], marks) -> dict[int, str]:
    """FNV-1a 64 over the int32 little-endian token ids of ids[:m], for each requested mark m <= len(ids).
    Same constants and walk as the server's /slots `prefix_hash` ladder (the server's offset basis has one digit
    fewer than the textbook FNV value: keep it identical)."""
    want = sorted({int(m) for m in marks if 0 <= int(m) <= len(ids)})
    out: dict[int, str] = {}
    h, iw = FNV_BASIS, 0
    for i in range(len(ids) + 1):
        while iw < len(want) and want[iw] == i:
            out[i] = f"{h:016x}"
            iw += 1
        if iw >= len(want) or i == len(ids):
            break
        t = int(ids[i]) & 0xFFFFFFFF
        for b in range(4):
            h ^= (t >> (8 * b)) & 0xFF
            h = (h * FNV_PRIME) & _M64
    return out


def lcp(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


# ---------------------------------------------------------------- the KPF-11 fork-position predictor
def slot_by_id(slots: list[dict], sid: int) -> dict | None:
    for s in slots or []:
        if s.get("id") == sid:
            return s
    return None


def checkpoint_set(slot: dict | None) -> list[dict]:
    """The checkpoints /slots reports for one slot, reduced to what the rule reads."""
    out = []
    for c in (slot or {}).get("checkpoints") or []:
        out.append({"n_tokens": int(c.get("n_tokens", 0)), "pos_max": c.get("pos_max"), "pinned": bool(c.get("pinned")),
                    "prefix_hash": c.get("prefix_hash") or ""})
    return out


def predict_fork(slots: list[dict], task_ids: list[int], dst_id: int, n_min: int, *, mode: str = "checkpoint",
                 n_swa: int = 0, lcp_by_slot: dict[int, int] | None = None, enabled: bool = True) -> dict:
    """What `find_fork_source` selects for a task sent to slot `dst_id`, from a /slots snapshot taken BEFORE the request.

    KPF-11 rule, checkpoint mode (recurrent/hybrid targets): the largest p over every other slot such that p is a
    checkpoint of that slot with  0 < p < n_task,  p >= n_min,  prefix_hash(task[:p]) == the checkpoint's hash (i.e.
    p <= lcp) and pos_max + 1 == p; plus kind "end": an IDLE source whose whole prompt is a prefix of the task
    (n_src == lcp < n_task, n_src >= n_min, no SWA). Ties: idle first, then the lowest slot id (the server prefers the
    most recently used; unknown from /slots, so a tie is flagged `tie`).
    KV mode (plain-attention targets): p = min(lcp, n_task - 1) for every source with lcp >= n_min; needs the exact
    lcp, supplied as `lcp_by_slot` (the runner keeps a shadow of what it sent to each slot).
    Not modelled: LoRA mismatch, WAIT_OTHER children, kv_altered slots, a source's posmax guard in KV mode.
    """
    n_task = len(task_ids)
    res = {"p": 0, "src": None, "kind": "none", "idle": None, "tie": False, "mode": mode, "n_min": n_min,
           "n_task": n_task, "candidates": [], "enabled": bool(enabled)}
    if not enabled or n_task <= 0:
        return res
    marks = set()
    for s in slots or []:
        if s.get("id") == dst_id:
            continue
        marks.add(int(s.get("n_prompt_tokens") or 0))
        for c in s.get("checkpoints") or []:
            marks.add(int(c.get("n_tokens", 0)))
    ladder = prefix_hashes(task_ids, [m for m in marks if 0 < m <= n_task])
    cands: list[dict] = []
    for s in sorted(slots or [], key=lambda x: x.get("id", 0)):
        sid = s.get("id")
        if sid == dst_id:
            continue
        n_src = int(s.get("n_prompt_tokens") or 0)
        if n_src <= 0:
            continue
        idle = not s.get("is_processing")
        if mode == "kv":
            l_ = (lcp_by_slot or {}).get(sid, 0)
            if l_ >= n_min and l_ > 0:
                cands.append({"src": sid, "p": min(l_, n_task - 1), "kind": "kv", "idle": idle})
            continue
        if idle and n_swa == 0 and 0 < n_src < n_task and n_src >= n_min and ladder.get(n_src) == s.get("prefix_hash"):
            cands.append({"src": sid, "p": n_src, "kind": "end", "idle": idle})
        for c in s.get("checkpoints") or []:
            p = int(c.get("n_tokens", 0))
            if p <= 0 or p >= n_task or p < n_min:
                continue
            if ladder.get(p) != c.get("prefix_hash"):
                continue                                    # p > lcp: this checkpoint is not a prefix of the task
            if c.get("pos_max") is not None and int(c["pos_max"]) + 1 != p:
                continue
            cands.append({"src": sid, "p": p, "kind": "checkpoint", "idle": idle})
    res["candidates"] = cands
    if not cands:
        return res
    best_p = max(c["p"] for c in cands)
    top = [c for c in cands if c["p"] == best_p]
    top.sort(key=lambda c: (not c["idle"], c["src"]))
    res.update(p=best_p, src=top[0]["src"], kind=top[0]["kind"], idle=top[0]["idle"],
               tie=len({(c["src"], c["kind"]) for c in top}) > 1)
    return res


def components(pred: dict, lcp_client: int | None, n_task: int, server_comps: dict | None) -> dict:
    """The per-component hit lengths KPF-18.2 item 2 wants (tgt_kv_hit, recurrent_ckpt_pos, drafter_kv_hit).
    The server does not report them today. tgt_kv_hit and recurrent_ckpt_pos are DERIVED client-side (lcp of the task
    with what the chosen source holds, capped at n_task - 1; the position of the checkpoint the rule picks);
    drafter_kv_hit is read from /slots[dst].fork if the build exposes it, else None = NOT MEASURED (never a guess).
    A server-reported value overrides the client-derived one and `source` says which. See SERVER_FORK_FIELDS."""
    sc = dict(server_comps or {})
    out = {"tgt_kv_hit": None, "recurrent_ckpt_pos": None, "drafter_kv_hit": None, "source": {}}
    if lcp_client is not None:
        out["tgt_kv_hit"] = min(int(lcp_client), max(0, n_task - 1))
        out["source"]["tgt_kv_hit"] = "client"
    if pred.get("kind") in ("checkpoint", "end"):
        out["recurrent_ckpt_pos"] = int(pred["p"])
        out["source"]["recurrent_ckpt_pos"] = "client-predicted"
    for k in ("tgt_kv_hit", "recurrent_ckpt_pos", "drafter_kv_hit"):
        if sc.get(k) is not None:
            out[k] = int(sc[k])
            out["source"][k] = "server"
    out["drafter_exposed"] = sc.get("drafter_kv_hit") is not None
    return out


SERVER_FORK_FIELDS = ("tgt_kv_hit", "recurrent_ckpt_pos", "drafter_kv_hit")


def server_components(slot: dict | None) -> dict:
    f = (slot or {}).get("fork") or {}
    return {k: f.get(k) for k in SERVER_FORK_FIELDS if f.get(k) is not None}


def make_cell(*, arm: str, cell_id: str, role: str, slot: int, n_task: int, lcp_client: int | None, timings: dict | None,
              n_fork_slot: int | None, pred: dict, ckpts_before: list[dict], expected: int | None = None,
              error: str | None = None, **extra) -> dict:
    """One sibling / cell, raw: nothing is dropped, errors and abstentions are cells too."""
    t = timings or {}
    n_fork = int(t.get("n_fork_tokens") or 0)
    c = {"arm": arm, "cell": cell_id, "role": role, "slot": slot, "n_task": n_task, "lcp_client": lcp_client,
         "n_fork_tokens": None if error else n_fork, "n_fork_tokens_slot": n_fork_slot,
         "fork_src_kind": t.get("fork_src_kind"), "fork_src_slot": t.get("fork_src_slot"),
         "cache_n": t.get("cache_n"), "prompt_n": t.get("prompt_n"),
         "predicted_p": pred.get("p"), "predicted_kind": pred.get("kind"), "predicted_src": pred.get("src"),
         "predicted_tie": pred.get("tie"), "n_candidates": len(pred.get("candidates") or []),
         "expected": expected, "checkpoint_set_before": [x["n_tokens"] for x in ckpts_before],
         "pinned_before": [x["n_tokens"] for x in ckpts_before if x["pinned"]], "error": error}
    c.update(extra)
    return c


# ---------------------------------------------------------------- predeclared rules (counts, not a pass word)
def _ok(c: dict) -> bool:
    return not c.get("error") and c.get("n_fork_tokens") is not None


def tally_expected(cells: list[dict]) -> dict:
    """Cells with an `expected` n_fork_tokens: how many equal it. Errors and abstentions are counted, never dropped."""
    use = [c for c in cells if c.get("expected") is not None]
    err = [c for c in use if not _ok(c)]
    eq = [c for c in use if _ok(c) and c["n_fork_tokens"] == c["expected"]]
    return {"n": len(use), "n_error": len(err), "n_equal_expected": len(eq),
            "n_not_equal_expected": len(use) - len(err) - len(eq),
            "observed": [c.get("n_fork_tokens") for c in use], "expected": [c["expected"] for c in use],
            "predeclared_rule_satisfied": bool(use) and len(eq) == len(use)}


def tally_prediction(cells: list[dict]) -> dict:
    """Observed n_fork_tokens vs the KPF-11 predictor, plus the 'zero while a usable checkpoint exists' defect count."""
    use = [c for c in cells if c.get("predicted_p") is not None]
    ok = [c for c in use if _ok(c)]
    eq = [c for c in ok if c["n_fork_tokens"] == c["predicted_p"]]
    zero_pos = [c["cell"] for c in ok if c["n_fork_tokens"] == 0 and c["predicted_p"] > 0]
    return {"n": len(use), "n_error": len(use) - len(ok), "n_equal_predicted": len(eq),
            "n_not_equal_predicted": len(ok) - len(eq), "n_zero_while_checkpoint_usable": len(zero_pos),
            "zero_while_checkpoint_usable": zero_pos,
            "n_ties": sum(1 for c in use if c.get("predicted_tie"))}


def within_floor(vs: dict | None, floor_max_abs: float | None, floor_max_kld: float | None) -> dict:
    """vs: gate.compare() dict child-vs-fresh. First-token top-1 equal AND max-abs prob and coarse KL no larger than the
    control floor (max over the run's fresh-vs-fresh pairs), both floors lifted to FLOOR_EPS."""
    if not vs:
        return {"top1": None, "maxabs_ok": None, "kld_ok": None, "within": None}
    fa = max(floor_max_abs or 0.0, FLOOR_EPS)
    fk = max(floor_max_kld or 0.0, FLOOR_EPS)
    m, k = vs.get("maxabs_union"), vs.get("kld")
    top1 = bool(vs.get("top1"))
    mo = (m is not None) and m <= fa
    ko = (k is not None) and k <= fk
    return {"top1": top1, "maxabs_ok": mo, "kld_ok": ko, "within": bool(top1 and mo and ko),
            "maxabs": m, "kld": k, "floor_maxabs_used": fa, "floor_kld_used": fk}


def floor_stats(floors: list[dict]) -> dict:
    mu = [f["maxabs_union"] for f in floors if f and f.get("maxabs_union") is not None]
    kl = [f["kld"] for f in floors if f and f.get("kld") is not None]
    pre = sorted(f["prefix"] for f in floors if f)
    return {"n": len(floors), "max_maxabs": max(mu) if mu else None, "max_kld": max(kl) if kl else None,
            "prefix_median": statistics.median(pre) if pre else None, "prefix_min": pre[0] if pre else None}


def median_or_none(xs: list[float]):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def rules_for(items: list[int]) -> dict:
    out = {}
    for it in items:
        if it in RULES:
            out[str(it)] = {"task": TASK_OF[it], **RULES[it]}
    return out


def rules_sha256(items: list[int]) -> str | None:
    r = rules_for(items)
    if not r:
        return None
    return hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------- drafter configurations
_SPEC_EXACT = {"-md", "--model-draft", "-ngld", "--n-gpu-layers-draft", "--gpu-layers-draft", "--device-draft", "-devd",
               "-cd", "--ctx-size-draft", "-td", "--threads-draft", "-tbd", "--threads-batch-draft", "-ctkd",
               "--cache-type-k-draft", "-ctvd", "--cache-type-v-draft"}


def _is_spec_flag(t: str) -> bool:
    return t in _SPEC_EXACT or t.startswith("--spec-") or t.startswith("--draft")


def flag_value(argv: list[str], names: tuple[str, ...]) -> str | None:
    for i, t in enumerate(argv):
        if t in names and i + 1 < len(argv):
            return argv[i + 1]
        for n in names:
            if t.startswith(n + "="):
                return t.split("=", 1)[1]
    return None


def strip_speculative(argv: list[str]) -> list[str]:
    """The same argv with every speculative / draft-model flag (and its value) removed: the 'no drafter' launch."""
    out, i = [], 0
    while i < len(argv):
        t = argv[i]
        if _is_spec_flag(t):
            if "=" not in t and i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                i += 2
            else:
                i += 1
            continue
        out.append(t)
        i += 1
    return out


def config_name(argv: list[str]) -> str:
    st = flag_value(argv, ("--spec-type",))
    md = flag_value(argv, ("-md", "--model-draft"))
    if st == "draft-mtp":
        return "mtp"
    if st == "draft-dflash":
        return "dflash2"
    if st or md:
        return f"draft-{st or 'model'}"
    return "nodraft"


def parse_configs(spec: str, extra: list[str] | None, base_argv: list[str],
                  load_argv: Callable[[str], list[str]]) -> list[tuple[str, list[str]]]:
    """`--kpf-configs` (comma list of 'as-given' | 'nodraft') plus repeated `--kpf-config-argv NAME=ARGVFILE`.
    Returns [(name, argv)], the first entry being the launched shape; names are unique."""
    out: list[tuple[str, list[str]]] = []
    seen: set[str] = set()

    def add(name: str, argv: list[str]) -> None:
        if name in seen:
            return
        seen.add(name)
        out.append((name, list(argv)))
    for tok in [t.strip() for t in (spec or "as-given").split(",") if t.strip()]:
        if tok == "as-given":
            add(config_name(base_argv), base_argv)
        elif tok == "nodraft":
            add("nodraft", strip_speculative(base_argv))
        else:
            raise ValueError(f"--kpf-configs: unknown token {tok!r} (as-given | nodraft)")
    for e in extra or []:
        name, _, path = e.partition("=")
        if not name or not path:
            raise ValueError(f"--kpf-config-argv expects NAME=ARGVFILE, got {e!r}")
        add(name, load_argv(path))
    return out


def ub_from_argv(argv: list[str], default: int = 512) -> int:
    v = flag_value(argv, ("-ub", "--ubatch-size"))
    return int(v) if v and v.isdigit() else default


# ---------------------------------------------------------------- planning helpers (pure)
def sweep_lengths(t_avail: int, ub: int, n: int) -> list[int]:
    """>= n trunk lengths at a fixed stride across ONE -ub period, all <= t_avail: L0 + k*stride, k < n, stride = ub // n,
    L0 the largest multiple of ub with L0 + ub <= t_avail. [] if the trunk cannot hold them."""
    if n < 1 or ub < n or t_avail < 2 * ub:
        return []
    stride = ub // n
    l0 = ((t_avail - ub) // ub) * ub
    if l0 <= 0:
        return []
    return [l0 + k * stride for k in range(n)]


def d_positions(junction: int, n_src: int, ub: int) -> dict[str, int]:
    """The checkpoint positions in force on a source of n_src tokens whose request pinned `junction`: the junction and
    the two near-end ones the server makes ({4 + n_ubatch, 4} tokens before the prompt end)."""
    return {"junction": junction, "n_minus_4_minus_ub": n_src - 4 - ub, "n_minus_4": n_src - 4}


def d_cells(positions: dict[str, int], deltas=(-1, 0, 1)) -> list[dict]:
    return [{"name": f"{nm}{d:+d}", "position_name": nm, "c": c, "delta": d, "lcp": c + d}
            for nm, c in positions.items() for d in deltas]


def holdout_lcp(positions: dict[str, int]) -> int:
    """A shared-prefix length at least 3 tokens away from every c-1, c, c+1 cell, between the junction and the first
    near-end checkpoint, so the rule can be checked at a length that was not used to set it."""
    vals = sorted(positions.values())
    h = (vals[0] + vals[1]) // 2 + 13
    while any(abs(h - v) <= 3 for v in vals):
        h += 7
    return h


def divergent_suffix(ids_after: list[list[int]], avoid_token: int | None, take: int) -> tuple[int, list[int]] | None:
    """The first candidate suffix whose first token differs from `avoid_token` (so the shared prefix ends exactly where
    intended), cut to `take` tokens. Returns (index, ids) or None."""
    for i, s in enumerate(ids_after):
        if s and (avoid_token is None or s[0] != avoid_token):
            return i, list(s[:take])
    return None


def dense_positions(n_prompt: int, u: int) -> list[int]:
    """Dense policy: checkpoint_at every u tokens, at most N_CKPT_AT_MAX of them, all strictly inside the prompt."""
    return [u * k for k in range(1, N_CKPT_AT_MAX + 1) if u * k < n_prompt]


def classify_end_ckpt(pinned_n_tokens: list[int], n_prompt: int, n_generated: int) -> dict:
    """Where does the pinned `checkpoint_at [-1]` checkpoint sit: at the prompt end, or at the generation end?"""
    at_prompt = [n for n in pinned_n_tokens if n == n_prompt]
    at_gen = [n for n in pinned_n_tokens if n_generated > 0 and n in (n_prompt + n_generated, n_prompt + n_generated - 1)]
    other = [n for n in pinned_n_tokens if n not in at_prompt and n not in at_gen]
    where = "prompt_end" if at_prompt else ("generation_end" if at_gen else ("other" if other else "absent"))
    return {"where": where, "pinned": sorted(pinned_n_tokens), "n_prompt": n_prompt, "n_generated": n_generated,
            "at_prompt_end": bool(at_prompt), "at_generation_end": bool(at_gen), "other": other}


def multi_turn_gap(n_turn1_prompt: int, n_reply: int, n_fork_tokens: int | None) -> dict:
    need = n_turn1_prompt + n_reply
    if n_fork_tokens is None:
        return {"required": need, "n_fork_tokens": None, "gap": None, "covered": None}
    return {"required": need, "n_fork_tokens": n_fork_tokens, "gap": need - n_fork_tokens,
            "covered": n_fork_tokens >= need}


def logprob_of_chosen(entry: dict) -> float | None:
    """log-probability of the sampled token in one completion_probabilities entry."""
    if entry.get("logprob") is not None:
        return float(entry["logprob"])
    tid = entry.get("id")
    for t in entry.get("top_logprobs") or []:
        if t.get("id") == tid and t.get("logprob") is not None:
            return float(t["logprob"])
    return None


def mean_abs_dlogprob(a: dict, b: dict, upto: int) -> dict:
    """mean |logprob_a - logprob_b| of the sampled token over the first `upto` generated positions (callers pass the
    greedy prefix common to every run being compared, so both sides share the context at every position)."""
    pa, pb = a.get("probs") or [], b.get("probs") or []
    ds = []
    for i in range(min(upto, len(pa), len(pb))):
        x, y = logprob_of_chosen(pa[i]), logprob_of_chosen(pb[i])
        if x is not None and y is not None:
            ds.append(abs(x - y))
    return {"n_positions": len(ds), "mean_abs_dlogprob": (sum(ds) / len(ds)) if ds else None,
            "max_abs_dlogprob": max(ds) if ds else None}


def common_prefix_len(runs: list[dict]) -> int:
    toks = [r.get("tokens") or [] for r in runs]
    if not toks:
        return 0
    n = min(len(t) for t in toks)
    i = 0
    while i < n and all(t[i] == toks[0][i] for t in toks):
        i += 1
    return i


# ---------------------------------------------------------------- CLI
def add_arguments(ap) -> None:
    g = ap.add_argument_group("KPF-18.1/.2/.3 arms (commands kpf181 | kpf182 | kpf183 | kpf18arms)")
    g.add_argument("--kpf-configs", default="as-given,nodraft",
                   help="drafter launch configurations: as-given (the argv file) and/or nodraft (it with every "
                        "speculative flag stripped)")
    g.add_argument("--kpf-config-argv", action="append", default=[], metavar="NAME=ARGVFILE",
                   help="an extra drafter configuration from its own argv file (e.g. dflash2=..., mtp=...)")
    g.add_argument("--kpf-n-swa", type=int, default=0, help="sliding-window width of the target (0 = none); the "
                   "predictor's end-kind source needs 0")
    g.add_argument("--kpf181-arms", default="A,B,C,D")
    g.add_argument("--kpf181-groups", default=None, help="trunk groups for arms A/B/D (default 0)")
    g.add_argument("--kpf181-sibs", type=int, default=3)
    g.add_argument("--kpf181-lengths", type=int, default=16, help="arm C: lengths across one -ub period (>= 16)")
    g.add_argument("--kpf181-c-group", default=None, help="arm C trunk group (default: the longest group <= 16k, else 0)")
    g.add_argument("--kpf182-parts", default="1,2,3,4,5")
    g.add_argument("--kpf182-groups", default=None, help="trunk groups for the parity/attribution protocol (default 0,2)")
    g.add_argument("--kpf182-sibs", type=int, default=3)
    g.add_argument("--kpf182-dense-u", type=int, default=1024, help="policy control: dense checkpoint spacing U")
    g.add_argument("--kpf183-parts", default=None, help="1 cost, 2 end-ckpt position + multi-turn, 3 lane, 4 >=50k "
                   "warm-vs-cold (default: GPU 1,4 ; --cpu 2,3)")
    g.add_argument("--kpf183-groups", default=None, help="part 1 trunk groups (default 0,3)")
    g.add_argument("--kpf183-reps", type=int, default=5, help="part 1 repeats per arm (A B A B ...)")
    g.add_argument("--kpf183-reply", type=int, default=64, help="part 2 turn-1 reply length")
    g.add_argument("--kpf183-long-group", default=None, help="part 4 trunk group (default: the first with T >= 50000)")
    g.add_argument("--kpf183-chat-probe-tokens", type=int, default=1500)


def groups_arg(spec: str | None, default: list[int]) -> list[int]:
    if not spec:
        return list(default)
    return [int(x) for x in spec.split(",") if x.strip() != ""]


# ---------------------------------------------------------------- wiring into the gate
def register(items: dict, gate) -> None:
    """Add the arm builders to the gate's ITEMS table. `gate` is the gate module object (never re-imported by name)."""
    for it, fn in BUILDERS.items():
        items[it] = (lambda ctx, fn=fn: fn(ctx, gate))


def preflight_arms(ctx, items: list[int], gate) -> tuple[list[str], list[str]]:
    probs: list[str] = []
    notes: list[str] = []
    for it in items:
        if it in PREFLIGHTS:
            p, n = PREFLIGHTS[it](ctx, gate)
            probs += p
            notes += n
    return probs, notes
