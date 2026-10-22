"""YARN-AGG-1: HELMET recall / RAG cells built LOCALLY (no HELMET eval.py, no inference).

Pinned source: princeton-nlp/HELMET @ aeadacc6593e9a222a84b0660b186a4e23d2efda, checked out at
``/mnt/raid0/llm/data/eval/helmet/repo``. Templates, metrics and config values below are ported from that
checkout's ``data.py`` / ``utils.py`` / ``configs/{recall,rag}.yaml`` (``validate_local_config`` re-checks the
templates against ``data.py`` text and the config keys against the yaml).

What is local and what is not
  * ``json_kv``: HELMET's data file (``data/json_kv/test_k1800_dep6.jsonl``) is not on disk, so the cell is a
    deterministic LOCAL RE-GENERATION of the Liu et al. key-value task in HELMET's prompt format, sized with the
    SERVED tokenizer. It is not HELMET's data.
  * ``ruler_niah_*``: the parent defaults to ``ruler_local``; ``build_from_data`` reads
    ``data/ruler/<task>/validation_<len>.jsonl`` for when the HELMET data exists.
  * ``nq`` / ``hotpotqa``: need HELMET's kilt retrieval files (``HelmetDataMissing`` while absent).
    128K reads ``data/kilt/<ds>-dev-multikilt_1000_k1000_dep{6|3}.jsonl`` + the ``k3`` demo file.
    240K needs ``..._k1875_...`` (k scales 1000 -> ceil(1000*245760/131072) = 1875, json_kv 1800 -> 3375); that file
    is not shipped, so when only the k1000 file exists the extra distractors are padded from other questions'
    passages (recorded as a deviation).

Metrics (utils.py): recall/json_kv/RAG all score with substring exact match (``sub_em``) or RULER recall.
Prompt: the chat-endpoint path of HELMET (``OpenAIModel.prepare_inputs``): ONE user message = ``user_template``
formatted; the ``system_template`` ("Answer:") answer prefix is not appended.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import string
import subprocess
import unicodedata  # noqa: F401  (kept for parity with HELMET utils)
import uuid
from pathlib import Path
from typing import Any, Callable

from scripts.benchmark.yarn_gate import cells

HELMET_REPO = "https://github.com/princeton-nlp/HELMET"
HELMET_PIN = "aeadacc6593e9a222a84b0660b186a4e23d2efda"
HELMET_CHECKOUT = "/mnt/raid0/llm/data/eval/helmet/repo"
DEFAULT_DATA_DIR = HELMET_CHECKOUT  # data.tar.gz extracts to ./data/... inside the checkout
DATA_TARBALL_URL = "https://huggingface.co/datasets/princeton-nlp/HELMET/resolve/main/data.tar.gz"
DATA_TARBALL_BYTES = 11271916108  # HEAD request 2026-10-22 (x-linked-size)

CELLS = {"recall": ("ruler_niah_mk_2", "ruler_niah_mk_3", "ruler_niah_mv", "json_kv"), "rag": ("nq", "hotpotqa")}
GROUP_OF = {t: g for g, ts in CELLS.items() for t in ts}
#: HELMET dataset name in the yaml `datasets:` list
HELMET_DATASET = {"ruler_niah_mk_2": "ruler_niah_mk_2", "ruler_niah_mk_3": "ruler_niah_mk_3",
                  "ruler_niah_mv": "ruler_niah_mv", "json_kv": "json_kv", "nq": "kilt_nq", "hotpotqa": "kilt_hotpotqa"}
GEN_MAX = {"ruler_niah_mk_2": 50, "ruler_niah_mk_3": 100, "ruler_niah_mv": 50, "json_kv": 100, "nq": 20, "hotpotqa": 20}
RULER_DIR = {"ruler_niah_mk_2": "niah_multikey_2", "ruler_niah_mk_3": "niah_multikey_3", "ruler_niah_mv": "niah_multivalue"}
YAML_OF_GROUP = {"recall": "configs/recall.yaml", "rag": "configs/rag.yaml"}
LEN_128K, LEN_240K = 131072, 245760
SHOTS = 2
FILL_LO = 0.95  # an item must land in [FILL_LO*target, target] real tokens

# ── templates, ported from data.py @ pin ───────────────────────────────────────────────────────────────
TEMPLATES = {
    "rag_user": "Use the given documents to write a concise and short answer to the question. Write your answer in "
                "the following format:\nAnswer: [answer]\n\n{demos}{context}\n\nQuestion: {question}",
    "rag_system": "Answer:",
    "rag_demo": "{documents}\n\nQuestion: {question}\nAnswer: {answer}",
    "rag_passage": "Document (Title: {title}): {text}",
    "kv_user": "{context}\n\nExtract the value corresponding to the specified key in the JSON object below.\n\n"
               "{demos}Key: {question}",
    "kv_system": "Corresponding value:",
    "kv_demo": "Key: {key}\nCorresponding value:{value}",
    "niah_user": "A special magic {type_needle_v} is hidden within the following text. Make sure to memorize it. I will "
                 "quiz you about the {type_needle_v} afterwards.\n{context}\nWhat is the special magic "
                 "{type_needle_v} for {query} mentioned in the provided text?",
    "niah_system": "The special magic {type_needle_v} for {query} mentioned in the provided text is",
    "niah_mv_user": "Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize "
                    "it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat are all the special "
                    "magic {type_needle_v} for {query} mentioned in the provided text?",
    "niah_mv_system": "The special magic {type_needle_v} for {query} mentioned in the provided text are",
}


class HelmetDataMissing(FileNotFoundError):
    def __init__(self, paths):
        self.paths = [str(p) for p in ([paths] if isinstance(paths, (str, Path)) else paths)]
        super().__init__("HELMET data missing (data.tar.gz is not extracted): " + ", ".join(self.paths))


class HelmetDataShort(ValueError):
    pass


# ── metrics, ported from utils.py @ pin (pure) ─────────────────────────────────────────────────────────
def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def substring_exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(ground_truth) in normalize_answer(prediction)


def _flat(answers) -> list[str]:
    if isinstance(answers, str):
        return [answers]
    if answers and isinstance(answers[0], list):
        return [a for sub in answers for a in sub]
    return list(answers)


def max_over_ground_truths(fn: Callable[[str, str], bool], prediction: str, answers) -> float:
    gts = _flat(answers)
    return max(float(fn(prediction, g)) for g in gts) if gts else 0.0


def parse_output(output: str, prefix: str = "Answer:"):
    """utils.parse_output: text after ``prefix`` on its line, else the first line."""
    for pat in (re.compile(f"(?:{prefix})(.*)(?:\n|$)", flags=re.IGNORECASE), re.compile(r"(?:^)(.*)(?:\n|$)")):
        m = pat.search(output)
        if m is not None:
            return re.sub(f"^{re.escape(prefix)}", "", m[1].strip(), flags=re.IGNORECASE).strip()
    return None


def ruler_recall(prediction: str, answers) -> float:
    """data.load_ruler post_process: fraction of gold strings that are case-insensitive substrings."""
    answers = _flat(answers)
    return sum(a.lower() in prediction.lower() for a in answers) / len(answers) if answers else 0.0


def sub_em_with_parse(prediction: str, answers, prefix: str) -> float:
    """default_post_process / load_json_kv: max of the raw and the parsed prediction."""
    best = max_over_ground_truths(substring_exact_match_score, prediction, answers)
    parsed = parse_output(prediction, prefix)
    if parsed is not None:
        best = max(best, max_over_ground_truths(substring_exact_match_score, parsed, answers))
    return best


@cells.register("helmet_sub_em")
def _score_rag(item, output, *, model):
    out = output or ""
    if (item.get("pins") or {}).get("stop_new_line"):  # HELMET stop_new_line: generation ends at the first newline
        out = out.lstrip().split("\n", 1)[0]
    return {"score": sub_em_with_parse(out, item["gold"], "Answer:"), "metric": "substring_exact_match"}


@cells.register("helmet_kv_sub_em")
def _score_kv(item, output, *, model):
    return {"score": sub_em_with_parse(output or "", item["gold"], "corresponding value:"),
            "metric": "substring_exact_match"}


@cells.register("helmet_ruler_recall")
def _score_ruler(item, output, *, model):
    return {"score": ruler_recall(output or "", item["gold"]), "metric": "ruler_recall"}


SCORER_OF = {"json_kv": "helmet_kv_sub_em", "nq": "helmet_sub_em", "hotpotqa": "helmet_sub_em",
             "ruler_niah_mk_2": "helmet_ruler_recall", "ruler_niah_mk_3": "helmet_ruler_recall",
             "ruler_niah_mv": "helmet_ruler_recall"}


# ── configs ────────────────────────────────────────────────────────────────────────────────────────────
def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _label(target_real: int) -> str:
    return f"{target_real // 1024}k"


def parse_cell(cell: str) -> tuple[str, int]:
    """``helmet_nq@240k#0003`` / ``nq@240k`` -> ("nq", 245760). 128k/240k use the exact nominal lengths."""
    base = cell.split("#")[0]
    task, _, lab = base.partition("@")
    task = task[len("helmet_"):] if task.startswith("helmet_") else task
    if task not in GROUP_OF:
        raise ValueError(f"unknown HELMET task {task!r}")
    m = re.fullmatch(r"(\d+)k", lab)
    if not m:
        raise ValueError(f"cell {cell!r}: label must look like '128k'")
    n = int(m.group(1)) * 1024
    return task, {128 * 1024: LEN_128K, 240 * 1024: LEN_240K}.get(n, n)


def _parse_yaml_flat(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        if line.strip() and not line.startswith("#") and ":" in line:
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip()
    return out


def pinned_config(group: str, checkout: str = HELMET_CHECKOUT) -> dict:
    p = Path(checkout) / YAML_OF_GROUP[group]
    return {"path": str(p), "sha256": _sha256_file(p), "values": _parse_yaml_flat(p.read_text())}


def _scale_k(k: int, target: int) -> int:
    return k if target == LEN_128K else math.ceil(k * target / LEN_128K)


def k_for(task: str, target_real: int) -> int:
    """The per-length passage / kv count HELMET would use: the 128K value scaled linearly with length."""
    base = {"nq": 1000, "hotpotqa": 1000, "json_kv": 1800}[task]
    return _scale_k(base, target_real)


def _rescale_test_files(files: str, target: int) -> str:
    parts = []
    for f in files.split(","):
        f = re.sub(r"_k(\d+)_dep", lambda m: f"_k{_scale_k(int(m.group(1)), target)}_dep", f)
        f = re.sub(r"validation_131072", f"validation_{target}", f)
        parts.append(f)
    return ",".join(parts)


def effective_config(group: str, target_real: int, checkout: str = HELMET_CHECKOUT) -> dict:
    """The config dict used at ``target_real``. 128K is the pinned yaml verbatim; any other length differs from it
    ONLY in input_max_length and the k / validation_<len> token inside test_files."""
    vals = dict(pinned_config(group, checkout)["values"])
    if target_real != LEN_128K:
        n = len(vals["input_max_length"].split(","))
        vals["input_max_length"] = ",".join([str(target_real)] * n)
        vals["test_files"] = _rescale_test_files(vals["test_files"], target_real)
    return {"helmet_config": vals, "templates": dict(TEMPLATES),
            "meta": {"pin": HELMET_PIN, "derived_from": YAML_OF_GROUP[group], "target_real": target_real,
                     "k_scale": target_real / LEN_128K, "pinned_verbatim": target_real == LEN_128K}}


def _config_text(cfg: dict) -> str:
    return json.dumps(cfg, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def local_config_name(group: str, target_real: int) -> str:
    return f"helmet_local_{_label(target_real)}_{group}.json"


def write_local_config(out_dir, cell: str) -> str:
    task, target = parse_cell(cell)
    group = GROUP_OF[task]
    p = Path(out_dir) / local_config_name(group, target)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_config_text(effective_config(group, target)), encoding="utf-8")
    return str(p)


def config_record(cell: str, *, out_dir=None, max_test_samples: int | None = None, checkout: str = HELMET_CHECKOUT) -> dict:
    """path + sha256 + max_test_samples + pin. 128K -> the pinned yaml. Other lengths -> the generated local config
    (written under ``out_dir`` when given; otherwise path is ``local:<name>`` and sha256 is that of the canonical text)."""
    task, target = parse_cell(cell)
    group = GROUP_OF[task]
    yaml_mts = int(pinned_config(group, checkout)["values"]["max_test_samples"])
    mts = yaml_mts if max_test_samples is None else max_test_samples
    if target == LEN_128K:
        pc = pinned_config(group, checkout)
        return {"path": pc["path"], "sha256": pc["sha256"], "max_test_samples": mts, "pin": HELMET_PIN,
                "kind": "pinned_yaml", "yaml_max_test_samples": yaml_mts}
    cfg = effective_config(group, target, checkout)
    if out_dir is not None:
        path = write_local_config(out_dir, cell)
        sha = _sha256_file(path)
    else:
        path = "local:" + local_config_name(group, target)
        sha = hashlib.sha256(_config_text(cfg).encode()).hexdigest()
    return {"path": path, "sha256": sha, "max_test_samples": mts, "pin": HELMET_PIN, "kind": "local_copy",
            "yaml_max_test_samples": yaml_mts}


def _git_head(checkout: str) -> str | None:
    try:
        return subprocess.run(["git", "-C", checkout, "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=20, check=True).stdout.strip()
    except Exception:
        return None


def validate_local_config(cfg: dict, group: str, *, checkout: str = HELMET_CHECKOUT, check_git: bool = True) -> list[str]:
    """Problems (empty list = valid). A local config must equal the pinned yaml except input_max_length and the
    k / validation_<len> token of test_files (scaled by target/131072), and carry templates that appear verbatim in
    the pinned data.py."""
    probs: list[str] = []
    pinned = pinned_config(group, checkout)["values"]
    meta, vals = cfg.get("meta", {}), cfg.get("helmet_config", {})
    target = meta.get("target_real")
    if meta.get("pin") != HELMET_PIN:
        probs.append(f"pin {meta.get('pin')!r} != {HELMET_PIN}")
    if check_git:
        head = _git_head(checkout)
        if head is not None and head != HELMET_PIN:
            probs.append(f"checkout HEAD {head} != pin")
    if set(vals) != set(pinned):
        probs.append(f"config keys differ: extra={sorted(set(vals) - set(pinned))} missing={sorted(set(pinned) - set(vals))}")
    if not isinstance(target, int) or target <= 0:
        probs.append("meta.target_real missing")
        return probs
    for k in sorted(set(vals) & set(pinned)):
        if k in ("input_max_length", "test_files"):
            continue
        if vals[k] != pinned[k]:
            probs.append(f"{k}: {vals[k]!r} != pinned {pinned[k]!r}")
    n = len(pinned["input_max_length"].split(","))
    if "input_max_length" in vals and vals["input_max_length"] != ",".join([str(target)] * n):
        probs.append(f"input_max_length {vals['input_max_length']!r} != {target} x{n}")
    if "test_files" in vals and vals["test_files"] != _rescale_test_files(pinned["test_files"], target):
        probs.append("test_files do not equal the pinned list with k / validation_<len> rescaled to the target")
    if "test_files" in vals and len(vals["test_files"].split(",")) != len(pinned["test_files"].split(",")):
        probs.append("test_files count differs")
    src = (Path(checkout) / "data.py").read_text()
    for name, tpl in (cfg.get("templates") or {}).items():
        if name not in TEMPLATES:
            probs.append(f"unknown template {name!r}")
        elif tpl != TEMPLATES[name]:
            probs.append(f"template {name!r} differs from the module's copy")
        if tpl.replace("\n", "\\n") not in src:
            probs.append(f"template {name!r} not found verbatim in pinned data.py")
    if set(cfg.get("templates") or {}) != set(TEMPLATES):
        probs.append("template set differs")
    return probs


# ── data files ─────────────────────────────────────────────────────────────────────────────────────────
def _data_dir(data_dir) -> Path:
    return Path(data_dir or os.environ.get("HELMET_DATA_DIR") or DEFAULT_DATA_DIR)


def needed_paths(task: str, target_real: int, data_dir=None) -> list[Path]:
    d = _data_dir(data_dir)
    if task == "json_kv":
        return []
    if task in RULER_DIR:
        return [d / "data" / "ruler" / RULER_DIR[task] / f"validation_{target_real}.jsonl"]
    dep = "dep6" if task == "nq" else "dep3"
    return [d / "data" / "kilt" / f"{task}-dev-multikilt_1000_k{k_for(task, target_real)}_{dep}.jsonl",
            d / "data" / "kilt" / f"{task}-train-multikilt_1000_k3_{dep}.jsonl"]


def available(task: str, *, data_dir=None) -> tuple[bool, str]:
    """(True, how) or (False, "<exact missing path(s)>"). Checked at 128K (the pinned config); 240K RAG additionally
    works from the k1000 file via pool padding (see module doc)."""
    if task not in GROUP_OF:
        return False, f"unknown HELMET task {task!r}"
    if task == "json_kv":
        return True, "local re-generation (no data file needed)"
    missing = [str(p) for p in needed_paths(task, LEN_128K, data_dir) if not p.exists()]
    if missing:
        return False, ", ".join(missing)
    return True, "HELMET data present"


def _read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ── counting ───────────────────────────────────────────────────────────────────────────────────────────
_TOK_CACHE: dict[str, Any] = {}


def make_counter(tokenizer_json: str | None = None) -> Callable[[str], int]:
    from tokenizers import Tokenizer
    path = tokenizer_json or cells.DEFAULT_TOKENIZER_JSON
    if path not in _TOK_CACHE:
        _TOK_CACHE[path] = Tokenizer.from_file(path)
    tok = _TOK_CACHE[path]
    return lambda text: len(tok.encode(text, add_special_tokens=False).ids)


def _tok_sha(tokenizer_json: str | None) -> str | None:
    path = tokenizer_json or cells.DEFAULT_TOKENIZER_JSON
    if path == cells.DEFAULT_TOKENIZER_JSON:
        return cells.DEFAULT_TOKENIZER_SHA256
    return _sha256_file(path) if os.path.exists(path) else None


def _rng(seed: int, *parts) -> random.Random:
    return random.Random("|".join([str(seed), *map(str, parts)]))


# ── json_kv local generator ────────────────────────────────────────────────────────────────────────────
DEPTHS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)  # assumed meaning of HELMET's "dep6": 6 gold-key depths, cycled


def _uuid(rng: random.Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


def _kv_render(pairs: list[tuple[str, str]], q_idx: int, demo_idx: tuple[int, ...]) -> tuple[str, str, str]:
    """-> (context, demos_text, question key). Pairs at demo_idx are shown as demos (also kept in the JSON)."""
    obj = {k: v for k, v in pairs}
    context = "JSON data:\n" + json.dumps(obj, indent=2)
    demos = "\n\n".join(TEMPLATES["kv_demo"].format(key=pairs[i][0], value=" " + pairs[i][1]) for i in demo_idx)
    return context, (demos + "\n\n") if demo_idx else "", pairs[q_idx][0]


def _kv_prompt(pairs, q_idx, demo_idx) -> str:
    context, demos, key = _kv_render(pairs, q_idx, demo_idx)
    return TEMPLATES["kv_user"].format(context=context, demos=demos, question=key)


def _kv_item_pairs(rng, k: int, depth: float, n_extra: list) -> tuple[list[tuple[str, str]], int, tuple[int, ...]]:
    pairs = n_extra[:k]
    q_idx = round(depth * (k - 1))
    # demos: SHOTS other pairs (never the queried one), deterministic positions
    others = [i for i in range(k) if i != q_idx]
    demo_idx = tuple(sorted(rng.sample(others, min(SHOTS, len(others)))))
    return pairs, q_idx, demo_idx


def build_json_kv(target_real: int, n: int, *, seed: int, count: Callable[[str], int]) -> list[dict]:
    label = _label(target_real)
    # calibrate tokens/pair + fixed overhead on a probe
    cal = _rng(seed, "json_kv", "cal", target_real)
    probe_pairs = [(_uuid(cal), _uuid(cal)) for _ in range(400)]
    c_small = count(_kv_prompt(probe_pairs[:100], 50, (1, 2)))
    c_big = count(_kv_prompt(probe_pairs, 200, (1, 2)))
    per_pair = max((c_big - c_small) / 300.0, 1e-6)
    overhead = c_small - 100 * per_pair
    items = []
    for i in range(n):
        rng = _rng(seed, "json_kv", label, i)
        depth = DEPTHS[i % len(DEPTHS)]
        k = max(int((target_real - overhead) / per_pair * 0.995), 3)
        pool: list[tuple[str, str]] = []
        best = None
        for _ in range(12):
            while len(pool) < k:
                pool.append((_uuid(rng), _uuid(rng)))
            sel_rng = _rng(seed, "json_kv", label, i, "demo", k)
            pairs, q_idx, demo_idx = _kv_item_pairs(sel_rng, k, depth, pool)
            text = _kv_prompt(pairs, q_idx, demo_idx)
            c = count(text)
            if c <= target_real and (best is None or c > best[0]):
                best = (c, pairs, q_idx, demo_idx, text)
            if target_real * 0.99 <= c <= target_real:
                break
            k = max(3, k + int((target_real * 0.995 - c) / per_pair) - (1 if c > target_real else 0))
        if best is None or best[0] < FILL_LO * target_real:
            raise HelmetDataShort(f"json_kv item {i}: could not land within [{FILL_LO},1] of {target_real}")
        c, pairs, q_idx, demo_idx, text = best
        context, demos, key = _kv_render(pairs, q_idx, demo_idx)
        items.append({"text": text, "gold": [pairs[q_idx][1]], "count": c, "key": key, "k": len(pairs),
                      "depth": depth, "q_idx": q_idx})
    return items


# ── item assembly ──────────────────────────────────────────────────────────────────────────────────────
def _deviations(task: str, target_real: int, extra: list[str] | None = None) -> list[str]:
    d = ["paper numbers non-comparable (different model / tokenizer-sized lengths)",
         "chat-endpoint prompt: user_template only; system_template answer prefix not appended (HELMET OpenAIModel path)",
         "lengths are real tokens of the served tokenizer.json over message content (chat-template tokens excluded)"]
    if target_real != LEN_128K:
        d.append(f"{_label(target_real)} is not a HELMET shipped length: local config copy (input_max_length/k scaled "
                 f"by {target_real}/{LEN_128K}); validate with validate_local_config")
    if task == "json_kv":
        d += ["LOCAL RE-GENERATION of the kv_retrieval task, not HELMET's data/json_kv/test_k1800_dep6.jsonl",
              "context prefix 'JSON data:' and json.dumps(indent=2) formatting follow Liu et al. as recalled; the "
              "HELMET data file was not inspected (not on disk)",
              "dep6 interpreted as 6 evenly spaced gold-key depths cycled over items",
              "demos (shots=2) are two other pairs of the same JSON object"]
    if task in ("nq", "hotpotqa"):
        d += ["max_test_samples subset drawn with a seeded RNG (HELMET uses unseeded random.sample)",
              "demos: 2 deterministic per-question draws from the k3 demo file (sha256 seed; HELMET's datasets.shuffle "
              "permutation not reproduced)",
              "passage list cut at passage granularity to fit the real-token target (HELMET truncates the text tail)",
              "stop_new_line=true applied in the scorer (first line only), not as a server stop sequence"]
    return (d + (extra or []))


def _make_item(task, label, i, target_real, n, seed, tok_sha, messages, gold, real_n, deviations, extra_pins=None,
               cell_record=None):
    item_id = f"helmet_{task}@{label}#{i:04d}"
    pins = {"helmet_repo": HELMET_REPO, "helmet_sha": HELMET_PIN, "config": cell_record,
            "max_test_samples": n, "seed": seed, "tokenizer_sha256": tok_sha, "deviations": deviations,
            "stop_new_line": task in ("nq", "hotpotqa")}
    pins.update(extra_pins or {})
    item = {"item_id": item_id, "cell": f"helmet_{task}@{label}", "suite": "helmet", "task": task,
            "context_id": item_id, "messages": messages, "max_tokens": GEN_MAX[task], "thinking": False,
            "scorer": SCORER_OF[task], "gold": gold, "nominal_len": target_real, "real_n_offline": real_n,
            "n_source": "model_tokenizer_json", "pins": pins}
    cells.validate_item(item)
    return item


def _ruler_user(task: str, row: dict) -> str:
    key = "niah_mv_user" if task == "ruler_niah_mv" else "niah_user"
    return TEMPLATES[key].format(type_needle_v=row["type_needle_v"], context=row["context"],
                                 query=row.get("query", row.get("question", "")))


def _rag_demo_text(demo_rows: list[dict], sample: dict, key: str) -> str:
    h = int(hashlib.sha256(str(sample[key]).encode("utf-8")).hexdigest(), 16) % 2**31
    pool = [d for d in demo_rows if d.get(key) != sample.get(key)]
    random.Random(h).shuffle(pool)
    seen, chosen = set(), []
    for d in pool:
        if d.get(key) in seen:
            continue
        seen.add(d.get(key))
        chosen.append(d)
        if len(chosen) == SHOTS:
            break
    parts = [TEMPLATES["rag_demo"].format(
        documents="\n\n".join(TEMPLATES["rag_passage"].format(title=c["title"], text=c["text"]) for c in d["ctxs"]),
        question=d["question"], answer=d["answers"][0]) for d in chosen]
    return "\n\n".join(parts) + "\n\n" if parts else ""


def _passage(c: dict) -> str:
    return TEMPLATES["rag_passage"].format(title=c["title"], text=c["text"])


def _build_rag_item(sample, demo_rows, other_rows, target_real, count, rng, key):
    """-> (user_text, count, padded_n) or None when the gold passage cannot be retained / item cannot fill."""
    demos = _rag_demo_text(demo_rows, sample, key)
    q = sample["question"]
    fixed = count(TEMPLATES["rag_user"].format(demos=demos, context="", question=q))
    budget = target_real - fixed
    ctxs = list(sample["ctxs"])
    per = {}

    def tc(c):
        t = _passage(c)
        if t not in per:
            per[t] = count(t + "\n\n")
        return per[t]

    padded = 0
    total = sum(tc(c) for c in ctxs)
    if total < budget * FILL_LO and other_rows:
        gold_norm = [normalize_answer(a) for a in _flat(sample["answers"])]
        extras = [c for r in other_rows if r is not sample for c in r["ctxs"]
                  if not c.get("has_answer") and not any(g and g in normalize_answer(c["text"]) for g in gold_norm)]
        rng.shuffle(extras)
        add = []
        for c in extras:
            if total >= budget * 1.02:
                break
            add.append(c)
            total += tc(c)
        padded = len(add)
        slots = sorted(rng.sample(range(len(ctxs) + len(add)), len(ctxs)))
        merged: list = [None] * (len(ctxs) + len(add))
        it_c, it_a = iter(ctxs), iter(add)
        slot_set = set(slots)
        for pos in range(len(merged)):
            merged[pos] = next(it_c) if pos in slot_set else next(it_a)
        ctxs = merged
    chosen, used = [], 0
    for c in ctxs:
        t = tc(c)
        if used + t > budget:
            break
        chosen.append(c)
        used += t
    if not any(c.get("has_answer") for c in chosen) and any(c.get("has_answer") for c in ctxs):
        return None
    while chosen:
        text = TEMPLATES["rag_user"].format(demos=demos, context="\n\n".join(_passage(c) for c in chosen), question=q)
        n_tok = count(text)
        if n_tok <= target_real:
            return (text, n_tok, padded) if n_tok >= FILL_LO * target_real else None
        chosen.pop()
    return None


def build_from_data(task: str, target_real: int, n: int, *, seed: int = 20261022, count=None, tokenizer_json=None,
                    data_dir=None, out_dir=None) -> list[dict]:
    """Items from HELMET's own data files (ruler_niah_* and nq / hotpotqa). Raises ``HelmetDataMissing``."""
    if task == "json_kv":
        raise ValueError("json_kv has no data file here; use build_cell")
    count = count or make_counter(tokenizer_json)
    label = _label(target_real)
    tok_sha = _tok_sha(tokenizer_json)
    cell = f"helmet_{task}@{label}"
    rec = config_record(cell, out_dir=out_dir, max_test_samples=n)
    items: list[dict] = []
    if task in RULER_DIR:
        (path,) = needed_paths(task, target_real, data_dir)
        if not path.exists():
            raise HelmetDataMissing([path])
        rows = _read_jsonl(path)
        _rng(seed, task, label).shuffle(rows)
        dropped = 0
        for row in rows:
            if len(items) == n:
                break
            text = _ruler_user(task, row)
            c = count(text)
            if c > target_real:
                dropped += 1
                continue
            gold = row.get("answer", row.get("outputs"))
            items.append(_make_item(task, label, len(items), target_real, n, seed, tok_sha,
                                    [{"role": "user", "content": text}], list(gold), c,
                                    _deviations(task, target_real, ["items whose real length exceeds the target are "
                                                                    "dropped, never truncated"]),
                                    {"source_file": str(path), "dropped_over_target": dropped}, rec))
        return items
    test_p, demo_p = needed_paths(task, target_real, data_dir)
    exact = test_p.exists()
    if not exact:  # fall back to the k1000 file with pool padding (only meaningful above 128K)
        test_p = needed_paths(task, LEN_128K, data_dir)[0]
    missing = [p for p in (test_p, demo_p) if not p.exists()]
    if missing:
        raise HelmetDataMissing(missing if exact or target_real == LEN_128K else
                                [needed_paths(task, target_real, data_dir)[0], *missing])
    rows, demo_rows = _read_jsonl(test_p), _read_jsonl(demo_p)
    key = "id" if rows and "id" in rows[0] else "question"
    keys = sorted({r[key] for r in rows})
    keep = set(_rng(seed, task, label, "sample").sample(keys, min(n, len(keys))))
    chosen_rows = [r for r in rows if r[key] in keep]
    chosen_rows.sort(key=lambda r: hashlib.sha256((str(seed) + str(r[key])).encode()).hexdigest())
    pad_ok = not exact
    skipped = 0
    for sample in chosen_rows:
        got = _build_rag_item(sample, demo_rows, rows if pad_ok else [], target_real, count,
                              _rng(seed, task, label, "pad", sample[key]), key)
        if got is None:
            skipped += 1
            continue
        text, c, padded = got
        extra = []
        if padded:
            extra.append(f"{padded} distractor passages padded from other questions (exact k file "
                         f"{needed_paths(task, target_real, data_dir)[0].name} absent)")
        items.append(_make_item(task, label, len(items), target_real, n, seed, tok_sha,
                                [{"role": "user", "content": text}], list(sample["answers"]), c,
                                _deviations(task, target_real, extra),
                                {"source_file": str(test_p), "demo_file": str(demo_p), "padded_passages": padded}, rec))
    if not items:
        raise HelmetDataShort(f"{task}@{label}: no item reached {FILL_LO} of {target_real} tokens "
                              f"(file {test_p.name} has too few passages; need k={k_for(task, target_real)})")
    for it in items:
        it["pins"]["skipped_unfillable_or_goldless"] = skipped
    return items


def build_cell(task: str, target_real: int, n: int, *, seed: int = 20261022, count=None, tokenizer_json=None,
               data_dir: str | None = None, out_dir=None) -> list[dict]:
    if task not in GROUP_OF:
        raise ValueError(f"unknown HELMET task {task!r}")
    if task != "json_kv":
        return build_from_data(task, target_real, n, seed=seed, count=count, tokenizer_json=tokenizer_json,
                               data_dir=data_dir, out_dir=out_dir)
    count = count or make_counter(tokenizer_json)
    label = _label(target_real)
    rec = config_record(f"helmet_{task}@{label}", out_dir=out_dir, max_test_samples=n)
    tok_sha = _tok_sha(tokenizer_json)
    out = []
    for i, g in enumerate(build_json_kv(target_real, n, seed=seed, count=count)):
        out.append(_make_item(task, label, i, target_real, n, seed, tok_sha,
                              [{"role": "user", "content": g["text"]}], g["gold"], g["count"],
                              _deviations(task, target_real),
                              {"kv_pairs": g["k"], "gold_depth": g["depth"], "k_helmet_equivalent": k_for(task, target_real)},
                              rec))
    return out
