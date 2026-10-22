"""YARN-AGG-1: Oolong-synth cells from the LOCAL fetched subset (no HuggingFace, no ``datasets``).

``oolong_adapter.load_rows`` loads from the HF hub; the gate window must not touch the network, so this
wrapper reads the JSONL files ``fetch_oolong_subset.py`` wrote (``oolong_synth_test_<nominal>.jsonl``,
revision ``81d2f8d9…``, the labelled column already excluded at fetch time) and reuses the adapter for
everything else: the labels-blind column guard, ``group_by_context``, ``build_messages`` (context first,
question last) and the UPSTREAM vendored scorer (``score_item``) — nothing is re-implemented.

Real-token sizes come from the offline re-measurement (``token_report.json`` beside the data, served
model's tokenizer.json) and are only a planning hint: zones are labelled from the server's prompt_n.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from scripts.benchmark import oolong_adapter as oa
from scripts.benchmark.yarn_gate import cells

DEFAULT_DIR = "/mnt/raid0/llm/tmp/yarn-agg-oolong-subset"
SOURCE = "oolongbench/oolong-synth"
FAMILY = "synth"
GATE_TASK_GROUPS = ("counting", "user")  # YARN-AGG-1(1): counting and user questions (timeline excluded)
#: OOL-1 thinking_on arm: fixed generation cap (architect-bench R2 precedent), PRE-REGISTERED here.
THINKING_ON_MAX_TOKENS = 16384
THINKING_OFF_MAX_TOKENS = 1024


def data_path(nominal: int, data_dir: str = DEFAULT_DIR) -> Path:
    return Path(data_dir) / f"oolong_synth_test_{nominal}.jsonl"


def available(nominal: int, data_dir: str = DEFAULT_DIR) -> tuple[bool, str]:
    p = data_path(nominal, data_dir)
    if not p.is_file():
        return False, f"missing {p} (fetch with yarn_gate/fetch_oolong_subset.py --lengths {nominal})"
    try:
        oa.load_eval_helpers()
    except Exception as e:  # dateutil missing, vendored sha mismatch ...
        return False, f"oolong scorer unavailable: {type(e).__name__}: {e}"
    return True, str(p)


def read_rows(path: Path, *, task_groups: Iterable[str] = GATE_TASK_GROUPS) -> list[dict]:
    """Rows of one local JSONL, labels-blind: a row carrying the labelled column is refused."""
    keep = set(task_groups)
    rows = []
    with Path(path).open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if oa.LABELLED_COLUMN in r:
                raise oa.LabelledColumnError(f"{path}: row {r.get('id')!r} carries {oa.LABELLED_COLUMN!r}")
            if r.get("task_group") in keep:
                rows.append({k: v for k, v in r.items() if k in oa.NEEDED_COLUMNS})
    return rows


def _offline_counts(data_dir: str) -> dict[tuple[int, Any], int]:
    p = Path(data_dir) / "token_report.json"
    if not p.is_file():
        return {}
    rep = json.loads(p.read_text())
    return {(int(c["context_len"]), c["context_window_id"]): int(c["real_n"]) for c in rep.get("contexts", [])}


def build_cell(nominal: int, *, data_dir: str = DEFAULT_DIR, thinking: bool = False,
               max_contexts: int | None = None, context_filter=None,
               task_groups: Iterable[str] = GATE_TASK_GROUPS, label: str | None = None,
               max_questions_per_context: int | None = None) -> list[dict]:
    """Items for one nominal bucket. ``context_filter(real_n_offline) -> bool`` selects contexts (e.g. the
    constructible ±10% band of a real-token target); ``max_contexts`` keeps the first N contexts in
    context_window_id order. Items of one context stay adjacent (one cached prefill per context)."""
    path = data_path(nominal, data_dir)
    rows = read_rows(path, task_groups=task_groups)
    counts = _offline_counts(data_dir)
    ctxs = sorted(oa.group_by_context(rows), key=lambda c: str(c["context_window_id"]))
    if context_filter is not None:
        ctxs = [c for c in ctxs if context_filter(counts.get((nominal, c["context_window_id"])))]
    if max_contexts is not None:
        ctxs = ctxs[:max_contexts]
    arm_tag = "think" if thinking else "nothink"
    cell = f"oolong_synth@{label or nominal}" + ("" if not thinking else "+think")
    pins = {"dataset": f"{SOURCE}@{oa.SOURCE_REVISIONS[SOURCE]}", "data_file": str(path),
            "upstream_sha": oa.UPSTREAM_SHA_PREFIX, "vendored_sha256": oa.VENDORED_SHA256,
            "task_groups": list(task_groups), "max_tokens_policy":
                f"fixed cap {THINKING_ON_MAX_TOKENS} (thinking_on, pre-registered)" if thinking
                else f"fixed cap {THINKING_OFF_MAX_TOKENS} (thinking_off)"}
    items = []
    for c in ctxs:
        cid = c["context_window_id"]
        for r in sorted(c["rows"], key=lambda r: str(r["id"]))[:max_questions_per_context]:
            dp = {k: v for k, v in r.items() if k != oa.CONTEXT_COLUMN}
            items.append({
                "item_id": f"{cell}#{r['id']}", "cell": cell, "suite": "oolong", "task": r["task_group"],
                "context_id": f"oolong:{nominal}:{cid}", "messages": oa.build_messages(c["text"], r["question"]),
                "max_tokens": THINKING_ON_MAX_TOKENS if thinking else THINKING_OFF_MAX_TOKENS,
                "thinking": bool(thinking), "scorer": "oolong_synth", "gold": r["answer"], "datapoint": dp,
                "nominal_len": nominal, "real_n_offline": counts.get((nominal, cid)),
                "n_source": "model_tokenizer_json" if (nominal, cid) in counts else None,
                "num_labels": r.get("num_labels"), "arm_tag": arm_tag, "pins": pins,
            })
    return items


def file_sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


@cells.register("oolong_synth")
def score_oolong(item, output: str, *, model: str) -> dict:
    """The upstream scorer via the OOL-1 adapter (``make_record`` semantics: empty output scores 0.0 with
    parse_confidence low; the failure class needs finish_reason, which the runner adds afterwards)."""
    if not (output or "").strip():
        return {"score": 0.0, "parse_confidence": "low", "attempted_parse": ""}
    s = oa.score_item(item["datapoint"], output, model, FAMILY)
    return {"score": float(s["score"]), "parse_confidence": s.get("parse_confidence"),
            "attempted_parse": str(s.get("attempted_parse"))}
