#!/usr/bin/env python3
"""OOL-1: Oolong long-context suite adapter (labels-blind by construction).

Requirements: handoffs/active/rao-redel-substrate-spike.md -> OOL-1 (epyc-root).

* Scoring is the UPSTREAM code: ``vendor/oolong/eval_helpers.py`` (repo
  abertsch72/oolong @ 0bb7eabe8, vendored UNMODIFIED, sha256 verified on every
  import).  Nothing here re-implements the parse/score rules.
* Only ``context_window_text`` is ever used.  ``context_window_text_with_labels``
  is refused (``LabelledColumnError``) everywhere except the explicitly named
  labels-given ablation, whose records carry ``labels_given: true`` and are
  excluded from every gate (a regex scored ~100% on the labelled text).
* One cached prefill per context: a context's questions share a byte-identical
  prefix (context first, question LAST), one llama-server slot per context,
  ``cache_prompt: true``.
* Persisted per item: parse_confidence, answered/attempted/no-answer, failure
  class (wrong | no_answer_budget | refusal | unparsable), arm (thinking
  on/off), labels_given, source_revision.  CIs are clustered by context.

Inference only happens in ``run_arm`` / ``main`` (guarded by WINDOW_CONFIRMED=1).
Everything else is pure and unit-testable with a tiny fixture.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import sys
import time
import types
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

SUITE = "oolong"

# ── pins (OOL-1) ──────────────────────────────────────────────────────────────
SOURCE_REVISIONS = {
    "oolongbench/oolong-synth": "81d2f8d9349f6ff329d695f9be494051e9f05cb9",
    "oolongbench/oolong-real": "3ae6ab58ffcc4253ed52daeb58e02ed4cfea5784",
}
UPSTREAM_SHA_PREFIX = "0bb7eabe8"
VENDORED_SHA256 = "247583a3b653b91c39fb88a102460802f1493175b23d05a7217aead0d685c64c"
_VENDOR_PATH = Path(__file__).resolve().parent / "vendor" / "oolong" / "eval_helpers.py"

CONTEXT_COLUMN = "context_window_text"
LABELLED_COLUMN = "context_window_text_with_labels"
NEEDED_COLUMNS = (
    "id", "context_window_id", "dataset", "question", "answer", "answer_type",
    "task_group", "task", "context_len", "num_labels", CONTEXT_COLUMN,
)
QUESTIONS_PER_CONTEXT = 25
FAILURE_CLASSES = ("wrong", "no_answer_budget", "refusal", "unparsable")
_REFUSAL_RE = re.compile(
    r"\b(i (?:can(?:no|')t|am unable to|'m unable to|won't)|i'm sorry,? but|"
    r"cannot (?:answer|determine|help)|unable to (?:answer|determine))\b", re.I)


class LabelledColumnError(ValueError):
    """The labelled column was supplied to a labels-blind path."""


class VendorIntegrityError(RuntimeError):
    """Vendored eval_helpers.py does not match the pinned sha256."""


# ── vendored scorer import ────────────────────────────────────────────────────
_HEAVY = ("litellm", "datasets", "jsonlines", "tiktoken", "transformers")
_helpers = None


def load_eval_helpers():
    """Import the vendored upstream module after verifying its sha256.

    Upstream does module-level ``import litellm/datasets/jsonlines/tiktoken``
    and ``from transformers import AutoTokenizer``; none is used by the scoring
    functions.  Absent ones are stubbed in ``sys.modules`` (never overriding a
    real install) so the file can be imported UNMODIFIED.
    """
    global _helpers
    if _helpers is not None:
        return _helpers
    digest = hashlib.sha256(_VENDOR_PATH.read_bytes()).hexdigest()
    if digest != VENDORED_SHA256:
        raise VendorIntegrityError(f"{_VENDOR_PATH} sha256 {digest} != {VENDORED_SHA256}")
    import dateutil.parser  # noqa: F401  (upstream does bare `import dateutil`)
    for name in _HEAVY:
        try:
            __import__(name)
        except Exception:
            mod = types.ModuleType(name)
            mod.load_dataset = lambda *a, **k: (_ for _ in ()).throw(
                RuntimeError("stub: not available"))
            mod.AutoTokenizer = object
            sys.modules[name] = mod
    spec = importlib.util.spec_from_file_location("oolong_eval_helpers", _VENDOR_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _helpers = mod
    return mod


# ── labels-blind row handling ─────────────────────────────────────────────────
def assert_labels_blind(column: str, labels_given: bool = False) -> str:
    """Return the column to read, refusing the labelled one unless labels_given."""
    if column == LABELLED_COLUMN or "with_labels" in column:
        if not labels_given:
            raise LabelledColumnError(
                f"refusing {column!r}: labelled text is only allowed in the "
                "labels-given ablation (labels_given=True, excluded from gates)")
    return column


def context_text(row: dict, labels_given: bool = False) -> str:
    """The prompt context for ``row``.  Labels-blind unless ablation."""
    col = LABELLED_COLUMN if labels_given else CONTEXT_COLUMN
    assert_labels_blind(col, labels_given)
    if col not in row or row[col] is None:
        raise LabelledColumnError(f"row {row.get('id')!r} lacks {col!r}") \
            if not labels_given else KeyError(col)
    return row[col]


def group_by_context(rows: Iterable[dict], labels_given: bool = False) -> list[dict]:
    """Cluster rows into contexts: [{context_window_id, text, rows:[...]}].

    The context text must be identical across a context's rows (it is the
    shared cached prefix); a mismatch is an error, not a silent pick.
    """
    ctxs: dict[Any, dict] = {}
    for r in rows:
        text = context_text(r, labels_given)
        cid = r["context_window_id"]
        c = ctxs.setdefault(cid, {"context_window_id": cid, "text": text, "rows": []})
        if c["text"] != text:
            raise ValueError(f"context {cid!r}: rows disagree on context text")
        c["rows"].append(r)
    return list(ctxs.values())


def build_messages(context: str, question: str) -> list[dict]:
    """Context first, question LAST => byte-identical cacheable prefix."""
    return [{"role": "user", "content": context + "\n\n" + question}]


# ── thin OpenAI-compatible client ─────────────────────────────────────────────
class LlamaServerClient:
    def __init__(self, base_url: str, model: str = "default", timeout: float = 3600.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def payload(self, messages, *, slot: int, thinking: bool, max_tokens: int,
                temperature: float = 0.0, seed: int = 0) -> dict:
        return {
            "model": self.model, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "seed": seed, "stream": False,
            "cache_prompt": True, "id_slot": slot,
            "chat_template_kwargs": {"enable_thinking": bool(thinking)},
        }

    def complete(self, messages, **kw) -> dict:
        req = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=json.dumps(self.payload(messages, **kw)).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.load(resp)
        ch = body["choices"][0]
        return {"content": ch["message"].get("content") or "",
                "finish_reason": ch.get("finish_reason"),
                "usage": body.get("usage", {})}


# ── scoring / classification ──────────────────────────────────────────────────
def score_item(datapoint: dict, output: str, model: str, family: str = "synth") -> dict:
    """Score with the UPSTREAM function; family synth|real(dnd)."""
    h = load_eval_helpers()
    fn = h.synth_process_response if family == "synth" else h.dnd_process_response
    return fn(datapoint, output, model)


def classify_failure(score: float, output: str, finish_reason: Optional[str],
                     parse_confidence: str) -> Optional[str]:
    """None when fully correct; else one of FAILURE_CLASSES.

    Order matters: budget exhaustion and refusals are not 'wrong answers'.
    """
    if finish_reason == "length" or not (output or "").strip():
        return "no_answer_budget"
    if score >= 1:
        return None
    if _REFUSAL_RE.search(output):
        return "refusal"
    if parse_confidence == "low":
        return "unparsable"
    return "wrong"


def make_record(datapoint: dict, comp: dict, *, model: str, arm: str, family: str,
                source: str, labels_given: bool = False) -> dict:
    out = comp["content"]
    answered = bool(out.strip()) and comp.get("finish_reason") != "length"
    scored = score_item(datapoint, out, model, family) if out.strip() else {
        "id": datapoint["id"], "context_window_id": datapoint["context_window_id"],
        "parse_confidence": "low", "score": 0.0, "attempted_parse": "",
        "full_answer": out, "answer": str(datapoint.get("answer"))}
    pc = scored["parse_confidence"]
    score = float(scored["score"])
    return {
        "suite": SUITE, "id": datapoint["id"],
        "context_window_id": datapoint["context_window_id"],
        "model": model, "arm": arm, "family": family,
        "labels_given": bool(labels_given),
        "source_revision": f"{source}@{SOURCE_REVISIONS[source]}",
        "upstream_sha": UPSTREAM_SHA_PREFIX, "vendored_sha256": VENDORED_SHA256,
        "context_len": datapoint.get("context_len"),
        "task_group": datapoint.get("task_group"),
        "attempted": True, "answered": answered, "no_answer": not answered,
        "parse_confidence": pc, "score": score,
        "failure_class": classify_failure(score, out, comp.get("finish_reason"), pc),
        "finish_reason": comp.get("finish_reason"),
        "usage": comp.get("usage", {}),
        "attempted_parse": str(scored.get("attempted_parse")),
        "full_answer": out, "gold": str(scored.get("answer")),
    }


# ── summaries ─────────────────────────────────────────────────────────────────
def gate_records(records: Iterable[dict]) -> list[dict]:
    """Records eligible for any gate: labels_given records are excluded."""
    return [r for r in records if not r.get("labels_given")]


def cluster_ci(records: list[dict], n_boot: int = 2000, seed: int = 0,
               alpha: float = 0.05) -> dict:
    """Mean score with a context-clustered bootstrap CI (resample contexts)."""
    by_ctx: dict[Any, list[float]] = defaultdict(list)
    for r in records:
        by_ctx[r["context_window_id"]].append(float(r["score"]))
    ctxs = list(by_ctx.values())
    n = sum(len(c) for c in ctxs)
    if not ctxs:
        return {"n_items": 0, "n_contexts": 0, "mean": None, "lo": None, "hi": None}
    mean = sum(sum(c) for c in ctxs) / n
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        pick = [ctxs[rng.randrange(len(ctxs))] for _ in ctxs]
        m = sum(len(c) for c in pick)
        boots.append(sum(sum(c) for c in pick) / m)
    boots.sort()
    lo = boots[int(math.floor(alpha / 2 * (n_boot - 1)))]
    hi = boots[int(math.ceil((1 - alpha / 2) * (n_boot - 1)))]
    return {"n_items": n, "n_contexts": len(ctxs), "mean": mean, "lo": lo, "hi": hi,
            "method": "context-clustered percentile bootstrap", "n_boot": n_boot,
            "seed": seed}


def summarize(records: list[dict]) -> dict:
    """Per (model, arm, context_len): counts, failure classes, low-confidence share, CI.

    Counts are over gate-eligible records only; labels_given are reported apart.
    """
    out: dict[str, Any] = {"cells": {}, "labels_given_n": 0}
    out["labels_given_n"] = len(records) - len(gate_records(records))
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in gate_records(records):
        cells[(r["model"], r["arm"], r.get("context_len"))].append(r)
    for (model, arm, clen), rs in sorted(cells.items(), key=lambda kv: str(kv[0])):
        fc = {k: sum(1 for r in rs if r["failure_class"] == k) for k in FAILURE_CLASSES}
        low = sum(1 for r in rs if r["parse_confidence"] == "low")
        out["cells"][f"{model}|{arm}|{clen}"] = {
            "attempted": sum(r["attempted"] for r in rs),
            "answered": sum(r["answered"] for r in rs),
            "no_answer": sum(r["no_answer"] for r in rs),
            "failure_classes": fc,
            "low_parse_confidence": low,
            "low_parse_confidence_share": low / len(rs),
            "ci": cluster_ci(rs),
        }
    return out


def sample_low_for_hand_review(records: list[dict], per_arm: int = 20, seed: int = 0) -> dict:
    """Deterministic sample of ``per_arm`` low-confidence items per (model, arm).

    OOL-1 requires hand-classifying these BEFORE any thinking-on vs off comparison.
    """
    pool: dict[tuple, list[dict]] = defaultdict(list)
    for r in gate_records(records):
        if r["parse_confidence"] == "low":
            pool[(r["model"], r["arm"])].append(r)
    rng = random.Random(seed)
    res = {}
    for k, rs in sorted(pool.items()):
        rs = sorted(rs, key=lambda r: str(r["id"]))
        res[f"{k[0]}|{k[1]}"] = rng.sample(rs, min(per_arm, len(rs)))
    return res


# ── data loading (download only inside the window runner) ────────────────────
def load_rows(source: str, split: str, config: Optional[str] = None,
              limit_contexts: Optional[int] = None) -> list[dict]:
    """Load pinned dataset rows, dropping the labelled column before use."""
    from datasets import load_dataset  # real package required here
    ds = load_dataset(source, config, split=split, revision=SOURCE_REVISIONS[source])
    drop = [c for c in ds.column_names if c not in NEEDED_COLUMNS]
    ds = ds.remove_columns(drop)
    rows = [dict(r) for r in ds]
    if limit_contexts is not None:
        keep = []
        for r in rows:
            if r["context_window_id"] not in keep:
                keep.append(r["context_window_id"])
        keep = set(keep[:limit_contexts])
        rows = [r for r in rows if r["context_window_id"] in keep]
    return rows


def run_arm(client: LlamaServerClient, rows: list[dict], *, model: str, thinking: bool,
            source: str, family: str, out_path: Path, max_tokens: int, n_slots: int = 4,
            labels_given: bool = False) -> list[dict]:
    """Inference path (window only).  Resumable JSONL, one record per question."""
    arm = ("labels_given_" if labels_given else "") + ("thinking_on" if thinking else "thinking_off")
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in out_path.read_text().splitlines() if l.strip()}
    recs = []
    with out_path.open("a") as fh:
        for ci, ctx in enumerate(group_by_context(rows, labels_given)):
            for r in ctx["rows"]:  # question last; same prefix => cache hit after first
                if r["id"] in done:
                    continue
                comp = client.complete(build_messages(ctx["text"], r["question"]),
                                       slot=ci % n_slots, thinking=thinking,
                                       max_tokens=max_tokens)
                rec = make_record(r, comp, model=model, arm=arm, family=family,
                                  source=source, labels_given=labels_given)
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                recs.append(rec)
    return recs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", choices=sorted(SOURCE_REVISIONS), default="oolongbench/oolong-synth")
    ap.add_argument("--split", default="test")
    ap.add_argument("--config", default=None)
    ap.add_argument("--arm", choices=["thinking_on", "thinking_off"], required=True)
    ap.add_argument("--labels-given", action="store_true",
                    help="ablation only; records carry labels_given: true, excluded from gates")
    ap.add_argument("--limit-contexts", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if os.environ.get("WINDOW_CONFIRMED") != "1":
        print("refusing: inference run needs an inference window (WINDOW_CONFIRMED=1)", file=sys.stderr)
        return 2
    family = "synth" if a.source.endswith("synth") else "real"
    rows = load_rows(a.source, a.split, a.config, a.limit_contexts)
    if a.labels_given:  # ablation: re-read the labelled column explicitly
        raise SystemExit("labels-given ablation loader not enabled in the gate run; "
                         "use run_arm(labels_given=True) with an explicitly loaded labelled column")
    run_arm(LlamaServerClient(a.base_url, a.model), rows, model=a.model,
            thinking=a.arm == "thinking_on", source=a.source, family=family,
            out_path=Path(a.out), max_tokens=a.max_tokens)
    return 0


if __name__ == "__main__":
    sys.exit(main())
