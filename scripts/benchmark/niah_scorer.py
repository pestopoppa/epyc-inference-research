#!/usr/bin/env python3
"""NIAH strict + lenient dual scorer (RLM contested claims, E1a).

Why this exists
---------------
The most-cited RLM "rescue" number (arXiv:2603.02615, DeepSeek OOLONG 0.0 -> 42.1) is
substantially a *format* artifact: the base model found the answer but failed a strict
exact-match scorer, and no rescoring was done (epyc-root
``handoffs/active/rlm-contested-claims-self-evaluation.md`` E0/E1a). E1 must therefore score
every response TWICE and report both numbers together, so a format failure can never be
read as a retrieval failure.

Contract (E1a)
--------------
* ``strict``  -- raw exact match: ``response.strip() == expected.strip()``. Only outer
  whitespace is forgiven (the same rule as ``answer_scoring.score_response``'s fallback);
  case, inner whitespace, Unicode form, punctuation and any surrounding prose all fail.
  This reproduces the strict-format scorer the literature used.
* ``lenient`` -- normalized, boundary-aware substring: both sides are NFKC-normalized,
  stripped of format characters (Unicode category ``Cf``, e.g. zero-width spaces/BOM),
  casefolded, and whitespace-collapsed; the expected needle must then occur in the response
  with no word character (``\\w``) immediately before or after it. The boundary rule stops
  ``"12"`` matching inside ``"123"`` or ``"key_12"``.

Invariant: ``strict`` implies ``lenient`` (tested).

Metric direction
----------------
``strict_accuracy`` and ``lenient_accuracy`` are fractions in [0, 1], **higher is better**.
``format_gap = lenient_accuracy - strict_accuracy`` is a DIAGNOSTIC with no better/worse
direction: it is the share of items that found the needle but failed the format check.
Report all three; never report one accuracy alone.

Determinism
-----------
Pure functions over strings, stdlib only (``re``, ``unicodedata``), no randomness, no I/O in
the scoring path, no LLM judge (the scoring trust boundary is human-amendment-only,
MEASUREMENT.md). The same inputs always produce byte-identical ``--out`` JSON.

Undecidable items
-----------------
An item whose expected needle is empty after normalization cannot be scored. It is counted
in ``n_undecidable`` and excluded from both denominators -- never silently scored as a pass
or a fail. An empty/missing response IS decidable: it fails both arms.

CLI (deterministic replay of saved outputs -- no inference)
-----------------------------------------------------------
    python3 scripts/benchmark/niah_scorer.py --in outputs.jsonl [--out report.json] \
        [--response-field response] [--expected-field expected] [--id-field id]

Each input line is a JSON object. Field names are configurable so fast-rlm
``benchmarks/niah_benchmark.py`` outputs and the in-repo ``long_context_adapters``
NIAH tasks (``ruler_niah_*``, ``needle_*``; gold in ``expected``) both score without a
converter. ``expected`` may be a string or a list of acceptable strings (any match counts).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Iterable

SCORER_ID = "epyc.niah_scorer.strict_lenient.v1"
METRIC_DIRECTION = {
    "strict_accuracy": "higher_better",
    "lenient_accuracy": "higher_better",
    "format_gap": "diagnostic_no_direction",
}

_WS = re.compile(r"\s+")


def normalize_lenient(text: str) -> str:
    """NFKC -> drop Cf format chars -> casefold -> collapse whitespace -> strip."""
    t = unicodedata.normalize("NFKC", text or "")
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Cf")
    t = t.casefold()
    return _WS.sub(" ", t).strip()


def score_strict(response: str, expected: str) -> bool:
    return (response or "").strip() == (expected or "").strip()


def score_lenient(response: str, expected: str) -> bool:
    needle = normalize_lenient(expected)
    if not needle:
        return False
    hay = normalize_lenient(response)
    pattern = r"(?<!\w)" + re.escape(needle) + r"(?!\w)"
    return re.search(pattern, hay) is not None


def _expected_list(expected) -> list[str]:
    if expected is None:
        return []
    if isinstance(expected, (list, tuple)):
        return [str(e) for e in expected if e is not None]
    return [str(expected)]


def score_niah(response, expected) -> dict:
    """Score one response under both arms. Returns a dict; never raises on bad input."""
    resp = "" if response is None else str(response)
    golds = _expected_list(expected)
    decidable = any(normalize_lenient(g) for g in golds)
    if not decidable:
        return {"decidable": False, "strict": None, "lenient": None,
                "cause": "no_reference"}
    return {
        "decidable": True,
        "strict": any(score_strict(resp, g) for g in golds),
        "lenient": any(score_lenient(resp, g) for g in golds),
        "cause": None,
    }


def aggregate(results: Iterable[dict]) -> dict:
    """Fold per-item results into the report block. Both accuracies always travel together."""
    rs = list(results)
    dec = [r for r in rs if r["decidable"]]
    n = len(dec)
    s = sum(1 for r in dec if r["strict"])
    l_ = sum(1 for r in dec if r["lenient"])
    sa = s / n if n else None
    la = l_ / n if n else None
    return {
        "scorer_id": SCORER_ID,
        "metric_direction": METRIC_DIRECTION,
        "n_items": len(rs),
        "n_scored": n,
        "n_undecidable": len(rs) - n,
        "strict_correct": s,
        "lenient_correct": l_,
        "strict_accuracy": sa,
        "lenient_accuracy": la,
        "format_gap": (la - sa) if n else None,
        "format_only_failures": sum(1 for r in dec if r["lenient"] and not r["strict"]),
    }


def score_records(records: Iterable[dict], response_field: str = "response",
                  expected_field: str = "expected", id_field: str = "id") -> dict:
    items = []
    for idx, rec in enumerate(records):
        r = score_niah(rec.get(response_field), rec.get(expected_field))
        r["id"] = rec.get(id_field, idx)
        items.append(r)
    return {"summary": aggregate(items), "items": items}


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--response-field", default="response")
    ap.add_argument("--expected-field", default="expected")
    ap.add_argument("--id-field", default="id")
    a = ap.parse_args(argv)
    report = score_records(_read_jsonl(a.inp), a.response_field, a.expected_field, a.id_field)
    report["input"] = str(a.inp)
    text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if a.out:
        a.out.write_text(text, encoding="utf-8")
    s = report["summary"]
    print(f"n_scored={s['n_scored']} undecidable={s['n_undecidable']} "
          f"strict={s['strict_accuracy']} lenient={s['lenient_accuracy']} "
          f"format_gap={s['format_gap']}", file=sys.stderr)
    if not a.out:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
