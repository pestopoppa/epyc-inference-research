#!/usr/bin/env python3
"""Re-score archived per-question JSONL with the current scorer.

The runner persists full response text, so a scorer fix can be applied to
completed runs without spending GPU time re-running inference. Writes
per_question.rescored.jsonl beside the original and never mutates it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v7_quality_gate_runner import (  # noqa: E402
    extract_exact_answer, extract_letter_answer,
)
# CJ-11: migrated off the two-valued `bool(resp) and score_response(...)` idiom.
# Imported from the canonical library directly rather than through the
# v7_quality_gate_runner re-export shim, so this migration does not perturb that
# module's sha256 (it is pinned by laguna_q4_cpu_bench_runner's provenance block).
from answer_scoring import score_response_or_error  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir", type=Path)
    ap.add_argument("--questions", type=Path, action="append", default=[],
                    help="pinned manifest(s) supplying scoring_method/config")
    ap.add_argument("--write", action="store_true", help="write rescored files")
    args = ap.parse_args()

    qmeta: dict[str, dict] = {}
    for man in args.questions:
        for suite, items in json.loads(man.read_text())["suites"].items():
            for q in items:
                qmeta[q["id"]] = q

    for pq in sorted(args.runs_dir.glob("*/*/per_question.jsonl")):
        rows = [json.loads(l) for l in pq.read_text().splitlines() if l.strip()]
        if not rows:
            continue
        changed = flips_to_correct = flips_to_wrong = undecided = 0
        causes: dict[str, int] = {}
        out = []
        for r in rows:
            q = qmeta.get(r["id"])
            if q is None:
                out.append(r)
                continue
            method = q.get("scoring_method", "multiple_choice")
            cfg = q.get("scoring_config", {}) or {}
            resp = r.get("response", "")
            # CJ-11: three-valued. The old idiom was
            # `bool(resp) and score_response(...)`, in which ANY truthy third
            # value coerces to a PASS. Here an undecidable row is RECORDED --
            # verdict `out-of-coverage` plus its CJ-8 cause code -- and is never
            # counted correct. `correct` stays a bool because the JSONL schema
            # and every downstream reader are two-valued; the third state lives
            # in the added `verdict`/`cause` fields, not in `correct`.
            verdict, cause = score_response_or_error(resp, r["expected"], q)
            new_ok = verdict is True
            new_got = (extract_letter_answer(resp) if method == "multiple_choice"
                       else extract_exact_answer(resp, cfg)) if resp else ""
            if verdict is None:
                undecided += 1
                causes[cause] = causes.get(cause, 0) + 1
            if new_ok != r["correct"]:
                changed += 1
                flips_to_correct += new_ok
                flips_to_wrong += (not new_ok)
            r2 = dict(r)
            r2["correct"], r2["extracted"] = new_ok, new_got
            r2["verdict"] = "out-of-coverage" if verdict is None else (
                "pass" if verdict else "fail")
            if cause is not None:
                r2["cause"] = cause
            r2["rescored"] = True
            out.append(r2)
        n = len(out)
        old = sum(r["correct"] for r in rows) / n
        new = sum(r["correct"] for r in out) / n
        noparse_old = sum(1 for r in rows if not r.get("extracted"))
        noparse_new = sum(1 for r in out if not r.get("extracted"))
        cause_note = ("; undecided %d [%s]" % (
            undecided, ", ".join(f"{k}={v}" for k, v in sorted(causes.items())))
            if undecided else "")
        print(f"{pq.parent.parent.name}/{pq.parent.name}: "
              f"{old:.1%} -> {new:.1%}  (+{flips_to_correct}/-{flips_to_wrong} flips; "
              f"noparse {noparse_old} -> {noparse_new}; n={n}{cause_note})")
        if args.write:
            dst = pq.with_name("per_question.rescored.jsonl")
            dst.write_text("".join(json.dumps(r) + "\n" for r in out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
