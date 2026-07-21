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
    extract_exact_answer, extract_letter_answer, score_response,
)


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
        changed = flips_to_correct = flips_to_wrong = 0
        out = []
        for r in rows:
            q = qmeta.get(r["id"])
            if q is None:
                out.append(r)
                continue
            method = q.get("scoring_method", "multiple_choice")
            cfg = q.get("scoring_config", {}) or {}
            resp = r.get("response", "")
            new_ok = bool(resp) and score_response(resp, r["expected"], q)
            new_got = (extract_letter_answer(resp) if method == "multiple_choice"
                       else extract_exact_answer(resp, cfg)) if resp else ""
            if new_ok != r["correct"]:
                changed += 1
                flips_to_correct += new_ok
                flips_to_wrong += (not new_ok)
            r2 = dict(r)
            r2["correct"], r2["extracted"] = new_ok, new_got
            r2["rescored"] = True
            out.append(r2)
        n = len(out)
        old = sum(r["correct"] for r in rows) / n
        new = sum(r["correct"] for r in out) / n
        noparse_old = sum(1 for r in rows if not r.get("extracted"))
        noparse_new = sum(1 for r in out if not r.get("extracted"))
        print(f"{pq.parent.parent.name}/{pq.parent.name}: "
              f"{old:.1%} -> {new:.1%}  (+{flips_to_correct}/-{flips_to_wrong} flips; "
              f"noparse {noparse_old} -> {noparse_new}; n={n})")
        if args.write:
            dst = pq.with_name("per_question.rescored.jsonl")
            dst.write_text("".join(json.dumps(r) + "\n" for r in out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
