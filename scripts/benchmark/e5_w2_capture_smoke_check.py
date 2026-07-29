#!/usr/bin/env python3
"""Verdict for the E5 W2 focused post-fix capture smoke.

HARD PRECONDITION for any decision-grade Gemma W2 sweep. The historic W0 Gemma
capture was 430/430 parse failures (43/43 in each of 10 cells) with NO raw SSE
ledger: reasoning-only output under the 64-token scout cap, with the answer
channel never persisted. Those rows are UNRECOVERABLE — not re-scoreable —
which is the only reason re-running inference is authorised for W2 at all.

This checks the three properties the smoke exists to prove, against a completed
smoke run directory:

  1. `reasoning_text` is persisted SEPARATELY from answer text.
  2. Nonempty answer-text deltas whenever tokens were generated — i.e. the
     `response_capture_missing_answer_text` fail-close never fired.
  3. The offline scorer sees SCOREABLE answer text.

(3) is the one that actually distinguishes a fixed capture from a broken one:
(1) and (2) can both hold while every answer still strips to nothing. So the
scorer is run for real and its parse_ok rate is the verdict, not a proxy.

Exit 0 = smoke PASSED, W2 may proceed to decision-grade.
Exit 1 = smoke FAILED, W2 stays quality-invalid. Never soft-passes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
OFFLINE_SCORER = REPO_ROOT / "scripts" / "benchmark" / "e5_w0_offline_score.py"

# The W0 Gemma failure was TOTAL (43/43 per cell). A fixed capture path should
# be near-perfect; the offline scorer's own per-cell budget is 2. Held to the
# same number here rather than inventing a looser smoke-only threshold.
MAX_PARSE_FAILURES = 2


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def check_capture(run_dir: Path) -> tuple[list[str], dict[str, Any]]:
    """Properties 1 and 2, from the persisted response ledger."""
    failures: list[str] = []
    responses = read_jsonl(run_dir / "responses.jsonl")
    if not responses:
        return (
            [f"no responses.jsonl rows under {run_dir} — nothing was captured"],
            {},
        )

    generated = [
        row
        for row in responses
        if int((row.get("timings") or {}).get("predicted_n") or 0) > 0
    ]
    empty_answer = [
        row for row in generated if not str(row.get("response_text") or "").strip()
    ]
    with_reasoning = [
        row for row in responses if str(row.get("reasoning_text") or "").strip()
    ]
    missing_field = [row for row in responses if "reasoning_text" not in row]

    if missing_field:
        failures.append(
            f"{len(missing_field)}/{len(responses)} rows lack a reasoning_text "
            "field: the answer/reasoning split is not being persisted"
        )
    if empty_answer:
        failures.append(
            f"{len(empty_answer)}/{len(generated)} generated responses have EMPTY "
            "answer text — the exact W0 Gemma failure mode (reasoning-only "
            "output with no answer channel)"
        )

    # The capture fail-close firing is itself a hard failure: it means the
    # server produced tokens the harness could not attribute to an answer.
    cells = read_jsonl(run_dir / "cells.jsonl")
    capture_errors = 0
    for cell in cells:
        for blocker in cell.get("decision_grade_blockers") or []:
            if "response_capture_failure" in blocker:
                capture_errors += 1
                failures.append(f"{cell.get('cell_id')}: {blocker}")

    return failures, {
        "responses": len(responses),
        "generated": len(generated),
        "empty_answer_text": len(empty_answer),
        "rows_with_reasoning_text": len(with_reasoning),
        "capture_fail_close_cells": capture_errors,
    }


def check_scorer(run_dir: Path, python: str) -> tuple[list[str], dict[str, Any]]:
    """Property 3: run the REAL offline scorer and read its verdict."""
    proc = subprocess.run(
        [python, str(OFFLINE_SCORER), "--run-dir", str(run_dir)],
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        return (
            [f"offline scorer exited {proc.returncode}:\n{proc.stdout[-2000:]}"],
            {},
        )

    scores = read_jsonl(run_dir / "offline_scores.jsonl")
    if not scores:
        return (["offline scorer produced no offline_scores.jsonl rows"], {})

    failures: list[str] = []
    by_cell: dict[str, dict[str, int]] = {}
    for row in scores:
        cell = str(row.get("cell_id"))
        bucket = by_cell.setdefault(cell, {"total": 0, "parse_ok": 0})
        bucket["total"] += 1
        if row.get("parse_ok"):
            bucket["parse_ok"] += 1

    for cell, bucket in sorted(by_cell.items()):
        bad = bucket["total"] - bucket["parse_ok"]
        if bad > MAX_PARSE_FAILURES:
            failures.append(
                f"{cell}: {bad}/{bucket['total']} parse failures "
                f"(budget {MAX_PARSE_FAILURES}) — the scorer still cannot see "
                "scoreable answer text"
            )
    return failures, {"parse_ok_by_cell": by_cell}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--python",
        default=str(REPO_ROOT / ".venv" / "bin" / "python"),
        help="interpreter used to invoke the offline scorer",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    run_dir = args.run_dir.resolve()
    capture_failures, capture_stats = check_capture(run_dir)
    scorer_failures, scorer_stats = check_scorer(run_dir, args.python)

    failures = capture_failures + scorer_failures
    verdict = {
        "artifact_type": "e5_w2_capture_smoke_verdict",
        "run_dir": str(run_dir),
        "passed": not failures,
        "failures": failures,
        "capture": capture_stats,
        "scorer": scorer_stats,
        "properties_checked": [
            "reasoning_text persisted separately from answer text",
            "nonempty answer-text deltas whenever tokens were generated",
            "offline scorer sees scoreable answer text",
        ],
    }
    text = json.dumps(verdict, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)

    if failures:
        print(
            "\nSMOKE FAILED — W2 stays quality-invalid; do NOT run a "
            "decision-grade Gemma W2 sweep.",
            file=sys.stderr,
        )
        return 1
    print("\nSMOKE PASSED — W2 may proceed to decision-grade.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
