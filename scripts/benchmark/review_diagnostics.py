#!/usr/bin/env python3
"""CLI tool for reviewing seeding diagnostic records.

Usage:
    python scripts/benchmark/review_diagnostics.py --last 10
    python scripts/benchmark/review_diagnostics.py --last 20 --anomalies-only
    python scripts/benchmark/review_diagnostics.py --summary
    python scripts/benchmark/review_diagnostics.py --question thinking/t1_q1
    python scripts/benchmark/review_diagnostics.py --signal format_violation
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

DIAGNOSTICS_PATH = Path("/mnt/raid0/llm/claude/logs/seeding_diagnostics.jsonl")


def load_records(path: Path) -> list[dict]:
    """Load all diagnostic records from JSONL file."""
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def print_record(rec: dict, verbose: bool = False) -> None:
    """Print a single diagnostic record in a readable format."""
    triggered = [
        name for name, active in rec.get("anomaly_signals", {}).items()
        if active
    ]
    passed = "PASS" if rec.get("passed") else "FAIL"
    anomaly_str = ", ".join(triggered) if triggered else "none"
    score = rec.get("anomaly_score", 0)

    print(f"  {rec.get('question_id', '?'):30s} [{passed}] "
          f"score={score:.2f}  {anomaly_str}")
    print(f"    config={rec.get('config', '?')}  role={rec.get('role', '?')}:{rec.get('mode', '?')}  "
          f"tokens={rec.get('tokens_generated', 0)}  elapsed={rec.get('elapsed_s', 0):.1f}s")

    if rec.get("error"):
        print(f"    error: {rec['error'][:100]}")

    if verbose:
        answer = rec.get("answer", "")
        preview = answer[:200]
        if len(answer) > 200:
            preview += f"... [{len(answer) - 200} chars more]"
        print(f"    answer: {preview}")

    print()


def show_last(records: list[dict], n: int, anomalies_only: bool, verbose: bool) -> None:
    """Show the last N records."""
    subset = records[-n:]
    if anomalies_only:
        subset = [r for r in subset if r.get("anomaly_score", 0) > 0]

    if not subset:
        print("No matching records.")
        return

    print(f"Showing {len(subset)} records:\n")
    for rec in subset:
        print_record(rec, verbose)


def show_summary(records: list[dict]) -> None:
    """Show aggregate summary statistics."""
    if not records:
        print("No diagnostic records found.")
        return

    total = len(records)
    passed = sum(1 for r in records if r.get("passed"))
    failed = total - passed
    anomalous = sum(1 for r in records if r.get("anomaly_score", 0) > 0)
    critical = sum(1 for r in records if r.get("anomaly_score", 0) >= 1.0)

    print(f"Total records:    {total}")
    print(f"Passed:           {passed} ({100 * passed / total:.1f}%)")
    print(f"Failed:           {failed} ({100 * failed / total:.1f}%)")
    print(f"Anomalous:        {anomalous} ({100 * anomalous / total:.1f}%)")
    print(f"Critical (>=1.0): {critical}")
    print()

    # Signal frequency
    signal_counts: Counter = Counter()
    for rec in records:
        for name, active in rec.get("anomaly_signals", {}).items():
            if active:
                signal_counts[name] += 1

    if signal_counts:
        print("Signal frequency:")
        for name, count in signal_counts.most_common():
            print(f"  {name:30s} {count:4d} ({100 * count / total:.1f}%)")
        print()

    # Per-suite breakdown
    suite_stats: dict[str, dict] = {}
    for rec in records:
        suite = rec.get("suite", "unknown")
        if suite not in suite_stats:
            suite_stats[suite] = {"total": 0, "passed": 0, "anomalous": 0}
        suite_stats[suite]["total"] += 1
        if rec.get("passed"):
            suite_stats[suite]["passed"] += 1
        if rec.get("anomaly_score", 0) > 0:
            suite_stats[suite]["anomalous"] += 1

    print("Per-suite breakdown:")
    print(f"  {'Suite':20s} {'Total':>6s} {'Pass%':>6s} {'Anomaly%':>8s}")
    for suite, stats in sorted(suite_stats.items()):
        pct_pass = 100 * stats["passed"] / stats["total"]
        pct_anom = 100 * stats["anomalous"] / stats["total"]
        print(f"  {suite:20s} {stats['total']:6d} {pct_pass:5.1f}% {pct_anom:7.1f}%")


def show_question(records: list[dict], question_id: str, verbose: bool) -> None:
    """Show all records for a specific question."""
    matches = [r for r in records if r.get("question_id") == question_id]
    if not matches:
        # Try partial match
        matches = [r for r in records if question_id in r.get("question_id", "")]

    if not matches:
        print(f"No records found for question '{question_id}'.")
        return

    print(f"Found {len(matches)} records for '{question_id}':\n")
    for rec in matches:
        print_record(rec, verbose=True)


def show_signal(records: list[dict], signal_name: str, verbose: bool) -> None:
    """Show all records where a specific signal fired."""
    matches = [
        r for r in records
        if r.get("anomaly_signals", {}).get(signal_name, False)
    ]
    if not matches:
        print(f"No records with signal '{signal_name}'.")
        return

    print(f"Found {len(matches)} records with signal '{signal_name}':\n")
    for rec in matches:
        print_record(rec, verbose)


def main() -> None:
    parser = argparse.ArgumentParser(description="Review seeding diagnostic records")
    parser.add_argument("--file", type=Path, default=DIAGNOSTICS_PATH,
                        help=f"Diagnostics JSONL file (default: {DIAGNOSTICS_PATH})")
    parser.add_argument("--last", type=int, default=0,
                        help="Show last N records")
    parser.add_argument("--anomalies-only", action="store_true",
                        help="Only show records with anomaly_score > 0")
    parser.add_argument("--summary", action="store_true",
                        help="Show aggregate summary statistics")
    parser.add_argument("--question", type=str, default=None,
                        help="Show records for a specific question ID")
    parser.add_argument("--signal", type=str, default=None,
                        help="Show records where a specific signal fired")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show answer text previews")

    args = parser.parse_args()

    records = load_records(args.file)

    if args.summary:
        show_summary(records)
    elif args.question:
        show_question(records, args.question, args.verbose)
    elif args.signal:
        show_signal(records, args.signal, args.verbose)
    elif args.last > 0:
        show_last(records, args.last, args.anomalies_only, args.verbose)
    else:
        # Default: show summary
        show_summary(records)


if __name__ == "__main__":
    main()
