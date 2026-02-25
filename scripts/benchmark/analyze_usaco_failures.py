#!/usr/bin/env python3
"""Analyze USACO failure patterns from 3-way eval JSONL files.

Reports pass/fail/timeout/token distribution grouped by role, identifies
whether failures are timeout-dominated or truncation-dominated, and flags
slow_delegation as a contributing factor.

Usage:
    python scripts/benchmark/analyze_usaco_failures.py [JSONL_PATH]

If no path given, uses the latest 3way JSONL in benchmarks/results/eval/.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def find_latest_3way() -> Path:
    """Find the most recent 3way JSONL file."""
    base = Path(__file__).resolve().parent.parent.parent / "benchmarks" / "results" / "eval"
    candidates = sorted(base.rglob("3way_*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        print("ERROR: No 3way JSONL files found in", base, file=sys.stderr)
        sys.exit(1)
    return candidates[-1]


def analyze(path: Path) -> None:
    print(f"Analyzing: {path}")
    print(f"{'='*72}\n")

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("suite", "").startswith("usaco"):
                records.append(d)

    if not records:
        print("No USACO records found in this file.")
        return

    print(f"Total USACO records: {len(records)}\n")

    # ── Per-role breakdown ──
    role_stats: dict[str, dict[str, int | list]] = defaultdict(
        lambda: {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "infra_error": 0,
            "timeout": 0,
            "tokens": [],
            "elapsed": [],
            "errors": Counter(),
            "anomalies": Counter(),
        }
    )

    for rec in records:
        qid = rec.get("question_id", "?")
        for role_key, rr in rec.get("role_results", {}).items():
            s = role_stats[role_key]
            s["total"] += 1

            error = rr.get("error") or ""
            error_type = rr.get("error_type", "")
            passed = rr.get("passed")
            tokens = rr.get("tokens_generated", 0)
            elapsed = rr.get("elapsed_seconds", 0.0)

            s["tokens"].append(tokens)
            s["elapsed"].append(elapsed)

            if error_type == "infrastructure":
                s["infra_error"] += 1
                if "504" in str(error) or "timeout" in str(error).lower():
                    s["timeout"] += 1
                s["errors"][error[:80] if error else "infra"] += 1
            elif passed:
                s["passed"] += 1
            else:
                s["failed"] += 1
                if error:
                    s["errors"][error[:80]] += 1

            # Check delegation events for slow_delegation
            for ev in rr.get("delegation_events", []):
                elapsed_ms = ev.get("elapsed_ms", 0)
                if elapsed_ms > 120_000:
                    s["anomalies"]["slow_delegation"] += 1

    for role_key in sorted(role_stats):
        s = role_stats[role_key]
        tokens = s["tokens"]
        elapsed = s["elapsed"]
        total = s["total"]

        print(f"── {role_key} ({total} records) ──")
        print(f"  Passed:      {s['passed']:4d}  ({100*s['passed']/total:.1f}%)")
        print(f"  Failed:      {s['failed']:4d}  ({100*s['failed']/total:.1f}%)")
        print(f"  Infra error: {s['infra_error']:4d}  ({100*s['infra_error']/total:.1f}%)")
        print(f"  Timeout:     {s['timeout']:4d}  ({100*s['timeout']/total:.1f}%)")

        if tokens:
            avg_tok = sum(tokens) / len(tokens)
            max_tok = max(tokens)
            min_tok = min(tokens)
            # Count truncation candidates (tokens near budget limits)
            near_256 = sum(1 for t in tokens if 240 <= t <= 260)
            near_768 = sum(1 for t in tokens if 740 <= t <= 780)
            near_1024 = sum(1 for t in tokens if 1000 <= t <= 1030)
            print(f"  Tokens: avg={avg_tok:.0f} min={min_tok} max={max_tok}")
            if near_256:
                print(f"  ⚠ Near 256 cap: {near_256} records (OLD truncation)")
            if near_768:
                print(f"  ⚠ Near 768 cap: {near_768} records (CURRENT cap)")
            if near_1024:
                print(f"  ⚠ Near 1024 cap: {near_1024} records")

        if elapsed:
            avg_el = sum(elapsed) / len(elapsed)
            max_el = max(elapsed)
            over_120 = sum(1 for e in elapsed if e > 120)
            over_300 = sum(1 for e in elapsed if e > 300)
            print(f"  Elapsed: avg={avg_el:.1f}s max={max_el:.1f}s")
            if over_120:
                print(f"  ⚠ Over 120s: {over_120} records")
            if over_300:
                print(f"  ⚠ Over 300s: {over_300} records (likely timeout)")

        if s["anomalies"]:
            print(f"  Anomalies: {dict(s['anomalies'])}")

        if s["errors"]:
            print(f"  Top errors:")
            for err, cnt in s["errors"].most_common(5):
                print(f"    [{cnt}x] {err}")
        print()

    # ── Summary diagnosis ──
    total_timeout = sum(s["timeout"] for s in role_stats.values())
    total_infra = sum(s["infra_error"] for s in role_stats.values())
    total_failed = sum(s["failed"] for s in role_stats.values())
    total_slow_deleg = sum(s["anomalies"].get("slow_delegation", 0) for s in role_stats.values())

    all_tokens = []
    for s in role_stats.values():
        all_tokens.extend(s["tokens"])
    near_768_all = sum(1 for t in all_tokens if 740 <= t <= 780)

    print(f"{'='*72}")
    print("DIAGNOSIS SUMMARY")
    print(f"  Total failures (non-infra): {total_failed}")
    print(f"  Total infra errors:         {total_infra}")
    print(f"  Total timeouts:             {total_timeout}")
    print(f"  Slow delegations (>120s):   {total_slow_deleg}")
    print(f"  Near 768-token cap:         {near_768_all}")
    print()

    if total_timeout > total_failed:
        print("  → TIMEOUT-DOMINATED: Most failures are infrastructure timeouts.")
        print("    Fix: Increase timeout budget or reduce model contention.")
    elif near_768_all > total_failed // 2:
        print("  → TRUNCATION-DOMINATED: Many responses near 768 token cap.")
        print("    Fix: Consider raising REPL/delegation token budget further.")
    elif total_slow_deleg > total_failed // 2:
        print("  → SLOW-DELEGATION-DOMINATED: Delegation latency burns time budget.")
        print("    Fix: Optimize specialist spawn or increase delegation timeout.")
    else:
        print("  → MIXED: No single dominant failure mode.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        path = find_latest_3way()
    analyze(path)
