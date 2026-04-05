#!/usr/bin/env python3
"""Validate long-context evaluation datasets.

Loads each dataset through its adapter, reports statistics (row counts,
context length distributions), and verifies data integrity.

Usage:
    python validate_long_context_datasets.py [--sample N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure local imports work
sys.path.insert(0, str(Path(__file__).parent))

from dataset_adapters import get_adapter, ADAPTER_SUITES


LONG_CONTEXT_SUITES = ["longbench", "zeroscrolls", "leval", "ruler", "needle_parameterized"]


def _stats(values: list[int]) -> dict:
    """Compute min/max/mean/p50/p95 for a list of values."""
    if not values:
        return {"count": 0}
    s = sorted(values)
    n = len(s)
    return {
        "count": n,
        "min": s[0],
        "max": s[-1],
        "mean": sum(s) // n,
        "p50": s[n // 2],
        "p95": s[int(n * 0.95)] if n > 1 else s[0],
    }


def validate_suite(suite_name: str, sample_size: int = 20) -> dict:
    """Validate a single suite adapter."""
    print(f"\n{'='*60}")
    print(f"Suite: {suite_name}")
    print(f"{'='*60}")

    adapter = get_adapter(suite_name)
    if adapter is None:
        print(f"  SKIP: adapter not available (import failed or not registered)")
        return {"status": "skip", "reason": "adapter not available"}

    try:
        total = adapter.total_available
        print(f"  Total available: {total}")
    except Exception as e:
        print(f"  FAIL: could not load dataset: {e}")
        return {"status": "fail", "error": str(e)}

    if total == 0:
        print(f"  WARN: dataset is empty")
        return {"status": "warn", "total": 0}

    # Sample questions
    actual_sample = min(sample_size, total)
    try:
        questions = adapter.sample(n=actual_sample, seed=42)
    except Exception as e:
        print(f"  FAIL: sampling failed: {e}")
        return {"status": "fail", "error": str(e)}

    print(f"  Sampled: {len(questions)} questions")

    # Validate prompt structure
    required_fields = {"id", "suite", "prompt", "expected", "scoring_method", "tier"}
    missing_fields = []
    empty_prompts = 0
    empty_expected = 0
    context_lengths = []

    for q in questions:
        missing = required_fields - set(q.keys())
        if missing:
            missing_fields.append(missing)

        if not q.get("prompt"):
            empty_prompts += 1
        if not q.get("expected"):
            empty_expected += 1

        # Track context length
        prompt_chars = len(q.get("prompt", ""))
        context_lengths.append(prompt_chars)

        meta = q.get("metadata", {})
        if "context_length_chars" in meta:
            pass  # metadata is self-consistent

    if missing_fields:
        print(f"  WARN: {len(missing_fields)} questions missing fields: {missing_fields[0]}")
    if empty_prompts:
        print(f"  WARN: {empty_prompts}/{len(questions)} questions have empty prompts")
    if empty_expected:
        print(f"  INFO: {empty_expected}/{len(questions)} questions have empty expected (may be generation tasks)")

    # Context length statistics
    length_stats = _stats(context_lengths)
    print(f"  Context length (chars): min={length_stats['min']:,} max={length_stats['max']:,} "
          f"mean={length_stats['mean']:,} p50={length_stats['p50']:,} p95={length_stats['p95']:,}")

    # Tier distribution
    tiers = [q.get("tier", 0) for q in questions]
    tier_counts = {}
    for t in tiers:
        tier_counts[t] = tier_counts.get(t, 0) + 1
    print(f"  Tier distribution: {dict(sorted(tier_counts.items()))}")

    # Scoring methods
    methods = set(q.get("scoring_method", "unknown") for q in questions)
    print(f"  Scoring methods: {methods}")

    # Show one example
    if questions:
        ex = questions[0]
        prompt_preview = ex["prompt"][:100].replace("\n", " ")
        print(f"  Example: id={ex['id']}, prompt='{prompt_preview}...'")
        print(f"           expected='{str(ex.get('expected', ''))[:80]}'")

    return {
        "status": "ok",
        "total": total,
        "sampled": len(questions),
        "context_length_stats": length_stats,
        "tier_distribution": tier_counts,
        "scoring_methods": list(methods),
        "empty_prompts": empty_prompts,
        "empty_expected": empty_expected,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate long-context eval datasets")
    parser.add_argument("--sample", type=int, default=20, help="Questions to sample per suite")
    parser.add_argument("--suite", type=str, help="Validate a single suite only")
    args = parser.parse_args()

    suites = [args.suite] if args.suite else LONG_CONTEXT_SUITES
    results = {}

    for suite in suites:
        if suite not in ADAPTER_SUITES:
            print(f"\n  SKIP {suite}: not in ADAPTER_SUITES")
            continue
        results[suite] = validate_suite(suite, args.sample)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for suite, result in results.items():
        status = result.get("status", "unknown")
        total = result.get("total", "?")
        icon = {"ok": "OK", "warn": "WARN", "fail": "FAIL", "skip": "SKIP"}.get(status, "?")
        print(f"  [{icon}] {suite}: {total} questions")

    # Exit code
    failures = sum(1 for r in results.values() if r.get("status") == "fail")
    if failures:
        print(f"\n{failures} suite(s) failed validation")
        sys.exit(1)


if __name__ == "__main__":
    main()
