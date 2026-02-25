#!/usr/bin/env python3
"""Calibrate timeout settings by running hardest questions through all models.

Samples T3 (hardest tier) questions from each dataset and measures actual
generation times to determine appropriate timeout values.

Usage:
    # Dry run - show what would be tested
    python3 scripts/benchmark/calibrate_timeouts.py --dry-run

    # Run calibration (requires orchestrator stack running)
    python3 scripts/benchmark/calibrate_timeouts.py --questions-per-suite 1

    # Test specific suite
    python3 scripts/benchmark/calibrate_timeouts.py --suite gpqa --questions-per-suite 3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "benchmark"))

from dataset_adapters import get_adapter, ADAPTER_SUITES
from seeding_types import DEFAULT_ROLES, DEFAULT_TIMEOUT, ROLE_PORT, HEAVY_PORTS

# Suites with real T3 questions worth calibrating
CALIBRATION_SUITES = [
    "gpqa",          # 292 T3 - graduate science
    "hotpotqa",      # 193 T3 - multi-hop reasoning
    "livecodebench", # 209 T3 - LeetCode hard
    "debugbench",    # 140 T3 - hard bugs
    "general",       # 335 T3 - college MMLU
]

# Roles to test (heaviest first - most likely to need longer timeouts)
CALIBRATION_ROLES = [
    "architect_general",   # 235B - slowest
    "architect_coding",    # 480B - largest
    "coder_escalation",    # 32B with spec decode
    "frontdoor",           # 30B MoE
]


def sample_hardest_questions(suite: str, n: int = 1, seed: int = 42) -> list[dict]:
    """Sample T3 (hardest) questions from a suite."""
    adapter = get_adapter(suite)
    if adapter is None:
        return []

    try:
        adapter._ensure_loaded()
    except Exception as e:
        print(f"  [!] Failed to load {suite}: {e}")
        return []

    if not adapter.total_available:
        return []

    # Find T3 indices
    t3_indices = []
    for i in range(adapter.total_available):
        if adapter._get_tier_for_index(i) == 3:
            t3_indices.append(i)

    if not t3_indices:
        # Fall back to T2 if no T3
        for i in range(adapter.total_available):
            if adapter._get_tier_for_index(i) == 2:
                t3_indices.append(i)

    if not t3_indices:
        # Last resort: random sample
        t3_indices = list(range(min(100, adapter.total_available)))

    rng = random.Random(seed)
    selected = rng.sample(t3_indices, min(n, len(t3_indices)))

    return [adapter._row_to_prompt(i, adapter._dataset[i]) for i in selected]


def check_orchestrator_health(url: str = "http://localhost:8000") -> bool:
    """Check if orchestrator is running."""
    import httpx
    try:
        resp = httpx.get(f"{url}/health", timeout=5)
        return resp.status_code == 200
    except Exception as e:
        return False


def run_single_question(
    prompt: str,
    role: str,
    url: str = "http://localhost:8000",
    timeout: int = DEFAULT_TIMEOUT,
) -> dict:
    """Run a single question and measure timing."""
    import httpx

    payload = {
        "prompt": prompt,
        "real_mode": True,
        "force_role": role,
        "force_mode": "direct",
    }

    start = time.time()
    try:
        response = httpx.post(
            f"{url}/chat",
            json=payload,
            timeout=timeout,
        )
        elapsed = time.time() - start

        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "elapsed_seconds": elapsed,
                "tokens_generated": data.get("tokens_generated", 0),
                "generation_ms": data.get("generation_ms", 0),
                "prompt_eval_ms": data.get("prompt_eval_ms", 0),
                "answer_preview": data.get("answer", "")[:100],
            }
        else:
            return {
                "success": False,
                "elapsed_seconds": elapsed,
                "error": f"HTTP {response.status_code}",
            }
    except httpx.TimeoutException:
        return {
            "success": False,
            "elapsed_seconds": timeout,
            "error": f"Timeout after {timeout}s",
        }
    except Exception as e:
        return {
            "success": False,
            "elapsed_seconds": time.time() - start,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Calibrate timeout settings")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without running")
    parser.add_argument("--suite", type=str, help="Test specific suite only")
    parser.add_argument("--role", type=str, help="Test specific role only")
    parser.add_argument("--questions-per-suite", type=int, default=1, help="Questions per suite")
    parser.add_argument("--timeout", type=int, default=600, help="Max timeout to test (seconds)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="Orchestrator URL")
    args = parser.parse_args()

    suites = [args.suite] if args.suite else CALIBRATION_SUITES
    roles = [args.role] if args.role else CALIBRATION_ROLES

    print("=" * 70)
    print("TIMEOUT CALIBRATION")
    print("=" * 70)
    print(f"Suites: {suites}")
    print(f"Roles: {roles}")
    print(f"Questions per suite: {args.questions_per_suite}")
    print(f"Max timeout: {args.timeout}s")
    print(f"Current DEFAULT_TIMEOUT: {DEFAULT_TIMEOUT}s")
    print()

    # Sample questions
    print("Sampling hardest (T3) questions...")
    questions_by_suite: dict[str, list[dict]] = {}
    for suite in suites:
        qs = sample_hardest_questions(suite, args.questions_per_suite, args.seed)
        if qs:
            questions_by_suite[suite] = qs
            print(f"  {suite}: {len(qs)} T3 questions")
            for q in qs:
                print(f"    - {q['id']}: {q['prompt'][:60]}...")
        else:
            print(f"  {suite}: No questions available")
    print()

    if args.dry_run:
        print("[DRY RUN] Would test the above questions through these roles:")
        for role in roles:
            port = ROLE_PORT.get(role, "?")
            heavy = "HEAVY" if port in HEAVY_PORTS else ""
            print(f"  - {role} (port {port}) {heavy}")
        print("\nRun without --dry-run to execute calibration.")
        return

    # Check orchestrator
    if not check_orchestrator_health(args.url):
        print(f"ERROR: Orchestrator not responding at {args.url}")
        print("Start with: python3 scripts/server/orchestrator_stack.py start --hot-only")
        sys.exit(1)

    # Run calibration
    results: list[dict] = []
    total_tests = len(questions_by_suite) * len(roles) * args.questions_per_suite
    test_num = 0

    for suite, questions in questions_by_suite.items():
        print(f"\n{'='*70}")
        print(f"SUITE: {suite} (using DEFAULT_TIMEOUT: {DEFAULT_TIMEOUT}s)")
        print("=" * 70)

        for q in questions:
            print(f"\nQuestion: {q['id']}")
            print(f"Prompt: {q['prompt'][:80]}...")

            for role in roles:
                test_num += 1
                port = ROLE_PORT.get(role, 8080)
                heavy_tag = " [HEAVY]" if port in HEAVY_PORTS else ""

                print(f"\n  [{test_num}/{total_tests}] {role}{heavy_tag}...")

                result = run_single_question(
                    prompt=q["prompt"],
                    role=role,
                    url=args.url,
                    timeout=args.timeout,
                )

                result.update({
                    "suite": suite,
                    "question_id": q["id"],
                    "role": role,
                    "default_timeout": DEFAULT_TIMEOUT,
                })
                results.append(result)

                if result["success"]:
                    elapsed = result["elapsed_seconds"]
                    tokens = result.get("tokens_generated", 0)
                    gen_ms = result.get("generation_ms", 0)

                    # Flag if default timeout would fail
                    timeout_ok = "OK" if elapsed < DEFAULT_TIMEOUT else "WOULD TIMEOUT!"
                    print(f"    ✓ {elapsed:.1f}s ({tokens} tok, {gen_ms:.0f}ms gen) [{timeout_ok}]")
                else:
                    print(f"    ✗ {result.get('error', 'Unknown error')}")

    # Summary
    print("\n" + "=" * 70)
    print("CALIBRATION SUMMARY")
    print("=" * 70)

    # Group by suite
    by_suite: dict[str, list[dict]] = {}
    for r in results:
        by_suite.setdefault(r["suite"], []).append(r)

    max_observed_overall = 0.0

    for suite, suite_results in by_suite.items():
        successes = [r for r in suite_results if r["success"]]
        failures = [r for r in suite_results if not r["success"]]

        if successes:
            max_time = max(r["elapsed_seconds"] for r in successes)
            avg_time = sum(r["elapsed_seconds"] for r in successes) / len(successes)
            max_observed_overall = max(max_observed_overall, max_time)
        else:
            max_time = avg_time = 0

        print(f"\n{suite}:")
        print(f"  Successes: {len(successes)}, Failures: {len(failures)}")
        if successes:
            print(f"  Max observed: {max_time:.1f}s, Avg: {avg_time:.1f}s")
            if max_time > DEFAULT_TIMEOUT:
                print(f"  WARNING: Max time exceeds DEFAULT_TIMEOUT ({DEFAULT_TIMEOUT}s)!")

    # Output recommendation
    print("\n" + "=" * 70)
    print("CALIBRATION RESULT:")
    print("=" * 70)
    print(f"Max observed time across all suites: {max_observed_overall:.1f}s")
    print(f"Current DEFAULT_TIMEOUT: {DEFAULT_TIMEOUT}s")
    recommended = int((max_observed_overall * 2 + 29) // 30 * 30)  # 2x with 30s rounding
    if recommended > DEFAULT_TIMEOUT:
        print(f"RECOMMENDED: Increase DEFAULT_TIMEOUT to {recommended}s")
    else:
        print(f"Current timeout of {DEFAULT_TIMEOUT}s is sufficient (2x margin = {recommended}s)")

    # Save raw results
    output_path = PROJECT_ROOT / "benchmarks" / "results" / "timeout_calibration.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "args": vars(args),
            "results": results,
            "max_observed_seconds": max_observed_overall,
            "default_timeout": DEFAULT_TIMEOUT,
        }, f, indent=2)
    print(f"\nRaw results saved to: {output_path}")


if __name__ == "__main__":
    main()
