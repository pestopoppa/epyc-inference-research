#!/usr/bin/env python3
"""A/B Test Harness for TOON vs JSON format evaluation.

This script compares TOON and JSON formats across:
1. Token count efficiency
2. Model comprehension accuracy (instruction_precision benchmark)
3. Time-to-first-token (TTFT) latency

Usage:
    # Token count comparison only (no inference required)
    python ab_test_harness.py --mode tokens

    # Full A/B test (requires running models)
    python ab_test_harness.py --mode full --port 8080

    # TTFT benchmark (requires running models)
    python ab_test_harness.py --mode ttft --port 8083
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.services.toon_encoder import encode, is_available, should_use_toon


@dataclass
class TestCase:
    """A single test case for A/B comparison."""

    name: str
    description: str
    data: dict[str, Any]
    expected_toon_savings: float  # Expected % savings (0.0-1.0)


@dataclass
class TokenResult:
    """Token count comparison result."""

    name: str
    json_chars: int
    toon_chars: int
    json_tokens_est: int  # Estimate: chars / 4
    toon_tokens_est: int
    savings_pct: float
    meets_target: bool


@dataclass
class ABTestResult:
    """Result from A/B test run."""

    name: str
    json_accuracy: float
    toon_accuracy: float
    json_avg_tokens: float
    toon_avg_tokens: float
    accuracy_delta: float  # Positive = TOON better


# Test cases representing real orchestrator data patterns
TEST_CASES = [
    TestCase(
        name="file_listing_small",
        description="Directory with 5 files",
        data={
            "path": "/workspace/src",
            "files": [
                {"name": "api.py", "type": "file", "size": 1234},
                {"name": "utils.py", "type": "file", "size": 567},
                {"name": "models", "type": "dir", "size": None},
                {"name": "routes", "type": "dir", "size": None},
                {"name": "__init__.py", "type": "file", "size": 89},
            ],
            "total": 5,
        },
        expected_toon_savings=0.40,
    ),
    TestCase(
        name="file_listing_large",
        description="Directory with 20 files",
        data={
            "path": "/workspace/src/api/routes",
            "files": [
                {"name": f"route_{i}.py", "type": "file", "size": 1000 + i * 100}
                for i in range(20)
            ],
            "total": 20,
        },
        expected_toon_savings=0.55,
    ),
    TestCase(
        name="procedure_listing",
        description="Procedure registry output (10 procedures)",
        data={
            "procedures": [
                {
                    "name": f"procedure_{i}",
                    "description": f"Description for procedure {i}",
                    "tier": "B" if i < 5 else "C",
                    "requires_approval": i % 2 == 0,
                }
                for i in range(10)
            ]
        },
        expected_toon_savings=0.50,
    ),
    TestCase(
        name="memory_recall",
        description="Episodic memory results (8 episodes)",
        data={
            "results": [
                {
                    "episode_id": f"ep_{i:04d}",
                    "task_type": "code" if i % 2 == 0 else "research",
                    "similarity": 0.95 - i * 0.05,
                    "outcome": "success" if i < 6 else "escalated",
                }
                for i in range(8)
            ]
        },
        expected_toon_savings=0.50,
    ),
    TestCase(
        name="escalation_context",
        description="Escalation with 3 previous attempts",
        data={
            "task_id": "task_abc123",
            "failure_count": 3,
            "error_category": "FORMAT",
            "gate_name": "schema",
            "error_preview": "JSON parse error at line 42: unexpected token",
            "previous_attempts": [
                {"role": "coder", "error": "Missing field 'id'", "tokens": 1234},
                {"role": "coder", "error": "Invalid JSON", "tokens": 1456},
                {"role": "coder_escalation", "error": "Schema mismatch", "tokens": 2100},
            ],
        },
        expected_toon_savings=0.40,
    ),
    TestCase(
        name="tool_output_mixed",
        description="Mixed tool output with nested structure",
        data={
            "tool": "analyze_document",
            "success": True,
            "sections": [
                {"id": f"s{i}", "title": f"Section {i}", "page": i + 1}
                for i in range(6)
            ],
            "metadata": {"pages": 12, "words": 3500, "language": "en"},
        },
        expected_toon_savings=0.35,
    ),
]


def estimate_tokens(text: str) -> int:
    """Rough token estimate (4 chars per token average)."""
    return len(text) // 4


def run_token_comparison() -> list[TokenResult]:
    """Compare token counts for JSON vs TOON across all test cases."""
    results = []

    if not is_available():
        print("WARNING: toon_format not installed. Install with: pip install toon_format")
        return []

    for tc in TEST_CASES:
        json_str = json.dumps(tc.data, indent=2)
        toon_str = encode(tc.data, fallback_to_json=False)

        json_chars = len(json_str)
        toon_chars = len(toon_str)
        json_tokens = estimate_tokens(json_str)
        toon_tokens = estimate_tokens(toon_str)

        savings = 1.0 - (toon_chars / json_chars) if json_chars > 0 else 0.0
        meets_target = savings >= tc.expected_toon_savings * 0.8  # 80% of expected

        results.append(
            TokenResult(
                name=tc.name,
                json_chars=json_chars,
                toon_chars=toon_chars,
                json_tokens_est=json_tokens,
                toon_tokens_est=toon_tokens,
                savings_pct=savings * 100,
                meets_target=meets_target,
            )
        )

    return results


def print_token_results(results: list[TokenResult]) -> None:
    """Print token comparison results table."""
    print("\n" + "=" * 80)
    print("TOKEN COUNT COMPARISON: JSON vs TOON")
    print("=" * 80)
    print(f"{'Test Case':<25} {'JSON tok':>10} {'TOON tok':>10} {'Savings':>10} {'Target':>8}")
    print("-" * 80)

    total_json = 0
    total_toon = 0
    all_meet_target = True

    for r in results:
        status = "PASS" if r.meets_target else "FAIL"
        print(
            f"{r.name:<25} {r.json_tokens_est:>10} {r.toon_tokens_est:>10} "
            f"{r.savings_pct:>9.1f}% {status:>8}"
        )
        total_json += r.json_tokens_est
        total_toon += r.toon_tokens_est
        if not r.meets_target:
            all_meet_target = False

    print("-" * 80)
    total_savings = (1.0 - total_toon / total_json) * 100 if total_json > 0 else 0
    print(f"{'TOTAL':<25} {total_json:>10} {total_toon:>10} {total_savings:>9.1f}%")
    print("=" * 80)

    if all_meet_target:
        print("RESULT: All test cases meet savings targets")
    else:
        print("RESULT: Some test cases below savings targets")


def run_ttft_benchmark(port: int = 8083, n_trials: int = 5) -> dict[str, float]:
    """Benchmark time-to-first-token with JSON vs TOON context.

    Requires a running llama-server on the specified port.

    Args:
        port: Server port (default 8083 for architect_general).
        n_trials: Number of trials per format.

    Returns:
        Dict with json_ttft_ms, toon_ttft_ms, speedup_pct.
    """
    try:
        import httpx
    except ImportError:
        print("ERROR: httpx not installed. Run: pip install httpx")
        return {}

    if not is_available():
        print("ERROR: toon_format not installed")
        return {}

    # Use a reasonably large test case
    test_data = TEST_CASES[1].data  # file_listing_large

    json_context = json.dumps(test_data, indent=2)
    toon_context = encode(test_data, fallback_to_json=False)

    base_prompt = "Analyze this data and list the 3 largest files:\n\n"

    json_times = []
    toon_times = []

    client = httpx.Client(timeout=60.0)
    url = f"http://localhost:{port}/completion"

    for trial in range(n_trials):
        # JSON trial
        start = time.perf_counter()
        try:
            resp = client.post(
                url,
                json={
                    "prompt": base_prompt + json_context,
                    "n_predict": 1,
                    "stream": False,
                },
            )
            elapsed = (time.perf_counter() - start) * 1000
            if resp.status_code == 200:
                json_times.append(elapsed)
        except Exception as e:
            print(f"JSON trial {trial} failed: {e}")

        # TOON trial
        start = time.perf_counter()
        try:
            resp = client.post(
                url,
                json={
                    "prompt": base_prompt + toon_context,
                    "n_predict": 1,
                    "stream": False,
                },
            )
            elapsed = (time.perf_counter() - start) * 1000
            if resp.status_code == 200:
                toon_times.append(elapsed)
        except Exception as e:
            print(f"TOON trial {trial} failed: {e}")

    client.close()

    if not json_times or not toon_times:
        print("ERROR: No successful trials. Is the server running?")
        return {}

    json_avg = sum(json_times) / len(json_times)
    toon_avg = sum(toon_times) / len(toon_times)
    speedup = ((json_avg - toon_avg) / json_avg) * 100 if json_avg > 0 else 0

    print("\n" + "=" * 60)
    print("TIME-TO-FIRST-TOKEN BENCHMARK")
    print("=" * 60)
    print(f"JSON context: {len(json_context)} chars ({estimate_tokens(json_context)} tokens est)")
    print(f"TOON context: {len(toon_context)} chars ({estimate_tokens(toon_context)} tokens est)")
    print(f"Trials: {n_trials}")
    print("-" * 60)
    print(f"JSON TTFT: {json_avg:.1f} ms")
    print(f"TOON TTFT: {toon_avg:.1f} ms")
    print(f"Speedup:   {speedup:.1f}%")
    print("=" * 60)

    return {"json_ttft_ms": json_avg, "toon_ttft_ms": toon_avg, "speedup_pct": speedup}


def main():
    parser = argparse.ArgumentParser(description="A/B test harness for TOON vs JSON")
    parser.add_argument(
        "--mode",
        choices=["tokens", "ttft", "full"],
        default="tokens",
        help="Test mode: tokens (no inference), ttft (latency), full (accuracy + tokens)",
    )
    parser.add_argument(
        "--port", type=int, default=8083, help="Server port for inference tests"
    )
    parser.add_argument(
        "--trials", type=int, default=5, help="Number of trials for TTFT benchmark"
    )
    args = parser.parse_args()

    print("TOON Format A/B Test Harness")
    print(f"Mode: {args.mode}")
    print(f"TOON available: {is_available()}")
    print()

    if args.mode in ("tokens", "full"):
        results = run_token_comparison()
        if results:
            print_token_results(results)

    if args.mode == "ttft":
        run_ttft_benchmark(port=args.port, n_trials=args.trials)

    if args.mode == "full":
        print("\n[INFO] Full accuracy benchmark requires instruction_precision suite.")
        print("Run: ./scripts/benchmark/run_overnight_benchmark_suite.sh --suite instruction_precision")
        print("With TOON_TOOL_OUTPUT=1 environment variable for TOON variant.")


if __name__ == "__main__":
    main()
