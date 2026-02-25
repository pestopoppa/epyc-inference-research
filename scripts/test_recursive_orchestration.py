#!/usr/bin/env python3
from __future__ import annotations

"""End-to-end validation for recursive orchestration.

This script validates the full recursive LLM pattern:
1. Root LM (frontdoor) receives a task
2. Root LM generates Python code that calls sub-LMs
3. Sub-LMs process chunks using RadixAttention
4. Results are aggregated and returned via FINAL()

Usage:
    # Start servers first, then:
    python scripts/test_recursive_orchestration.py

    # Test with custom API endpoint
    python scripts/test_recursive_orchestration.py --api http://localhost:8000

    # Verbose mode
    python scripts/test_recursive_orchestration.py -v

Requirements:
    - API server running: uvicorn src.api:app --port 8000
    - llama-server instances on ports 8080 (frontdoor) and 8082 (worker)
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class TestCase:
    """A test case for recursive orchestration."""

    name: str
    prompt: str
    context: str
    expected_behavior: str
    max_turns: int = 10


# Test cases that exercise the recursive pattern
TEST_CASES = [
    TestCase(
        name="simple_direct_answer",
        prompt="What is 2 + 2?",
        context="",
        expected_behavior="Should answer directly without sub-LM calls",
        max_turns=3,
    ),
    TestCase(
        name="context_peek",
        prompt="What is the first word in the context?",
        context="Hello world this is a test document with multiple words.",
        expected_behavior="Should use peek() to examine context and answer 'Hello'",
        max_turns=3,
    ),
    TestCase(
        name="context_grep",
        prompt="Find all lines containing 'error' in the context.",
        context="""
Log entry 1: System started
Log entry 2: Connection error occurred
Log entry 3: Processing complete
Log entry 4: Another error detected
Log entry 5: Shutdown initiated
        """.strip(),
        expected_behavior="Should use grep('error') to find matching lines",
        max_turns=3,
    ),
    TestCase(
        name="summarization_with_llm_call",
        prompt="Summarize the following document in one sentence.",
        context="""
The quick brown fox jumps over the lazy dog. This sentence contains every
letter of the English alphabet and has been used for typing practice since
the late 1800s. It was popularized by typewriter companies and later by
computer keyboard manufacturers. The sentence is exactly 35 letters long
when counting only the letters.
        """.strip(),
        expected_behavior="Should call llm_call() with extracted content",
        max_turns=5,
    ),
    TestCase(
        name="chunked_processing",
        prompt="Process each section separately and combine results.",
        context="""
## Section 1
First section content about topic A.

## Section 2
Second section content about topic B.

## Section 3
Third section content about topic C.
        """.strip(),
        expected_behavior="Should use grep() to find sections, then llm_batch() for parallel processing",
        max_turns=10,
    ),
]


def test_mock_mode(api_url: str, verbose: bool = False) -> dict:
    """Test mock mode endpoint."""
    import requests

    print("\n=== Testing Mock Mode ===")

    response = requests.post(
        f"{api_url}/chat",
        json={
            "prompt": "Hello, test!",
            "mock_mode": True,
        },
    )

    result = response.json()

    if verbose:
        print(f"Response: {json.dumps(result, indent=2)}")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert result["mock_mode"] is True, "Expected mock_mode=True"
    assert "[MOCK]" in result["answer"], "Expected [MOCK] in answer"

    print("  [PASS] Mock mode works correctly")
    return result


def test_real_mode_health(api_url: str, verbose: bool = False) -> dict:
    """Test real mode server health."""
    import requests

    print("\n=== Testing Real Mode Health ===")

    # First check API health
    response = requests.get(f"{api_url}/health")
    result = response.json()

    if verbose:
        print(f"Health: {json.dumps(result, indent=2)}")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert result["status"] == "ok", f"Expected status=ok, got {result['status']}"

    print("  [PASS] API is healthy")
    return result


def test_recursive_orchestration(
    api_url: str,
    test_case: TestCase,
    verbose: bool = False,
) -> dict:
    """Run a single test case."""
    import requests

    print(f"\n=== Test: {test_case.name} ===")
    print(f"  Expected: {test_case.expected_behavior}")

    start_time = time.time()

    try:
        response = requests.post(
            f"{api_url}/chat",
            json={
                "prompt": test_case.prompt,
                "context": test_case.context,
                "real_mode": True,
                "max_turns": test_case.max_turns,
            },
            timeout=120,  # 2 minute timeout per test
        )

        elapsed = time.time() - start_time

        if response.status_code != 200:
            print(f"  [FAIL] HTTP {response.status_code}: {response.text}")
            return {
                "name": test_case.name,
                "passed": False,
                "error": f"HTTP {response.status_code}",
                "elapsed": elapsed,
            }

        result = response.json()

        if verbose:
            print(f"  Response: {json.dumps(result, indent=2)}")

        # Basic validation
        passed = True
        issues = []

        if result.get("real_mode") is not True:
            issues.append("real_mode not True")
            passed = False

        if result.get("answer", "").startswith("[ERROR"):
            issues.append(f"Error in answer: {result['answer'][:100]}")
            passed = False

        if result.get("turns", 0) >= test_case.max_turns:
            issues.append(f"Hit max turns ({test_case.max_turns})")
            # Not necessarily a failure - some tasks may need more turns

        # Print result
        if passed:
            print(f"  [PASS] Completed in {result.get('turns', 0)} turns, {elapsed:.2f}s")
            if result.get("cache_stats"):
                for role, stats in result["cache_stats"].items():
                    if isinstance(stats, dict):
                        hit_rate = stats.get("router_hit_rate", 0) * 100
                        print(f"    Cache ({role}): {hit_rate:.1f}% hit rate")
        else:
            print(f"  [FAIL] {', '.join(issues)}")

        if verbose:
            print(f"  Answer: {result.get('answer', '')[:200]}...")

        return {
            "name": test_case.name,
            "passed": passed,
            "issues": issues,
            "turns": result.get("turns", 0),
            "elapsed": elapsed,
            "answer": result.get("answer", "")[:500],
            "cache_stats": result.get("cache_stats"),
        }

    except requests.exceptions.ConnectionError as e:
        print(f"  [FAIL] Connection error: {e}")
        return {
            "name": test_case.name,
            "passed": False,
            "error": "Connection error - is the API server running?",
            "elapsed": time.time() - start_time,
        }

    except requests.exceptions.Timeout:
        print(f"  [FAIL] Request timed out after 120s")
        return {
            "name": test_case.name,
            "passed": False,
            "error": "Timeout",
            "elapsed": 120,
        }

    except Exception as e:
        print(f"  [FAIL] Exception: {e}")
        return {
            "name": test_case.name,
            "passed": False,
            "error": str(e),
            "elapsed": time.time() - start_time,
        }


def run_all_tests(api_url: str, verbose: bool = False) -> dict:
    """Run all test cases and report results."""
    print("=" * 60)
    print("RECURSIVE ORCHESTRATION END-TO-END VALIDATION")
    print("=" * 60)
    print(f"API URL: {api_url}")

    results = {
        "api_url": api_url,
        "tests": [],
        "passed": 0,
        "failed": 0,
        "total_time": 0,
    }

    start_time = time.time()

    # Test mock mode first
    try:
        test_mock_mode(api_url, verbose)
        results["mock_mode"] = "pass"
    except Exception as e:
        print(f"  [FAIL] Mock mode test failed: {e}")
        results["mock_mode"] = f"fail: {e}"
        results["failed"] += 1

    # Test health
    try:
        test_real_mode_health(api_url, verbose)
        results["health"] = "pass"
    except Exception as e:
        print(f"  [FAIL] Health check failed: {e}")
        results["health"] = f"fail: {e}"
        results["failed"] += 1

    # Run test cases
    for test_case in TEST_CASES:
        result = test_recursive_orchestration(api_url, test_case, verbose)
        results["tests"].append(result)

        if result.get("passed"):
            results["passed"] += 1
        else:
            results["failed"] += 1

    results["total_time"] = time.time() - start_time

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Passed: {results['passed']}/{len(TEST_CASES)}")
    print(f"  Failed: {results['failed']}/{len(TEST_CASES)}")
    print(f"  Time:   {results['total_time']:.2f}s")
    print("=" * 60)

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="End-to-end validation for recursive orchestration"
    )
    parser.add_argument(
        "--api",
        default="http://localhost:8000",
        help="API endpoint URL",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--output",
        help="Save results to JSON file",
    )

    args = parser.parse_args()

    results = run_all_tests(args.api, args.verbose)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    # Return exit code based on failures
    return 0 if results["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
