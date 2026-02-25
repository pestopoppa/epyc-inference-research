#!/usr/bin/env python3
from __future__ import annotations

"""
Orchestrator Benchmark Runner

Runs functional benchmarks against the orchestrator API to measure:
- Routing correctness (t1_routing)
- Delegation patterns (t2_delegation)
- Escalation behavior (t3_escalation)
- Adversarial handling (t4_adversarial)

Usage:
    python scripts/benchmark/bench_orchestrator.py --suite all
    python scripts/benchmark/bench_orchestrator.py --suite t1_routing --dry-run
    python scripts/benchmark/bench_orchestrator.py --suite t2_delegation --config-from checkpoint.yaml
"""

import argparse
import json
import os
import sys
import time
import signal
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import re

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import httpx
    import yaml
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install httpx pyyaml")
    sys.exit(1)


# Constants
PROMPT_DIR = PROJECT_ROOT / "benchmarks" / "prompts" / "v1" / "orchestrator"
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results" / "orchestrator"
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 120  # seconds per test


@dataclass
class TestResult:
    """Result of a single test case."""
    test_id: str
    suite: str
    passed: bool
    turns: int
    latency_ms: float
    functions_called: list[str] = field(default_factory=list)
    escalation_triggered: bool = False
    escalation_to: Optional[str] = None
    final_called: bool = False
    error: Optional[str] = None
    response_preview: str = ""
    criteria_results: dict = field(default_factory=dict)


@dataclass
class SuiteResult:
    """Aggregated results for a test suite."""
    suite: str
    total: int
    passed: int
    failed: int
    pass_rate: float
    avg_latency_ms: float
    avg_turns: float
    tests: list[TestResult] = field(default_factory=list)


def load_prompt(prompt_path: Path) -> dict:
    """Load a prompt file and parse its metadata."""
    content = prompt_path.read_text()

    # Split on --- separator
    parts = content.split("---", 1)
    prompt_text = parts[0].strip()

    metadata = {}
    if len(parts) > 1:
        # Parse YAML-like metadata after ---
        for line in parts[1].strip().split("\n"):
            if ":" in line and not line.strip().startswith("#"):
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                # Handle basic types
                if value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False
                elif value.isdigit():
                    value = int(value)
                metadata[key] = value

    return {
        "prompt": prompt_text,
        "metadata": metadata,
        "file": prompt_path.name
    }


def load_ground_truth(suite: str) -> dict:
    """Load ground truth for a suite."""
    gt_path = PROMPT_DIR / "ground_truth" / f"{suite.replace('t', 't')}_expected.json"

    # Handle naming: t1_routing -> t1_expected.json
    suite_num = suite.split("_")[0]  # t1, t2, t3, t4
    gt_path = PROMPT_DIR / "ground_truth" / f"{suite_num}_expected.json"

    if not gt_path.exists():
        print(f"Warning: Ground truth not found at {gt_path}")
        return {}

    with open(gt_path) as f:
        return json.load(f)


def load_suite_prompts(suite: str) -> list[dict]:
    """Load all prompts for a suite."""
    suite_dir = PROMPT_DIR / suite
    if not suite_dir.exists():
        print(f"Error: Suite directory not found: {suite_dir}")
        return []

    prompts = []
    for prompt_file in sorted(suite_dir.glob("*.txt")):
        prompts.append(load_prompt(prompt_file))

    return prompts


def call_orchestrator_api(
    prompt: str,
    api_url: str,
    timeout: int,
    real_mode: bool = True,
    config: Optional[dict] = None
) -> dict:
    """Call the orchestrator API and return the response."""

    payload = {
        "prompt": prompt,
        "real_mode": real_mode,
        "mock_mode": not real_mode,
    }

    # Apply config overrides
    if config:
        if "temperature" in config:
            payload["temperature"] = config["temperature"]
        if "max_turns" in config:
            payload["max_turns"] = config["max_turns"]

    try:
        with httpx.Client(timeout=timeout) as client:
            start = time.perf_counter()
            response = client.post(
                f"{api_url}/chat",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            latency_ms = (time.perf_counter() - start) * 1000

            if response.status_code != 200:
                return {
                    "error": f"HTTP {response.status_code}: {response.text[:200]}",
                    "latency_ms": latency_ms
                }

            result = response.json()
            result["latency_ms"] = latency_ms
            return result

    except httpx.TimeoutException:
        return {"error": "Timeout", "latency_ms": timeout * 1000}
    except httpx.ConnectError as e:
        return {"error": f"Connection failed: {e}", "latency_ms": 0}
    except Exception as e:
        return {"error": f"API call failed: {e}", "latency_ms": 0}


def mock_orchestrator_response(prompt: str, test_id: str) -> dict:
    """Generate mock response for dry-run mode."""

    # Simulate different behaviors based on test ID
    mock_responses = {
        "t1_direct_answer": {
            "answer": "4",
            "turns": 1,
            "functions_called": [],
            "final_called": True
        },
        "t1_context_peek": {
            "answer": "The first word is 'The'.",
            "turns": 2,
            "functions_called": ["peek"],
            "final_called": True
        },
        "t1_context_grep": {
            "answer": "Found ERROR on lines 15, 42, 87.",
            "turns": 3,
            "functions_called": ["grep"],
            "final_called": True
        },
        "t1_simple_delegate": {
            "answer": "This document discusses machine learning concepts...",
            "turns": 3,
            "functions_called": ["llm_call"],
            "final_called": True
        },
        "t2_parallel_summarize": {
            "answer": "Combined summary of all sections...",
            "turns": 4,
            "functions_called": ["llm_batch"],
            "final_called": True
        },
        "t2_iterative_refine": {
            "answer": "Refined code with improvements...",
            "turns": 5,
            "functions_called": ["llm_call", "llm_call", "llm_call"],
            "final_called": True
        },
        "t2_code_explain": {
            "answer": "This is a bubble sort implementation with O(n^2) complexity...",
            "turns": 4,
            "functions_called": ["llm_call"],
            "final_called": True
        },
        "t2_structured_extract": {
            "answer": '{"name": "John Doe", "email": "john@example.com", "phone": "555-1234"}',
            "turns": 3,
            "functions_called": ["llm_call"],
            "final_called": True
        },
        "t3_complex_code_gen": {
            "answer": "class ThreadSafeLRUCache:\n    def __init__(self)...",
            "turns": 6,
            "functions_called": ["escalate", "llm_call"],
            "escalation_triggered": True,
            "escalation_to": "coder",
            "final_called": True
        },
        "t3_architecture_design": {
            "answer": "Microservices architecture with API gateway...",
            "turns": 7,
            "functions_called": ["escalate", "llm_call"],
            "escalation_triggered": True,
            "escalation_to": "architect",
            "final_called": True
        },
        "t3_failure_recovery": {
            "answer": '{"name": "John Doe", "age": 30}',
            "turns": 5,
            "functions_called": ["llm_call", "llm_call"],
            "final_called": True
        },
        "t3_long_context_64k": {
            "answer": "The secret code is SECRET_CODE_XYZ123.",
            "turns": 4,
            "functions_called": ["grep", "peek"],
            "final_called": True
        },
        "t3_ambiguous_task": {
            "answer": "I've added type hints and a docstring to improve the code...",
            "turns": 4,
            "functions_called": ["llm_call"],
            "final_called": True
        },
        "t4_infinite_loop_trap": {
            "answer": "The code is already simple and well-written. No further improvements needed.",
            "turns": 3,
            "functions_called": ["llm_call"],
            "final_called": True
        },
        "t4_malformed_context": {
            "answer": "The provided data appears to contain malformed/binary content. Unable to analyze.",
            "turns": 2,
            "functions_called": [],
            "final_called": True
        },
        "t4_timeout_heavy": {
            "answer": "Partial summary of quantum computing covering history and basic principles...",
            "turns": 8,
            "functions_called": ["llm_call", "llm_call"],
            "final_called": True
        }
    }

    # Default mock response
    default = {
        "answer": "Mock response for testing.",
        "turns": 2,
        "functions_called": ["llm_call"],
        "final_called": True
    }

    response = mock_responses.get(test_id, default)
    response["latency_ms"] = 100 + (len(prompt) % 500)  # Fake latency
    return response


def evaluate_criteria(
    response: dict,
    criteria: dict,
    ground_truth: dict
) -> tuple[bool, dict]:
    """Evaluate response against pass criteria."""

    results = {}
    all_passed = True

    # Check max_turns
    if "max_turns" in criteria:
        turns = response.get("turns", 0)
        passed = turns <= criteria["max_turns"]
        results["max_turns"] = {"passed": passed, "actual": turns, "max": criteria["max_turns"]}
        if not passed:
            all_passed = False

    # Check must_call_final
    if criteria.get("must_call_final"):
        final_called = response.get("final_called", False)
        results["must_call_final"] = {"passed": final_called}
        if not final_called:
            all_passed = False

    # Check answer_contains
    if "answer_contains" in criteria:
        answer = response.get("answer", "")
        required = criteria["answer_contains"]
        found = all(term.lower() in answer.lower() for term in required)
        results["answer_contains"] = {"passed": found, "required": required}
        if not found:
            all_passed = False

    # Check answer_contains_any
    if "answer_contains_any" in criteria:
        answer = response.get("answer", "")
        options = criteria["answer_contains_any"]
        found = any(term.lower() in answer.lower() for term in options)
        results["answer_contains_any"] = {"passed": found, "options": options}
        if not found:
            all_passed = False

    # Check uses_function
    if "uses_function" in criteria:
        functions = response.get("functions_called", [])
        required = criteria["uses_function"]
        found = all(f in functions for f in required)
        results["uses_function"] = {"passed": found, "required": required, "actual": functions}
        if not found:
            all_passed = False

    # Check no_llm_calls
    if criteria.get("no_llm_calls"):
        functions = response.get("functions_called", [])
        llm_calls = [f for f in functions if f.startswith("llm_")]
        passed = len(llm_calls) == 0
        results["no_llm_calls"] = {"passed": passed, "actual": llm_calls}
        if not passed:
            all_passed = False

    # Check llm_calls_count
    if "llm_calls_count" in criteria:
        functions = response.get("functions_called", [])
        llm_calls = len([f for f in functions if f.startswith("llm_")])
        min_calls = criteria["llm_calls_count"].get("min", 0)
        max_calls = criteria["llm_calls_count"].get("max", float("inf"))
        passed = min_calls <= llm_calls <= max_calls
        results["llm_calls_count"] = {"passed": passed, "actual": llm_calls, "range": [min_calls, max_calls]}
        if not passed:
            all_passed = False

    # Check escalation_triggered
    if "escalation_triggered" in criteria:
        triggered = response.get("escalation_triggered", False)
        passed = triggered == criteria["escalation_triggered"]
        results["escalation_triggered"] = {"passed": passed}
        if not passed:
            all_passed = False

    # Check escalation_to
    if "escalation_to" in criteria:
        target = response.get("escalation_to", "")
        passed = target == criteria["escalation_to"]
        results["escalation_to"] = {"passed": passed, "expected": criteria["escalation_to"], "actual": target}
        if not passed:
            all_passed = False

    # Check answer_is_valid_json
    if criteria.get("answer_is_valid_json"):
        answer = response.get("answer", "")
        try:
            json.loads(answer)
            passed = True
        except (json.JSONDecodeError, TypeError):
            # Try to extract JSON from answer
            try:
                match = re.search(r'\{[^{}]*\}', answer)
                if match:
                    json.loads(match.group())
                    passed = True
                else:
                    passed = False
            except Exception as e:
                passed = False
        results["answer_is_valid_json"] = {"passed": passed}
        if not passed:
            all_passed = False

    # Check json_has_keys
    if "json_has_keys" in criteria:
        answer = response.get("answer", "")
        required_keys = criteria["json_has_keys"]
        try:
            # Try to extract JSON
            match = re.search(r'\{[^{}]*\}', answer)
            if match:
                data = json.loads(match.group())
            else:
                data = json.loads(answer)
            has_keys = all(k in data for k in required_keys)
        except Exception as e:
            has_keys = False
        results["json_has_keys"] = {"passed": has_keys, "required": required_keys}
        if not has_keys:
            all_passed = False

    # Check must_terminate
    if criteria.get("must_terminate"):
        # If we got a response without timeout, it terminated
        error = response.get("error", "")
        terminated = "Timeout" not in error
        results["must_terminate"] = {"passed": terminated}
        if not terminated:
            all_passed = False

    # Check must_not_crash
    if criteria.get("must_not_crash"):
        error = response.get("error", "")
        # Allow specific expected errors
        crashed = error and "crash" in error.lower()
        results["must_not_crash"] = {"passed": not crashed}
        if crashed:
            all_passed = False

    # Check one_of (alternative criteria)
    if "one_of" in criteria:
        one_of_options = criteria["one_of"]
        any_passed = False
        for option in one_of_options:
            option_results = {}
            option_passed = True
            for key, value in option.items():
                if key == "asks_clarification":
                    answer = response.get("answer", "").lower()
                    passed = any(q in answer for q in ["?", "clarify", "what do you mean", "could you specify"])
                    option_results[key] = passed
                    if value and not passed:
                        option_passed = False
                elif key == "answer_contains_any":
                    answer = response.get("answer", "")
                    found = any(term.lower() in answer.lower() for term in value)
                    option_results[key] = found
                    if not found:
                        option_passed = False
            if option_passed:
                any_passed = True
                break
        results["one_of"] = {"passed": any_passed, "options": one_of_options}
        if not any_passed:
            all_passed = False

    return all_passed, results


def run_test(
    prompt_data: dict,
    ground_truth: dict,
    api_url: str,
    timeout: int,
    dry_run: bool,
    config: Optional[dict] = None
) -> TestResult:
    """Run a single test and return the result."""

    # Extract test ID from metadata or filename
    test_id = prompt_data["metadata"].get("test_id", prompt_data["file"].replace(".txt", ""))
    suite = test_id.split("_")[0] + "_" + test_id.split("_")[1] if "_" in test_id else "unknown"

    # Find criteria from ground truth
    criteria = {}
    test_name = prompt_data["file"].replace(".txt", "")
    if "tests" in ground_truth:
        if test_name in ground_truth["tests"]:
            criteria = ground_truth["tests"][test_name].get("pass_criteria", {})
        elif test_id in ground_truth["tests"]:
            criteria = ground_truth["tests"][test_id].get("pass_criteria", {})

    # Also use metadata criteria
    for key in ["max_turns", "timeout_seconds", "must_call_final"]:
        if key in prompt_data["metadata"]:
            criteria[key] = prompt_data["metadata"][key]

    # Get response
    if dry_run:
        response = mock_orchestrator_response(prompt_data["prompt"], test_id)
    else:
        response = call_orchestrator_api(
            prompt_data["prompt"],
            api_url,
            timeout,
            real_mode=True,
            config=config
        )

    # Check for errors
    if "error" in response:
        return TestResult(
            test_id=test_id,
            suite=suite,
            passed=False,
            turns=0,
            latency_ms=response.get("latency_ms", 0),
            error=response["error"],
            final_called=False
        )

    # Evaluate criteria
    passed, criteria_results = evaluate_criteria(response, criteria, ground_truth)

    return TestResult(
        test_id=test_id,
        suite=suite,
        passed=passed,
        turns=response.get("turns", 0),
        latency_ms=response.get("latency_ms", 0),
        functions_called=response.get("functions_called", []),
        escalation_triggered=response.get("escalation_triggered", False),
        escalation_to=response.get("escalation_to"),
        final_called=response.get("final_called", False),
        response_preview=str(response.get("answer", ""))[:200],
        criteria_results=criteria_results
    )


def run_suite(
    suite: str,
    api_url: str,
    timeout: int,
    dry_run: bool,
    config: Optional[dict] = None
) -> SuiteResult:
    """Run all tests in a suite."""

    prompts = load_suite_prompts(suite)
    ground_truth = load_ground_truth(suite)

    if not prompts:
        return SuiteResult(
            suite=suite,
            total=0,
            passed=0,
            failed=0,
            pass_rate=0.0,
            avg_latency_ms=0.0,
            avg_turns=0.0
        )

    results = []
    for prompt_data in prompts:
        print(f"  Running: {prompt_data['file']}...", end=" ", flush=True)
        result = run_test(prompt_data, ground_truth, api_url, timeout, dry_run, config)
        results.append(result)
        status = "✓" if result.passed else "✗"
        print(f"{status} ({result.latency_ms:.0f}ms, {result.turns} turns)")

        if result.error:
            print(f"    Error: {result.error}")
        elif not result.passed:
            for key, val in result.criteria_results.items():
                if not val.get("passed"):
                    print(f"    Failed: {key} - {val}")

    passed = sum(1 for r in results if r.passed)
    total_latency = sum(r.latency_ms for r in results)
    total_turns = sum(r.turns for r in results)

    return SuiteResult(
        suite=suite,
        total=len(results),
        passed=passed,
        failed=len(results) - passed,
        pass_rate=passed / len(results) if results else 0.0,
        avg_latency_ms=total_latency / len(results) if results else 0.0,
        avg_turns=total_turns / len(results) if results else 0.0,
        tests=results
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_results(results: dict, output_path: Path):
    """Save results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclasses to dicts
    output = {}
    for suite, result in results.items():
        output[suite] = {
            "suite": result.suite,
            "total": result.total,
            "passed": result.passed,
            "failed": result.failed,
            "pass_rate": result.pass_rate,
            "avg_latency_ms": result.avg_latency_ms,
            "avg_turns": result.avg_turns,
            "tests": [asdict(t) for t in result.tests]
        }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def generate_report(results: dict) -> str:
    """Generate a markdown report from results."""
    lines = [
        "# Orchestrator Benchmark Report",
        f"\nGenerated: {datetime.now().isoformat()}",
        "\n## Summary\n",
        "| Suite | Passed | Total | Pass Rate | Avg Latency | Avg Turns |",
        "|-------|--------|-------|-----------|-------------|-----------|"
    ]

    for suite, result in results.items():
        lines.append(
            f"| {suite} | {result.passed} | {result.total} | "
            f"{result.pass_rate:.1%} | {result.avg_latency_ms:.0f}ms | "
            f"{result.avg_turns:.1f} |"
        )

    # Overall
    total_passed = sum(r.passed for r in results.values())
    total_tests = sum(r.total for r in results.values())
    overall_rate = total_passed / total_tests if total_tests else 0

    lines.extend([
        "",
        f"**Overall: {total_passed}/{total_tests} ({overall_rate:.1%})**",
        "\n## Test Details\n"
    ])

    for suite, result in results.items():
        lines.append(f"\n### {suite}\n")
        for test in result.tests:
            status = "✅" if test.passed else "❌"
            lines.append(f"- {status} **{test.test_id}**: {test.turns} turns, {test.latency_ms:.0f}ms")
            if test.error:
                lines.append(f"  - Error: {test.error}")
            elif not test.passed:
                for key, val in test.criteria_results.items():
                    if not val.get("passed"):
                        lines.append(f"  - Failed: {key}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Orchestrator Benchmark Runner")
    parser.add_argument(
        "--suite",
        choices=["all", "t1_routing", "t2_delegation", "t3_escalation", "t4_adversarial"],
        default="all",
        help="Test suite to run"
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"Orchestrator API URL (default: {DEFAULT_API_URL})"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout per test in seconds (default: {DEFAULT_TIMEOUT})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use mock responses instead of calling API"
    )
    parser.add_argument(
        "--config-from",
        help="Load configuration from YAML file"
    )
    parser.add_argument(
        "--output",
        help="Output file for results (JSON)"
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate markdown report"
    )

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config_from:
        config = load_config(args.config_from)
        print(f"Loaded config from: {args.config_from}")

    # Determine suites to run
    if args.suite == "all":
        suites = ["t1_routing", "t2_delegation", "t3_escalation", "t4_adversarial"]
    else:
        suites = [args.suite]

    print(f"\n{'='*60}")
    print(f"Orchestrator Benchmark - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"API URL: {args.api_url}")
    print(f"Timeout: {args.timeout}s")
    print(f"Mode: {'Dry Run (Mock)' if args.dry_run else 'Live'}")
    print(f"Suites: {', '.join(suites)}")
    print(f"{'='*60}\n")

    # Run suites
    results = {}
    for suite in suites:
        print(f"\n[{suite}]")
        print("-" * 40)
        results[suite] = run_suite(
            suite,
            args.api_url,
            args.timeout,
            args.dry_run,
            config
        )
        print(f"\nResult: {results[suite].passed}/{results[suite].total} passed ({results[suite].pass_rate:.1%})")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    total_passed = sum(r.passed for r in results.values())
    total_tests = sum(r.total for r in results.values())
    overall_rate = total_passed / total_tests if total_tests else 0

    for suite, result in results.items():
        status = "✓" if result.pass_rate >= get_target_rate(suite) else "✗"
        print(f"{status} {suite}: {result.passed}/{result.total} ({result.pass_rate:.1%})")

    print(f"\nOverall: {total_passed}/{total_tests} ({overall_rate:.1%})")

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = RESULTS_DIR / f"bench_{timestamp}.json"

    save_results(results, output_path)

    # Generate report if requested
    if args.generate_report:
        report = generate_report(results)
        report_path = output_path.with_suffix(".md")
        report_path.write_text(report)
        print(f"Report saved to: {report_path}")

    # Return exit code based on results
    sys.exit(0 if overall_rate >= 0.5 else 1)


def get_target_rate(suite: str) -> float:
    """Get target pass rate for a suite."""
    targets = {
        "t1_routing": 1.0,     # Must pass 100%
        "t2_delegation": 0.9,  # Target >90%
        "t3_escalation": 0.7,  # Target >70%
        "t4_adversarial": 0.5  # Target >50%
    }
    return targets.get(suite, 0.5)


if __name__ == "__main__":
    main()
