#!/usr/bin/env python3
"""Comprehensive TOON Framework Test Suite.

Tests all TOON encoding paths across all large model invocation patterns
to determine if TOON provides meaningful benefits for orchestrator performance.

Usage:
    python scripts/toon/comprehensive_toon_test.py [--live] [--port PORT]

Options:
    --live      Run live inference tests (requires running servers)
    --port      Base port for orchestrator (default: 8080)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.services.toon_encoder import (
    is_available,
    encode,
    decode,
    should_use_toon,
    encode_list_dir,
    encode_grep_hits,
    encode_escalation_context,
    encode_procedures,
    encode_memory_results,
)


# =============================================================================
# Test Data Generators
# =============================================================================

def generate_file_listing(n: int, include_nested: bool = False) -> dict:
    """Generate realistic file listing data."""
    files = []
    for i in range(n):
        entry = {
            "name": f"file_{i:04d}.py" if i % 3 != 0 else f"directory_{i:04d}",
            "type": "file" if i % 3 != 0 else "dir",
            "size": (i * 1234) % 100000 if i % 3 != 0 else None,
        }
        if include_nested and i % 5 == 0:
            entry["metadata"] = {"modified": f"2026-01-{(i % 28) + 1:02d}", "perms": "644"}
        files.append(entry)
    return {"path": "/mnt/raid0/llm/claude/src", "files": files, "total": n}


def generate_escalation_context(
    failure_count: int,
    include_attempts: bool = True,
    attempt_count: int = 3
) -> dict:
    """Generate realistic escalation context."""
    ctx = {
        "task_id": "task_abc123",
        "failure_count": failure_count,
        "error_category": "FORMAT",
        "gate_name": "schema_validation",
        "error_preview": "JSON parse error at line 45: unexpected token '}'",
    }
    if include_attempts:
        ctx["previous_attempts"] = [
            {
                "attempt": i + 1,
                "role": "coder" if i < 2 else "coder_escalation",
                "error": f"Attempt {i+1} failed: {'JSON parse error' if i == 0 else 'Missing required field'}",
                "tokens_used": 1000 + i * 500,
                "duration_ms": 2500 + i * 1000,
            }
            for i in range(attempt_count)
        ]
    return ctx


def generate_procedures(n: int) -> list[dict]:
    """Generate realistic procedure registry entries."""
    procedures = [
        {"name": "code_review", "description": "Review code for quality and bugs", "trigger": "on_pr"},
        {"name": "run_tests", "description": "Execute test suite with coverage", "trigger": "on_commit"},
        {"name": "deploy_staging", "description": "Deploy to staging environment", "trigger": "manual"},
        {"name": "security_scan", "description": "Run security vulnerability scan", "trigger": "on_pr"},
        {"name": "performance_bench", "description": "Run performance benchmarks", "trigger": "nightly"},
    ]
    # Cycle through base procedures to reach n
    return [
        {**procedures[i % len(procedures)], "id": f"proc_{i:03d}"}
        for i in range(n)
    ]


def generate_memory_results(n: int, include_complex: bool = False) -> list[dict]:
    """Generate realistic episodic memory recall results."""
    results = []
    for i in range(n):
        entry = {
            "memory_id": f"mem_{i:06d}",
            "query": f"How to handle error type {i % 10}?",
            "similarity_score": 0.95 - (i * 0.05),
            "q_value": 0.8 - (i * 0.1),
            "action_taken": f"Applied fix strategy {chr(65 + (i % 5))}",
            "outcome": "success" if i % 3 != 0 else "partial",
        }
        if include_complex:
            entry["context"] = {
                "task_type": "code_fix",
                "files_modified": [f"src/module_{j}.py" for j in range(i % 4 + 1)],
                "error_trace": f"Traceback (most recent call last):\n  File 'main.py', line {i*10}..."[:200],
            }
        results.append(entry)
    return results


def generate_grep_hits(n_patterns: int, hits_per_pattern: int) -> list[dict]:
    """Generate realistic grep search results."""
    patterns = ["def main", "class Error", "import asyncio", "async def", "raise ValueError"]
    return [
        {
            "pattern": patterns[i % len(patterns)],
            "hits": [
                {
                    "file": f"src/module_{j}.py",
                    "line_num": 10 + j * 5,
                    "match": f"    {patterns[i % len(patterns)]}(arg{j}): # implementation"[:200],
                }
                for j in range(hits_per_pattern)
            ],
        }
        for i in range(n_patterns)
    ]


# =============================================================================
# Test Cases
# =============================================================================

@dataclass
class TestResult:
    """Result of a single TOON test."""
    name: str
    passed: bool
    json_chars: int
    toon_chars: int
    reduction_pct: float
    roundtrip_ok: bool
    error: str | None = None
    details: dict | None = None


def estimate_tokens(text: str) -> int:
    """Estimate token count (chars / 4 heuristic)."""
    return len(text) // 4


def test_roundtrip(data: Any, name: str) -> TestResult:
    """Test TOON encode/decode roundtrip fidelity."""
    if not is_available():
        return TestResult(
            name=name,
            passed=False,
            json_chars=0,
            toon_chars=0,
            reduction_pct=0,
            roundtrip_ok=False,
            error="TOON not available",
        )

    try:
        json_str = json.dumps(data, indent=2)
        toon_str = encode(data, fallback_to_json=False)
        decoded = decode(toon_str)

        # Check roundtrip fidelity
        roundtrip_ok = decoded == data

        reduction = 1 - (len(toon_str) / len(json_str)) if len(json_str) > 0 else 0

        return TestResult(
            name=name,
            passed=roundtrip_ok and reduction > 0,
            json_chars=len(json_str),
            toon_chars=len(toon_str),
            reduction_pct=reduction * 100,
            roundtrip_ok=roundtrip_ok,
            details={"json_tokens": estimate_tokens(json_str), "toon_tokens": estimate_tokens(toon_str)},
        )
    except Exception as e:
        return TestResult(
            name=name,
            passed=False,
            json_chars=0,
            toon_chars=0,
            reduction_pct=0,
            roundtrip_ok=False,
            error=str(e),
        )


def test_should_use_toon(data: Any, expected: bool, name: str) -> TestResult:
    """Test should_use_toon heuristic accuracy."""
    result = should_use_toon(data)
    passed = result == expected

    json_str = json.dumps(data, indent=2)

    return TestResult(
        name=name,
        passed=passed,
        json_chars=len(json_str),
        toon_chars=0,
        reduction_pct=0,
        roundtrip_ok=True,
        error=None if passed else f"Expected {expected}, got {result}",
        details={"should_use_toon": result, "expected": expected},
    )


# =============================================================================
# Test Suites by Model Invocation Pattern
# =============================================================================

def suite_file_listings() -> list[TestResult]:
    """Test file listing encoding across size ranges."""
    results = []

    # Edge cases
    results.append(test_roundtrip(generate_file_listing(0), "file_listing_empty"))
    results.append(test_roundtrip(generate_file_listing(1), "file_listing_single"))
    results.append(test_roundtrip(generate_file_listing(2), "file_listing_two"))

    # Should NOT use TOON
    results.append(test_should_use_toon(generate_file_listing(2), False, "heuristic_small_listing"))

    # Typical ranges
    results.append(test_roundtrip(generate_file_listing(5), "file_listing_5"))
    results.append(test_roundtrip(generate_file_listing(10), "file_listing_10"))
    results.append(test_roundtrip(generate_file_listing(25), "file_listing_25"))
    results.append(test_roundtrip(generate_file_listing(50), "file_listing_50"))
    results.append(test_roundtrip(generate_file_listing(100), "file_listing_100"))

    # Should use TOON
    results.append(test_should_use_toon(generate_file_listing(10), True, "heuristic_medium_listing"))

    # Large (stress test)
    results.append(test_roundtrip(generate_file_listing(500), "file_listing_500_stress"))

    # With nested metadata
    results.append(test_roundtrip(generate_file_listing(20, include_nested=True), "file_listing_nested"))

    # Test encode_list_dir helper
    listing = generate_file_listing(15)
    encoded = encode_list_dir(listing["path"], listing["files"], listing["total"])
    try:
        # Should be TOON, not JSON
        is_toon = not encoded.strip().startswith("{")
        results.append(TestResult(
            name="encode_list_dir_uses_toon",
            passed=is_toon,
            json_chars=len(json.dumps(listing, indent=2)),
            toon_chars=len(encoded),
            reduction_pct=(1 - len(encoded) / len(json.dumps(listing, indent=2))) * 100,
            roundtrip_ok=True,
        ))
    except Exception as e:
        results.append(TestResult(
            name="encode_list_dir_uses_toon",
            passed=False,
            json_chars=0,
            toon_chars=0,
            reduction_pct=0,
            roundtrip_ok=False,
            error=str(e),
        ))

    return results


def suite_escalation_context() -> list[TestResult]:
    """Test escalation context encoding for architect models."""
    results = []

    # No previous attempts
    results.append(test_roundtrip(
        generate_escalation_context(1, include_attempts=False),
        "escalation_no_attempts"
    ))

    # Single attempt (should NOT use TOON)
    ctx_1 = generate_escalation_context(1, include_attempts=True, attempt_count=1)
    results.append(test_roundtrip(ctx_1, "escalation_1_attempt"))
    results.append(test_should_use_toon(ctx_1, False, "heuristic_1_attempt"))

    # Multiple attempts (should use TOON)
    ctx_3 = generate_escalation_context(3, include_attempts=True, attempt_count=3)
    results.append(test_roundtrip(ctx_3, "escalation_3_attempts"))

    ctx_5 = generate_escalation_context(5, include_attempts=True, attempt_count=5)
    results.append(test_roundtrip(ctx_5, "escalation_5_attempts"))

    # Test encode_escalation_context helper
    encoded = encode_escalation_context(
        task_id="test_123",
        failure_count=3,
        error_category="FORMAT",
        gate_name="schema",
        error_message="Parse error at line 42",
        previous_attempts=[
            {"attempt": i, "role": "coder", "error": f"Error {i}"}
            for i in range(4)
        ],
    )
    is_toon = not encoded.strip().startswith("{")
    results.append(TestResult(
        name="encode_escalation_context_helper",
        passed=is_toon,
        json_chars=len(json.dumps(ctx_5, indent=2)),
        toon_chars=len(encoded),
        reduction_pct=0,  # Different structure
        roundtrip_ok=True,
    ))

    return results


def suite_procedures() -> list[TestResult]:
    """Test procedure listing encoding."""
    results = []

    # Edge cases
    results.append(test_roundtrip({"procedures": generate_procedures(0)}, "procedures_empty"))
    results.append(test_roundtrip({"procedures": generate_procedures(2)}, "procedures_2"))

    # Typical ranges
    results.append(test_roundtrip({"procedures": generate_procedures(5)}, "procedures_5"))
    results.append(test_roundtrip({"procedures": generate_procedures(10)}, "procedures_10"))
    results.append(test_roundtrip({"procedures": generate_procedures(25)}, "procedures_25"))

    # Test helper
    procs = generate_procedures(8)
    encoded = encode_procedures(procs)
    is_toon = not encoded.strip().startswith("[") and not encoded.strip().startswith("{")
    results.append(TestResult(
        name="encode_procedures_helper",
        passed=is_toon or "procedures" in encoded,  # Either TOON or wrapped
        json_chars=len(json.dumps(procs, indent=2)),
        toon_chars=len(encoded),
        reduction_pct=(1 - len(encoded) / len(json.dumps(procs, indent=2))) * 100 if len(json.dumps(procs)) > 0 else 0,
        roundtrip_ok=True,
    ))

    return results


def suite_memory_results() -> list[TestResult]:
    """Test episodic memory result encoding."""
    results = []

    # Edge cases
    results.append(test_roundtrip({"results": generate_memory_results(0)}, "memory_empty"))
    results.append(test_roundtrip({"results": generate_memory_results(2)}, "memory_2"))

    # Typical ranges (MemRL retrieval usually returns 5-20 results)
    results.append(test_roundtrip({"results": generate_memory_results(5)}, "memory_5"))
    results.append(test_roundtrip({"results": generate_memory_results(10)}, "memory_10"))
    results.append(test_roundtrip({"results": generate_memory_results(20)}, "memory_20"))

    # Complex with nested context
    results.append(test_roundtrip(
        {"results": generate_memory_results(10, include_complex=True)},
        "memory_10_complex"
    ))

    # Test helper
    mem = generate_memory_results(8)
    encoded = encode_memory_results(mem)
    results.append(TestResult(
        name="encode_memory_results_helper",
        passed=True,  # Just check it doesn't crash
        json_chars=len(json.dumps({"results": mem}, indent=2)),
        toon_chars=len(encoded),
        reduction_pct=(1 - len(encoded) / len(json.dumps({"results": mem}, indent=2))) * 100,
        roundtrip_ok=True,
    ))

    return results


def suite_grep_hits() -> list[TestResult]:
    """Test grep hit encoding (expect TOON to be WORSE here)."""
    results = []

    # Various sizes
    for patterns, hits in [(1, 3), (3, 5), (5, 10), (10, 5)]:
        grep_data = generate_grep_hits(patterns, hits)
        json_str = json.dumps(grep_data, indent=2)
        md_str = encode_grep_hits(grep_data)

        # Markdown should be more compact than JSON for grep
        md_better = len(md_str) < len(json_str)

        results.append(TestResult(
            name=f"grep_{patterns}p_{hits}h_markdown_vs_json",
            passed=md_better,
            json_chars=len(json_str),
            toon_chars=len(md_str),  # Using MD as "toon" slot for comparison
            reduction_pct=(1 - len(md_str) / len(json_str)) * 100 if len(json_str) > 0 else 0,
            roundtrip_ok=True,  # MD is not meant to roundtrip
            details={"format": "markdown", "patterns": patterns, "hits_per": hits},
        ))

    return results


def suite_edge_cases() -> list[TestResult]:
    """Test edge cases and potential failure modes."""
    results = []

    # Unicode handling
    unicode_data = {
        "files": [
            {"name": "日本語ファイル.py", "type": "file", "size": 123},
            {"name": "émoji_🎉.txt", "type": "file", "size": 456},
            {"name": "中文文件.md", "type": "file", "size": 789},
        ]
    }
    results.append(test_roundtrip(unicode_data, "unicode_filenames"))

    # Special characters in strings
    special_data = {
        "files": [
            {"name": "file with spaces.py", "type": "file", "size": 123},
            {"name": "file,with,commas.txt", "type": "file", "size": 456},
            {"name": "file:with:colons.md", "type": "file", "size": 789},
            {"name": "file\twith\ttabs.json", "type": "file", "size": 101},
        ]
    }
    results.append(test_roundtrip(special_data, "special_characters"))

    # Null values
    null_data = {
        "files": [
            {"name": "file.py", "type": "file", "size": None},
            {"name": "dir", "type": "dir", "size": None},
        ]
    }
    results.append(test_roundtrip(null_data, "null_values"))

    # Deeply nested (TOON may not help)
    deep_nested = {
        "level1": {
            "level2": {
                "level3": {
                    "items": [{"a": i, "b": i*2} for i in range(5)]
                }
            }
        }
    }
    results.append(test_roundtrip(deep_nested, "deeply_nested"))
    results.append(test_should_use_toon(deep_nested, True, "heuristic_deeply_nested"))

    # Non-uniform arrays (TOON should NOT help)
    non_uniform = {
        "items": [
            {"name": "a", "value": 1},
            {"name": "b", "count": 2},  # Different key
            {"name": "c", "value": 3},
        ]
    }
    results.append(test_should_use_toon(non_uniform, False, "heuristic_non_uniform"))

    # Mixed types in array
    mixed_types = {
        "data": [1, "two", 3.0, None, True, {"nested": "dict"}]
    }
    results.append(test_should_use_toon(mixed_types, False, "heuristic_mixed_types"))

    # Very long strings (potential truncation issues)
    long_strings = {
        "files": [
            {"name": "a" * 500, "type": "file", "size": i}
            for i in range(5)
        ]
    }
    results.append(test_roundtrip(long_strings, "very_long_strings"))

    # Empty strings
    empty_strings = {
        "files": [
            {"name": "", "type": "", "size": 0},
            {"name": "a", "type": "file", "size": 123},
        ]
    }
    results.append(test_roundtrip(empty_strings, "empty_strings"))

    # Boolean values
    bool_data = {
        "items": [
            {"name": f"item_{i}", "active": i % 2 == 0, "count": i}
            for i in range(5)
        ]
    }
    results.append(test_roundtrip(bool_data, "boolean_values"))

    # Numeric edge cases
    numeric_data = {
        "items": [
            {"name": "zero", "value": 0},
            {"name": "negative", "value": -123},
            {"name": "float", "value": 3.14159},
            {"name": "scientific", "value": 1.23e-10},
            {"name": "large", "value": 9999999999999},
        ]
    }
    results.append(test_roundtrip(numeric_data, "numeric_edge_cases"))

    return results


def suite_orchestration_scenarios() -> list[TestResult]:
    """Test realistic orchestration data combinations."""
    results = []

    # Scenario 1: Code review escalation with file context
    code_review = {
        "task": {
            "id": "review_pr_123",
            "type": "code_review",
            "priority": "high",
        },
        "files_changed": [
            {"path": f"src/module_{i}.py", "additions": i*10, "deletions": i*5}
            for i in range(8)
        ],
        "escalation": {
            "from_tier": "B1",
            "to_tier": "B3",
            "reason": "Complex architectural decision required",
            "attempts": [
                {"tier": "B1", "outcome": "needs_escalation", "tokens": 2500},
                {"tier": "B1", "outcome": "retry_failed", "tokens": 3200},
            ],
        },
    }
    results.append(test_roundtrip(code_review, "scenario_code_review"))

    # Scenario 2: Long context ingestion summary
    ingestion = {
        "document": {
            "id": "doc_whitepaper_001",
            "title": "Technical Whitepaper on Distributed Systems",
            "pages": 45,
            "word_count": 12500,
        },
        "sections": [
            {"id": f"s{i}", "title": f"Section {i}", "tokens": 500 + i*100}
            for i in range(12)
        ],
        "key_entities": [
            {"name": f"Entity_{i}", "type": "concept", "mentions": 5 + i}
            for i in range(15)
        ],
    }
    results.append(test_roundtrip(ingestion, "scenario_long_context"))

    # Scenario 3: Multi-turn REPL state
    repl_state = {
        "turn": 5,
        "artifacts": [
            {"key": f"artifact_{i}", "type": "dict" if i % 2 else "list", "size": 1000 + i*500}
            for i in range(6)
        ],
        "tool_calls": [
            {"tool": "grep", "args": {"pattern": f"pattern_{i}"}, "success": True}
            for i in range(4)
        ],
        "memory_recalls": [
            {"query": f"query_{i}", "score": 0.9 - i*0.1, "used": i < 2}
            for i in range(5)
        ],
    }
    results.append(test_roundtrip(repl_state, "scenario_repl_state"))

    # Scenario 4: Worker batch results
    worker_batch = {
        "batch_id": "batch_001",
        "task_count": 8,
        "results": [
            {
                "task_id": f"task_{i}",
                "worker": f"worker_{i % 3}",
                "status": "success" if i % 4 != 0 else "failed",
                "duration_ms": 1000 + i*200,
                "output_tokens": 500 + i*100,
            }
            for i in range(8)
        ],
    }
    results.append(test_roundtrip(worker_batch, "scenario_worker_batch"))

    return results


# =============================================================================
# Live Inference Tests (Optional)
# =============================================================================

def run_live_ttft_test(port: int, data: dict, name: str) -> TestResult | None:
    """Run live TTFT comparison test against a running server."""
    try:
        import httpx
    except ImportError:
        return None

    json_str = json.dumps(data, indent=2)
    toon_str = encode(data, fallback_to_json=False) if is_available() else json_str

    prompt_template = """Analyze this data and respond with a one-word summary:

{data}

Summary:"""

    try:
        client = httpx.Client(timeout=60.0)

        # JSON timing
        json_prompt = prompt_template.format(data=json_str)
        start = time.perf_counter()
        resp = client.post(
            f"http://127.0.0.1:{port}/completion",
            json={"prompt": json_prompt, "n_predict": 1, "stream": False},
        )
        json_ttft = time.perf_counter() - start

        # TOON timing
        toon_prompt = prompt_template.format(data=toon_str)
        start = time.perf_counter()
        resp = client.post(
            f"http://127.0.0.1:{port}/completion",
            json={"prompt": toon_prompt, "n_predict": 1, "stream": False},
        )
        toon_ttft = time.perf_counter() - start

        client.close()

        improvement = (json_ttft - toon_ttft) / json_ttft * 100 if json_ttft > 0 else 0

        return TestResult(
            name=f"live_ttft_{name}",
            passed=toon_ttft < json_ttft,
            json_chars=len(json_str),
            toon_chars=len(toon_str),
            reduction_pct=(1 - len(toon_str) / len(json_str)) * 100,
            roundtrip_ok=True,
            details={
                "json_ttft_ms": json_ttft * 1000,
                "toon_ttft_ms": toon_ttft * 1000,
                "ttft_improvement_pct": improvement,
            },
        )
    except Exception as e:
        return TestResult(
            name=f"live_ttft_{name}",
            passed=False,
            json_chars=0,
            toon_chars=0,
            reduction_pct=0,
            roundtrip_ok=False,
            error=str(e),
        )


# =============================================================================
# Main Runner
# =============================================================================

def print_results(results: list[TestResult], suite_name: str) -> tuple[int, int]:
    """Print test results and return (passed, total)."""
    print(f"\n{'='*60}")
    print(f"Suite: {suite_name}")
    print(f"{'='*60}")

    passed = 0
    for r in results:
        status = "✓" if r.passed else "✗"
        if r.passed:
            passed += 1

        print(f"{status} {r.name}")
        if r.reduction_pct != 0:
            print(f"    JSON: {r.json_chars:,} chars → TOON: {r.toon_chars:,} chars ({r.reduction_pct:+.1f}%)")
        if r.details:
            for k, v in r.details.items():
                print(f"    {k}: {v}")
        if r.error:
            print(f"    ERROR: {r.error}")

    print(f"\nPassed: {passed}/{len(results)}")
    return passed, len(results)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive TOON test suite")
    parser.add_argument("--live", action="store_true", help="Run live inference tests")
    parser.add_argument("--port", type=int, default=8080, help="Server port for live tests")
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    args = parser.parse_args()

    print("TOON Comprehensive Test Suite")
    print("=" * 60)
    print(f"TOON available: {is_available()}")

    if not is_available():
        print("\nWARNING: TOON library not installed!")
        print("Install with: pip install toon-format")
        print("Or: pip install git+https://github.com/toon-format/toon-python.git")
        print("\nRunning tests anyway (will use JSON fallback)...\n")

    all_results = []
    total_passed = 0
    total_tests = 0

    # Run all suites
    suites = [
        ("File Listings", suite_file_listings),
        ("Escalation Context", suite_escalation_context),
        ("Procedures", suite_procedures),
        ("Memory Results", suite_memory_results),
        ("Grep Hits (Markdown)", suite_grep_hits),
        ("Edge Cases", suite_edge_cases),
        ("Orchestration Scenarios", suite_orchestration_scenarios),
    ]

    for name, suite_fn in suites:
        results = suite_fn()
        all_results.extend(results)
        p, t = print_results(results, name)
        total_passed += p
        total_tests += t

    # Live tests if requested
    if args.live:
        print(f"\n{'='*60}")
        print("Live TTFT Tests (port {args.port})")
        print(f"{'='*60}")

        live_tests = [
            (generate_file_listing(20), "file_listing_20"),
            (generate_escalation_context(3), "escalation_3"),
            ({"results": generate_memory_results(10)}, "memory_10"),
        ]

        for data, name in live_tests:
            result = run_live_ttft_test(args.port, data, name)
            if result:
                all_results.append(result)
                status = "✓" if result.passed else "✗"
                print(f"{status} {result.name}")
                if result.details:
                    print(f"    JSON TTFT: {result.details['json_ttft_ms']:.1f}ms")
                    print(f"    TOON TTFT: {result.details['toon_ttft_ms']:.1f}ms")
                    print(f"    Improvement: {result.details['ttft_improvement_pct']:+.1f}%")
                if result.error:
                    print(f"    ERROR: {result.error}")
                if result.passed:
                    total_passed += 1
                total_tests += 1

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total: {total_passed}/{total_tests} tests passed ({total_passed/total_tests*100:.1f}%)")

    # Calculate average reduction for successful TOON encodings
    reductions = [r.reduction_pct for r in all_results if r.passed and r.reduction_pct > 0]
    if reductions:
        avg_reduction = sum(reductions) / len(reductions)
        print(f"Average token reduction: {avg_reduction:.1f}%")

    # Identify failures
    failures = [r for r in all_results if not r.passed]
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for f in failures:
            print(f"  - {f.name}: {f.error or 'check details'}")

    # JSON output if requested
    if args.json:
        report = {
            "toon_available": is_available(),
            "total_passed": total_passed,
            "total_tests": total_tests,
            "pass_rate": total_passed / total_tests if total_tests > 0 else 0,
            "avg_reduction_pct": sum(reductions) / len(reductions) if reductions else 0,
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "json_chars": r.json_chars,
                    "toon_chars": r.toon_chars,
                    "reduction_pct": r.reduction_pct,
                    "roundtrip_ok": r.roundtrip_ok,
                    "error": r.error,
                    "details": r.details,
                }
                for r in all_results
            ],
        }
        print("\n" + json.dumps(report, indent=2))

    return 0 if total_passed == total_tests else 1


if __name__ == "__main__":
    sys.exit(main())
