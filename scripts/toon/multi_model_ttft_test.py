#!/usr/bin/env python3
"""Multi-model TTFT comparison for TOON evaluation.

Tests TOON performance benefits across different model sizes to
determine if improvements scale with model complexity.

Usage:
    python scripts/toon/multi_model_ttft_test.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.services.toon_encoder import encode, is_available


# =============================================================================
# Test Configuration
# =============================================================================

MODELS = {
    "0.5B": {"port": 8080, "description": "Qwen2.5-Coder-0.5B-Q8_0"},
    "8B": {"port": 8081, "description": "Meta-Llama-3-8B-Instruct-Q4_K_M"},
}

TRIALS = 3  # Runs per test for averaging


# =============================================================================
# Test Data - Realistic Orchestration Scenarios
# =============================================================================

def scenario_frontdoor_routing() -> dict:
    """Data sent to frontdoor for routing decision."""
    return {
        "task_type": "code_review",
        "user_query": "Review this PR for security issues",
        "context": {
            "repo": "pestopoppa/llama.cpp",
            "pr_number": 18239,
            "files_changed": [
                {"path": f"src/module_{i}.cpp", "additions": 50+i*10, "deletions": 20+i*5}
                for i in range(12)
            ],
            "labels": ["security", "critical", "needs-review"],
        },
        "memory_recalls": [
            {"query": f"security review pattern {i}", "score": 0.9 - i*0.05, "action": f"pattern_{i}"}
            for i in range(5)
        ],
    }


def scenario_coder_escalation() -> dict:
    """Data sent to coder on escalation from worker."""
    return {
        "task_id": "task_code_fix_001",
        "escalation_level": 2,
        "from_tier": "C",
        "to_tier": "B1",
        "reason": "Worker failed schema validation twice",
        "previous_attempts": [
            {
                "attempt": i+1,
                "role": "worker_code",
                "error_type": "SCHEMA_VALIDATION",
                "error_message": f"Missing required field: {'output' if i == 0 else 'metadata'}",
                "tokens_used": 800 + i*200,
                "duration_ms": 1500 + i*500,
                "partial_output": f"def process(data):\n    # Attempt {i+1}...\n",
            }
            for i in range(3)
        ],
        "repl_state": {
            "artifacts": [
                {"key": f"artifact_{i}", "type": "code", "lines": 50+i*20}
                for i in range(4)
            ],
            "tool_history": [
                {"tool": "read_file", "path": f"src/mod_{i}.py", "success": True}
                for i in range(3)
            ],
        },
    }


def scenario_architect_complex() -> dict:
    """Data sent to architect (235B/480B) for complex decisions."""
    return {
        "task_id": "task_architecture_001",
        "escalation_level": 3,
        "complexity_score": 0.92,
        "decision_type": "architectural_refactor",
        "context": {
            "codebase_summary": {
                "total_files": 1234,
                "total_lines": 456789,
                "languages": ["Python", "C++", "TypeScript"],
                "modules": [
                    {"name": f"module_{i}", "files": 20+i*5, "complexity": 0.5+i*0.1}
                    for i in range(15)
                ],
            },
            "dependencies": [
                {"name": f"dep_{i}", "version": f"1.{i}.0", "critical": i < 3}
                for i in range(10)
            ],
            "test_coverage": {
                "overall": 78.5,
                "by_module": [
                    {"module": f"module_{i}", "coverage": 60+i*3}
                    for i in range(15)
                ],
            },
        },
        "previous_decisions": [
            {
                "decision_id": f"dec_{i}",
                "type": "refactor",
                "outcome": "success" if i % 3 != 0 else "partial",
                "impact_score": 0.8 - i*0.05,
            }
            for i in range(8)
        ],
        "constraints": [
            {"type": "performance", "requirement": "< 100ms p99 latency"},
            {"type": "memory", "requirement": "< 1GB peak usage"},
            {"type": "compatibility", "requirement": "Python 3.10+"},
        ],
    }


def scenario_long_context_ingest() -> dict:
    """Data sent to ingest model (80B) for document processing."""
    return {
        "document_id": "doc_whitepaper_001",
        "document_type": "technical_whitepaper",
        "metadata": {
            "title": "Efficient Inference on AMD EPYC Processors",
            "authors": ["Author A", "Author B", "Author C"],
            "pages": 45,
            "word_count": 12500,
            "language": "en",
        },
        "sections": [
            {
                "id": f"s{i}",
                "title": f"Section {i}: {'Introduction' if i == 0 else 'Analysis' if i < 5 else 'Results' if i < 10 else 'Conclusion'}",
                "start_page": i*4 + 1,
                "end_page": (i+1)*4,
                "tokens_estimate": 500 + i*100,
                "key_terms": [f"term_{i}_{j}" for j in range(3)],
            }
            for i in range(12)
        ],
        "entities": [
            {
                "name": f"Entity_{i}",
                "type": ["concept", "method", "framework", "tool"][i % 4],
                "mentions": 5 + i,
                "first_section": f"s{i % 12}",
            }
            for i in range(20)
        ],
        "references": [
            {"id": f"ref_{i}", "title": f"Reference Paper {i}", "year": 2020 + i % 6}
            for i in range(15)
        ],
    }


def scenario_worker_batch() -> dict:
    """Data representing batch results from parallel workers."""
    return {
        "batch_id": "batch_parallel_001",
        "task_type": "file_processing",
        "total_tasks": 16,
        "completed": 14,
        "failed": 2,
        "results": [
            {
                "task_id": f"subtask_{i:03d}",
                "worker_id": f"worker_{i % 4}",
                "status": "success" if i not in [5, 12] else "failed",
                "duration_ms": 1000 + i*150,
                "output_tokens": 400 + i*50,
                "file_path": f"src/components/module_{i}.tsx",
                "changes": {"additions": 20+i*5, "deletions": 10+i*2},
            }
            for i in range(16)
        ],
        "aggregated_metrics": {
            "total_duration_ms": 25000,
            "total_tokens": 12000,
            "avg_latency_ms": 1562,
            "success_rate": 0.875,
        },
    }


SCENARIOS = [
    ("frontdoor_routing", scenario_frontdoor_routing),
    ("coder_escalation", scenario_coder_escalation),
    ("architect_complex", scenario_architect_complex),
    ("long_context_ingest", scenario_long_context_ingest),
    ("worker_batch", scenario_worker_batch),
]


# =============================================================================
# Test Runner
# =============================================================================

@dataclass
class TTFTResult:
    scenario: str
    model: str
    json_chars: int
    toon_chars: int
    reduction_pct: float
    json_ttft_ms: float
    toon_ttft_ms: float
    ttft_improvement_pct: float
    trials: int


def measure_ttft(port: int, prompt: str, trials: int = 3) -> float:
    """Measure time-to-first-token averaged over trials."""
    client = httpx.Client(timeout=120.0)
    times = []

    for _ in range(trials):
        start = time.perf_counter()
        try:
            resp = client.post(
                f"http://127.0.0.1:{port}/completion",
                json={"prompt": prompt, "n_predict": 1, "stream": False},
            )
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # Convert to ms
        except Exception as e:
            print(f"    Warning: Request failed: {e}")
            continue

    client.close()
    return sum(times) / len(times) if times else 0


def run_scenario(scenario_name: str, data: dict, model_name: str, port: int) -> TTFTResult | None:
    """Run a single scenario test against a model."""
    json_str = json.dumps(data, indent=2)
    toon_str = encode(data, fallback_to_json=False) if is_available() else json_str

    # Build prompts
    prompt_template = """You are an AI assistant analyzing structured data. Respond with "UNDERSTOOD" only.

DATA:
{data}

Response:"""

    json_prompt = prompt_template.format(data=json_str)
    toon_prompt = prompt_template.format(data=toon_str)

    # Measure TTFT
    print(f"    Testing {scenario_name} on {model_name}...")
    json_ttft = measure_ttft(port, json_prompt, TRIALS)
    toon_ttft = measure_ttft(port, toon_prompt, TRIALS)

    if json_ttft == 0 or toon_ttft == 0:
        return None

    reduction = (1 - len(toon_str) / len(json_str)) * 100
    improvement = (json_ttft - toon_ttft) / json_ttft * 100

    return TTFTResult(
        scenario=scenario_name,
        model=model_name,
        json_chars=len(json_str),
        toon_chars=len(toon_str),
        reduction_pct=reduction,
        json_ttft_ms=json_ttft,
        toon_ttft_ms=toon_ttft,
        ttft_improvement_pct=improvement,
        trials=TRIALS,
    )


def check_server(port: int) -> bool:
    """Check if server is healthy."""
    try:
        resp = httpx.get(f"http://127.0.0.1:{port}/health", timeout=5.0)
        return resp.status_code == 200
    except Exception as e:
        return False


def main():
    print("=" * 70)
    print("TOON Multi-Model TTFT Comparison")
    print("=" * 70)
    print(f"TOON available: {is_available()}")
    print(f"Trials per test: {TRIALS}")
    print()

    # Check available models
    available_models = {}
    for name, config in MODELS.items():
        if check_server(config["port"]):
            available_models[name] = config
            print(f"✓ {name}: {config['description']} (port {config['port']})")
        else:
            print(f"✗ {name}: Not available (port {config['port']})")

    if not available_models:
        print("\nNo models available. Start servers first.")
        return 1

    print()

    # Run all scenarios on all available models
    results: list[TTFTResult] = []

    for model_name, config in available_models.items():
        print(f"\n{'='*70}")
        print(f"Model: {model_name} ({config['description']})")
        print(f"{'='*70}")

        for scenario_name, scenario_fn in SCENARIOS:
            data = scenario_fn()
            result = run_scenario(scenario_name, data, model_name, config["port"])
            if result:
                results.append(result)
                status = "✓" if result.ttft_improvement_pct > 0 else "✗"
                print(f"    {status} {scenario_name}:")
                print(f"        Chars: {result.json_chars:,} → {result.toon_chars:,} ({result.reduction_pct:+.1f}%)")
                print(f"        TTFT:  {result.json_ttft_ms:.1f}ms → {result.toon_ttft_ms:.1f}ms ({result.ttft_improvement_pct:+.1f}%)")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    # By model
    for model_name in available_models:
        model_results = [r for r in results if r.model == model_name]
        if model_results:
            avg_reduction = sum(r.reduction_pct for r in model_results) / len(model_results)
            avg_improvement = sum(r.ttft_improvement_pct for r in model_results) / len(model_results)
            print(f"\n{model_name}:")
            print(f"  Avg token reduction: {avg_reduction:.1f}%")
            print(f"  Avg TTFT improvement: {avg_improvement:.1f}%")

    # Overall
    if results:
        overall_reduction = sum(r.reduction_pct for r in results) / len(results)
        overall_improvement = sum(r.ttft_improvement_pct for r in results) / len(results)
        positive_improvements = sum(1 for r in results if r.ttft_improvement_pct > 0)

        print(f"\nOverall ({len(results)} tests):")
        print(f"  Avg token reduction: {overall_reduction:.1f}%")
        print(f"  Avg TTFT improvement: {overall_improvement:.1f}%")
        print(f"  Tests with positive improvement: {positive_improvements}/{len(results)}")

    # Detailed table
    print(f"\n{'='*70}")
    print("DETAILED RESULTS")
    print(f"{'='*70}")
    print(f"{'Scenario':<25} {'Model':<8} {'Reduction':<12} {'TTFT Improv':<12}")
    print("-" * 70)
    for r in results:
        print(f"{r.scenario:<25} {r.model:<8} {r.reduction_pct:>+8.1f}%    {r.ttft_improvement_pct:>+8.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
