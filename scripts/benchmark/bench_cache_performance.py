#!/usr/bin/env python3
from __future__ import annotations

"""Benchmark cache hit rates on RLM (Recursive LLM) workloads.

This script measures the effectiveness of RadixAttention-style prefix caching
for orchestrator workloads where prompts share common system prefixes.

Usage:
    # Basic benchmark
    python scripts/benchmark/bench_cache_performance.py

    # With custom server URL
    python scripts/benchmark/bench_cache_performance.py --server http://localhost:8082

    # With verbose output
    python scripts/benchmark/bench_cache_performance.py -v

    # Dry run (mock mode)
    python scripts/benchmark/bench_cache_performance.py --dry-run

Requirements:
    - llama-server running with cache_prompt support
    - RadixAttention infrastructure (src/prefix_cache.py)
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class BenchmarkConfig:
    """Configuration for cache benchmark."""

    server_url: str = "http://localhost:8082"
    num_slots: int = 4
    n_tokens: int = 50
    num_queries: int = 20
    num_rounds: int = 3
    verbose: bool = False
    dry_run: bool = False


@dataclass
class BenchmarkResult:
    """Results from cache benchmark."""

    config: BenchmarkConfig
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    hit_rate: float = 0.0
    avg_latency_ms: float = 0.0
    total_time_s: float = 0.0
    tokens_generated: int = 0
    avg_speed_tps: float = 0.0
    errors: list = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "server_url": self.config.server_url,
            "num_slots": self.config.num_slots,
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_pct": self.hit_rate * 100,
            "avg_latency_ms": self.avg_latency_ms,
            "total_time_s": self.total_time_s,
            "tokens_generated": self.tokens_generated,
            "avg_speed_tps": self.avg_speed_tps,
            "error_count": len(self.errors),
        }


# RLM-style system prompts (shared prefix patterns)
SYSTEM_PROMPTS = {
    "code_assistant": """You are a code assistant for a large software project.
You help with code generation, refactoring, debugging, and documentation.
Follow these guidelines:
1. Write clean, maintainable code
2. Include appropriate error handling
3. Add type hints where applicable
4. Keep functions focused and small
5. Use descriptive variable names

Current context: Working on the orchestrator module.
""",
    "document_analyzer": """You are a document analysis assistant.
You help with summarizing, extracting key points, and answering questions.
Follow these guidelines:
1. Be concise but comprehensive
2. Cite specific sections when relevant
3. Identify key themes and patterns
4. Note any inconsistencies or gaps
5. Provide actionable insights

Document type: Technical documentation.
""",
    "math_helper": """You are a mathematical reasoning assistant.
You help with proofs, calculations, and formal verification.
Follow these guidelines:
1. Show your work step by step
2. Verify intermediate results
3. State assumptions clearly
4. Consider edge cases
5. Provide final answer clearly

Domain: Optimization problems.
""",
}

# User queries to append to system prompts
USER_QUERIES = [
    "Write a function to sort a list",
    "Explain this error message",
    "Refactor this code to be cleaner",
    "Add error handling to this function",
    "Write unit tests for this module",
    "Summarize the main points",
    "What are the key takeaways?",
    "Find any inconsistencies",
    "Extract the action items",
    "Compare sections A and B",
    "Prove this statement",
    "Calculate the result",
    "Verify this equation",
    "Find the optimal solution",
    "Check for edge cases",
    "Document this function",
    "Explain the architecture",
    "Identify potential bugs",
    "Suggest improvements",
    "Review for security issues",
]


def run_mock_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Run benchmark in mock mode (no server required)."""
    from src.prefix_cache import PrefixRouter, canonicalize_prompt

    print("Running in mock mode (dry run)...")

    router = PrefixRouter(num_slots=config.num_slots)
    result = BenchmarkResult(config=config)

    start_time = time.time()

    # Run multiple rounds
    for round_num in range(config.num_rounds):
        if config.verbose:
            print(f"\n--- Round {round_num + 1}/{config.num_rounds} ---")

        for sys_name, sys_prompt in SYSTEM_PROMPTS.items():
            for query in USER_QUERIES[: config.num_queries]:
                prompt = sys_prompt + "\n\nUser: " + query
                prompt = canonicalize_prompt(prompt)

                # Route the prompt
                slot = router.get_slot_for_prompt(prompt)
                result.total_requests += 1
                result.tokens_generated += config.n_tokens

                if config.verbose:
                    print(f"  {sys_name}: slot={slot}")

    result.total_time_s = time.time() - start_time
    result.cache_hits = router.cache_hits
    result.cache_misses = router.cache_misses
    result.hit_rate = router.cache_hits / result.total_requests if result.total_requests > 0 else 0
    result.avg_latency_ms = (result.total_time_s * 1000) / result.total_requests if result.total_requests > 0 else 0
    result.avg_speed_tps = result.tokens_generated / result.total_time_s if result.total_time_s > 0 else 0

    return result


def run_live_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Run benchmark against live llama-server."""
    from src.backends.llama_server import LlamaServerBackend, ServerConfig
    from src.prefix_cache import CachingBackend, PrefixRouter, canonicalize_prompt
    from src.model_server import InferenceRequest
    from src.registry_loader import (
        RoleConfig,
        AccelerationConfig,
        ModelConfig,
        PerformanceMetrics,
        MemoryConfig,
    )

    print(f"Running against live server: {config.server_url}")

    # Setup backend
    server_config = ServerConfig(base_url=config.server_url, num_slots=config.num_slots)
    backend = LlamaServerBackend(server_config)
    router = PrefixRouter(num_slots=config.num_slots)
    caching = CachingBackend(backend, router)

    # Check server health
    if not backend.health_check(0):
        print(f"ERROR: Server not healthy at {config.server_url}")
        return BenchmarkResult(config=config, errors=["Server not healthy"])

    # Create role config with proper nested objects
    role_config = RoleConfig(
        name="benchmark_worker",
        tier="C",
        description="Cache benchmark worker",
        model=ModelConfig(
            name="benchmark-model",
            path="",
            quant="Q8_0",
            size_gb=0.5,
        ),
        acceleration=AccelerationConfig(type="baseline", temperature=0.0),
        performance=PerformanceMetrics(),
        memory=MemoryConfig(residency="warm"),
    )

    result = BenchmarkResult(config=config)
    latencies = []

    start_time = time.time()

    # Run multiple rounds
    for round_num in range(config.num_rounds):
        print(f"\n--- Round {round_num + 1}/{config.num_rounds} ---")

        for sys_name, sys_prompt in SYSTEM_PROMPTS.items():
            print(f"  System: {sys_name}")

            for i, query in enumerate(USER_QUERIES[: config.num_queries]):
                prompt = sys_prompt + "\n\nUser: " + query
                prompt = canonicalize_prompt(prompt)

                request = InferenceRequest(
                    role="worker",
                    prompt=prompt,
                    n_tokens=config.n_tokens,
                    temperature=0.0,
                )

                try:
                    req_start = time.time()
                    response = caching.infer(role_config, request)
                    req_elapsed = (time.time() - req_start) * 1000  # ms

                    if response.success:
                        latencies.append(req_elapsed)
                        result.total_requests += 1
                        result.tokens_generated += response.tokens_generated

                        if config.verbose:
                            print(f"    Query {i+1}: {req_elapsed:.0f}ms, {response.tokens_generated} tokens")
                    else:
                        result.errors.append(response.error_message)
                        print(f"    Query {i+1}: ERROR - {response.error_message}")

                except Exception as e:
                    result.errors.append(str(e))
                    print(f"    Query {i+1}: EXCEPTION - {e}")

    result.total_time_s = time.time() - start_time

    # Get cache stats
    stats = caching.get_stats()
    result.cache_hits = stats.get("router_cache_hits", 0)
    result.cache_misses = stats.get("router_cache_misses", 0)
    result.hit_rate = stats.get("router_hit_rate", 0)

    if latencies:
        result.avg_latency_ms = sum(latencies) / len(latencies)
        result.avg_speed_tps = result.tokens_generated / result.total_time_s if result.total_time_s > 0 else 0

    return result


def print_results(result: BenchmarkResult) -> None:
    """Print benchmark results."""
    print("\n" + "=" * 60)
    print("CACHE PERFORMANCE BENCHMARK RESULTS")
    print("=" * 60)

    data = result.to_dict()

    print(f"\nServer: {data['server_url']}")
    print(f"Slots:  {data['num_slots']}")
    print()

    print("Cache Performance:")
    print(f"  Total Requests: {data['total_requests']}")
    print(f"  Cache Hits:     {data['cache_hits']}")
    print(f"  Cache Misses:   {data['cache_misses']}")
    print(f"  Hit Rate:       {data['hit_rate_pct']:.1f}%")
    print()

    print("Latency & Throughput:")
    print(f"  Avg Latency:    {data['avg_latency_ms']:.1f} ms")
    print(f"  Total Time:     {data['total_time_s']:.1f} s")
    print(f"  Tokens Generated: {data['tokens_generated']}")
    print(f"  Avg Speed:      {data['avg_speed_tps']:.1f} t/s")
    print()

    if data['error_count'] > 0:
        print(f"Errors: {data['error_count']}")
        for err in result.errors[:5]:
            print(f"  - {err}")
    print()

    # Success criteria check
    print("Success Criteria:")
    hit_target = 50.0
    if data['hit_rate_pct'] >= hit_target:
        print(f"  [PASS] Hit rate >= {hit_target}%: {data['hit_rate_pct']:.1f}%")
    else:
        print(f"  [FAIL] Hit rate >= {hit_target}%: {data['hit_rate_pct']:.1f}%")

    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Benchmark cache performance on RLM workloads")
    parser.add_argument("--server", default="http://localhost:8082", help="Server URL")
    parser.add_argument("--slots", type=int, default=4, help="Number of slots")
    parser.add_argument("--tokens", type=int, default=50, help="Tokens per request")
    parser.add_argument("--queries", type=int, default=20, help="Queries per system prompt")
    parser.add_argument("--rounds", type=int, default=3, help="Number of rounds")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--dry-run", action="store_true", help="Run in mock mode")
    parser.add_argument("--output", type=str, help="Output JSON file")

    args = parser.parse_args()

    config = BenchmarkConfig(
        server_url=args.server,
        num_slots=args.slots,
        n_tokens=args.tokens,
        num_queries=args.queries,
        num_rounds=args.rounds,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )

    print("Cache Performance Benchmark")
    print(f"  Server: {config.server_url}")
    print(f"  Slots: {config.num_slots}")
    print(f"  Queries per system: {config.num_queries}")
    print(f"  Rounds: {config.num_rounds}")
    print()

    # Run benchmark
    if config.dry_run:
        result = run_mock_benchmark(config)
    else:
        result = run_live_benchmark(config)

    # Print results
    print_results(result)

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"Results saved to: {output_path}")

    # Return exit code based on hit rate target
    return 0 if result.hit_rate >= 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())
