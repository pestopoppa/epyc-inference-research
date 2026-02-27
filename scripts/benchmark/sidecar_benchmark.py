#!/usr/bin/env python3
"""Phase 2B-Sidecar benchmark: compare sidecar speculation vs Phase 2A baselines.

Sends the same 6 code generation prompts to a sidecar-enabled llama-server
and records speed/acceptance metrics. Compares against Phase 2A baseline
numbers already recorded in the progress report.

Usage:
    python scripts/benchmark/sidecar_benchmark.py --port 9081 --model 30b
    python scripts/benchmark/sidecar_benchmark.py --port 9082 --model 32b
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROMPTS = [
    {"id": "async_retry", "prompt": "Write a Python async HTTP client with retry logic, exponential backoff, and circuit breaker pattern. Include type hints and a usage example."},
    {"id": "bst_iterator", "prompt": "Implement a binary search tree in Python with an in-order iterator that uses O(h) memory where h is the height of the tree. Include insert, search, delete, and the iterator protocol (__iter__, __next__)."},
    {"id": "lru_cache", "prompt": "Write a thread-safe LRU cache in Python using a doubly-linked list and a dictionary. Support get, put, and resize operations. Include proper locking and a decorator version."},
    {"id": "json_parser", "prompt": "Write a recursive descent JSON parser in Python from scratch (no json module). Handle strings (with escapes), numbers (int and float), booleans, null, arrays, and objects. Return native Python types."},
    {"id": "rate_limiter", "prompt": "Implement a token bucket rate limiter in Python that supports per-key limits, burst capacity, and automatic refill. Make it work both synchronously and with asyncio."},
    {"id": "graph_shortest", "prompt": "Write Dijkstra's algorithm and A* search in Python. Support weighted directed graphs with an adjacency list representation. Include a priority queue implementation and path reconstruction."},
]

# Phase 2A baseline numbers (from progress/2026-02/2026-02-19.md)
PHASE2A_BASELINES = {
    "30b": {
        "no_corpus": {"async_retry": 25.8, "bst_iterator": 26.4, "lru_cache": 22.8, "json_parser": 26.7, "rate_limiter": 27.5, "graph_shortest": 25.5},
        "with_corpus": {"async_retry": 27.1, "bst_iterator": 30.4, "lru_cache": 28.7, "json_parser": 33.4, "rate_limiter": 27.8, "graph_shortest": 31.8},
    },
    "32b": {
        "no_corpus": {"async_retry": 10.4, "bst_iterator": 14.5, "lru_cache": 11.2, "json_parser": 13.8, "rate_limiter": 11.6, "graph_shortest": 13.3},
        "with_corpus": {"async_retry": 16.3, "bst_iterator": 14.5, "lru_cache": 15.5, "json_parser": 20.2, "rate_limiter": 13.2, "graph_shortest": 50.3},
    },
}


def warmup(port: int) -> None:
    log.info("Warming up port %d...", port)
    try:
        generate(port, "Say hello.", max_tokens=5)
        log.info("Warmup done.")
    except Exception as e:
        log.warning("Warmup failed: %s", e)


def generate(port: int, prompt: str, max_tokens: int = 1024) -> dict:
    url = f"http://localhost:{port}/v1/chat/completions"
    payload = {
        "model": "test",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    t0 = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=600)
    wall = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()

    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    timings = data.get("timings", {})

    return {
        "output": content,
        "tokens": usage.get("completion_tokens", len(content.split())),
        "speed": timings.get("predicted_per_second", 0),
        "draft_n": timings.get("draft_n", 0),
        "draft_accepted": timings.get("draft_n_accepted", 0),
        "wall_time": wall,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 2B sidecar benchmark")
    parser.add_argument("--port", type=int, required=True, help="Port of sidecar-enabled server")
    parser.add_argument("--model", choices=["30b", "32b"], required=True)
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    results_dir = Path("/mnt/raid0/llm/epyc-inference-research/benchmarks/results/runs")
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = results_dir / f"sidecar_{args.model}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    warmup(args.port)

    results = []
    baselines = PHASE2A_BASELINES.get(args.model, {})

    log.info("=== Phase 2B-Sidecar Benchmark: %s on port %d ===", args.model, args.port)

    for p in PROMPTS:
        log.info("[%s] %s — generating...", args.model, p["id"])
        r = generate(args.port, p["prompt"])

        accept_rate = (r["draft_accepted"] / r["draft_n"] * 100) if r["draft_n"] > 0 else 0

        entry = {
            "prompt_id": p["id"],
            "model": args.model,
            "speed_tps": r["speed"],
            "tokens": r["tokens"],
            "draft_n": r["draft_n"],
            "draft_accepted": r["draft_accepted"],
            "acceptance_rate": round(accept_rate, 1),
            "wall_time": round(r["wall_time"], 2),
        }
        results.append(entry)

        # Compare to Phase 2A baselines
        no_corpus = baselines.get("no_corpus", {}).get(p["id"], 0)
        with_corpus = baselines.get("with_corpus", {}).get(p["id"], 0)

        log.info(
            "  %s: sidecar=%.1f t/s, accept=%.1f%% | Phase2A baseline=%.1f, corpus=%.1f t/s",
            p["id"], r["speed"], accept_rate, no_corpus, with_corpus,
        )

        # Write incrementally
        out_file = out_dir / "results.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)

    # Summary table
    log.info("")
    log.info("=== RESULTS SUMMARY ===")
    log.info("%-20s %10s %10s %10s %10s %10s", "Prompt", "Sidecar", "Baseline", "Phase2A", "vs Base", "vs 2A")
    log.info("-" * 80)

    sidecar_speeds = []
    baseline_speeds = []
    phase2a_speeds = []

    for entry in results:
        pid = entry["prompt_id"]
        sidecar = entry["speed_tps"]
        no_corpus = baselines.get("no_corpus", {}).get(pid, 0)
        with_corpus = baselines.get("with_corpus", {}).get(pid, 0)

        vs_base = ((sidecar - no_corpus) / no_corpus * 100) if no_corpus > 0 else 0
        vs_2a = ((sidecar - with_corpus) / with_corpus * 100) if with_corpus > 0 else 0

        log.info("%-20s %8.1f %8.1f %8.1f %+8.0f%% %+8.0f%%", pid, sidecar, no_corpus, with_corpus, vs_base, vs_2a)

        sidecar_speeds.append(sidecar)
        baseline_speeds.append(no_corpus)
        phase2a_speeds.append(with_corpus)

    avg_sidecar = sum(sidecar_speeds) / len(sidecar_speeds) if sidecar_speeds else 0
    avg_base = sum(baseline_speeds) / len(baseline_speeds) if baseline_speeds else 0
    avg_2a = sum(phase2a_speeds) / len(phase2a_speeds) if phase2a_speeds else 0
    vs_base_avg = ((avg_sidecar - avg_base) / avg_base * 100) if avg_base > 0 else 0
    vs_2a_avg = ((avg_sidecar - avg_2a) / avg_2a * 100) if avg_2a > 0 else 0

    log.info("-" * 80)
    log.info("%-20s %8.1f %8.1f %8.1f %+8.0f%% %+8.0f%%", "AVERAGE", avg_sidecar, avg_base, avg_2a, vs_base_avg, vs_2a_avg)

    log.info("")
    log.info("Results saved to: %s", out_dir / "results.json")


if __name__ == "__main__":
    main()
