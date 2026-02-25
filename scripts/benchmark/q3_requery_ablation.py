#!/usr/bin/env python3
"""Q3: First-20-token re-query vs keyword-only retrieval ablation.

Compares two corpus query strategies:
  A) Keyword extraction from NL prompt (current production approach)
  B) First ~20 tokens of actual model output (simulated re-query)

Measures: n-gram overlap between retrieved snippets and full model output.
Higher overlap = more n-gram material for prompt lookup = faster spec decode.

Uses existing quality gate outputs (no inference needed) and V3 sharded corpus.

Usage:
    python3 scripts/benchmark/q3_requery_ablation.py
"""

import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.services.corpus_retrieval import (
    CorpusConfig,
    CorpusRetriever,
    extract_code_query,
)

# Quality gate prompts (same as corpus_quality_gate.py)
PROMPTS = [
    {
        "id": "async_retry",
        "prompt": "Write a Python async HTTP client with retry logic, exponential backoff, and circuit breaker pattern. Include type hints and a usage example.",
    },
    {
        "id": "bst_iterator",
        "prompt": "Implement a binary search tree in Python with an in-order iterator that uses O(h) memory where h is the height of the tree. Include insert, search, delete, and the iterator protocol (__iter__, __next__).",
    },
    {
        "id": "lru_cache",
        "prompt": "Write a thread-safe LRU cache in Python using a doubly-linked list and a dictionary. Support get, put, and resize operations. Include proper locking and a decorator version.",
    },
    {
        "id": "json_parser",
        "prompt": "Write a recursive descent JSON parser in Python from scratch (no json module). Handle strings (with escapes), numbers (int and float), booleans, null, arrays, and objects. Return native Python types.",
    },
    {
        "id": "rate_limiter",
        "prompt": "Implement a token bucket rate limiter in Python that supports per-key limits, burst capacity, and automatic refill. Make it work both synchronously and with asyncio.",
    },
    {
        "id": "graph_shortest",
        "prompt": "Write Dijkstra's algorithm and A* search in Python. Support weighted directed graphs with an adjacency list representation. Include a priority queue implementation and path reconstruction.",
    },
]

OUTPUT_DIR = "/mnt/raid0/llm/claude/benchmarks/results/runs/q3_requery"
QUALITY_GATE_RESULTS = "/mnt/raid0/llm/tmp/corpus_quality_gate.json"
V3_INDEX = "/mnt/raid0/llm/cache/corpus/v3_sharded"


def extract_first_n_tokens(text: str, n: int = 20) -> str:
    """Extract first N whitespace-delimited tokens from model output.

    Skips common preamble ("Certainly!", "Here's", etc.) to get to actual code.
    """
    lines = text.strip().split("\n")
    code_tokens = []
    in_code = False

    for line in lines:
        stripped = line.strip()
        # Skip markdown and preamble
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if not in_code and not stripped:
            continue
        if not in_code and any(stripped.startswith(w) for w in [
            "Certainly", "Here", "Below", "Sure", "This", "The following",
            "I'll", "Let me", "###", "##", "#", "**",
        ]):
            continue

        # Collect tokens from code or substantive text
        tokens = stripped.split()
        code_tokens.extend(tokens)
        if len(code_tokens) >= n:
            break

    return " ".join(code_tokens[:n])


def compute_ngram_overlap(snippets_text: str, output_text: str, n: int = 4) -> dict:
    """Compute n-gram overlap between retrieved snippets and model output.

    Returns overlap metrics that predict prompt lookup acceleration.
    """
    def extract_ngrams(text: str, n: int) -> set:
        words = text.lower().split()
        if len(words) < n:
            return set()
        return {tuple(words[i:i+n]) for i in range(len(words) - n + 1)}

    snippet_grams = extract_ngrams(snippets_text, n)
    output_grams = extract_ngrams(output_text, n)

    if not output_grams:
        return {"overlap_count": 0, "overlap_pct": 0.0, "snippet_grams": len(snippet_grams), "output_grams": 0}

    overlap = snippet_grams & output_grams
    return {
        "overlap_count": len(overlap),
        "overlap_pct": round(100.0 * len(overlap) / len(output_grams), 2),
        "snippet_grams": len(snippet_grams),
        "output_grams": len(output_grams),
    }


def main():
    print("Q3: First-20-token Re-query vs Keyword-only Retrieval")
    print("=" * 70)

    # Load existing quality gate outputs
    if not os.path.exists(QUALITY_GATE_RESULTS):
        print(f"ERROR: Quality gate results not found at {QUALITY_GATE_RESULTS}")
        sys.exit(1)

    with open(QUALITY_GATE_RESULTS) as f:
        gate_data = json.load(f)

    # Use 32b outputs (best quality, highest acceptance rates)
    model_key = "32b"
    if model_key not in gate_data:
        model_key = list(gate_data.keys())[0]
    model_outputs = {r["prompt_id"]: r for r in gate_data[model_key]}
    print(f"\nUsing {model_key} model outputs from quality gate ({len(model_outputs)} prompts)")

    # Init corpus retriever with V3 index
    config = CorpusConfig(
        enabled=True,
        index_path=V3_INDEX,
        max_snippets=3,
        max_chars=3000,
        min_score=0.3,  # Lower threshold to see more results
    )
    CorpusRetriever.reset_instance()
    retriever = CorpusRetriever.get_instance(config)

    results = []

    for prompt_info in PROMPTS:
        pid = prompt_info["id"]
        nl_prompt = prompt_info["prompt"]

        output_data = model_outputs.get(pid)
        if not output_data:
            print(f"\n  SKIP {pid}: no quality gate output")
            continue

        # Get the baseline (no-corpus) output — this is what the model naturally generates
        full_output = output_data["baseline"]["output"]

        print(f"\n{'─' * 70}")
        print(f"  Prompt: {pid}")
        print(f"  NL: {nl_prompt[:80]}...")

        # ── Strategy A: Keyword extraction from NL prompt (current production) ──
        keyword_query = extract_code_query(nl_prompt)
        print(f"\n  [A] Keyword query: \"{keyword_query}\"")

        t0 = time.perf_counter()
        keyword_snippets = retriever.retrieve(keyword_query)
        keyword_ms = (time.perf_counter() - t0) * 1000

        keyword_text = "\n".join(s.code for s in keyword_snippets)
        keyword_overlap = compute_ngram_overlap(keyword_text, full_output)

        print(f"      Snippets: {len(keyword_snippets)}, latency: {keyword_ms:.1f}ms")
        print(f"      Scores: {[round(s.score, 3) for s in keyword_snippets]}")
        print(f"      4-gram overlap with output: {keyword_overlap['overlap_count']} "
              f"({keyword_overlap['overlap_pct']}% of output grams)")

        # ── Strategy B: First-20-token re-query ──
        first_tokens = extract_first_n_tokens(full_output, n=20)
        print(f"\n  [B] First-20-token query: \"{first_tokens[:80]}...\"")

        t0 = time.perf_counter()
        token_snippets = retriever.retrieve(first_tokens)
        token_ms = (time.perf_counter() - t0) * 1000

        token_text = "\n".join(s.code for s in token_snippets)
        token_overlap = compute_ngram_overlap(token_text, full_output)

        print(f"      Snippets: {len(token_snippets)}, latency: {token_ms:.1f}ms")
        print(f"      Scores: {[round(s.score, 3) for s in token_snippets]}")
        print(f"      4-gram overlap with output: {token_overlap['overlap_count']} "
              f"({token_overlap['overlap_pct']}% of output grams)")

        # ── Strategy C: Combined (keyword + first tokens) ──
        combined_query = f"{keyword_query} {first_tokens}"
        print(f"\n  [C] Combined query: \"{combined_query[:80]}...\"")

        t0 = time.perf_counter()
        combined_snippets = retriever.retrieve(combined_query)
        combined_ms = (time.perf_counter() - t0) * 1000

        combined_text = "\n".join(s.code for s in combined_snippets)
        combined_overlap = compute_ngram_overlap(combined_text, full_output)

        print(f"      Snippets: {len(combined_snippets)}, latency: {combined_ms:.1f}ms")
        print(f"      Scores: {[round(s.score, 3) for s in combined_snippets]}")
        print(f"      4-gram overlap with output: {combined_overlap['overlap_count']} "
              f"({combined_overlap['overlap_pct']}% of output grams)")

        # ── Comparison ──
        best = max(
            [("keyword", keyword_overlap), ("first_tokens", token_overlap), ("combined", combined_overlap)],
            key=lambda x: x[1]["overlap_count"],
        )
        print(f"\n  WINNER: {best[0]} ({best[1]['overlap_count']} overlapping 4-grams)")

        result = {
            "prompt_id": pid,
            "keyword": {
                "query": keyword_query,
                "n_snippets": len(keyword_snippets),
                "scores": [round(s.score, 3) for s in keyword_snippets],
                "latency_ms": round(keyword_ms, 2),
                **keyword_overlap,
            },
            "first_tokens": {
                "query": first_tokens[:200],
                "n_snippets": len(token_snippets),
                "scores": [round(s.score, 3) for s in token_snippets],
                "latency_ms": round(token_ms, 2),
                **token_overlap,
            },
            "combined": {
                "query": combined_query[:200],
                "n_snippets": len(combined_snippets),
                "scores": [round(s.score, 3) for s in combined_snippets],
                "latency_ms": round(combined_ms, 2),
                **combined_overlap,
            },
            "winner": best[0],
        }
        results.append(result)

    # ── Summary Table ──
    print(f"\n{'=' * 70}")
    print("  SUMMARY: 4-gram overlap with model output (higher = better for spec decode)")
    print(f"{'=' * 70}")
    print(f"  {'Prompt':<20} {'Keyword':>10} {'1st-20tok':>10} {'Combined':>10} {'Winner':>12}")
    print(f"  {'-' * 62}")

    keyword_total = 0
    token_total = 0
    combined_total = 0

    for r in results:
        kw = r["keyword"]["overlap_count"]
        ft = r["first_tokens"]["overlap_count"]
        cb = r["combined"]["overlap_count"]
        keyword_total += kw
        token_total += ft
        combined_total += cb
        print(f"  {r['prompt_id']:<20} {kw:>10} {ft:>10} {cb:>10} {r['winner']:>12}")

    print(f"  {'-' * 62}")
    n = len(results)
    print(f"  {'TOTAL':<20} {keyword_total:>10} {token_total:>10} {combined_total:>10}")

    # Percentage improvement
    if keyword_total > 0:
        ft_pct = (token_total - keyword_total) / keyword_total * 100
        cb_pct = (combined_total - keyword_total) / keyword_total * 100
        print(f"\n  First-tokens vs keyword: {ft_pct:+.1f}%")
        print(f"  Combined vs keyword:     {cb_pct:+.1f}%")

    # ── Decision ──
    print(f"\n  DECISION:")
    if token_total <= keyword_total * 1.1:
        print("  First-20-token re-query does NOT meaningfully improve over keyword-only.")
        print("  Q3 CLOSED — keyword-only retrieval is sufficient.")
    else:
        improvement = (token_total - keyword_total) / keyword_total * 100
        print(f"  First-20-token re-query improves {improvement:.0f}% over keyword-only.")
        print("  Worth implementing if latency cost is acceptable.")

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
