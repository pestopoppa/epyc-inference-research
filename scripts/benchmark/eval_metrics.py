#!/usr/bin/env python3
"""Efficiency metrics for eval framework: OAA and PTI.

OAA (Overthinking-Adjusted Accuracy): Penalizes correct answers that use
excessive tokens. From OckBench (arXiv 2511.05722).

PTI (Per-Token Intelligence): Accuracy normalized by token cost.
Higher = more efficient model.

Usage as library:
    from eval_metrics import compute_oaa, compute_pti, compute_batch_oaa

Usage as CLI:
    python eval_metrics.py --results path/to/results.jsonl --alpha 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def compute_oaa(
    accuracy: float,
    actual_tokens: int,
    reference_tokens: int,
    alpha: float = 0.5,
) -> float:
    """Overthinking-Adjusted Accuracy.

    OAA = accuracy * (1 - alpha * excess_ratio)
    where excess_ratio = max(0, (actual_tokens - reference_tokens) / reference_tokens)

    Args:
        accuracy: Model accuracy (0.0-1.0).
        actual_tokens: Tokens used by this model.
        reference_tokens: Tokens used by shortest correct answer (across models).
        alpha: Penalty factor (default 0.5, higher = harsher penalty).

    Returns:
        OAA score. Can go negative with very high alpha + excess.
    """
    if reference_tokens <= 0:
        return accuracy
    excess_ratio = max(0.0, (actual_tokens - reference_tokens) / reference_tokens)
    return accuracy * (1.0 - alpha * excess_ratio)


def compute_pti(
    accuracy: float,
    tokens_per_correct: int,
) -> float:
    """Per-Token Intelligence.

    PTI = accuracy / tokens_per_correct_answer
    Higher = more efficient model.

    Args:
        accuracy: Model accuracy (0.0-1.0).
        tokens_per_correct: Average tokens per correct answer.

    Returns:
        PTI score (higher is better).
    """
    if tokens_per_correct <= 0:
        return 0.0
    return accuracy / tokens_per_correct


def compute_batch_oaa(
    results: list[dict],
    alpha: float = 0.5,
    token_key: str = "total_tokens",
    correct_key: str = "correct",
) -> dict:
    """Compute OAA and PTI for a batch of results.

    Determines reference_tokens as the minimum tokens among correct answers,
    then computes OAA for the overall run.

    Args:
        results: List of result dicts (from eval_trimr.py or similar).
        alpha: Penalty factor.
        token_key: Key for token count in result dicts.
        correct_key: Key for correctness boolean.

    Returns:
        {accuracy, oaa, pti, avg_tokens, reference_tokens}
    """
    if not results:
        return {
            "accuracy": 0.0, "oaa": 0.0, "pti": 0.0,
            "avg_tokens": 0.0, "reference_tokens": 0,
        }

    scored = [r for r in results if r.get(correct_key) is not None]
    correct = [r for r in scored if r[correct_key]]

    accuracy = len(correct) / max(len(scored), 1)
    avg_tokens = sum(r.get(token_key, 0) for r in scored) / max(len(scored), 1)

    correct_tokens = [r.get(token_key, 0) for r in correct if r.get(token_key, 0) > 0]
    reference_tokens = min(correct_tokens) if correct_tokens else int(avg_tokens)

    tokens_per_correct = (
        sum(correct_tokens) / len(correct_tokens) if correct_tokens else 0
    )

    oaa = compute_oaa(accuracy, int(avg_tokens), reference_tokens, alpha)
    pti = compute_pti(accuracy, int(tokens_per_correct))

    return {
        "accuracy": round(accuracy, 4),
        "oaa": round(oaa, 4),
        "pti": round(pti, 8),
        "avg_tokens": round(avg_tokens, 1),
        "reference_tokens": reference_tokens,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute OAA and PTI from eval results",
    )
    parser.add_argument(
        "--results", type=Path, required=True,
        help="Path to results JSONL file",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="OAA penalty factor (default: 0.5)",
    )
    parser.add_argument(
        "--token-key", type=str, default="total_tokens",
        help="Key for token count in result dicts",
    )
    parser.add_argument(
        "--correct-key", type=str, default="correct",
        help="Key for correctness boolean in result dicts",
    )
    args = parser.parse_args()

    if not args.results.exists():
        print(f"Error: {args.results} not found", file=sys.stderr)
        sys.exit(1)

    results = []
    with open(args.results) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    metrics = compute_batch_oaa(
        results,
        alpha=args.alpha,
        token_key=args.token_key,
        correct_key=args.correct_key,
    )

    print(f"Results: {len(results)} entries")
    print(f"Accuracy: {metrics['accuracy']:.1%}")
    print(f"OAA (alpha={args.alpha}): {metrics['oaa']:.4f}")
    print(f"PTI: {metrics['pti']:.6f}")
    print(f"Avg tokens: {metrics['avg_tokens']:.0f}")
    print(f"Reference tokens: {metrics['reference_tokens']}")


if __name__ == "__main__":
    main()
