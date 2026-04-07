#!/usr/bin/env python3
"""Helpfulness scoring calibration — validate heuristic weights against ground truth.

Compares the heuristic segment_helpfulness() scores against empirical "was this
segment referenced later?" ground truth from recorded session traces.

Requires model servers for LLM-based Δ_k measurement. Use --dry-run for offline.

Usage:
    python eval_helpfulness_calibration.py --dry-run
    python eval_helpfulness_calibration.py --traces-dir /mnt/raid0/llm/tmp/
    python eval_helpfulness_calibration.py --weight-sweep --output results/calibration.csv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


# Default weight configurations to sweep
WEIGHT_CONFIGS = [
    {"recency": 0.3, "overlap": 0.3, "outcome": 0.2, "sensitivity": 0.2},  # default
    {"recency": 0.4, "overlap": 0.3, "outcome": 0.2, "sensitivity": 0.1},  # recency-heavy
    {"recency": 0.2, "overlap": 0.4, "outcome": 0.2, "sensitivity": 0.2},  # overlap-heavy
    {"recency": 0.25, "overlap": 0.25, "outcome": 0.25, "sensitivity": 0.25},  # uniform
    {"recency": 0.1, "overlap": 0.5, "outcome": 0.3, "sensitivity": 0.1},  # overlap+outcome
]


def find_session_traces(traces_dir: Path) -> list[Path]:
    """Find session log files in the traces directory."""
    return sorted(traces_dir.glob("session_*.md"))


def calibrate_helpfulness(
    traces: list[Path],
    weight_configs: list[dict],
    *,
    dry_run: bool = False,
) -> list[dict]:
    """Run helpfulness calibration across traces and weight configs.

    Ground truth: For each segment, check if its identifiers appear
    in subsequent turns. Segments that ARE referenced later should
    have higher helpfulness scores.

    Returns list of result dicts with calibration metrics.
    """
    results = []

    for config_idx, weights in enumerate(weight_configs):
        if dry_run:
            # Mock calibration results
            result = {
                "config_idx": config_idx,
                "weights": weights,
                "n_traces": len(traces),
                "n_segments_scored": len(traces) * 12,
                "spearman_rho": 0.55 + config_idx * 0.05,  # mock correlation
                "precision_at_3": 0.70 + config_idx * 0.03,
                "ndcg": 0.65 + config_idx * 0.04,
                "mean_helpfulness_referenced": 0.72,
                "mean_helpfulness_unreferenced": 0.35,
                "separation": 0.72 - 0.35,
                "status": "dry_run",
            }
        else:
            result = run_calibration(traces, weights)

        results.append(result)

    return results


def run_calibration(traces: list[Path], weights: dict) -> dict:
    """Run calibration on traces with given weights.

    Pure heuristic — does NOT require model servers. Computes helpfulness
    scores from trace data and correlates with identifier-overlap ground truth.
    """
    from eval_helpers import parse_session_trace, extract_identifiers

    all_heuristic = []
    all_ground_truth = []
    outcome_map = {"ok": 1.0, "final": 1.0, "error": 0.5, "nudge": 0.2}

    for trace in traces:
        turns = parse_session_trace(trace)
        if len(turns) < 4:
            continue

        # Segment into groups of 2-3 consecutive turns
        seg_size = min(3, max(2, len(turns) // 4))
        segments = []
        for i in range(0, len(turns) - seg_size + 1, seg_size):
            segments.append(turns[i:i + seg_size])

        current_turn = turns[-1].turn_num

        for seg_turns in segments:
            seg_end = seg_turns[-1].turn_num
            distance = current_turn - seg_end

            # Recency signal
            recency = 1.0 / (1.0 + distance * 0.1)

            # Overlap signal: identifiers in segment vs subsequent turns
            seg_text = " ".join(t.raw_text for t in seg_turns)
            seg_ids = extract_identifiers(seg_text)
            subsequent = [t for t in turns if t.turn_num > seg_end]
            sub_text = " ".join(t.raw_text for t in subsequent)
            sub_ids = extract_identifiers(sub_text)
            overlap = len(seg_ids & sub_ids) / max(len(seg_ids), 1)

            # Outcome signal
            outcomes = [outcome_map.get(t.outcome, 0.5) for t in seg_turns]
            outcome = sum(outcomes) / len(outcomes)

            # Sensitivity signal
            has_code = any(t.code_hash for t in seg_turns)
            has_error = any(t.error for t in seg_turns)
            sensitivity = 1.0 if (has_code or has_error) else 0.5

            # Weighted combination
            heuristic = (
                weights["recency"] * recency
                + weights["overlap"] * overlap
                + weights["outcome"] * outcome
                + weights["sensitivity"] * sensitivity
            )

            # Ground truth: did ANY identifier from this segment appear later?
            ground_truth = 1.0 if len(seg_ids & sub_ids) > 0 else 0.0

            all_heuristic.append(heuristic)
            all_ground_truth.append(ground_truth)

    n = len(all_heuristic)
    if n < 2:
        return {
            "config_idx": 0,
            "weights": weights,
            "n_traces": len(traces),
            "n_segments_scored": n,
            "spearman_rho": 0.0,
            "precision_at_3": 0.0,
            "ndcg": 0.0,
            "mean_helpfulness_referenced": 0.0,
            "mean_helpfulness_unreferenced": 0.0,
            "separation": 0.0,
            "status": "insufficient_data",
        }

    rho = _spearman_correlation(all_heuristic, all_ground_truth)
    p_at_3 = _precision_at_k(all_heuristic, all_ground_truth, k=3)
    ndcg = _compute_ndcg(all_heuristic, all_ground_truth)

    referenced = [h for h, g in zip(all_heuristic, all_ground_truth) if g > 0]
    unreferenced = [h for h, g in zip(all_heuristic, all_ground_truth) if g == 0]
    mean_ref = sum(referenced) / max(len(referenced), 1)
    mean_unref = sum(unreferenced) / max(len(unreferenced), 1)

    return {
        "config_idx": 0,
        "weights": weights,
        "n_traces": len(traces),
        "n_segments_scored": n,
        "spearman_rho": round(rho, 4),
        "precision_at_3": round(p_at_3, 4),
        "ndcg": round(ndcg, 4),
        "mean_helpfulness_referenced": round(mean_ref, 4),
        "mean_helpfulness_unreferenced": round(mean_unref, 4),
        "separation": round(mean_ref - mean_unref, 4),
        "status": "live",
    }


def _rank(values: list[float]) -> list[float]:
    """Compute ranks with average tie-breaking."""
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 1) / 2  # 1-based average
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def _spearman_correlation(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation without scipy."""
    n = len(x)
    if n < 2:
        return 0.0
    rx = _rank(x)
    ry = _rank(y)
    d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1.0 - 6.0 * d_sq / (n * (n * n - 1))


def _precision_at_k(scores: list[float], labels: list[float], k: int = 3) -> float:
    """Precision of top-k scored items having positive ground truth."""
    if not scores or k <= 0:
        return 0.0
    paired = sorted(zip(scores, labels), key=lambda x: -x[0])
    top_k = paired[:k]
    return sum(1 for _, l in top_k if l > 0) / len(top_k)


def _compute_ndcg(scores: list[float], labels: list[float]) -> float:
    """Normalized Discounted Cumulative Gain."""
    if not scores:
        return 0.0
    paired = sorted(zip(scores, labels), key=lambda x: -x[0])
    dcg = sum(l / math.log2(i + 2) for i, (_, l) in enumerate(paired))
    ideal = sorted(labels, reverse=True)
    idcg = sum(l / math.log2(i + 2) for i, l in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate helpfulness scoring weights against ground truth",
    )
    parser.add_argument(
        "--traces-dir",
        type=Path,
        default=Path("/mnt/raid0/llm/tmp"),
        help="Directory containing session trace files",
    )
    parser.add_argument(
        "--weight-sweep",
        action="store_true",
        help="Sweep multiple weight configurations (default: just the default config)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: stdout as JSON)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate mock results without loading traces",
    )
    args = parser.parse_args()

    if args.weight_sweep:
        configs = WEIGHT_CONFIGS
    else:
        configs = [WEIGHT_CONFIGS[0]]  # just the default

    if args.dry_run:
        traces = [Path(f"synthetic_session_{i}.md") for i in range(10)]
    else:
        traces = find_session_traces(args.traces_dir)
        if not traces:
            print(f"No session traces found in {args.traces_dir}", file=sys.stderr)
            sys.exit(1)

    results = calibrate_helpfulness(traces, configs, dry_run=args.dry_run)

    if args.output:
        import csv
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [k for k in results[0].keys() if k != "weights"]
        fieldnames.extend(["w_recency", "w_overlap", "w_outcome", "w_sensitivity"])
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                flat = {k: v for k, v in r.items() if k != "weights"}
                flat["w_recency"] = r["weights"]["recency"]
                flat["w_overlap"] = r["weights"]["overlap"]
                flat["w_outcome"] = r["weights"]["outcome"]
                flat["w_sensitivity"] = r["weights"]["sensitivity"]
                writer.writerow(flat)
        print(f"Wrote {len(results)} results to {args.output}")
    else:
        for r in results:
            print(json.dumps(r, default=str))

    # Summary
    if args.dry_run:
        print(f"\n[DRY RUN] {len(results)} calibration configs evaluated")
        best = max(results, key=lambda r: r["spearman_rho"])
        print(f"  Best config: {best['weights']}")
        print(f"  Spearman rho: {best['spearman_rho']:.3f}")
        print(f"  Precision@3: {best['precision_at_3']:.3f}")
        print(f"  Separation: {best['separation']:.3f}")


if __name__ == "__main__":
    main()
