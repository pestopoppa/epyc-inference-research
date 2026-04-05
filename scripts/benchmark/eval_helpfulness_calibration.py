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
    """Run live calibration on traces with given weights.

    Steps per trace:
    1. Load session log, extract ConsolidatedSegments
    2. For each segment, compute heuristic helpfulness with given weights
    3. For each segment, compute ground truth: were its identifiers
       referenced in subsequent turns?
    4. Compute Spearman correlation between heuristic and ground truth
    """
    raise NotImplementedError(
        "Live calibration requires session traces with full TurnRecord data. "
        "Use --dry-run for offline testing."
    )


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
