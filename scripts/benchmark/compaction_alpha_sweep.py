#!/usr/bin/env python3
"""Offline dual-objective alpha sweep for context-folding summaries.

This implements the CF-2c.0/NIB2-43 training-free probe using existing
compaction/summarizer score CSVs. Raw summary text is not required: the
secondary objective is a leave-one-trace-out retrieval proxy over scored rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


DEFAULT_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class Candidate:
    source: str
    trace: str
    config: str
    faithfulness: float
    retention: float
    compression: float
    tokens_before: float
    tokens_after: float
    status: str

    @property
    def helpfulness(self) -> float:
        # Primary Phase-2a scorer proxy: factual/helpfulness quality only.
        # Retention is held out below as the downstream task-success proxy.
        return clamp01(self.faithfulness / 3.0)

    @property
    def task_success(self) -> int:
        # Downstream proxy: the judge could still recover the probe facts.
        return int(self.retention >= 3.0)

    @property
    def feature_vector(self) -> tuple[float, float, float]:
        compression = clamp01(self.compression)
        token_ratio = self.tokens_after / max(self.tokens_before, 1.0)
        return (
            clamp01(self.faithfulness / 3.0),
            compression,
            clamp01(token_ratio),
        )


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def parse_float(raw: str | None, default: float = -1.0) -> float:
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def load_candidates(paths: list[Path]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for path in paths:
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                status = row.get("status", "")
                faithfulness = parse_float(row.get("faithfulness"))
                retention = parse_float(row.get("retention"))
                if status != "live" or faithfulness < 0 or retention < 0:
                    continue
                compression = _compression_value(row)
                if compression <= 0:
                    continue
                candidates.append(
                    Candidate(
                        source=path.name,
                        trace=row["trace"],
                        config=_config_name(row),
                        faithfulness=faithfulness,
                        retention=retention,
                        compression=compression,
                        tokens_before=max(0.0, parse_float(row.get("tokens_before"), 0.0)),
                        tokens_after=max(0.0, parse_float(row.get("tokens_after"), 0.0)),
                        status=status,
                    )
                )
    return candidates


def _compression_value(row: dict[str, str]) -> float:
    if "compression_achieved" in row:
        return parse_float(row.get("compression_achieved"), 0.0)
    ratio = parse_float(row.get("compression_ratio"), 0.0)
    if ratio <= 0:
        return 0.0
    # summarizer_quality records before/after ratio, not fraction reduced.
    return 1.0 - (1.0 / ratio)


def _config_name(row: dict[str, str]) -> str:
    if row.get("level"):
        return f"L{row['level']}"
    if row.get("model_tier"):
        return f"{row['model_tier']}@{row.get('model_port', '?')}"
    return "unknown"


def estimate_task_success(candidates: list[Candidate], k: int) -> dict[int, float]:
    """Return leave-one-trace-out retrieval proxy probabilities by row index."""
    estimates: dict[int, float] = {}
    for idx, candidate in enumerate(candidates):
        neighbours: list[tuple[float, int]] = []
        for other_idx, other in enumerate(candidates):
            if idx == other_idx or candidate.trace == other.trace:
                continue
            neighbours.append((distance(candidate.feature_vector, other.feature_vector), other.task_success))
        if not neighbours:
            estimates[idx] = mean(c.task_success for c in candidates)
            continue
        nearest = sorted(neighbours, key=lambda item: item[0])[:k]
        estimates[idx] = mean(label for _, label in nearest)
    return estimates


def distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def alpha_sweep(candidates: list[Candidate], alphas: tuple[float, ...], k: int) -> list[dict]:
    success_proxy = estimate_task_success(candidates, k=k)
    rows: list[dict] = []
    for alpha in alphas:
        scored = []
        for idx, candidate in enumerate(candidates):
            score = alpha * candidate.helpfulness + (1.0 - alpha) * success_proxy[idx]
            scored.append((score, candidate.task_success))
        rows.append(
            {
                "alpha": alpha,
                "n": len(scored),
                "success_rate": round(mean(label for _, label in scored), 6),
                "average_precision": round(average_precision(scored), 6),
                "roc_auc": round(roc_auc(scored), 6),
                "top_10_success_rate": round(top_k_success_rate(scored, 10), 6),
                "spearman": round(spearman(scored), 6),
            }
        )
    return rows


def average_precision(scored: list[tuple[float, int]]) -> float:
    ranked = sorted(scored, key=lambda item: item[0], reverse=True)
    positives = sum(label for _, label in ranked)
    if positives == 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(ranked, start=1):
        if label:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / positives


def roc_auc(scored: list[tuple[float, int]]) -> float:
    positives = [score for score, label in scored if label]
    negatives = [score for score, label in scored if not label]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def top_k_success_rate(scored: list[tuple[float, int]], k: int) -> float:
    ranked = sorted(scored, key=lambda item: item[0], reverse=True)[:k]
    if not ranked:
        return 0.0
    return mean(label for _, label in ranked)


def spearman(scored: list[tuple[float, int]]) -> float:
    n = len(scored)
    if n < 2:
        return 0.0
    score_ranks = ranks([score for score, _ in scored])
    label_ranks = ranks([float(label) for _, label in scored])
    mean_score = mean(score_ranks)
    mean_label = mean(label_ranks)
    numerator = sum((a - mean_score) * (b - mean_label) for a, b in zip(score_ranks, label_ranks))
    denom_a = math.sqrt(sum((a - mean_score) ** 2 for a in score_ranks))
    denom_b = math.sqrt(sum((b - mean_label) ** 2 for b in label_ranks))
    if denom_a == 0 or denom_b == 0:
        return 0.0
    return numerator / (denom_a * denom_b)


def ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    output = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][1] == ordered[i][1]:
            j += 1
        rank = (i + j + 2) / 2.0
        for idx in range(i, j + 1):
            output[ordered[idx][0]] = rank
        i = j + 1
    return output


def write_outputs(rows: list[dict], output_json: Path | None, output_csv: Path | None) -> None:
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(rows, indent=2) + "\n")
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def parse_alphas(raw: str) -> tuple[float, ...]:
    alphas = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not alphas:
        raise ValueError("at least one alpha is required")
    for alpha in alphas:
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError(f"alpha must be in [0,1], got {alpha}")
    return alphas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="CSV files from compaction_sweep.py or eval_summarizer.py",
    )
    parser.add_argument("--alphas", default=",".join(str(a) for a in DEFAULT_ALPHAS))
    parser.add_argument("--k", type=int, default=5, help="Nearest neighbours for retrieval proxy")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()

    alphas = parse_alphas(args.alphas)
    candidates = load_candidates(args.inputs)
    if len(candidates) < 10:
        raise SystemExit(f"Need at least 10 valid candidates, found {len(candidates)}")

    rows = alpha_sweep(candidates, alphas, k=args.k)
    write_outputs(rows, args.output_json, args.output_csv)

    baseline = next((row for row in rows if row["alpha"] == 1.0), rows[-1])
    best = max(rows, key=lambda row: (row["average_precision"], row["roc_auc"]))
    improvement = best["average_precision"] - baseline["average_precision"]
    for row in rows:
        print(json.dumps(row))
    print(
        json.dumps(
            {
                "best_alpha": best["alpha"],
                "baseline_alpha": baseline["alpha"],
                "average_precision_delta": round(improvement, 6),
                "promote_dual_objective": best["alpha"] < 1.0 and improvement > 0.02,
            }
        )
    )


if __name__ == "__main__":
    main()
