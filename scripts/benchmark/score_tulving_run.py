#!/usr/bin/env python3
"""Score a Tulving episodic-memory benchmark result file offline.

``run_benchmark.py`` stores raw responses and generic throughput summaries.
This helper rehydrates Tulving ground truth from the dataset adapter and emits
the benchmark-specific Simple Recall and Chronological Awareness metrics.
It is safe to run on partial result files while a benchmark is still active.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tulving_episodic_adapter import (
    TulvingEpisodicAdapter,
    _extract_list_from_response,
    _token_f1,
    compute_chronological_awareness_score,
    compute_simple_recall_score,
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def build_prompt_index() -> dict[str, dict[str, Any]]:
    adapter = TulvingEpisodicAdapter()
    return {item["id"]: item for item in adapter.extract_all()}


def _kendall_tau(indices: list[int]) -> float:
    if len(indices) < 2:
        return 0.0
    concordant = 0
    discordant = 0
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            if indices[i] < indices[j]:
                concordant += 1
            elif indices[i] > indices[j]:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 0.0


def chronological_tau(response: str, prompt_dict: dict[str, Any], *, threshold: float = 0.5) -> float:
    """Approximate Tulving chronological-order score from a model response.

    The answer parser extracts ordered model items, then greedily maps each
    predicted item to its best unmatched ground-truth item. Kendall tau is
    computed over the matched ground-truth indices in predicted order.
    """
    ground_truth = list(prompt_dict.get("metadata", {}).get("ground_truth_items", []))
    if len(ground_truth) < 2:
        return 0.0

    predicted = _extract_list_from_response(response)
    matched_indices: list[int] = []
    used: set[int] = set()
    for item in predicted:
        best_idx = -1
        best_score = 0.0
        for idx, gt_item in enumerate(ground_truth):
            if idx in used:
                continue
            score = _token_f1(item, gt_item)
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_idx >= 0 and best_score >= threshold:
            used.add(best_idx)
            matched_indices.append(best_idx)

    return _kendall_tau(matched_indices)


def score_result_payload(payload: dict[str, Any], prompt_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    suite_results = payload.get("results", {}).get("tulving_episodic", {})
    per_question: list[dict[str, Any]] = []
    simple_inputs: list[dict[str, Any]] = []
    latest_inputs: list[dict[str, Any]] = []
    chronological_inputs: list[dict[str, Any]] = []
    missing_ground_truth: list[str] = []

    for question_id, row in sorted(suite_results.items()):
        prompt_dict = prompt_index.get(question_id)
        if prompt_dict is None:
            missing_ground_truth.append(question_id)
            continue

        response = row.get("response", "")
        score = TulvingEpisodicAdapter.compute_f1_for_result(response, prompt_dict)
        meta = prompt_dict.get("metadata", {})
        scored = {
            "question_id": question_id,
            "f1": score["f1"],
            "precision": score["precision"],
            "recall": score["recall"],
            "nb_gt": score["nb_gt"],
            "nb_pred": score["nb_pred"],
            "retrieval_type": score.get("retrieval_type", ""),
            "get_style": score.get("get_style", ""),
            "tokens_per_second": row.get("tokens_per_second"),
            "completion_tokens": row.get("completion_tokens"),
            "ground_truth_items": meta.get("ground_truth_items", []),
            "matched_gt_items": score.get("matched_gt_items", []),
        }
        if scored["get_style"] == "chronological":
            scored["kendall_tau"] = chronological_tau(response, prompt_dict)
            chronological_inputs.append(scored)
        if scored["get_style"] == "latest":
            latest_inputs.append(scored)

        per_question.append(scored)
        simple_inputs.append(scored)

    avg_f1 = sum(row["f1"] for row in per_question) / len(per_question) if per_question else 0.0
    tps_values = [
        row["tokens_per_second"]
        for row in per_question
        if isinstance(row.get("tokens_per_second"), (int, float))
    ]
    avg_tps = sum(tps_values) / len(tps_values) if tps_values else None

    by_retrieval: dict[str, dict[str, Any]] = {}
    for row in per_question:
        key = row["retrieval_type"] or "unknown"
        bucket = by_retrieval.setdefault(key, {"count": 0, "avg_f1": 0.0})
        bucket["count"] += 1
        bucket["avg_f1"] += row["f1"]
    for bucket in by_retrieval.values():
        bucket["avg_f1"] /= bucket["count"]

    summary = {
        "run_id": payload.get("run_id"),
        "model_role": payload.get("model_role"),
        "config_name": payload.get("config_name"),
        "result_questions": len(suite_results),
        "scored_questions": len(per_question),
        "missing_ground_truth": len(missing_ground_truth),
        "avg_f1": avg_f1,
        "simple_recall_score": compute_simple_recall_score(simple_inputs),
        "chronological_awareness_score": compute_chronological_awareness_score(
            latest_inputs, chronological_inputs
        ),
        "latest_questions": len(latest_inputs),
        "chronological_questions": len(chronological_inputs),
        "avg_tokens_per_second": avg_tps,
        "by_retrieval_type": by_retrieval,
    }
    return {
        "summary": summary,
        "missing_ground_truth_ids": missing_ground_truth,
        "per_question": per_question,
    }


def render_markdown(scored: dict[str, Any], result_path: Path) -> str:
    summary = scored["summary"]
    lines = [
        "# Tulving Episodic Run Score",
        "",
        f"- Result file: `{result_path}`",
        f"- Run ID: `{summary.get('run_id')}`",
        f"- Model role: `{summary.get('model_role')}`",
        f"- Config: `{summary.get('config_name')}`",
        f"- Scored questions: {summary['scored_questions']} / {summary['result_questions']}",
        f"- Missing ground truth: {summary['missing_ground_truth']}",
        f"- Average F1: {summary['avg_f1']:.4f}",
        f"- Simple Recall Score: {summary['simple_recall_score']:.4f}",
        f"- Chronological Awareness Score: {summary['chronological_awareness_score']:.4f}",
    ]
    if summary.get("avg_tokens_per_second") is not None:
        lines.append(f"- Average tokens/sec: {summary['avg_tokens_per_second']:.2f}")
    lines.extend(["", "## By Retrieval Type", ""])
    lines.append("| Retrieval type | Count | Avg F1 |")
    lines.append("|---|---:|---:|")
    for retrieval_type, bucket in sorted(summary["by_retrieval_type"].items()):
        lines.append(f"| {retrieval_type} | {bucket['count']} | {bucket['avg_f1']:.4f} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="Path to ingest_long_context_*.json")
    parser.add_argument("--out-json", type=Path, default=None, help="Write scored JSON report")
    parser.add_argument("--out-md", type=Path, default=None, help="Write Markdown summary")
    args = parser.parse_args()

    payload = _load_json(args.result)
    scored = score_result_payload(payload, build_prompt_index())

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(scored, indent=2) + "\n")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(render_markdown(scored, args.result) + "\n")

    print(json.dumps(scored["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
