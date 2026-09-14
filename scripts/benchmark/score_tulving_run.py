#!/usr/bin/env python3
"""Score a Tulving episodic-memory benchmark result file offline.

``run_benchmark.py`` stores raw responses and generic throughput summaries.
This helper rehydrates Tulving ground truth from the dataset adapter and emits
the benchmark-specific Simple Recall and Chronological Awareness metrics.
It is safe to run on partial result files while a benchmark is still active.

It performs NO inference: it reads stored responses out of a result file and
re-derives ground truth from the local dataset parquet. Nothing in this module
or in ``tulving_episodic_adapter`` opens a socket or loads a model.

Subset discipline (M-12e): Simple Recall is computed over the ``get == "all"``
subset ONLY, and the Kendall tau leg of Chronological Awareness fails closed
unless the matched set covers the full ordered ground truth. See
``SCORER_VERSION`` below and the ``tulving_episodic_adapter`` module docstring
for the evidence that fixes those definitions.

Belief-kernel write side (SC67): ``--belief-measurements`` emits a
producer-authored ``belief_measurements.jsonl`` sidecar beside ``--out-json``,
projected by ``epyc-root:scripts/vidya/adapters/tulving_episodic.py``. It
requires ``--arm``, ``--variant`` and ``--chapters``, because the result file does
not record them and a tuple that guesses the arm claims warrant the run never
captured. Runs scored before this hook existed emit zero rows, permanently —
including re-scores of them, whose arm identity was never recorded.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

from tulving_episodic_adapter import (
    CHRONOLOGICAL_GET_STYLE,
    LATEST_GET_STYLE,
    SIMPLE_RECALL_GET_STYLE,
    TulvingEpisodicAdapter,
    _extract_list_from_response,
    _token_f1,
    compute_chronological_awareness_score,
    compute_simple_recall_score,
    simple_recall_bin_basis,
    simple_recall_bin_counts,
)

#: Bumped whenever the scoring semantics change. Any consumer comparing two
#: scored artifacts MUST compare this first (M-12e).
#:   1  pre-2026-09-14: Simple Recall over EVERY question, Kendall tau with no
#:      coverage requirement, bins keyed on ground-truth item count.
#:   2  2026-09-14 (M-12e): Simple Recall over the ``get == "all"`` subset only,
#:      tau fails closed unless the matched set covers the full ground truth,
#:      bins keyed on matching-event count (``nb_events``).
SCORER_VERSION = 2


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


def chronological_tau_detail(
    response: str,
    prompt_dict: dict[str, Any],
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Kendall tau for one chronological question, with its coverage state.

    The answer parser extracts ordered model items, then greedily maps each
    predicted item to its best unmatched ground-truth item. Kendall tau is
    computed over the matched ground-truth indices in predicted order.

    **Coverage is a precondition, not a detail (M-12e).** Tau over a partial
    match scores the ordering of whatever the model happened to emit, so a model
    that emits two items in the right order out of nine ground-truth items
    scores 1.0 — the metric rewards emitting less. The reported ``tau``
    therefore **fails closed at 0.0** unless the matched set covers the FULL
    ground truth; ``tau_raw`` keeps the uncovered value as a diagnostic and must
    never be averaged into a headline.

    Returns a dict with:
      tau            the value to score with (0.0 unless full coverage)
      tau_raw        tau over the matched subset, whatever its coverage
      nb_gt          ground-truth item count
      nb_matched     matched ground-truth item count
      coverage       nb_matched / nb_gt (0.0 when nb_gt == 0)
      full_coverage  bool — whether ``tau`` was allowed to be non-zero
      status         "scored" | "partial" | "too_short"
    """
    ground_truth = list(prompt_dict.get("metadata", {}).get("ground_truth_items", []))
    nb_gt = len(ground_truth)
    if nb_gt < 2:
        # Ordering is undefined for fewer than two items; not a partial match.
        return {
            "tau": 0.0,
            "tau_raw": 0.0,
            "nb_gt": nb_gt,
            "nb_matched": 0,
            "coverage": 0.0,
            "full_coverage": False,
            "status": "too_short",
        }

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

    tau_raw = _kendall_tau(matched_indices)
    full_coverage = len(used) == nb_gt
    return {
        "tau": tau_raw if full_coverage else 0.0,
        "tau_raw": tau_raw,
        "nb_gt": nb_gt,
        "nb_matched": len(used),
        "coverage": len(used) / nb_gt,
        "full_coverage": full_coverage,
        "status": "scored" if full_coverage else "partial",
    }


def chronological_tau(response: str, prompt_dict: dict[str, Any], *, threshold: float = 0.5) -> float:
    """Fail-closed Kendall tau: 0.0 unless the match covers the full ground truth.

    Thin wrapper over :func:`chronological_tau_detail`; use that when you need
    to know WHY a question scored 0.0.
    """
    return chronological_tau_detail(response, prompt_dict, threshold=threshold)["tau"]


def score_result_payload(payload: dict[str, Any], prompt_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    suite_results = payload.get("results", {}).get("tulving_episodic", {})
    per_question: list[dict[str, Any]] = []
    simple_inputs: list[dict[str, Any]] = []
    latest_inputs: list[dict[str, Any]] = []
    chronological_inputs: list[dict[str, Any]] = []
    missing_ground_truth: list[str] = []
    unknown_get_styles: list[str] = []

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
            "nb_events": meta.get("nb_events"),
            "ground_truth_items": meta.get("ground_truth_items", []),
            "matched_gt_items": score.get("matched_gt_items", []),
        }
        if scored["get_style"] == CHRONOLOGICAL_GET_STYLE:
            detail = chronological_tau_detail(response, prompt_dict)
            scored["kendall_tau"] = detail["tau"]
            scored["kendall_tau_raw"] = detail["tau_raw"]
            scored["tau_coverage"] = detail["coverage"]
            scored["tau_matched_gt"] = detail["nb_matched"]
            scored["tau_full_coverage"] = detail["full_coverage"]
            scored["tau_status"] = detail["status"]
            chronological_inputs.append(scored)
        elif scored["get_style"] == LATEST_GET_STYLE:
            latest_inputs.append(scored)
        elif scored["get_style"] == SIMPLE_RECALL_GET_STYLE:
            # M-12e: ONLY the "all" get style is the Simple Recall subset.
            simple_inputs.append(scored)
        else:
            unknown_get_styles.append(question_id)

        per_question.append(scored)

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

    partial_tau = [
        row["question_id"] for row in chronological_inputs
        if not row.get("tau_full_coverage")
    ]

    summary = {
        "scorer_version": SCORER_VERSION,
        "run_id": payload.get("run_id"),
        "model_role": payload.get("model_role"),
        "config_name": payload.get("config_name"),
        "result_questions": len(suite_results),
        "scored_questions": len(per_question),
        "missing_ground_truth": len(missing_ground_truth),
        "unknown_get_style": len(unknown_get_styles),
        "avg_f1": avg_f1,
        "simple_recall_score": compute_simple_recall_score(simple_inputs),
        "simple_recall_questions": len(simple_inputs),
        "simple_recall_bin_basis": simple_recall_bin_basis(simple_inputs),
        "simple_recall_bins": simple_recall_bin_counts(simple_inputs),
        "chronological_awareness_score": compute_chronological_awareness_score(
            latest_inputs, chronological_inputs
        ),
        "latest_questions": len(latest_inputs),
        "chronological_questions": len(chronological_inputs),
        "chronological_partial_coverage": len(partial_tau),
        "avg_tokens_per_second": avg_tps,
        "by_retrieval_type": by_retrieval,
    }
    return {
        "summary": summary,
        "missing_ground_truth_ids": missing_ground_truth,
        "unknown_get_style_ids": unknown_get_styles,
        "chronological_partial_coverage_ids": partial_tau,
        "per_question": per_question,
    }


def render_markdown(scored: dict[str, Any], result_path: Path) -> str:
    summary = scored["summary"]
    lines = [
        "# Tulving Episodic Run Score",
        "",
        f"- Result file: `{result_path}`",
        f"- Scorer version: {summary.get('scorer_version')}",
        f"- Run ID: `{summary.get('run_id')}`",
        f"- Model role: `{summary.get('model_role')}`",
        f"- Config: `{summary.get('config_name')}`",
        f"- Scored questions: {summary['scored_questions']} / {summary['result_questions']}",
        f"- Missing ground truth: {summary['missing_ground_truth']}",
        f"- Average F1: {summary['avg_f1']:.4f}",
        f"- Simple Recall Score: {summary['simple_recall_score']:.4f}"
        f" (over {summary.get('simple_recall_questions', 0)}"
        f" `get={SIMPLE_RECALL_GET_STYLE}` questions,"
        f" bin basis `{summary.get('simple_recall_bin_basis')}`)",
        f"- Chronological Awareness Score: {summary['chronological_awareness_score']:.4f}",
        f"- Chronological questions failed closed for partial coverage: "
        f"{summary.get('chronological_partial_coverage', 0)}"
        f" / {summary.get('chronological_questions', 0)}",
    ]
    if summary.get("unknown_get_style"):
        lines.append(
            f"- **Unknown get styles (excluded from every subset): "
            f"{summary['unknown_get_style']}**"
        )
    if summary.get("avg_tokens_per_second") is not None:
        lines.append(f"- Average tokens/sec: {summary['avg_tokens_per_second']:.2f}")
    if summary.get("simple_recall_bins"):
        lines.extend(["", "## Simple Recall Bins (matching events)", ""])
        lines.append("| Bin | Count | Avg F1 |")
        lines.append("|---|---:|---:|")
        for label, bucket in summary["simple_recall_bins"].items():
            lines.append(f"| {label} | {bucket['count']} | {bucket['avg_f1']:.4f} |")
    lines.extend(["", "## By Retrieval Type", ""])
    lines.append("| Retrieval type | Count | Avg F1 |")
    lines.append("|---|---:|---:|")
    for retrieval_type, bucket in sorted(summary["by_retrieval_type"].items()):
        lines.append(f"| {retrieval_type} | {bucket['count']} | {bucket['avg_f1']:.4f} |")
    lines.append("")
    return "\n".join(lines)


#: Where the belief-kernel write-side vocabulary lives. It is hosted in epyc-root so the
#: writer and the strict reader cannot drift into two dialects of one schema; the scorer
#: imports it rather than re-deriving the row shape here (SC67, the CT-8 precedent).
_ROOT_CANDIDATES = (
    os.environ.get("EPYC_ROOT", ""),
    "/mnt/raid0/llm/epyc-root",
    "/workspace",
)


def _load_belief_capture():
    """Import ``tulving_episodic_capture`` from epyc-root, or explain why it is unavailable."""
    tried = []
    for root in _ROOT_CANDIDATES:
        if not root:
            continue
        adapters = Path(root) / "scripts" / "vidya" / "adapters"
        module = adapters / "tulving_episodic_capture.py"
        tried.append(str(module))
        if not module.is_file():
            continue
        spec = importlib.util.spec_from_file_location("tulving_episodic_capture", module)
        if spec is None or spec.loader is None:  # pragma: no cover - defensive
            continue
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        return loaded
    raise SystemExit(
        "--belief-measurements needs epyc-root's tulving_episodic_capture.py "
        "(set EPYC_ROOT). Looked in: " + ", ".join(tried)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="Path to ingest_long_context_*.json")
    parser.add_argument("--out-json", type=Path, default=None, help="Write scored JSON report")
    parser.add_argument("--out-md", type=Path, default=None, help="Write Markdown summary")
    belief = parser.add_argument_group(
        "belief kernel (SC67)",
        "Emit the producer-authored claim-tuple sidecar beside --out-json. Requires the arm "
        "and dataset identity, because the scored artifact does not carry them and nothing "
        "may be guessed on read.",
    )
    belief.add_argument("--belief-measurements", action="store_true",
                        help="Write belief_measurements.jsonl beside --out-json")
    belief.add_argument("--arm", default=None,
                        help="M-12a arm: none (memory-off) | retrieved | full (ceiling)")
    belief.add_argument("--variant", default=None, help="Dataset variant of this run")
    belief.add_argument("--chapters", type=int, default=None,
                        help="Chapter count of the book this run scored")
    belief.add_argument("--run-id", default=None,
                        help="Run id override (default: the result file's run_id)")
    belief.add_argument("--category", default="CANDIDATE",
                        choices=("OPTIMUM", "BASELINE", "CANDIDATE"))
    args = parser.parse_args()

    payload = _load_json(args.result)
    scored = score_result_payload(payload, build_prompt_index())

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(scored, indent=2) + "\n")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(render_markdown(scored, args.result) + "\n")

    if args.belief_measurements:
        if not args.out_json:
            raise SystemExit("--belief-measurements requires --out-json: the sidecar attests "
                             "the scored artifact, so the artifact must be durable first")
        missing = [name for name, value in
                   (("--arm", args.arm), ("--variant", args.variant),
                    ("--chapters", args.chapters)) if not value]
        if missing:
            raise SystemExit(
                "--belief-measurements requires " + ", ".join(missing)
                + ". The harness does not record the arm or the book identity, and a tuple "
                  "that guesses them claims warrant the run never captured.")
        capture = _load_belief_capture()
        run_id = args.run_id or scored["summary"].get("run_id")
        if not run_id:
            raise SystemExit("--belief-measurements needs a run id (--run-id)")
        sidecar = capture.write_belief_measurements(
            args.out_json,
            summary=scored["summary"],
            run_id=str(run_id),
            producer="score_tulving_run.py",
            arm=args.arm,
            variant=args.variant,
            chapters=args.chapters,
            category=args.category,
        )
        print(f"belief sidecar: {sidecar}")

    print(json.dumps(scored["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
