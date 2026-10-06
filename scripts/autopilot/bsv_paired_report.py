#!/usr/bin/env python3
"""Read-only BSV-2 paired behavior-signature report.

This CLI does not run inference and does not gate AutoPilot acceptance by itself.
It converts already-journaled same-question eval vectors into baseline/candidate
behavior signatures, compares them with `diff_signatures`, and emits the exact
evidence a future accept gate needs: shared-qid coverage, scalar paired delta,
signature severity, and blockers.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ORCH_ROOT))

try:
    from scripts.autopilot.paired_stats import (  # type: ignore[import-not-found]
        DEFAULT_JOURNAL_DIR,
        McNemarResult,
        QuestionOutcome,
        group_rows_by_fingerprint,
        iter_journal_rows,
        majority_vector,
        mcnemar_from_vectors,
        trial_vectors,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from paired_stats import (  # type: ignore[no-redef]
        DEFAULT_JOURNAL_DIR,
        McNemarResult,
        QuestionOutcome,
        group_rows_by_fingerprint,
        iter_journal_rows,
        majority_vector,
        mcnemar_from_vectors,
        trial_vectors,
    )

from src.autopilot_core.measurement_guards import (
    DISPOSITION_INFRA_FAILED,
    DISPOSITION_SCORED,
    DISPOSITION_SCORING_FAILED,
    DISPOSITION_TASK_FAILED,
    is_quality_admissible,
)
from src.behavior_signature import compute_behavior_signature, diff_signatures

DEFAULT_MIN_SHARED_QIDS = 35


def _route_path_from_row(row: dict[str, Any]) -> list[str] | None:
    routing = row.get("routing_distribution") or (row.get("eval_details") or {}).get(
        "routing_distribution"
    )
    if not isinstance(routing, dict):
        return None
    items = sorted(routing.items())
    return [f"{role}:{_weight_bucket(weight)}" for role, weight in items] or None


def _weight_bucket(weight: Any) -> str:
    try:
        w = float(weight)
    except (TypeError, ValueError):
        return "q?"
    return "q1" if w < 0.25 else "q2" if w < 0.5 else "q3" if w < 0.75 else "q4"


def _avg_prompt_tokens_from_rows(rows: list[dict[str, Any]]) -> float | None:
    values: list[float] = []
    for row in rows:
        for key in ("avg_prompt_tokens", "instruction_tokens"):
            value = row.get(key)
            if value is None:
                value = (row.get("eval_details") or {}).get(key)
            try:
                values.append(float(value))
                break
            except (TypeError, ValueError):
                continue
    return statistics.median(values) if values else None


def _sentinel_outcomes(vector: dict[str, QuestionOutcome]) -> dict[str, str]:
    return {
        qid: "pass" if outcome.correct else "fail"
        for qid, outcome in sorted(vector.items())
    }


def _signature(
    *,
    archive_member_id: str,
    trial_id: int | None,
    vector: dict[str, QuestionOutcome],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    route_path = _route_path_from_row(rows[-1]) if rows else None
    sig = compute_behavior_signature(
        archive_member_id=archive_member_id,
        trial_id=trial_id,
        sentinel_outcomes=_sentinel_outcomes(vector),
        route_path=route_path,
        total_tokens=_avg_prompt_tokens_from_rows(rows),
        signature_confidence="partial",
    )
    return {
        "archive_member_id": sig.archive_member_id,
        "trial_id": sig.trial_id,
        "sentinel_outcomes": sig.sentinel_outcomes or {},
        "route_path_hash": sig.route_path_hash,
        "tool_sequence_hash": sig.tool_sequence_hash,
        "escalation_path_hash": sig.escalation_path_hash,
        "latency_bucket": sig.latency_bucket,
        "token_bucket": sig.token_bucket,
        "signature_hash": sig.signature_hash,
        "signature_confidence": sig.signature_confidence,
    }


def _decision(
    *,
    severity: str,
    stats: McNemarResult,
    min_shared_qids: int,
    max_accuracy_regression: float,
) -> tuple[str, list[str]]:
    blockers: list[str] = []
    if stats.shared_qids < min_shared_qids:
        blockers.append(f"shared_qids {stats.shared_qids} < {min_shared_qids}")
    if stats.delta_b_minus_a < -abs(max_accuracy_regression):
        blockers.append(
            "candidate accuracy regression "
            f"{stats.delta_b_minus_a:.6f} < -{abs(max_accuracy_regression):.6f}"
        )
    if severity == "blocking":
        blockers.append("behavior signature severity is blocking")
    return ("pass" if not blockers else "block", blockers)


def build_trial_pair_report(
    rows: list[dict[str, Any]],
    *,
    baseline_trial: int,
    candidate_trial: int,
    min_shared_qids: int = DEFAULT_MIN_SHARED_QIDS,
    max_accuracy_regression: float = 0.0,
) -> dict[str, Any]:
    vectors = trial_vectors(rows)
    by_trial = {int(row["trial_id"]): row for row in rows if row.get("trial_id") is not None}
    try:
        baseline_vector = vectors[baseline_trial]
        candidate_vector = vectors[candidate_trial]
        baseline_row = by_trial[baseline_trial]
        candidate_row = by_trial[candidate_trial]
    except KeyError as exc:
        raise ValueError(f"trial has no question_results vector: {exc}") from exc

    stats = mcnemar_from_vectors(
        baseline_vector,
        candidate_vector,
        str(baseline_trial),
        str(candidate_trial),
    )
    baseline_sig = _signature(
        archive_member_id=f"trial:{baseline_trial}",
        trial_id=baseline_trial,
        vector=baseline_vector,
        rows=[baseline_row],
    )
    candidate_sig = _signature(
        archive_member_id=f"trial:{candidate_trial}",
        trial_id=candidate_trial,
        vector=candidate_vector,
        rows=[candidate_row],
    )
    return _paired_report(
        comparison_type="trial_pair",
        baseline_label=str(baseline_trial),
        candidate_label=str(candidate_trial),
        stats=stats,
        baseline_signature=baseline_sig,
        candidate_signature=candidate_sig,
        min_shared_qids=min_shared_qids,
        max_accuracy_regression=max_accuracy_regression,
    )


def build_fingerprint_pair_report(
    rows: list[dict[str, Any]],
    *,
    baseline_fingerprint: str,
    candidate_fingerprint: str,
    min_shared_qids: int = DEFAULT_MIN_SHARED_QIDS,
    max_accuracy_regression: float = 0.0,
) -> dict[str, Any]:
    grouped = group_rows_by_fingerprint(rows)
    baseline_rows = grouped.get(baseline_fingerprint, [])
    candidate_rows = grouped.get(candidate_fingerprint, [])
    if not baseline_rows:
        raise ValueError(f"baseline fingerprint has no question_results vector: {baseline_fingerprint}")
    if not candidate_rows:
        raise ValueError(f"candidate fingerprint has no question_results vector: {candidate_fingerprint}")

    baseline_vector = majority_vector(baseline_rows)
    candidate_vector = majority_vector(candidate_rows)
    stats = mcnemar_from_vectors(
        baseline_vector,
        candidate_vector,
        f"baseline:{baseline_fingerprint}",
        f"candidate:{candidate_fingerprint}",
    )
    baseline_sig = _signature(
        archive_member_id=f"fingerprint:{baseline_fingerprint}",
        trial_id=None,
        vector=baseline_vector,
        rows=baseline_rows,
    )
    candidate_sig = _signature(
        archive_member_id=f"fingerprint:{candidate_fingerprint}",
        trial_id=None,
        vector=candidate_vector,
        rows=candidate_rows,
    )
    report = _paired_report(
        comparison_type="fingerprint_pair",
        baseline_label=baseline_fingerprint,
        candidate_label=candidate_fingerprint,
        stats=stats,
        baseline_signature=baseline_sig,
        candidate_signature=candidate_sig,
        min_shared_qids=min_shared_qids,
        max_accuracy_regression=max_accuracy_regression,
    )
    report["baseline_trials"] = [row["trial_id"] for row in baseline_rows]
    report["candidate_trials"] = [row["trial_id"] for row in candidate_rows]
    return report


def build_eval_result_pair_report(
    baseline_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
    *,
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
    min_shared_qids: int = DEFAULT_MIN_SHARED_QIDS,
    max_accuracy_regression: float = 0.0,
) -> dict[str, Any]:
    baseline_row = _eval_result_row(
        baseline_payload,
        label=baseline_label,
        synthetic_trial_id=0,
    )
    candidate_row = _eval_result_row(
        candidate_payload,
        label=candidate_label,
        synthetic_trial_id=1,
    )
    baseline_vector = _vector_from_eval_result_row(baseline_row, baseline_label)
    candidate_vector = _vector_from_eval_result_row(candidate_row, candidate_label)
    stats = mcnemar_from_vectors(
        baseline_vector,
        candidate_vector,
        f"baseline:{baseline_label}",
        f"candidate:{candidate_label}",
    )
    baseline_sig = _signature(
        archive_member_id=_archive_member_id(baseline_row, fallback=f"eval:{baseline_label}"),
        trial_id=_trial_id_or_none(baseline_row),
        vector=baseline_vector,
        rows=[baseline_row],
    )
    candidate_sig = _signature(
        archive_member_id=_archive_member_id(candidate_row, fallback=f"eval:{candidate_label}"),
        trial_id=_trial_id_or_none(candidate_row),
        vector=candidate_vector,
        rows=[candidate_row],
    )
    return _paired_report(
        comparison_type="eval_result_pair",
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        stats=stats,
        baseline_signature=baseline_sig,
        candidate_signature=candidate_sig,
        min_shared_qids=min_shared_qids,
        max_accuracy_regression=max_accuracy_regression,
    )


def _eval_result_row(
    payload: dict[str, Any],
    *,
    label: str,
    synthetic_trial_id: int,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("eval result payload must be an object")
    if "eval_result" in payload:
        wrapped = payload["eval_result"]
        if not isinstance(wrapped, dict):
            raise ValueError("eval_result wrapper must contain an object")
        row = dict(wrapped)
    else:
        row = dict(payload)
    trial_id = _trial_id_or_none(row)
    row.setdefault("trial_id", trial_id if trial_id is not None else synthetic_trial_id)
    if "eval_details" in row and not isinstance(row["eval_details"], dict):
        raise ValueError("eval result eval_details must be an object")
    if "eval_details" not in row:
        row["eval_details"] = {}
    if "question_results" in row and "question_results" not in row["eval_details"]:
        row["eval_details"] = {
            **row["eval_details"],
            "question_results": row["question_results"],
        }
    if not isinstance(row.get("config_snapshot"), dict):
        row["config_snapshot"] = {}
    row["config_snapshot"].setdefault("config_fingerprint", label)
    return row


def _vector_from_eval_result_row(row: dict[str, Any], label: str) -> dict[str, QuestionOutcome]:
    locations: list[tuple[str, Any]] = []
    if "question_results" in row:
        locations.append(("question_results", row["question_results"]))

    eval_details = row.get("eval_details")
    if "eval_details" in row and not isinstance(eval_details, dict):
        raise ValueError(f"eval result has malformed eval_details: {label}")
    if isinstance(eval_details, dict):
        if "question_results" in eval_details:
            locations.append(("eval_details.question_results", eval_details["question_results"]))
        nested = eval_details.get("details")
        if "details" in eval_details and not isinstance(nested, dict):
            raise ValueError(f"eval result has malformed eval_details.details: {label}")
        if isinstance(nested, dict) and "question_results" in nested:
            locations.append(("eval_details.details.question_results", nested["question_results"]))

    if not locations:
        raise ValueError(f"eval result has no question_results vector: {label}")

    known_dispositions = {
        DISPOSITION_SCORED,
        DISPOSITION_INFRA_FAILED,
        DISPOSITION_SCORING_FAILED,
        DISPOSITION_TASK_FAILED,
    }

    def normalize(location: str, raw: Any) -> dict[str, tuple[str, bool, str]]:
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"{location} must be a non-empty list: {label}")
        normalized: dict[str, tuple[str, bool, str]] = {}
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ValueError(f"{location}[{index}] must be an object: {label}")
            qid = str(item.get("qid") or "").strip()
            if not qid:
                qid = str(item.get("question_id") or "").strip()
            if not qid:
                raise ValueError(f"{location}[{index}] has no usable qid/question_id: {label}")
            if qid in normalized:
                raise ValueError(f"{location} has duplicate normalized question id {qid!r}: {label}")

            if "correct" not in item or type(item["correct"]) is not bool:
                raise ValueError(
                    f"{location}[{index}].correct must be an explicit boolean: {label}"
                )

            raw_disposition = item.get("disposition")
            disposition = (
                DISPOSITION_SCORED
                if raw_disposition is None or raw_disposition == ""
                else raw_disposition
            )
            if not isinstance(disposition, str) or disposition not in known_dispositions:
                raise ValueError(f"{location}[{index}] has unknown disposition: {label}")
            if not is_quality_admissible(disposition):
                raise ValueError(
                    f"{location}[{index}] disposition {disposition!r} has no quality verdict: {label}"
                )
            if disposition == DISPOSITION_TASK_FAILED and item["correct"]:
                raise ValueError(
                    f"{location}[{index}] task_failed outcome cannot be correct: {label}"
                )

            suite = str(item.get("suite") or "").strip()
            normalized[qid] = (suite, item["correct"], disposition)
        return normalized

    vectors = [(name, normalize(name, raw)) for name, raw in locations]
    canonical = vectors[0][1]
    for name, vector in vectors[1:]:
        if vector != canonical:
            raise ValueError(f"conflicting question_results locations ({vectors[0][0]} and {name}): {label}")

    trial_id = _trial_id_or_none(row)
    return {
        qid: QuestionOutcome(qid=qid, suite=values[0], correct=values[1], trial_id=trial_id)
        for qid, values in canonical.items()
    }


def _trial_id_or_none(payload: dict[str, Any]) -> int | None:
    trial_id = payload.get("trial_id")
    try:
        return int(trial_id)
    except (TypeError, ValueError):
        return None


def _archive_member_id(payload: dict[str, Any], *, fallback: str) -> str:
    value = payload.get("archive_member_id") or (payload.get("eval_details") or {}).get(
        "archive_member_id"
    )
    return str(value) if value else fallback


def _paired_report(
    *,
    comparison_type: str,
    baseline_label: str,
    candidate_label: str,
    stats: McNemarResult,
    baseline_signature: dict[str, Any],
    candidate_signature: dict[str, Any],
    min_shared_qids: int,
    max_accuracy_regression: float,
) -> dict[str, Any]:
    severity, reasons = diff_signatures(baseline_signature, candidate_signature)
    decision, blockers = _decision(
        severity=severity,
        stats=stats,
        min_shared_qids=min_shared_qids,
        max_accuracy_regression=max_accuracy_regression,
    )
    return {
        "bsv_paired_report_version": "bsv-2-paired-report-v1",
        "comparison_type": comparison_type,
        "baseline": baseline_label,
        "candidate": candidate_label,
        "gate_decision": decision,
        "blockers": blockers,
        "thresholds": {
            "min_shared_qids": min_shared_qids,
            "max_accuracy_regression": abs(max_accuracy_regression),
        },
        "paired_stats": asdict(stats),
        "signature_diff": {
            "severity": severity,
            "reasons": reasons,
        },
        "baseline_signature": baseline_signature,
        "candidate_signature": candidate_signature,
    }


def render_markdown(report: dict[str, Any]) -> str:
    stats = report["paired_stats"]
    diff = report["signature_diff"]
    lines = [
        "# BSV-2 Paired Behavior Report",
        "",
        f"- Comparison: {report['comparison_type']}",
        f"- Baseline: {report['baseline']}",
        f"- Candidate: {report['candidate']}",
        f"- Gate decision: {report['gate_decision']}",
        f"- Shared qids: {stats['shared_qids']}",
        f"- Candidate accuracy delta: {stats['delta_b_minus_a']:+.6f}",
        f"- Signature severity: {diff['severity']}",
        "",
        "## Blockers",
        "",
    ]
    blockers = list(report.get("blockers") or [])
    lines.extend(f"- {item}" for item in blockers) if blockers else lines.append("- none")
    lines.extend(["", "## Signature Reasons", ""])
    lines.extend(f"- {item}" for item in diff.get("reasons") or [])
    return "\n".join(lines)


def _load_rows(path: Path | str) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_journal_rows(path)]


def _load_json_object(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _cmd_trial_pair(args: argparse.Namespace) -> int:
    report = build_trial_pair_report(
        _load_rows(args.journal),
        baseline_trial=args.baseline_trial,
        candidate_trial=args.candidate_trial,
        min_shared_qids=args.min_shared_qids,
        max_accuracy_regression=args.max_accuracy_regression,
    )
    _print_report(report, markdown=args.markdown)
    return 0 if args.no_fail or report["gate_decision"] == "pass" else 1


def _cmd_fingerprint_pair(args: argparse.Namespace) -> int:
    report = build_fingerprint_pair_report(
        _load_rows(args.journal),
        baseline_fingerprint=args.baseline_fingerprint,
        candidate_fingerprint=args.candidate_fingerprint,
        min_shared_qids=args.min_shared_qids,
        max_accuracy_regression=args.max_accuracy_regression,
    )
    _print_report(report, markdown=args.markdown)
    return 0 if args.no_fail or report["gate_decision"] == "pass" else 1


def _cmd_eval_result_pair(args: argparse.Namespace) -> int:
    baseline_label = args.baseline_label or Path(args.baseline_json).stem
    candidate_label = args.candidate_label or Path(args.candidate_json).stem
    report = build_eval_result_pair_report(
        _load_json_object(args.baseline_json),
        _load_json_object(args.candidate_json),
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        min_shared_qids=args.min_shared_qids,
        max_accuracy_regression=args.max_accuracy_regression,
    )
    _print_report(report, markdown=args.markdown)
    return 0 if args.no_fail or report["gate_decision"] == "pass" else 1


def _print_report(report: dict[str, Any], *, markdown: bool) -> None:
    if markdown:
        print(render_markdown(report))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only BSV-2 paired behavior report")
    parser.add_argument("--journal", default=str(DEFAULT_JOURNAL_DIR), help="journal dir or JSONL file")
    parser.add_argument("--min-shared-qids", type=int, default=DEFAULT_MIN_SHARED_QIDS)
    parser.add_argument("--max-accuracy-regression", type=float, default=0.0)
    parser.add_argument("--markdown", action="store_true", help="render a markdown report")
    parser.add_argument("--no-fail", action="store_true", help="always exit 0 after printing")
    sub = parser.add_subparsers(dest="cmd", required=True)

    trial = sub.add_parser("trial-pair", help="compare two vector-bearing trial ids")
    trial.add_argument("baseline_trial", type=int)
    trial.add_argument("candidate_trial", type=int)
    trial.set_defaults(func=_cmd_trial_pair)

    fp = sub.add_parser("fingerprint-pair", help="compare majority vectors by config fingerprint")
    fp.add_argument("baseline_fingerprint")
    fp.add_argument("candidate_fingerprint")
    fp.set_defaults(func=_cmd_fingerprint_pair)

    er = sub.add_parser(
        "eval-result-pair",
        help="compare two standalone EvalResult-like JSON artifacts",
    )
    er.add_argument("baseline_json")
    er.add_argument("candidate_json")
    er.add_argument("--baseline-label")
    er.add_argument("--candidate-label")
    er.set_defaults(func=_cmd_eval_result_pair)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
