#!/usr/bin/env python3
"""Offline, report-only count of persisted tool-use outcomes.

This reader consumes EvalTower's question-result JSONL sidecars. It does not
score answers, import AutoPilot, or make an admission/gating decision.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

TOOL_USE_SUITE = "tool_use"
_KNOWN_DISPOSITIONS = {"scored", "task_failed", "infra_failed", "scoring_failed"}
_NON_QUALITY_DISPOSITIONS = {"infra_failed", "scoring_failed"}


class _DuplicateJSONKeyError(ValueError):
    pass


class _NonFiniteJSONConstantError(ValueError):
    pass


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKeyError(f"duplicate object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise _NonFiniteJSONConstantError(f"non-finite JSON number: {value}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise _NonFiniteJSONConstantError(f"non-finite JSON number: {value}")
    return parsed


def _normalized_path(value: str | Path) -> Path:
    path = Path(value)
    try:
        return path.resolve()
    except (OSError, RuntimeError):
        return path.absolute()


def _batch_key(row: Mapping[str, Any]) -> tuple[str, str] | None:
    batch_id = row.get("eval_batch_id")
    label = row.get("label")
    if not isinstance(batch_id, str) or not batch_id.strip():
        return None
    if not isinstance(label, str) or not label.strip():
        return None
    return batch_id, label


def _new_batch(path: Path, batch_id: str, label: str) -> dict[str, Any]:
    return {
        "source": str(path),
        "eval_batch_id": batch_id,
        "label": label,
        "start_markers": 0,
        "complete_markers": 0,
        "completed_n": None,
        "question_result_rows": 0,
        "tool_use_rows": 0,
        "other_suite_rows": 0,
        "malformed_result_rows": 0,
        "known_attempt_rows": 0,
        "known_no_attempt_rows": 0,
        "unknown_trace_rows": 0,
        "malformed_trace_rows": 0,
        "tool_count_trace_disagreements": 0,
        "scored_tool_use_rows": 0,
        "known_attempt_scored_rows": 0,
        "tool_invoked_scored_wrong_rows": 0,
        "attempt_unscored_rows": 0,
        "missing_correct_rows": 0,
        "malformed_correct_rows": 0,
        "unscored_status_rows": 0,
        "malformed_score_state_rows": 0,
        "implicit_scored_status_rows": 0,
        "task_failed_quality_zero_rows": 0,
        "task_failed_correctness_overridden_rows": 0,
    }


def _is_nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _scoreable_result(result: Mapping[str, Any], counts: dict[str, Any]) -> bool:
    """Apply the persisted EvalTower compact-result scoring contract.

    QuestionResult defaults to disposition ``scored``. The compact writer
    omits that default and emits non-default dispositions. EvalTower's current
    denominator includes ``task_failed`` as a quality zero, excludes
    ``infra_failed``/``scoring_failed``, and excludes an error row whose status
    is omitted/default scored.
    """
    correct = result.get("correct")
    if "correct" not in result:
        counts["missing_correct_rows"] += 1
        return False
    if type(correct) is not bool:
        counts["malformed_correct_rows"] += 1
        return False

    error_present = "error" in result
    error = result.get("error")
    if error_present and type(error) is not bool:
        counts["malformed_score_state_rows"] += 1
        return False

    if "disposition" not in result:
        if error is True:
            counts["unscored_status_rows"] += 1
            return False
        counts["implicit_scored_status_rows"] += 1
        return True
    disposition = result["disposition"]
    if not isinstance(disposition, str) or disposition not in _KNOWN_DISPOSITIONS:
        counts["malformed_score_state_rows"] += 1
        return False
    if disposition == "task_failed":
        # The producer always persists error:true for task_failed. Require
        # that structural companion rather than inferring it from a message.
        if error is not True:
            counts["malformed_score_state_rows"] += 1
            return False
        counts["task_failed_quality_zero_rows"] += 1
        if correct is True:
            counts["task_failed_correctness_overridden_rows"] += 1
        return True
    if disposition in _NON_QUALITY_DISPOSITIONS:
        counts["unscored_status_rows"] += 1
        return False
    if error is True:
        counts["malformed_score_state_rows"] += 1
        return False
    return True


def _record_question_result(row: Mapping[str, Any], counts: dict[str, Any]) -> None:
    counts["question_result_rows"] += 1
    result = row.get("result")
    if not isinstance(result, Mapping):
        counts["malformed_result_rows"] += 1
        return
    suite = result.get("suite")
    if suite != TOOL_USE_SUITE:
        counts["other_suite_rows"] += 1
        return
    counts["tool_use_rows"] += 1

    trace_known = False
    attempted = False
    if "tools_called" not in result:
        counts["unknown_trace_rows"] += 1
    else:
        calls = result["tools_called"]
        if not isinstance(calls, list) or any(
            not isinstance(name, str) or not name.strip() for name in calls
        ):
            counts["malformed_trace_rows"] += 1
        else:
            trace_known = True
            attempted = bool(calls)
            if attempted:
                counts["known_attempt_rows"] += 1
            else:
                counts["known_no_attempt_rows"] += 1

    if "tools_used" in result:
        tools_used = result["tools_used"]
        if not _is_nonnegative_int(tools_used):
            counts["malformed_trace_rows"] += 1
        elif trace_known and (tools_used < 1 if attempted else tools_used != 0):
            counts["tool_count_trace_disagreements"] += 1
        elif not trace_known and tools_used > 0:
            # A scalar call count does not supply the missing named action trace.
            counts["tool_count_trace_disagreements"] += 1

    scoreable = _scoreable_result(result, counts)
    if scoreable:
        counts["scored_tool_use_rows"] += 1
    if attempted:
        if scoreable:
            counts["known_attempt_scored_rows"] += 1
            if result.get("disposition") == "task_failed" or result["correct"] is False:
                counts["tool_invoked_scored_wrong_rows"] += 1
        else:
            counts["attempt_unscored_rows"] += 1


def _batch_completeness(
    counts: Mapping[str, Any], file_meta: Mapping[str, Any]
) -> str:
    if any(
        file_meta.get(key, 0)
        for key in (
            "malformed_json_lines",
            "non_object_json_lines",
            "unidentified_rows",
            "read_errors",
            "decode_errors",
        )
    ):
        return "partial_or_ambiguous_file_data_issue"
    if counts["complete_markers"] != 1:
        return "partial_or_ambiguous"
    completed_n = counts["completed_n"]
    if not _is_nonnegative_int(completed_n):
        return "partial_or_ambiguous"
    if completed_n != counts["question_result_rows"]:
        return "partial_or_ambiguous"
    if any(
        counts[key]
        for key in (
            "malformed_result_rows",
            "malformed_trace_rows",
            "missing_correct_rows",
            "malformed_correct_rows",
            "malformed_score_state_rows",
        )
    ):
        return "complete_marker_and_row_count_match_with_attributable_data_issues"
    return "complete_marker_and_row_count_match"


def summarize_paths(paths: Iterable[Path]) -> dict[str, Any]:
    """Summarize sidecars without importing or executing project code."""
    batches: dict[tuple[str, str, str], dict[str, Any]] = {}
    file_meta: dict[str, dict[str, Any]] = {}
    files_read: list[str] = []

    for raw_path in paths:
        path = _normalized_path(raw_path)
        path_text = str(path)
        if path_text in files_read:
            continue
        files_read.append(path_text)
        metadata = file_meta.setdefault(
            path_text,
            {
                "malformed_json_lines": 0,
                "malformed_json_reasons": {},
                "non_object_json_lines": 0,
                "unidentified_rows": 0,
                "read_errors": 0,
                "decode_errors": 0,
                "data_errors": [],
            },
        )
        try:
            handle = path.open("r", encoding="utf-8")
        except OSError as exc:
            metadata["read_errors"] += 1
            metadata["data_errors"].append(
                f"read_error: {type(exc).__name__}: {exc}"
            )
            continue
        try:
            with handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        row = json.loads(
                            stripped,
                            object_pairs_hook=_object_without_duplicate_keys,
                            parse_constant=_reject_nonfinite_constant,
                            parse_float=_parse_finite_float,
                        )
                    except _DuplicateJSONKeyError:
                        metadata["malformed_json_lines"] += 1
                        reasons = metadata["malformed_json_reasons"]
                        reasons["duplicate_key"] = reasons.get("duplicate_key", 0) + 1
                        continue
                    except _NonFiniteJSONConstantError:
                        metadata["malformed_json_lines"] += 1
                        reasons = metadata["malformed_json_reasons"]
                        reasons["nonfinite_constant"] = reasons.get("nonfinite_constant", 0) + 1
                        continue
                    except (json.JSONDecodeError, RecursionError, ValueError):
                        metadata["malformed_json_lines"] += 1
                        reasons = metadata["malformed_json_reasons"]
                        reasons["invalid_json"] = reasons.get("invalid_json", 0) + 1
                        continue
                    if not isinstance(row, dict):
                        metadata["non_object_json_lines"] += 1
                        continue
                    row_type = row.get("row_type")
                    if row_type not in {
                        "batch_start",
                        "batch_complete",
                        "question_result",
                    }:
                        continue
                    identity = _batch_key(row)
                    if identity is None:
                        metadata["unidentified_rows"] += 1
                        continue
                    batch_id, label = identity
                    key = (path_text, batch_id, label)
                    counts = batches.setdefault(key, _new_batch(path, batch_id, label))
                    if row_type == "batch_start":
                        counts["start_markers"] += 1
                    elif row_type == "batch_complete":
                        if row.get("complete") is True:
                            counts["complete_markers"] += 1
                            if _is_nonnegative_int(row.get("completed_n")):
                                counts["completed_n"] = row["completed_n"]
                            else:
                                counts["completed_n"] = None
                    else:
                        _record_question_result(row, counts)
        except UnicodeDecodeError as exc:
            metadata["decode_errors"] += 1
            metadata["data_errors"].append(
                f"decode_error: UnicodeDecodeError: {exc}"
            )
        except OSError as exc:
            metadata["read_errors"] += 1
            metadata["data_errors"].append(
                f"read_error: {type(exc).__name__}: {exc}"
            )

    batch_rows: list[dict[str, Any]] = []
    for key in sorted(batches):
        counts = batches[key]
        counts["completeness"] = _batch_completeness(
            counts, file_meta.get(counts["source"], {})
        )
        denominator = counts["known_attempt_scored_rows"]
        counts["wrong_fraction"] = (
            counts["tool_invoked_scored_wrong_rows"] / denominator
            if denominator
            else None
        )
        batch_rows.append(counts)
    return {
        "report": "tool_use_outcome_divergence",
        "authority": "report_only",
        "interpretation": (
            "A quality-zero outcome with a persisted tool call is descriptive co-occurrence; "
            "it does not show the tool caused the error. Missing traces and "
            "platform/scorer-excluded rows are outside the attempted-and-scored denominator. "
            "Marker/count agreement is only "
            "an artifact-shape "
            "diagnostic, not a correctness, scoring-validity, or representativeness assertion."
        ),
        "files": sorted(files_read),
        "file_data_quality": dict(sorted(file_meta.items())),
        "batches": batch_rows,
        "tool_use_rows": sum(row["tool_use_rows"] for row in batch_rows),
        "known_attempt_scored_rows": sum(
            row["known_attempt_scored_rows"] for row in batch_rows
        ),
        "tool_invoked_scored_wrong_rows": sum(
            row["tool_invoked_scored_wrong_rows"] for row in batch_rows
        ),
        "wrong_fraction": (
            sum(row["tool_invoked_scored_wrong_rows"] for row in batch_rows)
            / sum(row["known_attempt_scored_rows"] for row in batch_rows)
            if sum(row["known_attempt_scored_rows"] for row in batch_rows)
            else None
        ),
    }


def expand_inputs(inputs: Iterable[str]) -> list[Path]:
    """Expand file/directory arguments to sorted sidecar paths."""
    paths: list[Path] = []
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            paths.extend(sorted(path.rglob("question_results*.jsonl")))
        else:
            paths.append(path)
    unique: dict[str, Path] = {}
    for path in paths:
        resolved = _normalized_path(path)
        unique.setdefault(str(resolved), resolved)
    return list(unique.values())


def _markdown_cell(value: Any) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def format_markdown(report: Mapping[str, Any]) -> str:
    batches = report["batches"]
    file_quality = report["file_data_quality"].values()
    unknown_traces = sum(row["unknown_trace_rows"] for row in batches)
    malformed_traces = sum(row["malformed_trace_rows"] for row in batches)
    missing_correct = sum(row["missing_correct_rows"] for row in batches)
    malformed_correct = sum(row["malformed_correct_rows"] for row in batches)
    malformed_score = sum(row["malformed_score_state_rows"] for row in batches)
    attempts_unscored = sum(row["attempt_unscored_rows"] for row in batches)
    task_failed_zeros = sum(row["task_failed_quality_zero_rows"] for row in batches)
    task_failed_overrides = sum(
        row["task_failed_correctness_overridden_rows"] for row in batches
    )
    malformed_json = sum(
        int(row.get("malformed_json_lines", 0)) for row in file_quality
    )
    non_object_json = sum(
        int(row.get("non_object_json_lines", 0)) for row in file_quality
    )
    read_errors = sum(
        int(row.get("read_errors", 0)) + int(row.get("decode_errors", 0))
        for row in file_quality
    )
    unidentified_rows = sum(
        int(row.get("unidentified_rows", 0)) for row in file_quality
    )
    lines = [
        "# Tool-use outcome divergence (report only)",
        "",
        "A quality-zero outcome after a recorded tool call is descriptive co-occurrence; "
        "it does not attribute cause.",
        "",
        f"Observed `tool_use` rows: {report['tool_use_rows']}",
        f"Attempted and quality-scoreable denominator: {report['known_attempt_scored_rows']}",
        f"Tool-invoked quality-zero (scored-wrong): {report['tool_invoked_scored_wrong_rows']}",
        "Quality-zero fraction of attempted-and-scoreable rows: "
        + (
            f"{report['wrong_fraction']:.4f}"
            if report["wrong_fraction"] is not None
            else "NA (denominator 0)"
        ),
        f"Unknown traces: {unknown_traces}; malformed traces: {malformed_traces}; "
        f"missing/malformed correctness: {missing_correct}/{malformed_correct}; "
        f"malformed score state: {malformed_score}; attempted but unscored: {attempts_unscored}; "
        f"task_failed quality-zero rows: {task_failed_zeros} "
        f"(native-correctness overrides: {task_failed_overrides}).",
        f"Input data issues: malformed JSON lines: {malformed_json}; non-object JSON rows: "
        f"{non_object_json}; unidentified rows: {unidentified_rows}; "
        f"read/decode errors: {read_errors}.",
        "",
        "| Source | Batch | Arm | Completeness | Tool-use rows | Known attempts | "
        "Quality-scoreable | "
        "Attempted and scoreable | Quality-zero | Unknown trace | Malformed | Attempt unscored |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["batches"]:
        display_row = dict(row)
        display_row.update(
            source=_markdown_cell(row["source"]),
            batch=_markdown_cell(row["eval_batch_id"]),
            label=_markdown_cell(row["label"]),
            completeness=_markdown_cell(row["completeness"]),
        )
        lines.append(
            "| {source} | {batch} | {label} | {completeness} | {tool_use_rows} | "
            "{known_attempt_rows} | {scored_tool_use_rows} | {known_attempt_scored_rows} | "
            "{tool_invoked_scored_wrong_rows} | {unknown_trace_rows} | "
            "{malformed_trace_rows} | {attempt_unscored_rows} |".format(**display_row)
        )
    if not report["batches"]:
        lines.append(
            "| *(no sidecar batches)* | — | — | — | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |"
        )
    lines.extend(
        [
            "",
            "Missing `tools_called` is unknown; an explicit empty list is a known no-call. "
            "No rows or a zero denominator means no estimate, not zero divergence.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        help="question_results*.jsonl files or directories to scan recursively",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of Markdown")
    args = parser.parse_args(argv)
    report = summarize_paths(expand_inputs(args.paths))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(format_markdown(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
