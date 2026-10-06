"""Synthetic contracts for the offline tool-use outcome report."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.autopilot.tool_use_divergence_report import (
    expand_inputs,
    format_markdown,
    main,
    summarize_paths,
)


def _row(
    row_type: str,
    result: dict | None = None,
    *,
    batch_id: str = "batch-1",
    label: str = "candidate",
    **fields: object,
) -> dict:
    row: dict = {
        "schema_version": 1,
        "row_type": row_type,
        "eval_batch_id": batch_id,
        "label": label,
        "requested_n": 1,
        "artifact_root_source": "window_output_dir",
        "recovery_contract": "complete_marker_required",
        "trial_id": 42,
        **fields,
    }
    if result is not None:
        row["result"] = result
    if row_type == "batch_start":
        row.update(concurrency=1, complete=False)
    elif row_type == "question_result":
        row.update(
            ordinal=0,
            answer="synthetic answer",
            complete=False,
            ended_at_s=1.0,
            elapsed_s=0.25,
            started_at_s=0.75,
        )
    elif row_type == "batch_complete":
        row.update(elapsed_s=0.25)
    return row


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_report_counts_only_known_attempted_scoreable_rows(tmp_path: Path) -> None:
    path = tmp_path / "question_results.candidate.jsonl"
    rows = [
        _row("batch_start"),
        _row(
            "question_result",
            {"suite": "tool_use", "tools_called": ["search"], "tools_used": 1, "correct": False},
        ),
        _row(
            "question_result",
            {"suite": "tool_use", "tools_called": ["read"], "tools_used": 1, "correct": True},
        ),
        # The producer omits empty tools_called. A positive scalar count cannot
        # fill in the missing named trace, so this row stays unknown.
        _row("question_result", {"suite": "tool_use", "tools_used": 2, "correct": False}),
        # EvalTower keeps task_failed in the quality denominator as a zero.
        _row(
            "question_result",
            {
                "suite": "tool_use",
                "tools_called": ["search"],
                "correct": False,
                "error": True,
                "disposition": "task_failed",
            },
        ),
        # EvalTower's quality policy forces task_failed to zero even if a
        # recovered/stale native correctness bit says true.
        _row(
            "question_result",
            {
                "suite": "tool_use",
                "tools_called": ["search"],
                "correct": True,
                "error": True,
                "disposition": "task_failed",
            },
        ),
        _row(
            "question_result",
            {"suite": "tool_use", "tools_called": [], "tools_used": 0, "correct": True},
        ),
        # The real writer's completed_n is len(out), with one wrapper per
        # returned result; it is not a tool_use-only count.
        _row("batch_complete", complete=True, completed_n=6),
    ]
    for row in rows:
        row["requested_n"] = 6
    _write(path, rows)

    report = summarize_paths([path])
    batch = report["batches"][0]
    assert report["tool_use_rows"] == 6
    assert batch["known_attempt_rows"] == 4
    assert batch["known_no_attempt_rows"] == 1
    assert batch["unknown_trace_rows"] == 1
    assert batch["known_attempt_scored_rows"] == 4
    assert batch["tool_invoked_scored_wrong_rows"] == 3
    assert batch["task_failed_quality_zero_rows"] == 2
    assert batch["task_failed_correctness_overridden_rows"] == 1
    assert batch["attempt_unscored_rows"] == 0
    assert batch["wrong_fraction"] == 0.75
    assert batch["completeness"] == "complete_marker_and_row_count_match"
    # Rendering exercises the CLI's Markdown path, including rows with metadata.
    assert "Tool-invoked quality-zero (scored-wrong): 3" in format_markdown(report)


def test_no_known_attempted_scored_rows_has_no_rate(tmp_path: Path) -> None:
    path = tmp_path / "question_results.jsonl"
    _write(
        path,
        [
            _row("question_result", {"suite": "tool_use", "correct": False}),
            _row(
                "question_result",
                {"suite": "tool_use", "tools_called": ["x"], "correct": False, "error": True},
            ),
            _row(
                "question_result",
                {
                    "suite": "tool_use",
                    "tools_called": ["x"],
                    "correct": False,
                    "error": True,
                    "disposition": "infra_failed",
                },
            ),
            _row(
                "question_result",
                {
                    "suite": "tool_use",
                    "tools_called": ["x"],
                    "correct": False,
                    "error": True,
                    "disposition": "scoring_failed",
                },
            ),
        ],
    )

    report = summarize_paths([path])
    assert report["known_attempt_scored_rows"] == 0
    assert report["tool_invoked_scored_wrong_rows"] == 0
    assert report["wrong_fraction"] is None
    assert report["batches"][0]["wrong_fraction"] is None
    assert report["batches"][0]["unscored_status_rows"] == 3
    assert report["batches"][0]["attempt_unscored_rows"] == 3


def test_malformed_rows_traces_and_scores_remain_visible(tmp_path: Path) -> None:
    path = tmp_path / "question_results.jsonl"
    _write(
        path,
        [
            _row("batch_start"),
            _row(
                "question_result",
                {"suite": "tool_use", "tools_called": "search", "correct": False},
            ),
            _row(
                "question_result",
                {"suite": "tool_use", "tools_called": ["search"], "correct": "false"},
            ),
            _row(
                "question_result",
                {
                    "suite": "tool_use",
                    "tools_called": ["search"],
                    "correct": False,
                    "disposition": "future-status",
                },
            ),
            _row("question_result", {"suite": "other", "correct": False}),
            _row("question_result", None),
            _row("batch_complete", complete=True, completed_n=5),
        ],
    )

    batch = summarize_paths([path])["batches"][0]
    assert batch["tool_use_rows"] == 3
    assert batch["malformed_trace_rows"] == 1
    assert batch["malformed_correct_rows"] == 1
    assert batch["malformed_score_state_rows"] == 1
    assert batch["other_suite_rows"] == 1
    assert batch["malformed_result_rows"] == 1
    assert batch["known_attempt_scored_rows"] == 0
    assert batch["completeness"] == (
        "complete_marker_and_row_count_match_with_attributable_data_issues"
    )


def test_batch_completion_requires_one_matching_complete_marker(tmp_path: Path) -> None:
    path = tmp_path / "question_results.jsonl"
    _write(
        path,
        [
            _row("question_result", {"suite": "tool_use", "tools_called": ["x"], "correct": True}),
            _row("batch_complete", complete=True, completed_n=0),
            _row("batch_complete", complete=True, completed_n=1),
        ],
    )

    assert summarize_paths([path])["batches"][0]["completeness"] == "partial_or_ambiguous"


def test_empty_input_and_known_no_call_do_not_create_zero_divergence_claim(tmp_path: Path) -> None:
    empty = summarize_paths([])
    assert empty["files"] == []
    assert empty["batches"] == []
    assert empty["wrong_fraction"] is None
    assert "no estimate" in format_markdown(empty)

    path = tmp_path / "question_results.known-none.jsonl"
    _write(
        path,
        [
            _row("batch_start"),
            _row(
                "question_result",
                {"suite": "tool_use", "tools_called": [], "tools_used": 0, "correct": False},
            ),
            _row(
                "question_result",
                {"suite": "tool_use", "tools_called": [], "tools_used": 2, "correct": False},
            ),
            _row("batch_complete", complete=True, completed_n=2),
        ],
    )
    batch = summarize_paths([path])["batches"][0]
    assert batch["known_no_attempt_rows"] == 2
    assert batch["tool_count_trace_disagreements"] == 1
    assert batch["known_attempt_scored_rows"] == 0
    assert batch["wrong_fraction"] is None


def test_cli_accepts_no_paths(capsys) -> None:  # noqa: ANN001
    assert main(["--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["files"] == []
    assert output["wrong_fraction"] is None


def test_arms_and_batches_remain_separate_and_paths_are_deduplicated(tmp_path: Path) -> None:
    path = tmp_path / "question_results.mixed.jsonl"
    rows = []
    for batch_id, label in (
        ("batch-1", "candidate"),
        ("batch-1", "control"),
        ("batch-2", "candidate"),
    ):
        rows.extend(
            [
                _row("batch_start", batch_id=batch_id, label=label),
                _row(
                    "question_result",
                    {
                        "suite": "tool_use",
                        "tools_called": ["search"],
                        "tools_used": 1,
                        "correct": False,
                    },
                    batch_id=batch_id,
                    label=label,
                ),
                _row(
                    "batch_complete",
                    batch_id=batch_id,
                    label=label,
                    complete=True,
                    completed_n=1,
                ),
            ]
        )
    _write(path, rows)

    report = summarize_paths([path, path])
    assert report["files"] == [str(path.resolve())]
    assert len(report["batches"]) == 3
    assert {(row["eval_batch_id"], row["label"]) for row in report["batches"]} == {
        ("batch-1", "candidate"),
        ("batch-1", "control"),
        ("batch-2", "candidate"),
    }
    assert report["tool_invoked_scored_wrong_rows"] == 3
    assert expand_inputs([str(path), str(path)]) == [path.resolve()]


def test_duplicate_keys_nonfinite_unidentified_and_nonobject_rows_taint_completion(
    tmp_path: Path,
) -> None:
    path = tmp_path / "question_results.bad-lines.jsonl"
    valid = _row(
        "question_result",
        {"suite": "tool_use", "tools_called": ["search"], "tools_used": 1, "correct": False},
    )
    rows = [
        json.dumps(_row("batch_start")),
        json.dumps(valid),
        '{"row_type":"question_result","row_type":"batch_complete"}',
        '{"row_type":"question_result","result":{"suite":"tool_use","latency_ms":NaN}}',
        '{"row_type":"question_result","result":{"suite":"tool_use","latency_ms":1e999}}',
        "[]",
        json.dumps(
            {
                "row_type": "batch_complete",
                "complete": True,
                "completed_n": 1,
                "eval_batch_id": "batch-1",
            }
        ),  # label missing: attribution is unsafe
        json.dumps(_row("batch_complete", complete=True, completed_n=1)),
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    report = summarize_paths([path])
    batch = report["batches"][0]
    file_issue = report["file_data_quality"][str(path.resolve())]
    assert file_issue["malformed_json_lines"] == 3
    assert file_issue["malformed_json_reasons"] == {"duplicate_key": 1, "nonfinite_constant": 2}
    assert file_issue["non_object_json_lines"] == 1
    assert file_issue["unidentified_rows"] == 1
    assert batch["completeness"] == "partial_or_ambiguous_file_data_issue"


def test_unreadable_and_invalid_utf8_inputs_are_reported(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"
    bad_utf8 = tmp_path / "question_results.invalid-utf8.jsonl"
    good_prefix = (
        json.dumps(_row("batch_start"))
        + "\n"
        + json.dumps(_row("batch_complete", complete=True, completed_n=0))
        + "\n"
    ).encode("utf-8")
    bad_utf8.write_bytes(good_prefix + b"\xff")

    report = summarize_paths([missing, bad_utf8])
    assert len(report["files"]) == 2
    assert report["file_data_quality"][str(missing.resolve())]["read_errors"] == 1
    assert report["file_data_quality"][str(bad_utf8.resolve())]["decode_errors"] == 1
    assert all(
        batch["completeness"] == "partial_or_ambiguous_file_data_issue"
        for batch in report["batches"]
    )
    assert "read_error" in report["file_data_quality"][str(missing.resolve())]["data_errors"][0]


def test_unidentified_batch_keys_are_not_collapsed_and_mark_file_partial(
    tmp_path: Path,
) -> None:
    path = tmp_path / "question_results.unidentified.jsonl"
    _write(
        path,
        [
            _row("batch_start"),
            _row("question_result", {"suite": "tool_use", "tools_called": ["x"], "correct": False}),
            _row("batch_complete", complete=True, completed_n=1),
            {"row_type": "batch_start", "label": "candidate"},
            {
                "row_type": "question_result",
                "eval_batch_id": "batch-1",
                "result": {"suite": "tool_use", "correct": False},
            },
        ],
    )

    report = summarize_paths([path])
    assert len(report["batches"]) == 1
    assert report["file_data_quality"][str(path.resolve())]["unidentified_rows"] == 2
    assert report["batches"][0]["completeness"] == "partial_or_ambiguous_file_data_issue"


def test_markdown_cells_escape_pipes_and_newlines(tmp_path: Path) -> None:
    path = tmp_path / "question_results.escape.jsonl"
    _write(
        path,
        [
            _row("batch_start", label="candidate|\ncontrol"),
            _row(
                "question_result",
                {"suite": "tool_use", "tools_called": ["x"], "correct": False},
                label="candidate|\ncontrol",
            ),
            _row("batch_complete", label="candidate|\ncontrol", complete=True, completed_n=1),
        ],
    )
    markdown = format_markdown(summarize_paths([path]))
    assert "candidate\\| control" in markdown
    assert "candidate|\ncontrol" not in markdown
