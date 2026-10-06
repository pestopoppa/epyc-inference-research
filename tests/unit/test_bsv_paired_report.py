from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.autopilot import bsv_paired_report


def _row(
    trial_id: int,
    fingerprint: str,
    outcomes: list[tuple[str, bool]],
    *,
    avg_prompt_tokens: int = 1000,
    routing_distribution: dict[str, float] | None = None,
) -> dict:
    return {
        "trial_id": trial_id,
        "config_snapshot": {"config_fingerprint": fingerprint},
        "avg_prompt_tokens": avg_prompt_tokens,
        "routing_distribution": routing_distribution or {"frontdoor": 1.0},
        "eval_details": {
            "question_results": [
                {"qid": qid, "suite": "suite", "correct": correct}
                for qid, correct in outcomes
            ]
        },
    }


def test_trial_pair_blocks_on_prior_pass_regression() -> None:
    rows = [
        _row(1, "base", [("q1", True), ("q2", True), ("q3", False)]),
        _row(2, "cand", [("q1", True), ("q2", False), ("q3", False)]),
    ]

    report = bsv_paired_report.build_trial_pair_report(
        rows,
        baseline_trial=1,
        candidate_trial=2,
        min_shared_qids=3,
    )

    assert report["gate_decision"] == "block"
    assert report["paired_stats"]["shared_qids"] == 3
    assert report["signature_diff"]["severity"] == "blocking"
    assert "behavior signature severity is blocking" in report["blockers"]
    assert report["baseline_signature"]["archive_member_id"] == "trial:1"
    assert report["candidate_signature"]["archive_member_id"] == "trial:2"


def test_trial_pair_passes_watch_when_no_regression_and_enough_coverage() -> None:
    rows = [
        _row(1, "base", [("q1", True), ("q2", False), ("q3", False)]),
        _row(
            2,
            "cand",
            [("q1", True), ("q2", True), ("q3", False)],
            routing_distribution={"frontdoor": 0.1, "worker": 0.9},
        ),
    ]

    report = bsv_paired_report.build_trial_pair_report(
        rows,
        baseline_trial=1,
        candidate_trial=2,
        min_shared_qids=3,
    )

    assert report["gate_decision"] == "pass"
    assert report["signature_diff"]["severity"] == "watch"
    assert report["paired_stats"]["delta_b_minus_a"] > 0
    assert report["blockers"] == []


def test_fingerprint_pair_uses_majority_vectors_and_records_trials() -> None:
    rows = [
        _row(1, "base", [("q1", True), ("q2", False), ("q3", False)]),
        _row(2, "base", [("q1", True), ("q2", False), ("q3", True)]),
        _row(3, "cand", [("q1", True), ("q2", True), ("q3", True)]),
        _row(4, "cand", [("q1", True), ("q2", True), ("q3", False)]),
    ]

    report = bsv_paired_report.build_fingerprint_pair_report(
        rows,
        baseline_fingerprint="base",
        candidate_fingerprint="cand",
        min_shared_qids=2,
    )

    assert report["comparison_type"] == "fingerprint_pair"
    assert report["gate_decision"] == "pass"
    assert report["paired_stats"]["shared_qids"] == 2
    assert report["baseline_trials"] == [1, 2]
    assert report["candidate_trials"] == [3, 4]


def test_eval_result_pair_compares_standalone_payloads() -> None:
    baseline = {
        "archive_member_id": "archive:base",
        "trial_id": 101,
        "avg_prompt_tokens": 900,
        "routing_distribution": {"frontdoor": 1.0},
        "question_results": [
            {"qid": "q1", "suite": "suite", "correct": True},
            {"qid": "q2", "suite": "suite", "correct": True},
            {"qid": "q3", "suite": "suite", "correct": False},
        ],
    }
    candidate = {
        "archive_member_id": "archive:cand",
        "trial_id": 102,
        "avg_prompt_tokens": 900,
        "routing_distribution": {"frontdoor": 1.0},
        "question_results": [
            {"qid": "q1", "suite": "suite", "correct": True},
            {"qid": "q2", "suite": "suite", "correct": False},
            {"qid": "q3", "suite": "suite", "correct": False},
        ],
    }

    report = bsv_paired_report.build_eval_result_pair_report(
        baseline,
        candidate,
        baseline_label="base-json",
        candidate_label="cand-json",
        min_shared_qids=3,
    )

    assert report["comparison_type"] == "eval_result_pair"
    assert report["gate_decision"] == "block"
    assert report["paired_stats"]["trial_a"] == "baseline:base-json"
    assert report["paired_stats"]["trial_b"] == "candidate:cand-json"
    assert report["baseline_signature"]["archive_member_id"] == "archive:base"
    assert report["candidate_signature"]["trial_id"] == 102


def test_eval_result_pair_uses_stable_qid_when_source_question_id_differs() -> None:
    baseline = {"question_results": [
        {"qid": "stable:one", "question_id": "dataset-17", "suite": "suite", "correct": True},
        {"qid": "stable:two", "question_id": "dataset-18", "suite": "suite", "correct": False},
    ]}
    candidate = {"eval_result": {"eval_details": {"details": {"question_results": [
        {"question_id": "stable:one", "suite": "suite", "correct": True},
        {"question_id": "stable:two", "suite": "suite", "correct": True},
    ]}}}}

    report = bsv_paired_report.build_eval_result_pair_report(
        baseline, candidate, min_shared_qids=2
    )

    assert report["paired_stats"]["shared_qids"] == 2
    assert report["paired_stats"]["delta_b_minus_a"] == 0.5


@pytest.mark.parametrize(
    "outcome",
    [
        {"qid": "q1"},
        {"qid": "q1", "correct": None},
        {"qid": "q1", "correct": "false"},
        {"qid": "q1", "correct": 0},
        None,
        "not-an-outcome",
    ],
)
def test_eval_result_pair_refuses_malformed_boolean_rows(outcome) -> None:  # noqa: ANN001
    valid = {"question_results": [{"qid": "q1", "correct": True}]}
    malformed = {"question_results": [outcome]}

    with pytest.raises(ValueError):
        bsv_paired_report.build_eval_result_pair_report(valid, malformed)


@pytest.mark.parametrize(
    "outcomes",
    [
        [{"correct": True}],
        [{"qid": "   ", "correct": True}],
        [
            {"qid": " q1 ", "correct": True},
            {"question_id": "q1", "correct": False},
        ],
    ],
)
def test_eval_result_pair_refuses_missing_or_duplicate_question_ids(outcomes) -> None:  # noqa: ANN001
    valid = {"question_results": [{"qid": "q1", "correct": True}]}
    malformed = {"question_results": outcomes}

    with pytest.raises(ValueError):
        bsv_paired_report.build_eval_result_pair_report(valid, malformed)


def test_eval_result_pair_requires_consistent_question_result_locations() -> None:
    valid = {"question_results": [{"qid": "q1", "correct": True}]}
    inconsistent = {
        "question_results": [{"qid": "q1", "correct": True}],
        "eval_details": {"question_results": [{"qid": "q1", "correct": False}]},
    }

    with pytest.raises(ValueError, match="conflicting question_results locations"):
        bsv_paired_report.build_eval_result_pair_report(valid, inconsistent)


@pytest.mark.parametrize("disposition", ["infra_failed", "scoring_failed", "unrecognized"])
def test_eval_result_pair_refuses_nonquality_or_unknown_dispositions(disposition: str) -> None:
    valid = {"question_results": [{"qid": "q1", "correct": True}]}
    nonquality = {
        "question_results": [
            {"qid": "q1", "correct": False, "disposition": disposition}
        ]
    }

    with pytest.raises(ValueError):
        bsv_paired_report.build_eval_result_pair_report(valid, nonquality)


@pytest.mark.parametrize("legacy_disposition", [None, ""])
def test_eval_result_pair_accepts_explicit_task_failure_and_legacy_scored_rows(
    legacy_disposition: str | None,
) -> None:
    baseline = {
        "question_results": [
            {"qid": "q1", "correct": False, "disposition": "task_failed"}
        ]
    }
    candidate = {
        "question_results": [
            {"qid": "q1", "correct": True, "disposition": legacy_disposition}
        ]
    }

    report = bsv_paired_report.build_eval_result_pair_report(baseline, candidate, min_shared_qids=1)

    assert report["paired_stats"]["shared_qids"] == 1
    assert report["paired_stats"]["delta_b_minus_a"] == 1.0


def test_eval_result_pair_refuses_task_failure_marked_correct() -> None:
    invalid = {
        "question_results": [
            {"qid": "q1", "correct": True, "disposition": "task_failed"}
        ]
    }
    valid = {"question_results": [{"qid": "q1", "correct": True}]}

    with pytest.raises(ValueError, match="task_failed outcome cannot be correct"):
        bsv_paired_report.build_eval_result_pair_report(invalid, valid)


def test_eval_result_pair_accepts_identical_question_result_aliases() -> None:
    outcomes = [
        {"qid": "stable:one", "question_id": "dataset-17", "correct": True},
        {"qid": "stable:two", "question_id": "dataset-18", "correct": False},
    ]
    payload = {
        "question_results": outcomes,
        "eval_details": {"details": {"question_results": outcomes}},
    }

    report = bsv_paired_report.build_eval_result_pair_report(
        payload,
        {"question_results": [
            {"qid": "stable:one", "correct": True},
            {"qid": "stable:two", "correct": False},
        ]},
        min_shared_qids=2,
    )

    assert report["paired_stats"]["shared_qids"] == 2


def test_eval_result_pair_cli_refusal_emits_no_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps({"question_results": [{"qid": "q1", "correct": True}]}))
    candidate.write_text(json.dumps({"question_results": [{"qid": "q1", "correct": None}]}))

    with pytest.raises(ValueError, match="explicit boolean"):
        bsv_paired_report.main([
            "eval-result-pair",
            str(baseline),
            str(candidate),
            "--min-shared-qids",
            "1",
        ])

    assert capsys.readouterr().out == ""


def test_blocks_when_shared_qid_coverage_is_too_low() -> None:
    rows = [
        _row(1, "base", [("q1", True)]),
        _row(2, "cand", [("q1", True)]),
    ]

    report = bsv_paired_report.build_trial_pair_report(
        rows,
        baseline_trial=1,
        candidate_trial=2,
        min_shared_qids=2,
    )

    assert report["gate_decision"] == "block"
    assert report["blockers"] == ["shared_qids 1 < 2"]


def test_cli_trial_pair_returns_nonzero_on_block(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    journal = tmp_path / "autopilot_journal.jsonl"
    rows = [
        _row(1, "base", [("q1", True), ("q2", True)]),
        _row(2, "cand", [("q1", True), ("q2", False)]),
    ]
    journal.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    code = bsv_paired_report.main([
        "--journal",
        str(journal),
        "--min-shared-qids",
        "2",
        "trial-pair",
        "1",
        "2",
    ])

    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["gate_decision"] == "block"


def test_cli_markdown_no_fail(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    journal = tmp_path / "autopilot_journal.jsonl"
    rows = [
        _row(1, "base", [("q1", True)]),
        _row(2, "cand", [("q1", True)]),
    ]
    journal.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    code = bsv_paired_report.main([
        "--journal",
        str(journal),
        "--min-shared-qids",
        "2",
        "--markdown",
        "--no-fail",
        "trial-pair",
        "1",
        "2",
    ])

    assert code == 0
    assert "# BSV-2 Paired Behavior Report" in capsys.readouterr().out


def test_cli_eval_result_pair_reads_json_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps({
        "eval_details": {
            "question_results": [
                {"qid": "q1", "suite": "suite", "correct": True},
                {"qid": "q2", "suite": "suite", "correct": False},
            ]
        }
    }))
    candidate.write_text(json.dumps({
        "eval_result": {
            "eval_details": {
                "details": {
                    "question_results": [
                        {"question_id": "q1", "suite": "suite", "correct": True},
                        {"question_id": "q2", "suite": "suite", "correct": True},
                    ]
                }
            }
        }
    }))

    code = bsv_paired_report.main([
        "--min-shared-qids",
        "2",
        "eval-result-pair",
        str(baseline),
        str(candidate),
        "--baseline-label",
        "base-file",
        "--candidate-label",
        "cand-file",
    ])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["comparison_type"] == "eval_result_pair"
    assert payload["gate_decision"] == "pass"
    assert payload["paired_stats"]["shared_qids"] == 2
