#!/usr/bin/env python3
"""No-inference tests for glm52_external_ground_truth_direct_runner.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parent / "glm52_external_ground_truth_direct_runner.py"
_SPEC = importlib.util.spec_from_file_location("glm52_external_ground_truth_direct_runner", _MODULE_PATH)
runner = importlib.util.module_from_spec(_SPEC)
sys.modules["glm52_external_ground_truth_direct_runner"] = runner
_SPEC.loader.exec_module(runner)


def _row(row_id: str, gold: str = "A") -> dict:
    return {
        "row_id": row_id,
        "task": "Pick the better answer.",
        "candidate": "Answer A",
        "candidate_b": "Answer B",
        "gold_label": gold,
        "gold_source": "judgebench",
        "gold_instrument_version": "local-file-sha256:test",
        "source_benchmark": "judgebench",
        "source_suite": "gpt",
    }


def _swe_row(row_id: str, gold: str = "accept") -> dict:
    return {
        "row_id": row_id,
        "task_kind": "patch_review_oracle",
        "task": "Fix the parser so lower-case commands are accepted.",
        "candidate": "diff --git a/parser.py b/parser.py\n@@\n- if token == 'READ':\n+ if token.upper() == 'READ':\n",
        "gold_label": gold,
        "gold_source": "swe-bench-verified",
        "gold_instrument_version": "file-sha256:test",
        "instance_id": "repo__project-1",
        "repo": "repo/project",
        "FAIL_TO_PASS": ["tests/test_parser.py::test_lowercase_read"],
        "PASS_TO_PASS": ["tests/test_parser.py::test_uppercase_read"],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_build_plan_refuses_missing_rows(tmp_path):
    rows_path = tmp_path / "rows.jsonl"
    _write_jsonl(rows_path, [])

    rc = runner.main(["--rows-jsonl", str(rows_path), "--output-dir", str(tmp_path / "out")])

    assert rc == 3
    plan = json.loads((tmp_path / "out" / "plan.json").read_text())
    assert plan["execution_allowed"] is False
    assert "no rows" in plan["refusal_reasons"]


def test_dry_run_writes_plan_without_responses(tmp_path):
    rows_path = tmp_path / "rows.jsonl"
    _write_jsonl(rows_path, [_row("r1", "A"), _row("r2", "B")])

    rc = runner.main(["--rows-jsonl", str(rows_path), "--output-dir", str(tmp_path / "out")])

    assert rc == 0
    plan = json.loads((tmp_path / "out" / "plan.json").read_text())
    assert plan["schema"] == runner.SCHEMA
    assert plan["rows"]["gold_label_counts"] == {"A": 1, "B": 1}
    assert plan["execution_allowed"] is True
    assert not (tmp_path / "out" / "decisions.jsonl").exists()


def test_score_saved_responses_writes_decisions_summary_and_manifest(tmp_path):
    rows_path = tmp_path / "rows.jsonl"
    responses_path = tmp_path / "responses.jsonl"
    _write_jsonl(rows_path, [_row("r1", "A"), _row("r2", "B")])
    _write_jsonl(
        responses_path,
        [
            {"row_id": "r1", "response_text": '{"decision":"A","confidence":0.8}'},
            {"row_id": "r2", "response_text": '{"decision":"A","confidence":85}'},
        ],
    )

    rc = runner.main(
        [
            "--rows-jsonl",
            str(rows_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--score-responses-jsonl",
            str(responses_path),
            "--measurement-protocol",
            "p_rev1",
            "--protocol-attestation",
            "TEST-ATTESTATION",
        ]
    )

    assert rc == 0
    decisions = [(json.loads(line)["candidate_id"], json.loads(line)["correct"]) for line in (tmp_path / "out" / "decisions.jsonl").read_text().splitlines()]
    assert decisions == [("r1", True), ("r2", False)]
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["score"]["accuracy"] == 0.5
    assert summary["score"]["confidence_warning_counts"] == {"confidence_scale_0_100": 1}
    assert summary["run_manifest"]["n_scored"] == 2
    assert summary["run_manifest"]["observation_only"] is False
    assert summary["run_manifest"]["protocol_attestation"] == "TEST-ATTESTATION"
    assert summary["server"]["not_started"] is True
    assert summary["server"]["log_file"] is None
    assert (tmp_path / "out" / "run_manifest.json").exists()


def test_parse_args_defaults_p_rev1_to_attested_era(tmp_path):
    rows_path = tmp_path / "rows.jsonl"
    _write_jsonl(rows_path, [_row("r1", "A")])

    args = runner.parse_args(["--rows-jsonl", str(rows_path), "--measurement-protocol", "p_rev1"])

    assert args.era == runner.P_REV1_ERA


def test_loads_swe_patch_review_oracle_without_candidate_b(tmp_path):
    rows_path = tmp_path / "rows.jsonl"
    _write_jsonl(rows_path, [_swe_row("swe-1")])

    rows = runner.load_rows(rows_path)

    assert rows[0]["row_id"] == "swe-1"
    assert runner.row_task_kind(rows[0]) == runner.ROW_KIND_PATCH_REVIEW_ORACLE


def test_swe_patch_review_uses_review_schema_and_oracle_prompt():
    row = _swe_row("swe-1")

    prompt_info = runner.fit_prompt_to_budget_for_row(
        row,
        context_length=4096,
        max_completion_tokens=128,
        prompt_context_guard_tokens=64,
        max_field_chars=1200,
    )
    schema = runner.response_schema_for_rows([row])

    assert "SWE-BENCH-VERIFIED TEST ORACLE" in prompt_info["prompt"]
    assert "FAIL_TO_PASS tests:" in prompt_info["prompt"]
    assert "PASS_TO_PASS tests:" in prompt_info["prompt"]
    assert prompt_info["truncation"]["review_mode"] == runner.ROW_KIND_PATCH_REVIEW_ORACLE
    assert schema["properties"]["decision"]["enum"] == ["approve", "reject"]
    assert "blocking" in schema["required"]
    assert "evidence" in schema["required"]


def test_score_saved_swe_responses_reports_false_reject_false_accept_and_parse_failure(tmp_path):
    rows_path = tmp_path / "rows.jsonl"
    responses_path = tmp_path / "responses.jsonl"
    _write_jsonl(rows_path, [_swe_row("a1", "accept"), _swe_row("r1", "reject"), _swe_row("a2", "accept")])
    _write_jsonl(
        responses_path,
        [
            {
                "row_id": "a1",
                "response_text": '{"decision":"reject","confidence":0.7,"blocking":{"tripwire":true},"evidence":{"basis":"misses parser path","risk":"lowercase still rejected"}}',
            },
            {
                "row_id": "r1",
                "response_text": '{"decision":"approve","confidence":0.8,"blocking":{"tripwire":false},"evidence":{"basis":"patch covers parser","risk":"no blocker found"}}',
            },
            {"row_id": "a2", "response_text": "not json"},
        ],
    )

    rc = runner.main(
        [
            "--rows-jsonl",
            str(rows_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--score-responses-jsonl",
            str(responses_path),
            "--measurement-protocol",
            "p_rev1",
            "--protocol-attestation",
            "TEST-ATTESTATION",
        ]
    )

    assert rc == 0
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["request"]["rubric_version"] == runner.DEFAULT_PATCH_REVIEW_RUBRIC_VERSION
    assert summary["score"]["accept_control_n"] == 2
    assert summary["score"]["hard_negative_n"] == 1
    assert summary["score"]["false_rejects"] == 1
    assert summary["score"]["false_accepts"] == 1
    assert summary["score"]["parse_failures"] == 1
    decisions = [json.loads(line) for line in (tmp_path / "out" / "decisions.jsonl").read_text().splitlines()]
    assert {row["domain"] for row in decisions} == {"patch_review"}
    assert decisions[0]["false_reject"] is True
    assert decisions[1]["false_accept"] is True


def test_mixed_pairwise_and_patch_rows_are_refused(tmp_path):
    rows_path = tmp_path / "rows.jsonl"
    _write_jsonl(rows_path, [_row("r1"), _swe_row("swe-1")])

    with pytest.raises(ValueError, match="mixed row task kinds"):
        runner.load_rows(rows_path)
