#!/usr/bin/env python3
"""No-inference tests for glm52_external_ground_truth_direct_runner.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

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
            {"row_id": "r2", "response_text": '{"decision":"A","confidence":0.6}'},
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
        ]
    )

    assert rc == 0
    decisions = [(json.loads(line)["candidate_id"], json.loads(line)["correct"]) for line in (tmp_path / "out" / "decisions.jsonl").read_text().splitlines()]
    assert decisions == [("r1", True), ("r2", False)]
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["score"]["accuracy"] == 0.5
    assert summary["run_manifest"]["n_scored"] == 2
    assert (tmp_path / "out" / "run_manifest.json").exists()
