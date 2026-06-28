from __future__ import annotations

import json
from pathlib import Path

from scripts.halo import closed_loop_observation_surface as halo


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_convert_records_builds_deterministic_loop_chain(tmp_path):
    source = tmp_path / "mixed.jsonl"
    _write_jsonl(
        source,
        [
            {
                "record_type": "benchmark",
                "run_id": "run-17",
                "question_id": "q-1",
                "suite": "math",
                "model": "demo",
                "outcome": "PASS",
                "completion_tokens": 32,
                "tokens_per_second": 12.5,
                "prompt": "solve 1+1",
                "response": "<answer>2</answer>",
            },
            {
                "record_type": "log",
                "run_id": "run-17",
                "level": "INFO",
                "logger": "halo",
                "message": "run started",
            },
            {
                "record_type": "report",
                "run_id": "run-17",
                "report_id": "report-17",
                "status": "complete",
                "verdict": "keep",
                "summary": "finished cleanly",
            },
        ],
    )

    records = halo.load_sources([source])
    observations = halo.convert_records(records)

    assert [obs["kind"] for obs in observations] == ["benchmark", "log", "report"]
    assert [obs["sequence"] for obs in observations] == [0, 1, 2]
    assert observations[0]["parent_span_id"] is None
    assert observations[1]["parent_span_id"] == observations[0]["span_id"]
    assert observations[2]["parent_span_id"] == observations[1]["span_id"]
    assert observations[0]["attributes"]["question_id"] == "q-1"
    assert observations[1]["attributes"]["message"] == "run started"
    assert observations[2]["attributes"]["verdict"] == "keep"
    assert observations[0]["loop_key"] == observations[1]["loop_key"] == observations[2]["loop_key"]


def test_analyze_observations_reports_closed_loop_summary():
    observations = [
        {
            "kind": "benchmark",
            "loop_key": "benchmark|run_id=\"r1\"",
            "source": {"path": "a.jsonl", "line": 1},
            "attributes": {"outcome": "PASS"},
        },
        {
            "kind": "log",
            "loop_key": "benchmark|run_id=\"r1\"",
            "source": {"path": "a.jsonl", "line": 2},
            "attributes": {"level": "INFO"},
        },
        {
            "kind": "report",
            "loop_key": "benchmark|run_id=\"r1\"",
            "source": {"path": "a.jsonl", "line": 3},
            "attributes": {"verdict": "keep"},
        },
        {
            "kind": "log",
            "loop_key": "log|run_id=\"r2\"",
            "source": {"path": "b.jsonl", "line": 1},
            "attributes": {"level": "WARN"},
        },
    ]

    summary = halo.analyze_observations(observations)

    assert summary["total_observations"] == 4
    assert summary["kind_counts"] == {"benchmark": 1, "log": 2, "report": 1}
    assert summary["benchmark_outcomes"] == {"PASS": 1}
    assert summary["log_levels"] == {"INFO": 1, "WARN": 1}
    assert summary["report_verdicts"] == {"keep": 1}
    assert summary["loop_cardinality"] == {
        "total": 2,
        "closed": 1,
        "max_observations": 3,
        "min_observations": 1,
    }
    assert summary["closed_loops"] == ['benchmark|run_id="r1"']


def test_cli_convert_and_analyze_round_trip(tmp_path, capsys):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_jsonl(
        input_dir / "records.jsonl",
        [
            {"record_type": "benchmark", "run_id": "run-1", "question_id": "q-1", "outcome": "FAIL"},
            {"record_type": "log", "run_id": "run-1", "level": "ERROR", "message": "boom"},
            {"record_type": "report", "run_id": "run-1", "verdict": "reject", "summary": "needs work"},
        ],
    )

    converted = tmp_path / "observations.jsonl"
    assert halo.main(["convert", str(input_dir), "--output", str(converted)]) == 0
    assert converted.exists()

    assert halo.main(["analyze", str(converted)]) == 0
    rendered = capsys.readouterr().out.strip()
    payload = json.loads(rendered)

    assert payload["schema"] == halo.SCHEMA_NAME
    assert payload["total_observations"] == 3
    assert payload["loop_cardinality"]["closed"] == 1
