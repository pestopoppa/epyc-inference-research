#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import e2_eval_driver_ab as e2


@dataclass(frozen=True)
class _Prompt:
    qid: str
    suite: str = "math"
    tier: int = 1
    prompt: str = "Solve 1+1."


def _fake_prompts(path, *, limit, seed, tier, suites):
    return [_Prompt(qid=f"q{i:02d}", tier=tier) for i in range(limit)]


def test_manifest_blocks_commands_when_host_health_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(e2, "load_prompt_batch", _fake_prompts)
    monkeypatch.setattr(e2, "collect_attestation", lambda: {"host": "test-host"})
    monkeypatch.setattr(e2, "host_health_warnings", lambda attestation: ["uptime exceeds policy"])

    args = e2.parse_args(
        [
            "--run-id",
            "e2-test",
            "--output-root",
            str(tmp_path),
            "--prompt-limit",
            "3",
        ]
    )
    output_dir = e2.write_outputs(args)

    manifest = json.loads((output_dir / "manifest.json").read_text())
    commands = (output_dir / "commands.sh").read_text()

    assert manifest["status"] == "blocked"
    assert manifest["decision_grade"] is False
    assert manifest["prompt_batch"]["qids"] == ["q00", "q01", "q02"]
    assert "# blocked: host-health preconditions failed" in commands
    assert "#   uv run --extra benchmark python scripts/benchmark/server_np_sweep.py" in commands
    assert "\n  uv run --extra benchmark python scripts/benchmark/server_np_sweep.py" not in commands


def test_allow_host_health_warning_emits_scout_commands(monkeypatch, tmp_path):
    monkeypatch.setattr(e2, "load_prompt_batch", _fake_prompts)
    monkeypatch.setattr(e2, "collect_attestation", lambda: {"host": "test-host"})
    monkeypatch.setattr(e2, "host_health_warnings", lambda attestation: ["existing llama processes present"])

    args = e2.parse_args(
        [
            "--run-id",
            "e2-scout",
            "--output-root",
            str(tmp_path),
            "--prompt-limit",
            "2",
            "--allow-host-health-warning",
            "--scout-skip-clean-check",
        ]
    )
    output_dir = e2.write_outputs(args)

    manifest = json.loads((output_dir / "manifest.json").read_text())
    commands = (output_dir / "commands.sh").read_text()

    assert manifest["status"] == "runnable"
    assert manifest["decision_grade"] is False
    assert manifest["allow_host_health_warning"] is True
    assert "--allow-host-health-warning --skip-clean-check" in commands
    assert "AUTOPILOT_EVAL_CONCURRENCY=3" in commands
    assert "current_quarters.jsonl" in commands


def test_clean_manifest_records_two_e2_arms(monkeypatch, tmp_path):
    monkeypatch.setattr(e2, "load_prompt_batch", _fake_prompts)
    monkeypatch.setattr(e2, "collect_attestation", lambda: {"host": "test-host"})
    monkeypatch.setattr(e2, "host_health_warnings", lambda attestation: [])

    args = e2.parse_args(
        [
            "--run-id",
            "e2-clean",
            "--output-root",
            str(tmp_path),
            "--prompt-limit",
            "4",
            "--batch-np",
            "8",
            "--current-concurrency",
            "3",
        ]
    )
    output_dir = e2.write_outputs(args)
    manifest = json.loads((output_dir / "manifest.json").read_text())

    assert manifest["status"] == "runnable"
    assert manifest["decision_grade"] is True
    assert [arm["name"] for arm in manifest["arms"]] == [
        "batch_np8_single_full_instance",
        "current_three_concurrent_quarters",
    ]
    assert manifest["arms"][0]["kind"] == "server_np_sweep"
    assert manifest["arms"][1]["kind"] == "core_v2_calibrate"
    assert manifest["comparison"]["metric"] == "wall_minutes_per_eval"


def test_current_arm_command_uses_absolute_research_artifact_path(monkeypatch, tmp_path):
    monkeypatch.setattr(e2, "load_prompt_batch", _fake_prompts)
    monkeypatch.setattr(e2, "collect_attestation", lambda: {"host": "test-host"})
    monkeypatch.setattr(e2, "host_health_warnings", lambda attestation: [])
    monkeypatch.chdir(tmp_path)

    args = e2.parse_args(
        [
            "--run-id",
            "e2-paths",
            "--output-root",
            "relative-output",
            "--prompt-limit",
            "1",
        ]
    )
    args.research_root = tmp_path / "research"
    args.orchestrator_root = tmp_path / "orchestrator"
    output_dir = e2.write_outputs(args)

    commands = (output_dir / "commands.sh").read_text()

    expected = (
        args.research_root
        / "relative-output"
        / "e2-paths"
        / "current_quarters.jsonl"
    )
    assert f"--out-jsonl {expected}" in commands


def _write_completed_e2_run(tmp_path: Path, *, decision_grade: bool = True) -> Path:
    run_dir = tmp_path / "e2-complete"
    batch_dir = run_dir / "serving" / "e2-complete-batch-np8"
    batch_dir.mkdir(parents=True)
    manifest = {
        "run_id": "e2-complete",
        "status": "runnable",
        "decision_grade": decision_grade,
        "arms": [
            {
                "kind": "server_np_sweep",
                "primary_artifacts": [str(batch_dir / "summary.csv")],
            },
            {
                "kind": "core_v2_calibrate",
                "primary_artifacts": [str(run_dir / "current_quarters.jsonl")],
            },
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    # Two candidate cells whose orderings DISAGREE: np=4 wins on tasks/hour
    # (900 > 516), np=8 wins on aggregate decode tok/s (200 > 100). Operator
    # ruling 2026-07-30 makes tok/s the selection metric, so np=8 must be
    # picked — and only np=8 yields the 5.0 wall-minutes/eval asserted below.
    # Ranking on the demoted metric would pick np=4 and fail loudly instead of
    # short-circuiting on a single-candidate fixture.
    (batch_dir / "summary.csv").write_text(
        "\n".join(
            [
                "model,np,success_count,total_count,error_rate,wall_seconds,"
                "aggregate_decode_tps,per_stream_decode_tps,tasks_per_hour,p95_latency_ms",
                "qwen36_q8_0,8,43,43,0.0,300.0,200.0,25.0,516.0,12000.0",
                "qwen36_q8_0,4,43,43,0.0,600.0,100.0,25.0,900.0,9000.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "current_quarters.jsonl").write_text(
        json.dumps(
            {
                "eval_wall_s": 600.0,
                "n_questions": 43,
                "eval_concurrency": 3,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir


def test_summarize_run_marks_fast_decision_grade_batch_keep_candidate(tmp_path):
    run_dir = _write_completed_e2_run(tmp_path)

    summary = e2.summarize_run(run_dir)

    assert summary["status"] == "keep_candidate"
    assert summary["decision_grade"] is True
    # batch row selected on tok/s, not tasks/hour (operator ruling 2026-07-30)
    assert summary["batch_arm"]["np"] == 8
    assert summary["batch_arm"]["aggregate_decode_tps"] == 200.0
    assert summary["batch_arm"]["per_stream_decode_tps"] == 25.0
    assert summary["batch_arm"]["tasks_per_hour"] == 516.0  # secondary, kept
    assert summary["batch_arm"]["wall_minutes_per_eval"] == 5.0
    assert summary["current_arm"]["wall_minutes_per_eval"] == 10.0
    assert summary["comparison"]["speedup_current_over_batch"] == 2.0


def test_summarize_run_keeps_scout_data_out_of_decisions(tmp_path):
    run_dir = _write_completed_e2_run(tmp_path, decision_grade=False)

    summary = e2.summarize_run(run_dir)

    assert summary["status"] == "scout_only"
    assert summary["decision_grade"] is False
    assert "not decision-grade" in summary["recommendation"]["reasons"][0]


def test_summarize_run_reports_missing_artifacts(tmp_path):
    run_dir = tmp_path / "e2-missing"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "e2-missing", "decision_grade": True}),
        encoding="utf-8",
    )

    summary = e2.summarize_run(run_dir)

    assert summary["status"] == "incomplete"
    assert summary["decision_grade"] is False
    assert any("missing batch summary" in reason for reason in summary["recommendation"]["reasons"])
    assert any("missing current-arm JSONL" in reason for reason in summary["recommendation"]["reasons"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
