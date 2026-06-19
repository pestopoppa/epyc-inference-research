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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
