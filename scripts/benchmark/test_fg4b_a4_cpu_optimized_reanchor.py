from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import fg4b_a4_cpu_optimized_reanchor as runner


def test_command_is_production_shaped_and_rejects_legacy_bench_shape() -> None:
    command = runner.build_server_command()
    assert command[:3] == ["taskset", "-c", runner.CPU_LIST]
    assert "numactl" not in command
    assert "llama-bench" not in " ".join(command)
    server_index = command.index(str(runner.LLAMA_SERVER))
    for option, value in (("-c", "32768"), ("-ub", "8192"), ("-ctk", "q8_0"), ("-ctv", "q8_0"), ("--spec-type", "draft-mtp"), ("--spec-draft-n-max", "4")):
        index = command.index(option, server_index)
        assert command[index + 1] == value
    assert "--reasoning" in command and command[command.index("--reasoning") + 1] == "off"


def test_dry_run_never_spawns_and_has_no_registry_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "start_server", lambda *_: pytest.fail("dry run spawned"))
    payload = runner.dry_run_payload(runner.parse_args([]))
    assert payload["registry_mutation"] is False
    assert payload["required_regions"] == ["q0", "q1"]
    assert payload["metric"] == "timings.predicted_per_second"


def test_region_check_rejects_q2_only_claim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "_region_status", lambda: [
        {"region": "q0", "global_held": False}, {"region": "q1", "global_held": False},
        {"region": "q2", "global_held": True}, {"region": "q3", "global_held": False},
    ])
    with pytest.raises(runner.ReanchorRefusal, match="q2-only"):
        runner.verify_held_footprint()


def test_region_check_accepts_actual_footprint(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"region": name, "global_held": name in {"q0", "q1"}} for name in ("q0", "q1", "q2", "q3")]
    monkeypatch.setattr(runner, "_region_status", lambda: rows)
    assert runner.verify_held_footprint() == rows


def test_long_decode_rejects_short_or_missing_timing() -> None:
    response = {"timings": {"predicted_n": 32, "predicted_per_second": 12.0}}
    with pytest.raises(runner.ReanchorRefusal, match="ended at 32"):
        runner.parse_sample(response, 1)
    response = {"timings": {"predicted_n": 512, "predicted_per_second": 0.0}}
    with pytest.raises(runner.ReanchorRefusal, match="no positive"):
        runner.parse_sample(response, 1)


def test_execute_refuses_without_explicit_window_grant() -> None:
    args = runner.parse_args(["--execute"])
    with pytest.raises(runner.ReanchorRefusal, match="i-have-operator-grant"):
        runner.execute(args)


def test_long_decode_accepts_server_timing() -> None:
    response = {
        "timings": {"predicted_n": 512, "predicted_per_second": 42.5, "prompt_n": 18},
        "choices": [{"message": {"content": "x" * 10}}],
    }
    sample = runner.parse_sample(response, 1)
    assert sample.predicted_per_second == 42.5
    assert sample.predicted_n == 512


def test_proposal_is_evidence_bound_and_non_applying() -> None:
    evidence = {"mean_tokens_per_second": 42.0, "runtime_identity": {"llama_commit": runner.EXPECTED_LLAMA_COMMIT}}
    result = runner.proposal(evidence)
    assert result["mode"] == "proposal_only"
    assert result["must_not_apply_automatically"] is True
    assert "llama-bench tg512" in result["not_comparable_to"]
    assert len(result["evidence_sha256"]) == 64
