import json
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.benchmark import dr3_frontdoor_opportunity_cost_gate as runner


def json_load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_dry_run_writes_plan_commands_and_non_serving_summary(tmp_path: Path) -> None:
    exit_code = runner.main(["--output-dir", str(tmp_path), "--context", "8192"])

    assert exit_code == 0
    plan = json_load(tmp_path / "plan.json")
    summary = json_load(tmp_path / "summary.json")
    commands = (tmp_path / "commands.sh").read_text(encoding="utf-8")

    assert set(plan["arms"]) == {
        "frontdoor_alone_before_eviction",
        "dr3_lane_active",
        "frontdoor_after_eviction_reload",
    }
    assert plan["fixed_k"] == 2
    assert plan["arms"]["dr3_lane_active"]["task_class"] == "long_repetitive_output"
    assert "--spec-type draft-mtp" in commands
    assert "--spec-draft-n-max 2" in commands
    assert "frontdoor alone before DR-3 lease" in commands
    assert summary["mode"] == "dry_run"
    assert summary["frontdoor_opportunity_cost_gate"]["status"] == "not_run"
    assert summary["decision_grade"] is False
    assert summary["observation_grade"] is False
    assert summary["serving_route_allowed"] is False
    assert summary["numeric_swarm_surface_allowed"] is False


def test_validate_inputs_refuses_production_v6_binary(tmp_path: Path) -> None:
    args = runner.parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--binary",
            "/mnt/raid0/llm/llama.cpp/build-hip/bin/llama-server",
        ]
    )

    with pytest.raises(ValueError, match="production v6"):
        runner.validate_inputs(args)


def test_build_summary_marks_passing_execute_gate_observation_only(tmp_path: Path) -> None:
    args = Namespace(execute=True, output_dir=tmp_path)
    plan = {"arms": {}}
    results = {
        "frontdoor_alone_before_eviction": {
            "status": "ok",
            "load_wall_clock_s": 2.0,
            "cleanup": {"status": "ok"},
            "results": [
                {
                    "decode_tps": 100.0,
                    "prompt_tps": 500.0,
                    "completion_tokens": 256,
                    "passed_min_completion": True,
                }
            ],
        },
        "dr3_lane_active": {
            "status": "ok",
            "load_wall_clock_s": 3.0,
            "cleanup": {"status": "ok"},
            "aggregate": {
                "decode_tps": 11.0,
                "prompt_tps": 120.0,
                "alpha": 0.88,
                "draft_tokens": 100,
                "accepted_draft_tokens": 88,
            },
        },
        "frontdoor_after_eviction_reload": {
            "status": "ok",
            "load_wall_clock_s": 2.5,
            "cleanup": {"status": "ok"},
            "results": [
                {
                    "decode_tps": 95.0,
                    "prompt_tps": 510.0,
                    "completion_tokens": 256,
                    "passed_min_completion": True,
                }
            ],
        },
    }

    summary = runner.build_summary(
        args,
        plan,
        pre_process={"lines": []},
        post_process={"lines": []},
        pre_rocm={"kfd_pids_observed": False},
        post_rocm={"kfd_pids_observed": False},
        results=results,
    )

    gate = summary["frontdoor_opportunity_cost_gate"]
    assert summary["status"] == "pass"
    assert summary["observation_grade"] is True
    assert summary["decision_grade"] is False
    assert summary["serving_route_allowed"] is False
    assert summary["numeric_swarm_surface_allowed"] is False
    assert gate["status"] == "pass"
    assert gate["after_vs_before_decode_ratio"] == pytest.approx(0.95)
    assert gate["dr3_lane_active"]["alpha"] == pytest.approx(0.88)
    assert summary["cleanup_proof"]["status"] == "pass"
    assert summary["p_gpu_1_gate"]["serving_blocker"] is True


def test_build_summary_fails_when_frontdoor_reload_fails(tmp_path: Path) -> None:
    args = Namespace(execute=True, output_dir=tmp_path)
    summary = runner.build_summary(
        args,
        {"arms": {}},
        pre_process={"lines": []},
        post_process={"lines": []},
        pre_rocm={"kfd_pids_observed": False},
        post_rocm={"kfd_pids_observed": False},
        results={
            "frontdoor_alone_before_eviction": {
                "status": "ok",
                "results": [{"decode_tps": 100.0, "passed_min_completion": True}],
                "cleanup": {"status": "ok"},
            },
            "dr3_lane_active": {"status": "ok", "aggregate": {}, "cleanup": {"status": "ok"}},
            "frontdoor_after_eviction_reload": {
                "status": "quality_fail",
                "results": [{"decode_tps": 0.0, "passed_min_completion": False}],
                "cleanup": {"status": "ok"},
            },
        },
    )

    assert summary["status"] == "fail"
    assert summary["observation_grade"] is False
    assert summary["frontdoor_opportunity_cost_gate"]["status"] == "fail"
    assert summary["frontdoor_opportunity_cost_gate"]["serving_blocker"] is True
