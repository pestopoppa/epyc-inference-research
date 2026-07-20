from pathlib import Path

import pytest

from scripts.benchmark import dr3_quant_asym_k2_admission_runner as runner


def test_parse_args_defaults_to_dry_run_and_fixed_contexts() -> None:
    args = runner.parse_args([])

    assert args.execute is False
    assert args.context_bands == [8192, 16384]
    assert runner.K_VALUE == 2
    assert args.binary == runner.dr0.EXPERIMENTAL_SERVER.resolve()


def test_parse_args_refuses_production_v6_binary() -> None:
    production_binary = (
        Path("/mnt/raid0/llm/llama.cpp") / "build-hip" / "bin" / "llama-server"
    )

    with pytest.raises(ValueError, match="production v6"):
        runner.parse_args(["--binary", str(production_binary)])


def test_dry_run_writes_task_packet_and_non_serving_summary(tmp_path: Path) -> None:
    exit_code = runner.main(
        [
            "--output-dir",
            str(tmp_path),
            "--context-band",
            "8192",
            "--rows-per-class",
            "1",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "task_packet.jsonl").exists()
    manifest = json_load(tmp_path / "manifest.json")
    summary = json_load(tmp_path / "summary.json")
    task_rows = (tmp_path / "task_packet.jsonl").read_text(encoding="utf-8").splitlines()

    assert manifest["live_runner"]["fixed_k"] == 2
    assert manifest["live_runner"]["task_row_count"] == 6
    assert len(task_rows) == 6
    assert all("max_tokens" in row for row in manifest["admission_task_rows"])
    assert summary["mode"] == "dry_run"
    assert summary["serving_route_allowed"] is False
    assert summary["numeric_swarm_surface_allowed"] is False
    assert summary["quality_gate"]["status"] == "not_run"


def test_build_arm_specs_has_only_baseline_and_k2() -> None:
    args = runner.parse_args(["--context-band", "8192", "--context-band", "16384"])
    specs = runner.build_arm_specs(args)

    assert [spec.id for spec in specs] == [
        "cpu_baseline_ctx8192",
        "combined_k2_ctx8192",
        "cpu_baseline_ctx16384",
        "combined_k2_ctx16384",
    ]
    assert {spec.k for spec in specs} == {None, 2}
    assert all("--spec-draft-n-max" not in spec.argv or "2" in spec.argv for spec in specs)


def test_quality_scoring_accepts_materialized_rows() -> None:
    args = runner.parse_args(["--context-band", "8192"])
    rows = runner.materialize_task_rows(args, 8192)
    by_class = {row["class_id"]: row for row in rows}

    structured_content = "\n".join(
        '{"index": %d, "status": "READY", "payload": "dr3-8192-0-%d"}' % (index, index)
        for index in range(8)
    )
    assert runner.score_admission_quality(by_class["structured_json_long"], structured_content)["pass"]
    assert runner.score_admission_quality(
        by_class["strict_formatting"],
        "\n".join(by_class["strict_formatting"]["expected"]["lines"]),
    )["pass"]
    assert runner.score_admission_quality(
        by_class["long_repetitive_output"],
        by_class["long_repetitive_output"]["expected"]["exact_text"],
    )["pass"]
    assert runner.score_admission_quality(
        by_class["long_context_tail"],
        by_class["long_context_tail"]["expected"]["exact_text"],
    )["pass"]


def test_execute_path_writes_observation_summary_with_mocked_arms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run_arm_spec(args, spec, task_rows):
        return {
            "arm": spec.id,
            "base_arm": spec.base_arm_id,
            "status": "ok",
            "fresh_server": True,
            "k": spec.k,
            "context_band": spec.context,
            "prompt_tokens": 10,
            "generated_tokens": 20,
            "draft_tokens": 10 if spec.k else 0,
            "accepted_draft_tokens": 9 if spec.k else 0,
            "alpha": 0.9 if spec.k else None,
            "wall_time_s": 1.0,
            "prompt_time_s": 0.25,
            "decode_time_s": 0.5,
            "prompt_tps": 40.0,
            "decode_tps": 40.0 if not spec.k else 50.0,
            "spec_telemetry_status": "observed" if spec.k else "missing",
            "spec_verify_steps": 2 if spec.k else 0,
            "spec_draft_time_s": 0.001 if spec.k else 0.0,
            "spec_verify_time_s": 0.012 if spec.k else 0.0,
            "spec_process_time_s": 0.002 if spec.k else 0.0,
            "spec_sample_accept_time_s": 0.003 if spec.k else 0.0,
            "spec_accept_by_depth": [1, 1] if spec.k else [],
            "load_wall_clock_s": 0.1,
            "request_count": 1,
            "error_count": 0,
            "quality_results": [
                {
                    "task_class": "mock",
                    "row_id": "row-a",
                    "status": "checked",
                    "pass": True,
                    "checker": "mock",
                    "details": {},
                }
            ],
            "task_results": [
                {
                    "task_class": "mock",
                    "status": "ok",
                    "content_sha256": "same",
                    "quality_pass": True,
                }
            ],
            "admission_task_results": [
                {
                    "row_id": "row-a",
                    "task_class": "mock",
                    "context_band": spec.context,
                    "status": "ok",
                    "content_sha256": "same",
                    "quality_pass": True,
                }
            ],
            "cleanup": {"status": "ok", "terminated": True, "port_open_after": False},
        }

    monkeypatch.setattr(runner, "validate_live_inputs", lambda args: None)
    monkeypatch.setattr(runner, "run_arm_spec", fake_run_arm_spec)
    monkeypatch.setattr(runner.dr0, "process_snapshot", lambda: {"lines": []})
    monkeypatch.setattr(runner.dr0, "rocm_smi_showpids", lambda: {"kfd_pids_observed": False})

    exit_code = runner.main(
        [
            "--execute",
            "--output-dir",
            str(tmp_path),
            "--context-band",
            "8192",
        ]
    )
    summary = json_load(tmp_path / "summary.json")

    assert exit_code == 0
    assert summary["mode"] == "execute"
    assert summary["decision_grade"] is False
    assert summary["observation_grade"] is True
    assert summary["quality_gate"]["status"] == "pass"
    assert summary["output_stability_gate"]["status"] == "pass"
    assert summary["context_coverage_gate"]["status"] == "pass"
    assert summary["cleanup_proof"]["status"] == "pass"
    assert summary["admission_result"]["serving_route_allowed"] is False
    assert summary["speed_economics"]["rows"][0]["decode_tps_ratio_vs_baseline"] == 1.25


def test_output_stability_detects_combined_mismatch() -> None:
    arms = {
        "cpu_baseline_ctx8192": {
            "context_band": 8192,
            "admission_task_results": [
                {"row_id": "row-a", "content_sha256": "baseline", "quality_pass": True}
            ],
        },
        "combined_k2_ctx8192": {
            "context_band": 8192,
            "admission_task_results": [
                {"row_id": "row-a", "content_sha256": "changed", "quality_pass": True}
            ],
        },
    }

    rows = runner.output_stability_rows(arms)

    assert rows == [
        {
            "arm": "combined_k2_ctx8192",
            "context_band": 8192,
            "baseline_arm": "cpu_baseline_ctx8192",
            "target_output_match_vs_baseline": {"row-a": False},
            "details": {
                "row-a": {
                    "equivalence_rule": "exact_hash_when_seeded",
                    "baseline_quality_pass": True,
                    "combined_quality_pass": True,
                    "content_hash_match": False,
                }
            },
            "pass": False,
        }
    ]


def test_output_stability_uses_semantic_rule_from_manifest() -> None:
    manifest = {
        "admission_task_rows": [
            {
                "row_id": "row-a",
                "equivalence_rule": "semantic_equivalence_plus_verdict_match",
            }
        ]
    }
    arms = {
        "cpu_baseline_ctx8192": {
            "context_band": 8192,
            "admission_task_results": [
                {"row_id": "row-a", "content_sha256": "baseline", "quality_pass": True}
            ],
        },
        "combined_k2_ctx8192": {
            "context_band": 8192,
            "admission_task_results": [
                {"row_id": "row-a", "content_sha256": "changed", "quality_pass": True}
            ],
        },
    }

    rows = runner.output_stability_rows(arms, manifest)

    assert rows[0]["target_output_match_vs_baseline"] == {"row-a": True}
    assert rows[0]["details"]["row-a"]["content_hash_match"] is False
    assert rows[0]["pass"] is True


def json_load(path: Path):
    import json

    return json.loads(path.read_text(encoding="utf-8"))
