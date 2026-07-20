from pathlib import Path

import pytest

from scripts.benchmark import dr0_quant_asym_self_spec_runner as runner


def test_parse_args_defaults_to_dry_run_and_experimental_binary() -> None:
    args = runner.parse_args([])

    assert args.execute is False
    assert args.binary == runner.EXPERIMENTAL_SERVER.resolve()
    assert args.cpu_verifier_model == runner.DEFAULT_CPU_VERIFIER_MODEL
    assert "Qwen3.5-122B-A10B-UD-Q4_K_M" in str(args.cpu_verifier_model)
    assert args.k == runner.DEFAULT_K_VALUES


def test_parse_args_refuses_production_v6_binary() -> None:
    production_binary = (
        Path("/mnt/raid0/llm/llama.cpp") / "build-hip" / "bin" / "llama-server"
    )

    with pytest.raises(ValueError, match="production v6"):
        runner.parse_args(["--binary", str(production_binary)])


def test_summary_schema_contains_required_accounting_markers(tmp_path: Path) -> None:
    args = runner.parse_args(["--output-dir", str(tmp_path), "--k", "2", "--k", "4"])
    manifest = runner.build_manifest(args)
    summary = runner.build_summary_skeleton(args, manifest)
    expected_variants = {variant.id for variant in runner.execution_variants(args)}

    assert summary["schema"] == "epyc.dr0_quant_asym_self_spec.summary.v1"
    assert set(summary["arms"]) == expected_variants
    assert summary["fh_accounting"]["k_values"] == [2, 4]
    assert summary["fh_accounting"]["F_K"]["status"] == runner.FH_NOT_OBSERVABLE
    assert summary["fh_accounting"]["H_K"]["status"] == runner.FH_NOT_OBSERVABLE
    for arm_summary in summary["arms"].values():
        assert "draft_tokens" in arm_summary
        assert "accepted_draft_tokens" in arm_summary
        assert "alpha" in arm_summary
        assert len(arm_summary["quality_results"]) == len(runner.TASK_CLASSES)


def test_dry_run_writes_manifest_commands_and_summary(tmp_path: Path) -> None:
    exit_code = runner.main(["--output-dir", str(tmp_path)])

    assert exit_code == 0
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "commands.sh").exists()
    assert (tmp_path / "summary.json").exists()
    assert "llama.cpp-experimental" in (tmp_path / "commands.sh").read_text(encoding="utf-8")
    manifest = json_load(tmp_path / "manifest.json")
    assert "quant_asymmetric_combined_k4" in {
        command["arm"] for command in manifest["command_templates"]
    }


def test_combined_arm_uses_external_mtp_drafter_template() -> None:
    args = runner.parse_args([])
    combined = next(arm for arm in runner.ARMS if arm.id == "quant_asymmetric_combined")
    argv = runner.arm_argv(args, combined, 19999)

    assert argv[argv.index("--spec-type") + 1] == "draft-mtp"
    assert argv[argv.index("-m") + 1] == str(runner.DEFAULT_CPU_VERIFIER_MODEL)
    assert argv[argv.index("-md") + 1] == str(runner.DEFAULT_MI210_DRAFTER_MODEL)
    assert argv[argv.index("--spec-draft-device") + 1] == "ROCm0"


def test_quality_checker_strict_format_task_passes() -> None:
    task = next(task for task in runner.TASK_CLASSES if task["id"] == "exact_format_strict_instruction")
    content = "\n".join(runner.STRICT_FORMAT_EXPECTED_LINES)

    result = runner.score_quality(task, content)

    assert result["pass"] is True


def test_quality_checker_strict_format_rejects_creative_fixture() -> None:
    task = next(task for task in runner.TASK_CLASSES if task["id"] == "exact_format_strict_instruction")
    content = "\n".join(
        [
            "DR0-1 the sky is blue today",
            "DR0-2 birds fly high above",
            "DR0-3 clouds drift slowly west",
            "DR0-4 wind blows soft and light",
            "DR0-5 sun shines warm and bright",
        ]
    )

    result = runner.score_quality(task, content)

    assert result["pass"] is False
    assert result["details"]["exact_match"] is False


def test_quality_checker_repetitive_structured_task_fits_token_cap() -> None:
    task = next(task for task in runner.TASK_CLASSES if task["id"] == "repetitive_structured_generation")
    content = "\n".join(
        f'{{"index": {index}, "status": "READY"}}'
        for index in range(runner.STRUCTURED_JSON_LINE_COUNT)
    )

    result = runner.score_quality(task, content)

    assert result["pass"] is True
    assert result["details"]["line_count"] == runner.STRUCTURED_JSON_LINE_COUNT


def test_execute_path_writes_summary_with_mocked_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_validate_live_inputs(args):
        return None

    def fake_run_arm_variant(args, variant, port):
        return {
            "arm": variant.id,
            "base_arm": variant.arm.id,
            "status": "ok",
            "fresh_server": True,
            "k": variant.k,
            "prompt_tokens": 10,
            "generated_tokens": 20,
            "draft_tokens": 10 if variant.k else 0,
            "accepted_draft_tokens": 5 if variant.k else 0,
            "alpha": 0.5 if variant.k else None,
            "wall_time_s": 1.0,
            "prompt_time_s": 0.25,
            "decode_time_s": 0.5,
            "prompt_tps": 40.0,
            "decode_tps": 40.0,
            "load_wall_clock_s": 0.1,
            "request_count": 1,
            "error_count": 0,
            "quality_results": [
                {
                    "task_class": "mock",
                    "status": "checked",
                    "pass": True,
                    "checker": "mock",
                    "details": {},
                }
            ],
            "task_results": [
                {
                    "task_class": "mock",
                    "content_sha256": runner.sha256_text("stable output"),
                }
            ],
            "cleanup": {"status": "ok", "terminated": True, "port_open_after": False},
        }

    monkeypatch.setattr(runner, "validate_live_inputs", fake_validate_live_inputs)
    monkeypatch.setattr(runner, "run_arm_variant", fake_run_arm_variant)
    monkeypatch.setattr(runner, "process_snapshot", lambda: {"lines": []})
    monkeypatch.setattr(runner, "rocm_smi_showpids", lambda: {"kfd_pids_observed": False})

    exit_code = runner.main(["--execute", "--output-dir", str(tmp_path), "--k", "1"])

    summary = json_load(tmp_path / "summary.json")
    assert exit_code == 0
    assert summary["mode"] == "execute"
    assert summary["dry_run_only"] is False
    assert summary["quality_gate"]["status"] == "pass"
    assert summary["output_stability_gate"]["status"] == "pass"
    assert summary["cleanup_proof"]["status"] == "pass"
    assert summary["observation_grade"] is True
    assert summary["decision_grade"] is False
    assert summary["fh_accounting"]["accounting_verdict"].startswith("not_decision_grade")


def test_execute_preflight_refuses_existing_llama_processes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "validate_live_inputs", lambda args: None)
    monkeypatch.setattr(
        runner,
        "process_snapshot",
        lambda: {"lines": ["123 llama-server -m contaminant.gguf"]},
    )
    monkeypatch.setattr(runner, "rocm_smi_showpids", lambda: {"kfd_pids_observed": False})

    with pytest.raises(RuntimeError, match="refusing contaminated"):
        runner.main(["--execute", "--output-dir", str(tmp_path), "--k", "1"])


def test_cleanup_status_fails_when_post_pid_remains(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = iter(
        [
            {"lines": []},
            {"lines": ["456 llama-server -m leaked.gguf"]},
        ]
    )

    def fake_run_arm_variant(args, variant, port):
        return {
            "arm": variant.id,
            "base_arm": variant.arm.id,
            "status": "ok",
            "fresh_server": True,
            "k": variant.k,
            "prompt_tokens": 10,
            "generated_tokens": 20,
            "draft_tokens": 0,
            "accepted_draft_tokens": 0,
            "alpha": None,
            "wall_time_s": 1.0,
            "prompt_time_s": 0.25,
            "decode_time_s": 0.5,
            "prompt_tps": 40.0,
            "decode_tps": 40.0,
            "load_wall_clock_s": 0.1,
            "request_count": 1,
            "error_count": 0,
            "quality_results": [
                {
                    "task_class": "mock",
                    "status": "checked",
                    "pass": True,
                    "checker": "mock",
                    "details": {},
                }
            ],
            "cleanup": {"status": "ok", "terminated": True, "port_open_after": False},
        }

    monkeypatch.setattr(runner, "validate_live_inputs", lambda args: None)
    monkeypatch.setattr(runner, "run_arm_variant", fake_run_arm_variant)
    monkeypatch.setattr(runner, "process_snapshot", lambda: next(calls))
    monkeypatch.setattr(runner, "rocm_smi_showpids", lambda: {"kfd_pids_observed": False})

    runner.main(["--execute", "--output-dir", str(tmp_path), "--k", "1"])
    summary = json_load(tmp_path / "summary.json")

    assert summary["cleanup_proof"]["status"] == "fail"
    assert summary["cleanup_proof"]["no_llama_process_leak"] is False
    assert summary["observation_grade"] is False


def test_quiet_preflight_ignores_earlyoom_ignore_pattern() -> None:
    args = runner.parse_args([])
    pre_process = {
        "lines": [],
        "stdout": (
            "1849379 /usr/local/bin/earlyoom --ignore ^(llama-server|sd-server)$ "
            "-N /mnt/raid0/llm/epyc-root/scripts/hooks/earlyoom_audit.sh\n"
        ),
    }

    runner.ensure_quiet_preflight(args, pre_process, {"kfd_pids_observed": False})


def test_row_from_response_extracts_spec_telemetry(tmp_path: Path) -> None:
    variant = next(
        variant
        for variant in runner.execution_variants(runner.parse_args(["--k", "2"]))
        if variant.id == "quant_asymmetric_combined_k2"
    )
    task = next(task for task in runner.TASK_CLASSES if task["id"] == "bounded_architect_reviewer_json_decision")
    response = {
        "choices": [{"message": {"content": '{"decision":"run","confidence":0.8,"rationale":"ok"}'}}],
        "timings": {
            "prompt_n": 10,
            "predicted_n": 20,
            "draft_n": 8,
            "draft_n_accepted": 6,
            "prompt_ms": 100.0,
            "predicted_ms": 200.0,
            "spec_verify_steps": 2,
            "spec_draft_ms": 1.0,
            "spec_verify_ms": 12.0,
            "spec_process_ms": 2.0,
            "spec_sample_accept_ms": 3.0,
            "spec_accept_by_depth": [2, 2, 1],
        },
    }

    row = runner.row_from_response(variant, task, response, tmp_path / "raw.json", 0.5)

    assert row["alpha"] == 0.75
    assert row["spec_telemetry"]["status"] == "observed"
    assert row["spec_telemetry"]["spec_verify_ms"] == 12.0
    assert row["spec_telemetry"]["spec_accept_by_depth"] == [2, 2, 1]


def test_fh_accounting_upgrades_when_spec_telemetry_is_observed() -> None:
    arms = {
        "quant_asymmetric_combined_k2": {
            "k": 2,
            "spec_telemetry_status": "observed",
            "spec_verify_time_s": 0.012,
            "spec_draft_time_s": 0.001,
            "spec_process_time_s": 0.002,
            "spec_sample_accept_time_s": 0.003,
            "spec_verify_steps": 2,
            "draft_tokens": 8,
            "accepted_draft_tokens": 6,
            "alpha": 0.75,
        }
    }

    rows = runner.observed_fh_rows(arms)

    assert rows == [
        {
            "arm": "quant_asymmetric_combined_k2",
            "k": 2,
            "F_K_verify_time_s": 0.012,
            "H_K_coordination_time_s": 0.006,
            "spec_draft_time_s": 0.001,
            "spec_process_time_s": 0.002,
            "spec_sample_accept_time_s": 0.003,
            "spec_verify_steps": 2,
            "draft_tokens": 8,
            "accepted_draft_tokens": 6,
            "alpha": 0.75,
        }
    ]


def test_target_output_stability_detects_combined_mismatch() -> None:
    baseline_hash = runner.sha256_text("baseline")
    changed_hash = runner.sha256_text("changed")
    arms = {
        "cpu_high_quant_verifier_baseline": {
            "task_results": [
                {"task_class": "strict", "content_sha256": baseline_hash},
            ]
        },
        "quant_asymmetric_combined_k2": {
            "k": 2,
            "task_results": [
                {"task_class": "strict", "content_sha256": changed_hash},
            ],
        },
    }

    rows = runner.target_output_match_rows(arms)

    assert rows == [
        {
            "arm": "quant_asymmetric_combined_k2",
            "k": 2,
            "target_output_match_vs_baseline": {"strict": False},
            "pass": False,
        }
    ]


def json_load(path: Path):
    import json

    return json.loads(path.read_text(encoding="utf-8"))
