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

    assert summary["schema"] == "epyc.dr0_quant_asym_self_spec.summary.v1"
    assert set(summary["arms"]) == {arm.id for arm in runner.ARMS}
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


def test_combined_arm_uses_external_mtp_drafter_template() -> None:
    args = runner.parse_args([])
    combined = next(arm for arm in runner.ARMS if arm.id == "quant_asymmetric_combined")
    argv = runner.arm_argv(args, combined, 19999)

    assert argv[argv.index("--spec-type") + 1] == "draft-mtp"
    assert argv[argv.index("-m") + 1] == str(runner.DEFAULT_CPU_VERIFIER_MODEL)
    assert argv[argv.index("-md") + 1] == str(runner.DEFAULT_MI210_DRAFTER_MODEL)
    assert argv[argv.index("--spec-draft-device") + 1] == "ROCm0"
