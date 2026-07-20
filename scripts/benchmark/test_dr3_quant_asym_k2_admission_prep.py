from pathlib import Path

import pytest

from scripts.benchmark import dr3_quant_asym_k2_admission_prep as prep


def test_parse_args_defaults_to_fixed_k2_and_context_bands() -> None:
    args = prep.parse_args([])

    assert args.binary == prep.dr0.EXPERIMENTAL_SERVER.resolve()
    assert args.context_bands == [8192, 16384]
    assert prep.K_VALUE == 2


def test_parse_args_refuses_production_v6_binary() -> None:
    production_binary = (
        Path("/mnt/raid0/llm/llama.cpp") / "build-hip" / "bin" / "llama-server"
    )

    with pytest.raises(ValueError, match="production v6"):
        prep.parse_args(["--binary", str(production_binary)])


def test_manifest_selects_k2_and_never_k4(tmp_path: Path) -> None:
    args = prep.parse_args(["--output-dir", str(tmp_path)])
    manifest = prep.build_manifest(args)

    assert manifest["fixed_k"] == 2
    assert manifest["decision"]["numeric_swarm_surface_allowed"] is False
    assert manifest["decision"]["serve_live_traffic"] is False
    assert "K4 added only 3.85%" in manifest["decision"]["k2_selected_reason"]
    assert {command["k"] for command in manifest["command_templates"]} == {None, 2}
    assert all("production-consolidated-v6" not in command["shell"] for command in manifest["command_templates"])


def test_manifest_contains_required_admission_gates(tmp_path: Path) -> None:
    args = prep.parse_args(["--output-dir", str(tmp_path)])
    manifest = prep.build_manifest(args)
    gate_ids = {gate["id"] for gate in manifest["required_gates"]}
    task_ids = {task["id"] for task in manifest["admission_task_classes"]}

    assert "cpu_target_equivalence" in gate_ids
    assert "frontdoor_opportunity_cost" in gate_ids
    assert "p_gpu_1_production_named" in gate_ids
    assert "structured_json_long" in task_ids
    assert "long_context_tail" in task_ids


def test_dry_run_writes_bundle(tmp_path: Path) -> None:
    exit_code = prep.main(["--output-dir", str(tmp_path), "--context-band", "8192"])

    assert exit_code == 0
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "task_packet.jsonl").exists()
    assert (tmp_path / "commands.sh").exists()
    assert (tmp_path / "operator_run.sh").exists()
    commands = (tmp_path / "commands.sh").read_text(encoding="utf-8")
    assert "--spec-draft-n-max 2" in commands
    assert "--spec-draft-n-max 4" not in commands
    summary = (tmp_path / "summary.json").read_text(encoding="utf-8")
    assert "admission_ready_to_execute" in summary
