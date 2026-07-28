from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))
import fg4b_a4_cpu_evidence_importer as importer


def write_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    artifact = tmp_path / "artifact"
    artifact.mkdir(parents=True)
    (artifact / "COMPLETE").write_text("")
    regions = json.dumps([{"region": "q0", "global_held": False}])
    (artifact / "region-status-before.json").write_text(regions)
    (artifact / "region-status-after.json").write_text(regions)
    (artifact / "provenance.txt").write_text("\n".join((
        f"protocol_id={importer.EXPECTED_PROTOCOL}",
        f"metric={importer.EXPECTED_METRIC}",
        "n_gen=512", "reps=2", f"model={importer.EXPECTED_MODEL}",
        "started_at=2026-07-28T09:00:00+00:00", "research_commit=abc", "llama_commit=def",
        "llama_branch=production-consolidated-v8", f"binary={importer.EXPECTED_BINARY}", "binary_version_exit_code=1",
        "finished_at=2026-07-28T10:00:00+00:00", "exit_code=0", "",
    )))
    (artifact / "binary.sha256").write_text("a" * 64 + f"  {importer.EXPECTED_BINARY}\n")
    (artifact / "model.sha256").write_text("b" * 64 + f"  {importer.EXPECTED_MODEL}\n")
    (artifact / "instrument.sha256").write_text(
        "c" * 64 + f"  {importer.PROJECT_ROOT}/scripts/benchmark/bench_canonical.sh\n" +
        "d" * 64 + f"  {importer.PROJECT_ROOT}/scripts/lib/canonical_recipe.py\n"
    )
    (artifact / "binary-version.txt").write_text(f"usage: {importer.EXPECTED_BINARY} [options]\n")
    (artifact / "launcher.log").write_text(
        "FG-4b A4 CPU re-anchor watcher started: 2026-07-28T09:00:00+00:00\n"
        f"Evidence directory: {artifact.resolve()}\n"
        f"FG-4b completed successfully: {artifact.resolve()}\n"
    )
    (artifact / "bench.log").write_text("\n".join((
        "Env:       GGML_IQK=1 GGML_IQK_Q8_0=1 OMP_PROC_BIND=spread",
        f"Cmd: taskset -c 0-95 numactl --interleave=all {importer.EXPECTED_BINARY} -t 96 -fa 1 -mmp 0 -m {importer.EXPECTED_MODEL} -p 0 -n 512 -r 2 -o md",
        "| model | size | params | backend | ngl | threads | split | device | test | t/s |",
        "| Qwen | 34.0 GiB | 35B | CPU | 0 | 96 | 0 | none | tg512 | 42.50 ± 0.25 |",
        "",
    )))
    registry_payload = {"server_mode": {"frontdoor": {"model_path": importer.EXPECTED_MODEL}}}
    research = tmp_path / "research.yaml"
    orchestrator = tmp_path / "orchestrator.yaml"
    for path in (research, orchestrator):
        path.write_text(yaml.safe_dump(registry_payload))
    return artifact, research, orchestrator


def test_import_emits_evidence_and_non_applying_proposal(tmp_path: Path) -> None:
    artifact, research, orchestrator = write_fixture(tmp_path)
    evidence, proposal = importer.import_evidence(artifact, research, orchestrator)
    assert evidence["mean_tokens_per_second"] == 42.5
    assert evidence["spread_tokens_per_second"] == 0.25
    assert evidence["date"] == "2026-07-28"
    assert evidence["instrument_sha256"]
    assert proposal["mode"] == "proposal_only"
    assert proposal["intended_registry_field_targets"] == [importer.REGISTRY_TARGET]
    assert proposal["json_patch"][0]["path"].endswith("fg4b_a4_cpu_reanchor_20260728")


def test_refuses_incomplete_armed_artifact(tmp_path: Path) -> None:
    artifact, research, orchestrator = write_fixture(tmp_path)
    (artifact / "COMPLETE").unlink()
    with pytest.raises(importer.EvidenceError, match="artifact is incomplete"):
        importer.import_evidence(artifact, research, orchestrator)


def test_refuses_noncanonical_command_or_multiple_tg512_rows(tmp_path: Path) -> None:
    artifact, research, orchestrator = write_fixture(tmp_path)
    bench = artifact / "bench.log"
    bench.write_text(bench.read_text().replace("-t 96", "-t 95"))
    with pytest.raises(importer.EvidenceError, match="requires -t 96"):
        importer.import_evidence(artifact, research, orchestrator)
    artifact, research, orchestrator = write_fixture(tmp_path / "third")
    bench = artifact / "bench.log"
    bench.write_text(bench.read_text() + "| Qwen | 34.0 GiB | 35B | CPU | 0 | 96 | 0 | none | tg512 | 42.50 ± 0.25 |\n")
    with pytest.raises(importer.EvidenceError, match="exactly one successful CPU tg512"):
        importer.import_evidence(artifact, research, orchestrator)


def test_cli_refuses_to_write_inside_artifact(tmp_path: Path) -> None:
    artifact, research, orchestrator = write_fixture(tmp_path)
    result = importer.main([
        "--artifact-dir", str(artifact), "--evidence-out", str(artifact / "evidence.json"),
        "--proposal-out", str(tmp_path / "proposal.json"), "--research-registry", str(research),
        "--orchestrator-registry", str(orchestrator),
    ])
    assert result == 2
    assert not (artifact / "evidence.json").exists()
