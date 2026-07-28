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
    regions = json.dumps([{"region": f"q{index}", "global_held": False} for index in range(4)])
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
    (artifact / "binary.sha256").write_text(importer.EXPECTED_BINARY_SHA256 + f"  {importer.EXPECTED_BINARY}\n")
    (artifact / "model.sha256").write_text(importer.EXPECTED_MODEL_SHA256 + f"  {importer.EXPECTED_MODEL}\n")
    (artifact / "instrument.sha256").write_text(
        importer.EXPECTED_BENCH_CANONICAL_SHA256 + f"  {importer.PROJECT_ROOT}/scripts/benchmark/bench_canonical.sh\n" +
        importer.EXPECTED_CANONICAL_RECIPE_SHA256 + f"  {importer.PROJECT_ROOT}/scripts/lib/canonical_recipe.py\n"
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
        "| model | size | params | backend | threads | fa | mmap | test | t/s |",
        "| Qwen | 34.0 GiB | 35B | CPU | 96 | 1 | 0 | tg512 | 42.50 ± 0.25 |",
        "",
    )))
    registry_payload = {
        "roles": {"frontdoor": {"model": {"path": importer.EXPECTED_MODEL}, "performance": {
            "baseline_tps": importer.EXPECTED_OLD_BASELINE_TPS,
            "optimized_tps": importer.EXPECTED_OLD_BASELINE_TPS,
            "benchmark_date": importer.EXPECTED_OLD_BENCHMARK_DATE,
        }}},
        "server_mode": {"frontdoor": {"model_path": importer.EXPECTED_MODEL, "throughput": importer.EXPECTED_OLD_BASELINE_TPS}},
    }
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
    assert proposal["intended_registry_field_targets"] == list(importer.REGISTRY_TARGETS)
    assert [operation["op"] for operation in proposal["json_patch"]] == ["test", "replace", "test", "replace"]
    assert proposal["json_patch"][1]["value"] == 42.5


def test_refuses_incomplete_armed_artifact(tmp_path: Path) -> None:
    artifact, research, orchestrator = write_fixture(tmp_path)
    (artifact / "COMPLETE").unlink()
    with pytest.raises(importer.EvidenceError, match="artifact is incomplete"):
        importer.import_evidence(artifact, research, orchestrator)


def test_refuses_wrong_model_argv_and_bad_cpu_table(tmp_path: Path) -> None:
    artifact, research, orchestrator = write_fixture(tmp_path)
    bench = artifact / "bench.log"
    bench.write_text(bench.read_text().replace(importer.EXPECTED_MODEL, "/wrong/model.gguf"))
    with pytest.raises(importer.EvidenceError, match="requires -m"):
        importer.import_evidence(artifact, research, orchestrator)
    artifact, research, orchestrator = write_fixture(tmp_path / "third")
    bench = artifact / "bench.log"
    bench.write_text(bench.read_text().replace("| model | size | params | backend | threads | fa | mmap | test | t/s |", "| model | size | params | backend | threads | mmap | test | t/s |"))
    with pytest.raises(importer.EvidenceError, match="current CPU tg512 table header"):
        importer.import_evidence(artifact, research, orchestrator)


def test_refuses_wrong_backend_region_set_and_forged_digest(tmp_path: Path) -> None:
    artifact, research, orchestrator = write_fixture(tmp_path)
    (artifact / "bench.log").write_text((artifact / "bench.log").read_text().replace("| CPU |", "| ROCm |"))
    with pytest.raises(importer.EvidenceError, match="exactly one successful CPU tg512"):
        importer.import_evidence(artifact, research, orchestrator)
    artifact, research, orchestrator = write_fixture(tmp_path / "regions")
    (artifact / "region-status-after.json").write_text(json.dumps([{"region": "q0", "global_held": False}]))
    with pytest.raises(importer.EvidenceError, match="q0, q1, q2, and q3 exactly once"):
        importer.import_evidence(artifact, research, orchestrator)
    artifact, research, orchestrator = write_fixture(tmp_path / "digest")
    (artifact / "model.sha256").write_text("0" * 64 + f"  {importer.EXPECTED_MODEL}\n")
    with pytest.raises(importer.EvidenceError, match="reviewed FG-4b identities"):
        importer.import_evidence(artifact, research, orchestrator)


def test_refuses_registry_without_patch_parent(tmp_path: Path) -> None:
    artifact, research, orchestrator = write_fixture(tmp_path)
    payload = yaml.safe_load(research.read_text())
    del payload["roles"]["frontdoor"]["performance"]
    research.write_text(yaml.safe_dump(payload))
    with pytest.raises(importer.EvidenceError, match="lacks server_mode.frontdoor.model_path"):
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


def test_cli_validates_both_destinations_before_writing_either(tmp_path: Path) -> None:
    artifact, research, orchestrator = write_fixture(tmp_path)
    evidence = tmp_path / "evidence.json"
    result = importer.main([
        "--artifact-dir", str(artifact), "--evidence-out", str(evidence),
        "--proposal-out", str(artifact / "proposal.json"), "--research-registry", str(research),
        "--orchestrator-registry", str(orchestrator),
    ])
    assert result == 2
    assert not evidence.exists()
