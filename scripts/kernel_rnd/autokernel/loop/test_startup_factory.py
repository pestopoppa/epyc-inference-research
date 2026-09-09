"""Factory tests use real campaign_cli output, never a hand-upgraded manifest."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from . import campaign, campaign_cli, scheduling, scoped_evidence, startup_factory as factory
from . import standalone_inputs, unified_driver
from .test_production_enrollment import _campaign_config, _export, _policy, _seal_recipe_artifacts
from .test_unified_planner import canonical_recipe, profile, scheduler
from .test_planned_serving import _prompts


def pin(path: Path, value: dict) -> dict:
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(raw)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def request_for(tmp_path: Path, *, backend: str = "cpu", missing_candidate: bool = False,
                include_candidate: bool = True) -> dict:
    export = _seal_recipe_artifacts(_export(tmp_path, backend=backend), tmp_path)
    export_pin = pin(tmp_path / "production-export.json", export)
    recipe = canonical_recipe(backend=backend)
    recipe_pin = pin(tmp_path / "local-candidate-recipe.json", recipe.to_dict())
    artifacts = {}
    for kind, role in (("model", recipe.model), ("build", recipe.executable)):
        ref = "local:candidate:" + kind
        artifacts[kind] = {ref: {"schema": campaign.ARTIFACT_SCHEMA, "kind": kind,
            "ref": ref, "path": role.path, "sha256": role.sha256}}
    artifacts["recipe"] = {"local:candidate:recipe": {
        "schema": campaign.ARTIFACT_SCHEMA, "kind": "recipe", "ref": "local:candidate:recipe",
        "path": recipe_pin["path"], "sha256": recipe_pin["sha256"]}}
    config = _campaign_config(backend)
    config["local_seeds"] = {"targets": [{
        "schema": campaign.TARGET_SCHEMA, "request_id": config["request_id"], "target_id": "candidate",
        "backend": backend, "model_ref": "local:candidate:model", "build_ref": "local:candidate:build",
        "recipe_ref": "local:candidate:recipe", "context": recipe.template.ctx,
        "concurrency": recipe.template.np, "speculation": "none",
        "env": {key: value for key, value in recipe.launch_env if key != "LD_LIBRARY_PATH"},
        "metric": "aggregate_tok_s", "metric_direction": "higher", "roles": ["candidate"],
        "required_obligations": []}], "artifacts": artifacts}
    if missing_candidate:
        artifacts["model"] = {}
    if not include_candidate:
        del config["local_seeds"]
    config_pin = pin(tmp_path / "campaign-config.json", config)
    resolved_path = tmp_path / "campaign-resolved.json"
    assert campaign_cli.main(["--production-enrollment", export_pin["path"],
        "--production-campaign-config", config_pin["path"], "--out", str(resolved_path)]) == 0
    resolved_pin = {"path": str(resolved_path),
                    "sha256": hashlib.sha256(resolved_path.read_bytes()).hexdigest()}
    configuration, _ = scheduler(backend)
    configuration = replace(configuration, config_id="production-test")
    evidence = scoped_evidence.EvidenceIndex((), current_epoch="explicit-fixture-epoch",
                                           projection_available=False)
    return {
        "schema": factory.REQUEST_SCHEMA, "resolved_export": resolved_pin,
        "production_export": export_pin, "candidate_target_ids": ["candidate"] if include_candidate else [],
        "store_path": str(tmp_path / "campaign-store"), "config_generation": 1,
        "scheduler_config": configuration.to_dict(), "scheduler_state": None,
        "environment_policy": _policy(), "targets": {}, "target_defaults": {backend: {
            "profile": None, "profile_request": {
                "adapter_id": "explicit-profiler-v1", "adapter_digest": "a" * 64,
                "estimated_duration_seconds": 5, "estimated_claims": configuration.capacity.to_dict(),
                "submitted_at": 1}, "execution": None, "runtime_dimensions": []}},
        "experiment_plans": {}, "evidence_index": pin(tmp_path / "evidence.json", evidence.to_dict()),
        "actor_identities": {}, "providers": {"lifecycle": "unavailable-explicit-lifecycle",
            "readiness": "unavailable-explicit-readiness", "evidence_verifier": "unavailable-explicit-verifier"},
        "native_artifact_sink_ref": "explicit-native-artifact-sink",
        "dry_run_runner": {"python": sys.executable,
                           "pythonpath": str(Path(factory.__file__).resolve().parents[2])},
    }


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_campaign_cli_to_typed_startup_without_grants(tmp_path, monkeypatch, backend):
    request = request_for(tmp_path, backend=backend)
    def forbidden(*args, **kwargs):
        raise AssertionError("factory must not launch, acquire a provider or verify model bytes")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(campaign_cli, "_verify_identity", forbidden)
    monkeypatch.setattr(standalone_inputs.ProviderRegistry, "get", forbidden)
    output = tmp_path / "startup-bundle"
    receipt = factory.build_startup(request, output_dir=output)
    manifest = standalone_inputs.StartupManifest.from_dict(json.loads((output / "startup.json").read_text()))
    materialized = standalone_inputs.materialize(manifest)
    assert len(materialized.resolved.targets) == 2
    assert len(materialized.inputs.profile_requests) == 2
    assert materialized.inputs.profiles == {}
    assert len(materialized.inputs.runtime_anchors.recipes) == 2
    for digest, preparation in materialized.inputs.profile_requests.items():
        assert isinstance(preparation, unified_driver.ProfilePreparationRequest)
        assert preparation.target_revision_digest == digest
        assert preparation.stage_proposal.stage_class == "prerequisite"
        assert preparation.stage_proposal.estimated_duration_seconds == 5
    candidate = receipt["target_revision_map"]["candidate"]
    assert receipt["anchor_prerequisites"][candidate] == [
        "local_artifact_byte_verification_required", "local_first_launch_correctness_required"]
    assert manifest.driver_config.profiles == {}
    assert receipt["preflight"]["execution_authorized"] is False
    assert "evidence_index:projection_unavailable" in receipt["preflight"]["missing_prerequisites"]
    assert any("profile_execution:explicit-profiler-v1:unavailable" in item
               for item in receipt["preflight"]["missing_prerequisites"])
    assert receipt["execution_authorized"] is False
    assert receipt["artifact_verification"]["status"] == "not_requested"
    assert not Path(request["store_path"]).exists()
    assert (output.stat().st_mode & 0o777) == 0o700
    for name, digest in receipt["output_files"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    assert receipt["loaded_sources"] and receipt["dry_run_sources"]


def test_missing_candidate_dependency_is_not_manufactured(tmp_path):
    request = request_for(tmp_path, missing_candidate=True)
    receipt = factory.build_startup(request, output_dir=tmp_path / "bundle")
    candidate = receipt["target_revision_map"]["candidate"]
    manifest = json.loads((tmp_path / "bundle/startup.json").read_text())
    assert candidate not in manifest["driver_config"]["runtime_anchors"]
    assert candidate not in manifest["driver_config"]["profile_requests"]
    assert any(row["resolution_status"] == "missing_artifact" for row in receipt["target_dispositions"])


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_production_only_enrollment_accepts_explicit_empty_candidates(tmp_path, backend):
    request = request_for(tmp_path, backend=backend, include_candidate=False)
    receipt = factory.build_startup(request, output_dir=tmp_path / "bundle")
    manifest = standalone_inputs.StartupManifest.from_dict(json.loads((tmp_path / "bundle/startup.json").read_text()))
    materialized = standalone_inputs.materialize(manifest)
    assert receipt["candidate_target_ids"] == []
    assert set(receipt["target_revision_map"]) == {"frontdoor@8070"}
    assert len(materialized.resolved.targets) == len(materialized.inputs.profile_requests) == 1
    assert materialized.resolved.targets[0].enrolled_as == ("production",)
    assert receipt["execution_authorized"] is False
    assert not Path(request["store_path"]).exists()


def test_unicode_paths_use_existing_planner_and_startup_digest_contracts(tmp_path):
    directory = tmp_path / "réal"
    directory.mkdir()
    request = request_for(directory)
    receipt = factory.build_startup(request, output_dir=directory / "bundle")
    manifest = standalone_inputs.StartupManifest.from_dict(json.loads((directory / "bundle/startup.json").read_text()))
    assert receipt["preflight"] == standalone_inputs.materialize(manifest).preflight()


def test_oversized_recipe_is_refused_before_existing_loader(tmp_path, monkeypatch):
    request = request_for(tmp_path)
    export = json.loads(Path(request["production_export"]["path"]).read_text())
    recipe = next(row for row in export["targets"][0]["artifacts"] if row["use"] == "recipe")
    with Path(recipe["path"]).open("wb") as stream:
        stream.truncate(factory.MAX_INPUT_BYTES + 1)
    def forbidden(*args, **kwargs):
        raise AssertionError("unbounded production reader must not precede the bounded sidecar check")
    monkeypatch.setattr(factory.production_enrollment, "load_export", forbidden)
    with pytest.raises(ValueError, match="unsafe file identity or size"):
        factory.build_startup(request, output_dir=tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("field", ["resolved_export", "production_export", "evidence_index"])
def test_input_sha_mismatch_refuses_before_output(tmp_path, field):
    request = request_for(tmp_path)
    request[field]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        factory.build_startup(request, output_dir=tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("change", ["diagnostics", "source", "omit_production", "artifact_path", "obligations"])
def test_resealed_resolved_tampering_refuses(tmp_path, change):
    request = request_for(tmp_path)
    path = Path(request["resolved_export"]["path"])
    envelope = json.loads(path.read_text())
    production = next(row for row in envelope["resolved_campaign"]["targets"] if "production" in row["enrolled_as"])
    if change == "diagnostics":
        envelope["production_enrollment"]["export_sha256"] = "0" * 64
    elif change == "source":
        envelope["resolved_campaign"]["source_snapshot"]["launcher"]["path"] += ".foreign"
    elif change == "omit_production":
        envelope["resolved_campaign"]["targets"].remove(production)
    elif change == "obligations":
        production["required_obligations"] = []
    else:
        production["execution"]["model"]["path"] += ".foreign"
    request["resolved_export"] = pin(path, envelope)
    with pytest.raises((ValueError, RuntimeError)):
        factory.build_startup(request, output_dir=tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("change", ["candidate_not_array", "production_as_candidate", "existing_store",
    "profile_budget", "gpu_on_cpu", "unknown_target", "unknown_request", "symlink_input"])
def test_explicit_choice_and_budget_boundaries(tmp_path, change):
    request = request_for(tmp_path)
    if change == "candidate_not_array":
        request["candidate_target_ids"] = None
    elif change == "production_as_candidate":
        request["candidate_target_ids"] = ["frontdoor@8070"]
    elif change == "existing_store":
        Path(request["store_path"]).mkdir()
    elif change == "profile_budget":
        request["target_defaults"]["cpu"]["profile_request"]["estimated_duration_seconds"] = 61
    elif change == "gpu_on_cpu":
        request["target_defaults"]["cpu"]["profile_request"]["estimated_claims"]["gpu_devices"] = ["ROCm0"]
    elif change == "unknown_target":
        request["targets"]["unresolved"] = deepcopy(request["target_defaults"]["cpu"])
    elif change == "unknown_request":
        request["grant"] = "not-a-capability"
    else:
        source = Path(request["evidence_index"]["path"])
        link = tmp_path / "evidence-link.json"
        link.symlink_to(source)
        request["evidence_index"]["path"] = str(link)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        factory.build_startup(request, output_dir=tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_factory_cli_emits_runnable_frozen_installed_dryrun(tmp_path):
    frozen = os.environ.get("AUTOKERNEL_FACTORY_DRY_RUN_TREE")
    if not frozen:
        pytest.skip("set AUTOKERNEL_FACTORY_DRY_RUN_TREE to an exact standalone delivery tree")
    root = Path(frozen)
    request = request_for(tmp_path)
    request["dry_run_runner"] = {"python": sys.executable,
                                     "pythonpath": str(root / "scripts/kernel_rnd")}
    request_path = tmp_path / "startup-request.json"
    pin(request_path, request)
    output = tmp_path / "bundle"
    factory_run = subprocess.run([sys.executable, "-B", "-m", "autokernel.loop.startup_factory",
        "--request", str(request_path), "--out-dir", str(output)], cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(Path(factory.__file__).resolve().parents[2]),
             "PYTHONDONTWRITEBYTECODE": "1"}, text=True, capture_output=True, timeout=30, check=False)
    assert factory_run.returncode == 0, factory_run.stderr
    receipt = json.loads(factory_run.stdout)
    command = receipt["command"]
    completed = subprocess.run(command["argv"], cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": command["PYTHONPATH"], "PYTHONDONTWRITEBYTECODE": "1"},
        text=True, capture_output=True, timeout=30, check=False)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["execution_authorized"] is False
    assert report["status"] == "unavailable"
    assert report == receipt["preflight"]
    assert not Path(request["store_path"]).exists()
    for path, expected in receipt["dry_run_sources"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected


def test_supplied_profile_and_execution_roundtrip_without_invention(tmp_path):
    request = request_for(tmp_path)
    envelope = json.loads(Path(request["resolved_export"]["path"]).read_text())
    resolved = campaign.ResolvedCampaign.from_dict(envelope["resolved_campaign"])
    target = next(row for row in resolved.targets if "candidate" in row.target_ids)
    existing = profile(target, opportunities=[])
    prompts = _prompts(canonical_recipe().template)
    request["targets"]["candidate"] = {
        "profile": pin(tmp_path / "supplied-profile.json", existing),
        "profile_request": None, "runtime_dimensions": [], "execution": {
            "prompt_manifest": pin(tmp_path / "prompts.json", prompts.to_dict()),
            "max_stage_seconds": 30, "teardown_seconds": 5, "instrument_id": "explicit-serving-v1"}}
    receipt = factory.build_startup(request, output_dir=tmp_path / "bundle")
    manifest = standalone_inputs.StartupManifest.from_dict(json.loads((tmp_path / "bundle/startup.json").read_text()))
    materialized = standalone_inputs.materialize(manifest)
    candidate = receipt["target_revision_map"]["candidate"]
    assert materialized.inputs.profiles[candidate].to_dict() == existing
    assert candidate not in materialized.inputs.profile_requests
    assert materialized.inputs.execution_inputs[candidate].prompt_manifest == prompts
    assert materialized.inputs.execution_inputs[candidate].instrument_id == "explicit-serving-v1"


def test_existing_scheduler_state_preserves_accounting(tmp_path):
    request = request_for(tmp_path)
    config = scheduling.SchedulerConfig.from_dict(request["scheduler_config"])
    state = replace(scheduling.initial_state(config, "production-test"),
                    campaign_attempts=2, campaign_charged_seconds=15)
    request["scheduler_state"] = pin(tmp_path / "state.json", state.to_dict())
    Path(request["store_path"]).mkdir()
    factory.build_startup(request, output_dir=tmp_path / "bundle")
    manifest = json.loads((tmp_path / "bundle/startup.json").read_text())
    assert manifest["driver_config"]["scheduler_state"] == state.to_dict()
    assert list(Path(request["store_path"]).iterdir()) == []


def test_foreign_profile_refuses(tmp_path):
    request = request_for(tmp_path)
    envelope = json.loads(Path(request["resolved_export"]["path"]).read_text())
    target = campaign.ResolvedCampaign.from_dict(envelope["resolved_campaign"]).targets[0]
    foreign = profile(target)
    foreign["target_revision_digest"] = "0" * 64
    request["target_defaults"]["cpu"]["profile_request"] = None
    request["target_defaults"]["cpu"]["profile"] = pin(tmp_path / "profile.json", foreign)
    with pytest.raises(ValueError, match="profile differs from enrolled target"):
        factory.build_startup(request, output_dir=tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_mutated_input_closure_refuses_before_publication(tmp_path, monkeypatch):
    request = request_for(tmp_path)
    original = unified_driver.DriverConfig.from_dict
    def mutate_after_parse(value):
        result = original(value)
        Path(request["evidence_index"]["path"]).write_text("{}")
        return result
    monkeypatch.setattr(unified_driver.DriverConfig, "from_dict", mutate_after_parse)
    with pytest.raises(ValueError, match="input closure changed"):
        factory.build_startup(request, output_dir=tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_cli_refusal_is_clean_and_does_not_create_output(tmp_path, capsys):
    request = request_for(tmp_path)
    request["providers"]["lifecycle"] = ""
    path = tmp_path / "request.json"
    pin(path, request)
    assert factory.main(["--request", str(path), "--out-dir", str(tmp_path / "bundle")]) == 2
    assert "startup factory refused:" in capsys.readouterr().err
    assert not (tmp_path / "bundle").exists()
