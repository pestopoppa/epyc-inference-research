"""Focused startup materializer and installed-entrypoint tests."""
from __future__ import annotations

import json
import copy
from dataclasses import replace
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.request import Request, urlopen

import pytest

from . import scheduling, scoped_evidence, standalone_inputs as inputs, worker_lifecycle
from . import unified_driver
from .test_driver_execution import HeldProvider
from .test_unified_driver import runtime_driver
from .test_unified_planner import canonical_recipe, runtime_anchor


def _verifier(rule="fixture-rule"):
    return inputs.EvidenceVerifierBinding(
        scope_verifier=lambda *_args: True,
        use_verifier=lambda *_args: True,
        result_verifier=lambda *_args: True,
        support_rule_identity=rule,
    )


class FullHeldProvider(HeldProvider):
    def inspect(self, grant, container_id, deadline):
        del grant, container_id, deadline
        return worker_lifecycle.RecoveryInspection(
            "absent_released", None, "fixture proves absence")

    def describe_active_observation_claim(self, **_request):
        return {"fixture": "not exercised by the v1 prepared-stage path"}


def _document(tmp_path):
    seed, engine, resolved, target, target_digest = runtime_driver(git_source=True)
    resolved_path = tmp_path / "resolved.json"
    resolved_path.write_text(json.dumps(resolved.to_dict()))
    config = {
        "schema": unified_driver.CONFIG_SCHEMA,
        "resolved_campaign_path": str(resolved_path.resolve()),
        "store_path": str((tmp_path / "store").resolve()),
        "scheduler_config": None,
        "scheduler_state": None,
        "runtime_anchors": {
            target_digest: runtime_anchor(target, canonical_recipe())},
        "runtime_dimensions": {
            key: [item.to_dict() for item in value]
            for key, value in seed.runtime_dimensions.items()},
        "profiles": {key: value.to_dict() for key, value in seed.profiles.items()},
        "experiment_plans": {
            key: value.to_dict() for key, value in seed.experiment_plans.items()},
        "profile_requests": {
            key: value.to_dict() for key, value in seed.profile_requests.items()},
        "execution_inputs": {
            key: value.to_dict() for key, value in seed.execution_inputs.items()},
        "native_artifact_sink_ref": seed.sink_ref,
        "config_generation": 1,
    }
    body = {
        "schema": inputs.MANIFEST_SCHEMA,
        "driver_config": config,
        "evidence_index": seed.evidence.to_dict(),
        "actor_identities": {"source": {"build": "1" * 64}},
        "lifecycle_provider_id": "fixture-lifecycle",
        "readiness_provider_id": "fixture-readiness",
        "evidence_verifier_id": "fixture-evidence",
    }
    scheduler_config = scheduling.SchedulerConfig.from_dict(
        engine.config.to_dict() | {"config_id": resolved.campaign_id})
    body["driver_config"]["scheduler_config"] = scheduler_config.to_dict()
    body["driver_config"]["scheduler_state"] = scheduling.initial_state(
        scheduler_config, resolved.campaign_id).to_dict()
    body["evidence_index"] = scoped_evidence.EvidenceIndex(
        (), current_epoch="fixture", support_rule_identity="fixture-rule").to_dict()
    return body | {"manifest_digest": inputs._digest(body)}


def _management_document(tmp_path):
    document = copy.deepcopy(_document(tmp_path))
    for profile in document["driver_config"]["profiles"].values():
        profile["opportunities"] = []
    document["driver_config"]["runtime_dimensions"] = {}
    document["driver_config"]["experiment_plans"] = {}
    document["driver_config"]["execution_inputs"] = {}
    document["manifest_digest"] = inputs._digest({
        key: value for key, value in document.items() if key != "manifest_digest"})
    return document


def test_dry_run_uses_installed_main_without_store_or_provider(tmp_path, capsys):
    document = _document(tmp_path)
    path = tmp_path / "startup.json"
    path.write_text(json.dumps(document))

    class Explodes:
        def get(self, _identifier):
            raise AssertionError("dry-run consulted provider registry")

    assert unified_driver.main(
        ["--config", str(path), "--dry-run"], provider_registry=Explodes()) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "unavailable"
    assert result["execution_authorized"] is False
    assert result["manifest_digest"] == document["manifest_digest"]
    assert set(result["missing_prerequisites"]) == {
        "lifecycle_provider:fixture-lifecycle:unavailable",
        "readiness_provider:fixture-readiness:unavailable",
        "evidence_verifier:fixture-evidence:unavailable",
        "native_observation:typed_source_runtime_consumer_unavailable",
    }
    assert not (tmp_path / "store").exists()


def test_manifest_hash_is_integrity_not_authority_and_old_reader_refuses(tmp_path):
    document = _document(tmp_path)
    path = tmp_path / "startup.json"
    path.write_text(json.dumps(document))
    with pytest.raises(unified_driver.DriverRefused, match="fields/schema"):
        unified_driver.load_config(path)
    document["actor_identities"]["source"]["build"] = "2" * 64
    with pytest.raises(inputs.StandaloneInputsRefused, match="digest"):
        inputs.StartupManifest.from_dict(document)


def test_parsed_manifest_is_detached_and_recursive_input_is_typed_refusal(tmp_path):
    document = _document(tmp_path)
    parsed = inputs.StartupManifest.from_dict(document)
    document["actor_identities"]["source"]["build"] = "f" * 64
    document["evidence_index"]["current_epoch"] = "mutated"
    materialized = inputs.materialize(parsed)
    assert materialized.manifest.actor_identities["source"]["build"] == "1" * 64
    assert materialized.inputs.evidence_index.current_epoch == "fixture"
    recursive = {}
    recursive["self"] = recursive
    with pytest.raises(inputs.StandaloneInputsRefused, match="canonical JSON"):
        inputs.StartupManifest.from_dict(recursive)


def test_evidence_reconstruction_does_not_invent_verifier_callbacks(tmp_path):
    materialized = inputs.materialize(inputs.StartupManifest.from_dict(_document(tmp_path)))
    evidence = materialized.inputs.evidence_index
    assert evidence._scope_verifier is None
    assert evidence._use_verifier is None
    assert evidence._result_verifier is None
    assert evidence._recorded_support_rule_identity == "fixture-rule"


def test_native_runtime_work_retains_explicit_unavailable_source_debt(tmp_path):
    materialized = inputs.materialize(inputs.StartupManifest.from_dict(_document(tmp_path)))
    assert ("native_observation:typed_source_runtime_consumer_unavailable"
            in materialized.missing_prerequisites)


def test_missing_profile_and_provider_are_explicit(tmp_path):
    document = _document(tmp_path)
    document["driver_config"]["profiles"] = {}
    document["manifest_digest"] = inputs._digest({
        key: value for key, value in document.items() if key != "manifest_digest"})
    materialized = inputs.materialize(inputs.StartupManifest.from_dict(document))
    report = materialized.preflight(inputs.ProviderRegistry({}))
    assert report["status"] == "unavailable"
    assert any(item.endswith(":profile_request_missing")
               for item in report["missing_prerequisites"])
    assert any(item.startswith("lifecycle_provider:")
               for item in report["missing_prerequisites"])


def test_required_actor_identity_is_derived_from_existing_profile_semantics(tmp_path):
    document = _document(tmp_path)
    document["actor_identities"] = {}
    profile = next(iter(document["driver_config"]["profiles"].values()))
    profile["opportunities"][0]["kind"] = "source"
    profile["opportunities"][0]["runtime_dimension_ids"] = []
    document["manifest_digest"] = inputs._digest({
        key: value for key, value in document.items() if key != "manifest_digest"})
    materialized = inputs.materialize(inputs.StartupManifest.from_dict(document))
    assert "actor_identity:source:unavailable" in materialized.missing_prerequisites


def test_schedulable_profile_prerequisite_is_target_pending_not_profile_missing(tmp_path):
    document = _document(tmp_path)
    target_digest, profile = next(iter(document["driver_config"]["profiles"].items()))
    target = inputs.campaign_service.load_resolved(
        Path(document["driver_config"]["resolved_campaign_path"])).targets[0]
    cost = scheduling.ResourceVector(1.0, (), 0)
    stage = scheduling.StageProposal(
        proposal_id="profile:fixture", submitted_at=1, backend="cpu",
        target_revision=target_digest, alias_identity=target.workload_signature,
        frontier_id=target_digest, production_frontier=True, seed_id=None,
        stage_class="prerequisite", estimated_duration_seconds=10,
        estimated_claims=cost, eligible=True, eligibility_ref="2" * 64,
        reservation_kind=None, full_region=True,
        compatibility_authority_refs=(), safe_chunking_declared=False)
    document["driver_config"]["profiles"] = {}
    document["driver_config"]["profile_requests"] = {target_digest: {
        "schema": unified_driver.PROFILE_REQUEST_SCHEMA,
        "target_revision_digest": target_digest,
        "stage_proposal": stage.to_dict(),
        "profile_contract": {
            "schema": unified_driver.PROFILE_CONTRACT_SCHEMA,
            "adapter_id": "fixture-profile-v1", "adapter_digest": "9" * 64},
    }}
    document["manifest_digest"] = inputs._digest({
        key: value for key, value in document.items() if key != "manifest_digest"})
    materialized = inputs.materialize(inputs.StartupManifest.from_dict(document))
    assert materialized.pending_profile_targets == (target_digest,)
    assert not any("profile_missing" in item or "profile_request_missing" in item
                   for item in materialized.missing_prerequisites)
    report = materialized.preflight()
    assert report["pending_profile_targets"] == [target_digest]
    assert (f"target:{target_digest}:profile_execution:fixture-profile-v1:unavailable"
            in report["missing_prerequisites"])


def test_regular_file_loader_refuses_fifo_without_blocking(tmp_path):
    fifo = tmp_path / "config.fifo"
    os.mkfifo(fifo)
    started = time.monotonic()
    with pytest.raises(unified_driver.DriverRefused, match="identity/size"):
        unified_driver.load_config_document(fifo)
    assert time.monotonic() - started < 0.5


def test_regular_file_loader_detects_named_path_replacement(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_document(tmp_path)))
    replacement = tmp_path / "replacement.json"
    replacement.write_text(path.read_text())
    real_read = os.read
    replaced = False

    def swapping_read(fd, size):
        nonlocal replaced
        result = real_read(fd, size)
        if result and not replaced:
            replaced = True
            os.replace(replacement, path)
        return result

    monkeypatch.setattr(unified_driver.os, "read", swapping_read)
    with pytest.raises(unified_driver.DriverRefused, match="changed during bounded read"):
        unified_driver.load_config_document(path)


def test_preflight_validates_provider_surface_and_exact_evidence_rule(tmp_path):
    materialized = inputs.materialize(inputs.StartupManifest.from_dict(_document(tmp_path)))
    malformed = inputs.ProviderRegistry({
        "fixture-lifecycle": inputs.ProviderBinding(lifecycle_provider=object()),
        "fixture-readiness": inputs.ProviderBinding(readiness_check=lambda: (True, None)),
    }, evidence_verifiers={"fixture-evidence": _verifier("foreign-rule")})
    missing = materialized.preflight(malformed)["missing_prerequisites"]
    assert "lifecycle_provider:fixture-lifecycle:invalid" in missing
    assert "evidence_verifier:fixture-evidence:rule_mismatch" in missing
    callbacks_missing = inputs.ProviderRegistry({}, evidence_verifiers={
        "fixture-evidence": inputs.EvidenceVerifierBinding(
            None, lambda *_args: True, lambda *_args: True, "fixture-rule")})
    assert "evidence_verifier:fixture-evidence:unavailable" in (
        materialized.preflight(callbacks_missing)["missing_prerequisites"])


def test_typed_factory_constructs_and_restarts_same_owner_inputs(tmp_path):
    document = _management_document(tmp_path)
    materialized = inputs.materialize(inputs.StartupManifest.from_dict(document))
    containers = tmp_path / "containers"
    containers.mkdir(mode=0o700)
    provider = FullHeldProvider(containers)
    registry = inputs.ProviderRegistry({
        "fixture-lifecycle": inputs.ProviderBinding(lifecycle_provider=provider),
        "fixture-readiness": inputs.ProviderBinding(readiness_check=lambda: (True, None)),
    }, evidence_verifiers={"fixture-evidence": _verifier()})
    factory = inputs.runtime_factory(materialized, registry)

    class Args:
        store = document["driver_config"]["store_path"]
        config_generation = 1
        snapshot_version = 3

    controller, runtime = factory(materialized.resolved, Args())
    try:
        assert runtime.recover().status == "recovered"
        assert runtime.driver.scheduler is controller._scheduler_engine
        assert runtime.driver.scheduler is not materialized.inputs.scheduler_engine
        first = controller.snapshot()
        assert first["schema"].endswith(".v3")
    finally:
        runtime.close()
        controller.close()

    restarted, rerun = factory(materialized.resolved, Args())
    try:
        assert rerun.recover().status == "recovered"
        assert rerun.driver.scheduler is restarted._scheduler_engine
        assert rerun.driver.scheduler is not runtime.driver.scheduler
        assert restarted.resolved.to_dict() == materialized.resolved.to_dict()
        assert restarted.store == controller.store
        assert rerun.driver.evidence.to_dict() == materialized.inputs.evidence_index.to_dict()
        assert rerun.driver.execution_inputs.keys() == materialized.inputs.execution_inputs.keys()
    finally:
        rerun.close()
        restarted.close()


def test_factory_refuses_materialized_scheduler_drift(tmp_path):
    document = _management_document(tmp_path)
    materialized = inputs.materialize(inputs.StartupManifest.from_dict(document))
    state = materialized.inputs.scheduler_engine.export_state().to_dict()
    state["successor_fences"] = ["fixture drift"]
    changed = scheduling.SchedulerEngine(
        scheduling.SchedulerConfig.from_dict(document["driver_config"]["scheduler_config"]),
        scheduling.SchedulerState.from_dict(state))
    materialized = replace(materialized, inputs=replace(
        materialized.inputs, scheduler_engine=changed))
    registry = inputs.ProviderRegistry({
        "fixture-lifecycle": inputs.ProviderBinding(
            lifecycle_provider=FullHeldProvider(tmp_path / "containers")),
        "fixture-readiness": inputs.ProviderBinding(readiness_check=lambda: (True, None)),
    }, evidence_verifiers={"fixture-evidence": _verifier()})
    (tmp_path / "containers").mkdir(mode=0o700)
    with pytest.raises(inputs.StandaloneInputsRefused,
                       match="differs from the immutable startup seed"):
        inputs.runtime_factory(materialized, registry)


def test_listen_entrypoint_delegates_exact_factory_to_existing_service(
        tmp_path, monkeypatch):
    document = _management_document(tmp_path)
    path = tmp_path / "startup.json"
    path.write_text(json.dumps(document))
    containers = tmp_path / "containers"
    containers.mkdir(mode=0o700)
    registry = inputs.ProviderRegistry({
        "fixture-lifecycle": inputs.ProviderBinding(
            lifecycle_provider=FullHeldProvider(containers)),
        "fixture-readiness": inputs.ProviderBinding(readiness_check=lambda: (True, None)),
    }, evidence_verifiers={"fixture-evidence": _verifier()})
    observed = {}

    def service(argv, *, runtime_factory):
        observed["argv"] = argv
        observed["factory"] = runtime_factory
        return 7

    monkeypatch.setattr(unified_driver.campaign_service, "main", service)
    assert unified_driver.main(
        ["--config", str(path), "--listen", "127.0.0.1:0"],
        provider_registry=registry) == 7
    assert observed["argv"][-2:] == ["--listen", "127.0.0.1:0"]
    assert callable(observed["factory"])


def test_real_installed_entrypoint_serves_typed_status_and_restarts_same_store(tmp_path):
    document = _management_document(tmp_path)
    path = tmp_path / "startup.json"
    path.write_text(json.dumps(document))
    containers = tmp_path / "containers"
    containers.mkdir(mode=0o700)
    script = r'''
from pathlib import Path
import sys
from autokernel.loop import standalone_inputs, unified_driver
from autokernel.loop.test_standalone_inputs import FullHeldProvider, _verifier

root = Path(sys.argv[1])
registry = standalone_inputs.ProviderRegistry({
    "fixture-lifecycle": standalone_inputs.ProviderBinding(
        lifecycle_provider=FullHeldProvider(root / "containers")),
    "fixture-readiness": standalone_inputs.ProviderBinding(
        readiness_check=lambda: (True, None)),
}, evidence_verifiers={"fixture-evidence": _verifier()})
raise SystemExit(unified_driver.main([
    "--config", str(root / "startup.json"),
    "--listen", f"127.0.0.1:{sys.argv[2]}",
], provider_registry=registry))
'''
    env = os.environ.copy()
    env["AUTOKERNEL_CONTROL_TOKEN"] = "startup-fixture-token"
    env["PYTHONPATH"] = str(Path(__file__).parents[2])

    def one_start():
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        process = subprocess.Popen(
            [sys.executable, "-c", script, str(tmp_path), str(port)],
            cwd=Path(__file__).parents[4], env=env)
        base = f"http://127.0.0.1:{port}"
        try:
            deadline = time.monotonic() + 4
            while True:
                assert process.poll() is None
                try:
                    with urlopen(base + "/health", timeout=0.2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                assert time.monotonic() < deadline
                time.sleep(0.02)
            request = Request(
                base + "/snapshot",
                headers={"Authorization": "Bearer startup-fixture-token"})
            with urlopen(request, timeout=1) as response:
                snapshot = json.loads(response.read())
            os.kill(process.pid, signal.SIGTERM)
            assert process.wait(timeout=5) == 0
            return snapshot
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=3)

    first = one_start()
    second = one_start()
    assert first["schema"].endswith(".v3") and second["schema"].endswith(".v3")
    for key in ("campaign_id", "config_digest", "config_generation"):
        assert second[key] == first[key]
    assert second["supervisor_incarnation"] == first["supervisor_incarnation"] + 1
    assert document["manifest_digest"] == inputs.StartupManifest.from_dict(
        json.loads(path.read_text())).manifest_digest
