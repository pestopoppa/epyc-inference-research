"""Installed profile owner over real tiny subprocesses; no hardware/science claims."""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import os
import stat
import time
from types import SimpleNamespace

import pytest

from .. import journal
from . import campaign_control as cc, planned_serving, scheduling
from . import profile_preparation as pp, standalone_inputs as si, standalone_runtime as sr
from . import target_profile_execution as tp, unified_driver as ud
from . import worker_lifecycle as wl
from .test_standalone_inputs import _document, _verifier, FullHeldProvider
from .test_campaign_control import _command
from .test_unified_driver import runtime_driver
from .test_unified_planner import canonical_recipe, runtime_anchor


class SyntheticSelectedProfileProvider(FullHeldProvider):
    """Test-only original author of selected metadata; no hardware authority."""

    def __init__(self, root, requests):
        super().__init__(root)
        self.requests = {"profile-" + pp.state.digest(item.to_dict())[:24]: item
                         for item in requests.values()}

    def close_held_receipt(self, *, authorization, request, worker_id,
                           worker_generation, container_identity,
                           lifecycle_started_at, released_at, deadline):
        del container_identity, deadline
        original = self.requests[request.request_id]
        proposal = original.stage_proposal
        assert json.loads(request.argv[1]) == original.to_dict()
        receipt = scheduling.HeldClaimReceipt(
            f"synthetic-profile:{worker_id}:{worker_generation}", proposal.proposal_id,
            proposal.backend, proposal.stage_class, lifecycle_started_at, released_at,
            worker_generation, authorization.grant.generation,
            (f"synthetic-profile-claim:{authorization.container_id}",),
            0.5, (), 0, ("0",), {proposal.proposal_id: 1.0})
        return wl.TrustedHeldClaimReceipt(
            request.request_id, request.plan_digest, worker_id, worker_generation,
            authorization.grant.grant_id, authorization.grant.generation,
            authorization.container_id, authorization.grant.clock_domain,
            lifecycle_started_at, released_at, receipt)


def _fixture(tmp_path, *, malformed=None, return_code=0, selected_provider=False, backend="cpu"):
    document = _document(tmp_path)
    if backend == "gpu":
        recipe = canonical_recipe(backend="gpu")
        seed, _, resolved, resolved_target, digest = runtime_driver(git_source=True, recipe=recipe)
        (tmp_path / "resolved.json").write_text(json.dumps(resolved.to_dict()))
        document["driver_config"].update({
            "runtime_anchors": {digest: runtime_anchor(resolved_target, recipe)},
            "profiles": {key: value.to_dict() for key, value in seed.profiles.items()},
            "runtime_dimensions": {}, "experiment_plans": {}, "execution_inputs": {}})
        document["manifest_digest"] = si._digest({key: value for key, value in document.items()
                                                 if key != "manifest_digest"})
    target, profile = next(iter(document["driver_config"]["profiles"].items()))
    profile = copy.deepcopy(profile)
    profile["opportunities"] = []
    initial = si.materialize(si.StartupManifest.from_dict(document))
    recipe = initial.inputs.runtime_anchors.recipes[target]
    arm = planned_serving.arm_identity(recipe.template, recipe)
    loaded = {"target_revision_digest": target, "model_digest": arm["model_digest"],
              "quantization": profile["quant"], "recipe_digest": recipe.snapshot_digest,
              "executable_digest": arm["executable_digest"], "dso_digest": arm["dso_set_digest"]}
    output = {"schema": tp.PROFILE_OUTPUT_SCHEMA, "profile_content": profile,
              "loaded_identity": loaded, "artifact_identity": {"kind": "tiny-fixture", "sha256": "a" * 64},
              "measurement_carrier": {"schema": "epyc.autokernel.profile_measurement_carrier.v1",
                  "profile_source_id": tp.PROFILE_SOURCE_ID, "validation_source_id": tp.VALIDATION_SOURCE_ID,
                  "run_id": "synthetic-profile-fixture", "profile_claim_tuple": {"fixture": True},
                  "validation_claim_tuple": {"fixture": True}}}
    if malformed:
        output[malformed[0]] = malformed[1]
    counter = tmp_path / "launch-count"
    profiler = tmp_path / "tiny-profiler"
    profiler.write_text("#!/usr/bin/env python3\nfrom pathlib import Path\nimport json,sys\n"
        "request=json.loads(sys.argv[1])\n"
        f"assert request['target_revision_digest']=={target!r}\n"
        f"p=Path({str(counter)!r});p.write_text((p.read_text() if p.exists() else '')+'x')\n"
        f"print({json.dumps(output)!r})\nraise SystemExit({return_code})\n")
    profiler.chmod(profiler.stat().st_mode | stat.S_IXUSR)
    mechanism = tp.ProfileMechanism("tiny-profile", profiler,
        hashlib.sha256(profiler.read_bytes()).hexdigest(), tmp_path,
        {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}, loaded, 2.0, 1.0, 65536)
    binding = pp.InstalledProfileMechanismBinding(mechanism, 120.0)
    stage = scheduling.StageProposal(
        proposal_id=f"profile:installed:{backend}", submitted_at=1.0, backend=recipe.backend,
        target_revision=target, alias_identity=initial.resolved.targets[0].workload_signature,
        frontier_id=target, production_frontier=True, seed_id=None,
        stage_class="prerequisite", estimated_duration_seconds=3.0,
        estimated_claims=scheduling.ResourceVector(1.0, (), 0), eligible=True,
        eligibility_ref="1" * 64, reservation_kind=None, full_region=True,
        compatibility_authority_refs=(), safe_chunking_declared=False)
    request = ud.ProfilePreparationRequest.from_dict({"schema": ud.PROFILE_REQUEST_SCHEMA,
        "target_revision_digest": target, "stage_proposal": stage.to_dict(),
        "profile_contract": {"schema": ud.PROFILE_CONTRACT_SCHEMA,
                             "adapter_id": "tiny-profile", "adapter_digest": binding.adapter_digest}})
    document["driver_config"]["profiles"] = {}
    document["driver_config"]["profile_requests"] = {target: request.to_dict()}
    document["manifest_digest"] = si._digest({key: value for key, value in document.items()
                                              if key != "manifest_digest"})
    materialized = si.materialize(si.StartupManifest.from_dict(document))
    containers = tmp_path / "containers"
    containers.mkdir()
    provider = (SyntheticSelectedProfileProvider(containers, {target: request})
                if selected_provider else FullHeldProvider(containers))
    installed = pp.InstalledProfilePreparationBinding({target: binding})
    registry = si.ProviderRegistry({
        "fixture-lifecycle": si.ProviderBinding(lifecycle_provider=provider),
        "fixture-readiness": si.ProviderBinding(readiness_check=lambda: (True, None))},
        evidence_verifiers={"fixture-evidence": _verifier()},
        profile_bindings={"tiny-profile": installed})
    return materialized, registry, target, binding, counter, profile


def test_one_installed_owner_selects_two_cpu_gpu_target_mechanisms(tmp_path):
    cpu_root, gpu_root = tmp_path / "cpu", tmp_path / "gpu"
    cpu_root.mkdir()
    gpu_root.mkdir()
    cpu = _fixture(cpu_root)
    gpu = _fixture(gpu_root, backend="gpu")
    document = cpu[0].manifest.to_dict()
    resolved = cpu[0].resolved.to_dict()
    resolved["targets"] += gpu[0].resolved.to_dict()["targets"]
    resolved["resources"]["gpu_ids"] = ["ROCm0"]
    # Both targets and all generated contracts are frozen before original issue.
    (cpu_root / "resolved.json").write_text(json.dumps(resolved))
    for field in ("runtime_anchors", "profile_requests"):
        document["driver_config"][field].update(gpu[0].manifest.to_dict()["driver_config"][field])
    for field in ("runtime_dimensions", "experiment_plans", "execution_inputs"):
        document["driver_config"][field] = {}
    document["manifest_digest"] = si._digest({key: value for key, value in document.items()
                                             if key != "manifest_digest"})
    materialized = si.materialize(si.StartupManifest.from_dict(document))
    installed = pp.InstalledProfilePreparationBinding({cpu[2]: cpu[3], gpu[2]: gpu[3]})
    provider = SyntheticSelectedProfileProvider(cpu_root / "containers", materialized.inputs.profile_requests)
    registry = si.ProviderRegistry({
        "fixture-lifecycle": si.ProviderBinding(lifecycle_provider=provider),
        "fixture-readiness": si.ProviderBinding(readiness_check=lambda: (True, None))},
        evidence_verifiers={"fixture-evidence": _verifier()},
        profile_bindings={"tiny-profile": installed})
    controller, runtime = _start(materialized, registry)
    try:
        first = runtime.tick()
        first_target = first.execution_receipt["profile_reference"]["target_revision_digest"]
        first_expiry = controller.current_verified_profile_result(first_target)["profile_event"]["valid_until"]
        runtime._clock = lambda: first_expiry
        results = [first, runtime.tick()]
        assert [result.status for result in results] == ["settled", "settled"]
        assert any("profile_refresh_unavailable" in reason
                   for reason in results[1].driver_outcome["reasons"])
        assert results[1].execution_receipt["profile_reference"]["target_revision_digest"] != first_target
        assert len(runtime.profile_executor.producer.mechanisms) == 2
        assert cpu[4].read_text() == gpu[4].read_text() == "x"
        assert set(runtime.profile_executor.planner_profiles(time.monotonic())) == {cpu[2], gpu[2]}
        for fixture in (cpu, gpu):
            snapshot = controller.current_verified_profile_result(fixture[2])
            assert snapshot["profile_event"]["loaded_identity"] == fixture[3].mechanism.loaded_identity
            assert snapshot["profile_event"]["profile_content"] == ud._freeze(fixture[5])
            assert snapshot["settlement"]["receipt"]["backend"] == materialized.inputs.profile_requests[
                fixture[2]].stage_proposal.backend
        assert {result.execution_receipt["provider_held_receipt"]["backend"] for result in results} == {"cpu", "gpu"}
    finally:
        runtime.close()
        controller.close()


def _start(materialized, registry):
    report = materialized.preflight(registry)
    assert report["status"] == "ready", report
    config = materialized.manifest.driver_config
    controller, runtime = si.runtime_factory(materialized, registry)(materialized.resolved,
        SimpleNamespace(store=config.store_path, config_generation=config.config_generation,
                        snapshot_version=3))
    controller.apply_command(_command(materialized.resolved, "resume", "resume", 0))
    assert runtime.recover().status == "recovered"
    return controller, runtime


def test_actual_installed_profile_retains_original_output_but_binding_gate_blocks_feedback(tmp_path):
    materialized, registry, target, binding, counter, profile = _fixture(tmp_path)
    controller, runtime = _start(materialized, registry)
    try:
        with pytest.raises(sr.StandaloneRuntimeUncertain, match="profile settlement unresolved"):
            runtime.tick()
        assert counter.read_text() == "x"
        snapshot = controller.current_verified_profile_result(target)
        assert snapshot["profile_event"]["profile_content"] == ud._freeze(profile)
        assert snapshot["profile_event"]["profile_request"]["profile_contract"]["adapter_digest"] == binding.adapter_digest
        assert snapshot["settlement"] is None
        assert runtime.profile_executor.planner_profiles(time.monotonic()) == {}
        assert runtime.driver.profiles == {}
        with pytest.raises(sr.StandaloneRuntimeUncertain):
            runtime.retry_pending()
        assert counter.read_text() == "x"
        assert not controller._actor_profile_execution_reservations
        events = journal.Journal(str(controller.store / "journal"),
                                  campaign_id=controller.resolved.campaign_id).read_all()
        assert sum(row.kind == journal.KIND_ACTOR_PREPARATION
                   and row.payload["event"] == "PROFILE_VERIFIED" for row in events) == 1
        assert not any(row.kind == journal.KIND_UNIFIED_DRIVER_SETTLED for row in events)
        assert controller.unified_driver_pending_intent() is not None
    finally:
        runtime.close()
        controller.close()


def test_synthetic_original_provider_settles_feedback_expiry_and_fresh_restart(tmp_path, monkeypatch):
    materialized, registry, target, _, counter, profile = _fixture(tmp_path, selected_provider=True)
    controller, runtime = _start(materialized, registry)
    try:
        result = runtime.tick()
        assert result.status == "settled"
        parsed = pp.ProfilePreparationExecutionReceipt.from_dict(result.execution_receipt)
        assert parsed.body["disposition"] == "profile_verified"
        snapshot = ud._thaw(controller.current_verified_profile_result(target))
        row = snapshot["profile_event"]
        assert snapshot["settlement"]["outcome"] == "prerequisite"
        assert snapshot["settlement"]["terminal_refs"] == [row["verifier_ref"]]
        assert runtime.profile_executor.planner_profiles(row["verified_at"])[target].to_dict() == profile
        assert runtime.profile_executor.planner_profiles(row["valid_until"]) == {}
        runtime.driver.refresh_installed_profiles(runtime.profile_executor, now=row["verified_at"])
        assert target in runtime.driver.profiles
        runtime.driver.refresh_installed_profiles(runtime.profile_executor, now=row["valid_until"])
        assert runtime.driver.profiles == {}
        runtime._clock = lambda: row["valid_until"]
        waiting = runtime.tick()
        assert waiting.status == "waiting"
        assert "profile_refresh_unavailable:original_request_consumed" in waiting.reason
        assert controller.unified_driver_pending_intent() is None
        assert runtime.tick().status == "waiting"
        assert counter.read_text() == "x"
        events = journal.Journal(str(controller.store / "journal"),
                                  campaign_id=controller.resolved.campaign_id).read_all()
        assert sum(event.kind == journal.KIND_UNIFIED_DRIVER_SETTLED for event in events) == 1
    finally:
        runtime.close()
        controller.close()
    materialized = si.materialize(materialized.manifest)
    config = materialized.manifest.driver_config
    reopened, restarted = si.runtime_factory(materialized, registry)(materialized.resolved,
        SimpleNamespace(store=config.store_path, config_generation=config.config_generation,
                        snapshot_version=3))
    try:
        assert restarted.recover().status == "recovered"
        assert ud._thaw(reopened.current_verified_profile_result(target)) == snapshot
        before = (reopened.store / "journal").stat().st_mtime_ns
        assert restarted.profile_executor.planner_profiles(row["verified_at"])[target].to_dict() == profile
        assert restarted.profile_executor.planner_profiles(row["valid_until"]) == {}
        restarted._clock = lambda: row["valid_until"]
        assert "profile_refresh_unavailable:original_request_consumed" in restarted.tick().reason
        monkeypatch.setattr(wl, "monotonic_clock_domain", lambda: "synthetic-other-boot")
        assert restarted.profile_executor.planner_profiles(row["verified_at"]) == {}
        assert (reopened.store / "journal").stat().st_mtime_ns == before
        assert not restarted.profile_executor._attempts
        assert counter.read_text() == "x"
    finally:
        restarted.close()
        reopened.close()


def test_configured_target_with_registered_request_installs_refresh_owner(tmp_path):
    materialized, registry, target, _, counter, profile = _fixture(tmp_path, selected_provider=True)
    document = materialized.manifest.to_dict()
    document["driver_config"]["profiles"] = {target: profile}
    document["manifest_digest"] = si._digest({key: value for key, value in document.items()
                                             if key != "manifest_digest"})
    configured = si.materialize(si.StartupManifest.from_dict(document))
    assert configured.pending_profile_targets == ()
    controller, runtime = _start(configured, registry)
    try:
        assert type(runtime.profile_executor) is pp.InstalledProfilePreparationOwner
        assert target in runtime.profile_executor.binding.mechanisms
        assert runtime.tick().status == "waiting"
        assert not counter.exists()
    finally:
        runtime.close()
        controller.close()


@pytest.mark.parametrize("field,value", [
    ("profile_content", None), ("profile_content", []), ("profile_content", "bad"),
    ("profile_content", {}), ("artifact_identity", None), ("artifact_identity", []),
    ("artifact_identity", 4), ("artifact_identity", {}),
    ("profile_content", {"nonfinite": float("nan")}),
])
def test_malformed_original_profile_closes_reservation_and_retains_cost(tmp_path, field, value):
    materialized, registry, target, _, counter, _ = _fixture(tmp_path, malformed=(field, value))
    controller, runtime = _start(materialized, registry)
    try:
        with pytest.raises(sr.StandaloneRuntimeUncertain):
            runtime.tick()
        assert counter.read_text() == "x"
        assert controller.current_verified_profile_result(target) is None
        assert not controller._actor_profile_execution_reservations
        assert len(controller._actor_profile_execution_settlements) == 1
        held, terminal = next(iter(controller._actor_profile_execution_settlements.values()))
        assert terminal.return_code == 0
        assert held.ended_at > held.started_at
        assert runtime.profile_executor.planner_profiles(time.monotonic()) == {}
    finally:
        runtime.close()
        controller.close()


@pytest.mark.parametrize("mutation", ["digest", "model", "recipe", "backend"])
def test_profile_binding_refuses_before_launch(tmp_path, mutation):
    materialized, registry, target, binding, counter, _ = _fixture(tmp_path)
    if mutation == "digest":
        request = materialized.inputs.profile_requests[target].to_dict()
        request["profile_contract"]["adapter_digest"] = "f" * 64
        request = ud.ProfilePreparationRequest.from_dict(request)
        with pytest.raises(pp.ProfilePreparationRefused, match="adapter/source/target"):
            binding.validate_request(request)
    else:
        request = materialized.inputs.profile_requests[target]
        loaded = dict(binding.mechanism.loaded_identity)
        if mutation == "backend":
            raw = request.to_dict()
            raw["stage_proposal"]["backend"] = "gpu"
            request = ud.ProfilePreparationRequest.from_dict(raw)
        else:
            loaded["model_digest" if mutation == "model" else "recipe_digest"] = "f" * 64
            binding = pp.InstalledProfileMechanismBinding(replace(binding.mechanism,
                loaded_identity=loaded), binding.valid_for_seconds)
        raw = request.to_dict()
        raw["profile_contract"]["adapter_digest"] = binding.adapter_digest
        request = ud.ProfilePreparationRequest.from_dict(raw)
        installed = pp.InstalledProfilePreparationBinding({target: binding})
        with pytest.raises(pp.ProfilePreparationRefused, match="model/build/DSO/recipe/backend"):
            installed.validate(materialized.resolved, materialized.inputs.runtime_anchors, {target: request})
    assert not counter.exists()
    assert not (tmp_path / "store").exists()


def test_source_closure_complete_and_frozen_but_changed_loaded_producer_refuses(tmp_path, monkeypatch):
    materialized, _, target, binding, _, _ = _fixture(tmp_path)
    original = copy.deepcopy(binding.body())
    def changed(self, **kwargs):
        raise AssertionError("must not execute changed profiler")
    monkeypatch.setattr(tp.TargetProfileExecution, "prepare", changed)
    assert binding.body() == original
    with pytest.raises(pp.ProfilePreparationRefused, match="adapter/source/target"):
        binding.validate_request(materialized.inputs.profile_requests[target])


def test_exact_loaded_driver_default_is_separately_pinned_and_mutation_refuses(tmp_path, monkeypatch):
    materialized, _, target, binding, _, _ = _fixture(tmp_path)
    original = binding.body()["source_closure"]["driver_tick"]
    assert original["callable"]["configuration_status"] == "unproven"
    assert original["kwdefaults"]["stop_requested"]["configuration_status"] == "pinned"
    def changed():
        return True
    monkeypatch.setattr(ud.UnifiedCampaignDriver.tick, "__kwdefaults__",
                        {"now": None, "stop_requested": changed})
    with pytest.raises(pp.ProfilePreparationRefused, match="adapter/source/target"):
        binding.validate_request(materialized.inputs.profile_requests[target])


@pytest.mark.parametrize("replacement", [len, None])
def test_non_python_driver_replacement_is_declared_source_refusal(tmp_path, monkeypatch, replacement):
    materialized, _, target, binding, _, _ = _fixture(tmp_path)
    monkeypatch.setattr(ud.UnifiedCampaignDriver, "tick", replacement)
    with pytest.raises(pp.ProfilePreparationRefused, match="driver tick default shape"):
        binding.validate_request(materialized.inputs.profile_requests[target])


def test_runtime_source_refusal_is_bounded_and_keeps_issued_intent(tmp_path, monkeypatch):
    materialized, registry, _, _, counter, _ = _fixture(tmp_path)
    controller, runtime = _start(materialized, registry)
    def changed(self, **kwargs):
        raise AssertionError("changed profiler must not launch")
    monkeypatch.setattr(tp.TargetProfileExecution, "prepare", changed)
    try:
        first = runtime.tick()
        assert first.status == "waiting"
        assert "adapter/source/target" in first.reason
        pending = controller.unified_driver_pending_intent()
        assert pending is not None
        for _ in range(runtime.config.max_exact_retries + 1):
            final = runtime.retry_pending()
            if final.status == "recovery_required":
                break
        assert final.status == "recovery_required"
        assert controller.unified_driver_pending_intent() == pending
        assert not counter.exists()
    finally:
        runtime.close()
        controller.close()


def test_actual_profile_restart_reopens_original_diagnosis_without_live_issuance(tmp_path):
    materialized, registry, target, _, counter, _ = _fixture(tmp_path)
    controller, runtime = _start(materialized, registry)
    with pytest.raises(sr.StandaloneRuntimeUncertain):
        runtime.tick()
    original = ud._thaw(controller.current_verified_profile_result(target))
    store = controller.store
    runtime.close()
    controller.close()
    # Fresh scheduler/controller/owner; no held receipt registry is carried over.
    materialized = si.materialize(materialized.manifest)
    config = materialized.manifest.driver_config
    reopened, restarted = si.runtime_factory(materialized, registry)(materialized.resolved,
        SimpleNamespace(store=config.store_path, config_generation=config.config_generation,
                        snapshot_version=3))
    try:
        assert reopened is not controller
        assert restarted.profile_executor is not runtime.profile_executor
        assert ud._thaw(reopened.current_verified_profile_result(target)) == original
        before = (store / "journal").stat().st_mtime_ns
        assert restarted.profile_executor.planner_profiles(time.monotonic()) == {}
        assert (store / "journal").stat().st_mtime_ns == before
        result = restarted.recover()
        assert result.status == "recovery_required"
        assert "original held issuance" in result.reason
        assert counter.read_text() == "x"
        assert not restarted.profile_executor._attempts
    finally:
        restarted.close()
        reopened.close()


def _synthetic_reporting_receipt(attempt):
    """Grammar fixture only: deliberately not a provider-issued settlement."""
    from .driver_execution import _terminal_body
    work = attempt.work
    proposal = work.selection.proposal
    held = replace(attempt.held, proposal_id=proposal.proposal_id,
                   backend=proposal.backend, stage_class=proposal.stage_class)
    settlement = ud._thaw(attempt.settlement)
    settlement["receipt"] = held.to_dict()
    body = {"schema": pp.RECEIPT_SCHEMA, "catalog_id": work.catalog_id,
        "transition_id": work.transition_id, "selected_work": work.to_dict(),
        "selected_work_digest": pp.state.digest(work.to_dict()),
        "request_id": attempt.terminal.request_id, "stage_id": attempt.terminal.stage_id,
        "terminal": _terminal_body(attempt.terminal), "provider_held_receipt": held.to_dict(),
        "profile_reference": ud._thaw(attempt.profile_reference),
        "disposition": "profile_verified" if attempt.profile_reference else "failed",
        "settlement_request": settlement,
        "settlement_receipt": {"schema": cc.DRIVER_SETTLEMENT_RECEIPT_SCHEMA,
            "transition_id": work.transition_id, "status": "accepted",
            "accounting_projection_digest": "f" * 64}}
    return body


@pytest.fixture
def diagnostic_attempt(tmp_path):
    materialized, registry, target, _, counter, _ = _fixture(tmp_path)
    controller, runtime = _start(materialized, registry)
    try:
        with pytest.raises(sr.StandaloneRuntimeUncertain):
            runtime.tick()
        attempt = next(iter(runtime.profile_executor._attempts.values()))
        yield controller, runtime, target, counter, attempt
    finally:
        runtime.close()
        controller.close()


def test_parsed_reporting_receipt_cannot_admit_fabricated_settlement(diagnostic_attempt):
    controller, runtime, target, _, attempt = diagnostic_attempt
    body = _synthetic_reporting_receipt(attempt)
    parsed = pp.ProfilePreparationExecutionReceipt(body)
    body["terminal"]["worker_generation"] += 1
    assert parsed.to_dict()["terminal"]["worker_generation"] == attempt.terminal.worker_generation
    with pytest.raises(pp.ProfilePreparationRefused, match="original profile attempt"):
        runtime.profile_executor.verify_settlement(ud._thaw(parsed.body["settlement_request"]))
    assert controller.current_verified_profile_result(target)["settlement"] is None
    assert runtime.profile_executor.planner_profiles(time.monotonic()) == {}


@pytest.mark.parametrize("field", ["target", "request", "terminal", "held", "selection", "reference", "unknown"])
def test_reporting_receipt_rejects_changed_original_joins(diagnostic_attempt, field):
    _, _, _, _, attempt = diagnostic_attempt
    body = _synthetic_reporting_receipt(attempt)
    if field == "target":
        body["profile_reference"]["target_revision_digest"] = "f" * 64
    elif field == "request":
        body["request_id"] += "other"
    elif field == "terminal":
        body["terminal"]["worker_generation"] += 1
    elif field == "held":
        body["provider_held_receipt"]["allocation_generation"] += 1
    elif field == "selection":
        body["settlement_request"]["transition_id"] = "f" * 64
    elif field == "reference":
        body["profile_reference"]["profile_request_digest"] = "f" * 64
    else:
        body["unknown"] = True
    with pytest.raises(pp.ProfilePreparationRefused):
        pp.ProfilePreparationExecutionReceipt(body)


@pytest.mark.parametrize("field", ["catalog_id", "transition_id", "profile_request_digest", "verifier_ref", "clock_domain"])
def test_older_profile_cannot_be_current_attempt_publication(diagnostic_attempt, field):
    controller, _, target, _, attempt = diagnostic_attempt
    snapshot = ud._thaw(controller.current_verified_profile_result(target))
    row = snapshot["profile_event"]
    assert pp._current_publication(snapshot, attempt.work, attempt.terminal,
        row["verified_at"], row["clock_domain"], row["valid_until"]) == row
    start, domain, expiry = row["verified_at"], row["clock_domain"], row["valid_until"]
    row[field] = "f" * 64 if field.endswith("digest") or field.endswith("id") else "historical-other"
    assert pp._current_publication(snapshot, attempt.work, attempt.terminal, start, domain, expiry) is None


def test_exact_retry_rejects_forged_outcome_even_with_same_transition(diagnostic_attempt):
    _, runtime, _, counter, attempt = diagnostic_attempt
    outcome = runtime._pending_outcome
    forged = ud.DriverOutcome(**(outcome.to_dict() | {"reasons": ["foreign-history"]}))
    with pytest.raises(pp.ProfilePreparationRefused, match="DriverOutcome identity"):
        runtime.profile_executor.execute(runtime.driver, forged)
    # Exercise the cached branch without pretending this reporting fixture is
    # original settlement authority: identity must be checked before any return.
    runtime.profile_executor._receipts[attempt.work.transition_id] = object()
    with pytest.raises(pp.ProfilePreparationRefused, match="DriverOutcome identity"):
        runtime.profile_executor.execute(runtime.driver, forged)
    assert counter.read_text() == "x"
