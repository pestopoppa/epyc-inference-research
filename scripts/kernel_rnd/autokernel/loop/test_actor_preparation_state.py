"""Closed actor-preparation native event and recovery tests."""
from __future__ import annotations

import copy
import hashlib
import os
import stat
import time
from types import SimpleNamespace

import pytest

from .. import journal
from . import actor_preparation, actor_preparation_state as state
from . import campaign_control, target_profile_execution
from . import scheduling, unified_driver, worker_lifecycle
from .test_actor_lifecycle import _held_receipt, _profile_receipt
from .test_actor_preparation import _budgets, _fake_executable, _profiles, _request, _resolved
from .test_campaign_control import _command
from .test_unified_driver import runtime_driver
from .test_unified_planner import scheduler as make_scheduler
from .test_worker_lifecycle import ReceiptProvider


def _dimensions(value):
    return {key: value for key in state.BUDGET_KEYS}


def _intent():
    return {
        "schema": state.INTENT_SCHEMA, "event": "INTENT",
        "reservation_id": "reservation-1", "campaign_id": "campaign",
        "config_generation": 1, "config_digest": "1" * 64,
        "supervisor_id": "supervisor", "supervisor_incarnation": 2,
        "control_revision": 3, "catalog_id": "2" * 64,
        "transition_id": "3" * 64, "request_digest": "4" * 64,
        "stage_plan_digest": "5" * 64, "target_revision_digest": "6" * 64,
        "target_profile_digest": "7" * 64,
        "target_profile_receipt_digest": "8" * 64,
        "actor_profile_digest": "9" * 64, "backend_key": "codex:model@low",
        "clock_domain": "monotonic:boot", "deadline": 100.0,
        "occurred_at": "2026-09-09T00:00:00Z",
        "budgets": _dimensions(10),
        "debits": {**_dimensions(0.0), "actor_calls_per_target": 1.0,
                   "actor_calls_per_campaign": 1.0},
    }


def _finish():
    intent = _intent()
    row = {key: value for key, value in intent.items()
           if key not in {"budgets", "debits"}}
    row.update({
        "schema": state.FINISH_SCHEMA, "event": "FINISH",
        "occurred_at": "2026-09-09T00:01:00Z", "outcome_digest": "a" * 64,
        "status": "completed", "failure_class": None, "charged_seconds": 2.5,
        "resource_enforced": True, "descendants_clean": True,
        "disposition": "proposal_ready",
        "charges": {**_dimensions(0.0), "actor_calls_per_target": 1.0,
                    "actor_calls_per_campaign": 1.0,
                    "provider_seconds_per_target": 2.5},
        "consecutive_failures": 0, "last_success": 90.0,
        "retry_after": None, "reset_at": None, "next_eligible_at": None,
    })
    return row


def test_intent_reserves_all_six_dimensions_and_restart_stays_unresolved():
    projection = state.project_events([_intent()])
    assert set(projection.reserved) == state.BUDGET_KEYS
    assert projection.reserved["actor_calls_per_target"] == 1.0
    assert projection.spent["actor_calls_per_target"] == 0.0
    assert "reservation-1" in projection.pending
    with pytest.raises(state.ActorStateRefused, match="duplicated"):
        state.project_events([_intent(), _intent()])


def test_exact_finish_moves_reservation_to_spent_and_backend_availability():
    projection = state.project_events([_intent(), _finish()])
    assert not projection.pending
    assert projection.reserved["actor_calls_per_target"] == 0.0
    assert projection.spent["actor_calls_per_target"] == 1.0
    assert projection.spent["provider_seconds_per_target"] == 2.5
    assert projection.availability["codex:model@low"]["last_success"] == 90.0
    with pytest.raises(state.ActorStateRefused, match="must not be appended twice"):
        state.project_events([_intent(), _finish(), _finish()])


@pytest.mark.parametrize("mutation", ["nan_budget", "missing_dimension", "wrong_charge",
                                       "binding", "failure_status"])
def test_closed_finite_events_and_exact_settlement(mutation):
    intent = _intent()
    finish = _finish()
    if mutation == "nan_budget":
        intent["budgets"]["provider_seconds_per_target"] = float("nan")
    elif mutation == "missing_dimension":
        intent["debits"].pop("contamination_events_per_target")
    elif mutation == "wrong_charge":
        finish["charges"]["provider_seconds_per_target"] = 3.0
    elif mutation == "binding":
        finish["transition_id"] = "b" * 64
    else:
        finish["status"] = "failed"
    with pytest.raises(state.ActorStateRefused):
        state.project_events([intent, finish])


def test_mutating_returned_input_does_not_hide_digest_difference():
    event = _intent()
    first = state.digest(event)
    changed = copy.deepcopy(event)
    changed["backend_key"] = "codex:other@low"
    assert state.digest(changed) != first


def test_native_journal_route_accepts_only_closed_actor_events(tmp_path):
    owner = journal.Journal(str(tmp_path / "journal"), campaign_id="campaign")
    owner.initialize()
    entry = owner.append(journal.KIND_ACTOR_PREPARATION, _intent())
    assert entry.payload["reservation_id"] == "reservation-1"
    malformed = _intent()
    malformed["extra"] = True
    with pytest.raises(ValueError, match="actor preparation event"):
        owner.append(journal.KIND_ACTOR_PREPARATION, malformed)


def test_controller_selected_intent_finish_and_restart_projection(tmp_path):
    import json
    import os
    import stat
    from . import target_profile_execution as profile_execution
    from .test_actor_lifecycle import _publish_test_owner_accessors
    from .test_target_profile_execution import _sha
    from .test_worker_lifecycle import MockProvider

    resolved = _resolved()
    binary = _fake_executable(tmp_path)
    profiles = _profiles(binary)
    request = _request(profiles["gpt-5.6-sol"])
    request_row = request.to_dict()
    request_digest = actor_preparation._digest(request_row)
    stage_plan_digest = request_row["cache_key"]
    selected_stage = "b" * 64
    issued = {
        "catalog_id": "c" * 64, "transition_id": "d" * 64,
        "selection": {"status": "selected", "proposal_digest": selected_stage},
        "catalog": {"work_by_stage_digest": {selected_stage: {
            "kind": "actor_preparation", "payload": request_row,
            "stage_plan_digest": stage_plan_digest,
            "stage_plan_binding": "preparation_contract",
        }}},
    }
    store = tmp_path / "store"
    (tmp_path / "containers").mkdir()
    controller = campaign_control.CampaignController(
        resolved, store, snapshot_version=2,
        lifecycle_provider=MockProvider(tmp_path / "containers"),
        readiness_check=lambda: (True, None))
    controller.__enter__()
    controller.apply_command(_command(resolved, "resume", "resume", 0))
    _publish_test_owner_accessors(controller)
    now = 20.0
    profile_request = _profile_receipt(resolved, now=now).to_dict()["profile_request"]
    profile_request["stage_proposal"] = {
        "target_revision": "a" * 64, "stage_class": "prerequisite"}
    profile_stage_plan = state.digest(profile_request)
    profile_stage = "e" * 64
    profile_issued = {
        "catalog_id": "f" * 64, "transition_id": "1" * 64,
        "selection": {"status": "selected", "proposal_digest": profile_stage},
        "catalog": {"work_by_stage_digest": {profile_stage: {
            "kind": "profile_preparation", "payload": profile_request,
            "stage_plan_digest": profile_stage_plan,
            "stage_plan_binding": "preparation_contract"}}},
    }
    profile_content = {"hotspots": [{"symbol": "kernel_x", "samples": 1}]}
    loaded_identity = {
        "target_revision_digest": "a" * 64, "model_digest": "3" * 64,
        "quantization": "Q4_K_M", "recipe_digest": "4" * 64,
        "executable_digest": "5" * 64, "dso_digest": "6" * 64,
    }
    artifact_identity = {"kind": "fixture", "sha256": "2" * 64}
    carrier = {
        "schema": "epyc.autokernel.profile_measurement_carrier.v1",
        "profile_source_id": profile_execution.PROFILE_SOURCE_ID,
        "validation_source_id": profile_execution.VALIDATION_SOURCE_ID,
        "run_id": "state-projection-profile",
        "profile_claim_tuple": {"claim_id": "profile"},
        "validation_claim_tuple": {"claim_id": "validation"},
    }
    profile_output = {
        "schema": profile_execution.PROFILE_OUTPUT_SCHEMA,
        "profile_content": profile_content, "loaded_identity": loaded_identity,
        "artifact_identity": artifact_identity, "measurement_carrier": carrier,
    }
    profiler = tmp_path / "profile-fixture"
    profiler.write_text("#!/usr/bin/env python3\nprint(" + repr(json.dumps(profile_output)) + ")\n")
    profiler.chmod(profiler.stat().st_mode | stat.S_IXUSR)
    controller._driver_issued = {profile_issued["catalog_id"]: profile_issued}
    mechanism = profile_execution.ProfileMechanism(
        "state-profile", profiler, _sha(profiler), tmp_path,
        {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}, loaded_identity,
        1.0, 0.5, 4096)
    verified = profile_execution.TargetProfileExecution(
        controller=controller, mechanism=mechanism).prepare(
            profile_request=profile_request, catalog_id=profile_issued["catalog_id"],
            transition_id=profile_issued["transition_id"],
            stage_plan_digest=profile_stage_plan,
            clock_domain=_profile_receipt(resolved, now=now).clock_domain,
            verified_at=now - 1.0, valid_until=now + 30.0)
    controller._driver_issued = {issued["catalog_id"]: issued}
    receipt = verified
    controller._current_actor_profile_receipts.clear()
    with pytest.raises(campaign_control.ControlRefused, match="current producer receipt"):
        controller.reserve_actor_preparation(
            request=request_row, request_digest=request_digest,
            stage_plan_digest=stage_plan_digest,
            actor_profile=profiles["gpt-5.6-sol"].to_dict(),
            actor_profile_digest=profiles["gpt-5.6-sol"].digest,
            target_profile_receipt=verified.to_dict(),
            target_profile_receipt_digest=verified.digest,
            budgets=_budgets().to_dict(), now=now, clock_domain=receipt.clock_domain)
    controller._current_actor_profile_receipts[verified.digest] = verified
    reservation = controller.reserve_actor_preparation(
        request=request_row, request_digest=request_digest,
        stage_plan_digest=stage_plan_digest,
        actor_profile=profiles["gpt-5.6-sol"].to_dict(),
        actor_profile_digest=profiles["gpt-5.6-sol"].digest,
        target_profile_receipt=verified.to_dict(),
        target_profile_receipt_digest=verified.digest,
        budgets=_budgets().to_dict(), now=now, clock_domain=receipt.clock_domain)
    assert reservation["transition_id"] == issued["transition_id"]
    with pytest.raises(campaign_control.ControlRefused, match="unresolved"):
        controller.reserve_actor_preparation(
            request=request_row, request_digest=request_digest,
            stage_plan_digest=stage_plan_digest,
            actor_profile=profiles["gpt-5.6-sol"].to_dict(),
            actor_profile_digest=profiles["gpt-5.6-sol"].digest,
            target_profile_receipt=verified.to_dict(),
            target_profile_receipt_digest=verified.digest,
            budgets=_budgets().to_dict(), now=now, clock_domain=receipt.clock_domain)
    outcome = actor_preparation.StageOutcome(
        reservation["reservation_id"], "completed", "{}", None, 0.25, True, True)
    fabricated = SimpleNamespace(
        request_id=reservation["reservation_id"], plan_digest=stage_plan_digest,
        held_started_at=1.0, held_ended_at=1.25)
    with pytest.raises(campaign_control.ControlRefused,
                       match="provider-authored held cost"):
        controller.finish_actor_preparation(
            reservation=reservation, outcome=outcome.to_dict(),
            disposition="proposal_ready", provider_cost_receipt=fabricated)
    assert reservation["reservation_id"] in controller._actor_preparation_state.pending
    terminal = SimpleNamespace(
        request_id=reservation["reservation_id"], plan_digest=stage_plan_digest,
        worker_id="worker-cost", worker_generation=1, grant_id="grant-cost",
        grant_generation=1, container_id="container-cost")
    held = _held_receipt(terminal, started_at=1.0, ended_at=1.25)
    controller._actor_trusted_held_receipts[reservation["reservation_id"]] = (
        held, terminal)
    controller.finish_actor_preparation(
        reservation=reservation, outcome=outcome.to_dict(), disposition="proposal_ready",
        provider_cost_receipt=held)
    controller.finish_actor_preparation(
        reservation=reservation, outcome=outcome.to_dict(), disposition="proposal_ready",
        provider_cost_receipt=held)
    controller.close()
    rows = [entry for entry in journal.Journal(
        str(store / "journal"), campaign_id=resolved.campaign_id).read_all()
            if entry.kind == journal.KIND_ACTOR_PREPARATION]
    assert [row.payload["event"] for row in rows] == [
        "PROFILE_VERIFIED", "INTENT", "FINISH"]
    with campaign_control.CampaignController(
            resolved, store, snapshot_version=2,
            readiness_check=lambda: (True, None)) as recovered:
        assert reservation["reservation_id"] in recovered._actor_preparation_state.finished
        recovered._driver_issued = {issued["catalog_id"]: issued}
        producer = target_profile_execution.TargetProfileExecution(
            controller=recovered,
            mechanism=target_profile_execution.ProfileMechanism(
                "fixture-profile", binary, hashlib.sha256(binary.read_bytes()).hexdigest(),
                tmp_path, {}, {
                    "target_revision_digest": "a" * 64, "model_digest": "3" * 64,
                    "quantization": "fixture", "recipe_digest": "4" * 64,
                    "executable_digest": "5" * 64, "dso_digest": "6" * 64,
                }, 1.0, 0.5, 4096))
        recovered.register_target_profile_producer(producer)
        refreshed = recovered.verified_target_profile(
            request=request_row, request_digest=request_digest,
            stage_plan_digest=stage_plan_digest,
            campaign_digest=recovered.config_digest,
            target_revision_digest="a" * 64, now=now,
            clock_domain=receipt.clock_domain)
        assert refreshed == verified
        changed_budgets = _budgets().to_dict()
        changed_budgets["actor_calls_per_campaign"] += 1
        with pytest.raises(campaign_control.ControlRefused,
                           match="durable campaign configuration"):
            recovered.reserve_actor_preparation(
                request=request_row, request_digest=request_digest,
                stage_plan_digest=stage_plan_digest,
                actor_profile=profiles["gpt-5.6-sol"].to_dict(),
                actor_profile_digest=profiles["gpt-5.6-sol"].digest,
                target_profile_receipt=refreshed.to_dict(),
                target_profile_receipt_digest=refreshed.digest,
                budgets=changed_budgets, now=now, clock_domain=receipt.clock_domain)


@pytest.mark.xfail(
    raises=scheduling.SchedulingRefused,
    reason=("StageRequest/provider receipt lacks selected proposal_id/backend/stage_class "
            "binding required for unified settlement"),
    strict=True)
def test_actual_driver_profile_requires_native_scheduler_binding(tmp_path):
    driver, _fixture_engine, enrolled, target, target_digest = runtime_driver()
    base_config, _ = make_scheduler()
    scheduler_config = scheduling.SchedulerConfig.from_dict(
        base_config.to_dict() | {"config_id": enrolled.campaign_id})
    engine = scheduling.SchedulerEngine(
        scheduler_config, scheduling.initial_state(scheduler_config, enrolled.campaign_id))
    driver.scheduler = engine
    cost = scheduling.ResourceVector(1.0, (), 0)
    profile_stage = scheduling.StageProposal(
        proposal_id="profile:actual", submitted_at=1, backend="cpu",
        target_revision=target_digest, alias_identity=target.workload_signature,
        frontier_id=target_digest, production_frontier=True, seed_id=None,
        stage_class="prerequisite", estimated_duration_seconds=10,
        estimated_claims=cost, eligible=True, eligibility_ref="2" * 64,
        reservation_kind=None, full_region=True,
        compatibility_authority_refs=(), safe_chunking_declared=False)
    profile_request = unified_driver.ProfilePreparationRequest.from_dict({
        "schema": unified_driver.PROFILE_REQUEST_SCHEMA,
        "target_revision_digest": target_digest,
        "stage_proposal": profile_stage.to_dict(),
        "profile_contract": {"schema": unified_driver.PROFILE_CONTRACT_SCHEMA,
                             "adapter_id": "fixture-profile",
                             "adapter_digest": "9" * 64}})
    driver.profiles = {}
    driver.profile_requests = {target_digest: profile_request}
    containers = tmp_path / "containers"
    containers.mkdir()
    provider = ReceiptProvider(containers)
    store = tmp_path / "actual-driver-store"
    controller = campaign_control.CampaignController(
        enrolled, store, snapshot_version=3, scheduler_engine=engine,
        lifecycle_provider=provider, readiness_check=lambda: (True, None))
    controller.__enter__()
    controller.apply_command(_command(enrolled, "resume", "resume", 0))
    driver.controller = controller
    issued_profile = driver.tick(now=1)
    selected_profile = driver.materialize_profile(issued_profile)
    loaded = {
        "target_revision_digest": target_digest, "model_digest": "1" * 64,
        "quantization": "fixture", "recipe_digest": "2" * 64,
        "executable_digest": "3" * 64, "dso_digest": "4" * 64}
    output = {
        "schema": target_profile_execution.PROFILE_OUTPUT_SCHEMA,
        "profile_content": {"hotspots": [{"symbol": "kernel_x", "samples": 2}]},
        "loaded_identity": loaded,
        "artifact_identity": {"kind": "profile-json", "sha256": "5" * 64},
        "measurement_carrier": {
            "schema": "epyc.autokernel.profile_measurement_carrier.v1",
            "profile_source_id": target_profile_execution.PROFILE_SOURCE_ID,
            "validation_source_id": target_profile_execution.VALIDATION_SOURCE_ID,
            "run_id": "actual-profile-run",
            "profile_claim_tuple": {"claim_id": "profile"},
            "validation_claim_tuple": {"claim_id": "validation"}}}
    profiler = tmp_path / "fake-profiler"
    profiler.write_text("#!/usr/bin/env python3\nimport json\nprint(" + repr(
        __import__("json").dumps(output)) + ")\n")
    profiler.chmod(profiler.stat().st_mode | stat.S_IXUSR)
    producer = target_profile_execution.TargetProfileExecution(
        controller=controller, mechanism=target_profile_execution.ProfileMechanism(
            "fixture-profile", profiler, hashlib.sha256(profiler.read_bytes()).hexdigest(),
            tmp_path, {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}, loaded,
            1.0, 0.5, 4096))
    now = time.monotonic()
    receipt = producer.prepare(
        profile_request=selected_profile.profile_request.to_dict(),
        catalog_id=selected_profile.catalog_id,
        transition_id=selected_profile.transition_id,
        stage_plan_digest=selected_profile.stage_plan_digest,
        clock_domain=worker_lifecycle.monotonic_clock_domain(),
        verified_at=now, valid_until=now + 30.0,
        selected_work=selected_profile)
    controller.register_target_profile_producer(producer)
    selection = scheduling.Selection.from_dict(issued_profile.selection)
    request_digest = state.digest(selected_profile.profile_request.to_dict())
    terminal = controller.worker_terminal_for_request(
        request_id="profile-" + request_digest[:24],
        plan_digest=selected_profile.stage_plan_digest,
        lineage_id=selected_profile.transition_id,
        stage_id="target-profile-" + request_digest[:24])
    assert terminal is not None
    held = controller.actor_held_claim_receipt(terminal)

    class ExactFixtureSettlementValidator:
        def __call__(self, value):
            assert value["receipt"] == held.to_dict()
            assert value["terminal_refs"] == [receipt.verifier_ref]
            return value

    controller.register_unified_settlement_validator(ExactFixtureSettlementValidator())
    controller.unified_driver_settle({
        "schema": campaign_control.DRIVER_SETTLEMENT_SCHEMA,
        "catalog_id": selected_profile.catalog_id,
        "transition_id": selected_profile.transition_id,
        "selection": selection.to_dict(), "receipt": held.to_dict(),
        "outcome": "prerequisite", "terminal_refs": [receipt.verifier_ref]})
