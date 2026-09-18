"""Public, non-authoritative selected profile materialization tests."""
from __future__ import annotations

import copy

import pytest

from . import campaign_control, scheduling, unified_driver as driver
from .test_campaign_control import _command
from .test_unified_driver import runtime_driver
from .test_unified_planner import profile, scheduler


def _profile_driver(tmp_path):
    instance, _old_engine, enrolled, target, target_digest = runtime_driver()
    base, _ = scheduler()
    config = scheduling.SchedulerConfig.from_dict(
        base.to_dict() | {"config_id": enrolled.campaign_id})
    engine = scheduling.SchedulerEngine(
        config, scheduling.initial_state(config, enrolled.campaign_id))
    raw_profile = profile(target)
    cost = scheduling.ResourceVector.from_dict(raw_profile["resource_cost"])
    stage = scheduling.StageProposal(
        proposal_id="profile:public", submitted_at=1.0, backend="cpu",
        target_revision=target_digest, alias_identity=target.workload_signature,
        frontier_id=target_digest, production_frontier=True, seed_id=None,
        stage_class="prerequisite", estimated_duration_seconds=10.0,
        estimated_claims=cost, eligible=True, eligibility_ref="2" * 64,
        reservation_kind=None, full_region=True,
        compatibility_authority_refs=(), safe_chunking_declared=False)
    request = driver.ProfilePreparationRequest.from_dict({
        "schema": driver.PROFILE_REQUEST_SCHEMA,
        "target_revision_digest": target_digest,
        "stage_proposal": stage.to_dict(),
        "profile_contract": {"schema": driver.PROFILE_CONTRACT_SCHEMA,
                             "adapter_id": "fixture-profile",
                             "adapter_digest": "9" * 64}})
    instance.scheduler = engine
    instance.controller = None
    instance.profiles = {}
    instance.profile_requests = {target_digest: request}
    controller = campaign_control.CampaignController(
        enrolled, tmp_path / "controller", snapshot_version=3,
        scheduler_engine=engine, readiness_check=lambda: (True, None))
    controller.__enter__()
    controller.apply_command(_command(enrolled, "resume-profile", "resume", 0))
    instance.controller = controller
    return instance, controller, request, config


def test_actual_tick_materializes_exact_public_profile_advice(tmp_path):
    instance, controller, request, _config = _profile_driver(tmp_path)
    try:
        outcome = instance.tick(now=1.0)
        selected = instance.materialize_profile(outcome)
        assert selected.selection.to_dict() == driver._thaw(outcome.selection)
        assert selected.profile_request == request
        assert selected.stage_plan_digest == driver._digest(request.to_dict())
        assert selected.controller_binding["supervisor_incarnation"] == 1
        assert selected.controller_binding["campaign_id"] == instance.resolved.campaign_id
        assert selected.execution_authorized is False
        assert driver.SelectedProfileWork.from_dict(selected.to_dict()) == selected

        detached = selected.to_dict()
        detached["profile_request"]["profile_contract"]["adapter_id"] = "mutated"
        assert selected.profile_request.profile_contract["adapter_id"] == "fixture-profile"
    finally:
        controller.close()


def test_forged_swapped_and_wrong_kind_outcomes_refuse(tmp_path):
    instance, controller, _request, _config = _profile_driver(tmp_path)
    try:
        outcome = instance.tick(now=1.0)
        malformed = copy.deepcopy(outcome.to_dict())
        malformed["selection"]["proposal"]["backend"] = "gpu"
        forged = driver.DriverOutcome(
            malformed["status"], tuple(malformed["reasons"]), malformed["transition_id"],
            malformed["selection"])
        with pytest.raises(driver.DriverRefused, match="selection is invalid"):
            instance.materialize_profile(forged)
    finally:
        controller.close()

    runtime, _engine, _enrolled, _target, _digest = runtime_driver()
    runtime_outcome = runtime.tick(now=1.0)
    with pytest.raises(driver.DriverRefused, match="not profile preparation"):
        runtime.materialize_profile(runtime_outcome)


def test_closed_or_restarted_owner_refuses_old_profile_outcome(tmp_path):
    instance, controller, _request, config = _profile_driver(tmp_path)
    outcome = instance.tick(now=1.0)
    enrolled = instance.resolved
    store = controller.store
    controller.close()
    replay = scheduling.SchedulerEngine(
        config, scheduling.initial_state(config, enrolled.campaign_id))
    with campaign_control.CampaignController(
            enrolled, store, snapshot_version=3, scheduler_engine=replay) as reopened:
        instance.controller = reopened
        with pytest.raises(driver.DriverRefused, match="controller refused"):
            instance.materialize_profile(outcome)


def test_selected_profile_record_rejects_nested_mutation_and_authority_flip(tmp_path):
    instance, controller, _request, _config = _profile_driver(tmp_path)
    try:
        selected = instance.materialize_profile(instance.tick(now=1.0))
        raw = selected.to_dict()
        raw["execution_authorized"] = True
        with pytest.raises(driver.DriverRefused, match="schema/authority"):
            driver.SelectedProfileWork.from_dict(raw)
        raw = selected.to_dict()
        raw["profile_request"]["stage_proposal"]["stage_class"] = "search"
        with pytest.raises(driver.DriverRefused):
            driver.SelectedProfileWork.from_dict(raw)
    finally:
        controller.close()
