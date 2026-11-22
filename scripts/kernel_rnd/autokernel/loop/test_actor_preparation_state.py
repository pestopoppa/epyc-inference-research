"""Closed actor-preparation native event and recovery tests."""
from __future__ import annotations

import copy
import os
import time
from types import SimpleNamespace

import pytest

from .. import journal
from . import actor_preparation, actor_preparation_state as state
from . import campaign_control, scheduling, unified_driver, worker_lifecycle
from .test_actor_preparation import _budgets

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

def _held_by_request(controller):
    """Provider-authored held facts retained by the controller's own worker lifecycle."""
    return {row.payload["request_id"]: row.payload["data"]["receipt"]
            for row in controller._journal.read_all()
            if row.kind == journal.KIND_WORKER_LIFECYCLE
            and row.payload["event"] == "OWNED_HELD_RECEIPT_RETAINED"}


def _public_actor_sequence(tmp_path):
    """AKU-06d: installed producer -> driver-issued profile -> genuine settlement ->
    driver-issued actor work -> contained fixture planner/critic children.

    Every authority comes from the public owner path: the installed profile binding
    registers the producer, the runtime settles through its trusted verifier, actor
    reservations/finishes use the provider-authored held cost of each owned terminal.
    No issuance, current-receipt or trusted-held projection is seeded."""
    from ..execution import worktree
    from . import actor_lifecycle
    from .test_profile_preparation_runtime import _start
    from .test_source_build_execution import _git, _profiled_build_runtime
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    repo_root = tmp_path / "source"
    repo_root.mkdir()
    repo, snapshot, head = _git(repo_root)
    plan = worktree.BuildPlan(
        snapshot.path,
        worktree.SandboxPath.create(repo_root / "build", sandbox_root=repo_root,
                                    production_trees=()),
        worktree.SandboxPath.create(repo_root / "actor", sandbox_root=repo_root,
                                    production_trees=()),
        worktree.BuildParallelism(1, cpu_list="0"), targets=("fixture",),
        cmake_defines=(("FIXTURE", "ON"),))
    materialized, registry, target, actor_profiles = _profiled_build_runtime(
        runtime_root, source_revision=head, build_plan=plan)
    controller, runtime = _start(materialized, registry)
    assert runtime.tick().status == "settled"
    assert controller.current_verified_profile_result(target)["settlement"] is not None
    driver = unified_driver.UnifiedCampaignDriver(
        resolved_campaign=runtime.driver.resolved, controller=controller,
        scheduler_engine=runtime.driver.scheduler,
        profiles=runtime.profile_executor.planner_profiles(time.monotonic()),
        evidence_index=runtime.driver.evidence,
        runtime_anchors=runtime.driver.runtime_anchors,
        runtime_dimensions=runtime.driver.runtime_dimensions,
        experiment_plans=runtime.driver.experiment_plans,
        profile_requests=runtime.driver.profile_requests,
        actor_identities=runtime.driver.actor_identities,
        native_artifact_sink_ref=runtime.driver.sink_ref,
        execution_inputs=runtime.driver.execution_inputs,
        executable_work_kinds={"actor_preparation"})
    selected = driver.materialize_actor(driver.tick(now=time.monotonic()))
    lifecycle = actor_lifecycle.ActorLifecycleAdapter(
        controller=controller, persistence=controller,
        config=actor_lifecycle.ActorLifecycleConfig(
            campaign_digest=controller.config_digest, cwd=runtime_root,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            max_stage_seconds=5.0, teardown_seconds=1.0, max_retained_output_bytes=4096),
        target_profile_owner=controller)
    consumer = actor_preparation.ActorPreparationConsumer(
        resolved_campaign=runtime.driver.resolved, profiles=actor_profiles,
        budgets=_budgets(), capability=lifecycle, clock=time.monotonic,
        clock_domain=worker_lifecycle.monotonic_clock_domain(), max_output_bytes=4096)
    result = consumer.prepare(selected.actor_request, stage_plan_digest=selected.stage_plan_digest)
    return SimpleNamespace(
        controller=controller, runtime=runtime, materialized=materialized, registry=registry,
        target=target, selected=selected, result=result, actor_profiles=actor_profiles,
        cleanup=lambda: repo.remove_worktree(snapshot.path, force=True))


def _close(sequence):
    sequence.runtime.close()
    sequence.controller.close()


def _settle_actor_operation(sequence):
    """Settle the selected operation once from its durable original planner/critic children.

    No build child ran in this preparation-only scope, so the closed outcome rules allow
    only a non-prerequisite settlement; the spent child costs stay durable."""
    controller, selected = sequence.controller, sequence.selected
    children = controller.unified_driver_children(selected.transition_id)
    assert [child["role"] for child in children] == ["planner", "critic"]
    return controller.unified_driver_settle_v2({
        "schema": campaign_control.DRIVER_SETTLEMENT_SCHEMA_V2,
        "catalog_id": selected.catalog_id, "transition_id": selected.transition_id,
        "selection": selected.selection.to_dict(), "outcome": "failed",
        "children": [child["digest"] for child in children], "denial_ref": None})


def _restart(sequence):
    from .test_profile_preparation_runtime import _start
    _close(sequence)
    sequence.controller, sequence.runtime = _start(sequence.materialized, sequence.registry)
    return sequence.controller


def _reserve(controller, sequence, receipt, budgets):
    request = sequence.selected.actor_request.to_dict()
    planner = sequence.actor_profiles["actor-planner"]
    return controller.reserve_actor_preparation(
        request=request, request_digest=state.digest(request),
        stage_plan_digest=sequence.selected.stage_plan_digest,
        actor_profile=planner.to_dict(), actor_profile_digest=planner.digest,
        target_profile_receipt=receipt.to_dict(), target_profile_receipt_digest=receipt.digest,
        budgets=budgets, now=time.monotonic(), clock_domain=receipt.clock_domain)


def test_driver_issued_profile_then_actor_preparation_finish_and_restart(tmp_path):
    from . import actor_lifecycle
    sequence = _public_actor_sequence(tmp_path)
    try:
        controller = sequence.controller
        assert sequence.result.status == "proposed"
        rows = [row.payload for row in controller._journal.read_all()
                if row.kind == journal.KIND_ACTOR_PREPARATION]
        assert [row["event"] for row in rows] == [
            "PROFILE_VERIFIED", "INTENT", "FINISH", "INTENT", "FINISH"]
        receipt = controller.current_actor_profile(sequence.target)
        held = _held_by_request(controller)
        for intent, finish in ((rows[1], rows[2]), (rows[3], rows[4])):
            assert intent["catalog_id"] == sequence.selected.catalog_id
            assert intent["transition_id"] == sequence.selected.transition_id
            assert intent["target_profile_receipt_digest"] == receipt.digest
            assert intent["target_profile_digest"] == rows[0]["target_profile_digest"]
            cost = held[intent["reservation_id"]]
            # exact held cost authored by the provider for this owned terminal
            assert finish["charged_seconds"] == cost["ended_at"] - cost["started_at"] > 0
            assert finish["status"] == "completed"
        # a schema-valid but non-current receipt is not producer authority
        forged = actor_lifecycle.TargetProfileReceipt(
            **(receipt.to_dict() | {"verifier_ref": "caller:forged-profile"}))
        with pytest.raises(campaign_control.ControlRefused, match="current producer receipt"):
            _reserve(controller, sequence, forged, _budgets().to_dict())
        settled = _settle_actor_operation(sequence)
        assert settled["status"] == "accepted"
        with pytest.raises(campaign_control.ControlRefused, match="one exact issued selected"):
            _reserve(controller, sequence, receipt, _budgets().to_dict())
        spent = controller.actor_preparation_projection().spent
        finished = set(controller.actor_preparation_projection().finished)
        durable = copy.deepcopy(controller._driver_settled[sequence.selected.transition_id])
        recovered = _restart(sequence)
        projection = recovered.actor_preparation_projection()
        assert set(projection.finished) == finished and not projection.pending
        assert projection.spent == spent
        assert projection.spent["actor_calls_per_campaign"] == 2.0
        assert recovered._driver_settled[sequence.selected.transition_id] == durable
        assert _settle_actor_operation(sequence)["status"] == "duplicate"
    finally:
        _close(sequence)
        sequence.cleanup()


def test_durable_budget_drift_refuses_with_valid_current_producer_receipt(tmp_path):
    sequence = _public_actor_sequence(tmp_path)
    try:
        controller = sequence.controller
        request = sequence.selected.actor_request.to_dict()
        receipt = controller.verified_target_profile(
            request=request, request_digest=state.digest(request),
            stage_plan_digest=sequence.selected.stage_plan_digest,
            campaign_digest=controller.config_digest,
            target_revision_digest=sequence.target, now=time.monotonic(),
            clock_domain=worker_lifecycle.monotonic_clock_domain())
        assert receipt is not None  # valid current producer authority
        drifted = _budgets().to_dict()
        drifted["actor_calls_per_campaign"] += 1
        with pytest.raises(campaign_control.ControlRefused,
                           match="durable campaign configuration"):
            _reserve(controller, sequence, receipt, drifted)
        # the unchanged durable budgets pass the drift gate and reach reinvocation refusal
        with pytest.raises(campaign_control.ControlRefused, match="already settled"):
            _reserve(controller, sequence, receipt, _budgets().to_dict())
        assert not controller.actor_preparation_projection().pending
    finally:
        _close(sequence)
        sequence.cleanup()


def test_actual_driver_profile_requires_native_scheduler_binding(tmp_path):
    from .test_target_profile_execution import prepare_selected, selected_profile_owner
    owner = selected_profile_owner(tmp_path)
    controller = owner.controller
    try:
        receipt = prepare_selected(owner)
        selected_profile = owner.selected
        selection = scheduling.Selection.from_dict(owner.issued.selection)
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
    finally:
        controller.close()
