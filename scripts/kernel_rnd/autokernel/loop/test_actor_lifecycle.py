"""Concrete adapter tests through CampaignController and its native Journal."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import time
from types import MethodType, SimpleNamespace

import pytest

from .. import journal as journal_module
from . import actor_lifecycle, actor_preparation, campaign_control, scheduling, worker_lifecycle
from .test_actor_preparation import _budgets, _fake_executable, _profiles, _request, _resolved
from .test_campaign_control import _command
from .test_worker_lifecycle import MockProvider


class SelectedPersistence:
    """Fixture for the still-primary-owned selected/profile/budget Journal seam."""

    def __init__(self, request, stage_digest, target_profile_digest,
                 *, control_revision=1):
        self.request = request.to_dict()
        self.stage_digest = stage_digest
        self.target_profile_digest = target_profile_digest
        self.control_revision = control_revision
        self.events = []
        self.receipts = []

    def reserve_actor_preparation(
            self, *, request, request_digest, stage_plan_digest, actor_profile,
            actor_profile_digest, target_profile_receipt,
            target_profile_receipt_digest, budgets, now, clock_domain):
        assert request == self.request
        assert request_digest == actor_preparation._digest(request)
        assert stage_plan_digest == self.stage_digest
        assert budgets == _budgets().to_dict()
        assert actor_preparation._digest(target_profile_receipt) == target_profile_receipt_digest
        self.receipts.append((dict(target_profile_receipt), target_profile_receipt_digest))
        # This event precedes WorkerLifecycle's OWNED_LAUNCH_INTENT and stands in
        # for the primary-owned durable callback that Rev10 explicitly reserves.
        self.events.append(("PREPARATION_INTENT", request_digest,
                            self.target_profile_digest, dict(budgets)))
        return {"schema": actor_preparation.RESERVATION_SCHEMA,
                "reservation_id": f"reservation-{len(self.events)}-{actor_profile['role']}",
                "request_digest": request_digest, "stage_plan_digest": stage_plan_digest,
                "transition_id": "selected-controller-transition",
                "target_profile_digest": self.target_profile_digest,
                "target_profile_receipt_digest": target_profile_receipt_digest,
                "actor_profile_digest": actor_profile_digest,
                "deadline": now + 10.0, "clock_domain": clock_domain,
                "control_revision": self.control_revision}

    def finish_actor_preparation(
            self, *, reservation, outcome, disposition, provider_cost_receipt):
        if outcome["charged_seconds"]:
            assert provider_cost_receipt is not None
        self.events.append(("PREPARATION_FINISH", reservation["reservation_id"],
                            outcome["charged_seconds"], disposition))


def _profile_receipt(resolved, *, now, valid_until=None, target="a" * 64):
    profile_request = {
        "schema": "epyc.autokernel.profile_preparation_request.v1",
        "target_revision_digest": target,
        "stage_proposal": {"opaque": "exact-primary-owned-scheduler-proposal"},
        "profile_contract": {"opaque": "exact-primary-owned-adapter-contract"},
    }
    return actor_lifecycle.TargetProfileReceipt(
        campaign_digest=actor_preparation._digest(resolved.to_dict()),
        profile_request=profile_request,
        profile_request_digest=actor_preparation._digest(profile_request),
        target_revision_digest=target, target_profile_digest="8" * 64,
        verified_at=now - 1.0,
        valid_until=now + 30.0 if valid_until is None else valid_until,
        clock_domain=worker_lifecycle.monotonic_clock_domain(),
        verifier_ref="primary:profile-preparation:fixture")


class VerifiedProfileOwner:
    def __init__(self, receipt):
        self.receipt = receipt
        self.requests = []

    def verified_target_profile(self, **kwargs):
        self.requests.append(dict(kwargs))
        return self.receipt


def _held_receipt(terminal, *, started_at=10.0, ended_at=10.25):
    return scheduling.HeldClaimReceipt(
        receipt_id=f"receipt-{terminal.worker_id}", proposal_id=terminal.request_id,
        backend="fixture", stage_class="prerequisite", started_at=started_at,
        ended_at=ended_at, ownership_generation=terminal.worker_generation,
        allocation_generation=terminal.grant_generation, physical_claim_ids=(),
        physical_region_fraction=0.0, gpu_device_ids=(), memory_reservation_bytes=0,
        affinity_cores=(), beneficiary_shares={terminal.request_id: 1.0})


def _publish_test_owner_accessors(controller):
    """Fixture form of the exact narrow public accessors requested from primary."""

    actual_run_worker_stage = controller.run_worker_stage
    terminals = {}
    controller._test_held_receipt_calls = 0

    def run_worker_stage(request):
        terminal = actual_run_worker_stage(request)
        terminals[(request.request_id, request.plan_digest,
                   request.lineage_id, request.stage_id)] = terminal
        return terminal

    def terminal_for_request(self, *, request_id, plan_digest, lineage_id, stage_id):
        return terminals.get((request_id, plan_digest, lineage_id, stage_id))

    def read_worker_stdout(
            self, *, request_id, plan_digest, lineage_id, stage_id, worker_id,
            worker_generation, result_digest, max_bytes):
        terminal = terminal_for_request(
            self, request_id=request_id, plan_digest=plan_digest,
            lineage_id=lineage_id, stage_id=stage_id)
        assert terminal is not None
        assert (terminal.worker_id, terminal.worker_generation, terminal.result_digest) == (
            worker_id, worker_generation, result_digest)
        leaf = hashlib.sha256(f"{worker_id}:{worker_generation}".encode()).hexdigest()[:24]
        return self._runtime_root.read_bytes(
            f"worker-{leaf}.stdout.log", limit=max_bytes)

    def worker_held_claim_receipt(self, terminal):
        self._test_held_receipt_calls += 1
        receipt = _held_receipt(terminal)
        return receipt

    controller.run_worker_stage = run_worker_stage
    controller.worker_terminal_for_request = MethodType(terminal_for_request, controller)
    controller.worker_held_claim_receipt = MethodType(worker_held_claim_receipt, controller)
    controller.read_worker_stdout = MethodType(read_worker_stdout, controller)


def test_actual_worker_lifecycle_runs_backend_and_retains_native_journal(tmp_path):
    binary = _fake_executable(tmp_path)
    profiles = _profiles(binary)
    actor_request = _request(profiles["gpt-5.6-sol"])
    stage_digest = "7" * 64
    target_profile_digest = "8" * 64
    persistence = SelectedPersistence(actor_request, stage_digest, target_profile_digest)
    (tmp_path / "containers").mkdir()
    provider = MockProvider(tmp_path / "containers")
    resolved = _resolved()
    controller = campaign_control.CampaignController(
        resolved, tmp_path / "store", snapshot_version=2,
        lifecycle_provider=provider, readiness_check=lambda: (True, None))
    controller.__enter__()
    controller.apply_command(_command(resolved, "resume", "resume", 0))
    _publish_test_owner_accessors(controller)
    journal = journal_module.Journal(
        str(tmp_path / "store" / "journal"), campaign_id=resolved.campaign_id)
    config = actor_lifecycle.ActorLifecycleConfig(
        campaign_digest=actor_preparation._digest(resolved.to_dict()),
        cwd=tmp_path, env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        max_stage_seconds=1.0, teardown_seconds=0.5,
        max_retained_output_bytes=4096)
    profile_owner = VerifiedProfileOwner(
        _profile_receipt(resolved, now=time.monotonic()))
    adapter = actor_lifecycle.ActorLifecycleAdapter(
        controller=controller, persistence=persistence, config=config,
        target_profile_owner=profile_owner)
    consumer = actor_preparation.ActorPreparationConsumer(
        resolved_campaign=resolved, profiles=profiles, budgets=_budgets(),
        capability=adapter, clock=time.monotonic,
        clock_domain=worker_lifecycle.monotonic_clock_domain(), max_output_bytes=4096)
    try:
        result = consumer.prepare(actor_request, stage_plan_digest=stage_digest)
        assert result.status == "proposed"
        assert result.proposed_output["target_symbol"] == "kernel_x"
        assert [event[0] for event in persistence.events] == [
            "PREPARATION_INTENT", "PREPARATION_FINISH",
            "PREPARATION_INTENT", "PREPARATION_FINISH"]
        entries = journal.read_all()
        lifecycle_events = [entry.payload["event"] for entry in entries
                            if entry.kind == journal_module.KIND_WORKER_LIFECYCLE]
        assert lifecycle_events.count("OWNED_CHILD_CAPTURED") == 2
        assert lifecycle_events.count("WORKER_RESULT_ACCEPTED") == 2
        assert lifecycle_events.count("OWNED_TERMINAL") == 2
        assert provider.authorization is not None
        assert not provider.authorization.container.populated()
        assert len(persistence.receipts) == 2
        assert len(profile_owner.requests) == 2
        assert all(item["target_revision_digest"] == "a" * 64
                   and item["campaign_digest"] == config.campaign_digest
                   for item in profile_owner.requests)
    finally:
        controller.close()


def test_adapter_contains_no_competing_process_launcher():
    source = Path(actor_lifecycle.__file__).read_text()
    assert "subprocess" not in source
    assert "Popen" not in source
    assert "self.lifecycle" not in source
    assert ".run_stage(" not in source
    assert "RuntimeRoot" not in source
    assert ".run_worker_stage(request)" in source


@pytest.mark.parametrize("held_ended_at", [20.75, 20.0])
def test_exact_accepted_nonzero_terminal_requires_provider_hold_before_failure(
        tmp_path, held_ended_at):
    binary = _fake_executable(tmp_path)
    profiles = _profiles(binary)
    profile = profiles["gpt-5.6-sol"]
    actor_request = _request(profile)
    resolved = _resolved()
    controller = campaign_control.CampaignController(
        resolved, tmp_path / "store", snapshot_version=2)
    controller.__enter__()
    _publish_test_owner_accessors(controller)
    persistence = SelectedPersistence(actor_request, "7" * 64, "8" * 64)
    now = time.monotonic()
    adapter = actor_lifecycle.ActorLifecycleAdapter(
        controller=controller, persistence=persistence,
        config=actor_lifecycle.ActorLifecycleConfig(
            campaign_digest=actor_preparation._digest(resolved.to_dict()), cwd=tmp_path,
            env={}, max_stage_seconds=1.0, teardown_seconds=0.5,
            max_retained_output_bytes=4096),
        target_profile_owner=VerifiedProfileOwner(_profile_receipt(resolved, now=now)))
    reservation = actor_preparation.StageReservation.from_dict(adapter.reserve(
        request=actor_request.to_dict(), stage_plan_digest="7" * 64,
        actor_profile=profile.to_dict(), budgets=_budgets().to_dict(), now=now,
        clock_domain=worker_lifecycle.monotonic_clock_domain()))
    seen = {}

    def run_stage(request):
        terminal = SimpleNamespace(
            request_id=request.request_id, plan_digest=request.plan_digest,
            lineage_id=request.lineage_id, stage_id=request.stage_id,
            worker_id="worker-nonzero", worker_generation=1,
            grant_id="grant-nonzero", grant_generation=1,
            container_id="container-nonzero", result_digest="1" * 64,
            return_code=9, accepted=True)
        seen["terminal"] = terminal
        return terminal

    controller.run_worker_stage = run_stage
    controller.worker_terminal_for_request = lambda **_kwargs: seen["terminal"]
    def held_cost(terminal):
        if held_ended_at == 20.0:
            return SimpleNamespace(started_at=20.0, ended_at=20.0)
        receipt = _held_receipt(terminal, started_at=20.0, ended_at=held_ended_at)
        return receipt

    controller.worker_held_claim_receipt = held_cost
    try:
        if held_ended_at == 20.0:
            with pytest.raises(actor_lifecycle.ActorInvocationUncertain,
                               match="cannot prove exact provider-held accounting"):
                adapter.invoke(reservation, profile.backend(), "prompt")
            assert [event[0] for event in persistence.events] == ["PREPARATION_INTENT"]
            return
        outcome = actor_preparation.StageOutcome.from_dict(
            adapter.invoke(reservation, profile.backend(), "prompt"))
        assert (outcome.status, outcome.failure_class, outcome.charged_seconds) == (
            "failed", "process_exit", 0.75)
        adapter.finish(reservation, outcome, "actor_failed")
        assert persistence.events[-1] == (
            "PREPARATION_FINISH", reservation.reservation_id, 0.75, "actor_failed")
    finally:
        controller.close()


@pytest.mark.parametrize("owner_state", ["paused", "drained", "closed"])
def test_campaign_controller_owner_refuses_closed_admission_with_native_journal(
        tmp_path, owner_state):
    binary = _fake_executable(tmp_path)
    profiles = _profiles(binary)
    actor_request = _request(profiles["gpt-5.6-sol"])
    resolved = _resolved()
    (tmp_path / "containers").mkdir()
    provider = MockProvider(tmp_path / "containers")
    store = tmp_path / "store"
    controller = campaign_control.CampaignController(
        resolved, store, snapshot_version=2, lifecycle_provider=provider,
        readiness_check=lambda: (True, None))
    controller.__enter__()
    _publish_test_owner_accessors(controller)
    if owner_state in {"drained", "closed"}:
        controller.apply_command(_command(resolved, "resume", "resume", 0))
    if owner_state == "drained":
        controller.apply_command(_command(resolved, "drain", "drain", 1))
    persistence = SelectedPersistence(
        actor_request, "7" * 64, "8" * 64,
        control_revision=controller.control_revision)
    config = actor_lifecycle.ActorLifecycleConfig(
        campaign_digest=actor_preparation._digest(resolved.to_dict()), cwd=tmp_path,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        max_stage_seconds=1.0, teardown_seconds=0.5,
        max_retained_output_bytes=4096)
    adapter = actor_lifecycle.ActorLifecycleAdapter(
        controller=controller, persistence=persistence, config=config,
        target_profile_owner=VerifiedProfileOwner(
            _profile_receipt(resolved, now=time.monotonic())))
    consumer = actor_preparation.ActorPreparationConsumer(
        resolved_campaign=resolved, profiles=profiles, budgets=_budgets(),
        capability=adapter, clock=time.monotonic,
        clock_domain=worker_lifecycle.monotonic_clock_domain(), max_output_bytes=4096)
    if owner_state == "closed":
        controller.close()
    try:
        with pytest.raises(actor_lifecycle.ActorInvocationUncertain,
                           match="negative-admission proof"):
            consumer.prepare(actor_request, stage_plan_digest="7" * 64)
        assert [event[0] for event in persistence.events] == ["PREPARATION_INTENT"]
        assert provider.authorization is None
        entries = journal_module.Journal(
            str(store / "journal"), campaign_id=resolved.campaign_id).read_all()
        assert not any(entry.kind == journal_module.KIND_WORKER_LIFECYCLE
                       for entry in entries)
        if owner_state == "drained":
            assert any(entry.kind == journal_module.KIND_CAMPAIGN_COMMAND_V2
                       for entry in entries)
    finally:
        if owner_state != "closed":
            controller.close()


@pytest.mark.parametrize(
    "receipt_kind", ["absent_owner", "missing", "stale", "mismatch", "arbitrary"])
def test_unverified_target_profile_never_reaches_persistence_or_run_stage(
        tmp_path, receipt_kind):
    binary = _fake_executable(tmp_path)
    profiles = _profiles(binary)
    actor_request = _request(profiles["gpt-5.6-sol"])
    resolved = _resolved()
    persistence = SelectedPersistence(actor_request, "7" * 64, "8" * 64)
    controller = campaign_control.CampaignController(
        resolved, tmp_path / "store", snapshot_version=2)
    controller.__enter__()
    _publish_test_owner_accessors(controller)
    now = 50.0
    receipt = None
    if receipt_kind == "stale":
        receipt = _profile_receipt(resolved, now=now, valid_until=now)
    elif receipt_kind == "mismatch":
        receipt = _profile_receipt(resolved, now=now, target="b" * 64)
    elif receipt_kind == "arbitrary":
        receipt = {"target_profile_digest": "8" * 64}
    config = actor_lifecycle.ActorLifecycleConfig(
        campaign_digest=actor_preparation._digest(resolved.to_dict()), cwd=tmp_path,
        env={}, max_stage_seconds=1.0, teardown_seconds=0.5,
        max_retained_output_bytes=4096)
    try:
        profile_owner = (None if receipt_kind == "absent_owner"
                         else VerifiedProfileOwner(receipt))
        adapter = actor_lifecycle.ActorLifecycleAdapter(
            controller=controller, persistence=persistence, config=config,
            target_profile_owner=profile_owner)
        consumer = actor_preparation.ActorPreparationConsumer(
            resolved_campaign=resolved, profiles=profiles, budgets=_budgets(),
            capability=adapter, clock=lambda: now,
            clock_domain=worker_lifecycle.monotonic_clock_domain(), max_output_bytes=4096)
        with pytest.raises(actor_preparation.PreparationRefused,
                           match="target-profile"):
            consumer.prepare(actor_request, stage_plan_digest="7" * 64)
        assert persistence.events == []
    finally:
        controller.close()
