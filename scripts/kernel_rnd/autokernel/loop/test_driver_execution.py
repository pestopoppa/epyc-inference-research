"""Hermetic connector tests; fake provider facts are not live containment proof."""
from __future__ import annotations

import hashlib
import os
import threading
import time

import pytest

from . import campaign_control
from . import driver_execution as de
from . import experiment_plan
from .. import journal as journal_module
from . import planned_serving
from . import scheduling
from . import unified_driver
from . import unified_worker
from . import worker_lifecycle as wl
from .test_unified_driver import runtime_driver
from .test_unified_planner import scheduler
from .test_unified_worker import _measure
from .test_worker_lifecycle import MockOwnedContainer


class HeldProvider:
    """Explicit test provider; its claims prove no real resource ownership."""

    def __init__(self, root):
        self.root = root
        self.authorization = None

    def authorize(self, request, container_id, deadline):
        del request
        grant = wl.GrantReceipt("test-grant", 1, deadline, wl.monotonic_clock_domain())
        self.authorization = wl.AuthorizedLaunch(
            grant, container_id, MockOwnedContainer(self.root, container_id))
        return self.authorization

    def refresh(self, authorization, deadline):
        del deadline
        return authorization.grant

    def release(self, authorization, deadline):
        del authorization, deadline
        return True

    def inspect_pending(self, identity, deadline):
        del identity, deadline
        return wl.PendingAcquisitionInspection(
            "absent", None, "test provider certifies no acquisition")

    def close_held_receipt(self, *, authorization, request, worker_id,
                           worker_generation, container_identity,
                           lifecycle_started_at, released_at, deadline):
        del container_identity, deadline
        receipt = scheduling.HeldClaimReceipt(
            f"held:{worker_id}:{worker_generation}", request.request_id, "cpu", "search",
            lifecycle_started_at, released_at, worker_generation,
            authorization.grant.generation, (f"test:{authorization.container_id}",),
            0.5, (), 1024, ("0",), {request.request_id: 1.0})
        return wl.TrustedHeldClaimReceipt(
            request.request_id, request.plan_digest, worker_id, worker_generation,
            authorization.grant.grant_id, authorization.grant.generation,
            authorization.container_id, authorization.grant.clock_domain,
            lifecycle_started_at, released_at, receipt)


class DirectUnitAuthority:
    """Test-only adapter feeding the same bounded parent evidence cache."""

    def __init__(self, parent, start):
        self.parent, self.start = parent, start

    def admit(self, *, sequence, plan_digest, unit, prior_completion_digest):
        del plan_digest, prior_completion_digest
        return unified_worker.UnitPermit(
            sequence, unit.unit_id, unit.process_id, f"test-fence:{sequence}",
            self.start.provider_deadline, self.start.grant_id,
            self.start.grant_generation, self.start.container_id, True, "test held")

    def complete(self, *, sequence, fence, observation):
        key, completion = self.parent.request_completion(
            start=self.start, sequence=sequence, fence=fence, observation=observation)
        deadline = time.monotonic() + 2
        while completion is None and time.monotonic() < deadline:
            time.sleep(0.001)
            completion = self.parent.poll_completion(key)
        assert completion is not None
        return completion

    def verify_continuation(self, raw, plan, prompts, previous_lineage_id):
        del raw, plan, prompts, previous_lineage_id
        return False


def _resume(controller, campaign_id):
    digest = campaign_control.command_digest(
        operation="resume", payload={}, campaign_id=campaign_id, config_generation=1)
    controller.apply_command({
        "schema": campaign_control.COMMAND_SCHEMA, "campaign_id": campaign_id,
        "config_generation": 1, "request_id": "resume-execution",
        "operation": "resume", "payload": {}, "payload_digest": digest,
        "expected_control_revision": 0})


def _command(controller, operation, request_id):
    payload = {}
    digest = campaign_control.command_digest(
        operation=operation, payload=payload,
        campaign_id=controller.resolved.campaign_id,
        config_generation=controller.config_generation)
    return controller.apply_command({
        "schema": campaign_control.COMMAND_SCHEMA,
        "campaign_id": controller.resolved.campaign_id,
        "config_generation": controller.config_generation,
        "request_id": request_id, "operation": operation, "payload": payload,
        "payload_digest": digest,
        "expected_control_revision": controller.control_revision})


def _open_fds():
    result = {}
    for name in os.listdir("/proc/self/fd"):
        try:
            result[int(name)] = os.readlink(f"/proc/self/fd/{name}")
        except FileNotFoundError:
            pass
    return result


def _owned_stack(tmp_path, monkeypatch, *, return_code=0):
    driver, _old, enrolled, _target, target_digest = runtime_driver(git_source=True)
    original_input = driver.execution_inputs[target_digest]
    prompt_body = original_input.prompt_manifest.to_dict()
    prompt_body["prompts"] = prompt_body["prompts"][:1]
    prompt_body["digest"] = unified_driver._digest({
        key: value for key, value in prompt_body.items() if key != "digest"})
    prompts = planned_serving.FrozenPromptManifest.from_dict(prompt_body)
    driver.execution_inputs = {target_digest: unified_driver.ExecutionInput(
        target_digest, prompts, original_input.max_stage_seconds,
        original_input.teardown_seconds, original_input.instrument_id)}
    plan_key, original_plan = next(iter(driver.experiment_plans.items()))
    plan_body = original_plan.to_dict()
    for unit in plan_body["expected_units"]:
        unit["expected_prompt_ids"] = ["p1"]
    driver.experiment_plans = {plan_key: experiment_plan.ExperimentPlan.from_dict(plan_body)}
    base, _ = scheduler()
    config = scheduling.SchedulerConfig.from_dict(
        base.to_dict() | {"config_id": enrolled.campaign_id})
    engine = scheduling.SchedulerEngine(
        config, scheduling.initial_state(config, enrolled.campaign_id))
    containers = tmp_path / "containers"
    containers.mkdir(mode=0o700)
    provider = HeldProvider(containers)
    controller = campaign_control.CampaignController(
        enrolled, tmp_path / "controller", snapshot_version=3,
        scheduler_engine=engine, readiness_check=lambda: (True, None),
        lifecycle_provider=provider)
    controller.__enter__()
    _resume(controller, enrolled.campaign_id)
    driver.controller = controller
    driver.scheduler = engine

    lifecycle = controller._worker_lifecycle
    assert lifecycle is not None

    def fake_wait(process, fd, nonce, contract_digest, request, authorization, held_grant,
                  deadline, *, planned_invocation, container_identity, worker_id,
                  worker_generation, container_id):
        del fd, authorization
        start = unified_worker.WorkerStart(
            nonce, request.request_id, planned_invocation.prepared.prepared_digest,
            request.plan_digest, request.lineage_id, request.stage_id,
            lifecycle.binding.campaign_id, lifecycle.binding.config_digest,
            lifecycle.binding.config_generation, lifecycle.binding.supervisor_id,
            lifecycle.binding.supervisor_incarnation, worker_id, worker_generation,
            held_grant.grant_id, held_grant.generation, container_id,
            wl.process_identity(os.getpid()), container_identity, held_grant.clock_domain,
            deadline)
        planned_invocation.start = start
        authority = DirectUnitAuthority(planned_invocation.parent_authority, start)
        times = iter(f"2026-09-09T00:00:{index:02d}Z" for index in range(20))
        reference = unified_worker.run_prepared_stage(
            planned_invocation.prepared, start, _test_authority=authority,
            _test_membership_probe=lambda _: None, _test_measure=_measure,
            clock=lambda: 1.0, wall_clock=times.__next__)
        planned_invocation._reference = reference
        planned_invocation._reference_digest = unified_worker._digest(reference.to_dict())
        return {"schema": wl.OUTCOME_SCHEMA, "nonce": nonce,
                "contract_digest": contract_digest, "child_pid": process.pid,
                "return_code": return_code, "forwarded_signal": None, "stdout_bytes": 0,
                "stderr_bytes": 0, "stdout_sha256": hashlib.sha256(b"").hexdigest(),
                "stderr_sha256": hashlib.sha256(b"").hexdigest(),
                "stdout_truncated": False, "stderr_truncated": False}

    monkeypatch.setattr(lifecycle, "_wait_outcome", fake_wait)
    return driver, controller, lifecycle, engine


def test_nonzero_owned_terminal_is_charged_once_without_native_acceptance(
        tmp_path, monkeypatch):
    driver, controller, lifecycle, engine = _owned_stack(
        tmp_path, monkeypatch, return_code=7)
    connector = de.UnifiedDriverExecution(driver=driver, controller=controller)
    calls = {"lifecycle": 0}
    real_run = lifecycle.run_stage

    def count_run(*args, **kwargs):
        calls["lifecycle"] += 1
        return real_run(*args, **kwargs)

    monkeypatch.setattr(lifecycle, "run_stage", count_run)
    real_stop = de.UnknownParentEvidenceProducer.stop_and_join

    def stop_then_report(*args, **kwargs):
        real_stop(*args, **kwargs)
        raise RuntimeError("injected producer shutdown report")

    monkeypatch.setattr(de.UnknownParentEvidenceProducer, "stop_and_join", stop_then_report)
    try:
        issued = driver.tick(now=1.0)
        receipt = connector.execute(issued)
        assert receipt.settlement_request["outcome"] == "failed"
        assert receipt.native_measurement_ids == ()
        assert receipt.result_reference is None
        assert receipt.terminal["return_code"] == 7
        assert engine.accounting_view().receipt_count == 1
        assert connector.execute(issued).to_dict() == receipt.to_dict()
        assert calls == {"lifecycle": 1}
        assert engine.accounting_view().receipt_count == 1
    finally:
        connector.close()
        controller.close()


def test_accepted_terminal_with_stopped_producer_error_is_coherent_failed_receipt(
        tmp_path, monkeypatch):
    driver, controller, _lifecycle, engine = _owned_stack(tmp_path, monkeypatch)
    connector = de.UnifiedDriverExecution(driver=driver, controller=controller)
    original = de.UnknownParentEvidenceProducer.stop_and_join

    def stop_then_report(producer, *args, **kwargs):
        original(producer, *args, **kwargs)
        raise RuntimeError("producer reported after stopping")

    monkeypatch.setattr(de.UnknownParentEvidenceProducer, "stop_and_join", stop_then_report)
    try:
        issued = driver.tick(now=1.0)
        receipt = connector.execute(issued)
        assert receipt.terminal["accepted"] is True
        assert receipt.result_reference is not None
        assert receipt.native_measurement_ids == ()
        assert receipt.settlement_request["outcome"] == "failed"
        assert de.DriverExecutionReceipt.from_dict(receipt.to_dict()) == receipt
        assert connector.execute(issued).to_dict() == receipt.to_dict()
        assert engine.accounting_view().receipt_count == 1
    finally:
        connector.close()
        controller.close()


def test_unresolved_producer_thread_fences_successor_after_charging_attempt(
        tmp_path, monkeypatch):
    driver, controller, _lifecycle, engine = _owned_stack(tmp_path, monkeypatch)
    connector = de.UnifiedDriverExecution(driver=driver, controller=controller)
    original_run = de.UnknownParentEvidenceProducer._run
    original_stop = de.UnknownParentEvidenceProducer.stop_and_join
    release = threading.Event()
    calls = {"stop": 0}

    def blocked_after_work(producer):
        original_run(producer)
        release.wait(2)

    def bounded_stop(producer, *_args, **_kwargs):
        calls["stop"] += 1
        return original_stop(producer, timeout=0.02)

    monkeypatch.setattr(de.UnknownParentEvidenceProducer, "_run", blocked_after_work)
    monkeypatch.setattr(de.UnknownParentEvidenceProducer, "stop_and_join", bounded_stop)
    try:
        first = driver.tick(now=1.0)
        receipt = connector.execute(first)
        assert receipt.terminal["accepted"] is True
        assert receipt.settlement_request["outcome"] == "failed"
        assert engine.accounting_view().receipt_count == 1
        assert connector.execute(first).to_dict() == receipt.to_dict()
        successor = unified_driver.DriverOutcome(
            first.status, first.reasons, "f" * 64,
            unified_driver._thaw(first.selection))
        with pytest.raises(de.DriverExecutionUncertain, match="remains alive"):
            connector.execute(successor)
        assert engine.accounting_view().receipt_count == 1
        with pytest.raises(de.DriverExecutionUncertain, match="teardown remains unresolved"):
            connector.close()
        assert connector.execute(first).to_dict() == receipt.to_dict()
        release.set()
        connector.close()
        with pytest.raises(de.DriverExecutionRefused, match="closed"):
            connector.execute(first)
    finally:
        release.set()
        if not connector._closed:
            connector.close()
        controller.close()
    assert calls["stop"] >= 2


def test_partially_started_live_producer_is_retained_until_close_retry(
        tmp_path, monkeypatch):
    driver, controller, _lifecycle, _engine = _owned_stack(tmp_path, monkeypatch)
    connector = de.UnifiedDriverExecution(driver=driver, controller=controller)
    issued = driver.tick(now=1.0)
    original_run = de.UnknownParentEvidenceProducer._run
    original_start = de.UnknownParentEvidenceProducer.start
    original_stop = de.UnknownParentEvidenceProducer.stop_and_join
    release = threading.Event()

    def blocked_after_work(producer):
        original_run(producer)
        release.wait(2)

    def start_then_fail(producer):
        original_start(producer)
        raise RuntimeError("producer failed after thread start")

    monkeypatch.setattr(de.UnknownParentEvidenceProducer, "_run", blocked_after_work)
    monkeypatch.setattr(de.UnknownParentEvidenceProducer, "start", start_then_fail)
    monkeypatch.setattr(
        de.UnknownParentEvidenceProducer, "stop_and_join",
        lambda producer, *_args, **_kwargs: original_stop(producer, timeout=0.02))
    try:
        with pytest.raises(RuntimeError, match="after thread start"):
            connector.execute(issued)
        assert connector._successor_fence is not None
        assert len(connector._unresolved_producers) == 1
        with pytest.raises(de.DriverExecutionUncertain, match="teardown remains unresolved"):
            connector.close()
        release.set()
        connector.close()
    finally:
        release.set()
        if not connector._closed:
            connector.close()
        controller.close()


def test_pause_refusal_has_no_acquisition_and_same_issue_can_resume(
        tmp_path, monkeypatch):
    driver, controller, _lifecycle, engine = _owned_stack(tmp_path, monkeypatch)
    connector = de.UnifiedDriverExecution(driver=driver, controller=controller)
    try:
        issued = driver.tick(now=1.0)
        assert _command(controller, "pause", "pause-before-worker")["completed"]
        assert controller.worker_attempt_status(
            request_id="unseen-request", plan_digest="1" * 64,
            lineage_id="unseen-lineage", stage_id="unseen-stage") == "unknown"
        with pytest.raises(campaign_control.ControlRefused, match="admission is closed"):
            connector.execute(issued)
        assert connector._launched == set()
        assert controller._acquisition_projection.pending is None
        assert controller._worker_projection.active == {}
        assert _command(controller, "resume", "resume-before-worker")["completed"]
        receipt = connector.execute(issued)
        assert receipt.settlement_request["outcome"] == "invalid"
        assert engine.accounting_view().receipt_count == 1
    finally:
        connector.close()
        controller.close()


def test_invocation_fds_close_when_stage_request_or_producer_start_fails(
        tmp_path, monkeypatch):
    driver, controller, _lifecycle, _engine = _owned_stack(tmp_path, monkeypatch)
    connector = de.UnifiedDriverExecution(driver=driver, controller=controller)
    issued = driver.tick(now=1.0)
    # Materialization pins the controller-owned artifact root once; this descriptor
    # is intentionally lifetime-scoped and is not an invocation channel.
    driver.materialize_runtime(issued)
    baseline = _open_fds()
    original_request = unified_worker.PlannedWorkerInvocation.stage_request
    original_start = de.UnknownParentEvidenceProducer.start
    try:
        with monkeypatch.context() as scoped:
            scoped.setattr(
                unified_worker.PlannedWorkerInvocation, "stage_request",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    ValueError("injected request failure")))
            with pytest.raises(ValueError, match="request failure"):
                connector.execute(issued)
        assert _open_fds() == baseline

        with monkeypatch.context() as scoped:
            def start_then_fail(producer):
                original_start(producer)
                raise RuntimeError("injected producer failure")

            scoped.setattr(
                de.UnknownParentEvidenceProducer, "start",
                start_then_fail)
            with pytest.raises(RuntimeError, match="producer failure"):
                connector.execute(issued)
        assert _open_fds() == baseline
        assert connector._launched == set()
        assert original_request is unified_worker.PlannedWorkerInvocation.stage_request
        assert original_start is de.UnknownParentEvidenceProducer.start
        assert connector.execute(issued).settlement_request["outcome"] == "invalid"
    finally:
        connector.close()
        controller.close()


def test_concurrent_duplicate_submission_runs_one_controller_owned_worker(
        tmp_path, monkeypatch):
    driver, controller, lifecycle, engine = _owned_stack(tmp_path, monkeypatch)
    connector = de.UnifiedDriverExecution(driver=driver, controller=controller)
    issued = driver.tick(now=1.0)
    entered, release = threading.Event(), threading.Event()
    calls = []
    original = lifecycle._wait_outcome

    def blocking_wait(*args, **kwargs):
        calls.append("worker")
        entered.set()
        assert release.wait(2)
        return original(*args, **kwargs)

    monkeypatch.setattr(lifecycle, "_wait_outcome", blocking_wait)
    results, errors = [], []

    def execute():
        try:
            results.append(connector.execute(issued))
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=execute)
    second = threading.Thread(target=execute)
    try:
        first.start()
        assert entered.wait(2)
        assert controller._worker_run_active is True
        assert controller.publish_snapshot()["active_worker"] is not None
        second.start()
        time.sleep(0.02)
        assert second.is_alive()
        release.set()
        first.join(3)
        second.join(3)
        assert not errors and len(results) == 2
        assert results[0].to_dict() == results[1].to_dict()
        assert calls == ["worker"]
        assert engine.accounting_view().receipt_count == 1
        assert controller._worker_run_active is False
    finally:
        release.set()
        first.join(3)
        second.join(3)
        connector.close()
        controller.close()


def test_close_waits_for_active_execute_and_then_fences_new_submission(
        tmp_path, monkeypatch):
    driver, controller, lifecycle, _engine = _owned_stack(tmp_path, monkeypatch)
    connector = de.UnifiedDriverExecution(driver=driver, controller=controller)
    issued = driver.tick(now=1.0)
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    original = lifecycle._wait_outcome

    def blocking_wait(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return original(*args, **kwargs)

    monkeypatch.setattr(lifecycle, "_wait_outcome", blocking_wait)
    results, errors = [], []

    def execute():
        try:
            results.append(connector.execute(issued))
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=execute)
    closer = threading.Thread(target=lambda: (connector.close(), closed.set()))
    try:
        worker.start()
        assert entered.wait(2)
        closer.start()
        time.sleep(0.02)
        assert not closed.is_set()
        release.set()
        worker.join(3)
        closer.join(3)
        assert not errors and len(results) == 1 and closed.is_set()
        with pytest.raises(de.DriverExecutionRefused, match="closed"):
            connector.execute(issued)
    finally:
        release.set()
        worker.join(3)
        closer.join(3)
        connector.close()
        controller.close()


def test_selected_runtime_reaches_native_journal_invalid_accounting_and_replay(
        tmp_path, monkeypatch):
    driver, controller, lifecycle, engine = _owned_stack(tmp_path, monkeypatch)
    connector = de.UnifiedDriverExecution(driver=driver, controller=controller)
    try:
        issued = driver.tick(now=1.0)
        receipt = connector.execute(issued)
        assert receipt.settlement_request["outcome"] == "invalid"
        assert len(receipt.native_measurement_ids) == 2
        assert all(controller.native_capture(item) is not None
                   for item in receipt.native_measurement_ids)
        assert engine.accounting_view().receipt_count == 1
        entries = controller._journal.read_all()
        assert any(item.kind == journal_module.KIND_WORKER_ACQUISITION for item in entries)
        assert any(item.kind == journal_module.KIND_WORKER_LIFECYCLE for item in entries)
        assert controller.publish_snapshot()["active_worker"] is None
        assert controller._worker_run_active is False
        assert controller.worker_attempt_status(
            request_id=receipt.request_id, plan_digest=receipt.terminal["plan_digest"],
            lineage_id=receipt.lineage_id, stage_id=receipt.stage_id) == "terminal"
        assert connector.execute(issued).to_dict() == receipt.to_dict()
        assert engine.accounting_view().receipt_count == 1
    finally:
        connector.close()
        controller.close()

    replay_engine = scheduling.SchedulerEngine(
        engine.config, scheduling.initial_state(engine.config, driver.resolved.campaign_id))
    with campaign_control.CampaignController(
            driver.resolved, tmp_path / "controller", snapshot_version=3,
            scheduler_engine=replay_engine) as replayed:
        assert replay_engine.accounting_view().receipt_count == 1
        assert replayed.worker_attempt_status(
            request_id=receipt.request_id, plan_digest=receipt.terminal["plan_digest"],
            lineage_id=receipt.lineage_id, stage_id=receipt.stage_id) == "unknown"
        duplicate = de.UnifiedDriverExecution.retry_durable(replayed, receipt)
        assert duplicate["status"] == "duplicate"
        assert replay_engine.accounting_view().receipt_count == 1


def test_default_missing_provider_is_safe_to_retry_without_acquisition(
        tmp_path, monkeypatch):
    driver, _old, enrolled, _target, _digest = runtime_driver(git_source=True)
    base, _ = scheduler()
    config = scheduling.SchedulerConfig.from_dict(
        base.to_dict() | {"config_id": enrolled.campaign_id})
    engine = scheduling.SchedulerEngine(
        config, scheduling.initial_state(config, enrolled.campaign_id))
    with campaign_control.CampaignController(
            enrolled, tmp_path / "controller", snapshot_version=3,
            scheduler_engine=engine, readiness_check=lambda: (True, None)) as controller:
        _resume(controller, enrolled.campaign_id)
        driver.controller, driver.scheduler = controller, engine
        connector = de.UnifiedDriverExecution(driver=driver, controller=controller)
        issued = driver.tick(now=1.0)
        real_stop = de.UnknownParentEvidenceProducer.stop_and_join

        def noisy_stop(*args, **kwargs):
            real_stop(*args, **kwargs)
            raise RuntimeError("secondary producer stop failure")

        monkeypatch.setattr(de.UnknownParentEvidenceProducer, "stop_and_join", noisy_stop)
        for _ in range(2):
            with pytest.raises(wl.WaitingAuthority, match="provider is unavailable"):
                connector.execute(issued)
        assert engine.accounting_view().receipt_count == 0
        assert controller._native_records == {}
        assert connector._launched == set()
        assert controller._acquisition_projection.pending is None
        assert controller._worker_projection.active == {}
        connector.close()


def test_ambiguous_authorization_never_relaunches_without_reconciliation(
        tmp_path, monkeypatch):
    driver, controller, _lifecycle, engine = _owned_stack(tmp_path, monkeypatch)
    connector = de.UnifiedDriverExecution(driver=driver, controller=controller)
    issued = driver.tick(now=1.0)
    provider = controller._lifecycle_provider
    original_authorize = provider.authorize
    provider.authorize = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        TimeoutError("test authorization reply lost"))
    try:
        with pytest.raises(de.DriverExecutionUncertain, match="exact terminal recovery"):
            connector.execute(issued)
        with pytest.raises(de.DriverExecutionUncertain, match="not rerun"):
            connector.execute(issued)
        assert controller._acquisition_projection.pending is not None
        assert engine.accounting_view().receipt_count == 0
        provider.authorize = original_authorize
        assert controller.reconcile_workers() is None
        assert controller._acquisition_projection.pending is None
        with pytest.raises(de.DriverExecutionUncertain, match="not rerun"):
            connector.execute(issued)
    finally:
        provider.authorize = original_authorize
        connector.close()
        controller.close()


def test_exact_provider_denial_is_durable_and_safe_to_retry(
        tmp_path, monkeypatch):
    driver, controller, lifecycle, engine = _owned_stack(tmp_path, monkeypatch)
    connector = de.UnifiedDriverExecution(driver=driver, controller=controller)
    issued = driver.tick(now=1.0)
    provider = controller._lifecycle_provider
    original_authorize = provider.authorize
    calls = {"authorize": 0}

    def deny_once(request, container_id, deadline):
        calls["authorize"] += 1
        if calls["authorize"] == 1:
            return wl.AuthorizationDenied(
                request.request_id, container_id, "test capacity unavailable")
        return original_authorize(request, container_id, deadline)

    provider.authorize = deny_once
    try:
        with pytest.raises(wl.WaitingAuthority, match="capacity unavailable"):
            connector.execute(issued)
        assert connector._launched == set()
        assert controller._acquisition_projection.pending is None
        selection = scheduling.Selection.from_dict(issued.selection)
        assert controller.worker_attempt_status(
            request_id=selection.proposal.proposal_id,
            plan_digest=driver.materialize_runtime(issued).plan.digest,
            lineage_id=f"driver:{issued.transition_id}",
            stage_id=f"runtime:{issued.transition_id}") == "not_acquired"
        receipt = connector.execute(issued)
        assert receipt.terminal["worker_generation"] == 2
        assert engine.accounting_view().receipt_count == 1
        acquisition_rows = [
            item.payload for item in controller._journal.read_all()
            if item.kind == journal_module.KIND_WORKER_ACQUISITION]
        assert [row["phase"] for row in acquisition_rows[:2]] == ["INTENT", "RESOLVED"]
        assert acquisition_rows[1]["data"]["outcome"] == "denied"
        assert lifecycle._worker_generation == 2
    finally:
        provider.authorize = original_authorize
        connector.close()
        controller.close()


def test_denial_proof_does_not_cross_controller_incarnation(
        tmp_path, monkeypatch):
    driver, controller, _lifecycle, engine = _owned_stack(tmp_path, monkeypatch)
    connector = de.UnifiedDriverExecution(driver=driver, controller=controller)
    issued = driver.tick(now=1.0)
    prepared = driver.materialize_runtime(issued)
    selection = scheduling.Selection.from_dict(issued.selection)
    provider = controller._lifecycle_provider
    provider.authorize = lambda request, container_id, _deadline: wl.AuthorizationDenied(
        request.request_id, container_id, "test denied")
    with pytest.raises(wl.WaitingAuthority, match="test denied"):
        connector.execute(issued)
    assert controller.worker_attempt_status(
        request_id=selection.proposal.proposal_id, plan_digest=prepared.plan.digest,
        lineage_id=f"driver:{issued.transition_id}",
        stage_id=f"runtime:{issued.transition_id}") == "not_acquired"
    connector.close()
    controller.close()

    replay_engine = scheduling.SchedulerEngine(
        engine.config, scheduling.initial_state(engine.config, driver.resolved.campaign_id))
    with campaign_control.CampaignController(
            driver.resolved, tmp_path / "controller", snapshot_version=3,
            scheduler_engine=replay_engine) as replayed:
        assert replayed.worker_attempt_status(
            request_id=selection.proposal.proposal_id, plan_digest=prepared.plan.digest,
            lineage_id=f"driver:{issued.transition_id}",
            stage_id=f"runtime:{issued.transition_id}") == "unknown"


def test_lost_settlement_reply_retries_native_and_accounting_without_execution(
        tmp_path, monkeypatch):
    driver, controller, lifecycle, engine = _owned_stack(tmp_path, monkeypatch)
    connector = de.UnifiedDriverExecution(driver=driver, controller=controller)
    issued = driver.tick(now=1.0)
    original = controller.unified_driver_settle
    calls = {"settle": 0, "lifecycle": 0}
    real_run = lifecycle.run_stage

    def count_run(*args, **kwargs):
        calls["lifecycle"] += 1
        return real_run(*args, **kwargs)

    def lose_reply(value):
        result = original(value)
        calls["settle"] += 1
        if calls["settle"] == 1:
            raise OSError("reply lost after durable settlement")
        return result

    monkeypatch.setattr(lifecycle, "run_stage", count_run)
    monkeypatch.setattr(controller, "unified_driver_settle", lose_reply)
    try:
        with pytest.raises(de.DriverExecutionUncertain, match="settlement"):
            connector.execute(issued)
        assert engine.accounting_view().receipt_count == 1
        receipt = connector.execute(issued)
        assert receipt.settlement_receipt["status"] == "duplicate"
        assert calls == {"settle": 2, "lifecycle": 1}
        assert engine.accounting_view().receipt_count == 1
    finally:
        connector.close()
        controller.close()


def test_settlement_verifier_refuses_caller_mutation_and_receipt_is_detached(
        tmp_path, monkeypatch):
    driver, controller, _lifecycle, engine = _owned_stack(tmp_path, monkeypatch)
    connector = de.UnifiedDriverExecution(driver=driver, controller=controller)
    try:
        receipt = connector.execute(driver.tick(now=1.0))
        restored = de.DriverExecutionReceipt.from_dict(receipt.to_dict())
        assert restored.to_dict() == receipt.to_dict()
        malformed = receipt.to_dict()
        malformed["settlement_request"]["outcome"] = "valid_comparison"
        malformed["receipt_digest"] = unified_driver._digest({
            key: value for key, value in malformed.items() if key != "receipt_digest"})
        with pytest.raises(de.DriverExecutionRefused, match="settlement request differs"):
            de.DriverExecutionReceipt.from_dict(malformed)
        changed = receipt.to_dict()["settlement_request"]
        changed["outcome"] = "valid_comparison"
        with pytest.raises(de.DriverExecutionRefused, match="differs from the owned"):
            connector._verify_settlement(changed)
        with pytest.raises(campaign_control.ControlRefused, match="conflicts"):
            controller.unified_driver_settle(changed)
        assert receipt.settlement_request["outcome"] == "invalid"
        assert engine.accounting_view().receipt_count == 1
        assert not any(thread.name == "autokernel-parent-evidence" and thread.is_alive()
                       for thread in __import__("threading").enumerate())
    finally:
        connector.close()
        controller.close()
