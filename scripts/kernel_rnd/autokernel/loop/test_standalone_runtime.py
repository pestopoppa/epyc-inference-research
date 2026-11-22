"""Hermetic tests for the real standalone driver/controller composition."""

from __future__ import annotations

import threading
import time
import hashlib
import fcntl
import os
import copy

import pytest

from .. import journal as journal_module
from . import (campaign_control, driver_execution, scheduling, standalone_runtime as sr,
               unified_driver, unified_worker, worker_lifecycle)
from .test_driver_execution import (_command, _owned_stack, _measure,
                                    DirectUnitAuthority)


def _runtime(tmp_path, monkeypatch, *, return_code=0, config=None):
    seed, controller, lifecycle, engine = _owned_stack(
        tmp_path, monkeypatch, return_code=return_code
    )
    inputs = sr.StandaloneRuntimeInputs(
        seed.resolved,
        engine,
        seed.profiles,
        seed.evidence,
        seed.runtime_anchors,
        seed.runtime_dimensions,
        seed.experiment_plans,
        seed.profile_requests,
        seed.actor_identities,
        seed.execution_inputs,
        seed.sink_ref,
    )
    runtime = sr.StandaloneRuntime.compose(
        controller=controller,
        inputs=inputs,
        config=config
        or sr.StandaloneRuntimeConfig(
            idle_interval_seconds=0.001,
            unavailable_backoff_seconds=0.002,
            max_backoff_seconds=0.004,
            shutdown_timeout_seconds=0.05,
        ),
    )
    return runtime, controller, lifecycle, engine


def _crash_release(controller):
    """Fixture-only process-death lock release; retained owners remain journaled."""
    artifact, controller._driver_artifact_store = controller._driver_artifact_store, None
    fd, controller._lease_fd = controller._lease_fd, None
    root, controller._runtime_root = controller._runtime_root, None
    controller._entered = False
    controller._journal = None
    if artifact is not None:
        artifact.close()
    assert fd is not None and root is not None
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)
    root.close()


def _reopen_runtime(runtime, controller, engine, monkeypatch, *, child_counter=None,
                    crashed=False):
    old_driver = runtime.driver
    store = controller.store
    provider = controller._lifecycle_provider
    if crashed:
        provider.inspect = lambda grant, container_id, deadline: (
            worker_lifecycle.RecoveryInspection(
                "exact", provider.authorization,
                "fixture returns the exact retained authorization")
            if provider.authorization.container.path.exists()
            else worker_lifecycle.RecoveryInspection(
                "absent_released", None, "fixture proves absent and released"))
    if crashed:
        _crash_release(controller)
    else:
        runtime.close()
        controller.close()
    replay_engine = scheduling.SchedulerEngine(
        engine.config, scheduling.initial_state(engine.config, old_driver.resolved.campaign_id))
    reopened = campaign_control.CampaignController(
        old_driver.resolved, store, snapshot_version=3, scheduler_engine=replay_engine,
        readiness_check=lambda: (True, None), lifecycle_provider=provider)
    reopened.__enter__()
    inputs = sr.StandaloneRuntimeInputs(
        old_driver.resolved, replay_engine, old_driver.profiles, old_driver.evidence,
        old_driver.runtime_anchors, old_driver.runtime_dimensions,
        old_driver.experiment_plans, old_driver.profile_requests,
        old_driver.actor_identities, old_driver.execution_inputs, old_driver.sink_ref)
    result = sr.StandaloneRuntime.compose(
        controller=reopened, inputs=inputs,
        config=sr.StandaloneRuntimeConfig(0.001, 0.002, 0.004, 0.05))
    lifecycle = reopened._worker_lifecycle
    assert lifecycle is not None

    def fake_wait(process, fd, nonce, contract_digest, request, authorization, held_grant,
                  deadline, *, planned_invocation, container_identity, worker_id,
                  worker_generation, container_id):
        del fd, authorization
        if child_counter is not None:
            child_counter["count"] += 1
        start = unified_worker.WorkerStart(
            nonce, request.request_id, planned_invocation.prepared.prepared_digest,
            request.plan_digest, request.lineage_id, request.stage_id,
            lifecycle.binding.campaign_id, lifecycle.binding.config_digest,
            lifecycle.binding.config_generation, lifecycle.binding.supervisor_id,
            lifecycle.binding.supervisor_incarnation, worker_id, worker_generation,
            held_grant.grant_id, held_grant.generation, container_id,
            worker_lifecycle.process_identity(os.getpid()), container_identity,
            held_grant.clock_domain, deadline)
        planned_invocation.start = start
        authority = DirectUnitAuthority(planned_invocation.parent_authority, start)
        times = iter(f"2026-09-09T00:01:{index:02d}Z" for index in range(20))
        reference = unified_worker.run_prepared_stage(
            planned_invocation.prepared, start, _test_authority=authority,
            _test_membership_probe=lambda _: None, _test_measure=_measure,
            clock=lambda: 1.0, wall_clock=times.__next__)
        planned_invocation._reference = reference
        planned_invocation._reference_digest = unified_driver._digest(reference.to_dict())
        return {"schema": worker_lifecycle.OUTCOME_SCHEMA, "nonce": nonce,
                "contract_digest": contract_digest, "child_pid": process.pid,
                "return_code": 0, "forwarded_signal": None, "stdout_bytes": 0,
                "stderr_bytes": 0, "stdout_sha256": hashlib.sha256(b"").hexdigest(),
                "stderr_sha256": hashlib.sha256(b"").hexdigest(),
                "stdout_truncated": False, "stderr_truncated": False}

    monkeypatch.setattr(lifecycle, "_wait_outcome", fake_wait)
    return result, reopened, lifecycle, replay_engine


def test_restart_after_issue_restores_exact_intent_without_reselection(tmp_path, monkeypatch):
    runtime, controller, _lifecycle, engine = _runtime(tmp_path, monkeypatch)
    runtime.recover()
    issued = runtime.driver.tick(now=1.0)
    transition_id = issued.transition_id
    children = {"count": 0}
    reopened_runtime, reopened, _new_lifecycle, replay_engine = _reopen_runtime(
        runtime, controller, engine, monkeypatch, child_counter=children)
    try:
        recovered = reopened_runtime.recover()
        assert recovered.status == "settled", recovered.reason
        assert recovered.driver_outcome["transition_id"] == transition_id
        assert recovered.driver_outcome["reasons"] == ("restored",)
        assert children["count"] == 1
        assert len(replay_engine.export_state().receipts) == 1
    finally:
        reopened_runtime.close()
        reopened.close()


def test_restart_recovery_retries_same_restored_intent_after_transient_wait(
    tmp_path, monkeypatch
):
    runtime, controller, _lifecycle, engine = _runtime(tmp_path, monkeypatch)
    runtime.recover()
    transition_id = runtime.driver.tick(now=1.0).transition_id
    reopened_runtime, reopened, _new_lifecycle, replay_engine = _reopen_runtime(
        runtime, controller, engine, monkeypatch)
    real = reopened_runtime.executor.recover_issued
    waits = {"once": True}

    def wait_once(outcome):
        if waits.pop("once", False):
            raise worker_lifecycle.WaitingAuthority("fixture provider temporarily unavailable")
        return real(outcome)

    monkeypatch.setattr(reopened_runtime.executor, "recover_issued", wait_once)
    try:
        first = reopened_runtime.recover()
        assert first.status == "recovery_required"
        second = reopened_runtime.recover()
        assert second.status == "settled", second.reason
        assert second.driver_outcome["transition_id"] == transition_id
        assert len(replay_engine.export_state().receipts) == 1
    finally:
        reopened_runtime.close()
        reopened.close()


def test_replayed_intent_is_detached_and_digest_tampering_refuses(tmp_path, monkeypatch):
    runtime, controller, _lifecycle, engine = _runtime(tmp_path, monkeypatch)
    runtime.recover()
    runtime.driver.tick(now=1.0)
    reopened_runtime, reopened, _new_lifecycle, _replay_engine = _reopen_runtime(
        runtime, controller, engine, monkeypatch)
    try:
        pending = reopened.unified_driver_pending_intent()
        forged = copy.deepcopy(pending)
        forged["transition_id"] = "0" * 64
        with pytest.raises(unified_driver.DriverRefused, match="identity/projection"):
            reopened_runtime.driver.restore_issued_intent(forged)
        pending["selection"]["status"] = "blocked"
        assert reopened.unified_driver_pending_intent()["selection"]["status"] == "selected"
    finally:
        reopened_runtime.close()
        reopened.close()


def test_restart_after_terminal_without_durable_held_receipt_stays_fenced(
    tmp_path, monkeypatch
):
    runtime, controller, lifecycle, engine = _runtime(tmp_path, monkeypatch)
    runtime.recover()
    children = {"count": 0}
    real_run = lifecycle.run_stage
    monkeypatch.setattr(lifecycle, "run_stage", lambda *a, **k: (
        children.__setitem__("count", children["count"] + 1) or real_run(*a, **k)))
    monkeypatch.setattr(
        controller, "unified_driver_settle",
        lambda _value: (_ for _ in ()).throw(OSError("fixture crash before settlement")))
    with pytest.raises(sr.StandaloneRuntimeUncertain):
        runtime.tick()
    reopened_runtime, reopened, _new_lifecycle, replay_engine = _reopen_runtime(
        runtime, controller, engine, monkeypatch, child_counter=children)
    try:
        recovered = reopened_runtime.recover()
        assert recovered.status == "recovery_required"
        assert "ownership is unknown" in recovered.reason
        assert children["count"] == 1
        assert not replay_engine.export_state().receipts
    finally:
        reopened_runtime.close()
        reopened.close()


def test_restart_after_settlement_reply_loss_does_not_reopen_work(tmp_path, monkeypatch):
    runtime, controller, lifecycle, engine = _runtime(tmp_path, monkeypatch)
    runtime.recover()
    children = {"count": 0}
    real_run = lifecycle.run_stage
    real_settle = controller.unified_driver_settle
    monkeypatch.setattr(lifecycle, "run_stage", lambda *a, **k: (
        children.__setitem__("count", children["count"] + 1) or real_run(*a, **k)))

    def lose_reply(value):
        real_settle(value)
        raise OSError("fixture lost durable settlement reply")

    monkeypatch.setattr(controller, "unified_driver_settle", lose_reply)
    with pytest.raises(sr.StandaloneRuntimeUncertain):
        runtime.tick()
    reopened_runtime, reopened, _new_lifecycle, replay_engine = _reopen_runtime(
        runtime, controller, engine, monkeypatch, child_counter=children)
    try:
        assert reopened_runtime.recover().status == "recovered"
        assert children["count"] == 1
        assert len(replay_engine.export_state().receipts) == 1
    finally:
        reopened_runtime.close()
        reopened.close()


@pytest.mark.parametrize("phase", ["OWNED_LAUNCH_INTENT", "OWNED_CHILD_CAPTURED"])
def test_restart_reconciles_acquired_or_started_attempt_without_second_child(
    tmp_path, monkeypatch, phase
):
    runtime, controller, lifecycle, engine = _runtime(tmp_path, monkeypatch)
    runtime.recover()
    fired = {"value": False}

    def crash(current):
        if current == phase and not fired["value"]:
            fired["value"] = True
            raise worker_lifecycle.SimulatedCrash(current)

    lifecycle.fault_hook = crash
    with pytest.raises(sr.StandaloneRuntimeUncertain):
        runtime.tick()
    children = {"count": 0}
    reopened_runtime, reopened, _new_lifecycle, replay_engine = _reopen_runtime(
        runtime, controller, engine, monkeypatch, child_counter=children, crashed=True)
    try:
        recovered = reopened_runtime.recover()
        assert recovered.status == "recovery_required"
        assert "trusted held receipt" in recovered.reason
        assert children["count"] == 0
        assert not replay_engine.export_state().receipts
        provider = reopened._lifecycle_provider
        assert not provider.authorization.container.path.exists()
        assert not reopened._worker_projection.active
    finally:
        reopened_runtime.close()
        reopened.close()


def test_requires_recovery_then_crosses_driver_journal_lifecycle_and_settlement(
    tmp_path, monkeypatch
):
    runtime, controller, _lifecycle, engine = _runtime(tmp_path, monkeypatch)
    try:
        before = engine.export_state()
        blocked = runtime.tick()
        assert blocked.status == "recovery_required"
        assert engine.export_state() == before
        assert runtime.recover().status == "recovered"
        result = runtime.tick()
        assert result.status == "settled"
        assert result.driver_outcome["status"] == "intent_recorded"
        assert result.execution_receipt["settlement_receipt"]["status"] == "accepted"
        assert result.execution_receipt["native_measurement_ids"]
        snapshot = campaign_control.validate_snapshot_v3(result.to_dict()["controller_snapshot"])
        assert snapshot["schema"] == campaign_control.SNAPSHOT_SCHEMA_V3
        assert snapshot["unified"]["candidate"]["status"] == "not_connected"
        assert snapshot["unified"]["evidence"]["status"] == "not_connected"
        assert len(engine.export_state().receipts) == 1
        rows = controller._journal.read_all()
        kinds = {item.kind for item in rows}
        assert journal_module.KIND_UNIFIED_DRIVER_ISSUED in kinds
        assert journal_module.KIND_UNIFIED_DRIVER_SETTLED in kinds
    finally:
        runtime.close()
        controller.close()


def test_default_provider_waits_and_backoff_is_bounded(tmp_path):
    from .test_unified_driver import runtime_driver

    seed, _engine, enrolled, _target, _digest = runtime_driver(git_source=True)
    config = scheduling.SchedulerConfig.from_dict(
        seed.scheduler.config.to_dict() | {"config_id": enrolled.campaign_id}
    )
    engine = scheduling.SchedulerEngine(
        config, scheduling.initial_state(config, enrolled.campaign_id)
    )
    controller = campaign_control.CampaignController(
        enrolled, tmp_path / "controller", snapshot_version=3, scheduler_engine=engine
    )
    controller.__enter__()
    _command(controller, "resume", "resume")
    inputs = sr.StandaloneRuntimeInputs(
        seed.resolved,
        engine,
        seed.profiles,
        seed.evidence,
        seed.runtime_anchors,
        seed.runtime_dimensions,
        seed.experiment_plans,
        seed.profile_requests,
        seed.actor_identities,
        seed.execution_inputs,
        seed.sink_ref,
    )
    runtime = sr.StandaloneRuntime.compose(
        controller=controller,
        inputs=inputs,
        config=sr.StandaloneRuntimeConfig(0.001, 0.002, 0.004, 0.05),
    )
    try:
        assert runtime.recover().status == "recovered"
        delays = [runtime.tick().retry_after_seconds for _ in range(8)]
        assert delays[:3] == [0.002, 0.004, 0.004]
        assert max(delays) == 0.004
        assert not engine.export_state().issued_selection_digests
    finally:
        runtime.close()
        controller.close()


def test_pause_and_drain_prevent_new_runtime_issue(tmp_path, monkeypatch):
    runtime, controller, _lifecycle, engine = _runtime(tmp_path, monkeypatch)
    try:
        runtime.recover()
        _command(controller, "pause", "pause")
        assert runtime.tick().status == "waiting"
        assert not engine.export_state().issued_selection_digests
        _command(controller, "resume", "resume-again")
        _command(controller, "drain", "drain")
        result = runtime.tick()
        assert result.status == "waiting"
        assert result.controller_snapshot["desired_state"] == "drained"
        assert not engine.export_state().issued_selection_digests
    finally:
        runtime.close()
        controller.close()


def test_exact_driver_reply_retry_keeps_transition_identity(tmp_path, monkeypatch):
    runtime, controller, _lifecycle, _engine = _runtime(tmp_path, monkeypatch)
    runtime.recover()
    real = controller.unified_driver_transaction
    lost = {"once": True}

    def lose_reply(value):
        result = real(value)
        if lost.pop("once", False):
            raise OSError("fixture lost reply after durable append")
        return result

    monkeypatch.setattr(controller, "unified_driver_transaction", lose_reply)
    try:
        with pytest.raises(sr.StandaloneRuntimeUncertain):
            runtime.tick()
        result = runtime.retry_pending()
        assert result.status == "settled"
        assert result.driver_outcome["reasons"] == ("duplicate",)
        assert result.execution_receipt["settlement_receipt"]["status"] == "accepted"
    finally:
        runtime.close()
        controller.close()


def test_temporarily_refused_exact_driver_retry_keeps_fresh_work_fenced(tmp_path, monkeypatch):
    runtime, controller, _lifecycle, _engine = _runtime(tmp_path, monkeypatch)
    runtime.recover()
    real_transaction = controller.unified_driver_transaction
    lost = {"once": True}

    def lose_reply(value):
        result = real_transaction(value)
        if lost.pop("once", False):
            raise OSError("fixture lost durable transaction reply")
        return result

    monkeypatch.setattr(controller, "unified_driver_transaction", lose_reply)
    with pytest.raises(sr.StandaloneRuntimeUncertain):
        runtime.tick()
    real_retry = runtime.driver.retry_pending
    refused = {"once": True}

    def refuse_retry():
        if refused.pop("once", False):
            raise campaign_control.ControlRefused("fixture controller temporarily unavailable")
        return real_retry()

    monkeypatch.setattr(runtime.driver, "retry_pending", refuse_retry)
    try:
        assert runtime.retry_pending().status == "waiting"
        assert runtime.tick().status == "recovery_required"
        assert runtime.retry_pending().status == "settled"
    finally:
        runtime.close()
        controller.close()


def test_refused_before_acquisition_retains_exact_intent_for_bounded_retry(tmp_path, monkeypatch):
    from . import worker_lifecycle

    runtime, controller, _lifecycle, engine = _runtime(tmp_path, monkeypatch)
    runtime.recover()
    real = runtime.executor.execute
    refused = {"once": True}

    def refuse_once(outcome):
        if refused.pop("once", False):
            raise worker_lifecycle.WaitingAuthority("fixture provider unavailable")
        return real(outcome)

    monkeypatch.setattr(runtime.executor, "execute", refuse_once)
    try:
        first = runtime.tick()
        assert first.status == "waiting"
        assert first.reason == "fixture provider unavailable"
        assert len(engine.export_state().issued_selection_digests) == 1
        assert runtime.tick().status == "recovery_required"
        retried = runtime.recover()
        assert retried.status == "settled"
        assert retried.driver_outcome["transition_id"] == first.driver_outcome["transition_id"]
    finally:
        runtime.close()
        controller.close()


def test_lost_settlement_reply_retries_finished_attempt_without_reexecution(tmp_path, monkeypatch):
    runtime, controller, lifecycle, engine = _runtime(tmp_path, monkeypatch)
    runtime.recover()
    real_settle = controller.unified_driver_settle
    lost = {"once": True}
    executions = {"count": 0}
    real_run = lifecycle.run_stage

    def count_run(*args, **kwargs):
        executions["count"] += 1
        return real_run(*args, **kwargs)

    def lose_reply(value):
        result = real_settle(value)
        if lost.pop("once", False):
            raise OSError("fixture lost settlement reply after durable append")
        return result

    monkeypatch.setattr(lifecycle, "run_stage", count_run)
    monkeypatch.setattr(controller, "unified_driver_settle", lose_reply)
    try:
        with pytest.raises(sr.StandaloneRuntimeUncertain):
            runtime.tick()
        retried = runtime.retry_pending()
        assert retried.status == "settled"
        assert retried.execution_receipt["settlement_receipt"]["status"] == "duplicate"
        assert executions["count"] == 1
        assert len(engine.export_state().receipts) == 1
    finally:
        runtime.close()
        controller.close()


def test_run_autoretries_driver_transaction_reply_loss(tmp_path, monkeypatch):
    runtime, controller, _lifecycle, engine = _runtime(tmp_path, monkeypatch)
    stop = threading.Event()
    real_transaction = controller.unified_driver_transaction
    real_execute = runtime.executor.execute
    lost = {"once": True}

    def lose_reply(value):
        result = real_transaction(value)
        if lost.pop("once", False):
            raise OSError("fixture lost transaction reply")
        return result

    def execute_then_stop(outcome):
        receipt = real_execute(outcome)
        stop.set()
        return receipt

    monkeypatch.setattr(controller, "unified_driver_transaction", lose_reply)
    monkeypatch.setattr(runtime.executor, "execute", execute_then_stop)
    try:
        result = runtime.run(stop)
        assert result.status == "settled"
        assert result.driver_outcome["reasons"][0] == "duplicate"
        assert len(engine.export_state().receipts) == 1
    finally:
        runtime.close()
        controller.close()


def test_run_autoretries_lost_settlement_reply_without_second_child(tmp_path, monkeypatch):
    runtime, controller, lifecycle, engine = _runtime(tmp_path, monkeypatch)
    stop = threading.Event()
    real_settle = controller.unified_driver_settle
    real_execute = runtime.executor.execute
    real_run = lifecycle.run_stage
    lost = {"once": True}
    children = {"count": 0}

    def lose_reply(value):
        result = real_settle(value)
        if lost.pop("once", False):
            raise OSError("fixture lost settlement reply")
        return result

    def count_child(*args, **kwargs):
        children["count"] += 1
        return real_run(*args, **kwargs)

    def execute_then_stop(outcome):
        receipt = real_execute(outcome)
        stop.set()
        return receipt

    monkeypatch.setattr(controller, "unified_driver_settle", lose_reply)
    monkeypatch.setattr(lifecycle, "run_stage", count_child)
    monkeypatch.setattr(runtime.executor, "execute", execute_then_stop)
    try:
        result = runtime.run(stop)
        assert result.status == "settled"
        assert result.execution_receipt["settlement_receipt"]["status"] == "duplicate"
        assert children["count"] == 1
        assert len(engine.export_state().receipts) == 1
    finally:
        runtime.close()
        controller.close()


def test_run_bounds_permanent_uncertainty_and_never_reselects(tmp_path, monkeypatch):
    runtime, controller, _lifecycle, engine = _runtime(
        tmp_path,
        monkeypatch,
        config=sr.StandaloneRuntimeConfig(0.001, 0.001, 0.002, 0.05, max_exact_retries=2),
    )
    stop = threading.Event()
    attempts = []

    def unresolved(outcome):
        attempts.append(outcome.transition_id)
        raise driver_execution.DriverExecutionUncertain("fixture ownership unresolved")

    monkeypatch.setattr(runtime.executor, "execute", unresolved)
    try:
        result = runtime.run(stop)
        assert result.status == "recovery_required"
        assert "retry budget exhausted" in result.reason
        assert len(attempts) == 3
        assert len(set(attempts)) == 1
        assert len(engine.export_state().issued_selection_digests) == 1
        assert not engine.export_state().receipts
    finally:
        runtime.close()
        controller.close()


def test_stop_interrupts_exact_retry_backoff_before_second_attempt(tmp_path, monkeypatch):
    from . import worker_lifecycle

    runtime, controller, _lifecycle, _engine = _runtime(
        tmp_path,
        monkeypatch,
        config=sr.StandaloneRuntimeConfig(0.001, 0.5, 1.0, 0.05),
    )
    stop = threading.Event()
    attempted = threading.Event()
    calls = {"count": 0}
    result = {}

    def unavailable(_outcome):
        calls["count"] += 1
        attempted.set()
        raise worker_lifecycle.WaitingAuthority("fixture remains unavailable")

    monkeypatch.setattr(runtime.executor, "execute", unavailable)
    thread = threading.Thread(target=lambda: result.setdefault("value", runtime.run(stop)))
    started = time.monotonic()
    thread.start()
    assert attempted.wait(1)
    stop.set()
    thread.join(0.2)
    try:
        assert not thread.is_alive()
        assert result["value"].status == "waiting"
        assert calls["count"] == 1
        assert time.monotonic() - started < 0.3
    finally:
        runtime.close()
        controller.close()


def test_shutdown_incomplete_does_not_close_active_executor(tmp_path, monkeypatch):
    runtime, controller, _lifecycle, _engine = _runtime(tmp_path, monkeypatch)
    runtime.recover()
    entered = threading.Event()
    release = threading.Event()
    real_tick = runtime.driver.tick

    def blocked_tick(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return real_tick(*args, **kwargs)

    monkeypatch.setattr(runtime.driver, "tick", blocked_tick)
    thread = threading.Thread(target=runtime.tick)
    thread.start()
    assert entered.wait(1)
    try:
        result = runtime.close(deadline=time.monotonic() + 0.01)
        assert result.status == "shutdown_incomplete"
        assert runtime.executor._closed is False
    finally:
        release.set()
        thread.join(2)
        runtime.close(deadline=time.monotonic() + 1)
        controller.close()


def test_executor_teardown_uncertainty_is_retryable(tmp_path, monkeypatch):
    runtime, controller, _lifecycle, _engine = _runtime(tmp_path, monkeypatch)
    runtime.recover()
    calls = {"count": 0}
    real_close = runtime.executor.close

    def uncertain_once():
        calls["count"] += 1
        if calls["count"] == 1:
            raise driver_execution.DriverExecutionUncertain("fixture producer remains")
        real_close()

    monkeypatch.setattr(runtime.executor, "close", uncertain_once)
    try:
        first = runtime.close()
        assert first.status == "shutdown_incomplete"
        assert runtime.close().status == "closed"
    finally:
        controller.close()


def test_runtime_close_is_idempotent_after_controller_close(tmp_path, monkeypatch):
    runtime, controller, _lifecycle, _engine = _runtime(tmp_path, monkeypatch)
    first = runtime.close()
    controller.close()
    assert runtime.close() is first


def test_stop_event_interrupts_bounded_wait(tmp_path):
    from .test_unified_driver import runtime_driver

    seed, _engine, enrolled, _target, _digest = runtime_driver(git_source=True)
    config = scheduling.SchedulerConfig.from_dict(
        seed.scheduler.config.to_dict() | {"config_id": enrolled.campaign_id}
    )
    engine = scheduling.SchedulerEngine(
        config, scheduling.initial_state(config, enrolled.campaign_id)
    )
    controller = campaign_control.CampaignController(
        enrolled, tmp_path / "controller", snapshot_version=3, scheduler_engine=engine
    )
    controller.__enter__()
    _command(controller, "resume", "resume")
    inputs = sr.StandaloneRuntimeInputs(
        seed.resolved,
        engine,
        seed.profiles,
        seed.evidence,
        seed.runtime_anchors,
        seed.runtime_dimensions,
        seed.experiment_plans,
        seed.profile_requests,
        seed.actor_identities,
        seed.execution_inputs,
        seed.sink_ref,
    )
    runtime = sr.StandaloneRuntime.compose(
        controller=controller, inputs=inputs, config=sr.StandaloneRuntimeConfig(0.01, 0.5, 1.0, 0.1)
    )
    stop = threading.Event()
    thread = threading.Thread(target=runtime.run, args=(stop,))
    started = time.monotonic()
    thread.start()
    time.sleep(0.02)
    stop.set()
    thread.join(0.2)
    try:
        assert not thread.is_alive()
        assert time.monotonic() - started < 0.3
    finally:
        runtime.close()
        controller.close()


def test_paused_journal_restart_recovers_before_remaining_closed(tmp_path):
    from .test_driver_execution import HeldProvider
    from .test_unified_driver import runtime_driver

    seed, _engine, enrolled, _target, _digest = runtime_driver(git_source=True)
    scheduler_config = scheduling.SchedulerConfig.from_dict(
        seed.scheduler.config.to_dict() | {"config_id": enrolled.campaign_id}
    )
    store = tmp_path / "controller"
    containers = tmp_path / "containers"
    containers.mkdir(mode=0o700)

    def open_runtime():
        engine = scheduling.SchedulerEngine(
            scheduler_config, scheduling.initial_state(scheduler_config, enrolled.campaign_id)
        )
        controller = campaign_control.CampaignController(
            enrolled,
            store,
            snapshot_version=3,
            scheduler_engine=engine,
            readiness_check=lambda: (True, None),
            lifecycle_provider=HeldProvider(containers),
        )
        controller.__enter__()
        inputs = sr.StandaloneRuntimeInputs(
            seed.resolved,
            engine,
            seed.profiles,
            seed.evidence,
            seed.runtime_anchors,
            seed.runtime_dimensions,
            seed.experiment_plans,
            seed.profile_requests,
            seed.actor_identities,
            seed.execution_inputs,
            seed.sink_ref,
        )
        return sr.StandaloneRuntime.compose(controller=controller, inputs=inputs), controller

    first, first_controller = open_runtime()
    try:
        assert first.recover().status == "recovered"
        assert first.tick().status == "waiting"
    finally:
        first.close()
        first_controller.close()
    second, second_controller = open_runtime()
    try:
        assert second_controller.snapshot()["desired_state"] == "paused"
        assert second.tick().status == "recovery_required"
        assert second.recover().status == "recovered"
        assert second.tick().status == "waiting"
    finally:
        second.close()
        second_controller.close()


def test_config_rejects_boolean_nonfinite_and_unbounded_order():
    for bad in (True, 0, -1, float("inf"), float("nan")):
        with pytest.raises(sr.StandaloneRuntimeRefused):
            sr.StandaloneRuntimeConfig(idle_interval_seconds=bad)
    with pytest.raises(sr.StandaloneRuntimeRefused):
        sr.StandaloneRuntimeConfig(unavailable_backoff_seconds=2, max_backoff_seconds=1)
    for bad_retries in (True, 0, -1, sr.MAX_EXACT_RETRIES + 1):
        with pytest.raises(sr.StandaloneRuntimeRefused):
            sr.StandaloneRuntimeConfig(max_exact_retries=bad_retries)


def test_shutdown_drain_between_planning_and_issue_is_typed_stop(tmp_path, monkeypatch):
    runtime, controller, _lifecycle, _engine = _runtime(tmp_path, monkeypatch)
    assert runtime.recover().status == "recovered"
    entered = threading.Event()
    release = threading.Event()
    stop = threading.Event()
    observed = {}
    transaction = controller.unified_driver_transaction

    def delayed_transaction(value):
        entered.set()
        assert release.wait(2)
        return transaction(value)

    monkeypatch.setattr(controller, "unified_driver_transaction", delayed_transaction)

    def run():
        try:
            observed["result"] = runtime.run(stop)
        except BaseException as exc:
            observed["error"] = exc

    thread = threading.Thread(target=run, daemon=False)
    thread.start()
    try:
        assert entered.wait(2)
        controller.request_shutdown_drain()
        runtime.request_stop()
        stop.set()
        release.set()
        thread.join(2)
        assert not thread.is_alive()
        assert "error" not in observed
        assert observed["result"].status == "stopped"
    finally:
        release.set()
        runtime.request_stop()
        stop.set()
        thread.join(2)
        runtime.close()
        controller.close()


def test_concurrent_stop_does_not_swallow_unrelated_driver_refusal(tmp_path, monkeypatch):
    runtime, controller, _lifecycle, _engine = _runtime(tmp_path, monkeypatch)
    assert runtime.recover().status == "recovered"
    entered = threading.Event()
    release = threading.Event()
    stop = threading.Event()
    observed = {}

    def unrelated_refusal(_value):
        entered.set()
        assert release.wait(2)
        raise campaign_control.ControlRefused("fixture unrelated refusal")

    monkeypatch.setattr(controller, "unified_driver_transaction", unrelated_refusal)

    def run():
        try:
            observed["result"] = runtime.run(stop)
        except BaseException as exc:
            observed["error"] = exc

    thread = threading.Thread(target=run, daemon=False)
    thread.start()
    try:
        assert entered.wait(2)
        runtime.request_stop()
        stop.set()
        release.set()
        thread.join(2)
        assert not thread.is_alive()
        assert isinstance(observed.get("error"), unified_driver.DriverRefused)
        assert "fixture unrelated refusal" in str(observed["error"])
        assert "result" not in observed
    finally:
        release.set()
        runtime.request_stop()
        stop.set()
        thread.join(2)
        runtime.close()
        controller.close()


# AKU-07j restart consumer: durable held receipts settle the ORIGINAL unsettled driver
# selection exactly once after process restart. All receipts here are labelled fixtures
# from HeldProvider; they prove accounting wiring, not real resource ownership.


def _crash_at(runtime, lifecycle, phase):
    fired = {"value": False}

    def crash(current):
        if current == phase and not fired["value"]:
            fired["value"] = True
            raise worker_lifecycle.SimulatedCrash(current)

    lifecycle.fault_hook = crash
    with pytest.raises(sr.StandaloneRuntimeUncertain):
        runtime.tick()


def _payloads(controller, kind):
    return [entry.payload for entry in controller._journal.read_all() if entry.kind == kind]


def _no_reacquire(controller):
    calls = {"authorize": 0}
    provider = controller._lifecycle_provider
    real = provider.authorize

    def counted(*args, **kwargs):
        calls["authorize"] += 1
        return real(*args, **kwargs)

    provider.authorize = counted
    return calls


def _lookup_args(controller):
    pending = controller.unified_driver_pending_intent()
    selection = pending["selection"]
    work = pending["catalog"]["work_by_stage_digest"][selection["proposal_digest"]]
    return {"catalog_id": pending["catalog_id"], "transition_id": pending["transition_id"],
            "selection": selection, "plan_digest": work["stage_plan_digest"]}


def _assert_single_held_settlement(controller, transition_id):
    worker = _payloads(controller, journal_module.KIND_WORKER_LIFECYCLE)
    assert [row["event"] for row in worker].count("WORKER_RESULT_STALE") == 1
    assert "WORKER_RESULT_ACCEPTED" not in {row["event"] for row in worker}
    held, = [row for row in worker if row["event"] == "OWNED_HELD_RECEIPT_RETAINED"]
    settled, = _payloads(controller, journal_module.KIND_UNIFIED_DRIVER_SETTLED)
    assert settled["transition_id"] == transition_id
    assert settled["receipt"] == held["data"]["receipt"]
    assert settled["outcome"] == "failed"
    assert [ref.split(":")[0] for ref in settled["terminal_refs"]] == ["lifecycle"]
    assert not controller._native_records
    return settled


@pytest.mark.parametrize("settle_crash", [False, True])
def test_restart_after_held_append_settles_original_transition_once_and_idempotently(
    tmp_path, monkeypatch, settle_crash
):
    runtime, controller, lifecycle, engine = _runtime(tmp_path, monkeypatch)
    runtime.recover()
    _crash_at(runtime, lifecycle, "OWNED_HELD_RECEIPT_RETAINED")
    transition_id = controller.unified_driver_pending_intent()["transition_id"]
    children = {"count": 0}
    current = _reopen_runtime(runtime, controller, engine, monkeypatch,
                              child_counter=children, crashed=True)
    calls = _no_reacquire(current[1])
    if settle_crash:
        # Crash after the durable stale event but before settlement: the next process
        # settles from the accounting-only index (no live lifecycle terminal exists).
        monkeypatch.setattr(current[1], "unified_driver_settle", lambda _value: (
            _ for _ in ()).throw(OSError("fixture crash before settlement")))
        assert current[0].recover().status == "recovery_required"
        assert not _payloads(current[1], journal_module.KIND_UNIFIED_DRIVER_SETTLED)
        current = _reopen_runtime(*current[:2], current[3], monkeypatch,
                                  child_counter=children, crashed=True)
        calls = _no_reacquire(current[1])
        assert current[1]._worker_lifecycle._terminals == {}
    recovered = current[0].recover()
    assert recovered.status == "settled", recovered.reason
    assert recovered.driver_outcome["transition_id"] == transition_id
    assert not recovered.execution_receipt["native_measurement_ids"]
    settled = _assert_single_held_settlement(current[1], transition_id)
    assert len(current[3].export_state().receipts) == 1
    assert children["count"] == 0 and calls["authorize"] == 0
    before_state = current[3].export_state()
    before = [(e.kind, e.payload) for e in current[1]._journal.read_all()
              if e.kind != journal_module.KIND_CAMPAIGN_SUPERVISOR_EVENT]
    again = _reopen_runtime(*current[:2], current[3], monkeypatch, child_counter=children)
    try:
        calls = _no_reacquire(again[1])
        assert again[1].unified_driver_pending_intent() is None
        assert again[0].recover().status == "recovered"
        assert again[3].export_state() == before_state
        assert again[1]._journal.read_all()[-1].kind == \
            journal_module.KIND_CAMPAIGN_SUPERVISOR_EVENT
        after = [(e.kind, e.payload) for e in again[1]._journal.read_all()
                 if e.kind != journal_module.KIND_CAMPAIGN_SUPERVISOR_EVENT]
        assert after == before
        assert _assert_single_held_settlement(again[1], transition_id) == settled
        assert children["count"] == 0 and calls["authorize"] == 0
    finally:
        again[0].close()
        again[1].close()


def test_restored_held_lookup_is_bound_to_the_original_unsettled_selection(
    tmp_path, monkeypatch
):
    runtime, controller, lifecycle, engine = _runtime(tmp_path, monkeypatch)
    runtime.recover()
    _crash_at(runtime, lifecycle, "OWNED_HELD_RECEIPT_RETAINED")
    first = _reopen_runtime(runtime, controller, engine, monkeypatch, crashed=True)
    monkeypatch.setattr(first[1], "unified_driver_settle", lambda _value: (
        _ for _ in ()).throw(OSError("fixture crash before settlement")))
    assert first[0].recover().status == "recovery_required"
    reopened_runtime, reopened, _lifecycle, _engine = _reopen_runtime(
        *first[:2], first[3], monkeypatch, crashed=True)
    try:
        args = _lookup_args(reopened)
        terminal, receipt = reopened.restored_held_settlement(**args)
        assert terminal.accepted is False and receipt.proposal_id == terminal.request_id
        # Accounting only: the restored terminal grants no lifecycle result authority.
        assert reopened.worker_terminal_for_request(
            request_id=terminal.request_id, plan_digest=terminal.plan_digest,
            lineage_id=terminal.lineage_id, stage_id=terminal.stage_id) is None
        with pytest.raises(worker_lifecycle.LifecycleRefused):
            reopened.worker_held_claim_receipt(terminal)
        other_request = copy.deepcopy(args["selection"])
        other_request["proposal"]["proposal_id"] = "fixture-other-request"
        for bad in ({"transition_id": "0" * 64}, {"catalog_id": "1" * 64},
                    {"plan_digest": "2" * 64}, {"selection": other_request}):
            with pytest.raises(campaign_control.ControlRefused, match="original unsettled"):
                reopened.restored_held_settlement(**(args | bad))
        assert reopened_runtime.recover().status == "settled"
        with pytest.raises(campaign_control.ControlRefused, match="original unsettled"):
            reopened.restored_held_settlement(**args)
    finally:
        reopened_runtime.close()
        reopened.close()


def test_legacy_restart_without_held_record_keeps_existing_fence(tmp_path, monkeypatch):
    runtime, controller, lifecycle, engine = _runtime(tmp_path, monkeypatch)
    runtime.recover()
    _crash_at(runtime, lifecycle, "OWNED_TERMINAL")
    first = _reopen_runtime(runtime, controller, engine, monkeypatch, crashed=True)
    recovered = first[0].recover()
    assert recovered.status == "recovery_required"
    assert "trusted held receipt" in recovered.reason
    second = _reopen_runtime(*first[:2], first[3], monkeypatch, crashed=True)
    try:
        assert second[1]._restored_held_accounting == {}
        assert second[1].restored_held_settlement(**_lookup_args(second[1])) is None
        again = second[0].recover()
        assert again.status == "recovery_required"
        assert "ownership is unknown" in again.reason
        assert not second[3].export_state().receipts
        assert not _payloads(second[1], journal_module.KIND_UNIFIED_DRIVER_SETTLED)
    finally:
        second[0].close()
        second[1].close()
