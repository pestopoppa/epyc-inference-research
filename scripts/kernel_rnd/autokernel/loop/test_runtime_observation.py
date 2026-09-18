"""Runtime facts are diagnostics, never execution or scientific authority."""
from __future__ import annotations

import copy
import threading

import pytest

from . import campaign_control as cc, standalone_runtime as sr
from .test_driver_execution import _owned_stack
from .test_standalone_runtime import _runtime, _reopen_runtime
from .test_discovery_runtime_guard import _discovery, _reason
from .test_profile_preparation_runtime import _fixture, _start


def observation(controller):
    return controller.snapshot()["unified"]["runtime"]


def discovery_runtime(tmp_path, monkeypatch):
    seed, controller, _lifecycle, _engine = _owned_stack(tmp_path, monkeypatch)
    key, original = next(iter(seed.experiment_plans.items()))
    declared = _discovery(original)
    inputs = sr.StandaloneRuntimeInputs(
        seed.resolved, seed.scheduler, seed.profiles, seed.evidence, seed.runtime_anchors,
        seed.runtime_dimensions, {key: declared}, seed.profile_requests,
        seed.actor_identities, seed.execution_inputs, seed.sink_ref)
    runtime = sr.StandaloneRuntime.compose(controller=controller, inputs=inputs)
    return runtime, controller, declared


@pytest.mark.parametrize("failure_report_also_fails", [False, True])
def test_first_recovery_publication_failure_does_not_become_execution_uncertainty(
        tmp_path, monkeypatch, failure_report_also_fails):
    runtime, controller, _lifecycle, _engine = _runtime(tmp_path, monkeypatch)
    try:
        def fail(*_args, **_kwargs):
            raise OSError("fixture diagnostic write fault")
        monkeypatch.setattr(controller, "record_runtime_observation", fail)
        if failure_report_also_fails:
            monkeypatch.setattr(controller, "runtime_observation_failed", fail)
        result = runtime.recover()
        assert result.status == "recovered"
        assert runtime._uncertain is None and runtime._recovered
        row = observation(controller)
        assert row["status"] == "not_reported" and row["observed_at"] is None
        assert row["publication_error"] == (None if failure_report_also_fails else
                                            "OSError: fixture diagnostic write fault")
        assert row["observation_sequence"] == (0 if failure_report_also_fails else 1)
    finally:
        runtime.close()
        controller.close()


def test_settlement_survives_diagnostic_fault_without_retry_or_reexecution(tmp_path, monkeypatch):
    runtime, controller, lifecycle, engine = _runtime(tmp_path, monkeypatch)
    try:
        runtime.recover()
        old = observation(controller)
        calls = []
        original = lifecycle.run_stage
        monkeypatch.setattr(lifecycle, "run_stage", lambda *a, **k: (
            calls.append(1) or original(*a, **k)))
        def fail(*_args, **_kwargs):
            raise OSError("after durable settlement")
        monkeypatch.setattr(controller, "record_runtime_observation", fail)
        result = runtime.tick()
        assert result.status == "settled" and result.execution_receipt is not None
        assert len(engine.export_state().receipts) == 1 and calls == [1]
        assert runtime._uncertain is None and runtime._pending_outcome is None
        row = observation(controller)
        assert row["observed_at"] == old["observed_at"]
        assert row["publication_error"] == "OSError: after durable settlement"
    finally:
        runtime.close()
        controller.close()


def test_actual_runtime_selection_and_discovery_wait_survive_publisher_heartbeats(tmp_path, monkeypatch):
    runtime, controller, _lifecycle, _engine = _runtime(tmp_path, monkeypatch)
    try:
        runtime.recover()
        result = runtime.tick()
        row = result.to_dict()["controller_snapshot"]["unified"]["runtime"]
        assert row["status"] == "settled" and row["publication_error"] is None
        assert row["work_kind"] == "runtime_comparison"
        assert row["settlement_outcome"] == result.execution_receipt["settlement_request"]["outcome"]
        assert row["transition_id"] == result.driver_outcome["transition_id"]
        assert row["target_revision"] == result.driver_outcome["selection"]["proposal"]["target_revision"]
        assert observation(controller) == row
    finally:
        runtime.close()
        controller.close()
    # Separate campaign: configure before issue, no discovery launch or permit.
    (tmp_path / "discovery").mkdir()
    runtime, controller, declared = discovery_runtime(tmp_path / "discovery", monkeypatch)
    try:
        runtime.recover()
        result = runtime.tick()
        assert result.status == "waiting" and _reason(declared) in result.reason
        snapshot = controller.publish_snapshot()
        assert snapshot["observed_state"] == "running"  # admission, not operational readiness
        assert snapshot["unified"]["runtime"]["reason"] == result.reason
        assert snapshot["unified"]["runtime"]["work_kind"] is None
        assert controller.publish_snapshot()["unified"]["runtime"] == snapshot["unified"]["runtime"]
        assert controller.unified_driver_pending_intent() is None
    finally:
        runtime.close()
        controller.close()


def test_recovered_settlement_and_owner_reset_keep_original_identity(tmp_path, monkeypatch):
    runtime, controller, _lifecycle, engine = _runtime(tmp_path, monkeypatch)
    old_result = runtime.recover()
    issued = runtime.driver.tick(now=1)
    new_runtime, new_controller, _new_lifecycle, new_engine = _reopen_runtime(
        runtime, controller, engine, monkeypatch)
    try:
        initial = observation(new_controller)
        assert initial["status"] == "not_reported" and initial["observed_at"] is None
        with pytest.raises(cc.ControlRefused, match="owner/incarnation"):
            new_controller.record_runtime_observation(runtime, old_result)
        result = new_runtime.recover()
        row = observation(new_controller)
        assert result.status == row["status"] == "settled"
        assert row["transition_id"] == issued.transition_id
        assert row["publication_error"] is None and len(new_engine.export_state().receipts) == 1
    finally:
        new_runtime.close()
        new_controller.close()


def test_installed_profile_settlement_and_expiry_debt_are_not_scientific_success(tmp_path):
    materialized, registry, target, _binding, counter, _profile = _fixture(tmp_path, selected_provider=True)
    controller, runtime = _start(materialized, registry)
    try:
        result = runtime.tick()
        row = observation(controller)
        assert result.status == row["status"] == "settled"
        assert row["work_kind"] == "profile_preparation" and row["target_revision"] == target
        assert row["settlement_outcome"] == "prerequisite" and row["publication_error"] is None
        runtime._clock = lambda: controller.current_verified_profile_result(target)["profile_event"]["valid_until"]
        waited = runtime.tick()
        assert "profile_refresh_unavailable" in waited.reason
        assert observation(controller)["reason"] == waited.reason and counter.read_text() == "x"
        assert controller.snapshot()["last_scientific_result_at"] is None
    finally:
        runtime.close()
        controller.close()


def test_active_deadline_is_checked_by_original_owner_not_activity_or_foreign_clock(tmp_path, monkeypatch):
    runtime, controller, lifecycle, _engine = _runtime(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    results, errors = [], []
    def barrier(phase):
        if phase == "WORKER_STAGE":
            entered.set()
            assert release.wait(5)
    lifecycle.fault_hook = barrier
    def run():
        try:
            results.append(runtime.tick())
        except Exception as exc:
            errors.append(exc)
    runtime.recover()
    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert entered.wait(5)
        original = controller.snapshot()
        timing = original["unified"]["worker_timing"]
        assert timing["state"] == "within_deadline" and timing["remaining_seconds"] > 0
        assert timing["worker_id"] == original["active_worker"]["worker_id"]
        assert timing["checked_at"] == original["generated_at"]
        # Only clock observation boundaries are replaced; original intent/worker is untouched.
        with monkeypatch.context() as clocks:
            clocks.setattr(cc.time, "monotonic", lambda: original["active_worker"]["provider_deadline"] + 1)
            expired = controller.snapshot()
        assert expired["unified"]["worker_timing"]["state"] == "deadline_elapsed"
        assert expired["worker_activity_at"] == original["worker_activity_at"]
        with monkeypatch.context() as clocks:
            clocks.setattr(controller, "_runtime_clock_domain", "foreign-domain")
            unknown = controller.snapshot()["unified"]["worker_timing"]
        assert unknown["state"] == "clock_unavailable" and unknown["remaining_seconds"] is None
    finally:
        release.set()
        thread.join(5)
        runtime.close()
        controller.close()
    assert not thread.is_alive() and not errors and results[0].status == "settled"


def test_closed_runtime_observation_bounds_and_selected_refs(tmp_path, monkeypatch):
    runtime, controller, _lifecycle, _engine = _runtime(tmp_path, monkeypatch)
    try:
        runtime.recover()
        row = observation(controller)
        for change in ({"invented": True}, {"work_kind": "actor_preparation"},
                       {"observed_at": "2026-01-01"}, {"status": []},
                       {"publication_error": "x" * 4097}, {"observation_sequence": True}):
            with pytest.raises(cc.ControlRefused):
                cc.validate_runtime_observation(dict(copy.deepcopy(row), **change))
        long_reason = "🚀" * 5000
        runtime._observed_result("waiting", long_reason, 1, controller.snapshot())
        row = observation(controller)
        assert len(row["reason"]) == 4096 and row["reason_truncated"]
        assert row["publication_error"] is None
    finally:
        runtime.close()
        controller.close()
