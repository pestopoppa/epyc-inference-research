"""Actor/profile refusal regressions with the existing owned hermetic test worker."""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading
from dataclasses import replace

import pytest

from . import actor_lifecycle, actor_preparation, campaign_control, target_profile_execution
from .test_target_profile_execution import test_selected_profile_requires_success_after_binding_held_cost as _run_owned_profiler
from .test_worker_lifecycle import MockProvider
from .test_actor_lifecycle import test_actual_worker_lifecycle_runs_backend_and_retains_native_journal as _run_owned_actor
from .test_actor_lifecycle import _profile_receipt
from .test_actor_preparation import _resolved
from .test_campaign_control import _command


def test_cancel_after_actual_owned_terminal_before_cost_retrieval(tmp_path, monkeypatch):
    controllers = []
    observed = {}

    class BoundaryObserved(BaseException):
        pass

    def at_first_cost_read(controller, terminal):
        controllers.append(controller)
        reservation = next(iter(controller._actor_profile_execution_reservations.values()))
        observed["attempt_status"] = controller.worker_attempt_status(
            request_id=terminal.request_id, plan_digest=terminal.plan_digest,
            lineage_id=terminal.lineage_id, stage_id=terminal.stage_id)
        observed["provider_authorized"] = controller._lifecycle_provider.authorization is not None
        observed["provider_released"] = bool(controller._lifecycle_provider.released)
        observed["trusted_receipt_cached"] = terminal.request_id in controller._actor_trusted_held_receipts
        observed["reservations_before"] = len(controller._actor_profile_execution_reservations)
        with pytest.raises(campaign_control.ControlRefused, match="negative-admission proof"):
            controller.cancel_target_profile_execution(reservation)
        observed["reservations_after"] = len(controller._actor_profile_execution_reservations)
        raise BoundaryObserved

    monkeypatch.setattr(campaign_control.CampaignController, "actor_held_claim_receipt", at_first_cost_read)
    try:
        with pytest.raises(BoundaryObserved):
            _run_owned_profiler(tmp_path, 0)
    finally:
        for controller in controllers:
            controller.close()
    assert observed == {"attempt_status": "terminal", "provider_authorized": True,
                        "provider_released": True, "trusted_receipt_cached": False,
                        "reservations_before": 1, "reservations_after": 1}
    print(json.dumps(observed, sort_keys=True))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0, -1, True])
@pytest.mark.parametrize("field", ["max_stage_seconds", "teardown_seconds"])
def test_lifecycle_config_refuses_nonfinite_or_nonpositive_budget(tmp_path, value, field):
    kwargs = {"campaign_digest": "a" * 64, "cwd": tmp_path, "env": {},
              "max_stage_seconds": 1, "teardown_seconds": 1, "max_retained_output_bytes": 4096}
    kwargs[field] = value
    with pytest.raises(actor_preparation.PreparationRefused, match="finite and positive"):
        actor_lifecycle.ActorLifecycleConfig(**kwargs)


def test_lifecycle_config_detaches_and_freezes_caller_environment(tmp_path):
    environment = {"PATH": "/initial"}
    config = actor_lifecycle.ActorLifecycleConfig("a" * 64, tmp_path, environment, 1, 1, 4096)
    environment["PATH"] = "/changed-after-validation"
    assert config.env["PATH"] == "/initial"
    with pytest.raises(TypeError):
        config.env["PATH"] = "/mutated"


def test_profile_mechanism_detaches_and_freezes_identity_and_environment(tmp_path):
    loaded = {"target_revision_digest": "a" * 64, "model_digest": "1" * 64,
              "quantization": "Q4_K_M", "recipe_digest": "2" * 64,
              "executable_digest": "3" * 64, "dso_digest": "4" * 64}
    environment = {"PATH": "/initial"}
    mechanism = target_profile_execution.ProfileMechanism(
        "explicit-profile", tmp_path / "profiler", "a" * 64, tmp_path,
        environment, loaded, 1, 1, 4096)
    loaded["target_revision_digest"] = "foreign-invalid-after-validation"
    environment["PATH"] = "/changed-after-validation"
    assert mechanism.loaded_identity["target_revision_digest"] == "a" * 64
    assert mechanism.env["PATH"] == "/initial"
    with pytest.raises(TypeError):
        mechanism.loaded_identity["target_revision_digest"] = "mutated"
    with pytest.raises(TypeError):
        mechanism.env["PATH"] = "/mutated"


def test_cancel_prelaunch_fences_later_public_worker_invocation(tmp_path, monkeypatch):
    original = campaign_control.CampaignController.reserve_target_profile_execution
    seen = []
    def cancel_before_return(controller, **kwargs):
        reservation = original(controller, **kwargs)
        seen.append(controller)
        assert controller.worker_attempt_status(
            request_id=reservation.request_id, plan_digest=reservation.stage_plan_digest,
            lineage_id=reservation.transition_id, stage_id=reservation.stage_id) == "unknown"
        controller.cancel_target_profile_execution(reservation)
        assert not controller._actor_profile_execution_reservations
        return reservation
    monkeypatch.setattr(campaign_control.CampaignController, "reserve_target_profile_execution", cancel_before_return)
    try:
        with pytest.raises(campaign_control.ControlRefused, match="admission was cancelled"):
            _run_owned_profiler(tmp_path, 0)
        assert seen[0]._lifecycle_provider.authorization is None
    finally:
        for controller in seen:
            controller.close()


def test_pre_engine_no_acquisition_refusal_remains_safely_cancellable(tmp_path, monkeypatch):
    original_reserve = campaign_control.CampaignController.reserve_target_profile_execution
    original_run = campaign_control.CampaignController.run_worker_stage
    seen, requests = [], []
    def pause_before_return(controller, **kwargs):
        reservation = original_reserve(controller, **kwargs)
        seen.append((controller, reservation))
        controller.apply_command(_command(controller.resolved, "pause-before-engine", "pause",
                                           controller.control_revision))
        return reservation
    def capture_request(controller, request, **kwargs):
        requests.append(request)
        return original_run(controller, request, **kwargs)
    monkeypatch.setattr(campaign_control.CampaignController, "reserve_target_profile_execution", pause_before_return)
    monkeypatch.setattr(campaign_control.CampaignController, "run_worker_stage", capture_request)
    try:
        with pytest.raises(campaign_control.ControlRefused, match="worker admission is closed"):
            _run_owned_profiler(tmp_path, 0)
        controller, reservation = seen[0]
        assert controller.worker_attempt_status(
            request_id=reservation.request_id, plan_digest=reservation.stage_plan_digest,
            lineage_id=reservation.transition_id, stage_id=reservation.stage_id) == "not_acquired"
        controller.cancel_target_profile_execution(reservation)
        assert not controller._actor_profile_execution_reservations
        controller.apply_command(_command(controller.resolved, "resume-after-cancel", "resume",
                                           controller.control_revision))
        request = replace(requests[0], control_revision=controller.control_revision)
        with pytest.raises(campaign_control.ControlRefused, match="admission was cancelled"):
            controller.run_worker_stage(request)
        assert controller._lifecycle_provider.authorization is None
    finally:
        for controller, _ in seen:
            controller.close()


def test_profile_receipt_defensively_freezes_nested_request_and_returns_detached_dict():
    body = _profile_receipt(_resolved(), now=10).to_dict()
    body["profile_request"]["stage_proposal"]["nested"] = [{"value": "original"}]
    body["profile_request_digest"] = actor_preparation._digest(body["profile_request"])
    receipt = actor_lifecycle.TargetProfileReceipt(**body)
    original_digest = receipt.digest
    body["profile_request"]["stage_proposal"]["nested"][0]["value"] = "caller-mutated"
    with pytest.raises(TypeError):
        receipt.profile_request["stage_proposal"]["nested"][0]["value"] = "mutated"
    detached = receipt.to_dict()
    detached["profile_request"]["stage_proposal"]["nested"][0]["value"] = "serialized-mutated"
    assert receipt.profile_request["stage_proposal"]["nested"][0]["value"] == "original"
    assert receipt.digest == original_digest
    assert replace(receipt).to_dict() == receipt.to_dict()


def test_registered_producer_receipt_cannot_be_mutated_through_public_accessor(tmp_path, monkeypatch):
    original = campaign_control.CampaignController.register_target_profile_producer
    observed = []
    def verify_immutable(controller, producer):
        original(controller, producer)
        target = producer.mechanism.loaded_identity["target_revision_digest"]
        receipt = controller.current_actor_profile(target)
        digest = receipt.digest
        with pytest.raises(TypeError):
            receipt.profile_request["stage_proposal"]["stage_class"] = "forged"
        receipt.to_dict()["profile_request"]["stage_proposal"]["stage_class"] = "forged"
        assert receipt.digest == digest
        assert controller.current_actor_profile(target) is receipt
        observed.append(digest)
    monkeypatch.setattr(campaign_control.CampaignController, "register_target_profile_producer", verify_immutable)
    _run_owned_profiler(tmp_path, 0)
    assert len(observed) == 1


@pytest.mark.parametrize("failure", ["changed", "fifo", "oversized"])
def test_prelaunch_binary_refusal_cancels_without_stranding_reservation(tmp_path, monkeypatch, failure):
    original = campaign_control.CampaignController.reserve_target_profile_execution
    seen = []
    def change_after_reserve(controller, **kwargs):
        reservation = original(controller, **kwargs)
        seen.append(controller)
        binary = tmp_path / "fake-profiler"
        if failure == "fifo":
            binary.unlink()
            os.mkfifo(binary)
        elif failure == "oversized":
            from .observation_binding import _MAX_PROVIDER_BYTES
            with binary.open("wb") as stream:
                stream.truncate(_MAX_PROVIDER_BYTES + 1)
        else:
            binary.write_text("changed after profile selection")
        return reservation
    monkeypatch.setattr(campaign_control.CampaignController, "reserve_target_profile_execution", change_after_reserve)
    try:
        with pytest.raises(target_profile_execution.ProfileExecutionRefused, match="unreadable|bytes changed"):
            _run_owned_profiler(tmp_path, 0)
        assert seen[0]._lifecycle_provider.authorization is None
        assert not seen[0]._actor_profile_execution_reservations
        assert seen[0]._actor_profile_cancelled_attempts
    finally:
        for controller in seen:
            controller.close()


def test_unknown_attempt_after_engine_entry_retains_reservation(tmp_path, monkeypatch):
    original = campaign_control.CampaignController.reserve_target_profile_execution
    seen = []
    def install_fault(controller, **kwargs):
        reservation = original(controller, **kwargs)
        seen.append((controller, reservation))
        def uncertain(*args, **kwargs):
            raise RuntimeError("injected engine-boundary uncertainty")
        monkeypatch.setattr(controller._worker_lifecycle, "run_stage", uncertain)
        return reservation
    monkeypatch.setattr(campaign_control.CampaignController, "reserve_target_profile_execution", install_fault)
    try:
        with pytest.raises(RuntimeError, match="engine-boundary uncertainty"):
            _run_owned_profiler(tmp_path, 0)
        controller, reservation = seen[0]
        assert controller.worker_attempt_status(
            request_id=reservation.request_id, plan_digest=reservation.stage_plan_digest,
            lineage_id=reservation.transition_id, stage_id=reservation.stage_id) == "unknown"
        with pytest.raises(campaign_control.ControlRefused, match="attempting/unknown"):
            controller.cancel_target_profile_execution(reservation)
        assert controller._actor_profile_execution_reservations[reservation.reservation_id] is reservation
    finally:
        for controller, _ in seen:
            controller.close()


def test_cancel_racing_actual_provider_admission_refuses_then_worker_finishes(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original_reserve = campaign_control.CampaignController.reserve_target_profile_execution
    original_authorize = MockProvider.authorize
    seen, failures = [], []
    def capture(controller, **kwargs):
        reservation = original_reserve(controller, **kwargs)
        seen.append((controller, reservation))
        return reservation
    def gated_authorize(provider, *args):
        entered.set()
        assert release.wait(5), "review gate was not released"
        return original_authorize(provider, *args)
    def run():
        try:
            _run_owned_profiler(tmp_path, 0)
        except BaseException as exc:
            failures.append(exc)
    monkeypatch.setattr(campaign_control.CampaignController, "reserve_target_profile_execution", capture)
    monkeypatch.setattr(MockProvider, "authorize", gated_authorize)
    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert entered.wait(5)
        controller, reservation = seen[0]
        with pytest.raises(campaign_control.ControlRefused, match="attempting/unresolved"):
            controller.cancel_target_profile_execution(reservation)
        assert controller._actor_profile_execution_reservations[reservation.reservation_id] is reservation
        assert not controller._actor_trusted_held_receipts
    finally:
        release.set()
        worker.join(10)
        assert not worker.is_alive()
        for controller, _ in seen:
            controller.close()
    assert not failures, failures


@pytest.mark.parametrize("failure", ["fifo", "oversized"])
def test_actor_uses_bounded_regular_executable_read_before_provider(tmp_path, monkeypatch, failure):
    original = actor_lifecycle.ActorLifecycleAdapter.invoke
    providers = []
    def replace_binary(adapter, reservation, backend, prompt):
        providers.append(adapter.controller._lifecycle_provider)
        binary = Path(backend.binary)
        if failure == "fifo":
            binary.unlink()
            os.mkfifo(binary)
        else:
            from .observation_binding import _MAX_PROVIDER_BYTES
            with binary.open("wb") as stream:
                stream.truncate(_MAX_PROVIDER_BYTES + 1)
        return original(adapter, reservation, backend, prompt)
    monkeypatch.setattr(actor_lifecycle.ActorLifecycleAdapter, "invoke", replace_binary)
    with pytest.raises(actor_preparation.PreparationRefused, match="settled by owner"):
        _run_owned_actor(tmp_path)
    assert providers and all(provider.authorization is None for provider in providers)
