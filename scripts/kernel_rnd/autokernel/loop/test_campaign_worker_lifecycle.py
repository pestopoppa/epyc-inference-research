"""V2 controller/journal integration over tiny workers and mock containment."""
from __future__ import annotations

from dataclasses import replace
import fcntl
import os
import json
from pathlib import Path
import sys
import threading
import time

import pytest

from .. import journal as journal_module
from . import campaign_control as control
from . import campaign_service as service
from . import worker_lifecycle as lifecycle
from .test_campaign_control import _command, _resolved
from .test_worker_lifecycle import MockProvider, ReceiptProvider


def _request(root: Path, revision: int, code: str) -> lifecycle.StageRequest:
    return lifecycle.StageRequest(
        "worker-request", "1" * 64, "lineage-v2", "stage-v2", "sampling",
        (sys.executable, "-B", "-c", code),
        {"PATH": os.environ.get("PATH", "/usr/bin"),
         "PYTHONDONTWRITEBYTECODE": "1"}, root, "2" * 64,
        1.0, 0.5, revision)


def test_v2_is_explicit_closed_and_absent_provider_cannot_launch(tmp_path):
    resolved = _resolved("campaign-v2-absent")
    with control.CampaignController(
            resolved, tmp_path / "store", snapshot_version=2) as controller:
        snapshot = controller.publish_snapshot()
        assert snapshot["schema"] == control.SNAPSHOT_SCHEMA_V2
        assert set(snapshot) == control.SNAPSHOT_V2_FIELDS
        assert snapshot["active_worker"] is None
        assert snapshot["execution_authorized"] is False
        assert snapshot["execution_capability_available"] is False
        assert snapshot["worker_lifecycle_revision"] == 0
        with pytest.raises(control.ControlRefused, match="closed: paused"):
            controller.run_worker_stage(_request(tmp_path, 0, "pass"))
        resumed = controller.apply_command(_command(resolved, "resume", "resume", 0))
        assert resumed["completed"] is True
        assert set(resumed) == lifecycle.COMMAND_RESULT_FIELDS
        with pytest.raises(lifecycle.WaitingAuthority, match="unavailable"):
            controller.run_worker_stage(_request(tmp_path, 1, "pass"))
    entries = journal_module.Journal(
        str(tmp_path / "store" / "journal"), campaign_id=resolved.campaign_id).read_all()
    assert entries[0].kind == journal_module.KIND_CAMPAIGN_SUPERVISOR_EVENT
    assert entries[0].payload["schema"] == \
        journal_module.CAMPAIGN_SUPERVISOR_EVENT_SCHEMA_V2
    # A v1 reader cannot silently project a store that acquired v2 events.
    with pytest.raises(control.ControlRefused, match="downgrade is refused"):
        control.CampaignController(resolved, tmp_path / "store").__enter__()


def test_v1_to_v2_start_fence_appends_without_rewriting_old_bytes(tmp_path):
    resolved = _resolved("campaign-v1-to-v2-fence")
    store = tmp_path / "store"
    with control.CampaignController(resolved, store):
        pass
    journal_path = store / "journal" / journal_module.BASE_SHARD_NAME
    old_bytes = journal_path.read_bytes()
    with control.CampaignController(resolved, store, snapshot_version=2):
        pass
    new_bytes = journal_path.read_bytes()
    assert new_bytes.startswith(old_bytes) and len(new_bytes) > len(old_bytes)
    with control.CampaignController(resolved, store, snapshot_version=2):
        pass
    with pytest.raises(control.ControlRefused, match="downgrade is refused"):
        control.CampaignController(resolved, store).__enter__()


def test_pause_ack_is_fast_incomplete_then_durably_completes_after_quiescence(tmp_path):
    resolved = _resolved("campaign-v2-pause")
    provider = MockProvider(tmp_path / "containers")
    (tmp_path / "containers").mkdir(mode=0o700)
    store = tmp_path / "store"
    controller = control.CampaignController(
        resolved, store, snapshot_version=2, lifecycle_provider=provider,
        readiness_check=lambda: (True, None))
    controller.__enter__()
    try:
        resume = controller.apply_command(_command(resolved, "resume", "resume", 0))
        assert resume["completed"] and resume["observed_state"] == "running"
        results = []
        child_thread = threading.Thread(target=lambda: results.append(
            controller.run_worker_stage(_request(
                tmp_path, 1, "import time; time.sleep(0.2)"))))
        child_thread.start()
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            if controller.snapshot()["worker_activity_at"] is not None:
                break
            time.sleep(0.005)
        started = time.monotonic()
        pause = controller.apply_command(_command(resolved, "pause", "pause", 1))
        assert time.monotonic() - started < 0.15
        assert pause["accepted"] is True and pause["completed"] is False
        assert pause["observed_state"] == "pausing"
        assert set(pause) == lifecycle.COMMAND_RESULT_FIELDS
        child_thread.join(2)
        assert not child_thread.is_alive() and results[0].accepted
        completed = controller.command_results["pause"]
        assert completed["completed"] is True
        assert completed["observed_state"] == "paused"
        assert completed["completed_at"] is not None
        snapshot = controller.publish_snapshot()
        assert snapshot["active_worker"] is None
        assert snapshot["worker_activity_at"] is None
        assert snapshot["last_scientific_result_at"] is None
        assert snapshot["execution_authorized"] is False
    finally:
        controller.close()

    with control.CampaignController(
            resolved, store, snapshot_version=2,
            lifecycle_provider=provider, readiness_check=lambda: (True, None)) as replayed:
        result = replayed.command_results["pause"]
        assert result["completed"] is True and result["observed_state"] == "paused"
        duplicate = dict(_command(resolved, "pause", "pause", 1),
                         expected_control_revision=999)
        assert replayed.apply_command(duplicate) == result


def test_journal_refuses_open_or_malformed_lifecycle_payload(tmp_path):
    journal = journal_module.Journal(str(tmp_path / "journal"), campaign_id="campaign")
    journal.initialize()
    malformed = {
        "schema": lifecycle.EVENT_SCHEMA, "event": "OWNED_CONTAINER_CREATED",
        "campaign_id": "campaign", "config_digest": "3" * 64,
        "config_generation": 1, "supervisor_id": "supervisor",
        "supervisor_incarnation": 1, "worker_id": "worker", "worker_generation": 1,
        "request_id": "request", "plan_digest": "4" * 64, "lineage_id": "lineage",
        "stage_id": "stage", "grant_id": "grant", "grant_generation": 1,
        "container_id": "epyc-autokernel-fixture", "control_revision": 0,
        "occurred_at": "2026-09-09T00:00:00Z", "data": {"unrelated": True},
    }
    with pytest.raises(ValueError, match="missing/unknown"):
        journal.append(journal_module.KIND_WORKER_LIFECYCLE, malformed)


def test_pending_acquisition_is_visible_and_keeps_pause_incomplete(tmp_path):
    resolved = _resolved("campaign-v2-pending-acquisition")
    provider = MockProvider(tmp_path / "containers")
    (tmp_path / "containers").mkdir(mode=0o700)
    entered = threading.Event()
    finish = threading.Event()

    def ambiguous_authorize(_request, _container_id, _deadline):
        entered.set()
        assert finish.wait(1.0)
        raise TimeoutError("fixture ambiguous acquisition")

    provider.authorize = ambiguous_authorize
    controller = control.CampaignController(
        resolved, tmp_path / "store", snapshot_version=2,
        lifecycle_provider=provider, readiness_check=lambda: (True, None))
    controller.__enter__()
    controller.apply_command(_command(resolved, "resume", "resume", 0))
    failures = []
    thread = threading.Thread(target=lambda: _capture_failure(
        failures, controller.run_worker_stage, _request(tmp_path, 1, "pass")))
    thread.start()
    assert entered.wait(1.0)
    snapshot = controller.snapshot()
    assert snapshot["active_worker"] is None
    assert snapshot["observed_state"] == "ownership_unresolved"
    assert snapshot["prerequisite_reason"].startswith("worker_acquisition_pending:")
    pause = controller.apply_command(_command(resolved, "pause", "pause", 1))
    assert pause["accepted"] and not pause["completed"]
    finish.set()
    thread.join(2)
    assert not thread.is_alive() and isinstance(failures[0], lifecycle.ContainmentFailure)
    assert not controller.command_results["pause"]["completed"]
    assert controller.reconcile_workers() is None
    assert controller.command_results["pause"]["completed"]
    controller.close()


def test_pending_acquisition_replays_and_resolves_by_exact_provider_inspection(tmp_path):
    resolved = _resolved("campaign-v2-pending-replay")
    provider = MockProvider(tmp_path / "containers")
    (tmp_path / "containers").mkdir(mode=0o700)
    original_authorize = provider.authorize
    provider.authorize = lambda *_args: (_ for _ in ()).throw(
        TimeoutError("fixture lost authorization reply"))
    store = tmp_path / "store"
    crashed = control.CampaignController(
        resolved, store, snapshot_version=2, lifecycle_provider=provider,
        readiness_check=lambda: (True, None))
    crashed.__enter__()
    crashed.apply_command(_command(resolved, "resume", "resume", 0))
    with pytest.raises(lifecycle.ContainmentFailure, match="ambiguous"):
        crashed.run_worker_stage(_request(tmp_path, 1, "pass"))
    before_revision = crashed.snapshot()["worker_lifecycle_revision"]
    with pytest.raises(control.ControlRefused, match="ownership retained"):
        crashed.close()
    # Test-only supervisor crash cut: release local descriptors without changing
    # the durable Journal that the successor must project and reconcile.
    crashed._active_acquisition_events = []
    crashed._acquisition_projection = lifecycle.project_acquisitions([])
    crashed.close()
    provider.authorize = original_authorize

    with control.CampaignController(
            resolved, store, snapshot_version=2, lifecycle_provider=provider,
            readiness_check=lambda: (True, None)) as recovered:
        pending = recovered.snapshot()
        assert pending["active_worker"] is None
        assert pending["observed_state"] == "ownership_unresolved"
        assert pending["prerequisite_reason"].startswith("worker_acquisition_pending:")
        assert recovered.reconcile_workers() is None
        settled = recovered.snapshot()
        assert settled["observed_state"] == "running"
        assert settled["prerequisite_reason"] is None
        assert settled["worker_lifecycle_revision"] == before_revision + 1
        successor = recovered.run_worker_stage(lifecycle.StageRequest(
            "worker-after-resolved-gap", "1" * 64, "lineage-v2", "stage-after-gap",
            "sampling", (sys.executable, "-B", "-c", "pass"),
            {"PATH": os.environ.get("PATH", "/usr/bin"),
             "PYTHONDONTWRITEBYTECODE": "1"}, tmp_path, "2" * 64,
            1.0, 0.5, 1))
        assert successor.accepted and successor.worker_generation == 2


def test_restart_finishes_handoff_and_old_owner_lifecycle_without_stage_repeat(tmp_path):
    resolved = _resolved("campaign-v2-handoff-replay")
    provider = MockProvider(tmp_path / "containers")
    (tmp_path / "containers").mkdir(mode=0o700)
    store = tmp_path / "store"
    crashed = control.CampaignController(
        resolved, store, snapshot_version=2, lifecycle_provider=provider,
        readiness_check=lambda: (True, None))
    crashed.__enter__()
    crashed.apply_command(_command(resolved, "resume", "resume", 0))

    def crash_after_launch_intent(phase):
        if phase == "OWNED_LAUNCH_INTENT":
            raise lifecycle.SimulatedCrash()

    crashed._worker_lifecycle.fault_hook = crash_after_launch_intent
    with pytest.raises(lifecycle.SimulatedCrash):
        crashed.run_worker_stage(_request(
            tmp_path, 1, "raise SystemExit('must never execute')"))
    assert crashed.snapshot()["active_worker"]["state"] == "intent"
    crashed._active_acquisition_events = []
    crashed._acquisition_projection = lifecycle.project_acquisitions([])
    crashed._active_worker_events = []
    crashed._worker_projection = lifecycle.project_events([])
    crashed.close()

    with control.CampaignController(
            resolved, store, snapshot_version=2, lifecycle_provider=provider,
            readiness_check=lambda: (True, None)) as recovered:
        terminal = recovered.reconcile_workers()
        assert terminal is not None and not terminal.accepted
        assert recovered.snapshot()["active_worker"] is None
        assert recovered.observed_state == "running"
        assert recovered.prerequisite_reason is None


def test_denied_attempt_generation_gap_replays_before_next_success(tmp_path):
    resolved = _resolved("campaign-v2-denied-generation-gap")
    provider = MockProvider(tmp_path / "containers")
    (tmp_path / "containers").mkdir(mode=0o700)
    store = tmp_path / "store"
    with control.CampaignController(
            resolved, store, snapshot_version=2, lifecycle_provider=provider,
            readiness_check=lambda: (True, None)) as controller:
        controller.apply_command(_command(resolved, "resume", "resume", 0))
        first = controller.run_worker_stage(_request(tmp_path, 1, "pass"))
        original_authorize = provider.authorize
        provider.authorize = lambda request, container_id, _deadline: (
            lifecycle.AuthorizationDenied(request.request_id, container_id,
                                          "fixture definite denial"))
        with pytest.raises(lifecycle.WaitingAuthority):
            controller.run_worker_stage(replace(
                _request(tmp_path, 1, "pass"), request_id="denied-attempt"))
        provider.authorize = original_authorize
        third = controller.run_worker_stage(replace(
            _request(tmp_path, 1, "pass"), request_id="third-attempt"))
        assert third.worker_generation == first.worker_generation + 2
    with control.CampaignController(
            resolved, store, snapshot_version=2, lifecycle_provider=provider,
            readiness_check=lambda: (True, None)) as recovered:
        fourth = recovered.run_worker_stage(replace(
            _request(tmp_path, 1, "pass"), request_id="fourth-attempt"))
        assert fourth.worker_generation == third.worker_generation + 1


def test_controller_reads_only_exact_pinned_terminal_stdout(tmp_path):
    resolved = _resolved("campaign-v2-stdout")
    provider = MockProvider(tmp_path / "containers")
    (tmp_path / "containers").mkdir(mode=0o700)
    controller = control.CampaignController(
        resolved, tmp_path / "store", snapshot_version=2,
        lifecycle_provider=provider, readiness_check=lambda: (True, None))
    controller.__enter__()
    try:
        controller.apply_command(_command(resolved, "resume", "resume", 0))
        request = replace(
            _request(tmp_path, 1, "print('actor-json')"),
            request_id="stdout-request", plan_digest="7" * 64,
            lineage_id="stdout-lineage", stage_id="stdout-stage")
        terminal = controller.run_worker_stage(request)
        assert terminal.accepted and terminal.result_digest is not None
        arguments = {
            "request_id": request.request_id, "plan_digest": request.plan_digest,
            "lineage_id": request.lineage_id, "stage_id": request.stage_id,
            "worker_id": terminal.worker_id,
            "worker_generation": terminal.worker_generation,
            "result_digest": terminal.result_digest, "max_bytes": 4096,
        }
        assert controller.read_worker_stdout(**arguments) == b"actor-json\n"
        with pytest.raises(lifecycle.LifecycleRefused, match="identity differs"):
            controller.read_worker_stdout(**(arguments | {"result_digest": "8" * 64}))
        with pytest.raises(lifecycle.LifecycleRefused, match="oversized"):
            controller.read_worker_stdout(**(arguments | {"max_bytes": 1}))

        leaf = lifecycle.WorkerLifecycle.stdout_leaf(
            terminal.worker_id, terminal.worker_generation)
        locked_fd = os.open(leaf, os.O_RDONLY, dir_fd=controller._runtime_root.fd)
        try:
            fcntl.flock(locked_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            started = time.monotonic()
            with pytest.raises(lifecycle.LifecycleRefused, match="unavailable"):
                controller.read_worker_stdout(**arguments)
            assert time.monotonic() - started < 0.15
        finally:
            os.close(locked_fd)

        stdout_path = tmp_path / "store" / leaf
        with stdout_path.open("r+b") as stream:
            stream.write(b"ACTOR-JSON\n")
            stream.flush()
            os.fsync(stream.fileno())
        with pytest.raises(lifecycle.LifecycleRefused, match="content differs"):
            controller.read_worker_stdout(**arguments)

        replacement = tmp_path / "store" / "replacement.stdout"
        replacement.write_bytes(b"actor-json\n")
        replacement.chmod(0o600)
        os.replace(replacement, tmp_path / "store" / leaf)
        with pytest.raises(lifecycle.LifecycleRefused, match="replaced"):
            controller.read_worker_stdout(**arguments)
        (tmp_path / "store" / leaf).unlink()
        os.mkfifo(tmp_path / "store" / leaf, mode=0o600)
        started = time.monotonic()
        with pytest.raises(lifecycle.LifecycleRefused, match="replaced"):
            controller.read_worker_stdout(**arguments)
        assert time.monotonic() - started < 0.15
        (tmp_path / "store" / leaf).unlink()
        (tmp_path / "store" / leaf).mkdir(mode=0o700)
        with pytest.raises(lifecycle.LifecycleRefused, match="replaced"):
            controller.read_worker_stdout(**arguments)
        (tmp_path / "store" / leaf).rmdir()
        with pytest.raises(lifecycle.LifecycleRefused, match="unavailable"):
            controller.read_worker_stdout(**arguments)
    finally:
        controller.close()


def test_controller_stdout_read_does_not_hold_mutex_and_close_fences_return(
        tmp_path, monkeypatch):
    resolved = _resolved("campaign-v2-stdout-close")
    provider = MockProvider(tmp_path / "containers")
    (tmp_path / "containers").mkdir(mode=0o700)
    controller = control.CampaignController(
        resolved, tmp_path / "store", snapshot_version=2,
        lifecycle_provider=provider, readiness_check=lambda: (True, None))
    controller.__enter__()
    controller.apply_command(_command(resolved, "resume", "resume", 0))
    request = replace(
        _request(tmp_path, 1, "print('bounded')"),
        request_id="stdout-close-request", plan_digest="8" * 64,
        lineage_id="stdout-close-lineage", stage_id="stdout-close-stage")
    terminal = controller.run_worker_stage(request)
    assert terminal.result_digest is not None
    entered, release = threading.Event(), threading.Event()
    original = control.read_stable_fd

    def blocked_read(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return original(*args, **kwargs)

    monkeypatch.setattr(control, "read_stable_fd", blocked_read)
    errors = []

    def read():
        try:
            controller.read_worker_stdout(
                request_id=request.request_id, plan_digest=request.plan_digest,
                lineage_id=request.lineage_id, stage_id=request.stage_id,
                worker_id=terminal.worker_id,
                worker_generation=terminal.worker_generation,
                result_digest=terminal.result_digest, max_bytes=4096)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=read)
    try:
        thread.start()
        assert entered.wait(2)
        started = time.monotonic()
        controller.close()
        assert time.monotonic() - started < 0.15
        release.set()
        thread.join(2)
        assert not thread.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], lifecycle.LifecycleRefused)
    finally:
        release.set()
        thread.join(2)
        controller.close()


@pytest.mark.parametrize("failure", ["secure_open", "fstat"])
def test_stdout_identity_capture_failure_preserves_terminal_and_held_cost(
        tmp_path, monkeypatch, failure):
    resolved = _resolved(f"campaign-v2-stdout-capture-{failure}")
    provider = ReceiptProvider(tmp_path / "containers")
    (tmp_path / "containers").mkdir(mode=0o700)
    with control.CampaignController(
            resolved, tmp_path / "store", snapshot_version=2,
            lifecycle_provider=provider,
            readiness_check=lambda: (True, None)) as controller:
        controller.apply_command(_command(resolved, "resume", "resume", 0))
        engine = controller._worker_lifecycle
        request = replace(
            _request(tmp_path, 1, "print('retained terminal')"),
            request_id=f"stdout-capture-{failure}", plan_digest="a" * 64,
            lineage_id="stdout-capture-lineage", stage_id="stdout-capture-stage")
        if failure == "secure_open":
            original_open = engine.runtime.open_leaf

            def faulty_open(name, flags, mode=0o600):
                if (name.endswith(".stdout.log")
                        and flags & os.O_ACCMODE == os.O_RDONLY):
                    raise lifecycle.SecureRuntimeError("fixture secure refusal")
                return original_open(name, flags, mode)

            monkeypatch.setattr(engine.runtime, "open_leaf", faulty_open)
        else:
            original_open = engine.runtime.open_leaf
            original_fstat = lifecycle.os.fstat
            targeted = {"fd": -1}

            def tracked_open(name, flags, mode=0o600):
                fd = original_open(name, flags, mode)
                if (name.endswith(".stdout.log")
                        and flags & os.O_ACCMODE == os.O_RDONLY):
                    targeted["fd"] = fd
                return fd

            def faulty_fstat(fd):
                if fd == targeted["fd"]:
                    targeted["fd"] = -1
                    raise OSError("fixture fstat refusal")
                return original_fstat(fd)

            monkeypatch.setattr(engine.runtime, "open_leaf", tracked_open)
            monkeypatch.setattr(lifecycle.os, "fstat", faulty_fstat)
        terminal = controller.run_worker_stage(request)
        assert terminal.accepted and terminal.result_digest is not None
        assert controller.worker_terminal_for_request(
            request_id=request.request_id, plan_digest=request.plan_digest,
            lineage_id=request.lineage_id, stage_id=request.stage_id) == terminal
        assert controller.worker_held_claim_receipt(terminal).proposal_id == request.request_id
        with pytest.raises(lifecycle.LifecycleRefused, match="unavailable"):
            controller.read_worker_stdout(
                request_id=request.request_id, plan_digest=request.plan_digest,
                lineage_id=request.lineage_id, stage_id=request.stage_id,
                worker_id=terminal.worker_id,
                worker_generation=terminal.worker_generation,
                result_digest=terminal.result_digest, max_bytes=4096)


def test_truncated_stdout_is_not_available_as_authenticated_actor_output(tmp_path):
    resolved = _resolved("campaign-v2-stdout-truncated")
    provider = ReceiptProvider(tmp_path / "containers")
    (tmp_path / "containers").mkdir(mode=0o700)
    with control.CampaignController(
            resolved, tmp_path / "store", snapshot_version=2,
            lifecycle_provider=provider,
            readiness_check=lambda: (True, None)) as controller:
        controller.apply_command(_command(resolved, "resume", "resume", 0))
        request = replace(
            _request(tmp_path, 1, "import os; os.write(1, b'x' * 65537)"),
            request_id="stdout-truncated-request", plan_digest="b" * 64,
            lineage_id="stdout-truncated-lineage", stage_id="stdout-truncated-stage")
        terminal = controller.run_worker_stage(request)
        assert terminal.accepted and terminal.result_digest is not None
        assert controller.worker_held_claim_receipt(terminal).proposal_id == request.request_id
        with pytest.raises(lifecycle.LifecycleRefused, match="truncated"):
            controller.read_worker_stdout(
                request_id=request.request_id, plan_digest=request.plan_digest,
                lineage_id=request.lineage_id, stage_id=request.stage_id,
                worker_id=terminal.worker_id,
                worker_generation=terminal.worker_generation,
                result_digest=terminal.result_digest, max_bytes=65537)


def test_missing_stdout_identity_does_not_erase_terminal_or_held_cost(
        tmp_path, monkeypatch):
    resolved = _resolved("campaign-v2-stdout-missing")
    provider = ReceiptProvider(tmp_path / "containers")
    (tmp_path / "containers").mkdir(mode=0o700)
    with control.CampaignController(
            resolved, tmp_path / "store", snapshot_version=2,
            lifecycle_provider=provider,
            readiness_check=lambda: (True, None)) as controller:
        controller.apply_command(_command(resolved, "resume", "resume", 0))
        engine = controller._worker_lifecycle
        monkeypatch.setattr(engine, "_stdout_identity", lambda *_args: None)
        request = replace(
            _request(tmp_path, 1, "print('retained terminal')"),
            request_id="stdout-missing-request", plan_digest="9" * 64,
            lineage_id="stdout-missing-lineage", stage_id="stdout-missing-stage")
        terminal = controller.run_worker_stage(request)
        assert terminal.accepted and terminal.result_digest is not None
        assert controller.worker_terminal_for_request(
            request_id=request.request_id, plan_digest=request.plan_digest,
            lineage_id=request.lineage_id, stage_id=request.stage_id) == terminal
        held = controller.worker_held_claim_receipt(terminal)
        assert held.proposal_id == request.request_id
        with pytest.raises(lifecycle.LifecycleRefused, match="unavailable"):
            controller.read_worker_stdout(
                request_id=request.request_id, plan_digest=request.plan_digest,
                lineage_id=request.lineage_id, stage_id=request.stage_id,
                worker_id=terminal.worker_id,
                worker_generation=terminal.worker_generation,
                result_digest=terminal.result_digest, max_bytes=4096)


def _capture_failure(target, callback, *args):
    try:
        callback(*args)
    except BaseException as exc:
        target.append(exc)


def test_cleanup_failure_blocks_replacement_and_leaves_pause_incomplete(tmp_path):
    resolved = _resolved("campaign-v2-cleanup-failure")
    provider = MockProvider(tmp_path / "containers", release_ok=False)
    (tmp_path / "containers").mkdir(mode=0o700)
    controller = control.CampaignController(
        resolved, tmp_path / "store", snapshot_version=2,
        lifecycle_provider=provider, readiness_check=lambda: (True, None))
    controller.__enter__()
    controller.apply_command(_command(resolved, "resume", "resume", 0))
    try:
        with pytest.raises(lifecycle.ContainmentFailure):
            controller.run_worker_stage(_request(tmp_path, 1, "pass"))
        pause = controller.apply_command(_command(resolved, "pause", "pause", 1))
        assert pause["accepted"] and not pause["completed"]
        assert controller.snapshot()["active_worker"]["state"] == "teardown_failed"
        with pytest.raises(control.ControlRefused, match="active or unresolved"):
            controller.run_worker_stage(_request(tmp_path, 2, "pass"))
        with pytest.raises(control.ControlRefused, match="ownership retained"):
            controller.close()
    finally:
        # Mock release refusal is the retained authority seam; process/container
        # cleanup itself was proven complete, so allow exact controller teardown.
        controller._worker_projection = lifecycle.project_events([])
        controller._worker_run_active = False
        controller.close()


def test_cli_v2_selection_changes_projection_but_creates_no_authority(tmp_path, capsys):
    resolved = _resolved("campaign-v2-cli")
    resolved_path = tmp_path / "resolved.json"
    resolved_path.write_text(json.dumps(resolved.to_dict()), encoding="utf-8")
    assert service.main([
        "--resolved-campaign", str(resolved_path), "--store", str(tmp_path / "store"),
        "--snapshot-version", "2", "--once",
    ]) == 0
    snapshot = json.loads(capsys.readouterr().out)
    assert snapshot["schema"] == control.SNAPSHOT_SCHEMA_V2
    assert snapshot["execution_authorized"] is False
    assert snapshot["execution_capability_available"] is False
    assert set(snapshot) == control.SNAPSHOT_V2_FIELDS


def test_pending_pause_can_escalate_to_terminal_drain_without_late_overwrite(tmp_path):
    resolved = _resolved("campaign-v2-drain-escalation")
    provider = MockProvider(tmp_path / "containers")
    (tmp_path / "containers").mkdir(mode=0o700)
    store = tmp_path / "store"
    controller = control.CampaignController(
        resolved, store, snapshot_version=2, lifecycle_provider=provider,
        readiness_check=lambda: (True, None))
    controller.__enter__()
    controller.apply_command(_command(resolved, "resume", "resume", 0))
    results = []
    thread = threading.Thread(target=lambda: results.append(controller.run_worker_stage(
        _request(tmp_path, 1, "import time; time.sleep(0.2)"))))
    thread.start()
    deadline = time.monotonic() + 1
    while controller.snapshot()["worker_activity_at"] is None \
            and time.monotonic() < deadline:
        time.sleep(0.005)
    pause_command = _command(resolved, "pause", "pause", 1)
    pause = controller.apply_command(pause_command)
    assert not pause["completed"]
    drain_command = _command(resolved, "drain", "drain", 2)
    drain = controller.apply_command(drain_command)
    assert drain["accepted"] and not drain["completed"]
    superseded_pause = controller.command_results["pause"]
    assert superseded_pause["completed"] is True
    assert superseded_pause["desired_state"] == "drained"
    assert "superseded" in superseded_pause["completion_reason"]
    assert controller.desired_state == "drained"
    thread.join(2)
    assert not thread.is_alive() and results[0].accepted
    assert controller.command_results["drain"]["completed"] is True
    assert controller.observed_state == "drained"
    assert controller.apply_command(dict(drain_command,
                                         expected_control_revision=999)) == \
        controller.command_results["drain"]
    controller.close()

    with control.CampaignController(
            resolved, store, snapshot_version=2, lifecycle_provider=provider,
            readiness_check=lambda: (True, None)) as replayed:
        assert replayed.desired_state == replayed.observed_state == "drained"
        assert replayed.command_results["pause"]["desired_state"] == "drained"
        assert replayed.command_results["drain"]["completed"] is True
        with pytest.raises(control.ControlRefused, match="terminal"):
            replayed.apply_command(_command(resolved, "resume-again", "resume", 3))


@pytest.mark.parametrize("changes", [
    {"desired_state": "running", "observed_state": "running",
     "completion_reason": "running"},
    {"completion_reason": "invented completion"},
    {"completed_at": "2000-01-01T00:00:00Z"},
    {"observed_state": "draining"},
])
def test_v2_command_result_rejects_nonproducer_completion_semantics(tmp_path,
                                                                    changes):
    resolved = _resolved("campaign-v2-result-semantics")
    with control.CampaignController(
            resolved, tmp_path / "store", snapshot_version=2) as controller:
        result = controller.apply_command(_command(resolved, "pause", "pause", 0))
        result.update(changes)
        with pytest.raises(lifecycle.LifecycleRefused):
            lifecycle.validate_command_result_v2(result)


@pytest.mark.parametrize(("field", "value"), [
    ("config_digest", "bad"), ("sequence", True), ("stream_epoch", 0),
    ("desired_state", "invented"), ("generated_at", "not-a-time"),
])
def test_v2_snapshot_validator_rejects_malformed_inherited_fields(tmp_path, field, value):
    resolved = _resolved("campaign-v2-invalid-snapshot")
    with control.CampaignController(
            resolved, tmp_path / "store", snapshot_version=2) as controller:
        snapshot = controller.snapshot()
        snapshot[field] = value
        with pytest.raises(control.ControlRefused):
            control.validate_snapshot_v2(snapshot)


def test_v2_snapshot_validator_rejects_arbitrary_worker_state(tmp_path):
    resolved = _resolved("campaign-v2-invalid-worker")
    provider = MockProvider(tmp_path / "containers")
    (tmp_path / "containers").mkdir(mode=0o700)
    with control.CampaignController(
            resolved, tmp_path / "store", snapshot_version=2,
            lifecycle_provider=provider, readiness_check=lambda: (True, None)) as controller:
        controller.apply_command(_command(resolved, "resume", "resume", 0))
        thread = threading.Thread(target=lambda: controller.run_worker_stage(
            _request(tmp_path, 1, "import time; time.sleep(0.1)")))
        thread.start()
        deadline = time.monotonic() + 1
        while controller.snapshot()["active_worker"] is None and time.monotonic() < deadline:
            time.sleep(0.005)
        snapshot = controller.snapshot()
        snapshot["active_worker"]["state"] = "browser-invented"
        with pytest.raises(control.ControlRefused, match="state"):
            control.validate_snapshot_v2(snapshot)
        snapshot = controller.snapshot()
        snapshot["observed_state"] = "paused"
        with pytest.raises(control.ControlRefused, match="quiescence"):
            control.validate_snapshot_v2(snapshot)
        thread.join(2)
        assert not thread.is_alive()


def test_lifecycle_hot_path_uses_compact_active_index_not_journal_history(tmp_path,
                                                                         monkeypatch):
    resolved = _resolved("campaign-v2-compact-index")
    provider = MockProvider(tmp_path / "containers")
    (tmp_path / "containers").mkdir(mode=0o700)
    with control.CampaignController(
            resolved, tmp_path / "store", snapshot_version=2,
            lifecycle_provider=provider, readiness_check=lambda: (True, None)) as controller:
        controller.apply_command(_command(resolved, "resume", "resume", 0))
        assert controller.run_worker_stage(_request(tmp_path, 1, "pass")).accepted
        assert controller._active_worker_events == []
        assert not hasattr(controller, "_worker_events")
        monkeypatch.setattr(
            controller._journal, "read_all",
            lambda: (_ for _ in ()).throw(AssertionError("hot path rescanned journal")))
        second = _request(tmp_path, 1, "pass")
        second = lifecycle.StageRequest(
            "worker-request-2", second.plan_digest, second.lineage_id, "stage-v2-2",
            second.stage, second.argv, second.env, second.cwd,
            second.artifact_contract_digest, second.max_stage_seconds,
            second.teardown_seconds, second.control_revision)
        assert controller.run_worker_stage(second).worker_generation == 2
        assert controller.snapshot()["worker_lifecycle_revision"] > 0
