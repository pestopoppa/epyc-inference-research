#!/usr/bin/env python3
"""Hermetic lifecycle tests; mock containment is not real cgroup evidence."""
from __future__ import annotations

import copy
from dataclasses import replace
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import pytest

from ..controller.discovery_supervisor_secure import RuntimeRoot
from . import worker_bootstrap as bootstrap
from . import worker_lifecycle as lifecycle


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class MockOwnedContainer:
    """PID-identity mock only; it makes no real containment claim."""

    def __init__(self, root: Path, name: str) -> None:
        self.path = root / name
        self._owned: dict[int, lifecycle.ProcessIdentity] = {}
        self.signals: list[tuple[int, int, int]] = []
        self.killed: list[tuple[int, int]] = []

    def create(self) -> None:
        self.path.mkdir(mode=0o700)

    def identity(self):
        info = self.path.stat()
        return {"path": str(self.path), "dev": info.st_dev, "ino": info.st_ino,
                "uid": info.st_uid, "nlink": info.st_nlink,
                "mode": info.st_mode & 0o777}

    def add(self, pid: int) -> None:
        self._owned[pid] = lifecycle.process_identity(pid)

    def _capture_descendants(self) -> None:
        pending = list(self._owned)
        seen = set(pending)
        while pending:
            parent = pending.pop()
            try:
                raw = Path(f"/proc/{parent}/task/{parent}/children").read_text()
            except OSError:
                continue
            for text in raw.split():
                pid = int(text)
                if pid in seen:
                    continue
                seen.add(pid)
                try:
                    self._owned[pid] = lifecycle.process_identity(pid)
                except lifecycle.LifecycleRefused:
                    continue
                pending.append(pid)

    def pids(self) -> tuple[int, ...]:
        self._capture_descendants()
        return tuple(sorted(pid for pid, identity in self._owned.items()
                            if lifecycle.same_process(identity)))

    def populated(self) -> bool:
        return bool(self.pids())

    def signal_all(self, signum: int, identities):
        sent = False
        self._capture_descendants()
        for pid in self.pids():
            identity = self._owned[pid]
            if identities.get(pid) != identity.start_ticks:
                continue
            os.kill(pid, signum)
            self.signals.append((pid, identity.start_ticks, signum))
            sent = True
        return sent

    def kill(self) -> None:
        self._capture_descendants()
        for pid in reversed(self.pids()):
            identity = self._owned[pid]
            if lifecycle.same_process(identity):
                try:
                    os.kill(pid, signal.SIGKILL)
                    self.killed.append((pid, identity.start_ticks))
                except ProcessLookupError:
                    pass

    def wait_empty(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.populated():
                return True
            time.sleep(0.005)
        return not self.populated()

    def close_and_remove(self) -> None:
        if self.populated():
            raise lifecycle.ContainmentFailure("mock container remains populated")
        self.path.rmdir()


class MockProvider:
    def __init__(self, root: Path, *, refresh=None, release_ok: bool = True) -> None:
        self.root = root
        self.refresh_value = refresh
        self.release_ok = release_ok
        self.authorization = None
        self.released: list[tuple[str, float]] = []

    def authorize(self, request, container_id, deadline):
        del request, deadline
        grant = lifecycle.GrantReceipt(
            "fixture-grant", 1, time.monotonic() + 5.0,
            lifecycle.monotonic_clock_domain())
        self.authorization = lifecycle.AuthorizedLaunch(
            grant, container_id, MockOwnedContainer(self.root, container_id))
        return self.authorization

    def refresh(self, authorization, deadline):
        del deadline
        return self.refresh_value or authorization.grant

    def release(self, authorization, deadline):
        self.released.append((authorization.container_id, deadline))
        return self.release_ok

    def inspect_pending(self, identity, deadline):
        del deadline
        if self.authorization and self.authorization.container_id == identity.container_id:
            return lifecycle.PendingAcquisitionInspection(
                "exact", self.authorization, "fixture exact pending acquisition")
        return lifecycle.PendingAcquisitionInspection(
            "absent", None, "fixture certifies no pending acquisition")

    def inspect(self, grant, container_id, deadline):
        del deadline
        if self.authorization and self.authorization.container_id == container_id:
            if not self.authorization.container.path.exists():
                self.released.append((container_id, time.monotonic()))
                return lifecycle.RecoveryInspection(
                    "absent_released", None, "fixture confirms absent/released")
            refreshed = lifecycle.AuthorizedLaunch(
                lifecycle.GrantReceipt(grant.grant_id, grant.generation,
                                       time.monotonic() + 2.0,
                                       lifecycle.monotonic_clock_domain()),
                container_id, self.authorization.container)
            return lifecycle.RecoveryInspection("exact", refreshed, "fixture exact container")
        return lifecycle.RecoveryInspection(
            "absent_released", None, "fixture confirms absent/released")


class Harness:
    def __init__(self, *, provider=True, binding_current=True, refresh=None,
                 release_ok=True) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name)
        self.runtime = RuntimeRoot.create_or_open(self.path / "runtime")
        self.events = []
        self.acquisitions = []
        self.provider = (MockProvider(self.path / "containers", refresh=refresh,
                                      release_ok=release_ok) if provider else None)
        (self.path / "containers").mkdir(mode=0o700)
        self.binding = lifecycle.CampaignBinding(
            "campaign-fixture", "a" * 64, 1, "supervisor-fixture", 1)
        self.engine = lifecycle.WorkerLifecycle(
            binding=self.binding, runtime=self.runtime,
            event_sink=self._record_event,
            provider=self.provider,
            admission_fence=lambda _request, _grant, _now_, _until: lifecycle.StageAdmission(
                True, "admitted"),
            binding_fence=lambda _binding: binding_current,
            runtime_fence=lambda _request, _now_: lifecycle.RuntimeDirective(
                "continue", "held stage remains authorized"),
            wall_clock=_now)

    def _record_event(self, row):
        target = (self.acquisitions if row.get("schema") == lifecycle.ACQUISITION_SCHEMA
                  else self.events)
        target.append(dict(row))

    def request(self, code: str, *, stage_seconds=1.0, teardown_seconds=0.5):
        return lifecycle.StageRequest(
            "request-fixture", "b" * 64, "lineage-fixture", "stage-fixture",
            "sampling", (sys.executable, "-B", "-c", code),
            {"PATH": os.environ.get("PATH", "/usr/bin"),
             "PYTHONDONTWRITEBYTECODE": "1"}, self.path, "c" * 64,
            stage_seconds, teardown_seconds, 0)

    def close(self) -> None:
        self.runtime.close()
        self.temporary.cleanup()


def _captured(events):
    identities = [lifecycle.ProcessIdentity(**row["data"]["process"])
                  for row in events if row["event"] == "OWNED_CHILD_CAPTURED"]
    for identity in identities:
        print(f"OWNED_TEST_PID pid={identity.pid} start_ticks={identity.start_ticks} "
              f"boot_id={identity.boot_id} alive={lifecycle.same_process(identity)}")
    return identities


def test_contract_is_closed_digest_bound_and_rejects_secrets():
    contract = bootstrap.make_contract(
        nonce="0123456789abcdef", argv=("/bin/true",), env={"PATH": "/bin"}, cwd="/tmp")
    assert bootstrap.validate_contract(contract) == contract
    with pytest.raises(bootstrap.BootstrapRefused, match="missing/unknown"):
        bootstrap.validate_contract(contract | {"shell": True})
    harness = Harness()
    try:
        with pytest.raises(lifecycle.LifecycleRefused, match="secret-bearing"):
            replace(harness.request("pass"), env={"API_TOKEN": "must-not-be-persisted"})
    finally:
        harness.close()


def test_absent_provider_waits_without_intent_or_process():
    harness = Harness(provider=False)
    try:
        with pytest.raises(lifecycle.WaitingAuthority, match="unavailable"):
            harness.engine.run_stage(harness.request("pass"))
        assert harness.events == []
    finally:
        harness.close()


def test_ambiguous_authorize_error_blocks_successor_without_claiming_denial():
    harness = Harness()
    try:
        def ambiguous_authorize(_request, _container_id, _deadline):
            raise TimeoutError("reply lost after possible acquisition")

        harness.provider.authorize = ambiguous_authorize
        with pytest.raises(lifecycle.ContainmentFailure, match="outcome is ambiguous"):
            harness.engine.run_stage(harness.request("raise SystemExit('must not execute')"))
        assert harness.events == []
        assert [row["phase"] for row in harness.acquisitions] == ["INTENT"]
        assert lifecycle.project_acquisitions(harness.acquisitions).pending is not None
        with pytest.raises(lifecycle.ContainmentFailure, match="remains unresolved"):
            harness.engine.run_stage(harness.request("raise SystemExit('must not execute')"))
    finally:
        harness.close()


def test_prospective_digest_binds_full_request_and_preassigned_routing():
    harness = Harness()
    try:
        request = harness.request("pass")
        kwargs = {"worker_id": "worker-fixed", "worker_generation": 1,
                  "container_id": "epyc-autokernel-fixed"}
        digest = lifecycle.prospective_request_digest(harness.binding, request, **kwargs)
        assert digest != lifecycle.prospective_request_digest(
            harness.binding, replace(request, env={**request.env, "OMP_NUM_THREADS": "2"}),
            **kwargs)
        assert digest != lifecycle.prospective_request_digest(
            harness.binding, replace(request, teardown_seconds=0.75), **kwargs)
        assert digest != lifecycle.prospective_request_digest(
            replace(harness.binding, supervisor_incarnation=2), request, **kwargs)
        assert digest != lifecycle.prospective_request_digest(
            harness.binding, request, **(kwargs | {"worker_generation": 2}))
    finally:
        harness.close()


def test_typed_denial_durably_resolves_without_lifecycle_or_child():
    harness = Harness()
    try:
        harness.provider.authorize = lambda request, container_id, _deadline: (
            lifecycle.AuthorizationDenied(request.request_id, container_id,
                                          "fixture definite denial"))
        with pytest.raises(lifecycle.WaitingAuthority, match="definite denial"):
            harness.engine.run_stage(harness.request("raise SystemExit('must not execute')"))
        assert harness.events == []
        assert [row["phase"] for row in harness.acquisitions] == ["INTENT", "RESOLVED"]
        assert harness.acquisitions[-1]["data"]["outcome"] == "denied"
        assert lifecycle.project_acquisitions(harness.acquisitions).pending is None
    finally:
        harness.close()


def test_pending_exact_empty_acquisition_is_released_without_launch():
    harness = Harness()
    try:
        original = harness.provider.authorize

        def acquire_then_lose_reply(request, container_id, deadline):
            original(request, container_id, deadline)
            raise TimeoutError("lost reply")

        harness.provider.authorize = acquire_then_lose_reply
        with pytest.raises(lifecycle.ContainmentFailure, match="ambiguous"):
            harness.engine.run_stage(harness.request("raise SystemExit('must not execute')"))
        recovery = lifecycle.WorkerLifecycle(
            binding=harness.binding, runtime=harness.runtime,
            event_sink=harness._record_event, provider=harness.provider,
            admission_fence=lambda *_args: lifecycle.StageAdmission(True, "fixture"),
            binding_fence=lambda _binding: True,
            runtime_fence=lambda _request, _now_: lifecycle.RuntimeDirective(
                "continue", "fixture"), wall_clock=_now)
        assert recovery.reconcile_acquisition(harness.acquisitions, []) == "resolved"
        assert harness.acquisitions[-1]["data"]["outcome"] == "exact_released"
        assert harness.events == [] and harness.provider.released
    finally:
        harness.close()


def test_pending_populated_container_remains_unresolved_without_signal_or_release():
    harness = Harness()
    try:
        original = harness.provider.authorize

        def acquire_populated_then_lose_reply(request, container_id, deadline):
            authorization = original(request, container_id, deadline)
            authorization.container.create()
            authorization.container.populated = lambda: True
            raise TimeoutError("lost reply")

        harness.provider.authorize = acquire_populated_then_lose_reply
        with pytest.raises(lifecycle.ContainmentFailure, match="ambiguous"):
            harness.engine.run_stage(harness.request("raise SystemExit('must not execute')"))
        recovery = lifecycle.WorkerLifecycle(
            binding=harness.binding, runtime=harness.runtime,
            event_sink=harness._record_event, provider=harness.provider,
            admission_fence=lambda *_args: lifecycle.StageAdmission(True, "fixture"),
            binding_fence=lambda _binding: True,
            runtime_fence=lambda _request, _now_: lifecycle.RuntimeDirective(
                "continue", "fixture"), wall_clock=_now)
        with pytest.raises(lifecycle.ContainmentFailure, match="unexpectedly populated"):
            recovery.reconcile_acquisition(harness.acquisitions, [])
        assert harness.provider.released == []
        assert harness.provider.authorization.container.signals == []
        assert harness.provider.authorization.container.killed == []
        harness.provider.authorization.container.populated = lambda: False
        harness.provider.authorization.container.close_and_remove()
    finally:
        harness.close()


def test_crash_after_real_launch_intent_replays_exact_acquisition_handoff():
    harness = Harness()
    try:
        def crash_after_launch_intent(phase):
            if phase == "OWNED_LAUNCH_INTENT":
                raise lifecycle.SimulatedCrash()

        harness.engine.fault_hook = crash_after_launch_intent
        with pytest.raises(lifecycle.SimulatedCrash):
            harness.engine.run_stage(harness.request("raise SystemExit('must not execute')"))
        assert [row["phase"] for row in harness.acquisitions] == ["INTENT"]
        assert [row["event"] for row in harness.events] == ["OWNED_LAUNCH_INTENT"]
        recovered_binding = replace(harness.binding, supervisor_incarnation=2)
        recovery = lifecycle.WorkerLifecycle(
            binding=recovered_binding, runtime=harness.runtime,
            event_sink=harness._record_event, provider=harness.provider,
            admission_fence=lambda *_args: lifecycle.StageAdmission(True, "fixture"),
            binding_fence=lambda _binding: True,
            runtime_fence=lambda _request, _now_: lifecycle.RuntimeDirective(
                "continue", "fixture"), wall_clock=_now)
        assert recovery.reconcile_acquisition(
            harness.acquisitions, harness.events) == "handoff"
        terminal = recovery.reconcile(harness.events)
        assert terminal is not None and not terminal.accepted
        assert harness.acquisitions[-1]["data"]["outcome"] == "lifecycle_handoff"
        assert not lifecycle.project_events(harness.events).active
    finally:
        harness.close()


def test_lifecycle_handoff_rejects_request_content_not_bound_by_prospective_digest():
    harness = Harness()
    try:
        assert harness.engine.run_stage(harness.request("pass")).accepted
        launch = copy.deepcopy(next(
            row for row in harness.events if row["event"] == "OWNED_LAUNCH_INTENT"))
        launch["data"]["contract"]["env"]["OMP_NUM_THREADS"] = "999"
        with pytest.raises(lifecycle.LifecycleRefused, match="request digest differs"):
            lifecycle.validate_lifecycle_handoff(harness.acquisitions[0], launch)
    finally:
        harness.close()


def test_real_tiny_worker_is_captured_gated_bounded_and_gone():
    harness = Harness()
    try:
        terminal = harness.engine.run_stage(harness.request(
            "import sys; sys.stdout.write('x' * 200000); sys.stderr.write('e' * 7)"))
        names = [row["event"] for row in harness.events]
        assert names.index("OWNED_LAUNCH_INTENT") < names.index("OWNED_CONTAINER_CREATED")
        assert names.index("OWNED_CHILD_CAPTURED") < names.index("OWNED_EXEC_RELEASE_INTENT")
        assert names.index("OWNED_EXEC_RELEASE_INTENT") < names.index("OWNED_EXEC_RELEASED")
        assert names.index("OWNED_TERMINAL") < names.index("WORKER_RESULT_ACCEPTED")
        assert terminal.accepted and terminal.return_code == 0
        result_row = next(row for row in harness.events
                          if row["event"] == "WORKER_RESULT_RETAINED")
        assert result_row["data"]["return_code"] == 0
        logs = list((harness.path / "runtime").glob("worker-*.stdout.log"))
        assert len(logs) == 1 and logs[0].stat().st_size == bootstrap.MAX_LOG_BYTES_PER_STREAM
        identities = _captured(harness.events)
        assert identities and all(not lifecycle.same_process(item) for item in identities)
        fence = harness.engine.trusted_result_fence(terminal)
        assert fence.current and fence.result_accepted
        assert harness.provider.released
    finally:
        harness.close()


def test_bootstrap_uses_trusted_absolute_origin_and_minimal_environment(monkeypatch):
    harness = Harness()
    marker = harness.path / "shadow-bootstrap-ran"
    shadow = harness.path / "autokernel" / "loop"
    shadow.mkdir(parents=True)
    (harness.path / "autokernel" / "__init__.py").write_text("", encoding="utf-8")
    (shadow / "__init__.py").write_text("", encoding="utf-8")
    (shadow / "worker_bootstrap.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('shadow')\n",
        encoding="utf-8")
    monkeypatch.setenv("SYNTHETIC_PARENT_PASSWORD", "fixture-must-not-cross")
    original_popen = lifecycle.subprocess.Popen
    captured = []

    def record_popen(*args, **kwargs):
        captured.append((args, kwargs))
        return original_popen(*args, **kwargs)

    monkeypatch.setattr(lifecycle.subprocess, "Popen", record_popen)
    try:
        assert harness.engine.run_stage(harness.request("pass")).accepted
        bootstrap_argv = captured[0][0][0]
        bootstrap_env = captured[0][1]["env"]
        assert bootstrap_argv[:4] == (sys.executable, "-I", "-S", "-B")
        assert Path(bootstrap_argv[4]).resolve() == Path(bootstrap.__file__).resolve()
        assert Path(bootstrap_argv[4]).is_absolute()
        assert "-m" not in bootstrap_argv
        assert bootstrap_env == {"PYTHONDONTWRITEBYTECODE": "1"}
        assert "SYNTHETIC_PARENT_PASSWORD" not in bootstrap_env
        assert not marker.exists()
    finally:
        harness.close()


def test_pipe_setup_failure_closes_every_already_owned_descriptor(monkeypatch):
    harness = Harness()
    before = len(os.listdir("/proc/self/fd"))
    original_pipe2 = lifecycle.os.pipe2
    calls = 0

    def fail_second_pipe(flags):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected second-pipe failure")
        return original_pipe2(flags)

    monkeypatch.setattr(lifecycle.os, "pipe2", fail_second_pipe)
    try:
        with pytest.raises(lifecycle.LifecycleRefused, match="second-pipe failure"):
            harness.engine.run_stage(harness.request("pass"))
        assert len(os.listdir("/proc/self/fd")) == before
    finally:
        harness.close()


def test_completed_predecessor_cannot_regain_current_result_authority():
    harness = Harness()
    try:
        predecessor = harness.engine.run_stage(harness.request("pass"))
        assert harness.engine.trusted_result_fence(predecessor).current
        successor = harness.engine.run_stage(replace(
            harness.request("pass"), request_id="request-successor",
            stage_id="stage-successor"))
        assert harness.engine.trusted_result_fence(successor).current
        assert not harness.engine.trusted_result_fence(predecessor).current
        assert predecessor.result_digest is not None
    finally:
        harness.close()


def test_transient_renewal_failure_before_gate_blocks_future_stage_and_cleans():
    failed = lifecycle.GrantReceipt(
        "fixture-grant", 1, time.monotonic() + 5.0,
        lifecycle.monotonic_clock_domain(), renewal_ok=False)
    harness = Harness(refresh=failed)
    try:
        with pytest.raises(lifecycle.LifecycleRefused, match="renewal_failed_future_admission"):
            harness.engine.run_stage(harness.request("raise SystemExit('must not execute')"))
        names = [row["event"] for row in harness.events]
        assert "OWNED_CHILD_CAPTURED" in names
        assert "OWNED_EXEC_RELEASED" not in names
        assert "OWNED_TERMINAL" in names
        assert all(not lifecycle.same_process(item) for item in _captured(harness.events))
    finally:
        harness.close()


def test_timeout_kills_only_owned_identities_and_not_foreign_sibling():
    foreign = subprocess.Popen(
        (sys.executable, "-B", "-c", "import time; time.sleep(5)"),
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    foreign_identity = lifecycle.process_identity(foreign.pid)
    harness = Harness()
    try:
        with pytest.raises(lifecycle.LifecycleRefused, match="deadline expired"):
            harness.engine.run_stage(harness.request(
                "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(5)",
                stage_seconds=0.1, teardown_seconds=0.5))
        assert lifecycle.same_process(foreign_identity)
        assert all(not lifecycle.same_process(item) for item in _captured(harness.events))
        owned_signals = harness.provider.authorization.container.signals
        assert owned_signals
        assert all(pid != foreign.pid for pid, _ticks, _signal in owned_signals)
        print(f"FOREIGN_TEST_PID pid={foreign_identity.pid} "
              f"start_ticks={foreign_identity.start_ticks} untouched_alive=true")
    finally:
        if lifecycle.same_process(foreign_identity):
            foreign.terminate()
        foreign.wait(timeout=2)
        print(f"FOREIGN_TEST_PID_CLEANUP pid={foreign_identity.pid} "
              f"start_ticks={foreign_identity.start_ticks} "
              f"alive={lifecycle.same_process(foreign_identity)}")


def test_teardown_journal_failure_still_kills_exact_owned_child_not_foreign():
    foreign = subprocess.Popen(
        (sys.executable, "-B", "-c", "import time; time.sleep(5)"),
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    foreign_identity = lifecycle.process_identity(foreign.pid)
    harness = Harness()

    def fail_teardown_start(row):
        if row.get("event") == "OWNED_TEARDOWN_STARTED":
            raise OSError("injected teardown journal failure")
        harness._record_event(row)

    harness.engine.event_sink = fail_teardown_start
    try:
        with pytest.raises(OSError, match="teardown journal failure"):
            harness.engine.run_stage(harness.request(
                "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(5)",
                stage_seconds=0.1, teardown_seconds=0.5))
        identities = _captured(harness.events)
        assert identities and all(not lifecycle.same_process(item) for item in identities)
        assert lifecycle.same_process(foreign_identity)
        names = [row["event"] for row in harness.events]
        assert "OWNED_TERMINAL" not in names
        assert "WORKER_RESULT_ACCEPTED" not in names
        assert harness.engine._terminals == {}
        print(f"FOREIGN_TEARDOWN_FAILURE_PID pid={foreign_identity.pid} "
              f"start_ticks={foreign_identity.start_ticks} untouched_alive=true")
    finally:
        if lifecycle.same_process(foreign_identity):
            foreign.terminate()
        foreign.wait(timeout=2)
        print(f"FOREIGN_TEARDOWN_FAILURE_CLEANUP pid={foreign_identity.pid} "
              f"start_ticks={foreign_identity.start_ticks} "
              f"alive={lifecycle.same_process(foreign_identity)}")
        harness.close()
        harness.close()


def test_stale_binding_retains_diagnostic_result_but_refuses_current_fence():
    harness = Harness()
    try:
        harness.engine.binding_fence = lambda _binding: (
            harness.provider.authorization is None
            or harness.provider.authorization.container.populated())
        terminal = harness.engine.run_stage(harness.request("pass"))
        assert not terminal.accepted and terminal.result_digest
        assert harness.events[-1]["event"] == "WORKER_RESULT_STALE"
        fence = harness.engine.trusted_result_fence(terminal)
        assert not fence.current and not fence.result_accepted
    finally:
        harness.close()


def test_cleanup_failure_retains_owned_state_and_reports_containment():
    harness = Harness(release_ok=False)
    try:
        with pytest.raises(lifecycle.ContainmentFailure, match="claim release"):
            harness.engine.run_stage(harness.request("pass"))
        names = [row["event"] for row in harness.events]
        assert "OWNED_TEARDOWN_FAILED" in names
        assert "OWNED_TERMINAL" not in names
        assert all(not lifecycle.same_process(item) for item in _captured(harness.events))
    finally:
        harness.close()


def test_failed_claim_release_can_reconcile_without_repeating_stage():
    harness = Harness(release_ok=False)
    try:
        with pytest.raises(lifecycle.ContainmentFailure):
            harness.engine.run_stage(harness.request("pass"))
        assert lifecycle.project_events(harness.events).active
        harness.provider.release_ok = True
        recovery = lifecycle.WorkerLifecycle(
            binding=harness.binding, runtime=harness.runtime,
            event_sink=lambda row: harness.events.append(dict(row)),
            provider=harness.provider,
            admission_fence=lambda *_args: lifecycle.StageAdmission(True, "recovery"),
            binding_fence=lambda _binding: True,
            runtime_fence=lambda _request, _now_: lifecycle.RuntimeDirective(
                "continue", "recovery"), wall_clock=_now)
        terminal = recovery.reconcile(harness.events)
        assert terminal is not None and not terminal.accepted
        assert not lifecycle.project_events(harness.events).active
    finally:
        harness.close()


def test_planned_serving_adapter_explicitly_refuses_unproven_placement():
    harness = Harness()
    try:
        with pytest.raises(Exception, match="placement is not proven"):
            harness.engine.planned_serving_guard()
    finally:
        harness.close()


def test_renewal_outage_does_not_kill_already_held_stage():
    harness = Harness()
    calls = 0

    def refresh(authorization, deadline):
        nonlocal calls
        del deadline
        calls += 1
        return replace(authorization.grant, renewal_ok=calls == 1)

    harness.provider.refresh = refresh
    try:
        terminal = harness.engine.run_stage(harness.request(
            "import time; time.sleep(0.08)"))
        assert terminal.accepted and calls >= 2
        assert all(not lifecycle.same_process(item) for item in _captured(harness.events))
    finally:
        harness.close()


def test_live_revocation_ends_stage_with_owned_cleanup():
    harness = Harness()
    calls = 0

    def refresh(authorization, deadline):
        nonlocal calls
        del deadline
        calls += 1
        return replace(authorization.grant, revoked=calls > 1)

    harness.provider.refresh = refresh
    try:
        with pytest.raises(lifecycle.LifecycleRefused, match="revoked"):
            harness.engine.run_stage(harness.request("import time; time.sleep(5)"))
        assert harness.events[-1]["event"] == "WORKER_RESULT_STALE"
        assert all(not lifecycle.same_process(item) for item in _captured(harness.events))
    finally:
        harness.close()


def test_unadmitted_grant_is_released_without_spawn():
    harness = Harness()
    harness.engine.admission_fence = lambda *_args: lifecycle.StageAdmission(
        False, "control fence closed")
    try:
        with pytest.raises(lifecycle.WaitingAuthority, match="control fence"):
            harness.engine.run_stage(harness.request("pass"))
        assert harness.events == []
        assert harness.provider.released
    finally:
        harness.close()


def test_wrong_clock_grant_is_released_without_spawn():
    harness = Harness()
    original = harness.provider.authorize

    def authorize(request, container_id, deadline):
        value = original(request, container_id, deadline)
        value = replace(value, grant=replace(value.grant, clock_domain="foreign-clock"))
        harness.provider.authorization = value
        return value

    harness.provider.authorize = authorize
    try:
        with pytest.raises(lifecycle.WaitingAuthority, match="clock domain"):
            harness.engine.run_stage(harness.request("pass"))
        assert harness.events == [] and harness.provider.released
    finally:
        harness.close()


def test_replaced_container_identity_is_not_signalled_or_removed():
    harness = Harness()

    def replace_container(authorization, deadline):
        del deadline
        path = authorization.container.path
        path.rename(path.with_name(path.name + "-original"))
        path.mkdir(mode=0o700)
        return authorization.grant

    harness.provider.refresh = replace_container
    identities = []
    try:
        with pytest.raises(lifecycle.ContainmentFailure):
            harness.engine.run_stage(harness.request("import time; time.sleep(5)"))
        identities = _captured(harness.events)
        assert identities
        assert harness.provider.authorization.container.signals == []
        assert harness.provider.authorization.container.path.exists()
    finally:
        for identity in identities:
            if lifecycle.same_process(identity):
                os.kill(identity.pid, signal.SIGKILL)
                try:
                    os.waitpid(identity.pid, 0)
                except ChildProcessError:
                    pass
            print(f"OWNED_TEST_PID_CLEANUP pid={identity.pid} "
                  f"start_ticks={identity.start_ticks} alive={lifecycle.same_process(identity)}")
        container = harness.provider.authorization.container
        container._owned.clear()
        if container.path.exists():
            container.path.rmdir()
        original = container.path.with_name(container.path.name + "-original")
        if original.exists():
            original.rmdir()
        harness.close()


@pytest.mark.parametrize("phase", [
    "OWNED_LAUNCH_INTENT", "OWNED_CONTAINER_CREATED", "OWNED_CHILD_CAPTURED",
    "OWNED_EXEC_RELEASE_INTENT", "OWNED_EXEC_RELEASED",
])
def test_crash_windows_reconcile_only_exact_preassigned_container(phase):
    harness = Harness()
    fired = False

    def crash(current):
        nonlocal fired
        if current == phase and not fired:
            fired = True
            raise lifecycle.SimulatedCrash(current)

    harness.engine.fault_hook = crash
    try:
        with pytest.raises(lifecycle.SimulatedCrash):
            harness.engine.run_stage(harness.request("import time; time.sleep(5)"))
        recovery = lifecycle.WorkerLifecycle(
            binding=harness.binding, runtime=harness.runtime,
            event_sink=lambda row: harness.events.append(dict(row)),
            provider=harness.provider,
            admission_fence=lambda _request, _grant, _now_, _until: lifecycle.StageAdmission(
                True, "admitted"),
            binding_fence=lambda _binding: True,
            runtime_fence=lambda _request, _now_: lifecycle.RuntimeDirective(
                "continue", "recovery"), wall_clock=_now)
        if phase == "OWNED_CONTAINER_CREATED":
            with pytest.raises(lifecycle.ContainmentFailure, match="before PID receipt"):
                recovery.reconcile(harness.events)
            assert harness.events[-1]["event"] == "WORKER_UNRESOLVED"
        else:
            terminal = recovery.reconcile(harness.events)
            assert terminal is not None and not terminal.accepted
            assert harness.events[-1]["event"] == "WORKER_RESULT_STALE"
            assert all(not lifecycle.same_process(item) for item in _captured(harness.events))
            assert not harness.provider.authorization.container.path.exists()
    finally:
        container = (harness.provider.authorization.container
                     if harness.provider and harness.provider.authorization else None)
        if container is not None:
            container.kill()
            container.wait_empty(0.2)
            if container.path.exists() and not container.populated():
                container.close_and_remove()
        harness.close()


def test_crash_immediately_after_popen_before_attach_stays_unresolved():
    harness = Harness()
    harness.engine.fault_hook = lambda phase: (
        (_ for _ in ()).throw(lifecycle.SimulatedCrash(phase))
        if phase == "AFTER_POPEN_BEFORE_ATTACH" else None)
    identity = None
    try:
        with pytest.raises(lifecycle.SimulatedCrash):
            harness.engine.run_stage(harness.request("import time; time.sleep(5)"))
        identity = harness.engine._unattached_identity
        assert identity is not None
        recovery = lifecycle.WorkerLifecycle(
            binding=harness.binding, runtime=harness.runtime,
            event_sink=lambda row: harness.events.append(dict(row)),
            provider=harness.provider,
            admission_fence=lambda *_args: lifecycle.StageAdmission(True, "recovery"),
            binding_fence=lambda _binding: True,
            runtime_fence=lambda _request, _now_: lifecycle.RuntimeDirective(
                "continue", "recovery"), wall_clock=_now)
        with pytest.raises(lifecycle.ContainmentFailure, match="before PID receipt"):
            recovery.reconcile(harness.events)
        assert harness.events[-1]["event"] == "WORKER_UNRESOLVED"
        # The test owns this exact identity; cleanup never scans names or PIDs.
        deadline = time.monotonic() + 0.5
        while lifecycle.same_process(identity) and time.monotonic() < deadline:
            try:
                waited, _status = os.waitpid(identity.pid, os.WNOHANG)
            except ChildProcessError:
                break
            if waited == 0:
                time.sleep(0.005)
        if lifecycle.same_process(identity):
            os.kill(identity.pid, signal.SIGKILL)
            try:
                os.waitpid(identity.pid, 0)
            except ChildProcessError:
                pass
        assert not lifecycle.same_process(identity)
        print(f"OWNED_UNATTACHED_TEST_PID pid={identity.pid} "
              f"start_ticks={identity.start_ticks} alive=false")
    finally:
        if identity is not None and lifecycle.same_process(identity):
            os.kill(identity.pid, signal.SIGKILL)
            try:
                os.waitpid(identity.pid, 0)
            except ChildProcessError:
                pass
        container = harness.provider.authorization.container
        container._owned.clear()
        if container.path.exists():
            container.path.rmdir()
        harness.close()
