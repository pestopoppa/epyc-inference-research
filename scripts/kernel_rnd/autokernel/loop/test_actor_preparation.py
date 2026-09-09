"""Hermetic tests for the lifecycle-authorized actor preparation consumer."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import time

import pytest

from . import actor_preparation as preparation
from . import campaign
from . import worker_lifecycle
from .test_campaign import _manifest, _registry, _target


@dataclass(frozen=True)
class _ForwardActorPreparationRequest:
    row: dict

    def to_dict(self):
        return json.loads(json.dumps(self.row))


def _resolved(*, fallback=True):
    raw = _manifest(production=[_target("prod")])
    if not fallback:
        raw["fallbacks"]["planner"] = []
    return campaign.resolve_manifest(campaign.CampaignManifest.from_dict(raw),
                                     registry_snapshot=_registry())


def _fake_executable(tmp_path: Path) -> Path:
    path = tmp_path / "fake-actor"
    path.write_text("""#!/usr/bin/env python3
import json, os, subprocess, sys
prompt = sys.argv[-1]
mode = os.environ.get("FAKE_MODE", "ok")
if mode == "fail_primary" and "gpt-5.6-sol" in sys.argv:
    raise SystemExit(7)
if mode == "child":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                             start_new_session=True)
    open(os.environ["CHILD_PID"], "w").write(str(child.pid))
if "Review this proposed preparation" in prompt:
    print(json.dumps({"accepted": mode != "reject", "reason": "reviewed"}))
elif mode == "invalid":
    print(json.dumps({"compiled": True}))
elif "build_system" in prompt:
    print(json.dumps({"build_system": "cmake", "configured_options": ["-DX=1"],
                      "artifact_expectations": "unverified binary"}))
else:
    print(json.dumps({"mechanism": "tile loop", "target_surface": "src/x.cpp",
                      "target_symbol": "kernel_x", "implementation_plan": "edit one loop"}))
""")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _profile(profile_id, role, binary, *, model=None):
    return preparation.ActorProfile.from_dict({
        "schema": preparation.PROFILE_SCHEMA, "profile_id": profile_id, "role": role,
        "provider": "fake-provider", "model": model or profile_id, "effort": "low",
        "backend_kind": "codex", "binary": str(binary), "binary_sha256": _sha(binary),
    })


def _profiles(binary):
    return {item.profile_id: item for item in (
        _profile("gpt-5.6-sol", "planner", binary),
        _profile("gpt-5.5", "planner", binary),
        _profile("fable-5.1", "critic", binary),
    )}


def _request(profile, *, kind="source", marker="one"):
    row = {"schema": preparation.REQUEST_SCHEMA, "actor_kind": kind,
           "actor_identity": profile.to_dict(),
           "prompt": f"fresh target profile/model/quant/evidence {marker}",
           "mandatory_conflicts": [{"finding": "do not compose"}],
           "proposal": {"proposal_id": "proposal-1", "target_revision_digest": "a" * 64}}
    row["cache_key"] = preparation._digest({
        key: row[key] for key in ("actor_kind", "actor_identity", "prompt",
                                  "mandatory_conflicts", "proposal")})
    return _ForwardActorPreparationRequest(row)


def _budgets():
    return preparation.ActorBudgets(4, 2, 60, 2, 1, 20)


class FakeOwnedLifecycle:
    """Test-only provider; production must adapt the accepted WorkerLifecycle path."""

    def __init__(self, *, stage="stage-1", deny_models=(), profile_ready=True,
                 mode="ok", child_pid_path=None, outcome_status=None):
        self.stage = stage
        self.deny_models = set(deny_models)
        self.profile_ready = profile_ready
        self.mode = mode
        self.child_pid_path = child_pid_path
        self.outcome_status = outcome_status
        self.events = []
        self.admission_times = []

    def reserve(self, *, request, stage_plan_digest, actor_profile, budgets, now,
                clock_domain):
        self.events.append(("INTENT", actor_profile["model"], dict(budgets)))
        self.admission_times.append(now)
        failure = None
        if stage_plan_digest != self.stage:
            failure = "selected_stage_mismatch"
        elif not self.profile_ready:
            failure = "target_profile_pending"
        elif actor_profile["model"] in self.deny_models:
            failure = "provider_cooldown"
        elif _sha(Path(actor_profile["binary"])) != actor_profile["binary_sha256"]:
            failure = "actor_executable_changed"
        if failure:
            return {"schema": preparation.AVAILABILITY_SCHEMA,
                    "actor_profile_digest": preparation._digest(actor_profile),
                    "failure_class": failure, "consecutive_failures": 1,
                    "last_success": None, "retry_after": now + 30,
                    "reset_at": None, "clock_domain": clock_domain,
                    "next_eligible_at": now + 30}
        return {"schema": preparation.RESERVATION_SCHEMA,
                "reservation_id": f"reservation-{len(self.events)}",
                "request_digest": preparation._digest(request),
                "stage_plan_digest": stage_plan_digest,
                "transition_id": "controller-transition-1",
                "target_profile_digest": "e" * 64,
                "target_profile_receipt_digest": "f" * 64,
                "actor_profile_digest": preparation._digest(actor_profile), "deadline": 10.0,
                "clock_domain": clock_domain, "control_revision": 1}

    def invoke(self, reservation, backend, prompt):
        self.events.append(("INVOKE", backend.describe()))
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "FAKE_MODE": self.mode}
        if self.child_pid_path:
            env["CHILD_PID"] = str(self.child_pid_path)
        done = subprocess.Popen(backend.argv(prompt, Path("/tmp")), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, env=env)
        done.wait(timeout=3)
        clean = True
        if self.child_pid_path and self.child_pid_path.exists():
            child = int(self.child_pid_path.read_text())
            try:
                os.kill(child, signal.SIGTERM)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                try:
                    os.kill(child, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.01)
            else:
                os.kill(child, signal.SIGKILL)
            try:
                os.kill(child, 0)
            except ProcessLookupError:
                clean = True
            else:
                # A killed child can remain a zombie owned by init briefly; the fake
                # authority has still terminated it, but does not claim reaping here.
                clean = Path(f"/proc/{child}/stat").read_text().split()[2] == "Z"
        stdout, _stderr = done.communicate(timeout=1)
        status = self.outcome_status or ("completed" if done.returncode == 0 else "failed")
        return {"schema": preparation.OUTCOME_SCHEMA,
                "reservation_id": reservation.reservation_id,
                "status": status, "stdout": stdout,
                "failure_class": None if status == "completed" else status,
                "charged_seconds": 0.25, "resource_enforced": True,
                "descendants_clean": clean}

    def finish(self, reservation, outcome, disposition):
        self.events.append(("FINISH", reservation.reservation_id,
                            outcome.charged_seconds, disposition))


def _consumer(resolved, profiles, capability):
    return preparation.ActorPreparationConsumer(
        resolved_campaign=resolved, profiles=profiles, budgets=_budgets(),
        capability=capability, clock=lambda: 1.0, clock_domain="test-monotonic",
        max_output_bytes=4096)


def test_default_missing_provider_refuses_before_any_backend_call(tmp_path):
    profiles = _profiles(_fake_executable(tmp_path))
    request = _request(profiles["gpt-5.6-sol"])
    with pytest.raises(preparation.PreparationRefused, match="capability is unavailable"):
        _consumer(_resolved(), profiles, None).prepare(request, stage_plan_digest="stage-1")


def test_real_backend_mapping_owned_invocation_review_and_preintent(tmp_path):
    binary = _fake_executable(tmp_path)
    profiles = _profiles(binary)
    authority = FakeOwnedLifecycle()
    result = _consumer(_resolved(), profiles, authority).prepare(
        _request(profiles["gpt-5.6-sol"]), stage_plan_digest="stage-1")
    assert result.status == "proposed"
    assert result.proposed_output["target_symbol"] == "kernel_x"
    assert [event[0] for event in authority.events] == [
        "INTENT", "INVOKE", "FINISH", "INTENT", "INVOKE", "FINISH"]
    assert authority.events[1][1] == "codex:gpt-5.6-sol@low"
    assert authority.events[3][1] == "fable-5.1"
    assert authority.events[0][2] == _budgets().to_dict()


def test_strings_cannot_manufacture_selection_or_profile_prerequisite(tmp_path):
    profiles = _profiles(_fake_executable(tmp_path))
    request = _request(profiles["gpt-5.6-sol"])
    wrong_stage = FakeOwnedLifecycle(stage="different")
    assert _consumer(_resolved(), profiles, wrong_stage).prepare(
        request, stage_plan_digest="freely-constructed").status == "cooldown"
    missing_profile = FakeOwnedLifecycle(profile_ready=False)
    assert _consumer(_resolved(), profiles, missing_profile).prepare(
        request, stage_plan_digest="stage-1").status == "cooldown"
    assert not any(event[0] == "INVOKE" for event in (*wrong_stage.events,
                                                       *missing_profile.events))


def test_runtime_never_calls_authority(tmp_path):
    profiles = _profiles(_fake_executable(tmp_path))
    authority = FakeOwnedLifecycle()
    request = _request(profiles["gpt-5.6-sol"], kind="runtime_recipe")
    with pytest.raises(preparation.PreparationRefused, match="runtime work"):
        _consumer(_resolved(), profiles, authority).prepare(request, stage_plan_digest="stage-1")
    assert authority.events == []


def test_only_configured_fallback_follows_primary_denial(tmp_path):
    profiles = _profiles(_fake_executable(tmp_path))
    authority = FakeOwnedLifecycle(deny_models={"gpt-5.6-sol"})
    result = _consumer(_resolved(), profiles, authority).prepare(
        _request(profiles["gpt-5.6-sol"]), stage_plan_digest="stage-1")
    assert result.status == "proposed"
    assert [event[1] for event in authority.events if event[0] == "INTENT"] == [
        "gpt-5.6-sol", "gpt-5.5", "fable-5.1"]


@pytest.mark.parametrize("mode,status,failure", [
    ("invalid", "cooldown", "planner_unavailable"),
    ("reject", "refused", "review_rejected"),
])
def test_invalid_source_and_critic_rejection_remain_non_authoritative(
        tmp_path, mode, status, failure):
    profiles = _profiles(_fake_executable(tmp_path))
    authority = FakeOwnedLifecycle(mode=mode)
    result = _consumer(_resolved(fallback=False), profiles, authority).prepare(
        _request(profiles["gpt-5.6-sol"]), stage_plan_digest="stage-1")
    assert result.status == status
    assert result.failure_class == failure
    assert result.proposed_output is None


def test_owned_provider_proves_detached_descendant_cleanup(tmp_path):
    profiles = _profiles(_fake_executable(tmp_path))
    child_path = tmp_path / "child.pid"
    authority = FakeOwnedLifecycle(mode="child", child_pid_path=child_path)
    result = _consumer(_resolved(), profiles, authority).prepare(
        _request(profiles["gpt-5.6-sol"]), stage_plan_digest="stage-1")
    assert result.status == "proposed"
    child = int(child_path.read_text())
    assert not Path(f"/proc/{child}").exists() \
        or Path(f"/proc/{child}/stat").read_text().split()[2] == "Z"


def test_unproven_containment_refuses_result(tmp_path):
    profiles = _profiles(_fake_executable(tmp_path))

    class LyingProvider(FakeOwnedLifecycle):
        def invoke(self, reservation, backend, prompt):
            row = super().invoke(reservation, backend, prompt)
            row["descendants_clean"] = False
            return row

    authority = LyingProvider()
    with pytest.raises(preparation.PreparationRefused, match="did not prove"):
        _consumer(_resolved(), profiles, authority).prepare(
            _request(profiles["gpt-5.6-sol"]), stage_plan_digest="stage-1")
    assert authority.events[-1][-1] == "containment_refused"


@pytest.mark.parametrize("outcome_status", ["failed", "deadline", "output_limit"])
def test_lifecycle_faults_are_charged_before_fallback(tmp_path, outcome_status):
    profiles = _profiles(_fake_executable(tmp_path))
    authority = FakeOwnedLifecycle(outcome_status=outcome_status)
    result = _consumer(_resolved(fallback=False), profiles, authority).prepare(
        _request(profiles["gpt-5.6-sol"]), stage_plan_digest="stage-1")
    assert result.status == "cooldown"
    expected = outcome_status if outcome_status in {"deadline", "output_limit"} else "actor_failed"
    assert authority.events[-1][-1] == expected
    assert authority.events[-1][2] == 0.25


def test_changed_profile_cannot_reuse_request_identity(tmp_path):
    binary = _fake_executable(tmp_path)
    profiles = _profiles(binary)
    request = _request(profiles["gpt-5.6-sol"])
    profiles["gpt-5.6-sol"] = _profile(
        "gpt-5.6-sol", "planner", binary, model="different")
    with pytest.raises(preparation.PreparationRefused, match="differs from configured"):
        _consumer(_resolved(), profiles, FakeOwnedLifecycle()).prepare(
            request, stage_plan_digest="stage-1")


def test_provider_refuses_changed_executable_before_invoke(tmp_path):
    binary = _fake_executable(tmp_path)
    profiles = _profiles(binary)
    request = _request(profiles["gpt-5.6-sol"])
    binary.write_text(binary.read_text() + "\n# changed\n")
    authority = FakeOwnedLifecycle()
    result = _consumer(_resolved(), profiles, authority).prepare(
        request, stage_plan_digest="stage-1")
    assert result.status == "cooldown"
    assert result.failure_class == "actor_executable_changed"
    assert not any(event[0] == "INVOKE" for event in authority.events)


@pytest.mark.parametrize("failure_kind", ["malformed", "exception"])
def test_invocation_decode_or_callback_exception_is_settled(tmp_path, failure_kind):
    profiles = _profiles(_fake_executable(tmp_path))

    class BrokenOutcome(FakeOwnedLifecycle):
        def invoke(self, reservation, backend, prompt):
            if failure_kind == "exception":
                raise RuntimeError("secret details must not escape")
            row = super().invoke(reservation, backend, prompt)
            row["failure_class"] = "exit"
            return row

    authority = BrokenOutcome()
    expected = "malformed_outcome" if failure_kind == "malformed" else "invocation_exception"
    with pytest.raises(preparation.PreparationRefused, match=expected):
        _consumer(_resolved(), profiles, authority).prepare(
            _request(profiles["gpt-5.6-sol"]), stage_plan_digest="stage-1")
    assert authority.events[-1][-1] == expected
    assert "secret details" not in str(authority.events[-1])


def test_lifecycle_crash_boundary_is_not_falsely_finished_or_reinvoked(tmp_path):
    profiles = _profiles(_fake_executable(tmp_path))

    class CrashingLifecycle(FakeOwnedLifecycle):
        def invoke(self, reservation, backend, prompt):
            self.events.append(("CRASH", reservation.reservation_id))
            raise worker_lifecycle.SimulatedCrash("OWNED_LAUNCH_INTENT")

    authority = CrashingLifecycle()
    with pytest.raises(worker_lifecycle.SimulatedCrash):
        _consumer(_resolved(), profiles, authority).prepare(
            _request(profiles["gpt-5.6-sol"]), stage_plan_digest="stage-1")
    assert [event[0] for event in authority.events] == ["INTENT", "CRASH"]


def test_finish_failure_is_uncertain_and_forbids_reinvocation(tmp_path):
    profiles = _profiles(_fake_executable(tmp_path))

    class UncertainFinish(FakeOwnedLifecycle):
        def finish(self, reservation, outcome, disposition):
            self.events.append(("FINISH_ATTEMPT", reservation.reservation_id, disposition))
            raise OSError("persistence details must not escape")

    authority = UncertainFinish()
    with pytest.raises(preparation.PreparationSettlementUncertain,
                       match="reinvocation is forbidden") as caught:
        _consumer(_resolved(), profiles, authority).prepare(
            _request(profiles["gpt-5.6-sol"]), stage_plan_digest="stage-1")
    assert [event[0] for event in authority.events] == [
        "INTENT", "INVOKE", "FINISH_ATTEMPT"]
    assert "persistence details" not in str(caught.value)


def test_clock_refreshes_for_each_fallback_and_critic_admission(tmp_path):
    profiles = _profiles(_fake_executable(tmp_path))
    authority = FakeOwnedLifecycle(deny_models={"gpt-5.6-sol"})
    ticks = iter((1.0, 2.0, 3.0))
    consumer = preparation.ActorPreparationConsumer(
        resolved_campaign=_resolved(), profiles=profiles, budgets=_budgets(),
        capability=authority, clock=lambda: next(ticks), clock_domain="test-monotonic",
        max_output_bytes=4096)
    assert consumer.prepare(_request(profiles["gpt-5.6-sol"]),
                            stage_plan_digest="stage-1").status == "proposed"
    assert authority.admission_times == [1.0, 2.0, 3.0]


def test_completed_huge_stdout_is_terminalized_as_output_limit(tmp_path):
    profiles = _profiles(_fake_executable(tmp_path))

    class HugeProvider(FakeOwnedLifecycle):
        def invoke(self, reservation, backend, prompt):
            row = super().invoke(reservation, backend, prompt)
            row["stdout"] = "x" * 5000
            return row

    authority = HugeProvider()
    result = _consumer(_resolved(fallback=False), profiles, authority).prepare(
        _request(profiles["gpt-5.6-sol"]), stage_plan_digest="stage-1")
    assert result.status == "cooldown"
    assert authority.events[-1][-1] == "output_limit"


@pytest.mark.parametrize("status,failure", [
    ("completed", "exit"), ("failed", None), ("deadline", "exit"),
    ("output_limit", "deadline"),
])
def test_outcome_status_and_failure_class_must_agree(status, failure):
    row = {"schema": preparation.OUTCOME_SCHEMA, "reservation_id": "reservation",
           "status": status, "stdout": "", "failure_class": failure,
           "charged_seconds": 1.0, "resource_enforced": True,
           "descendants_clean": True}
    with pytest.raises(preparation.PreparationRefused, match="failure_class"):
        preparation.StageOutcome.from_dict(row)


@pytest.mark.parametrize("mutation", ["expired", "wrong_clock"])
def test_reservation_deadline_uses_named_current_clock(tmp_path, mutation):
    profiles = _profiles(_fake_executable(tmp_path))

    class BadReservation(FakeOwnedLifecycle):
        def reserve(self, **kwargs):
            row = super().reserve(**kwargs)
            row["deadline"] = 1.0 if mutation == "expired" else row["deadline"]
            row["clock_domain"] = "wall" if mutation == "wrong_clock" else row["clock_domain"]
            return row

    with pytest.raises(preparation.PreparationRefused, match="reservation differs"):
        _consumer(_resolved(), profiles, BadReservation()).prepare(
            _request(profiles["gpt-5.6-sol"]), stage_plan_digest="stage-1")


@pytest.mark.parametrize("mutation", ["zero_streak", "past", "wrong_clock", "mismatch"])
def test_stale_or_non_cooldown_availability_cannot_trigger_fallback(tmp_path, mutation):
    profiles = _profiles(_fake_executable(tmp_path))

    class BadAvailability(FakeOwnedLifecycle):
        def reserve(self, **kwargs):
            self.deny_models.add(kwargs["actor_profile"]["model"])
            row = super().reserve(**kwargs)
            if mutation == "zero_streak":
                row["consecutive_failures"] = 0
            elif mutation == "past":
                row["retry_after"] = row["next_eligible_at"] = kwargs["now"]
            elif mutation == "wrong_clock":
                row["clock_domain"] = "wall"
            else:
                row["next_eligible_at"] += 1
            return row

    authority = BadAvailability()
    with pytest.raises(preparation.PreparationRefused, match="availability"):
        _consumer(_resolved(), profiles, authority).prepare(
            _request(profiles["gpt-5.6-sol"]), stage_plan_digest="stage-1")
    assert len(authority.events) == 1
