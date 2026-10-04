"""G1 executor: lease, 60-min cap, drain-or-refuse, serving proof, watchdog, AP3 gate.

Offline: every stack action goes through a fake ``StackOps``; window, lease,
schedule, marker and device-lock paths all live under ``tmp_path``.
"""

from __future__ import annotations

import fcntl
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.runtime import gpu_window as gw
from src.runtime import gpu_window_executor as gwe

PORT = 8083
ROLES = ["architect_critic", "coder_escalation"]


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


class FakeStack:
    """A one-port llama-server, scriptable per test."""

    def __init__(self, clock: list[float]) -> None:
        self.clock = clock
        self.up = True
        self.busy = 0
        self.busy_until: float | None = None
        self.slots_ok = True
        self.completion_text = " Paris."
        self.libdir = "/k/production/gpu/bin"
        self.calls: list[tuple[str, str]] = []
        self.device_held = False
        self.reload_brings_up = True

    def ops(self) -> gwe.StackOps:
        return gwe.StackOps(
            stop=self.stop, reload=self.reload, get_json=self.get_json,
            post_json=self.post_json, pids_on_port=self.pids, proc_maps=self.maps,
            proc_exe=lambda pid: f"{self.libdir}/llama-server",
            device_held=lambda dev: self.device_held,
            now=lambda: self.clock[0], sleep=self.sleep)

    def sleep(self, s: float) -> None:
        self.clock[0] += s

    def stop(self, comp: str) -> bool:
        self.calls.append(("stop", comp))
        self.up = False
        return True

    def reload(self, comp: str) -> bool:
        self.calls.append(("reload", comp))
        self.up = self.reload_brings_up
        return True

    def get_json(self, url: str, timeout: float):
        if not self.up:
            return 0, None
        if url.endswith("/slots"):
            if not self.slots_ok:
                return 501, None
            busy = self.busy if (self.busy_until is None or self.clock[0] < self.busy_until) else 0
            return 200, [{"id": i, "is_processing": i < busy} for i in range(4)]
        if url.endswith("/health"):
            return 200, {"status": "ok"}
        if url.endswith("/v1/models"):
            return 200, {"data": [{"id": "qwen"}]}
        return 404, None

    def post_json(self, url: str, body: dict, timeout: float):
        self.calls.append(("completion", url))
        if not self.up:
            return 0, None
        return 200, {"choices": [{"text": self.completion_text}]}

    def pids(self, port: int):
        return [4242] if self.up else []

    def maps(self, pid: int):
        return [f"7f00-7f10 r-xp 0 0 0 {self.libdir}/libggml-base.so",
                f"7f10-7f20 r-xp 0 0 0 {self.libdir}/libggml-hip.so"]


@pytest.fixture
def env(tmp_path, monkeypatch):
    window = tmp_path / "gpu-window" / "mi210.json"
    window.parent.mkdir()
    monkeypatch.setenv(gw.PATH_ENV, str(window))
    monkeypatch.setenv("ORCHESTRATOR_TMP_DIR", str(tmp_path / "locks"))
    monkeypatch.setenv(gwe.PENDING_ENV, str(tmp_path / "stack_change_pending.json"))
    monkeypatch.setenv(gwe.TOKENS_ENV, str(tmp_path / "tokens"))
    clock = [1_900_000_000.0]
    fake = FakeStack(clock)
    schedule = window.with_name("schedule.json")
    schedule.write_text(json.dumps({
        "schema": gwe.SCHEDULE_SCHEMA,
        "entries": [{"id": "s1", "device_id": "mi210_0", "holder": "autokernel",
                     "campaign_id": "ak", "purpose": "gpu batch",
                     "start": _iso(clock[0] - 60), "end": _iso(clock[0] + 3500)}]}))
    ex = gwe.Executor(window, fake.ops())
    auth = gwe.Authority("stack_owner", "workspace-ec", "grant-1")
    return SimpleNamespace(window=window, fake=fake, clock=clock, ex=ex, auth=auth,
                           schedule=schedule, tmp=tmp_path)


def _open(env, minutes: int = 45, **kw):
    args = dict(roles=ROLES, ports=[PORT], components=["architect_critic"],
                expected_end=_iso(env.clock[0] + minutes * 60), authority=env.auth,
                schedule_ref="s1", campaign_id="ak")
    args.update(kw)
    return env.ex.open(**args)


def _window(env) -> dict:
    return json.loads(env.window.read_text())


# ── open ─────────────────────────────────────────────────────────────────────


def test_open_drains_stops_and_grants(env):
    env.fake.busy, env.fake.busy_until = 2, env.clock[0] + 30
    lease = _open(env)
    w = _window(env)
    assert w["holder"] == "autokernel" and w["grant_state"] == "granted"
    assert w["written_by"] == gwe.WRITER and w["window_id"] == lease["window_id"]
    assert w["device_id"] == "mi210_0" and w["parked_ports"] == [PORT]
    assert lease["state"] == "open" and ("stop", "architect_critic") in env.fake.calls
    assert gw.is_parked("architect_critic")


def test_open_caps_window_at_60_min(env):
    with pytest.raises(gwe.WindowRefused) as exc:
        _open(env, minutes=90)
    assert exc.value.reason == "window_too_long"
    assert not env.window.exists() and env.fake.calls == []


def test_one_window_at_a_time(env):
    _open(env)
    with pytest.raises(gwe.WindowRefused) as exc:
        _open(env)
    assert exc.value.reason == "window_held"
    with pytest.raises(gw.WindowHeld):
        gw.park(roles=ROLES, ports=[PORT], path=env.window)


def test_lease_lock_refuses_a_concurrent_transition(env):
    with gwe.LeaseLock(env.window):
        with pytest.raises(gwe.WindowRefused) as exc:
            _open(env)
    assert exc.value.reason == "executor_busy"


def test_drain_timeout_refuses_and_never_stops(env):
    env.fake.busy = 1  # never finishes
    with pytest.raises(gwe.WindowRefused) as exc:
        _open(env)
    assert exc.value.reason == "drain_timeout"
    assert not any(c[0] == "stop" for c in env.fake.calls)
    assert _window(env)["holder"] == "production"
    assert json.loads(gwe.lease_path(env.window).read_text())["state"] == "closed"


def test_drain_without_slots_endpoint_refuses(env):
    env.fake.slots_ok = False
    with pytest.raises(gwe.WindowRefused) as exc:
        _open(env)
    assert exc.value.reason == "drain_unverifiable"
    assert _window(env)["holder"] == "production"


def test_pending_stack_change_refuses(env):
    gwe.set_pending("chg-1", "bring-up", by="test")
    with pytest.raises(gwe.WindowRefused) as exc:
        _open(env)
    assert exc.value.reason == "stack_change_pending"
    gwe.pending_marker_path().write_text("garbled{")
    with pytest.raises(gwe.WindowRefused):  # unreadable marker = still pending
        _open(env)
    gwe.clear_pending()
    _open(env)


def test_unscheduled_window_refuses(env):
    with pytest.raises(gwe.WindowRefused) as exc:
        _open(env, schedule_ref="nope")
    assert exc.value.reason == "schedule_unscheduled"


def test_device_busy_refuses(env):
    env.fake.device_held = True
    with pytest.raises(gwe.WindowRefused) as exc:
        _open(env)
    assert exc.value.reason == "device_busy"


# ── authority ────────────────────────────────────────────────────────────────


def test_authority_needs_owner_with_grant_or_operator_token(env, monkeypatch):
    with pytest.raises(gwe.WindowRefused):
        gwe.authorize(stack_owner_session=None, operator_token=None, compute_grant="g")
    with pytest.raises(gwe.WindowRefused):  # a peer session name is not the owner
        gwe.authorize(stack_owner_session="mainB", operator_token=None, compute_grant="g")
    with pytest.raises(gwe.WindowRefused):  # owner, but not routed via a compute-grant
        gwe.authorize(stack_owner_session="workspace-ec", operator_token=None,
                      compute_grant=None)
    assert gwe.authorize(stack_owner_session="workspace-ec", operator_token=None,
                         compute_grant="g").kind == "stack_owner"
    with pytest.raises(gwe.WindowRefused):
        gwe.authorize(stack_owner_session=None, operator_token="s3cret", compute_grant=None)
    import hashlib

    (env.tmp / "tokens").write_text(hashlib.sha256(b"s3cret").hexdigest() + "\n")
    assert gwe.authorize(stack_owner_session=None, operator_token="s3cret",
                         compute_grant=None).kind == "operator_token"


# ── close / serving proof ────────────────────────────────────────────────────


def test_close_reloads_proves_then_writes_production(env):
    _open(env)
    out = env.ex.close(reason="manual")
    assert out["outcome"] == "restored"
    w = _window(env)
    assert w["holder"] == "production" and w["written_by"] == gwe.WRITER
    assert ("reload", "architect_critic") in env.fake.calls
    assert not gw.is_parked("architect_critic")


@pytest.mark.parametrize("defect", ["empty_completion", "foreign_ggml", "no_hip", "down"])
def test_failed_serving_proof_never_writes_production(env, defect):
    _open(env)
    if defect == "empty_completion":
        env.fake.completion_text = "  "
    elif defect == "down":
        env.fake.reload_brings_up = False
    fake_maps = env.fake.maps
    if defect == "foreign_ggml":
        env.fake.maps = lambda pid: fake_maps(pid) + [
            "7f20-7f30 r-xp 0 0 0 /mnt/raid0/llm/llama.cpp/build/bin/libggml-cpu.so"]
    elif defect == "no_hip":
        env.fake.maps = lambda pid: [fake_maps(pid)[0]]
    env.ex.ops = env.fake.ops()
    out = env.ex.close(reason="manual")
    assert out["outcome"] == "restore_failed"
    w = _window(env)
    assert w["holder"] == "released" and w["grant_state"] == "restoring"
    assert gw.is_parked("architect_critic")  # still honestly refused


def test_close_refuses_while_ak_holds_the_device(env):
    _open(env)
    env.fake.device_held = True
    with pytest.raises(gwe.WindowRefused) as exc:
        env.ex.close(reason="manual")
    assert exc.value.reason == "device_held"
    assert _window(env)["holder"] == "autokernel"
    assert not any(c[0] == "reload" for c in env.fake.calls)


def test_reload_attempts_are_bounded(env):
    _open(env)
    env.fake.reload_brings_up = False
    assert env.ex.close(reason="manual")["outcome"] == "restore_failed"
    for _ in range(5):
        env.ex.tick()
    reloads = [c for c in env.fake.calls if c[0] == "reload"]
    env.clock[0] += 3 * 3600
    env.ex.tick()
    assert len(reloads) == gwe.MAX_RELOAD_ATTEMPTS
    assert len([c for c in env.fake.calls if c[0] == "reload"]) == gwe.MAX_RELOAD_ATTEMPTS


# ── watchdog ─────────────────────────────────────────────────────────────────


def test_watchdog_restores_at_end_plus_grace_without_ak(env):
    _open(env, minutes=30)
    env.clock[0] += 30 * 60 + gwe.RESTORE_GRACE_S - 5
    assert env.ex.tick()["holder"] == "autokernel"
    env.clock[0] += 10
    status = env.ex.tick()
    assert status["last_action"] == "close_expiry" and status["holder"] == "production"
    assert json.loads(gwe.status_path(env.window).read_text())["verdict"] == "ok"


def test_watchdog_restores_once_ak_releases(env):
    _open(env)
    env.fake.device_held = True
    assert env.ex.tick()["last_action"] == "none"
    env.fake.device_held = False
    assert env.ex.tick()["last_action"] == "close_ak_released"
    assert _window(env)["holder"] == "production"


def test_watchdog_restores_after_preempt_when_device_free(env):
    _open(env)
    gw.request_preempt("frontdoor request", role="architect_critic", path=env.window)
    assert env.ex.tick()["last_action"] == "close_preempt"


def test_overdue_with_ak_still_holding_is_an_alarm_not_a_restore(env):
    _open(env, minutes=10)
    env.fake.device_held = True
    env.clock[0] += 3600
    status = env.ex.tick()
    assert status["last_action"] == "refused:device_held" and status["verdict"] == "alarm"
    assert _window(env)["holder"] == "autokernel"


def test_reboot_reconcile_neutralises_and_never_starts_servers(env, monkeypatch):
    _open(env)
    monkeypatch.setattr(gwe, "boot_id", lambda: "new-boot")
    env.fake.up = False  # the reboot took the stack down
    status = env.ex.tick()
    assert status["last_action"] == "reboot_reconcile"
    w = _window(env)
    assert w["holder"] == "released" and w["grant_state"] == "restoring"
    env.ex.tick()
    assert not any(c[0] == "reload" for c in env.fake.calls)
    env.fake.up = True  # stack owner brings the stack up
    assert env.ex.tick()["holder"] == "production"


def test_watchdog_restores_a_manual_park_too(env):
    gw.park(roles=ROLES, ports=[PORT], expected_end=_iso(env.clock[0] + 600),
            path=env.window)
    env.fake.up = False
    env.clock[0] += 600 + gwe.RESTORE_GRACE_S
    assert env.ex.tick()["holder"] == "production"
    assert ("reload", "architect_critic") in env.fake.calls


def test_tick_on_production_is_a_noop(env):
    status = env.ex.tick()
    assert status["last_action"] == "none" and status["verdict"] == "ok"
    assert env.fake.calls == []


# ── schedule validator ───────────────────────────────────────────────────────


def test_schedule_validator(env):
    base = env.clock[0]
    good = {"id": "a", "device_id": "mi210_0", "holder": "autokernel", "campaign_id": "c",
            "purpose": "p", "start": _iso(base), "end": _iso(base + 1800)}
    assert gwe.validate_schedule({"schema": gwe.SCHEDULE_SCHEMA, "entries": [good]}) == []
    long = dict(good, id="b", start=_iso(base + 7200), end=_iso(base + 7200 + 7200))
    overlap = dict(good, id="c", start=_iso(base + 900), end=_iso(base + 1000))
    errors = gwe.validate_schedule({"schema": gwe.SCHEDULE_SCHEMA,
                                    "entries": [good, long, overlap, dict(good)]})
    text = " ".join(errors)
    assert "operator_approved" in text and "overlap" in text and "duplicate id" in text
    assert gwe.validate_schedule({"schema": gwe.SCHEDULE_SCHEMA,
                                  "entries": [dict(long, operator_approved=True)]}) == []
    assert gwe.main(["validate-schedule", str(env.schedule)]) == 0


# ── CLI routes manual park/restore through the lease ─────────────────────────


def test_cli_park_capped_and_restore_runs_the_proof(env, monkeypatch, capsys):
    monkeypatch.setattr(gwe, "live_ops", env.fake.ops)
    assert gw.main(["park", "--roles", ",".join(ROLES), "--ports", "8083",
                    "--expected-end", "+4h"]) == 2
    assert not env.window.exists()
    assert gw.main(["park", "--roles", ",".join(ROLES), "--ports", "8083",
                    "--expected-end", "+30m"]) == 0
    assert gw.main(["park", "--roles", "x", "--expected-end", "+30m"]) == 2  # held
    assert gw.main(["restore"]) == 0
    assert _window(env)["holder"] == "production"
    assert any(c[0] == "completion" for c in env.fake.calls)


# ── AP3 role restart respects the window and gpu-quiet ───────────────────────


def test_ap3_restart_refused_during_window_and_gpu_quiet(env, monkeypatch):
    from scripts.autopilot import config_applicator as ca

    ran = []
    monkeypatch.setattr(ca.subprocess, "run", lambda *a, **k: ran.append(a))
    gw.park(roles=ROLES, ports=[PORT], expected_end=_iso(env.clock[0] + 600),
            path=env.window)
    out = ca._reload_role_via_stack(role="architect_critic", env_overrides=None, env_unset=[])
    assert out["status"] == "refused" and out["refusal"]["gate"] == "gpu_window"
    env.window.write_text("{garbled")
    assert ca._reload_role_via_stack(role="x", env_overrides=None,
                                     env_unset=[])["status"] == "refused"
    gw.restore(path=env.window)

    from src.runtime.gpu_quiet_lock import gpu_quiet_lock_path

    lock = gpu_quiet_lock_path()
    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, "a+b") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        out = ca._reload_role_via_stack(role="frontdoor", env_overrides=None, env_unset=[])
    assert out["status"] == "refused" and out["refusal"]["gate"] == "gpu_quiet"
    assert ran == []
    out = ca._reload_role_via_stack(role="frontdoor", env_overrides=None, env_unset=[])
    assert out["status"] == "ok" and len(ran) == 1


# ── hub data plane: panel + health probe ─────────────────────────────────────


def test_gpu_window_panel_and_probe(env):
    import asyncio

    from src.api.routes import dashboard as d

    def probe():
        resp = asyncio.run(d.gpu_window_health())
        return resp.status_code, json.loads(resp.body)

    code, body = probe()
    assert code == 503 and "watchdog not running" in body["reasons"][0]
    env.ex.tick()
    # the status file is stamped with the fake clock; age it against the same clock
    payload = d._gpu_window_payload(now=env.clock[0] + 5)
    assert payload["health"]["status"] == "ok" and payload["executor"]["holder"] == "production"
    assert d._gpu_window_payload(now=env.clock[0] + 600)["health"]["status"] == "degraded"
    _open(env, minutes=10)
    env.fake.device_held = True
    env.clock[0] += 3600
    env.ex.tick()
    alarm = d._gpu_window_payload(now=env.clock[0])
    assert alarm["health"]["status"] == "degraded"
    assert any("alarm" in r for r in alarm["health"]["reasons"])
    body = json.loads(asyncio.run(d.gpu_window_panel()).body)
    assert "_freshness" in body and body["window"]["holder"] == "autokernel"


def test_watchdog_closes_an_interrupted_open_at_once(env, monkeypatch):
    def boom(*a, **k):
        raise OSError("executor killed mid-drain")

    monkeypatch.setattr(gwe, "drain", boom)
    with pytest.raises(OSError):
        _open(env)
    assert _window(env)["grant_state"] == "draining"
    status = env.ex.tick()
    assert status["last_action"] == "interrupted_open" and status["holder"] == "production"
    assert not any(c[0] in ("stop", "reload") for c in env.fake.calls)
