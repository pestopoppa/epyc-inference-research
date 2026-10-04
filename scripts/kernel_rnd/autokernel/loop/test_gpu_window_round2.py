"""G1 round-2 review (tmp/g1-review-20261004/REVIEW-ROUND2.md), AutoKernel half.

F4 window-scoped device flock in the serial_run parent; F5 live compute-request
handshake; F6 expected_end honoured mid-measurement; item 8 llama-bench 62 GiB abort
and the fail-closed sampler. Temp dirs only, no GPU, no live bus, no real lock file.
"""
import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from . import bench, claim, gpu_window as gw, residency, serial_run

NOW = datetime.now(timezone.utc)


def _iso(dt):
    return dt.isoformat()


def _window(**over):
    body = {"holder": "autokernel", "since": _iso(datetime.now(timezone.utc)),
            "expected_end": _iso(datetime.now(timezone.utc) + timedelta(minutes=30)),
            "preempt_requested_at": None, "preempt_reason": None,
            "parked_roles": ["architect_critic"], "parked_ports": [8083],
            "window_id": "w-1", "device_id": "mi210_0", "written_by": "gpu_window_executor",
            "grant_state": "granted", "schedule_ref": "sched-4", "campaign_id": "ak-27b"}
    body.update(over)
    return body


def _write(path, body):
    """Atomic, like the executor's `_atomic_write`: a reader never sees a torn file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(body))
    os.replace(tmp, path)
    return path


def _executor_probe(lock_path) -> bool:
    """The executor's `device_flock_held`: True when some OTHER open file description
    holds the flock (a fresh open + LOCK_EX|LOCK_NB fails)."""
    with open(lock_path, "rb") as fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        return False


@pytest.fixture
def lock_and_window(tmp_path):
    lock = tmp_path / "gpu_device.mi210_0.lock"
    window = _write(tmp_path / "mi210.json", _window())
    return lock, window


# ------------------------------------------------------------------ F4 parent hold

def test_window_hold_is_atomic_and_confirms_the_window_after_the_lock(lock_and_window):
    lock, window = lock_and_window
    hold = claim.WindowDeviceHold(lock, window_path=window)
    grant = hold.acquire()
    assert hold.held and grant.window_id == "w-1"
    assert _executor_probe(lock) is True
    inode = lock.stat().st_ino
    assert hold.release() is True and hold.release() is False
    assert _executor_probe(lock) is False
    assert lock.stat().st_ino == inode  # never unlinked / recreated


def test_window_hold_refuses_and_lets_go_when_the_window_is_not_granted(lock_and_window):
    lock, window = lock_and_window
    _write(window, _window(grant_state="restoring"))
    hold = claim.WindowDeviceHold(lock, window_path=window)
    with pytest.raises(gw.GpuWindowPreempted, match="restoring"):
        hold.acquire()
    assert not hold.held and _executor_probe(lock) is False


def test_window_hold_refuses_a_busy_device_without_queueing(lock_and_window):
    lock, window = lock_and_window
    with open(lock, "a") as other:
        fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(claim.ClaimRefused, match="another holder"):
            claim.WindowDeviceHold(lock, window_path=window).acquire()


def test_flock_survives_between_batches_and_children_reenter_it(lock_and_window, tmp_path):
    """The F4 defect: the flock dropped between batches. Two real child processes
    re-enter the parent's hold via the inherited fd; a child opening its OWN fd is
    refused; between and after the children the executor still sees the device held."""
    lock, window = lock_and_window
    hold = claim.WindowDeviceHold(lock, window_path=window)
    hold.acquire()
    package_root = Path(__file__).resolve().parents[4]
    child = (
        "import fcntl, os, sys\n"
        "from scripts.kernel_rnd.autokernel.loop import claim\n"
        f"lock = {str(lock)!r}\n"
        "from pathlib import Path\n"
        "with claim.hold(Path(lock), device_id='mi210_0') as receipt:\n"
        "    assert receipt.get('window_hold_pid') == int(sys.argv[1]), receipt\n"
        "    assert receipt.observe()['status'] == 'held', receipt.observe()\n"
        "fh = open(lock, 'a')\n"
        "try:\n"
        "    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
        "    print('OWN-FD-ACQUIRED')\n"
        "except BlockingIOError:\n"
        "    print('OWN-FD-REFUSED')\n")
    try:
        for _batch in range(2):
            kwargs = hold.popen_kwargs()
            done = subprocess.run([sys.executable, "-c", child, str(os.getpid())],
                                  cwd=package_root, capture_output=True, text=True,
                                  timeout=60, **kwargs)
            assert done.returncode == 0, done.stderr
            assert "OWN-FD-REFUSED" in done.stdout
            assert _executor_probe(lock) is True  # no gap between batches
    finally:
        hold.release()
    assert _executor_probe(lock) is False


def test_hold_reenters_an_inherited_fd_and_never_unlocks_it(lock_and_window, monkeypatch):
    lock, window = lock_and_window
    hold = claim.WindowDeviceHold(lock, window_path=window)
    hold.acquire()
    try:
        env = hold.popen_kwargs()["env"]
        fd_s, dev, ino, pid = env[claim.INHERITED_DEVICE_LOCK_ENV].split(":")
        dup = os.dup(int(fd_s))  # same open file description, as a forked child has
        monkeypatch.setenv(claim.INHERITED_DEVICE_LOCK_ENV, f"{dup}:{dev}:{ino}:{pid}")
        with claim.hold(lock) as receipt:
            assert receipt["window_hold_pid"] == os.getpid()
        assert _executor_probe(lock) is True  # the child's exit did not drop the parent's hold
        os.close(dup)
        assert _executor_probe(lock) is True
    finally:
        hold.release()


def test_hold_ignores_a_foreign_inherited_fd_and_fails_closed(lock_and_window, tmp_path,
                                                                monkeypatch):
    lock, window = lock_and_window
    hold = claim.WindowDeviceHold(lock, window_path=window)
    hold.acquire()
    other = tmp_path / "other.lock"
    other.write_text("")
    with open(other, "a") as fh:
        st = os.fstat(fh.fileno())
        monkeypatch.setenv(claim.INHERITED_DEVICE_LOCK_ENV,
                           f"{fh.fileno()}:{st.st_dev}:{st.st_ino}:{os.getpid()}")
        with pytest.raises(claim.ClaimRefused, match="another holder"):
            with claim.hold(lock):
                pytest.fail("a second holder")
    hold.release()


def test_hold_verify_runs_after_the_lock_and_releases_on_refusal(lock_and_window):
    lock, window = lock_and_window
    seen = []

    def verify():
        seen.append(_executor_probe(lock))
        raise gw.GpuWindowPreempted("window not granted")

    with pytest.raises(gw.GpuWindowPreempted):
        with claim.hold(lock, verify=verify):
            pytest.fail("measured outside a window")
    assert seen == [True] and _executor_probe(lock) is False


def test_window_hold_boundary_and_abort_checks(lock_and_window):
    lock, window = lock_and_window
    hold = claim.WindowDeviceHold(lock, window_path=window)
    hold.acquire()
    try:
        assert hold.refusal() is None and hold.expired_or_revoked() is None
        _write(window, _window(preempt_requested_at=_iso(NOW)))
        assert "preempt" in hold.refusal()
        assert hold.expired_or_revoked() is None  # a preempt is a drain, not a kill
        _write(window, _window(window_id="w-2"))
        assert "window changed" in hold.refusal()
        assert hold.expired_or_revoked() is None  # first sight only
        assert "replaced" in hold.expired_or_revoked()  # persisted: revoked
    finally:
        hold.release()
    late = claim.WindowDeviceHold(lock, window_path=window,
                                  clock=lambda: datetime.now(timezone.utc) + timedelta(hours=1))
    _write(window, _window())
    with pytest.raises(gw.GpuWindowPreempted, match="expired"):
        late.acquire()


class _FakeHandshake:
    def __init__(self, window_id="w-1"):
        self.calls = []
        self.window_id = window_id

    def request(self, request):
        self.calls.append(("request", request.campaign_id))
        return "msg-1"

    def wait_grant(self, corr_id, timeout_s):
        self.calls.append(("wait", corr_id, timeout_s))
        return gw.WindowGrant(holder="autokernel", expected_end=None, parked_ports=(8083,),
                              preempt_requested_at=None, window_id=self.window_id,
                              compute_grant="msg-g")

    def release(self, grant, *, hold=None, campaign_id=None):
        self.calls.append(("release", grant.compute_grant, campaign_id))


def _session(lock, window, handshake):
    request = {"request": gw.WindowRequest("ak-27b", "p", 600, schedule_ref="sched-4"),
               "timeout_s": 5.0, "bus_root": None}
    return serial_run._GpuWindowSession(
        window, request, hold=claim.WindowDeviceHold(lock, window_path=window),
        handshake=handshake)


def test_serial_session_requests_once_holds_across_batches_and_releases(lock_and_window):
    lock, window = lock_and_window
    handshake = _FakeHandshake()
    session = _session(lock, window, handshake)
    assert session.ensure() is None
    assert session.ensure() is None  # second GPU batch: re-check only
    assert [c[0] for c in handshake.calls] == ["request", "wait"]
    assert "pass_fds" in session.popen_kwargs() and _executor_probe(lock) is True
    session.close()
    assert handshake.calls[-1] == ("release", "msg-g", "ak-27b")
    assert _executor_probe(lock) is False
    session.close()
    assert [c[0] for c in handshake.calls].count("release") == 1


def test_serial_session_refuses_a_window_that_is_not_the_granted_one(lock_and_window):
    lock, window = lock_and_window
    session = _session(lock, window, _FakeHandshake(window_id="w-other"))
    assert "not the granted window" in session.ensure()
    assert _executor_probe(lock) is False


def test_serial_session_ends_the_hold_when_the_window_stops_granting(lock_and_window):
    lock, window = lock_and_window
    session = _session(lock, window, _FakeHandshake())
    assert session.ensure() is None
    _write(window, _window(preempt_requested_at=_iso(NOW)))
    assert "preempt" in session.ensure()
    assert _executor_probe(lock) is False  # released at the refusing boundary


def test_serial_run_cli_refuses_inconsistent_window_flags(tmp_path, capsys):
    with pytest.raises(SystemExit):
        # `_main` is the argparse core; `main` is the lane_watchdog supervisor wrapper.
        serial_run._main(["--state-dir", str(tmp_path), "--batch-iterations", "1",
                          "--target-args", str(tmp_path / "t.json"),
                          "--gpu-window-request-schedule-ref", "s"])
    assert "needs --gpu-window required" in capsys.readouterr().err
    gpu_target = ["--target-id", "g", "--gpu-window", "required"]
    assert serial_run._gpu_window_target_refusal("off", [["--target-id", "g"]]) is None
    assert serial_run._gpu_window_target_refusal("required", [gpu_target]) is None
    assert "same flag" in serial_run._gpu_window_target_refusal(
        "required", [["--target-id", "g"]])
    assert "gpu-quiet off" in serial_run._gpu_window_target_refusal(
        "required", [[*gpu_target, "--cpu-measurement-gpu-quiet", "off"]])
    assert serial_run._gpu_window_target_refusal(
        "required", [["--target-id", "c", "--cpu-serving-launch", "x.json"]]) is None


def test_run_refuses_gpu_quiet_off_under_a_required_window(tmp_path, capsys):
    from . import run
    with pytest.raises(SystemExit):
        run.main(["--worktree", str(tmp_path), "--anchor-build", str(tmp_path),
                  "--store", str(tmp_path), "--gpu-window", "required",
                  "--cpu-measurement-gpu-quiet", "off", "--dry-run"])
    assert "cannot run with --cpu-measurement-gpu-quiet off" in capsys.readouterr().err


# ------------------------------------------------------------------ F5 handshake

def _schedule(tmp_path, start, minutes=60):
    path = tmp_path / "schedule.json"
    path.write_text(json.dumps({"schema": gw.SCHEDULE_SCHEMA, "entries": [{
        "id": "sched-4", "device_id": "mi210_0", "holder": "autokernel",
        "campaign_id": "ak-27b", "purpose": "rocprof", "start": _iso(start),
        "end": _iso(start + timedelta(minutes=minutes))}]}))
    return path


class _Bus:
    """A temp bus root, an appender recording AK's outbox rows, and a fake clock."""

    def __init__(self, tmp_path):
        self.root = tmp_path / "bus"
        (self.root / "inbox").mkdir(parents=True)
        self.inbox = self.root / "inbox" / "inference.jsonl"
        self.inbox.write_text("")
        self.sent = []
        self.t = 0.0
        self.on_sleep = []

    def append(self, msg):
        self.sent.append(json.loads(json.dumps(msg)))

    def deliver(self, **row):
        with self.inbox.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

    def sleep(self, s):
        self.t += s
        for hook in list(self.on_sleep):
            hook()

    def monotonic(self):
        return self.t


def _handshake(tmp_path, bus, *, window=None):
    now = datetime.now(timezone.utc)
    window = window or (tmp_path / "mi210.json")
    return gw.BusStackOwnerWindow(
        bus_root=bus.root, window_path=window, schedule_path=_schedule(tmp_path, now),
        appender=bus.append, sleep=bus.sleep, monotonic=bus.monotonic, poll_s=1.0)


def _LATER(minutes):
    """Placeholder resolved at test time (the module is imported long before it runs)."""
    return f"@later:{minutes}"


def _grant_row(request_id, *, end=None, **over):
    end = end or datetime.now(timezone.utc) + timedelta(minutes=9)
    row = {"kind": "compute-grant", "from": "coordinator-daemon", "to": "inference",
           "id": "msg-20261004T120000Z-1-coordinator-daemon", "corr_id": request_id,
           "payload": {"task": "t", "window": "now", "device_id": "mi210_0",
                       "expected_end": _iso(end), "campaign_id": "ak-27b"}}
    row.update(over)
    return row


def _open_window(tmp_path, grant_id, *, window_id="w-1", **over):
    window = _write(tmp_path / "mi210.json", _window(
        window_id=window_id,
        expected_end=_iso(datetime.now(timezone.utc) + timedelta(minutes=8)), **over))
    gw.lease_path(window).write_text(json.dumps({
        "window_id": window_id, "state": "open",
        "authority": {"kind": "stack_owner", "principal": "workspace-ec",
                      "compute_grant": grant_id}}))
    return window


def test_request_appends_one_schema_shaped_compute_request(tmp_path):
    bus = _Bus(tmp_path)
    hs = _handshake(tmp_path, bus)
    corr = hs.request(gw.WindowRequest("ak-27b", "p", 600, schedule_ref="sched-4"))
    assert re.fullmatch(r"msg-[0-9]{8}T[0-9]{6}Z-[0-9]+-inference", corr)
    (msg,) = bus.sent
    assert msg["id"] == corr and msg["kind"] == "compute-request"
    assert msg["to"] == "coordinator-daemon" and msg["from"] == "inference"
    for key in ("task", "window", "device_region", "est_h", "release_condition"):
        assert msg["payload"][key] not in (None, "")


def test_a_compute_deny_fails_closed(tmp_path):
    bus = _Bus(tmp_path)
    hs = _handshake(tmp_path, bus)
    corr = hs.request(gw.WindowRequest("ak-27b", "p", 600, schedule_ref="sched-4"))
    bus.deliver(kind="compute-deny", **{"from": "coordinator-daemon"}, corr_id=corr,
                payload={"task": "t", "reason": "gpu reserved", "next": "retry 22:00"})
    with pytest.raises(gw.WindowDenied, match="gpu reserved"):
        hs.wait_grant(corr, timeout_s=10)


def test_no_grant_times_out_and_other_requests_grants_are_ignored(tmp_path):
    bus = _Bus(tmp_path)
    hs = _handshake(tmp_path, bus)
    corr = hs.request(gw.WindowRequest("ak-27b", "p", 600, schedule_ref="sched-4"))
    bus.deliver(**_grant_row("msg-someone-else"))
    with pytest.raises(gw.WindowGrantTimeout, match="no compute-grant"):
        hs.wait_grant(corr, timeout_s=5)
    with pytest.raises(gw.GrantUnverified, match="not a request"):
        hs.wait_grant("msg-never-sent", timeout_s=5)


@pytest.mark.parametrize("over, match", [
    ({"from": "mainB"}, "not coordinator-daemon"),
    ({"payload": {"task": "t", "window": "w", "device_id": "mi100_0",
                  "expected_end": _LATER(5)}}, "mi100_0"),
    ({"payload": {"task": "t", "window": "w", "device_id": "mi210_0"}}, "expected_end"),
    ({"payload": {"task": "t", "window": "w", "device_id": "mi210_0",
                  "expected_end": _LATER(40)}}, "past the requested"),
    ({"payload": {"task": "t", "window": "w", "device_id": "mi210_0",
                  "expected_end": _LATER(-1)}}, "already passed"),
    ({"payload": {"task": "t", "window": "w", "device_id": "mi210_0", "campaign_id": "x",
                  "expected_end": _LATER(5)}}, "campaign"),
])
def test_a_mismatching_bus_grant_is_refused(tmp_path, over, match):
    over = json.loads(json.dumps(over))  # never mutate the shared parameter
    payload = over.get("payload")
    if payload and isinstance(payload.get("expected_end"), str) \
            and payload["expected_end"].startswith("@later:"):
        minutes = int(payload["expected_end"].split(":")[1])
        payload["expected_end"] = _iso(datetime.now(timezone.utc) + timedelta(minutes=minutes))
    bus = _Bus(tmp_path)
    hs = _handshake(tmp_path, bus)
    corr = hs.request(gw.WindowRequest("ak-27b", "p", 600, schedule_ref="sched-4"))
    bus.deliver(**_grant_row(corr, **over))
    with pytest.raises(gw.GrantUnverified, match=match):
        hs.wait_grant(corr, timeout_s=10)


def test_a_verified_grant_waits_for_the_executor_window_bound_to_it(tmp_path):
    bus = _Bus(tmp_path)
    hs = _handshake(tmp_path, bus)
    corr = hs.request(gw.WindowRequest("ak-27b", "p", 600, schedule_ref="sched-4"))
    grant = _grant_row(corr)
    bus.deliver(**grant)
    polls = []

    def executor_opens():  # the executor drains, stops, then writes granted + lease
        polls.append(1)
        if len(polls) == 3:
            _open_window(tmp_path, grant["id"])

    bus.on_sleep.append(executor_opens)
    window = hs.wait_grant(corr, timeout_s=60)
    assert window.window_id == "w-1" and window.compute_grant == grant["id"]
    assert len(polls) == 3


@pytest.mark.parametrize("window_over, lease_grant, match", [
    ({}, "msg-other-grant", "compute_grant"),
    ({"campaign_id": "other"}, None, "campaign"),
    ({"schedule_ref": "sched-9"}, None, "schedule_ref"),
    ({"grant_state": "draining"}, None, "grant_state"),
    ({"expected_end": _iso(NOW + timedelta(minutes=30))}, None, "past the grant"),
])
def test_a_window_not_opened_for_this_grant_is_never_used(tmp_path, window_over,
                                                          lease_grant, match):
    bus = _Bus(tmp_path)
    hs = _handshake(tmp_path, bus)
    corr = hs.request(gw.WindowRequest("ak-27b", "p", 600, schedule_ref="sched-4"))
    grant = _grant_row(corr)
    bus.deliver(**grant)
    window = _open_window(tmp_path, lease_grant or grant["id"])
    if window_over:
        _write(window, {**json.loads(window.read_text()), **window_over})
    with pytest.raises(gw.WindowGrantTimeout, match=match):
        hs.wait_grant(corr, timeout_s=3)


def test_a_missing_lease_fails_closed(tmp_path):
    bus = _Bus(tmp_path)
    hs = _handshake(tmp_path, bus)
    corr = hs.request(gw.WindowRequest("ak-27b", "p", 600, schedule_ref="sched-4"))
    grant = _grant_row(corr)
    bus.deliver(**grant)
    window = _open_window(tmp_path, grant["id"])
    gw.lease_path(window).unlink()
    with pytest.raises(gw.WindowGrantTimeout, match="lease unreadable"):
        hs.wait_grant(corr, timeout_s=3)


def test_release_drops_the_flock_then_sends_a_status_note(tmp_path, lock_and_window):
    lock, window = lock_and_window
    bus = _Bus(tmp_path)
    hs = _handshake(tmp_path, bus, window=window)
    hold = claim.WindowDeviceHold(lock, window_path=window)
    grant = hold.acquire()
    hs.release(grant, hold=hold, campaign_id="ak-27b")
    assert _executor_probe(lock) is False
    (note,) = bus.sent
    assert note["kind"] == "status" and note["payload"]["event"] == "gpu-window-released"
    assert note["payload"]["window_id"] == "w-1"


def test_the_live_cli_accepts_the_request_and_release_rows(tmp_path):
    """Schema + single-writer conformance against the real session_bus.py CLI, on a
    TEMP bus root seeded with copies of the live roster and schema (read-only)."""
    live = Path("/workspace/coordination/session-bus")
    script = Path("/workspace/scripts/coordination/session_bus.py")
    if not (script.is_file() and (live / "config.yaml").is_file()
            and (live / "session_bus.schema.json").is_file()):
        pytest.skip("live session-bus CLI/roster not present on this host")
    root = tmp_path / "bus"
    for sub in ("inbox", "outbox", "heartbeats", "cursors"):
        (root / sub).mkdir(parents=True)
    for name in ("config.yaml", "session_bus.schema.json"):
        shutil.copy(live / name, root / name)
    now = datetime.now(timezone.utc)
    hs = gw.BusStackOwnerWindow(bus_root=root, window_path=tmp_path / "mi210.json",
                                schedule_path=_schedule(tmp_path, now))
    corr = hs.request(gw.WindowRequest("ak-27b", "p", 600, schedule_ref="sched-4"))
    rows = [json.loads(line) for line in
            (root / "outbox" / "inference.jsonl").read_text().splitlines()]
    assert rows[-1]["id"] == corr and rows[-1]["kind"] == "compute-request"
    hs.release(gw.read_grant(_write(tmp_path / "w.json", _window())), campaign_id="ak-27b")
    rows = [json.loads(line) for line in
            (root / "outbox" / "inference.jsonl").read_text().splitlines()]
    assert rows[-1]["kind"] == "status" and len(rows) == 2


# ------------------------------------------------------------------ F6 expected_end

def test_abort_reason_covers_expiry_and_revocation_but_not_preempt(tmp_path):
    path = _write(tmp_path / "mi210.json", _window())
    grant = gw.read_grant(path)
    assert gw.abort_reason(grant, path) is None
    later = datetime.now(timezone.utc) + timedelta(hours=1)
    assert "expected_end" in gw.abort_reason(grant, path, now=later)
    _write(path, _window(preempt_requested_at=_iso(NOW)))
    assert gw.abort_reason(grant, path) is None
    for over, match in (({"holder": "production"}, "holder"),
                        ({"grant_state": "restoring"}, "restoring"),
                        ({"window_id": "w-9"}, "replaced"),
                        ({"expected_end": _iso(NOW - timedelta(seconds=1))}, "reached")):
        _write(path, _window(**over))
        assert match in gw.abort_reason(grant, path)
    path.unlink()
    assert gw.abort_reason(grant, path) is not None  # unreadable is not "still mine"


def test_expiry_watcher_needs_two_reads_to_believe_a_revocation(tmp_path):
    path = _write(tmp_path / "mi210.json", _window())
    fired = []
    watcher = gw.ExpiryWatcher(gw.read_grant(path), fired.append, path=path)
    path.write_text("{torn")  # one transient unreadable read
    assert watcher.check() is None and fired == []
    _write(path, _window())
    assert watcher.check() is None  # streak reset
    _write(path, _window(grant_state="restoring"))
    assert watcher.check() is None and fired == []
    assert "restoring" in watcher.check() and len(fired) == 1


def test_expiry_watcher_fires_once(tmp_path):
    path = _write(tmp_path / "mi210.json", _window())
    fired = []
    late = datetime.now(timezone.utc) + timedelta(hours=1)
    watcher = gw.ExpiryWatcher(gw.read_grant(path), fired.append, path=path, poll_s=0.01,
                               clock=lambda: late)
    with watcher:
        deadline = time.monotonic() + 5
        while not fired and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.05)
    assert len(fired) == 1 and "expected_end" in fired[0]


def _sleeper(seconds=60, ignore_term=False):
    code = ("import signal, sys, time\n"
            + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore_term else "")
            + "print('ready', flush=True)\n"
            + f"time.sleep({seconds})\n")
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "ready"
    return proc


def test_measurement_in_flight_is_aborted_at_expected_end_and_never_returned(tmp_path):
    from . import loop, run
    path = _write(tmp_path / "mi210.json", _window(
        expected_end=_iso(datetime.now(timezone.utc) + timedelta(seconds=1.0))))
    stops = []
    proc = _sleeper()
    try:
        with pytest.raises(run.GpuWindowMeasurementAborted, match="no result") as info:
            with run._gpu_window_boundary(True, path, on_preempt=lambda: stops.append(1),
                                          poll_s=0.05):
                residency.OWN_LAUNCHES.register(proc)
                try:
                    proc.wait(timeout=30)  # the measurement's own child, killed by the watcher
                finally:
                    residency.OWN_LAUNCHES.unregister(proc)
                result = {"effect": 9.9}  # whatever the body computed is discarded
        assert isinstance(info.value, loop.TailRefused)
        assert proc.poll() is not None and stops
        assert residency.OWN_LAUNCHES.reason is None  # reset for the next measurement
        assert result  # the body ran to its end; the boundary still refused it
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_measurement_that_finishes_before_expected_end_is_untouched(tmp_path):
    from . import run
    path = _write(tmp_path / "mi210.json", _window())
    with run._gpu_window_boundary(True, path, on_preempt=lambda: pytest.fail("aborted"),
                                  poll_s=0.01):
        time.sleep(0.05)


def test_a_preempt_after_the_gpu_quiet_wait_refuses_the_measurement(tmp_path):
    from . import loop, run
    path = _write(tmp_path / "mi210.json", _window())
    stops = []

    class _Wait:  # gpu-quiet wait during which the window gets a preempt request
        def __enter__(self):
            _write(path, _window(preempt_requested_at=_iso(NOW)))

        def __exit__(self, *exc):
            return False

    with pytest.raises(loop.TailRefused, match="preempt"):
        with run._stacked(
                run._gpu_window_boundary(True, path, on_preempt=lambda: stops.append(1),
                                         poll_s=0.01),
                run._stacked(_Wait(), run._gpu_window_boundary(
                    True, path, watch=False, on_preempt=lambda: stops.append(2)))):
            pytest.fail("measured after a preempt")
    assert 2 in stops


def test_own_launches_terminates_a_late_registration_after_abort():
    launches = residency.OwnLaunches()
    launches.abort("window over")
    proc = _sleeper()
    launches.register(proc)
    assert proc.poll() is not None
    with pytest.raises(residency.MeasurementAborted, match="window over"):
        launches.raise_if_aborted()
    launches.reset()
    launches.raise_if_aborted()


def test_bench_does_not_retry_or_capture_an_aborted_launch(monkeypatch):
    from unittest import mock
    calls, captured = [], []

    def fake_launch(argv, **kwargs):
        calls.append(argv)
        residency.OWN_LAUNCHES.abort("window expected_end reached")
        return subprocess.CompletedProcess(argv, -15, "", "")

    capture = mock.Mock()
    capture.invocation.return_value = mock.MagicMock()
    capture.completed.side_effect = lambda *a: captured.append(a)
    try:
        with mock.patch.object(bench, "_launch", side_effect=fake_launch), \
                mock.patch.object(bench.residency, "Sampler"), \
                mock.patch.object(bench.residency, "loader_env", return_value={}):
            with pytest.raises(residency.MeasurementAborted):
                bench.run_once(Path("/b"), Path("/m"), pp=0, tg=128, sleep=lambda _s: None,
                               capture=capture)
    finally:
        residency.OWN_LAUNCHES.reset()
    assert len(calls) == 1 and captured == []


# ------------------------------------------------------------------ item 8 bench abort

def test_terminate_own_process_escalates_to_sigkill_and_verifies():
    proc = _sleeper(ignore_term=True)
    assert residency.terminate_own_process(proc, grace_s=0.3) is True
    assert proc.returncode == -signal.SIGKILL


def test_llama_bench_child_is_killed_above_62_gib(monkeypatch):
    monkeypatch.setattr(bench, "llama_bench_argv", lambda *a, **k: [
        sys.executable, "-c", "import time; time.sleep(60)"])
    monkeypatch.setattr(residency, "loader_env", lambda _b: dict(os.environ))
    monkeypatch.setattr(residency, "vram_bytes", lambda: 63 << 30)
    monkeypatch.setattr(residency, "kfd_pid_vram", lambda _pid: (62 << 30) + 4096)
    monkeypatch.setattr(residency, "kfd_processes", lambda: 1)
    monkeypatch.setattr(residency, "sclk_mhz", lambda: 0)
    monkeypatch.setattr(residency, "pid_peak_rss", lambda _pid: None)
    seen = {}
    real = residency.terminate_own_process

    def spy(proc, **kw):
        seen["pid"] = proc.pid
        ok = real(proc, **kw)
        seen["dead"] = proc.poll() is not None
        return ok

    monkeypatch.setattr(residency, "terminate_own_process", spy)
    started = time.monotonic()
    with pytest.raises(residency.VramCeilingExceeded, match="62.00 GiB ceiling"):
        bench.run_once(Path("/b"), Path("/m"), pp=0, tg=128, sleep=lambda _s: None)
    assert seen["dead"] is True and time.monotonic() - started < 30


def test_bench_launch_returns_a_completed_process_for_a_normal_child():
    class _Sampler:
        watched = None

        def watch_pid(self, pid):
            self.watched = pid

        def abort_above(self, ceiling, terminate):
            self.ceiling = ceiling

    sampler = _Sampler()
    done = bench._launch([sys.executable, "-c", "print('[]')"], timeout=30,
                         env=dict(os.environ), sampler=sampler)
    assert done.returncode == 0 and done.stdout.strip() == "[]"
    assert sampler.watched and sampler.ceiling == 62 << 30


def _sampler(monkeypatch, *, device, own):
    monkeypatch.setattr(residency, "vram_bytes", lambda: device)
    monkeypatch.setattr(residency, "kfd_pid_vram", lambda _pid: own)
    monkeypatch.setattr(residency, "pid_peak_rss", lambda _pid: None)
    sampler = residency.Sampler(interval=0.01)
    sampler.watch_pid(4242)
    killed = []
    sampler.abort_above(62 << 30, lambda: killed.append(1))
    return sampler, killed


def test_sampler_unread_own_kfd_falls_back_to_the_device_total(monkeypatch):
    sampler, killed = _sampler(monkeypatch, device=63 << 30, own=None)
    sampler._check_ceiling(residency.vram_bytes(), sampler._watch_once())
    assert killed == [1] and sampler.ceiling_breach["source"] == "device_vram"


def test_sampler_fails_closed_after_consecutive_unreadable_samples(monkeypatch):
    sampler, killed = _sampler(monkeypatch, device=-1, own=None)
    for _ in range(residency.UNREADABLE_SAMPLES_FAIL_CLOSED - 1):
        sampler._check_ceiling(residency.vram_bytes(), sampler._watch_once())
    assert killed == []
    sampler._check_ceiling(-1, None)  # a readable sample would have reset the streak
    assert killed == [1] and sampler.ceiling_breach["source"] == "unreadable"
    with pytest.raises(residency.VramCeilingExceeded, match="fail closed"):
        sampler.raise_if_ceiling_breached()


def test_sampler_streak_resets_on_a_readable_sample(monkeypatch):
    sampler, killed = _sampler(monkeypatch, device=-1, own=None)
    for _ in range(residency.UNREADABLE_SAMPLES_FAIL_CLOSED - 1):
        sampler._check_ceiling(-1, None)
    sampler._check_ceiling(1 << 30, None)
    for _ in range(residency.UNREADABLE_SAMPLES_FAIL_CLOSED - 1):
        sampler._check_ceiling(-1, None)
    assert killed == [] and sampler.ceiling_breach is None


def test_serial_router_hands_the_hold_to_gpu_children_and_releases_at_the_end():
    source = open(serial_run.__file__, encoding="utf-8").read()
    boundary = source.index('gpu_batch = (gpu_window == "required"')
    ensure = source.index("window_session.ensure()", boundary)
    popen = source.index("process = subprocess.Popen(", boundary)
    assert boundary < ensure < popen
    handoff = source.index("window_session.popen_kwargs_for(child_env)", boundary)
    assert ensure < handoff < popen
    assert "pass_fds=capability_fds" in source[popen:popen + 400]
    assert "window_session.hold.expired_or_revoked()" in source
    tail = source[source.index("def _drive("):]
    assert "window_session.close()  # every child is reaped by now" in tail[tail.rindex(
        "    finally:\n        try:\n            if service is not None:"):]
