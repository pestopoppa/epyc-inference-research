"""DS41-C84: no AutoKernel process outlives the call, gate or run that spawned it.

Every real process these tests start is a `sleep` (no CPU): the live campaign measures
on this host. CPU-burning "foreign" processes are simulated in a synthetic /proc tree.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time

import pytest

from . import actors, ak_check, procguard, scratch, serving
from .loop import MeasurementInvalid

SLEEP = "/bin/sleep" if os.path.exists("/bin/sleep") else "sleep"


# -- helpers -----------------------------------------------------------------------------

def _gone(pid: int, timeout_s: float = 8.0) -> bool:
    """Dead: no /proc entry, or a zombie (its new parent reaps it)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        stat = procguard.read_stat(Path("/proc"), pid)
        if stat is None or stat[2] == "Z":
            return True
        time.sleep(0.05)
    return False


def _wait_for_pid(path: Path, timeout_s: float = 10.0) -> int:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            text = path.read_text().strip()
            if text:
                return int(text)
        except (OSError, ValueError):
            pass
        time.sleep(0.05)
    raise AssertionError(f"no pid appeared in {path}")


def _kill_own(pid: int) -> None:
    """Cleanup of a process THIS test started (pid captured by the test itself)."""
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


#: A "tool" that escapes exactly like C84: its sleeper runs in a NEW session and the
#: tool exits without waiting, so the sleeper is reparented away from the actor.
ESCAPING_TOOL = r"""
import subprocess, sys, time
child = subprocess.Popen([sys.argv[1], "300"], start_new_session=True,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
open(sys.argv[2], "w").write(str(child.pid))
time.sleep(float(sys.argv[3]))
"""


def _plant_orphan(cwd: Path, pidfile: Path) -> int:
    """A double-forked sleeper in `cwd`: NOT a descendant of this test process and
    WITHOUT a cookie -- the shape of the C84 orphan, which predates the cookie."""
    env = {k: v for k, v in os.environ.items() if k != procguard.ENV_SCOPE}
    subprocess.run([sys.executable, "-c", ESCAPING_TOOL, SLEEP, str(pidfile), "0"],
                   cwd=str(cwd), env=env, check=True, timeout=30)
    pid = _wait_for_pid(pidfile)
    assert procguard.read_stat(Path("/proc"), pid)[0] != os.getpid()
    return pid


# -- the call scope: actor / validator ---------------------------------------------------

def test_setsid_grandchild_of_an_actor_is_dead_when_the_call_returns(tmp_path):
    pidfile = tmp_path / "sleeper.pid"
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        done = actors._run_stoppable(
            [sys.executable, "-c", ESCAPING_TOOL, SLEEP, str(pidfile), "0"],
            out=out, err=err, timeout_s=60, cwd=tmp_path, should_stop=lambda: False,
            extra={})
    assert done.returncode == 0
    pid = _wait_for_pid(pidfile)
    try:
        assert _gone(pid), "the setsid'd tool child outlived the actor call"
        record = procguard.current().records[-1]
        assert record["reason"] == "call_scope_closed"
        assert pid in [row["pid"] for row in record["targets"]]
        assert record["survivors"] == []
    finally:
        _kill_own(pid)


def test_setsid_grandchild_is_dead_after_the_actor_timeout_path(tmp_path):
    pidfile = tmp_path / "sleeper.pid"
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        with pytest.raises(subprocess.TimeoutExpired):
            actors._run_stoppable(
                [sys.executable, "-c", ESCAPING_TOOL, SLEEP, str(pidfile), "300"],
                out=out, err=err, timeout_s=3, cwd=tmp_path, should_stop=lambda: False,
                extra={}, grace_s=1.0)
    pid = _wait_for_pid(pidfile)
    try:
        assert _gone(pid), "a timed-out actor's setsid'd tool child survived"
    finally:
        _kill_own(pid)


def test_setsid_grandchild_is_dead_after_a_stop(tmp_path):
    pidfile = tmp_path / "sleeper.pid"
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        with pytest.raises(actors._StoppedChild):
            actors._run_stoppable(
                [sys.executable, "-c", ESCAPING_TOOL, SLEEP, str(pidfile), "300"],
                out=out, err=err, timeout_s=60, cwd=tmp_path,
                should_stop=lambda: pidfile.exists() and bool(pidfile.read_text().strip()),
                extra={}, grace_s=1.0, poll_s=0.1)
    pid = _wait_for_pid(pidfile)
    try:
        assert _gone(pid)
    finally:
        _kill_own(pid)


def test_call_scope_sweeps_only_its_own_cookie(tmp_path):
    """A sibling call's process (another cookie) is never touched by this call's end."""
    guard = procguard.Guard()
    with guard.call_scope() as other_env:
        other = subprocess.Popen([SLEEP, "300"], env=other_env)
        try:
            with guard.call_scope() as env:
                mine = subprocess.Popen([SLEEP, "300"], env=env)
            assert mine.wait(timeout=10) is not None
            assert other.poll() is None, "a different call's process was swept"
        finally:
            _kill_own(other.pid)
            other.wait(timeout=10)


# -- ak-check: the root cause ------------------------------------------------------------

AK_HELPER = r"""
import sys
sys.path.insert(0, sys.argv[1])
import ak_check
ak_check.install_signal_cleanup()
ak_check.run_bounded([sys.argv[2], "300"], cpus=[int(sys.argv[3])], timeout_s=600, cwd=".")
"""


def _child_named(parent: int, comm: str, timeout_s: float = 10.0) -> int:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for proc in procguard.snapshot().values():
            if proc.ppid == parent:
                try:
                    if Path(f"/proc/{proc.pid}/comm").read_text().strip() == comm:
                        return proc.pid
                except OSError:
                    pass
        time.sleep(0.05)
    raise AssertionError(f"no {comm} child of {parent}")


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGKILL])
def test_ak_check_child_dies_with_ak_check(tmp_path, sig):
    """C84 exactly: ak-check is signalled (opencode's shell-tool timeout sends TERM to
    its own group, then KILL) while its own-session child runs."""
    cpu = sorted(os.sched_getaffinity(0))[-1]
    helper = subprocess.Popen([sys.executable, "-c", AK_HELPER,
                               str(Path(ak_check.__file__).parent), SLEEP, str(cpu)],
                              cwd=str(tmp_path))
    child = None
    try:
        child = _child_named(helper.pid, "sleep")
        assert os.getsid(child) == child and os.getsid(child) != os.getsid(helper.pid)
        os.kill(helper.pid, sig)
        helper.wait(timeout=15)
        assert _gone(child), f"ak-check's child outlived a {sig.name} of ak-check"
    finally:
        if child is not None:
            _kill_own(child)
        _kill_own(helper.pid)
        helper.wait(timeout=10)


def test_run_bounded_timeout_ends_the_group_and_verifies_it(tmp_path):
    cpu = sorted(os.sched_getaffinity(0))[-1]
    started = time.monotonic()
    code, _out, timed_out = ak_check.run_bounded(
        ["sh", "-c", f"{SLEEP} 300 & {SLEEP} 300"], cpus=[cpu], timeout_s=1, cwd=tmp_path)
    assert timed_out and code == -9 and time.monotonic() - started < 30
    with ak_check._LIVE_LOCK:
        assert not ak_check._LIVE_GROUPS


def test_the_fence_slot_is_inherited_by_the_child(tmp_path):
    """The lock follows the work: while the child runs it holds the slot fd."""
    import threading
    fence = tmp_path / ak_check.FENCE_DIR_NAME
    cpu = sorted(os.sched_getaffinity(0))[-1]
    seen = {}
    with ak_check.sandbox_slot(fence):
        fds = list(ak_check._INHERITED_LOCK_FDS)
        assert len(fds) == 1

        def run():
            seen["result"] = ak_check.run_bounded([SLEEP, "2"], cpus=[cpu], timeout_s=30,
                                                  cwd=tmp_path)
        thread = threading.Thread(target=run)
        thread.start()
        child = _child_named(os.getpid(), "sleep")
        assert os.path.exists(f"/proc/{child}/fd/{fds[0]}")
        thread.join(timeout=30)
    assert seen["result"][0] == 0
    assert ak_check._INHERITED_LOCK_FDS == []


# -- ownership classes on a synthetic /proc (no real CPU load) ---------------------------

class FakeProc:
    def __init__(self, root: Path):
        self.root = root

    def add(self, pid, *, ppid=1, start=1000, state="S", cpu=0, tty=0, cwd="/", exe="/usr/bin/x",
            scope=None):
        d = self.root / str(pid)
        d.mkdir(parents=True, exist_ok=True)
        tail = [state, ppid, pid, pid, tty, -1, 0, 0, 0, 0, 0, cpu, 0, 0, 0, 20, 0, 1, 0, start,
                0, 0]
        (d / "stat").write_text(f"{pid} (x) " + " ".join(str(v) for v in tail) + "\n")
        env = b"PATH=/bin\0" + ((procguard.ENV_SCOPE + "=" + scope).encode() + b"\0"
                                 if scope else b"")
        (d / "environ").write_bytes(env)
        for name, target in (("cwd", cwd), ("exe", exe)):
            link = d / name
            if link.is_symlink():
                link.unlink()
            os.symlink(target, link)

    def burn(self, pid, cpu):
        stat = (self.root / str(pid) / "stat").read_text().split()
        stat[13] = str(cpu)      # utime (field 14)
        (self.root / str(pid) / "stat").write_text(" ".join(stat) + "\n")


def _synthetic(tmp_path):
    store = tmp_path / "campaign" / "store"
    (store / "scratch" / "author" / "lane0" / "ak-check").mkdir(parents=True)
    fake = FakeProc(tmp_path / "proc")
    ended = []

    def terminate(targets, still_selected):
        ended.extend(p.pid for p in targets if still_selected(p))
        return {"terminated": [p.pid for p in targets], "killed": [], "survivors": []}

    guard = procguard.Guard(store=store, proc_root=fake.root, identity=(100, 5000),
                            terminate=terminate,
                            never_sweep=("/home/node/.codex/packages/app-server-daemon/",))
    key = guard.key
    open_scope = guard.new_scope()
    guard._open.add(open_scope)
    closed_scope = guard.new_scope()
    s = str(store)
    fake.add(1, ppid=0, start=1)
    fake.add(50, ppid=1, start=10)                                     # the serial parent
    fake.add(100, ppid=50, start=5000)                                 # this loop
    fake.add(101, ppid=100, cwd=s + "/anchor-gen-009", exe=s + "/anchor-gen-009/bin/llama-server")
    fake.add(200, scope=closed_scope, cwd="/tmp")                     # this owner, closed call
    fake.add(201, scope=open_scope, cwd="/mnt/raid0/llm/tmp/ak-loop-lanes/w0")
    fake.add(202, scope=f"{key}.999.77.deadbeef")                      # owner is gone
    fake.add(300, start=4242)
    fake.add(203, scope=f"{key}.300.4242.cafef00d")                    # a live peer loop
    fake.add(204, cwd=s + "/scratch/author/lane0/ak-check (deleted)",  # the C84 orphan
             exe=s + "/.anchor-prune-private-x/anchor-gen-005/bin/test-backend-ops (deleted)")
    fake.add(205, cwd=s + "/scratch/author/lane0", tty=34816)         # someone's shell
    fake.add(210, ppid=300, cwd=s + "/scratch/author/lane0")          # a live peer's child
    fake.add(206, cwd=s, exe="/usr/bin/python3.13")                    # campaign, not ours
    fake.add(207, cwd="/home/node", exe="/usr/bin/vim")                # unrelated
    fake.add(208, scope=closed_scope,
             exe="/home/node/.codex/packages/app-server-daemon/releases/x/bin/codex")
    fake.add(209, scope=f"otherkey.999.77.deadbeef", cwd="/tmp")        # another campaign
    return guard, fake, ended


def test_classes_are_decided_by_provenance_not_names(tmp_path):
    guard, _fake, _ended = _synthetic(tmp_path)
    classes = {proc.pid: cls for cls, proc in guard.census()}
    assert classes == {100: "self", 50: "ancestor", 1: "ancestor", 101: "descendant",
                       200: "leak", 201: "live", 202: "stale", 203: "peer",
                       204: "orphan_scratch", 205: "campaign_foreign", 210: "campaign_foreign",
                       206: "campaign_foreign", 208: "exempt"}


def test_stale_sweep_ends_exactly_the_provably_ours_orphans(tmp_path):
    guard, _fake, ended = _synthetic(tmp_path)
    record = guard.sweep_stale("run_start")
    assert sorted(ended) == [200, 202, 204]
    assert sorted(row["pid"] for row in record["targets"]) == [200, 202, 204]


def test_measurement_flags_a_foreign_campaign_process_burning_cpu(tmp_path):
    guard, fake, ended = _synthetic(tmp_path)
    now = [0.0]
    watch = procguard.MeasurementWatch(guard, "unit", background=False,
                                       clock=lambda: now[0]).start()
    assert sorted(ended) == [200, 202, 204]                      # swept BEFORE
    for step in range(1, 4):                                      # 3 samples = 2 intervals
        now[0] = float(step)
        fake.burn(206, step * procguard.CLK_TCK)                  # one full core
        fake.burn(101, step * procguard.CLK_TCK)                  # our own server: fine
        fake.burn(201, step * procguard.CLK_TCK)                  # a live call: fine
        fake.burn(203, step * 2)                                  # 2% of a core: noise
        watch.sample()
    conditions = watch.finish()
    assert [row["pid"] for row in conditions] == [206]
    row = conditions[0]
    assert row["condition"] == procguard.CONTENTION and row["class"] == "campaign_foreign"
    assert row["observations"] >= 2 and row["max_cpu_fraction"] >= 0.9


def test_one_busy_interval_is_not_a_contradiction(tmp_path):
    guard, fake, _ended = _synthetic(tmp_path)
    now = [0.0]
    watch = procguard.MeasurementWatch(guard, "unit", background=False,
                                       clock=lambda: now[0]).start()
    now[0] = 1.0
    fake.burn(206, procguard.CLK_TCK)
    watch.sample()
    now[0] = 2.0
    watch.sample()                                                # idle again
    assert watch.finish() == []


# -- real processes: before-measurement sweep, scratch release ----------------------------

def test_measurement_watch_sweeps_a_planted_orphan_in_the_campaign_scratch(tmp_path):
    store = tmp_path / "store"
    lane = store / "scratch" / "author" / "lane0" / "ak-check"
    lane.mkdir(parents=True)
    other = store / "anchor-gen-009"
    other.mkdir()
    orphan = _plant_orphan(lane, tmp_path / "orphan.pid")
    foreign = _plant_orphan(other, tmp_path / "foreign.pid")
    guard = procguard.install(procguard.Guard(store=store, grace_s=2.0))
    try:
        watch = procguard.measurement_watch("unit", background=False)
        assert watch is not None
        assert _gone(orphan), "a no-cookie orphan inside the campaign scratch survived"
        assert not _gone(foreign, timeout_s=0.5), \
            "a campaign process outside the scratch is flagged, never killed"
        classes = {proc.pid: cls for cls, proc in guard.census()}
        assert classes.get(foreign) == "campaign_foreign"
        assert watch.finish() == []                               # asleep: no contention
    finally:
        procguard.uninstall(guard)
        _kill_own(orphan)
        _kill_own(foreign)


def test_scratch_release_ends_a_process_inside_before_deleting(tmp_path):
    reg = scratch.ScratchRegistry(tmp_path / "scratch", owner={"campaign": "t"})
    guard = procguard.install(procguard.Guard(scratch_roots=(reg.root,), grace_s=2.0))
    pid = None
    try:
        with reg.scope("iteration", "it") as scope:
            path = scope.dir("ak-check-build", "lane0")
            pid = _plant_orphan(path, tmp_path / "inside.pid")
        assert _gone(pid)
        assert not path.exists()
    finally:
        procguard.uninstall(guard)
        if pid is not None:
            _kill_own(pid)


def test_scratch_release_with_a_survivor_is_refused(tmp_path):
    reg = scratch.ScratchRegistry(tmp_path / "scratch", owner={"campaign": "t"})
    guard = procguard.install(procguard.Guard(
        scratch_roots=(reg.root,),
        terminate=lambda targets, _sel: {"terminated": [], "killed": [],
                                         "survivors": [p.to_dict() for p in targets]}))
    pid = None
    try:
        with reg.scope("iteration", "it") as scope:
            path = scope.dir("ak-check-build", "lane0")
            pid = _plant_orphan(path, tmp_path / "inside.pid")
        assert path.exists(), "a dir with a live process inside was deleted"
        events = [json.loads(line)["event"] for line in reg.journal_path.read_text().splitlines()]
        assert "release_failed" in events
    finally:
        procguard.uninstall(guard)
        if pid is not None:
            _kill_own(pid)


# -- the serving launch refuses a contended sample ----------------------------------------

def test_a_contended_cpu_launch_is_measurement_invalid(tmp_path, monkeypatch):
    from .test_legacy_cpu_serving import _requests, _server
    from .test_serving_residency import _proof, _sampler_class
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    model = tmp_path / "not-a-model"
    model.write_bytes(b"no inference is performed")
    recipe = serving.Recipe(name="cpu-guard", model=str(model), device="none", ngl=0,
                            np=2, n_predict=8, cpu_list=None)
    anchor, _log, _pids = _server(tmp_path / "anchor", recipe, port, 10.0)
    monkeypatch.setattr(serving.residency, "Sampler",
                        _sampler_class(_proof(peak=0, median=0, kfd=0)))
    planted = {"condition": procguard.CONTENTION, "class": "orphan_scratch", "pid": 930091,
               "start_ticks": 1, "cwd": "/x/store/scratch/author/lane0/ak-check",
               "exe": "/x/store/anchor-gen-005/bin/test-backend-ops", "observations": 2,
               "max_cpu_fraction": 4.8, "label": "serving:cpu-guard", "scope": None}

    class Watch:
        def finish(self):
            return [planted]
    monkeypatch.setattr(procguard, "measurement_watch", lambda label, **kw: Watch())
    with pytest.raises(MeasurementInvalid) as caught:
        serving._measure_once(recipe, Path(anchor.build_dir), port, resolved_recipe=anchor,
                              frozen_requests=_requests())
    assert procguard.CONTENTION in str(caught.value)
    assert planted in caught.value.record["failed_conditions"]


# -- fixtures: an out-of-process fake never outlives its test ----------------------------

def test_fixture_fake_exits_when_its_parent_is_gone(tmp_path):
    from .test_legacy_cpu_serving import FIXTURE_PARENT_WATCH
    script = tmp_path / "fake-server"
    script.write_text(f"#!{sys.executable}\n{FIXTURE_PARENT_WATCH}\nimport time\ntime.sleep(300)\n")
    script.chmod(0o700)
    pidfile = tmp_path / "fake.pid"
    parent = (f"import subprocess, sys, time\n"
              f"p = subprocess.Popen([{str(script)!r}], start_new_session=True)\n"
              f"open({str(pidfile)!r}, 'w').write(str(p.pid))\n"
              f"time.sleep(1.0)\n")                         # then die without waiting
    subprocess.run([sys.executable, "-c", parent], check=True, timeout=30)
    pid = _wait_for_pid(pidfile)
    try:
        assert _gone(pid), "a fixture fake outlived the process that started it"
    finally:
        _kill_own(pid)
