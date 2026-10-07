"""Local owners release between phases, including failed compute bodies."""
from contextlib import contextmanager
import fcntl
from types import SimpleNamespace

import pytest

from . import claim, gpu_phases as gp
from .test_cpu_quiet_region_backoff import native


def _quiet_module(path):
    @contextmanager
    def owner(*args, **kwargs):
        with path.open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield {"lock_path": str(path)}
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)
    return SimpleNamespace(gpu_quiet_lock=owner, GPU_QUIET_EXCLUSIVE="exclusive")


def test_compute_retains_only_real_quiet_intervals_and_releases_on_failure(tmp_path, monkeypatch,
                                                                         real_gpu_quiet):
    path = tmp_path / "gpu_quiet.lock"
    monkeypatch.setattr(claim, "_gpu_quiet_module", lambda: _quiet_module(path))
    monkeypatch.setattr(claim, "hold_gpu_quiet_measurement",
                        real_gpu_quiet["hold_gpu_quiet_measurement"])
    owner = gp.LocalPhases(cpu_list="88-91", quiet=True, should_stop=lambda: False,
                           on_wait=lambda kind: None)
    with owner.compute():
        with owner.compute():
            pass  # same owner; a nested measurement cannot self-deadlock
    with pytest.raises(ValueError, match="failed compute"):
        with owner.compute():
            raise ValueError("failed compute")
    assert len(owner.phases) == 2
    for row in owner.phases:
        component = row["component"]
        assert component["open"]["status"] == component["close"]["status"] == "held"
        assert component["released"] is True
        assert component["physical_region_fraction"] == 0.
        assert component["gpu_device_ids"] == []
    assert owner.phases[0]["component"]["ended_at"] <= owner.phases[1]["component"]["started_at"]
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_build_requests_narrow_role_without_gpu_quiet_and_records_original_release(monkeypatch):
    events = []
    @contextmanager
    def cpu_owner(cpu_list, **kwargs):
        entered = kwargs.pop("on_acquired")
        events.append(("acquire", cpu_list, kwargs))
        entered()
        class Receipt:
            def retained_interval(self):
                assert events[-1] == "release"
                return {"original": True}
        try:
            yield Receipt()
        finally:
            events.append("release")
    monkeypatch.setattr(claim, "hold_cpu", cpu_owner)
    owner = gp.LocalPhases(cpu_list="88-91", quiet=True, should_stop=lambda: False,
                           on_wait=lambda kind: None)
    with owner.build():
        with owner.build():
            events.append("compile")
    assert events == [("acquire", "88-91", {"role": "build"}), "compile", "release"]
    assert owner.phases == [{"kind": "build", "component": {"original": True}}]


def test_stopped_phase_never_enters_provider(monkeypatch):
    monkeypatch.setattr(claim, "hold_cpu", lambda *a, **k: pytest.fail("acquired after stop"))
    owner = gp.LocalPhases(cpu_list="88-91", quiet=True, should_stop=lambda: True,
                           on_wait=lambda kind: None)
    with pytest.raises(gp.loop.TailRefused):
        with owner.build():
            pytest.fail("computed after stop")


def test_missing_child_release_never_settles_as_an_idle_gap(tmp_path):
    owner = gp.LocalPhases(cpu_list="88-91", quiet=True, should_stop=lambda: False,
                           on_wait=lambda kind: None)
    owner.child_dir = tmp_path
    (tmp_path / "123.pending").write_text('{"pid":123}')
    with pytest.raises(claim.ClaimRefused, match="release receipt"):
        owner.closed_phases()


def test_failed_owner_release_never_settles_as_an_idle_gap(monkeypatch):
    @contextmanager
    def cpu_owner(*args, **kwargs):
        kwargs["on_acquired"]()
        try:
            yield {}
        finally:
            raise OSError("release unproved")
    monkeypatch.setattr(claim, "hold_cpu", cpu_owner)
    owner = gp.LocalPhases(cpu_list="88-91", quiet=True, should_stop=lambda: False,
                           on_wait=lambda kind: None)
    with pytest.raises(OSError):
        with owner.build():
            pass
    with pytest.raises(claim.ClaimRefused, match="capture failed"):
        owner.closed_phases()


def test_failed_child_release_preserves_parent_refusal_marker(tmp_path, monkeypatch):
    @contextmanager
    def cpu_owner(*args, **kwargs):
        kwargs["on_acquired"]()
        try:
            yield {}
        finally:
            raise OSError("child release unproved")
    monkeypatch.setattr(claim, "hold_cpu", cpu_owner)
    monkeypatch.setenv(gp.CHILD_PHASE_DIR, str(tmp_path))
    with pytest.raises(OSError):
        with gp.child_build("88-91"):
            pass
    assert list(tmp_path.glob("*.pending"))
    parent = gp.LocalPhases(cpu_list="88-91", quiet=True, should_stop=lambda: False,
                            on_wait=lambda kind: None)
    parent.child_dir = tmp_path
    with pytest.raises(claim.ClaimRefused, match="release receipt"):
        parent.closed_phases()


def _native_local(kind, native, monkeypatch, real_gpu_quiet):
    monkeypatch.setattr(claim, "hold_gpu_quiet_measurement",
                        real_gpu_quiet["hold_gpu_quiet_measurement"])
    local = gp.LocalPhases(cpu_list="0", quiet=True, should_stop=lambda: False,
                           on_wait=lambda kind: pytest.fail("post-entry failure retried"))
    def assert_held():
        if kind == "quiet":
            assert native[0].lock_owners() == {gp.os.getpid(): "exclusive"}
        else:
            owners = claim.observe_gpu_quiet(native[1].global_region_lock_path("q0"))["owners"]
            assert [row["pid"] for row in owners] == [gp.os.getpid()]
    def assert_released():
        assert native[0].lock_owners() == {}
        with native[1].cpu_region_lock("private-release-probe", {"q0"}, timeout_s=.1):
            pass
    return local, local.compute if kind == "quiet" else local.build, assert_held, assert_released


@pytest.mark.parametrize("kind", ["cpu", "quiet"])
@pytest.mark.parametrize("stage", ["constructor", "open"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_original_native_entry_before_observer_failure_poisoned(
        native, monkeypatch, real_gpu_quiet, kind, stage, error_type):
    local, hold, assert_held, assert_released = _native_local(
        kind, native, monkeypatch, real_gpu_quiet)
    # Earlier closed facts cannot be published across the unobserved native tail.
    with hold():
        pass
    earlier = list(local.phases)
    def fault(*args, **kwargs):
        assert_held()  # The ORIGINAL provider acquired before receipt/open failure.
        raise error_type("native observer failed")
    if stage == "constructor":
        monkeypatch.setattr(claim, "HeldCpuClaim", fault)
    else:
        monkeypatch.setattr(claim.HeldCpuClaim, "observe", fault)
    with pytest.raises(error_type, match="native observer failed"):
        with hold():
            pytest.fail("observer failure yielded")
    assert_released()
    assert local.phases == earlier
    assert local.capture_error is not None
    with pytest.raises(claim.ClaimRefused, match="capture failed"):
        local.closed_phases()
    with pytest.raises(claim.ClaimRefused, match="capture failed"):
        with hold():
            pytest.fail("poisoned local owner reacquired")


@pytest.mark.parametrize("kind", ["cpu", "quiet"])
@pytest.mark.parametrize("stage", ["close", "release_receipt", "retained", "native_exit"])
@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_baseexception_teardown_exits_original_native_owner_and_refuses_capture(
        native, monkeypatch, real_gpu_quiet, kind, stage, error_type):
    local, hold, assert_held, assert_released = _native_local(
        kind, native, monkeypatch, real_gpu_quiet)
    def fault(*args, **kwargs):
        if stage == "close":
            assert_held()
        else:
            assert_released()
        raise error_type("native teardown failed")
    if stage == "native_exit":
        module = native[0] if kind == "quiet" else native[1]
        name = "gpu_quiet_lock" if kind == "quiet" else "cpu_region_lock"
        original = getattr(module, name)
        @contextmanager
        def exiting(*args, **kwargs):
            with original(*args, **kwargs) as grant:
                yield grant
            fault()
        monkeypatch.setattr(module, name, exiting)
        # The release probe must use the unpatched original context.
        def assert_released():
            assert native[0].lock_owners() == {}
            region_owner = original if kind == "cpu" else native[1].cpu_region_lock
            with region_owner("private-release-probe", {"q0"}, timeout_s=.1):
                pass
    else:
        method = {"close": "_closing", "release_receipt": "_released_now",
                  "retained": "retained_interval"}[stage]
        monkeypatch.setattr(claim.HeldCpuClaim, method, fault)
    with pytest.raises(error_type, match="native teardown failed"):
        with hold():
            assert_held()
    assert_released()
    assert local.phases == [] and local.capture_error is not None
    with pytest.raises(claim.ClaimRefused, match="capture failed"):
        local.closed_phases()


@pytest.mark.parametrize("stage", ["constructor", "open"])
def test_original_child_native_entry_fault_retains_pending_refusal(
        native, tmp_path, monkeypatch, stage):
    monkeypatch.setenv(gp.CHILD_PHASE_DIR, str(tmp_path))
    def fault(*args, **kwargs):
        owners = claim.observe_gpu_quiet(native[1].global_region_lock_path("q0"))["owners"]
        assert [row["pid"] for row in owners] == [gp.os.getpid()]
        raise KeyboardInterrupt("original child capture failed")
    if stage == "constructor":
        monkeypatch.setattr(claim, "HeldCpuClaim", fault)
    else:
        monkeypatch.setattr(claim.HeldCpuClaim, "observe", fault)
    with pytest.raises(KeyboardInterrupt, match="original child capture failed"):
        with gp.child_build("0"):
            pytest.fail("child observer failure yielded")
    assert list(tmp_path.glob("*.pending")) and not list(tmp_path.glob("*.json"))
    parent = gp.LocalPhases(cpu_list="0", quiet=True, should_stop=lambda: False,
                            on_wait=lambda kind: None)
    parent.child_dir = tmp_path
    with pytest.raises(claim.ClaimRefused, match="release receipt"):
        parent.closed_phases()
    with native[1].cpu_region_lock("private-release-probe", {"q0"}, timeout_s=.1):
        pass


@pytest.mark.parametrize("kind", ["cpu", "quiet"])
def test_only_pre_native_busy_is_retryable(native, monkeypatch, real_gpu_quiet, kind):
    local, hold, assert_held, assert_released = _native_local(
        kind, native, monkeypatch, real_gpu_quiet)
    original = claim.hold_cpu if kind == "cpu" else claim.hold_gpu_quiet_measurement
    attempts, waits = [], []
    @contextmanager
    def busy_once(*args, **kwargs):
        attempts.append("attempt")
        if len(attempts) == 1:
            raise native[1].CpuRegionLockTimeout("pre-native busy")
        with original(*args, **kwargs) as receipt:
            yield receipt
    monkeypatch.setattr(claim, "hold_cpu" if kind == "cpu" else "hold_gpu_quiet_measurement",
                        busy_once)
    local.on_wait = waits.append
    with hold():
        assert_held()
    assert len(attempts) == 2 and len(waits) == 1
    assert local.capture_error is None and len(local.closed_phases()) == 1
    assert_released()


@pytest.mark.parametrize("kind", ["cpu", "quiet"])
def test_busy_named_failure_after_original_native_entry_cannot_retry(
        native, monkeypatch, real_gpu_quiet, kind):
    local, hold, assert_held, assert_released = _native_local(
        kind, native, monkeypatch, real_gpu_quiet)
    def fault(*args, **kwargs):
        assert_held()
        raise native[1].CpuRegionLockTimeout("post-native observer busy")
    monkeypatch.setattr(claim, "HeldCpuClaim", fault)
    with pytest.raises(native[1].CpuRegionLockTimeout, match="post-native observer busy"):
        with hold():
            pytest.fail("post-entry busy yielded")
    assert_released()
    with pytest.raises(claim.ClaimRefused, match="capture failed"):
        local.closed_phases()


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_unexpected_entry_failure_without_native_latch_is_not_zero_warrant(monkeypatch, error_type):
    @contextmanager
    def uncertain(*args, **kwargs):
        raise error_type("ownership entry uncertain")
        yield  # The provider failed before yielding; absence is not a receipt.
    monkeypatch.setattr(claim, "hold_cpu", uncertain)
    local = gp.LocalPhases(cpu_list="0", quiet=False, should_stop=lambda: False,
                           on_wait=lambda kind: pytest.fail("uncertain entry retried"))
    with pytest.raises(error_type, match="ownership entry uncertain"):
        with local.build():
            pytest.fail("uncertain owner yielded")
    with pytest.raises(claim.ClaimRefused, match="capture failed"):
        local.closed_phases()


def test_missing_native_callback_closes_provider_and_preserves_child_refusal(tmp_path, monkeypatch):
    events = []
    @contextmanager
    def missing_callback(*args, **kwargs):
        events.append("entered")
        try:
            yield {}
        finally:
            events.append("exited")
    monkeypatch.setattr(claim, "hold_cpu", missing_callback)
    monkeypatch.setenv(gp.CHILD_PHASE_DIR, str(tmp_path))
    with pytest.raises(claim.ClaimRefused, match="omitted native-entry callback"):
        with gp.child_build("0"):
            pytest.fail("missing callback yielded a phase")
    assert events == ["entered", "exited"]
    assert list(tmp_path.glob("*.pending")) and not list(tmp_path.glob("*.json"))


@pytest.mark.parametrize("route", ["ordinary", "author"])
@pytest.mark.parametrize("outcome", ["success", "error", "timeout", "cleanup_uncertain"])
def test_all_compile_routes_sweep_redirected_descendant_before_native_release(
        native, tmp_path, monkeypatch, route, outcome):
    """Real subprocess forks, native cookie cleanup and original CPU owner on both routes."""
    import ast
    import subprocess
    import sys
    from pathlib import Path
    from . import ak_check, gates, procguard, scratch
    registry = scratch.ScratchRegistry(tmp_path / "scratch",
        {"campaign": "private-compiler-cookie", "state_dir": str(tmp_path), "run_id": "compile"}, 0)
    guard = procguard.Guard(store=tmp_path, grace_s=.1)
    original_sweep, original_cpu = guard.sweep_scope, native[1].cpu_region_lock
    original_close = claim.HeldCpuClaim._closing
    events, cookies = [], []
    child_pid = [None]
    pid_file = tmp_path / "captured-background-compiler.pid"
    script = tmp_path / "fixture-cmake"
    script.write_text("#!/usr/bin/python3\n"
        "import os,subprocess,sys,time\n"
        "if '--build' in sys.argv: sys.exit(0)\n"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'], "
        "cwd=os.environ['COMPILER_OUTSIDE'],start_new_session=True, "
        "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
        "open(os.environ['COMPILER_PID_FILE'],'w').write(str(child.pid))\n"
        "mode=os.environ['COMPILER_OUTCOME']\n"
        "if mode=='timeout': time.sleep(60)\n"
        "sys.exit(2 if mode=='error' else 0)\n")
    script.chmod(0o755)
    # The forked/background descendant is outside all protected scratch paths.
    monkeypatch.setenv("COMPILER_OUTSIDE", str(tmp_path))
    monkeypatch.setenv("COMPILER_PID_FILE", str(pid_file))
    monkeypatch.setenv("COMPILER_OUTCOME", outcome)
    monkeypatch.setattr(procguard, "current", lambda: guard)
    monkeypatch.setattr(gates, "BUILD_TIMEOUT_S", .5)
    local = gp.LocalPhases(cpu_list="0", quiet=False, should_stop=lambda: False,
                           on_wait=lambda kind: pytest.fail("unexpected claim retry"))
    protected = []

    def assert_held():
        owners = claim.observe_gpu_quiet(native[1].global_region_lock_path("q0"))["owners"]
        assert [row["pid"] for row in owners] == [gp.os.getpid()]
        assert events.count("native_acquired") == 1 and "native_released" not in events

    def assert_dead():
        if child_pid[0] is not None:
            child = procguard.read_proc(Path("/proc"), child_pid[0])
            assert child is None or child.state == "Z", child

    @contextmanager
    def cpu(*args, **kwargs):
        with original_cpu(*args, **kwargs) as receipt:
            events.append("native_acquired")
            yield receipt
        assert_dead()
        events.append("native_released")

    def sweep(token):
        assert_held()
        events.append("sweep_held")
        cookies.append(token)
        if child_pid[0] is None:
            child_pid[0] = int(pid_file.read_text())
            child = procguard.read_proc(Path("/proc"), child_pid[0])
            assert child is not None and child.state != "Z"
            assert child.cwd == str(tmp_path) and child.scope == token
        for path in protected:
            marker = scratch.read_marker("dir", path)
            assert marker["release_block_token"] == token
            assert "release_blocked" in marker
        record = original_sweep(token)  # Actual captured PID/start-identity cleanup.
        assert not record["survivors"]
        assert_dead()
        events.append("descendant_verified_dead")
        if outcome == "cleanup_uncertain":
            raise OSError("injected original cleanup receipt failure")
        return record

    def closing(receipt):
        assert_held()
        assert "descendant_verified_dead" in events
        assert_dead()
        events.append("close_observed")
        return original_close(receipt)

    monkeypatch.setattr(native[1], "cpu_region_lock", cpu)
    monkeypatch.setattr(guard, "sweep_scope", sweep)
    monkeypatch.setattr(claim.HeldCpuClaim, "_closing", closing)
    scratch.install(registry)
    try:
        with registry.scope("run", name="compiled-paths") as scope:
            source = scope.dir("fixture-source", "source")
            build = scope.dir("ak-check-build", "build")
            protected[:] = [build] if route == "author" else [source, build]
            if route == "author":
                phase_dir = tmp_path / "child-phases"
                phase_dir.mkdir()
                monkeypatch.setenv(gp.CHILD_PHASE_DIR, str(phase_dir))
                def invoke():
                    with gp.child_build("0", paths=(build,)):
                        return ak_check.run_bounded([str(script)], cpus=tuple(sorted(gp.os.sched_getaffinity(0)))[:1],
                            timeout_s=.5, cwd=tmp_path)
            else:
                tree = ast.parse((Path(__file__).parent / "run.py").read_text())
                main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
                forwarder = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == "local_compiles")
                from contextlib import nullcontext
                namespace = {"gates": gates, "gpu_local": local, "nullcontext": nullcontext}
                exec(compile(ast.Module(body=[forwarder], type_ignores=[]), "actual-local-compile", "exec"), namespace)
                def invoke():
                    return namespace["local_compiles"](source, build, cmake_defines=(), jobs=1,
                        cpu_list=None, cmake=str(script), targets=("original-target",),
                        env=dict(gp.os.environ))
            if outcome == "cleanup_uncertain":
                with pytest.raises(scratch.ScratchRefused, match="cleanup uncertain"):
                    invoke()
            elif outcome == "timeout" and route == "ordinary":
                with pytest.raises(subprocess.TimeoutExpired):
                    invoke()
            else:
                result = invoke()
                if route == "ordinary":
                    assert result.passed == (outcome == "success")
                else:
                    assert result[2] == (outcome == "timeout")
                    assert (result[0] == 0) == (outcome == "success")
            assert cookies and len(set(cookies)) == 1
            assert events[0] == "native_acquired"
            assert events[-2:] == ["close_observed", "native_released"]
            assert_dead()
            if route == "author":
                if outcome == "cleanup_uncertain":
                    assert list(phase_dir.glob("*.pending")) and not list(phase_dir.glob("*.json"))
                else:
                    assert not list(phase_dir.glob("*.pending")) and len(list(phase_dir.glob("*.json"))) == 1
            elif outcome == "cleanup_uncertain":
                with pytest.raises(claim.ClaimRefused, match="capture failed"):
                    local.closed_phases()
            else:
                assert len(local.closed_phases()) == 1
            if outcome == "cleanup_uncertain":
                for path in protected:
                    assert "release_blocked" in scratch.read_marker("dir", path)
                # A stale parent retention write cannot erase the child's disk fence.
                scope.retain("parent handles uncertain compiler cleanup")
        if outcome == "cleanup_uncertain":
            assert all(path.exists() and "release_blocked" in scratch.read_marker("dir", path)
                       for path in protected)
            later = scratch.ScratchRegistry(tmp_path / "scratch",
                {"campaign": "private-compiler-cookie", "state_dir": str(tmp_path), "run_id": "later"}, 0)
            try:
                assert later.sweep()["removed"] == []
            finally:
                later.close()
        else:
            assert all(not path.exists() for path in protected)
        with original_cpu("private-release-probe", {"q0"}, timeout_s=.1):
            pass
    finally:
        scratch.uninstall(registry)
        registry.close()
