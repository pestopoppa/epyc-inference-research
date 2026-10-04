"""The CPU/GPU measurement quiet window on the orchestrator's gpu-quiet lock.

DEFAULT ON since 2026-10-04 (operator ruling: GPU benches and CPU measurements never
overlap). `--cpu-measurement-gpu-quiet lock`: a CPU run holds gpu-quiet SHARED with its
region claim (`claim.hold_cpu(..., gpu_quiet=True)`, yielded with it by the CPU window);
a GPU measurement holds it EXCLUSIVE for its body only and takes no CPU region.
`off` is the explicit opt-out; `q3` is a deprecated spelling of `lock`.

Every lock here is a temp file: the package conftest points ORCHESTRATOR_TMP_DIR and
the device lock into the test's tmp dir."""
import argparse
from contextlib import contextmanager
import fcntl
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

from . import claim, loop, run


# ------------------------------------------------------------------ policy and CLI

def test_default_policy_is_lock_and_off_is_the_opt_out():
    assert run.CPU_MEASUREMENT_GPU_QUIET_DEFAULT == run.CPU_MEASUREMENT_GPU_QUIET_LOCK == "lock"
    assert run.CPU_MEASUREMENT_GPU_QUIET_POLICIES == ("off", "lock")


def test_q3_is_a_deprecated_spelling_of_lock(capsys):
    assert run._gpu_quiet_policy("q3") == "lock"
    assert "deprecated" in capsys.readouterr().err
    assert run._gpu_quiet_policy("off") == "off"
    assert run._gpu_quiet_policy("lock") == "lock"
    assert capsys.readouterr().err == ""
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=run._gpu_quiet_policy,
                        choices=run.CPU_MEASUREMENT_GPU_QUIET_POLICIES)
    assert parser.parse_args(["--q", "q3"]).q == "lock"
    with pytest.raises(SystemExit):
        parser.parse_args(["--q", "on"])


def test_cli_declares_gpu_quiet_policy_default_lock(capsys):
    with pytest.raises(SystemExit) as exited:
        run.main(["--help"])
    assert exited.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    assert "--cpu-measurement-gpu-quiet {off,lock}" in text
    tail = text.split("--cpu-measurement-gpu-quiet {off,lock}")[-1][:900]
    assert "(default: lock)" in tail and "deprecated" in tail
    body = (Path(__file__).parent / "run.py").read_text(encoding="utf-8")
    assert re.search(r'"--cpu-measurement-gpu-quiet",\n\s+type=_gpu_quiet_policy,', body)


def test_gpu_preflight_runs_at_startup_and_refuses_without_the_owner(monkeypatch, real_gpu_quiet):
    """The GPU half is proven at STARTUP (claim.gpu_quiet_preflight behind parser.error),
    not discovered at the first measurement after a build."""
    monkeypatch.setattr(claim, "_ensure_orchestrator_importable", lambda: None)
    monkeypatch.setitem(sys.modules, "src.runtime.gpu_quiet_lock", None)
    with pytest.raises(claim.ClaimRefused, match="no gpu-quiet lock"):
        real_gpu_quiet["gpu_quiet_preflight"]()
    body = (Path(__file__).parent / "run.py").read_text(encoding="utf-8")
    assert re.search(r"if \(direct_launch and not cpu_launch\n\s+and args\.cpu_measurement_gpu_quiet "
                     r"!= CPU_MEASUREMENT_GPU_QUIET_OFF\):\n(?:.*\n){1,6}?\s+claim\.gpu_quiet_preflight\(\)"
                     r"\n\s+except \(claim\.ClaimRefused, ImportError, OSError\) as exc:\n\s+parser\.error",
                     body)


def test_run_wires_shared_gpu_quiet_into_cpu_claims_only():
    """A CPU run's region claim carries gpu-quiet SHARED; a GPU run's host-CPU claim must
    not (its measurements take it EXCLUSIVE -- a shared hold of its own would
    self-deadlock). The q3-honouring marker and the device flock window are gone."""
    body = (Path(__file__).parent / "run.py").read_text(encoding="utf-8")
    assert re.search(r'gpu_quiet_kw = \(\{"gpu_quiet": True\} if cpu_launch and '
                     r'args\.cpu_measurement_gpu_quiet\n\s+!= CPU_MEASUREMENT_GPU_QUIET_OFF else \{\}\)',
                     body)
    assert body.count("claim.hold_cpu(cpu_list, **gpu_quiet_kw)") == 2
    for gone in ("q3_honouring_marker", "_q3_cpu_gpu_quiet_window", "hold_q3_measurement",
                 "device_holder_honours_q3"):
        assert gone not in body
        assert not hasattr(claim, gone)


# ------------------------------------------------------------------ GPU half: the window

def _busy():
    class CpuRegionLockTimeout(RuntimeError):
        pass
    return CpuRegionLockTimeout("gpu-quiet busy")


def test_gpu_measurement_holds_gpu_quiet_for_its_body_only():
    events = []

    @contextmanager
    def hold():
        events.append("acquire")
        yield {"gpu_quiet": "exclusive"}
        events.append("release")

    with run._gpu_quiet_measurement_window(True, on_wait=lambda: pytest.fail("wait"),
                                           should_stop=lambda: False, hold=hold):
        events.append("measure")
    assert events == ["acquire", "measure", "release"]


def test_default_hold_is_the_exclusive_gpu_quiet_measurement(gpu_quiet_holds):
    with run._gpu_quiet_measurement_window(True, on_wait=lambda: pytest.fail("wait"),
                                           should_stop=lambda: False):
        assert gpu_quiet_holds == ["enter"]
    assert gpu_quiet_holds == ["enter", "exit"]


def test_gpu_measurement_waits_on_busy_lock_then_refuses_on_stop():
    notices = []

    def hold():
        raise _busy()

    with pytest.raises(loop.TailRefused, match="gpu-quiet lock"):
        with run._gpu_quiet_measurement_window(True, on_wait=lambda: notices.append("w"),
                                               should_stop=lambda: bool(notices), hold=hold):
            pytest.fail("measured without gpu-quiet")
    assert notices == ["w"]


def test_gpu_measurement_retries_busy_lock_until_granted(monkeypatch):
    monkeypatch.setattr(run.time, "sleep", lambda _s: None)
    attempts = []

    @contextmanager
    def granted():
        yield {}

    def hold():
        attempts.append(1)
        if len(attempts) < 3:
            raise _busy()
        return granted()

    with run._gpu_quiet_measurement_window(True, on_wait=lambda: None,
                                           should_stop=lambda: False, hold=hold):
        pass
    assert len(attempts) == 3


def test_gpu_measurement_window_noops_when_off_or_for_a_cpu_run():
    def hold():
        pytest.fail("gpu-quiet taken")
    with run._gpu_quiet_measurement_window(True, on_wait=lambda: None, should_stop=lambda: False,
                                           hold=hold, policy=run.CPU_MEASUREMENT_GPU_QUIET_OFF):
        pass
    with run._gpu_quiet_measurement_window(False, on_wait=lambda: None,
                                           should_stop=lambda: False, hold=hold):
        pass


def test_gpu_measurement_propagates_a_non_busy_refusal():
    def hold():
        raise claim.ClaimRefused("preflight failed")
    with pytest.raises(claim.ClaimRefused, match="preflight"):
        with run._gpu_quiet_measurement_window(True, on_wait=lambda: None,
                                               should_stop=lambda: False, hold=hold):
            pass


def test_unknown_gpu_quiet_policy_is_refused():
    with pytest.raises(ValueError, match="quiet policy"):
        with run._gpu_quiet_measurement_window(True, on_wait=lambda: None,
                                               should_stop=lambda: False, policy="q3"):
            pass


def test_nested_measurement_window_enters_the_quiet_window_once():
    depth = run.threading.local()
    entered = []

    @contextmanager
    def window():
        entered.append("in")
        yield
        entered.append("out")

    with run._reentrant(depth, window()):
        with run._reentrant(depth, window()):
            pass
    assert entered == ["in", "out"]


def test_invalid_arm_reschedule_reacquires_measurement_window(monkeypatch):
    events = []

    @contextmanager
    def window():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    invalid = loop.MeasurementInvalid("invalid arm", {"arm": "candidate"})
    invalid.reschedule = lambda: {"continued": True}
    monkeypatch.setattr(run, "ServingComparison", lambda row, scope: (row, scope))

    def initial():
        raise invalid

    with pytest.raises(loop.MeasurementInvalid) as caught:
        run._serving_comparison(initial, "fixture", measurement_window=window)
    assert events == ["enter", "exit"]
    assert caught.value.reschedule() == ({"continued": True}, "fixture")
    assert events == ["enter", "exit", "enter", "exit"]


# ------------------------------------------------------------------ CPU half: hold_cpu

@pytest.fixture
def fake_owner(tmp_path, monkeypatch):
    """A fake orchestrator: real temp flocks for the regions, a recording gpu-quiet."""
    modules = {name: ModuleType(name) for name in (
        "src.runtime.cpu_region_lock", "src.runtime.instance_topology",
        "src.runtime.region_lock_cli", "src.runtime.gpu_quiet_lock")}
    paths = [tmp_path / "role.lock", tmp_path / "global.lock"]
    events = []

    @contextmanager
    def regions(*args, **kwargs):
        events.append("regions")
        handles = [path.open("a") for path in paths]
        try:
            for handle in handles:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield {"fixture-region": paths[0]}
        finally:
            for handle in handles:
                handle.close()
            events.append("regions released")

    @contextmanager
    def gpu_quiet_then_regions(role, regs, *, gpu_quiet, timeout_s, cancel_check=None,
                               request_tag=None):
        events.append(f"gpu-quiet {gpu_quiet}")
        try:
            with regions() as held:
                yield {"regions": held, "gpu_quiet": {"mode": gpu_quiet}}
        finally:
            events.append("gpu-quiet released")

    cpu = modules["src.runtime.cpu_region_lock"]
    cpu.cpu_region_lock = regions
    cpu.global_region_lock_path = lambda region: paths[1]
    modules["src.runtime.instance_topology"].cpu_list_to_regions = lambda cpus: ["fixture-region"]
    modules["src.runtime.instance_topology"].ATOMIC_REGIONS = ("fixture-region",)
    modules["src.runtime.region_lock_cli"]._preflight = lambda **kwargs: None
    quiet = modules["src.runtime.gpu_quiet_lock"]
    quiet.GPU_QUIET_SHARED, quiet.GPU_QUIET_EXCLUSIVE = "shared", "exclusive"
    quiet.gpu_quiet_then_regions = gpu_quiet_then_regions
    quiet.gpu_quiet_lock_path = lambda: tmp_path / "gpu_quiet.lock"
    src = ModuleType("src")
    runtime = ModuleType("src.runtime")
    runtime.gpu_quiet_lock = quiet
    monkeypatch.setitem(sys.modules, "src", src)
    monkeypatch.setitem(sys.modules, "src.runtime", runtime)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(claim, "_ensure_orchestrator_importable", lambda: None)
    monkeypatch.setenv("ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT", "1")
    return events


def test_cpu_claim_takes_gpu_quiet_shared_before_its_regions(fake_owner, tmp_path):
    with claim.hold_cpu("0", gpu_quiet=True) as receipt:
        assert fake_owner == ["gpu-quiet shared", "regions"]
        assert receipt["gpu_quiet"] == {"mode": "shared",
                                        "lock_path": str(tmp_path / "gpu_quiet.lock")}
        assert receipt["regions"] == ["fixture-region"]
        assert receipt.observe()["status"] == "held"
    assert fake_owner[-2:] == ["regions released", "gpu-quiet released"]


def test_cpu_claim_without_gpu_quiet_is_unchanged(fake_owner):
    with claim.hold_cpu("0") as receipt:
        assert "gpu_quiet" not in receipt
    assert fake_owner == ["regions", "regions released"]


def test_yield_lease_releases_and_retakes_gpu_quiet_with_the_regions(fake_owner):
    """Actor phases free gpu-quiet for a GPU bench; every re-acquire takes it first."""
    with claim.hold_cpu("0", gpu_quiet=True) as receipt:
        lease = claim.yield_lease(receipt)
        assert lease.release(reason="actor:planner")
        assert fake_owner[-2:] == ["regions released", "gpu-quiet released"]
        lease.reacquire(reason="measurement", poll_s=0.01)
        assert fake_owner[-2:] == ["gpu-quiet shared", "regions"]
        assert lease.generation == 2
    assert fake_owner[-2:] == ["regions released", "gpu-quiet released"]


def test_cpu_claim_refuses_when_the_owner_has_no_gpu_quiet(fake_owner, monkeypatch):
    monkeypatch.setitem(sys.modules, "src.runtime.gpu_quiet_lock", None)
    monkeypatch.delattr(sys.modules["src.runtime"], "gpu_quiet_lock")
    with pytest.raises(claim.ClaimRefused, match="no gpu-quiet lock"):
        with claim.hold_cpu("0", gpu_quiet=True):
            pytest.fail("claimed without gpu-quiet")


# ------------------------------------------------------------------ the real owner

def _real_owner():
    try:
        return claim._gpu_quiet_module()
    except claim.ClaimRefused as exc:
        pytest.skip(f"resolved orchestrator has no gpu-quiet owner yet: {exc}")


def test_real_exclusive_measurement_waits_for_a_shared_cpu_holder(real_gpu_quiet):
    """Against the orchestrator's own module, on the conftest's temp lock dir."""
    import os
    quiet = _real_owner()
    lock = quiet.gpu_quiet_lock_path()
    assert lock.parent == Path(os.environ["ORCHESTRATOR_TMP_DIR"])  # never the host's
    hold = real_gpu_quiet["hold_gpu_quiet_measurement"]
    with quiet.gpu_quiet_lock("shared", role="autokernel-cpu", request_tag="t"):
        with pytest.raises(Exception) as busy:
            with hold(timeout_s=0.1):
                pytest.fail("exclusive granted over a shared CPU holder")
        assert claim.region_lock_busy(busy.value)
    with hold(timeout_s=1.0) as record:
        assert record["gpu_quiet"] == "exclusive" and record["lock_path"] == str(lock)
        state = quiet.holders()
        assert state["mode"] == "exclusive"
        assert [h["role"] for h in state["holders"]] == [claim.GPU_QUIET_ROLE]
    assert quiet.holders()["held"] is False
