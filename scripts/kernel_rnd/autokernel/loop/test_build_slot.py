"""AKU-20/21: build slot (pinned, bounded jobs, role `build`, per-lane cap); CPU arm role `bench`."""
import inspect
import time

import pytest

from . import build_slot, claim, gates


def test_bound_jobs_clamps_to_pinned_cpus_and_refuses_unpinned():
    assert build_slot.cpu_count("0-3,8") == 5
    assert build_slot.bound_jobs(64, "0-3") == 4
    assert build_slot.bound_jobs(2, "0-3") == 2
    for bad in ("", None, "x", "5-2"):
        with pytest.raises(build_slot.BuildSlotRefused):
            build_slot.bound_jobs(1, bad)
    for bad in (0, -1, True, 1.5):
        with pytest.raises(build_slot.BuildSlotRefused):
            build_slot.bound_jobs(bad, "0-3")


def test_compiles_bounds_jobs_and_pins(tmp_path, monkeypatch):
    argv = []

    class Done:
        returncode, stdout, stderr = 0, "", ""

    monkeypatch.setattr(gates.subprocess, "run", lambda cmd, **kw: argv.append(cmd) or Done())
    monkeypatch.setattr(gates, "_clear_stale_build_dir", lambda *a, **k: False)
    assert gates.compiles(tmp_path, tmp_path / "b", cmake_defines=(), jobs=64, cpu_list="0-3",
                          targets=("t",)).passed
    assert all(cmd[:3] == ["taskset", "-c", "0-3"] or cmd[:3] == ("taskset", "-c", "0-3")
               for cmd in argv)
    assert argv[-1][argv[-1].index("-j") + 1] == "4"


def test_lane_cap_refuses_fast_and_lanes_are_independent(tmp_path):
    with build_slot.lane_slot("a", 1, directory=tmp_path):
        started = time.monotonic()
        with pytest.raises(build_slot.BuildSlotBusy):
            with build_slot.lane_slot("a", 1, directory=tmp_path):
                pytest.fail("cap of 1 admitted a second build in one lane")
        assert time.monotonic() - started < 0.5
        with build_slot.lane_slot("b", 1, directory=tmp_path):
            pass
    with build_slot.lane_slot("a", 2, directory=tmp_path) as first:
        with build_slot.lane_slot("a", 2, directory=tmp_path) as second:
            assert {first, second} == {0, 1}
            with pytest.raises(build_slot.BuildSlotBusy):
                with build_slot.lane_slot("a", 2, directory=tmp_path):
                    pass
    with build_slot.lane_slot("a", 1, directory=tmp_path):  # released on exit
        pass


def test_busy_lane_is_retryable_like_a_busy_region():
    assert claim.region_lock_busy(build_slot.BuildSlotBusy("x"))
    assert not claim.region_lock_busy(build_slot.BuildSlotRefused("x"))


def test_cpu_arm_defaults_to_bench_role():
    assert claim.CPU_ARM_ROLE == "bench"
    assert inspect.signature(claim.hold_cpu).parameters["role"].default == "bench"


@pytest.fixture
def native(tmp_path, monkeypatch):
    claim._ensure_orchestrator_importable()
    import importlib
    cli = importlib.import_module("src.runtime.region_lock_cli")
    monkeypatch.setenv("ORCHESTRATOR_TMP_DIR", str(tmp_path / "private-native-locks"))
    monkeypatch.setenv("ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT", "1")
    monkeypatch.setenv("EPYC_LOCK_FIFO", "1")
    monkeypatch.setenv("AK_BUILD_SLOT_DIR", str(tmp_path / "slots"))
    monkeypatch.setattr(cli, "_preflight", lambda **kwargs: None)
    return importlib.import_module("src.runtime.cpu_region_lock")


def test_build_claim_inside_bench_grant_on_disjoint_cores_works(native):
    with claim.hold_cpu("24-47") as bench:  # q1, default role bench
        assert set(bench["regions"]) == {"q1"}
        with build_slot.build_slot("48-71", 64, lane="t") as build:  # q2
            assert set(build["regions"]) == {"q2"}
            assert build_slot.plan("48-71", 64)["jobs"] == 24
            assert any("build" in path for path in build["lock_paths"].values())
        assert bench.observe()["status"] == "held"


def test_build_claim_overlapping_a_bench_grant_is_refused_fast(native):
    with claim.hold_cpu("24-47"):
        started = time.monotonic()
        with pytest.raises(Exception) as refused:
            with build_slot.build_slot("24-47", 8, lane="t"):
                pytest.fail("a build took a region its bench ancestor holds")
        assert time.monotonic() - started < 5.0
        assert type(refused.value).__name__ in {"NestedLockError", "CpuRegionLockTimeout"}


def test_yielded_bench_grant_lets_a_child_build_on_the_same_cores(native):
    with claim.hold_cpu("24-47") as bench:
        lease = claim.yield_lease(bench)
        lease.release(reason="spawn child")
        with build_slot.build_slot("24-47", 8, lane="t"):
            pass
        lease.reacquire(reason="child done")
        assert bench.observe()["status"] == "held"
