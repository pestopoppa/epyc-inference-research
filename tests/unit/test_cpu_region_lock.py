"""Unit tests for per-CPU-region locking (`src.runtime.cpu_region_lock`)
and the instance topology table (`src.runtime.instance_topology`).

Covers:
- Atomic-region derivation from cpu_list strings (NUMA_CONFIG-style)
- build_instance_regions on synthetic configs
- Overlap predicate (instances_overlap)
- Single-region lock acquire/release
- Multi-region lock acquire/release (full instance)
- Non-overlapping quarters acquire concurrently
- Overlapping instances block each other (with deadline timeout)
- All-or-nothing rollback on partial acquire failure
- Cross-process behavior via multiprocessing
- Convenience wrapper cpu_region_lock_for_instance
"""

from __future__ import annotations

import multiprocessing
import os
import threading
import time
from pathlib import Path

import pytest


# Redirect lock files to a per-test tmpdir BEFORE importing the module
@pytest.fixture(autouse=True)
def _lock_tmpdir(tmp_path, monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_TMP_DIR", str(tmp_path))
    # Force the config-aware path to be the env override even if src.config
    # is importable in this environment (avoids leaking lock files into
    # the production /mnt/raid0/llm/tmp during tests).
    monkeypatch.setenv("ORCHESTRATOR_INFERENCE_LOCK_POLL_MS", "10")
    yield


from src.runtime.instance_topology import (  # noqa: E402
    ATOMIC_REGIONS,
    REGION_CORE_RANGE,
    build_instance_regions,
    cores_to_regions,
    cpu_list_to_regions,
    instances_overlap,
    parse_cpu_list,
)
from src.runtime.cpu_region_lock import (  # noqa: E402
    CpuRegionLockTimeout,
    cpu_region_lock,
    cpu_region_lock_for_instance,
    read_region_occupancy,
    read_region_lock_payload,
    region_lock_path,
    sweep_stale_region_lock_payloads,
)
import src.runtime.cpu_region_lock as cpu_region_lock_module  # noqa: E402


# ───────────────────────────── topology tests ─────────────────────────────


class TestTopology:
    def test_atomic_regions_partition(self):
        """The four atomic regions must cover all 96 physical cores exactly."""
        covered = set()
        for region, (lo, hi) in REGION_CORE_RANGE.items():
            for c in range(lo, hi + 1):
                assert c not in covered, f"core {c} covered twice"
                covered.add(c)
        assert covered == set(range(96))
        assert set(REGION_CORE_RANGE.keys()) == set(ATOMIC_REGIONS)

    def test_parse_cpu_list_basic(self):
        assert parse_cpu_list("0-23,96-119") == set(range(24))  # HT range dropped
        assert parse_cpu_list("48-71") == set(range(48, 72))
        assert parse_cpu_list("0-95") == set(range(96))
        assert parse_cpu_list("") == set()
        assert parse_cpu_list("5") == {5}
        assert parse_cpu_list("5,10,15") == {5, 10, 15}
        assert parse_cpu_list(" 0-3 , 7-8 ") == {0, 1, 2, 3, 7, 8}

    def test_parse_cpu_list_drops_invalid(self):
        # Out-of-range and non-numeric segments are silently dropped
        assert parse_cpu_list("foo,5,bar") == {5}
        assert parse_cpu_list("96-100,5") == {5}  # 96-100 are HTs, dropped
        assert parse_cpu_list("-,5") == {5}

    def test_cores_to_regions(self):
        assert cores_to_regions({0}) == {"q0"}
        assert cores_to_regions({23}) == {"q0"}
        assert cores_to_regions({24}) == {"q1"}
        assert cores_to_regions(range(48)) == {"q0", "q1"}
        assert cores_to_regions(range(96)) == {"q0", "q1", "q2", "q3"}
        assert cores_to_regions(set()) == frozenset()

    def test_cpu_list_to_regions(self):
        # Equivalent to NUMA_NODE0
        assert cpu_list_to_regions("0-47,96-143") == {"q0", "q1"}
        # NUMA_FULL = entire machine
        assert cpu_list_to_regions("0-95") == {"q0", "q1", "q2", "q3"}
        # NUMA_Q0A
        assert cpu_list_to_regions("0-23,96-119") == {"q0"}
        # NUMA_Q1B
        assert cpu_list_to_regions("72-95,168-191") == {"q3"}

    def test_build_instance_regions_synthetic(self):
        cfg = {
            "frontdoor": {
                "instances": [
                    ("0-47,96-143", 8070, 96),  # full → q0,q1
                    ("0-23,96-119", 8080, 48),  # q0
                    ("24-47,120-143", 8180, 48),  # q1
                ],
            },
            "worker_general": {
                "instances": [
                    ("0-95", 8072, 96),  # full → all
                    ("0-23,96-119", 8082, 48),  # q0
                ],
            },
            "architect_general": {
                "instances": [
                    ("0-95", 8083, 96),  # full → all
                ],
            },
        }
        out = build_instance_regions(cfg)
        assert out[("frontdoor", 0)] == {"q0", "q1"}
        assert out[("frontdoor", 1)] == {"q0"}
        assert out[("frontdoor", 2)] == {"q1"}
        assert out[("worker_general", 0)] == {"q0", "q1", "q2", "q3"}
        assert out[("worker_general", 1)] == {"q0"}
        assert out[("architect_general", 0)] == {"q0", "q1", "q2", "q3"}

    def test_instances_overlap(self):
        cfg = {
            "frontdoor": {
                "instances": [
                    ("0-47,96-143", 8070, 96),  # full
                    ("0-23,96-119", 8080, 48),  # q0
                    ("24-47,120-143", 8180, 48),  # q1
                    ("48-71,144-167", 8280, 48),  # q2
                    ("72-95,168-191", 8380, 48),  # q3
                ],
            },
        }
        r = build_instance_regions(cfg)

        # Full overlaps with q0 and q1 (it spans 0-47)
        assert instances_overlap(r, ("frontdoor", 0), ("frontdoor", 1)) is True
        assert instances_overlap(r, ("frontdoor", 0), ("frontdoor", 2)) is True

        # Full does NOT overlap with q2 and q3 (it's only NUMA_NODE0)
        assert instances_overlap(r, ("frontdoor", 0), ("frontdoor", 3)) is False
        assert instances_overlap(r, ("frontdoor", 0), ("frontdoor", 4)) is False

        # q0 and q2 are disjoint
        assert instances_overlap(r, ("frontdoor", 1), ("frontdoor", 3)) is False

        # q0 and q1 are disjoint
        assert instances_overlap(r, ("frontdoor", 1), ("frontdoor", 2)) is False

    def test_unknown_instance_treated_as_no_conflict(self):
        r = build_instance_regions({})
        assert instances_overlap(r, ("x", 0), ("y", 0)) is False


# ─────────────────────────── lock primitive tests ──────────────────────────


class TestCpuRegionLockBasic:
    def test_region_lock_path_uses_tmp_dir(self, tmp_path):
        p = region_lock_path("frontdoor", "q0")
        assert p.parent == tmp_path
        assert p.name == "cpu_region.frontdoor.q0.lock"

    def test_acquire_single_region(self, tmp_path):
        with cpu_region_lock("frontdoor", {"q0"}) as paths:
            assert "q0" in paths
            assert paths["q0"].exists()
            assert paths["q0"].name == "cpu_region.frontdoor.q0.lock"
        # File persists after release (zero-content; that's fine)
        assert (tmp_path / "cpu_region.frontdoor.q0.lock").exists()

    def test_acquire_multiple_regions(self):
        with cpu_region_lock("frontdoor", {"q0", "q1"}) as paths:
            assert set(paths.keys()) == {"q0", "q1"}

    def test_lock_payload_written_while_held_and_cleared_on_release(self):
        with cpu_region_lock(
            "frontdoor",
            {"q1", "q0"},
            instance_idx=0,
            request_tag="unit-request",
        ) as paths:
            payload = read_region_lock_payload(paths["q0"])
            assert payload is not None
            assert payload["schema_version"] == 1
            assert payload["pid"] == os.getpid()
            assert payload["role"] == "frontdoor"
            assert payload["region"] == "q0"
            assert payload["regions"] == ["q0", "q1"]
            assert payload["instance_idx"] == 0
            assert payload["request_tag"] == "unit-request"
            assert isinstance(payload["started_at"], float)

        assert read_region_lock_payload(paths["q0"]) is None

    def test_startup_sweep_clears_unlocked_stale_payload(self):
        path = region_lock_path("worker_general", "q2")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"pid": 839651, "region": "q2"}\n', encoding="utf-8")

        assert sweep_stale_region_lock_payloads() == 1
        assert path.exists()
        assert path.read_text(encoding="utf-8") == ""

    def test_startup_sweep_never_clears_payload_when_flock_is_held(self, monkeypatch):
        path = region_lock_path("worker_general", "q3")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = '{"pid": 839651, "region": "q3"}\n'
        path.write_text(payload, encoding="utf-8")
        monkeypatch.setattr(cpu_region_lock_module, "_try_flock", lambda *_args: False)

        assert sweep_stale_region_lock_payloads() == 0
        assert path.read_text(encoding="utf-8") == payload

    def test_empty_region_set_is_noop(self):
        with cpu_region_lock("frontdoor", set()) as paths:
            assert paths == {}

    def test_path_traversal_sanitization(self):
        # Slashes in role/region get neutered to prevent escaping tmp_dir
        with cpu_region_lock("foo/bar", {"a/b"}) as paths:
            for p in paths.values():
                assert ".." not in str(p)


class TestCpuRegionLockExclusion:
    def test_non_overlapping_quarters_acquire_concurrently(self):
        """q0 + q2 are disjoint — both should acquire without blocking."""
        acquired = []

        def take_q2():
            with cpu_region_lock("frontdoor", {"q2"}, timeout_s=2):
                acquired.append("q2")
                time.sleep(0.1)

        with cpu_region_lock("frontdoor", {"q0"}, timeout_s=2):
            acquired.append("q0")
            t = threading.Thread(target=take_q2)
            t.start()
            t.join(timeout=2)
            assert not t.is_alive(), "q2 should have acquired immediately"

        assert "q2" in acquired
        assert "q0" in acquired

    def test_overlapping_instances_block_each_other(self):
        """A 'full' instance (q0+q1) blocks a 'quarter' (q0). The quarter
        should time out without acquiring."""
        with cpu_region_lock("frontdoor", {"q0", "q1"}, timeout_s=5):
            with pytest.raises(CpuRegionLockTimeout):
                with cpu_region_lock("frontdoor", {"q0"}, timeout_s=0.2):
                    pytest.fail("should not have acquired")

    def test_same_region_serializes(self):
        """Two attempts to acquire the same region — second blocks until
        the first releases."""
        order = []
        ready = threading.Event()

        def second_acquirer():
            ready.wait()
            with cpu_region_lock("frontdoor", {"q0"}, timeout_s=5):
                order.append("second")

        t = threading.Thread(target=second_acquirer)

        with cpu_region_lock("frontdoor", {"q0"}, timeout_s=5):
            order.append("first")
            t.start()
            ready.set()
            time.sleep(0.2)  # let second_acquirer start blocking
            assert order == ["first"], "second must still be blocked"

        t.join(timeout=2)
        assert order == ["first", "second"]


class TestCpuRegionLockSafety:
    def test_all_or_nothing_on_timeout(self):
        """If we try to acquire {q0, q1} but q1 is held, we must release
        the q0 lock before raising — otherwise no one can ever acquire q0
        again."""
        # Hold q1 in a thread
        block_q1_ready = threading.Event()
        release_q1 = threading.Event()

        def hold_q1():
            with cpu_region_lock("frontdoor", {"q1"}, timeout_s=10):
                block_q1_ready.set()
                release_q1.wait(timeout=5)

        t = threading.Thread(target=hold_q1)
        t.start()
        try:
            block_q1_ready.wait(timeout=2)
            # Try {q0, q1} with a short timeout — q0 will acquire, q1 won't
            with pytest.raises(CpuRegionLockTimeout):
                with cpu_region_lock("frontdoor", {"q0", "q1"}, timeout_s=0.3):
                    pytest.fail("should not have acquired both")

            # After the timeout, q0 must be free again
            acquired_again = []
            with cpu_region_lock("frontdoor", {"q0"}, timeout_s=2):
                acquired_again.append("q0")
            assert acquired_again == ["q0"]
        finally:
            release_q1.set()
            t.join(timeout=5)

    def test_release_on_exception_in_body(self):
        """Exception inside the with-body must still release the lock."""
        with pytest.raises(RuntimeError, match="body fail"):
            with cpu_region_lock("frontdoor", {"q0"}, timeout_s=2):
                raise RuntimeError("body fail")

        # Lock should be releasable again
        with cpu_region_lock("frontdoor", {"q0"}, timeout_s=2):
            pass

    def test_cancel_check_aborts(self):
        """A cancel_check returning True should abort the acquire."""
        cancelled = [False]

        def cancel_after(deadline):
            def check():
                if time.perf_counter() > deadline:
                    cancelled[0] = True
                    return True
                return False

            return check

        # Hold q0 in a thread
        block_ready = threading.Event()
        release = threading.Event()

        def holder():
            with cpu_region_lock("frontdoor", {"q0"}, timeout_s=10):
                block_ready.set()
                release.wait(timeout=5)

        t = threading.Thread(target=holder)
        t.start()
        try:
            block_ready.wait(timeout=2)
            deadline = time.perf_counter() + 0.3
            with pytest.raises(CpuRegionLockTimeout, match="cancelled"):
                with cpu_region_lock(
                    "frontdoor",
                    {"q0"},
                    timeout_s=10,
                    cancel_check=cancel_after(deadline),
                ):
                    pytest.fail("should not have acquired")
            assert cancelled[0] is True
        finally:
            release.set()
            t.join(timeout=5)

    def test_lock_order_prevents_deadlock(self):
        """Two threads acquiring {q0,q1} from different orders must not
        deadlock. The lock module sorts internally, so both threads
        take q0 first."""
        acquired = []

        def thread_a():
            with cpu_region_lock("frontdoor", {"q1", "q0"}, timeout_s=5):
                acquired.append("A")
                time.sleep(0.05)

        def thread_b():
            with cpu_region_lock("frontdoor", {"q0", "q1"}, timeout_s=5):
                acquired.append("B")
                time.sleep(0.05)

        threads = [threading.Thread(target=thread_a) for _ in range(3)] + [
            threading.Thread(target=thread_b) for _ in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
            assert not t.is_alive(), "deadlock — thread did not finish"

        assert sorted(acquired) == ["A", "A", "A", "B", "B", "B"]


class TestCpuRegionLockCrossProcess:
    @staticmethod
    def _hold_region(role, regions, hold_s, tmp_dir, ready_path):
        """Subprocess worker — acquires region lock and holds it."""
        os.environ["ORCHESTRATOR_TMP_DIR"] = tmp_dir
        os.environ["ORCHESTRATOR_INFERENCE_LOCK_POLL_MS"] = "10"
        # Re-import to pick up env override
        import importlib
        import src.runtime.cpu_region_lock as crl

        importlib.reload(crl)
        with crl.cpu_region_lock(role, set(regions), timeout_s=10):
            Path(ready_path).touch()
            time.sleep(hold_s)

    @staticmethod
    def _hold_shared_cohort(role, regions, hold_s, tmp_dir, ready_path):
        os.environ["ORCHESTRATOR_TMP_DIR"] = tmp_dir
        os.environ["ORCHESTRATOR_INFERENCE_LOCK_POLL_MS"] = "10"
        os.environ["ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT"] = "1"
        import importlib
        import src.runtime.cpu_region_lock as crl

        importlib.reload(crl)
        with crl.cpu_region_lock(
            role,
            set(regions),
            instance_idx=0,
            timeout_s=10,
            shared=True,
            capacity=4,
        ):
            Path(ready_path).touch()
            time.sleep(hold_s)

    def test_cross_process_exclusion(self, tmp_path, monkeypatch):
        """fcntl locks are per-fd-pair — verify they hold across
        processes too. Spawn a subprocess that takes q0; assert the
        parent cannot acquire q0 until the subprocess exits."""
        # Use a multiprocessing context that inherits env via 'fork' so
        # the ORCHESTRATOR_TMP_DIR redirect carries over without us
        # having to re-export it in the worker (cleaner test).
        ctx = multiprocessing.get_context("fork")
        ready_path = tmp_path / "child_ready"
        p = ctx.Process(
            target=self._hold_region,
            args=("frontdoor", ["q0"], 1.5, str(tmp_path), str(ready_path)),
        )
        p.start()
        try:
            # Wait for child to actually hold the lock
            deadline = time.time() + 5
            while time.time() < deadline:
                if ready_path.exists():
                    break
                time.sleep(0.05)
            assert ready_path.exists(), "child failed to acquire in time"

            # Parent attempt should time out (child holding for 1.5s)
            with pytest.raises(CpuRegionLockTimeout):
                with cpu_region_lock("frontdoor", {"q0"}, timeout_s=0.5):
                    pytest.fail("should not have acquired across process")

            # After child exits, lock is free
            p.join(timeout=5)
            with cpu_region_lock("frontdoor", {"q0"}, timeout_s=2):
                pass
        finally:
            if p.is_alive():
                p.terminate()
            p.join(timeout=2)

    def test_cross_process_shared_cohort_allows_same_server_only(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT", "1")
        ctx = multiprocessing.get_context("fork")
        ready_path = tmp_path / "shared_child_ready"
        p = ctx.Process(
            target=self._hold_shared_cohort,
            args=("frontdoor", ["q0"], 1.5, str(tmp_path), str(ready_path)),
        )
        p.start()
        try:
            deadline = time.time() + 5
            while time.time() < deadline and not ready_path.exists():
                time.sleep(0.05)
            assert ready_path.exists(), "child failed to establish shared cohort"

            with cpu_region_lock(
                "frontdoor",
                {"q0"},
                instance_idx=0,
                timeout_s=0.5,
                shared=True,
                capacity=4,
            ):
                assert p.is_alive(), "same-server request should join before child exits"

            with pytest.raises(CpuRegionLockTimeout, match="admission timeout|admission deadline exceeded"):
                with cpu_region_lock(
                    "worker_general",
                    {"q0"},
                    instance_idx=0,
                    timeout_s=0.2,
                    shared=True,
                    capacity=4,
                ):
                    pytest.fail("different serving process must not join shared cohort")
        finally:
            if p.is_alive():
                p.terminate()
            p.join(timeout=2)


class TestForInstanceConvenience:
    def test_known_instance_uses_topology(self, monkeypatch):
        """cpu_region_lock_for_instance reads regions from the topology
        table and acquires them. Patch the lookup to return a known set."""
        from src.runtime import instance_topology

        monkeypatch.setattr(
            instance_topology,
            "_INSTANCE_REGIONS_CACHE",
            {("test_role", 0): frozenset({"q0", "q1"})},
        )
        with cpu_region_lock_for_instance("test_role", 0) as paths:
            assert set(paths.keys()) == {"q0", "q1"}

    def test_unknown_instance_is_noop(self, monkeypatch):
        from src.runtime import instance_topology

        monkeypatch.setattr(
            instance_topology,
            "_INSTANCE_REGIONS_CACHE",
            {},
        )
        with cpu_region_lock_for_instance("unknown_role", 999) as paths:
            assert paths == {}


class TestSharedNativeBatchOccupancy:
    def test_shared_holders_report_fractional_load_and_block_exclusive(self, monkeypatch):
        monkeypatch.setenv("ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT", "1")
        with cpu_region_lock(
            "frontdoor", {"q0"}, shared=True, capacity=4, request_tag="batch-a"
        ):
            with cpu_region_lock(
                "frontdoor", {"q0"}, shared=True, capacity=4, request_tag="batch-a"
            ):
                q0 = read_region_occupancy()["per_region"]["q0"]
                assert q0["active"] == 2
                assert q0["capacity"] == 4
                assert q0["load"] == pytest.approx(0.5)
                with pytest.raises(CpuRegionLockTimeout):
                    with cpu_region_lock("other", {"q0"}, timeout_s=0.05):
                        pytest.fail("exclusive placement must conflict with shared lease")

        assert read_region_occupancy()["entries"] == []

    def test_shared_cohort_enforces_certified_capacity(self, monkeypatch):
        monkeypatch.setenv("ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT", "1")
        with cpu_region_lock(
            "frontdoor", {"q0"}, instance_idx=0, shared=True, capacity=2
        ):
            with cpu_region_lock(
                "frontdoor", {"q0"}, instance_idx=0, shared=True, capacity=2
            ):
                with pytest.raises(CpuRegionLockTimeout, match="admission timeout|admission deadline exceeded"):
                    with cpu_region_lock(
                        "frontdoor",
                        {"q0"},
                        instance_idx=0,
                        shared=True,
                        capacity=2,
                        timeout_s=0.05,
                    ):
                        pytest.fail("cohort exceeded certified native slot capacity")
