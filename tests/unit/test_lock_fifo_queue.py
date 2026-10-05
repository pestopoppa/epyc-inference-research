"""Unit tests for the FIFO ticket queue (`src.runtime.lock_queue`) and its
integration into `cpu_region_lock` / `gpu_quiet_lock` behind `EPYC_LOCK_FIFO=1`.

Covers (handoff requirement #9):
- FIFO order across 3 contenders
- No barging by a re-requesting releaser
- An all-region request admitted ahead of a newer single-region request,
  after current single-region holders release
- Dead-ticket reaping (fake pid / start time)
- pid-reuse defense
- Nested descendant acquire never deadlocks (refuses fast instead)
- gpu-quiet writer preference
- An old (queue-ignoring) client coexists without deadlock or double grant
- Ticket cleanup on timeout

All cross-process scenarios use `multiprocessing.get_context("fork")` so the
`ORCHESTRATOR_TMP_DIR` redirect carries over without re-exporting it in the
worker, mirroring `test_cpu_region_lock.py::test_cross_process_exclusion`.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import time
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _lock_tmpdir(tmp_path, monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_TMP_DIR", str(tmp_path))
    monkeypatch.setenv("ORCHESTRATOR_INFERENCE_LOCK_POLL_MS", "10")
    monkeypatch.setenv("EPYC_LOCK_FIFO", "1")
    yield


import src.runtime.lock_queue as lq  # noqa: E402
from src.runtime.cpu_region_lock import (  # noqa: E402
    CpuRegionLockTimeout,
    cpu_region_lock,
)
from src.runtime.gpu_quiet_lock import (  # noqa: E402
    GPU_QUIET_EXCLUSIVE,
    GPU_QUIET_SHARED,
)


def _ctx():
    return multiprocessing.get_context("fork")


def _append_event(results_path: str, label: str, event: str) -> None:
    with open(results_path, "a", encoding="utf-8") as fh:
        fh.write(f"{label},{event},{time.time()}\n")


def _read_events(results_path: Path) -> list[tuple[str, str, float]]:
    if not Path(results_path).exists():
        return []
    out = []
    for line in Path(results_path).read_text(encoding="utf-8").strip().splitlines():
        label, event, ts = line.split(",")
        out.append((label, event, float(ts)))
    return out


# ───────────────────────────── worker entry points (module-level for fork) ─────────────────────────────


def _cpu_worker(
    tmp_dir: str,
    results_path: str,
    label: str,
    role: str,
    regions: list[str],
    hold_s: float,
    timeout_s: float,
    fifo: bool,
    start_delay: float = 0.0,
    ready_path: str | None = None,
    cross_role: bool = False,
) -> None:
    os.environ["ORCHESTRATOR_TMP_DIR"] = tmp_dir
    os.environ["ORCHESTRATOR_INFERENCE_LOCK_POLL_MS"] = "10"
    os.environ["EPYC_LOCK_FIFO"] = "1" if fifo else "0"
    if cross_role:
        os.environ["ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT"] = "1"
    import src.runtime.cpu_region_lock as crl

    if start_delay:
        time.sleep(start_delay)
    try:
        with crl.cpu_region_lock(role, set(regions), timeout_s=timeout_s, request_tag=label):
            _append_event(results_path, label, "acquire")
            if ready_path:
                Path(ready_path).touch()
            time.sleep(hold_s)
            _append_event(results_path, label, "release")
    except crl.CpuRegionLockTimeout:
        _append_event(results_path, label, "timeout")


def _cpu_worker_n_phase(
    tmp_dir: str,
    results_path: str,
    role: str,
    regions: list[str],
    labels: list[str],
    hold_s: float,
    gap_s: float,
    cross_role: bool = True,
    start_delay: float = 0.0,
    ready_path: str | None = None,
) -> None:
    """Acquire/hold/release `len(labels)` times in sequence — each a brand-new
    (younger) ticket — to model a role that keeps re-requesting (the
    "alternating" starvation shape)."""
    os.environ["ORCHESTRATOR_TMP_DIR"] = tmp_dir
    os.environ["ORCHESTRATOR_INFERENCE_LOCK_POLL_MS"] = "10"
    os.environ["EPYC_LOCK_FIFO"] = "1"
    if cross_role:
        os.environ["ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT"] = "1"
    import src.runtime.cpu_region_lock as crl

    if start_delay:
        time.sleep(start_delay)
    for i, label in enumerate(labels):
        with crl.cpu_region_lock(role, set(regions), timeout_s=10, request_tag=label):
            _append_event(results_path, label, "acquire")
            if ready_path and i == 0:
                Path(ready_path).touch()
            time.sleep(hold_s)
            _append_event(results_path, label, "release")
        if i < len(labels) - 1:
            time.sleep(gap_s)


def _cpu_worker_two_phase(
    tmp_dir: str,
    results_path: str,
    role: str,
    regions: list[str],
    hold_s: float,
    gap_s: float,
    ready_path: str,
) -> None:
    os.environ["ORCHESTRATOR_TMP_DIR"] = tmp_dir
    os.environ["ORCHESTRATOR_INFERENCE_LOCK_POLL_MS"] = "10"
    os.environ["EPYC_LOCK_FIFO"] = "1"
    import src.runtime.cpu_region_lock as crl

    with crl.cpu_region_lock(role, set(regions), timeout_s=10, request_tag="A1"):
        _append_event(results_path, "A1", "acquire")
        Path(ready_path).touch()
        time.sleep(hold_s)
        _append_event(results_path, "A1", "release")
    time.sleep(gap_s)
    with crl.cpu_region_lock(role, set(regions), timeout_s=10, request_tag="A2"):
        _append_event(results_path, "A2", "acquire")
        time.sleep(hold_s)
        _append_event(results_path, "A2", "release")


def _nested_worker(tmp_dir: str, role: str, regions: list[str], result_path: str) -> None:
    os.environ["ORCHESTRATOR_TMP_DIR"] = tmp_dir
    os.environ["ORCHESTRATOR_INFERENCE_LOCK_POLL_MS"] = "10"
    os.environ["EPYC_LOCK_FIFO"] = "1"
    import src.runtime.cpu_region_lock as crl
    import src.runtime.lock_queue as lq2

    try:
        with crl.cpu_region_lock(role, set(regions), timeout_s=2):
            Path(result_path).write_text("UNEXPECTED_ACQUIRED")
    except lq2.NestedLockError:
        Path(result_path).write_text("NESTED_REFUSED")
    except Exception as exc:  # noqa: BLE001
        Path(result_path).write_text(f"OTHER:{type(exc).__name__}")


def _gpu_worker(
    tmp_dir: str,
    results_path: str,
    label: str,
    mode: str,
    hold_s: float,
    timeout_s: float,
    fifo: bool,
    start_delay: float = 0.0,
    ready_path: str | None = None,
) -> None:
    os.environ["ORCHESTRATOR_TMP_DIR"] = tmp_dir
    os.environ["ORCHESTRATOR_INFERENCE_LOCK_POLL_MS"] = "10"
    os.environ["EPYC_LOCK_FIFO"] = "1" if fifo else "0"
    import src.runtime.gpu_quiet_lock as gql

    if start_delay:
        time.sleep(start_delay)
    try:
        with gql.gpu_quiet_lock(mode, role="bench", timeout_s=timeout_s, request_tag=label):
            _append_event(results_path, label, "acquire")
            if ready_path:
                Path(ready_path).touch()
            time.sleep(hold_s)
            _append_event(results_path, label, "release")
    except Exception as exc:  # noqa: BLE001
        _append_event(results_path, label, f"error:{type(exc).__name__}")


# ───────────────────────────── pure unit tests on lock_queue ─────────────────────────────


class TestTicketMechanics:
    def test_ticket_id_lexical_order_matches_arrival(self, tmp_path):
        qdir = lq.queue_dir(tmp_path)
        ids = []
        for _ in range(5):
            tid, _path = lq.write_ticket(qdir, resource_keys=["q0"], mode="exclusive", tag="t")
            ids.append(tid)
            time.sleep(0.001)
        assert ids == sorted(ids)

    def test_dead_ticket_is_reaped_on_read(self, tmp_path):
        qdir = lq.queue_dir(tmp_path)
        qdir.mkdir(parents=True)
        fake_pid = 2**30  # never a real pid on this host
        payload = {
            "ticket_id": "00000000000000000001-00000001-deadbeef",
            "pid": fake_pid,
            "start_ticks": 123,
            "boot_id": lq.read_boot_id(),
            "regions": ["q0"],
            "mode": "exclusive",
            "tag": "dead",
            "client_version": lq.CLIENT_VERSION,
            "created_at": time.time(),
        }
        path = qdir / "00000000000000000001-00000001-deadbeef.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        live = lq.read_live_tickets(qdir)
        assert live == []
        assert not path.exists(), "dead ticket must be reaped on read"

    def test_pid_reuse_defense(self):
        pid = os.getpid()
        real_ticks = lq.proc_start_ticks(pid)
        payload = {
            "ticket_id": "x",
            "pid": pid,
            "start_ticks": (real_ticks or 0) + 999_999,
            "boot_id": lq.read_boot_id(),
            "regions": ["q0"],
            "mode": "exclusive",
            "tag": "reused",
            "client_version": lq.CLIENT_VERSION,
            "created_at": time.time(),
        }
        assert lq.ticket_is_live(payload) is False

    def test_pid_reuse_defense_boot_id_mismatch(self):
        pid = os.getpid()
        payload = {
            "ticket_id": "x",
            "pid": pid,
            "start_ticks": lq.proc_start_ticks(pid),
            "boot_id": "not-the-real-boot-id",
            "regions": ["q0"],
            "mode": "exclusive",
            "tag": "rebooted",
            "client_version": lq.CLIENT_VERSION,
            "created_at": time.time(),
        }
        assert lq.ticket_is_live(payload) is False

    def test_older_overlapping_blocks(self, tmp_path):
        qdir = lq.queue_dir(tmp_path)
        tid_a, _ = lq.write_ticket(qdir, resource_keys=["q0", "q1"], mode="exclusive", tag="a")
        time.sleep(0.002)
        tid_b, _ = lq.write_ticket(qdir, resource_keys=["q1"], mode="exclusive", tag="b")
        live = lq.read_live_tickets(qdir)
        blockers_b = lq.older_overlapping_blocks(live, tid_b, ["q1"])
        assert [b["tag"] for b in blockers_b] == ["a"]
        blockers_a = lq.older_overlapping_blocks(live, tid_a, ["q0", "q1"])
        assert blockers_a == []

    def test_nested_check_detects_queued_ancestor_ticket(self, tmp_path, monkeypatch):
        """Exercises the ticket-branch of `check_nesting` directly: an ancestor
        that is merely QUEUED (not yet holding the flock) must still block a
        nested descendant acquire — refuse fast, never queue behind it."""
        qdir = lq.queue_dir(tmp_path)
        lq.write_ticket(qdir, resource_keys=["q0"], mode="exclusive", tag="ancestor")
        monkeypatch.setattr(lq, "ancestor_pids", lambda pid, max_depth=64: [os.getpid()])
        with pytest.raises(lq.NestedLockError):
            lq.check_nesting(
                ["q0"], holder_pids=lambda: set(), queue_dir_path=qdir, label="test"
            )

    def test_no_ancestors_is_a_fast_noop(self, tmp_path):
        """A process with no live ancestors (e.g. pid 1's immediate child in a
        container) must never be blocked by nesting checks."""
        qdir = lq.queue_dir(tmp_path)
        lq.check_nesting(["q0"], holder_pids=lambda: set(), queue_dir_path=qdir, label="test")


# ───────────────────────────── cross-process FIFO behavior ─────────────────────────────


class TestCpuRegionFifoOrdering:
    def test_fifo_order_three_contenders(self, tmp_path):
        """Three waiters queued behind a holder are admitted in arrival order,
        not reverse or random order — the bug this feature fixes."""
        results = tmp_path / "events.log"
        ready = tmp_path / "blocker_ready"
        ctx = _ctx()
        blocker = ctx.Process(
            target=_cpu_worker,
            args=(str(tmp_path), str(results), "blocker", "roleA", ["q0"], 1.8, 10, True, 0.0, str(ready)),
        )
        blocker.start()
        deadline = time.time() + 5
        while time.time() < deadline and not ready.exists():
            time.sleep(0.02)
        assert ready.exists(), "blocker failed to acquire in time"

        waiters = []
        for i, label in enumerate(["w1", "w2", "w3"]):
            p = ctx.Process(
                target=_cpu_worker,
                args=(str(tmp_path), str(results), label, "roleA", ["q0"], 0.3, 10, True, i * 0.4, None),
            )
            p.start()
            waiters.append(p)

        blocker.join(timeout=10)
        for p in waiters:
            p.join(timeout=10)
            assert p.exitcode == 0

        acquires = [label for label, event, _ts in _read_events(results) if event == "acquire"]
        assert acquires == ["blocker", "w1", "w2", "w3"]

    def test_no_barging_by_rerequesting_releaser(self, tmp_path):
        """A holder that releases and immediately re-requests must get a NEW
        ticket, not barge an already-queued older waiter."""
        results = tmp_path / "events.log"
        a_ready = tmp_path / "a1_ready"
        ctx = _ctx()
        a = ctx.Process(
            target=_cpu_worker_two_phase,
            args=(str(tmp_path), str(results), "roleB", ["q1"], 0.3, 0.05, str(a_ready)),
        )
        a.start()
        deadline = time.time() + 5
        while time.time() < deadline and not a_ready.exists():
            time.sleep(0.02)
        assert a_ready.exists()

        b = ctx.Process(
            target=_cpu_worker,
            args=(str(tmp_path), str(results), "B", "roleB", ["q1"], 0.1, 10, True, 0.0, None),
        )
        b.start()

        a.join(timeout=10)
        b.join(timeout=10)
        assert a.exitcode == 0
        assert b.exitcode == 0

        acquires = [label for label, event, _ts in _read_events(results) if event == "acquire"]
        assert acquires == ["A1", "B", "A2"], acquires

    def test_all_region_request_admitted_before_newer_single_region_waiter(self, tmp_path):
        """An all-region request at the head of the queue gets in once the
        current single-region holders release, while a NEWER single-region
        request (queued after it) keeps waiting."""
        results = tmp_path / "events.log"
        ctx = _ctx()
        q0_ready = tmp_path / "q0_ready"
        q1_ready = tmp_path / "q1_ready"
        holder_q0 = ctx.Process(
            target=_cpu_worker,
            args=(str(tmp_path), str(results), "holder_q0", "roleC", ["q0"], 0.5, 10, True, 0.0, str(q0_ready)),
        )
        holder_q1 = ctx.Process(
            target=_cpu_worker,
            args=(str(tmp_path), str(results), "holder_q1", "roleC", ["q1"], 0.5, 10, True, 0.0, str(q1_ready)),
        )
        holder_q0.start()
        holder_q1.start()
        deadline = time.time() + 5
        while time.time() < deadline and not (q0_ready.exists() and q1_ready.exists()):
            time.sleep(0.02)
        assert q0_ready.exists() and q1_ready.exists()

        all_waiter = ctx.Process(
            target=_cpu_worker,
            args=(str(tmp_path), str(results), "all", "roleC", ["q0", "q1", "q2", "q3"], 0.1, 10, True, 0.0, None),
        )
        all_waiter.start()
        # Give the all-region ticket time to be written and admitted (it has
        # no older overlapping ticket yet, so it blocks on the physical
        # flocks only) before the newer single-region waiter shows up.
        time.sleep(0.3)

        new_q0 = ctx.Process(
            target=_cpu_worker,
            args=(str(tmp_path), str(results), "new_q0", "roleC", ["q0"], 0.1, 10, True, 0.0, None),
        )
        new_q0.start()

        for p in (holder_q0, holder_q1, all_waiter, new_q0):
            p.join(timeout=10)
            assert p.exitcode == 0

        acquires = [label for label, event, _ts in _read_events(results) if event == "acquire"]
        assert acquires.index("all") < acquires.index("new_q0"), acquires
        assert acquires.index("all") > acquires.index("holder_q0")
        assert acquires.index("all") > acquires.index("holder_q1")

    def test_ticket_cleanup_on_timeout(self, tmp_path):
        """A waiter that times out must remove its own ticket — the queue
        never carries a dead row forward for the next contender to trip on."""
        ctx = _ctx()
        ready = tmp_path / "holder_ready"
        results = tmp_path / "events.log"
        holder = ctx.Process(
            target=_cpu_worker,
            args=(str(tmp_path), str(results), "holder", "roleE", ["q3"], 1.0, 10, True, 0.0, str(ready)),
        )
        holder.start()
        deadline = time.time() + 5
        while time.time() < deadline and not ready.exists():
            time.sleep(0.02)
        assert ready.exists()

        with pytest.raises(CpuRegionLockTimeout):
            with cpu_region_lock("roleE", {"q3"}, timeout_s=0.2, request_tag="waiter"):
                pytest.fail("must not acquire while holder is live")

        qdir = lq.queue_dir(tmp_path)
        assert lq.read_live_tickets(qdir) == [], "timed-out waiter's ticket must be removed"

        holder.join(timeout=10)
        assert holder.exitcode == 0

    def test_legacy_client_coexists_without_deadlock_or_double_grant(self, tmp_path):
        """A client that never sets EPYC_LOCK_FIFO (ignores the queue
        entirely) and a FIFO-aware client contending the same region must
        never both be inside the critical section at once, and neither may
        hang waiting on the other."""
        results = tmp_path / "events.log"
        ctx = _ctx()
        legacy = ctx.Process(
            target=_cpu_worker,
            args=(str(tmp_path), str(results), "legacy", "roleD", ["q0"], 0.3, 5, False, 0.0, None),
        )
        fifo_client = ctx.Process(
            target=_cpu_worker,
            args=(str(tmp_path), str(results), "fifo", "roleD", ["q0"], 0.3, 5, True, 0.05, None),
        )
        legacy.start()
        fifo_client.start()
        legacy.join(timeout=10)
        fifo_client.join(timeout=10)
        assert legacy.exitcode == 0, "legacy client must not hang/deadlock"
        assert fifo_client.exitcode == 0, "fifo client must not hang/deadlock"

        spans: dict[str, dict[str, float]] = {}
        for label, event, ts in _read_events(results):
            spans.setdefault(label, {})[event] = ts
        a, b = spans["legacy"], spans["fifo"]
        assert "acquire" in a and "release" in a
        assert "acquire" in b and "release" in b
        # Mutual exclusion is the flock's job regardless of who has a ticket:
        # the two holds must never overlap.
        assert a["release"] <= b["acquire"] or b["release"] <= a["acquire"], (
            "legacy and fifo clients held the region concurrently — double grant"
        )

    def test_nested_descendant_refuses_fast_never_blocks(self, tmp_path):
        """A descendant of the current holder must refuse immediately, never
        queue behind its own ancestor (which would deadlock: the ancestor is
        typically waiting on the descendant process to exit)."""
        result_path = tmp_path / "nested_result"
        ctx = _ctx()
        with cpu_region_lock("roleF", {"q2"}, timeout_s=10):
            child = ctx.Process(
                target=_nested_worker, args=(str(tmp_path), "roleF", ["q2"], str(result_path))
            )
            start = time.time()
            child.start()
            child.join(timeout=5)
            elapsed = time.time() - start
        assert child.exitcode == 0
        assert elapsed < 3.0, "nested acquire must fail fast, not block for the ancestor to release"
        assert result_path.read_text() == "NESTED_REFUSED"


class TestGpuQuietFifoOrdering:
    def test_writer_preference(self, tmp_path):
        """A queued EXCLUSIVE ticket blocks NEW shared acquires (younger
        tickets) while an existing shared holder finishes undisturbed."""
        results = tmp_path / "events.log"
        ctx = _ctx()
        s1_ready = tmp_path / "s1_ready"
        s1 = ctx.Process(
            target=_gpu_worker,
            args=(str(tmp_path), str(results), "S1", GPU_QUIET_SHARED, 0.5, 10, True, 0.0, str(s1_ready)),
        )
        s1.start()
        deadline = time.time() + 5
        while time.time() < deadline and not s1_ready.exists():
            time.sleep(0.02)
        assert s1_ready.exists()

        e = ctx.Process(
            target=_gpu_worker,
            args=(str(tmp_path), str(results), "E", GPU_QUIET_EXCLUSIVE, 0.2, 10, True, 0.05, None),
        )
        e.start()
        time.sleep(0.25)  # let E's ticket register and block behind S1

        s2 = ctx.Process(
            target=_gpu_worker,
            args=(str(tmp_path), str(results), "S2", GPU_QUIET_SHARED, 0.1, 10, True, 0.0, None),
        )
        s2.start()

        for p in (s1, e, s2):
            p.join(timeout=10)
            assert p.exitcode == 0

        acquires = [label for label, event, _ts in _read_events(results) if event == "acquire"]
        assert acquires == ["S1", "E", "S2"], acquires


class TestCrossRoleGlobalFifoOrdering:
    """The main case per the coordinator's review: AutoKernel loops claim CPU
    regions as role `autokernel-cpu` and benches as role `bench` — DIFFERENT
    per-role lock files, so they never contend the per-role flock at all. The
    only place they actually serialize is the cross-role GLOBAL mutex
    (`global_region_lock_path`, gated by
    ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT=1), which a same-role-only
    ticket (the first version of this layer) never ordered. These tests run
    with that flag on, which is how `region-lock run` and `claim.hold_cpu`
    both already run it in production."""

    def test_all_region_bench_request_admitted_before_newer_autokernel_request(self, tmp_path):
        """A `bench` all-region request queued behind an `autokernel-cpu`
        single-region holder (via the GLOBAL mutex, not the per-role lock —
        they're different roles) gets in before a NEWER `autokernel-cpu`
        request for that same region."""
        results = tmp_path / "events.log"
        ctx = _ctx()
        ak_ready = tmp_path / "ak_ready"
        ak_holder = ctx.Process(
            target=_cpu_worker,
            args=(str(tmp_path), str(results), "ak_holder", "autokernel-cpu", ["q0"], 0.6, 10, True, 0.0, str(ak_ready), True),
        )
        ak_holder.start()
        deadline = time.time() + 5
        while time.time() < deadline and not ak_ready.exists():
            time.sleep(0.02)
        assert ak_ready.exists()

        bench_all = ctx.Process(
            target=_cpu_worker,
            args=(str(tmp_path), str(results), "bench_all", "bench", ["q0", "q1", "q2", "q3"], 0.1, 10, True, 0.0, None, True),
        )
        bench_all.start()
        # Give bench's ticket time to register (global:q0 overlap with the
        # holder would be moot — the holder has no ticket — but it must be
        # registered and admitted/blocked on the physical GLOBAL q0 flock
        # before the newer autokernel-cpu request shows up).
        time.sleep(0.3)

        ak_newer = ctx.Process(
            target=_cpu_worker,
            args=(str(tmp_path), str(results), "ak_newer", "autokernel-cpu", ["q0"], 0.1, 10, True, 0.0, None, True),
        )
        ak_newer.start()

        for p in (ak_holder, bench_all, ak_newer):
            p.join(timeout=10)
            assert p.exitcode == 0

        acquires = [label for label, event, _ts in _read_events(results) if event == "acquire"]
        assert acquires.index("bench_all") < acquires.index("ak_newer"), acquires
        assert acquires.index("bench_all") > acquires.index("ak_holder"), acquires

    def test_two_roles_alternating_cannot_starve_a_third_waiting_role(self, tmp_path):
        """roleA and roleB keep releasing and immediately re-requesting
        (each re-request is a brand-new, younger ticket) while roleC's
        ticket sits older than all of those re-requests — roleC must be
        admitted right after the first holder releases, not starved by the
        ongoing A/B alternation."""
        results = tmp_path / "events.log"
        ctx = _ctx()
        a_ready = tmp_path / "a_ready"
        # roleA holds first (uncontended), then alternates A2.
        a = ctx.Process(
            target=_cpu_worker_n_phase,
            args=(str(tmp_path), str(results), "roleA", ["q0"], ["A1", "A2"], 1.0, 0.05, True, 0.0, str(a_ready)),
        )
        a.start()
        deadline = time.time() + 5
        while time.time() < deadline and not a_ready.exists():
            time.sleep(0.02)
        assert a_ready.exists()

        # roleC queues immediately — its ticket predates both A2 and roleB's
        # request, which are written later. A1's long hold (1.0s) gives C's
        # (and B's) process-start/ticket-write overhead a wide margin to land
        # well before A2's ticket is written at ~1.05s.
        c = ctx.Process(
            target=_cpu_worker,
            args=(str(tmp_path), str(results), "C", "roleC", ["q0"], 0.1, 10, True, 0.0, None, True),
        )
        c.start()
        time.sleep(0.3)

        b = ctx.Process(
            target=_cpu_worker_n_phase,
            args=(str(tmp_path), str(results), "roleB", ["q0"], ["B1", "B2"], 0.1, 0.02, True, 0.0, None),
        )
        b.start()

        for p in (a, c, b):
            p.join(timeout=10)
            assert p.exitcode == 0

        acquires = [label for label, event, _ts in _read_events(results) if event == "acquire"]
        # A1 is the uncontested first holder; C's ticket predates every
        # subsequent A/B request, so C must be admitted immediately after A1
        # releases — strictly before A2, B1 and B2, regardless of how many
        # times A and B keep alternating afterward.
        assert acquires[0] == "A1", acquires
        c_pos = acquires.index("C")
        for later in ("A2", "B1", "B2"):
            assert acquires.index(later) > c_pos, acquires

    def test_no_deadlock_across_per_role_and_global_queues_three_contenders(self, tmp_path):
        """Three different roles requesting overlapping-but-not-identical
        region sets (a circular shape: A wants q0+q1, B wants q1+q2, C wants
        q0+q2) must all complete — the single outermost ticket plus the
        existing sorted-region GLOBAL-then-per-role acquire order must stay
        deadlock free with the queue layer added on top."""
        results = tmp_path / "events.log"
        ctx = _ctx()
        procs = [
            ctx.Process(
                target=_cpu_worker,
                args=(str(tmp_path), str(results), label, role, regions, 0.2, 10, True, delay, None, True),
            )
            for label, role, regions, delay in [
                ("A", "roleX", ["q0", "q1"], 0.0),
                ("B", "roleY", ["q1", "q2"], 0.02),
                ("C", "roleZ", ["q0", "q2"], 0.04),
            ]
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=15)
            assert p.exitcode == 0, "a hung/killed process means a deadlock"

        acquires = {label for label, event, _ts in _read_events(results) if event == "acquire"}
        releases = {label for label, event, _ts in _read_events(results) if event == "release"}
        assert acquires == {"A", "B", "C"}
        assert releases == {"A", "B", "C"}


class TestNestingAudit:
    """2026-10-05 audit (coordinator review): does any EXISTING caller do a
    nested acquire that would newly raise NestedLockError under
    EPYC_LOCK_FIFO=1?

    Grepped (both repos) for `--inside`, `region-lock run` invocations, and
    the AutoKernel loop's `ak-check` / tail fence:

    - `--inside`: no match anywhere in either repo.
    - `src/runtime/region_lock_cli.py` (`region-lock run`): the only
      in-tree caller that shells out to it is documentation/comments
      (`scripts/kernel_rnd/autokernel/loop/cpu_window.py`,
      `scripts/kernel_rnd/autokernel/loop/ak_check.py`) describing a PEER
      process (a different session/loop) taking the SAME cores after the
      AutoKernel loop has released them — never a child of a process that
      still holds the claim.
    - `scripts/kernel_rnd/autokernel/loop/ak_check.py` (the AK loop's
      ak-check, run inside the loop's tail fence): `default_peer_status` /
      `_flock_currently_held` only probe with a non-blocking
      `LOCK_EX|LOCK_NB` + immediate `LOCK_UN` — they NEVER call
      `cpu_region_lock`/`gpu_quiet_lock`/`write_ticket`. ak-check cannot
      trigger `NestedLockError` because it never enters the ticket machinery
      at all; its own docstring states the loop "yields its own CPU-region
      claim during actor phases" (`cpu_window.py`'s `CpuClaimLease` yield/
      reacquire) before any such child process runs.
    - `scripts/kernel_rnd/autokernel/loop/claim.py::hold_cpu`: the one
      in-process caller of `cpu_region_lock("autokernel-cpu", ...)`. It
      acquires once per lease and the lease is explicitly released
      (`CpuClaimLease`) before the loop's actor/build/test phases spawn any
      subprocess — by design there is no window where a child of the holding
      process would call back into the same lock while it is still held.

    Conclusion: no existing caller performs a same-lock nested acquire today.
    If that design invariant (yield-before-spawn) is ever violated, the
    FAILURE MODE CHANGES FOR THE BETTER under EPYC_LOCK_FIFO=1: today it
    would hang forever (the parent thread waiting on the child, the child
    blocked on the parent's flock); under the flag it fails fast with
    `NestedLockError` instead. That is a visible new exception type an
    operator could see for the first time post-flip, which is exactly why
    this is called out here rather than silently assumed safe.
    """

    def test_claim_hold_cpu_style_acquire_then_child_reacquire_refuses_fast(self, tmp_path):
        """Models the one-in-process caller (`claim.hold_cpu`) if its
        yield-before-spawn invariant were ever violated: a child of the
        holding process calling back into the SAME role+region must refuse
        fast, not hang — this is the safety net for that invariant, not
        evidence it is currently violated (see class docstring)."""
        result_path = tmp_path / "nested_result"
        ctx = _ctx()
        with cpu_region_lock("autokernel-cpu", {"q1"}, timeout_s=10):
            child = ctx.Process(
                target=_nested_worker,
                args=(str(tmp_path), "autokernel-cpu", ["q1"], str(result_path)),
            )
            start = time.time()
            child.start()
            child.join(timeout=5)
            elapsed = time.time() - start
        assert child.exitcode == 0
        assert elapsed < 3.0
        assert result_path.read_text() == "NESTED_REFUSED"
