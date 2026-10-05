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
    # Explicitly CLEAR, never just "don't set": when this suite runs under
    # `region-lock run ... --role bench`, region_lock_cli.py's own
    # `os.environ.setdefault("ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT", "1")`
    # (module import time) has already set it in THIS process's environ, and
    # the pytest subprocess inherits it — a bare "don't set" leaves it leaked
    # in from the wrapper, not actually unset (2026-10 review finding).
    monkeypatch.delenv("ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT", raising=False)
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
    else:
        # Defense in depth: the fork should already have inherited the
        # fixture's delenv, but a worker that runs cross_role=False must
        # never see this flag on regardless of HOW it got invoked.
        os.environ.pop("ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT", None)
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
    fifo: bool = True,
) -> None:
    """Acquire/hold/release `len(labels)` times in sequence — each a brand-new
    (younger) ticket — to model a role that keeps re-requesting (the
    "alternating" starvation shape). `fifo=False` is the non-vacuity control:
    EPYC_LOCK_FIFO off means releases/re-requests just race the bare flock."""
    os.environ["ORCHESTRATOR_TMP_DIR"] = tmp_dir
    os.environ["ORCHESTRATOR_INFERENCE_LOCK_POLL_MS"] = "10"
    os.environ["EPYC_LOCK_FIFO"] = "1" if fifo else "0"
    if cross_role:
        os.environ["ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT"] = "1"
    else:
        os.environ.pop("ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT", None)
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
    fifo: bool = True,
) -> None:
    os.environ["ORCHESTRATOR_TMP_DIR"] = tmp_dir
    os.environ["ORCHESTRATOR_INFERENCE_LOCK_POLL_MS"] = "10"
    os.environ["EPYC_LOCK_FIFO"] = "1" if fifo else "0"
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


def _sigterm_ticket_worker(tmp_dir: str, ready_path: str) -> None:
    os.environ["ORCHESTRATOR_TMP_DIR"] = tmp_dir
    import src.runtime.lock_queue as lq2

    qdir = lq2.queue_dir(Path(tmp_dir))
    _tid, path = lq2.write_ticket(qdir, resource_keys=["q0"], mode="exclusive", tag="sigterm-test")
    lq2.install_ticket_sigterm_cleanup(path)
    Path(ready_path).write_text(str(path))
    time.sleep(30)  # waits to be SIGTERMed by the test


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

    def test_zombie_pid_is_dead_for_ticket_liveness(self):
        """2026-10 review finding: a SIGKILLed waiter not yet reaped by its
        parent still passes `kill(pid, 0)` (it IS a real pid, just a zombie)
        -- its ticket must not be treated as live, or it blocks every
        younger overlapping waiter until something happens to reap it."""
        pid = os.fork()
        if pid == 0:
            os._exit(0)
        try:
            deadline = time.time() + 2
            state = None
            while time.time() < deadline:
                fields = lq._proc_stat_fields(pid)
                state = fields[0] if fields else None
                if state == "Z":
                    break
                time.sleep(0.01)
            assert state == "Z", f"child did not become a zombie in time (state={state!r})"
            assert lq._pid_alive(pid) is False
            payload = {
                "pid": pid,
                "start_ticks": lq.proc_start_ticks(pid),
                "boot_id": lq.read_boot_id(),
            }
            assert lq.ticket_is_live(payload) is False
        finally:
            os.waitpid(pid, 0)

    def test_sigterm_handler_removes_own_ticket(self, tmp_path):
        """A SIGTERMed waiter must remove its own outstanding ticket rather
        than leaving it for liveness-reaping to eventually clear."""
        ready_path = tmp_path / "ticket_path.txt"
        ctx = _ctx()
        p = ctx.Process(target=_sigterm_ticket_worker, args=(str(tmp_path), str(ready_path)))
        p.start()
        deadline = time.time() + 5
        while time.time() < deadline and not ready_path.exists():
            time.sleep(0.02)
        assert ready_path.exists(), "worker never installed its ticket/handler"
        ticket_path = Path(ready_path.read_text())
        assert ticket_path.exists()

        p.terminate()  # multiprocessing.Process.terminate() sends SIGTERM
        p.join(timeout=5)
        assert not p.is_alive(), "SIGTERM handler must still let the process terminate"
        assert not ticket_path.exists(), "SIGTERM handler must remove the ticket"


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

    @pytest.mark.parametrize(
        "fifo_on",
        [
            True,
            pytest.param(
                False,
                marks=pytest.mark.xfail(
                    strict=False,  # best-effort: real flock wake order is not adversarial, so an occasional XPASS is expected noise, not a signal
                    reason="non-vacuity control (2026-10 review): with EPYC_LOCK_FIFO "
                    "unset, A's immediate re-request races the bare flock against B's "
                    "already-pending wait with no ordering guarantee, and B does not "
                    "reliably win -- this must fail, or the fifo_on=True pass above is "
                    "vacuous (a 50ms+ re-request gap used to let a plain flock waiter "
                    "win anyway; this is why the gap below is zero).",
                ),
            ),
        ],
    )
    def test_no_barging_by_rerequesting_releaser(self, tmp_path, fifo_on):
        """A holder that releases and immediately re-requests must get a NEW
        ticket, not barge an already-queued older waiter."""
        results = tmp_path / "events.log"
        a_ready = tmp_path / "a1_ready"
        ctx = _ctx()
        a = ctx.Process(
            target=_cpu_worker_two_phase,
            # A1's hold is long (1.0s) so B's process-start/ticket-write
            # overhead has a wide margin to land well before A1 releases,
            # even under host load (2026-10 review: a tight hold here flaked
            # under concurrent host activity).
            args=(str(tmp_path), str(results), "roleB", ["q1"], 1.0, 0.0, str(a_ready), fifo_on),
        )
        a.start()
        deadline = time.time() + 5
        while time.time() < deadline and not a_ready.exists():
            time.sleep(0.02)
        assert a_ready.exists()

        b = ctx.Process(
            target=_cpu_worker,
            args=(str(tmp_path), str(results), "B", "roleB", ["q1"], 0.1, 10, fifo_on, 0.0, None),
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

    @pytest.mark.parametrize(
        "fifo_on",
        [
            True,
            pytest.param(
                False,
                marks=pytest.mark.xfail(
                    strict=False,  # best-effort: real flock wake order is not adversarial, so an occasional XPASS is expected noise, not a signal
                    reason="non-vacuity control (2026-10 review): with EPYC_LOCK_FIFO "
                    "unset, A and B's immediate re-requests race the bare flock with no "
                    "ordering guarantee against C's already-pending wait, so C is not "
                    "reliably admitted first -- this must fail.",
                ),
            ),
        ],
    )
    def test_two_roles_alternating_cannot_starve_a_third_waiting_role(self, tmp_path, fifo_on):
        """roleA and roleB keep releasing and IMMEDIATELY re-requesting (zero
        gap — each re-request is a brand-new, younger ticket) while roleC's
        ticket sits older than all of those re-requests — roleC must be
        admitted right after the first holder releases, not starved by the
        ongoing A/B alternation."""
        results = tmp_path / "events.log"
        ctx = _ctx()
        a_ready = tmp_path / "a_ready"
        # roleA holds first (uncontended), then alternates A2 with zero gap.
        a = ctx.Process(
            target=_cpu_worker_n_phase,
            args=(str(tmp_path), str(results), "roleA", ["q0"], ["A1", "A2"], 1.0, 0.0, True, 0.0, str(a_ready), fifo_on),
        )
        a.start()
        deadline = time.time() + 5
        while time.time() < deadline and not a_ready.exists():
            time.sleep(0.02)
        assert a_ready.exists()

        # roleC queues immediately — its ticket predates both A2 and roleB's
        # request, which are written later. A1's long hold (1.0s) gives C's
        # (and B's) process-start/ticket-write overhead a wide margin to land
        # well before A2's ticket is written just after A1 releases.
        c = ctx.Process(
            target=_cpu_worker,
            args=(str(tmp_path), str(results), "C", "roleC", ["q0"], 0.1, 10, fifo_on, 0.0, None, True),
        )
        c.start()
        time.sleep(0.3)

        b = ctx.Process(
            target=_cpu_worker_n_phase,
            args=(str(tmp_path), str(results), "roleB", ["q0"], ["B1", "B2"], 0.1, 0.0, True, 0.0, None, fifo_on),
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

    @pytest.mark.parametrize(
        "fifo_on",
        [
            True,
            pytest.param(
                False,
                marks=pytest.mark.xfail(
                    strict=False,  # best-effort: real flock wake order is not adversarial, so an occasional XPASS is expected noise, not a signal
                    reason="non-vacuity control (2026-10 review): with EPYC_LOCK_FIFO "
                    "unset, A and B's immediate re-requests race the bare flock with no "
                    "ordering guarantee against C's already-pending wait, so C is not "
                    "reliably admitted before A2/B2 -- this must fail. The deadlock-"
                    "freedom assertions (every process completes) are expected to keep "
                    "passing in both modes; only the fairness ordering should flip.",
                ),
            ),
        ],
    )
    def test_no_deadlock_across_per_role_and_global_queues_three_contenders(self, tmp_path, fifo_on):
        """Three different roles: A wants q0 only, B wants q1 only (DISJOINT
        from each other — both can hold concurrently, same as real
        full+quarter placement), C wants BOTH q0 and q1 (overlaps each of
        them, the same "all-region vs singles" shape as
        test_all_region_request_admitted_before_newer_single_region_waiter,
        but across THREE different roles and TWO independent tickets/flock
        pairs instead of one). A and B each cycle twice with an IMMEDIATE
        (zero-gap) re-request; C's single request is queued while A1/B1 both
        hold, so it is older than A2/B2.

        Two independent properties: (1) deadlock freedom — the single
        outermost ticket plus the existing sorted-region GLOBAL-then-per-role
        acquire order must stay deadlock free with the queue layered on top,
        checked unconditionally; (2) fairness — C must not be starved by A/B's
        alternation, checked only with FIFO on (see the xfail control)."""
        results = tmp_path / "events.log"
        ctx = _ctx()
        a_ready = tmp_path / "a_ready"
        b_ready = tmp_path / "b_ready"
        a = ctx.Process(
            target=_cpu_worker_n_phase,
            args=(str(tmp_path), str(results), "roleX", ["q0"], ["A1", "A2"], 0.6, 0.0, True, 0.0, str(a_ready), fifo_on),
        )
        b = ctx.Process(
            target=_cpu_worker_n_phase,
            args=(str(tmp_path), str(results), "roleY", ["q1"], ["B1", "B2"], 0.6, 0.0, True, 0.0, str(b_ready), fifo_on),
        )
        a.start()
        b.start()
        deadline = time.time() + 5
        while time.time() < deadline and not (a_ready.exists() and b_ready.exists()):
            time.sleep(0.02)
        assert a_ready.exists() and b_ready.exists()

        c = ctx.Process(
            target=_cpu_worker,
            args=(str(tmp_path), str(results), "C", "roleZ", ["q0", "q1"], 0.1, 10, fifo_on, 0.0, None, True),
        )
        c.start()

        for p in (a, b, c):
            p.join(timeout=15)
            assert p.exitcode == 0, "a hung/killed process means a deadlock"

        acquires_list = [label for label, event, _ts in _read_events(results) if event == "acquire"]
        releases = {label for label, event, _ts in _read_events(results) if event == "release"}
        # Deadlock freedom: unconditional, must hold in BOTH modes.
        assert set(acquires_list) == {"A1", "A2", "B1", "B2", "C"}
        assert releases == {"A1", "A2", "B1", "B2", "C"}
        # Fairness: C's ticket predates A2 and B2 (both registered only after
        # A1/B1 release) -- it must be admitted before either of them.
        c_pos = acquires_list.index("C")
        assert acquires_list.index("A2") > c_pos, acquires_list
        assert acquires_list.index("B2") > c_pos, acquires_list


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

    def test_cross_role_mutex_off_does_not_falsely_serialize_different_roles(self, tmp_path):
        """Discriminating test for the resource-key fix (coordinator review
        2026-10-05): per-role lock FILES are genuinely separate
        (cpu_region.<role>.<region>.lock), so two different roles requesting
        the same region with the cross-role GLOBAL mutex OFF do not
        physically conflict at all and must be able to run CONCURRENTLY.

        A ticket keyed on the bare region name (the pre-fix design) would
        wrongly serialize them anyway -- manufacturing contention between
        roles that never touch the same lock file. This test fails on that
        design and passes on the role+global composite-key design, so it
        guards the fix rather than just re-confirming the mutual-exclusion
        tests above."""
        results = tmp_path / "events.log"
        ctx = _ctx()
        # Deliberately NOT cross_role=True: this models the primitive used
        # directly, without hold_cpu/region-lock's self-forced
        # ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT=1 -- the GLOBAL mutex is
        # off, exactly as the coordinator found in /etc/environment.
        bench = ctx.Process(
            target=_cpu_worker,
            args=(str(tmp_path), str(results), "bench", "bench", ["q0"], 0.4, 10, True, 0.0, None, False),
        )
        ak = ctx.Process(
            target=_cpu_worker,
            args=(str(tmp_path), str(results), "ak", "autokernel-cpu", ["q0"], 0.4, 10, True, 0.0, None, False),
        )
        bench.start()
        ak.start()
        bench.join(timeout=10)
        ak.join(timeout=10)
        assert bench.exitcode == 0
        assert ak.exitcode == 0

        spans: dict[str, dict[str, float]] = {}
        for label, event, ts in _read_events(results):
            spans.setdefault(label, {})[event] = ts
        b, a = spans["bench"], spans["ak"]
        assert "acquire" in b and "release" in b
        assert "acquire" in a and "release" in a
        # They must OVERLAP: with the GLOBAL mutex off, different roles never
        # contend the same lock file, so a region-based (role-blind) ticket
        # key would be the only thing serializing them -- and it must not.
        overlap = min(b["release"], a["release"]) - max(b["acquire"], a["acquire"])
        assert overlap > 0, (
            "bench and autokernel-cpu were serialized with the GLOBAL mutex "
            f"off -- ticket key is not scoped to the real contended resource: {spans}"
        )
