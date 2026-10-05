"""FIFO ticket queue shared by `cpu_region_lock` and `gpu_quiet_lock`.

WHY. Both locks are `flock`-based, and `flock` has no fairness guarantee: a
holder that releases and immediately re-requests can win the race against a
waiter that has been blocked for hours (observed: two AutoKernel loops
starving a third claimant; an all-region request never seeing all 4 regions
free at once because a stream of single-region requests keeps barging in).

DESIGN. `flock` stays the sole mutual-exclusion fact — this module never
replaces it, only orders *when a waiter is allowed to attempt it*:

1. Before attempting its underlying flock(s), a waiter writes a ticket into
   `<lock dir>/queue/<ticket_id>.json`. `ticket_id` is `monotonic_ns` (system
   clock, comparable across processes on one host, immune to wall-clock
   jumps) zero-padded plus a pid+random tiebreak, so lexical sort == arrival
   order.
2. The waiter then blocks in `wait_for_admission` until no OLDER *live*
   ticket requests an overlapping resource (region, or the single
   "gpu-quiet" pseudo-region). Only then does it proceed to the real,
   already-existing flock-acquisition loop, which still legitimately blocks
   on whatever CURRENTLY holds the flock (holders have no ticket — a ticket
   exists only for the waiting window).
3. The ticket is removed the moment the underlying flock is actually
   acquired, and in `finally` on any failure/timeout/cancel path, so the
   queue never accumulates dead rows under normal operation. Rows are also
   reaped-on-read via `pid` + `/proc/<pid>/stat` start-time + boot id
   liveness, defending against crashes and pid reuse.
4. A releaser that re-acquires is a brand-new `write_ticket` call — it gets
   a new (younger) ticket id and so cannot barge an older waiter.
5. A single-resource lock (gpu-quiet) gets writer preference for free: the
   "overlap" check is universal on one resource, so a queued EXCLUSIVE
   ticket blocks every younger SHARED ticket from even attempting the flock,
   while shared holders that already passed the gate (and so hold no
   ticket) finish undisturbed.

GATING. Entirely opt-in behind `EPYC_LOCK_FIFO=1` (default OFF). An old
client that never calls into this module does not write tickets and is
never gated by `wait_for_admission` — it just flocks directly, exactly as
today. That is "unfair but correct": the flock remains the single source of
mutual exclusion, so old and new clients can never double-grant, only race.

NESTING. A descendant of a process that already holds (or is queued for) an
overlapping lock must never queue behind its own ancestor — that is a
guaranteed deadlock (the ancestor is typically waiting on the descendant
process to exit). `flock`'s mutual exclusion is per *open file description*,
not per-process, so a descendant cannot transparently "inherit" an ancestor's
already-open fd through the existing code paths (they re-`open()` the lock
file and re-`flock()` it). We therefore refuse fast with `NestedLockError`
rather than attempt a that would never be correct — see
`check_nesting`.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Callable, Iterable, Optional

logger = logging.getLogger(__name__)

#: Bumped if the ticket schema changes incompatibly. Recorded in every
#: ticket and every jsonl event for forward diagnostics.
CLIENT_VERSION = 2

#: Single-resource pseudo-region name used by gpu_quiet_lock tickets.
GPU_QUIET_RESOURCE = "gpu-quiet"

_TRUTHY = {"1", "true", "yes", "on"}


def is_fifo_enabled() -> bool:
    """`EPYC_LOCK_FIFO=1` gate. Default OFF until all live loops restart."""
    return os.environ.get("EPYC_LOCK_FIFO", "0").strip().lower() in _TRUTHY


class LockQueueTimeout(RuntimeError):
    """Raised when a ticket could not be admitted within the time budget."""


class NestedLockError(RuntimeError):
    """Raised when a descendant would queue behind its own ancestor."""


# ───────────────────────────── /proc liveness helpers ─────────────────────────────


def read_boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _proc_stat_fields(pid: int) -> list[str] | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    # comm can contain spaces/parens; split on the LAST ") " to skip past it.
    rest = stat.rpartition(") ")[2]
    if not rest:
        return None
    return rest.split()


def proc_start_ticks(pid: int) -> Optional[int]:
    """Field 22 overall (index 19 after stripping `pid (comm) `) — process start time."""
    fields = _proc_stat_fields(pid)
    if fields is None or len(fields) <= 19:
        return None
    try:
        return int(fields[19])
    except ValueError:
        return None


def proc_ppid(pid: int) -> Optional[int]:
    fields = _proc_stat_fields(pid)
    if fields is None or len(fields) <= 1:
        return None
    try:
        return int(fields[1])
    except ValueError:
        return None


def ancestor_pids(pid: int, *, max_depth: int = 64) -> list[int]:
    """Walk the ppid chain up to (not including) pid 1 / init, bounded."""
    out: list[int] = []
    seen: set[int] = {pid}
    cur = pid
    for _ in range(max_depth):
        parent = proc_ppid(cur)
        if parent is None or parent <= 1 or parent in seen:
            break
        out.append(parent)
        seen.add(parent)
        cur = parent
    return out


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ───────────────────────────── ticket read/write ─────────────────────────────


def queue_dir(lock_dir: Path) -> Path:
    return lock_dir / "queue"


def write_ticket(
    qdir: Path,
    *,
    regions: Iterable[str],
    mode: str,
    tag: Optional[str],
    client_version: int = CLIENT_VERSION,
) -> tuple[str, Path]:
    qdir.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    ticket_id = f"{time.monotonic_ns():020d}-{pid:08d}-{uuid.uuid4().hex[:8]}"
    payload = {
        "ticket_id": ticket_id,
        "pid": pid,
        "start_ticks": proc_start_ticks(pid),
        "boot_id": read_boot_id(),
        "regions": sorted(regions),
        "mode": mode,
        "tag": tag or "",
        "client_version": client_version,
        "created_at": time.time(),
    }
    path = qdir / f"{ticket_id}.json"
    tmp = path.with_name(path.name + f".{pid}.tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
    return ticket_id, path


def remove_ticket(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def ticket_is_live(payload: dict) -> bool:
    """A ticket is live only if pid + start time + boot_id all match."""
    try:
        pid = int(payload.get("pid"))
    except (TypeError, ValueError):
        return False
    if not _pid_alive(pid):
        return False
    recorded_ticks = payload.get("start_ticks")
    current_ticks = proc_start_ticks(pid)
    if recorded_ticks is not None and current_ticks != recorded_ticks:
        return False  # pid reused by an unrelated process
    recorded_boot = payload.get("boot_id")
    if recorded_boot and recorded_boot != read_boot_id():
        return False  # survived a reboot — a reused pid on the new boot
    return True


def read_live_tickets(qdir: Path, *, reap: bool = True) -> list[dict]:
    """Live tickets sorted by arrival order (lexical == arrival by construction).

    Dead tickets (crashed owner, reused pid, stale boot id, corrupt JSON) are
    unlinked as a side effect unless `reap=False` (used by read-only status
    views that must not race a live writer's own cleanup).
    """
    try:
        paths = sorted(qdir.glob("*.json"))
    except OSError:
        return []
    live: list[dict] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            if reap:
                remove_ticket(path)
            continue
        if not isinstance(payload, dict) or "ticket_id" not in payload:
            if reap:
                remove_ticket(path)
            continue
        payload = dict(payload)
        payload["_path"] = str(path)
        if ticket_is_live(payload):
            live.append(payload)
        elif reap:
            remove_ticket(path)
    live.sort(key=lambda t: str(t.get("ticket_id") or ""))
    return live


def older_overlapping_blocks(
    live_tickets: list[dict], self_ticket_id: str, self_regions: Iterable[str]
) -> list[dict]:
    want = set(self_regions)
    return [
        ticket
        for ticket in live_tickets
        if str(ticket.get("ticket_id") or "") < self_ticket_id
        and want.intersection(str(r) for r in ticket.get("regions") or [])
    ]


def wait_for_admission(
    qdir: Path,
    ticket_id: str,
    regions: Iterable[str],
    *,
    deadline_s: Optional[float],
    timeout_s: Optional[float],
    cancel_check: Optional[Callable[[], bool]],
    poll_s: float,
    tag: Optional[str],
    label: str = "lock",
    log_every_s: float = 15.0,
) -> float:
    """Block until no older live ticket requests an overlapping resource.

    Returns elapsed wait seconds. Raises `LockQueueTimeout` on cancel/deadline/
    timeout — the ticket is NOT removed here; callers must remove it in
    `finally` regardless of outcome.
    """
    regions = list(regions)
    start = time.perf_counter()
    abs_deadline = None if not timeout_s or timeout_s <= 0 else start + timeout_s
    last_log = start
    while True:
        live = read_live_tickets(qdir)
        blockers = older_overlapping_blocks(live, ticket_id, regions)
        if not blockers:
            return time.perf_counter() - start
        if cancel_check is not None and cancel_check():
            raise LockQueueTimeout(f"{label} queue wait cancelled (tag={tag})")
        now = time.perf_counter()
        if deadline_s is not None and now >= deadline_s:
            raise LockQueueTimeout(f"{label} queue admission deadline exceeded (tag={tag})")
        if abs_deadline is not None and now >= abs_deadline:
            raise LockQueueTimeout(
                f"{label} queue admission timeout after {timeout_s:.1f}s (tag={tag})"
            )
        if now - last_log >= log_every_s:
            logger.info(
                "still queued for %s regions=%s elapsed=%.1fs blockers=%s",
                label,
                sorted(regions),
                now - start,
                [b.get("ticket_id") for b in blockers],
            )
            last_log = now
        wake_at = now + poll_s
        if deadline_s is not None:
            wake_at = min(wake_at, deadline_s)
        if abs_deadline is not None:
            wake_at = min(wake_at, abs_deadline)
        time.sleep(max(0.001, wake_at - now))


def check_nesting(
    regions: Iterable[str],
    *,
    holder_pids: Callable[[], set[int]],
    queue_dir_path: Path,
    label: str = "lock",
    max_depth: int = 64,
) -> None:
    """Refuse fast (`NestedLockError`) if an ancestor holds or queues for an
    overlapping resource. Never blocks — a nested acquire must fail
    immediately, not queue, or it can deadlock against the very ancestor it
    would be waiting on (e.g. a parent `region-lock run` waiting on a child
    process that itself calls back into the same lock).
    """
    pid = os.getpid()
    ancestors = set(ancestor_pids(pid, max_depth=max_depth))
    if not ancestors:
        return
    want = set(regions)
    try:
        held_by = set(holder_pids())
    except Exception:  # best-effort probe; never let it mask a real nesting bug
        held_by = set()
    overlap_holders = ancestors & held_by
    if overlap_holders:
        raise NestedLockError(
            f"{label}: refusing nested acquire — ancestor pid(s) {sorted(overlap_holders)} "
            f"already hold this lock (regions={sorted(want)}); a descendant acquire would "
            "self-deadlock (flock is per open-file-description, not inherited)"
        )
    live = read_live_tickets(queue_dir_path)
    conflicting_pids = {
        int(t["pid"])
        for t in live
        if int(t.get("pid", -1)) in ancestors and want.intersection(t.get("regions") or [])
    }
    if conflicting_pids:
        raise NestedLockError(
            f"{label}: refusing nested acquire — ancestor pid(s) {sorted(conflicting_pids)} "
            f"are queued for an overlapping resource (regions={sorted(want)}); a descendant "
            "acquire would self-deadlock"
        )


# ───────────────────────────── observability ─────────────────────────────


def append_event_log(lock_dir: Path, event: dict) -> None:
    """Append one jsonl record. Best-effort — must never affect lock correctness."""
    path = lock_dir / "lock_events.jsonl"
    try:
        lock_dir.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, sort_keys=True, default=str) + "\n")
    except OSError:
        pass


def queue_status(qdir: Path) -> list[dict]:
    """Ordered `[{position, age_s, pid, tag, regions, mode}, ...]` for display."""
    now = time.time()
    live = read_live_tickets(qdir)
    rows = []
    for position, ticket in enumerate(live):
        rows.append(
            {
                "position": position,
                "age_s": max(0.0, now - float(ticket.get("created_at") or now)),
                "pid": ticket.get("pid"),
                "tag": ticket.get("tag") or "",
                "regions": ticket.get("regions") or [],
                "mode": ticket.get("mode") or "",
                "client_version": ticket.get("client_version"),
            }
        )
    return rows


__all__ = [
    "CLIENT_VERSION",
    "GPU_QUIET_RESOURCE",
    "LockQueueTimeout",
    "NestedLockError",
    "ancestor_pids",
    "append_event_log",
    "check_nesting",
    "is_fifo_enabled",
    "older_overlapping_blocks",
    "proc_ppid",
    "proc_start_ticks",
    "queue_dir",
    "queue_status",
    "read_boot_id",
    "read_live_tickets",
    "remove_ticket",
    "ticket_is_live",
    "wait_for_admission",
    "write_ticket",
]
