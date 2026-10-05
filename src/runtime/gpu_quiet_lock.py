"""gpu-quiet — one host-wide reader/writer flock between GPU benches and CPU measurements.

WHY. Pinned GPU host threads (184-191) raise the CPU A/A noise floor ~9x
(INF-70, 2026-09-08): pinning controls placement, not contention. GPU benches
used to keep CPU measurements off the host by taking CPU REGION claims
(``region-lock run --cpu-list 0-95``) for 15-30 minutes, but the orchestrator's
per-call serving claims take the same region flocks, so every CPU role placed
on those regions answered 503 ``contention_denied`` for the whole bench.

THE LOCK. ``{tmp_dir}/gpu_quiet.lock`` beside the region lock files (same
directory resolution as ``cpu_region_lock.region_lock_path``):

* a GPU bench takes it EXCLUSIVE (and no CPU region claim);
* a CPU measurement takes it SHARED, alongside its region claim, so it waits
  for any exclusive holder and many CPU measurements coexist;
* the orchestrator's per-call serving claims never look at it.

ORDERING. Whoever takes both takes gpu-quiet FIRST, then regions
(``gpu_quiet_then_regions``). That combined acquire also backs off: when the
regions do not come within one attempt slice it releases gpu-quiet and starts
over, so it never sits on gpu-quiet while blocked behind a region holder. A
holder that (against the rule) waits on gpu-quiet while holding regions can
therefore never close a cycle with it.

HOLDER RECORDS. One shared flock file cannot carry one payload per reader, so
each holder writes ``gpu_quiet.holders/<pid>-<nonce>.json`` after it acquires
and removes it before it releases. The flock stays the fact: ``holders()``
reports a record only while its pid is a current owner of the lock in
``/proc/locks`` (realized-first, as ``region-lock status`` treats region
payloads).
"""

from __future__ import annotations

import fcntl
import json
import os
import time
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Callable, Iterator, Optional

from src.runtime.cpu_region_lock import (
    CpuRegionLockTimeout,
    _acquire_one_with_timeout,
    _tmp_dir,
    _try_flock,
    cpu_region_lock,
)
from src.runtime.lock_queue import (
    CLIENT_VERSION,
    GPU_QUIET_RESOURCE,
    LockQueueTimeout,
    append_event_log,
    check_nesting,
    is_fifo_enabled,
    queue_dir as _fifo_queue_dir,
    remove_ticket as _fifo_remove_ticket,
    wait_for_admission,
    write_ticket,
)

GPU_QUIET_EXCLUSIVE = "exclusive"
GPU_QUIET_SHARED = "shared"
GPU_QUIET_MODES = (GPU_QUIET_EXCLUSIVE, GPU_QUIET_SHARED)

#: How long one combined attempt waits on the regions while holding gpu-quiet
#: before backing off (releasing gpu-quiet and retrying).
DEFAULT_REGION_ATTEMPT_S = 5.0
_BACKOFF_S = 0.2

SCHEMA = "epyc.orchestrator.gpu_quiet_holder.v1"


def gpu_quiet_lock_path() -> Path:
    """``{tmp_dir}/gpu_quiet.lock`` — the region lock directory, not a ``cpu_region.*`` name.

    Deliberately outside the ``cpu_region.*.*.lock`` namespace so region sweeps,
    status globs and dashboard scans never mistake it for a region.
    """
    return _tmp_dir() / "gpu_quiet.lock"


def gpu_quiet_holders_dir() -> Path:
    return _tmp_dir() / "gpu_quiet.holders"


def _start_ticks(pid: int) -> int | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return int(stat.rpartition(") ")[2].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def lock_owners(path: Path | None = None, proc_locks: Path = Path("/proc/locks")) -> dict[int, str] | None:
    """{pid: "exclusive"|"shared"} for FLOCK owners of the gpu-quiet file; None if unreadable."""
    path = gpu_quiet_lock_path() if path is None else path
    try:
        st = os.stat(path)
        lines = proc_locks.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    owners: dict[int, str] = {}
    for line in lines:
        fields = line.split()
        # "1: FLOCK  ADVISORY  WRITE 1234 08:02:5678 0 EOF"; "->" rows are waiters.
        if "->" in fields or len(fields) < 6 or fields[1] != "FLOCK":
            continue
        # Matched by inode, as `cpu_region_lock._current_lock_owner_pids` does: the
        # device numbers /proc/locks prints need not equal st_dev under overlay/bind
        # mounts, and the lock directory holds only a handful of files.
        try:
            same = int(fields[5].rsplit(":", 1)[-1]) == st.st_ino
            pid = int(fields[4])
        except ValueError:
            continue
        if same:
            owners[pid] = GPU_QUIET_EXCLUSIVE if fields[3] == "WRITE" else GPU_QUIET_SHARED
    return owners


def _sweep_stale_records(directory: Path) -> None:
    """Remove records whose process is gone (or whose pid was reused)."""
    try:
        records = list(directory.glob("*.json"))
    except OSError:
        return
    for record in records:
        try:
            body = json.loads(record.read_text(encoding="utf-8"))
            pid = int(body["pid"])
        except (OSError, ValueError, KeyError, TypeError):
            # Unparseable: possibly mid-write by its owner; leave it unless old.
            try:
                if time.time() - record.stat().st_mtime > 3600:
                    record.unlink()
            except OSError:
                pass
            continue
        ticks = _start_ticks(pid)
        if ticks is None or (body.get("start_ticks") is not None and ticks != body["start_ticks"]):
            try:
                record.unlink()
            except OSError:
                pass


def holders(*, proc_locks: Path = Path("/proc/locks")) -> dict[str, object]:
    """Who holds gpu-quiet now: ``{"path", "held", "mode", "holders": [record, ...]}``.

    Read-only. A record counts only while its pid currently owns the flock; when
    /proc/locks is unreadable, a non-blocking probe decides ``held`` and records
    of live pids are reported unverified.
    """
    path = gpu_quiet_lock_path()
    owners = lock_owners(path, proc_locks) if path.exists() else {}
    records: list[dict[str, object]] = []
    try:
        files = sorted(gpu_quiet_holders_dir().glob("*.json"))
    except OSError:
        files = []
    for record in files:
        try:
            body = json.loads(record.read_text(encoding="utf-8"))
            pid = int(body["pid"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if owners is None:
            if _start_ticks(pid) is None:
                continue
            body = {**body, "verified": False}
        elif pid not in owners:
            continue
        records.append(body)
    if owners is None:
        held = _probe_held(path)
        mode = None
    else:
        held = bool(owners)
        modes = set(owners.values())
        mode = (GPU_QUIET_EXCLUSIVE if GPU_QUIET_EXCLUSIVE in modes
                else GPU_QUIET_SHARED if modes else None)
    return {"path": str(path), "held": held, "mode": mode, "holders": records}


def _probe_held(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with open(path, "a+b") as fh:
            if not _try_flock(fh.fileno(), fcntl.LOCK_EX):
                return True
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            return False
    except OSError:
        return True


@contextmanager
def gpu_quiet_lock(
    mode: str,
    *,
    role: str,
    timeout_s: float = 0.0,
    deadline_s: Optional[float] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    request_tag: Optional[str] = None,
) -> Iterator[dict[str, object]]:
    """Hold gpu-quiet in ``mode`` for the body; yields this holder's record.

    ``timeout_s`` <= 0 blocks indefinitely (waiting logs from the region-lock
    waiter); on expiry raises ``CpuRegionLockTimeout`` — the same exception the
    region claims raise, so existing busy-retry callers need no new case.
    """
    if mode not in GPU_QUIET_MODES:
        raise ValueError(f"gpu-quiet mode must be one of {GPU_QUIET_MODES}, got {mode!r}")
    path = gpu_quiet_lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+b")
    record_path: Path | None = None
    fifo_enabled = is_fifo_enabled()
    fifo_qdir = _fifo_queue_dir(_tmp_dir())
    fifo_ticket_id: str | None = None
    fifo_ticket_path: Path | None = None
    fifo_wait_s = 0.0
    acquire_wall_start = time.perf_counter()

    def _gpu_quiet_holder_pids() -> set[int]:
        owners = lock_owners(path)
        return set(owners.keys()) if owners else set()

    flock_acquired = False
    try:
        if fifo_enabled:
            check_nesting(
                [GPU_QUIET_RESOURCE],
                holder_pids=_gpu_quiet_holder_pids,
                queue_dir_path=fifo_qdir,
                label="gpu_quiet",
            )
            fifo_ticket_id, fifo_ticket_path = write_ticket(
                fifo_qdir,
                regions=[GPU_QUIET_RESOURCE],
                mode=mode,
                tag=request_tag,
                client_version=CLIENT_VERSION,
            )
            try:
                fifo_wait_s = wait_for_admission(
                    fifo_qdir,
                    fifo_ticket_id,
                    [GPU_QUIET_RESOURCE],
                    deadline_s=deadline_s,
                    timeout_s=timeout_s,
                    cancel_check=cancel_check,
                    poll_s=0.05,
                    tag=request_tag,
                    label="gpu_quiet",
                )
            except LockQueueTimeout as exc:
                raise CpuRegionLockTimeout(str(exc)) from exc
        _acquire_one_with_timeout(
            fh,
            region="gpu-quiet",
            role=role,
            timeout_s=timeout_s,
            deadline_s=deadline_s,
            cancel_check=cancel_check,
            request_tag=request_tag,
            lock_type=fcntl.LOCK_EX if mode == GPU_QUIET_EXCLUSIVE else fcntl.LOCK_SH,
        )
        flock_acquired = True
        if fifo_ticket_path is not None:
            _fifo_remove_ticket(fifo_ticket_path)
            append_event_log(
                _tmp_dir(),
                {
                    "event": "acquire",
                    "lock": "gpu_quiet",
                    "ts": time.time(),
                    "pid": os.getpid(),
                    "tag": request_tag,
                    "regions": [GPU_QUIET_RESOURCE],
                    "mode": mode,
                    "wait_s": time.perf_counter() - acquire_wall_start,
                    "client_version": CLIENT_VERSION,
                },
            )
            fifo_ticket_path = None
        pid = os.getpid()
        record: dict[str, object] = {
            "schema": SCHEMA,
            "pid": pid,
            "start_ticks": _start_ticks(pid),
            "role": role,
            "mode": mode,
            "request_tag": request_tag,
            "started_at": time.time(),
            "lock_path": str(path),
        }
        directory = gpu_quiet_holders_dir()
        try:
            directory.mkdir(parents=True, exist_ok=True)
            _sweep_stale_records(directory)
            record_path = directory / f"{pid}-{uuid.uuid4().hex[:12]}.json"
            tmp = record_path.with_name(record_path.name + ".tmp")
            tmp.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(tmp, record_path)
        except OSError:
            record_path = None  # attribution is diagnostic; the flock is the fact
        yield record
    finally:
        if record_path is not None:
            try:
                record_path.unlink()
            except OSError:
                pass
        if fifo_ticket_path is not None:
            _fifo_remove_ticket(fifo_ticket_path)
        if fifo_enabled and flock_acquired:
            append_event_log(
                _tmp_dir(),
                {
                    "event": "release",
                    "lock": "gpu_quiet",
                    "ts": time.time(),
                    "pid": os.getpid(),
                    "tag": request_tag,
                    "regions": [GPU_QUIET_RESOURCE],
                    "mode": mode,
                    "hold_s": time.perf_counter() - acquire_wall_start - fifo_wait_s,
                    "client_version": CLIENT_VERSION,
                },
            )
        fh.close()  # closing the fd releases the flock


@contextmanager
def gpu_quiet_then_regions(
    role: str,
    regions,
    *,
    gpu_quiet: Optional[str],
    timeout_s: float = 0.0,
    request_tag: Optional[str] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    region_attempt_s: float = DEFAULT_REGION_ATTEMPT_S,
) -> Iterator[dict[str, object]]:
    """Take gpu-quiet (when ``gpu_quiet`` is a mode) FIRST, then ``regions``.

    Yields ``{"regions": {region: Path}, "gpu_quiet": record | None}``.
    ``gpu_quiet=None`` is exactly ``cpu_region_lock(role, regions, timeout_s=...)``.
    With a mode, the regions are tried for at most ``region_attempt_s`` per
    attempt while gpu-quiet is held; on a miss gpu-quiet is released and the
    whole acquisition restarts, until ``timeout_s`` (<= 0: forever) expires
    with ``CpuRegionLockTimeout``.
    """
    regions = frozenset(regions or ())
    if gpu_quiet is None:
        with cpu_region_lock(role, regions, timeout_s=timeout_s, request_tag=request_tag,
                             cancel_check=cancel_check) as held:
            yield {"regions": held, "gpu_quiet": None}
        return
    if gpu_quiet not in GPU_QUIET_MODES:
        raise ValueError(f"gpu-quiet mode must be one of {GPU_QUIET_MODES}, got {gpu_quiet!r}")
    deadline = None if timeout_s <= 0 else time.monotonic() + timeout_s

    def remaining() -> float:
        """0.0 = block forever (the lock module's convention)."""
        if deadline is None:
            return 0.0
        left = deadline - time.monotonic()
        if left <= 0:
            raise CpuRegionLockTimeout(
                f"gpu-quiet+regions timeout after {timeout_s:.1f}s "
                f"(role={role}, regions={sorted(regions)}, tag={request_tag})")
        return left

    while True:
        with ExitStack() as stack:
            record = stack.enter_context(gpu_quiet_lock(
                gpu_quiet, role=role, timeout_s=remaining(), cancel_check=cancel_check,
                request_tag=request_tag))
            held: dict = {}
            if regions:
                left = remaining()
                attempt = region_attempt_s if left == 0.0 else min(region_attempt_s, left)
                try:
                    held = stack.enter_context(cpu_region_lock(
                        role, regions, timeout_s=attempt, request_tag=request_tag,
                        cancel_check=cancel_check))
                except CpuRegionLockTimeout:
                    if cancel_check is not None and cancel_check():
                        raise
                    stack.close()  # back off: never sit on gpu-quiet behind a region holder
                    remaining()  # raises once the overall budget is spent
                    time.sleep(_BACKOFF_S)
                    continue
            yield {"regions": held, "gpu_quiet": record}
            return


__all__ = [
    "DEFAULT_REGION_ATTEMPT_S",
    "GPU_QUIET_EXCLUSIVE",
    "GPU_QUIET_MODES",
    "GPU_QUIET_SHARED",
    "gpu_quiet_holders_dir",
    "gpu_quiet_lock",
    "gpu_quiet_lock_path",
    "gpu_quiet_then_regions",
    "holders",
    "lock_owners",
]
