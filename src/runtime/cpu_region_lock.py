"""Cross-process per-CPU-region locking — replacement for the single
global heavy_model.lock.

Motivation: the orchestrator has multiple llama-server instances per role
(e.g. frontdoor: 1×full + 4×quarter via `stack_numa.NUMA_CONFIG`). The
"full" instance uses all of CPUs 0-47 (NUMA_NODE0); a "quarter" uses
24 cores. Multiple non-overlapping instances *should* be able to run
concurrently, but the existing `inference_lock` takes a single
exclusive flock on `heavy_model.lock`, serializing every heavy
inference globally. With multi-worker uvicorn, in-process state in
`ConcurrencyAwareBackend._select` cannot coordinate across processes,
so the global file lock was the only safe primitive.

This module adds per-atomic-region file locks. A given (role,
instance_idx) maps to a set of atomic regions via
`src.runtime.instance_topology`. To dispatch through an instance, the
caller acquires LOCK_EX on the region file for *each* region the
instance occupies — in lexicographic order to prevent deadlock.

The "full" instance acquires multiple region locks; a "quarter" acquires
exactly one. Two quarter instances on disjoint regions can run
concurrently because their lock sets don't intersect.

Lock files live under `tmp_dir/cpu_region.{role}.{region}.lock` (the
orchestrator's configured tmp_dir, defaulting to /mnt/raid0/llm/tmp).
Files are created on-demand on first acquisition. They are empty when
idle, and while held carry a small JSON attribution payload; the flock
state remains the source of liveness truth.

Behavior modeled on `src.runtime.inference_lock`:
- Honors a `deadline_s` absolute deadline + per-acquire `timeout_s`.
- Honors a `cancel_check` callable for early abort.
- Polls at `ORCHESTRATOR_INFERENCE_LOCK_POLL_MS` (default 50ms) — same
  knob as the existing lock module.
- Logs periodic "still waiting" diagnostics when blocked.

Concurrency safety:
- All-or-nothing acquisition: if any region lock times out, all
  previously-acquired locks are released before raising. No partial
  state.
- LIFO release on context exit: ensures fcntl's per-fd state stays
  consistent.

2026-05-22: introduced for the full+quarter concurrency project; lives
alongside the legacy `inference_lock` until phase-3 migration completes.
"""

from __future__ import annotations

import errno
import fcntl
import json
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, IO, Optional

from src.runtime.lock_queue import (
    CLIENT_VERSION,
    LockQueueTimeout,
    append_event_log,
    check_nesting,
    is_fifo_enabled,
    queue_dir as _fifo_queue_dir,
    remove_ticket as _fifo_remove_ticket,
    wait_for_admission,
    write_ticket,
)

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _tmp_dir() -> Path:
    """Resolve the orchestrator's configured tmp_dir.

    Resolution order (env override wins so tests can redirect):
      1. ORCHESTRATOR_TMP_DIR env var (explicit override)
      2. ORCHESTRATOR_PATHS_TMP_DIR (the configured PathsConfig override)
      3. /mnt/raid0/llm/tmp hard-coded fallback

    This module is imported by the standalone ``region-lock`` CLI.  It must
    not import ``src.config`` here: configuration startup reads runtime facts,
    which imports the stack path helpers and can re-enter configuration while
    the runtime-facts module is still initializing.  The paths configuration
    is environment-backed, so the documented override preserves configurable
    lock placement without that import cycle.
    """
    for name in ("ORCHESTRATOR_TMP_DIR", "ORCHESTRATOR_PATHS_TMP_DIR"):
        env_override = os.environ.get(name)
        if env_override:
            return Path(env_override)
    return Path("/mnt/raid0/llm/tmp")


def _lock_poll_s() -> float:
    return max(0.005, _env_float("ORCHESTRATOR_INFERENCE_LOCK_POLL_MS", 50.0) / 1000.0)


def _lock_log_every_s() -> float:
    return max(1.0, _env_float("ORCHESTRATOR_INFERENCE_LOCK_LOG_EVERY_S", 15.0))


def _default_timeout_s() -> float:
    return _env_float("ORCHESTRATOR_INFERENCE_LOCK_TIMEOUT_S", 180.0)


def region_lock_path(role: str, region: str) -> Path:
    """Return the lock-file path for one atomic region of a role.

    Path format: {tmp_dir}/cpu_region.{role}.{region}.lock
    Both `role` and `region` are sanitized to remove path separators —
    callers should never pass attacker-controlled input but defense is
    cheap.
    """
    safe_role = role.replace("/", "_").replace("\\", "_")
    safe_region = region.replace("/", "_").replace("\\", "_")
    return _tmp_dir() / f"cpu_region.{safe_role}.{safe_region}.lock"


# Reserved pseudo-role for the cross-role global region mutex (A-1). Never a
# real role, so `active_region_holders` (which iterates the topology table)
# never surfaces it as a holder — attribution stays role-only.
_GLOBAL_MUTEX_ROLE = "GLOBAL"


def _occupancy_path() -> Path:
    return _tmp_dir() / "cpu_region.occupancy.json"


def _occupancy_mutex_path() -> Path:
    return _tmp_dir() / "cpu_region.occupancy.lock"


def _pid_is_alive(pid: object) -> bool:
    try:
        numeric_pid = int(pid)
    except (TypeError, ValueError):
        return False
    if numeric_pid <= 0:
        return False
    try:
        os.kill(numeric_pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_occupancy_entries_unlocked() -> list[dict[str, object]]:
    try:
        raw = json.loads(_occupancy_path().read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return []
    entries = raw.get("entries") if isinstance(raw, dict) else None
    if not isinstance(entries, list):
        return []
    return [dict(entry) for entry in entries if isinstance(entry, dict)]


def _write_occupancy_entries_unlocked(entries: list[dict[str, object]]) -> None:
    path = _occupancy_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    payload = {
        "schema_version": 1,
        "updated_at": time.time(),
        "entries": entries,
    }
    tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _update_region_occupancy(
    *,
    add: dict[str, object] | None = None,
    remove_token: str | None = None,
) -> None:
    mutex_path = _occupancy_mutex_path()
    mutex_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mutex_path, "a+b") as mutex:
        fcntl.flock(mutex.fileno(), fcntl.LOCK_EX)
        entries = [
            entry
            for entry in _read_occupancy_entries_unlocked()
            if _pid_is_alive(entry.get("pid"))
            and str(entry.get("token") or "") != str(remove_token or "")
        ]
        if add is not None:
            entries.append(add)
        _write_occupancy_entries_unlocked(entries)


def _new_occupancy_entry(
    *,
    role: str,
    regions: list[str],
    instance_idx: Optional[int],
    capacity: int,
    request_tag: Optional[str],
    shared: bool,
) -> dict[str, object]:
    return {
        "token": f"{os.getpid()}-{threading.get_ident()}-{uuid.uuid4().hex}",
        "pid": os.getpid(),
        "role": role,
        "regions": list(regions),
        "instance_idx": instance_idx,
        "capacity": max(1, int(capacity)),
        "shared": bool(shared),
        "request_tag": request_tag or "",
        "started_at": time.time(),
    }


def _reserve_region_admission(
    *,
    role: str,
    regions: list[str],
    instance_idx: Optional[int],
    capacity: int,
    request_tag: Optional[str],
    shared: bool,
    timeout_s: float,
    deadline_s: Optional[float],
    cancel_check: Optional[Callable[[], bool]],
) -> str:
    """Atomically join or establish one cross-process placement cohort.

    ``flock(LOCK_SH)`` alone cannot express "readers from the same serving
    process only": two different overlapping llama-server processes would
    both be admitted as readers.  This crash-pruned registry is therefore the
    cohort admission boundary.  Same-role/same-instance shared requests may
    join up to the certified capacity; every other overlapping placement waits.
    """
    entry = _new_occupancy_entry(
        role=role,
        regions=regions,
        instance_idx=instance_idx,
        capacity=capacity,
        request_tag=request_tag,
        shared=shared,
    )
    token = str(entry["token"])
    requested = set(regions)
    started = time.perf_counter()
    timeout_deadline = None if timeout_s <= 0 else started + timeout_s
    last_log = started
    mutex_path = _occupancy_mutex_path()
    mutex_path.parent.mkdir(parents=True, exist_ok=True)

    while True:
        blockers: list[dict[str, object]] = []
        with open(mutex_path, "a+b") as mutex:
            fcntl.flock(mutex.fileno(), fcntl.LOCK_EX)
            previous = _read_occupancy_entries_unlocked()
            entries = [item for item in previous if _pid_is_alive(item.get("pid"))]
            overlaps = [
                item
                for item in entries
                if requested.intersection(str(region) for region in item.get("regions") or [])
            ]
            same_cohort = bool(shared) and all(
                bool(item.get("shared"))
                and str(item.get("role") or "") == role
                and item.get("instance_idx") == instance_idx
                for item in overlaps
            )
            per_region_active = max(
                (
                    sum(
                        1
                        for item in overlaps
                        if region in {str(value) for value in item.get("regions") or []}
                    )
                    for region in requested
                ),
                default=0,
            )
            cohort_capacity = min(
                [max(1, int(item.get("capacity") or 1)) for item in overlaps]
                + [max(1, int(capacity))]
            )
            compatible = not overlaps or (
                same_cohort and per_region_active < cohort_capacity
            )
            if compatible:
                entries.append(entry)
                _write_occupancy_entries_unlocked(entries)
                return token
            blockers = overlaps
            if len(entries) != len(previous):
                _write_occupancy_entries_unlocked(entries)

        if cancel_check is not None and cancel_check():
            raise CpuRegionLockTimeout(
                f"region admission cancelled before acquire (role={role}, tag={request_tag})"
            )
        now = time.perf_counter()
        if deadline_s is not None and now >= deadline_s:
            raise CpuRegionLockTimeout(
                f"region admission deadline exceeded (role={role}, tag={request_tag})"
            )
        if timeout_deadline is not None and now >= timeout_deadline:
            blocker_roles = sorted({str(item.get("role") or "") for item in blockers})
            raise CpuRegionLockTimeout(
                f"region admission timeout after {timeout_s:.1f}s "
                f"(role={role}, blockers={blocker_roles}, tag={request_tag})"
            )
        if now - last_log >= _lock_log_every_s():
            logger.info(
                "still waiting for region cohort role=%s regions=%s elapsed=%.1fs blockers=%s",
                role,
                ",".join(regions),
                now - started,
                ",".join(sorted({str(item.get("role") or "") for item in blockers}))
                or "(unknown)",
            )
            last_log = now
        wake_at = now + _lock_poll_s()
        if deadline_s is not None:
            wake_at = min(wake_at, deadline_s)
        if timeout_deadline is not None:
            wake_at = min(wake_at, timeout_deadline)
        time.sleep(max(0.001, wake_at - now))


@contextmanager
def cpu_region_occupancy(
    *,
    role: str,
    regions: list[str],
    instance_idx: Optional[int],
    capacity: int,
    request_tag: Optional[str],
    shared: bool,
):
    """Register one live inference for staged dashboard occupancy telemetry."""
    if not regions:
        yield
        return
    entry = _new_occupancy_entry(
        role=role,
        regions=regions,
        instance_idx=instance_idx,
        capacity=capacity,
        request_tag=request_tag,
        shared=shared,
    )
    token = str(entry["token"])
    _update_region_occupancy(add=entry)
    try:
        yield
    finally:
        try:
            _update_region_occupancy(remove_token=token)
        except Exception:
            logger.warning("failed to remove region occupancy token %s", token, exc_info=True)


def read_region_occupancy() -> dict[str, object]:
    """Return live occupancy records and an aggregated per-region load view."""
    mutex_path = _occupancy_mutex_path()
    mutex_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mutex_path, "a+b") as mutex:
        fcntl.flock(mutex.fileno(), fcntl.LOCK_SH)
        entries = [
            entry
            for entry in _read_occupancy_entries_unlocked()
            if _pid_is_alive(entry.get("pid"))
        ]
    per_region: dict[str, dict[str, object]] = {}
    for entry in entries:
        capacity = max(1, int(entry.get("capacity") or 1))
        for region in entry.get("regions") or []:
            key = str(region)
            bucket = per_region.setdefault(
                key,
                {"active": 0, "capacity": 0, "roles": [], "tokens": []},
            )
            bucket["active"] = int(bucket["active"]) + 1
            bucket["capacity"] = max(int(bucket["capacity"]), capacity)
            role = str(entry.get("role") or "")
            if role and role not in bucket["roles"]:
                bucket["roles"].append(role)
            bucket["tokens"].append(str(entry.get("token") or ""))
    for bucket in per_region.values():
        cap = max(1, int(bucket["capacity"]))
        bucket["load"] = min(1.0, int(bucket["active"]) / cap)
    return {"entries": entries, "per_region": per_region}


def global_region_lock_path(region: str) -> Path:
    """Lock-file path for the role-AGNOSTIC global mutex on one atomic region.

    Path format: {tmp_dir}/cpu_region.GLOBAL.{region}.lock

    Acquired (when ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT is enabled)
    before the per-role region lock, so two DIFFERENT roles cannot hold the
    same physical region concurrently. The per-role locks remain the
    attribution layer (`active_region_holders`); this is a separate exclusion
    layer that does not affect attribution.
    """
    safe_region = region.replace("/", "_").replace("\\", "_")
    return _tmp_dir() / f"cpu_region.{_GLOBAL_MUTEX_ROLE}.{safe_region}.lock"


def _cross_role_mutex_enabled() -> bool:
    """A-1: gate the global cross-role region mutex behind the same flag as the
    placement-side change. Off by default → byte-identical legacy behavior
    (per-role locks only; cross-role same-region overlap remains possible)."""
    return os.environ.get("ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT", "0").strip() in {
        "1",
        "true",
        "yes",
        "on",
    }


class CpuRegionLockTimeout(RuntimeError):
    """Raised when one of the region locks could not be acquired within
    the time budget. All locks acquired before the timeout are released
    before this exception propagates.
    """


_LOCAL_REGION_COND = threading.Condition()
_LOCAL_SHARED_HOLDERS: dict[tuple[str, str], int] = {}
_LOCAL_EXCLUSIVE_HOLDERS: dict[tuple[str, str], int] = {}


def _acquire_local_region_locks(
    keys: list[tuple[str, str]],
    *,
    shared: bool,
    timeout_s: float,
    deadline_s: Optional[float],
    cancel_check: Optional[Callable[[], bool]],
    request_tag: Optional[str],
) -> None:
    """Same-process reader/writer guard complementing cross-process flock."""
    started = time.perf_counter()
    timeout_deadline = None if timeout_s <= 0 else started + timeout_s
    with _LOCAL_REGION_COND:
        while True:
            blocked = any(
                _LOCAL_EXCLUSIVE_HOLDERS.get(key, 0) > 0
                or (not shared and _LOCAL_SHARED_HOLDERS.get(key, 0) > 0)
                for key in keys
            )
            if not blocked:
                target = _LOCAL_SHARED_HOLDERS if shared else _LOCAL_EXCLUSIVE_HOLDERS
                for key in keys:
                    target[key] = target.get(key, 0) + 1
                return
            if cancel_check is not None and cancel_check():
                raise CpuRegionLockTimeout(
                    f"local region lock cancelled before acquire (tag={request_tag})"
                )
            now = time.perf_counter()
            if deadline_s is not None and now >= deadline_s:
                raise CpuRegionLockTimeout(
                    f"local region lock deadline exceeded (tag={request_tag})"
                )
            if timeout_deadline is not None and now >= timeout_deadline:
                raise CpuRegionLockTimeout(
                    f"local region lock timeout after {timeout_s:.1f}s (tag={request_tag})"
                )
            wake_candidates = [now + _lock_poll_s()]
            if deadline_s is not None:
                wake_candidates.append(deadline_s)
            if timeout_deadline is not None:
                wake_candidates.append(timeout_deadline)
            _LOCAL_REGION_COND.wait(timeout=max(0.001, min(wake_candidates) - now))


def _release_local_region_locks(keys: list[tuple[str, str]], *, shared: bool) -> None:
    target = _LOCAL_SHARED_HOLDERS if shared else _LOCAL_EXCLUSIVE_HOLDERS
    with _LOCAL_REGION_COND:
        for key in keys:
            remaining = max(0, target.get(key, 0) - 1)
            if remaining:
                target[key] = remaining
            else:
                target.pop(key, None)
        _LOCAL_REGION_COND.notify_all()


def _try_flock(fd: int, lock_type: int) -> bool:
    """Non-blocking fcntl.flock — returns True on success, False if the
    lock is held by another process. Any other OSError propagates.
    """
    try:
        fcntl.flock(fd, lock_type | fcntl.LOCK_NB)
        return True
    except OSError as e:
        if e.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
            return False
        raise


def _acquire_one_with_timeout(
    fh: IO[bytes],
    *,
    region: str,
    role: str,
    timeout_s: float,
    deadline_s: Optional[float],
    cancel_check: Optional[Callable[[], bool]],
    request_tag: Optional[str],
    lock_type: int = fcntl.LOCK_EX,
) -> float:
    """Block until the requested flock mode is acquired, or raise on timeout/cancel.

    Returns elapsed wait seconds. Logs every ~15s while waiting.
    """
    start = time.perf_counter()
    poll_s = _lock_poll_s()
    log_every_s = _lock_log_every_s()
    last_log = start
    last_holders_logged: tuple[str, ...] | None = None
    abs_deadline = None if timeout_s <= 0 else (start + timeout_s)

    while True:
        if _try_flock(fh.fileno(), lock_type):
            return time.perf_counter() - start
        if cancel_check is not None and cancel_check():
            raise CpuRegionLockTimeout(
                f"region lock cancelled before acquire (role={role}, region={region}, "
                f"tag={request_tag})"
            )
        if deadline_s is not None and time.perf_counter() >= deadline_s:
            raise CpuRegionLockTimeout(
                f"region lock deadline exceeded (role={role}, region={region}, tag={request_tag})"
            )
        now = time.perf_counter()
        if abs_deadline is not None and now >= abs_deadline:
            raise CpuRegionLockTimeout(
                f"region lock timeout after {timeout_s:.1f}s "
                f"(role={role}, region={region}, tag={request_tag})"
            )
        if now - last_log >= log_every_s:
            holders = _current_lock_owner_pids(Path(fh.name))
            holder_tuple = tuple(holders)
            if holder_tuple != last_holders_logged:
                logger.info(
                    "still waiting for region lock role=%s region=%s elapsed=%.1fs holders=%s",
                    role,
                    region,
                    now - start,
                    ",".join(holders) or "(unknown)",
                )
                last_holders_logged = holder_tuple
            last_log = now
        time.sleep(poll_s)


def _current_lock_owner_pids(lock_file: Path) -> list[str]:
    """Best-effort lock owners from /proc/locks for the target inode."""
    try:
        inode = str(lock_file.stat().st_ino)
    except Exception:
        return []
    owners: set[str] = set()
    try:
        with open("/proc/locks", "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 6:
                    continue
                pid = parts[4]
                dev_inode = parts[5]
                if not pid.isdigit():
                    continue
                if dev_inode.rsplit(":", 1)[-1] == inode:
                    owners.add(pid)
    except Exception:
        return []
    return sorted(owners)


def _lock_payload_for_region(
    *,
    role: str,
    region: str,
    regions: list[str],
    instance_idx: Optional[int],
    request_tag: Optional[str],
    started_at: float,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "pid": os.getpid(),
        "role": role,
        "region": region,
        "regions": list(regions),
        "instance_idx": instance_idx,
        "request_tag": request_tag,
        "started_at": started_at,
    }


def _write_lock_payload(fh: IO[bytes], payload: dict[str, object]) -> None:
    try:
        fh.seek(0)
        fh.truncate(0)
        fh.write(json.dumps(payload, sort_keys=True).encode("utf-8"))
        fh.write(b"\n")
        fh.flush()
    except Exception as exc:  # noqa: BLE001
        logger.debug("could not write cpu-region lock payload for %s: %s", fh.name, exc)


def _clear_lock_payload(fh: IO[bytes]) -> None:
    try:
        fh.seek(0)
        fh.truncate(0)
        fh.flush()
    except Exception:
        pass


def sweep_stale_region_lock_payloads() -> int:
    """Clear diagnostic payloads whose lock file is currently unlocked.

    The flock is the sole liveness and occupancy fact.  A process terminated
    without running the context manager's ``finally`` block releases that
    flock but leaves its JSON attribution bytes behind.  Probe and hold an
    exclusive flock before clearing so an active dispatcher's payload is never
    changed; PID values are deliberately not consulted because they can be
    reused.

    Returns the number of non-empty, unlocked role lock files cleared.  This
    is best-effort diagnostic hygiene and must never affect dispatch safety.
    """
    cleared = 0
    lock_dir = _tmp_dir()
    try:
        lock_paths = tuple(lock_dir.glob("cpu_region.*.*.lock"))
    except OSError:
        return cleared

    for lock_path in lock_paths:
        # GLOBAL locks are exclusion-only and have no attribution payload.
        if lock_path.name.startswith(f"cpu_region.{_GLOBAL_MUTEX_ROLE}."):
            continue
        try:
            with open(lock_path, "r+b") as fh:
                try:
                    acquired = _try_flock(fh.fileno(), fcntl.LOCK_EX)
                except OSError:
                    continue
                if not acquired:
                    continue
                try:
                    fh.seek(0)
                    if not fh.read().strip():
                        continue
                    _clear_lock_payload(fh)
                    cleared += 1
                finally:
                    try:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                    except OSError:
                        pass
        except OSError:
            # A concurrent filesystem cleanup must not affect dispatch.
            continue
    return cleared


def read_region_lock_payload(lock_file: Path) -> dict[str, object] | None:
    """Read the JSON payload from a region lock file, if present and valid."""
    try:
        raw = lock_file.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw:
        return None
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


def active_region_holders(
    instance_regions: dict[tuple[str, int], frozenset[str]] | None = None,
) -> dict[str, list[int]]:
    """Return {role: [instance_idx, ...]} for instances currently holding
    any CPU region lock (i.e. actively dispatched under PER_REGION_LOCKS=1).

    This is the canonical cross-process source of "which role is decoding"
    used by the cross-role admission gate (Phase B of the cross-role-bw-aware
    routing handoff). Returns empty dict if PER_REGION_LOCKS is disabled or
    no instances are dispatching.

    `instance_regions`: optional override (e.g. for tests). Defaults to the
    live mapping from `src.runtime.instance_topology.get_instance_regions()`.

    The mapping is cheap to call (one /proc/locks scan + per-region stat),
    typically <2 ms even on a busy host. Safe to call from request-handling
    threads.
    """
    if instance_regions is None:
        try:
            from src.runtime.instance_topology import get_instance_regions

            instance_regions = get_instance_regions()
        except Exception:
            return {}

    if not instance_regions:
        return {}

    # Cache lock-file holder lookup per region — many instances share regions.
    region_held: dict[str, bool] = {}

    def _region_has_holder(role: str, region: str) -> bool:
        key = f"{role}.{region}"
        if key in region_held:
            return region_held[key]
        lock_path = region_lock_path(role, region)
        if not lock_path.exists():
            region_held[key] = False
            return False
        held = bool(_current_lock_owner_pids(lock_path))
        region_held[key] = held
        return held

    out: dict[str, list[int]] = {}
    for (role, idx), regions in instance_regions.items():
        if not regions:
            continue
        # Instance is "active" if ANY of its regions has a current holder.
        # That matches how cpu_region_lock acquires the union of region locks
        # for an instance — if even one is held, the instance is dispatched.
        for region in regions:
            if _region_has_holder(role, region):
                out.setdefault(role, []).append(idx)
                break
    # Sort instance indices for deterministic output (helps tests + logs).
    for role in out:
        out[role] = sorted(out[role])
    return out


def active_region_holder_instances(
    instance_regions: dict[tuple[str, int], frozenset[str]] | None = None,
) -> dict[str, list[int]]:
    """Return exact active holder instances by grouping held regions per PID.

    This is the display/metrics counterpart to `active_region_holders`.
    `active_region_holders` is an attribution view: if q0 is held it reports
    every configured instance that contains q0, which is useful for overlap
    checks but over-counts activity on the dashboard. This helper first groups
    held region locks by (role, lock-owner PID), then resolves each exact held
    region set back to the configured instance shape.

    Example: one single-slot MTP worker holding q0+q1+q2+q3 resolves to
    `worker_general: [0]`, not `[0, 1, 2, 3, 4]`.
    """
    if instance_regions is None:
        try:
            from src.runtime.instance_topology import get_instance_regions

            instance_regions = get_instance_regions()
        except Exception:
            return {}

    if not instance_regions:
        return {}

    role_regions: dict[str, set[str]] = {}
    regions_to_idx_by_role: dict[str, dict[frozenset[str], int]] = {}
    for (role, idx), regions in instance_regions.items():
        if not regions:
            continue
        role_regions.setdefault(role, set()).update(regions)
        # If duplicate shapes ever exist, prefer the lowest topology index for
        # stable display; the region set is the physical truth for this view.
        regions_to_idx_by_role.setdefault(role, {}).setdefault(frozenset(regions), idx)

    role_pid_regions: dict[str, dict[str, set[str]]] = {}
    for role, regions in role_regions.items():
        for region in regions:
            lock_path = region_lock_path(role, region)
            if not lock_path.exists():
                continue
            for pid in _current_lock_owner_pids(lock_path):
                role_pid_regions.setdefault(role, {}).setdefault(pid, set()).add(region)

    out: dict[str, set[int]] = {}
    for role, pid_regions in role_pid_regions.items():
        shapes = regions_to_idx_by_role.get(role, {})
        for regions in pid_regions.values():
            idx = shapes.get(frozenset(regions))
            if idx is not None:
                out.setdefault(role, set()).add(idx)

    return {role: sorted(idxs) for role, idxs in out.items()}


def held_regions_by_role(
    instance_regions: dict[tuple[str, int], frozenset[str]] | None = None,
) -> dict[str, frozenset[str]]:
    """EXACT cross-role region view: {role: frozenset(regions with a held lock)}.

    Unlike `active_region_holders` (an ATTRIBUTION view that marks an *instance*
    active when ANY of its regions is held — and therefore over-reports, since a
    held quarter q0 also flags the role's `full` instance that contains q0),
    this reports ONLY the physical regions whose per-role lock file is actually
    held. It is the correct input for shape-aware contention/placement decisions
    (audit P1) — placement disjointness must be computed from the precise held
    region set, not from over-reported instance membership.

    `instance_regions` defaults to the live topology map; pass an override for
    tests. Roles with no held region are omitted (no empty entries). Reuses the
    same `/proc/locks` detection as `active_region_holders`; that function is
    UNCHANGED — this is a separate, additive helper.
    """
    if instance_regions is None:
        try:
            from src.runtime.instance_topology import get_instance_regions

            instance_regions = get_instance_regions()
        except Exception:
            return {}

    if not instance_regions:
        return {}

    # The atomic regions any role could occupy, keyed by role → set(regions),
    # so we probe each (role, region) lock file exactly once.
    role_regions: dict[str, set[str]] = {}
    for (role, _idx), regions in instance_regions.items():
        if regions:
            role_regions.setdefault(role, set()).update(regions)

    out: dict[str, frozenset[str]] = {}
    for role, regions in role_regions.items():
        held = {
            region
            for region in regions
            if (lp := region_lock_path(role, region)).exists() and _current_lock_owner_pids(lp)
        }
        if held:
            out[role] = frozenset(held)
    return out


@contextmanager
def cpu_region_lock(
    role: str,
    regions: frozenset[str] | set[str] | list[str] | tuple[str, ...],
    *,
    instance_idx: Optional[int] = None,
    timeout_s: Optional[float] = None,
    deadline_s: Optional[float] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    request_tag: Optional[str] = None,
    shared: bool = False,
    capacity: int = 1,
    telemetry: Optional[dict[str, Any]] = None,
):
    """Acquire each region's file lock in exclusive or certified-shared mode.

    ``shared=True`` is reserved for requests sent to the *same* serving
    process, where llama-server's native slots are the concurrency boundary.
    When cross-role placement is enabled, the crash-pruned occupancy registry
    atomically establishes a serving-process cohort: only matching shared
    requests may join it, and every overlapping different process waits until
    it drains. Per-role flock files remain the attribution/legacy boundary.

    Yields a dict {region: Path} so callers can inspect / log which lock
    files were taken. Releases all locks (LIFO order) on context exit,
    including on exceptions.

    Args:
        role: Role name (e.g. "frontdoor", "worker_general"). Used only
            for the lock filename and diagnostic logs.
        regions: Set of atomic region identifiers (e.g. {"q0", "q1"}).
            Iteration order is sorted internally; callers can pass any
            iterable.
        timeout_s: Per-acquire timeout in seconds. None → use the env
            default (`ORCHESTRATOR_INFERENCE_LOCK_TIMEOUT_S` or 180).
            0 or negative disables timeout (blocks indefinitely).
        deadline_s: Absolute deadline as `time.perf_counter()` value.
            Both `timeout_s` and `deadline_s` may be supplied; whichever
            is more restrictive applies per-region. If you have a budget
            for the whole multi-lock acquisition, pass `deadline_s`.
        cancel_check: Optional zero-arg callable that returns True to
            cancel an in-progress acquire (e.g. client disconnected).
        request_tag: Opaque tag included in diagnostic logs to help
            identify which inference triggered the lock attempt.

    Raises:
        CpuRegionLockTimeout: if any region cannot be acquired within
            the budget. All previously-acquired locks released first.

    Example:
        from src.runtime.instance_topology import get_instance_regions
        regions = get_instance_regions().get(("frontdoor", 0), frozenset())
        with cpu_region_lock("frontdoor", regions, deadline_s=time.perf_counter() + 60):
            # safe to dispatch to frontdoor's full instance — no other
            # instance can be running on q0..q3 right now
            run_inference()
    """
    if not regions:
        # Empty region set → no-op context. Treat as no-CPU-conflict
        # (some embedder paths route through this for code simplicity).
        yield {}
        return

    if timeout_s is None:
        timeout_s = _default_timeout_s()

    sorted_regions = sorted(regions)
    started_at = time.time()
    handles: list[tuple[str, Path, IO[bytes]]] = []
    acquired_paths: dict[str, Path] = {}
    lock_type = fcntl.LOCK_SH if shared else fcntl.LOCK_EX

    def _acquire(lock_role: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Open in 'a+b' so the file is created if missing and we have a
        # binary fd suitable for fcntl.flock.
        fh = open(path, "a+b")
        try:
            _acquire_one_with_timeout(
                fh,
                region=region,
                role=lock_role,
                timeout_s=timeout_s,
                deadline_s=deadline_s,
                cancel_check=cancel_check,
                request_tag=request_tag,
                lock_type=lock_type,
            )
        except BaseException:
            fh.close()
            raise
        # Shared holders cannot safely own one mutable attribution payload:
        # one completion would clear another live request's JSON.  /proc lock
        # ownership + the occupancy ledger provide exact shared attribution.
        if lock_role != _GLOBAL_MUTEX_ROLE and not shared:
            _write_lock_payload(
                fh,
                _lock_payload_for_region(
                    role=lock_role,
                    region=region,
                    regions=sorted_regions,
                    instance_idx=instance_idx,
                    request_tag=request_tag,
                    started_at=started_at,
                ),
            )
        handles.append((region, path, fh))

    cross_role_mutex = _cross_role_mutex_enabled()
    reservation_token: str | None = None
    emitted_lease_token: str | None = None
    local_keys = (
        [(_GLOBAL_MUTEX_ROLE, region) for region in sorted_regions]
        if cross_role_mutex
        else []
    ) + [(role, region) for region in sorted_regions]
    local_acquired = False

    fifo_enabled = is_fifo_enabled()
    fifo_qdir = _fifo_queue_dir(_tmp_dir())
    fifo_ticket_id: str | None = None
    fifo_ticket_path: Path | None = None
    fifo_wait_s = 0.0
    acquire_wall_start = time.perf_counter()

    # The ACTUAL serialization domain this acquire will contend, so the ticket
    # orders exactly what the flocks exclude — never more (that would create
    # fairness-induced contention between roles that never physically
    # conflict), never less (that would leave the real contention path
    # unordered). Two components:
    #   - the per-role lock, always taken: `role:<role>:<region>` — only
    #     orders waiters of the SAME role for the SAME region (e.g. two
    #     AutoKernel loops both claiming role "autokernel-cpu").
    #   - the cross-role GLOBAL mutex, taken only when
    #     ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT is on AND this is not a
    #     shared cohort join: `global:<region>` — orders waiters across
    #     DIFFERENT roles (e.g. "autokernel-cpu" vs "bench"), which is the
    #     cross-role starvation case this layer exists for.
    # One ticket, one admission wait, written/awaited at the OUTERMOST point
    # before either flock layer is attempted: there is no separate "per-role
    # queue" and "global queue" to invert against each other, because both
    # live in the same ticket (same file, same monotonic_ns arrival order).
    fifo_resource_keys = [f"role:{role}:{region}" for region in sorted_regions]
    if cross_role_mutex and not shared:
        fifo_resource_keys += [f"global:{region}" for region in sorted_regions]

    def _region_holder_pids() -> set[int]:
        pids: set[int] = set()
        for region in sorted_regions:
            pids.update(int(p) for p in _current_lock_owner_pids(region_lock_path(role, region)))
            if cross_role_mutex and not shared:
                pids.update(int(p) for p in _current_lock_owner_pids(global_region_lock_path(region)))
        return pids

    try:
        if fifo_enabled:
            check_nesting(
                fifo_resource_keys,
                holder_pids=_region_holder_pids,
                queue_dir_path=fifo_qdir,
                label=f"cpu_region:{role}",
            )
            fifo_ticket_id, fifo_ticket_path = write_ticket(
                fifo_qdir,
                resource_keys=fifo_resource_keys,
                display_regions=sorted_regions,
                mode="shared" if shared else "exclusive",
                tag=request_tag,
                client_version=CLIENT_VERSION,
            )
            try:
                fifo_wait_s = wait_for_admission(
                    fifo_qdir,
                    fifo_ticket_id,
                    fifo_resource_keys,
                    deadline_s=deadline_s,
                    timeout_s=timeout_s,
                    cancel_check=cancel_check,
                    poll_s=_lock_poll_s(),
                    tag=request_tag,
                    label=f"cpu_region:{role}",
                )
            except LockQueueTimeout as exc:
                raise CpuRegionLockTimeout(str(exc)) from exc
        if cross_role_mutex:
            reservation_token = _reserve_region_admission(
                role=role,
                regions=sorted_regions,
                instance_idx=instance_idx,
                capacity=max(1, int(capacity)),
                request_tag=request_tag,
                shared=shared,
                timeout_s=timeout_s,
                deadline_s=deadline_s,
                cancel_check=cancel_check,
            )
        _acquire_local_region_locks(
            local_keys,
            shared=shared,
            timeout_s=timeout_s,
            deadline_s=deadline_s,
            cancel_check=cancel_check,
            request_tag=request_tag,
        )
        local_acquired = True
        # A-1: when enabled, acquire the role-AGNOSTIC global region mutex for
        # exclusive cohort only. Shared cohorts are mutually excluded by the
        # atomic admission registry; taking LOCK_SH here would incorrectly let
        # two different serving-process cohorts overlap. Consistent ordering
        # (GLOBAL-all-then-role-all, each region-sorted) prevents deadlock.
        if cross_role_mutex and not shared:
            for region in sorted_regions:
                _acquire(_GLOBAL_MUTEX_ROLE, global_region_lock_path(region))
        for region in sorted_regions:
            path = region_lock_path(role, region)
            _acquire(role, path)
            acquired_paths[region] = path
        if fifo_ticket_path is not None:
            # The ticket's job (ordering the WAIT) ends the moment the flock
            # is actually held; the flock itself is now the liveness fact.
            _fifo_remove_ticket(fifo_ticket_path)
            append_event_log(
                _tmp_dir(),
                {
                    "event": "acquire",
                    "lock": "cpu_region",
                    "ts": time.time(),
                    "pid": os.getpid(),
                    "tag": request_tag,
                    "regions": sorted_regions,
                    "mode": "shared" if shared else "exclusive",
                    "wait_s": time.perf_counter() - acquire_wall_start,
                    "client_version": CLIENT_VERSION,
                },
            )
            fifo_ticket_path = None
        emitted_lease_token = reservation_token or (
            f"flock-{os.getpid()}-{threading.get_ident()}-{uuid.uuid4().hex}"
        )
        try:
            from src.runtime.live_telemetry import emit_lifecycle_transition

            identity = dict(telemetry or {})
            identity.setdefault("role", role)
            if not any(identity.get(key) for key in ("request_id", "task_id", "batch_id")):
                tag = str(request_tag or "")
                if tag.startswith(("chat-", "task-")):
                    identity["task_id"] = tag
                elif tag:
                    identity["batch_id"] = tag
            emit_lifecycle_transition(
                "placement_lease_acquired",
                lease_token=emitted_lease_token,
                details={
                    "regions": sorted_regions,
                    "instance_idx": instance_idx,
                    "capacity": max(1, int(capacity)),
                    "shared": bool(shared),
                },
                **identity,
            )
        except Exception:
            pass
        if reservation_token is not None:
            yield acquired_paths
        else:
            with cpu_region_occupancy(
                role=role,
                regions=sorted_regions,
                instance_idx=instance_idx,
                capacity=max(1, int(capacity)),
                request_tag=request_tag,
                shared=shared,
            ):
                yield acquired_paths
    finally:
        # LIFO release: close in reverse acquire order. Closing the file
        # descriptor releases the fcntl lock automatically.
        released_any_flock = bool(handles)
        for _region, _path, fh in reversed(handles):
            try:
                if not shared:
                    _clear_lock_payload(fh)
                fh.close()
            except Exception:
                pass
        if local_acquired:
            _release_local_region_locks(local_keys, shared=shared)
        # Covers every early-exit path (nesting refusal, admission timeout,
        # cancellation, or a mid-acquisition failure): the ticket must never
        # outlive the attempt it represents.
        if fifo_ticket_path is not None:
            _fifo_remove_ticket(fifo_ticket_path)
        if fifo_enabled and released_any_flock:
            append_event_log(
                _tmp_dir(),
                {
                    "event": "release",
                    "lock": "cpu_region",
                    "ts": time.time(),
                    "pid": os.getpid(),
                    "tag": request_tag,
                    "regions": sorted_regions,
                    "mode": "shared" if shared else "exclusive",
                    "hold_s": time.perf_counter() - acquire_wall_start - fifo_wait_s,
                    "client_version": CLIENT_VERSION,
                },
            )
        if reservation_token is not None:
            try:
                _update_region_occupancy(remove_token=reservation_token)
            except Exception:
                logger.warning(
                    "failed to remove region admission token %s",
                    reservation_token,
                    exc_info=True,
                )
        if emitted_lease_token is not None:
            try:
                from src.runtime.live_telemetry import emit_lifecycle_transition

                identity = dict(telemetry or {})
                identity.setdefault("role", role)
                emit_lifecycle_transition(
                    "placement_lease_released",
                    lease_token=emitted_lease_token,
                    details={
                        "regions": sorted_regions,
                        "instance_idx": instance_idx,
                        "capacity": max(1, int(capacity)),
                        "shared": bool(shared),
                    },
                    **identity,
                )
            except Exception:
                pass


@contextmanager
def cpu_region_lock_for_instance(
    role: str,
    instance_idx: int,
    *,
    timeout_s: Optional[float] = None,
    deadline_s: Optional[float] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    request_tag: Optional[str] = None,
    shared: bool = False,
    capacity: int = 1,
    telemetry: Optional[dict[str, Any]] = None,
):
    """Convenience: look up regions for (role, instance_idx) via
    `instance_topology.get_instance_regions()` and acquire them.

    If the (role, instance_idx) pair is unknown to the topology table
    (e.g. an embedder or a role not declared in NUMA_CONFIG), yields
    an empty path-set and runs the body without any locking. Callers
    that need strict locking should use `cpu_region_lock(role, regions)`
    directly and assemble `regions` themselves.
    """
    from src.runtime.instance_topology import get_instance_regions

    regions = get_instance_regions().get((role, instance_idx), frozenset())
    with cpu_region_lock(
        role,
        regions,
        instance_idx=instance_idx,
        timeout_s=timeout_s,
        deadline_s=deadline_s,
        cancel_check=cancel_check,
        request_tag=request_tag,
        shared=shared,
        capacity=capacity,
        telemetry=telemetry,
    ) as paths:
        yield paths
