#!/usr/bin/env python3
"""The GPU claim. Acquired, held, re-verified, released.

Invariant 5: a claim is ACQUIRED, never observed. The flock is the fact; looking at
`rocm-smi` and concluding the device is free is a TOCTOU race with whoever is about
to take it.

Re-verification at window CLOSE is not ceremony. `P-AK-SEARCH-1` precondition 1
requires the claim to be still held, by the same holder, at close as well as open --
a measurement taken across a lost claim is a measurement of a contended device.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Iterator

DEVICE_LOCK = Path("/mnt/raid0/llm/tmp/gpu_device.mi210_0.lock")
DEVICE_ID = "mi210_0"


class ClaimRefused(RuntimeError):
    """The device is held by someone else, or the claim did not survive the window."""


def _ensure_orchestrator_importable() -> None:
    """Put the configured CPU-lock owner on the import path.

    Root lane worktrees do not contain the untracked ``repos`` symlinks from the
    shared checkout.  Their Git common directory still identifies that checkout,
    so a child launched from the research repository can resolve the same owner
    without requiring operator-supplied ``PYTHONPATH`` glue.
    """
    configured = Path(os.environ.get("EPYC_ROOT_REPO", "/workspace")).resolve()
    candidates = [configured, configured / "repos" / "epyc-orchestrator"]
    git_marker = configured / ".git"
    try:
        if git_marker.is_file():
            prefix, separator, value = git_marker.read_text(encoding="utf-8").strip().partition(":")
            if prefix == "gitdir" and separator:
                git_dir = Path(value.strip())
                if not git_dir.is_absolute():
                    git_dir = configured / git_dir
                common_dir = (git_dir / "commondir").read_text(encoding="utf-8").strip()
                common_git = Path(common_dir)
                if not common_git.is_absolute():
                    common_git = git_dir / common_git
                candidates.append(common_git.resolve().parent / "repos" / "epyc-orchestrator")
    except OSError:
        pass

    for candidate in candidates:
        if (candidate / "src" / "runtime" / "cpu_region_lock.py").is_file():
            path = str(candidate)
            if path not in sys.path:
                sys.path.insert(0, path)
            return
    raise ClaimRefused(
        f"EPYC_ROOT_REPO={configured} does not resolve the orchestrator CPU-lock owner")


class HeldCpuClaim(dict):
    """The existing acquired context, with read-only open/close observations.

    Dictionary serialization remains the old receipt. This live object only observes
    locks its owning context already acquired; it cannot acquire, restore or transfer
    ownership and is inactive after that context exits. A CPU context's yield lease
    (`CpuClaimLease`, `yield_lease`) is owned by `hold_cpu`, not by this receipt.
    """

    def __init__(self, receipt, lock_paths, *, region_fraction=0.0, affinity=()):
        super().__init__(receipt)
        self._receipt = dict(receipt)
        self._owner_pid = os.getpid()
        self._lock_paths = tuple(Path(path) for path in lock_paths)
        self._active = True
        self._started_at = time.monotonic()
        self._ended_at = None
        self._released = False
        self._closed = None
        self._region_fraction = region_fraction
        self._affinity = tuple(affinity)
        # These are the first acquisition/allocation of THIS direct context, not
        # controller generations. Nothing can reconstruct a live context from it.
        self._domain = {"kind": "direct_loop", "clock": "monotonic",
                        "pid": self._owner_pid, "boot_id": None,
                        "process_start_ticks": None, "error": None}
        try:
            self._domain["boot_id"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            process = Path(f"/proc/{self._owner_pid}/stat").read_text()
            self._domain["process_start_ticks"] = int(process[process.rfind(")") + 2:].split()[19])
        except (OSError, ValueError, IndexError) as exc:
            self._domain["error"] = f"{type(exc).__name__}: {exc}"
        self._opened = self.observe()
        self._context_id = hashlib.sha256(json.dumps(
            {"domain": self._domain, "started_at": self._started_at,
             "locks": self._opened["locks"]}, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    def observe(self):
        started = time.monotonic()
        result = {"observed_at": time.time(), "started_monotonic_s": started,
                  "owner_pid": self._owner_pid, "locks": [], "error": None,
                  "status": "unavailable"}
        try:
            if not self._active or os.getpid() != self._owner_pid:
                raise ClaimRefused("original CPU claim context is no longer active here")
            before = [path.stat() for path in self._lock_paths]
            with Path("/proc/locks").open("r", encoding="utf-8") as stream:
                raw = stream.read((1 << 20) + 1)
            if len(raw) > 1 << 20:
                raise ClaimRefused("kernel lock observation exceeded byte bound")
            lines = [line.split() for line in raw.splitlines()]
            for path, identity in zip(self._lock_paths, before):
                owners = []
                for fields in lines:
                    if len(fields) < 8 or fields[1:4] != ["FLOCK", "ADVISORY", "WRITE"]:
                        continue
                    device = fields[5].split(":")
                    if len(device) != 3:
                        continue
                    if (int(device[0], 16), int(device[1], 16), int(device[2])) == (
                            os.major(identity.st_dev), os.minor(identity.st_dev), identity.st_ino):
                        owners.append({"pid": int(fields[4]), "kernel_row": " ".join(fields)})
                after = path.stat()
                unchanged = (after.st_dev, after.st_ino) == (identity.st_dev, identity.st_ino)
                result["locks"].append({"path": str(path), "device": identity.st_dev,
                    "inode": identity.st_ino, "path_unchanged": unchanged, "owners": owners,
                    "same_holder": unchanged and [row["pid"] for row in owners] == [self._owner_pid]})
            result["status"] = ("held" if result["locks"] and all(
                row["same_holder"] for row in result["locks"]) else "lost")
        except (OSError, ValueError, ClaimRefused) as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        result["ended_monotonic_s"] = time.monotonic()
        return result

    def _closing(self):
        self._closed = self.observe()
        self._active = False

    def _close_observation_failed(self, error):
        self._active = False
        self._closed = {"status": "unavailable", "locks": [], "owner_pid": self._owner_pid,
                        "error": f"{type(error).__name__}: {error}"}

    def _released_now(self):
        self._ended_at = time.monotonic()
        self._released = True

    def retained_interval(self):
        """Closed original owner facts only; not execution permission or a replay grant."""
        if not self._released or self._ended_at is None or self._closed is None:
            raise ClaimRefused("original claim has not completed its owning release")
        locks = self._opened["locks"]
        return {"context_id": self._context_id, "domain": dict(self._domain),
                "ownership_generation": 1, "allocation_generation": 1,
                "started_at": self._started_at, "ended_at": self._ended_at,
                "device_id": self._receipt["device_id"],
                "physical_claim_ids": [
                    f"{self._domain['boot_id']}:flock:{row['device']}:{row['inode']}"
                    for row in locks],
                "physical_region_fraction": self._region_fraction,
                "gpu_device_ids": ([] if self._receipt["device_id"] == "cpu"
                                   else [self._receipt["device_id"]]),
                "memory_reservation_bytes": 0, "affinity_cores": list(self._affinity),
                "open": self._opened, "close": self._closed, "released": True}


def publish_intervals(store, selection, contexts, *, target):
    """Retain original component intervals; the scheduler owns their partition.

    A GPU run's CPU prefix and suffix must not be flattened into its GPU interval.
    This writer neither allocates resources nor mints a scheduler selection.
    """
    from .measurement_capture import ArtifactStore
    from .scheduling import Selection
    if type(store) is not ArtifactStore or type(selection) is not Selection:
        raise ClaimRefused("original artifact store and selected accounting identity are required")
    if not 1 <= len(contexts) <= 2 or any(type(row) is not HeldCpuClaim for row in contexts):
        raise ClaimRefused("original CPU/GPU owning contexts are required")
    components = [row.retained_interval() for row in contexts]
    if len({row["context_id"] for row in components}) != len(components):
        raise ClaimRefused("duplicate original held context")
    body = {"schema": "epyc.autokernel.direct_held_intervals.v1",
            "selection": selection.to_dict(), "selection_digest": selection.digest,
            "target": target, "components": components}
    return store.write("direct-held-intervals", body)


@contextmanager
def hold(lock_path: Path | None = None, *, device_id: str = DEVICE_ID) -> Iterator[dict]:
    """Hold an exclusive claim for the whole window, or refuse.

    Non-blocking on purpose: a loop that waits on a lock behind an unknown holder is
    a loop that looks alive while doing nothing. Refusing tells the operator the
    device is busy, which is a fact worth surfacing.
    """
    lock_path = DEVICE_LOCK if lock_path is None else lock_path
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise ClaimRefused(
            f"{device_id} is claimed by another holder ({lock_path}); "
            f"this loop does not queue behind an unknown holder") from exc

    receipt = None
    try:
        receipt = HeldCpuClaim({"device_id": device_id, "lock_path": str(lock_path),
                               "pid": os.getpid()}, [lock_path])
        yield receipt
    finally:
        try:
            # Re-verify before releasing: if this raises, the claim did not survive
            # the window and every measurement inside it is suspect.
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            raise ClaimRefused(
                f"{device_id} claim did not survive the measurement window; "
                f"results taken under it cannot be trusted")
        try:
            if receipt is not None:
                try:
                    receipt._closing()
                except Exception as error:
                    receipt._close_observation_failed(error)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()
            if receipt is not None:
                receipt._released_now()


@contextmanager
def hold_cpu(cpu_list: str) -> Iterator[dict]:
    """Use the installed orchestrator's physical region owner, never a new flock."""
    os.environ.setdefault("ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT", "1")
    if os.environ["ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT"].lower() not in {
            "1", "true", "yes", "on"}:
        raise ClaimRefused("CPU run requires the existing cross-role region mutex")
    _ensure_orchestrator_importable()
    from src.runtime.cpu_region_lock import cpu_region_lock, global_region_lock_path
    from src.runtime.instance_topology import ATOMIC_REGIONS, cpu_list_to_regions
    from src.runtime.region_lock_cli import _preflight

    reason = _preflight(strict=True)
    if reason:
        raise ClaimRefused(reason)
    regions = cpu_list_to_regions(cpu_list)
    if not regions:
        raise ClaimRefused("CPU affinity maps to no physical regions")
    receipt = None
    lease = None
    with ExitStack() as release:
        owner = cpu_region_lock("autokernel-cpu", regions, timeout_s=1.0,
                                request_tag="autokernel-experimental-serving")
        held = owner.__enter__()
        # The provider context currently holding the regions. Only a yield lease
        # (`CpuClaimLease`, opt-in by the loop's CPU window) ever empties or
        # refills it; without one this is the original owner until close.
        slot = [owner]

        def close_original(*error):
            try:
                if receipt is not None:
                    if slot[0] is None and lease is not None:
                        # Yielded at close: take the same regions back so the close
                        # observation is a real one. A failure leaves it "lost".
                        try:
                            lease.reacquire(reason="close")
                        except Exception as reacquire_error:
                            print(f"cpu claim close could not reacquire: {reacquire_error}",
                                  file=sys.stderr)
                    try:
                        receipt._closing()
                    except Exception as observation_error:
                        receipt._close_observation_failed(observation_error)
            finally:
                current = slot[0]
                slot[0] = None
                result = current.__exit__(*error) if current is not None else False
            # Reached only after the original provider's exit returned. An
            # uncertain release never produces completed interval evidence.
            if receipt is not None:
                receipt._released_now()
            return result

        release.push(close_original)
        if set(held) != set(regions):
            raise ClaimRefused("CPU owner did not acquire the requested physical regions")
        receipt = HeldCpuClaim({"device_id": "cpu", "cpu_list": cpu_list, "regions": sorted(held),
               "lock_paths": {key: str(value) for key, value in held.items()},
               "pid": os.getpid()},
               [*held.values(), *(global_region_lock_path(region) for region in sorted(held))],
               region_fraction=len(regions) / len(ATOMIC_REGIONS),
               affinity=tuple(str(cpu) for cpu in sorted(_cpu_numbers(cpu_list))))
        lease = CpuClaimLease(
            receipt, slot, regions=regions,
            acquire=lambda timeout_s, cancel_check: cpu_region_lock(
                "autokernel-cpu", regions, timeout_s=timeout_s, cancel_check=cancel_check,
                request_tag="autokernel-experimental-serving"),
            preflight=lambda: _preflight(strict=True))
        receipt._yield_lease = lease
        yield receipt


#: The q3 MEASUREMENT claim a GPU measurement takes (operator 2026-10-04 backlog
#: schedule: GPU benches and CPU measurements never overlap). GPU host threads run on
#: 184-191, the SMT siblings of physical cores 88-95 (region q3); a pinned GPU bench
#: raised a CPU A/A floor 0.80% -> 7.22% (INF-70, 2026-09-08). Claimed as `88-95`, NOT
#: `184-191`: the orchestrator's `cpu_list_to_regions` does not fold SMT siblings yet
#: (REGION-SIBLING-1, still on a branch), so `184-191` maps to NO region today and
#: would claim nothing. Measurement-only: production :8083 serving never takes it.
GPU_Q3_MEASUREMENT_CPU_LIST = "88-95"
GPU_Q3_MEASUREMENT_ROLE = "autokernel-gpu-measurement"


def q3_measurement_preflight() -> set:
    """Everything `hold_q3_measurement` needs short of the lock itself: the cross-role
    mutex enabled, the orchestrator importable, its region-lock preflight green and
    `88-95` mapping to exactly q3. Raises ClaimRefused (or ImportError) so a GPU run
    refuses at STARTUP, not at its first measurement after a build."""
    os.environ.setdefault("ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT", "1")
    if os.environ["ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT"].lower() not in {
            "1", "true", "yes", "on"}:
        raise ClaimRefused("GPU q3 measurement claim requires the cross-role region mutex")
    _ensure_orchestrator_importable()
    from src.runtime.instance_topology import cpu_list_to_regions
    from src.runtime.region_lock_cli import _preflight

    reason = _preflight(strict=True)
    if reason:
        raise ClaimRefused(reason)
    regions = cpu_list_to_regions(GPU_Q3_MEASUREMENT_CPU_LIST)
    if set(regions) != {"q3"}:
        raise ClaimRefused(f"{GPU_Q3_MEASUREMENT_CPU_LIST} maps to {sorted(regions)}, not q3")
    return set(regions)


@contextmanager
def hold_q3_measurement(timeout_s: float = 1.0) -> Iterator[dict]:
    """Hold region q3 through the orchestrator's region owner for ONE GPU measurement.

    Raises the provider's `CpuRegionLockTimeout` when another holder has q3
    (`region_lock_busy` recognises it); the caller waits and retries between
    attempts, never pre-empting the holder.
    """
    regions = q3_measurement_preflight()
    from src.runtime.cpu_region_lock import cpu_region_lock
    with cpu_region_lock(GPU_Q3_MEASUREMENT_ROLE, regions, timeout_s=timeout_s,
                         request_tag="autokernel-gpu-measurement-q3") as held:
        if set(held) != {"q3"}:
            raise ClaimRefused("region owner did not grant q3 for the GPU measurement")
        yield {"device_id": "cpu", "purpose": "gpu_measurement_quiet", "regions": ["q3"],
               "cpu_list": GPU_Q3_MEASUREMENT_CPU_LIST,
               "lock_paths": {key: str(value) for key, value in held.items()},
               "pid": os.getpid()}


# ---------------------------------------------------------------- q3-honouring GPU holder
#
# Deadlock this avoids: a GPU AutoKernel run holds `mi210_0` for its whole life and
# takes q3 only around each measurement; a CPU run holds its q3 region claim and, under
# the q3 quiet window, wants `mi210_0` around each measurement. Each would wait on the
# other forever. But a CPU run that already holds q3 is ALREADY excluded from every
# measurement of a GPU holder that takes q3 for its measurements, so it need not wait
# on that holder's device flock. The GPU run advertises this with a marker written
# while it holds the flock; the CPU side trusts it only when the marker's PID is alive
# with the same start time AND is the only process holding the flock (/proc/locks).

def q3_honouring_marker_path(lock_path: Path | None = None) -> Path:
    lock_path = DEVICE_LOCK if lock_path is None else lock_path
    return lock_path.with_name(lock_path.name + ".q3-honouring.json")


def _start_ticks(pid: int) -> int | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return int(stat.rpartition(") ")[2].split()[19])
    except (OSError, ValueError, IndexError):
        return None


@contextmanager
def q3_honouring_marker(lock_path: Path | None = None) -> Iterator[Path]:
    """Advertise, while this process holds the device flock, that its GPU measurements
    take the q3 measurement claim. Removed on exit (only if still ours)."""
    path = q3_honouring_marker_path(lock_path)
    pid = os.getpid()
    body = {"pid": pid, "start_ticks": _start_ticks(pid),
            "role": GPU_Q3_MEASUREMENT_ROLE, "cpu_list": GPU_Q3_MEASUREMENT_CPU_LIST}
    tmp = path.with_name(path.name + f".{pid}.tmp")
    tmp.write_text(json.dumps(body, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
    try:
        yield path
    finally:
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("pid") == pid:
                path.unlink()
        except (OSError, ValueError):
            pass


def _flock_holders(lock_path: Path, proc_locks: Path = Path("/proc/locks")) -> set[int] | None:
    """PIDs holding an FLOCK on `lock_path` (by inode); None if unreadable."""
    try:
        inode = os.stat(lock_path).st_ino
        lines = proc_locks.read_text().splitlines()
    except OSError:
        return None
    holders = set()
    for line in lines:
        fields = line.split()
        if "->" in fields or len(fields) < 6 or fields[1] != "FLOCK":
            continue
        try:
            if int(fields[5].rsplit(":", 1)[1]) == inode:
                holders.add(int(fields[4]))
        except (ValueError, IndexError):
            continue
    return holders


def device_holder_honours_q3(lock_path: Path | None = None, *,
                             proc_locks: Path = Path("/proc/locks")) -> bool:
    lock_path = DEVICE_LOCK if lock_path is None else lock_path
    try:
        marker = json.loads(q3_honouring_marker_path(lock_path).read_text(encoding="utf-8"))
        pid = int(marker["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return False
    if marker.get("start_ticks") is None or _start_ticks(pid) != marker.get("start_ticks"):
        return False
    return _flock_holders(lock_path, proc_locks) == {pid}


def region_lock_busy(error: BaseException) -> bool:
    """True when the provider refused because another holder has the regions.

    Matched by the provider's class name: the orchestrator owns
    `CpuRegionLockTimeout`, and importing it here would make this module's import
    depend on the orchestrator checkout being resolvable.
    """
    return type(error).__name__ == "CpuRegionLockTimeout"


class CpuClaimLease:
    """Yield and re-acquire the SAME physical regions under the original CPU context.

    Opt-in (the loop's `--cpu-window-yield on`); a context whose lease is never
    used behaves exactly as before. The receipt (`HeldCpuClaim`) is unchanged: it
    still cannot acquire or restore anything itself, and its open/close
    observations remain real kernel-lock observations taken while this process
    holds the regions. What a lease adds is honesty about the gaps: every
    release/re-acquire is a numbered `segments` row (`generation` counts
    acquisitions), published beside the held-claim evidence by the loop, so the
    envelope interval is never read as continuous ownership.

    Release is the provider's own exit (the region-lock protocol has no other
    release: closing the flock fds IS the release, exactly as `region-lock run`
    releases when its child exits). Re-acquire is a fresh provider acquisition of
    the same regions, verified to land on the same lock files and to observe as
    held by this process before it is reported held.
    """

    def __init__(self, receipt, slot, *, regions, acquire, preflight=None,
                 clock=time.time, monotonic=time.monotonic, sleep=time.sleep):
        self._receipt = receipt
        self._slot = slot
        self._regions = frozenset(regions)
        self._acquire = acquire
        self._preflight = preflight
        self._clock, self._monotonic, self._sleep = clock, monotonic, sleep
        self._mutex = threading.RLock()
        self.generation = 1
        self.segments = [{"generation": 1, "acquired_at": clock(), "released_at": None,
                          "release_reason": None, "acquire_reason": "initial",
                          "waited_s": 0.0, "wait_exceeded_bound": False}]

    @property
    def held(self) -> bool:
        return self._slot[0] is not None

    def release(self, *, reason: str) -> bool:
        """Exit the current provider context. False when already released."""
        with self._mutex:
            owner = self._slot[0]
            if owner is None:
                return False
            self._slot[0] = None
            try:
                owner.__exit__(None, None, None)
            finally:
                self.segments[-1]["released_at"] = self._clock()
                self.segments[-1]["release_reason"] = reason
            return True

    def reacquire(self, *, reason: str, should_stop=None, wait_bound_s: float | None = None,
                  poll_s: float = 5.0, on_wait=None, on_bound=None) -> dict:
        """Take the same regions back, waiting (never bounded) while a peer holds them.

        `on_wait(waited_s)` is called about once per `poll_s` while waiting and
        `on_bound(waited_s)` once when `wait_bound_s` is exceeded; waiting then
        CONTINUES -- no measurement may start without the claim, and a peer's
        process is never touched. `should_stop()` true while waiting raises
        `ClaimRefused` with the regions still released.
        """
        with self._mutex:
            if self._slot[0] is not None:
                return self.segments[-1]
            if self._preflight is not None:
                problem = self._preflight()
                if problem:
                    raise ClaimRefused(problem)
            started = self._monotonic()
            warned = False
            while True:
                owner = self._acquire(poll_s, should_stop)
                try:
                    held = owner.__enter__()
                except Exception as exc:
                    if not region_lock_busy(exc):
                        raise
                    waited = self._monotonic() - started
                    if should_stop is not None and should_stop():
                        raise ClaimRefused(
                            f"stop requested while waiting {waited:.0f} s to re-acquire the "
                            "CPU regions from a peer; nothing was measured") from None
                    if (not warned and wait_bound_s is not None and waited >= wait_bound_s):
                        warned = True
                        if on_bound is not None:
                            on_bound(waited)
                    if on_wait is not None:
                        on_wait(waited)
                    continue
                if set(held) != set(self._regions) or {
                        key: str(value) for key, value in held.items()} != self._receipt["lock_paths"]:
                    owner.__exit__(None, None, None)
                    raise ClaimRefused("re-acquired CPU regions are not the original lock files")
                self._slot[0] = owner
                observed = self._receipt.observe()
                if observed["status"] != "held":
                    self._slot[0] = None
                    owner.__exit__(None, None, None)
                    raise ClaimRefused(
                        f"re-acquired CPU claim does not observe as held: {observed['error']}")
                waited = round(self._monotonic() - started, 3)
                self.generation += 1
                self.segments.append({
                    "generation": self.generation, "acquired_at": self._clock(),
                    "released_at": None, "release_reason": None, "acquire_reason": reason,
                    "waited_s": waited, "wait_exceeded_bound": warned})
                return self.segments[-1]

    def ledger(self) -> dict:
        """The yield ledger of this context: never scheduler evidence by itself."""
        with self._mutex:
            return {"schema": "epyc.autokernel.cpu_yield_segments.v1",
                    "context_id": getattr(self._receipt, "_context_id", None),
                    "cpu_list": self._receipt.get("cpu_list"),
                    "regions": sorted(self._regions), "generation": self.generation,
                    "held": self.held, "segments": [dict(row) for row in self.segments],
                    "continuous": len(self.segments) == 1}


def yield_lease(receipt) -> CpuClaimLease | None:
    """The yield lease of an original CPU context, or None (GPU claim, other provider)."""
    if type(receipt) is not HeldCpuClaim:
        return None
    lease = getattr(receipt, "_yield_lease", None)
    return lease if type(lease) is CpuClaimLease else None


def _cpu_numbers(cpu_list):
    from ..execution.cpu_region_claim import parse_cpu_list
    return parse_cpu_list(cpu_list)


__all__ = ["ClaimRefused", "CpuClaimLease", "DEVICE_ID", "DEVICE_LOCK", "q3_measurement_preflight",
           "GPU_Q3_MEASUREMENT_CPU_LIST", "GPU_Q3_MEASUREMENT_ROLE",
           "device_holder_honours_q3", "hold", "hold_cpu", "hold_q3_measurement",
           "q3_honouring_marker", "q3_honouring_marker_path", "region_lock_busy",
           "yield_lease"]
