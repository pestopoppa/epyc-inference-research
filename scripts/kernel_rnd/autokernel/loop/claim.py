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


def _process_start_ticks(pid: int) -> int | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return int(stat[stat.rfind(")") + 2:].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def _ancestor_pids(pid: int) -> list[int]:
    """This process's ancestors, nearest first (bounded; read-only /proc walk)."""
    chain = []
    current = pid
    for _ in range(64):
        try:
            stat = Path(f"/proc/{current}/stat").read_text()
            parent = int(stat[stat.rfind(")") + 2:].split()[1])
        except (OSError, ValueError, IndexError):
            break
        if parent <= 0 or parent in chain:
            break
        chain.append(parent)
        current = parent
    return chain


def observe_gpu_quiet(lock_path: Path, *, proc_locks: Path = Path("/proc/locks")) -> dict:
    """Read-only observation of who owns the host-wide gpu-quiet flock right now.

    A GPU run with no CPU region claim (`--gpu-cpu-region-claim off`) runs under a
    gpu-quiet EXCLUSIVE hold taken by its launcher (`region-lock run --gpu-quiet
    exclusive -- serial_run ...`), so the owner is an ANCESTOR of this process, not
    this process. Every CPU-lane measurement takes gpu-quiet SHARED with its regions,
    so an exclusive hold spanning the device interval is the GPU-only run's host
    resource receipt. This never acquires anything: it records the kernel's lock
    rows (waiters excluded), each owner's mode, start ticks and relation to us.
    """
    pid = os.getpid()
    result = {"path": str(lock_path), "device": None, "inode": None,
              "path_unchanged": False, "owners": [], "error": None}
    try:
        before = Path(lock_path).stat()
        with Path(proc_locks).open("r", encoding="utf-8") as stream:
            raw = stream.read((1 << 20) + 1)
        if len(raw) > 1 << 20:
            raise ClaimRefused("kernel lock observation exceeded byte bound")
        ancestors = _ancestor_pids(pid)
        for line in raw.splitlines():
            fields = line.split()
            if "->" in fields or len(fields) < 6 or fields[1] != "FLOCK":
                continue
            try:
                same = int(fields[5].rsplit(":", 1)[-1]) == before.st_ino
                owner = int(fields[4])
            except ValueError:
                continue
            if not same:
                continue
            result["owners"].append({
                "pid": owner, "start_ticks": _process_start_ticks(owner),
                "mode": "exclusive" if fields[3] == "WRITE" else "shared",
                "relation": ("self" if owner == pid else
                             "ancestor" if owner in ancestors else "other"),
                "kernel_row": " ".join(fields)})
        after = Path(lock_path).stat()
        result.update(device=before.st_dev, inode=before.st_ino,
                      path_unchanged=(after.st_dev, after.st_ino)
                      == (before.st_dev, before.st_ino))
    except (OSError, ValueError, ClaimRefused) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def gpu_quiet_exclusive_holder(observation: dict) -> dict | None:
    """The single exclusive owner (self or ancestor) of one observation, else None."""
    owners = observation.get("owners")
    if observation.get("error") is not None or observation.get("path_unchanged") is not True \
            or not isinstance(owners, list) or len(owners) != 1:
        return None
    owner = owners[0]
    if owner.get("mode") != "exclusive" or owner.get("relation") not in {"self", "ancestor"} \
            or type(owner.get("start_ticks")) is not int:
        return None
    return owner


class HeldCpuClaim(dict):
    """The existing acquired context, with read-only open/close observations.

    Dictionary serialization remains the old receipt. This live object only observes
    locks its owning context already acquired; it cannot acquire, restore or transfer
    ownership and is inactive after that context exits. A CPU context's yield lease
    (`CpuClaimLease`, `yield_lease`) is owned by `hold_cpu`, not by this receipt.
    """

    def __init__(self, receipt, lock_paths, *, region_fraction=0.0, affinity=(),
                 gpu_quiet_path=None):
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
        # GPU-only run (no CPU region): the launcher's gpu-quiet EXCLUSIVE hold is
        # observed beside the device flock at open and close (`observe_gpu_quiet`).
        self._gpu_quiet_path = None if gpu_quiet_path is None else Path(gpu_quiet_path)
        self._gpu_quiet = ({"open": observe_gpu_quiet(self._gpu_quiet_path), "close": None}
                           if self._gpu_quiet_path is not None else None)
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

    def gpu_quiet_open(self):
        """The open gpu-quiet observation, or None when this context does not carry one."""
        return None if self._gpu_quiet is None else self._gpu_quiet["open"]

    def _closing(self):
        self._closed = self.observe()
        if self._gpu_quiet is not None:
            self._gpu_quiet["close"] = observe_gpu_quiet(self._gpu_quiet_path)
        self._active = False

    def _close_observation_failed(self, error):
        self._active = False
        self._closed = {"status": "unavailable", "locks": [], "owner_pid": self._owner_pid,
                        "error": f"{type(error).__name__}: {error}"}
        if self._gpu_quiet is not None and self._gpu_quiet["close"] is None:
            self._gpu_quiet["close"] = {"path": str(self._gpu_quiet_path), "device": None,
                "inode": None, "path_unchanged": False, "owners": [],
                "error": f"{type(error).__name__}: {error}"}

    def _released_now(self):
        self._ended_at = time.monotonic()
        self._released = True

    def retained_interval(self):
        """Closed original owner facts only; not execution permission or a replay grant."""
        if not self._released or self._ended_at is None or self._closed is None:
            raise ClaimRefused("original claim has not completed its owning release")
        locks = self._opened["locks"]
        quiet = ({} if self._gpu_quiet is None
                 else {"gpu_quiet": {"open": self._gpu_quiet["open"],
                                     "close": self._gpu_quiet["close"]}})
        return {**quiet, "context_id": self._context_id, "domain": dict(self._domain),
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
def hold(lock_path: Path | None = None, *, device_id: str = DEVICE_ID,
         gpu_quiet_path: Path | None = None) -> Iterator[dict]:
    """Hold an exclusive claim for the whole window, or refuse.

    Non-blocking on purpose: a loop that waits on a lock behind an unknown holder is
    a loop that looks alive while doing nothing. Refusing tells the operator the
    device is busy, which is a fact worth surfacing.

    `gpu_quiet_path` (GPU-only runs, `--gpu-cpu-region-claim off`): also observe the
    launcher's gpu-quiet EXCLUSIVE hold at open and close; it is retained as the
    host-side resource receipt of a run that holds no CPU region.
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
                               "pid": os.getpid()}, [lock_path],
                               gpu_quiet_path=gpu_quiet_path)
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


def _gpu_quiet_module():
    """The orchestrator's gpu-quiet owner (`src/runtime/gpu_quiet_lock.py`), or refuse."""
    _ensure_orchestrator_importable()
    try:
        import importlib
        gpu_quiet_lock = importlib.import_module("src.runtime.gpu_quiet_lock")
    except ImportError as exc:
        raise ClaimRefused(
            "the resolved orchestrator checkout has no gpu-quiet lock "
            "(src/runtime/gpu_quiet_lock.py); update it, or pass "
            "--cpu-measurement-gpu-quiet off only with the operator's leave") from exc
    return gpu_quiet_lock


@contextmanager
def hold_cpu(cpu_list: str, *, gpu_quiet: bool = False) -> Iterator[dict]:
    """Use the installed orchestrator's physical region owner, never a new flock.

    `gpu_quiet=True` (a CPU MEASUREMENT run under `--cpu-measurement-gpu-quiet lock`)
    also holds the orchestrator's host-wide gpu-quiet lock SHARED, acquired BEFORE the
    regions and released after them, by the orchestrator's own combined owner
    (`gpu_quiet_then_regions`: it backs off gpu-quiet while the regions are busy). The
    yield lease releases and re-acquires both together, so a GPU bench holding
    gpu-quiet EXCLUSIVE fits into this run's actor phases and never into its tails.
    A GPU run's own host-CPU claim passes False: its measurements take gpu-quiet
    EXCLUSIVE (`hold_gpu_quiet_measurement`), which a shared hold of its own would
    self-deadlock (flock conflicts between open file descriptions of one process).
    """
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
    quiet = _gpu_quiet_module() if gpu_quiet else None

    def provider(timeout_s, cancel_check=None):
        """The owner context: the regions, preceded by gpu-quiet SHARED when asked."""
        if quiet is None:
            return cpu_region_lock("autokernel-cpu", regions, timeout_s=timeout_s,
                                   cancel_check=cancel_check,
                                   request_tag="autokernel-experimental-serving")
        return _regions_of(quiet.gpu_quiet_then_regions(
            "autokernel-cpu", regions, gpu_quiet=quiet.GPU_QUIET_SHARED, timeout_s=timeout_s,
            cancel_check=cancel_check, request_tag="autokernel-experimental-serving"))

    receipt = None
    lease = None
    with ExitStack() as release:
        owner = provider(1.0)
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
        body = {"device_id": "cpu", "cpu_list": cpu_list, "regions": sorted(held),
                "lock_paths": {key: str(value) for key, value in held.items()},
                "pid": os.getpid()}
        if quiet is not None:
            # Only when taken: a receipt without gpu-quiet stays byte-identical. The
            # shared flock is not in the observed lock paths -- `observe` proves
            # sole WRITE ownership, and a shared lock has co-owners by design.
            body["gpu_quiet"] = {"mode": quiet.GPU_QUIET_SHARED,
                                 "lock_path": str(quiet.gpu_quiet_lock_path())}
        receipt = HeldCpuClaim(body,
               [*held.values(), *(global_region_lock_path(region) for region in sorted(held))],
               region_fraction=len(regions) / len(ATOMIC_REGIONS),
               affinity=tuple(str(cpu) for cpu in sorted(_cpu_numbers(cpu_list))))
        lease = CpuClaimLease(
            receipt, slot, regions=regions,
            acquire=provider,
            preflight=lambda: _preflight(strict=True))
        receipt._yield_lease = lease
        yield receipt


@contextmanager
def _regions_of(combined):
    """Present the combined gpu-quiet+regions grant as the plain `{region: Path}` the
    region owner yields, so the receipt and the lease compare the same lock files."""
    with combined as grant:
        yield grant["regions"]


#: The role a GPU measurement holds gpu-quiet EXCLUSIVE under (operator 2026-10-04:
#: GPU benches and CPU measurements never overlap). GPU host threads run on 184-191
#: and a pinned GPU bench raised a CPU A/A floor 0.80% -> 7.22%, 9x (INF-70,
#: 2026-09-08). The lock is host-wide and region-free: it replaces the q3 region claim
#: GPU measurements used to take, which also 503'd every serving role placed on q3.
#: Measurement-only: production :8083 serving and orchestrator per-call claims never
#: take it.
GPU_QUIET_ROLE = "autokernel-gpu-measurement"


def gpu_quiet_preflight() -> Path:
    """Everything `hold_gpu_quiet_measurement` needs short of the lock: the orchestrator
    importable with its gpu-quiet owner. Raises ClaimRefused so a GPU run refuses at
    STARTUP, not at its first measurement after a build. Returns the lock path."""
    return _gpu_quiet_module().gpu_quiet_lock_path()


@contextmanager
def hold_gpu_quiet_measurement(timeout_s: float = 1.0) -> Iterator[dict]:
    """Hold gpu-quiet EXCLUSIVE through the orchestrator's owner for ONE GPU measurement.

    Raises the provider's `CpuRegionLockTimeout` while a CPU measurement holds it
    shared (`region_lock_busy` recognises it); the caller waits and retries between
    attempts, never pre-empting the holder. Takes no CPU region.
    """
    quiet = _gpu_quiet_module()
    with quiet.gpu_quiet_lock(quiet.GPU_QUIET_EXCLUSIVE, role=GPU_QUIET_ROLE,
                              timeout_s=timeout_s,
                              request_tag="autokernel-gpu-measurement") as record:
        yield {"device_id": "host", "purpose": "gpu_measurement_quiet",
               "gpu_quiet": quiet.GPU_QUIET_EXCLUSIVE, "lock_path": record["lock_path"],
               "pid": os.getpid()}


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


__all__ = ["ClaimRefused", "CpuClaimLease", "DEVICE_ID", "DEVICE_LOCK", "GPU_QUIET_ROLE",
           "gpu_quiet_preflight", "hold", "hold_cpu", "hold_gpu_quiet_measurement",
           "region_lock_busy", "yield_lease"]
