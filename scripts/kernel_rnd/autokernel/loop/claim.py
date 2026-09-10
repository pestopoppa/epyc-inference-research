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
import time
from typing import Iterator

DEVICE_LOCK = Path("/mnt/raid0/llm/tmp/gpu_device.mi210_0.lock")
DEVICE_ID = "mi210_0"


class ClaimRefused(RuntimeError):
    """The device is held by someone else, or the claim did not survive the window."""


class HeldCpuClaim(dict):
    """The existing acquired context, with read-only open/close observations.

    Dictionary serialization remains the old receipt. This live object only observes
    locks its owning context already acquired; it cannot acquire, restore or transfer
    ownership and is inactive after that context exits.
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
def hold(lock_path: Path = DEVICE_LOCK, *, device_id: str = DEVICE_ID) -> Iterator[dict]:
    """Hold an exclusive claim for the whole window, or refuse.

    Non-blocking on purpose: a loop that waits on a lock behind an unknown holder is
    a loop that looks alive while doing nothing. Refusing tells the operator the
    device is busy, which is a fact worth surfacing.
    """
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
    with ExitStack() as release:
        owner = cpu_region_lock("autokernel-cpu", regions, timeout_s=1.0,
                                request_tag="autokernel-experimental-serving")
        held = owner.__enter__()

        def close_original(*error):
            try:
                if receipt is not None:
                    try:
                        receipt._closing()
                    except Exception as observation_error:
                        receipt._close_observation_failed(observation_error)
            finally:
                result = owner.__exit__(*error)
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
        yield receipt


def _cpu_numbers(cpu_list):
    from ..execution.cpu_region_claim import parse_cpu_list
    return parse_cpu_list(cpu_list)


__all__ = ["ClaimRefused", "DEVICE_ID", "DEVICE_LOCK", "hold", "hold_cpu"]
