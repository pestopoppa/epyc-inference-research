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

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
from typing import Iterator

DEVICE_LOCK = Path("/mnt/raid0/llm/tmp/gpu_device.mi210_0.lock")
DEVICE_ID = "mi210_0"


class ClaimRefused(RuntimeError):
    """The device is held by someone else, or the claim did not survive the window."""


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

    receipt = {"device_id": device_id, "lock_path": str(lock_path), "pid": os.getpid()}
    try:
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
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


@contextmanager
def hold_cpu(cpu_list: str) -> Iterator[dict]:
    """Use the installed orchestrator's physical region owner, never a new flock."""
    os.environ.setdefault("ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT", "1")
    if os.environ["ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT"].lower() not in {
            "1", "true", "yes", "on"}:
        raise ClaimRefused("CPU run requires the existing cross-role region mutex")
    from src.runtime.cpu_region_lock import cpu_region_lock
    from src.runtime.instance_topology import cpu_list_to_regions
    from src.runtime.region_lock_cli import _preflight

    reason = _preflight(strict=True)
    if reason:
        raise ClaimRefused(reason)
    regions = cpu_list_to_regions(cpu_list)
    if not regions:
        raise ClaimRefused("CPU affinity maps to no physical regions")
    with cpu_region_lock("autokernel-cpu", regions, timeout_s=1.0,
                         request_tag="autokernel-experimental-serving") as held:
        if set(held) != set(regions):
            raise ClaimRefused("CPU owner did not acquire the requested physical regions")
        yield {"device_id": "cpu", "cpu_list": cpu_list, "regions": sorted(held),
               "lock_paths": {key: str(value) for key, value in held.items()},
               "pid": os.getpid()}


__all__ = ["ClaimRefused", "DEVICE_ID", "DEVICE_LOCK", "hold", "hold_cpu"]
