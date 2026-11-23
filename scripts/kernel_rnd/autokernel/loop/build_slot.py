"""AKU-20: a build runs in a PINNED slot, with BOUNDED jobs, under region lock role `build`,
and at most `cap` builds per lane at once.

Parent/child rule: this takes the region lock itself. The caller must not already hold a region
the build's cores map to (the orchestrator raises NestedLockError for a descendant under its own
ancestor); `claim.hold_cpu` yields before it spawns children, and a build-role claim on cores
DISJOINT from a bench grant nests cleanly.
"""
from contextlib import ExitStack, contextmanager
import errno
import fcntl
import os
from pathlib import Path
import re
import time

from . import claim

ROLE = "build"
DEFAULT_LANE_CAP = 1
SLOT_DIR_ENV, LANE_CAP_ENV = "AK_BUILD_SLOT_DIR", "AK_BUILD_LANE_CAP"
_DEFAULT_DIR = "/mnt/raid0/llm/tmp/ak-build-slots"


class BuildSlotRefused(claim.ClaimRefused):
    """The build is not pinned/bounded."""


class BuildSlotBusy(BuildSlotRefused):
    """The lane's concurrency cap is reached (retryable, like a busy region)."""


def cpu_count(cpu_list):
    """Logical CPUs named by a taskset-style list, or 0 when empty/unparsable."""
    cpus = set()
    for part in filter(None, str(cpu_list or "").replace(" ", "").split(",")):
        if not re.fullmatch(r"\d+(-\d+)?", part):
            return 0
        low, _, high = part.partition("-")
        if int(high or low) < int(low):
            return 0
        cpus.update(range(int(low), int(high or low) + 1))
    return len(cpus)


def bound_jobs(jobs, cpu_list):
    """`-j` for a pinned build: at least 1, never more than the pinned CPUs. Unpinned refuses."""
    n = cpu_count(cpu_list)
    if not n:
        raise BuildSlotRefused(f"build is not pinned: cpu_list {cpu_list!r} names no CPUs")
    if type(jobs) is not int or jobs < 1:
        raise BuildSlotRefused(f"build jobs {jobs!r} is not a positive integer")
    return min(jobs, n)


def plan(cpu_list, jobs):
    """The pinned, bounded build command parts: taskset prefix and `-j` value."""
    return {"prefix": ("taskset", "-c", cpu_list), "jobs": bound_jobs(jobs, cpu_list)}


@contextmanager
def lane_slot(lane, cap=None, *, directory=None, timeout_s=0.0):
    """One of `cap` per-lane flock slots; BuildSlotBusy when all stay taken for `timeout_s` (0 = fail fast)."""
    cap = int(os.environ.get(LANE_CAP_ENV, DEFAULT_LANE_CAP)) if cap is None else cap
    if type(cap) is not int or cap < 1:
        raise BuildSlotRefused(f"lane cap {cap!r} must be a positive integer")
    root = Path(directory or os.environ.get(SLOT_DIR_ENV, _DEFAULT_DIR))
    root.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(lane)) or "lane"
    deadline = time.monotonic() + timeout_s
    while True:
        with ExitStack() as stack:
            for index in range(cap):
                handle = stack.enter_context(open(root / f"{name}.{index}.lock", "a+"))
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    if exc.errno not in (errno.EWOULDBLOCK, errno.EAGAIN):
                        raise
                    continue
                yield index
                return
        if time.monotonic() >= deadline:
            raise BuildSlotBusy(f"lane {lane!r} already runs {cap} build(s): per-lane cap reached")
        time.sleep(0.1)


@contextmanager
def build_slot(cpu_list, jobs, *, lane, cap=None, directory=None, on_acquired=None, timeout_s=0.0):
    """Lane cap, then the `build`-role region claim on `cpu_list`; yields the claim receipt; `plan()` gives the bounded jobs."""
    bound_jobs(jobs, cpu_list)
    with lane_slot(lane, cap, directory=directory, timeout_s=timeout_s):
        with claim.hold_cpu(cpu_list, role=ROLE, on_acquired=on_acquired) as receipt:
            yield receipt
