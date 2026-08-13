"""Cross-process exclusion for the brief interval in which a model is resident.

CPU and GPU AutoKernel work may prepare candidates concurrently.  Their actual
model load/inference calls must not overlap, however: GPU loading perturbs the
same memory system used by the CPU benchmark.  This module provides that one
narrow mutex.  It is deliberately independent of CPU-region and device claims;
those describe *where* a call may run, while this lock describes *when* either
backend may make a model resident.

The flock is the sole occupancy and liveness fact.  The lock file is never
unlinked (avoiding split-inode locks), carries no holder payload that could go
stale, and the kernel releases it if a holder exits or is killed.
"""
from __future__ import annotations

import fcntl
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence


DEFAULT_LOCK_PATH = Path("/mnt/raid0/llm/ak-claims/inference-call-window.lock")
DEFAULT_POLL_S = 0.010
WINDOWED_CPU_ROLE = "autokernel-windowed-controls"


class InferenceWindowTimeout(TimeoutError):
    """The model-call mutex remained occupied for the caller's whole budget."""


@dataclass
class InferenceWindowLease:
    """One held flock.  Closing/releasing is idempotent and never unlinks it."""

    path: Path
    fd: int
    waited_s: float
    acquired_monotonic_s: float
    _released: bool = False

    @property
    def held(self) -> bool:
        return not self._released

    def release(self) -> None:
        if self._released:
            return
        try:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            os.close(self.fd)
            self._released = True

    def __enter__(self) -> "InferenceWindowLease":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.release()


class InferenceCallWindow:
    """Acquire a recoverable host-wide mutex around one model invocation.

    Acquisition polls with ``LOCK_NB`` so callers may bound how long they wait.
    No stale-owner reclamation exists because none is necessary: ``flock`` is
    attached to the open file description and disappears with the dead process.
    """

    def __init__(self, path: Path | str = DEFAULT_LOCK_PATH, *,
                 timeout_s: float | None = None, poll_s: float = DEFAULT_POLL_S,
                 monotonic: Callable[[], float] = time.monotonic,
                 wait: Callable[[float], None] = time.sleep) -> None:
        self.path = Path(path)
        if timeout_s is not None and timeout_s < 0:
            raise ValueError("timeout_s must be non-negative or None")
        if poll_s <= 0:
            raise ValueError("poll_s must be positive")
        self.timeout_s = timeout_s
        self.poll_s = float(poll_s)
        self._monotonic = monotonic
        self._wait = wait

    def acquire(self) -> InferenceWindowLease:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o660)
        started = self._monotonic()
        deadline = None if self.timeout_s is None else started + self.timeout_s
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = self._monotonic()
                    return InferenceWindowLease(
                        path=self.path, fd=fd, waited_s=max(0.0, acquired - started),
                        acquired_monotonic_s=acquired)
                except BlockingIOError:
                    now = self._monotonic()
                    if deadline is not None and now >= deadline:
                        raise InferenceWindowTimeout(
                            f"inference-call window {self.path} remained occupied for "
                            f"{now - started:.3f}s")
                    remaining = None if deadline is None else max(0.0, deadline - now)
                    self._wait(self.poll_s if remaining is None
                               else min(self.poll_s, remaining))
        except BaseException:
            os.close(fd)
            raise

    @contextmanager
    def hold(self) -> Iterator[InferenceWindowLease]:
        lease = self.acquire()
        try:
            yield lease
        finally:
            lease.release()


class WindowedSpawner:
    """Apply the mutex to one delegate spawn, not to setup around the runner.

    ``spawner_id`` is preserved because MicrobenchRunner uses it to require the
    live-process integrity checks.  Wrapping a live spawner must not make it
    look like a fixture/replay spawner.
    """

    def __init__(self, delegate: Any, call_window: InferenceCallWindow) -> None:
        if not callable(getattr(delegate, "run", None)):
            raise TypeError("delegate must expose run(argv, env, timeout_s=, cwd=)")
        if not isinstance(call_window, InferenceCallWindow):
            raise TypeError("call_window must be an InferenceCallWindow")
        self._delegate = delegate
        self._call_window = call_window
        self.spawner_id = str(getattr(delegate, "spawner_id", "windowed/unknown"))

    def run(self, argv: Sequence[str], env: Mapping, *, timeout_s: float,
            cwd: str | None = None) -> Any:
        with self._call_window.hold() as lease:
            result = self._delegate.run(argv, env, timeout_s=timeout_s, cwd=cwd)
        receipt = {
            "schema": "epyc.autokernel.inference_call_window.v1",
            "lock_path": str(lease.path),
            "waited_s": lease.waited_s,
            "held_s": max(0.0, self._call_window._monotonic()
                          - lease.acquired_monotonic_s),
            "scope": "model_load_and_inference_only",
            "released": lease.held is False,
        }
        # SpawnResult is frozen, so attach the per-invocation proof by
        # replacement.  Generic delegates used by narrow lock tests remain
        # valid and simply return their own result shape unchanged.
        try:
            return replace(result, inference_window_receipt=receipt)
        except (TypeError, ValueError):
            return result


class ReleaseUnderWindow:
    """Drain model calls before releasing the CPU coverage they may borrow."""

    def __init__(self, claim: Any, call_window: InferenceCallWindow) -> None:
        if not callable(getattr(claim, "release", None)):
            raise TypeError("claim must expose release()")
        self._claim = claim
        self._call_window = call_window

    def __enter__(self) -> Any:
        return self._claim

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        with self._call_window.hold():
            self._claim.release()


@dataclass
class BorrowedCpuCoverage:
    """Read-only proof that a window-aware CPU campaign covers GPU helpers."""

    cpu_list: str
    claim_id: str
    campaign_id: str
    holder_pid: int
    lock_paths: tuple[str, ...]
    payloads: tuple[dict, ...]
    _validator: Callable[[], None]

    borrowed: bool = True

    def validate(self) -> None:
        self._validator()

    def release(self) -> None:
        # We do not own the claim.  Validation at close proves its owner kept
        # the physical coverage until after this model process exited.
        self.validate()

    def to_dict(self) -> dict:
        return {
            "schema": "epyc.autokernel.borrowed_cpu_coverage.v1",
            "borrowed": True,
            "cpu_list": self.cpu_list,
            "claim_id": self.claim_id,
            "campaign_id": self.campaign_id,
            "holder_pid": self.holder_pid,
            "lock_paths": list(self.lock_paths),
            "windowed_role": WINDOWED_CPU_ROLE,
        }


def _locked(path: Path) -> bool:
    try:
        fd = os.open(path, os.O_RDWR | os.O_CLOEXEC)
    except OSError as exc:
        raise RuntimeError(f"CPU coverage lock unreadable: {path}: {exc}") from exc
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def borrow_windowed_cpu_coverage(cpu_list: str, *, lock_root: Path | str | None = None
                                 ) -> BorrowedCpuCoverage:
    """Borrow q-region coverage from a live-controls claimant while holding the window.

    The special role is the compatibility marker: legacy/current processes do
    not opt into the shared model-call mutex and therefore cannot be borrowed.
    Every role and GLOBAL flock is rechecked before and after the GPU call.
    """
    from . import cpu_region_claim  # lazy: keeps the small lock module acyclic

    root = (cpu_region_claim.default_region_lock_dir() if lock_root is None
            else Path(lock_root))
    regions = cpu_region_claim.cpu_list_to_regions(cpu_list)
    if not regions:
        raise RuntimeError(f"GPU helper footprint {cpu_list!r} resolves to no region")

    def read_and_validate() -> tuple[tuple[str, ...], tuple[dict, ...]]:
        paths: list[str] = []
        payloads: list[dict] = []
        expected: tuple[str, str, int] | None = None
        for region in regions:
            global_path = cpu_region_claim.global_region_lock_path(region, root)
            role_path = cpu_region_claim.region_lock_path(
                WINDOWED_CPU_ROLE, region, root)
            if not _locked(global_path) or not _locked(role_path):
                raise RuntimeError(
                    f"window-aware CPU coverage for {region} is not held")
            try:
                payload = json.loads(role_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"window-aware CPU coverage payload unreadable: {role_path}: {exc}") from exc
            holder = payload.get("holder")
            identity = (payload.get("claim_id"), payload.get("campaign_id"),
                        holder.get("pid") if isinstance(holder, dict) else None)
            purpose = str(payload.get("purpose", ""))
            if (payload.get("autokernel_schema")
                    != cpu_region_claim.CPU_REGION_CLAIM_SCHEMA
                    or payload.get("claim_role") != WINDOWED_CPU_ROLE
                    or payload.get("state") != cpu_region_claim.STATE_HELD
                    or not str(payload.get("campaign_id", "")).startswith("ak-")
                    or "AutoKernel" not in purpose
                    or region not in payload.get("regions", [])):
                raise RuntimeError(
                    f"{role_path} is not a borrowable window-aware AutoKernel claim")
            if expected is None:
                expected = identity
            elif identity != expected:
                raise RuntimeError("GPU helper regions belong to different CPU claims")
            liveness = cpu_region_claim.assess_holder_liveness(holder)
            if liveness.state != cpu_region_claim.LIVE:
                raise RuntimeError(
                    f"window-aware CPU claim holder is {liveness.state}: {liveness.reason}")
            paths.extend((str(global_path), str(role_path)))
            payloads.append(payload)
        return tuple(paths), tuple(payloads)

    paths, payloads = read_and_validate()
    first = payloads[0]

    def validator() -> None:
        current_paths, current = read_and_validate()
        if current_paths != paths or any(
                row.get("claim_id") != first["claim_id"] for row in current):
            raise RuntimeError("borrowed CPU coverage changed during the GPU model call")

    return BorrowedCpuCoverage(
        cpu_list=cpu_list, claim_id=first["claim_id"],
        campaign_id=first["campaign_id"], holder_pid=int(first["holder"]["pid"]),
        lock_paths=paths, payloads=payloads, _validator=validator)
