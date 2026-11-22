"""CTC-2: stage admission between the serial supervisor and its target children.

The parent owns one `Admission` queue (no daemon, no external service). A child
asks for one stage at a time over a local control channel bound to the child
identity the parent captured at spawn; the grant says "you may now try to take
the physical claims", and `acquire()` then takes them transactionally in the
fixed order admission -> CPU regions (canonical order) -> devices, releasing any
partial set before reporting failure. The flock/provider claim stays the
ownership fact: a grant is never a substitute for it.

Fairness: FIFO among compatible requests. A newer request may start ahead of an
older waiting one only if it conflicts neither with running stages nor with any
OLDER waiting request, so repeated conflicting work cannot starve the oldest.
Each target has at most one outstanding request or running stage.
"""
from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

from . import stage_footprint as sf


class AdmissionRefused(RuntimeError):
    pass


@dataclass(frozen=True)
class ChildIdentity:
    """Captured by the parent at spawn (pid plus /proc start time); never self-declared."""
    pid: int
    start_ticks: int
    target: str


@dataclass
class _Request:
    seq: int
    child: ChildIdentity
    attempt: str
    footprint: sf.StageFootprint
    granted: bool = False
    event: threading.Event = field(default_factory=threading.Event)


class Admission:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seq = 0
        self._queue: list[_Request] = []      # waiting + running, in arrival order
        self._children: dict[tuple, ChildIdentity] = {}

    def register(self, child: ChildIdentity) -> None:
        with self._lock:
            self._children[(child.pid, child.start_ticks)] = child

    def _bound(self, child: ChildIdentity, footprint: sf.StageFootprint) -> None:
        known = self._children.get((child.pid, child.start_ticks))
        if known is None or known != child:
            raise AdmissionRefused("request from an unknown child identity")
        if footprint.target != child.target:
            raise AdmissionRefused("footprint target differs from the child's captured target")

    def request(self, child: ChildIdentity, attempt: str,
                footprint: sf.StageFootprint) -> _Request:
        with self._lock:
            self._bound(child, footprint)
            if any(r.child.target == child.target for r in self._queue):
                raise AdmissionRefused("target already has an outstanding stage")
            self._seq += 1
            req = _Request(self._seq, child, attempt, footprint)
            self._queue.append(req)
            self._schedule()
            return req

    def _schedule(self) -> None:
        running = [r for r in self._queue if r.granted]
        waiting_before: list[_Request] = []
        for req in self._queue:
            if req.granted:
                continue
            blockers = running + waiting_before
            if not any(sf.conflicts(req.footprint, other.footprint) for other in blockers):
                req.granted = True
                running.append(req)
                req.event.set()
            else:
                waiting_before.append(req)

    def release(self, req: _Request) -> None:
        with self._lock:
            if req in self._queue:
                self._queue.remove(req)
            self._schedule()

    def cancel_child(self, child: ChildIdentity) -> int:
        """Stop/crash: drop the child's pending and running entries."""
        with self._lock:
            mine = [r for r in self._queue if r.child == child]
            for req in mine:
                self._queue.remove(req)
            self._schedule()
            return len(mine)

    def status(self) -> list[dict]:
        with self._lock:
            return [{"seq": r.seq, "target": r.child.target, "kind": r.footprint.kind,
                     "attempt": r.attempt,
                     "state": "running" if r.granted else "waiting_resource"}
                    for r in self._queue]


def acquire(footprint: sf.StageFootprint, *, cpu_region: Callable[[str], object],
            device: Callable[[str], object]) -> list:
    """Take regions (canonical order) then devices; release the partial set on failure.

    Providers return objects with `release()`; a raising provider means unavailable."""
    held: list = []
    try:
        for region in footprint.regions:      # cores_to_regions already sorts canonically
            held.append(cpu_region(region))
        for dev in sorted(footprint.devices):
            held.append(device(dev))
    except BaseException:
        for item in reversed(held):
            try:
                item.release()
            except Exception:
                pass
        raise
    return held


@contextmanager
def stage(admission: Admission, child: ChildIdentity, attempt: str,
          footprint: sf.StageFootprint, *, cpu_region, device,
          timeout_s: Optional[float] = None) -> Iterator[list]:
    """stage(kind, target, attempt, footprint): yield currently held claim objects.

    The stage timeout starts at admission, so `timeout_s` bounds only the wait."""
    req = admission.request(child, attempt, footprint)
    try:
        if not req.event.wait(timeout_s):
            raise AdmissionRefused("stage admission wait timed out")
        held = acquire(footprint, cpu_region=cpu_region, device=device)
        try:
            yield held
        finally:
            for item in reversed(held):
                item.release()
    finally:
        admission.release(req)


# ---- parent/child control channel (newline-delimited JSON on a socketpair) ------

MESSAGES = frozenset({"request", "grant", "release", "cancel", "status", "refused"})


def encode(kind: str, **body) -> bytes:
    if kind not in MESSAGES:
        raise ValueError(f"unknown control message {kind!r}")
    return (json.dumps({"kind": kind, **body}, sort_keys=True) + "\n").encode()


def decode(line: bytes) -> dict:
    msg = json.loads(line)
    if not isinstance(msg, dict) or msg.get("kind") not in MESSAGES:
        raise AdmissionRefused("malformed control message")
    return msg
