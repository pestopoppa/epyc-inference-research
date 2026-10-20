#!/usr/bin/env python3
"""CS-48 controller event log (JSONL) and its structural validator.

One line per event: ``{"i": seq, "t_ns": int, "ev": name, ...}``; ``t_ns`` is nanoseconds since the run's t0 on the
run clock (the same instant sample 0 of every wav is anchored to). The validator is the first mutation gate: a log
whose sequence is not 0..n-1, whose clock goes backwards, or whose cancel hooks are missing, duplicated or out of the
controller's order is refused before any score is computed.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

HOOK_ORDER = ("queue", "generation", "vocoder", "orchestrator")


class LogRefused(ValueError):
    """The event log violates the CS-48 contract."""


class EventLog:
    def __init__(self, clock, t0_ns: int) -> None:
        self.clock, self.t0, self.events, self._lock = clock, t0_ns, [], threading.Lock()

    def emit(self, ev: str, **kw) -> dict:
        with self._lock:
            rec = {"i": len(self.events), "t_ns": self.clock.now_ns() - self.t0, "ev": ev, **kw}
            self.events.append(rec)
            return rec

    def write(self, path: str | Path) -> None:
        Path(path).write_text("".join(json.dumps(e, sort_keys=True, separators=(",", ":")) + "\n" for e in self.events))


def read_log(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate_log(events: list[dict]) -> dict:
    """Raise LogRefused on any structural violation; return {cancels: [per-cancel hook record]}."""
    if not events:
        raise LogRefused("empty event log")
    last = None
    for n, e in enumerate(events):
        if e.get("i") != n:
            raise LogRefused(f"event {n}: sequence index {e.get('i')} (missing or duplicated line)")
        t = e.get("t_ns")
        if not isinstance(t, int):
            raise LogRefused(f"event {n}: t_ns missing or not integer")
        if last is not None and t < last:
            raise LogRefused(f"event {n}: clock went backwards ({t} < {last})")
        last = t
    cancels, cur = [], None
    for e in events:
        if e["ev"] == "cancel_call":
            if cur is not None:
                raise LogRefused("cancel_call inside an unfinished cancel chain")
            cur = {"call": e, "hooks": {}}
        elif e["ev"] == "hook":
            if cur is None:
                raise LogRefused(f"hook {e.get('hook')} outside a cancel chain")
            name = e.get("hook")
            if name in cur["hooks"]:
                raise LogRefused(f"hook {name} duplicated within one cancel chain")
            expect = HOOK_ORDER[len(cur["hooks"])] if len(cur["hooks"]) < len(HOOK_ORDER) else None
            if name != expect:
                raise LogRefused(f"hook {name} out of order (expected {expect})")
            cur["hooks"][name] = e
        elif e["ev"] == "cancel_return":
            if cur is None:
                raise LogRefused("cancel_return without cancel_call")
            # controller.cancel attempts EVERY hook even when one raises; a missing hook entry is a defect
            got = tuple(cur["hooks"])
            late = got == HOOK_ORDER[:1] and cur["call"].get("turn_live") is False  # turn already retired: queue hook only
            if got != HOOK_ORDER and not late:
                raise LogRefused(f"cancel chain returned with hooks {got}, expected {HOOK_ORDER}")
            cur["late"] = late
            cur["ret"] = e
            cancels.append(cur)
            cur = None
    if cur is not None:
        raise LogRefused("cancel chain never returned")
    return {"cancels": cancels}
