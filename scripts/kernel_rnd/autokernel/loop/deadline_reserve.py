"""Cleanup reserve: stop dispatching while a granted window still has time to finish.

The launcher exports FILLER_DEADLINE (epoch; start no new work after it). `run.py` folds
`tripped` into `should_stop`, so the pool declines new iterations at the next boundary
and the existing stop path does the drain, publishing and teardown inside the window.
The run ends with the resumable terminal `deadline_parked`.

Reserve: AK_CLEANUP_RESERVE_S when set, else the p95 of held tail seconds recorded in
this store (at least MIN_SAMPLES), else 0, which stops exactly at FILLER_DEADLINE.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import time
from typing import Callable, Iterable, Mapping

from . import tail_timing

ENV_DEADLINE = "FILLER_DEADLINE"
ENV_RESERVE = "AK_CLEANUP_RESERVE_S"
MIN_SAMPLES = 5


def p95_tail_held(payloads: Iterable[Mapping | None]) -> float | None:
    held = []
    for payload in payloads:
        timing = payload.get(tail_timing.FIELD) if isinstance(payload, Mapping) else None
        sessions = timing.get("tail_sessions") if isinstance(timing, Mapping) else None
        for session in sessions if isinstance(sessions, list) else ():
            value = session.get("held_s") if isinstance(session, Mapping) else None
            if isinstance(value, (int, float)):
                held.append(float(value))
    if len(held) < MIN_SAMPLES:
        return None
    held.sort()
    return held[min(len(held) - 1, math.ceil(0.95 * len(held)) - 1)]


def resolve_reserve_s(store: Path | None, env: Mapping[str, str] = os.environ) -> float:
    raw = env.get(ENV_RESERVE)
    if raw is not None:
        value = float(raw)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{ENV_RESERVE} must be a finite non-negative number")
        return value
    try:
        return p95_tail_held(tail_timing.read_store(Path(store))) or 0.0
    except Exception:  # noqa: BLE001 -- an unreadable store means no measured default
        return 0.0


class DeadlineReserve:
    """Trips once the granted deadline minus the reserve has passed; latches."""

    def __init__(self, deadline: float | None, reserve_s: float = 0.0,
                 clock: Callable[[], float] = time.time) -> None:
        self.deadline, self.reserve_s, self._clock = deadline, reserve_s, clock
        self._tripped = False

    @classmethod
    def from_env(cls, store: Path | None, env: Mapping[str, str] = os.environ,
                 clock: Callable[[], float] = time.time) -> "DeadlineReserve":
        raw = env.get(ENV_DEADLINE)
        if not raw:
            return cls(None, clock=clock)
        return cls(float(raw), resolve_reserve_s(store, env), clock)

    @property
    def tripped(self) -> bool:
        if self.deadline is not None and self._clock() >= self.deadline - self.reserve_s:
            self._tripped = True
        return self._tripped
