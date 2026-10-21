"""AKR-SG-1: load-driver self-check and survivorship rule.

A load generator that cannot start its requests when it says it will measures itself,
not the server. Every run therefore states (a) the requested versus achieved start rate and
(b) the distribution of client scheduling delay (actual start minus intended start), and a
run is VOIDED -- never averaged, never down-weighted -- when the median delay exceeds
`MAX_MEDIAN_DELAY_MS` or the achieved rate departs from the request by more than
`RATE_TOLERANCE` (intake-2330#record, inference-perf's generator self-check).

Survivorship: the headline metrics of a session replay are taken over ALL sessions started
(`session_headline`); failed and cancelled sessions count in the denominator and are
reported beside the headline, never dropped.

Additive. `serving._measure_once` consults this module only after `enable()`; with it off
(the default) the serving path is byte-identical.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import concurrent.futures as cf
import math
import statistics
import time

from .loop import MeasurementInvalid

SCHEMA = "epyc.autokernel.loadgen_selfcheck.v1"
MAX_MEDIAN_DELAY_MS = 10.0
RATE_TOLERANCE = 0.05

_ENABLED = False


def enable(on: bool = True) -> None:
    global _ENABLED
    _ENABLED = bool(on)


def enabled() -> bool:
    return _ENABLED


class LoadDriverVoided(MeasurementInvalid):
    """The generator failed its own self-check; the run is void, not a null."""


def _pct(sorted_vals: Sequence[float], q: float) -> float:
    return sorted_vals[min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1))))]


class SchedulingLog:
    """Intended versus actual start (monotonic seconds) of every request."""

    def __init__(self, requested_rate_hz: float | None = None):
        self.requested_rate_hz = requested_rate_hz
        self.rows: list[tuple[float, float]] = []

    def record(self, intended_s: float, actual_s: float) -> None:
        self.rows.append((float(intended_s), float(actual_s)))

    def report(self, *, max_median_delay_ms: float = MAX_MEDIAN_DELAY_MS,
               rate_tolerance: float = RATE_TOLERANCE,
               requested_n: int | None = None) -> dict:
        reasons: list[str] = []
        n = len(self.rows)
        delays = sorted(max(0.0, (a - i)) * 1000.0 for i, a in self.rows)
        dist = None
        if n:
            dist = {"n": n, "median_ms": statistics.median(delays),
                    "p95_ms": _pct(delays, 0.95), "max_ms": delays[-1]}
        else:
            reasons.append("no request start was recorded")
        if dist and dist["median_ms"] > max_median_delay_ms:
            reasons.append(f"median scheduling delay {dist['median_ms']:.3f} ms exceeds "
                           f"{max_median_delay_ms:.3f} ms")
        if requested_n is not None and n != requested_n:
            reasons.append(f"started {n} requests, requested {requested_n}")
        achieved = None
        requested = self.requested_rate_hz
        if requested is not None and n >= 2:
            span = max(a for _, a in self.rows) - min(a for _, a in self.rows)
            achieved = (n - 1) / span if span > 0 else math.inf
            if requested <= 0 or abs(achieved - requested) / requested > rate_tolerance:
                reasons.append(f"achieved rate {achieved:.4f}/s departs from requested "
                               f"{requested:.4f}/s by more than {rate_tolerance:.0%}")
        elif requested is not None:
            reasons.append("rate requested but fewer than 2 starts recorded")
        return {"schema": SCHEMA, "requested_rate_hz": requested,
                "achieved_rate_hz": None if achieved is None or math.isinf(achieved) else achieved,
                "scheduling_delay_ms": dist, "max_median_delay_ms": max_median_delay_ms,
                "rate_tolerance": rate_tolerance, "voided": bool(reasons),
                "void_reasons": reasons}

    def require_valid(self, **kw) -> dict:
        rep = self.report(**kw)
        if rep["voided"]:
            raise LoadDriverVoided("load driver self-check failed: " + "; ".join(rep["void_reasons"]),
                                   rep)
        return rep


def session_headline(sessions: Sequence[Mapping]) -> dict:
    """Headline over ALL sessions started. Each session: {"status": "ok"|"failed"|"cancelled",
    "success": bool}. Denominator is every session started; survivors-only is never offered."""
    started = len(sessions)
    failed = sum(1 for s in sessions if s.get("status") == "failed")
    cancelled = sum(1 for s in sessions if s.get("status") == "cancelled")
    solved = sum(1 for s in sessions if s.get("status", "ok") == "ok" and s.get("success"))
    return {"sessions_started": started, "failed": failed, "cancelled": cancelled,
            "solved": solved, "success_rate_over_started": (solved / started) if started else None,
            "denominator": "all sessions started"}


def run_open_loop(fn: Callable[[int], object], n: int, rate_hz: float, *,
                  inject_delay_s: float = 0.0, workers: int = 8,
                  clock=time.monotonic, sleep=time.sleep) -> tuple[list, SchedulingLog]:
    """Fire `fn(i)` at i/rate_hz after t0 on a thread pool, logging intended/actual start.
    `inject_delay_s` simulates a scheduler stall before each start (test and calibration)."""
    log = SchedulingLog(requested_rate_hz=rate_hz)
    t0 = clock() + 0.05

    def one(i: int):
        intended = t0 + i / rate_hz
        wait = intended - clock()
        if wait > 0:
            sleep(wait)
        if inject_delay_s:
            sleep(inject_delay_s)
        log.record(intended, clock())
        return fn(i)

    with cf.ThreadPoolExecutor(workers) as ex:
        out = list(ex.map(one, range(n)))
    return out, log
