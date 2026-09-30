"""Per-stage routing-decision latency for live ``/chat`` requests (RI-16).

A routed ``/chat`` request makes several embed+KNN lookups before and after the
answer is generated, and none of them was timed. This module times each routing
stage with ``time.perf_counter`` and keeps the result in a request-scoped
``ContextVar``: instrumentation only, no behaviour change, no I/O. The numbers reach
disk through the progress JSONL writes that already happen:

* ``routing_decision`` event — ``routing_path`` + ``stage_ms`` snapshot taken when the
  pre-execution route is decided (``route_total`` and the stages inside it).
* ``task_completed`` / ``task_failed`` event — the same two keys, final: adds ``mode``,
  ``routing_context``, ``review_gate``, ``review_verdict`` and the final ``total``.

``stage_ms`` always carries every key in :data:`STAGE_KEYS` (stable schema). A stage
that did not run on this request is ``None`` — never ``0``. Stages that can run more
than once per request (the review gate, the verdict) accumulate.

Which embed+KNN lookup each stage contains:

* ``priors``          — ``_heuristic_role_priors`` -> ``classify_and_route``
                        (``ClassificationRetriever`` when ``use_memrl``).
* ``route``           — ``select_initial_route`` -> ``HybridRouter.route`` (episodic KNN),
                        or the rules classifier again when there is no learned router.
* ``mode``            — ``_select_mode`` -> ``HybridRouter.route_with_mode`` (KNN).
* ``routing_context`` — streaming turn 0 ``build_routing_context`` (KNN).
* ``review_gate``     — ``_should_review`` (KNN over the ANSWER text).
* ``review_verdict``  — the architect verdict LLM call the gate triggers. Reported, but
                        NOT part of ``total``: it is generation, not a routing decision.

``total`` = ``route_total`` + ``mode`` + ``routing_context`` + ``review_gate``, so it is
always >= the sum of the sequential stages it covers (``route_total`` is the wall time
of the whole pre-execution routing function, which contains ``memrl_init``,
``priors``, ``route``, ``xmas``, ``factual_risk``, ``failure_veto``, ``difficulty`` and
``trinity``).
"""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterator, TypeVar

#: Every stage key, in pipeline order. The emitted ``stage_ms`` always has all of them.
STAGE_KEYS: tuple[str, ...] = (
    "memrl_init",
    "priors",
    "route",
    "xmas",
    "factual_risk",
    "failure_veto",
    "difficulty",
    "trinity",
    "route_total",
    "mode",
    "routing_context",
    "review_gate",
    "review_verdict",
    "total",
)

#: Stages that sum into ``total`` (``review_verdict`` is generation, excluded).
TOTAL_COMPONENTS: tuple[str, ...] = ("route_total", "mode", "routing_context", "review_gate")

#: ``routing_path`` values: the non-streaming pipeline, the unified streaming adapter,
#: and the legacy inline streaming generator.
PATH_CHAT = "chat"
PATH_UNIFIED_STREAM = "unified_stream"
PATH_LEGACY_STREAM = "legacy_stream"

_F = TypeVar("_F", bound=Callable[..., Any])
_R = TypeVar("_R")


class RoutingStageTiming:
    """Stage timings for one request. Mutated only by the request that owns it."""

    __slots__ = ("task_id", "path", "_ms")

    def __init__(self, task_id: str, path: str) -> None:
        self.task_id = task_id
        self.path = path
        self._ms: dict[str, float | None] = dict.fromkeys(STAGE_KEYS)

    def record(self, stage: str, elapsed_ms: float, *, accumulate: bool = False) -> None:
        if stage not in self._ms or stage == "total":
            return
        prior = self._ms[stage]
        if accumulate and prior is not None:
            elapsed_ms += prior
        self._ms[stage] = elapsed_ms
        self._ms["total"] = sum(self._ms[key] or 0.0 for key in TOTAL_COMPONENTS)

    def stage_ms(self) -> dict[str, float | None]:
        """A rounded copy — safe to put in a buffered progress entry."""
        return {
            key: (round(value, 3) if value is not None else None)
            for key, value in self._ms.items()
        }

    def telemetry(self) -> dict[str, Any]:
        return {"routing_path": self.path, "stage_ms": self.stage_ms()}


_CURRENT: ContextVar[RoutingStageTiming | None] = ContextVar(
    "routing_stage_timing", default=None
)


def begin(task_id: str, path: str) -> RoutingStageTiming:
    """Start timing for the request running in the current context."""
    timing = RoutingStageTiming(task_id, path)
    _CURRENT.set(timing)
    return timing


def current() -> RoutingStageTiming | None:
    return _CURRENT.get()


def telemetry_for(task_id: str) -> dict[str, Any] | None:
    """``{"routing_path", "stage_ms"}`` for ``task_id`` if it is the timed request."""
    timing = _CURRENT.get()
    if timing is None or timing.task_id != task_id:
        return None
    return timing.telemetry()


def clear() -> None:
    """Drop the current context's timing (tests)."""
    _CURRENT.set(None)


@contextmanager
def timed(stage: str, *, accumulate: bool = False) -> Iterator[None]:
    """Time the enclosed block into ``stage``; a no-op when no request is timed."""
    timing = _CURRENT.get()
    if timing is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        timing.record(stage, (time.perf_counter() - start) * 1000.0, accumulate=accumulate)


def call_timed(
    stage: str, fn: Callable[..., _R], /, *args: Any, accumulate: bool = False, **kwargs: Any
) -> _R:
    """``fn(*args, **kwargs)`` timed into ``stage`` — usable inside an expression.

    The caller passes the function it already references, so a test that patches that
    name at the call site still takes effect.
    """
    timing = _CURRENT.get()
    if timing is None:
        return fn(*args, **kwargs)
    start = time.perf_counter()
    try:
        return fn(*args, **kwargs)
    finally:
        timing.record(stage, (time.perf_counter() - start) * 1000.0, accumulate=accumulate)


def timed_stage(stage: str, *, accumulate: bool = False) -> Callable[[_F], _F]:
    """Decorator form of :func:`timed` (one ``ContextVar.get`` when untimed)."""

    def decorate(fn: _F) -> _F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            timing = _CURRENT.get()
            if timing is None:
                return fn(*args, **kwargs)
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                timing.record(
                    stage, (time.perf_counter() - start) * 1000.0, accumulate=accumulate
                )

        return wrapper  # type: ignore[return-value]

    return decorate
