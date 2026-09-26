"""Is a guarded serving instance decoding right now? (REPL-EMB-1.4)

Two readers, combined in the policy's order:

``ledger``  The orchestrator's own dispatch ledger: the per-region CPU locks
            (``src.runtime.cpu_region_lock``) an instance holds for the whole of a
            dispatched inference, resolved to exact instances by
            ``active_region_holder_instances``. Cross-process (it reads /proc/locks), cheap,
            no HTTP. It is POSITIVE-ONLY: traffic that bypasses the orchestrator (scouts,
            scripts, a measurement gate) never takes a lock, so "no holder" proves nothing.
            (DAR-LAT-1's ``SlotCapacity`` snapshot will be the in-process ledger once it is
            built; it plugs in here as another positive source.)
``slots``   The instance's live ``GET /slots`` (``is_processing``), parsed by the same
            ``parse_slots`` the scout stage's admission uses. Sees every client. Authoritative
            both ways; an unreadable ``/slots`` is UNKNOWN, never idle.

A port the ledger already marks busy is not probed.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from src.embedding_pool.topology import GuardedInstance

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BusySnapshot:
    busy: frozenset[int] = frozenset()
    unknown: frozenset[int] = frozenset()
    source_by_port: Mapping[int, str] = field(default_factory=dict)
    taken_at: float = 0.0

    def is_busy(self, port: int, *, unknown_is_busy: bool) -> bool:
        return port in self.busy or (unknown_is_busy and port in self.unknown)

    def as_dict(self) -> dict[str, Any]:
        return {
            "busy": sorted(self.busy),
            "unknown": sorted(self.unknown),
            "source_by_port": {str(k): v for k, v in sorted(self.source_by_port.items())},
        }


class BusySource(Protocol):
    name: str

    async def read(self, guarded: Sequence[GuardedInstance], http: Any) -> dict[int, bool | None]:
        """port -> True (busy) / False (idle, authoritative) / None (no opinion)."""


class LedgerBusySource:
    """Positive-only reader of the orchestrator's region-lock dispatch ledger."""

    name = "ledger"

    def __init__(self, holders_fn: Callable[[], Mapping[str, Iterable[int]]] | None = None) -> None:
        self._holders_fn = holders_fn

    def _holders(self) -> Mapping[str, Iterable[int]]:
        if self._holders_fn is not None:
            return self._holders_fn()
        from src.runtime.cpu_region_lock import active_region_holder_instances

        return active_region_holder_instances()

    async def read(self, guarded: Sequence[GuardedInstance], http: Any) -> dict[int, bool | None]:
        try:
            holders = await asyncio.to_thread(self._holders)
        except Exception as exc:  # noqa: BLE001 — an unreadable ledger has no opinion
            log.debug("embedding pool: dispatch ledger unreadable: %s", exc)
            return {g.port: None for g in guarded}
        held = {(role, int(i)) for role, idxs in (holders or {}).items() for i in idxs}
        return {g.port: (True if (g.role, g.instance_idx) in held else None) for g in guarded}


class SlotsBusySource:
    """Live ``/slots``: busy iff any slot ``is_processing``."""

    name = "slots"

    def __init__(self, timeout_s: float = 0.5) -> None:
        self.timeout_s = float(timeout_s)

    async def _one(self, g: GuardedInstance, http: Any) -> bool | None:
        from src.backends.context_limits import parse_slots

        try:
            resp = await http.get(f"{g.url}/slots", timeout=self.timeout_s)
            if resp.status_code != 200:
                return None
            occ = parse_slots(g.url, resp.json())
        except Exception as exc:  # noqa: BLE001 — unknown, never idle
            log.debug("embedding pool: GET %s/slots failed: %s", g.url, exc)
            return None
        if occ is None or not occ.slots:
            return None
        return occ.processing > 0

    async def read(self, guarded: Sequence[GuardedInstance], http: Any) -> dict[int, bool | None]:
        if not guarded:
            return {}
        results = await asyncio.gather(*(self._one(g, http) for g in guarded))
        return {g.port: r for g, r in zip(guarded, results)}


class StaticBusySource:
    """Fixed answers, for tests and for callers that already know."""

    name = "static"

    def __init__(self, busy: Iterable[int] = (), unknown: Iterable[int] = ()) -> None:
        self.busy = set(int(p) for p in busy)
        self.unknown = set(int(p) for p in unknown)

    async def read(self, guarded: Sequence[GuardedInstance], http: Any) -> dict[int, bool | None]:
        return {g.port: (None if g.port in self.unknown else g.port in self.busy) for g in guarded}


class CompositeBusySource:
    """Apply sources in order; a port decided busy by an earlier source is not re-probed."""

    name = "composite"

    def __init__(self, sources: Sequence[BusySource]) -> None:
        self.sources = list(sources)

    async def snapshot(
        self,
        guarded: Sequence[GuardedInstance],
        http: Any,
        clock: Callable[[], float] = time.monotonic,
    ) -> BusySnapshot:
        decided: dict[int, bool] = {}
        source_by_port: dict[int, str] = {}
        for source in self.sources:
            todo = [g for g in guarded if decided.get(g.port) is not True]
            if not todo:
                break
            try:
                answers = await source.read(todo, http)
            except Exception as exc:  # noqa: BLE001
                log.debug("embedding pool: busy source %s failed: %s", source.name, exc)
                continue
            for port, verdict in answers.items():
                if verdict is None:
                    continue
                if verdict or port not in decided:
                    decided[port] = bool(verdict)
                    source_by_port[port] = source.name
        busy = frozenset(p for p, v in decided.items() if v)
        unknown = frozenset(g.port for g in guarded if g.port not in decided)
        return BusySnapshot(
            busy=busy, unknown=unknown, source_by_port=source_by_port, taken_at=clock()
        )


def busy_source_from_policy(
    names: Sequence[str], *, slots_timeout_s: float, ledger: LedgerBusySource | None = None
) -> CompositeBusySource:
    sources: list[BusySource] = []
    for name in names:
        if name == "ledger":
            sources.append(ledger or LedgerBusySource())
        elif name == "slots":
            sources.append(SlotsBusySource(timeout_s=slots_timeout_s))
        else:
            raise ValueError(f"unknown busy source {name!r}")
    return CompositeBusySource(sources)


__all__ = [
    "BusySnapshot",
    "BusySource",
    "CompositeBusySource",
    "LedgerBusySource",
    "SlotsBusySource",
    "StaticBusySource",
    "busy_source_from_policy",
]
