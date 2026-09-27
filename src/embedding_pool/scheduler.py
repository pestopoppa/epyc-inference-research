"""Placement-aware slot scheduler for the embedding pool (REPL-EMB-1.1 + REPL-EMB-1.4).

The unit of admission is a SLOT: one text being embedded. An instance declared ``-np 4``
holds at most 4 texts in flight, so a list batch sent to it carries at most its headroom.

Effective cap of an embedder instance:

* ``slots`` normally;
* ``min(slots, neighbour_cap.max_in_flight)`` while any guarded serving instance on its NUMA
  node is busy (REPL-EMB-1.4, operator decision D3). An instance whose guarded neighbour's
  state is unknown counts as capped when ``unknown_is_busy``.

Selection (operator decision D1): texts go, one slot at a time, to the best candidate by
``(tier, load)``:

* tier 0 — "an idle instance anywhere": not capped (no busy neighbour); least-loaded first,
  so a fully idle instance always beats a partly used one;
* tier 1 — "the instance sharing the requesting model's hardware": capped instances on a
  node of the requester's serving instance, up to the cap;
* tier 2 — other capped instances up to the cap (``use_capped_foreign_instances``);
* nothing left — the caller waits up to its admission budget, then answers lexically now and
  indexes later (the client raises ``EmbeddingUnavailable("saturated")``).

The scheduler is thread-safe and event-loop agnostic, so one instance can be shared by every
client in the process (async clients on the API loop and the sync facade's own loop); the
in-flight accounting — and therefore the cap — is process-wide.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from src.embedding_pool.busy import BusySnapshot, CompositeBusySource, busy_source_from_policy
from src.embedding_pool.policy import EmbeddingPoolPolicy
from src.embedding_pool.topology import PoolTopology

TIER_IDLE = "idle"
TIER_REQUESTER = "requester_hardware"
TIER_CAPPED_FOREIGN = "capped_foreign"
_TIER_ORDER = {TIER_IDLE: 0, TIER_REQUESTER: 1, TIER_CAPPED_FOREIGN: 2}


@dataclass(frozen=True)
class Grant:
    port: int
    url: str
    n: int
    tier: str
    capped: bool
    grant_id: int


class EmbeddingScheduler:
    def __init__(
        self,
        topology: PoolTopology,
        policy: EmbeddingPoolPolicy,
        *,
        busy_source: CompositeBusySource | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.topology = topology
        self.policy = policy
        cap = policy.neighbour_cap
        self.busy_source = busy_source or busy_source_from_policy(
            cap.busy_sources, slots_timeout_s=cap.slots_timeout_s
        )
        self._clock = clock
        self._lock = threading.Lock()
        self._inflight: dict[int, int] = {p: 0 for p in topology.ports}
        # Two peaks, both since construction or the last ``reset_peaks()``:
        # ``_peak`` is the raw high-water mark of the in-flight ledger. It includes phases in
        # which the instance was NOT capped (idle neighbours, a warm-up), and texts granted
        # before busy detection flipped, which drain at their pre-flip width (admission never
        # preempts). It is therefore not evidence about the cap.
        # ``_peak_capped`` is the high-water mark reached by grants made WHILE the instance was
        # capped: the enforcement claim is ``_peak_capped[p] <= neighbour_cap.max_in_flight``.
        self._peak: dict[int, int] = {p: 0 for p in topology.ports}
        self._peak_capped: dict[int, int] = {p: 0 for p in topology.ports}
        self._backoff_until: dict[int, float] = {}
        self._busy = BusySnapshot(taken_at=float("-inf"))
        self._refreshing = False
        self._rr = 0
        self._grant_seq = 0
        self._stats: Counter[str] = Counter()

    # -- busy state -----------------------------------------------------------------------
    @property
    def cap_enabled(self) -> bool:
        cap = self.policy.neighbour_cap
        return cap.enabled and bool(self.topology.guarded)

    def busy_snapshot(self) -> BusySnapshot:
        with self._lock:
            return self._busy

    def set_busy_snapshot(self, snapshot: BusySnapshot) -> None:
        with self._lock:
            self._busy = snapshot

    def busy_is_stale(self) -> bool:
        with self._lock:
            return (self._clock() - self._busy.taken_at) >= self.policy.neighbour_cap.busy_ttl_s

    async def refresh_busy(self, http: Any, *, force: bool = False) -> BusySnapshot:
        """Re-read busy state if stale. Concurrent callers share one refresh: a caller that
        finds a refresh already running WAITS for it rather than admitting on stale state
        (at startup the stale state is "nothing known", which would admit uncapped)."""
        if not self.cap_enabled:
            return self.busy_snapshot()
        while True:
            with self._lock:
                stale = (
                    self._clock() - self._busy.taken_at
                ) >= self.policy.neighbour_cap.busy_ttl_s
                if not stale and not force:
                    return self._busy
                if not self._refreshing:
                    self._refreshing = True
                    break
            await asyncio.sleep(0.002)
            force = False  # someone else's refresh satisfies a forced one too
        try:
            snap = await self.busy_source.snapshot(self.topology.guarded, http, clock=self._clock)
            with self._lock:
                self._busy = snap
                self._stats["busy_refreshes"] += 1
            return snap
        finally:
            with self._lock:
                self._refreshing = False

    # -- capacity -------------------------------------------------------------------------
    def _capped_locked(self, port: int) -> bool:
        if not self.cap_enabled:
            return False
        unknown_is_busy = self.policy.neighbour_cap.unknown_is_busy
        return any(
            self._busy.is_busy(g.port, unknown_is_busy=unknown_is_busy)
            for g in self.topology.neighbours(port)
        )

    def _effective_cap_locked(self, port: int) -> int:
        slots = self.topology.embedder(port).slots
        if self._capped_locked(port):
            return max(0, min(slots, int(self.policy.neighbour_cap.max_in_flight)))
        return slots

    def effective_caps(self) -> dict[int, int]:
        with self._lock:
            return {p: self._effective_cap_locked(p) for p in self.topology.ports}

    def in_flight(self) -> dict[int, int]:
        with self._lock:
            return dict(self._inflight)

    def total_in_flight(self) -> int:
        with self._lock:
            return sum(self._inflight.values())

    def peak_in_flight(self) -> dict[int, int]:
        with self._lock:
            return dict(self._peak)

    def peak_in_flight_capped(self) -> dict[int, int]:
        with self._lock:
            return dict(self._peak_capped)

    def reset_peaks(self) -> None:
        """Start a new measurement window: the raw peak restarts from the current in-flight
        (texts already admitted are still in flight), the capped peak from zero."""
        with self._lock:
            self._peak = dict(self._inflight)
            self._peak_capped = {p: 0 for p in self.topology.ports}

    def healthy_ports(self) -> list[int]:
        now = self._clock()
        with self._lock:
            return [p for p in self.topology.ports if self._backoff_until.get(p, 0.0) <= now]

    # -- admission ------------------------------------------------------------------------
    def try_reserve(
        self, n: int, *, requester_port: int | None = None, exclude: Iterable[int] = ()
    ) -> list[Grant]:
        """Reserve up to ``n`` slots now. Returns zero or more grants (one per instance)."""
        if n <= 0:
            return []
        excluded = set(exclude)
        requester_nodes = self.topology.nodes_of_port(requester_port)
        use_foreign = self.policy.neighbour_cap.use_capped_foreign_instances
        with self._lock:
            now = self._clock()
            ports = self.topology.ports
            order = {p: (i - self._rr) % len(ports) for i, p in enumerate(ports)}
            candidates: list[
                tuple[int, str, bool, int, int]
            ] = []  # (tier_rank, tier, capped, port, headroom)
            for port in ports:
                if port in excluded or self._backoff_until.get(port, 0.0) > now:
                    continue
                capped = self._capped_locked(port)
                headroom = self._effective_cap_locked(port) - self._inflight[port]
                if headroom <= 0:
                    continue
                if not capped:
                    tier = TIER_IDLE
                elif self.topology.embedder(port).numa_node in requester_nodes:
                    tier = TIER_REQUESTER
                elif use_foreign:
                    tier = TIER_CAPPED_FOREIGN
                else:
                    continue
                candidates.append((_TIER_ORDER[tier], tier, capped, port, headroom))
            alloc: dict[int, int] = {}
            for _ in range(n):
                best = None
                for rank, tier, capped, port, headroom in candidates:
                    if alloc.get(port, 0) >= headroom:
                        continue
                    key = (rank, self._inflight[port] + alloc.get(port, 0), order[port])
                    if best is None or key < best[0]:
                        best = (key, port)
                if best is None:
                    break
                alloc[best[1]] = alloc.get(best[1], 0) + 1
            grants: list[Grant] = []
            meta = {port: (tier, capped) for _r, tier, capped, port, _h in candidates}
            for port, k in alloc.items():
                self._inflight[port] += k
                self._peak[port] = max(self._peak[port], self._inflight[port])
                self._grant_seq += 1
                tier, capped = meta[port]
                if capped:
                    self._peak_capped[port] = max(self._peak_capped[port], self._inflight[port])
                grants.append(
                    Grant(
                        port=port,
                        url=self.topology.embedder(port).url,
                        n=k,
                        tier=tier,
                        capped=capped,
                        grant_id=self._grant_seq,
                    )
                )
                self._stats[f"grants_{tier}"] += 1
                self._stats[f"texts_{tier}"] += k
                if capped:
                    self._stats["grants_capped"] += 1
            if grants:
                self._rr = (self._rr + 1) % len(ports)
            else:
                self._stats["reserve_denied"] += 1
            return sorted(grants, key=lambda g: order[g.port])

    def release(self, grant: Grant) -> None:
        with self._lock:
            self._inflight[grant.port] = max(0, self._inflight[grant.port] - grant.n)

    def mark_failure(self, port: int) -> None:
        with self._lock:
            self._backoff_until[port] = self._clock() + self.policy.client.failure_backoff_s
            self._stats["instance_failures"] += 1

    def mark_success(self, port: int) -> None:
        with self._lock:
            self._backoff_until.pop(port, None)

    def note(self, key: str, k: int = 1) -> None:
        with self._lock:
            self._stats[key] += k

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "counters": dict(self._stats),
                "in_flight": dict(self._inflight),
                "peak_in_flight": dict(self._peak),
                "peak_in_flight_capped": dict(self._peak_capped),
                "effective_caps": {p: self._effective_cap_locked(p) for p in self.topology.ports},
                "busy": self._busy.as_dict(),
            }


__all__ = ["EmbeddingScheduler", "Grant", "TIER_CAPPED_FOREIGN", "TIER_IDLE", "TIER_REQUESTER"]
