"""UFH-12 REPL-EMB-1.4: does the neighbour cap enforce once a frontdoor turns busy? (offline)

The post-cap live record (data/embedder_placement/post-cap-20260927.json) showed
``effective_caps_at_end`` = 1 on every neighbour of the busy frontdoor but ``peak_in_flight``
= 4 on every port in every sample. These tests reproduce that saturated shape offline — the
load starts while the frontdoor is idle (the gate's warm-up), then a fake frontdoor turns busy —
and pin the verdict:

* the cap IS enforced: no grant made while capped ever lifts an instance above the cap, and
  once the pre-flip texts drain, in-flight on every capped port stays <= the cap;
* the 4 was the lifetime high-water mark: ``peak_in_flight`` includes the uncapped warm-up and
  the texts granted before busy detection flipped (admission never preempts), so it is not
  evidence about the cap. ``peak_in_flight_capped`` is.
"""

from __future__ import annotations

import time

from scripts.server import embedder_placement_gate as gate
from src.embedding_pool.busy import BusySnapshot, CompositeBusySource, StaticBusySource
from src.embedding_pool.client import PooledEmbeddingClient
from src.embedding_pool.fake import FakeEmbeddingServers
from src.embedding_pool.policy import EmbeddingPoolPolicy
from src.embedding_pool.scheduler import EmbeddingScheduler
from tests.unit.test_embedding_pool_scheduler import NODES, make_topology

CAP = 1
FULL_NEIGHBOURS = (8090, 8091, 8092, 8093, 8094, 8095)  # :8070 spans nodes 0-3
HALF_B_NEIGHBOURS = (8092, 8093, 8094, 8095)  # :8180 spans nodes 2-3


def _policy() -> EmbeddingPoolPolicy:
    # slots only: the ledger source reads the HOST's /proc/locks, and a live orchestrator
    # holding a frontdoor region lock would make an offline test depend on live traffic.
    return (
        EmbeddingPoolPolicy()
        .with_client(embedding_dim=None)
        .with_cap(busy_sources=("slots",), max_in_flight=CAP)
    )


def _set_busy(sched: EmbeddingScheduler, busy: set[int]) -> None:
    sched.set_busy_snapshot(BusySnapshot(busy=frozenset(busy), taken_at=1e18))


# ── deterministic: the ledger across the idle -> busy flip ────────────────────────────────


def test_flip_to_busy_never_grants_above_the_cap_and_lifetime_peak_keeps_the_warmup():
    sched = EmbeddingScheduler(
        make_topology(),
        _policy(),
        busy_source=CompositeBusySource([StaticBusySource()]),
    )
    _set_busy(sched, set())
    warm = sched.try_reserve(4 * len(NODES))  # warm-up: frontdoor idle, every port full
    assert sched.in_flight() == {p: 4 for p in NODES}

    _set_busy(sched, {8070})  # busy detection flips; pre-flip texts are still in flight
    assert all(sched.effective_caps()[p] == CAP for p in FULL_NEIGHBOURS)
    assert sched.try_reserve(24) == []  # over the cap: nothing more is admitted

    # drain one text at a time while the offered load keeps knocking
    for g in warm:
        sched.release(g)
        for got in sched.try_reserve(24):
            assert got.capped
            assert sched.in_flight()[got.port] <= CAP
    # steady saturated state under the cap: exactly CAP per port, never more
    for _ in range(50):
        grants = sched.try_reserve(24)
        assert all(sched.in_flight()[p] <= CAP for p in FULL_NEIGHBOURS)
        for g in grants:
            sched.release(g)

    stats = sched.stats()
    # the record's shape: cap 1 at the end, lifetime peak 4 everywhere
    assert stats["effective_caps"] == {p: CAP for p in FULL_NEIGHBOURS}
    assert stats["peak_in_flight"] == {p: 4 for p in FULL_NEIGHBOURS}
    # ...while every grant made under the cap respected it
    assert all(0 < v <= CAP for v in stats["peak_in_flight_capped"].values())

    # a new window restarts the raw peak from what is in flight now, the capped peak from 0
    sched.reset_peaks()
    assert sched.peak_in_flight() == sched.in_flight()
    assert sched.peak_in_flight_capped() == {p: 0 for p in NODES}


# ── the gate's saturated shape, through the pooled client and /slots ─────────────────────


def _factory(fake: FakeEmbeddingServers):
    def build(ports):
        pol = _policy()
        sched = EmbeddingScheduler(make_topology().restricted_to(ports), pol)
        return PooledEmbeddingClient(sched, policy=pol, transport=fake.transport())

    return build


def _run_flip(busy_port: int, capped_ports: tuple[int, ...], delay_s: float = 0.02):
    fake = FakeEmbeddingServers(NODES, dim=16, delay_s=delay_s)
    load = gate._SchedulerLoad(
        list(NODES), 4, client_factory=_factory(fake), warmup_s=0.2, admission_wait_s=5.0
    )
    violations: list[tuple[int, int]] = []
    with load:
        sched = load._client.scheduler
        # warm-up ran with the frontdoor idle: every port was saturated at its full 4 slots
        assert fake.peak and all(fake.peak[p] == 4 for p in NODES)
        fake.processing = {busy_port: 1}  # the frontdoor starts decoding
        deadline = time.monotonic() + 3.0
        while busy_port not in sched.busy_snapshot().busy:
            assert time.monotonic() < deadline, "busy detection never flipped"
            time.sleep(0.005)
        time.sleep(4 * delay_s)  # texts granted before the flip drain (admission never preempts)
        end = time.monotonic() + 0.4
        while time.monotonic() < end:
            inflight = sched.in_flight()
            caps = sched.effective_caps()
            for p in capped_ports:
                assert caps[p] == CAP
                if inflight[p] > CAP or fake.inflight.get(p, 0) > CAP:
                    violations.append((p, max(inflight[p], fake.inflight.get(p, 0))))
            time.sleep(0.003)
    return load.summary(), violations


def test_full_frontdoor_busy_caps_all_six_neighbours_while_busy():
    summary, violations = _run_flip(8070, FULL_NEIGHBOURS)
    assert violations == []
    assert all(summary["peak_in_flight_capped"][p] <= CAP for p in FULL_NEIGHBOURS)
    assert summary["counters_delta"].get("grants_capped", 0) > 0


def test_half_busy_caps_only_its_neighbours():
    summary, violations = _run_flip(8180, HALF_B_NEIGHBOURS)
    assert violations == []
    capped_peak = summary["peak_in_flight_capped"]
    assert all(capped_peak[p] <= CAP for p in HALF_B_NEIGHBOURS)
    assert capped_peak[8090] == capped_peak[8091] == 0  # never capped: node 1 is half A
    assert summary["counters_delta"].get("grants_idle", 0) > 0
