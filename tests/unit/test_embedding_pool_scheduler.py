"""UFH-12 REPL-EMB-1.1/1.4: pool topology, policy, busy sources and the slot scheduler.

Offline: synthetic topologies mirror the 2026-09-26 Phase-0 layout (8090/8091 on node 1,
8092/8093 on node 2, 8094/8095 on node 3; frontdoor full :8070 on nodes 0-3, half A :8080 on
0-1, half B :8180 on 2-3), and one test derives the same facts from the repo's own manifests.
"""

from __future__ import annotations

import asyncio

import pytest

from src.embedding_pool.busy import (
    BusySnapshot,
    CompositeBusySource,
    LedgerBusySource,
    SlotsBusySource,
    StaticBusySource,
    busy_source_from_policy,
)
from src.embedding_pool.fake import FakeEmbeddingServers
from src.embedding_pool.policy import EmbeddingPoolPolicy, load_policy, parse_policy
from src.embedding_pool.scheduler import (
    TIER_CAPPED_FOREIGN,
    TIER_IDLE,
    TIER_REQUESTER,
    EmbeddingScheduler,
)
from src.embedding_pool.topology import (
    EmbedderInstance,
    GuardedInstance,
    PoolTopology,
    live_topology,
    topology_from_declarations,
)

NODES = {8090: 1, 8091: 1, 8092: 2, 8093: 2, 8094: 3, 8095: 3}
GUARDED = {8070: {0, 1, 2, 3}, 8080: {0, 1}, 8180: {2, 3}}


def make_topology(ports=tuple(NODES), slots=4) -> PoolTopology:
    emb = [
        EmbedderInstance(
            port=p, url=f"http://127.0.0.1:{p}", numa_node=NODES[p], slots=slots, model_id="bge"
        )
        for p in ports
    ]
    guarded = [
        GuardedInstance(
            role="frontdoor",
            instance_idx=i,
            port=p,
            url=f"http://127.0.0.1:{p}",
            numa_nodes=frozenset(n),
        )
        for i, (p, n) in enumerate(GUARDED.items())
    ]
    return PoolTopology.build(emb, guarded)


def make_scheduler(busy=(), unknown=(), policy=None, topology=None) -> EmbeddingScheduler:
    sched = EmbeddingScheduler(
        topology or make_topology(),
        policy or EmbeddingPoolPolicy(),
        busy_source=CompositeBusySource([StaticBusySource(busy, unknown)]),
    )
    sched.set_busy_snapshot(
        BusySnapshot(busy=frozenset(busy), unknown=frozenset(unknown), taken_at=1e18)
    )
    return sched


def by_port(grants):
    return {g.port: g.n for g in grants}


# ── topology ─────────────────────────────────────────────────────────────────────────────


def test_neighbours_follow_numa_nodes():
    topo = make_topology()
    assert [g.port for g in topo.neighbours(8090)] == [8070, 8080]
    assert [g.port for g in topo.neighbours(8093)] == [8070, 8180]
    assert [g.port for g in topo.neighbours(8095)] == [8070, 8180]
    assert topo.nodes_of_port(8180) == frozenset({2, 3})
    assert topo.nodes_of_port(None) == frozenset()


def test_pool_refuses_two_models():
    emb = [EmbedderInstance(8090, "u", 1, 4, "bge"), EmbedderInstance(8091, "u", 1, 4, "granite")]
    with pytest.raises(ValueError, match="ONE embedding model"):
        PoolTopology.build(emb)


def test_topology_from_declarations_derives_nodes_and_slots():
    topo = topology_from_declarations(
        placement={8090: ("136-139", 1), 8092: ("144-147", 2)},
        recipes={
            8090: {"model_path": "/m/bge.gguf", "slots": 4},
            8092: {"model_path": "/m/bge.gguf", "slots": 4},
        },
        pool_ports=[8090, 8092],
        numa_config={
            "frontdoor": {
                "instances": [
                    ("0-95", 8070, 96),
                    ("0-47,96-143", 8080, 48),
                    ("48-95,144-191", 8180, 48),
                ]
            }
        },
        nodes_of_cpuset=lambda spec: {
            "0-95": [0, 1, 2, 3],
            "0-47,96-143": [0, 1],
            "48-95,144-191": [2, 3],
        }[spec],
        guarded_roles=["frontdoor"],
    )
    assert topo.model_id == "bge.gguf"
    assert [g.port for g in topo.neighbours(8092)] == [8070, 8180]
    with pytest.raises(ValueError, match="no declared placement"):
        topology_from_declarations(
            placement={},
            recipes={},
            pool_ports=[8090],
            numa_config={},
            nodes_of_cpuset=lambda s: [],
            guarded_roles=[],
        )


def test_live_topology_is_derived_from_the_repo_declarations():
    """Placement is data: the live pool comes from launch_manifest + stack_topology."""
    from scripts.server.stack_manifest import EMBEDDER_PORTS, EMBEDDING_PLACEMENT

    topo = live_topology()
    assert list(topo.ports) == sorted(EMBEDDER_PORTS)
    for e in topo.embedders:
        assert e.numa_node == EMBEDDING_PLACEMENT[e.port].numa_node
        assert e.slots > 0
        # every embedder shares a node with at least the full-machine frontdoor
        assert topo.neighbours(e.port), e.port
    assert {g.role for g in topo.guarded} == {"frontdoor"}


# ── policy ───────────────────────────────────────────────────────────────────────────────


def test_repo_policy_loads_with_the_operator_cap():
    policy = load_policy()
    cap = policy.neighbour_cap
    assert cap.enabled is True and cap.max_in_flight == 1
    assert cap.guarded_roles == ("frontdoor",)
    assert cap.busy_sources == ("ledger", "slots")
    assert policy.client.admission_wait_s == 0.0


@pytest.mark.parametrize(
    "doc, match",
    [
        ({"version": 2}, "version"),
        ({"version": 1, "extra": 1}, "unknown top-level"),
        ({"version": 1, "client": {"nope": 1}}, "unknown key"),
        ({"version": 1, "neighbour_cap": {"busy_sources": ["metrics"]}}, "unknown busy source"),
        ({"version": 1, "neighbour_cap": {"max_in_flight": -1}}, "non-negative int"),
        ({"version": 1, "neighbour_cap": {"enabled": "yes"}}, "boolean"),
        ({"version": 1, "client": {"request_timeout_s": 0}}, "timeouts"),
    ],
)
def test_policy_rejects_bad_documents(doc, match):
    with pytest.raises(ValueError, match=match):
        parse_policy(doc)


# ── scheduler: slots, balance, cap, tiers ────────────────────────────────────────────────


def test_idle_pool_balances_and_respects_slots():
    sched = make_scheduler()
    grants = sched.try_reserve(30)
    assert by_port(grants) == {p: 4 for p in NODES}  # 24 slots, never more than -np per port
    assert all(g.tier == TIER_IDLE and not g.capped for g in grants)
    assert sched.try_reserve(1) == []
    for g in grants:
        sched.release(g)
    assert sched.total_in_flight() == 0


def test_small_batches_spread_least_loaded_first():
    sched = make_scheduler()
    first = sched.try_reserve(6)
    assert by_port(first) == {p: 1 for p in NODES}


def test_busy_half_b_caps_its_node_neighbours_and_idle_instances_win():
    sched = make_scheduler(busy={8180})
    caps = sched.effective_caps()
    assert caps == {8090: 4, 8091: 4, 8092: 1, 8093: 1, 8094: 1, 8095: 1}
    grants = sched.try_reserve(8)
    assert by_port(grants) == {8090: 4, 8091: 4}  # idle anywhere first
    more = sched.try_reserve(8)
    assert by_port(more) == {8092: 1, 8093: 1, 8094: 1, 8095: 1}  # then capped, <= 1 each
    assert all(g.capped for g in more)
    assert sched.try_reserve(1) == []


def test_busy_full_frontdoor_caps_every_instance():
    sched = make_scheduler(busy={8070})
    assert set(sched.effective_caps().values()) == {1}


def test_requester_hardware_before_foreign_capped_instances():
    sched = make_scheduler(busy={8070})  # everything capped
    grants = sched.try_reserve(2, requester_port=8080)
    assert by_port(grants) == {8090: 1, 8091: 1}
    assert {g.tier for g in grants} == {TIER_REQUESTER}
    rest = sched.try_reserve(10, requester_port=8080)
    assert by_port(rest) == {8092: 1, 8093: 1, 8094: 1, 8095: 1}
    assert {g.tier for g in rest} == {TIER_CAPPED_FOREIGN}


def test_foreign_capped_instances_can_be_disabled():
    policy = EmbeddingPoolPolicy().with_cap(use_capped_foreign_instances=False)
    sched = make_scheduler(busy={8070}, policy=policy)
    assert by_port(sched.try_reserve(10, requester_port=8180)) == {
        8092: 1,
        8093: 1,
        8094: 1,
        8095: 1,
    }
    assert sched.try_reserve(1, requester_port=None) == []  # no requester -> lexical


def test_cap_zero_means_lexical_when_every_neighbour_is_busy():
    sched = make_scheduler(busy={8070}, policy=EmbeddingPoolPolicy().with_cap(max_in_flight=0))
    assert sched.try_reserve(4, requester_port=8080) == []
    assert sched.stats()["counters"]["reserve_denied"] == 1


def test_unknown_state_counts_as_busy_by_default():
    assert make_scheduler(unknown={8080}).effective_caps()[8090] == 1
    lax = EmbeddingPoolPolicy().with_cap(unknown_is_busy=False)
    assert make_scheduler(unknown={8080}, policy=lax).effective_caps()[8090] == 4


def test_cap_disabled_restores_full_slots():
    sched = make_scheduler(busy={8070}, policy=EmbeddingPoolPolicy().with_cap(enabled=False))
    assert set(sched.effective_caps().values()) == {4}


def test_backoff_and_exclude_skip_an_instance():
    now = [100.0]
    sched = EmbeddingScheduler(
        make_topology(),
        EmbeddingPoolPolicy(),
        clock=lambda: now[0],
        busy_source=CompositeBusySource([StaticBusySource()]),
    )
    sched.set_busy_snapshot(BusySnapshot(taken_at=1e18))
    sched.mark_failure(8090)
    assert 8090 not in by_port(sched.try_reserve(24))
    now[0] += 5.0
    assert 8090 in sched.healthy_ports()
    assert 8091 not in by_port(sched.try_reserve(4, exclude={8091}))


def test_concurrent_reservations_never_exceed_the_cap():
    """The in-flight ledger is thread-safe: 8 threads hammering reserve/release."""
    import threading

    sched = make_scheduler(busy={8180})
    violations = []

    def worker():
        for _ in range(300):
            for g in sched.try_reserve(3):
                cap = sched.effective_caps()[g.port]
                if sched.in_flight()[g.port] > cap:
                    violations.append(g.port)
                sched.release(g)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    peak = sched.peak_in_flight()
    assert not violations
    assert all(peak[p] <= 1 for p in (8092, 8093, 8094, 8095))
    assert sched.total_in_flight() == 0


# ── busy sources ─────────────────────────────────────────────────────────────────────────


def _guarded():
    return make_topology().guarded


async def test_slots_source_reads_is_processing_and_unknown_on_error():
    import httpx

    fake = FakeEmbeddingServers([])
    fake.processing = {8180: 1}
    fake.slots_fail_ports = {8080}
    async with httpx.AsyncClient(transport=fake.transport()) as http:
        got = await SlotsBusySource().read(_guarded(), http)
    assert got == {8070: False, 8080: None, 8180: True}


async def test_ledger_source_is_positive_only():
    ledger = LedgerBusySource(holders_fn=lambda: {"frontdoor": [2], "coder": [0]})
    got = await ledger.read(_guarded(), None)
    assert got == {8070: None, 8080: None, 8180: True}
    broken = LedgerBusySource(holders_fn=lambda: (_ for _ in ()).throw(OSError("no /proc")))
    assert await broken.read(_guarded(), None) == {8070: None, 8080: None, 8180: None}


async def test_composite_prefers_ledger_and_skips_probing_ledger_busy_ports():
    import httpx

    fake = FakeEmbeddingServers([])
    fake.processing = {8080: 2}
    ledger = LedgerBusySource(holders_fn=lambda: {"frontdoor": [0]})
    source = busy_source_from_policy(["ledger", "slots"], slots_timeout_s=0.5, ledger=ledger)
    async with httpx.AsyncClient(transport=fake.transport()) as http:
        snap = await source.snapshot(_guarded(), http)
    assert snap.busy == {8070, 8080}
    assert snap.unknown == frozenset()
    assert snap.source_by_port[8070] == "ledger" and snap.source_by_port[8080] == "slots"
    assert 8070 not in fake.slots_reads  # already busy by the ledger: not probed


async def test_scheduler_refresh_is_ttl_bound():
    now = [0.0]
    calls = []

    class Counting(StaticBusySource):
        async def read(self, guarded, http):
            calls.append(1)
            return await super().read(guarded, http)

    sched = EmbeddingScheduler(
        make_topology(),
        EmbeddingPoolPolicy(),
        clock=lambda: now[0],
        busy_source=CompositeBusySource([Counting(busy={8180})]),
    )
    await sched.refresh_busy(None)
    await sched.refresh_busy(None)
    assert len(calls) == 1
    now[0] += 0.3  # > busy_ttl_s 0.25
    await asyncio.gather(sched.refresh_busy(None), sched.refresh_busy(None))
    assert len(calls) == 2
    assert sched.busy_snapshot().busy == {8180}


async def test_concurrent_first_refresh_waits_instead_of_admitting_on_nothing_known():
    """Regression: at startup, callers that found a refresh in progress used the empty
    initial snapshot and admitted UNCAPPED (peak 4 next to a busy half in the gate test)."""
    calls = []

    class Slow(StaticBusySource):
        async def read(self, guarded, http):
            calls.append(1)
            await asyncio.sleep(0.05)
            return await super().read(guarded, http)

    sched = EmbeddingScheduler(
        make_topology(), EmbeddingPoolPolicy(), busy_source=CompositeBusySource([Slow(busy={8180})])
    )
    snaps = await asyncio.gather(*(sched.refresh_busy(None) for _ in range(5)))
    assert all(s.busy == {8180} for s in snaps)
    assert len(calls) == 1
