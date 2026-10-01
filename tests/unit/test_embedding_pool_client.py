"""UFH-12 REPL-EMB-1.1/1.4: the pooled async client, the sync facade, the cache and the flag.

Offline: every HTTP request goes to FakeEmbeddingServers (an httpx.MockTransport).
"""

from __future__ import annotations

import asyncio
import threading

import numpy as np
import pytest

from src.embedding_pool import (
    FEATURE_FLAG,
    get_pooled_embedder,
    get_sync_embedder,
    pool_enabled,
    reset_pool,
)
from src.embedding_pool.busy import CompositeBusySource, StaticBusySource
from src.embedding_pool.cache import EmbeddingLRUCache, chunk_key
from src.embedding_pool.client import (
    MODE_LEXICAL,
    EmbeddingUnavailable,
    PooledEmbeddingClient,
    SyncPooledEmbedder,
    normalise_vector,
    parse_embedding_response,
)
from src.embedding_pool.fake import FakeEmbeddingServers, FakePooledEmbedder, fake_vector
from src.embedding_pool.policy import EmbeddingPoolPolicy
from src.embedding_pool.scheduler import EmbeddingScheduler
from tests.unit.test_embedding_pool_scheduler import NODES, make_topology

DIM = 32


def policy(**client) -> EmbeddingPoolPolicy:
    base = dict(embedding_dim=DIM, failure_backoff_s=60.0)
    base.update(client)
    # slots only (as test_embedding_pool_cap_enforcement): the ledger source reads the HOST's
    # region locks, so a live orchestrator holding a frontdoor lock made these offline tests
    # see every guarded instance busy and cap every embedder to 1 (4 failures, 2026-10-01).
    return EmbeddingPoolPolicy().with_client(**base).with_cap(busy_sources=("slots",))


def build(busy=(), pol=None, delay_s=0.0, topology=None):
    pol = pol or policy()
    fake = FakeEmbeddingServers(NODES, dim=DIM, delay_s=delay_s)
    fake.processing = {p: 1 for p in busy}
    # the REAL /slots reader against the fake servers, so the cap is exercised end to end
    sched = EmbeddingScheduler(topology or make_topology(), pol)
    client = PooledEmbeddingClient(sched, policy=pol, transport=fake.transport())
    return fake, sched, client


def unit(text):
    v = fake_vector(text, DIM)
    return v / np.linalg.norm(v)


TEXTS = [f"chunk {i} of the spill file about hash maps and collisions" for i in range(40)]


# ── basics ───────────────────────────────────────────────────────────────────────────────


async def test_embeds_in_order_normalised_and_list_batched():
    fake, sched, client = build()
    async with client:
        vecs = await client.embed_many(TEXTS)
    assert vecs.shape == (40, DIM) and vecs.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, rtol=1e-5)
    for text, row in zip(TEXTS, vecs):
        np.testing.assert_allclose(row, unit(text), rtol=1e-5, atol=1e-6)
    # list batches, never more than a port's 4 slots per request, spread over the pool
    assert all(1 <= n <= 4 for _p, n in fake.requests)
    assert len(fake.requests) < len(TEXTS)
    assert {p for p, _n in fake.requests} == set(NODES)
    assert max(fake.peak.values()) <= 4
    assert sched.total_in_flight() == 0


async def test_one_client_one_http_session_across_calls():
    fake, _sched, client = build()
    await client.embed_many(TEXTS[:3])
    first = client._http
    await client.embed_many(TEXTS[3:6])
    assert client._http is first and not first.is_closed
    await client.aclose()
    assert first.is_closed


async def test_cache_is_sha256_lru_and_skips_http():
    fake, sched, client = build()
    await client.embed_many(TEXTS[:5])
    n = len(fake.requests)
    again = await client.embed_many(TEXTS[:5] + TEXTS[:5])
    assert len(fake.requests) == n  # all hits
    assert again.shape == (10, DIM)
    assert sched.stats()["counters"]["cache_hits"] == 5
    assert client.cache.get(chunk_key(client.model_id, TEXTS[0])) is not None
    # bypass (the measurement gate uses this) goes to the pool again
    await client.embed_many(TEXTS[:1], use_cache=False)
    assert len(fake.requests) == n + 1


def test_lru_evicts_oldest_and_keys_bind_the_model():
    cache = EmbeddingLRUCache(max_entries=2)
    cache.put("a", np.ones(2))
    cache.put("b", np.ones(2))
    cache.get("a")
    cache.put("c", np.ones(2))
    assert cache.get("b") is None and cache.get("a") is not None
    assert chunk_key("bge", "x") != chunk_key("granite", "x")
    with pytest.raises(ValueError):
        cache.get("a")[0] = 5.0  # stored read-only


async def test_duplicate_texts_in_one_call_are_embedded_once():
    fake, _s, client = build()
    await client.embed_many(["same text"] * 6)
    assert sum(n for _p, n in fake.requests) == 1


# ── degenerate guard, no hash fallback ───────────────────────────────────────────────────


def test_normalise_vector_guard():
    np.testing.assert_allclose(
        np.linalg.norm(normalise_vector([3.0, 4.0], dim=2, min_norm=1e-6)), 1.0
    )
    for bad, dim in (([0.0, 0.0], 2), ([np.nan, 1.0], 2), ([1.0, 2.0, 3.0], 2), ([], None)):
        with pytest.raises(Exception, match="norm|non-finite|dimension|empty"):
            normalise_vector(bad, dim=dim, min_norm=1e-6)


def test_parse_embedding_response_shapes():
    assert parse_embedding_response(
        [{"index": 1, "embedding": [[2.0]]}, {"index": 0, "embedding": [[1.0]]}], 2
    ) == [[1.0], [2.0]]
    assert parse_embedding_response({"data": [{"embedding": [1.0]}]}, 1) == [[1.0]]
    with pytest.raises(Exception, match="expected 2"):
        parse_embedding_response([{"embedding": [1.0]}], 2)
    with pytest.raises(Exception, match="per-token"):
        parse_embedding_response([{"embedding": [[1.0], [2.0]]}], 1)


async def test_degenerate_vector_is_retried_elsewhere_then_refused_never_faked():
    fake, sched, client = build()
    fake.degenerate_ports = {8090}
    vecs = await client.embed_many(TEXTS[:24])  # some chunks land on 8090 and are retried
    assert np.all(np.isfinite(vecs)) and sched.stats()["counters"]["degenerate_vectors"] >= 1
    fake.degenerate_ports = set(NODES)
    client.cache.clear()
    with pytest.raises(EmbeddingUnavailable) as exc:
        await client.embed_many(["new text never seen"])
    assert exc.value.reason == "degenerate"
    assert client.cache.get(chunk_key(client.model_id, "new text never seen")) is None


async def test_empty_text_is_degenerate_not_a_pseudo_vector():
    _fake, _s, client = build()
    out = await client.try_embed_many([""])
    assert out.mode == MODE_LEXICAL and out.vectors is None and out.reason == "degenerate"


# ── retry and failure ────────────────────────────────────────────────────────────────────


async def test_failed_instance_is_retried_on_the_next_and_backed_off():
    fake, sched, client = build()
    fake.fail_ports = {8090, 8092}
    vecs = await client.embed_many(TEXTS[:24])
    assert vecs.shape == (24, DIM)
    stats = sched.stats()["counters"]
    assert stats["retries"] >= 1 and stats["instance_failures"] >= 1
    assert 8090 not in sched.healthy_ports()


async def test_all_instances_failing_raises_all_failed():
    fake, _s, client = build()
    fake.fail_ports = set(NODES)
    with pytest.raises(EmbeddingUnavailable) as exc:
        await client.embed_many(TEXTS[:2])
    assert exc.value.reason == "all_failed"


async def test_timeout_is_bounded():
    fake, sched, client = build(delay_s=5.0, pol=policy(call_timeout_s=0.2))
    with pytest.raises(EmbeddingUnavailable) as exc:
        await client.embed_many(TEXTS[:2])
    assert exc.value.reason == "timeout"
    assert fake.total_inflight() == 0 and sched.total_in_flight() == 0


# ── REPL-EMB-1.4: the neighbour cap, end to end over /slots ─────────────────────────────


async def test_busy_half_b_caps_its_neighbours_to_one_in_flight():
    fake, sched, client = build(busy={8180}, delay_s=0.02)
    await client.embed_many(TEXTS, admission_wait_s=5.0)
    for port in (8092, 8093, 8094, 8095):
        assert fake.peak.get(port, 0) <= 1, (port, fake.peak)
    assert fake.peak[8090] == 4 and fake.peak[8091] == 4  # idle instances took the bulk
    assert sched.busy_snapshot().busy == {8180}
    assert 8180 in fake.slots_reads


async def test_busy_full_frontdoor_caps_the_whole_pool():
    fake, _s, client = build(busy={8070}, delay_s=0.02)
    await client.embed_many(TEXTS[:20], admission_wait_s=5.0)
    assert max(fake.peak.values()) <= 1


async def test_saturated_pool_answers_lexically_now():
    pol = policy().with_cap(max_in_flight=0)
    fake, _s, client = build(busy={8070}, pol=pol)
    out = await client.try_embed_many(TEXTS[:3])
    assert out.mode == MODE_LEXICAL and out.reason == "saturated"
    assert fake.requests == []


async def test_requester_hardware_is_preferred_when_everything_is_capped():
    fake, sched, client = build(busy={8070}, delay_s=0.01)
    await client.embed_many(TEXTS[:2], requester_port=8080)
    assert {p for p, _n in fake.requests} <= {8090, 8091}
    assert sched.stats()["counters"].get("texts_requester_hardware") == 2


async def test_cap_engages_when_a_decode_starts_mid_batch():
    pol = policy().with_cap(busy_ttl_s=0.0)
    fake, _s, client = build(pol=pol, delay_s=0.03)
    switch = {}

    async def start_decode():
        await asyncio.sleep(0.05)
        fake.processing[8180] = 1
        switch["at"] = len(fake.requests)

    await asyncio.gather(
        client.embed_many(TEXTS * 3 + [f"x{i}" for i in range(60)], admission_wait_s=5.0),
        start_decode(),
    )
    capped = (8092, 8093, 8094, 8095)
    before = [n for p, n in fake.requests[: switch["at"]] if p in capped]
    # one scheduling round may still use the pre-decode snapshot; after that, cap 1 holds
    after = [n for p, n in fake.requests[switch["at"] + len(capped) :] if p in capped]
    assert max(before) == 4
    assert after and max(after) == 1


# ── R2: nothing outlives the call ────────────────────────────────────────────────────────


async def test_task_cancellation_cancels_every_request_and_releases_slots():
    fake, sched, client = build(delay_s=10.0)
    task = asyncio.ensure_future(client.embed_many(TEXTS[:12]))
    await asyncio.sleep(0.05)
    assert fake.total_inflight() == 12
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert fake.total_inflight() == 0 and fake.cancelled >= 1
    assert sched.total_in_flight() == 0


async def test_cancel_event_stops_the_call():
    fake, sched, client = build(delay_s=10.0)
    ev = threading.Event()

    async def trip():
        await asyncio.sleep(0.05)
        ev.set()

    with pytest.raises(EmbeddingUnavailable) as exc:
        await asyncio.gather(client.embed_many(TEXTS[:4], cancel_event=ev), trip())
    assert exc.value.reason == "cancelled"
    assert fake.total_inflight() == 0 and sched.total_in_flight() == 0


async def test_client_is_bound_to_one_loop():
    _f, _s, client = build()
    await client.embed_many(["a b"])

    def other_loop():
        return asyncio.run(client.embed_many(["c d"]))

    with pytest.raises(RuntimeError, match="another event loop"):
        await asyncio.to_thread(other_loop)


# ── sync facade ──────────────────────────────────────────────────────────────────────────


def test_sync_facade_reuses_its_client_and_shares_the_cap():
    fake = FakeEmbeddingServers(NODES, dim=DIM)
    pol = policy()
    sched = EmbeddingScheduler(
        make_topology(), pol, busy_source=CompositeBusySource([StaticBusySource()])
    )
    with SyncPooledEmbedder(sched, policy=pol, transport=fake.transport()) as sync:
        a = sync.embed_many(TEXTS[:3])
        http = sync.client._http
        b = sync.embed("another text")
        assert sync.client._http is http and not http.is_closed
        out = sync.try_embed_many(TEXTS[3:5])
        assert out.is_dense and a.shape == (3, DIM) and b.shape == (DIM,)
    assert http.is_closed
    with pytest.raises(RuntimeError, match="closed"):
        sync.embed("x")


async def test_sync_facade_works_from_inside_a_running_loop():
    fake = FakeEmbeddingServers(NODES, dim=DIM)
    pol = policy()
    sched = EmbeddingScheduler(
        make_topology(), pol, busy_source=CompositeBusySource([StaticBusySource()])
    )
    sync = SyncPooledEmbedder(sched, policy=pol, transport=fake.transport())
    try:
        vec = await asyncio.to_thread(sync.embed, "from a worker thread")
        assert vec.shape == (DIM,)
    finally:
        sync.close()


# ── fake embedder for consumer tests ─────────────────────────────────────────────────────


async def test_fake_pooled_embedder_has_the_client_surface():
    fake = FakePooledEmbedder(dim=16)
    vecs = await fake.embed_many(["alpha beta", "alpha gamma"])
    assert vecs.shape == (2, 16)
    fake.available = False
    out = await fake.try_embed_many(["x"])
    assert out.mode == MODE_LEXICAL and out.reason == "saturated"


# ── feature flag ─────────────────────────────────────────────────────────────────────────


async def test_flag_is_off_by_default_and_factories_return_none():
    from src.features import Features

    assert FEATURE_FLAG == "repl_embedding_pool"
    assert Features().repl_embedding_pool is False
    assert pool_enabled() is False
    assert get_pooled_embedder() is None
    assert get_sync_embedder() is None


async def test_flag_on_builds_one_client_per_loop_over_the_live_topology():
    from src.features import Features, features, reset_features, set_features

    set_features(Features(**{**features().summary(), FEATURE_FLAG: True}))
    try:
        reset_pool()
        a = get_pooled_embedder()
        b = get_pooled_embedder()
        assert a is not None and a is b
        assert a.scheduler.policy.neighbour_cap.max_in_flight == 1
        assert a.scheduler.topology.guarded  # frontdoor instances, derived
    finally:
        reset_pool()
        reset_features()
