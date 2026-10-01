"""UFH-12 REPL-EMB-1.4: the placement gate's scheduler load mode (offline).

The raw mode is the Phase-0 method and stays the default; the scheduler mode offers the same
load through the pooled client so the neighbour cap is exercised. Nothing here touches a
live server: the client factory points the pool at FakeEmbeddingServers.
"""

from __future__ import annotations

import time

import pytest

from scripts.server import embedder_placement_gate as gate
from src.embedding_pool.client import PooledEmbeddingClient
from src.embedding_pool.fake import FakeEmbeddingServers
from src.embedding_pool.policy import EmbeddingPoolPolicy
from src.embedding_pool.scheduler import EmbeddingScheduler
from tests.unit.test_embedding_pool_scheduler import NODES, make_topology


def _factory(fake: FakeEmbeddingServers):
    def build(ports):
        # slots only (as test_embedding_pool_client): the default ledger source reads the HOST's
        # region locks, so a live frontdoor lock capped every fake embedder to 1 (2026-10-01).
        pol = EmbeddingPoolPolicy().with_client(embedding_dim=None).with_cap(busy_sources=("slots",))
        sched = EmbeddingScheduler(make_topology().restricted_to(ports), pol)
        return PooledEmbeddingClient(sched, policy=pol, transport=fake.transport())

    return build


def test_raw_mode_is_the_unchanged_default():
    assert gate._load_for("raw") is gate._Load
    import inspect

    assert inspect.signature(gate.g1).parameters["load_factory"].default is gate._Load
    assert inspect.signature(gate.g2).parameters["load_factory"].default is gate._Load
    with pytest.raises(ValueError):
        gate._load_for("bogus")


def test_scheduler_load_saturates_through_the_cap():
    fake = FakeEmbeddingServers(NODES, dim=16, delay_s=0.01)
    fake.processing = {8180: 1}  # half B decoding
    load = gate._SchedulerLoad(
        list(NODES), 4, client_factory=_factory(fake), warmup_s=0.1, admission_wait_s=5.0
    )
    with load:
        time.sleep(0.4)
    summary = load.summary()
    assert load.errors == 0 and load.done > 0
    assert summary["texts_per_s"] > 0
    assert summary["effective_caps_at_end"] == {
        8090: 4,
        8091: 4,
        8092: 1,
        8093: 1,
        8094: 1,
        8095: 1,
    }
    for port in (8092, 8093, 8094, 8095):
        assert fake.peak.get(port, 0) <= 1
    assert fake.peak[8090] == 4
    assert summary["counters_delta"].get("grants_capped", 0) > 0
    assert fake.total_inflight() == 0  # every call ended with the sample


def test_g1_and_g2_record_the_scheduler_load(monkeypatch):
    fake = FakeEmbeddingServers(NODES, dim=16, delay_s=0.005)
    monkeypatch.setattr(gate, "_decode_tps", lambda port, n: (time.sleep(0.05), 50.0)[1])
    factory = gate._load_for(
        "scheduler", client_factory=_factory(fake), warmup_s=0.05, admission_wait_s=5.0
    )
    out = gate.g1([8180], pairs=2, n_predict=8, load_factory=factory)
    row = out["8180"]
    assert row["s_over_q_median"] == pytest.approx(1.0)
    assert len(row["s_embedding_load"]) == 2 and row["s_embed_texts_per_s_median"] > 0
    g2 = gate.g2(0.1, factory)
    assert g2["scaling_ratio"] > 0 and "load" in g2["whole_pool"]
