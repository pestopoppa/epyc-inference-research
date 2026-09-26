"""Embedding pool: placement-aware pooled embedding client (UFH-12 REPL-EMB-1.1 + 1.4).

Behind the ``repl_embedding_pool`` feature flag (default OFF in test and prod). With the flag
off, :func:`get_pooled_embedder` and :func:`get_sync_embedder` return None and nothing in this
package is constructed, so landing it changes no live behaviour. No consumer is migrated yet.

Modules:
    policy     knobs from orchestration/embedding_pool_policy.yaml
    topology   embedder/guarded-instance placement, derived from the stack declarations
    busy       is a guarded serving instance decoding? (dispatch ledger, live /slots)
    scheduler  slot admission, the neighbour cap, D1 selection order
    client     PooledEmbeddingClient (async) + SyncPooledEmbedder (blocking facade)
    cache      sha256 LRU
    fake       offline fakes for tests
"""

from __future__ import annotations

import asyncio
import threading
import weakref

from src.embedding_pool.cache import EmbeddingLRUCache, chunk_key
from src.embedding_pool.client import (
    MODE_DENSE,
    MODE_LEXICAL,
    EmbeddingOutcome,
    EmbeddingUnavailable,
    PooledEmbeddingClient,
    SyncPooledEmbedder,
)
from src.embedding_pool.policy import EmbeddingPoolPolicy, load_policy
from src.embedding_pool.scheduler import EmbeddingScheduler, Grant
from src.embedding_pool.topology import PoolTopology, live_topology

FEATURE_FLAG = "repl_embedding_pool"

_lock = threading.Lock()
_scheduler: EmbeddingScheduler | None = None
_cache: EmbeddingLRUCache | None = None
_clients: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, PooledEmbeddingClient]" = (
    weakref.WeakKeyDictionary()
)
_sync: SyncPooledEmbedder | None = None


def pool_enabled() -> bool:
    from src.features import features

    return bool(getattr(features(), FEATURE_FLAG, False))


def get_scheduler() -> EmbeddingScheduler:
    """The process-wide scheduler (one in-flight ledger, so the cap holds across clients)."""
    global _scheduler, _cache
    with _lock:
        if _scheduler is None:
            policy = load_policy()
            topology = live_topology(policy.neighbour_cap.guarded_roles)
            _scheduler = EmbeddingScheduler(topology, policy)
            _cache = EmbeddingLRUCache(policy.client.cache_max_entries)
        return _scheduler


def _shared_cache() -> EmbeddingLRUCache:
    get_scheduler()
    assert _cache is not None
    return _cache


def get_pooled_embedder() -> PooledEmbeddingClient | None:
    """Async client for the CURRENT event loop, or None when the flag is off."""
    if not pool_enabled():
        return None
    loop = asyncio.get_running_loop()
    scheduler = get_scheduler()
    with _lock:
        client = _clients.get(loop)
        if client is None:
            client = PooledEmbeddingClient(scheduler, cache=_cache)
            _clients[loop] = client
        return client


def get_sync_embedder() -> SyncPooledEmbedder | None:
    """Process-wide blocking facade, or None when the flag is off."""
    global _sync
    if not pool_enabled():
        return None
    scheduler = get_scheduler()
    with _lock:
        if _sync is None:
            _sync = SyncPooledEmbedder(scheduler, cache=_cache)
        return _sync


def reset_pool() -> None:
    """Drop the process-wide singletons (tests; after a topology change)."""
    global _scheduler, _cache, _sync
    with _lock:
        sync, _sync = _sync, None
        _scheduler = None
        _cache = None
        _clients.clear()
    if sync is not None:
        sync.close()


__all__ = [
    "EmbeddingLRUCache",
    "EmbeddingOutcome",
    "EmbeddingPoolPolicy",
    "EmbeddingScheduler",
    "EmbeddingUnavailable",
    "FEATURE_FLAG",
    "Grant",
    "MODE_DENSE",
    "MODE_LEXICAL",
    "PoolTopology",
    "PooledEmbeddingClient",
    "SyncPooledEmbedder",
    "chunk_key",
    "get_pooled_embedder",
    "get_scheduler",
    "get_sync_embedder",
    "live_topology",
    "load_policy",
    "pool_enabled",
    "reset_pool",
]
