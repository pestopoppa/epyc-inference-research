"""Pooled async embedding client (UFH-12 REPL-EMB-1.1) with the neighbour cap (REPL-EMB-1.4).

What it does, and what the old ``ParallelEmbedderClient`` did instead:

* one shared ``httpx.AsyncClient`` per client, kept open across calls (the old sync wrapper
  closed its client after every call);
* placement-aware slot scheduling through :class:`EmbeddingScheduler` — idle instance
  anywhere, else the requester's hardware, else lexical — instead of probe-first "first
  responder wins" with six ``/health`` probes per text;
* list batches (``POST /embedding {"content": [...]}``), each carrying at most the chosen
  instance's headroom (<= its ``-np`` slots), instead of one request per text;
* per-request timeouts, a wall-clock bound per call, and retry of a failed chunk on the next
  instance (the failed one sits out ``failure_backoff_s``);
* L2 normalisation plus a degenerate-vector guard (non-finite, wrong dimension, or a norm
  below ``min_norm`` is rejected and never cached);
* a sha256 LRU cache keyed by (model, text);
* NO hash-vector fallback. When the pool cannot embed, the call raises
  :class:`EmbeddingUnavailable` with a reason, and :meth:`try_embed_many` returns a result
  LABELLED lexical, so the caller degrades explicitly.

Lifecycle (R2 — nothing outlives ``/chat``): every HTTP request a call starts is awaited or
cancelled before the call returns or raises, including when the calling task is cancelled
or the optional ``cancel_event`` trips; slot reservations are released in ``finally``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from src.embedding_pool.cache import EmbeddingLRUCache, chunk_key
from src.embedding_pool.policy import EmbeddingPoolPolicy
from src.embedding_pool.scheduler import EmbeddingScheduler, Grant

log = logging.getLogger(__name__)

MODE_DENSE = "dense"
MODE_LEXICAL = "lexical"

REASONS = frozenset(
    {"no_instances", "saturated", "all_failed", "degenerate", "timeout", "cancelled", "disabled"}
)


class EmbeddingUnavailable(RuntimeError):
    """The pool cannot embed this call now. Degrade to (labelled) lexical; never substitute a
    pseudo-embedding."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason if reason in REASONS else "all_failed"
        self.detail = detail
        super().__init__(f"embedding unavailable ({self.reason}){': ' + detail if detail else ''}")


@dataclass(frozen=True)
class EmbeddingOutcome:
    """Result of :meth:`PooledEmbeddingClient.try_embed_many`: dense vectors, or an explicit
    lexical label with the reason."""

    mode: str
    vectors: np.ndarray | None
    reason: str | None = None
    detail: str = ""

    @property
    def is_dense(self) -> bool:
        return self.mode == MODE_DENSE


class _InstanceError(Exception):
    pass


class _DegenerateVector(Exception):
    pass


def normalise_vector(raw: Any, *, dim: int | None, min_norm: float) -> np.ndarray:
    """L2-normalise one vector; raise ``_DegenerateVector`` for a vector no index should hold."""
    vec = np.asarray(raw, dtype=np.float32).ravel()
    if vec.size == 0:
        raise _DegenerateVector("empty vector")
    if dim is not None and vec.size != dim:
        raise _DegenerateVector(f"dimension {vec.size} != expected {dim}")
    if not np.all(np.isfinite(vec)):
        raise _DegenerateVector("non-finite component")
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm < min_norm:
        raise _DegenerateVector(f"norm {norm:.3g} below {min_norm:.3g}")
    return (vec / norm).astype(np.float32)


def parse_embedding_response(data: Any, n: int) -> list[Any]:
    """Vectors from a llama-server ``/embedding`` list reply (or an OpenAI-style ``data``)."""
    items = data.get("data") if isinstance(data, dict) else data
    if not isinstance(items, list) or len(items) != n:
        raise _InstanceError(
            f"expected {n} embeddings, got {type(items).__name__}"
            f"{'[' + str(len(items)) + ']' if isinstance(items, list) else ''}"
        )
    if all(isinstance(it, dict) and isinstance(it.get("index"), int) for it in items):
        items = sorted(items, key=lambda it: it["index"])
    out: list[Any] = []
    for it in items:
        emb = it.get("embedding") if isinstance(it, dict) else it
        if isinstance(emb, list) and emb and isinstance(emb[0], list):
            if len(emb) != 1:
                raise _InstanceError("per-token embeddings returned; the server must pool")
            emb = emb[0]
        if not isinstance(emb, list):
            raise _InstanceError("malformed embedding item")
        out.append(emb)
    return out


@dataclass
class _Chunk:
    indices: list[int]
    tried: frozenset[int]
    degenerate_hits: int = 0


class PooledEmbeddingClient:
    """Async client over one embedding model's instance pool. Bind it to one event loop."""

    def __init__(
        self,
        scheduler: EmbeddingScheduler,
        *,
        cache: EmbeddingLRUCache | None = None,
        policy: EmbeddingPoolPolicy | None = None,
        transport: Any = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.scheduler = scheduler
        self.policy = policy or scheduler.policy
        self.cache = (
            cache if cache is not None else EmbeddingLRUCache(self.policy.client.cache_max_entries)
        )
        self._transport = transport
        self._clock = clock
        self._http: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._dim: int | None = self.policy.client.embedding_dim
        self._closed = False

    @property
    def model_id(self) -> str:
        return self.scheduler.topology.model_id

    @property
    def embedding_dim(self) -> int | None:
        return self._dim

    async def _client(self) -> Any:
        if self._closed:
            raise RuntimeError("PooledEmbeddingClient is closed")
        loop = asyncio.get_running_loop()
        if self._http is None:
            import httpx

            c = self.policy.client
            total_slots = sum(e.slots for e in self.scheduler.topology.embedders)
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=c.connect_timeout_s,
                    read=c.request_timeout_s,
                    write=c.request_timeout_s,
                    pool=c.request_timeout_s,
                ),
                limits=httpx.Limits(
                    max_connections=total_slots + 2 * len(self.scheduler.topology.guarded) + 4,
                    max_keepalive_connections=total_slots + 4,
                ),
                transport=self._transport,
                trust_env=False,
            )
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError(
                "PooledEmbeddingClient is bound to another event loop; "
                "use one client per loop (get_pooled_embedder does this)"
            )
        return self._http

    async def aclose(self) -> None:
        self._closed = True
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def __aenter__(self) -> "PooledEmbeddingClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # -- one instance request ---------------------------------------------------------------
    async def _run(self, http: Any, grant: Grant, texts: list[str]) -> list[np.ndarray]:
        try:
            try:
                resp = await http.post(f"{grant.url}/embedding", json={"content": texts})
            except Exception as exc:  # noqa: BLE001 — httpx transport/timeout errors (not CancelledError)
                raise _InstanceError(f"{type(exc).__name__}: {exc}") from exc
            if resp.status_code != 200:
                raise _InstanceError(f"HTTP {resp.status_code}")
            try:
                payload = resp.json()
            except ValueError as exc:
                raise _InstanceError("reply is not JSON") from exc
            raws = parse_embedding_response(payload, len(texts))
            c = self.policy.client
            vecs = [normalise_vector(r, dim=self._dim, min_norm=c.min_norm) for r in raws]
            if self._dim is None and vecs:
                self._dim = int(vecs[0].size)
            return vecs
        finally:
            self.scheduler.release(grant)

    # -- public API -------------------------------------------------------------------------
    async def embed(self, text: str, **kwargs: Any) -> np.ndarray:
        return (await self.embed_many([text], **kwargs))[0]

    async def embed_many(
        self,
        texts: Sequence[str],
        *,
        requester_port: int | None = None,
        timeout_s: float | None = None,
        admission_wait_s: float | None = None,
        cancel_event: threading.Event | None = None,
        use_cache: bool = True,
    ) -> np.ndarray:
        """Embed ``texts`` (N x D float32, unit rows, input order) or raise EmbeddingUnavailable."""
        texts = list(texts)
        if any(not isinstance(t, str) for t in texts):
            raise TypeError("embed_many takes a sequence of str")
        if not texts:
            return np.zeros((0, self._dim or 0), dtype=np.float32)
        if not self.scheduler.topology.ports:
            raise EmbeddingUnavailable("no_instances")
        c = self.policy.client
        model = self.model_id
        results: list[np.ndarray | None] = [None] * len(texts)
        positions: dict[str, list[int]] = {}
        for i, t in enumerate(texts):
            positions.setdefault(t, []).append(i)
        to_fetch: list[str] = []
        for t, idxs in positions.items():
            hit = self.cache.get(chunk_key(model, t)) if use_cache else None
            if hit is not None:
                for i in idxs:
                    results[i] = hit
            else:
                to_fetch.append(t)
        if use_cache and len(to_fetch) < len(positions):
            self.scheduler.note("cache_hits", len(positions) - len(to_fetch))
        if to_fetch:
            fetched = await self._fetch(
                to_fetch,
                requester_port=requester_port,
                timeout_s=c.call_timeout_s if timeout_s is None else float(timeout_s),
                wait_s=c.admission_wait_s if admission_wait_s is None else float(admission_wait_s),
                cancel_event=cancel_event,
            )
            for t, vec in zip(to_fetch, fetched):
                if use_cache:
                    self.cache.put(chunk_key(model, t), vec)
                for i in positions[t]:
                    results[i] = vec
        return np.stack([np.asarray(r, dtype=np.float32) for r in results])  # type: ignore[arg-type]

    async def try_embed_many(self, texts: Sequence[str], **kwargs: Any) -> EmbeddingOutcome:
        """Like :meth:`embed_many` but never raises EmbeddingUnavailable: returns a LABELLED
        lexical outcome instead (invariant 3 — degrade explicitly, never a pseudo-vector)."""
        try:
            vecs = await self.embed_many(texts, **kwargs)
        except EmbeddingUnavailable as exc:
            return EmbeddingOutcome(
                mode=MODE_LEXICAL, vectors=None, reason=exc.reason, detail=exc.detail
            )
        return EmbeddingOutcome(mode=MODE_DENSE, vectors=vecs)

    async def _fetch(
        self,
        texts: list[str],
        *,
        requester_port: int | None,
        timeout_s: float,
        wait_s: float,
        cancel_event: threading.Event | None,
    ) -> list[np.ndarray]:
        sched = self.scheduler
        poll = self.policy.client.poll_interval_s
        n_ports = len(sched.topology.ports)
        http = await self._client()
        deadline = self._clock() + timeout_s
        fetched: list[np.ndarray | None] = [None] * len(texts)
        pending: deque[_Chunk] = deque([_Chunk(list(range(len(texts))), frozenset())])
        running: dict[asyncio.Future, tuple[_Chunk, Grant]] = {}
        stalled_since: float | None = None
        try:
            while pending or running:
                if cancel_event is not None and cancel_event.is_set():
                    raise EmbeddingUnavailable("cancelled")
                now = self._clock()
                if now >= deadline:
                    raise EmbeddingUnavailable("timeout", f"{timeout_s:.1f}s call budget spent")
                await sched.refresh_busy(http)
                for chunk in list(pending):
                    if len(chunk.tried) >= n_ports:
                        raise EmbeddingUnavailable(
                            "degenerate" if chunk.degenerate_hits else "all_failed",
                            f"tried every instance {sorted(chunk.tried)}",
                        )
                    grants = sched.try_reserve(
                        len(chunk.indices), requester_port=requester_port, exclude=chunk.tried
                    )
                    if not grants:
                        continue
                    pending.remove(chunk)
                    pos = 0
                    for g in grants:
                        sub = _Chunk(
                            chunk.indices[pos : pos + g.n], chunk.tried, chunk.degenerate_hits
                        )
                        pos += g.n
                        task = asyncio.ensure_future(
                            self._run(http, g, [texts[i] for i in sub.indices])
                        )
                        running[task] = (sub, g)
                    if pos < len(chunk.indices):
                        pending.append(
                            _Chunk(chunk.indices[pos:], chunk.tried, chunk.degenerate_hits)
                        )
                if running:
                    stalled_since = None
                    remaining = max(0.0, deadline - self._clock())
                    # Poll while work is queued (busy state / headroom can change) or a
                    # cancel flag must be watched; otherwise just wait for a completion.
                    watch = bool(pending) or cancel_event is not None
                    done, _ = await asyncio.wait(
                        list(running),
                        timeout=min(poll, remaining) if watch else remaining,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in done:
                        sub, g = running.pop(task)
                        exc = task.exception()
                        if exc is None:
                            sched.mark_success(g.port)
                            for i, vec in zip(sub.indices, task.result()):
                                fetched[i] = vec
                        elif isinstance(exc, _DegenerateVector):
                            sched.note("degenerate_vectors")
                            log.warning(
                                "embedding pool: degenerate vector from :%d (%s)", g.port, exc
                            )
                            if sub.degenerate_hits >= 1:
                                raise EmbeddingUnavailable("degenerate", str(exc))
                            pending.append(
                                _Chunk(sub.indices, sub.tried | {g.port}, sub.degenerate_hits + 1)
                            )
                        elif isinstance(exc, _InstanceError):
                            sched.mark_failure(g.port)
                            sched.note("retries")
                            log.info(
                                "embedding pool: :%d failed (%s); retrying on the next instance",
                                g.port,
                                exc,
                            )
                            pending.append(
                                _Chunk(sub.indices, sub.tried | {g.port}, sub.degenerate_hits)
                            )
                        else:
                            raise exc
                elif pending:
                    if stalled_since is None:
                        stalled_since = now
                    if now - stalled_since >= wait_s:
                        if not sched.healthy_ports():
                            raise EmbeddingUnavailable(
                                "all_failed", "every instance is in failure backoff"
                            )
                        sched.note("saturated")
                        raise EmbeddingUnavailable(
                            "saturated", "no instance has headroom under the policy"
                        )
                    await asyncio.sleep(min(poll, max(0.0, deadline - now)))
        finally:
            if running:
                for task in running:
                    task.cancel()
                await asyncio.gather(*running, return_exceptions=True)
        return fetched  # type: ignore[return-value]

    def stats(self) -> dict[str, Any]:
        return {
            "scheduler": self.scheduler.stats(),
            "cache_entries": len(self.cache),
            "cache_hits": self.cache.hits,
            "cache_misses": self.cache.misses,
        }


class SyncPooledEmbedder:
    """Blocking facade for sync callers. Owns one background event loop and ONE long-lived
    async client on it; nothing is closed between calls. Call :meth:`close` when done."""

    def __init__(
        self,
        scheduler: EmbeddingScheduler,
        *,
        cache: EmbeddingLRUCache | None = None,
        policy: EmbeddingPoolPolicy | None = None,
        transport: Any = None,
    ) -> None:
        self.policy = policy or scheduler.policy
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="embedding-pool-sync", daemon=True
        )
        self._thread.start()
        self.client = PooledEmbeddingClient(
            scheduler, cache=cache, policy=self.policy, transport=transport
        )
        self._closed = False
        self._close_lock = threading.Lock()

    def _submit(self, coro: Any, timeout_s: float) -> Any:
        if self._closed:
            coro.close()
            raise RuntimeError("SyncPooledEmbedder is closed")
        if threading.current_thread() is self._thread:
            coro.close()
            raise RuntimeError(
                "SyncPooledEmbedder called from its own loop thread (would deadlock)"
            )
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return fut.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            fut.cancel()
            raise EmbeddingUnavailable("timeout", "sync facade wait exceeded") from None

    def _budget(self, kwargs: dict[str, Any]) -> float:
        c = self.policy.client
        call = kwargs.get("timeout_s") or c.call_timeout_s
        wait = kwargs.get("admission_wait_s") or c.admission_wait_s
        return float(call) + float(wait) + 5.0

    def embed_many(self, texts: Sequence[str], **kwargs: Any) -> np.ndarray:
        return self._submit(self.client.embed_many(texts, **kwargs), self._budget(kwargs))

    def embed(self, text: str, **kwargs: Any) -> np.ndarray:
        return self.embed_many([text], **kwargs)[0]

    def try_embed_many(self, texts: Sequence[str], **kwargs: Any) -> EmbeddingOutcome:
        return self._submit(self.client.try_embed_many(texts, **kwargs), self._budget(kwargs))

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            try:
                asyncio.run_coroutine_threadsafe(self.client.aclose(), self._loop).result(timeout=5)
            except Exception:  # noqa: BLE001
                log.debug("embedding pool: sync facade close failed", exc_info=True)
            self._closed = True
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
            if not self._thread.is_alive():
                self._loop.close()

    def __enter__(self) -> "SyncPooledEmbedder":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = [
    "EmbeddingOutcome",
    "EmbeddingUnavailable",
    "MODE_DENSE",
    "MODE_LEXICAL",
    "PooledEmbeddingClient",
    "REASONS",
    "SyncPooledEmbedder",
    "normalise_vector",
    "parse_embedding_response",
]
