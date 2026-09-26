"""Offline fakes for the embedding pool. Nothing here opens a socket.

* :func:`fake_vector` — deterministic bag-of-words vector, so similar texts are close (useful
  for retrieval tests) and the empty string is the zero vector (useful for the
  degenerate-vector guard). Returned UN-normalised on purpose.
* :class:`FakeEmbeddingServers` — an ``httpx.MockTransport`` that emulates llama-server
  ``/embedding`` (list batches), ``/slots`` and ``/health`` on any number of ports, with
  per-port latency, failure and degenerate modes, and records the in-flight texts per port
  (current and peak) so tests can assert the slot limit and the neighbour cap.
* :class:`FakePooledEmbedder` — the client's public API with no HTTP at all, for consumer
  tests (chunk index, hybrid search) that only need vectors or a lexical label.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from typing import Any, Iterable, Sequence

import numpy as np

from src.embedding_pool.client import (
    MODE_DENSE,
    MODE_LEXICAL,
    EmbeddingOutcome,
    EmbeddingUnavailable,
)

_WORD = re.compile(r"[a-z0-9_]+")


def fake_vector(text: str, dim: int = 32) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for word in _WORD.findall(text.lower()):
        h = hashlib.sha256(word.encode()).digest()
        vec[int.from_bytes(h[:4], "little") % dim] += (
            1.0  # counts never cancel: non-empty -> non-zero
        )
    return vec * 3.0


class FakeEmbeddingServers:
    def __init__(
        self, ports: Iterable[int], *, dim: int = 32, delay_s: float = 0.0, slots_per_port: int = 4
    ) -> None:
        self.ports = set(int(p) for p in ports)
        self.dim = dim
        self.delay_s = delay_s
        self.delay_by_port: dict[int, float] = {}
        self.slots_per_port = slots_per_port
        self.fail_ports: set[int] = set()
        self.degenerate_ports: set[int] = set()
        self.slots_fail_ports: set[int] = set()
        self.processing: dict[int, int] = {}
        self.inflight: dict[int, int] = {}
        self.peak: dict[int, int] = {}
        self.requests: list[tuple[int, int]] = []
        self.slots_reads: list[int] = []
        self.completed = 0
        self.cancelled = 0

    def transport(self) -> Any:
        import httpx

        return httpx.MockTransport(self.handle)

    def total_inflight(self) -> int:
        return sum(self.inflight.values())

    async def handle(self, request: Any) -> Any:
        import httpx

        port = request.url.port
        path = request.url.path
        if path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if path == "/slots":
            self.slots_reads.append(port)
            if port in self.slots_fail_ports:
                return httpx.Response(503, json={"error": "unavailable"})
            busy = self.processing.get(port, 0)
            return httpx.Response(
                200,
                json=[
                    {"id": i, "n_ctx": 4096, "is_processing": i < busy, "n_prompt_tokens": 0}
                    for i in range(max(self.slots_per_port, busy))
                ],
            )
        if path != "/embedding" or port not in self.ports:
            return httpx.Response(404, json={"error": "not found"})
        body = json.loads(request.content or b"{}")
        content = body.get("content")
        texts = content if isinstance(content, list) else [content]
        n = len(texts)
        self.requests.append((port, n))
        self.inflight[port] = self.inflight.get(port, 0) + n
        self.peak[port] = max(self.peak.get(port, 0), self.inflight[port])
        try:
            delay = self.delay_by_port.get(port, self.delay_s)
            if delay:
                await asyncio.sleep(delay)
            if port in self.fail_ports:
                return httpx.Response(503, json={"error": "loading model"})
            out = []
            for i, t in enumerate(texts):
                vec = (
                    np.zeros(self.dim, dtype=np.float32)
                    if port in self.degenerate_ports
                    else fake_vector(t, self.dim)
                )
                out.append({"index": i, "embedding": [vec.tolist()]})
            self.completed += 1
            return httpx.Response(200, json=out)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.inflight[port] -= n


class FakePooledEmbedder:
    """Same public surface as PooledEmbeddingClient / SyncPooledEmbedder, no HTTP."""

    def __init__(self, dim: int = 32, *, available: bool = True, reason: str = "saturated") -> None:
        self.dim = dim
        self.available = available
        self.reason = reason
        self.calls: list[list[str]] = []
        self.model_id = "fake-embedder"

    @property
    def embedding_dim(self) -> int:
        return self.dim

    def _vectors(self, texts: Sequence[str]) -> np.ndarray:
        self.calls.append(list(texts))
        if not self.available:
            raise EmbeddingUnavailable(self.reason, "fake embedder set unavailable")
        rows = []
        for t in texts:
            v = fake_vector(t, self.dim)
            n = float(np.linalg.norm(v))
            if n == 0.0:
                raise EmbeddingUnavailable("degenerate", "empty text")
            rows.append(v / n)
        return np.stack(rows).astype(np.float32) if rows else np.zeros((0, self.dim), np.float32)

    def _outcome(self, texts: Sequence[str]) -> EmbeddingOutcome:
        try:
            return EmbeddingOutcome(mode=MODE_DENSE, vectors=self._vectors(texts))
        except EmbeddingUnavailable as exc:
            return EmbeddingOutcome(
                mode=MODE_LEXICAL, vectors=None, reason=exc.reason, detail=exc.detail
            )

    async def embed_many(self, texts: Sequence[str], **_: Any) -> np.ndarray:
        return self._vectors(texts)

    async def embed(self, text: str, **_: Any) -> np.ndarray:
        return self._vectors([text])[0]

    async def try_embed_many(self, texts: Sequence[str], **_: Any) -> EmbeddingOutcome:
        return self._outcome(texts)

    def embed_many_sync(self, texts: Sequence[str], **_: Any) -> np.ndarray:
        return self._vectors(texts)

    def try_embed_many_sync(self, texts: Sequence[str], **_: Any) -> EmbeddingOutcome:
        return self._outcome(texts)


__all__ = ["FakeEmbeddingServers", "FakePooledEmbedder", "fake_vector"]
