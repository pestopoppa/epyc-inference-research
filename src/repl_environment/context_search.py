"""UFH-12 (minimal REPL-EMB-1.2/1.3 + 4.1): ``context.search`` over a request's context bundle.

Hybrid retrieval over the sections of a :class:`~src.repl_environment.context_bundle.ContextBundle`
that returns POINTERS, never text. Each hit is ``{section, start, end, line, score, via, mode}``:
char offsets into ``context.get(section)``. The model reads a hit through the existing counted
pull path, ``context.get(section, offset=start, max_chars=end - start)``, so OAB-12 pull-byte
accounting stays exact: a search moves zero bytes of section text into the REPL.

Pieces, all per request and built lazily on the first ``search`` (the bundle is constructed
twice per request -- once by the ChatRequest validator, once by the REPL executor -- and only
the REPL's copy is ever searched):

* **chunker** -- line-aware: a chunk is a run of whole lines of ONE section, at most
  ``CHUNK_MAX_CHARS`` chars; a single longer line is cut at the char limit. Every chunk keeps
  its section, char span and first line number.
* **lexical** -- Okapi BM25 in pure Python over lower-cased word tokens; identifiers like
  ``ggml_vec_dot_q8_0`` also contribute their ``_``-separated parts.
* **dense** -- an EXISTING embedding pool, through the orchestration API's own scheduler
  (``src.embedding_pool``: placement-aware slot admission, the busy-frontdoor neighbour cap,
  the sha256 chunk-vector cache). Nothing here starts, picks or addresses a server. There is
  no hash-vector fallback; every hit carries a ``mode`` saying what ranked it:

  - ``"hybrid"`` -- BM25 and dense both took part;
  - ``"lexical"`` -- dense is OFF for this request (the ``repl_embedding_pool`` flag is off, or
    the caller set no embedder): a deliberate lexical arm, not a failure;
  - ``"lexical_fallback:<reason>"`` -- an embedder exists but dense could not take part this
    time, e.g. ``index_timeout`` (the index build ran out of budget; it resumes on the next
    search), ``saturated``/``all_failed`` (the pool refused), ``too_large`` (above
    ``MAX_DENSE_CHUNKS``), ``query_<reason>`` (the query embedding failed) or
    ``model_mismatch``. An experiment counts these per search in ``context_pulls.search.modes``.

  **Index build budget and limits.** The chunk vectors are embedded in batches of
  ``DENSE_BATCH_CHUNKS``; each finished batch is kept on the index (and cached by the pool by
  sha256), so a build that runs out of time keeps its progress and the NEXT search resumes it --
  "lexical now, index later" (D1). One search waits at most
  ``index_build_timeout_s(remaining) = clamp(INDEX_TIMEOUT_BASE_S + remaining /
  INDEX_ASSUMED_TEXTS_PER_S, INDEX_TIMEOUT_MIN_S, INDEX_TIMEOUT_MAX_S)`` = clamp(5 + n/25,
  10, 60) s. 25 texts/s is the pool's rate under the busy-frontdoor cap (~27 texts/s, post-cap
  G1 load 2026-09-27; 45.6 texts/s uncapped, G2 2026-09-26), so up to ~1,400 chunks (~1.1 MB of
  bundle) index within one search, and a bundle at ``MAX_DENSE_CHUNKS`` (4,096 chunks, ~3.2 MB)
  needs ~164 s, i.e. three searches. An attempt that embeds NOTHING counts as a failure: a
  refusal is retried on later searches up to ``MAX_DENSE_ATTEMPTS`` times, and a timeout or a
  degenerate vector with no progress is not retried in this request.
* **fusion** -- ``src.trace.navigation.rrf_fuse`` over the BM25 and cosine rankings.

An index belongs to ONE embedding model (UFH-12 invariant 4): the dense vectors carry the
model id they were made with, and a query embedded by a different model is never scored
against them -- the dense side is rebuilt for the new model instead.
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

log = logging.getLogger(__name__)

#: Chunk size. The live BGE-large pool serves 512-token slots (``/slots`` n_ctx 512, 2026-10-01);
#: 800 chars of prose is ~200 tokens and of dense JSON/hex ~400, so a chunk fits one slot.
CHUNK_MAX_CHARS = 800
#: Hard clip on the text sent to the embedder (defence in depth for the slot limit).
DENSE_INPUT_MAX_CHARS = 800
SEARCH_DEFAULT_K = 8
SEARCH_MAX_K = 50
QUERY_MAX_CHARS = 2000
#: Candidates each ranker contributes to the fusion.
CANDIDATES_PER_RANKER = 64
#: Above this many chunks the dense side is skipped (``lexical_fallback:too_large``). At the
#: assumed capped rate a bundle this size indexes in ~164 s, three searches at the ceiling.
MAX_DENSE_CHUNKS = 4096
#: Chunks per embed call during an index build; progress is kept per finished batch.
DENSE_BATCH_CHUNKS = 128
#: Index-build attempts that embedded NOTHING (pool refusals) before dense gives up for the
#: request. An attempt that finished at least one batch is progress and never counts.
MAX_DENSE_ATTEMPTS = 3
#: Reasons a no-progress attempt is not retried at all: a build that embedded nothing within
#: its whole budget, or a degenerate vector, would only spend the same budget again.
_NO_RETRY = frozenset({"index_timeout", "degenerate"})
#: Index-build budget per search (s): clamp(BASE + remaining / RATE, MIN, MAX). See the module
#: docstring for where the rate comes from.
INDEX_ASSUMED_TEXTS_PER_S = 25.0
INDEX_TIMEOUT_BASE_S = 5.0
INDEX_TIMEOUT_MIN_S = 10.0
INDEX_TIMEOUT_MAX_S = 60.0
INDEX_ADMISSION_WAIT_S = 2.0
QUERY_TIMEOUT_S = 5.0
QUERY_ADMISSION_WAIT_S = 0.5
BM25_K1 = 1.2
BM25_B = 0.75
RRF_K = 60

#: Monotonic clock for the index-build deadline (a seam for tests).
_clock = time.perf_counter

MODE_HYBRID = "hybrid"
MODE_LEXICAL = "lexical"
MODE_FALLBACK_PREFIX = "lexical_fallback:"


def index_build_timeout_s(remaining_chunks: int) -> float:
    """One search's index-build budget for ``remaining_chunks`` not yet embedded."""
    want = INDEX_TIMEOUT_BASE_S + max(0, int(remaining_chunks)) / INDEX_ASSUMED_TEXTS_PER_S
    return float(min(INDEX_TIMEOUT_MAX_S, max(INDEX_TIMEOUT_MIN_S, want)))


def mode_for(reason: str | None) -> str:
    """``hybrid`` when dense took part; ``lexical`` when dense is off; else a labelled fallback."""
    if reason is None:
        return MODE_HYBRID
    if reason == "disabled":
        return MODE_LEXICAL
    return MODE_FALLBACK_PREFIX + reason


_WORD = re.compile(r"[A-Za-z0-9_]+")
#: Retrieval-query instructions for embedding models that were trained with one.
_QUERY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("bge-large-en-v1.5", "Represent this sentence for searching relevant passages: "),
    ("bge-base-en-v1.5", "Represent this sentence for searching relevant passages: "),
    ("bge-small-en-v1.5", "Represent this sentence for searching relevant passages: "),
)


# ───────────────────────────────────────────────────────────────────────────── chunker


@dataclass(frozen=True)
class Chunk:
    section: str
    start: int  # char offset within the section text
    end: int
    line: int  # 1-based line number of ``start`` within the section
    text: str


def _lines_keepends(text: str):
    pos = 0
    while pos < len(text):
        nl = text.find("\n", pos)
        end = len(text) if nl < 0 else nl + 1
        yield text[pos:end]
        pos = end


def chunk_sections(sections: Sequence[Any], max_chars: int = CHUNK_MAX_CHARS) -> list[Chunk]:
    """Line-aware chunks of every section (objects with ``name`` and ``text``)."""
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    chunks: list[Chunk] = []
    for s in sections:
        text = s.text
        if not text:
            continue
        cur_start: int | None = None
        cur_line = 1
        pos = 0
        lineno = 1
        # split on "\n" only, so line numbers match context.grep's
        for line in _lines_keepends(text):
            line_start, line_end = pos, pos + len(line)
            pos = line_end
            if cur_start is not None and line_end - cur_start > max_chars:
                chunks.append(Chunk(s.name, cur_start, line_start, cur_line, text[cur_start:line_start]))
                cur_start = None
            if line_end - line_start > max_chars:
                # one line longer than a chunk: cut it at the limit
                a = line_start
                while line_end - a > max_chars:
                    chunks.append(Chunk(s.name, a, a + max_chars, lineno, text[a:a + max_chars]))
                    a += max_chars
                cur_start, cur_line = a, lineno
            elif cur_start is None:
                cur_start, cur_line = line_start, lineno
            lineno += 1
        if cur_start is not None and cur_start < len(text):
            chunks.append(Chunk(s.name, cur_start, len(text), cur_line, text[cur_start:]))
    return chunks


# ───────────────────────────────────────────────────────────────────────────── lexical


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens; a ``_``-joined identifier also yields its parts."""
    out: list[str] = []
    for word in _WORD.findall(text):
        w = word.lower()
        out.append(w)
        if "_" in w:
            out.extend(p for p in w.split("_") if p and p != w)
    return out


class BM25Index:
    """Okapi BM25 over a fixed list of documents. Pure Python, no dependencies."""

    def __init__(self, docs: Sequence[str], *, k1: float = BM25_K1, b: float = BM25_B) -> None:
        self.k1, self.b = k1, b
        self._tf: list[Counter[str]] = [Counter(tokenize(d)) for d in docs]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = (sum(self._len) / len(self._len)) if self._len else 0.0
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(self._tf)
        self._idf = {t: math.log(1.0 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def __len__(self) -> int:
        return len(self._tf)

    def scores(self, query: str, ids: Sequence[int] | None = None) -> list[tuple[int, float]]:
        """``(doc id, score)`` for every candidate with a positive score, best first."""
        terms = [t for t in dict.fromkeys(tokenize(query)) if t in self._idf]
        if not terms:
            return []
        cand = range(len(self._tf)) if ids is None else ids
        avg = self._avg or 1.0
        out: list[tuple[int, float]] = []
        for i in cand:
            tf = self._tf[i]
            norm = self.k1 * (1.0 - self.b + self.b * self._len[i] / avg)
            score = 0.0
            for t in terms:
                f = tf.get(t)
                if f:
                    score += self._idf[t] * f * (self.k1 + 1.0) / (f + norm)
            if score > 0.0:
                out.append((i, score))
        out.sort(key=lambda r: (-r[1], r[0]))
        return out


# ───────────────────────────────────────────────────────────────────────────── dense


@dataclass(frozen=True)
class DenseResult:
    """Unit-row vectors (N x D) from one model, or the reason there are none."""

    vectors: Any  # numpy.ndarray | None
    model_id: str | None
    reason: str | None = None


class DenseEmbedder(Protocol):
    model_id: str

    def embed(self, texts: Sequence[str], *, timeout_s: float, admission_wait_s: float) -> DenseResult:
        ...


class OutcomeEmbedder:
    """Adapts any ``try_embed_many(texts, **kw) -> EmbeddingOutcome`` callable (the pool's
    ``SyncPooledEmbedder.try_embed_many``, or a fake's ``try_embed_many_sync``)."""

    def __init__(self, try_embed_many: Callable[..., Any], model_id: str) -> None:
        self._try = try_embed_many
        self.model_id = str(model_id)

    def embed(self, texts: Sequence[str], *, timeout_s: float, admission_wait_s: float) -> DenseResult:
        try:
            outcome = self._try(list(texts), timeout_s=timeout_s, admission_wait_s=admission_wait_s)
        except Exception as exc:  # noqa: BLE001 -- a dense failure degrades to labelled lexical
            reason = getattr(exc, "reason", None) or type(exc).__name__
            return DenseResult(None, self.model_id, str(reason))
        if getattr(outcome, "is_dense", False) and outcome.vectors is not None:
            return DenseResult(outcome.vectors, self.model_id)
        return DenseResult(None, self.model_id, getattr(outcome, "reason", None) or "unavailable")


def pool_dense_embedder() -> DenseEmbedder | None:
    """The orchestration API's pooled embedder (UFH-12 REPL-EMB-1.1/1.4), or None while the
    ``repl_embedding_pool`` flag is off. Scheduling, placement and the neighbour cap are the
    pool's; this only adapts its sync facade."""
    try:
        from src.embedding_pool import get_sync_embedder

        sync = get_sync_embedder()
    except Exception:  # noqa: BLE001 -- a broken pool must not break the REPL
        log.warning("context.search: embedding pool unavailable", exc_info=True)
        return None
    if sync is None:
        return None
    return OutcomeEmbedder(sync.try_embed_many, sync.client.model_id)


def query_text(model_id: str | None, query: str) -> str:
    low = (model_id or "").lower()
    for marker, prefix in _QUERY_PREFIXES:
        if marker in low:
            return prefix + query
    return query


# ───────────────────────────────────────────────────────────────────────────── index


class ContextSearchIndex:
    """Chunks + BM25 + (optionally) the dense vectors of ONE embedding model."""

    def __init__(self, sections: Sequence[Any], *, chunk_max_chars: int = CHUNK_MAX_CHARS) -> None:
        t0 = time.perf_counter()
        self.chunk_max_chars = int(chunk_max_chars)
        self.chunks = chunk_sections(sections, self.chunk_max_chars)
        self.bm25 = BM25Index([c.text for c in self.chunks])
        self.lexical_build_ms = round((time.perf_counter() - t0) * 1000, 2)
        self._by_section: dict[str, list[int]] = {}
        for i, c in enumerate(self.chunks):
            self._by_section.setdefault(c.section, []).append(i)
        self.vectors: Any = None  # N x D once every chunk is embedded
        self.model_id: str | None = None  # the model of ``vectors`` AND of any partial rows
        self._parts: list[Any] = []  # finished batches, chunk order
        self.dense_done = 0  # chunks embedded so far
        self.dense_reason: str | None = "not_built"
        self.dense_attempts = 0
        self.dense_failures = 0  # attempts that embedded nothing
        self.dense_build_ms: float | None = None
        self.dense_timeout_s: float | None = None  # the last attempt's budget

    # -- dense side --------------------------------------------------------------------------
    def _reset_dense(self) -> None:
        self.vectors, self.model_id, self._parts = None, None, []
        self.dense_done = self.dense_attempts = self.dense_failures = 0
        self.dense_reason, self.dense_build_ms, self.dense_timeout_s = "not_built", None, None

    def ensure_dense(self, embedder: DenseEmbedder | None) -> str | None:
        """Make the dense side usable with ``embedder``; return None or the reason it isn't.

        Embeds the chunks not yet embedded, batch by batch, within
        ``index_build_timeout_s(remaining)``; finished batches are kept, so a timed-out build
        resumes on the next call."""
        if embedder is None:
            return "disabled"
        if self.model_id is not None and self.model_id != embedder.model_id:
            # invariant 4: never score a query from one model against another's vectors, and
            # never finish one model's partial index with another model's batches
            self._reset_dense()
        if self.vectors is not None:
            return None
        if not self.chunks:
            return "empty"
        if len(self.chunks) > MAX_DENSE_CHUNKS:
            self.dense_reason = "too_large"
            return self.dense_reason
        if self.dense_failures and (
            self.dense_failures >= MAX_DENSE_ATTEMPTS or self.dense_reason in _NO_RETRY
        ):
            return self.dense_reason or "unavailable"
        import numpy as np

        self.model_id = embedder.model_id
        self.dense_attempts += 1
        budget = index_build_timeout_s(len(self.chunks) - self.dense_done)
        self.dense_timeout_s = budget
        t0 = _clock()
        deadline = t0 + budget
        progressed = False
        reason: str | None = None
        while self.dense_done < len(self.chunks):
            left = deadline - _clock()
            if left <= 0:
                reason = "index_timeout"
                break
            batch = self.chunks[self.dense_done : self.dense_done + DENSE_BATCH_CHUNKS]
            res = embedder.embed(
                [c.text[:DENSE_INPUT_MAX_CHARS] for c in batch],
                timeout_s=left,
                admission_wait_s=INDEX_ADMISSION_WAIT_S,
            )
            if res.vectors is None or len(res.vectors) != len(batch):
                raw = res.reason or "shape_mismatch"
                reason = "index_timeout" if raw == "timeout" else raw
                break
            self._parts.append(np.asarray(res.vectors, dtype=np.float32))
            self.dense_done += len(batch)
            progressed = True
        self.dense_build_ms = round((self.dense_build_ms or 0.0) + (_clock() - t0) * 1000, 2)
        if self.dense_done == len(self.chunks):
            self.vectors = np.vstack(self._parts)
            self._parts, self.dense_reason = [], None
            return None
        if not progressed:
            self.dense_failures += 1
        self.dense_reason = reason or "unavailable"
        return self.dense_reason

    # -- query -------------------------------------------------------------------------------
    def candidates(self, section: str | None) -> list[int] | None:
        if section is None:
            return None
        return list(self._by_section.get(section, []))

    def search(
        self, query: str, k: int, *, section: str | None, embedder: DenseEmbedder | None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Fused pointers and a small meta record (mode, reason, timings)."""
        from src.trace.navigation import rrf_fuse

        ids = self.candidates(section)
        lexical = [
            {"id": i, "_rank_source": "bm25"}
            for i, _ in self.bm25.scores(query, ids)[:CANDIDATES_PER_RANKER]
        ]
        ranked = [lexical]
        reason = self.ensure_dense(embedder)
        dense_ms = None
        if reason is None:
            t0 = time.perf_counter()
            assert embedder is not None
            q = embedder.embed(
                [query_text(self.model_id, query)[:DENSE_INPUT_MAX_CHARS]],
                timeout_s=QUERY_TIMEOUT_S,
                admission_wait_s=QUERY_ADMISSION_WAIT_S,
            )
            dense_ms = round((time.perf_counter() - t0) * 1000, 2)
            if q.vectors is None or len(q.vectors) != 1:
                reason = "query_" + (q.reason or "failed")
            elif q.model_id != self.model_id:
                reason = "model_mismatch"
            else:
                ranked.append(self._dense_ranking(q.vectors[0], ids))
        mode = mode_for(reason)
        fused = rrf_fuse(ranked, key="id", k=RRF_K, limit=k)
        hits = []
        for row in fused:
            c = self.chunks[row["id"]]
            hits.append(
                {
                    "section": c.section,
                    "start": c.start,
                    "end": c.end,
                    "line": c.line,
                    "score": row["_rrf_score"],
                    "via": row["_rrf_sources"],
                    "mode": mode,
                }
            )
        meta = {"mode": mode, "reason": reason, "dense_query_ms": dense_ms}
        return hits, meta

    def _dense_ranking(self, qvec: Any, ids: Sequence[int] | None) -> list[dict[str, Any]]:
        import numpy as np

        mat = self.vectors if ids is None else self.vectors[list(ids)]
        if len(mat) == 0:
            return []
        sims = mat @ np.asarray(qvec, dtype=np.float32)
        order = np.argsort(-sims, kind="stable")[:CANDIDATES_PER_RANKER]
        index_of = list(range(len(self.chunks))) if ids is None else list(ids)
        return [{"id": index_of[int(j)], "_rank_source": "dense"} for j in order]

    def describe(self) -> dict[str, Any]:
        return {
            "chunks": len(self.chunks),
            "chunk_max_chars": self.chunk_max_chars,
            "embedder_model": self.model_id,
            "dense": self.vectors is not None,
            "dense_reason": self.dense_reason,
            "dense_chunks_done": self.dense_done,
            "dense_attempts": self.dense_attempts,
            "dense_failures": self.dense_failures,
            "dense_timeout_s": self.dense_timeout_s,
            "lexical_build_ms": self.lexical_build_ms,
            "dense_build_ms": self.dense_build_ms,
        }


__all__ = [
    "BM25Index",
    "CHUNK_MAX_CHARS",
    "Chunk",
    "ContextSearchIndex",
    "DenseEmbedder",
    "DenseResult",
    "MODE_FALLBACK_PREFIX",
    "MODE_HYBRID",
    "MODE_LEXICAL",
    "OutcomeEmbedder",
    "SEARCH_DEFAULT_K",
    "SEARCH_MAX_K",
    "chunk_sections",
    "index_build_timeout_s",
    "mode_for",
    "pool_dense_embedder",
    "query_text",
    "tokenize",
]
