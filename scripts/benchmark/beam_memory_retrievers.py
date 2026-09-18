#!/usr/bin/env python3
"""M-12 B2: the two BEAM memory arms, over identical chunks with an identical budget.

M-12c(6) requires identical chunking and an identical retrieval budget for the
naive-memory control and for the arm under test. M-12c(7) requires one prompt
contract for every arm. This module owns the chunking and both retrievers.
``BEAMAdapter`` owns the prompt.

* **Chunking = BEAM ``pair_chunk``.** Each chunk is one user turn together with
  the assistant turn(s) that answer it, rendered exactly as the full-history
  transcript renders them (time anchor, role label). Both roles are ingested
  (M-12c(2)).
* **``rag`` (the naive-memory control) = ``pair_chunk`` x BM25.** Okapi BM25
  (k1 = 1.5, b = 0.75) runs in process over lower-cased alphanumeric tokens, and
  the whole question is the query. This is one of the retriever cells BEAM's own
  harness offers (``answer_generation.py:275-280``). It is fully local and uses
  no model.
* **``trace`` (the arm under test) = the same chunks through the production
  memory surface.** Every chunk is one ``Event`` in a PRIVATE epyc-orchestrator
  trace store: SQLite + FTS5, built with the orchestrator's own
  ``ensure_schema``, with ``session_id`` set to the conversation. Chunks are
  found through ``src.trace.navigation.search_records(order="relevance")``,
  which is bm25-ordered since orchestrator ``6302b381``. An orchestrator older
  than that has no ``order`` parameter; that raises
  :class:`TraceRetrieverUnavailable` and never silently falls back to recency
  order.

Both retrievers return ``top_k`` chunks in CONVERSATION order, because order is
information the temporal and event-ordering questions need. Both are offline:
no socket, no embedding service, and the production trace DB is never touched.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import weakref
from collections import Counter
from pathlib import Path
from typing import Mapping, Optional, Sequence

CHUNKING = "pair_chunk"
BM25_RETRIEVER = "bm25_okapi_k1.5_b0.75"
TRACE_RETRIEVER = "trace_fts5_navigation_relevance"
TRACE_SOURCE = "beam_pair_chunk"
BM25_K1 = 1.5
BM25_B = 0.75
_TOKEN = re.compile(r"[a-z0-9]+")


def render_message(msg: Mapping) -> str:
    """One message, rendered exactly as ``BEAMAdapter._render_transcript`` renders it."""
    role = str(msg.get("role", "")).strip().lower()
    label = "User" if role == "user" else "Assistant" if role == "assistant" else role
    anchor = msg.get("time_anchor")
    prefix = f"[{anchor}] " if anchor else ""
    return f"{prefix}{label}: {msg.get('content', '')}"


def pair_chunks(messages: Sequence[Mapping]) -> list[str]:
    """BEAM ``pair_chunk``: a chunk starts at each user turn and runs to the next one.

    Messages before the first user turn form their own leading chunk, so no turn is
    ever dropped. Joining the chunks with a blank line reproduces the transcript.
    """
    chunks: list[list[str]] = []
    for msg in messages:
        is_user = str(msg.get("role", "")).strip().lower() == "user"
        if is_user or not chunks:
            chunks.append([])
        chunks[-1].append(render_message(msg))
    return ["\n\n".join(lines) for lines in chunks]


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


class BM25PairChunkRetriever:
    """``rag`` arm: Okapi BM25 over each conversation's pair chunks."""

    name = BM25_RETRIEVER

    def __init__(self, chunks_by_conversation: Mapping[str, Sequence[str]]):
        self._chunks = {cid: list(chunks) for cid, chunks in chunks_by_conversation.items()}
        self._index = {}
        for cid, chunks in self._chunks.items():
            docs = [Counter(_tokens(c)) for c in chunks]
            lengths = [sum(d.values()) for d in docs]
            df: Counter = Counter()
            for d in docs:
                df.update(d.keys())
            n = len(docs)
            idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
            avg = (sum(lengths) / n) if n else 0.0
            self._index[cid] = (docs, lengths, idf, avg)

    def scores(self, question: str, conversation_id: str) -> list[float]:
        docs, lengths, idf, avg = self._index[conversation_id]
        query = _tokens(question)
        out = []
        for doc, length in zip(docs, lengths):
            s = 0.0
            for term in query:
                tf = doc.get(term, 0)
                if tf:
                    norm = tf + BM25_K1 * (1 - BM25_B + BM25_B * length / (avg or 1))
                    s += idf[term] * tf * (BM25_K1 + 1) / norm
            out.append(s)
        return out

    def search(self, question: str, *, conversation_id: str, top_k: int) -> list[int]:
        scored = self.scores(question, conversation_id)
        ranked = sorted((i for i, s in enumerate(scored) if s > 0), key=lambda i: (-scored[i], i))
        return sorted(ranked[:top_k])

    def __call__(self, question: str, *, conversation_id: str, top_k: int) -> list[str]:
        chunks = self._chunks[conversation_id]
        return [chunks[i] for i in self.search(question, conversation_id=conversation_id,
                                                top_k=top_k)]


class TraceRetrieverUnavailable(RuntimeError):
    """The orchestrator trace surface cannot serve the ``trace`` arm faithfully."""


class TracePairChunkRetriever:
    """``trace`` arm: the same chunks through the orchestrator trace store + navigation."""

    name = TRACE_RETRIEVER

    def __init__(self, chunks_by_conversation: Mapping[str, Sequence[str]], *,
                 store_id: str = "", db_dir: Optional[Path | str] = None,
                 orchestrator_root: Optional[Path | str] = None):
        from tulving_trace_retriever import DB_DIR_ENV, DEFAULT_DB_DIR, _load_trace_modules
        from tulving_trace_retriever import TraceRetrieverUnavailable as _TulvingUnavailable

        try:
            self._navigation, store = _load_trace_modules(
                Path(orchestrator_root) if orchestrator_root else None)
        except _TulvingUnavailable as exc:
            raise TraceRetrieverUnavailable(str(exc)) from exc
        self._chunks = {cid: list(chunks) for cid, chunks in chunks_by_conversation.items()}
        digest = hashlib.sha256(json.dumps(self._chunks, sort_keys=True).encode()).hexdigest()[:16]
        self.store_id = store_id or digest
        base = Path(db_dir or os.environ.get(DB_DIR_ENV) or DEFAULT_DB_DIR)
        base.mkdir(parents=True, exist_ok=True)
        self._tmpdir = tempfile.mkdtemp(prefix=f"beam-{digest}-", dir=str(base))
        self._cleanup = weakref.finalize(self, shutil.rmtree, self._tmpdir, True)
        self.db_path = Path(self._tmpdir) / "events.sqlite"
        conn = store.ensure_schema(self.db_path)
        try:
            events = [
                store.Event(
                    ts_utc=f"2000-01-01T00:00:00.{index:06d}+00:00",
                    source=TRACE_SOURCE,
                    source_path=f"beam://{self.store_id}/{cid}",
                    source_line=index,
                    session_id=str(cid),
                    category="docs",
                    summary=f"conversation {cid} pair {index}",
                    detail_json=json.dumps({"conversation": cid, "pair": index, "text": text}),
                )
                for cid, chunks in self._chunks.items()
                for index, text in enumerate(chunks)
            ]
            inserted, skipped = store.upsert_events(conn, events)
        finally:
            conn.close()
        expected = sum(len(c) for c in self._chunks.values())
        if inserted != expected or skipped:
            raise TraceRetrieverUnavailable(
                f"indexed {inserted}/{expected} pair chunks ({skipped} duplicates)")

    def search(self, question: str, *, conversation_id: str, top_k: int) -> list[int]:
        from tulving_trace_retriever import fts_query

        query = fts_query(question)
        if not query or top_k <= 0:
            return []
        try:
            rows = self._navigation.search_records(
                query, db_path=self.db_path, limit=int(top_k), order="relevance",
                source=TRACE_SOURCE, session_id=str(conversation_id))
        except TypeError as exc:
            raise TraceRetrieverUnavailable(
                "navigation.search_records has no order= parameter; this orchestrator "
                f"predates bm25 ranking (6302b381) and would rank by recency ({exc})") from exc
        return sorted(int(row["source_line"]) for row in rows)

    def __call__(self, question: str, *, conversation_id: str, top_k: int) -> list[str]:
        chunks = self._chunks[conversation_id]
        return [chunks[i] for i in self.search(question, conversation_id=conversation_id,
                                                top_k=top_k)]


__all__ = [
    "CHUNKING", "BM25_RETRIEVER", "TRACE_RETRIEVER", "TRACE_SOURCE",
    "render_message", "pair_chunks", "BM25PairChunkRetriever",
    "TracePairChunkRetriever", "TraceRetrieverUnavailable",
]
