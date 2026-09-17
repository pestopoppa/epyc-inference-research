#!/usr/bin/env python3
"""CME-4: the default ``retrieved`` arm for the Tulving adapter, using the trace FTS5 surface.

The M-12a ``retrieved`` arm has to go through the memory surface production actually
has: the epyc-orchestrator unified trace store (``src/trace/store.py``, SQLite + FTS5)
and its read-only navigation tools (``src/trace/navigation.py``). This module indexes
one Tulving book, one trace ``Event`` per chapter, into a **private** store built with
the orchestrator's own ``ensure_schema``. It then answers each question through
``navigation.search_records``.

Offline by construction. It opens no socket, loads no model, calls no embedding
service, and never touches the production trace DB (``data/trace/events.sqlite``).
The store is a throwaway file under ``db_dir``.

Ranking. ``navigation.search_records`` delegates to ``src.trace.query.query``. That
function's docstring says "rank by bm25", but its SQL orders every result by
``ts_utc DESC`` (checked 2026-09-16). A ``limit`` below the match count would
therefore keep the LATEST chapters, not the most relevant ones. So this retriever
asks navigation for EVERY matching chapter (``limit`` = chapter count), which is the
candidate set, and orders those candidates by FTS5 ``bm25()`` over the same store. The
top-k passages are returned in book order, because the task is episodic and chapter
order is information the model needs for the chronological questions.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import weakref
from pathlib import Path
from typing import Callable, Optional

from tulving_episodic_adapter import split_book_chapters

ORCHESTRATOR_ROOT_ENV = "EPYC_ORCHESTRATOR_ROOT"
DEFAULT_ORCHESTRATOR_ROOT = Path("/mnt/raid0/llm/epyc-orchestrator")
DB_DIR_ENV = "TULVING_TRACE_DB_DIR"
DEFAULT_DB_DIR = Path("/mnt/raid0/llm/tmp/tulving-trace")
TRACE_SOURCE = "tulving_book"

#: Words that match every chapter or carry no content, so they only add noise to an OR query.
_STOPWORDS = frozenset("""
a an and are as at be been by did do does for from had has have he her his how i in into is
it its list me of on or she that the their them they this to was were what when where which
who whom why will with you your all any chapter chapters book story event events times
places mentioned describe provide
""".split())
_TOKEN = re.compile(r"[A-Za-z0-9]+")

#: ``retriever(question, *, cue, top_k) -> list[str]`` — the contract the adapter calls.
Retriever = Callable[..., list]


class TraceRetrieverUnavailable(RuntimeError):
    """The orchestrator trace surface cannot be imported. The retrieved arm must not run."""


def _load_trace_modules(orchestrator_root: Optional[Path] = None):
    root = Path(orchestrator_root or os.environ.get(ORCHESTRATOR_ROOT_ENV)
                or DEFAULT_ORCHESTRATOR_ROOT).resolve()
    if not (root / "src" / "trace" / "navigation.py").is_file():
        raise TraceRetrieverUnavailable(
            f"orchestrator trace surface not found under {root} (set {ORCHESTRATOR_ROOT_ENV})")
    loaded = sys.modules.get("src")
    if loaded is not None:
        loaded_from = Path(getattr(loaded, "__path__", [""])[0] or "").resolve()
        if loaded_from != root / "src":
            raise TraceRetrieverUnavailable(
                f"a different 'src' package is already imported from {loaded_from}; refusing "
                "to mix it with the orchestrator trace surface")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from src.trace import navigation, store  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise TraceRetrieverUnavailable(f"cannot import src.trace from {root}: {exc}") from exc
    return navigation, store


def fts_query(text: str) -> str:
    """An FTS5 OR query over the content words of ``text``. Each term is quoted, so no
    FTS5 operator syntax can leak in from the question."""
    seen: list[str] = []
    for token in _TOKEN.findall(text or ""):
        low = token.lower()
        if len(low) < 3 or low in _STOPWORDS or low in seen:
            continue
        seen.append(low)
    return " OR ".join(f'"{t}"' for t in seen)


class TraceFTSChapterRetriever:
    """Chapter-level lexical retrieval over one book through the trace FTS5 surface."""

    def __init__(self, book_text: str, *, book_id: str = "",
                 db_dir: Optional[Path | str] = None,
                 orchestrator_root: Optional[Path | str] = None):
        self.chapters = split_book_chapters(book_text)
        self._navigation, store = _load_trace_modules(
            Path(orchestrator_root) if orchestrator_root else None)
        digest = hashlib.sha256(book_text.encode("utf-8")).hexdigest()[:16]
        self.book_id = book_id or digest
        base = Path(db_dir or os.environ.get(DB_DIR_ENV) or DEFAULT_DB_DIR)
        base.mkdir(parents=True, exist_ok=True)
        self._tmpdir = tempfile.mkdtemp(prefix=f"tulving-{digest}-", dir=str(base))
        # The store is throwaway: remove it when this retriever is collected or at exit.
        self._cleanup = weakref.finalize(self, shutil.rmtree, self._tmpdir, True)
        self.db_path = Path(self._tmpdir) / "events.sqlite"
        conn = store.ensure_schema(self.db_path)
        try:
            events = [
                store.Event(
                    ts_utc=f"2000-01-01T00:00:00.{number:06d}+00:00",
                    source=TRACE_SOURCE,
                    source_path=f"tulving://{self.book_id}",
                    source_line=number,
                    category="docs",
                    summary=f"Chapter {number}",
                    detail_json=json.dumps({"chapter": number, "text": text}),
                )
                for number, text in self.chapters
            ]
            inserted, skipped = store.upsert_events(conn, events)
        finally:
            conn.close()
        if inserted != len(self.chapters) or skipped:
            raise TraceRetrieverUnavailable(
                f"indexed {inserted}/{len(self.chapters)} chapters ({skipped} duplicates): "
                "duplicate chapter numbers in the book")
        self._text_by_chapter = dict(self.chapters)

    def search(self, question: str, *, cue: str = "", top_k: int = 5) -> list[tuple[int, float]]:
        """``(chapter, bm25)`` for the best ``top_k`` matching chapters, best first."""
        query = fts_query(f"{question} {cue}")
        if not query or top_k <= 0:
            return []
        candidates = self._navigation.search_records(
            query, db_path=self.db_path, limit=len(self.chapters), source=TRACE_SOURCE)
        ids = [int(row["id"]) for row in candidates]
        if not ids:
            return []
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        try:
            marks = ",".join("?" for _ in ids)
            scored = conn.execute(
                "SELECT e.source_line, bm25(event_fts) AS score FROM event_fts "
                "JOIN event e ON e.id = event_fts.rowid "
                f"WHERE event_fts MATCH ? AND event_fts.rowid IN ({marks}) "
                "ORDER BY score, e.source_line LIMIT ?",
                [query, *ids, int(top_k)],
            ).fetchall()
        finally:
            conn.close()
        return [(int(chapter), float(score)) for chapter, score in scored]

    def __call__(self, question: str, *, cue: str = "", top_k: int = 5) -> list[str]:
        hits = sorted(chapter for chapter, _ in self.search(question, cue=cue, top_k=top_k))
        return [self._text_by_chapter[chapter] for chapter in hits]


__all__ = [
    "ORCHESTRATOR_ROOT_ENV", "DB_DIR_ENV", "TRACE_SOURCE", "Retriever",
    "TraceRetrieverUnavailable", "TraceFTSChapterRetriever", "fts_query",
]
