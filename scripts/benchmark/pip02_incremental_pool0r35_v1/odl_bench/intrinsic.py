"""Ekimetrics intrinsic chunk-quality metrics wired for odl_bench (ODL-011).

Phase-3 contract (handoffs/active/opendataloader-pipeline-integration.md :539-540):
compare the Ekimetrics intrinsic suite and HOPE side by side on our chunker
output, EXCLUDE unverified RC/FMRE, and do not gate on intrinsic scores alone.
This module is the "instrumented harness" half of that contract: the Ekimetrics
MIT scaffold's four non-coref metrics, lifted so any chunker's output can be
scored next to the bench's NID/TEDS/MHS rows.

Lifted from ``ekimetrics/adaptive-chunking`` (MIT; de Moura Junior / Lelong /
Blangero, LREC 2026, https://github.com/ekimetrics/adaptive-chunking, main @
2026-08-13). Four metrics are implemented faithfully:

* ``compute_size_compliance``   (SC)  — fraction of chunks within token bounds
* ``compute_block_integrity``   (BI)  — fraction of gold blocks not cut in half
* ``compute_intrachunk_cohesion`` (ICC) — mean sentence-vs-chunk embedding sim
* ``compute_contextual_coherence`` (DCC) — chunk-vs-context-window embedding sim

EXCLUDED BY CONTRACT — Filtered Missing Reference Error (FMRE / "RC"):
the metric needs ``maverick-coref`` (CC BY-NC-SA 4.0, non-commercial), and the
upstream RC=99.0 figure is unverified: ``str.find()`` returned -1 while the
code caught ``ValueError``, silently corrupting reference boundaries until
upstream ``649ece1`` (2026-07-06). Both reasons are load-bearing for Phase 2/3
quality design and are recorded in the handoff under 2026-07-25 intake dive
corrections. No coref code, no ``maverick`` import, and no license exposure is
introduced here.

Scaffold deltas (documented, all deliberate):

* ``count_tokens`` defaults to a deterministic whitespace approximation of the
  upstream ``tiktoken gpt-4o`` counter so the harness runs on every EPYC venv
  with zero new dependencies. Pass ``count_tokens_func`` to reproduce upstream
  numbers exactly. Relative comparisons are unaffected by the approximation.
* ICC/DCC need an embedding model. Following the backend convention in
  ``backends.py`` (never crash on absence; record the reason), the embedder is
  injected and defaults to None; when absent the metric rows carry
  ``value=None`` with the availability reason in ``detail``.
* ``detect_block_boundaries`` supplies BI's gold split points from the
  prediction markdown's own structure (headings / blank-line paragraphs) as a
  stand-in until the ODL-structure-based block source lands in Phase 2.
* ``DefaultChunker`` is a deterministic heading/paragraph-aware splitter that
  mirrors the Phase-2 heading-driven chunker direction, so the harness is
  runnable today with no model; the real Phase-2 chunker slots in by passing
  its chunk list directly to ``score_chunks``.

All pure metrics are stdlib-only; ICC/DCC use numpy (present on all three EPYC
venvs) only when an embedder is provided.
"""

from __future__ import annotations

import re
from typing import Callable, Protocol

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------
def count_tokens(text: str) -> int:
    """Deterministic whitespace token approximation (no tiktoken dependency).

    Upstream uses ``tiktoken gpt-4o``; tiktoken is not installed on any EPYC
    venv at wiring time. Whitespace splitting is deterministic across
    interpreters, which is what the harness needs; absolute token counts are
    approximate (typically lower than BPE counts). Pass an explicit
    ``count_tokens_func`` for tiktoken-exact numbers.
    """
    return len(text.split())


# ---------------------------------------------------------------------------
# Chunk location helper (lifted from upstream postprocessing.py)
# ---------------------------------------------------------------------------
def find_chunks_start_and_end(chunks: list[str], text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` char offsets of each chunk within ``text``.

    Handles overlaps: searches backward from the previous chunk's end first,
    then falls forward. Raises ``ValueError`` if a chunk is absent from text.
    """
    if not chunks:
        return []

    end_of_previous_chunk = 0
    starts_and_ends = []
    for chunk in chunks:
        # backward search inside a window around the previous chunk's end
        search_start_at = max(0, end_of_previous_chunk - len(chunk))
        search_stop_at = end_of_previous_chunk + len(chunk)
        current_start_index = text.rfind(chunk, search_start_at, search_stop_at)
        # if not found, fall back to a forward search over the rest of the text
        if current_start_index == -1:
            current_start_index = text.find(chunk, end_of_previous_chunk, len(text))
            if current_start_index == -1:
                raise ValueError("Chunk not found in text.")
        starts_and_ends.append((current_start_index, current_start_index + len(chunk)))
        end_of_previous_chunk = current_start_index + len(chunk)
    return starts_and_ends


# ---------------------------------------------------------------------------
# SC — Size Compliance (lifted from upstream metrics.py)
# ---------------------------------------------------------------------------
def compute_size_compliance(
    chunks: list[str],
    max_tokens: int = 1100,
    min_tokens: int = 100,
    count_tokens_func: Callable[[str], int] = count_tokens,
) -> float | None:
    """Fraction of chunks within ``[min_tokens, max_tokens]`` token bounds.

    None when there are no chunks. Higher is better (1.0 = every chunk in
    bounds).
    """
    if not chunks:
        return None
    out_of_span = 0
    for chunk in chunks:
        chunk_len = count_tokens_func(chunk)
        if chunk_len > max_tokens or chunk_len < min_tokens:
            out_of_span += 1
    return 1 - out_of_span / len(chunks)


# ---------------------------------------------------------------------------
# BI — Block Integrity (lifted from upstream metrics.py)
# ---------------------------------------------------------------------------
def compute_block_integrity(
    chunks: list[str],
    doc_split_points: list[int],
    full_text: str,
    tolerance_chars: int = 5,
) -> float | None:
    """Fraction of gold-standard blocks that the chunking did not cut in half.

    A gold block is intact if no predicted split lies strictly inside it,
    allowing ``tolerance_chars`` leeway at both ends. None when there are no
    chunks; 1.0 for a single chunk (nothing can be cut).
    """
    if not chunks:
        return None
    if len(chunks) == 1:
        return 1.0

    starts_and_ends = find_chunks_start_and_end(chunks=chunks, text=full_text)
    starts = [start for start, _end in starts_and_ends]
    predicted_split_points = sorted({s for s in starts[1:] if s is not None})

    gold_sorted = sorted(doc_split_points)
    doc_len = len(full_text)
    block_bounds = [0] + gold_sorted + [doc_len]

    intact = 0
    for left, right in zip(block_bounds, block_bounds[1:]):
        block_broken = any(
            (left < p < right)
            and (p - left) > tolerance_chars
            and (right - p) > tolerance_chars
            for p in predicted_split_points
        )
        if not block_broken:
            intact += 1

    total = len(block_bounds) - 1
    return intact / total if total else None


# ---------------------------------------------------------------------------
# Block boundary detection (harness stand-in for gold blocks)
# ---------------------------------------------------------------------------
_HEADING_RE = re.compile(r"^#{1,6}\s+\S.*$", re.MULTILINE)


def detect_block_boundaries(text: str) -> list[int]:
    """Return char offsets of structural block starts in extracted markdown.

    Stand-in for the ODL-structure-based gold blocks that land in Phase 2.
    Blocks are: markdown headings (``^#{1,6} ``) and blank-line-separated
    paragraphs. A block's start is the offset of its first non-whitespace char.
    """
    if not text:
        return []
    boundaries: list[int] = []
    for m in _HEADING_RE.finditer(text):
        boundaries.append(m.start())
    # paragraph boundaries: a line whose previous line is blank
    lines = [(m.start(), m.group(0)) for m in re.finditer(r"[^\n]*\n?", text)]
    prev_blank = True  # start-of-text is a block boundary
    for start, content in lines:
        is_blank = content.strip() == ""
        if not is_blank:
            if prev_blank:
                boundaries.append(start)
        prev_blank = is_blank
    return sorted({b for b in boundaries if b > 0})


# ---------------------------------------------------------------------------
# Embedder protocol + availability (mirrors backends.available() convention)
# ---------------------------------------------------------------------------
class Embedder(Protocol):
    """Minimal sentence-transformers-compatible embedding surface."""

    def encode(
        self,
        texts: list[str],
        batch_size: int = 16,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
    ) -> "object":  # np.ndarray at runtime
        ...


def resolve_embedder() -> tuple[Embedder | None, str]:
    """Return a cached sentence-transformers embedder, or (None, reason).

    Returns ``(None, reason)`` without raising when the model library is
    absent, so the intrinsic harness degrades to SC/BI rows exactly like a
    missing backend degrades to ``available=False``. Model download is never
    attempted implicitly.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:  # pragma: no cover - env dependent
        return None, f"sentence-transformers not importable: {type(exc).__name__}: {exc}"
    try:
        return SentenceTransformer("all-MiniLM-L6-v2"), ""
    except Exception as exc:  # pragma: no cover - env dependent
        return None, f"SentenceTransformer load failed: {type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# ICC — Intrachunk Cohesion (lifted from upstream metrics.py)
# ---------------------------------------------------------------------------
def _normalize_chunk_embeddings(chunk_embeddings: "object") -> "object":
    """Guard: downstream math assumes unit vectors. Normalize in place-safe way."""
    import numpy as np

    arr = np.asarray(chunk_embeddings)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def compute_intrachunk_cohesion(
    chunks: list[str],
    full_text: str,
    split_points: list[int],
    model: Embedder,
    chunk_embeddings: "object | None" = None,
    batch_size: int = 16,
    progress_bar: bool = False,
) -> float | None:
    """Mean cosine similarity between a chunk's sentences and its embedding.

    Sentence boundaries are provided in ``split_points`` as char offsets
    relative to ``full_text``. Returns None for degenerate inputs (no
    multi-sentence chunks, no chunks, or chunk/embedding length mismatch).
    """
    import numpy as np

    if not chunks:
        return None
    if chunk_embeddings is None:
        chunk_embeddings = _embed_chunks(chunks, model, batch_size, progress_bar)
    else:
        chunk_embeddings = _normalize_chunk_embeddings(chunk_embeddings)
    if len(chunk_embeddings) != len(chunks):
        raise ValueError("chunk_embeddings length must equal number of chunks.")

    sorted_splits = sorted(split_points)
    chunk_starts_and_ends = find_chunks_start_and_end(chunks=chunks, text=full_text)

    chunk_sentences: list[list[str]] = []
    for chunk, (chunk_start, chunk_end) in zip(chunks, chunk_starts_and_ends):
        local_split_points = [
            split_point - chunk_start
            for split_point in sorted_splits
            if chunk_start <= split_point < chunk_end
        ]
        boundaries = sorted({0, *local_split_points, chunk_end})
        sentences: list[str] = []
        for i in range(len(boundaries) - 1):
            sentence = chunk[boundaries[i]:boundaries[i + 1]]
            if sentence:
                sentences.append(sentence)
        if not sentences:
            sentences = [chunk]
        chunk_sentences.append(sentences)

    flattened: list[str] = [s for sent_list in chunk_sentences for s in sent_list]
    if not flattened:
        return None
    sentence_embeddings = np.asarray(
        model.encode(
            flattened,
            batch_size=batch_size,
            show_progress_bar=progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    )

    cohesion_scores: list[float] = []
    sent_idx = 0
    for idx, sentences in enumerate(chunk_sentences):
        num_sents = len(sentences)
        if num_sents < 2:
            sent_idx += num_sents
            continue
        sent_embeds = sentence_embeddings[sent_idx: sent_idx + num_sents]
        sent_idx += num_sents
        sims = np.dot(sent_embeds, chunk_embeddings[idx])
        cohesion_scores.append(np.mean(sims))

    if not cohesion_scores:
        return None
    return float(np.clip(np.mean(cohesion_scores), 0.0, 1.0))


# ---------------------------------------------------------------------------
# DCC — Contextual Coherence (lifted from upstream metrics.py)
# ---------------------------------------------------------------------------
def compute_contextual_coherence(
    chunks: list[str],
    full_text: str,
    model: Embedder,
    window_context_tokens: int = 3000,
    count_tokens_func: Callable[[str], int] = count_tokens,
    window_step: int = 1,
    batch_size: int = 16,
    chunk_embeddings: "object | None" = None,
    progress_bar: bool = False,
) -> float | None:
    """Mean cosine similarity of each chunk to its sliding context window.

    Windows are built so the sum of non-overlapping token counts stays below
    ``window_context_tokens``, sliding forward ``window_step`` chunks at a
    time. Returns None with fewer than 2 chunks or no windows.
    """
    import numpy as np

    n = len(chunks)
    if n < 2:
        return None
    if chunk_embeddings is None:
        chunk_embeddings = _embed_chunks(chunks, model, batch_size, progress_bar)
    else:
        chunk_embeddings = _normalize_chunk_embeddings(chunk_embeddings)
    if len(chunk_embeddings) != len(chunks):
        raise ValueError("chunk_embeddings length must equal number of chunks.")

    chunk_bounds = find_chunks_start_and_end(chunks=chunks, text=full_text)

    text_windows: list[str] = []
    chunks_indices_per_window: list[list[int]] = []

    i = 0
    while i < n:
        current_end = chunk_bounds[i][0]
        window_token_count = 0
        window_parts: list[str] = []
        chunks_in_window: list[int] = []

        j = i
        while j < n:
            start_j, end_j = chunk_bounds[j]
            slice_start = max(current_end, start_j)
            slice_end = end_j
            if slice_start < slice_end:
                tail_tokens = count_tokens_func(full_text[slice_start:slice_end])
            else:
                tail_tokens = 0
            if chunks_in_window and window_token_count + tail_tokens > window_context_tokens:
                break
            if tail_tokens:
                window_parts.append(full_text[slice_start:slice_end])
                window_token_count += tail_tokens
                current_end = slice_end
            chunks_in_window.append(j)
            j += 1

        if len(chunks_indices_per_window) > 0:
            if chunks_in_window[-1] != chunks_indices_per_window[-1][-1] and len(chunks_in_window) > 1:
                text_windows.append("".join(window_parts))
                chunks_indices_per_window.append(chunks_in_window)
        elif len(chunks_in_window) > 1:
            text_windows.append("".join(window_parts))
            chunks_indices_per_window.append(chunks_in_window)

        i += window_step

    if not text_windows:
        return None

    window_embeddings = np.asarray(
        model.encode(
            text_windows,
            batch_size=batch_size,
            show_progress_bar=progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    )

    cohesion_scores: list[float] = []
    for window_idx, window_chunks in enumerate(chunks_indices_per_window):
        window_embed = window_embeddings[window_idx]
        for chunk_idx in window_chunks:
            sim = np.dot(window_embed, chunk_embeddings[chunk_idx])
            cohesion_scores.append(sim)

    if not cohesion_scores:
        return None
    return float(np.clip(np.mean(cohesion_scores), 0.0, 1.0))


def _embed_chunks(chunks: list[str], model: Embedder, batch_size: int, progress_bar: bool) -> "object":
    """Embed each chunk once (shared by ICC and DCC)."""
    import numpy as np

    embeddings = np.asarray(
        model.encode(
            chunks,
            batch_size=batch_size,
            show_progress_bar=progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    )
    return _normalize_chunk_embeddings(embeddings)


# ---------------------------------------------------------------------------
# Default deterministic chunker (Phase-2 heading-driven direction)
# ---------------------------------------------------------------------------
class DefaultChunker:
    """Deterministic heading/paragraph-aware splitter (no model, no regex LLM).

    Mirrors the Phase-2 heading-driven chunker direction: chunks start at
    markdown headings or blank-line paragraph boundaries, then oversized chunks
    are split on the next available separator (newline -> space -> char) using
    the same recursive strategy as upstream ``RecursiveSplitter``. The output
    is fully deterministic for a given text + settings.

    Phase-2's real chunker (heading hierarchy from ODL JSON) replaces this by
    passing its chunk list straight to ``score_chunks``; this default exists so
    the intrinsic harness is runnable today.
    """

    def __init__(
        self,
        chunk_size: int = 600,
        min_chunk_tokens: int = 50,
        count_tokens_func: Callable[[str], int] = count_tokens,
    ):
        self.chunk_size = chunk_size
        self.min_chunk_tokens = min_chunk_tokens
        self.length_function = count_tokens_func

    def split_text(self, text: str) -> list[str]:
        if not text:
            return []
        boundary_offsets = [0] + detect_block_boundaries(text) + [len(text)]
        boundary_offsets = sorted(set(boundary_offsets))
        blocks = [
            text[boundary_offsets[k]:boundary_offsets[k + 1]]
            for k in range(len(boundary_offsets) - 1)
            if text[boundary_offsets[k]:boundary_offsets[k + 1]].strip()
        ]
        chunks: list[str] = []
        for block in blocks:
            if self.length_function(block) <= self.chunk_size:
                chunks.append(block)
            else:
                chunks.extend(self._recursive_split(block, ["\n", " ", ""]))
        chunks = self._merge_small_chunks(chunks)
        # Drop leading/trailing whitespace-only chunks: they sit at the text
        # edges, so removing them cannot break the interior contiguity that
        # find_chunks_start_and_end relies on (interior whitespace chunks
        # were already merged into a neighbour above).
        while chunks and not chunks[0].strip():
            chunks.pop(0)
        while chunks and not chunks[-1].strip():
            chunks.pop()
        return chunks

    def _merge_small_chunks(self, chunks: list[str]) -> list[str]:
        """Greedily merge adjacent chunks whose combined size fits the target.

        Mirrors upstream ``merge_small_chunks_to_neighbours``: a chunk below
        ``min_chunk_tokens`` merges into its neighbour when the union stays at
        or under ``chunk_size``. Deterministic (forward pass, merge-right).
        """
        if len(chunks) < 2:
            return chunks
        merged = chunks[:]
        i = 0
        while i < len(merged) - 1:
            size_i = self.length_function(merged[i])
            size_next = self.length_function(merged[i + 1])
            if (
                size_i < self.min_chunk_tokens
                and size_i + size_next <= self.chunk_size
            ):
                merged[i] = merged[i] + merged[i + 1]
                del merged[i + 1]
                continue  # re-examine i against its new neighbour
            i += 1
        return merged

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if not text:
            return []
        if self.length_function(text) <= self.chunk_size:
            return [text]
        separator = separators[0]
        remaining = separators[1:]
        if separator == "":
            return self._hard_split(text)
        pieces = self._split_keep_separator(text, separator)
        final: list[str] = []
        for piece in pieces:
            if self.length_function(piece) > self.chunk_size:
                final.extend(self._recursive_split(piece, remaining if remaining else [""]))
            else:
                final.append(piece)
        return final

    @staticmethod
    def _split_keep_separator(text: str, separator: str) -> list[str]:
        """Split ``text`` on ``separator`` keeping the separator attached to
        the FOLLOWING piece, so every returned piece is a contiguous slice of
        ``text`` (concatenating the pieces reproduces ``text`` exactly).

        ``re.split`` would drop the separators, turning adjacent pieces into
        non-contiguous fragments; later merges of those fragments would no
        longer be substrings of ``text``, and ``find_chunks_start_and_end``
        (which locates each chunk inside the full text by exact match) would
        raise ``ValueError`` on real documents. Keeping separators preserves
        the substring invariant that BI/ICC/DCC depend on.
        """
        if not separator:
            return [text] if text else []
        pieces: list[str] = []
        cursor = 0
        for match in re.finditer(re.escape(separator), text):
            pieces.append(text[cursor:match.start()])
            cursor = match.start()
        pieces.append(text[cursor:])
        return [piece for piece in pieces if piece]

    def _hard_split(self, text: str) -> list[str]:
        chunks: list[str] = []
        remaining = text
        while remaining:
            low, high = 0, len(remaining)
            best_end = 0
            while low <= high:
                mid = (low + high) // 2
                if self.length_function(remaining[:mid]) <= self.chunk_size:
                    best_end = mid
                    low = mid + 1
                else:
                    high = mid - 1
            if best_end == 0:
                best_end = 1
            chunks.append(remaining[:best_end])
            remaining = remaining[best_end:]
        return chunks


# ---------------------------------------------------------------------------
# Row-producing scoring entry points (the odl_bench surface)
# ---------------------------------------------------------------------------
def score_chunks(
    chunks: list[str],
    full_text: str,
    *,
    engine: str,
    doc_split_points: list[int] | None = None,
    split_points: list[int] | None = None,
    embedder: Embedder | None = None,
    embedder_reason: str = "",
    min_tokens: int = 100,
    max_tokens: int = 1100,
) -> list["MetricRow"]:  # noqa: F821  (imported lazily to avoid a cycle)
    """Score one chunked document -> intrinsic MetricRows (SC/BI/ICC/DCC).

    Contract note (Phase-2 :539): these scores are informational next to
    NID/TEDS/MHS, never a gate on their own, and FMRE/RC is excluded. When
    ``embedder`` is None, ICC/DCC rows are emitted with ``value=None`` and the
    availability reason in ``detail`` — the same degrade convention as a
    missing extraction backend.
    """
    from .schemas import METRIC_INTRINSIC, MetricRow

    rows: list[MetricRow] = []

    sc = compute_size_compliance(chunks, max_tokens=max_tokens, min_tokens=min_tokens)
    rows.append(MetricRow(
        engine=engine,
        metric_family=METRIC_INTRINSIC,
        metric_name="Ekimetrics.SC",
        value=sc,
        n=len(chunks),
        detail=(
            "size compliance: fraction of chunks within "
            f"[{min_tokens},{max_tokens}] whitespace tokens; HIGHER is better"
        ),
    ))

    bi = compute_block_integrity(
        chunks,
        doc_split_points or detect_block_boundaries(full_text),
        full_text,
    )
    rows.append(MetricRow(
        engine=engine,
        metric_family=METRIC_INTRINSIC,
        metric_name="Ekimetrics.BI",
        value=bi,
        n=len(chunks),
        detail=(
            "block integrity: fraction of structural blocks (headings/paragraphs) "
            "not cut; HIGHER is better"
        ),
    ))

    if embedder is None:
        reason = embedder_reason or "no embedder provided (install sentence-transformers)"
        rows.append(MetricRow(
            engine=engine,
            metric_family=METRIC_INTRINSIC,
            metric_name="Ekimetrics.ICC",
            value=None,
            n=len(chunks),
            detail=f"intrachunk cohesion unavailable: {reason}",
        ))
        rows.append(MetricRow(
            engine=engine,
            metric_family=METRIC_INTRINSIC,
            metric_name="Ekimetrics.DCC",
            value=None,
            n=len(chunks),
            detail=f"contextual coherence unavailable: {reason}",
        ))
        return rows

    split_points = split_points or detect_block_boundaries(full_text)
    icc = compute_intrachunk_cohesion(
        chunks, full_text, split_points, model=embedder,
    )
    rows.append(MetricRow(
        engine=engine,
        metric_family=METRIC_INTRINSIC,
        metric_name="Ekimetrics.ICC",
        value=icc,
        n=len(chunks),
        detail="intrachunk cohesion: mean sentence-vs-chunk cosine; HIGHER is better",
    ))
    dcc = compute_contextual_coherence(
        chunks, full_text, model=embedder,
    )
    rows.append(MetricRow(
        engine=engine,
        metric_family=METRIC_INTRINSIC,
        metric_name="Ekimetrics.DCC",
        value=dcc,
        n=len(chunks),
        detail="contextual coherence: chunk-vs-window cosine; HIGHER is better",
    ))
    return rows


def score_prediction_dir(
    prediction_dir: str,
    *,
    engine: str,
    embedder: Embedder | None = None,
    chunker: DefaultChunker | None = None,
    aggregate: str = "mean",
    min_tokens: int = 100,
    max_tokens: int = 1100,
) -> list["MetricRow"]:  # noqa: F821
    """Score every prediction ``*.md`` in a dir, aggregated into engine rows.

    Each prediction file is treated as one document: chunked (default chunker),
    scored, then the per-document scores are aggregated (mean) into a single
    set of intrinsic rows for ``engine``. ``n`` = number of documents scored.
    Missing/empty files are skipped; a dir with no scorable documents yields
    rows with ``value=None``.
    """
    from pathlib import Path

    from .schemas import METRIC_INTRINSIC, MetricRow

    chunker = chunker or DefaultChunker()
    docs: list[str] = []
    for md_path in sorted(Path(prediction_dir).glob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        if text.strip():
            docs.append(text)

    if not docs:
        return [
            MetricRow(
                engine=engine,
                metric_family=METRIC_INTRINSIC,
                metric_name=name,
                value=None,
                n=0,
                detail=f"no non-empty prediction files under {prediction_dir}",
            )
            for name in ("Ekimetrics.SC", "Ekimetrics.BI", "Ekimetrics.ICC", "Ekimetrics.DCC")
        ]

    per_doc: dict[str, list[float]] = {
        "Ekimetrics.SC": [], "Ekimetrics.BI": [], "Ekimetrics.ICC": [], "Ekimetrics.DCC": []
    }
    unavailable_detail: dict[str, str] = {}
    skipped_docs = 0
    skip_errors: list[str] = []
    for text in docs:
        try:
            chunks = chunker.split_text(text)
            rows = score_chunks(
                chunks,
                text,
                engine=engine,
                embedder=embedder,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - a pathological doc never aborts the run
            skipped_docs += 1
            skip_errors.append(f"{type(exc).__name__}: {exc}")
            continue
        for row in rows:
            if row.value is not None:
                per_doc[row.metric_name].append(float(row.value))
            elif row.metric_name not in unavailable_detail:
                unavailable_detail[row.metric_name] = row.detail

    if skipped_docs:
        sample = skip_errors[0][:120]
        more = f" (+{len(skip_errors) - 1} more)" if len(skip_errors) > 1 else ""
        skip_note = (
            f"{len(docs)} prediction documents, {skipped_docs} skipped "
            f"(document-level scoring error: {sample}{more})"
        )
        for name in ("Ekimetrics.SC", "Ekimetrics.BI", "Ekimetrics.ICC", "Ekimetrics.DCC"):
            if name not in unavailable_detail:
                unavailable_detail[name] = skip_note

    rows: list[MetricRow] = []
    for name in ("Ekimetrics.SC", "Ekimetrics.BI", "Ekimetrics.ICC", "Ekimetrics.DCC"):
        values = per_doc[name]
        value: float | None
        if values:
            value = sum(values) / len(values) if aggregate == "mean" else min(values)
            detail = f"{aggregate} over {len(docs)} prediction documents"
        else:
            value = None
            detail = (
                unavailable_detail.get(name)
                or f"not computable over {len(docs)} prediction documents"
            )
        rows.append(MetricRow(
            engine=engine,
            metric_family=METRIC_INTRINSIC,
            metric_name=name,
            value=value,
            n=len(docs),
            detail=detail,
        ))
    return rows
