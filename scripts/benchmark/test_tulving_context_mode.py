#!/usr/bin/env python3
"""CME-4 (the ``context_mode`` arm axis) and M-12e-a (a loud dataframe load) for the Tulving adapter.

The things pinned here:

* ``full`` produces byte-for-byte the prompt the adapter always built, so the stored run
  ``20260619_141212`` stays reproducible;
* ``none`` carries no book text; ``retrieved`` carries only the retriever's passages, and it
  raises rather than degrading to another arm when no retriever exists;
* ground truth and question ids are identical across arms, and the arm is readable back
  from the prompt header (the scorer's ``--arm`` cross-check depends on this);
* the default retriever goes through the orchestrator trace FTS5 surface into a private
  store, and ranks by bm25 rather than the navigation layer's ts-desc order;
* a missing pandas/pyarrow, or an unreadable parquet, raises instead of returning an empty
  dataset that would score as all-missing.

Run with:
    pytest scripts/benchmark/test_tulving_context_mode.py -q
"""

import builtins
import gc
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import tulving_episodic_adapter as tea  # noqa: E402
from tulving_episodic_adapter import (  # noqa: E402
    CONTEXT_FULL,
    CONTEXT_NONE,
    CONTEXT_RETRIEVED,
    TulvingEpisodicAdapter,
    context_mode_of_prompt,
    split_book_chapters,
)

BOOK = (
    "Chapter 1\n\nAt the Harbor Market, Ezra Edwards juggled oranges on March 3, 2025.\n\n"
    "Chapter 2\n\nMila Chen rehearsed a dramatic scene at the Lyric Theatre on May 9, 2025.\n\n"
    "Chapter 3\n\nEzra Edwards returned to the Harbor Market and sold kites on June 1, 2025.\n"
)

ROW = {
    "question": "Where was Ezra Edwards observed?",
    "correct_answer": ["Harbor Market"],
    "retrieval_type": "Spaces",
    "get": "all",
    "cue": "(*, Ezra Edwards, *, *)",
    "chapter": 1,
    "n_chapters_correct_answer": 2,
}


def _adapter(mode, **kwargs):
    adapter = TulvingEpisodicAdapter(context_mode=mode, **kwargs)
    adapter._dataset = [dict(ROW)]
    adapter._book_text = BOOK
    return adapter


def _legacy_prompt(book, question_block):
    """The exact construction used before CME-4."""
    return "Book narrative:\n" f"{book.strip()}\n\n" "---\n\n" f"{question_block}"


class RecordingRetriever:
    def __init__(self, passages):
        self.passages = passages
        self.calls = []

    def __call__(self, question, *, cue, top_k):
        self.calls.append((question, cue, top_k))
        return list(self.passages)


# ── the three arms ────────────────────────────────────────────────────────────

def test_full_is_byte_identical_to_the_legacy_prompt():
    prompt = _adapter(CONTEXT_FULL)._row_to_prompt(0, ROW)
    bare = _adapter(CONTEXT_NONE)._row_to_prompt(0, ROW)["prompt"]
    assert prompt["prompt"] == _legacy_prompt(BOOK, bare)
    assert prompt["context"] == BOOK
    assert prompt["metadata"]["context_mode"] == CONTEXT_FULL


def test_default_mode_is_full(monkeypatch):
    monkeypatch.delenv(tea.CONTEXT_MODE_ENV, raising=False)
    assert TulvingEpisodicAdapter().context_mode == CONTEXT_FULL


def test_none_has_no_book_text():
    prompt = _adapter(CONTEXT_NONE)._row_to_prompt(0, ROW)
    assert "Harbor Market, Ezra" not in prompt["prompt"]
    assert prompt["prompt"].startswith(ROW["question"])
    assert prompt["context"] == ""
    assert prompt["metadata"]["context_mode"] == CONTEXT_NONE


def test_retrieved_uses_only_the_injected_passages():
    retriever = RecordingRetriever(["Chapter 3\n\nEzra sold kites."])
    adapter = _adapter(CONTEXT_RETRIEVED, retriever=retriever, retrieval_top_k=2)
    prompt = adapter._row_to_prompt(0, ROW)
    assert prompt["prompt"].startswith(tea.RETRIEVED_HEADER + "Chapter 3\n\nEzra sold kites.")
    assert "Lyric Theatre" not in prompt["prompt"]
    assert retriever.calls == [(ROW["question"], ROW["cue"], 2)]
    assert prompt["metadata"]["retrieved_passages"] == 1
    assert prompt["context"] == ""


def test_retrieved_with_no_hits_says_so():
    prompt = _adapter(CONTEXT_RETRIEVED, retriever=RecordingRetriever([]))._row_to_prompt(0, ROW)
    assert "(no passages retrieved)" in prompt["prompt"]
    assert context_mode_of_prompt(prompt["prompt"]) == CONTEXT_RETRIEVED
    assert prompt["metadata"]["retrieved_passages"] == 0


def test_retrieved_without_a_retriever_raises_not_degrades():
    adapter = _adapter(CONTEXT_RETRIEVED)
    with pytest.raises(RuntimeError, match="no retriever"):
        adapter._row_to_prompt(0, ROW)


def test_invalid_mode_and_retriever_are_refused():
    with pytest.raises(ValueError, match="context_mode"):
        TulvingEpisodicAdapter(context_mode="half")
    with pytest.raises(TypeError):
        TulvingEpisodicAdapter(context_mode=CONTEXT_RETRIEVED, retriever="not callable")
    with pytest.raises(ValueError):
        TulvingEpisodicAdapter(context_mode=CONTEXT_RETRIEVED, retrieval_top_k=0)


def test_env_selects_the_arm_for_get_adapter(monkeypatch):
    monkeypatch.setenv(tea.CONTEXT_MODE_ENV, CONTEXT_NONE)
    from dataset_adapters import get_adapter
    assert get_adapter("tulving_episodic").context_mode == CONTEXT_NONE
    monkeypatch.setenv(tea.CONTEXT_MODE_ENV, "bogus")
    with pytest.raises(ValueError):
        TulvingEpisodicAdapter()


def test_explicit_argument_beats_env(monkeypatch):
    monkeypatch.setenv(tea.CONTEXT_MODE_ENV, CONTEXT_NONE)
    assert TulvingEpisodicAdapter(context_mode=CONTEXT_FULL).context_mode == CONTEXT_FULL


def test_ground_truth_and_ids_are_identical_across_arms():
    prompts = {
        CONTEXT_NONE: _adapter(CONTEXT_NONE)._row_to_prompt(0, ROW),
        CONTEXT_FULL: _adapter(CONTEXT_FULL)._row_to_prompt(0, ROW),
        CONTEXT_RETRIEVED: _adapter(
            CONTEXT_RETRIEVED, retriever=RecordingRetriever(["x"]))._row_to_prompt(0, ROW),
    }
    ids = {p["id"] for p in prompts.values()}
    assert len(ids) == 1
    strip = lambda p: {k: v for k, v in p["metadata"].items()  # noqa: E731
                       if k not in ("context_mode", "retrieved_passages")}
    assert strip(prompts[CONTEXT_NONE]) == strip(prompts[CONTEXT_FULL]) \
        == strip(prompts[CONTEXT_RETRIEVED])
    assert prompts[CONTEXT_NONE]["expected"] == prompts[CONTEXT_FULL]["expected"]


def test_the_arm_reads_back_from_the_prompt_header():
    for mode, kwargs in ((CONTEXT_NONE, {}), (CONTEXT_FULL, {}),
                         (CONTEXT_RETRIEVED, {"retriever": RecordingRetriever(["p"])})):
        prompt = _adapter(mode, **kwargs)._row_to_prompt(0, ROW)["prompt"]
        assert context_mode_of_prompt(prompt) == mode


# ── chapter splitting ─────────────────────────────────────────────────────────

def test_split_book_chapters():
    chapters = split_book_chapters(BOOK)
    assert [n for n, _ in chapters] == [1, 2, 3]
    assert chapters[1][1].startswith("Chapter 2\n\nMila Chen")
    assert "Chapter 3" not in chapters[1][1]


def test_split_refuses_an_undivided_book():
    with pytest.raises(ValueError):
        split_book_chapters("one long blob with no headings")


# ── the default trace-FTS5 retriever ──────────────────────────────────────────

def _trace_available():
    try:
        import tulving_trace_retriever as ttr
        ttr._load_trace_modules()
        return True
    except Exception:
        return False


trace_only = pytest.mark.skipif(not _trace_available(),
                                reason="epyc-orchestrator src/trace not importable here")


def test_fts_query_quotes_terms_and_drops_noise():
    from tulving_trace_retriever import fts_query
    q = fts_query('Where was "Ezra" Edwards observed? chapter AND OR NEAR(')
    assert q == '"ezra" OR "edwards" OR "observed" OR "near"'
    assert fts_query("the of a") == ""


@trace_only
def test_trace_retriever_finds_the_right_chapters(tmp_path):
    from tulving_trace_retriever import TraceFTSChapterRetriever
    retriever = TraceFTSChapterRetriever(BOOK, db_dir=tmp_path)
    hits = retriever.search("Where was Ezra Edwards observed?", top_k=2)
    assert sorted(chapter for chapter, _ in hits) == [1, 3]
    passages = retriever("Where was Ezra Edwards observed?", top_k=2)
    assert [p.split("\n", 1)[0] for p in passages] == ["Chapter 1", "Chapter 3"]  # book order
    assert retriever("Lyric Theatre rehearsal", top_k=5)[0].startswith("Chapter 2")


@trace_only
def test_trace_retriever_ranks_by_bm25_not_recency(tmp_path):
    """navigation.search_records returns ts-desc; top_k=1 must still pick the best match."""
    from tulving_trace_retriever import TraceFTSChapterRetriever
    book = ("Chapter 1\n\nkites kites kites kites kites at the pier.\n\n"
            "Chapter 2\n\nA quiet day.\n\n"
            "Chapter 3\n\nOne kite, briefly, among many other unrelated words here.\n")
    retriever = TraceFTSChapterRetriever(book, db_dir=tmp_path)
    assert retriever.search("kites", top_k=1)[0][0] == 1


@trace_only
def test_trace_retriever_uses_a_private_store_and_cleans_it(tmp_path):
    from tulving_trace_retriever import TraceFTSChapterRetriever
    retriever = TraceFTSChapterRetriever(BOOK, db_dir=tmp_path)
    store_dir = Path(retriever._tmpdir)
    assert store_dir.parent == tmp_path and retriever.db_path.is_file()
    assert "data/trace/events.sqlite" not in str(retriever.db_path)
    del retriever
    gc.collect()
    assert not store_dir.exists()


@trace_only
def test_adapter_builds_the_default_retriever_on_load(tmp_path, monkeypatch):
    monkeypatch.setenv("TULVING_TRACE_DB_DIR", str(tmp_path / "trace"))
    adapter = TulvingEpisodicAdapter(context_mode=CONTEXT_RETRIEVED, data_dir=tmp_path)
    monkeypatch.setattr(adapter, "_load_qa_from_variant",
                        lambda _d: (setattr(adapter, "_book_text", BOOK), [dict(ROW)])[1])
    (tmp_path / adapter._variant).mkdir()
    adapter._ensure_loaded()
    prompt = adapter._row_to_prompt(0, ROW)["prompt"]
    assert prompt.startswith(tea.RETRIEVED_HEADER + "Chapter 1")
    assert "Lyric Theatre" not in prompt


def test_retrieved_load_without_a_book_raises(tmp_path, monkeypatch):
    adapter = TulvingEpisodicAdapter(context_mode=CONTEXT_RETRIEVED, data_dir=tmp_path)
    monkeypatch.setattr(adapter, "_load_qa_from_variant", lambda _d: [dict(ROW)])
    (tmp_path / adapter._variant).mkdir()
    with pytest.raises(RuntimeError, match="book text"):
        adapter._ensure_loaded()


def test_missing_trace_surface_is_loud(tmp_path):
    import tulving_trace_retriever as ttr
    with pytest.raises(ttr.TraceRetrieverUnavailable):
        ttr.TraceFTSChapterRetriever(BOOK, db_dir=tmp_path, orchestrator_root=tmp_path)


# ── M-12e-a: the dataframe load fails loudly ─────────────────────────────────

def _variant_with_parquet(tmp_path):
    book_dir = (tmp_path / "Udefault_Sdefault_seed0" / "books"
                / "model_claude_itermax_10_Idefault_nbchapters_19_nbtokens_1")
    book_dir.mkdir(parents=True)
    (book_dir / "df_qa.parquet").write_bytes(b"not a parquet file")
    (book_dir / "book.json").write_text(json.dumps(BOOK))
    return tmp_path


def test_missing_pandas_raises_instead_of_scoring_everything_missing(tmp_path, monkeypatch):
    data = _variant_with_parquet(tmp_path)
    real_import = builtins.__import__

    def no_pandas(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("No module named 'pandas'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pandas)
    adapter = TulvingEpisodicAdapter(data_dir=data)
    with pytest.raises(RuntimeError, match="pandas \\+ pyarrow"):
        adapter._ensure_loaded()


def test_unreadable_parquet_raises(tmp_path):
    pytest.importorskip("pandas")
    data = _variant_with_parquet(tmp_path)
    adapter = TulvingEpisodicAdapter(data_dir=data)
    with pytest.raises(RuntimeError, match="could not read QA parquet"):
        adapter._ensure_loaded()


def test_absent_data_dir_is_still_an_empty_dataset():
    """Unchanged: no extracted data at all is reported and yields nothing to run."""
    adapter = TulvingEpisodicAdapter(data_dir="/nonexistent/path")
    adapter._ensure_loaded()
    assert adapter._dataset == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
