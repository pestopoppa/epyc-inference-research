#!/usr/bin/env python3
"""M-12 B2: the BEAM memory arms (naive-RAG control and trace arm), offline.

The trace surface is mocked in most tests. One test runs against the real
orchestrator trace store (a private SQLite file) when that checkout is present.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

import beam_memory_retrievers as bmr
import judge_beam_run
import long_context_adapters as lca
import score_beam_run
from long_context_adapters import (
    BEAM_FULL_HEADER,
    BEAM_NO_EXCERPTS,
    BEAM_RETRIEVED_HEADER,
    BEAMAdapter,
    beam_context_kind_of_prompt,
)
from test_beam_adapter import _judged_payload, write_parquet_fixture

MESSAGES = [
    {"role": "assistant", "content": "welcome", "time_anchor": None},
    {"role": "user", "content": "my sprint ends March 29", "time_anchor": "March-16-2024"},
    {"role": "assistant", "content": "noted the sprint deadline"},
    {"role": "assistant", "content": "anything else?"},
    {"role": "user", "content": "I prefer tabs over spaces", "time_anchor": "March-17-2024"},
    {"role": "assistant", "content": "tabs it is"},
    {"role": "user", "content": "the budget tracker uses Flask", "time_anchor": "March-18-2024"},
    {"role": "assistant", "content": "Flask is a good choice"},
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(lca.BEAM_CONTEXT_MODE_ENV, raising=False)
    monkeypatch.delenv(lca.BEAM_RETRIEVAL_TOP_K_ENV, raising=False)


# ── chunking ──────────────────────────────────────────────────────────────────

def test_pair_chunks_start_at_user_turns_and_rebuild_the_transcript():
    chunks = bmr.pair_chunks(MESSAGES)
    assert len(chunks) == 4                      # leading assistant + 3 user-led pairs
    assert chunks[0] == "Assistant: welcome"
    assert chunks[1].startswith("[March-16-2024] User: my sprint")
    assert chunks[1].endswith("Assistant: anything else?")      # both roles kept (M-12c(2))
    assert "\n\n".join(chunks) == BEAMAdapter._render_transcript(MESSAGES)


# ── the naive control: pair_chunk x BM25 ──────────────────────────────────────

def test_bm25_finds_the_right_pair_and_keeps_conversation_order():
    r = bmr.BM25PairChunkRetriever({"c": bmr.pair_chunks(MESSAGES)})
    assert r.search("Which web framework does the budget tracker use?",
                    conversation_id="c", top_k=1) == [3]
    both = r.search("sprint deadline and Flask", conversation_id="c", top_k=2)
    assert both == [1, 3]                        # conversation order, not score order
    assert r("tabs or spaces?", conversation_id="c", top_k=1)[0].count("tabs") == 2


def test_bm25_budget_and_isolation():
    r = bmr.BM25PairChunkRetriever({"a": bmr.pair_chunks(MESSAGES), "b": ["User: Flask Flask"]})
    assert len(r("assistant user march", conversation_id="a", top_k=2)) == 2
    assert r("Flask", conversation_id="b", top_k=10) == ["User: Flask Flask"]
    assert r("zebra", conversation_id="a", top_k=3) == []       # no zero-score filler


# ── the adapter arms ──────────────────────────────────────────────────────────

class Recording:
    name = "recording"

    def __init__(self, out):
        self.out, self.calls = out, []

    def __call__(self, question, *, conversation_id, top_k):
        self.calls.append((question, conversation_id, top_k))
        return list(self.out)


def test_full_arm_prompt_is_unchanged(tmp_path):
    item = BEAMAdapter(data_dir=write_parquet_fixture(tmp_path)).extract_all()[0]
    assert item["prompt"].startswith(
        "The following is the complete history of your conversation with the user.\n\n")
    assert item["provenance"] == {"suite": "beam", "split": "100K",
                                  "beam_source": "hf_parquet", "context_mode": "full"}
    assert beam_context_kind_of_prompt(item["prompt"]) == "full"


def test_memory_arms_are_prompt_matched_and_differ_only_in_the_excerpts(tmp_path):
    data = write_parquet_fixture(tmp_path)
    rag = BEAMAdapter(data_dir=data, context_mode="rag", retriever=Recording(["EXCERPT"]))
    trace = BEAMAdapter(data_dir=data, context_mode="trace", retriever=Recording(["EXCERPT"]))
    full = BEAMAdapter(data_dir=data)
    a, b, f = rag.extract_all(), trace.extract_all(), full.extract_all()
    assert [p["prompt"] for p in a] == [p["prompt"] for p in b]   # M-12c(7)
    assert all(p["prompt"].startswith(BEAM_RETRIEVED_HEADER + "EXCERPT\n\n---\n\nUser: ")
               for p in a)
    # Same ids, nuggets, question and reference in every arm.
    for x, y in zip(a, f):
        assert x["id"] == y["id"] and x["scoring_config"] == y["scoring_config"]
        assert x["expected"] == y["expected"]
        assert x["prompt"].split("---\n\nUser: ")[-1] == y["prompt"].split("---\n\nUser: ")[-1]
    assert a[0]["provenance"]["context_mode"] == "rag"
    assert b[0]["provenance"]["context_mode"] == "trace"
    assert a[0]["provenance"]["retrieval_top_k"] == 10
    assert a[0]["provenance"]["chunking"] == "pair_chunk"


def test_injected_retriever_gets_conversation_and_budget(tmp_path):
    rec = Recording([])
    adapter = BEAMAdapter(data_dir=write_parquet_fixture(tmp_path), context_mode="rag",
                          retriever=rec, retrieval_top_k=3)
    item = adapter.extract_all()[0]
    assert rec.calls[0] == ("[1] abstention question 0?", "1", 3)
    assert BEAM_NO_EXCERPTS in item["prompt"] and item["metadata"]["retrieved_chunks"] == 0


def test_default_rag_arm_retrieves_only_from_its_own_conversation(tmp_path):
    adapter = BEAMAdapter(data_dir=write_parquet_fixture(tmp_path), context_mode="rag",
                          retrieval_top_k=1)
    items = adapter.extract_all()
    assert adapter.provenance()["retriever"] == bmr.BM25_RETRIEVER
    for item in items:
        conv = item["metadata"]["conversation_id"]
        body = item["prompt"][len(BEAM_RETRIEVED_HEADER):].split("\n\n---\n\nUser: ")[0]
        assert item["metadata"]["retrieved_chunks"] <= 1
        if body != BEAM_NO_EXCERPTS:
            assert body in adapter._chunks[conv]


def test_env_selects_the_arm_and_budget(monkeypatch, tmp_path):
    monkeypatch.setenv(lca.BEAM_CONTEXT_MODE_ENV, "rag")
    monkeypatch.setenv(lca.BEAM_RETRIEVAL_TOP_K_ENV, "4")
    adapter = BEAMAdapter(data_dir=write_parquet_fixture(tmp_path))
    assert adapter.context_mode == "rag" and adapter.provenance()["retrieval_top_k"] == 4
    assert BEAMAdapter(data_dir=tmp_path, context_mode="full").context_mode == "full"


@pytest.mark.parametrize("kwargs, exc", [
    ({"context_mode": "retrieved"}, ValueError),
    ({"context_mode": "rag", "retrieval_top_k": 0}, ValueError),
    ({"context_mode": "rag", "retriever": "bm25"}, TypeError),
])
def test_bad_arm_configuration_is_refused(kwargs, exc):
    with pytest.raises(exc):
        BEAMAdapter(**kwargs)


def test_a_memory_arm_without_a_retriever_never_degrades(tmp_path):
    adapter = BEAMAdapter(data_dir=write_parquet_fixture(tmp_path), context_mode="rag")
    adapter._ensure_loaded()
    adapter._retriever = None
    with pytest.raises(RuntimeError, match="silently be another arm"):
        adapter._row_to_prompt(0, adapter._dataset[0])


# ── the trace arm (mocked navigation) ─────────────────────────────────────────

class FakeStore:
    class Event:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    def __init__(self):
        self.events = []

    def ensure_schema(self, path):
        store = self

        class Conn:
            def close(self):
                pass
        store.path = path
        return Conn()

    def upsert_events(self, conn, events):
        events = list(events)
        self.events.extend(events)
        return len(events), 0


class FakeNavigation:
    def __init__(self, store, *, legacy=False):
        self.store, self.legacy, self.calls = store, legacy, []

    def search_records(self, text, *, db_path, limit, order=None, **filters):
        if self.legacy and order is not None:
            raise TypeError("query() got an unexpected keyword argument 'order'")
        self.calls.append({"text": text, "limit": limit, "order": order, **filters})
        terms = [t.strip('"') for t in text.split(" OR ")]
        rows = [e for e in self.store.events
                if e.session_id == filters["session_id"]
                and any(t in e.detail_json.lower() for t in terms)]
        rows.sort(key=lambda e: -sum(e.detail_json.lower().count(t) for t in terms))
        return [{"id": i, "source_line": e.source_line} for i, e in enumerate(rows[:limit])]


def _patch_trace(monkeypatch, tmp_path, *, legacy=False):
    import tulving_trace_retriever as ttr

    store = FakeStore()
    nav = FakeNavigation(store, legacy=legacy)
    monkeypatch.setattr(ttr, "_load_trace_modules", lambda root=None: (nav, store))
    monkeypatch.setenv(ttr.DB_DIR_ENV, str(tmp_path / "db"))
    return store, nav


def test_trace_arm_goes_through_navigation_by_relevance_within_the_conversation(
        monkeypatch, tmp_path):
    store, nav = _patch_trace(monkeypatch, tmp_path)
    r = bmr.TracePairChunkRetriever({"c": bmr.pair_chunks(MESSAGES), "d": ["User: Flask"]})
    assert {e.session_id for e in store.events} == {"c", "d"} and len(store.events) == 5
    assert all(e.source == bmr.TRACE_SOURCE for e in store.events)
    out = r("What framework does the budget tracker use?", conversation_id="c", top_k=2)
    call = nav.calls[-1]
    assert call["order"] == "relevance" and call["session_id"] == "c"
    assert call["limit"] == 2 and call["source"] == bmr.TRACE_SOURCE
    assert out and all(chunk in bmr.pair_chunks(MESSAGES) for chunk in out)


def test_trace_arm_refuses_a_recency_ordered_orchestrator(monkeypatch, tmp_path):
    _patch_trace(monkeypatch, tmp_path, legacy=True)
    r = bmr.TracePairChunkRetriever({"c": bmr.pair_chunks(MESSAGES)})
    with pytest.raises(bmr.TraceRetrieverUnavailable, match="6302b381"):
        r("budget", conversation_id="c", top_k=2)


def test_adapter_builds_the_trace_arm_and_cleans_its_store(monkeypatch, tmp_path):
    _patch_trace(monkeypatch, tmp_path)
    adapter = BEAMAdapter(data_dir=write_parquet_fixture(tmp_path), context_mode="trace")
    items = adapter.extract_all()
    assert adapter.provenance()["retriever"] == bmr.TRACE_RETRIEVER
    assert len(items) == 40 and all(p["prompt"].startswith(BEAM_RETRIEVED_HEADER) for p in items)
    store_dir = Path(adapter._retriever._tmpdir)
    assert store_dir.parent == tmp_path / "db"
    adapter._retriever._cleanup()
    assert not store_dir.exists()


def test_missing_trace_surface_is_loud(tmp_path, monkeypatch):
    monkeypatch.setenv("EPYC_ORCHESTRATOR_ROOT", str(tmp_path / "nowhere"))
    with pytest.raises(bmr.TraceRetrieverUnavailable):
        BEAMAdapter(data_dir=write_parquet_fixture(tmp_path), context_mode="trace").extract_all()


def _real_orchestrator() -> Path | None:
    root = Path("/mnt/raid0/llm/epyc-orchestrator")
    query = root / "src" / "trace" / "query.py"
    if query.is_file() and "ORDER_RELEVANCE" in query.read_text():
        return root
    return None


@pytest.mark.skipif(_real_orchestrator() is None, reason="bm25-ordered orchestrator not present")
def test_trace_arm_against_the_real_trace_store(tmp_path, monkeypatch):
    if "src" in sys.modules and not str(getattr(sys.modules["src"], "__path__", [""])[0]).startswith(
            str(_real_orchestrator())):
        pytest.skip("another 'src' package is already imported")
    monkeypatch.setenv("TULVING_TRACE_DB_DIR", str(tmp_path / "db"))
    r = bmr.TracePairChunkRetriever({"c": bmr.pair_chunks(MESSAGES), "d": ["User: Flask Flask"]},
                                    orchestrator_root=_real_orchestrator())
    assert r.search("Which framework does the budget tracker use?",
                    conversation_id="c", top_k=1) == [3]
    assert r.search("Flask", conversation_id="d", top_k=5) == [0]


# ── harness, judge and scorer wiring ──────────────────────────────────────────

def test_suite_keeps_conversations_contiguous_and_pins_params(monkeypatch, tmp_path):
    import dataset_adapters
    import suites

    data = write_parquet_fixture(tmp_path)
    monkeypatch.setattr(dataset_adapters, "get_adapter",
                        lambda name: BEAMAdapter(data_dir=data, context_mode="rag"))
    suite = suites._load_adapter_suite("beam")
    convs = [q.provenance and q.id.split("_")[2] for q in suite.questions]
    assert convs == sorted(convs, key=lambda c: convs.index(c))    # contiguous blocks
    assert convs[:20] == ["1"] * 20 and convs[20:] == ["2"] * 20
    assert suite.questions[0].provenance["context_mode"] == "rag"
    params = suites.get_inference_params(suite)
    assert params["max_tokens"] == 2048 and params["enable_thinking"] is False
    assert params["temperature"] == 0.0 and params["cache_prompt"] is True


def test_judged_payload_carries_the_recorded_arm():
    payload = {"run_id": "r", "results": {"beam": {
        "q1": {"response": "x", "provenance": {"context_mode": "trace"}},
        "q2": {"response": "x", "provenance": {"context_mode": "trace"}},
        "q3": {"response": "x"}}}}
    judged = judge_beam_run.build_judged_payload(
        payload, {"records": [], "unjudged": []}, split="100K", judge_model="gemma")
    assert judged["context_mode_by_row"] == {"trace": 2, "unrecorded": 1}


def _score_main(tmp_path, monkeypatch, arms: dict, arm: str):
    payload = _judged_payload()
    payload["context_mode_by_row"] = arms
    judged = tmp_path / "judged.json"
    judged.write_text(json.dumps(payload))
    monkeypatch.setattr(score_beam_run, "_load_belief_capture",
                        lambda: types.SimpleNamespace(
                            write_belief_measurements=lambda *a, **k: tmp_path / "sidecar"))
    monkeypatch.setattr("sys.argv", ["score_beam_run.py", str(judged), "--out-json",
                                     str(tmp_path / "s.json"), "--belief-measurements",
                                     "--arm", arm])
    return score_beam_run.main()


def test_scorer_refuses_an_arm_the_rows_do_not_record(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="disagrees"):
        _score_main(tmp_path, monkeypatch, {"rag": 40}, "trace")
    with pytest.raises(SystemExit, match="disagrees"):
        _score_main(tmp_path, monkeypatch, {"unrecorded": 40}, "full")


def test_scorer_accepts_the_recorded_arm(tmp_path, monkeypatch):
    assert _score_main(tmp_path, monkeypatch, {"trace": 40}, "trace") == 0
    summary = json.loads((tmp_path / "s.json").read_text())["summary"]
    assert summary["context_mode_by_row"] == {"trace": 40}
