"""UFH-12: ``context.search`` over a request's context bundle returns POINTERS.

Offline and inference-free (dense uses fake embedders). Acceptance:
  * hits are pointers only -- ``{section, start, end, line, score, via, mode}`` -- and carry
    no section text; the search echo carries none either;
  * BM25 and dense rankings are fused by ``rrf_fuse`` in the expected order;
  * pull accounting is unchanged by searching: totals, per-section and per-turn records are
    identical with and without searches, and pulling a hit is counted exactly like any get;
  * no embedder / a refusing pool -> every hit is LABELLED lexical (no pseudo-embedding);
  * an index belongs to one embedding model;
  * default off: without ``context_search`` the view, the root block and the echo are as before.

Run: nice -n 19 .venv/bin/python -m pytest tests/unit/test_ufh12_context_search.py -q
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from pydantic import ValidationError

from src.api.models import ChatRequest
from src.embedding_pool.fake import FakePooledEmbedder
from src.repl_environment import REPLEnvironment
from src.repl_environment.context_bundle import ContextBundle, repl_view
from src.repl_environment import context_search as cs
from src.repl_environment.context_search import (
    DENSE_BATCH_CHUNKS,
    INDEX_TIMEOUT_MAX_S,
    INDEX_TIMEOUT_MIN_S,
    MAX_DENSE_ATTEMPTS,
    MAX_DENSE_CHUNKS,
    BM25Index,
    ContextSearchIndex,
    DenseResult,
    OutcomeEmbedder,
    chunk_sections,
    index_build_timeout_s,
    tokenize,
)
from src.trace.navigation import rrf_fuse

SENTINEL = "ZQX-SENTINEL-7f3a"
POINTER_KEYS = {"section", "start", "end", "line", "score", "via", "mode"}


def _payload():
    return {
        "sections": [
            {"name": "preamble", "text": "## Task\nPropose ONE kernel change.\n", "inline": True},
            {"name": "profile",
             "text": "\n".join(f"op {i}: ggml_vec_dot_q8_0 share {i}% of decode" for i in range(60))
                     + "\nflash attention kernel dominates prefill at long context\n"},
            {"name": "big", "text": "\n".join(f"row {i}: filler {'x' * 60}" for i in range(300))
                                    + f"\nneedle: {SENTINEL} lives in the numa interleave note\n"},
            {"name": "target", "kind": "json",
             "text": json.dumps({"recipe": {"model": "Qwen3.8-27B-Q8_0.gguf", "threads": 96}})},
        ]
    }


def _bundle(**kw) -> ContextBundle:
    return ContextBundle.from_payload(_payload(), **kw)


def _fake(available: bool = True, model: str = "fake-embedder") -> tuple[OutcomeEmbedder, FakePooledEmbedder]:
    fake = FakePooledEmbedder(dim=64, available=available)
    return OutcomeEmbedder(fake.try_embed_many_sync, model), fake


class ScriptedEmbedder:
    """Returns a fixed vector per text by first matching marker; the query is its own key."""

    def __init__(self, table: dict[str, list[float]], model_id: str = "scripted") -> None:
        self.table = table
        self.model_id = model_id
        self.calls: list[list[str]] = []

    def embed(self, texts, *, timeout_s, admission_wait_s):
        self.calls.append(list(texts))
        rows = []
        for t in texts:
            vec = next((v for marker, v in self.table.items() if marker in t), None)
            if vec is None:
                vec = [0.0, 0.0, 0.0, 1.0]
            v = np.asarray(vec, dtype=np.float32)
            rows.append(v / np.linalg.norm(v))
        return DenseResult(np.stack(rows), self.model_id)


# ─────────────────────────────────────────────────────────────── chunker / lexical


def test_chunks_are_line_aligned_and_reconstruct_every_section():
    bundle = _bundle()
    chunks = chunk_sections(bundle.sections, max_chars=200)
    for s in bundle.sections:
        mine = [c for c in chunks if c.section == s.name]
        assert "".join(c.text for c in mine) == s.text           # a partition, nothing lost
        assert all(c.text == s.text[c.start:c.end] for c in mine)
        assert all(len(c.text) <= 200 for c in mine)
        for c in mine:
            # line-aware: a chunk starts at a line start, and `line` matches context.grep's
            assert c.start == 0 or s.text[c.start - 1] == "\n" or len(s.text.split("\n")[c.line - 1]) > 200
            assert s.text[:c.start].count("\n") + 1 == c.line


def test_a_line_longer_than_a_chunk_is_cut():
    class S:
        name, text = "s", "short\n" + "y" * 450 + "\ntail\n"

    chunks = chunk_sections([S], max_chars=200)
    assert "".join(c.text for c in chunks) == S.text
    assert max(len(c.text) for c in chunks) <= 200
    assert [c.line for c in chunks if "y" in c.text] == [2, 2, 2]


def test_bm25_ranks_term_overlap_and_splits_identifiers():
    assert "vec" in tokenize("ggml_vec_dot_q8_0") and "ggml_vec_dot_q8_0" in tokenize("ggml_vec_dot_q8_0")
    idx = BM25Index(["numa interleave numa", "numa once", "nothing relevant here"])
    ranked = idx.scores("numa interleave")
    assert [i for i, _ in ranked] == [0, 1]                       # no-overlap docs excluded
    assert idx.scores("absent words") == []


# ─────────────────────────────────────────────────────────────── pointers only


def test_search_returns_pointers_only_and_no_text():
    bundle = _bundle(search_enabled=True)
    emb, _ = _fake()
    bundle.set_embedder(emb)
    hits = bundle.search("numa interleave needle", k=5)
    assert hits and all(set(h) == POINTER_KEYS for h in hits)
    assert hits[0]["section"] == "big" and hits[0]["mode"] == "hybrid"
    for h in hits:
        assert isinstance(h["start"], int) and isinstance(h["end"], int) and h["end"] > h["start"]
        assert isinstance(h["score"], float)
        # no section text in any value
        flat = json.dumps(h)
        assert SENTINEL not in flat and "filler" not in flat and "ggml" not in flat
    # the echo keeps pointers, never text
    echo = json.dumps(bundle.accounting()["search"])
    assert SENTINEL not in echo and "filler" not in echo
    # a pointer pulls exactly its span through get()
    h = hits[0]
    text = bundle.get(h["section"], offset=h["start"], max_chars=h["end"] - h["start"])
    assert SENTINEL in text and text == bundle.sections[2].text[h["start"]:h["end"]]


def test_search_section_filter_and_bad_input():
    bundle = _bundle(search_enabled=True)
    bundle.set_embedder(None)
    hits = bundle.search("kernel", k=8, section="profile")
    assert hits and {h["section"] for h in hits} == {"profile"}
    with pytest.raises(KeyError):
        bundle.search("kernel", section="nope")
    with pytest.raises(ValueError):
        bundle.search("   ")


# ─────────────────────────────────────────────────────────────── RRF fusion order


def test_rrf_fusion_order_matches_rrf_fuse_over_both_rankings():
    # three one-chunk sections; lexical ranks A > C (B has no overlap), dense ranks B > C > A
    payload = {"sections": [
        {"name": "A", "text": "alpha beta alpha beta marker-a\n"},
        {"name": "B", "text": "unrelated words marker-b\n"},
        {"name": "C", "text": "alpha gamma delta epsilon zeta marker-c\n"},
    ]}
    bundle = ContextBundle.from_payload(payload, search_enabled=True)
    emb = ScriptedEmbedder({
        "marker-a": [0.0, 0.0, 1.0, 0.0],
        "marker-b": [1.0, 0.0, 0.0, 0.0],
        "marker-c": [0.8, 0.6, 0.0, 0.0],
        "alpha beta": [1.0, 0.0, 0.0, 0.0],   # the query
    })
    bundle.set_embedder(emb)
    hits = bundle.search("alpha beta", k=3)
    lexical = [{"id": "A", "_rank_source": "bm25"}, {"id": "C", "_rank_source": "bm25"}]
    dense = [{"id": n, "_rank_source": "dense"} for n in ("B", "C", "A")]
    expected = rrf_fuse([lexical, dense], key="id", k=60, limit=3)
    assert [h["section"] for h in hits] == [r["id"] for r in expected] == ["A", "C", "B"]
    assert [h["score"] for h in hits] == [r["_rrf_score"] for r in expected]
    assert [h["via"] for h in hits] == [["bm25", "dense"], ["bm25", "dense"], ["dense"]]
    assert all(h["mode"] == "hybrid" for h in hits)
    # the query was embedded separately from the chunks (index once, then one query)
    assert [len(c) for c in emb.calls] == [3, 1]


def test_lexical_only_is_labelled_and_uses_bm25_alone():
    bundle = _bundle(search_enabled=True)
    bundle.set_embedder(None)
    hits = bundle.search("flash attention prefill", k=3)
    assert hits[0]["section"] == "profile"
    assert all(h["mode"] == "lexical" and h["via"] == ["bm25"] for h in hits)
    entry = bundle.accounting()["search"]["log"][0]
    assert entry["mode"] == "lexical" and entry["reason"] == "disabled"


def test_flag_off_pool_means_lexical_disabled(monkeypatch):
    import src.embedding_pool as pool

    monkeypatch.setattr(pool, "pool_enabled", lambda: False)
    bundle = _bundle(search_enabled=True)                      # default embedder = the pool
    hits = bundle.search("numa interleave")
    assert hits and all(h["mode"] == "lexical" for h in hits)
    assert bundle.accounting()["search"]["log"][0]["reason"] == "disabled"


def test_refusing_pool_degrades_to_labelled_lexical_with_bounded_retries():
    bundle = _bundle(search_enabled=True)
    emb, fake = _fake(available=False)
    bundle.set_embedder(emb)
    for _ in range(MAX_DENSE_ATTEMPTS + 2):
        hits = bundle.search("numa interleave")
        assert hits and all(h["mode"] == "lexical_fallback:saturated" for h in hits)
    assert len(fake.calls) == MAX_DENSE_ATTEMPTS                 # index retried, then given up
    search = bundle.accounting()["search"]
    assert search["modes"] == {"lexical_fallback:saturated": MAX_DENSE_ATTEMPTS + 2}
    assert search["log"][0]["reason"] == "saturated"
    assert search["index"]["dense"] is False


def test_a_timed_out_index_build_with_no_progress_is_not_retried():
    class Slow:
        model_id = "slow"
        calls = 0

        def embed(self, texts, **kw):
            Slow.calls += 1
            return DenseResult(None, self.model_id, "timeout")

    bundle = _bundle(search_enabled=True)
    bundle.set_embedder(Slow())
    for _ in range(3):
        assert all(h["mode"] == "lexical_fallback:index_timeout" for h in bundle.search("numa"))
    assert Slow.calls == 1
    assert bundle.accounting()["search"]["modes"] == {"lexical_fallback:index_timeout": 3}


def test_an_index_belongs_to_one_embedding_model():
    bundle = _bundle(search_enabled=True)
    e1, f1 = _fake(model="model-one")
    bundle.set_embedder(e1)
    bundle.search("numa")
    n_chunks = bundle.accounting()["search"]["index"]["chunks"]
    assert [len(c) for c in f1.calls] == [n_chunks, 1]
    e2, f2 = _fake(model="model-two")
    bundle.set_embedder(e2)
    bundle.search("numa")
    # the new model re-embeds every chunk; its query is never scored against model-one vectors
    assert [len(c) for c in f2.calls] == [n_chunks, 1]
    log = bundle.accounting()["search"]["log"]
    assert [e["embedder_model"] for e in log] == ["model-one", "model-two"]


def test_query_from_another_model_is_refused():
    idx = ContextSearchIndex(_bundle().sections)

    class Liar(ScriptedEmbedder):
        def embed(self, texts, **kw):
            res = super().embed(texts, **kw)
            return res if len(texts) > 1 else DenseResult(res.vectors, "other-model")

    hits, meta = idx.search("numa", 3, section=None, embedder=Liar({}))
    assert meta["reason"] == "model_mismatch"
    assert all(h["mode"] == "lexical_fallback:model_mismatch" for h in hits)


# ─────────────────────────────────────────────────────────────── pull accounting unchanged


def test_pull_accounting_is_unchanged_by_search():
    plain = _bundle()
    searched = _bundle(search_enabled=True)
    emb, _ = _fake()
    searched.set_embedder(emb)
    for b, do_search in ((plain, False), (searched, True)):
        b.begin_turn(1)
        if do_search:
            b.search("numa interleave")
            b.search("flash attention", section="profile")
        b.get("big", max_chars=100, offset=50)
        b.grep(SENTINEL)
        b.begin_turn(2)
        if do_search:
            b.search("kernel change")
        b.get("profile", max_chars=200)
    acc_plain = plain.accounting()
    acc_searched = searched.accounting()
    assert "search" not in acc_plain                             # default echo is as before
    assert acc_searched.pop("search")["calls"] == 3
    assert acc_searched == acc_plain                             # totals, sections, turns: identical


def test_search_alone_pulls_nothing_and_ignores_the_pull_budget():
    bundle = _bundle(search_enabled=True, pull_budget_bytes=1)
    bundle.set_embedder(None)
    assert bundle.search("needle")
    totals = bundle.accounting()["totals"]
    assert totals["bytes_pulled"] == 0 and totals["pull_calls"] == 0 and totals["refused"] == 0


# ─────────────────────────────────────────────────────────────── REPL / prompt / contract


def test_repl_search_then_pull_is_counted_like_any_get():
    bundle = _bundle(search_enabled=True)
    emb, _ = _fake()
    bundle.set_embedder(emb)
    repl = REPLEnvironment(context="root prompt text")
    repl.attach_context_bundle(bundle)
    r = repl.execute(
        "hits = context.search('where is the numa interleave needle', k=3)\n"
        "h = hits[0]\n"
        "span = context.get(h['section'], offset=h['start'], max_chars=h['end'] - h['start'])\n"
        "print(h['mode'], h['section'], SENT in span)".replace("SENT", repr(SENTINEL))
    )
    assert r.error is None, r.error
    assert r.output.strip() == "hybrid big True"
    acc = bundle.accounting()
    ops = [p["op"] for t in acc["turns"] for p in t["pulls"]]
    assert ops == ["get"]                                        # the search is not a pull
    span = acc["search"]["log"][0]["hits"][0]
    assert acc["totals"]["bytes_pulled"] == len(bundle.sections[2].text[span[1]:span[2]].encode())


def test_view_and_root_block_are_unchanged_unless_enabled():
    off, on = _bundle(), _bundle(search_enabled=True)
    assert [a for a in dir(repl_view(off)) if not a.startswith("__")] == \
        ["get", "grep", "index", "json", "keys"]
    assert [a for a in dir(repl_view(on)) if not a.startswith("__")] == \
        ["get", "grep", "index", "json", "keys", "search"]
    assert "context.search" not in off.render_root_block() and "search" not in repr(repl_view(off))
    block = on.render_root_block()
    assert "context.search(query, k=8, section=None)" in block and "POINTERS" in block
    assert "context.search" in repr(repl_view(on))
    assert on.render_root_block().replace(block, "") == ""       # deterministic
    # the only difference between the two blocks is the one search line
    extra = [ln for ln in block.split("\n") if ln not in off.render_root_block().split("\n")]
    assert len(extra) == 1 and extra[0].startswith("- context.search(")


def test_request_contract():
    with pytest.raises(ValidationError, match="context_search requires context_bundle"):
        ChatRequest(prompt="x", force_mode="repl", context_search=True)
    req = ChatRequest(prompt="x", force_mode="repl", context_bundle=_payload(), context_search=True)
    assert req.context_search is True
    assert ChatRequest(prompt="x", force_mode="repl", context_bundle=_payload()).context_search is False


def test_search_is_an_exploration_turn_not_a_final_answer():
    from src.prompt_builders.code_utils import auto_wrap_final

    code = "hits = context.search('numa')\nhits"
    assert auto_wrap_final(code) == code


# ─────────────────────────────────────────────────────────────── the actor CLI


def _bundle_file(tmp_path):
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    return path


def test_cli_sends_context_search_and_requires_the_echo(tmp_path):
    from tests.unit.test_autokernel_actor_cli import MockChat, chat_response, run_main

    pulls = {"schema": "epyc.orchestrator.context_pulls.v1", "totals": {},
             "search": {"enabled": True, "calls": 2}}
    sidecar = tmp_path / "prov.json"
    with MockChat(body=chat_response('{"ok": true}', context_pulls=pulls)) as mock:
        code, out, err = run_main(["--root", str(tmp_path), "--url", mock.url,
                                   "--context-bundle", str(_bundle_file(tmp_path)),
                                   "--context-search", "--provenance-out", str(sidecar)])
    assert code == 0, err
    assert mock.requests[0]["body"]["context_search"] is True
    record = json.loads(sidecar.read_text())
    assert record["context_search_acknowledged"] is True
    assert record["request"]["context_bundle"]["search"] is True

    no_search = {"schema": "epyc.orchestrator.context_pulls.v1", "totals": {}}
    with MockChat(body=chat_response('{"ok": true}', context_pulls=no_search)) as mock:
        code, out, err = run_main(["--root", str(tmp_path), "--url", mock.url,
                                   "--context-bundle", str(_bundle_file(tmp_path)),
                                   "--context-search"])
    assert code == 1 and "did not enable context.search" in err

    with MockChat(body=chat_response('{"ok": true}', context_pulls=no_search)) as mock:
        code, out, err = run_main(["--root", str(tmp_path), "--url", mock.url,
                                   "--context-bundle", str(_bundle_file(tmp_path))])
    assert code == 0, err
    assert "context_search" not in mock.requests[0]["body"]      # off by default

    code, _, err = run_main(["--root", str(tmp_path), "--context-search"])
    assert code == 1 and "need --context-bundle" in err


# ─────────────────────────────────────────────────────────────── index build budget


def test_index_build_budget_scales_with_chunks_under_a_hard_ceiling():
    assert index_build_timeout_s(0) == INDEX_TIMEOUT_MIN_S == 10.0
    assert index_build_timeout_s(100) == INDEX_TIMEOUT_MIN_S         # 5 + 4 s, floored
    assert index_build_timeout_s(500) == pytest.approx(25.0)         # 5 + 500 / 25
    assert index_build_timeout_s(1375) == pytest.approx(60.0)
    assert index_build_timeout_s(MAX_DENSE_CHUNKS) == INDEX_TIMEOUT_MAX_S == 60.0
    # a bundle at the chunk cap finishes in at most three searches at the assumed rate
    per_search = (INDEX_TIMEOUT_MAX_S - cs.INDEX_TIMEOUT_BASE_S) * cs.INDEX_ASSUMED_TEXTS_PER_S
    assert 3 * per_search >= MAX_DENSE_CHUNKS


def _many_sections_payload(n_chunks: int):
    # one ~700-char line per chunk, each with its own marker word
    line = "{} " + "pad " * 170
    text = "".join(line.format(f"mark{i}").rstrip() + "\n" for i in range(n_chunks))
    return {"sections": [{"name": "spill", "text": text}]}


class BatchBudget:
    """Embeds ``per_attempt`` batches per index-build attempt, then reports a pool timeout;
    a one-text call (the query) always succeeds. A pool too slow to finish in one budget."""

    model_id = "batchy"

    def __init__(self, per_attempt: int) -> None:
        self.per_attempt = per_attempt
        self.batches_this_attempt = 0
        self.batch_sizes: list[int] = []

    def embed(self, texts, *, timeout_s, admission_wait_s):
        if len(texts) == 1:
            return DenseResult(np.ones((1, 4), dtype=np.float32) / 2.0, self.model_id)
        if self.batches_this_attempt >= self.per_attempt:
            self.batches_this_attempt = 0
            return DenseResult(None, self.model_id, "timeout")
        self.batches_this_attempt += 1
        self.batch_sizes.append(len(texts))
        return DenseResult(np.ones((len(texts), 4), dtype=np.float32) / 2.0, self.model_id)


def test_a_timed_out_build_keeps_its_progress_labels_it_and_resumes():
    n = 3 * DENSE_BATCH_CHUNKS + 10                      # four batches
    bundle = ContextBundle.from_payload(_many_sections_payload(n), search_enabled=True)
    emb = BatchBudget(per_attempt=2)
    bundle.set_embedder(emb)

    first = bundle.search("mark3", k=3)
    index = bundle.accounting()["search"]["index"]
    assert index["chunks"] == n and index["dense"] is False
    assert index["dense_chunks_done"] == 2 * DENSE_BATCH_CHUNKS     # progress kept
    assert index["dense_reason"] == "index_timeout"
    assert index["dense_timeout_s"] == pytest.approx(index_build_timeout_s(n))
    assert first and all(h["mode"] == "lexical_fallback:index_timeout" for h in first)
    assert all(h["via"] == ["bm25"] for h in first)                 # never a partial dense ranking

    second = bundle.search("mark3", k=3)                            # resumes, finishes
    index = bundle.accounting()["search"]["index"]
    assert index["dense"] is True and index["dense_chunks_done"] == n
    assert index["dense_attempts"] == 2 and index["dense_failures"] == 0
    assert index["dense_timeout_s"] == pytest.approx(index_build_timeout_s(n - 2 * DENSE_BATCH_CHUNKS))
    assert second and all(h["mode"] == "hybrid" for h in second)
    # every chunk embedded exactly once, in batches of at most DENSE_BATCH_CHUNKS
    assert sum(emb.batch_sizes) == n and max(emb.batch_sizes) == DENSE_BATCH_CHUNKS
    # the experiment can count the fallback per search
    assert bundle.accounting()["search"]["modes"] == {
        "lexical_fallback:index_timeout": 1, "hybrid": 1}


def test_the_build_budget_is_a_deadline_across_batches(monkeypatch):
    n = 3 * DENSE_BATCH_CHUNKS
    idx = ContextSearchIndex(ContextBundle.from_payload(_many_sections_payload(n)).sections)
    clock = [1000.0]
    monkeypatch.setattr(cs, "_clock", lambda: clock[0])
    seen: list[float] = []

    class Ticking:
        """Each batch takes 8 s; a batch given less than that times out, as the pool does."""

        model_id = "tick"

        def embed(self, texts, *, timeout_s, admission_wait_s):
            seen.append(timeout_s)
            if timeout_s < 8.0:
                clock[0] += timeout_s
                return DenseResult(None, self.model_id, "timeout")
            clock[0] += 8.0
            return DenseResult(np.ones((len(texts), 4), dtype=np.float32) / 2.0, self.model_id)

    budget = index_build_timeout_s(n)                        # 5 + 384 / 25 = 20.36 s
    assert budget == pytest.approx(20.36)
    assert idx.ensure_dense(Ticking()) == "index_timeout"   # 3 batches need 24 s
    assert seen == pytest.approx([budget, budget - 8.0, budget - 16.0])
    assert idx.dense_done == 2 * DENSE_BATCH_CHUNKS and idx.vectors is None
    assert idx.dense_failures == 0                           # progress: not a failure
    assert idx.ensure_dense(Ticking()) is None               # the next search finishes it
    assert idx.vectors.shape == (n, 4)


def test_too_large_and_query_failures_are_labelled_fallbacks(monkeypatch):
    monkeypatch.setattr(cs, "MAX_DENSE_CHUNKS", 2)
    bundle = _bundle(search_enabled=True)
    emb, _ = _fake()
    bundle.set_embedder(emb)
    assert all(h["mode"] == "lexical_fallback:too_large" for h in bundle.search("numa"))
    monkeypatch.undo()

    class QueryTimesOut(ScriptedEmbedder):
        def embed(self, texts, **kw):
            res = super().embed(texts, **kw)
            return res if len(texts) > 1 else DenseResult(None, self.model_id, "timeout")

    idx = ContextSearchIndex(_bundle().sections)
    hits, meta = idx.search("numa", 3, section=None, embedder=QueryTimesOut({}))
    assert meta["reason"] == "query_timeout"
    assert all(h["mode"] == "lexical_fallback:query_timeout" for h in hits)


# ─────────────────────────────────────────────────────────────── golden: field off

#: sha256 of each surface for ``_GOLDEN_PAYLOAD``, recorded on orchestrator main 08edc054 (before
#: UFH-12 context.search existed). With ``context_search`` off, the bundle the model sees -- root
#: block, REPL view, pull echo, payload -- must stay byte-identical to that pre-feature build.
_GOLDEN_SHA256 = {
    "root_block": "01595ea912e161913675ca5f353ade0c7b76d8eac05dbe2c717d90e3616083c3",
    "repr": "48c397c66047c1b73f891941006440d1e2a3bfff8f40b696e14d9bddd81d0ddd",
    "dir": "4b7f611a09a33c2205bca02f3bfa19a8223947039f7cdda43568ccab8fcce26e",
    "accounting": "a1c29e8f3aed1cc68bb053b9f16a52093372dd34ef678936580a61ec24437cef",
    "payload": "cf1623d8fb59e782e15b1a8f319a96cb4049e5a5b85f3f75193667211ea8ffcc",
}
_GOLDEN_PAYLOAD = {"sections": [
    {"name": "spec", "text": "# Spec\nThe NUMA policy pins weights.\nUse interleave for decode.\n" * 20},
    {"name": "data", "kind": "json", "text": json.dumps({"a": [1, 2, {"b": "numa"}], "c": "x" * 300})},
    {"name": "notes", "text": "line one\nggml_vec_dot_q8_0 is hot\n" * 50, "inline": False,
     "description": "perf notes"},
]}


@pytest.mark.parametrize("explicit", [False, True])
def test_field_off_is_byte_identical_to_the_pre_feature_build(explicit):
    import hashlib

    kw = {"search_enabled": False} if explicit else {}
    b = ContextBundle.from_payload(_GOLDEN_PAYLOAD, print_cap_bytes=1234, pull_budget_bytes=100000, **kw)
    v = repl_view(b)
    b.begin_turn(1)
    v.get("spec", max_chars=50, offset=10)
    v.grep("numa", k=3)
    v.json("data.a[2].b")
    v["notes"]
    b.begin_turn(2)
    b.record_printed(100, 80, True)
    surfaces = {
        "root_block": b.render_root_block(),
        "repr": repr(v),
        "dir": [a for a in dir(v) if not a.startswith("__")],
        "accounting": b.accounting(),
        "payload": b.to_payload(),
    }
    got = {
        name: hashlib.sha256(
            (obj if isinstance(obj, str) else json.dumps(obj, sort_keys=True)).encode()
        ).hexdigest()
        for name, obj in surfaces.items()
    }
    assert got == _GOLDEN_SHA256
    assert "search" not in surfaces["accounting"]
