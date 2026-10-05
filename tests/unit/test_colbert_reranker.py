"""Tests for ColBERT reranker module.

Every test here MOCKS the shared encoder: no ONNX session is created and no
model is loaded. The reranker is a thin consumer — what is worth testing is the
CONTRACT it uses the encoder with (role per call, model-declared token caps) and
the ordering it derives from returned scores.
"""

import inspect
import json

import numpy as np
import pytest

from src.retrieval import colbert_encoder
from src.tools.web.colbert_reranker import (
    _maxsim,
    rerank_snippets,
    is_available,
)


@pytest.fixture
def stub_encoder(monkeypatch):
    """Record every `colbert_encoder.encode()` call the reranker makes.

    Returns the call list; each entry is (text, max_tokens, role). The default
    embedding is a fixed unit vector, so a test that does not care about scores
    gets deterministic, finite ones.
    """
    calls = []
    default = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    def fake_encode(text, max_tokens, *, role):
        calls.append((text, max_tokens, role))
        return default

    monkeypatch.setattr(colbert_encoder, "ensure_loaded", lambda: True)
    monkeypatch.setattr(colbert_encoder, "prefix_tokens_available", lambda: True)
    monkeypatch.setattr(colbert_encoder, "encode", fake_encode)
    return calls


class TestMaxSim:
    """Test MaxSim scoring function."""

    def test_identical_embeddings_score_one(self):
        """Identical normalized embeddings → MaxSim = 1.0."""
        emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        assert _maxsim(emb, emb) == pytest.approx(1.0)

    def test_orthogonal_embeddings_score_zero(self):
        """Orthogonal query/doc tokens → MaxSim = 0.0."""
        query = np.array([[1.0, 0.0]], dtype=np.float32)
        doc = np.array([[0.0, 1.0]], dtype=np.float32)
        assert _maxsim(query, doc) == pytest.approx(0.0)

    def test_partial_overlap_intermediate_score(self):
        """Partial overlap → score between 0 and 1."""
        query = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        doc = np.array([[0.707, 0.707]], dtype=np.float32)  # 45-degree
        score = _maxsim(query, doc)
        assert 0.0 < score < 1.0

    def test_multiple_doc_tokens_max_per_query(self):
        """MaxSim takes max per query token across all doc tokens."""
        query = np.array([[1.0, 0.0]], dtype=np.float32)
        doc = np.array([
            [0.0, 1.0],  # low sim
            [0.9, 0.1],  # high sim
        ], dtype=np.float32)
        # Normalize doc tokens
        doc = doc / np.linalg.norm(doc, axis=1, keepdims=True)
        score = _maxsim(query, doc)
        assert score > 0.8  # Should pick the high-sim doc token

    def test_delegates_to_the_shared_implementation(self):
        """`_maxsim` must not be a second copy of the scoring maths."""
        query = np.array([[1.0, 0.0], [0.3, 0.9]], dtype=np.float32)
        doc = np.array([[0.6, 0.8], [0.0, 1.0]], dtype=np.float32)
        assert _maxsim(query, doc) == colbert_encoder.maxsim(query, doc)


class TestRerankSnippets:
    """Test rerank_snippets with the shared encoder mocked."""

    def test_empty_snippets_returns_empty(self):
        result = rerank_snippets("query", [], top_k=3)
        assert result == []

    def test_model_unavailable_returns_original_order(self, monkeypatch):
        """When the encoder cannot load, returns original snippets (graceful)."""
        monkeypatch.setattr(colbert_encoder, "ensure_loaded", lambda: False)
        snippets = [
            {"title": "A", "snippet": "First"},
            {"title": "B", "snippet": "Second"},
        ]
        result = rerank_snippets("query", snippets, top_k=2)
        assert len(result) == 2
        assert result[0]["title"] == "A"  # Original order preserved

    def test_reranks_by_maxsim(self, monkeypatch):
        """Snippets reranked by MaxSim when model available."""
        # Query embedding: points in x direction
        query_emb = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

        # Doc embeddings: B is more aligned with query than A
        doc_a = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)  # orthogonal
        doc_b = np.array([[0.9, 0.1, 0.0]], dtype=np.float32)  # aligned

        def encode_side_effect(text, max_tokens, *, role):
            if "irrelevant" in text.lower():
                return doc_a
            elif "relevant" in text.lower():
                return doc_b
            return query_emb  # query

        monkeypatch.setattr(colbert_encoder, "ensure_loaded", lambda: True)
        monkeypatch.setattr(colbert_encoder, "prefix_tokens_available", lambda: True)
        monkeypatch.setattr(colbert_encoder, "encode", encode_side_effect)

        snippets = [
            {"title": "Irrelevant", "snippet": "Irrelevant content about cooking"},
            {"title": "Relevant", "snippet": "Relevant content about the query topic"},
        ]

        result = rerank_snippets("test query", snippets, top_k=2)
        assert len(result) == 2
        assert result[0]["title"] == "Relevant"
        assert "rerank_score" in result[0]
        assert result[0]["rerank_score"] > result[1]["rerank_score"]

    def test_ranking_follows_a_fake_score_matrix_exactly(self, monkeypatch):
        """Order is the descending score order, for scores we fully control.

        One query token per axis and one-hot document embeddings make MaxSim
        read the chosen coefficient straight off, so the expected permutation is
        arithmetic rather than approximate.
        """
        query_emb = np.eye(1, 4, dtype=np.float32)  # [[1,0,0,0]]
        wanted = {"d0": 0.10, "d1": 0.90, "d2": 0.50, "d3": 0.70}

        def encode_side_effect(text, max_tokens, *, role):
            # Keyed on TEXT, not role, so this test measures ordering only —
            # the role contract is TestRoleContract's job.
            if text == "q":
                return query_emb
            key = text.split(".")[0]
            vec = np.zeros((1, 4), dtype=np.float32)
            vec[0, 0] = wanted[key]
            vec[0, 1] = float(np.sqrt(max(0.0, 1.0 - wanted[key] ** 2)))
            return vec

        monkeypatch.setattr(colbert_encoder, "ensure_loaded", lambda: True)
        monkeypatch.setattr(colbert_encoder, "prefix_tokens_available", lambda: True)
        monkeypatch.setattr(colbert_encoder, "encode", encode_side_effect)

        snippets = [{"title": k, "snippet": "body"} for k in wanted]
        result = rerank_snippets("q", snippets, top_k=4)

        assert [r["title"] for r in result] == ["d1", "d3", "d2", "d0"]
        assert [r["rerank_score"] for r in result] == [0.9, 0.7, 0.5, 0.1]

    def test_top_k_limits_output(self, stub_encoder):
        """Returns at most top_k snippets."""
        snippets = [{"snippet": f"doc {i}"} for i in range(10)]
        result = rerank_snippets("query", snippets, top_k=3)
        assert len(result) == 3

    def test_empty_snippet_text_scores_zero(self, monkeypatch):
        """Snippets with no text get score 0.0 (no encoding attempted)."""
        snippets = [
            {"title": "", "snippet": ""},
            {"title": "Real", "snippet": "Has actual content"},
        ]

        query_emb = np.array([[1.0, 0.0]], dtype=np.float32)
        doc_emb = np.array([[0.8, 0.2]], dtype=np.float32)
        encoded_texts = []

        def encode_side_effect(text, max_tokens, *, role):
            encoded_texts.append(text)
            return doc_emb if "actual" in text.lower() else query_emb

        monkeypatch.setattr(colbert_encoder, "ensure_loaded", lambda: True)
        monkeypatch.setattr(colbert_encoder, "prefix_tokens_available", lambda: True)
        monkeypatch.setattr(colbert_encoder, "encode", encode_side_effect)

        result = rerank_snippets("query", snippets, top_k=2)
        # Empty snippet gets score 0.0 (no text → skipped)
        assert result[0]["title"] == "Real"
        assert not any(t.strip().strip(".") == "" for t in encoded_texts)

    def test_encode_failure_returns_original(self, monkeypatch):
        """Encoding failure → returns original order (top_k sliced)."""
        monkeypatch.setattr(colbert_encoder, "ensure_loaded", lambda: True)
        monkeypatch.setattr(colbert_encoder, "prefix_tokens_available", lambda: True)
        monkeypatch.setattr(
            colbert_encoder, "encode", lambda text, max_tokens, *, role: None
        )
        snippets = [{"snippet": f"doc {i}"} for i in range(5)]
        result = rerank_snippets("query", snippets, top_k=3)
        assert len(result) == 3


class TestRoleContract:
    """PREFIX-1: the reranker must ASK for the trained `[Q]`/`[D]` roles.

    The old private `_encode` had no role parameter at all, so query and document
    were both encoded off-distribution. Measured on the reference model, the
    missing prefix moves MaxSim by max |delta| 1.63e-01 and flips top-1 on 37.5%
    of queries — ~25x the perturbation of the INT8 quantization we accept. A
    future A/B run through a prefix-less encoder measures the encoder.
    """

    def test_query_is_encoded_with_role_query(self, stub_encoder):
        rerank_snippets("what causes aurora?", [{"snippet": "northern lights"}], top_k=1)
        assert stub_encoder, "the reranker encoded nothing"
        text, _, role = stub_encoder[0]
        assert text == "what causes aurora?"
        assert role == colbert_encoder.ROLE_QUERY

    def test_every_candidate_is_encoded_with_role_document(self, stub_encoder):
        snippets = [{"title": f"t{i}", "snippet": f"body {i}"} for i in range(4)]
        rerank_snippets("q", snippets, top_k=2)

        doc_calls = stub_encoder[1:]
        assert len(doc_calls) == 4, "one encode per candidate expected"
        assert {role for _, _, role in doc_calls} == {colbert_encoder.ROLE_DOCUMENT}
        assert [text for text, _, _ in doc_calls] == [
            f"t{i}. body {i}" for i in range(4)
        ]

    def test_roles_never_mix_conventions(self, stub_encoder):
        """Both sides prefixed, or (fallback) neither — never one of each."""
        rerank_snippets("q", [{"snippet": "a"}, {"snippet": "b"}], top_k=2)
        roles = {role for _, _, role in stub_encoder}
        assert roles == {colbert_encoder.ROLE_QUERY, colbert_encoder.ROLE_DOCUMENT}

    def test_missing_prefix_tokens_degrade_both_sides_to_role_none(self, monkeypatch):
        """A checkpoint without the trained tokens must not raise on every call.

        `encode()` REFUSES a prefixed role when the tokenizer lacks the token, so
        blindly asking for it would break reranking entirely. Falling back is
        legitimate only because both sides fall back together.
        """
        calls = []

        def fake_encode(text, max_tokens, *, role):
            calls.append(role)
            return np.array([[1.0, 0.0]], dtype=np.float32)

        monkeypatch.setattr(colbert_encoder, "ensure_loaded", lambda: True)
        monkeypatch.setattr(colbert_encoder, "prefix_tokens_available", lambda: False)
        monkeypatch.setattr(colbert_encoder, "encode", fake_encode)

        rerank_snippets("q", [{"snippet": "a"}], top_k=1)
        assert set(calls) == {colbert_encoder.ROLE_NONE}


class TestTokenCaps:
    """Caps come from the checkpoint's declared config, not a local constant."""

    @staticmethod
    def _point_at_config(monkeypatch, tmp_path, **cfg):
        import src.tools.web.colbert_reranker as cr

        (tmp_path / "onnx_config.json").write_text(json.dumps(cfg))
        monkeypatch.setattr(colbert_encoder, "_MODEL_DIR", tmp_path)
        monkeypatch.setattr(colbert_encoder, "_declared_lengths_cache", {})
        monkeypatch.delenv(cr._DOC_TOKEN_BUDGET_ENV, raising=False)

    def test_query_cap_is_the_model_declared_value(self, monkeypatch, tmp_path, stub_encoder):
        """LateOn declares query_length 32; the old code hardcoded GTE's 48."""
        self._point_at_config(monkeypatch, tmp_path, query_length=32, document_length=300)

        rerank_snippets("q", [{"snippet": "a"}], top_k=1)

        query_cap = stub_encoder[0][1]
        assert query_cap == 32 == colbert_encoder.max_query_tokens()

    def test_query_cap_tracks_a_different_checkpoint(self, monkeypatch, tmp_path, stub_encoder):
        """Same code, GTE's declaration → 48. The cap is data, not a constant."""
        self._point_at_config(monkeypatch, tmp_path, query_length=48, document_length=300)

        rerank_snippets("q", [{"snippet": "a"}], top_k=1)

        assert stub_encoder[0][1] == 48

    def test_doc_cap_never_exceeds_the_declared_document_length(
        self, monkeypatch, tmp_path, stub_encoder
    ):
        """A model declaring a SHORT document_length clamps the snippet budget."""
        self._point_at_config(monkeypatch, tmp_path, query_length=32, document_length=24)

        rerank_snippets("q", [{"snippet": "a"}], top_k=1)

        assert stub_encoder[1][1] == 24

    def test_doc_budget_may_spend_fewer_tokens_than_declared(
        self, monkeypatch, tmp_path, stub_encoder
    ):
        """Snippets are short; obeying document_length=300 would pad for nothing."""
        import src.tools.web.colbert_reranker as cr

        self._point_at_config(monkeypatch, tmp_path, query_length=32, document_length=300)

        rerank_snippets("q", [{"snippet": "a"}], top_k=1)

        assert stub_encoder[1][1] == cr._SNIPPET_DOC_TOKEN_BUDGET < 300

    def test_doc_budget_env_override_is_still_clamped(
        self, monkeypatch, tmp_path, stub_encoder
    ):
        import src.tools.web.colbert_reranker as cr

        self._point_at_config(monkeypatch, tmp_path, query_length=32, document_length=96)
        monkeypatch.setenv(cr._DOC_TOKEN_BUDGET_ENV, "512")

        rerank_snippets("q", [{"snippet": "a"}], top_k=1)

        assert stub_encoder[1][1] == 96

    @pytest.mark.parametrize("bad", ["not-a-number", "0", "-4", ""])
    def test_invalid_doc_budget_override_falls_back(self, monkeypatch, bad):
        import src.tools.web.colbert_reranker as cr

        monkeypatch.setenv(cr._DOC_TOKEN_BUDGET_ENV, bad)
        assert cr._doc_token_budget() == cr._SNIPPET_DOC_TOKEN_BUDGET

    def test_unreadable_config_falls_back_without_raising(self, monkeypatch, tmp_path):
        """No config at all → documented fallbacks, not a crash."""
        monkeypatch.setattr(colbert_encoder, "_MODEL_DIR", tmp_path)
        monkeypatch.setattr(colbert_encoder, "_declared_lengths_cache", {})
        assert colbert_encoder.max_query_tokens() == colbert_encoder._FALLBACK_QUERY_TOKENS
        assert (
            colbert_encoder.max_document_tokens()
            == colbert_encoder._FALLBACK_DOCUMENT_TOKENS
        )

    def test_absurdly_small_declared_cap_is_treated_as_unstated(self, monkeypatch, tmp_path):
        """A cap that cannot hold a prefix plus text is a bad declaration."""
        (tmp_path / "onnx_config.json").write_text(json.dumps({"query_length": 1}))
        monkeypatch.setattr(colbert_encoder, "_MODEL_DIR", tmp_path)
        monkeypatch.setattr(colbert_encoder, "_declared_lengths_cache", {})
        assert colbert_encoder.max_query_tokens() == colbert_encoder._FALLBACK_QUERY_TOKENS


class TestNoDuplicatedEncoder:
    """Static guard: the private encoder copy must not come back.

    The web path drifted for a month because it owned a second implementation
    that no KB-side fix ever reached. A behavioural test cannot catch the
    reintroduction of a parallel path; this can.
    """

    def test_module_defines_no_private_encode(self):
        import src.tools.web.colbert_reranker as cr

        assert not hasattr(cr, "_encode"), (
            "colbert_reranker._encode is back. Encoding belongs to "
            "src.retrieval.colbert_encoder, which is where the [Q]/[D] role, "
            "token_type_ids, and do_lower_case handling live."
        )
        assert not hasattr(cr, "_ensure_loaded"), (
            "colbert_reranker owns a model load again; use "
            "colbert_encoder.ensure_loaded() so there is ONE session."
        )
        for attr in ("_session", "_tokenizer"):
            assert not hasattr(cr, attr), f"colbert_reranker._{attr} is a second singleton"

    def test_module_hardcodes_no_query_token_cap(self):
        import src.tools.web.colbert_reranker as cr

        assert not hasattr(cr, "_MAX_QUERY_TOKENS"), (
            "_MAX_QUERY_TOKENS was 48 (GTE's number) while the LateOn slot "
            "declares 32. Use colbert_encoder.max_query_tokens()."
        )
        assert not hasattr(cr, "_MAX_DOC_TOKENS")

        # Assert on the resolver's BODY, not the module text: a comment
        # mentioning the accessor would satisfy a whole-module substring check.
        resolver = inspect.getsource(cr._token_caps)
        assert "colbert_encoder.max_query_tokens()" in resolver
        assert "colbert_encoder.max_document_tokens()" in resolver
        assert not any(
            token.isdigit() and int(token) > 8
            for token in resolver.replace("(", " ").replace(")", " ").split()
        ), f"a literal token cap crept back into _token_caps:\n{resolver}"

    def test_module_does_not_import_onnxruntime_or_tokenizers(self):
        import src.tools.web.colbert_reranker as cr

        source = inspect.getsource(cr)
        for dep in ("onnxruntime", "from tokenizers"):
            assert dep not in source, (
                f"colbert_reranker references {dep}; model plumbing belongs to "
                "the shared encoder."
            )


class TestIsAvailable:
    """Test model availability check."""

    def test_is_available_delegates_to_the_shared_encoder(self, monkeypatch):
        """Loadability, not file presence — and one load, not two."""
        monkeypatch.setattr(colbert_encoder, "ensure_loaded", lambda: True)
        assert is_available() is True
        monkeypatch.setattr(colbert_encoder, "ensure_loaded", lambda: False)
        assert is_available() is False


class TestModelPathOverride:
    """Test ColBERT reranker model-path slot selection.

    The selector now lives in the shared encoder (it owns the session), but
    importing the reranker must re-resolve it — that is how
    scripts/benchmark/bench_colbert_rerank.py switches slots.
    """

    @staticmethod
    def _reload_unloaded(monkeypatch):
        """Reload the reranker with no session loaded, so a re-point is clean."""
        import importlib

        import src.tools.web.colbert_reranker as cr

        monkeypatch.setattr(colbert_encoder, "_session", None)
        monkeypatch.setattr(colbert_encoder, "_tokenizer", None)
        return importlib.reload(cr)

    def test_default_points_to_gte_moderncolbert(self, monkeypatch):
        """With no env var, module resolves to the GTE-ModernColBERT-v1 directory."""
        monkeypatch.delenv("LATEON_MODEL_PATH", raising=False)
        monkeypatch.delenv("REASON_MXBAI_MODEL_PATH", raising=False)
        cr = self._reload_unloaded(monkeypatch)
        assert str(cr._MODEL_DIR) == "/mnt/raid0/llm/models/gte-moderncolbert-v1-onnx"
        assert cr._MODEL_SLOT == "gte_moderncolbert"
        assert cr._MODEL_PATH.name == "model_int8.onnx"

    def test_env_var_overrides_to_lateon(self, monkeypatch):
        """LATEON_MODEL_PATH redirects the module-level constants."""
        monkeypatch.setenv("LATEON_MODEL_PATH", "/mnt/raid0/llm/models/lateon-onnx-int8")
        monkeypatch.delenv("REASON_MXBAI_MODEL_PATH", raising=False)
        cr = self._reload_unloaded(monkeypatch)
        assert str(cr._MODEL_DIR) == "/mnt/raid0/llm/models/lateon-onnx-int8"
        assert cr._MODEL_SLOT == "lateon"
        assert cr._MODEL_PATH == cr._MODEL_DIR / "model_int8.onnx"
        assert cr._TOKENIZER_PATH == cr._MODEL_DIR / "tokenizer.json"
        # Restore default for subsequent tests.
        monkeypatch.delenv("LATEON_MODEL_PATH", raising=False)
        self._reload_unloaded(monkeypatch)

    def test_reason_mxbai_env_var_selects_fallback_slot(self, monkeypatch):
        """REASON_MXBAI_MODEL_PATH redirects when LateOn is unset."""
        monkeypatch.delenv("LATEON_MODEL_PATH", raising=False)
        monkeypatch.setenv(
            "REASON_MXBAI_MODEL_PATH",
            "/mnt/raid0/llm/models/reason-mxbai-colbert-v0-32m-onnx-int8",
        )
        cr = self._reload_unloaded(monkeypatch)
        assert str(cr._MODEL_DIR) == (
            "/mnt/raid0/llm/models/reason-mxbai-colbert-v0-32m-onnx-int8"
        )
        assert cr._MODEL_SLOT == "reason_mxbai"
        assert cr._MODEL_PATH == cr._MODEL_DIR / "model_int8.onnx"
        assert cr._TOKENIZER_PATH == cr._MODEL_DIR / "tokenizer.json"
        monkeypatch.delenv("REASON_MXBAI_MODEL_PATH", raising=False)
        self._reload_unloaded(monkeypatch)

    def test_lateon_precedes_reason_mxbai_when_both_are_set(self, monkeypatch):
        """LateOn remains the primary slot when both overrides are configured."""
        monkeypatch.setenv("LATEON_MODEL_PATH", "/mnt/raid0/llm/models/lateon-onnx-int8")
        monkeypatch.setenv(
            "REASON_MXBAI_MODEL_PATH",
            "/mnt/raid0/llm/models/reason-mxbai-colbert-v0-32m-onnx-int8",
        )
        cr = self._reload_unloaded(monkeypatch)
        assert str(cr._MODEL_DIR) == "/mnt/raid0/llm/models/lateon-onnx-int8"
        assert cr._MODEL_SLOT == "lateon"
        monkeypatch.delenv("LATEON_MODEL_PATH", raising=False)
        monkeypatch.delenv("REASON_MXBAI_MODEL_PATH", raising=False)
        self._reload_unloaded(monkeypatch)

    def test_reranker_import_repoints_the_shared_encoder_too(self, monkeypatch):
        """A reload must move the ENCODER, not just the reranker's copies.

        Otherwise the bench harness reports one slot and scores another.
        """
        monkeypatch.setenv("LATEON_MODEL_PATH", "/mnt/raid0/llm/models/lateon-onnx-int8")
        monkeypatch.delenv("REASON_MXBAI_MODEL_PATH", raising=False)
        cr = self._reload_unloaded(monkeypatch)
        assert cr._MODEL_DIR == colbert_encoder._MODEL_DIR
        assert cr._MODEL_PATH == colbert_encoder._MODEL_PATH
        monkeypatch.delenv("LATEON_MODEL_PATH", raising=False)
        self._reload_unloaded(monkeypatch)

    def test_repointing_drops_a_session_loaded_for_another_model(self, monkeypatch):
        """A stale session would serve the OLD checkpoint under the new name."""
        monkeypatch.delenv("LATEON_MODEL_PATH", raising=False)
        monkeypatch.delenv("REASON_MXBAI_MODEL_PATH", raising=False)
        self._reload_unloaded(monkeypatch)

        monkeypatch.setattr(colbert_encoder, "_session", object())
        monkeypatch.setattr(colbert_encoder, "_tokenizer", object())
        monkeypatch.setenv("LATEON_MODEL_PATH", "/mnt/raid0/llm/models/lateon-onnx-int8")

        colbert_encoder.refresh_model_dir()
        assert colbert_encoder._session is None
        assert colbert_encoder._tokenizer is None

        monkeypatch.delenv("LATEON_MODEL_PATH", raising=False)
        self._reload_unloaded(monkeypatch)


class TestOnnxThreadBound:
    """The reranker's historical thread knob must still reach the ONNX session.

    Regression guard for the 2026-08-12 measurement: ORT's default pool (one
    thread per visible core, 192 here) is both slower and far noisier than a
    small bound for this 1+N single-row forward-pass workload, and it spins up
    192 threads per rerank call on a shared host. The session is now the shared
    encoder's, so the guard is that COLBERT_RERANK_ONNX_THREADS was not silently
    dropped when the session moved.
    """

    @pytest.mark.parametrize("visible_cpu_count", [1, 8, 16, 192])
    def test_shared_default_is_deterministic_across_visible_cpu_counts(
        self, monkeypatch, visible_cpu_count
    ):
        import os

        monkeypatch.setattr(os, "cpu_count", lambda: visible_cpu_count)
        monkeypatch.delenv("COLBERT_ENCODE_ONNX_THREADS", raising=False)
        monkeypatch.delenv("COLBERT_RERANK_ONNX_THREADS", raising=False)
        assert colbert_encoder._DEFAULT_ONNX_THREADS == 8
        assert colbert_encoder._onnx_threads() == 8

    def test_rerank_env_knob_is_honoured_as_an_alias(self, monkeypatch):
        monkeypatch.delenv("COLBERT_ENCODE_ONNX_THREADS", raising=False)
        monkeypatch.setenv("COLBERT_RERANK_ONNX_THREADS", "3")
        assert colbert_encoder._onnx_threads() == 3

    def test_encode_knob_wins_over_the_alias(self, monkeypatch):
        monkeypatch.setenv("COLBERT_ENCODE_ONNX_THREADS", "5")
        monkeypatch.setenv("COLBERT_RERANK_ONNX_THREADS", "3")
        assert colbert_encoder._onnx_threads() == 5

    @pytest.mark.parametrize("bad", ["not-a-number", "0", "-4", ""])
    def test_invalid_override_falls_back_to_default(self, monkeypatch, bad):
        """Garbage, zero and negative overrides never reach ORT."""
        monkeypatch.delenv("COLBERT_ENCODE_ONNX_THREADS", raising=False)
        monkeypatch.setenv("COLBERT_RERANK_ONNX_THREADS", bad)
        assert colbert_encoder._onnx_threads() == colbert_encoder._DEFAULT_ONNX_THREADS
