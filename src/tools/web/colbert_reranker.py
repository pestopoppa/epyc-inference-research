"""ColBERT snippet reranker (thin consumer of the shared ColBERT encoder).

Provides semantic reranking of search snippets via late-interaction MaxSim
scoring. All model loading, tokenization, role prefixing and MaxSim live in
`src.retrieval.colbert_encoder`; this module only decides WHAT to encode, with
WHICH role, and how to order the result.

Model slot selection (`LATEON_MODEL_PATH` / `REASON_MXBAI_MODEL_PATH` /
GTE-ModernColBERT-v1 default) is the shared encoder's, re-exported here as
`_MODEL_DIR` / `_MODEL_SLOT` / `_MODEL_PATH` / `_TOKENIZER_PATH` for the
benchmark harness and tests.

Why this module has no `_encode` of its own (PREFIX-1, 2026-09-14)
-----------------------------------------------------------------
It used to. That private copy predated the shared encoder and never received
the OP-24 / K1 / K6 / K8 fixes the KB side got in `fe55b228` and `4e5e84c0`:

- no `[Q]`/`[D]` role prefix — measured 25x more perturbing than the INT8
  quantization we accept (max |ΔMaxSim| 1.63e-01, top-1 flipped on 37.5% of
  queries) and enough to make any A/B a measurement of a broken encoder;
- a hardcoded two-input feed, so a BERT-family graph declaring
  `token_type_ids` returned None for every text (an empty rerank, logged at
  debug, presenting as an ordinary miss);
- no `do_lower_case` handling;
- `_MAX_QUERY_TOKENS = 48`, the GTE number, while the LateOn slot this file's
  own docstring points at declares `query_length: 32`.

Every one of those is fixed once, in the shared encoder, or not at all. The
prefix change is safe to make here with no re-embedding: unlike the KB there is
no stored web index, so query and document are always encoded together by the
same live code and move as one.

Model loaded lazily on first call, cached as a module-level singleton inside the
shared encoder. ONNX inference session is thread-safe for prediction.

Usage:
    from src.tools.web.colbert_reranker import rerank_snippets

    ranked = rerank_snippets(
        query="What causes aurora borealis?",
        snippets=[
            {"title": "Aurora", "snippet": "Northern lights are caused by..."},
            {"title": "Cooking", "snippet": "Best pasta recipes..."},
        ],
        top_k=3,
    )
    # ranked[0] = highest relevance snippet

Feature-gated: ORCHESTRATOR_WEB_RESEARCH_RERANK=1
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import numpy as np

from src.retrieval import colbert_encoder

logger = logging.getLogger(__name__)

# The slot selector lives in the shared encoder (it owns the single ONNX
# session, so it must be the one authority on which checkpoint is live).
# Re-read here at import so `importlib.reload(colbert_reranker)` after an env
# change re-points the encoder too — the pattern
# scripts/benchmark/bench_colbert_rerank.py uses to switch slots.
_MODEL_DIR, _MODEL_SLOT = colbert_encoder.refresh_model_dir()
_MODEL_PATH = colbert_encoder._MODEL_PATH
_TOKENIZER_PATH = colbert_encoder._TOKENIZER_PATH

# Snippet-side token budget.
#
# Token CAPS are the model's (`colbert_encoder.max_query_tokens()` /
# `max_document_tokens()`): the query cap is taken verbatim, because feeding a
# model a longer query than it declares is off-distribution and differs per slot
# (GTE 48, LateOn 32, Reason-mxbai 256).
#
# Documents are different: search snippets are two or three sentences, and the
# shared encoder pads every input to `max_tokens`, so obeying GTE's declared
# `document_length: 300` would pad each of N snippets to 300 tokens for no
# content gain. Spending FEWER tokens than declared is a caller's latency
# choice; the budget is therefore min(declared, this). 64 is the value the
# 2026-08-12 latency measurements on this path were taken at. Raise it with
# COLBERT_RERANK_MAX_DOC_TOKENS, which is still clamped by the declared cap.
_SNIPPET_DOC_TOKEN_BUDGET = 64
_DOC_TOKEN_BUDGET_ENV = "COLBERT_RERANK_MAX_DOC_TOKENS"

# One-shot warning latch for the prefix-less fallback.
_warned_no_prefix = False


def _doc_token_budget() -> int:
    """Snippet document budget (env-overridable, positive int)."""
    raw = os.environ.get(_DOC_TOKEN_BUDGET_ENV)
    if not raw:
        return _SNIPPET_DOC_TOKEN_BUDGET
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not an integer; using %d",
            _DOC_TOKEN_BUDGET_ENV, raw, _SNIPPET_DOC_TOKEN_BUDGET,
        )
        return _SNIPPET_DOC_TOKEN_BUDGET
    if value <= 0:
        logger.warning(
            "%s=%d must be positive; using %d",
            _DOC_TOKEN_BUDGET_ENV, value, _SNIPPET_DOC_TOKEN_BUDGET,
        )
        return _SNIPPET_DOC_TOKEN_BUDGET
    return value


def _token_caps() -> tuple[int, int]:
    """(query_cap, doc_cap): model-declared, doc side clamped to the budget."""
    query_cap = colbert_encoder.max_query_tokens()
    doc_cap = min(colbert_encoder.max_document_tokens(), _doc_token_budget())
    return query_cap, doc_cap


def _roles() -> tuple[str, str]:
    """(query_role, document_role) for this checkpoint.

    Prefixed roles when the tokenizer maps `[Q] `/`[D] ` to the trained single
    tokens, `ROLE_NONE` otherwise. Unlike the KB there is no stored web index to
    stay consistent with, so this is a free choice per call and both sides always
    move together — but a prefixed role against a tokenizer that lacks the token
    RAISES in the encoder, so a checkpoint without them must degrade, loudly and
    once, to the legacy prefix-free convention rather than fail every rerank.
    """
    global _warned_no_prefix
    if colbert_encoder.prefix_tokens_available():
        return colbert_encoder.ROLE_QUERY, colbert_encoder.ROLE_DOCUMENT
    if not _warned_no_prefix:
        _warned_no_prefix = True
        logger.warning(
            "ColBERT rerank: %s has no trained [Q]/[D] prefix tokens; scoring "
            "prefix-free (off-distribution, ~25x the perturbation of INT8). "
            "Any A/B run in this state measures the encoder, not the model.",
            colbert_encoder._MODEL_DIR,
        )
    return colbert_encoder.ROLE_NONE, colbert_encoder.ROLE_NONE


def _maxsim(query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
    """MaxSim score — thin alias for the shared implementation."""
    return colbert_encoder.maxsim(query_emb, doc_emb)


def _snippet_text(snippet_dict: dict[str, Any]) -> str:
    """Flatten a snippet dict into the text that gets encoded."""
    text = ""
    if "title" in snippet_dict:
        text += snippet_dict["title"] + ". "
    if "snippet" in snippet_dict:
        text += snippet_dict["snippet"]
    elif "body" in snippet_dict:
        text += snippet_dict["body"]
    return text


@colbert_encoder.state_transaction
def rerank_snippets(
    query: str,
    snippets: list[dict[str, Any]],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Rerank search snippets by ColBERT MaxSim relevance.

    Each snippet dict should have at least 'snippet' or 'title' keys.
    Returns snippets sorted by relevance, with 'rerank_score' added.

    If the model is not loaded or reranking fails, returns the original
    snippets unchanged (graceful degradation).

    Args:
        query: Search query.
        snippets: List of snippet dicts from web search.
        top_k: Number of top snippets to return.

    Returns:
        Top-k snippets sorted by relevance, with 'rerank_score' field.
    """
    if not snippets:
        return []

    if not colbert_encoder.ensure_loaded():
        logger.debug("ColBERT reranker not available, returning original order")
        return snippets[:top_k]

    start = time.perf_counter()

    query_role, doc_role = _roles()
    query_cap, doc_cap = _token_caps()

    query_emb = colbert_encoder.encode(query, query_cap, role=query_role)
    if query_emb is None:
        return snippets[:top_k]

    # Score each snippet
    scored = []
    for snippet_dict in snippets:
        text = _snippet_text(snippet_dict)

        # Skip snippets with no meaningful text (only punctuation/whitespace)
        cleaned = text.strip().strip(".")
        if not cleaned.strip():
            scored.append((snippet_dict, 0.0))
            continue

        doc_emb = colbert_encoder.encode(text, doc_cap, role=doc_role)
        if doc_emb is None:
            scored.append((snippet_dict, 0.0))
            continue

        score = _maxsim(query_emb, doc_emb)
        scored.append((snippet_dict, score))

    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "ColBERT rerank: %d snippets in %.0fms, roles=%s/%s, caps=%d/%d, scores=[%s]",
        len(snippets), elapsed_ms, query_role, doc_role, query_cap, doc_cap,
        ", ".join(f"{s:.3f}" for _, s in scored[:5]),
    )

    # Add scores and return top-k
    result = []
    for snippet_dict, score in scored[:top_k]:
        enriched = dict(snippet_dict)
        enriched["rerank_score"] = round(score, 4)
        result.append(enriched)

    return result


def is_available() -> bool:
    """Check if the reranker model is loadable, not merely present on disk."""
    return colbert_encoder.ensure_loaded()
