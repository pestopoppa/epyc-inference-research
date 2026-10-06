"""Shared ColBERT encoder primitives (ONNX Runtime + MaxSim).

Exposes corpus-agnostic encode/maxsim/ensure_loaded primitives so multiple
consumers (web_research reranker, internal KB-RAG) reuse one model load
and one tokenizer.

Model path resolution is a THREE-SLOT selector, owned here because both
consumers must agree on which checkpoint is live (this module holds the single
ONNX session):

1. `LATEON_MODEL_PATH`  — LightOn LateOn primary slot (intake-430, +2.55 pp
   BEIR vs GTE-ModernColBERT-v1; same ModernBERT backbone/output width).
2. `REASON_MXBAI_MODEL_PATH` — Reason-mxbai 32M natural-language
   edge-reasoning fallback (intake-453; A/B gated on AR-3 Package D).
3. Default GTE-ModernColBERT-v1 ONNX INT8 (128-dim per-token, ~144 MB).

Public API (corpus-agnostic, max-token configurable per call):
    is_available() -> bool
    ensure_loaded() -> bool
    encode(text: str, max_tokens: int, *, role: str) -> np.ndarray | None
    maxsim(query_emb, doc_emb) -> float
    max_query_tokens() / max_document_tokens() -> int   (model-DECLARED caps)
    resolve_model_dir() / refresh_model_dir() -> (Path, slot)

Token caps are model-declared, never hardcoded by a caller
-----------------------------------------------------------
`query_length` / `document_length` in the checkpoint's own config are the caps
the model was configured for, and they DIFFER per slot: GTE declares 48/300,
LateOn 32/300, Reason-mxbai 256/2048. A caller that hardcodes one model's
number silently feeds another model longer queries than it was set up for, so
ask `max_query_tokens()` / `max_document_tokens()` instead. A caller may spend
FEWER tokens than declared (a latency budget is a legitimate caller choice);
exceeding the declared cap is not.

ColBERT role prefixes (OP-24, 2026-08-12)
-----------------------------------------
Both GTE-ModernColBERT-v1 and the LateOn candidate declare
``query_prefix "[Q] "`` / ``document_prefix "[D] "`` in
``config_sentence_transformers.json``, backed by DEDICATED TRAINED token ids
(50368 / 50369 for the deployed model) that pylate inserts at index 1, right
after ``[CLS]``. Encoding without them is off-distribution: measured on the
reference model with no ONNX in the loop, the missing prefix moves MaxSim by
max |delta| 1.63e-01 and flips top-1 on 37.5% of queries — roughly 25x more
perturbing than the INT8 quantization we already accept (6.60e-03, top-1
agreement 100%).

`role` is therefore REQUIRED and keyword-only. There is deliberately no
default: a silent default is exactly how the prefix-less path survived
undetected, so every call site must state its intent. `ROLE_NONE` is the
explicit legacy escape hatch for stores built before this change — it is a
choice a caller makes visibly, not a fallback.

Prepending the prefix STRING is equivalent to pylate's id insertion only when
the prefix ENCODES to exactly one token; `ensure_loaded()` verifies that by
encoding it and refuses the prefixed roles otherwise.

The distinction is load-bearing and was wrong here until 2026-08-22. The check
used to be `token_to_id(prefix) is not None`, which reads the BASE vocabulary.
For a prefix like "[unused0]" that always answers yes, including on a tokenizer
that never promoted the string into `added_tokens` -- and only the added-token
trie splits a literal before WordPiece runs, so `encode()` then produced
['[', 'unused', '##0', ']'] with the guard satisfied. The upstream answerai
repository ships that exact pairing, so "just copy the missing config across"
built a silently wrong index. "[Q] " / "[D] " were never the dangerous case:
`token_to_id` returns None for them and the loader correctly refused.

ONNX session and tokenizer are module-level singletons, lazy-loaded on first
call. ONNX inference is thread-safe for prediction.

Per handoffs/active/internal-kb-rag.md K1.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ── Model path resolution: the three-slot selector (see module docstring) ────
DEFAULT_MODEL_DIR = Path("/mnt/raid0/llm/models/gte-moderncolbert-v1-onnx")

SLOT_LATEON = "lateon"
SLOT_REASON_MXBAI = "reason_mxbai"
SLOT_DEFAULT = "gte_moderncolbert"

LATEON_ENV = "LATEON_MODEL_PATH"
REASON_MXBAI_ENV = "REASON_MXBAI_MODEL_PATH"


def resolve_model_dir() -> "tuple[Path, str]":
    """Resolve (model_dir, slot_name) from the environment. Pure; reads env only."""
    if lateon_path := os.environ.get(LATEON_ENV):
        return Path(lateon_path), SLOT_LATEON
    if reason_path := os.environ.get(REASON_MXBAI_ENV):
        return Path(reason_path), SLOT_REASON_MXBAI
    return DEFAULT_MODEL_DIR, SLOT_DEFAULT


_MODEL_DIR, _MODEL_SLOT = resolve_model_dir()
_MODEL_PATH = _MODEL_DIR / "model_int8.onnx"
_TOKENIZER_PATH = _MODEL_DIR / "tokenizer.json"

# ONNX Runtime intra-op thread bound.
#
# `encode()` runs ONE single-row forward pass over <=max_tokens padded tokens, so
# the parallel work per call is tiny and ORT's default pool (one thread per visible
# core) oversubscribes badly on this 192-thread host. Measured 2026-08-12 on EPYC
# 9655, model_int8.onnx, batch=1, 20 calls x 3 interleaved rounds, medians in ms:
#
#     2t=20.54  4t=17.34  8t=17.88  16t=18.58  32t=23.92  192t=33.94
#
# Unbounded costs 1.96x. 4 measured nominally best, but 8 is within 3% — inside
# shared-host run-to-run variance — and matches the sibling reranker, which has the
# same single-row shape. Chosen for shape-consistency over a third magic number.
#
# Do NOT copy this value to a BATCHED consumer: cross_encoder.score_pairs() feeds N
# rows in one run() and measures best at 16 (40% faster than 8 at batch=50). The
# optimum is call-shape dependent, so measure before reusing.
_DEFAULT_ONNX_THREADS = 8


# `COLBERT_RERANK_ONNX_THREADS` is the web reranker's historical knob. That module
# used to own its own ONNX session; it now shares this one, so the knob is honoured
# here as a deprecated alias rather than silently dropped. Both were measured at 8.
_ONNX_THREADS_ENV = "COLBERT_ENCODE_ONNX_THREADS"
_ONNX_THREADS_ENV_ALIASES = ("COLBERT_RERANK_ONNX_THREADS",)


def _onnx_threads() -> int:
    """Resolve the ONNX intra-op thread count (env-overridable, positive int)."""
    raw = os.environ.get(_ONNX_THREADS_ENV)
    if not raw:
        for alias in _ONNX_THREADS_ENV_ALIASES:
            if alias_raw := os.environ.get(alias):
                raw = alias_raw
                break
    if not raw:
        return _DEFAULT_ONNX_THREADS
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "COLBERT_ENCODE_ONNX_THREADS=%r is not an integer; using %d",
            raw, _DEFAULT_ONNX_THREADS,
        )
        return _DEFAULT_ONNX_THREADS
    if value <= 0:
        logger.warning(
            "COLBERT_ENCODE_ONNX_THREADS=%d must be positive; using %d",
            value, _DEFAULT_ONNX_THREADS,
        )
        return _DEFAULT_ONNX_THREADS
    return value


# ── ColBERT role prefixes ────────────────────────────────────────────────────
#
# Fallbacks only; ensure_loaded() prefers whatever the model's own
# config_sentence_transformers.json declares, so a model swap cannot silently
# keep the wrong convention.
_FALLBACK_QUERY_PREFIX = "[Q] "
_FALLBACK_DOCUMENT_PREFIX = "[D] "

ROLE_QUERY = "query"
ROLE_DOCUMENT = "document"
ROLE_NONE = "none"
VALID_ROLES = (ROLE_QUERY, ROLE_DOCUMENT, ROLE_NONE)

# Identifier stamped alongside stored vectors so an index can always say which
# convention produced it. Bump when the prefix semantics change.
PREFIX_CONVENTION = "qd-v1"
# Convention of every embedding written before OP-24 (2026-08-12).
LEGACY_CONVENTION = "none"

# Module-level singletons (lazy-loaded).
_session = None
_tokenizer = None
_tokenizer_sha256: str | None = None
_query_prefix = _FALLBACK_QUERY_PREFIX
_document_prefix = _FALLBACK_DOCUMENT_PREFIX
_prefix_tokens_ok = False
# Input names the loaded graph declares, captured at load (K1). Empty until
# ensure_loaded() runs; encode() feeds exactly the inputs named here.
_input_names: tuple = ()
# Whether this checkpoint declares that text is lower-cased BEFORE tokenizing
# (K8). mxbai-edge-colbert does, and applies it OUTSIDE the tokenizer -- no
# `Lowercase` normalizer appears in tokenizer.json -- so nothing in this module
# would fold case and a cased query would be fed to a lower-case-trained model.
_do_lower_case = False


def _load_declared_prefixes(model_dir: Path) -> tuple[str, str]:
    """Read query/document prefixes from the model's sentence-transformers config."""
    cfg_path = model_dir / "config_sentence_transformers.json"
    try:
        import json

        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.warning(
            "ColBERT: could not read %s (%s); using fallback prefixes %r / %r",
            cfg_path, e, _FALLBACK_QUERY_PREFIX, _FALLBACK_DOCUMENT_PREFIX,
        )
        return _FALLBACK_QUERY_PREFIX, _FALLBACK_DOCUMENT_PREFIX
    q = cfg.get("query_prefix")
    d = cfg.get("document_prefix")
    if not isinstance(q, str) or not q or not isinstance(d, str) or not d:
        logger.warning(
            "ColBERT: %s declares no usable query/document prefix; using fallbacks",
            cfg_path,
        )
        return _FALLBACK_QUERY_PREFIX, _FALLBACK_DOCUMENT_PREFIX
    return q, d


def _load_declared_config(model_dir: Path) -> dict:
    """Merged serving contract for this checkpoint, `onnx_config.json` winning.

    Two filenames carry the same contract and neither is universal.
    `config_sentence_transformers.json` is what this module has always read;
    `onnx_config.json` is what the NextPlaid Rust reader REQUIRES and the only
    one a current `pylate-onnx-export` writes, so a freshly exported checkpoint
    has no file we would have read. Read both, prefer the format-mandated one,
    and let either supply keys the other omits.

    Returns {} when neither is readable — callers must treat every key as
    optional, because most checkpoints on disk declare only a subset.
    """
    import json

    merged: dict = {}
    for name in ("config_sentence_transformers.json", "onnx_config.json"):
        path = model_dir / name
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            merged.update(data)
    return merged


def _load_declared_prefix_ids(model_dir: Path) -> dict:
    """Declared `{"query": id, "document": id}`; a key is absent when unstated.

    Used to make the prefix probe assert IDENTITY, not just single-token-ness:
    a tokenizer can encode the prefix to one token that is nevertheless not the
    id the model was trained against.
    """
    cfg = _load_declared_config(model_dir)
    out: dict = {}
    for role, key in (("query", "query_prefix_id"), ("document", "document_prefix_id")):
        value = cfg.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            out[role] = value
    return out


def _prefix_encodes_to_one_token(tokenizer, prefix: str, declared_id: "int | None") -> bool:
    """True iff `prefix` ENCODES to exactly one token (optionally `declared_id`).

    `token_to_id(prefix)` is NOT this test and must not be used for it. It
    consults the base vocabulary, where a prefix like "[unused0]" is present by
    construction in every BERT checkpoint, so it answers "yes" on a tokenizer
    that never promoted the string into `added_tokens` -- and `encode()` then
    shreds it into ['[', 'unused', '##0', ']']. Only the added-token trie splits
    a literal before the WordPiece pre-tokenizer runs, and only ENCODING
    observes the trie. The upstream answerai repository ships exactly that
    combination (a PyLate-shaped config beside a tokenizer carrying 5 added
    tokens, not 7), so the failure is reachable from real published artifacts
    and it is SILENT: the guard passes, the index builds, the vectors are wrong.

    Padding and truncation are cleared for the probe because `encode()` sets
    them globally on this shared tokenizer; a padded probe would return
    max_tokens ids and reject every prefix.
    """
    try:
        tokenizer.no_padding()
        tokenizer.no_truncation()
        ids = tokenizer.encode(prefix, add_special_tokens=False).ids
    except Exception as e:  # noqa: BLE001 — a guard that raises must not pass
        logger.error("ColBERT: prefix probe for %r raised %s; treating as unusable", prefix, e)
        return False
    if len(ids) != 1:
        return False
    return declared_id is None or ids[0] == declared_id


def refresh_model_dir() -> "tuple[Path, str]":
    """Re-read the slot env vars and re-point the module at the resolved model.

    Needed because the slot selector is read at IMPORT time but consumers
    (`scripts/benchmark/bench_colbert_rerank.py`) switch slots by setting an env
    var and reloading a module. A loaded session for a DIFFERENT directory is
    dropped and logged — keeping it would serve the old checkpoint while every
    reported path said otherwise, which is the failure mode this exists to stop.
    """
    global _MODEL_DIR, _MODEL_SLOT, _MODEL_PATH, _TOKENIZER_PATH
    global _session, _tokenizer, _prefix_tokens_ok, _input_names, _do_lower_case
    global _count_tokenizer, _tokenizer_sha256

    model_dir, slot = resolve_model_dir()
    if model_dir == _MODEL_DIR:
        return _MODEL_DIR, _MODEL_SLOT

    _tokenizer_sha256 = None

    if _session is not None or _tokenizer is not None:
        logger.warning(
            "ColBERT encoder re-pointed from %s to %s (slot %s); dropping the "
            "loaded session so the next encode uses the model now configured.",
            _MODEL_DIR, model_dir, slot,
        )
        _session = None
        _tokenizer = None
        _prefix_tokens_ok = False
        _input_names = ()
        _do_lower_case = False
        _count_tokenizer = None

    _MODEL_DIR, _MODEL_SLOT = model_dir, slot
    _MODEL_PATH = _MODEL_DIR / "model_int8.onnx"
    _TOKENIZER_PATH = _MODEL_DIR / "tokenizer.json"
    return _MODEL_DIR, _MODEL_SLOT


# ── Model-declared token caps ────────────────────────────────────────────────
#
# Used only when the checkpoint declares neither `query_length` nor
# `document_length` (i.e. no readable config at all). These are the values the
# code hardcoded before this became model-derived, so an unreadable config
# preserves historical behaviour rather than inventing a new number.
_FALLBACK_QUERY_TOKENS = 48
_FALLBACK_DOCUMENT_TOKENS = 256
# A declared cap below this cannot hold a prefix token plus useful text, so it is
# treated as unstated rather than obeyed.
_MIN_DECLARED_TOKENS = 8

_declared_lengths_cache: dict = {}


def _declared_lengths(model_dir: Path) -> "tuple[int | None, int | None]":
    """(query_length, document_length) as DECLARED by the checkpoint, or None."""
    key = str(model_dir)
    if key in _declared_lengths_cache:
        return _declared_lengths_cache[key]
    cfg = _load_declared_config(model_dir)
    values = []
    for cfg_key in ("query_length", "document_length"):
        value = cfg.get(cfg_key)
        ok = (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value >= _MIN_DECLARED_TOKENS
        )
        values.append(value if ok else None)
    out = (values[0], values[1])
    _declared_lengths_cache[key] = out
    return out


def max_query_tokens() -> int:
    """Query truncation cap declared by the LIVE checkpoint (`query_length`)."""
    declared, _ = _declared_lengths(_MODEL_DIR)
    if declared is None:
        logger.warning(
            "ColBERT: %s declares no usable query_length; using fallback %d",
            _MODEL_DIR, _FALLBACK_QUERY_TOKENS,
        )
        return _FALLBACK_QUERY_TOKENS
    return declared


def max_document_tokens() -> int:
    """Document truncation cap declared by the LIVE checkpoint (`document_length`)."""
    _, declared = _declared_lengths(_MODEL_DIR)
    if declared is None:
        logger.warning(
            "ColBERT: %s declares no usable document_length; using fallback %d",
            _MODEL_DIR, _FALLBACK_DOCUMENT_TOKENS,
        )
        return _FALLBACK_DOCUMENT_TOKENS
    return declared


def prefix_for_role(role: str) -> str:
    """Return the literal prefix string for `role` ("" for ROLE_NONE)."""
    if role == ROLE_QUERY:
        return _query_prefix
    if role == ROLE_DOCUMENT:
        return _document_prefix
    if role == ROLE_NONE:
        return ""
    raise ValueError(f"unknown ColBERT role {role!r}; expected one of {VALID_ROLES}")


def prefix_tokens_available() -> bool:
    """True iff the loaded tokenizer maps both prefixes to single trained tokens."""
    return _prefix_tokens_ok


def is_available() -> bool:
    """Return True iff model files exist on disk. Does not load."""
    return _MODEL_PATH.exists() and _TOKENIZER_PATH.exists()


def ensure_loaded() -> bool:
    """Lazily load ONNX session + tokenizer. Returns True on success.

    Subsequent calls are no-ops when already loaded. Returns False if
    dependencies are missing or model files cannot be opened.
    """
    global _session, _tokenizer, _query_prefix, _document_prefix, _prefix_tokens_ok
    global _input_names, _do_lower_case, _count_tokenizer, _tokenizer_sha256

    if _session is not None and _tokenizer is not None:
        return True
    _count_tokenizer = None
    _tokenizer_sha256 = None

    if not is_available():
        logger.warning("ColBERT ONNX model not found at %s", _MODEL_PATH)
        return False

    try:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        start = time.perf_counter()
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = _onnx_threads()
        sess_options.inter_op_num_threads = 1
        _session = ort.InferenceSession(
            str(_MODEL_PATH),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        # Parse and fingerprint the same byte buffer: the file may change while
        # this singleton stays loaded, so hashing its path alone is insufficient.
        tokenizer_bytes = _TOKENIZER_PATH.read_bytes()
        _tokenizer = Tokenizer.from_str(tokenizer_bytes.decode("utf-8"))
        _tokenizer_sha256 = hashlib.sha256(tokenizer_bytes).hexdigest()

        # K1: which inputs this graph actually declares. BERT-family
        # late-interaction exports (answerai-colbert-small-v1, ColBERTv2,
        # Jina-ColBERT-v2) declare a third input `token_type_ids` with NO
        # initializer default, so a hardcoded two-input feed raises
        # InvalidArgument and the encoder returns None for every text. Pattern
        # borrowed from cross_encoder.py:148, which already does this.
        _input_names = tuple(i.name for i in _session.get_inputs())

        _query_prefix, _document_prefix = _load_declared_prefixes(_MODEL_DIR)
        declared = _load_declared_prefix_ids(_MODEL_DIR)
        _do_lower_case = bool(_load_declared_config(_MODEL_DIR).get("do_lower_case", False))
        q_id = _tokenizer.token_to_id(_query_prefix)
        d_id = _tokenizer.token_to_id(_document_prefix)
        # K6: encode-round-trip, NOT base-vocab membership. See
        # _prefix_encodes_to_one_token for why token_to_id cannot answer this.
        _prefix_tokens_ok = (
            _prefix_encodes_to_one_token(_tokenizer, _query_prefix, declared.get("query"))
            and _prefix_encodes_to_one_token(_tokenizer, _document_prefix, declared.get("document"))
        )
        if not _prefix_tokens_ok:
            logger.error(
                "ColBERT: prefixes %r/%r do not ENCODE to single tokens in %s "
                "(base-vocab ids %r/%r, declared ids %r/%r) — prefixed roles will be "
                "refused; only ROLE_NONE can be encoded. A non-None base-vocab id here "
                "with a failing probe is the silent-corruption case: the config declares "
                "a prefix the tokenizer never promoted into added_tokens.",
                _query_prefix, _document_prefix, _TOKENIZER_PATH, q_id, d_id,
                declared.get("query"), declared.get("document"),
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "ColBERT encoder loaded: %s (%.0fms), prefixes %r=%r %r=%r",
            _MODEL_PATH.name,
            elapsed_ms,
            _query_prefix, q_id, _document_prefix, d_id,
        )
        return True
    except ImportError as e:
        _session = None
        _tokenizer = None
        _tokenizer_sha256 = None
        logger.warning("ColBERT encoder dependencies missing: %s", e)
        return False
    except Exception as e:  # noqa: BLE001 — defensive; caller checks return.
        _session = None
        _tokenizer = None
        _tokenizer_sha256 = None
        logger.error("ColBERT encoder load failed: %s", e)
        return False


def encode(text: str, max_tokens: int, *, role: str) -> np.ndarray | None:
    """Encode text into per-token L2-normalized ColBERT embeddings.

    Args:
        text: Input text.
        max_tokens: Max tokens; tokenizer truncates beyond. The role prefix
            occupies one of them, matching pylate.
        role: REQUIRED. `ROLE_QUERY`, `ROLE_DOCUMENT`, or `ROLE_NONE`.
            Queries and documents MUST use the same convention as the store
            they are compared against — see the module docstring. There is no
            default on purpose.

    Returns:
        Array shape (n_real_tokens, 128) or None on failure.

    Raises:
        ValueError: `role` is not one of `VALID_ROLES`, or a prefixed role was
            requested from a model whose tokenizer lacks the trained prefix
            tokens (encoding it would silently emit garbage sub-word tokens).
    """
    prefix = prefix_for_role(role)  # raises on an unknown role
    if prefix and not _prefix_tokens_ok:
        raise ValueError(
            f"ColBERT role {role!r} needs prefix {prefix!r}, but the loaded "
            f"tokenizer has no such token ({_TOKENIZER_PATH})"
        )

    if _session is None or _tokenizer is None:
        return None

    try:
        _tokenizer.enable_truncation(max_length=max_tokens)
        _tokenizer.enable_padding(length=max_tokens)
        # K8: fold case only when the checkpoint declares it, and only on the
        # TEXT -- never the prefix, which is a literal added token and would
        # stop matching the trie if lower-cased.
        body = text.lower() if _do_lower_case else text
        encoded = _tokenizer.encode(prefix + body)

        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

        # K1: feed exactly the inputs this graph declares. `token_type_ids` is
        # all-zeros for single-segment input, which is all we ever encode.
        feed = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in _input_names:
            feed["token_type_ids"] = np.zeros_like(input_ids)
        unsatisfied = [n for n in _input_names if n not in feed]
        if unsatisfied:
            # Loud, not silent: an unknown required input means this graph is
            # not one we can drive, and returning None here would present as an
            # empty index rather than a load error.
            raise RuntimeError(
                f"ColBERT graph declares input(s) {unsatisfied} that this encoder "
                f"does not supply (declared: {list(_input_names)})"
            )

        outputs = _session.run(None, feed)

        embeddings = outputs[0][0]  # (max_tokens, hidden_dim)
        mask = attention_mask[0]  # (max_tokens,)
        token_embeddings = embeddings[mask == 1]

        # L2 normalize.
        norms = np.linalg.norm(token_embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        token_embeddings = token_embeddings / norms
        return token_embeddings
    except Exception as e:  # noqa: BLE001
        # K7: warning, not debug. At debug level an ONNX InvalidArgument
        # ("Missing Input: token_type_ids") is invisible and the caller sees an
        # ordinary miss, so a model that CANNOT be encoded at all presents as an
        # empty index rather than a failure.
        logger.warning("ColBERT encode failed (%s): %s", type(e).__name__, e)
        return None


_count_tokenizer = None


def count_tokens(text: str, *, role: str) -> int | None:
    """UNTRUNCATED, unpadded token count of `text` as `encode()` would see it.

    Same prefix, same case folding, same special tokens as `encode()` — but on a
    private tokenizer copy with truncation and padding disabled. `encode()` sets
    `enable_truncation(max_length)` + `enable_padding(length)` on the shared
    tokenizer, so any count read from its output is always exactly the cap; a
    query-length instrument built on that would report a 0 % over-cap rate forever
    (KB-RAG H2). A copy is used, never the shared instance, so a concurrent
    `encode()` can never observe truncation switched off.

    Returns None when the encoder is not loaded or tokenization fails; the
    caller records nothing rather than a fabricated count.
    """
    global _count_tokenizer
    prefix = prefix_for_role(role)  # raises on an unknown role
    tok = _tokenizer
    if tok is None:
        return None
    try:
        if _count_tokenizer is None:
            from tokenizers import Tokenizer

            copy = Tokenizer.from_str(tok.to_str())
            copy.no_truncation()
            copy.no_padding()
            _count_tokenizer = copy
        body = text.lower() if _do_lower_case else text
        return len(_count_tokenizer.encode(prefix + body).ids)
    except Exception as e:  # noqa: BLE001 — instrumentation must never fail a query
        logger.debug("ColBERT count_tokens failed (%s): %s", type(e).__name__, e)
        return None


def maxsim(query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
    """ColBERT MaxSim score: avg over query tokens of max cosine to any doc token.

    Both inputs must be L2-normalized along the last axis (encode() does this).
    """
    sim_matrix = query_emb @ doc_emb.T  # (n_q, n_d)
    return float(sim_matrix.max(axis=1).mean())
