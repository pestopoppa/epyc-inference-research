"""Native candidate-scoring fast path for the typed decision plane (TD-1b/1c).

``run_typed_decisions_native`` answers a whole ``Question`` catalogue in ONE
grammar-constrained ``LLMPrimitives.llm_call``. In TD-1b it generated exactly
one token per question, in catalogue order, under a GBNF grammar whose
position ``i`` only accepted question ``i``'s candidate TOKEN IDS. TD-1c
changes the generation layout because the live measurement showed that bare
concatenation is NOT semantically equivalent to JSON mode (agreement 11/16,
native accuracy 68.8% vs JSON 91.7% on the same 16 questions, a false-bias
on noul positions):

TD-1c — why the answer cue is REPLAYED, not merely listed:
    A single completion has exactly one "next token" position, so a prompt
    that merely lists N questions with N answer cues cannot condition answer
    ``i`` on cue ``i`` — the model sees every cue before it emits anything.
    The live failure was exactly that: at position 1 (no prior emission) the
    model answered the first noul correctly, and positions 2..8 collapsed to
    ``false`` even when the JSON arm answered ``true``. TD-1c makes the
    decoder re-emit each question's cue as fixed token ids BETWEEN the answer
    slots::

        <cue 1 tokens> <answer 1 token> <cue 2 tokens> <answer 2 token> ...

    so every answer token is generated immediately after its own question's
    cue (question id + question text + ``Answer (one of: ...):``), which is
    the conditioning the JSON arm gets from writing one answer object per
    question. The cue tokens are obtained from the SAME tokenization seam the
    candidates use (``POST /tokenize`` by default) and are compiled into the
    grammar as exact-token terminals, so the layout is deterministic and the
    "one constrained token per question" readout is preserved.

TD-1d — cue styles (speed at parity):
    The TD-1c replay restored conditioning at the cost of ~300+ grammar-forced
    cue tokens for a 24-question catalogue (523 generated tokens, 2.08x vs
    JSON). ``CueStyle`` selects how much of that cue is replayed: ``FULL`` is
    the TD-1c layout; ``SHORT`` replays ``Q <id>: <first six words>``;
    ``ID_ONLY`` replays ``<id>: `` alone and is the default since TD-6 (cue
    sweep: 11.98x at 15/16 agreement, ``bench-cue-sweep-worker.json``; n=1
    and contested — the 2026-09-18 n=4 re-measurement gives 9.60x, 6.40x per
    decision, below the 10x bar, pending TD-1d.0). The
    numbered catalogue rendered by ``build_native_prompt`` is byte-identical
    across styles — the prompt carries the grounding, the cue only re-conditions
    the answer position — so a sweep varies exactly one thing: which cue tokens
    are replayed. The tokenizer bind, the exact-token grammar terminals, the
    probability slicing and every fail-closed contract are unchanged.

Probability semantics (investigated 2026-09-17 against the frozen v9 tree;
revised 2026-09-24, TD-1d.2, against the frozen v10 tree):
    ``post_sampling_probs=false`` was the default through TD-1d (this module
    never set it) and captures the PRE-sampler distribution:
    ``server-context.cpp`` sets ``need_pre_sample_logits = n_probs > 0 &&
    !post_sampling_probs`` and therefore disables backend sampling, so
    ``get_token_probabilities`` (``server-common.cpp``) softmaxes the raw
    full-vocab logits from ``llama_get_logits_ith``. The GBNF grammar mask is
    applied in the CPU sampler chain to a COPY, so ``top_logprobs`` never
    reflect it. Consequences: (a) the slice is raw model mass, and mass that
    landed outside the declared candidates is invisible in the runner's
    renormalized distribution — for a heavily grammar-narrowed position (e.g.
    a boolean or a 1-5 score digit against a 100k+-token vocabulary) this can
    mean NONE of the declared candidates appear in the raw top-k at all, even
    though (b) the emitted token is always the greedy argmax over the
    grammar-masked candidates. Confirmed live 2026-09-24 (TD-1d.2): the /v1
    lane's native:full/native:id_only arms got ``native_unknown_candidate``
    ("none of the declared candidate tokens appears in the captured top
    probabilities") on 15/16 and 16/16 supported positions respectively —
    ``_match_row_label`` (reads the row's OWN ``id``/``token``) matched every
    one of them, so the row's own emission was never in doubt; only the
    *coverage* of its sibling ``top_logprobs`` list was the problem. Reading
    ``tools/server/server-task.cpp``/``server-context.cpp`` at the exact
    frozen commit confirmed both ``/completion`` and ``/v1/chat/completions``
    build ``completion_probabilities``/``logprobs.content`` from the IDENTICAL
    ``completion_token_output::probs_vector_to_json`` — same shape, ``id``
    always present on both row and entries on this server — so this was never
    a ``/v1``-specific shape bug; it is the pre-existing raw-mass coverage gap
    biting harder on this workload than it did on 2026-09-17.

    This module now sets ``post_sampling_probs=True`` (threaded through
    ``LLMPrimitives.llm_call`` → ``InferenceRequest.post_sampling_probs`` →
    both backend payload builders in ``src/backends/llama_server.py``,
    opt-in and False everywhere else) so the server's top-k comes from
    ``populate_token_probs``'s ``post_sampling=true`` branch:
    ``common_sampler_get_candidates(slot.smpl.get(), true)``, i.e. candidates
    AFTER the full sampler chain including the grammar mask. Illegal tokens
    carry probability 0 there and the loop breaks on the first zero, so the
    reported top-k can only ever contain still-legal candidates — the
    coverage gap is closed by construction, not papered over by rescuing a
    single unconfirmed weight (a genuinely-uncovered capture still fails
    closed via ``REASON_NATIVE_UNKNOWN_CANDIDATE``, unchanged; see
    ``test_no_candidate_token_in_the_capture_is_failure_not_uniform``, which
    pins that fail-closed behavior and must never be "fixed" by fabricating a
    weight-of-one distribution for an uncorroborated token). ``prob``
    (linear, not log) replaces ``logprob`` in this mode — already-supported,
    pinned in ``test_post_sampling_top_probs_shape_is_accepted``.
    ``native_diagnostics`` still exposes the captured top-k live for
    debugging: it reports the raw top-k rows, the sliced mass, the mass
    outside the candidate set, and the argmax per question.

Natural noul surface forms:
    ``true``/``false`` are the JSON labels but not always the model's natural
    boolean answer tokens. For ``noul`` questions native mode ALSO binds the
    single-token surface forms ``yes``/``no`` (with and without a leading
    space) to the ``true``/``false`` labels, so a model that prefers ``Yes`` /
    ``No`` still contributes its probability mass to the right label. The
    grammar accepts them, the sliced weights SUM per label, and the ``Decision``
    probability keys stay ``"true"``/``"false"``. The cue echoes the declared
    labels only; yes/no are read as insurance against vocabulary mismatch.
    One consequence is pinned deliberately: ``Decision.value`` is the argmax of
    the SUMMED label distribution, so when a label's surface forms split the
    raw mass (e.g. ``false`` 0.15 versus ``true`` 0.10 + ``yes`` 0.10), the
    emitted token can map to a different label than the reported value.
    ``native_diagnostics`` exposes ``emitted_matches_argmax`` for exactly that
    audit; a real capture with the question's own cue in context should keep
    the two aligned.

TD-29 — single-token keys for multi-token closed sets:
    The 2026-09-29 champion-sidecar native run of the TD-4 tool-args pilot
    lost 6/18 cases, all to the ``p1``..``p4`` severity enum: each label is
    two tokens, so TD-1b excluded the question and argument assembly failed.
    A set whose labels are NOT all single tokens is now re-keyed with the
    TD-9 routing code alphabet (``A``..``Z``, ``0``..``9``; reused from
    ``routing_replay``): each label takes the next code whose bare or spaced
    form is exactly one token for the SERVED tokenizer (probed through the
    same ``/tokenize`` seam) and whose ids collide with no earlier key. The
    prompt renders ``A = p1 | B = p2 ...`` and asks for the key, the grammar
    binds the key ids, and the decision maps back: ``Decision.value`` is the
    original label and ``Decision.native_key`` the code chosen; the layout
    carries both. Gated on need: a set whose labels already bind is never
    re-keyed, so its prompt, grammar, layout and decisions are byte-identical
    to TD-1b. More labels than codes, or an alphabet exhausted by multi-token
    or colliding codes, falls back exactly as before
    (``native_unsupported_candidates``) with the reason in the detail.
    ``single_token_keys=False`` restores the TD-1b behavior outright.

Tokenizer seam:
    ``tokenize_fn: Callable[[str], Sequence[int] | None]`` is injectable on
    this runner and forwarded by ``runner.run_typed_decisions`` in native mode
    only. ``None`` (or an exception) from a call means "this text could not be
    tokenized" and fails the affected question closed. When not injected,
    ``_resolve_tokenize_fn`` derives the role's backend base URL from the
    primitives object and POSTs to ``{base_url}/tokenize`` via httpx (client
    pattern: ``src/backends/llama_server.py``).
    ``src/llm_primitives/tokenizer.py::LlamaTokenizer`` is deliberately NOT
    reused as the instrument: it returns counts only and silently falls back
    to a ``len(text) // 4`` heuristic on error, which cannot satisfy the
    exact-single-token-id contract (a heuristic count of 1 would fabricate
    eligibility). Its ``base_url`` is used as a last-resort resolution source.
    The same seam tokenizes each question's cue text; a cue that cannot be
    tokenized (or that tokenizes to nothing) fails that question closed.

Contract:
    * ONE call, ``len(native_questions)`` answer tokens plus the fixed cue
      tokens between them. ``n_tokens`` defaults to that exact total;
      temperature ``0.0`` and seed ``0``.
    * ``n_probs`` defaults to the largest per-question token-alternative count
      plus ``_N_PROBS_BUFFER``, capped at ``_MAX_N_PROBS`` (128 — the
      llama-server payload cap in ``src/backends/llama_server.py``). The
      buffer exists because the captured top-K is global (all tokens), not
      per-candidate.
    * At each ANSWER position the returned top-probability entries are matched
      to that question's candidate token ids (text fallback only for a whole
      row without ids), giving every declared candidate a weight (absent
      candidates weigh ``0.0``), then
      ``confidence.normalize_probabilities`` renormalizes. The argmax of the
      renormalized distribution is the ``Decision.value``;
      ``Decision.token_logprob`` is the raw model log-probability of that
      value's token (``None`` when the payload only carried linear probs).
      ``Decision.confidence`` uses ``choice_confidence`` for choice/noul and
      ``score_confidence`` for score, exactly like the JSON runner.
    * A position whose emitted token id/text is not one of the declared
      candidates, whose row is missing, or whose top-probability slice
      contains no declared candidate token yields a ``ParseFailure``
      (``native_unknown_candidate``) — never a default, and never a uniform
      fallback over an empty slice.
    * Questions that cannot be tokenized into single tokens — or whose
      candidates or cue cannot be tokenized — are NEVER forced into the
      native batch. Multi-token candidates are returned as
      ``ParseFailure(native_unsupported_candidates)``; an unresolvable or
      unresponsive tokenizer yields ``ParseFailure(native_tokenizer_unavailable)``
      for every affected question. Both are emitted alongside the native
      decisions so the caller can re-ask exactly those questions in JSON
      mode, which remains the correctness fallback. This module never falls
      back itself: a silent mode switch would make the receipt lie about how
      the answer was produced. With no resolvable tokenizer NOTHING is sent
      to the model — no tokenizer, no call, no guess.
    * Transport failures (``llm_call`` returning an ``"[ERROR: ...]"`` string)
      short-circuit to a single ``transport_error`` failure; the stale
      ``_last_inference_meta`` is deliberately NOT read on that path.

Layout side channel (for ``native_diagnostics``):
    Immediately before the call, ``run_typed_decisions_native`` attaches the
    exact per-position layout (question ids, candidate token bindings, cue
    lengths, answer row indices, the prompt, ``n_probs``) to the primitives
    object as ``_last_native_layout``. This is the same instance-level
    side-channel trade-off as ``_last_inference_meta`` and shares its
    concurrency caveat. ``native_diagnostics(result, primitives_snapshot)``
    accepts the primitives object (best), a mapping with
    ``{"native_layout": ..., "meta"/"completion_probabilities": ...}``, or a
    bare meta mapping (row-level output only — no question binding).

Row-shape note (production-consolidated-v9):
    ``/completion`` with ``n_probs`` set returns ``completion_probabilities``
    rows shaped ``{"id": int, "token": str, "bytes": [...], "logprob": float,
    "top_logprobs": [{"id", "token", "bytes", "logprob"}, ...]}``
    (``tools/server/server-task.cpp::probs_vector_to_json`` with
    ``post_sampling_probs=false``, the default). One row is captured per
    generated token, cues included; answer rows are addressed by their
    precomputed index. Older builds shipped the legacy
    ``{"content": str, "probs": [{"tok_str", "prob"}, ...]}`` shape and
    ``post_sampling_probs=true`` ships ``top_probs`` with linear ``prob`` and
    no ids; for a row without ``id`` the match falls back to the exact token
    text against the variant strings sent to ``/tokenize`` (see above).

Concurrency caveat:
    ``primitives._last_inference_meta`` and ``primitives._last_native_layout``
    are INSTANCE-level attributes, not request-scoped return values
    (``src/llm_primitives/inference.py``). This module reads them immediately
    after its single call, but a concurrent ``llm_call`` on the SAME
    primitives object can overwrite them. Native scoring therefore requires
    serialized use of one primitives object; sharing one across threads is
    unsupported.
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx

from src.typed_decisions.confidence import (
    choice_confidence,
    normalize_probabilities,
    score_confidence,
)
from src.typed_decisions.runner import REASON_TRANSPORT_ERROR, _validated_catalogue
from src.typed_decisions.types import (
    Decision,
    DecisionResult,
    ParseFailure,
    Question,
    QuestionKind,
)

__all__ = [
    "REASON_NATIVE_TOKENIZER_UNAVAILABLE",
    "REASON_NATIVE_UNKNOWN_CANDIDATE",
    "REASON_NATIVE_UNSUPPORTED_CANDIDATES",
    "CueStyle",
    "TokenizeFn",
    "build_native_prompt",
    "native_diagnostics",
    "run_typed_decisions_native",
    "run_typed_decisions_native_parallel",
]

logger = logging.getLogger(__name__)

# Text -> token ids, or ``None`` when the text could not be tokenized (the
# endpoint failed, or the response shape was unusable). ``None`` is never
# coerced into a count: the caller fails closed instead.
TokenizeFn = Callable[[str], Sequence[int] | None]


class CueStyle(str, Enum):
    """How much of a question's cue is replayed before its answer token (TD-1d).

    The cue is the only thing a style changes: the prompt built by
    ``build_native_prompt`` and every fail-closed contract are identical, so a
    sweep isolates the conditioning value of the replayed tokens.

    ``ID_ONLY`` — ``"\\n<id>: "``; the prompt's numbered catalogue carries the
    grounding. **Native default since TD-6**: the cue sweep measured 11.98x at
    15/16 agreement vs the JSON arm (``bench-cue-sweep-worker.json``; n=1,
    contested: 9.60x at n=4 on 2026-09-18, pending TD-1d.0).
    ``FULL``    — ``"\\nQ <id>: <text>\\nAnswer (one of: ...): "`` (TD-1c;
    selectable, still the cue a ``full`` request replays).
    ``SHORT``   — ``"\\nQ <id>: <first _SHORT_CUE_WORDS words>\\n"``.
    """

    FULL = "full"
    SHORT = "short"
    ID_ONLY = "id_only"


def _normalize_cue_style(cue_style: CueStyle | str) -> CueStyle:
    """Coerce a style value to ``CueStyle``; unknown values raise ``ValueError``.

    ``CueStyle`` is a ``str`` enum, so the plain value strings ("full", "short",
    "id_only") are accepted wherever the enum is.
    """
    if isinstance(cue_style, CueStyle):
        return cue_style
    try:
        return CueStyle(cue_style)
    except ValueError:
        expected = [style.value for style in CueStyle]
        raise ValueError(f"unknown cue style: {cue_style!r}; expected one of {expected}") from None


# Deterministic decode, identical policy to the JSON runner: temperature 0.0
# plus a pinned seed. The grammar removes most sampling freedom anyway (the
# token still comes from the model's masked distribution).
_DECODE_SEED = 0

# The captured top-K list is global over the vocabulary, so K must exceed the
# per-question token-alternative count for every declared candidate token to be
# captured; the buffer covers the near-miss tokens between candidates.
_N_PROBS_BUFFER = 4

# Mirrors the payload clamp in src/backends/llama_server.py
# (``payload["n_probs"] = min(128, int(request.n_probs))``).
_MAX_N_PROBS = 128

# /tokenize is a local, CPU-cheap endpoint; a short timeout keeps an
# unresponsive server from stalling the batch.
_TOKENIZE_TIMEOUT_S = 2.0

# SHORT cues replay the question id plus this many leading words of the text.
_SHORT_CUE_WORDS = 6

REASON_NATIVE_UNSUPPORTED_CANDIDATES = "native_unsupported_candidates"
REASON_NATIVE_UNKNOWN_CANDIDATE = "native_unknown_candidate"
REASON_NATIVE_TOKENIZER_UNAVAILABLE = "native_tokenizer_unavailable"

# Single-token surface forms bound to a noul label in addition to the declared
# text (and its leading-space variant). A chat model asked a yes/no question
# answers "Yes"/"No" far more naturally than the JSON label "true"/"false";
# reading both token families measures the same semantic dimension.
_NOUL_SURFACE_FORMS: Mapping[str, tuple[str, ...]] = {
    "true": ("yes",),
    "false": ("no",),
}

_NATIVE_INSTRUCTIONS = """\
Answer the numbered questions below. Each block ends with its own
"Answer (one of: ...):" cue. The decoder writes each question, in order, as
"Q <id>: <question text>" followed by that cue, and allows EXACTLY ONE answer
token at each cue; the token must be one of the block's declared candidates.
Answer every question with one of its declared candidates, in question order.
Emit nothing else: no separators, prose, punctuation or explanation."""


@dataclass(frozen=True)
class _NativeCandidate:
    """One declared label bound to its exact token id(s).

    ``token_ids`` / ``token_texts`` hold one entry per single-token variant
    found by the tokenizer, in probe order: the declared text and, when
    distinct, its space-prefixed form; noul labels additionally carry their
    natural surface forms (see ``_NOUL_SURFACE_FORMS``). More than one id
    means the label has several tokenizations; their captured weights sum.

    ``key`` is set only when the question was re-keyed (TD-29 single-token
    keys, see ``_bind_single_token_keys``): ``token_ids`` / ``token_texts``
    then bind the KEY's single-token variants, while ``label`` stays the
    original declared value the key resolves to.
    """

    label: str
    token_ids: tuple[int, ...]
    token_texts: tuple[str, ...]
    key: str | None = None

    @property
    def alternatives(self) -> int:
        """Number of grammar alternatives this candidate contributes."""
        return len(self.token_ids)


@dataclass(frozen=True)
class _NativeQuestion:
    """A question whose every candidate (and cue) is bound to exact token ids."""

    question: Question
    candidates: tuple[_NativeCandidate, ...]
    cue_token_ids: tuple[int, ...] = ()

    @property
    def keys(self) -> tuple[str, ...] | None:
        """The single-token keys in candidate order, or ``None`` when not re-keyed."""
        if not any(candidate.key is not None for candidate in self.candidates):
            return None
        return tuple(str(candidate.key) for candidate in self.candidates)

    @property
    def alternatives(self) -> int:
        """Total token alternatives in this question's answer rule."""
        return sum(candidate.alternatives for candidate in self.candidates)

    @property
    def cue_length(self) -> int:
        """Number of fixed cue tokens generated immediately before the answer."""
        return len(self.cue_token_ids)


class _CandidateTokenizationError(Exception):
    """A question's candidates or cue could not be bound to exact token ids.

    Carries the ``ParseFailure`` reason so the catalogue split can record it
    verbatim: ``native_unsupported_candidates`` (multi-token label or an
    id collision) or ``native_tokenizer_unavailable`` (the tokenizer could not
    answer for a candidate/cue text).
    """

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def run_typed_decisions_native(
    primitives: Any,
    *,
    state: str,
    questions: Sequence[Question],
    role: str,
    n_tokens: int | None = None,
    n_probs: int | None = None,
    cue_style: CueStyle | str = CueStyle.ID_ONLY,
    tokenize_fn: TokenizeFn | None = None,
    single_token_keys: bool = True,
) -> DecisionResult:
    """Score one question catalogue in a single constrained generation.

    Args:
        primitives: The ``LLMPrimitives`` seam. Only the serial
            ``llm_call(prompt, role=..., n_tokens=..., grammar=...,
            temperature=..., seed=..., n_probs=...)`` contract is used, and
            ``_last_inference_meta`` / ``_last_native_layout`` are read
            immediately after the call (see the module docstring's concurrency
            caveat). Also the source of the default tokenizer's base URL when
            ``tokenize_fn`` is not given.
        state: Task/context state injected after the stable prefix.
        questions: The catalogue; ids must be unique and non-empty.
        role: Registry role the call is charged to.
        n_tokens: Output budget; defaults to exactly the cue tokens plus one
            answer token per native-capable question. A smaller explicit value
            truncates the batch (the missing positions fail typed); a larger
            one is capped by the grammar itself, which ends after the last
            answer slot.
        n_probs: Top-K probability capture override. Defaults to the largest
            per-question token-alternative count (after tokenization) plus
            ``_N_PROBS_BUFFER``; always clamped to ``[1, _MAX_N_PROBS]``.
            Values below 1 raise ``ValueError``.
        cue_style: Which cue text is replayed before each answer token (TD-1d;
            see ``CueStyle``). ``CueStyle.ID_ONLY`` is the default since TD-6
            (cue sweep: 11.98x at 15/16 agreement, n=1; 9.60x at n=4 —
            contested, pending TD-1d.0); "full" and "short" remain
            selectable via the value strings "full"/"short"/"id_only". Unknown
            values raise ``ValueError``. The prompt, grammar shape, probability
            slicing and failure contracts do not depend on the style — only the
            replayed cue tokens do — so a sweep changes one variable at a time.
        tokenize_fn: Text -> token ids seam used to bind candidates and cue
            text to exact tokens (see module docstring). When ``None``, a
            default resolver derives the role's backend base URL from
            ``primitives`` and uses its ``POST /tokenize`` endpoint. When no
            tokenizer can be resolved, no model call is made and every
            question fails with ``native_tokenizer_unavailable``.
        single_token_keys: TD-29 single-token keys (default on). A closed set
            whose labels are not all single tokens is re-keyed with
            single-token codes (``_bind_single_token_keys``); the model picks
            a code and the decision carries the original value plus
            ``Decision.native_key``. Sets whose labels already bind are never
            re-keyed. ``False`` restores the TD-1b behavior (the whole
            question fails ``native_unsupported_candidates``).

    Returns:
        ``DecisionResult`` with ``mode="native"``. ``decisions`` holds the
        native-capable questions in catalogue order; ``failures`` holds the
        transport error first (when the call failed), then one
        ``native_unsupported_candidates`` / ``native_tokenizer_unavailable``
        failure per excluded question, then one ``native_unknown_candidate``
        failure per unresolved position, all in catalogue order. Questions are
        never silently defaulted.
    """
    catalogue = _validated_catalogue(questions)
    style = _normalize_cue_style(cue_style)
    tokenize = tokenize_fn if tokenize_fn is not None else _resolve_tokenize_fn(primitives, role)
    own_tokenizer = (
        tokenize if tokenize_fn is None and isinstance(tokenize, _HttpTokenizer) else None
    )
    try:
        return _score_native_batch(
            primitives,
            state=state,
            catalogue=catalogue,
            role=role,
            n_tokens=n_tokens,
            n_probs=n_probs,
            cue_style=style,
            tokenize=tokenize,
            single_token_keys=single_token_keys,
        )
    finally:
        if own_tokenizer is not None:
            own_tokenizer.close()


def _score_native_batch(
    primitives: Any,
    *,
    state: str,
    catalogue: Sequence[Question],
    role: str,
    n_tokens: int | None,
    n_probs: int | None,
    cue_style: CueStyle,
    tokenize: TokenizeFn | None,
    single_token_keys: bool = True,
) -> DecisionResult:
    """Tokenize, generate and slice one catalogue (tokenizer already resolved)."""
    if tokenize is None:
        native_questions: list[_NativeQuestion] = []
        tokenizer_failures = [
            ParseFailure(
                REASON_NATIVE_TOKENIZER_UNAVAILABLE,
                f"question {question.id!r}: no tokenizer could be resolved from the "
                "primitives object; ask this question in JSON mode",
            )
            for question in catalogue
        ]
    else:
        native_questions, tokenizer_failures = _tokenize_catalogue(
            catalogue, tokenize, cue_style=cue_style, single_token_keys=single_token_keys
        )

    # The token-bound questions (not bare ``Question`` objects) are rendered so
    # a re-keyed set shows its ``key = label`` legend; unkeyed sets render
    # byte-identically either way.
    prompt = build_native_prompt(state, native_questions)
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    if not native_questions:
        # Nothing is grammar-forceable; do not call the model at all.
        _record_native_layout(
            primitives,
            _native_layout(
                [],
                prompt=prompt,
                prompt_sha256=prompt_sha256,
                n_probs=0,
                n_tokens=0,
                cue_style=cue_style,
                excluded=tokenizer_failures,
            ),
        )
        return DecisionResult(
            decisions=(),
            failures=tuple(tokenizer_failures),
            raw_text="",
            mode="native",
            elapsed_ms=0.0,
            prompt_sha256=prompt_sha256,
        )

    if n_tokens is None:
        n_tokens = _total_generated_tokens(native_questions)
    if n_probs is None:
        n_probs = min(
            _MAX_N_PROBS,
            max(native.alternatives for native in native_questions) + _N_PROBS_BUFFER,
        )
    else:
        n_probs = int(n_probs)
        if n_probs < 1:
            raise ValueError(f"n_probs must be >= 1, got {n_probs}")
        n_probs = min(_MAX_N_PROBS, n_probs)

    grammar = _build_native_grammar(native_questions)
    _record_native_layout(
        primitives,
        _native_layout(
            native_questions,
            prompt=prompt,
            prompt_sha256=prompt_sha256,
            n_probs=n_probs,
            n_tokens=n_tokens,
            cue_style=cue_style,
            excluded=tokenizer_failures,
        ),
    )
    started = time.perf_counter()
    raw_text = str(
        primitives.llm_call(
            prompt,
            role=role,
            n_tokens=n_tokens,
            grammar=grammar,
            temperature=0.0,
            seed=_DECODE_SEED,
            n_probs=n_probs,
            # TD-1d.2: without this, the server's n_probs top-k is the
            # PRE-grammar raw-vocab distribution (module docstring,
            # "Probability semantics"), so a heavily grammar-narrowed
            # position's declared candidates are frequently absent from it
            # even though the grammar-forced sampled token IS one of them —
            # observed live as native_unknown_candidate on nearly every
            # answered position. post_sampling_probs=True captures the top-k
            # AFTER the full sampler chain (grammar mask included), so it
            # can only ever contain still-legal candidates.
            post_sampling_probs=True,
        )
        or ""
    )
    # TD-21.33b: this read is in the SAME synchronous call as its own
    # `llm_call` above — no thread/task boundary crossed — so the per-call-safe
    # getter applies cleanly (module docstring's concurrency caveat is about
    # the plain attribute, which this bypasses). Runtime `isinstance` (not a
    # `TYPE_CHECKING` import) so a hand-rolled test double that predates the
    # getter keeps its pre-existing exact behavior.
    from src.llm_primitives import LLMPrimitives

    meta = (
        primitives.get_last_inference_meta()
        if isinstance(primitives, LLMPrimitives)
        else getattr(primitives, "_last_inference_meta", None)
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    if raw_text.strip().startswith("[ERROR:"):
        # A dead transport cannot be fixed by slicing stale probabilities.
        transport_failure = ParseFailure(REASON_TRANSPORT_ERROR, raw_text.strip())
        return DecisionResult(
            decisions=(),
            failures=tuple([transport_failure, *tokenizer_failures]),
            raw_text=raw_text,
            mode="native",
            elapsed_ms=elapsed_ms,
            prompt_sha256=prompt_sha256,
        )

    decisions, position_failures = _decisions_from_rows(meta, native_questions)
    return DecisionResult(
        decisions=tuple(decisions),
        failures=tuple(tokenizer_failures + position_failures),
        raw_text=raw_text,
        mode="native",
        elapsed_ms=elapsed_ms,
        prompt_sha256=prompt_sha256,
    )


# ── generation layout ─────────────────────────────────────────────────────


def _total_generated_tokens(questions: Sequence[_NativeQuestion]) -> int:
    """Cue tokens plus one answer token per question."""
    return sum(native.cue_length for native in questions) + len(questions)


def _answer_row_index(questions: Sequence[_NativeQuestion], index: int) -> int:
    """Row index of question ``index``'s answer token in the capture.

    The generation order is ``cue-0 answer-0 cue-1 answer-1 ...``, so
    ``answer-i`` sits at ``i + sum(cue lengths of questions 0..i)``.
    """
    return index + sum(native.cue_length for native in questions[: index + 1])


def _native_layout(
    questions: Sequence[_NativeQuestion],
    *,
    prompt: str,
    prompt_sha256: str,
    n_probs: int,
    n_tokens: int,
    cue_style: CueStyle,
    excluded: Sequence[ParseFailure],
) -> dict[str, Any]:
    """JSON-safe per-position layout attached to the primitives before the call."""
    positions: list[dict[str, Any]] = []
    for index, native in enumerate(questions):
        question = native.question
        positions.append(
            {
                "position": index,
                "row_index": _answer_row_index(questions, index),
                "cue_token_ids": list(native.cue_token_ids),
                "cue_length": native.cue_length,
                "question": {
                    "id": question.id,
                    "kind": question.kind.value,
                    "text": question.text,
                    "options": list(question.options),
                    "levels": list(question.levels),
                    "criteria": list(question.criteria),
                },
                "candidates": [_layout_candidate(candidate) for candidate in native.candidates],
            }
        )
    return {
        "prompt": prompt,
        "prompt_sha256": prompt_sha256,
        "cue_style": cue_style.value,
        "n_probs": n_probs,
        "n_tokens": n_tokens,
        "total_tokens": _total_generated_tokens(questions),
        "positions": positions,
        "excluded": [{"reason": failure.reason, "detail": failure.detail} for failure in excluded],
    }


def _layout_candidate(candidate: _NativeCandidate) -> dict[str, Any]:
    """One candidate's layout binding; ``key`` appears ONLY for a re-keyed set.

    Omitting the field for directly bound labels keeps the pre-TD-29 layout
    byte-comparable.
    """
    entry: dict[str, Any] = {
        "label": candidate.label,
        "token_ids": list(candidate.token_ids),
        "token_texts": list(candidate.token_texts),
    }
    if candidate.key is not None:
        entry["key"] = candidate.key
    return entry


def _record_native_layout(primitives: Any, layout: Mapping[str, Any]) -> None:
    """Best-effort attach of the layout for ``native_diagnostics``.

    Never fails the run: a primitives object that refuses attribute writes
    only loses diagnostics, not the answer.
    """
    try:
        setattr(primitives, "_last_native_layout", dict(layout))
    except Exception:  # noqa: BLE001 - diagnostics are best-effort
        logger.debug("native layout could not be attached to the primitives object")


# ── tokenizer resolution and the /tokenize seam ───────────────────────────


class _HttpTokenizer:
    """Best-effort llama-server ``/tokenize`` client returning token ids.

    Mirrors the pooled ``httpx.Client`` pattern of
    ``src/backends/llama_server.py``, but unlike
    ``src/llm_primitives/tokenizer.py::LlamaTokenizer`` it NEVER substitutes a
    character-count heuristic: any transport/shape failure returns ``None`` so
    the caller fails closed instead of fabricating eligibility.
    """

    def __init__(self, base_url: str, timeout: float = _TOKENIZE_TIMEOUT_S) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def __call__(self, text: str) -> list[int] | None:
        try:
            response = self._client.post(
                f"{self.base_url}/tokenize",
                # add_special=False: no BOS is prepended, so the ids describe
                # exactly ``text``. parse_special stays at the server default.
                json={"content": text, "add_special": False, "with_pieces": False},
            )
            response.raise_for_status()
            tokens = response.json().get("tokens")
        except Exception as exc:  # noqa: BLE001 - every failure is "no answer"
            logger.debug("native /tokenize failed for %r: %s", text, exc)
            return None
        if not isinstance(tokens, list):
            return None
        ids: list[int] = []
        for token in tokens:
            if not _is_token_id(token):
                return None
            ids.append(token)
        return ids

    def close(self) -> None:
        self._client.close()


def _resolve_tokenize_fn(primitives: Any, role: str) -> TokenizeFn | None:
    """Best-effort default tokenizer for this primitives object.

    Resolution order (first usable base URL wins): the role's backend
    (unwrapping CachingBackend / RoundRobinBackend / ConcurrencyAwareBackend
    wrappers), ``primitives.server_urls[role]`` (or the sole entry), then a
    ``LlamaTokenizer`` already attached by ``LLMPrimitives.__init__``. The
    endpoint is probed once with an empty content; a dead endpoint resolves no
    tokenizer, which makes the caller fail closed with zero model calls.
    """
    base_url = _resolve_base_url(primitives, role)
    if base_url is None:
        return None
    tokenizer = _HttpTokenizer(base_url)
    if tokenizer("") is None:
        tokenizer.close()
        logger.warning("native tokenizer unavailable at %s/tokenize", base_url)
        return None
    return tokenizer


def _resolve_base_url(primitives: Any, role: str) -> str | None:
    backend: Any = None
    get_backend = getattr(primitives, "get_backend", None)
    if callable(get_backend):
        try:
            backend = get_backend(role)
        except Exception:  # noqa: BLE001 - resolution is best-effort
            backend = None
    backends = getattr(primitives, "_backends", None)
    if backend is None and isinstance(backends, Mapping):
        backend = backends.get(role)
        if backend is None and len(backends) == 1:
            backend = next(iter(backends.values()))
    if backend is not None:
        url = _backend_base_url(backend)
        if url:
            return url

    server_urls = getattr(primitives, "server_urls", None)
    if isinstance(server_urls, Mapping) and server_urls:
        raw = server_urls.get(role)
        if raw is None and len(server_urls) == 1:
            raw = next(iter(server_urls.values()))
        url = _normalize_server_url(raw)
        if url:
            return url

    attached = getattr(primitives, "_tokenizer", None)
    url = getattr(attached, "base_url", None)
    if isinstance(url, str) and url:
        return url
    return None


def _backend_base_url(backend: Any) -> str | None:
    """Unwrap backend wrappers to the first concrete ``config.base_url``.

    Mirrors and extends ``src/backends/concurrency_aware.py::_get_base_url``
    (which only handles CachingBackend): RoundRobinBackend lists and
    ConcurrencyAwareBackend full/quarter members are traversed too. Cyclic
    wrappers are impossible in practice but guarded with a visited set.
    """
    queue: list[Any] = [backend]
    visited: set[int] = set()
    while queue:
        node = queue.pop(0)
        if node is None or id(node) in visited:
            continue
        visited.add(id(node))
        config = getattr(node, "config", None)
        base_url = getattr(config, "base_url", None)
        if isinstance(base_url, str) and base_url:
            return base_url
        for attr in ("backend", "_backend", "_full", "full_backend"):
            inner = getattr(node, attr, None)
            if inner is not None:
                queue.append(inner)
        for attr in ("backends", "_quarters", "quarter_backends"):
            members = getattr(node, attr, None)
            if isinstance(members, (list, tuple)):
                queue.extend(member for member in members if member is not None)
    return None


def _normalize_server_url(raw: Any) -> str | None:
    """First URL of a (possibly comma-separated, possibly ``full:``) role URL."""
    if not isinstance(raw, str):
        return None
    first = raw.split(",", 1)[0].strip()
    if first.startswith("full:"):
        first = first[len("full:") :].strip()
    return first or None


# ── catalogue tokenization ────────────────────────────────────────────────


def _tokenize_catalogue(
    questions: Sequence[Question],
    tokenize: TokenizeFn,
    cue_style: CueStyle = CueStyle.ID_ONLY,
    single_token_keys: bool = True,
) -> tuple[list[_NativeQuestion], list[ParseFailure]]:
    """Partition the catalogue into token-bound questions and typed failures.

    Tokens are memoized per run (the same label appears in many questions), and
    every failure is recorded in catalogue order with the reason that keeps the
    question out of the native batch. ``cue_style`` selects the cue text bound
    to exact tokens (TD-1d); candidate binding is style-independent.
    """
    cache: dict[str, Sequence[int] | None] = {}

    def tokenize_cached(text: str) -> Sequence[int] | None:
        if text not in cache:
            try:
                cache[text] = tokenize(text)
            except Exception as exc:  # noqa: BLE001 - any failure is "no answer"
                logger.warning("native tokenizer raised for %r: %s", text, exc)
                cache[text] = None
        return cache[text]

    native: list[_NativeQuestion] = []
    failures: list[ParseFailure] = []
    for question in questions:
        try:
            native.append(
                _tokenize_question(
                    question,
                    tokenize_cached,
                    cue_style=cue_style,
                    single_token_keys=single_token_keys,
                )
            )
        except _CandidateTokenizationError as exc:
            failures.append(ParseFailure(exc.reason, f"question {question.id!r}: {exc.detail}"))
    return native, failures


def _tokenize_question(
    question: Question,
    tokenize: TokenizeFn,
    cue_style: CueStyle = CueStyle.ID_ONLY,
    single_token_keys: bool = True,
) -> _NativeQuestion:
    """Bind every candidate label of one question — and its cue — to token ids.

    When some label is not a single token and ``single_token_keys`` is on, the
    WHOLE set is re-keyed with single-token codes (TD-29, see
    ``_bind_single_token_keys``); a set whose labels all bind is never
    re-keyed, so its binding, prompt and grammar are byte-identical to TD-1b.

    Raises:
        _CandidateTokenizationError: when a candidate, key or cue text cannot
            be tokenized at all (``native_tokenizer_unavailable``), or when a
            label is not a single token and no single-token key set could be
            bound (keys disabled, the set exceeds the key alphabet, or the
            alphabet ran out of usable codes), or two labels collide on one
            token id (``native_unsupported_candidates``). Partial eligibility
            is not a thing: an answer that could have chosen an unsupported
            label must go to the JSON fallback as a whole question.
    """
    candidates: list[_NativeCandidate] = []
    unsupported: list[str] = []
    for label in _candidate_labels(question):
        token_ids: list[int] = []
        token_texts: list[str] = []
        for text in _label_variants(label, question.kind):
            ids = tokenize(text)
            if ids is None:
                raise _CandidateTokenizationError(
                    REASON_NATIVE_TOKENIZER_UNAVAILABLE,
                    f"the tokenizer returned no ids for candidate text {text!r}; "
                    "ask this question in JSON mode",
                )
            if len(ids) == 1:
                token_ids.append(ids[0])
                token_texts.append(text)
        if not token_ids:
            unsupported.append(label)
            continue
        candidates.append(
            _NativeCandidate(
                label=label,
                token_ids=tuple(token_ids),
                token_texts=tuple(token_texts),
            )
        )
    if unsupported:
        not_single = (
            f"candidate labels {unsupported!r} do not tokenize to exactly one token "
            "(with or without a leading space)"
        )
        if not single_token_keys:
            raise _CandidateTokenizationError(
                REASON_NATIVE_UNSUPPORTED_CANDIDATES,
                f"{not_single}; ask this question in JSON mode",
            )
        candidates = _bind_single_token_keys(question, tokenize, not_single=not_single)
    _reject_token_id_collisions(candidates)
    keys = tuple(str(candidate.key) for candidate in candidates) if unsupported else None
    cue_text = _cue_text(question, cue_style, keys=keys)
    cue_ids = tokenize(cue_text)
    if cue_ids is None or not cue_ids:
        raise _CandidateTokenizationError(
            REASON_NATIVE_TOKENIZER_UNAVAILABLE,
            f"the tokenizer returned no ids for the cue text {cue_text!r}; "
            "ask this question in JSON mode",
        )
    return _NativeQuestion(
        question=question,
        candidates=tuple(candidates),
        cue_token_ids=tuple(cue_ids),
    )


def _single_token_key_alphabet() -> str:
    """The TD-9 single-token code alphabet (A-Z then 0-9, 36 symbols).

    Reused from ``routing_replay`` so routing codes and native keys are one
    mechanism. Lazy import: ``routing_replay`` pulls ``llm_primitives``
    statistics that the native hot path should not load at import time.
    """
    from src.typed_decisions.routing_replay import _CODE_ALPHABET

    return _CODE_ALPHABET


def _bind_single_token_keys(
    question: Question,
    tokenize: TokenizeFn,
    *,
    not_single: str,
) -> list[_NativeCandidate]:
    """Re-key a closed set whose labels are not all single tokens (TD-29).

    Each declared label, in declaration order, takes the next code of the
    TD-9 alphabet whose bare or space-prefixed form is exactly ONE token for
    the served tokenizer (probed through the same ``/tokenize`` seam as the
    labels) and whose token id(s) collide with no key already bound. A code
    that is itself a DIFFERENT label of this set is skipped so the legend can
    never read ``B = A``. The assignment depends only on candidate order and
    the tokenizer, so it is stable across runs. ``label`` stays the original
    value; the grammar binds the key's ids and the decision maps back.

    The key set is all-or-nothing and it falls back exactly as before: more
    labels than codes, or an alphabet exhausted by multi-token/colliding
    codes, raises ``native_unsupported_candidates`` with the reason recorded
    in the detail; a key probe the tokenizer cannot answer raises
    ``native_tokenizer_unavailable``.
    """
    alphabet = _single_token_key_alphabet()
    labels = _candidate_labels(question)
    if len(labels) > len(alphabet):
        raise _CandidateTokenizationError(
            REASON_NATIVE_UNSUPPORTED_CANDIDATES,
            f"{not_single}, and single-token keys cannot be bound: {len(labels)} "
            f"candidates exceed the {len(alphabet)} single-token key codes (A-Z, 0-9); "
            "ask this question in JSON mode",
        )
    label_texts = {label.strip() for label in labels}
    used_ids: set[int] = set()
    rejected: list[str] = []
    codes = iter(alphabet)
    candidates: list[_NativeCandidate] = []
    for label in labels:
        for code in codes:
            if code in label_texts and code != label.strip():
                rejected.append(f"{code} (a different label of this set)")
                continue
            token_ids: list[int] = []
            token_texts: list[str] = []
            for text in _candidate_variants(code):
                ids = tokenize(text)
                if ids is None:
                    raise _CandidateTokenizationError(
                        REASON_NATIVE_TOKENIZER_UNAVAILABLE,
                        f"{not_single}, and the tokenizer returned no ids for the "
                        f"single-token key probe {text!r}; ask this question in JSON mode",
                    )
                if len(ids) == 1:
                    token_ids.append(ids[0])
                    token_texts.append(text)
            if not token_ids:
                rejected.append(f"{code} (not a single token)")
                continue
            if used_ids.intersection(token_ids):
                rejected.append(f"{code} (token id collides with an earlier key)")
                continue
            used_ids.update(token_ids)
            candidates.append(
                _NativeCandidate(
                    label=label,
                    token_ids=tuple(token_ids),
                    token_texts=tuple(token_texts),
                    key=code,
                )
            )
            break
        else:
            raise _CandidateTokenizationError(
                REASON_NATIVE_UNSUPPORTED_CANDIDATES,
                f"{not_single}, and single-token keys cannot be bound: the "
                f"{len(alphabet)}-code alphabet ran out after binding {len(candidates)} "
                f"of {len(labels)} candidates (rejected codes: {rejected!r}); "
                "ask this question in JSON mode",
            )
    return candidates


def _candidate_variants(label: str) -> tuple[str, ...]:
    """The candidate texts probed for single-token tokenization.

    llama.cpp's pre-tokenizer distinguishes a word at a boundary (``false``)
    from the same word after a space (`` false``); either can be the token the
    decoder is about to emit, so both are probed (and both kept when both are
    single tokens). A label declared with a leading space is probed as declared
    and with that space removed.
    """
    if label.startswith(" "):
        stripped = label[1:]
        return (label, stripped) if stripped else (label,)
    return (label, " " + label)


def _label_variants(label: str, kind: QuestionKind) -> tuple[str, ...]:
    """All token texts bound to one declared label.

    The declared text and its leading-space variant, plus (for ``noul``
    labels) the natural surface forms from ``_NOUL_SURFACE_FORMS`` and their
    leading-space variants. Order is probe order and deduplicated.
    """
    texts: list[str] = []
    for text in _candidate_variants(label):
        if text not in texts:
            texts.append(text)
    if kind is QuestionKind.NOUL:
        for form in _NOUL_SURFACE_FORMS.get(label, ()):
            for text in _candidate_variants(form):
                if text not in texts:
                    texts.append(text)
    return tuple(texts)


def _reject_token_id_collisions(candidates: Sequence[_NativeCandidate]) -> None:
    """Two labels bound to the same token id are indistinguishable natively."""
    seen: dict[int, str] = {}
    for candidate in candidates:
        for token_id in candidate.token_ids:
            other = seen.get(token_id)
            if other is not None and other != candidate.label:
                raise _CandidateTokenizationError(
                    REASON_NATIVE_UNSUPPORTED_CANDIDATES,
                    f"candidate labels {other!r} and {candidate.label!r} both tokenize "
                    f"to token id {token_id}; the native grammar cannot distinguish "
                    "them, ask this question in JSON mode",
                )
            seen[token_id] = candidate.label


# ── grammar and prompt ────────────────────────────────────────────────────


def _build_native_grammar(questions: Sequence[_NativeQuestion]) -> str:
    """Build the cue/answer token-sequence grammar for the native batch.

    The generation order is ``cue-0 answer-0 cue-1 answer-1 ...``: cue rule
    ``i`` pins the question's cue to its exact token ids, answer rule ``i`` is
    the alternation of question ``i``'s candidate token alternatives. This is
    NOT ``schema.build_gbnf`` (that one builds the JSON-object grammar):
    native mode never emits JSON, it emits a fixed token layout with one
    constrained answer per question.

    Token-id terminals are load-bearing: a quoted literal would be parsed as
    character elements by ``llama-grammar.cpp::parse_sequence``, and the
    decoder could satisfy it with a partial piece (the TD-1a ``fal`` bug).
    ``<[id]>`` can only be advanced by exactly that token, so each cue and
    answer consumes exactly the tokens the layout assumes.
    """
    root_parts: list[str] = []
    rules: list[str] = []
    for index, native in enumerate(questions):
        cue = " ".join(f"<[{token_id}]>" for token_id in native.cue_token_ids)
        alternatives = " | ".join(
            f"<[{token_id}]>" for candidate in native.candidates for token_id in candidate.token_ids
        )
        rules.append(f"cue-{index} ::= {cue}")
        rules.append(f"answer-{index} ::= {alternatives}")
        root_parts.extend((f"cue-{index}", f"answer-{index}"))
    root = "root ::= " + " ".join(root_parts)
    return "\n".join([root, *rules]) + "\n"


def _cue_text(
    question: Question,
    cue_style: CueStyle | str = CueStyle.ID_ONLY,
    *,
    keys: Sequence[str] | None = None,
) -> str:
    """The fixed cue replayed immediately before this question's answer token.

    ``ID_ONLY`` (default since TD-6) keeps just ``"\\n<id>: "``: a minimal
    delimiter whose grounding comes from the numbered catalogue in
    ``build_native_prompt``. ``FULL`` is the TD-1c cue: id, full question text
    and the declared labels, ending on an explicit answer delimiter. ``SHORT``
    keeps the id and the first ``_SHORT_CUE_WORDS`` words of the question.
    Every style starts with a newline so the generated transcript stays
    readable. ``keys`` (a re-keyed set, TD-29) replaces the declared labels in
    the ``FULL`` answer list, since the answer token is a key; ``ID_ONLY`` and
    ``SHORT`` never list candidates and are unaffected.
    """
    style = _normalize_cue_style(cue_style)
    if style is CueStyle.ID_ONLY:
        return f"\n{question.id}: "
    if style is CueStyle.SHORT:
        excerpt = " ".join(question.text.split()[:_SHORT_CUE_WORDS])
        return f"\nQ {question.id}: {excerpt}\n"
    labels = list(keys) if keys is not None else _candidate_labels(question)
    return f"\nQ {question.id}: {question.text}\nAnswer (one of: {', '.join(labels)}): "


def build_native_prompt(
    state: str,
    questions: Sequence[Question | _NativeQuestion],
) -> str:
    """Build the deterministic native prompt for a batch, in catalogue order.

    Pure helper (no tokenizer, no model): identical inputs give a
    byte-identical prompt. Each question is its own numbered block with the
    candidates echoed and an explicit ``Answer (one of: ...):`` cue. The
    decoder replays a cue as fixed tokens between the answer slots (see
    ``_cue_text`` and ``_build_native_grammar``).

    Deliberately cue-style invariant (TD-1d): the catalogue is byte-identical
    for every ``CueStyle``, so a sweep changes exactly which cue tokens the
    decoder replays between answers — and shorter cues draw their grounding
    from this catalogue.

    Accepts plain ``Question`` objects or the runner's token-bound
    ``_NativeQuestion`` (whose ``.question`` is used; a re-keyed one renders
    its ``key = label`` legend and asks for the key, TD-29); the runner calls it with
    only the native-capable questions, so a preview built from a full
    catalogue may be longer than the exact runtime prompt. The exact runtime
    prompt is stored on the layout side channel (``_last_native_layout``).
    """
    lines = [_NATIVE_INSTRUCTIONS, ""]
    lines.append(f"STATE:\n{state}")
    lines.append("")
    lines.append("QUESTION SEQUENCE:")
    for index, item in enumerate(questions, start=1):
        question = _as_question(item)
        labels = _candidate_labels(question)
        keys = item.keys if isinstance(item, _NativeQuestion) else None
        lines.append(f"{index}. id={question.id} kind={question.kind.value}")
        lines.append(f"   question: {question.text}")
        if keys is None:
            lines.append(f"   candidates: {' | '.join(labels)}")
            lines.append(f"   Answer (one of: {', '.join(labels)}):")
        else:
            # TD-29 single-token keys: the label stays the display text, the
            # answer token is its key.
            legend = " | ".join(f"{key} = {label}" for key, label in zip(keys, labels))
            lines.append(f"   candidates: {legend}")
            lines.append(f"   Answer with the key (one of: {', '.join(keys)}):")
    return "\n".join(lines) + "\n"


def _as_question(item: Question | _NativeQuestion) -> Question:
    if isinstance(item, _NativeQuestion):
        return item.question
    if isinstance(item, Question):
        return item
    raise TypeError(
        "build_native_prompt expects Question or _NativeQuestion entries, "
        f"got {type(item).__name__}"
    )


def _candidate_labels(question: Question) -> list[str]:
    """Candidate labels in declaration order (the grammar's alternative order)."""
    if question.kind is QuestionKind.CHOICE:
        return list(question.options)
    if question.kind is QuestionKind.SCORE:
        return [str(level) for level in question.levels]
    return ["true", "false"]


# ── TD-1d option (b): per-question reads against one cached prefix ────────
#
# PARKED, UNWIRED (TD-21.33b, landed 2026-09-24 from 2026-09-22 shared-clone
# WIP — see handoffs/active/typed-decision-plane.md TD-1d.4). Not called from
# any production path. TD-1d.0's 2026-09-18 re-measurement found this shape a
# LOSER on the frozen stack (native_pq3 2.67x, slower than id_only-serial's
# 9.60x; the exclusive `heavy_model` lock serializes concurrent `llm_call`s
# end to end today, so the fan-out buys nothing over that lock). It is kept,
# still exercised by unit tests against fake primitives, as the harness for
# any FUTURE runtime choice that removes that serialization (a real
# multi-slot readout, or a lock policy change) — re-litigate against TD-1d.0
# before adopting it anywhere.


def run_typed_decisions_native_parallel(
    primitives_pool: Sequence[Any],
    *,
    state: str,
    questions: Sequence[Question],
    role: str,
    n_probs: int | None = None,
    cue_style: CueStyle | str = CueStyle.ID_ONLY,
    tokenize_fn: TokenizeFn | None = None,
    pin_slots: bool = False,
    warm_prefix: bool = False,
    single_token_keys: bool = True,
) -> DecisionResult:
    """Answer the catalogue with ONE constrained read per question, fanned out.

    The TD-1c serial layout pays one forward pass per grammar-forced cue
    token. This shape moves every cue into the PROMPT instead: each question
    gets its own request whose prompt is the byte-identical catalogue prefix
    (``build_native_prompt`` over the native-capable questions) followed by
    that question's cue, a grammar that admits exactly one answer token, and
    ``n_tokens=1``. The shared prefix is prefilled once per server slot and
    then hit from the prompt cache; the cue tokens are prefilled (one batched
    pass) rather than decoded (one pass each). The requests are issued
    concurrently, one worker per entry in ``primitives_pool``.

    Args:
        primitives_pool: Independent primitives objects, one per worker. They
            must NOT share instance state: ``_last_inference_meta`` is
            instance-level (module docstring), so each worker reads its own
            object's meta right after its own call. ``len(pool)`` is the
            fan-out width; a pool of one is the same shape run serially.
        state, questions, role, n_probs, cue_style, tokenize_fn,
            single_token_keys: As for ``run_typed_decisions_native``. ``n_probs`` defaults to the same
            per-batch value the serial runner would use.
        pin_slots: When true, worker ``i`` passes ``slot_id=i`` on every call
            so all of its reads share one server slot's prompt cache (the
            seam must accept a ``slot_id`` keyword; ``LLMPrimitives.llm_call``
            does not, so this is for direct-backend seams). Measured
            2026-09-18: llama-server's automatic slot choice repeatedly placed
            reads on a slot whose cache missed the shared prefix
            (``cache_n=0``, a full ~1.1k-token prefill per read), while a
            slot holding the prefix served the read in ~45 ms.
        warm_prefix: When true, every worker first sends the bare prefix
            (``n_tokens=1``, no grammar) on its own slot, INSIDE the timed
            window, before its reads. Load-bearing on hybrid/recurrent
            models (``qwen35moe`` has SSM layers): llama-server can only
            restore a slot's cache at a saved checkpoint, and the checkpoint
            written after a read sits past the shared prefix, so back-to-back
            reads on the same slot reuse nothing (measured ``cache_n=0``, ~450
            ms each). A prefix-only request leaves a checkpoint exactly at
            the prefix end; reads then reuse it (measured ``cache_n=1105``,
            ~128 ms each). The warm costs ~56 ms when the checkpoint already
            exists and a full prefill when it does not, so the arm is
            self-healing after another prompt evicted its slot.

    Returns:
        ``DecisionResult`` with ``mode="native_parallel"``. ``elapsed_ms`` is
        the wall time of the whole fan-out (first request issued to last
        response sliced). ``raw_text`` joins the per-question emitted tokens in
        catalogue order. Failures follow the serial contract exactly, plus
        ``transport_error`` per failed request. The first pool entry receives
        the layout side channel (``_last_native_layout``) with one position
        per question and ``per_request`` meta snapshots (slot id, prompt /
        cache token counts) for residency and cache-hit auditing.
    """
    pool = list(primitives_pool)
    if not pool:
        raise ValueError("run_typed_decisions_native_parallel requires a non-empty primitives pool")
    catalogue = _validated_catalogue(questions)
    style = _normalize_cue_style(cue_style)
    tokenize = tokenize_fn if tokenize_fn is not None else _resolve_tokenize_fn(pool[0], role)
    own_tokenizer = (
        tokenize if tokenize_fn is None and isinstance(tokenize, _HttpTokenizer) else None
    )
    try:
        if tokenize is None:
            native_questions: list[_NativeQuestion] = []
            tokenizer_failures = [
                ParseFailure(
                    REASON_NATIVE_TOKENIZER_UNAVAILABLE,
                    f"question {question.id!r}: no tokenizer could be resolved from the "
                    "primitives object; ask this question in JSON mode",
                )
                for question in catalogue
            ]
        else:
            native_questions, tokenizer_failures = _tokenize_catalogue(
                catalogue, tokenize, cue_style=style, single_token_keys=single_token_keys
            )
    finally:
        if own_tokenizer is not None:
            own_tokenizer.close()

    prefix = build_native_prompt(state, native_questions)
    prompt_sha256 = hashlib.sha256(prefix.encode("utf-8")).hexdigest()
    if not native_questions:
        return DecisionResult(
            decisions=(),
            failures=tuple(tokenizer_failures),
            raw_text="",
            mode="native_parallel",
            elapsed_ms=0.0,
            prompt_sha256=prompt_sha256,
        )

    if n_probs is None:
        n_probs = min(
            _MAX_N_PROBS,
            max(native.alternatives for native in native_questions) + _N_PROBS_BUFFER,
        )
    else:
        n_probs = int(n_probs)
        if n_probs < 1:
            raise ValueError(f"n_probs must be >= 1, got {n_probs}")
        n_probs = min(_MAX_N_PROBS, n_probs)

    # One request per question: the cue is prompt text here, so the answer
    # row is row 0 — the layout copy carries no cue tokens on purpose.
    jobs: list[tuple[int, _NativeQuestion, str, str]] = []
    for index, native in enumerate(native_questions):
        answer_only = _NativeQuestion(question=native.question, candidates=native.candidates)
        alternatives = " | ".join(
            f"<[{token_id}]>" for candidate in native.candidates for token_id in candidate.token_ids
        )
        grammar = f"root ::= {alternatives}\n"
        prompt = prefix + _cue_text(native.question, style, keys=native.keys)
        jobs.append((index, answer_only, prompt, grammar))

    outcomes: dict[int, tuple[str, Any]] = {}
    per_request: dict[int, dict[str, Any]] = {}
    warm_requests: dict[int, dict[str, Any]] = {}

    def worker(
        worker_index: int, assigned: Sequence[tuple[int, _NativeQuestion, str, str]]
    ) -> None:
        # TD-21.33b: each worker owns exactly one `pool[worker_index]` object
        # (never shared across threads per this function's contract above),
        # and every meta read below happens in THIS thread, immediately after
        # THIS thread's own call — no thread/task boundary is crossed between
        # the write and the read, so the per-call-safe getter applies
        # cleanly. A plain `threading.Thread` (unlike `asyncio.to_thread`)
        # starts with its own empty context, so `get_last_inference_meta()`
        # is additionally immune to any OTHER worker's concurrent call even
        # if a caller violated the "independent objects" contract and passed
        # the same primitives instance twice — the plain attribute this
        # replaces would not be. Runtime `isinstance` (not a `TYPE_CHECKING`
        # import) so a hand-rolled test double that predates the getter keeps
        # its pre-existing exact behavior.
        from src.llm_primitives import LLMPrimitives

        primitives = pool[worker_index]
        is_real_primitives = isinstance(primitives, LLMPrimitives)
        extra: dict[str, Any] = {"slot_id": worker_index} if pin_slots else {}
        if warm_prefix:
            started_warm = time.perf_counter()
            primitives.llm_call(
                prefix,
                role=role,
                n_tokens=1,
                temperature=0.0,
                seed=_DECODE_SEED,
                **extra,
            )
            warm_meta = (
                primitives.get_last_inference_meta()
                if is_real_primitives
                else getattr(primitives, "_last_inference_meta", None)
            )
            snapshot = {
                "worker": worker_index,
                "call_ms": (time.perf_counter() - started_warm) * 1000.0,
            }
            if isinstance(warm_meta, Mapping):
                for key in ("id_slot", "prompt_n", "cache_n", "prompt_ms"):
                    if key in warm_meta:
                        snapshot[key] = warm_meta[key]
            warm_requests[worker_index] = snapshot
        for index, _native, prompt, grammar in assigned:
            started_call = time.perf_counter()
            raw = str(
                primitives.llm_call(
                    prompt,
                    role=role,
                    n_tokens=1,
                    grammar=grammar,
                    temperature=0.0,
                    seed=_DECODE_SEED,
                    n_probs=n_probs,
                    # TD-1d.2: same fix as the single-call path — see the
                    # module docstring's "Probability semantics" section.
                    post_sampling_probs=True,
                    **extra,
                )
                or ""
            )
            meta = (
                primitives.get_last_inference_meta()
                if is_real_primitives
                else getattr(primitives, "_last_inference_meta", None)
            )
            outcomes[index] = (raw, meta)
            snapshot = {
                "worker": worker_index,
                "call_ms": (time.perf_counter() - started_call) * 1000.0,
            }
            if isinstance(meta, Mapping):
                for key in ("id_slot", "prompt_n", "cache_n", "tokens", "prompt_ms", "gen_ms"):
                    if key in meta:
                        snapshot[key] = meta[key]
            per_request[index] = snapshot

    assignments: list[list[tuple[int, _NativeQuestion, str, str]]] = [[] for _ in pool]
    for job in jobs:
        assignments[job[0] % len(pool)].append(job)
    threads = [
        threading.Thread(target=worker, args=(worker_index, assigned), daemon=True)
        for worker_index, assigned in enumerate(assignments)
        if assigned
    ]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    decisions: list[Decision] = []
    position_failures: list[ParseFailure] = []
    raw_parts: list[str] = []
    for index, answer_only, _prompt, _grammar in jobs:
        raw, meta = outcomes.get(index, ("", None))
        raw_parts.append(raw)
        if raw.strip().startswith("[ERROR:") or index not in outcomes:
            position_failures.append(
                ParseFailure(
                    REASON_TRANSPORT_ERROR,
                    f"question {answer_only.question.id!r}: {raw.strip() or 'no response'}",
                )
            )
            continue
        got, failed = _decisions_from_rows(meta, [answer_only])
        decisions.extend(got)
        position_failures.extend(failed)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    _record_native_layout(
        pool[0],
        {
            **_native_layout(
                [job[1] for job in jobs],
                prompt=prefix,
                prompt_sha256=prompt_sha256,
                n_probs=n_probs,
                n_tokens=len(jobs),
                cue_style=style,
                excluded=tokenizer_failures,
            ),
            "shape": "per_question_parallel",
            "workers": len(pool),
            "pin_slots": pin_slots,
            "warm_prefix": warm_prefix,
            "warm_requests": [warm_requests.get(worker_index) for worker_index in range(len(pool))],
            "per_request": [per_request.get(index) for index, *_ in jobs],
        },
    )
    return DecisionResult(
        decisions=tuple(decisions),
        failures=tuple(tokenizer_failures + position_failures),
        raw_text="".join(raw_parts),
        mode="native_parallel",
        elapsed_ms=elapsed_ms,
        prompt_sha256=prompt_sha256,
    )


# ── completion_probabilities slicing ──────────────────────────────────────


def _decisions_from_rows(
    meta: Any,
    questions: Sequence[_NativeQuestion],
) -> tuple[list[Decision], list[ParseFailure]]:
    """Turn captured probability rows into typed decisions / per-position failures.

    Rows are 1:1 with generated tokens (cue rows included), so question
    ``i``'s answer row is addressed by ``_answer_row_index``.
    """
    rows = _rows_from_meta(meta)
    decisions: list[Decision] = []
    failures: list[ParseFailure] = []

    for index, native in enumerate(questions):
        question = native.question
        row_index = _answer_row_index(questions, index)
        if row_index >= len(rows):
            failures.append(
                ParseFailure(
                    REASON_NATIVE_UNKNOWN_CANDIDATE,
                    f"position {index + 1} question {question.id!r}: no "
                    f"completion_probabilities row was captured at generated-token "
                    f"index {row_index} (rows captured: {len(rows)})",
                )
            )
            continue
        row = rows[row_index]
        id_to_label = {
            token_id: candidate.label
            for candidate in native.candidates
            for token_id in candidate.token_ids
        }
        text_to_label = {
            text: candidate.label
            for candidate in native.candidates
            for text in candidate.token_texts
        }
        emitted_label = _match_row_label(row, id_to_label, text_to_label)
        if emitted_label is None:
            failures.append(
                ParseFailure(
                    REASON_NATIVE_UNKNOWN_CANDIDATE,
                    f"position {index + 1} question {question.id!r}: emitted token "
                    f"({_describe_row_token(row)}) is not one of the declared "
                    f"candidates {_candidate_labels(question)!r}",
                )
            )
            continue
        entries = _row_entries(row)
        # Text fallback is scoped to a row that lacks ``id`` altogether: when
        # the row carries an id the capture is a modern one, and every entry
        # must resolve by id (a text fallback there could resurrect a stale
        # piece from a mixed shape).
        text_fallback = not _is_token_id(row.get("id"))
        weights = _candidate_weights(entries, native, text_fallback=text_fallback)
        if not any(weight > 0.0 for weight in weights.values()):
            # normalize_probabilities would otherwise fabricate a uniform
            # distribution over an all-zero slice; the capture is unusable.
            failures.append(
                ParseFailure(
                    REASON_NATIVE_UNKNOWN_CANDIDATE,
                    f"position {index + 1} question {question.id!r}: none of the "
                    "declared candidate tokens appears in the captured top probabilities",
                )
            )
            continue
        decisions.append(_decision_from_weights(native, weights, entries, text_fallback))
    return decisions, failures


def _match_row_label(
    row: Mapping[str, Any],
    id_to_label: Mapping[int, str],
    text_to_label: Mapping[str, str],
) -> str | None:
    """Resolve the emitted row to a candidate label.

    The v9 row carries ``id``, which is authoritative: when an ``id`` is
    present the text is NOT consulted (the text is derived from the id, and
    mixing keys would let a stale text override the sampler's token). Text
    matching is the documented fallback for legacy rows that lack ``id``.
    """
    token_id = row.get("id")
    if _is_token_id(token_id):
        return id_to_label.get(token_id)
    text = _row_token_text(row)
    if isinstance(text, str):
        return text_to_label.get(text)
    return None


def _describe_row_token(row: Mapping[str, Any]) -> str:
    """Human-readable identity of the emitted row token for failure details."""
    token_id = row.get("id")
    text = _row_token_text(row)
    parts = []
    if _is_token_id(token_id):
        parts.append(f"id={token_id}")
    parts.append(f"text={text!r}")
    return " ".join(parts)


def _decision_from_weights(
    native: _NativeQuestion,
    weights: Mapping[str, float],
    entries: Sequence[Mapping[str, Any]],
    text_fallback: bool,
) -> Decision:
    """Renormalize candidate weights and build the typed ``Decision``.

    ``weights`` carries one entry per declared label (absent candidates at
    ``0.0``), so the renormalized distribution always covers the full
    candidate set. The value is the argmax of the slice (ties resolve to the
    first declared candidate). When every label maps to exactly one token
    this coincides with the emitted token under the grammar's masked greedy
    decode; when a label has several accepted surface forms the summed mass
    can outrank the emitted token's own label, so the emitted token was only
    used as a membership check upstream (see the module docstring and
    ``emitted_matches_argmax`` in ``native_diagnostics``).
    """
    question = native.question
    probabilities = normalize_probabilities(weights)
    value_label = max(weights, key=lambda label: weights[label])
    value_candidate = next(
        candidate for candidate in native.candidates if candidate.label == value_label
    )
    token_logprob = _logprob_for(entries, value_candidate, text_fallback=text_fallback)

    if question.kind is QuestionKind.SCORE:
        typed_probabilities: Mapping[str | int, float] = {
            int(label): probability for label, probability in probabilities.items()
        }
        value: object = int(value_label)
        confidence = score_confidence(typed_probabilities)
    elif question.kind is QuestionKind.NOUL:
        typed_probabilities = probabilities
        value = value_label == "true"
        confidence = choice_confidence(typed_probabilities)
    else:
        typed_probabilities = probabilities
        value = value_label
        confidence = choice_confidence(typed_probabilities)

    return Decision(
        question_id=question.id,
        kind=question.kind,
        value=value,
        probabilities=typed_probabilities,
        confidence=confidence,
        mode="native",
        token_logprob=token_logprob,
        native_key=value_candidate.key,
    )


# ── live diagnostics ──────────────────────────────────────────────────────


def native_diagnostics(result: DecisionResult, primitives_snapshot: Any) -> dict[str, Any]:
    """Report, per question, the raw capture at its answer position.

    Answers "was the candidate slice representative, or did the model's mass
    land outside the candidates?" from the runner's own evidence, without
    another model call.

    Args:
        result: The ``DecisionResult`` returned by
            ``run_typed_decisions_native`` (or by
            ``run_typed_decisions(mode="native")``).
        primitives_snapshot: The primitives object used for the run (it carries
            both ``_last_inference_meta`` and the layout side channel), a
            mapping with ``{"native_layout": ..., "meta"/"completion_probabilities":
            ...}``, or a bare meta/``completion_probabilities`` mapping (then
            output is row-level only — there is no question binding).

    Returns:
        A JSON-safe dict. With a layout: the replayed ``cue_style`` and one
        entry per question carrying the
        question id, its answer row index, the emitted token, the raw captured
        weights per candidate label (``candidate_weights_raw``), the sliced raw
        mass, the exact mass OUTSIDE the candidate set when every candidate
        variant was captured (``mass_outside_candidates``), the slice's argmax,
        whether the emitted token maps to that argmax (``emitted_matches_argmax``
        — must be true on a real grammar-constrained capture), the full raw
        top-k row entries, and the subset of top-k entries that are not declared
        candidate tokens. Without a layout: one entry per captured row with the
        raw top-k. ``result``'s decisions/failures fill ``resolved_value`` /
        ``failure`` per question.
    """
    rows = _snapshot_rows(primitives_snapshot)
    layout = _snapshot_layout(primitives_snapshot)
    decisions = {decision.question_id: decision for decision in result.decisions}
    report: dict[str, Any] = {
        "mode": result.mode,
        "prompt_sha256": result.prompt_sha256,
        "layout_present": layout is not None,
        "rows_captured": len(rows),
        "cue_style": None,
        "total_tokens_expected": None,
        "n_probs": None,
        "excluded": [],
        "failures": [
            {"reason": failure.reason, "detail": failure.detail} for failure in result.failures
        ],
        "positions": [],
        "notes": [],
    }
    if layout is None:
        report["notes"].append(
            "no native layout found on the snapshot: pass the primitives object (or a "
            "mapping with 'native_layout') to bind rows to question ids; reporting raw "
            "rows only"
        )
        for row_index, row in enumerate(rows):
            report["positions"].append(_row_diagnostic(row_index, row))
        return report

    report["cue_style"] = layout.get("cue_style")
    report["total_tokens_expected"] = layout.get("total_tokens")
    report["n_probs"] = layout.get("n_probs")
    excluded = layout.get("excluded")
    if isinstance(excluded, list):
        report["excluded"] = list(excluded)
    layout_sha = layout.get("prompt_sha256")
    if isinstance(layout_sha, str) and layout_sha:
        report["prompt_sha256"] = layout_sha
    positions = layout.get("positions")
    if not isinstance(positions, list):
        report["notes"].append("native layout carries no positions list")
        return report
    for position in positions:
        if not isinstance(position, Mapping):
            continue
        row_index = position.get("row_index")
        row = None
        if isinstance(row_index, int) and 0 <= row_index < len(rows):
            row = rows[row_index]
        report["positions"].append(_position_diagnostic(position, row, decisions, result.failures))
    if len(rows) < int(report["total_tokens_expected"] or 0):
        report["notes"].append(
            "fewer probability rows than expected generated tokens: positions beyond "
            "rows_captured have no capture (truncated generation or dropped rows)"
        )
    return report


def _snapshot_rows(primitives_snapshot: Any) -> list[Mapping[str, Any]]:
    """Extract the captured rows from any supported snapshot shape.

    TD-21.33b: ``native_diagnostics`` (this helper's only caller) has no
    production call site today — every caller is a test that reads the
    snapshot in the SAME synchronous call as the run it diagnoses — so the
    per-call-safe getter applies cleanly, with a runtime ``isinstance`` guard
    so a bare-mapping snapshot or a hand-rolled test double that predates the
    getter keeps its pre-existing exact behavior.
    """
    meta: Any = primitives_snapshot
    if not isinstance(meta, Mapping):
        from src.llm_primitives import LLMPrimitives

        meta = (
            primitives_snapshot.get_last_inference_meta()
            if isinstance(primitives_snapshot, LLMPrimitives)
            else getattr(primitives_snapshot, "_last_inference_meta", None)
        )
    elif isinstance(meta.get("meta"), Mapping):
        meta = meta["meta"]
    return _rows_from_meta(meta)


def _snapshot_layout(primitives_snapshot: Any) -> Mapping[str, Any] | None:
    """Extract the native layout from any supported snapshot shape."""
    layout: Any = None
    if isinstance(primitives_snapshot, Mapping):
        layout = primitives_snapshot.get("native_layout") or primitives_snapshot.get("layout")
    if layout is None:
        layout = getattr(primitives_snapshot, "_last_native_layout", None)
    return layout if isinstance(layout, Mapping) else None


def _row_diagnostic(row_index: int, row: Mapping[str, Any]) -> dict[str, Any]:
    """Row-level diagnostic for snapshots without a layout (no question binding)."""
    row_id = row.get("id")
    return {
        "position": None,
        "row_index": row_index,
        "question_id": None,
        "resolved_value": None,
        "failure": None,
        "emitted": {
            "id": row_id if _is_token_id(row_id) else None,
            "text": _row_token_text(row),
        },
        "row_logprob": _finite_or_none(_entry_logprob(row)),
        "candidate_weights_raw": None,
        "candidate_mass_raw": None,
        "mass_outside_candidates": None,
        "all_candidate_variants_captured": None,
        "argmax_label": None,
        "emitted_matches_argmax": None,
        "top_k": [_entry_diagnostic(entry) for entry in _row_entries(row)],
        "top_k_outside_candidates": [],
        "note": "no layout in snapshot: candidate slicing unavailable",
    }


def _position_diagnostic(
    position: Mapping[str, Any],
    row: Mapping[str, Any] | None,
    decisions: Mapping[str, Decision],
    failures: Sequence[ParseFailure],
) -> dict[str, Any]:
    """Per-question diagnostic: raw capture, sliced mass, argmax, emitted, value."""
    question_id = str(position.get("question_id") or _layout_question_id(position))
    diagnostic: dict[str, Any] = {
        "position": position.get("position"),
        "row_index": position.get("row_index"),
        "question_id": question_id,
        "kind": position.get("kind") or _layout_question_kind(position),
        "resolved_value": None,
        "confidence": None,
        "token_logprob": None,
        "failure": None,
        "emitted": None,
        "emitted_label": None,
        "candidate_weights_raw": None,
        "candidate_weights_normalized": None,
        "candidate_mass_raw": None,
        "mass_outside_candidates": None,
        "all_candidate_variants_captured": None,
        "argmax_label": None,
        "emitted_matches_argmax": None,
        "top_k": [],
        "top_k_outside_candidates": [],
    }
    decision = decisions.get(question_id)
    if decision is not None:
        diagnostic["resolved_value"] = decision.value
        diagnostic["confidence"] = decision.confidence
        diagnostic["token_logprob"] = decision.token_logprob
    else:
        diagnostic["failure"] = _failure_for(question_id, failures)
    if row is None:
        diagnostic["note"] = "no captured row at this index"
        return diagnostic

    native = _native_from_layout(position)
    row_id = row.get("id")
    diagnostic["emitted"] = {
        "id": row_id if _is_token_id(row_id) else None,
        "text": _row_token_text(row),
    }
    entries = _row_entries(row)
    text_fallback = not _is_token_id(row_id)
    id_to_label = {
        token_id: candidate.label
        for candidate in native.candidates
        for token_id in candidate.token_ids
    }
    text_to_label = {
        text: candidate.label for candidate in native.candidates for text in candidate.token_texts
    }
    diagnostic["emitted_label"] = _match_row_label(row, id_to_label, text_to_label)
    weights = _candidate_weights(entries, native, text_fallback=text_fallback)
    diagnostic["candidate_weights_raw"] = {
        candidate.label: weights[candidate.label] for candidate in native.candidates
    }
    diagnostic["candidate_mass_raw"] = sum(weights.values())
    diagnostic["candidate_weights_normalized"] = normalize_probabilities(weights)
    if any(weight > 0.0 for weight in weights.values()):
        diagnostic["argmax_label"] = max(weights, key=lambda label: weights[label])
    diagnostic["emitted_matches_argmax"] = (
        diagnostic["argmax_label"] == diagnostic["emitted_label"]
        if diagnostic["emitted_label"] is not None and diagnostic["argmax_label"] is not None
        else None
    )
    diagnostic["all_candidate_variants_captured"] = _all_variants_captured(
        entries, native, text_fallback=text_fallback
    )
    if diagnostic["all_candidate_variants_captured"]:
        # Every declared variant is somewhere in the captured top-K, so the
        # slice sums exactly and the remainder is the true outside mass.
        diagnostic["mass_outside_candidates"] = max(
            0.0, 1.0 - float(diagnostic["candidate_mass_raw"])
        )
    top_k = [_entry_diagnostic(entry) for entry in entries]
    diagnostic["top_k"] = top_k
    diagnostic["top_k_outside_candidates"] = [
        diagnostic_entry
        for raw, diagnostic_entry in zip(entries, top_k)
        if not _entry_is_candidate(raw, id_to_label, text_to_label, text_fallback=text_fallback)
    ]
    return diagnostic


def _entry_is_candidate(
    entry: Mapping[str, Any],
    id_to_label: Mapping[int, str],
    text_to_label: Mapping[str, str],
    *,
    text_fallback: bool,
) -> bool:
    """Whether one captured entry resolves to a declared candidate token.

    Same row-scoped matching rules as ``_candidate_weights``: id-bearing
    entries by id only, id-less entries by text only under ``text_fallback``.
    """
    token_id = entry.get("id")
    if _is_token_id(token_id):
        return token_id in id_to_label
    if not text_fallback:
        return False
    text = _entry_token_text(entry)
    return text is not None and text in text_to_label


def _failure_for(question_id: str, failures: Sequence[ParseFailure]) -> dict[str, str] | None:
    needle = f"question {question_id!r}"
    for failure in failures:
        if needle in failure.detail:
            return {"reason": failure.reason, "detail": failure.detail}
    return None


def _layout_question_id(position: Mapping[str, Any]) -> str:
    question = position.get("question")
    return str(question.get("id")) if isinstance(question, Mapping) else ""


def _layout_question_kind(position: Mapping[str, Any]) -> str | None:
    question = position.get("question")
    return str(question.get("kind")) if isinstance(question, Mapping) else None


def _native_from_layout(position: Mapping[str, Any]) -> _NativeQuestion:
    """Rebuild the runner's token binding for one layout position."""
    raw_question = position.get("question")
    question_data = raw_question if isinstance(raw_question, Mapping) else {}
    question = Question(
        id=str(question_data.get("id", "")),
        kind=question_data.get("kind"),
        text=str(question_data.get("text", "")),
        options=tuple(question_data.get("options", ()) or ()),
        levels=tuple(question_data.get("levels", ()) or ()),
        criteria=tuple(question_data.get("criteria", ()) or ()),
    )
    candidates = tuple(
        _NativeCandidate(
            label=str(candidate.get("label", "")),
            token_ids=tuple(int(token_id) for token_id in candidate.get("token_ids", ()) or ()),
            token_texts=tuple(str(text) for text in candidate.get("token_texts", ()) or ()),
            key=str(candidate["key"]) if candidate.get("key") is not None else None,
        )
        for candidate in position.get("candidates", ()) or ()
        if isinstance(candidate, Mapping)
    )
    cue_token_ids = tuple(int(token_id) for token_id in position.get("cue_token_ids", ()) or ())
    return _NativeQuestion(
        question=question,
        candidates=candidates,
        cue_token_ids=cue_token_ids,
    )


def _all_variants_captured(
    entries: Sequence[Mapping[str, Any]],
    native: _NativeQuestion,
    *,
    text_fallback: bool,
) -> bool:
    """True when every declared candidate variant appears in the captured rows.

    Mirrors ``_candidate_weights``' matching rules: id-bearing entries resolve
    by id only; id-less entries resolve by exact text only when
    ``text_fallback`` is set.
    """
    captured_ids = {entry.get("id") for entry in entries if _is_token_id(entry.get("id"))}
    captured_texts = {text for entry in entries if (text := _entry_token_text(entry)) is not None}
    for candidate in native.candidates:
        for token_id, token_text in zip(candidate.token_ids, candidate.token_texts):
            if token_id in captured_ids:
                continue
            if text_fallback and token_text in captured_texts:
                continue
            return False
    return True


def _entry_diagnostic(entry: Mapping[str, Any]) -> dict[str, Any]:
    """One raw top-k entry, JSON-safe."""
    token_id = entry.get("id")
    return {
        "id": token_id if _is_token_id(token_id) else None,
        "text": _entry_token_text(entry),
        "logprob": _finite_or_none(_entry_logprob(entry)),
        "weight": _finite_or_none(_entry_weight(entry)),
    }


def _finite_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _rows_from_meta(meta: Any) -> list[Mapping[str, Any]]:
    """Extract the ``completion_probabilities`` rows from an inference meta dict.

    Missing / malformed meta yields ``[]``, which every position then reports
    as a typed failure — never as a default answer.
    """
    if not isinstance(meta, Mapping):
        return []
    rows = meta.get("completion_probabilities")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def _row_token_text(row: Mapping[str, Any]) -> str | None:
    """The emitted token's text: ``token`` (v9) or ``content`` (legacy)."""
    for key in ("token", "content"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    return None


def _row_entries(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The row's top-probability entries: logprob- or prob-keyed, any generation."""
    for key in ("top_logprobs", "top_probs", "probs"):
        entries = row.get(key)
        if isinstance(entries, list):
            return [entry for entry in entries if isinstance(entry, Mapping)]
    return []


def _entry_token_text(entry: Mapping[str, Any]) -> str | None:
    """The candidate token's text: ``token`` (v9) or ``tok_str`` (legacy)."""
    for key in ("token", "tok_str"):
        value = entry.get(key)
        if isinstance(value, str):
            return value
    return None


def _entry_weight(entry: Mapping[str, Any]) -> float | None:
    """One entry's linear probability weight (``prob`` directly, ``logprob`` exp'd).

    Non-finite or unconvertible values yield ``None`` (the entry is skipped),
    never a fabricated ``0.0``-vs-missing ambiguity.
    """
    probability = entry.get("prob")
    if isinstance(probability, (int, float)) and not isinstance(probability, bool):
        value = float(probability)
        return value if math.isfinite(value) else None
    logprob = entry.get("logprob")
    if isinstance(logprob, (int, float)) and not isinstance(logprob, bool):
        value = float(logprob)
        if not math.isfinite(value):
            return None
        try:
            return math.exp(value)
        except OverflowError:
            return None
    return None


def _entry_logprob(entry: Mapping[str, Any]) -> float | None:
    """One entry's log-probability, re-derived from ``prob`` when necessary."""
    logprob = entry.get("logprob")
    if isinstance(logprob, (int, float)) and not isinstance(logprob, bool):
        value = float(logprob)
        if math.isfinite(value):
            return value
        return None
    probability = _entry_weight(entry)
    if probability is None or probability <= 0.0:
        return None
    return math.log(probability)


def _candidate_weights(
    entries: Sequence[Mapping[str, Any]],
    native: _NativeQuestion,
    *,
    text_fallback: bool,
) -> dict[str, float]:
    """Slice entries to the declared candidates; ids first, then text.

    Every declared label gets an entry (``0.0`` when its token is absent from
    the captured top-K, i.e. below the capture cutoff), so the caller's
    renormalization always covers the full candidate set. Entries carrying an
    ``id`` are matched by id only (an id outside the candidate set is ignored,
    never reinterpreted as text); each distinct id contributes once. Ids-less
    entries match by exact token text against the variant strings sent to
    ``/tokenize`` (each distinct text contributes once) ONLY when
    ``text_fallback`` is set, i.e. when the enclosing row carries no id. A
    label with several single-token variants has its captured weights summed.
    """
    id_to_label = {
        token_id: candidate.label
        for candidate in native.candidates
        for token_id in candidate.token_ids
    }
    text_to_label = {
        text: candidate.label for candidate in native.candidates for text in candidate.token_texts
    }
    weights = {candidate.label: 0.0 for candidate in native.candidates}
    seen_ids: set[int] = set()
    seen_texts: set[str] = set()
    for entry in entries:
        weight = _entry_weight(entry)
        if weight is None:
            continue
        token_id = entry.get("id")
        if _is_token_id(token_id):
            if token_id in seen_ids:
                continue
            seen_ids.add(token_id)
            label = id_to_label.get(token_id)
            if label is not None:
                weights[label] += weight
            continue
        if not text_fallback:
            continue
        text = _entry_token_text(entry)
        if text is None or text in seen_texts:
            continue
        seen_texts.add(text)
        label = text_to_label.get(text)
        if label is not None:
            weights[label] += weight
    return weights


def _logprob_for(
    entries: Sequence[Mapping[str, Any]],
    candidate: _NativeCandidate,
    *,
    text_fallback: bool,
) -> float | None:
    """Raw log-probability of a candidate's token in the captured row, if present.

    A label with several single-token variants returns the first listed match
    (llama.cpp lists top entries in descending probability), i.e. the most
    probable captured variant. Text matching follows the same row-scoped
    ``text_fallback`` rule as ``_candidate_weights``.
    """
    token_ids = set(candidate.token_ids)
    token_texts = set(candidate.token_texts)
    for entry in entries:
        token_id = entry.get("id")
        if _is_token_id(token_id):
            if token_id in token_ids:
                return _entry_logprob(entry)
            continue
        if not text_fallback:
            continue
        text = _entry_token_text(entry)
        if text is not None and text in token_texts:
            return _entry_logprob(entry)
    return None


def _is_token_id(value: Any) -> bool:
    """True for a real integer token id (``bool`` is not a token id)."""
    return isinstance(value, int) and not isinstance(value, bool)
