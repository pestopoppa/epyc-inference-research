"""Unit tests for the TD-1b/TD-1c native candidate-scoring path.

Mirrors the fake-primitives pattern of ``tests/unit/test_typed_decisions.py``
(one canned response, captured call kwargs) but adds the instance-level
``_last_inference_meta`` that carries synthetic ``completion_probabilities``
rows, plus a fake tokenizer seam.

The TD-1c generation layout is ``cue-0 answer-0 cue-1 answer-1 ...``: each
question's cue is replayed as fixed tokens so every answer token is generated
immediately after its own question. The tests therefore build metas whose rows
are 1:1 with generated tokens (cue rows included) via ``_build_meta``, and the
answer row indices are computed from the tokenizer's cue lengths.

The primary row shape pinned here is the production-consolidated-v9
``/completion`` shape emitted by
``tools/server/server-task.cpp::probs_vector_to_json`` with
``post_sampling_probs=false``::

    {"id": int, "token": str, "bytes": [int],
     "logprob": float, "top_logprobs": [{"id", "token", "bytes", "logprob"}, ...]}

The legacy ``{"content", "probs": [{"tok_str", "prob"}]}`` shape and the
``top_probs`` linear-probability variant (both WITHOUT token ids) are pinned
alongside it: they exercise the documented text fallback. No model/server
call.

The fake tokenizer makes both the bare and the space-prefixed form of every
fixture label a single token with distinct ids — as llama.cpp tokenizers
commonly do — so eligibility, grammar construction and id-based slicing are
all exercised against realistic tokenizer output. Texts outside the fake
vocabulary fall back to one id per character (multi-token) unless the test
passes a single-id ``fallback`` for short, pinned cues.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from src.typed_decisions import (
    DecisionResult,
    Question,
    QuestionKind,
    run_typed_decisions,
    run_typed_decisions_native,
)
from src.typed_decisions.confidence import choice_confidence, score_confidence
from src.typed_decisions.native import (
    REASON_NATIVE_TOKENIZER_UNAVAILABLE,
    REASON_NATIVE_UNKNOWN_CANDIDATE,
    REASON_NATIVE_UNSUPPORTED_CANDIDATES,
    REASON_TRANSPORT_ERROR,
    CueStyle,
    _cue_text,
    _DECODE_SEED,
    build_native_prompt,
    native_diagnostics,
    run_typed_decisions_native_parallel,
)

ROLE = "worker"
STATE = "unit-test state: native candidate scoring over a three-question catalogue."

# A single token id substituted for every text absent from a fake vocabulary
# when a test passes ``fallback=(_CUE_TOKEN,)``: keeps cues one token long and
# the pinned grammar strings short.
_CUE_TOKEN = 990


# ── fake tokenizer ────────────────────────────────────────────────────────


def _default_vocab() -> dict[str, tuple[int, ...]]:
    """Bare and space-prefixed single-token ids for every fixture label."""
    labels = (
        "red",
        "blue",
        "green",
        "0",
        "1",
        "2",
        "3",
        "true",
        "false",
        "yes",
        "no",
        "purple",
        "x",
        "y",
        "fal",
    )
    vocab: dict[str, tuple[int, ...]] = {}
    for index, label in enumerate(labels):
        vocab[label] = (1000 + 2 * index,)
        vocab[" " + label] = (1001 + 2 * index,)
    return vocab


_DEFAULT_VOCAB = _default_vocab()
_DEFAULT_IDS = {text: ids[0] for text, ids in _DEFAULT_VOCAB.items()}


class _FakeTokenizer:
    """Text -> token ids with a recorded call log.

    ``failing`` texts return ``None`` (the seam's "no answer" signal). Texts
    absent from ``vocab`` tokenize to ``fallback`` when one is given, else to
    one id per character, so multi-token candidates are representable without
    special-casing.
    """

    def __init__(
        self,
        vocab: Mapping[str, tuple[int, ...]] | None = None,
        failing: Sequence[str] = (),
        fallback: Sequence[int] | None = None,
    ) -> None:
        self.vocab = {text: tuple(ids) for text, ids in (vocab or _DEFAULT_VOCAB).items()}
        self.failing = set(failing)
        self.fallback = tuple(fallback) if fallback is not None else None
        self.calls: list[str] = []

    def __call__(self, text: str) -> list[int] | None:
        self.calls.append(text)
        if text in self.failing:
            return None
        if text in self.vocab:
            return list(self.vocab[text])
        if self.fallback is not None:
            return list(self.fallback)
        return [ord(char) for char in text]


class _FakePrimitives:
    """Canned-response stand-in for ``LLMPrimitives`` with inference meta.

    ``responses`` is consumed in order; the last entry repeats. The meta is
    instance-level exactly as in ``src/llm_primitives/inference.py``, so the
    tests can also prove that a transport failure does not fall back to stale
    probability rows. No ``_backends`` / ``server_urls`` / ``_tokenizer`` is
    configured, so the default tokenizer resolver must fail closed.
    """

    def __init__(self, responses: str | Sequence[str], meta: dict[str, Any] | None = None):
        if isinstance(responses, str):
            responses = [responses]
        self.responses = list(responses)
        self.calls: list[dict] = []
        self._last_inference_meta = meta if meta is not None else {}

    def llm_call(self, prompt: str, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index]


# ── fixtures ──────────────────────────────────────────────────────────────

CHOICE = Question(
    id="choice",
    kind=QuestionKind.CHOICE,
    text="Pick a colour.",
    options=("red", "blue", "green"),
)
SCORE = Question(
    id="score",
    kind=QuestionKind.SCORE,
    text="Rate the change.",
    levels=(0, 1, 2, 3),
)
NOUL = Question(id="noul", kind=QuestionKind.NOUL, text="Ship it?")
QUESTIONS = (CHOICE, SCORE, NOUL)

MULTI_TOKEN = Question(
    id="multi",
    kind=QuestionKind.CHOICE,
    text="Pick a multi-token option.",
    options=("do the thing", "do nothing"),
)
SINGLE_TOKEN = Question(
    id="single",
    kind=QuestionKind.CHOICE,
    text="Pick a single-token option.",
    options=("yes", "no"),
)


# ── Production-v9 row shape ───────────────────────────────────────────────


def _v9_row(
    emitted: str,
    logprob: float,
    top: Sequence[tuple[str, float]],
    *,
    ids: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """One ``completion_probabilities`` row in the pinned v9 shape."""
    table = _DEFAULT_IDS if ids is None else ids

    def token_id(text: str) -> int:
        return table[text]

    row = {
        "id": token_id(emitted),
        "token": emitted,
        "bytes": list(emitted.encode("utf-8")),
        "logprob": logprob,
        "top_logprobs": [
            {
                "id": token_id(token),
                "token": token,
                "bytes": list(token.encode("utf-8")),
                "logprob": token_logprob,
            }
            for token, token_logprob in top
        ],
    }
    return row


def _cue_length(
    tokenizer: _FakeTokenizer,
    question: Question,
    cue_style: CueStyle | str = CueStyle.FULL,
) -> int:
    return len(tokenizer(_cue_text(question, cue_style)))


def _build_meta(
    questions: Sequence[Question],
    answer_rows: Sequence[Mapping[str, Any]],
    tokenizer: _FakeTokenizer,
    cue_style: CueStyle | str = CueStyle.FULL,
) -> dict[str, Any]:
    """A meta whose rows are 1:1 with the TD-1c generated tokens.

    ``answer_rows[i]`` is placed at question ``i``'s answer index; the cue
    tokens before it become opaque filler rows (the runner never slices them).
    ``cue_style`` selects the cue whose tokenized length the filler matches.
    """
    rows: list[Mapping[str, Any]] = []
    for question, answer_row in zip(questions, answer_rows):
        for _ in range(_cue_length(tokenizer, question, cue_style)):
            rows.append({"id": _CUE_TOKEN, "token": "", "logprob": 0.0, "top_logprobs": []})
        rows.append(answer_row)
    return {"completion_probabilities": rows}


def _main_answer_rows() -> list[dict[str, Any]]:
    """Three rows: choice (0.5/0.2/0.1), score (0.6/0.25/0.1), noul (0.9/0.1)."""
    return [
        _v9_row(
            "blue",
            math.log(0.5),
            [("blue", math.log(0.5)), ("red", math.log(0.2)), ("green", math.log(0.1))],
        ),
        _v9_row(
            "2",
            math.log(0.6),
            [("2", math.log(0.6)), ("1", math.log(0.25)), ("0", math.log(0.1))],
        ),
        _v9_row(
            "true",
            math.log(0.9),
            [("true", math.log(0.9)), ("false", math.log(0.1))],
        ),
    ]


def _main_meta(tokenizer: _FakeTokenizer | None = None) -> dict[str, Any]:
    tokenizer = tokenizer or _FakeTokenizer()
    return _build_meta(QUESTIONS, _main_answer_rows(), tokenizer)


def _by_id(result: DecisionResult) -> dict[str, Any]:
    return {decision.question_id: decision for decision in result.decisions}


def _run(primitives: _FakePrimitives, questions: Sequence[Question], **kwargs):
    return run_typed_decisions_native(
        primitives,
        state=STATE,
        questions=questions,
        role=ROLE,
        tokenize_fn=kwargs.pop("tokenize_fn", _FakeTokenizer()),
        **kwargs,
    )


_MAIN_GRAMMAR = (
    "root ::= cue-0 answer-0 cue-1 answer-1 cue-2 answer-2\n"
    "cue-0 ::= <[990]>\n"
    "answer-0 ::= <[1000]> | <[1001]> | <[1002]> | <[1003]> | <[1004]> | <[1005]>\n"
    "cue-1 ::= <[990]>\n"
    "answer-1 ::= <[1006]> | <[1007]> | <[1008]> | <[1009]> | <[1010]> | <[1011]> | <[1012]> | <[1013]>\n"
    "cue-2 ::= <[990]>\n"
    "answer-2 ::= <[1014]> | <[1015]> | <[1018]> | <[1019]> | <[1016]> | <[1017]> | <[1020]> | <[1021]>\n"
)


def _single_cue_run(questions: Sequence[Question]):
    """A run whose cue texts are one token (``_CUE_TOKEN``) and answers follow."""
    tokenizer = _FakeTokenizer(fallback=(_CUE_TOKEN,))
    answer_rows = _main_answer_rows()[: len(questions)]
    primitives = _FakePrimitives("", meta=_build_meta(questions, answer_rows, tokenizer))
    return _run(primitives, questions, tokenize_fn=tokenizer), primitives, tokenizer


# ── 1. Pinned row shape, slicing, renormalization, argmax ─────────────────


class TestNativeSlicing:
    def test_production_v9_row_shape_is_pinned(self):
        tokenizer = _FakeTokenizer(fallback=(_CUE_TOKEN,))
        row = _v9_row(
            "blue",
            math.log(0.5),
            [("blue", math.log(0.5)), ("red", math.log(0.2)), ("green", math.log(0.1))],
        )

        assert set(row) == {"id", "token", "bytes", "logprob", "top_logprobs"}
        assert isinstance(row["id"], int)
        assert isinstance(row["bytes"], list)
        for entry in row["top_logprobs"]:
            assert set(entry) == {"id", "token", "bytes", "logprob"}

        question = Question(
            id="colour",
            kind=QuestionKind.CHOICE,
            text="Pick a colour.",
            options=("red", "blue", "green"),
        )
        primitives = _FakePrimitives("", meta=_build_meta([question], [row], tokenizer))

        result = _run(primitives, [question], tokenize_fn=tokenizer)

        assert len(result.decisions) == 1
        assert result.failures == ()

    def test_slice_is_renormalized_and_argmax_is_reported(self):
        tokenizer = _FakeTokenizer(fallback=(_CUE_TOKEN,))
        primitives = _FakePrimitives("", meta=_main_meta(tokenizer))

        result = _run(primitives, QUESTIONS, tokenize_fn=tokenizer)

        assert isinstance(result, DecisionResult)
        assert result.mode == "native"
        assert result.raw_text == ""
        assert result.failures == ()
        decisions = _by_id(result)
        assert set(decisions) == {"choice", "score", "noul"}

        # choice: candidate weights 0.5/0.2/0.1 (sum 0.8) -> renormalized.
        choice = decisions["choice"]
        assert choice.value == "blue"
        assert dict(choice.probabilities) == {
            "red": pytest.approx(0.25),
            "blue": pytest.approx(0.625),
            "green": pytest.approx(0.125),
        }
        assert sum(choice.probabilities.values()) == pytest.approx(1.0)
        assert choice.confidence == pytest.approx(0.4375)
        assert choice.token_logprob == pytest.approx(math.log(0.5))
        assert choice.mode == "native"

        # score: level 3 absent from the top-K -> explicit zero, no default.
        score = decisions["score"]
        assert score.value == 2
        expected_score = {0: 0.1 / 0.95, 1: 0.25 / 0.95, 2: 0.6 / 0.95, 3: 0.0}
        assert dict(score.probabilities) == {
            level: pytest.approx(probability) for level, probability in expected_score.items()
        }
        assert sum(score.probabilities.values()) == pytest.approx(1.0)
        assert score.confidence == pytest.approx(score_confidence(expected_score))
        assert score.token_logprob == pytest.approx(math.log(0.6))

        # noul: bool value, "true"/"false" labels.
        noul = decisions["noul"]
        assert noul.value is True
        assert dict(noul.probabilities) == {
            "true": pytest.approx(0.9),
            "false": pytest.approx(0.1),
        }
        assert noul.confidence == pytest.approx(choice_confidence({"true": 0.9, "false": 0.1}))
        assert noul.token_logprob == pytest.approx(math.log(0.9))

    def test_legacy_content_probs_shape_is_accepted(self):
        tokenizer = _FakeTokenizer(fallback=(_CUE_TOKEN,))
        answer_row = {
            "content": "red",
            "probs": [
                {"tok_str": "red", "prob": 0.6},
                {"tok_str": "blue", "prob": 0.3},
                {"tok_str": "green", "prob": 0.1},
            ],
        }
        question = Question(
            id="colour",
            kind=QuestionKind.CHOICE,
            text="Pick a colour.",
            options=("red", "blue", "green"),
        )
        primitives = _FakePrimitives("", meta=_build_meta([question], [answer_row], tokenizer))

        result = _run(primitives, [question], tokenize_fn=tokenizer)

        decision = result.decisions[0]
        assert decision.value == "red"
        assert dict(decision.probabilities) == {
            "red": pytest.approx(0.6),
            "blue": pytest.approx(0.3),
            "green": pytest.approx(0.1),
        }
        assert decision.token_logprob == pytest.approx(math.log(0.6))

    def test_post_sampling_top_probs_shape_is_accepted(self):
        tokenizer = _FakeTokenizer(fallback=(_CUE_TOKEN,))
        answer_row = {
            "token": "false",
            "prob": 0.7,
            "top_probs": [
                {"token": "false", "prob": 0.7},
                {"token": "true", "prob": 0.3},
            ],
        }
        question = Question(id="flag", kind=QuestionKind.NOUL, text="Ship it?")
        primitives = _FakePrimitives("", meta=_build_meta([question], [answer_row], tokenizer))

        result = _run(primitives, [question], tokenize_fn=tokenizer)

        decision = result.decisions[0]
        assert decision.value is False
        assert dict(decision.probabilities) == {
            "true": pytest.approx(0.3),
            "false": pytest.approx(0.7),
        }
        assert decision.token_logprob == pytest.approx(math.log(0.7))

    def test_post_sampling_probs_id_bearing_v1_shaped_row_covers_the_candidate(self):
        """TD-1d.2: the /v1 lane's ``logprobs.content`` row is id-bearing too
        (verified against ``tools/server/server-task.cpp`` at the frozen
        commit: both ``/completion`` and ``/v1/chat/completions`` build their
        row from the identical ``probs_vector_to_json``), so this is NOT the
        idless-row case — it is the id-bearing case, but with
        ``post_sampling_probs=True``'s guarantee that the top-k can only ever
        contain still-legal (grammar-masked) candidates. Contrast with
        ``test_no_candidate_token_in_the_capture_is_failure_not_uniform``,
        which pins the OLD (``post_sampling_probs=false``) raw-logit failure
        mode on an otherwise-identical id-bearing row.
        """
        tokenizer = _FakeTokenizer(fallback=(_CUE_TOKEN,))
        answer_rows = _main_answer_rows()
        answer_rows[0] = {
            "id": _DEFAULT_IDS["blue"],
            "token": "blue",
            "prob": 0.7,
            "top_probs": [
                {"id": _DEFAULT_IDS["red"], "token": "red", "prob": 0.2},
                {"id": _DEFAULT_IDS["blue"], "token": "blue", "prob": 0.7},
                {"id": _DEFAULT_IDS["green"], "token": "green", "prob": 0.1},
            ],
        }
        primitives = _FakePrimitives("", meta=_build_meta(QUESTIONS, answer_rows, tokenizer))

        result = _run(primitives, QUESTIONS, tokenize_fn=tokenizer)

        decisions = _by_id(result)
        assert "choice" in decisions
        choice = decisions["choice"]
        assert choice.value == "blue"
        assert dict(choice.probabilities) == {
            "red": pytest.approx(0.2),
            "blue": pytest.approx(0.7),
            "green": pytest.approx(0.1),
        }
        assert sum(choice.probabilities.values()) == pytest.approx(1.0)


# ── 2. Call contract: cue/answer grammar, n_probs, n_tokens, determinism ──


class TestNativeCallContract:
    def test_captured_kwargs_carry_cue_grammar_and_probability_capture(self):
        tokenizer = _FakeTokenizer(fallback=(_CUE_TOKEN,))
        primitives = _FakePrimitives("", meta=_main_meta(tokenizer))

        _run(primitives, QUESTIONS, tokenize_fn=tokenizer)

        assert len(primitives.calls) == 1
        call = primitives.calls[0]
        assert call["role"] == ROLE
        # 3 one-token cues + one answer token per question.
        assert call["n_tokens"] == 6
        # score/noul contribute the most alternatives: 8 + buffer 4.
        assert call["n_probs"] == 12
        # TD-1d.2: without this, the server's n_probs top-k is the raw
        # pre-grammar distribution, which frequently omits a heavily
        # grammar-narrowed position's declared candidates entirely — see
        # the module docstring's "Probability semantics" section.
        assert call["post_sampling_probs"] is True
        assert call["temperature"] == 0.0
        assert call["seed"] == 0
        assert "json_schema" not in call
        assert call["grammar"] == _MAIN_GRAMMAR

    def test_n_probs_accounts_for_tokenized_alternatives(self):
        # yes: bare + spaced single; no: bare single only -> 3 alternatives.
        vocab = {"yes": (41,), " yes": (42,), "no": (43,), " no": (900, 901)}
        tokenizer = _FakeTokenizer(vocab, fallback=(_CUE_TOKEN,))
        answer_row = _v9_row(
            "yes",
            math.log(0.8),
            [("yes", math.log(0.8)), ("no", math.log(0.2))],
            ids={"yes": 41, "no": 43},
        )
        primitives = _FakePrimitives("", meta=_build_meta([SINGLE_TOKEN], [answer_row], tokenizer))

        _run(primitives, [SINGLE_TOKEN], tokenize_fn=tokenizer)

        call = primitives.calls[0]
        assert call["n_probs"] == 7  # max alternatives (3) + buffer 4
        assert call["n_tokens"] == 2  # one cue token + one answer token
        assert call["grammar"] == (
            "root ::= cue-0 answer-0\ncue-0 ::= <[990]>\nanswer-0 ::= <[41]> | <[42]> | <[43]>\n"
        )

    def test_prompt_is_deterministic_and_hashed(self):
        tokenizer = _FakeTokenizer(fallback=(_CUE_TOKEN,))
        first = _FakePrimitives("", meta=_main_meta(tokenizer))
        second = _FakePrimitives("", meta=_main_meta(tokenizer))

        first_result = _run(first, QUESTIONS, tokenize_fn=tokenizer)
        second_result = _run(second, QUESTIONS, tokenize_fn=tokenizer)

        prompt = first.calls[0]["prompt"]
        assert prompt == second.calls[0]["prompt"]
        assert prompt == build_native_prompt(STATE, QUESTIONS)
        assert first_result.prompt_sha256 == second_result.prompt_sha256
        assert first_result.prompt_sha256 == hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        assert "candidates: red | blue | green" in prompt
        assert prompt.index("id=choice") < prompt.index("id=score") < prompt.index("id=noul")

    def test_explicit_n_probs_is_forwarded_and_capped(self):
        tokenizer = _FakeTokenizer(fallback=(_CUE_TOKEN,))
        low = _FakePrimitives("", meta=_main_meta(tokenizer))
        high = _FakePrimitives("", meta=_main_meta(tokenizer))

        _run(low, QUESTIONS, tokenize_fn=tokenizer, n_probs=7)
        _run(high, QUESTIONS, tokenize_fn=tokenizer, n_probs=1000)

        assert low.calls[0]["n_probs"] == 7
        assert high.calls[0]["n_probs"] == 128

    def test_non_positive_n_probs_is_rejected(self):
        tokenizer = _FakeTokenizer(fallback=(_CUE_TOKEN,))
        with pytest.raises(ValueError, match="n_probs must be >= 1"):
            _run(
                _FakePrimitives("", meta=_main_meta(tokenizer)),
                QUESTIONS,
                tokenize_fn=tokenizer,
                n_probs=0,
            )

    def test_explicit_n_tokens_is_forwarded(self):
        tokenizer = _FakeTokenizer(fallback=(_CUE_TOKEN,))
        primitives = _FakePrimitives("", meta=_main_meta(tokenizer))

        _run(primitives, QUESTIONS, tokenize_fn=tokenizer, n_tokens=16)

        assert primitives.calls[0]["n_tokens"] == 16


# ── 3. Tokenizer-aware eligibility and grammar ────────────────────────────


class TestTokenizedEligibility:
    def test_each_candidate_binds_to_its_single_token_variant(self):
        # true: only the bare form is one token; false: only the spaced form.
        tokenizer = _FakeTokenizer({"true": (11,), " false": (23,)})
        cue_length = _cue_length(tokenizer, NOUL, CueStyle.FULL)
        answer_row = _v9_row(
            " false",
            math.log(0.7),
            [(" false", math.log(0.7)), ("true", math.log(0.3))],
            ids={" false": 23, "true": 11},
        )
        primitives = _FakePrimitives("", meta=_build_meta([NOUL], [answer_row], tokenizer))

        result = _run(primitives, [NOUL], tokenize_fn=tokenizer, cue_style=CueStyle.FULL)

        call = primitives.calls[0]
        assert call["grammar"] == (
            "root ::= cue-0 answer-0\n"
            f"cue-0 ::= {' '.join('<[%d]>' % ord(char) for char in _cue_text(NOUL, CueStyle.FULL))}\n"
            "answer-0 ::= <[11]> | <[23]>\n"
        )
        assert call["n_tokens"] == cue_length + 1
        assert call["n_probs"] == 6  # 2 alternatives + buffer 4
        decision = result.decisions[0]
        assert decision.value is False
        assert dict(decision.probabilities) == {
            "true": pytest.approx(0.3),
            "false": pytest.approx(0.7),
        }
        assert decision.token_logprob == pytest.approx(math.log(0.7))
        # The multi-token variants were probed, not guessed.
        assert " true" in tokenizer.calls
        assert "false" in tokenizer.calls
        assert "yes" in tokenizer.calls  # noul surface forms probed too

    def test_two_id_candidates_are_deferred_to_json_mode(self):
        vocab = {
            "true": (11,),
            " true": (12,),
            "false": (21,),
            " false": (22,),
            "lock": (31, 32),
            " lock": (33, 34),
            "unlock": (35, 36),
            " unlock": (37, 38),
        }
        tokenizer = _FakeTokenizer(vocab)
        choice = Question(
            id="lock", kind=QuestionKind.CHOICE, text="Lock it?", options=("lock", "unlock")
        )
        noul = Question(id="confirm", kind=QuestionKind.NOUL, text="Confirm?")
        cue_length = _cue_length(tokenizer, noul, CueStyle.FULL)
        answer_row = _v9_row(
            " true",
            math.log(0.8),
            [(" true", math.log(0.8)), ("false", math.log(0.2))],
            ids={" true": 12, "false": 21},
        )
        primitives = _FakePrimitives("", meta=_build_meta([noul], [answer_row], tokenizer))

        # TD-1b legacy path pinned: with TD-29 single-token keys on, the
        # multi-token set would be re-keyed (see TestSingleTokenKeys).
        result = _run(
            primitives,
            [choice, noul],
            tokenize_fn=tokenizer,
            cue_style=CueStyle.FULL,
            single_token_keys=False,
        )

        # Only the single-token noul question entered the native batch, in order.
        call = primitives.calls[0]
        assert call["n_tokens"] == cue_length + 1
        assert call["grammar"].startswith("root ::= cue-0 answer-0\n")
        assert "answer-0 ::= <[11]> | <[12]> | <[21]> | <[22]>" in call["grammar"]
        assert call["n_probs"] == 8
        assert [decision.question_id for decision in result.decisions] == ["confirm"]
        # id-based slicing: the spaced " true" variant carries the 0.8.
        confirm = result.decisions[0]
        assert confirm.value is True
        assert dict(confirm.probabilities) == {
            "true": pytest.approx(0.8),
            "false": pytest.approx(0.2),
        }
        assert confirm.token_logprob == pytest.approx(math.log(0.8))

        failure = result.failures[0]
        assert failure.reason == REASON_NATIVE_UNSUPPORTED_CANDIDATES
        assert "lock" in failure.detail and "unlock" in failure.detail
        assert "two tokens" not in failure.detail  # never a fabricated reason

    def test_token_id_collision_between_labels_is_unsupported(self):
        vocab = {"red": (5,), " red": (6,), "blue": (5,), " blue": (6,)}
        tokenizer = _FakeTokenizer(vocab)
        question = Question(
            id="colour", kind=QuestionKind.CHOICE, text="Pick.", options=("red", "blue")
        )
        primitives = _FakePrimitives("", meta=_main_meta(tokenizer))

        result = _run(primitives, [question], tokenize_fn=tokenizer)

        assert primitives.calls == []
        assert result.decisions == ()
        assert result.failures[0].reason == REASON_NATIVE_UNSUPPORTED_CANDIDATES
        assert "both tokenize" in result.failures[0].detail


# ── 4. Multi-token candidates -> JSON-mode fallback failures ──────────────


class TestMultiTokenFallback:
    def test_unsupported_question_is_excluded_and_recorded(self):
        questions = (SINGLE_TOKEN, MULTI_TOKEN, NOUL)
        tokenizer = _FakeTokenizer(fallback=(_CUE_TOKEN,))
        answer_rows = [
            _v9_row("yes", math.log(0.8), [("yes", math.log(0.8)), ("no", math.log(0.2))]),
            _v9_row(
                "false",
                math.log(0.6),
                [("false", math.log(0.6)), ("true", math.log(0.4))],
            ),
        ]
        primitives = _FakePrimitives(
            "",
            meta=_build_meta([SINGLE_TOKEN, NOUL], answer_rows, tokenizer),
        )

        result = _run(primitives, questions, tokenize_fn=tokenizer)

        assert [decision.question_id for decision in result.decisions] == ["single", "noul"]
        assert len(result.failures) == 1
        failure = result.failures[0]
        assert failure.reason == REASON_NATIVE_UNSUPPORTED_CANDIDATES
        assert "multi" in failure.detail
        assert "JSON mode" in failure.detail

        call = primitives.calls[0]
        assert call["n_tokens"] == 4  # two cue tokens + two answer tokens
        assert call["grammar"] == (
            "root ::= cue-0 answer-0 cue-1 answer-1\n"
            "cue-0 ::= <[990]>\n"
            "answer-0 ::= <[1018]> | <[1019]> | <[1020]> | <[1021]>\n"
            "cue-1 ::= <[990]>\n"
            "answer-1 ::= <[1014]> | <[1015]> | <[1018]> | <[1019]> | <[1016]> | "
            "<[1017]> | <[1020]> | <[1021]>\n"
        )
        assert "do the thing" not in call["grammar"]
        assert call["n_probs"] == 12  # max alternatives (8) + buffer 4

    def test_all_unsupported_questions_make_no_call_at_all(self):
        tokenizer = _FakeTokenizer(fallback=(_CUE_TOKEN,))
        primitives = _FakePrimitives("", meta=_main_meta(tokenizer))
        other_multi = Question(
            id="multi-2",
            kind=QuestionKind.CHOICE,
            text="Pick another multi-token option.",
            options=("do the other thing", "do nothing"),
        )
        questions = (MULTI_TOKEN, other_multi)

        result = _run(primitives, questions, tokenize_fn=tokenizer)

        assert primitives.calls == []
        assert result.decisions == ()
        assert result.raw_text == ""
        assert result.elapsed_ms == 0.0
        assert [failure.reason for failure in result.failures] == [
            REASON_NATIVE_UNSUPPORTED_CANDIDATES,
            REASON_NATIVE_UNSUPPORTED_CANDIDATES,
        ]
        assert len(result.prompt_sha256) == 64

    def test_partial_piece_label_is_ineligible_before_generation(self):
        # The TD-1a live failure: the grammar forced a partial piece ("fal")
        # of a two-token "false". With real tokenization the label is excluded
        # up front and the partial piece is never generated or accepted.
        vocab = {"true": (11,), " true": (12,), "false": (901, 902), " false": (903, 904)}
        tokenizer = _FakeTokenizer(vocab)
        primitives = _FakePrimitives("", meta=_main_meta(tokenizer))

        # TD-1b legacy path pinned (keys off); with keys on, the partial piece
        # is still never generated because the grammar binds whole key tokens.
        result = _run(primitives, [NOUL], tokenize_fn=tokenizer, single_token_keys=False)

        assert primitives.calls == []
        assert result.decisions == ()
        assert result.failures[0].reason == REASON_NATIVE_UNSUPPORTED_CANDIDATES
        assert "false" in result.failures[0].detail
        assert REASON_NATIVE_UNKNOWN_CANDIDATE not in {
            failure.reason for failure in result.failures
        }


# ── 5. Tokenizer unavailable -> fail closed, zero model calls ─────────────


class TestTokenizerUnavailable:
    def test_unresolvable_tokenizer_fails_every_question_without_a_call(self):
        primitives = _FakePrimitives("", meta=_main_meta())

        result = run_typed_decisions_native(primitives, state=STATE, questions=QUESTIONS, role=ROLE)

        assert primitives.calls == []
        assert result.decisions == ()
        assert result.raw_text == ""
        assert result.elapsed_ms == 0.0
        assert [failure.reason for failure in result.failures] == [
            REASON_NATIVE_TOKENIZER_UNAVAILABLE,
        ] * 3
        assert all(
            "no tokenizer could be resolved" in failure.detail for failure in result.failures
        )
        assert len(result.prompt_sha256) == 64

    def test_injected_tokenizer_returning_none_fails_closed(self):
        primitives = _FakePrimitives("", meta=_main_meta())

        result = _run(primitives, QUESTIONS, tokenize_fn=lambda text: None)

        assert primitives.calls == []
        assert result.decisions == ()
        assert [failure.reason for failure in result.failures] == [
            REASON_NATIVE_TOKENIZER_UNAVAILABLE,
        ] * 3
        assert all("tokenizer returned no ids" in failure.detail for failure in result.failures)

    def test_raising_tokenizer_is_treated_as_unavailable(self):
        def boom(text: str) -> list[int]:
            raise RuntimeError("tokenizer exploded")

        primitives = _FakePrimitives("", meta=_main_meta())

        result = _run(primitives, QUESTIONS, tokenize_fn=boom)

        assert primitives.calls == []
        assert [failure.reason for failure in result.failures] == [
            REASON_NATIVE_TOKENIZER_UNAVAILABLE,
        ] * 3

    def test_unavailable_question_does_not_block_eligible_questions(self):
        tokenizer = _FakeTokenizer(failing={"true", " true"})
        answer_row = _v9_row("yes", math.log(0.8), [("yes", math.log(0.8)), ("no", math.log(0.2))])
        cue_length = _cue_length(tokenizer, SINGLE_TOKEN, CueStyle.FULL)
        primitives = _FakePrimitives("", meta=_build_meta([SINGLE_TOKEN], [answer_row], tokenizer))

        result = _run(
            primitives, (SINGLE_TOKEN, NOUL), tokenize_fn=tokenizer, cue_style=CueStyle.FULL
        )

        assert len(primitives.calls) == 1
        call = primitives.calls[0]
        assert call["n_tokens"] == cue_length + 1
        assert "answer-0 ::= <[1018]> | <[1019]> | <[1020]> | <[1021]>" in call["grammar"]
        assert [decision.question_id for decision in result.decisions] == ["single"]
        assert [failure.reason for failure in result.failures] == [
            REASON_NATIVE_TOKENIZER_UNAVAILABLE
        ]
        assert "noul" in result.failures[0].detail

    def test_cue_tokenization_failure_fails_the_question_closed(self):
        # Candidates tokenize fine; the cue text is refused by the seam.
        tokenizer = _FakeTokenizer(failing={_cue_text(NOUL)})
        primitives = _FakePrimitives("", meta=_main_meta())

        result = _run(primitives, [NOUL], tokenize_fn=tokenizer)

        assert primitives.calls == []
        assert result.decisions == ()
        assert result.failures[0].reason == REASON_NATIVE_TOKENIZER_UNAVAILABLE
        assert "cue text" in result.failures[0].detail


# ── 6. Default resolver: role backend base URL -> /tokenize ───────────────


class _BackendConfig:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url


class _InnerBackend:
    def __init__(self, base_url: str) -> None:
        self.config = _BackendConfig(base_url)


class _WrappedBackend:
    """CachingBackend-shaped wrapper (``.backend.config.base_url``)."""

    def __init__(self, base_url: str) -> None:
        self.backend = _InnerBackend(base_url)


class _ResolverPrimitives(_FakePrimitives):
    def __init__(self, base_url: str, **kwargs: Any) -> None:
        super().__init__("", **kwargs)
        self._backends = {"worker": _WrappedBackend(base_url)}


def _install_fake_http_client(monkeypatch, responder):
    """Replace ``httpx.Client`` with a recorder; returns the created clients."""
    created: list[Any] = []

    class _Response:
        def __init__(self, payload: Any) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> Any:
            return self._payload

    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.requests: list[tuple[str, dict[str, Any]]] = []
            self.closed = False
            created.append(self)

        def post(self, url: str, json: dict[str, Any]):
            self.requests.append((url, json))
            return _Response(responder(json))

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr("src.typed_decisions.native.httpx.Client", _Client)
    return created


class TestDefaultTokenizerResolver:
    def test_role_backend_url_is_used_and_owned_client_is_closed(self, monkeypatch):
        ids = {
            "true": 7,
            " true": 8,
            "false": 9,
            " false": 10,
            "yes": 11,
            " yes": 12,
            "no": 13,
            " no": 14,
            _cue_text(NOUL): 990,
        }

        def responder(payload: dict[str, Any]) -> dict[str, Any]:
            text = payload.get("content", "")
            if text == "":
                return {"tokens": []}
            # Unknown texts (other cues) are one opaque token.
            return {"tokens": [ids[text]] if text in ids else [991]}

        created = _install_fake_http_client(monkeypatch, responder)
        meta = {
            "completion_probabilities": [
                {"id": 990, "token": "", "logprob": 0.0, "top_logprobs": []},
                _v9_row(
                    " false",
                    math.log(0.7),
                    [(" false", math.log(0.7)), ("true", math.log(0.3))],
                    ids=ids,
                ),
            ]
        }
        primitives = _ResolverPrimitives("http://test-host:8123", meta=meta)

        result = run_typed_decisions_native(
            primitives,
            state=STATE,
            questions=[NOUL],
            role=ROLE,
            # tokenize_fn omitted: the default resolver must do the work.
        )

        assert len(created) == 1
        client = created[0]
        assert client.requests[0][0] == "http://test-host:8123/tokenize"
        assert client.requests[0][1]["add_special"] is False
        assert client.requests[0][1]["content"] == ""  # probe
        probe_and_candidates = {request[1]["content"] for request in client.requests}
        assert {"", "true", " true", "false", " false"} <= probe_and_candidates
        assert _cue_text(NOUL) in probe_and_candidates
        assert client.closed is True  # the owned tokenizer is closed after the run

        assert primitives.calls[0]["grammar"] == (
            "root ::= cue-0 answer-0\n"
            "cue-0 ::= <[990]>\n"
            "answer-0 ::= <[7]> | <[8]> | <[11]> | <[12]> | <[9]> | <[10]> | <[13]> | <[14]>\n"
        )
        decision = result.decisions[0]
        assert decision.value is False
        assert dict(decision.probabilities) == {
            "true": pytest.approx(0.3),
            "false": pytest.approx(0.7),
        }

    def test_unreachable_tokenize_endpoint_fails_closed(self, monkeypatch):
        def responder(payload: dict[str, Any]) -> dict[str, Any]:
            raise ConnectionError("connection refused")

        created = _install_fake_http_client(monkeypatch, responder)
        primitives = _ResolverPrimitives("http://test-host:8123", meta=_main_meta())

        result = run_typed_decisions_native(primitives, state=STATE, questions=QUESTIONS, role=ROLE)

        assert primitives.calls == []  # zero model calls
        assert result.decisions == ()
        assert [failure.reason for failure in result.failures] == [
            REASON_NATIVE_TOKENIZER_UNAVAILABLE,
        ] * 3
        assert created[0].closed is True

    def test_server_urls_are_used_when_no_backend_is_registered(self, monkeypatch):
        ids = {"true": 7, " true": 8, "false": 9, " false": 10}

        def responder(payload: dict[str, Any]) -> dict[str, Any]:
            text = payload.get("content", "")
            if text == "":
                return {"tokens": []}
            return {"tokens": [ids[text]] if text in ids else [990]}

        created = _install_fake_http_client(monkeypatch, responder)
        primitives = _FakePrimitives("", meta=_main_meta())
        primitives.server_urls = {"worker": "http://fallback-host:9000,http://other:9001"}

        run_typed_decisions_native(primitives, state=STATE, questions=[NOUL], role=ROLE)

        assert created[0].requests[0][0] == "http://fallback-host:9000/tokenize"


# ── 7. Typed failure paths (never a default) ──────────────────────────────


class TestNativeFailurePaths:
    def test_emitted_token_outside_candidates_is_typed_failure(self):
        tokenizer = _FakeTokenizer(fallback=(_CUE_TOKEN,))
        answer_rows = _main_answer_rows()
        answer_rows[0] = _v9_row(
            "purple",
            math.log(0.9),
            [("purple", math.log(0.9)), ("red", math.log(0.1))],
        )
        primitives = _FakePrimitives("", meta=_build_meta(QUESTIONS, answer_rows, tokenizer))

        result = _run(primitives, QUESTIONS, tokenize_fn=tokenizer)

        decisions = _by_id(result)
        assert set(decisions) == {"score", "noul"}
        assert result.failures[0].reason == REASON_NATIVE_UNKNOWN_CANDIDATE
        assert "purple" in result.failures[0].detail
        assert "id=" in result.failures[0].detail

    def test_idless_row_outside_candidates_still_fails(self):
        tokenizer = _FakeTokenizer(fallback=(_CUE_TOKEN,))
        answer_row = {
            "content": "purple",
            "probs": [{"tok_str": "purple", "prob": 0.9}, {"tok_str": "red", "prob": 0.1}],
        }
        question = Question(
            id="colour", kind=QuestionKind.CHOICE, text="Pick.", options=("red", "blue")
        )
        primitives = _FakePrimitives("", meta=_build_meta([question], [answer_row], tokenizer))

        result = _run(primitives, [question], tokenize_fn=tokenizer)

        assert result.decisions == ()
        assert result.failures[0].reason == REASON_NATIVE_UNKNOWN_CANDIDATE
        assert "purple" in result.failures[0].detail

    def test_id_bearing_row_does_not_text_match_ids_less_entries(self):
        # The text fallback is scoped to rows that lack an id altogether.
        tokenizer = _FakeTokenizer(fallback=(_CUE_TOKEN,))
        answer_row = {
            "id": _DEFAULT_IDS["red"],
            "token": "red",
            "bytes": [114, 101, 100],
            "logprob": math.log(0.9),
            "probs": [{"tok_str": "red", "prob": 0.9}],
        }
        question = Question(
            id="colour", kind=QuestionKind.CHOICE, text="Pick.", options=("red", "blue")
        )
        primitives = _FakePrimitives("", meta=_build_meta([question], [answer_row], tokenizer))

        result = _run(primitives, [question], tokenize_fn=tokenizer)

        assert result.decisions == ()
        assert result.failures[0].reason == REASON_NATIVE_UNKNOWN_CANDIDATE
        assert "none of the declared candidate tokens" in result.failures[0].detail

    def test_no_candidate_token_in_the_capture_is_failure_not_uniform(self):
        tokenizer = _FakeTokenizer(fallback=(_CUE_TOKEN,))
        answer_rows = _main_answer_rows()
        answer_rows[0] = _v9_row(
            "blue",
            math.log(0.9),
            [("x", math.log(0.5)), ("y", math.log(0.5))],
        )
        primitives = _FakePrimitives("", meta=_build_meta(QUESTIONS, answer_rows, tokenizer))

        result = _run(primitives, QUESTIONS, tokenize_fn=tokenizer)

        assert "choice" not in _by_id(result)
        assert result.failures[0].reason == REASON_NATIVE_UNKNOWN_CANDIDATE
        assert "none of the declared candidate tokens" in result.failures[0].detail

    def test_missing_meta_yields_typed_failures_for_every_position(self):
        tokenizer = _FakeTokenizer(fallback=(_CUE_TOKEN,))
        primitives = _FakePrimitives("", meta=None)

        result = _run(primitives, QUESTIONS, tokenize_fn=tokenizer)

        assert result.decisions == ()
        assert len(result.failures) == 3
        assert all(failure.reason == REASON_NATIVE_UNKNOWN_CANDIDATE for failure in result.failures)
        assert all(
            "no completion_probabilities row" in failure.detail for failure in result.failures
        )

    def test_short_row_count_fails_only_the_missing_positions(self):
        tokenizer = _FakeTokenizer(fallback=(_CUE_TOKEN,))
        meta = _main_meta(tokenizer)
        # Keep cue-0, answer-0, cue-1, answer-1: the noul answer row is gone.
        meta["completion_probabilities"] = meta["completion_probabilities"][:4]
        primitives = _FakePrimitives("", meta=meta)

        result = _run(primitives, QUESTIONS, tokenize_fn=tokenizer)

        assert [decision.question_id for decision in result.decisions] == ["choice", "score"]
        assert len(result.failures) == 1
        assert result.failures[0].reason == REASON_NATIVE_UNKNOWN_CANDIDATE
        assert "noul" in result.failures[0].detail

    def test_transport_error_short_circuits_and_ignores_stale_meta(self):
        tokenizer = _FakeTokenizer(fallback=(_CUE_TOKEN,))
        primitives = _FakePrimitives("[ERROR: connection refused]", meta=_main_meta(tokenizer))

        result = _run(primitives, QUESTIONS, tokenize_fn=tokenizer)

        assert len(primitives.calls) == 1  # no retry
        assert result.decisions == ()
        assert result.failures[0].reason == "transport_error"
        assert result.raw_text == "[ERROR: connection refused]"

    def test_transport_failure_precedes_unsupported_question_failures(self):
        questions = (MULTI_TOKEN, SINGLE_TOKEN)
        tokenizer = _FakeTokenizer(fallback=(_CUE_TOKEN,))
        primitives = _FakePrimitives("[ERROR: timeout]", meta={"completion_probabilities": []})

        result = _run(primitives, questions, tokenize_fn=tokenizer)

        assert [failure.reason for failure in result.failures] == [
            "transport_error",
            REASON_NATIVE_UNSUPPORTED_CANDIDATES,
        ]


# ── 8. Prompt cueing (TD-1c) ──────────────────────────────────────────────


class TestPromptCueing:
    def test_prompt_blocks_have_explicit_answer_cue_and_echoed_candidates(self):
        prompt = build_native_prompt(STATE, QUESTIONS)

        for index, question in enumerate(QUESTIONS, start=1):
            assert f"{index}. id={question.id} kind={question.kind.value}" in prompt
            assert f"   question: {question.text}" in prompt
        assert "   Answer (one of: red, blue, green):" in prompt
        assert "   Answer (one of: 0, 1, 2, 3):" in prompt
        assert "   Answer (one of: true, false):" in prompt
        # One answer cue per block, in catalogue order.
        assert prompt.index("Answer (one of: red, blue, green):") < prompt.index(
            "Answer (one of: 0, 1, 2, 3):"
        )
        assert prompt.index("Answer (one of: 0, 1, 2, 3):") < prompt.index(
            "Answer (one of: true, false):"
        )

    def test_prompt_is_identical_for_plain_and_tokenized_questions(self):
        tokenizer = _FakeTokenizer(fallback=(_CUE_TOKEN,))
        from src.typed_decisions.native import _tokenize_catalogue

        native, failures = _tokenize_catalogue(QUESTIONS, tokenizer)

        assert failures == []
        assert build_native_prompt(STATE, native) == build_native_prompt(STATE, QUESTIONS)

    def test_build_native_prompt_rejects_unknown_entries(self):
        with pytest.raises(TypeError, match="expects Question or _NativeQuestion"):
            build_native_prompt(STATE, ["not-a-question"])  # type: ignore[list-item]

    def test_cue_replay_is_grammar_forced_between_answers(self):
        tokenizer = _FakeTokenizer(fallback=(_CUE_TOKEN,))
        primitives = _FakePrimitives("", meta=_main_meta(tokenizer))

        _run(primitives, QUESTIONS, tokenize_fn=tokenizer)

        grammar = primitives.calls[0]["grammar"]
        assert grammar.startswith("root ::= cue-0 answer-0 cue-1 answer-1 cue-2 answer-2\n")
        assert grammar.count("cue-0 ::=") == 1
        assert "cue-2 ::= <[990]>" in grammar


# ── 8b. Cue styles (TD-1d) ────────────────────────────────────────────────


class TestCueStyles:
    LONG_QUESTION = Question(
        id="s01",
        kind=QuestionKind.SCORE,
        text="Using the number priority rules, what is the priority of Item B?",
        levels=(0, 1, 2, 3),
    )

    def test_cue_style_is_a_str_enum(self):
        assert issubclass(CueStyle, str)
        assert CueStyle.FULL == "full"
        assert {style.value for style in CueStyle} == {"full", "short", "id_only"}

    def test_id_only_is_the_native_default_and_full_remains_selectable(self):
        # TD-6 flipped the native default to id_only (cue sweep 11.98x at
        # 15/16 agreement); full stays selectable.
        full_cue = "\nQ choice: Pick a colour.\nAnswer (one of: red, blue, green): "

        assert _cue_text(CHOICE) == "\nchoice: "
        assert _cue_text(CHOICE, CueStyle.ID_ONLY) == "\nchoice: "
        assert _cue_text(CHOICE, CueStyle.FULL) == full_cue
        assert _cue_text(CHOICE, "full") == full_cue

    def test_short_cue_is_id_plus_first_six_words(self):
        assert _cue_text(self.LONG_QUESTION, CueStyle.SHORT) == (
            "\nQ s01: Using the number priority rules, what\n"
        )

    def test_short_cue_keeps_a_short_question_whole(self):
        assert _cue_text(NOUL, CueStyle.SHORT) == "\nQ noul: Ship it?\n"

    def test_id_only_cue_is_the_minimal_delimiter(self):
        assert _cue_text(CHOICE, CueStyle.ID_ONLY) == "\nchoice: "
        assert _cue_text(SCORE, CueStyle.ID_ONLY) == "\nscore: "

    def test_plain_value_strings_are_accepted(self):
        assert _cue_text(CHOICE, "short") == _cue_text(CHOICE, CueStyle.SHORT)
        assert _cue_text(CHOICE, "id_only") == _cue_text(CHOICE, CueStyle.ID_ONLY)

    def test_unknown_cue_style_is_rejected(self):
        with pytest.raises(ValueError, match="unknown cue style"):
            _cue_text(CHOICE, "medium")

    def test_each_style_replays_its_own_cue_as_exact_token_terminals(self):
        tokenizer = _FakeTokenizer()
        primitives = _FakePrimitives(
            "", meta=_build_meta(QUESTIONS, _main_answer_rows(), tokenizer, CueStyle.ID_ONLY)
        )

        result = _run(primitives, QUESTIONS, tokenize_fn=tokenizer, cue_style=CueStyle.ID_ONLY)

        call = primitives.calls[0]
        expected_cue = " ".join(f"<[{ord(char)}]>" for char in _cue_text(CHOICE, CueStyle.ID_ONLY))
        assert f"cue-0 ::= {expected_cue}" in call["grammar"]
        assert (
            f"cue-2 ::= {' '.join(f'<[{ord(char)}]>' for char in _cue_text(NOUL, CueStyle.ID_ONLY))}"
            in call["grammar"]
        )
        # Exact-token terminals only (no quoted literals), and the answer cue
        # prose is prompt-side: the grammar replays just the id delimiter.
        assert '"' not in call["grammar"]
        assert "Answer (one of" not in call["grammar"]
        # The ID_ONLY cue text — and only it — was tokenized for the replay.
        assert _cue_text(CHOICE, CueStyle.ID_ONLY) in tokenizer.calls
        assert _cue_text(CHOICE, CueStyle.FULL) not in tokenizer.calls
        # The answer readout is unchanged.
        assert [decision.value for decision in result.decisions] == ["blue", 2, True]
        assert result.failures == ()

    def test_cue_style_changes_only_the_replayed_tokens(self):
        full_tokenizer = _FakeTokenizer()
        id_tokenizer = _FakeTokenizer()
        full = _FakePrimitives(
            "", meta=_build_meta(QUESTIONS, _main_answer_rows(), full_tokenizer, CueStyle.FULL)
        )
        id_only = _FakePrimitives(
            "", meta=_build_meta(QUESTIONS, _main_answer_rows(), id_tokenizer, CueStyle.ID_ONLY)
        )

        full_result = _run(full, QUESTIONS, tokenize_fn=full_tokenizer, cue_style="full")
        id_result = _run(id_only, QUESTIONS, tokenize_fn=id_tokenizer, cue_style="id_only")

        full_call, id_call = full.calls[0], id_only.calls[0]
        # Byte-identical prompt and hash: the numbered catalogue grounds every
        # style, so the cue replay is the only variable.
        assert full_call["prompt"] == id_call["prompt"]
        assert full_result.prompt_sha256 == id_result.prompt_sha256
        assert "   question: Pick a colour." in id_call["prompt"]
        assert "   candidates: red | blue | green" in id_call["prompt"]
        # Only the replayed cue terminals (and therefore the budget) differ.
        assert full_call["grammar"] != id_call["grammar"]
        assert id_call["n_tokens"] < full_call["n_tokens"]
        assert [decision.value for decision in full_result.decisions] == [
            decision.value for decision in id_result.decisions
        ]

    def test_layout_records_the_cue_style_and_per_style_cue_lengths(self):
        tokenizer = _FakeTokenizer()
        layouts: dict[str, dict[str, Any]] = {}
        for style in (CueStyle.FULL, CueStyle.SHORT, CueStyle.ID_ONLY):
            primitives = _FakePrimitives(
                "", meta=_build_meta(QUESTIONS, _main_answer_rows(), tokenizer, style)
            )
            _run(primitives, QUESTIONS, tokenize_fn=tokenizer, cue_style=style)
            layouts[style.value] = primitives._last_native_layout

        for style, layout in layouts.items():
            assert layout["cue_style"] == style
            assert layout["total_tokens"] == sum(
                position["cue_length"] for position in layout["positions"]
            ) + len(QUESTIONS)

        for index, question in enumerate(QUESTIONS):
            full_length = layouts["full"]["positions"][index]["cue_length"]
            short_length = layouts["short"]["positions"][index]["cue_length"]
            id_length = layouts["id_only"]["positions"][index]["cue_length"]
            assert full_length == len(_cue_text(question, CueStyle.FULL))
            assert short_length == len(_cue_text(question, CueStyle.SHORT))
            assert id_length == len(_cue_text(question, CueStyle.ID_ONLY))
            assert id_length < short_length < full_length

        # Answer rows are addressed by the style's own cue lengths.
        expected_rows = [
            index
            + sum(len(_cue_text(question, CueStyle.ID_ONLY)) for question in QUESTIONS[: index + 1])
            for index in range(len(QUESTIONS))
        ]
        assert [p["row_index"] for p in layouts["id_only"]["positions"]] == expected_rows
        assert layouts["id_only"]["positions"][0]["row_index"] == len(
            _cue_text(CHOICE, CueStyle.ID_ONLY)
        )


# ── 9. Natural noul surface forms ─────────────────────────────────────────


class TestNoulSurfaceForms:
    _VOCAB = {
        "true": (11,),
        " true": (12,),
        "yes": (13,),
        " yes": (14,),
        "false": (21,),
        " false": (22,),
        "no": (23,),
        " no": (24,),
    }

    def test_yes_and_no_are_bound_to_the_boolean_labels(self):
        tokenizer = _FakeTokenizer(self._VOCAB, fallback=(_CUE_TOKEN,))
        answer_row = _v9_row(
            "yes",
            math.log(0.6),
            [("yes", math.log(0.6)), ("no", math.log(0.4))],
            ids={text: ids[0] for text, ids in self._VOCAB.items()},
        )
        primitives = _FakePrimitives("", meta=_build_meta([NOUL], [answer_row], tokenizer))

        result = _run(primitives, [NOUL], tokenize_fn=tokenizer)

        assert primitives.calls[0]["grammar"] == (
            "root ::= cue-0 answer-0\n"
            "cue-0 ::= <[990]>\n"
            "answer-0 ::= <[11]> | <[12]> | <[13]> | <[14]> | <[21]> | <[22]> | <[23]> | <[24]>\n"
        )
        decision = result.decisions[0]
        assert decision.value is True
        assert dict(decision.probabilities) == {
            "true": pytest.approx(0.6),
            "false": pytest.approx(0.4),
        }
        assert decision.token_logprob == pytest.approx(math.log(0.6))

    def test_surface_mass_sums_into_the_declared_label(self):
        tokenizer = _FakeTokenizer(self._VOCAB, fallback=(_CUE_TOKEN,))
        answer_row = _v9_row(
            " true",
            math.log(0.5),
            [
                (" true", math.log(0.5)),
                ("yes", math.log(0.2)),
                ("false", math.log(0.2)),
                (" no", math.log(0.1)),
            ],
            ids={text: ids[0] for text, ids in self._VOCAB.items()},
        )
        primitives = _FakePrimitives("", meta=_build_meta([NOUL], [answer_row], tokenizer))

        result = _run(primitives, [NOUL], tokenize_fn=tokenizer)

        decision = result.decisions[0]
        assert decision.value is True
        # true = " true" 0.5 + yes 0.2; false = "false" 0.2 + " no" 0.1.
        assert dict(decision.probabilities) == {
            "true": pytest.approx(0.7),
            "false": pytest.approx(0.3),
        }

    def test_variant_split_can_make_the_summed_label_argmax_win_over_emitted(self):
        # Pinned consequence of summing surface forms: the emitted token is the
        # masked per-token argmax, the value is the argmax of the summed label
        # distribution, and diagnostics exposes the disagreement for audit.
        tokenizer = _FakeTokenizer(self._VOCAB, fallback=(_CUE_TOKEN,))
        answer_row = _v9_row(
            "false",
            math.log(0.15),
            [("false", math.log(0.15)), ("true", math.log(0.1)), ("yes", math.log(0.1))],
            ids={text: ids[0] for text, ids in self._VOCAB.items()},
        )
        primitives = _FakePrimitives("", meta=_build_meta([NOUL], [answer_row], tokenizer))

        result = _run(primitives, [NOUL], tokenize_fn=tokenizer)
        report = native_diagnostics(result, primitives)
        entry = report["positions"][0]

        decision = result.decisions[0]
        assert decision.value is True  # true 0.1 + yes 0.1 > false 0.15
        assert dict(decision.probabilities) == {
            "true": pytest.approx(0.2 / 0.35),
            "false": pytest.approx(0.15 / 0.35),
        }
        assert entry["emitted_label"] == "false"
        assert entry["argmax_label"] == "true"
        assert entry["emitted_matches_argmax"] is False


# ── 10. native_diagnostics ────────────────────────────────────────────────


class TestNativeDiagnostics:
    def _run_with_mixed_rows(self):
        tokenizer = _FakeTokenizer(fallback=(_CUE_TOKEN,))
        answer_rows = _main_answer_rows()
        # Noul answer row: every candidate variant is captured, plus two
        # outside tokens. true: 4x0.05=0.20, false: 4x0.10=0.40, outside: 0.40.
        answer_rows[2] = _v9_row(
            "false",
            math.log(0.4),
            [
                ("true", math.log(0.05)),
                (" true", math.log(0.05)),
                ("yes", math.log(0.05)),
                (" yes", math.log(0.05)),
                ("false", math.log(0.1)),
                (" false", math.log(0.1)),
                ("no", math.log(0.1)),
                (" no", math.log(0.1)),
                ("purple", math.log(0.3)),
                ("x", math.log(0.1)),
            ],
        )
        primitives = _FakePrimitives("", meta=_build_meta(QUESTIONS, answer_rows, tokenizer))
        result = _run(primitives, QUESTIONS, tokenize_fn=tokenizer)
        return result, primitives

    def test_primitives_snapshot_reports_per_question_raw_capture(self):
        result, primitives = self._run_with_mixed_rows()

        report = native_diagnostics(result, primitives)

        assert report["layout_present"] is True
        assert report["rows_captured"] == 6
        assert report["total_tokens_expected"] == 6
        assert report["n_probs"] == 12
        assert report["excluded"] == []
        by_question = {entry["question_id"]: entry for entry in report["positions"]}
        assert set(by_question) == {"choice", "score", "noul"}
        assert by_question["choice"]["row_index"] == 1
        assert by_question["score"]["row_index"] == 3
        assert by_question["noul"]["row_index"] == 5

        noul = by_question["noul"]
        assert noul["resolved_value"] is False
        assert noul["emitted"] == {"id": _DEFAULT_IDS["false"], "text": "false"}
        assert noul["emitted_label"] == "false"
        assert noul["candidate_weights_raw"] == {
            "true": pytest.approx(0.2),
            "false": pytest.approx(0.4),
        }
        assert noul["candidate_mass_raw"] == pytest.approx(0.6)
        assert noul["all_candidate_variants_captured"] is True
        assert noul["mass_outside_candidates"] == pytest.approx(0.4)
        assert noul["argmax_label"] == "false"
        assert noul["emitted_matches_argmax"] is True
        outside_ids = {entry["id"] for entry in noul["top_k_outside_candidates"]}
        assert outside_ids == {_DEFAULT_IDS["purple"], _DEFAULT_IDS["x"]}
        assert len(noul["top_k"]) == 10

        choice = by_question["choice"]
        assert choice["candidate_mass_raw"] == pytest.approx(0.8)
        assert choice["all_candidate_variants_captured"] is False
        assert choice["mass_outside_candidates"] is None
        assert choice["argmax_label"] == "blue"

    def test_bare_meta_snapshot_degrades_to_row_level(self):
        result, primitives = self._run_with_mixed_rows()

        report = native_diagnostics(result, primitives._last_inference_meta)

        assert report["layout_present"] is False
        assert len(report["positions"]) == report["rows_captured"] == 6
        assert all(entry["question_id"] is None for entry in report["positions"])
        assert report["positions"][5]["emitted"]["text"] == "false"
        assert any("no native layout" in note for note in report["notes"])

    def test_mapping_snapshot_with_layout_binds_questions(self):
        result, primitives = self._run_with_mixed_rows()

        report = native_diagnostics(
            result,
            {
                "native_layout": primitives._last_native_layout,
                "meta": primitives._last_inference_meta,
            },
        )

        assert report["layout_present"] is True
        assert {entry["question_id"] for entry in report["positions"]} == {
            "choice",
            "score",
            "noul",
        }

    def test_diagnostics_classifies_idless_entries_by_text(self):
        tokenizer = _FakeTokenizer(fallback=(_CUE_TOKEN,))
        answer_row = {
            "content": "red",
            "probs": [
                {"tok_str": "red", "prob": 0.6},
                {"tok_str": "green", "prob": 0.3},
                {"tok_str": "purple", "prob": 0.1},
            ],
        }
        question = Question(
            id="colour", kind=QuestionKind.CHOICE, text="Pick.", options=("red", "blue", "green")
        )
        primitives = _FakePrimitives("", meta=_build_meta([question], [answer_row], tokenizer))

        result = _run(primitives, [question], tokenize_fn=tokenizer)
        report = native_diagnostics(result, primitives)
        entry = report["positions"][0]

        assert [outside["text"] for outside in entry["top_k_outside_candidates"]] == ["purple"]
        assert entry["all_candidate_variants_captured"] is False  # "blue" absent
        assert entry["mass_outside_candidates"] is None

    def test_diagnostics_reports_missing_variant_as_unknown_outside_mass(self):
        tokenizer = _FakeTokenizer(fallback=(_CUE_TOKEN,))
        answer_rows = _main_answer_rows()
        # " no" never appears in the top-K: candidate mass is undercounted, so
        # the exact outside mass must stay None rather than be fabricated.
        answer_rows[2] = _v9_row(
            "true",
            math.log(0.9),
            [("true", math.log(0.9)), ("false", math.log(0.1))],
        )
        primitives = _FakePrimitives("", meta=_build_meta(QUESTIONS, answer_rows, tokenizer))

        result = _run(primitives, QUESTIONS, tokenize_fn=tokenizer)
        report = native_diagnostics(result, primitives)
        noul = {entry["question_id"]: entry for entry in report["positions"]}["noul"]

        assert noul["all_candidate_variants_captured"] is False
        assert noul["mass_outside_candidates"] is None
        assert noul["candidate_mass_raw"] == pytest.approx(1.0)

    def test_diagnostics_without_rows_reports_no_capture(self):
        tokenizer = _FakeTokenizer(fallback=(_CUE_TOKEN,))
        primitives = _FakePrimitives("[ERROR: refused]", meta={})

        result = _run(primitives, QUESTIONS, tokenize_fn=tokenizer)
        report = native_diagnostics(result, primitives)

        assert report["rows_captured"] == 0
        assert len(report["positions"]) == 3
        assert all(entry["emitted"] is None for entry in report["positions"])
        assert report["failures"][0]["reason"] == "transport_error"
        assert report["positions"][0]["failure"] is None

    def test_diagnostics_reports_the_replayed_cue_style(self):
        tokenizer = _FakeTokenizer()
        primitives = _FakePrimitives(
            "", meta=_build_meta(QUESTIONS, _main_answer_rows(), tokenizer, CueStyle.SHORT)
        )

        result = _run(primitives, QUESTIONS, tokenize_fn=tokenizer, cue_style=CueStyle.SHORT)
        report = native_diagnostics(result, primitives)

        assert report["cue_style"] == "short"


# ── 11. Runner dispatch and JSON-mode non-regression ──────────────────────


class TestRunnerDispatch:
    def test_runner_native_mode_dispatches_to_native_runner(self):
        tokenizer = _FakeTokenizer(fallback=(_CUE_TOKEN,))
        primitives = _FakePrimitives("", meta=_main_meta(tokenizer))

        result = run_typed_decisions(
            primitives,
            state=STATE,
            questions=QUESTIONS,
            role=ROLE,
            mode="native",
            tokenize_fn=tokenizer,
        )

        assert result.mode == "native"
        assert [decision.value for decision in result.decisions] == ["blue", 2, True]
        assert primitives.calls[0]["grammar"] == _MAIN_GRAMMAR
        assert "json_schema" not in primitives.calls[0]

    def test_runner_native_mode_forwards_explicit_n_tokens(self):
        tokenizer = _FakeTokenizer(fallback=(_CUE_TOKEN,))
        primitives = _FakePrimitives("", meta=_main_meta(tokenizer))

        run_typed_decisions(
            primitives,
            state=STATE,
            questions=QUESTIONS,
            role=ROLE,
            mode="native",
            n_tokens=11,
            tokenize_fn=tokenizer,
        )

        assert primitives.calls[0]["n_tokens"] == 11

    def test_runner_native_mode_fails_closed_without_a_tokenizer(self):
        primitives = _FakePrimitives("", meta=_main_meta())

        result = run_typed_decisions(
            primitives, state=STATE, questions=QUESTIONS, role=ROLE, mode="native"
        )

        assert primitives.calls == []
        assert result.decisions == ()
        assert all(
            failure.reason == REASON_NATIVE_TOKENIZER_UNAVAILABLE for failure in result.failures
        )

    def test_json_mode_still_uses_the_schema_path_only(self):
        response = (
            '{"answers": {"noul": {"noul": true, "probabilities": '
            '{"true": 0.8, "false": 0.2}, "confidence": 0.8}}}'
        )
        primitives = _FakePrimitives(response, meta=_main_meta())

        result = run_typed_decisions(
            primitives, state=STATE, questions=[NOUL], role=ROLE, mode="json"
        )

        assert result.mode == "json"
        assert result.decisions[0].value is True
        call = primitives.calls[0]
        assert "json_schema" in call
        assert "grammar" not in call
        assert "n_probs" not in call

    def test_runner_native_mode_defaults_to_id_only_cues(self):
        # TD-6: the native default is id_only (11.98x at 15/16 agreement).
        tokenizer = _FakeTokenizer()
        primitives = _FakePrimitives(
            "", meta=_build_meta(QUESTIONS, _main_answer_rows(), tokenizer, CueStyle.ID_ONLY)
        )

        run_typed_decisions(
            primitives,
            state=STATE,
            questions=QUESTIONS,
            role=ROLE,
            mode="native",
            tokenize_fn=tokenizer,
        )

        assert primitives._last_native_layout["cue_style"] == "id_only"
        assert _cue_text(CHOICE, CueStyle.ID_ONLY) in tokenizer.calls
        assert _cue_text(CHOICE, CueStyle.FULL) not in tokenizer.calls

    def test_runner_native_mode_forwards_cue_style(self):
        tokenizer = _FakeTokenizer()
        primitives = _FakePrimitives(
            "", meta=_build_meta(QUESTIONS, _main_answer_rows(), tokenizer, CueStyle.ID_ONLY)
        )

        result = run_typed_decisions(
            primitives,
            state=STATE,
            questions=QUESTIONS,
            role=ROLE,
            mode="native",
            cue_style="id_only",
            tokenize_fn=tokenizer,
        )

        assert result.mode == "native"
        assert [decision.value for decision in result.decisions] == ["blue", 2, True]
        assert primitives._last_native_layout["cue_style"] == "id_only"
        assert _cue_text(CHOICE, CueStyle.ID_ONLY) in tokenizer.calls
        assert _cue_text(CHOICE, CueStyle.FULL) not in tokenizer.calls

    def test_runner_native_mode_rejects_an_unknown_cue_style(self):
        with pytest.raises(ValueError, match="unknown cue style"):
            run_typed_decisions(
                _FakePrimitives("", meta=_main_meta()),
                state=STATE,
                questions=QUESTIONS,
                role=ROLE,
                mode="native",
                cue_style="medium",
                tokenize_fn=_FakeTokenizer(),
            )

    def test_json_mode_ignores_cue_style(self):
        response = (
            '{"answers": {"noul": {"noul": true, "probabilities": '
            '{"true": 0.8, "false": 0.2}, "confidence": 0.8}}}'
        )
        primitives = _FakePrimitives(response, meta=_main_meta())

        result = run_typed_decisions(
            primitives,
            state=STATE,
            questions=[NOUL],
            role=ROLE,
            mode="json",
            cue_style="id_only",
        )

        assert result.mode == "json"
        assert result.decisions[0].value is True
        assert "grammar" not in primitives.calls[0]

    def test_native_runner_rejects_an_unknown_cue_style(self):
        with pytest.raises(ValueError, match="unknown cue style"):
            _run(
                _FakePrimitives("", meta=_main_meta()),
                QUESTIONS,
                tokenize_fn=_FakeTokenizer(),
                cue_style="medium",
            )


# ── TD-1d option (b): run_typed_decisions_native_parallel (TD-21.33b) ──────


def _parallel_meta(row: Mapping[str, Any]) -> dict[str, Any]:
    """A one-row meta: the parallel shape's cue lives in the PROMPT, so the
    answer is always row 0 (no cue filler rows, unlike ``_build_meta``)."""
    return {"completion_probabilities": [row]}


class TestNativeParallel:
    """``run_typed_decisions_native_parallel`` (parked TD-1d.4 harness, landed
    unwired 2026-09-24 — TD-1d.0 is not settled and the 2026-09-18
    re-measurement found this shape a loser on the frozen stack; it is kept as
    a harness for a future runtime choice, not called from production)."""

    def test_empty_pool_raises(self):
        with pytest.raises(ValueError, match="non-empty primitives pool"):
            run_typed_decisions_native_parallel(
                [],
                state=STATE,
                questions=QUESTIONS,
                role=ROLE,
                tokenize_fn=_FakeTokenizer(),
            )

    def test_no_native_eligible_questions_makes_no_calls(self):
        tokenizer = _FakeTokenizer()
        pool = [_FakePrimitives(""), _FakePrimitives("")]

        result = run_typed_decisions_native_parallel(
            pool,
            state=STATE,
            questions=[MULTI_TOKEN],
            role=ROLE,
            tokenize_fn=tokenizer,
            single_token_keys=False,  # TD-1b legacy path pinned
        )

        assert result.mode == "native_parallel"
        assert result.decisions == ()
        assert result.elapsed_ms == 0.0
        assert len(result.failures) == 1
        assert result.failures[0].reason == REASON_NATIVE_UNSUPPORTED_CANDIDATES
        assert pool[0].calls == []
        assert pool[1].calls == []

    def test_one_worker_per_question_resolves_every_decision(self):
        """Pool width == catalogue width: each worker answers exactly one
        question, proving the per-question fan-out end to end against fakes."""
        tokenizer = _FakeTokenizer()
        rows = _main_answer_rows()
        pool = [
            _FakePrimitives("blue", meta=_parallel_meta(rows[0])),
            _FakePrimitives("2", meta=_parallel_meta(rows[1])),
            _FakePrimitives("true", meta=_parallel_meta(rows[2])),
        ]

        result = run_typed_decisions_native_parallel(
            pool,
            state=STATE,
            questions=QUESTIONS,
            role=ROLE,
            tokenize_fn=tokenizer,
        )

        assert result.mode == "native_parallel"
        assert result.failures == ()
        by_id = _by_id(result)
        assert by_id["choice"].value == "blue"
        assert by_id["score"].value == 2
        assert by_id["noul"].value is True
        assert result.raw_text == "blue2true"  # catalogue order: choice, score, noul
        # Each worker made exactly one call, none pinned to a slot, no warm call.
        for worker in pool:
            assert len(worker.calls) == 1
            call = worker.calls[0]
            assert call["n_tokens"] == 1
            assert "slot_id" not in call
            assert call["prompt"].startswith(build_native_prompt(STATE, [q for q in QUESTIONS]))

    def test_round_robin_assignment_across_fewer_workers_than_questions(self):
        """2 workers, 3 questions: jobs 0/2 -> worker 0, job 1 -> worker 1."""
        tokenizer = _FakeTokenizer()
        rows = _main_answer_rows()
        worker0 = _FakePrimitives(["blue", "true"], meta=_parallel_meta(rows[0]))
        worker1 = _FakePrimitives("2", meta=_parallel_meta(rows[1]))
        pool = [worker0, worker1]

        # worker0's meta is fixed at construction (fake limitation): it
        # correctly answers whichever of its two assigned questions the
        # decision-row happens to describe, so pin the assertions to worker1
        # (unambiguous) and to the call COUNT on worker0 (its fan-out width).
        run_typed_decisions_native_parallel(
            pool,
            state=STATE,
            questions=QUESTIONS,
            role=ROLE,
            tokenize_fn=tokenizer,
        )

        assert len(worker0.calls) == 2  # jobs 0 ("choice") and 2 ("noul")
        assert len(worker1.calls) == 1  # job 1 ("score")

    def test_pin_slots_passes_slot_id_per_worker(self):
        tokenizer = _FakeTokenizer()
        rows = _main_answer_rows()
        pool = [
            _FakePrimitives("blue", meta=_parallel_meta(rows[0])),
            _FakePrimitives("2", meta=_parallel_meta(rows[1])),
            _FakePrimitives("true", meta=_parallel_meta(rows[2])),
        ]

        run_typed_decisions_native_parallel(
            pool,
            state=STATE,
            questions=QUESTIONS,
            role=ROLE,
            tokenize_fn=tokenizer,
            pin_slots=True,
        )

        for worker_index, worker in enumerate(pool):
            assert worker.calls[0]["slot_id"] == worker_index

    def test_warm_prefix_issues_one_unconstrained_call_before_the_read(self):
        tokenizer = _FakeTokenizer()
        rows = _main_answer_rows()
        pool = [
            _FakePrimitives(["ignored", "blue"], meta=_parallel_meta(rows[0])),
            _FakePrimitives(["ignored", "2"], meta=_parallel_meta(rows[1])),
            _FakePrimitives(["ignored", "true"], meta=_parallel_meta(rows[2])),
        ]

        result = run_typed_decisions_native_parallel(
            pool,
            state=STATE,
            questions=QUESTIONS,
            role=ROLE,
            tokenize_fn=tokenizer,
            warm_prefix=True,
        )

        assert result.failures == ()
        for worker in pool:
            assert len(worker.calls) == 2
            warm_call, read_call = worker.calls
            assert warm_call["n_tokens"] == 1
            assert "grammar" not in warm_call
            assert "grammar" in read_call
        layout = pool[0]._last_native_layout
        assert layout["shape"] == "per_question_parallel"
        assert layout["workers"] == 3
        assert layout["warm_prefix"] is True
        assert len(layout["warm_requests"]) == 3
        assert all(entry is not None for entry in layout["warm_requests"])

    def test_transport_error_fails_only_its_own_question(self):
        tokenizer = _FakeTokenizer()
        rows = _main_answer_rows()
        pool = [
            _FakePrimitives("blue", meta=_parallel_meta(rows[0])),
            _FakePrimitives("[ERROR: connection refused]", meta=_parallel_meta(rows[1])),
            _FakePrimitives("true", meta=_parallel_meta(rows[2])),
        ]

        result = run_typed_decisions_native_parallel(
            pool,
            state=STATE,
            questions=QUESTIONS,
            role=ROLE,
            tokenize_fn=tokenizer,
        )

        by_id = _by_id(result)
        assert by_id["choice"].value == "blue"
        assert by_id["noul"].value is True
        assert "score" not in by_id
        assert len(result.failures) == 1
        assert result.failures[0].reason == REASON_TRANSPORT_ERROR
        assert "score" in result.failures[0].detail

    def test_layout_side_channel_records_per_request_snapshots(self):
        tokenizer = _FakeTokenizer()
        rows = _main_answer_rows()
        pool = [
            _FakePrimitives("blue", meta=_parallel_meta(rows[0])),
            _FakePrimitives("2", meta=_parallel_meta(rows[1])),
            _FakePrimitives("true", meta=_parallel_meta(rows[2])),
        ]

        run_typed_decisions_native_parallel(
            pool,
            state=STATE,
            questions=QUESTIONS,
            role=ROLE,
            tokenize_fn=tokenizer,
        )

        layout = pool[0]._last_native_layout
        assert layout["shape"] == "per_question_parallel"
        assert layout["workers"] == 3
        assert layout["pin_slots"] is False
        assert layout["warm_prefix"] is False
        assert len(layout["per_request"]) == 3
        assert all(entry is not None and "call_ms" in entry for entry in layout["per_request"])

    def test_default_cue_style_matches_the_serial_default(self):
        """TD-6: id_only is the production default for BOTH runners — the
        parked WIP's diff had drifted this back to ``full`` pre-TD-6; this
        pins the corrected default so a future edit cannot silently reopen it.
        """
        import inspect

        default = (
            inspect.signature(run_typed_decisions_native_parallel).parameters["cue_style"].default
        )
        assert default is CueStyle.ID_ONLY


class TestNativeParallelWithRealPrimitives:
    """Threading correctness (TD-21.33b): each worker's meta read must observe
    only ITS OWN call, never a concurrent worker's — proven against the real
    ``LLMPrimitives.get_last_inference_meta()`` context-var machinery, not the
    fakes above (which cannot exercise a real race)."""

    @staticmethod
    def _primitives_with_backend(role: str, output: str, row: Mapping[str, Any]):
        from unittest.mock import Mock

        from src.llm_primitives import LLMPrimitives
        from src.model_server import InferenceResult

        prims = LLMPrimitives(mock_mode=False, server_urls={role: "http://localhost:0"})
        backend = Mock(spec=[])
        backend.infer = Mock(
            return_value=InferenceResult(
                role=role,
                output=output,
                tokens_generated=1,
                generation_speed=1.0,
                elapsed_time=0.001,
                success=True,
                prompt_eval_ms=0.1,
                generation_ms=0.1,
                http_overhead_ms=0.0,
                completion_reason="stop",
                completion_probabilities=[row],
            )
        )
        prims._backends[role] = backend
        return prims, backend

    def test_each_worker_observes_only_its_own_call(self):
        tokenizer = _FakeTokenizer()
        rows = _main_answer_rows()
        pool_and_backends = [
            self._primitives_with_backend(ROLE, "blue", rows[0]),
            self._primitives_with_backend(ROLE, "2", rows[1]),
            self._primitives_with_backend(ROLE, "true", rows[2]),
        ]
        pool = [prims for prims, _backend in pool_and_backends]

        # Warm the lazy `from src.inference_lock import ...`-style imports
        # `_real_call` performs on its FIRST invocation, serially, before the
        # threaded fan-out below: CPython's import lock serializes concurrent
        # imports of the SAME module, but a module whose own import triggers
        # a nested import of a module already mid-import elsewhere raises
        # ImportError ("partially initialized module") rather than blocking —
        # a first-import ordering hazard orthogonal to the getter migration
        # this test exists to prove, so it is neutralized here rather than
        # left to flake on interpreter/test-order state.
        pool[0].llm_call("warm", role=ROLE, n_tokens=1, temperature=0.0, seed=_DECODE_SEED)
        for prims, backend in pool_and_backends:
            backend.infer.reset_mock()

        result = run_typed_decisions_native_parallel(
            pool,
            state=STATE,
            questions=QUESTIONS,
            role=ROLE,
            tokenize_fn=tokenizer,
        )

        assert result.failures == ()
        by_id = _by_id(result)
        # Each decision matches ONLY its own worker's backend output -- if the
        # per-call-safe getter's read inside `worker()` ever observed a
        # DIFFERENT thread's meta (the exact race this migration closes), a
        # question would resolve to a sibling worker's answer instead, or the
        # slicing would fail closed. `get_last_inference_meta()` itself is NOT
        # asserted here from this (main) thread: it is a ContextVar scoped to
        # the thread that set it, so reading it back from the thread that
        # DIDN'T make the call (as a test easily could, by mistake) would
        # prove nothing -- exactly the context-propagation edge TD-21.33's
        # `_best_effort_last_inference_meta` docstring calls out. The
        # implementation itself captures each meta INSIDE its own worker
        # thread, immediately after its own call (see `outcomes[index]` in
        # `run_typed_decisions_native_parallel`), which is the only place the
        # read is valid -- these decisions are that capture's output.
        assert by_id["choice"].value == "blue"
        assert by_id["score"].value == 2
        assert by_id["noul"].value is True
        for _prims, backend in pool_and_backends:
            backend.infer.assert_called_once()


# ── Wire-level: real LLMPrimitives, mocked HTTP, no fake primitives double ──


class TestPostSamplingProbsOnTheWire:
    """TD-1d.2 window-diag (2026-09-24): after 71be6ed3 (which threads
    ``post_sampling_probs=True`` through ``LLMPrimitives.llm_call`` ->
    ``InferenceRequest`` -> both backend payload builders), the LIVE re-bench
    is unchanged -- native:full 1/15 ``native_unknown_candidate``, id_only
    0/16 -- as if the flag never reached the server.

    Every existing ``post_sampling_probs`` test (``test_llama_server.py``'s
    ``test_build_payload_forwards_post_sampling_probs`` and
    ``test_post_sampling_probs_translates_to_openai_chat_params``,
    ``test_typed_decisions_native.py``'s
    ``test_captured_kwargs_carry_cue_grammar_and_probability_capture``)
    starts from an already-built ``InferenceRequest`` or a ``_FakePrimitives``
    double that just records kwargs -- none of them drives the REAL upper
    stack bench.py actually uses: ``run_typed_decisions_native`` ->
    ``LLMPrimitives.llm_call`` -> ``_llm_call_impl`` -> ``_real_call`` ->
    ``_real_call_impl`` -> ``_call_caching_backend`` (constructs the
    ``InferenceRequest``) -> ``CachingBackend`` -> ``LlamaServerBackend``. A
    drop anywhere in THAT stack -- a kwarg the primitives-level `llm_call`
    signature doesn't forward, a caching/admission wrapper that rebuilds the
    request without every field -- would be invisible to those tests and
    would reproduce exactly the live symptom. This test drives the real
    stack (only the httpx transport is mocked, matching bench's own
    ``_live_primitives()`` construction: ``LLMPrimitives(mock_mode=False,
    server_urls=..., num_slots=...)``, no ``registry=``) and asserts the
    flag is on the wire.
    """

    def _live_primitives(self, monkeypatch):
        """Mirror ``src/typed_decisions/bench.py::_live_primitives`` exactly,
        against a fake server URL, with the shared ``heavy_model.lock`` file
        (a REAL, production-shared resource -- role=frontdoor is a heavy
        role) replaced by a no-op so this test never touches it."""
        import contextlib

        from src.llm_primitives import LLMPrimitives

        # frontdoor routes through /v1/chat/completions in production (J12);
        # force the same routing here rather than depending on whatever
        # generated stack-priors artifact (or lack of one) this process sees.
        monkeypatch.setenv("ORCHESTRATOR_USE_CHAT_COMPLETIONS_ROLES", "frontdoor")

        @contextlib.contextmanager
        def _noop_lock(*_args, **_kwargs):
            yield

        monkeypatch.setattr("src.runtime.inference_lock.inference_lock", _noop_lock)

        primitives = LLMPrimitives(
            mock_mode=False,
            server_urls={"frontdoor": "http://test-native-wire:8080"},
            num_slots=2,
        )
        assert primitives._backends.get("frontdoor") is not None, (
            "test setup: no CachingBackend built for frontdoor -- fix the "
            "fixture, not the assertion below"
        )
        return primitives

    def _chat_response(self, *, n_filler_rows: int):
        """A /v1/chat/completions response in the post_sampling_probs=True
        shape (``prob``/``top_probs``, not ``logprob``/``top_logprobs``) --
        ``n_filler_rows`` opaque cue-token rows followed by the real "true"
        answer row, matching native.py's cue+answer generated-token layout
        (mirrors this file's ``_build_meta`` helper, but wire-shaped).
        """
        from unittest.mock import Mock

        filler = [
            {"id": _CUE_TOKEN, "token": "", "prob": 0.0, "top_probs": []}
            for _ in range(n_filler_rows)
        ]
        answer = {
            "id": _DEFAULT_IDS["true"],
            "token": "true",
            "prob": 0.9,
            "top_probs": [
                {"id": _DEFAULT_IDS["true"], "token": "true", "prob": 0.9},
                {"id": _DEFAULT_IDS["false"], "token": "false", "prob": 0.1},
            ],
        }
        response = Mock()
        response.status_code = 200
        response.raise_for_status = Mock()
        response.json.return_value = {
            "choices": [
                {
                    "message": {"content": "true"},
                    "finish_reason": "stop",
                    "logprobs": {"content": [*filler, answer]},
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": len(filler) + 1},
            "timings": {"prompt_ms": 5.0, "predicted_ms": 5.0, "predicted_per_second": 30.0},
        }
        return response

    def test_post_sampling_probs_reaches_the_wire_through_the_real_stack(self, monkeypatch):
        primitives = self._live_primitives(monkeypatch)
        backend = primitives._backends["frontdoor"].backend
        captured: dict[str, Any] = {}

        tokenizer = _FakeTokenizer()

        def _post(_path, json=None, timeout=None):
            captured.update(json or {})
            # The answer token is always the LAST of the request's own
            # max_tokens budget (native.py sizes n_tokens exactly to its
            # forced cue+answer sequence) -- read it back from the just-
            # captured request rather than recomputing the cue length by a
            # second, easily-mismatched route.
            max_tokens = int((json or {}).get("max_tokens") or 1)
            return self._chat_response(n_filler_rows=max(0, max_tokens - 1))

        monkeypatch.setattr(backend.client, "post", _post)

        result = run_typed_decisions_native(
            primitives,
            state=STATE,
            questions=[NOUL],
            role="frontdoor",
            tokenize_fn=tokenizer,
        )

        assert captured, "backend.client.post was never called -- fix the fixture, not this assert"
        assert captured.get("post_sampling_probs") is True, (
            "post_sampling_probs did not reach the /v1/chat/completions wire "
            f"through the real LLMPrimitives.llm_call stack; captured payload "
            f"keys={sorted(captured)}"
        )
        # Full round trip resolves too -- not just "the key is present somewhere".
        assert result.failures == (), result.failures
        assert _by_id(result)["noul"].value is True


# ── TD-29: single-token keys for multi-token closed sets ──────────────────

_SEVERITY = Question(
    id="severity",
    kind=QuestionKind.CHOICE,
    text="Choose the value for argument 'severity' of tool 'file_ticket'.",
    options=("p1", "p2", "p3", "p4"),
)


def _key_vocab(extra: Mapping[str, tuple[int, ...]] | None = None) -> dict[str, tuple[int, ...]]:
    """Default vocab + multi-token p1..p4 + single-token A..Z / 0..9 keys."""
    vocab = dict(_DEFAULT_VOCAB)
    for index, label in enumerate(("p1", "p2", "p3", "p4")):
        vocab[label] = (200 + index, 300 + index)  # "p" + digit: two tokens
        vocab[" " + label] = (210 + index, 300 + index)
    from src.typed_decisions.routing_replay import _CODE_ALPHABET

    for index, code in enumerate(_CODE_ALPHABET):
        vocab[code] = (3000 + 2 * index,)
        vocab[" " + code] = (3001 + 2 * index,)
    vocab.update(extra or {})
    return vocab


def _key_ids(vocab: Mapping[str, tuple[int, ...]]) -> dict[str, int]:
    return {text: ids[0] for text, ids in vocab.items() if len(ids) == 1}


class TestSingleTokenKeys:
    def _severity_run(self, emitted: str, top, **kwargs):
        vocab = _key_vocab()
        tokenizer = _FakeTokenizer(vocab, fallback=(_CUE_TOKEN,))
        row = _v9_row(emitted, top[0][1], top, ids=_key_ids(vocab))
        primitives = _FakePrimitives("", meta=_build_meta([_SEVERITY], [row], tokenizer, "id_only"))
        result = _run(primitives, [_SEVERITY], tokenize_fn=tokenizer, **kwargs)
        return result, primitives, tokenizer

    def test_multi_token_enum_resolves_through_keys(self):
        result, primitives, tokenizer = self._severity_run(
            " C",
            [(" C", math.log(0.7)), ("A", math.log(0.2)), (" B", math.log(0.1))],
        )

        assert result.failures == ()
        decision = result.decisions[0]
        assert decision.value == "p3"  # the ORIGINAL label, never the key
        assert decision.native_key == "C"
        assert dict(decision.probabilities) == {
            "p1": pytest.approx(0.2),
            "p2": pytest.approx(0.1),
            "p3": pytest.approx(0.7),
            "p4": pytest.approx(0.0),
        }
        assert decision.token_logprob == pytest.approx(math.log(0.7))

        call = primitives.calls[0]
        # The grammar binds the KEY tokens (bare and spaced), in candidate order.
        assert (
            "answer-0 ::= <[3000]> | <[3001]> | <[3002]> | <[3003]> | <[3004]> | "
            in (call["grammar"])
        )
        assert "<[200]>" not in call["grammar"]
        # The label stays the display text; the model is asked for the key.
        assert "   candidates: A = p1 | B = p2 | C = p3 | D = p4\n" in call["prompt"]
        assert "   Answer with the key (one of: A, B, C, D):\n" in call["prompt"]
        # Keys were verified through the tokenizer seam, not assumed.
        assert {"A", " A", "D", " D"} <= set(tokenizer.calls)
        assert "E" not in tokenizer.calls
        # Receipts: the layout carries key AND label for every candidate.
        layout = primitives._last_native_layout
        assert [(c["key"], c["label"]) for c in layout["positions"][0]["candidates"]] == [
            ("A", "p1"),
            ("B", "p2"),
            ("C", "p3"),
            ("D", "p4"),
        ]

    @pytest.mark.parametrize(("key", "label"), [("A", "p1"), ("B", "p2"), ("C", "p3"), ("D", "p4")])
    def test_key_to_value_round_trip(self, key, label):
        result, primitives, _ = self._severity_run(key, [(key, math.log(0.9))])

        decision = result.decisions[0]
        assert (decision.native_key, decision.value) == (key, label)
        # The layout side channel rebuilds the same binding for diagnostics.
        diagnostics = native_diagnostics(result, primitives)
        question = diagnostics["positions"][0]
        assert question["emitted_label"] == label
        assert question["argmax_label"] == label

    def test_tool_arguments_assemble_from_keyed_severity(self):
        # The TD-29 failure shape: every file_ticket case failed native with
        # "required argument 'severity' was not answered" (p1..p4 multi-token).
        from src.typed_decisions.tool_args import assemble_arguments, tool_schema_to_questions
        from src.typed_decisions.tool_args_pilot import build_cases

        vocab = _key_vocab({str(level): (4000 + level,) for level in range(4, 9)})
        ids = _key_ids(vocab)
        for case in (case for case in build_cases() if case.tool == "file_ticket"):
            mapping = tool_schema_to_questions(case.tool, case.parameters)
            severity_key = "ABCD"[("p1", "p2", "p3", "p4").index(case.expected["severity"])]
            tokens = []
            for question in mapping.questions:
                if question.id == "severity":
                    tokens.append(severity_key)
                elif question.kind is QuestionKind.SCORE:
                    tokens.append(str(case.expected["effort_points"]))
                else:
                    if question.id == "escalate":
                        truth = case.expected["escalate"]
                    else:
                        truth = question.id.split("__", 1)[1] in case.expected["categories"]
                    tokens.append("true" if truth else "false")
            rows = [_v9_row(token, 0.0, [(token, 0.0)], ids=ids) for token in tokens]
            tokenizer = _FakeTokenizer(vocab, fallback=(_CUE_TOKEN,))
            primitives = _FakePrimitives(
                "", meta=_build_meta(mapping.questions, rows, tokenizer, "id_only")
            )

            result = _run(primitives, mapping.questions, tokenize_fn=tokenizer)

            assert result.failures == (), case.case_id
            arguments = assemble_arguments(mapping.questions, result.decisions, case.parameters)
            assert arguments == case.expected, case.case_id
            keyed = {d.question_id: d.native_key for d in result.decisions if d.native_key}
            assert keyed == {"severity": severity_key}

    def test_single_token_sets_are_byte_identical_with_keys_on(self):
        runs = []
        for keys_on in (True, False):
            tokenizer = _FakeTokenizer(fallback=(_CUE_TOKEN,))
            primitives = _FakePrimitives("", meta=_main_meta(tokenizer))
            result = _run(primitives, QUESTIONS, tokenize_fn=tokenizer, single_token_keys=keys_on)
            runs.append((result, primitives, tokenizer))

        (on, p_on, t_on), (off, p_off, t_off) = runs
        assert p_on.calls == p_off.calls  # prompt, grammar, n_tokens, n_probs
        assert p_on._last_native_layout == p_off._last_native_layout
        assert on.decisions == off.decisions
        assert on.prompt_sha256 == off.prompt_sha256
        assert all(decision.native_key is None for decision in on.decisions)
        assert all(
            "key" not in c for p in p_on._last_native_layout["positions"] for c in p["candidates"]
        )
        assert t_on.calls == t_off.calls  # no key was even probed

    def test_mixed_catalogue_keys_only_the_multi_token_set(self):
        vocab = _key_vocab()
        ids = _key_ids(vocab)
        tokenizer = _FakeTokenizer(vocab, fallback=(_CUE_TOKEN,))
        rows = [
            _v9_row("blue", 0.0, [("blue", 0.0)], ids=ids),
            _v9_row(" B", 0.0, [(" B", 0.0)], ids=ids),
        ]
        primitives = _FakePrimitives(
            "", meta=_build_meta([CHOICE, _SEVERITY], rows, tokenizer, "id_only")
        )

        result = _run(primitives, [CHOICE, _SEVERITY], tokenize_fn=tokenizer)

        by_id = _by_id(result)
        assert (by_id["choice"].value, by_id["choice"].native_key) == ("blue", None)
        assert (by_id["severity"].value, by_id["severity"].native_key) == ("p2", "B")
        prompt = primitives.calls[0]["prompt"]
        assert "   candidates: red | blue | green\n" in prompt
        assert "   candidates: A = p1 | B = p2 | C = p3 | D = p4\n" in prompt

    def test_colliding_and_multi_token_codes_are_skipped_deterministically(self):
        vocab = _key_vocab(
            {
                "B": (3000,),  # collides with A's bare id
                " B": (3000,),
                "C": (91, 92),  # multi-token both ways
                " C": (93, 94),
            }
        )
        ids = _key_ids(vocab)
        tokenizer = _FakeTokenizer(vocab, fallback=(_CUE_TOKEN,))
        row = _v9_row("E", 0.0, [("E", 0.0)], ids=ids)
        primitives = _FakePrimitives("", meta=_build_meta([_SEVERITY], [row], tokenizer, "id_only"))

        result = _run(primitives, [_SEVERITY], tokenize_fn=tokenizer)

        keys = [c["key"] for c in primitives._last_native_layout["positions"][0]["candidates"]]
        assert keys == ["A", "D", "E", "F"]
        assert (result.decisions[0].value, result.decisions[0].native_key) == ("p3", "E")
        # No two keys share a token id.
        bound = [
            token_id
            for c in primitives._last_native_layout["positions"][0]["candidates"]
            for token_id in c["token_ids"]
        ]
        assert len(bound) == len(set(bound))

    def test_code_equal_to_another_label_is_skipped(self):
        question = Question(
            id="pick",
            kind=QuestionKind.CHOICE,
            text="Pick.",
            options=("B", "the long option"),
        )
        tokenizer = _FakeTokenizer(_key_vocab())  # "the long option" -> one id per char
        primitives = _FakePrimitives("", meta={"completion_probabilities": []})

        _run(primitives, [question], tokenize_fn=tokenizer)

        prompt = primitives.calls[0]["prompt"]
        assert "   candidates: A = B | C = the long option\n" in prompt

    def test_overflow_falls_back_with_reason(self):
        options = tuple(f"option number {index}" for index in range(37))
        question = Question(id="big", kind=QuestionKind.CHOICE, text="Pick.", options=options)
        tokenizer = _FakeTokenizer(_key_vocab())
        primitives = _FakePrimitives("")

        result = _run(primitives, [question], tokenize_fn=tokenizer)

        assert primitives.calls == []
        assert result.decisions == ()
        failure = result.failures[0]
        assert failure.reason == REASON_NATIVE_UNSUPPORTED_CANDIDATES
        assert "37 candidates exceed the 36 single-token key codes" in failure.detail
        assert "JSON mode" in failure.detail

    def test_exhausted_alphabet_falls_back_with_reason(self):
        # Every code tokenizes to two ids, bare and spaced: nothing can be keyed.
        from src.typed_decisions.routing_replay import _CODE_ALPHABET

        multi = {code: (7, 8) for code in _CODE_ALPHABET}
        multi.update({" " + code: (7, 8) for code in _CODE_ALPHABET})
        tokenizer = _FakeTokenizer(_key_vocab(multi))
        primitives = _FakePrimitives("")

        result = _run(primitives, [_SEVERITY], tokenize_fn=tokenizer)

        assert primitives.calls == []
        failure = result.failures[0]
        assert failure.reason == REASON_NATIVE_UNSUPPORTED_CANDIDATES
        assert "alphabet ran out after binding 0 of 4 candidates" in failure.detail
        assert "(not a single token)" in failure.detail

    def test_unanswered_key_probe_is_tokenizer_unavailable(self):
        tokenizer = _FakeTokenizer(_key_vocab(), failing=(" A",))
        primitives = _FakePrimitives("")

        result = _run(primitives, [_SEVERITY], tokenize_fn=tokenizer)

        assert primitives.calls == []
        assert result.failures[0].reason == REASON_NATIVE_TOKENIZER_UNAVAILABLE
        assert "key probe ' A'" in result.failures[0].detail

    def test_keys_off_keeps_td1b_failure(self):
        tokenizer = _FakeTokenizer(_key_vocab())
        primitives = _FakePrimitives("")

        result = _run(primitives, [_SEVERITY], tokenize_fn=tokenizer, single_token_keys=False)

        assert primitives.calls == []
        assert result.failures[0].reason == REASON_NATIVE_UNSUPPORTED_CANDIDATES
        assert "single-token keys" not in result.failures[0].detail
        assert "A" not in tokenizer.calls

    def test_full_cue_lists_keys_not_labels(self):
        vocab = _key_vocab()
        tokenizer = _FakeTokenizer(vocab)
        primitives = _FakePrimitives("", meta={"completion_probabilities": []})

        _run(primitives, [_SEVERITY], tokenize_fn=tokenizer, cue_style=CueStyle.FULL)

        keyed_cue = _cue_text(_SEVERITY, CueStyle.FULL, keys=("A", "B", "C", "D"))
        assert keyed_cue.endswith("Answer (one of: A, B, C, D): ")
        assert keyed_cue in tokenizer.calls
        assert _cue_text(_SEVERITY, CueStyle.FULL) not in tokenizer.calls
        assert _cue_text(_SEVERITY, CueStyle.ID_ONLY, keys=("A",)) == "\nseverity: "

    def test_parallel_path_resolves_keys(self):
        vocab = _key_vocab()
        ids = _key_ids(vocab)
        tokenizer = _FakeTokenizer(vocab, fallback=(_CUE_TOKEN,))
        row = _v9_row(" D", 0.0, [(" D", 0.0)], ids=ids)
        pool = [_FakePrimitives("", meta={"completion_probabilities": [row]})]

        result = run_typed_decisions_native_parallel(
            pool, state=STATE, questions=[_SEVERITY], role=ROLE, tokenize_fn=tokenizer
        )

        assert result.failures == ()
        assert (result.decisions[0].value, result.decisions[0].native_key) == ("p4", "D")
        prompt = pool[0].calls[0]["prompt"]
        assert "   candidates: A = p1 | B = p2 | C = p3 | D = p4\n" in prompt
        assert prompt.endswith("\nseverity: ")
