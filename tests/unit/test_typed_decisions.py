"""Unit tests for the TD-1 typed-decision core (src/typed_decisions).

Mirrors the fake-primitives pattern of ``tests/unit/test_consultation.py``
(one canned response, captured call kwargs) and the validator usage of
``tests/test_review_grammar.py`` (Draft202012Validator). No model/server call.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

import pytest
from jsonschema import Draft202012Validator

from src.typed_decisions import (
    Decision,
    DecisionResult,
    ParseFailure,
    Question,
    QuestionKind,
    run_typed_decisions,
)
from src.typed_decisions.confidence import (
    choice_confidence,
    normalize_probabilities,
    score_confidence,
)
from src.typed_decisions.schema import build_gbnf, build_response_schema
from src.typed_decisions.runner import _build_catalog

ROLE = "worker"
STATE = "unit-test state: refactor candidate A against candidate B."


class _FakePrimitives:
    """Canned-response stand-in for ``LLMPrimitives``.

    ``responses`` is consumed in order; the last entry repeats, which lets
    the persistent-failure case reuse one canned string for every attempt.
    """

    def __init__(self, responses: str | Sequence[str]):
        if isinstance(responses, str):
            responses = [responses]
        self.responses = list(responses)
        self.calls: list[dict] = []

    def llm_call(self, prompt: str, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index]


def _question_fixture() -> tuple[Question, ...]:
    choice = [
        Question(
            id=f"choice-{index}",
            kind=QuestionKind.CHOICE,
            text=f"Pick the deployment strategy for service {index}.",
            options=("alpha", "beta", "gamma"),
            criteria=("must be reversible",) if index == 0 else (),
        )
        for index in range(8)
    ]
    score = [
        Question(
            id=f"score-{index}",
            kind=QuestionKind.SCORE,
            text=f"Rate the risk of change {index}.",
            levels=(0, 1, 2, 3),
        )
        for index in range(6)
    ]
    noul = [
        Question(
            id=f"noul-{index}",
            kind=QuestionKind.NOUL,
            text=f"Is invariant {index} satisfied?",
        )
        for index in range(6)
    ]
    return tuple(choice + score + noul)


QUESTIONS = _question_fixture()


def _probabilities(labels: Sequence[str]) -> dict[str, float]:
    weights = [1.0 / (index + 2) for index in range(len(labels))]
    total = sum(weights)
    return {label: weight / total for label, weight in zip(labels, weights)}


def _valid_response(questions: Sequence[Question]) -> str:
    answers: dict[str, dict] = {}
    for question in questions:
        if question.kind is QuestionKind.CHOICE:
            labels = list(question.options)
            answers[question.id] = {
                "choice": labels[0],
                "probabilities": _probabilities(labels),
                "confidence": 0.8,
            }
        elif question.kind is QuestionKind.SCORE:
            labels = [str(level) for level in question.levels]
            answers[question.id] = {
                "score": question.levels[0],
                "probabilities": _probabilities(labels),
                "confidence": 0.7,
            }
        else:
            answers[question.id] = {
                "noul": True,
                "probabilities": {"true": 0.9, "false": 0.1},
                "confidence": 0.9,
            }
    return json.dumps({"answers": answers})


# ── 1. Full batch over a canned valid response ────────────────────────────


class TestJsonRun:
    def test_twenty_question_batch_produces_schema_valid_decisions(self):
        response = _valid_response(QUESTIONS)
        primitives = _FakePrimitives(response)

        result = run_typed_decisions(primitives, state=STATE, questions=QUESTIONS, role=ROLE)

        assert isinstance(result, DecisionResult)
        assert result.mode == "json"
        assert result.failures == ()
        assert len(result.decisions) == 20
        assert result.raw_text == response
        assert result.elapsed_ms >= 0.0

        by_id = {decision.question_id: decision for decision in result.decisions}
        assert set(by_id) == {question.id for question in QUESTIONS}
        for question in QUESTIONS:
            decision = by_id[question.id]
            assert isinstance(decision, Decision)
            assert decision.kind is question.kind
            assert decision.mode == "json"
            assert decision.token_logprob is None
            assert sum(decision.probabilities.values()) == pytest.approx(1.0)
            assert 0.0 <= decision.confidence <= 1.0
            if question.kind is QuestionKind.CHOICE:
                assert decision.value in question.options
            elif question.kind is QuestionKind.SCORE:
                assert decision.value in question.levels
            else:
                assert decision.value is True

        # The canned response must itself satisfy the generated schema.
        Draft202012Validator(build_response_schema(QUESTIONS)).validate(json.loads(response))

        schema = build_response_schema(QUESTIONS)
        call = primitives.calls[0]
        assert len(primitives.calls) == 1
        assert call["role"] == ROLE
        assert call["json_schema"] == schema
        assert call["temperature"] == 0.0
        assert call["seed"] == 0
        assert isinstance(call["n_tokens"], int) and call["n_tokens"] > 0
        assert "grammar" not in call
        assert result.prompt_sha256 == hashlib.sha256(call["prompt"].encode("utf-8")).hexdigest()

    def test_n_tokens_override_is_forwarded(self):
        primitives = _FakePrimitives(_valid_response(QUESTIONS))

        run_typed_decisions(
            primitives,
            state=STATE,
            questions=QUESTIONS,
            role=ROLE,
            n_tokens=777,
        )

        assert primitives.calls[0]["n_tokens"] == 777


class TestDefaultNTokens:
    """TD-1d.2 window-diag (2026-09-24): the JSON arm's live re-bench against
    the real ``decision_set_v1`` catalogue (24 questions: 8 noul, 8 choice,
    8 score) ended ``completion_reason: length`` at the old flat budget
    (``_default_n_tokens(24) == 1600``), 0 decisions, 2x ``no_json`` -- the
    model never emitted a balanced JSON object in either attempt.

    Offline measurement (Qwen3-family tokenizer as a proxy for frontdoor's
    Qwen3.6 tokenizer -- same BPE vocab family) of the MINIMAL valid answer
    for that exact catalogue: 2273 chars / 999 tokens compact (no
    whitespace), 4089 chars / 1621 tokens pretty-printed (2-space indent,
    a formatting choice models commonly make even under schema-constrained
    decoding). The pretty-printed minimal answer ALONE (1621 tokens) already
    exceeded the old 1600-token TOTAL budget, before any preamble/formatting
    variance -- the old heuristic could not have succeeded regardless of
    model behavior. These two measured counts are hardcoded here (not
    re-tokenized at test time) so the regression does not depend on a
    tokenizer file being present in every environment that runs this suite.
    """

    _DECISION_SET_V1_MEASURED_COMPACT_TOKENS = 999
    _DECISION_SET_V1_MEASURED_PRETTY_TOKENS = 1621
    _OLD_FLAT_BUDGET = 1600  # _TOKENS_PER_QUESTION(64) * 24 + _TOKENS_OVERHEAD(64)

    def test_decision_set_v1_budget_clears_measured_pretty_json_with_margin(self):
        from src.typed_decisions.runner import _default_n_tokens

        n_tokens = _default_n_tokens(QUESTIONS_DECISION_SET_V1)

        assert n_tokens > self._OLD_FLAT_BUDGET
        assert n_tokens > self._DECISION_SET_V1_MEASURED_PRETTY_TOKENS
        # Comfortable margin left over for preamble/formatting variance, not
        # just enough to fit the minimal pretty-printed answer exactly.
        assert (
            n_tokens - self._DECISION_SET_V1_MEASURED_PRETTY_TOKENS
        ) >= self._DECISION_SET_V1_MEASURED_PRETTY_TOKENS * 0.2

    def test_budget_scales_with_label_count_not_just_question_count(self):
        # An all-noul (2-label) catalogue needs less than an equal-sized
        # all-choice (4-label) catalogue -- the flat per-question heuristic
        # this replaces was blind to that difference.
        from src.typed_decisions.runner import _default_n_tokens

        noul_only = tuple(
            Question(id=f"n{i}", kind=QuestionKind.NOUL, text="t") for i in range(8)
        )
        choice_only = tuple(
            Question(
                id=f"c{i}", kind=QuestionKind.CHOICE, text="t",
                options=("a", "b", "c", "d"),
            )
            for i in range(8)
        )
        assert _default_n_tokens(choice_only) > _default_n_tokens(noul_only)

    def test_never_below_min_floor(self):
        from src.typed_decisions.runner import _MIN_N_TOKENS, _default_n_tokens

        one_question = (Question(id="n1", kind=QuestionKind.NOUL, text="t"),)
        assert _default_n_tokens(one_question) >= _MIN_N_TOKENS


QUESTIONS_DECISION_SET_V1 = tuple(
    Question(
        id=item["id"],
        kind=item["kind"],
        text=item["text"],
        options=tuple(item.get("options", ())),
        levels=tuple(item.get("levels", ())),
    )
    for item in [
        {"id": "n01", "kind": "noul", "text": "Item B is divisible by 3."},
        {"id": "n02", "kind": "noul", "text": "Item D is a prime number."},
        {"id": "n03", "kind": "noul", "text": "Item F is a prime number."},
        {"id": "n04", "kind": "noul", "text": "Item A is a fish."},
        {"id": "n05", "kind": "noul", "text": "Item G is a bird."},
        {"id": "n06", "kind": "noul", "text": "Item E is a reptile."},
        {"id": "n07", "kind": "noul", "text": "The shipping region of Item C is vermont."},
        {"id": "n08", "kind": "noul", "text": "Item H is divisible by 3."},
        {
            "id": "c01", "kind": "choice", "text": "What is the animal type of Item A?",
            "options": ["fish", "bird", "mammal", "reptile"],
        },
        {
            "id": "c02", "kind": "choice", "text": "What is the animal type of Item C?",
            "options": ["fish", "bird", "mammal", "reptile"],
        },
        {
            "id": "c03", "kind": "choice", "text": "What is the animal type of Item E?",
            "options": ["fish", "bird", "mammal", "reptile"],
        },
        {
            "id": "c04", "kind": "choice", "text": "What is the animal type of Item G?",
            "options": ["fish", "bird", "mammal", "reptile"],
        },
        {
            "id": "c05", "kind": "choice", "text": "What is the shipping region of Item A?",
            "options": ["oregon", "vermont", "arizona", "alaska"],
        },
        {
            "id": "c06", "kind": "choice", "text": "What is the shipping region of Item C?",
            "options": ["oregon", "vermont", "arizona", "alaska"],
        },
        {
            "id": "c07", "kind": "choice", "text": "What is the shipping region of Item E?",
            "options": ["oregon", "vermont", "arizona", "alaska"],
        },
        {
            "id": "c08", "kind": "choice", "text": "What is the shipping region of Item G?",
            "options": ["oregon", "vermont", "arizona", "alaska"],
        },
        {
            "id": "s01", "kind": "score",
            "text": "Using the number priority rules, what is the priority of Item B?",
            "levels": [0, 1, 2, 3],
        },
        {
            "id": "s02", "kind": "score",
            "text": "Using the number priority rules, what is the priority of Item D?",
            "levels": [0, 1, 2, 3],
        },
        {
            "id": "s03", "kind": "score",
            "text": "Using the number priority rules, what is the priority of Item F?",
            "levels": [0, 1, 2, 3],
        },
        {
            "id": "s04", "kind": "score",
            "text": "Using the number priority rules, what is the priority of Item H?",
            "levels": [0, 1, 2, 3],
        },
        {
            "id": "s05", "kind": "score",
            "text": "Using the animal priority table, what is the priority of Item A?",
            "levels": [0, 1, 2, 3],
        },
        {
            "id": "s06", "kind": "score",
            "text": "Using the animal priority table, what is the priority of Item C?",
            "levels": [0, 1, 2, 3],
        },
        {
            "id": "s07", "kind": "score",
            "text": "Using the animal priority table, what is the priority of Item E?",
            "levels": [0, 1, 2, 3],
        },
        {
            "id": "s08", "kind": "score",
            "text": "Using the animal priority table, what is the priority of Item G?",
            "levels": [0, 1, 2, 3],
        },
    ]
)


# ── 2. Corrective retry recovers ──────────────────────────────────────────


class TestCorrectiveRetry:
    def test_malformed_first_response_recovers_via_one_corrective_retry(self):
        malformed = json.dumps({"answers": {}})
        valid = _valid_response(QUESTIONS)
        primitives = _FakePrimitives([malformed, valid])

        result = run_typed_decisions(
            primitives, state=STATE, questions=QUESTIONS, role=ROLE, max_retries=1
        )

        assert len(primitives.calls) == 2
        assert "CORRECTION" in primitives.calls[1]["prompt"]
        assert primitives.calls[1]["prompt"].startswith(primitives.calls[0]["prompt"])
        assert len(result.decisions) == 20
        assert result.raw_text == valid
        # The retry is recorded: attempt 1's schema failure stays visible.
        assert [failure.reason for failure in result.failures] == ["schema_violation"]

    def test_persistent_schema_violation_returns_typed_failure(self):
        primitives = _FakePrimitives(json.dumps({"answers": {}}))

        result = run_typed_decisions(
            primitives, state=STATE, questions=QUESTIONS, role=ROLE, max_retries=1
        )

        assert result.decisions == ()
        assert result.failures
        assert all(failure.reason == "schema_violation" for failure in result.failures)
        assert len(primitives.calls) == 1 + 1  # first attempt + max_retries
        assert "CORRECTION" in primitives.calls[1]["prompt"]

    def test_transport_error_short_circuits_without_retry(self):
        primitives = _FakePrimitives("[ERROR: connection refused]")

        result = run_typed_decisions(
            primitives, state=STATE, questions=QUESTIONS, role=ROLE, max_retries=1
        )

        assert result.decisions == ()
        assert isinstance(result.failures[0], ParseFailure)
        assert [failure.reason for failure in result.failures] == ["transport_error"]
        assert len(primitives.calls) == 1

    def test_no_json_emission_is_typed_and_retried(self):
        primitives = _FakePrimitives(["I cannot answer that.", _valid_response(QUESTIONS)])

        result = run_typed_decisions(
            primitives, state=STATE, questions=QUESTIONS, role=ROLE, max_retries=1
        )

        assert [failure.reason for failure in result.failures] == ["no_json"]
        assert len(result.decisions) == 20
        assert len(primitives.calls) == 2


# ── 3. Confidence formula vectors ─────────────────────────────────────────


class TestConfidenceFormulas:
    def test_choice_uniform_scores_zero(self):
        assert choice_confidence([0.25, 0.25, 0.25, 0.25]) == pytest.approx(0.0)
        assert choice_confidence([0.5, 0.5]) == pytest.approx(0.0)

    def test_choice_two_outcome_vector_scores_064(self):
        assert choice_confidence([0.82, 0.18]) == pytest.approx(0.64)

    def test_choice_single_outcome_scores_one(self):
        assert choice_confidence([1.0]) == 1.0

    def test_choice_mapping_input(self):
        assert choice_confidence({"a": 0.82, "b": 0.18}) == pytest.approx(0.64)

    def test_choice_empty_input_scores_zero(self):
        assert choice_confidence([]) == 0.0

    def test_score_degenerate_and_uniform(self):
        assert score_confidence({0: 1.0, 1: 0.0, 2: 0.0}) == pytest.approx(1.0)
        assert score_confidence({0: 1 / 3, 1: 1 / 3, 2: 1 / 3}) == pytest.approx(0.0)
        assert score_confidence([1.0, 0.0]) == pytest.approx(1.0)

    def test_score_moderate_peak_is_in_range(self):
        confidence = score_confidence({0: 0.7, 1: 0.2, 2: 0.1})
        assert confidence == pytest.approx(0.4)
        assert 0.0 <= confidence <= 1.0

    def test_normalize_rescales_invalid_sum(self):
        assert normalize_probabilities({"a": 0.6, "b": 0.6}) == {"a": 0.5, "b": 0.5}

    def test_normalize_zero_total_falls_back_to_uniform(self):
        assert normalize_probabilities({"a": 0.0, "b": 0.0}) == {"a": 0.5, "b": 0.5}

    def test_normalize_clamps_negative_and_non_finite(self):
        assert normalize_probabilities({"a": -1.0, "b": 2.0}) == {"a": 0.0, "b": 1.0}
        assert normalize_probabilities({"a": float("nan"), "b": 1.0}) == {"a": 0.0, "b": 1.0}

    def test_normalize_empty_stays_empty(self):
        assert normalize_probabilities({}) == {}


# ── 4. Schema and GBNF builders ───────────────────────────────────────────


class TestSchemaBuilders:
    def test_response_schema_is_valid_draft_2020_12(self):
        Draft202012Validator.check_schema(build_response_schema(QUESTIONS))

    def test_response_schema_requires_every_question_and_forbids_extras(self):
        questions = [QUESTIONS[0], next(q for q in QUESTIONS if q.kind is QuestionKind.SCORE)]
        schema = build_response_schema(questions)
        answers = schema["properties"]["answers"]

        assert schema["additionalProperties"] is False
        assert answers["additionalProperties"] is False
        assert answers["required"] == [question.id for question in questions]

        choice_schema = answers["properties"][questions[0].id]
        assert choice_schema["required"] == ["choice", "probabilities", "confidence"]
        assert choice_schema["properties"]["choice"]["enum"] == list(questions[0].options)
        assert choice_schema["properties"]["probabilities"]["required"] == list(
            questions[0].options
        )

        score_schema = answers["properties"][questions[1].id]
        assert score_schema["properties"]["score"] == {
            "type": "integer",
            "enum": list(questions[1].levels),
        }
        assert score_schema["properties"]["probabilities"]["required"] == ["0", "1", "2", "3"]

    def test_gbnf_returns_none_for_multi_token_candidates(self):
        question = Question(
            id="multi",
            kind=QuestionKind.CHOICE,
            text="Pick one.",
            options=("do the thing", "do nothing"),
        )
        assert build_gbnf([question]) is None

    def test_gbnf_for_single_token_candidates(self):
        questions = [
            Question(
                id="single",
                kind=QuestionKind.CHOICE,
                text="Pick a colour.",
                options=("red", "blue"),
            ),
            Question(id="flag", kind=QuestionKind.NOUL, text="Ship it?"),
        ]
        gbnf = build_gbnf(questions)

        assert gbnf is not None
        assert gbnf.startswith("root ::=")
        assert '"\\"red\\""' in gbnf
        assert '"\\"flag\\""' in gbnf
        assert "boolean" in gbnf

    def test_gbnf_returns_none_for_empty_catalogue(self):
        assert build_gbnf([]) is None


# ── 5. Prompt stability and mode contract ─────────────────────────────────


class TestPromptStability:
    def test_identical_inputs_produce_identical_prompt_sha256(self):
        first = _FakePrimitives(_valid_response(QUESTIONS))
        second = _FakePrimitives(_valid_response(QUESTIONS))

        first_result = run_typed_decisions(first, state=STATE, questions=QUESTIONS, role=ROLE)
        second_result = run_typed_decisions(second, state=STATE, questions=QUESTIONS, role=ROLE)

        assert first.calls[0]["prompt"] == second.calls[0]["prompt"]
        assert first_result.prompt_sha256 == second_result.prompt_sha256
        assert (
            first_result.prompt_sha256
            == hashlib.sha256(first.calls[0]["prompt"].encode("utf-8")).hexdigest()
        )

    def test_retry_does_not_change_the_canonical_prompt_hash(self):
        first_attempt = _FakePrimitives(["not json", _valid_response(QUESTIONS)])
        clean = _FakePrimitives(_valid_response(QUESTIONS))

        retried = run_typed_decisions(first_attempt, state=STATE, questions=QUESTIONS, role=ROLE)
        direct = run_typed_decisions(clean, state=STATE, questions=QUESTIONS, role=ROLE)

        assert retried.prompt_sha256 == direct.prompt_sha256


class TestModeContract:
    def test_native_mode_dispatches_to_the_native_runner(self):
        primitives = _FakePrimitives("")
        primitives._last_inference_meta = {
            "completion_probabilities": [
                # TD-1c: the question's cue is replayed as one fixed token,
                # so the answer row sits at index 1.
                {"id": 97, "token": "", "logprob": 0.0, "top_logprobs": []},
                {
                    "id": 1,
                    "token": "alpha",
                    "bytes": [97],
                    "logprob": -0.1,
                    "top_logprobs": [
                        {"id": 1, "token": "alpha", "bytes": [97], "logprob": -0.1},
                        {"id": 2, "token": "beta", "bytes": [98], "logprob": -2.0},
                    ],
                },
            ]
        }

        # TD-1b: native mode binds candidates to token ids through the
        # tokenizer seam; this test injects one so the dispatch path is
        # exercised without a live server. TD-1c reuses the seam for the cue
        # text, so anything unmapped (the cue) tokenizes to one opaque token.
        def tokenize(text: str) -> list[int]:
            return {
                "alpha": [1],
                " alpha": [11],
                "beta": [2],
                " beta": [12],
                "gamma": [3],
                " gamma": [13],
            }.get(text, [97])

        result = run_typed_decisions(
            primitives,
            state=STATE,
            questions=QUESTIONS[:1],
            role=ROLE,
            mode="native",
            tokenize_fn=tokenize,
        )

        assert result.mode == "native"
        assert [decision.value for decision in result.decisions] == ["alpha"]
        assert primitives.calls[0]["grammar"] is not None
        assert "json_schema" not in primitives.calls[0]

    def test_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError, match="unknown typed-decisions mode"):
            run_typed_decisions(
                _FakePrimitives(""),
                state=STATE,
                questions=QUESTIONS[:1],
                role=ROLE,
                mode="yolo",
            )

    def test_empty_catalogue_is_rejected(self):
        with pytest.raises(ValueError, match="at least one Question"):
            run_typed_decisions(_FakePrimitives(""), state=STATE, questions=[], role=ROLE)

    def test_duplicate_question_ids_are_rejected(self):
        duplicate = [QUESTIONS[0], QUESTIONS[0]]
        with pytest.raises(ValueError, match="duplicate question id"):
            run_typed_decisions(_FakePrimitives(""), state=STATE, questions=duplicate, role=ROLE)


# ── 6. Question invariants ────────────────────────────────────────────────


class TestQuestionInvariants:
    def test_choice_requires_two_unique_options(self):
        with pytest.raises(ValueError, match=">= 2 options"):
            Question(id="x", kind=QuestionKind.CHOICE, text="?", options=("only",))
        with pytest.raises(ValueError, match="duplicate options"):
            Question(id="x", kind=QuestionKind.CHOICE, text="?", options=("a", "a"))

    def test_score_requires_unique_integer_levels(self):
        with pytest.raises(ValueError, match="duplicate levels"):
            Question(id="x", kind=QuestionKind.SCORE, text="?", levels=(0, 0))
        with pytest.raises(ValueError, match="levels must be integers"):
            Question(id="x", kind=QuestionKind.SCORE, text="?", levels=(0, "1"))

    def test_kind_coerces_from_plain_string(self):
        question = Question(id="x", kind="noul", text="?")
        assert question.kind is QuestionKind.NOUL

    def test_option_descriptions_are_optional_ordered_choice_metadata(self):
        legacy = Question("x", QuestionKind.CHOICE, "Pick.", ("alpha", "beta"), (), ())
        described = Question(
            "x", QuestionKind.CHOICE, "Pick.", ("alpha", "beta"), (), (),
            ("first meaning", None),
        )

        assert legacy.option_descriptions == ()
        assert described.option_descriptions == ("first meaning", None)
        assert _build_catalog((legacy,)) == (
            "Q1 id=x kind=choice\n  question: Pick.\n  candidates: alpha | beta"
        )
        assert _build_catalog((described,)) == (
            "Q1 id=x kind=choice\n  question: Pick.\n  candidates: alpha | beta\n"
            "  candidate description [alpha]: first meaning"
        )
        assert build_response_schema((described,)) == build_response_schema((legacy,))

    @pytest.mark.parametrize("raw", ["xy", b"xy", bytearray(b"xy")])
    def test_option_descriptions_reject_text_as_the_outer_sequence(self, raw):
        with pytest.raises(ValueError, match="sequence, not text or bytes"):
            Question(
                id="x", kind=QuestionKind.CHOICE, text="Pick.",
                options=("alpha", "beta"), option_descriptions=raw,
            )

    @pytest.mark.parametrize("raw", [{"a": "first", "b": "second"}, {"first", "second"}])
    def test_option_descriptions_reject_unordered_outer_containers(self, raw):
        with pytest.raises(ValueError, match="preserve candidate order"):
            Question(
                id="x", kind=QuestionKind.CHOICE, text="Pick.",
                options=("alpha", "beta"), option_descriptions=raw,
            )

    def test_option_descriptions_validate_kind_arity_and_content(self):
        with pytest.raises(ValueError, match="same length and order"):
            Question(
                id="x", kind=QuestionKind.CHOICE, text="Pick.",
                options=("alpha", "beta"), option_descriptions=("only one",),
            )
        with pytest.raises(ValueError, match="only valid for choice"):
            Question(
                id="x", kind=QuestionKind.SCORE, text="Rate.", levels=(0, 1),
                option_descriptions=("low", "high"),
            )
        for invalid in ("", "   ", "first\nsecond", "first\rsecond", "first\u2028second", 7):
            with pytest.raises(ValueError):
                Question(
                    id="x", kind=QuestionKind.CHOICE, text="Pick.",
                    options=("alpha", "beta"), option_descriptions=(invalid, None),
                )

    def test_descriptions_do_not_change_resolved_choice_labels(self):
        question = Question(
            id="choice-described", kind=QuestionKind.CHOICE, text="Pick.",
            options=("alpha", "beta"), option_descriptions=("first", "second"),
        )
        primitives = _FakePrimitives(_valid_response((question,)))

        result = run_typed_decisions(
            primitives, state=STATE, questions=(question,), role=ROLE
        )

        assert result.decisions[0].value == "alpha"
        assert primitives.calls[0]["json_schema"] == build_response_schema((question,))
