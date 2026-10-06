"""Deterministic arithmetic/status fixtures for RC-10 confidence preparation."""

from dataclasses import FrozenInstanceError
from inspect import signature

import pytest

from src.typed_decisions.confidence_expectation import (
    PROBABILITY_SUM_ABS_TOLERANCE,
    SYNTHETIC_EXACT_COVERAGE,
    expected_score_from_result,
)
from src.typed_decisions.types import Decision, DecisionResult, ParseFailure, QuestionKind


ANSWER_QUALITY = {
    "choice_strings": ["A", "B", "C", "D", "E"],
    "choice_scores": {"A": 1.0, "B": 0.6, "C": 0.3, "D": 0.0, "E": 0.8},
}
ROUTING_OPTIMALITY = {
    "choice_strings": ["A", "B", "C", "D"],
    "choice_scores": {"A": 1.0, "B": 0.7, "C": 0.3, "D": 0.0},
}
SYNTHESIS_COHERENCE = {
    "choice_strings": ["A", "B", "C", "D"],
    "choice_scores": {"A": 1.0, "B": 0.7, "C": 0.3, "D": 0.0},
}


def _result(probabilities, *, value="A", kind=QuestionKind.CHOICE, failures=()):
    decision = Decision(
        question_id="grade",
        kind=kind,
        value=value,
        probabilities=probabilities,
        confidence=0.5,
        mode="synthetic-fixture",
    )
    return DecisionResult(
        decisions=(decision,),
        failures=tuple(failures),
        raw_text="synthetic fixture",
        mode="synthetic-fixture",
        elapsed_ms=0.0,
        prompt_sha256="fixture",
    )


def _evaluate(result, spec=ANSWER_QUALITY, provenance=SYNTHETIC_EXACT_COVERAGE):
    return expected_score_from_result(
        result,
        question_id="grade",
        grading_spec=spec,
        provenance=provenance,
    )


def test_uses_selected_spec_phi_in_host_expectation():
    result = _result({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.15, "E": 0.10})
    outcome = _evaluate(result)
    assert outcome.status == "computed"
    assert outcome.expectation == pytest.approx(0.555)
    assert outcome.decision is result.decisions[0]
    assert outcome.failures is result.failures


def test_uses_routing_spec_coefficients():
    result = _result({"A": 0.2, "B": 0.3, "C": 0.1, "D": 0.4})
    outcome = _evaluate(result, ROUTING_OPTIMALITY)
    assert outcome.status == "computed"
    assert outcome.expectation == pytest.approx(0.44)


def test_uses_synthesis_spec_coefficients():
    result = _result({"A": 0.2, "B": 0.3, "C": 0.1, "D": 0.4})
    outcome = _evaluate(result, SYNTHESIS_COHERENCE)
    assert outcome.status == "computed"
    assert outcome.expectation == pytest.approx(0.44)


@pytest.mark.parametrize(
    ("probabilities", "reason"),
    [
        ({"A": 0.5, "B": 0.5, "C": 0.0, "D": 0.0}, "probabilities_missing_labels"),
        (
            {"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2, "X": 0.0},
            "probabilities_extra_labels",
        ),
        ({"A": True, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0}, "probability_not_numeric"),
        ({"A": float("nan"), "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0}, "probability_nonfinite"),
        ({"A": -0.1, "B": 0.3, "C": 0.3, "D": 0.3, "E": 0.2}, "probability_negative"),
        (
            {
                "A": -2 * PROBABILITY_SUM_ABS_TOLERANCE,
                "B": 1.0 + 2 * PROBABILITY_SUM_ABS_TOLERANCE,
                "C": 0.0,
                "D": 0.0,
                "E": 0.0,
            },
            "probability_negative",
        ),
        (
            {"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2 + 2 * PROBABILITY_SUM_ABS_TOLERANCE},
            "probabilities_not_simplex",
        ),
        ({"A": 10**10000, "B": 0, "C": 0, "D": 0, "E": 0}, "probability_nonfinite"),
        ({"A": float("inf"), "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0}, "probability_nonfinite"),
        ({"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.1}, "probabilities_not_simplex"),
    ],
)
def test_bad_probability_map_fails_closed_without_normalization(probabilities, reason):
    outcome = _evaluate(_result(probabilities))
    assert outcome.status in {"missing", "unsupported", "unresolved"}
    assert outcome.expectation is None
    assert outcome.reason == reason


def test_accepts_declared_absolute_tolerance_without_changing_input():
    probabilities = {
        "A": 0.2,
        "B": 0.2,
        "C": 0.2,
        "D": 0.2,
        "E": 0.2 + PROBABILITY_SUM_ABS_TOLERANCE / 2,
    }
    before = probabilities.copy()
    result = _result(probabilities)
    decision = result.decisions[0]
    outcome = _evaluate(result)
    assert outcome.status == "computed"
    assert probabilities == before
    assert outcome.decision is decision


def test_wrong_provenance_and_wrong_question_identity_are_not_estimates():
    assert (
        _evaluate(
            _result({"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0}), provenance="native"
        ).status
        == "unsupported"
    )
    other = _result({"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0})
    changed = DecisionResult(
        decisions=(
            Decision(
                "other",
                QuestionKind.CHOICE,
                "A",
                other.decisions[0].probabilities,
                0.5,
                "synthetic-fixture",
            ),
        ),
        failures=(),
        raw_text="",
        mode="synthetic-fixture",
        elapsed_ms=0.0,
        prompt_sha256="",
    )
    assert _evaluate(changed).status == "missing"


def test_missing_spec_fields_and_missing_score_labels_are_explicit():
    probs = {"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2}
    assert (
        _evaluate(_result(probs), {"choice_strings": list(probs)}).reason
        == "grading_spec_fields_missing"
    )
    incomplete = {"choice_strings": list(probs), "choice_scores": {"A": 1, "B": 0.6}}
    assert _evaluate(_result(probs), incomplete).reason == "choice_scores_missing_labels"


def test_unresolved_decision_failures_are_preserved_and_recovered_decision_keeps_history():
    failure = ParseFailure("schema_violation", "attempt 1/2")
    unresolved = DecisionResult((), (failure,), "", "json", 0.0, "fixture")
    assert _evaluate(unresolved).status == "unresolved"
    assert _evaluate(unresolved).failures is unresolved.failures

    recovered = _result({"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0}, failures=(failure,))
    outcome = _evaluate(recovered)
    assert outcome.status == "computed"
    assert outcome.failures is recovered.failures
    assert outcome.failures == (failure,)


@pytest.mark.parametrize("value", [["A"], {"label": "A"}, "", "Z", None, 1])
def test_malformed_or_out_of_catalogue_decision_values_are_unsupported(value):
    result = _result({"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0}, value=value)
    outcome = _evaluate(result)
    assert outcome.status == "unsupported"
    assert outcome.reason in {"decision_value_invalid", "decision_value_outside_catalogue"}
    assert outcome.decision is result.decisions[0]


def test_duplicate_or_wrong_kind_decision_is_not_scored():
    base = _result({"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0})
    duplicate = DecisionResult(
        decisions=base.decisions * 2,
        failures=(),
        raw_text="",
        mode="synthetic-fixture",
        elapsed_ms=0.0,
        prompt_sha256="",
    )
    assert _evaluate(duplicate).reason == "decision_duplicate"
    wrong_kind = _result(
        {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0}, kind=QuestionKind.SCORE
    )
    assert _evaluate(wrong_kind).reason == "decision_kind_not_choice"


def test_discrete_decision_is_immutable_and_helper_has_no_call_seam():
    result = _result({"A": 0.6, "B": 0.1, "C": 0.1, "D": 0.1, "E": 0.1})
    decision = result.decisions[0]
    original = dict(decision.probabilities)
    outcome = _evaluate(result)
    assert outcome.decision is decision
    assert decision.value == "A"
    assert dict(decision.probabilities) == original
    with pytest.raises(FrozenInstanceError):
        decision.value = "B"
    assert tuple(signature(expected_score_from_result).parameters) == (
        "result",
        "question_id",
        "grading_spec",
        "provenance",
    )


def test_invalid_score_map_values_are_rejected_without_a_second_ladder():
    probabilities = {"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2}
    invalid = {
        "choice_strings": list(probabilities),
        "choice_scores": {**ANSWER_QUALITY["choice_scores"], "E": float("inf")},
    }
    outcome = _evaluate(_result(probabilities), invalid)
    assert outcome.status == "unsupported"
    assert outcome.reason == "choice_score_nonfinite"

    boolean = {
        "choice_strings": list(probabilities),
        "choice_scores": {**ANSWER_QUALITY["choice_scores"], "E": True},
    }
    outcome = _evaluate(_result(probabilities), boolean)
    assert outcome.status == "unsupported"
    assert outcome.reason == "choice_score_not_numeric"
