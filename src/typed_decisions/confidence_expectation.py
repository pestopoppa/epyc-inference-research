"""Pure host-side expectation over an explicitly synthetic candidate simplex.

This module prepares arithmetic for RC-10 fixtures only. Its provenance gate is
an assertion from the caller, not evidence that a live model readout matches
EV-15b/c. In particular, native candidate probabilities do not carry retained
mass here and must not be treated as the historical T=1 score-token instrument.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from src.typed_decisions.types import Decision, DecisionResult, ParseFailure, QuestionKind


SYNTHETIC_EXACT_COVERAGE = "synthetic_exact_coverage"
PROBABILITY_SUM_ABS_TOLERANCE = 1e-9
ExpectationStatus = Literal["computed", "unsupported", "missing", "unresolved"]


@dataclass(frozen=True)
class ConfidenceExpectation:
    """Host-computed expected score with its original typed evidence preserved."""

    status: ExpectationStatus
    expectation: float | None
    reason: str | None
    decision: Decision | None
    failures: tuple[ParseFailure, ...]


def expected_score_from_result(
    result: DecisionResult,
    *,
    question_id: str,
    grading_spec: Mapping[str, Any],
    provenance: str,
) -> ConfidenceExpectation:
    """Compute ``sum(p[label] * choice_scores[label])`` on exact synthetic input.

    This pure helper never calls a model and never changes the discrete
    ``Decision`` or ``DecisionResult``. The caller must supply the exact
    selected grading spec and explicitly mark the probability map as
    ``synthetic_exact_coverage``. Missing values are never filled and the
    probability map is never normalized.
    """
    failures = result.failures
    if provenance != SYNTHETIC_EXACT_COVERAGE:
        return _outcome("unsupported", "provenance_not_synthetic_exact_coverage", failures)
    if not isinstance(question_id, str) or not question_id:
        return _outcome("unsupported", "invalid_question_id", failures)
    if not isinstance(grading_spec, Mapping):
        return _outcome("unsupported", "grading_spec_not_mapping", failures)

    if "choice_strings" not in grading_spec or "choice_scores" not in grading_spec:
        return _outcome("missing", "grading_spec_fields_missing", failures)
    raw_labels = grading_spec["choice_strings"]
    raw_scores = grading_spec["choice_scores"]
    if not isinstance(raw_labels, (list, tuple)) or not raw_labels:
        return _outcome("unsupported", "choice_strings_invalid", failures)
    if any(not isinstance(label, str) or not label for label in raw_labels):
        return _outcome("unsupported", "choice_label_invalid", failures)
    labels = tuple(raw_labels)
    if len(set(labels)) != len(labels):
        return _outcome("unsupported", "choice_labels_duplicate", failures)
    if not isinstance(raw_scores, Mapping):
        return _outcome("unsupported", "choice_scores_not_mapping", failures)
    score_labels = set(raw_scores)
    label_set = set(labels)
    if score_labels - label_set:
        return _outcome("unsupported", "choice_scores_extra_labels", failures)
    if label_set - score_labels:
        return _outcome("missing", "choice_scores_missing_labels", failures)

    scores: dict[str, float] = {}
    for label in labels:
        value = raw_scores[label]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return _outcome("unsupported", "choice_score_not_numeric", failures)
        try:
            score = float(value)
        except OverflowError:
            return _outcome("unsupported", "choice_score_nonfinite", failures)
        if not math.isfinite(score):
            return _outcome("unsupported", "choice_score_nonfinite", failures)
        scores[label] = score

    matching = [decision for decision in result.decisions if decision.question_id == question_id]
    if not matching:
        if result.failures:
            return _outcome("unresolved", "decision_unresolved_with_failures", failures)
        return _outcome("missing", "decision_missing", failures)
    if len(matching) != 1:
        return _outcome("unresolved", "decision_duplicate", failures)

    decision = matching[0]
    if decision.kind is not QuestionKind.CHOICE:
        return _outcome("unsupported", "decision_kind_not_choice", failures, decision)
    if not isinstance(decision.value, str) or not decision.value:
        return _outcome("unsupported", "decision_value_invalid", failures, decision)
    if decision.value not in label_set:
        return _outcome("unsupported", "decision_value_outside_catalogue", failures, decision)
    probabilities = decision.probabilities
    if not isinstance(probabilities, Mapping):
        return _outcome("missing", "probabilities_missing", failures, decision)
    probability_labels = set(probabilities)
    if probability_labels - label_set:
        return _outcome("unsupported", "probabilities_extra_labels", failures, decision)
    if label_set - probability_labels:
        return _outcome("missing", "probabilities_missing_labels", failures, decision)

    checked: dict[str, float] = {}
    for label in labels:
        value = probabilities[label]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return _outcome("unresolved", "probability_not_numeric", failures, decision)
        try:
            probability = float(value)
        except OverflowError:
            return _outcome("unresolved", "probability_nonfinite", failures, decision)
        if not math.isfinite(probability):
            return _outcome("unresolved", "probability_nonfinite", failures, decision)
        if probability < 0.0:
            return _outcome("unresolved", "probability_negative", failures, decision)
        checked[label] = probability

    total = sum(checked.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=PROBABILITY_SUM_ABS_TOLERANCE):
        return _outcome("unresolved", "probabilities_not_simplex", failures, decision)

    expectation = sum(checked[label] * scores[label] for label in labels)
    if not math.isfinite(expectation):
        return _outcome("unresolved", "expectation_nonfinite", failures, decision)
    return ConfidenceExpectation(
        status="computed",
        expectation=expectation,
        reason=None,
        decision=decision,
        failures=failures,
    )


def _outcome(
    status: ExpectationStatus,
    reason: str,
    failures: tuple[ParseFailure, ...],
    decision: Decision | None = None,
) -> ConfidenceExpectation:
    return ConfidenceExpectation(
        status=status,
        expectation=None,
        reason=reason,
        decision=decision,
        failures=failures,
    )
