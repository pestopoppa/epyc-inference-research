"""Reference-normalized log-time scoring and matched budget curves.

This is the transferable scoring protocol from RE-Bench, not an adoption of
its H100 Triton environment or its published human/model anchors.  A score of
zero is the measured starting implementation and one is the measured strong
reference implementation.  Correctness is a hard precondition.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .. import schemas


SCHEMA = "epyc.autokernel.rebench_log_time.v1"
DEFAULT_BUDGET_HOURS = (2.0, 8.0, 32.0)


def _positive_finite(value: object, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


@dataclass(frozen=True)
class LogTimeScore:
    baseline_seconds: float
    reference_seconds: float
    candidate_seconds: float
    score: float | None
    behavior_check: schemas.Check

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "baseline_seconds": self.baseline_seconds,
            "reference_seconds": self.reference_seconds,
            "candidate_seconds": self.candidate_seconds,
            "score": self.score,
            "anchors": {"starting_state": 0.0, "strong_reference": 1.0},
            "behavior_check": {
                "outcome": self.behavior_check.outcome,
                "reasons": list(self.behavior_check.reasons),
            },
        }


def score_log_time(
        *, baseline_seconds: float, reference_seconds: float,
        candidate_seconds: float, behavior_check: schemas.Check) -> LogTimeScore:
    """Score a behavior-preserving runtime on the baseline-to-reference axis.

    The score is ``log(candidate / baseline) / log(reference / baseline)``.
    It is deliberately not clipped: a candidate may be worse than the start
    (negative) or stronger than the reference (>1).  A non-PASS behavior check
    withholds the score instead of turning incorrect work into a fast result.
    """
    baseline = _positive_finite(baseline_seconds, "baseline_seconds")
    reference = _positive_finite(reference_seconds, "reference_seconds")
    candidate = _positive_finite(candidate_seconds, "candidate_seconds")
    if not isinstance(behavior_check, schemas.Check):
        raise TypeError("behavior_check must be a schemas.Check")
    if reference >= baseline:
        raise ValueError("the strong reference must be faster than the starting state")
    if behavior_check.outcome != schemas.PASS:
        return LogTimeScore(
            baseline, reference, candidate, None, behavior_check)
    score = math.log(candidate / baseline) / math.log(reference / baseline)
    return LogTimeScore(baseline, reference, candidate, score, behavior_check)


@dataclass(frozen=True)
class TimedAttempt:
    completed_hours: float
    score: LogTimeScore

    def __post_init__(self) -> None:
        _positive_finite(self.completed_hours, "completed_hours")
        if not isinstance(self.score, LogTimeScore):
            raise TypeError("score must be a LogTimeScore")


@dataclass(frozen=True)
class BudgetPoint:
    budget_hours: float
    best_score: float | None
    eligible_attempts: int

    def to_dict(self) -> dict:
        return {
            "budget_hours": self.budget_hours,
            "best_score": self.best_score,
            "eligible_attempts": self.eligible_attempts,
        }


def time_budget_curve(
        attempts: Sequence[TimedAttempt], *,
        budgets_hours: Sequence[float] = DEFAULT_BUDGET_HOURS,
        ) -> tuple[BudgetPoint, ...]:
    """Return best behavior-preserving score available at each matched budget."""
    budgets = tuple(_positive_finite(value, "budget_hours")
                    for value in budgets_hours)
    if not budgets or any(right <= left for left, right in zip(budgets, budgets[1:])):
        raise ValueError("budgets_hours must be a non-empty strictly increasing sequence")
    if any(not isinstance(attempt, TimedAttempt) for attempt in attempts):
        raise TypeError("attempts must contain TimedAttempt values")
    points = []
    for budget in budgets:
        admitted = [attempt.score.score for attempt in attempts
                    if attempt.completed_hours <= budget
                    and attempt.score.score is not None]
        points.append(BudgetPoint(
            budget_hours=budget,
            best_score=max(admitted) if admitted else None,
            eligible_attempts=len(admitted)))
    return tuple(points)
