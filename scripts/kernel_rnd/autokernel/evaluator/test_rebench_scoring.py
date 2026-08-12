from __future__ import annotations

import math
import unittest

from .. import schemas
from . import rebench_scoring as R


PASS = schemas.Check(schemas.PASS)


class TestLogTimeScore(unittest.TestCase):
    def test_start_and_reference_are_zero_and_one(self):
        start = R.score_log_time(
            baseline_seconds=10.0, reference_seconds=5.0,
            candidate_seconds=10.0, behavior_check=PASS)
        reference = R.score_log_time(
            baseline_seconds=10.0, reference_seconds=5.0,
            candidate_seconds=5.0, behavior_check=PASS)
        self.assertAlmostEqual(start.score, 0.0)
        self.assertAlmostEqual(reference.score, 1.0)

    def test_log_interpolation_and_beyond_reference_are_not_clipped(self):
        middle = R.score_log_time(
            baseline_seconds=10.0, reference_seconds=5.0,
            candidate_seconds=math.sqrt(50.0), behavior_check=PASS)
        stronger = R.score_log_time(
            baseline_seconds=10.0, reference_seconds=5.0,
            candidate_seconds=2.5, behavior_check=PASS)
        worse = R.score_log_time(
            baseline_seconds=10.0, reference_seconds=5.0,
            candidate_seconds=20.0, behavior_check=PASS)
        self.assertAlmostEqual(middle.score, 0.5)
        self.assertAlmostEqual(stronger.score, 2.0)
        self.assertLess(worse.score, 0.0)

    def test_incorrect_or_unverified_work_has_no_score(self):
        for outcome in (schemas.FAIL, schemas.COULD_NOT_CHECK):
            report = R.score_log_time(
                baseline_seconds=10.0, reference_seconds=5.0,
                candidate_seconds=1.0,
                behavior_check=schemas.Check(outcome, ("not behavior preserving",)))
            self.assertIsNone(report.score)

    def test_invalid_runtime_or_reference_refuses(self):
        with self.assertRaisesRegex(ValueError, "strong reference"):
            R.score_log_time(
                baseline_seconds=10.0, reference_seconds=10.0,
                candidate_seconds=9.0, behavior_check=PASS)
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            R.score_log_time(
                baseline_seconds=10.0, reference_seconds=5.0,
                candidate_seconds=0.0, behavior_check=PASS)


class TestBudgetCurve(unittest.TestCase):
    @staticmethod
    def attempt(hours: float, seconds: float,
                outcome: str = schemas.PASS) -> R.TimedAttempt:
        return R.TimedAttempt(
            completed_hours=hours,
            score=R.score_log_time(
                baseline_seconds=10.0, reference_seconds=5.0,
                candidate_seconds=seconds,
                behavior_check=schemas.Check(outcome)))

    def test_curve_is_best_so_far_at_matched_budgets(self):
        points = R.time_budget_curve((
            self.attempt(1.0, 8.0), self.attempt(3.0, 6.0),
            self.attempt(7.0, 1.0, schemas.FAIL), self.attempt(20.0, 4.0)))
        self.assertEqual([point.budget_hours for point in points], [2.0, 8.0, 32.0])
        self.assertEqual([point.eligible_attempts for point in points], [1, 2, 3])
        self.assertLess(points[0].best_score, points[1].best_score)
        self.assertLess(points[1].best_score, points[2].best_score)

    def test_empty_prefix_and_bad_budgets_are_explicit(self):
        points = R.time_budget_curve((self.attempt(3.0, 6.0),))
        self.assertIsNone(points[0].best_score)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            R.time_budget_curve((), budgets_hours=(8.0, 2.0))


if __name__ == "__main__":
    unittest.main()
