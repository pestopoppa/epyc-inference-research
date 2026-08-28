"""The screen estimator must use ONE statistic on BOTH arms.

The superseded rule centred on mean(anchor) and reported median(candidate effects)
against it. Across all 25 historical two-arm screens that mismatch injected
+2.014pp, flipped 10 signs, and took nominations from 3 to 7. These tests pin the
correction and are mutation-tested against the exact defect they exist to catch.
"""
from statistics import median, mean
import unittest

from scripts.benchmark import run_autokernel_gpu_discovery as gpu


# A realistic anchor arm: eight samples around 400 plus one cold-start low outlier,
# tuned so median(anchor)/mean(anchor) - 1 lands on the +1.96% measured across all 25
# historical screens (autokernel_rescore_estimator.py). This is the shape that made
# the superseded rule report improvement on every run.
ANCHOR = [330.0, 398.0, 399.0, 400.0, 400.5, 401.0, 401.5, 402.0, 403.0]
# A candidate that is genuinely NEUTRAL: same distribution as the anchor's steady state.
NEUTRAL_CANDIDATE = [398.5, 399.5, 400.0, 400.5, 401.0, 401.0, 401.5, 402.0, 402.5]


class ScreenEffectEstimator(unittest.TestCase):

    def test_estimator_is_median_over_median(self):
        got = gpu.screen_effect(
            anchor_samples=ANCHOR, anchor_runs=[], candidate_samples=NEUTRAL_CANDIDATE,
            candidate_runs=[], pair_max=False)
        self.assertEqual(got["estimator"], "median_over_median")
        self.assertAlmostEqual(
            got["effect"], median(NEUTRAL_CANDIDATE) / median(ANCHOR) - 1.0, places=12)
        self.assertAlmostEqual(got["center"], median(ANCHOR), places=12)
        self.assertAlmostEqual(got["candidate_statistic"], median(NEUTRAL_CANDIDATE),
                               places=12)

    def test_a_neutral_candidate_scores_near_zero(self):
        """The defect in one assertion: a candidate that is not faster must not read faster."""
        got = gpu.screen_effect(
            anchor_samples=ANCHOR, anchor_runs=[], candidate_samples=NEUTRAL_CANDIDATE,
            candidate_runs=[], pair_max=False)
        # Corrected: within half a percent of zero.
        self.assertLess(abs(got["effect"]), 0.005,
                        f"neutral candidate scored {got['effect']:+.4%}")
        # Superseded rule on the same samples: reports the candidate as faster purely
        # from the estimator mismatch. Production measured this at +1.96pp on average.
        self.assertGreater(got["legacy_median_relative"], 0.015,
                           "fixture no longer reproduces the historical bias; if the "
                           "anchor's cold-start outlier was removed this test is vacuous")
        self.assertGreater(got["legacy_median_relative"] - got["effect"], 0.015)
        # And the anchor's own median-vs-mean gap is the mechanism.
        self.assertAlmostEqual(
            got["anchor_median"] / got["anchor_mean"] - 1.0, 0.0196, places=3)

    def test_legacy_field_reproduces_the_superseded_rule_exactly(self):
        got = gpu.screen_effect(
            anchor_samples=ANCHOR, anchor_runs=[], candidate_samples=NEUTRAL_CANDIDATE,
            candidate_runs=[], pair_max=False)
        anchor_mean = mean(ANCHOR)
        self.assertAlmostEqual(
            got["legacy_median_relative"],
            median([(v - anchor_mean) / anchor_mean for v in NEUTRAL_CANDIDATE]),
            places=12)

    def test_a_genuinely_faster_candidate_still_reads_faster(self):
        """The correction must not be unfalsifiable in the positive direction."""
        faster = [v * 1.06 for v in NEUTRAL_CANDIDATE]
        got = gpu.screen_effect(
            anchor_samples=ANCHOR, anchor_runs=[], candidate_samples=faster,
            candidate_runs=[], pair_max=False)
        self.assertGreater(got["effect"], 0.03)

    def test_a_slower_candidate_reads_slower(self):
        slower = [v * 0.94 for v in NEUTRAL_CANDIDATE]
        got = gpu.screen_effect(
            anchor_samples=ANCHOR, anchor_runs=[], candidate_samples=slower,
            candidate_runs=[], pair_max=False)
        self.assertLess(got["effect"], -0.03)

    def test_pair_max_compares_like_statistics(self):
        got = gpu.screen_effect(
            anchor_samples=ANCHOR, anchor_runs=[{"metric": 400.0}],
            candidate_samples=NEUTRAL_CANDIDATE, candidate_runs=[{"metric": 412.0}],
            pair_max=True)
        self.assertEqual(got["estimator"], "pair_max_metric_over_pair_max_metric")
        self.assertAlmostEqual(got["effect"], 412.0 / 400.0 - 1.0, places=12)

    def test_pair_max_refuses_without_both_runs(self):
        with self.assertRaises(ValueError):
            gpu.screen_effect(anchor_samples=ANCHOR, anchor_runs=[],
                              candidate_samples=NEUTRAL_CANDIDATE, candidate_runs=[],
                              pair_max=True)

    def test_empty_arms_refuse_rather_than_score(self):
        """An empty input must not vacuously produce an effect."""
        with self.assertRaises(ValueError):
            gpu.screen_effect(anchor_samples=[], anchor_runs=[],
                              candidate_samples=NEUTRAL_CANDIDATE, candidate_runs=[],
                              pair_max=False)
        with self.assertRaises(ValueError):
            gpu.screen_effect(anchor_samples=ANCHOR, anchor_runs=[],
                              candidate_samples=[], candidate_runs=[], pair_max=False)

    def test_relative_effects_are_diagnostics_not_the_decision(self):
        """median(relative_effects) must NOT equal the reported effect.

        If it did, the centre would still be doing double duty and the mismatch
        would be back.
        """
        got = gpu.screen_effect(
            anchor_samples=ANCHOR, anchor_runs=[{"metric": 400.0}],
            candidate_samples=NEUTRAL_CANDIDATE, candidate_runs=[{"metric": 412.0}],
            pair_max=True)
        self.assertNotAlmostEqual(median(got["relative_effects"]), got["effect"], places=6)


if __name__ == "__main__":
    unittest.main()
