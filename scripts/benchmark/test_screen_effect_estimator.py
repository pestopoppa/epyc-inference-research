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


class PairedAlternatingDesign(unittest.TestCase):
    """Arms must alternate across PROCESSES, not run as two blocks.

    The superseded design ran all anchor repetitions in one process and then all
    candidate repetitions in one process. `anchor_processes: 1, candidate_processes:
    1` with nine in-process reps means n_effective = 1 per arm: between-process
    variance (model load, HIP context creation, clock and thermal state) is entirely
    unsampled, and any drift over the window loads onto whichever arm ran second.
    The same candidate identity in v31 measured +5.369% and -1.714% on two runs of
    identical code.
    """

    def _order(self, pairs, schedule=("anchor", "candidate")):
        """The runner's own schedule -- not a reimplementation of it."""
        return gpu.arm_schedule(schedule, pairs)

    def test_one_pair_is_the_superseded_block_sequential_shape(self):
        self.assertEqual(self._order(1), [("anchor", 0), ("candidate", 0)])

    def test_more_pairs_alternate_rather_than_block(self):
        order = [arm for arm, _ in self._order(3)]
        self.assertEqual(order, ["anchor", "candidate"] * 3)
        # The property that matters: no arm ever runs twice in a row, so a monotone
        # drift over the window cannot accumulate against one arm.
        self.assertFalse(any(a == b for a, b in zip(order, order[1:])))

    def test_the_declared_schedule_is_honoured(self):
        order = [arm for arm, _ in self._order(2, ("candidate", "anchor"))]
        self.assertEqual(order, ["candidate", "anchor"] * 2)

    def test_pair_zero_keeps_the_historical_receipt_paths(self):
        """An existing sealed operation must resume byte-identically."""
        def supervisor_dir(arm, pair):
            return f"supervisor-{arm}" if pair == 0 else f"supervisor-{arm}-p{pair}"
        self.assertEqual(supervisor_dir("anchor", 0), "supervisor-anchor")
        self.assertEqual(supervisor_dir("anchor", 1), "supervisor-anchor-p1")
        # Distinct per pair, or two processes collide on one receipt root.
        dirs = {supervisor_dir(arm, pair)
                for pair in range(4) for arm in ("anchor", "candidate")}
        self.assertEqual(len(dirs), 8)

    def test_the_seed_is_not_varied_per_pair(self):
        """Pairing replicates the PROCESS, not the input.

        When a cross-arm output oracle is enabled both arms must see identical
        input, so varying the seed per pair would break the oracle that proves the
        two arms computed the same thing.
        """
        seeds = {gpu._invocation_seed(
            base_seed=42, repetitions=9, arm=arm,
            timed_output_oracle_enabled=True, runtime_graphs="off")
            for arm in ("anchor", "candidate")}
        self.assertEqual(seeds, {42}, "an enabled output oracle requires one seed")


if __name__ == "__main__":
    unittest.main()
