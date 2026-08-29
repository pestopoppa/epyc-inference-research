"""The decision arithmetic: what counts as a measurement at all.

Two rules carry everything: one statistic on BOTH arms, and an arm that is still
warming is not resolving anything.
"""
from pathlib import Path
import unittest

from autokernel.loop import bench

class ADriftingArmIsNotAMeasurement(unittest.TestCase):
    """The force-MMQ probe's real failure mode, pinned with its own numbers.

    The candidate climbed +4.324% across five pairs while the anchor stayed flat, and
    per-pair effects marched -4.491% -> -0.037%. `spread_is_suspect` missed it entirely:
    a monotonic trend gives max/min = 1.043, far under the 1.3 bimodality bar. The
    headline -1.469% described first-use cost, not throughput.
    """

    #: The measured samples from that probe, verbatim.
    ANCHOR = [12658.0, 12598.3, 12552.7, 12626.8, 12617.1]
    CANDIDATE = [12089.6, 12431.7, 12413.0, 12613.1, 12612.4]

    def _comparison(self, anchor, candidate, floor=0.973):
        import statistics as st
        return bench.Comparison(
            surface="pp512", anchor_samples=anchor, candidate_samples=candidate,
            effect=(st.median(candidate) / st.median(anchor)) - 1.0,
            estimator="median_over_median", pairs=len(anchor),
            noise_floor_pct=floor, residency={},
            anchor_drift_pct=bench.drift_pct(anchor),
            candidate_drift_pct=bench.drift_pct(candidate))

    def test_the_bimodality_check_does_not_catch_this(self):
        self.assertFalse(bench.spread_is_suspect(self.CANDIDATE),
                         "a monotonic warm-up is not bimodal; that is why it slipped")

    def test_the_drift_detector_does(self):
        self.assertGreater(bench.drift_pct(self.CANDIDATE), 2.0)
        self.assertLess(abs(bench.drift_pct(self.ANCHOR)), 1.0)

    def test_a_drifting_arm_makes_the_result_not_decisive(self):
        comparison = self._comparison(self.ANCHOR, self.CANDIDATE)
        self.assertTrue(comparison.drifting)
        self.assertLess(comparison.effect * 100.0, -1.0,
                        "the raw effect still looks large")
        self.assertFalse(comparison.decisive,
                         "an arm that is still warming resolves nothing")

    def test_two_settled_arms_are_still_decisive(self):
        anchor = [100.0, 100.2, 99.9, 100.1, 100.0]
        candidate = [105.0, 105.1, 104.9, 105.2, 105.0]
        comparison = self._comparison(anchor, candidate)
        self.assertFalse(comparison.drifting)
        self.assertTrue(comparison.decisive,
                        "the drift veto must not swallow a real effect")

    def test_compare_runs_and_discards_warmup_pairs(self):
        from unittest import mock
        calls = []

        def fake_run_once(binary, model, *, pp, tg, reps=9, timeout_s=3600):
            calls.append(str(binary))
            return 100.0, {"resident": True, "peak_vram_bytes": 1 << 31,
                           "peak_kfd_processes": 1}

        with mock.patch.object(bench, "run_once", side_effect=fake_run_once):
            bench.compare(bench.Arm("a", Path("/a")), bench.Arm("c", Path("/c")),
                          Path("/m.gguf"), pp=512, tg=0, pairs=2,
                          noise_floor_pct=1.0, warmup_pairs=1)
        # 1 discarded pair + 2 measured pairs, two arms each.
        self.assertEqual(len(calls), (1 + 2) * 2)


class ADriftVetoMustNotReadAsACleanNull(unittest.TestCase):
    """Run 10 recorded `akm-q8-1-wave64-eight-block` as "effect +1.126% did not clear
    the 1.175% noise floor". True on its face and misleading: that run was VETOED for
    drift (anchor +1.302%). The mechanism reads as tested and unpromising when it was
    never resolved, which invites the planner to abandon a live idea -- the same class
    of error as a fabricated refusal.
    """

    def _drifting(self):
        return bench.Comparison(
            surface="tg128", anchor_samples=[1.0], candidate_samples=[1.0],
            effect=0.01126, estimator="median_over_median", pairs=9,
            noise_floor_pct=1.175, residency={},
            anchor_drift_pct=1.302, candidate_drift_pct=0.855)

    def _settled(self):
        return bench.Comparison(
            surface="tg128", anchor_samples=[1.0], candidate_samples=[1.0],
            effect=0.00109, estimator="median_over_median", pairs=9,
            noise_floor_pct=1.175, residency={},
            anchor_drift_pct=0.483, candidate_drift_pct=0.104)

    def test_a_drift_veto_says_UNTESTED_not_unpromising(self):
        from autokernel.loop import loop as loop_mod
        reason = loop_mod._null_reason(self._drifting())
        self.assertIn("NOT RESOLVED", reason)
        self.assertIn("UNTESTED", reason)
        self.assertIn("re-run", reason)
        self.assertIn("+1.302", reason, "name the arm that moved")

    def test_a_settled_null_still_reads_as_a_null(self):
        from autokernel.loop import loop as loop_mod
        reason = loop_mod._null_reason(self._settled())
        self.assertIn("did not clear", reason)
        self.assertNotIn("NOT RESOLVED", reason)


class TheClockIsPartOfTheEvidence(unittest.TestCase):
    """gfx90a exposes 500/800/1700 MHz and the host runs performance level `auto`, so a
    benchmark can begin at 800 and ramp to 1700 -- a 2.125x clock change mid-measurement.
    That is a candidate mechanism for the drift the veto keeps catching, so a drifting
    result must be diagnosable from its own record rather than re-investigated by hand.
    """

    def test_the_sampler_reports_the_clock_range_and_stability(self):
        from autokernel.loop import residency
        sampler = residency.Sampler()
        sampler.min_sclk, sampler.max_sclk = 800, 1700
        proof = sampler.proof
        self.assertEqual(proof["sclk_min_mhz"], 800)
        self.assertEqual(proof["sclk_max_mhz"], 1700)
        self.assertFalse(proof["clock_stable"], "800 -> 1700 is not a stable clock")

    def test_a_steady_clock_reads_stable(self):
        from autokernel.loop import residency
        sampler = residency.Sampler()
        sampler.min_sclk = sampler.max_sclk = 1700
        self.assertTrue(sampler.proof["clock_stable"])

    def test_an_unread_clock_is_not_reported_as_stable(self):
        """Zero means the sysfs was unreadable, not that the clock never moved."""
        from autokernel.loop import residency
        self.assertFalse(residency.Sampler().proof["clock_stable"])
