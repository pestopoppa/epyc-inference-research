"""The decision arithmetic: what counts as a measurement at all.

Two rules carry everything: one statistic on BOTH arms, and an arm that is still
warming is not resolving anything.
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

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

        def fake_run_once(binary, model, *, pp, tg, ubatch=None, reps=9,
                          timeout_s=3600):
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

    #: A genuinely non-stationary arm: monotone in position, rho = 1.0.
    RAMPING = [260.0, 262.0, 264.0, 266.0, 268.0, 270.0, 272.0, 274.0, 276.0]
    #: The measured anchor of run 11's akm-fattn-single-partition-direct. Its
    #: median-of-halves drift was +1.599% and it was vetoed; its rank trend is
    #: rho=+0.100, p=0.81 -- no trend at all. The arm minimum sits at position 7.
    NOISY_BUT_FLAT = [261.27, 273.79, 266.99, 264.99, 271.83, 276.37, 260.66,
                      270.29, 268.96]

    def _comparison(self, anchor, effect):
        return bench.Comparison(
            surface="tg128", anchor_samples=anchor, candidate_samples=list(anchor),
            effect=effect, estimator="median_over_median", pairs=9,
            noise_floor_pct=1.175, residency={},
            anchor_drift_pct=bench.drift_pct(anchor),
            candidate_drift_pct=bench.drift_pct(anchor))

    def _drifting(self):
        return self._comparison(self.RAMPING, 0.01126)

    def _settled(self):
        return self._comparison(self.NOISY_BUT_FLAT, 0.00109)

    def test_the_flagship_false_veto_now_resolves(self):
        """Real samples, not synthetic drift fields: this arm was vetoed at +1.599%
        median-of-halves while having no rank trend whatsoever."""
        self.assertGreater(abs(bench.drift_pct(self.NOISY_BUT_FLAT)), 1.175,
                           "the OLD gate fired on this arm")
        self.assertLess(abs(bench.trend_rho(self.NOISY_BUT_FLAT)), 0.700,
                        "the NEW gate must not")
        self.assertFalse(self._comparison(self.NOISY_BUT_FLAT, 0.02621).drifting)

    def test_a_real_ramp_is_still_caught(self):
        self.assertGreaterEqual(abs(bench.trend_rho(self.RAMPING)), 0.700)
        self.assertTrue(self._comparison(self.RAMPING, 0.01126).drifting)

    def test_a_drift_veto_says_UNTESTED_not_unpromising(self):
        from autokernel.loop import loop as loop_mod
        reason = loop_mod._null_reason(self._drifting())
        self.assertIn("NOT RESOLVED", reason)
        self.assertIn("UNTESTED", reason)
        self.assertIn("re-run", reason)
        expected = f"{bench.drift_pct(self.RAMPING):+.3f}"
        self.assertIn(expected, reason, "name the arm that moved, with its size")

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


class AnExternalKillIsRetriedAndACrashIsNot(unittest.TestCase):
    """Run 12 died because llama-bench came back rc=-9 mid-measurement. earlyoom on
    this host ignores llama-server and NOT llama-bench, so a memory-pressure reaper
    ending a benchmark is a standing hazard we cannot fix from userspace -- only
    survive.

    A crash is different in kind: -11 is the CANDIDATE failing, and retrying it into
    looking healthy would fabricate a result."""

    def _run_with(self, codes):
        from unittest import mock
        seen = iter(codes)
        slept = []

        def fake(*a, **k):
            return mock.Mock(returncode=next(seen), stdout="[]", stderr="killed")

        with mock.patch.object(bench.subprocess, "run", side_effect=fake), \
             mock.patch.object(bench.residency, "Sampler"), \
             mock.patch.object(bench.residency, "loader_env", return_value={}):
            try:
                bench.run_once(Path("/b"), Path("/m"), pp=0, tg=128,
                               sleep=slept.append)
            except bench.BenchFailed as exc:
                return slept, str(exc)
        return slept, None

    def test_a_sigkill_is_retried(self):
        slept, error = self._run_with([-9, -9, -9, -9])
        self.assertEqual(len(slept), bench.KILL_RETRIES, "must back off and retry")
        self.assertIn("external kill", error)

    def test_a_segfault_is_never_retried(self):
        slept, error = self._run_with([-11])
        self.assertEqual(slept, [], "a crashing candidate must not be retried")
        self.assertNotIn("external kill", error)

    def test_a_kill_that_then_succeeds_costs_only_the_backoff(self):
        from unittest import mock
        seen = iter([-9, 0])
        slept = []
        rows = '[{"n_prompt":0,"n_gen":128,"avg_ts":100.0}]'

        def fake(*a, **k):
            code = next(seen)
            return mock.Mock(returncode=code, stdout=rows, stderr="")

        with mock.patch.object(bench.subprocess, "run", side_effect=fake), \
             mock.patch.object(bench.residency, "Sampler"), \
             mock.patch.object(bench.residency, "loader_env", return_value={}):
            value, _ = bench.run_once(Path("/b"), Path("/m"), pp=0, tg=128,
                                      sleep=slept.append)
        self.assertEqual(value, 100.0)
        self.assertEqual(len(slept), 1, "one kill, one backoff, then the measurement")


class DriftMustBeBigEnoughToExplainTheEffect(unittest.TestCase):
    """A trend test says an arm MOVED. It says nothing about whether the movement is
    large enough to be the answer.

    Run 14 vetoed a +6.293% result on a candidate drift of +1.049% -- 17% of the
    effect. Subtracting ALL of it still leaves +5.244%, over four times the floor.
    That is a strong effect on a slightly moving arm, not an unresolved measurement.

    The veto exists because the force-MMQ probe's +4.324% ramp WAS the whole result.
    Both cases must come out right."""

    #: Run 14's akm-q4k-reuse-q8-sum, verbatim.
    def _big_effect_small_drift(self):
        return bench.Comparison(
            surface="tg128", anchor_samples=[100.0] * 20,
            candidate_samples=[106.3] * 20, effect=0.06293,
            estimator="median_over_median", pairs=20, noise_floor_pct=1.188,
            residency={}, anchor_drift_pct=0.561, candidate_drift_pct=1.049)

    def _drift_is_the_result(self):
        """The force-MMQ shape: the drift is as big as the effect."""
        return bench.Comparison(
            surface="pp512", anchor_samples=[100.0] * 5,
            candidate_samples=[101.4] * 5, effect=0.01469,
            estimator="median_over_median", pairs=5, noise_floor_pct=0.973,
            residency={}, anchor_drift_pct=-0.324, candidate_drift_pct=4.324)

    def test_a_small_drift_does_not_veto_a_large_effect(self):
        comparison = self._big_effect_small_drift()
        self.assertFalse(comparison.drift_explains_the_effect,
                         "1.049% cannot have manufactured 6.293%")

    def test_a_drift_the_size_of_the_effect_still_vetoes(self):
        comparison = self._drift_is_the_result()
        self.assertTrue(comparison.drift_explains_the_effect,
                        "4.324% absolutely can manufacture 1.469%")

    def test_the_threshold_is_a_fraction_of_the_effect_not_the_floor(self):
        """Comparing drift to the FLOOR is what over-vetoed: the floor knows nothing
        about how big the effect is."""
        comparison = self._big_effect_small_drift()
        self.assertGreater(abs(comparison.candidate_drift_pct),
                           comparison.noise_floor_pct * 0.8,
                           "this drift is comparable to the floor")
        self.assertFalse(comparison.drift_explains_the_effect,
                         "yet it is small next to the effect, which is what matters")


class AnUncalibratedSurfaceRefusesToDecide(unittest.TestCase):
    """The run-22 surface-extension discipline, pinned against the historical defect.

    pp512 once carried a stale floor row and the loop published fake `decisive=True`
    off it. So the gate is the CALIBRATED flag, never the mere presence of a floor
    number: a numeric floor on an uncalibrated surface must still refuse to decide,
    and the refusal is None ("undecidable"), not False ("did not clear").
    """

    SETTLED_A = [100.0, 100.2, 99.9, 100.1, 100.0]
    SETTLED_C = [110.0, 110.1, 109.9, 110.2, 110.0]      # +10%, far over any floor

    def _comparison(self, *, calibrated, floor=1.0):
        import statistics as st
        return bench.Comparison(
            surface="dec-b4", anchor_samples=self.SETTLED_A,
            candidate_samples=self.SETTLED_C,
            effect=(st.median(self.SETTLED_C) / st.median(self.SETTLED_A)) - 1.0,
            estimator="median_over_median", pairs=5, noise_floor_pct=floor,
            residency={}, calibrated=calibrated)

    def test_a_fake_floor_cannot_make_an_uncalibrated_surface_decisive(self):
        """THE historical defect: floor number present, calibration absent."""
        comparison = self._comparison(calibrated=False, floor=1.0)
        self.assertIsNone(comparison.decisive,
                          "a floor nobody calibrated must not decide anything")

    def test_decisive_none_is_falsy_so_every_keep_gate_refuses_it(self):
        self.assertFalse(self._comparison(calibrated=False).decisive)

    def test_a_calibrated_surface_still_decides_true(self):
        """The other mutation direction: calibration must not refuse EVERYTHING."""
        self.assertIs(self._comparison(calibrated=True).decisive, True)

    def test_a_calibrated_miss_is_false_not_none(self):
        comparison = bench.Comparison(
            surface="tg128", anchor_samples=self.SETTLED_A,
            candidate_samples=[100.1, 100.3, 100.0, 100.2, 100.1], effect=0.001,
            estimator="median_over_median", pairs=5, noise_floor_pct=1.0,
            residency={}, calibrated=True)
        self.assertIs(comparison.decisive, False,
                      "'did not clear' and 'undecidable' are different facts")

    def test_to_dict_publishes_the_refusal_and_its_reason_flag(self):
        row = self._comparison(calibrated=False).to_dict()
        self.assertIsNone(row["decisive"])
        self.assertIs(row["calibrated"], False)

    def test_the_default_is_calibrated_for_the_two_measured_surfaces(self):
        row = self._comparison(calibrated=True).to_dict()
        self.assertIs(row["calibrated"], True)


class TheSurfaceTableDrivesTheBatchWidth(unittest.TestCase):
    """dec-b* rows must reach llama-bench as `-b N -ub N` -- the args VERIFIED against
    tools/llama-bench/llama-bench.cpp in the champion tree (test_prompt submits
    min(remaining, n_batch) sequential tokens per llama_decode; test_gen is hardwired
    to one token, so no tg surface can ever express ne11 > 1)."""

    def _argv_for(self, *, ubatch):
        captured = {}

        class FakeSampler:
            proof = {"resident": True, "peak_vram_bytes": 1 << 31,
                     "peak_kfd_processes": 1}
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                return False

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            return mock.Mock(returncode=0, stdout=json.dumps(
                [{"n_prompt": 512, "n_gen": 0, "avg_ts": 100.0}]), stderr="")

        with mock.patch.object(bench.subprocess, "run", side_effect=fake_run), \
             mock.patch.object(bench.residency, "Sampler", FakeSampler), \
             mock.patch.object(bench.residency, "loader_env", lambda _b: {}):
            bench.run_once(Path("/b"), Path("/m.gguf"), pp=512, tg=0, ubatch=ubatch)
        return captured["argv"]

    def test_a_dec_surface_caps_both_batch_and_ubatch(self):
        argv = self._argv_for(ubatch=4)
        self.assertIn("-b", argv)
        self.assertIn("-ub", argv)
        self.assertEqual(argv[argv.index("-b") + 1], "4")
        self.assertEqual(argv[argv.index("-ub") + 1], "4")

    def test_the_classic_surfaces_pass_no_batch_flags(self):
        argv = self._argv_for(ubatch=None)
        self.assertNotIn("-b", argv)
        self.assertNotIn("-ub", argv)

    def test_compare_carries_the_surface_name_and_width_through(self):
        seen = []

        def fake_run_once(binary, model, *, pp, tg, ubatch=None, reps=9,
                          timeout_s=3600):
            seen.append(ubatch)
            return 100.0, {"resident": True, "peak_vram_bytes": 1 << 31,
                           "peak_kfd_processes": 1}

        with mock.patch.object(bench, "run_once", side_effect=fake_run_once):
            comparison = bench.compare(
                bench.Arm("a", Path("/a")), bench.Arm("c", Path("/c")),
                Path("/m.gguf"), pp=512, tg=0, pairs=2, noise_floor_pct=None,
                warmup_pairs=1, surface="dec-b4", ubatch=4, calibrated=False)
        self.assertEqual(comparison.surface, "dec-b4",
                         "the record must name the surface, not masquerade as pp512")
        self.assertIsNone(comparison.decisive)
        self.assertEqual(set(seen), {4}, "every invocation, warm-up included, "
                                         "must run at the surface's width")

    def test_every_surface_row_is_wellformed(self):
        for name, (pp, tg, ubatch) in bench.SURFACES.items():
            with self.subTest(surface=name):
                self.assertTrue((pp > 0) != (tg > 0),
                                "exactly one of pp/tg drives a surface")
                self.assertTrue(ubatch is None or 2 <= ubatch <= 8,
                                "dec widths live in the verify regime ne11 2..8")

    def test_the_seed_facing_names_exist(self):
        """Seeds 05/07/10's scope lines name these surfaces; renaming them silently
        would orphan the planner's own instructions."""
        for name in ("dec-b2", "dec-b4", "dec-b8"):
            self.assertIn(name, bench.SURFACES)


class TheFloorComesFromCalibrationOnly(unittest.TestCase):
    """`floor_rows` may answer from the measured built-in table or a store-written
    A/A record -- NEVER from a default or a neighbouring surface."""

    def test_builtin_surfaces_answer_from_the_measured_table(self):
        self.assertIs(bench.floor_rows("tg128"), bench.MEASURED_FLOOR_PCT["tg128"])
        self.assertIs(bench.floor_rows("pp512"), bench.MEASURED_FLOOR_PCT["pp512"])

    def test_an_unknown_surface_is_uncalibrated_not_defaulted(self):
        self.assertIsNone(bench.floor_rows("dec-b4"))

    def test_a_store_record_calibrates_and_round_trips_int_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            (store / "calibration").mkdir()
            (store / "calibration" / "dec-b4.json").write_text(json.dumps(
                {"floor_pct": {"1": 3.0, "5": 2.0, "9": 1.5, "20": 1.0}}))
            rows = bench.floor_rows("dec-b4", store=store)
        self.assertEqual(rows, {1: 3.0, 5: 2.0, 9: 1.5, 20: 1.0})

    def test_a_store_without_the_record_is_still_uncalibrated(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(bench.floor_rows("dec-b4", store=Path(tmp)))


class TheBootstrapFloorReproducesTheD8Row(unittest.TestCase):
    """The archived 2026-08-29 SETTLED A/A samples must reproduce the ENFORCED tg128
    floor: the k=5 bootstrap row IS where 2.422 came from. A method change that moves
    it is a recalibration and must say so."""

    STORE = Path("/mnt/raid0/llm/autokernel/loop-memory/aa-campaign/aa-campaign.json")

    def _samples(self):
        if not self.STORE.is_file():          # pragma: no cover - store-less host
            self.skipTest("live store not present on this host")
        settled = json.loads(self.STORE.read_text())["conditions"][0]
        return settled["anchor_samples"], settled["candidate_samples"]

    def test_k5_reproduces_the_enforced_floor_byte_for_byte(self):
        anchor, candidate = self._samples()
        rows = bench.bootstrap_floor(anchor, candidate)
        self.assertEqual(rows[5], bench.MEASURED_FLOOR_PCT["tg128"][5])

    def test_rows_cover_the_parametric_seed_and_are_monotone(self):
        anchor, candidate = self._samples()
        rows = bench.bootstrap_floor(anchor, candidate)
        self.assertIn(1, rows, "run.noise_floor_pct seeds sigma/sqrt(n) from k=1")
        ks = sorted(rows)
        self.assertTrue(all(rows[a] >= rows[b] for a, b in zip(ks, ks[1:])),
                        "more pairs may only lower a bootstrap floor")

    def test_the_p95_is_a_tail_statistic_not_a_centre(self):
        rows = bench.bootstrap_floor([100.0] * 20, [100.0] * 20)
        self.assertEqual(set(rows.values()), {0.0},
                         "identical arms have a zero floor at every k")
