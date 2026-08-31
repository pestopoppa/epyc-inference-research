"""The uncalibrated-floor discipline at the RUN level, and the calibration mode.

The historical defect this pins shut: pp512 once carried a stale floor row and the
loop published fake `decisive=True` off it. Three layers now refuse that, and each
is tested with the others mutated away in mind: `Comparison.decisive` returns None
(test_bench), `iterate` cannot commit a falsy decisive (test_loop), and here the
commit path's OWN `refuse_uncalibrated_keep` -- which must hold even against a
doctored comparison object whose decisive lies True.
"""
import importlib.util
import json
from pathlib import Path
import statistics as st
import subprocess
import tempfile
import types
import unittest
from unittest import mock

from autokernel.loop import bench, loop
from autokernel.loop import run as run_mod


def _load_aa_campaign():
    path = Path(run_mod.__file__).resolve().parents[3] / "benchmark" / \
        "autokernel_aa_campaign.py"
    spec = importlib.util.spec_from_file_location("aa_campaign_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TheFloorIsCalibrationOrNothing(unittest.TestCase):
    """`run.noise_floor_pct` may answer from the measured table or a store-written
    A/A record. For anything else the answer is None -- never a borrowed number."""

    def test_the_builtin_surfaces_are_unchanged_by_the_extension(self):
        self.assertAlmostEqual(run_mod.noise_floor_pct("pp512", 5), 0.9727, places=3)
        self.assertAlmostEqual(run_mod.noise_floor_pct("tg128", 5), 2.422, places=3)

    def test_an_unknown_surface_has_no_floor_not_a_default_one(self):
        self.assertIsNone(run_mod.noise_floor_pct("dec-b4", 5))
        self.assertIsNone(run_mod.noise_floor_pct("dec-b8", 20))

    def test_a_store_calibration_gives_the_surface_a_real_floor(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            (store / "calibration").mkdir()
            (store / "calibration" / "dec-b4.json").write_text(json.dumps(
                {"floor_pct": {"1": 3.0, "3": 2.4, "5": 2.0, "9": 1.6, "20": 1.2}}))
            floor = run_mod.noise_floor_pct("dec-b4", 5, store=store)
            # max(parametric 3.0/sqrt(5)=1.342, measured row 2.0) -- the same
            # two-bound arithmetic the built-in surfaces get.
            self.assertAlmostEqual(floor, 2.0, places=6)
            self.assertIsNone(run_mod.noise_floor_pct("dec-b8", 5, store=store),
                              "one surface's calibration must not leak to another")


class TheCommitPathRefusesForItself(unittest.TestCase):
    """`refuse_uncalibrated_keep` re-derives the refusal from the run-level
    calibration fact, so a doctored comparison cannot advance the champion."""

    def test_a_lying_decisive_true_is_still_refused_when_uncalibrated(self):
        doctored = types.SimpleNamespace(decisive=True)     # the fake-floor defect
        with self.assertRaises(loop.RunAborted) as caught:
            run_mod.refuse_uncalibrated_keep("dec-b4", False, doctored)
        self.assertIn("UNCALIBRATED", str(caught.exception))
        self.assertIn("dec-b4", str(caught.exception))

    def test_a_calibrated_decisive_keep_passes(self):
        run_mod.refuse_uncalibrated_keep(
            "tg128", True, types.SimpleNamespace(decisive=True))

    def test_a_calibrated_indecisive_comparison_is_refused_too(self):
        for decisive in (False, None):
            with self.subTest(decisive=decisive):
                with self.assertRaises(loop.RunAborted):
                    run_mod.refuse_uncalibrated_keep(
                        "tg128", True, types.SimpleNamespace(decisive=decisive))


class TheCalibrateModeDispatchesToTheD8Instrument(unittest.TestCase):
    """`--calibrate-surface` must run the ONE home of the method with the store
    wired through -- not a second copy of the campaign."""

    def test_the_dispatch_names_a_real_script_and_wires_the_store(self):
        args = types.SimpleNamespace(
            surface="dec-b4", calibrate_surface=20,
            anchor_build=Path("/tmp/anchor"), worktree=Path("/tmp/tree"),
            model=Path("/tmp/m.gguf"), store=Path("/tmp/store"))
        captured = {}

        def fake_run(argv):
            captured["argv"] = argv
            return types.SimpleNamespace(returncode=7)

        self.assertEqual(run_mod.calibrate(args, run=fake_run), 7,
                         "the campaign's exit code is the mode's exit code")
        argv = captured["argv"]
        script = Path(argv[1])
        self.assertTrue(script.is_file(), f"{script} must exist")
        self.assertEqual(script.name, "autokernel_aa_campaign.py")
        self.assertEqual(argv[argv.index("--surface") + 1], "dec-b4")
        self.assertEqual(argv[argv.index("--pairs") + 1], "20")
        self.assertEqual(argv[argv.index("--write-calibration") + 1], "/tmp/store")


class TheCampaignWritesWhatTheLoopReads(unittest.TestCase):
    """End to end with injected doubles: `--write-calibration` produces exactly the
    record `bench.floor_rows` parses, after which the surface stops being refused."""

    PAIRS = 6

    def _fake_compare(self, anchor, candidate, model, *, pp, tg, pairs,
                      noise_floor_pct=None, warmup_pairs=1, surface=None,
                      ubatch=None, calibrated=True, reps=9):
        a = [100.0 + 0.1 * index for index in range(pairs)]
        c = [100.05 + 0.1 * index for index in range(pairs)]
        return bench.Comparison(
            surface=surface or "dec-b4", anchor_samples=a, candidate_samples=c,
            effect=(st.median(c) / st.median(a)) - 1.0,
            estimator="median_over_median", pairs=pairs,
            noise_floor_pct=noise_floor_pct, residency={"invocations": 2 * pairs,
                                                        "resident": 2 * pairs},
            calibrated=calibrated)

    def test_the_round_trip_calibrates_the_surface(self):
        aa = _load_aa_campaign()

        class FakeClaim:
            def __enter__(self):
                return {"device_id": "mi210_0"}

            def __exit__(self, *exc):
                return False

        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "store"
            self.assertIsNone(run_mod.noise_floor_pct("dec-b4", 5, store=store),
                              "before calibration the surface must refuse")
            with mock.patch.object(aa.bench, "compare", self._fake_compare), \
                 mock.patch.object(aa.claim, "hold", FakeClaim), \
                 mock.patch.object(aa.gates, "compiles",
                                   lambda *a, **k: types.SimpleNamespace(
                                       gate="compile", passed=True)), \
                 mock.patch.object(aa.time, "sleep", lambda _s: None), \
                 mock.patch.object(subprocess, "run",
                                   lambda *a, **k: types.SimpleNamespace(
                                       stdout="feedcafe\n", returncode=0)):
                rc = aa.main(["--surface", "dec-b4", "--pairs", str(self.PAIRS),
                              "--worktree", tmp, "--model", "/tmp/m.gguf",
                              "--out", str(store / "calibration" / "aa-dec-b4"),
                              "--write-calibration", str(store)])
            self.assertEqual(rc, 0)
            record = json.loads(
                (store / "calibration" / "dec-b4.json").read_text())
            self.assertEqual(record["surface"], "dec-b4")
            self.assertEqual(record["bench_args"], {"pp": 512, "tg": 0, "ubatch": 4})
            self.assertEqual(len(record["conditions"]), 3,
                             "SETTLED, PREHEATED and POST_BUILD all land as provenance")
            self.assertEqual(record["anchor_commit"], "feedcafe")
            floor = run_mod.noise_floor_pct("dec-b4", 5, store=store)
            self.assertIsInstance(floor, float,
                                  "after calibration the surface decides again")
            self.assertGreater(floor, 0.0)
            # And the campaign artifact itself still lands beside it.
            self.assertTrue((store / "calibration" / "aa-dec-b4" /
                             "aa-campaign.json").is_file())


if __name__ == "__main__":
    unittest.main()
