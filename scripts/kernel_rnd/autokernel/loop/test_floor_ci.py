"""A FLOOR MUST CARRY ITS n AND ITS CI (R23-61a).

WHY THESE TESTS EXIST. `floor_pct` is p95 |deviation from median| over N server launches --
an EXTREME order statistic. The standing champion floor was 4.581% at n=10; a bootstrap of
n=10 draws from 24 same-configuration launches spanned 4.200%-7.821% (5th-95th pct), so the
point estimate carried no usable precision. `n` is already enforced (a floor that cannot
state it must not gate); this pins the interval that travels next to it:

  * `calibrate_floor` states a `floor_ci` computed over the SAME statistic as `floor_pct`;
  * it is deterministic (fixed seed), so a replay reproduces it byte-for-byte;
  * `write_floor` re-derives it from the row's own runs and REFUSES a stated interval that
    does not re-derive, or one stated with no runs to check it against;
  * the reader is backward compatible: a floor written before R23-61a loads with `ci=None`
    and gates exactly as before; a matched (v2) floor exposes its sealed `interval`;
  * the CI is descriptive -- `gate_floor` returns the same bar with or without it.

No server is launched: synthetic rows and temp stores throughout.
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from autokernel.loop import serving


RECIPE = serving.Recipe(name="t", model="/m/target.gguf", np=4, ctx=16384)

#: 24 launch values with a real tail, shaped like the champion's A/A (~161 tok/s, ~4-7%).
RUNS = [163.1, 154.2, 154.3, 168.5, 160.6, 168.1, 161.5, 154.2, 163.2, 159.2, 171.9, 166.0,
        158.4, 162.7, 150.3, 165.5, 167.0, 160.1, 157.9, 164.4, 169.8, 155.6, 161.9, 162.2]


def _calibrated(runs=RUNS) -> dict:
    with mock.patch.object(serving, "_measure_once", side_effect=list(runs)):
        return serving.calibrate_floor(RECIPE, Path("/b"), samples=len(runs))


class TheCalibratedRowStatesItsInterval(unittest.TestCase):
    def test_the_interval_is_over_the_same_statistic_and_brackets_it(self):
        row = _calibrated()
        ci = row["floor_ci"]
        self.assertEqual(ci["n"], 24)
        self.assertEqual(row["n"], 24)
        self.assertEqual(ci["level"], 0.95)
        self.assertEqual(ci["use"], "descriptive_only_not_gate_endpoint")
        self.assertLessEqual(ci["low_pct"], ci["median_pct"])
        self.assertLessEqual(ci["median_pct"], ci["high_pct"])
        # The point estimate must sit inside its own interval.
        self.assertLessEqual(ci["low_pct"], row["floor_pct"])
        self.assertLessEqual(row["floor_pct"], ci["high_pct"])
        self.assertLess(ci["low_pct"], ci["high_pct"], "a real tail yields a real width")

    def test_it_is_deterministic(self):
        self.assertEqual(serving.floor_ci(RUNS), serving.floor_ci(list(RUNS)))
        self.assertNotEqual(serving.floor_ci(RUNS)["seed"],
                            serving.floor_ci(RUNS, seed=serving.FLOOR_CI_SEED + 1)["seed"])

    def test_the_floor_statistic_is_unchanged_by_the_shared_helper(self):
        """`_spread` was refactored onto the helper the CI uses; its p95 must not move."""
        import statistics
        med = statistics.median(RUNS)
        legacy = sorted(abs(r / med - 1.0) * 100.0 for r in RUNS)
        expected = round(legacy[min(23, int(round(0.95 * 23)))], 3)
        self.assertEqual(_calibrated()["floor_pct"], expected)

    def test_too_few_launches_cannot_state_an_interval(self):
        with self.assertRaises(serving.RecipeError):
            serving.floor_ci([100.0])
        self.assertIsNone(_calibrated([100.0])["floor_ci"])


class TheWriterChecksTheInterval(unittest.TestCase):
    def test_round_trip_carries_n_and_ci_next_to_the_percentage(self):
        row = _calibrated()
        with tempfile.TemporaryDirectory() as tmp:
            body = json.loads(serving.write_floor(
                tmp, RECIPE, row, unit=serving.CALIBRATION_UNIT).read_text())
            reading = serving.load_floor(tmp, RECIPE)
        self.assertEqual(body["n"], 24)
        self.assertEqual(body["floor_ci"], serving.floor_ci(RUNS))
        self.assertEqual(reading.ci, body["floor_ci"])
        self.assertEqual(reading.n, 24)
        self.assertEqual(reading.gate_floor(effect_unit=serving.COMPARE_EFFECT_UNIT),
                         row["floor_pct"])

    def test_a_row_without_a_ci_gets_one_derived_from_its_runs(self):
        row = {k: v for k, v in _calibrated().items() if k != "floor_ci"}
        with tempfile.TemporaryDirectory() as tmp:
            body = json.loads(serving.write_floor(
                tmp, RECIPE, row, unit=serving.CALIBRATION_UNIT).read_text())
        self.assertEqual(body["floor_ci"], serving.floor_ci(RUNS))

    def test_a_stated_interval_that_does_not_rederive_refuses_and_writes_nothing(self):
        row = _calibrated()
        row["floor_ci"] = dict(row["floor_ci"], high_pct=1.0)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(serving.ServingFloorMismatch):
                serving.write_floor(tmp, RECIPE, row, unit=serving.CALIBRATION_UNIT)
            self.assertFalse(serving.floor_path(tmp, RECIPE).exists())

    def test_an_interval_with_no_runs_to_check_it_refuses(self):
        row = {k: v for k, v in _calibrated().items() if k != "runs"}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(serving.ServingFloorMismatch):
                serving.write_floor(tmp, RECIPE, row, unit=serving.CALIBRATION_UNIT)


class OldRecordsStillRead(unittest.TestCase):
    def test_a_floor_written_before_r23_61a_loads_with_no_ci_and_gates_as_before(self):
        row = {k: v for k, v in _calibrated().items() if k != "floor_ci"}
        with tempfile.TemporaryDirectory() as tmp:
            path = serving.floor_path(tmp, RECIPE)
            path.write_text(json.dumps(row), encoding="utf-8")
            reading = serving.load_floor(tmp, RECIPE)
        self.assertIsNone(reading.ci)
        self.assertEqual(reading.gate_floor(effect_unit=serving.COMPARE_EFFECT_UNIT),
                         row["floor_pct"])

    def test_a_matched_floor_exposes_its_sealed_interval(self):
        reading = serving.FloorReading(2.0, "verified", Path("/x"),
                                       {"interval": {"level": 0.95, "low_pct": 1.0,
                                                     "high_pct": 3.0}})
        self.assertEqual(reading.ci["high_pct"], 3.0)


if __name__ == "__main__":
    unittest.main()
