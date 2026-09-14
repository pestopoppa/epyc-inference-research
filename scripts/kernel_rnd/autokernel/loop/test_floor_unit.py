"""A FLOOR WITHOUT ITS UNIT IS A 1200-FOLD ERROR (R23-55 / INF-73 U2).

WHY THESE TESTS EXIST. A floor is a DISPERSION, and a dispersion only exists relative to
what was RESAMPLED between two readings. INF-70's RETEST-1 measured the same host on the
same day in two units:

  * within one server session (an `arm`-scoped knob toggled between requests): sd 0.501%
  * between two separate process launches:                                     sd 2.793%

~13x coarser between processes. The floor a gate uses is therefore meaningless without the
unit it was measured in, and getting it wrong is not a rounding error: INF-70's 0.171%
ARM-unit floor sized CHAMP-2 THP at 4 sessions per side where the correct session-unit
answer is 4,780 -- a 1200-fold underestimate that would have been spent as real host time.

What must hold:
  * `write_floor` is GIVEN the unit; there is no default, and a call that does not state
    one refuses and writes nothing;
  * a row that already states a unit must AGREE with the caller, and a row with no sample
    count refuses too (R23-61: a floor that cannot state its `n` must not gate);
  * a legacy floor already on disk loads, is marked `unit=None` / `legacy=True`, is NEVER
    rewritten, and REFUSES to be used as a gate bar -- naming the file and the fix;
  * an effect is judged only against a floor of its OWN unit; a mismatch refuses with BOTH
    units named, and is never warned about or rescaled;
  * the whole record round-trips: unit and n survive write -> read.

No server is launched: synthetic rows and temp stores throughout.
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from autokernel.loop import bench, instruments, serving


RECIPE = serving.Recipe(name="t", model="/m/target.gguf", np=4, ctx=16384)


def _row(recipe: serving.Recipe = RECIPE, floor_pct: float = 3.536, **extra) -> dict:
    """What `calibrate_floor`'s legacy path returns, without launching anything."""
    return {"schema": "epyc.autokernel.serving_floor.v1", "recipe": recipe.name,
            "recipe_hash": recipe.recipe_hash, "recipe_env": dict(recipe.env or {}),
            "recipe_describe": recipe.describe(), "metric": recipe.metric,
            "np": recipe.np, "samples": 24, "unit": serving.CALIBRATION_UNIT, "n": 24,
            "median_tok_s": 161.07, "floor_pct": floor_pct, "runs": [161.0] * 24,
            "cv_pct": 3.136, **extra}


class WritingAFloorRequiresItsUnit(unittest.TestCase):
    def test_a_write_that_does_not_state_the_unit_refuses_and_writes_nothing(self):
        """The defect R23-55 names is the MISSING field, so the writer cannot have a
        default: a default would answer the question nobody asked."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(serving.FloorUnitMismatch) as caught:
                serving.write_floor(tmp, RECIPE, dict(_row(), unit=None))
            self.assertFalse(serving.floor_path(tmp, RECIPE).exists())
        message = str(caught.exception)
        for expected in ("unit", "arm", "session", "process"):
            self.assertIn(expected, message)

    def test_an_unknown_unit_is_refused_rather_than_stored(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(serving.FloorUnitMismatch):
                serving.write_floor(tmp, RECIPE, _row(), unit="launch")
            self.assertFalse(serving.floor_path(tmp, RECIPE).exists())

    def test_a_row_that_states_another_unit_refuses_with_both_named(self):
        """The row is the harness's own statement about what it resampled. A caller that
        files it under a different unit is relabelling a measurement."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(serving.FloorUnitMismatch) as caught:
                serving.write_floor(tmp, RECIPE, dict(_row(), unit=serving.UNIT_ARM),
                                    unit=serving.UNIT_PROCESS)
            self.assertFalse(serving.floor_path(tmp, RECIPE).exists())
        self.assertIn("arm", str(caught.exception))
        self.assertIn("process", str(caught.exception))

    def test_a_row_with_no_sample_count_refuses(self):
        """R23-61: a floor is an extreme order statistic, so a record that cannot state
        its `n` carries no usable precision and must not gate."""
        row = {k: v for k, v in _row().items() if k not in ("n", "samples")}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(serving.FloorUnitMismatch) as caught:
                serving.write_floor(tmp, RECIPE, row, unit=serving.CALIBRATION_UNIT)
            self.assertFalse(serving.floor_path(tmp, RECIPE).exists())
        self.assertIn("n", str(caught.exception))

    def test_the_legacy_n_field_is_promoted_under_its_own_name(self):
        row = {k: v for k, v in _row().items() if k != "n"}
        with tempfile.TemporaryDirectory() as tmp:
            body = json.loads(serving.write_floor(
                tmp, RECIPE, row, unit=serving.CALIBRATION_UNIT).read_text())
        self.assertEqual(body["n"], 24)
        self.assertEqual(body["unit"], serving.CALIBRATION_UNIT)

    def test_a_matched_floor_must_carry_unit_and_n_inside_its_sealed_content(self):
        """`content_sha256` seals a matched floor. Stamping the two fields a gate depends
        on AFTER the seal would put them where an edit leaves no trace."""
        row = {"schema": "epyc.autokernel.serving_floor.v2", "floor_pct": 1.0,
               "calibration_pairs": 24, "recipe_hash": RECIPE.recipe_hash}
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(serving, "_validate_matched_floor", lambda *a, **k: None):
                with self.assertRaises(serving.FloorUnitMismatch) as caught:
                    serving.write_floor(tmp, RECIPE, row, unit=serving.UNIT_PROCESS,
                                        instrument=serving.MATCHED_INSTRUMENT, pairs=5)
        self.assertIn("sealed", str(caught.exception))

    def test_the_calibrated_row_states_the_unit_the_harness_actually_resampled(self):
        """`calibrate_floor` relaunches the server per sample, so its dispersion is
        between-PROCESS. The unit is DERIVED from the harness, never passed in."""
        with mock.patch.object(serving, "_measure_once", side_effect=[100.0] * 5):
            row = serving.calibrate_floor(RECIPE, Path("/b"), samples=5)
        self.assertEqual(row["unit"], serving.UNIT_PROCESS)
        self.assertEqual(row["n"], 5)


class AFullRecordRoundTrips(unittest.TestCase):
    def test_calibrate_write_load_preserves_the_unit_and_the_n(self):
        with mock.patch.object(serving, "_measure_once", side_effect=[100.0] * 24):
            row = serving.calibrate_floor(RECIPE, Path("/b"), samples=24)
        with tempfile.TemporaryDirectory() as tmp:
            path = serving.write_floor(tmp, RECIPE, row, unit=serving.CALIBRATION_UNIT,
                                       conditions={"host_state": "loop DOWN"})
            body = json.loads(path.read_text())
            reading = serving.load_floor(tmp, RECIPE)
        self.assertEqual(body["unit"], serving.UNIT_PROCESS)
        self.assertEqual(body["n"], 24)
        self.assertEqual(reading.unit, serving.UNIT_PROCESS)
        self.assertEqual(reading.n, 24)
        self.assertFalse(reading.legacy)
        self.assertTrue(reading.verified)
        # ...and it is usable as the bar for an effect of the same unit.
        self.assertEqual(reading.gate_floor(effect_unit=serving.UNIT_PROCESS),
                         reading.floor_pct)


class ALegacyFloorLoadsAndRefusesToGate(unittest.TestCase):
    """A floor file written before R23-55 says nothing about its unit. It is READ (nothing
    on disk is rewritten) and it cannot gate: the two candidate answers differ by ~13x."""

    def _legacy(self, tmp) -> Path:
        row = {k: v for k, v in _row().items() if k != "unit"}
        path = serving.floor_path(tmp, RECIPE)
        path.write_text(json.dumps(row), encoding="utf-8")
        return path

    def test_it_loads_marked_unit_none_and_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._legacy(Path(tmp))
            reading = serving.load_floor(tmp, RECIPE)
        self.assertIsNone(reading.unit)
        self.assertTrue(reading.legacy)
        self.assertEqual(reading.floor_pct, 3.536)      # the value is not destroyed
        self.assertEqual(reading.provenance, "verified")  # the RECIPE identity is fine

    def test_the_gate_refuses_it_naming_the_file_and_the_fix(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._legacy(Path(tmp))
            before = path.read_bytes()
            reading = serving.load_floor(tmp, RECIPE)
            with self.assertRaises(serving.FloorUnitMismatch) as caught:
                reading.gate_floor(effect_unit=serving.COMPARE_EFFECT_UNIT)
            self.assertEqual(path.read_bytes(), before, "no floor file may be rewritten")
        message = str(caught.exception)
        self.assertIn(str(path), message)
        self.assertIn("recalibrate", message.lower())
        self.assertIn("0.501", message)
        self.assertIn("2.793", message)

    def test_an_instrument_refuses_it_before_spending_any_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._legacy(Path(tmp))
            with self.assertRaises(instruments.InstrumentRefusal) as caught:
                instruments.read_floor(tmp, RECIPE,
                                       effect_unit=serving.COMPARE_EFFECT_UNIT,
                                       echo=lambda *_: None)
        self.assertIn("unit", str(caught.exception))

    def test_an_absent_floor_is_still_just_absent(self):
        """Uncalibrated already fails closed everywhere downstream; the unit rule must not
        turn "no floor" into a different kind of error."""
        with tempfile.TemporaryDirectory() as tmp:
            reading = serving.load_floor(tmp, RECIPE)
        self.assertEqual(reading.provenance, "absent")
        self.assertFalse(reading.legacy)
        self.assertIsNone(reading.gate_floor(effect_unit=serving.COMPARE_EFFECT_UNIT))


class AnEffectIsJudgedOnlyAgainstItsOwnUnit(unittest.TestCase):
    def test_matching_units_pass_and_decide(self):
        with mock.patch.object(serving, "_measure_once", side_effect=[100.0, 110.0]):
            out = serving.compare(RECIPE, Path("/a"), Path("/c"), pairs=1, floor_pct=1.0,
                                  floor_unit=serving.UNIT_PROCESS)
        self.assertIs(out["decisive"], True)
        self.assertEqual(out["effect_unit"], serving.UNIT_PROCESS)
        self.assertEqual(out["floor_unit"], serving.UNIT_PROCESS)

    def test_a_mismatched_unit_refuses_with_both_units_named(self):
        """The arm-unit floor is the exact instance that mis-sized CHAMP-2 by 1200x."""
        def forbidden(*args, **kwargs):
            self.fail("a mismatched unit reached a measurement")
        with mock.patch.object(serving, "_measure_once", forbidden):
            with self.assertRaises(serving.FloorUnitMismatch) as caught:
                serving.compare(RECIPE, Path("/a"), Path("/c"), pairs=1, floor_pct=0.171,
                                floor_unit=serving.UNIT_ARM)
        message = str(caught.exception)
        self.assertIn(serving.UNIT_ARM, message)
        self.assertIn(serving.UNIT_PROCESS, message)
        self.assertNotIn("WARNING", message)

    def test_a_bar_with_no_unit_refuses_rather_than_being_assumed(self):
        def forbidden(*args, **kwargs):
            self.fail("an unlabelled bar reached a measurement")
        with mock.patch.object(serving, "_measure_once", forbidden):
            with self.assertRaises(serving.FloorUnitMismatch):
                serving.compare(RECIPE, Path("/a"), Path("/c"), pairs=1, floor_pct=1.0)

    def test_a_unit_without_a_floor_refuses(self):
        """Nothing is being gated, so there is no comparison for a unit to describe --
        and the pairing is the only thing that makes the field meaningful."""
        with mock.patch.object(serving, "_measure_once", side_effect=[100.0, 100.0]):
            with self.assertRaises(serving.FloorUnitMismatch):
                serving.compare(RECIPE, Path("/a"), Path("/c"), pairs=1, floor_pct=None,
                                floor_unit=serving.UNIT_PROCESS)

    def test_the_record_disagreeing_with_the_caller_refuses(self):
        with self.assertRaises(serving.FloorUnitMismatch):
            serving.compare(RECIPE, Path("/a"), Path("/c"), pairs=1, floor_pct=1.0,
                            floor_unit=serving.UNIT_PROCESS,
                            floor_record={"unit": serving.UNIT_ARM, "floor_pct": 1.0})

    def test_check_unit_is_the_one_rule_and_returns_the_agreed_unit(self):
        self.assertEqual(
            serving.check_unit(serving.UNIT_SESSION, serving.UNIT_SESSION), "session")
        for floor_unit, effect_unit in ((serving.UNIT_ARM, serving.UNIT_SESSION),
                                        (None, serving.UNIT_PROCESS),
                                        (serving.UNIT_PROCESS, None)):
            with self.subTest(floor=floor_unit, effect=effect_unit):
                with self.assertRaises(serving.FloorUnitMismatch):
                    serving.check_unit(floor_unit, effect_unit)


class TheBenchFloorCarriesItsUnitToo(unittest.TestCase):
    """The screen floor is the same rule on the other instrument: `bench.compare`
    alternates across llama-bench INVOCATIONS, so its floors are process-unit."""

    MODEL = "Qwen3.8-27B-Q8_0.gguf"

    def _store(self, tmp, body) -> Path:
        store = Path(tmp)
        (store / "calibration").mkdir()
        (store / "calibration" / f"dec-b4.{Path(self.MODEL).stem}.json").write_text(
            json.dumps({"model": self.MODEL, "floor_pct": {"5": 2.0}, **body}))
        return store

    def test_the_two_modules_share_one_vocabulary(self):
        """`bench` repeats the literal because importing `serving` there is a cycle. This
        is what stops the two copies drifting."""
        self.assertIn(bench.FLOOR_UNIT, serving.FLOOR_UNITS)
        self.assertEqual(bench.FLOOR_UNIT, serving.UNIT_PROCESS)

    def test_an_explicit_unit_is_honoured(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp, {"unit": bench.FLOOR_UNIT, "n": 24})
            self.assertEqual(bench.floor_rows("dec-b4", self.MODEL, store=store), {5: 2.0})

    def test_a_record_of_another_unit_refuses_with_both_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp, {"unit": serving.UNIT_ARM, "n": 24})
            with self.assertRaises(serving.FloorUnitMismatch) as caught:
                bench.floor_rows("dec-b4", self.MODEL, store=store)
        self.assertIn("arm", str(caught.exception))
        self.assertIn("process", str(caught.exception))

    def test_a_pre_r23_55_record_takes_its_unit_from_its_schema_not_from_silence(self):
        """The schema has exactly ONE writer and that writer alternates processes, so the
        unit is derived. It is not a guess, and it keeps the live store readable without
        rewriting a single floor file."""
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp, {"schema": bench.CALIBRATION_SCHEMA,
                                      "pairs_per_condition": 20})
            self.assertEqual(bench.floor_rows("dec-b4", self.MODEL, store=store), {5: 2.0})

    def test_a_record_with_neither_a_unit_nor_a_known_schema_cannot_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp, {"n": 20})
            with self.assertRaises(serving.FloorUnitMismatch):
                bench.floor_rows("dec-b4", self.MODEL, store=store)

    def test_a_record_without_its_n_cannot_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp, {"unit": bench.FLOOR_UNIT})
            with self.assertRaises(serving.FloorUnitMismatch) as caught:
                bench.floor_rows("dec-b4", self.MODEL, store=store)
        self.assertIn("`n`", str(caught.exception))

    def test_the_comparison_row_states_both_units(self):
        row = bench.Comparison(surface="tg128", anchor_samples=[1.0],
                               candidate_samples=[1.0], effect=0.0,
                               estimator="median_over_median", pairs=1,
                               noise_floor_pct=1.0, residency={}).to_dict()
        self.assertEqual(row["effect_unit"], bench.FLOOR_UNIT)
        self.assertEqual(row["floor_unit"], bench.FLOOR_UNIT)
        uncalibrated = bench.Comparison(surface="tg128", anchor_samples=[1.0],
                                        candidate_samples=[1.0], effect=0.0,
                                        estimator="median_over_median", pairs=1,
                                        noise_floor_pct=None, residency={}).to_dict()
        self.assertIsNone(uncalibrated["floor_unit"], "no bar, no bar's unit")

    def test_the_bench_floor_writer_stamps_the_unit_and_the_n(self):
        """The record `floor_rows` reads is written by exactly one place, and R23-55 is
        only closed if that writer states both fields."""
        source = (instruments.REPO_ROOT
                  / "scripts/benchmark/autokernel_aa_campaign.py").read_text()
        self.assertIn('"unit": bench.FLOOR_UNIT', source)
        self.assertIn('"n": args.pairs', source)


class TheGatesPassTheUnitTruthfully(unittest.TestCase):
    """Structural: every serving gate must hand `compare` the unit of the floor it read,
    not a literal. A hard-coded unit would make the check unfalsifiable."""

    def _source(self, name: str) -> str:
        return (Path(__file__).with_name(name)).read_text(encoding="utf-8")

    def test_the_hand_run_serving_gate_passes_the_loaded_floors_own_unit(self):
        source = self._source("serving_gate.py")
        self.assertIn("effect_unit=serving.COMPARE_EFFECT_UNIT", source)
        self.assertIn("floor_unit=reading.unit", source)

    def test_the_loop_turns_a_reading_into_a_bar_in_exactly_one_place(self):
        source = self._source("run.py")
        self.assertIn("def _gate_floor(reading)", source)
        self.assertIn("floor_unit=serving_floor_unit", source)
        # ...and never bypasses it by reading the number straight off the file.
        self.assertNotIn("floor = serving_floor_pct = floor_reading.floor_pct", source)

    def test_the_loo_battery_admits_its_floor_before_the_first_arm(self):
        source = self._source("source_loo.py")
        self.assertIn("floor.gate_floor(effect_unit=serving.COMPARE_EFFECT_UNIT)", source)
        self.assertIn("floor_unit=floor_gate_unit", source)

    def test_the_hand_supplied_bench_floors_state_their_unit(self):
        for name in ("fold2_gates.py", "seed_bundle.py"):
            with self.subTest(instrument=name):
                source = self._source(name)
                self.assertIn("--floor-unit", source)
                self.assertIn("serving.check_unit(args.floor_unit, bench.FLOOR_UNIT",
                              source)

    def test_the_recalibration_states_the_unit_it_measured(self):
        source = self._source("recal_serving_floor.py")
        self.assertIn("unit=serving.CALIBRATION_UNIT", source)


if __name__ == "__main__":
    unittest.main()
