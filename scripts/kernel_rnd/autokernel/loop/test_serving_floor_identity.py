"""The serving floor is keyed by recipe IDENTITY, not by recipe NAME.

WHY THESE TESTS EXIST. `serving-floor.<name>.json` is a name-keyed cache of a number that
is only meaningful under ONE measured condition, and nothing checked that the file on disk
was produced by the recipe now being gated. So any recipe edit silently reused the old
floor and the gate judged one condition against another's bar. It has already bitten once:
R23-49 pinned `cpu_list` to 184-191, which changed the measured condition and voided the
calibrated floor -- and only a human noticing forced the recalibration. `env` makes it
sharper, because an env arm's entire PURPOSE is to change dispersion, i.e. the floor itself.

What must hold:
  * a floor written for this recipe carries its `recipe_hash`, `recipe_describe` and name
    at TOP LEVEL, so the identity is in the file and not merely in some caller's memory;
  * loading a floor whose hash disagrees REFUSES, naming both hashes -- fail-closed, and
    distinguishable from "no floor at all", which merely blocks the gate (R23-54) and
    would read as a cadence bug;
  * a floor written before stamping existed still loads (hard-failing would block the loop
    on every floor that exists today) but is marked `unverified` wherever it is recorded;
  * a `with_env` arm never resolves to its base arm's floor file, on any construction path;
  * an env VALUE reaches the recipe name, so the filename derived from it is sanitised
    deterministically and can never collide two recipes onto one floor.

No server is launched: synthetic rows, temp stores.
"""
import dataclasses
import json
from pathlib import Path
import tempfile
import unittest

from autokernel.loop import serving


BASE = serving.Recipe(name="t", model="/m/target.gguf", np=4, ctx=16384)
SHIPPED = (Path(__file__).resolve().parents[4]
           / "artifacts/serving-recipes/qwen3.8-27b-q8-gpu-dflash2-np4.json")


def _floor_row(recipe: serving.Recipe, floor_pct: float = 3.536) -> dict:
    """What `calibrate_floor` returns, without launching anything."""
    return {"schema": "epyc.autokernel.serving_floor.v1", "recipe": recipe.name,
            "recipe_hash": recipe.recipe_hash, "recipe_env": dict(recipe.env or {}),
            "recipe_describe": recipe.describe(), "metric": recipe.metric,
            "np": recipe.np, "samples": 10, "median_tok_s": 161.07,
            "floor_pct": floor_pct, "runs": [161.0] * 10, "cv_pct": 3.136}


class FloorFileCarriesTheIdentity(unittest.TestCase):
    def test_the_written_file_carries_hash_describe_and_name_at_top_level(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = serving.write_floor(tmp, BASE, _floor_row(BASE),
                                       conditions={"host_state": "loop DOWN"})
            body = json.loads(path.read_text())
        self.assertEqual(body["recipe_hash"], BASE.recipe_hash)
        self.assertEqual(body["recipe"], BASE.name)
        self.assertEqual(body["recipe_describe"], BASE.describe())
        self.assertEqual(body["floor_pct"], 3.536)
        # free-form provenance is merged, never at the cost of the identity keys
        self.assertEqual(body["conditions"], {"host_state": "loop DOWN"})

    def test_calibrate_floor_output_is_exactly_what_write_floor_accepts(self):
        """The row the calibrator produces must file without a translation step -- a
        hand-copied field is a field that can be copied wrong."""
        import unittest.mock as mock
        with mock.patch.object(serving, "_measure_once",
                               side_effect=[100.0, 101.0, 99.0, 100.5, 99.5]):
            row = serving.calibrate_floor(BASE, Path("/b"), samples=5)
        with tempfile.TemporaryDirectory() as tmp:
            body = json.loads(serving.write_floor(tmp, BASE, row).read_text())
            self.assertEqual(serving.load_floor(tmp, BASE).provenance, "verified")
        self.assertEqual(body["recipe_hash"], BASE.recipe_hash)

    def test_filing_another_recipes_row_is_refused(self):
        other = dataclasses.replace(BASE, np=8)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(serving.ServingFloorMismatch):
                serving.write_floor(tmp, BASE, _floor_row(other))


class LoadingAFloorChecksIt(unittest.TestCase):
    def test_a_matching_hash_proceeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            serving.write_floor(tmp, BASE, _floor_row(BASE))
            reading = serving.load_floor(tmp, BASE)
        self.assertEqual(reading.provenance, "verified")
        self.assertTrue(reading.verified)
        self.assertEqual(reading.floor_pct, 3.536)

    def test_a_differing_hash_refuses_and_names_both_hashes(self):
        """Same NAME, different condition: exactly the R23-49 shape (a `cpu_list` pin)."""
        pinned = dataclasses.replace(BASE, cpu_list="184-191")
        self.assertEqual(pinned.name, BASE.name)          # the name cannot tell them apart
        with tempfile.TemporaryDirectory() as tmp:
            serving.write_floor(tmp, BASE, _floor_row(BASE))
            # the pinned recipe resolves to the SAME file -- and must refuse it
            self.assertEqual(serving.floor_path(tmp, pinned),
                             serving.floor_path(tmp, BASE))
            with self.assertRaises(serving.ServingFloorMismatch) as caught:
                serving.load_floor(tmp, pinned)
        message = str(caught.exception)
        self.assertIn(BASE.recipe_hash, message)          # the floor's
        self.assertIn(pinned.recipe_hash, message)        # the live recipe's
        self.assertIn("recalibrate", message.lower())

    def test_a_legacy_floor_with_no_hash_proceeds_as_unverified(self):
        """Grandfathered. Hard-failing would block the loop at relaunch on every floor
        that exists today; proceeding silently would make it indistinguishable from a
        checked one."""
        legacy = _floor_row(BASE)
        legacy.pop("recipe_hash")
        legacy.pop("recipe_describe")
        with tempfile.TemporaryDirectory() as tmp:
            serving.floor_path(tmp, BASE).write_text(json.dumps(legacy))
            reading = serving.load_floor(tmp, BASE)
        self.assertEqual(reading.provenance, "unverified")
        self.assertFalse(reading.verified)
        self.assertEqual(reading.floor_pct, 3.536)        # it is USED, not discarded

    def test_the_refusal_is_distinguishable_from_having_no_floor_at_all(self):
        """A mismatch must never degrade to the uncalibrated path: an absent floor blocks
        both gate triggers (R23-54), so a silent downgrade reads as a cadence bug rather
        than as the stale floor it is."""
        with tempfile.TemporaryDirectory() as tmp:
            absent = serving.load_floor(tmp, BASE)        # nothing on disk: no exception
            self.assertEqual(absent.provenance, "absent")
            self.assertIsNone(absent.floor_pct)
            serving.write_floor(tmp, BASE, _floor_row(BASE))
            with self.assertRaises(serving.ServingFloorMismatch):
                serving.load_floor(tmp, dataclasses.replace(BASE, np=8))

    def test_the_live_store_floor_would_be_grandfathered_not_refused(self):
        """The floors on disk today predate stamping. If this ever starts raising, the
        loop cannot relaunch -- which is exactly why `unverified` exists."""
        with tempfile.TemporaryDirectory() as tmp:
            for missing in ({}, {"recipe": BASE.name}, {"floor_pct": 4.581}):
                with self.subTest(row=missing):
                    serving.floor_path(tmp, BASE).write_text(json.dumps(missing))
                    self.assertEqual(serving.load_floor(tmp, BASE).provenance,
                                     "unverified")


class FloorFilenames(unittest.TestCase):
    def test_a_safe_name_is_used_verbatim_so_the_shipped_path_is_unchanged(self):
        shipped = serving.Recipe.load(SHIPPED)
        self.assertEqual(serving.floor_path("/store", shipped).name,
                         f"serving-floor.{shipped.name}.json")
        self.assertEqual(shipped.name, "qwen3.8-27b-q8-gpu-dflash2-np4")

    def test_a_with_env_arm_does_not_resolve_to_its_bases_floor(self):
        arm = BASE.with_env(GGML_NOHUGEPAGE_PROCESS="1")
        self.assertNotEqual(serving.floor_path("/s", arm), serving.floor_path("/s", BASE))
        # readable on disk: `+` and `=` are filesystem-safe and stay verbatim
        self.assertEqual(serving.floor_path("/s", arm).name,
                         "serving-floor.t+GGML_NOHUGEPAGE_PROCESS=1.json")

    def test_a_recipe_loaded_from_json_that_already_has_env_still_derives_a_new_name(self):
        """The construction path that matters in practice: the arm is derived from a
        SHIPPED recipe, not from a literal built in a test."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.json"
            path.write_text(json.dumps(
                {**BASE.to_dict(), "name": "base", "env": {"A": "1"}}))
            loaded = serving.Recipe.load(path)
            self.assertEqual(loaded.env, {"A": "1"})
            derived = loaded.with_env(B="2")
            self.assertEqual(derived.env, {"A": "1", "B": "2"})
            self.assertNotEqual(serving.floor_path(tmp, derived),
                                serving.floor_path(tmp, loaded))
            # and the round trip through JSON does not lose the derived identity
            self.assertEqual(serving.Recipe.from_dict(derived.to_dict()).recipe_hash,
                             derived.recipe_hash)

    def test_an_unsafe_env_value_gives_a_safe_stable_bounded_filename(self):
        """An env VALUE reaches the name through `with_env`, and a path or a spaced value
        is legal env. `_`-substitution alone is not injective, so the digest of the
        ORIGINAL name is what keeps two recipes off one floor file."""
        arm = BASE.with_env(GGML_TRACE="/tmp/a b*?")
        name = serving.floor_path("/s", arm).name
        for bad in ("/", " ", "*", "?"):
            self.assertNotIn(bad, name[len("serving-floor."):-len(".json")], bad)
        self.assertEqual(name, serving.floor_path("/s", arm).name)          # stable
        self.assertEqual(name, serving.floor_path("/s", BASE.with_env(
            GGML_TRACE="/tmp/a b*?")).name)                                  # reproducible
        # two names that sanitise to the same string keep different files
        self.assertNotEqual(serving.floor_key("a/b"), serving.floor_key("a b"))

    def test_a_runaway_derived_name_stays_inside_the_filename_limit(self):
        arm = BASE
        for i in range(12):
            arm = arm.with_env(**{f"GGML_VERY_LONG_VARIABLE_NAME_{i}": "1"})
        key = serving.floor_key(arm.name)
        self.assertLessEqual(len(key), serving.FLOOR_KEY_MAX)
        self.assertLess(len(serving.floor_path("/s", arm).name.encode()), 255)
        self.assertEqual(key, serving.floor_key(arm.name))                   # deterministic

    def test_an_empty_name_is_refused_rather_than_written_to_a_stub_path(self):
        with self.assertRaises(serving.RecipeError):
            serving.floor_key("")

    def test_a_deliberate_name_collision_is_still_caught_by_the_hash(self):
        """`with_env(name=...)` is a documented escape hatch. It can point an arm at
        another arm's floor FILE -- and the identity check is what stops that file from
        being used as this arm's bar."""
        arm = BASE.with_env(name=BASE.name, GGML_NOHUGEPAGE_PROCESS="1")
        with tempfile.TemporaryDirectory() as tmp:
            serving.write_floor(tmp, BASE, _floor_row(BASE))
            self.assertEqual(serving.floor_path(tmp, arm), serving.floor_path(tmp, BASE))
            with self.assertRaises(serving.ServingFloorMismatch):
                serving.load_floor(tmp, arm)


class TheLoopActuallyUsesIt(unittest.TestCase):
    """The check only exists if `run.py` calls it. A behavioural test of `serving` passes
    just as happily while the loop keeps its own inline `json.loads` of the floor file."""

    def _source(self) -> str:
        return (Path(__file__).resolve().parent / "run.py").read_text(encoding="utf-8")

    def test_run_loads_the_floor_through_the_identity_checking_loader(self):
        source = self._source()
        self.assertIn("serving.load_floor(args.store, serving_recipe)", source)
        # and no longer builds the name-keyed path itself
        self.assertNotIn('f"serving-floor.{serving_recipe.name}.json"', source)

    def test_run_stamps_the_provenance_into_the_serving_record_and_the_status(self):
        source = self._source()
        self.assertIn('"floor_provenance": serving_floor_provenance', source)
        self.assertIn('"serving_floor_provenance": serving_floor_provenance', source)

    def test_run_does_not_swallow_the_refusal(self):
        """Fail-closed means the mismatch reaches the operator. A `try` around the load
        that fell back to `serving_floor_pct = None` would be the silent downgrade."""
        source = self._source()
        block = source.split("floor_reading = serving.load_floor", 1)[1][:400]
        self.assertNotIn("except", block)


if __name__ == "__main__":
    unittest.main()
