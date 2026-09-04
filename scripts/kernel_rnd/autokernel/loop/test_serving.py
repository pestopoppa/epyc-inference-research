"""The serving-throughput measurement, tested WITHOUT launching a server.

What must hold before this gates a keep: the recipe is general over spec-decode type
(a model with no drafter emits no drafter flags), the champion recipe reproduces the
DF2-5-validated np4 command, and the paired A/B / floor arithmetic is right and
fail-closed when uncalibrated.
"""
import dataclasses
from pathlib import Path
import unittest
from unittest import mock

from autokernel.loop import serving


RECIPE = serving.Recipe(
    name="t", model="/m/target.gguf",
    spec_decode={"type": "draft-dflash", "drafter": "/m/draft.gguf", "ngld": 99, "draft_n_max": 8},
    np=4, ctx=16384)


class RecipeArgv(unittest.TestCase):
    def test_the_dflash_recipe_emits_the_drafter_flags(self):
        argv = RECIPE.server_argv(Path("/B"), 18311)
        self.assertEqual(argv[argv.index("-np") + 1], "4")
        self.assertEqual(argv[argv.index("--spec-type") + 1], "draft-dflash")
        self.assertEqual(argv[argv.index("-md") + 1], "/m/draft.gguf")
        self.assertEqual(argv[argv.index("--spec-draft-n-max") + 1], "8")
        self.assertIn("--no-kv-unified", argv)

    def test_a_none_spec_recipe_emits_no_drafter_flags(self):
        plain = dataclasses.replace(RECIPE, spec_decode={"type": "none"})
        argv = plain.server_argv(Path("/B"), 1)
        self.assertNotIn("-md", argv)
        self.assertNotIn("--spec-type", argv)
        # still a valid server command
        self.assertEqual(argv[argv.index("-np") + 1], "4")

    def test_mtp_is_general_too(self):
        mtp = dataclasses.replace(RECIPE, spec_decode={"type": "draft-mtp", "drafter": "/m/mtp.gguf"})
        argv = mtp.server_argv(Path("/B"), 1)
        self.assertEqual(argv[argv.index("--spec-type") + 1], "draft-mtp")

    def test_kv_unified_flips_the_flag(self):
        self.assertIn("--kv-unified", dataclasses.replace(RECIPE, kv_unified=True).server_argv(Path("/B"), 1))

    def test_the_shipped_champion_recipe_matches_df2_5_np4(self):
        r = serving.Recipe.load(Path(__file__).resolve().parents[4]
                                / "artifacts/serving-recipes/qwen3.8-27b-q8-gpu-dflash2-np4.json")
        argv = " ".join(r.server_argv(Path("/mnt/raid0/llm/tmp/champ2/build-hip"), 18099))
        for token in ("-np 4", "-c 16384", "--spec-type draft-dflash",
                      "--spec-draft-n-max 8", "--no-kv-unified", "-ngld 99"):
            self.assertIn(token, argv, token)


class Arithmetic(unittest.TestCase):
    def test_compare_is_fail_closed_without_a_floor(self):
        with mock.patch.object(serving, "_measure_once", side_effect=[100.0, 120.0]):
            out = serving.compare(RECIPE, Path("/a"), Path("/c"), pairs=1, floor_pct=None)
        self.assertAlmostEqual(out["effect_pct"], 20.0, places=3)
        self.assertIsNone(out["decisive"])  # uncalibrated -> never decisive

    def test_compare_is_decisive_above_the_floor(self):
        with mock.patch.object(serving, "_measure_once", side_effect=[100.0, 120.0]):
            out = serving.compare(RECIPE, Path("/a"), Path("/c"), pairs=1, floor_pct=1.0)
        self.assertTrue(out["decisive"])

    def test_a_within_floor_effect_is_not_decisive(self):
        with mock.patch.object(serving, "_measure_once", side_effect=[100.0, 100.4]):
            out = serving.compare(RECIPE, Path("/a"), Path("/c"), pairs=1, floor_pct=1.0)
        self.assertFalse(out["decisive"])

    def test_floor_is_p95_of_the_aa_spread(self):
        with mock.patch.object(serving, "_measure_once",
                               side_effect=[100.0, 101.0, 99.0, 100.5, 99.5]):
            out = serving.calibrate_floor(RECIPE, Path("/b"), samples=5)
        self.assertGreater(out["floor_pct"], 0.0)
        self.assertEqual(out["samples"], 5)


if __name__ == "__main__":
    unittest.main()
