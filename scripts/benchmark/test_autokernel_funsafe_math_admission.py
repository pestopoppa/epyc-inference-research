"""The admission harness's decision arithmetic, tested without a GPU or a build.

The harness itself is prepared-not-executed until the run-21->22 boundary; what can
be wrong TODAY is the pure logic the operator's verdict rides on -- the divergence
detector, the one-line-geometry refusal, and the wiring defaults naming the real
branch. Injected doubles for everything that touches git.
"""
import importlib.util
from pathlib import Path
import unittest
from unittest import mock

_PATH = Path(__file__).resolve().parent / "autokernel_funsafe_math_admission.py"
_spec = importlib.util.spec_from_file_location("funsafe_admission_under_test", _PATH)
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)


class TheDivergenceDetector(unittest.TestCase):
    """(a) of the operator's question: do argmax outputs diverge at all?"""

    def test_bit_identical_streams_report_no_divergence(self):
        self.assertIsNone(harness.divergence("the same text", "the same text"))

    def test_a_single_flipped_token_is_found_at_its_position(self):
        split = harness.divergence("alpha beta gamma delta", "alpha beta GAMMA delta")
        self.assertIsNotNone(split)
        self.assertEqual(split["char_index"], 11)
        self.assertEqual(split["tokens_before_split"], 2)

    def test_a_truncated_stream_diverges_at_the_truncation(self):
        split = harness.divergence("one two three", "one two")
        self.assertEqual(split["char_index"], 7)

    def test_empty_versus_empty_is_parity(self):
        self.assertIsNone(harness.divergence("", ""))

    def test_the_continuations_show_both_arms(self):
        split = harness.divergence("x A follows", "x B follows")
        self.assertTrue(split["flag_on_continuation"].startswith("A"))
        self.assertTrue(split["flag_off_continuation"].startswith("B"))


class TheOneLineGeometryGuard(unittest.TestCase):
    """The A/B is void if the arms differ by anything beyond the CMake line."""

    def test_the_clean_pair_passes_and_names_the_parent(self):
        with mock.patch.object(harness, "_git", side_effect=[
                "parentsha", "ggml/src/ggml-hip/CMakeLists.txt"]):
            self.assertEqual(
                harness.verify_one_line_geometry(Path("/t"), "admsha"), "parentsha")

    def test_a_confounded_pair_is_refused(self):
        with mock.patch.object(harness, "_git", side_effect=[
                "parentsha",
                "ggml/src/ggml-hip/CMakeLists.txt\nggml/src/ggml-cuda/common.cuh"]):
            with self.assertRaises(SystemExit) as caught:
                harness.verify_one_line_geometry(Path("/t"), "admsha")
        self.assertIn("confounded", str(caught.exception))

    def test_an_empty_diff_is_also_refused(self):
        """Two identical commits measure the instrument, not the flag."""
        with mock.patch.object(harness, "_git", side_effect=["parentsha", ""]):
            with self.assertRaises(SystemExit):
                harness.verify_one_line_geometry(Path("/t"), "admsha")


class TheWiringNamesTheRealArtifacts(unittest.TestCase):

    def test_the_default_ref_is_the_prepared_admission_branch(self):
        """Captured by intercepting the parser, not by grepping the source: a
        default that argparse never receives would pass a text search."""
        import argparse
        captured = {}

        def capture(parser, argv=None):
            captured.update({action.dest: action.default
                             for action in parser._actions})
            raise SystemExit(0)

        with mock.patch.object(argparse.ArgumentParser, "parse_args", capture):
            with self.assertRaises(SystemExit):
                harness.main([])
        self.assertEqual(captured["admission_ref"],
                         "ak/admission/remove-funsafe-math-20260831")
        self.assertEqual(captured["pairs"], 20,
                         "a ~2%% effect needs the 20-pair floor row (1.188%%)")

    def test_the_probe_set_mixes_regimes_and_is_fixed(self):
        self.assertGreaterEqual(len(harness.PROMPTS), 8)
        self.assertEqual(len(set(harness.PROMPTS)), len(harness.PROMPTS))

    def test_greedy_defaults_are_temperature_zero_fixed_seed(self):
        self.assertEqual(harness.GREEDY_SEED, 42)
        self.assertGreaterEqual(harness.GEN_TOKENS, 64)


if __name__ == "__main__":
    unittest.main()
