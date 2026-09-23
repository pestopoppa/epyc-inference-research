"""Tests for the PRELIMINARY DeepSeek-V4.1-Flash canonical recipe.

These are contract tests, not artifact tests: nothing here touches the 483 GiB
GGUF, the experimental build, or the host. They exist so the two limitations the
module advertises cannot be quietly removed — a recipe that stops SAYING it is
spec-dec-incomplete while still BEING spec-dec-incomplete is the exact failure
the module was written to prevent.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import canonical_recipe as CR                 # noqa: E402
import deepseek_v41_flash_recipe as R         # noqa: E402


class TestStatusIsLoud(unittest.TestCase):
    def test_grade_is_preliminary(self):
        self.assertEqual(R.STATUS["grade"], "PRELIMINARY")

    def test_category_is_candidate_not_optimum(self):
        # A draft path EXISTS for this model; we have not built it. That makes
        # today's no-draft number a CANDIDATE, never an OPTIMUM.
        self.assertEqual(R.STATUS["measurement_category"], "CANDIDATE")

    def test_measured_numbers_declare_bench_class_and_no_protocol(self):
        self.assertEqual(R.MEASURED["instrument_class"], "bench")
        self.assertIsNone(R.MEASURED["protocol"])

    def test_banner_names_both_limitations(self):
        self.assertIn("PRELIMINARY", R.STATUS_BANNER)
        self.assertIn("PENDING", R.STATUS_BANNER)


class TestSpecDecFieldsExistAndArePending(unittest.TestCase):
    def test_status_is_pending_not_absent(self):
        self.assertEqual(R.SPEC_DEC_STATUS, "pending")
        self.assertFalse(R.SPEC_DEC["satisfied"])

    def test_every_flag_field_is_present_and_none(self):
        # PRESENT so the recipe cannot be read as "this model has no draft path";
        # None so it cannot be read as configured.
        for key in ("draft_gguf", "spec_type", "spec_draft_n_max",
                    "spec_draft_p_min", "alpha", "drafted_per_token"):
            self.assertIn(key, R.SPEC_DEC, f"{key} must EXIST, pending is not absent")
            self.assertIsNone(R.SPEC_DEC[key], f"{key} must be None while pending")

    def test_flip_conditions_are_enumerated(self):
        self.assertGreaterEqual(len(R.SPEC_DEC["flips_on"]), 5)

    def test_artifact_has_no_mtp_tensors(self):
        self.assertEqual(R.TRUNK_MTP_TENSORS, 0)

    def test_serve_refuses_by_default(self):
        with self.assertRaises(R.RecipeViolation):
            R.build_serve_command()

    def test_serve_allows_an_explicit_no_draft_launch(self):
        cmd = R.build_serve_command(spec_dec=False)
        self.assertNotIn("--spec-type", cmd)
        self.assertNotIn("-md", cmd)

    def test_write_up_guard_rejects_a_spec_dec_claim(self):
        with self.assertRaises(R.RecipeViolation):
            R.assert_no_silent_spec_dec_claim("Measured with speculative decoding enabled.")


class TestThreads(unittest.TestCase):
    def test_decode_is_48_and_prefill_is_96(self):
        self.assertEqual(R.THREADS, 48)
        self.assertEqual(R.THREADS_BATCH, 96)

    def test_serve_command_emits_both(self):
        cmd = R.build_serve_command(spec_dec=False)
        self.assertEqual(cmd[cmd.index("-t") + 1], "48")
        self.assertEqual(cmd[cmd.index("-tb") + 1], "96")
        R.assert_decode_threads(cmd)
        R.assert_no_bad_flag_forms(cmd)

    def test_192_is_refused_not_offered(self):
        self.assertIn("REFUSED", R.THREADS_REJECTED[192])

    def test_measured_sweep_supports_the_choice(self):
        tg = R.MEASURED["tg128_tps_by_threads"]
        self.assertEqual(max(tg, key=lambda k: max(_as_tuple(tg[k]))), 48)
        self.assertLess(max(_as_tuple(tg[192])), max(_as_tuple(tg[48])) / 2)


def _as_tuple(v):
    return v if isinstance(v, tuple) else (v,)


class TestInheritsGlobalRecipe(unittest.TestCase):
    def test_inherits_not_forks(self):
        R.assert_inherits_canonical()

    def test_prefix_and_omp_are_the_same_objects(self):
        self.assertIs(R.PREFIX, CR.CANONICAL_PREFIX)
        self.assertIs(R.OMP_ENV, CR.CANONICAL_OMP_ENV)

    def test_iqk_fa_and_no_mmap_are_carried(self):
        self.assertEqual(R.OMP_ENV["GGML_IQK"], "1")
        self.assertIn("-fa", R.INHERITED_BENCH_FLAGS)
        self.assertIn("-mmp", R.INHERITED_BENCH_FLAGS)
        cmd = R.build_serve_command(spec_dec=False)
        self.assertIn("--no-mmap", cmd)
        self.assertEqual(cmd[cmd.index("-fa") + 1], "on")

    def test_pre_evict_and_placement_proof_are_inherited_and_required(self):
        self.assertEqual(R.PRE_EVICT_GIB, CR.CANONICAL_PRE_EVICT_GIB)
        self.assertTrue(R.PLACEMENT_PROOF_REQUIRED)


class TestBinaryIdentity(unittest.TestCase):
    def test_production_discovery_is_off(self):
        # Production does not know the `deepseek41` architecture.
        self.assertFalse(R.USE_PRODUCTION_DISCOVERY)

    def test_identity_is_the_triple_not_the_build_number(self):
        self.assertEqual(R.KERNEL_COMMIT, "7c18bb8c1")
        self.assertEqual(R.KERNEL_BRANCH, "experimental/deepseek41-port-20260923")
        self.assertIn("collides", R.KERNEL_BUILD_NUMBER_IS_NOT_AN_IDENTITY)

    def test_serve_command_uses_the_experimental_bindir(self):
        cmd = R.build_serve_command(spec_dec=False)
        self.assertTrue(any(R.KERNEL_BINDIR in tok for tok in cmd))


class TestArtifact(unittest.TestCase):
    def test_arch_and_joined_size(self):
        self.assertEqual(R.TRUNK_ARCH, "deepseek41")
        self.assertEqual(R.TRUNK_BYTES, 518_596_067_328)
        self.assertEqual(R.TRUNK_TENSORS, 1046)


if __name__ == "__main__":
    unittest.main()
