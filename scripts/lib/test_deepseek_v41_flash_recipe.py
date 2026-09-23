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
        # 2026-09-23b: spec-dec is no longer PENDING, so the banner's second
        # limitation is now the GREEDY one. The banner must still carry two.
        self.assertIn("GREEDY", R.STATUS_BANNER)
        self.assertIn("EXACTNESS", R.STATUS_BANNER)


class TestSpecDecLanded(unittest.TestCase):
    """2026-09-23b: the drafter exists. These tests replace the pending ones."""

    def test_status_is_measured(self):
        self.assertEqual(R.SPEC_DEC_STATUS, "measured")

    def test_the_landed_fields_are_set(self):
        self.assertEqual(R.SPEC_DEC["spec_type"], "draft-dspark")
        self.assertEqual(R.SPEC_DEC["draft_gguf"], R.DRAFT_GGUF)
        self.assertIsNotNone(R.SPEC_DEC["alpha"])
        self.assertIsNotNone(R.SPEC_DEC["spec_draft_n_max"])

    def test_the_unmeasured_fields_are_still_none_not_guessed(self):
        # p_min was never swept and drafted_per_token never extracted. Absent
        # stays None; a plausible default would be a fabricated measurement.
        self.assertIsNone(R.SPEC_DEC["spec_draft_p_min"])
        self.assertIsNone(R.SPEC_DEC["drafted_per_token"])

    def test_flip_conditions_are_retained_with_their_state(self):
        self.assertGreaterEqual(len(R.SPEC_DEC["flips_on"]), 5)
        state = R.SPEC_DEC["flips_on_state_20260923b"]
        self.assertIn("OPEN", state["5_rederive_threads_with_drafter_on"])

    def test_artifact_has_no_mtp_tensors(self):
        # Still true, and still the reason the drafter is a SEPARATE file.
        self.assertEqual(R.TRUNK_MTP_TENSORS, 0)

    def test_mtp_spec_type_is_refused_by_name(self):
        self.assertIn("--spec-type draft-mtp", R.KNOWN_BAD_FLAG_FORMS)


class TestGreedyExactness(unittest.TestCase):
    def test_upstream_exactness_claim_is_recorded_as_false_here(self):
        self.assertFalse(R.GREEDY_EXACTNESS["holds_here"])
        self.assertFalse(R.GREEDY_EXACTNESS["measured_here"])

    def test_the_greedy_batched_variant_is_measured_and_carries_its_evidence(self):
        # DS41-T5 ran 2026-09-23 and the operator adopted it. The variant may now
        # build a command, but it must still carry the parity evidence and the
        # not-bit-exact warning, or a later reader could mistake it for exact.
        v = R.SPEC_DEC_VARIANTS["greedy-batched"]
        self.assertEqual(v["status"], "measured")
        R.build_serve_command(variant="greedy-batched")
        self.assertIn("VACUOUS", v["flipped_on"])     # the p09/p10 correction survives
        self.assertIn("synonym", v["flipped_on"].lower())
        self.assertIn("NOT bit-exact", v["honest_note"])

    def test_the_vacuous_low_entropy_pair_is_recorded_as_proving_nothing(self):
        # The original low-entropy prompts returned 1 token + EOS in every arm.
        # Zero divergence there was silence, not evidence; nothing may cite it.
        self.assertIn("must not be cited",
                      R.SPEC_DEC_VARIANTS["greedy-batched"]["flipped_on"])

    def test_block_5_is_refused_by_name_not_rediscovered(self):
        self.assertIn("REJECTED", R.SPEC_DEC_BLOCK_REJECTED[5])

    def test_greedy_is_recorded_as_a_loss_not_a_win(self):
        greedy_ctrl = R.MEASURED_SERVING["arms_tps"]["control_no_drafter"][0]
        greedy_b3 = R.MEASURED_SERVING["arms_tps"]["dspark_block_3"][0]
        self.assertLess(greedy_b3, greedy_ctrl)
        self.assertIn("LOSS", R.MEASURED_SERVING["headline"])

    def test_serving_numbers_are_observations(self):
        self.assertEqual(R.MEASURED_SERVING["instrument_class"], "serving")
        self.assertIsNone(R.MEASURED_SERVING["protocol"])
        self.assertIsNone(R.MEASURED_SERVING["reps"])


class TestSpecDecRefusals(unittest.TestCase):
    def test_serve_refuses_an_unknown_variant(self):
        with self.assertRaises(R.RecipeViolation):
            R.build_serve_command(variant="fastest-please")

    def test_serve_allows_an_explicit_no_draft_launch(self):
        cmd, env = R.build_serve_command(spec_dec=False)
        self.assertNotIn("--spec-type", cmd)
        self.assertNotIn("-md", cmd)
        self.assertEqual(env, {})

    def test_the_default_variant_emits_the_drafter(self):
        cmd, env = R.build_serve_command()
        self.assertIn("-md", cmd)
        self.assertEqual(cmd[cmd.index("--spec-type") + 1], "draft-dspark")
        self.assertEqual(cmd[cmd.index("--spec-draft-n-max") + 1], "2")
        # The default is now greedy-batched, which MUST carry the env var: without
        # it a greedy request silently falls back to the serial path and loses the
        # win (7.18 vs 11.43 t/s), which is the failure this assertion guards.
        self.assertEqual(R.SPEC_DEC_VARIANT_DEFAULT, "greedy-batched")
        self.assertEqual(env, {"LLAMA_SPEC_EXACT": "batched-greedy-inexact"})

    def test_the_exact_variant_is_still_reachable_for_reproducibility(self):
        cmd, env = R.build_serve_command(variant="greedy-exact")
        self.assertIn("-md", cmd)
        self.assertNotIn("LLAMA_SPEC_EXACT", env)

    def test_greedy_exact_variant_uses_block_3(self):
        cmd, _ = R.build_serve_command(variant="greedy-exact")
        self.assertEqual(cmd[cmd.index("--spec-draft-n-max") + 1], "3")

    def test_spec_dec_refuses_multi_slot(self):
        with self.assertRaises(R.RecipeViolation):
            R.build_serve_command(parallel_slots=4)

    def test_write_up_guard_rejects_an_exactness_claim(self):
        with self.assertRaises(R.RecipeViolation):
            R.assert_no_silent_spec_dec_claim(
                "Speculative decoding here is bit-exact at greedy.")

    def test_write_up_guard_rejects_a_greedy_speedup_claim(self):
        with self.assertRaises(R.RecipeViolation):
            R.assert_no_silent_spec_dec_claim("We measured 1.27x at greedy.")

    def test_write_up_guard_allows_an_honest_sentence(self):
        R.assert_no_silent_spec_dec_claim(
            "At temp 0.7 the DSpark drafter gives 10.48 t/s against an 8.28 control.")


class TestThreads(unittest.TestCase):
    def test_decode_is_48_and_prefill_is_96(self):
        self.assertEqual(R.THREADS, 48)
        self.assertEqual(R.THREADS_BATCH, 96)

    def test_serve_command_emits_both(self):
        cmd, _ = R.build_serve_command(spec_dec=False)
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
        cmd, _ = R.build_serve_command(spec_dec=False)
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
        # A commit hash alone is NOT the identity of the DSpark numbers.
        self.assertIn("uncommitted", R.KERNEL_WORKTREE_PATCHES)
        self.assertEqual(R.KERNEL_BRANCH, "experimental/deepseek41-port-20260923")
        self.assertIn("collides", R.KERNEL_BUILD_NUMBER_IS_NOT_AN_IDENTITY)

    def test_serve_command_uses_the_experimental_bindir(self):
        cmd, _ = R.build_serve_command(spec_dec=False)
        self.assertTrue(any(R.KERNEL_BINDIR in tok for tok in cmd))


class TestArtifact(unittest.TestCase):
    def test_arch_and_joined_size(self):
        self.assertEqual(R.TRUNK_ARCH, "deepseek41")
        self.assertEqual(R.TRUNK_BYTES, 518_596_067_328)
        self.assertEqual(R.TRUNK_TENSORS, 1046)


if __name__ == "__main__":
    unittest.main()
