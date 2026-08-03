#!/usr/bin/env python3
"""Unit tests for `adapters/qwentts_tts.py` — the `qwentts_tts` backend adapter.

NO inference, NO synthesis, NO benchmark, NO build, NO model, NO audio. Nothing here
starts, stops or signals a process, and nothing here writes a file. Every "verifier
report" is a fixture string shaped exactly like
`scripts/utils/verify_ggml_linkage.sh`'s own `printf` output, and every path is a
made-up experimental worktree except where a test deliberately checks that a real
production path is REFUSED.

The suite is organised around this backend's three asymmetries and its two scars:

  * the stable path points at `build`, NOT `build/bin` like the other three;
  * `ggml` is a git SUBMODULE, so the frozen production change is `ggml | 2 +-` — one
    line — in the superproject and 4 files / 115 lines in the submodule. A closure or
    a complexity assessment computed without traversal under-reports by two orders of
    magnitude;
  * it runs ggml 0.17.0 between llama.cpp's 0.16.0 and whisper.cpp's 0.18.0, so a
    binary inheriting another tree's ggml runs silently wrong;
  * `test-backend-ops` reported `ARGSORT 46/46` — 100 % pass — while the failing
    gfx90a shapes were silently skipped; after the fix the same suite reported
    `74/74`. Only the ENUMERATION distinguishes the two readings;
  * round-trip WER measured 0.0 % on the CPU anchor, i.e. saturated, and is gameable
    by a cached waveform or a flat robotic monotone.

Run standalone (no pytest needed):
    python3 -m unittest scripts/kernel_rnd/autokernel/adapters/test_qwentts_tts.py
    python3 -W error::ResourceWarning -m unittest \
        scripts/kernel_rnd/autokernel/adapters/test_qwentts_tts.py
"""
from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

_KERNEL_RND = str(Path(__file__).resolve().parents[2])
if _KERNEL_RND not in sys.path:
    sys.path.insert(0, _KERNEL_RND)

from autokernel import schemas as S  # noqa: E402
from autokernel import storage  # noqa: E402
from autokernel.adapters import qwentts_tts as Q  # noqa: E402
from autokernel.adapters import whisper_stt as W  # noqa: E402
from autokernel.evaluator import correctness, integrity  # noqa: E402

#: The sibling adapter's module path, used only to prove that ITS source does not
#: satisfy THIS module's self-audit. Read as text.
W_MODULE_PATH = str(Path(__file__).resolve().parent / "whisper_stt.py")

EXP_TREE = "/mnt/raid0/llm/qwentts.cpp-experimental"
EXP_BIN = "/mnt/raid0/llm/qwentts.cpp-experimental/build/qwen-tts"
EXP_LIB = "/mnt/raid0/llm/qwentts.cpp-experimental/build"


def _sha(tag: str) -> str:
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


def _report(rows, *, expect=EXP_LIB,
            trailer="PASS: all linked ggml libraries resolve inside "):
    out = [f"binary : {EXP_BIN}", f"expect : libraries under {expect}", ""]
    for state, name, path in rows:
        out.append("  %s %-28s -> %s" % ("OK  " if state == "OK" else "BAD ", name, path))
    out += ["", "LD_LIBRARY_PATH order as the loader sees it:", f"     1  {expect}", ""]
    out.append(trailer + expect)
    return "\n".join(out) + "\n"


_GOOD_ROWS = [
    ("OK", "libggml-base.so.0.17.0", f"{EXP_LIB}/libggml-base.so.0.17.0"),
    ("OK", "libggml-cpu.so.0.17.0", f"{EXP_LIB}/libggml-cpu.so.0.17.0"),
    ("OK", "libggml.so.0.17.0", f"{EXP_LIB}/libggml.so.0.17.0"),
]


class TreeIdentityTest(unittest.TestCase):
    def test_identity_matches_the_speech_freeze_receipt(self):
        facts = Q.tree_facts()
        self.assertEqual(facts.backend, "qwentts_tts")
        self.assertEqual(facts.source_tree, "qwentts.cpp")
        self.assertEqual(facts.frozen_branch, "production-speech-v1")
        self.assertEqual(facts.frozen_commit,
                         "2c1b5182e7e9f1acaa04405ff21747d8a7acf4d5")
        self.assertEqual(facts.frozen_ggml_submodule_commit,
                         "b86f660238dcc1a83b7cbf5a72d355a965de9245")
        self.assertEqual(facts.ggml_generation, "0.17.0")

    def test_the_three_trees_run_three_different_ggml_generations(self):
        # The reason per-launcher LD_LIBRARY_PATH isolation is load-bearing rather
        # than cosmetic.
        self.assertNotEqual(Q.GGML_GENERATION, W.GGML_GENERATION)
        self.assertEqual({Q.GGML_GENERATION, W.GGML_GENERATION}, {"0.17.0", "0.18.0"})

    def test_ggml_is_a_submodule_here_and_in_tree_in_the_sibling(self):
        self.assertEqual(Q.GGML_VENDORING, "submodule")
        self.assertEqual(Q.SUBMODULE_PATHS, ("ggml",))
        self.assertEqual(W.GGML_VENDORING, "in_tree")

    def test_the_stable_path_does_not_end_in_bin_unlike_the_other_three(self):
        self.assertFalse(Q.STABLE_TARGET.endswith("/bin"))
        self.assertEqual(Q.STABLE_TARGET, "/mnt/raid0/llm/qwentts.cpp/build")
        for target in Q.SIBLING_STABLE_TARGETS.values():
            self.assertTrue(target.endswith("/bin"), target)

    def test_a_transaction_that_appended_bin_is_caught_at_the_dry_run(self):
        check = Q.check_stable_path_assumption("/mnt/raid0/llm/qwentts.cpp/build/bin")
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("assume uniformity", " ".join(check.reasons))

    def test_the_correct_target_passes(self):
        self.assertEqual(Q.check_stable_path_assumption(Q.STABLE_TARGET).outcome, S.PASS)

    def test_production_tree_mirror_agrees_with_the_two_other_copies(self):
        self.assertEqual(set(storage.PRODUCTION_TREES) - set(Q.PRODUCTION_TREE_ROOTS),
                         set())
        self.assertEqual(set(correctness.PRODUCTION_TREE_ROOTS),
                         set(Q.PRODUCTION_TREE_ROOTS))


class FreezeScopeTest(unittest.TestCase):
    def test_qwentts_is_independently_freezable(self):
        scope = Q.freeze_scope()
        self.assertTrue(scope.independently_freezable)
        self.assertEqual(scope.backends, ("qwentts_tts",))
        self.assertEqual(scope.shares_tree_with, ())

    def test_the_two_speech_trees_are_independent_of_each_other(self):
        self.assertNotEqual(Q.SOURCE_TREE, W.SOURCE_TREE)
        self.assertTrue(Q.freeze_scope().independently_freezable)
        self.assertTrue(W.freeze_scope().independently_freezable)

    def test_joining_another_trees_champion_is_refused(self):
        with self.assertRaises(Q.WrongReleasePath):
            Q.refuse_llama_champion("llama.cpp")
        with self.assertRaises(Q.WrongReleasePath):
            Q.refuse_llama_champion("whisper.cpp")
        Q.refuse_llama_champion("qwentts.cpp")

    def test_the_stack_change_path_is_refused_outright(self):
        with self.assertRaises(Q.WrongReleasePath):
            Q.refuse_stack_change_path()


class PathDenialTest(unittest.TestCase):
    def test_a_candidate_path_inside_the_production_tree_is_refused(self):
        with self.assertRaises(Q.ProductionPathRefused):
            Q.binary_path("/mnt/raid0/llm/qwentts.cpp", "qwen-tts")

    def test_no_inventory_entry_contains_a_bin_segment(self):
        for spec in Q.binary_inventory():
            self.assertTrue(spec.rel_path.startswith("build/"), spec.rel_path)
            self.assertNotIn("/bin/", spec.rel_path)

    def test_the_tts_server_binary_matches_the_freeze_receipt_location(self):
        self.assertEqual(Q.binary_path(EXP_TREE, "tts-server"),
                         f"{EXP_TREE}/build/tts-server")

    def test_an_experimental_sibling_directory_is_not_a_production_path(self):
        Q.check_not_production_path(EXP_BIN)

    def test_an_unknown_binary_is_refused_by_name(self):
        with self.assertRaises(Q.UnknownBinary):
            Q.binary_path(EXP_TREE, "whisper-cli")

    def test_an_anchor_must_be_the_frozen_production_binary(self):
        Q.expect_production_anchor("/mnt/raid0/llm/qwentts.cpp/build/tts-server")
        with self.assertRaises(Q.QwenTtsAdapterError):
            Q.expect_production_anchor(EXP_BIN)

    # --- regression: aliases of the frozen tree ------------------------------
    # Before the fix both of these returned None (no refusal) while naming a file
    # inside the frozen production tree.

    def test_a_leading_double_slash_does_not_walk_through_the_refusal(self):
        for root in Q.PRODUCTION_TREE_ROOTS:
            with self.assertRaises(Q.QwenTtsAdapterError):
                Q.check_not_production_path("/" + root + "/build/x")
        with self.assertRaises(Q.QwenTtsAdapterError):
            Q.binary_path("//mnt/raid0/llm/qwentts.cpp", "qwen-tts")
        with self.assertRaises(Q.ProductionPathRefused):
            Q.check_not_production_path("///mnt/raid0/llm/qwentts.cpp/build/x")

    def test_the_stable_kernel_symlink_is_refused_as_a_production_path(self):
        with self.assertRaises(Q.ProductionPathRefused):
            Q.check_not_production_path(Q.STABLE_PATH + "/qwen-tts")
        for alias in Q.PRODUCTION_PATH_ALIASES:
            with self.assertRaises(Q.ProductionPathRefused):
                Q.check_not_production_path(alias + "/tts/qwen-tts")

    def test_a_sibling_of_the_alias_root_is_not_refused(self):
        Q.check_not_production_path("/mnt/raid0/llm/kernels-experimental/tts/x")


class LinkageTest(unittest.TestCase):
    def test_the_environment_is_declared_in_full(self):
        inv = Q.linkage_command(EXP_BIN, library_path_entries=[EXP_LIB, "/opt/rocm/lib"])
        self.assertEqual(inv.argv, (Q.LINKAGE_VERIFIER, EXP_BIN, EXP_LIB))
        self.assertEqual(inv.env["LD_LIBRARY_PATH"], f"{EXP_LIB}:/opt/rocm/lib")

    def test_the_binarys_own_directory_must_be_first(self):
        with self.assertRaises(Q.QwenTtsAdapterError):
            Q.linkage_command(EXP_BIN,
                              library_path_entries=["/mnt/raid0/llm/whisper.cpp/build/bin",
                                                    EXP_LIB])

    def test_a_clean_report_is_pass(self):
        self.assertEqual(Q.interpret_linkage_report(_report(_GOOD_ROWS), 0).check.outcome,
                         S.PASS)

    def test_another_trees_ggml_fails_and_names_the_generation_spread(self):
        rows = list(_GOOD_ROWS)
        rows[0] = ("BAD", "libggml-base.so.0",
                   "/mnt/raid0/llm/whisper.cpp/build/bin/libggml-base.so.0")
        verdict = Q.interpret_linkage_report(
            _report(rows, trailer="FAIL: 1 library/libraries resolve OUTSIDE "), 1)
        self.assertEqual(verdict.check.outcome, S.FAIL)
        self.assertIn("0.17.0", " ".join(verdict.check.reasons))

    def test_zero_resolved_libraries_is_could_not_check_despite_exit_zero(self):
        text = (f"binary : {EXP_BIN}\n\n  (no ggml/whisper/llama libs in ldd output "
                f"— statically linked, or ldd failed)\n\nPASS: all linked ggml "
                f"libraries resolve inside {EXP_LIB}\n")
        verdict = Q.interpret_linkage_report(text, 0)
        self.assertEqual(verdict.check.outcome, S.COULD_NOT_CHECK)
        self.assertFalse(verdict.check.passed)

    def test_the_static_qwen_core_archive_is_not_an_expected_shared_library(self):
        # `libqwen-core.a` never appears in ldd output, so listing it would make
        # every report look incomplete.
        self.assertNotIn("libqwen-core.so", Q.EXPECTED_SHARED_LIBRARIES)
        self.assertEqual(Q.interpret_linkage_report(_report(_GOOD_ROWS), 0).missing_expected,
                         ())

    def test_a_library_outside_the_scripts_name_filter_is_could_not_check(self):
        rows = [r for r in _GOOD_ROWS if not r[1].startswith("libggml-cpu")]
        verdict = Q.interpret_linkage_report(_report(rows), 0)
        self.assertEqual(verdict.check.outcome, S.COULD_NOT_CHECK)
        self.assertIn("libggml*", " ".join(verdict.check.reasons))

    def test_use_gpu_equals_one_alone_is_not_device_evidence(self):
        check = Q.check_device_evidence("qwen_tts: use gpu = 1\n", expected_lane="gpu")
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("REQUESTED", " ".join(check.reasons))

    def test_a_real_device_line_passes(self):
        self.assertEqual(
            Q.check_device_evidence("ggml_cuda_init: Device 0: AMD Instinct MI210\n",
                                    expected_lane="gpu").outcome, S.PASS)


class MetricTest(unittest.TestCase):
    def test_a_bare_real_time_factor_is_refused_by_name(self):
        # The receipt records rtf 0.169 (lower-better); the handoff records
        # xRT 5.47x (higher-better). They are reciprocals of one another.
        with self.assertRaises(Q.UnknownMetric) as ctx:
            Q.metric_direction("real_time_factor")
        self.assertIn("reciprocal", str(ctx.exception))

    def test_mos_is_refused_because_no_signal_here_is_one(self):
        with self.assertRaises(Q.UnknownMetric):
            Q.metric_direction("mos")

    def test_rtf_and_xrt_are_reciprocals_and_convert_explicitly(self):
        self.assertEqual(Q.metric_direction("rtf"), "lower_better")
        self.assertEqual(Q.metric_direction("xrt"), "higher_better")
        self.assertAlmostEqual(Q.rtf_from_xrt(5.47), 0.18282, places=5)
        self.assertAlmostEqual(Q.xrt_from_rtf(0.169), 5.9172, places=4)

    def test_a_zero_xrt_is_a_categorical_failure_not_a_slow_run(self):
        # `utt0: NO AUDIO PRODUCED` was a real outcome of the 2026-07-31 GPU bench.
        with self.assertRaises(Q.QwenTtsAdapterError):
            Q.rtf_from_xrt(0.0)

    def test_no_declared_metric_is_a_task_rate(self):
        for metric in Q.METRIC_DIRECTIONS:
            self.assertEqual(Q.check_metric_commensurable(metric).outcome, S.PASS, metric)

    def test_phases_are_the_engines_own_stages(self):
        self.assertNotIn(Q.BACKEND, S.PHASES_BY_BACKEND)
        for phase in ("talker", "code_predictor", "codec_decode", "end_to_end"):
            self.assertEqual(Q.check_phase(phase), phase)
        with self.assertRaises(Q.UnknownPhase):
            Q.check_phase("decode")


class ComplexityAndClosureTest(unittest.TestCase):
    def test_the_ceiling_uses_the_expanded_submodule_figures(self):
        ceiling = Q.complexity_ceiling()
        self.assertEqual(ceiling.max_diff_lines, 115)
        self.assertEqual(ceiling.max_files_touched, 4)
        self.assertTrue(ceiling.shared_core_modification_requires_review)

    def test_the_superproject_only_figures_would_understate_by_two_orders(self):
        self.assertEqual(Q._SUPERPROJECT_ONLY_MAX_CHANGED_LINES, 2)
        self.assertGreater(Q._EXPANDED_MAX_CHANGED_LINES / 2, 50)
        self.assertIn("b86f6602", Q.CEILING_DERIVATION)

    def test_assessing_complexity_without_traversal_is_refused(self):
        diff = integrity.SourceDiff(files=(
            integrity.FileDiff(path="ggml", old_path=None, added_lines=1,
                               removed_lines=1, hunks=1, is_new_file=False,
                               is_deleted_file=False, is_rename=False, is_binary=False,
                               observed_old_extent=1),))
        with self.assertRaises(Q.SubmoduleClosureMissing):
            Q.assess_complexity(diff, change_class="arithmetic", domains=["ggml"],
                                submodule_traversed=False)

    def test_the_expanded_argsort_change_is_marked_for_human_review(self):
        # The real production patch: argsort.cu 70+/26-, argsort.cuh 3+/0-,
        # ggml-cuda.cu 3+/1-, vendors/hip.h 10+/2-.
        diff = integrity.SourceDiff(files=(
            integrity.FileDiff(path="ggml/src/ggml-cuda/argsort.cu", old_path=None,
                               added_lines=70, removed_lines=26, hunks=6,
                               is_new_file=False, is_deleted_file=False,
                               is_rename=False, is_binary=False, observed_old_extent=120),
            integrity.FileDiff(path="ggml/src/ggml-cuda/argsort.cuh", old_path=None,
                               added_lines=3, removed_lines=0, hunks=1,
                               is_new_file=False, is_deleted_file=False,
                               is_rename=False, is_binary=False, observed_old_extent=10),
            integrity.FileDiff(path="ggml/src/ggml-cuda/ggml-cuda.cu", old_path=None,
                               added_lines=3, removed_lines=1, hunks=1,
                               is_new_file=False, is_deleted_file=False,
                               is_rename=False, is_binary=False, observed_old_extent=900),
            integrity.FileDiff(path="ggml/src/ggml-cuda/vendors/hip.h", old_path=None,
                               added_lines=10, removed_lines=2, hunks=2,
                               is_new_file=False, is_deleted_file=False,
                               is_rename=False, is_binary=False, observed_old_extent=60),
        ))
        self.assertEqual(diff.total_changed, 115)
        assessment = Q.assess_complexity(diff, change_class="arithmetic",
                                         domains=["ggml"], submodule_traversed=True)
        # 115 == the ceiling, so size alone does not trip it; shared core does.
        self.assertTrue(assessment.requires_human_code_review)
        self.assertIn("shared ggml core", " ".join(assessment.reasons))

    def test_the_shared_core_marking_is_traced_from_the_diff_not_declared(self):
        # Regression. A small change inside the expanded submodule sits under both
        # size ceilings (115 lines / 4 files), so the shared-core clause is the only
        # reason to mark it — and declaring `domains=["src"]` used to remove that
        # reason, yielding `requires_human_code_review: false` on a ggml edit.
        diff = integrity.SourceDiff(files=(
            integrity.FileDiff(path="ggml/src/ggml-cuda/argsort.cu", old_path=None,
                               added_lines=1, removed_lines=1, hunks=1,
                               is_new_file=False, is_deleted_file=False,
                               is_rename=False, is_binary=False,
                               observed_old_extent=120),))
        self.assertEqual(Q.diff_domains(diff), ("ggml",))
        self.assertEqual(Q.shared_core_paths(diff), ("ggml/src/ggml-cuda/argsort.cu",))
        assessment = Q.assess_complexity(diff, change_class="parameter",
                                         domains=["src"], submodule_traversed=True)
        self.assertTrue(assessment.requires_human_code_review, assessment.reasons)
        self.assertTrue(assessment.measured["touches_shared_core"])

    def test_an_under_declared_domain_list_fails_against_its_own_diff(self):
        diff = integrity.SourceDiff(files=(
            integrity.FileDiff(path="ggml/src/ggml-cuda/argsort.cu", old_path=None,
                               added_lines=1, removed_lines=1, hunks=1,
                               is_new_file=False, is_deleted_file=False,
                               is_rename=False, is_binary=False,
                               observed_old_extent=120),))
        check = Q.check_declared_domains_cover_diff(diff, ["src"])
        self.assertEqual(check.outcome, S.FAIL, check.reasons)
        self.assertIn("ggml", " ".join(check.reasons))
        self.assertEqual(
            Q.check_declared_domains_cover_diff(diff, ["ggml"]).outcome, S.PASS)

    def test_a_closure_that_skipped_the_submodule_fails(self):
        check = Q.check_closure_traversed_submodules(["src", "include"])
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("ggml", " ".join(check.reasons))

    def test_a_closure_that_traversed_the_submodule_passes(self):
        self.assertEqual(
            Q.check_closure_traversed_submodules(["src", "ggml"]).outcome, S.PASS)

    def test_an_untraversed_stage_one_cannot_be_classified_at_all(self):
        check = Q.classify_unchanged_result(stage1_closure_empty=True,
                                            stage2_normalized_identical=True,
                                            submodule_traversed=False)
        self.assertEqual(check.outcome, S.COULD_NOT_CHECK)
        self.assertIn("silence, not evidence", " ".join(check.reasons))

    def test_a_no_op_candidate_is_refused(self):
        check = Q.classify_unchanged_result(stage1_closure_empty=True,
                                            stage2_normalized_identical=True,
                                            submodule_traversed=True)
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("NO-OP", " ".join(check.reasons))

    def test_a_genuine_change_passes(self):
        self.assertEqual(
            Q.classify_unchanged_result(stage1_closure_empty=False,
                                        stage2_normalized_identical=False,
                                        submodule_traversed=True).outcome, S.PASS)

    def test_every_envelope_is_a_valid_change_class(self):
        envelopes = Q.change_class_envelopes()
        for name, env in envelopes.items():
            self.assertIn(name, S.CHANGE_CLASSES)
            self.assertIs(integrity.envelope_for(envelopes, name), env)


class InputIdentityTest(unittest.TestCase):
    def _record(self, **over):
        base = {
            "prompt_text_sha256": _sha("hello"), "prompt_text_bytes": 5,
            "tokenizer_sha256": _sha("tok"), "talker_weights_sha256": _sha("talker"),
            "code_predictor_weights_sha256": _sha("cp"),
            "speaker_conditioning_sha256": None,
            "sampling_policy": {"greedy": True}, "cache_state": "cold",
        }
        base.update(over)
        return base

    def test_a_complete_record_passes(self):
        self.assertEqual(Q.check_input_identity(self._record()).outcome, S.PASS)

    def test_a_missing_clone_reference_field_fails_but_an_explicit_none_passes(self):
        # Absent and none are different facts.
        record = self._record()
        del record["speaker_conditioning_sha256"]
        self.assertEqual(Q.check_input_identity(record).outcome, S.FAIL)
        self.assertEqual(
            Q.check_input_identity(
                self._record(speaker_conditioning_sha256=_sha("freeman.spk"))).outcome,
            S.PASS)

    def test_a_served_from_cache_arm_fails_as_control_threes_shape(self):
        check = Q.check_input_identity(self._record(cache_state="served_from_cache"))
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("degraded-negative", " ".join(check.reasons))

    def test_an_unknown_cache_state_fails(self):
        self.assertEqual(Q.check_input_identity(self._record(cache_state="hot")).outcome,
                         S.FAIL)

    def test_a_placeholder_digest_is_refused(self):
        self.assertEqual(
            Q.check_input_identity(self._record(tokenizer_sha256="f" * 64)).outcome,
            S.FAIL)


class GreedyArmTest(unittest.TestCase):
    def test_a_greedy_arm_passes(self):
        self.assertEqual(Q.check_greedy_arm(greedy=True, temperature=0.0).outcome, S.PASS)

    def test_a_sampled_arm_may_not_carry_a_release_verdict(self):
        check = Q.check_greedy_arm(greedy=False, temperature=0.7)
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("diagnostic", " ".join(check.reasons))

    def test_an_undeclared_sampling_policy_is_could_not_check(self):
        self.assertEqual(Q.check_greedy_arm(greedy=None, temperature=None).outcome,
                         S.COULD_NOT_CHECK)

    def test_greedy_true_with_a_nonzero_temperature_is_a_contradiction(self):
        self.assertEqual(Q.check_greedy_arm(greedy=True, temperature=0.7).outcome, S.FAIL)


class IdentityOracleTest(unittest.TestCase):
    def test_matching_code_sequences_pass(self):
        digest = _sha("codes")
        self.assertEqual(
            Q.check_code_sequence_identity(candidate_sha256=digest,
                                           anchor_sha256=digest).outcome, S.PASS)

    def test_diverging_code_sequences_fail(self):
        check = Q.check_code_sequence_identity(candidate_sha256=_sha("a"),
                                               anchor_sha256=_sha("b"))
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("LM half", " ".join(check.reasons))

    def test_a_build_that_does_not_expose_codes_is_could_not_check(self):
        check = Q.check_code_sequence_identity(candidate_sha256=None,
                                               anchor_sha256=_sha("b"))
        self.assertEqual(check.outcome, S.COULD_NOT_CHECK)
        self.assertNotEqual(check.outcome, S.PASS)

    def test_a_bitwise_stable_anchor_gives_a_zero_tolerance(self):
        self.assertEqual(
            Q.derive_waveform_tolerance(anchor_aa_dispersion=0.0,
                                        determinism_class="bitwise_stable"), 0.0)

    def test_a_bitwise_stable_claim_with_nonzero_dispersion_is_a_hard_finding(self):
        with self.assertRaises(Q.QwenTtsAdapterError):
            Q.derive_waveform_tolerance(anchor_aa_dispersion=0.01,
                                        determinism_class="bitwise_stable")

    def test_an_unmeasured_determinism_class_cannot_derive_a_tolerance(self):
        with self.assertRaises(Q.DerivationImpossible):
            Q.derive_waveform_tolerance(anchor_aa_dispersion=0.0,
                                        determinism_class="not_measured")

    def test_the_tolerance_is_the_measured_dispersion_never_a_literal(self):
        self.assertEqual(
            Q.derive_waveform_tolerance(anchor_aa_dispersion=0.004,
                                        determinism_class="bitwise_unstable"), 0.004)

    def test_a_sample_count_change_is_categorical_not_a_distance(self):
        check = Q.check_waveform_identity(candidate_sample_count=140_000,
                                          anchor_sample_count=140_160,
                                          max_abs_difference=0.0,
                                          spectral_distance=0.0, tolerance=1.0)
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("categorical", " ".join(check.reasons))

    def test_no_audio_produced_is_a_failure_not_a_zero_length_pass(self):
        check = Q.check_waveform_identity(candidate_sample_count=0,
                                          anchor_sample_count=140_160,
                                          max_abs_difference=0.0,
                                          spectral_distance=0.0, tolerance=1.0)
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("NO AUDIO PRODUCED", " ".join(check.reasons))

    def test_a_distance_within_the_derived_tolerance_passes(self):
        self.assertEqual(
            Q.check_waveform_identity(candidate_sample_count=140_160,
                                      anchor_sample_count=140_160,
                                      max_abs_difference=0.002,
                                      spectral_distance=0.003, tolerance=0.004).outcome,
            S.PASS)

    def test_a_distance_beyond_the_tolerance_fails(self):
        self.assertEqual(
            Q.check_waveform_identity(candidate_sample_count=140_160,
                                      anchor_sample_count=140_160,
                                      max_abs_difference=0.5,
                                      spectral_distance=0.0, tolerance=0.004).outcome,
            S.FAIL)

    def test_the_combined_verdict_takes_the_worst_of_the_two_layers(self):
        verdict = Q.combine_identity(layer1=S.Check(S.PASS),
                                     layer2=S.Check(S.FAIL, ("waveform drifted",)))
        self.assertEqual(verdict.combined.outcome, S.FAIL)
        verdict = Q.combine_identity(layer1=S.Check(S.COULD_NOT_CHECK, ("no codes",)),
                                     layer2=S.Check(S.PASS))
        self.assertEqual(verdict.combined.outcome, S.COULD_NOT_CHECK)


class NumericalSafetyTest(unittest.TestCase):
    def _kwargs(self, **over):
        base = {"nan_count": 0, "inf_count": 0, "clipping_fraction": 0.001,
                "clipping_band": [0.0, 0.01], "dc_offset": 0.0002,
                "dc_band": [-0.001, 0.001]}
        base.update(over)
        return base

    def test_clean_audio_passes(self):
        self.assertEqual(Q.check_numerical_safety(**self._kwargs()).outcome, S.PASS)

    def test_any_nan_fails_regardless_of_how_the_audio_sounds(self):
        check = Q.check_numerical_safety(**self._kwargs(nan_count=1))
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("regardless of how the audio sounds", " ".join(check.reasons))

    def test_an_unscanned_buffer_is_could_not_check(self):
        self.assertEqual(
            Q.check_numerical_safety(**self._kwargs(nan_count=None)).outcome,
            S.COULD_NOT_CHECK)

    def test_clipping_outside_the_anchor_band_fails(self):
        self.assertEqual(
            Q.check_numerical_safety(**self._kwargs(clipping_fraction=0.4)).outcome,
            S.FAIL)

    def test_a_bound_with_no_band_is_could_not_check(self):
        self.assertEqual(
            Q.check_numerical_safety(**self._kwargs(clipping_band=None)).outcome,
            S.COULD_NOT_CHECK)


class IntelligibilityTest(unittest.TestCase):
    def _instrument(self, **over):
        base = {"stt_binary_sha256": _sha("whisper-cli"),
                "stt_model_sha256": _sha("large-v3-turbo"),
                "stt_decode_parameters": {"greedy": True},
                "stt_normalizer_id": "stt_norm/v1",
                "stt_normalizer_sha256": _sha("norm"),
                "stt_binary_is_frozen_production": True}
        base.update(over)
        return base

    def test_a_pinned_frozen_production_scorer_passes(self):
        self.assertEqual(
            Q.check_intelligibility_instrument(self._instrument()).outcome, S.PASS)

    def test_the_stt_champion_may_not_be_the_tts_oracle(self):
        check = Q.check_intelligibility_instrument(
            self._instrument(stt_binary_is_frozen_production=False))
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("champion", " ".join(check.reasons))

    def test_an_unrecorded_scorer_makes_a_regression_unattributable(self):
        record = self._instrument()
        del record["stt_normalizer_sha256"]
        check = Q.check_intelligibility_instrument(record)
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("cannot be attributed", " ".join(check.reasons))

    def test_the_floor_is_derived_from_the_anchor_plus_aa_dispersion(self):
        self.assertAlmostEqual(
            Q.derive_intelligibility_floor(anchor_roundtrip_wer_pct=1.49,
                                           aa_dispersion_pp=0.2), 1.69, places=6)

    def test_a_saturated_anchor_is_labelled_so(self):
        # The 2026-07-31 CPU Q8_0 pair measured 0.0 % — word-perfect.
        self.assertEqual(Q.saturation_label(0.0, 0.2), "saturated")
        self.assertEqual(Q.saturation_label(1.49, 0.2), "unsaturated")

    def test_the_floor_alone_is_not_sufficient(self):
        check = Q.check_intelligibility(roundtrip_wer_pct=0.0, floor_pct=0.2,
                                        anchor_roundtrip_wer_pct=0.0,
                                        spectral_distance=None, spectral_band=None)
        self.assertEqual(check.outcome, S.COULD_NOT_CHECK)
        self.assertIn("non-saturated companion", " ".join(check.reasons))

    def test_both_signals_within_band_passes(self):
        self.assertEqual(
            Q.check_intelligibility(roundtrip_wer_pct=0.1, floor_pct=0.2,
                                    anchor_roundtrip_wer_pct=0.0,
                                    spectral_distance=0.4,
                                    spectral_band=[0.0, 0.5]).outcome, S.PASS)

    def test_a_perfect_roundtrip_wer_does_not_rescue_a_drifted_waveform(self):
        # The gaming direction control 3 exists to catch: a flat robotic monotone
        # is perfectly legible and worse speech.
        check = Q.check_intelligibility(roundtrip_wer_pct=0.0, floor_pct=0.2,
                                        anchor_roundtrip_wer_pct=0.0,
                                        spectral_distance=9.0,
                                        spectral_band=[0.0, 0.5])
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("spectral_distance", " ".join(check.reasons))


class StageAndOpCoverageTest(unittest.TestCase):
    def test_all_three_stages_are_required(self):
        check = Q.check_stage_attribution({"talker": 1313.7, "code_predictor": 1091.8},
                                          total_ms=6820.7, tolerance_ms=1.0)
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("codec_decode", " ".join(check.reasons))

    def test_the_2026_07_31_cpu_stage_split_reconciles(self):
        stages = {"talker": 1313.7, "code_predictor": 1091.8, "codec_decode": 4362.9}
        total = sum(stages.values())
        self.assertEqual(
            Q.check_stage_attribution(stages, total_ms=total, tolerance_ms=0.5).outcome,
            S.PASS)

    def test_an_unaccounted_remainder_is_a_finding(self):
        stages = {"talker": 1313.7, "code_predictor": 1091.8, "codec_decode": 4362.9}
        check = Q.check_stage_attribution(stages, total_ms=6820.7, tolerance_ms=0.5)
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("unaccounted", " ".join(check.reasons))

    def test_parts_exceeding_the_whole_fail(self):
        stages = {"talker": 10.0, "code_predictor": 10.0, "codec_decode": 10.0}
        check = Q.check_stage_attribution(stages, total_ms=20.0, tolerance_ms=0.1)
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("exceed the whole", " ".join(check.reasons))

    def test_a_shrinking_op_enumeration_fails_at_one_hundred_percent_pass(self):
        # The gfx90a ARGSORT precedent: 46/46 and 74/74 are both "100 % pass".
        check = Q.check_op_coverage(candidate_attempted=46, candidate_skipped=0,
                                    candidate_passed=46, anchor_attempted=74,
                                    skip_reasons=[])
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("shrinking enumeration", " ".join(check.reasons))

    def test_a_growing_enumeration_passes(self):
        self.assertEqual(
            Q.check_op_coverage(candidate_attempted=74, candidate_skipped=0,
                                candidate_passed=74, anchor_attempted=46,
                                anchor_skipped=0, skip_reasons=[]).outcome, S.PASS)

    def test_an_unexplained_skip_is_could_not_check(self):
        check = Q.check_op_coverage(candidate_attempted=74, candidate_skipped=2,
                                    candidate_passed=72, anchor_attempted=74,
                                    anchor_skipped=2, skip_reasons=None)
        self.assertEqual(check.outcome, S.COULD_NOT_CHECK)

    def test_an_explained_skip_the_anchor_also_took_passes(self):
        self.assertEqual(
            Q.check_op_coverage(candidate_attempted=74, candidate_skipped=2,
                                candidate_passed=72, anchor_attempted=74,
                                anchor_skipped=2,
                                skip_reasons=["unsupported dtype", "shape > 1024"]
                                ).outcome, S.PASS)

    # --- regression: a skip is not a pass -----------------------------------
    # Before the fix these three all returned PASS, because `passed >= attempted -
    # skipped` stays true however many cases move into the skip bucket and nothing
    # compared the candidate's skips with the anchor's.

    def test_skipping_every_case_is_not_a_pass(self):
        # attempted parity with the anchor, every reason reported, ZERO cases passed.
        check = Q.check_op_coverage(candidate_attempted=74, candidate_skipped=74,
                                    candidate_passed=0, anchor_attempted=74,
                                    anchor_skipped=0, skip_reasons=["x"] * 74)
        self.assertEqual(check.outcome, S.FAIL, check.reasons)
        self.assertIn("a skip is not a pass", " ".join(check.reasons))

    def test_a_skip_the_anchor_did_not_take_is_a_coverage_regression(self):
        check = Q.check_op_coverage(candidate_attempted=74, candidate_skipped=30,
                                    candidate_passed=44, anchor_attempted=74,
                                    anchor_skipped=2, skip_reasons=["x"] * 30)
        self.assertEqual(check.outcome, S.FAIL, check.reasons)
        self.assertIn("28 cases RAN on the anchor", " ".join(check.reasons))

    def test_an_unknown_anchor_skip_count_is_could_not_check_never_a_pass(self):
        check = Q.check_op_coverage(candidate_attempted=74, candidate_skipped=0,
                                    candidate_passed=74, anchor_attempted=74,
                                    skip_reasons=[])
        self.assertEqual(check.outcome, S.COULD_NOT_CHECK, check.reasons)
        self.assertIn("anchor's skip count", " ".join(check.reasons))

    def test_passing_more_than_attempted_fails(self):
        self.assertEqual(
            Q.check_op_coverage(candidate_attempted=10, candidate_skipped=0,
                                candidate_passed=11, anchor_attempted=10,
                                skip_reasons=[]).outcome, S.FAIL)


class ReleaseReadinessTest(unittest.TestCase):
    def test_release_is_blocked_while_the_protocol_family_is_a_draft(self):
        check = Q.release_gate_readiness(["P-AK-SEARCH-1"])
        self.assertEqual(check.outcome, S.COULD_NOT_CHECK)
        reasons = " ".join(check.reasons)
        self.assertIn("BLOCKED", reasons)
        self.assertIn("remains legal", reasons)

    def test_p_stt_3_is_a_cross_family_release_dependency(self):
        # TTS stability and op coverage are governed by P-STT-3 rather than
        # duplicated into a P-TTS-4 (where a rule already lives, the amendment goes).
        self.assertIn("P-STT-3", Q.RELEASE_PROTOCOL_IDS)
        check = Q.release_gate_readiness(["P-AK-SEARCH-1", "P-TTS-1", "P-TTS-2",
                                          "P-TTS-3", "P-TTS-REL-1"])
        self.assertEqual(check.outcome, S.COULD_NOT_CHECK)
        self.assertIn("P-STT-3", " ".join(check.reasons))

    def test_it_passes_only_once_every_required_protocol_is_ratified(self):
        self.assertEqual(
            Q.release_gate_readiness(["P-AK-SEARCH-1", *Q.RELEASE_PROTOCOL_IDS]).outcome,
            S.PASS)

    def test_search_itself_is_blocked_without_the_search_protocol(self):
        self.assertEqual(Q.release_gate_readiness([]).outcome, S.COULD_NOT_CHECK)


class SelfAuditTest(unittest.TestCase):
    def test_the_module_cannot_write_or_signal(self):
        source = Path(Q.__file__).read_text(encoding="utf-8")
        check = Q.audit_no_write_or_process_paths(source)
        self.assertEqual(check.outcome, S.PASS, check.reasons)

    def test_no_source_is_could_not_check_not_a_pass(self):
        self.assertEqual(Q.audit_no_write_or_process_paths().outcome, S.COULD_NOT_CHECK)

    def test_a_module_that_wrote_a_file_would_fail_the_audit(self):
        self.assertEqual(
            Q.audit_no_write_or_process_paths("p.write_text('x')\n").outcome, S.FAIL)

    # --- regression: the audit must be ABOUT this module ---------------------
    # Before the fix every one of these returned PASS.

    def test_the_empty_string_does_not_earn_the_no_write_guarantee(self):
        for text in ("", "   \n", "x = 1\n", "def f():\n    return 1\n"):
            check = Q.audit_no_write_or_process_paths(text)
            self.assertEqual(check.outcome, S.COULD_NOT_CHECK, text)
            self.assertIn("identity", " ".join(check.reasons))

    def test_the_sibling_adapters_source_is_not_this_modules_audit(self):
        other = Path(W_MODULE_PATH).read_text(encoding="utf-8")
        self.assertEqual(Q.audit_no_write_or_process_paths(other).outcome,
                         S.COULD_NOT_CHECK)

    def test_a_forbidden_import_still_fails_even_unbound(self):
        self.assertEqual(
            Q.audit_no_write_or_process_paths("import subprocess\n").outcome, S.FAIL)


if __name__ == "__main__":
    unittest.main()
