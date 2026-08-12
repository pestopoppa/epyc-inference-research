#!/usr/bin/env python3
"""Unit tests for `adapters/whisper_stt.py` — the `whisper_stt` backend adapter.

NO inference, NO transcription, NO benchmark, NO build, NO model, NO audio. Nothing
here starts, stops or signals a process, and nothing here writes a file. Every
"verifier report" is a fixture string shaped exactly like
`scripts/utils/verify_ggml_linkage.sh`'s own `printf` output, and every path is a
made-up experimental worktree except where a test is deliberately checking that a
real production path is REFUSED.

The suite is organised around the ways this backend can be got wrong:

  * a binary that resolves another tree's ggml and reports plausible numbers
    (INC-20260731-ggml-linkage-silent-cpu-fallback) — including the two states the
    raw verifier script cannot express: "ldd found nothing" and "the library you care
    about is outside my name filter", both of which exit 0;
  * `use gpu = 1` accepted as evidence that a GPU loaded;
  * a speech number carried with no metric direction, in a project that already
    carries `rtf: 0.169` and `xRT 5.47x` for one engine;
  * a corpus WER computed as a mean of per-utterance rates, or averaging a
    categorical failure into a rate (the Qwen3-ASR repetition-loop precedent);
  * a threshold supplied as a literal where the constitution requires a derivation;
  * a release gate that reports PASS when its caller omits a ratified prerequisite.

Run standalone (no pytest needed):
    python3 -m unittest scripts/kernel_rnd/autokernel/adapters/test_whisper_stt.py
    python3 -W error::ResourceWarning -m unittest \
        scripts/kernel_rnd/autokernel/adapters/test_whisper_stt.py
"""
from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

# Import through the PACKAGE, never by putting this directory on `sys.path`.
_KERNEL_RND = str(Path(__file__).resolve().parents[2])
if _KERNEL_RND not in sys.path:
    sys.path.insert(0, _KERNEL_RND)

from autokernel import schemas as S  # noqa: E402
from autokernel import storage  # noqa: E402
from autokernel.adapters import whisper_stt as W  # noqa: E402
from autokernel.evaluator import correctness, integrity  # noqa: E402
from autokernel.release import plan as P  # noqa: E402

#: The sibling adapter's module path, used only to prove that ITS source does not
#: satisfy THIS module's self-audit. Read as text; nothing here imports it.
Q_MODULE_PATH = str(Path(__file__).resolve().parent / "qwentts_tts.py")

EXP_TREE = "/mnt/raid0/llm/whisper.cpp-experimental"
EXP_BIN = "/mnt/raid0/llm/whisper.cpp-experimental/build/bin/whisper-cli"
EXP_LIB = "/mnt/raid0/llm/whisper.cpp-experimental/build/bin"


def _sha(tag: str) -> str:
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


def _report(lines, *, expect=EXP_LIB, member="whisper-cli",
            trailer="PASS: all linked ggml libraries resolve inside "):
    """Rebuild the verifier's own output shape from `(state, name, path)` rows.

    `member` names the binary the report was CAPTURED AGAINST, and it goes into the
    `binary :` header exactly as `verify_ggml_linkage.sh` prints it. A fixture that
    always said `whisper-cli` was how one member's report got graded as another's.
    """
    out = [f"binary : {expect}/{member}", f"expect : libraries under {expect}", ""]
    for state, name, path in lines:
        out.append("  %s %-28s -> %s" % ("OK  " if state == "OK" else "BAD ", name, path))
    out += ["", "LD_LIBRARY_PATH order as the loader sees it:", f"     1  {expect}", ""]
    out.append(trailer + expect)
    return "\n".join(out) + "\n"


def _diff(path: str, *, added: int, removed: int, hunks: int = 1
          ) -> integrity.SourceDiff:
    """One-file `SourceDiff` fixture. No file is read and no `git` is run."""
    return integrity.SourceDiff(files=(
        integrity.FileDiff(path=path, old_path=None, added_lines=added,
                           removed_lines=removed, hunks=hunks, is_new_file=False,
                           is_deleted_file=False, is_rename=False, is_binary=False,
                           observed_old_extent=max(added + removed, 1) * 10),))


_GOOD_ROWS = [
    ("OK", "libggml-base.so.0.18.0", f"{EXP_LIB}/libggml-base.so.0.18.0"),
    ("OK", "libggml-cpu.so.0.18.0", f"{EXP_LIB}/libggml-cpu.so.0.18.0"),
    ("OK", "libggml.so.0.18.0", f"{EXP_LIB}/libggml.so.0.18.0"),
    ("OK", "libwhisper.so.1.9.1", f"{EXP_LIB}/libwhisper.so.1.9.1"),
]


class TreeIdentityTest(unittest.TestCase):
    """Facts, bound to the ratified operator receipt that froze the tree."""

    def test_identity_matches_the_speech_freeze_receipt(self):
        facts = W.tree_facts()
        self.assertEqual(facts.backend, "whisper_stt")
        self.assertEqual(facts.source_tree, "whisper.cpp")
        self.assertEqual(facts.frozen_branch, "production-speech-v1")
        self.assertEqual(facts.frozen_commit,
                         "b307379226d93d9c5ed790d7cea0626613c0ef4b")
        self.assertEqual(facts.ggml_generation, "0.18.0")

    def test_backend_and_source_tree_agree_with_the_schema_vocabulary(self):
        self.assertIn(W.BACKEND, S.BACKENDS)
        self.assertEqual(S.SOURCE_TREE_BY_BACKEND[W.BACKEND], W.SOURCE_TREE)

    def test_ggml_is_vendored_in_tree_not_as_a_submodule(self):
        # The sibling TTS tree is the opposite case. A shared assumption would be
        # wrong for one of the two, which is why each adapter declares its own.
        self.assertEqual(W.GGML_VENDORING, "in_tree")
        self.assertEqual(W.SUBMODULE_PATHS, ())

    def test_stable_path_resolves_through_bin_unlike_the_tts_sibling(self):
        self.assertTrue(W.STABLE_TARGET.endswith("/build/bin"))
        self.assertEqual(W.LIBRARY_DIR_REL, "build/bin")

    def test_production_tree_mirror_agrees_with_the_two_other_copies(self):
        # Two independent copies of a security boundary is how one of them quietly
        # loses an entry, so the duplication is checked rather than trusted.
        self.assertEqual(set(storage.PRODUCTION_TREES) - set(W.PRODUCTION_TREE_ROOTS),
                         set())
        self.assertEqual(set(correctness.PRODUCTION_TREE_ROOTS),
                         set(W.PRODUCTION_TREE_ROOTS))


class FreezeScopeTest(unittest.TestCase):
    def test_whisper_is_independently_freezable(self):
        scope = W.freeze_scope()
        self.assertTrue(scope.independently_freezable)
        self.assertEqual(scope.backends, ("whisper_stt",))
        self.assertEqual(scope.shares_tree_with, ())

    def test_the_llama_backends_are_not_independently_freezable(self):
        # The property is derived from the shared schema map, so this test also
        # proves the derivation is real rather than a hard-coded True.
        shared = [b for b, t in S.SOURCE_TREE_BY_BACKEND.items() if t == "llama.cpp"]
        self.assertEqual(sorted(shared), ["llama_cpu", "llama_gpu"])

    def test_joining_the_llama_champion_is_refused(self):
        with self.assertRaises(W.WrongReleasePath):
            W.refuse_llama_champion("llama.cpp")
        W.refuse_llama_champion("whisper.cpp")  # its own tree is fine

    def test_the_stack_change_path_is_refused_outright(self):
        with self.assertRaises(W.WrongReleasePath):
            W.refuse_stack_change_path()


class PathDenialTest(unittest.TestCase):
    def test_a_candidate_path_inside_the_production_tree_is_refused(self):
        with self.assertRaises(W.ProductionPathRefused):
            W.binary_path("/mnt/raid0/llm/whisper.cpp", "whisper-cli")

    def test_every_production_tree_is_refused_not_just_this_backends(self):
        for root in W.PRODUCTION_TREE_ROOTS:
            with self.assertRaises(W.ProductionPathRefused):
                W.check_not_production_path(root + "/build/bin/x")

    def test_an_experimental_sibling_directory_is_not_a_production_path(self):
        # `startswith` would call this production and refuse the very tree
        # candidates are supposed to be built in.
        W.check_not_production_path(EXP_BIN)
        self.assertEqual(W.binary_path(EXP_TREE, "whisper-cli"), EXP_BIN)

    def test_the_bin_segment_lives_in_the_inventory_not_in_the_caller(self):
        for spec in W.binary_inventory():
            self.assertTrue(spec.rel_path.startswith("build/bin/"), spec.rel_path)

    def test_an_unknown_binary_is_refused_by_name(self):
        with self.assertRaises(W.UnknownBinary):
            W.binary_path(EXP_TREE, "qwen-tts")

    def test_an_anchor_must_be_the_frozen_production_binary(self):
        W.expect_production_anchor("/mnt/raid0/llm/whisper.cpp/build/bin/whisper-cli")
        with self.assertRaises(W.WhisperAdapterError):
            W.expect_production_anchor(EXP_BIN)

    def test_relative_and_dotdot_paths_are_refused(self):
        with self.assertRaises(W.WhisperAdapterError):
            W.check_not_production_path("build/bin/whisper-cli")
        with self.assertRaises(W.WhisperAdapterError):
            W.check_not_production_path("/mnt/raid0/llm/x/../whisper.cpp/build/bin/y")

    # --- regression: aliases of the frozen tree ------------------------------
    # Before the fix both of these returned None (no refusal at all) while naming
    # a file inside the frozen production tree.

    def test_a_leading_double_slash_does_not_walk_through_the_refusal(self):
        # POSIX leaves `//` implementation-defined and PurePosixPath keeps it as a
        # distinct root, so the segment-wise containment test compares unequal while
        # Linux opens the identical file.
        for root in W.PRODUCTION_TREE_ROOTS:
            with self.assertRaises(W.WhisperAdapterError):
                W.check_not_production_path("/" + root + "/build/bin/x")
        with self.assertRaises(W.WhisperAdapterError):
            W.binary_path("//mnt/raid0/llm/whisper.cpp", "whisper-cli")
        # Three or more slashes collapse to one, so that form was always refused and
        # must stay refused as a ProductionPathRefused rather than a syntax refusal.
        with self.assertRaises(W.ProductionPathRefused):
            W.check_not_production_path("///mnt/raid0/llm/whisper.cpp/build/bin/x")

    def test_the_stable_kernel_symlink_is_refused_as_a_production_path(self):
        # `kernels/README.md`: production/<backend> is "the only path anything should
        # name", and this module declares STABLE_PATH -> STABLE_TARGET itself.
        with self.assertRaises(W.ProductionPathRefused):
            W.check_not_production_path(W.STABLE_PATH + "/whisper-cli")
        for alias in W.PRODUCTION_PATH_ALIASES:
            with self.assertRaises(W.ProductionPathRefused):
                W.check_not_production_path(alias + "/stt/whisper-cli")

    def test_a_sibling_of_the_alias_root_is_not_refused(self):
        # The guard must be segment-wise here too, or it would refuse an unrelated
        # directory that merely shares a prefix.
        W.check_not_production_path("/mnt/raid0/llm/kernels-experimental/stt/x")


class LinkageCommandTest(unittest.TestCase):
    def test_the_environment_is_declared_in_full_never_a_prepend(self):
        inv = W.linkage_command(EXP_BIN, library_path_entries=[EXP_LIB, "/opt/rocm/lib"])
        self.assertEqual(inv.argv, (W.LINKAGE_VERIFIER, EXP_BIN, EXP_LIB))
        self.assertEqual(inv.env, {"LD_LIBRARY_PATH": f"{EXP_LIB}:/opt/rocm/lib"})
        self.assertNotIn("$", inv.env["LD_LIBRARY_PATH"])

    def test_the_verifier_lives_in_the_research_repo_not_epyc_root(self):
        self.assertTrue(W.LINKAGE_VERIFIER.startswith(
            "/mnt/raid0/llm/epyc-inference-research/"))

    def test_an_empty_library_path_is_refused(self):
        with self.assertRaises(W.WhisperAdapterError):
            W.linkage_command(EXP_BIN, library_path_entries=[])

    def test_the_binarys_own_directory_must_be_first(self):
        # This ordering IS the property being verified; accepting any order would
        # verify nothing.
        with self.assertRaises(W.WhisperAdapterError):
            W.linkage_command(EXP_BIN,
                              library_path_entries=["/mnt/raid0/llm/llama.cpp/build/bin",
                                                    EXP_LIB])


class LinkageInterpretationTest(unittest.TestCase):
    def test_a_clean_report_is_pass(self):
        verdict = W.interpret_linkage_report(_report(_GOOD_ROWS), 0, binary="whisper-cli")
        self.assertEqual(verdict.check.outcome, S.PASS)
        self.assertEqual(verdict.bad_libraries, ())
        self.assertEqual(verdict.resolved_count, 4)

    def test_a_bad_line_fails_and_names_the_offender(self):
        rows = list(_GOOD_ROWS)
        rows[2] = ("BAD", "libggml.so.0.18.0",
                   "/mnt/raid0/llm/llama.cpp/build/bin/libggml.so.0")
        verdict = W.interpret_linkage_report(
            _report(rows, trailer="FAIL: 1 library/libraries resolve OUTSIDE "), 1,
            binary="whisper-cli")
        self.assertEqual(verdict.check.outcome, S.FAIL)
        self.assertEqual(len(verdict.bad_libraries), 1)
        self.assertIn("llama.cpp", " ".join(verdict.check.reasons))

    def test_zero_resolved_libraries_is_could_not_check_even_though_the_script_exits_zero(self):
        # THE defect this interpreter exists for: the script prints its
        # "statically linked, or ldd failed" marker and then `exit 0`, so an
        # exit-status consumer records a PASS for a check that did not run.
        text = (f"binary : {EXP_BIN}\nexpect : libraries under {EXP_LIB}\n\n"
                "  (no ggml/whisper/llama libs in ldd output — statically linked, "
                "or ldd failed)\n\nPASS: all linked ggml libraries resolve inside "
                f"{EXP_LIB}\n")
        verdict = W.interpret_linkage_report(text, 0, binary="whisper-cli")
        self.assertEqual(verdict.check.outcome, S.COULD_NOT_CHECK)
        self.assertNotEqual(verdict.check.outcome, S.PASS)
        self.assertFalse(verdict.check.passed)
        self.assertEqual(verdict.resolved_count, 0)

    def test_an_expected_library_absent_from_the_report_is_could_not_check(self):
        # The script's name filter is libggml*/libwhisper*/libllama*/libmtmd*; a
        # library outside it is never examined, and its absence is silence.
        rows = [r for r in _GOOD_ROWS if not r[1].startswith("libwhisper")]
        verdict = W.interpret_linkage_report(_report(rows), 0, binary="whisper-cli")
        self.assertEqual(verdict.check.outcome, S.COULD_NOT_CHECK)
        self.assertIn("libwhisper.so", verdict.missing_expected)
        self.assertIn("name filter", " ".join(verdict.check.reasons))

    def test_a_nonzero_exit_with_no_bad_line_is_a_fail_not_a_pass(self):
        verdict = W.interpret_linkage_report(_report(_GOOD_ROWS), 3, binary="whisper-cli")
        self.assertEqual(verdict.check.outcome, S.FAIL)

    def test_versioned_sonames_are_matched_by_stem(self):
        verdict = W.interpret_linkage_report(_report(_GOOD_ROWS), 0, binary="whisper-cli")
        self.assertEqual(verdict.missing_expected, ())

    def test_a_non_string_report_is_refused_rather_than_coerced(self):
        with self.assertRaises(W.WhisperAdapterError):
            W.interpret_linkage_report(None, 0, binary="whisper-cli")


class PerMemberLibrarySetTest(unittest.TestCase):
    """§10.2 phase 2 was UNRUNNABLE for any member that links a subset.

    One `EXPECTED_SHARED_LIBRARIES` for the whole inventory grades every member by
    the strictest one: a report for a member that never linked `libwhisper.so` is
    missing a library, so the verdict is COULD_NOT_CHECK on every run and the phase
    can never pass for it. On a three-ggml-generation host that is not cosmetic —
    the gate is the one that catches a binary inheriting another tree's ggml.
    """

    def test_a_subset_member_is_gradeable_on_a_report_with_no_engine_library(self):
        # ggml core only — exactly what a tool member that does not link the engine
        # library resolves. Before the fix this was COULD_NOT_CHECK forever.
        rows = [r for r in _GOOD_ROWS if not r[1].startswith("libwhisper")]
        verdict = W.interpret_linkage_report(_report(rows, member="whisper-quantize"), 0,
                                             binary="whisper-quantize")
        self.assertEqual(verdict.check.outcome, S.PASS)
        self.assertEqual(verdict.missing_expected, ())
        self.assertEqual(verdict.binary, "whisper-quantize")
        self.assertNotIn("libwhisper.so", verdict.required_libraries)

    def test_the_same_report_is_still_could_not_check_for_a_member_that_needs_more(self):
        # The relaxation is PER MEMBER, not a global loosening.
        rows = [r for r in _GOOD_ROWS if not r[1].startswith("libwhisper")]
        verdict = W.interpret_linkage_report(_report(rows), 0, binary="whisper-cli")
        self.assertEqual(verdict.check.outcome, S.COULD_NOT_CHECK)
        self.assertIn("libwhisper.so", verdict.missing_expected)

    def test_a_report_cannot_be_graded_without_naming_the_member(self):
        # No default: the union is not reachable as a gate input.
        with self.assertRaises(TypeError):
            W.interpret_linkage_report(_report(_GOOD_ROWS), 0)
        with self.assertRaises(W.UnknownBinary):
            W.interpret_linkage_report(_report(_GOOD_ROWS), 0, binary="qwen-tts")

    def test_every_member_declares_its_own_set_with_provenance(self):
        for spec in W.binary_inventory():
            self.assertTrue(W.CORE_SHARED_LIBRARIES <= spec.required_libraries, spec.name)
            self.assertFalse(spec.required_libraries & spec.optional_libraries, spec.name)
            self.assertTrue(spec.linkage_provenance.strip(), spec.name)
            self.assertEqual(W.expected_shared_libraries(spec.name),
                             spec.required_libraries)

    def test_the_inventory_union_is_a_description_and_never_a_gate_input(self):
        union = W.all_declared_shared_libraries()
        self.assertIn("libwhisper.so", union)
        self.assertIn("libparakeet.so", union)
        # No member is graded against the union: the strictest member's set is a
        # proper subset of it, so grading by the union would import optional
        # libraries into every member's requirement.
        for spec in W.binary_inventory():
            self.assertTrue(spec.required_libraries < union, spec.name)

    def test_a_member_that_requires_nothing_or_drops_the_ggml_core_is_refused(self):
        with self.assertRaises(W.WhisperAdapterError):
            W.BinarySpec("x", "build/bin/x", "op_and_unit_test",
                         required_libraries=frozenset(),
                         optional_libraries=frozenset(), linkage_provenance="test")
        with self.assertRaises(W.WhisperAdapterError):
            W.BinarySpec("x", "build/bin/x", "op_and_unit_test",
                         required_libraries=frozenset({"libwhisper.so"}),
                         optional_libraries=frozenset(), linkage_provenance="test")

    def test_a_library_cannot_be_both_required_and_optional(self):
        with self.assertRaises(W.WhisperAdapterError):
            W.BinarySpec("x", "build/bin/x", "op_and_unit_test",
                         required_libraries=W.CORE_SHARED_LIBRARIES,
                         optional_libraries=frozenset({"libggml.so"}),
                         linkage_provenance="test")

    def test_a_member_set_without_provenance_is_refused(self):
        with self.assertRaises(W.WhisperAdapterError):
            W.BinarySpec("x", "build/bin/x", "op_and_unit_test",
                         required_libraries=W.CORE_SHARED_LIBRARIES,
                         optional_libraries=frozenset(), linkage_provenance="   ")

    # --- COMPLIANT-PATH CONTROL ---------------------------------------------

    def test_the_per_member_rule_does_not_break_the_full_linkage_report(self):
        """The engine members still grade a complete report exactly as before.

        Per-member sets must not turn into "every member requires less": the member
        that DOES link the engine library is still held to it, on the same report
        shape the verifier actually emits.
        """
        for name in ("whisper-cli", "whisper-server", "whisper-bench"):
            verdict = W.interpret_linkage_report(_report(_GOOD_ROWS, member=name), 0,
                                                 binary=name)
            self.assertEqual(verdict.check.outcome, S.PASS, name)
            self.assertIn("libwhisper.so", verdict.required_libraries, name)
            self.assertIn("libwhisper.so", W.expected_shared_libraries(name))


class DeviceEvidenceTest(unittest.TestCase):
    def test_a_real_device_line_is_required_for_a_gpu_cell(self):
        log = "whisper_init: Device 0: AMD Instinct MI210, gfx90a\n"
        self.assertEqual(W.check_device_evidence(log, expected_lane="gpu").outcome, S.PASS)

    def test_use_gpu_equals_one_alone_is_not_device_evidence(self):
        # The exact 2026-07-31 signature: a HIP build printing `use gpu = 1` while
        # running full-CPU against the production ggml.
        check = W.check_device_evidence("whisper_init: use gpu    = 1\n",
                                        expected_lane="gpu")
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("REQUESTED", " ".join(check.reasons))

    def test_a_cpu_cell_that_names_a_device_fails(self):
        log = "whisper_init: Device 0: AMD Instinct MI210\n"
        self.assertEqual(W.check_device_evidence(log, expected_lane="cpu").outcome, S.FAIL)

    def test_a_clean_cpu_log_passes(self):
        self.assertEqual(
            W.check_device_evidence("whisper_init: using 96 threads\n",
                                    expected_lane="cpu").outcome, S.PASS)

    def test_an_empty_log_is_could_not_check(self):
        self.assertEqual(W.check_device_evidence("   \n", expected_lane="gpu").outcome,
                         S.COULD_NOT_CHECK)

    def test_an_undeclared_lane_is_refused(self):
        with self.assertRaises(W.WhisperAdapterError):
            W.check_device_evidence("x", expected_lane="hip")


class DeviceNameVocabularyTest(unittest.TestCase):
    """A device LINE is necessary and not sufficient — the NAME has to denote a GPU.

    The carried-forward defect: `check_device_evidence(expected_lane="gpu")` required
    a `Device N: <name>` line and never asked what `<name>` was, so `Device 0: CPU`
    — precisely what a silently-fallen-back ggml prints — satisfied a GPU cell. That
    is the 2026-07-31 incident surviving the check written to catch it.
    """

    def test_device_zero_cpu_no_longer_satisfies_a_gpu_cell(self):
        check = W.check_device_evidence("whisper_init: Device 0: CPU\n",
                                        expected_lane="gpu")
        self.assertEqual(check.outcome, S.FAIL)
        self.assertNotEqual(check.outcome, S.PASS)
        self.assertIn("host devices", " ".join(check.reasons))

    def test_a_cpu_device_line_beside_a_request_flag_is_the_incident_signature(self):
        log = "whisper_init: use gpu = 1\nwhisper_init: Device 0: CPU\n"
        self.assertEqual(W.check_device_evidence(log, expected_lane="gpu").outcome,
                         S.FAIL)

    def test_a_blas_device_is_a_host_device_not_an_accelerator(self):
        self.assertEqual(
            W.check_device_evidence("Device 0: BLAS\n", expected_lane="gpu").outcome,
            S.FAIL)

    def test_an_unrecognised_device_name_is_could_not_check_never_pass(self):
        # A device the vocabulary cannot name is a device it cannot vouch for.
        check = W.check_device_evidence("Device 0: Some Future Accelerator\n",
                                        expected_lane="gpu")
        self.assertEqual(check.outcome, S.COULD_NOT_CHECK)
        self.assertNotEqual(check.outcome, S.PASS)

    def test_every_device_line_is_read_not_only_the_first(self):
        # ggml enumerates more than one device; grading `search()` alone lets the
        # first line decide a cell the second contradicts, in both directions.
        log = ("ggml_cuda_init: found 1 ROCm devices:\n"
               "  Device 0: CPU\n"
               "  Device 1: AMD Instinct MI210, gfx90a:sramecc+:xnack-\n")
        self.assertEqual(W.check_device_evidence(log, expected_lane="gpu").outcome,
                         S.PASS)
        self.assertEqual(W.check_device_evidence(log, expected_lane="cpu").outcome,
                         S.FAIL)
        self.assertEqual(W.device_names_in_log(log),
                         ("CPU", "AMD Instinct MI210"))

    def test_a_cpu_cell_whose_log_names_the_cpu_device_is_correct_evidence(self):
        self.assertEqual(
            W.check_device_evidence("Device 0: CPU\n", expected_lane="cpu").outcome,
            S.PASS)

    def test_the_vocabulary_is_not_local_to_this_adapter(self):
        # Two copies diverge; the audit proves this one has none of its own.
        source = Path(W.__file__).read_text(encoding="utf-8")
        self.assertEqual(W.audit_device_vocabulary_delegation(source).outcome, S.PASS)

    def test_the_delegation_audit_is_could_not_check_on_empty_and_foreign_source(self):
        self.assertEqual(W.audit_device_vocabulary_delegation("").outcome,
                         S.COULD_NOT_CHECK)
        self.assertEqual(W.audit_device_vocabulary_delegation(None).outcome,
                         S.COULD_NOT_CHECK)
        foreign = Path(Q_MODULE_PATH).read_text(encoding="utf-8")
        self.assertEqual(W.audit_device_vocabulary_delegation(foreign).outcome,
                         S.COULD_NOT_CHECK)

    def test_the_delegation_audit_bites_on_a_local_vocabulary(self):
        doctored = (
            'BACKEND = "whisper_stt"\n'
            'GPU_DEVICE_NAMES = {"AMD Instinct MI210", "gfx90a"}\n'
            "def check_not_production_path(p):\n    return p\n"
            "def interpret_linkage_report(s, e):\n    return s\n"
            "def release_gate_readiness(r):\n    return r\n"
            "def check_device_evidence(log, *, expected_lane):\n"
            "    return devices.check_device_names([log], expected_lane=expected_lane)\n"
        )
        check = W.audit_device_vocabulary_delegation(doctored)
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("MI210", " ".join(check.reasons))

    def test_the_delegation_audit_bites_on_a_checker_that_grades_names_itself(self):
        doctored = (
            'BACKEND = "whisper_stt"\n'
            "def check_not_production_path(p):\n    return p\n"
            "def interpret_linkage_report(s, e):\n    return s\n"
            "def release_gate_readiness(r):\n    return r\n"
            "def check_device_evidence(log, *, expected_lane):\n"
            "    return schemas.Check(schemas.PASS) if 'Device' in log else None\n"
        )
        check = W.audit_device_vocabulary_delegation(doctored)
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("check_device_names", " ".join(check.reasons))

    # --- COMPLIANT-PATH CONTROL ---------------------------------------------

    def test_the_audit_does_not_forbid_this_adapters_own_legitimate_literals(self):
        """Library names and lane keys are not a device vocabulary.

        `libggml-cpu.so` carries the token `cpu` on a word boundary and this adapter
        must keep naming it; the TTS sibling keys `SIBLING_STABLE_TARGETS` by lane
        (`cpu`/`gpu`/`stt`). A guard that banned either would be banning the idiom it
        exists to protect, so both real adapter sources must pass it.
        """
        source = Path(W.__file__).read_text(encoding="utf-8")
        self.assertEqual(W.audit_device_vocabulary_delegation(source).outcome, S.PASS)
        self.assertIn("libggml-cpu.so", W.all_declared_shared_libraries())
        self.assertIn("libggml-cpu.so", W.expected_shared_libraries("whisper-cli"))
        # And a real, complete GPU startup log still passes on the GPU lane.
        log = ("whisper_init: use gpu = 1\n"
               "ggml_cuda_init: found 1 ROCm devices:\n"
               "  Device 0: AMD Instinct MI210, gfx90a:sramecc+:xnack-, VMM: no\n")
        self.assertEqual(W.check_device_evidence(log, expected_lane="gpu").outcome,
                         S.PASS)


class MetricAndPhaseTest(unittest.TestCase):
    def test_a_bare_real_time_factor_is_refused_by_name(self):
        with self.assertRaises(W.UnknownMetric) as ctx:
            W.metric_direction("real_time_factor")
        self.assertIn("reciprocal", str(ctx.exception))

    def test_rtf_and_xrt_carry_opposite_directions(self):
        self.assertEqual(W.metric_direction("rtf"), "lower_better")
        self.assertEqual(W.metric_direction("xrt"), "higher_better")

    def test_every_declared_direction_is_in_the_schema_vocabulary(self):
        for direction in W.METRIC_DIRECTIONS.values():
            self.assertIn(direction, S.METRIC_DIRECTIONS)

    def test_no_declared_metric_is_a_task_rate(self):
        for metric in W.METRIC_DIRECTIONS:
            self.assertEqual(W.check_metric_commensurable(metric).outcome, S.PASS, metric)

    def test_an_undeclared_metric_is_refused(self):
        with self.assertRaises(W.UnknownMetric):
            W.metric_direction("tokens_per_s")

    def test_phases_are_declared_in_the_shared_schema(self):
        self.assertEqual(set(W.PHASES), set(S.PHASES_BY_BACKEND[W.BACKEND]))
        self.assertEqual(W.check_phase("encode"), "encode")
        with self.assertRaises(W.UnknownPhase):
            W.check_phase("prefill")

    def test_the_resource_lane_is_never_the_stack_lane(self):
        self.assertEqual(W.resource_lane(device=None), "cpu")
        self.assertEqual(W.resource_lane(device="ROCm0"), "gpu")
        self.assertIn(W.resource_lane(device="ROCm0"), S.RESOURCE_LANES)


class DomainOwnershipTest(unittest.TestCase):
    def test_an_unowned_domain_fails(self):
        check = W.check_domains_owned(["src", "tools/server"])
        self.assertEqual(check.outcome, S.FAIL)

    def test_no_declared_domain_is_could_not_check_not_a_pass(self):
        self.assertEqual(W.check_domains_owned([]).outcome, S.COULD_NOT_CHECK)

    def test_ggml_counts_as_shared_core_for_this_tree(self):
        self.assertTrue(W.touches_shared_core(["ggml"]))
        self.assertFalse(W.touches_shared_core(["examples"]))

    def test_the_domains_a_diff_reaches_are_read_off_the_diff(self):
        diff = _diff("ggml/src/ggml-cuda/vendors/hip.h", added=1, removed=1)
        self.assertEqual(W.diff_domains(diff), ("ggml",))
        self.assertEqual(W.shared_core_paths(diff),
                         ("ggml/src/ggml-cuda/vendors/hip.h",))

    def test_an_under_declared_domain_list_fails_against_its_own_diff(self):
        diff = _diff("ggml/src/ggml-cuda/vendors/hip.h", added=1, removed=1)
        check = W.check_declared_domains_cover_diff(diff, ["src"])
        self.assertEqual(check.outcome, S.FAIL, check.reasons)
        self.assertIn("ggml", " ".join(check.reasons))
        self.assertEqual(
            W.check_declared_domains_cover_diff(diff, ["ggml"]).outcome, S.PASS)


class ComplexityCeilingTest(unittest.TestCase):
    def test_the_ceiling_is_the_observed_production_history_not_a_round_number(self):
        ceiling = W.complexity_ceiling()
        self.assertEqual(ceiling.max_diff_lines, 2)
        self.assertEqual(ceiling.max_files_touched, 1)
        self.assertTrue(ceiling.shared_core_modification_requires_review)
        self.assertIn("b3073792", ceiling.declared_by)

    def test_the_derivation_is_stated_in_the_declared_by_field(self):
        self.assertIn("production-speech-v1", W.CEILING_DERIVATION)
        self.assertIn("2 changed lines", W.CEILING_DERIVATION)

    def test_a_modest_change_is_already_marked_for_human_review(self):
        # Deliberate: whisper.cpp is a third-party tree this project does not own.
        diff = integrity.SourceDiff(files=(
            integrity.FileDiff(path="ggml/src/ggml-cuda/argsort.cu", old_path=None,
                               added_lines=40, removed_lines=10, hunks=3,
                               is_new_file=False, is_deleted_file=False,
                               is_rename=False, is_binary=False,
                               observed_old_extent=200),))
        assessment = W.assess_complexity(diff, change_class="arithmetic",
                                         domains=["ggml"])
        self.assertTrue(assessment.requires_human_code_review)
        self.assertIn(integrity.REQUIRES_HUMAN_CODE_REVIEW, assessment.first_page_notice)

    def test_the_shared_core_marking_is_traced_from_the_diff_not_declared(self):
        # Regression. This is the frozen tree's OWN production change: 1 file, 2
        # lines, inside `ggml/src/ggml-cuda/` — i.e. inside both size ceilings, so
        # the shared-core clause is the only reason to mark it. Declaring
        # `domains=["src"]` used to remove that reason and the package came out
        # `requires_human_code_review: false` on a shared-core edit.
        diff = _diff("ggml/src/ggml-cuda/vendors/hip.h", added=1, removed=1)
        assessment = W.assess_complexity(diff, change_class="parameter",
                                         domains=["src"])
        self.assertTrue(assessment.requires_human_code_review, assessment.reasons)
        self.assertIn("shared ggml core", " ".join(assessment.reasons))
        self.assertTrue(assessment.measured["touches_shared_core"])

    def test_declaring_shared_core_can_only_add_a_reason_never_remove_one(self):
        # A diff outside shared core, declared as shared core, is still marked: the
        # declared list is OR-ed in, never used to subtract.
        diff = _diff("examples/cli/main.cpp", added=1, removed=1)
        self.assertEqual(W.shared_core_paths(diff), ())
        traced = W.assess_complexity(diff, change_class="parameter",
                                     domains=["examples"])
        self.assertFalse(traced.measured["touches_shared_core"])
        declared = W.assess_complexity(diff, change_class="parameter",
                                       domains=["examples", "ggml"])
        self.assertTrue(declared.measured["touches_shared_core"])

    def test_a_core_header_change_is_marked_regardless_of_size(self):
        diff = integrity.SourceDiff(files=(
            integrity.FileDiff(path="ggml/include/ggml.h", old_path=None,
                               added_lines=1, removed_lines=1, hunks=1,
                               is_new_file=False, is_deleted_file=False,
                               is_rename=False, is_binary=False,
                               observed_old_extent=10),))
        assessment = W.assess_complexity(diff, change_class="core_header",
                                         domains=["include"])
        self.assertTrue(assessment.requires_human_code_review)

    def test_every_envelope_is_a_valid_change_class(self):
        envelopes = W.change_class_envelopes()
        for name, env in envelopes.items():
            self.assertIn(name, S.CHANGE_CLASSES)
            self.assertIs(integrity.envelope_for(envelopes, name), env)
            self.assertLessEqual(env.max_file_shrinkage_ratio, 0.60)

    def test_an_undeclared_change_class_raises_rather_than_defaulting(self):
        with self.assertRaises(integrity.EnvelopeNotDeclared):
            integrity.envelope_for(W.change_class_envelopes(), "recurrent")


class UnchangedTestTest(unittest.TestCase):
    def test_no_transfer_is_available_for_a_single_backend_tree(self):
        plan = W.unchanged_test_plan()
        self.assertFalse(plan.transfer_available)
        self.assertTrue(plan.stage2_required)
        self.assertEqual(plan.traverse_submodules, ())

    def test_a_no_op_candidate_is_refused(self):
        check = W.classify_unchanged_result(stage1_closure_empty=True,
                                            stage2_normalized_identical=True)
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("NO-OP", " ".join(check.reasons))

    def test_a_genuine_change_passes(self):
        self.assertEqual(
            W.classify_unchanged_result(stage1_closure_empty=False,
                                        stage2_normalized_identical=False).outcome,
            S.PASS)

    def test_stage_disagreement_is_a_hard_finding(self):
        check = W.classify_unchanged_result(stage1_closure_empty=True,
                                            stage2_normalized_identical=False)
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("build-identity", " ".join(check.reasons))

    def test_stage_two_not_run_is_could_not_check(self):
        self.assertEqual(
            W.classify_unchanged_result(stage1_closure_empty=False,
                                        stage2_normalized_identical=None).outcome,
            S.COULD_NOT_CHECK)


class NormalizerContractTest(unittest.TestCase):
    def test_the_contracted_pipeline_passes(self):
        check = W.check_normalizer_contract(steps=list(W.NORMALIZATION_STEPS),
                                            transforms_used=["nfkc", "casefold"])
        self.assertEqual(check.outcome, S.PASS)

    def test_order_is_normative_not_merely_membership(self):
        shuffled = list(W.NORMALIZATION_STEPS)
        shuffled[0], shuffled[1] = shuffled[1], shuffled[0]  # casefold before NFKC
        check = W.check_normalizer_contract(steps=shuffled, transforms_used=[])
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("order is normative", " ".join(check.reasons))

    def test_a_forbidden_transform_fails(self):
        for transform in ("stemming", "spell_correction", "fuzzy_token_match",
                          "synonym_mapping", "truncation"):
            check = W.check_normalizer_contract(steps=list(W.NORMALIZATION_STEPS),
                                                transforms_used=[transform])
            self.assertEqual(check.outcome, S.FAIL, transform)

    def test_numeral_normalization_is_part_of_the_pipeline(self):
        # The 2026-07-31 harness did not do it at all, which makes `1920` against
        # `nineteen twenty` a substitution PLUS a deletion for a perfect transcript.
        self.assertIn("numerals_hypothesis_to_reference_form", W.NORMALIZATION_STEPS)

    def test_apostrophes_are_preserved_and_punctuation_becomes_a_separator(self):
        self.assertIn("preserve_apostrophes", W.NORMALIZATION_STEPS)
        self.assertIn("punctuation_to_separator", W.NORMALIZATION_STEPS)

    def test_the_nonlexical_marker_list_is_closed_and_enumerated(self):
        self.assertIn("[BLANK_AUDIO]", W.NONLEXICAL_MARKERS)
        self.assertIsInstance(W.NONLEXICAL_MARKERS, tuple)

    def test_unasserted_normalizer_properties_are_could_not_check(self):
        check = W.check_normalizer_properties(symmetric=True, idempotent=None,
                                              deterministic=True)
        self.assertEqual(check.outcome, S.COULD_NOT_CHECK)

    def test_a_violated_normalizer_property_fails(self):
        check = W.check_normalizer_properties(symmetric=False, idempotent=True,
                                              deterministic=True)
        self.assertEqual(check.outcome, S.FAIL)

    def test_all_three_properties_asserted_passes(self):
        self.assertEqual(
            W.check_normalizer_properties(symmetric=True, idempotent=True,
                                          deterministic=True).outcome, S.PASS)


class AudioIdentityTest(unittest.TestCase):
    def _record(self, **over):
        base = {"utterance_id": "1089-134686-0000", "pcm_sha256": _sha("pcm"),
                "sample_rate_hz": 16000, "channels": 1, "sample_format": "s16le",
                "sample_count": 168_000}
        base.update(over)
        return base

    def test_a_complete_record_passes(self):
        self.assertEqual(W.check_audio_identity(self._record()).outcome, S.PASS)

    def test_a_missing_field_fails(self):
        record = self._record()
        del record["sample_count"]
        self.assertEqual(W.check_audio_identity(record).outcome, S.FAIL)

    def test_a_placeholder_digest_is_refused(self):
        # A fabricated hash is indistinguishable from a measured one downstream.
        self.assertEqual(W.check_audio_identity(self._record(pcm_sha256="0" * 64)).outcome,
                         S.FAIL)

    def test_a_zero_sample_count_fails(self):
        self.assertEqual(W.check_audio_identity(self._record(sample_count=0)).outcome,
                         S.FAIL)

    def test_a_non_mapping_is_could_not_check(self):
        self.assertEqual(W.check_audio_identity(["not", "a", "mapping"]).outcome,
                         S.COULD_NOT_CHECK)

    def test_a_differing_corpus_voids_rather_than_failing_the_candidate(self):
        anchor = {"u1": _sha("a"), "u2": _sha("b")}
        candidate = {"u1": _sha("a"), "u2": _sha("DIFFERENT")}
        check = W.compare_corpus_identity(anchor, candidate)
        self.assertEqual(check.outcome, S.FAIL)
        reasons = " ".join(check.reasons)
        self.assertIn("VOID", reasons)
        self.assertIn("NOT a candidate correctness failure", reasons)

    def test_an_identical_corpus_passes(self):
        manifest = {"u1": _sha("a")}
        self.assertEqual(W.compare_corpus_identity(manifest, dict(manifest)).outcome,
                         S.PASS)

    def test_the_mismatch_evidence_states_its_own_truncation(self):
        # Regression: the reason listed `missing[:8]` with no count, so 800 missing
        # utterances and 8 missing utterances rendered as the same-looking list.
        anchor = {f"u{i}": _sha(str(i)) for i in range(50)}
        candidate = {"u0": _sha("0")}
        check = W.compare_corpus_identity(anchor, candidate)
        self.assertEqual(check.outcome, S.FAIL)
        reasons = " ".join(check.reasons)
        self.assertIn("49 missing", reasons)
        self.assertIn("first 8 of 49", reasons)


class PooledWerTest(unittest.TestCase):
    def test_it_reproduces_the_2026_07_31_arms_from_their_own_counts(self):
        # Descriptive, not a threshold: 44/1870 and 63/1870 are the two arms in
        # `/mnt/raid0/llm/tmp/stt_wer_results.json`. The receipt's `wer_pct: 2.35`
        # is the FIRST of these — the faster-whisper CPU incumbent — while
        # whisper.cpp on the MI210 measured 3.37 %.
        self.assertAlmostEqual(W.pooled_corpus_wer([44], [1870]), 2.3529, places=3)
        self.assertAlmostEqual(W.pooled_corpus_wer([63], [1870]), 3.3690, places=3)

    def test_it_is_pooled_and_not_the_mean_of_per_utterance_rates(self):
        # 1 error in 1 word plus 0 errors in 99 words: pooled 1 %, mean-of-ratios
        # 50 %. This module offers no function computing the latter.
        self.assertAlmostEqual(W.pooled_corpus_wer([1, 0], [1, 99]), 1.0, places=6)
        self.assertFalse(any("mean" in name for name in dir(W)))

    def test_an_empty_corpus_raises_rather_than_returning_zero(self):
        with self.assertRaises(W.DerivationImpossible):
            W.pooled_corpus_wer([], [])

    def test_mismatched_lengths_raise_rather_than_dropping_utterances(self):
        with self.assertRaises(W.WhisperAdapterError):
            W.pooled_corpus_wer([1, 2], [10])

    def test_an_utterance_with_no_reference_tokens_raises(self):
        with self.assertRaises(W.WhisperAdapterError):
            W.pooled_corpus_wer([1], [0])


class DerivationTest(unittest.TestCase):
    def test_the_repetition_envelope_is_the_anchors_own_observed_maximum(self):
        self.assertEqual(W.derive_repetition_envelope([0.9, 1.4, 1.1]), 1.4)

    def test_the_repetition_envelope_cannot_be_defaulted(self):
        with self.assertRaises(W.DerivationImpossible):
            W.derive_repetition_envelope([])

    def test_corpus_size_scales_as_n_to_the_minus_one_half(self):
        # 2026-07-31 precedent: paired half-width 0.67 pp at n=100.
        self.assertEqual(W.derive_corpus_size(observed_halfwidth_pp=0.67,
                                              observed_n=100,
                                              contribution_floor_pp=0.30), 499)
        self.assertEqual(W.derive_corpus_size(observed_halfwidth_pp=0.67,
                                              observed_n=100,
                                              contribution_floor_pp=0.10), 4489)

    def test_a_0_1pp_floor_exceeds_all_of_librispeech_test_clean(self):
        # 2620 utterances exist. The campaign must confront that at calibration
        # time rather than after spending its budget.
        required = W.derive_corpus_size(observed_halfwidth_pp=0.67, observed_n=100,
                                        contribution_floor_pp=0.10)
        self.assertGreater(required, 2620)

    def test_a_zero_or_negative_floor_is_refused(self):
        for floor in (0.0, -1.0):
            with self.assertRaises(W.WhisperAdapterError):
                W.derive_corpus_size(observed_halfwidth_pp=0.67, observed_n=100,
                                     contribution_floor_pp=floor)

    def test_a_bitwise_stable_instrument_has_a_zero_correctness_margin(self):
        margin = W.derive_correctness_margin(aa_noise_floor=0.0, contribution_floor=0.3,
                                             determinism_class="bitwise_stable")
        self.assertEqual(margin, 0.0)

    def test_a_bitwise_stable_claim_with_a_nonzero_floor_is_a_hard_finding(self):
        with self.assertRaises(W.WhisperAdapterError):
            W.derive_correctness_margin(aa_noise_floor=0.2, contribution_floor=0.3,
                                        determinism_class="bitwise_stable")

    def test_an_unmeasured_determinism_class_cannot_derive_a_margin(self):
        with self.assertRaises(W.DerivationImpossible):
            W.derive_correctness_margin(aa_noise_floor=0.0, contribution_floor=0.3,
                                        determinism_class="not_measured")

    def test_an_unstable_instrument_takes_the_larger_of_floor_and_noise(self):
        self.assertEqual(
            W.derive_correctness_margin(aa_noise_floor=0.5, contribution_floor=0.3,
                                        determinism_class="bitwise_unstable"), 0.5)
        self.assertEqual(
            W.derive_correctness_margin(aa_noise_floor=0.1, contribution_floor=0.3,
                                        determinism_class="bitwise_unstable"), 0.3)


class FailureTaxonomyTest(unittest.TestCase):
    def _counts(self, **over):
        base = {c: 0 for c in W.FAILURE_CLASSES}
        base["ok"] = 100
        base.update(over)
        return base

    def test_a_clean_corpus_passes(self):
        self.assertEqual(W.check_failure_taxonomy(self._counts(), n_utterances=100).outcome,
                         S.PASS)

    def test_a_repetition_loop_is_a_categorical_failure_not_a_wer_contribution(self):
        # The Qwen3-ASR precedent: 21 of 100 rows carried 94.7 % of all errors, and
        # the clean rows were at 2.27 %. Averaging that into a rate reports a
        # uniformly mediocre model where the truth is a different production risk.
        counts = self._counts(ok=79, repetition_loop=21)
        check = W.check_failure_taxonomy(counts, n_utterances=100)
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("no speed rank at all", " ".join(check.reasons))

    def test_every_utterance_must_receive_exactly_one_class(self):
        self.assertEqual(
            W.check_failure_taxonomy(self._counts(ok=99), n_utterances=100).outcome,
            S.FAIL)

    def test_an_omitted_class_is_could_not_check_not_a_zero(self):
        counts = self._counts()
        del counts["unknown_marker"]
        self.assertEqual(W.check_failure_taxonomy(counts, n_utterances=100).outcome,
                         S.COULD_NOT_CHECK)

    def test_a_class_outside_the_vocabulary_fails(self):
        counts = self._counts(ok=99)
        counts["weird"] = 1
        self.assertEqual(W.check_failure_taxonomy(counts, n_utterances=100).outcome,
                         S.FAIL)

    def test_excluded_classes_do_not_fail_the_candidate(self):
        counts = self._counts(ok=98, numeral_uncovered=1, unknown_marker=1)
        self.assertEqual(W.check_failure_taxonomy(counts, n_utterances=100).outcome,
                         S.PASS)

    def test_the_exclusion_cap_is_derived_from_the_anchor_plus_aa_dispersion(self):
        two_percent = W.ExclusionRateDispersion.from_fraction(0.02)
        ok = W.check_exclusion_rate(candidate_excluded=3, anchor_excluded=2,
                                    n_utterances=100,
                                    aa_dispersion_fraction=two_percent)
        self.assertEqual(ok.outcome, S.PASS)
        bad = W.check_exclusion_rate(candidate_excluded=9, anchor_excluded=2,
                                     n_utterances=100,
                                     aa_dispersion_fraction=two_percent)
        self.assertEqual(bad.outcome, S.FAIL)
        self.assertIn("derived cap", " ".join(bad.reasons))

    def test_a_missing_aa_dispersion_is_could_not_check(self):
        self.assertEqual(
            W.check_exclusion_rate(candidate_excluded=1, anchor_excluded=0,
                                   n_utterances=100,
                                   aa_dispersion_fraction=None).outcome,
            S.COULD_NOT_CHECK)


class ExclusionDispersionUnitTest(unittest.TestCase):
    """The unit hazard: `aa_dispersion=0.02` meant two things and said neither.

    An exclusion rate is a fraction of 1 (a count over a count). An A/A dispersion of
    a rate is routinely quoted in percentage POINTS on this project — the TTS
    sibling's `derive_intelligibility_floor` takes `aa_dispersion_pp`, and
    `stt_wer_results.json` quotes WER in percent. `0.02` is legal in both units and
    the two readings of it differ 50-fold, in the direction that WIDENS the cap.
    """

    def test_a_bare_number_is_refused_because_its_unit_is_unknowable(self):
        # Before the fix this was accepted and silently read as a fraction.
        with self.assertRaises(W.UnitAmbiguity) as ctx:
            W.check_exclusion_rate(candidate_excluded=3, anchor_excluded=2,
                                   n_utterances=100, aa_dispersion_fraction=0.02)
        message = str(ctx.exception)
        self.assertIn("percentage points", message)
        self.assertIn("from_fraction", message)

    def test_the_two_units_are_different_caps_and_both_are_expressible(self):
        # 2 percentage points IS 0.02 as a fraction; 0.02 percentage points is not.
        self.assertAlmostEqual(
            W.ExclusionRateDispersion.from_percentage_points(2.0).fraction_of_1, 0.02)
        self.assertAlmostEqual(
            W.ExclusionRateDispersion.from_percentage_points(0.02).fraction_of_1, 0.0002)
        # The same candidate passes under one reading and fails under the other, which
        # is exactly why the bare number could not be interpreted.
        wide = W.check_exclusion_rate(candidate_excluded=4, anchor_excluded=2,
                                      n_utterances=100,
                                      aa_dispersion_fraction=
                                      W.ExclusionRateDispersion.from_percentage_points(2.0))
        narrow = W.check_exclusion_rate(candidate_excluded=4, anchor_excluded=2,
                                        n_utterances=100,
                                        aa_dispersion_fraction=
                                        W.ExclusionRateDispersion.from_percentage_points(0.02))
        self.assertEqual(wide.outcome, S.PASS)
        self.assertEqual(narrow.outcome, S.FAIL)

    def test_the_unit_travels_with_the_value_into_the_reason(self):
        check = W.check_exclusion_rate(
            candidate_excluded=9, anchor_excluded=2, n_utterances=100,
            aa_dispersion_fraction=W.ExclusionRateDispersion.from_percentage_points(2.0))
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("percentage_points", " ".join(check.reasons))

    def test_a_percentage_point_value_typed_into_the_fraction_field_is_refused(self):
        # The honest mistake, caught by the field's own range: 2.0 is not a fraction.
        with self.assertRaises(W.WhisperAdapterError):
            W.ExclusionRateDispersion.from_fraction(2.0)
        with self.assertRaises(W.WhisperAdapterError):
            W.ExclusionRateDispersion(fraction_of_1=2.0, source_unit="fraction_of_1")

    def test_an_undeclared_source_unit_is_refused(self):
        with self.assertRaises(W.UnitAmbiguity):
            W.ExclusionRateDispersion(fraction_of_1=0.02, source_unit="percent")

    def test_a_negative_or_non_numeric_dispersion_is_refused(self):
        with self.assertRaises(W.WhisperAdapterError):
            W.ExclusionRateDispersion.from_fraction(-0.01)
        with self.assertRaises(W.WhisperAdapterError):
            W.ExclusionRateDispersion.from_fraction("0.02")
        with self.assertRaises(W.WhisperAdapterError):
            W.ExclusionRateDispersion.from_fraction(True)

    # --- COMPLIANT-PATH CONTROL ---------------------------------------------

    def test_the_guard_does_not_forbid_a_dispersion_that_really_is_a_fraction(self):
        """A typed fraction — the idiom the guard exists to require — still works.

        A refusal that also refused the compliant call would be worked around rather
        than obeyed. Both constructors reach a verdict, and a zero dispersion (a
        bitwise-stable instrument) is a legal cap, not a refusal.

        `from_fraction(1.0)` was in this list and has been removed: a dispersion of
        100 % puts the cap at or above 1, which no exclusion rate can exceed, so the
        "compliant path" it asserted was a comparison no candidate could ever fail.
        Its verdict is now COULD_NOT_CHECK and it has its own test below.
        """
        for dispersion in (W.ExclusionRateDispersion.from_fraction(0.0),
                           W.ExclusionRateDispersion.from_fraction(0.02),
                           W.ExclusionRateDispersion.from_percentage_points(0.0),
                           W.ExclusionRateDispersion.from_percentage_points(2.0),
                           W.ExclusionRateDispersion.from_fraction(0.5)):
            check = W.check_exclusion_rate(candidate_excluded=2, anchor_excluded=2,
                                           n_utterances=100,
                                           aa_dispersion_fraction=dispersion)
            self.assertEqual(check.outcome, S.PASS, dispersion)
        self.assertEqual(
            W.ExclusionRateDispersion.from_fraction(0.02).to_dict(),
            {"fraction_of_1": 0.02, "source_unit": "fraction_of_1"})


class ReleaseReadinessTest(unittest.TestCase):
    def test_release_is_blocked_when_the_supplied_registry_is_incomplete(self):
        check = W.release_gate_readiness(["P-AK-SEARCH-1", "P-BENCH-1", "P-GPU-1"])
        self.assertEqual(check.outcome, S.COULD_NOT_CHECK)
        self.assertNotEqual(check.outcome, S.PASS)
        reasons = " ".join(check.reasons)
        self.assertIn("BLOCKED", reasons)
        self.assertIn("Search under P-AK-SEARCH-1 remains legal", reasons)

    def test_search_itself_is_blocked_without_the_search_protocol(self):
        check = W.release_gate_readiness(["P-BENCH-1"])
        self.assertEqual(check.outcome, S.COULD_NOT_CHECK)
        self.assertIn("not even T0-T2 search", " ".join(check.reasons))

    def test_it_passes_only_once_every_required_protocol_is_ratified(self):
        ratified = ["P-AK-SEARCH-1", *W.RELEASE_PROTOCOL_IDS]
        self.assertEqual(W.release_gate_readiness(ratified).outcome, S.PASS)

    def test_the_ratified_set_is_supplied_never_baked_in(self):
        # A constant here would go stale silently the moment the operator ratified
        # — or declined — the family.
        with self.assertRaises(TypeError):
            W.release_gate_readiness()  # noqa: E1120 - deliberate arity check

    def test_the_protocol_locator_points_at_ratified_annex_s(self):
        self.assertEqual(W.RELEASE_PROTOCOL_LOCATOR,
                         "measurement/protocols/speech.md")

    def test_pure_binding_has_exact_tree_linkage_and_ratified_prerequisite(self):
        protocols = {
            phase: P.PhaseProtocol(phase=phase, protocol_id="P-STT-2",
                                   metric="rtf", direction="lower_better")
            for phase in W.PHASES
        }
        binding = W.release_binding(
            protocols=protocols,
            ratified_protocol_ids=["P-AK-SEARCH-1", *W.RELEASE_PROTOCOL_IDS])
        self.assertEqual(binding.production_tree_path, W.PRODUCTION_TREE_ROOT)
        self.assertEqual(binding.binary_roots, (W.STABLE_PATH, W.STABLE_TARGET))
        self.assertEqual(binding.linkage.ggml_generation, W.GGML_GENERATION)
        self.assertEqual(binding.linkage.required_ld_library_path, (W.STABLE_TARGET,))
        self.assertEqual(binding.prerequisites["ratified_protocol_registry"].outcome,
                         S.PASS)
        # No adapter invents calibration thresholds.
        self.assertTrue(all(not protocol.thresholds
                            for protocol in binding.protocols.values()))

    def test_binding_keeps_missing_ratification_fail_closed(self):
        binding = W.release_binding(protocols={}, ratified_protocol_ids=[])
        self.assertEqual(binding.prerequisites["ratified_protocol_registry"].outcome,
                         S.COULD_NOT_CHECK)


class SelfAuditTest(unittest.TestCase):
    def test_the_module_cannot_write_or_signal(self):
        source = Path(W.__file__).read_text(encoding="utf-8")
        check = W.audit_no_write_or_process_paths(source)
        self.assertEqual(check.outcome, S.PASS, check.reasons)

    def test_no_source_is_could_not_check_not_a_pass(self):
        self.assertEqual(W.audit_no_write_or_process_paths().outcome, S.COULD_NOT_CHECK)

    def test_unparseable_source_is_could_not_check(self):
        self.assertEqual(W.audit_no_write_or_process_paths("def (").outcome,
                         S.COULD_NOT_CHECK)

    def test_a_module_that_launched_a_process_would_fail_the_audit(self):
        self.assertEqual(
            W.audit_no_write_or_process_paths("import subprocess\n").outcome, S.FAIL)

    # --- regression: the audit must be ABOUT this module ---------------------
    # Before the fix every one of these returned PASS: the guarantee was obtainable
    # by deleting the thing the check inspects.

    def test_the_empty_string_does_not_earn_the_no_write_guarantee(self):
        for text in ("", "   \n", "x = 1\n", "def f():\n    return 1\n"):
            check = W.audit_no_write_or_process_paths(text)
            self.assertEqual(check.outcome, S.COULD_NOT_CHECK, text)
            self.assertIn("identity", " ".join(check.reasons))

    def test_another_adapters_source_is_not_this_modules_audit(self):
        other = Path(Q_MODULE_PATH).read_text(encoding="utf-8")
        self.assertEqual(W.audit_no_write_or_process_paths(other).outcome,
                         S.COULD_NOT_CHECK)

    def test_a_forbidden_construct_still_fails_even_unbound(self):
        # A FAIL is a finding about the text whoever the text belongs to, so the
        # binding must not convert a positive detection into COULD_NOT_CHECK.
        self.assertEqual(
            W.audit_no_write_or_process_paths("p.write_text('x')\n").outcome, S.FAIL)


class LinkageReportBindingTest(unittest.TestCase):
    """RED TEAM: per-member sets made the member's identity load-bearing, and the
    member was named by the CALLER with nothing tying it to the report.

    `verify_ggml_linkage.sh` prints `binary : $BIN` as its first line, so the report
    states what it is. Grading was done against `binary=`, the claim of the party
    being gated. On identical bytes:

        whisper-cli's own report, missing libwhisper.so
          graded as whisper-cli      -> COULD_NOT_CHECK   (the engine member is held
                                                           to its engine library)
          graded as whisper-quantize -> PASS              (a tool member is not)

    A wrongly-linked or half-captured engine report was therefore one keyword away
    from a clean §10.2 phase-2 verdict.
    """

    def _core_only(self):
        return [r for r in _GOOD_ROWS if not r[1].startswith("libwhisper")]

    def test_a_report_cannot_be_graded_as_a_member_it_was_not_captured_against(self):
        report = _report(self._core_only(), member="whisper-cli")
        self.assertEqual(
            W.interpret_linkage_report(report, 0, binary="whisper-cli").check.outcome,
            S.COULD_NOT_CHECK)
        verdict = W.interpret_linkage_report(report, 0, binary="whisper-quantize")
        self.assertEqual(verdict.check.outcome, S.COULD_NOT_CHECK)
        self.assertNotEqual(verdict.check.outcome, S.PASS)
        self.assertFalse(verdict.check.passed)
        reasons = " ".join(verdict.check.reasons)
        self.assertIn("whisper-cli", reasons)
        self.assertIn("whisper-quantize", reasons)

    def test_the_relabelling_cannot_manufacture_a_pass_for_any_other_member(self):
        report = _report(_GOOD_ROWS, member="whisper-cli")
        for name in ("whisper-server", "whisper-bench", "whisper-quantize",
                     "test-vad", "test-vad-full"):
            verdict = W.interpret_linkage_report(report, 0, binary=name)
            self.assertEqual(verdict.check.outcome, S.COULD_NOT_CHECK, name)

    def test_a_report_that_names_no_binary_cannot_be_bound_to_a_member(self):
        # Deleting the header is the other way to make grading float free of the
        # evidence, so absence is COULD_NOT_CHECK too — never PASS.
        stripped = "\n".join(line for line in _report(_GOOD_ROWS).splitlines()
                             if not line.startswith("binary :")) + "\n"
        self.assertIsNone(W.report_binary_name(stripped))
        verdict = W.interpret_linkage_report(stripped, 0, binary="whisper-cli")
        self.assertEqual(verdict.check.outcome, S.COULD_NOT_CHECK)
        self.assertIn("no `binary : <path>` header", " ".join(verdict.check.reasons))

    def test_a_wrong_tree_resolution_is_still_a_fail_whoever_the_report_belongs_to(self):
        # A BAD line is a finding about the text, and stays FAIL rather than being
        # softened into "not my member" — the fail-closed direction.
        rows = list(_GOOD_ROWS)
        rows[2] = ("BAD", "libggml.so.0.18.0",
                   "/mnt/raid0/llm/llama.cpp/build/bin/libggml.so.0")
        verdict = W.interpret_linkage_report(
            _report(rows, member="whisper-cli",
                    trailer="FAIL: 1 library/libraries resolve OUTSIDE "), 1,
            binary="whisper-quantize")
        self.assertEqual(verdict.check.outcome, S.FAIL)

    # --- COMPLIANT-PATH CONTROL ---------------------------------------------

    def test_every_member_still_grades_its_own_report_exactly_as_before(self):
        """The binding must not make any member ungradeable.

        Each member, graded on a report captured against ITSELF — the only thing a
        runner can actually produce — reaches the same verdict it did before the
        binding existed: engine members PASS the full report, tool members PASS the
        ggml-core-only report, and the engine members are still held to
        `libwhisper.so`.
        """
        for name in ("whisper-cli", "whisper-server", "whisper-bench"):
            verdict = W.interpret_linkage_report(_report(_GOOD_ROWS, member=name), 0,
                                                 binary=name)
            self.assertEqual(verdict.check.outcome, S.PASS, name)
        for name in ("whisper-quantize", "test-vad", "test-vad-full"):
            verdict = W.interpret_linkage_report(
                _report(self._core_only(), member=name), 0, binary=name)
            self.assertEqual(verdict.check.outcome, S.PASS, name)
        held = W.interpret_linkage_report(
            _report(self._core_only(), member="whisper-cli"), 0, binary="whisper-cli")
        self.assertEqual(held.check.outcome, S.COULD_NOT_CHECK)
        self.assertIn("libwhisper.so", held.missing_expected)

    def test_the_header_is_read_the_way_the_script_prints_it(self):
        # `echo "binary : $BIN"` — an absolute path, and the member is its basename.
        self.assertEqual(
            W.report_binary_name(f"binary : {EXP_LIB}/whisper-cli\nexpect : x\n"),
            "whisper-cli")
        self.assertEqual(W.report_binary_name("binary : whisper-cli\n"), "whisper-cli")
        self.assertEqual(W.report_binary_name("binary :   /a/b/test-vad-full  \n"),
                         "test-vad-full")


class ExclusionCapVacuityTest(unittest.TestCase):
    """RED TEAM: naming the unit fixed which number was meant, not how wide it is.

    `check_exclusion_rate` derives `cap = anchor_rate + dispersion` and FAILs a
    candidate above it. An exclusion rate is a count over a count, so it cannot
    exceed 1 — and a legally-typed `ExclusionRateDispersion.from_fraction(1.0)` puts
    the cap at 1.0, where a candidate that dropped EVERY utterance in the corpus
    passes. That is the same defect the TTS sibling's stage tolerance was bounded
    against in this very package: a slack at or above the quantity it bounds is not a
    loose check, it is no check.
    """

    def test_a_cap_no_candidate_could_exceed_is_not_a_pass(self):
        for dispersion in (W.ExclusionRateDispersion.from_fraction(1.0),
                           W.ExclusionRateDispersion.from_percentage_points(100.0),
                           W.ExclusionRateDispersion.from_fraction(0.99)):
            check = W.check_exclusion_rate(candidate_excluded=100, anchor_excluded=1,
                                           n_utterances=100,
                                           aa_dispersion_fraction=dispersion)
            self.assertEqual(check.outcome, S.COULD_NOT_CHECK, dispersion)
            self.assertNotEqual(check.outcome, S.PASS, dispersion)
            self.assertFalse(check.passed, dispersion)
            self.assertIn("vacuous", " ".join(check.reasons))

    def test_the_anchors_own_rate_counts_toward_the_vacuity_bound(self):
        # cap = 0.90 + 0.10 = 1.00. The dispersion alone looks modest; the cap does
        # not, and the cap is what grades.
        check = W.check_exclusion_rate(
            candidate_excluded=100, anchor_excluded=90, n_utterances=100,
            aa_dispersion_fraction=W.ExclusionRateDispersion.from_fraction(0.10))
        self.assertEqual(check.outcome, S.COULD_NOT_CHECK)

    # --- COMPLIANT-PATH CONTROL ---------------------------------------------

    def test_a_real_dispersion_still_reaches_both_verdicts(self):
        """The bound must not turn a working check into a permanent refusal.

        A realistic A/A dispersion still PASSes a candidate inside the cap and FAILs
        one outside it — the check keeps discriminating, which is the property the
        bound exists to protect.
        """
        dispersion = W.ExclusionRateDispersion.from_fraction(0.02)
        self.assertEqual(
            W.check_exclusion_rate(candidate_excluded=3, anchor_excluded=2,
                                   n_utterances=100,
                                   aa_dispersion_fraction=dispersion).outcome, S.PASS)
        self.assertEqual(
            W.check_exclusion_rate(candidate_excluded=9, anchor_excluded=2,
                                   n_utterances=100,
                                   aa_dispersion_fraction=dispersion).outcome, S.FAIL)
        # …and a cap just under 1 on a corpus where it can still be exceeded.
        self.assertEqual(
            W.check_exclusion_rate(
                candidate_excluded=1000, anchor_excluded=0, n_utterances=1000,
                aa_dispersion_fraction=W.ExclusionRateDispersion.from_fraction(0.999)
            ).outcome, S.FAIL)


if __name__ == "__main__":
    unittest.main()
