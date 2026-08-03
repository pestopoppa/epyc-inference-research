#!/usr/bin/env python3
"""Unit tests for `evaluator/devices.py` — the device-NAME vocabulary.

NO inference, NO process, NO device is opened, NO file is written. Every "startup
log" is a fixture string shaped like the lines ggml actually prints.

The suite is organised around the defect the module closes and the ways a
vocabulary can be got wrong:

  * `Device 0: CPU` accepted as evidence that a GPU loaded — the 2026-07-31 silent
    CPU fallback surviving the check written to catch it;
  * a vendor word (`AMD`) shared by the MI210 and the EPYC 9655, so a token table
    that used one would classify every device name as both;
  * a name the table does not know, resolved by guessing instead of by saying so;
  * an audit that can be satisfied by handing it a different string, or that forbids
    its own consumers' legitimate literals (`libggml-cpu.so`, the lane key `cpu`).

Run standalone (no pytest needed):
    python3 -m unittest scripts/kernel_rnd/autokernel/evaluator/test_devices.py
"""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

_KERNEL_RND = str(Path(__file__).resolve().parents[2])
if _KERNEL_RND not in sys.path:
    sys.path.insert(0, _KERNEL_RND)

from autokernel import schemas as S  # noqa: E402
from autokernel.evaluator import api  # noqa: E402
from autokernel.evaluator import devices as D  # noqa: E402

_ADAPTERS = Path(__file__).resolve().parents[1] / "adapters"

#: A conforming consumer, minimal but complete: it declares an identity and its
#: device checker delegates. Used as the audit's COMPLIANT control alongside the two
#: real adapters.
_CONFORMING_CONSUMER = (
    'BACKEND = "whisper_stt"\n'
    'EXPECTED = frozenset({"libggml-cpu.so", "libggml.so"})\n'
    'LANES = {"cpu": "/a", "gpu": "/b"}\n'
    "def check_not_production_path(p):\n    return p\n"
    "def interpret_linkage_report(s, e):\n    return s\n"
    "def release_gate_readiness(r):\n    return r\n"
    "def check_device_evidence(log, *, expected_lane):\n"
    "    return devices.check_device_names([log], expected_lane=expected_lane)\n"
)

_IDENTITY = ("check_not_production_path", "interpret_linkage_report",
             "release_gate_readiness")


class ClassificationTest(unittest.TestCase):
    def test_the_hosts_accelerator_classifies_as_a_gpu(self):
        for name in ("AMD Instinct MI210", "MI210", "gfx90a",
                     "AMD Instinct MI210 (gfx90a)"):
            self.assertEqual(D.classify_device_name(name).device_class, D.GPU, name)

    def test_the_ggml_cpu_backends_device_classifies_as_a_host_device(self):
        for name in ("CPU", "cpu", "BLAS", "AMD EPYC 9655 96-Core Processor"):
            self.assertEqual(D.classify_device_name(name).device_class, D.CPU, name)

    def test_an_unknown_name_is_unknown_not_guessed(self):
        for name in ("Some Future Accelerator", "", "   ", "device 0"):
            self.assertEqual(D.classify_device_name(name).device_class, D.UNKNOWN, name)

    def test_a_name_matching_both_classes_is_ambiguous_never_resolved_by_precedence(self):
        verdict = D.classify_device_name("MI210 CPU shim")
        self.assertEqual(verdict.device_class, D.AMBIGUOUS)
        self.assertFalse(verdict.is_gpu)
        self.assertFalse(verdict.is_cpu)

    def test_the_shared_vendor_word_is_not_a_token(self):
        # `AMD Instinct MI210` and `AMD EPYC 9655` share it; a vendor token would
        # make every device name on this host ambiguous.
        self.assertEqual(D.classify_device_name("AMD").device_class, D.UNKNOWN)
        self.assertNotIn("amd", [e.token for e in D.device_vocabulary()])

    def test_matching_is_on_word_boundaries_not_substrings(self):
        # `MI2100` is not an MI210, and `epycx` is not an EPYC.
        self.assertEqual(D.classify_device_name("MI2100").device_class, D.UNKNOWN)
        self.assertEqual(D.classify_device_name("epycx").device_class, D.UNKNOWN)
        self.assertEqual(D.classify_device_name("gfx90a:sramecc+:xnack-").device_class,
                         D.GPU)

    def test_matching_is_case_insensitive(self):
        self.assertEqual(D.classify_device_name("amd instinct mi210").device_class,
                         D.GPU)
        self.assertEqual(D.classify_device_name("Cpu").device_class, D.CPU)

    def test_a_non_string_name_is_refused_rather_than_coerced(self):
        with self.assertRaises(D.DeviceVocabularyError):
            D.classify_device_name(None)
        with self.assertRaises(D.DeviceVocabularyError):
            D.classify_device_name(0)

    def test_the_verdict_names_which_entries_matched(self):
        verdict = D.classify_device_name("AMD Instinct MI210, gfx90a")
        self.assertEqual(set(verdict.matched_tokens), {"instinct", "mi210", "gfx90a"})
        self.assertEqual(verdict.to_dict()["device_class"], D.GPU)


class LaneGateTest(unittest.TestCase):
    """The defect: a device LINE was accepted as proof an accelerator loaded."""

    def test_a_cpu_device_does_not_satisfy_a_gpu_lane(self):
        check = D.check_device_name("CPU", expected_lane="gpu")
        self.assertEqual(check.outcome, S.FAIL)
        self.assertNotEqual(check.outcome, S.PASS)
        self.assertIn("silent CPU fallback", " ".join(check.reasons))

    def test_a_gpu_device_satisfies_a_gpu_lane(self):
        self.assertEqual(
            D.check_device_name("AMD Instinct MI210", expected_lane="gpu").outcome,
            S.PASS)

    def test_a_gpu_device_fails_a_cpu_lane(self):
        check = D.check_device_name("AMD Instinct MI210", expected_lane="cpu")
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("declared one", " ".join(check.reasons))

    def test_a_host_device_satisfies_a_cpu_lane(self):
        self.assertEqual(D.check_device_name("CPU", expected_lane="cpu").outcome,
                         S.PASS)

    def test_an_unknown_name_is_could_not_check_on_either_lane_never_pass(self):
        for lane in ("cpu", "gpu"):
            check = D.check_device_name("Some Future Accelerator", expected_lane=lane)
            self.assertEqual(check.outcome, S.COULD_NOT_CHECK, lane)
            self.assertNotEqual(check.outcome, S.PASS, lane)
            self.assertFalse(check.passed, lane)

    def test_an_ambiguous_name_is_could_not_check_on_either_lane(self):
        for lane in ("cpu", "gpu"):
            self.assertEqual(
                D.check_device_name("MI210 CPU", expected_lane=lane).outcome,
                S.COULD_NOT_CHECK, lane)

    def test_an_accelerator_listed_beside_host_devices_still_passes_the_gpu_lane(self):
        # ggml enumerates the host backends too; that does not unload the MI210.
        self.assertEqual(
            D.check_device_names(("CPU", "BLAS", "AMD Instinct MI210"),
                                 expected_lane="gpu").outcome, S.PASS)

    def test_any_accelerator_in_the_enumeration_fails_the_cpu_lane(self):
        self.assertEqual(
            D.check_device_names(("CPU", "AMD Instinct MI210"),
                                 expected_lane="cpu").outcome, S.FAIL)

    def test_an_empty_enumeration_is_could_not_check_not_a_pass(self):
        for lane in ("cpu", "gpu"):
            check = D.check_device_names((), expected_lane=lane)
            self.assertEqual(check.outcome, S.COULD_NOT_CHECK, lane)
            self.assertIn("not evidence", " ".join(check.reasons))

    def test_an_undeclared_lane_is_refused_by_name(self):
        for lane in ("hip", "stack", "GPU", None):
            with self.assertRaises(D.DeviceVocabularyError):
                D.check_device_names(("CPU",), expected_lane=lane)

    def test_the_stack_lane_names_no_device_and_cannot_be_graded(self):
        # `serving_runtime` travels the §11.6 stack-change path and enumerates no
        # device; grading it here would invent evidence.
        self.assertIn("stack", S.RESOURCE_LANES)
        self.assertNotIn("stack", D.GRADEABLE_LANES)


class VocabularyIntegrityTest(unittest.TestCase):
    def test_every_entry_carries_what_it_denotes_and_where_it_came_from(self):
        for entry in D.device_vocabulary():
            self.assertIn(entry.device_class, D.DECLARABLE_CLASSES)
            self.assertTrue(entry.denotes.strip(), entry.token)
            self.assertTrue(entry.provenance.strip(), entry.token)
            self.assertEqual(entry.token, entry.token.lower())

    def test_both_classes_are_represented_or_a_lane_could_never_be_graded(self):
        classes = {e.device_class for e in D.device_vocabulary()}
        self.assertEqual(classes, D.DECLARABLE_CLASSES)

    def test_an_entry_cannot_declare_unknown_or_ambiguous(self):
        for bad in (D.UNKNOWN, D.AMBIGUOUS, "accelerator"):
            with self.assertRaises(D.DeviceVocabularyError):
                D.DeviceVocabularyEntry(token="x", device_class=bad, denotes="d",
                                        provenance="p")

    def test_an_entry_without_meaning_or_origin_is_refused(self):
        with self.assertRaises(D.DeviceVocabularyError):
            D.DeviceVocabularyEntry(token="x", device_class=D.GPU, denotes="",
                                    provenance="p")
        with self.assertRaises(D.DeviceVocabularyError):
            D.DeviceVocabularyEntry(token="x", device_class=D.GPU, denotes="d",
                                    provenance="  ")

    def test_a_mixed_case_token_is_refused_because_matching_is_case_insensitive(self):
        with self.assertRaises(D.DeviceVocabularyError):
            D.DeviceVocabularyEntry(token="MI210", device_class=D.GPU, denotes="d",
                                    provenance="p")

    def test_an_empty_or_one_sided_vocabulary_is_refused_at_validation(self):
        with self.assertRaises(D.DeviceVocabularyError):
            D._validate_vocabulary(())
        gpu_only = tuple(e for e in D.device_vocabulary() if e.device_class == D.GPU)
        with self.assertRaises(D.DeviceVocabularyError):
            D._validate_vocabulary(gpu_only)

    def test_a_duplicate_token_is_refused(self):
        entry = D.device_vocabulary()[0]
        with self.assertRaises(D.DeviceVocabularyError):
            D._validate_vocabulary((entry, entry,
                                    D.DeviceVocabularyEntry(token="cpu",
                                                            device_class=D.CPU,
                                                            denotes="d",
                                                            provenance="p")))

    def test_an_invalid_device_class_cannot_be_stamped_onto_a_verdict(self):
        with self.assertRaises(D.DeviceVocabularyError):
            D.DeviceNameVerdict(name="x", device_class="accelerator",
                                matched_tokens=())


class DelegationAuditTest(unittest.TestCase):
    """The audit must not be satisfiable by deleting what it inspects."""

    def _audit(self, source, backend="whisper_stt"):
        return D.audit_delegates_device_vocabulary(
            source, expected_backend=backend, identity_functions=_IDENTITY)

    def test_empty_source_is_could_not_check_never_pass(self):
        for source in ("", None, "   \n"):
            check = self._audit(source)
            self.assertEqual(check.outcome, S.COULD_NOT_CHECK, repr(source))
            self.assertNotEqual(check.outcome, S.PASS, repr(source))

    def test_foreign_source_is_could_not_check(self):
        foreign = (_ADAPTERS / "qwentts_tts.py").read_text(encoding="utf-8")
        check = self._audit(foreign, backend="whisper_stt")
        self.assertEqual(check.outcome, S.COULD_NOT_CHECK)
        self.assertIn("qwentts_tts", " ".join(check.reasons))

    def test_source_without_the_identity_functions_is_could_not_check(self):
        check = self._audit('BACKEND = "whisper_stt"\n'
                            "def check_device_evidence(log, *, expected_lane):\n"
                            "    return devices.check_device_names(\n"
                            "        [log], expected_lane=expected_lane)\n")
        self.assertEqual(check.outcome, S.COULD_NOT_CHECK)

    def test_source_with_no_device_checker_at_all_is_could_not_check(self):
        # Deleting the checker must not read as a clean bill of health.
        source = _CONFORMING_CONSUMER.replace("check_device_evidence", "unrelated")
        check = self._audit(source)
        self.assertEqual(check.outcome, S.COULD_NOT_CHECK)
        self.assertNotEqual(check.outcome, S.PASS)

    def test_unparsable_source_is_could_not_check(self):
        self.assertEqual(self._audit("def (:\n").outcome, S.COULD_NOT_CHECK)

    def test_a_local_device_vocabulary_fails(self):
        source = _CONFORMING_CONSUMER.replace(
            'LANES = {"cpu": "/a", "gpu": "/b"}\n',
            'LANES = {"cpu": "/a", "gpu": "/b"}\n'
            'GPU_NAMES = ("AMD Instinct MI210", "gfx90a")\n')
        check = self._audit(source)
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("MI210", " ".join(check.reasons))

    def test_a_local_vocabulary_of_printed_host_device_names_fails(self):
        source = _CONFORMING_CONSUMER.replace(
            'BACKEND = "whisper_stt"\n',
            'BACKEND = "whisper_stt"\nHOST = {"CPU", "BLAS"}\n')
        self.assertEqual(self._audit(source).outcome, S.FAIL)

    def test_a_checker_that_grades_names_itself_fails(self):
        source = _CONFORMING_CONSUMER.replace(
            "    return devices.check_device_names([log], expected_lane=expected_lane)\n",
            "    return 'MI210' in log\n")
        check = self._audit(source)
        self.assertEqual(check.outcome, S.FAIL)
        self.assertIn("check_device_names", " ".join(check.reasons))

    def test_a_finding_is_returned_unbound_because_it_is_about_the_text(self):
        # Foreign source with a local vocabulary: FAIL, not COULD_NOT_CHECK. A
        # forbidden construct is a finding about whoever's text it is.
        source = ('BACKEND = "something_else"\n'
                  'NAMES = ("AMD Instinct MI210",)\n')
        self.assertEqual(self._audit(source).outcome, S.FAIL)

    def test_a_non_string_source_or_backend_is_refused(self):
        with self.assertRaises(D.DeviceVocabularyError):
            self._audit(b"BACKEND = 'whisper_stt'")
        with self.assertRaises(D.DeviceVocabularyError):
            D.audit_delegates_device_vocabulary(_CONFORMING_CONSUMER,
                                                expected_backend="")

    # --- COMPLIANT-PATH CONTROL ---------------------------------------------

    def test_the_audit_passes_every_real_consumer_and_the_minimal_conforming_one(self):
        """A guard that forbade its consumers' own idioms would be routed around.

        Both real adapters name `libggml-cpu.so` in a frozenset and one keys a dict
        by lane (`cpu`/`gpu`/`stt`) — strings that carry the vocabulary's `cpu` and
        `gpu` tokens on word boundaries. All three sources must PASS.
        """
        self.assertEqual(self._audit(_CONFORMING_CONSUMER).outcome, S.PASS)
        for backend, filename in (("whisper_stt", "whisper_stt.py"),
                                  ("qwentts_tts", "qwentts_tts.py")):
            source = (_ADAPTERS / filename).read_text(encoding="utf-8")
            self.assertEqual(self._audit(source, backend=backend).outcome, S.PASS,
                             filename)

    def test_lane_keys_and_sonames_are_not_device_names(self):
        self.assertFalse(D._is_device_name_literal("cpu"))
        self.assertFalse(D._is_device_name_literal("gpu"))
        self.assertFalse(D._is_device_name_literal("libggml-cpu.so"))
        self.assertFalse(D._is_device_name_literal("/mnt/raid0/llm/llama.cpp/build/bin"))
        self.assertFalse(D._is_device_name_literal("libggml*"))
        # …but the PRINTED device name still is, and that is what a divergent local
        # table would actually contain.
        self.assertTrue(D._is_device_name_literal("CPU"))
        self.assertTrue(D._is_device_name_literal("AMD Instinct MI210"))


class MandatoryTokenTest(unittest.TestCase):
    """RED TEAM: the table's guarantee must not be deletable one row at a time.

    "Both classes are represented" is satisfied after deleting the `cpu` row, because
    `epyc` and `blas` still carry the host class. With that row gone, `Device 0: CPU`
    — the exact string the 2026-07-31 silent fallback prints, and the whole reason
    this module exists — classifies as `unknown`, so a GPU cell grades
    COULD_NOT_CHECK instead of FAIL. The defect would be reopened by deleting one row
    of the table that closes it.
    """

    def _without(self, token):
        return tuple(e for e in D.device_vocabulary() if e.token != token)

    def test_deleting_a_mandatory_token_is_refused_at_import_time_validation(self):
        for token in sorted(D.MANDATORY_TOKENS):
            reduced = self._without(token)
            # The one-sided-vocabulary rule must NOT be what catches this: both
            # classes are still represented after the deletion.
            self.assertEqual({e.device_class for e in reduced}, D.DECLARABLE_CLASSES,
                             token)
            with self.assertRaises(D.DeviceVocabularyError, msg=token) as ctx:
                D._validate_vocabulary(reduced)
            self.assertIn(token, str(ctx.exception))

    def test_deleting_the_cpu_row_is_what_it_would_cost(self):
        # Stated as a fact about the classifier, so the reason the row is mandatory
        # is checked and not merely asserted in a comment.
        patterns = D._TOKEN_PATTERNS
        D._TOKEN_PATTERNS = tuple((e, p) for e, p in patterns if e.token != "cpu")
        try:
            self.assertEqual(D.check_device_name("CPU", expected_lane="gpu").outcome,
                             S.COULD_NOT_CHECK)
        finally:
            D._TOKEN_PATTERNS = patterns
        self.assertEqual(D.check_device_name("CPU", expected_lane="gpu").outcome, S.FAIL)

    def test_a_mandatory_token_cannot_be_reclassed_into_the_other_lane(self):
        # Presence alone is not enough: a `cpu` row declared GPU would satisfy a
        # presence-only rule and invert every verdict that depends on it.
        flipped = tuple(
            D.DeviceVocabularyEntry(token=e.token,
                                    device_class=(D.GPU if e.device_class == D.CPU
                                                  else D.CPU),
                                    denotes=e.denotes, provenance=e.provenance)
            if e.token == "cpu" else e
            for e in D.device_vocabulary())
        with self.assertRaises(D.DeviceVocabularyError) as ctx:
            D._validate_vocabulary(flipped)
        self.assertIn("cpu", str(ctx.exception))

    # --- COMPLIANT-PATH CONTROL ---------------------------------------------

    def test_the_shipped_vocabulary_satisfies_its_own_mandatory_rule(self):
        """The rule must not forbid the table it was written for.

        It also must not be a rule about nothing: every mandatory token is really in
        the shipped table, in the class the rule pins it to, and extending the table
        with new optional rows stays legal.
        """
        D._validate_vocabulary(D.device_vocabulary())
        by_token = {e.token: e for e in D.device_vocabulary()}
        for token, (device_class, why) in D.MANDATORY_TOKENS.items():
            self.assertIn(token, by_token)
            self.assertEqual(by_token[token].device_class, device_class, token)
            self.assertTrue(why.strip(), token)
        extended = D.device_vocabulary() + (
            D.DeviceVocabularyEntry(token="mi300x", device_class=D.GPU,
                                    denotes="a future accelerator",
                                    provenance="test: extending the table is legal"),)
        D._validate_vocabulary(extended)


class HiddenVocabularyAuditTest(unittest.TestCase):
    """RED TEAM: the delegation audit inspected MODULE-LEVEL literals only.

    A table one indent deeper is the same table. Moving it inside
    `check_device_evidence` — and delegating only on the fall-through, which
    satisfies rule 1 — passed the audit outright.
    """

    def _audit(self, source, backend="whisper_stt"):
        return D.audit_delegates_device_vocabulary(
            source, expected_backend=backend, identity_functions=_IDENTITY)

    def test_a_vocabulary_hidden_inside_the_checker_fails(self):
        source = _CONFORMING_CONSUMER.replace(
            "    return devices.check_device_names([log], expected_lane=expected_lane)\n",
            '    local_names = ("AMD Instinct MI210", "gfx90a")\n'
            "    if any(n in log for n in local_names):\n"
            "        return 'PASS'\n"
            "    return devices.check_device_names([log], expected_lane=expected_lane)\n")
        check = self._audit(source)
        self.assertEqual(check.outcome, S.FAIL)
        self.assertNotEqual(check.outcome, S.PASS)
        self.assertIn("MI210", " ".join(check.reasons))

    def test_a_vocabulary_hidden_in_a_helper_function_fails(self):
        source = _CONFORMING_CONSUMER + (
            "def _looks_like_a_gpu(name):\n"
            '    table = {"CPU": False, "AMD Instinct MI210": True}\n'
            "    return table.get(name, False)\n")
        self.assertEqual(self._audit(source).outcome, S.FAIL)

    def test_a_vocabulary_written_as_a_membership_test_fails(self):
        # The table written without ever being bound to a name.
        source = _CONFORMING_CONSUMER.replace(
            "    return devices.check_device_names([log], expected_lane=expected_lane)\n",
            '    if log in ("CPU", "BLAS"):\n'
            "        return 'FAIL'\n"
            "    return devices.check_device_names([log], expected_lane=expected_lane)\n")
        self.assertEqual(self._audit(source).outcome, S.FAIL)

    # --- COMPLIANT-PATH CONTROL ---------------------------------------------

    def test_a_checker_may_still_name_the_device_it_rejected_in_its_own_reason(self):
        """The idiom the widened rule must not forbid.

        Both real adapters build a FAIL reason containing the sentence *"a CPU cell's
        log carries `use gpu = 1`"*. That string names a device in PROSE, as an
        argument to `schemas.Check` — it is a finding, not a vocabulary. A rule that
        walked every collection literal in the module would FAIL both adapters on it,
        and a guard that forbids its consumer from naming what it rejected is worked
        around rather than obeyed.
        """
        source = _CONFORMING_CONSUMER.replace(
            "    return devices.check_device_names([log], expected_lane=expected_lane)\n",
            "    if not log:\n"
            "        return Check(FAIL, (\n"
            "            \"a CPU cell's log carries `use gpu = 1`; the request \"\n"
            '            "contradicts the declared lane",))\n'
            "    return devices.check_device_names([log], expected_lane=expected_lane)\n")
        self.assertEqual(self._audit(source).outcome, S.PASS)

    def test_both_real_adapters_still_pass_the_widened_rule(self):
        for backend, filename in (("whisper_stt", "whisper_stt.py"),
                                  ("qwentts_tts", "qwentts_tts.py")):
            source = (_ADAPTERS / filename).read_text(encoding="utf-8")
            self.assertEqual(self._audit(source, backend=backend).outcome, S.PASS,
                             filename)


class NoWriteNoProcessTest(unittest.TestCase):
    """RED TEAM (surface): this module classifies a string and nothing else.

    The bundle's conformance suite audits `api`, `controls`, `correctness` and
    `integrity` from a hard-coded list; a module added later is not in it. The
    property is checked here instead of assumed, and it is BOUND to this file's real
    text — the empty string passes every rule the auditor has.
    """

    def test_the_vocabulary_module_can_neither_write_nor_spawn(self):
        source = Path(D.__file__).read_text(encoding="utf-8")
        self.assertIn("def classify_device_name", source)
        self.assertIn("HOST_DEVICE_VOCABULARY", source)
        self.assertEqual(api.audit_no_write_or_process_paths(source).outcome, S.PASS)

    def test_it_imports_nothing_that_could_touch_a_file_a_process_or_a_device(self):
        tree = ast.parse(Path(D.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        forbidden = {"os", "subprocess", "shutil", "socket", "signal", "pathlib",
                     "multiprocessing", "ctypes", "urllib", "http", "tempfile"}
        self.assertEqual(imported & forbidden, set(), sorted(imported))


if __name__ == "__main__":
    unittest.main()
