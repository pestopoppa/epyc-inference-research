"""C6-CTRL-MULMAT-ORACLE: new-fail-vs-baseline correctness and NMSE numerics.

Operator rule (INF-48, 2026-11-10): a candidate fails only on cases the
champion baseline passes. Quantized MUL_MAT numerics gate on test-backend-ops'
NMSE bound, not on an elementwise matched ratio against a CPU reference.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace
import unittest

from scripts.kernel_rnd import c6_reward_integrity as R
from . import gpu_source_evidence as E

FIXTURES = Path(__file__).parent / "fixtures"
C10_LOG_SHA256 = E.C10_MUL_MAT_CORRECTNESS_BASELINE["source_log_sha256"]
BASELINE_PARAMS = next(iter(E.C10_MUL_MAT_CORRECTNESS_BASELINE["fail_multiset"]))
ARGV = ("/x/test-backend-ops", "test", "-o", "MUL_MAT", "-b", "ROCm0", "-j", "1",
        "--suite-seed", "2026081301", "--autokernel-properties")


def recorded_c10_stdout() -> str:
    raw = gzip.decompress(base64.b64decode(
        (FIXTURES / "recorded_c10_mul_mat_properties_stdout.txt.gz.b64").read_bytes()))
    if hashlib.sha256(raw).hexdigest() != C10_LOG_SHA256:
        raise AssertionError("recorded C10 MUL_MAT fixture changed")
    return raw.decode("utf-8")


def plan(*, baseline=E.C10_MUL_MAT_CORRECTNESS_BASELINE, cases=1202):
    return SimpleNamespace(correctness_backend="ROCm0", correctness_op="MUL_MAT",
                           expected_correctness_cases=cases,
                           correctness_baseline=baseline)


def console(verdicts: list[tuple[str, str]], *, failing_block: bool = True) -> str:
    """Real-shaped console: ANSI FAIL, a Failing-tests block, backend FAIL."""
    lines = ["Testing 2 devices", "", "Backend 1/2: ROCm0"]
    failed = []
    for params, verdict in verdicts:
        if verdict == "OK":
            lines.append(f"  MUL_MAT({params}): \x1b[1;32mOK\x1b[0m")
        else:
            failed.append(params)
            lines.append(f"  MUL_MAT({params}): test failed \x1b[1;31mFAIL\x1b[0m")
    ok = len(verdicts) - len(failed)
    lines.append(f"  {ok}/{len(verdicts)} tests passed")
    if failed and failing_block:
        lines += ["", "Failing tests:", *(f"  MUL_MAT({p})" for p in failed)]
    status = "FAIL" if failed else "OK"
    lines += [f"  Backend ROCm0: {status}", "Backend 2/2: CPU", "  Skipping",
              f"{1 if failed else 2}/2 backends passed", status, ""]
    return "\n".join(lines)


def baseline_for(cases: int, fails: dict[str, int]):
    return {**E._baseline_json(E.C10_MUL_MAT_CORRECTNESS_BASELINE),
            "expected_cases": cases, "fail_multiset": fails}


class BaselineCorrectnessTests(unittest.TestCase):
    def test_real_c10_log_reproducing_only_the_baseline_fail_passes(self) -> None:
        parsed = E._parse_correctness(recorded_c10_stdout(), plan())
        self.assertEqual(parsed.baseline_fails, (BASELINE_PARAMS,))
        self.assertEqual(parsed.summary, "1201/1202 tests passed")
        self.assertEqual(parsed.overall, "FAIL")
        self.assertEqual(E._expected_correctness_exit(parsed), 1)
        self.assertEqual(E._baseline_receipt_fields(parsed)["new_fails"], [])

    def test_real_c10_log_is_still_refused_under_the_strict_contract(self) -> None:
        with self.assertRaisesRegex(E.CorrectnessParseRefusal, "did not report OK"):
            E._parse_correctness(recorded_c10_stdout(), plan(baseline=None))

    def test_a_new_fail_is_refused_and_named(self) -> None:
        raw = console([("a=1", "OK"), ("a=2", "FAIL"), ("a=3", "FAIL")])
        with self.assertRaisesRegex(E.CorrectnessParseRefusal,
                                    r"not failed by baseline .*MUL_MAT\(a=3\)"):
            E._parse_correctness(raw, plan(baseline=baseline_for(3, {"a=2": 1}), cases=3))

    def test_baseline_count_is_a_multiset_not_a_set(self) -> None:
        raw = console([("a=1", "OK"), ("a=2", "FAIL"), ("a=2", "FAIL")])
        with self.assertRaisesRegex(E.CorrectnessParseRefusal, "not failed by baseline"):
            E._parse_correctness(raw, plan(baseline=baseline_for(3, {"a=2": 1}), cases=3))

    def test_baseline_fail_that_now_passes_is_accepted_as_clean(self) -> None:
        raw = console([("a=1", "OK"), ("a=2", "OK")])
        parsed = E._parse_correctness(raw, plan(baseline=baseline_for(2, {"a=2": 1}), cases=2))
        self.assertEqual(parsed.baseline_fails, ())
        self.assertEqual(E._expected_correctness_exit(parsed), 0)

    def test_truncated_log_is_refused(self) -> None:
        raw = recorded_c10_stdout()
        cut = raw[: len(raw) // 2]
        with self.assertRaises(E.CorrectnessParseRefusal):
            E._parse_correctness(cut, plan())

    def test_crash_mid_case_is_refused(self) -> None:
        raw = console([("a=1", "OK"), ("a=2", "FAIL")])
        crashed = raw.split("  MUL_MAT(a=2)")[0] + "  MUL_MAT(a=2): ggml_cuda_x: path ran\n"
        with self.assertRaises(E.CorrectnessParseRefusal):
            E._parse_correctness(crashed, plan(baseline=baseline_for(2, {"a=2": 1}), cases=2))

    def test_missing_case_is_refused_even_if_all_present_pass(self) -> None:
        raw = console([("a=1", "OK")])
        with self.assertRaisesRegex(E.CorrectnessParseRefusal, "exact expected case count"):
            E._parse_correctness(raw, plan(baseline=baseline_for(2, {"a=2": 1}), cases=2))

    def test_failing_tests_list_must_match_the_tolerated_fails(self) -> None:
        raw = console([("a=1", "OK"), ("a=2", "FAIL")], failing_block=False)
        with self.assertRaisesRegex(E.CorrectnessParseRefusal, "tolerated baseline"):
            E._parse_correctness(raw, plan(baseline=baseline_for(2, {"a=2": 1}), cases=2))

    def test_crash_exit_status_never_matches_a_parse(self) -> None:
        parsed = E._parse_correctness(recorded_c10_stdout(), plan())
        for status in (-6, -11, 134, 2):
            self.assertNotEqual(status, E._expected_correctness_exit(parsed))

    def test_baseline_bound_to_its_measured_argv(self) -> None:
        E._validate_correctness_baseline(
            E.C10_MUL_MAT_CORRECTNESS_BASELINE, argv=ARGV, backend="ROCm0",
            op="MUL_MAT", expected_cases=1202)
        with self.assertRaisesRegex(E.EvidenceProducerError, "different correctness argv"):
            E._validate_correctness_baseline(
                E.C10_MUL_MAT_CORRECTNESS_BASELINE, argv=ARGV[:-1], backend="ROCm0",
                op="MUL_MAT", expected_cases=1202)
        with self.assertRaisesRegex(E.EvidenceProducerError, "case count"):
            E._validate_correctness_baseline(
                E.C10_MUL_MAT_CORRECTNESS_BASELINE, argv=ARGV, backend="ROCm0",
                op="MUL_MAT", expected_cases=1139)


def _floats(hex_text: str) -> list[float]:
    raw = bytes.fromhex(hex_text)
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


class QuantizedNmseNumericsTests(unittest.TestCase):
    def setUp(self) -> None:
        fixture = json.loads((Path(__file__).parents[1] / "native" /
                              "fixture_c6_native_residency_r2.json").read_text())
        self.reference = _floats(fixture["cpu_reference_f32le_hex"])
        self.honest = _floats(fixture["rocm0_candidate_f32le_hex"])
        self.structural = R.StructuralPrecisionEvidence("f32", "f32", "0" * 64)
        self.nmse = R.PrecisionContract(
            required_output_dtype="f32", required_accumulator_dtype="f32",
            atol=0.01, rtol=0.01, required_matched_ratio=0.95, lowbit=True,
            numeric_predicate="nmse", nmse_max=R.TEST_BACKEND_OPS_MUL_MAT_NMSE_MAX)
        self.matched = R.PrecisionContract(
            required_output_dtype="f32", required_accumulator_dtype="f32",
            atol=0.01, rtol=0.01, required_matched_ratio=0.95, lowbit=True)

    def test_honest_q4k_gpu_output_passes_on_nmse(self) -> None:
        verdict = R.evaluate_numerics(self.reference, self.honest,
                                      structural=self.structural, policy=self.nmse)
        self.assertTrue(verdict.correct, verdict)
        self.assertEqual(verdict.predicate, "nmse")
        self.assertLess(verdict.nmse, 5e-4)
        # The matched ratio is still recorded and would have refused it.
        self.assertLess(verdict.matched_ratio, 0.95)
        self.assertFalse(R.evaluate_numerics(
            self.reference, self.honest, structural=self.structural,
            policy=self.matched).correct)

    def test_planted_q4k_defect_is_refused_through_nmse(self) -> None:
        # Wrong per-row scale on one row in four (planted q4_K d/dmin defect).
        defect = [value * (1.25 if index % 4 == 0 else 1.0)
                  for index, value in enumerate(self.honest)]
        verdict = R.evaluate_numerics(self.reference, defect,
                                      structural=self.structural, policy=self.nmse)
        self.assertFalse(verdict.correct)
        self.assertEqual(verdict.reason, "nmse_bound_exceeded")

    def test_nmse_bound_cannot_be_loosened(self) -> None:
        with self.assertRaises(R.EvaluatorPolicyError):
            R.PrecisionContract(required_output_dtype="f32",
                                required_accumulator_dtype="f32",
                                numeric_predicate="nmse", nmse_max=1e-3)
        with self.assertRaises(R.EvaluatorPolicyError):
            R.PrecisionContract(required_output_dtype="f32",
                                required_accumulator_dtype="f32", nmse_max=1e-4)

    def test_f32_contract_keeps_matched_ratio(self) -> None:
        policy = R.PrecisionContract(required_output_dtype="f32",
                                     required_accumulator_dtype="f32",
                                     atol=0.001, rtol=0.001)
        verdict = R.evaluate_numerics([1.0, 2.0], [1.0, 2.1],
                                      structural=self.structural, policy=policy)
        self.assertEqual(verdict.predicate, "matched_ratio")
        self.assertFalse(verdict.correct)


if __name__ == "__main__":
    unittest.main()
