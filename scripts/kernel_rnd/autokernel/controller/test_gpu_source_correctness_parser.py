"""Regression tests for the typed GPU correctness-output contract."""

from __future__ import annotations

import base64
import gzip
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest

from . import gpu_source_evidence as E
from .test_gpu_source_evidence import correctness_console


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "recorded_v10_correctness_stdout.txt.gz.b64"
)
V10_STDOUT_SHA256 = "f342b8503ad53e88a9f6d8280a21e50069dc203562aa51b07a5f6183f9fb466c"


def plan(*, backend: str, op: str, cases: int) -> SimpleNamespace:
    return SimpleNamespace(
        correctness_backend=backend,
        correctness_op=op,
        expected_correctness_cases=cases,
    )


def recorded_v10_stdout() -> str:
    compressed = base64.b64decode(FIXTURE.read_bytes())
    raw = gzip.decompress(compressed)
    if hashlib.sha256(raw).hexdigest() != V10_STDOUT_SHA256:
        raise AssertionError("recorded v10 correctness fixture changed")
    return raw.decode("utf-8")


class GpuSourceCorrectnessParserTests(unittest.TestCase):
    def test_exact_v10_gpu_pass_with_skipped_cpu_is_accepted(self) -> None:
        raw = recorded_v10_stdout()
        self.assertEqual(len(raw.encode("utf-8")), 160_038)
        parsed = E._parse_correctness(
            raw, plan(backend="ROCm0", op="MUL_MAT", cases=1139))
        self.assertEqual(parsed.summary, "1139/1139 tests passed")
        self.assertEqual(parsed.skipped_backends, ("CPU",))
        self.assertEqual((parsed.backends_passed, parsed.backends_total), (2, 2))
        self.assertEqual(parsed.overall, "OK")

    def test_each_eligible_strategy_uses_its_template_case_contract(self) -> None:
        # Q5 and Q8 intentionally share MUL_MAT's current suite size; FA and
        # RMS prove the reducer is not specialized to that operation or count.
        strategies = (
            ("q5", "MUL_MAT", 1139),
            ("q8", "MUL_MAT", 1139),
            ("fa", "FLASH_ATTN_EXT", 2868),
            ("rms", "RMS_NORM", 21),
        )
        for strategy, op, cases in strategies:
            with self.subTest(strategy=strategy):
                parsed = E._parse_correctness(
                    correctness_console(
                        f"{cases}/{cases} tests passed", op=op),
                    plan(backend="ROCm0", op=op, cases=cases))
                self.assertEqual(parsed.passed_cases, cases)
                self.assertEqual(parsed.operation, op)

    def test_zero_supported_cases_cannot_pass(self) -> None:
        raw = "\n".join((
            "Testing 2 devices", "", "Backend 1/2: ROCm0",
            "  0/0 tests passed", "  Backend ROCm0: OK",
            "Backend 2/2: CPU", "  Skipping", "2/2 backends passed", "OK", ""))
        with self.assertRaisesRegex(
                E.CorrectnessParseRefusal, "zero supported cases"):
            E._parse_correctness(
                raw, plan(backend="ROCm0", op="RMS_NORM", cases=1))

    def test_unsupported_cases_do_not_satisfy_the_count(self) -> None:
        raw = "\n".join((
            "Testing 2 devices", "", "Backend 1/2: ROCm0",
            "  RMS_NORM(case=0): not supported [ROCm0]",
            "  0/0 tests passed", "  Backend ROCm0: OK",
            "Backend 2/2: CPU", "  Skipping", "2/2 backends passed", "OK", ""))
        with self.assertRaisesRegex(
                E.CorrectnessParseRefusal, "zero supported cases"):
            E._parse_correctness(
                raw, plan(backend="ROCm0", op="RMS_NORM", cases=1))

    def test_wrong_backend_cannot_masquerade_as_target_evidence(self) -> None:
        raw = correctness_console(
            "1/1 tests passed", backend="CUDA0", op="RMS_NORM")
        with self.assertRaisesRegex(
                E.CorrectnessParseRefusal, "exactly one target backend"):
            E._parse_correctness(
                raw, plan(backend="ROCm0", op="RMS_NORM", cases=1))

    def test_duplicate_summary_is_a_typed_parse_refusal(self) -> None:
        raw = correctness_console("1/1 tests passed", op="RMS_NORM")
        raw = raw.replace(
            "2/2 backends passed\nOK",
            "2/2 backends passed\n2/2 backends passed\nOK")
        with self.assertRaisesRegex(
                E.CorrectnessParseRefusal, "more than one backends-passed"):
            E._parse_correctness(
                raw, plan(backend="ROCm0", op="RMS_NORM", cases=1))

    def test_failed_case_cannot_be_hidden_by_a_summary(self) -> None:
        raw = correctness_console("1/2 tests passed", op="RMS_NORM")
        with self.assertRaisesRegex(
                E.CorrectnessParseRefusal, "did not report OK"):
            E._parse_correctness(
                raw, plan(backend="ROCm0", op="RMS_NORM", cases=2))

    def test_backend_status_must_name_the_active_frame(self) -> None:
        raw = correctness_console("1/1 tests passed", op="RMS_NORM")
        raw = raw.replace("Backend ROCm0: OK", "Backend CUDA0: OK")
        with self.assertRaisesRegex(
                E.CorrectnessParseRefusal, "active backend frame"):
            E._parse_correctness(
                raw, plan(backend="ROCm0", op="RMS_NORM", cases=1))


if __name__ == "__main__":
    unittest.main()
