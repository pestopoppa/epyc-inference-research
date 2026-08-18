#!/usr/bin/env python3
"""Red acceptance gate for the v10 targeted-correctness summary boundary.

This module is deliberately hardware-free.  The one live fixture is immutable
stdout already produced by v10; every other console transcript is synthetic.
The gate specifies the join that failed after the real GPU run: the discovery
producer must delegate console grammar to ``execution.t0_provider``, validate
the target selected by the sealed argv, and durably stage a structured
correctness receipt before either profiling arm starts.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from ..execution import t0_provider
from . import discovery_deployment_factory as F
from . import gpu_source_evidence as E
from . import test_gpu_source_evidence as fixtures


V10_STDOUT = Path(os.environ.get(
    "AUTOKERNEL_V10_CORRECTNESS_STDOUT",
    "/mnt/raid0/llm/autokernel/deployments/"
    "gpu-discovery-quant-ladder-occupancy-v10/operations/"
    "8248d04bef0f60a72c96804357dcea86039c0f3af122e1a54a82f6a959095f4d/"
    "proof/correctness/stdout.txt",
)).resolve()
V10_STDOUT_SHA256 = "f342b8503ad53e88a9f6d8280a21e50069dc203562aa51b07a5f6183f9fb466c"


def _contract(*, op: str, cases: int, backend: str = "ROCm0") -> object:
    return SimpleNamespace(
        correctness_argv=("/fixture/test-backend-ops", "test", "-o", op,
                          "-b", backend, "-j", "1", "--suite-seed", "2026081301"),
        correctness_backend=backend,
        correctness_op=op,
        expected_correctness_cases=cases,
    )


def _console(*, op: str, cases: int, backend: str = "ROCm0",
             skipped: tuple[str, ...] = ("CPU",), status: str = "OK",
             duplicate_summary: bool = False, omit_summary: bool = False,
             case_op: str | None = None) -> str:
    devices = 1 + len(skipped)
    rows = [f"Testing {devices} devices", "", f"Backend 1/{devices}: {backend}"]
    emitted_op = case_op or op
    passed = cases if status == "OK" else max(cases - 1, 0)
    for index in range(cases):
        verdict = "FAIL" if status == "FAIL" and index == cases - 1 else "OK"
        rows.append(f"  {emitted_op}(fixture={index}): {verdict}")
    if not omit_summary:
        rows.append(f"  {passed}/{cases} tests passed")
        if duplicate_summary:
            rows.append(f"  {passed}/{cases} tests passed")
    rows.append(f"  Backend {backend}: {status}")
    for offset, name in enumerate(skipped, start=2):
        rows.extend((f"Backend {offset}/{devices}: {name}", "  Skipping"))
    backends_passed = devices if status == "OK" else devices - 1
    rows.extend((f"{backends_passed}/{devices} backends passed", status))
    return "\n".join(rows) + "\n"


def _expected(*, op: str, cases: int, backend: str = "ROCm0",
              skipped: tuple[str, ...] = ("CPU",)) -> dict[str, object]:
    total = 1 + len(skipped)
    return {
        "summary": f"{cases}/{cases} tests passed",
        "target_backend": backend,
        "target_op": op,
        "passed_cases": cases,
        "total_cases": cases,
        "target_status": "OK",
        "skipped_backends": list(skipped),
        "backends_passed": total,
        "backends_total": total,
        "overall": "OK",
    }


class CorrectnessSummaryAcceptance(unittest.TestCase):
    """The producer consumes the authoritative parser's structured result."""

    def _parse(self, text: str, plan: object) -> dict[str, object]:
        with mock.patch.object(
                t0_provider, "parse_backend_ops_console",
                wraps=t0_provider.parse_backend_ops_console) as parser:
            result = E._parse_correctness(text, plan)
        parser.assert_called_once_with(text)
        return {
            "summary": result.summary,
            "target_backend": result.backend,
            "target_op": result.operation,
            "passed_cases": result.passed_cases,
            "total_cases": result.total_cases,
            "target_status": "OK",
            "skipped_backends": list(result.skipped_backends),
            "backends_passed": result.backends_passed,
            "backends_total": result.backends_total,
            "overall": result.overall,
        }

    @unittest.skipUnless(V10_STDOUT.is_file(),
                         "requires immutable v10 correctness stdout")
    def test_exact_v10_stdout_is_a_valid_two_backend_run(self) -> None:
        raw = V10_STDOUT.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), V10_STDOUT_SHA256)
        text = raw.decode("utf-8")

        # Independent audit of the evidence before exercising the integration.
        run = t0_provider.parse_backend_ops_console(text)
        run.reconcile()
        self.assertEqual((run.backends_passed, run.backends_total), (2, 2))
        self.assertEqual(run.overall, "OK")
        self.assertEqual(run.failing_tests, ())
        self.assertEqual(len(run.backends), 2)
        rocm, cpu = run.backends
        self.assertEqual((rocm.name, rocm.skipped, rocm.status),
                         ("ROCm0", False, "OK"))
        self.assertEqual((rocm.reported_passed, rocm.reported_total),
                         (1139, 1139))
        compared = tuple(case for case in rocm.cases
                         if case.status != "not_supported")
        self.assertEqual(len(compared), 1139)
        self.assertTrue(all(case.op == "MUL_MAT" and case.passed
                            for case in compared))
        self.assertEqual((cpu.name, cpu.skipped, cpu.skip_reason),
                         ("CPU", True, "Skipping"))

        parsed = self._parse(text, _contract(op="MUL_MAT", cases=1139))
        self.assertEqual(parsed, _expected(op="MUL_MAT", cases=1139))

    def test_four_strategy_contracts_are_parameterized_by_template(self) -> None:
        registry = F._template_registry()
        strategies = {
            "q5": "cuda-vecdotq-v1",
            "q8_quantizer": "cuda-quantize-q8-v1",
            "fattention_gqa7": "cuda-fattn-tile-v1",
            "rmsnorm": "cuda-norm-v2",
        }
        for strategy, template_id in strategies.items():
            template = registry.templates[template_id]
            op = template.semantics["correctness_op"]
            cases = template.semantics["expected_correctness_cases"]
            with self.subTest(strategy=strategy, template=template_id,
                              op=op, cases=cases):
                plan = _contract(op=op, cases=cases)
                text = _console(op=op, cases=cases)
                self.assertEqual(self._parse(text, plan),
                                 _expected(op=op, cases=cases))

    def test_zero_skipped_wrong_backend_wrong_op_and_failures_refuse(self) -> None:
        plan = _contract(op="RMS_NORM", cases=3)
        invalid = {
            "zero-cases": _console(op="RMS_NORM", cases=0),
            "target-skipped": (
                "Testing 1 devices\n\nBackend 1/1: ROCm0\n  Skipping\n"
                "1/1 backends passed\nOK\n"),
            "wrong-backend": _console(op="RMS_NORM", cases=3, backend="CUDA0"),
            "wrong-op": _console(op="RMS_NORM", case_op="MUL_MAT", cases=3),
            "failed-case": _console(op="RMS_NORM", cases=3, status="FAIL"),
        }
        for label, text in invalid.items():
            with self.subTest(case=label), self.assertRaises(E.EvidenceProducerError):
                E._parse_correctness(text, plan)

    def test_malformed_missing_and_duplicate_summaries_refuse(self) -> None:
        plan = _contract(op="MUL_MAT", cases=3)
        invalid = {
            "not-console": "3/3 tests passed\nBackend ROCm0: OK\n",
            "missing-target-summary": _console(
                op="MUL_MAT", cases=3, omit_summary=True),
            "duplicate-target-summary": _console(
                op="MUL_MAT", cases=3, duplicate_summary=True),
            "truncated": "Testing 1 devices\n\nBackend 1/1: ROCm0\n"
                         "  MUL_MAT(fixture=0): OK\n",
        }
        for label, text in invalid.items():
            with self.subTest(case=label), self.assertRaises(E.EvidenceProducerError):
                E._parse_correctness(text, plan)


class CorrectnessReceiptStageAcceptance(unittest.TestCase):
    """A parsed receipt is durable before the first profiler invocation."""

    def test_structured_correctness_receipt_precedes_candidate_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            plan = fixtures.plan(root / "inputs")
            tool = plan.correctness_argv[0]
            original_tail = plan.correctness_argv[1:]
            plan = replace(
                plan,
                correctness_argv=(tool, "test", "-o", "RMS_NORM", "-b", "ROCm0",
                                  "-j", "1", "--suite-seed", "2026081301",
                                  *original_tail),
                correctness_backend="ROCm0",
                correctness_op="RMS_NORM",
                expected_correctness_cases=3,
            )
            # The policy receipts the exact plan, so refresh the test-owned
            # carrier after changing the correctness contract.
            plan.policy.path.write_text(
                json.dumps(E._policy_payload(plan), sort_keys=True), encoding="utf-8")
            plan = replace(plan, policy=E.BoundInputFile(
                "execution_policy", plan.policy.path,
                hashlib.sha256(plan.policy.path.read_bytes()).hexdigest()))

            delegates = fixtures.FakeExecutors(
                correctness_summary=_console(op="RMS_NORM", cases=3).rstrip("\n"))
            observed: list[dict[str, object]] = []
            proof_root = root / "proof"

            def correctness(invocation: E.CommandInvocation) -> E.ExecutionCapture:
                return delegates.correctness(invocation)

            def rocprof(invocation: E.CommandInvocation) -> E.ExecutionCapture:
                receipt_path = proof_root / "correctness/receipt.json"
                self.assertTrue(receipt_path.is_file(),
                                "profiling started before correctness was durable")
                receipt = E.proofs.load_receipt(
                    receipt_path, schema=E.CORRECTNESS_SCHEMA)["body"]
                observed.append({
                    "arm": invocation.arm,
                    "parsed_summary": {
                        "summary": receipt.get("summary"),
                        "target_backend": receipt.get("correctness_backend"),
                        "target_op": receipt.get("correctness_op"),
                        "passed_cases": receipt.get("passed_cases"),
                        "total_cases": receipt.get("expected_cases"),
                        "target_status": "OK" if receipt.get("result") == "PASS" else None,
                        "skipped_backends": receipt.get("skipped_backends"),
                        "backends_passed": receipt.get("backends_passed"),
                        "backends_total": receipt.get("backends_total"),
                        "overall": receipt.get("overall"),
                    },
                    "stdout_sha256": receipt.get("stdout_sha256"),
                })
                return delegates.rocprof(invocation)

            E.produce_gpu_source_evidence(
                output_root=proof_root, plan=plan,
                correctness_executor=correctness, rocprof_executor=rocprof,
                claim_journal=object(), claim_acquirer=fixtures.ClaimFactory(),
                claim_verifier=lambda _receipt: True, claim_timeout_s=0,
            )
            self.assertEqual([row["arm"] for row in observed],
                             ["candidate", "anchor"])
            self.assertEqual(observed[0]["parsed_summary"],
                             _expected(op="RMS_NORM", cases=3))
            self.assertEqual(
                observed[0]["stdout_sha256"],
                hashlib.sha256(_console(op="RMS_NORM", cases=3).encode()).hexdigest())


if __name__ == "__main__":
    unittest.main()
