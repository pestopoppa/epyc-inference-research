#!/usr/bin/env python3
"""Inference-free planted tests for the C3/EPYC suite contracts."""
from __future__ import annotations

import hashlib
import unittest

from .. import schemas
from . import c3_epyc_suite as C


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


PASS = schemas.Check(schemas.PASS)


class C3EpycSuiteTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cases = C.epyc_op_suite()
        self.case = self.cases[0]
        self.surface = C.ExactOpSurface.create(
            case_id=self.case.case_id,
            device_id="ROCm0",
            model_sha256=digest("model"),
            quant="bf16",
            operation="mla_paged_prefill",
            shape=(17, 16, 512, 64),
            dtype="bf16",
            tensor_manifest_sha256=digest("attention-tensors"),
            recipe_id="epyc.op.attention.v1",
            recipe_sha256=digest("attention-recipe"),
            harness_build_sha256=digest("harness"),
            factors={"graphs": "off", "stream_sync": "full_device", "warmup": 10,
                     "timed_outputs": "validated",
                     "input_rotation": "address_and_content"},
        )

    def make_surface(self, factors) -> C.ExactOpSurface:
        return C.ExactOpSurface.create(
            case_id=self.case.case_id, device_id="ROCm0",
            model_sha256=digest("model"), quant="bf16",
            operation="mla_paged_prefill", shape=(17, 16, 512, 64),
            dtype="bf16", tensor_manifest_sha256=digest("attention-tensors"),
            recipe_id="epyc.op.attention.v1",
            recipe_sha256=digest("attention-recipe"),
            harness_build_sha256=digest("harness"), factors=factors)

    def test_timing_surface_refuses_phase_capture_content_and_unsynchronized_stream_gaps(self):
        base = {"graphs": "off", "stream_sync": "full_device", "warmup": 10,
                "timed_outputs": "validated",
                "input_rotation": "address_and_content"}
        for changed, message in (
                ({"graphs": "on"}, "phase/capture/content"),
                ({"timed_outputs": "unchecked"}, "phase/capture/content"),
                ({"input_rotation": "address_only"}, "phase/capture/content"),
                ({"stream_sync": "event_only"}, "stream integrity")):
            with self.subTest(changed=changed), self.assertRaisesRegex(
                    C.C3ContractError, message):
                self.make_surface({**base, **changed})

    def test_tracked_stream_timing_requires_fence_after_start_and_join_before_stop(self):
        base = {"graphs": "off", "stream_sync": "tracked_fence_join_v1",
                "warmup": 10, "timed_outputs": "validated",
                "input_rotation": "address_and_content",
                "stream_join": "before_stop"}
        with self.assertRaisesRegex(C.C3ContractError, "fence after start"):
            self.make_surface(base)
        surface = self.make_surface({**base, "stream_fence": "after_start"})
        self.assertIn(("stream_sync", "tracked_fence_join_v1"), surface.factors)

    def observation(self, provider: str, samples=(100.0, 101.0, 99.0), *,
                    surface=None, suffix="baseline") -> C.TimingObservation:
        implementation = digest(f"implementation-{suffix}")
        production_baseline = None
        if provider == C.LLAMA_CPP_PRODUCTION_V9:
            production_baseline = C.FrozenProductionBaseline(
                branch=C.PRODUCTION_V9_BRANCH,
                source_commit=C.PRODUCTION_V9_COMMIT,
                version=C.PRODUCTION_V9_VERSION,
                binary_sha256=implementation,
                linkage_sha256=digest("production-v9-linkage"),
                attestation_ref=C.PRODUCTION_V9_FREEZE_ATTESTATION_REF,
                attestation_sha256=C.PRODUCTION_V9_FREEZE_ATTESTATION_SHA256)
        return C.TimingObservation(
            provider=provider,
            surface=self.surface if surface is None else surface,
            implementation_sha256=implementation,
            samples_ns=tuple(samples),
            evidence_ref=f"evidence://{suffix}",
            evidence_sha256=digest(f"evidence-{suffix}"),
            production_baseline=production_baseline,
        )

    def floor(self) -> C.VendorFloor:
        return C.select_vendor_floor(
            self.case, self.surface,
            (self.observation(C.TORCH_ROCM_COMPILE),))

    def test_suite_is_exact_and_dequant_is_not_relabelled_as_c5(self):
        self.assertEqual(
            tuple(case.case_id for case in self.cases),
            ("epyc.attention.mla_paged_prefill.k228",
             "epyc.moe.sparse_expert_dispatch.k175",
             "epyc.dequant.q4_k_decode_gemv"),
        )
        self.assertEqual(self.cases[0].source_ref, "hyra-sol-execbench/k228")
        self.assertEqual(self.cases[1].source_ref, "hyra-sol-execbench/k175")
        self.assertEqual(self.cases[2].source_kind, "epyc_native_contract")
        self.assertNotIn("hyra", self.cases[2].source_ref)
        for case in self.cases[:2]:
            self.assertEqual(case.required_baseline_providers,
                             (C.TORCH_ROCM_COMPILE,))
            self.assertFalse(case.to_dict()["baseline"]["eager_allowed"])
        self.assertEqual(self.cases[2].required_baseline_providers,
                         (C.LLAMA_CPP_PRODUCTION_V9,))

    def test_dequant_requires_exact_frozen_v9_baseline_identity(self):
        case = self.cases[2]
        surface = C.ExactOpSurface.create(
            case_id=case.case_id, device_id="ROCm0", model_sha256=digest("model"),
            quant="Q4_K", operation=case.operator_family, shape=(1, 256), dtype="f32",
            tensor_manifest_sha256=digest("q4-tensors"), recipe_id="q4.v1",
            recipe_sha256=digest("q4-recipe"), harness_build_sha256=digest("harness"),
            factors={"graphs": "off", "stream_sync": "full_device", "warmup": 10,
                     "timed_outputs": "validated",
                     "input_rotation": "address_and_content"})
        with self.assertRaisesRegex(C.C3ContractError, "exact frozen-v9"):
            C.TimingObservation(
                provider=C.LLAMA_CPP_PRODUCTION_V9, surface=surface,
                implementation_sha256=digest("v9-binary"), samples_ns=(1.0, 1.1, 0.9),
                evidence_ref="evidence://v9", evidence_sha256=digest("v9-evidence"))
        with self.assertRaisesRegex(C.IdentityMismatch, "identity drifted"):
            C.FrozenProductionBaseline(
                branch=C.PRODUCTION_V9_BRANCH, source_commit="f" * 40,
                version=C.PRODUCTION_V9_VERSION, binary_sha256=digest("v9-binary"),
                linkage_sha256=digest("v9-linkage"),
                attestation_ref=C.PRODUCTION_V9_FREEZE_ATTESTATION_REF,
                attestation_sha256=C.PRODUCTION_V9_FREEZE_ATTESTATION_SHA256)
        with self.assertRaisesRegex(C.C3ContractError, "one observation per provider"):
            C.select_vendor_floor(
                case, surface,
                (C.TimingObservation(
                    provider=C.TORCH_ROCM_COMPILE, surface=surface,
                    implementation_sha256=digest("torch"), samples_ns=(1.0, 1.1, 0.9),
                    evidence_ref="evidence://torch",
                    evidence_sha256=digest("torch-evidence")),))

    def test_eager_or_candidate_baseline_is_refused(self):
        with self.assertRaisesRegex(C.C3ContractError, "unsupported timing provider"):
            self.observation("torch_eager")
        with self.assertRaisesRegex(C.C3ContractError, "candidate cannot serve"):
            C.select_vendor_floor(
                self.case, self.surface,
                (self.observation(C.CANDIDATE_PROVIDER),))

    def test_required_vendor_provider_and_exact_surface_are_fail_closed(self):
        with self.assertRaisesRegex(C.C3ContractError, "one observation per provider"):
            C.select_vendor_floor(self.case, self.surface, ())
        other = C.ExactOpSurface.create(
            case_id=self.case.case_id, device_id="ROCm0",
            model_sha256=digest("other-model"), quant="bf16",
            operation="mla_paged_prefill", shape=(17, 16, 512, 64), dtype="bf16",
            tensor_manifest_sha256=digest("attention-tensors"),
            recipe_id="epyc.op.attention.v1", recipe_sha256=digest("attention-recipe"),
            harness_build_sha256=digest("harness"),
            factors={"graphs": "off", "stream_sync": "full_device", "warmup": 10,
                     "timed_outputs": "validated",
                     "input_rotation": "address_and_content"})
        with self.assertRaisesRegex(C.IdentityMismatch, "another exact surface"):
            C.select_vendor_floor(
                self.case, self.surface,
                (self.observation(C.TORCH_ROCM_COMPILE, surface=other),))

    def test_fast_p_is_correctness_first_and_uses_vendor_floor(self):
        candidate = self.observation(
            C.CANDIDATE_PROVIDER, (80.0, 81.0, 79.0), suffix="candidate")
        gate = C.score_fast_p(
            floor=self.floor(), candidate=candidate, p=1.2,
            correctness=PASS, integrity=PASS)
        self.assertEqual(gate.check.outcome, schemas.PASS)
        self.assertEqual(gate.speedup, 1.25)
        self.assertEqual(gate.baseline_provider, C.TORCH_ROCM_COMPILE)

        failed = C.score_fast_p(
            floor=self.floor(), candidate=candidate, p=1.0,
            correctness=schemas.Check(schemas.FAIL, ("planted wrong output",)),
            integrity=PASS)
        self.assertEqual(failed.check.outcome, schemas.FAIL)
        self.assertIsNone(failed.speedup)

    def test_fast_p_missing_evidence_is_not_a_zero_or_a_pass(self):
        result = C.score_fast_p(
            floor=None, candidate=None, p=1.0,
            correctness=PASS, integrity=PASS)
        self.assertEqual(result.check.outcome, schemas.COULD_NOT_CHECK)
        self.assertIsNone(result.speedup)

    def test_suite_fast_p_requires_every_exact_case(self):
        gates = tuple(C.FastPGate(
            case.case_id, 1.0, 1.1, PASS, C.TORCH_ROCM_COMPILE,
            "evidence://baseline", "evidence://candidate",
            digest(f"candidate-{case.case_id}")) for case in self.cases)
        report = C.aggregate_fast_p(self.cases, gates, p=1.0)
        self.assertEqual(report.fast_p, 1.0)
        self.assertEqual(report.scored_cases, 3)
        self.assertEqual(report.authority, C.SEARCH_EXIT_AUTHORITY)
        with self.assertRaisesRegex(C.C3ContractError, "exactly one gate"):
            C.aggregate_fast_p(self.cases, gates[:-1], p=1.0)

    def test_suite_fast_p_is_withheld_when_any_case_is_unavailable(self):
        gates = [C.FastPGate(
            case.case_id, 1.0, 1.1, PASS, C.TORCH_ROCM_COMPILE,
            "evidence://baseline", "evidence://candidate",
            digest(f"candidate-{case.case_id}")) for case in self.cases]
        gates[-1] = C.FastPGate(
            self.cases[-1].case_id, 1.0, None,
            schemas.Check(schemas.COULD_NOT_CHECK, ("not measured",)),
            None, None, None, None)
        report = C.aggregate_fast_p(self.cases, gates, p=1.0)
        self.assertIsNone(report.fast_p)
        self.assertEqual(report.scored_cases, 2)

    def workload(self) -> C.CapturedWorkload:
        return C.CapturedWorkload(
            workload_id="epyc.production.prefill.capture.v1",
            model_sha256=digest("model"),
            tensor_manifest_sha256=digest("whole-model-tensors"),
            capture_receipt_ref="evidence://tensor-capture",
            capture_receipt_sha256=digest("tensor-capture-receipt"))

    def whole_surface(self, *, workload=None) -> C.WholeModelSurface:
        return C.WholeModelSurface.create(
            workload=self.workload() if workload is None else workload,
            device_id="ROCm0", quant="bf16",
            recipe_id="apex.epyc.whole_model.v1",
            recipe_sha256=digest("whole-model-recipe"),
            factors={"graphs": "off", "stage": "prefill", "warmup": 10})

    def diagnostic_integration(self) -> C.DiagnosticProviderBinding:
        return C.DiagnosticProviderBinding(
            runner_id=C.APEX_PYTHON_OVERLAY, runner_revision=C.PINNED_APEX_REVISION,
            patch_bundle_sha256=digest("patch"),
            candidate_source_sha256=digest("implementation-candidate"),
            candidate_build_sha256=digest("candidate-build"),
            candidate_binary_sha256=digest("candidate-binary"),
            receipt_ref="evidence://hot-patch",
            receipt_sha256=digest("hot-patch-receipt"))

    def integration(self) -> C.IntegratedLlamaGpuBinding:
        return C.IntegratedLlamaGpuBinding(
            candidate_branch="ak/c3-integrated/source",
            production_base_commit="1" * 40,
            candidate_source_commit="2" * 40,
            patch_bundle_sha256=digest("patch"),
            candidate_source_sha256=digest("implementation-candidate"),
            candidate_build_sha256=digest("candidate-build"),
            candidate_binary_sha256=digest("candidate-binary"),
            candidate_linkage_sha256=digest("candidate-linkage"),
            toolchain_manifest_sha256=digest("toolchain"),
            isolation_root="/mnt/raid0/llm/autokernel/providers/c3-test",
            receipt_ref="evidence://integrated-llama-gpu",
            receipt_sha256=digest("integrated-receipt"))

    def whole_observation(self, arm: str, samples, *, surface=None,
                          build="anchor-build", binary="anchor-binary"):
        return C.WholeModelObservation(
            arm=arm, surface=self.whole_surface() if surface is None else surface,
            build_sha256=digest(build), binary_sha256=digest(binary),
            samples_ns=tuple(samples), evidence_ref=f"evidence://{arm}",
            evidence_sha256=digest(f"whole-{arm}"))

    def passed_operator_gate(self) -> C.FastPGate:
        candidate = self.observation(
            C.CANDIDATE_PROVIDER, (80.0, 81.0, 79.0), suffix="candidate")
        return C.score_fast_p(
            floor=self.floor(), candidate=candidate, p=1.0,
            correctness=PASS, integrity=PASS)

    def test_whole_model_exit_requires_real_integration_and_matched_capture(self):
        missing = C.evaluate_whole_model_exit(
            operator_gate=self.passed_operator_gate(), integration=None,
            anchor=None, candidate=None, correctness=PASS, integrity=PASS)
        self.assertEqual(missing.check.outcome, schemas.COULD_NOT_CHECK)
        self.assertIsNone(missing.speedup)

        surface = self.whole_surface()
        anchor = self.whole_observation(
            "unpatched_anchor", (1000.0, 1001.0, 999.0), surface=surface)
        candidate = self.whole_observation(
            "integrated_candidate", (900.0, 901.0, 899.0), surface=surface,
            build="candidate-build", binary="candidate-binary")
        report = C.evaluate_whole_model_exit(
            operator_gate=self.passed_operator_gate(), integration=self.integration(),
            anchor=anchor, candidate=candidate, correctness=PASS, integrity=PASS)
        self.assertEqual(report.check.outcome, schemas.PASS)
        self.assertAlmostEqual(report.speedup, 1000.0 / 900.0)
        self.assertFalse(report.promotion_authorized)
        self.assertEqual(report.authority_boundary, C.NO_PROMOTION_AUTHORITY)

    def test_whole_model_surface_or_integration_identity_mismatch_refuses(self):
        surface = self.whole_surface()
        other_workload = C.CapturedWorkload(
            workload_id="other", model_sha256=digest("model"),
            tensor_manifest_sha256=digest("other-tensors"),
            capture_receipt_ref="evidence://other",
            capture_receipt_sha256=digest("other-receipt"))
        other_surface = self.whole_surface(workload=other_workload)
        anchor = self.whole_observation(
            "unpatched_anchor", (1000.0, 1001.0, 999.0), surface=surface)
        candidate = self.whole_observation(
            "integrated_candidate", (900.0, 901.0, 899.0), surface=other_surface,
            build="candidate-build", binary="candidate-binary")
        with self.assertRaisesRegex(C.IdentityMismatch, "different captured"):
            C.evaluate_whole_model_exit(
                operator_gate=self.passed_operator_gate(), integration=self.integration(),
                anchor=anchor, candidate=candidate, correctness=PASS, integrity=PASS)

        candidate = self.whole_observation(
            "integrated_candidate", (900.0, 901.0, 899.0), surface=surface,
            build="wrong-build", binary="candidate-binary")
        with self.assertRaisesRegex(C.IdentityMismatch, "different candidate build"):
            C.evaluate_whole_model_exit(
                operator_gate=self.passed_operator_gate(), integration=self.integration(),
                anchor=anchor, candidate=candidate, correctness=PASS, integrity=PASS)

    def test_diagnostic_provider_cannot_satisfy_integrated_exit(self):
        surface = self.whole_surface()
        anchor = self.whole_observation(
            "unpatched_anchor", (1000.0, 1001.0, 999.0), surface=surface)
        candidate = self.whole_observation(
            "integrated_candidate", (900.0, 901.0, 899.0), surface=surface,
            build="candidate-build", binary="candidate-binary")
        report = C.evaluate_whole_model_exit(
            operator_gate=self.passed_operator_gate(),
            integration=self.diagnostic_integration(), anchor=anchor,
            candidate=candidate, correctness=PASS, integrity=PASS)
        self.assertEqual(report.check.outcome, schemas.COULD_NOT_CHECK)
        self.assertIn("diagnostic provider", report.check.reasons[0])

    def test_integrated_binding_refuses_shared_rocm_and_wrong_backend(self):
        values = dict(self.integration().__dict__)
        values["isolation_root"] = "/opt/rocm"
        with self.assertRaisesRegex(C.C3ContractError, "prohibited prefix"):
            C.IntegratedLlamaGpuBinding(**values)
        values = dict(self.integration().__dict__)
        values["backend"] = "llama_cpu"
        with self.assertRaisesRegex(C.C3ContractError, "through llama_gpu"):
            C.IntegratedLlamaGpuBinding(**values)

    def test_pinned_apex_and_no_execution_authority_are_mechanical(self):
        with self.assertRaisesRegex(C.C3ContractError, "pinned revision"):
            C.HotPatchBinding(
                runner_id=C.APEX_PYTHON_OVERLAY,
                runner_revision="0" * 40,
                patch_bundle_sha256=digest("patch"),
                candidate_source_sha256=digest("source"),
                candidate_build_sha256=digest("build"),
                candidate_binary_sha256=digest("binary"),
                receipt_ref="evidence://patch", receipt_sha256=digest("receipt"))
        self.assertEqual(C.audit_no_execution_paths().outcome, schemas.PASS)
        requirements = C.external_artifact_requirements()
        self.assertTrue(requirements)
        self.assertTrue(all(not item.presence_asserted for item in requirements))


if __name__ == "__main__":
    unittest.main()
