#!/usr/bin/env python3
"""Deterministic JSON compiler tests; all numeric fixtures are planted."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import unittest
from contextlib import redirect_stdout

from .. import schemas
from . import c3_epyc_compiler as P
from . import c3_epyc_suite as C


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def evidence_check(label: str, outcome: str = schemas.PASS) -> dict:
    return {
        "outcome": outcome,
        "reasons": [] if outcome == schemas.PASS else [f"planted {label} finding"],
        "evidence_ref": f"evidence://{label}",
        "evidence_sha256": digest(f"evidence-{label}"),
    }


class C3EpycCompilerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = P.compile_plan()
        self.cases = C.epyc_op_suite()

    def surface(self, case, index: int) -> dict:
        return {
            "case_id": case.case_id,
            "device_id": "ROCm0",
            "model_sha256": digest("model"),
            "quant": "Q4_K" if "dequant" in case.case_id else "bf16",
            "operation": case.operator_family,
            "shape": [index + 1, 16, 512, 64],
            "dtype": "bf16",
            "tensor_manifest_sha256": digest(f"tensor-manifest-{index}"),
            "recipe_id": f"epyc.op.case-{index}.v1",
            "recipe_sha256": digest(f"recipe-{index}"),
            "harness_build_sha256": digest("harness"),
            "factors": {"graphs": "off", "stream_sync": "full_device", "warmup": 10},
        }

    def observed_case(self, case, index: int) -> dict:
        provider = (C.LLAMA_CPP_PRODUCTION_V9 if index == 2
                    else C.TORCH_ROCM_COMPILE)
        implementation_sha256 = digest(f"vendor-{index}")
        production_baseline = ({
            "branch": C.PRODUCTION_V9_BRANCH,
            "source_commit": C.PRODUCTION_V9_COMMIT,
            "version": C.PRODUCTION_V9_VERSION,
            "binary_sha256": implementation_sha256,
            "linkage_sha256": digest("production-v9-linkage"),
            "attestation_ref": C.PRODUCTION_V9_FREEZE_ATTESTATION_REF,
            "attestation_sha256": C.PRODUCTION_V9_FREEZE_ATTESTATION_SHA256,
        } if provider == C.LLAMA_CPP_PRODUCTION_V9 else None)
        return {
            "case_id": case.case_id,
            "state": "observed",
            "surface": self.surface(case, index),
            "vendor_observations": [{
                "provider": provider,
                "implementation_sha256": implementation_sha256,
                "samples_ns": [100.0 + index, 101.0 + index, 99.0 + index],
                "evidence_ref": f"evidence://vendor-{index}",
                "evidence_sha256": digest(f"vendor-evidence-{index}"),
                **({"production_baseline": production_baseline}
                   if production_baseline is not None else {}),
            }],
            "candidate_observation": {
                "provider": C.CANDIDATE_PROVIDER,
                "implementation_sha256": digest(f"candidate-{index}"),
                "samples_ns": [80.0 + index, 81.0 + index, 79.0 + index],
                "evidence_ref": f"evidence://candidate-{index}",
                "evidence_sha256": digest(f"candidate-evidence-{index}"),
            },
            "correctness": evidence_check(f"correctness-{index}"),
            "integrity": evidence_check(f"integrity-{index}"),
        }

    def whole_model(self, target_index: int = 0) -> dict:
        target = self.cases[target_index]
        surface = {
            "workload": {
                "workload_id": "epyc.production.capture.v1",
                "model_sha256": digest("model"),
                "tensor_manifest_sha256": digest("whole-tensor-manifest"),
                "capture_receipt_ref": "evidence://capture",
                "capture_receipt_sha256": digest("capture-receipt"),
            },
            "device_id": "ROCm0", "quant": "bf16",
            "recipe_id": "epyc.whole-model.v1",
            "recipe_sha256": digest("whole-recipe"),
            "factors": {"graphs": "off", "stage": "prefill", "warmup": 10},
        }
        return {
            "state": "observed", "target_case_id": target.case_id,
            "surface": surface,
            "integration": {
                "candidate_branch": "ak/c3-integrated-test",
                "production_base_commit": C.PRODUCTION_V9_COMMIT,
                "candidate_source_commit": digest("source-commit")[:40],
                "patch_bundle_sha256": digest("patch"),
                "candidate_source_sha256": digest(f"candidate-{target_index}"),
                "candidate_build_sha256": digest("candidate-build"),
                "candidate_binary_sha256": digest("candidate-binary"),
                "candidate_linkage_sha256": digest("candidate-linkage"),
                "toolchain_manifest_sha256": digest("candidate-toolchain"),
                "isolation_root": "/mnt/raid0/llm/autokernel/c3/test-candidate",
                "receipt_ref": "evidence://integration",
                "receipt_sha256": digest("integration-receipt"),
                "source_tree": "llama.cpp", "backend": "llama_gpu",
                "tree_clean": True, "ancestry_clean": True,
            },
            "anchor": {
                "arm": "unpatched_anchor",
                "build_sha256": digest("anchor-build"),
                "binary_sha256": digest("anchor-binary"),
                "samples_ns": [1000.0, 1001.0, 999.0],
                "evidence_ref": "evidence://whole-anchor",
                "evidence_sha256": digest("whole-anchor"),
            },
            "candidate": {
                "arm": "integrated_candidate",
                "build_sha256": digest("candidate-build"),
                "binary_sha256": digest("candidate-binary"),
                "samples_ns": [900.0, 901.0, 899.0],
                "evidence_ref": "evidence://whole-candidate",
                "evidence_sha256": digest("whole-candidate"),
            },
            "correctness": evidence_check("whole-correctness"),
            "integrity": evidence_check("whole-integrity"),
        }

    def observed_input(self, target_index: int = 0) -> dict:
        return {
            "schema": P.INPUT_SCHEMA,
            "plan_sha256": self.plan["plan_sha256"],
            "policy": dict(self.plan["policy"]),
            "cases": [self.observed_case(case, index)
                      for index, case in enumerate(self.cases)],
            "whole_model": self.whole_model(target_index),
        }

    def unavailable_input(self) -> dict:
        return {
            "schema": P.INPUT_SCHEMA,
            "plan_sha256": self.plan["plan_sha256"],
            "policy": dict(self.plan["policy"]),
            "cases": [{"case_id": case.case_id, "state": "unavailable",
                       "reason": "empirical evidence has not been captured"}
                      for case in self.cases],
            "whole_model": {"state": "unavailable",
                            "reason": "matched whole-model rebench is absent"},
        }

    def test_plan_is_exact_hash_bound_non_numeric_and_names_real_entrypoints(self):
        self.assertEqual(self.plan, P.compile_plan())
        self.assertRegex(self.plan["compiler_source_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(self.plan["contract_source_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(len(self.plan["cases"]), 3)
        self.assertEqual(
            [row["case_id"] for row in self.plan["cases"]],
            [case.case_id for case in self.cases])
        apex = self.plan["runner_bindings"][C.APEX_PYTHON_OVERLAY]
        self.assertEqual(apex["revision"], C.PINNED_APEX_REVISION)
        self.assertEqual(apex["capture"]["python_entrypoint"],
                         "scripts.kernel_rnd.autokernel.evaluator."
                         "c3_apex_runner.execute_trace")
        self.assertEqual(apex["capture"]["pinned_downstream_entrypoint"],
                         "pipeline.kernel_tracing.runner.run_trace_kernel")
        self.assertIn("c3_apex_case_mapping.v2", apex["capture"]["cli"])
        self.assertEqual(self.plan["external_artifacts"][0]["artifact_id"],
                         "c3_apex_case_mapping")
        self.assertIn("_run_final_benchmark",
                      apex["whole_model"]["python_entrypoint"])
        self.assertIn("no system C++", apex["boundary"])
        dequant = self.plan["cases"][2]
        self.assertEqual(dequant["runner_binding_id"], C.EPYC_EXPERIMENTAL_BINARY)
        baseline = self.plan["runner_bindings"][C.EPYC_EXPERIMENTAL_BINARY]["baseline"]
        self.assertEqual(baseline["provider"], C.LLAMA_CPP_PRODUCTION_V9)
        self.assertEqual(baseline["source_commit"], C.PRODUCTION_V9_COMMIT)
        self.assertEqual(baseline["attestation_sha256"],
                         C.PRODUCTION_V9_FREEZE_ATTESTATION_SHA256)
        rendered = json.dumps(self.plan)
        self.assertNotIn("samples_ns", rendered)
        self.assertNotIn('"speedup":', rendered)
        self.assertNotIn('"fast_p": {"value"', rendered)

        mutated = P.compile_plan()
        mutated["runner_bindings"][C.APEX_PYTHON_OVERLAY]["revision"] = "changed"
        self.assertEqual(P.compile_plan(), self.plan)

    def test_unavailable_receipt_withholds_all_derived_values(self):
        receipt = P.compile_receipt(self.unavailable_input())
        self.assertIsNone(receipt["fast_p"]["value"])
        self.assertEqual(receipt["fast_p"]["scored_cases"], 0)
        self.assertIsNone(receipt["whole_model_exit"]["speedup"])
        self.assertEqual(receipt["whole_model_exit"]["check"]["outcome"],
                         schemas.COULD_NOT_CHECK)
        self.assertFalse(receipt["promotion_authorized"])

    def test_observed_receipt_validates_and_derives_fast_p_and_whole_model(self):
        payload = self.observed_input()
        receipt = P.compile_receipt(payload)
        self.assertEqual(receipt, P.compile_receipt(copy.deepcopy(payload)))
        self.assertEqual(receipt["fast_p"]["value"], 1.0)
        self.assertEqual(receipt["fast_p"]["scored_cases"], 3)
        self.assertGreater(receipt["whole_model_exit"]["speedup"], 1.0)
        self.assertEqual(receipt["whole_model_exit"]["check"]["outcome"],
                         schemas.PASS)
        self.assertRegex(receipt["receipt_sha256"], r"^[0-9a-f]{64}$")

    def test_dequant_uses_experimental_binary_not_apex_overlay(self):
        receipt = P.compile_receipt(self.observed_input(target_index=2))
        self.assertEqual(receipt["whole_model_exit"]["target_case_id"],
                         "epyc.dequant.q4_k_decode_gemv")
        payload = self.observed_input(target_index=2)
        payload["whole_model"]["integration"] = {
            "runner_id": C.EPYC_EXPERIMENTAL_BINARY,
            "runner_revision": digest("source-commit")[:40],
            "patch_bundle_sha256": digest("patch"),
            "candidate_source_sha256": digest("candidate-2"),
            "candidate_build_sha256": digest("candidate-build"),
            "candidate_binary_sha256": digest("candidate-binary"),
            "receipt_ref": "evidence://diagnostic-integration",
            "receipt_sha256": digest("diagnostic-integration"),
        }
        receipt = P.compile_receipt(payload)
        self.assertIsNone(receipt["whole_model_exit"]["speedup"])
        self.assertEqual(receipt["whole_model_exit"]["check"]["outcome"],
                         schemas.COULD_NOT_CHECK)

    def test_plan_case_and_evidence_identity_tampering_refuse(self):
        payload = self.observed_input()
        payload["plan_sha256"] = digest("other-plan")
        with self.assertRaisesRegex(P.C3CompilerError, "different plan"):
            P.compile_receipt(payload)

        payload = self.observed_input()
        payload["cases"][0]["case_id"], payload["cases"][1]["case_id"] = (
            payload["cases"][1]["case_id"], payload["cases"][0]["case_id"])
        with self.assertRaisesRegex(P.C3CompilerError, "order/identity"):
            P.compile_receipt(payload)

        payload = self.observed_input()
        del payload["cases"][0]["correctness"]["evidence_sha256"]
        with self.assertRaisesRegex(P.C3CompilerError, "fields differ"):
            P.compile_receipt(payload)

    def test_eager_baseline_and_candidate_source_mismatch_refuse(self):
        payload = self.observed_input()
        payload["cases"][0]["vendor_observations"][0]["provider"] = "torch_eager"
        with self.assertRaisesRegex(C.C3ContractError, "unsupported timing provider"):
            P.compile_receipt(payload)

        payload = self.observed_input()
        payload["whole_model"]["integration"]["candidate_source_sha256"] = digest("other")
        with self.assertRaisesRegex(C.IdentityMismatch, "different candidate source"):
            P.compile_receipt(payload)

    def test_dequant_baseline_missing_or_wrong_frozen_identity_refuses(self):
        payload = self.observed_input()
        del payload["cases"][2]["vendor_observations"][0]["production_baseline"]
        with self.assertRaisesRegex(P.C3CompilerError, "must be an object"):
            P.compile_receipt(payload)

        payload = self.observed_input()
        payload["cases"][2]["vendor_observations"][0]["production_baseline"][
            "source_commit"] = "f" * 40
        with self.assertRaisesRegex(C.IdentityMismatch, "identity drifted"):
            P.compile_receipt(payload)

        payload = self.observed_input()
        row = payload["cases"][2]["vendor_observations"][0]
        row["provider"] = C.TORCH_ROCM_COMPILE
        del row["production_baseline"]
        with self.assertRaisesRegex(C.C3ContractError, "one observation per provider"):
            P.compile_receipt(payload)

    def test_direct_backend_and_cli_emit_same_plan(self):
        self.assertEqual(P.C3EpycBackend().compile_plan(), self.plan)
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(P.main(["plan"]), 0)
        self.assertEqual(json.loads(output.getvalue()), self.plan)


if __name__ == "__main__":
    unittest.main()
