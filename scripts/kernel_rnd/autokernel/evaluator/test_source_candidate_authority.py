"""Inference-free tests for source-candidate correctness authority."""
from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

_KERNEL_RND = str(Path(__file__).resolve().parents[2])
if _KERNEL_RND not in sys.path:
    sys.path.insert(0, _KERNEL_RND)

from autokernel import schemas  # noqa: E402
from autokernel.evaluator import api  # noqa: E402
from autokernel.evaluator import correctness as C  # noqa: E402
from autokernel.evaluator import source_candidate_authority as A  # noqa: E402
from autokernel.evaluator import sensitivity as SENS  # noqa: E402
from autokernel.evaluator.test_correctness import evidence, policy, request  # noqa: E402


def sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def sensitivity_report(outcome: str = schemas.PASS) -> SENS.SensitivityReport:
    reasons = () if outcome == schemas.PASS else ("planted sensitivity finding",)
    return SENS.SensitivityReport(
        suite_version="0db32c06e", units=(),
        check=schemas.Check(outcome, reasons))


def provenance(label: str, *, bundle: str = sha("evaluator-bundle"),
               capture_mode: str = "measured") -> A.EvidenceProvenance:
    return A.EvidenceProvenance(
        evidence_ref=f"evidence://{label}", evidence_sha256=sha(label),
        evaluator_bundle_sha256=bundle, capture_mode=capture_mode)


def bindings(*, source: str, bundle: str, capture_mode: str = "measured") -> tuple:
    return (
        A.bind_sensitivity(
            sensitivity_report(), candidate_source_sha256=source,
            provenance=provenance("sensitivity", bundle=bundle,
                                  capture_mode=capture_mode)),
        A.bind_oracle_check(
            "hostile_distributions", schemas.Check(schemas.PASS),
            suite_version="0db32c06e", candidate_source_sha256=source,
            provenance=provenance("hostile", bundle=bundle,
                                  capture_mode=capture_mode)),
        A.bind_oracle_check(
            "checker_isolation", schemas.Check(schemas.PASS),
            suite_version="0db32c06e", candidate_source_sha256=source,
            provenance=provenance("checker", bundle=bundle,
                                  capture_mode=capture_mode)),
    )


class SourceCandidateAuthorityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.request = request()
        self.source = self.request.artifact.source_sha256
        self.bundle = self.request.evaluator.bundle_sha256

    def evaluate(self, prerequisites=()) -> C.T0Report:
        return C.evaluate_t0(
            self.request,
            evidence(source_candidate=True, source_prerequisites=tuple(prerequisites)),
            policy())

    def gate(self, report: C.T0Report, gate_id: str):
        return next(gate for gate in report.gates if gate.gate_id == gate_id)

    def test_no_evidence_fails_closed_before_speed_authority(self):
        report = self.evaluate()
        op_gate = self.gate(report, C.GID_OP_UNITS)
        ref_gate = self.gate(report, C.GID_EXACT_REFERENCE)
        self.assertEqual(op_gate.check.outcome, schemas.COULD_NOT_CHECK)
        self.assertEqual(ref_gate.check.outcome, schemas.COULD_NOT_CHECK)
        self.assertIn("no hash-bound input_sensitivity", " ".join(op_gate.check.reasons))
        self.assertIn("no hash-bound checker_isolation", " ".join(ref_gate.check.reasons))

    def test_complete_exact_bindings_pass_and_enter_event_vectors(self):
        report = self.evaluate(bindings(source=self.source, bundle=self.bundle))
        self.assertEqual(self.gate(report, C.GID_OP_UNITS).check.outcome, schemas.PASS)
        self.assertEqual(
            self.gate(report, C.GID_EXACT_REFERENCE).check.outcome, schemas.PASS)
        measurements = (
            self.gate(report, C.GID_OP_UNITS).measurements +
            self.gate(report, C.GID_EXACT_REFERENCE).measurements)
        self.assertEqual(
            {item["prerequisite_id"] for item in measurements},
            set(C.SOURCE_PREREQUISITE_IDS))
        self.assertTrue(all(item["candidate_source_sha256"] == self.source
                            for item in measurements))
        self.assertTrue(all(item["evaluator_bundle_sha256"] == self.bundle
                            for item in measurements))
        self.assertTrue(all(item["evidence_ref"].startswith("evidence://")
                            and len(item["evidence_sha256"]) == 64
                            for item in measurements))
        vector = api._vector(report.gates, api.GATE_CORRECTNESS)
        self.assertEqual(
            {item["prerequisite_id"]
             for item in vector[C.GID_OP_UNITS]["measurements"]},
            {"input_sensitivity", "hostile_distributions"})
        self.assertEqual(
            vector[C.GID_EXACT_REFERENCE]["measurements"][0]["prerequisite_id"],
            "checker_isolation")

    def test_dry_run_is_could_not_check_and_cannot_mint_pass(self):
        dry = bindings(
            source=self.source, bundle=self.bundle, capture_mode="dry_run")
        report = self.evaluate(dry)
        self.assertEqual(self.gate(report, C.GID_OP_UNITS).check.outcome,
                         schemas.COULD_NOT_CHECK)
        self.assertIn("dry-run", " ".join(
            self.gate(report, C.GID_OP_UNITS).check.reasons))
        with self.assertRaisesRegex(ValueError, "dry-run.*cannot carry PASS"):
            C.SourcePrerequisiteEvidence(
                prerequisite_id="input_sensitivity",
                candidate_source_sha256=self.source,
                evaluator_bundle_sha256=self.bundle,
                suite_version="0db32c06e", producer_id="trusted_evaluator",
                capture_mode="dry_run", evidence_ref="evidence://dry",
                evidence_sha256=sha("dry"), check=schemas.Check(schemas.PASS))

    def test_source_or_evaluator_identity_mismatch_fails_closed(self):
        wrong_source = bindings(source=sha("other-source"), bundle=self.bundle)
        report = self.evaluate(wrong_source)
        self.assertEqual(self.gate(report, C.GID_OP_UNITS).check.outcome,
                         schemas.COULD_NOT_CHECK)
        self.assertIn("different candidate source", " ".join(
            self.gate(report, C.GID_OP_UNITS).check.reasons))

        wrong_bundle = bindings(source=self.source, bundle=sha("other-bundle"))
        report = self.evaluate(wrong_bundle)
        self.assertEqual(self.gate(report, C.GID_EXACT_REFERENCE).check.outcome,
                         schemas.COULD_NOT_CHECK)
        self.assertIn("different evaluator bundle", " ".join(
            self.gate(report, C.GID_EXACT_REFERENCE).check.reasons))

    def test_standalone_report_is_not_t0_evidence(self):
        with self.assertRaisesRegex(TypeError, "SourcePrerequisiteEvidence"):
            evidence(source_candidate=True,
                     source_prerequisites=(sensitivity_report(),))

    def test_parameter_candidate_cannot_smuggle_source_prerequisites(self):
        with self.assertRaisesRegex(ValueError, "parameter/no-source"):
            evidence(
                source_candidate=False,
                source_prerequisites=bindings(source=self.source, bundle=self.bundle))


if __name__ == "__main__":
    unittest.main()
