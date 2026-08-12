"""Bind offline ROCm correctness reducers into live source-candidate T0 authority.

The reducers remain inference-free and cannot rank or mutate a candidate.  Their
outputs become verdict-bearing only after this module binds them to the exact
candidate source, evaluator bundle, evidence object, suite version and capture
mode consumed by :func:`correctness.evaluate_t0`.
"""
from __future__ import annotations

from dataclasses import dataclass

from .. import schemas
from . import correctness, sensitivity


@dataclass(frozen=True)
class EvidenceProvenance:
    evidence_ref: str
    evidence_sha256: str
    evaluator_bundle_sha256: str
    capture_mode: str

    def __post_init__(self) -> None:
        schemas.require.str(self.evidence_ref, "provenance.evidence_ref")
        schemas.require.sha256(self.evidence_sha256, "provenance.evidence_sha256")
        schemas.require.sha256(
            self.evaluator_bundle_sha256, "provenance.evaluator_bundle_sha256")
        if self.capture_mode not in ("measured", "dry_run"):
            raise ValueError("provenance.capture_mode must be measured or dry_run")


def _bound_check(check: schemas.Check, *, capture_mode: str,
                 prerequisite_id: str) -> schemas.Check:
    if not isinstance(check, schemas.Check):
        raise TypeError(f"{prerequisite_id} reducer output must be a schemas.Check")
    if capture_mode == "dry_run":
        return schemas.Check(
            schemas.COULD_NOT_CHECK,
            (f"{prerequisite_id} was compiled from a dry run, not measured evidence",))
    return check


def bind_sensitivity(
        report: sensitivity.SensitivityReport, *, candidate_source_sha256: str,
        provenance: EvidenceProvenance) -> correctness.SourcePrerequisiteEvidence:
    """Bind a sensitivity report; the report alone has no T0 authority."""
    if not isinstance(report, sensitivity.SensitivityReport):
        raise TypeError("report must be a sensitivity.SensitivityReport")
    return correctness.SourcePrerequisiteEvidence(
        prerequisite_id="input_sensitivity",
        candidate_source_sha256=candidate_source_sha256,
        evaluator_bundle_sha256=provenance.evaluator_bundle_sha256,
        suite_version=report.suite_version,
        producer_id=sensitivity.TRUSTED_PRODUCER,
        capture_mode=provenance.capture_mode,
        evidence_ref=provenance.evidence_ref,
        evidence_sha256=provenance.evidence_sha256,
        check=_bound_check(
            report.check, capture_mode=provenance.capture_mode,
            prerequisite_id="input_sensitivity"))


def bind_oracle_check(
        prerequisite_id: str, check: schemas.Check, *, suite_version: str,
        candidate_source_sha256: str,
        provenance: EvidenceProvenance) -> correctness.SourcePrerequisiteEvidence:
    """Bind one hostile-distribution or checker-isolation reducer output."""
    if prerequisite_id not in ("hostile_distributions", "checker_isolation"):
        raise ValueError("oracle prerequisite must be hostile_distributions or checker_isolation")
    schemas.require.str(suite_version, "suite_version")
    return correctness.SourcePrerequisiteEvidence(
        prerequisite_id=prerequisite_id,
        candidate_source_sha256=candidate_source_sha256,
        evaluator_bundle_sha256=provenance.evaluator_bundle_sha256,
        suite_version=suite_version,
        producer_id=sensitivity.TRUSTED_PRODUCER,
        capture_mode=provenance.capture_mode,
        evidence_ref=provenance.evidence_ref,
        evidence_sha256=provenance.evidence_sha256,
        check=_bound_check(
            check, capture_mode=provenance.capture_mode,
            prerequisite_id=prerequisite_id))


__all__ = [
    "EvidenceProvenance", "bind_sensitivity", "bind_oracle_check",
]
