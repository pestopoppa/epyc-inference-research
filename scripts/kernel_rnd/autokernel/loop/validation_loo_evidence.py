"""Immutable exact references for candidate LOO evidence without store scans."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .. import schemas
from . import candidate_manifest as cm
from . import measurement_capture as mc
from .validation_objective_decision import (
    ObjectiveDecisionReference,
    reopen_readiness_decision,
)


EVIDENCE_SCHEMA = "epyc.autokernel.validation_loo_evidence.v1"
REFERENCE_SCHEMA = "epyc.autokernel.validation_loo_evidence_reference.v1"
PRODUCER_ID = "autokernel.loop.validation_loo_evidence/v1"


class LOOEvidenceError(ValueError):
    """LOO evidence is missing, replaced, or bound to different semantics."""


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not schemas.SHA256_RE.fullmatch(value) \
            or schemas.is_placeholder_digest(value):
        raise LOOEvidenceError(f"{label} must be a non-placeholder SHA-256")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LOOEvidenceError(f"{label} must be non-empty text")
    return value


def _exact(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise LOOEvidenceError(f"{label} has missing or unknown fields")
    return value


@dataclass(frozen=True)
class RowEvidencePointer:
    receipt: cm.RowReceipt

    @classmethod
    def from_dict(cls, value: Any) -> "RowEvidencePointer":
        row = _exact(value, {"receipt"}, "LOO row evidence pointer")
        return cls(cm.RowReceipt.from_dict(row["receipt"]))

    @property
    def row_id(self) -> str:
        return self.receipt.row_id

    @property
    def evidence_digest(self) -> str:
        return self.receipt.native_evidence_digest

    @property
    def locator(self) -> str:
        return self.receipt.native_evidence_ref

    @property
    def sha256(self) -> str:
        return self.receipt.native_evidence_digest

    def to_dict(self) -> dict[str, Any]:
        return {"receipt": self.receipt.to_dict()}


@dataclass(frozen=True)
class LOOEvidenceReference:
    evidence_digest: str
    locator: str
    sha256: str

    @classmethod
    def from_dict(cls, value: Any) -> "LOOEvidenceReference":
        row = _exact(value, {"schema", "evidence_digest", "locator", "sha256"},
                     "LOO evidence reference")
        if row["schema"] != REFERENCE_SCHEMA:
            raise LOOEvidenceError("LOO evidence reference schema is unsupported")
        return cls(_sha(row["evidence_digest"], "LOO evidence digest"),
                   _text(row["locator"], "LOO evidence locator"),
                   _sha(row["sha256"], "LOO evidence artifact"))

    def to_dict(self) -> dict[str, str]:
        return {"schema": REFERENCE_SCHEMA, **self.__dict__}


def evidence_body(*, plan: cm.LOOPlan, candidate: cm.CandidateManifest,
                  row_set: cm.RequiredRowSet,
                  row_evidence: Sequence[RowEvidencePointer],
                  objective_decisions: Sequence[ObjectiveDecisionReference]) -> dict[str, Any]:
    plan = cm.LOOPlan.from_dict(plan.to_dict())
    candidate = candidate.validated()
    row_set = cm.RequiredRowSet.from_dict(row_set.to_dict())
    rows = tuple(RowEvidencePointer.from_dict(item.to_dict()) for item in row_evidence)
    decisions = tuple(ObjectiveDecisionReference.from_dict(item.to_dict())
                      for item in objective_decisions)
    if plan.status != "planned" or plan.derived_manifest_digest is None:
        raise LOOEvidenceError("only identifiable planned LOO can carry evidence")
    if plan.candidate_manifest_digest != candidate.manifest_digest:
        raise LOOEvidenceError("LOO plan names a different candidate")
    required_rows = {item.row_id for item in row_set.rows if item.required}
    if {item.row_id for item in rows} != required_rows \
            or len({item.row_id for item in rows}) != len(rows):
        raise LOOEvidenceError("LOO evidence must index every required row exactly once")
    if len({item.evidence_digest for item in rows}) != len(rows):
        raise LOOEvidenceError("LOO evidence reuses a row receipt digest")
    if len(decisions) != len(rows) \
            or len({item.decision_id for item in decisions}) != len(decisions):
        raise LOOEvidenceError("LOO needs one distinct objective decision per required row")
    return {"schema": EVIDENCE_SCHEMA, "producer": PRODUCER_ID,
            "plan": plan.to_dict(),
            "candidate_manifest_digest": candidate.manifest_digest,
            "row_set": row_set.to_dict(),
            "row_evidence": [item.to_dict() for item in sorted(rows,
                                                               key=lambda item: item.row_id)],
            "objective_decisions": [item.to_dict() for item in sorted(
                decisions, key=lambda item: item.objective_digest)]}


def evidence_digest(**kwargs: Any) -> str:
    return schemas.content_hash(evidence_body(**kwargs))


def _verify_referenced_closure(*, store: mc.ArtifactStore, body: Mapping[str, Any]) -> None:
    """Reopen every exact row/decision reference; do not trust the index alone."""
    candidate_digest = body["candidate_manifest_digest"]
    row_set = cm.RequiredRowSet.from_dict(body["row_set"])
    required = {item.row_id: item for item in row_set.rows if item.required}
    seen_decisions: dict[str, Mapping[str, Any]] = {}
    for reference in body["objective_decisions"]:
        decision = reopen_readiness_decision(
            store=store, reference=ObjectiveDecisionReference.from_dict(reference))
        binding = decision["candidate_binding"]
        if (binding["candidate_manifest_digest"] != candidate_digest
                or binding["row_set_digest"] != row_set.row_set_digest
                or binding["row_id"] not in required
                or decision["objective_digest"]
                != required[binding["row_id"]].objective_digest
                or binding["row_id"] in seen_decisions):
            raise LOOEvidenceError("objective decision does not bind an exact required LOO row")
        seen_decisions[binding["row_id"]] = decision
    for pointer_value in body["row_evidence"]:
        pointer = RowEvidencePointer.from_dict(pointer_value)
        try:
            bundle = mc._plain(store.read(pointer.locator, pointer.sha256))
            store.verify("candidate-validation-row", bundle)
        except (ValueError, mc.CaptureError, mc.SecureRuntimeError) as exc:
            raise LOOEvidenceError("LOO row evidence cannot be reopened") from exc
        try:
            sealed_candidate = cm.CandidateManifest.from_dict(bundle["candidate"])
            sealed_comparator = cm.CandidateManifest.from_dict(bundle["comparator"])
            sealed_batch = cm.ValidationBatch.from_dict(bundle["batch"])
            sealed_row_set = cm.RequiredRowSet.from_dict(bundle["row_set"])
            sealed_row = cm.ValidationRow.from_dict(bundle["row"])
        except (KeyError, TypeError, ValueError, cm.CandidateError) as exc:
            raise LOOEvidenceError("LOO row evidence closure is malformed") from exc
        if (bundle.get("schema") != "epyc.autokernel.validation_native_pair.v1"
                or bundle.get("fixture_only_authority") is not True):
            raise LOOEvidenceError(
                "LOO v1 closure is fixture-only pending native-v2 authority")
        if (pointer.sha256 != pointer.evidence_digest
                or sealed_row.row_id != pointer.row_id
                or sealed_candidate.manifest_digest != candidate_digest
                or sealed_row_set.row_set_digest != row_set.row_set_digest
                or not sealed_batch.matches_receipt(pointer.receipt)
                or pointer.receipt.candidate_manifest_digest
                != sealed_candidate.manifest_digest
                or pointer.receipt.comparator_manifest_digest
                != sealed_comparator.manifest_digest
                or pointer.receipt.row_set_digest != sealed_row_set.row_set_digest
                or pointer.receipt.intended_use != "validate"
                or pointer.receipt.use_disposition != "permitted"
                or pointer.row_id not in seen_decisions):
            raise LOOEvidenceError("LOO row evidence differs from its exact semantic binding")
        decision = seen_decisions[pointer.row_id]
        binding = decision["candidate_binding"]
        if (binding != {
                "batch_id": pointer.receipt.batch_id,
                "row_id": pointer.receipt.row_id,
                "row_set_digest": pointer.receipt.row_set_digest,
                "candidate_manifest_digest": pointer.receipt.candidate_manifest_digest,
                "comparator_manifest_digest": pointer.receipt.comparator_manifest_digest}):
            raise LOOEvidenceError(
                "objective decision batch/comparator differs from exact row receipt")
        native = decision["native_binding"]
        sealed_native = bundle.get("native")
        calibration = bundle.get("calibration")
        try:
            anchor = sealed_native["anchor"]
            candidate_arm = sealed_native["candidate"]
            derived_native = {
                "plan_digest": bundle["plan_digest"],
                "anchor_measurement_id": anchor["measurement_id"],
                "anchor_carrier_digest": anchor["payload"]["carrier"]["carrier_digest"],
                "candidate_measurement_id": candidate_arm["measurement_id"],
                "candidate_carrier_digest":
                    candidate_arm["payload"]["carrier"]["carrier_digest"],
                "calibration_digest": calibration["digest"],
                "instrument_identity_digest": sealed_row.instrument_digest,
            }
        except (KeyError, TypeError) as exc:
            raise LOOEvidenceError("LOO native row binding is incomplete") from exc
        if any(native[key] != value for key, value in derived_native.items()):
            raise LOOEvidenceError(
                "objective decision native binding differs from sealed row evidence")


def seal_loo_evidence(*, store: mc.ArtifactStore, result: cm.LOOResult,
                      plan: cm.LOOPlan, candidate: cm.CandidateManifest,
                      row_set: cm.RequiredRowSet,
                      row_evidence: Sequence[RowEvidencePointer],
                      objective_decisions: Sequence[ObjectiveDecisionReference]
                      ) -> LOOEvidenceReference:
    result = cm.LOOResult.from_dict(result.to_dict())
    body = evidence_body(plan=plan, candidate=candidate, row_set=row_set,
                         row_evidence=row_evidence,
                         objective_decisions=objective_decisions)
    _verify_referenced_closure(store=store, body=body)
    digest = schemas.content_hash(body)
    if (result.evidence_digest != digest
            or result.plan_digest != plan.plan_digest
            or result.candidate_manifest_digest != candidate.manifest_digest
            or result.keep_id != plan.keep_id
            or result.derived_manifest_digest != plan.derived_manifest_digest
            or result.row_set_digest != row_set.row_set_digest
            or set(result.receipt_digests)
               != {item.evidence_digest for item in row_evidence}
            or result.disposition == "inconclusive"
            or result.deletion_authorized):
        raise LOOEvidenceError("LOO result differs from its exact evidence index")
    artifact = store.write("validation-loo-evidence", body)
    return LOOEvidenceReference(digest, artifact.locator, artifact.sha256)


def reopen_loo_evidence(*, store: mc.ArtifactStore,
                        reference: LOOEvidenceReference,
                        result: cm.LOOResult) -> Mapping[str, Any]:
    reference = LOOEvidenceReference.from_dict(reference.to_dict())
    result = cm.LOOResult.from_dict(result.to_dict())
    try:
        body = mc._plain(store.read(reference.locator, reference.sha256))
    except (ValueError, mc.CaptureError, mc.SecureRuntimeError) as exc:
        raise LOOEvidenceError("LOO evidence artifact cannot be reopened") from exc
    expected_fields = {"schema", "producer", "plan", "candidate_manifest_digest",
                       "row_set", "row_evidence", "objective_decisions"}
    _exact(body, expected_fields, "LOO evidence")
    if body["schema"] != EVIDENCE_SCHEMA or body["producer"] != PRODUCER_ID:
        raise LOOEvidenceError("LOO evidence schema/producer is unsupported")
    plan = cm.LOOPlan.from_dict(body["plan"])
    row_set = cm.RequiredRowSet.from_dict(body["row_set"])
    rows = tuple(RowEvidencePointer.from_dict(item) for item in body["row_evidence"])
    decisions = tuple(ObjectiveDecisionReference.from_dict(item)
                      for item in body["objective_decisions"])
    _verify_referenced_closure(store=store, body=body)
    digest = schemas.content_hash(body)
    if (digest != reference.evidence_digest or digest != result.evidence_digest
            or plan.plan_digest != result.plan_digest
            or plan.keep_id != result.keep_id
            or body["candidate_manifest_digest"] != result.candidate_manifest_digest
            or row_set.row_set_digest != result.row_set_digest
            or set(result.receipt_digests) != {item.evidence_digest for item in rows}
            or len(decisions) != len(rows)):
        raise LOOEvidenceError("LOO evidence/result/reference binding changed")
    store.verify("validation-loo-evidence", body)
    return body


class LOOEvidenceIndex:
    """Exact caller-owned index; lookups never guess filenames or scan a store."""

    def __init__(self, references: Sequence[LOOEvidenceReference]):
        items = tuple(LOOEvidenceReference.from_dict(item.to_dict()) for item in references)
        self._references = {item.evidence_digest: item for item in items}
        if len(self._references) != len(items):
            raise LOOEvidenceError("LOO evidence index repeats a digest")

    def exact(self, digest: str) -> LOOEvidenceReference:
        digest = _sha(digest, "LOO evidence digest")
        try:
            return self._references[digest]
        except KeyError as exc:
            raise LOOEvidenceError("LOO evidence digest is not in the exact index") from exc


__all__ = ["LOOEvidenceError", "LOOEvidenceIndex", "LOOEvidenceReference",
           "RowEvidencePointer", "evidence_body", "evidence_digest",
           "reopen_loo_evidence", "seal_loo_evidence"]
