"""Seal the actual AutoKernel readiness result without upgrading its authority."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .. import schemas
from ..release import readiness
from . import candidate_manifest as cm
from . import measurement_capture as mc


DECISION_SCHEMA = "epyc.autokernel.validation_objective_decision.v1"
REFERENCE_SCHEMA = "epyc.autokernel.validation_objective_decision_reference.v1"
PRODUCER_ID = "autokernel.loop.validation_objective_decision/v1"
READINESS_SOURCE_SHA256 = "5f3ff5b1236f575b3fab90ac58c07e388d490885b19e0c3e94bf0fdb03e76aa5"


class ObjectiveDecisionError(ValueError):
    """An objective result cannot be bound to this candidate validation row."""


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not schemas.SHA256_RE.fullmatch(value) \
            or schemas.is_placeholder_digest(value):
        raise ObjectiveDecisionError(f"{label} must be a non-placeholder SHA-256")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ObjectiveDecisionError(f"{label} must be non-empty text")
    return value


def _exact(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ObjectiveDecisionError(f"{label} has missing or unknown fields")
    return value


@dataclass(frozen=True)
class NativePairBinding:
    plan_digest: str
    anchor_measurement_id: str
    anchor_carrier_digest: str
    candidate_measurement_id: str
    candidate_carrier_digest: str
    calibration_digest: str
    instrument_identity_digest: str
    instrument_locator: str
    instrument_artifact_sha256: str

    @classmethod
    def from_dict(cls, value: Any) -> "NativePairBinding":
        fields = {"plan_digest", "anchor_measurement_id", "anchor_carrier_digest",
                  "candidate_measurement_id", "candidate_carrier_digest",
                  "calibration_digest", "instrument_identity_digest",
                  "instrument_locator", "instrument_artifact_sha256"}
        row = _exact(value, fields, "native pair binding")
        return cls(_sha(row["plan_digest"], "plan digest"),
                   _sha(row["anchor_measurement_id"], "anchor measurement id"),
                   _sha(row["anchor_carrier_digest"], "anchor carrier digest"),
                   _sha(row["candidate_measurement_id"], "candidate measurement id"),
                   _sha(row["candidate_carrier_digest"], "candidate carrier digest"),
                   _sha(row["calibration_digest"], "calibration digest"),
                   _sha(row["instrument_identity_digest"], "instrument identity"),
                   _text(row["instrument_locator"], "instrument locator"),
                   _sha(row["instrument_artifact_sha256"], "instrument artifact"))

    def to_dict(self) -> dict[str, str]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class ObjectiveDecisionReference:
    decision_id: str
    objective_digest: str
    disposition: str
    locator: str
    sha256: str

    @classmethod
    def from_dict(cls, value: Any) -> "ObjectiveDecisionReference":
        row = _exact(value, {"schema", "decision_id", "objective_digest",
                             "disposition", "locator", "sha256"},
                     "objective decision reference")
        if row["schema"] != REFERENCE_SCHEMA \
                or row["disposition"] not in {"refused", "policy_undefined"}:
            raise ObjectiveDecisionError("objective decision reference is unsupported")
        return cls(_text(row["decision_id"], "decision id"),
                   _sha(row["objective_digest"], "objective digest"),
                   row["disposition"], _text(row["locator"], "decision locator"),
                   _sha(row["sha256"], "decision artifact digest"))

    def to_dict(self) -> dict[str, str]:
        return {"schema": REFERENCE_SCHEMA, **self.__dict__}


def _validated_decision(value: Any) -> Mapping[str, Any]:
    fields = {"schema", "producer", "decision_id", "source_identity",
              "source_result", "objective_digest", "requested_intended_use",
              "owner_scope", "source_standing", "disposition", "reasons",
              "decided_proposition", "candidate_binding", "native_binding"}
    row = _exact(value, fields, "objective decision")
    if row["schema"] != DECISION_SCHEMA or row["producer"] != PRODUCER_ID:
        raise ObjectiveDecisionError("objective decision schema/producer is unsupported")
    source = _exact(row["source_identity"], {"module_id", "source_sha256"},
                    "objective source identity")
    if source != {"module_id": readiness.MODULE_ID,
                  "source_sha256": READINESS_SOURCE_SHA256}:
        raise ObjectiveDecisionError("objective decision source identity changed")
    result = _exact(row["source_result"], {"locator", "sha256", "content_digest"},
                    "objective source result")
    _text(result["locator"], "objective source result locator")
    _sha(result["sha256"], "objective source result artifact")
    _sha(result["content_digest"], "objective source result content")
    _sha(row["objective_digest"], "objective digest")
    if row["requested_intended_use"] != "validate_production" \
            or row["owner_scope"] != "advisory_readiness_only" \
            or row["source_standing"] not in readiness.STANDINGS \
            or row["disposition"] not in {"refused", "policy_undefined"}:
        raise ObjectiveDecisionError("objective decision authority labels are invalid")
    if not isinstance(row["reasons"], (list, tuple)) or not row["reasons"] \
            or any(not isinstance(item, str) or not item for item in row["reasons"]):
        raise ObjectiveDecisionError("objective decision needs explicit reasons")
    _text(row["decided_proposition"], "decided proposition")
    candidate = _exact(row["candidate_binding"], {
        "batch_id", "row_id", "row_set_digest", "candidate_manifest_digest",
        "comparator_manifest_digest"}, "candidate binding")
    _text(candidate["batch_id"], "batch id")
    _text(candidate["row_id"], "row id")
    for key in ("row_set_digest", "candidate_manifest_digest",
                "comparator_manifest_digest"):
        _sha(candidate[key], key)
    NativePairBinding.from_dict(row["native_binding"])
    expected = schemas.content_hash({key: row[key] for key in row if key != "decision_id"})
    if row["decision_id"] != f"objective-{expected[:24]}":
        raise ObjectiveDecisionError("objective decision id does not rederive")
    return row


def seal_readiness_decision(*, store: mc.ArtifactStore,
                            signal: readiness.ReadinessSignal,
                            batch: cm.ValidationBatch, row: cm.ValidationRow,
                            row_set: cm.RequiredRowSet,
                            candidate: cm.CandidateManifest,
                            comparator: cm.CandidateManifest,
                            native: NativePairBinding) -> ObjectiveDecisionReference:
    """Bind the real reducer result; never turn advisory readiness into permission."""
    if not isinstance(signal, readiness.ReadinessSignal):
        raise ObjectiveDecisionError("signal must be the actual ReadinessSignal type")
    batch, candidate, comparator = batch.validated(), candidate.validated(), comparator.validated()
    row_set = cm.RequiredRowSet.from_dict(row_set.to_dict())
    row = cm.ValidationRow.from_dict(row.to_dict())
    native = NativePairBinding.from_dict(native.to_dict())
    try:
        cm._validate_batch_obligations(batch, row_set, candidate, comparator)
    except (cm.CandidateError, cm.TransitionError) as exc:
        raise ObjectiveDecisionError("objective decision batch closure is invalid") from exc
    objective_digest = schemas.content_hash(signal.objective.to_dict())
    if (row.objective_digest != objective_digest
            or signal.backend != {"cpu": "llama_cpu", "gpu": "llama_gpu"}[row.backend]
            or row.protocol_id not in set(signal.objective.protocol_by_phase.values())
            or batch.row_set_digest != row_set.row_set_digest
            or batch.candidate_manifest_digest != candidate.manifest_digest
            or batch.comparator_manifest_digest != comparator.manifest_digest
            or row.instrument_digest != native.instrument_identity_digest):
        raise ObjectiveDecisionError("readiness result does not bind the frozen validation row")
    source_body = signal.to_dict()
    source_artifact = store.write("validation-readiness-result", source_body)
    source_digest = schemas.content_hash(source_body)
    proposition = (f"{readiness.MODULE_ID} computed {signal.standing} for backend "
                   f"{signal.backend} objective {objective_digest}")
    if signal.standing == readiness.STANDING_MET:
        disposition = "policy_undefined"
        reasons = ["readiness is advisory (is_trigger=false); no owning production-validation "
                   "numerical policy/result is installed"]
    else:
        disposition = "refused"
        reasons = [f"actual readiness standing is {signal.standing}", *signal.blockers]
    body = {
        "schema": DECISION_SCHEMA, "producer": PRODUCER_ID,
        "source_identity": {"module_id": readiness.MODULE_ID,
                            "source_sha256": READINESS_SOURCE_SHA256},
        "source_result": {"locator": source_artifact.locator,
                          "sha256": source_artifact.sha256,
                          "content_digest": source_digest},
        "objective_digest": objective_digest,
        "requested_intended_use": "validate_production",
        "owner_scope": "advisory_readiness_only",
        "source_standing": signal.standing, "disposition": disposition,
        "reasons": reasons, "decided_proposition": proposition,
        "candidate_binding": {"batch_id": batch.batch_id, "row_id": row.row_id,
            "row_set_digest": row_set.row_set_digest,
            "candidate_manifest_digest": candidate.manifest_digest,
            "comparator_manifest_digest": comparator.manifest_digest},
        "native_binding": native.to_dict(),
    }
    body["decision_id"] = "objective-" + schemas.content_hash(body)[:24]
    _validated_decision(body)
    artifact = store.write("validation-objective-decision", body)
    return ObjectiveDecisionReference(body["decision_id"], objective_digest,
                                      disposition, artifact.locator, artifact.sha256)


def reopen_readiness_decision(*, store: mc.ArtifactStore,
                              reference: ObjectiveDecisionReference) -> Mapping[str, Any]:
    reference = ObjectiveDecisionReference.from_dict(reference.to_dict())
    try:
        decision = _validated_decision(mc._plain(
            store.read(reference.locator, reference.sha256)))
    except (ValueError, mc.CaptureError, mc.SecureRuntimeError) as exc:
        raise ObjectiveDecisionError("objective decision artifact cannot be reopened") from exc
    result_ref = decision["source_result"]
    try:
        result = mc._plain(store.read(result_ref["locator"], result_ref["sha256"]))
        store.verify("validation-readiness-result", result)
    except (ValueError, mc.CaptureError, mc.SecureRuntimeError) as exc:
        raise ObjectiveDecisionError("objective source result cannot be reopened") from exc
    if (schemas.content_hash(result) != result_ref["content_digest"]
            or result.get("reducer_id") != readiness.MODULE_ID
            or result.get("is_trigger") is not False
            or result.get("standing") != decision["source_standing"]
            or schemas.content_hash(result.get("objective")) != decision["objective_digest"]
            or reference.decision_id != decision["decision_id"]
            or reference.objective_digest != decision["objective_digest"]
            or reference.disposition != decision["disposition"]):
        raise ObjectiveDecisionError("objective decision reference/result binding changed")
    return decision


__all__ = ["NativePairBinding", "ObjectiveDecisionError",
           "ObjectiveDecisionReference", "reopen_readiness_decision",
           "seal_readiness_decision"]
