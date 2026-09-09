"""Immutable references to canonical ROOT projection-and-grade results.

This is a provenance carrier, not a grader.  Creation and replay live on the
hash-pinned :class:`PinnedRootProjection`, which reopens the native source and
calls the registered projector plus ROOT's sole ``grade`` function.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .. import schemas


RECEIPT_SCHEMA = "epyc.autokernel.canonical_claim_grade_receipt.v1"
REFERENCE_SCHEMA = "epyc.autokernel.canonical_claim_grade_receipt_reference.v1"
PAIR_REFERENCE_SCHEMA = "epyc.autokernel.canonical_claim_grade_receipt_pair.v1"
PRODUCER_ID = "autokernel.loop.validation_semantic_adapter/v1"


class ClaimReceiptError(ValueError):
    """A canonical receipt is malformed, replaced, or incompletely bound."""


def _sha(value: Any, label: str) -> str:
    if (not isinstance(value, str) or not schemas.SHA256_RE.fullmatch(value)
            or schemas.is_placeholder_digest(value)):
        raise ClaimReceiptError(f"{label} must be a non-placeholder SHA-256")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ClaimReceiptError(f"{label} must be non-empty text")
    return value


def _exact(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ClaimReceiptError(f"{label} has missing or unknown fields")
    return value


@dataclass(frozen=True)
class ClaimGradeReceiptReference:
    receipt_id: str
    locator: str
    sha256: str

    @classmethod
    def from_dict(cls, value: Any) -> "ClaimGradeReceiptReference":
        row = _exact(value, {"schema", "receipt_id", "locator", "sha256"},
                     "claim-grade receipt reference")
        if row["schema"] != REFERENCE_SCHEMA:
            raise ClaimReceiptError("claim-grade receipt reference schema is unsupported")
        return cls(_text(row["receipt_id"], "receipt id"),
                   _text(row["locator"], "receipt locator"),
                   _sha(row["sha256"], "receipt artifact"))

    def to_dict(self) -> dict[str, str]:
        return {"schema": REFERENCE_SCHEMA, **self.__dict__}


@dataclass(frozen=True)
class ClaimGradeReceiptPairReference:
    anchor: ClaimGradeReceiptReference
    candidate: ClaimGradeReceiptReference
    anchor_measurement_id: str
    candidate_measurement_id: str
    admissible_view_digest: str

    @classmethod
    def from_dict(cls, value: Any) -> "ClaimGradeReceiptPairReference":
        row = _exact(value, {"schema", "anchor", "candidate", "anchor_measurement_id",
                            "candidate_measurement_id", "admissible_view_digest"}, "claim-grade pair")
        if row["schema"] != PAIR_REFERENCE_SCHEMA:
            raise ClaimReceiptError("claim-grade pair schema is unsupported")
        return cls(ClaimGradeReceiptReference.from_dict(row["anchor"]),
                   ClaimGradeReceiptReference.from_dict(row["candidate"]),
                   _sha(row["anchor_measurement_id"], "anchor measurement"),
                   _sha(row["candidate_measurement_id"], "candidate measurement"),
                   _sha(row["admissible_view_digest"], "admissible view"))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": PAIR_REFERENCE_SCHEMA, "anchor": self.anchor.to_dict(),
                "candidate": self.candidate.to_dict(),
                "anchor_measurement_id": self.anchor_measurement_id,
                "candidate_measurement_id": self.candidate_measurement_id,
                "admissible_view_digest": self.admissible_view_digest}


def validate_receipt_body(value: Any) -> Mapping[str, Any]:
    from . import validation_projection_source as current
    if isinstance(value, Mapping) and value.get("schema") == current.RECEIPT_SCHEMA:
        try:
            return current.validate_current_receipt(value)
        except (ValueError, TypeError, KeyError) as exc:
            raise ClaimReceiptError("current canonical receipt is malformed") from exc
    fields = {"schema", "producer", "receipt_id", "source_identity",
              "source_event", "projection", "native_binding", "authority_scope"}
    row = _exact(value, fields, "claim-grade receipt")
    if row["schema"] != RECEIPT_SCHEMA or row["producer"] != PRODUCER_ID:
        raise ClaimReceiptError("claim-grade receipt schema/producer is unsupported")
    source = _exact(row["source_identity"], {
        "root_commit", "claim_tuple_sha256", "adapter_sha256", "adapter_id",
        "projector_name", "capture_schema", "measurement_capture_source_sha256",
        "measurement_capture_producer_id", "measurement_capture_schema",
        "observation_binding_source_sha256", "observation_binding_producer_id",
        "observation_binding_schemas"}, "projection source identity")
    _text(source["root_commit"], "ROOT commit")
    _sha(source["claim_tuple_sha256"], "ClaimTuple source")
    _sha(source["adapter_sha256"], "arm adapter source")
    for key in ("adapter_id", "projector_name", "capture_schema",
                "measurement_capture_producer_id", "measurement_capture_schema",
                "observation_binding_producer_id"):
        _text(source[key], key)
    _sha(source["measurement_capture_source_sha256"], "measurement-capture source")
    _sha(source["observation_binding_source_sha256"], "observation-binding source")
    schemas_row = _exact(source["observation_binding_schemas"], {
        "loaded_instrument_reference", "observation_unit_binding",
        "lifecycle_observation_reference", "lifecycle_observation_link"},
        "observation-binding schemas")
    for key, item in schemas_row.items():
        _text(item, f"observation binding schema {key}")
    event = _exact(row["source_event"], {"locator", "sha256"}, "source event")
    _text(event["locator"], "source event locator")
    _sha(event["sha256"], "source event artifact")
    projection = _exact(row["projection"], {
        "claim_tuple", "claim_tuple_digest", "source_grade", "trace_grade", "reasons"},
        "canonical projection")
    if not isinstance(projection["claim_tuple"], Mapping):
        raise ClaimReceiptError("canonical ClaimTuple must be an object")
    _sha(projection["claim_tuple_digest"], "ClaimTuple digest")
    _text(projection["source_grade"], "source grade")
    _text(projection["trace_grade"], "trace grade")
    if (not isinstance(projection["reasons"], list)
            or any(not isinstance(item, str) or not item for item in projection["reasons"])):
        raise ClaimReceiptError("grade reasons must be an array of non-empty text")
    binding = _exact(row["native_binding"], {
        "measurement_id", "arm", "plan_digest", "lineage_id",
        "comparison_identities_digest", "instrument_identity_sha256"},
        "canonical native binding")
    _text(binding["measurement_id"], "measurement id")
    if binding["arm"] not in {"anchor", "candidate"}:
        raise ClaimReceiptError("canonical receipt arm is unsupported")
    for key in ("plan_digest", "comparison_identities_digest",
                "instrument_identity_sha256"):
        _sha(binding[key], key)
    _text(binding["lineage_id"], "lineage id")
    if row["authority_scope"] not in {"compatibility_only", "final_pinned_source"}:
        raise ClaimReceiptError("canonical receipt authority scope is unsupported")
    if projection["claim_tuple_digest"] != schemas.content_hash(projection["claim_tuple"]):
        raise ClaimReceiptError("canonical ClaimTuple digest does not rederive")
    expected = schemas.content_hash({key: row[key] for key in row if key != "receipt_id"})
    if row["receipt_id"] != f"claim-grade-{expected[:24]}":
        raise ClaimReceiptError("canonical receipt id does not rederive")
    return row


__all__ = ["ClaimGradeReceiptReference", "ClaimGradeReceiptPairReference",
           "PAIR_REFERENCE_SCHEMA", "ClaimReceiptError", "PRODUCER_ID",
           "RECEIPT_SCHEMA", "REFERENCE_SCHEMA", "validate_receipt_body"]
