"""Versioned prospective build-enrollment INTENT/ACTIVATED records (AKU-10i).

Additive and opt-in: an integration that carries no enrollment record is unchanged.
The INTENT record names the verified BuildIdentity, the exact recipe snapshot and the
retained source refs plus the source/build/evaluation expiry authority.  It rides in the
journaled candidate-transaction intent (so Journal native payload validation recovers it)
and is persisted as an immutable object before integration.  ACTIVATED joins the exact
INTENT digest to closed original evidence.  Nothing here issues a disposition, infers a
rejection, backfills history or changes deletion policy; expiry only reports status.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from . import candidate_manifest as cm

INTENT_SCHEMA = "epyc.autokernel.build_enrollment_intent.v1"
ACTIVATED_SCHEMA = "epyc.autokernel.build_enrollment_activated.v1"
EXPIRY_KEYS = ("source_valid_until", "build_valid_until", "evaluation_valid_until")
EVIDENCE_KEYS = ("evaluation_ref", "native_ref", "worker_ref", "held_cost_ref",
                 "physical_owner_ref")
_INTENT = {"schema", "event", "enrollment_id", "campaign_id", "integration_request_id",
           "candidate_manifest_digest", "build_identity", "recipe_snapshot_digest",
           "source_refs", "expiry", "clock_domain", "issued_at", "occurred_at"}
_ACTIVATED = {"schema", "event", "enrollment_id", "intent_digest",
              "integration_request_id", "candidate_state_digest", "evidence",
              "activated_at", "occurred_at"}


class EnrollmentRefused(ValueError):
    pass


def digest(value: Any) -> str:
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise EnrollmentRefused("enrollment record is not canonical finite JSON") from exc
    return hashlib.sha256(raw.encode()).hexdigest()


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EnrollmentRefused(f"{label} must be nonempty text")
    return value


def _sha(value: Any, label: str) -> str:
    value = _text(value, label)
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise EnrollmentRefused(f"{label} must be lowercase SHA-256")
    return value


def _time(value: Any, label: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise EnrollmentRefused(f"{label} must be finite and positive")
    return float(value)


def validate_record(value: Mapping[str, Any]) -> dict[str, Any]:
    """Closed validator for one INTENT or ACTIVATED record."""
    if not isinstance(value, Mapping):
        raise EnrollmentRefused("enrollment record must be an object")
    row = dict(value)
    event = row.get("event")
    if event == "INTENT":
        if set(row) != _INTENT or row["schema"] != INTENT_SCHEMA:
            raise EnrollmentRefused("enrollment intent fields/schema differ")
        for name in ("enrollment_id", "campaign_id", "integration_request_id",
                     "clock_domain", "occurred_at"):
            _text(row[name], name)
        _sha(row["candidate_manifest_digest"], "candidate_manifest_digest")
        _sha(row["recipe_snapshot_digest"], "recipe_snapshot_digest")
        try:
            build = cm.BuildIdentity.from_dict(row["build_identity"])
        except (cm.CandidateError, KeyError, TypeError, ValueError) as exc:
            raise EnrollmentRefused(f"build identity is invalid: {exc}") from exc
        if build.to_dict() != row["build_identity"]:
            raise EnrollmentRefused("build identity is not in canonical form")
        refs = row["source_refs"]
        if not isinstance(refs, Sequence) or isinstance(refs, (str, bytes)) or not refs:
            raise EnrollmentRefused("source_refs must be a nonempty list")
        try:
            sources = [cm.SourceIdentity.from_dict(item) for item in refs]
        except (cm.CandidateError, KeyError, TypeError, ValueError) as exc:
            raise EnrollmentRefused(f"source ref is invalid: {exc}") from exc
        if len({item.repo_id for item in sources}) != len(sources):
            raise EnrollmentRefused("source_refs repeat a repository")
        if cm.source_set_digest(sources) != build.source_set_digest:
            raise EnrollmentRefused("source refs do not match the build source set")
        issued = _time(row["issued_at"], "issued_at")
        expiry = row["expiry"]
        if not isinstance(expiry, Mapping) or set(expiry) != set(EXPIRY_KEYS):
            raise EnrollmentRefused("expiry must carry exactly source/build/evaluation")
        for name in EXPIRY_KEYS:
            if _time(expiry[name], name) <= issued:
                raise EnrollmentRefused(f"{name} must follow issued_at")
        return row
    if event == "ACTIVATED":
        if set(row) != _ACTIVATED or row["schema"] != ACTIVATED_SCHEMA:
            raise EnrollmentRefused("enrollment activation fields/schema differ")
        for name in ("enrollment_id", "integration_request_id", "occurred_at"):
            _text(row[name], name)
        _sha(row["intent_digest"], "intent_digest")
        _sha(row["candidate_state_digest"], "candidate_state_digest")
        _time(row["activated_at"], "activated_at")
        evidence = row["evidence"]
        if not isinstance(evidence, Mapping) or set(evidence) != set(EVIDENCE_KEYS):
            raise EnrollmentRefused("activation needs exactly the closed evidence set")
        for name in EVIDENCE_KEYS:
            _sha(evidence[name], f"evidence.{name}")
        return row
    raise EnrollmentRefused("enrollment event must be INTENT or ACTIVATED")


def project_records(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """enrollment_id -> {intent, intent_digest, activated}; refuses any broken join."""
    out: dict[str, dict[str, Any]] = {}
    for value in records:
        row = validate_record(value)
        key = row["enrollment_id"]
        if row["event"] == "INTENT":
            if key in out:
                raise EnrollmentRefused("enrollment intent is duplicated")
            out[key] = {"intent": row, "intent_digest": digest(row), "activated": None}
            continue
        entry = out.get(key)
        if entry is None:
            raise EnrollmentRefused("activation lacks its original intent")
        if entry["activated"] is not None:
            raise EnrollmentRefused("enrollment is already activated")
        if row["intent_digest"] != entry["intent_digest"]:
            raise EnrollmentRefused("activation intent digest differs")
        if row["integration_request_id"] != entry["intent"]["integration_request_id"]:
            raise EnrollmentRefused("activation integration request differs")
        if row["activated_at"] >= min(entry["intent"]["expiry"].values()):
            raise EnrollmentRefused("activation occurs after an expiry")
        entry["activated"] = row
    return out


def expiry_status(entry: Mapping[str, Any], *, now: float, clock_domain: str
                  ) -> dict[str, str]:
    """Report valid/expired per authority.  Status only: never a disposition."""
    intent = entry["intent"]
    if clock_domain != intent["clock_domain"]:
        raise EnrollmentRefused("expiry is compared across clock domains")
    now = _time(now, "now")
    return {name: ("valid" if now < intent["expiry"][name] else "expired")
            for name in EXPIRY_KEYS}


def bind_to_integration(record: Mapping[str, Any], *, request_id: str,
                        candidate: cm.CandidateManifest) -> dict[str, Any]:
    """INTENT must bind this exact integration and a build/source set of the candidate."""
    row = validate_record(record)
    if row["event"] != "INTENT":
        raise EnrollmentRefused("integration carries an INTENT record")
    if row["integration_request_id"] != request_id:
        raise EnrollmentRefused("enrollment binds a different integration request")
    if row["candidate_manifest_digest"] != candidate.manifest_digest:
        raise EnrollmentRefused("enrollment binds a different candidate manifest")
    build = cm.BuildIdentity.from_dict(row["build_identity"])
    if build.to_dict() not in [item.to_dict() for item in candidate.builds]:
        raise EnrollmentRefused("enrolled build is not in the candidate manifest")
    sources = sorted((cm.SourceIdentity.from_dict(item).to_dict()
                      for item in row["source_refs"]), key=lambda r: r["repo_id"])
    if sources != [item.to_dict() for item in candidate.sources]:
        raise EnrollmentRefused("enrolled source refs differ from the candidate sources")
    return row


__all__ = ["ACTIVATED_SCHEMA", "EVIDENCE_KEYS", "EXPIRY_KEYS", "EnrollmentRefused",
           "INTENT_SCHEMA", "bind_to_integration", "digest", "expiry_status",
           "project_records", "validate_record"]
