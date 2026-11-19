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
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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


SOURCE_BUILD_ENROLLMENT_SCHEMA = "epyc.autokernel.source_build_enrollment.v1"
EVIDENCE_RECORD_SCHEMA = "epyc.autokernel.enrollment_evidence_record.v1"


def mint_intent_from_source_build(
        source_enrollment: Mapping[str, Any], *, candidate: cm.CandidateManifest,
        request_id: str, enrollment_id: str, recipe_snapshot_digest: str,
        expiry: Mapping[str, float], clock_domain: str, issued_at: float,
        occurred_at: str) -> dict[str, Any]:
    """Mint an INTENT from the real source/build owner's verified enrollment (AKU-10i).

    The owner (`SourceBuildStageExecutor.execute`) returns a verified, published
    worktree BuildIdentity enrollment.  This binds that exact identity (binary and
    library digests, content hash) to one build of the candidate manifest; nothing is
    inferred: an unverified artifact, a digest that does not recompute, or a binary or
    DSO set that matches no (or more than one) candidate build is refused.  Opt-in: an
    integration that never calls this is unchanged.
    """
    from .. import schemas
    if not isinstance(source_enrollment, Mapping):
        raise EnrollmentRefused("source/build enrollment must be an object")
    if source_enrollment.get("schema") != SOURCE_BUILD_ENROLLMENT_SCHEMA:
        raise EnrollmentRefused("source/build enrollment schema differs")
    artifact = source_enrollment.get("artifact")
    if not isinstance(artifact, Mapping) or artifact.get("verified") is not True:
        raise EnrollmentRefused("source/build enrollment artifact is not verified")
    identity = source_enrollment.get("build_identity")
    if not isinstance(identity, Mapping):
        raise EnrollmentRefused("source/build enrollment lacks a build identity")
    if schemas.content_hash(dict(identity)) != source_enrollment.get("build_identity_digest"):
        raise EnrollmentRefused("source/build build identity digest does not recompute")
    binary = _sha(identity.get("output_binary_sha256"), "output_binary_sha256")
    try:
        libraries = {_sha(item[1], "library sha256") for item in identity["library_sha256s"]}
    except (KeyError, TypeError, IndexError) as exc:
        raise EnrollmentRefused("source/build library digests are malformed") from exc
    matches = [item for item in candidate.builds
               if item.executable.sha256 == binary
               and {dso.sha256 for dso in item.dsos} == libraries]
    if len(matches) != 1:
        raise EnrollmentRefused(
            f"verified build matches {len(matches)} candidate builds (need exactly one)")
    row = {"schema": INTENT_SCHEMA, "event": "INTENT", "enrollment_id": enrollment_id,
           "campaign_id": source_enrollment.get("campaign_id"),
           "integration_request_id": request_id,
           "candidate_manifest_digest": candidate.manifest_digest,
           "build_identity": matches[0].to_dict(),
           "recipe_snapshot_digest": recipe_snapshot_digest,
           "source_refs": [item.to_dict() for item in candidate.sources],
           "expiry": dict(expiry), "clock_domain": clock_domain, "issued_at": issued_at,
           "occurred_at": occurred_at}
    return validate_record(row)


def build_activated(intent: Mapping[str, Any], *, candidate_state_digest: str,
                    evidence: Mapping[str, str], activated_at: float,
                    occurred_at: str) -> dict[str, Any]:
    """ACTIVATED joining the exact INTENT digest; must precede every expiry."""
    intent = validate_record(intent)
    if intent["event"] != "INTENT":
        raise EnrollmentRefused("activation joins an INTENT record")
    row = {"schema": ACTIVATED_SCHEMA, "event": "ACTIVATED",
           "enrollment_id": intent["enrollment_id"], "intent_digest": digest(intent),
           "integration_request_id": intent["integration_request_id"],
           "candidate_state_digest": candidate_state_digest, "evidence": dict(evidence),
           "activated_at": activated_at, "occurred_at": occurred_at}
    row = validate_record(row)
    project_records([intent, row])  # expiry precedence + exact joins
    return row


class EnrollmentStore:
    """Write-once durable INTENT/ACTIVATED files; restart reconciles by re-projection."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, enrollment_id: str, event: str) -> Path:
        name = hashlib.sha256(enrollment_id.encode()).hexdigest()
        return self.root / f"{name}.{event}.json"

    def _write_once(self, path: Path, row: Mapping[str, Any]) -> bool:
        raw = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if path.exists():
            if path.read_text() != raw:
                raise EnrollmentRefused("enrollment record already persisted with different content")
            return False
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=".tmp-")
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(tmp, path)  # atomic and fails if a racer already won
            except FileExistsError:
                if path.read_text() != raw:
                    raise EnrollmentRefused(
                        "enrollment record already persisted with different content")
                return False
        finally:
            os.unlink(tmp)
        dirfd = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(dirfd)
        finally:
            os.close(dirfd)
        return True

    def record_intent(self, intent: Mapping[str, Any]) -> bool:
        row = validate_record(intent)
        if row["event"] != "INTENT":
            raise EnrollmentRefused("record_intent requires an INTENT record")
        return self._write_once(self._path(row["enrollment_id"], "INTENT"), row)

    def record_activated(self, activated: Mapping[str, Any]) -> bool:
        row = validate_record(activated)
        if row["event"] != "ACTIVATED":
            raise EnrollmentRefused("record_activated requires an ACTIVATED record")
        intent_path = self._path(row["enrollment_id"], "INTENT")
        if not intent_path.exists():
            raise EnrollmentRefused("activation lacks its persisted original intent")
        project_records([json.loads(intent_path.read_text()), row])
        return self._write_once(self._path(row["enrollment_id"], "ACTIVATED"), row)

    def reconcile(self) -> dict[str, dict[str, Any]]:
        """enrollment_id -> {state, entry}; refuses orphan/misfiled/divergent files."""
        intents, activations = [], []
        for path in sorted(self.root.glob("*.json")):
            row = validate_record(json.loads(path.read_text()))
            if path != self._path(row["enrollment_id"], row["event"]):
                raise EnrollmentRefused(f"enrollment file {path.name} is misfiled")
            (intents if row["event"] == "INTENT" else activations).append(row)
        projected = project_records([*intents, *activations])
        return {key: {"state": "activated" if entry["activated"] else "intent_only",
                      "entry": entry} for key, entry in projected.items()}


@dataclass(frozen=True)
class DispositionDecision:
    """Explicit refuse / allow-with-reasons.  Allowing issues nothing by itself."""
    allowed: bool
    reasons: tuple[str, ...]
    expired: tuple[str, ...] = ()
    # Constants documenting what an allow never does.
    infers_rejection: bool = False
    changes_deletion_policy: bool = False


EvidenceReader = Callable[[str, str], "Mapping[str, Any] | None"]


def disposition_gate(entry: Mapping[str, Any] | None, *, now: float, clock_domain: str,
                     read_evidence: EvidenceReader,
                     existing_dispositions: Sequence[str] = ()) -> DispositionDecision:
    """Evidence + ownership gate that must pass before any disposition is issued.

    Refuses unless the enrollment is activated (original INTENT joined, no backfill),
    at least one authority has expired on the intent's own clock domain, every closed
    original evidence ref (evaluation/native/worker/held-cost/physical-owner) re-reads
    through its owner as a content-addressed record bound to this INTENT and CLOSED, the
    physical owner is exclusive (derived from the owner list, never a label) and bound
    to this build, and no disposition record already exists (no duplicate accounting).
    Never infers a rejection, accepts a caller label, backfills or changes deletion
    policy; `expiry_status` itself stays status-only.
    """
    if entry is None or entry.get("activated") is None:
        return DispositionDecision(False, ("not_activated",))
    intent, activated = entry["intent"], entry["activated"]
    reasons: list[str] = []
    if activated["activated_at"] <= intent["issued_at"]:
        reasons.append("backfill_activation_not_after_intent")
    if activated["intent_digest"] != entry["intent_digest"] or \
            digest(intent) != entry["intent_digest"]:
        reasons.append("intent_digest_mismatch")
    try:
        status = expiry_status(entry, now=now, clock_domain=clock_domain)
    except EnrollmentRefused as exc:
        return DispositionDecision(False, (*reasons, f"expiry_unreadable:{exc}"))
    expired = tuple(name for name, value in status.items() if value == "expired")
    if not expired:
        reasons.append("nothing_expired")
    if list(existing_dispositions):
        reasons.append("duplicate_disposition_record")
    build = cm.BuildIdentity.from_dict(intent["build_identity"])
    for key in EVIDENCE_KEYS:
        ref = activated["evidence"][key]
        try:
            record = read_evidence(key, ref)
        except Exception:  # an unreadable owner is missing evidence, never a pass
            record = None
        if not isinstance(record, Mapping):
            reasons.append(f"evidence_missing:{key}")
            continue
        try:
            addressed = digest(dict(record)) == ref
        except EnrollmentRefused:
            addressed = False
        if (not addressed or record.get("schema") != EVIDENCE_RECORD_SCHEMA
                or record.get("kind") != key
                or record.get("enrollment_id") != intent["enrollment_id"]
                or record.get("intent_digest") != entry["intent_digest"]):
            reasons.append(f"evidence_unbound:{key}")
        elif record.get("state") != "closed":
            reasons.append(f"evidence_not_closed:{key}")
        elif key == "physical_owner_ref":
            owners = record.get("owners")
            if (not isinstance(owners, list) or len(owners) != 1
                    or not isinstance(owners[0], str) or not owners[0]):
                reasons.append("ownership_not_exclusive")
            elif record.get("build_execution_digest") != build.execution_digest:
                reasons.append("ownership_unbound_to_build")
    if reasons:
        return DispositionDecision(False, tuple(reasons), expired)
    return DispositionDecision(
        True, ("activated", "evidence_closed", "ownership_exclusive",
               "no_duplicate_disposition", "expired:" + ",".join(expired)), expired)


__all__ = ["ACTIVATED_SCHEMA", "DispositionDecision", "EVIDENCE_KEYS",
           "EVIDENCE_RECORD_SCHEMA", "EXPIRY_KEYS", "EnrollmentRefused", "EnrollmentStore",
           "INTENT_SCHEMA", "SOURCE_BUILD_ENROLLMENT_SCHEMA", "bind_to_integration",
           "build_activated", "digest", "disposition_gate", "expiry_status",
           "mint_intent_from_source_build", "project_records", "validate_record"]
