#!/usr/bin/env python3
"""Scoped, replayable evidence projection for AutoKernel research.

This is a pure projection over supplied native events.  It neither persists an
event log nor grades claims.  Certificate authority is available only through
trusted callbacks supplied by an eventual registered adapter.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import math
import re
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence, cast

from .. import schemas


CLAIM_KEY_SCHEMA = "epyc.autokernel.scoped_claim_key.v1"
SOURCE_REF_SCHEMA = "epyc.autokernel.source_ref.v1"
FINDING_SCHEMA = "epyc.autokernel.scoped_finding.v1"
INVALIDATION_SCHEMA = "epyc.autokernel.evidence_invalidation.v1"
INDEX_SCHEMA = "epyc.autokernel.evidence_index.v1"
RETRIEVAL_SCHEMA = "epyc.autokernel.scoped_retrieval.v1"
PROPOSAL_SCHEMA = "epyc.autokernel.evidence_proposal_snapshot.v1"
LOCAL_FENCE_SCHEMA = "epyc.autokernel.local_evidence_fences.v1"
QUARANTINE_SCHEMA = "epyc.autokernel.evidence_quarantine.v1"
TRANSFER_SCHEMA = "epyc.autokernel.transfer_receipt.v1"
ROUTE_SCHEMA = "epyc.autokernel.mechanism_route.v1"
AUDIT_SCHEMA = "epyc.autokernel.reject_audit_decision.v1"
REVOCATION_SCHEMA = "epyc.autokernel.route_revocation.v1"
COEXISTENCE_SCHEMA = "epyc.autokernel.coexistence_receipt.v1"

_SHA = re.compile(r"^[0-9a-f]{64}$")
_CONCLUSIONS = frozenset({"positive", "null", "refutation", "conflict", "retraction"})
_RECORD_CLASSES = frozenset({
    "discovery_screen", "strict_search", "observation", "registered_claim",
})
_DIRECTIONS = frozenset({"higher", "lower"})
_TRANSFER_TYPES = frozenset({"correctness", "local_work", "serving_effect"})
_ROUTE_DISPOSITIONS = frozenset({"exploration_only", "may_screen_out"})
_CERTIFICATE_USES = frozenset({
    "screen_out", "reject", "certify", "validate", "promotion", "timing",
    "admission", "coexistence", "bank", "validate_production",
    "certify_transfer", "certify_overlap", "headline", "release",
})
_VERIFIED_USES = _CERTIFICATE_USES | frozenset({"rank"})
_INTENDED_USES = frozenset({"explore", "nominate", "rank"}) | _CERTIFICATE_USES
_LIFECYCLE_PHASES = frozenset({
    "setup", "placement", "load", "warmup", "steady", "bursts", "teardown",
})


class EvidenceValidationError(ValueError):
    """A supplied event or projection object violates its versioned schema."""


def _exact(obj: Mapping[str, Any], fields: set[str], label: str) -> None:
    if not isinstance(obj, Mapping):
        raise EvidenceValidationError(f"{label}: expected object")
    missing, extra = fields - set(obj), set(obj) - fields
    if missing or extra:
        parts = []
        if missing:
            parts.append("missing " + ", ".join(sorted(missing)))
        if extra:
            parts.append("unknown " + ", ".join(sorted(extra)))
        raise EvidenceValidationError(f"{label}: {'; '.join(parts)}")


def _text(value: Any, label: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not value.strip():
        raise EvidenceValidationError(f"{label}: expected non-empty text")
    return value


def _enum(value: Any, choices: frozenset[str], label: str) -> str:
    result = cast(str, _text(value, label))
    if result not in choices:
        raise EvidenceValidationError(f"{label}: unsupported {result!r}")
    return result


def _int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EvidenceValidationError(f"{label}: expected integer >= {minimum}")
    return value


def _number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceValidationError(f"{label}: expected finite number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise EvidenceValidationError(f"{label}: expected finite number"
                                      + (f" >= {minimum}" if minimum is not None else ""))
    return result


def _digest(value: Any, label: str) -> str:
    result = cast(str, _text(value, label))
    if not _SHA.fullmatch(result):
        raise EvidenceValidationError(f"{label}: expected lowercase SHA-256")
    return result


def _freeze(value: Any, label: str) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise EvidenceValidationError(f"{label}: mapping keys must be strings")
        frozen = MappingProxyType({key: _freeze(item, f"{label}.{key}")
                                   for key, item in value.items()})
    elif isinstance(value, (list, tuple)):
        frozen = tuple(_freeze(item, f"{label}[]") for item in value)
    else:
        frozen = value
    try:
        schemas.canonical_json(_thaw(frozen))
    except (TypeError, ValueError) as exc:
        raise EvidenceValidationError(f"{label}: {exc}") from exc
    return frozen


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _mapping(value: Any, label: str, *, nonempty: bool = True) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or (nonempty and not value):
        raise EvidenceValidationError(
            f"{label}: expected {'non-empty ' if nonempty else ''}object")
    return _freeze(_thaw(value), label)


def _strings(value: Any, label: str, *, empty: bool = False,
             unique: bool = True) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise EvidenceValidationError(f"{label}: expected array")
    result = tuple(cast(str, _text(item, f"{label}[]")) for item in value)
    if not empty and not result:
        raise EvidenceValidationError(f"{label}: must not be empty")
    if unique and len(set(result)) != len(result):
        raise EvidenceValidationError(f"{label}: duplicate identifiers")
    return result


def _dependency_identities(value: Any, label: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise EvidenceValidationError(f"{label}: expected non-empty object")
    result: dict[str, str] = {}
    for dep, identity in value.items():
        dep = cast(str, _text(dep, f"{label} key"))
        result[dep] = _digest(identity, f"{label}.{dep}")
    return MappingProxyType(result)


def _generations(value: Any, label: str) -> Mapping[str, int]:
    if not isinstance(value, Mapping) or not value:
        raise EvidenceValidationError(f"{label}: expected non-empty object")
    result = {}
    for dep, generation in value.items():
        dep = cast(str, _text(dep, f"{label} key"))
        result[dep] = _int(generation, f"{label}.{dep}")
    return MappingProxyType(result)


def _generations_allow_empty(value: Any, label: str) -> Mapping[str, int]:
    if not isinstance(value, Mapping):
        raise EvidenceValidationError(f"{label}: expected object")
    result = {}
    for dep, generation in value.items():
        dep = cast(str, _text(dep, f"{label} key"))
        result[dep] = _int(generation, f"{label}.{dep}")
    return MappingProxyType(result)


def _digest_map(value: Any, label: str, *, empty: bool = True) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or (not empty and not value):
        raise EvidenceValidationError(f"{label}: expected object")
    result: dict[str, str] = {}
    for key, digest in value.items():
        key = cast(str, _text(key, f"{label} key"))
        result[key] = _digest(digest, f"{label}.{key}")
    return MappingProxyType(result)


def _question(value: Any, label: str) -> Mapping[str, Any]:
    _exact(value, {"kind", "bound", "unit"}, label)
    if value["kind"] != "absolute_effect_bound":
        raise EvidenceValidationError(f"{label}.kind: unsupported {value['kind']!r}")
    return MappingProxyType({
        "kind": "absolute_effect_bound",
        "bound": _number(value["bound"], f"{label}.bound", minimum=0.0),
        "unit": cast(str, _text(value["unit"], f"{label}.unit")),
    })


def _callback(value: Any, label: str) -> Any:
    if value is not None and not callable(value):
        raise EvidenceValidationError(f"{label}: expected trusted callable or null")
    return value


def _verify(callback: Callable[..., bool | str], *args: Any) -> bool | str:
    try:
        return callback(*args)
    except Exception as exc:
        return f"trusted verifier failed closed: {exc}"


@dataclass(frozen=True)
class ClaimKey:
    schema: str
    target_scope: Mapping[str, Any]
    control_identity: Mapping[str, Any]
    intervention_identity: Mapping[str, Any]
    mechanism_identity: Mapping[str, Any]
    estimand: str
    metric: str
    metric_direction: str
    effect_question: Mapping[str, Any]
    dependency_identities: Mapping[str, str]

    FIELDS = {"schema", "target_scope", "control_identity", "intervention_identity",
              "mechanism_identity", "estimand", "metric", "metric_direction",
              "effect_question", "dependency_identities"}
    SCOPE_FIELDS = {"target", "backend", "model", "quant", "workload", "allocation"}

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "ClaimKey":
        _exact(obj, cls.FIELDS, "ClaimKey")
        if obj["schema"] != CLAIM_KEY_SCHEMA:
            raise EvidenceValidationError(f"ClaimKey.schema: unsupported {obj['schema']!r}")
        scope = obj["target_scope"]
        _exact(scope, cls.SCOPE_FIELDS, "ClaimKey.target_scope")
        scope = {name: cast(str, _text(scope[name], f"ClaimKey.target_scope.{name}"))
                 for name in cls.SCOPE_FIELDS}
        return cls(
            CLAIM_KEY_SCHEMA, _freeze(scope, "ClaimKey.target_scope"),
            _mapping(obj["control_identity"], "ClaimKey.control_identity"),
            _mapping(obj["intervention_identity"], "ClaimKey.intervention_identity"),
            _mapping(obj["mechanism_identity"], "ClaimKey.mechanism_identity"),
            cast(str, _text(obj["estimand"], "ClaimKey.estimand")),
            cast(str, _text(obj["metric"], "ClaimKey.metric")),
            _enum(obj["metric_direction"], _DIRECTIONS, "ClaimKey.metric_direction"),
            _question(obj["effect_question"], "ClaimKey.effect_question"),
            _dependency_identities(obj["dependency_identities"],
                                   "ClaimKey.dependency_identities"))

    def to_dict(self) -> dict[str, Any]:
        return {name: _thaw(getattr(self, name)) for name in self.FIELDS}

    @property
    def digest(self) -> str:
        return schemas.content_hash(self.to_dict())


@dataclass(frozen=True)
class SourceRef:
    schema: str
    event_id: str
    artifact_digest: str
    locator: str

    FIELDS = {"schema", "event_id", "artifact_digest", "locator"}

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "SourceRef":
        _exact(obj, cls.FIELDS, "SourceRef")
        if obj["schema"] != SOURCE_REF_SCHEMA:
            raise EvidenceValidationError(f"SourceRef.schema: unsupported {obj['schema']!r}")
        return cls(SOURCE_REF_SCHEMA,
                   cast(str, _text(obj["event_id"], "SourceRef.event_id")),
                   _digest(obj["artifact_digest"], "SourceRef.artifact_digest"),
                   cast(str, _text(obj["locator"], "SourceRef.locator")))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "event_id": self.event_id,
                "artifact_digest": self.artifact_digest, "locator": self.locator}


@dataclass(frozen=True)
class Finding:
    schema: str
    finding_id: str
    source: SourceRef
    claim_key: ClaimKey
    conclusion: str
    value: float | None
    tested_scope: Mapping[str, Any]
    tested_question: Mapping[str, Any]
    raw_grade: Mapping[str, Any]
    epoch: str
    record_class: str
    dependency_generations: Mapping[str, int]
    intended_use_disposition: Mapping[str, Any]
    authority_reference: str | None
    frontier: int

    FIELDS = {"schema", "finding_id", "source", "claim_key", "conclusion",
              "value", "tested_scope", "tested_question", "raw_grade", "epoch",
              "record_class", "dependency_generations", "intended_use_disposition",
              "authority_reference", "frontier"}

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "Finding":
        _exact(obj, cls.FIELDS, "Finding")
        if obj["schema"] != FINDING_SCHEMA:
            raise EvidenceValidationError(f"Finding.schema: unsupported {obj['schema']!r}")
        conclusion = _enum(obj["conclusion"], _CONCLUSIONS, "Finding.conclusion")
        value = obj["value"]
        if value is not None:
            value = _number(value, "Finding.value")
        disposition = obj["intended_use_disposition"]
        _exact(disposition, {"intended_use", "disposition"},
               "Finding.intended_use_disposition")
        disposition = {
            "intended_use": _enum(
                disposition["intended_use"], _INTENDED_USES,
                "Finding.intended_use_disposition.intended_use"),
            "disposition": _enum(
                disposition["disposition"],
                frozenset({"exploration_only", "certificate_candidate"}),
                "Finding.intended_use_disposition.disposition"),
        }
        claim_key = ClaimKey.from_dict(obj["claim_key"])
        dependency_generations = _generations(
            obj["dependency_generations"], "Finding.dependency_generations")
        if set(dependency_generations) != set(claim_key.dependency_identities):
            raise EvidenceValidationError(
                "Finding.dependency_generations must exactly cover claim dependencies")
        return cls(
            FINDING_SCHEMA, cast(str, _text(obj["finding_id"], "Finding.finding_id")),
            SourceRef.from_dict(obj["source"]), claim_key,
            conclusion, value, _mapping(obj["tested_scope"], "Finding.tested_scope"),
            _question(obj["tested_question"], "Finding.tested_question"),
            _mapping(obj["raw_grade"], "Finding.raw_grade", nonempty=False),
            cast(str, _text(obj["epoch"], "Finding.epoch")),
            _enum(obj["record_class"], _RECORD_CLASSES, "Finding.record_class"),
            dependency_generations,
            _freeze(disposition, "Finding.intended_use_disposition"),
            _text(obj["authority_reference"], "Finding.authority_reference", nullable=True),
            _int(obj["frontier"], "Finding.frontier"))

    def to_dict(self) -> dict[str, Any]:
        result = {name: _thaw(getattr(self, name)) for name in self.FIELDS
                  if name not in {"source", "claim_key"}}
        result["source"] = self.source.to_dict()
        result["claim_key"] = self.claim_key.to_dict()
        return result

    @property
    def digest(self) -> str:
        return schemas.content_hash(self.to_dict())


@dataclass(frozen=True)
class InvalidationEvent:
    schema: str
    event_id: str
    dependency_id: str
    generation: int
    kind: str
    frontier: int

    FIELDS = {"schema", "event_id", "dependency_id", "generation", "kind", "frontier"}

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "InvalidationEvent":
        _exact(obj, cls.FIELDS, "InvalidationEvent")
        if obj["schema"] != INVALIDATION_SCHEMA:
            raise EvidenceValidationError(
                f"InvalidationEvent.schema: unsupported {obj['schema']!r}")
        return cls(
            INVALIDATION_SCHEMA,
            cast(str, _text(obj["event_id"], "InvalidationEvent.event_id")),
            cast(str, _text(obj["dependency_id"], "InvalidationEvent.dependency_id")),
            _int(obj["generation"], "InvalidationEvent.generation", minimum=1),
            _enum(obj["kind"], frozenset({"recipe", "topology", "retraction", "dependency"}),
                  "InvalidationEvent.kind"),
            _int(obj["frontier"], "InvalidationEvent.frontier"))

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.FIELDS}

    @property
    def digest(self) -> str:
        return schemas.content_hash(self.to_dict())


@dataclass(frozen=True)
class Quarantine:
    schema: str
    event_id: str
    event_digest: str
    reason: str
    affected_dependencies: tuple[str, ...]
    global_scope: bool
    frontier: int

    FIELDS = {"schema", "event_id", "event_digest", "reason",
              "affected_dependencies", "global_scope", "frontier"}

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "Quarantine":
        _exact(obj, cls.FIELDS, "Quarantine")
        if obj["schema"] != QUARANTINE_SCHEMA:
            raise EvidenceValidationError(f"Quarantine.schema: unsupported {obj['schema']!r}")
        if not isinstance(obj["global_scope"], bool):
            raise EvidenceValidationError("Quarantine.global_scope: expected boolean")
        dependencies = _strings(obj["affected_dependencies"],
                                "Quarantine.affected_dependencies", empty=True)
        if obj["global_scope"] == bool(dependencies):
            raise EvidenceValidationError(
                "Quarantine global_scope must be true exactly when dependencies are unknown")
        return cls(
            QUARANTINE_SCHEMA, cast(str, _text(obj["event_id"], "Quarantine.event_id")),
            _digest(obj["event_digest"], "Quarantine.event_digest"),
            cast(str, _text(obj["reason"], "Quarantine.reason")), dependencies,
            obj["global_scope"], _int(obj["frontier"], "Quarantine.frontier"))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "event_id": self.event_id,
                "event_digest": self.event_digest,
                "reason": self.reason,
                "affected_dependencies": list(self.affected_dependencies),
                "global_scope": self.global_scope, "frontier": self.frontier}


@dataclass(frozen=True)
class RetrievedFinding:
    finding: Finding
    applicability: str
    magnitude_status: str
    ranking_value: float | None

    def to_dict(self) -> dict[str, Any]:
        return {"finding": self.finding.to_dict(), "applicability": self.applicability,
                "magnitude_status": self.magnitude_status,
                "ranking_value": self.ranking_value}


@dataclass(frozen=True)
class RetrievalResult:
    schema: str
    findings: tuple[RetrievedFinding, ...]
    mandatory_conflicts: tuple[RetrievedFinding, ...]
    dependency_generations: Mapping[str, int]
    snapshot_frontier: int
    retrieval_complete: bool
    supported_for_intended_use: bool
    reasons: tuple[str, ...]
    quarantines: tuple[Quarantine, ...]
    support_basis_digest: str
    result_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema,
                "findings": [row.to_dict() for row in self.findings],
                "mandatory_conflicts": [row.to_dict() for row in self.mandatory_conflicts],
                "dependency_generations": dict(self.dependency_generations),
                "snapshot_frontier": self.snapshot_frontier,
                "retrieval_complete": self.retrieval_complete,
                "supported_for_intended_use": self.supported_for_intended_use,
                "reasons": list(self.reasons),
                "quarantines": [row.to_dict() for row in self.quarantines],
                "support_basis_digest": self.support_basis_digest,
                "result_digest": self.result_digest}

    @property
    def complete_for_intended_use(self) -> bool:
        """Compatibility reading: both complete retrieval and supported use."""
        return self.retrieval_complete and self.supported_for_intended_use


@dataclass(frozen=True)
class LocalFenceSnapshot:
    schema: str
    available: bool
    global_fence_generation: int
    dependency_generations: Mapping[str, int]
    dependency_fence_generations: Mapping[str, int]
    dependency_frontiers: Mapping[str, int]
    dependency_evidence_digests: Mapping[str, str]
    semantic_fences: Mapping[str, str]
    current_epoch: str
    support_rule_identity: str | None
    frontier: int

    FIELDS = {"schema", "available", "global_fence_generation",
              "dependency_generations", "dependency_fence_generations",
              "dependency_frontiers", "dependency_evidence_digests",
              "semantic_fences", "current_epoch", "support_rule_identity", "frontier"}

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "LocalFenceSnapshot":
        _exact(obj, cls.FIELDS, "LocalFenceSnapshot")
        if obj["schema"] != LOCAL_FENCE_SCHEMA:
            raise EvidenceValidationError(
                f"LocalFenceSnapshot.schema: unsupported {obj['schema']!r}")
        if not isinstance(obj["available"], bool):
            raise EvidenceValidationError("LocalFenceSnapshot.available: expected boolean")
        generations = _generations_allow_empty(
            obj["dependency_generations"], "LocalFenceSnapshot.dependency_generations")
        fences = _generations_allow_empty(
            obj["dependency_fence_generations"],
            "LocalFenceSnapshot.dependency_fence_generations")
        frontiers = _generations_allow_empty(
            obj["dependency_frontiers"], "LocalFenceSnapshot.dependency_frontiers")
        return cls(
            LOCAL_FENCE_SCHEMA, obj["available"],
            _int(obj["global_fence_generation"],
                 "LocalFenceSnapshot.global_fence_generation"),
            generations, fences, frontiers,
            _digest_map(obj["dependency_evidence_digests"],
                        "LocalFenceSnapshot.dependency_evidence_digests"),
            _digest_map(obj["semantic_fences"],
                        "LocalFenceSnapshot.semantic_fences"),
            cast(str, _text(obj["current_epoch"], "LocalFenceSnapshot.current_epoch")),
            _text(obj["support_rule_identity"],
                  "LocalFenceSnapshot.support_rule_identity", nullable=True),
            _int(obj["frontier"], "LocalFenceSnapshot.frontier"))

    def to_dict(self) -> dict[str, Any]:
        return {name: _thaw(getattr(self, name)) for name in self.FIELDS}


@dataclass(frozen=True)
class ProposalSnapshot:
    schema: str
    claim_digest: str
    intended_use: str
    dependency_generations: Mapping[str, int]
    dependency_fence_generations: Mapping[str, int]
    dependency_frontiers: Mapping[str, int]
    dependency_evidence_digests: Mapping[str, str]
    semantic_fences: Mapping[str, str]
    current_epoch: str
    support_rule_identity: str | None
    global_fence_generation: int
    snapshot_frontier: int
    retrieval_result_digest: str
    retrieval_complete: bool
    supported_for_intended_use: bool
    index_digest: str

    FIELDS = {"schema", "claim_digest", "intended_use", "dependency_generations",
              "dependency_fence_generations", "dependency_frontiers",
              "dependency_evidence_digests", "semantic_fences", "current_epoch",
              "support_rule_identity",
              "global_fence_generation",
              "snapshot_frontier", "retrieval_result_digest", "retrieval_complete",
              "supported_for_intended_use", "index_digest"}

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "ProposalSnapshot":
        _exact(obj, cls.FIELDS, "ProposalSnapshot")
        if obj["schema"] != PROPOSAL_SCHEMA:
            raise EvidenceValidationError(
                f"ProposalSnapshot.schema: unsupported {obj['schema']!r}")
        if not isinstance(obj["retrieval_complete"], bool):
            raise EvidenceValidationError("ProposalSnapshot.retrieval_complete: expected boolean")
        if not isinstance(obj["supported_for_intended_use"], bool):
            raise EvidenceValidationError(
                "ProposalSnapshot.supported_for_intended_use: expected boolean")
        intended_use = _enum(obj["intended_use"], _INTENDED_USES,
                             "ProposalSnapshot.intended_use")
        generations = _generations(obj["dependency_generations"],
                                   "ProposalSnapshot.dependency_generations")
        fences = _generations(obj["dependency_fence_generations"],
                              "ProposalSnapshot.dependency_fence_generations")
        frontiers = _generations(obj["dependency_frontiers"],
                                 "ProposalSnapshot.dependency_frontiers")
        evidence_digests = _digest_map(
            obj["dependency_evidence_digests"],
            "ProposalSnapshot.dependency_evidence_digests", empty=False)
        if not (set(generations) == set(fences) == set(frontiers)
                == set(evidence_digests)):
            raise EvidenceValidationError(
                "ProposalSnapshot dependency generation/fence/frontier/evidence keys differ")
        rule_identity = _text(
            obj["support_rule_identity"],
            "ProposalSnapshot.support_rule_identity", nullable=True)
        if (intended_use in _VERIFIED_USES
                and obj["supported_for_intended_use"] and rule_identity is None):
            raise EvidenceValidationError(
                "ProposalSnapshot supported verified use requires support_rule_identity")
        return cls(PROPOSAL_SCHEMA, _digest(obj["claim_digest"], "ProposalSnapshot.claim_digest"),
                   intended_use,
                   generations, fences, frontiers, evidence_digests,
                   _digest_map(obj["semantic_fences"],
                               "ProposalSnapshot.semantic_fences", empty=False),
                   cast(str, _text(obj["current_epoch"],
                                   "ProposalSnapshot.current_epoch")),
                   rule_identity,
                   _int(obj["global_fence_generation"],
                        "ProposalSnapshot.global_fence_generation"),
                   _int(obj["snapshot_frontier"], "ProposalSnapshot.snapshot_frontier"),
                   _digest(obj["retrieval_result_digest"],
                           "ProposalSnapshot.retrieval_result_digest"),
                   obj["retrieval_complete"], obj["supported_for_intended_use"],
                   _digest(obj["index_digest"], "ProposalSnapshot.index_digest"))

    def to_dict(self) -> dict[str, Any]:
        return {name: _thaw(getattr(self, name)) for name in self.FIELDS}


@dataclass(frozen=True)
class AdmissionResult:
    status: str
    affected_dependencies: tuple[str, ...]
    reasons: tuple[str, ...]


def _normalize(cls: Any, value: Any, label: str) -> Any:
    if not isinstance(value, cls):
        raise EvidenceValidationError(f"{label}: expected {cls.__name__}")
    try:
        return cls.from_dict(value.to_dict())
    except EvidenceValidationError:
        raise
    except Exception as exc:
        raise EvidenceValidationError(f"{label}: invalid direct construction: {exc}") from exc


def _mandatory_signature(claim: ClaimKey) -> tuple[str, ...]:
    scope = claim.target_scope
    return (
        schemas.content_hash(_thaw(claim.mechanism_identity)),
        schemas.content_hash(_thaw(claim.effect_question)), claim.estimand, claim.metric,
        scope["backend"], scope["model"], scope["quant"], scope["workload"],
    )


def _mandatory_signature_digest(claim: ClaimKey) -> str:
    return schemas.content_hash(list(_mandatory_signature(claim)))


_EMPTY_EVIDENCE_DIGEST = schemas.content_hash([])


@dataclass(frozen=True)
class PlanningEvidence:
    retrieval: RetrievalResult
    snapshot: ProposalSnapshot


@dataclass(frozen=True)
class ProjectionCompleteness:
    """Additional negative evidence only; this never grants supported use."""

    evicted_keys: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        keys = tuple(sorted(set(tuple(item) for item in self.evicted_keys)))
        if any(len(item) != 2 or item[0] not in {"dependency", "signature"}
               or not isinstance(item[1], str) or not item[1] for item in keys):
            raise EvidenceValidationError("invalid projection completeness keys")
        object.__setattr__(self, "evicted_keys", keys)

    @property
    def digest(self) -> str:
        return schemas.content_hash({"evicted_keys": [list(item) for item in self.evicted_keys]})

    def bind(self, fences: Mapping[str, str]) -> Mapping[str, str]:
        return MappingProxyType({key: schemas.content_hash({
            "local": value, "projection_completeness": self.digest})
            for key, value in fences.items()})


class EvidenceIndex:
    """In-memory projection with prebuilt claim/dependency reverse indices."""

    def __init__(self, findings: Iterable[Finding], invalidations: Iterable[Any] = (), *,
                 current_epoch: str, projection_available: bool = True,
                 scope_verifier: Callable[[Finding, ClaimKey, Mapping[str, Any]], bool | str] | None = None,
                 use_verifier: Callable[[Finding, str], bool | str] | None = None,
                 result_verifier: Callable[[tuple[RetrievedFinding, ...],
                                            tuple[RetrievedFinding, ...],
                                            ClaimKey, str], bool | str] | None = None,
                 support_rule_identity: str | None = None):
        self.current_epoch = cast(str, _text(current_epoch, "EvidenceIndex.current_epoch"))
        if not isinstance(projection_available, bool):
            raise EvidenceValidationError("EvidenceIndex.projection_available: expected boolean")
        self.projection_available = projection_available
        self._scope_verifier = _callback(scope_verifier, "EvidenceIndex.scope_verifier")
        self._use_verifier = _callback(use_verifier, "EvidenceIndex.use_verifier")
        self._result_verifier = _callback(result_verifier, "EvidenceIndex.result_verifier")
        self._support_rule_identity = _text(
            support_rule_identity, "EvidenceIndex.support_rule_identity", nullable=True)
        self._recorded_support_rule_identity = self._support_rule_identity
        normalized = tuple(_normalize(Finding, row, "EvidenceIndex.findings")
                           for row in findings)
        ids = [row.finding_id for row in normalized]
        if len(set(ids)) != len(ids):
            raise EvidenceValidationError("EvidenceIndex.findings: duplicate finding_id")
        self._findings = list(normalized)
        self._finding_digests = {row.finding_id: row.digest for row in normalized}
        self._by_claim: dict[str, list[Finding]] = {}
        claim_build: dict[str, list[Finding]] = defaultdict(list)
        mandatory_build: dict[tuple[str, ...], list[Finding]] = defaultdict(list)
        observed_signatures: set[tuple[str, ...]] = set()
        dep_build: dict[str, set[str]] = defaultdict(set)
        for row in normalized:
            claim_build[row.claim_key.digest].append(row)
            observed_signatures.add(_mandatory_signature(row.claim_key))
            if row.conclusion in {"refutation", "conflict", "retraction"}:
                mandatory_build[_mandatory_signature(row.claim_key)].append(row)
            for dep in row.claim_key.dependency_identities:
                dep_build[dep].add(row.claim_key.digest)
        self._by_claim = dict(claim_build)
        self._mandatory_by_signature = {
            key: list(value) for key, value in mandatory_build.items()}
        self._semantic_fences = {
            schemas.content_hash(list(key)): schemas.content_hash(
                sorted(row.digest for row in mandatory_build.get(key, ())))
            for key in observed_signatures}
        self._mandatory_digests = {
            schemas.content_hash(list(key)): {row.digest for row in rows}
            for key, rows in mandatory_build.items()}
        self._dirty_signatures: set[str] = set()
        self._dependency_claims = {key: set(value) for key, value in dep_build.items()}
        self._generations: dict[str, int] = {
            dep: 0 for row in normalized for dep in row.claim_key.dependency_identities}
        self._dependency_fences: dict[str, int] = {
            dep: 0 for dep in self._generations}
        self._dependency_frontiers: dict[str, int] = {
            dep: max((row.frontier for row in normalized
                      if dep in row.claim_key.dependency_identities), default=0)
            for dep in self._generations}
        self._dependency_finding_digests = {
            dep: {row.digest for row in normalized
                  if dep in row.claim_key.dependency_identities}
            for dep in self._generations}
        self._dependency_evidence_digests: dict[str, str] = {
            dep: schemas.content_hash(sorted(digests))
            for dep, digests in self._dependency_finding_digests.items()}
        self._dirty_dependencies: set[str] = set()
        self._global_fence_generation = 0
        self._invalidations: list[InvalidationEvent] = []
        self._quarantines: list[Quarantine] = []
        self._quarantine_digests: dict[str, str] = {}
        self._seen_events: dict[str, str] = {}
        self._frontier = max((row.frontier for row in normalized), default=0)
        for raw in invalidations:
            self._ingest_invalidation(raw)

    @classmethod
    def replay(cls, findings: Iterable[Finding], invalidations: Iterable[Any], **kwargs: Any) -> "EvidenceIndex":
        return cls(findings, invalidations, **kwargs)

    @property
    def frontier(self) -> int:
        return self._frontier

    @property
    def dependency_generations(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self._generations))

    @property
    def findings(self) -> tuple[Finding, ...]:
        return tuple(self._findings)

    @property
    def quarantines(self) -> tuple[Quarantine, ...]:
        return tuple(self._quarantines)

    def ingest_finding(self, raw: Finding | Mapping[str, Any]) -> bool:
        """Incrementally add one finding without rebuilding unrelated indices."""
        finding = (Finding.from_dict(raw) if isinstance(raw, Mapping)
                   else _normalize(Finding, raw, "EvidenceIndex.ingest_finding"))
        prior = self._finding_digests.get(finding.finding_id)
        if prior is not None:
            if prior != finding.digest:
                raise EvidenceValidationError(
                    f"finding_id {finding.finding_id!r} reused with different content")
            return False
        self._finding_digests[finding.finding_id] = finding.digest
        self._findings.append(finding)
        claim_digest = finding.claim_key.digest
        self._by_claim.setdefault(claim_digest, []).append(finding)
        signature = _mandatory_signature(finding.claim_key)
        signature_digest = schemas.content_hash(list(signature))
        self._semantic_fences.setdefault(signature_digest, _EMPTY_EVIDENCE_DIGEST)
        if finding.conclusion in {"refutation", "conflict", "retraction"}:
            self._mandatory_by_signature.setdefault(signature, []).append(finding)
            self._mandatory_digests.setdefault(signature_digest, set()).add(finding.digest)
            self._dirty_signatures.add(signature_digest)
        for dep in finding.claim_key.dependency_identities:
            self._dependency_claims.setdefault(dep, set()).add(claim_digest)
            self._generations.setdefault(dep, 0)
            self._dependency_fences.setdefault(dep, 0)
            self._dependency_frontiers[dep] = max(
                self._dependency_frontiers.get(dep, 0), finding.frontier)
            self._dependency_finding_digests.setdefault(dep, set()).add(finding.digest)
            self._dirty_dependencies.add(dep)
        self._frontier = max(self._frontier, finding.frontier)
        return True

    def _refresh_fence_digests(self) -> None:
        for dep in self._dirty_dependencies:
            self._dependency_evidence_digests[dep] = schemas.content_hash(
                sorted(self._dependency_finding_digests[dep]))
        self._dirty_dependencies.clear()
        for signature in self._dirty_signatures:
            self._semantic_fences[signature] = schemas.content_hash(
                sorted(self._mandatory_digests.get(signature, ())))
        self._dirty_signatures.clear()

    def ingest_invalidation(self, raw: InvalidationEvent | Mapping[str, Any]) -> None:
        """Incrementally apply one dependency-generation event."""
        self._ingest_invalidation(raw)

    def ingest_quarantine(self, raw: Quarantine | Mapping[str, Any]) -> bool:
        """Incrementally apply one already-validated feed quarantine."""
        row = (Quarantine.from_dict(raw) if isinstance(raw, Mapping)
               else _normalize(Quarantine, raw, "EvidenceIndex.ingest_quarantine"))
        digest = schemas.content_hash(row.to_dict())
        prior = self._quarantine_digests.get(row.event_id)
        if prior is not None:
            if prior != digest:
                raise EvidenceValidationError(
                    f"quarantine event_id {row.event_id!r} reused with different content")
            return False
        self._quarantine_digests[row.event_id] = digest
        self._quarantines.append(row)
        if row.global_scope:
            self._global_fence_generation += 1
        else:
            for dep in row.affected_dependencies:
                self._dependency_fences[dep] = self._dependency_fences.get(dep, 0) + 1
                self._generations.setdefault(dep, 0)
                self._dependency_evidence_digests.setdefault(
                    dep, _EMPTY_EVIDENCE_DIGEST)
                self._dependency_finding_digests.setdefault(dep, set())
                self._dependency_frontiers[dep] = max(
                    self._dependency_frontiers.get(dep, 0), row.frontier)
        self._frontier = max(self._frontier, row.frontier)
        return True

    def set_projection_available(self, available: bool) -> None:
        if not isinstance(available, bool):
            raise EvidenceValidationError("projection availability must be boolean")
        self.projection_available = available

    def _quarantine(self, raw: Any, reason: str) -> None:
        deps: tuple[str, ...] = ()
        event_id = "unidentified"
        frontier = self._frontier
        if isinstance(raw, Mapping):
            candidate = raw.get("event_id")
            if isinstance(candidate, str) and candidate:
                event_id = candidate
            candidate_frontier = raw.get("frontier")
            if isinstance(candidate_frontier, int) and not isinstance(candidate_frontier, bool):
                frontier = max(0, candidate_frontier)
            candidates = raw.get("dependency_ids")
            if isinstance(candidates, list) and all(isinstance(item, str) and item for item in candidates):
                deps = tuple(sorted(set(candidates)))
            elif isinstance(raw.get("dependency_id"), str) and raw["dependency_id"]:
                deps = (raw["dependency_id"],)
        try:
            digest = schemas.content_hash(_thaw(raw))
        except Exception:
            digest = schemas.content_hash({"unhashable_type": type(raw).__name__,
                                           "event_id": event_id})
        for prior in self._quarantines:
            if prior.event_id == event_id and prior.event_digest == digest:
                return
        if any(prior.event_id == event_id for prior in self._quarantines):
            event_id = f"{event_id}#conflict-{digest[:12]}"
        self._quarantines.append(Quarantine(
            QUARANTINE_SCHEMA, event_id, digest, reason, deps, not bool(deps), frontier))
        if deps:
            for dep in deps:
                self._dependency_fences[dep] = self._dependency_fences.get(dep, 0) + 1
                self._generations.setdefault(dep, 0)
                self._dependency_evidence_digests.setdefault(
                    dep, _EMPTY_EVIDENCE_DIGEST)
                self._dependency_frontiers[dep] = max(
                    self._dependency_frontiers.get(dep, 0), frontier)
        else:
            self._global_fence_generation += 1
        self._frontier = max(self._frontier, frontier)

    def _ingest_invalidation(self, raw: Any) -> None:
        try:
            event = (InvalidationEvent.from_dict(raw) if isinstance(raw, Mapping)
                     else _normalize(InvalidationEvent, raw, "invalidation"))
        except EvidenceValidationError as exc:
            self._quarantine(raw, str(exc))
            return
        prior_digest = self._seen_events.get(event.event_id)
        if prior_digest is not None:
            if prior_digest == event.digest:
                return
            self._quarantine(event.to_dict(), "event_id reused with different content")
            return
        current = self._generations.get(event.dependency_id, 0)
        if event.generation != current + 1:
            self._quarantine(
                event.to_dict(),
                f"out-of-order/conflicting generation: expected {current + 1}, got {event.generation}")
            return
        self._seen_events[event.event_id] = event.digest
        self._generations[event.dependency_id] = event.generation
        self._dependency_fences.setdefault(event.dependency_id, 0)
        self._dependency_evidence_digests.setdefault(
            event.dependency_id, _EMPTY_EVIDENCE_DIGEST)
        self._dependency_frontiers[event.dependency_id] = max(
            self._dependency_frontiers.get(event.dependency_id, 0), event.frontier)
        self._invalidations.append(event)
        self._frontier = max(self._frontier, event.frontier)

    def _claim_quarantines(self, claim: ClaimKey) -> tuple[Quarantine, ...]:
        dependencies = set(claim.dependency_identities)
        return tuple(row for row in self._quarantines
                     if row.global_scope or dependencies.intersection(row.affected_dependencies))

    def _finding_stale(self, finding: Finding) -> bool:
        return any(self._generations.get(dep, 0) != finding.dependency_generations.get(dep, -1)
                   for dep in finding.claim_key.dependency_identities)

    def retrieve(self, scope: Mapping[str, Any], claim_key: ClaimKey,
                 intended_use: str, limit: int = 40, *,
                 projection_completeness: ProjectionCompleteness | None = None
                 ) -> RetrievalResult:
        claim = _normalize(ClaimKey, claim_key, "retrieve.claim_key")
        normalized_scope = _mapping(scope, "retrieve.scope")
        if _thaw(normalized_scope) != _thaw(claim.target_scope):
            raise EvidenceValidationError("retrieve.scope differs from ClaimKey.target_scope")
        limit = _int(limit, "retrieve.limit", minimum=1)
        if limit > 40:
            raise EvidenceValidationError("retrieve.limit must be <= 40")
        intended_use = _enum(intended_use, _INTENDED_USES, "retrieve.intended_use")
        verified_use = intended_use in _VERIFIED_USES
        retrieval_reasons: list[str] = []
        support_reasons: list[str] = []
        if projection_completeness is not None:
            if not isinstance(projection_completeness, ProjectionCompleteness):
                raise EvidenceValidationError("projection completeness must be typed")
            if projection_completeness.evicted_keys:
                retrieval_reasons.append(
                    "relevant durable projection was evicted: "
                    + projection_completeness.digest)
        if not self.projection_available:
            retrieval_reasons.append("asynchronous projection unavailable; local fences retained")
        if verified_use and (self._scope_verifier is None or self._use_verifier is None
                             or self._result_verifier is None):
            support_reasons.append(
                "trusted registered scope/use/full-result verifier is not connected")
        if verified_use and self._support_rule_identity is None:
            support_reasons.append(
                "registered support verifier/rule identity is not connected")

        ordinary: list[RetrievedFinding] = []
        mandatory: list[RetrievedFinding] = []
        exact_rows = tuple(self._by_claim.get(claim.digest, ()))
        broad_mandatory = tuple(self._mandatory_by_signature.get(
            _mandatory_signature(claim), ()))
        candidates = list(exact_rows)
        exact_ids = {row.finding_id for row in exact_rows}
        candidates.extend(row for row in broad_mandatory if row.finding_id not in exact_ids)
        for finding in candidates:
            is_mandatory = finding.conclusion in {"refutation", "conflict", "retraction"}
            same_claim = finding.claim_key.digest == claim.digest
            exact = (_thaw(finding.tested_scope) == _thaw(normalized_scope)
                     and _thaw(finding.tested_question) == _thaw(claim.effect_question))
            if not is_mandatory and (not same_claim or not exact):
                continue
            if is_mandatory and _thaw(finding.tested_question) != _thaw(claim.effect_question):
                continue
            stale_dependency = self._finding_stale(finding)
            applicability = "stale_dependency" if stale_dependency else (
                "exact" if exact else "broader_scope_candidate")
            scope_check: bool | str = exact
            if not stale_dependency and self._scope_verifier is not None:
                scope_check = _verify(self._scope_verifier, finding, claim, normalized_scope)
                if scope_check is not True:
                    if (is_mandatory and not exact and isinstance(scope_check, str)
                            and scope_check.startswith("trusted verifier failed closed")):
                        retrieval_reasons.append(
                            "broader-scope applicability verifier failed closed")
                    if is_mandatory and not exact:
                        continue
                    applicability = "scope_unverified"
                elif not exact:
                    applicability = "trusted_broader_scope"
            elif is_mandatory and not exact:
                retrieval_reasons.append(
                    "broader-scope mandatory candidate lacks trusted applicability verifier")
                continue
            use_check: bool | str = False
            if verified_use and not stale_dependency and self._use_verifier is not None:
                use_check = _verify(self._use_verifier, finding, intended_use)
                if use_check is not True:
                    applicability = "use_unverified"
            is_search = finding.record_class in {"discovery_screen", "strict_search"}
            stale_epoch = is_search and finding.epoch != self.current_epoch
            ranking_value = (finding.value if finding.conclusion == "positive"
                             and is_search and not stale_epoch
                             and intended_use == "rank" and use_check is True
                             and applicability == "exact" else None)
            magnitude = ("stale_cross_epoch" if stale_epoch else
                         "tested_null" if finding.conclusion == "null" else
                         "not_numeric" if finding.value is None else "current_epoch")
            row = RetrievedFinding(finding, applicability, magnitude, ranking_value)
            if is_mandatory:
                mandatory.append(row)
            else:
                ordinary.append(row)

        if verified_use and any(row.applicability == "stale_dependency"
                                for row in ordinary + mandatory):
            support_reasons.append("applicable finding dependency generation is stale")
        if verified_use and any(row.applicability in {"scope_unverified", "use_unverified"}
                                for row in ordinary + mandatory):
            support_reasons.append("trusted verifier refused one or more applicable findings")
        if verified_use and not ordinary and not mandatory:
            support_reasons.append("no applicable findings for requested use")

        relevant_dependencies = set(claim.dependency_identities)
        for row in ordinary + mandatory:
            relevant_dependencies.update(row.finding.claim_key.dependency_identities)
        quarantine = tuple(row for row in self._quarantines
                           if row.global_scope or relevant_dependencies.intersection(
                               row.affected_dependencies))
        if quarantine:
            retrieval_reasons.append(
                "applicable invalidation quarantine makes retrieval incomplete")
        retrieval_complete = not retrieval_reasons
        supported = intended_use == "explore"
        if verified_use and retrieval_complete and not support_reasons and self._result_verifier:
            verdict = _verify(
                self._result_verifier, tuple(ordinary), tuple(mandatory), claim, intended_use)
            if verdict is True:
                supported = True
            else:
                support_reasons.append(
                    str(verdict) if isinstance(verdict, str)
                    else "trusted full-result verifier refused intended use")
        elif not verified_use and intended_use != "explore":
            support_reasons.append("intended use has no registered structural route")

        def rank(row: RetrievedFinding) -> tuple[Any, ...]:
            value = row.ranking_value
            numeric = value is not None and row.applicability == "exact"
            directed = (-value if claim.metric_direction == "higher" else value) if numeric else 0.0
            return (0 if numeric else 1, directed, row.finding.frontier, row.finding.finding_id)

        ordinary.sort(key=rank)
        mandatory.sort(key=lambda row: (row.finding.frontier, row.finding.finding_id))
        reasons = retrieval_reasons + support_reasons
        support_body = {
            "claim_digest": claim.digest, "intended_use": intended_use,
            "current_epoch": self.current_epoch,
            "support_rule_identity": self._support_rule_identity,
            "ordinary": [row.to_dict() for row in ordinary],
            "mandatory": [row.to_dict() for row in mandatory]}
        if projection_completeness is not None:
            support_body["projection_completeness"] = projection_completeness.digest
        support_basis_digest = schemas.content_hash(support_body)
        body = {"schema": RETRIEVAL_SCHEMA,
                "findings": [row.to_dict() for row in ordinary[:limit]],
                "mandatory_conflicts": [row.to_dict() for row in mandatory],
                "dependency_generations": {dep: self._generations.get(dep, 0)
                                           for dep in sorted(relevant_dependencies)},
                "snapshot_frontier": self._frontier,
                "retrieval_complete": retrieval_complete,
                "supported_for_intended_use": supported,
                "reasons": reasons,
                "quarantines": [row.to_dict() for row in quarantine],
                "support_basis_digest": support_basis_digest}
        return RetrievalResult(
            RETRIEVAL_SCHEMA, tuple(ordinary[:limit]), tuple(mandatory),
            MappingProxyType(body["dependency_generations"]), self._frontier,
            retrieval_complete, supported, tuple(reasons), quarantine,
            support_basis_digest, schemas.content_hash(body))

    def proposal_snapshot(self, claim_key: ClaimKey, *,
                          intended_use: str) -> ProposalSnapshot:
        return self.planning_evidence(claim_key, intended_use=intended_use).snapshot

    def planning_evidence(self, claim_key: ClaimKey, *, intended_use: str,
                          projection_completeness: ProjectionCompleteness | None = None
                          ) -> PlanningEvidence:
        """One verifier evaluation supplies both prompt retrieval and its snapshot."""
        self._refresh_fence_digests()
        claim = _normalize(ClaimKey, claim_key, "proposal_snapshot.claim_key")
        intended_use = _enum(intended_use, _INTENDED_USES,
                             "proposal_snapshot.intended_use")
        before = self.fence_snapshot()
        retrieval = self.retrieve(
            claim.target_scope, claim, intended_use, limit=40,
            projection_completeness=projection_completeness)
        if self.fence_snapshot() != before:
            raise EvidenceValidationError("evidence changed during its exact planning query")
        generations = MappingProxyType(dict(retrieval.dependency_generations))
        fences = MappingProxyType({dep: self._dependency_fences.get(dep, 0)
                                   for dep in generations})
        dependency_frontiers = MappingProxyType({
            dep: self._dependency_frontiers.get(dep, 0) for dep in generations})
        dependency_evidence_digests = MappingProxyType({
            dep: self._dependency_evidence_digests.get(dep, _EMPTY_EVIDENCE_DIGEST)
            for dep in generations})
        signature_key = _mandatory_signature_digest(claim)
        semantic_fences = MappingProxyType({
            signature_key: self._semantic_fences.get(
                signature_key, _EMPTY_EVIDENCE_DIGEST)})
        index_body = {"frontier": self._frontier,
                      "dependency_generations": dict(self._generations),
                      "dependency_fence_generations": dict(self._dependency_fences),
                      "dependency_evidence_digests": dict(
                          self._dependency_evidence_digests),
                      "semantic_fences": dict(self._semantic_fences),
                      "current_epoch": self.current_epoch,
                      "support_rule_identity": self._support_rule_identity,
                      "global_fence_generation": self._global_fence_generation,
                      "quarantines": [row.to_dict() for row in self._quarantines]}
        snapshot = ProposalSnapshot(PROPOSAL_SCHEMA, claim.digest, intended_use,
                                generations, fences, dependency_frontiers,
                                dependency_evidence_digests, semantic_fences,
                                self.current_epoch, self._support_rule_identity,
                                self._global_fence_generation,
                                self._frontier, retrieval.result_digest,
                                retrieval.retrieval_complete,
                                retrieval.supported_for_intended_use,
                                schemas.content_hash(index_body))
        if projection_completeness is not None:
            snapshot = replace(snapshot,
                semantic_fences=projection_completeness.bind(snapshot.semantic_fences),
                index_digest=schemas.content_hash({
                    "index": snapshot.index_digest,
                    "projection_completeness": projection_completeness.digest}))
        return PlanningEvidence(retrieval, snapshot)

    def fence_snapshot(self) -> LocalFenceSnapshot:
        self._refresh_fence_digests()
        return LocalFenceSnapshot(
            LOCAL_FENCE_SCHEMA, self.projection_available,
            self._global_fence_generation,
            MappingProxyType(dict(sorted(self._generations.items()))),
            MappingProxyType(dict(sorted(self._dependency_fences.items()))),
            MappingProxyType(dict(sorted(self._dependency_frontiers.items()))),
            MappingProxyType(dict(sorted(self._dependency_evidence_digests.items()))),
            MappingProxyType(dict(sorted(self._semantic_fences.items()))),
            self.current_epoch, self._support_rule_identity,
            self._frontier)

    @staticmethod
    def admit_cached(
            proposal: ProposalSnapshot, local_fences: LocalFenceSnapshot, *,
            intended_use: str,
            authority_verifier: Callable[[ProposalSnapshot], bool | str] | None = None,
            ) -> AdmissionResult:
        proposal = _normalize(ProposalSnapshot, proposal, "admit_cached.proposal")
        if not isinstance(local_fences, LocalFenceSnapshot):
            raise EvidenceValidationError(
                "admit_cached.local_fences: expected LocalFenceSnapshot")
        local = local_fences
        if local.schema != LOCAL_FENCE_SCHEMA:
            raise EvidenceValidationError(
                f"admit_cached.local_fences.schema: unsupported {local.schema!r}")
        if not isinstance(local.available, bool):
            raise EvidenceValidationError(
                "admit_cached.local_fences.available: expected boolean")
        _int(local.global_fence_generation,
             "admit_cached.local_fences.global_fence_generation")
        _int(local.frontier, "admit_cached.local_fences.frontier")
        if not isinstance(local.dependency_generations, Mapping):
            raise EvidenceValidationError(
                "admit_cached.local_fences.dependency_generations: expected object")
        if not isinstance(local.dependency_fence_generations, Mapping):
            raise EvidenceValidationError(
                "admit_cached.local_fences.dependency_fence_generations: expected object")
        if not isinstance(local.dependency_frontiers, Mapping):
            raise EvidenceValidationError(
                "admit_cached.local_fences.dependency_frontiers: expected object")
        if not isinstance(local.dependency_evidence_digests, Mapping):
            raise EvidenceValidationError(
                "admit_cached.local_fences.dependency_evidence_digests: expected object")
        if not isinstance(local.semantic_fences, Mapping):
            raise EvidenceValidationError(
                "admit_cached.local_fences.semantic_fences: expected object")
        local_epoch = cast(str, _text(
            local.current_epoch, "admit_cached.local_fences.current_epoch"))
        local_rule = _text(
            local.support_rule_identity,
            "admit_cached.local_fences.support_rule_identity", nullable=True)
        intended_use = _enum(intended_use, _INTENDED_USES, "admit_cached.intended_use")
        _callback(authority_verifier, "admit_cached.authority_verifier")
        if intended_use != proposal.intended_use:
            return AdmissionResult(
                "incomplete", (), ("cached proposal intended_use differs from admission use",))
        if not local.available:
            return AdmissionResult(
                "incomplete", (), ("local fence snapshot is unavailable",))
        if local_epoch != proposal.current_epoch:
            return AdmissionResult(
                "stale", (), ("support epoch changed",))
        if intended_use in _VERIFIED_USES and (
                proposal.support_rule_identity is None or local_rule is None):
            return AdmissionResult(
                "incomplete", (),
                ("registered support verifier/rule identity is unavailable",))
        if local_rule != proposal.support_rule_identity:
            return AdmissionResult(
                "incomplete", (), ("registered support verifier/rule identity changed",))
        if local.global_fence_generation != proposal.global_fence_generation:
            return AdmissionResult(
                "stale", (), ("global uncertainty fence changed",))
        affected = []
        # Deliberately O(number of proposal dependencies): no finding/index scan or I/O.
        for dep, generation in proposal.dependency_generations.items():
            try:
                local_generation = local.dependency_generations[dep]
            except KeyError:
                return AdmissionResult(
                    "incomplete", (dep,), ("local dependency generation is missing",))
            try:
                local_fence = local.dependency_fence_generations[dep]
            except KeyError:
                return AdmissionResult(
                    "incomplete", (dep,), ("local dependency fence is missing",))
            try:
                local_frontier = local.dependency_frontiers[dep]
            except KeyError:
                return AdmissionResult(
                    "incomplete", (dep,), ("local dependency frontier is missing",))
            try:
                local_evidence_digest = local.dependency_evidence_digests[dep]
            except KeyError:
                return AdmissionResult(
                    "incomplete", (dep,),
                    ("local dependency evidence digest is missing",))
            local_generation = _int(
                local_generation,
                f"admit_cached.local_fences.dependency_generations.{dep}")
            local_fence = _int(
                local_fence,
                f"admit_cached.local_fences.dependency_fence_generations.{dep}")
            local_frontier = _int(
                local_frontier,
                f"admit_cached.local_fences.dependency_frontiers.{dep}")
            local_evidence_digest = _digest(
                local_evidence_digest,
                f"admit_cached.local_fences.dependency_evidence_digests.{dep}")
            if local_generation != generation:
                affected.append(dep)
            elif local_fence != proposal.dependency_fence_generations[dep]:
                affected.append(dep)
            elif local_frontier != proposal.dependency_frontiers[dep]:
                affected.append(dep)
            elif local_evidence_digest != proposal.dependency_evidence_digests[dep]:
                affected.append(dep)
        if affected:
            return AdmissionResult("stale", tuple(sorted(affected)),
                                   ("relevant dependency generation or fence changed",))
        changed_semantic = []
        for signature, expected in proposal.semantic_fences.items():
            try:
                actual = local.semantic_fences[signature]
            except KeyError:
                return AdmissionResult(
                    "incomplete", (),
                    ("local semantic retrieval-set fence is missing",))
            actual = _digest(
                actual, f"admit_cached.local_fences.semantic_fences.{signature}")
            if actual != expected:
                changed_semantic.append(signature)
        if changed_semantic:
            return AdmissionResult(
                "stale", (), ("relevant semantic retrieval-set fence changed",))
        if not proposal.retrieval_complete:
            return AdmissionResult("incomplete", (),
                                   ("proposal retrieval was incomplete",))
        if not proposal.supported_for_intended_use:
            return AdmissionResult("incomplete", (),
                                   ("proposal was not supported for its intended use",))
        if authority_verifier is None:
            return AdmissionResult("incomplete", (),
                                   ("trusted cached-admission verifier is not connected",))
        verified = _verify(authority_verifier, proposal)
        if verified is not True:
            return AdmissionResult(
                "incomplete", (),
                (str(verified) if isinstance(verified, str)
                 else "trusted cached-admission verifier refused",))
        return AdmissionResult("eligible", (), ())

    def to_dict(self) -> dict[str, Any]:
        self._refresh_fence_digests()
        body = {"schema": INDEX_SCHEMA, "current_epoch": self.current_epoch,
                "projection_available": self.projection_available,
                "findings": [row.to_dict() for row in self.findings],
                "invalidations": [row.to_dict() for row in self._invalidations],
                "quarantines": [row.to_dict() for row in self._quarantines],
                "dependency_generations": dict(sorted(self._generations.items())),
                "dependency_fence_generations": dict(
                    sorted(self._dependency_fences.items())),
                "dependency_frontiers": dict(sorted(self._dependency_frontiers.items())),
                "dependency_evidence_digests": dict(
                    sorted(self._dependency_evidence_digests.items())),
                "semantic_fences": dict(sorted(self._semantic_fences.items())),
                "support_rule_identity": self._recorded_support_rule_identity,
                "global_fence_generation": self._global_fence_generation,
                "snapshot_frontier": self._frontier}
        return body | {"index_digest": schemas.content_hash(body)}

    @classmethod
    def from_dict(
            cls, obj: Mapping[str, Any], *,
            scope_verifier: Callable[[Finding, ClaimKey, Mapping[str, Any]], bool | str] | None = None,
            use_verifier: Callable[[Finding, str], bool | str] | None = None,
            result_verifier: Callable[[tuple[RetrievedFinding, ...],
                                       tuple[RetrievedFinding, ...], ClaimKey, str],
                                      bool | str] | None = None,
            support_rule_identity: str | None = None) -> "EvidenceIndex":
        fields = {"schema", "current_epoch", "projection_available", "findings",
                  "invalidations", "quarantines", "dependency_generations",
                  "dependency_fence_generations", "dependency_frontiers",
                  "dependency_evidence_digests", "semantic_fences",
                  "support_rule_identity",
                  "global_fence_generation",
                  "snapshot_frontier", "index_digest"}
        _exact(obj, fields, "EvidenceIndex projection")
        if obj["schema"] != INDEX_SCHEMA:
            raise EvidenceValidationError(
                f"EvidenceIndex.schema: unsupported {obj['schema']!r}")
        unsigned = {key: _thaw(value) for key, value in obj.items()
                    if key != "index_digest"}
        if _digest(obj["index_digest"], "EvidenceIndex.index_digest") != schemas.content_hash(unsigned):
            raise EvidenceValidationError("EvidenceIndex.index_digest does not verify")
        if not isinstance(obj["projection_available"], bool):
            raise EvidenceValidationError("EvidenceIndex.projection_available: expected boolean")
        if not isinstance(obj["findings"], (list, tuple)):
            raise EvidenceValidationError("EvidenceIndex.findings: expected array")
        if not isinstance(obj["invalidations"], (list, tuple)):
            raise EvidenceValidationError("EvidenceIndex.invalidations: expected array")
        if not isinstance(obj["quarantines"], (list, tuple)):
            raise EvidenceValidationError("EvidenceIndex.quarantines: expected array")
        index = cls(
            tuple(Finding.from_dict(row) for row in obj["findings"]),
            tuple(InvalidationEvent.from_dict(row) for row in obj["invalidations"]),
            current_epoch=obj["current_epoch"],
            projection_available=obj["projection_available"],
            scope_verifier=scope_verifier, use_verifier=use_verifier,
            result_verifier=result_verifier,
            support_rule_identity=support_rule_identity)
        recorded_rule = _text(
            obj["support_rule_identity"],
            "EvidenceIndex.support_rule_identity", nullable=True)
        index._recorded_support_rule_identity = recorded_rule
        quarantines = tuple(Quarantine.from_dict(row) for row in obj["quarantines"])
        quarantine_ids = [row.event_id for row in quarantines]
        if len(set(quarantine_ids)) != len(quarantine_ids):
            raise EvidenceValidationError("EvidenceIndex.quarantines: duplicate event_id")
        index._quarantines = list(quarantines)
        index._quarantine_digests = {
            row.event_id: schemas.content_hash(row.to_dict()) for row in quarantines}
        index._dependency_fences = {dep: 0 for dep in index._generations}
        index._global_fence_generation = 0
        for row in quarantines:
            if row.global_scope:
                index._global_fence_generation += 1
            else:
                for dep in row.affected_dependencies:
                    index._dependency_fences[dep] = index._dependency_fences.get(dep, 0) + 1
                    index._generations.setdefault(dep, 0)
        expected_generations = dict(_generations_allow_empty(
            obj["dependency_generations"], "EvidenceIndex.dependency_generations"))
        expected_fences = dict(_generations_allow_empty(
            obj["dependency_fence_generations"],
            "EvidenceIndex.dependency_fence_generations"))
        if expected_generations != index._generations:
            raise EvidenceValidationError("EvidenceIndex dependency generations do not replay")
        if expected_fences != index._dependency_fences:
            raise EvidenceValidationError("EvidenceIndex dependency fences do not replay")
        expected_dependency_frontiers = dict(_generations_allow_empty(
            obj["dependency_frontiers"], "EvidenceIndex.dependency_frontiers"))
        for row in quarantines:
            for dep in row.affected_dependencies:
                index._dependency_frontiers[dep] = max(
                    index._dependency_frontiers.get(dep, 0), row.frontier)
        if expected_dependency_frontiers != index._dependency_frontiers:
            raise EvidenceValidationError("EvidenceIndex dependency frontiers do not replay")
        expected_evidence_digests = dict(_digest_map(
            obj["dependency_evidence_digests"],
            "EvidenceIndex.dependency_evidence_digests"))
        if expected_evidence_digests != index._dependency_evidence_digests:
            raise EvidenceValidationError(
                "EvidenceIndex dependency evidence digests do not replay")
        expected_semantic_fences = dict(_digest_map(
            obj["semantic_fences"], "EvidenceIndex.semantic_fences"))
        if expected_semantic_fences != index._semantic_fences:
            raise EvidenceValidationError(
                "EvidenceIndex semantic fences do not replay")
        if (_int(obj["global_fence_generation"], "EvidenceIndex.global_fence_generation")
                != index._global_fence_generation):
            raise EvidenceValidationError("EvidenceIndex global fence does not replay")
        frontier = _int(obj["snapshot_frontier"], "EvidenceIndex.snapshot_frontier")
        recomputed_frontier = max(
            [index._frontier] + [row.frontier for row in quarantines])
        if frontier != recomputed_frontier:
            raise EvidenceValidationError("EvidenceIndex snapshot frontier does not replay")
        index._frontier = frontier
        return index


@dataclass(frozen=True)
class TransferReceipt:
    schema: str
    receipt_id: str
    source_scope: Mapping[str, Any]
    destination_scope: Mapping[str, Any]
    transfer_type: str
    mechanism_identity: Mapping[str, Any]
    intervention_identity: Mapping[str, Any]
    effect_question: Mapping[str, Any]
    dependency_identities: Mapping[str, str]
    authority_reference: str
    source_ref: SourceRef

    FIELDS = {"schema", "receipt_id", "source_scope", "destination_scope",
              "transfer_type", "mechanism_identity", "intervention_identity",
              "effect_question", "dependency_identities", "authority_reference",
              "source_ref"}

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "TransferReceipt":
        _exact(obj, cls.FIELDS, "TransferReceipt")
        if obj["schema"] != TRANSFER_SCHEMA:
            raise EvidenceValidationError(
                f"TransferReceipt.schema: unsupported {obj['schema']!r}")
        return cls(
            TRANSFER_SCHEMA,
            cast(str, _text(obj["receipt_id"], "TransferReceipt.receipt_id")),
            _mapping(obj["source_scope"], "TransferReceipt.source_scope"),
            _mapping(obj["destination_scope"], "TransferReceipt.destination_scope"),
            _enum(obj["transfer_type"], _TRANSFER_TYPES, "TransferReceipt.transfer_type"),
            _mapping(obj["mechanism_identity"], "TransferReceipt.mechanism_identity"),
            _mapping(obj["intervention_identity"], "TransferReceipt.intervention_identity"),
            _question(obj["effect_question"], "TransferReceipt.effect_question"),
            _dependency_identities(obj["dependency_identities"],
                                   "TransferReceipt.dependency_identities"),
            cast(str, _text(obj["authority_reference"],
                            "TransferReceipt.authority_reference")),
            SourceRef.from_dict(obj["source_ref"]))

    def to_dict(self) -> dict[str, Any]:
        result = {name: _thaw(getattr(self, name)) for name in self.FIELDS
                  if name != "source_ref"}
        result["source_ref"] = self.source_ref.to_dict()
        return result

    @property
    def digest(self) -> str:
        return schemas.content_hash(self.to_dict())


@dataclass(frozen=True)
class TransferDisposition:
    status: str
    receipt_id: str | None
    reasons: tuple[str, ...]


def transfer_disposition(
        receipts: Iterable[TransferReceipt], source_claim: ClaimKey,
        destination_claim: ClaimKey, intended_use: str, *,
        authority_verifier: Callable[[TransferReceipt, str], bool | str] | None
        ) -> TransferDisposition:
    """Check only an exact directed edge; never search paths or reverse it."""
    source = _normalize(ClaimKey, source_claim, "transfer.source_claim")
    destination = _normalize(ClaimKey, destination_claim, "transfer.destination_claim")
    _callback(authority_verifier, "transfer.authority_verifier")
    transfer_uses = frozenset({"explore", "correctness", "local_work",
                               "screen_out", "timing"})
    intended_use = _enum(intended_use, transfer_uses, "transfer.intended_use")
    allowed_uses = {
        "correctness": frozenset({"correctness"}),
        "local_work": frozenset({"explore", "local_work", "screen_out"}),
        "serving_effect": frozenset({"explore", "timing", "screen_out"}),
    }
    normalized_receipts = tuple(_normalize(TransferReceipt, raw, "transfer.receipt")
                                for raw in receipts)
    receipt_ids = [receipt.receipt_id for receipt in normalized_receipts]
    if len(set(receipt_ids)) != len(receipt_ids):
        raise EvidenceValidationError("transfer.receipts: duplicate receipt_id")
    for receipt in normalized_receipts:
        if (_thaw(receipt.source_scope) != _thaw(source.target_scope)
                or _thaw(receipt.destination_scope) != _thaw(destination.target_scope)):
            continue
        exact = (
            _thaw(receipt.mechanism_identity) == _thaw(source.mechanism_identity)
            == _thaw(destination.mechanism_identity)
            and _thaw(receipt.intervention_identity) == _thaw(source.intervention_identity)
            == _thaw(destination.intervention_identity)
            and _thaw(receipt.effect_question) == _thaw(source.effect_question)
            == _thaw(destination.effect_question)
            and _thaw(receipt.dependency_identities)
            == _thaw(destination.dependency_identities))
        if not exact:
            continue
        if intended_use not in allowed_uses[receipt.transfer_type]:
            return TransferDisposition(
                "refused", receipt.receipt_id,
                (f"{receipt.transfer_type} receipt does not authorize {intended_use}",))
        if authority_verifier is None:
            return TransferDisposition(
                "policy_undefined", receipt.receipt_id,
                ("trusted registered transfer-authority verifier is not connected",))
        verified = _verify(authority_verifier, receipt, intended_use)
        if verified is not True:
            return TransferDisposition(
                "refused", receipt.receipt_id,
                (str(verified) if isinstance(verified, str)
                 else "registered transfer-authority verifier refused",))
        return TransferDisposition("permitted", receipt.receipt_id, ())
    return TransferDisposition("unsupported", None,
                               ("no exact direct transfer receipt",))


@dataclass(frozen=True)
class Route:
    schema: str
    route_id: str
    revision: str
    source_scope: Mapping[str, Any]
    destination_scope: Mapping[str, Any]
    mechanism_identity: Mapping[str, Any]
    intervention_identity: Mapping[str, Any]
    dependency_identities: Mapping[str, str]
    preserved_dimensions: tuple[str, ...]
    required_path_witnesses: tuple[str, ...]
    covered_targets: tuple[Mapping[str, Any], ...]
    disposition: Mapping[str, Any]

    FIELDS = {"schema", "route_id", "revision", "source_scope", "destination_scope",
              "mechanism_identity", "intervention_identity", "dependency_identities",
              "preserved_dimensions", "required_path_witnesses", "covered_targets",
              "disposition"}

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "Route":
        _exact(obj, cls.FIELDS, "Route")
        if obj["schema"] != ROUTE_SCHEMA:
            raise EvidenceValidationError(f"Route.schema: unsupported {obj['schema']!r}")
        targets_obj = obj["covered_targets"]
        if not isinstance(targets_obj, (list, tuple)) or not targets_obj:
            raise EvidenceValidationError("Route.covered_targets: expected non-empty array")
        targets = tuple(_mapping(item, "Route.covered_targets[]") for item in targets_obj)
        target_digests = [schemas.content_hash(_thaw(item)) for item in targets]
        if len(set(target_digests)) != len(target_digests):
            raise EvidenceValidationError("Route.covered_targets: duplicate scope")
        disposition = obj["disposition"]
        _exact(disposition, {"kind", "scope", "effect_question"}, "Route.disposition")
        kind = _enum(disposition["kind"], _ROUTE_DISPOSITIONS, "Route.disposition.kind")
        if kind == "exploration_only":
            if disposition["scope"] is not None or disposition["effect_question"] is not None:
                raise EvidenceValidationError(
                    "Route.disposition: exploration_only requires null scope/question")
            normalized_disposition = {"kind": kind, "scope": None, "effect_question": None}
        else:
            normalized_disposition = {
                "kind": kind, "scope": _thaw(_mapping(
                    disposition["scope"], "Route.disposition.scope")),
                "effect_question": _thaw(_question(
                    disposition["effect_question"], "Route.disposition.effect_question")),
            }
        return cls(
            ROUTE_SCHEMA, cast(str, _text(obj["route_id"], "Route.route_id")),
            cast(str, _text(obj["revision"], "Route.revision")),
            _mapping(obj["source_scope"], "Route.source_scope"),
            _mapping(obj["destination_scope"], "Route.destination_scope"),
            _mapping(obj["mechanism_identity"], "Route.mechanism_identity"),
            _mapping(obj["intervention_identity"], "Route.intervention_identity"),
            _dependency_identities(obj["dependency_identities"], "Route.dependency_identities"),
            _strings(obj["preserved_dimensions"], "Route.preserved_dimensions"),
            _strings(obj["required_path_witnesses"], "Route.required_path_witnesses"),
            targets, _freeze(normalized_disposition, "Route.disposition"))

    def to_dict(self) -> dict[str, Any]:
        return {name: (_thaw(getattr(self, name))) for name in self.FIELDS}


@dataclass(frozen=True)
class RouteRevocation:
    schema: str
    event_id: str
    route_id: str
    route_revision: str
    destination_scope: Mapping[str, Any]
    dependency_generation: int
    audit_finding_id: str
    target_scale_success: bool

    FIELDS = {"schema", "event_id", "route_id", "route_revision",
              "destination_scope", "dependency_generation", "audit_finding_id",
              "target_scale_success"}

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "RouteRevocation":
        _exact(obj, cls.FIELDS, "RouteRevocation")
        if obj["schema"] != REVOCATION_SCHEMA:
            raise EvidenceValidationError(
                f"RouteRevocation.schema: unsupported {obj['schema']!r}")
        if not isinstance(obj["target_scale_success"], bool):
            raise EvidenceValidationError("RouteRevocation.target_scale_success: expected boolean")
        return cls(
            REVOCATION_SCHEMA, cast(str, _text(obj["event_id"], "RouteRevocation.event_id")),
            cast(str, _text(obj["route_id"], "RouteRevocation.route_id")),
            cast(str, _text(obj["route_revision"], "RouteRevocation.route_revision")),
            _mapping(obj["destination_scope"], "RouteRevocation.destination_scope"),
            _int(obj["dependency_generation"], "RouteRevocation.dependency_generation"),
            cast(str, _text(obj["audit_finding_id"], "RouteRevocation.audit_finding_id")),
            obj["target_scale_success"])

    def to_dict(self) -> dict[str, Any]:
        return {name: _thaw(getattr(self, name)) for name in self.FIELDS}


@dataclass(frozen=True)
class RouteDisposition:
    status: str
    reasons: tuple[str, ...]


def route_disposition(
        route: Route, source_claim: ClaimKey, destination_claim: ClaimKey, *,
        path_witnesses: Mapping[str, SourceRef], dependency_generation: int,
        revocations: Iterable[RouteRevocation] = (),
        authority_verifier: Callable[[Route, ClaimKey, ClaimKey, int,
                                      Mapping[str, SourceRef], str], bool | str] | None = None,
        audit_decision: "RejectAuditDecision | None" = None,
        ) -> RouteDisposition:
    route = _normalize(Route, route, "route_disposition.route")
    source = _normalize(ClaimKey, source_claim, "route_disposition.source")
    destination = _normalize(ClaimKey, destination_claim, "route_disposition.destination")
    generation = _int(dependency_generation, "route_disposition.dependency_generation")
    _callback(authority_verifier, "route_disposition.authority_verifier")
    if not isinstance(path_witnesses, Mapping):
        raise EvidenceValidationError("route_disposition.path_witnesses: expected mapping")
    witnesses = {name: _normalize(SourceRef, ref, f"path witness {name}")
                 for name, ref in path_witnesses.items()
                 if isinstance(name, str) and name}
    if len(witnesses) != len(path_witnesses):
        raise EvidenceValidationError("route_disposition.path_witnesses: invalid witness name")
    exact = (_thaw(route.source_scope) == _thaw(source.target_scope)
             and _thaw(route.destination_scope) == _thaw(destination.target_scope)
             and any(_thaw(target) == _thaw(destination.target_scope)
                     for target in route.covered_targets)
             and _thaw(route.mechanism_identity) == _thaw(source.mechanism_identity)
             == _thaw(destination.mechanism_identity)
             and _thaw(route.intervention_identity) == _thaw(source.intervention_identity)
             == _thaw(destination.intervention_identity)
             and _thaw(route.dependency_identities)
             == _thaw(destination.dependency_identities))
    if not exact:
        return RouteDisposition("exploration_only", ("route does not exactly cover destination",))
    if route.disposition["kind"] != "may_screen_out":
        return RouteDisposition("exploration_only", ("route has exploration-only disposition",))
    if (_thaw(route.disposition["scope"]) != _thaw(destination.target_scope)
            or _thaw(route.disposition["effect_question"])
            != _thaw(destination.effect_question)):
        return RouteDisposition("exploration_only", ("screen-out scope/question differs",))
    missing = set(route.required_path_witnesses) - set(witnesses)
    if missing:
        return RouteDisposition(
            "exploration_only", ("missing executed-path witnesses: " + ", ".join(sorted(missing)),))
    normalized_revocations = tuple(
        _normalize(RouteRevocation, raw, "route_disposition.revocation")
        for raw in revocations)
    event_ids = [row.event_id for row in normalized_revocations]
    if len(set(event_ids)) != len(event_ids):
        raise EvidenceValidationError("route_disposition.revocations: duplicate event_id")
    for revoked in normalized_revocations:
        if (revoked.target_scale_success and revoked.route_id == route.route_id
                and revoked.route_revision == route.revision
                and revoked.dependency_generation == generation
                and _thaw(revoked.destination_scope) == _thaw(destination.target_scope)):
            return RouteDisposition(
                "revoked", ("successful target-scale audit revoked this scoped route authority",))
    if authority_verifier is None:
        return RouteDisposition(
            "exploration_only", ("trusted route/path authority verifier is not connected",))
    verified = _verify(authority_verifier, route, source, destination, generation,
                       MappingProxyType(witnesses), "screen_out")
    if verified is not True:
        return RouteDisposition(
            "exploration_only",
            (str(verified) if isinstance(verified, str) else "route verifier refused",))
    if audit_decision is None:
        return RouteDisposition(
            "exploration_only", ("deterministic reject-audit selection is missing",))
    audit = _normalize(RejectAuditDecision, audit_decision, "route_disposition.audit")
    expected_digest = _audit_sample_digest(
        destination, route, audit.mechanism_stratum, audit.allocation_stratum)
    expected_selected = int(expected_digest, 16) / float(2 ** 256) < audit.selection_probability
    if audit.stable_sample_digest != expected_digest or audit.selected != expected_selected:
        raise EvidenceValidationError("route_disposition.audit: stable selection does not verify")
    if audit.status == "selected_budget_unavailable":
        return RouteDisposition(
            "audit_budget_unavailable",
            (f"selected audit lacks target-confirmation budget {audit.target_confirmation_budget_id}",))
    if audit.status == "selected":
        return RouteDisposition(
            "target_audit_required",
            (f"selected for target confirmation budget {audit.target_confirmation_budget_id}",))
    return RouteDisposition("may_screen_out", ())


@dataclass(frozen=True)
class RejectAuditDecision:
    schema: str
    status: str
    selected: bool
    selection_probability: float
    mechanism_stratum: str
    allocation_stratum: str
    target_confirmation_budget_id: str
    stable_sample_digest: str

    FIELDS = {"schema", "status", "selected", "selection_probability",
              "mechanism_stratum", "allocation_stratum",
              "target_confirmation_budget_id", "stable_sample_digest"}

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "RejectAuditDecision":
        _exact(obj, cls.FIELDS, "RejectAuditDecision")
        if obj["schema"] != AUDIT_SCHEMA:
            raise EvidenceValidationError(
                f"RejectAuditDecision.schema: unsupported {obj['schema']!r}")
        if not isinstance(obj["selected"], bool):
            raise EvidenceValidationError("RejectAuditDecision.selected: expected boolean")
        probability = _number(obj["selection_probability"],
                              "RejectAuditDecision.selection_probability", minimum=0.0)
        if probability > 1.0:
            raise EvidenceValidationError(
                "RejectAuditDecision.selection_probability must be <= 1")
        status = _enum(obj["status"], frozenset({
            "not_selected", "selected", "selected_budget_unavailable"}),
            "RejectAuditDecision.status")
        if (status == "not_selected") == obj["selected"]:
            raise EvidenceValidationError(
                "RejectAuditDecision.status disagrees with selected")
        return cls(
            AUDIT_SCHEMA, status, obj["selected"], probability,
            cast(str, _text(obj["mechanism_stratum"],
                            "RejectAuditDecision.mechanism_stratum")),
            cast(str, _text(obj["allocation_stratum"],
                            "RejectAuditDecision.allocation_stratum")),
            cast(str, _text(obj["target_confirmation_budget_id"],
                            "RejectAuditDecision.target_confirmation_budget_id")),
            _digest(obj["stable_sample_digest"],
                    "RejectAuditDecision.stable_sample_digest"))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "status": self.status,
                "selected": self.selected,
                "selection_probability": self.selection_probability,
                "mechanism_stratum": self.mechanism_stratum,
                "allocation_stratum": self.allocation_stratum,
                "target_confirmation_budget_id": self.target_confirmation_budget_id,
                "stable_sample_digest": self.stable_sample_digest}


def select_reject_audit(
        claim_key: ClaimKey, route: Route, *, probability: float,
        mechanism_stratum: str, allocation_stratum: str,
        target_confirmation_budget_id: str, budget_remaining: int,
        ) -> RejectAuditDecision:
    claim = _normalize(ClaimKey, claim_key, "select_reject_audit.claim_key")
    route = _normalize(Route, route, "select_reject_audit.route")
    probability = _number(probability, "select_reject_audit.probability", minimum=0.0)
    if probability > 1.0:
        raise EvidenceValidationError("select_reject_audit.probability must be <= 1")
    mechanism_stratum = cast(str, _text(mechanism_stratum, "mechanism_stratum"))
    allocation_stratum = cast(str, _text(allocation_stratum, "allocation_stratum"))
    budget_id = cast(str, _text(target_confirmation_budget_id,
                                "target_confirmation_budget_id"))
    budget = _int(budget_remaining, "budget_remaining")
    sample_digest = _audit_sample_digest(
        claim, route, mechanism_stratum, allocation_stratum)
    selected = int(sample_digest, 16) / float(2 ** 256) < probability
    status = ("not_selected" if not selected else
              "selected" if budget > 0 else "selected_budget_unavailable")
    return RejectAuditDecision(AUDIT_SCHEMA, status, selected, probability,
                               mechanism_stratum, allocation_stratum, budget_id,
                               sample_digest)


def _audit_sample_digest(claim: ClaimKey, route: Route,
                         mechanism_stratum: str, allocation_stratum: str) -> str:
    return schemas.content_hash({
        "claim_key": claim.to_dict(), "route_id": route.route_id,
        "route_revision": route.revision, "mechanism_stratum": mechanism_stratum,
        "allocation_stratum": allocation_stratum})


@dataclass(frozen=True)
class CoexistenceReceipt:
    schema: str
    receipt_id: str
    victim_scope: Mapping[str, Any]
    neighbor_mode: str
    neighbors: tuple[Mapping[str, Any], ...]
    pressure_envelope: Mapping[str, Any] | None
    physical_claims: tuple[str, ...]
    lifecycle_phases: tuple[str, ...]
    dependency_identities: Mapping[str, str]
    estimands: tuple[str, ...]
    equivalence_margin: Mapping[str, Any] | None
    uncertainty: Mapping[str, Any] | None
    authority_reference: str | None

    FIELDS = {"schema", "receipt_id", "victim_scope", "neighbor_mode", "neighbors",
              "pressure_envelope", "physical_claims", "lifecycle_phases",
              "dependency_identities", "estimands", "equivalence_margin",
              "uncertainty", "authority_reference"}

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "CoexistenceReceipt":
        _exact(obj, cls.FIELDS, "CoexistenceReceipt")
        if obj["schema"] != COEXISTENCE_SCHEMA:
            raise EvidenceValidationError(
                f"CoexistenceReceipt.schema: unsupported {obj['schema']!r}")
        mode = _enum(obj["neighbor_mode"], frozenset({"multiset", "pressure_envelope"}),
                     "CoexistenceReceipt.neighbor_mode")
        neighbors_obj = obj["neighbors"]
        if not isinstance(neighbors_obj, (list, tuple)):
            raise EvidenceValidationError("CoexistenceReceipt.neighbors: expected array")
        neighbors = tuple(_mapping(item, "CoexistenceReceipt.neighbors[]")
                          for item in neighbors_obj)
        envelope = obj["pressure_envelope"]
        if mode == "multiset":
            if not neighbors or envelope is not None:
                raise EvidenceValidationError(
                    "CoexistenceReceipt.multiset requires neighbors and null envelope")
            normalized_envelope = None
        else:
            if neighbors or not isinstance(envelope, Mapping) or not envelope:
                raise EvidenceValidationError(
                    "CoexistenceReceipt.pressure_envelope requires envelope and no neighbors")
            normalized_envelope = _mapping(envelope, "CoexistenceReceipt.pressure_envelope")
        phases = _strings(obj["lifecycle_phases"], "CoexistenceReceipt.lifecycle_phases")
        if set(phases) != _LIFECYCLE_PHASES:
            raise EvidenceValidationError(
                "CoexistenceReceipt.lifecycle_phases must cover setup, placement, load, "
                "warmup, steady, bursts, teardown exactly")
        margin = obj["equivalence_margin"]
        if margin is not None:
            margin = _mapping(margin, "CoexistenceReceipt.equivalence_margin")
        uncertainty = obj["uncertainty"]
        if uncertainty is not None:
            uncertainty = _mapping(uncertainty, "CoexistenceReceipt.uncertainty")
        return cls(
            COEXISTENCE_SCHEMA,
            cast(str, _text(obj["receipt_id"], "CoexistenceReceipt.receipt_id")),
            _mapping(obj["victim_scope"], "CoexistenceReceipt.victim_scope"), mode,
            neighbors, normalized_envelope,
            _strings(obj["physical_claims"], "CoexistenceReceipt.physical_claims"), phases,
            _dependency_identities(obj["dependency_identities"],
                                   "CoexistenceReceipt.dependency_identities"),
            _strings(obj["estimands"], "CoexistenceReceipt.estimands"), margin,
            uncertainty, _text(obj["authority_reference"],
                               "CoexistenceReceipt.authority_reference", nullable=True))

    def to_dict(self) -> dict[str, Any]:
        return {name: _thaw(getattr(self, name)) for name in self.FIELDS}


@dataclass(frozen=True)
class CoexistenceDisposition:
    status: str
    reasons: tuple[str, ...]


def coexistence_disposition(
        receipt: CoexistenceReceipt, victim_scope: Mapping[str, Any], *,
        neighbors: Sequence[Mapping[str, Any]], lifecycle_phases: Sequence[str],
        equivalence_verifier: Callable[[CoexistenceReceipt], bool | str] | None,
        pressure_verifier: Callable[[CoexistenceReceipt, Sequence[Mapping[str, Any]]], bool | str] | None = None,
        ) -> CoexistenceDisposition:
    receipt = _normalize(CoexistenceReceipt, receipt, "coexistence.receipt")
    _callback(equivalence_verifier, "coexistence.equivalence_verifier")
    _callback(pressure_verifier, "coexistence.pressure_verifier")
    victim = _mapping(victim_scope, "coexistence.victim_scope")
    if _thaw(victim) != _thaw(receipt.victim_scope):
        return CoexistenceDisposition("serialized_owned", ("victim direction differs",))
    phases = _strings(lifecycle_phases, "coexistence.lifecycle_phases")
    if set(phases) != set(receipt.lifecycle_phases):
        return CoexistenceDisposition("serialized_owned", ("lifecycle phase coverage differs",))
    normalized_neighbors = tuple(_mapping(item, "coexistence.neighbors[]") for item in neighbors)
    if receipt.neighbor_mode == "multiset":
        wanted = Counter(schemas.content_hash(_thaw(item)) for item in receipt.neighbors)
        actual = Counter(schemas.content_hash(_thaw(item)) for item in normalized_neighbors)
        if wanted != actual:
            return CoexistenceDisposition(
                "serialized_owned", ("complete neighbor multiset differs",))
    else:
        if pressure_verifier is None:
            return CoexistenceDisposition(
                "serialized_owned", ("registered pressure-envelope verifier is absent",))
        pressure = _verify(pressure_verifier, receipt, normalized_neighbors)
        if pressure is not True:
            return CoexistenceDisposition(
                "serialized_owned",
                (str(pressure) if isinstance(pressure, str)
                 else "pressure envelope does not cover neighbors",))
    if (receipt.equivalence_margin is None or receipt.uncertainty is None
            or receipt.authority_reference is None):
        return CoexistenceDisposition(
            "serialized_owned", ("registered equivalence margin/uncertainty/authority absent",))
    if equivalence_verifier is None:
        return CoexistenceDisposition(
            "serialized_owned", ("trusted equivalence verifier is not connected",))
    verified = _verify(equivalence_verifier, receipt)
    if verified is not True:
        return CoexistenceDisposition(
            "serialized_owned",
            (str(verified) if isinstance(verified, str)
             else "registered equivalence verifier refused",))
    return CoexistenceDisposition("coexistence_supported", ())


__all__ = [
    "CLAIM_KEY_SCHEMA", "SOURCE_REF_SCHEMA", "FINDING_SCHEMA",
    "INVALIDATION_SCHEMA", "INDEX_SCHEMA", "RETRIEVAL_SCHEMA", "PROPOSAL_SCHEMA",
    "LOCAL_FENCE_SCHEMA", "QUARANTINE_SCHEMA",
    "TRANSFER_SCHEMA", "ROUTE_SCHEMA", "AUDIT_SCHEMA", "REVOCATION_SCHEMA",
    "COEXISTENCE_SCHEMA", "EvidenceValidationError", "ClaimKey", "SourceRef",
    "Finding", "InvalidationEvent", "Quarantine", "RetrievedFinding",
    "RetrievalResult", "LocalFenceSnapshot", "ProposalSnapshot", "AdmissionResult",
    "EvidenceIndex", "PlanningEvidence", "ProjectionCompleteness",
    "TransferReceipt", "TransferDisposition", "transfer_disposition", "Route",
    "RouteRevocation", "RouteDisposition", "route_disposition",
    "RejectAuditDecision", "select_reject_audit", "CoexistenceReceipt",
    "CoexistenceDisposition", "coexistence_disposition",
]
