"""Immutable retained-artifact closure and dry expiry planning.

This module decides only which declared artifacts are still referenced.  It neither
discovers filesystem dependencies nor deletes bytes; reclamation authority remains
in :mod:`autokernel.storage` and its tombstone-first policy.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .. import storage

NODE_SCHEMA = "epyc.autokernel.retention_node.v1"
ROOT_SCHEMA = "epyc.autokernel.retention_root.v1"
EXPIRY_SCHEMA = "epyc.autokernel.retention_expiry_descriptor.v1"
SNAPSHOT_SCHEMA = "epyc.autokernel.retention_snapshot.v1"
PLAN_SCHEMA = "epyc.autokernel.retention_plan.v1"

ARTIFACT_KINDS = frozenset({
    "source_ref", "build_dir", "shared_dso_dir", "runtime_recipe",
    "evidence_record", "candidate_artifact", "calibration_artifact",
})
ROOT_KINDS = frozenset({
    "production", "rollback", "integration_candidate", "validated_candidate",
    "pending_validation", "pending_loo", "pending_comparator", "pending_calibration",
    "active_worker", "launch_intent", "integration_intent", "orphan_ref",
    "retained_evidence",
})


class RetentionRefused(ValueError):
    """Retention input cannot prove a safe reclamation boundary."""


def _mapping(value: Any, fields: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RetentionRefused(f"{label} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise RetentionRefused(f"{label} keys must be strings")
    actual = set(value)
    if actual != fields:
        raise RetentionRefused(
            f"{label} fields differ: missing={sorted(fields - actual)}, "
            f"unknown={sorted(actual - fields)}")
    return value


def _text(value: Any, label: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RetentionRefused(f"{label} must be non-empty text")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RetentionRefused(f"{label} must be an integer >= {minimum}")
    return value


def _texts(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise RetentionRefused(f"{label} must be an array")
    result = tuple(_text(item, f"{label}[]") for item in value)
    if len(result) != len(set(result)):
        raise RetentionRefused(f"{label} contains duplicates")
    return tuple(str(item) for item in result)


def _sha(value: Any, label: str) -> str:
    value = _text(value, label)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise RetentionRefused(f"{label} must be a lowercase SHA-256")
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _freeze_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RetentionRefused(f"{label} must be an object")
    result: dict[str, Any] = {}
    for key, item in value.items():
        key = _text(key, f"{label} key")
        if isinstance(item, Mapping):
            item = _freeze_mapping(item, f"{label}.{key}")
        elif isinstance(item, (list, tuple)):
            item = tuple(_freeze_value(part, f"{label}.{key}[]") for part in item)
        else:
            item = _freeze_value(item, f"{label}.{key}")
        result[str(key)] = item
    return MappingProxyType(result)


def _freeze_value(value: Any, label: str) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, Mapping):
        return _freeze_mapping(value, label)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item, f"{label}[]") for item in value)
    raise RetentionRefused(f"{label} contains an unsupported/nonfinite value")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ExpiryDescriptor:
    campaign_id: str
    sha256: str
    durability_class: str
    expirable_kind: str
    reason: str
    rule_id: str
    actor: str
    preconditions: Mapping[str, Any]
    schema: str = EXPIRY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != EXPIRY_SCHEMA:
            raise RetentionRefused("expiry descriptor schema is unsupported")
        for name in ("campaign_id", "expirable_kind", "reason", "rule_id", "actor"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "sha256", _sha(self.sha256, "expiry sha256"))
        durability = _text(self.durability_class, "durability_class")
        if durability not in storage.schemas.DURABILITY_CLASSES:
            raise RetentionRefused("durability_class is unsupported")
        object.__setattr__(self, "durability_class", durability)
        if self.expirable_kind not in storage.EXPIRY_RULES:
            raise RetentionRefused("expirable_kind is unsupported by storage policy")
        object.__setattr__(self, "preconditions",
                           _freeze_mapping(self.preconditions, "preconditions"))

    @classmethod
    def from_dict(cls, value: Any) -> "ExpiryDescriptor":
        fields = frozenset({"schema", "campaign_id", "sha256", "durability_class",
                            "expirable_kind", "reason", "rule_id", "actor",
                            "preconditions"})
        return cls(**dict(_mapping(value, fields, "expiry descriptor")))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "campaign_id": self.campaign_id,
                "sha256": self.sha256, "durability_class": self.durability_class,
                "expirable_kind": self.expirable_kind, "reason": self.reason,
                "rule_id": self.rule_id, "actor": self.actor,
                "preconditions": _thaw(self.preconditions)}


@dataclass(frozen=True, slots=True)
class ArtifactNode:
    artifact_id: str
    artifact_kind: str
    dependencies: tuple[str, ...]
    path: str | None
    retention_class: str
    declared_size_bytes: int | None
    expiry: ExpiryDescriptor | None
    schema: str = NODE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != NODE_SCHEMA:
            raise RetentionRefused("retention node schema is unsupported")
        object.__setattr__(self, "artifact_id", _text(self.artifact_id, "artifact_id"))
        if self.artifact_kind not in ARTIFACT_KINDS:
            raise RetentionRefused("artifact_kind is unsupported")
        object.__setattr__(self, "dependencies", _texts(self.dependencies, "dependencies"))
        object.__setattr__(self, "path", _text(self.path, "path", nullable=True))
        if self.path is not None:
            lexical = Path(self.path)
            if not lexical.is_absolute():
                raise RetentionRefused("artifact path must be absolute")
            if ".." in lexical.parts or str(lexical) != self.path:
                raise RetentionRefused(
                    "artifact path must be lexically canonical without dot segments")
        if self.retention_class not in storage.RETENTION_CLASSES:
            raise RetentionRefused("retention_class is unsupported")
        if self.declared_size_bytes is not None:
            object.__setattr__(self, "declared_size_bytes", _integer(
                self.declared_size_bytes, "declared_size_bytes"))
        if self.expiry is not None:
            object.__setattr__(self, "expiry", ExpiryDescriptor.from_dict(
                self.expiry.to_dict() if isinstance(self.expiry, ExpiryDescriptor) else self.expiry))
        if self.retention_class == "expirable":
            if self.path is None or self.expiry is None:
                raise RetentionRefused("expirable node requires path and expiry descriptor")
            if self.artifact_kind == "source_ref":
                raise RetentionRefused("source refs cannot be classified expirable")
        elif self.expiry is not None:
            raise RetentionRefused("non-expirable node cannot carry an expiry descriptor")

    @classmethod
    def from_dict(cls, value: Any) -> "ArtifactNode":
        fields = frozenset({"schema", "artifact_id", "artifact_kind", "dependencies",
                            "path", "retention_class", "declared_size_bytes", "expiry"})
        row = dict(_mapping(value, fields, "retention node"))
        if row["expiry"] is not None:
            row["expiry"] = ExpiryDescriptor.from_dict(row["expiry"])
        return cls(**row)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "artifact_id": self.artifact_id,
                "artifact_kind": self.artifact_kind,
                "dependencies": list(self.dependencies), "path": self.path,
                "retention_class": self.retention_class,
                "declared_size_bytes": self.declared_size_bytes,
                "expiry": None if self.expiry is None else self.expiry.to_dict()}


@dataclass(frozen=True, slots=True)
class RetentionRoot:
    root_id: str
    root_kind: str
    artifact_ids: tuple[str, ...]
    schema: str = ROOT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ROOT_SCHEMA:
            raise RetentionRefused("retention root schema is unsupported")
        object.__setattr__(self, "root_id", _text(self.root_id, "root_id"))
        if self.root_kind not in ROOT_KINDS:
            raise RetentionRefused("root_kind is unsupported")
        artifacts = _texts(self.artifact_ids, "root artifact_ids")
        if not artifacts:
            raise RetentionRefused("retention root must name at least one artifact")
        object.__setattr__(self, "artifact_ids", artifacts)

    @classmethod
    def from_dict(cls, value: Any) -> "RetentionRoot":
        fields = frozenset({"schema", "root_id", "root_kind", "artifact_ids"})
        return cls(**dict(_mapping(value, fields, "retention root")))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "root_id": self.root_id,
                "root_kind": self.root_kind, "artifact_ids": list(self.artifact_ids)}


@dataclass(frozen=True, slots=True)
class RetentionSnapshot:
    snapshot_id: str
    generation: int
    nodes: tuple[ArtifactNode, ...]
    roots: tuple[RetentionRoot, ...]
    uncertain_scopes: tuple[str, ...]
    snapshot_digest: str
    schema: str = SNAPSHOT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SNAPSHOT_SCHEMA:
            raise RetentionRefused("retention snapshot schema is unsupported")
        object.__setattr__(self, "snapshot_id", _text(self.snapshot_id, "snapshot_id"))
        object.__setattr__(self, "generation", _integer(
            self.generation, "generation", minimum=1))
        if not isinstance(self.nodes, (list, tuple)) or not isinstance(self.roots, (list, tuple)):
            raise RetentionRefused("nodes and roots must be arrays")
        nodes = tuple(ArtifactNode.from_dict(item.to_dict() if isinstance(item, ArtifactNode)
                                             else item) for item in self.nodes)
        roots = tuple(RetentionRoot.from_dict(item.to_dict() if isinstance(item, RetentionRoot)
                                              else item) for item in self.roots)
        if not roots:
            raise RetentionRefused("retention snapshot must declare at least one root")
        if len({node.artifact_id for node in nodes}) != len(nodes):
            raise RetentionRefused("duplicate artifact IDs")
        if len({root.root_id for root in roots}) != len(roots):
            raise RetentionRefused("duplicate root IDs")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "roots", roots)
        object.__setattr__(self, "uncertain_scopes", _texts(
            self.uncertain_scopes, "uncertain_scopes"))
        object.__setattr__(self, "snapshot_digest", _sha(
            self.snapshot_digest, "snapshot_digest"))
        if self.snapshot_digest != digest(self._body()):
            raise RetentionRefused("snapshot digest does not match content")

    def _body(self) -> dict[str, Any]:
        return {"schema": self.schema, "snapshot_id": self.snapshot_id,
                "generation": self.generation,
                "nodes": [node.to_dict() for node in self.nodes],
                "roots": [root.to_dict() for root in self.roots],
                "uncertain_scopes": list(self.uncertain_scopes)}

    @classmethod
    def create(cls, *, snapshot_id: str, generation: int,
               nodes: Sequence[ArtifactNode | Mapping[str, Any]],
               roots: Sequence[RetentionRoot | Mapping[str, Any]],
               uncertain_scopes: Sequence[str] = ()) -> "RetentionSnapshot":
        normalized_nodes = tuple(ArtifactNode.from_dict(
            item.to_dict() if isinstance(item, ArtifactNode) else item) for item in nodes)
        normalized_roots = tuple(RetentionRoot.from_dict(
            item.to_dict() if isinstance(item, RetentionRoot) else item) for item in roots)
        body = {"schema": SNAPSHOT_SCHEMA, "snapshot_id": snapshot_id,
                "generation": generation,
                "nodes": [node.to_dict() for node in normalized_nodes],
                "roots": [root.to_dict() for root in normalized_roots],
                "uncertain_scopes": list(uncertain_scopes)}
        return cls(snapshot_id=snapshot_id, generation=generation, nodes=normalized_nodes,
                   roots=normalized_roots, uncertain_scopes=tuple(uncertain_scopes),
                   snapshot_digest=digest(body))

    @classmethod
    def from_dict(cls, value: Any) -> "RetentionSnapshot":
        fields = frozenset({"schema", "snapshot_id", "generation", "nodes", "roots",
                            "uncertain_scopes", "snapshot_digest"})
        return cls(**dict(_mapping(value, fields, "retention snapshot")))

    def to_dict(self) -> dict[str, Any]:
        return self._body() | {"snapshot_digest": self.snapshot_digest}


def _reason_map(value: Any) -> Mapping[str, tuple[str, ...]]:
    if not isinstance(value, Mapping):
        raise RetentionRefused("retained_reasons must be an object")
    result = {str(_text(key, "retained reason key")): _texts(items, "retained reasons")
              for key, items in value.items()}
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class RetentionPlan:
    snapshot_id: str
    snapshot_generation: int
    snapshot_digest: str
    retained_ids: tuple[str, ...]
    retained_reasons: Mapping[str, tuple[str, ...]]
    retained_paths: tuple[str, ...]
    unknown_closure: tuple[str, ...]
    expirable_ids: tuple[str, ...]
    expirable_bytes: Mapping[str, int]
    expirable_total_bytes: int | None
    complete: bool
    plan_digest: str
    schema: str = PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PLAN_SCHEMA:
            raise RetentionRefused("retention plan schema is unsupported")
        object.__setattr__(self, "snapshot_id", _text(self.snapshot_id, "snapshot_id"))
        object.__setattr__(self, "snapshot_generation", _integer(
            self.snapshot_generation, "snapshot_generation", minimum=1))
        object.__setattr__(self, "snapshot_digest", _sha(
            self.snapshot_digest, "snapshot_digest"))
        for name in ("retained_ids", "retained_paths", "unknown_closure", "expirable_ids"):
            object.__setattr__(self, name, _texts(getattr(self, name), name))
        object.__setattr__(self, "retained_reasons", _reason_map(self.retained_reasons))
        if set(self.retained_reasons) != set(self.retained_ids):
            raise RetentionRefused("retained reasons must exactly cover retained IDs")
        if not isinstance(self.expirable_bytes, Mapping):
            raise RetentionRefused("expirable_bytes must be an object")
        byte_map = {str(_text(key, "expirable byte key")):
                    _integer(value, "expirable bytes")
                    for key, value in self.expirable_bytes.items()}
        if not set(byte_map).issubset(self.expirable_ids):
            raise RetentionRefused("expirable byte keys must be expirable IDs")
        object.__setattr__(self, "expirable_bytes", MappingProxyType(byte_map))
        if self.expirable_total_bytes is not None:
            object.__setattr__(self, "expirable_total_bytes", _integer(
                self.expirable_total_bytes, "expirable_total_bytes"))
            if len(byte_map) != len(self.expirable_ids) \
                    or self.expirable_total_bytes != sum(byte_map.values()):
                raise RetentionRefused("expirable total does not match known candidate bytes")
        if not isinstance(self.complete, bool) or self.complete == bool(self.unknown_closure):
            raise RetentionRefused("complete must be exactly the inverse of unknown closure")
        object.__setattr__(self, "plan_digest", _sha(self.plan_digest, "plan_digest"))
        if self.plan_digest != digest(self._body()):
            raise RetentionRefused("plan digest does not match content")

    def _body(self) -> dict[str, Any]:
        return {"schema": self.schema, "snapshot_id": self.snapshot_id,
                "snapshot_generation": self.snapshot_generation,
                "snapshot_digest": self.snapshot_digest,
                "retained_ids": list(self.retained_ids),
                "retained_reasons": {key: list(value)
                                     for key, value in self.retained_reasons.items()},
                "retained_paths": list(self.retained_paths),
                "unknown_closure": list(self.unknown_closure),
                "expirable_ids": list(self.expirable_ids),
                "expirable_bytes": dict(self.expirable_bytes),
                "expirable_total_bytes": self.expirable_total_bytes,
                "complete": self.complete}

    @classmethod
    def from_dict(cls, value: Any) -> "RetentionPlan":
        fields = frozenset({"schema", "snapshot_id", "snapshot_generation",
                            "snapshot_digest", "retained_ids", "retained_reasons",
                            "retained_paths", "unknown_closure", "expirable_ids",
                            "expirable_bytes", "expirable_total_bytes", "complete",
                            "plan_digest"})
        return cls(**dict(_mapping(value, fields, "retention plan")))

    def to_dict(self) -> dict[str, Any]:
        return self._body() | {"plan_digest": self.plan_digest}


def _make_plan(snapshot: RetentionSnapshot, *, retained: set[str],
               reasons: Mapping[str, set[str]], unknown: set[str],
               expirable: Sequence[ArtifactNode]) -> RetentionPlan:
    retained_ids = tuple(sorted(retained))
    retained_reasons = {item: tuple(sorted(reasons[item])) for item in retained_ids}
    retained_paths = tuple(sorted({node.path for node in snapshot.nodes
                                   if node.artifact_id in retained and node.path is not None}))
    candidates = () if unknown else tuple(sorted(node.artifact_id for node in expirable))
    by_id = {node.artifact_id: node for node in expirable}
    byte_map = {item: by_id[item].declared_size_bytes for item in candidates
                if by_id[item].declared_size_bytes is not None}
    total = sum(byte_map.values()) if len(byte_map) == len(candidates) else None
    body = {"schema": PLAN_SCHEMA, "snapshot_id": snapshot.snapshot_id,
            "snapshot_generation": snapshot.generation,
            "snapshot_digest": snapshot.snapshot_digest,
            "retained_ids": list(retained_ids),
            "retained_reasons": {key: list(value) for key, value in retained_reasons.items()},
            "retained_paths": list(retained_paths),
            "unknown_closure": sorted(unknown), "expirable_ids": list(candidates),
            "expirable_bytes": byte_map, "expirable_total_bytes": total,
            "complete": not unknown}
    return RetentionPlan(**{key: value for key, value in body.items() if key != "schema"},
                         plan_digest=digest(body))


def plan_retention(value: RetentionSnapshot | Mapping[str, Any]) -> RetentionPlan:
    """Traverse the declared graph; uncertainty makes the whole supplied store non-reclaimable."""
    snapshot = RetentionSnapshot.from_dict(
        value.to_dict() if isinstance(value, RetentionSnapshot) else value)
    nodes = {node.artifact_id: node for node in snapshot.nodes}
    retained: set[str] = set()
    reasons: dict[str, set[str]] = {}
    unknown = {f"uncertain scope: {scope}" for scope in snapshot.uncertain_scopes}
    for node in snapshot.nodes:
        for dependency in node.dependencies:
            if dependency not in nodes:
                unknown.add(
                    f"artifact {node.artifact_id} references missing artifact {dependency}")
    work_roots = [
        (f"{root.root_kind}:{root.root_id}", root.artifact_ids)
        for root in snapshot.roots]
    work_roots.extend(
        (f"retention_class:{node.retention_class}:{node.artifact_id}",
         (node.artifact_id,))
        for node in snapshot.nodes if node.retention_class != "expirable")
    def retain_closure(artifact_ids: Sequence[str], root_reason: str) -> None:
        stack = list(reversed(artifact_ids))
        seen: set[str] = set()
        while stack:
            artifact_id = stack.pop()
            if artifact_id in seen:
                continue
            seen.add(artifact_id)
            node = nodes.get(artifact_id)
            if node is None:
                unknown.add(f"{root_reason} references missing artifact {artifact_id}")
                continue
            retained.add(artifact_id)
            reasons.setdefault(artifact_id, set()).add(root_reason)
            stack.extend(reversed(node.dependencies))

    for root_reason, artifact_ids in sorted(work_roots):
        retain_closure(artifact_ids, root_reason)

    # Artifact IDs describe graph objects, not disjoint bytes. If any retained path
    # equals, contains, or is contained by another declared path, both names share a
    # physical deletion boundary. Retain the overlapping node and its dependency
    # closure, repeating because those dependencies may expose another overlap. This
    # is deliberately lexical: live symlink identity belongs in the native snapshot's
    # uncertainty, not in this pure planner.
    while True:
        retained_nodes = tuple(node for node in snapshot.nodes
                               if node.artifact_id in retained and node.path is not None)
        additions: list[tuple[ArtifactNode, ArtifactNode]] = []
        for node in snapshot.nodes:
            if node.artifact_id in retained or node.path is None:
                continue
            candidate = Path(node.path)
            for retained_node in retained_nodes:
                retained_path = Path(retained_node.path)
                if (candidate == retained_path or candidate in retained_path.parents
                        or retained_path in candidate.parents):
                    additions.append((node, retained_node))
                    break
        if not additions:
            break
        for node, retained_node in additions:
            retain_closure((node.artifact_id,),
                           f"physical_path_overlap:{retained_node.artifact_id}")
    expirable = tuple(node for node in snapshot.nodes
                      if node.artifact_id not in retained
                      and node.retention_class == "expirable")
    return _make_plan(snapshot, retained=retained, reasons=reasons,
                      unknown=unknown, expirable=expirable)


def validate_plan_snapshot(plan: RetentionPlan | Mapping[str, Any],
                           snapshot: RetentionSnapshot | Mapping[str, Any]
                           ) -> tuple[RetentionPlan, RetentionSnapshot]:
    plan = RetentionPlan.from_dict(plan.to_dict() if isinstance(plan, RetentionPlan) else plan)
    snapshot = RetentionSnapshot.from_dict(
        snapshot.to_dict() if isinstance(snapshot, RetentionSnapshot) else snapshot)
    if (plan.snapshot_id != snapshot.snapshot_id
            or plan.snapshot_generation != snapshot.generation
            or plan.snapshot_digest != snapshot.snapshot_digest):
        raise RetentionRefused("retention plan no longer matches snapshot generation/content")
    expected = plan_retention(snapshot)
    if plan.to_dict() != expected.to_dict():
        raise RetentionRefused(
            "retention plan does not match the deterministic closure of its snapshot")
    if not plan.complete:
        raise RetentionRefused("retention closure is unknown; unified reclamation is withheld")
    return plan, snapshot


def plan_expiry_candidates(plan: RetentionPlan | Mapping[str, Any],
                           snapshot: RetentionSnapshot | Mapping[str, Any],
                           policy: storage.StoragePolicy, *,
                           now: datetime | None = None) -> tuple[storage.ExpiryOutcome, ...]:
    """Apply the existing storage policy to complete, unretained candidates; never delete."""
    plan, snapshot = validate_plan_snapshot(plan, snapshot)
    nodes = {node.artifact_id: node for node in snapshot.nodes}
    outcomes = []
    for artifact_id in plan.expirable_ids:
        node = nodes[artifact_id]
        if node.retention_class != "expirable" or node.expiry is None or node.path is None:
            raise RetentionRefused(f"candidate {artifact_id} is not explicitly expirable")
        descriptor = node.expiry
        artifact = storage.ExpirableArtifact(
            path=node.path, campaign_id=descriptor.campaign_id, sha256=descriptor.sha256,
            durability_class=descriptor.durability_class,
            expirable_kind=descriptor.expirable_kind, reason=descriptor.reason,
            rule_id=descriptor.rule_id, actor=descriptor.actor,
            retention_class=node.retention_class,
            preconditions=_thaw(descriptor.preconditions),
            declared_size_bytes=node.declared_size_bytes)
        outcomes.append(storage.plan_expiry(artifact, policy, now=now))
    return tuple(outcomes)


__all__ = [
    "ARTIFACT_KINDS", "ArtifactNode", "EXPIRY_SCHEMA", "ExpiryDescriptor",
    "NODE_SCHEMA", "PLAN_SCHEMA", "ROOT_KINDS", "ROOT_SCHEMA", "RetentionPlan",
    "RetentionRefused", "RetentionRoot", "RetentionSnapshot", "SNAPSHOT_SCHEMA",
    "digest", "plan_expiry_candidates", "plan_retention", "validate_plan_snapshot",
]
