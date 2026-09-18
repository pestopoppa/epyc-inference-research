"""Pure immutable candidate, LOO, validation-batch, and pointer contracts.

This module is not a WAL, evidence grader, launcher, or production promotion path.
Persisting a ``passed`` row is never sufficient to advance a validated pointer: that
transition requires an injected trusted verifier to revalidate every required receipt.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .resolved_recipe import ArtifactDigest


CANDIDATE_SCHEMA = "epyc.autokernel.candidate_manifest.v1"
SOURCE_SCHEMA = "epyc.autokernel.source_identity.v1"
BUILD_SCHEMA = "epyc.autokernel.candidate_build.v1"
TARGET_SCHEMA = "epyc.autokernel.candidate_target.v1"
KEEP_SCHEMA = "epyc.autokernel.keep_treatment.v1"
CHANGE_SCHEMA = "epyc.autokernel.field_change.v1"
LOO_PLAN_SCHEMA = "epyc.autokernel.loo_plan.v1"
LOO_RESULT_SCHEMA = "epyc.autokernel.loo_result.v1"
ROW_SCHEMA = "epyc.autokernel.validation_row.v1"
ROW_SET_SCHEMA = "epyc.autokernel.required_row_set.v1"
RECEIPT_SCHEMA = "epyc.autokernel.validation_row_receipt.v1"
ROW_STATE_SCHEMA = "epyc.autokernel.validation_row_state.v1"
BATCH_SCHEMA = "epyc.autokernel.validation_batch.v1"
STATE_SCHEMA = "epyc.autokernel.candidate_state.v1"
EQUIVALENCE_SCHEMA = "epyc.autokernel.equivalence_receipt.v1"

KEEP_KINDS = frozenset({"source", "build", "runtime"})
ROW_STATUSES = frozenset({"pending", "prerequisite_missing", "running", "passed", "failed",
                          "inconclusive", "unsupported", "nonidentifiable"})
TERMINAL_ROW_STATUSES = ROW_STATUSES - {"pending", "running"}
EQUIVALENCE_USES = frozenset({"exact", "correctness", "local_work", "timing"})


class CandidateError(ValueError):
    """Malformed or internally contradictory immutable candidate data."""


class TransitionError(RuntimeError):
    """A state transition violates identity, ordering, CAS, or trust requirements."""


class TrustedVerificationRequired(TransitionError):
    """Validated-pointer advancement lacks its trusted evidence adapter."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateError(f"{label} must be an object")
    return dict(value)


def _keys(value: Mapping[str, Any], required: set[str], label: str) -> None:
    missing, extra = required - set(value), set(value) - required
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing {sorted(missing)}")
        if extra:
            parts.append(f"unknown {sorted(extra)}")
        raise CandidateError(f"{label}: " + "; ".join(parts))


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CandidateError(f"{label} must be a non-empty string")
    return value


def _sha(value: Any, label: str) -> str:
    digest = _text(value, label)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise CandidateError(f"{label} must be lowercase SHA-256")
    return digest


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise CandidateError(f"{label} must be an array")
    return value


def _strings(value: Any, label: str, *, nonempty: bool = False) -> tuple[str, ...]:
    result = tuple(_text(item, f"{label}[]") for item in _sequence(value, label))
    if nonempty and not result:
        raise CandidateError(f"{label} must not be empty")
    if len(set(result)) != len(result):
        raise CandidateError(f"{label} contains duplicates")
    return result


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)):
        raise CandidateError(f"{label} must be a finite number")
    return float(value)


@dataclass(frozen=True)
class SourceIdentity:
    repo_id: str
    path: str
    object_format: str
    commit: str
    tree: str

    @classmethod
    def from_dict(cls, value: Any) -> "SourceIdentity":
        row = _object(value, "source identity")
        _keys(row, {"schema", "repo_id", "path", "object_format", "commit", "tree"},
              "source identity")
        if row["schema"] != SOURCE_SCHEMA:
            raise CandidateError(f"source identity: unsupported schema {row['schema']!r}")
        path = _text(row["path"], "source identity.path")
        if not Path(path).is_absolute():
            raise CandidateError("source identity.path must be absolute")
        object_format = _text(row["object_format"], "source identity.object_format")
        lengths = {"sha1": 40, "sha256": 64}
        if object_format not in lengths:
            raise CandidateError("source identity.object_format must be sha1 or sha256")
        def oid(value: Any, label: str) -> str:
            result = _text(value, label)
            if len(result) != lengths[object_format] or any(
                    char not in "0123456789abcdef" for char in result):
                raise CandidateError(f"{label} does not match {object_format}")
            return result
        return cls(_text(row["repo_id"], "source identity.repo_id"), path, object_format,
                   oid(row["commit"], "source identity.commit"),
                   oid(row["tree"], "source identity.tree"))

    def to_dict(self) -> dict[str, str]:
        return {"schema": SOURCE_SCHEMA, "repo_id": self.repo_id, "path": self.path,
                "object_format": self.object_format, "commit": self.commit, "tree": self.tree}


def source_set_digest(sources: Sequence[SourceIdentity]) -> str:
    return _digest(sorted(({"repo_id": item.repo_id, "object_format": item.object_format,
                            "commit": item.commit, "tree": item.tree}
                           for item in sources), key=lambda row: row["repo_id"]))


@dataclass(frozen=True)
class BuildIdentity:
    build_id: str
    build_recipe_digest: str
    source_set_digest: str
    executable: ArtifactDigest
    dsos: tuple[ArtifactDigest, ...]

    @classmethod
    def from_dict(cls, value: Any) -> "BuildIdentity":
        row = _object(value, "candidate build")
        _keys(row, {"schema", "build_id", "build_recipe_digest", "source_set_digest", "executable", "dsos"},
              "candidate build")
        if row["schema"] != BUILD_SCHEMA:
            raise CandidateError(f"candidate build: unsupported schema {row['schema']!r}")
        dsos = tuple(ArtifactDigest.from_dict(item, role="dso")
                     for item in _sequence(row["dsos"], "candidate build.dsos"))
        load_names = [Path(item.path).name for item in dsos]
        if not dsos or len(set(load_names)) != len(dsos):
            raise CandidateError("candidate build needs unique DSO loader names")
        return cls(_text(row["build_id"], "candidate build.build_id"),
                   _sha(row["build_recipe_digest"], "candidate build.build_recipe_digest"),
                   _sha(row["source_set_digest"], "candidate build.source_set_digest"),
                   ArtifactDigest.from_dict(row["executable"], role="executable"),
                   tuple(sorted(dsos, key=lambda item: Path(item.path).name)))

    @property
    def execution_digest(self) -> str:
        return _digest({"recipe": self.build_recipe_digest,
                        "source": self.source_set_digest,
                        "executable": self.executable.sha256,
                        "dsos": [{"load_name": Path(item.path).name, "sha256": item.sha256}
                                 for item in self.dsos]})

    def to_dict(self) -> dict[str, Any]:
        return {"schema": BUILD_SCHEMA, "build_id": self.build_id,
                "build_recipe_digest": self.build_recipe_digest,
                "source_set_digest": self.source_set_digest,
                "executable": self.executable.to_dict(),
                "dsos": [item.to_dict() for item in self.dsos]}


@dataclass(frozen=True)
class CandidateTarget:
    target_id: str
    target_revision_digest: str
    backend: str
    build_execution_digest: str
    resolved_recipe_execution_digest: str
    resolved_recipe_snapshot_digest: str
    model_digest: str
    drafter_digest: str | None
    workload_digest: str
    production_required: bool

    @classmethod
    def from_dict(cls, value: Any) -> "CandidateTarget":
        row = _object(value, "candidate target")
        _keys(row, {"schema", "target_id", "target_revision_digest", "backend",
                    "build_execution_digest",
                    "resolved_recipe_execution_digest", "resolved_recipe_snapshot_digest",
                    "model_digest", "drafter_digest", "workload_digest", "production_required"},
              "candidate target")
        if row["schema"] != TARGET_SCHEMA:
            raise CandidateError(f"candidate target: unsupported schema {row['schema']!r}")
        backend = _text(row["backend"], "candidate target.backend")
        if backend not in {"cpu", "gpu", "both"}:
            raise CandidateError("candidate target.backend must be cpu, gpu, or both")
        if not isinstance(row["production_required"], bool):
            raise CandidateError("candidate target.production_required must be boolean")
        drafter = row["drafter_digest"]
        return cls(_text(row["target_id"], "candidate target.target_id"),
                   _sha(row["target_revision_digest"], "target revision digest"), backend,
                   _sha(row["build_execution_digest"], "target build execution digest"),
                   _sha(row["resolved_recipe_execution_digest"], "recipe execution digest"),
                   _sha(row["resolved_recipe_snapshot_digest"], "recipe snapshot digest"),
                   _sha(row["model_digest"], "model digest"),
                   None if drafter is None else _sha(drafter, "drafter digest"),
                   _sha(row["workload_digest"], "workload digest"), row["production_required"])

    def to_dict(self) -> dict[str, Any]:
        return {"schema": TARGET_SCHEMA,
                "target_id": self.target_id,
                "target_revision_digest": self.target_revision_digest,
                "backend": self.backend,
                "build_execution_digest": self.build_execution_digest,
                "resolved_recipe_execution_digest": self.resolved_recipe_execution_digest,
                "resolved_recipe_snapshot_digest": self.resolved_recipe_snapshot_digest,
                "model_digest": self.model_digest, "drafter_digest": self.drafter_digest,
                "workload_digest": self.workload_digest,
                "production_required": self.production_required}


@dataclass(frozen=True)
class FieldChange:
    field_id: str
    previous_digest: str
    current_digest: str

    @classmethod
    def from_dict(cls, value: Any) -> "FieldChange":
        row = _object(value, "field change")
        _keys(row, {"schema", "field_id", "previous_digest", "current_digest"}, "field change")
        if row["schema"] != CHANGE_SCHEMA:
            raise CandidateError(f"field change: unsupported schema {row['schema']!r}")
        result = cls(_text(row["field_id"], "field change.field_id"),
                     _sha(row["previous_digest"], "field change.previous_digest"),
                     _sha(row["current_digest"], "field change.current_digest"))
        if result.previous_digest == result.current_digest:
            raise CandidateError("field change must change identity")
        return result

    def to_dict(self) -> dict[str, str]:
        return {"schema": CHANGE_SCHEMA, "field_id": self.field_id,
                "previous_digest": self.previous_digest, "current_digest": self.current_digest}


@dataclass(frozen=True)
class KeepTreatment:
    request_id: str
    keep_id: str
    parent_manifest_digest: str
    kind: str
    changes: tuple[FieldChange, ...]
    affected_scopes: tuple[str, ...]
    dependencies: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: Any) -> "KeepTreatment":
        row = _object(value, "keep treatment")
        _keys(row, {"schema", "request_id", "keep_id", "parent_manifest_digest", "kind",
                    "changes", "affected_scopes", "dependencies"}, "keep treatment")
        if row["schema"] != KEEP_SCHEMA:
            raise CandidateError(f"keep treatment: unsupported schema {row['schema']!r}")
        kind = _text(row["kind"], "keep treatment.kind")
        if kind not in KEEP_KINDS:
            raise CandidateError(f"keep treatment.kind must be one of {sorted(KEEP_KINDS)}")
        changes = tuple(FieldChange.from_dict(item)
                        for item in _sequence(row["changes"], "keep treatment.changes"))
        if not changes or len({item.field_id for item in changes}) != len(changes):
            raise CandidateError("keep treatment needs unique changed field identities")
        return cls(_text(row["request_id"], "keep treatment.request_id"),
                   _text(row["keep_id"], "keep treatment.keep_id"),
                   _sha(row["parent_manifest_digest"], "keep treatment.parent_manifest_digest"),
                   kind, tuple(sorted(changes, key=lambda item: item.field_id)),
                   tuple(sorted(_strings(row["affected_scopes"], "affected scopes", nonempty=True))),
                   tuple(sorted(_strings(row["dependencies"], "keep dependencies"))))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": KEEP_SCHEMA, "request_id": self.request_id, "keep_id": self.keep_id,
                "parent_manifest_digest": self.parent_manifest_digest, "kind": self.kind,
                "changes": [item.to_dict() for item in self.changes],
                "affected_scopes": list(self.affected_scopes),
                "dependencies": list(self.dependencies)}


@dataclass(frozen=True)
class CandidateManifest:
    manifest_id: str
    parent_manifest_digest: str
    production_ref_digest: str
    sources: tuple[SourceIdentity, ...]
    builds: tuple[BuildIdentity, ...]
    targets: tuple[CandidateTarget, ...]
    keeps: tuple[KeepTreatment, ...]
    dependency_digest: str

    @classmethod
    def from_dict(cls, value: Any) -> "CandidateManifest":
        row = _object(value, "candidate manifest")
        _keys(row, {"schema", "manifest_id", "parent_manifest_digest", "production_ref_digest",
                    "sources", "builds", "targets", "keeps", "dependency_digest"},
              "candidate manifest")
        if row["schema"] != CANDIDATE_SCHEMA:
            raise CandidateError(f"candidate manifest: unsupported schema {row['schema']!r}")
        sources = tuple(SourceIdentity.from_dict(item)
                        for item in _sequence(row["sources"], "candidate manifest.sources"))
        if not sources or len({item.repo_id for item in sources}) != len(sources):
            raise CandidateError("candidate manifest needs unique source repositories")
        builds = tuple(BuildIdentity.from_dict(item)
                       for item in _sequence(row["builds"], "candidate manifest.builds"))
        if not builds or len({item.build_id for item in builds}) != len(builds) \
                or len({item.execution_digest for item in builds}) != len(builds):
            raise CandidateError("candidate manifest needs unique build ids and identities")
        if any(build.source_set_digest != source_set_digest(sources) for build in builds):
            raise CandidateError("candidate build was not produced from the manifest source set")
        targets = tuple(CandidateTarget.from_dict(item)
                        for item in _sequence(row["targets"], "candidate manifest.targets"))
        if not targets or len({item.target_id for item in targets}) != len(targets) \
                or len({item.target_revision_digest for item in targets}) != len(targets):
            raise CandidateError("candidate manifest needs unique targets")
        build_digests = {item.execution_digest for item in builds}
        if any(target.build_execution_digest not in build_digests for target in targets):
            raise CandidateError("candidate target selects an unknown build identity")
        keeps = tuple(KeepTreatment.from_dict(item)
                      for item in _sequence(row["keeps"], "candidate manifest.keeps"))
        seen: set[str] = set()
        requests: dict[str, str] = {}
        for keep in keeps:
            if keep.keep_id in seen:
                raise CandidateError(f"duplicate keep id {keep.keep_id!r}")
            if not set(keep.dependencies) <= seen:
                raise CandidateError(f"keep {keep.keep_id!r} has missing/forward/cyclic dependencies")
            payload = _digest(keep.to_dict())
            if keep.request_id in requests and requests[keep.request_id] != payload:
                raise CandidateError(f"keep request {keep.request_id!r} conflicts")
            requests[keep.request_id] = payload
            seen.add(keep.keep_id)
        result = cls(_text(row["manifest_id"], "candidate manifest.manifest_id"),
                     _sha(row["parent_manifest_digest"], "parent manifest digest"),
                     _sha(row["production_ref_digest"], "production ref digest"),
                     tuple(sorted(sources, key=lambda item: item.repo_id)),
                     tuple(sorted(builds, key=lambda item: item.build_id)),
                     tuple(sorted(targets, key=lambda item: item.target_id)), keeps,
                     _sha(row["dependency_digest"], "dependency digest"))
        for index, keep in enumerate(result.keeps):
            later_fields = {change.field_id for later in result.keeps[index + 1:]
                            for change in later.changes}
            active_changes = [change for change in keep.changes
                              if change.field_id not in later_fields]
            if not active_changes:
                continue
            current = {item.current_digest for item in active_changes}
            if keep.kind == "source" and source_set_digest(result.sources) not in current:
                raise CandidateError("source keep does not bind the candidate source/build identity")
            if keep.kind == "build" and not current & {
                    build.execution_digest for build in result.builds}:
                raise CandidateError("build keep does not bind the candidate binary identity")
            if keep.kind == "runtime" and not current & {
                    target.resolved_recipe_execution_digest for target in result.targets}:
                raise CandidateError("runtime keep does not bind a target resolved recipe identity")
        return result

    @property
    def manifest_digest(self) -> str:
        return _digest(self.to_dict())

    @property
    def build_set_digest(self) -> str:
        return _digest([item.execution_digest for item in self.builds])

    def to_dict(self) -> dict[str, Any]:
        return {"schema": CANDIDATE_SCHEMA, "manifest_id": self.manifest_id,
                "parent_manifest_digest": self.parent_manifest_digest,
                "production_ref_digest": self.production_ref_digest,
                "sources": [item.to_dict() for item in self.sources],
                "builds": [item.to_dict() for item in self.builds],
                "targets": [item.to_dict() for item in self.targets],
                "keeps": [item.to_dict() for item in self.keeps],
                "dependency_digest": self.dependency_digest}

    def validated(self) -> "CandidateManifest":
        return CandidateManifest.from_dict(self.to_dict())


@dataclass(frozen=True)
class LOOPlan:
    candidate_manifest_digest: str
    keep_id: str
    status: str
    reason: str | None
    binary_execution_digest: str
    revert_changes: tuple[FieldChange, ...]
    derived_manifest_digest: str | None = None

    @classmethod
    def from_dict(cls, value: Any) -> "LOOPlan":
        row = _object(value, "LOO plan")
        _keys(row, {"schema", "candidate_manifest_digest", "keep_id", "status", "reason",
                    "binary_execution_digest", "revert_changes", "derived_manifest_digest"},
              "LOO plan")
        if row["schema"] != LOO_PLAN_SCHEMA:
            raise CandidateError("unsupported LOO plan schema")
        status = _text(row["status"], "LOO plan.status")
        if status not in {"planned", "nonidentifiable", "unsupported"}:
            raise CandidateError("unknown LOO plan status")
        reason = row["reason"]
        if status == "planned" and reason is not None:
            raise CandidateError("planned LOO cannot have refusal reason")
        if status != "planned":
            reason = _text(reason, "LOO plan.reason")
        derived = row["derived_manifest_digest"]
        if derived is not None:
            derived = _sha(derived, "LOO derived manifest")
        if status != "planned" and derived is not None:
            raise CandidateError("unplanned LOO cannot name a derived manifest")
        changes = tuple(FieldChange.from_dict(item)
                        for item in _sequence(row["revert_changes"], "LOO revert changes"))
        return cls(_sha(row["candidate_manifest_digest"], "LOO candidate"),
                   _text(row["keep_id"], "LOO keep id"), status, reason,
                   _sha(row["binary_execution_digest"], "LOO build set"), changes, derived)

    @property
    def plan_digest(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {"schema": LOO_PLAN_SCHEMA,
                "candidate_manifest_digest": self.candidate_manifest_digest,
                "keep_id": self.keep_id, "status": self.status, "reason": self.reason,
                "binary_execution_digest": self.binary_execution_digest,
                "revert_changes": [item.to_dict() for item in self.revert_changes],
                "derived_manifest_digest": self.derived_manifest_digest}


TreatmentResolver = Callable[[CandidateManifest, KeepTreatment], CandidateManifest]


def plan_loo(candidate: CandidateManifest, keep_id: str, *,
             treatment_resolver: TreatmentResolver | None = None) -> LOOPlan:
    candidate = candidate.validated()
    by_id = {keep.keep_id: (index, keep) for index, keep in enumerate(candidate.keeps)}
    if keep_id not in by_id:
        raise CandidateError(f"unknown keep id {keep_id!r}")
    index, selected = by_id[keep_id]
    later = candidate.keeps[index + 1:]
    dependent_ids = {keep_id}
    dependent: list[str] = []
    for keep in later:
        if dependent_ids & set(keep.dependencies):
            dependent.append(keep.keep_id)
            dependent_ids.add(keep.keep_id)
    overwritten = sorted({change.field_id for change in selected.changes}
                         & {change.field_id for keep in later for change in keep.changes})
    if dependent or overwritten:
        reason = (f"dependent keeps {dependent}" if dependent else
                  f"later overwrite of {overwritten}")
        if treatment_resolver is None:
            return LOOPlan(candidate.manifest_digest, keep_id, "nonidentifiable", reason,
                           candidate.build_set_digest, ())
    if selected.kind != "runtime" and treatment_resolver is None:
        return LOOPlan(candidate.manifest_digest, keep_id, "unsupported",
                       "source/build LOO requires a registered treatment resolver",
                       candidate.build_set_digest, ())
    derived_digest = None
    if treatment_resolver is None:
        raw = candidate.to_dict()
        raw["manifest_id"] = f"{candidate.manifest_id}:loo:{keep_id}"
        raw["keeps"] = [item.to_dict() for item in candidate.keeps if item.keep_id != keep_id]
        substitutions = {item.current_digest: item.previous_digest for item in selected.changes}
        for target in raw["targets"]:
            current = target["resolved_recipe_execution_digest"]
            if current in substitutions:
                target["resolved_recipe_execution_digest"] = substitutions[current]
        derived_digest = CandidateManifest.from_dict(raw).manifest_digest
    if treatment_resolver is not None:
        derived = treatment_resolver(candidate, selected).validated()
        if any(keep.keep_id == keep_id for keep in derived.keeps):
            raise CandidateError("treatment resolver did not remove the selected keep")
        if selected.kind == "runtime" and {
                item.execution_digest for item in derived.builds} != {
                item.execution_digest for item in candidate.builds}:
            raise CandidateError("runtime-only LOO changed the candidate binary")
        derived_digest = derived.manifest_digest
    return LOOPlan(candidate.manifest_digest, keep_id, "planned", None,
                   candidate.build_set_digest, selected.changes, derived_digest)


@dataclass(frozen=True)
class LOOResult:
    plan_digest: str
    candidate_manifest_digest: str
    keep_id: str
    derived_manifest_digest: str
    row_set_digest: str
    receipt_digests: tuple[str, ...]
    disposition: str
    evidence_digest: str
    deletion_authorized: bool = False

    @classmethod
    def from_dict(cls, value: Any) -> "LOOResult":
        row = _object(value, "LOO result")
        _keys(row, {"schema", "plan_digest", "candidate_manifest_digest", "keep_id",
                    "derived_manifest_digest", "row_set_digest", "receipt_digests",
                    "disposition", "evidence_digest",
                    "deletion_authorized"}, "LOO result")
        if row["schema"] != LOO_RESULT_SCHEMA:
            raise CandidateError(f"LOO result: unsupported schema {row['schema']!r}")
        disposition = _text(row["disposition"], "LOO result.disposition")
        if disposition not in {"neutral", "supports_keep", "supports_removal", "inconclusive"}:
            raise CandidateError("unknown LOO disposition")
        if row["deletion_authorized"] is not False:
            raise CandidateError("a LOO result cannot grant deletion authority")
        receipts = tuple(sorted(_strings(row["receipt_digests"], "LOO receipt digests",
                                         nonempty=True)))
        return cls(_sha(row["plan_digest"], "LOO plan digest"),
                   _sha(row["candidate_manifest_digest"], "LOO candidate digest"),
                   _text(row["keep_id"], "LOO keep id"),
                   _sha(row["derived_manifest_digest"], "LOO derived manifest"),
                   _sha(row["row_set_digest"], "LOO row set"), receipts, disposition,
                   _sha(row["evidence_digest"], "LOO evidence digest"), False)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": LOO_RESULT_SCHEMA, "plan_digest": self.plan_digest,
                "candidate_manifest_digest": self.candidate_manifest_digest,
                "keep_id": self.keep_id,
                "derived_manifest_digest": self.derived_manifest_digest,
                "row_set_digest": self.row_set_digest,
                "receipt_digests": list(self.receipt_digests),
                "disposition": self.disposition, "evidence_digest": self.evidence_digest,
                "deletion_authorized": False}


@dataclass(frozen=True)
class ValidationRow:
    row_id: str
    row_kind: str
    required: bool
    target_revision_digest: str
    control_target_revision_digest: str
    backend: str
    candidate_build_digest: str
    control_build_digest: str
    model_digest: str
    drafter_digest: str | None
    category: str
    control_recipe_digest: str
    candidate_recipe_digest: str
    instrument_digest: str
    protocol_id: str
    objective_digest: str
    workload_digest: str
    exact_candidate_required: bool

    @classmethod
    def from_dict(cls, value: Any) -> "ValidationRow":
        row = _object(value, "validation row")
        required = {"schema", "row_id", "row_kind", "required", "target_revision_digest",
                    "control_target_revision_digest",
                    "backend", "candidate_build_digest", "control_build_digest", "model_digest",
                    "drafter_digest", "category", "control_recipe_digest", "candidate_recipe_digest",
                    "instrument_digest", "protocol_id", "objective_digest", "workload_digest",
                    "exact_candidate_required"}
        _keys(row, required, "validation row")
        if row["schema"] != ROW_SCHEMA or not isinstance(row["required"], bool) \
                or not isinstance(row["exact_candidate_required"], bool):
            raise CandidateError("malformed validation row schema/boolean")
        row_kind = _text(row["row_kind"], "validation row.row_kind")
        category = _text(row["category"], "validation row.category")
        if row_kind not in {"production", "seed"}:
            raise CandidateError("validation row kind must be production or seed")
        if row["required"] and (row_kind != "production" or category != "OPTIMUM"):
            raise CandidateError("required rows must be production OPTIMUM rows")
        if row_kind == "seed" and row["required"]:
            raise CandidateError("seed rows cannot be required")
        backend = _text(row["backend"], "validation row.backend")
        if backend not in {"cpu", "gpu"}:
            raise CandidateError("validation row backend must be cpu or gpu")
        drafter = row["drafter_digest"]
        return cls(_text(row["row_id"], "validation row.row_id"), row_kind, row["required"],
                   _sha(row["target_revision_digest"], "row target revision"),
                   _sha(row["control_target_revision_digest"], "row control target revision"),
                   backend,
                   _sha(row["candidate_build_digest"], "row candidate build"),
                   _sha(row["control_build_digest"], "row control build"),
                   _sha(row["model_digest"], "row model"),
                   None if drafter is None else _sha(drafter, "row drafter"), category,
                   _sha(row["control_recipe_digest"], "control recipe digest"),
                   _sha(row["candidate_recipe_digest"], "candidate recipe digest"),
                   _sha(row["instrument_digest"], "instrument digest"),
                   _text(row["protocol_id"], "protocol id"),
                   _sha(row["objective_digest"], "objective digest"),
                   _sha(row["workload_digest"], "workload digest"),
                   row["exact_candidate_required"])

    def to_dict(self) -> dict[str, Any]:
        return {"schema": ROW_SCHEMA, **self.__dict__}


@dataclass(frozen=True)
class RequiredRowSet:
    version: str
    rows: tuple[ValidationRow, ...]

    @classmethod
    def from_dict(cls, value: Any) -> "RequiredRowSet":
        row = _object(value, "required row set")
        _keys(row, {"schema", "version", "rows", "row_set_digest"}, "required row set")
        if row["schema"] != ROW_SET_SCHEMA:
            raise CandidateError(f"required row set: unsupported schema {row['schema']!r}")
        rows = tuple(ValidationRow.from_dict(item)
                     for item in _sequence(row["rows"], "required row set.rows"))
        if not rows or len({item.row_id for item in rows}) != len(rows):
            raise CandidateError("required row set needs unique rows")
        if not any(item.required for item in rows):
            raise CandidateError("required row set needs at least one required production row")
        result = cls(_text(row["version"], "required row set.version"),
                     tuple(sorted(rows, key=lambda item: item.row_id)))
        if _sha(row["row_set_digest"], "row set digest") != result.row_set_digest:
            raise CandidateError("required row set digest mismatch")
        return result

    @property
    def row_set_digest(self) -> str:
        return _digest({"version": self.version, "rows": [item.to_dict() for item in self.rows]})

    def to_dict(self) -> dict[str, Any]:
        return {"schema": ROW_SET_SCHEMA, "version": self.version,
                "rows": [item.to_dict() for item in self.rows],
                "row_set_digest": self.row_set_digest}


@dataclass(frozen=True)
class RowReceipt:
    batch_id: str
    candidate_manifest_digest: str
    comparator_manifest_digest: str
    row_set_digest: str
    row_id: str
    native_evidence_ref: str
    native_evidence_digest: str
    intended_use: str
    use_disposition: str

    @classmethod
    def from_dict(cls, value: Any) -> "RowReceipt":
        row = _object(value, "row receipt")
        _keys(row, {"schema", "batch_id", "candidate_manifest_digest",
                    "comparator_manifest_digest", "row_set_digest", "row_id",
                    "native_evidence_ref", "native_evidence_digest", "intended_use",
                    "use_disposition"}, "row receipt")
        if row["schema"] != RECEIPT_SCHEMA:
            raise CandidateError(f"row receipt: unsupported schema {row['schema']!r}")
        intended = _text(row["intended_use"], "row receipt.intended_use")
        if intended != "validate":
            raise CandidateError("row receipt intended_use must be validate")
        return cls(_text(row["batch_id"], "row receipt.batch_id"),
                   _sha(row["candidate_manifest_digest"], "receipt candidate"),
                   _sha(row["comparator_manifest_digest"], "receipt comparator"),
                   _sha(row["row_set_digest"], "receipt row set"),
                   _text(row["row_id"], "row receipt.row_id"),
                   _text(row["native_evidence_ref"], "native evidence ref"),
                   _sha(row["native_evidence_digest"], "native evidence digest"), intended,
                   _text(row["use_disposition"], "row receipt.use_disposition"))

    def to_dict(self) -> dict[str, str]:
        return {"schema": RECEIPT_SCHEMA, **self.__dict__}


@dataclass(frozen=True)
class ValidationRowState:
    row_id: str
    status: str
    reason: str | None
    receipt: RowReceipt | None

    @classmethod
    def from_dict(cls, value: Any) -> "ValidationRowState":
        row = _object(value, "validation row state")
        _keys(row, {"schema", "row_id", "status", "reason", "receipt"},
              "validation row state")
        if row["schema"] != ROW_STATE_SCHEMA:
            raise CandidateError(f"row state: unsupported schema {row['schema']!r}")
        status = _text(row["status"], "row state.status")
        if status not in ROW_STATUSES:
            raise CandidateError("unknown validation row status")
        reason = row["reason"]
        if reason is not None:
            reason = _text(reason, "row state.reason")
        receipt = None if row["receipt"] is None else RowReceipt.from_dict(row["receipt"])
        if status == "passed" and receipt is None:
            raise CandidateError("passed row needs a receipt")
        if status in {"prerequisite_missing", "failed", "inconclusive", "unsupported",
                      "nonidentifiable"} and reason is None:
            raise CandidateError(f"{status} row needs a reason")
        if status in {"pending", "running"} and (reason is not None or receipt is not None):
            raise CandidateError(f"{status} row cannot carry result data")
        return cls(_text(row["row_id"], "row state.row_id"), status, reason, receipt)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": ROW_STATE_SCHEMA, "row_id": self.row_id, "status": self.status,
                "reason": self.reason, "receipt": self.receipt.to_dict() if self.receipt else None}


@dataclass(frozen=True)
class ValidationBatch:
    batch_id: str
    candidate_manifest_digest: str
    comparator_manifest_digest: str
    row_set_digest: str
    expected_validated_predecessor: str | None
    launch_integration_tip: str
    accounted_keeps_since_gate: int
    accounted_threshold_generation: int
    required_loo_keep_ids: tuple[str, ...]
    required_row_ids: tuple[str, ...]
    rows: tuple[ValidationRowState, ...]

    @classmethod
    def from_dict(cls, value: Any) -> "ValidationBatch":
        row = _object(value, "validation batch")
        _keys(row, {"schema", "batch_id", "candidate_manifest_digest",
                    "comparator_manifest_digest", "row_set_digest",
                    "expected_validated_predecessor", "launch_integration_tip",
                    "accounted_keeps_since_gate", "accounted_threshold_generation",
                    "required_loo_keep_ids", "required_row_ids", "rows"},
              "validation batch")
        if row["schema"] != BATCH_SCHEMA:
            raise CandidateError(f"validation batch: unsupported schema {row['schema']!r}")
        predecessor = row["expected_validated_predecessor"]
        if predecessor is not None:
            predecessor = _sha(predecessor, "batch expected predecessor")
        accounted = row["accounted_keeps_since_gate"]
        if isinstance(accounted, bool) or not isinstance(accounted, int) or accounted < 0:
            raise CandidateError("batch accounted cadence must be a non-negative integer")
        threshold_generation = row["accounted_threshold_generation"]
        if isinstance(threshold_generation, bool) or not isinstance(threshold_generation, int) \
                or threshold_generation < 0:
            raise CandidateError("batch accounted threshold generation must be non-negative")
        rows = tuple(ValidationRowState.from_dict(item)
                     for item in _sequence(row["rows"], "validation batch.rows"))
        if not rows or len({item.row_id for item in rows}) != len(rows):
            raise CandidateError("validation batch needs unique row states")
        result = cls(_text(row["batch_id"], "validation batch.batch_id"),
                     _sha(row["candidate_manifest_digest"], "batch candidate"),
                     _sha(row["comparator_manifest_digest"], "batch comparator"),
                     _sha(row["row_set_digest"], "batch row set"), predecessor,
                     _sha(row["launch_integration_tip"], "batch launch integration tip"),
                     accounted, threshold_generation,
                     tuple(sorted(_strings(row["required_loo_keep_ids"], "required LOO keeps"))),
                     tuple(sorted(_strings(row["required_row_ids"], "required row ids",
                                           nonempty=True))),
                     tuple(sorted(rows, key=lambda item: item.row_id)))
        for state in result.rows:
            if state.receipt and (state.receipt.row_id != state.row_id
                                  or not result.matches_receipt(state.receipt)):
                raise CandidateError(f"receipt for row {state.row_id!r} mismatches its batch")
        return result

    def matches_receipt(self, receipt: RowReceipt) -> bool:
        return (receipt.batch_id == self.batch_id
                and receipt.candidate_manifest_digest == self.candidate_manifest_digest
                and receipt.comparator_manifest_digest == self.comparator_manifest_digest
                and receipt.row_set_digest == self.row_set_digest)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": BATCH_SCHEMA, "batch_id": self.batch_id,
                "candidate_manifest_digest": self.candidate_manifest_digest,
                "comparator_manifest_digest": self.comparator_manifest_digest,
                "row_set_digest": self.row_set_digest,
                "expected_validated_predecessor": self.expected_validated_predecessor,
                "launch_integration_tip": self.launch_integration_tip,
                "accounted_keeps_since_gate": self.accounted_keeps_since_gate,
                "accounted_threshold_generation": self.accounted_threshold_generation,
                "required_loo_keep_ids": list(self.required_loo_keep_ids),
                "required_row_ids": list(self.required_row_ids),
                "rows": [item.to_dict() for item in self.rows]}

    def validated(self) -> "ValidationBatch":
        return ValidationBatch.from_dict(self.to_dict())


TrustedVerifier = Callable[[RowReceipt, ValidationRow, CandidateManifest, CandidateManifest], bool]
TrustedLOOVerifier = Callable[[LOOResult, LOOPlan, CandidateManifest, RequiredRowSet], bool]


def record_row(batch: ValidationBatch, row_set: RequiredRowSet, state: ValidationRowState) \
        -> ValidationBatch:
    batch = batch.validated()
    row_set = RequiredRowSet.from_dict(row_set.to_dict())
    if batch.row_set_digest != row_set.row_set_digest:
        raise TransitionError("row set does not match frozen batch")
    states = {item.row_id: item for item in batch.rows}
    if state.row_id not in states:
        raise TransitionError("row result is not in the frozen batch")
    state = ValidationRowState.from_dict(state.to_dict())
    old = states[state.row_id]
    if old in (state,):
        return batch
    if old.status in TERMINAL_ROW_STATUSES:
        raise TransitionError("late/conflicting result for terminal row")
    if state.receipt and (state.receipt.row_id != state.row_id
                          or not batch.matches_receipt(state.receipt)):
        raise TransitionError("row receipt identities mismatch")
    states[state.row_id] = state
    return replace(batch, rows=tuple(sorted(states.values(), key=lambda item: item.row_id))).validated()


@dataclass(frozen=True)
class CandidateState:
    production_ref_digest: str
    integration_tip: str
    validated_candidate: str | None
    keeps_since_gate: int
    threshold_generation: int
    covered_threshold_generation: int
    validation_debt: tuple[str, ...]
    active_batches: tuple[ValidationBatch, ...]
    completed_batches: tuple[ValidationBatch, ...]
    integrated_requests: tuple[tuple[str, str], ...]
    outstanding_summary_refs: tuple[str, ...]
    stale_summary_refs: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: Any) -> "CandidateState":
        row = _object(value, "candidate state")
        required = {"schema", "production_ref_digest", "integration_tip",
                    "validated_candidate", "keeps_since_gate", "threshold_generation",
                    "covered_threshold_generation",
                    "validation_debt",
                    "active_batches", "completed_batches", "integrated_requests",
                    "outstanding_summary_refs", "stale_summary_refs"}
        _keys(row, required, "candidate state")
        if row["schema"] != STATE_SCHEMA:
            raise CandidateError(f"candidate state: unsupported schema {row['schema']!r}")
        cadence = row["keeps_since_gate"]
        if isinstance(cadence, bool) or not isinstance(cadence, int) or cadence < 0:
            raise CandidateError("keeps_since_gate must be a non-negative integer")
        threshold_generation = row["threshold_generation"]
        covered_generation = row["covered_threshold_generation"]
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0
               for item in (threshold_generation, covered_generation)) \
                or covered_generation > threshold_generation:
            raise CandidateError("threshold generations must be ordered non-negative integers")
        validated = row["validated_candidate"]
        if validated is not None:
            validated = _sha(validated, "validated candidate")
        request_map = _object(row["integrated_requests"], "integrated requests")
        requests = tuple(sorted((_text(key, "integrated request id"),
                                 _sha(item, "integrated request digest"))
                                for key, item in request_map.items()))
        active = tuple(ValidationBatch.from_dict(item)
                       for item in _sequence(row["active_batches"], "active batches"))
        completed = tuple(ValidationBatch.from_dict(item)
                          for item in _sequence(row["completed_batches"], "completed batches"))
        ids = [item.batch_id for item in active + completed]
        if len(ids) != len(set(ids)):
            raise CandidateError("candidate state contains duplicate batch ids")
        return cls(_sha(row["production_ref_digest"], "production ref"),
                   _sha(row["integration_tip"], "integration tip"), validated, cadence,
                   threshold_generation, covered_generation,
                   tuple(sorted(_strings(row["validation_debt"], "validation debt"))), active,
                   completed, requests,
                   tuple(sorted(_strings(row["outstanding_summary_refs"], "outstanding summaries"))),
                   tuple(sorted(_strings(row["stale_summary_refs"], "stale summaries"))))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": STATE_SCHEMA, "production_ref_digest": self.production_ref_digest,
                "integration_tip": self.integration_tip,
                "validated_candidate": self.validated_candidate,
                "keeps_since_gate": self.keeps_since_gate,
                "threshold_generation": self.threshold_generation,
                "covered_threshold_generation": self.covered_threshold_generation,
                "validation_debt": list(self.validation_debt),
                "active_batches": [item.to_dict() for item in self.active_batches],
                "completed_batches": [item.to_dict() for item in self.completed_batches],
                "integrated_requests": dict(self.integrated_requests),
                "outstanding_summary_refs": list(self.outstanding_summary_refs),
                "stale_summary_refs": list(self.stale_summary_refs)}

    def validated(self) -> "CandidateState":
        return CandidateState.from_dict(self.to_dict())

    @property
    def gate_due(self) -> bool:
        return (self.keeps_since_gate >= 4
                or self.threshold_generation > self.covered_threshold_generation)


def integrate_candidate(state: CandidateState, previous: CandidateManifest,
                        candidate: CandidateManifest, *, request_id: str,
                        threshold_signal: bool = False) -> tuple[CandidateState, bool]:
    state, previous, candidate = state.validated(), previous.validated(), candidate.validated()
    request_id = _text(request_id, "integration request id")
    if not isinstance(threshold_signal, bool):
        raise TransitionError("existing threshold signal must be a boolean supplied by its owner")
    payload = _digest({"previous": previous.manifest_digest,
                       "candidate": candidate.manifest_digest,
                       "threshold_signal": threshold_signal})
    requests = dict(state.integrated_requests)
    if request_id in requests:
        if requests[request_id] != payload:
            raise TransitionError("integration request id reused with different payload")
        return state, state.gate_due
    if state.integration_tip != previous.manifest_digest \
            or candidate.parent_manifest_digest != previous.manifest_digest:
        raise TransitionError("integration tip/parent CAS mismatch")
    old_keeps = {item.keep_id for item in previous.keeps}
    new_keeps = [item for item in candidate.keeps if item.keep_id not in old_keeps]
    if len(new_keeps) != 1 or candidate.keeps[:-1] != previous.keeps \
            or candidate.keeps[-1] != new_keeps[0]:
        raise TransitionError("integration must add exactly one unique keep")
    if new_keeps[0].parent_manifest_digest != previous.manifest_digest:
        raise TransitionError("integrated keep parent does not match previous manifest")
    if new_keeps[0].kind == "runtime" and (candidate.sources != previous.sources
                                            or candidate.builds != previous.builds):
        raise TransitionError("runtime keep cannot change source or build identity")
    if new_keeps[0].kind == "build" and candidate.sources != previous.sources:
        raise TransitionError("build keep cannot change source identity")
    changed_pairs: set[tuple[str, str]] = set()
    if previous.sources != candidate.sources:
        changed_pairs.add((source_set_digest(previous.sources), source_set_digest(candidate.sources)))
    if previous.builds != candidate.builds:
        changed_pairs.add((previous.build_set_digest, candidate.build_set_digest))
    if len(previous.targets) != len(candidate.targets):
        raise TransitionError("one keep cannot silently add or drop targets")
    old_targets = {item.target_id: item for item in previous.targets}
    new_targets = {item.target_id: item for item in candidate.targets}
    if set(old_targets) != set(new_targets):
        raise TransitionError("one keep cannot silently add or drop logical targets")
    for target_id, old in old_targets.items():
        new = new_targets[target_id]
        old_values, new_values = old.to_dict(), new.to_dict()
        for key in old_values.keys() - {"schema"}:
            if old_values[key] != new_values[key]:
                if not isinstance(old_values[key], str) or not isinstance(new_values[key], str):
                    raise TransitionError("one keep cannot alter untyped target obligations")
                changed_pairs.add((old_values[key], new_values[key]))
    if previous.dependency_digest != candidate.dependency_digest:
        changed_pairs.add((previous.dependency_digest, candidate.dependency_digest))
    declared_pairs = {(item.previous_digest, item.current_digest)
                      for item in new_keeps[0].changes}
    if not changed_pairs or not changed_pairs <= declared_pairs:
        raise TransitionError("candidate contains changes not declared by its unique keep")
    if candidate.production_ref_digest != state.production_ref_digest:
        raise TransitionError("integration cannot change frozen production_ref")
    requests[request_id] = payload
    result = replace(state, integration_tip=candidate.manifest_digest,
                     keeps_since_gate=state.keeps_since_gate + 1,
                     threshold_generation=(state.threshold_generation
                                           + (1 if threshold_signal else 0)),
                     integrated_requests=tuple(sorted(requests.items()))).validated()
    return result, result.gate_due


def _validate_row_bindings(row_set: RequiredRowSet, candidate: CandidateManifest,
                           comparator: CandidateManifest) -> None:
    candidate_targets = {item.target_revision_digest: item for item in candidate.targets}
    for row in row_set.rows:
        if not row.required:
            continue
        target = candidate_targets.get(row.target_revision_digest)
        if target is None or (target.backend not in {row.backend, "both"}
                or row.candidate_build_digest != target.build_execution_digest
                or row.candidate_recipe_digest != target.resolved_recipe_execution_digest
                or row.model_digest != target.model_digest
                or row.drafter_digest != target.drafter_digest
                or row.workload_digest != target.workload_digest):
            raise TransitionError(f"required row {row.row_id!r} target identity mismatch")
        controls = [item for item in comparator.targets
                    if item.target_revision_digest == row.control_target_revision_digest
                    and item.backend in {row.backend, "both"}
                    and item.build_execution_digest == row.control_build_digest
                    and item.resolved_recipe_execution_digest == row.control_recipe_digest
                    and item.model_digest == row.model_digest
                    and item.drafter_digest == row.drafter_digest
                    and item.workload_digest == row.workload_digest]
        if not controls:
            raise TransitionError(f"required row {row.row_id!r} comparator identity mismatch")


def _validate_batch_obligations(batch: ValidationBatch, row_set: RequiredRowSet,
                                candidate: CandidateManifest,
                                comparator: CandidateManifest) -> None:
    if (batch.row_set_digest != row_set.row_set_digest
            or batch.candidate_manifest_digest != candidate.manifest_digest
            or batch.comparator_manifest_digest != comparator.manifest_digest
            or candidate.production_ref_digest != comparator.production_ref_digest):
        raise TransitionError("batch obligation identity mismatch")
    if {item.row_id for item in batch.rows} != {item.row_id for item in row_set.rows}:
        raise TransitionError("batch must freeze exactly the row-set rows")
    required_rows = [item for item in row_set.rows if item.required]
    required_ids = {item.row_id for item in required_rows}
    if not required_rows or set(batch.required_row_ids) != required_ids:
        raise TransitionError("batch required-row obligation set mismatch")
    if set(batch.required_loo_keep_ids) != {item.keep_id for item in candidate.keeps}:
        raise TransitionError("batch must require LOO coverage for every candidate keep")
    target_rows: dict[str, set[str]] = {}
    for item in required_rows:
        target_rows.setdefault(item.target_revision_digest, set()).add(item.backend)
    for target in candidate.targets:
        if not target.production_required:
            continue
        needed = {"cpu", "gpu"} if target.backend == "both" else {target.backend}
        if not needed <= target_rows.get(target.target_revision_digest, set()):
            raise TransitionError("required row set omits a production-required target/backend")
    _validate_row_bindings(row_set, candidate, comparator)


def start_batch(state: CandidateState, batch: ValidationBatch, row_set: RequiredRowSet,
                candidate: CandidateManifest, comparator: CandidateManifest) -> CandidateState:
    state, batch = state.validated(), batch.validated()
    row_set = RequiredRowSet.from_dict(row_set.to_dict())
    candidate, comparator = candidate.validated(), comparator.validated()
    existing = next((item for item in state.active_batches + state.completed_batches
                     if item.batch_id == batch.batch_id), None)
    if existing is not None:
        if existing == batch:
            _validate_batch_obligations(batch, row_set, candidate, comparator)
            return state
        raise TransitionError("batch id reused with different payload")
    _validate_batch_obligations(batch, row_set, candidate, comparator)
    if (batch.candidate_manifest_digest != candidate.manifest_digest
            or batch.comparator_manifest_digest != comparator.manifest_digest
            or batch.launch_integration_tip != state.integration_tip
            or batch.launch_integration_tip != candidate.manifest_digest
            or batch.expected_validated_predecessor != state.validated_candidate
            or candidate.production_ref_digest != state.production_ref_digest
            or comparator.production_ref_digest != state.production_ref_digest):
        raise TransitionError("batch launch candidate/comparator/predecessor identity mismatch")
    if state.validated_candidate is not None \
            and comparator.manifest_digest != state.validated_candidate:
        raise TransitionError("batch comparator is not the expected validated predecessor")
    if batch.accounted_keeps_since_gate != state.keeps_since_gate \
            or batch.accounted_threshold_generation != state.threshold_generation:
        raise TransitionError("batch launch cadence snapshot mismatch")
    if state.active_batches:
        raise TransitionError("a validation batch is already active")
    return replace(state, active_batches=state.active_batches + (batch,),
                   validation_debt=tuple(sorted(set(state.validation_debt) | {batch.batch_id}))).validated()


def refuse_gate_start(state: CandidateState, batch_id: str, reason: str) -> CandidateState:
    state.validated()
    _text(batch_id, "batch id")
    _text(reason, "gate refusal reason")
    # Refusal is intentionally not a completed run and therefore cannot reset cadence.
    return state


def complete_batch(state: CandidateState, batch: ValidationBatch) -> CandidateState:
    state, batch = state.validated(), batch.validated()
    active = {item.batch_id: item for item in state.active_batches}
    if batch.batch_id not in active:
        raise TransitionError("completed batch was not active")
    frozen = active[batch.batch_id]
    if (batch.candidate_manifest_digest, batch.comparator_manifest_digest,
            batch.row_set_digest, batch.expected_validated_predecessor,
            batch.launch_integration_tip, batch.accounted_keeps_since_gate,
            batch.accounted_threshold_generation, batch.required_loo_keep_ids,
            batch.required_row_ids,
            tuple(item.row_id for item in batch.rows)) != (
            frozen.candidate_manifest_digest, frozen.comparator_manifest_digest,
            frozen.row_set_digest, frozen.expected_validated_predecessor,
            frozen.launch_integration_tip, frozen.accounted_keeps_since_gate,
            frozen.accounted_threshold_generation, frozen.required_loo_keep_ids,
            frozen.required_row_ids,
            tuple(item.row_id for item in frozen.rows)):
        raise TransitionError("completed batch retargets its frozen identity")
    states = {item.row_id: item for item in batch.rows}
    if any(states[row_id].status not in TERMINAL_ROW_STATUSES
           for row_id in batch.required_row_ids):
        raise TransitionError("batch cannot complete with pending/running required rows")
    del active[batch.batch_id]
    attempted = any(states[row_id].status in {"passed", "failed", "inconclusive"}
                    for row_id in batch.required_row_ids)
    if state.keeps_since_gate < batch.accounted_keeps_since_gate:
        raise TransitionError("batch cadence snapshot exceeds current cadence")
    remaining = max(0, state.keeps_since_gate - batch.accounted_keeps_since_gate)
    return replace(state, keeps_since_gate=remaining if attempted else state.keeps_since_gate,
                   covered_threshold_generation=(max(state.covered_threshold_generation,
                                                      batch.accounted_threshold_generation)
                                                 if attempted
                                                 else state.covered_threshold_generation),
                   active_batches=tuple(sorted(active.values(), key=lambda item: item.batch_id)),
                   completed_batches=state.completed_batches + (batch,),
                   validation_debt=state.validation_debt).validated()


def advance_validated(state: CandidateState, batch: ValidationBatch,
                      row_set: RequiredRowSet, candidate: CandidateManifest,
                      comparator: CandidateManifest, *, verifier: TrustedVerifier | None,
                      loo_plans: Mapping[str, LOOPlan] | None = None,
                      loo_results: Mapping[str, LOOResult] | None = None,
                      loo_verifier: TrustedLOOVerifier | None = None) -> CandidateState:
    state, batch = state.validated(), batch.validated()
    row_set = RequiredRowSet.from_dict(row_set.to_dict())
    candidate, comparator = candidate.validated(), comparator.validated()
    if verifier is None:
        raise TrustedVerificationRequired("trusted evidence validator adapter is not connected")
    _validate_batch_obligations(batch, row_set, candidate, comparator)
    if candidate.production_ref_digest != state.production_ref_digest:
        raise TransitionError("batch candidate production reference mismatches state")
    if state.validated_candidate != batch.expected_validated_predecessor:
        raise TransitionError("validated-candidate predecessor CAS mismatch")
    if state.validated_candidate is not None \
            and comparator.manifest_digest != state.validated_candidate:
        raise TransitionError("comparator is not the expected validated predecessor")
    if not any(item == batch for item in state.completed_batches):
        raise TransitionError("validated batch is not an immutable completed batch")
    states = {item.row_id: item for item in batch.rows}
    candidate_targets = {item.target_revision_digest: item for item in candidate.targets}
    for row in row_set.rows:
        if not row.required:
            continue
        state_row = states.get(row.row_id)
        if state_row is None or state_row.status != "passed" or state_row.receipt is None:
            raise TransitionError(f"required row {row.row_id!r} is not passed")
        if row.target_revision_digest not in candidate_targets:
            raise TransitionError(f"required row {row.row_id!r} names the wrong candidate target")
        target = candidate_targets[row.target_revision_digest]
        if (target.backend not in {row.backend, "both"}
                or row.candidate_build_digest != target.build_execution_digest
                or row.candidate_recipe_digest != target.resolved_recipe_execution_digest
                or row.model_digest != target.model_digest
                or row.drafter_digest != target.drafter_digest
                or row.workload_digest != target.workload_digest):
            raise TransitionError(f"required row {row.row_id!r} target identity mismatch")
        controls = [item for item in comparator.targets
                    if item.target_revision_digest == row.control_target_revision_digest
                    and item.backend in {row.backend, "both"}
                    and item.build_execution_digest == row.control_build_digest
                    and item.resolved_recipe_execution_digest == row.control_recipe_digest
                    and item.model_digest == row.model_digest
                    and item.drafter_digest == row.drafter_digest
                    and item.workload_digest == row.workload_digest]
        if not controls:
            raise TransitionError(f"required row {row.row_id!r} comparator identity mismatch")
        if row.exact_candidate_required and row.candidate_recipe_digest != \
                target.resolved_recipe_execution_digest:
            raise TransitionError(f"row {row.row_id!r} lacks exact-candidate identity")
        if not batch.matches_receipt(state_row.receipt) \
                or state_row.receipt.row_id != row.row_id \
                or state_row.receipt.use_disposition != "permitted":
            raise TransitionError(f"required row {row.row_id!r} receipt is inapplicable")
        try:
            verified = verifier(state_row.receipt, row, candidate, comparator)
        except Exception as exc:
            raise TransitionError(f"trusted verifier errored for row {row.row_id!r}") from exc
        if verified is not True:
            raise TransitionError(f"trusted verifier refused row {row.row_id!r}")
    plans, results = dict(loo_plans or {}), dict(loo_results or {})
    required_loo = set(batch.required_loo_keep_ids)
    if required_loo != {item.keep_id for item in candidate.keeps}:
        raise TransitionError("batch lacks complete candidate keep LOO coverage")
    if required_loo - set(results) or required_loo - set(plans):
        raise TransitionError("required LOO results are missing")
    if required_loo and loo_verifier is None:
        raise TrustedVerificationRequired("trusted LOO evidence adapter is not connected")
    for keep_id in batch.required_loo_keep_ids:
        result = LOOResult.from_dict(results[keep_id].to_dict())
        plan = LOOPlan.from_dict(plans[keep_id].to_dict())
        if (plan.status != "planned" or plan.derived_manifest_digest is None
                or plan.candidate_manifest_digest != candidate.manifest_digest
                or plan.keep_id != keep_id
                or plan.binary_execution_digest != candidate.build_set_digest
                or result.plan_digest != plan.plan_digest
                or result.candidate_manifest_digest != candidate.manifest_digest
                or result.keep_id != keep_id
                or result.derived_manifest_digest != plan.derived_manifest_digest
                or result.row_set_digest != row_set.row_set_digest
                or result.disposition not in {"neutral", "supports_keep", "supports_removal"}):
            raise TransitionError(f"LOO result for {keep_id!r} is not eligible")
        assert loo_verifier is not None
        try:
            loo_verified = loo_verifier(result, plan, candidate, row_set)
        except Exception as exc:
            raise TransitionError(f"trusted verifier errored for LOO {keep_id!r}") from exc
        if loo_verified is not True:
            raise TransitionError(f"trusted verifier refused LOO {keep_id!r}")
    debt = set(state.validation_debt)
    debt.discard(batch.batch_id)
    return replace(state, validated_candidate=candidate.manifest_digest,
                   validation_debt=tuple(sorted(debt)),
                   stale_summary_refs=tuple(sorted(set(state.stale_summary_refs)
                                                   | set(state.outstanding_summary_refs))),
                   outstanding_summary_refs=()).validated()


@dataclass(frozen=True)
class EquivalenceReceipt:
    source_digest: str
    destination_digest: str
    workload_digest: str
    dependency_digest: str
    intended_use: str
    native_evidence_digest: str

    @classmethod
    def from_dict(cls, value: Any) -> "EquivalenceReceipt":
        row = _object(value, "equivalence receipt")
        _keys(row, {"schema", "source_digest", "destination_digest", "workload_digest",
                    "dependency_digest", "intended_use", "native_evidence_digest"},
              "equivalence receipt")
        if row["schema"] != EQUIVALENCE_SCHEMA:
            raise CandidateError("unsupported equivalence receipt schema")
        use = _text(row["intended_use"], "equivalence intended use")
        if use not in EQUIVALENCE_USES - {"exact"}:
            raise CandidateError("invalid equivalence intended use")
        return cls(_sha(row["source_digest"], "equivalence source"),
                   _sha(row["destination_digest"], "equivalence destination"),
                   _sha(row["workload_digest"], "equivalence workload"),
                   _sha(row["dependency_digest"], "equivalence dependency"), use,
                   _sha(row["native_evidence_digest"], "equivalence evidence"))

    def to_dict(self) -> dict[str, str]:
        return {"schema": EQUIVALENCE_SCHEMA, **self.__dict__}


EquivalenceVerifier = Callable[[EquivalenceReceipt], bool]


def equivalence_permits(*, actual_digest: str, required_digest: str, use: str,
                        workload_digest: str | None = None,
                        dependency_digest: str | None = None,
                        receipt: EquivalenceReceipt | None = None,
                        verifier: EquivalenceVerifier | None = None) -> bool:
    _sha(actual_digest, "actual artifact digest")
    _sha(required_digest, "required artifact digest")
    if use not in EQUIVALENCE_USES:
        raise CandidateError("unknown equivalence use")
    if actual_digest == required_digest:
        return True
    if use in {"exact", "timing"}:
        return False
    if receipt is None or verifier is None or workload_digest is None \
            or dependency_digest is None:
        return False
    receipt = EquivalenceReceipt.from_dict(receipt.to_dict())
    identities_match = (receipt.source_digest == actual_digest
            and receipt.destination_digest == required_digest
            and receipt.workload_digest == _sha(workload_digest, "equivalence workload")
            and receipt.dependency_digest == _sha(dependency_digest, "equivalence dependency")
            and receipt.intended_use == use)
    if not identities_match:
        return False
    try:
        return verifier(receipt) is True
    except Exception as exc:
        raise TransitionError("trusted equivalence verifier errored") from exc


__all__ = ["BATCH_SCHEMA", "BUILD_SCHEMA", "CANDIDATE_SCHEMA", "CHANGE_SCHEMA",
           "EQUIVALENCE_SCHEMA", "EquivalenceReceipt",
           "KEEP_SCHEMA", "LOO_PLAN_SCHEMA", "LOO_RESULT_SCHEMA", "RECEIPT_SCHEMA",
           "ROW_SCHEMA", "ROW_SET_SCHEMA", "ROW_STATE_SCHEMA", "SOURCE_SCHEMA",
           "STATE_SCHEMA", "TARGET_SCHEMA", "BuildIdentity", "CandidateError",
           "CandidateManifest", "CandidateState", "CandidateTarget", "FieldChange",
           "KeepTreatment", "LOOPlan", "LOOResult", "RequiredRowSet", "RowReceipt",
           "SourceIdentity", "TransitionError", "TrustedVerificationRequired",
           "ValidationBatch", "ValidationRow", "ValidationRowState", "advance_validated",
           "complete_batch", "equivalence_permits", "integrate_candidate", "plan_loo",
           "record_row", "refuse_gate_start", "source_set_digest", "start_batch"]
