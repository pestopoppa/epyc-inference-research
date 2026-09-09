"""Pure campaign enrollment for the unified AutoKernel loop.

This module resolves a versioned JSON/YAML declaration into immutable local
identities.  It deliberately does not launch, download, build, claim resources,
or decide statistical gates.  Those are later consumers of ``ResolvedCampaign``.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml


MANIFEST_SCHEMA = "epyc.autokernel.campaign_manifest.v1"
RESOLVED_SCHEMA = "epyc.autokernel.resolved_campaign.v1"
TARGET_SCHEMA = "epyc.autokernel.target_spec.v1"
TARGET_REVISION_SCHEMA = "epyc.autokernel.target_revision.v1"
EXECUTION_SCHEMA = "epyc.autokernel.resolved_execution.v1"
RESOURCE_SCHEMA = "epyc.autokernel.resource_request.v1"
ARTIFACT_SCHEMA = "epyc.autokernel.local_artifact.v1"

BACKENDS = frozenset({"cpu", "gpu", "both"})
METRIC_DIRECTIONS = frozenset({"higher", "lower"})
TARGET_STATUSES = frozenset({"ready", "missing_artifact", "unsupported_capability",
                             "mismatched_ref"})


class ManifestError(ValueError):
    """The declaration is malformed or conflicts with a frozen request."""


FrozenPairs = tuple[tuple[str, str], ...]


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestError(f"{label} must be an object")
    return dict(value)


def _keys(value: Mapping[str, Any], *, required: set[str], optional: set[str],
          label: str) -> None:
    missing = required - set(value)
    extra = set(value) - required - optional
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing {sorted(missing)}")
        if extra:
            parts.append(f"unknown {sorted(extra)}")
        raise ManifestError(f"{label}: " + "; ".join(parts))


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{label} must be a non-empty string")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ManifestError(f"{label} must be a positive finite integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ManifestError(f"{label} must be a non-negative integer")
    return value


def _strings(value: Any, label: str, *, nonempty: bool = False) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ManifestError(f"{label} must be an array of strings")
    result = tuple(_text(item, f"{label}[]") for item in value)
    if nonempty and not result:
        raise ManifestError(f"{label} must not be empty")
    if len(set(result)) != len(result):
        raise ManifestError(f"{label} contains duplicates")
    return result


def _pairs(value: Any, label: str) -> FrozenPairs:
    row = _mapping(value, label)
    return tuple(sorted((_text(key, f"{label} key"), _text(item, f"{label}.{key}"))
                        for key, item in row.items()))


def _env_pairs(value: Any, label: str) -> FrozenPairs:
    """Freeze an environment mapping while preserving valid empty values."""
    row = _mapping(value, label)
    pairs: list[tuple[str, str]] = []
    for key, item in row.items():
        env_key = _text(key, f"{label} key")
        if not isinstance(item, str):
            raise ManifestError(f"{label}.{key} must be a string")
        pairs.append((env_key, item))
    return tuple(sorted(pairs))


def _pairs_dict(value: FrozenPairs) -> dict[str, str]:
    return dict(value)


@dataclass(frozen=True)
class ResourceRequest:
    cpu_logical: tuple[int, ...]
    gpu_ids: tuple[str, ...]
    stage_timeout_s: int
    build_timeout_s: int
    build_jobs: int
    max_builds: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResourceRequest":
        row = _mapping(value, "resources")
        _keys(row, required={"schema", "cpu_logical", "gpu_ids", "stage_timeout_s",
                             "build_timeout_s", "build_jobs", "max_builds"},
              optional=set(), label="resources")
        if row["schema"] != RESOURCE_SCHEMA:
            raise ManifestError(f"resources: unsupported schema {row['schema']!r}")
        cpus = row["cpu_logical"]
        if isinstance(cpus, (str, bytes)) or not isinstance(cpus, Sequence):
            raise ManifestError("resources.cpu_logical must be an integer array")
        cpu_logical = tuple(_nonnegative_int(cpu, "resources.cpu_logical[]") for cpu in cpus)
        if len(set(cpu_logical)) != len(cpu_logical):
            raise ManifestError("resources.cpu_logical contains duplicates")
        return cls(cpu_logical=cpu_logical,
                   gpu_ids=_strings(row["gpu_ids"], "resources.gpu_ids"),
                   stage_timeout_s=_positive_int(row["stage_timeout_s"],
                                                 "resources.stage_timeout_s"),
                   build_timeout_s=_positive_int(row["build_timeout_s"],
                                                 "resources.build_timeout_s"),
                   build_jobs=_positive_int(row["build_jobs"], "resources.build_jobs"),
                   max_builds=_positive_int(row["max_builds"], "resources.max_builds"))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": RESOURCE_SCHEMA, "cpu_logical": list(self.cpu_logical),
                "gpu_ids": list(self.gpu_ids), "stage_timeout_s": self.stage_timeout_s,
                "build_timeout_s": self.build_timeout_s, "build_jobs": self.build_jobs,
                "max_builds": self.max_builds}


@dataclass(frozen=True)
class TargetSpec:
    request_id: str
    target_id: str
    backend: str
    model_ref: str
    build_ref: str
    recipe_ref: str
    baseline_ref: str | None
    context: int
    concurrency: int
    speculation: str
    drafter_ref: str | None
    env: FrozenPairs
    metric: str
    metric_direction: str
    roles: tuple[str, ...]
    required_obligations: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TargetSpec":
        row = _mapping(value, "target")
        _keys(row, required={"schema", "request_id", "target_id", "backend", "model_ref",
                             "build_ref", "recipe_ref", "context", "concurrency",
                             "speculation", "env", "metric", "metric_direction", "roles",
                             "required_obligations"},
              optional={"baseline_ref", "drafter_ref"}, label="target")
        if row["schema"] != TARGET_SCHEMA:
            raise ManifestError(f"target: unsupported schema {row['schema']!r}")
        backend = _text(row["backend"], "target.backend")
        if backend not in BACKENDS:
            raise ManifestError(f"target.backend must be one of {sorted(BACKENDS)}")
        direction = _text(row["metric_direction"], "target.metric_direction")
        if direction not in METRIC_DIRECTIONS:
            raise ManifestError("target.metric_direction must be 'higher' or 'lower'")
        baseline = row.get("baseline_ref")
        drafter = row.get("drafter_ref")
        if baseline is not None:
            baseline = _text(baseline, "target.baseline_ref")
        if drafter is not None:
            drafter = _text(drafter, "target.drafter_ref")
        return cls(request_id=_text(row["request_id"], "target.request_id"),
                   target_id=_text(row["target_id"], "target.target_id"),
                   backend=backend, model_ref=_text(row["model_ref"], "target.model_ref"),
                   build_ref=_text(row["build_ref"], "target.build_ref"),
                   recipe_ref=_text(row["recipe_ref"], "target.recipe_ref"),
                   baseline_ref=baseline,
                   context=_positive_int(row["context"], "target.context"),
                   concurrency=_positive_int(row["concurrency"], "target.concurrency"),
                   speculation=_text(row["speculation"], "target.speculation"),
                   drafter_ref=drafter, env=_env_pairs(row["env"], "target.env"),
                   metric=_text(row["metric"], "target.metric"),
                   metric_direction=direction,
                   roles=_strings(row["roles"], "target.roles", nonempty=True),
                   required_obligations=_strings(row["required_obligations"],
                                                 "target.required_obligations"))

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "schema": TARGET_SCHEMA, "request_id": self.request_id,
            "target_id": self.target_id, "backend": self.backend,
            "model_ref": self.model_ref, "build_ref": self.build_ref,
            "recipe_ref": self.recipe_ref, "context": self.context,
            "concurrency": self.concurrency, "speculation": self.speculation,
            "env": _pairs_dict(self.env), "metric": self.metric,
            "metric_direction": self.metric_direction, "roles": list(self.roles),
            "required_obligations": list(self.required_obligations),
        }
        if self.baseline_ref is not None:
            row["baseline_ref"] = self.baseline_ref
        if self.drafter_ref is not None:
            row["drafter_ref"] = self.drafter_ref
        return row

    @property
    def spec_digest(self) -> str:
        return _digest(self.to_dict())


@dataclass(frozen=True)
class CampaignManifest:
    campaign_id: str
    request_id: str
    source_snapshot: FrozenPairs
    resources: ResourceRequest
    objective_ref: str
    actors: FrozenPairs
    fallbacks: tuple[tuple[str, tuple[str, ...]], ...]
    production: tuple[TargetSpec, ...]
    seeds: tuple[TargetSpec, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CampaignManifest":
        row = _mapping(value, "manifest")
        _keys(row, required={"schema", "campaign_id", "request_id", "source_snapshot",
                             "resources", "objective_ref", "actors", "fallbacks",
                             "production", "seeds"}, optional=set(), label="manifest")
        if row["schema"] != MANIFEST_SCHEMA:
            raise ManifestError(f"manifest: unsupported schema {row['schema']!r}")
        actors = _pairs(row["actors"], "manifest.actors")
        if not actors:
            raise ManifestError("manifest.actors must not be empty")
        fallback_map = _mapping(row["fallbacks"], "manifest.fallbacks")
        if set(fallback_map) != {key for key, _ in actors}:
            raise ManifestError("manifest.fallbacks must explicitly name exactly every actor role")
        fallbacks = tuple(sorted((key, _strings(value, f"manifest.fallbacks.{key}"))
                                 for key, value in fallback_map.items()))
        production = tuple(TargetSpec.from_dict(item) for item in _target_list(
            row["production"], "manifest.production"))
        seeds = tuple(TargetSpec.from_dict(item) for item in _target_list(
            row["seeds"], "manifest.seeds"))
        if not production and not seeds:
            raise ManifestError("manifest needs at least one production selector or local seed")
        resources = ResourceRequest.from_dict(row["resources"])
        all_targets = production + seeds
        if any(item.backend in {"cpu", "both"} for item in all_targets) and not resources.cpu_logical:
            raise ManifestError("CPU/both targets require non-empty resources.cpu_logical")
        if any(item.backend in {"gpu", "both"} for item in all_targets) and not resources.gpu_ids:
            raise ManifestError("GPU/both targets require non-empty resources.gpu_ids")
        request_id = _text(row["request_id"], "manifest.request_id")
        if any(item.request_id != request_id for item in all_targets):
            raise ManifestError("every target.request_id must equal manifest.request_id")
        source_snapshot = _pairs(row["source_snapshot"], "manifest.source_snapshot")
        if not source_snapshot:
            raise ManifestError("manifest.source_snapshot must not be empty")
        return cls(campaign_id=_text(row["campaign_id"], "manifest.campaign_id"),
                   request_id=request_id,
                   source_snapshot=source_snapshot,
                   resources=resources,
                   objective_ref=_text(row["objective_ref"], "manifest.objective_ref"),
                   actors=actors, fallbacks=fallbacks,
                   production=production, seeds=seeds)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": MANIFEST_SCHEMA, "campaign_id": self.campaign_id,
                "request_id": self.request_id,
                "source_snapshot": _pairs_dict(self.source_snapshot),
                "resources": self.resources.to_dict(), "objective_ref": self.objective_ref,
                "actors": _pairs_dict(self.actors),
                "fallbacks": {key: list(values) for key, values in self.fallbacks},
                "production": [item.to_dict() for item in self.production],
                "seeds": [item.to_dict() for item in self.seeds]}

    @property
    def manifest_digest(self) -> str:
        return _digest(self.to_dict())


def _target_list(value: Any, label: str) -> Sequence[Mapping[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ManifestError(f"{label} must be an array")
    return value


@dataclass(frozen=True)
class ArtifactIdentity:
    kind: str
    ref: str
    path: str
    sha256: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, kind: str, ref: str) -> "ArtifactIdentity":
        row = _mapping(value, f"resolved {kind} {ref}")
        _keys(row, required={"schema", "kind", "ref", "path", "sha256"}, optional=set(),
              label=f"resolved {kind} {ref}")
        if row["schema"] != ARTIFACT_SCHEMA or row["kind"] != kind or row["ref"] != ref:
            raise ManifestError(f"resolved {kind} {ref}: identity does not match request")
        path = _text(row["path"], f"resolved {kind} {ref}.path")
        if not Path(path).is_absolute():
            raise ManifestError(f"resolved {kind} {ref}.path must be absolute")
        digest = _text(row["sha256"], f"resolved {kind} {ref}.sha256")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ManifestError(f"resolved {kind} {ref}.sha256 must be lowercase SHA-256")
        return cls(kind=kind, ref=ref, path=path, sha256=digest)

    def to_dict(self) -> dict[str, str]:
        return {"schema": ARTIFACT_SCHEMA, "kind": self.kind, "ref": self.ref,
                "path": self.path, "sha256": self.sha256}

    def operational_dict(self) -> dict[str, str]:
        """Identity fields that alter execution, excluding alias labels and copy paths."""
        return {"kind": self.kind, "sha256": self.sha256}


@dataclass(frozen=True)
class ResolvedExecution:
    backend: str
    model: ArtifactIdentity | None
    build: ArtifactIdentity | None
    recipe: ArtifactIdentity | None
    drafter: ArtifactIdentity | None
    model_ref: str
    build_ref: str
    recipe_ref: str
    drafter_ref: str | None
    context: int
    concurrency: int
    speculation: str
    env: FrozenPairs
    metric: str
    metric_direction: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResolvedExecution":
        row = _mapping(value, "resolved execution")
        _keys(row, required={"schema", "backend", "model", "build", "recipe", "drafter",
                             "model_ref", "build_ref", "recipe_ref", "drafter_ref", "context",
                             "concurrency", "speculation", "env", "metric", "metric_direction"},
              optional=set(), label="resolved execution")
        if row["schema"] != EXECUTION_SCHEMA:
            raise ManifestError(f"resolved execution: unsupported schema {row['schema']!r}")
        backend = _text(row["backend"], "resolved execution.backend")
        if backend not in BACKENDS:
            raise ManifestError(f"resolved execution.backend must be one of {sorted(BACKENDS)}")
        direction = _text(row["metric_direction"], "resolved execution.metric_direction")
        if direction not in METRIC_DIRECTIONS:
            raise ManifestError("resolved execution.metric_direction must be 'higher' or 'lower'")
        drafter_ref = row["drafter_ref"]
        if drafter_ref is not None:
            drafter_ref = _text(drafter_ref, "resolved execution.drafter_ref")
        model = _artifact_from_optional(row["model"], "model")
        build = _artifact_from_optional(row["build"], "build")
        recipe = _artifact_from_optional(row["recipe"], "recipe")
        drafter = _artifact_from_optional(row["drafter"], "model")
        model_ref = _text(row["model_ref"], "resolved execution.model_ref")
        build_ref = _text(row["build_ref"], "resolved execution.build_ref")
        recipe_ref = _text(row["recipe_ref"], "resolved execution.recipe_ref")
        for label, artifact, ref in (("model", model, model_ref), ("build", build, build_ref),
                                     ("recipe", recipe, recipe_ref),
                                     ("drafter", drafter, drafter_ref)):
            if artifact is not None and artifact.ref != ref:
                raise ManifestError(f"resolved execution.{label} does not match its ref")
        return cls(backend=backend, model=model, build=build, recipe=recipe, drafter=drafter,
                   model_ref=model_ref, build_ref=build_ref, recipe_ref=recipe_ref,
                   drafter_ref=drafter_ref,
                   context=_positive_int(row["context"], "resolved execution.context"),
                   concurrency=_positive_int(row["concurrency"],
                                             "resolved execution.concurrency"),
                   speculation=_text(row["speculation"], "resolved execution.speculation"),
                   env=_env_pairs(row["env"], "resolved execution.env"),
                   metric=_text(row["metric"], "resolved execution.metric"),
                   metric_direction=direction)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": EXECUTION_SCHEMA, "backend": self.backend,
                "model": _artifact_dict(self.model), "build": _artifact_dict(self.build),
                "recipe": _artifact_dict(self.recipe), "drafter": _artifact_dict(self.drafter),
                "model_ref": self.model_ref, "build_ref": self.build_ref,
                "recipe_ref": self.recipe_ref, "drafter_ref": self.drafter_ref,
                "context": self.context, "concurrency": self.concurrency,
                "speculation": self.speculation, "env": _pairs_dict(self.env),
                "metric": self.metric, "metric_direction": self.metric_direction}

    def operational_dict(self) -> dict[str, Any]:
        """Canonical execution identity; refs/paths do not split byte-identical aliases."""
        return {"backend": self.backend,
                "model": _operational_artifact(self.model, self.model_ref),
                "build": _operational_artifact(self.build, self.build_ref),
                "recipe": _operational_artifact(self.recipe, self.recipe_ref),
                "context": self.context, "concurrency": self.concurrency,
                "speculation": self.speculation,
                "drafter": _operational_artifact(self.drafter, self.drafter_ref),
                "env": _pairs_dict(self.env), "metric": self.metric,
                "metric_direction": self.metric_direction}


@dataclass(frozen=True)
class TargetRevision:
    target_ids: tuple[str, ...]
    revision: int
    spec_digests: tuple[str, ...]
    status: str
    missing: tuple[str, ...]
    enrolled_as: tuple[str, ...]
    roles: tuple[str, ...]
    required_obligations: tuple[str, ...]
    seed_boost_units: int
    workload_signature: str
    execution: ResolvedExecution
    baseline_ref: str | None
    baseline: ArtifactIdentity | None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TargetRevision":
        row = _mapping(value, "target revision")
        _keys(row, required={"schema", "target_ids", "revision", "spec_digests", "status",
                             "missing", "enrolled_as", "roles", "required_obligations",
                             "seed_boost_units", "workload_signature", "execution", "baseline_ref",
                             "baseline"},
              optional=set(),
              label="target revision")
        if row["schema"] != TARGET_REVISION_SCHEMA:
            raise ManifestError(f"target revision: unsupported schema {row['schema']!r}")
        status = _text(row["status"], "target revision.status")
        if status not in TARGET_STATUSES:
            raise ManifestError(f"target revision.status must be one of {sorted(TARGET_STATUSES)}")
        signature = _sha(row["workload_signature"], "target revision.workload_signature")
        execution = ResolvedExecution.from_dict(row["execution"])
        baseline_ref = row["baseline_ref"]
        if baseline_ref is not None:
            baseline_ref = _text(baseline_ref, "target revision.baseline_ref")
        baseline = _artifact_from_optional(row["baseline"], "build")
        if baseline is not None and baseline_ref is not None and baseline.ref != baseline_ref:
            raise ManifestError("target revision.baseline does not match baseline_ref")
        if signature != _workload_signature(execution, baseline, baseline_ref):
            raise ManifestError("target revision.workload_signature does not match execution/baseline")
        enrolled_as = _strings(row["enrolled_as"], "target revision.enrolled_as", nonempty=True)
        if not set(enrolled_as) <= {"production", "seed"}:
            raise ManifestError("target revision.enrolled_as contains unknown provenance")
        seed_boost = _nonnegative_int(row["seed_boost_units"],
                                      "target revision.seed_boost_units")
        if seed_boost > 1:
            raise ManifestError("target revision.seed_boost_units must be 0 or 1")
        if seed_boost != int("seed" in enrolled_as):
            raise ManifestError(
                "target revision.seed_boost_units must match seed provenance")
        missing = _strings(row["missing"], "target revision.missing")
        if status == "ready":
            if missing:
                raise ManifestError("ready target revision cannot carry missing reasons")
            if any(item is None for item in (execution.model, execution.build,
                                             execution.recipe, baseline)):
                raise ManifestError("ready target revision needs model/build/recipe/baseline")
            if execution.drafter_ref is not None and execution.drafter is None:
                raise ManifestError("ready target revision needs its declared drafter")
        elif not missing:
            raise ManifestError("non-ready target revision needs an exact reason")
        return cls(target_ids=_strings(row["target_ids"], "target revision.target_ids", nonempty=True),
                   revision=_positive_int(row["revision"], "target revision.revision"),
                   spec_digests=tuple(_sha(item, "target revision.spec_digests[]")
                                      for item in _strings(row["spec_digests"],
                                                           "target revision.spec_digests", nonempty=True)),
                   status=status, missing=missing, enrolled_as=enrolled_as,
                   roles=_strings(row["roles"], "target revision.roles", nonempty=True),
                   required_obligations=_strings(row["required_obligations"],
                                                 "target revision.required_obligations"),
                   seed_boost_units=seed_boost, workload_signature=signature,
                   execution=execution, baseline_ref=baseline_ref, baseline=baseline)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": TARGET_REVISION_SCHEMA, "target_ids": list(self.target_ids),
                "revision": self.revision, "spec_digests": list(self.spec_digests),
                "status": self.status, "missing": list(self.missing),
                "enrolled_as": list(self.enrolled_as), "roles": list(self.roles),
                "required_obligations": list(self.required_obligations),
                "seed_boost_units": self.seed_boost_units,
                "workload_signature": self.workload_signature,
                "execution": self.execution.to_dict(),
                "baseline_ref": self.baseline_ref,
                "baseline": _artifact_dict(self.baseline)}


def _artifact_dict(value: ArtifactIdentity | None) -> dict[str, str] | None:
    return value.to_dict() if value is not None else None


def _operational_artifact(value: ArtifactIdentity | None,
                          unresolved_ref: str | None) -> dict[str, Any] | None:
    if value is not None:
        return value.operational_dict()
    if unresolved_ref is None:
        return None
    # Preserve which unresolved object was requested: two absent drafters are not aliases.
    return {"kind": "unresolved", "ref": unresolved_ref}


def _workload_signature(execution: ResolvedExecution,
                        baseline: ArtifactIdentity | None,
                        baseline_ref: str | None) -> str:
    return _digest({"execution": execution.operational_dict(),
                    "baseline": _operational_artifact(baseline, baseline_ref)})


def _artifact_from_optional(value: Any, kind: str) -> ArtifactIdentity | None:
    if value is None:
        return None
    row = _mapping(value, f"target revision.{kind}")
    return ArtifactIdentity.from_dict(row, kind=kind, ref=_text(row.get("ref"), "artifact.ref"))


def _sha(value: Any, label: str) -> str:
    digest = _text(value, label)
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ManifestError(f"{label} must be lowercase SHA-256")
    return digest


@dataclass(frozen=True)
class ResolvedCampaign:
    campaign_id: str
    request_id: str
    manifest_digest: str
    source_refs: FrozenPairs
    source_snapshot: tuple[tuple[str, ArtifactIdentity | None], ...]
    resources: ResourceRequest
    objective_ref: str
    actors: FrozenPairs
    fallbacks: tuple[tuple[str, tuple[str, ...]], ...]
    targets: tuple[TargetRevision, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResolvedCampaign":
        row = _mapping(value, "resolved campaign")
        _keys(row, required={"schema", "campaign_id", "request_id", "manifest_digest",
                             "source_refs", "source_snapshot", "resources", "objective_ref", "actors",
                             "fallbacks", "targets"}, optional=set(), label="resolved campaign")
        if row["schema"] != RESOLVED_SCHEMA:
            raise ManifestError(f"resolved campaign: unsupported schema {row['schema']!r}")
        actors = _pairs(row["actors"], "resolved campaign.actors")
        fallback_map = _mapping(row["fallbacks"], "resolved campaign.fallbacks")
        if set(fallback_map) != {key for key, _ in actors}:
            raise ManifestError("resolved campaign fallbacks must match actor roles")
        targets = _target_list(row["targets"], "resolved campaign.targets")
        parsed_targets = tuple(TargetRevision.from_dict(item) for item in targets)
        if not parsed_targets:
            raise ManifestError("resolved campaign.targets must not be empty")
        resources = ResourceRequest.from_dict(row["resources"])
        if (any(item.execution.backend in {"cpu", "both"} for item in parsed_targets)
                and not resources.cpu_logical):
            raise ManifestError("resolved CPU/both targets require CPU resources")
        if (any(item.execution.backend in {"gpu", "both"} for item in parsed_targets)
                and not resources.gpu_ids):
            raise ManifestError("resolved GPU/both targets require GPU resources")
        source_refs = _pairs(row["source_refs"], "resolved campaign.source_refs")
        if not source_refs:
            raise ManifestError("resolved campaign.source_refs must not be empty")
        source_row = _mapping(row["source_snapshot"], "resolved campaign.source_snapshot")
        if set(source_row) != {name for name, _ in source_refs}:
            raise ManifestError("resolved source snapshot must name exactly every source ref")
        source_snapshot = tuple(sorted(
            (name, _artifact_from_optional(source_row[name], "source"))
            for name, _ in source_refs))
        for name, ref in source_refs:
            artifact = dict(source_snapshot)[name]
            if artifact is not None and artifact.ref != ref:
                raise ManifestError(f"resolved source {name!r} does not match its frozen ref")
        if any(artifact is None for _, artifact in source_snapshot) and any(
                target.status == "ready" for target in parsed_targets):
            raise ManifestError("ready target cannot depend on an unresolved source snapshot")
        return cls(campaign_id=_text(row["campaign_id"], "resolved campaign.campaign_id"),
                   request_id=_text(row["request_id"], "resolved campaign.request_id"),
                   manifest_digest=_sha(row["manifest_digest"],
                                        "resolved campaign.manifest_digest"),
                   source_refs=source_refs, source_snapshot=source_snapshot,
                   resources=resources,
                   objective_ref=_text(row["objective_ref"],
                                       "resolved campaign.objective_ref"),
                   actors=actors,
                   fallbacks=tuple(sorted((key, _strings(items, f"fallbacks.{key}"))
                                          for key, items in fallback_map.items())),
                   targets=parsed_targets)

    def targets_for(self, backend: str) -> tuple[TargetRevision, ...]:
        """Return requested surfaces; ``both`` means the union of CPU and GPU targets."""
        if backend not in BACKENDS:
            raise ManifestError(f"backend must be one of {sorted(BACKENDS)}")
        if backend == "both":
            return self.targets
        return tuple(item for item in self.targets
                     if item.execution.backend in {backend, "both"})

    def to_dict(self) -> dict[str, Any]:
        return {"schema": RESOLVED_SCHEMA, "campaign_id": self.campaign_id,
                "request_id": self.request_id, "manifest_digest": self.manifest_digest,
                "source_refs": _pairs_dict(self.source_refs),
                "source_snapshot": {name: _artifact_dict(artifact)
                                    for name, artifact in self.source_snapshot},
                "resources": self.resources.to_dict(), "objective_ref": self.objective_ref,
                "actors": _pairs_dict(self.actors),
                "fallbacks": {key: list(values) for key, values in self.fallbacks},
                "targets": [item.to_dict() for item in self.targets]}


ArtifactResolver = Callable[[str, str, Mapping[str, Any]], Mapping[str, Any] | None]


def load_manifest(path: Path | str) -> CampaignManifest:
    """Load JSON or YAML by suffix; parsing never resolves or touches an artifact."""
    source = Path(path)
    try:
        if source.suffix.lower() == ".json":
            value = json.loads(source.read_text(encoding="utf-8"))
        elif source.suffix.lower() in {".yaml", ".yml"}:
            value = yaml.safe_load(source.read_text(encoding="utf-8"))
        else:
            raise ManifestError("manifest suffix must be .json, .yaml, or .yml")
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ManifestError(f"cannot load manifest {source}: {exc}") from exc
    return CampaignManifest.from_dict(value)


def _default_resolver(kind: str, ref: str,
                      registry: Mapping[str, Any]) -> Mapping[str, Any] | None:
    group = registry.get(kind, {})
    return group.get(ref) if isinstance(group, Mapping) else None


def _deep_freeze(value: Any) -> Any:
    """Copy caller state, then recursively remove every mutation surface."""
    if isinstance(value, Mapping):
        return MappingProxyType({copy.deepcopy(key): _deep_freeze(item)
                                 for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_deep_freeze(item) for item in value)
    return copy.deepcopy(value)


def _resolve(kind: str, ref: str | None, snapshot: Mapping[str, Any],
             resolver: ArtifactResolver) -> tuple[ArtifactIdentity | None, str | None]:
    if ref is None:
        return None, None
    value = resolver(kind, ref, snapshot)
    if value is None:
        return None, f"{kind}:missing"
    if isinstance(value, Mapping) and value.get("status") == "unsupported_capability":
        reason = _text(value.get("reason"), f"unsupported {kind} {ref}.reason")
        return None, f"{kind}:unsupported:{reason}"
    return ArtifactIdentity.from_dict(value, kind=kind, ref=ref), None


def _execution(spec: TargetSpec, *, model: ArtifactIdentity | None,
               build: ArtifactIdentity | None, recipe: ArtifactIdentity | None,
               drafter: ArtifactIdentity | None) -> ResolvedExecution:
    return ResolvedExecution(backend=spec.backend, model=model, build=build, recipe=recipe,
                             drafter=drafter, model_ref=spec.model_ref,
                             build_ref=spec.build_ref, recipe_ref=spec.recipe_ref,
                             drafter_ref=spec.drafter_ref, context=spec.context,
                             concurrency=spec.concurrency, speculation=spec.speculation,
                             env=spec.env, metric=spec.metric,
                             metric_direction=spec.metric_direction)


def resolve_manifest(manifest: CampaignManifest, *, registry_snapshot: Mapping[str, Any],
                     resolve_artifact: ArtifactResolver | None = None,
                     previous: ResolvedCampaign | None = None) -> ResolvedCampaign:
    """Resolve one frozen request without launching, downloading, building, or claiming.

    Reusing a request id with identical bytes returns the prior object.  Reusing it
    with different bytes is a conflict.  A later request may create a new immutable
    target revision, but an unchanged baseline reference remains pinned; movement is
    reported as ``mismatched_ref`` instead of silently updating the baseline.
    """
    if not isinstance(registry_snapshot, Mapping):
        raise ManifestError("registry_snapshot must be an immutable caller snapshot mapping")
    if previous is not None and previous.campaign_id != manifest.campaign_id:
        raise ManifestError("previous resolved campaign belongs to a different campaign")
    if previous is not None:
        if previous.request_id == manifest.request_id:
            if previous.manifest_digest != manifest.manifest_digest:
                raise ManifestError("conflicting manifest for an existing campaign/request id")
            return previous
    resolver = resolve_artifact or _default_resolver
    frozen_registry = _deep_freeze(registry_snapshot)
    resolved_cache: dict[tuple[str, str], tuple[ArtifactIdentity | None, str | None]] = {}

    def resolve_once(kind: str, ref: str | None) -> tuple[ArtifactIdentity | None, str | None]:
        if ref is None:
            return None, None
        key = (kind, ref)
        if key not in resolved_cache:
            resolved_cache[key] = _resolve(kind, ref, frozen_registry, resolver)
        return resolved_cache[key]

    resolved_sources = tuple((name, resolve_once("source", ref)[0])
                             for name, ref in manifest.source_snapshot)
    source_issues = tuple(f"source:{name}:{issue}"
                          for name, ref in manifest.source_snapshot
                          for _, issue in (resolve_once("source", ref),) if issue is not None)

    prior_by_id: dict[str, TargetRevision] = {}
    if previous is not None:
        for revision in previous.targets:
            for target_id in revision.target_ids:
                prior_by_id[target_id] = revision

    rows: list[tuple[TargetSpec, str, ArtifactIdentity | None, ArtifactIdentity | None,
                     ArtifactIdentity | None, ArtifactIdentity | None,
                     ArtifactIdentity | None, tuple[str, ...], str]] = []
    for enrolled_as, specs in (("production", manifest.production), ("seed", manifest.seeds)):
        for spec in specs:
            model, model_issue = resolve_once("model", spec.model_ref)
            build, build_issue = resolve_once("build", spec.build_ref)
            recipe, recipe_issue = resolve_once("recipe", spec.recipe_ref)
            drafter, drafter_issue = resolve_once("model", spec.drafter_ref)
            baseline_now, baseline_issue = resolve_once("build", spec.baseline_ref)
            issues = source_issues + tuple(issue for issue in (
                model_issue, build_issue, recipe_issue, drafter_issue, baseline_issue)
                                           if issue is not None)
            status = ("unsupported_capability" if any(":unsupported:" in item for item in issues)
                      else "missing_artifact" if issues else "ready")
            # A local seed without an explicit comparator is pinned to the exact build
            # resolved at enrollment; it never means "whatever this ref points at later".
            baseline = build if spec.baseline_ref is None else baseline_now
            prior = prior_by_id.get(spec.target_id)
            if prior is not None and prior.baseline is not None:
                if spec.baseline_ref is None and prior.baseline_ref is None:
                    baseline = prior.baseline
                    if build is not None and build != prior.baseline:
                        status = "mismatched_ref"
                        issues = tuple(sorted(set(
                            issues + ("implicit_baseline_build_moved",))))
                elif (spec.baseline_ref is not None
                      and prior.baseline.ref == spec.baseline_ref):
                    baseline = prior.baseline
                    if baseline_now is not None and baseline_now != prior.baseline:
                        status = "mismatched_ref"
                        issues = tuple(sorted(set(issues + ("baseline_ref_moved",))))
            rows.append((spec, enrolled_as, model, build, recipe, drafter, baseline,
                         issues, status))

    grouped: dict[str, list[tuple[Any, ...]]] = {}
    signature_by_target_id: dict[str, str] = {}
    for row in rows:
        spec, _, model, build, recipe, drafter, baseline, _, _ = row
        execution = _execution(spec, model=model, build=build, recipe=recipe, drafter=drafter)
        signature = _workload_signature(execution, baseline, spec.baseline_ref)
        old_signature = signature_by_target_id.setdefault(spec.target_id, signature)
        if old_signature != signature:
            raise ManifestError(f"target_id {spec.target_id!r} names conflicting workloads")
        grouped.setdefault(signature, []).append(row)

    revisions: list[TargetRevision] = []
    for signature, aliases in sorted(grouped.items()):
        specs = [row[0] for row in aliases]
        baseline_refs = {spec.baseline_ref for spec in specs}
        if len(baseline_refs) > 1:
            raise ManifestError(
                "content-identical target aliases declare different baseline refs; "
                "use one pinned comparator or distinct workloads")
        first = aliases[0]
        statuses = {row[8] for row in aliases}
        status = ("mismatched_ref" if "mismatched_ref" in statuses else
                  "unsupported_capability" if "unsupported_capability" in statuses else
                  "missing_artifact" if "missing_artifact" in statuses else "ready")
        target_ids = tuple(sorted({spec.target_id for spec in specs}))
        prior_revisions = {prior_by_id[target_id].revision for target_id in target_ids
                           if target_id in prior_by_id}
        revision = max(prior_revisions, default=0) + 1
        revisions.append(TargetRevision(
            target_ids=target_ids, revision=revision,
            spec_digests=tuple(sorted({spec.spec_digest for spec in specs})), status=status,
            missing=tuple(sorted({item for row in aliases for item in row[7]})),
            enrolled_as=tuple(sorted({row[1] for row in aliases})),
            roles=tuple(sorted({role for spec in specs for role in spec.roles})),
            required_obligations=tuple(sorted({ob for spec in specs
                                               for ob in spec.required_obligations})),
            seed_boost_units=1 if any(row[1] == "seed" for row in aliases) else 0,
            workload_signature=signature,
            execution=_execution(first[0], model=first[2], build=first[3],
                                 recipe=first[4], drafter=first[5]),
            baseline_ref=first[0].baseline_ref,
            baseline=first[6]))

    return ResolvedCampaign(campaign_id=manifest.campaign_id, request_id=manifest.request_id,
                            manifest_digest=manifest.manifest_digest,
                            source_refs=manifest.source_snapshot,
                            source_snapshot=resolved_sources,
                            resources=manifest.resources, objective_ref=manifest.objective_ref,
                            actors=manifest.actors, fallbacks=manifest.fallbacks,
                            targets=tuple(revisions))


__all__ = ["ARTIFACT_SCHEMA", "MANIFEST_SCHEMA", "RESOLVED_SCHEMA", "RESOURCE_SCHEMA",
           "TARGET_REVISION_SCHEMA", "TARGET_SCHEMA", "ArtifactIdentity", "CampaignManifest",
           "ManifestError", "ResolvedCampaign", "ResolvedExecution", "ResourceRequest", "TargetRevision",
           "TargetSpec", "load_manifest", "resolve_manifest"]
