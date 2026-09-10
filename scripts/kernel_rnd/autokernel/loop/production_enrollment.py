"""Offline consumer for canonical production-launcher enrollment exports.

The producer owns command construction.  This module validates and freezes its bytes,
projects declared artifacts into the existing Campaign resolver, and invokes the
existing ResolvedRecipe resolver only when the exported command is exactly expressible
by that instrument.  Unsupported launch shapes remain per-target diagnostics.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import stat
from typing import Any, Mapping

from . import campaign
from .resolved_recipe import ARTIFACT_SCHEMA as LAUNCH_ARTIFACT_SCHEMA
from .resolved_recipe import (EnvironmentPolicy, ResolutionError,
                              canonical_recipe_projection, resolve_canonical_launch)

EXPORT_SCHEMA = "autokernel-production-enrollment/v1"
CAMPAIGN_CONFIG_SCHEMA = "epyc.autokernel.production_campaign_config.v1"
DIAGNOSTICS_SCHEMA = "epyc.autokernel.production_enrollment_diagnostics.v1"
RECIPE_ARTIFACT_SCHEMA = "autokernel-production-launch-recipe/v1"
_RECIPE_FIELDS = (
    "backend", "port", "numa_instance", "command_argv", "environment",
    "environment_unsets", "workload", "topology", "runtime_requirements",
    "source_revisions", "source_revision_kinds", "speculation",
)


class ProductionEnrollmentError(campaign.ManifestError):
    pass


def _json_value(value: Any, label: str = "production enrollment") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProductionEnrollmentError(f"{label} contains non-finite numeric data")
        return
    if isinstance(value, list):
        for item in value:
            _json_value(item, f"{label}[]")
        return
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ProductionEnrollmentError(f"{label} contains non-string keys")
        for key, item in value.items():
            _json_value(item, f"{label}.{key}")
        return
    raise ProductionEnrollmentError(f"{label} contains non-JSON data")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


def _recipe_body(target: Mapping[str, Any]) -> dict[str, Any]:
    if any(field not in target for field in _RECIPE_FIELDS):
        raise ProductionEnrollmentError("production target recipe fields are incomplete")
    artifacts = [dict(item) for item in target.get("artifacts", [])
                 if isinstance(item, Mapping) and item.get("use") != "recipe"]
    return {"schema": RECIPE_ARTIFACT_SCHEMA,
            "launch": {field: target[field] for field in _RECIPE_FIELDS},
            "artifacts": sorted(artifacts,
                                key=lambda row: (str(row.get("use")), str(row.get("path"))))}


def _validate_recipe_artifact(target: Mapping[str, Any]) -> None:
    recipes = [item for item in target.get("artifacts", [])
               if isinstance(item, Mapping) and item.get("use") == "recipe"]
    if len(recipes) > 1:
        raise ProductionEnrollmentError("production target repeats its recipe artifact")
    if not recipes:
        return
    artifact = recipes[0]
    if set(artifact) != {"use", "path", "sha256"}:
        raise ProductionEnrollmentError("production recipe artifact is malformed")
    path = Path(artifact["path"]) if isinstance(artifact.get("path"), str) else Path()
    digest = artifact.get("sha256")
    if not path.is_absolute() or not isinstance(digest, str) or len(digest) != 64:
        raise ProductionEnrollmentError("production recipe artifact identity is malformed")
    try:
        info = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(info.st_mode):
            raise ProductionEnrollmentError("production recipe artifact is not a regular file")
        raw = path.read_bytes()
    except OSError as exc:
        raise ProductionEnrollmentError(f"cannot read production recipe artifact: {exc}") from exc
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ProductionEnrollmentError("production recipe artifact SHA-256 differs")
    try:
        parsed = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProductionEnrollmentError("production recipe artifact is not JSON") from exc
    if parsed != _recipe_body(target):
        raise ProductionEnrollmentError("production recipe artifact differs from exported launch")


def _local_seeds(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]],
                                                      dict[str, dict[str, Any]]]:
    raw = config.get("local_seeds")
    if raw is None:
        return [], {}
    if not isinstance(raw, Mapping) or set(raw) != {"targets", "artifacts"}:
        raise ProductionEnrollmentError("local_seeds must contain targets and artifacts")
    targets = raw["targets"]
    artifacts = raw["artifacts"]
    if not isinstance(targets, list) or not isinstance(artifacts, Mapping):
        raise ProductionEnrollmentError("local_seeds targets/artifacts are malformed")
    normalized_targets = [campaign.TargetSpec.from_dict(item).to_dict() for item in targets]
    if len({item["target_id"] for item in normalized_targets}) != len(normalized_targets):
        raise ProductionEnrollmentError("local seed target IDs must be unique")
    normalized_artifacts: dict[str, dict[str, Any]] = {}
    unknown = set(artifacts) - {"model", "build", "recipe"}
    if unknown:
        raise ProductionEnrollmentError(f"local seed artifacts have unknown kinds {sorted(unknown)}")
    for kind, group in artifacts.items():
        if not isinstance(group, Mapping):
            raise ProductionEnrollmentError(f"local seed {kind} artifacts must be an object")
        normalized_artifacts[kind] = {}
        for ref, identity in group.items():
            if not isinstance(ref, str) or not ref or ref.startswith("production:"):
                raise ProductionEnrollmentError(
                    "local seed artifact refs must be non-production non-empty strings")
            normalized_artifacts[kind][ref] = campaign.ArtifactIdentity.from_dict(
                identity, kind=kind, ref=ref).to_dict()
    return normalized_targets, normalized_artifacts


def merge_local_seed_registry(registry: Mapping[str, Mapping[str, Any]], *,
                              campaign_config: Mapping[str, Any]
                              ) -> dict[str, dict[str, Any]]:
    """Merge declared local seed pins without replacing canonical production pins."""
    _, additions = _local_seeds(campaign_config)
    merged = {kind: dict(group) for kind, group in registry.items()}
    for kind, group in additions.items():
        target = merged.setdefault(kind, {})
        overlap = set(target).intersection(group)
        if overlap:
            raise ProductionEnrollmentError(
                f"local seed artifacts collide with production refs: {sorted(overlap)}")
        target.update(group)
    return merged


def _validate_local_seed_production_bindings(
        targets: list[dict[str, Any]], export: Mapping[str, Any]) -> None:
    backends = {target["target_id"]: target.get("backend") for target in export["targets"]}
    for target in targets:
        for field, suffix in (("build_ref", ":executable"), ("recipe_ref", ":recipe")):
            ref = target[field]
            if not ref.startswith("production:"):
                continue
            if not ref.endswith(suffix):
                raise ProductionEnrollmentError(
                    f"local seed {field} has an unsupported production ref")
            source_target = ref[len("production:"):-len(suffix)]
            if backends.get(source_target) != target["backend"]:
                raise ProductionEnrollmentError(
                    f"local seed {field} backend differs from production target")


def load_export(value: Path | str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, (Path, str)):
        try:
            row = json.loads(Path(value).read_text(encoding="utf-8"),
                             parse_constant=lambda token: (_ for _ in ()).throw(
                                 ValueError(f"non-finite JSON token {token}")))
        except (OSError, ValueError) as exc:
            raise ProductionEnrollmentError(f"cannot read production enrollment: {exc}") from exc
    else:
        if not isinstance(value, Mapping):
            raise ProductionEnrollmentError("production enrollment must be an object")
        row = dict(value)
    _json_value(row)
    if set(row) != {"schema", "context", "targets", "disposition", "export_sha256"}:
        raise ProductionEnrollmentError("production enrollment has unknown or missing fields")
    if row["schema"] != EXPORT_SCHEMA or not isinstance(row["targets"], list):
        raise ProductionEnrollmentError("unsupported production enrollment schema")
    context = row["context"]
    if not isinstance(context, Mapping) or not isinstance(context.get("sources"), list):
        raise ProductionEnrollmentError("production enrollment context/sources are malformed")
    source_names: set[str] = set()
    for source in context["sources"]:
        if (not isinstance(source, Mapping)
                or not all(isinstance(source.get(key), str) and source.get(key)
                           for key in ("name", "path", "sha256", "revision"))
                or len(source.get("sha256", "")) != 64
                or any(char not in "0123456789abcdef" for char in source.get("sha256", ""))
                or source["name"] in source_names):
            raise ProductionEnrollmentError("production source pins are malformed or duplicate")
        source_names.add(source["name"])
    expected = row.pop("export_sha256")
    actual = _digest(row)
    row["export_sha256"] = expected
    if not isinstance(expected, str) or expected != actual:
        raise ProductionEnrollmentError("production enrollment integrity check failed")
    seen: set[str] = set()
    for target in row["targets"]:
        if not isinstance(target, Mapping):
            raise ProductionEnrollmentError("production target must be an object")
        target_id = target.get("target_id")
        if not isinstance(target_id, str) or not target_id or target_id in seen:
            raise ProductionEnrollmentError("production target ids must be unique strings")
        seen.add(target_id)
        if target.get("status") not in {"ready", "waiting_artifact", "unsupported"}:
            raise ProductionEnrollmentError(f"target {target_id} has invalid status")
        reasons = target.get("reasons", [])
        if (not isinstance(reasons, list)
                or not all(isinstance(reason, str) and reason for reason in reasons)):
            raise ProductionEnrollmentError(f"target {target_id} has malformed reasons")
        argv = target.get("argv", [])
        env = target.get("environment", {})
        if (not isinstance(argv, list) or not all(isinstance(x, str) and x for x in argv)
                or not isinstance(env, Mapping)
                or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items())):
            raise ProductionEnrollmentError(f"target {target_id} has malformed launch bytes")
        for field in ("aliases", "obligations"):
            values = target.get(field, [])
            if (not isinstance(values, list)
                    or not all(isinstance(item, str) and item for item in values)):
                raise ProductionEnrollmentError(f"target {target_id} has malformed {field}")
        artifacts = target.get("artifacts", [])
        if not isinstance(artifacts, list):
            raise ProductionEnrollmentError(f"target {target_id} has malformed artifacts")
        seen_artifacts: set[tuple[str, str]] = set()
        for artifact in artifacts:
            if (not isinstance(artifact, Mapping)
                    or set(artifact) != {"use", "path", "sha256"}
                    or artifact.get("use") not in {"model", "drafter", "executable",
                                                   "dso", "recipe"}
                    or not isinstance(artifact.get("path"), str)
                    or not Path(artifact["path"]).is_absolute()
                    or not isinstance(artifact.get("sha256"), str)
                    or len(artifact["sha256"]) != 64
                    or any(char not in "0123456789abcdef"
                           for char in artifact["sha256"])):
                raise ProductionEnrollmentError(
                    f"target {target_id} has malformed artifact identity")
            key = (artifact["use"], artifact["path"])
            if key in seen_artifacts:
                raise ProductionEnrollmentError(f"target {target_id} repeats an artifact")
            seen_artifacts.add(key)
        _validate_recipe_artifact(target)
    return json.loads(json.dumps(row))


def _campaign_projection_reason(target: Mapping[str, Any]) -> str | None:
    if target.get("backend") not in {"cpu", "gpu"}:
        return "backend_not_in_campaign_v1"
    workload = target.get("workload")
    if not isinstance(workload, Mapping):
        return "workload_missing"
    for field in ("context", "np"):
        raw = workload.get(field)
        if not isinstance(raw, str) or not raw.isdecimal() or int(raw) <= 0:
            return f"workload_{field}_invalid"
    primary = target.get("primary_role")
    if not isinstance(primary, str) or not primary:
        return "primary_role_missing"
    return None


def production_enrollment_diagnostics(
        value: Path | str | Mapping[str, Any]) -> dict[str, Any]:
    """Preserve every exported target while naming which rows entered Campaign v1."""
    export = load_export(value)
    rows = []
    for target in export["targets"]:
        projection_reason = _campaign_projection_reason(target)
        reasons = list(target.get("reasons", []))
        if projection_reason is not None:
            reasons.append(f"campaign_projection:{projection_reason}")
        rows.append({"target_id": target["target_id"], "status": target["status"],
                     "reasons": sorted(set(reasons)),
                     "enrolled_target_ids": ([] if projection_reason is not None
                                             else [target["target_id"]])})
    return {"schema": DIAGNOSTICS_SCHEMA, "export_sha256": export["export_sha256"],
            "targets": rows}


def _entry_artifacts(target: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Keep argv entry shards in singular launch slots; export retains all pins."""
    artifacts = target.get("artifacts", [])
    command = target.get("command_argv", [])
    entries = {}
    for use, flags in (("model", ("-m", "--model")),
                       ("drafter", ("-md", "--model-draft"))):
        if sum(item.get("use") == use for item in artifacts) > 1:
            entries[use] = next((command[i + 1] for i, token in enumerate(command[:-1])
                                 if token in flags), None)
    return [item for item in artifacts
            if item["use"] not in entries or item["path"] == entries[item["use"]]]


def registry_snapshot_from_export(value: Path | str | Mapping[str, Any]
                                  ) -> dict[str, dict[str, Any]]:
    """Project pins into Campaign's existing registry shape; grant no readiness."""
    export = load_export(value)
    registry: dict[str, dict[str, Any]] = {kind: {} for kind in
                                          ("source", "model", "build", "recipe")}
    for source in export["context"].get("sources", []):
        ref = f"production-source:{source['name']}:{source['revision']}"
        registry["source"][ref] = {
            "schema": campaign.ARTIFACT_SCHEMA, "kind": "source", "ref": ref,
            "path": source["path"], "sha256": source["sha256"]}
    for target in export["targets"]:
        target_id = target["target_id"]
        for artifact in _entry_artifacts(target):
            kind = {"model": "model", "drafter": "model",
                    "executable": "build", "recipe": "recipe"}.get(artifact.get("use"))
            if kind is None:
                continue
            ref = f"production:{target_id}:{artifact['use']}"
            registry[kind][ref] = {
                "schema": campaign.ARTIFACT_SCHEMA, "kind": kind, "ref": ref,
                "path": artifact["path"], "sha256": artifact["sha256"]}
        recipe_ref = f"production:{target_id}:recipe"
        # Sealing the launch bytes records identity, not readiness.  Preserve the
        # producer's per-target prerequisite disposition in the Campaign resolver:
        # an unsupported launch is a scoped capability refusal and a launch still
        # waiting on any artifact (including a DSO outside Campaign's four artifact
        # kinds) remains unresolved rather than becoming ready from its sidecar.
        if target["status"] == "unsupported":
            registry["recipe"][recipe_ref] = {
                "status": "unsupported_capability",
                "reason": "production_launcher_target_unsupported"}
        elif target["status"] == "waiting_artifact":
            registry["recipe"].pop(recipe_ref, None)
        elif recipe_ref not in registry["recipe"]:
            registry["recipe"][recipe_ref] = {
                "status": "unsupported_capability",
                "reason": "production_launcher_recipe_artifact_not_sealed"}
    return registry


def manifest_from_export(value: Path | str | Mapping[str, Any], *,
                         campaign_config: Mapping[str, Any]) -> campaign.CampaignManifest:
    """Derive the complete CPU/GPU target roster; retain operator-owned campaign inputs."""
    export = load_export(value)
    config = dict(campaign_config) if isinstance(campaign_config, Mapping) else {}
    required = {"schema", "campaign_id", "request_id", "resources", "objective_ref",
                "actors", "fallbacks", "metric", "metric_direction"}
    if (not required <= set(config) or not set(config) - required <= {"local_seeds"}
            or config.get("schema") != CAMPAIGN_CONFIG_SCHEMA):
        raise ProductionEnrollmentError("malformed production campaign configuration")
    local_targets, _ = _local_seeds(config)
    _validate_local_seed_production_bindings(local_targets, export)
    production = []
    seeds = []
    for target in export["targets"]:
        if _campaign_projection_reason(target) is not None:
            continue
        backend = target["backend"]
        workload = target.get("workload")
        assert isinstance(workload, Mapping)
        raw_context = workload.get("context")
        raw_np = workload.get("np")
        assert isinstance(raw_context, str) and isinstance(raw_np, str)
        context, concurrency = int(raw_context), int(raw_np)
        artifacts = {item.get("use"): item for item in target.get("artifacts", [])
                     if isinstance(item, Mapping)}
        target_id = target["target_id"]
        primary_role = target.get("primary_role")
        assert isinstance(primary_role, str) and primary_role
        row = {
            "schema": campaign.TARGET_SCHEMA, "request_id": config["request_id"],
            "target_id": target_id, "backend": backend,
            "model_ref": f"production:{target_id}:model",
            "build_ref": f"production:{target_id}:executable",
            "recipe_ref": f"production:{target_id}:recipe", "context": context,
            "concurrency": concurrency, "speculation": str(target.get("speculation", "none")),
            "env": {key: val for key, val in target.get("environment", {}).items()
                    if key != "LD_LIBRARY_PATH"},
            "metric": config["metric"], "metric_direction": config["metric_direction"],
            "roles": list(target.get("aliases", [])) + [primary_role],
            "required_obligations": list(target.get("obligations", [])),
        }
        if "drafter" in artifacts:
            row["drafter_ref"] = f"production:{target_id}:drafter"
        (seeds if target.get("optional_seed") is True else production).append(row)
    raw = {"schema": campaign.MANIFEST_SCHEMA, "campaign_id": config["campaign_id"],
           "request_id": config["request_id"],
           "source_snapshot": {source["name"]:
                               f"production-source:{source['name']}:{source['revision']}"
                               for source in export["context"].get("sources", [])},
           "resources": config["resources"], "objective_ref": config["objective_ref"],
           "actors": config["actors"], "fallbacks": config["fallbacks"],
           "production": production, "seeds": seeds + local_targets}
    return campaign.CampaignManifest.from_dict(raw)


def resolve_exported_recipes(value: Path | str | Mapping[str, Any], *,
                             environment_policy: EnvironmentPolicy | Mapping[str, Any]
                             ) -> dict[str, Any]:
    """Resolve exactly expressible rows; retain every other row independently."""
    export = load_export(value)
    rows = []
    for target in export["targets"]:
        result = {"target_id": target["target_id"], "source_argv": list(target.get("argv", [])),
                  "source_environment": dict(target.get("environment", {})),
                  "status": "unsupported", "reason": None, "resolved_recipe": None}
        if target["status"] != "ready":
            result["reason"] = f"source_{target['status']}"
            rows.append(result)
            continue
        # V1 deliberately refuses commands that cannot be reconstructed through Recipe.
        # The canonical launcher currently uses a richer flag grammar, so no string label
        # is promoted into an execution identity.
        try:
            command = target.get("command_argv")
            topology = target.get("topology")
            if not isinstance(command, list) or not isinstance(topology, Mapping):
                raise ResolutionError("canonical command/topology declaration is absent")
            prefix = topology.get("argv_prefix")
            recipe = canonical_recipe_projection(
                name=f"production:{target['target_id']}", command_argv=command,
                topology_prefix=prefix)
            artifacts = {"model": None, "drafter": None, "executable": None, "dsos": []}
            for item in _entry_artifacts(target):
                role = item["use"]
                if role == "dso":
                    artifacts["dsos"].append({"schema": LAUNCH_ARTIFACT_SCHEMA,
                                               "role": "dso", "path": item["path"],
                                               "sha256": item["sha256"]})
                elif role in artifacts:
                    artifacts[role] = {"schema": LAUNCH_ARTIFACT_SCHEMA,
                                       "role": role, "path": item["path"],
                                       "sha256": item["sha256"]}
            executable = next(item for item in target["artifacts"]
                              if item["use"] == "executable")
            build_dir = str(Path(executable["path"]).parent.parent)
            runtime = target.get("runtime_requirements")
            if not isinstance(runtime, Mapping):
                raise ResolutionError("runtime requirements are absent")
            provenance = {"export_sha256": export["export_sha256"],
                          "instance_mode": str(export["context"].get("instance_mode", "unknown"))}
            for source in export["context"].get("sources", []):
                provenance[f"source:{source['name']}"] = source["sha256"]
            resolved = resolve_canonical_launch(
                recipe, build_dir=build_dir, command_argv=command, topology_prefix=prefix,
                launch_environment=target["environment"], artifact_identities=artifacts,
                backend=target["backend"], environment_policy=environment_policy,
                port=target["port"], runtime_binary_dir=runtime.get("binary_dir"),
                runtime_ld_paths=runtime.get("ld_library_path") or (), provenance=provenance)
            if list(resolved.argv) != target["argv"] or dict(resolved.launch_env) != target["environment"]:
                raise ResolutionError("resolved launch bytes differ from production export")
            result.update(status="resolved", reason=None, resolved_recipe=resolved.to_dict())
        except (KeyError, StopIteration, TypeError, ValueError, ResolutionError) as exc:
            result["reason"] = f"not_exactly_resolvable:{type(exc).__name__}"
        rows.append(result)
    return {"schema": "epyc.autokernel.production_recipe_resolution.v1",
            "export_sha256": export["export_sha256"], "admission_ready": False,
            "targets": rows}


__all__ = ["CAMPAIGN_CONFIG_SCHEMA", "DIAGNOSTICS_SCHEMA", "EXPORT_SCHEMA",
           "ProductionEnrollmentError",
           "load_export", "manifest_from_export", "merge_local_seed_registry",
           "registry_snapshot_from_export",
           "production_enrollment_diagnostics", "resolve_exported_recipes"]
