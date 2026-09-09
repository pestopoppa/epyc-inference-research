"""Pure unified planning and scheduler handoff for enrolled CPU/GPU targets.

This module deliberately stops at a typed dispatch request.  A scheduler selection is
advice, not a grant, and neither the planner nor a supplied recorder may launch work.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from . import campaign, experiment_plan, scheduling, scoped_evidence, serving
from .resolved_recipe import (CanonicalResolvedRecipe, ResolvedRecipe, ResolutionError,
                              EnvironmentPolicy,
                              resolve_canonical_launch, resolve_recipe,
                              resolved_recipe_from_dict)

PROPOSAL_SCHEMA = "epyc.autokernel.unified_proposal.v1"
PROFILE_SCHEMA = "epyc.autokernel.target_profile.v1"
OPPORTUNITY_SCHEMA = "epyc.autokernel.planning_opportunity.v1"
DIMENSION_SCHEMA = "epyc.autokernel.runtime_dimension.v1"
PAIR_SCHEMA = "epyc.autokernel.runtime_arm_pair.v1"
ANCHOR_SCHEMA = "epyc.autokernel.runtime_anchor.v1"
DISPATCH_SCHEMA = "epyc.autokernel.planner_dispatch_request.v1"
INTENT_SCHEMA = "epyc.autokernel.experiment_intent.v1"
RESULT_SCHEMA = "epyc.autokernel.unified_planning_result.v1"

KINDS = frozenset({"source", "build_recipe", "runtime_recipe"})
PROFILE_STATES = frozenset({"fresh", "stale", "missing"})
OBSERVATION_STATES = frozenset({
    "accepted_unmeasured", "unknown", "inconclusive", "failed_correctness",
    "invalid_instrument", "bounded_null", "positive",
})
DIMENSIONS = frozenset({"threads", "cpu_list", "numa_policy", "env", "batch", "ubatch"})


class PlanningRefused(ValueError):
    pass


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise PlanningRefused(f"value is not finite canonical JSON: {exc}") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise PlanningRefused(f"{label} must be an object with string keys")
    _canonical(value)
    return value


def _exact(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    row = _mapping(value, label)
    if set(row) != fields:
        raise PlanningRefused(f"{label} fields differ")
    return row


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlanningRefused(f"{label} must be nonempty text")
    return value


def _texts(value: Any, label: str, *, nonempty: bool = False) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PlanningRefused(f"{label} must be an array")
    result = tuple(_text(item, f"{label}[]") for item in value)
    if nonempty and not result:
        raise PlanningRefused(f"{label} must not be empty")
    if len(result) != len(set(result)):
        raise PlanningRefused(f"{label} contains duplicates")
    return result


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or (positive and float(value) <= 0)):
        raise PlanningRefused(f"{label} must be a finite{' positive' if positive else ''} number")
    return float(value)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class RuntimeDimension:
    dimension_id: str
    kind: str
    anchor: Any
    candidate: Any
    authority_ref: str
    schema: str = DIMENSION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != DIMENSION_SCHEMA or self.kind not in DIMENSIONS:
            raise PlanningRefused("runtime dimension schema/kind is unsupported")
        object.__setattr__(self, "dimension_id", _text(self.dimension_id, "dimension_id"))
        object.__setattr__(self, "authority_ref", _text(self.authority_ref, "authority_ref"))
        _canonical(self.anchor)
        _canonical(self.candidate)
        if self.anchor == self.candidate:
            raise PlanningRefused("runtime dimension is an exact no-op")
        if self.kind in {"threads", "batch", "ubatch"}:
            for value in (self.anchor, self.candidate):
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise PlanningRefused(f"{self.kind} values must be positive integers")
        elif self.kind in {"cpu_list", "numa_policy"}:
            _text(self.anchor, "cpu_list.anchor")
            _text(self.candidate, "cpu_list.candidate")
            if self.kind == "numa_policy":
                for value in (self.anchor, self.candidate):
                    if not (value.startswith("--interleave=") or value.startswith("--membind=")):
                        raise PlanningRefused("NUMA policy is outside the canonical grammar")
        elif self.kind == "env":
            for label, value in (("anchor", self.anchor), ("candidate", self.candidate)):
                row = _exact(value, {"key", "value"}, f"env.{label}")
                _text(row["key"], f"env.{label}.key")
                if row["value"] is not None and not isinstance(row["value"], str):
                    raise PlanningRefused("env dimension values must be strings or null")
            if self.anchor["key"] != self.candidate["key"]:
                raise PlanningRefused("env dimension must change one exact key")
        object.__setattr__(self, "anchor", _freeze(self.anchor))
        object.__setattr__(self, "candidate", _freeze(self.candidate))

    @classmethod
    def from_dict(cls, value: Any) -> "RuntimeDimension":
        return cls(**dict(_exact(value, {"schema", "dimension_id", "kind", "anchor",
                                        "candidate", "authority_ref"}, "runtime dimension")))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "dimension_id": self.dimension_id, "kind": self.kind,
                "anchor": _thaw(self.anchor), "candidate": _thaw(self.candidate),
                "authority_ref": self.authority_ref}


def _artifacts(recipe: ResolvedRecipe | CanonicalResolvedRecipe) -> dict[str, Any]:
    return {"model": recipe.model.to_dict(), "drafter": (None if recipe.drafter is None
            else recipe.drafter.to_dict()), "executable": recipe.executable.to_dict(),
            "dsos": [item.to_dict() for item in recipe.dsos]}


def _replace_flag(argv: Sequence[str], flag: str, value: str) -> tuple[str, ...]:
    result = list(argv)
    positions = [index for index, token in enumerate(result) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(result):
        raise PlanningRefused(f"canonical command does not expose exactly one {flag}")
    result[positions[0] + 1] = value
    return tuple(result)


def _mutate_template(template: serving.Recipe, dimension: RuntimeDimension,
                     value: Any) -> serving.Recipe:
    if dimension.kind in {"threads", "batch", "ubatch"}:
        return replace(template, **{dimension.kind: value})
    if dimension.kind == "cpu_list":
        return replace(template, cpu_list=value)
    if dimension.kind == "numa_policy":
        return template
    # Canonical production recipes intentionally project argv into a neutral
    # template. Environment state is frozen separately under EnvironmentPolicy.
    assert dimension.kind == "env"
    return template


def _resolve_variant(anchor: ResolvedRecipe | CanonicalResolvedRecipe,
                     dimension: RuntimeDimension, value: Any
                     ) -> ResolvedRecipe | CanonicalResolvedRecipe:
    template = _mutate_template(anchor.template, dimension, value) if isinstance(
        anchor, CanonicalResolvedRecipe) else _mutate_template_from_v1(anchor, dimension, value)
    if isinstance(anchor, CanonicalResolvedRecipe):
        command = anchor.command_argv
        prefix = anchor.topology_prefix
        if dimension.kind in {"threads", "batch", "ubatch"}:
            flag = {"threads": "-t", "batch": "-b", "ubatch": "-ub"}[dimension.kind]
            command = _replace_flag(command, flag, str(value))
            # The accepted canonical grammar binds background threads to serving
            # threads; changing only -t would manufacture an invalid mixed setting.
            if dimension.kind == "threads" and "-tb" in command:
                command = _replace_flag(command, "-tb", str(value))
        elif dimension.kind == "cpu_list":
            if len(prefix) >= 3 and prefix[-3:-1] == ("taskset", "-c"):
                prefix = prefix[:-1] + (value,)
            else:
                raise PlanningRefused("CPU-list sweep requires canonical taskset topology")
        elif dimension.kind == "numa_policy":
            if len(prefix) >= 6 and prefix[0] == "numactl" and prefix[2] == "--":
                prefix = ("numactl", value, "--") + prefix[3:]
            else:
                raise PlanningRefused("NUMA-policy sweep requires canonical numactl topology")
        launch = dict(anchor.launch_env)
        if dimension.kind == "env":
            if value["value"] is None:
                launch.pop(value["key"], None)
            else:
                launch[value["key"]] = value["value"]
        return resolve_canonical_launch(
            template, build_dir=anchor.build_dir, command_argv=command,
            topology_prefix=prefix, launch_environment=launch,
            artifact_identities=_artifacts(anchor), backend=anchor.backend,
            environment_policy=anchor.environment_policy, port=anchor.port,
            runtime_binary_dir=anchor.runtime_binary_dir,
            runtime_ld_paths=anchor.runtime_ld_paths, provenance=dict(anchor.provenance))
    return resolve_recipe(template, build_dir=anchor.build_dir,
                          artifact_identities=_artifacts(anchor), backend=anchor.backend,
                          environment_policy=anchor.environment_policy,
                          inherited_environment=dict(anchor.launch_env), port=anchor.port)


def _mutate_template_from_v1(anchor: ResolvedRecipe, dimension: RuntimeDimension,
                             value: Any) -> serving.Recipe:
    # v1 embeds only a hash, so recover the exact semantic template from its argv is
    # intentionally unsupported.  This prevents a hash label becoming recipe authority.
    raise PlanningRefused("legacy resolved recipe lacks a reconstructable frozen template")


@dataclass(frozen=True)
class RuntimeArmPair:
    dimension: RuntimeDimension
    anchor: CanonicalResolvedRecipe
    candidate: CanonicalResolvedRecipe
    schema: str = PAIR_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PAIR_SCHEMA or not isinstance(self.dimension, RuntimeDimension):
            raise PlanningRefused("runtime arm pair schema/dimension is unsupported")
        self.validate()

    @classmethod
    def from_dict(cls, value: Any) -> "RuntimeArmPair":
        row = _exact(value, {"schema", "dimension", "anchor", "candidate"},
                     "runtime arm pair")
        if row["schema"] != PAIR_SCHEMA:
            raise PlanningRefused("runtime arm pair schema is unsupported")
        dimension = RuntimeDimension.from_dict(row["dimension"])
        try:
            anchor = resolved_recipe_from_dict(row["anchor"])
            candidate = resolved_recipe_from_dict(row["candidate"])
        except ResolutionError as exc:
            raise PlanningRefused(f"runtime arm recipe is invalid: {exc}") from exc
        if not isinstance(anchor, CanonicalResolvedRecipe) or not isinstance(
                candidate, CanonicalResolvedRecipe):
            raise PlanningRefused("runtime sweep currently requires canonical launch recipes")
        result = cls(dimension, anchor, candidate)
        result.validate()
        return result

    def validate(self) -> None:
        for recipe in (self.anchor, self.candidate):
            try:
                recipe.validate_launch(recipe.template, recipe.build_dir, recipe.port)
            except Exception as exc:
                raise PlanningRefused(f"runtime arm is not launch-valid: {exc}") from exc
        for name in ("backend", "build_dir", "port"):
            if getattr(self.anchor, name) != getattr(self.candidate, name):
                raise PlanningRefused(f"runtime arms differ in sealed {name}")
        if _artifacts(self.anchor) != _artifacts(self.candidate):
            raise PlanningRefused("runtime arms must share exact model/executable/DSOs")
        expected_anchor = _resolve_variant(self.anchor, self.dimension, self.dimension.anchor)
        expected_candidate = _resolve_variant(self.anchor, self.dimension, self.dimension.candidate)
        if expected_anchor.to_dict() != self.anchor.to_dict() \
                or expected_candidate.to_dict() != self.candidate.to_dict():
            raise PlanningRefused("runtime arm pair differs beyond its registered dimension")
        if self.anchor.execution_digest == self.candidate.execution_digest:
            raise PlanningRefused("runtime arm pair is an execution no-op")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "dimension": self.dimension.to_dict(),
                "anchor": self.anchor.to_dict(), "candidate": self.candidate.to_dict()}


@dataclass(frozen=True)
class RuntimeAnchor:
    target_revision_digest: str
    target_id: str
    production_export: Mapping[str, Any]
    environment_policy: EnvironmentPolicy
    schema: str = ANCHOR_SCHEMA

    @classmethod
    def from_dict(cls, value: Any) -> "RuntimeAnchor":
        row = _exact(value, {"schema", "target_revision_digest", "target_id",
                             "production_export", "environment_policy"}, "runtime anchor")
        if row["schema"] != ANCHOR_SCHEMA:
            raise PlanningRefused("runtime anchor schema is unsupported")
        digest = _text(row["target_revision_digest"], "target_revision_digest")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise PlanningRefused("target_revision_digest must be SHA-256")
        try:
            export = _mapping(row["production_export"], "runtime anchor production_export")
            policy = EnvironmentPolicy.from_dict(row["environment_policy"])
        except (ValueError, TypeError, ResolutionError) as exc:
            raise PlanningRefused(f"runtime anchor artifact/policy is invalid: {exc}") from exc
        target_id = _text(row["target_id"], "runtime anchor target_id")
        return cls(digest, target_id, _freeze(export), policy)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema,
                "target_revision_digest": self.target_revision_digest,
                "target_id": self.target_id,
                "production_export": _thaw(self.production_export),
                "environment_policy": self.environment_policy.to_dict()}


def _bind_runtime_anchor(target: campaign.TargetRevision, anchor: RuntimeAnchor,
                         export: Mapping[str, Any],
                         resolved_rows: Mapping[str, CanonicalResolvedRecipe]
                         ) -> CanonicalResolvedRecipe:
    execution = target.execution
    expected_digest = _target_digest(target)
    if anchor.target_revision_digest != expected_digest:
        raise PlanningRefused("runtime anchor targets a different revision")
    if any(item is None for item in (execution.model, execution.build, execution.recipe)):
        raise PlanningRefused("runtime anchor target lacks enrolled artifacts")
    if not (execution.build.ref.startswith("production:")
            and execution.build.ref.endswith(":executable")):
        raise PlanningRefused(
            "runtime sweep cannot interpret a generic build pin as executable bytes")
    try:
        target_row = next(item for item in export["targets"]
                          if item["target_id"] == anchor.target_id)
        uses = [item["use"] for item in target_row["artifacts"]]
        if (uses.count("model") != 1 or uses.count("executable") != 1
                or uses.count("recipe") != 1 or uses.count("drafter") > 1
                or len([item for item in target_row["artifacts"] if item["use"] == "dso"])
                   != len({item["path"] for item in target_row["artifacts"]
                           if item["use"] == "dso"})):
            raise PlanningRefused("runtime export artifact roles are missing or duplicated")
        recipe_artifact = next(item for item in target_row["artifacts"]
                               if item["use"] == "recipe")
        if recipe_artifact["sha256"] != execution.recipe.sha256:
            raise PlanningRefused(
                "validated export recipe bytes differ from enrolled recipe identity")
        recipe = resolved_rows[anchor.target_id]
    except (KeyError, TypeError, ValueError, ResolutionError) as exc:
        raise PlanningRefused(f"runtime recipe artifact cannot be rederived: {exc}") from exc
    checks = {
        "backend": recipe.backend == execution.backend,
        "model": recipe.model.sha256 == execution.model.sha256,
        "executable": recipe.executable.sha256 == execution.build.sha256,
        "drafter_presence": (recipe.drafter is None) == (execution.drafter is None),
        "drafter": ((recipe.drafter is None and execution.drafter is None) or
                    (recipe.drafter is not None and execution.drafter is not None
                     and recipe.drafter.sha256 == execution.drafter.sha256)),
        "context": recipe.template.ctx == execution.context,
        "concurrency": recipe.template.np == execution.concurrency,
        "speculation": recipe.template.spec_decode.get("type", "none")
                       == execution.speculation,
        "environment": ({key: value for key, value in recipe.launch_env
                         if key != "LD_LIBRARY_PATH"} == dict(execution.env)),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise PlanningRefused(
            f"runtime anchor differs from enrolled execution identity: {failed}")
    return recipe


_PREPARED_TOKEN = object()


@dataclass(frozen=True)
class PreparedRuntimeAnchors:
    campaign_id: str
    manifest_digest: str
    resolved_campaign_digest: str
    recipes: Mapping[str, CanonicalResolvedRecipe]
    export_digests: tuple[str, ...]
    _token: object

    def __post_init__(self) -> None:
        if self._token is not _PREPARED_TOKEN:
            raise PlanningRefused(
                "prepared runtime anchors must come from prepare_runtime_anchors")


def prepare_runtime_anchors(
        resolved_campaign: campaign.ResolvedCampaign,
        anchors: Mapping[str, Mapping[str, Any] | RuntimeAnchor],
        ) -> PreparedRuntimeAnchors:
    """Validate each unique export/policy once, outside the planning hot path."""
    if not isinstance(resolved_campaign, campaign.ResolvedCampaign):
        raise PlanningRefused("resolved_campaign must be validated before anchor preparation")
    try:
        resolved_campaign = campaign.ResolvedCampaign.from_dict(
            resolved_campaign.to_dict())
    except (campaign.ManifestError, TypeError, ValueError) as exc:
        raise PlanningRefused(f"resolved campaign is invalid: {exc}") from exc
    targets = {_target_digest(target): target for target in resolved_campaign.targets}
    normalized: dict[str, RuntimeAnchor] = {}
    groups: dict[tuple[str, str], list[RuntimeAnchor]] = {}
    for key, raw in anchors.items():
        if key not in targets:
            raise PlanningRefused("runtime anchor names an unknown target revision")
        anchor = RuntimeAnchor.from_dict(raw.to_dict() if isinstance(raw, RuntimeAnchor) else raw)
        if anchor.target_revision_digest != key:
            raise PlanningRefused("runtime anchor map key differs from its target revision")
        export_label = anchor.production_export.get("export_sha256")
        if not isinstance(export_label, str):
            raise PlanningRefused("runtime anchor export lacks an identity")
        group = (export_label, _digest(anchor.environment_policy.to_dict()))
        groups.setdefault(group, []).append(anchor)
        normalized[key] = anchor
    prepared: dict[str, CanonicalResolvedRecipe] = {}
    verified_exports: dict[tuple[str, str], Mapping[str, Any]] = {}
    for group, members in groups.items():
        from .production_enrollment import (ProductionEnrollmentError,
                                            resolve_exported_recipes)
        first = members[0]
        # Work from one detached immutable snapshot.  The resolver performs the
        # complete export and sidecar validation; calling load_export separately
        # would validate and read every sidecar twice at this boundary.
        export = json.loads(_canonical(_thaw(first.production_export)))
        # Same claimed group must be byte-identical, not merely share a digest label.
        for member in members[1:]:
            if _thaw(member.production_export) != export:
                raise PlanningRefused("runtime anchors conflict for one export identity")
        try:
            resolved = resolve_exported_recipes(
                export, environment_policy=first.environment_policy)
        except (ProductionEnrollmentError, ResolutionError, ValueError, TypeError) as exc:
            raise PlanningRefused(
                f"runtime production export cannot be prepared: {exc}") from exc
        rows: dict[str, CanonicalResolvedRecipe] = {}
        for row in resolved["targets"]:
            if row["status"] != "resolved":
                continue
            recipe = resolved_recipe_from_dict(row["resolved_recipe"])
            if not isinstance(recipe, CanonicalResolvedRecipe):
                raise PlanningRefused("production resolver returned non-canonical recipe")
            rows[row["target_id"]] = recipe
        for member in members:
            if member.target_id not in rows:
                raise PlanningRefused("runtime anchor target is not canonically resolvable")
        verified_exports[group] = export
        for key, member in normalized.items():
            if member in members:
                prepared[key] = _bind_runtime_anchor(
                    targets[key], member, export, rows)
    return PreparedRuntimeAnchors(
        resolved_campaign.campaign_id, resolved_campaign.manifest_digest,
        _digest(resolved_campaign.to_dict()), MappingProxyType(prepared),
        tuple(sorted(key[0] for key in verified_exports)), _PREPARED_TOKEN)


def enumerate_runtime_dimensions(anchor: Mapping[str, Any] | CanonicalResolvedRecipe,
                                 dimensions: Sequence[Mapping[str, Any] | RuntimeDimension]
                                 ) -> tuple[RuntimeArmPair, ...]:
    """Concrete deterministic adapter over accepted canonical recipe validators."""
    try:
        base = (resolved_recipe_from_dict(anchor) if isinstance(anchor, Mapping) else
                resolved_recipe_from_dict(anchor.to_dict()))
    except ResolutionError as exc:
        raise PlanningRefused(f"runtime anchor is invalid: {exc}") from exc
    if not isinstance(base, CanonicalResolvedRecipe):
        raise PlanningRefused("runtime enumeration requires canonical launch v1")
    normalized = tuple(item if isinstance(item, RuntimeDimension)
                       else RuntimeDimension.from_dict(item) for item in dimensions)
    ids = [item.dimension_id for item in normalized]
    if len(ids) != len(set(ids)):
        raise PlanningRefused("runtime dimensions repeat an id")
    pairs = []
    for dimension in sorted(normalized, key=lambda item: item.dimension_id):
        try:
            left = _resolve_variant(base, dimension, dimension.anchor)
            right = _resolve_variant(base, dimension, dimension.candidate)
            pair = RuntimeArmPair(dimension, left, right)  # type: ignore[arg-type]
            pair.validate()
        except (ResolutionError, serving.RecipeError, PlanningRefused) as exc:
            raise PlanningRefused(
                f"runtime dimension {dimension.dimension_id} unsupported: {exc}") from exc
        pairs.append(pair)
    return tuple(pairs)


@dataclass(frozen=True)
class Opportunity:
    opportunity_id: str
    kind: str
    mechanism_id: str
    estimand: str
    metric: str
    metric_direction: str
    effect_question: Mapping[str, Any]
    changed_factors: tuple[str, ...]
    instrument: str
    unit: str
    required_witnesses: tuple[str, ...]
    stage_class: str
    estimated_duration_seconds: float
    runtime_dimension_ids: tuple[str, ...]
    claim_key: scoped_evidence.ClaimKey
    schema: str = OPPORTUNITY_SCHEMA

    @classmethod
    def from_dict(cls, value: Any) -> "Opportunity":
        row = dict(_exact(value, {"schema", "opportunity_id", "kind", "mechanism_id",
            "estimand", "metric", "metric_direction", "effect_question", "changed_factors",
            "instrument", "unit", "required_witnesses", "stage_class",
            "estimated_duration_seconds", "runtime_dimension_ids", "claim_key"}, "opportunity"))
        if row.pop("schema") != OPPORTUNITY_SCHEMA or row["kind"] not in KINDS:
            raise PlanningRefused("opportunity schema/kind is unsupported")
        for field in ("opportunity_id", "mechanism_id", "estimand", "metric", "instrument", "unit"):
            row[field] = _text(row[field], field)
        if row["metric_direction"] not in {"higher", "lower"}:
            raise PlanningRefused("metric_direction is unsupported")
        row["effect_question"] = _freeze(_mapping(row["effect_question"], "effect_question"))
        row["changed_factors"] = _texts(row["changed_factors"], "changed_factors", nonempty=True)
        row["required_witnesses"] = _texts(row["required_witnesses"], "required_witnesses", nonempty=True)
        row["runtime_dimension_ids"] = _texts(row["runtime_dimension_ids"], "runtime_dimension_ids")
        row["estimated_duration_seconds"] = _number(row["estimated_duration_seconds"], "duration", positive=True)
        if row["stage_class"] not in scheduling.STAGE_CLASSES:
            raise PlanningRefused("stage_class is unsupported")
        row["claim_key"] = scoped_evidence.ClaimKey.from_dict(row["claim_key"])
        if row["kind"] == "runtime_recipe":
            if len(row["changed_factors"]) != 1 or len(row["runtime_dimension_ids"]) != 1:
                raise PlanningRefused("runtime A2 opportunity requires one exact factor/dimension")
        elif row["runtime_dimension_ids"]:
            raise PlanningRefused("source/build opportunity cannot carry runtime dimensions")
        return cls(**row)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "opportunity_id": self.opportunity_id,
                "kind": self.kind, "mechanism_id": self.mechanism_id,
                "estimand": self.estimand, "metric": self.metric,
                "metric_direction": self.metric_direction,
                "effect_question": _thaw(self.effect_question),
                "changed_factors": list(self.changed_factors), "instrument": self.instrument,
                "unit": self.unit, "required_witnesses": list(self.required_witnesses),
                "stage_class": self.stage_class,
                "estimated_duration_seconds": self.estimated_duration_seconds,
                "runtime_dimension_ids": list(self.runtime_dimension_ids),
                "claim_key": self.claim_key.to_dict()}


@dataclass(frozen=True)
class TargetProfile:
    target_revision_digest: str
    freshness: str
    quant: str
    hotspots: tuple[str, ...]
    observation_states: tuple[str, ...]
    kept_scope: tuple[str, ...]
    resource_cost: scheduling.ResourceVector
    opportunities: tuple[Opportunity, ...]
    schema: str = PROFILE_SCHEMA

    @classmethod
    def from_dict(cls, value: Any) -> "TargetProfile":
        row = dict(_exact(value, {"schema", "target_revision_digest", "freshness", "quant",
            "hotspots", "observation_states", "kept_scope", "resource_cost", "opportunities"},
            "target profile"))
        if row.pop("schema") != PROFILE_SCHEMA:
            raise PlanningRefused("target profile schema is unsupported")
        digest = _text(row["target_revision_digest"], "target_revision_digest")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise PlanningRefused("target_revision_digest must be SHA-256")
        row["target_revision_digest"] = digest
        if row["freshness"] not in PROFILE_STATES:
            raise PlanningRefused("profile freshness is unsupported")
        row["quant"] = _text(row["quant"], "quant")
        row["hotspots"] = _texts(row["hotspots"], "hotspots")
        states = _texts(row["observation_states"], "observation_states")
        if not set(states) <= OBSERVATION_STATES:
            raise PlanningRefused("profile observation state is unsupported")
        row["observation_states"] = states
        row["kept_scope"] = _texts(row["kept_scope"], "kept_scope")
        row["resource_cost"] = scheduling.ResourceVector.from_dict(row["resource_cost"])
        if not isinstance(row["opportunities"], (list, tuple)):
            raise PlanningRefused("opportunities must be an array")
        row["opportunities"] = tuple(Opportunity.from_dict(item) for item in row["opportunities"])
        return cls(**row)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "target_revision_digest": self.target_revision_digest,
                "freshness": self.freshness, "quant": self.quant,
                "hotspots": list(self.hotspots), "observation_states": list(self.observation_states),
                "kept_scope": list(self.kept_scope), "resource_cost": self.resource_cost.to_dict(),
                "opportunities": [item.to_dict() for item in self.opportunities]}


@dataclass(frozen=True)
class UnifiedProposal:
    proposal_id: str
    target_revision_digest: str
    backend: str
    parent_identity: Mapping[str, Any]
    control_identity: Mapping[str, Any]
    intervention_identity: Mapping[str, Any]
    kind: str
    mechanism_id: str
    estimand: str
    metric: str
    metric_direction: str
    effect_question: Mapping[str, Any]
    changed_factors: tuple[str, ...]
    instrument: str
    unit: str
    required_witnesses: tuple[str, ...]
    stage_class: str
    estimated_duration_seconds: float
    experiment_plan_digest: str | None
    native_artifact_sink_ref: str
    runtime_pair: Mapping[str, Any] | None
    claim_key: scoped_evidence.ClaimKey
    evidence_snapshot: Mapping[str, Any]
    schema: str = PROPOSAL_SCHEMA

    @classmethod
    def from_dict(cls, value: Any) -> "UnifiedProposal":
        fields = {"schema", "proposal_id", "target_revision_digest", "backend",
            "parent_identity", "control_identity", "intervention_identity", "kind",
            "mechanism_id", "estimand", "metric", "metric_direction", "effect_question",
            "changed_factors", "instrument", "unit", "required_witnesses", "stage_class",
            "estimated_duration_seconds", "experiment_plan_digest", "native_artifact_sink_ref",
            "runtime_pair", "claim_key", "evidence_snapshot"}
        row = dict(_exact(value, fields, "unified proposal"))
        if row.pop("schema") != PROPOSAL_SCHEMA or row["kind"] not in KINDS:
            raise PlanningRefused("unified proposal schema/kind is unsupported")
        for field in ("proposal_id", "target_revision_digest", "mechanism_id", "estimand",
                      "metric", "instrument", "unit",
                      "native_artifact_sink_ref"):
            row[field] = _text(row[field], field)
        plan_digest = row["experiment_plan_digest"]
        if plan_digest is not None:
            plan_digest = _text(plan_digest, "experiment_plan_digest")
            if len(plan_digest) != 64 or any(c not in "0123456789abcdef" for c in plan_digest):
                raise PlanningRefused("experiment_plan_digest must be SHA-256 or null")
        row["experiment_plan_digest"] = plan_digest
        if row["backend"] not in {"cpu", "gpu"} or row["metric_direction"] not in {"higher", "lower"}:
            raise PlanningRefused("proposal backend/direction is unsupported")
        for field in ("parent_identity", "control_identity", "intervention_identity",
                      "effect_question", "evidence_snapshot"):
            row[field] = _freeze(_mapping(row[field], field))
        row["claim_key"] = scoped_evidence.ClaimKey.from_dict(row["claim_key"])
        if row["kind"] == "runtime_recipe":
            pair = RuntimeArmPair.from_dict(row["runtime_pair"])
            row["runtime_pair"] = _freeze(pair.to_dict())
            if ({"execution_digest": pair.anchor.execution_digest}
                    != _thaw(row["control_identity"]) or
                    {"execution_digest": pair.candidate.execution_digest}
                    != _thaw(row["intervention_identity"])):
                raise PlanningRefused("runtime pair differs from proposal arm identities")
        elif row["runtime_pair"] is not None:
            raise PlanningRefused("source/build proposal cannot carry a runtime pair")
        if (row["claim_key"].estimand != row["estimand"]
                or row["claim_key"].metric != row["metric"]
                or row["claim_key"].metric_direction != row["metric_direction"]
                or _thaw(row["claim_key"].effect_question) != _thaw(row["effect_question"])):
            raise PlanningRefused("proposal semantic fields differ from exact ClaimKey")
        if (_thaw(row["claim_key"].control_identity) != _thaw(row["control_identity"])
                or _thaw(row["claim_key"].intervention_identity)
                   != _thaw(row["intervention_identity"])):
            raise PlanningRefused("proposal arm identities differ from exact ClaimKey")
        row["changed_factors"] = _texts(row["changed_factors"], "changed_factors", nonempty=True)
        row["required_witnesses"] = _texts(row["required_witnesses"], "required_witnesses", nonempty=True)
        row["estimated_duration_seconds"] = _number(row["estimated_duration_seconds"], "duration", positive=True)
        if row["stage_class"] not in scheduling.STAGE_CLASSES:
            raise PlanningRefused("proposal stage_class is unsupported")
        if row["kind"] == "runtime_recipe" and len(row["changed_factors"]) != 1:
            raise PlanningRefused("runtime A2 proposal must change exactly one factor")
        return cls(**row)

    def to_dict(self) -> dict[str, Any]:
        result = {field: (_thaw(getattr(self, field))) for field in (
            "schema", "proposal_id", "target_revision_digest", "backend", "parent_identity",
            "control_identity", "intervention_identity", "kind", "mechanism_id", "estimand",
            "metric", "metric_direction", "effect_question", "changed_factors", "instrument",
            "unit", "required_witnesses", "stage_class", "estimated_duration_seconds",
            "experiment_plan_digest", "native_artifact_sink_ref", "runtime_pair",
            "evidence_snapshot")}
        result["claim_key"] = self.claim_key.to_dict()
        return result

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())


@dataclass(frozen=True)
class DispatchRequest:
    selection: Mapping[str, Any]
    proposal: Mapping[str, Any]
    experiment_intent: Mapping[str, Any]
    execution_authorized: bool = False
    schema: str = DISPATCH_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != DISPATCH_SCHEMA or self.execution_authorized is not False:
            raise PlanningRefused("dispatch request cannot grant execution authority")
        selection = scheduling.Selection.from_dict(_thaw(self.selection))
        proposal = UnifiedProposal.from_dict(_thaw(self.proposal))
        intent = _exact(_thaw(self.experiment_intent), {
            "schema", "status", "proposal_digest", "stage_proposal_digest",
            "experiment_plan_digest", "claim_key",
            "effect_question", "arm_scalars_are_gain_evidence"}, "experiment intent")
        if (intent["schema"] != INTENT_SCHEMA
                or intent["status"] != ("ready_comparison" if
                    proposal.experiment_plan_digest is not None else "pending_plan_preparation")
                or intent["proposal_digest"] != proposal.digest
                or selection.proposal is None
                or intent["stage_proposal_digest"] != selection.proposal.digest
                or intent["experiment_plan_digest"] != proposal.experiment_plan_digest
                or intent["claim_key"] != proposal.claim_key.to_dict()
                or intent["effect_question"] != _thaw(proposal.effect_question)
                or intent["arm_scalars_are_gain_evidence"] is not False
                or selection.proposal.proposal_id != proposal.proposal_id
                or selection.proposal.target_revision != proposal.target_revision_digest
                or selection.proposal.backend != proposal.backend
                or selection.proposal.stage_class != (
                    "prerequisite" if proposal.experiment_plan_digest is None
                    else proposal.stage_class)
                or selection.proposal.estimated_duration_seconds
                   != proposal.estimated_duration_seconds):
            raise PlanningRefused("dispatch experiment intent differs from selected proposal")
        object.__setattr__(self, "selection", _freeze(selection.to_dict()))
        object.__setattr__(self, "proposal", _freeze(proposal.to_dict()))
        object.__setattr__(self, "experiment_intent", _freeze(dict(intent)))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "selection": _thaw(self.selection),
                "proposal": _thaw(self.proposal),
                "experiment_intent": _thaw(self.experiment_intent),
                "execution_authorized": self.execution_authorized}


def _target_digest(target: campaign.TargetRevision) -> str:
    return _digest(target.to_dict())


def _prompt(target: campaign.TargetRevision, profile: TargetProfile,
            opportunity: Opportunity, retrieval: scoped_evidence.RetrievalResult,
            *, max_chars: int) -> tuple[str, tuple[Mapping[str, Any], ...]]:
    mandatory = tuple(item.to_dict() for item in retrieval.mandatory_conflicts)
    ordinary = []
    for item in retrieval.findings:
        row = item.to_dict()
        if item.magnitude_status == "stale_cross_epoch":
            row["finding"]["value"] = None
        ordinary.append(row)
    body = {"target": target.to_dict(), "profile": profile.to_dict(),
            "opportunity": opportunity.to_dict(),
            "ordinary_evidence": ordinary}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"))
    if len(text) > max_chars:
        text = text[:max_chars]
    return text, mandatory


def _runtime_proposal(target: campaign.TargetRevision, opportunity: Opportunity,
                      pair: RuntimeArmPair, snapshot: scoped_evidence.ProposalSnapshot,
                      plan_digest: str, sink_ref: str) -> UnifiedProposal:
    return UnifiedProposal.from_dict({
        "schema": PROPOSAL_SCHEMA,
        "proposal_id": f"{opportunity.opportunity_id}:{pair.dimension.dimension_id}",
        "target_revision_digest": _target_digest(target), "backend": target.execution.backend,
        "parent_identity": {"target_ids": list(target.target_ids),
                            "target_revision": target.revision},
        "control_identity": {"execution_digest": pair.anchor.execution_digest},
        "intervention_identity": {"execution_digest": pair.candidate.execution_digest},
        "kind": "runtime_recipe", "mechanism_id": opportunity.mechanism_id,
        "estimand": opportunity.estimand, "metric": opportunity.metric,
        "metric_direction": opportunity.metric_direction,
        "effect_question": _thaw(opportunity.effect_question),
        "changed_factors": list(opportunity.changed_factors),
        "instrument": opportunity.instrument, "unit": opportunity.unit,
        "required_witnesses": list(opportunity.required_witnesses),
        "stage_class": opportunity.stage_class,
        "estimated_duration_seconds": opportunity.estimated_duration_seconds,
        "experiment_plan_digest": plan_digest, "native_artifact_sink_ref": sink_ref,
        "runtime_pair": pair.to_dict(),
        "claim_key": opportunity.claim_key.to_dict(),
        "evidence_snapshot": snapshot.to_dict(),
    })


@dataclass(frozen=True)
class PlanningResult:
    proposals: tuple[UnifiedProposal, ...]
    stage_proposals: tuple[scheduling.StageProposal, ...]
    dispositions: tuple[Mapping[str, Any], ...]
    selection: scheduling.Selection | None
    dispatch: DispatchRequest | None
    scheduler_state: scheduling.SchedulerState
    schema: str = RESULT_SCHEMA


def plan_iteration(*, resolved_campaign: campaign.ResolvedCampaign,
                   profiles: Mapping[str, Mapping[str, Any] | TargetProfile],
                   evidence_index: scoped_evidence.EvidenceIndex,
                   runtime_anchors: PreparedRuntimeAnchors,
                   runtime_dimensions: Mapping[str, Sequence[Mapping[str, Any] | RuntimeDimension]],
                   source_actor: Callable[[str, tuple[Mapping[str, Any], ...],
                                           scoped_evidence.ProposalSnapshot], Mapping[str, Any]],
                   build_actor: Callable[[str, tuple[Mapping[str, Any], ...],
                                          scoped_evidence.ProposalSnapshot], Mapping[str, Any]],
                   scheduler_engine: scheduling.SchedulerEngine,
                   experiment_plans: Mapping[str, experiment_plan.ExperimentPlan | Mapping[str, Any]],
                   now: float, native_artifact_sink_ref: str,
                   stage_handler: Callable[[DispatchRequest], None] | None = None,
                   stop_requested: Callable[[], bool] = lambda: False,
                   max_prompt_chars: int = 12000) -> PlanningResult:
    """Plan once, ask the real scheduler, and optionally record its dispatch request."""
    if not isinstance(resolved_campaign, campaign.ResolvedCampaign):
        raise PlanningRefused("resolved_campaign must be a validated ResolvedCampaign")
    if not isinstance(evidence_index, scoped_evidence.EvidenceIndex):
        raise PlanningRefused("evidence_index must be an EvidenceIndex")
    if (not isinstance(runtime_anchors, PreparedRuntimeAnchors)
            or runtime_anchors._token is not _PREPARED_TOKEN
            or runtime_anchors.campaign_id != resolved_campaign.campaign_id
            or runtime_anchors.manifest_digest != resolved_campaign.manifest_digest
            or runtime_anchors.resolved_campaign_digest
               != _digest(resolved_campaign.to_dict())):
        raise PlanningRefused("prepared runtime anchors differ from resolved campaign")
    _number(now, "now")
    if isinstance(max_prompt_chars, bool) or not isinstance(max_prompt_chars, int) or max_prompt_chars < 256:
        raise PlanningRefused("max_prompt_chars must be an integer >= 256")
    if not isinstance(scheduler_engine, scheduling.SchedulerEngine):
        raise PlanningRefused("scheduler_engine must be the persistent operational scheduler")
    sink_ref = _text(native_artifact_sink_ref, "native_artifact_sink_ref")
    proposals: list[UnifiedProposal] = []
    stages: list[scheduling.StageProposal] = []
    dispositions: list[Mapping[str, Any]] = []
    stop = False
    for target in sorted(resolved_campaign.targets, key=lambda row: row.target_ids):
        if stop_requested():
            stop = True
            dispositions.append(_freeze({"target_ids": list(target.target_ids), "status": "stopped"}))
            break
        digest = _target_digest(target)
        if target.status != "ready":
            dispositions.append(_freeze({"target_ids": list(target.target_ids),
                                         "status": "prerequisite", "reasons": list(target.missing)}))
            continue
        raw_profile = profiles.get(digest)
        if raw_profile is None:
            dispositions.append(_freeze({"target_ids": list(target.target_ids),
                                         "status": "missing_profile"}))
            continue
        profile = raw_profile if isinstance(raw_profile, TargetProfile) else TargetProfile.from_dict(raw_profile)
        profile = TargetProfile.from_dict(profile.to_dict())
        if profile.target_revision_digest != digest or profile.freshness != "fresh":
            dispositions.append(_freeze({"target_ids": list(target.target_ids),
                                         "status": "stale_or_mismatched_profile"}))
            continue
        for opportunity in sorted(profile.opportunities, key=lambda item: item.opportunity_id):
            if stop_requested():
                stop = True
                dispositions.append(_freeze({"opportunity_id": opportunity.opportunity_id,
                                             "status": "stopped"}))
                break
            scope = opportunity.claim_key.target_scope
            mechanism = opportunity.claim_key.mechanism_identity
            expected_model = target.execution.model.sha256 if target.execution.model else ""
            if (scope["target"] not in target.target_ids
                    or scope["backend"] != target.execution.backend
                    or scope["model"] != expected_model
                    or scope["quant"] != profile.quant
                    or scope["workload"] != target.workload_signature
                    or scope["allocation"] != profile.resource_cost.digest
                    or mechanism.get("id") != opportunity.mechanism_id):
                dispositions.append(_freeze({"opportunity_id": opportunity.opportunity_id,
                                             "status": "claim_scope_mismatch"}))
                continue
            retrieval = evidence_index.retrieve(opportunity.claim_key.target_scope,
                                                opportunity.claim_key, "rank", limit=40)
            snapshot = evidence_index.proposal_snapshot(opportunity.claim_key, intended_use="rank")
            prompt, conflicts = _prompt(target, profile, opportunity, retrieval,
                                         max_chars=max_prompt_chars)
            generated: list[UnifiedProposal] = []
            if opportunity.kind == "runtime_recipe":
                anchor = runtime_anchors.recipes.get(digest)
                dimensions = runtime_dimensions.get(digest, ())
                if anchor is None:
                    dispositions.append(_freeze({"opportunity_id": opportunity.opportunity_id,
                                                 "status": "runtime_anchor_missing"}))
                    continue
                try:
                    pairs = enumerate_runtime_dimensions(anchor, dimensions)
                except PlanningRefused as exc:
                    dispositions.append(_freeze({"opportunity_id": opportunity.opportunity_id,
                                                 "status": "unsupported_runtime_dimension",
                                                 "reason": str(exc)}))
                    continue
                wanted = set(opportunity.runtime_dimension_ids)
                for pair in pairs:
                    if pair.dimension.dimension_id in wanted:
                        proposal_id = f"{opportunity.opportunity_id}:{pair.dimension.dimension_id}"
                        raw_plan = experiment_plans.get(proposal_id)
                        if raw_plan is None:
                            dispositions.append(_freeze({
                                "opportunity_id": opportunity.opportunity_id,
                                "status": "pending_experiment_plan", "proposal_id": proposal_id}))
                            continue
                        try:
                            plan = (experiment_plan.ExperimentPlan.from_dict(raw_plan.to_dict())
                                    if isinstance(raw_plan, experiment_plan.ExperimentPlan)
                                    else experiment_plan.ExperimentPlan.from_dict(raw_plan))
                        except Exception as exc:
                            raise PlanningRefused(f"experiment plan is invalid: {exc}") from exc
                        if (plan.campaign_id != resolved_campaign.campaign_id
                                or plan.target_revision != digest
                                or plan.metric != opportunity.metric
                                or plan.metric_direction != opportunity.metric_direction
                                or plan.estimand != opportunity.estimand
                                or plan.instrument_class != opportunity.instrument
                                or plan.unit != opportunity.unit
                                or plan.changed_factors != opportunity.changed_factors
                                or plan.required_witnesses != opportunity.required_witnesses
                                or _thaw(plan.anchor_identity).get("execution_digest")
                                   != pair.anchor.execution_digest
                                or _thaw(plan.candidate_identity).get("execution_digest")
                                   != pair.candidate.execution_digest):
                            raise PlanningRefused(
                                "experiment plan differs from proposal/arm identities")
                        generated.append(_runtime_proposal(target, opportunity, pair, snapshot,
                                                           plan.digest, sink_ref))
            else:
                actor = source_actor if opportunity.kind == "source" else build_actor
                try:
                    generated.append(UnifiedProposal.from_dict(actor(prompt, conflicts, snapshot)))
                except Exception as exc:
                    dispositions.append(_freeze({"opportunity_id": opportunity.opportunity_id,
                                                 "status": "actor_failed", "reason": str(exc)}))
                    continue
            for proposal in generated:
                if opportunity.kind != "runtime_recipe" \
                        and proposal.experiment_plan_digest is not None:
                    raise PlanningRefused(
                        "source/build preparation cannot claim a final experiment plan")
                if (proposal.target_revision_digest != digest
                        or proposal.kind != opportunity.kind
                        or proposal.backend != target.execution.backend
                        or proposal.native_artifact_sink_ref != sink_ref
                        or proposal.claim_key.to_dict() != opportunity.claim_key.to_dict()
                        or proposal.mechanism_id != opportunity.mechanism_id
                        or proposal.changed_factors != opportunity.changed_factors
                        or proposal.estimand != opportunity.estimand
                        or proposal.metric != opportunity.metric
                        or proposal.metric_direction != opportunity.metric_direction
                        or _thaw(proposal.effect_question) != _thaw(opportunity.effect_question)
                        or proposal.instrument != opportunity.instrument
                        or proposal.unit != opportunity.unit
                        or proposal.required_witnesses != opportunity.required_witnesses
                        or proposal.stage_class != opportunity.stage_class
                        or proposal.estimated_duration_seconds
                           != opportunity.estimated_duration_seconds
                        or _thaw(proposal.evidence_snapshot) != snapshot.to_dict()):
                    raise PlanningRefused("actor proposal changed frozen planner bindings")
                proposals.append(proposal)
                stage_class = ("prerequisite" if proposal.experiment_plan_digest is None
                               else proposal.stage_class)
                stages.append(scheduling.StageProposal(
                    proposal_id=proposal.proposal_id, submitted_at=now,
                    backend=proposal.backend, target_revision=digest,
                    alias_identity=target.workload_signature,
                    frontier_id=(digest if "production" in target.enrolled_as else None),
                    production_frontier=("production" in target.enrolled_as
                                         and bool(target.required_obligations)),
                    seed_id=(target.target_ids[0] if "seed" in target.enrolled_as else None),
                    stage_class=stage_class,
                    estimated_duration_seconds=proposal.estimated_duration_seconds,
                    estimated_claims=profile.resource_cost,
                    eligible=True, eligibility_ref=_digest(snapshot.to_dict()),
                    reservation_kind=("seed" if "seed" in target.enrolled_as else None),
                    full_region=profile.resource_cost.physical_region_fraction == 1.0,
                    compatibility_authority_refs=(), safe_chunking_declared=False))
    if stop or stop_requested():
        return PlanningResult(tuple(proposals), tuple(stages), tuple(dispositions), None,
                              None, scheduler_engine.export_state())
    selection = scheduler_engine.select_stage(stages, now=now)
    next_state = scheduler_engine.export_state()
    dispatch = None
    if selection.status == "selected" and selection.proposal is not None:
        selected = next(item for item in proposals
                        if item.proposal_id == selection.proposal.proposal_id)
        intent = _freeze({"schema": INTENT_SCHEMA,
                          "status": ("ready_comparison" if
                                     selected.experiment_plan_digest is not None
                                     else "pending_plan_preparation"),
                          "proposal_digest": selected.digest,
                          "stage_proposal_digest": selection.proposal.digest,
                          "experiment_plan_digest": selected.experiment_plan_digest,
                          "claim_key": selected.claim_key.to_dict(),
                          "effect_question": _thaw(selected.effect_question),
                          "arm_scalars_are_gain_evidence": False})
        dispatch = DispatchRequest(_freeze(selection.to_dict()),
                                   _freeze(selected.to_dict()), intent)
        if stage_handler is not None:
            stage_handler(dispatch)
    return PlanningResult(tuple(proposals), tuple(stages), tuple(dispositions), selection,
                          dispatch, next_state)


__all__ = ["DispatchRequest", "Opportunity", "PlanningRefused", "PlanningResult",
           "PreparedRuntimeAnchors", "RuntimeAnchor", "RuntimeArmPair", "RuntimeDimension",
           "TargetProfile", "UnifiedProposal", "enumerate_runtime_dimensions",
           "plan_iteration", "prepare_runtime_anchors"]
