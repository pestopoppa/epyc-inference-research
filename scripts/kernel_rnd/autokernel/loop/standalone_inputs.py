"""Closed startup materialization for the standalone v3 runtime owner.

The manifest hash protects file integrity only.  Provider capability comes solely
from an application-installed :class:`ProviderRegistry`; evidence authority remains
whatever the reconstructed EvidenceIndex can verify with its installed callbacks.
"""
from __future__ import annotations

import hashlib
import json
import copy
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

from . import (campaign, campaign_control, campaign_service, experiment_plan,
               scheduling, scoped_evidence, standalone_runtime, unified_driver,
               unified_planner)

MANIFEST_SCHEMA = "epyc.autokernel.standalone_inputs.v1"
PREFLIGHT_SCHEMA = "epyc.autokernel.standalone_inputs_preflight.v1"
_PROVIDER_METHODS = (
    "authorize", "inspect_pending", "refresh", "release", "inspect",
    "close_held_receipt", "describe_active_observation_claim",
)


class StandaloneInputsRefused(RuntimeError):
    """Startup input is malformed, stale, foreign, or unavailable."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise StandaloneInputsRefused("startup manifest is not canonical JSON") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise StandaloneInputsRefused(f"{label} must be an object with string keys")
    _canonical(value)
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StandaloneInputsRefused(f"{label} must be nonempty text")
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({copy.deepcopy(key): _freeze(item)
                                 for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return copy.deepcopy(value)


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {copy.deepcopy(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return copy.deepcopy(value)


@dataclass(frozen=True)
class StartupManifest:
    driver_config: unified_driver.DriverConfig
    evidence_index: Mapping[str, Any]
    actor_identities: Mapping[str, Mapping[str, Any]]
    lifecycle_provider_id: str
    readiness_provider_id: str
    evidence_verifier_id: str
    manifest_digest: str
    schema: str = MANIFEST_SCHEMA

    @classmethod
    def from_dict(cls, value: Any) -> "StartupManifest":
        row = dict(_mapping(value, "startup manifest"))
        fields = {"schema", "driver_config", "evidence_index", "actor_identities",
                  "lifecycle_provider_id", "readiness_provider_id",
                  "evidence_verifier_id", "manifest_digest"}
        if set(row) != fields or row["schema"] != MANIFEST_SCHEMA:
            raise StandaloneInputsRefused("startup manifest fields/schema differ")
        supplied_digest = row.pop("manifest_digest")
        if (not isinstance(supplied_digest, str) or len(supplied_digest) != 64
                or supplied_digest != _digest(row)):
            raise StandaloneInputsRefused("startup manifest digest does not verify")
        try:
            config = unified_driver.DriverConfig.from_dict(row["driver_config"])
        except Exception as exc:
            raise StandaloneInputsRefused(f"driver config is invalid: {exc}") from exc
        evidence = _freeze(_mapping(row["evidence_index"], "evidence_index"))
        actor_rows = _mapping(row["actor_identities"], "actor_identities")
        if not set(actor_rows) <= {"source", "build"}:
            raise StandaloneInputsRefused("actor identities contain an unsupported actor kind")
        actors = {}
        for kind, identity in actor_rows.items():
            actors[kind] = _freeze(_mapping(identity, f"actor_identities.{kind}"))
            if not actors[kind]:
                raise StandaloneInputsRefused("actor identity must not be empty")
        return cls(
            config, evidence, MappingProxyType(actors),
            _text(row["lifecycle_provider_id"], "lifecycle_provider_id"),
            _text(row["readiness_provider_id"], "readiness_provider_id"),
            _text(row["evidence_verifier_id"], "evidence_verifier_id"),
            supplied_digest,
        )

    def body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "driver_config": {
                "schema": self.driver_config.schema,
                "resolved_campaign_path": self.driver_config.resolved_campaign_path,
                "store_path": self.driver_config.store_path,
                "scheduler_config": unified_driver._thaw(self.driver_config.scheduler_config),
                "scheduler_state": unified_driver._thaw(self.driver_config.scheduler_state),
                "runtime_anchors": unified_driver._thaw(self.driver_config.runtime_anchors),
                "runtime_dimensions": unified_driver._thaw(self.driver_config.runtime_dimensions),
                "profiles": unified_driver._thaw(self.driver_config.profiles),
                "experiment_plans": unified_driver._thaw(self.driver_config.experiment_plans),
                "profile_requests": unified_driver._thaw(self.driver_config.profile_requests),
                "execution_inputs": unified_driver._thaw(self.driver_config.execution_inputs),
                "native_artifact_sink_ref": self.driver_config.native_artifact_sink_ref,
                "config_generation": self.driver_config.config_generation,
            },
            "evidence_index": _thaw(self.evidence_index),
            "actor_identities": {
                key: _thaw(value) for key, value in self.actor_identities.items()},
            "lifecycle_provider_id": self.lifecycle_provider_id,
            "readiness_provider_id": self.readiness_provider_id,
            "evidence_verifier_id": self.evidence_verifier_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.body() | {"manifest_digest": self.manifest_digest}


@dataclass(frozen=True)
class ProviderBinding:
    lifecycle_provider: Any = None
    readiness_check: Callable[[], tuple[bool, str | None]] | None = None


@dataclass(frozen=True)
class EvidenceVerifierBinding:
    scope_verifier: Callable[..., bool | str]
    use_verifier: Callable[..., bool | str]
    result_verifier: Callable[..., bool | str]
    support_rule_identity: str


class ProviderRegistry:
    """Application-owned locator; configuration cannot construct its values.

    Profile entries are deliberately non-authoritative ``Any`` locators while their
    owner module is absent. They are never consumed here and, when that module is
    present, are checked against its concrete TargetProfileExecution type.
    """

    def __init__(self, bindings: Mapping[str, ProviderBinding], *,
                 evidence_verifiers: Mapping[str, EvidenceVerifierBinding] | None = None,
                 profile_executions: Mapping[str, Any] | None = None) -> None:
        if not isinstance(bindings, Mapping):
            raise StandaloneInputsRefused("provider registry must be a mapping")
        normalized = {}
        for identifier, binding in bindings.items():
            _text(identifier, "provider registry identifier")
            if not isinstance(binding, ProviderBinding):
                raise StandaloneInputsRefused("provider registry values must be ProviderBinding")
            normalized[identifier] = binding
        self._bindings = MappingProxyType(normalized)
        verifiers = {}
        for identifier, binding in (evidence_verifiers or {}).items():
            _text(identifier, "evidence verifier identifier")
            if not isinstance(binding, EvidenceVerifierBinding):
                raise StandaloneInputsRefused(
                    "evidence registry values must be EvidenceVerifierBinding")
            verifiers[identifier] = binding
        self._evidence_verifiers = MappingProxyType(verifiers)
        profiles = {}
        for identifier, execution in (profile_executions or {}).items():
            _text(identifier, "profile execution identifier")
            profiles[identifier] = execution
        self._profile_executions = MappingProxyType(profiles)

    def get(self, identifier: str) -> ProviderBinding | None:
        return self._bindings.get(identifier)

    def evidence_verifier(self, identifier: str) -> EvidenceVerifierBinding | None:
        return self._evidence_verifiers.get(identifier)

    def profile_execution(self, identifier: str) -> Any:
        return self._profile_executions.get(identifier)


@dataclass(frozen=True)
class MaterializedInputs:
    manifest: StartupManifest
    resolved: campaign.ResolvedCampaign
    inputs: standalone_runtime.StandaloneRuntimeInputs
    missing_prerequisites: tuple[str, ...]
    pending_profile_targets: tuple[str, ...] = ()

    def preflight(self, registry: ProviderRegistry | None = None) -> dict[str, Any]:
        missing = list(self.missing_prerequisites)
        if registry is None:
            missing.extend((
                f"lifecycle_provider:{self.manifest.lifecycle_provider_id}:unavailable",
                f"readiness_provider:{self.manifest.readiness_provider_id}:unavailable",
                f"evidence_verifier:{self.manifest.evidence_verifier_id}:unavailable",
            ))
            for target in self.pending_profile_targets:
                mechanism_id = self.inputs.profile_requests[
                    target].profile_contract["adapter_id"]
                missing.append(
                    f"target:{target}:profile_execution:{mechanism_id}:unavailable")
        else:
            lifecycle = registry.get(self.manifest.lifecycle_provider_id)
            readiness = registry.get(self.manifest.readiness_provider_id)
            if lifecycle is None or lifecycle.lifecycle_provider is None:
                missing.append(
                    f"lifecycle_provider:{self.manifest.lifecycle_provider_id}:unavailable")
            elif any(not callable(getattr(lifecycle.lifecycle_provider, name, None))
                     for name in _PROVIDER_METHODS):
                missing.append(
                    f"lifecycle_provider:{self.manifest.lifecycle_provider_id}:invalid")
            if readiness is None or not callable(readiness.readiness_check):
                missing.append(
                    f"readiness_provider:{self.manifest.readiness_provider_id}:unavailable")
            verifier = registry.evidence_verifier(self.manifest.evidence_verifier_id)
            callbacks = (() if verifier is None else (
                verifier.scope_verifier, verifier.use_verifier, verifier.result_verifier))
            if verifier is None or any(not callable(item) for item in callbacks):
                missing.append(
                    f"evidence_verifier:{self.manifest.evidence_verifier_id}:unavailable")
            elif (not verifier.support_rule_identity
                  or verifier.support_rule_identity
                  != self.inputs.evidence_index._recorded_support_rule_identity):
                missing.append(
                    f"evidence_verifier:{self.manifest.evidence_verifier_id}:rule_mismatch")
            for target in self.pending_profile_targets:
                request = self.inputs.profile_requests[target]
                mechanism_id = request.profile_contract["adapter_id"]
                execution = registry.profile_execution(mechanism_id)
                if execution is None:
                    missing.append(
                        f"target:{target}:profile_execution:{mechanism_id}:unavailable")
                    continue
                try:
                    from .target_profile_execution import TargetProfileExecution
                except ImportError:
                    missing.append(
                        f"target:{target}:profile_execution:{mechanism_id}:source_unavailable")
                    continue
                if not isinstance(execution, TargetProfileExecution):
                    missing.append(
                        f"target:{target}:profile_execution:{mechanism_id}:invalid")
                else:
                    missing.append(
                        f"target:{target}:profile_execution:{mechanism_id}:runtime_consumer_unavailable")
        return {
            "schema": PREFLIGHT_SCHEMA,
            "status": "ready" if not missing else "unavailable",
            "manifest_digest": self.manifest.manifest_digest,
            "campaign_id": self.resolved.campaign_id,
            "config_generation": self.manifest.driver_config.config_generation,
            "store_path": self.manifest.driver_config.store_path,
            "missing_prerequisites": sorted(set(missing)),
            "pending_profile_targets": list(self.pending_profile_targets),
            "execution_authorized": False,
        }


def materialize(value: StartupManifest) -> MaterializedInputs:
    """Validate all file-supplied inputs without acquiring a store or provider."""
    if not isinstance(value, StartupManifest):
        raise StandaloneInputsRefused("materialization requires a StartupManifest")
    if value.manifest_digest != _digest(value.body()):
        raise StandaloneInputsRefused("startup manifest changed after digest validation")
    config = value.driver_config
    try:
        resolved = campaign_service.load_resolved(Path(config.resolved_campaign_path))
        scheduler_config = scheduling.SchedulerConfig.from_dict(
            unified_driver._thaw(config.scheduler_config))
        scheduler_state = scheduling.SchedulerState.from_dict(
            unified_driver._thaw(config.scheduler_state))
        scheduler_engine = scheduling.SchedulerEngine(scheduler_config, scheduler_state)
        anchors = unified_planner.prepare_runtime_anchors(
            resolved, unified_driver._thaw(config.runtime_anchors))
        evidence = scoped_evidence.EvidenceIndex.from_dict(_thaw(value.evidence_index))
        profiles = {}
        for key, item in config.profiles.items():
            profile = unified_planner.TargetProfile.from_dict(unified_driver._thaw(item))
            if key != profile.target_revision_digest:
                raise StandaloneInputsRefused("profile key differs from target revision")
            profiles[key] = profile
        dimensions = {}
        target_digests = {_digest(target.to_dict()) for target in resolved.targets}
        for key, items in config.runtime_dimensions.items():
            if key not in target_digests or not isinstance(items, (list, tuple)):
                raise StandaloneInputsRefused("runtime dimensions name an unknown target")
            dimensions[key] = tuple(unified_planner.RuntimeDimension.from_dict(
                unified_driver._thaw(item)) for item in items)
        plans = {key: experiment_plan.ExperimentPlan.from_dict(
            unified_driver._thaw(item)) for key, item in config.experiment_plans.items()}
        requests = {}
        for key, item in config.profile_requests.items():
            request = unified_driver.ProfilePreparationRequest.from_dict(
                unified_driver._thaw(item))
            if key != request.target_revision_digest or key not in target_digests:
                raise StandaloneInputsRefused("profile request key differs from target revision")
            requests[key] = request
        execution = {}
        for key, item in config.execution_inputs.items():
            parsed = unified_driver.ExecutionInput.from_dict(unified_driver._thaw(item))
            if key != parsed.target_revision_digest or key not in target_digests:
                raise StandaloneInputsRefused("execution input key differs from target revision")
            execution[key] = parsed
    except StandaloneInputsRefused:
        raise
    except Exception as exc:
        raise StandaloneInputsRefused(f"startup input materialization failed: {exc}") from exc
    missing = []
    pending_profiles = []
    required_actor_kinds = set()
    native_runtime_requested = False
    for target in resolved.targets:
        digest = _digest(target.to_dict())
        if target.status == "ready" and digest not in profiles:
            if digest in requests:
                pending_profiles.append(digest)
            else:
                missing.append(f"target:{digest}:profile_request_missing")
        profile = profiles.get(digest)
        if profile is not None:
            required_actor_kinds.update(
                item.kind for item in profile.opportunities
                if item.kind in {"source", "build"})
            native_runtime_requested = native_runtime_requested or any(
                item.kind == "runtime_recipe" for item in profile.opportunities)
    for kind in sorted(required_actor_kinds - set(value.actor_identities)):
        missing.append(f"actor_identity:{kind}:unavailable")
    if not evidence.projection_available:
        missing.append("evidence_index:projection_unavailable")
    if native_runtime_requested:
        missing.append("native_observation:typed_source_runtime_consumer_unavailable")
    inputs = standalone_runtime.StandaloneRuntimeInputs(
        resolved, scheduler_engine, MappingProxyType(profiles), evidence, anchors,
        MappingProxyType(dimensions), MappingProxyType(plans), MappingProxyType(requests),
        value.actor_identities, MappingProxyType(execution), config.native_artifact_sink_ref)
    return MaterializedInputs(
        value, resolved, inputs, tuple(sorted(missing)), tuple(sorted(pending_profiles)))


def runtime_factory(materialized: MaterializedInputs, registry: ProviderRegistry):
    """Return the exact typed factory consumed by campaign_service.main."""
    if not isinstance(materialized, MaterializedInputs):
        raise StandaloneInputsRefused("runtime factory requires materialized inputs")
    if not isinstance(registry, ProviderRegistry):
        raise StandaloneInputsRefused("runtime factory requires an installed provider registry")
    report = materialized.preflight(registry)
    if report["status"] != "ready":
        raise StandaloneInputsRefused(
            "startup prerequisites unavailable: " + ", ".join(report["missing_prerequisites"]))
    lifecycle = registry.get(materialized.manifest.lifecycle_provider_id)
    readiness = registry.get(materialized.manifest.readiness_provider_id)
    verifier = registry.evidence_verifier(materialized.manifest.evidence_verifier_id)
    assert lifecycle is not None and readiness is not None and verifier is not None
    provider = lifecycle.lifecycle_provider
    if any(not callable(getattr(provider, name, None)) for name in _PROVIDER_METHODS):
        raise StandaloneInputsRefused("lifecycle provider lacks the trusted provider contract")
    if not callable(readiness.readiness_check):
        raise StandaloneInputsRefused("readiness provider lacks a bounded readiness check")
    try:
        evidence = scoped_evidence.EvidenceIndex.from_dict(
            _thaw(materialized.manifest.evidence_index),
            scope_verifier=verifier.scope_verifier,
            use_verifier=verifier.use_verifier,
            result_verifier=verifier.result_verifier,
            support_rule_identity=verifier.support_rule_identity)
    except Exception as exc:
        raise StandaloneInputsRefused(f"evidence verifier binding is invalid: {exc}") from exc
    if evidence._recorded_support_rule_identity != verifier.support_rule_identity:
        raise StandaloneInputsRefused("evidence verifier support rule differs from projection")
    verified_inputs = replace(materialized.inputs, evidence_index=evidence)

    def build(resolved: campaign.ResolvedCampaign, args):
        config = materialized.manifest.driver_config
        if (resolved.to_dict() != materialized.resolved.to_dict()
                or Path(args.store).absolute() != Path(config.store_path).absolute()
                or args.config_generation != config.config_generation
                or args.snapshot_version != 3):
            raise StandaloneInputsRefused("service and startup manifest identities differ")
        controller = campaign_control.CampaignController(
            resolved, Path(config.store_path), config_generation=config.config_generation,
            readiness_check=readiness.readiness_check, snapshot_version=3,
            scheduler_engine=materialized.inputs.scheduler_engine,
            lifecycle_provider=provider)
        controller.__enter__()
        try:
            runtime = standalone_runtime.StandaloneRuntime.compose(
                controller=controller, inputs=verified_inputs)
        except BaseException:
            controller.close()
            raise
        return controller, runtime

    return build


__all__ = ["MANIFEST_SCHEMA", "PREFLIGHT_SCHEMA", "EvidenceVerifierBinding", "MaterializedInputs",
           "ProviderBinding", "ProviderRegistry", "StandaloneInputsRefused",
           "StartupManifest", "materialize", "runtime_factory"]
