"""Closed startup materialization for the standalone v3 runtime owner.

The manifest hash protects file integrity only.  Provider capability comes solely
from an application-installed :class:`ProviderRegistry`; evidence authority remains
whatever the reconstructed EvidenceIndex can verify with its installed callbacks.
"""
from __future__ import annotations

import hashlib
import json
import copy
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

from . import (campaign, campaign_control, campaign_service, experiment_plan,
               scheduling, scoped_evidence, standalone_runtime, unified_driver,
               unified_planner, feed_runtime)

MANIFEST_SCHEMA = "epyc.autokernel.standalone_inputs.v1"
FEED_MANIFEST_SCHEMA = "epyc.autokernel.standalone_inputs.v2"
NATIVE_MANIFEST_SCHEMA = "epyc.autokernel.standalone_inputs.v3"
NATIVE_EVIDENCE_SCHEMA = "epyc.autokernel.standalone_native_evidence.v1"
SCIENTIFIC_SELECTION_SCHEMA = "epyc.autokernel.scientific_adapter_selection.v1"
PREFLIGHT_SCHEMA = "epyc.autokernel.standalone_inputs_preflight.v1"
NATIVE_PREFLIGHT_SCHEMA = "epyc.autokernel.standalone_inputs_preflight.v2"
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
    evidence_feed: feed_runtime.FeedConfig | None = None
    native_evidence: Mapping[str, Any] | None = None

    @classmethod
    def from_dict(cls, value: Any) -> "StartupManifest":
        row = dict(_mapping(value, "startup manifest"))
        fields = {"schema", "driver_config", "evidence_index", "actor_identities",
                  "lifecycle_provider_id", "readiness_provider_id",
                  "evidence_verifier_id", "manifest_digest"}
        feed_mode = row.get("schema") in {FEED_MANIFEST_SCHEMA, NATIVE_MANIFEST_SCHEMA}
        if feed_mode:
            fields = (fields - {"evidence_index", "evidence_verifier_id"}) | {"evidence_feed"}
        native_mode = row.get("schema") == NATIVE_MANIFEST_SCHEMA
        if native_mode:
            fields |= {"native_evidence"}
        if set(row) != fields or row["schema"] not in {
                MANIFEST_SCHEMA, FEED_MANIFEST_SCHEMA, NATIVE_MANIFEST_SCHEMA}:
            raise StandaloneInputsRefused("startup manifest fields/schema differ")
        supplied_digest = row.pop("manifest_digest")
        if (not isinstance(supplied_digest, str) or len(supplied_digest) != 64
                or supplied_digest != _digest(row)):
            raise StandaloneInputsRefused("startup manifest digest does not verify")
        try:
            config = unified_driver.DriverConfig.from_dict(row["driver_config"])
        except Exception as exc:
            raise StandaloneInputsRefused(f"driver config is invalid: {exc}") from exc
        evidence = _freeze(_mapping(row.get("evidence_index", {}), "evidence_index"))
        feed = feed_runtime.FeedConfig.from_dict(row["evidence_feed"]) if feed_mode else None
        actor_rows = _mapping(row["actor_identities"], "actor_identities")
        if not set(actor_rows) <= {"source", "build"}:
            raise StandaloneInputsRefused("actor identities contain an unsupported actor kind")
        actors = {}
        for kind, identity in actor_rows.items():
            actors[kind] = _freeze(_mapping(identity, f"actor_identities.{kind}"))
            if not actors[kind]:
                raise StandaloneInputsRefused("actor identity must not be empty")
        native = (_freeze(_mapping(row["native_evidence"], "native_evidence"))
                  if native_mode else None)
        return cls(
            config, evidence, MappingProxyType(actors),
            _text(row["lifecycle_provider_id"], "lifecycle_provider_id"),
            _text(row["readiness_provider_id"], "readiness_provider_id"),
            "" if feed_mode else _text(row["evidence_verifier_id"], "evidence_verifier_id"),
            supplied_digest, row["schema"], feed, native,
        )

    def body(self) -> dict[str, Any]:
        result = {
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
        if self.schema in {FEED_MANIFEST_SCHEMA, NATIVE_MANIFEST_SCHEMA}:
            if self.evidence_feed is None:
                raise StandaloneInputsRefused("feed manifest requires typed feed configuration")
            result.pop("evidence_index")
            result.pop("evidence_verifier_id")
            result["evidence_feed"] = self.evidence_feed.to_dict()
        if self.schema == NATIVE_MANIFEST_SCHEMA:
            if self.native_evidence is None:
                raise StandaloneInputsRefused("native manifest requires native evidence configuration")
            result["native_evidence"] = _thaw(self.native_evidence)
        return result

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
                 profile_executions: Mapping[str, Any] | None = None,
                 evidence_feeds: Mapping[str, feed_runtime.InstalledFeedBinding] | None = None
                 ) -> None:
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
        feeds = {}
        for identifier, binding in (evidence_feeds or {}).items():
            _text(identifier, "feed binding identifier")
            if not isinstance(binding, feed_runtime.InstalledFeedBinding):
                raise StandaloneInputsRefused("evidence feed binding must be installed and typed")
            feeds[identifier] = binding
        self._evidence_feeds = MappingProxyType(feeds)

    def get(self, identifier: str) -> ProviderBinding | None:
        return self._bindings.get(identifier)

    def evidence_verifier(self, identifier: str) -> EvidenceVerifierBinding | None:
        return self._evidence_verifiers.get(identifier)

    def profile_execution(self, identifier: str) -> Any:
        return self._profile_executions.get(identifier)

    def evidence_feed(self, identifier: str) -> feed_runtime.InstalledFeedBinding | None:
        return self._evidence_feeds.get(identifier)


def _installed_scientific_adapters(value: Any):
    """Construct one closed installed adapter registry without restoring issuance."""
    from . import native_scientific_witness as nsw
    selection = dict(_mapping(value, "scientific adapter selection"))
    if set(selection) != {"schema", "correctness", "purpose", "contention", "residency"} \
            or selection["schema"] != SCIENTIFIC_SELECTION_SCHEMA \
            or any(selection[name] is not None for name in ("purpose", "contention", "residency")):
        raise StandaloneInputsRefused("scientific adapter selection is unsupported")
    correctness = dict(_mapping(selection["correctness"], "correctness adapter selection"))
    if set(correctness) != {"adapter_id", "max_units"} \
            or correctness["adapter_id"] != nsw.ADAPTER_ID:
        raise StandaloneInputsRefused("configured correctness adapter is not installed")
    try:
        return nsw.ParentScientificWitnessAdapters(
            correctness=nsw.NativeT0WitnessAdapter(max_units=correctness["max_units"]))
    except Exception as exc:
        raise StandaloneInputsRefused(f"scientific adapter selection is invalid: {exc}") from exc


def native_evidence_document(*, scientific_adapters: Any,
                             model_preparations: Any,
                             artifact_root: str,
                             observation_configuration: Any,
                             search_window_configuration: Any = None) -> dict[str, Any]:
    """Create the v3 manifest projection from a closed installed selection."""
    from . import observation_binding as ob, serving
    adapters = _installed_scientific_adapters(scientific_adapters)
    identity_kwargs = {"measurement_callable": serving._measure_once,
                       "fence_clock": time.monotonic, "serving_timer": time.time,
                       "scientific_adapters": adapters}
    if search_window_configuration is not None:
        identity_kwargs["search_window_configuration"] = search_window_configuration
    identity = ob._plain(ob.loaded_planned_serving_identity(**identity_kwargs))
    root = Path(_text(artifact_root, "native artifact root"))
    if not root.is_absolute():
        raise StandaloneInputsRefused("native artifact root must be absolute")
    return {"schema": NATIVE_EVIDENCE_SCHEMA,
            "scientific_adapters": _thaw(scientific_adapters),
            "model_preparations": _thaw(model_preparations),
            "loaded_instrument_identity": identity, "artifact_root": str(root),
            "observation_configuration": _thaw(observation_configuration),
            "search_window_configuration": (None if search_window_configuration is None
                else search_window_configuration.to_dict())}


def _native_evidence(value: Any):
    """Rebuild the one installed factual configuration selected by startup v3."""
    from . import measurement_capture as mc, native_parent_service as nps
    from . import observation_binding as ob, serving
    row = dict(_mapping(_thaw(value), "native evidence configuration"))
    if set(row) != {"schema", "scientific_adapters", "model_preparations",
                    "loaded_instrument_identity", "artifact_root",
                    "observation_configuration", "search_window_configuration"} \
            or row["schema"] != NATIVE_EVIDENCE_SCHEMA:
        raise StandaloneInputsRefused("native evidence configuration fields/schema differ")
    adapters = _installed_scientific_adapters(row["scientific_adapters"])
    search_window = None
    if row["search_window_configuration"] is not None:
        try:
            from .search_window import InstalledSearchWindowConfiguration
            search_window = InstalledSearchWindowConfiguration.from_dict(
                row["search_window_configuration"])
        except Exception as exc:
            raise StandaloneInputsRefused(
                f"search window configuration is invalid: {exc}") from exc
    try:
        configuration = nps.NativeFactualEvidenceConfiguration(
            schema=nps.FACTUAL_CONFIGURATION_SCHEMA_V2,
            scientific_adapters=adapters, model_preparations=row["model_preparations"],
            search_window_configuration=search_window)
        identity_kwargs = {"measurement_callable": serving._measure_once,
                           "fence_clock": time.monotonic, "serving_timer": time.time,
                           "scientific_adapters": adapters}
        if search_window is not None:
            identity_kwargs["search_window_configuration"] = search_window
        identity = ob._plain(ob.loaded_planned_serving_identity(**identity_kwargs))
    except Exception as exc:
        raise StandaloneInputsRefused(f"native evidence configuration is invalid: {exc}") from exc
    supplied = ob._plain(_mapping(row["loaded_instrument_identity"],
                                  "loaded instrument identity"))
    if supplied != identity:
        raise StandaloneInputsRefused(
            "configured loaded instrument differs from the installed adapters")
    name, encoded, _ = mc.ArtifactStore._identity(
        f"loaded-instrument:{identity['sha256']}", identity)
    reference = ob.LoadedInstrumentReference(
        identity["sha256"], identity["configuration_complete"],
        mc.StoredArtifact(name, hashlib.sha256(encoded).hexdigest(), True))
    root = Path(_text(row["artifact_root"], "native artifact root"))
    if not root.is_absolute():
        raise StandaloneInputsRefused("native artifact root must be absolute")
    observation = dict(_mapping(row["observation_configuration"],
                                "observation configuration"))
    if set(observation) != {"requested_effective_states", "required_gpu_dsos",
                            "cadence_s", "gap_limit_s", "budgets"}:
        raise StandaloneInputsRefused("observation configuration fields differ")
    try:
        parsed_observation = ob.ParentObservationConfiguration(**observation)
    except Exception as exc:
        raise StandaloneInputsRefused(f"observation configuration is invalid: {exc}") from exc
    return configuration, _freeze(identity), reference, root, parsed_observation


@dataclass(frozen=True)
class MaterializedInputs:
    manifest: StartupManifest
    resolved: campaign.ResolvedCampaign
    inputs: standalone_runtime.StandaloneRuntimeInputs
    missing_prerequisites: tuple[str, ...]
    pending_profile_targets: tuple[str, ...] = ()

    def preflight(self, registry: ProviderRegistry | None = None) -> dict[str, Any]:
        missing = list(self.missing_prerequisites)
        feed = self.manifest.evidence_feed
        if registry is None:
            missing.extend((
                f"lifecycle_provider:{self.manifest.lifecycle_provider_id}:unavailable",
                f"readiness_provider:{self.manifest.readiness_provider_id}:unavailable",
            ))
            missing.append(f"evidence_feed:{feed.binding_id}:unavailable" if feed is not None
                           else f"evidence_verifier:{self.manifest.evidence_verifier_id}:unavailable")
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
            if feed is not None:
                binding = registry.evidence_feed(feed.binding_id)
                if binding is None:
                    missing.append(f"evidence_feed:{feed.binding_id}:unavailable")
                elif binding.current_epoch != feed.expected_epoch:
                    missing.append(f"evidence_feed:{feed.binding_id}:epoch_mismatch")
            elif verifier is None or any(not callable(item) for item in callbacks):
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
        result = {
            "schema": (NATIVE_PREFLIGHT_SCHEMA if self.manifest.schema == NATIVE_MANIFEST_SCHEMA
                       else PREFLIGHT_SCHEMA),
            "status": "ready" if not missing else "unavailable",
            "manifest_digest": self.manifest.manifest_digest,
            "campaign_id": self.resolved.campaign_id,
            "config_generation": self.manifest.driver_config.config_generation,
            "store_path": self.manifest.driver_config.store_path,
            "missing_prerequisites": sorted(set(missing)),
            "pending_profile_targets": list(self.pending_profile_targets),
            "execution_authorized": False,
        }
        if self.manifest.schema == NATIVE_MANIFEST_SCHEMA:
            result["native_instrument_runtime_status"] = "planned_unpublished"
            result["native_retention_catalog_status"] = "planned_unpublished"
        return result


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
        if value.evidence_feed is None:
            evidence = scoped_evidence.EvidenceIndex.from_dict(_thaw(value.evidence_index))
        else:
            feed_runtime.validate_paths(value.evidence_feed, Path(config.store_path))
            evidence = scoped_evidence.EvidenceIndex(
                (), current_epoch=value.evidence_feed.expected_epoch, projection_available=False)
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
        if value.evidence_feed is not None and any(
                plan.epoch != value.evidence_feed.expected_epoch for plan in plans.values()):
            raise StandaloneInputsRefused("experiment plan epoch differs from configured feed epoch")
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
        native_configuration = None
        native_identity = None
        native_reference = None
        native_artifact_root = None
        observation_configuration = None
        retention_catalog_seed = None
        retention_runtime_recipes = None
        if value.schema == NATIVE_MANIFEST_SCHEMA:
            (native_configuration, native_identity, native_reference,
             native_artifact_root, observation_configuration) = _native_evidence(
                 value.native_evidence)
            if (native_artifact_root != Path(config.native_artifact_sink_ref)
                    or native_artifact_root
                       != Path(config.store_path) / "unified-native-artifacts"):
                raise StandaloneInputsRefused(
                    "native artifact root differs from controller/startup configuration")
            expected_reference = native_reference.to_dict()
            for key, plan in plans.items():
                if (plan.schema != experiment_plan.PLAN_SCHEMA_V2
                        or _thaw(plan.loaded_instrument) != expected_reference):
                    raise StandaloneInputsRefused(
                        f"experiment plan {key} differs from configured loaded instrument")
                target_execution = execution.get(plan.target_revision)
                if (target_execution is None
                        or target_execution.instrument_id != native_reference.identity_sha256):
                    raise StandaloneInputsRefused(
                        f"experiment plan {key} lacks matching native execution input")
            from . import lifecycle_observation as lo
            expected_recipes = {}
            expected_target_recipes = {}
            for target_digest, anchor in anchors.recipes.items():
                recipes = [anchor]
                recipes.extend(arm for pair in unified_planner.enumerate_runtime_dimensions(
                    anchor, dimensions.get(target_digest, ()))
                               for arm in (pair.anchor, pair.candidate))
                for recipe in recipes:
                    expected_target_recipes[target_digest, recipe.execution_digest] = recipe
                    previous = expected_recipes.setdefault(recipe.execution_digest, recipe)
                    if previous.to_dict() != recipe.to_dict():
                        raise StandaloneInputsRefused(
                            "one execution digest identifies different native recipes")
            states = observation_configuration.requested_effective_states
            dsos = observation_configuration.required_gpu_dsos
            if set(states) != set(expected_recipes) or set(dsos) - set(expected_recipes):
                raise StandaloneInputsRefused(
                    "observation configuration does not cover exact native recipes")
            for digest, recipe in expected_recipes.items():
                if recipe.template.cpu_list is None:
                    raise StandaloneInputsRefused(
                        "native observation requires explicit recipe CPU placement")
                if list(states[digest]["logical_cpus"]) != sorted(
                        lo.parse_cpu_list(recipe.template.cpu_list)):
                    raise StandaloneInputsRefused(
                        "observation CPU placement differs from native recipe")
                expected_dsos = ([] if recipe.backend == "cpu" else
                                 [item.to_dict() for item in recipe.dsos])
                if _thaw(dsos.get(digest, ())) != expected_dsos:
                    raise StandaloneInputsRefused(
                        "observation GPU DSO identities differ from native recipe")
            configured_pairs = {(target, recipe) for target, recipes in
                                native_configuration.model_preparations.items()
                                for recipe in recipes}
            if configured_pairs != set(expected_target_recipes):
                raise StandaloneInputsRefused(
                    "model preparations do not cover exact target/recipe pairs")
            for (target, digest), recipe in expected_target_recipes.items():
                preparation = native_configuration.preparation(target, digest)
                if (preparation is None
                        or (preparation.entry_path, preparation.entry_sha256)
                           != (recipe.model.path, recipe.model.sha256)):
                    raise StandaloneInputsRefused(
                        "model preparation entry differs from native recipe model")
            from . import native_retention_catalog
            retention_runtime_recipes = MappingProxyType({
                target: MappingProxyType({
                    recipe: expected_target_recipes[target, recipe]
                    for known_target, recipe in expected_target_recipes
                    if known_target == target})
                for target in sorted({target for target, _recipe in expected_target_recipes})})
            retention_catalog_seed = native_retention_catalog.build_seed(
                resolved, anchors,
                config_digest=campaign_control.resolved_config_digest(resolved),
                model_preparations=native_configuration.model_preparations,
                runtime_recipes=retention_runtime_recipes,
                artifact_root=native_artifact_root)
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
    if value.evidence_feed is None and not evidence.projection_available:
        missing.append("evidence_index:projection_unavailable")
    if native_runtime_requested and value.schema != NATIVE_MANIFEST_SCHEMA:
        missing.append("native_observation:typed_source_runtime_consumer_unavailable")
    inputs = standalone_runtime.StandaloneRuntimeInputs(
        resolved, scheduler_engine, MappingProxyType(profiles), evidence, anchors,
        MappingProxyType(dimensions), MappingProxyType(plans), MappingProxyType(requests),
        value.actor_identities, MappingProxyType(execution), config.native_artifact_sink_ref,
        None, native_configuration, native_identity,
        None if native_reference is None else _freeze(native_reference.to_dict()),
        native_artifact_root, observation_configuration)
    if retention_catalog_seed is not None:
        inputs = replace(
            inputs, retention_catalog_seed=retention_catalog_seed,
            retention_runtime_recipes=retention_runtime_recipes)
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
    assert lifecycle is not None and readiness is not None
    provider = lifecycle.lifecycle_provider
    if any(not callable(getattr(provider, name, None)) for name in _PROVIDER_METHODS):
        raise StandaloneInputsRefused("lifecycle provider lacks the trusted provider contract")
    if not callable(readiness.readiness_check):
        raise StandaloneInputsRefused("readiness provider lacks a bounded readiness check")
    if materialized.manifest.evidence_feed is not None:
        binding = registry.evidence_feed(materialized.manifest.evidence_feed.binding_id)
        assert binding is not None
        verified_inputs = materialized.inputs
    else:
        assert verifier is not None
        verified_inputs = _verified_snapshot_inputs(materialized, verifier)

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
            current_inputs = verified_inputs
            if materialized.manifest.evidence_feed is not None:
                current_inputs = replace(verified_inputs, feed_owner=feed_runtime.FeedRuntimeOwner(
                    materialized.manifest.evidence_feed, binding))
            if current_inputs.native_evidence_configuration is not None:
                controller.install_native_retention_catalog(
                    current_inputs.retention_catalog_seed,
                    runtime_anchors=current_inputs.runtime_anchors,
                    model_preparations=(
                        current_inputs.native_evidence_configuration.model_preparations),
                    runtime_recipes=current_inputs.retention_runtime_recipes,
                    artifact_root=current_inputs.native_artifact_root)
                from . import measurement_capture as mc, observation_binding as ob
                identity = _thaw(current_inputs.loaded_instrument_identity)
                reference = ob.LoadedInstrumentReference.from_dict(
                    _thaw(current_inputs.loaded_instrument_reference))
                store = mc.ArtifactStore(current_inputs.native_artifact_root)
                try:
                    written = store.write(
                        f"loaded-instrument:{identity['sha256']}", identity)
                finally:
                    store.close()
                if written.to_dict() != reference.artifact.to_dict():
                    raise StandaloneInputsRefused(
                        "runtime loaded instrument publication differs from startup")
            runtime = standalone_runtime.StandaloneRuntime.compose(
                controller=controller, inputs=current_inputs)
        except BaseException:
            controller.close()
            raise
        return controller, runtime

    return build


def _verified_snapshot_inputs(materialized, verifier):
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
    return replace(materialized.inputs, evidence_index=evidence)


__all__ = ["MANIFEST_SCHEMA", "FEED_MANIFEST_SCHEMA", "NATIVE_MANIFEST_SCHEMA",
           "NATIVE_EVIDENCE_SCHEMA", "SCIENTIFIC_SELECTION_SCHEMA", "PREFLIGHT_SCHEMA",
           "NATIVE_PREFLIGHT_SCHEMA",
           "EvidenceVerifierBinding", "MaterializedInputs",
           "ProviderBinding", "ProviderRegistry", "StandaloneInputsRefused",
           "StartupManifest", "materialize", "native_evidence_document", "runtime_factory"]
