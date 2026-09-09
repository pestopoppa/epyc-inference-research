"""Concrete bounded parent service: owned facts in, no scientific policy out."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from types import MappingProxyType
from typing import Any, Mapping

from . import driver_execution as de
from . import lifecycle_observation as lo
from . import measurement_capture as mc
from . import native_parent_evidence as npe
from . import native_parent_receipt_replay as replay
from . import observation_binding as ob
from . import unified_worker as uw
from . import worker_lifecycle as wl

FACTUAL_CONFIGURATION_SCHEMA = "epyc.autokernel.native_factual_evidence_configuration.v1"
FACTUAL_CONFIGURATION_SCHEMA_V2 = "epyc.autokernel.native_factual_evidence_configuration.v2"


@dataclass(frozen=True)
class NativeFactualEvidenceConfiguration:
    """Explicit factual mode; not eligibility, correctness, purpose, or GPU authority."""
    schema: str = FACTUAL_CONFIGURATION_SCHEMA
    scientific_adapters: Any = None
    model_preparations: Mapping[str, Any] = field(default_factory=dict)
    search_window_configuration: Any = None

    def __post_init__(self) -> None:
        from .native_scientific_witness import ParentScientificWitnessAdapters
        if self.schema not in {FACTUAL_CONFIGURATION_SCHEMA,
                               FACTUAL_CONFIGURATION_SCHEMA_V2}:
            raise de.DriverExecutionRefused("unsupported factual evidence configuration")
        if self.scientific_adapters is not None and type(self.scientific_adapters) is not ParentScientificWitnessAdapters:
            raise de.DriverExecutionRefused("scientific configuration requires its concrete registry")
        if self.search_window_configuration is not None:
            from .search_window import InstalledSearchWindowConfiguration
            if type(self.search_window_configuration) is not InstalledSearchWindowConfiguration:
                raise de.DriverExecutionRefused(
                    "window configuration requires its concrete installed type")
        from .native_model_preparation import ScheduledModelPreparation
        prepared = {}
        if not isinstance(self.model_preparations, Mapping):
            raise de.DriverExecutionRefused("model preparations must be a mapping")
        def parsed(value: Any) -> ScheduledModelPreparation:
            try:
                return (ScheduledModelPreparation.from_dict(value.to_dict())
                        if isinstance(value, ScheduledModelPreparation)
                        else ScheduledModelPreparation.from_dict(value))
            except Exception as exc:
                raise de.DriverExecutionRefused("model preparation is invalid") from exc
        if self.schema == FACTUAL_CONFIGURATION_SCHEMA:
            for key, value in self.model_preparations.items():
                if not isinstance(key, str):
                    raise de.DriverExecutionRefused("model preparation keys must be text")
                item = parsed(value)
                if key != item.recipe_execution_digest or key in prepared:
                    raise de.DriverExecutionRefused(
                        "model preparation key differs or is duplicated")
                prepared[key] = item
        else:
            for target_key, recipes in self.model_preparations.items():
                if not isinstance(target_key, str) or not isinstance(recipes, Mapping):
                    raise de.DriverExecutionRefused(
                        "v2 model preparations require target/recipe mappings")
                target_prepared = {}
                for recipe_key, value in recipes.items():
                    if not isinstance(recipe_key, str):
                        raise de.DriverExecutionRefused("model preparation recipe key must be text")
                    item = parsed(value)
                    if (target_key != item.target_revision_digest
                            or recipe_key != item.recipe_execution_digest
                            or recipe_key in target_prepared):
                        raise de.DriverExecutionRefused(
                            "model preparation target/recipe key differs or is duplicated")
                    target_prepared[recipe_key] = item
                prepared[target_key] = MappingProxyType(target_prepared)
        if prepared and (self.scientific_adapters is None
                         or self.scientific_adapters.correctness is None):
            raise de.DriverExecutionRefused(
                "model preparation requires the configured concrete T0 adapter")
        object.__setattr__(self, "model_preparations", MappingProxyType(prepared))

    def preparation(self, target_digest: str, recipe_digest: str):
        if self.schema == FACTUAL_CONFIGURATION_SCHEMA:
            item = self.model_preparations.get(recipe_digest)
            return item if item is None or item.target_revision_digest == target_digest else None
        return self.model_preparations.get(target_digest, {}).get(recipe_digest)


class NativeParentEvidenceService(de.UnknownParentEvidenceProducer):
    def __init__(self, authority: uw.ParentUnitEvidenceAuthority,
                 prepared: uw.PreparedPlannedServingStage, lifecycle: wl.WorkerLifecycle,
                 observation_configuration: ob.ParentObservationConfiguration, *,
                 registry: replay.IssuedNativeEvidenceRegistry,
                 runtime_probe: lo.FilesystemProbe | None = None,
                 scientific_adapters: Any = None,
                 model_preparations: Mapping[str, Any] | None = None,
                 factual_configuration: NativeFactualEvidenceConfiguration | None = None) -> None:
        super().__init__(authority, prepared, lifecycle, observation_configuration)
        if (prepared.schema != uw.PREPARED_SCHEMA_V2
                or type(registry) is not replay.IssuedNativeEvidenceRegistry
                or lifecycle is None or observation_configuration is None):
            raise de.DriverExecutionRefused("native factual service requires v2 parent authority")
        if runtime_probe is not None and type(runtime_probe) is not lo.FilesystemProbe:
            raise de.DriverExecutionRefused("native readback requires the concrete bounded reader")
        self.registry = registry
        if factual_configuration is not None:
            if (type(factual_configuration) is not NativeFactualEvidenceConfiguration
                    or scientific_adapters is not None or model_preparations is not None):
                raise de.DriverExecutionRefused("native factual configuration is ambiguous")
            configured = factual_configuration
        else:
            configured = NativeFactualEvidenceConfiguration(
                scientific_adapters=scientific_adapters,
                model_preparations={} if model_preparations is None else model_preparations)
        self.factual_configuration = configured
        self.scientific_adapters = configured.scientific_adapters
        self.model_preparations = configured.model_preparations
        self._native_store = mc.ArtifactStore(prepared.artifact_root)
        self._native_probe = runtime_probe
        self._unit_producers: dict[str, npe.NativeUnitEvidenceProducer] = {}
        self._phase_requests: dict[tuple[str, str], Mapping[str, Any]] = {}
        self._phase_results: dict[tuple[str, str], Mapping[str, Any]] = {}
        self._model_preparation_receipts: dict[str, mc.StoredArtifact] = {}
        self.search_window_owner = None
        self.search_window_receipts: dict[str, Any] = {}
        if configured.search_window_configuration is not None:
            from .search_window import SameAttemptWindowOwner
            try:
                self.search_window_owner = SameAttemptWindowOwner(
                    config=configured.search_window_configuration, prepared=prepared,
                    lifecycle=lifecycle, store=self._native_store)
            except BaseException:
                self._native_store.close()
                raise

    def _before_observation_binding(self, *, start: uw.WorkerStart, unit: Any,
                                    fence: Any, recipe: Any,
                                    claim: Mapping[str, Any]) -> None:
        if self.scientific_adapters is None or self.scientific_adapters.correctness is None:
            return
        from .native_model_preparation import (
            ActiveObservationPreparationClaim, BINDING_SCHEMA)
        target = self.prepared.dispatch["proposal"]["target_revision_digest"]
        preparation = self.factual_configuration.preparation(
            target, recipe.execution_digest)
        if preparation is None:
            raise de.DriverExecutionRefused(
                "selected native recipe lacks scheduled complete-model preparation")
        if (preparation.target_revision_digest != target
                or preparation.recipe_execution_digest != recipe.execution_digest
                or (preparation.entry_path, preparation.entry_sha256)
                   != (recipe.model.path, recipe.model.sha256)):
            raise de.DriverExecutionRefused(
                "model preparation differs from selected target/recipe/model")
        live_claim = ActiveObservationPreparationClaim(
            lifecycle=self.lifecycle, start=start, unit=unit, fence=fence,
            initial_claim=claim)
        receipt = self.scientific_adapters.correctness.prepare_model_identity(
            store=self._native_store, identity=preparation.identity(),
            entry_path=Path(preparation.entry_path), preparation_claim=live_claim)
        original = self._native_store.read(receipt.locator, receipt.sha256)
        if (original.get("entry_path"), original.get("entry_sha256")) != (
                recipe.model.path, recipe.model.sha256):
            raise de.DriverExecutionRefused(
                "verified model inventory entry differs from selected recipe model")
        binding_body = {"schema": BINDING_SCHEMA,
            "preparation": preparation.to_dict(),
            "selected_model": recipe.model.to_dict(),
            "original_model_receipt": receipt.to_dict(),
            "preparation_claim_id": original["preparation_claim_id"]}
        binding = self._native_store.write(
            f"scheduled-model-preparation:{preparation.preparation_digest}", binding_body)
        if not live_claim.is_held():
            raise de.DriverExecutionRefused(
                "model preparation claim expired before observation binding")
        prior = self._model_preparation_receipts.get(recipe.execution_digest)
        if prior is not None and prior != binding:
            raise de.DriverExecutionRefused("model preparation retry changed its receipt")
        self._model_preparation_receipts[recipe.execution_digest] = binding

    def _binding_ready(self, *, start: uw.WorkerStart, binding: ob.ObservationUnitBinding,
                       claim: Mapping[str, Any]) -> None:
        if self.search_window_owner is not None:
            self.search_window_owner.open_unit(start=start, binding=binding, claim=claim)

    def _poll_live_evidence(self) -> None:
        if self.search_window_owner is not None:
            self.search_window_owner.poll()

    def _evidence_poll_delay(self) -> float:
        return 0.05 if self.search_window_owner is None else self.search_window_owner.poll_delay()

    def _producer_for(self, notice: Mapping[str, Any]) -> npe.NativeUnitEvidenceProducer:
        start, fence = notice["start"], notice["fence"]
        if not isinstance(start, uw.WorkerStart):
            raise de.DriverExecutionRefused("native notice lacks its actual worker start")
        unit = next((unit for unit in self.plan.expected_units
                     if unit.unit_id == fence.unit_id), None)
        if unit is None:
            raise de.DriverExecutionRefused("native unit is absent from the frozen plan")
        key = (start.worker_id, start.worker_generation, unit.unit_id, unit.process_id,
               fence.fence_id)
        binding, target, claim = self._bindings.get(key), self._targets.get(key), self._claims.get(key)
        if binding is None or target is None or claim is None:
            raise de.DriverExecutionRefused("native unit lacks original parent claim/binding/target")
        producer = self._unit_producers.get(unit.unit_id)
        if producer is not None:
            if producer.context.fence != fence or producer.context.nonce != start.nonce:
                raise de.DriverExecutionRefused("native unit owner changed")
            return producer
        assert self.lifecycle is not None
        event = self.lifecycle.owned_descendant_event(binding_ref=target["binding_ref"])
        recipe = (self.prepared.runtime_pair.anchor if unit.arm == "anchor"
                  else self.prepared.runtime_pair.candidate)
        context = npe.ParentUnitContext(self.plan, unit.unit_id, recipe, self.prepared.prompts,
            fence, binding, event, claim, start.nonce)
        producer = npe.NativeUnitEvidenceProducer(
            store=self._native_store, context=context, runtime_probe=self._native_probe,
            scientific_adapters=self.scientific_adapters)
        self._unit_producers[unit.unit_id] = producer
        return producer

    def _phase_for(self, notice: Mapping[str, Any]) -> Mapping[str, Any]:
        producer = self._producer_for(notice)
        context, packet = producer.context, notice["request"]
        if (packet["phase"] not in uw.OBSERVATION_WINDOW_MARKERS
                or packet["binding_digest"] != context.binding.to_dict()["binding_digest"]
                or packet["descendant_binding_ref"] != wl._digest(ob._plain(context.descendant_event))
                or packet["boundary_monotonic_s"] > time.monotonic()):
            raise de.DriverExecutionRefused("phase notice differs from actual parent context")
        key = (context.unit_id, packet["phase"])
        prior = self._phase_requests.get(key)
        if prior is not None:
            if prior != packet:
                raise de.DriverExecutionRefused("parent health notice retry conflicts")
            return self._phase_results[key]
        result = {"outcome": "unavailable"}
        if packet["phase"] == "health":
            receipt = producer.capture_runtime_readback(phase="health")
            raw = self._native_store.read(receipt.locator, receipt.sha256)
            result = {"outcome": "captured" if raw["error"] is None else "unavailable"}
        if self.search_window_owner is not None:
            self.search_window_owner.marker(context=context, packet=packet)
            result = {"outcome": "captured"}
        self._phase_requests[key] = ob._freeze(ob._plain(packet))
        self._phase_results[key] = ob._freeze(result)
        return self._phase_results[key]

    def _artifact_completion_for(self, notice: Mapping[str, Any]):
        producer = self._producer_for(notice)
        result = producer.evaluate(notice["request"])
        if self.search_window_owner is not None:
            reference = notice["request"]["native_observation"]
            native = self._native_store.read(reference["locator"], reference["sha256"])
            self.search_window_receipts[producer.context.unit_id] = self.search_window_owner.seal(
                context=producer.context, native=native, parent_result=result)
        self.registry.record(producer=producer, request=notice["request"], result=result)
        return result.completion

    def stop_and_join(self, timeout: float = 2.0) -> None:
        try:
            super().stop_and_join(timeout)
        finally:
            if self.stopped:
                self._native_store.close()
