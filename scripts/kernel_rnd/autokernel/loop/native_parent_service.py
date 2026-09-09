"""Concrete bounded parent service: owned facts in, no scientific policy out."""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping

from . import driver_execution as de
from . import lifecycle_observation as lo
from . import measurement_capture as mc
from . import native_parent_evidence as npe
from . import native_parent_receipt_replay as replay
from . import observation_binding as ob
from . import unified_worker as uw
from . import worker_lifecycle as wl


@dataclass(frozen=True)
class NativeFactualEvidenceConfiguration:
    """Explicit factual mode; not eligibility, correctness, purpose, or GPU authority."""
    schema: str = "epyc.autokernel.native_factual_evidence_configuration.v1"
    scientific_adapters: Any = None

    def __post_init__(self) -> None:
        from .native_scientific_witness import ParentScientificWitnessAdapters
        if self.schema != "epyc.autokernel.native_factual_evidence_configuration.v1":
            raise de.DriverExecutionRefused("unsupported factual evidence configuration")
        if self.scientific_adapters is not None and type(self.scientific_adapters) is not ParentScientificWitnessAdapters:
            raise de.DriverExecutionRefused("scientific configuration requires its concrete registry")


class NativeParentEvidenceService(de.UnknownParentEvidenceProducer):
    def __init__(self, authority: uw.ParentUnitEvidenceAuthority,
                 prepared: uw.PreparedPlannedServingStage, lifecycle: wl.WorkerLifecycle,
                 observation_configuration: ob.ParentObservationConfiguration, *,
                 registry: replay.IssuedNativeEvidenceRegistry,
                 runtime_probe: lo.FilesystemProbe | None = None,
                 scientific_adapters: Any = None) -> None:
        super().__init__(authority, prepared, lifecycle, observation_configuration)
        if (prepared.schema != uw.PREPARED_SCHEMA_V2
                or type(registry) is not replay.IssuedNativeEvidenceRegistry
                or lifecycle is None or observation_configuration is None):
            raise de.DriverExecutionRefused("native factual service requires v2 parent authority")
        if runtime_probe is not None and type(runtime_probe) is not lo.FilesystemProbe:
            raise de.DriverExecutionRefused("native readback requires the concrete bounded reader")
        self.registry = registry
        self.scientific_adapters = NativeFactualEvidenceConfiguration(
            scientific_adapters=scientific_adapters).scientific_adapters
        self._native_store = mc.ArtifactStore(prepared.artifact_root)
        self._native_probe = runtime_probe
        self._unit_producers: dict[str, npe.NativeUnitEvidenceProducer] = {}
        self._phase_requests: dict[str, Mapping[str, Any]] = {}
        self._phase_results: dict[str, Mapping[str, Any]] = {}

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
        if (packet["phase"] != "health"
                or packet["binding_digest"] != context.binding.to_dict()["binding_digest"]
                or packet["descendant_binding_ref"] != wl._digest(ob._plain(context.descendant_event))
                or packet["boundary_monotonic_s"] > time.monotonic()):
            raise de.DriverExecutionRefused("phase notice differs from actual parent context")
        prior = self._phase_requests.get(context.unit_id)
        if prior is not None:
            if prior != packet:
                raise de.DriverExecutionRefused("parent health notice retry conflicts")
            return self._phase_results[context.unit_id]
        receipt = producer.capture_runtime_readback(phase="health")
        raw = self._native_store.read(receipt.locator, receipt.sha256)
        result = {"outcome": "captured" if raw["error"] is None else "unavailable"}
        self._phase_requests[context.unit_id] = ob._freeze(ob._plain(packet))
        self._phase_results[context.unit_id] = ob._freeze(result)
        return self._phase_results[context.unit_id]

    def _artifact_completion_for(self, notice: Mapping[str, Any]):
        producer = self._producer_for(notice)
        result = producer.evaluate(notice["request"])
        self.registry.record(producer=producer, request=notice["request"], result=result)
        return result.completion

    def stop_and_join(self, timeout: float = 2.0) -> None:
        try:
            super().stop_and_join(timeout)
        finally:
            if self.stopped:
                self._native_store.close()
