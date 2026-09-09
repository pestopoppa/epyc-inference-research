"""Replay original parent-issued factual evidence; hashes alone do not prove issuance."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any, Mapping

from . import lifecycle_observation as lo
from . import measurement_capture as mc
from . import native_parent_evidence as npe
from . import observation_binding as ob
from . import worker_lifecycle as wl


class ParentReceiptRefused(mc.CaptureError):
    pass


@dataclass(frozen=True)
class _Issued:
    context: npe.ParentUnitContext
    request: Mapping[str, Any]
    result: npe.NativeUnitEvidenceResult
    body: Mapping[str, Any]
    native_validation_source_pins: Mapping[str, Any]
    scientific_adapters: Any = None


def _native_validation_pins() -> Mapping[str, Any]:
    # Original loaded validation identity retained by this parent. This memory-only
    # pin is not relabelled as a durable field of the published carrier.
    from . import native_capture_control as nc
    return ob._freeze({
        "module_artifact": lo.prepare_artifact_identity(Path(nc.__file__)),
        "callables": {function.__qualname__: lo.callable_identity(function) for function in (
            nc.NativeCaptureValidator.prevalidate, nc.NativeCaptureValidator._verify_artifacts,
            nc.NativeCaptureValidator._verify_observations,
            nc.NativeCaptureValidator.validate_prevalidated)}})


class IssuedNativeEvidenceRegistry:
    """Parent memory only, bounded to one prepared attempt; never restored from child JSON."""
    def __init__(self, *, artifact_root: Path, max_units: int) -> None:
        if type(max_units) is not int or not 1 <= max_units <= 1024:
            raise ParentReceiptRefused("issued evidence registry bound is invalid")
        self.artifact_root = Path(artifact_root).absolute()
        self.max_units = max_units
        self._lock = threading.Lock()
        self._entries: dict[tuple[Any, ...], _Issued] = {}

    @staticmethod
    def _key(plan_digest: str, unit_id: str, fence_id: str,
             worker_id: str, incarnation: int) -> tuple[Any, ...]:
        return plan_digest, unit_id, fence_id, worker_id, incarnation

    def record(self, *, producer: npe.NativeUnitEvidenceProducer,
               request: Mapping[str, Any], result: npe.NativeUnitEvidenceResult) -> None:
        if (type(producer) is not npe.NativeUnitEvidenceProducer
                or type(result) is not npe.NativeUnitEvidenceResult
                or producer._result is not result
                or producer._request_digest != request.get("request_digest")
                or producer.store.root != self.artifact_root):
            raise ParentReceiptRefused("issuance must come from the actual evaluated parent producer")
        context = producer.context
        key = self._key(context.plan.digest, context.unit_id, context.fence.fence_id,
                        context.fence.worker_id, context.fence.worker_incarnation)
        body = producer.store.read(result.receipt.locator, result.receipt.sha256)
        if wl._digest(ob._plain(body)) != result.receipt_digest:
            raise ParentReceiptRefused("parent evidence receipt digest differs")
        producer.store.verify(f"parent-unit-evidence:{result.receipt_digest}", body)
        entry = _Issued(context, ob._freeze(ob._plain(request)), result,
                        ob._freeze(ob._plain(body)), _native_validation_pins(),
                        producer.scientific_adapters)
        with self._lock:
            prior = self._entries.get(key)
            if prior is not None and prior != entry:
                raise ParentReceiptRefused("same parent unit issuance conflicts")
            if prior is None and len(self._entries) >= self.max_units:
                raise ParentReceiptRefused("issued evidence registry capacity exhausted")
            self._entries[key] = entry

    def _lookup(self, carrier: Mapping[str, Any], native: Mapping[str, Any]) -> _Issued:
        context = carrier["capture_context"]
        key = self._key(wl._digest(ob._plain(carrier["plan"])), native["unit_id"],
                        native["fence_id"], context["worker_id"], context["worker_incarnation"])
        with self._lock:
            found = self._entries.get(key)
        if found is None:
            raise ParentReceiptRefused("original parent issuance is unavailable")
        return found


class NativeParentReceiptReplayer:
    """Concrete replayer scoped by its owner to one retained original registry."""
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._registry: IssuedNativeEvidenceRegistry | None = None

    @contextmanager
    def using(self, registry: IssuedNativeEvidenceRegistry):
        if type(registry) is not IssuedNativeEvidenceRegistry:
            raise ParentReceiptRefused("concrete parent-issued registry required")
        with self._lock:
            if self._registry is not None:
                raise ParentReceiptRefused("parent receipt replay scope already active")
            self._registry = registry
            try:
                yield self
            finally:
                self._registry = None

    def replay(self, carrier: Mapping[str, Any], *, store: mc.ArtifactStore) -> None:
        with self._lock:
            registry = self._registry
            if registry is None or registry.artifact_root != store.root:
                raise ParentReceiptRefused("original parent evidence scope is unavailable")
            natives = [item["document"] for item in carrier["raw_artifacts"]
                       if item["document"].get("kind") == "native_observation"]
            attempts = {item["document"]["unit_id"]: item["document"]
                        for item in carrier["raw_artifacts"]
                        if item["document"].get("kind") == "completed_attempt"}
            for native in natives:
                entry = registry._lookup(carrier, native)
                self._replay_unit(entry, native, attempts[native["unit_id"]], store)

    def selected_scientific_adapters(self):
        """Selected original parent configuration, never inferred from child bytes."""
        with self._lock:
            if self._registry is None:
                raise ParentReceiptRefused("original scientific registry scope is unavailable")
            with self._registry._lock:
                entries = tuple(self._registry._entries.values())
            if not entries:
                raise ParentReceiptRefused("original scientific issuance is unavailable")
            first = entries[0].scientific_adapters
            if any(entry.scientific_adapters is not first for entry in entries):
                raise ParentReceiptRefused("original scientific adapter configuration differs by unit")
            return first

    @staticmethod
    def _replay_unit(entry: _Issued, native: Mapping[str, Any], attempt: Mapping[str, Any],
                     store: mc.ArtifactStore) -> None:
        result, context = entry.result, entry.context
        npe._equal(entry.native_validation_source_pins, _native_validation_pins(),
                   "original loaded native validation source")
        body = ob._plain(store.read(result.receipt.locator, result.receipt.sha256))
        npe._equal(body, entry.body, "original parent receipt bytes")
        npe._equal(wl._digest(ob._plain(body)), result.receipt_digest, "parent receipt digest")
        store.verify(f"parent-unit-evidence:{result.receipt_digest}", body)
        npe._equal(body["identity"], context.identity, "parent receipt identity")
        npe._equal(body["completion_request_digest"], entry.request["request_digest"],
                   "parent receipt completion request")
        npe._equal(body["parent_descendant_event"], context.descendant_event, "issued descendant")
        npe._equal(body["parent_active_claim"], context.active_claim, "issued claim")
        npe._equal(body["parent_observation_binding"], context.binding.to_dict(), "issued binding")
        npe._equal(body["recipe_source_pins"], context.recipe.to_dict(), "issued recipe pins")
        native_ref = npe._artifact(body["native_observation"])
        npe._equal(native_ref.to_dict(), entry.request["native_observation"], "issued native reference")
        reopened = ob._plain(store.read(native_ref.locator, native_ref.sha256))
        npe._equal(reopened, native, "attempt native artifact")
        store.verify(f"raw:{native['artifact_digest']}", reopened)
        producer = npe.NativeUnitEvidenceProducer(store=store, context=context,
                                                  scientific_adapters=entry.scientific_adapters)
        npe._equal(body["producer_supporting_pins"], producer._source_pins, "producer supporting pins")
        npe._equal(body["producer_source_pin"], lo.callable_identity(type(producer).evaluate),
                   "producer implementation pin")
        producer._join_native(reopened)
        reference = ob.LifecycleObservationReference.from_dict(body["lifecycle_observation"])
        npe._equal(reference.to_dict(), native["lifecycle_observation"], "issued lifecycle reference")
        link = ob.validate_reopened_observation(reference, store=store,
            expected={"unit_id": context.unit_id, "process_generation_id": context.unit.process_id,
                "fence_id": context.fence.fence_id, "active_claim_ref": context.binding.active_claim_ref,
                "container_id": context.binding.container_id,
                "capture_context": ob._plain(context.binding.worker_binding)},
            instrument=context.binding.instrument)
        observation = lo.validate_observation(ob._plain(store.read(
            reference.artifact.locator, reference.artifact.sha256)))
        producer._join_observation(native, observation, reference)
        producer._readbacks = [npe._artifact(item) for item in body["runtime_readbacks"]]
        # These refs came from original issuance, never from this child's carrier.
        for ref in producer._readbacks:
            raw = store.read(ref.locator, ref.sha256)
            store.verify(f"parent-status-readback:{wl._digest(ob._plain(raw))}", raw)
        requests_ok, requests = producer._requests(native)
        coverage_ok, coverage = producer._coverage(observation)
        npe._equal(body["phase_coverage"], coverage, "replayed phase coverage")
        findings = body["findings"]
        npe._equal(findings["request_completeness"]["facts"], requests, "replayed request facts")
        npe._equal(findings["request_completeness"]["status"],
                   "pass" if requests_ok else "fail", "replayed request status")
        npe._equal(findings["placement"], producer._placement(observation, coverage_ok),
                   "replayed placement facts")
        npe._equal(findings["runtime_readback"], producer._runtime(observation),
                   "replayed runtime facts")
        if producer.scientific_adapters is not None:
            scientific = producer.scientific_adapters.findings(context, native, link,
                tuple(producer._readbacks), store=store)
            for name, finding in scientific.items():
                npe._equal(findings[name], finding, "replayed original scientific finding")
        expected = {name: witness.to_dict() for name, witness
                    in result.completion.stage_witnesses.items()}
        npe._equal(attempt["stage_witnesses"], expected, "issued whole witness map")
        for name, witness in expected.items():
            npe._equal(findings[name]["status"], witness["status"], "issued witness finding")
            if witness["status"] != "unknown":
                npe._equal(witness["ref"], f"parent-unit-evidence:{result.receipt_digest}#{name}",
                           "issued witness receipt reference")
        npe._equal(attempt["provider_recorded_screen"], result.completion.recorded_screen,
                   "issued provider screen")
        if (attempt["terminal"] and not result.completion.terminal
                or attempt["recorded_screen"] == "clean"
                   and result.completion.recorded_screen != "clean"):
            raise ParentReceiptRefused("child upgraded its parent completion")


def has_parent_receipt_refs(carrier: Mapping[str, Any]) -> bool:
    return any(isinstance(witness.get("ref"), str)
               and witness["ref"].startswith("parent-unit-evidence:")
               for item in carrier.get("raw_artifacts", ())
               for witness in item.get("document", {}).get("stage_witnesses", {}).values())
