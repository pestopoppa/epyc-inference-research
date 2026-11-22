"""Connect one durably selected runtime comparison to the owned worker bridge.

This module supplies no grant or scientific authority.  The lifecycle owns process
execution and provider receipts; the controller owns both native writes and scheduler
settlement.  In the absence of a registered scientific evaluator, completed native
observations are charged as ``invalid`` rather than promoted to comparisons.
"""
from __future__ import annotations

import copy
import json
import queue
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from . import restart_native_citations as citations
from . import campaign_control
from . import experiment_plan as ep
from . import measurement_capture as mc
from . import native_capture_control as nc
from . import observation_binding as ob
from . import operation_accounting as oa
from . import planned_serving as ps
from . import scheduling
from . import unified_driver
from . import unified_worker
from . import worker_lifecycle
# re-exported: moved verbatim to driver_execution_receipt.py
from .driver_execution_receipt import (  # noqa: F401
    DriverExecutionReceipt, DriverExecutionRefused, EXECUTION_RECEIPT_SCHEMA,
    EXECUTION_RECEIPT_SCHEMA_V2, _sha, _text,
)


class DriverExecutionUncertain(RuntimeError):
    """An owned attempt started and must be reconciled, never launched again."""


class RetainedResultFenced(DriverExecutionRefused):
    """AKU-07h: a restored ACCEPTED result cannot be reopened; its transition stays fenced."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"retained result fenced: {detail}")
        self.code = code  # one of unified_worker.RETAINED_RESULT_FENCES


def _terminal_body(value: worker_lifecycle.TerminalWorker) -> dict[str, Any]:
    if not isinstance(value, worker_lifecycle.TerminalWorker):
        raise DriverExecutionRefused("lifecycle returned an untyped terminal")
    return {
        "worker_id": value.worker_id, "worker_generation": value.worker_generation,
        "request_id": value.request_id, "plan_digest": value.plan_digest,
        "lineage_id": value.lineage_id, "stage_id": value.stage_id,
        "grant_id": value.grant_id, "grant_generation": value.grant_generation,
        "container_id": value.container_id, "return_code": value.return_code,
        "result_digest": value.result_digest, "accepted": value.accepted,
        "reason": value.reason,
    }


class UnknownParentEvidenceProducer:
    """Bounded producer that preserves observations while granting no witness pass.

    A future registered evaluator can replace this concrete producer.  This default
    makes successful transport terminal, but every required witness remains unknown,
    so the result is never a valid scientific comparison.
    """

    def __init__(self, authority: unified_worker.ParentUnitEvidenceAuthority,
                 prepared: unified_worker.PreparedPlannedServingStage,
                 lifecycle: worker_lifecycle.WorkerLifecycle | None = None,
                 observation_configuration: ob.ParentObservationConfiguration | None = None
                 ) -> None:
        if not isinstance(authority, unified_worker.ParentUnitEvidenceAuthority):
            raise DriverExecutionRefused("parent evidence cache is not the fixed bounded type")
        self.authority = authority
        self.prepared = unified_worker.PreparedPlannedServingStage.from_dict(
            prepared.to_dict())
        self.plan = self.prepared.plan
        if lifecycle is not None and not isinstance(lifecycle, worker_lifecycle.WorkerLifecycle):
            raise DriverExecutionRefused("observation lifecycle owner is untyped")
        if (observation_configuration is not None
                and not isinstance(observation_configuration,
                                   ob.ParentObservationConfiguration)):
            raise DriverExecutionRefused("observation configuration is untyped")
        self.lifecycle = lifecycle
        self.observation_configuration = observation_configuration
        if self.plan.schema == ep.PLAN_SCHEMA_V2 and observation_configuration is not None:
            ob.validate_planned_sample_capacity(observation_configuration,
                max_stage_seconds=self.prepared.max_stage_seconds,
                teardown_seconds=self.prepared.teardown_seconds)
        self._bindings: dict[tuple[str, int, str, str, str],
                             ob.ObservationUnitBinding] = {}
        self._targets: dict[tuple[str, int, str, str, str], Mapping[str, Any]] = {}
        self._claims: dict[tuple[str, int, str, str, str], Mapping[str, Any]] = {}
        self._stop = threading.Event()
        self._errors: list[BaseException] = []
        self._thread = threading.Thread(target=self._run, name="autokernel-parent-evidence",
                                        daemon=False)

    def start(self) -> None:
        self._thread.start()

    def _completion_for(self, fence: ps.StageFence,
                        observation: Mapping[str, Any]) -> ps.StageCompletion:
        del observation
        witnesses = MappingProxyType({
            name: ep.Witness("unknown", None)
            for name in self.plan.required_witnesses})
        return ps.StageCompletion(
            fence.fence_id, True, witnesses, "flagged_but_retained",
            "registered parent observation evaluator is unavailable")

    def _record_target(self, binding_key: tuple[str, int, str, str, str],
                       binding: ob.ObservationUnitBinding,
                       target: Mapping[str, Any]) -> None:
        del binding
        prior = self._targets.get(binding_key)
        if prior is not None and prior != target:
            raise DriverExecutionRefused(
                "observation target retry conflicts with prior authority")
        self._targets[binding_key] = target

    def _artifact_completion_for(self, notice: Mapping[str, Any]) -> ps.StageCompletion:
        """Default still grants no warrant; even its v2 input must be an actual artifact."""
        packet = notice["request"]
        ref = unified_worker._native_reference(packet["native_observation"])
        store = mc.ArtifactStore(self.prepared.artifact_root)
        try:
            native = store.read(ref["locator"], ref["sha256"])
            native_digest = native["artifact_digest"]
            if native_digest != unified_worker._digest({
                    key: value for key, value in native.items() if key != "artifact_digest"}):
                raise DriverExecutionRefused("native completion artifact digest differs")
            store.verify(f"raw:{native_digest}", native)
            observation = native["selected_observation"]
            if not isinstance(observation, Mapping):
                raise DriverExecutionRefused("native completion observation is malformed")
            return self._completion_for(notice["fence"], observation)
        finally:
            store.close()

    def _phase_for(self, notice: Mapping[str, Any]) -> Mapping[str, Any]:
        del notice
        return {"outcome": "unavailable"}

    def _before_observation_binding(self, *, start: unified_worker.WorkerStart,
                                    unit: ep.UnitSpec, fence: ps.StageFence,
                                    recipe: Any,
                                    claim: Mapping[str, Any]) -> None:
        """Optional parent preparation while the child is blocked on its binding."""
        del start, unit, fence, recipe, claim

    def _binding_ready(self, *, start: unified_worker.WorkerStart,
                       binding: ob.ObservationUnitBinding,
                       claim: Mapping[str, Any]) -> None:
        """Post-preparation, pre-spawn hook on the owning evidence thread."""

    def _poll_live_evidence(self) -> None:
        """At most one bounded collector step; the default performs no I/O."""

    def _evidence_poll_delay(self) -> float:
        return 0.05

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                notice = self.authority.next_notice(timeout=self._evidence_poll_delay())
            except queue.Empty:
                try:
                    self._poll_live_evidence()
                except BaseException as exc:
                    self._errors.append(exc)
                    self._stop.set()
                continue
            try:
                kind = notice.get("kind")
                key = notice.get("key")
                if kind == "completion":
                    fence = notice.get("fence")
                    observation = notice.get("observation")
                    if (not isinstance(key, str) or not isinstance(fence, ps.StageFence)
                            or not isinstance(observation, Mapping)):
                        raise DriverExecutionRefused("completion notice is malformed")
                    self.authority.publish_completion(
                        key, self._completion_for(fence, observation))
                elif kind == "artifact_completion":
                    if (not isinstance(key, str)
                            or not isinstance(notice.get("request"), Mapping)):
                        raise DriverExecutionRefused("artifact completion notice is malformed")
                    self.authority.publish_completion(key, self._artifact_completion_for(notice))
                elif kind == "observation_phase":
                    if not isinstance(key, str):
                        raise DriverExecutionRefused("observation phase notice is malformed")
                    self.authority.publish_observation_phase(key, self._phase_for(notice))
                elif kind == "continuation":
                    if not isinstance(key, str):
                        raise DriverExecutionRefused("continuation notice is malformed")
                    self.authority.publish_continuation(key, False)
                elif kind == "observation_binding":
                    start = notice.get("start")
                    unit = notice.get("unit")
                    fence = notice.get("fence")
                    recipe_digest = notice.get("recipe_identity_digest")
                    if (not isinstance(key, str)
                            or not isinstance(start, unified_worker.WorkerStart)
                            or not isinstance(unit, ep.UnitSpec)
                            or not isinstance(fence, ps.StageFence)
                            or not isinstance(recipe_digest, str)
                            or self.lifecycle is None
                            or self.observation_configuration is None):
                        raise DriverExecutionRefused(
                            "observation binding notice lacks parent authority")
                    recipe = (self.prepared.runtime_pair.anchor
                              if unit.arm == "anchor"
                              else self.prepared.runtime_pair.candidate)
                    if recipe.execution_digest != recipe_digest:
                        raise DriverExecutionRefused(
                            "observation binding recipe identity differs")
                    configuration = self.observation_configuration
                    requested = configuration.requested_effective_states.get(recipe_digest)
                    if requested is None:
                        raise DriverExecutionRefused(
                            "observation recipe configuration is unavailable")
                    claim = self.lifecycle.describe_active_observation_claim(
                        start=start, unit_id=unit.unit_id,
                        process_generation_id=unit.process_id,
                        deadline=fence.valid_until)
                    held = claim["held_claim"]
                    if list(requested["logical_cpus"]) != list(held["logical_cpus"]):
                        raise DriverExecutionRefused(
                            "requested CPU placement differs from the active held claim")
                    required_dsos = configuration.required_gpu_dsos.get(recipe_digest, ())
                    if (recipe.backend == "cpu" and (held["gpu_devices"] or required_dsos)) \
                            or (recipe.backend == "gpu"
                                and (not held["gpu_devices"] or not required_dsos)):
                        raise DriverExecutionRefused(
                            "observation backend differs from held GPU/DSO evidence")
                    self._before_observation_binding(
                        start=start, unit=unit, fence=fence, recipe=recipe, claim=claim)
                    instrument = ob.LoadedInstrumentReference.from_dict(
                        unified_worker._plain(self.plan.loaded_instrument))
                    worker_binding = {
                        "worker_id": start.worker_id,
                        "worker_incarnation": start.worker_generation,
                        "grant_id": start.grant_id,
                        "grant_generation": start.grant_generation,
                        "container_identity": unified_worker._plain(start.cgroup_identity)}
                    observation_id = "observation-" + unified_worker._digest({
                        "start": start.to_dict(), "sequence": notice.get("sequence"),
                        "unit": unit.to_dict(), "fence_id": fence.fence_id,
                        "claim_digest": claim["claim_digest"]})
                    binding = ob.ObservationUnitBinding.from_dict(
                        ob.ObservationUnitBinding(
                            observation_id, unit.unit_id, unit.process_id,
                            fence.fence_id, start.clock_domain,
                            start.child_process.boot_id, worker_binding,
                            start.container_id, claim["active_claim_ref"], held,
                            requested,
                            tuple(key for key, _value in recipe.readback_expectations),
                            tuple(required_dsos), configuration.cadence_s,
                            configuration.gap_limit_s, configuration.budgets,
                            instrument).to_dict())
                    binding_key = (start.worker_id, start.worker_generation,
                                   unit.unit_id, unit.process_id, fence.fence_id)
                    prior = self._bindings.get(binding_key)
                    if prior is not None and prior != binding:
                        raise DriverExecutionRefused(
                            "observation binding retry conflicts with prior authority")
                    self._bindings[binding_key] = binding
                    prior_claim = self._claims.get(binding_key)
                    if prior_claim is not None and prior_claim != claim:
                        raise DriverExecutionRefused("observation claim retry conflicts")
                    self._claims[binding_key] = claim
                    self._binding_ready(start=start, binding=binding, claim=claim)
                    self.authority.publish_observation_binding(key, binding)
                elif kind == "observation_target":
                    start = notice.get("start")
                    unit = notice.get("unit")
                    fence = notice.get("fence")
                    pid = notice.get("pid")
                    if (not isinstance(key, str)
                            or not isinstance(start, unified_worker.WorkerStart)
                            or not isinstance(unit, ep.UnitSpec)
                            or not isinstance(fence, ps.StageFence)
                            or not isinstance(pid, int) or isinstance(pid, bool)
                            or self.lifecycle is None):
                        raise DriverExecutionRefused("observation target notice is malformed")
                    binding_key = (start.worker_id, start.worker_generation,
                                   unit.unit_id, unit.process_id, fence.fence_id)
                    binding = self._bindings.get(binding_key)
                    if binding is None:
                        raise DriverExecutionRefused(
                            "observation target lacks its exact parent binding")
                    captured = self.lifecycle.capture_owned_descendant(
                        start=start, unit_id=unit.unit_id,
                        process_generation_id=unit.process_id,
                        fence_id=fence.fence_id, pid=pid,
                        binding_digest=binding.to_dict()["binding_digest"])
                    target = {name: captured[name] for name in (
                        "pid", "start_ticks", "boot_id", "worker_binding",
                        "binding_ref")}
                    self._record_target(binding_key, binding, target)
                    self.authority.publish_observation_target(key, target)
                else:
                    raise DriverExecutionRefused("parent evidence notice kind is unsupported")
            except BaseException as exc:
                self._errors.append(exc)
                self._stop.set()

            if not self._stop.is_set():
                try:
                    self._poll_live_evidence()
                except BaseException as exc:
                    self._errors.append(exc)
                    self._stop.set()

    def stop_and_join(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread.ident is None:
            if self._errors:
                raise DriverExecutionUncertain(
                    f"parent evidence producer failed: {self._errors[0]}") \
                    from self._errors[0]
            return
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise DriverExecutionUncertain(
                "parent evidence producer did not stop; successor execution is fenced")
        if self._errors:
            raise DriverExecutionUncertain(
                f"parent evidence producer failed: {self._errors[0]}") from self._errors[0]

    @property
    def stopped(self) -> bool:
        return self._thread.ident is None or not self._thread.is_alive()


@dataclass
class _FinishedAttempt:
    catalog_id: str
    transition_id: str
    selection: scheduling.Selection
    prepared: unified_worker.PreparedPlannedServingStage
    start: unified_worker.WorkerStart
    terminal: worker_lifecycle.TerminalWorker
    reference: unified_worker.PlannedWorkerResultReference
    fence: nc.TrustedWorkerResultFence
    held: scheduling.HeldClaimReceipt


@dataclass
class _FailedAttempt:
    catalog_id: str
    transition_id: str
    selection: scheduling.Selection
    prepared: unified_worker.PreparedPlannedServingStage
    terminal: worker_lifecycle.TerminalWorker
    held: scheduling.HeldClaimReceipt
    reference: unified_worker.PlannedWorkerResultReference | None = None
    recovered: bool = False
    restored: bool = False  # AKU-07j: from the controller's accounting-only held index


class UnifiedDriverExecution:
    """Single-owner selected-runtime executor with exact retry boundaries."""

    def __init__(self, *, driver: unified_driver.UnifiedCampaignDriver,
                 controller: Any,
                 observation_verifiers: ob.ParentObservationVerifiers =
                 ob.ParentObservationVerifiers(),
                 observation_configuration: ob.ParentObservationConfiguration | None = None,
                 native_evidence_configuration: Any | None = None,
                 profile_executor: Any | None = None,
                 operation_settlement: bool = False
                 ) -> None:
        if (not isinstance(driver, unified_driver.UnifiedCampaignDriver)
                or not isinstance(controller, campaign_control.CampaignController)):
            raise DriverExecutionRefused(
                "driver/controller must be the current typed owners")
        if driver.controller is not controller:
            raise DriverExecutionRefused("driver/controller ownership differs")
        if not isinstance(observation_verifiers, ob.ParentObservationVerifiers):
            raise DriverExecutionRefused(
                "observation verifiers must be the concrete parent adapter set")
        if (observation_configuration is not None
                and not isinstance(observation_configuration,
                                   ob.ParentObservationConfiguration)):
            raise DriverExecutionRefused(
                "observation configuration must be the concrete parent type")
        self.driver = driver
        self.controller = controller
        if profile_executor is not None:
            from .profile_preparation import InstalledProfilePreparationOwner
            if (type(profile_executor) is not InstalledProfilePreparationOwner
                    or profile_executor.controller is not controller):
                raise DriverExecutionRefused("profile executor is not this concrete owner")
        self._profile_executor = profile_executor
        # AKU-06l: settle single-stage work as one durable child + SETTLED.v2 (opt-in;
        # off keeps the exact v1 settlement bytes).
        if type(operation_settlement) is not bool:
            raise DriverExecutionRefused("operation_settlement must be a bool")
        self._operation_settlement = operation_settlement
        if profile_executor is not None:
            profile_executor.operation_settlement = operation_settlement
        self._launched: set[str] = set()
        self._finished: dict[str, _FinishedAttempt | _FailedAttempt] = {}
        self._trusted_settlements: dict[str, dict[str, Any]] = {}
        self._receipts: dict[str, DriverExecutionReceipt] = {}
        self._capture_store: mc.ArtifactStore | None = None
        self._active_fence: nc.TrustedWorkerResultFence | None = None
        self._observation_verifiers = observation_verifiers
        self._observation_configuration = observation_configuration
        if native_evidence_configuration is not None:
            from .native_parent_service import NativeFactualEvidenceConfiguration
            from .native_parent_receipt_replay import NativeParentReceiptReplayer
            if type(native_evidence_configuration) is not NativeFactualEvidenceConfiguration:
                raise DriverExecutionRefused("native factual evidence configuration is untyped")
            self._parent_receipt_replayer = NativeParentReceiptReplayer()
        else:
            self._parent_receipt_replayer = None
        self._native_evidence_configuration = native_evidence_configuration
        self._parent_evidence_registries: dict[str, Any] = {}
        self._lock = threading.RLock()
        self._closed = False
        self._admission_closed = False
        self._successor_fence: str | None = None
        self._unresolved_producers: list[UnknownParentEvidenceProducer] = []
        try:
            controller.register_unified_settlement_validator(self._verify_settlement)
        except Exception as exc:
            raise DriverExecutionRefused(
                f"controller settlement authority is unavailable: {exc}") from exc

    def _verify_settlement(self, supplied: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(supplied, Mapping):
            raise DriverExecutionRefused("settlement candidate is malformed")
        if supplied.get("schema") == campaign_control.DRIVER_SETTLEMENT_SCHEMA_V2:
            return self._verify_operation_settlement(supplied)
        return self._verify_v1(supplied)

    def _verify_v1(self, supplied: Mapping[str, Any]) -> Mapping[str, Any]:
        transition_id = supplied.get("transition_id")
        if (self._profile_executor is not None
                and self._profile_executor.owns_settlement(transition_id)):
            with self._lock:
                if self._closed:
                    raise DriverExecutionRefused("driver execution connector is closed")
            return self._profile_executor.verify_settlement(supplied)
        with self._lock:
            if self._closed:
                raise DriverExecutionRefused("driver execution connector is closed")
            trusted = copy.deepcopy(self._trusted_settlements.get(transition_id))
            attempt = self._finished.get(transition_id)
        if trusted is None or attempt is None or dict(supplied) != trusted:
            raise DriverExecutionRefused("settlement differs from the owned finished attempt")
        if isinstance(attempt, _FailedAttempt) and attempt.restored:
            restored = self.controller.restored_held_settlement(
                catalog_id=attempt.catalog_id, transition_id=transition_id,
                selection=attempt.selection.to_dict(), plan_digest=attempt.terminal.plan_digest)
            if restored is None or restored[0] != attempt.terminal:
                raise DriverExecutionRefused("restored held accounting differs")
            held = restored[1]
        else:
            held = self.controller.worker_held_claim_receipt(attempt.terminal)
        if held.to_dict() != trusted["receipt"]:
            raise DriverExecutionRefused("trusted held receipt differs")
        if isinstance(attempt, _FinishedAttempt):
            fence = self.controller.worker_result_fence(attempt.terminal)
            if (not fence.current or not fence.result_accepted
                    or unified_worker._digest(attempt.reference.to_dict())
                       != attempt.terminal.result_digest):
                raise DriverExecutionRefused("terminal/result authority is stale or differs")
            for measurement_id in trusted["terminal_refs"][:-1]:
                if measurement_id.startswith("calibration-collected:"):
                    from . import serving_preparation as preparation
                    if (attempt.prepared.schema != unified_worker.PREPARED_SCHEMA_V3
                            or self.driver.preparation_owner is None):
                        raise DriverExecutionRefused("unexpected calibration collection reference")
                    reference = preparation.CollectedCalibrationReference.from_dict(json.loads(
                        measurement_id.removeprefix("calibration-collected:")))
                    request = preparation.CalibrationPreparationDispatch.from_dict(
                        attempt.prepared.dispatch).request
                    outcome = self.driver.preparation_owner.collection_outcome(reference, request=request)
                    if outcome != trusted["outcome"]:
                        raise DriverExecutionRefused("raw collection and settlement dispositions differ")
                    continue
                if not measurement_id.startswith("native:"):
                    raise DriverExecutionRefused("native terminal reference is malformed")
                native_id = measurement_id.removeprefix("native:")
                if self.controller.native_capture(native_id) is None:
                    raise DriverExecutionRefused(
                        "native terminal reference is not durably recorded")
            expected_result = f"result:{unified_worker._digest(attempt.reference.to_dict())}"
            if trusted["terminal_refs"][-1] != expected_result:
                raise DriverExecutionRefused("sealed result terminal reference differs")
        else:
            terminal_ref = f"lifecycle:{unified_driver._digest(_terminal_body(attempt.terminal))}"
            recovered = (attempt.terminal if attempt.recovered else
                         self.controller.worker_terminal_for_request(
                             request_id=attempt.terminal.request_id,
                             plan_digest=attempt.terminal.plan_digest,
                             lineage_id=attempt.terminal.lineage_id,
                             stage_id=attempt.terminal.stage_id))
            if recovered != attempt.terminal or trusted["terminal_refs"] != [
                    terminal_ref, *citations.refs(self.controller, attempt)]:
                raise DriverExecutionRefused("failed lifecycle terminal differs")
            if attempt.reference is not None and attempt.restored:
                if (self._reopen_retained(attempt.prepared, attempt.terminal)
                        != (attempt.prepared, attempt.reference)):
                    raise DriverExecutionRefused("restored retained result differs")
            elif attempt.reference is not None:
                fence = self.controller.worker_result_fence(attempt.terminal)
                if (not attempt.terminal.accepted or not fence.current
                        or not fence.result_accepted
                        or unified_worker._digest(attempt.reference.to_dict())
                           != attempt.terminal.result_digest):
                    raise DriverExecutionRefused(
                        "operational failure terminal/result differs")
        return copy.deepcopy(trusted)

    def _verify_operation_settlement(self, supplied: Mapping[str, Any]) -> Mapping[str, Any]:
        """AKU-06l: the sole registered validator for a v2 operation settlement.

        Reads only controller-owned durable state: the exact ordered child prefix, the
        closed role/disposition rules, and each current-lifetime child's lifecycle terminal
        and provider-held receipt. A v1 caller receipt is never accepted here."""
        with self._lock:
            if self._closed:
                raise DriverExecutionRefused("driver execution connector is closed")
        transition_id = supplied.get("transition_id")
        children = self.controller.unified_driver_children(transition_id)
        if [child["digest"] for child in children] != list(supplied.get("children", ())):
            raise DriverExecutionRefused("operation settlement children differ from the durable prefix")
        enrollment = supplied.get("enrollment_ref")
        try:
            oa.check_outcome_against_children(
                supplied.get("outcome"), children, supplied.get("denial_ref"),
                enrollment_ref=enrollment, uncertain=len(
                    self.controller.unified_driver_intent_resolutions(transition_id)))
        except oa.OperationAccountingRefused as exc:
            raise DriverExecutionRefused(f"operation settlement refused: {exc}") from exc
        if (supplied.get("outcome") == "prerequisite" and children
                and children[-1]["operation"] == "source_build" and enrollment is None):
            raise DriverExecutionRefused(
                "source-build prerequisite requires its enrollment reference")
        if oa.single_stage(children):
            # The one child carries the original receipt; the owner's exact v1 attempt
            # checks (terminal, result, native and profile references) still decide.
            candidate = {"schema": campaign_control.DRIVER_SETTLEMENT_SCHEMA,
                         "receipt": children[0]["receipt"],
                         "terminal_refs": supplied.get("evidence_refs"),
                         **{name: supplied.get(name) for name in (
                             "catalog_id", "transition_id", "selection", "outcome")}}
            if dict(self._verify_v1(candidate)) != candidate:
                raise DriverExecutionRefused("single-stage owner attempt differs")
        for child in children:
            terminal = worker_lifecycle.TerminalWorker(**child["terminal"])
            exact, held = self.controller.operation_child_proof(  # AKU-06l3: or restored index
                terminal, child["binding_digest"])
            if exact is None and (child["supervisor_incarnation"]
                                  < self.controller.supervisor_incarnation):
                continue  # a prior lifetime's child: its durable verified row is the proof
            if exact != terminal or held is None or held.to_dict() != child["receipt"]:
                raise DriverExecutionRefused(
                    "operation child terminal/held receipt is not the current lifecycle proof")
        return copy.deepcopy(dict(supplied))

    def _install_capture_validator(self, prepared: unified_worker.PreparedPlannedServingStage) \
            -> None:
        if self._capture_store is not None:
            if self._capture_store.root != prepared.artifact_root:
                raise DriverExecutionRefused("driver artifact root changed within one owner")
            return
        base = prepared.capture_context_base
        self._capture_store = mc.ArtifactStore(prepared.artifact_root)
        validator = nc.NativeCaptureValidator(
            binding=nc.NativeCaptureBinding(
                base["campaign_id"], base["config_digest"], base["config_generation"],
                base["supervisor_id"], base["supervisor_incarnation"]),
            store=self._capture_store,
            fence_provider=lambda _measurement_id, _context: self._active_fence,
            observation_verifiers=self._observation_verifiers,
            parent_receipt_replayer=self._parent_receipt_replayer)
        try:
            self.controller.register_native_capture(validator)
        except Exception:
            self._capture_store.close()
            self._capture_store = None
            raise

    def execute(self, outcome: unified_driver.DriverOutcome) -> DriverExecutionReceipt:
        with self._lock:
            if self._closed:
                raise DriverExecutionRefused("driver execution connector is closed")
            if (self._admission_closed
                    and outcome.transition_id not in self._receipts
                    and outcome.transition_id not in self._finished):
                raise DriverExecutionUncertain(
                    "driver execution admission is closed during producer teardown")
            return self._execute_locked(outcome)

    def recover_issued(self, outcome: unified_driver.DriverOutcome) -> DriverExecutionReceipt:
        """Reconcile and resume one exact restored durable issue without reselection."""
        with self._lock:
            if self._closed:
                raise DriverExecutionRefused("driver execution connector is closed")
            transition_id = outcome.transition_id
            if transition_id is None or self.driver.issued_work_kind(outcome) not in {
                    "runtime_comparison", "calibration_preparation"}:
                raise DriverExecutionRefused("recovery requires an exact restored serving intent")
            prepared = self._materialize_serving(outcome)
            selection = scheduling.Selection.from_dict(outcome.selection)
            if selection.proposal is None or self.driver._issued_catalog is None:
                raise DriverExecutionRefused("restored runtime selection is incomplete")
            request_id = selection.proposal.proposal_id
            lineage_id = f"driver:{transition_id}"
            stage_id = f"runtime:{transition_id}"
            terminal = self.controller.reconcile_workers()
            restored = None
            if terminal is None:
                # AKU-07j: a durable stale terminal + held record of this exact selection
                # settles once from accounting-only indexes; nothing is reacquired or rerun.
                restored = self.controller.restored_held_settlement(
                    catalog_id=self.driver._issued_catalog.catalog_id,
                    transition_id=transition_id, selection=selection.to_dict(),
                    plan_digest=prepared.plan.digest)
                terminal = None if restored is None else restored[0]
            if terminal is not None:
                if (terminal.request_id != request_id
                        or terminal.plan_digest != prepared.plan.digest
                        or terminal.lineage_id != lineage_id
                        or terminal.stage_id != stage_id):
                    raise DriverExecutionUncertain(
                        "reconciled lifecycle belongs to another issued request")
                try:
                    held = (restored[1] if restored is not None
                            else self.controller.worker_held_claim_receipt(terminal))
                except Exception as exc:
                    raise worker_lifecycle.WaitingAuthority(
                        "reconciled terminal lacks a trusted held receipt") from exc
                # AKU-07h: a durably ACCEPTED result is reopened by digest and retained
                # diagnostically; an old-incarnation result is never natively accepted.
                reference = None
                if restored is not None and terminal.accepted:
                    prepared, reference = self._reopen_retained(prepared, terminal)
                failed = _FailedAttempt(
                    self.driver._issued_catalog.catalog_id, transition_id, selection,
                    prepared, terminal, held, reference, recovered=True,
                    restored=restored is not None)
                self._launched.add(transition_id)
                self._finished[transition_id] = failed
                return self._finish(failed)
            status = self.controller.worker_attempt_status(
                request_id=request_id, plan_digest=prepared.plan.digest,
                lineage_id=lineage_id, stage_id=stage_id)
            if status == "not_acquired":
                return self._execute_locked(outcome)
            raise worker_lifecycle.WaitingAuthority(
                f"restored lifecycle ownership is {status}; exact recovery remains fenced")

    def _reopen_retained(self, prepared, terminal):
        try:
            return unified_worker.reopen_retained_result(
                prepared=prepared, terminal=terminal,
                durable=self.controller.restored_retained_acceptance(terminal))
        except unified_worker.RetainedResultUnavailable as exc:
            raise RetainedResultFenced(exc.code, str(exc)) from exc

    def _stop_producer(self, producer: UnknownParentEvidenceProducer) \
            -> BaseException | None:
        error = None
        try:
            producer.stop_and_join()
        except BaseException as exc:
            error = exc
        if producer.stopped:
            self._unresolved_producers = [
                item for item in self._unresolved_producers if item is not producer]
            return error
        if all(item is not producer for item in self._unresolved_producers):
            self._unresolved_producers.append(producer)
        self._successor_fence = (
            "parent evidence producer remains alive; successor execution is fenced")
        return error or DriverExecutionUncertain(self._successor_fence)

    def _materialize_serving(self, outcome: unified_driver.DriverOutcome):
        kind = self.driver.issued_work_kind(outcome)
        if kind == unified_driver.CALIBRATION_WORK_KIND:
            return self.driver.materialize_calibration(outcome)
        if kind != "runtime_comparison":
            raise DriverExecutionRefused("selected work has no installed serving executor")
        return self.driver.materialize_runtime(outcome)

    def _execute_locked(self, outcome: unified_driver.DriverOutcome) \
            -> DriverExecutionReceipt:
        transition_id = outcome.transition_id
        if transition_id is None:
            raise DriverExecutionRefused("execution requires a durably issued transition")
        if transition_id in self._receipts:
            stored = self._receipts[transition_id]
            duplicate = self.controller.unified_driver_settle(single_stage_request(
                self.controller, unified_driver._thaw(stored.settlement_request)))
            if duplicate.get("status") != "duplicate":
                raise DriverExecutionUncertain("durable execution retry was not idempotent")
            return stored
        if transition_id in self._finished:
            return self._finish(self._finished[transition_id])
        if transition_id in self._launched:
            raise DriverExecutionUncertain(
                "started lifecycle attempt requires owned reconciliation; it is not rerun")
        if self._successor_fence is not None:
            raise DriverExecutionUncertain(self._successor_fence)
        prepared = self._materialize_serving(outcome)
        if (self._native_evidence_configuration is not None
                and not prepared.native_observed):
            raise DriverExecutionRefused("native factual evidence requires an already-issued v2 plan")
        if prepared.native_observed:
            lifecycle = getattr(self.controller, "_worker_lifecycle", None)
            provider = getattr(self.controller, "_lifecycle_provider", None)
            if (not isinstance(lifecycle, worker_lifecycle.WorkerLifecycle)
                    or self._observation_configuration is None
                    or not callable(getattr(
                        provider, "describe_active_observation_claim", None))):
                raise worker_lifecycle.WaitingAuthority(
                    "v2 parent observation claim/configuration authority is unavailable")
        selection = scheduling.Selection.from_dict(outcome.selection)
        if selection.proposal is None:
            raise DriverExecutionRefused("issued runtime selection lacks a proposal")
        catalog = self.driver._issued_catalog
        if catalog is None:
            raise DriverExecutionRefused("driver no longer owns the issued catalog")
        readiness = self.controller.unified_driver_readiness()
        records_per_unit = (3 + len(unified_worker.OBSERVATION_WINDOW_MARKERS)
                            if prepared.native_observed else 2)
        record_capacity = max(8, len(prepared.plan.expected_units) * records_per_unit)
        if record_capacity > 1024:
            raise DriverExecutionRefused("prepared parent notices exceed the fixed evidence cache bound")
        authority = unified_worker.ParentUnitEvidenceAuthority(max_records=record_capacity)
        if self._native_evidence_configuration is None:
            producer = UnknownParentEvidenceProducer(
                authority, prepared,
                getattr(self.controller, "_worker_lifecycle", None),
                self._observation_configuration)
        else:
            from .native_parent_service import NativeParentEvidenceService
            from .native_parent_receipt_replay import IssuedNativeEvidenceRegistry
            registry = IssuedNativeEvidenceRegistry(
                artifact_root=prepared.artifact_root, max_units=len(prepared.plan.expected_units))
            producer = NativeParentEvidenceService(
                authority, prepared, self.controller._worker_lifecycle,
                self._observation_configuration, registry=registry,
                factual_configuration=self._native_evidence_configuration)
            self._parent_evidence_registries[transition_id] = registry
        invocation = unified_worker.PlannedWorkerInvocation.open(prepared, authority)
        request_id = selection.proposal.proposal_id
        try:
            request = invocation.stage_request(
                request_id=request_id,
                lineage_id=f"driver:{transition_id}", stage_id=f"runtime:{transition_id}",
                control_revision=readiness["control_revision"],
                accounting_binding=(
                    worker_lifecycle.AccountingBinding.from_selection(
                        catalog_id=catalog.catalog_id, transition_id=transition_id,
                        selection=selection)
                    if getattr(self.controller, "snapshot_version", None) == 3 else None))
            producer.start()
        except BaseException:
            self._stop_producer(producer)
            invocation.close()
            raise
        self._launched.add(transition_id)
        terminal = None
        lifecycle_error: BaseException | None = None
        try:
            terminal = self.controller.run_worker_stage(
                request, planned_invocation=invocation)
        except BaseException as exc:
            lifecycle_error = exc
        finally:
            producer_error = self._stop_producer(producer)
            if lifecycle_error is None and producer_error is not None:
                lifecycle_error = producer_error
            invocation.close()
        if lifecycle_error is not None:
            terminal = self.controller.worker_terminal_for_request(
                request_id=request.request_id, plan_digest=request.plan_digest,
                lineage_id=request.lineage_id, stage_id=request.stage_id)
            if terminal is None:
                status = self.controller.worker_attempt_status(
                    request_id=request.request_id, plan_digest=request.plan_digest,
                    lineage_id=request.lineage_id, stage_id=request.stage_id)
                typed = (isinstance(lifecycle_error, worker_lifecycle.AdmissionDenied)
                         and status in {"not_acquired", "unknown"})
                if typed and lifecycle_error.denial_class == "terminal" \
                        and getattr(self.controller, "snapshot_version", None) == 3:
                    # AKU-06l: a typed terminal denial before any held interval is the
                    # explicit zero-cost no-receipt cancellation, settled exactly once.
                    denial_ref = unified_driver._digest({
                        "transition_id": transition_id, "request_id": request.request_id,
                        "stage_id": request.stage_id, "denial_class": "terminal",
                        "reason": str(lifecycle_error)})
                    try:
                        self.controller.unified_driver_settle_v2({
                            "schema": campaign_control.DRIVER_SETTLEMENT_SCHEMA_V2,
                            "catalog_id": catalog.catalog_id,
                            "transition_id": transition_id,
                            "selection": selection.to_dict(), "outcome": "failed",
                            "children": [], "denial_ref": denial_ref})
                    except BaseException as exc:
                        raise DriverExecutionUncertain(
                            "terminal-denial cancellation is uncertain; exact retry required"
                        ) from exc
                    raise lifecycle_error
                if (typed or (isinstance(lifecycle_error, (
                        worker_lifecycle.WaitingAuthority,
                        campaign_control.ControlRefused)) and status == "not_acquired")):
                    self._launched.discard(transition_id)
                    raise lifecycle_error
                raise DriverExecutionUncertain(
                    "lifecycle attempt started; exact terminal recovery is required") \
                    from lifecycle_error
            try:
                held = self.controller.worker_held_claim_receipt(terminal)
            except Exception as exc:
                raise DriverExecutionUncertain(
                    "failed lifecycle has no trusted held receipt; reconciliation required") \
                    from exc
            reference = None
            if terminal.accepted:
                try:
                    reference = invocation.result_reference()
                except BaseException as exc:
                    raise DriverExecutionUncertain(
                        "accepted terminal lacks its retained diagnostic result") from exc
            failed = _FailedAttempt(catalog.catalog_id, transition_id, selection,
                                    prepared, terminal, held, reference)
            self._finished[transition_id] = failed
            return self._finish(failed)
        assert terminal is not None
        if not terminal.accepted:
            try:
                held = self.controller.worker_held_claim_receipt(terminal)
            except Exception as exc:
                raise DriverExecutionUncertain(
                    "failed lifecycle has no trusted held receipt; reconciliation required") \
                    from exc
            failed = _FailedAttempt(catalog.catalog_id, transition_id, selection,
                                    prepared, terminal, held)
            self._finished[transition_id] = failed
            return self._finish(failed)
        reference = invocation.result_reference()
        start = invocation.start
        if start is None:
            raise DriverExecutionUncertain("accepted planned worker lacks its exact start")
        fence = self.controller.worker_result_fence(terminal)
        held = self.controller.worker_held_claim_receipt(terminal)
        attempt = _FinishedAttempt(catalog.catalog_id, transition_id, selection, prepared,
                                   start, terminal, reference, fence, held)
        self._finished[transition_id] = attempt
        return self._finish(attempt)

    def _finish(self, attempt: _FinishedAttempt | _FailedAttempt) \
            -> DriverExecutionReceipt:
        if isinstance(attempt, _FailedAttempt):
            terminal_ref = f"lifecycle:{unified_driver._digest(_terminal_body(attempt.terminal))}"
            request = {
                "schema": campaign_control.DRIVER_SETTLEMENT_SCHEMA,
                "catalog_id": attempt.catalog_id, "transition_id": attempt.transition_id,
                "selection": attempt.selection.to_dict(), "receipt": attempt.held.to_dict(),
                "outcome": "failed",
                "terminal_refs": [terminal_ref, *citations.refs(self.controller, attempt)],
            }
            with self._lock:
                self._trusted_settlements[attempt.transition_id] = copy.deepcopy(request)
            try:
                settlement = self._settle(request, attempt)
            except BaseException as exc:
                raise DriverExecutionUncertain(
                    "failed-attempt settlement is uncertain; exact retry required") from exc
            receipt = DriverExecutionReceipt(
                attempt.catalog_id, attempt.transition_id, attempt.prepared.prepared_digest,
                attempt.terminal.request_id, attempt.terminal.lineage_id,
                attempt.terminal.stage_id, _terminal_body(attempt.terminal),
                (None if attempt.reference is None else attempt.reference.to_dict()), (),
                request, settlement, schema=(EXECUTION_RECEIPT_SCHEMA_V2
                    if attempt.prepared.schema == unified_worker.PREPARED_SCHEMA_V3
                    else EXECUTION_RECEIPT_SCHEMA))
            self._receipts[attempt.transition_id] = receipt
            self._parent_evidence_registries.pop(attempt.transition_id, None)
            if attempt.prepared.schema == unified_worker.PREPARED_SCHEMA_V3:
                self.driver.preparation_owner.refresh_settled()
            return receipt
        self._install_capture_validator(attempt.prepared)
        self._active_fence = attempt.fence
        try:
            registry = self._parent_evidence_registries.get(attempt.transition_id)
            scope = (self._parent_receipt_replayer.using(registry)
                     if self._parent_receipt_replayer is not None else nullcontext())
            with scope:
                with self.controller.native_capture_callback() as capture:
                    entries = unified_worker.ingest_deferred_result(
                        attempt.reference, prepared=attempt.prepared, start=attempt.start,
                        terminal=attempt.terminal, fence=attempt.fence,
                        capture_transaction=capture)
        finally:
            self._active_fence = None
        measurement_ids = tuple(entry.record_id for entry in entries)
        if any(not isinstance(item, str) or not item for item in measurement_ids):
            raise DriverExecutionUncertain("native journal returned malformed record identities")
        terminal_refs = [f"native:{item}" for item in measurement_ids]
        calibration = attempt.prepared.schema == unified_worker.PREPARED_SCHEMA_V3
        outcome = "invalid"
        if calibration:
            from . import serving_preparation as preparation
            owner = self.driver.preparation_owner
            if owner is None:
                raise DriverExecutionUncertain("installed raw preparation owner is unavailable")
            reference = owner.accept(prepared=attempt.prepared, start=attempt.start,
                terminal=attempt.terminal, reference=attempt.reference, fence=attempt.fence)
            original_request = preparation.CalibrationPreparationDispatch.from_dict(
                attempt.prepared.dispatch).request
            outcome = owner.collection_outcome(reference, request=original_request)
            terminal_refs.append("calibration-collected:" + unified_driver._canonical(
                reference.to_dict()).decode())
        terminal_refs.append(
            f"result:{unified_worker._digest(attempt.reference.to_dict())}")
        request = {
            "schema": campaign_control.DRIVER_SETTLEMENT_SCHEMA,
            "catalog_id": attempt.catalog_id, "transition_id": attempt.transition_id,
            "selection": attempt.selection.to_dict(), "receipt": attempt.held.to_dict(),
            "outcome": outcome, "terminal_refs": terminal_refs,
        }
        with self._lock:
            self._trusted_settlements[attempt.transition_id] = copy.deepcopy(request)
        try:
            settlement = self._settle(request, attempt)
        except BaseException as exc:
            raise DriverExecutionUncertain(
                "settlement append/reply is uncertain; retry exact finished attempt") from exc
        receipt = DriverExecutionReceipt(
            attempt.catalog_id, attempt.transition_id, attempt.prepared.prepared_digest,
            attempt.start.request_id, attempt.start.lineage_id, attempt.start.stage_id,
            _terminal_body(attempt.terminal), attempt.reference.to_dict(), measurement_ids,
            request, settlement, schema=(EXECUTION_RECEIPT_SCHEMA_V2 if calibration
                                         else EXECUTION_RECEIPT_SCHEMA))
        self._receipts[attempt.transition_id] = receipt
        self._parent_evidence_registries.pop(attempt.transition_id, None)
        if calibration:
            self.driver.preparation_owner.refresh_settled()
        return receipt

    def _settle(self, request: Mapping[str, Any], attempt) -> Mapping[str, Any]:
        if not self._operation_settlement:
            return self.controller.unified_driver_settle(request)
        calibration = attempt.prepared.schema == unified_worker.PREPARED_SCHEMA_V3
        return settle_single_stage(
            self.controller, request, attempt.terminal, attempt.held,
            "calibration_preparation" if calibration else "runtime_comparison")

    @staticmethod
    def retry_durable(controller: Any, receipt: DriverExecutionReceipt) -> Mapping[str, Any]:
        """Recover a lost reply only when the controller already has exact durable bytes."""
        if not isinstance(receipt, DriverExecutionReceipt):
            raise DriverExecutionRefused("durable retry requires a typed execution receipt")
        result = controller.unified_driver_settle(single_stage_request(
            controller, unified_driver._thaw(receipt.settlement_request)))
        if result.get("status") != "duplicate":
            raise DriverExecutionRefused(
                "retry did not match an already durable settlement; live authority is required")
        return result

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._admission_closed = True
            errors = [self._stop_producer(producer)
                      for producer in tuple(self._unresolved_producers)]
            if self._unresolved_producers:
                cause = next((item for item in errors if item is not None), None)
                raise DriverExecutionUncertain(
                    "parent evidence producer teardown remains unresolved") from cause
            if self._capture_store is not None:
                self._capture_store.close()
                self._capture_store = None
            if self.driver.preparation_owner is not None:
                self.driver.preparation_owner.close()
            self._closed = True


def single_stage_request(controller: Any, request: Mapping[str, Any]) -> dict[str, Any]:
    """The exact SETTLED.v2 request of a v1-shaped single-stage attempt once its one
    child is durable (AKU-06l); otherwise the v1 request itself, unchanged."""
    children = controller.unified_driver_children(request.get("transition_id"))
    if not oa.single_stage(children) or request.get("schema") \
            != campaign_control.DRIVER_SETTLEMENT_SCHEMA:
        return dict(request)
    return {"schema": campaign_control.DRIVER_SETTLEMENT_SCHEMA_V2,
            **{name: request[name] for name in (
                "catalog_id", "transition_id", "selection", "outcome")},
            "children": [child["digest"] for child in children], "denial_ref": None,
            "evidence_refs": list(request["terminal_refs"])}


def settle_single_stage(controller: Any, request: Mapping[str, Any], terminal, held,
                        operation: str) -> Mapping[str, Any]:
    """Account the one original child (exact retry = duplicate), then settle it as v2."""
    selection = scheduling.Selection.from_dict(request["selection"])
    if held.to_dict() != request["receipt"]:
        raise DriverExecutionRefused("single-stage held receipt differs from its attempt")
    controller.unified_driver_account_child(
        binding=worker_lifecycle.AccountingBinding.from_selection(
            catalog_id=request["catalog_id"], transition_id=request["transition_id"],
            selection=selection),
        terminal=terminal, receipt=held, role=oa.SINGLE_STAGE_ROLE, operation=operation)
    return controller.unified_driver_settle_v2(single_stage_request(controller, request))


__all__ = ["DriverExecutionReceipt", "DriverExecutionRefused", "RetainedResultFenced",
           "DriverExecutionUncertain", "EXECUTION_RECEIPT_SCHEMA", "EXECUTION_RECEIPT_SCHEMA_V2",
           "UnifiedDriverExecution", "UnknownParentEvidenceProducer"]
