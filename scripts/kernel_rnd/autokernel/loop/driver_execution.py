"""Connect one durably selected runtime comparison to the owned worker bridge.

This module supplies no grant or scientific authority.  The lifecycle owns process
execution and provider receipts; the controller owns both native writes and scheduler
settlement.  In the absence of a registered scientific evaluator, completed native
observations are charged as ``invalid`` rather than promoted to comparisons.
"""
from __future__ import annotations

import copy
import queue
import threading
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from . import campaign_control
from . import experiment_plan as ep
from . import measurement_capture as mc
from . import native_capture_control as nc
from . import planned_serving as ps
from . import scheduling
from . import unified_driver
from . import unified_worker
from . import worker_lifecycle

EXECUTION_RECEIPT_SCHEMA = "epyc.autokernel.unified_driver_execution_receipt.v1"


class DriverExecutionRefused(RuntimeError):
    """The selected work cannot safely cross the execution boundary."""


class DriverExecutionUncertain(RuntimeError):
    """An owned attempt started and must be reconciled, never launched again."""


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DriverExecutionRefused(f"{label} must be nonempty text")
    return value


def _sha(value: Any, label: str) -> str:
    value = _text(value, label)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise DriverExecutionRefused(f"{label} must be lowercase SHA-256")
    return value


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
                 plan: ep.ExperimentPlan) -> None:
        if not isinstance(authority, unified_worker.ParentUnitEvidenceAuthority):
            raise DriverExecutionRefused("parent evidence cache is not the fixed bounded type")
        self.authority = authority
        self.plan = ep.ExperimentPlan.from_dict(plan.to_dict())
        self._stop = threading.Event()
        self._errors: list[BaseException] = []
        self._thread = threading.Thread(target=self._run, name="autokernel-parent-evidence",
                                        daemon=False)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                notice = self.authority.next_notice(timeout=0.05)
            except queue.Empty:
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
                    witnesses = MappingProxyType({
                        name: ep.Witness("unknown", None)
                        for name in self.plan.required_witnesses})
                    self.authority.publish_completion(
                        key, ps.StageCompletion(
                            fence.fence_id, True, witnesses, "flagged_but_retained",
                            "registered parent observation evaluator is unavailable"))
                elif kind == "continuation":
                    if not isinstance(key, str):
                        raise DriverExecutionRefused("continuation notice is malformed")
                    self.authority.publish_continuation(key, False)
                else:
                    raise DriverExecutionRefused("parent evidence notice kind is unsupported")
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


@dataclass(frozen=True)
class DriverExecutionReceipt:
    catalog_id: str
    transition_id: str
    prepared_digest: str
    request_id: str
    lineage_id: str
    stage_id: str
    terminal: Mapping[str, Any]
    result_reference: Mapping[str, Any] | None
    native_measurement_ids: tuple[str, ...]
    settlement_request: Mapping[str, Any]
    settlement_receipt: Mapping[str, Any]
    schema: str = EXECUTION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != EXECUTION_RECEIPT_SCHEMA:
            raise DriverExecutionRefused("execution receipt schema is unsupported")
        for name in ("catalog_id", "transition_id", "prepared_digest"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        for name in ("request_id", "lineage_id", "stage_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if len(set(self.native_measurement_ids)) != len(self.native_measurement_ids):
            raise DriverExecutionRefused("execution receipt native identities are invalid")
        for value in self.native_measurement_ids:
            _text(value, "native measurement id")
        for name in ("terminal", "settlement_request", "settlement_receipt"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise DriverExecutionRefused(f"{name} must be an object")
            object.__setattr__(self, name, unified_driver._freeze(copy.deepcopy(dict(value))))
        terminal_fields = {"worker_id", "worker_generation", "request_id", "plan_digest",
            "lineage_id", "stage_id", "grant_id", "grant_generation", "container_id",
            "return_code", "result_digest", "accepted", "reason"}
        if set(self.terminal) != terminal_fields \
                or type(self.terminal["accepted"]) is not bool \
                or self.terminal["request_id"] != self.request_id \
                or self.terminal["lineage_id"] != self.lineage_id \
                or self.terminal["stage_id"] != self.stage_id:
            raise DriverExecutionRefused("execution receipt terminal binding differs")
        reference = None
        if self.result_reference is not None:
            if not isinstance(self.result_reference, Mapping):
                raise DriverExecutionRefused("result_reference must be an object or null")
            reference = unified_worker.PlannedWorkerResultReference.from_dict(
                unified_driver._thaw(self.result_reference))
            object.__setattr__(self, "result_reference", unified_driver._freeze(
                reference.to_dict()))
        for name in ("worker_generation", "grant_generation"):
            if (not isinstance(self.terminal[name], int)
                    or isinstance(self.terminal[name], bool) or self.terminal[name] < 1):
                raise DriverExecutionRefused(f"terminal {name} is invalid")
        _sha(self.terminal["plan_digest"], "terminal plan digest")
        selection_request = unified_driver._thaw(self.settlement_request)
        expected_request = {"schema", "catalog_id", "transition_id", "selection",
                            "receipt", "outcome", "terminal_refs"}
        if (set(selection_request) != expected_request
                or selection_request["schema"] != campaign_control.DRIVER_SETTLEMENT_SCHEMA
                or selection_request["catalog_id"] != self.catalog_id
                or selection_request["transition_id"] != self.transition_id
                or selection_request["outcome"] not in {"invalid", "failed"}):
            raise DriverExecutionRefused("execution receipt settlement request differs")
        selection = scheduling.Selection.from_dict(selection_request["selection"])
        held = scheduling.HeldClaimReceipt.from_dict(selection_request["receipt"])
        if (selection.proposal is None or held.proposal_id != selection.proposal.proposal_id
                or held.proposal_id != self.request_id
                or held.ownership_generation != self.terminal["worker_generation"]
                or held.allocation_generation != self.terminal["grant_generation"]):
            raise DriverExecutionRefused("execution receipt held/selection binding differs")
        if self.terminal["accepted"]:
            if (reference is None
                    or reference.worker_id != self.terminal["worker_id"]
                    or reference.worker_generation != self.terminal["worker_generation"]
                    or reference.prepared_digest != self.prepared_digest
                    or self.terminal["return_code"] != 0
                    or self.terminal["reason"] is not None):
                raise DriverExecutionRefused("execution result reference/terminal differs")
            result_envelope_digest = unified_worker._digest(reference.to_dict())
            if self.terminal["result_digest"] != result_envelope_digest:
                raise DriverExecutionRefused("execution receipt terminal references differ")
            if selection_request["outcome"] == "invalid":
                expected_refs = [f"native:{item}" for item in self.native_measurement_ids] + [
                    f"result:{result_envelope_digest}"]
                if (not self.native_measurement_ids
                        or selection_request["terminal_refs"] != expected_refs):
                    raise DriverExecutionRefused(
                        "execution receipt terminal references differ")
            else:
                terminal_digest = unified_driver._digest(
                    unified_driver._thaw(self.terminal))
                if (self.native_measurement_ids
                        or selection_request["terminal_refs"]
                           != [f"lifecycle:{terminal_digest}"]):
                    raise DriverExecutionRefused(
                        "operational failure receipt binding differs")
        else:
            terminal_digest = unified_driver._digest(unified_driver._thaw(self.terminal))
            if (reference is not None or self.native_measurement_ids
                    or selection_request["outcome"] != "failed"
                    or selection_request["terminal_refs"]
                       != [f"lifecycle:{terminal_digest}"]):
                raise DriverExecutionRefused("failed execution receipt binding differs")
        settlement = self.settlement_receipt
        if (set(settlement) != {"schema", "transition_id", "status",
                               "accounting_projection_digest"}
                or settlement["schema"] != campaign_control.DRIVER_SETTLEMENT_RECEIPT_SCHEMA
                or settlement["transition_id"] != self.transition_id
                or settlement["status"] not in {"accepted", "duplicate"}):
            raise DriverExecutionRefused("execution settlement receipt differs")
        _sha(settlement["accounting_projection_digest"], "accounting projection digest")

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema": self.schema, "catalog_id": self.catalog_id,
            "transition_id": self.transition_id, "prepared_digest": self.prepared_digest,
            "request_id": self.request_id, "lineage_id": self.lineage_id,
            "stage_id": self.stage_id, "terminal": unified_driver._thaw(self.terminal),
            "result_reference": (None if self.result_reference is None
                                 else unified_driver._thaw(self.result_reference)),
            "native_measurement_ids": list(self.native_measurement_ids),
            "settlement_request": unified_driver._thaw(self.settlement_request),
            "settlement_receipt": unified_driver._thaw(self.settlement_receipt),
        }
        return {**body, "receipt_digest": unified_driver._digest(body)}

    @classmethod
    def from_dict(cls, value: Any) -> "DriverExecutionReceipt":
        fields = {"schema", "catalog_id", "transition_id", "prepared_digest",
                  "request_id", "lineage_id", "stage_id", "terminal",
                  "result_reference", "native_measurement_ids", "settlement_request",
                  "settlement_receipt", "receipt_digest"}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise DriverExecutionRefused("execution receipt fields differ")
        body = {key: copy.deepcopy(value[key]) for key in fields - {"receipt_digest"}}
        supplied = _sha(value["receipt_digest"], "execution receipt digest")
        if supplied != unified_driver._digest(body):
            raise DriverExecutionRefused("execution receipt digest differs")
        ids = body.pop("native_measurement_ids")
        if not isinstance(ids, (list, tuple)):
            raise DriverExecutionRefused("native measurement ids must be an array")
        return cls(native_measurement_ids=tuple(ids), **body)


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


class UnifiedDriverExecution:
    """Single-owner selected-runtime executor with exact retry boundaries."""

    def __init__(self, *, driver: unified_driver.UnifiedCampaignDriver,
                 controller: Any) -> None:
        if (not isinstance(driver, unified_driver.UnifiedCampaignDriver)
                or not isinstance(controller, campaign_control.CampaignController)):
            raise DriverExecutionRefused(
                "driver/controller must be the current typed owners")
        if driver.controller is not controller:
            raise DriverExecutionRefused("driver/controller ownership differs")
        self.driver = driver
        self.controller = controller
        self._launched: set[str] = set()
        self._finished: dict[str, _FinishedAttempt | _FailedAttempt] = {}
        self._trusted_settlements: dict[str, dict[str, Any]] = {}
        self._receipts: dict[str, DriverExecutionReceipt] = {}
        self._capture_store: mc.ArtifactStore | None = None
        self._active_fence: nc.TrustedWorkerResultFence | None = None
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
        transition_id = supplied.get("transition_id")
        with self._lock:
            if self._closed:
                raise DriverExecutionRefused("driver execution connector is closed")
            trusted = copy.deepcopy(self._trusted_settlements.get(transition_id))
            attempt = self._finished.get(transition_id)
        if trusted is None or attempt is None or dict(supplied) != trusted:
            raise DriverExecutionRefused("settlement differs from the owned finished attempt")
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
            recovered = self.controller.worker_terminal_for_request(
                request_id=attempt.terminal.request_id,
                plan_digest=attempt.terminal.plan_digest,
                lineage_id=attempt.terminal.lineage_id,
                stage_id=attempt.terminal.stage_id)
            terminal_ref = f"lifecycle:{unified_driver._digest(_terminal_body(attempt.terminal))}"
            if recovered != attempt.terminal or trusted["terminal_refs"] != [terminal_ref]:
                raise DriverExecutionRefused("failed lifecycle terminal differs")
            if attempt.reference is not None:
                fence = self.controller.worker_result_fence(attempt.terminal)
                if (not attempt.terminal.accepted or not fence.current
                        or not fence.result_accepted
                        or unified_worker._digest(attempt.reference.to_dict())
                           != attempt.terminal.result_digest):
                    raise DriverExecutionRefused(
                        "operational failure terminal/result differs")
        return copy.deepcopy(trusted)

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
            fence_provider=lambda _measurement_id, _context: self._active_fence)
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

    def _execute_locked(self, outcome: unified_driver.DriverOutcome) \
            -> DriverExecutionReceipt:
        transition_id = outcome.transition_id
        if transition_id is None:
            raise DriverExecutionRefused("execution requires a durably issued transition")
        if transition_id in self._receipts:
            stored = self._receipts[transition_id]
            duplicate = self.controller.unified_driver_settle(
                unified_driver._thaw(stored.settlement_request))
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
        prepared = self.driver.materialize_runtime(outcome)
        selection = scheduling.Selection.from_dict(outcome.selection)
        if selection.proposal is None:
            raise DriverExecutionRefused("issued runtime selection lacks a proposal")
        catalog = self.driver._issued_catalog
        if catalog is None:
            raise DriverExecutionRefused("driver no longer owns the issued catalog")
        readiness = self.controller.unified_driver_readiness()
        authority = unified_worker.ParentUnitEvidenceAuthority(
            max_records=max(8, len(prepared.plan.expected_units) * 2))
        producer = UnknownParentEvidenceProducer(authority, prepared.plan)
        invocation = unified_worker.PlannedWorkerInvocation.open(prepared, authority)
        request_id = selection.proposal.proposal_id
        try:
            request = invocation.stage_request(
                request_id=request_id,
                lineage_id=f"driver:{transition_id}", stage_id=f"runtime:{transition_id}",
                control_revision=readiness["control_revision"])
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
                if (isinstance(lifecycle_error, (
                        worker_lifecycle.WaitingAuthority,
                        campaign_control.ControlRefused))
                        and self.controller.worker_attempt_status(
                            request_id=request.request_id,
                            plan_digest=request.plan_digest,
                            lineage_id=request.lineage_id,
                            stage_id=request.stage_id) == "not_acquired"):
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
                "outcome": "failed", "terminal_refs": [terminal_ref],
            }
            with self._lock:
                self._trusted_settlements[attempt.transition_id] = copy.deepcopy(request)
            try:
                settlement = self.controller.unified_driver_settle(request)
            except BaseException as exc:
                raise DriverExecutionUncertain(
                    "failed-attempt settlement is uncertain; exact retry required") from exc
            receipt = DriverExecutionReceipt(
                attempt.catalog_id, attempt.transition_id, attempt.prepared.prepared_digest,
                attempt.terminal.request_id, attempt.terminal.lineage_id,
                attempt.terminal.stage_id, _terminal_body(attempt.terminal),
                (None if attempt.reference is None else attempt.reference.to_dict()), (),
                request, settlement)
            self._receipts[attempt.transition_id] = receipt
            return receipt
        self._install_capture_validator(attempt.prepared)
        self._active_fence = attempt.fence
        try:
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
        terminal_refs.append(
            f"result:{unified_worker._digest(attempt.reference.to_dict())}")
        request = {
            "schema": campaign_control.DRIVER_SETTLEMENT_SCHEMA,
            "catalog_id": attempt.catalog_id, "transition_id": attempt.transition_id,
            "selection": attempt.selection.to_dict(), "receipt": attempt.held.to_dict(),
            "outcome": "invalid", "terminal_refs": terminal_refs,
        }
        with self._lock:
            self._trusted_settlements[attempt.transition_id] = copy.deepcopy(request)
        try:
            settlement = self.controller.unified_driver_settle(request)
        except BaseException as exc:
            raise DriverExecutionUncertain(
                "settlement append/reply is uncertain; retry exact finished attempt") from exc
        receipt = DriverExecutionReceipt(
            attempt.catalog_id, attempt.transition_id, attempt.prepared.prepared_digest,
            attempt.start.request_id, attempt.start.lineage_id, attempt.start.stage_id,
            _terminal_body(attempt.terminal), attempt.reference.to_dict(), measurement_ids,
            request, settlement)
        self._receipts[attempt.transition_id] = receipt
        return receipt

    @staticmethod
    def retry_durable(controller: Any, receipt: DriverExecutionReceipt) -> Mapping[str, Any]:
        """Recover a lost reply only when the controller already has exact durable bytes."""
        if not isinstance(receipt, DriverExecutionReceipt):
            raise DriverExecutionRefused("durable retry requires a typed execution receipt")
        result = controller.unified_driver_settle(
            unified_driver._thaw(receipt.settlement_request))
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
            self._closed = True


__all__ = ["DriverExecutionReceipt", "DriverExecutionRefused",
           "DriverExecutionUncertain", "EXECUTION_RECEIPT_SCHEMA",
           "UnifiedDriverExecution", "UnknownParentEvidenceProducer"]
