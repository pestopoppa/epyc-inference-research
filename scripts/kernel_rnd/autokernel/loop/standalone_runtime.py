"""Single-owner composition of the unified planner and controller execution path.

This module grants no provider, scientific, validation, or promotion authority.  It
only drives the already typed driver/executor pair one bounded selection at a time.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from pathlib import Path
import threading
import time
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from . import campaign, campaign_control, driver_execution, experiment_plan, feed_runtime
from . import scheduling, scoped_evidence, unified_driver, unified_planner, worker_lifecycle

CONFIG_SCHEMA = "epyc.autokernel.standalone_runtime_config.v1"
RESULT_SCHEMA = "epyc.autokernel.standalone_runtime_result.v1"
SHUTDOWN_SCHEMA = "epyc.autokernel.standalone_runtime_shutdown.v1"
MAX_EXACT_RETRIES = 32
_RETRYABLE_STATES = frozenset(
    {"driver_transaction_retry_required", "execution_exact_retry_required"}
)


class StandaloneRuntimeRefused(RuntimeError):
    """The requested operation is malformed or conflicts with owned state."""


class StandaloneRuntimeUncertain(RuntimeError):
    """Durable state may have advanced and only an exact retry is safe."""


def _seconds(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StandaloneRuntimeRefused(f"{label} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise StandaloneRuntimeRefused(f"{label} must be a positive finite number")
    return result


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({copy.deepcopy(key): _freeze(item) for key, item in value.items()})
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
class StandaloneRuntimeConfig:
    idle_interval_seconds: float = 0.05
    unavailable_backoff_seconds: float = 0.25
    max_backoff_seconds: float = 5.0
    shutdown_timeout_seconds: float = 10.0
    max_exact_retries: int = 3
    schema: str = CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CONFIG_SCHEMA:
            raise StandaloneRuntimeRefused("runtime config schema is unsupported")
        for name in (
            "idle_interval_seconds",
            "unavailable_backoff_seconds",
            "max_backoff_seconds",
            "shutdown_timeout_seconds",
        ):
            object.__setattr__(self, name, _seconds(getattr(self, name), name))
        if self.unavailable_backoff_seconds > self.max_backoff_seconds:
            raise StandaloneRuntimeRefused("unavailable backoff cannot exceed maximum backoff")
        if (
            not isinstance(self.max_exact_retries, int)
            or isinstance(self.max_exact_retries, bool)
            or self.max_exact_retries < 1
            or self.max_exact_retries > MAX_EXACT_RETRIES
        ):
            raise StandaloneRuntimeRefused(
                f"max_exact_retries must be from 1 through {MAX_EXACT_RETRIES}"
            )


@dataclass(frozen=True)
class StandaloneRuntimeInputs:
    resolved_campaign: campaign.ResolvedCampaign
    scheduler_engine: scheduling.SchedulerEngine
    profiles: Mapping[str, unified_planner.TargetProfile | Mapping[str, Any]]
    evidence_index: scoped_evidence.EvidenceIndex
    runtime_anchors: unified_planner.PreparedRuntimeAnchors
    runtime_dimensions: Mapping[str, Sequence[unified_planner.RuntimeDimension | Mapping[str, Any]]]
    experiment_plans: Mapping[str, experiment_plan.ExperimentPlan | Mapping[str, Any]]
    profile_requests: Mapping[str, unified_driver.ProfilePreparationRequest | Mapping[str, Any]]
    actor_identities: Mapping[str, Mapping[str, Any]]
    execution_inputs: Mapping[str, unified_driver.ExecutionInput | Mapping[str, Any]]
    native_artifact_sink_ref: str
    feed_owner: Any = None
    native_evidence_configuration: Any = None
    loaded_instrument_identity: Mapping[str, Any] | None = None
    loaded_instrument_reference: Mapping[str, Any] | None = None
    native_artifact_root: Path | None = None
    observation_configuration: Any = None


@dataclass(frozen=True)
class RuntimeTickResult:
    status: str
    reason: str
    retry_after_seconds: float
    controller_snapshot: Mapping[str, Any]
    driver_outcome: Mapping[str, Any] | None = None
    execution_receipt: Mapping[str, Any] | None = None
    schema: str = RESULT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RESULT_SCHEMA or self.status not in {
            "recovered",
            "waiting",
            "settled",
            "stopped",
            "recovery_required",
        }:
            raise StandaloneRuntimeRefused("runtime result schema/status is invalid")
        if not isinstance(self.reason, str) or not self.reason:
            raise StandaloneRuntimeRefused("runtime result requires a reason")
        delay = float(self.retry_after_seconds)
        if not math.isfinite(delay) or delay < 0:
            raise StandaloneRuntimeRefused("runtime retry delay is invalid")
        snapshot = campaign_control.validate_snapshot_v3(self.controller_snapshot)
        object.__setattr__(self, "retry_after_seconds", delay)
        object.__setattr__(self, "controller_snapshot", _freeze(snapshot))
        if self.driver_outcome is not None:
            object.__setattr__(self, "driver_outcome", _freeze(self.driver_outcome))
        if self.execution_receipt is not None:
            object.__setattr__(self, "execution_receipt", _freeze(self.execution_receipt))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "status": self.status,
            "reason": self.reason,
            "retry_after_seconds": self.retry_after_seconds,
            "controller_snapshot": _thaw(self.controller_snapshot),
            "driver_outcome": _thaw(self.driver_outcome),
            "execution_receipt": _thaw(self.execution_receipt),
        }


@dataclass(frozen=True)
class RuntimeShutdownResult:
    status: str
    reason: str
    controller_snapshot: Mapping[str, Any]
    schema: str = SHUTDOWN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SHUTDOWN_SCHEMA or self.status not in {"closed", "shutdown_incomplete"}:
            raise StandaloneRuntimeRefused("runtime shutdown schema/status is invalid")
        if not isinstance(self.reason, str) or not self.reason:
            raise StandaloneRuntimeRefused("runtime shutdown requires a reason")
        object.__setattr__(
            self,
            "controller_snapshot",
            _freeze(campaign_control.validate_snapshot_v3(self.controller_snapshot)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "status": self.status,
            "reason": self.reason,
            "controller_snapshot": _thaw(self.controller_snapshot),
        }


class StandaloneRuntime:
    """One bounded, recover-first runtime owner independent of HTTP handling."""

    def __init__(
        self,
        *,
        controller: campaign_control.CampaignController,
        driver: unified_driver.UnifiedCampaignDriver,
        executor: driver_execution.UnifiedDriverExecution,
        config: StandaloneRuntimeConfig,
        monotonic_clock,
    ) -> None:
        self.controller = controller
        self.driver = driver
        self.executor = executor
        self.config = config
        self._clock = monotonic_clock
        self._condition = threading.Condition(threading.RLock())
        self._active = False
        self._recovered = False
        self._stop_requested = False
        self._closed = False
        self._close_result: RuntimeShutdownResult | None = None
        self._wait_count = 0
        self._pending_outcome: unified_driver.DriverOutcome | None = None
        self._uncertain: str | None = None
        self._retry_failures = 0
        self._uncertainty_reason: str | None = None

    @classmethod
    def compose(
        cls,
        *,
        controller: campaign_control.CampaignController,
        inputs: StandaloneRuntimeInputs,
        config: StandaloneRuntimeConfig | None = None,
        monotonic_clock=time.monotonic,
    ) -> "StandaloneRuntime":
        if not isinstance(controller, campaign_control.CampaignController):
            raise StandaloneRuntimeRefused("runtime requires the current CampaignController")
        if not isinstance(inputs, StandaloneRuntimeInputs):
            raise StandaloneRuntimeRefused("runtime requires typed composition inputs")
        if not isinstance(inputs.resolved_campaign, campaign.ResolvedCampaign):
            raise StandaloneRuntimeRefused("runtime requires a resolved campaign")
        if controller.snapshot_version != 3:
            raise StandaloneRuntimeRefused("runtime requires controller snapshot v3")
        if controller.resolved.to_dict() != inputs.resolved_campaign.to_dict():
            raise StandaloneRuntimeRefused("controller and runtime campaign differ")
        if not isinstance(inputs.scheduler_engine, scheduling.SchedulerEngine):
            raise StandaloneRuntimeRefused("runtime requires the indexed SchedulerEngine")
        if not callable(monotonic_clock):
            raise StandaloneRuntimeRefused("runtime clock must be callable")
        if inputs.feed_owner is not None:
            if not isinstance(inputs.feed_owner, feed_runtime.FeedRuntimeOwner):
                raise StandaloneRuntimeRefused("runtime requires a concrete feed owner")
            feed_runtime.validate_paths(inputs.feed_owner.config, controller.store)
        if inputs.native_evidence_configuration is not None:
            from . import measurement_capture as mc, observation_binding as ob
            if (inputs.native_artifact_root is None
                    or inputs.loaded_instrument_identity is None
                    or inputs.loaded_instrument_reference is None):
                raise StandaloneRuntimeRefused(
                    "native runtime requires its exact published instrument")
            store = mc.ArtifactStore(inputs.native_artifact_root)
            try:
                expected = ob.LoadedInstrumentReference.from_dict(
                    unified_driver._thaw(inputs.loaded_instrument_reference))
                actual = store.verify(
                    f"loaded-instrument:{inputs.loaded_instrument_identity['sha256']}",
                    unified_driver._thaw(inputs.loaded_instrument_identity))
            finally:
                store.close()
            if actual.to_dict() != expected.artifact.to_dict():
                raise StandaloneRuntimeRefused(
                    "published native instrument differs from startup")
        readiness = controller.unified_driver_readiness()
        if (
            readiness["scheduler_projection_digest"]
            != inputs.scheduler_engine.operational_projection().projection_digest
        ):
            raise StandaloneRuntimeRefused("controller and runtime scheduler differ")
        driver = unified_driver.UnifiedCampaignDriver(
            resolved_campaign=inputs.resolved_campaign,
            controller=controller,
            scheduler_engine=inputs.scheduler_engine,
            profiles=inputs.profiles,
            evidence_index=inputs.evidence_index,
            runtime_anchors=inputs.runtime_anchors,
            runtime_dimensions=inputs.runtime_dimensions,
            experiment_plans=inputs.experiment_plans,
            profile_requests=inputs.profile_requests,
            actor_identities=inputs.actor_identities,
            native_artifact_sink_ref=inputs.native_artifact_sink_ref,
            execution_inputs=inputs.execution_inputs,
            monotonic_clock=monotonic_clock,
            executable_work_kinds={"runtime_comparison"},
            feed_owner=inputs.feed_owner,
        )
        executor = driver_execution.UnifiedDriverExecution(
            driver=driver, controller=controller,
            observation_configuration=inputs.observation_configuration,
            native_evidence_configuration=inputs.native_evidence_configuration)
        return cls(
            controller=controller,
            driver=driver,
            executor=executor,
            config=config or StandaloneRuntimeConfig(),
            monotonic_clock=monotonic_clock,
        )

    def _snapshot(self) -> Mapping[str, Any]:
        return campaign_control.validate_snapshot_v3(self.controller.snapshot())

    def _delay(self, unavailable: bool = False) -> float:
        if not unavailable:
            self._wait_count = 0
            return self.config.idle_interval_seconds
        self._wait_count += 1
        return min(
            self.config.max_backoff_seconds,
            self.config.unavailable_backoff_seconds * (2.0 ** min(self._wait_count - 1, 1023)),
        )

    def _enter(self) -> bool:
        with self._condition:
            if self._closed:
                raise StandaloneRuntimeRefused("runtime is closed")
            if self._active:
                raise StandaloneRuntimeRefused("a runtime operation is already active")
            if self._stop_requested:
                return False
            self._active = True
            return True

    def _leave(self) -> None:
        with self._condition:
            self._active = False
            self._condition.notify_all()

    def _mark_uncertain(self, state: str, reason: str, *, retry: bool) -> bool:
        with self._condition:
            if retry and self._uncertain == state:
                self._retry_failures += 1
            else:
                self._retry_failures = 0
            if retry and self._retry_failures >= self.config.max_exact_retries:
                self._uncertain = "execution_recovery_required"
                self._uncertainty_reason = (
                    f"exact retry budget exhausted after {self._retry_failures} failures: {reason}"
                )
                return False
            self._uncertain = state
            self._uncertainty_reason = reason
            return True

    def recover(self) -> RuntimeTickResult:
        if not self._enter():
            return RuntimeTickResult("stopped", "stop requested", 0, self._snapshot())
        try:
            pending_record = self.controller.unified_driver_pending_intent()
            with self._condition:
                pending = self._pending_outcome
            if pending_record is None:
                if pending is not None:
                    raise StandaloneRuntimeRefused(
                        "restored local intent is no longer pending in the controller")
            elif pending is None:
                pending = self.driver.restore_issued_intent(pending_record)
            elif pending.transition_id != pending_record["transition_id"]:
                raise StandaloneRuntimeRefused(
                    "restored local intent conflicts with the controller pending intent")
            receipt = None
            if pending is None:
                self.controller.reconcile_workers()
            else:
                receipt = self.executor.recover_issued(pending)
            with self._condition:
                self._recovered = True
                self._pending_outcome = None
                self._uncertain = None
                self._retry_failures = 0
                self._uncertainty_reason = None
            if receipt is not None:
                return RuntimeTickResult(
                    "settled",
                    "restored issued execution durably settled",
                    0,
                    self._snapshot(),
                    pending.to_dict(),
                    receipt.to_dict(),
                )
            return RuntimeTickResult(
                "recovered", "owned lifecycle recovery completed", 0, self._snapshot()
            )
        except Exception as exc:
            with self._condition:
                if 'pending' in locals() and pending is not None:
                    self._pending_outcome = pending
                self._uncertain = "execution_recovery_required"
                self._uncertainty_reason = str(exc)
            return RuntimeTickResult(
                "recovery_required", str(exc), self._delay(unavailable=True), self._snapshot()
            )
        finally:
            self._leave()

    def _drive(self, *, retry: bool, external_stop=lambda: False) -> RuntimeTickResult:
        if not self._enter():
            return RuntimeTickResult("stopped", "stop requested", 0, self._snapshot())
        outcome: unified_driver.DriverOutcome | None = None
        try:
            with self._condition:
                recovered = self._recovered
                uncertain = self._uncertain
                pending = self._pending_outcome
                uncertainty_reason = self._uncertainty_reason
            if not recovered:
                return RuntimeTickResult(
                    "recovery_required",
                    "worker recovery must precede admission",
                    self._delay(unavailable=True),
                    self._snapshot(),
                )
            if uncertain is not None and not retry:
                return RuntimeTickResult(
                    "recovery_required",
                    uncertainty_reason or uncertain,
                    self._delay(unavailable=True),
                    self._snapshot(),
                )
            if retry and uncertain == "driver_transaction_retry_required":
                outcome = self.driver.retry_pending()
            elif retry and pending is not None:
                outcome = pending
            elif retry:
                raise StandaloneRuntimeRefused("runtime has no exact pending operation")
            else:
                outcome = self.driver.tick(
                    stop_requested=lambda: self._stop_requested or external_stop()
                )
            if outcome.status == "waiting":
                return RuntimeTickResult(
                    "waiting",
                    "; ".join(outcome.reasons),
                    self._delay(unavailable=True),
                    self._snapshot(),
                    outcome.to_dict(),
                )
            if outcome.status == "stopped":
                return RuntimeTickResult(
                    "stopped", "; ".join(outcome.reasons), 0, self._snapshot(), outcome.to_dict()
                )
            if self.driver.issued_work_kind(outcome) != "runtime_comparison":
                raise StandaloneRuntimeRefused("standalone runtime received unavailable work")
            with self._condition:
                self._pending_outcome = outcome
            receipt = self.executor.execute(outcome)
            with self._condition:
                self._pending_outcome = None
                self._uncertain = None
                self._retry_failures = 0
                self._uncertainty_reason = None
            return RuntimeTickResult(
                "settled",
                "runtime attempt durably settled",
                self._delay(),
                self._snapshot(),
                outcome.to_dict(),
                receipt.to_dict(),
            )
        except feed_runtime.FeedRuntimeRefused as exc:
            return RuntimeTickResult(
                "recovery_required", str(exc), self._delay(unavailable=True), self._snapshot())
        except unified_driver.DriverTransactionUncertain as exc:
            self._mark_uncertain("driver_transaction_retry_required", str(exc), retry=retry)
            raise StandaloneRuntimeUncertain(str(exc)) from exc
        except campaign_control.DriverAdmissionClosed as exc:
            stopping = self._stop_requested or external_stop()
            return RuntimeTickResult(
                "stopped" if stopping else "waiting",
                str(exc),
                0 if stopping else self._delay(unavailable=True),
                self._snapshot(),
            )
        except (worker_lifecycle.WaitingAuthority, campaign_control.ControlRefused) as exc:
            if outcome is None:
                if retry:
                    retryable = self._mark_uncertain(
                        uncertain or "driver_transaction_retry_required",
                        str(exc),
                        retry=True,
                    )
                    if not retryable:
                        return RuntimeTickResult(
                            "recovery_required",
                            self._uncertainty_reason or str(exc),
                            self._delay(unavailable=True),
                            self._snapshot(),
                        )
                else:
                    with self._condition:
                        self._uncertain = None
                        self._retry_failures = 0
                        self._uncertainty_reason = None
                return RuntimeTickResult(
                    "waiting", str(exc), self._delay(unavailable=True), self._snapshot()
                )
            retryable = self._mark_uncertain(
                "execution_exact_retry_required", str(exc), retry=retry
            )
            return RuntimeTickResult(
                "waiting" if retryable else "recovery_required",
                str(exc) if retryable else self._uncertainty_reason or str(exc),
                self._delay(unavailable=True),
                self._snapshot(),
                outcome.to_dict(),
            )
        except driver_execution.DriverExecutionUncertain as exc:
            self._mark_uncertain("execution_exact_retry_required", str(exc), retry=retry)
            raise StandaloneRuntimeUncertain(str(exc)) from exc
        finally:
            self._leave()

    def tick(self) -> RuntimeTickResult:
        return self._drive(retry=False)

    def retry_pending(self) -> RuntimeTickResult:
        return self._drive(retry=True)

    def run(self, stop_event: threading.Event) -> RuntimeTickResult:
        try:
            return self._run_owned(stop_event)
        finally:
            owner = self.driver.feed_owner
            if owner is not None:
                owner.close()

    def _run_owned(self, stop_event: threading.Event) -> RuntimeTickResult:
        if not isinstance(stop_event, threading.Event):
            raise StandaloneRuntimeRefused("run requires a threading.Event")
        last = (
            self.recover()
            if not self._recovered
            else RuntimeTickResult(
                "waiting",
                "runtime already recovered",
                self.config.idle_interval_seconds,
                self._snapshot(),
            )
        )
        if last.status == "recovery_required" and not self._recovered:
            return last
        while not stop_event.is_set() and not self._stop_requested:
            with self._condition:
                retry = self._uncertain in _RETRYABLE_STATES
            try:
                last = self._drive(retry=retry, external_stop=stop_event.is_set)
            except StandaloneRuntimeUncertain as exc:
                with self._condition:
                    retryable = self._uncertain in _RETRYABLE_STATES
                    reason = self._uncertainty_reason or str(exc)
                if not retryable:
                    return RuntimeTickResult(
                        "recovery_required",
                        reason,
                        self._delay(unavailable=True),
                        self._snapshot(),
                    )
                last = RuntimeTickResult(
                    "waiting",
                    reason,
                    self._delay(unavailable=True),
                    self._snapshot(),
                )
            if last.status in {"stopped", "recovery_required"}:
                break
            stop_event.wait(last.retry_after_seconds)
        return last

    def request_stop(self) -> None:
        with self._condition:
            self._stop_requested = True
            self._condition.notify_all()

    def close(self, *, deadline: float | None = None) -> RuntimeShutdownResult:
        already_closed = False
        timed_out = False
        with self._condition:
            if self._closed:
                already_closed = True
            else:
                self._stop_requested = True
                if deadline is None:
                    deadline = self._clock() + self.config.shutdown_timeout_seconds
                if (
                    isinstance(deadline, bool)
                    or not isinstance(deadline, (int, float))
                    or not math.isfinite(deadline)
                ):
                    raise StandaloneRuntimeRefused("shutdown deadline must be finite")
                while self._active:
                    remaining = deadline - self._clock()
                    if remaining <= 0:
                        timed_out = True
                        break
                    self._condition.wait(remaining)
        if already_closed:
            assert self._close_result is not None
            return self._close_result
        if timed_out:
            return RuntimeShutdownResult(
                "shutdown_incomplete", "runtime operation remains active", self._snapshot()
            )
        owner = self.driver.feed_owner
        if owner is not None:
            try:
                feed_closed = owner.close_if_owner()
            except Exception as exc:
                return RuntimeShutdownResult(
                    "shutdown_incomplete", f"evidence cleanup failed: {exc}", self._snapshot())
            if not feed_closed:
                return RuntimeShutdownResult(
                    "shutdown_incomplete", "execution-thread evidence cleanup remains active",
                    self._snapshot())
        try:
            self.executor.close()
        except driver_execution.DriverExecutionUncertain as exc:
            return RuntimeShutdownResult("shutdown_incomplete", str(exc), self._snapshot())
        result = RuntimeShutdownResult(
            "closed", "executor closed; controller remains owned", self._snapshot()
        )
        with self._condition:
            self._closed = True
            self._close_result = result
        return result


__all__ = [
    "CONFIG_SCHEMA",
    "MAX_EXACT_RETRIES",
    "RESULT_SCHEMA",
    "SHUTDOWN_SCHEMA",
    "RuntimeShutdownResult",
    "RuntimeTickResult",
    "StandaloneRuntime",
    "StandaloneRuntimeConfig",
    "StandaloneRuntimeInputs",
    "StandaloneRuntimeRefused",
    "StandaloneRuntimeUncertain",
]
