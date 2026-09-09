#!/usr/bin/env python3
"""Opt-in standalone consumer for unified planning and durable dispatch intent.

The driver owns no grant, journal, or worker authority.  It asks the existing
campaign controller for a closed readiness/transaction interface and refuses to
issue a scheduler selection until both are present.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import stat
import time
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from . import campaign, campaign_control, campaign_service, experiment_plan, planned_serving
from . import scheduling, scoped_evidence, unified_planner

CONFIG_SCHEMA = "epyc.autokernel.unified_driver_config.v1"
PROFILE_REQUEST_SCHEMA = "epyc.autokernel.profile_preparation_request.v1"
PROFILE_CONTRACT_SCHEMA = "epyc.autokernel.profile_preparation_contract.v1"
READINESS_SCHEMA = "epyc.autokernel.unified_driver_readiness.v1"
CATALOG_SCHEMA = "epyc.autokernel.unified_driver_planning_catalog.v1"
TRANSACTION_RECEIPT_SCHEMA = "epyc.autokernel.unified_driver_transaction_receipt.v1"
OUTCOME_SCHEMA = "epyc.autokernel.unified_driver_outcome.v1"
EXECUTION_INPUT_SCHEMA = "epyc.autokernel.unified_execution_input.v1"
MATERIALIZATION_BINDING_SCHEMA = "epyc.autokernel.unified_materialization_binding.v1"
SELECTED_PROFILE_WORK_SCHEMA = "epyc.autokernel.selected_profile_work.v1"


class DriverRefused(RuntimeError):
    pass


class DriverTransactionUncertain(DriverRefused):
    pass


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise DriverRefused(f"value is not finite canonical JSON: {exc}") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise DriverRefused(f"{label} must be an object with string keys")
    _canonical(value)
    return value


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


def _sha(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise DriverRefused(f"{label} must be lowercase SHA-256")
    return value


def _finite_positive(value: Any, label: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise DriverRefused(f"{label} must be finite and positive")
    return float(value)


@dataclass(frozen=True)
class ExecutionInput:
    """Startup-resolved non-authoritative input for one enrolled runtime target."""

    target_revision_digest: str
    prompt_manifest: planned_serving.FrozenPromptManifest
    max_stage_seconds: float
    teardown_seconds: float
    instrument_id: str
    schema: str = EXECUTION_INPUT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != EXECUTION_INPUT_SCHEMA:
            raise DriverRefused("execution input schema is unsupported")
        object.__setattr__(self, "target_revision_digest", _sha(
            self.target_revision_digest, "execution target revision"))
        try:
            prompts = planned_serving.FrozenPromptManifest.from_dict(
                self.prompt_manifest.to_dict())
        except Exception as exc:
            raise DriverRefused(f"execution prompt manifest is invalid: {exc}") from exc
        object.__setattr__(self, "prompt_manifest", prompts)
        object.__setattr__(self, "max_stage_seconds", _finite_positive(
            self.max_stage_seconds, "max_stage_seconds"))
        object.__setattr__(self, "teardown_seconds", _finite_positive(
            self.teardown_seconds, "teardown_seconds"))
        if not isinstance(self.instrument_id, str) or not self.instrument_id.strip():
            raise DriverRefused("execution instrument_id must be nonempty")

    @classmethod
    def from_dict(cls, value: Any) -> "ExecutionInput":
        row = dict(_mapping(value, "execution input"))
        if set(row) != {"schema", "target_revision_digest", "prompt_manifest",
                        "max_stage_seconds", "teardown_seconds", "instrument_id"} \
                or row.pop("schema") != EXECUTION_INPUT_SCHEMA:
            raise DriverRefused("execution input fields/schema differ")
        try:
            prompts = planned_serving.FrozenPromptManifest.from_dict(row["prompt_manifest"])
        except Exception as exc:
            raise DriverRefused(f"execution prompt manifest is invalid: {exc}") from exc
        return cls(row["target_revision_digest"], prompts, row["max_stage_seconds"],
                   row["teardown_seconds"], row["instrument_id"])

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema,
                "target_revision_digest": self.target_revision_digest,
                "prompt_manifest": self.prompt_manifest.to_dict(),
                "max_stage_seconds": self.max_stage_seconds,
                "teardown_seconds": self.teardown_seconds,
                "instrument_id": self.instrument_id}


@dataclass(frozen=True)
class ProfilePreparationRequest:
    target_revision_digest: str
    stage_proposal: scheduling.StageProposal
    profile_contract: Mapping[str, Any]
    schema: str = PROFILE_REQUEST_SCHEMA

    @classmethod
    def from_dict(cls, value: Any) -> "ProfilePreparationRequest":
        row = _mapping(value, "profile preparation request")
        if set(row) != {"schema", "target_revision_digest", "stage_proposal",
                        "profile_contract"} or row["schema"] != PROFILE_REQUEST_SCHEMA:
            raise DriverRefused("profile preparation request fields/schema differ")
        target = _sha(row["target_revision_digest"], "profile target revision")
        try:
            stage = scheduling.StageProposal.from_dict(row["stage_proposal"])
        except Exception as exc:
            raise DriverRefused(f"profile stage proposal is invalid: {exc}") from exc
        contract_row = _mapping(row["profile_contract"], "profile contract")
        if set(contract_row) != {"schema", "adapter_id", "adapter_digest"} \
                or contract_row["schema"] != PROFILE_CONTRACT_SCHEMA \
                or not isinstance(contract_row["adapter_id"], str) \
                or not contract_row["adapter_id"].strip():
            raise DriverRefused("profile preparation contract fields/schema differ")
        _sha(contract_row["adapter_digest"], "profile adapter_digest")
        contract = _freeze(contract_row)
        if (stage.target_revision != target or stage.stage_class != "prerequisite"
                or stage.eligible is not True):
            raise DriverRefused("profile request differs from its scheduler proposal")
        return cls(target, stage, contract)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema,
                "target_revision_digest": self.target_revision_digest,
                "stage_proposal": self.stage_proposal.to_dict(),
                "profile_contract": _thaw(self.profile_contract)}


@dataclass(frozen=True)
class SelectedProfileWork:
    """Exact current selected profile advice; never grant or execution authority."""

    catalog_id: str
    transition_id: str
    selection: scheduling.Selection
    profile_request: ProfilePreparationRequest
    stage_plan_digest: str
    controller_binding: Mapping[str, Any]
    execution_authorized: bool = False
    schema: str = SELECTED_PROFILE_WORK_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SELECTED_PROFILE_WORK_SCHEMA or self.execution_authorized is not False:
            raise DriverRefused("selected profile work schema/authority is invalid")
        object.__setattr__(self, "catalog_id", _sha(self.catalog_id, "profile catalog_id"))
        object.__setattr__(self, "transition_id", _sha(
            self.transition_id, "profile transition_id"))
        selection = (scheduling.Selection.from_dict(self.selection.to_dict())
                     if isinstance(self.selection, scheduling.Selection)
                     else scheduling.Selection.from_dict(self.selection))
        request = (ProfilePreparationRequest.from_dict(self.profile_request.to_dict())
                   if isinstance(self.profile_request, ProfilePreparationRequest)
                   else ProfilePreparationRequest.from_dict(self.profile_request))
        plan_digest = _sha(self.stage_plan_digest, "profile stage_plan_digest")
        binding = dict(_mapping(_thaw(self.controller_binding), "profile controller binding"))
        expected_binding = {"schema", "campaign_id", "config_digest", "config_generation",
                            "supervisor_id", "supervisor_incarnation", "artifact_root"}
        if (set(binding) != expected_binding
                or binding["schema"] != MATERIALIZATION_BINDING_SCHEMA
                or not isinstance(binding["campaign_id"], str)
                or not binding["campaign_id"].strip()
                or not isinstance(binding["supervisor_id"], str)
                or not binding["supervisor_id"].strip()
                or not isinstance(binding["artifact_root"], str)
                or not Path(binding["artifact_root"]).is_absolute()):
            raise DriverRefused("profile controller binding is malformed")
        _sha(binding["config_digest"], "profile controller config_digest")
        for name in ("config_generation", "supervisor_incarnation"):
            if (not isinstance(binding[name], int) or isinstance(binding[name], bool)
                    or binding[name] < 1):
                raise DriverRefused(f"profile controller {name} must be positive")
        if (selection.status != "selected" or selection.proposal is None
                or selection.proposal != request.stage_proposal
                or request.target_revision_digest != selection.proposal.target_revision
                or plan_digest != _digest(request.to_dict())
                or self.transition_id != _digest({
                    "catalog_id": self.catalog_id, "selection": selection.to_dict()})):
            raise DriverRefused("selected profile work bindings differ")
        object.__setattr__(self, "selection", selection)
        object.__setattr__(self, "profile_request", request)
        object.__setattr__(self, "stage_plan_digest", plan_digest)
        object.__setattr__(self, "controller_binding", _freeze(binding))

    @classmethod
    def from_dict(cls, value: Any) -> "SelectedProfileWork":
        row = dict(_mapping(value, "selected profile work"))
        expected = {"schema", "catalog_id", "transition_id", "selection",
                    "profile_request", "stage_plan_digest", "controller_binding",
                    "execution_authorized"}
        if set(row) != expected:
            raise DriverRefused("selected profile work fields differ")
        return cls(**row)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "catalog_id": self.catalog_id,
                "transition_id": self.transition_id,
                "selection": self.selection.to_dict(),
                "profile_request": self.profile_request.to_dict(),
                "stage_plan_digest": self.stage_plan_digest,
                "controller_binding": _thaw(self.controller_binding),
                "execution_authorized": False}


@dataclass(frozen=True)
class PlanningCatalog:
    campaign_digest: str
    controller_binding: Mapping[str, Any]
    scheduler_projection_digest: str
    observed_at: float
    stage_proposals: tuple[Mapping[str, Any], ...]
    work_by_stage_digest: Mapping[str, Mapping[str, Any]]
    schema: str = CATALOG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CATALOG_SCHEMA:
            raise DriverRefused("planning catalog schema is unsupported")
        _sha(self.campaign_digest, "campaign_digest")
        binding = _freeze(_mapping(self.controller_binding, "controller binding"))
        _sha(self.scheduler_projection_digest, "scheduler_projection_digest")
        if (isinstance(self.observed_at, bool)
                or not isinstance(self.observed_at, (int, float))
                or not math.isfinite(self.observed_at)):
            raise DriverRefused("catalog observed_at must be finite")
        stages = tuple(scheduling.StageProposal.from_dict(_thaw(item))
                       for item in self.stage_proposals)
        if not stages or len({item.digest for item in stages}) != len(stages):
            raise DriverRefused("catalog stage proposals must be nonempty and unique")
        work = _mapping(self.work_by_stage_digest, "catalog work map")
        if set(work) != {item.digest for item in stages}:
            raise DriverRefused("catalog work must exactly cover stage proposal digests")
        normalized = {}
        for stage in stages:
            row = dict(_mapping(work[stage.digest], "catalog work"))
            if set(row) != {"kind", "stage_plan_binding", "stage_plan_digest", "payload"}:
                raise DriverRefused("catalog work fields differ")
            if row["kind"] not in {"runtime_comparison", "actor_preparation",
                                    "profile_preparation"}:
                raise DriverRefused("catalog work kind is unsupported")
            if row["stage_plan_binding"] not in {"experiment_plan",
                                                  "preparation_contract"}:
                raise DriverRefused("catalog plan binding is unsupported")
            if ((row["kind"] == "runtime_comparison")
                    != (row["stage_plan_binding"] == "experiment_plan")):
                raise DriverRefused("only runtime work may bind an ExperimentPlan")
            _sha(row["stage_plan_digest"], "stage_plan_digest")
            payload = _mapping(row["payload"], "catalog work payload")
            if row["kind"] == "runtime_comparison":
                if set(payload) != {"proposal", "experiment_plan"}:
                    raise DriverRefused("runtime catalog payload fields differ")
                proposal = unified_planner.UnifiedProposal.from_dict(payload["proposal"])
                plan = experiment_plan.ExperimentPlan.from_dict(payload["experiment_plan"])
                if (proposal.proposal_id != stage.proposal_id
                        or proposal.target_revision_digest != stage.target_revision
                        or proposal.backend != stage.backend
                        or proposal.stage_class != stage.stage_class
                        or proposal.experiment_plan_digest != plan.digest
                        or plan.digest != row["stage_plan_digest"]):
                    raise DriverRefused("runtime catalog Plan/proposal/stage binding differs")
            elif row["kind"] == "actor_preparation":
                required = {"schema", "actor_kind", "actor_identity", "prompt",
                            "mandatory_conflicts", "proposal", "cache_key"}
                if set(payload) != required:
                    raise DriverRefused("actor preparation payload fields differ")
                proposal = unified_planner.UnifiedProposal.from_dict(payload["proposal"])
                preparation = unified_planner.ActorPreparationRequest(
                    payload["actor_kind"], payload["actor_identity"], payload["prompt"],
                    tuple(payload["mandatory_conflicts"]), proposal, payload["cache_key"],
                    payload["schema"])
                if (proposal.proposal_id != stage.proposal_id
                        or proposal.target_revision_digest != stage.target_revision
                        or proposal.backend != stage.backend
                        or stage.stage_class != "prerequisite"
                        or preparation.cache_key != row["stage_plan_digest"]):
                    raise DriverRefused("actor preparation proposal/stage binding differs")
            else:
                profile = ProfilePreparationRequest.from_dict(payload)
                if (profile.stage_proposal.digest != stage.digest
                        or _digest(profile.to_dict()) != row["stage_plan_digest"]):
                    raise DriverRefused("profile preparation/stage binding differs")
            normalized[stage.digest] = _freeze(row)
        object.__setattr__(self, "controller_binding", binding)
        object.__setattr__(self, "stage_proposals",
                           tuple(_freeze(item.to_dict()) for item in stages))
        object.__setattr__(self, "work_by_stage_digest", MappingProxyType(normalized))

    @property
    def catalog_id(self) -> str:
        return _digest(self.body())

    def body(self) -> dict[str, Any]:
        return {"schema": self.schema, "campaign_digest": self.campaign_digest,
                "controller_binding": _thaw(self.controller_binding),
                "scheduler_projection_digest": self.scheduler_projection_digest,
                "observed_at": self.observed_at,
                "stage_proposals": [_thaw(item) for item in self.stage_proposals],
                "work_by_stage_digest": _thaw(self.work_by_stage_digest)}

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "catalog_id": self.catalog_id}


@dataclass(frozen=True)
class DriverOutcome:
    status: str
    reasons: tuple[str, ...]
    transition_id: str | None
    selection: Mapping[str, Any] | None
    execution_authorized: bool = False
    schema: str = OUTCOME_SCHEMA

    def __post_init__(self) -> None:
        if (self.schema != OUTCOME_SCHEMA or self.status not in {
                "waiting", "intent_recorded", "stopped"}
                or self.execution_authorized is not False):
            raise DriverRefused("driver outcome is invalid")
        if not self.reasons or any(not isinstance(item, str) or not item for item in self.reasons):
            raise DriverRefused("driver outcome requires reasons")
        if self.transition_id is not None:
            _sha(self.transition_id, "transition_id")
        object.__setattr__(self, "reasons", tuple(self.reasons))
        if self.selection is not None:
            object.__setattr__(self, "selection", _freeze(_mapping(
                self.selection, "outcome selection")))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "status": self.status,
                "reasons": list(self.reasons), "transition_id": self.transition_id,
                "selection": None if self.selection is None else _thaw(self.selection),
                "execution_authorized": False}


def _pinned_source_revision(resolved: campaign.ResolvedCampaign) -> str:
    revisions = set()
    for _name, artifact in resolved.source_snapshot:
        if artifact is None:
            raise DriverRefused("runtime materialization has an unresolved source snapshot")
        candidate = artifact.ref.rsplit(":", 1)[-1]
        if (len(candidate) not in {40, 64}
                or any(char not in "0123456789abcdef" for char in candidate)):
            raise DriverRefused(
                "runtime materialization lacks an exact pinned Git source revision")
        revisions.add(candidate)
    if len(revisions) != 1:
        raise DriverRefused(
            "runtime materialization cannot bind multiple source revisions to one arm")
    return next(iter(revisions))


class UnifiedCampaignDriver:
    """One-tick consumer; actual execution remains with the controller-owned bridge."""

    def __init__(self, *, resolved_campaign: campaign.ResolvedCampaign,
                 controller: Any, scheduler_engine: scheduling.SchedulerEngine,
                 profiles: Mapping[str, Mapping[str, Any] | unified_planner.TargetProfile],
                 evidence_index: scoped_evidence.EvidenceIndex,
                 runtime_anchors: unified_planner.PreparedRuntimeAnchors,
                 runtime_dimensions: Mapping[str, Sequence[Mapping[str, Any]]],
                 experiment_plans: Mapping[str, experiment_plan.ExperimentPlan | Mapping[str, Any]],
                 profile_requests: Mapping[str, Mapping[str, Any] | ProfilePreparationRequest],
                 actor_identities: Mapping[str, Mapping[str, Any]],
                 native_artifact_sink_ref: str,
                 execution_inputs: Mapping[str, Mapping[str, Any] | ExecutionInput] | None = None,
                 monotonic_clock=time.monotonic) -> None:
        self.resolved = campaign.ResolvedCampaign.from_dict(resolved_campaign.to_dict())
        self.controller = controller
        if not isinstance(scheduler_engine, scheduling.SchedulerEngine):
            raise DriverRefused("driver requires a persistent SchedulerEngine")
        self.scheduler = scheduler_engine
        self.profiles = MappingProxyType({
            key: unified_planner.TargetProfile.from_dict(
                value.to_dict() if isinstance(value, unified_planner.TargetProfile) else value)
            for key, value in profiles.items()})
        self.evidence = evidence_index
        self.runtime_anchors = runtime_anchors
        self.runtime_dimensions = MappingProxyType({
            key: tuple(item if isinstance(item, unified_planner.RuntimeDimension)
                       else unified_planner.RuntimeDimension.from_dict(item)
                       for item in values)
            for key, values in runtime_dimensions.items()})
        self.experiment_plans = MappingProxyType({
            key: (experiment_plan.ExperimentPlan.from_dict(value.to_dict())
                  if isinstance(value, experiment_plan.ExperimentPlan)
                  else experiment_plan.ExperimentPlan.from_dict(value))
            for key, value in experiment_plans.items()})
        self.profile_requests = MappingProxyType({
            key: (ProfilePreparationRequest.from_dict(item.to_dict())
                  if isinstance(item, ProfilePreparationRequest)
                  else ProfilePreparationRequest.from_dict(item))
            for key, item in profile_requests.items()})
        self.actor_identities = _freeze(_thaw(actor_identities))
        parsed_inputs = {}
        for key, item in (execution_inputs or {}).items():
            if not isinstance(key, str):
                raise DriverRefused("execution input keys must be target revision digests")
            parsed = (ExecutionInput.from_dict(item.to_dict())
                      if isinstance(item, ExecutionInput) else ExecutionInput.from_dict(item))
            if key != parsed.target_revision_digest:
                raise DriverRefused("execution input key differs from target revision")
            parsed_inputs[key] = parsed
        self.execution_inputs = MappingProxyType(parsed_inputs)
        self.sink_ref = native_artifact_sink_ref
        if not callable(monotonic_clock):
            raise DriverRefused("monotonic_clock must be callable")
        self._monotonic_clock = monotonic_clock
        self._pending: PlanningCatalog | None = None
        self._issued_catalog: PlanningCatalog | None = None
        self._issued_transition_id: str | None = None
        self._poisoned = False

    @property
    def campaign_digest(self) -> str:
        return _digest(self.resolved.to_dict())

    def _readiness(self) -> Mapping[str, Any] | None:
        refresh = getattr(self.controller, "refresh_unified_driver_readiness", None)
        if callable(refresh):
            refresh()
        probe = getattr(self.controller, "unified_driver_readiness", None)
        transaction = getattr(self.controller, "unified_driver_transaction", None)
        if not callable(probe) or not callable(transaction):
            return None
        row = _mapping(probe(), "controller driver readiness")
        required = {"schema", "campaign_id", "config_digest", "config_generation",
                    "supervisor_incarnation", "control_revision", "admission_open",
                    "provider_available", "reason", "scheduler_projection_digest"}
        if set(row) != required or row["schema"] != READINESS_SCHEMA:
            raise DriverRefused("controller driver readiness schema differs")
        if (row["campaign_id"] != self.resolved.campaign_id
                or row["config_digest"]
                   != campaign_control.resolved_config_digest(self.resolved)
                or type(row["admission_open"]) is not bool
                or type(row["provider_available"]) is not bool
                or not isinstance(row["reason"], str) or not row["reason"]):
            raise DriverRefused("controller driver readiness binding is invalid")
        for key in ("config_generation", "supervisor_incarnation"):
            if not isinstance(row[key], int) or isinstance(row[key], bool) or row[key] < 1:
                raise DriverRefused(f"controller readiness {key} is invalid")
        if (not isinstance(row["control_revision"], int)
                or isinstance(row["control_revision"], bool)
                or row["control_revision"] < 0):
            raise DriverRefused("controller readiness control_revision is invalid")
        _sha(row["config_digest"], "controller config_digest")
        _sha(row["scheduler_projection_digest"], "scheduler projection digest")
        return _freeze(row)

    def _record(self, catalog: PlanningCatalog) -> DriverOutcome:
        callback = getattr(self.controller, "unified_driver_transaction")
        try:
            receipt = _mapping(callback(catalog.to_dict()), "driver transaction receipt")
        except campaign_control.ControlRefused as exc:
            raise DriverRefused(f"controller refused driver catalog: {exc}") from exc
        except BaseException as exc:
            self._pending = catalog
            self._poisoned = True
            raise DriverTransactionUncertain(
                "driver transaction outcome is uncertain; exact retry required") from exc
        if (set(receipt) != {"schema", "catalog_id", "transition_id", "status", "selection"}
                or receipt["schema"] != TRANSACTION_RECEIPT_SCHEMA
                or receipt["catalog_id"] != catalog.catalog_id
                or receipt["status"] not in {"accepted", "duplicate", "not_selected"}):
            self._pending = catalog
            self._poisoned = True
            raise DriverTransactionUncertain("driver transaction returned an invalid receipt")
        try:
            selection = scheduling.Selection.from_dict(receipt["selection"])
        except Exception as exc:
            self._pending = catalog
            self._poisoned = True
            raise DriverTransactionUncertain("driver transaction selection is invalid") from exc
        stages = {scheduling.StageProposal.from_dict(_thaw(item)).digest
                  for item in catalog.stage_proposals}
        if receipt["transition_id"] != _digest({
                    "catalog_id": catalog.catalog_id, "selection": selection.to_dict()}):
            self._pending = catalog
            self._poisoned = True
            raise DriverTransactionUncertain("driver transaction identity differs")
        if receipt["status"] == "not_selected":
            if selection.status == "selected":
                raise DriverTransactionUncertain("not-selected receipt contains a selection")
            return DriverOutcome("waiting", tuple(selection.reasons) or (selection.status,),
                                 None, selection.to_dict())
        if (selection.status != "selected" or selection.proposal is None
                or selection.proposal.digest not in stages):
            self._pending = catalog
            self._poisoned = True
            raise DriverTransactionUncertain("driver transaction selected outside exact catalog")
        self._pending = None
        self._poisoned = False
        self._issued_catalog = catalog
        self._issued_transition_id = receipt["transition_id"]
        return DriverOutcome("intent_recorded", (receipt["status"],),
                             receipt["transition_id"], selection.to_dict())

    def retry_pending(self) -> DriverOutcome:
        if self._pending is None:
            raise DriverRefused("driver has no uncertain transition to retry")
        return self._record(self._pending)

    def materialize_runtime(self, outcome: DriverOutcome):
        """Build the one bridge-owned prepared record selected by the controller.

        This is still advice: the returned record has no grant, worker, or execution
        authority. The lifecycle bridge must acquire and fence those independently.
        """
        if (outcome.status != "intent_recorded" or outcome.transition_id is None
                or outcome.selection is None or self._issued_catalog is None
                or outcome.transition_id != self._issued_transition_id):
            raise DriverRefused("runtime materialization requires the current issued intent")
        selection = scheduling.Selection.from_dict(_thaw(outcome.selection))
        if selection.proposal is None:
            raise DriverRefused("issued intent has no selected proposal")
        work = self._issued_catalog.work_by_stage_digest.get(selection.proposal.digest)
        if work is None or work["kind"] != "runtime_comparison":
            raise DriverRefused("selected work is not a runtime comparison")
        payload = _mapping(_thaw(work["payload"]), "selected runtime work")
        proposal = unified_planner.UnifiedProposal.from_dict(payload["proposal"])
        plan = experiment_plan.ExperimentPlan.from_dict(payload["experiment_plan"])
        execution_input = self.execution_inputs.get(proposal.target_revision_digest)
        if execution_input is None:
            raise DriverRefused(
                "selected target is waiting for frozen prompts and stage budgets")
        planning = unified_planner.PlanningResult(
            (proposal,), (selection.proposal,), (), (), None, None, None)
        dispatch = unified_planner.materialize_selection(planning, selection)
        if dispatch is None:
            raise DriverRefused("selected runtime dispatch could not be materialized")
        binding_callback = getattr(
            self.controller, "unified_driver_materialization_binding", None)
        if not callable(binding_callback):
            raise DriverRefused("controller-owned artifact/capture binding is unavailable")
        try:
            binding_value = binding_callback(
                catalog_id=self._issued_catalog.catalog_id,
                transition_id=outcome.transition_id,
                selection=selection.to_dict())
        except campaign_control.ControlRefused as exc:
            raise DriverRefused(
                f"controller refused runtime materialization: {exc}") from exc
        binding = dict(_mapping(binding_value, "materialization binding"))
        expected = {"schema", "campaign_id", "config_digest", "config_generation",
                    "supervisor_id", "supervisor_incarnation", "artifact_root"}
        if (set(binding) != expected
                or binding.pop("schema") != MATERIALIZATION_BINDING_SCHEMA
                or binding["campaign_id"] != self.resolved.campaign_id
                or binding["config_digest"]
                   != campaign_control.resolved_config_digest(self.resolved)):
            raise DriverRefused("controller materialization binding is stale or foreign")
        revision = _pinned_source_revision(self.resolved)
        if proposal.runtime_pair is None:
            raise DriverRefused("runtime proposal lacks its frozen arm pair")
        pair = unified_planner.RuntimeArmPair.from_dict(_thaw(proposal.runtime_pair))
        sources = {}
        for name, recipe in (("anchor", pair.anchor), ("candidate", pair.candidate)):
            sources[name] = {
                "source_revision": revision,
                "model_sha256": recipe.model.sha256,
                "build_sha256": recipe.executable.sha256,
                "recipe_hash": recipe.template.recipe_hash,
            }
        capture_base = {
            "campaign_id": binding["campaign_id"],
            "config_digest": binding["config_digest"],
            "supervisor_id": binding["supervisor_id"],
            "supervisor_incarnation": binding["supervisor_incarnation"],
            "config_generation": binding["config_generation"],
            "instrument_id": execution_input.instrument_id,
            "protocol_id": plan.protocol_ref,
            "protocol_status": plan.protocol_status,
            "source_identities": sources,
        }
        from . import unified_worker
        body = {
            "schema": unified_worker.PREPARED_SCHEMA,
            "dispatch": dispatch.to_dict(), "plan": plan.to_dict(),
            "prompt_manifest": execution_input.prompt_manifest.to_dict(),
            "runtime_pair": pair.to_dict(), "capture_context_base": capture_base,
            "artifact_root": binding["artifact_root"], "previous": None,
            "max_stage_seconds": execution_input.max_stage_seconds,
            "teardown_seconds": execution_input.teardown_seconds,
        }
        return unified_worker.PreparedPlannedServingStage.from_dict(
            {**body, "prepared_digest": _digest(body)})

    def materialize_profile(self, outcome: DriverOutcome) -> SelectedProfileWork:
        """Resolve exact selected profile advice through the current controller owner."""
        if (not isinstance(outcome, DriverOutcome)
                or outcome.status != "intent_recorded"
                or outcome.transition_id is None or outcome.selection is None
                or self._issued_catalog is None
                or outcome.transition_id != self._issued_transition_id):
            raise DriverRefused("profile materialization requires the current issued intent")
        try:
            selection = scheduling.Selection.from_dict(_thaw(outcome.selection))
        except Exception as exc:
            raise DriverRefused("issued profile selection is invalid") from exc
        if selection.proposal is None:
            raise DriverRefused("issued profile intent has no selected proposal")
        work = self._issued_catalog.work_by_stage_digest.get(selection.proposal.digest)
        if work is None or work["kind"] != "profile_preparation":
            raise DriverRefused("selected work is not profile preparation")
        request = ProfilePreparationRequest.from_dict(_thaw(work["payload"]))
        if (request.stage_proposal != selection.proposal
                or work["stage_plan_binding"] != "preparation_contract"
                or work["stage_plan_digest"] != _digest(request.to_dict())):
            raise DriverRefused("selected profile catalog binding differs")
        callback = getattr(self.controller, "unified_driver_materialization_binding", None)
        if not callable(callback):
            raise DriverRefused("controller-owned profile binding is unavailable")
        try:
            binding = callback(
                catalog_id=self._issued_catalog.catalog_id,
                transition_id=outcome.transition_id,
                selection=selection.to_dict())
        except campaign_control.ControlRefused as exc:
            raise DriverRefused(f"controller refused profile materialization: {exc}") from exc
        result = SelectedProfileWork(
            self._issued_catalog.catalog_id, outcome.transition_id, selection, request,
            work["stage_plan_digest"], binding)
        if (result.controller_binding["campaign_id"] != self.resolved.campaign_id
                or result.controller_binding["config_digest"]
                   != campaign_control.resolved_config_digest(self.resolved)):
            raise DriverRefused("profile controller binding is stale or foreign")
        return SelectedProfileWork.from_dict(result.to_dict())

    def tick(self, *, now: float | None = None, stop_requested=lambda: False) -> DriverOutcome:
        if self._pending is not None or self._poisoned:
            raise DriverTransactionUncertain("exact pending transition retry is required")
        if now is None:
            now = self._monotonic_clock()
        if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now):
            raise DriverRefused("driver clock must be finite")
        if stop_requested():
            return DriverOutcome("stopped", ("stop requested before scheduling",), None, None)
        readiness = self._readiness()
        if readiness is None:
            return DriverOutcome("waiting", ("controller driver transaction unavailable",),
                                 None, None)
        if not readiness["admission_open"] or not readiness["provider_available"]:
            return DriverOutcome("waiting", (readiness["reason"],), None, None)
        planning = unified_planner.plan_iteration(
            resolved_campaign=self.resolved, profiles=self.profiles,
            evidence_index=self.evidence, runtime_anchors=self.runtime_anchors,
            runtime_dimensions=self.runtime_dimensions, source_actor=None, build_actor=None,
            scheduler_engine=self.scheduler, experiment_plans=self.experiment_plans,
            now=now, native_artifact_sink_ref=self.sink_ref,
            defer_actor_preparation=True, issue_selection=False,
            actor_identities=_thaw(self.actor_identities),
            stop_requested=stop_requested)
        proposals_by_id = {item.proposal_id: item for item in planning.proposals}
        actors_by_id = {item.proposal.proposal_id: item
                        for item in planning.actor_preparations}
        runtime_stages = []
        missing_inputs = []
        for stage in planning.stage_proposals:
            proposal = proposals_by_id.get(stage.proposal_id)
            if proposal is None:
                raise DriverRefused("scheduler stage lacks exact planner proposal")
            if (proposal.proposal_id not in actors_by_id
                    and proposal.target_revision_digest not in self.execution_inputs):
                missing_inputs.append(
                    f"target:{proposal.target_revision_digest}:execution_input_missing")
                continue
            runtime_stages.append(stage)
        profile_stages = tuple(request.stage_proposal
                               for key, request in self.profile_requests.items()
                               if key not in self.profiles)
        stages = (*runtime_stages, *profile_stages)
        if not stages:
            return DriverOutcome(
                "waiting", tuple(missing_inputs) or ("no scheduler-ready work",), None, None)
        if stop_requested():
            return DriverOutcome("stopped", ("stop requested before scheduling",), None, None)
        work_by_digest: dict[str, Mapping[str, Any]] = {}
        profiles_by_digest = {item.stage_proposal.digest: item
                              for item in self.profile_requests.values()}
        for stage in stages:
            profile = profiles_by_digest.get(stage.digest)
            if profile is not None:
                payload = profile.to_dict()
                work_by_digest[stage.digest] = {
                    "kind": "profile_preparation", "payload": payload,
                    "stage_plan_binding": "preparation_contract",
                    "stage_plan_digest": _digest(payload)}
                continue
            proposal = proposals_by_id.get(stage.proposal_id)
            if proposal is None:
                raise DriverRefused("scheduler stage lacks exact planner proposal")
            actor = actors_by_id.get(proposal.proposal_id)
            if actor is not None:
                work_by_digest[stage.digest] = {
                    "kind": "actor_preparation", "payload": actor.to_dict(),
                    "stage_plan_binding": "preparation_contract",
                    "stage_plan_digest": actor.cache_key}
            else:
                plan = self.experiment_plans.get(proposal.proposal_id)
                if plan is None:
                    raise DriverRefused("runtime proposal lacks exact ExperimentPlan")
                work_by_digest[stage.digest] = {
                    "kind": "runtime_comparison",
                    "payload": {"proposal": proposal.to_dict(),
                                "experiment_plan": plan.to_dict()},
                    "stage_plan_binding": "experiment_plan",
                    "stage_plan_digest": plan.digest}
        catalog = PlanningCatalog(
            self.campaign_digest, _thaw(readiness),
            readiness["scheduler_projection_digest"], float(now),
            tuple(stage.to_dict() for stage in stages), work_by_digest)
        return self._record(catalog)


@dataclass(frozen=True)
class DriverConfig:
    resolved_campaign_path: str
    store_path: str
    scheduler_config: Mapping[str, Any]
    scheduler_state: Mapping[str, Any]
    runtime_anchors: Mapping[str, Any]
    runtime_dimensions: Mapping[str, Any]
    profiles: Mapping[str, Any]
    experiment_plans: Mapping[str, Any]
    profile_requests: Mapping[str, Any]
    execution_inputs: Mapping[str, Any]
    native_artifact_sink_ref: str
    config_generation: int
    schema: str = CONFIG_SCHEMA

    @classmethod
    def from_dict(cls, value: Any) -> "DriverConfig":
        fields = {"schema", "resolved_campaign_path", "store_path", "scheduler_config",
                  "scheduler_state", "runtime_anchors", "runtime_dimensions", "profiles",
                  "experiment_plans", "profile_requests", "native_artifact_sink_ref",
                  "execution_inputs", "config_generation"}
        row = dict(_mapping(value, "driver config"))
        if set(row) != fields or row.pop("schema") != CONFIG_SCHEMA:
            raise DriverRefused("driver config fields/schema differ")
        for name in ("resolved_campaign_path", "store_path"):
            path = Path(row[name]) if isinstance(row[name], str) else Path()
            if not path.is_absolute():
                raise DriverRefused(f"{name} must be absolute")
            row[name] = str(path)
        if (not isinstance(row["config_generation"], int)
                or isinstance(row["config_generation"], bool)
                or row["config_generation"] < 1):
            raise DriverRefused("config_generation must be positive")
        if not isinstance(row["native_artifact_sink_ref"], str) \
                or not row["native_artifact_sink_ref"]:
            raise DriverRefused("native_artifact_sink_ref must be nonempty")
        for name in ("scheduler_config", "scheduler_state", "runtime_anchors",
                     "runtime_dimensions", "profiles", "experiment_plans",
                     "profile_requests", "execution_inputs"):
            row[name] = _freeze(_mapping(row[name], name))
        return cls(**row)


def load_config(path: Path) -> DriverConfig:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            before = os.fstat(fd)
            if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
                    or before.st_nlink != 1 or before.st_size > 4 * 1024 * 1024):
                raise DriverRefused("unified driver config file identity/size is unsafe")
            chunks = []
            remaining = 4 * 1024 * 1024 + 1
            while remaining:
                chunk = os.read(fd, min(remaining, 65536))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            after = os.fstat(fd)
            if (len(b"".join(chunks)) > 4 * 1024 * 1024
                    or (before.st_dev, before.st_ino, before.st_size)
                    != (after.st_dev, after.st_ino, after.st_size)):
                raise DriverRefused("unified driver config changed during bounded read")
        finally:
            os.close(fd)
        return DriverConfig.from_dict(json.loads(b"".join(chunks).decode("utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DriverRefused(f"cannot load unified driver config: {exc}") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--once", action="store_true")
    modes.add_argument("--listen", metavar="HOST:PORT")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    resolved = campaign_service.load_resolved(Path(config.resolved_campaign_path))
    scheduler_config = scheduling.SchedulerConfig.from_dict(_thaw(config.scheduler_config))
    scheduler_state = scheduling.SchedulerState.from_dict(_thaw(config.scheduler_state))
    scheduler_engine = scheduling.SchedulerEngine(scheduler_config, scheduler_state)
    anchors = unified_planner.prepare_runtime_anchors(
        resolved, _thaw(config.runtime_anchors))
    controller = campaign_control.CampaignController(
        resolved, Path(config.store_path), config_generation=config.config_generation,
        snapshot_version=3, scheduler_engine=scheduler_engine)
    with controller:
        driver = UnifiedCampaignDriver(
            resolved_campaign=resolved, controller=controller,
            scheduler_engine=scheduler_engine, profiles=_thaw(config.profiles),
            evidence_index=scoped_evidence.EvidenceIndex((), current_epoch="driver-start"),
            runtime_anchors=anchors,
            runtime_dimensions=_thaw(config.runtime_dimensions),
            experiment_plans=_thaw(config.experiment_plans),
            profile_requests=_thaw(config.profile_requests),
            actor_identities={},
            native_artifact_sink_ref=config.native_artifact_sink_ref,
            execution_inputs=_thaw(config.execution_inputs))
        if args.once:
            print(json.dumps(driver.tick().to_dict(), sort_keys=True))
            return 0
        host, port = campaign_service.parse_listen(args.listen)
        token = os.environ.get("AUTOKERNEL_CONTROL_TOKEN", "")
        if not token:
            raise DriverRefused("AUTOKERNEL_CONTROL_TOKEN is required")
        service = campaign_service.CampaignHTTPService(controller, host, port, token)
        try:
            service.start()
            while service.thread is not None and service.thread.is_alive():
                driver.tick()
                service.thread.join(timeout=0.25)
        finally:
            service.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CATALOG_SCHEMA", "CONFIG_SCHEMA", "DriverConfig", "DriverOutcome", "DriverRefused",
           "DriverTransactionUncertain", "PlanningCatalog", "OUTCOME_SCHEMA",
           "EXECUTION_INPUT_SCHEMA", "ExecutionInput",
           "PROFILE_CONTRACT_SCHEMA", "PROFILE_REQUEST_SCHEMA", "ProfilePreparationRequest",
           "SELECTED_PROFILE_WORK_SCHEMA", "SelectedProfileWork",
           "READINESS_SCHEMA",
           "TRANSACTION_RECEIPT_SCHEMA", "UnifiedCampaignDriver",
           "load_config", "main"]
