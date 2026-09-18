"""Installed selected-profile execution and original settled-profile feedback.

This is an ownership connector, not a profiler, grant provider or scientific grader.
Serialized receipts and unpublished/unsettled profiles never grant planner authority.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import threading
import time
from types import FunctionType, MappingProxyType
from typing import Any, Mapping

from . import actor_preparation_state as state
from . import campaign, campaign_control as cc, lifecycle_observation as lo
from . import planned_serving, scheduling, target_profile_execution as tp
from . import unified_driver as ud, unified_planner as up, worker_lifecycle as wl

BINDING_SCHEMA = "epyc.autokernel.installed_profile_mechanism.v1"
SOURCE_SCHEMA = "epyc.autokernel.installed_profile_source.v1"
RECEIPT_SCHEMA = "epyc.autokernel.profile_preparation_execution_receipt.v1"
MAX_TARGETS = 4096


class ProfilePreparationRefused(ValueError):
    pass


def _plain(value):
    return ud._thaw(value)


def _driver_tick_source():
    value = ud.UnifiedCampaignDriver.tick
    if type(value) is not FunctionType:
        raise ProfilePreparationRefused("profile driver tick default shape differs")
    defaults = value.__kwdefaults__
    if (value.__defaults__ is not None
            or value.__closure__ is not None or type(defaults) is not dict
            or set(defaults) != {"now", "stop_requested"} or defaults["now"] is not None
            or type(defaults["stop_requested"]) is not FunctionType):
        raise ProfilePreparationRefused("profile driver tick default shape differs")
    identity = lo.callable_identity(value)
    stop = lo.callable_identity(defaults["stop_requested"])
    if (identity["implementation_status"] != "pinned"
            or stop["implementation_status"] != "pinned"
            or stop["configuration_status"] != "pinned"):
        raise ProfilePreparationRefused("profile driver tick default source is incomplete")
    # The generic identity remains unproven for configuration. This closed
    # projection accounts for the exact loaded callable default independently;
    # it is not a cloned/stripped function or a generic object serializer.
    return {"callable": identity, "kwdefaults": {"now": None, "stop_requested": stop}}


def _source_closure():
    from . import runtime_aggregates as aggregate
    roles = {
        "source_closure": _source_closure,
        "binding_body": InstalledProfileMechanismBinding.body,
        "binding_validate": InstalledProfileMechanismBinding.validate_request,
        "installation_validate": InstalledProfilePreparationBinding.validate,
        "original_producer": tp.TargetProfileExecution.prepare,
        "producer_construction": tp.TargetProfileExecution.__init__,
        "profile_request": tp._profile_request,
        "profile_event_validate": state.validate_event,
        "profile_digest": state.digest,
        "profile_json": state._canonical,
        "profile_reserve": cc.CampaignController.reserve_target_profile_execution,
        "profile_finish": cc.CampaignController.finish_target_profile_execution,
        "profile_stdout": cc.CampaignController.read_worker_stdout,
        "profile_held_receipt": cc.CampaignController.actor_held_claim_receipt,
        "profile_driver_settlement": cc.CampaignController.unified_driver_settle,
        "profile_publish": cc.CampaignController.record_verified_target_profile,
        "profile_read": cc.CampaignController.current_verified_profile_result,
        "profile_execute": InstalledProfilePreparationOwner.execute,
        "profile_settle": InstalledProfilePreparationOwner._settle,
        "profile_verify": InstalledProfilePreparationOwner.verify_settlement,
        "profile_replay": InstalledProfilePreparationOwner.planner_profiles,
        "profile_refresh_debt": InstalledProfilePreparationOwner.consumed_request_debt,
        "profile_observation_row": InstalledProfilePreparationOwner._profile_observation_row,
        "profile_observation_failed": InstalledProfilePreparationOwner._profile_observation_failed,
        "profile_debt_observation": InstalledProfilePreparationOwner._record_profile_debt,
        "profile_observation_snapshot": InstalledProfilePreparationOwner.observation_snapshot,
        "profile_driver_refresh": ud.UnifiedCampaignDriver.refresh_installed_profiles,
        "profile_driver_tick_source": _driver_tick_source,
        "profile_snapshot": InstalledProfilePreparationOwner._profile,
        "profile_current_publication": _current_publication,
        "profile_receipt": ProfilePreparationExecutionReceipt.__post_init__,
        "profile_receipt_parse": ProfilePreparationExecutionReceipt.from_dict.__func__,
        "selected_implementation": _selected_implementation,
        "profile_content": up.TargetProfile.from_dict.__func__,
    }
    identities = {name: lo.callable_identity(value) for name, value in roles.items()}
    incomplete = [name for name, row in identities.items()
                  if row["implementation_status"] != "pinned"
                  or row["configuration_status"] != "pinned"]
    if incomplete:
        raise ProfilePreparationRefused(
            "installed profile source closure is incomplete: " + ",".join(incomplete))
    return {"schema": SOURCE_SCHEMA, "callables": identities,
            "observation_codec": aggregate.source_identity(),
            "driver_tick": _driver_tick_source(),
            "profile_output_schema": tp.PROFILE_OUTPUT_SCHEMA,
            "profile_event_schema": state.PROFILE_SCHEMA,
            "profile_request_schema": ud.PROFILE_REQUEST_SCHEMA,
            "receipt_schema": RECEIPT_SCHEMA, "max_targets": MAX_TARGETS}


@dataclass(frozen=True)
class InstalledProfileMechanismBinding:
    mechanism: tp.ProfileMechanism
    valid_for_seconds: float
    source_closure: Mapping[str, Any] = field(init=False, repr=False)

    def __post_init__(self):
        if type(self.mechanism) is not tp.ProfileMechanism:
            raise ProfilePreparationRefused("profile mechanism is not the concrete installed type")
        state._finite(self.valid_for_seconds, "profile validity", minimum=1e-12)
        object.__setattr__(self, "source_closure", ud._freeze(_source_closure()))

    def body(self):
        mechanism = self.mechanism
        return {"schema": BINDING_SCHEMA, "mechanism_id": mechanism.mechanism_id,
                "binary": str(mechanism.binary), "binary_sha256": mechanism.binary_sha256,
                "cwd": str(mechanism.cwd), "env": dict(mechanism.env),
                "loaded_identity": dict(mechanism.loaded_identity),
                "max_stage_seconds": mechanism.max_stage_seconds,
                "teardown_seconds": mechanism.teardown_seconds,
                "max_output_bytes": mechanism.max_output_bytes,
                "valid_for_seconds": self.valid_for_seconds,
                "source_closure": _plain(self.source_closure)}

    @property
    def adapter_digest(self):
        return state.digest(self.body())

    def validate_request(self, request):
        if type(request) is not ud.ProfilePreparationRequest:
            raise ProfilePreparationRefused("profile request must be typed")
        request = ud.ProfilePreparationRequest.from_dict(request.to_dict())
        contract = request.profile_contract
        if (contract["adapter_id"] != self.mechanism.mechanism_id
                or contract["adapter_digest"] != self.adapter_digest
                or request.target_revision_digest
                   != self.mechanism.loaded_identity["target_revision_digest"]
                or _source_closure() != _plain(self.source_closure)):
            raise ProfilePreparationRefused("installed profile adapter/source/target differs")


@dataclass(frozen=True)
class InstalledProfilePreparationBinding:
    mechanisms: Mapping[str, InstalledProfileMechanismBinding]

    def __post_init__(self):
        if not isinstance(self.mechanisms, Mapping):
            raise ProfilePreparationRefused("profile mechanisms must be a mapping")
        rows = dict(self.mechanisms)
        if not rows or len(rows) > MAX_TARGETS:
            raise ProfilePreparationRefused("profile target count is outside its bound")
        for target, value in rows.items():
            if (type(value) is not InstalledProfileMechanismBinding
                    or target != value.mechanism.loaded_identity["target_revision_digest"]):
                raise ProfilePreparationRefused("installed profile target/type differs")
        object.__setattr__(self, "mechanisms", MappingProxyType(rows))

    def validate(self, resolved, runtime_anchors, requests):
        if (type(resolved) is not campaign.ResolvedCampaign
                or type(runtime_anchors) is not up.PreparedRuntimeAnchors
                or runtime_anchors.resolved_campaign_digest != state.digest(resolved.to_dict())):
            raise ProfilePreparationRefused("profile prepared campaign identity differs")
        targets = {state.digest(target.to_dict()): target for target in resolved.targets}
        for target, binding in self.mechanisms.items():
            request = requests.get(target)
            recipe = runtime_anchors.recipes.get(target)
            if target not in targets or recipe is None or request is None:
                raise ProfilePreparationRefused("profile target lacks original prepared recipe/request")
            binding.validate_request(request)
            loaded = binding.mechanism.loaded_identity
            arm = planned_serving.arm_identity(recipe.template, recipe)
            if (loaded["model_digest"] != arm["model_digest"]
                    or loaded["executable_digest"] != arm["executable_digest"]
                    or loaded["dso_digest"] != arm["dso_set_digest"]
                    or loaded["recipe_digest"] != recipe.snapshot_digest
                    or request.stage_proposal.backend != recipe.backend):
                raise ProfilePreparationRefused("profile model/build/DSO/recipe/backend differs")

    def install(self, *, controller, resolved, runtime_anchors, requests, configured_profiles):
        if type(controller) is not cc.CampaignController:
            raise ProfilePreparationRefused("profile controller must be the concrete owner")
        self.validate(resolved, runtime_anchors, requests)
        if controller.resolved.to_dict() != resolved.to_dict():
            raise ProfilePreparationRefused("profile controller campaign differs")
        producer = tp.TargetProfileExecution(controller=controller, mechanisms={
            target: binding.mechanism for target, binding in self.mechanisms.items()})
        controller.register_target_profile_producer(producer)
        return InstalledProfilePreparationOwner(controller, self, producer, configured_profiles)


@dataclass(frozen=True)
class ProfilePreparationExecutionReceipt:
    """Integrity-checked reporting value; parsing never restores original issuance."""
    body: Mapping[str, Any]

    def __post_init__(self):
        row = _plain(self.body)
        fields = {"schema", "catalog_id", "transition_id", "selected_work", "selected_work_digest",
                  "request_id", "stage_id", "terminal", "provider_held_receipt",
                  "profile_reference", "disposition", "settlement_request", "settlement_receipt"}
        if (not isinstance(row, Mapping) or set(row) != fields
                or row["schema"] != RECEIPT_SCHEMA
                or row["disposition"] not in {"profile_verified", "failed"}):
            raise ProfilePreparationRefused("profile execution receipt fields/schema differ")
        state._canonical(row)
        work = ud.SelectedProfileWork.from_dict(row["selected_work"])
        held = scheduling.HeldClaimReceipt.from_dict(row["provider_held_receipt"])
        terminal = row["terminal"]
        terminal_fields = {"worker_id", "worker_generation", "request_id", "plan_digest",
            "lineage_id", "stage_id", "grant_id", "grant_generation", "container_id",
            "return_code", "result_digest", "accepted", "reason"}
        if (not isinstance(terminal, Mapping) or set(terminal) != terminal_fields
                or type(terminal["accepted"]) is not bool):
            raise ProfilePreparationRefused("profile receipt terminal fields differ")
        request_digest = state.digest(work.profile_request.to_dict())
        if (row["catalog_id"] != work.catalog_id or row["transition_id"] != work.transition_id
                or row["selected_work_digest"] != state.digest(work.to_dict())
                or row["request_id"] != "profile-" + request_digest[:24]
                or row["stage_id"] != "target-profile-" + request_digest[:24]
                or terminal["request_id"] != row["request_id"]
                or terminal["stage_id"] != row["stage_id"]
                or terminal["plan_digest"] != work.stage_plan_digest
                or terminal["lineage_id"] != work.transition_id
                or held.ownership_generation != terminal["worker_generation"]
                or held.allocation_generation != terminal["grant_generation"]
                or held.proposal_id != work.selection.proposal.proposal_id
                or held.backend != work.selection.proposal.backend
                or held.stage_class != work.selection.proposal.stage_class):
            raise ProfilePreparationRefused("profile receipt selected/terminal/held identity differs")
        settlement = row["settlement_request"]
        settlement_fields = {"schema", "catalog_id", "transition_id", "selection", "receipt",
                             "outcome", "terminal_refs"}
        if (not isinstance(settlement, Mapping) or set(settlement) != settlement_fields
                or settlement["schema"] != cc.DRIVER_SETTLEMENT_SCHEMA
                or settlement["catalog_id"] != work.catalog_id
                or settlement["transition_id"] != work.transition_id
                or settlement["selection"] != work.selection.to_dict()
                or settlement["receipt"] != held.to_dict()):
            raise ProfilePreparationRefused("profile receipt settlement identity differs")
        reference = row["profile_reference"]
        if row["disposition"] == "profile_verified":
            ref_fields = {"profile_event_digest", "target_profile_digest", "profile_request_digest",
                          "catalog_id", "transition_id", "verifier_ref", "target_revision_digest"}
            if (not isinstance(reference, Mapping) or set(reference) != ref_fields
                    or terminal["accepted"] is not True or terminal["return_code"] != 0
                    or terminal["reason"] is not None
                    or reference["catalog_id"] != work.catalog_id
                    or reference["transition_id"] != work.transition_id
                    or reference["profile_request_digest"] != request_digest
                    or reference["target_revision_digest"] != work.profile_request.target_revision_digest
                    or reference["verifier_ref"] !=
                       f"controller-worker:{terminal['worker_id']}:{terminal['worker_generation']}"
                    or settlement["outcome"] != "prerequisite"
                    or settlement["terminal_refs"] != [reference["verifier_ref"]]):
                raise ProfilePreparationRefused("profile receipt original profile reference differs")
            for name in ("profile_event_digest", "target_profile_digest"):
                state._sha(reference[name], name)
            state._sha(terminal["result_digest"], "profile terminal result")
        elif (reference is not None or settlement["outcome"] != "failed"
                or settlement["terminal_refs"] != ["lifecycle:" + state.digest(terminal)]):
            raise ProfilePreparationRefused("failed profile receipt cannot carry a profile")
        accepted = row["settlement_receipt"]
        if (not isinstance(accepted, Mapping)
                or set(accepted) != {"schema", "transition_id", "status", "accounting_projection_digest"}
                or accepted["schema"] != cc.DRIVER_SETTLEMENT_RECEIPT_SCHEMA
                or accepted["transition_id"] != work.transition_id
                or accepted["status"] not in {"accepted", "duplicate"}):
            raise ProfilePreparationRefused("profile receipt durable settlement differs")
        state._sha(accepted["accounting_projection_digest"], "accounting projection")
        object.__setattr__(self, "body", ud._freeze(row))

    def to_dict(self):
        row = _plain(self.body)
        return row | {"receipt_digest": state.digest(row)}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, Mapping) or "receipt_digest" not in value:
            raise ProfilePreparationRefused("profile reporting receipt lacks integrity digest")
        row = _plain(value)
        digest = row.pop("receipt_digest")
        if digest != state.digest(row):
            raise ProfilePreparationRefused("profile reporting receipt digest differs")
        return cls(row)


@dataclass(frozen=True)
class _OriginalAttempt:
    work: ud.SelectedProfileWork
    terminal: wl.TerminalWorker
    held: scheduling.HeldClaimReceipt
    profile_reference: Mapping[str, Any] | None
    settlement: Mapping[str, Any]


def _current_publication(snapshot, work, terminal, started, domain, expires):
    if snapshot is None:
        return None
    row = snapshot["profile_event"]
    if (row["catalog_id"] != work.catalog_id or row["transition_id"] != work.transition_id
            or row["profile_request_digest"] != state.digest(work.profile_request.to_dict())
            or row["stage_plan_digest"] != work.stage_plan_digest
            or row["target_revision_digest"] != work.profile_request.target_revision_digest
            or row["verifier_ref"] != f"controller-worker:{terminal.worker_id}:{terminal.worker_generation}"
            or terminal.return_code != 0 or not terminal.accepted
            or row["verified_at"] != started or row["valid_until"] != expires
            or row["clock_domain"] != domain):
        return None
    return row


def _selected_implementation(controller, producer):
    for instance, owner, names in (
            (producer, tp.TargetProfileExecution, ("prepare",)),
            (controller, cc.CampaignController, (
                "reserve_target_profile_execution", "record_verified_target_profile",
                "finish_target_profile_execution", "current_verified_profile_result",
                "actor_held_claim_receipt", "read_worker_stdout", "unified_driver_settle"))):
        for name in names:
            selected = getattr(instance, name)
            if (getattr(selected, "__self__", None) is not instance
                    or getattr(selected, "__func__", None) is not getattr(owner, name)):
                raise ProfilePreparationRefused("selected profile implementation differs: " + name)


class InstalledProfilePreparationOwner:
    def __init__(self, controller, binding, producer, configured_profiles):
        if (type(controller) is not cc.CampaignController
                or type(binding) is not InstalledProfilePreparationBinding
                or type(producer) is not tp.TargetProfileExecution
                or producer.controller is not controller):
            raise ProfilePreparationRefused("profile owner installation is not concrete")
        self.controller, self.binding, self.producer = controller, binding, producer
        self._configured = MappingProxyType({target: up.TargetProfile.from_dict(
            value.to_dict() if isinstance(value, up.TargetProfile) else _plain(value))
            for target, value in configured_profiles.items()})
        self._attempts = {}
        self._receipts = {}
        self._entered = set()
        self._pending = {}
        self._outcomes = {}
        self._lock = threading.RLock()

    def owns_settlement(self, transition_id):
        with self._lock:
            return transition_id in self._attempts

    def _profile(self, target):
        _selected_implementation(self.controller, self.producer)
        snapshot = self.controller.current_verified_profile_result(target)
        if snapshot is None:
            return None
        result = _plain(snapshot)
        row = state.validate_event(result["profile_event"])
        binding = self.binding.mechanisms.get(target)
        if binding is None:
            raise ProfilePreparationRefused("original profile lacks installed target")
        request = ud.ProfilePreparationRequest.from_dict(row["profile_request"])
        binding.validate_request(request)
        if (row["loaded_identity"] != dict(binding.mechanism.loaded_identity)
                or row["config_digest"] != self.controller.config_digest
                or row["target_revision_digest"] != target):
            raise ProfilePreparationRefused("original profile loaded identity differs")
        issued = result["issued"]
        if (not isinstance(issued, Mapping) or issued["catalog_id"] != row["catalog_id"]
                or issued["transition_id"] != row["transition_id"]):
            raise ProfilePreparationRefused("original profile issuance is missing")
        selection = scheduling.Selection.from_dict(issued["selection"])
        if selection.proposal != request.stage_proposal:
            raise ProfilePreparationRefused("original profile selected proposal differs")
        work = issued["catalog"]["work_by_stage_digest"].get(selection.proposal.digest)
        if (work is None or work["kind"] != "profile_preparation"
                or work["payload"] != request.to_dict()
                or work["stage_plan_digest"] != row["stage_plan_digest"]):
            raise ProfilePreparationRefused("original profile catalog work differs")
        return result

    def planner_profiles(self, now):
        now = state._finite(now, "profile planning time")
        domain = wl.monotonic_clock_domain()
        profiles = dict(self._configured)
        observations = []
        observation_error = None
        for target in self.binding.mechanisms:
            snapshot = self._profile(target)
            try:
                if len(observations) < 24:
                    observations.append(self._profile_observation_row(target, snapshot, domain, now))
            except Exception as exc:
                observation_error = exc
            if snapshot is None:
                continue
            # A generated original supersedes a configured snapshot; expiry
            # must not fall back to the obsolete configured row.
            profiles.pop(target, None)
            row, settled = snapshot["profile_event"], snapshot["settlement"]
            if (settled is None or settled["outcome"] != "prerequisite"
                    or settled["catalog_id"] != row["catalog_id"]
                    or settled["transition_id"] != row["transition_id"]
                    or settled["selection"] != snapshot["issued"]["selection"]
                    or settled["terminal_refs"] != [row["verifier_ref"]]
                    or settled["supervisor_incarnation"] != row["supervisor_incarnation"]
                    or row["clock_domain"] != domain
                    or not row["verified_at"] <= now < row["valid_until"]):
                continue
            held = scheduling.HeldClaimReceipt.from_dict(settled["receipt"])
            selected = scheduling.Selection.from_dict(settled["selection"]).proposal
            if (held.proposal_id != selected.proposal_id or held.backend != selected.backend
                    or held.stage_class != selected.stage_class):
                raise ProfilePreparationRefused("original profile settlement accounting differs")
            # Reopen the original schema; never construct missing planner fields.
            try:
                profile = up.TargetProfile.from_dict(row["profile_content"])
            except (ValueError, TypeError):
                continue
            if (profile.target_revision_digest != target
                    or profile.quant != row["loaded_identity"]["quantization"]):
                raise ProfilePreparationRefused("profile content target/quant differs")
            profiles[target] = profile
        result = MappingProxyType(profiles)
        try:
            from . import runtime_aggregates as aggregate
            if observation_error is not None:
                raise observation_error
            for item in observations:
                item["available_at_planning"] = item["target_revision"] in result
            data = aggregate.freeze({
                "configured_count": len(self._configured), "usable_count": len(result),
                "mechanism_count": len(self.binding.mechanisms), "debt_count": None,
                "planning_observed_at": aggregate.utc_now(), "items": observations,
                "items_total": len(self.binding.mechanisms),
                "items_truncated": len(self.binding.mechanisms) > len(observations)})
            self._observation_view = result
            self._observation_profiles = data
            self._observation_attempted_at = data["planning_observed_at"]
            self._observation_error = None
            self._observation_generation = getattr(self, "_observation_generation", 0) + 1
        except Exception as exc:
            self._profile_observation_failed(exc)
        return result

    @staticmethod
    def _profile_observation_row(target, snapshot, domain, now):
        row = None if snapshot is None else snapshot["profile_event"]
        same_clock = row is not None and row["clock_domain"] == domain
        return {"target_revision": target,
            "profile_digest": None if row is None else row["target_profile_digest"],
            "transition_id": None if row is None else row["transition_id"],
            "available_at_planning": False,
            "settled": snapshot is not None and snapshot["settlement"] is not None,
            "remaining_seconds": max(0., row["valid_until"] - now) if same_clock else None,
            "clock_known": same_clock, "consumed_request_debt": None}

    def _profile_observation_failed(self, exc):
        try:
            from . import runtime_aggregates as aggregate
            self._observation_error = str(exc)[:512] or type(exc).__name__
            self._observation_attempted_at = aggregate.utc_now()
        except Exception:
            pass  # retain the earlier immutable cache, never alter profile selection

    def consumed_request_debt(self, requests, profiles):
        """Exclude an already consumed fixed request without inventing renewal."""
        debt = {}
        for target, request in requests.items():
            if target in profiles or target not in self.binding.mechanisms:
                continue
            snapshot = self._profile(target)
            if (snapshot is not None
                    and snapshot["profile_event"]["profile_request_digest"]
                        == state.digest(request.to_dict())):
                debt[target] = (f"target:{target}:profile_refresh_unavailable:"
                    "original_request_consumed:fresh_predeclared_request_required")
        result = MappingProxyType(debt)
        try:
            self._record_profile_debt(profiles, result)
        except Exception as exc:
            self._profile_observation_failed(exc)
        return result

    def _record_profile_debt(self, profiles, debt):
        if profiles is getattr(self, "_observation_view", None):
            from . import runtime_aggregates as aggregate
            data = aggregate.plain(self._observation_profiles)
            data["debt_count"] = len(debt)
            for item in data["items"]:
                item["consumed_request_debt"] = item["target_revision"] in debt
            self._observation_profiles = aggregate.freeze(data)

    def observation_snapshot(self):
        """Only the latest actual owning reduction; never rerun it for publication."""
        from . import runtime_aggregates as aggregate
        data = getattr(self, "_observation_profiles", None)
        stamp = None if data is None else data["planning_observed_at"]
        error = getattr(self, "_observation_error", None)
        return aggregate.observation("profile", data=data, observed_at=stamp,
            attempted_at=getattr(self, "_observation_attempted_at", stamp), error=error,
            generation=getattr(self, "_observation_generation", 0),
            status="available" if data is not None and data["debt_count"] is not None and error is None else "unknown",
            reason="original settled profile reduction; counts apply at planning observation")

    def execute(self, driver, outcome):
        if type(driver) is not ud.UnifiedCampaignDriver or driver.controller is not self.controller:
            raise ProfilePreparationRefused("profile driver/controller differs")
        if type(outcome) is not ud.DriverOutcome or outcome.status != "intent_recorded":
            raise ProfilePreparationRefused("profile requires exact issued DriverOutcome")
        self.controller.unified_driver_readiness()
        _selected_implementation(self.controller, self.producer)
        transition = outcome.transition_id
        with self._lock:
            if transition in self._outcomes and self._outcomes[transition] != ud._freeze(outcome.to_dict()):
                raise ProfilePreparationRefused("profile retry DriverOutcome identity differs")
            if transition in self._receipts:
                return self._receipts[transition]
            attempt = self._attempts.get(transition)
        if attempt is not None:
            return self._settle(attempt)
        work = driver.materialize_profile(outcome)
        binding = self.binding.mechanisms.get(work.profile_request.target_revision_digest)
        if binding is None:
            raise ProfilePreparationRefused("selected profile mechanism unavailable")
        binding.validate_request(work.profile_request)
        if self.producer.mechanisms.get(work.profile_request.target_revision_digest) is not binding.mechanism:
            raise ProfilePreparationRefused("selected producer mechanism is not installed binding")
        with self._lock:
            pending = self._pending.get(transition)
            if pending is not None and pending[0] != work:
                raise ProfilePreparationRefused("pending original profile work differs")
        launch = pending is None
        if launch:
            started, domain = time.monotonic(), wl.monotonic_clock_domain()
            expires = started + binding.valid_for_seconds
            if not math.isfinite(expires):
                raise ProfilePreparationRefused("profile validity overflow")
            with self._lock:
                if transition in self._entered:
                    raise cc.ControlRefused("profile original attempt in flight; no second launch")
                self._entered.add(transition)
                self._pending[transition] = (work, started, domain, expires)
                self._outcomes[transition] = ud._freeze(outcome.to_dict())
        else:
            _, started, domain, expires = pending
        failure = None
        if launch:
            try:
                self.producer.prepare(
                    profile_request=work.profile_request.to_dict(), catalog_id=work.catalog_id,
                    transition_id=work.transition_id, stage_plan_digest=work.stage_plan_digest,
                    clock_domain=domain, verified_at=started, valid_until=expires,
                    selected_work=work)
            except Exception as exc:
                failure = exc
        request_digest = state.digest(work.profile_request.to_dict())
        terminal = self.controller.worker_terminal_for_request(
            request_id="profile-" + request_digest[:24], plan_digest=work.stage_plan_digest,
            lineage_id=work.transition_id, stage_id="target-profile-" + request_digest[:24])
        if terminal is None:
            raise cc.ControlRefused("profile lacks original terminal/held issuance") from failure
        held = self.controller.actor_held_claim_receipt(terminal)
        snapshot = self._profile(work.profile_request.target_revision_digest)
        row = _current_publication(snapshot, work, terminal, started, domain, expires)
        reference = None if row is None else ud._freeze({
            "profile_event_digest": state.digest(row),
            **{name: row[name] for name in ("target_profile_digest", "profile_request_digest",
                                          "catalog_id", "transition_id", "verifier_ref",
                                          "target_revision_digest")}})
        from .driver_execution import _terminal_body
        terminal_ref = (row["verifier_ref"] if row is not None else
                        "lifecycle:" + state.digest(_terminal_body(terminal)))
        settlement = ud._freeze({"schema": cc.DRIVER_SETTLEMENT_SCHEMA,
            "catalog_id": work.catalog_id, "transition_id": work.transition_id,
            "selection": work.selection.to_dict(), "receipt": held.to_dict(),
            "outcome": "prerequisite" if row is not None else "failed",
            "terminal_refs": [terminal_ref]})
        attempt = _OriginalAttempt(work, terminal, held, reference, settlement)
        with self._lock:
            self._attempts[transition] = attempt
        return self._settle(attempt)

    def verify_settlement(self, supplied):
        with self._lock:
            attempt = self._attempts.get(supplied.get("transition_id"))
        if attempt is None or _plain(attempt.settlement) != dict(supplied):
            raise ProfilePreparationRefused("settlement lacks original profile attempt")
        terminal = self.controller.worker_terminal_for_request(
            request_id=attempt.terminal.request_id, plan_digest=attempt.terminal.plan_digest,
            lineage_id=attempt.terminal.lineage_id, stage_id=attempt.terminal.stage_id)
        if (terminal != attempt.terminal
                or self.controller.actor_held_claim_receipt(terminal) != attempt.held):
            raise ProfilePreparationRefused("profile original terminal/held receipt differs")
        if attempt.profile_reference is not None:
            original = self._profile(attempt.work.profile_request.target_revision_digest)
            if (original is None or state.digest(original["profile_event"])
                    != attempt.profile_reference["profile_event_digest"]):
                raise ProfilePreparationRefused("profile original publication differs")
        return _plain(attempt.settlement)

    def _settle(self, attempt):
        from .driver_execution import _terminal_body, DriverExecutionUncertain
        try:
            settled = self.controller.unified_driver_settle(_plain(attempt.settlement))
        except Exception as exc:
            raise DriverExecutionUncertain(
                "profile settlement unresolved; original provider binding/held cost required") from exc
        receipt = ProfilePreparationExecutionReceipt({
            "schema": RECEIPT_SCHEMA, "catalog_id": attempt.work.catalog_id,
            "transition_id": attempt.work.transition_id,
            "selected_work": attempt.work.to_dict(),
            "selected_work_digest": state.digest(attempt.work.to_dict()),
            "request_id": attempt.terminal.request_id, "stage_id": attempt.terminal.stage_id,
            "terminal": _terminal_body(attempt.terminal),
            "provider_held_receipt": attempt.held.to_dict(),
            "profile_reference": _plain(attempt.profile_reference),
            "disposition": "profile_verified" if attempt.profile_reference else "failed",
            "settlement_request": _plain(attempt.settlement), "settlement_receipt": settled})
        with self._lock:
            self._receipts[attempt.work.transition_id] = receipt
        return receipt

    def recover_issued(self, driver, outcome):
        if (type(driver) is not ud.UnifiedCampaignDriver or driver.controller is not self.controller
                or driver.issued_work_kind(outcome) != "profile_preparation"):
            raise ProfilePreparationRefused("profile recovery requires the original typed work")
        self.controller.reconcile_workers()
        with self._lock:
            attempt = self._attempts.get(outcome.transition_id)
        if attempt is not None:
            return self.execute(driver, outcome)
        raise cc.ControlRefused(
            "profile recovery lacks original held issuance; no reacquisition or second child")
