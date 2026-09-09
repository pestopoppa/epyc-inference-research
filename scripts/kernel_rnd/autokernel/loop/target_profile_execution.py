"""Selected target-profile execution through the campaign-controller worker owner."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from . import actor_preparation_state as state
from . import campaign_control, scheduling, worker_lifecycle
from . import observation_binding

PROFILE_REQUEST_SCHEMA = "epyc.autokernel.profile_preparation_request.v1"
PROFILE_OUTPUT_SCHEMA = "epyc.autokernel.target_profile_output.v1"
PROFILE_SOURCE_ID = "VB-AK-UNIFIED-PROFILE"
VALIDATION_SOURCE_ID = "VB-AK-UNIFIED-VALIDATION"


class ProfileExecutionRefused(ValueError):
    pass


@dataclass(frozen=True)
class TargetProfileExecutionReservation:
    """Lifetime-bound controller admission; its fields grant no authority."""

    reservation_id: str
    catalog_id: str
    transition_id: str
    profile_request_digest: str
    stage_plan_digest: str
    request_id: str
    stage_id: str
    supervisor_incarnation: int
    control_revision: int
    max_output_bytes: int
    _capability: object


@dataclass(frozen=True)
class ProfileMechanism:
    mechanism_id: str
    binary: Path
    binary_sha256: str
    cwd: Path
    env: Mapping[str, str]
    loaded_identity: Mapping[str, Any]
    max_stage_seconds: float
    teardown_seconds: float
    max_output_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.mechanism_id, str) or not self.mechanism_id:
            raise ProfileExecutionRefused("profile mechanism_id must be nonempty")
        for path, label in ((self.binary, "binary"), (self.cwd, "cwd")):
            if not isinstance(path, Path) or not path.is_absolute():
                raise ProfileExecutionRefused(f"profile {label} must be absolute")
        state._sha(self.binary_sha256, "profile binary_sha256")
        if not isinstance(self.env, Mapping) or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in self.env.items()):
            raise ProfileExecutionRefused("profile environment must contain text pairs")
        required = {"target_revision_digest", "model_digest", "quantization",
                    "recipe_digest", "executable_digest", "dso_digest"}
        if not isinstance(self.loaded_identity, Mapping) or set(self.loaded_identity) != required:
            raise ProfileExecutionRefused("loaded identity fields differ")
        for name in required - {"quantization"}:
            state._sha(self.loaded_identity[name], f"loaded_identity.{name}")
        if not isinstance(self.loaded_identity["quantization"], str) \
                or not self.loaded_identity["quantization"]:
            raise ProfileExecutionRefused("loaded identity quantization must be nonempty")
        state._finite(self.max_stage_seconds, "max_stage_seconds", minimum=1e-12)
        state._finite(self.teardown_seconds, "teardown_seconds", minimum=1e-12)
        if (isinstance(self.max_output_bytes, bool) or not isinstance(self.max_output_bytes, int)
                or self.max_output_bytes < 1):
            raise ProfileExecutionRefused("max_output_bytes must be positive")
        object.__setattr__(self, "env", MappingProxyType(dict(self.env)))
        object.__setattr__(self, "loaded_identity", MappingProxyType(dict(self.loaded_identity)))


def _profile_request(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProfileExecutionRefused("profile request must be an object")
    row = dict(value)
    if (set(row) != {"schema", "target_revision_digest", "stage_proposal",
                     "profile_contract"}
            or row["schema"] != PROFILE_REQUEST_SCHEMA
            or not isinstance(row["stage_proposal"], Mapping)
            or not isinstance(row["profile_contract"], Mapping)):
        raise ProfileExecutionRefused("profile request fields/schema differ")
    state._sha(row["target_revision_digest"], "target_revision_digest")
    if row["stage_proposal"].get("target_revision") != row["target_revision_digest"]:
        raise ProfileExecutionRefused("profile stage target differs")
    return row


class TargetProfileExecution:
    """Execute one configured profiler and publish its immutable native result."""

    def __init__(self, *, controller: campaign_control.CampaignController,
                 mechanism: ProfileMechanism) -> None:
        if not isinstance(controller, campaign_control.CampaignController):
            raise TypeError("controller must be CampaignController")
        if not isinstance(mechanism, ProfileMechanism):
            raise ProfileExecutionRefused("target profiling mechanism is unsupported/absent")
        self.controller = controller
        self.mechanism = mechanism

    def prepare(self, *, profile_request: Mapping[str, Any], catalog_id: str,
                transition_id: str, stage_plan_digest: str, clock_domain: str,
                verified_at: float, valid_until: float,
                selected_work: Any = None) -> Mapping[str, Any]:
        request = _profile_request(profile_request)
        for value, label in ((catalog_id, "catalog_id"), (transition_id, "transition_id"),
                             (stage_plan_digest, "stage_plan_digest")):
            state._sha(value, label)
        if request["target_revision_digest"] != self.mechanism.loaded_identity[
                "target_revision_digest"]:
            raise ProfileExecutionRefused("mechanism loaded identity targets another revision")
        request_digest = state.digest(request)
        reservation = self.controller.reserve_target_profile_execution(
            profile_request=request, profile_request_digest=request_digest,
            catalog_id=catalog_id, transition_id=transition_id,
            stage_plan_digest=stage_plan_digest,
            max_output_bytes=self.mechanism.max_output_bytes,
            selected_work=selected_work)
        try:
            current_sha = observation_binding._artifact_identity(self.mechanism.binary)["sha256"]
        except (OSError, observation_binding.ObservationBindingError) as exc:
            self.controller.cancel_target_profile_execution(reservation)
            raise ProfileExecutionRefused("profiling mechanism is unreadable") from exc
        if current_sha != self.mechanism.binary_sha256:
            self.controller.cancel_target_profile_execution(reservation)
            raise ProfileExecutionRefused("profiling mechanism bytes changed")
        worker_request = worker_lifecycle.StageRequest(
            request_id=reservation.request_id, plan_digest=stage_plan_digest,
            lineage_id=transition_id, stage_id=reservation.stage_id, stage="setup",
            argv=(str(self.mechanism.binary), json.dumps(
                request, sort_keys=True, separators=(",", ":"))),
            env=dict(self.mechanism.env), cwd=self.mechanism.cwd,
            artifact_contract_digest=state.digest({
                "profile_request": request_digest,
                "loaded_identity": dict(self.mechanism.loaded_identity)}),
            max_stage_seconds=self.mechanism.max_stage_seconds,
            teardown_seconds=self.mechanism.teardown_seconds,
            control_revision=self.controller.control_revision)
        terminal = self.controller.run_worker_stage(worker_request)
        exact = self.controller.worker_terminal_for_request(
            request_id=worker_request.request_id, plan_digest=worker_request.plan_digest,
            lineage_id=worker_request.lineage_id, stage_id=worker_request.stage_id)
        if terminal != exact or terminal is None or terminal.result_digest is None:
            raise ProfileExecutionRefused("profile worker lacks an exact terminal")
        held = self.controller.actor_held_claim_receipt(terminal)
        if (not isinstance(held, scheduling.HeldClaimReceipt)
                or held.ownership_generation != terminal.worker_generation
                or held.allocation_generation != terminal.grant_generation
                or not math.isfinite(held.ended_at - held.started_at)
                or held.ended_at <= held.started_at):
            raise ProfileExecutionRefused("profile worker lacks provider-authored held cost")
        if terminal.return_code != 0 or not terminal.accepted:
            self.controller.finish_target_profile_execution(
                reservation=reservation, terminal=terminal,
                provider_cost_receipt=held)
            raise ProfileExecutionRefused("profile worker terminal was not successful")
        try:
            raw = self.controller.read_worker_stdout(
                request_id=terminal.request_id, plan_digest=terminal.plan_digest,
                lineage_id=terminal.lineage_id, stage_id=terminal.stage_id,
                worker_id=terminal.worker_id, worker_generation=terminal.worker_generation,
                result_digest=terminal.result_digest,
                max_bytes=self.mechanism.max_output_bytes)
        except Exception as exc:
            self.controller.finish_target_profile_execution(
                reservation=reservation, terminal=terminal,
                provider_cost_receipt=held)
            raise ProfileExecutionRefused("profile output is unavailable") from exc
        try:
            output = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self.controller.finish_target_profile_execution(
                reservation=reservation, terminal=terminal,
                provider_cost_receipt=held)
            raise ProfileExecutionRefused("profile output is not one UTF-8 JSON object") from exc
        if (not isinstance(output, Mapping)
                or set(output) != {"schema", "profile_content", "loaded_identity",
                                   "artifact_identity", "measurement_carrier"}
                or output["schema"] != PROFILE_OUTPUT_SCHEMA
                or output["loaded_identity"] != dict(self.mechanism.loaded_identity)):
            self.controller.finish_target_profile_execution(
                reservation=reservation, terminal=terminal,
                provider_cost_receipt=held)
            raise ProfileExecutionRefused("profile output fields/loaded identity differ")
        carrier = output["measurement_carrier"]
        if (not isinstance(carrier, Mapping)
                or set(carrier) != {"schema", "profile_source_id",
                                    "validation_source_id", "run_id",
                                    "profile_claim_tuple", "validation_claim_tuple"}
                or carrier.get("schema")
                   != "epyc.autokernel.profile_measurement_carrier.v1"
                or carrier.get("profile_source_id") != PROFILE_SOURCE_ID
                or carrier.get("validation_source_id") != VALIDATION_SOURCE_ID
                or not isinstance(carrier.get("run_id"), str)
                or not carrier["run_id"]
                or not isinstance(carrier.get("profile_claim_tuple"), Mapping)
                or not carrier["profile_claim_tuple"]
                or not isinstance(carrier.get("validation_claim_tuple"), Mapping)
                or not carrier["validation_claim_tuple"]):
            self.controller.finish_target_profile_execution(
                reservation=reservation, terminal=terminal,
                provider_cost_receipt=held)
            raise ProfileExecutionRefused("profile output lacks registered write-side carriers")
        profile_digest = state.digest({
            "profile_content": output["profile_content"],
            "loaded_identity": output["loaded_identity"],
            "artifact_identity": output["artifact_identity"]})
        event = {
            "schema": state.PROFILE_SCHEMA, "event": "PROFILE_VERIFIED",
            "campaign_id": self.controller.resolved.campaign_id,
            "config_generation": self.controller.config_generation,
            "config_digest": self.controller.config_digest,
            "supervisor_id": self.controller._supervisor_id,
            "supervisor_incarnation": self.controller.supervisor_incarnation,
            "control_revision": self.controller.control_revision,
            "catalog_id": catalog_id, "transition_id": transition_id,
            "stage_plan_digest": stage_plan_digest,
            "profile_request_digest": request_digest,
            "target_revision_digest": request["target_revision_digest"],
            "target_profile_digest": profile_digest,
            "profile_request": request, "profile_content": dict(output["profile_content"]),
            "artifact_identity": dict(output["artifact_identity"]),
            "loaded_identity": dict(output["loaded_identity"]),
            "measurement_carrier": dict(carrier), "clock_domain": clock_domain,
            "verifier_ref": f"controller-worker:{terminal.worker_id}:{terminal.worker_generation}",
            "verified_at": verified_at, "valid_until": valid_until,
            "occurred_at": self.controller.clock(),
        }
        try:
            return self.controller.record_verified_target_profile(
                event, reservation=reservation, terminal=terminal,
                provider_cost_receipt=held)
        except Exception:
            self.controller.finish_target_profile_execution(
                reservation=reservation, terminal=terminal,
                provider_cost_receipt=held)
            raise

    def verified_target_profile(self, **request):
        """Return only a persisted current-owner profile matching this admission."""
        receipt = self.controller.current_actor_profile(
            request.get("target_revision_digest"))
        if receipt is None:
            return None
        if (receipt.campaign_digest != request.get("campaign_digest")
                or receipt.profile_request_digest
                   != state.digest(receipt.to_dict()["profile_request"])
                or receipt.clock_domain != request.get("clock_domain")
                or receipt.verified_at > request.get("now")
                or receipt.valid_until <= request.get("now")):
            return None
        return receipt


__all__ = ["PROFILE_OUTPUT_SCHEMA", "PROFILE_SOURCE_ID", "VALIDATION_SOURCE_ID",
           "ProfileExecutionRefused", "ProfileMechanism", "TargetProfileExecution",
           "TargetProfileExecutionReservation"]
