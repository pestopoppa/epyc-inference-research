"""Concrete actor-stage adapter over the owning campaign-controller path.

Selection/profile/budget persistence remains an injected parent callback because its
Journal schema and controller transaction are primary-owned. Process creation and
descendant cleanup are performed only below ``CampaignController.run_worker_stage``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol

from . import actor_preparation as preparation
from . import actors, campaign_control, scheduling, worker_lifecycle
from . import observation_binding

TARGET_PROFILE_RECEIPT_SCHEMA = "epyc.autokernel.target_profile_receipt.v1"
_ACTOR_ENV_KEYS = frozenset({"LANG", "LC_ALL", "LC_CTYPE", "PATH", "TMPDIR"})


class ActorInvocationUncertain(BaseException):
    """Controller may have crossed provider admission; retain INTENT for recovery."""


@dataclass(frozen=True)
class TargetProfileReceipt:
    """Verified result of the distinct, primary-owned target-profile job.

    Construction is not authority. The controller supplies these typed receipts from
    its selected persistence fold; the adapter refuses mappings and missing entries.
    """

    campaign_digest: str
    profile_request: Mapping[str, Any]
    profile_request_digest: str
    target_revision_digest: str
    target_profile_digest: str
    verified_at: float
    valid_until: float
    clock_domain: str
    verifier_ref: str
    status: str = "verified"
    schema: str = TARGET_PROFILE_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        from .unified_driver import _freeze, _thaw

        if self.schema != TARGET_PROFILE_RECEIPT_SCHEMA or self.status != "verified":
            raise preparation.PreparationRefused(
                "target-profile receipt schema/status is not verified")
        for name in ("campaign_digest", "profile_request_digest",
                     "target_revision_digest", "target_profile_digest"):
            preparation._sha256(getattr(self, name), f"target-profile receipt {name}")
        for name in ("verified_at", "valid_until"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value)):
                raise preparation.PreparationRefused(
                    f"target-profile receipt {name} must be finite")
        preparation._text(self.clock_domain, "target-profile receipt clock_domain")
        preparation._text(self.verifier_ref, "target-profile receipt verifier_ref")
        profile_request = preparation._mapping(
            _thaw(self.profile_request), "target-profile receipt profile_request")
        if (set(profile_request) != {"schema", "target_revision_digest", "stage_proposal",
                                    "profile_contract"}
                or profile_request["schema"] !=
                "epyc.autokernel.profile_preparation_request.v1"
                or profile_request["target_revision_digest"] != self.target_revision_digest
                or preparation._digest(profile_request) != self.profile_request_digest):
            raise preparation.PreparationRefused(
                "target-profile receipt does not bind an exact ProfilePreparationRequest")
        object.__setattr__(self, "profile_request", _freeze(profile_request))

    def to_dict(self) -> dict[str, Any]:
        from .unified_driver import _thaw

        return {"schema": self.schema, "status": self.status,
                "campaign_digest": self.campaign_digest,
                "profile_request": _thaw(self.profile_request),
                "profile_request_digest": self.profile_request_digest,
                "target_revision_digest": self.target_revision_digest,
                "target_profile_digest": self.target_profile_digest,
                "verified_at": self.verified_at, "valid_until": self.valid_until,
                "clock_domain": self.clock_domain, "verifier_ref": self.verifier_ref}

    @property
    def digest(self) -> str:
        return preparation._digest(self.to_dict())


class ActorPersistence(Protocol):
    """Exact primary-owned selection, budget and settlement seam.

    ``reserve_actor_preparation`` must atomically validate the selected driver
    catalog/transition and fresh ProfilePreparationRequest result, append durable
    preparation/budget INTENT, and return the exact reservation or typed availability.
    ``finish_actor_preparation`` must idempotently settle actual charged time and the
    terminal availability/disposition for that reservation.
    """

    def reserve_actor_preparation(
            self, *, request: Mapping[str, Any], request_digest: str,
            stage_plan_digest: str, actor_profile: Mapping[str, Any],
            actor_profile_digest: str, target_profile_receipt: Mapping[str, Any],
            target_profile_receipt_digest: str, budgets: Mapping[str, int],
            now: float, clock_domain: str) -> Mapping[str, Any]: ...

    def finish_actor_preparation(
            self, *, reservation: Mapping[str, Any], outcome: Mapping[str, Any],
            disposition: str, provider_cost_receipt: Any | None) -> None: ...


class TargetProfileOwner(Protocol):
    """Producer-owned capability; receipt serialization alone grants no authority."""

    def verified_target_profile(
            self, *, request: Mapping[str, Any], request_digest: str,
            stage_plan_digest: str, campaign_digest: str,
            target_revision_digest: str, now: float,
            clock_domain: str) -> TargetProfileReceipt | None: ...


@dataclass(frozen=True)
class ActorLifecycleConfig:
    campaign_digest: str
    cwd: Path
    env: Mapping[str, str]
    max_stage_seconds: float
    teardown_seconds: float
    max_retained_output_bytes: int

    def __post_init__(self) -> None:
        preparation._sha256(self.campaign_digest, "actor lifecycle campaign_digest")
        if not isinstance(self.cwd, Path) or not self.cwd.is_absolute():
            raise preparation.PreparationRefused("actor lifecycle cwd must be absolute")
        if not isinstance(self.env, Mapping):
            raise preparation.PreparationRefused("actor lifecycle env must be a mapping")
        if (set(self.env) - _ACTOR_ENV_KEYS
                or any(not isinstance(item, str) for item in self.env.values())):
            raise preparation.PreparationRefused(
                "actor lifecycle env contains a non-allowlisted key or non-text value")
        for value, label in ((self.max_stage_seconds, "max_stage_seconds"),
                             (self.teardown_seconds, "teardown_seconds")):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise preparation.PreparationRefused(f"{label} must be finite and positive")
        if (isinstance(self.max_retained_output_bytes, bool)
                or not isinstance(self.max_retained_output_bytes, int)
                or self.max_retained_output_bytes <= 0):
            raise preparation.PreparationRefused(
                "max_retained_output_bytes must be positive")
        object.__setattr__(self, "env", MappingProxyType(dict(self.env)))


class ActorLifecycleAdapter:
    """Translate preparation calls through the real campaign-controller owner."""

    def __init__(self, *, controller: campaign_control.CampaignController,
                 persistence: ActorPersistence,
                 config: ActorLifecycleConfig,
                 target_profile_owner: TargetProfileOwner | None = None) -> None:
        if not isinstance(controller, campaign_control.CampaignController):
            raise TypeError("controller must be CampaignController")
        for method in ("run_worker_stage", "worker_terminal_for_request",
                       "actor_held_claim_receipt", "read_worker_stdout"):
            if not callable(getattr(controller, method, None)):
                raise TypeError(f"controller lacks required {method} owner method")
        if (not callable(getattr(persistence, "reserve_actor_preparation", None))
                or not callable(getattr(persistence, "finish_actor_preparation", None))):
            raise TypeError("persistence lacks actor preparation methods")
        self.controller = controller
        self.persistence = persistence
        self.config = config
        if target_profile_owner is not None and not callable(
                getattr(target_profile_owner, "verified_target_profile", None)):
            raise TypeError("target_profile_owner lacks producer verification method")
        self.target_profile_owner = target_profile_owner
        self._bindings: dict[str, tuple[dict[str, Any], preparation.ActorProfile]] = {}
        self._invoked: set[str] = set()
        self._held_costs: dict[str, Any] = {}

    def reserve(self, *, request: Mapping[str, Any], stage_plan_digest: str,
                actor_profile: Mapping[str, Any], budgets: Mapping[str, int],
                now: float, clock_domain: str) -> Mapping[str, Any]:
        profile = preparation.ActorProfile.from_dict(actor_profile)
        request_row = dict(request)
        request_digest = preparation._digest(request_row)
        proposal = preparation._mapping(request_row.get("proposal"), "actor proposal")
        target_revision = preparation._sha256(
            proposal.get("target_revision_digest"), "target revision digest")
        if self.target_profile_owner is None:
            raise preparation.PreparationRefused(
                "verified target-profile producer capability is unavailable")
        receipt = self.target_profile_owner.verified_target_profile(
            request=request_row, request_digest=request_digest,
            stage_plan_digest=stage_plan_digest,
            campaign_digest=self.config.campaign_digest,
            target_revision_digest=target_revision, now=now,
            clock_domain=clock_domain)
        if not isinstance(receipt, TargetProfileReceipt):
            raise preparation.PreparationRefused(
                "target-profile producer returned no typed verified receipt")
        if (receipt.target_revision_digest != target_revision
                or receipt.campaign_digest != self.config.campaign_digest
                or receipt.clock_domain != clock_domain
                or receipt.verified_at > now or receipt.valid_until <= now):
            raise preparation.PreparationRefused(
                "target-profile receipt is stale or differs from selected binding")
        raw = self.persistence.reserve_actor_preparation(
            request=request_row, request_digest=request_digest,
            stage_plan_digest=stage_plan_digest, actor_profile=profile.to_dict(),
            actor_profile_digest=profile.digest,
            target_profile_receipt=receipt.to_dict(),
            target_profile_receipt_digest=receipt.digest,
            budgets=dict(budgets), now=now, clock_domain=clock_domain)
        if not isinstance(raw, Mapping):
            raise preparation.PreparationRefused(
                "actor persistence returned an untyped reservation")
        if raw.get("schema") == preparation.RESERVATION_SCHEMA:
            reservation = preparation.StageReservation.from_dict(raw)
            if (reservation.target_profile_digest != receipt.target_profile_digest
                    or reservation.target_profile_receipt_digest != receipt.digest):
                raise preparation.PreparationRefused(
                    "reservation differs from verified target-profile receipt")
            if reservation.reservation_id in self._bindings:
                old_request, old_profile = self._bindings[reservation.reservation_id]
                if old_request != request_row or old_profile != profile:
                    raise preparation.PreparationRefused(
                        "reservation id was reused for different actor bytes")
            self._bindings[reservation.reservation_id] = (request_row, profile)
        return raw

    def invoke(self, reservation: preparation.StageReservation,
               backend: actors.Backend, prompt: str) -> Mapping[str, Any]:
        binding = self._bindings.get(reservation.reservation_id)
        if binding is None:
            raise preparation.PreparationRefused("actor reservation was not issued by adapter")
        if reservation.reservation_id in self._invoked:
            raise preparation.PreparationRefused(
                "actor reservation is one-shot; settlement/recovery cannot re-invoke")
        self._invoked.add(reservation.reservation_id)
        _request, profile = binding
        if backend != profile.backend():
            raise preparation.PreparationRefused("actor backend differs from reserved profile")
        try:
            current_binary = observation_binding._artifact_identity(Path(profile.binary))["sha256"]
        except (OSError, observation_binding.ObservationBindingError) as exc:
            raise preparation.PreparationRefused("actor executable is unreadable") from exc
        if current_binary != profile.binary_sha256:
            raise preparation.PreparationRefused("actor executable changed after reservation")
        artifact_contract_digest = preparation._digest({
            "request_digest": reservation.request_digest,
            "actor_profile_digest": reservation.actor_profile_digest,
            "target_profile_digest": reservation.target_profile_digest,
        })
        request = worker_lifecycle.StageRequest(
            request_id=reservation.reservation_id,
            plan_digest=reservation.stage_plan_digest,
            lineage_id=reservation.transition_id,
            stage_id=f"actor-{profile.role}-{reservation.reservation_id}",
            stage="setup", argv=tuple(backend.argv(prompt, self.config.cwd)),
            env=dict(self.config.env), cwd=self.config.cwd,
            artifact_contract_digest=artifact_contract_digest,
            max_stage_seconds=self.config.max_stage_seconds,
            teardown_seconds=self.config.teardown_seconds,
            control_revision=reservation.control_revision)
        try:
            returned = self.controller.run_worker_stage(request)
            terminal = self.controller.worker_terminal_for_request(
                request_id=request.request_id, plan_digest=request.plan_digest,
                lineage_id=request.lineage_id, stage_id=request.stage_id)
        except Exception as exc:
            raise ActorInvocationUncertain(
                "controller invocation has no authoritative negative-admission proof") from exc
        if (terminal is None or returned != terminal
                or terminal.request_id != reservation.reservation_id
                or terminal.plan_digest != reservation.stage_plan_digest
                or terminal.stage_id != request.stage_id):
            raise ActorInvocationUncertain(
                "controller invocation lacks one exact accepted terminal")
        try:
            held = self.controller.actor_held_claim_receipt(terminal)
        except Exception as exc:
            raise ActorInvocationUncertain(
                "controller cannot prove exact provider-held accounting") from exc
        if (not isinstance(held, scheduling.HeldClaimReceipt)
                or held.ownership_generation != terminal.worker_generation
                or held.allocation_generation != terminal.grant_generation):
            raise ActorInvocationUncertain(
                "controller lacks exact provider-authored held accounting")
        charged = held.ended_at - held.started_at
        if not math.isfinite(charged) or charged <= 0:
            raise ActorInvocationUncertain(
                "provider-authored held accounting duration is invalid")
        self._held_costs[reservation.reservation_id] = held
        if terminal.return_code != 0:
            return preparation.StageOutcome(
                reservation.reservation_id, "failed", "", "process_exit",
                charged, True, True).to_dict()
        if not terminal.accepted:
            return preparation.StageOutcome(
                reservation.reservation_id, "failed", "", "worker_terminal_refused",
                charged, False, False).to_dict()
        try:
            raw = self.controller.read_worker_stdout(
                request_id=terminal.request_id, plan_digest=terminal.plan_digest,
                lineage_id=terminal.lineage_id, stage_id=terminal.stage_id,
                worker_id=terminal.worker_id,
                worker_generation=terminal.worker_generation,
                result_digest=terminal.result_digest,
                max_bytes=self.config.max_retained_output_bytes)
            if not isinstance(raw, bytes):
                return preparation.StageOutcome(
                    reservation.reservation_id, "output_limit", "", "output_limit",
                    charged, True, True).to_dict()
            stdout = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError, worker_lifecycle.LifecycleRefused):
            return preparation.StageOutcome(
                reservation.reservation_id, "output_limit", "", "output_limit",
                charged, True, True).to_dict()
        return preparation.StageOutcome(
            reservation.reservation_id, "completed", stdout, None, charged,
            True, True).to_dict()

    def finish(self, reservation: preparation.StageReservation,
               outcome: preparation.StageOutcome, disposition: str) -> None:
        self.persistence.finish_actor_preparation(
            reservation=reservation.to_dict(), outcome=outcome.to_dict(),
            disposition=disposition,
            provider_cost_receipt=self._held_costs.get(reservation.reservation_id))


__all__ = ["ActorInvocationUncertain", "ActorLifecycleAdapter", "ActorLifecycleConfig",
           "ActorPersistence", "TargetProfileOwner", "TargetProfileReceipt"]
