"""Selected source/build actor preparation over an owned-worker authority seam.

This consumer is not a process supervisor. Its injected capability is implemented by
the controller/WorkerLifecycle owner: that boundary proves selection, durably records
intent and budget debit before launch, contains every descendant, and returns bounded
output. No public record here grants authority.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

from . import actors, campaign

PROFILE_SCHEMA = "epyc.autokernel.actor_profile.v1"
RESERVATION_SCHEMA = "epyc.autokernel.actor_stage_reservation.v1"
OUTCOME_SCHEMA = "epyc.autokernel.actor_stage_outcome.v1"
AVAILABILITY_SCHEMA = "epyc.autokernel.actor_availability.v1"
RESULT_SCHEMA = "epyc.autokernel.actor_preparation_result.v1"
REQUEST_SCHEMA = "epyc.autokernel.actor_preparation.v1"


class PreparationRefused(ValueError):
    """Preparation lacks an exact binding or accepted lifecycle authority."""


class PreparationSettlementUncertain(RuntimeError):
    """Owner did not acknowledge terminal settlement; never re-invoke blindly."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PreparationRefused(f"{label} must be a non-empty string")
    return value


def _sha256(value: Any, label: str) -> str:
    digest = _text(value, label)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise PreparationRefused(f"{label} must be lowercase SHA-256")
    return digest


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PreparationRefused(f"{label} must be an object")
    return dict(value)


@dataclass(frozen=True)
class ActorProfile:
    """Pinned provider/model/effort/executable configuration selected by campaign."""

    profile_id: str
    role: str
    provider: str
    model: str
    effort: str
    backend_kind: str
    binary: str
    binary_sha256: str
    schema: str = PROFILE_SCHEMA

    @classmethod
    def from_dict(cls, value: Any) -> "ActorProfile":
        row = _mapping(value, "actor profile")
        fields = {"schema", "profile_id", "role", "provider", "model", "effort",
                  "backend_kind", "binary", "binary_sha256"}
        if set(row) != fields or row.pop("schema") != PROFILE_SCHEMA:
            raise PreparationRefused("actor profile fields/schema differ")
        binary = Path(_text(row["binary"], "actor profile binary"))
        if not binary.is_absolute():
            raise PreparationRefused("actor profile binary must be an absolute pinned path")
        _sha256(row["binary_sha256"], "actor profile binary_sha256")
        result = cls(**{key: _text(row[key], f"actor profile {key}")
                        for key in fields - {"schema"}})
        result.backend().argv("probe", Path("/tmp"))
        return result

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "profile_id": self.profile_id, "role": self.role,
                "provider": self.provider, "model": self.model, "effort": self.effort,
                "backend_kind": self.backend_kind, "binary": self.binary,
                "binary_sha256": self.binary_sha256}

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())

    def backend(self) -> actors.Backend:
        return actors.Backend(self.backend_kind, self.model, self.effort, self.binary)


@dataclass(frozen=True)
class ActorBudgets:
    """Independent limits submitted to the durable owner; never charged locally."""

    actor_calls_per_target: int
    patch_repairs_per_target: int
    provider_seconds_per_target: int
    resource_failures_per_target: int
    contamination_events_per_target: int
    actor_calls_per_campaign: int

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise PreparationRefused(f"{name} must be a positive integer")

    def to_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class StageReservation:
    """Receipt returned by trusted lifecycle authority after durable INTENT."""

    reservation_id: str
    request_digest: str
    stage_plan_digest: str
    transition_id: str
    target_profile_digest: str
    target_profile_receipt_digest: str
    actor_profile_digest: str
    deadline: float
    clock_domain: str
    control_revision: int
    schema: str = RESERVATION_SCHEMA

    @classmethod
    def from_dict(cls, value: Any) -> "StageReservation":
        row = _mapping(value, "actor reservation")
        fields = {"schema", "reservation_id", "request_digest", "stage_plan_digest",
                  "transition_id", "target_profile_digest", "actor_profile_digest", "deadline",
                  "target_profile_receipt_digest", "clock_domain", "control_revision"}
        if set(row) != fields or row.pop("schema") != RESERVATION_SCHEMA:
            raise PreparationRefused("actor reservation fields/schema differ")
        deadline = row.pop("deadline")
        if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) \
                or not math.isfinite(deadline):
            raise PreparationRefused("actor reservation deadline is invalid")
        control_revision = row.pop("control_revision")
        if (isinstance(control_revision, bool) or not isinstance(control_revision, int)
                or control_revision < 0):
            raise PreparationRefused("actor reservation control_revision is invalid")
        parsed = {key: _text(item, f"actor reservation {key}")
                  for key, item in row.items()}
        for name in ("request_digest", "target_profile_digest",
                     "target_profile_receipt_digest", "actor_profile_digest"):
            parsed[name] = _sha256(parsed[name], f"actor reservation {name}")
        return cls(**parsed, deadline=float(deadline), control_revision=control_revision)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "reservation_id": self.reservation_id,
                "request_digest": self.request_digest,
                "stage_plan_digest": self.stage_plan_digest,
                "transition_id": self.transition_id,
                "target_profile_digest": self.target_profile_digest,
                "target_profile_receipt_digest": self.target_profile_receipt_digest,
                "actor_profile_digest": self.actor_profile_digest, "deadline": self.deadline,
                "clock_domain": self.clock_domain, "control_revision": self.control_revision}


@dataclass(frozen=True)
class StageOutcome:
    """Bounded output and lifecycle facts from the owned-child implementation."""

    reservation_id: str
    status: str
    stdout: str
    failure_class: str | None
    charged_seconds: float
    resource_enforced: bool
    descendants_clean: bool
    schema: str = OUTCOME_SCHEMA

    @classmethod
    def from_dict(cls, value: Any) -> "StageOutcome":
        row = _mapping(value, "actor outcome")
        fields = {"schema", "reservation_id", "status", "stdout", "failure_class",
                  "charged_seconds", "resource_enforced", "descendants_clean"}
        if set(row) != fields or row.pop("schema") != OUTCOME_SCHEMA:
            raise PreparationRefused("actor outcome fields/schema differ")
        if row["status"] not in {"completed", "failed", "deadline", "output_limit"}:
            raise PreparationRefused("actor outcome status differs")
        if type(row["resource_enforced"]) is not bool or type(row["descendants_clean"]) is not bool:
            raise PreparationRefused("actor outcome enforcement facts must be boolean")
        seconds = row["charged_seconds"]
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) \
                or not math.isfinite(seconds) or seconds < 0:
            raise PreparationRefused("actor outcome charged_seconds is invalid")
        failure = row["failure_class"]
        if failure is not None:
            failure = _text(failure, "actor outcome failure_class")
        status = row["status"]
        if status == "completed" and failure is not None:
            raise PreparationRefused("completed actor outcome cannot carry failure_class")
        if status != "completed" and failure is None:
            raise PreparationRefused("failed actor outcome requires failure_class")
        if status in {"deadline", "output_limit"} and failure != status:
            raise PreparationRefused("actor outcome status/failure_class differ")
        if not isinstance(row["stdout"], str):
            raise PreparationRefused("actor outcome stdout must be a string")
        stdout = row["stdout"]
        return cls(_text(row["reservation_id"], "actor outcome reservation_id"),
                   status, stdout, failure, float(seconds),
                   row["resource_enforced"], row["descendants_clean"])

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "reservation_id": self.reservation_id,
                "status": self.status, "stdout": self.stdout,
                "failure_class": self.failure_class,
                "charged_seconds": self.charged_seconds,
                "resource_enforced": self.resource_enforced,
                "descendants_clean": self.descendants_clean}


@dataclass(frozen=True)
class ActorAvailability:
    """Durable denial/cooldown view returned by the parent-owned Journal fold."""

    actor_profile_digest: str
    failure_class: str
    consecutive_failures: int
    last_success: float | None
    retry_after: float | None
    reset_at: float | None
    clock_domain: str
    next_eligible_at: float
    schema: str = AVAILABILITY_SCHEMA

    @classmethod
    def from_dict(cls, value: Any) -> "ActorAvailability":
        row = _mapping(value, "actor availability")
        fields = {"schema", "actor_profile_digest", "failure_class", "consecutive_failures",
                  "last_success", "retry_after", "reset_at", "clock_domain",
                  "next_eligible_at"}
        if set(row) != fields or row.pop("schema") != AVAILABILITY_SCHEMA:
            raise PreparationRefused("actor availability fields/schema differ")
        streak = row.pop("consecutive_failures")
        if isinstance(streak, bool) or not isinstance(streak, int) or streak < 1:
            raise PreparationRefused("actor availability streak is invalid")
        times: dict[str, float | None] = {}
        for name in ("last_success", "retry_after", "reset_at", "next_eligible_at"):
            item = row.pop(name)
            if item is not None and (isinstance(item, bool) or not isinstance(item, (int, float))
                                     or not math.isfinite(item)):
                raise PreparationRefused(f"actor availability {name} is invalid")
            times[name] = None if item is None else float(item)
        if (times["next_eligible_at"] is None or times["retry_after"] is None
                or times["next_eligible_at"] != times["retry_after"]):
            raise PreparationRefused("actor availability retry/next eligibility differ")
        return cls(**{key: _text(item, f"actor availability {key}")
                      for key, item in row.items()}, consecutive_failures=streak, **times)  # type: ignore[arg-type]


class ActorStageCapability(Protocol):
    """Controller-owned adapter to the accepted WorkerLifecycle stage path.

    ``reserve`` validates the real selected driver catalog/transition, fresh
    target-profile prerequisite, and executable bytes against ``binary_sha256``, then
    appends selection/budget INTENT before return.
    A typed availability record is a denial/cooldown. ``invoke`` owns/bounds the
    descendant container.
    ``finish`` records actual cost, availability streak/retry/reset and disposition.
    Exact append-uncertainty retries are the owner's idempotent responsibility.
    """

    def reserve(self, *, request: Mapping[str, Any], stage_plan_digest: str,
                actor_profile: Mapping[str, Any], budgets: Mapping[str, int],
                now: float, clock_domain: str) -> Mapping[str, Any] | None: ...

    def invoke(self, reservation: StageReservation, backend: actors.Backend,
               prompt: str) -> Mapping[str, Any]: ...

    def finish(self, reservation: StageReservation, outcome: StageOutcome,
               disposition: str) -> None: ...


@dataclass(frozen=True)
class PreparationResult:
    status: str
    request_digest: str
    target_revision_digest: str
    actor_profile_digest: str | None
    proposed_output: Mapping[str, Any] | None
    failure_class: str | None = None
    retry_after: float | None = None
    schema: str = RESULT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RESULT_SCHEMA or self.status not in {"proposed", "cooldown", "refused"}:
            raise PreparationRefused("preparation result schema/status differs")
        object.__setattr__(self, "proposed_output", None if self.proposed_output is None
                           else MappingProxyType(dict(self.proposed_output)))


def _request_dict(request: Any) -> dict[str, Any]:
    if not hasattr(request, "to_dict"):
        raise PreparationRefused("request must be an ActorPreparationRequest")
    row = _mapping(request.to_dict(), "actor request")
    fields = {"schema", "actor_kind", "actor_identity", "prompt", "mandatory_conflicts",
              "proposal", "cache_key"}
    if set(row) != fields or row["schema"] != REQUEST_SCHEMA:
        raise PreparationRefused("actor request fields/schema differ")
    if row["actor_kind"] not in {"source", "build_recipe"}:
        raise PreparationRefused("runtime work never enters actor preparation")
    expected = _digest({key: row[key] for key in ("actor_kind", "actor_identity", "prompt",
                                                   "mandatory_conflicts", "proposal")})
    if row["cache_key"] != expected:
        raise PreparationRefused("actor request cache key differs from dependencies")
    return row


_FORBIDDEN = {"compiled", "compile_succeeded", "verified_dispatch", "scientific_warrant",
              "experiment_plan_digest", "candidate_integrated", "allocation", "grant_id"}


def _validate_output(kind: str, raw: str) -> dict[str, Any]:
    body = actors._extract_json(raw)
    forbidden = sorted(_FORBIDDEN.intersection(body))
    if forbidden:
        raise PreparationRefused(f"actor asserted parent-owned fields {forbidden}")
    required = ({"mechanism", "target_surface", "target_symbol", "implementation_plan"}
                if kind == "source" else
                {"build_system", "configured_options", "artifact_expectations"})
    if set(body) != required:
        raise PreparationRefused(f"{kind} actor output fields differ")
    for key, item in body.items():
        if key == "configured_options":
            if (isinstance(item, (str, bytes)) or not isinstance(item, Sequence)
                    or not all(isinstance(value, str) and value for value in item)):
                raise PreparationRefused("configured_options must be a string array")
        else:
            _text(item, f"actor output {key}")
    return body


def _actor_prompt(row: Mapping[str, Any]) -> str:
    shape = ('{"mechanism":"...","target_surface":"...","target_symbol":"...",'
             '"implementation_plan":"..."}' if row["actor_kind"] == "source" else
             '{"build_system":"...","configured_options":["..."],'
             '"artifact_expectations":"..."}')
    return (f"{row['prompt']}\n\nMandatory conflicts:\n{_canonical(row['mandatory_conflicts'])}\n\n"
            "Return advice only; do not claim build, dispatch, allocation, integration, or "
            f"scientific success. Reply with exactly one JSON object:\n{shape}")


def _review_prompt(row: Mapping[str, Any], output: Mapping[str, Any]) -> str:
    return ("Review this proposed preparation against the frozen request and mandatory conflicts. "
            'Reply exactly as JSON: {"accepted":true|false,"reason":"required on rejection"}\n'
            f"request={_canonical(row)}\nproposal={_canonical(output)}")


def _validate_review(raw: str) -> tuple[bool, str]:
    body = actors._extract_json(raw)
    if set(body) != {"accepted", "reason"} or type(body["accepted"]) is not bool \
            or not isinstance(body["reason"], str):
        raise PreparationRefused("critic output fields differ")
    if not body["accepted"] and not body["reason"].strip():
        raise PreparationRefused("critic rejection requires a reason")
    return body["accepted"], body["reason"]


class ActorPreparationConsumer:
    """Backend-blind consumer; authority and durable accounting remain upstream."""

    def __init__(self, *, resolved_campaign: campaign.ResolvedCampaign,
                 profiles: Mapping[str, ActorProfile | Mapping[str, Any]],
                 budgets: ActorBudgets, capability: ActorStageCapability | None,
                 clock: Any, clock_domain: str, max_output_bytes: int) -> None:
        self.campaign = campaign.ResolvedCampaign.from_dict(resolved_campaign.to_dict())
        self.profiles = MappingProxyType({
            key: item if isinstance(item, ActorProfile) else ActorProfile.from_dict(item)
            for key, item in profiles.items()})
        self.budgets = budgets
        self.capability = capability
        if not callable(clock):
            raise PreparationRefused("clock must be callable")
        self.clock = clock
        self.clock_domain = _text(clock_domain, "actor clock domain")
        if (isinstance(max_output_bytes, bool) or not isinstance(max_output_bytes, int)
                or max_output_bytes <= 0):
            raise PreparationRefused("max_output_bytes must be a positive integer")
        self.max_output_bytes = max_output_bytes

    def _chain(self, role: str) -> tuple[ActorProfile, ...]:
        configured = dict(self.campaign.actors)
        fallbacks = dict(self.campaign.fallbacks)
        if role not in configured:
            raise PreparationRefused(f"campaign lacks configured {role} actor")
        identities = (configured[role], *fallbacks[role])
        try:
            result = tuple(self.profiles[item] for item in identities)
        except KeyError as exc:
            raise PreparationRefused("configured actor profile is not loaded") from exc
        if any(profile.profile_id != identity or profile.role != role
               for identity, profile in zip(identities, result, strict=True)):
            raise PreparationRefused("loaded actor profile differs from campaign role")
        return result

    def _now(self) -> float:
        now = self.clock()
        if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now):
            raise PreparationRefused("actor clock must be finite")
        return float(now)

    def _finish(self, reservation: StageReservation, outcome: StageOutcome,
                disposition: str) -> None:
        if self.capability is None:  # guarded before every reservation
            raise PreparationRefused("owned actor-stage lifecycle capability is unavailable")
        try:
            self.capability.finish(reservation, outcome, disposition)
        except Exception as exc:
            raise PreparationSettlementUncertain(
                f"actor reservation {reservation.reservation_id} settlement is uncertain; "
                "exact owner retry is required and actor reinvocation is forbidden") from exc

    def _invoke(self, row: Mapping[str, Any], profile: ActorProfile, prompt: str,
                stage_plan_digest: str, now: float
                ) -> tuple[StageReservation, StageOutcome] | ActorAvailability:
        if self.capability is None:
            raise PreparationRefused("owned actor-stage lifecycle capability is unavailable")
        raw = self.capability.reserve(
            request=row, stage_plan_digest=stage_plan_digest,
            actor_profile=profile.to_dict(), budgets=self.budgets.to_dict(), now=now,
            clock_domain=self.clock_domain)
        if raw is None:
            raise PreparationRefused("actor authority returned an untyped denial")
        if raw.get("schema") == AVAILABILITY_SCHEMA:
            availability = ActorAvailability.from_dict(raw)
            if availability.actor_profile_digest != profile.digest:
                raise PreparationRefused("actor availability differs from requested profile")
            if (availability.clock_domain != self.clock_domain
                    or availability.next_eligible_at <= now):
                raise PreparationRefused("actor availability is stale or uses another clock")
            return availability
        reservation = StageReservation.from_dict(raw)
        if (reservation.request_digest != _digest(row)
                or reservation.stage_plan_digest != stage_plan_digest
                or reservation.actor_profile_digest != profile.digest
                or reservation.clock_domain != self.clock_domain
                or reservation.deadline <= now):
            raise PreparationRefused("reservation differs from selected request/profile")
        try:
            supplied_outcome = self.capability.invoke(reservation, profile.backend(), prompt)
            outcome = StageOutcome.from_dict(supplied_outcome)
        except Exception as exc:
            failure_class = ("malformed_outcome" if isinstance(exc, PreparationRefused)
                             else "invocation_exception")
            terminal = StageOutcome(
                reservation.reservation_id, "failed", "", failure_class, 0.0, False, False)
            self._finish(reservation, terminal, failure_class)
            raise PreparationRefused(
                f"actor lifecycle {failure_class} settled by owner") from exc
        if outcome.reservation_id != reservation.reservation_id:
            raise PreparationRefused("actor outcome differs from reservation")
        if not outcome.resource_enforced or not outcome.descendants_clean:
            self._finish(reservation, outcome, "containment_refused")
            raise PreparationRefused("lifecycle did not prove resource/descendant containment")
        if len(outcome.stdout.encode("utf-8")) > self.max_output_bytes:
            bounded = StageOutcome(
                outcome.reservation_id, "output_limit", "", "output_limit",
                outcome.charged_seconds, outcome.resource_enforced, outcome.descendants_clean)
            return reservation, bounded
        return reservation, outcome

    def prepare(self, request: Any, *, stage_plan_digest: str) -> PreparationResult:
        row = _request_dict(request)
        proposal = _mapping(row["proposal"], "actor proposal")
        target = _text(proposal.get("target_revision_digest"), "target revision digest")
        expected_identity = _mapping(row["actor_identity"], "actor identity")
        planner_output = None
        planner_profile = None
        last_availability = None
        for index, profile in enumerate(self._chain("planner")):
            if index == 0 and expected_identity != profile.to_dict():
                raise PreparationRefused("request actor identity differs from configured primary")
            invoked = self._invoke(
                row, profile, _actor_prompt(row), stage_plan_digest, self._now())
            if isinstance(invoked, ActorAvailability):
                last_availability = invoked
                continue
            reservation, outcome = invoked
            if outcome.status != "completed":
                disposition = (outcome.status if outcome.status in {"deadline", "output_limit"}
                               else "actor_failed")
                self._finish(reservation, outcome, disposition)
                continue
            try:
                planner_output = _validate_output(row["actor_kind"], outcome.stdout)
            except PreparationRefused:
                self._finish(reservation, outcome, "invalid_output")
                continue
            self._finish(reservation, outcome, "proposal_ready")
            planner_profile = profile
            break
        if planner_output is None or planner_profile is None:
            return PreparationResult("cooldown", _digest(row), target, None, None,
                                     (last_availability.failure_class if last_availability
                                      else "planner_unavailable"),
                                     last_availability.retry_after if last_availability else None)
        last_availability = None
        for critic in self._chain("critic"):
            invoked = self._invoke(row, critic, _review_prompt(row, planner_output),
                                   stage_plan_digest, self._now())
            if isinstance(invoked, ActorAvailability):
                last_availability = invoked
                continue
            reservation, outcome = invoked
            if outcome.status != "completed":
                disposition = (outcome.status if outcome.status in {"deadline", "output_limit"}
                               else "critic_failed")
                self._finish(reservation, outcome, disposition)
                continue
            try:
                accepted, _ = _validate_review(outcome.stdout)
            except PreparationRefused:
                self._finish(reservation, outcome, "invalid_review")
                continue
            disposition = "review_accepted" if accepted else "review_rejected"
            self._finish(reservation, outcome, disposition)
            combined = _digest({"planner": planner_profile.digest, "critic": critic.digest})
            if not accepted:
                return PreparationResult("refused", _digest(row), target, combined, None,
                                         "review_rejected", None)
            return PreparationResult("proposed", _digest(row), target, combined, planner_output)
        return PreparationResult("cooldown", _digest(row), target, planner_profile.digest,
                                 None, (last_availability.failure_class if last_availability
                                        else "critic_unavailable"),
                                 last_availability.retry_after if last_availability else None)


__all__ = ["ActorAvailability", "ActorBudgets", "ActorPreparationConsumer", "ActorProfile",
           "ActorStageCapability", "PreparationRefused", "PreparationResult",
           "PreparationSettlementUncertain",
           "StageOutcome", "StageReservation"]
