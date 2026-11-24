"""Closed native state machine for selected actor-preparation accounting."""
from __future__ import annotations

from .field_validation import require_sha256 as _require_sha256

from .field_validation import canonical_json as _canonical_json

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

INTENT_SCHEMA = "epyc.autokernel.actor_preparation_intent.v1"
FINISH_SCHEMA = "epyc.autokernel.actor_preparation_finish.v1"
PROFILE_SCHEMA = "epyc.autokernel.target_profile_verified.v1"
# AKU-06i additive successor semantics.  v1 stays byte-exact and keeps its refusal of a
# second distinct PROFILE_VERIFIED per target; v2 names its predecessor and settlement.
PROFILE_REQUEST_V2_SCHEMA = "epyc.autokernel.target_profile_request.v2"
PROFILE_V2_SCHEMA = "epyc.autokernel.target_profile_verified.v2"
BUDGET_KEYS = frozenset({
    "actor_calls_per_target", "patch_repairs_per_target",
    "provider_seconds_per_target", "resource_failures_per_target",
    "contamination_events_per_target", "actor_calls_per_campaign",
})
STATUSES = frozenset({"completed", "failed", "deadline", "output_limit"})


class ActorStateRefused(ValueError):
    pass


def _canonical(value: Any) -> bytes:
    return _canonical_json(value, ActorStateRefused, 'actor state is not canonical finite JSON')


def digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ActorStateRefused(f"{label} must be nonempty text")
    return value


def _sha(value: Any, label: str) -> str:
    return _require_sha256(_text(value, label), label, ActorStateRefused)


def _finite(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < minimum):
        raise ActorStateRefused(f"{label} must be finite and >= {minimum}")
    return float(value)


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ActorStateRefused(f"{label} must be a positive integer")
    return value


def _budget_map(value: Any, label: str, *, integral: bool) -> dict[str, float | int]:
    if not isinstance(value, Mapping) or set(value) != BUDGET_KEYS:
        raise ActorStateRefused(f"{label} must carry exactly all six dimensions")
    if integral:
        return {key: _positive_int(value[key], f"{label}.{key}") for key in BUDGET_KEYS}
    return {key: _finite(value[key], f"{label}.{key}") for key in BUDGET_KEYS}


_COMMON = {
    "schema", "event", "reservation_id", "campaign_id", "config_generation",
    "config_digest", "supervisor_id", "supervisor_incarnation", "control_revision",
    "catalog_id", "transition_id", "request_digest", "stage_plan_digest",
    "target_revision_digest", "target_profile_digest", "target_profile_receipt_digest",
    "actor_profile_digest", "backend_key", "clock_domain", "deadline", "occurred_at",
}
_INTENT = _COMMON | {"budgets", "debits"}
_FINISH = _COMMON | {
    "outcome_digest", "status", "failure_class", "charged_seconds",
    "resource_enforced", "descendants_clean", "disposition", "charges",
    "consecutive_failures", "last_success", "retry_after", "reset_at",
    "next_eligible_at",
}
_PROFILE = {
    "schema", "event", "campaign_id", "config_generation", "config_digest",
    "supervisor_id", "supervisor_incarnation", "control_revision", "catalog_id",
    "transition_id", "stage_plan_digest", "profile_request_digest",
    "target_revision_digest", "target_profile_digest", "profile_request",
    "profile_content", "artifact_identity", "loaded_identity", "measurement_carrier",
    "clock_domain", "verifier_ref", "verified_at", "valid_until", "occurred_at",
}
_SUCCESSOR_JOINS = {"generation", "predecessor_profile_digest",
                    "predecessor_request_digest", "predecessor_settlement_digest"}
_PROFILE_V2 = _PROFILE | _SUCCESSOR_JOINS
_PROFILE_REQUEST_V2 = {
    "schema", "event", "campaign_id", "config_generation", "config_digest",
    "supervisor_id", "supervisor_incarnation", "control_revision",
    "target_revision_digest", "profile_request", "profile_request_digest",
    "template_digest", "attempt_cap", "clock_domain", "requested_at", "occurred_at",
} | _SUCCESSOR_JOINS


def _validate_successor_joins(row: Mapping[str, Any]) -> None:
    generation = row["generation"]
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 2:
        raise ActorStateRefused("successor generation must be an integer >= 2")
    for name in ("predecessor_profile_digest", "predecessor_request_digest",
                 "predecessor_settlement_digest"):
        _sha(row[name], name)


def _validate_request_v2(row: dict[str, Any]) -> dict[str, Any]:
    if set(row) != _PROFILE_REQUEST_V2 or row.get("schema") != PROFILE_REQUEST_V2_SCHEMA:
        raise ActorStateRefused("profile request fields/schema differ")
    for name in ("campaign_id", "supervisor_id", "clock_domain", "occurred_at"):
        _text(row[name], name)
    for name in ("config_generation", "supervisor_incarnation"):
        _positive_int(row[name], name)
    if (isinstance(row["control_revision"], bool)
            or not isinstance(row["control_revision"], int)
            or row["control_revision"] < 0):
        raise ActorStateRefused("control_revision must be a nonnegative integer")
    for name in ("config_digest", "target_revision_digest", "profile_request_digest",
                 "template_digest"):
        _sha(row[name], name)
    _finite(row["requested_at"], "requested_at")
    _validate_successor_joins(row)
    cap = _positive_int(row["attempt_cap"], "attempt_cap")
    if row["generation"] > cap:
        raise ActorStateRefused("request generation exceeds the attempt cap")
    request = row["profile_request"]
    if not isinstance(request, Mapping) or not request:
        raise ActorStateRefused("profile_request must be a nonempty object")
    if digest(request) != row["profile_request_digest"]:
        raise ActorStateRefused("profile request digest differs")
    if row["profile_request_digest"] == row["predecessor_request_digest"]:
        raise ActorStateRefused("successor request must differ from its predecessor")
    return row


def validate_event(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ActorStateRefused("actor event must be an object")
    row = dict(value)
    event = row.get("event")
    if event == "PROFILE_REQUEST":
        return _validate_request_v2(row)
    if event == "PROFILE_VERIFIED" and row.get("schema") == PROFILE_V2_SCHEMA:
        if set(row) != _PROFILE_V2:
            raise ActorStateRefused("actor event fields/schema differ")
        _validate_successor_joins(row)
        base = {k: v for k, v in row.items() if k not in _SUCCESSOR_JOINS}
        base["schema"] = PROFILE_SCHEMA
        validate_event(base)
        return row
    expected = (_INTENT if event == "INTENT" else _FINISH if event == "FINISH"
                else _PROFILE if event == "PROFILE_VERIFIED" else set())
    schema = (INTENT_SCHEMA if event == "INTENT" else FINISH_SCHEMA
              if event == "FINISH" else PROFILE_SCHEMA)
    if not expected or set(row) != expected or row.get("schema") != schema:
        raise ActorStateRefused("actor event fields/schema differ")
    text_names = ["campaign_id", "supervisor_id", "clock_domain", "occurred_at"]
    if event != "PROFILE_VERIFIED":
        text_names += ["reservation_id", "backend_key"]
    else:
        text_names += ["verifier_ref"]
    for name in text_names:
        _text(row[name], name)
    for name in ("config_generation", "supervisor_incarnation"):
        _positive_int(row[name], name)
    if (isinstance(row["control_revision"], bool)
            or not isinstance(row["control_revision"], int)
            or row["control_revision"] < 0):
        raise ActorStateRefused("control_revision must be a nonnegative integer")
    sha_names = ["config_digest", "catalog_id", "transition_id", "stage_plan_digest",
                 "target_revision_digest", "target_profile_digest"]
    if event == "PROFILE_VERIFIED":
        sha_names += ["profile_request_digest"]
    else:
        sha_names += ["request_digest", "target_profile_receipt_digest",
                      "actor_profile_digest"]
    for name in sha_names:
        _sha(row[name], name)
    if event == "PROFILE_VERIFIED":
        verified = _finite(row["verified_at"], "verified_at")
        valid = _finite(row["valid_until"], "valid_until", minimum=1e-12)
        if valid <= verified:
            raise ActorStateRefused("profile validity interval is empty")
        for name in ("profile_request", "profile_content", "artifact_identity",
                     "loaded_identity", "measurement_carrier"):
            if not isinstance(row[name], Mapping) or not row[name]:
                raise ActorStateRefused(f"{name} must be a nonempty object")
        if digest(row["profile_request"]) != row["profile_request_digest"]:
            raise ActorStateRefused("profile request digest differs")
        if digest({"profile_content": row["profile_content"],
                   "loaded_identity": row["loaded_identity"],
                   "artifact_identity": row["artifact_identity"]}) \
                != row["target_profile_digest"]:
            raise ActorStateRefused("target profile digest differs from immutable result")
        return row
    _finite(row["deadline"], "deadline", minimum=1e-12)
    if event == "INTENT":
        _budget_map(row["budgets"], "budgets", integral=True)
        debits = _budget_map(row["debits"], "debits", integral=False)
        if debits["actor_calls_per_target"] != 1 \
                or debits["actor_calls_per_campaign"] != 1:
            raise ActorStateRefused("intent must debit exactly one target/campaign actor call")
        return row
    _sha(row["outcome_digest"], "outcome_digest")
    if row["status"] not in STATUSES:
        raise ActorStateRefused("finish status is unsupported")
    failure = row["failure_class"]
    if ((row["status"] == "completed") != (failure is None)):
        raise ActorStateRefused("finish status/failure_class disagree")
    if failure is not None:
        _text(failure, "failure_class")
    _finite(row["charged_seconds"], "charged_seconds")
    if type(row["resource_enforced"]) is not bool or type(row["descendants_clean"]) is not bool:
        raise ActorStateRefused("finish containment facts must be boolean")
    _text(row["disposition"], "disposition")
    charges = _budget_map(row["charges"], "charges", integral=False)
    if charges["provider_seconds_per_target"] != row["charged_seconds"]:
        raise ActorStateRefused("provider charge differs from provider-authored duration")
    streak = row["consecutive_failures"]
    if isinstance(streak, bool) or not isinstance(streak, int) or streak < 0:
        raise ActorStateRefused("consecutive_failures must be nonnegative")
    for name in ("last_success", "retry_after", "reset_at", "next_eligible_at"):
        if row[name] is not None:
            _finite(row[name], name)
    if row["retry_after"] != row["next_eligible_at"]:
        raise ActorStateRefused("retry_after and next_eligible_at differ")
    return row


@dataclass(frozen=True)
class ActorPreparationProjection:
    pending: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    finished: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    reserved: Mapping[str, float] = field(
        default_factory=lambda: {key: 0.0 for key in BUDGET_KEYS})
    spent: Mapping[str, float] = field(
        default_factory=lambda: {key: 0.0 for key in BUDGET_KEYS})
    availability: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    profiles: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    # AKU-06i: open (selected, unconsumed) successor request per target, and the
    # generation of the current profile per target (1 for a legacy v1 profile).
    profile_requests: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    profile_generations: Mapping[str, int] = field(default_factory=dict)


def project_events(events: Sequence[Mapping[str, Any]]) -> ActorPreparationProjection:
    pending: dict[str, dict[str, Any]] = {}
    finished: dict[str, dict[str, Any]] = {}
    reserved = {key: 0.0 for key in BUDGET_KEYS}
    spent = {key: 0.0 for key in BUDGET_KEYS}
    availability: dict[str, dict[str, Any]] = {}
    profiles: dict[str, dict[str, Any]] = {}
    requests: dict[str, dict[str, Any]] = {}
    generations: dict[str, int] = {}
    seen_requests: set[str] = set()
    for value in events:
        row = validate_event(value)
        if row["event"] == "PROFILE_REQUEST":
            target = row["target_revision_digest"]
            prior = profiles.get(target)
            _join_successor(row, prior, generations.get(target, 1))
            if target in requests:
                raise ActorStateRefused("target already has an open successor request")
            if row["profile_request_digest"] in seen_requests:
                raise ActorStateRefused("successor request digest was already used")
            seen_requests.add(row["profile_request_digest"])
            requests[target] = row
            continue
        if row["event"] == "PROFILE_VERIFIED" and row["schema"] == PROFILE_V2_SCHEMA:
            target = row["target_revision_digest"]
            prior = profiles.get(target)
            request = requests.get(target)
            _join_successor(row, prior, generations.get(target, 1))
            if request is None:
                raise ActorStateRefused("successor profile lacks its selected request")
            for name in ("profile_request_digest", "profile_request", "generation",
                         "predecessor_profile_digest", "predecessor_request_digest",
                         "predecessor_settlement_digest"):
                if row[name] != request[name]:
                    raise ActorStateRefused(f"successor profile {name} differs from request")
            profiles[target] = row
            generations[target] = row["generation"]
            del requests[target]
            continue
        if row["event"] == "PROFILE_VERIFIED":
            target = row["target_revision_digest"]
            prior = profiles.get(target)
            if prior is not None and prior != row:
                raise ActorStateRefused("target profile is replaced without a new owner epoch")
            profiles[target] = row
            generations.setdefault(target, 1)
            continue
        reservation = row["reservation_id"]
        if row["event"] == "INTENT":
            if reservation in pending or reservation in finished:
                raise ActorStateRefused("reservation intent is duplicated or reused")
            pending[reservation] = row
            for key, amount in row["debits"].items():
                reserved[key] += float(amount)
            continue
        intent = pending.get(reservation)
        if intent is None:
            if reservation in finished and finished[reservation] == row:
                raise ActorStateRefused("identical finish must not be appended twice")
            raise ActorStateRefused("finish lacks one exact unresolved intent")
        for name in _COMMON - {"schema", "event", "occurred_at"}:
            if row[name] != intent[name]:
                raise ActorStateRefused(f"finish {name} differs from intent")
        del pending[reservation]
        finished[reservation] = row
        for key, amount in intent["debits"].items():
            reserved[key] -= float(amount)
        for key, amount in row["charges"].items():
            spent[key] += float(amount)
        availability[row["backend_key"]] = {
            key: row[key] for key in ("failure_class", "consecutive_failures",
                                      "last_success", "retry_after", "reset_at",
                                      "next_eligible_at", "clock_domain")}
    return ActorPreparationProjection(pending, finished, reserved, spent, availability,
                                      profiles, requests, generations)


def _join_successor(row: Mapping[str, Any], prior: Mapping[str, Any] | None,
                    generation: int) -> None:
    """Bind a v2 request/profile to the exact settled predecessor (generation + 1)."""
    if prior is None:
        raise ActorStateRefused("successor has no settled predecessor profile")
    if row["generation"] != generation + 1:
        raise ActorStateRefused("successor generation is not predecessor + 1")
    if (row["predecessor_profile_digest"] != prior["target_profile_digest"]
            or row["predecessor_request_digest"] != prior["profile_request_digest"]):
        raise ActorStateRefused("successor does not join the exact predecessor")
    if row["target_revision_digest"] != prior["target_revision_digest"]:
        raise ActorStateRefused("successor target differs from predecessor")


__all__ = ["ActorPreparationProjection", "ActorStateRefused", "BUDGET_KEYS",
           "FINISH_SCHEMA", "INTENT_SCHEMA", "PROFILE_SCHEMA",
           "PROFILE_REQUEST_V2_SCHEMA", "PROFILE_V2_SCHEMA", "digest", "project_events",
           "validate_event"]
