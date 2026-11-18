"""Pure bounded profile-refresh selection (AKU-06i).

Derives a successor request from the installed template, the exact settled predecessor
profile and the attempt cap, before any selection.  An already-recorded open request is
always reused (restart never mints a second one).  No grant, launch or hardware authority.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

from . import actor_preparation_state as state

_BINDING = ("campaign_id", "config_generation", "config_digest", "supervisor_id",
            "supervisor_incarnation", "control_revision")


@dataclass(frozen=True)
class RefreshDecision:
    status: str  # no_predecessor | not_due | reuse_open_request | derived | exhausted
    generation: int | None = None
    request: Mapping[str, Any] | None = None


def derive_successor_request(*, template: Mapping[str, Any], predecessor: Mapping[str, Any],
                             generation: int, settlement_digest: str, attempt_cap: int,
                             binding: Mapping[str, Any], now: float,
                             clock_domain: str, occurred_at: str) -> dict[str, Any]:
    """Deterministic PROFILE_REQUEST v2 from template + settled predecessor."""
    stage = template.get("stage_proposal")
    if not isinstance(stage, Mapping) or not isinstance(stage.get("proposal_id"), str):
        raise state.ActorStateRefused("template lacks a stage proposal id")
    if template.get("target_revision_digest") != predecessor["target_revision_digest"]:
        raise state.ActorStateRefused("template targets a different revision")
    request = copy.deepcopy(dict(template))
    request["stage_proposal"]["proposal_id"] = f"{stage['proposal_id']}:g{generation}"
    row = {
        "schema": state.PROFILE_REQUEST_V2_SCHEMA, "event": "PROFILE_REQUEST",
        **{name: binding[name] for name in _BINDING},
        "target_revision_digest": predecessor["target_revision_digest"],
        "profile_request": request, "profile_request_digest": state.digest(request),
        "template_digest": state.digest(template), "attempt_cap": attempt_cap,
        "generation": generation,
        "predecessor_profile_digest": predecessor["target_profile_digest"],
        "predecessor_request_digest": predecessor["profile_request_digest"],
        "predecessor_settlement_digest": settlement_digest,
        "clock_domain": clock_domain, "requested_at": now, "occurred_at": occurred_at,
    }
    return state.validate_event(row)


def select_refresh(projection: state.ActorPreparationProjection, *, target: str,
                   template: Mapping[str, Any], settlement_digest: str, attempt_cap: int,
                   binding: Mapping[str, Any], now: float, clock_domain: str,
                   occurred_at: str) -> RefreshDecision:
    profile = projection.profiles.get(target)
    if profile is None:
        return RefreshDecision("no_predecessor")
    generation = projection.profile_generations.get(target, 1)
    open_request = projection.profile_requests.get(target)
    if open_request is not None:  # durable reuse of the original selected request
        return RefreshDecision("reuse_open_request", open_request["generation"],
                               copy.deepcopy(dict(open_request)))
    # A different boot clock domain cannot prove validity (no cross-boot inference), so
    # the profile counts as stale; the attempt cap still bounds the cost.
    if profile["clock_domain"] == clock_domain and now < profile["valid_until"]:
        return RefreshDecision("not_due", generation)
    if generation >= attempt_cap:
        return RefreshDecision("exhausted", generation)
    request = derive_successor_request(
        template=template, predecessor=profile, generation=generation + 1,
        settlement_digest=settlement_digest, attempt_cap=attempt_cap, binding=binding,
        now=now, clock_domain=clock_domain, occurred_at=occurred_at)
    return RefreshDecision("derived", generation + 1, request)


__all__ = ["RefreshDecision", "derive_successor_request", "select_refresh"]
