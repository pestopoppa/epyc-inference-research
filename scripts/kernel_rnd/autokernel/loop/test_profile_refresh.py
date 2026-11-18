"""Hermetic AKU-06i tests: versioned profile request/successor semantics (synthetic)."""
from __future__ import annotations

import copy

import pytest

from .. import journal
from . import actor_preparation_state as state
from . import profile_refresh as refresh

TARGET = "6" * 64
BINDING = {"campaign_id": "campaign", "config_generation": 1, "config_digest": "1" * 64,
           "supervisor_id": "supervisor", "supervisor_incarnation": 2,
           "control_revision": 3}
TEMPLATE = {"schema": "epyc.autokernel.profile_preparation_request.v1",
            "target_revision_digest": TARGET,
            "stage_proposal": {"proposal_id": "profile:t"},
            "profile_contract": {"adapter_id": "synthetic"}}


def _verified(request=None, *, valid_until=100.0, domain="boot-a", extra=None,
              schema=state.PROFILE_SCHEMA, content="c1"):
    request = request or TEMPLATE
    loaded, artifact = {"m": content}, {"a": content}
    pc = {"p": content}
    row = {
        "schema": schema, "event": "PROFILE_VERIFIED", **BINDING,
        "catalog_id": "2" * 64, "transition_id": "3" * 64,
        "stage_plan_digest": "5" * 64, "profile_request_digest": state.digest(request),
        "target_revision_digest": TARGET,
        "target_profile_digest": state.digest({
            "profile_content": pc, "loaded_identity": loaded,
            "artifact_identity": artifact}),
        "profile_request": request, "profile_content": pc,
        "artifact_identity": artifact, "loaded_identity": loaded,
        "measurement_carrier": {"c": 1}, "clock_domain": domain,
        "verifier_ref": "v", "verified_at": 10.0, "valid_until": valid_until,
        "occurred_at": "t"}
    row.update(extra or {})
    return row


def _decide(projection, *, now, domain="boot-a", cap=3):
    return refresh.select_refresh(
        projection, target=TARGET, template=TEMPLATE, settlement_digest="9" * 64,
        attempt_cap=cap, binding=BINDING, now=now, clock_domain=domain, occurred_at="t")


def _successor_verified(req):
    return _verified(
        req["profile_request"], valid_until=300.0, schema=state.PROFILE_V2_SCHEMA,
        content="c2", extra={k: req[k] for k in (
            "generation", "predecessor_profile_digest", "predecessor_request_digest",
            "predecessor_settlement_digest")})


def test_v1_behaviour_is_unchanged_second_distinct_profile_still_refused():
    first = _verified()
    assert state.validate_event(first) == first
    assert state.project_events([first, first]).profile_generations[TARGET] == 1
    with pytest.raises(state.ActorStateRefused, match="replaced"):
        state.project_events([first, _verified(content="c2")])
    with pytest.raises(state.ActorStateRefused):  # v1 schema cannot carry v2 joins
        state.validate_event(_verified(extra={"generation": 2}))


def test_not_due_then_derived_then_reused_after_restart():
    proj = state.project_events([_verified()])
    assert _decide(proj, now=50.0).status == "not_due"
    decision = _decide(proj, now=150.0)
    assert decision.status == "derived" and decision.generation == 2
    req = decision.request
    assert req["predecessor_request_digest"] == state.digest(TEMPLATE)
    assert req["profile_request"]["stage_proposal"]["proposal_id"] == "profile:t:g2"
    assert _decide(proj, now=150.0).request == req  # deterministic
    # Journal native validation accepts it; projection holds it as the open request.
    assert not journal._validate_native_payload(journal.KIND_ACTOR_PREPARATION, req)
    persisted = state.project_events([_verified(), req])
    again = _decide(persisted, now=999.0, domain="boot-b")  # restart, later, new boot
    assert again.status == "reuse_open_request" and again.request == req


def test_successor_verified_joins_request_and_predecessor():
    req = _decide(state.project_events([_verified()]), now=150.0).request
    ok = _successor_verified(req)
    proj = state.project_events([_verified(), req, ok])
    assert proj.profile_generations[TARGET] == 2 and TARGET not in proj.profile_requests
    assert proj.profiles[TARGET] == ok
    bad_cases = [
        ({"predecessor_settlement_digest": "8" * 64}, "differs from request"),
        ({"predecessor_profile_digest": "8" * 64}, "exact predecessor"),
        ({"generation": 3}, "predecessor \\+ 1"),
    ]
    for extra, match in bad_cases:
        bad = _successor_verified(req)
        bad.update(extra)
        with pytest.raises(state.ActorStateRefused, match=match):
            state.project_events([_verified(), req, bad])
    with pytest.raises(state.ActorStateRefused, match="lacks its selected request"):
        state.project_events([_verified(), ok])
    with pytest.raises(state.ActorStateRefused, match="no settled predecessor"):
        state.project_events([req])
    with pytest.raises(state.ActorStateRefused, match="open successor"):
        state.project_events([_verified(), req, req])


def test_request_is_closed_and_bound_to_cap_and_predecessor():
    req = _decide(state.project_events([_verified()]), now=150.0).request
    for mutate in (lambda r: r.update(extra=1),
                   lambda r: r.update(attempt_cap=1),
                   lambda r: r.update(generation=1),
                   lambda r: r.update(profile_request_digest="a" * 64),
                   lambda r: r.update(template_digest="zz"),
                   lambda r: r.update(profile_request_digest=r["predecessor_request_digest"])):
        row = copy.deepcopy(req)
        mutate(row)
        with pytest.raises(state.ActorStateRefused):
            state.validate_event(row)
    stale = copy.deepcopy(req)
    stale["generation"], stale["attempt_cap"] = 3, 3
    with pytest.raises(state.ActorStateRefused, match="predecessor \\+ 1"):
        state.project_events([_verified(), stale])


def test_expiry_other_boot_exhausted_budget_and_no_predecessor():
    proj = state.project_events([_verified()])
    assert _decide(proj, now=5.0, domain="boot-b").status == "derived"  # no cross-boot trust
    assert _decide(proj, now=150.0, cap=1).status == "exhausted"
    assert _decide(state.project_events([]), now=150.0).status == "no_predecessor"
    req = _decide(proj, now=150.0).request
    two = state.project_events([_verified(), req, _successor_verified(req)])
    assert _decide(two, now=999.0, cap=2).status == "exhausted"
    third = _decide(two, now=999.0, cap=3)
    assert third.status == "derived" and third.generation == 3
    assert third.request["predecessor_request_digest"] == req["profile_request_digest"]
    # Chain of three profiles replays; generation 3 verified joins generation 2.
    final = _verified(third.request["profile_request"], schema=state.PROFILE_V2_SCHEMA,
                      content="c3", valid_until=900.0,
                      extra={k: third.request[k] for k in (
                          "generation", "predecessor_profile_digest",
                          "predecessor_request_digest", "predecessor_settlement_digest")})
    proj3 = state.project_events([_verified(), req, _successor_verified(req),
                                  third.request, final])
    assert proj3.profile_generations[TARGET] == 3


def test_native_journal_route_accepts_v2_and_rejects_malformed():
    req = _decide(state.project_events([_verified()]), now=150.0).request
    ok = _successor_verified(req)
    assert not journal._validate_native_payload(journal.KIND_ACTOR_PREPARATION, ok)
    bad = copy.deepcopy(ok)
    bad["generation"] = 1
    assert journal._validate_native_payload(journal.KIND_ACTOR_PREPARATION, bad)
