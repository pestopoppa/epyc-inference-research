"""Pure closed A2 execution-transition and replay tests."""
from __future__ import annotations

import pytest

from . import a2_execution_state as S
from . import discovery_screen as D
from .test_discovery_screen import Invoker, _pair, _plan

PAIR = _pair()
PLAN = _plan(PAIR)
FRAME = "2" * 64

def _event(phase="anchor_bank", state="INTENT", index=0):
    offset = 0 if phase == "anchor_bank" else 3
    unit = sorted(PLAN.expected_units, key=lambda item: item.order_index)[offset + index]
    recipe = PAIR.anchor if phase == "anchor_bank" else PAIR.candidate
    producer = {"frame_digest": FRAME,
                "recipe_snapshot_digest": recipe.snapshot_digest,
                "recipe_execution_digest": recipe.execution_digest}
    if state == "INTENT":
        payload = {"unit": unit.to_dict(), "producer_identity": producer}
    elif state == "TERMINAL":
        invoker = Invoker(PLAN)
        invoker.frame_digest = FRAME
        result = invoker.invoke(unit, recipe)
        payload = {"status": "valid", "reason": None, "result": result.to_dict()}
    else:
        raise AssertionError("seal fixtures are not needed here")
    return D._event(phase, state, index, PLAN.digest, FRAME, payload)


def _transition(event=None, *, supervisor=1, logical_id="execution-1"):
    return S.make_transition(
        campaign_id="campaign-1", config_generation=1, config_digest="3" * 64,
        supervisor_incarnation=supervisor, logical_id=logical_id,
        event=event or _event())


def test_transition_identity_excludes_supervisor_but_retains_write_owner():
    first = _transition(supervisor=1)
    restarted = _transition(supervisor=2)
    assert first["execution_id"] == restarted["execution_id"]
    assert first["supervisor_incarnation"] == 1
    assert restarted["supervisor_incarnation"] == 2
    assert S.validate_transition(first) == first


def test_transition_is_closed_and_rederives_execution_identity():
    row = _transition()
    malformed = dict(row, execution_id="4" * 64)
    with pytest.raises(S.A2ExecutionStateRefused, match="identity differs"):
        S.validate_transition(malformed)
    malformed = dict(row, extra=True)
    with pytest.raises(S.A2ExecutionStateRefused, match="fields differ"):
        S.validate_transition(malformed)


def test_projection_preserves_pending_intent_and_refuses_out_of_order_events():
    intent = _transition()
    projected = S.project_transitions([intent])
    assert projected.pending_intents == ({
        "phase": "anchor_bank", "index": 0,
        "intent_event_digest": intent["event"]["event_digest"],
        "request_identity": intent["attempt_identity"]},)
    terminal = _transition(_event(state="TERMINAL"))
    with pytest.raises(S.A2ExecutionStateRefused, match="lacks its exact pending"):
        S.project_transitions([terminal])
    candidate = _transition(_event(phase="candidate_screen"))
    with pytest.raises(S.A2ExecutionStateRefused, match="precedes sealed"):
        S.project_transitions([candidate])


def test_projection_rejects_duplicate_events_and_fixed_bound():
    row = _transition()
    with pytest.raises(S.A2ExecutionStateRefused, match="repeats"):
        S.project_transitions([row, row])
    with pytest.raises(S.A2ExecutionStateRefused, match="bound"):
        S.project_transitions([row] * (S.MAX_PHASE_EVENTS + 1))


def test_terminal_must_match_the_exact_declared_unit():
    intent = _transition(_event(index=0))
    terminal = _transition(_event(state="TERMINAL", index=1))
    with pytest.raises(S.A2ExecutionStateRefused, match="exact pending intent"):
        S.project_transitions([intent, terminal])
