"""Pure hermetic tests for the AKU-04f per-unit A2 record chain (labelled synthetic data)."""
from __future__ import annotations

import pytest

from . import a2_execution_state as AX
from . import a2_unit_permit as P
from .test_a2_execution_state import PAIR, PLAN, FRAME, _event, _transition

EXEC = "5" * 64
PRODUCER = {"frame_digest": FRAME, "recipe_snapshot_digest": "6" * 64,
            "recipe_execution_digest": "7" * 64}
RESULT = "8" * 64


def _intent(owner=1, attempt=1, index=1, rng=(1, 1), prev=None):
    r = P.select_unit_range(PLAN, *rng)
    return P.make_intent(plan=PLAN, unit_range=r, unit_index=index, execution_id=EXEC,
                         frame_digest=FRAME, producer_identity=PRODUCER,
                         owner_incarnation=owner, attempt=attempt, prev_digest=prev)


def _chain(owner=1, upto="LAUNCHED"):
    intent = _intent(owner)
    rows = [intent]
    for kind in ("PERMIT_ISSUED", "LAUNCHED"):
        rows.append(P.make_follow(kind, intent, owner_incarnation=owner,
                                  prev_digest=rows[-1]["record_digest"]))
        if kind == upto:
            break
    return intent, rows


def _terminal(intent, rows, status="valid", reason=None, result=RESULT, owner=1):
    return P.make_follow("TERMINAL", intent, owner_incarnation=owner,
                         prev_digest=rows[-1]["record_digest"],
                         payload={"status": status, "reason": reason, "result_digest": result})


def test_range_is_digest_covered_and_membership_rederived():
    r = P.select_unit_range(PLAN, 1, 2)
    assert r["plan_digest"] == PLAN.digest and (r["first"], r["last"]) == (1, 2)
    with pytest.raises(P.A2UnitRefused, match="outside"):
        P.select_unit_range(PLAN, 0, 99)
    with pytest.raises(P.A2UnitRefused, match="digest-covered"):
        P.make_intent(plan=PLAN, unit_range=dict(r, members_digest="0" * 64), unit_index=1,
                      execution_id=EXEC, frame_digest=FRAME, producer_identity=PRODUCER,
                      owner_incarnation=1)
    with pytest.raises(P.A2UnitRefused, match="not a member"):
        _intent(index=4, rng=(1, 2))
    intent = _intent(index=1, rng=(0, 2))
    assert intent["plan_digest"] == PLAN.digest          # full plan, no one-unit rehash


def test_happy_path_one_shot_permit_then_terminal():
    intent, rows = _chain(upto="PERMIT_ISSUED")
    state = P.project_unit(rows)
    assert state.phase == "permit_issued"
    assert P.authorize_launch(state, current_owner=1, permit=state.permit_id)
    assert not P.authorize_launch(state, current_owner=2, permit=state.permit_id)
    assert not P.authorize_launch(state, current_owner=1, permit="0" * 64)
    intent, rows = _chain()
    state = P.project_unit(rows)
    assert state.phase == "launched_unknown"
    assert not P.authorize_launch(state, current_owner=1, permit=state.permit_id)  # consumed
    rows.append(_terminal(intent, rows))
    assert P.project_unit(rows).terminal_status == "valid"


def test_replayed_intent_never_authorizes_and_unknown_is_not_absence():
    intent = _intent()
    state = P.project_unit([intent])
    assert state.phase == "intent_only" and state.permit_id is None
    assert not P.authorize_launch(state, current_owner=1, permit="0" * 64)
    assert not state.absence_proven and not P.may_relaunch(state)
    _, rows = _chain()
    state = P.project_unit(rows)                         # replay after crash: launch unknown
    assert state.phase == "launched_unknown"
    assert not state.absence_proven and not P.may_relaunch(state)
    _, rows = _chain(upto="PERMIT_ISSUED")               # permit from an older incarnation
    state = P.project_unit(rows)
    assert not P.authorize_launch(state, current_owner=2, permit=state.permit_id)
    assert not P.may_relaunch(state)


def test_duplicate_intent_and_double_permit_and_double_launch_refused():
    intent, rows = _chain(upto="PERMIT_ISSUED")
    with pytest.raises(P.A2UnitRefused, match="first-issue"):
        P.project_unit(rows + [P.make_follow("PERMIT_ISSUED", intent, owner_incarnation=1,
                                             prev_digest=rows[-1]["record_digest"])])
    intent, rows = _chain()
    with pytest.raises(P.A2UnitRefused, match="unconsumed"):
        P.project_unit(rows + [P.make_follow("LAUNCHED", intent, owner_incarnation=1,
                                             prev_digest=rows[-1]["record_digest"])])
    with pytest.raises(P.A2UnitRefused, match="duplicate"):
        P.project_unit(rows + [_intent(attempt=2, prev=rows[-1]["record_digest"])])


def test_permit_by_other_owner_refused():
    intent = _intent(owner=1)
    bad = P.make_follow("PERMIT_ISSUED", intent, owner_incarnation=2,
                        prev_digest=intent["record_digest"])
    with pytest.raises(P.A2UnitRefused, match="first-issue"):
        P.project_unit([intent, bad])


def test_not_acquired_retry_preserves_exact_identity():
    intent, rows = _chain(upto="PERMIT_ISSUED")
    rows.append(P.make_follow("NOT_ACQUIRED", intent, owner_incarnation=1,
                              prev_digest=rows[-1]["record_digest"],
                              payload={"reason": "claim refused before launch"}))
    state = P.project_unit(rows)
    assert state.absence_proven and P.may_relaunch(state)
    retry = _intent(attempt=2, prev=rows[-1]["record_digest"])
    state = P.project_unit(rows + [retry])
    assert state.attempt == 2 and state.phase == "intent_only"
    assert state.intent["unit"] == intent["unit"]
    changed = P.make_intent(plan=PLAN, unit_range=P.select_unit_range(PLAN, 1, 1), unit_index=1,
                            execution_id=EXEC, frame_digest=FRAME,
                            producer_identity=dict(PRODUCER, recipe_execution_digest="9" * 64),
                            owner_incarnation=1, attempt=2, prev_digest=rows[-1]["record_digest"])
    with pytest.raises(P.A2UnitRefused, match="identity"):
        P.project_unit(rows + [changed])
    skip = _intent(attempt=3, prev=rows[-1]["record_digest"])
    with pytest.raises(P.A2UnitRefused, match="attempt"):
        P.project_unit(rows + [skip])
    # not-acquired cannot be written after a launch, nor by another owner
    intent2, rows2 = _chain()
    with pytest.raises(P.A2UnitRefused, match="not-acquired"):
        P.project_unit(rows2 + [P.make_follow("NOT_ACQUIRED", intent2, owner_incarnation=1,
                        prev_digest=rows2[-1]["record_digest"], payload={"reason": "x"})])


def test_diagnostic_terminal_is_terminal_and_recovers_unknown():
    intent, rows = _chain()
    rows.append(_terminal(intent, rows, status="diagnostic", reason="owner lost; unknown",
                          result=None, owner=2))
    state = P.project_unit(rows)
    assert state.phase == "terminal" and state.terminal_status == "diagnostic"
    assert not P.may_relaunch(state) and not state.absence_proven
    with pytest.raises(P.A2UnitRefused, match="follows a terminal"):
        P.project_unit(rows + [_intent(attempt=2, prev=rows[-1]["record_digest"])])
    with pytest.raises(P.A2UnitRefused, match="reason"):
        P.validate_record(_terminal(intent, rows[:-1], status="diagnostic", reason=None,
                                    result=None))


def test_valid_terminal_requires_launch_and_result_digest():
    intent = _intent()
    with pytest.raises(P.A2UnitRefused, match="requires a recorded launch"):
        P.project_unit([intent, _terminal(intent, [intent])])
    intent, rows = _chain()
    with pytest.raises(P.A2UnitRefused):
        P.validate_record(_terminal(intent, rows, result=None))


def test_records_are_closed_and_digest_checked():
    intent = _intent()
    assert P.validate_record(intent) == intent
    with pytest.raises(P.A2UnitRefused, match="fields differ"):
        P.validate_record(dict(intent, extra=1))
    with pytest.raises(P.A2UnitRefused, match="digest differs"):
        P.validate_record(dict(intent, attempt=2))
    with pytest.raises(P.A2UnitRefused, match="schema"):
        P.validate_record(dict(intent, schema="epyc.autokernel.a2_unit_record.v1"))
    intent2, rows = _chain(upto="PERMIT_ISSUED")
    broken = [rows[0], dict(rows[1], prev_digest="0" * 64)]
    with pytest.raises(P.A2UnitRefused):
        P.project_unit(broken)


def test_legacy_v1_transition_stream_unchanged_by_v2_records():
    # v1 events/validator are untouched: a v1 transition still validates and projects,
    # and v2 records are not accepted as v1 transitions (old-reader refusal kept).
    row = _transition(_event())
    assert AX.validate_transition(row) == row
    assert AX.project_transitions([row]).pending_intents
    with pytest.raises(AX.A2ExecutionStateRefused):
        AX.validate_transition(_intent())
