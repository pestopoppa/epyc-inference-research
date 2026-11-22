"""AKU-06d2: an actor-only operation settles under the explicit abstained policy."""
from __future__ import annotations

import pytest

from . import actor_settlement_policy as policy
from . import operation_accounting as oa
from .test_operation_accounting_v2_children import _prepare, _settle_request
from .test_operation_settlement_v2 import SHA, _close, _fixture, _record, _reopen


def _actor(role, previous=None, **changes):
    return _record(role, previous, operation="actor_preparation", **changes)


def test_pure_policy_admits_only_accepted_actor_only_chains():
    planner = _actor("planner")
    critic = _actor("critic", planner)
    oa.check_outcome_against_children("abstained", [planner, critic], None)
    assert policy.outcome_for([planner, critic]) == "abstained"
    refused = [
        ([planner], None),  # critic never ran
        ([planner, _actor("critic", planner, disposition="diagnostic")], None),
        ([planner, critic], "a" * 64),  # a denial is not a policy settlement
        ([_record("planner"), _record("critic", _record("planner"))], None),  # source build
    ]
    for rows, denial in refused:
        with pytest.raises(oa.OperationAccountingRefused, match="actor-only"):
            oa.check_outcome_against_children("abstained", rows, denial)
    assert policy.outcome_for([planner]) == "failed"
    payload = {"schema": oa.SETTLED_V2_SCHEMA, "catalog_id": SHA, "campaign_id": "c",
               "config_generation": 1, "config_digest": SHA, "supervisor_incarnation": 1,
               "selection": {}, "prior_projection_digest": SHA, "after_projection_digest": SHA,
               "transition_id": SHA, "outcome": "abstained", "terminal_refs": ["a"],
               "children": [planner["digest"], critic["digest"]], "denial_ref": None}
    assert oa.validate_settled_v2(payload) == []


def test_actor_only_operation_settles_abstained_and_replays(tmp_path, monkeypatch):
    from . import actor_lifecycle
    real = actor_lifecycle.account_child  # FIXTURE: account this actor as a source actor
    monkeypatch.setattr(actor_lifecycle, "account_child", lambda *a, **k: real(
        *a, **{**k, "operation": "actor_preparation"}))
    box = _fixture(tmp_path)
    try:
        assert _prepare(box).status == "proposed"
        c = box.controller
        chain = c.unified_driver_children(box.selected.transition_id)
        assert {row["operation"] for row in chain} == {"actor_preparation"}
        outcome = policy.outcome_for(chain)
        assert outcome == "abstained"
        settled = c.unified_driver_settle_v2(_settle_request(box, outcome))
        assert settled["status"] == "accepted"
        assert c._driver_settled[box.selected.transition_id]["outcome"] == "abstained"
        box.runtime.close()
        c.close()
        box.controller, box.runtime = _reopen(box.materialized, box.registry)
        assert box.controller._driver_settled[box.selected.transition_id]["outcome"] \
            == "abstained"
    finally:
        _close(box)


def test_source_build_operation_cannot_use_the_actor_only_policy(tmp_path):
    box = _fixture(tmp_path)
    try:
        assert _prepare(box).status == "proposed"
        with pytest.raises(Exception, match="actor-only"):
            box.controller.unified_driver_settle_v2(_settle_request(box, "abstained"))
    finally:
        _close(box)
