"""AKU-06l: registered v2 validator, fallback child attempts, build intent before launch.

Synthetic labelled fixtures only (tiny owned fixture children); no hardware, server or
campaign claim."""
from __future__ import annotations

import copy
import dataclasses

import pytest

from .. import journal
from . import (actor_preparation, campaign_control, operation_accounting as oa,
               scheduling as S, source_build_execution as execution)
from .test_operation_accounting import child, engine_and_selection
from .test_operation_settlement_v2 import (SHA, _close, _execute, _fixture, _record,
                                           _reopen, _terminal)


def _settle_request(box, outcome):
    rows = box.controller.unified_driver_children(box.selected.transition_id)
    return {"schema": oa.SETTLEMENT_REQUEST_V2_SCHEMA,
            "catalog_id": box.selected.catalog_id,
            "transition_id": box.selected.transition_id,
            "selection": box.selected.selection.to_dict(), "outcome": outcome,
            "children": [r["digest"] for r in rows], "denial_ref": None}


def _prepare(box):
    return box.owner.actor_consumer.prepare(
        box.selected.actor_request, stage_plan_digest=box.selected.stage_plan_digest)


# ------------------------------------------------------------ fallback attempts (pure) ---

def test_fallback_attempts_are_ordered_v2_children_and_v1_stays_first_invocation():
    planner = _record("planner", disposition="diagnostic")
    assert planner["schema"] == oa.CHILD_ACCOUNTED_SCHEMA and "attempt" not in planner
    fallback = _record("planner", planner, name="rc-planner-2", start=0.5, end=0.9,
                       terminal=_terminal("req-planner-2"), attempt=1)
    assert fallback["schema"] == oa.CHILD_ACCOUNTED_SCHEMA_V2 and fallback["attempt"] == 1
    critic = _record("critic", fallback)
    build = _record("build", critic)
    chain = [planner, fallback, critic, build]
    assert oa.validate_child_chain(chain) == chain
    # the final invocation of each role decides success; an earlier diagnostic does not
    oa.check_outcome_against_children("prerequisite", chain, None)
    assert journal._validate_native_payload(
        journal.KIND_UNIFIED_DRIVER_CHILD_ACCOUNTED, fallback) == []
    skipped = _record("planner", planner, name="rc-p3", start=0.5, end=0.9,
                      terminal=_terminal("req-p3"), attempt=2)
    with pytest.raises(oa.OperationAccountingRefused):  # attempt gap
        oa.validate_child_chain([planner, skipped])
    revisit = _record("planner", critic, name="rc-p4", start=9.0, end=9.5,
                      terminal=_terminal("req-p4"), attempt=1)
    with pytest.raises(oa.OperationAccountingRefused):  # a role is never revisited
        oa.validate_child_chain([planner, critic, revisit])
    with pytest.raises(oa.OperationAccountingRefused):  # v2 requires attempt >= 1
        oa.validate_child_record({**fallback, "attempt": 0})
    with pytest.raises(oa.OperationAccountingRefused):  # a v1 record cannot carry attempt
        oa.validate_child_record({**planner, "attempt": 1})
    failed_last = [planner, fallback, {**critic, "disposition": "diagnostic"}]
    with pytest.raises(oa.OperationAccountingRefused):
        oa.check_outcome_against_children("prerequisite", failed_last, None)
    # the v2 settlement validator admits more than three children; >3 digests were
    # refused before, so a pre-fallback reader still refuses such a settlement
    payload = {"schema": oa.SETTLED_V2_SCHEMA, "catalog_id": SHA, "campaign_id": "c",
               "config_generation": 1, "config_digest": SHA, "supervisor_incarnation": 1,
               "selection": {}, "prior_projection_digest": SHA,
               "after_projection_digest": SHA, "transition_id": SHA,
               "outcome": "prerequisite", "terminal_refs": ["a"],
               "children": [r["digest"] for r in chain], "denial_ref": None}
    assert oa.validate_settled_v2(payload) == []


def test_scheduler_accounts_fallback_children_once_and_refuses_skips():
    engine, selection, _ = engine_and_selection()
    kids = [child(selection, "planner", "p1", start=0.0, end=1.0),
            child(selection, "planner", "p2", start=1.0, end=3.0),
            child(selection, "critic", "c1", start=3.0, end=4.0)]
    assert engine.account_operation(selection, kids, outcome="failed") is True
    assert engine.campaign_attempts == 1
    assert engine.campaign_charged_seconds == pytest.approx(4.0)
    assert [r.receipt_id for r in engine.export_state().receipts] == ["p1", "p2", "c1"]
    assert not S.ordered_operation_roles(["planner", "build"])
    assert not S.ordered_operation_roles(["planner", "critic", "planner"])
    assert not S.ordered_operation_roles([])
    assert S.ordered_operation_roles(["planner", "planner", "critic", "critic", "build"])


# ------------------------------------------------------------ installed path ------------

def test_planner_fallback_invocation_is_accounted_in_order_exactly_once(tmp_path, monkeypatch):
    box = _fixture(tmp_path)
    try:
        consumer = box.owner.actor_consumer
        primary = consumer._chain("planner")[0]
        fallback = dataclasses.replace(primary, profile_id=primary.profile_id + "-fallback")
        original_chain = consumer._chain
        monkeypatch.setattr(consumer, "_chain", lambda role: (
            (primary, fallback) if role == "planner" else original_chain(role)))
        original_validate = actor_preparation._validate_output
        calls = []

        def first_invalid(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise actor_preparation.PreparationRefused("fixture: first planner invalid")
            return original_validate(*args, **kwargs)

        monkeypatch.setattr(actor_preparation, "_validate_output", first_invalid)
        assert _prepare(box).status == "proposed"
        c = box.controller
        chain = c.unified_driver_children(box.selected.transition_id)
        assert [(r["role"], oa.child_attempt(r)) for r in chain] == [
            ("planner", 0), ("planner", 1), ("critic", 0)]
        assert [r["schema"] for r in chain] == [
            oa.CHILD_ACCOUNTED_SCHEMA, oa.CHILD_ACCOUNTED_SCHEMA_V2, oa.CHILD_ACCOUNTED_SCHEMA]
        assert len({r["receipt"]["receipt_id"] for r in chain}) == 3
        before = c._scheduler_engine.accounting_view().receipt_count
        assert c.unified_driver_settle_v2(_settle_request(box, "failed"))["status"] \
            == "accepted"
        assert c._scheduler_engine.accounting_view().receipt_count == before + 3
        durable = copy.deepcopy(chain)
        box.runtime.close()
        c.close()
        box.controller, box.runtime = _reopen(box.materialized, box.registry)
        assert box.controller.unified_driver_children(box.selected.transition_id) \
            == tuple(durable)
    finally:
        _close(box)


def test_registered_validator_is_consulted_for_v2_and_a_v1_only_validator_refuses(tmp_path):
    box = _fixture(tmp_path)
    try:
        c = box.controller
        assert _prepare(box).status == "proposed"
        trusted = c._driver_settlement_validator
        assert trusted is not None  # the installed runtime's sole execution validator
        request = _settle_request(box, "failed")
        # the driver's validator rechecks the durable prefix and closed outcome rules
        with pytest.raises(Exception, match="differ"):
            trusted({**request, "children": list(reversed(request["children"]))})
        with pytest.raises(Exception, match="prerequisite"):
            trusted({**request, "outcome": "prerequisite"})
        assert trusted(request) == request
        seen = []

        def v1_only(supplied):  # an old reader: knows only v1 receipts
            seen.append(supplied["schema"])
            return {**supplied, "schema": campaign_control.DRIVER_SETTLEMENT_SCHEMA}

        c._driver_settlement_validator = v1_only
        with pytest.raises(campaign_control.ControlRefused, match="verifier"):
            c.unified_driver_settle_v2(request)
        assert seen == [oa.SETTLEMENT_REQUEST_V2_SCHEMA]
        assert box.selected.transition_id not in c._driver_settled
        spy = []
        c._driver_settlement_validator = lambda supplied: spy.append(supplied) or trusted(
            supplied)
        assert c.unified_driver_settle_v2(request)["status"] == "accepted"
        assert spy == [request]
        assert c.unified_driver_settle_v2(request)["status"] == "duplicate"
        assert len(spy) == 1  # an exact duplicate needs no re-verification
    finally:
        _close(box)


def test_build_intent_is_durable_before_launch_and_joins_the_build_child(tmp_path):
    box = _fixture(tmp_path)
    try:
        enrollment = _execute(box)
        assert enrollment["candidate_id"] == "akc-v2-fixture"
        c = box.controller
        transition = box.selected.transition_id
        intents = c._driver_child_intents[transition]
        build = c.unified_driver_children(transition)[-1]
        assert list(intents) == [build["request_id"]]
        assert intents[build["request_id"]]["binding_digest"] == build["binding_digest"]
        kinds = [e.kind for e in c._journal.read_all()]
        assert kinds.index(journal.KIND_UNIFIED_DRIVER_CHILD_INTENT) \
            < kinds.index(journal.KIND_UNIFIED_DRIVER_CHILD_ACCOUNTED, kinds.index(
                journal.KIND_UNIFIED_DRIVER_CHILD_INTENT))
        assert c.unified_driver_uncertain_children(transition) == ()
        with pytest.raises(campaign_control.ControlRefused, match="intent"):
            c._join_child_intent_locked({**build, "request_id": "build:other"})
        bad = {**intents[build["request_id"]], "role": "planner"}
        assert journal._validate_native_payload(journal.KIND_UNIFIED_DRIVER_CHILD_INTENT, bad)
        durable = copy.deepcopy(intents)
        box.runtime.close()
        c.close()
        box.controller, box.runtime = _reopen(box.materialized, box.registry)
        assert box.controller._driver_child_intents[transition] == durable
    finally:
        _close(box)


class _Crash(BaseException):
    pass


def test_crash_between_build_intent_and_launch_replays_uncertain_never_success(
        tmp_path, monkeypatch):
    box = _fixture(tmp_path)
    try:
        c = box.controller
        transition = box.selected.transition_id
        calls, launched = [], []
        original_call = execution.SourceBuildStageExecutor.__call__

        def recording_call(self, *args, **kwargs):
            calls.append((args, kwargs, self._accounting_binding))
            return original_call(self, *args, **kwargs)

        run_actor = c.run_worker_stage

        def crash(request):
            if request.stage != "build":
                return run_actor(request)
            launched.append(request)
            raise _Crash("fixture crash after durable build intent, before launch")

        monkeypatch.setattr(execution.SourceBuildStageExecutor, "__call__", recording_call)
        monkeypatch.setattr(c, "run_worker_stage", crash)
        with pytest.raises(_Crash):
            _execute(box, settle=False)
        assert len(launched) == 1 and len(calls) == 1
        assert [r["role"] for r in c.unified_driver_children(transition)] \
            == ["planner", "critic"]
        assert c.unified_driver_uncertain_children(transition) == (launched[0].request_id,)
        box.runtime.close()
        c.close()
        box.controller, box.runtime = _reopen(box.materialized, box.registry)
        c = box.controller
        assert c.unified_driver_uncertain_children(transition) == (launched[0].request_id,)
        for outcome in ("failed", "prerequisite"):  # neither zero cost nor success
            with pytest.raises(campaign_control.ControlRefused, match="uncertain"):
                c.unified_driver_settle_v2(_settle_request(box, outcome))
        assert transition not in c._driver_settled
        assert c.unified_driver_child_intent(launched[0])["replayed"] is True
        # the same executor, re-run against the reopened controller, never relaunches
        relaunched = []
        monkeypatch.setattr(c, "run_worker_stage", lambda r: relaunched.append(r))
        stage = box.owner.build_executor
        args, kwargs, binding = calls[0]
        stage.controller, stage._accounting_binding = c, binding
        with pytest.raises(execution.SourceBuildExecutionUncertain, match="uncertain"):
            original_call(stage, *args, **kwargs)
        assert relaunched == []
    finally:
        _close(box)
