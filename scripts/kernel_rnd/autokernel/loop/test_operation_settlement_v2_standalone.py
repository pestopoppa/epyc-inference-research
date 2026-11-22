"""AKU-06l: single-stage SETTLED.v2, enrollment reference, uncertain-intent resolution.

Synthetic labelled fixtures only (tiny owned fixture children); no hardware, server or
campaign claim."""
from __future__ import annotations

import copy

import pytest

from .. import journal
from . import (campaign_control, driver_execution as de, operation_accounting as oa,
               scheduling, standalone_runtime, unified_driver as ud)
from .test_driver_execution import _owned_stack
from .test_operation_accounting import engine_and_selection
from .test_operation_accounting_v2_children import _Crash, _settle_request
from .test_operation_settlement_v2 import (SHA, _close, _execute, _fixture, _receipt_dict,
                                           _record, _reopen, _terminal)
from .test_profile_preparation_runtime import _fixture as profile_fixture, _start


def _stage(operation="runtime_comparison", **changes):
    kwargs = dict(
        campaign_id="camp", config_generation=1, config_digest=SHA, supervisor_incarnation=1,
        catalog_id="b" * 64, transition_id="c" * 64, selection_digest="d" * 64,
        operation=operation, role=oa.SINGLE_STAGE_ROLE, binding_digest="e" * 64,
        terminal=_terminal("req-stage"), receipt=_receipt_dict("rc-stage", start=0.0, end=1.0),
        disposition="accepted", predecessor=None)
    kwargs.update(changes)
    return oa.make_child_record(**kwargs)


def _settled(children, outcome, **extra):
    return {"schema": oa.SETTLED_V2_SCHEMA, "catalog_id": SHA, "campaign_id": "camp",
            "config_generation": 1, "config_digest": SHA, "supervisor_incarnation": 1,
            "selection": {}, "prior_projection_digest": SHA, "after_projection_digest": SHA,
            "transition_id": SHA, "outcome": outcome, "terminal_refs": ["native:x"],
            "children": children, "denial_ref": None, **extra}


# ------------------------------------------------------------ pure records -------------

def test_single_stage_child_is_v2_attempt_zero_and_v1_first_invocation_is_unchanged():
    row = _stage()
    assert (row["schema"], row["attempt"], row["ordinal"]) == (
        oa.CHILD_ACCOUNTED_SCHEMA_V2, 0, 0)
    assert oa.validate_child_chain([row]) == [row]
    assert not journal._validate_native_payload(
        journal.KIND_UNIFIED_DRIVER_CHILD_ACCOUNTED, row)
    planner = _record("planner")
    assert planner["schema"] == oa.CHILD_ACCOUNTED_SCHEMA and "attempt" not in planner
    v1_shaped = {k: v for k, v in row.items() if k != "attempt"} | {
        "schema": oa.CHILD_ACCOUNTED_SCHEMA}
    v1_shaped["digest"] = oa.child_digest(v1_shaped)
    for bad in (lambda: oa.validate_child_record(v1_shaped),  # v1 never carries a stage
                lambda: _stage(operation="source_build"),
                lambda: _record("planner", operation="profile_preparation"),
                lambda: _stage(predecessor=row["digest"]),
                lambda: oa.validate_child_chain([row, planner])):
        with pytest.raises(oa.OperationAccountingRefused):
            bad()
    assert oa.ordered_roles("profile_preparation", ["stage"])
    assert not oa.ordered_roles("profile_preparation", ["stage", "stage"])
    assert not oa.ordered_roles("source_build", ["stage"])


def test_single_stage_outcomes_and_settled_v2_optional_keys_are_closed():
    row, diagnostic = _stage(), _stage(disposition="diagnostic")
    for outcome in ("invalid", "prerequisite", "calibration", "failed"):
        oa.check_outcome_against_children(outcome, [row], None)
    oa.check_outcome_against_children("failed", [diagnostic], None)
    for outcome, rows, denial, extra in (
            ("prerequisite", [diagnostic], None, {}), ("invalid", [diagnostic], None, {}),
            ("failed", [row], SHA, {}), ("failed", [row], None, {"uncertain": 1}),
            ("bogus", [row], None, {})):
        with pytest.raises(oa.OperationAccountingRefused):
            oa.check_outcome_against_children(outcome, rows, denial, **extra)
    base = _settled([row["digest"]], "invalid", operation="runtime_comparison")
    assert oa.validate_settled_v2(base) == []
    legacy = {k: v for k, v in base.items() if k != "operation"} | {"outcome": "failed"}
    assert oa.validate_settled_v2(legacy) == []  # legacy v2 shape is unchanged
    for bad in ({**base, "operation": "source_build"}, {**base, "children": [SHA, "f" * 64]},
                {**base, "denial_ref": SHA}, {**base, "unknown": 1},
                {**legacy, "outcome": "calibration"}):
        assert oa.validate_settled_v2(bad)
    # the pre-AKU-06l2 v2 reader required the exact field set; the v1 reader the v1 schema
    assert set(base) != oa._SETTLED_V2_FIELDS
    assert journal._validate_native_payload(
        journal.KIND_UNIFIED_DRIVER_SETTLED, {**base, "schema": journal.UNIFIED_DRIVER_SETTLED_SCHEMA})


def test_enrollment_ref_binds_the_accepted_build_child_and_is_never_completion():
    _engine, selection, _prop = engine_and_selection()
    planner = _record("planner", selection_digest=selection.digest)
    critic = _record("critic", planner, selection_digest=selection.digest)
    chain = [planner, critic, _record("build", critic, selection_digest=selection.digest)]
    enrollment = {"transition_id": "c" * 64, "selection": selection.to_dict(),
                  "build_identity_digest": "9" * 64, "candidate_id": "fixture"}
    ref = oa.make_enrollment_ref(enrollment, chain)
    assert ref["build_child_digest"] == chain[-1]["digest"]
    assert ref["enrollment_digest"] == scheduling.digest(enrollment)
    oa.check_outcome_against_children("prerequisite", chain, None, enrollment_ref=ref)
    diagnostic = chain[:2] + [_record("build", critic, selection_digest=selection.digest,
                                      disposition="diagnostic")]
    for bad in (lambda: oa.check_outcome_against_children(
                    "failed", chain, None, enrollment_ref=ref),
                lambda: oa.check_outcome_against_children(  # enrollment is not a build
                    "prerequisite", chain[:2], None, enrollment_ref=ref),
                lambda: oa.check_outcome_against_children(
                    "prerequisite", diagnostic, None, enrollment_ref=ref),
                lambda: oa.check_enrollment_ref({**ref, "build_child_digest": SHA}, chain),
                lambda: oa.make_enrollment_ref({**enrollment, "transition_id": SHA}, chain)):
        with pytest.raises(oa.OperationAccountingRefused):
            bad()
    payload = _settled([c["digest"] for c in chain], "prerequisite", enrollment_ref=ref)
    assert oa.validate_settled_v2(payload) == []
    assert oa.validate_settled_v2({**payload, "outcome": "failed"})
    assert oa.validate_settled_v2({**payload, "enrollment_ref": {**ref, "schema": "x"}})


def test_intent_resolution_is_a_typed_uncertain_diagnostic_of_an_earlier_lifetime():
    intent = oa.make_child_intent(
        campaign_id="camp", config_generation=1, config_digest=SHA, supervisor_incarnation=1,
        catalog_id="b" * 64, transition_id="c" * 64, selection_digest="d" * 64,
        operation="source_build", role="build", binding_digest="e" * 64,
        request_id="req-build", plan_digest=SHA, lineage_id="lin", stage_id="s",
        artifact_contract_digest="f" * 64)
    row = oa.make_intent_resolution(intent, supervisor_incarnation=2, reason="fixture crash")
    assert (row["resolution"], row["cost"], row["intent_digest"]) == (
        "uncertain_diagnostic", "uncertain", intent["digest"])
    assert journal.KIND_UNIFIED_DRIVER_CHILD_INTENT_RESOLVED in journal.NATIVE_KINDS
    kind = journal.KIND_UNIFIED_DRIVER_CHILD_INTENT_RESOLVED
    assert journal._validate_native_payload(kind, row) == []
    with pytest.raises(oa.OperationAccountingRefused):
        oa.make_intent_resolution(intent, supervisor_incarnation=1, reason="same lifetime")
    for bad in ({**row, "resolution": "success"}, {**row, "cost": "zero"}, {**row, "x": 1}):
        assert journal._validate_native_payload(kind, bad)


# ------------------------------------------------------------ installed paths ----------

@pytest.mark.parametrize("return_code,outcome", [(0, "invalid"), (7, "failed")])
def test_standalone_runtime_settles_one_stage_child_as_v2_with_the_v1_transition(
        tmp_path, monkeypatch, return_code, outcome):
    driver, controller, _lifecycle, engine = _owned_stack(
        tmp_path, monkeypatch, return_code=return_code)
    connector = de.UnifiedDriverExecution(
        driver=driver, controller=controller, operation_settlement=True)
    try:
        issued = driver.tick(now=1.0)
        twin = engine._clone()
        receipt = connector.execute(issued)
        transition = receipt.transition_id
        settled = copy.deepcopy(controller._driver_settled[transition])
        children = controller.unified_driver_children(transition)
        assert settled["schema"] == journal.UNIFIED_DRIVER_SETTLED_SCHEMA_V2
        assert (settled["operation"], settled["outcome"]) == ("runtime_comparison", outcome)
        assert [c["role"] for c in children] == ["stage"] and "receipt" not in settled
        assert settled["children"] == [children[0]["digest"]]
        assert children[0]["disposition"] == ("accepted" if return_code == 0 else "diagnostic")
        request = ud._thaw(receipt.settlement_request)
        assert children[0]["receipt"] == request["receipt"]
        assert settled["terminal_refs"] == request["terminal_refs"]
        # the exact v1 scheduler transition over the one original receipt
        assert twin.preview_accounting(
            request["selection"], request["receipt"], outcome=outcome
        ).after.projection_digest == settled["after_projection_digest"]
        assert engine.accounting_view().receipt_count == 1
        assert connector.execute(issued).to_dict() == receipt.to_dict()
        assert engine.accounting_view().receipt_count == 1
        kinds = [e.kind for e in controller._journal.read_all()]
        assert kinds.count(journal.KIND_UNIFIED_DRIVER_SETTLED) == 1
        assert kinds.count(journal.KIND_UNIFIED_DRIVER_CHILD_ACCOUNTED) == 1
    finally:
        connector.close()
        controller.close()
    replay_engine = scheduling.SchedulerEngine(
        engine.config, scheduling.initial_state(engine.config, driver.resolved.campaign_id))
    with campaign_control.CampaignController(
            driver.resolved, tmp_path / "controller", snapshot_version=3,
            scheduler_engine=replay_engine) as replayed:
        assert replayed._driver_settled[transition] == settled
        assert replay_engine.operational_projection().projection_digest \
            == settled["after_projection_digest"]
        assert de.UnifiedDriverExecution.retry_durable(replayed, receipt)["status"] \
            == "duplicate"


def test_standalone_profile_settles_v2_and_feeds_planner_after_restart(tmp_path, monkeypatch):
    original = standalone_runtime.StandaloneRuntime.compose.__func__
    monkeypatch.setattr(standalone_runtime.StandaloneRuntime, "compose", classmethod(
        lambda cls, **kw: original(cls, **kw, config=standalone_runtime.StandaloneRuntimeConfig(
            operation_settlement=True))))
    materialized, registry, target, _, counter, profile = profile_fixture(
        tmp_path, selected_provider=True)
    controller, runtime = _start(materialized, registry)
    try:
        result = runtime.tick()
        assert result.status == "settled"
        snapshot = ud._thaw(controller.current_verified_profile_result(target))
        settled, row = snapshot["settlement"], snapshot["profile_event"]
        children = controller.unified_driver_children(settled["transition_id"])
        assert settled["schema"] == journal.UNIFIED_DRIVER_SETTLED_SCHEMA_V2
        assert (settled["operation"], settled["outcome"]) == ("profile_preparation",
                                                              "prerequisite")
        assert settled["terminal_refs"] == [row["verifier_ref"]]
        assert [c["role"] for c in children] == ["stage"]
        assert children[0]["receipt"] == ud._thaw(
            result.execution_receipt["provider_held_receipt"])
        assert runtime.profile_executor.planner_profiles(
            row["verified_at"])[target].to_dict() == profile
    finally:
        runtime.close()
        controller.close()
    controller, runtime = _start(materialized, registry)
    try:
        assert ud._thaw(controller.current_verified_profile_result(target)) == snapshot
        assert runtime.profile_executor.planner_profiles(
            row["verified_at"])[target].to_dict() == profile
        assert counter.read_text() == "x"
    finally:
        runtime.close()
        controller.close()


def test_resolved_uncertain_build_intent_settles_failed_never_success_or_relaunch(
        tmp_path, monkeypatch):
    box = _fixture(tmp_path)
    try:
        c = box.controller
        transition = box.selected.transition_id
        launched = []
        run_actor = c.run_worker_stage

        def crash(request):
            if request.stage != "build":
                return run_actor(request)
            launched.append(request)
            raise _Crash("fixture crash after durable build intent")

        monkeypatch.setattr(c, "run_worker_stage", crash)
        with pytest.raises(_Crash):
            _execute(box, settle=False)
        request_id = launched[0].request_id
        with pytest.raises(campaign_control.ControlRefused, match="earlier lifetime"):
            c.unified_driver_resolve_uncertain_intent(transition, request_id, reason="early")
        box.runtime.close()
        c.close()
        box.controller, box.runtime = _reopen(box.materialized, box.registry)
        c = box.controller
        before = c._scheduler_engine.accounting_view().receipt_count
        with pytest.raises(campaign_control.ControlRefused, match="intent"):
            c.unified_driver_resolve_uncertain_intent(transition, "build:other", reason="x")
        resolved = c.unified_driver_resolve_uncertain_intent(
            transition, request_id, reason="fixture: crash before build launch")
        assert resolved["status"] == "accepted"
        assert c.unified_driver_resolve_uncertain_intent(
            transition, request_id, reason="again")["status"] == "duplicate"
        assert c.unified_driver_uncertain_children(transition) == ()
        with pytest.raises(campaign_control.ControlRefused, match="never relaunched"):
            c.unified_driver_child_intent(launched[0])
        for outcome in ("prerequisite", "invalid"):
            with pytest.raises(campaign_control.ControlRefused):
                c.unified_driver_settle_v2(_settle_request(box, outcome))
        settled = c.unified_driver_settle_v2(_settle_request(box, "failed"))
        assert settled["status"] == "accepted"
        durable = copy.deepcopy(c._driver_settled[transition])
        assert f"uncertain-intent:{resolved['digest']}" in durable["terminal_refs"]
        # only the accounted planner/critic receipts are charged; nothing is fabricated
        assert c._scheduler_engine.accounting_view().receipt_count == before + 2
        digest = c._scheduler_engine.operational_projection().projection_digest
        box.runtime.close()
        c.close()
        box.controller, box.runtime = _reopen(box.materialized, box.registry)
        c = box.controller
        assert c._driver_settled[transition] == durable
        assert len(c.unified_driver_intent_resolutions(transition)) == 1
        assert c._scheduler_engine.operational_projection().projection_digest == digest
        assert c.unified_driver_settle_v2(_settle_request(box, "failed"))["status"] \
            == "duplicate"
    finally:
        _close(box)
