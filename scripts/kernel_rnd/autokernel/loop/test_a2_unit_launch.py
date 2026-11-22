"""AKU-04f through the real selected-unit call site (`CampaignController.run_worker_stage`).

Labelled fixtures: the controller, Journal, worker lifecycle and provider are the real owned
stack from test_driver_execution; denials/crashes are injected at the provider or lifecycle
seam. No server, measurement or scheduler discovery. The happy path through a real child is
asserted in test_driver_execution's real-controller helper (selected orders 0/3/5)."""
from __future__ import annotations

import pytest

from .. import journal as journal_module
from . import a2_unit_permit as P
from . import a2_unit_transport as T
from . import campaign_control, scheduling, serving
from . import unified_worker as worker
from . import worker_lifecycle as wl
from .test_driver_execution import _as_observed_v2, _owned_stack, _resume
from .test_planned_unit_selection import _six_unit_transport


def _stack(tmp_path, monkeypatch, order=3):
    driver, controller, lifecycle, engine = _owned_stack(tmp_path, monkeypatch)
    original, _ = _as_observed_v2(driver.materialize_runtime(driver.tick(now=1.0)),
                                  serving._measure_once)
    return driver, controller, engine, original, _six_unit_transport(original, order)


def _run(controller, prepared):
    invocation = worker.PlannedWorkerInvocation.open(prepared,
                                                     worker.ParentUnitEvidenceAuthority())
    dispatch = prepared.selected_dispatch
    request = invocation.stage_request(
        request_id=dispatch.request_id if dispatch else "legacy-fixture",
        lineage_id=dispatch.lineage_id if dispatch else "legacy-lineage",
        stage_id=dispatch.stage_id if dispatch else "legacy-stage",
        control_revision=controller.control_revision)
    return controller.run_worker_stage(request, planned_invocation=invocation)


def _rows(controller):
    return [e.payload for e in controller._journal.read_all()
            if e.kind == journal_module.KIND_A2_RUNTIME_EXECUTION]


def _identity(controller, prepared):
    return T.selected_launch_identity(prepared, campaign_id=controller.resolved.campaign_id,
                                      config_generation=controller.config_generation,
                                      config_digest=controller.config_digest)


def _restart(tmp_path, driver, controller, engine):
    provider = controller._lifecycle_provider
    controller.close()
    restarted = campaign_control.CampaignController(
        driver.resolved, tmp_path / "controller", snapshot_version=3,
        scheduler_engine=scheduling.SchedulerEngine(engine.config, scheduling.initial_state(
            engine.config, driver.resolved.campaign_id)),
        readiness_check=lambda: (True, None), lifecycle_provider=provider)
    restarted.__enter__()
    _resume(restarted, driver.resolved.campaign_id)
    return restarted


def _deny(controller, calls):
    def deny(request, container_id, _deadline):
        calls.append(request.request_id)
        return wl.AuthorizationDenied(request.request_id, container_id, "fixture denied")
    controller._lifecycle_provider.authorize = deny


def test_identity_comes_from_real_prepared_and_controller_not_caller(tmp_path, monkeypatch):
    _driver, controller, _engine, original, prepared = _stack(tmp_path, monkeypatch)
    try:
        ident = _identity(controller, prepared)
        unit = prepared.selected_range.units(prepared.plan)[0]
        assert ident["plan"].digest == prepared.plan.digest and len(
            ident["plan"].expected_units) == 6                     # full plan, no rehash
        assert ident["unit_index"] == unit.order_index == 3
        assert (ident["unit_range"]["first"], ident["unit_range"]["last"]) == (3, 3)
        recipe = getattr(prepared.runtime_pair, unit.arm)
        assert ident["producer_identity"] == {
            "frame_digest": ident["frame_digest"],
            "recipe_snapshot_digest": recipe.snapshot_digest,
            "recipe_execution_digest": recipe.execution_digest}
        # a new transport attempt id is the SAME chain: it cannot reopen a unit
        retry = _six_unit_transport(original, 3, attempt_id="fixture-attempt-1")
        assert T.identity_key(_identity(controller, retry)) == T.identity_key(ident)
        with pytest.raises(P.A2UnitRefused, match="campaign/config"):
            T.selected_launch_identity(prepared, campaign_id=controller.resolved.campaign_id,
                                       config_generation=controller.config_generation + 1,
                                       config_digest=controller.config_digest)
        with pytest.raises(P.A2UnitRefused, match="not a selected"):
            _identity(controller, original)
    finally:
        controller.close()


def test_unselected_unit_is_refused_before_any_record(tmp_path, monkeypatch):
    _driver, controller, _engine, _original, prepared = _stack(tmp_path, monkeypatch)
    try:
        sink = []
        for index in (2, 4):   # neighbours inside the full plan, outside the selection
            with pytest.raises(P.A2UnitRefused, match="not a member"):
                T.reserve({**_identity(controller, prepared), "unit_index": index},
                          owner_incarnation=controller.supervisor_incarnation,
                          prior_records=(), append=sink.append)
        assert sink == [] and _rows(controller) == []
    finally:
        controller.close()


def test_not_acquired_is_durable_and_retry_preserves_exact_identity(tmp_path, monkeypatch):
    driver, controller, engine, _original, prepared = _stack(tmp_path, monkeypatch)
    calls = []
    _deny(controller, calls)
    with pytest.raises(wl.WaitingAuthority, match="fixture denied"):
        _run(controller, prepared)
    rows = _rows(controller)
    assert [r["kind"] for r in rows] == ["INTENT", "PERMIT_ISSUED", "NOT_ACQUIRED"]
    assert P.project_unit(rows).absence_proven and len(calls) == 1
    with pytest.raises(wl.WaitingAuthority):
        _run(controller, prepared)              # retry: same identity, attempt 2
    rows = _rows(controller)
    assert [r["kind"] for r in rows[3:]] == ["INTENT", "PERMIT_ISSUED", "NOT_ACQUIRED"]
    assert rows[3]["attempt"] == 2 and len({P.unit_key(r) for r in rows}) == 1
    assert {r["owner_incarnation"] for r in rows} == {controller.supervisor_incarnation}
    restarted = _restart(tmp_path, driver, controller, engine)
    try:   # durable absence survives restart; the next retry is a new owner's attempt 3
        with pytest.raises(wl.WaitingAuthority):
            _run(restarted, prepared)
        rows = _rows(restarted)
        assert rows[6]["attempt"] == 3 and rows[6]["owner_incarnation"] == (
            restarted.supervisor_incarnation) != rows[0]["owner_incarnation"]
        assert len({P.unit_key(r) for r in rows}) == 1 and len(calls) == 3
    finally:
        restarted.close()


def test_crash_after_intent_is_unknown_and_never_relaunched(tmp_path, monkeypatch):
    driver, controller, engine, _original, prepared = _stack(tmp_path, monkeypatch)
    calls = []
    _deny(controller, calls)
    real_append = controller.append_a2_unit_record

    def crash_after_intent(record):
        if record["kind"] == "PERMIT_ISSUED":
            raise OSError("fixture crash after fsynced INTENT")
        return real_append(record)
    controller.append_a2_unit_record = crash_after_intent
    with pytest.raises(campaign_control.ControlRefused, match="fixture crash"):
        _run(controller, prepared)
    assert [r["kind"] for r in _rows(controller)] == ["INTENT"] and calls == []
    restarted = _restart(tmp_path, driver, controller, engine)
    try:
        with pytest.raises(campaign_control.ControlRefused, match="intent_only"):
            _run(restarted, prepared)
        state = P.project_unit(_rows(restarted))
        assert not state.absence_proven and not P.may_relaunch(state) and calls == []
    finally:
        restarted.close()


def test_stale_incarnation_permit_is_refused_after_restart(tmp_path, monkeypatch):
    driver, controller, engine, _original, prepared = _stack(tmp_path, monkeypatch)
    lifecycle = controller._worker_lifecycle

    def crash(*_args, **_kwargs):
        raise RuntimeError("fixture crash after permit, outcome unknown")
    monkeypatch.setattr(lifecycle, "run_stage", crash)
    with pytest.raises(RuntimeError, match="fixture crash"):
        _run(controller, prepared)
    rows = _rows(controller)
    assert [r["kind"] for r in rows] == ["INTENT", "PERMIT_ISSUED"]   # UNKNOWN, not absent
    restarted = _restart(tmp_path, driver, controller, engine)
    try:
        state = P.project_unit(list(restarted.replay_a2_unit_records(P.unit_key(rows[0]))))
        assert state.phase == "permit_issued" and state.permit_owner == rows[0][
            "owner_incarnation"] != restarted.supervisor_incarnation
        assert not P.authorize_launch(state, current_owner=restarted.supervisor_incarnation,
                                      permit=state.permit_id)
        with pytest.raises(campaign_control.ControlRefused, match="current owner"):
            restarted.append_a2_unit_record(P.make_follow(
                "LAUNCHED", rows[0], owner_incarnation=state.permit_owner,
                prev_digest=rows[-1]["record_digest"]))
        with pytest.raises(campaign_control.ControlRefused, match="permit_issued"):
            _run(restarted, prepared)
        assert len(_rows(restarted)) == 2
    finally:
        restarted.close()


def test_legacy_unselected_prepared_writes_no_unit_records(tmp_path, monkeypatch):
    _driver, controller, _engine, original, _prepared = _stack(tmp_path, monkeypatch)
    calls = []
    _deny(controller, calls)
    try:
        with pytest.raises(wl.WaitingAuthority, match="fixture denied"):
            _run(controller, original)
        assert calls == ["legacy-fixture"] and _rows(controller) == []
    finally:
        controller.close()
