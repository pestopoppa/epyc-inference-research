"""AKU-04f: v2 unit records through Journal validation, controller replay and the opt-in glue.

Hermetic: synthetic plan/records, a fake launcher, a real CampaignController + Journal in tmp.
No launch, no scheduler discovery."""
from __future__ import annotations

import copy
import json

import pytest

from .. import journal as J
from . import a2_execution_state as AX
from . import a2_unit_permit as P
from . import a2_unit_transport as T
from . import campaign_control
from .test_a2_execution_state import FRAME, PLAN, _transition
from .test_a2_unit_permit import EXEC, PRODUCER, RESULT, _chain, _intent
from .test_campaign_control import _resolved

K = J.KIND_A2_RUNTIME_EXECUTION


def _launch(prior, sink, launcher, owner=1):
    return T.launch_selected_unit(
        plan=PLAN, unit_range=P.select_unit_range(PLAN, 1, 1), unit_index=1,
        execution_id=EXEC, frame_digest=FRAME, producer_identity=PRODUCER,
        owner_incarnation=owner, prior_records=prior, append=sink, launcher=launcher)


def test_journal_validator_accepts_v2_and_refuses_malformed_or_forged():
    _, rows = _chain()
    for row in rows:
        assert J._validate_native_payload(K, row) == []
    forged = copy.deepcopy(rows[1])
    forged["payload"]["permit_id"] = "0" * 64          # digest no longer matches
    assert J._validate_native_payload(K, forged)
    extra = dict(rows[0], surprise=1)
    assert J._validate_native_payload(K, extra)
    bad_kind = dict(rows[0], kind="BOGUS")
    assert J._validate_native_payload(K, bad_kind)
    assert J._validate_native_payload(K, {"schema": P.RECORD_SCHEMA})


def test_legacy_v1_validation_and_bytes_unchanged_and_old_reader_refuses_v2():
    v1 = _transition()
    assert J._validate_native_payload(K, v1) == []
    assert AX.validate_transition(v1) == v1            # v1 validator untouched
    assert json.dumps(v1, sort_keys=True) == json.dumps(
        AX.make_transition(campaign_id="campaign-1", config_generation=1,
                           config_digest="3" * 64, supervisor_incarnation=1,
                           logical_id="execution-1", event=v1["event"]), sort_keys=True)
    assert J._validate_native_payload(K, {"schema": "x"})  # unknown schema still refused
    _, rows = _chain()
    with pytest.raises(AX.A2ExecutionStateRefused):     # the legacy (old-reader) validator
        AX.validate_transition(rows[0])


def test_replay_units_projects_permit_state_and_replay_never_authorizes():
    intent, rows = _chain(upto="PERMIT_ISSUED")
    states = P.replay_units([_transition(), *rows])     # v1 payloads ignored
    (state,) = states.values()
    assert state.phase == "permit_issued"
    assert not P.authorize_launch(state, current_owner=2, permit=state.permit_id)
    _, launched = _chain()
    (state,) = P.replay_units(launched).values()
    assert state.phase == "launched_unknown" and not P.may_relaunch(state)
    broken = [launched[0], launched[2]]
    with pytest.raises(P.A2UnitRefused):
        P.replay_units(broken)


def test_glue_happy_path_orders_records_and_replays_terminal():
    sink: list = []
    seen = []
    out = _launch([], sink.append, lambda permit: seen.append(permit) or
                  {"status": "valid", "reason": None, "result_digest": RESULT})
    assert [r["kind"] for r in sink] == ["INTENT", "PERMIT_ISSUED", "LAUNCHED", "TERMINAL"]
    assert seen == [sink[1]["payload"]["permit_id"]] and out["phase"] == "terminal"
    with pytest.raises(P.A2UnitRefused, match="never authorizes"):
        _launch(sink, sink.append, lambda p: pytest.fail("must not launch"))


def test_glue_not_acquired_allows_exact_retry_but_unknown_never_relaunches():
    sink: list = []

    def refuse(permit):
        raise T.AcquisitionFailed("claim refused")
    out = _launch([], sink.append, refuse)
    assert [r["kind"] for r in sink] == ["INTENT", "PERMIT_ISSUED", "NOT_ACQUIRED"]
    assert out["phase"] == "not_acquired"
    out = _launch(list(sink), sink.append,
                  lambda p: {"status": "valid", "reason": None, "result_digest": RESULT})
    assert out["attempt"] == 2 and out["phase"] == "terminal"
    # a launcher crash after the permit leaves UNKNOWN: relaunch refused
    crash: list = []

    def boom(permit):
        raise RuntimeError("process vanished")
    with pytest.raises(RuntimeError):
        _launch([], crash.append, boom)
    assert [r["kind"] for r in crash] == ["INTENT", "PERMIT_ISSUED"]
    with pytest.raises(P.A2UnitRefused, match="never authorizes"):
        _launch(crash, crash.append, lambda p: pytest.fail("must not launch"))


def test_glue_sink_failure_aborts_before_launch():
    calls = []

    def sink(record):
        if record["kind"] == "PERMIT_ISSUED":
            raise OSError("disk full")
    with pytest.raises(OSError):
        _launch([], sink, lambda p: calls.append(p))
    assert calls == []


def test_controller_journal_round_trip_restart_replays_permit_state(tmp_path):
    store = tmp_path / "service"
    with campaign_control.CampaignController(_resolved("camp-1"), store) as c:
        out = _launch([], c.append_a2_unit_record, lambda p: {
            "status": "valid", "reason": None, "result_digest": RESULT})
        assert out["phase"] == "terminal"
        key = P.unit_key(out["records"][0])
        assert c.append_a2_unit_record(out["records"][0]) == out["records"][0]  # exact dup
        with pytest.raises(campaign_control.ControlRefused):
            c.append_a2_unit_record(P.make_follow(   # a record after the terminal
                "LAUNCHED", out["records"][0], owner_incarnation=1,
                prev_digest=out["records"][-1]["record_digest"]))
    with campaign_control.CampaignController(_resolved("camp-1"), store) as c2:
        replayed = c2.replay_a2_unit_records(key)
        assert [r["kind"] for r in replayed] == ["INTENT", "PERMIT_ISSUED", "LAUNCHED",
                                                "TERMINAL"]
        assert P.project_unit(replayed).phase == "terminal"
        with pytest.raises(P.A2UnitRefused):
            _launch(list(replayed), c2.append_a2_unit_record, lambda p: None)
