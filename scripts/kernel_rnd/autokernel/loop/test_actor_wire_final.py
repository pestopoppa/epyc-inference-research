"""actor-wire-final.json is written on PASS, FAIL and abort; the root reader accepts every one.

No providers, no inference: run_seat is replaced by a fake that writes the same evidence file shape.
The root-reader cross-check needs the root vidya worktree on PYTHONPATH (skipped otherwise).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from . import actors, actor_wire_gate as gate, actor_wire_test as wire
from .. import actor_wire_final as final

RECEIPT = {"cleanup_verified": True, "census_verified": True, "survivors": [], "scope": "t"}


def seat(role):
    backend = actors.Backend("codex", "gpt-5.6-sol", "high", sys.executable)
    return gate.describe(role, backend, actors.ActorSeat(bounded=False), timeout_s=20)


def request():
    return gate.request([seat("planner"), seat("author"), seat("critic")])


class Fake:
    """Replacement for run_seat: seat index -> behaviour ('pass' | 'fail' | exception instance)."""

    def __init__(self, plan, receipts=(RECEIPT,)):
        self.plan, self.calls, self.receipts = plan, 0, list(receipts)

    def __call__(self, row, evidence_dir=None):
        what = self.plan[self.calls]
        self.calls += 1
        reqs = [{"path": "/v1/x", "body": {"n": self.calls}}]
        if evidence_dir is not None:
            gate.write(Path(evidence_dir) / (gate.digest(row) + ".json"),
                       {"seat": row, "requests": reqs, "native_captures": {}, "suite": gate.suite_identity(),
                        "diagnostics": {"owned_call_cleanup": self.receipts}})
        if what == "fail":
            raise AssertionError("fixture wire check failed")
        if what != "pass":
            raise what
        return {"checks": dict.fromkeys(gate.CHECKS, True),
                "evidence_kinds": {k: list(v) for k, v in gate.EVIDENCE_KINDS.items()},
                "requests_sha256": gate.digest(reqs), "request_count": 1, "provider_protocol": "responses",
                "readonly_denial": "passed" if row["read_only"] else "not-readonly"}


def drive(monkeypatch, tmp_path, plan, **kw):
    monkeypatch.setattr(wire, "run_seat", Fake(plan, **kw))
    evidence = tmp_path / "ev"
    evidence.mkdir()
    req = request()
    err = None
    try:
        wire.run(req, evidence)
    except BaseException as exc:  # noqa: BLE001 - the abort paths are the point
        err = exc
    return req, evidence, err, json.loads((evidence / final.FINAL_NAME).read_text())


def reader():
    return pytest.importorskip("scripts.vidya.adapters.autokernel_actor_wire",
                               reason="root vidya worktree not on PYTHONPATH")


def assert_root_accepts(evidence):
    mod = reader()
    path = evidence / final.FINAL_NAME
    record = json.loads(path.read_text())
    assert mod.validate_final(record) == []
    assert mod.seal(record) == record
    rows = mod.native_rows(path)  # raises ProjectionError on a refused v1 record
    assert rows
    for row in rows:
        mod.project(row)
    assert mod.frames_for_path(path, as_of="2026-11-03T12:00:00Z")


def test_pass(monkeypatch, tmp_path):
    req, ev, err, rec = drive(monkeypatch, tmp_path, ["pass"] * 3)
    assert err is None and rec["status"] == "PASS" and rec["abort"] is None
    assert rec["attempted"] == rec["expected"] == 3 and rec["unvisited"] == []
    assert rec["schema"] == final.FINAL_SCHEMA and set(rec["cleanup"]["seats"]) == set(rec["evidence"])
    assert all(v["sha256"] for v in rec["evidence"].values())
    assert all(v == {"receipts": 1, "all_verified": True} for v in rec["cleanup"]["seats"].values())
    assert rec == final.seal(rec)
    assert (ev / "matrix-results.json").is_file()  # existing output unchanged
    assert not list(ev.glob(".actor-wire-*"))  # atomic write left no temporary
    assert_root_accepts(ev)


def test_fail(monkeypatch, tmp_path):
    req, ev, err, rec = drive(monkeypatch, tmp_path, ["pass", "fail", "pass"])
    assert isinstance(err, AssertionError)
    assert rec["status"] == "FAIL" and rec["abort"] is None and rec["attempted"] == 3
    assert [o["status"] for o in rec["outcomes"]] == ["PASS", "FAIL", "PASS"]
    assert "fixture wire check failed" in rec["outcomes"][1]["error"]
    assert len(rec["results"]) == 2
    assert_root_accepts(ev)


def test_abort_mid_matrix(monkeypatch, tmp_path):
    req, ev, err, rec = drive(monkeypatch, tmp_path, ["pass", RuntimeError("provider exploded")])
    assert isinstance(err, RuntimeError)
    assert rec["status"] == "FAIL" and rec["abort"] == "RuntimeError: provider exploded"
    assert rec["attempted"] == 1 and rec["expected"] == 3
    assert rec["unvisited"] == [gate.digest(r) for r in req["seats"][1:]]
    assert rec["cleanup"]["cleanup_uncertain"] is False
    assert_root_accepts(ev)


def test_abort_keyboard_interrupt_first_seat(monkeypatch, tmp_path):
    req, ev, err, rec = drive(monkeypatch, tmp_path, [KeyboardInterrupt()])
    assert isinstance(err, KeyboardInterrupt) and rec["attempted"] == 0 and rec["outcomes"] == []
    assert rec["abort"].startswith("KeyboardInterrupt") and rec["evidence"] == {}
    assert_root_accepts(ev)


def test_cleanup_uncertain_finalizes_as_aborted_fail(monkeypatch, tmp_path):
    bad = {**RECEIPT, "cleanup_verified": False}
    req, ev, err, rec = drive(monkeypatch, tmp_path, ["pass", wire.CleanupUncertain("survivor")], receipts=[bad])
    assert rec["status"] == "FAIL" and rec["cleanup"]["cleanup_uncertain"] is True
    assert rec["cleanup"]["seats"][rec["outcomes"][0]["seat"]]["all_verified"] is False
    assert rec["outcomes"][0]["status"] == "PASS"  # reader would refuse: PASS seat without verified cleanup
    mod = reader()
    assert mod.validate_final(rec)  # the reader refuses an unverified PASS seat; we never forge verification


def test_stop_file_abort_is_recorded(monkeypatch, tmp_path):
    monkeypatch.setattr(wire, "run_seat", Fake(["pass"] * 3))
    ev, stop = tmp_path / "ev", tmp_path / "STOP"
    ev.mkdir()
    stop.write_text("")
    with pytest.raises(RuntimeError):
        wire.run(request(), ev, stop_file=stop)
    rec = json.loads((ev / final.FINAL_NAME).read_text())
    assert rec["status"] == "FAIL" and "STOP" in rec["abort"] and rec["attempted"] == 0
    assert_root_accepts(ev)
