"""tail_timing: per-iteration stage records (AK-TAIL-SPLIT phase 1) + read-only report."""
from __future__ import annotations

import json
import sqlite3
import threading
import time

from autokernel.loop import gates, tail_timing as tt
from autokernel.loop.test_pipeline import _Planner, _drive, _workers


def _all_stages(t):
    yield from t["actor"]
    for s in t["tail_sessions"]:
        yield s
        yield from s["stages"]


def _assert_monotone(case_t):
    for rec in _all_stages(case_t):
        start = rec.get("start", rec.get("ready"))
        assert rec["end"] >= start, rec
        if "acquired" in rec:
            assert rec["ready"] <= rec["acquired"] <= rec["end"], rec
        for key in ("duration_s", "wait_s", "held_s"):
            if key in rec:
                assert rec[key] >= 0, rec


def test_unit_timer_every_stage_end_not_before_start():
    clock = iter(float(i) for i in range(1000))
    t = tt.IterationTimer(lane="l0", clock=lambda: next(clock), wall=lambda: "w")
    for label in ("proposing a hypothesis", "critic pass 1: reviewing", "authoring the patch",
                  "critic pass 2: reviewing the diff", "building and gating"):
        t.step(label)
    with tt.timed_session(t, _null()):
        t.timed("gate", lambda: (True, [gates.Verdict("compile", True, elapsed_s=3.0),
                                        gates.Verdict("correctness", True, elapsed_s=2.0)]))
        t.timed("measure", lambda: None)
        t.timed("commit", lambda: "h")
    d = t.to_dict("kept")
    assert d["schema"] == tt.SCHEMA and d["died_at"] is None
    assert [a["stage"] for a in d["actor"]] == ["propose", "critic1", "author", "critic2"]
    gate = d["tail_sessions"][0]["stages"][0]
    assert gate["build_s"] == 3.0 and gate["correctness_s"] == 2.0
    _assert_monotone(d)
    json.dumps(d)


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def _timings(outcomes):
    return [getattr(o, tt.ATTR) for o in outcomes]


def test_tail_wait_recorded_under_contention_two_lanes():
    barrier = threading.Barrier(2)

    class P(_Planner):
        def propose(self, context):
            barrier.wait(timeout=10)
            return super().propose(context)

    def slow_gate(worker):
        def gate(h, paths):
            time.sleep(0.3)
            return True, [gates.Verdict("compile", True)]
        return gate

    outcomes, _ = _drive(workers=_workers(2), iterations=2, planner=P(), gate=slow_gate,
                         commit=lambda w, h, p, c: "0" * 40)
    timings = _timings(outcomes)
    assert len(timings) == 2
    for t in timings:
        _assert_monotone(t)
        assert t["tail_sessions"], t
    waits = sorted(s.get("wait_s", 0) for t in timings for s in t["tail_sessions"])
    assert waits[-1] >= 0.2, waits     # the second lane waited for the first's gate
    assert waits[0] < 0.2, waits


def test_correctness_fail_is_attributed():
    def failing_gate(worker):
        return lambda h, paths: (False, [
            gates.Verdict("compile", True, elapsed_s=1.0),
            gates.Verdict("reference_comparison", False, "wrong", elapsed_s=0.5)])

    outcomes, _ = _drive(workers=_workers(1), iterations=1, gate=failing_gate)
    t = _timings(outcomes)[0]
    _assert_monotone(t)
    assert t["died_at"] == "correctness_fail", t
    gate = t["tail_sessions"][0]["stages"][0]
    assert gate["failing_gate"] == "reference_comparison"
    assert gate["build_s"] == 1.0 and gate["correctness_s"] == 0.5
    assert not any(st["stage"] == "measure" for s in t["tail_sessions"] for st in s["stages"])


def test_report_on_fixture_store(tmp_path):
    def payload(died, held, corr, meas):
        stages = [{"stage": "gate", "start": 1.0, "end": 2.0, "duration_s": 1.0,
                   "build_s": 0.4, "correctness_s": corr, "died_at": died}]
        if meas:
            stages.append({"stage": "measure", "start": 2.0, "end": 2.0 + meas,
                           "duration_s": meas})
        return {tt.FIELD: {"schema": tt.SCHEMA, "died_at": died, "outcome": "x",
                           "actor": [{"stage": "propose", "start": 0, "end": 1,
                                      "duration_s": 1.0}],
                           "tail_sessions": [{"ready": 0.5, "acquired": 1.0, "end": 5,
                                              "wait_s": 0.5, "held_s": held,
                                              "stages": stages}]}}
    db = sqlite3.connect(tmp_path / "experiments.db")
    db.execute("CREATE TABLE experiments (status TEXT, recorded_at TEXT, payload TEXT)")
    for i, p in enumerate([payload("correctness_fail", 10.0, 3.0, None),
                           payload(None, 20.0, 2.0, 15.0), {"status": "legacy"}]):
        db.execute("INSERT INTO experiments VALUES (?,?,?)", ("s", f"t{i}", json.dumps(p)))
    db.commit()
    db.close()
    r = tt.summarize(tt.read_store(tmp_path))
    assert r["iterations_timed"] == 2 and r["iterations_untimed"] == 1
    assert r["tail_wait_total_s"] == 1.0
    assert r["tail_held_on_failed_candidates"]["correctness_fail_held_s"] == 10.0
    assert r["device_held_split"] == {"correctness_s": 5.0, "timing_s": 15.0}
    assert r["stages"]["tail_held"]["p90_s"] == 20.0
    assert tt.main([str(tmp_path)]) == 0
