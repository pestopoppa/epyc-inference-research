"""R24-7 keep-drought trigger tests (pure Python, synthetic rows only)."""
from __future__ import annotations

import sqlite3

import pytest

from autokernel.controller import keep_drought as kd


def row(status, effect=0.0, floor=1.0):
    return {"status": status, "effect_fraction": effect, "floor_pct": floor}


def test_fires_after_k_consecutive_non_keeps():
    rows = [row("measured_null", 0.05) for _ in range(12)]
    hit = kd.detect(rows, k=5)
    assert hit and hit.non_keeps == 5 and hit.advisory


def test_keep_resets_the_drought():
    rows = [row("measured_null")] * 4 + [row("kept", 0.1)] + [row("regression", -0.1)] * 4
    assert kd.detect(rows, k=5) is None


def test_harness_events_are_not_measured_candidates():
    rows = [row("lane_error")] * 50 + [row("planner_transient")] * 50
    assert kd.detect(rows, k=3) is None


def test_window_best_inside_floor():
    rows = [row("measured_null", 0.002, 1.0)] * 6
    hit = kd.detect_window_best(rows, k=6)
    assert hit and hit.reason == "window_best_inside_floor"


def test_window_best_outside_floor_does_not_fire():
    rows = [row("measured_null", 0.002, 1.0)] * 5 + [row("measured_null", 0.03, 1.0)]
    assert kd.detect_window_best(rows, k=6) is None


def test_unknown_effect_is_not_evidence_of_null():
    rows = [row("measured_null", None, 1.0) for _ in range(10)]
    assert kd.detect_window_best(rows, k=3) is None
    assert kd.detect(rows, k=3) is not None   # still a non-keep run


def test_retrodict_bounds_and_open_naming_half():
    rows = [row("kept", 0.2)] + [row("measured_null", 0.0)] * 30
    result = kd.retrodict(rows, k=10)
    assert result["fired"] and result["within_bound"]
    assert result["names_defect"].startswith("OPEN")
    late = kd.retrodict([row("measured_null")] * 40, k=30)
    assert late["fired"] and not late["within_bound"]


def test_k_below_two_refused():
    with pytest.raises(ValueError):
        kd.detect([], k=1)


def test_load_rows_read_only(tmp_path):
    db = tmp_path / "experiments.db"
    con = sqlite3.connect(db)
    con.executescript("""
    CREATE TABLE experiments (attempt_id TEXT PRIMARY KEY, recorded_at TEXT, status TEXT,
        effect_fraction REAL, mechanism_id TEXT);
    CREATE TABLE same_code_measurements (attempt_id TEXT PRIMARY KEY, work_key TEXT,
        effect_fraction REAL, floor_pct REAL);
    INSERT INTO experiments VALUES ('a','2026-01-01','measured_null',0.001,'m');
    INSERT INTO same_code_measurements VALUES ('a','k',0.001,0.7);
    """)
    con.commit(); con.close()
    rows = kd.load_rows(db)
    assert rows[0]["floor_pct"] == 0.7
    with pytest.raises(sqlite3.OperationalError):
        c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        c.execute("INSERT INTO experiments VALUES ('b','x','kept',1,'m')")


def test_plan_validation():
    ok = {"schema": kd.SCHEMA, "trigger": {}, "diagnosis": {"repetition": "x"},
          "ranked_actions": [], "do_not_try": [{"experiment_id": "e1"}]}
    assert kd.validate_plan(ok) == []
    bad = dict(ok, diagnosis={"vibes": 1}, do_not_try=[{}], gate=True)
    defects = kd.validate_plan(bad)
    assert len(defects) == 3


# ---- ADVISORY ONLY (AutoKernel ruling): never a gate, never a blocking verdict ----

import copy
import dataclasses


def _drought_rows():
    return [row("kept", 0.2)] + [row("measured_null", 0.0)] * 30


def test_advisory_only_flag_and_fired_is_always_advisory():
    assert kd.ADVISORY_ONLY is True
    hit = kd.first_firing(_drought_rows(), k=5)
    assert hit is not None and hit.advisory is True
    with pytest.raises(dataclasses.FrozenInstanceError):
        hit.advisory = False


def test_no_result_carries_a_blocking_verdict_key():
    rows = _drought_rows()
    hit = kd.first_firing(rows, k=5)
    keys = set(dataclasses.asdict(hit)) | set(kd.retrodict(rows, k=5))
    assert not (keys & kd.FORBIDDEN_VERDICT_KEYS)
    assert set(dataclasses.asdict(hit)) == {"index", "non_keeps", "reason", "advisory"}


def test_detection_never_alters_rows_or_a_gate_decision():
    rows = _drought_rows()
    before = copy.deepcopy(rows)
    gate_decision = {"admit": True, "reason": "floor cleared"}
    gate_before = copy.deepcopy(gate_decision)
    kd.detect(rows, k=3); kd.detect_window_best(rows, k=3)
    kd.first_firing(rows, k=3); kd.retrodict(rows, k=3)
    assert rows == before and gate_decision == gate_before


def test_firing_does_not_change_the_firing_of_later_rows():
    """Rows after a firing are measured exactly as before: no state carries a block."""
    rows = _drought_rows()
    assert kd.first_firing(rows, k=5) is not None
    more = rows + [row("kept", 0.3)]
    assert kd.first_firing(more, k=5) == kd.first_firing(rows, k=5)
    assert kd.first_firing([row("kept", 0.3)] * 3, k=5) is None


def test_module_has_no_gate_surface():
    public = {n for n in dir(kd) if not n.startswith("_")}
    assert not {n for n in public if n.lower() in {"gate", "block", "admit", "refuse"}}
    assert not any(n.lower().startswith(("gate", "block", "enforce")) for n in public
                   if callable(getattr(kd, n)))


def test_plan_that_tries_to_gate_is_refused():
    plan = {"schema": kd.SCHEMA, "trigger": {}, "diagnosis": {"repetition": "x"},
            "ranked_actions": [], "do_not_try": [], "gate": {"block": True}}
    assert any("advisory" in d for d in kd.validate_plan(plan))
    assert kd.validate_plan(dict(plan, gate=None)) == []
