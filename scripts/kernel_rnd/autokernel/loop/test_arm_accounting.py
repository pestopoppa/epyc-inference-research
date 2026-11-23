"""AKU-23: arm-seconds budgets and utilisation. Pure arithmetic, no compute."""
import pytest

from . import arm_accounting as A


def test_held_vs_idle_while_claimed():
    r = A.account(100.0, 25.0, source="x")
    assert r["arm_seconds_idle_while_claimed"] == 75.0 and r["utilisation"] == 0.25
    assert r["idle_fraction_while_claimed"] == 0.75


def test_unobserved_held_is_never_replaced_by_wall_time():
    r = A.account(100.0, None, source="x")
    assert r["utilisation"] is None and r["arm_seconds_held"] is None and r["unavailable_reason"]


def test_held_longer_than_claim_is_refused_not_clamped_silently():
    r = A.account(10.0, 50.0, source="x")
    assert r["utilisation"] is None and "exceeds" in r["unavailable_reason"]


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), True, "3"])
def test_rejects_malformed_seconds(bad):
    with pytest.raises(ValueError):
        A.account(bad, 0.0, source="x")


def test_intervals_union_overlaps_once_and_clip_to_claim():
    r = A.from_intervals(100.0, 200.0, [(90, 120), (110, 130), (190, 250)], source="cpu")
    assert r["arm_seconds_held"] == 40.0 and r["arm_seconds_idle_while_claimed"] == 60.0


def test_gfx_trace_source_and_unavailable_trace():
    ok = A.from_gfx_trace(200.0, {"approx_gfx_activity_weighted_seconds": 50.0})
    assert ok["utilisation"] == 0.25 and ok["source"] == "gfx_counter_weighted"
    bad = A.from_gfx_trace(200.0, {"approx_gfx_activity_weighted_seconds": None,
                                   "unavailable_reason": "fewer than two counter samples"})
    assert bad["utilisation"] is None and "fewer than two" in bad["unavailable_reason"]
    assert A.from_gfx_trace(5.0, None)["utilisation"] is None


def test_total_counts_utilisation_over_measured_arms_only():
    rows = [A.account(100, 50, source="a"), A.account(100, None, source="b")]
    t = A.total(rows)
    assert t["arms"] == 2 and t["arms_measured"] == 1 and t["utilisation"] == 0.5
    assert t["arm_seconds_claimed"] == 200.0


def test_budget_is_in_arm_seconds_not_iterations():
    b = A.ArmSecondsBudget({"gpu": 600.0, "cpu": 6000.0})
    b.charge("gpu", A.account(100, 10, source="g"))
    b.charge("cpu", A.account(1800, 900, source="c"))      # one CPU arm = 18 GPU arms
    snap = b.snapshot()
    assert snap["surfaces"]["gpu"]["remaining"] == 500.0
    assert snap["surfaces"]["cpu"]["share_of_spend"] > 0.9
    assert b.allows("gpu", 500.0) and not b.allows("gpu", 500.1)


def test_budget_unknown_surface_and_empty_refused():
    with pytest.raises(ValueError):
        A.ArmSecondsBudget({})
    with pytest.raises(KeyError):
        A.ArmSecondsBudget({"gpu": 1}).charge("cpu", A.account(1, 1, source="x"))


# --- AKU-23 wiring: every row records utilisation (observed, or None + reason) -----------
import ast
import math
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


def _gpu_reading(cpu_launch, counter):
    node = next(n for n in ast.walk(ast.parse((Path(__file__).parent / "run.py").read_text()))
                if isinstance(n, ast.FunctionDef) and n.name == "gpu_reading")
    snap = SimpleNamespace(sample=lambda *_: None, snapshot=lambda: counter)
    scope = {"time": time, "math": math, "claim_started": 100.0, "cpu_launch": cpu_launch,
             "direct_launch": False, "gfx_counter": [snap if counter is not None else None],
             "arm_accounting": A}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "run.py", "exec"), scope)
    with mock.patch.object(time, "monotonic", return_value=200.0):
        return scope["gpu_reading"]()


def test_gpu_reading_uses_the_observed_gfx_counter():
    row = _gpu_reading(False, {"approx_gfx_activity_weighted_seconds": 25.0, "samples": []})
    assert row["arm_accounting"]["utilisation"] == 0.25
    assert row["idle_fraction_while_claimed"] == 0.75 and row["gpu_seconds_idle_while_claimed"] == 75.0


def test_gpu_reading_without_a_counter_is_none_with_reason_never_wall_time():
    row = _gpu_reading(False, None)
    assert row["idle_fraction_while_claimed"] is None and row["gpu_seconds_idle_while_claimed"] is None
    assert row["arm_accounting"]["unavailable_reason"] and row["claim_held_s"] == 100.0


def test_cpu_arm_row_records_utilisation_field_unobserved_with_reason():
    acct = _gpu_reading(True, None)["arm_accounting"]
    assert acct["utilisation"] is None and acct["source"] == "cpu_work_intervals"
    assert acct["arm_seconds_claimed"] == 100.0 and acct["unavailable_reason"]
    seen = A.from_cpu_work(100.0, [(110, 150)], claim_start=100.0)
    assert seen["arm_seconds_held"] == 40.0 and seen["utilisation"] == 0.4


def test_comparison_and_measurement_rows_carry_arm_utilisation():
    from . import bench
    c = bench.Comparison(surface="tg128", anchor_samples=[1.0], candidate_samples=[1.0], effect=0.0,
                         estimator="m", pairs=1, noise_floor_pct=None, residency={}, device_seconds=30.0)
    row = c.to_dict()["arm_utilisation"]
    assert row["utilisation"] is None and row["arm_seconds_claimed"] == 30.0 and row["unavailable_reason"]
    m = bench.Measurement(surface="tg128", samples=[1.0], model="m", launches=1, residency={}, device_seconds=9.0)
    assert m.to_dict()["arm_utilisation"]["arm_seconds_claimed"] == 9.0
    observed = bench._utilisation(40.0, lambda claimed: A.account(claimed, 10.0, source="obs"))
    assert observed["utilisation"] == 0.25
    broken = bench._utilisation(40.0, lambda claimed: 1 / 0)
    assert broken["utilisation"] is None and "ZeroDivisionError" in broken["unavailable_reason"]
