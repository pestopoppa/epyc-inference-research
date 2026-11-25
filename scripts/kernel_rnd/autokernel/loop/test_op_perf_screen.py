from __future__ import annotations

import pytest

from . import op_perf_screen as ops
from . import served_shape_cases as ssc

N_SET = (1, 3, 4, 8, 16)


def _cell(key, ratio, floor=1.0, pairs=6, base=100.0):
    """Anchor `base` us; candidate base/(1+ratio): ratio>0 is a speedup."""
    return {"key": key, "anchor": [base] * pairs, "candidate": [base / (1.0 + ratio)] * pairs,
            "floor_pct": floor}


def test_one_cell_win_with_regression_elsewhere_is_rejected():
    # A is +50% on 90% of the weight, B regresses 5% (floor 1%): aggregate is a large win, still rejected.
    d = ops.decide([_cell("a", 0.50), _cell("b", -0.05)], {"a": ops.weight(9, 1.0), "b": ops.weight(1, 1.0)})
    assert d["aggregate_effect_pct"] > 30
    assert d["screen"] == ops.REJECT and d["regressed"] == ["b"] and d["screening_only"] is True
    r = ops.polar_result(d)
    assert r["polar"] == "FAIL" and r["verdict"].startswith("FAIL_OP_PERF_REGRESSION")


def test_measured_regression_rejects_even_when_other_cells_are_unmeasured():   # N1: the regression is checked BEFORE UNMEASURED
    d = ops.decide([_cell("a", 0.50), _cell("b", -0.05)], {"a": 1, "b": 1, "ghost": 1})   # "ghost" is weighted and never measured
    assert d["unmeasured"] == ["ghost"] and d["regressed"] == ["b"] and d["screen"] == ops.REJECT
    r = ops.polar_result(d)
    assert r["polar"] == "FAIL" and r["verdict"].startswith("FAIL_OP_PERF_REGRESSION")
    nowt = ops.decide([_cell("a", 0.50), _cell("b", -0.05)], {"a": 1, "b": 1, "ghost": 1} | {"a": None})   # an unweighted cell too
    assert nowt["screen"] == ops.REJECT
    assert ops.decide([_cell("a", 0.5)], {"a": 1, "ghost": 1})["screen"] == ops.UNMEASURED   # no regression: still UNMEASURED


def test_regression_inside_its_own_floor_does_not_mark_the_cell():
    d = ops.decide([_cell("a", 0.30), _cell("b", -0.005, floor=1.0)], {"a": 1, "b": 1})
    assert d["regressed"] == [] and d["screen"] == ops.PASS
    assert ops.polar_result(d)["polar"] == "PASS"


def test_floor_is_per_cell():
    # -3% is noise on a 5% floor and a regression on a 1% floor.
    assert ops.decide([_cell("a", 0.4), _cell("b", -0.03, floor=5.0)], {"a": 1, "b": 1})["regressed"] == []
    assert ops.decide([_cell("a", 0.4), _cell("b", -0.03, floor=1.0)], {"a": 1, "b": 1})["regressed"] == ["b"]


@pytest.mark.parametrize("break_cell", [
    lambda c: c.update(floor_pct=None), lambda c: c.update(anchor=c["anchor"][:3], candidate=c["candidate"][:3]),
    lambda c: c.update(candidate=c["candidate"][:-1]), lambda c: c.update(candidate=[float("nan")] * 6),
    lambda c: c.update(anchor=[]), lambda c: c.pop("candidate")])
def test_unmeasured_cell_is_rejected(break_cell):
    bad = _cell("b", 0.2)
    break_cell(bad)
    d = ops.decide([_cell("a", 0.5), bad], {"a": 1, "b": 1})
    assert d["screen"] == ops.UNMEASURED and d["unmeasured"] == ["b"] and d["screening_only"] is True
    r = ops.polar_result(d)
    assert "polar" not in r and r["verdict"].startswith("FUTILITY_HARNESS_INSTRUMENT")


def test_weighted_cell_with_no_measurement_and_unweighted_cell_are_rejected():
    assert ops.decide([_cell("a", 0.5)], {"a": 1, "ghost": 1})["unmeasured"] == ["ghost"]
    assert ops.decide([_cell("a", 0.5), _cell("b", 0.5)], {"a": 1})["unmeasured"] == ["b"]
    assert ops.decide([], {})["screen"] == ops.UNMEASURED


def test_aggregate_must_clear_weighted_floor_and_is_weighted():
    flat = ops.decide([_cell("a", 0.002), _cell("b", 0.002)], {"a": 1, "b": 1})
    assert flat["screen"] == ops.REJECT and flat["regressed"] == []
    assert ops.polar_result(flat)["verdict"].startswith("FAIL_OP_PERF_NO_GAIN")
    # Heavy weight on the winner dominates the aggregate (but never overrides a marked cell).
    d = ops.decide([_cell("a", 0.20), _cell("b", 0.0)], {"a": 9, "b": 1})
    assert d["aggregate_effect_pct"] == pytest.approx(18.0) and d["screen"] == ops.PASS


def test_direction_positive_effect_is_a_speedup():
    d = ops.decide([_cell("a", 1.0)], {"a": 1})  # candidate takes half the time
    assert d["cells"][0]["effect_pct"] == pytest.approx(100.0)


def test_polar_result_merges_base():
    d = ops.decide([_cell("a", 0.5)], {"a": 1})
    r = ops.polar_result(d, {"schema": "s", "stage": "x", "row": "r1", "complete": True})
    assert r["row"] == "r1" and r["stage"] == ops.STAGE and r["screening_only"] is True


def test_weight_rejects_nonpositive():
    for args in ((0, 1), (3, 0), (float("nan"), 1)):
        with pytest.raises(ValueError):
            ops.weight(*args)


@pytest.mark.parametrize("lane", ["q38fn", "ds41"])
def test_grid_equals_real_lane_gguf_shapes_times_n_set(lane):
    served = ssc.lane_served_shapes(lane)  # real call, GGUF-derived dims, no binary
    for family in ("MUL_MAT", "MUL_MAT_ID"):
        cells = ops.grid({"op_family": family, "dims": {}}, served, N_SET)
        want = {(s.name, n) for s in served if s.op == family for n in N_SET}
        assert {(c["shape"], c["n"]) for c in cells} == want and len(cells) == len(want)
        for c in cells:
            s = next(s for s in served if s.name == c["shape"])
            assert c["dims"] == {"k": s.k, "m": s.m, "n_mats": s.n_mats, "n_used": s.n_used}


def test_grid_class_only_narrows_served_shapes():
    served = ssc.lane_served_shapes("q38fn")
    p = ssc.LANE_PROFILES["q38fn"]
    cells = ops.grid({"op_family": "MUL_MAT", "dims": {"k": [p.hidden], "m": [p.shexp_ff, 999999]}},
                     served, [1, 8])
    assert {c["shape"] for c in cells} == {"q38fn_shexp_gate_up"} and len(cells) == 2
    # a dim value the lane does not serve adds nothing
    assert ops.grid({"op_family": "MUL_MAT", "dims": {"k": [5120]}}, served, [1]) == []
    with pytest.raises(ValueError):
        ops.grid({"op_family": "MUL_MAT", "dims": {"bogus": [1]}}, served, [1])
    with pytest.raises(ValueError):
        ops.grid({"op_family": "MUL_MAT", "dims": {}}, served, [])
