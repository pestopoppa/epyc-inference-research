"""R23-44 compound-then-gate policy. Pure logic, no GPU: the loop injects the compounded
bench number and the serving row, and these pin the decisions those inputs must yield."""
from autokernel.loop import accumulate as A


def _bundle(cor="cor0", tip="cor0", keeps=(), pct=0.0):
    b = A.Bundle(champion_of_record=cor, tip=tip)
    for i, k in enumerate(keeps):
        b.add_keep(k, f"c{i}", pct)
    return b


def test_fire_threshold_is_multiple_times_floor():
    p = A.AccumulatorPolicy(fire_multiple=2.5)
    assert abs(p.fire_threshold_pct(3.536) - 8.84) < 0.01
    assert abs(p.fire_threshold_pct(2.0) - 5.0) < 1e-9


def test_below_threshold_accumulates():
    p = A.AccumulatorPolicy(fire_multiple=2.5)
    b = _bundle(keeps=["m1"], pct=8.83)  # just under 2.5 * 3.536
    assert A.decide_after_keep(b, 3.536, p) is A.Decision.ACCUMULATE


def test_at_or_above_threshold_fires():
    p = A.AccumulatorPolicy(fire_multiple=2.5)
    b = _bundle(keeps=["m1", "m2"], pct=8.84)
    assert A.decide_after_keep(b, 3.536, p) is A.Decision.FIRE_SERVING
    b2 = _bundle(keeps=["m1"], pct=12.0)
    assert A.decide_after_keep(b2, 3.536, p) is A.Decision.FIRE_SERVING


def test_uncalibrated_floor_never_fires():
    # fail-closed: a gate that cannot judge is never spent (R23-43 grammar)
    p = A.AccumulatorPolicy()
    b = _bundle(keeps=["m1"], pct=99.0)
    assert A.decide_after_keep(b, None, p) is A.Decision.ACCUMULATE


def test_classify_promote_only_on_decisive_positive():
    assert A.classify_serving({"decisive": True, "effect": 0.07}) is A.Outcome.PROMOTE
    assert A.classify_serving({"decisive": True, "effect": -0.05}) is A.Outcome.DIVERGED
    assert A.classify_serving({"decisive": True, "effect": 0.0}) is A.Outcome.DIVERGED
    assert A.classify_serving({"decisive": False, "effect": 0.5}) is A.Outcome.DIVERGED
    assert A.classify_serving({"decisive": None, "effect": 0.5}) is A.Outcome.DIVERGED


def test_resolve_promote_advances_champion_to_tip():
    p = A.AccumulatorPolicy()
    b = _bundle(cor="cor0", keeps=["m1", "m2"], pct=9.1)  # tip = last keep's commit
    r = A.resolve(b, {"decisive": True, "effect": 0.07, "effect_pct": 7.0,
                      "noise_floor_pct": 3.536}, p)
    assert r["outcome"] is A.Outcome.PROMOTE
    assert r["new_champion_of_record"] == b.tip and b.tip == "c1"
    assert r["action"] is None


def test_default_divergence_is_hold_with_planner_evidence():
    # operator 2026-09-04: default HOLD, and the divergence must reach the planner as
    # evidence naming the bundled keeps so it can revert/revise one of them.
    p = A.AccumulatorPolicy()
    assert p.on_divergence is A.DivergenceAction.HOLD
    b = _bundle(cor="cor0", keeps=["m1", "m2", "m3"], pct=10.4)
    r = A.resolve(b, {"decisive": False, "effect": 0.005, "effect_pct": 0.5,
                      "noise_floor_pct": 3.536}, p)
    assert r["outcome"] is A.Outcome.DIVERGED
    assert r["new_champion_of_record"] == "cor0"          # HOLDS, does not move
    assert r["action"] is A.DivergenceAction.HOLD
    ev = r["planner_evidence"]
    assert ev["kind"] == "serving_divergence"
    assert ev["bundled_keeps"] == ["m1", "m2", "m3"]      # named, so the planner can revise one
    assert ev["compounded_bench_pct"] == 10.4
    assert "revert" in ev["hint"] or "revis" in ev["hint"]


def test_rollback_action_still_available_when_selected():
    p = A.AccumulatorPolicy(on_divergence=A.DivergenceAction.ROLLBACK)
    b = _bundle(cor="cor0", keeps=["m1"], pct=9.0)
    r = A.resolve(b, {"decisive": True, "effect": -0.02, "effect_pct": -2.0,
                      "noise_floor_pct": 3.536}, p)
    assert r["action"] is A.DivergenceAction.ROLLBACK
    assert r["new_champion_of_record"] == "cor0"


def test_promote_carries_no_planner_evidence():
    p = A.AccumulatorPolicy()
    b = _bundle(cor="cor0", keeps=["m1"], pct=9.1)
    r = A.resolve(b, {"decisive": True, "effect": 0.07, "effect_pct": 7.0,
                      "noise_floor_pct": 3.536}, p)
    assert "planner_evidence" not in r


def test_bundle_add_keep_tracks_tip_and_compounded():
    b = _bundle()
    b.add_keep("m1", "c1", 3.0)
    b.add_keep("m2", "c2", 6.5)
    assert b.tip == "c2" and b.keeps == ["m1", "m2"] and b.compounded_bench_pct == 6.5
    assert not _bundle().is_empty.__self__.keeps  # empty bundle is empty
