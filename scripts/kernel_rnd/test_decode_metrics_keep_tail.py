import math, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rowlib import decode_metrics as dm, keep_tail as kt

FX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rowlib", "fixtures")


def test_step_ms_formula_and_degenerate():
    assert dm.step_ms({"predicted_ms": 1000.0, "predicted_n": 256, "draft_n_accepted": 156}) == 10.0
    assert dm.step_ms({"predicted_ms": 1000.0, "predicted_n": 10, "draft_n_accepted": 10}) is None
    assert dm.step_ms({"predicted_n": 10}) is None
    assert abs(dm.step_ms({"tok_s": 10.0, "predicted_n": 100, "draft_n_accepted": 50}) - 200.0) < 1e-9


def test_c14_r3_matches_recorded_verdict_numbers():
    # verdict.json: P2 ratio C13/C14 1.052, P3 0.973 (mean of per-step ms)
    cells = dm.cells_from_visits(dm.load_jsonl(f"{FX}/c14_r3_visits.jsonl"), "c13", "c14", ["P2", "P3"])
    assert abs(cells["P2"]["mean_ratio"] - 1.052) < 0.002
    assert abs(cells["P3"]["mean_ratio"] - 0.973) < 0.002


def test_c15_r2_matches_recorded_verdict_numbers():
    cells = dm.cells_from_visits(dm.load_jsonl(f"{FX}/c15_r2_visits.jsonl"), "c14", "c15", ["P2", "P3"])
    assert abs(cells["P2"]["mean_ratio"] - 1.055) < 0.002
    assert abs(cells["P3"]["mean_ratio"] - 0.914) < 0.002
    assert abs(dm.cells_geomean(cells, "median_ratio") - 0.998) < 0.002


def test_percentile_and_P():
    assert kt.percentile([1, 2, 3, 4, 5], 0.5) == 3
    assert kt.p95_inflation([100, 100, 100, 100]) == 0.0
    assert kt.p95_inflation([100]) is None
    p = kt.p95_inflation([100, 100, 100, 110], [100, 100, 100, 110])
    assert 0.0 < p <= 0.1000001


def test_floor_formula():
    assert kt.widened_floor(None) == 0.02
    assert kt.widened_floor(0.005) == 0.02
    assert abs(kt.widened_floor(0.039) - 0.0552) < 1e-4  # the box's 3.90% -> 5.52% example


def test_rule_c14_r3_and_c15_r2_rescore():
    c14 = dm.cells_from_visits(dm.load_jsonl(f"{FX}/c14_r3_visits.jsonl"), "c13", "c14", ["P2", "P3"])
    r = kt.apply_rule(c14)
    assert r["legacy"] == "FAIL" and r["F"] > 0.05 and r["widened"] == "PASS" and r["verdict_changed"]
    c15 = dm.cells_from_visits(dm.load_jsonl(f"{FX}/c15_r2_visits.jsonl"), "c14", "c15", ["P2", "P3"])
    r = kt.apply_rule(c15)
    assert r["legacy"] == "FAIL" and r["widened"] == "PASS"


def test_refuses_gain_inside_tail():
    cells = {"x": {"a": [100, 130, 100, 100], "b": [100, 100, 100, 130], "mean_ratio": 1.03, "median_ratio": 1.0}}
    r = kt.apply_rule(cells, min_gain=0.02)
    assert r["refusal"] and r["widened"] == "REFUSED_INSIDE_TAIL"


def test_rka_recs_step_ms_positive():
    rows = dm.load_jsonl(f"{FX}/rka_mmvf8_r2_recs.jsonl")
    xs = [dm.step_ms(r) for row in rows for r in row["recs"]]
    assert len(xs) == 128 and all(x and x > 0 for x in xs)
    unset = [dm.step_ms(r) for row in rows if row["karm"] == "UNSET" and row["dmax"] == 4 for r in row["recs"] if r["depth"] == "32k"]
    k8 = [dm.step_ms(r) for row in rows if row["karm"] == "K8" and row["dmax"] == 4 for r in row["recs"] if r["depth"] == "32k"]
    assert dm.median(unset) / dm.median(k8) > 1.05  # the recorded +13.6% knob gain
    ap = kt.apply_rule({"d4_32k": {"a": unset, "b": k8, "mean_ratio": dm.mean(unset) / dm.mean(k8)}}, min_gain=0.02)
    assert ap["P"] is not None and ap["F"] >= 0.02
