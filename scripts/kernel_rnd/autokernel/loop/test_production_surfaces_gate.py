"""AKX-ALL-LB2.1: production_surfaces.assess on the 27B PROMOTE path (accumulate.resolve). Pure logic."""
from autokernel.loop import accumulate as A, production_surfaces as PS

ROW = {"decisive": True, "effect": 0.07, "effect_pct": 7.0, "noise_floor_pct": 1.0,
       "recipe": "q27b-gpu-dflash2-np1"}
ALL = {"n16": 0.5, "n32": 0.2, "ragged_1_8_8_8": -0.4}


def _bundle():
    b = A.Bundle(champion_of_record="cor0", tip="cor0")
    b.add_keep("m1", "c0", 9.0)
    return b


def test_default_off_labels_but_promotes(monkeypatch):
    monkeypatch.delenv(PS.GATE_ENV, raising=False)
    assert PS.GATE_DEFAULT is False
    r = A.resolve(_bundle(), dict(ROW), A.AccumulatorPolicy())
    assert r["outcome"] is A.Outcome.PROMOTE
    assert r["production_surfaces"]["label"] == PS.NOT_YET_TRANSFERABLE
    assert r["production_surfaces"]["gate_enabled"] is False


def test_gate_on_refuses_anchor_only_keep(monkeypatch):
    monkeypatch.setenv(PS.GATE_ENV, "1")
    r = A.resolve(_bundle(), dict(ROW), A.AccumulatorPolicy())
    assert r["outcome"] is A.Outcome.DIVERGED and r["new_champion_of_record"] == "cor0"
    assert "not-yet-transferable" in r["reason"] and "n16" in r["reason"]


def test_gate_on_refuses_regression_and_promotes_transferable(monkeypatch):
    monkeypatch.setenv(PS.GATE_ENV, "1")
    bad = dict(ROW, production_surfaces=dict(ALL, n32=-1.5))
    r = A.resolve(_bundle(), bad, A.AccumulatorPolicy())
    assert r["outcome"] is A.Outcome.DIVERGED and r["production_surfaces"]["regressed"] == ["n32"]
    r = A.resolve(_bundle(), dict(ROW, production_surfaces=ALL), A.AccumulatorPolicy())
    assert r["outcome"] is A.Outcome.PROMOTE and r["production_surfaces"]["label"] == PS.TRANSFERABLE


def test_other_families_untouched(monkeypatch):
    monkeypatch.setenv(PS.GATE_ENV, "1")
    r = A.resolve(_bundle(), dict(ROW, recipe="a3b-cpu-mtp"), A.AccumulatorPolicy())
    assert r["outcome"] is A.Outcome.PROMOTE and r["production_surfaces"] is None
    r = A.resolve(_bundle(), {k: v for k, v in ROW.items() if k != "recipe"}, A.AccumulatorPolicy())
    assert r["outcome"] is A.Outcome.PROMOTE
