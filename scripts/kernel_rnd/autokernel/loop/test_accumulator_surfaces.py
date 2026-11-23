import json

import pytest

from . import accumulate
from . import accumulator_surfaces as acs
from . import cpu_keep_bundle as ckb

POLICY = accumulate.AccumulatorPolicy()


def bundle(tmp_path, name, keeps=3, pct=6.0, validity="current_snapshot", since=2):
    p = tmp_path / name
    p.write_text(json.dumps({"keeps": [f"k{i}" for i in range(keeps)], "compounded_bench_pct": pct,
                             "measurement_validity": validity, "keeps_since_serving_gate": since}))
    return p


def sec(tmp_path, **kw):
    floor = kw.pop("floor", 2.0)
    return acs.surface_section("gpu", bundle(tmp_path, "b.json", **kw), floor_pct=floor)


def test_no_floor_is_cannot_decide_never_a_guess(tmp_path):
    s = sec(tmp_path, floor=None, since=99)
    assert s["fire"]["decision"] == "cannot_decide"


def test_measured_figure_clearing_threshold_fires(tmp_path):
    s = sec(tmp_path, pct=6.0, floor=2.0)          # 2.5 x 2.0 = 5.0
    assert (s["fire"]["decision"], s["fire"]["trigger"]) == ("fire", "threshold")
    assert s["figure"]["kind"] == "measured"


def test_estimate_can_only_fire_on_cadence(tmp_path):
    s = sec(tmp_path, pct=60.0, validity="stale_external_tip_advance", since=2)
    assert s["figure"]["kind"] == "estimate" and s["figure"]["label"].startswith("ESTIMATE")
    assert s["fire"]["decision"] == "accumulate"
    s = sec(tmp_path, pct=0.1, validity="stale_external_tip_advance", since=POLICY.every_keeps)
    assert (s["fire"]["decision"], s["fire"]["trigger"]) == ("fire", "cadence")


def test_both_and_stale_baseline_and_empty(tmp_path):
    assert sec(tmp_path, pct=9.0, since=POLICY.every_keeps)["fire"]["trigger"] == "both"
    assert sec(tmp_path, validity="stale_baseline")["fire"]["decision"] == "cannot_decide"
    s = sec(tmp_path, keeps=0, pct=0.0, since=0)
    assert s["figure"]["kind"] == "none" and s["fire"]["decision"] == "accumulate"


def test_cpu_fold_estimates_are_always_labelled_estimate(tmp_path):
    keep = dict(keep_id="k", effect_pct=4.0, magnitude="provisional", contention_label="unproven",
                statistic="s", floor_pct=5.0, floor_unit="process")
    ckb.write(tmp_path / "cpu", ckb.build(recipe="r.h1", recipe_hash="h", champion_of_record="c" * 40,
                                           tip="t" * 40, keeps=[keep]))
    s = acs.surface_section("cpu", bundle(tmp_path, "c.json", keeps=0, pct=0.0, since=0),
                            floor_pct=3.0, cpu_fold_dir=tmp_path / "cpu")
    (fe,) = s["fold_estimates"]
    assert fe["label"].startswith("ESTIMATE") and fe["provisional_keeps"] == 1 and fe["recipe"] == "r.h1"
    # the fold estimate rides along, it never feeds the decision
    assert s["fire"]["decision"] == "accumulate"


def test_publish_roundtrip_and_cli(tmp_path):
    b = bundle(tmp_path, "g.json")
    assert acs.main(["--surface", f"gpu={b}:2.0", "--out", str(tmp_path / "o")]) == 0
    rec = json.loads((tmp_path / "o" / acs.FILENAME).read_text())
    assert rec["schema"] == acs.SCHEMA and rec["surfaces"][0]["fire"]["decision"] == "fire"
    assert not list((tmp_path / "o").glob(".accumulator-surfaces-*"))
