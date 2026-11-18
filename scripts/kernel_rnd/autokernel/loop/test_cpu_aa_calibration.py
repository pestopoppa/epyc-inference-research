import random

import pytest

from . import cpu_aa_calibration as aa

HS = {"store_digest": "d" * 64, "cpu_list": "24-71", "threads": 48, "numa": "interleave=all",
      "recipe_env_digest": "e" * 16}


def samples(n, sd=0.01, seed=1):
    r = random.Random(seed)
    return [127.0 * (1 + r.gauss(0, sd)) for _ in range(n)], [127.0 * (1 + r.gauss(0, sd)) for _ in range(n)]


def build(n=24, label="quiet", busy=(3.0, 5.0), unit="process", **kw):
    a, c = samples(n)
    return aa.build_record(recipe="cpu-q38fn", recipe_hash="h", store_digest="d" * 64, unit=unit,
                           host_state=HS, contention_label=label, foreign_busy_p95=busy,
                           surfaces={"P4K": {"anchor": a, "candidate": c, "gating": True}},
                           serving_pairs=4, source_row="r", **kw)


def test_record_has_both_floors_interval_unit_and_hash():
    r = build(determinism={"kld_aa": 0, "row": "c15-cpu-fa-splitkv-xmodel-20261118"})
    s = r["surfaces"]["P4K"]
    assert r["unit"] == "process" and len(r["host_state_hash"]) == 64 and s["n"] == 24
    assert s["screen_floor"]["pairs"] == 5 and s["serving_floor"]["pairs"] == 4
    assert s["serving_floor"]["interval"]["low_pct"] <= s["serving_floor"]["interval"]["high_pct"]
    assert r["determinism"]["kld_aa"] == 0


def test_refusals():
    with pytest.raises(aa.CpuAACalibrationError, match="n=23"):
        build(n=23)
    with pytest.raises(aa.CpuAACalibrationError, match="unit"):
        build(unit="arm")
    with pytest.raises(aa.CpuAACalibrationError, match="quiet"):
        build(busy=(3.0, 40.0))
    with pytest.raises(aa.CpuAACalibrationError, match="kld_aa"):
        build(determinism={"kld_aa": 0.001, "row": "x"})
    assert build(label="co-tenant", busy=(40.0,))["contention_label"] == "co-tenant"


def test_host_state_hash_needs_state_and_is_stable():
    with pytest.raises(aa.CpuAACalibrationError):
        aa.host_state_hash({**HS, "cpu_list": ""})
    assert aa.host_state_hash(HS) == aa.host_state_hash(dict(reversed(list(HS.items()))))


def test_write(tmp_path):
    p = aa.write(tmp_path, build())
    assert p.name == "cpu-aa-calibration.cpu-q38fn.quiet.json"
