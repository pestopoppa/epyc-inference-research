"""variance_keep: known-answer synthetic arms (LABELLED SYNTHETIC; no hardware claim)."""
import json
import random
from pathlib import Path

import pytest

from . import run, variance_keep as vk

FLOOR = 1.0  # percent, synthetic session-unit floor


def arms(n, a_sd, c_sd, c_shift=0.0, seed=1):
    """Synthetic paired units: anchor ~ 100*(1+N(0,a_sd)), candidate ~ 100*(1+c_shift)*(1+N(0,c_sd))."""
    rng = random.Random(seed)
    return [{"anchor": 100.0 * (1 + rng.gauss(0, a_sd)),
             "candidate": 100.0 * (1 + c_shift) * (1 + rng.gauss(0, c_sd))} for _ in range(n)]


def ev(units, **kw):
    kw.setdefault("floor_pct", FLOOR)
    kw.setdefault("floor_unit", "session")
    return vk.evaluate(units, **kw)


def test_clearly_tighter_passes():
    v = ev(arms(30, 0.02, 0.002))
    assert v["verdict"] == vk.KEEP_VARIANCE and v["reasons"] == []
    assert v["log_ratio_upper95"] < 0 and v["effect_pct_lcb95"] > -FLOOR
    assert v["multiplicity"] == vk.MULTIPLICITY


def test_equal_spread_fails():
    v = ev(arms(30, 0.02, 0.02, seed=7))
    assert v["verdict"] == vk.REFUSE and vk.R_NOT_TIGHTER in v["reasons"]


def test_n13_refused_n14_evaluated():
    v = ev(arms(13, 0.02, 0.002))
    assert v["verdict"] == vk.REFUSE and v["reasons"] == [vk.R_INSUFFICIENT_N]
    assert "log_ratio" not in v
    assert "log_ratio" in ev(arms(14, 0.02, 0.002))


def test_throughput_regression_refused_even_if_tighter():
    v = ev(arms(30, 0.02, 0.002, c_shift=-0.10))
    assert v["verdict"] == vk.REFUSE and v["reasons"] == [vk.R_THROUGHPUT]
    assert v["log_ratio_upper95"] < 0


def test_floor_is_an_input_not_hardcoded():
    units = arms(30, 0.02, 0.002, c_shift=-0.03)
    assert ev(units, floor_pct=1.0)["verdict"] == vk.REFUSE
    assert ev(units, floor_pct=20.0)["verdict"] == vk.KEEP_VARIANCE


def test_floor_defects_refused():
    u = arms(30, 0.02, 0.002)
    assert vk.R_FLOOR_UNIT in ev(u, floor_unit="process")["reasons"]
    for bad in (0, -1, None, float("nan"), True):
        assert vk.R_BAD_FLOOR in ev(u, floor_pct=bad)["reasons"]


def test_deterministic_same_seed_and_seed_matters():
    u = arms(20, 0.02, 0.01)
    assert ev(u) == ev(u)
    assert ev(u, seed=1)["log_ratio_upper95"] != ev(u, seed=2)["log_ratio_upper95"]
    assert ev(u, draws=9999)["reasons"] == [vk.R_BAD_INPUT]


def test_bad_input_and_degenerate_anchor_refuse_not_raise():
    assert ev([{"anchor": 1}] * 20)["reasons"] == [vk.R_BAD_INPUT]
    assert ev([{"anchor": -1.0, "candidate": 1.0}] * 20)["reasons"] == [vk.R_BAD_INPUT]
    flat = [{"anchor": 100.0, "candidate": 100.0 + i} for i in range(20)]
    assert ev(flat)["reasons"] == [vk.R_DEGENERATE_ANCHOR]


def test_semideviation_known_answer():
    # median 100; only 90 is below: sqrt(mean([0.1^2,0,0,0,0])) = 0.1/sqrt(5)
    assert vk.semideviation([90, 100, 100, 110, 120]) == pytest.approx(0.1 / 5 ** 0.5)
    assert vk.semideviation([100, 100, 100]) == 0.0


def _write(tmp_path, units, cid="c1", floor_unit="session"):
    p = tmp_path / "vin.json"
    p.write_text(json.dumps({"schema": vk.INPUT_SCHEMA, "candidate_id": cid, "floor_pct": FLOOR,
                             "floor_unit": floor_unit, "units": units}))
    return p


def test_keep_record_fields_writes_spread_and_retains(tmp_path):
    fp = tmp_path / "keep-frontier.json"
    r1 = vk.keep_record_fields(_write(tmp_path, arms(30, 0.02, 0.002), "c1"), fp)
    assert r1["variance_keep"]["verdict"] == vk.KEEP_VARIANCE
    k = r1["keep_frontier"]
    assert k["schema"] == "epyc.autokernel.keep_frontier.v1" and k["admitted"] and k["spread"] > 0
    r2 = vk.keep_record_fields(_write(tmp_path, arms(30, 0.05, 0.05, c_shift=0.1, seed=3), "c2"), fp)
    assert r2["keep_frontier"]["frontier_size"] == 2  # faster-noisier and slower-tighter coexist
    assert len(json.loads(fp.read_text())["points"]) == 2


def test_keep_record_fields_never_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    r = vk.keep_record_fields(bad, tmp_path / "f.json")
    assert r["variance_keep"]["verdict"] == vk.REFUSE and "keep_frontier" not in r
    r = vk.keep_record_fields(tmp_path / "missing.json", tmp_path / "f.json")
    assert r["variance_keep"]["reasons"] == [vk.R_BAD_INPUT]
    short = vk.keep_record_fields(_write(tmp_path, arms(5, 0.02, 0.002)), tmp_path / "f.json")
    assert short["variance_keep"]["reasons"] == [vk.R_INSUFFICIENT_N] and "keep_frontier" not in short


def test_default_off_identity_in_run():
    """No --variance-input: the flag defaults to None and every run.py wiring site is gated on
    it, so the keep path executes no new statement and the attempt row gains no key."""
    src = Path(run.__file__).read_text()
    assert '"--variance-input", type=Path, default=None' in Path(run.__file__).with_name("run_args.py").read_text()
    assert src.count("variance_keep.keep_record_fields(") == 1
    assert 'if args.variance_input is not None:\n                variance_keep_by_head[head]' in src
    # attempt only gains keys for heads that were recorded; an empty dict is a no-op
    assert 'champion_head in variance_keep_by_head:\n                attempt.update(' in src
    attempt = {"status": "kept"}
    before = json.dumps(attempt, sort_keys=True)
    by_head: dict = {}
    if "h" in by_head:
        attempt.update(by_head.pop("h"))
    assert json.dumps(attempt, sort_keys=True) == before
