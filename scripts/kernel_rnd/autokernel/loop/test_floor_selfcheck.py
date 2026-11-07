"""AKX-AUTO-4: floor calibration self-check, replayed against the recorded Q38FN floors."""
import json
from pathlib import Path

import pytest

from autokernel.loop import floor_selfcheck as fs

FIX = json.loads((Path(__file__).parent / "fixtures" / "q38fn_floor_samples_20261005.json").read_text())
BY = {f["identity"]: f for f in FIX["floors"]}
DEGRADED = BY["c625b1ae"]      # 6.528%: the 2026-10-05 5.5-min degraded block
CLEAN_PRIORS = [BY["303da9d5"]["floor_pct"], BY["b270fd68"]["floor_pct"]]


def test_fixture_is_the_motivating_floor():
    assert DEGRADED["floor_pct"] == 6.528 and len(DEGRADED["anchor_samples"]) == 24


def test_replay_rejects_the_6528_block():
    a = fs.assess_floor(DEGRADED["anchor_samples"], DEGRADED["candidate_samples"],
                        DEGRADED["floor_pct"], CLEAN_PRIORS)
    assert a.verdict == fs.REJECT_DEGRADED_BLOCK
    (first, last), = a.degraded_blocks[:1]
    assert first == 8 and last >= 15
    assert a.spread_ratio > 5          # also anomalous vs the ~0.8% prior


def test_replay_spread_alone_flags_6528():
    a = fs.assess_spread(6.528, CLEAN_PRIORS)
    assert a.verdict == fs.RECALIBRATE_SPREAD and a.spread_ratio > 5


@pytest.mark.parametrize("ident", ["303da9d5", "b270fd68", "537fb4f3"])
def test_clean_floors_pass_block_screen(ident):
    row = BY[ident]
    assert fs.find_degraded_block(row["anchor_samples"], row["candidate_samples"])[0] == []


def test_guarded_remeasure_2483_is_not_spread_anomalous():
    # the accepted re-measure: 2.483% vs priors ~0.65/0.96% stays under 3x of the 1% clip
    assert fs.assess_spread(2.483, CLEAN_PRIORS).ok


def test_d69f_second_degraded_calibration_rejected():
    row = BY["d69f3cc2"]
    assert fs.assess_floor(row["anchor_samples"], row["candidate_samples"]).verdict == fs.REJECT_DEGRADED_BLOCK


def test_single_low_pair_is_not_a_block():
    a = [50.0] * 24
    c = [50.0] * 24
    a[10] = c[10] = 30.0
    assert fs.find_degraded_block(a, c)[0] == []


def test_two_low_pairs_not_a_block_three_are():
    a = [50.0] * 24
    for n, expect in ((2, 0), (3, 1)):
        x = list(a)
        for i in range(5, 5 + n):
            x[i] = 40.0
        assert len(fs.find_degraded_block(x, list(x))[0]) == expect


def test_degenerate_inputs_judge_nothing():
    assert fs.find_degraded_block([], [])[0] == []
    assert fs.find_degraded_block([0.0] * 5, [1.0] * 5)[0] == []
    with pytest.raises(ValueError):
        fs.find_degraded_block([1.0], [1.0, 2.0])
    assert fs.assess_spread(9.0, []).ok


def test_persistence_accepts_agreeing_second_calibration():
    assert fs.assess_spread(6.5, [0.7], refused_floor_pcts=[6.0]).persisted
    assert not fs.assess_spread(6.5, [0.7], refused_floor_pcts=[2.0]).ok


def test_prior_clip_keeps_quiet_host_wobble_ok():
    assert fs.assess_spread(0.4, [0.1]).ok
    assert not fs.assess_spread(3.5, [0.1]).ok


def test_lineage_helpers(tmp_path):
    for ident in ("303da9d5", "b270fd68", "c625b1ae"):
        row = BY[ident]
        d = tmp_path / ident
        d.mkdir()
        (d / "serving-floor.x.json").write_text(json.dumps(row))
    priors = fs.prior_floor_pcts(tmp_path)
    assert sorted(priors) == sorted(CLEAN_PRIORS)       # degraded 6.528 never becomes a baseline
    assert fs.prior_floor_pcts(tmp_path, exclude=tmp_path / "303da9d5" / "serving-floor.x.json") == [BY["b270fd68"]["floor_pct"]]
    a = fs.assess_spread(6.0, priors)
    fs.record_refusal(tmp_path, a)
    fs.record_refusal(tmp_path, fs.assess_floor(DEGRADED["anchor_samples"], DEGRADED["candidate_samples"], 6.5, priors))
    assert fs.refused_floor_pcts(tmp_path) == [6.0]     # block refusals cannot vouch


def test_check_before_write_refuses_then_accepts_persistent(tmp_path):
    lineage = tmp_path / "runtime-source-floors" / "cff9ad900700"
    for ident in ("303da9d5", "b270fd68"):
        (lineage / ident).mkdir(parents=True)
        (lineage / ident / "serving-floor.x.json").write_text(json.dumps(BY[ident]))
    with pytest.raises(fs.FloorSelfcheckRefused):           # degraded block
        fs.check_before_write(tmp_path, {**DEGRADED, "recipe_hash": "cff9ad900700"})
    assert fs.refused_floor_pcts(lineage) == []             # block refusals are not recorded
    clean = {**BY["537fb4f3"], "recipe_hash": "cff9ad900700"}
    high = {**clean, "floor_pct": 6.0}
    with pytest.raises(fs.FloorSelfcheckRefused):           # spread only: first time refused
        fs.check_before_write(tmp_path, high)
    assert fs.check_before_write(tmp_path, {**high, "floor_pct": 5.8}).persisted
    assert fs.check_before_write(tmp_path, clean).ok


def test_check_before_write_abstains_without_per_pair_samples(tmp_path):
    """A calibration row with no per-pair samples cannot be judged: the check abstains
    (returns None, writes nothing) instead of crashing the caller's floor write."""
    from autokernel.loop import floor_selfcheck as fs
    assert fs.check_before_write(tmp_path, {"floor_pct": 1.0, "recipe_hash": "r"}) is None
    assert fs.check_before_write(tmp_path, {"anchor_samples": [1.0], "candidate_samples": None,
                                            "recipe_hash": "r"}) is None
    assert not (tmp_path / "runtime-source-floors").exists()
