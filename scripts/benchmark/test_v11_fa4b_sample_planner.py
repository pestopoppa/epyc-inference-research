import copy

import pytest

from scripts.benchmark import v11_fa4b_sample_planner as planner


def valid_report():
    cell = {"n_decidable_pairs": 200, "n_scored_observed_per_arm": {"ON": 200, "OFF": 200},
            "mcnemar": {"on_only": 0, "off_only": 0}, "delta_on_minus_off": 0.0,
            "termination": {arm: {"score": {"truncated": 100, "unknown": 0}} for arm in ("ON", "OFF")}}
    return {"schema": "v11-fa4b-quality.m2", "cells": {c: copy.deepcopy(cell) for c in planner.CELLS}}


def test_zero_discordance_point_scenario_is_not_a_quality_claim():
    result = planner.analyze(valid_report(), [200], 4, 42)
    assert result["cells"]["q8_0:off"]["size_sensitivity"][0]["fraction"] == 1
    assert "Zero observed discordance does not prove zero true discordance" in result["scope"]
    assert "Uncapped quality and runtime are unknown" in result["scope"]


def test_seed_reproduces_simulation_and_interval_is_bounded():
    report = valid_report()
    for cell in report["cells"].values():
        cell["mcnemar"] = {"on_only": 3, "off_only": 1}
        cell["delta_on_minus_off"] = 0.01
    a = planner.analyze(report, [200], 8, 42)
    assert a == planner.analyze(report, [200], 8, 42)
    low, high = a["cells"]["f16:on"]["size_sensitivity"][0]["monte_carlo_wilson_95"]
    assert 0 <= low <= high <= 1


def test_refuses_incomplete_inconsistent_and_unknown_inputs():
    report = valid_report()
    report["cells"]["q8_0:off"]["n_decidable_pairs"] = 199
    with pytest.raises(ValueError, match="200-pair"):
        planner.analyze(report, [200], 2, 42)
    report = valid_report()
    report["cells"]["q8_0:off"]["delta_on_minus_off"] = 0.03
    with pytest.raises(ValueError, match="delta disagree"):
        planner.analyze(report, [200], 2, 42)
    report = valid_report()
    del report["cells"]["q8_0:off"]["termination"]["ON"]["score"]["unknown"]
    with pytest.raises(ValueError, match="cap accounting"):
        planner.analyze(report, [200], 2, 42)
