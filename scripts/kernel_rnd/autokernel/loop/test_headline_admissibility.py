from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile

from unittest import mock

from autokernel.loop import bench, headline_admissibility as H, production, serving
from autokernel.loop.test_production import _comparison, _refresh, _warm_baseline


def test_planned_n_is_reproducible_and_explicit():
    assert H.MIN_LAUNCHES_PER_ARM == 14
    assert H.MIN_LAUNCHES == 28
    assert H.contract() == {
        "unit": "process", "n": 28, "n_per_arm": 14, "mde_pct": 3.0,
        "between_launch_sd_pct": 2.793, "alpha": 0.05, "power": 0.8,
        "sidedness": "two-sided",
        "planning_method": "normal_two_independent_samples_equal_n",
        "ci_level": 0.95,
        "ci_method": "paired_bootstrap_median_ratio_95_seed_20260915_draws_20000"}


def test_under_n_comparison_refuses_and_writes_no_headline():
    with tempfile.TemporaryDirectory() as tmp:
        store = Path(tmp)
        base = _warm_baseline(store)
        measured = _comparison()
        thin = replace(measured, anchor_samples=measured.anchor_samples[:13],
                       candidate_samples=measured.candidate_samples[:13], pairs=13)
        result = _refresh(store, base, compare=lambda _a, _b: thin)
        assert not result.published
        assert "REFUSED" in result.reason
        assert "below N=28" in result.reason
        assert not (store / production.FILENAME).exists()


def test_n_launches_publish_session_unit_ci_and_full_contract():
    with tempfile.TemporaryDirectory() as tmp:
        store = Path(tmp)
        base = _warm_baseline(store)
        measured = _comparison()
        enough = replace(measured, anchor_samples=measured.anchor_samples[:14],
                         candidate_samples=measured.candidate_samples[:14], pairs=14)
        result = _refresh(store, base, compare=lambda _a, _b: enough)
        assert result.published, result.reason
        body = json.loads((store / production.FILENAME).read_text())
        assert body["launches"] == 28
        assert body["headline_admissibility"] == H.contract()
        assert body["confidence_interval"]["unit"] == bench.FLOOR_UNIT == "process"
        assert body["confidence_interval"]["method"] == H.CI_METHOD
        assert body["confidence_interval"]["lower_effect_fraction"] \
            < body["confidence_interval"]["upper_effect_fraction"]


def test_floor_producers_name_the_same_headline_contract():
    source = (Path(__file__).parents[3] / "benchmark" / "autokernel_aa_campaign.py").read_text()
    assert '"headline_admissibility": headline_admissibility.contract()' in source
    with mock.patch.object(serving, "_measure_once", side_effect=[100.0] * 24):
        row = serving.calibrate_floor(
            serving.Recipe(name="floor-contract", model="/m.gguf"),
            Path("/build"), samples=24)
    assert row["headline_admissibility"] == H.contract()


def test_the_unpaired_interval_resamples_each_arm_on_its_own():
    """Carry-over: the baseline arm was recorded in another session, so there are no
    A/B blocks. BROKEN READS: the paired method's name on an unpaired interval, or an
    interval that refuses arms of unequal length (the record and the champion need
    not have the same n)."""
    measured = _comparison()
    baseline = list(measured.anchor_samples)
    champion = list(measured.candidate_samples)[:16]
    ci = H.unpaired_confidence_interval(baseline, champion)
    assert ci["method"] == H.UNPAIRED_CI_METHOD != H.CI_METHOD
    assert ci["unit"] == "process"
    assert "drift" in ci["excludes"]
    import statistics
    effect = statistics.median(champion) / statistics.median(baseline) - 1.0
    assert ci["lower_effect_fraction"] <= effect <= ci["upper_effect_fraction"]
    assert H.unpaired_confidence_interval(baseline, champion) == ci


def test_the_unpaired_interval_keeps_the_per_arm_minimum():
    measured = _comparison()
    try:
        H.unpaired_confidence_interval(measured.anchor_samples,
                                       measured.candidate_samples[:13])
    except H.HeadlineInadmissible:
        return
    raise AssertionError("13 champion launches must not support a headline")
