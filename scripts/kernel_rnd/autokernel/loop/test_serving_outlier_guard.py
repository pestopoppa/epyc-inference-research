"""AKX-FLOOR-REMEASURE-1 (A): the matched A/A floor's outlier guard.

WHY THIS EXISTS. `serving.calibrate_floor`'s matched branch had no trimming and no
degraded-host guard: one 5.5-minute degraded block during a 24-pair A/A inflated
Q38FN's floor from ~0.65% to 6.528%, and residency records explicitly say ordinary
load/PSI are "diagnostic, not blockers" -- nothing in the per-launch serving path would
ever refuse a degraded launch on its own. The threat model is honest-but-fallible, not
adversarial, so the guard's job is RE-MEASURE, bounded, never a silent drop: a flagged
pair gets one bounded re-measurement, and only a host that is STILL degraded after that
refuses outright.

Synthetic rates only: `serving._measure_once` is replaced, no server is launched.
"""
from pathlib import Path

import pytest

from . import claim, serving
from .test_resolved_recipe import BUILD, _resolve

MODE = {"instrument": serving.MATCHED_INSTRUMENT, "pairs": 5}
REQUESTS = (("p", b'{"prompt":[1],"n_predict":8}'),)


@pytest.fixture(autouse=True)
def _isolated_gpu_exclusion(tmp_path, monkeypatch):
    """Synthetic CPU windows must not wait on the host's real MI210 runner."""
    monkeypatch.setattr(claim, "DEVICE_LOCK", tmp_path / "mi210.lock")


def _inputs():
    recipe = serving.Recipe(name="outlier-guard", model="/fixture-model", device="none",
                            ngl=0, np=1)
    return recipe, _resolve(recipe, backend="cpu")


def _residency(idx, *, degraded):
    record = {"schema": serving.RESIDENCY_SCHEMA, "backend": "cpu",
             "status": "not_applicable", "window_start": float(idx),
             "window_end": float(idx), "samples": 0}
    if degraded:
        record["measurement_validity"] = {"status": "invalid",
                                          "failed_conditions": ["cpu_lifecycle"]}
    return record


def _measure_with_degraded_calls(calls, degraded_calls, *, remeasure_clean=True):
    """A monotone, MAD-quiet series (no genuine outlier in the numbers) that flags the
    CALLS in `degraded_calls` (1-indexed, matching `len(calls)` after the append) via
    the residency `measurement_validity` channel alone -- isolating the degraded-window
    detector from the MAD detector. `remeasure_clean=False` keeps flagging the SAME
    pair's calls even past the initial 48, to exercise the persistence refusal."""
    def measure(recipe, build, port, *, evidence, **kwargs):
        calls.append(1)
        idx = len(calls)
        degraded = idx in degraded_calls and (remeasure_clean is False or idx <= 48)
        evidence.append(_residency(idx, degraded=degraded))
        return 10.0 + idx / 1000.0
    return measure


def test_one_degraded_pair_is_replaced_and_the_floor_comes_out_tight(monkeypatch):
    """A single pair flagged via a recorded degraded residency window (not a numeric
    spike: the series itself is calm) is re-measured once, and the resulting floor is
    exactly what a clean 24-pair run would have produced -- no evidence of the
    degradation survives into the sealed row except the guard's own audit trail."""
    recipe, launch = _inputs()
    calls = []
    # Pair 0 is calls 1 and 2 (the plan iterates pair-by-pair, two calls each).
    monkeypatch.setattr(serving, "_measure_once",
                        _measure_with_degraded_calls(calls, {1, 2}))
    row = serving.calibrate_floor(recipe, BUILD, samples=24, resolved_recipe=launch,
                                  frozen_requests=REQUESTS, **MODE)

    guard = row["outlier_guard"]
    assert guard["applied"] is True
    assert guard["initial_outlier_pair_indices"] == [0]
    assert guard["replaced_pair_indices"] == [0]
    assert guard["final_outlier_pair_indices"] == []
    assert guard["excluded"][0]["reason"] == "degraded_residency_window"
    assert guard["excluded"][0]["pair_index"] == 0
    assert len(calls) == 48 + 2  # the original 48, plus one bounded re-measurement pair

    # Tight: every surviving/replaced value sits on the same smooth, low-dispersion
    # series, so the floor is nowhere near the ~6.528% a genuinely contaminated A/A
    # produced in the wild (origin incident: AKX-FLOOR-REMEASURE-1).
    assert row["floor_pct"] < 1.0

    # The replaced pair's residency record is the CLEAN re-measurement, not the
    # degraded original -- a reader of the sealed row never sees the bad launch as if
    # it were the one that defined the floor.
    assert row["anchor_residency"][0].get("measurement_validity") != {
        "status": "invalid", "failed_conditions": ["cpu_lifecycle"]}
    assert row["candidate_residency"][0].get("measurement_validity") != {
        "status": "invalid", "failed_conditions": ["cpu_lifecycle"]}


def test_too_many_outlier_pairs_up_front_refuses_without_remeasuring(monkeypatch):
    """More than the bounded replacement budget (6) is not "a few noisy launches" --
    refused outright, unmeasured, rather than spending 2x6 extra launches chasing it."""
    recipe, launch = _inputs()
    calls = []
    # Seven pairs (14 calls) flagged -- one more than MAX_OUTLIER_REPLACEMENT_PAIRS.
    degraded_calls = {i for pair in range(7) for i in (2 * pair + 1, 2 * pair + 2)}
    monkeypatch.setattr(serving, "_measure_once",
                        _measure_with_degraded_calls(calls, degraded_calls))
    with pytest.raises(serving.HostDegradedDuringCalibration) as caught:
        serving.calibrate_floor(recipe, BUILD, samples=24, resolved_recipe=launch,
                                frozen_requests=REQUESTS, **MODE)
    assert "host degraded during floor calibration" in str(caught.value)
    assert len(calls) == 48  # no bounded replacement budget was spent chasing it


def test_persistent_degradation_still_refuses_after_bounded_remeasurement(monkeypatch):
    """The host does not recover DURING this calibration: the same pairs come back
    degraded after the one bounded re-measurement pass, so the guard refuses rather
    than seal a floor built on a host that never quieted down."""
    recipe, launch = _inputs()
    calls = []
    # Three pairs (<= the 6-pair budget) flagged up front, but `remeasure_clean=False`
    # keeps the SAME call positions flagged even on their replacement launches.
    degraded_calls = {1, 2, 11, 12, 21, 22}

    def measure(recipe, build, port, *, evidence, **kwargs):
        calls.append(1)
        idx = len(calls)
        # All three flagged pairs are persistently degraded: their re-measurement
        # launches (idx > 48) come back degraded too, exactly like the original ones.
        degraded = idx in degraded_calls or idx > 48
        evidence.append(_residency(idx, degraded=degraded))
        return 10.0 + idx / 1000.0

    monkeypatch.setattr(serving, "_measure_once", measure)
    with pytest.raises(serving.HostDegradedDuringCalibration) as caught:
        serving.calibrate_floor(recipe, BUILD, samples=24, resolved_recipe=launch,
                                frozen_requests=REQUESTS, **MODE)
    assert "remain robust outliers after re-measuring" in str(caught.value)
    assert len(calls) == 48 + 2 * 3  # the bounded replacement budget was spent, in vain


def test_mad_outlier_pair_is_also_flagged_without_any_residency_evidence(monkeypatch):
    """The MAD channel catches a numeric spike even with no residency evidence at all
    (the GPU backend, or no CPU observer) -- the two detectors are independent."""
    recipe, launch = _inputs()
    calls = []

    def measure(recipe, build, port, *, evidence, **kwargs):
        calls.append(1)
        idx = len(calls)
        evidence.append({"schema": serving.RESIDENCY_SCHEMA, "backend": "cpu",
                         "status": "not_applicable", "window_start": float(idx),
                         "window_end": float(idx), "samples": 0})
        # Pair 3 (calls 7, 8) gets a lone, large relative spike; every other pair is
        # a tight, smooth series.
        if idx in (7, 8) and idx <= 48:
            return 10.0 + idx / 1000.0 + (0.5 if idx == 8 else 0.0)
        return 10.0 + idx / 1000.0

    monkeypatch.setattr(serving, "_measure_once", measure)
    row = serving.calibrate_floor(recipe, BUILD, samples=24, resolved_recipe=launch,
                                  frozen_requests=REQUESTS, **MODE)
    guard = row["outlier_guard"]
    assert guard["initial_outlier_pair_indices"] == [3]
    assert guard["excluded"][0]["reason"] == "mad_outlier"
    assert guard["replaced_pair_indices"] == [3]
    assert guard["final_outlier_pair_indices"] == []
