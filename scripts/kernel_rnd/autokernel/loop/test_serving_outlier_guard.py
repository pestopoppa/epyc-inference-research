"""AKX-FLOOR-REMEASURE-1 (A): the matched A/A floor's outlier guard.

WHY THIS EXISTS. `serving.calibrate_floor`'s matched branch had no trimming and no
degraded-host guard: one 5.5-minute degraded block during a 24-pair A/A inflated
Q38FN's floor from ~0.65% to 6.528%, and residency records explicitly say ordinary
load/PSI are "diagnostic, not blockers" -- nothing in the per-launch serving path would
ever refuse a degraded launch on its own. The threat model is honest-but-fallible, not
adversarial, so the guard's job is RE-MEASURE, bounded, never a silent drop: a flagged
pair gets one bounded re-measurement, and only a host that is STILL degraded after that
refuses outright.

SELECTION BIAS (review correction, Codex Astra, 2026-10-07): the first revision
excluded/replaced a pair because ITS OWN |effect| was a MAD outlier. That biases the
floor LOW and can admit real noise as a kernel effect -- a pair's own effect being
unusual is exactly what a floor exists to characterize. The guard now excludes or
replaces a pair ONLY on INDEPENDENT evidence that its OWN launch was compromised
(`measurement_validity.status == "invalid"`, or `clock_stable is False`), never on the
magnitude of what it measured. A MAD outlier with no such evidence is KEPT in the
floor untouched, reported as a diagnostic (`unexplained_outliers`), and the row is
marked `suspect` so an operator -- not the guard -- decides whether to file a
REMEASURE_REQUEST. `test_periodic_tail_pairs_with_no_independent_evidence_are_not_dropped`
below is the regression test for exactly the bug the first revision had.

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
    the residency `measurement_validity` channel alone -- the INDEPENDENT-evidence
    detector, isolated from the MAD diagnostic. `remeasure_clean=False` keeps flagging
    the same call positions even past the initial 48, to exercise the persistence
    refusal."""
    def measure(recipe, build, port, *, evidence, **kwargs):
        calls.append(1)
        idx = len(calls)
        degraded = idx in degraded_calls and (remeasure_clean is False or idx <= 48)
        evidence.append(_residency(idx, degraded=degraded))
        return 10.0 + idx / 1000.0
    return measure


def test_one_evidence_backed_pair_is_replaced_and_the_floor_comes_out_tight(monkeypatch):
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
    assert guard["schema"] == serving.OUTLIER_GUARD_SCHEMA_V2
    assert guard["semantics"] == serving.OUTLIER_GUARD_EVIDENCE_BASED
    assert guard["initial_evidence_pair_indices"] == [0]
    assert guard["replaced_pair_indices"] == [0]
    assert guard["final_evidence_pair_indices"] == []
    assert guard["excluded"][0]["reason"] == "cpu_lifecycle_invalid"
    assert guard["excluded"][0]["pair_index"] == 0
    assert guard["unexplained_outliers"] == []
    assert row["suspect"] is False
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


def test_too_many_evidence_backed_pairs_up_front_refuses_without_remeasuring(monkeypatch):
    """More than the bounded replacement budget (6) is not "a few compromised
    launches" -- refused outright, unmeasured, rather than spending 2x6 extra
    launches chasing it."""
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
    assert "cpu_lifecycle_invalid" in str(caught.value)
    assert len(calls) == 48  # no bounded replacement budget was spent chasing it


def test_persistent_degradation_still_refuses_after_bounded_remeasurement(monkeypatch):
    """The host does not recover DURING this calibration: the same pairs come back
    degraded after the one bounded re-measurement pass, so the guard refuses rather
    than seal a floor built on a host that never quieted down."""
    recipe, launch = _inputs()
    calls = []
    # Three pairs (<= the 6-pair budget) flagged up front, and the degraded condition
    # persists on their replacement launches too (idx > 48).
    degraded_calls = {1, 2, 11, 12, 21, 22}

    def measure(recipe, build, port, *, evidence, **kwargs):
        calls.append(1)
        idx = len(calls)
        degraded = idx in degraded_calls or idx > 48
        evidence.append(_residency(idx, degraded=degraded))
        return 10.0 + idx / 1000.0

    monkeypatch.setattr(serving, "_measure_once", measure)
    with pytest.raises(serving.HostDegradedDuringCalibration) as caught:
        serving.calibrate_floor(recipe, BUILD, samples=24, resolved_recipe=launch,
                                frozen_requests=REQUESTS, **MODE)
    assert "still carry independent evidence" in str(caught.value)
    assert len(calls) == 48 + 2 * 3  # the bounded replacement budget was spent, in vain


def test_gpu_clock_instability_is_also_independent_evidence(monkeypatch):
    """`clock_stable is False` -- the GPU's own sclk governor moving during the launch
    window -- is independent evidence just like a CPU lifecycle invalidity, and is
    named as its own `evidence_sources` entry."""
    recipe, launch = _inputs()
    calls = []

    def measure(recipe, build, port, *, evidence, **kwargs):
        calls.append(1)
        idx = len(calls)
        record = {"schema": serving.RESIDENCY_SCHEMA, "backend": "gpu",
                 "status": "proven", "window_start": float(idx), "window_end": float(idx),
                 "samples": 1, "clock_stable": idx not in (1, 2)}
        evidence.append(record)
        return 10.0 + idx / 1000.0

    monkeypatch.setattr(serving, "_measure_once", measure)
    row = serving.calibrate_floor(recipe, BUILD, samples=24, resolved_recipe=launch,
                                  frozen_requests=REQUESTS, **MODE)
    guard = row["outlier_guard"]
    assert guard["initial_evidence_pair_indices"] == [0]
    assert guard["excluded"][0]["reason"] == "gpu_clock_unstable"
    assert guard["replaced_pair_indices"] == [0]
    assert serving.OUTLIER_EVIDENCE_SOURCES == ("cpu_lifecycle_invalid", "gpu_clock_unstable")
    assert guard["evidence_sources"] == list(serving.OUTLIER_EVIDENCE_SOURCES)


def test_mad_outlier_with_no_independent_evidence_is_kept_not_replaced(monkeypatch):
    """THE CORE CORRECTION: a lone numeric spike with NO corroborating evidence is
    NEVER excluded or replaced on its own -- it is kept exactly as measured, reported
    as a diagnostic, and the row is marked `suspect` for an operator to act on."""
    recipe, launch = _inputs()
    calls = []

    def measure(recipe, build, port, *, evidence, **kwargs):
        calls.append(1)
        idx = len(calls)
        evidence.append(_residency(idx, degraded=False))  # no evidence, ever
        # Pair 3 (calls 7, 8) gets a lone, large relative spike; every other pair is a
        # tight, smooth series.
        if idx == 8:
            return 10.0 + idx / 1000.0 + 0.5
        return 10.0 + idx / 1000.0

    monkeypatch.setattr(serving, "_measure_once", measure)
    row = serving.calibrate_floor(recipe, BUILD, samples=24, resolved_recipe=launch,
                                  frozen_requests=REQUESTS, **MODE)
    guard = row["outlier_guard"]
    assert guard["initial_evidence_pair_indices"] == []
    assert guard["replaced_pair_indices"] == []
    assert guard["excluded"] == []
    assert len(calls) == 48  # NOT remeasured: no independent evidence, ever
    assert [item["pair_index"] for item in guard["unexplained_outliers"]] == [3]
    assert row["suspect"] is True
    # The spiked pair's actual measured values are exactly what went into the floor
    # (unordered: which label landed on which call is a randomized plan detail).
    assert sorted((row["anchor_samples"][3], row["candidate_samples"][3])) == sorted(
        (10.0 + 7 / 1000.0, 10.0 + 8 / 1000.0 + 0.5))


def test_periodic_tail_pairs_with_no_independent_evidence_are_not_dropped(monkeypatch):
    """Regression (post-Codex-Astra review): this is the ORIGINAL periodic-tail
    fixture (`10 + (len(calls) % 7) / 100`) that triggered the selection-bias bug in
    AKX-FLOOR-REMEASURE-1's first revision -- a handful of pairs straddle the modulo
    wraparound and read as clean MAD outliers, and the first revision silently
    replaced them with re-measured values, biasing the floor low. They must now
    survive completely untouched: no independent evidence exists for any of them, so
    none may be excluded or replaced, however unusual their own |effect| looks."""
    recipe, launch = _inputs()
    calls = []

    def measure(recipe, build, port, *, evidence, **kwargs):
        calls.append(1)
        idx = len(calls)
        evidence.append(_residency(idx, degraded=False))
        return 10 + (idx % 7) / 100

    monkeypatch.setattr(serving, "_measure_once", measure)
    row = serving.calibrate_floor(recipe, BUILD, samples=24, resolved_recipe=launch,
                                  frozen_requests=REQUESTS, **MODE)
    guard = row["outlier_guard"]

    # Nothing was excluded, replaced, or remeasured: no pair carries independent
    # evidence of a compromised launch, so NONE may be dropped on the strength of its
    # own |effect| alone -- the exact defect this regression test guards against.
    assert guard["initial_evidence_pair_indices"] == []
    assert guard["replaced_pair_indices"] == []
    assert guard["excluded"] == []
    assert len(calls) == 48

    # The periodic-tail discontinuity DOES still show up as a MAD diagnostic -- this
    # confirms the detector still runs, it just no longer acts unilaterally.
    assert guard["unexplained_outliers"]
    assert row["suspect"] is True

    # And every sample in the sealed floor is EXACTLY what the fixture measured --
    # reconstructed independently here (as unordered per-pair value sets, since which
    # label landed on which call is a randomized plan detail, not a measured fact), so
    # this assertion would fail if any pair had been silently replaced.
    expected = [10 + (i % 7) / 100 for i in range(1, 49)]
    expected_pairs = sorted(tuple(sorted((expected[2 * i], expected[2 * i + 1])))
                            for i in range(24))
    measured_pairs = sorted(tuple(sorted((a, c))) for a, c in
                            zip(row["anchor_samples"], row["candidate_samples"]))
    assert measured_pairs == expected_pairs
