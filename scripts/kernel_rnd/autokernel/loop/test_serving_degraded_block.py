"""AKX-AUTO-4: `serving.calibrate_floor` refuses a contiguous degraded block that carries
no per-launch evidence (a level shift both arms share). Synthetic rates, no server."""
import pytest

from . import claim, serving
from .test_resolved_recipe import BUILD, _resolve
from .test_serving_outlier_guard import MODE, REQUESTS, _inputs, _residency


@pytest.fixture(autouse=True)
def _isolated_gpu_exclusion(tmp_path, monkeypatch):
    monkeypatch.setattr(claim, "DEVICE_LOCK", tmp_path / "mi210.lock")


def _series(monkeypatch, degraded_calls):
    calls = []

    def measure(recipe, build, port, *, evidence, **kwargs):
        calls.append(1)
        evidence.append(_residency(len(calls), degraded=False))
        return 50.0 * (0.8 if len(calls) in degraded_calls else 1.0) + len(calls) * 1e-4
    monkeypatch.setattr(serving, "_measure_once", measure)


def test_degraded_plateau_without_evidence_is_refused(monkeypatch):
    recipe, launch = _inputs()
    _series(monkeypatch, set(range(17, 41)))          # pairs 8..19
    with pytest.raises(serving.HostDegradedDuringCalibration, match="AKX-AUTO-4"):
        serving.calibrate_floor(recipe, BUILD, samples=24, resolved_recipe=launch,
                                frozen_requests=REQUESTS, **MODE)


def test_two_degraded_pairs_are_not_a_block(monkeypatch):
    recipe, launch = _inputs()
    _series(monkeypatch, set(range(17, 21)))          # pairs 8..9 only
    row = serving.calibrate_floor(recipe, BUILD, samples=24, resolved_recipe=launch,
                                  frozen_requests=REQUESTS, **MODE)
    assert row["calibration_pairs"] == 24


def test_refusal_persists_partial_evidence_and_no_floor(monkeypatch, tmp_path):
    """A refused calibration leaves `*.refused-partial.json` and NO floor record."""
    import json
    recipe, launch = _inputs()
    _series(monkeypatch, set(range(17, 41)))
    with pytest.raises(serving.HostDegradedDuringCalibration, match="AKX-AUTO-4"):
        serving.calibrate_floor(recipe, BUILD, samples=24, resolved_recipe=launch,
                                frozen_requests=REQUESTS, partial_store=tmp_path, **MODE)
    partials = list(tmp_path.glob("*.refused-partial.json"))
    assert len(partials) == 1
    record = json.loads(partials[0].read_text())
    assert record["is_floor"] is False and "floor_pct" not in record
    assert len(record["anchor_samples"]) == len(record["candidate_samples"]) == 24
    assert len(record["anchor_residency"]) == 24
    assert record["reference_pair_indices"] == list(range(24))
    assert record["wall_clock"]["refused_at"] >= record["wall_clock"]["calibration_started_at"]
    # never resolvable as a floor
    floor_file = serving.floor_path(tmp_path, recipe, frozen_requests=REQUESTS, **MODE)
    assert not floor_file.exists() and partials[0] != floor_file
    assert list(tmp_path.glob("serving-floor.*.json")) == partials
