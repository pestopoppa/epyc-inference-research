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
