"""The clock policy: strict by default, a band only on ratified config and full proof."""
import json

import pytest

from . import claim, clock_policy as cp, residency, serving
from .test_resolved_recipe import BUILD
from .test_serving_degraded_block import MODE, REQUESTS, _inputs

#: Test fixture values only; the real band is the operator's to ratify.
BAND = {"schema": cp.SCHEMA, "mode": cp.BAND, "pinned_level": "high", "nominal_sclk_mhz": 1000,
        "droop_band_mhz": 50, "min_request_samples": 4, "max_sample_gap_s": 1.0,
        "ratification": "fixture"}


def policy(**overrides):
    return cp.parse(json.dumps({**BAND, **overrides}))


def record(mhz=(1000, 980, 970, 1000), load_mhz=600, level="high", **overrides):
    clocks = [load_mhz, load_mhz, *mhz]
    return {"backend": "gpu", "clock_stable": False, "window_start": 100.0, "window_end": 110.0,
            "request_start": 101.0, "request_end": 103.0,
            "clock_trace": {"t0": 100.0, "t_ms": [250 + 500 * i for i in range(len(clocks))],
                            "mhz": list(clocks), "level_changes": [[0, level]]}, **overrides}


def test_shipped_policy_is_the_ratified_band():
    active = cp.load()
    assert active.mode == cp.BAND and active.droop_band_mhz == 120


def test_strict_policy_keeps_the_historical_rule():
    active = cp.ClockPolicy()
    assert cp.launch_is_unstable(record(), active)
    assert not cp.launch_is_unstable(record(clock_stable=True), active)
    assert serving._independent_evidence(record(), policy=active) == "gpu_clock_unstable"


@pytest.mark.parametrize("bad", [{"droop_band_mhz": None}, {"droop_band_mhz": 0},
                                 {"droop_band_mhz": True}, {"pinned_level": "auto"},
                                 {"ratification": " "}, {"droop_band_mhz": 1000}])
def test_band_without_valid_ratified_values_is_refused(bad):
    with pytest.raises(cp.ClockPolicyError):
        policy(**bad)


def test_missing_band_key_and_unknown_mode_are_refused():
    with pytest.raises(cp.ClockPolicyError):
        cp.parse(json.dumps({"schema": cp.SCHEMA, "mode": cp.BAND}))
    with pytest.raises(cp.ClockPolicyError):
        cp.parse(json.dumps({"schema": cp.SCHEMA, "mode": "off"}))


def test_load_phase_clock_is_not_held_against_the_request_phase():
    verdict = cp.band_verdict(record(), policy())
    assert verdict["stable"] and verdict["request_sclk_min_mhz"] == 970


@pytest.mark.parametrize("mutate,reason", [
    (dict(mhz=(1000, 900, 1000, 1000)), "droop_below_band"),
    (dict(mhz=(1000, 1010, 1000, 1000)), "clock_above_nominal"),
    (dict(mhz=(1000, 0, 1000, 1000)), "unreadable_clock_in_request_phase"),
    (dict(level="auto"), "not_pinned_high"),
    (dict(window_end=102.0), "request_phase_not_covered"),
    (dict(clock_trace=None), "clock_trace_missing"),
])
def test_band_fails_closed(mutate, reason):
    verdict = cp.band_verdict(record(**mutate), policy())
    assert not verdict["stable"] and reason in verdict["reasons"]


def test_band_refuses_a_level_change_during_requests_and_sparse_sampling():
    moved = record()
    moved["clock_trace"]["level_changes"].append([2000, "auto"])
    assert "not_pinned_high" in cp.band_verdict(moved, policy())["reasons"]
    sparse = record()
    sparse["clock_trace"].update(t_ms=[250, 750, 1250, 1500, 1750, 1800], mhz=[600, 600, 1000, 1000, 1000, 1000])
    assert "sample_gap_exceeds_limit" in cp.band_verdict(sparse, policy())["reasons"]
    assert "too_few_request_samples" in cp.band_verdict(sparse, policy(min_request_samples=9))["reasons"]


def test_cpu_records_never_reach_the_band_path():
    assert not cp.launch_is_unstable({"backend": "cpu"}, policy())


def test_other_evidence_outranks_a_clean_band():
    invalid = record(measurement_validity={"status": "invalid"})
    assert serving._independent_evidence(invalid, policy=policy()) == "cpu_lifecycle_invalid"


def test_smoke_fails_closed():
    pinned = (lambda: "high", lambda: 1000)
    assert cp.smoke([record()], policy(), *pinned) == []
    assert cp.smoke([], policy(), *pinned) == ["no_gpu_launch_records"]
    assert cp.smoke([{"backend": "gpu"}], cp.ClockPolicy(), *pinned)
    assert cp.smoke([record()], cp.ClockPolicy(), *pinned)
    assert cp.smoke([record()], policy(), lambda: "auto", lambda: 1000)
    assert cp.smoke([record()], policy(), lambda: "high", lambda: 0)


def test_smoke_cli_refuses_a_malformed_policy(tmp_path, capsys):
    (tmp_path / "records.json").write_text(json.dumps([record()]))
    (tmp_path / "policy.json").write_text("{}")
    assert cp.main(["--records", str(tmp_path / "records.json"),
                    "--policy", str(tmp_path / "policy.json")]) == 1
    assert "REFUSED" in capsys.readouterr().err


def test_sampler_trace_separates_phases_and_stays_out_of_the_proof(monkeypatch):
    clocks = iter([900, 0, 1000])
    monkeypatch.setattr(residency, "sclk_mhz", lambda: next(clocks, 1000))
    monkeypatch.setattr(residency, "performance_level", lambda: "high")
    sampler = residency.Sampler(interval=0.001)
    with sampler:
        while sampler.samples < 3:
            pass
    trace = sampler.clock_trace
    assert trace["mhz"][:3] == [900, 0, 1000] and len(trace["t_ms"]) == len(trace["mhz"])
    assert trace["level_changes"][0][1] == "high" and len(trace["level_changes"]) == 1
    assert "clock_trace" not in sampler.proof


def test_residency_record_carries_trace_markers_and_strict_diagnostics():
    class Fake:
        proof = {"samples": 3, "vram_reads": 3, "resident": True, "clock_stable": False}
        clock_trace = {"t0": 100.0, "t_ms": [1500], "mhz": [970], "level_changes": [[0, "high"]],
                       "interval_s": 0.25}
    out = serving._residency_record(Fake(), window_start=100.0, window_end=110.0,
                                    request_start=101.0, request_end=102.0)
    assert out["clock_trace"]["mhz"] == [970]
    assert out["clock_phase_markers"] == {"window_start": 100.0, "request_start": 101.0,
                                          "request_end": 102.0, "window_end": 110.0}
    assert out["clock_stable"] is False


@pytest.fixture
def _lock(tmp_path, monkeypatch):
    monkeypatch.setattr(claim, "DEVICE_LOCK", tmp_path / "mi210.lock")


def test_refusal_keeps_the_windows_of_replaced_launches(monkeypatch, tmp_path, _lock):
    recipe, launch = _inputs()
    calls = []

    def measure(recipe_, build, port, *, evidence, **kwargs):
        calls.append(1)
        evidence.append({"backend": "gpu", "clock_stable": False, "call": len(calls)})
        return 50.0 + len(calls) * 1e-4
    monkeypatch.setattr(serving, "_measure_once", measure)
    monkeypatch.setattr(serving, "MAX_OUTLIER_REPLACEMENT_PAIRS", 24)
    with pytest.raises(serving.HostDegradedDuringCalibration, match="persistence"):
        serving.calibrate_floor(recipe, BUILD, samples=24, resolved_recipe=launch,
                                frozen_requests=REQUESTS, partial_store=tmp_path, **MODE)
    saved = json.loads(next(tmp_path.glob("*.refused-partial.json")).read_text())
    assert len(saved["superseded_residency"]) == 24
    first = saved["superseded_residency"][0]
    assert first["pair_index"] == 0 and first["reason"] == "gpu_clock_unstable"
    assert first["anchor"]["call"] <= 48 and first["candidate"]["call"] <= 48
    assert saved["anchor_residency"][0]["call"] > 48


def test_a_second_refusal_does_not_overwrite_the_first(monkeypatch, tmp_path, _lock):
    recipe, launch = _inputs()
    monkeypatch.setattr(serving, "_measure_once", lambda *a, evidence, **k: (
        evidence.append({"backend": "gpu", "clock_stable": False}), 50.0)[1])
    for _ in range(2):
        with pytest.raises(serving.HostDegradedDuringCalibration):
            serving.calibrate_floor(recipe, BUILD, samples=24, resolved_recipe=launch,
                                    frozen_requests=REQUESTS, partial_store=tmp_path, **MODE)
    assert len(list(tmp_path.glob("*.refused-partial.json"))) == 2
