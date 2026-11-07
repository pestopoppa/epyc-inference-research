"""SL-2: the speculative keep gate decides on target steps/s against its OWN steps/s floor,
guarded against acceptance regression. Offline: no server is launched."""
import dataclasses
from pathlib import Path

import pytest

from autokernel.loop import serving

SPEC = serving.Recipe(
    name="t-spec", model="/m/target.gguf",
    spec_decode={"type": "draft-dflash", "drafter": "/m/draft.gguf", "ngld": 99, "draft_n_max": 8},
    np=2, ctx=16384)
PLAIN = serving.Recipe(name="t-plain", model="/m/target.gguf", np=2, ctx=16384)


def _launch(steps_per_s, accepted_per_step, *, tok_s=None, counters=True, ms=1000.0):
    """One launch's aggregate built through the real slot/aggregate code."""
    slots = []
    for _ in range(2):
        steps = int(round(steps_per_s / 2 * ms / 1000.0))
        accepted = int(round(accepted_per_step * steps))
        timings = {"predicted_n": steps + accepted, "predicted_ms": ms}
        if counters:
            timings.update(draft_n=accepted * 2 + 1, draft_n_accepted=accepted)
        slots.append(serving._target_steps_slot(timings))
    return serving._target_steps_aggregate(slots, tok_s or 1.0)


def _patch(monkeypatch, arms):
    """arms: build-name -> list of (tok_s, launch) consumed in order."""
    queues = {k: list(v) for k, v in arms.items()}

    def measure(recipe, build, port, *, evidence=None, target_sample_steps=None, **_):
        tok_s, launch = queues[Path(build).name].pop(0)
        if evidence is not None:
            evidence.append({})
        if target_sample_steps is not None:
            target_sample_steps.append(launch)
        return tok_s
    monkeypatch.setattr(serving, "_measure_once", measure)


def _floor(monkeypatch, recipe=SPEC, *, acceptance=True, samples=6):
    vals = [1000.0, 1004.0, 996.0, 1002.0, 998.0, 1001.0][:samples]
    acc = [0.5, 0.502, 0.498, 0.501, 0.499, 0.5][:samples]
    _patch(monkeypatch, {"a": [(v, _launch(v, a, counters=acceptance)) for v, a in zip(vals, acc)]})
    row = serving.calibrate_floor(recipe, Path("/a"), samples=samples)
    return row, serving.steps_floor_row(recipe, row)


def _compare(monkeypatch, steps_floor, *, cand_steps, cand_acc, cand_tok, counters=True,
             tok_floor=1.0, pairs=3):
    _patch(monkeypatch, {
        "a": [(100.0, _launch(1000.0, 0.5)) for _ in range(pairs)],
        "c": [(cand_tok, _launch(cand_steps, cand_acc, counters=counters)) for _ in range(pairs)]})
    return serving.compare(SPEC, Path("/a"), Path("/c"), pairs=pairs, floor_pct=tok_floor,
                           floor_unit=serving.COMPARE_EFFECT_UNIT,
                           steps_floor_record=steps_floor)


def test_calibration_emits_steps_and_acceptance_aa(monkeypatch):
    row, steps = _floor(monkeypatch)
    block = row["target_sample_steps_est"]
    assert block["status"] == "ok" and block["estimator_family"] == "estimated"
    assert block["floor_pct"] > 0 and block["acceptance_floor_pct"] > 0
    assert steps["schema"] == serving.STEPS_FLOOR_SCHEMA
    assert steps["acceptance_floor_pct"] == block["acceptance_floor_pct"]


def test_faster_steps_equal_acceptance_is_kept(monkeypatch):
    _, steps = _floor(monkeypatch)
    out = _compare(monkeypatch, steps, cand_steps=1100.0, cand_acc=0.5, cand_tok=110.0)
    assert out["decisive"] is True and out["gate"]["acceptance_guard"]["passed"]
    assert out["gate_metric"] == serving.TARGET_STEPS_METRIC
    assert out["effect"] == pytest.approx(0.10)          # headline stays tok/s
    assert out["gate"]["effect"] == pytest.approx(0.10, rel=1e-2)


def test_faster_steps_with_dropped_acceptance_is_refused(monkeypatch):
    _, steps = _floor(monkeypatch)
    out = _compare(monkeypatch, steps, cand_steps=1100.0, cand_acc=0.4, cand_tok=104.0)
    assert out["gate"]["steps_decisive"] is True
    assert out["decisive"] is None
    assert out["gate"]["refusal"].startswith("acceptance_guard: acceptance_regression")
    assert out["gate"]["acceptance_guard"]["acceptance_effect"] < 0


def test_missing_acceptance_counter_fails_closed(monkeypatch):
    _, steps = _floor(monkeypatch)
    out = _compare(monkeypatch, steps, cand_steps=1100.0, cand_acc=0.5, cand_tok=110.0,
                   counters=False)
    assert out["decisive"] is None
    assert "acceptance_unavailable" in out["gate"]["refusal"]


def test_no_acceptance_floor_falls_back_to_tok_s_non_regression(monkeypatch):
    _, steps = _floor(monkeypatch)
    steps = dict(steps, acceptance_floor_pct=None)
    steps.pop("content_sha256")
    steps["content_sha256"] = serving._digest(steps)
    kept = _compare(monkeypatch, steps, cand_steps=1100.0, cand_acc=0.45, cand_tok=101.0)
    assert kept["decisive"] is True
    assert kept["gate"]["acceptance_guard"]["basis"] == "tok_s_fallback_no_acceptance_floor"
    refused = _compare(monkeypatch, steps, cand_steps=1100.0, cand_acc=0.3, cand_tok=95.0)
    assert refused["decisive"] is None and "tok_s_regression" in refused["gate"]["refusal"]


def test_speculative_without_steps_floor_fails_closed_even_with_tok_floor(monkeypatch):
    out = _compare(monkeypatch, None, cand_steps=1100.0, cand_acc=0.5, cand_tok=110.0)
    assert out["decisive"] is None and out["tok_s_decisive"] is True
    assert out["gate"]["refusal"].startswith("speculative_requires_steps_floor")


def test_steps_regression_stays_decisive(monkeypatch):
    _, steps = _floor(monkeypatch)
    out = _compare(monkeypatch, steps, cand_steps=900.0, cand_acc=0.5, cand_tok=90.0)
    assert out["decisive"] is True and out["gate"]["effect"] < 0


def test_tok_s_floor_is_never_a_steps_floor(monkeypatch):
    row, _ = _floor(monkeypatch)
    with pytest.raises(serving.ServingFloorMismatch, match="tok/s floor is never reused"):
        _compare(monkeypatch, row, cand_steps=1100.0, cand_acc=0.5, cand_tok=110.0)


def test_non_speculative_unchanged_and_refuses_steps_floor(monkeypatch):
    _patch(monkeypatch, {"a": [(100.0, _launch(100.0, 0.0, counters=False))],
                         "c": [(110.0, _launch(110.0, 0.0, counters=False))]})
    out = serving.compare(PLAIN, Path("/a"), Path("/c"), pairs=1, floor_pct=1.0,
                          floor_unit=serving.COMPARE_EFFECT_UNIT)
    assert out["decisive"] is True and "gate" not in out
    _, steps = _floor(monkeypatch)
    with pytest.raises(serving.ServingFloorMismatch, match="non-speculative"):
        serving.compare(PLAIN, Path("/a"), Path("/c"), pairs=1, floor_pct=1.0,
                        floor_unit=serving.COMPARE_EFFECT_UNIT, steps_floor_record=steps)


def test_steps_floor_file_roundtrip_and_key_separation(tmp_path, monkeypatch):
    row, _ = _floor(monkeypatch)
    path = serving.write_steps_floor(tmp_path, SPEC, row)
    assert path == serving.steps_floor_path(tmp_path, SPEC)
    assert path != serving.floor_path(tmp_path, SPEC)
    assert serving.load_steps_floor(tmp_path, SPEC)["floor_pct"] > 0
    other = dataclasses.replace(SPEC, ctx=8192)
    with pytest.raises(serving.ServingFloorMismatch):
        serving.steps_floor_row(other, row)
    # A steps/s file placed at the tok/s path is refused by the tok/s loader.
    serving.floor_path(tmp_path, SPEC).write_text(path.read_text())
    with pytest.raises(serving.ServingFloorMismatch, match="steps/s floor"):
        serving.load_floor(tmp_path, SPEC)


def test_calibration_without_steps_refuses_steps_floor(monkeypatch):
    _patch(monkeypatch, {"a": [(100.0, {"status": "unavailable", "value": None})] * 4})
    row = serving.calibrate_floor(SPEC, Path("/a"), samples=4)
    with pytest.raises(serving.ServingFloorMismatch, match="no complete target steps/s"):
        serving.steps_floor_row(SPEC, row)
