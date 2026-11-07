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
    # A tok/s regression beyond its floor is a decisive regression of the row.
    assert refused["decisive"] is True and refused["regression"] is True
    assert "tok_s_regression" in refused["gate"]["refusal"]


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


# ---- SL-4 (a): legacy arm order ----------------------------------------------------

def _orders(monkeypatch, recipe, **kw):
    seen = []

    def measure(recipe_, build, port, *, evidence=None, target_sample_steps=None, **_):
        seen.append(Path(build).name)
        if evidence is not None:
            evidence.append({})
        return 100.0
    monkeypatch.setattr(serving, "_measure_once", measure)
    out = serving.compare(recipe, Path("/a"), Path("/c"), pairs=8, floor_pct=None, **kw)
    return out, [seen[i:i + 2] for i in range(0, len(seen), 2)]


def test_short_legacy_stays_anchor_first_and_row_unchanged(monkeypatch):
    out, orders = _orders(monkeypatch, PLAIN)
    assert all(o == ["a", "c"] for o in orders) and "arm_order_plan" not in out


def test_prefill_heavy_legacy_is_randomized_seeded_and_recorded(monkeypatch):
    import json
    body = json.dumps({"prompt": [1] * serving.PREFILL_HEAVY_TOKENS}).encode()
    requests = (("p0", body), ("p1", body + b" "))
    assert serving.prefill_heavy(requests)
    out, orders = _orders(monkeypatch, PLAIN, frozen_requests=requests, arm_order_seed="ab" * 16)
    plan = out["arm_order_plan"]
    assert plan["reason"] == "prefill_heavy" and plan["seed"] == "ab" * 16
    assert plan == {**serving._legacy_order_plan(8, seed="ab" * 16), "reason": "prefill_heavy"}
    assert [["a" if arm == "anchor" else "c" for arm in o] for o in plan["orders"]] == orders
    assert any(o == ["c", "a"] for o in orders) and any(o == ["a", "c"] for o in orders)
    with pytest.raises(serving.RecipeError, match="anchor-first"):
        _orders(monkeypatch, PLAIN, frozen_requests=requests, arm_order="anchor_first")


def test_requested_randomization_and_matched_refusal(monkeypatch):
    out, _ = _orders(monkeypatch, PLAIN, arm_order="randomized")
    assert out["arm_order_plan"]["reason"] == "requested"
    with pytest.raises(serving.RecipeError, match="needs a randomized"):
        _orders(monkeypatch, PLAIN, arm_order_seed="ab" * 16)


# ---- SL-2 fail-closed on regressions (either metric) -------------------------------

def test_no_steps_floor_with_tok_s_regression_is_flagged(monkeypatch):
    from autokernel.loop import accumulate, fold2_gates, lane_targets
    out = _compare(monkeypatch, None, cand_steps=900.0, cand_acc=0.5, cand_tok=90.0)
    assert out["regression"] is True and out["decisive"] is True
    assert out["gate"]["regression_basis"] == ["tok_s"]
    assert serving.is_regression(out) and serving.verdict_effect(out) < 0
    assert fold2_gates.ab_verdict(out) == "FAIL"
    assert accumulate.classify_serving(out) is accumulate.Outcome.DIVERGED
    assert accumulate.negative_beyond_floor(out)
    assert lane_targets.decide(out, {"mode": "floor", "floor_pct": 1.0})[0] is False


def test_steps_up_with_tok_s_decisively_down_is_flagged(monkeypatch):
    from autokernel.loop import accumulate, fold2_gates
    _, steps = _floor(monkeypatch)
    out = _compare(monkeypatch, steps, cand_steps=1100.0, cand_acc=0.3, cand_tok=80.0)
    assert out["gate"]["steps_decisive"] is True and out["gate"]["effect"] > 0
    assert out["regression"] is True and out["decisive"] is True
    assert out["gate"]["regression_basis"] == ["tok_s"]
    assert serving.verdict_effect(out) == pytest.approx(-0.2)
    assert fold2_gates.ab_verdict(out) == "FAIL"
    assert accumulate.classify_serving(out) is accumulate.Outcome.DIVERGED


def test_steps_down_with_tok_s_up_is_flagged_not_kept(monkeypatch):
    from autokernel.loop import accumulate, fold2_gates
    _, steps = _floor(monkeypatch)
    out = _compare(monkeypatch, steps, cand_steps=900.0, cand_acc=0.8, cand_tok=105.0)
    assert out["effect"] > 0 and out["regression"] is True and out["decisive"] is True
    assert out["gate"]["regression_basis"] == ["steps_s"] and serving.verdict_effect(out) < 0
    assert fold2_gates.ab_verdict(out) == "FAIL"
    assert accumulate.classify_serving(out) is accumulate.Outcome.DIVERGED


def test_keep_row_verdict_effect_is_the_steps_effect(monkeypatch):
    _, steps = _floor(monkeypatch)
    out = _compare(monkeypatch, steps, cand_steps=1100.0, cand_acc=0.5, cand_tok=110.0)
    assert out["regression"] is False and serving.verdict_effect(out) == out["gate"]["effect"]
