"""AK-SURFACE-SERVING-1 follow-up: two-case (steady-state best, burst worst) screening of serving-knob arms."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from . import serving_knob_measure as m
from .serving_knobs import ScreeningInvalid

HERE = Path(__file__).parent
BURST_SAMPLE = HERE / "testdata" / "serving_knob_burst_sample.jsonl"       # first 3 REAL launches of the floor row
BURST_REAL = Path("/mnt/raid0/llm/tmp/ak-27b-served-np4-floor-gpu-20261124/out/results.jsonl")
BURST_RESULT = Path("/mnt/raid0/llm/tmp/ak-27b-served-np4-floor-gpu-20261124/out/RESULT.json")
GIB = 1 << 30


# --- builders in the rows' own shapes ---
def burst_unit(agg=30.0, accepted=280, ms_per_tok=40.0, *, valid=True, why="ok", vram=50 * GIB, invalid_why=()):
    reqs = [{"predicted_n": 384, "predicted_ms": 384 * ms_per_tok * (1 + 0.05 * k), "draft_n": 700,
             "draft_n_accepted": accepted, "prompt_n": 3824} for k in range(4)]
    return {"uid": "launch-1-a0", "arm": "A", "why": why, "reaped": True, "valid": valid, "invalid_why": list(invalid_why),
            "rec": {"aggregate_tok_s": agg, "requests": reqs, "prompt_n": [3824] * 4, "predicted_n": [384] * 4},
            "res": {"resident": True, "own_vram_b": vram}, "res_post": {"resident": True, "own_vram_b": vram}}


def burst(n=3, **kw):
    return m.parse_burst_results([burst_unit(agg=kw.pop("agg", 30.0) + i * 0.1, **kw) for i in range(n)])


def cell(agg, tps=3.0, itl=0.4):
    return {"K": 4, "valid": True, "invalid_why": [], "aggregate_tok_s": agg, "per_request_decode_tok_s_p50": agg / 4,
            "tokens_per_verify_step": tps, "ttft_p95_s": 20.0, "itl_p95_s": itl, "n_completed": 30}


def arm_run(label, k1, k4, *, tps=3.0, valid=True, probe=None, ready=True, alive=True, oom=None, vram=50 * GIB):
    return {"kind": "arm_run", "label": label, "arm": label, "valid": valid, "invalid_why": [] if valid else ["x"],
            "cells": {"1": cell(k1, tps), "2": cell((k1 + k4) / 2, tps), "4": cell(k4, tps)}, "probe": {"garbage": probe},
            "info": {"ready": ready, "alive": alive, "oom_line": oom, "residency": {"resident": True, "own_vram_b": vram}}}


def arms(**overrides):
    base = {"control": m.ArmMeasurement("control", m.parse_steady_state([arm_run("A", 100, 160)], {"A": "control"})["control"],
                                        burst())}
    base.update(overrides)
    return base


def arm(arm_id, k4, *, best_kw=None, worst_kw=None, k1=100):
    best = m.parse_steady_state([arm_run("A", k1, k4, **(best_kw or {}))], {"A": arm_id})[arm_id]
    return m.ArmMeasurement(arm_id, best, burst(**(worst_kw or {})))


# --- parsers ---
def test_tokens_per_verify_step_formula():
    assert m.tokens_per_verify_step(384, 299) == pytest.approx(384 / 85)
    assert m.tokens_per_verify_step(384, 384) is None


def test_burst_parser_on_real_launch_records():
    w = m.parse_burst_results(BURST_SAMPLE)
    assert len(w.launches) == 3 and all(x.valid for x in w.launches)
    first = w.launches[0]
    assert first.aggregate_tok_s == pytest.approx(24.899103011847362)
    assert first.predicted_n == 1536 and first.draft_n_accepted == 299 + 273 + 279 + 279
    assert first.own_vram_b == 53520916480   # res_post (after the round), not the pre-round 52750503936
    assert first.decode_ms_per_token[0] == pytest.approx(39712.323 / 384)
    assert w.tokens_per_verify_step == pytest.approx(
        sum(x.predicted_n for x in w.launches) / sum(x.predicted_n - x.draft_n_accepted for x in w.launches))
    assert w.itl_p95_ms and w.aggregate_tok_s == statistics_median(w)


def statistics_median(w):
    import statistics
    return statistics.median(x.aggregate_tok_s for x in w.launches)


@pytest.mark.skipif(not BURST_REAL.exists(), reason="floor row output not on this host")
def test_burst_parser_on_the_full_filed_floor_row():
    w = m.parse_burst_results(BURST_REAL)
    assert len(w.launches) == 48 and len(w.good) == 48
    assert 28 < w.aggregate_tok_s < 32 and 3.0 < w.tokens_per_verify_step < 4.5
    floor = m.parse_burst_floor(json.loads(BURST_RESULT.read_text()))
    assert floor["sd_ln_ratio"] == pytest.approx(0.0798, abs=1e-3) and floor["n_pairs"] == 24 and floor["unit"] == "process"


def test_burst_floor_refuses_a_result_without_a_process_floor():
    with pytest.raises(ScreeningInvalid):
        m.parse_burst_floor({"verdict": "FUTILITY_NOISE"})
    with pytest.raises(ScreeningInvalid):
        m.parse_burst_results([])


def test_steady_state_parser_reads_cells_fit_and_probe():
    got = m.parse_steady_state([arm_run("A", 100, 160), arm_run("B", 110, 150, probe="too short")], {"A": "control", "B": "f16"})
    assert got["control"].cells[4]["aggregate_tok_s"] == 160 and got["control"].k4_over_k1 == pytest.approx(1.6)
    assert got["control"].fits and got["control"].valid and got["control"].own_vram_b == 50 * GIB
    assert got["f16"].probe_garbage == "too short"


def test_steady_state_dropped_arm_is_a_memory_misfit_and_unmapped_label_raises():
    dropped = {"kind": "dropped", "label": "B", "reason": "hipErrorOutOfMemory", "info": {}}
    got = m.parse_steady_state([dropped, {"kind": "arm_run", "smoke": True, "label": "ZZ"}], {"B": "f16"})
    assert got["f16"].fits is False and got["f16"].fit_why == "hipErrorOutOfMemory"
    with pytest.raises(ScreeningInvalid, match="B65K"):
        m.parse_steady_state([dropped | {"label": "B65K"}], {"B": "f16"})


def test_steady_state_oom_line_in_info_means_does_not_fit():
    got = m.parse_steady_state([arm_run("A", 1, 2, oom="failed to allocate")], {"A": "a"})
    assert got["a"].fits is False and got["a"].fit_why == "failed to allocate"


def test_steady_state_reads_the_results_file(tmp_path):
    p = tmp_path / "results.jsonl"
    p.write_text("\n".join(json.dumps(u) for u in [arm_run("A", 100, 160)]) + "\n")
    assert m.parse_steady_state(p, {"A": "control"})["control"].cells[1]["aggregate_tok_s"] == 100


# --- evaluator ---
def test_ranks_by_best_case_aggregate_not_worst_case():
    a = arms(fast=arm("fast", 200, worst_kw={"agg": 20.0}), mid=arm("mid", 170, worst_kw={"agg": 40.0}))
    r = m.screen_cases(a)
    assert [e["arm_id"] for e in r["ranking"]] == ["fast", "mid", "control"]
    assert r["best_arm"] == "fast" and r["screening_only"] and r["promotable"] is False and not r["rejected"]
    assert r["ranking"][0]["best"]["vs_control_pct"] == pytest.approx(25.0)
    assert r["ranking"][0]["best"]["k4_over_k1"] == pytest.approx(2.0)
    assert r["ranking"][0]["worst"]["aggregate_tok_s"] < r["ranking"][1]["worst"]["aggregate_tok_s"]


def test_rank_k_selects_the_cell():
    a = arms(x=arm("x", 150, k1=130))           # beats control at K=1 (130 > 100), loses at K=4 (150 < 160)
    assert m.screen_cases(a, rank_k=1)["best_arm"] == "x" and m.screen_cases(a)["best_arm"] is None
    with pytest.raises(ScreeningInvalid):
        m.screen_cases(a, rank_k=3)


@pytest.mark.parametrize("worst_kw,reason", [
    ({"ms_per_tok": 49.0}, "itl-regression>20%"),
    ({"vram": 65 * GIB}, "does-not-fit-memory"),
    ({"valid": False, "why": "ok", "invalid_why": ["predicted_n short"]}, "incorrect-output"),
    ({"valid": False, "why": "not-resident"}, "does-not-fit-memory"),
    ({"valid": False, "why": "error: boom"}, "launch-failed:error: boom"),
    ({"n": 2}, "insufficient-burst-launches:2<3"),
])
def test_a_faster_arm_failing_a_worst_case_guard_is_rejected(worst_kw, reason):
    r = m.screen_cases(arms(fast=arm("fast", 300, worst_kw=worst_kw), ok=arm("ok", 165)))
    assert [e["arm_id"] for e in r["ranking"]] == ["ok", "control"]
    assert reason in r["rejected"][0]["reasons"] and r["rejected"][0]["arm_id"] == "fast"


@pytest.mark.parametrize("best_kw,reason", [({"probe": "too short"}, "incorrect-output"),
                                            ({"ready": False}, "does-not-fit-memory"),
                                            ({"valid": False}, "best-case-invalid"),
                                            ({"vram": 70 * GIB}, "does-not-fit-memory")])
def test_best_case_validity_fit_and_probe_also_reject(best_kw, reason):
    r = m.screen_cases(arms(x=arm("x", 300, best_kw=best_kw)))
    assert reason in r["rejected"][0]["reasons"] and r["best_arm"] is None


def test_itl_exactly_at_the_limit_passes_and_is_reported():
    r = m.screen_cases(arms(x=arm("x", 200, worst_kw={"ms_per_tok": 48.0})))
    assert r["best_arm"] == "x" and r["ranking"][0]["worst"]["itl_regression_pct"] == pytest.approx(20.0)


def test_missing_case_rejects_the_arm_and_an_unsound_control_refuses():
    only_best = m.ArmMeasurement("b", m.parse_steady_state([arm_run("A", 1, 999)], {"A": "b"})["b"], None)
    only_worst = m.ArmMeasurement("w", None, burst())
    r = m.screen_cases(arms(b=only_best, w=only_worst))
    assert {e["arm_id"]: e["reasons"] for e in r["rejected"]} == {"b": ["worst-case-missing"], "w": ["best-case-missing"]}
    with pytest.raises(ScreeningInvalid, match="no 'control'"):
        m.screen_cases({"x": only_best})
    with pytest.raises(ScreeningInvalid, match="both"):
        m.screen_cases({"control": only_best})
    bad = arm("control", 160, worst_kw={"valid": False, "why": "not-resident"})
    with pytest.raises(ScreeningInvalid, match="unsound"):
        m.screen_cases({"control": bad})


def test_tokens_per_step_is_carried_from_both_cases_and_decomposed_against_control():
    """A drafter win (more tokens/step, same step rate) and a step-cost win (same tokens/step, faster steps) both
    raise aggregate tok/s; the decomposition tells them apart and its factors multiply back to the aggregate ratio."""
    drafter = m.ArmMeasurement("drafter", m.parse_steady_state([arm_run("A", 100, 192, tps=3.6)], {"A": "drafter"})["drafter"],
                               burst(agg=36.0, accepted=336))                     # tps 384/(384-336)=8 vs control 384/104=3.69
    stepcost = m.ArmMeasurement("stepcost", m.parse_steady_state([arm_run("A", 100, 192, tps=3.0)], {"A": "stepcost"})["stepcost"],
                                burst(agg=36.0))
    r = m.screen_cases(arms(drafter=drafter, stepcost=stepcost))
    by = {e["arm_id"]: e for e in r["ranking"]}
    d_best, s_best = by["drafter"]["best"]["decomposition"], by["stepcost"]["best"]["decomposition"]
    assert d_best["aggregate_ratio"] == pytest.approx(1.2) and s_best["aggregate_ratio"] == pytest.approx(1.2)
    assert d_best["tokens_per_step_ratio"] == pytest.approx(1.2) and d_best["steps_per_s_ratio"] == pytest.approx(1.0)
    assert s_best["tokens_per_step_ratio"] == pytest.approx(1.0) and s_best["steps_per_s_ratio"] == pytest.approx(1.2)
    for e in by.values():
        for case in ("best", "worst"):
            d = e[case]["decomposition"]
            assert d["tokens_per_step_ratio"] * d["steps_per_s_ratio"] == pytest.approx(d["aggregate_ratio"])
            assert e[case]["tokens_per_verify_step"] is not None
    w = by["drafter"]["worst"]["decomposition"]
    assert w["tokens_per_step_ratio"] > 2 and w["steps_per_s_ratio"] < 1


def test_noise_floor_is_attached_but_never_changes_a_verdict():
    floor = {"sd_ln_ratio": 0.0798, "screen_floor_pct": 15.64, "serving_floor_pct": 3.37, "n_pairs": 24, "unit": "process"}
    a = arms(x=arm("x", 161))
    with_floor, without = m.screen_cases(a, noise_floor=floor), m.screen_cases(a)
    assert with_floor["burst_noise_floor"] == floor and without["burst_noise_floor"] is None
    assert with_floor["ranking"] == without["ranking"] and with_floor["best_arm"] == "x"


def test_parsed_real_burst_and_synthetic_steady_state_round_trip_through_the_screen():
    worst = m.parse_burst_results(BURST_SAMPLE)
    best = m.parse_steady_state([arm_run("A", 100, 160)], {"A": "control"})
    r = m.screen_cases(m.measurements(best, {"control": worst}))
    assert r["ranking"][0]["arm_id"] == "control" and r["ranking"][0]["worst"]["n_launches"] == 3
