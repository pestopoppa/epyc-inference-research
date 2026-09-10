"""Prospective process-pair instrument; synthetic rates and tiny HTTP children only."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import time
from types import SimpleNamespace
from unittest import mock

import pytest

from . import archive, bench, cpu_profile, loop, pool, run, serving
from .test_resolved_recipe import BUILD, _resolve
from .test_legacy_cpu_serving import _requests, _server
from .test_serving_residency import _proof, _sampler_class

MODE = {"instrument": serving.MATCHED_INSTRUMENT, "pairs": 5}


def _inputs():
    recipe = serving.Recipe(name="matched", model="/fixture-model", device="none", ngl=0, np=1)
    return recipe, _resolve(recipe, backend="cpu"), (("p", b'{"prompt":[1],"n_predict":8}'),)


def _measure(calls):
    def measure(recipe, build, port, *, evidence, **kwargs):
        started = time.time()
        calls.append((str(build), kwargs["frozen_requests"]))
        evidence.append({"schema": serving.RESIDENCY_SCHEMA, "backend": "cpu",
                         "status": "not_applicable", "window_start": started,
                         "window_end": time.time(), "samples": 0})
        return 10 + (len(calls) % 7) / 100
    return measure


def _floor(monkeypatch):
    recipe, launch, requests = _inputs()
    calls = []
    monkeypatch.setattr(serving, "_measure_once", _measure(calls))
    row = serving.calibrate_floor(recipe, BUILD, samples=24, resolved_recipe=launch,
                                  frozen_requests=requests, **MODE)
    return recipe, launch, requests, row, calls


def test_24_original_pairs_48_launches_replay_five_pair_scalar_and_keep_v1(tmp_path, monkeypatch):
    recipe, launch, requests, row, calls = _floor(monkeypatch)
    assert len(calls) == row["process_launches"] == 48
    assert row["calibration_pairs"] == 24 and row["comparison_pairs"] == 5
    assert row["floor_pct"] == bench.bootstrap_floor(row["anchor_samples"], row["candidate_samples"], ks=(5,))[5]
    assert row["interval"]["level"] == .95 and row["unit"] == "process"
    assert sum(order[0] == "anchor" for order in row["calibration_plan"]["orders"]) == 12
    legacy = serving.write_floor(tmp_path, recipe, {"floor_pct": 7.801,
        "recipe_hash": recipe.recipe_hash, "request_digest": serving.request_digest(recipe, requests)},
        frozen_requests=requests)
    original = legacy.read_bytes()
    path = serving.write_floor(tmp_path, recipe, row, frozen_requests=requests, **MODE)
    reading = serving.load_floor(tmp_path, recipe, frozen_requests=requests, **MODE)
    assert reading.row == row and path != legacy and legacy.read_bytes() == original
    assert serving.load_floor(tmp_path, recipe, frozen_requests=requests).floor_pct == 7.801
    out = serving.compare(recipe, BUILD, BUILD, pairs=5, floor_pct=reading.floor_pct,
        floor_record=reading.row, instrument=serving.MATCHED_INSTRUMENT,
        anchor_resolved_recipe=launch, candidate_resolved_recipe=launch, frozen_requests=requests,
        floor_request_digest=reading.request_digest)
    assert len(calls) == 58 and out["schema"] == "epyc.autokernel.serving_ab.v2"
    assert out["floor_sha256"] == row["content_sha256"]
    assert out["decisive"] == (abs(out["effect"]) * 100 >= row["floor_pct"])
    assert all(request == requests for _, request in calls)
    assert len(out["launch_membership"]) == 10
    assert out["belief_capture"]["schema"] == "epyc.vidya.legacy_serving_capture.v3"


@pytest.mark.parametrize("change", ["unit", "count", "interval", "sample", "scalar", "recipe"])
def test_incompatible_or_rehashed_changed_floor_refused_before_launch(monkeypatch, change):
    recipe, launch, requests, row, calls = _floor(monkeypatch)
    if change == "unit":
        row["unit"] = "arm"
    elif change == "count":
        row["comparison_pairs"] = 3
    elif change == "interval":
        row["interval"]["high_pct"] += 1
    elif change == "sample":
        row["anchor_samples"] = [value * 2 for value in row["anchor_samples"]]
    elif change == "scalar":
        row["floor_pct"] += 1
    else:
        row["recipe_hash"] = "0" * 64
    row["content_sha256"] = serving._digest({k: v for k, v in row.items() if k != "content_sha256"})
    with pytest.raises(serving.ServingFloorMismatch):
        serving.compare(recipe, BUILD, BUILD, pairs=5, floor_pct=row["floor_pct"], floor_record=row,
            instrument=serving.MATCHED_INSTRUMENT, anchor_resolved_recipe=launch,
            candidate_resolved_recipe=launch, frozen_requests=requests,
            floor_request_digest=serving.request_digest(recipe, requests))
    assert len(calls) == 48


def test_ba_invalid_arm_resumes_ordinal_without_repeating_valid_arm(monkeypatch):
    recipe, anchor, requests = _inputs()
    candidate = _resolve(recipe, backend="cpu", build=Path("/candidate"))
    plan = serving._matched_plan(2, seed="1" * 32)
    original_plan = serving._matched_plan
    monkeypatch.setattr(serving, "_matched_plan", lambda n, seed=None: original_plan(n, seed="1" * 32))
    target = next(2 * i + 1 for i, order in enumerate(plan["orders"]) if order[0] == "candidate")
    calls, values = [], []
    def measure(recipe, build, port, *, evidence, **kwargs):
        calls.append("anchor" if build == BUILD else "candidate")
        if len(calls) - 1 == target:
            raise loop.MeasurementInvalid("fixture placement contradiction", {"original": "invalid"})
        evidence.append({"window_start": float(len(calls)), "window_end": float(len(calls)), "status": "unproven"})
        values.append(build)
        return 10.0
    monkeypatch.setattr(serving, "_measure_once", measure)
    with pytest.raises(loop.MeasurementInvalid) as caught:
        run._serving_comparison(lambda: serving.compare(recipe, BUILD, Path("/candidate"), pairs=2,
            floor_pct=None, instrument=serving.MATCHED_INSTRUMENT, anchor_resolved_recipe=anchor,
            candidate_resolved_recipe=candidate, frozen_requests=requests), "fixture")
    assert caught.value.record["failed_ordinal"] == target
    assert caught.value.record["measurement_plan"] == plan
    result = caught.value.reschedule()
    assert isinstance(result, run.ServingComparison) and result.effect == 0
    assert len(values) == 4 and len(calls) == 5
    assert calls == [arm for order in plan["orders"] for arm in order][:target + 1] + \
        [arm for order in plan["orders"] for arm in order][target:]
    assert [row["ordinal"] for row in result.row["launch_membership"]] == list(range(4))


def test_actual_http_counterbalanced_requests_archive_and_teardown(tmp_path, monkeypatch):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    model = tmp_path / "not-a-model"
    model.write_bytes(b"no inference")
    recipe = serving.Recipe(name="matched-http", model=str(model), device="none", ngl=0,
                            np=2, n_predict=8, cpu_list=None)
    anchor, a_log, a_pids = _server(tmp_path / "anchor", recipe, port, 10.0)
    candidate, c_log, c_pids = _server(tmp_path / "candidate", recipe, port, 12.0)
    monkeypatch.setattr(serving.residency, "Sampler", _sampler_class(_proof(peak=0, median=0, kfd=0)))
    requests = _requests()
    row = serving.compare(recipe, Path(anchor.build_dir), Path(candidate.build_dir), pairs=2,
        floor_pct=None, instrument=serving.MATCHED_INSTRUMENT, port=port,
        anchor_resolved_recipe=anchor, candidate_resolved_recipe=candidate, frozen_requests=requests)
    assert row["effect"] == pytest.approx(.2) and row["decisive"] is None
    assert sorted(row["measurement_plan"]["orders"]) == [["anchor", "candidate"], ["candidate", "anchor"]]
    assert len(row["launch_membership"]) == 4
    expected = sorted(body.hex() for _, body in requests) * 4
    for log in (a_log, c_log):
        assert sorted(json.loads(line) for line in log.read_text().splitlines()) == sorted(expected)
    for log in (a_pids, c_pids):
        pids = [int(line) for line in log.read_text().splitlines()]
        assert len(pids) == 2
        for pid in pids:
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
    attempt = {"status": "measured_null", "mechanism_id": "synthetic-http", "comparison": row}
    archive.record(tmp_path / "memory", attempt, epoch="fixture", recorded_at="2026-09-10T00:00:00Z",
                   campaign_id="synthetic-http")


def test_actual_run_startup_compare_then_floor_reuse(monkeypatch):
    from . import test_promotion_targets as fixtures
    from .test_cpu_screen import _inputs as cpu_inputs
    fixture = fixtures.TheKeepBuildsAProductionCompleteAnchor()
    fixture.setUp()
    try:
        launch, prompts, options, _ = cpu_inputs(fixture)
        requests = prompts.requests(("glm-fixed2029",), launch.template)
        calls, comparisons = [], []
        real_main = run.main
        @contextmanager
        def hold(*_args): yield {"device_id": "synthetic-cpu-claim"}
        def drive(**kwargs):
            measure = kwargs["make_measure"](SimpleNamespace(build_dir=fixture.startup_anchor))
            comparison = measure(SimpleNamespace(runtime_pair=None), ())
            assert isinstance(comparison, run.ServingComparison)
            assert comparison.row["measurement_plan"]["instrument"] == serving.MATCHED_INSTRUMENT
            assert kwargs["build_context"]()["serving_instrument"] == MODE
            comparisons.append(comparison)
            return pool.PoolResult()
        def invoke(argv):
            argv += options + ["--serving-pairs", "5", "--serving-instrument", serving.MATCHED_INSTRUMENT]
            with mock.patch.object(run.claim, "hold_cpu", hold), \
                    mock.patch.object(run.os, "sched_getaffinity", return_value={0, 1}), \
                    mock.patch.object(run.os, "sched_setaffinity"), \
                    mock.patch.object(run.workload_contract, "read_census", return_value=SimpleNamespace(
                        n_embd=1536, dominant_quant="Q5_0")), \
                    mock.patch.object(cpu_profile, "profile_loop", side_effect=cpu_profile.CpuProfileRefused("fixture")), \
                    mock.patch.object(serving, "_measure_once", _measure(calls)), \
                    mock.patch.object(pool, "provision", return_value=[]), \
                    mock.patch.object(pool, "drive", drive):
                return real_main(argv)
        with mock.patch.object(run, "main", invoke):
            assert fixture._run_one_keep()[0] == 0
            assert len(calls) == 58
            path = serving.floor_path(fixture.store, launch.template, frozen_requests=requests, **MODE)
            original = path.read_bytes()
            assert fixture._run_one_keep()[0] == 0
            assert len(calls) == 68 and path.read_bytes() == original
        assert len(comparisons) == 2
    finally:
        fixture.doCleanups()


@pytest.mark.parametrize("field,value", [("pairs", None), ("pairs", True), ("pairs", 0),
                                         ("seed", []), ("seed", ""), ("seed", "z" * 32)])
def test_malformed_matched_plan_has_controlled_validation_refusal(field, value):
    from . import surface_validation
    row = {"schema": "epyc.autokernel.serving_ab.v2", "pairs": 5,
           "measurement_plan": serving._matched_plan(5)}
    if field == "pairs":
        row["pairs"] = value
    else:
        row["measurement_plan"][field] = value
    with pytest.raises(surface_validation.SurfaceValidationRefused, match="original serving"):
        surface_validation.classify(row, intended_target=False)


def test_actual_matched_keep_validation_reuse_and_other_target_compare():
    from .test_serial_run import test_actual_cpu_keep_cross_target_then_older_history_uses_latest_source
    real_main = run.main
    compared = []
    original_compare = serving.compare
    def compare(*args, **kwargs):
        assert kwargs["instrument"] == serving.MATCHED_INSTRUMENT
        assert kwargs["floor_record"]["comparison_pairs"] == kwargs["pairs"]
        row = original_compare(*args, **kwargs)
        compared.append(row)
        return row
    def matched(argv):
        return real_main([*argv, "--serving-instrument", serving.MATCHED_INSTRUMENT,
                          "--cpu-calibrate-serving", "24"])
    # The existing test keeps the exact source/build, reuses its original A/B
    # without another launch, then validates a distinct target under its own floor.
    with mock.patch.object(run, "main", matched), mock.patch.object(serving, "compare", compare):
        test_actual_cpu_keep_cross_target_then_older_history_uses_latest_source(False)
    assert compared and all(row["schema"] == "epyc.autokernel.serving_ab.v2" for row in compared)
