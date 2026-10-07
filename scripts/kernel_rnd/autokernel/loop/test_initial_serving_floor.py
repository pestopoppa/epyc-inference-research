"""Actual startup/calibration/file owners; synthetic launches, never hardware."""
from contextlib import contextmanager, nullcontext
import json
from types import SimpleNamespace
from unittest import mock

import pytest

from . import cpu_profile, pool, run, serial_run, serving
from .loop import ActorStopped
from . import test_promotion_targets as fixtures
from .test_cpu_screen import _inputs as cpu_inputs
from .test_existing_gpu_serving_run import _inputs as gpu_inputs


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("case", ["absent", "reuse", "new_request", "override",
                                  "mismatch", "malformed", "dry_run", "failed", "stop", "stop_matched"])
def test_selected_startup_prepares_only_missing_exact_floor(backend, case):
    fixture = fixtures.TheKeepBuildsAProductionCompleteAnchor()
    fixture.setUp()
    try:
        if backend == "cpu":
            launch, prompts, options, _ = cpu_inputs(fixture)
        else:
            launch, prompts, _, options, _ = gpu_inputs(fixture, experimental=True)
        requests = prompts.requests(("glm-fixed2029",), launch.template)
        floor_path = serving.floor_path(fixture.store, launch.template,
                                        frozen_requests=requests)
        retained_path = None
        retained_bytes = None
        if case in {"reuse", "new_request", "override", "mismatch", "malformed"}:
            original_requests = (("different-original-request", requests[0][1]),) \
                if case == "new_request" else requests
            retained_path = serving.write_floor(fixture.store, launch.template, {
                "floor_pct": 7.801, "n": 5, "recipe_hash": launch.template.recipe_hash,
                "request_digest": serving.request_digest(launch.template, original_requests),
                "fixture": "synthetic retained floor, not a new hardware claim"},
                frozen_requests=original_requests, unit=serving.CALIBRATION_UNIT)
            if case == "mismatch":
                row = json.loads(retained_path.read_text())
                row["recipe_hash"] = "0" * 64
                retained_path.write_text(json.dumps(row))
            elif case == "malformed":
                retained_path.write_text("{malformed original floor")
            retained_bytes = retained_path.read_bytes()

        held, launches, pool_calls = [], [], []
        original_main = run.main
        original_calibrate = serving.calibrate_floor
        stop_requested = [False]

        def calibrate(*args, **kwargs):
            if case == "stop":
                assert callable(kwargs.get("should_stop"))
                stop_requested[0] = True
                assert kwargs["should_stop"]()
                raise ActorStopped("synthetic stop during startup floor calibration")
            return original_calibrate(*args, **kwargs)

        @contextmanager
        def hold(*_args, **_quiet):
            held.append("open")
            try:
                yield {"device_id": "synthetic-owned-" + backend}
            finally:
                held.append("close")

        def observe(recipe, build, port, *, resolved_recipe, frozen_requests, **_kwargs):
            assert held and held[-1] == "open"
            assert frozen_requests == requests
            # Canonical JSON normalizes absent env to {}; compare original recipe
            # identity, not the pre-serialization dataclass's None spelling.
            assert recipe.recipe_hash == launch.template.recipe_hash
            assert resolved_recipe.backend == backend
            resolved_recipe.validate_launch(recipe, build, port)
            assert build == fixture.startup_anchor
            launches.append(resolved_recipe.execution_digest)
            if case == "failed":
                raise serving.ServerDied("synthetic original launch failure")
            if case == "stop_matched":
                stop_requested[0] = True
            return 10.0

        def source_pool(**_kwargs):
            # The actual startup must reopen its own completed floor before it
            # hands off to source research. No actor/build is needed for this test.
            reading = serving.load_floor(fixture.store, launch.template,
                                          frozen_requests=requests)
            assert reading.floor_pct == (7.801 if case == "reuse" else 0.0)
            assert reading.request_digest == serving.request_digest(launch.template, requests)
            pool_calls.append(reading)
            return pool.PoolResult()

        def invoke(argv):
            argv += options + ["--serving-pairs", "3", "--out", str(fixture.root / "result")]
            if case == "override":
                argv += [f"--{backend}-calibrate-serving", "4"]
            if case == "dry_run":
                argv += ["--dry-run"]
            if case == "stop_matched":
                argv += ["--serving-instrument", serving.MATCHED_INSTRUMENT]
            with mock.patch.object(run.claim, "hold_cpu", hold), \
                    mock.patch.object(run.claim, "hold", hold), \
                    mock.patch.object(run.claim, "hold_gpu_quiet_measurement",
                                      side_effect=lambda *_args, **_kwargs: nullcontext()) as quiet_hold, \
                    mock.patch.object(run.os, "sched_getaffinity", return_value={0, 1}), \
                    mock.patch.object(run.os, "sched_setaffinity"), \
                    mock.patch.object(run.workload_contract, "read_census", return_value=SimpleNamespace(
                        n_embd=1536, dominant_quant="Q5_0")), \
                    mock.patch.object(cpu_profile, "profile_loop", side_effect=cpu_profile.CpuProfileRefused(
                        "synthetic test; no hardware profiling")), \
                    mock.patch.object(serving, "_measure_once", observe), \
                    mock.patch.object(serving, "calibrate_floor", side_effect=calibrate), \
                    mock.patch.object(serving, "_matched_frame", return_value={}), \
                    mock.patch.object(serving, "_matched_plan", return_value={
                        "orders": [["anchor", "candidate"]]}), \
                    mock.patch.object(run, "_write_new_source_floor",
                                      wraps=run._write_new_source_floor) as floor_write, \
                    mock.patch.object(pool, "provision", return_value=[]) as provision, \
                    mock.patch.object(pool, "stop_requested",
                                      side_effect=lambda *_args: stop_requested[0]), \
                    mock.patch.object(pool, "drive", source_pool), \
                    mock.patch.object(run.build_recipe, "check_rocm_version",
                                      return_value=(True, "6.2.0")):
                result = original_main(argv)
                if case in {"stop", "stop_matched"}:
                    if backend == "gpu" and case == "stop_matched":
                        quiet_hold.assert_called_once_with()
                    provision.assert_not_called()
                    floor_write.assert_not_called()
                    output = fixture.root / "result"
                    original_run = json.loads((output / "loop-run.json").read_text())
                    continuation = json.loads((output / "loop-continuation.json").read_text())
                    terminal = json.loads((fixture.store / "loop-status.json").read_text())
                    assert result == 0
                    assert original_run["iterations"] == []
                    assert original_run["continuation"]["terminal"] == "stopped"
                    assert continuation == original_run["continuation"]
                    reopened, _digest = serial_run.load_completed(
                        output / "loop-continuation.json", expected_argv=argv)
                    assert reopened == continuation
                    assert terminal["state"] == "complete"
                    assert stop_requested[0]
                return result

        with mock.patch.object(run, "main", invoke):
            if case in {"mismatch", "malformed", "failed"}:
                error = {"mismatch": serving.ServingFloorMismatch,
                         "malformed": json.JSONDecodeError,
                         "failed": serving.ServerDied}[case]
                with pytest.raises(error):
                    fixture._run_one_keep()
            else:
                rc, builds, planners, _, log = fixture._run_one_keep()
                assert rc == 0, log
                assert not builds and not planners
        expected = 4 if case == "override" else 3 if case in {"absent", "new_request"} \
            else 1 if case in {"failed", "stop_matched"} else 0
        assert len(launches) == expected
        assert len(pool_calls) == int(case in {"absent", "reuse", "new_request", "override"})
        if case in {"mismatch", "malformed", "dry_run"}:
            assert not held
        else:
            assert held.count("open") == held.count("close") == (1 if backend == "cpu" else 2)
        if retained_path is not None and case != "override":
            assert retained_path.read_bytes() == retained_bytes
        if case in {"dry_run", "failed", "stop", "stop_matched"}:
            assert not floor_path.exists()
        if case == "new_request":
            assert retained_path != floor_path and floor_path.exists()
    finally:
        fixture.doCleanups()
