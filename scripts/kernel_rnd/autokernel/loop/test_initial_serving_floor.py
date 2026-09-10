"""Actual startup/calibration/file owners; synthetic launches, never hardware."""
from contextlib import contextmanager
import json
from types import SimpleNamespace
from unittest import mock

import pytest

from . import cpu_profile, pool, run, serving
from . import test_promotion_targets as fixtures
from .test_cpu_screen import _inputs as cpu_inputs
from .test_existing_gpu_serving_run import _inputs as gpu_inputs


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("case", ["absent", "reuse", "new_request", "override",
                                  "mismatch", "malformed", "dry_run", "failed"])
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
                "floor_pct": 7.801, "recipe_hash": launch.template.recipe_hash,
                "request_digest": serving.request_digest(launch.template, original_requests),
                "fixture": "synthetic retained floor, not a new hardware claim"},
                frozen_requests=original_requests)
            if case == "mismatch":
                row = json.loads(retained_path.read_text())
                row["recipe_hash"] = "0" * 64
                retained_path.write_text(json.dumps(row))
            elif case == "malformed":
                retained_path.write_text("{malformed original floor")
            retained_bytes = retained_path.read_bytes()

        held, launches, pool_calls = [], [], []
        original_main = run.main

        @contextmanager
        def hold(*_args):
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
            with mock.patch.object(run.claim, "hold_cpu", hold), \
                    mock.patch.object(run.claim, "hold", hold), \
                    mock.patch.object(run.os, "sched_getaffinity", return_value={0, 1}), \
                    mock.patch.object(run.os, "sched_setaffinity"), \
                    mock.patch.object(run.workload_contract, "read_census", return_value=SimpleNamespace(
                        n_embd=1536, dominant_quant="Q5_0")), \
                    mock.patch.object(cpu_profile, "profile_loop", side_effect=cpu_profile.CpuProfileRefused(
                        "synthetic test; no hardware profiling")), \
                    mock.patch.object(serving, "_measure_once", observe), \
                    mock.patch.object(pool, "provision", return_value=[]), \
                    mock.patch.object(pool, "drive", source_pool):
                return original_main(argv)

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
            else 1 if case == "failed" else 0
        assert len(launches) == expected
        assert len(pool_calls) == int(case in {"absent", "reuse", "new_request", "override"})
        if case in {"mismatch", "malformed", "dry_run"}:
            assert not held
        else:
            assert held.count("open") == held.count("close") == (1 if backend == "cpu" else 2)
        if retained_path is not None and case != "override":
            assert retained_path.read_bytes() == retained_bytes
        if case in {"dry_run", "failed"}:
            assert not floor_path.exists()
        if case == "new_request":
            assert retained_path != floor_path and floor_path.exists()
    finally:
        fixture.doCleanups()
