"""Original GPU runtime plumbing; synthetic device/host facts, no model hardware."""
from contextlib import closing, contextmanager
from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import json

import pytest

from .. import schemas
from ..evaluator import devices
from ..execution import device_sampler as ds
from . import actors, claim, gates, measurement_capture as mc, resolved_recipe as rr
from . import runtime_admission as ra, runtime_calibration as rc, runtime_window as rw, serving
from .test_runtime_calibration import original_claim
from .test_runtime_treatment import _fixture
from .test_serving_preparation import statistics
from .test_planned_serving import _prompts
from .test_serving_residency import _proof, _sampler_class
from .unified_planner import RuntimeDimension, enumerate_runtime_dimensions


def gpu_pair(tmp_path):
    cpu, log, pids = _fixture(tmp_path)
    old = cpu.anchor
    binary = Path(old.executable.path)
    # A tiny real HTTP child, not a model or GPU instruction. Long enough to
    # exercise the injected sampler's request-phase boundaries deterministically.
    binary.write_text(binary.read_text().replace("raw = self.rfile.read", "import time; time.sleep(.08)\n        raw = self.rfile.read"))
    template = replace(old.template, device="ROCm0", ngl=99)
    artifacts = {"model": old.model.to_dict(), "drafter": None,
        "executable": {**old.executable.to_dict(), "sha256": hashlib.sha256(binary.read_bytes()).hexdigest()},
        "dsos": [row.to_dict() for row in old.dsos]}
    anchor = rr.resolve_canonical_launch(template, build_dir=old.build_dir,
        command_argv=template.server_argv(Path(old.build_dir), old.port), topology_prefix=(),
        launch_environment=dict(old.launch_env), artifact_identities=artifacts,
        backend="gpu", environment_policy=old.environment_policy, port=old.port,
        runtime_binary_dir=old.runtime_binary_dir, runtime_ld_paths=old.runtime_ld_paths,
        provenance=dict(old.provenance))
    pair = enumerate_runtime_dimensions(anchor, (RuntimeDimension("gpu-threads", "threads",
        template.threads, template.threads + 1, "original-hypothesis:gpu-threads"),))[0]
    return pair, log, pids


def test_gpu_actor_preserves_original_model_requests_binary_and_device(tmp_path):
    pair, _, _ = gpu_pair(tmp_path)
    context = {"target": {"recipe": pair.anchor.to_dict()},
        "runtime_anchor": pair.anchor.to_dict(), "runtime_env_keys": []}
    actual = actors._runtime_pair({"kind": "threads", "candidate": pair.candidate.template.threads},
                                 context, "gpu-threads")
    assert actual == pair
    assert actual.anchor.backend == actual.candidate.backend == "gpu"
    assert actual.anchor.model == actual.candidate.model
    assert actual.anchor.executable == actual.candidate.executable
    assert actual.anchor.dsos == actual.candidate.dsos
    assert actual.candidate.template.device == "ROCm0"
    with pytest.raises(actors.ProviderTransient, match="installed runtime keys"):
        actors._runtime_pair({"kind": "env", "candidate": {"key": "GGML_IQK", "value": "1"}},
                            context, "not-gpu-evidence")


def device_body(*, clock=1700, count=3, offsets=None):
    samples = tuple(ds.TimedDeviceStateSample(float(offset),
        devices.DeviceStateSample(clock, 1200, 100, 60, True))
        for offset in (range(1, count + 1) if offsets is None else offsets))
    receipt = ds.DeviceSamplingReceipt("original-fixture", "ROCm0", "synthetic-device",
        "2026-09-10T00:00:00Z", "2026-09-10T00:00:04Z", 1, 4, ("synthetic-reader",), samples)
    return {"gpu_device": {"receipt": receipt.to_dict(), "error": None,
        "started_monotonic_s": 10, "ended_monotonic_s": 14}}


def test_original_device_parser_preserves_throttle_missing_and_exact_request_window():
    responses = [{"raw": {"phase": "measurement", "started_monotonic_s": 10.5,
                           "ended_monotonic_s": 13.5}}]
    assert rw.gpu_device_state(device_body(), responses).check().outcome == schemas.PASS
    assert rw.gpu_device_state(device_body(clock=1000), responses).check().outcome == schemas.FAIL
    with pytest.raises(ValueError, match="fewer than two"):
        rw.gpu_device_state(device_body(count=1), responses)
    outside = [{"raw": {"phase": "measurement", "started_monotonic_s": 20,
                        "ended_monotonic_s": 23}}]
    with pytest.raises(ValueError, match="enclose"):
        rw.gpu_device_state(device_body(), outside)
    moved = device_body()
    moved["gpu_device"]["receipt"]["samples"][0]["sclk_mhz"] = 200
    with pytest.raises(ValueError, match="differs"):
        rw.gpu_device_state(moved, responses)


def test_gpu_during_request_interior_gap_is_not_covered_by_valid_edges(monkeypatch):
    monkeypatch.setattr(rw, "health", lambda body: schemas.Check(schemas.PASS, ("synthetic host",)))
    responses = [{"raw": {"phase": "measurement", "started_monotonic_s": 10.5,
                           "ended_monotonic_s": 13.5}}]
    host = {"samples": [{"phase_contained": True, "snapshot": {"started": offset,
        "ended": offset}} for offset in (11, 12, 13)]}
    assert rw.launch_health({**host, **device_body()}, responses,
                            max_gap_s=1.1).outcome == schemas.PASS
    gap = rw.launch_health({**host, **device_body(offsets=(1, 3))}, responses, max_gap_s=1.1)
    assert gap.outcome == schemas.COULD_NOT_CHECK
    assert any("gap" in reason for reason in gap.reasons)


def test_gpu_calibration_actual_http_keeps_both_original_claims_and_device_trace(tmp_path, monkeypatch):
    pair, log, pids = gpu_pair(tmp_path)
    sampler_init = ds.RocmSmiSampler.__init__
    def numeric(argv, timeout):
        return ds.SnapshotResult(tuple(argv), 0,
            "sclk clock level: 0: (1700Mhz)\nmclk clock level: 0: (1200Mhz)\n"
            "Average Graphics Package Power (W): 100\nTemperature (Sensor junction) (C): 60\n", "")
    def fixture_sampler(self, **kwargs):
        sampler_init(self, interval_s=.02, runner=numeric, source="synthetic numeric GPU fixture")
    monkeypatch.setattr(ds.RocmSmiSampler, "__init__", fixture_sampler)
    monkeypatch.setattr(serving.residency, "Sampler", _sampler_class(_proof()))
    monkeypatch.setattr(rw, "snapshot", lambda *a, **k:
        {"complete": False, "errors": ["synthetic host, not health evidence"]})
    monkeypatch.setattr(rw, "launch_health", lambda *a, **k:
        schemas.Check(schemas.PASS, ("synthetic host fixture only",)))
    with closing(mc.ArtifactStore(tmp_path / "captures")) as store, original_claim(tmp_path / "cpu.lock") as cpu, \
            claim.hold(tmp_path / "gpu.lock") as gpu:
        neutral = rc.neutral_material(store=store, anchor=pair.anchor)
        kwargs = dict(store=store, held_claim=cpu, campaign_id="calibration-campaign", epoch="gpu",
            anchor=pair.anchor, neutral=neutral, prompts=_prompts(pair.anchor.template),
            statistical=statistics(), host_state={"fixture": "not hardware health"})
        with pytest.raises(rc.RuntimeCalibrationRefused, match="original CPU-host"):
            rc.DirectCalibration(**kwargs)
        frame = rc.DirectCalibration(**kwargs, gpu_claim=gpu)
        assert frame.frame["backend"] == "llama_gpu"
        assert frame.frame["gpu_claim_footprint"]["device_id"] == claim.DEVICE_ID
        context = {"campaign_id": "calibration-campaign", "epoch": "gpu",
            "comparison_id": frame.identity + ":aa", "arm": "anchor", "launch_index": 0}
        frame._launch(("aa", 0, "anchor"), pair.anchor, context)
        body, rows = frame._reopen_launch(frame.launches[0], ["aa", 0, "anchor"], pair.anchor, context)
        assert body["claim_open"]["locks"] != body["gpu_claim_open"]["locks"]
        assert rw.same_claim(body["gpu_claim_open"], body["gpu_claim_close"])
        assert rw.gpu_device_state(body["during_work"], rows).check().outcome == schemas.PASS
        assert body["during_work"]["gpu_device"]["receipt"]["sample_count"] >= 2
        assert len(pids.read_text().splitlines()) == 1
        assert len(log.read_text().splitlines()) == 2 * pair.anchor.template.np
        before = list(frame.launches)
        monkeypatch.setattr(gpu, "observe", lambda: {"status": "lost"})
        with pytest.raises(rc.RuntimeCalibrationRefused, match="GPU claim"):
            frame._launch(("aa", 0, "candidate"), pair.candidate, {**context, "arm": "candidate"})
        assert frame.launches == before and frame.pending is None
        assert len(pids.read_text().splitlines()) == 1


def test_missing_gpu_control_supplier_cannot_launch_cpu_controls_or_calibration(tmp_path, monkeypatch):
    pair, _, _ = gpu_pair(tmp_path)
    owner = ra.RuntimeAdmission.__new__(ra.RuntimeAdmission)
    monkeypatch.setattr(ra.RuntimeAdmission, "calibration", lambda *a, **k: pytest.fail("calibration spent"))
    with pytest.raises(rc.RuntimeCalibrationRefused, match="GPU positive/historical"):
        owner._controls(pair.anchor, 0)


def test_gpu_oracle_uses_actual_backend_and_original_launch_environment(tmp_path, monkeypatch):
    pair, _, _ = gpu_pair(tmp_path)
    oracle = Path(pair.anchor.build_dir) / "bin/test-backend-ops"
    oracle.write_text("fixture-not-executed")
    oracle.chmod(0o700)
    calls = []
    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return type("Result", (), {"returncode": 0, "stdout": "1/1 backends passed", "stderr": ""})()
    monkeypatch.setattr(gates.subprocess, "run", original)
    result = gates.op_correctness(Path(pair.anchor.build_dir), backend="ROCm0", resolved_recipe=pair.anchor)
    assert calls and calls[0][0][0][calls[0][0][0].index("-b") + 1] == "ROCm0"
    assert calls[0][1]["env"] == dict(pair.anchor.launch_env)
    assert result.gate != "oracle_unavailable"
    with pytest.raises(ValueError, match="backend differs"):
        gates.op_correctness(Path(pair.anchor.build_dir), backend="CPU", resolved_recipe=pair.anchor)


def test_actual_gpu_main_runtime_observation_preserves_source_and_skips_unavailable_setup():
    from . import loop, run
    from . import test_existing_gpu_serving_run as gpu_inputs
    from .test_promotion_targets import TheKeepBuildsAProductionCompleteAnchor
    fixture = TheKeepBuildsAProductionCompleteAnchor()
    fixture.setUp()
    try:
        manifest_factory = gpu_inputs._campaign
        with mock.patch.object(gpu_inputs, "_campaign", lambda **kwargs:
                {**manifest_factory(**kwargs), "campaign_id": "ak-gpu-runtime-fixture"}):
            selected, manifest, _resolved, options, _install = gpu_inputs._inputs(fixture, experimental=True)
        requests = manifest.requests(("glm-fixed2029",), selected.template)
        serving.write_floor(fixture.store, selected.template, {
            "floor_pct": 7.8, "recipe_hash": selected.template.recipe_hash,
            "request_digest": serving.request_digest(selected.template, requests)}, frozen_requests=requests)
        actual_main, actual_gpu_hold = run.main, claim.hold
        observed, proposed = [], []

        @contextmanager
        def host(_cpus):
            with original_claim(fixture.root / "private-host.lock") as holder:
                yield holder

        @contextmanager
        def gpu():
            with actual_gpu_hold(fixture.root / "private-device.lock") as holder:
                yield holder

        def forbidden(*args, **kwargs):
            pytest.fail("GPU runtime observation reached a source build/author/critic or unavailable calibration")

        def propose(context):
            original = rr.CanonicalResolvedRecipe.from_dict(context["runtime_anchor"])
            assert original.backend == "gpu"
            pair = actors._runtime_pair({"kind": "threads", "candidate": original.template.threads + 1},
                                        context, "gpu-runtime-original")
            proposed.append(pair)
            return loop.Hypothesis("gpu-runtime-original", "host threading", "same rate", "runtime", "threads",
                                   runtime_pair=pair)

        def measure(template, build, port, *, resolved_recipe, frozen_requests, **kwargs):
            assert resolved_recipe.backend == "gpu" and frozen_requests == requests
            assert Path(build) == fixture.startup_anchor
            resolved_recipe.validate_launch(template, build, port)
            observed.append(resolved_recipe)
            return 10.0

        def gpu_main(argv):
            with mock.patch.object(run.claim, "hold", gpu), mock.patch.object(run.claim, "hold_cpu", host), \
                    mock.patch.object(run.os, "sched_setaffinity"), \
                    mock.patch.object(run.workload_contract, "read_census", return_value=SimpleNamespace(
                        n_embd=1536, dominant_quant="Q5_0")), \
                    mock.patch.object(run.actors, "AgentPlanner", return_value=SimpleNamespace(
                        propose=propose, author=forbidden)), \
                    mock.patch.object(run.actors, "AgentCritic", return_value=SimpleNamespace(
                        review_hypothesis=forbidden, review_patch=forbidden)), \
                    mock.patch.object(gates, "compiles", forbidden), \
                    mock.patch.object(gates, "op_correctness", side_effect=lambda *a, **k:
                        gates.Verdict("correctness", k["backend"] == "ROCm0", "synthetic GPU oracle")), \
                    mock.patch.object(rc.DirectCalibration, "collect", forbidden), \
                    mock.patch.object(serving, "calibrate_floor", forbidden), \
                    mock.patch.object(serving, "_measure_once", measure):
                return actual_main(argv + options + ["--calibrate-runtime", "--out", str(fixture.root / "result")])

        before = run._git(fixture.repo, "rev-parse", "HEAD")
        with mock.patch.object(run, "main", gpu_main):
            result, builds, _planners, _scratch, output = fixture._run_one_keep()
        body = json.loads((fixture.root / "result/loop-run.json").read_text())
        assert result == 0, output
        assert len(proposed) == 1 and len(observed) == 4, body["iterations"]
        assert not builds
        assert run._git(fixture.repo, "rev-parse", "HEAD") == before
        assert "runtime calibration not started: GPU positive/historical" in output
        assert body["runtime_preparation"]["status"] == "observed_not_admitted"
        assert body["iterations"][0]["status"] == "runtime_observed"
        assert body["iterations"][0]["comparison"]["decisive"] is None
    finally:
        fixture.tearDown()
