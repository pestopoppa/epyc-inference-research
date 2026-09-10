"""Existing pool/git/keep owners; all hardware/provider observations are synthetic."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from . import campaign, gates, legacy_targets, pool, resolved_recipe as rr, run, serial_run, serving
from .test_campaign import _manifest as _campaign, _registry, _target
from .test_glm_frozen_requests import _manifest, _request
from .test_resolved_recipe import _policy
from . import test_promotion_targets as fixture_module


def _inputs(fixture, *, experimental):
    model = fixture.root / f"{run.bench.MEASURED_FLOOR_MODEL_STEM}.gguf"
    branch = "ak/experimental/gpu-serving" if experimental else run.champion.CANONICAL_BRANCH
    if experimental:
        run._git(fixture.repo, "checkout", "-b", branch)
    recipe = serving.Recipe(name="selected-gpu", model=str(model), np=1, ctx=8192,
                            n_predict=512, temperature=0.0, top_k=1, cpu_list=None)

    def install(build):
        binary_dir = Path(build) / "bin"
        binary_dir.mkdir(parents=True, exist_ok=True)
        for name in ("llama-server", "libggml-hip.so"):
            (binary_dir / name).write_bytes(b"synthetic-not-executed-" + Path(build).name.encode())

    install(fixture.startup_anchor)

    def identity(role, path):
        return rr.ArtifactDigest(role, str(path), hashlib.sha256(path.read_bytes()).hexdigest()).to_dict()

    build = fixture.startup_anchor
    selected = rr.resolve_canonical_launch(
        recipe, build_dir=build, command_argv=recipe.server_argv(build, 18311), topology_prefix=(),
        launch_environment={"LD_LIBRARY_PATH": str(build / "bin")},
        artifact_identities={"model": identity("model", model), "drafter": None,
            "executable": identity("executable", build / "bin/llama-server"),
            "dsos": [identity("dso", build / "bin/libggml-hip.so")]},
        backend="gpu", environment_policy=_policy(), port=18311,
        runtime_binary_dir=str(build / "bin"), runtime_ld_paths=(str(build / "bin"),),
        provenance={"export_sha256": "a" * 64, "instance_mode": "full",
                    "source:fixture": "synthetic observations, no hardware"})
    registry = _registry()
    registry["model"]["model-a"].update(path=str(model), sha256=selected.model.sha256)
    target = _target("selected", backend="gpu", context=recipe.ctx, concurrency=recipe.np)
    target.update(speculation=selected.capability.speculation, env={})
    resolved = campaign.resolve_manifest(campaign.CampaignManifest.from_dict(
        _campaign(**{"seeds" if experimental else "production": [target]})),
        registry_snapshot=registry)
    manifest = _manifest(_request())
    paths = {"--gpu-serving-launch": ("gpu.json", selected.to_dict()),
             "--frozen-prompts": ("prompts.json", manifest.to_dict()),
             "--resolved-campaign": ("campaign.json", resolved.to_dict())}
    argv = ["--target-id", "selected", "--serving-pairs", "2"]
    for flag, (name, body) in paths.items():
        path = fixture.root / name
        path.write_text(json.dumps(body))
        argv += [flag, str(path)]
    if experimental:
        argv += ["--experimental-branch", branch]
    return selected, manifest, resolved, argv, install


@pytest.mark.parametrize("experimental,calibrate,cor_wins", [
    (True, False, False), (True, True, True), (False, True, True), (False, True, False),
])
def test_existing_gpu_pool_uses_selected_requests_and_original_keep_owners(
        experimental, calibrate, cor_wins):
    fixture = fixture_module.TheKeepBuildsAProductionCompleteAnchor()
    fixture.setUp()
    try:
        selected, manifest, resolved, options, install = _inputs(fixture, experimental=experimental)
        requests = manifest.requests(("glm-fixed2029",), selected.template)
        measured, held, bench_calls, calibration_calls, cpu_held, affinities = [], [], [], [], [], []
        original_main, original_calibrate = run.main, serving.calibrate_floor
        real_compare = serving.compare
        comparisons = []

        @contextmanager
        def hold():
            assert cpu_held[-1] is True
            held.append(True)
            try:
                yield {"device_id": "synthetic-GPU-claim"}
            finally:
                held.append(False)

        @contextmanager
        def hold_host(cpus):
            assert cpus == "0-1"
            cpu_held.append(True)
            try:
                yield {"device_id": "synthetic-host-claim"}
            finally:
                cpu_held.append(False)

        def observe(recipe, build, port, *, resolved_recipe, frozen_requests, evidence, **kwargs):
            assert held[-1] is True
            assert frozen_requests == requests
            assert resolved_recipe.backend == "gpu"
            resolved_recipe.validate_launch(recipe, build, port)
            assert resolved_recipe.executable.sha256 == hashlib.sha256(
                (Path(build) / "bin/llama-server").read_bytes()).hexdigest()
            measured.append((Path(build), resolved_recipe))
            # These are explicitly synthetic rates, not invented GPU residency.
            if Path(build) == fixture.startup_anchor:
                return 10.0
            if Path(build).name == "lane0-build":
                return 11.0
            return 12.0 if cor_wins else 9.0  # promoted and same-source guard both agree

        def calibrate_original(*args, **kwargs):
            calibration_calls.append(kwargs)
            return original_calibrate(*args, **kwargs)

        def compare_original(*args, **kwargs):
            row = real_compare(*args, **kwargs)
            comparisons.append((args, kwargs, row))
            return row

        def gpu_main(argv):
            base_compile = gates.compiles

            def compile_gpu(*args, **kwargs):
                assert held[-1] is True
                assert dict(kwargs["cmake_defines"])["GGML_HIP"] == "ON"
                assert "llama-server" in kwargs["targets"]
                assert kwargs["cpu_list"] == "0-1" and kwargs["jobs"] <= resolved.resources.build_jobs
                result = base_compile(*args, **kwargs)
                install(args[1])
                return result

            def bench_auxiliary(a, c, model, **kwargs):
                assert not experimental
                assert a.name == "champion_of_record"  # never candidate measurement or guard
                assert kwargs["surface"] == "pp512" and kwargs["noise_floor_pct"] == 99.0
                bench_calls.append((a, c, kwargs))
                return run.bench.Comparison(surface="pp512", anchor_samples=[10.0] * 2,
                    candidate_samples=[11.0] * 2, effect=.1, estimator="median_over_median",
                    pairs=2, noise_floor_pct=99.0, residency={}, calibrated=True)

            argv += options + ["--out", str(fixture.root / "result")]
            if calibrate:
                argv += ["--gpu-calibrate-serving", "2"]
            with mock.patch.object(gates, "compiles", compile_gpu), \
                    mock.patch.object(run.claim, "hold", hold), \
                    mock.patch.object(run.claim, "hold_cpu", hold_host), \
                    mock.patch.object(run.os, "sched_getaffinity", return_value={3, 4}), \
                    mock.patch.object(run.os, "sched_setaffinity", side_effect=lambda pid, cpus:
                                      affinities.append((pid, cpus))), \
                    mock.patch.object(run, "noise_floor_pct", return_value=99.0), \
                    mock.patch.object(run.workload_contract, "read_census", return_value=SimpleNamespace(
                        n_embd=1536, dominant_quant="Q5_0")), \
                    mock.patch.object(run.workload_contract, "verify_workload", side_effect=AssertionError(
                        "selected serving is not legacy production K-quant screen")), \
                    mock.patch.object(run.bench, "compare", bench_auxiliary), \
                    mock.patch.object(run.hotspots, "profile", side_effect=AssertionError("wrong profile")), \
                    mock.patch.object(serving, "_measure_once", observe), \
                    mock.patch.object(serving, "calibrate_floor", calibrate_original), \
                    mock.patch.object(serving, "compare", compare_original), \
                    mock.patch.object(pool, "prune_anchor_generations", return_value=pool.PruneReport("complete")):
                rc = original_main(argv)
                receipt = fixture.root / "result/loop-continuation.json"
                prior, _ = serial_run.load_completed(receipt, expected_argv=argv,
                                                     expected_binding=serial_run.input_binding(argv))
                assert "--gpu-serving-launch" in prior["binding"]["documents"]
                assert prior["selected_target"]["scope"] == "gpu_serving_selected_workload"
                before = (len(measured), len(calibration_calls), len(held))
                # Fresh main parses the original receipt, verifies/rebinds the actual
                # retained anchor and reopens the request floor. Dry-run must not
                # rebuild, recalibrate or acquire another claim merely to switch back.
                assert original_main(argv + ["--resume-run", str(receipt), "--dry-run",
                                             "--out", str(fixture.root / "resume")]) == 0
                assert (len(measured), len(calibration_calls), len(held)) == before
                return rc

        with mock.patch.object(run, "main", gpu_main):
            rc, builds, planners, _, log = fixture._run_one_keep()
        assert rc == 0, log
        assert held == [True, False]
        assert cpu_held == [True, False] and affinities == [(0, {0, 1}), (0, {3, 4})]
        # Missing exact request floors prepare automatically; the explicit option
        # remains an override, not a prerequisite for this selected GPU target.
        assert len(calibration_calls) == 1
        assert calibration_calls[0]["samples"] == 2
        result = json.loads((fixture.root / "result/loop-run.json").read_text())
        outcome = result["iterations"][0]
        assert outcome["status"] == "kept"
        row = outcome["comparison"]
        assert row["request_digest"] == serving.request_digest(selected.template, requests)
        assert row["noise_floor_pct"] == 0.0  # never bench's 99%
        assert row["pairs"] == result["pairs"] == 2
        assert row["decisive"] is True
        assert row["baseline_scope"] == ("experimental_candidate_not_champion" if experimental
                                         else "canonical_candidate_vs_current_anchor")
        context = planners[0].contexts[0]
        assert "CPU EXPERIMENTAL TARGET" not in context["program"]
        assert context["target"]["scope"] == "selected GPU serving workload"
        assert context["target"]["recipe"]["backend"] == "gpu"
        assert context["target"]["enrollment"] == result["target"] == run.status.read(fixture.store)["target"]
        assert result["target"]["original_target"] == resolved.targets[0].to_dict()
        assert result["target"]["scope"] == "gpu_serving_selected_workload"
        assert run.status.read(fixture.store)["gpu"]["device_seconds_under_load"] is None
        assert bool(bench_calls) is (not experimental)
        if experimental:
            assert result["continuation"]["cor_anchor"] is None
            assert run._git(fixture.repo, "rev-parse", run.champion.CANONICAL_BRANCH) == fixture.tip
        else:
            # Original whole-COR gate runs separately; a marginal keep never promotes it by itself.
            cor_comparison = comparisons[-1]
            assert cor_comparison[0][1] == fixture.startup_anchor
            assert cor_comparison[0][2] == fixture.store / "anchor-gen-002"
            assert cor_comparison[1]["anchor_resolved_recipe"].build_dir == str(fixture.startup_anchor)
            assert cor_comparison[1]["candidate_resolved_recipe"].build_dir == str(fixture.store / "anchor-gen-002")
            assert cor_comparison[1]["frozen_requests"] == requests
            cor = result["continuation"]["cor_anchor"]
            assert cor["commit"] == (run._git(fixture.repo, "rev-parse", "HEAD") if cor_wins else fixture.tip)
            assert cor_comparison[2]["effect"] > 0 if cor_wins else cor_comparison[2]["effect"] < 0
        assert len(builds) == 3
        assert measured and all(path.exists() for path, _ in measured)
    finally:
        fixture.doCleanups()


@pytest.mark.parametrize("key,value,accepted", [
    ("gpu_ids", ["ROCm0"], True), ("gpu_ids", ["mi210_0"], True),
    ("gpu_ids", ["ROCm1"], False), ("cpu_logical", [], False),
    ("visibility", "0", True), ("visibility", "1", False), ("visibility", "0,1", False),
])
def test_existing_resource_aliases_and_visibility_are_not_relabelled(key, value, accepted):
    body = _campaign()["resources"]
    environment = {}
    if key == "visibility":
        environment["ROCR_VISIBLE_DEVICES"] = value
    else:
        body[key] = value
    resources = campaign.ResourceRequest.from_dict(body)
    if accepted:
        assert legacy_targets.validate_resources(resources, None, backend="gpu",
                                                  environment=environment) == "0-1"
    else:
        with pytest.raises(legacy_targets.TargetSelectionRefused):
            legacy_targets.validate_resources(resources, None, backend="gpu", environment=environment)


def test_selected_host_claim_failure_restores_original_affinity_without_launching():
    fixture = fixture_module.TheKeepBuildsAProductionCompleteAnchor()
    fixture.setUp()
    try:
        _, _, _, options, _ = _inputs(fixture, experimental=True)
        real_main = run.main
        affinity = []

        def selected_main(argv):
            with mock.patch.object(run.os, "sched_getaffinity", return_value={7, 8}), \
                    mock.patch.object(run.os, "sched_setaffinity", side_effect=lambda pid, cpus:
                                      affinity.append((pid, cpus))), \
                    mock.patch.object(run.workload_contract, "read_census", return_value=SimpleNamespace(
                        n_embd=1536, dominant_quant="Q5_0")), \
                    mock.patch.object(run.claim, "hold_cpu", side_effect=run.claim.ClaimRefused("owned region busy")), \
                    mock.patch.object(run.claim, "hold", side_effect=AssertionError("GPU claim after refusal")), \
                    mock.patch.object(run.pool, "provision", side_effect=AssertionError("launch after refusal")):
                return real_main(argv + options)

        with mock.patch.object(run, "main", selected_main), pytest.raises(run.claim.ClaimRefused):
            fixture._run_one_keep()
        assert affinity == [(0, {0, 1}), (0, {7, 8})]
        assert run.status.read(fixture.store)["state"] == "failed"
    finally:
        fixture.doCleanups()


@pytest.mark.parametrize("case", ["unenrolled", "both", "cpu_backend", "missing_requests",
                                  "foreign_model", "foreign_context", "calibration_backend"])
def test_gpu_serving_bad_inputs_refuse_before_startup(case):
    fixture = fixture_module.TheKeepBuildsAProductionCompleteAnchor()
    fixture.setUp()
    try:
        selected, manifest, _, options, _ = _inputs(fixture, experimental=True)
        argv = ["--worktree", str(fixture.repo), "--anchor-build", str(fixture.startup_anchor),
                "--store", str(fixture.store), *options]
        if case in ("unenrolled", "missing_requests"):
            flag = "--resolved-campaign" if case == "unenrolled" else "--frozen-prompts"
            pos = argv.index(flag)
            del argv[pos:pos + 2]
        elif case == "both":
            argv += ["--cpu-serving-launch", str(fixture.root / "gpu.json")]
        elif case == "cpu_backend":
            argv[argv.index("--gpu-serving-launch")] = "--cpu-serving-launch"
        elif case == "foreign_model":
            argv += ["--model", "/foreign/model"]
        elif case == "calibration_backend":
            argv += ["--cpu-calibrate-serving", "2"]
        else:
            registry = _registry()
            registry["model"]["model-a"].update(path=selected.model.path, sha256=selected.model.sha256)
            resolved = campaign.resolve_manifest(campaign.CampaignManifest.from_dict(_campaign(seeds=[
                _target("selected", backend="gpu", context=4096, concurrency=1)])),
                registry_snapshot=registry)
            (fixture.root / "campaign.json").write_text(json.dumps(resolved.to_dict()))
        with mock.patch.object(run.champion, "verify_startup", side_effect=AssertionError("startup")), \
                pytest.raises(SystemExit) as error:
            run.main(argv)
        assert error.value.code == 2
    finally:
        fixture.doCleanups()
