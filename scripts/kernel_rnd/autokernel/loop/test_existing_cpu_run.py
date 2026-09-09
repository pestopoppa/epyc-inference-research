"""Five iterations through the installed legacy lifecycle; observations are synthetic."""
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from unittest import mock

import pytest

from . import claim, gates, pool, resolved_recipe as rr, run, serving
from .test_glm_frozen_requests import _canonical_launch, _manifest, _request
from . import test_promotion_targets as promotion_fixture


@pytest.mark.parametrize("dry_run", [True, False])
def test_existing_main_cpu_five_iterations_preserves_canonical_champion(dry_run):
    fixture = promotion_fixture.TheKeepBuildsAProductionCompleteAnchor()
    fixture.setUp()
    try:
        model = fixture.root / f"{run.bench.MEASURED_FLOOR_MODEL_STEM}.gguf"
        original_head = fixture.tip
        branch = "ak/experimental/cpu-serving-test"
        run._git(fixture.repo, "checkout", "-b", branch)
        _, source = _canonical_launch(18311)
        command = list(source.command_argv)
        command[0] = str(fixture.startup_anchor / "bin/llama-server")
        command[command.index("-m") + 1] = str(model)
        template = rr.canonical_recipe_projection(
            name="cpu-fixture", command_argv=command, topology_prefix=source.topology_prefix,
            n_predict=512, temperature=0.0, top_k=1)

        def install_binaries(build):
            root = Path(build) / "bin"
            root.mkdir(parents=True, exist_ok=True)
            for name in ("llama-server", "libggml-cpu.so"):
                (root / name).write_bytes(b"synthetic fixture artifact")

        install_binaries(fixture.startup_anchor)

        def identity(role, path):
            return rr.ArtifactDigest(role, str(path), hashlib.sha256(path.read_bytes()).hexdigest()).to_dict()

        selected = rr.resolve_canonical_launch(
            template, build_dir=fixture.startup_anchor, command_argv=command,
            topology_prefix=source.topology_prefix,
            launch_environment={"LD_LIBRARY_PATH": str(fixture.startup_anchor / "bin")},
            artifact_identities={"model": identity("model", model), "drafter": None,
                "executable": identity("executable", fixture.startup_anchor / "bin/llama-server"),
                "dsos": [identity("dso", fixture.startup_anchor / "bin/libggml-cpu.so")]},
            backend="cpu", environment_policy=source.environment_policy, port=18311,
            runtime_binary_dir=str(fixture.startup_anchor / "bin"),
            runtime_ld_paths=(str(fixture.startup_anchor / "bin"),),
            provenance={**dict(source.provenance),
                        "fixture": "synthetic-observations-not-a-model-run"})
        manifest = _manifest(_request())
        launch_file, prompt_file = fixture.root / "launch.json", fixture.root / "prompts.json"
        launch_file.write_text(json.dumps(selected.to_dict()))
        prompt_file.write_text(json.dumps(manifest.to_dict()))
        expected_requests = manifest.requests(("glm-fixed2029",), template)
        held, issued, measured, builds, oracles = [], [], [], [], []
        real_main = run.main

        @contextmanager
        def cpu_hold(cpu_list):
            assert cpu_list == "0-95"
            held.append(True)
            try:
                yield {"device_id": "cpu", "regions": ["q0", "q1", "q2", "q3"]}
            finally:
                held.append(False)

        def observe(recipe, build, port, *, evidence=None, resolved_recipe=None,
                    frozen_requests=None, **kwargs):
            assert held[-1] is True
            assert frozen_requests == expected_requests
            resolved_recipe.validate_launch(recipe, build, port)
            assert resolved_recipe.backend == "cpu"
            measured.append((str(build), tuple(resolved_recipe.argv), frozen_requests))
            if evidence is not None:
                evidence.append({"backend": "cpu", "status": serving.RESIDENCY_NOT_APPLICABLE})
            if Path(build).name == "lane0-build":
                return 9.9 if len(issued) == 4 else 8.1 if len(issued) == 5 else 9.0
            return 9.0

        def cpu_main(argv):
            # The existing fixture has installed actor/build observation doubles.
            # Retain its real pool, git branch/patch/keep and anchor verification.
            base_compile, base_planner = gates.compiles, run.actors.AgentPlanner

            def compile_cpu(*args, **kwargs):
                assert held[-1] is True
                assert kwargs["cpu_list"] == "0-95"
                assert dict(kwargs["cmake_defines"])["GGML_HIP"] == "OFF"
                assert "llama-server" in kwargs["targets"]
                builds.append(kwargs)
                verdict = base_compile(*args, **kwargs)
                install_binaries(args[1])
                return verdict

            def planner(*args, **kwargs):
                actor = base_planner(*args, **kwargs)
                base_propose = actor.propose

                def propose(context):
                    assert context["target"]["scope"] == "experimental candidate, NOT canonical champion"
                    hypothesis = replace(base_propose(context), mechanism_id=f"cpu-{len(issued) + 1}")
                    issued.append(hypothesis)
                    return hypothesis

                def author(hypothesis, _context):
                    (actor.workspace / "kernel.c").write_text(hypothesis.mechanism_id)
                    return ("kernel.c",)

                actor.propose, actor.author = propose, author
                return actor

            def oracle(build, *, backend):
                assert held[-1] is True and backend == "CPU"
                oracles.append(str(build))
                return gates.Verdict("correctness", True, "synthetic observation")

            argv += ["--cpu-serving-launch", str(launch_file), "--frozen-prompts", str(prompt_file),
                     "--experimental-branch", branch, "--iterations", "5", "--serving-pairs", "2",
                     "--cpu-calibrate-serving", "3", "--out", str(fixture.root / "result")]
            if dry_run:
                argv.append("--dry-run")
            with mock.patch.object(gates, "compiles", compile_cpu), \
                    mock.patch.object(gates, "op_correctness", oracle), \
                    mock.patch.object(run.actors, "AgentPlanner", planner), \
                    mock.patch.object(run.workload_contract, "read_census", run.workload_contract.verify_workload), \
                    mock.patch.object(claim, "hold_cpu", cpu_hold), \
                    mock.patch.object(claim, "hold", side_effect=AssertionError("GPU claim")), \
                    mock.patch.object(run.bench, "compare", side_effect=AssertionError("GPU bench")), \
                    mock.patch.object(run, "noise_floor_pct", side_effect=AssertionError("GPU floor")), \
                    mock.patch.object(run.hotspots, "profile", side_effect=AssertionError("GPU profile")), \
                    mock.patch.object(run.production, "refresh", side_effect=AssertionError("production")), \
                    mock.patch.object(serving, "_measure_once", observe), \
                    mock.patch.object(pool, "prune_anchor_generations", return_value=pool.PruneReport("complete")):
                return real_main(argv)

        with mock.patch.object(run, "main", cpu_main):
            rc, _calls, _planners, _scratch, log = fixture._run_one_keep()
        assert rc == 0, log
        assert run._git(fixture.repo, "rev-parse", run.champion.CANONICAL_BRANCH) == original_head
        if dry_run:
            assert not any((held, issued, measured, builds, oracles))
            assert "DRY RUN" in log
        else:
            assert len(issued) == 5 and len(oracles) == 5
            assert held == [True, False]
            result = json.loads((fixture.root / "result/loop-run.json").read_text())
            assert result["baseline_scope"] == "experimental_candidate_not_champion"
            assert [row["status"] for row in result["iterations"]] == [
                "measured_null", "measured_null", "measured_null", "kept", "measured_null"]
            assert all(row["comparison"]["request_digest"] == result["floor_request_digest"]
                       for row in result["iterations"])
            assert run._git(fixture.repo, "rev-parse", branch) != original_head
    finally:
        fixture.doCleanups()
