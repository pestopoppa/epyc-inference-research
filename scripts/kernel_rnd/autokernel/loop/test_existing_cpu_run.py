"""Five iterations through the installed legacy lifecycle; observations are synthetic."""
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
from unittest import mock

import pytest

from . import campaign, campaign_cli, claim, cpu_profile, gates, pool, resolved_recipe as rr, run, serving
from .test_glm_frozen_requests import _canonical_launch, _manifest, _request
from .test_campaign import _manifest as _campaign_manifest, _registry, _target
from . import test_promotion_targets as promotion_fixture


@pytest.mark.parametrize("dry_run", [True, False])
def test_existing_main_cpu_five_iterations_preserves_canonical_champion(
        dry_run, feedback_root=None, profile_observer=None, profile_contexts=None, runtime_only=False,
        invalid_once=False, enrolled_pair=False, runtime_transition=None):
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
                (root / name).write_bytes(b"synthetic fixture artifact" + (
                    Path(build).name.encode() if feedback_root is not None or profile_observer is not None else b""))

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
        resolved_file = fixture.root / "resolved.json"
        if enrolled_pair:
            targets = []
            for target_id in ("target-a", "target-b"):
                target = _target(target_id, backend="cpu", context=template.ctx,
                                 concurrency=template.np)
                target.update(speculation=selected.capability.speculation,
                              env={key: value for key, value in selected.launch_env
                                   if key != "LD_LIBRARY_PATH"})
                targets.append(target)
            registry = _registry()
            registry["model"]["model-a"].update(
                path=str(model), sha256=hashlib.sha256(model.read_bytes()).hexdigest())
            declaration = _campaign_manifest(seeds=targets)
            declaration["resources"]["cpu_logical"] = list(range(96))
            resolved = campaign.resolve_manifest(
                campaign.CampaignManifest.from_dict(declaration), registry_snapshot=registry)
            resolved_file.write_text(json.dumps(campaign_cli.build_output(
                resolved, verify_artifacts=False)))
        expected_requests = manifest.requests(("glm-fixed2029",), template)
        held, issued, measured, builds, oracles = [], [], [], [], []
        invalidated = []
        builds_at_invalid = []
        feedback_contexts = []
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
            candidate = (recipe.threads != template.threads if runtime_only
                         else Path(build).name == "lane0-build")
            if invalid_once and issued and candidate:
                if not invalidated:
                    builds_at_invalid.append(len(builds))
                    invalidated.append(resolved_recipe.to_dict())
                    raise run.loop.MeasurementInvalid("synthetic observed placement contradiction", {
                        "resolved_recipe": resolved_recipe.to_dict(), "teardown": "terminated",
                        "fixture": "synthetic whole-arm observation, not real inference"})
                elif len(invalidated) == 1:
                    assert len(builds) == builds_at_invalid[0]
                    import sqlite3
                    # Actual original owner wrote its invalid outcome before the
                    # same-build retry; no source reset or substitute candidate.
                    with sqlite3.connect(fixture.store / "experiments.db") as db:
                        stored = json.loads(db.execute(
                            "SELECT payload FROM experiments WHERE status='measurement_invalid'"
                        ).fetchone()[0])
                    assert stored["invalid_measurement"]["invalid_arm"]["resolved_recipe"] == resolved_recipe.to_dict()
                    assert invalidated[0] == resolved_recipe.to_dict()
                    invalidated.append(stored)
            if evidence is not None:
                now = time.time()
                evidence.append({"backend": "cpu", "status": serving.RESIDENCY_NOT_APPLICABLE,
                                 "window_start": now, "window_end": now})
            if Path(build).name == "lane0-build":
                position = (len(issued) - 1) % 5 + 1
                return 9.9 if position == 4 else 8.1 if position == 5 else 9.0
            return 9.0

        def cpu_main(argv):
            # The existing fixture has installed actor/build observation doubles.
            # Retain its real pool, git branch/patch/keep and anchor verification.
            base_compile, base_planner = gates.compiles, run.actors.AgentPlanner

            def compile_cpu(*args, **kwargs):
                assert not runtime_only, "runtime treatment must not compile"
                assert not (runtime_transition is not None and len(issued) == 1), "recipe-only keep must not compile"
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
                    if profile_contexts is not None:
                        profile_contexts.append(dict(context["cpu_profile"]))
                    feedback_contexts.append(context["serving_observations"])
                    assert context["target"]["scope"] == "experimental candidate, NOT canonical champion"
                    assert context["program"].startswith("CPU EXPERIMENTAL TARGET")
                    assert "overrides inapplicable GPU instructions below" in context["program"]
                    assert "Author/review source only" in context["program"]
                    assert context["program"].endswith(run.loop.PROGRAM.read_text(encoding="utf-8"))
                    hypothesis = replace(base_propose(context), mechanism_id=f"cpu-{len(issued) + 1}")
                    if "runtime_anchor" in context:
                        assert context["target"]["recipe"] == context["runtime_anchor"]
                    else:
                        # The historical aku-* enrolled pair remains source-only.
                        assert enrolled_pair and not runtime_only and runtime_transition is None
                    if runtime_only or (runtime_transition is not None and not issued):
                        treatment = run.actors._runtime_pair(
                            {"kind": "threads", "candidate": template.threads + 1},
                            context, hypothesis.mechanism_id)
                        hypothesis = replace(hypothesis, runtime_pair=treatment)
                    issued.append(hypothesis)
                    return hypothesis

                def author(hypothesis, _context):
                    assert not runtime_only, "runtime treatment must not author source"
                    (actor.workspace / "kernel.c").write_text(hypothesis.mechanism_id)
                    return ("kernel.c",)

                actor.propose, actor.author = propose, author
                return actor

            def oracle(build, *, backend, resolved_recipe=None):
                assert held[-1] is True and backend == "CPU"
                if runtime_only:
                    assert resolved_recipe.template.threads == template.threads + 1
                    assert Path(build) == fixture.startup_anchor
                oracles.append(str(build))
                return gates.Verdict("correctness", True, "synthetic observation")

            argv += ["--cpu-serving-launch", str(launch_file), "--frozen-prompts", str(prompt_file),
                     "--experimental-branch", branch, "--iterations", "5", "--serving-pairs", "2",
                     "--cpu-calibrate-serving", "3", "--out", str(fixture.root / "result")]
            if enrolled_pair:
                argv += ["--resolved-campaign", str(resolved_file), "--target-id", "target-a"]
            if dry_run:
                argv.append("--dry-run")
            if feedback_root is not None:
                argv += ["--belief-root-repo", str(feedback_root)]
            with mock.patch.object(gates, "compiles", compile_cpu), \
                    mock.patch.object(gates, "op_correctness", oracle), \
                    mock.patch.object(run.actors, "AgentPlanner", planner), \
                    mock.patch.object(run.workload_contract, "read_census", run.workload_contract.verify_workload), \
                    mock.patch.object(claim, "hold_cpu", cpu_hold), \
                    mock.patch.object(claim, "hold", side_effect=AssertionError("GPU claim")), \
                    mock.patch.object(run.bench, "compare", side_effect=AssertionError("GPU bench")), \
                    mock.patch.object(run, "noise_floor_pct", side_effect=AssertionError("GPU floor")), \
                    mock.patch.object(run.hotspots, "profile", side_effect=AssertionError("GPU profile")), \
                    mock.patch.object(cpu_profile, "profile_loop", side_effect=profile_observer or
                        cpu_profile.CpuProfileRefused("test-only profiler unavailable")), \
                    mock.patch.object(run.production, "refresh", side_effect=AssertionError("production")), \
                    mock.patch.object(serving, "_measure_once", observe), \
                    mock.patch.object(pool, "prune_anchor_generations",
                        side_effect=pool.prune_anchor_generations if runtime_transition else None,
                        return_value=pool.PruneReport("complete")):
                return real_main(argv)

        with mock.patch.object(run, "main", cpu_main):
            rc, _calls, _planners, _scratch, log = fixture._run_one_keep()
        assert rc == 0, log
        assert run._git(fixture.repo, "rev-parse", run.champion.CANONICAL_BRANCH) == original_head
        if dry_run:
            assert not any((held, issued, measured, builds, oracles))
            assert "DRY RUN" in log
        else:
            assert len(issued) == (4 if invalid_once else 10 if enrolled_pair else 5)
            assert len(oracles) == len(issued)
            assert held == ([True, False, True, False] if enrolled_pair else [True, False])
            if enrolled_pair:
                # The caller owns the A -> B -> A identity/history assertions.
                # This fixture only establishes both original source/build keeps.
                return
            result = json.loads((fixture.root / "result/loop-run.json").read_text())
            epoch_inputs = {"cpu_execution_digest": selected.execution_digest,
                            "frozen_prompt_digest": manifest.digest}
            expected_epoch = run.archive.epoch_for(
                anchor_commit=original_head, build_recipe=run.build_recipe.NATIVE_CPU_RECIPE.to_dict(),
                host_state=epoch_inputs)
            assert result["epoch"] == expected_epoch
            # Same source/build but a different target or original request must not
            # be ranked as same-epoch evidence. No altered live recipe is launched.
            for field in epoch_inputs:
                changed = {**epoch_inputs, field: "0" * 64}
                assert run.archive.epoch_for(
                    anchor_commit=original_head,
                    build_recipe=run.build_recipe.NATIVE_CPU_RECIPE.to_dict(),
                    host_state=changed) != expected_epoch
            assert result["baseline_scope"] == "experimental_candidate_not_champion"
            assert run.status.read(fixture.store)["baseline_scope"] == result["baseline_scope"]
            if runtime_transition is not None:
                runtime_transition(result, measured, builds)
                return
            if invalid_once:
                rows = result["iterations"]
                assert len(rows) == 5 and rows[0]["status"] == "measurement_invalid"
                assert "comparison" not in rows[0] and "effect_fraction" not in rows[0]
                assert rows[1]["comparison"]["rescheduled_invalid_arms"][0]["failed_arm"] == "candidate"
                assert run.status.read(fixture.store)["measurements_reached"] == 4
                assert len(invalidated) == 2
                assert len(list((fixture.store / "serving-beliefs").glob("*.json"))) == 4
                assert rows[0]["mechanism_id"] == rows[1]["mechanism_id"] == "cpu-1"
                if runtime_only:
                    assert rows[0]["runtime_pair"] == rows[1]["runtime_pair"]
                else:
                    assert list((fixture.store / "patches").glob("cpu-1.lane0.*.patch"))
                # The three A/A launches precede A,B(invalid),B(rescheduled).
                assert measured[4] == measured[5]
                assert measured[3] != measured[4]
                return
            if runtime_only:
                assert not builds
                assert run.status.read(fixture.store)["measurements_reached"] == 5
                assert [row["status"] for row in result["iterations"]] == ["runtime_observed"] * 5
                assert all(row["comparison"]["decisive"] is None for row in result["iterations"])
                assert all(row["runtime_pair"]["anchor"]["build_dir"] == str(fixture.startup_anchor)
                           for row in result["iterations"])
                assert run._git(fixture.repo, "rev-parse", branch) == original_head
                return
            assert [row["status"] for row in result["iterations"]] == [
                "measured_null", "measured_null", "measured_null", "kept", "measured_null"]
            assert all(row["comparison"]["request_digest"] == result["floor_request_digest"]
                       for row in result["iterations"])
            assert run._git(fixture.repo, "rev-parse", branch) != original_head
            if feedback_root is not None:
                assert feedback_contexts[0]["rows"] == []
                first_null = feedback_contexts[1]["rows"][0]
                assert first_null["anchor_tok_s"] == first_null["candidate_tok_s"] == 9.0
                after_keep = feedback_contexts[4]
                assert not after_keep["errors"]
                assert after_keep["scope"]["anchor_build"] != str(fixture.startup_anchor)
                assert after_keep["scope"]["anchor_execution_digest"] != selected.execution_digest
                assert after_keep["scope"]["anchor_execution_digest"] == run._cpu_arm(
                    selected, Path(after_keep["scope"]["anchor_build"])).execution_digest
                assert all(row["anchor_execution_digest"] == after_keep["scope"]["anchor_execution_digest"]
                           for row in after_keep["rows"])
                recovered = run.serving_beliefs.PlannerFeedback(fixture.store, feedback_root).context(
                    after_keep["scope"])
                last_null = next(row for row in recovered["rows"] if row["mechanism_id"] == "cpu-5")
                assert last_null["anchor_tok_s"] == 9.0 and last_null["candidate_tok_s"] == 8.1
                assert last_null["anchor_execution_digest"] == after_keep["scope"]["anchor_execution_digest"]
    finally:
        fixture.doCleanups()


def test_gpu_actor_program_and_epoch_inputs_remain_legacy_exact():
    fixture = promotion_fixture.TheKeepBuildsAProductionCompleteAnchor()
    fixture.setUp()
    try:
        original_epoch = run.archive.epoch_for
        with mock.patch.object(run.archive, "epoch_for", wraps=original_epoch) as epoch:
            rc, _calls, planners, _scratch, log = fixture._run_one_keep()
        assert rc == 0, log
        assert "baseline_scope" not in run.status.read(fixture.store)
        epoch.assert_called_once_with(
            anchor_commit=fixture.tip, build_recipe=run.build_recipe.HOUSE_GPU_RECIPE.to_dict())
        assert planners[0].contexts[0]["program"] == run.loop.PROGRAM.read_text(encoding="utf-8")
    finally:
        fixture.doCleanups()
