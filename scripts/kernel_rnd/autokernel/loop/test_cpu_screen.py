"""Existing source/pool/serving owners; hardware/provider edges are synthetic."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from . import campaign, cpu_profile, cpu_screen, dispatch_guard, gates, loop, pool, resolved_recipe as rr
from . import run, serial_run as sr, serving
from .test_campaign import _manifest as campaign_manifest, _registry, _target
from .test_glm_frozen_requests import _canonical_launch, _manifest, _request
from . import test_promotion_targets as fixtures
from .test_shared_history import _record


def test_retained_critic_accepts_only_exact_screened_candidate():
    hypothesis = loop.Hypothesis("mechanism", "statement", "falsifier", "surface", "symbol")
    critic = cpu_screen.RetainedCritic({"hypothesis": hypothesis.to_dict(),
                                       "paths": ["ggml/src/ggml-cpu/file.c"]})
    assert critic.review_hypothesis(hypothesis, {}).accepted
    assert critic.review_patch(hypothesis, ("ggml/src/ggml-cpu/file.c",), {}).accepted
    changed = loop.Hypothesis("other", "statement", "falsifier", "surface", "symbol")
    assert not critic.review_hypothesis(changed, {}).accepted
    assert not critic.review_patch(hypothesis, ("other.c",), {}).accepted


def test_reduced_subfloor_positive_creates_confirmation_debt_not_a_keep():
    """A valid sub-floor positive is provisional until full-target confirmation."""
    hypothesis = loop.Hypothesis(
        "reduced-subfloor", "small positive on reduced CPU scope", "floor miss",
        "serving:glm53", "ggml/src/ggml-cpu/file.c")
    paths = ("ggml/src/ggml-cpu/file.c",)
    planner = mock.Mock()
    planner.propose.return_value = hypothesis
    planner.author.return_value = paths
    critic = cpu_screen.RetainedCritic({
        "hypothesis": hypothesis.to_dict(), "paths": list(paths)})
    comparison = run.bench.Comparison(
        surface="serving:glm53", anchor_samples=[100.0] * 5,
        candidate_samples=[100.5] * 5, effect=0.005,
        estimator="matched_process_v2", pairs=5, noise_floor_pct=1.0,
        residency={}, anchor_drift_pct=0.0, calibrated=True)
    commit = mock.Mock(side_effect=loop.ConfirmVetoed(
        "reduced positive retained; original full-target confirmation pending"))

    outcome = loop.iterate(
        planner=planner, critic=critic, context={},
        measure=lambda _hypothesis, _paths: comparison,
        gate=lambda _hypothesis, _paths: (True, [gates.Verdict("oracle", True)]),
        commit=commit, accumulate_valid_positive=True)

    assert outcome.status == "keep_candidate"
    assert outcome.hypothesis == hypothesis
    assert outcome.comparison is comparison
    assert "full-target confirmation pending" in " ".join(outcome.reasons)
    commit.assert_called_once_with(hypothesis, paths, comparison)


def test_reduced_subfloor_archive_requires_calibrated_stationary_positive(tmp_path):
    hypothesis = loop.Hypothesis(
        "reduced-subfloor", "small positive on reduced CPU scope", "floor miss",
        "serving:glm53", "ggml/src/ggml-cpu/file.c")
    worker = SimpleNamespace(worktree=tmp_path, name="lane0")
    full = SimpleNamespace(to_dict=lambda: {"full": "target"})
    capture = {"inputs": {"resolved_arms": {"anchor": {"a": 1},
                                                 "candidate": {"b": 2}}}}

    def comparison(effect=0.005, calibrated=True, drifting=False):
        value = mock.Mock()
        value.effect, value.decisive = effect, False if calibrated else None
        value.noise_floor_pct, value.drifting = 1.0, drifting
        value.to_dict.return_value = {"belief_capture": capture,
                                      "request_digest": "request", "noise_floor_pct": 1.0}
        return value

    with mock.patch.object(cpu_screen.archive, "retain_patch", return_value=None) as retained:
        with pytest.raises(cpu_screen.ScreenRefused, match="no original source patch"):
            cpu_screen.retain_candidate(store_root=tmp_path, origin_batch=tmp_path,
                worker=worker, target={"id": "glm"}, hypothesis=hypothesis,
                paths=["ggml/src/ggml-cpu/file.c"], full_target=full,
                comparison=comparison())
        retained.assert_called_once()
        for rejected in (comparison(effect=-0.005), comparison(calibrated=False),
                         comparison(drifting=True)):
            with pytest.raises(cpu_screen.ScreenRefused, match="valid positive"):
                cpu_screen.retain_candidate(store_root=tmp_path, origin_batch=tmp_path,
                    worker=worker, target={"id": "glm"}, hypothesis=hypothesis,
                    paths=["ggml/src/ggml-cpu/file.c"], full_target=full,
                    comparison=rejected)
        retained.assert_called_once()


def test_recovery_reads_original_null_and_patch_without_regrading_or_writing(tmp_path):
    store = tmp_path / "store"
    original = tmp_path / "old-batch"
    original.mkdir()
    worker_root = tmp_path / "workers"
    build_root = tmp_path / "builds"
    capture_dir = store / "serving-beliefs"
    source_dir = capture_dir / "sources"
    source_dir.mkdir(parents=True)
    native = source_dir / "capture.json"
    native.write_bytes(b"temporary native observation")
    receipt_file = capture_dir / "capture.json"
    receipt_file.write_text(json.dumps({"native_reference": {
        "path": "sources/capture.json", "sha256": hashlib.sha256(native.read_bytes()).hexdigest()}}))
    patch_dir = store / "patches"
    patch_dir.mkdir()
    patch = patch_dir / "source.patch"
    patch.write_bytes(b"diff --git a/ggml/src/kernel.c b/ggml/src/kernel.c\n")
    metadata_path = patch_dir / "source.json"
    metadata_path.write_text(json.dumps({
        "schema": "epyc.autokernel.source_patch_archive.v1", "original_head": "a" * 40,
        "mechanism_id": "recover-me", "worktree": str(worker_root / "lane0"),
        "patch_file": patch.name, "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest()}))
    continuation_path = original / "loop-continuation.json"
    receipt = {"terminal": "complete", "iterations_completed": 1,
        "outcome_counts": {"measured_null": 1}, "cpu_screen": {
            "scope": "half", "candidate": None, "full_execution_digest": "f" * 64},
        "current_anchor": {"commit": "a" * 40}, "selected_target": {"id": "glm"},
        "result_file": "loop-run.json", "input_argv": ["--store", str(store)],
        "last_outcome_reference": {"mechanism_id": "recover-me", "serving_receipt": {
            "path": str(receipt_file),
            "sha256": hashlib.sha256(receipt_file.read_bytes()).hexdigest()}}}
    comparison = {"decisive": False, "effect": 0.005, "noise_floor_pct": 0.7,
        "request_digest": "request", "belief_capture": {"capture_id": "capture",
            "native_sha256": "native", "inputs": {"resolved_arms": {
                "anchor": {"build_dir": str(store / "anchor")},
                "candidate": {"build_dir": str(build_root / "lane0")}}}}}
    outcome = {"status": "measured_null", "mechanism_id": "recover-me",
        "statement": "source change", "falsifier": "full comparison",
        "target_surface": "ggml/src/kernel.c", "target_symbol": "kernel",
        "comparison": comparison,
        "candidate_diff_sha256": hashlib.sha256(
            dispatch_guard.normalized_diff(patch.read_bytes()).encode()).hexdigest()}
    native.write_text(json.dumps({"mechanism_id": "recover-me", "comparison": comparison}))
    receipt_file.write_text(json.dumps({"native_reference": {
        "path": "sources/capture.json", "sha256": hashlib.sha256(native.read_bytes()).hexdigest()}}))
    receipt["last_outcome_reference"]["serving_receipt"]["sha256"] = hashlib.sha256(
        receipt_file.read_bytes()).hexdigest()
    (original / "loop-run.json").write_text(json.dumps(
        {"continuation": receipt, "iterations": [outcome]}))
    full = SimpleNamespace(execution_digest="f" * 64, to_dict=lambda: {"full": True})
    def git_result(_repo, *args, **_kwargs):
        return "a" * 40 if args[0] == "rev-parse" else patch.read_text()

    with mock.patch.object(sr, "load_completed", return_value=(receipt, "old-receipt-sha")), \
            mock.patch.object(cpu_screen.archive, "_verified_repo"), \
            mock.patch.object(cpu_screen.archive, "_git", side_effect=git_result):
        recovered = cpu_screen.recovery_from(continuation_path, metadata_path,
            full_target=full, selected_target={"id": "glm"}, request_digest="request",
            original_head="a" * 40, worker_root=worker_root, worker_build_root=build_root)
        assert recovered["screen_assessment"]["effect"] == 0.005
        assert recovered["paths"] == ["ggml/src/kernel.c"]
        assert recovered["recovery_origin"]["sha256"] == "old-receipt-sha"
        comparison["effect"] = -0.005
        (original / "loop-run.json").write_text(json.dumps(
            {"continuation": receipt, "iterations": [outcome]}))
        with pytest.raises(cpu_screen.ScreenRefused, match="sub-floor positive"):
            cpu_screen.recovery_from(continuation_path, metadata_path,
                full_target=full, selected_target={"id": "glm"}, request_digest="request",
                original_head="a" * 40, worker_root=worker_root, worker_build_root=build_root)
        comparison["effect"] = 0.006
        (original / "loop-run.json").write_text(json.dumps(
            {"continuation": receipt, "iterations": [outcome]}))
        with pytest.raises(cpu_screen.ScreenRefused, match="native comparison"):
            cpu_screen.recovery_from(continuation_path, metadata_path,
                full_target=full, selected_target={"id": "glm"}, request_digest="request",
                original_head="a" * 40, worker_root=worker_root, worker_build_root=build_root)


@pytest.mark.parametrize("scope,threads,regions", [("quarter", 24, ("q0",)),
                                                   ("half", 48, ("q0", "q1"))])
def test_original_glm_common_scope_preserves_request_and_full_target(scope, threads, regions):
    template, full = _canonical_launch(18311)
    before = full.to_dict()
    result = cpu_screen.prepare_launch(full, scope, range(96))
    reduced = result["launch"]
    assert full.to_dict() == before
    assert reduced.template.threads == threads and result["regions"] == regions
    assert reduced.template.spec_decode == full.template.spec_decode
    assert reduced.model == full.model and reduced.executable == full.executable
    assert reduced.dsos == full.dsos and reduced.launch_env == full.launch_env
    assert reduced.topology_prefix[:3] == full.topology_prefix[:3]
    prompts = _manifest(_request())
    assert prompts.requests(("glm-fixed2029",), reduced.template) == prompts.requests(
        ("glm-fixed2029",), template)
    assert reduced.execution_digest != full.execution_digest
    assert result["region_fraction"] == len(regions) / 4


def _inputs(fixture):
    _, source = _canonical_launch(18311)
    model = fixture.root / f"{run.bench.MEASURED_FLOOR_MODEL_STEM}.gguf"
    branch = "ak/experimental/reduced-source"
    run._git(fixture.repo, "checkout", "-b", branch)

    def install(build):
        directory = Path(build) / "bin"
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("llama-server", "libggml-cpu.so"):
            (directory / name).write_bytes(b"synthetic-not-executed-" + Path(build).name.encode())

    install(fixture.startup_anchor)
    command = list(source.command_argv)
    command[0] = str(fixture.startup_anchor / "bin/llama-server")
    command[command.index("-m") + 1] = str(model)
    template = rr.canonical_recipe_projection(name="cpu-common-screen", command_argv=command,
        topology_prefix=source.topology_prefix, n_predict=512, temperature=0.0, top_k=1)

    def identity(role, path):
        return rr.ArtifactDigest(role, str(path), hashlib.sha256(path.read_bytes()).hexdigest()).to_dict()

    build = fixture.startup_anchor
    launch = rr.resolve_canonical_launch(template, build_dir=build, command_argv=command,
        topology_prefix=source.topology_prefix, launch_environment={"LD_LIBRARY_PATH": str(build / "bin")},
        artifact_identities={"model": identity("model", model), "drafter": None,
            "executable": identity("executable", build / "bin/llama-server"),
            "dsos": [identity("dso", build / "bin/libggml-cpu.so")]},
        backend="cpu", environment_policy=source.environment_policy, port=18311,
        runtime_binary_dir=str(build / "bin"), runtime_ld_paths=(str(build / "bin"),),
        provenance=dict(source.provenance))
    registry = _registry()
    registry["model"]["model-a"].update(path=str(model), sha256=launch.model.sha256)
    target = _target("selected", backend="cpu", context=template.ctx, concurrency=template.np)
    target.update(speculation=launch.capability.speculation, env={})
    declaration = campaign_manifest(seeds=[target], production=[])
    declaration["resources"].update(cpu_logical=list(range(96)), gpu_ids=[])
    resolved = campaign.resolve_manifest(campaign.CampaignManifest.from_dict(declaration),
                                         registry_snapshot=registry)
    prompts = _manifest(_request())
    options = ["--target-id", "selected", "--experimental-branch", branch, "--serving-pairs", "2"]
    for flag, name, body in (("--cpu-serving-launch", "cpu.json", launch.to_dict()),
                             ("--frozen-prompts", "prompts.json", prompts.to_dict()),
                             ("--resolved-campaign", "campaign.json", resolved.to_dict())):
        path = fixture.root / name
        path.write_text(json.dumps(body))
        options += [flag, str(path)]
    return launch, prompts, options, install


@pytest.mark.parametrize("confirm_gain", [True, False])
def test_actual_owner_reduced_positive_automatically_confirms_same_source_build(confirm_gain):
    fixture = fixtures.TheKeepBuildsAProductionCompleteAnchor()
    fixture.setUp()
    try:
        full, prompts, options, install = _inputs(fixture)
        requests = prompts.requests(("glm-fixed2029",), full.template)
        original_main = run.main
        measurements, claims, builds, oracle, contexts = [], [], [], [], []
        first_candidate = []
        _record(fixture.store, statement="remove redundant arithmetic in local vec_dot",
                research_scope={"model": {"path": full.model.path}, "backend": "cpu"})

        @contextmanager
        def hold(cpus):
            claims.append(cpus)
            yield {"device_id": "synthetic-cpu-region-owner"}

        def observe(recipe, build, port, *, resolved_recipe, frozen_requests, evidence=None, **_kw):
            assert frozen_requests == requests
            resolved_recipe.validate_launch(recipe, build, port)
            measurements.append((recipe.threads, recipe.cpu_list, str(build)))
            if evidence is not None:
                evidence.append({"window_start": 1.0, "window_end": 2.0,
                                 "backend": "cpu", "status": serving.RESIDENCY_NOT_APPLICABLE,
                                 "fixture": "synthetic window, not placement evidence"})
            candidate = Path(build).name == "lane0-build"
            return 11.0 if candidate and (recipe.threads == 24 or confirm_gain) else 10.0

        def invoke(original):
            base_compile, base_planner = gates.compiles, run.actors.AgentPlanner

            def compile_source(*args, **kwargs):
                builds.append((Path(args[1]), kwargs["cpu_list"]))
                verdict = base_compile(*args, **kwargs)
                install(args[1])
                return verdict

            def planner(*args, **kwargs):
                actor = base_planner(*args, **kwargs)
                propose = actor.propose

                def captured(context):
                    contexts.append(context)
                    return propose(context)
                actor.propose = captured
                return actor

            def check(build, **kwargs):
                oracle.append((Path(build), kwargs))
                if len(oracle) > 1:
                    assert kwargs["resolved_recipe"].template.cpu_list == "0-95"
                return gates.Verdict("oracle", True, "synthetic hardware edge")

            argv = original + options
            with mock.patch.object(gates, "compiles", compile_source), \
                    mock.patch.object(gates, "op_correctness", check), \
                    mock.patch.object(run.actors, "AgentPlanner", planner), \
                    mock.patch.object(run.claim, "hold_cpu", hold), \
                    mock.patch.object(run.claim, "hold", side_effect=AssertionError("GPU")), \
                    mock.patch.object(run.os, "sched_getaffinity", return_value={3}), \
                    mock.patch.object(run.os, "sched_setaffinity"), \
                    mock.patch.object(run.workload_contract, "read_census", return_value=SimpleNamespace(
                        n_embd=1536, dominant_quant="Q4_K")), \
                    mock.patch.object(cpu_profile, "profile_loop", side_effect=cpu_profile.CpuProfileRefused(
                        "test-only unavailable")), \
                    mock.patch.object(serving, "_measure_once", observe), \
                    mock.patch.object(pool, "prune_anchor_generations", return_value=pool.PruneReport("complete")):
                batch1 = fixture.root / "batch-000000"
                first = sr._batch_argv(argv, None, 1, batch1)
                assert sr.option(first, "--cpu-screen-scope") == "quarter"
                assert original_main(first) == 0
                first_receipt, sha = sr.load_completed(batch1 / "loop-continuation.json")
                assert first_receipt["outcome_counts"] == {"keep_candidate": 1}, (
                    batch1 / "loop-run.json").read_text()
                assert run._git(fixture.repo, "rev-parse", "HEAD") == fixture.tip
                candidate = cpu_screen.read_candidate(first_receipt["cpu_screen"]["candidate"])
                first_candidate.append(candidate)
                assert len(builds) == 1
                assert candidate["candidate_launch"]["executable"]["sha256"] == hashlib.sha256(
                    (fixture.root / "lane0-build/bin/llama-server").read_bytes()).hexdigest()
                prior = {"path": str(batch1 / "loop-continuation.json"), "sha256": sha}
                from .pipeline import Worker
                worker = Worker("lane0", fixture.root / "lane0", fixture.root / "lane0-build")
                reduced = cpu_screen.prepare_launch(full, "quarter", range(96))["launch"]
                binary = worker.build_dir / "bin/llama-server"
                original_bytes = binary.read_bytes()
                binary.write_bytes(original_bytes + b"changed")
                try:
                    with pytest.raises(cpu_screen.ScreenRefused, match="binary/DSO"):
                        cpu_screen.verify_candidate(candidate, worker, reduced)
                finally:
                    binary.write_bytes(original_bytes)
                with pytest.raises(cpu_screen.ScreenRefused, match="base/target/request"):
                    cpu_screen.confirmation_from(prior["path"], full_target=full,
                        selected_target=first_receipt["selected_target"], request_digest="0" * 64,
                        original_head=fixture.tip)
                with pytest.raises(cpu_screen.ScreenRefused, match="single-candidate"):
                    cpu_screen.preview_batch(argv, prior, batch_iterations=2)
                # A second alias on this branch (even a distinct detached
                # worktree/build root) must not supersede the pending source.
                alias_tree = fixture.root / "other-target"
                fixtures._sh(fixture.repo, "worktree", "add", "--detach", str(alias_tree), fixture.tip)
                alias = sr._without(argv, {"--target-id", "--worktree", "--worker-root", "--worker-build-root"})
                alias += ["--target-id", "alias", "--worktree", str(alias_tree),
                          "--worker-root", str(fixture.root / "other-workers"),
                          "--worker-build-root", str(fixture.root / "other-builds")]
                unrelated = sr._without(alias, {"--experimental-branch"}) + [
                    "--experimental-branch", "ak/experimental/independent"]
                debts = cpu_screen.pending_collisions([argv, alias, unrelated], {"0": prior})
                assert set(debts) == {1} and "pending full confirmation" in debts[1]
                assert not cpu_screen.pending_collisions([argv, alias], {})
                # Original selection is stable even after its history changed.
                with mock.patch.object(cpu_screen, "mechanism_hint", side_effect=AssertionError("reselected")):
                    assert sr._batch_argv(argv, None, 1, batch1) == first
                    second = sr._batch_argv(argv, prior, 1, fixture.root / "batch-000001")
                assert sr.option(second, "--cpu-confirm-from") == prior["path"]
                assert original_main(second) == 0
                second_receipt, _ = sr.load_completed(fixture.root / "batch-000001/loop-continuation.json")
                assert second_receipt["cpu_screen"]["scope"] == "full_confirmation"
                assert second_receipt["cpu_screen"]["candidate"] is None
                assert second_receipt["outcome_counts"] == {"kept" if confirm_gain else "measured_null": 1}, (
                    fixture.root / "batch-000001/loop-run.json").read_text()
                # Full confirmation does not rebuild its original candidate. A real
                # full keep still builds/promotes its new anchor through the old owner.
                assert sum(path.name == "lane0-build" for path, _cpus in builds) == 1
                assert len(contexts) == 1  # No second provider proposal/author.
                assert len(oracle) == 2
                assert claims == [",".join(map(str, range(24))), "0-95"]
                assert len([x for x in measurements if x[0] == 24]) == 6  # 2 A/A + 2 pairs
                assert all(cpus == "0-95" for t, cpus, _b in measurements if t == 48)
                assert (run._git(fixture.repo, "rev-parse", "HEAD") != fixture.tip) is confirm_gain
                return 0

        with mock.patch.object(run, "main", invoke):
            rc, _builds, _planners, _scratch, log = fixture._run_one_keep()
        assert rc == 0, log
        assert "both a/b arms" in contexts[0]["program"].lower()
        assert contexts[0]["cpu_screen"]["full_target"] == full.to_dict()
        rendered = run.actors.render_context(contexts[0])
        assert "full_transfer_target" in rendered and full.execution_digest in rendered
        assert "reduced null cannot globally retire" in rendered
        assert run._git(fixture.repo, "rev-parse", run.champion.CANONICAL_BRANCH) == fixture.tip
    finally:
        fixture.tearDown()


@pytest.mark.parametrize("statement,scope", [
    ("remove redundant SIMD arithmetic", "quarter"),
    ("change cache traffic via prefetch layout", "half"),
    ("improve thread synchronization barrier scaling", "full"),
])
def test_original_qualitative_hint_not_magnitude_or_global_null_retirement(tmp_path, statement, scope):
    _template, full = _canonical_launch(18311)
    _record(tmp_path, statement=statement, effect_fraction=-0.99,
            research_scope={"model": {"path": full.model.path}, "backend": "cpu"})
    before = (tmp_path / "experiments.db").read_bytes()
    hint = cpu_screen.mechanism_hint(tmp_path, full)
    assert hint["scope"] == scope and hint["statement"] == statement
    assert "not transfer evidence" in hint["basis"]
    assert not any("effect" in key for key in hint)
    assert (tmp_path / "experiments.db").read_bytes() == before


def test_absent_or_unreadable_hint_uses_nonpromotable_half_screen(tmp_path):
    _template, full = _canonical_launch(18311)
    assert cpu_screen.mechanism_hint(tmp_path / "missing", full)["scope"] == "half"
    _record(tmp_path, statement="remove SIMD instructions")
    assert cpu_screen.mechanism_hint(tmp_path, full)["scope"] == "half"


@pytest.mark.parametrize("mechanism_id,scope", [
    ("akm-moe-up-gate-safe-fusion", "half"),
    ("akm-q8-multirow-batched-dot", "half"),
    ("akm-barrier-spin-policy", "full"),
])
def test_sealed_recent_mechanism_routes_without_scanning_history(
        tmp_path, mechanism_id, scope):
    _template, full = _canonical_launch(18311)
    hint = cpu_screen.mechanism_hint(
        tmp_path / "missing", full,
        recent={"mechanism_id": mechanism_id, "status": "measured_null"})
    assert hint["scope"] == scope
    assert hint["mechanism_id"] == mechanism_id
    assert "continuation" in hint["basis"]


def test_scope_refuses_foreign_resources_and_incomplete_receipt():
    _template, full = _canonical_launch(18311)
    with pytest.raises(cpu_screen.ScreenRefused, match="outside"):
        cpu_screen.prepare_launch(full, "quarter", range(8))
    with pytest.raises(cpu_screen.ScreenRefused, match="routing fields"):
        cpu_screen.routing({"scope": "quarter"}, [])


def test_one_shot_recovery_routes_only_as_full_confirmation():
    argv = ["--cpu-serving-launch", "/tmp/full.json", "--resolved-campaign", "/tmp/campaign.json",
            "--iterations", "1", "--cpu-recover-from", "/tmp/original.json"]
    route = {"scope": "full_confirmation", "full_execution_digest": "a" * 64,
             "measured_execution_digest": "a" * 64, "candidate": None}
    assert cpu_screen.routing(route, argv) == route
    with pytest.raises(cpu_screen.ScreenRefused, match="differs"):
        cpu_screen.routing(route, argv + ["--cpu-confirm-from", "/tmp/other.json"])
