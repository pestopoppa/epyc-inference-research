"""Enrolled inputs reach the existing CLI; no hardware, models or providers run."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from . import campaign, campaign_cli, legacy_targets as lt, planned_serving, run
from .test_campaign import _manifest, _registry, _target
from .test_glm_frozen_requests import _canonical_launch, _manifest as _prompts, _request
from . import test_promotion_targets as promotion_fixture


def _resolved(*, backend="cpu", seed=False, changes=None, missing=False):
    _, launch = _canonical_launch(18311)
    row = _target("selected", backend=backend, context=launch.template.ctx,
                  concurrency=launch.template.np)
    row.update(speculation=launch.capability.speculation,
               env={key: value for key, value in launch.launch_env if key != "LD_LIBRARY_PATH"})
    row.update(changes or {})
    registry = _registry()
    registry["model"]["model-a"].update(path=launch.model.path, sha256=launch.model.sha256)
    if missing:
        del registry["model"]["model-a"]
    declaration = _manifest(**{"seeds" if seed else "production": [row]})
    resolved = campaign.resolve_manifest(campaign.CampaignManifest.from_dict(declaration),
                                         registry_snapshot=registry)
    return resolved, launch


def _argv(tmp_path, resolved, launch, *, cpu, envelope=False):
    path = tmp_path / "resolved.json"
    body = campaign_cli.build_output(resolved, verify_artifacts=False) if envelope else resolved.to_dict()
    path.write_text(json.dumps(body))
    argv = ["--worktree", str(tmp_path / "source"), "--anchor-build", launch.build_dir,
            "--store", str(tmp_path / "store"), "--resolved-campaign", str(path),
            "--target-id", "selected", "--dry-run"]
    if cpu:
        recipe_path, prompts_path = tmp_path / "launch.json", tmp_path / "prompts.json"
        recipe_path.write_text(json.dumps(launch.to_dict()))
        prompts_path.write_text(json.dumps(_prompts(_request()).to_dict()))
        argv += ["--cpu-serving-launch", str(recipe_path), "--frozen-prompts", str(prompts_path),
                 "--experimental-branch", "ak/experimental/connector"]
    return argv


def _forbid_execution(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("selection reached execution or a provider")
    for module, names in ((run.claim, ("hold", "hold_cpu")),
                          (run.pool, ("provision", "drive")),
                          (run.gates, ("compiles", "op_correctness")),
                          (run.actors, ("AgentPlanner", "AgentCritic"))):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)


@pytest.mark.parametrize("backend,seed,envelope", [
    ("cpu", False, False), ("cpu", True, True), ("gpu", False, True), ("gpu", True, False),
])
def test_actual_main_selects_target_and_retains_existing_dry_run(
        tmp_path, monkeypatch, capsys, backend, seed, envelope):
    resolved, launch = _resolved(backend=backend, seed=seed)
    argv = _argv(tmp_path, resolved, launch, cpu=backend == "cpu", envelope=envelope)
    _forbid_execution(monkeypatch)
    startups, censuses = [], []

    def startup(**kwargs):
        startups.append(kwargs)
        return "a" * 40

    def census(model):
        censuses.append(model)
        return SimpleNamespace(n_embd=4096, dominant_quant="Q4_K")

    monkeypatch.setattr(run.champion, "verify_startup", startup)
    monkeypatch.setattr(run, "_git", lambda *_args: "a" * 40)
    monkeypatch.setattr(run.workload_contract, "read_census", census if backend == "cpu"
                        else lambda *_args: pytest.fail("GPU bypassed production workload check"))
    monkeypatch.setattr(run.workload_contract, "verify_workload", census if backend == "gpu"
                        else lambda *_args: pytest.fail("CPU used GPU workload check"))
    monkeypatch.setattr(run, "noise_floor_pct", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run.actors, "backend_for", lambda *_args: SimpleNamespace(describe=lambda: "fixture"))
    assert run.main(argv) == 0
    assert censuses == [Path(launch.model.path)]
    assert startups == [{"worktree": tmp_path / "source",
                         "branch": "ak/experimental/connector" if backend == "cpu" else run.champion.CANONICAL_BRANCH,
                         "anchor_build": Path(launch.build_dir), "allow_unverified_anchor": False,
                         "experimental_identity": backend == "cpu"}]
    output = capsys.readouterr().out
    assert ("cpu_serving_selected_workload" if backend == "cpu" else "legacy_gpu_screen") in output
    assert "DRY RUN" in output
    assert resolved.targets[0].enrolled_as == (("seed",) if seed else ("production",))


@pytest.mark.parametrize("case", ["unknown", "ambiguous", "missing", "cpu_as_gpu", "gpu_as_cpu",
    "both", "model", "flag_pair", "no_model", "cpu_context", "cpu_build", "cpu_request"])
def test_cli_refuses_bad_selection_before_startup_or_provider(tmp_path, monkeypatch, capsys, case):
    backend = "gpu" if case == "gpu_as_cpu" else "both" if case == "both" else "cpu"
    resolved, launch = _resolved(backend=backend, missing=case == "missing",
                                changes={"context": 16384} if case == "cpu_context" else None)
    if case == "ambiguous":
        resolved = replace(resolved, targets=resolved.targets + resolved.targets)
    argv = _argv(tmp_path, resolved, launch, cpu=case != "cpu_as_gpu")
    if case == "unknown":
        argv[argv.index("selected")] = "not-enrolled"
    elif case == "model":
        argv += ["--model", "/foreign/model.gguf"]
    elif case == "flag_pair":
        index = argv.index("--target-id")
        del argv[index:index + 2]
    elif case == "no_model":
        argv = ["--worktree", "/source", "--anchor-build", "/build", "--store", str(tmp_path)]
    elif case == "cpu_build":
        argv[argv.index("--anchor-build") + 1] = "/different-build"
    elif case == "cpu_request":
        request = _request()
        request["n_predict"] = 1
        (tmp_path / "prompts.json").write_text(json.dumps(_prompts(request).to_dict()))
    _forbid_execution(monkeypatch)
    monkeypatch.setattr(run.champion, "verify_startup", lambda **_kwargs: pytest.fail("startup before refusal"))
    monkeypatch.setattr(run.actors, "backend_for", lambda *_args: pytest.fail("provider setup before refusal"))
    with pytest.raises((SystemExit, planned_serving.PlannedServingError)) as error:
        run.main(argv)
    if isinstance(error.value, SystemExit):
        assert error.value.code == 2
        assert capsys.readouterr().err


@pytest.mark.parametrize("field,value", [
    ("context", 1234), ("concurrency", 3), ("speculation", "none"), ("env", {"UNDECLARED": "1"}),
])
def test_cpu_workload_differences_are_not_silently_selected(field, value):
    resolved, launch = _resolved(changes={field: value})
    selected = lt.select_target(resolved, "selected", cpu_serving=True)
    with pytest.raises(lt.TargetSelectionRefused, match=field if field != "env" else "environment"):
        lt.validate_cpu_workload(selected, launch)


def test_model_digest_and_drafter_differ_even_when_path_is_unchanged():
    resolved, launch = _resolved(seed=True)
    selected = lt.select_target(resolved, "selected", cpu_serving=True)
    changed = replace(launch, model=replace(launch.model, sha256="f" * 64))
    with pytest.raises(lt.TargetSelectionRefused, match="model"):
        lt.validate_cpu_workload(selected, changed)
    changed = replace(launch, drafter=replace(launch.model, role="drafter"))
    with pytest.raises(lt.TargetSelectionRefused, match="drafter"):
        lt.validate_cpu_workload(selected, changed)


def test_alias_is_selected_once_and_original_build_is_not_relabelled():
    resolved, launch = _resolved(seed=True)
    target = replace(resolved.targets[0], target_ids=("selected", "alias"))
    resolved = replace(resolved, targets=(target,))
    assert lt.select_target(resolved, "alias", cpu_serving=True) == target
    original_build = target.execution.build
    assert original_build.path != launch.executable.path
    lt.validate_cpu_workload(target, launch)
    assert target.execution.build == original_build


def test_glm_original_mtp_type_joins_owning_speculation_enum():
    resolved, launch = _resolved(seed=True)
    assert launch.template.spec_decode["type"] == "draft-mtp"
    assert launch.capability.speculation == "self_draft"
    assert resolved.targets[0].execution.speculation == "self_draft"
    lt.validate_cpu_workload(resolved.targets[0], launch)


def test_same_file_model_symlink_is_not_a_different_selected_model(tmp_path):
    resolved, _launch = _resolved()
    original = tmp_path / "model.gguf"
    original.write_bytes(b"not a model, no byte verification requested")
    link = tmp_path / "model-link.gguf"
    link.symlink_to(original)
    row = _target("selected", backend="gpu")
    registry = _registry()
    registry["model"]["model-a"]["path"] = str(original)
    resolved = campaign.resolve_manifest(campaign.CampaignManifest.from_dict(
        _manifest(production=[row])), registry_snapshot=registry)
    assert lt.select_target(resolved, "selected", cpu_serving=False, model=link) == resolved.targets[0]


def test_selected_identity_survives_existing_pool_context_status_output_and_epoch():
    """Real legacy pool/git/archive; existing fixture doubles all hardware/providers."""
    fixture = promotion_fixture.TheKeepBuildsAProductionCompleteAnchor()
    fixture.setUp()
    try:
        registry = _registry()
        model = fixture.root / f"{run.bench.MEASURED_FLOOR_MODEL_STEM}.gguf"
        registry["model"]["model-a"]["path"] = str(model)
        resolved = campaign.resolve_manifest(campaign.CampaignManifest.from_dict(
            _manifest(seeds=[_target("selected", backend="gpu")])), registry_snapshot=registry)
        path = fixture.root / "selected.json"
        path.write_text(json.dumps(resolved.to_dict()))
        original_main = run.main

        def selected_main(argv):
            return original_main([*argv, "--resolved-campaign", str(path), "--target-id", "selected",
                                  "--out", str(fixture.root / "result")])

        with mock.patch.object(run, "main", selected_main), \
                mock.patch.object(run.archive, "epoch_for", wraps=run.archive.epoch_for) as epoch:
            rc, _calls, planners, _scratch, log = fixture._run_one_keep()
        assert rc == 0, log
        result = json.loads((fixture.root / "result/loop-run.json").read_text())
        status = run.status.read(fixture.store)
        target = status["target"]
        assert target == result["target"] == planners[0].contexts[0]["target"]["enrollment"]
        assert target["original_target"] == resolved.targets[0].to_dict()
        assert target["scope"] == "legacy_gpu_screen"
        assert target["campaign_id"] == resolved.campaign_id
        assert target["request_id"] == resolved.request_id
        assert target["selected_id"] == "selected"
        assert target["manifest_digest"] == resolved.manifest_digest
        assert epoch.call_args.kwargs["host_state"]["enrolled_manifest_digest"] == resolved.manifest_digest
        assert len(epoch.call_args.kwargs["host_state"]["enrolled_target_digest"]) == 64
        assert result["epoch"] == status["epoch_sha256"]
        assert result["epoch"] != run.archive.epoch_for(
            anchor_commit=fixture.tip, build_recipe=run.build_recipe.HOUSE_GPU_RECIPE.to_dict())
        assert "NOT enrolled serving recipe" in planners[0].contexts[0]["target"]["scope"]
    finally:
        fixture.doCleanups()
