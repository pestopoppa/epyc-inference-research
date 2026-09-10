"""Generated enrolled inputs reach the existing owner; no hardware or builds."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import campaign, claim, legacy_targets, run, serial_roster, serial_run as sr, resolved_recipe as rr
from .test_campaign import _manifest, _registry, _target
from .test_glm_frozen_requests import _recipe, _manifest as _prompts, _request
from .test_legacy_targets import _forbid_execution
from .test_resolved_recipe import _artifacts, _policy
from .test_serial_run import CHILD


def _inputs(tmp_path, *, backends=("cpu", "gpu"), unowned=False, missing=False, experimental_gpu=False):
    registry, production, seeds, owners = _registry(), [], [], {}
    for index, backend in enumerate(backends):
        name = f"{backend}-{index}"
        root = tmp_path / name
        root.mkdir()
        build = root / "original-build"
        template = _recipe()
        if backend == "gpu":
            template = replace(template, device="ROCm0", ngl=99, cpu_list="0-7", threads=8)
        command = template.server_argv(build, 18311)[3:]
        template = rr.canonical_recipe_projection(name=template.name, command_argv=command,
            topology_prefix=["taskset", "-c", template.cpu_list], n_predict=512,
            temperature=0.0, top_k=1)
        launch = rr.resolve_canonical_launch(template, build_dir=build, command_argv=command,
            topology_prefix=["taskset", "-c", template.cpu_list],
            launch_environment={"LD_LIBRARY_PATH": str(build / "bin")},
            artifact_identities=_artifacts(template, build=build), backend=backend,
            environment_policy=_policy(), port=18311, runtime_binary_dir=str(build / "bin"),
            runtime_ld_paths=(str(build / "bin"),),
            provenance={"export_sha256": "a" * 64, "instance_mode": "full", "source:fixture": "b" * 64})
        recipe_path, prompt_path = root / "launch.json", root / "prompts.json"
        recipe_path.write_text(json.dumps(launch.to_dict()))
        prompt_path.write_text(json.dumps(_prompts(_request()).to_dict()))
        registry["recipe"][name] = {**registry["recipe"]["recipe-a"], "ref": name,
            "path": str(recipe_path), "sha256": hashlib.sha256(recipe_path.read_bytes()).hexdigest()}
        registry["model"]["model-a"].update(path=launch.model.path, sha256=launch.model.sha256)
        target = _target(name, backend=backend, recipe=name, context=template.ctx, concurrency=template.np)
        target.update(speculation=launch.capability.speculation,
                      env={k: v for k, v in launch.launch_env if k != "LD_LIBRARY_PATH"})
        (seeds if backend == "cpu" or experimental_gpu else production).append(target)
        owners[name] = {"worktree": str(root / "source"), "anchor_build": str(build),
            "branch": ("ak/experimental/roster" if backend == "cpu" or experimental_gpu
                       else sr.champion.CANONICAL_BRANCH),
            "frozen_prompts": str(prompt_path), "calibrate_serving": 2}
    if unowned:
        production.append(_target("unowned", model="model-b"))
    if missing:
        seeds.append(_target("offdisk", model="missing-model"))
    declaration = _manifest(production=production, seeds=seeds)
    declaration["resources"].update(cpu_logical=list(range(192)), gpu_ids=[claim.DEVICE_ID])
    resolved = campaign.resolve_manifest(
        campaign.CampaignManifest.from_dict(declaration),
        registry_snapshot=registry)
    resolved_path, owned_path = tmp_path / "resolved.json", tmp_path / "owned.json"
    resolved_path.write_text(json.dumps(resolved.to_dict()))
    owned_path.write_text(json.dumps(owners))
    state = tmp_path / "router"
    argv = ["--resolved-campaign", str(resolved_path), "--owned-targets", str(owned_path),
            "--state-dir", str(state), "--batch-iterations", "1", "--rounds", "2"]
    return resolved, owners, argv


def _build(argv):
    return serial_roster.build_targets(Path(sr.option(argv, "--resolved-campaign")),
        Path(sr.option(argv, "--owned-targets")),
        target_root=Path(sr.option(argv, "--state-dir")) / "targets")


def test_ready_production_and_candidate_roster_preserves_original_inputs(tmp_path):
    resolved, owners, argv = _inputs(tmp_path, unowned=True, missing=True)
    targets, skipped, cpus = _build(argv)
    assert cpus == tuple(range(192))
    assert len(targets) == 2
    assert {tuple(row["target_ids"]): row["reason"] for row in skipped} == {
        ("unowned",): "no owned source/anchor/request inputs",
        ("offdisk",): "missing_artifact: model:missing"}
    for args in targets:
        alias = sr.option(args, "--target-id")
        original = next(t for t in resolved.targets if alias in t.target_ids)
        backend = original.execution.backend
        assert sr.option(args, "--model") == original.execution.model.path
        assert sr.option(args, f"--{backend}-serving-launch") == original.execution.recipe.path
        assert sr.option(args, "--anchor-build") == owners[alias]["anchor_build"]
        assert sr.option(args, "--frozen-prompts") == owners[alias]["frozen_prompts"]
        assert sr.option(args, "--store").endswith(sr._digest(original.to_dict()) + "/store")
        assert sr.input_binding(args)["documents"][f"--{backend}-serving-launch"] == original.execution.recipe.sha256
        assert sr.option(args, "--planner-model") == dict(resolved.actors)["planner"]
    assert not (tmp_path / "router").exists()


@pytest.mark.parametrize("backend,experimental", [("cpu", True), ("gpu", False), ("gpu", True)])
def test_generated_dry_run_reaches_actual_owner_without_side_effects(
        tmp_path, monkeypatch, capsys, backend, experimental):
    _, owners, argv = _inputs(tmp_path, backends=(backend,), unowned=True,
                             experimental_gpu=experimental)
    _forbid_execution(monkeypatch)
    starts = []
    monkeypatch.setattr(run.champion, "verify_startup", lambda **kw: starts.append(kw) or "a" * 40)
    monkeypatch.setattr(run, "_git", lambda *_a: "a" * 40)
    monkeypatch.setattr(run.workload_contract, "read_census",
        lambda *_a: SimpleNamespace(n_embd=4096, dominant_quant="Q4_K"))
    monkeypatch.setattr(run.actors, "backend_for", lambda *_a: SimpleNamespace(describe=lambda: "fixture"))
    assert sr.main(argv + ["--dry-run"]) == 0
    assert len(starts) == 1 and starts[0]["experimental_identity"] is experimental
    assert starts[0]["anchor_build"] == Path(owners[f"{backend}-0"]["anchor_build"])
    assert "no owned source/anchor/request inputs" in capsys.readouterr().out
    assert not (tmp_path / "router").exists()


@pytest.mark.parametrize("experimental_gpu", [False, True])
def test_generated_roster_drives_actual_children_and_completed_restart(tmp_path, monkeypatch, experimental_gpu):
    _, _, argv = _inputs(tmp_path, experimental_gpu=experimental_gpu)
    child = tmp_path / "tiny.py"
    child.write_text(CHILD.replace('"pid": __import__(\'os\').getpid()',
        '"pid": __import__(\'os\').getpid(), "affinity": sorted(__import__(\'os\').sched_getaffinity(0))')
        .replace('selected = legacy_targets',
                 'gpu = sr.option(argv, "--gpu-serving-launch") is not None\n'
                 'experimental = cpu or (gpu and sr.option(argv, "--experimental-branch") is not None)\n'
                 'selected = legacy_targets')
        .replace('if cpu else "legacy_gpu_screen"',
                 'if cpu else "gpu_serving_selected_workload" if gpu else "legacy_gpu_screen"')
        .replace('if cpu else sr.champion.CANONICAL_BRANCH', 'if experimental else sr.champion.CANONICAL_BRANCH')
        .replace('None if cpu else anchor', 'None if experimental else anchor')
        .replace('None if cpu else "a" * 40', 'None if experimental else "a" * 40'))
    monkeypatch.setattr(sr, "_child_command", lambda args: [sr.sys.executable, str(child), *args])
    monkeypatch.setenv("PYTHONPATH", str(Path(sr.__file__).resolve().parents[4]))
    assert sr.main(argv) == 0
    state = tmp_path / "router"
    seen = [json.loads(line) for line in (state / "seen.jsonl").read_text().splitlines()]
    # Distinct recipe artifacts keep these as two distinct enrolled targets.
    assert len(seen) == 4
    assert all(set(row["affinity"]) == set(os.sched_getaffinity(0)) & set(range(192)) for row in seen)
    assert sr.option(seen[2]["argv"], "--resume-run")
    assert [sr.option(row["argv"], "--target-id") for row in seen[:2]] == [
        sr.option(row["argv"], "--target-id") for row in seen[2:]]
    for row in seen[2:]:
        assert sr.option(row["argv"], "--cpu-calibrate-serving") is None
        assert sr.option(row["argv"], "--gpu-calibrate-serving") is None
    assert sr.main(argv) == 0
    assert len((state / "seen.jsonl").read_text().splitlines()) == 4


def test_identical_owned_aliases_schedule_once_and_missing_ownership_is_not_success(tmp_path):
    resolved, owners, argv = _inputs(tmp_path, backends=("cpu",))
    first = resolved.targets[0]
    resolved = replace(resolved, targets=(replace(first, target_ids=first.target_ids + ("alias",)),))
    owners["alias"] = owners[first.target_ids[0]]
    Path(sr.option(argv, "--resolved-campaign")).write_text(json.dumps(resolved.to_dict()))
    Path(sr.option(argv, "--owned-targets")).write_text(json.dumps(owners))
    targets, skipped, _cpus = _build(argv)
    assert len(targets) == 1 and not skipped
    assert sr.option(targets[0], "--target-id") == first.target_ids[0]
    Path(sr.option(argv, "--owned-targets")).write_text("{}")
    with pytest.raises(sr.SerialRefused, match="1–64"):
        _build(argv)


def test_actual_target_args_dry_run_refuses_calibration_before_owner(tmp_path, monkeypatch):
    _, _, argv = _inputs(tmp_path, backends=("cpu",))
    targets, _, _ = _build(argv)
    args_path = tmp_path / "args.json"
    args_path.write_text(json.dumps(targets[0] + ["--calibrate-surface=5"]))
    monkeypatch.setattr(run, "main", lambda *_a: pytest.fail("calibration reached owner"))
    with pytest.raises(SystemExit) as caught:
        sr.main(["--target-args", str(args_path), "--batch-iterations", "1",
                 "--state-dir", str(tmp_path / "state"), "--dry-run"])
    assert caught.value.code == 2


@pytest.mark.parametrize("field,value", [("cpu_logical", [190, 191]), ("gpu_ids", ["unowned-gpu"])])
def test_roster_rejects_launch_outside_declared_resources(tmp_path, field, value):
    resolved, _, argv = _inputs(tmp_path)
    resolved = replace(resolved, resources=replace(resolved.resources, **{field: tuple(value)}))
    Path(sr.option(argv, "--resolved-campaign")).write_text(json.dumps(resolved.to_dict()))
    with pytest.raises(legacy_targets.TargetSelectionRefused, match="resources|installed ROCm0"):
        _build(argv)


@pytest.mark.parametrize("case", ["unknown_alias", "alias_conflict", "recipe_changed", "anchor_changed",
                                 "requests_changed", "foreign_common", "overlap", "production_branch"])
def test_invalid_generated_inputs_refuse_before_launch(tmp_path, monkeypatch, case):
    resolved, owners, argv = _inputs(tmp_path, backends=("cpu", "cpu"))
    path = Path(sr.option(argv, "--owned-targets"))
    if case == "unknown_alias":
        owners["foreign"] = owners.pop("cpu-0")
    elif case == "alias_conflict":
        first = resolved.targets[0]
        resolved = replace(resolved, targets=(replace(first, target_ids=first.target_ids + ("alias",)),
                                              *resolved.targets[1:]))
        Path(sr.option(argv, "--resolved-campaign")).write_text(json.dumps(resolved.to_dict()))
        owners["alias"] = {**owners[first.target_ids[0]], "branch": "ak/other"}
    elif case == "recipe_changed":
        Path(resolved.targets[0].execution.recipe.path).write_text("{}")
    elif case == "anchor_changed":
        owners["cpu-0"]["anchor_build"] = "/different-original"
    elif case == "requests_changed":
        request = _request()
        request["n_predict"] = 1
        Path(owners["cpu-0"]["frozen_prompts"]).write_text(json.dumps(_prompts(request).to_dict()))
    elif case == "foreign_common":
        common = tmp_path / "common.json"
        common.write_text(json.dumps(["--model", "/foreign-model"]))
        argv += ["--common-args", str(common)]
    elif case == "overlap":
        owners["cpu-1"]["worktree"] = owners["cpu-0"]["worktree"]
    elif case == "production_branch":
        owners["cpu-0"]["branch"] = "production-consolidated-v9"
    path.write_text(json.dumps(owners))
    monkeypatch.setattr(sr, "_drive", lambda *_a, **_kw: pytest.fail("invalid roster reached launch"))
    with pytest.raises(SystemExit) as caught:
        sr.main(argv)
    assert caught.value.code == 2
    assert not (tmp_path / "router").exists()
