"""Generated enrolled inputs reach the existing owner; no hardware or builds."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import textwrap
from types import SimpleNamespace

import pytest

from . import campaign, claim, legacy_targets, run, serial_roster, serial_run as sr, resolved_recipe as rr
from .test_campaign import _manifest, _registry, _target
from .test_glm_frozen_requests import _recipe, _manifest as _prompts, _request
from .test_legacy_targets import _forbid_execution
from .test_resolved_recipe import _artifacts, _policy
from .test_serial_run import CHILD


def _confine_fixture_child(tmp_path, monkeypatch):
    """Keep simulated owner children inside the test runner's actual CPU claim."""
    wrapper = tmp_path / "taskset-within-test-affinity"
    wrapper.write_text(textwrap.dedent(f"""\
        #!{os.sys.executable}
        import os, sys
        args = sys.argv[1:]
        if len(args) < 3 or args[0] != "-c":
            raise SystemExit(64)
        requested = set()
        for part in args[1].split(","):
            bounds = part.split("-", 1)
            requested.update(range(int(bounds[0]), int(bounds[-1]) + 1))
        effective = sorted(os.sched_getaffinity(0) & requested)
        if not effective:
            raise SystemExit(70)
        os.sched_setaffinity(0, effective)
        os.execv(args[2], args[2:])
    """))
    wrapper.chmod(0o755)
    real_which = shutil.which
    monkeypatch.setattr(sr.shutil, "which",
                        lambda name: str(wrapper) if name == "taskset" else real_which(name))


def _inputs(tmp_path, *, backends=("cpu", "gpu"), unowned=False, missing=False,
            experimental_gpu=False, cpu_logical=None):
    registry, production, seeds, owners = _registry(), [], [], {}
    for index, backend in enumerate(backends):
        name = f"{backend}-{index}"
        root = tmp_path / name
        root.mkdir()
        build = root / "original-build"
        template = _recipe()
        if backend == "gpu":
            owned_cpus = sorted(os.sched_getaffinity(0))
            template = replace(template, device="ROCm0", ngl=99,
                               cpu_list=",".join(str(cpu) for cpu in owned_cpus),
                               threads=len(owned_cpus))
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
            "frozen_prompts": str(prompt_path), "calibrate_serving": 24 if backend == "cpu" else 2}
    if unowned:
        production.append(_target("unowned", model="model-b"))
    if missing:
        seeds.append(_target("offdisk", model="missing-model"))
    declaration = _manifest(production=production, seeds=seeds)
    declared_cpus = list(range(192)) if cpu_logical is None else list(cpu_logical)
    declaration["resources"].update(cpu_logical=declared_cpus, gpu_ids=[claim.DEVICE_ID])
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


def test_resolved_campaign_refuses_batched_scheduled_child_before_launch(tmp_path, monkeypatch, capsys):
    # Several targets: a longer stage would distort the scheduler's apportionment, so a
    # batched child stays refused (the single-target roster below is admitted).
    _, _, argv = _inputs(tmp_path, backends=("cpu", "gpu"))
    argv[argv.index("--batch-iterations") + 1] = "2"
    _forbid_execution(monkeypatch)
    with pytest.raises(SystemExit) as caught:
        sr.main(argv + ["--dry-run"])
    assert caught.value.code == 2
    err = capsys.readouterr().err
    assert "scheduled serial mode requires one iteration per child" in err
    assert "exactly one target (it has 2)" in err
    assert not (tmp_path / "router").exists()


def test_single_target_derived_schedule_admits_a_batched_child_and_scales_its_bound(
        tmp_path, monkeypatch, capsys):
    resolved, _, argv = _inputs(tmp_path, backends=("cpu",))
    argv[argv.index("--batch-iterations") + 1] = "2"
    _forbid_execution(monkeypatch)
    monkeypatch.setattr(run, "main", lambda child: 0 if "--dry-run" in child else
                        pytest.fail("a dry run launched a child"))
    assert sr.main(argv + ["--dry-run"]) == 0
    out = capsys.readouterr().out
    printed = json.JSONDecoder().raw_decode(out)[0]
    one = (3 * resolved.resources.build_timeout_s + 8 * resolved.resources.stage_timeout_s)
    assert printed["scheduler"]["config"]["max_stage_seconds"] == 2 * one
    assert not (tmp_path / "router").exists()
    # One iteration per child derives the historical manifest byte for byte.
    targets, _, _ = _build(argv)
    targets = [sr._validate_target_args(row, owner_anchor_waiver=True) for row in targets]
    resolved_path = Path(sr.option(argv, "--resolved-campaign"))
    assert (sr._derived_scheduler_manifest(targets, resolved_path, 2).to_dict()
            == sr._derived_scheduler_manifest(targets, resolved_path, 2,
                                              batch_iterations=1).to_dict())
    assert sr._derived_scheduler_manifest(targets, resolved_path, 2).config.max_stage_seconds == one


def test_batched_schedule_refuses_an_explicit_manifest_and_a_pending_screen_seed(
        tmp_path, monkeypatch, capsys):
    assert sr._scheduled_batch_refusal(batch_iterations=1, explicit_manifest=True,
                                       targets=[[], []]) is None
    assert "explicit --scheduler-manifest" in sr._scheduled_batch_refusal(
        batch_iterations=2, explicit_manifest=True, targets=[[]])
    assert sr._scheduled_batch_refusal(batch_iterations=3, explicit_manifest=False,
                                       targets=[[]]) is None
    # A seed carrying a reduced-screen candidate owes its one-candidate confirmation.
    _, _, argv = _inputs(tmp_path, backends=("cpu",))
    argv[argv.index("--batch-iterations") + 1] = "2"
    seed = tmp_path / "seed" / "loop-continuation.json"
    _forbid_execution(monkeypatch)
    monkeypatch.setattr(sr, "_initial_continuation", lambda path, targets: {
        "target_index": 0, "path": str(path), "sha256": "0" * 64})
    monkeypatch.setattr(sr, "load_completed", lambda path, **_kw: (
        {"cpu_screen": {"candidate": {"hypothesis": {}}}}, "0" * 64))
    with pytest.raises(SystemExit) as caught:
        sr.main(argv + ["--initial-continuation", str(seed), "--dry-run"])
    assert caught.value.code == 2
    assert "pending CPU-screen candidate" in capsys.readouterr().err


def test_single_target_batched_schedule_drives_children_and_accounts_each_stage(
        tmp_path, monkeypatch):
    _, _, argv = _inputs(tmp_path, backends=("cpu",))
    argv[argv.index("--batch-iterations") + 1] = "2"
    child = tmp_path / "tiny.py"
    child.write_text(CHILD)
    monkeypatch.setattr(sr, "_child_command", lambda args: [sr.sys.executable, str(child), *args])
    _confine_fixture_child(tmp_path, monkeypatch)
    here = Path(sr.__file__).resolve()
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(here.parents[4]), str(here.parents[2]))))
    assert sr.main(argv) == 0
    state_dir = tmp_path / "router"
    seen = [json.loads(line)["argv"] for line in (state_dir / "seen.jsonl").read_text().splitlines()]
    assert len(seen) == 2 and all(sr.option(row, "--iterations") == "2" for row in seen)
    assert all(sr.option(row, "--scheduler-selection") for row in seen)
    state = json.loads((state_dir / "serial-state.json").read_text())
    assert state["batch_iterations"] == 2 and state["next_batch"] == 2
    scheduler = state["scheduler_state"]
    assert scheduler["campaign_attempts"] == 2 and not scheduler["successor_fences"]
    assert {row["outcome"] for row in scheduler["accounted_receipts"]} == {"valid_comparison"}
    for number in range(2):
        body, _sha = sr.load_completed(
            state_dir / "batches" / f"batch-{number:06d}" / "loop-continuation.json")
        assert body["iterations_completed"] == 2 and body["outcome_counts"] == {"measured_null": 2}
    # A completed restart replays nothing.
    assert sr.main(argv) == 0
    assert len((state_dir / "seen.jsonl").read_text().splitlines()) == 2


@pytest.mark.parametrize("experimental_gpu", [False, True])
def test_generated_roster_drives_actual_children_and_completed_restart(tmp_path, monkeypatch, experimental_gpu):
    owned_cpus = sorted(os.sched_getaffinity(0))
    # Preserve the campaign's full virtual CPU geometry. The fixture taskset shim
    # confines tiny children to this runner's narrower native CPU claim.
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
    _confine_fixture_child(tmp_path, monkeypatch)
    here = Path(sr.__file__).resolve()
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(here.parents[4]), str(here.parents[2]))))
    assert sr.main(argv) == 0
    state = tmp_path / "router"
    seen = [json.loads(line) for line in (state / "seen.jsonl").read_text().splitlines()]
    # Distinct recipe artifacts keep these as two distinct enrolled targets.
    assert len(seen) == 4
    assert all(set(row["affinity"]) == set(owned_cpus) for row in seen)
    assert {sr.option(row["argv"], "--target-id") for row in seen} == {"cpu-0", "gpu-1"}
    resumed = [row for row in seen if sr.option(row["argv"], "--resume-run")]
    assert resumed
    for row in resumed:
        assert sr.option(row["argv"], "--cpu-calibrate-serving") is None
        assert sr.option(row["argv"], "--gpu-calibrate-serving") is None
    assert sr.main(argv) == 0
    assert len((state / "seen.jsonl").read_text().splitlines()) == 4


def test_multiple_cpu_targets_serialize_one_owned_source_without_fabricating_shared_keep(
        tmp_path, monkeypatch):
    _, owners, argv = _inputs(tmp_path, backends=("cpu", "cpu"))
    owners["cpu-1"]["worktree"] = owners["cpu-0"]["worktree"]
    owners["cpu-1"]["branch"] = owners["cpu-0"]["branch"]
    Path(sr.option(argv, "--owned-targets")).write_text(json.dumps(owners))
    child = tmp_path / "tiny.py"
    child.write_text(CHILD)
    monkeypatch.setattr(sr, "_child_command", lambda args: [sr.sys.executable, str(child), *args])
    _confine_fixture_child(tmp_path, monkeypatch)
    here = Path(sr.__file__).resolve()
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(here.parents[4]), str(here.parents[2]))))
    assert sr.main(argv) == 0
    seen = [json.loads(line)["argv"] for line in (
        tmp_path / "router" / "seen.jsonl").read_text().splitlines()]
    assert len(seen) == 4
    for number, row in enumerate(seen[1:], start=1):
        # Cost-informed selection need not alternate seed targets. With only
        # measured nulls, the selected target resumes its own latest result;
        # no cross-target source continuation may be fabricated without a keep.
        reference = (sr.option(row, "--source-anchor-continuation")
                     or sr.option(row, "--resume-run"))
        assert reference.endswith(f"batch-{number - 1:06d}/loop-continuation.json")
    state = json.loads((tmp_path / "router/serial-state.json").read_text())
    assert state["source_results"] == {}
    assert all(sr.option(row, "--source-anchor-continuation") is None for row in seen)


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


def test_explicit_unverified_anchor_waiver_reaches_existing_owner_only(tmp_path):
    _resolved, owners, argv = _inputs(tmp_path, backends=("cpu",))
    owners["cpu-0"]["allow_unverified_anchor"] = True
    Path(sr.option(argv, "--owned-targets")).write_text(json.dumps(owners))
    targets, _skipped, _cpus = _build(argv)
    assert "--allow-unverified-anchor" in targets[0]
    owners["cpu-0"]["allow_unverified_anchor"] = "yes"
    Path(sr.option(argv, "--owned-targets")).write_text(json.dumps(owners))
    with pytest.raises(sr.SerialRefused, match="must be boolean"):
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
        owners["cpu-1"]["worktree"] = str(Path(owners["cpu-0"]["worktree"]) / "nested")
    elif case == "production_branch":
        owners["cpu-0"]["branch"] = "production-consolidated-v9"
    path.write_text(json.dumps(owners))
    monkeypatch.setattr(sr, "_drive", lambda *_a, **_kw: pytest.fail("invalid roster reached launch"))
    with pytest.raises(SystemExit) as caught:
        sr.main(argv)
    assert caught.value.code == 2
    assert not (tmp_path / "router").exists()


def test_a_validation_stage_stays_one_iteration_under_a_batched_schedule(tmp_path):
    original = ["--target-id", "t", "--worktree", str(tmp_path / "source")]
    ordinary = sr._batch_argv(original, None, 2, tmp_path / "b0")
    assert sr.option(ordinary, "--iterations") == "2"
    for flags in ({"validate_source": True}, {"validate_loo": True}):
        staged = sr._batch_argv(original, None, 2, tmp_path / "b1", **flags)
        assert sr.option(staged, "--iterations") == "1"
    # Batch size 1: the historical argv whatever the stage.
    assert sr._batch_argv(original, None, 1, tmp_path / "b2") == \
        sr._batch_argv(original, None, 1, tmp_path / "b2", validate_source=True)


def test_common_args_admit_the_longctx_surface_and_lane_binding(tmp_path):
    """The long-context surface is keep POLICY like the lane binding (serial_run.
    LONGCTX_SURFACE_FLAGS: resume_binding excludes it so turning it on at a batch
    boundary carries the lineage). The common-args admission list must accept it too,
    or the only way to opt in -- the shared common args -- refuses at startup."""
    _resolved, _owners, argv = _inputs(tmp_path, backends=("cpu",))
    common = tmp_path / "common.json"
    common.write_text(json.dumps(["--workers", "1", "--longctx-surface", "/x/spec.json",
                                  "--lane-targets", "/x/lt.json", "--lane", "lane0"]))
    targets, _skipped, _cpus = serial_roster.build_targets(
        Path(sr.option(argv, "--resolved-campaign")), Path(sr.option(argv, "--owned-targets")),
        target_root=Path(sr.option(argv, "--state-dir")) / "targets", common_path=common)
    assert sr.option(targets[0], "--longctx-surface") == "/x/spec.json"
    assert sr.option(targets[0], "--lane") == "lane0"


def test_keep_policy_flags_are_shared_policy_not_resume_identity():
    """27B GPU slot 6b: keep dimensions and the GPU claim posture ride the common argv and
    never orphan a continuation (serial_run.KEEP_POLICY_FLAGS)."""
    base = ["--target-id", "gpu-0", "--model", "/m.gguf"]
    extra = ["--keep-dimensions", "short_decode,capacity", "--keep-capacity-limit-gib", "62",
             "--gpu-cpu-region-claim", "off", "--longctx-surface", "/s.json"]
    assert sr.resume_binding(base + extra) == sr.resume_binding(base)
