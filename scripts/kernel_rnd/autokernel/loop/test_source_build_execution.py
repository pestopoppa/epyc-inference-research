from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

import pytest

from .. import schemas
from ..execution import worktree
from . import (actor_lifecycle, actor_preparation, campaign, campaign_control,
               measurement_capture, profile_preparation, scheduling,
               source_build_preparation, standalone_inputs, target_profile_execution,
               unified_driver, worker_lifecycle)
from . import source_build_execution as execution
from . import source_build_worker as worker
from .test_campaign_control import _command
from .test_actor_preparation import _budgets, _profile
from .test_profile_preparation_runtime import (
    SyntheticSelectedProfileProvider, _fixture as profile_fixture, _start)
from .test_standalone_inputs import FullHeldProvider, _verifier
from .test_source_build_preparation import advice
from .test_unified_driver import runtime_driver
from .test_unified_planner import opportunity, prepared, profile, runtime_anchor, scheduler
from .test_worker_lifecycle import ReceiptProvider


class _BuildReceiptProvider(ReceiptProvider):
    def authorize(self, request, container_id, deadline):
        del request, deadline
        grant = worker_lifecycle.GrantReceipt(
            "fixture-build-grant", 1, time.monotonic() + 180.0,
            worker_lifecycle.monotonic_clock_domain())
        self.authorization = worker_lifecycle.AuthorizedLaunch(
            grant, container_id,
            type(self).container_type(self.root, container_id))
        return self.authorization

    @staticmethod
    def container_type(root, container_id):
        from .test_worker_lifecycle import MockOwnedContainer
        return MockOwnedContainer(root, container_id)


class _ProfileActorBuildProvider(SyntheticSelectedProfileProvider):
    """Profile metadata author plus the unchanged generic held-receipt path."""

    def close_held_receipt(self, **request):
        if request["request"].request_id in self.requests:
            return super().close_held_receipt(**request)
        return FullHeldProvider.close_held_receipt(self, **request)


def _different_plan(_value):
    raise AssertionError("changed producer must be refused before plan execution")


def _plan(tmp_path: Path) -> worktree.BuildPlan:
    source = tmp_path / "source"
    actor = tmp_path / "actor"
    source.mkdir()
    actor.mkdir()
    return worktree.BuildPlan(
        worktree.SandboxPath.create(source, sandbox_root=tmp_path, production_trees=()),
        worktree.SandboxPath.create(tmp_path / "build", sandbox_root=tmp_path,
                                    production_trees=()),
        worktree.SandboxPath.create(actor, sandbox_root=tmp_path, production_trees=()),
        worktree.BuildParallelism(1), targets=("fixture",),
        cmake_defines=(("FIXTURE", "ON"),), cmake="/usr/bin/cmake")


def _disposition(argv):
    return {"argv": list(argv), "pid": 41, "pgid": 41, "exit_code": 0,
            "timed_out": False, "signals_sent": [], "verified_dead": True,
            "duration_s": 0.1, "started_at": "2026-09-09T00:00:00Z",
            "sandbox_receipt": None, "sandbox_teardown": None}


def _receipt(tmp_path: Path, plan: worktree.BuildPlan):
    facts = {"configured": True, "build_dir_from_log": plan.build_dir.path,
             "compiler_ids": [["CXX", "fixture"]], "ccache_enabled": False,
             "ggml_version": None, "ggml_commit": None, "ggml_commit_dirty": False,
             "built_targets": ["fixture"], "linked_outputs": ["fixture"],
             "compile_units": 1, "warning_count": 0, "errors": [],
             "make_failures": [], "succeeded_by_log": True}
    body = {"schema": "epyc.autokernel.build_process_result.v1",
            "plan": plan.to_dict(), "configure": _disposition(plan.configure_argv()),
            "build": _disposition(plan.build_argv()),
            "log_path": str(tmp_path / "build.log"), "log_sha256": "a" * 64,
            "log_identity": {"device": 1, "inode": 2}, "facts": facts,
            "build_dir_pre_build_digest": "b" * 64,
            "build_dir_created_for_this_build": True,
            "load_average_at_start": None}
    payload = body | {"receipt_sha256": schemas.content_hash(body)}
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path = tmp_path / "result.json"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def test_worker_reconstructs_exact_plan_and_rejects_changed_command(tmp_path):
    plan = _plan(tmp_path)
    assert worker._plan(plan.to_dict()).to_dict() == plan.to_dict()
    changed = plan.to_dict()
    changed["build_command"] = [*changed["build_command"], "--forged"]
    with pytest.raises(worker.SourceBuildWorkerRefused, match="differs"):
        worker._plan(changed)


def test_parent_reopens_exact_self_hashed_build_result(tmp_path):
    plan = _plan(tmp_path)
    path, digest = _receipt(tmp_path, plan)
    result = execution.reopen_build_result(str(path), digest, plan)
    assert result.plan == plan and result.succeeded and result.facts.succeeded_by_log
    raw = bytearray(path.read_bytes())
    raw[10] ^= 1
    path.write_bytes(raw)
    with pytest.raises(execution.SourceBuildExecutionRefused, match="bytes differ"):
        execution.reopen_build_result(str(path), digest, plan)


def test_parent_refuses_child_nominated_same_bytes_at_foreign_path(tmp_path):
    plan = _plan(tmp_path)
    path, digest = _receipt(tmp_path, plan)
    foreign = tmp_path / "foreign.json"
    foreign.write_bytes(path.read_bytes())
    with pytest.raises(execution.SourceBuildExecutionRefused,
                       match="differs from admitted plan"):
        execution.reopen_build_result(
            str(foreign), digest, plan, expected_log_path=str(tmp_path / "build.log"))


def test_parent_refuses_receipt_with_foreign_log_identity(tmp_path):
    plan = _plan(tmp_path)
    log = tmp_path / "build.log"
    log.write_bytes(b"owned build log\n")
    log.chmod(0o600)
    path, _digest = _receipt(tmp_path, plan)
    row = json.loads(path.read_bytes())
    body = dict(row)
    body.pop("receipt_sha256")
    body["log_path"] = str(log)
    body["log_sha256"] = hashlib.sha256(log.read_bytes()).hexdigest()
    body["log_identity"] = {
        "device": 1, "inode": 2, "uid": os.geteuid(), "mode": 0o600,
        "nlink": 1, "size": len(log.read_bytes()), "mtime_ns": 3, "ctime_ns": 4,
    }
    raw = json.dumps(body | {"receipt_sha256": schemas.content_hash(body)},
                     sort_keys=True, separators=(",", ":")).encode() + b"\n"
    receipt = Path(str(log) + ".result.json")
    receipt.write_bytes(raw)
    with pytest.raises(execution.SourceBuildExecutionRefused,
                       match="differs from its owned identity"):
        execution.reopen_build_result(
            str(receipt), hashlib.sha256(raw).hexdigest(), plan,
            expected_log_path=str(log), max_log_bytes=4096)


def test_worker_refuses_changed_loaded_producer_before_run(tmp_path, monkeypatch):
    plan = _plan(tmp_path)
    request = {"schema": worker.REQUEST_SCHEMA, "build_plan": plan.to_dict(),
               "log_path": str(tmp_path / "log"), "configure_timeout_s": 1,
               "build_timeout_s": 1, "env": {},
               "sandbox_cgroup_root": "/unavailable",
               "producer_identity": worker.producer_identity()}
    monkeypatch.setattr(worker, "_plan", _different_plan)
    with pytest.raises(worker.SourceBuildWorkerRefused, match="producer identity differs"):
        worker.run(request)


def test_executor_refuses_loaded_change_between_construction_and_launch(
        tmp_path, monkeypatch):
    instance, controller, _outcome = _selected_build(tmp_path, "d" * 40)
    store = measurement_capture.ArtifactStore(tmp_path / "artifacts")
    module_root = Path(__file__).resolve().parents[2]
    python = Path(sys.executable).resolve()
    try:
        executor = execution.SourceBuildStageExecutor(
            driver=instance, controller=controller, artifact_store=store,
            artifact_root=store.root, python_executable=python,
            python_sha256=hashlib.sha256(python.read_bytes()).hexdigest(),
            module_root=module_root,
            max_build_log_bytes=1024 * 1024,
            worker_source_sha256=hashlib.sha256(
                (module_root / "autokernel/loop/source_build_worker.py").read_bytes()
            ).hexdigest())
        plan = _plan(tmp_path)
        with pytest.raises(execution.SourceBuildExecutionRefused,
                           match="owner namespace"):
            executor(
                plan, log_path=str(tmp_path / "foreign.log"),
                configure_timeout_s=1, build_timeout_s=1, env={},
                require_fresh_build_dir=True,
                sandbox_cgroup_root=worktree.process_sandbox.default_cgroup_root())
        monkeypatch.setattr(worker, "_plan", _different_plan)
        with pytest.raises(execution.SourceBuildExecutionRefused,
                           match="producer identity changed before launch"):
            executor(
                plan, log_path=executor.log_path_for(plan),
                configure_timeout_s=1, build_timeout_s=1, env={},
                require_fresh_build_dir=True,
                sandbox_cgroup_root=worktree.process_sandbox.default_cgroup_root())
        assert list((tmp_path / "artifacts").rglob("source-build-request")) == []
    finally:
        store.close()
        controller.close()


def test_parent_receipt_reader_refuses_fifo_without_blocking(tmp_path):
    fifo = tmp_path / "result.fifo"
    os.mkfifo(fifo)
    with pytest.raises(execution.SourceBuildExecutionRefused, match="bounded private file"):
        execution.reopen_build_result(str(fifo), "a" * 64, _plan(tmp_path))


def test_parent_receipt_reader_refuses_oversize(tmp_path):
    path = tmp_path / "huge.json"
    path.write_bytes(b"x" * (execution.MAX_BUILD_RECEIPT_BYTES + 1))
    with pytest.raises(execution.SourceBuildExecutionRefused, match="bounded private file"):
        execution.reopen_build_result(str(path), hashlib.sha256(path.read_bytes()).hexdigest(),
                                      _plan(tmp_path))


def _git(tmp_path: Path, *, shadow_marker: Path | None = None):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    subprocess.run(("git", "init", "-q", str(repo_path)), check=True)
    subprocess.run(("git", "-C", str(repo_path), "config", "user.email", "fixture@example"),
                   check=True)
    subprocess.run(("git", "-C", str(repo_path), "config", "user.name", "fixture"),
                   check=True)
    (repo_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\nproject(fixture C)\n"
        "add_executable(fixture main.c)\n", encoding="utf-8")
    (repo_path / "main.c").write_text("int main(void){return 0;}\n", encoding="utf-8")
    if shadow_marker is not None:
        shadow = repo_path / "autokernel" / "loop"
        shadow.mkdir(parents=True)
        (shadow.parent / "__init__.py").write_text("", encoding="utf-8")
        (shadow / "__init__.py").write_text("", encoding="utf-8")
        (shadow / "source_build_entrypoint.py").write_text(
            "from pathlib import Path\n"
            f"Path({str(shadow_marker)!r}).write_text('shadow imported')\n"
            "raise RuntimeError('actor source shadow was imported')\n",
            encoding="utf-8")
    subprocess.run(("git", "-C", str(repo_path), "add", "."), check=True)
    subprocess.run(("git", "-C", str(repo_path), "commit", "-q", "-m", "fixture"),
                   check=True)
    repo = worktree.GitRepo(repo_path)
    head = repo.head_commit()
    destination = worktree.SandboxPath.create(
        tmp_path / "snapshot", sandbox_root=tmp_path, production_trees=())
    snapshot = repo.add_worktree(destination, head, detach=True)
    return repo, snapshot, head


def _profiled_build_runtime(tmp_path: Path, *, source_revision: str,
                            build_plan: worktree.BuildPlan):
    materialized, _registry, target, old_binding, counter, original_profile = \
        profile_fixture(tmp_path, selected_provider=True)
    actor_binary = tmp_path / "tiny-build-actor"
    build_advice = {
        "build_system": "cmake",
        "configured_options": [
            f"-D{name}={value}" for name, value in build_plan.effective_defines],
        "artifact_expectations": "owner must verify output",
    }
    actor_binary.write_text(
        "#!/usr/bin/env python3\nimport json,sys\np=sys.argv[-1]\n"
        "if 'Review this proposed preparation' in p:\n"
        " print(json.dumps({'accepted':True,'reason':'reviewed'}))\n"
        f"else:\n print(json.dumps({build_advice!r}))\n")
    actor_binary.chmod(actor_binary.stat().st_mode | stat.S_IXUSR)
    actor_profiles = {item.profile_id: item for item in (
        _profile("actor-planner", "planner", actor_binary),
        _profile("actor-critic", "critic", actor_binary),
    )}

    resolved_path = Path(materialized.manifest.driver_config.resolved_campaign_path)
    resolved = json.loads(resolved_path.read_text())
    resolved["campaign_id"] = "ak-profile-actor-build"
    resolved["actors"] = {"planner": "actor-planner", "critic": "actor-critic"}
    resolved["fallbacks"] = {"planner": [], "critic": []}
    source_name = next(iter(resolved["source_refs"]))
    source_ref = f"production-source:{source_name}:{source_revision}"
    resolved["source_refs"][source_name] = source_ref
    resolved["source_snapshot"][source_name]["ref"] = source_ref
    resolved_path.write_text(json.dumps(resolved))

    target_row = materialized.resolved.targets[0]
    allocation = scheduling.ResourceVector(1.0, (), 0)
    build_opportunity = opportunity(
        kind="build_recipe", target=target_row, allocation=allocation.digest)
    profile_row = copy.deepcopy(original_profile)
    profile_row["opportunities"] = [build_opportunity]
    loaded = dict(old_binding.mechanism.loaded_identity)
    output = {
        "schema": target_profile_execution.PROFILE_OUTPUT_SCHEMA,
        "profile_content": profile_row, "loaded_identity": loaded,
        "artifact_identity": {"kind": "tiny-fixture", "sha256": "a" * 64},
        "measurement_carrier": {
            "schema": "epyc.autokernel.profile_measurement_carrier.v1",
            "profile_source_id": target_profile_execution.PROFILE_SOURCE_ID,
            "validation_source_id": target_profile_execution.VALIDATION_SOURCE_ID,
            "run_id": "source-build-profile", "profile_claim_tuple": {"fixture": True},
            "validation_claim_tuple": {"fixture": True},
        },
    }
    profiler = old_binding.mechanism.binary
    profiler.write_text(
        "#!/usr/bin/env python3\nfrom pathlib import Path\nimport json,sys\n"
        f"p=Path({str(counter)!r});p.write_text((p.read_text() if p.exists() else '')+'x')\n"
        f"print({json.dumps(output)!r})\n")
    profiler.chmod(profiler.stat().st_mode | stat.S_IXUSR)
    mechanism = target_profile_execution.ProfileMechanism(
        "tiny-profile", profiler, hashlib.sha256(profiler.read_bytes()).hexdigest(),
        tmp_path, {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}, loaded,
        2.0, 1.0, 65536)
    binding = profile_preparation.InstalledProfileMechanismBinding(mechanism, 120.0)
    request = next(iter(materialized.inputs.profile_requests.values())).to_dict()
    request["profile_contract"]["adapter_digest"] = binding.adapter_digest
    request = unified_driver.ProfilePreparationRequest.from_dict(request)
    document = materialized.manifest.to_dict()
    scheduler_config = scheduling.SchedulerConfig.from_dict(
        document["driver_config"]["scheduler_config"]
        | {"config_id": resolved["campaign_id"]})
    document["driver_config"]["scheduler_config"] = scheduler_config.to_dict()
    document["driver_config"]["scheduler_state"] = scheduling.initial_state(
        scheduler_config, resolved["campaign_id"]).to_dict()
    document["driver_config"]["profile_requests"] = {target: request.to_dict()}
    document["actor_identities"] = {"build_recipe": actor_profiles["actor-planner"].to_dict()}
    document["manifest_digest"] = standalone_inputs._digest({
        key: value for key, value in document.items() if key != "manifest_digest"})
    materialized = standalone_inputs.materialize(
        standalone_inputs.StartupManifest.from_dict(document))
    provider = _ProfileActorBuildProvider(tmp_path / "containers", {target: request})
    installed = profile_preparation.InstalledProfilePreparationBinding({target: binding})
    registry = standalone_inputs.ProviderRegistry({
        "fixture-lifecycle": standalone_inputs.ProviderBinding(
            lifecycle_provider=provider),
        "fixture-readiness": standalone_inputs.ProviderBinding(
            readiness_check=lambda: (True, None)),
    }, evidence_verifiers={"fixture-evidence": _verifier()},
        profile_bindings={"tiny-profile": installed})
    return materialized, registry, target, actor_profiles


def _selected_build(tmp_path: Path, revision: str):
    instance, _old_engine, enrolled, target, target_digest = runtime_driver(git_source=True)
    raw = enrolled.to_dict()
    raw["campaign_id"] = "ak-source-build-owner"
    ref = f"production-source:kernel:{revision}"
    raw["source_refs"] = {"kernel": ref}
    raw["source_snapshot"]["kernel"]["ref"] = ref
    enrolled = campaign.ResolvedCampaign.from_dict(raw)
    instance.runtime_anchors = prepared(enrolled, {target_digest: runtime_anchor(
        target, instance.runtime_anchors.recipes[target_digest])})
    instance.resolved = enrolled
    base, _ = scheduler()
    config = scheduling.SchedulerConfig.from_dict(
        base.to_dict() | {"config_id": enrolled.campaign_id})
    engine = scheduling.SchedulerEngine(
        config, scheduling.initial_state(config, enrolled.campaign_id))
    instance.scheduler = engine
    cost = scheduling.ResourceVector(1.0, (), 0)
    instance.profiles = {target_digest: profile(target, opportunities=[opportunity(
        kind="build_recipe", target=target, allocation=cost.digest)])}
    containers = tmp_path / "containers"
    containers.mkdir()
    controller = campaign_control.CampaignController(
        enrolled, tmp_path / "owner", snapshot_version=3,
        scheduler_engine=engine, readiness_check=lambda: (True, None),
        lifecycle_provider=_BuildReceiptProvider(containers))
    controller.__enter__()
    controller.apply_command(_command(enrolled, "resume", "resume", 0))
    instance.controller = controller
    outcome = instance.tick(now=1.0)
    return instance, controller, outcome


def test_actual_selected_build_runs_owned_child_and_publishes_enrollment(tmp_path):
    repo, snapshot, head = _git(tmp_path)
    instance, controller, outcome = _selected_build(tmp_path, head)
    store = measurement_capture.ArtifactStore(tmp_path / "artifacts")
    try:
        selected = instance.materialize_actor(outcome)
        plan = worktree.BuildPlan(
            snapshot.path,
            worktree.SandboxPath.create(tmp_path / "build", sandbox_root=tmp_path,
                                        production_trees=()),
            worktree.SandboxPath.create(tmp_path / "actor", sandbox_root=tmp_path,
                                        production_trees=()),
            worktree.BuildParallelism(
                1, cpu_list=str(instance.resolved.resources.cpu_logical[0])),
            targets=("fixture",))
        Path(plan.actor_worktree.path).mkdir()
        proposed = advice(selected, options=[f"-D{k}={v}" for k, v in plan.effective_defines])
        bound = source_build_preparation.bind_build_preparation(
            selected_actor_work=selected, preparation_result=proposed,
            resolved_campaign=instance.resolved, source_commit=head,
            build_plan=plan, source_worktree=snapshot)
        module_root = Path(__file__).resolve().parents[2]
        python = Path(sys.executable).resolve()
        executor = execution.SourceBuildStageExecutor(
            driver=instance, controller=controller, artifact_store=store,
            artifact_root=store.root, python_executable=python,
            python_sha256=hashlib.sha256(python.read_bytes()).hexdigest(),
            module_root=module_root,
            max_build_log_bytes=1024 * 1024,
            worker_source_sha256=hashlib.sha256(
                (module_root / "autokernel/loop/source_build_worker.py").read_bytes()).hexdigest())
        enrollment = executor.execute(
            bound, source_worktree=snapshot, candidate_id="akc-source-build-owner",
            output_binary=Path(plan.build_dir.path) / "fixture",
            toolchain="cmake fixture", libraries={}, linkage_sha256="c" * 64,
            log_path=executor.log_path_for(plan), configure_timeout_s=30,
            build_timeout_s=30, env={"PATH": os.environ["PATH"], "LANG": "C"},
            sandbox_cgroup_root=worktree.process_sandbox.default_cgroup_root())
        assert enrollment["build_identity"]["output_binary_sha256"]
        assert enrollment["artifact"]["verified"] is True
        assert store.read(enrollment["artifact"]["locator"],
                          enrollment["artifact"]["sha256"])["candidate_id"] \
            == "akc-source-build-owner"
        kinds = [item.kind for item in controller._journal.read_all()]
        assert "WORKER_LIFECYCLE" in kinds and "WORKER_ACQUISITION" in kinds
    finally:
        store.close()
        controller.close()
        repo.remove_worktree(snapshot.path, force=True)


def test_genuine_profile_then_actor_consumer_runs_owned_build(tmp_path):
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    repo_root = tmp_path / "source"
    repo_root.mkdir()
    shadow_marker = tmp_path / "actor-shadow-imported"
    repo, snapshot, head = _git(repo_root, shadow_marker=shadow_marker)
    plan = worktree.BuildPlan(
        snapshot.path,
        worktree.SandboxPath.create(repo_root / "build", sandbox_root=repo_root,
                                    production_trees=()),
        worktree.SandboxPath.create(repo_root / "actor", sandbox_root=repo_root,
                                    production_trees=()),
        worktree.BuildParallelism(1, cpu_list="0"),
        targets=("fixture",), cmake_defines=(("FIXTURE", "ON"),))
    materialized, registry, target, actor_profiles = _profiled_build_runtime(
        runtime_root, source_revision=head, build_plan=plan)
    controller, runtime = _start(materialized, registry)
    store = None
    try:
        profile_result = runtime.tick()
        assert profile_result.status == "settled"
        assert controller.current_verified_profile_result(target)["settlement"] is not None
        profiles = runtime.profile_executor.planner_profiles(time.monotonic())
        build_driver = unified_driver.UnifiedCampaignDriver(
            resolved_campaign=runtime.driver.resolved, controller=controller,
            scheduler_engine=runtime.driver.scheduler, profiles=profiles,
            evidence_index=runtime.driver.evidence,
            runtime_anchors=runtime.driver.runtime_anchors,
            runtime_dimensions=runtime.driver.runtime_dimensions,
            experiment_plans=runtime.driver.experiment_plans,
            profile_requests=runtime.driver.profile_requests,
            actor_identities=runtime.driver.actor_identities,
            native_artifact_sink_ref=runtime.driver.sink_ref,
            execution_inputs=runtime.driver.execution_inputs,
            executable_work_kinds={"actor_preparation"})
        outcome = build_driver.tick(now=time.monotonic())
        selected = build_driver.materialize_actor(outcome)
        assert selected.actor_request.actor_kind == "build_recipe"

        lifecycle = actor_lifecycle.ActorLifecycleAdapter(
            controller=controller, persistence=controller,
            config=actor_lifecycle.ActorLifecycleConfig(
                campaign_digest=controller.config_digest, cwd=runtime_root,
                env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
                max_stage_seconds=5.0, teardown_seconds=1.0,
                max_retained_output_bytes=4096),
            target_profile_owner=controller)
        consumer = actor_preparation.ActorPreparationConsumer(
            resolved_campaign=runtime.driver.resolved, profiles=actor_profiles,
            budgets=_budgets(), capability=lifecycle, clock=time.monotonic,
            clock_domain=worker_lifecycle.monotonic_clock_domain(),
            max_output_bytes=4096)
        artifact_root = Path(selected.controller_binding["artifact_root"])
        store = measurement_capture.ArtifactStore(artifact_root)
        module_root = Path(__file__).resolve().parents[2]
        python = Path(sys.executable).resolve()
        stage = execution.SourceBuildStageExecutor(
            driver=build_driver, controller=controller, artifact_store=store,
            artifact_root=artifact_root, python_executable=python,
            python_sha256=hashlib.sha256(python.read_bytes()).hexdigest(),
            module_root=module_root, max_build_log_bytes=1024 * 1024,
            worker_source_sha256=hashlib.sha256(
                (module_root / "autokernel/loop/source_build_worker.py").read_bytes()
            ).hexdigest())
        owner = execution.SourceBuildExecutionOwner(
            driver=build_driver, actor_consumer=consumer, build_executor=stage)
        assert int(plan.parallelism.cpu_list) in runtime.driver.resolved.resources.cpu_logical
        Path(plan.actor_worktree.path).mkdir()
        enrollment = owner.execute_selected(
            selected, source_worktree=snapshot, source_commit=head,
            materialized_source=None, build_plan=plan,
            candidate_id="akc-profile-actor-build",
            output_binary=Path(plan.build_dir.path) / "fixture",
            toolchain="cmake fixture", libraries={}, linkage_sha256="c" * 64,
            log_path=stage.log_path_for(plan), configure_timeout_s=30,
            build_timeout_s=30, env={"PATH": os.environ["PATH"], "LANG": "C"},
            sandbox_cgroup_root=worktree.process_sandbox.default_cgroup_root())
        assert enrollment["candidate_id"] == "akc-profile-actor-build"
        assert not shadow_marker.exists()
        events = controller._journal.read_all()
        assert sum(row.kind == "ACTOR_PREPARATION" for row in events) >= 5
        assert sum(row.kind == "WORKER_LIFECYCLE" and
                   row.payload["event"] == "WORKER_RESULT_ACCEPTED" for row in events) >= 4
    finally:
        if store is not None:
            store.close()
        runtime.close()
        controller.close()
        repo.remove_worktree(snapshot.path, force=True)
