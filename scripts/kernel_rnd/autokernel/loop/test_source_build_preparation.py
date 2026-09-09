"""No models or CMake: public selection, real tiny Git fixture, recording builder."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from .. import source_candidate as source
from ..controller.discovery_controller import AuthoringAssignment
from ..execution import worktree as W
from .. import test_source_candidate as source_fixture
from ..test_source_candidate import PATH, SYMBOL
from . import actor_preparation as actor, campaign, campaign_control, scheduling
from . import source_build_preparation as bridge, unified_driver as driver
from .test_campaign_control import _command
from .test_unified_driver import runtime_driver
from .test_unified_planner import opportunity, prepared, profile, runtime_anchor, scheduler


@pytest.fixture
def tiny_source():
    fixture = source_fixture.SourceCase("runTest")
    fixture.setUp()
    try:
        yield fixture
    finally:
        fixture.doCleanups()


def selected_actor(tmp_path, *, kind="source", revision="a" * 40):
    instance, _, enrolled, target, target_digest = runtime_driver(git_source=True)
    raw = enrolled.to_dict()
    raw["campaign_id"] = "ak-source-bridge"
    ref = f"production-source:kernel:{revision}"
    raw["source_refs"] = {"kernel": ref}
    raw["source_snapshot"]["kernel"]["ref"] = ref
    enrolled = campaign.ResolvedCampaign.from_dict(raw)
    instance.runtime_anchors = prepared(
        enrolled, {target_digest: runtime_anchor(
            target, instance.runtime_anchors.recipes[target_digest])})
    instance.resolved = enrolled
    base, _ = scheduler()
    config = scheduling.SchedulerConfig.from_dict(
        base.to_dict() | {"config_id": enrolled.campaign_id})
    engine = scheduling.SchedulerEngine(
        config, scheduling.initial_state(config, enrolled.campaign_id))
    instance.scheduler = engine
    cost = scheduling.ResourceVector(1.0, (), 0)
    instance.profiles = {target_digest: profile(
        target, opportunities=[opportunity(kind=kind, target=target, allocation=cost.digest)])}
    controller = campaign_control.CampaignController(
        enrolled, tmp_path / "owner", snapshot_version=3,
        scheduler_engine=engine, readiness_check=lambda: (True, None))
    controller.__enter__()
    controller.apply_command(_command(enrolled, "resume", "resume", 0))
    instance.controller = controller
    outcome = instance.tick(now=1.0)
    return instance, controller, outcome


def advice(selected, *, options=()):
    value = ({"mechanism": "threads", "target_surface": PATH,
              "target_symbol": SYMBOL, "implementation_plan": "change the fixture addition"}
             if selected.actor_request.actor_kind == "source" else
             {"build_system": "cmake", "configured_options": list(options),
              "artifact_expectations": "unverified output; owner must build and identify it"})
    return actor.PreparationResult(
        "proposed", actor._digest(selected.actor_request.to_dict()),
        selected.actor_request.proposal.target_revision_digest, "b" * 64, value)


def source_inputs(fixture, selected):
    assignment = AuthoringAssignment(
        selected.controller_binding["campaign_id"], "akp-source-bridge", "akc-source-bridge",
        fixture.prod, fixture.base)
    manifest = fixture.manifest(campaign_id=assignment.campaign_id,
                                proposal_id=assignment.proposal_id,
                                candidate_id=assignment.candidate_id, mechanism_id="threads")
    proposal = fixture.proposal() | {"proposal_id": assignment.proposal_id}
    return assignment, manifest, proposal


def bound_source(instance, selected, fixture, result=None, **changes):
    assignment, manifest, proposal = source_inputs(fixture, selected)
    inputs = dict(selected_actor_work=selected, preparation_result=result or advice(selected),
                  resolved_campaign=instance.resolved, assignment=assignment,
                  authored_manifest=manifest, legacy_proposal=proposal)
    inputs.update(changes)
    return bridge.bind_source_preparation(**inputs)


@pytest.mark.parametrize("kind", ["source", "build_recipe"])
def test_public_actor_selection_and_deeply_immutable_materialization(tmp_path, kind):
    instance, controller, outcome = selected_actor(tmp_path, kind=kind)
    try:
        selected = instance.materialize_actor(outcome)
        assert selected.actor_request.actor_kind == kind
        assert selected.stage_plan_digest == selected.actor_request.cache_key
        assert selected.selection.to_dict() == driver._thaw(outcome.selection)
        assert selected.execution_authorized is False
        assert driver.SelectedActorWork.from_dict(selected.to_dict()) == selected
        detached = selected.to_dict()
        detached["actor_request"]["proposal"]["parent_identity"]["target_ids"].append("forged")
        assert "forged" not in selected.actor_request.proposal.parent_identity["target_ids"]
        with pytest.raises(TypeError):
            selected.controller_binding["campaign_id"] = "foreign"
    finally:
        controller.close()


@pytest.mark.parametrize("field", ["authority", "transition", "stage", "request", "owner"])
def test_selected_actor_record_refuses_binding_changes(tmp_path, field):
    instance, controller, outcome = selected_actor(tmp_path)
    try:
        raw = instance.materialize_actor(outcome).to_dict()
        if field == "authority":
            raw["execution_authorized"] = True
        elif field == "transition":
            raw["transition_id"] = "f" * 64
        elif field == "stage":
            raw["stage_plan_digest"] = "e" * 64
        elif field == "request":
            raw["actor_request"]["cache_key"] = "d" * 64
        else:
            raw["controller_binding"]["supervisor_incarnation"] = True
        with pytest.raises((driver.DriverRefused, RuntimeError)):
            driver.SelectedActorWork.from_dict(raw)
    finally:
        controller.close()


def test_wrong_kind_and_closed_owner_cannot_materialize(tmp_path):
    runtime, _, _, _, _ = runtime_driver()
    with pytest.raises(driver.DriverRefused, match="not actor preparation"):
        runtime.materialize_actor(runtime.tick(now=1.0))
    instance, controller, outcome = selected_actor(tmp_path)
    controller.close()
    with pytest.raises(driver.DriverRefused, match="controller refused"):
        instance.materialize_actor(outcome)


def test_real_guarded_source_materialization_not_a_build(tmp_path, tiny_source):
    instance, controller, outcome = selected_actor(tmp_path, revision=tiny_source.base)
    try:
        selected = instance.materialize_actor(outcome)
        bound = bound_source(instance, selected, tiny_source)
        before = tiny_source.actor.head_commit()
        receipt = bridge.materialize_source(bound, campaign_driver=instance,
                                            actor_worktree=tiny_source.actor)
        assert receipt["source_commit"] == tiny_source.actor.head_commit() != before
        assert receipt["actual_files"] == (PATH,)
        assert receipt["source_tree_digest"] == tiny_source.actor.snapshot_digest().sha256
        assert receipt["build_status"] == "pending"
        assert receipt["execution_authorized"] is False
        assert "build_sha256" not in receipt
        assert tiny_source.actor.is_clean()
        assert tiny_source.repo.head_commit() == tiny_source.base
        assert "return x + 2" in Path(tiny_source.actor.path.path, PATH).read_text()
    finally:
        controller.close()


def test_source_carrier_owns_patch_and_nested_declarations(tmp_path, tiny_source):
    instance, controller, outcome = selected_actor(tmp_path, revision=tiny_source.base)
    try:
        selected = instance.materialize_actor(outcome)
        assignment, manifest, proposal = source_inputs(tiny_source, selected)
        result = advice(selected)
        bound = bound_source(instance, selected, tiny_source, result,
                             authored_manifest=manifest, legacy_proposal=proposal)
        digest = bound.digest
        manifest.declared_symbols[PATH] = ("forged",)
        proposal["change"]["estimated_diff_size"] = 999
        bound.to_dict()["authored_manifest"]["declared_symbols"][PATH].append("forged")
        assert bound.digest == digest
        assert "forged" not in bound.to_dict()["authored_manifest"]["declared_symbols"][PATH]
        with pytest.raises(bridge.PreparationBindingRefused, match="immutable bytes"):
            bridge.BoundSourcePreparation(bytearray(bound.canonical))
    finally:
        controller.close()


@pytest.mark.parametrize("mutation", ["status", "target", "request", "mechanism", "symbol",
                                     "campaign", "manifest_base", "legacy_id", "estimate",
                                     "small_estimate"])
def test_source_mismatches_refuse_before_any_application(tmp_path, tiny_source, mutation):
    instance, controller, outcome = selected_actor(tmp_path, revision=tiny_source.base)
    try:
        selected = instance.materialize_actor(outcome)
        result = advice(selected)
        assignment, manifest, proposal = source_inputs(tiny_source, selected)
        kwargs = {}
        if mutation == "status":
            result = replace(result, status="cooldown")
        elif mutation == "target":
            result = replace(result, target_revision_digest="d" * 64)
        elif mutation == "request":
            result = replace(result, request_digest="d" * 64)
        elif mutation in {"mechanism", "symbol"}:
            key = "mechanism" if mutation == "mechanism" else "target_symbol"
            result = replace(result, proposed_output=dict(result.proposed_output) | {key: "forged"})
        elif mutation == "campaign":
            kwargs["assignment"] = replace(assignment, campaign_id="ak-foreign")
        elif mutation == "manifest_base":
            kwargs["authored_manifest"] = replace(manifest, instrument_commit="c" * 40)
        elif mutation == "legacy_id":
            kwargs["legacy_proposal"] = proposal | {"proposal_id": "akp-foreign"}
        else:
            kwargs["legacy_proposal"] = proposal | {"change": proposal["change"] |
                                                   {"estimated_diff_size":
                                                    1 if mutation == "small_estimate" else True}}
        with pytest.raises((bridge.PreparationBindingRefused, source.SourceCandidateError)):
            bound_source(instance, selected, tiny_source, result, **kwargs)
        assert tiny_source.actor.head_commit() == tiny_source.base
        assert tiny_source.actor.is_clean()
    finally:
        controller.close()


def build_inputs(tmp_path, fixture):
    destination = W.SandboxPath.create(str(tmp_path / "snapshot"), sandbox_root=str(tmp_path),
                                       production_trees=())
    snapshot = fixture.repo.add_worktree(destination, fixture.base, detach=True)
    fixture.addCleanup(lambda: fixture.repo.remove_worktree(destination, force=True))
    plan = W.BuildPlan(
        snapshot.path,
        W.SandboxPath.create(str(tmp_path / "build"), sandbox_root=str(tmp_path),
                             production_trees=()),
        fixture.actor.path, W.BuildParallelism(1, cpu_list="0"),
        targets=("fixture",), cmake_defines=(("X", "1"),), cmake="/usr/bin/cmake")
    return plan, snapshot


def test_build_only_exact_native_runner_delegation_without_execution(tmp_path, tiny_source,
                                                                   monkeypatch):
    instance, controller, outcome = selected_actor(
        tmp_path, kind="build_recipe", revision=tiny_source.base)
    try:
        selected = instance.materialize_actor(outcome)
        plan, snapshot = build_inputs(tmp_path, tiny_source)
        result = advice(selected, options=[f"-D{k}={v}" for k, v in plan.effective_defines])
        bound = bridge.bind_build_preparation(
            selected_actor_work=selected, preparation_result=result,
            resolved_campaign=instance.resolved, source_commit=tiny_source.base,
            build_plan=plan, source_worktree=snapshot)
        with pytest.raises(bridge.PreparationBindingRefused, match="immutable bytes"):
            bridge.BoundBuildPreparation(bytearray(bound.canonical), plan)
        calls = []
        sentinel = object()
        def recorded(native_plan, **kwargs):
            calls.append((native_plan, kwargs))
            return sentinel
        monkeypatch.setattr(W, "run_build", recorded)
        received = bridge.delegate_build(
            bound, campaign_driver=instance, runner=W.run_build, source_worktree=snapshot,
            log_path=str(tmp_path / "build.log"), configure_timeout_s=5, build_timeout_s=10,
            env={"LANG": "C"}, sandbox_cgroup_root="/fixture-owned-cgroup")
        assert received is sentinel  # no invented BuildResult/identity
        forwarded, kwargs = calls[0]
        assert forwarded.configure_argv() == plan.configure_argv()
        assert forwarded.build_argv() == plan.build_argv()
        assert kwargs == {"log_path": str(tmp_path / "build.log"), "configure_timeout_s": 5,
                          "build_timeout_s": 10, "env": {"LANG": "C"},
                          "require_fresh_build_dir": True,
                          "sandbox_cgroup_root": "/fixture-owned-cgroup"}
        assert tiny_source.actor.head_commit() == tiny_source.base
        assert not (tmp_path / "build").exists()
    finally:
        controller.close()


@pytest.mark.parametrize("mutation", ["options", "system", "source", "deadline", "owner",
                                     "snapshot"])
def test_build_binding_and_delegation_fail_closed(tmp_path, tiny_source, mutation):
    instance, controller, outcome = selected_actor(
        tmp_path, kind="build_recipe", revision=tiny_source.base)
    try:
        selected = instance.materialize_actor(outcome)
        plan, snapshot = build_inputs(tmp_path, tiny_source)
        result = advice(selected, options=[f"-D{k}={v}" for k, v in plan.effective_defines])
        if mutation == "options":
            result = replace(result, proposed_output=dict(result.proposed_output) |
                             {"configured_options": ["-DUNREVIEWED=ON"]})
        if mutation == "system":
            result = replace(result, proposed_output=dict(result.proposed_output) |
                             {"build_system": "shell"})
        inputs = dict(selected_actor_work=selected, preparation_result=result,
                      resolved_campaign=instance.resolved,
                      source_commit="c" * 40 if mutation == "source" else tiny_source.base,
                      build_plan=plan, source_worktree=snapshot)
        calls = []
        with pytest.raises((bridge.PreparationBindingRefused, driver.DriverRefused)):
            bound = bridge.bind_build_preparation(**inputs)
            if mutation == "owner":
                controller.close()
            if mutation == "snapshot":
                (tmp_path / "snapshot" / PATH).write_text("changed")
            bridge.delegate_build(
                bound, campaign_driver=instance, runner=lambda *a, **k: calls.append((a, k)),
                source_worktree=snapshot, log_path=str(tmp_path / "log"),
                configure_timeout_s=float("inf") if mutation == "deadline" else 5,
                build_timeout_s=10, env={}, sandbox_cgroup_root="/fixture-cgroup")
        assert calls == []
    finally:
        controller.close()


def test_source_owner_replacement_refuses_before_mutation(tmp_path, tiny_source):
    instance, controller, outcome = selected_actor(tmp_path, revision=tiny_source.base)
    selected = instance.materialize_actor(outcome)
    bound = bound_source(instance, selected, tiny_source)
    controller.close()
    with pytest.raises(driver.DriverRefused):
        bridge.materialize_source(bound, campaign_driver=instance, actor_worktree=tiny_source.actor)
    assert tiny_source.actor.head_commit() == tiny_source.base


@pytest.mark.parametrize("mutation", ["jobs", "cpu", "no_affinity", "duplicate_option",
                                     "wide_timeout", "nan_timeout"])
def test_build_cannot_widen_configured_resources(tmp_path, tiny_source, mutation):
    instance, controller, outcome = selected_actor(
        tmp_path, kind="build_recipe", revision=tiny_source.base)
    try:
        selected = instance.materialize_actor(outcome)
        plan, snapshot = build_inputs(tmp_path, tiny_source)
        if mutation in {"jobs", "cpu", "no_affinity"}:
            plan = replace(plan, parallelism=W.BuildParallelism(
                2 if mutation == "jobs" else 1,
                cpu_list=None if mutation == "no_affinity" else
                "99" if mutation == "cpu" else "0"))
        if mutation == "duplicate_option":
            plan = replace(plan, cmake_defines=(("X", "1"), ("X", "2")))
        result = advice(selected, options=[f"-D{k}={v}" for k, v in plan.effective_defines])
        calls = []
        with pytest.raises(bridge.PreparationBindingRefused):
            bound = bridge.bind_build_preparation(
                selected_actor_work=selected, preparation_result=result,
                resolved_campaign=instance.resolved, source_commit=tiny_source.base,
                build_plan=plan, source_worktree=snapshot)
            bridge.delegate_build(
                bound, campaign_driver=instance, runner=lambda *a, **k: calls.append((a, k)),
                source_worktree=snapshot, log_path=str(tmp_path / "log"),
                configure_timeout_s=float("nan") if mutation == "nan_timeout" else 50,
                build_timeout_s=50, env={}, sandbox_cgroup_root="/fixture-cgroup")
        assert not calls
    finally:
        controller.close()


def test_native_source_failure_is_not_relabelled_as_prepared(tmp_path, tiny_source, monkeypatch):
    instance, controller, outcome = selected_actor(tmp_path, revision=tiny_source.base)
    try:
        bound = bound_source(instance, instance.materialize_actor(outcome), tiny_source)
        failure = source.SourceCandidateError("original source refusal")
        def refused(*args, **kwargs):
            raise failure
        monkeypatch.setattr(source, "apply_source_candidate", refused)
        with pytest.raises(source.SourceCandidateError) as caught:
            bridge.materialize_source(bound, campaign_driver=instance,
                                      actor_worktree=tiny_source.actor)
        assert caught.value is failure
        assert tiny_source.actor.head_commit() == tiny_source.base
    finally:
        controller.close()


def test_unsupported_missing_and_mutated_source_inputs_are_not_advice_authority(tmp_path,
                                                                            tiny_source):
    instance, controller, outcome = selected_actor(tmp_path, revision=tiny_source.base)
    try:
        selected = instance.materialize_actor(outcome)
        with pytest.raises(bridge.PreparationBindingRefused, match="actual authored"):
            bound_source(instance, selected, tiny_source, authored_manifest=None)
        result = replace(advice(selected), actor_profile_digest=None)
        with pytest.raises(bridge.PreparationBindingRefused):
            bound_source(instance, selected, tiny_source, result)
        bound = bound_source(instance, selected, tiny_source)
        Path(tiny_source.actor.path.path, PATH).write_text("unexpected dirty source")
        with pytest.raises(source.SourceCandidateError, match="dirty"):
            bridge.materialize_source(bound, campaign_driver=instance,
                                      actor_worktree=tiny_source.actor)
        assert tiny_source.actor.head_commit() == tiny_source.base
    finally:
        controller.close()


def test_self_consistent_substituted_selected_payload_is_not_current_authority(tmp_path,
                                                                            tiny_source):
    instance, controller, outcome = selected_actor(tmp_path, revision=tiny_source.base)
    try:
        raw = instance.materialize_actor(outcome).to_dict()
        request = raw["actor_request"]
        request["prompt"] += "\\nsubstituted request"
        request["cache_key"] = actor._digest({key: request[key] for key in (
            "actor_kind", "actor_identity", "prompt", "mandatory_conflicts", "proposal")})
        raw["stage_plan_digest"] = request["cache_key"]
        substituted = driver.SelectedActorWork.from_dict(raw)
        bound = bound_source(instance, substituted, tiny_source)
        with pytest.raises(bridge.PreparationBindingRefused, match="current selected"):
            bridge.materialize_source(bound, campaign_driver=instance,
                                      actor_worktree=tiny_source.actor)
        assert tiny_source.actor.head_commit() == tiny_source.base
    finally:
        controller.close()
