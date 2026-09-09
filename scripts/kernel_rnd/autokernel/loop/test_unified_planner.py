from __future__ import annotations

from pathlib import Path
import hashlib
import json
import tempfile

import pytest

from . import campaign, scheduling, scoped_evidence as evidence, serving
from . import unified_planner as planner
from .resolved_recipe import (ARTIFACT_SCHEMA, ENVIRONMENT_POLICY_SCHEMA,
                              CanonicalResolvedRecipe, resolve_canonical_launch)
from .test_experiment_plan import plan_dict


def artifact(kind: str, ref: str, digit: str) -> dict:
    return {"schema": campaign.ARTIFACT_SCHEMA, "kind": kind, "ref": ref,
            "path": f"/artifact/{kind}/{ref}", "sha256": digit * 64}


def resolved_campaign(*, backend="cpu", include_missing=False, model_digit="a",
                      build_digit="b", recipe_digit="d", context=128,
                      concurrency=1, target_env=None, recipe_sha=None,
                      campaign_id="unified-test", request_id="request-1"):
    def target(target_id, model="model"):
        return {"schema": campaign.TARGET_SCHEMA, "request_id": request_id,
                "target_id": target_id, "backend": backend, "model_ref": model,
                "build_ref": "production:test:executable",
                "recipe_ref": "production:test:recipe", "baseline_ref": "base",
                "context": context, "concurrency": concurrency,
                "speculation": "none", "env": target_env or {},
                "metric": "aggregate_tok_s", "metric_direction": "higher",
                "roles": [target_id], "required_obligations": [target_id]}
    targets = [target("prod")]
    if include_missing:
        targets.append(target("future", "missing-model"))
    raw = {"schema": campaign.MANIFEST_SCHEMA, "campaign_id": campaign_id,
           "request_id": request_id, "source_snapshot": {"kernel": "source"},
           "resources": {"schema": campaign.RESOURCE_SCHEMA, "cpu_logical": [0, 1],
                         "gpu_ids": ["ROCm0"] if backend == "gpu" else [],
                         "stage_timeout_s": 60, "build_timeout_s": 60,
                         "build_jobs": 1, "max_builds": 1},
           "objective_ref": "objective", "actors": {"planner": "offline"},
           "fallbacks": {"planner": []}, "production": targets, "seeds": []}
    registry = {"source": {"source": artifact("source", "source", "1")},
                "model": {"model": artifact("model", "model", model_digit)},
                "build": {"production:test:executable": artifact(
                              "build", "production:test:executable", build_digit),
                          "base": artifact("build", "base", "4")},
                "recipe": {"production:test:recipe": artifact(
                    "recipe", "production:test:recipe", recipe_digit)}}
    if recipe_sha is not None:
        registry["recipe"]["production:test:recipe"]["sha256"] = recipe_sha
    return campaign.resolve_manifest(campaign.CampaignManifest.from_dict(raw),
                                     registry_snapshot=registry)


def canonical_recipe(*, backend="cpu", threads=4, policy_keys=(), launch_extra=None,
                     numa_policy=None, background_threads=True,
                     instance_mode="full") -> CanonicalResolvedRecipe:
    root = Path("/build")
    template = serving.Recipe(
        name=f"{backend}-recipe", model="/models/model.gguf",
        device="none" if backend == "cpu" else "ROCm0",
        ngl=0 if backend == "cpu" else 99, cpu_list="0-3", threads=threads,
        np=1, ctx=128)
    full = template.server_argv(root, 8000)
    if not background_threads:
        offset = full.index("-tb")
        del full[offset:offset + 2]
    policy = {"schema": ENVIRONMENT_POLICY_SCHEMA, "version": "test-policy-v1",
              "measurement_keys": list(policy_keys),
              "allowed_inherit_keys": [],
              "witnesses": {key: "runtime_set" for key in policy_keys}}
    launch = template.server_env(root, base={})
    launch.update(launch_extra or {})
    artifacts = {
        "model": {"schema": ARTIFACT_SCHEMA, "role": "model",
                  "path": template.model, "sha256": "a" * 64},
        "drafter": None,
        "executable": {"schema": ARTIFACT_SCHEMA, "role": "executable",
                       "path": "/build/bin/llama-server", "sha256": "b" * 64},
        "dsos": [{"schema": ARTIFACT_SCHEMA, "role": "dso",
                  "path": "/build/bin/libggml.so", "sha256": "c" * 64}],
    }
    prefix = full[:3]
    if numa_policy is not None:
        prefix = ["numactl", numa_policy, "--", *prefix]
    return resolve_canonical_launch(
        template, build_dir=root, topology_prefix=prefix, command_argv=full[3:],
        launch_environment=launch,
        artifact_identities=artifacts, backend=backend, environment_policy=policy,
        port=8000, runtime_binary_dir="/build/bin", runtime_ld_paths=["/build/bin"],
        provenance={"export_sha256": "d" * 64, "instance_mode": instance_mode,
                    "source:launcher": "e" * 64})


def dimension(kind="threads", anchor=4, candidate=8, ident="threads-4-8"):
    return {"schema": planner.DIMENSION_SCHEMA, "dimension_id": ident, "kind": kind,
            "anchor": anchor, "candidate": candidate,
            "authority_ref": "runtime-registry:v1"}


def claim(backend="cpu", quant="Q8_0", pair=None, target=None, allocation=None):
    target_id = target.target_ids[0] if target else "prod"
    model = target.execution.model.sha256 if target else "a" * 64
    workload = target.workload_signature if target else "0" * 64
    return evidence.ClaimKey.from_dict({
        "schema": evidence.CLAIM_KEY_SCHEMA,
        "target_scope": {"target": target_id, "backend": backend, "model": model,
                         "quant": quant, "workload": workload,
                         "allocation": allocation or "0" * 64},
        "control_identity": ({"execution_digest": pair.anchor.execution_digest}
                             if pair else {"recipe": "anchor"}),
        "intervention_identity": ({"execution_digest": pair.candidate.execution_digest}
                                  if pair else {"recipe": "candidate"}),
        "mechanism_identity": {"id": "threads"}, "estimand": "level",
        "metric": "aggregate_tok_s", "metric_direction": "higher",
        "effect_question": {"kind": "absolute_effect_bound", "bound": 1.0,
                            "unit": "percent"},
        "dependency_identities": {"recipe": "f" * 64}})


def opportunity(*, backend="cpu", kind="runtime_recipe", quant="Q8_0", pair=None,
                target=None, allocation=None):
    return {"schema": planner.OPPORTUNITY_SCHEMA, "opportunity_id": f"opp-{kind}",
            "kind": kind, "mechanism_id": "threads", "estimand": "level",
            "metric": "aggregate_tok_s", "metric_direction": "higher",
            "effect_question": {"kind": "absolute_effect_bound", "bound": 1.0,
                                "unit": "percent"},
            "changed_factors": ["threads"], "instrument": "serving",
            "unit": "process", "required_witnesses": ["native-capture-v1"],
            "stage_class": "search", "estimated_duration_seconds": 10,
            "runtime_dimension_ids": (["threads-4-8"] if kind == "runtime_recipe" else []),
            "claim_key": claim(backend, quant, pair, target, allocation).to_dict()}


def profile(target, *, backend="cpu", opportunities=None, quant="Q8_0", pair=None):
    resource = {"schema": scheduling.VECTOR_SCHEMA,
                "physical_region_fraction": 1.0,
                "gpu_devices": ["ROCm0"] if backend == "gpu" else [],
                "memory_reservation_bytes": 0}
    allocation = scheduling.ResourceVector.from_dict(resource).digest
    return {"schema": planner.PROFILE_SCHEMA,
            "target_revision_digest": planner._target_digest(target), "freshness": "fresh",
            "quant": quant, "hotspots": ["small-op-overhead"],
            "observation_states": ["unknown"], "kept_scope": [],
            "resource_cost": resource,
            "opportunities": opportunities or [opportunity(backend=backend, quant=quant,
                                                             pair=pair, target=target,
                                                             allocation=allocation)]}


def recipe_artifact_json(recipe):
    artifacts = []
    for use, item in (("model", recipe.model), ("drafter", recipe.drafter),
                      ("executable", recipe.executable)):
        if item is not None:
            artifacts.append({"use": use, "path": item.path, "sha256": item.sha256})
    artifacts.extend({"use": "dso", "path": item.path, "sha256": item.sha256}
                     for item in recipe.dsos)
    launch = {
        "backend": recipe.backend, "port": recipe.port,
        "numa_instance": "test", "command_argv": list(recipe.command_argv),
        "environment": dict(recipe.launch_env),
        "environment_unsets": list(recipe.absent_environment),
        "workload": {"context": str(recipe.template.ctx),
                     "np": str(recipe.template.np),
                     "threads": str(recipe.template.threads)},
        "topology": {"argv_prefix": list(recipe.topology_prefix)},
        "runtime_requirements": {"binary_dir": recipe.runtime_binary_dir,
                                 "ld_library_path": list(recipe.runtime_ld_paths)},
        "source_revisions": {}, "source_revision_kinds": {},
        "speculation": recipe.capability.speculation,
    }
    return json.dumps({"schema": "autokernel-production-launch-recipe/v1",
                       "launch": launch,
                       "artifacts": sorted(artifacts,
                                           key=lambda row: (row["use"], row["path"]))},
                      sort_keys=True, separators=(",", ":"))


def campaign_for_recipe(recipe, **kwargs):
    raw = recipe_artifact_json(recipe)
    return resolved_campaign(backend=recipe.backend,
                             recipe_sha=hashlib.sha256(raw.encode()).hexdigest(), **kwargs)


def runtime_anchor(target, recipe):
    raw = recipe_artifact_json(recipe)
    root = Path(tempfile.mkdtemp(prefix="autokernel-planner-anchor-"))
    sidecar = root / "recipe.json"
    sidecar.write_text(raw)
    artifact = json.loads(raw)
    launch = artifact["launch"]
    export_target = {
        "target_id": target.target_ids[0], "status": "ready", "reasons": [],
        "argv": list(recipe.argv), "aliases": [],
        "obligations": list(target.required_obligations),
        "primary_role": target.target_ids[0], "optional_seed": False,
        **launch,
        "artifacts": artifact["artifacts"] + [{"use": "recipe", "path": str(sidecar),
                                                "sha256": hashlib.sha256(raw.encode()).hexdigest()}],
    }
    export = {"schema": "autokernel-production-enrollment/v1",
              "context": {"sources": [{"name": "test", "path": "/source/test.py",
                                          "sha256": "e" * 64, "revision": "rev"}],
                          "instance_mode": dict(recipe.provenance)["instance_mode"]},
              "targets": [export_target],
              "disposition": {"ready": 1, "waiting_artifact": 0, "unsupported": 0}}
    export["export_sha256"] = hashlib.sha256(json.dumps(
        export, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"schema": planner.ANCHOR_SCHEMA,
            "target_revision_digest": planner._target_digest(target),
            "target_id": target.target_ids[0], "production_export": export,
            "environment_policy": recipe.environment_policy.to_dict()}


def prepared(enrolled, anchors=None):
    return planner.prepare_runtime_anchors(enrolled, anchors or {})


def scheduler(backend="cpu"):
    vector = scheduling.ResourceVector(1.0, ("ROCm0",) if backend == "gpu" else (), 0)
    config = scheduling.SchedulerConfig(
        config_id="test", max_stage_seconds=60, noncoverage_slots=0,
        reservation_slots={}, reservation_shares={}, campaign_attempt_cap=10,
        campaign_charged_seconds_cap=600, seed_attempt_cap=3,
        seed_charged_seconds_cap=180, capacity=vector,
        weights_source="fixed", apportionment_rule="coverage-first",
        adaptive_rule_id=None)
    state = scheduling.initial_state(config, "test-scheduler")
    return config, scheduling.SchedulerEngine(config, state)


def experiment(pair, target_digest, campaign_id="unified-test"):
    raw = plan_dict(instrument="serving", unit="process")
    raw.update(campaign_id=campaign_id, target_revision=target_digest,
               metric="aggregate_tok_s", changed_factors=["threads"],
               required_witnesses=["native-capture-v1"],
               anchor_identity={"execution_digest": pair.anchor.execution_digest},
               candidate_identity={"execution_digest": pair.candidate.execution_digest})
    return raw


def run(*, backend="cpu", include_missing=False, source_actor=None, build_actor=None,
        stop=lambda: False, stage_handler=None):
    anchor = canonical_recipe(backend=backend)
    enrolled = campaign_for_recipe(anchor, include_missing=include_missing)
    ready = next(row for row in enrolled.targets if row.status == "ready")
    digest = planner._target_digest(ready)
    _, engine = scheduler(backend)
    pair = planner.enumerate_runtime_dimensions(anchor, [dimension()])[0]
    return planner.plan_iteration(
        resolved_campaign=enrolled, profiles={digest: profile(ready, backend=backend, pair=pair)},
        evidence_index=evidence.EvidenceIndex((), current_epoch="epoch-1"),
        runtime_anchors=prepared(enrolled, {digest: runtime_anchor(ready, anchor)}),
        runtime_dimensions={digest: [dimension()]},
        source_actor=source_actor or (lambda *_: (_ for _ in ()).throw(AssertionError())),
        build_actor=build_actor or (lambda *_: (_ for _ in ()).throw(AssertionError())),
        scheduler_engine=engine,
        experiment_plans={"opp-runtime_recipe:threads-4-8": experiment(pair, digest)},
        now=1.0, native_artifact_sink_ref="native:capture",
        stage_handler=stage_handler, stop_requested=stop)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_runtime_sweep_revalidates_canonical_recipe_skips_actors_and_dispatches_advice(backend):
    recorded = []
    result = run(backend=backend, stage_handler=recorded.append)
    assert len(result.proposals) == 1
    assert result.proposals[0].kind == "runtime_recipe"
    assert result.selection.status == "selected"
    assert recorded == [result.dispatch]
    assert result.dispatch.execution_authorized is False
    assert result.dispatch.proposal["experiment_plan_digest"]
    assert result.dispatch.proposal["native_artifact_sink_ref"] == "native:capture"
    assert result.dispatch.experiment_intent["proposal_digest"] == result.proposals[0].digest
    assert result.dispatch.experiment_intent["experiment_plan_digest"] == \
        result.dispatch.proposal["experiment_plan_digest"]
    assert result.dispatch.experiment_intent["arm_scalars_are_gain_evidence"] is False


def test_threads_and_cpu_list_are_concrete_validated_dimensions_and_env_refuses_scoped():
    base = canonical_recipe()
    pairs = planner.enumerate_runtime_dimensions(base, [
        dimension(),
        dimension("cpu_list", "0-3", "4-7", "numa-placement"),
    ])
    assert [pair.dimension.dimension_id for pair in pairs] == [
        "numa-placement", "threads-4-8"]
    assert all(pair.anchor.executable == pair.candidate.executable for pair in pairs)
    env_base = canonical_recipe(policy_keys=("KNOB", "OMP_PROC_BIND"),
                                launch_extra={"OMP_PROC_BIND": "close"})
    env_pairs = planner.enumerate_runtime_dimensions(env_base, [
        dimension("env", {"key": "KNOB", "value": None},
                  {"key": "KNOB", "value": "1"}, "set-knob")])
    assert dict(env_pairs[0].candidate.launch_env)["KNOB"] == "1"
    assert dict(env_pairs[0].candidate.launch_env)["OMP_PROC_BIND"] == "close"
    unset = planner.enumerate_runtime_dimensions(env_pairs[0].candidate, [
        dimension("env", {"key": "KNOB", "value": "1"},
                  {"key": "KNOB", "value": None}, "unset-knob")])
    assert "KNOB" in unset[0].candidate.absent_environment
    with pytest.raises(planner.PlanningRefused, match="unsupported"):
        planner.enumerate_runtime_dimensions(base, [
            dimension("env", {"key": "UNKNOWN", "value": None},
                      {"key": "UNKNOWN", "value": "1"}, "unknown")])
    numa = canonical_recipe(numa_policy="--membind=0")
    numa_pairs = planner.enumerate_runtime_dimensions(numa, [
        dimension("numa_policy", "--membind=0", "--interleave=0", "numa-policy")])
    assert numa_pairs[0].candidate.topology_prefix[1] == "--interleave=0"
    without_tb = planner.enumerate_runtime_dimensions(
        canonical_recipe(background_threads=False), [dimension()])
    assert "-tb" not in without_tb[0].candidate.command_argv


def test_noop_multifactor_and_tampered_arm_are_refused():
    with pytest.raises(planner.PlanningRefused, match="no-op"):
        planner.RuntimeDimension.from_dict(dimension(candidate=4))
    raw = opportunity()
    raw["changed_factors"] = ["threads", "env:KNOB"]
    with pytest.raises(planner.PlanningRefused, match="one exact factor"):
        planner.Opportunity.from_dict(raw)
    pair = planner.enumerate_runtime_dimensions(canonical_recipe(), [dimension()])[0]
    bad = pair.to_dict()
    bad["candidate"]["execution_digest"] = "0" * 64
    with pytest.raises(planner.PlanningRefused, match="invalid"):
        planner.RuntimeArmPair.from_dict(bad)


def test_missing_optional_target_and_missing_profile_do_not_block_ready_peer():
    result = run(include_missing=True)
    assert result.selection.status == "selected"
    assert any(row["status"] == "prerequisite" for row in result.dispositions)


def test_quant_scope_is_exact_and_not_transferred():
    enrolled = resolved_campaign()
    target = enrolled.targets[0]
    bad_profile = profile(target, quant="Q4_K", opportunities=[opportunity(quant="Q8_0")])
    normalized = planner.TargetProfile.from_dict(bad_profile)
    digest = planner._target_digest(target)
    _, engine = scheduler()
    result = planner.plan_iteration(
        resolved_campaign=enrolled, profiles={digest: normalized},
        evidence_index=evidence.EvidenceIndex((), current_epoch="epoch-1"),
        runtime_anchors=prepared(enrolled),
        runtime_dimensions={digest: [dimension()]},
        source_actor=lambda *_: pytest.fail("must not author"),
        build_actor=lambda *_: pytest.fail("must not build"),
        scheduler_engine=engine, experiment_plans={}, now=1,
        native_artifact_sink_ref="native:capture")
    assert not result.proposals
    assert result.dispositions[0]["status"] == "claim_scope_mismatch"


def test_stop_before_next_target_or_stage_handler_preserves_no_execution():
    calls = 0
    def stop():
        nonlocal calls
        calls += 1
        return True
    result = run(stop=stop, stage_handler=lambda _: pytest.fail("must not dispatch"))
    assert not result.proposals
    assert result.selection is None
    assert result.scheduler_state.round_number == 0
    assert result.dispatch is None


def test_direct_object_revalidation_rejects_nonfinite_duration():
    raw = opportunity()
    raw["estimated_duration_seconds"] = float("nan")
    with pytest.raises(planner.PlanningRefused, match="finite"):
        planner.Opportunity.from_dict(raw)


@pytest.mark.parametrize("kind", ["source", "build_recipe"])
def test_source_and_build_route_only_their_actor_as_pending_plan_prerequisite(kind):
    enrolled = resolved_campaign()
    target = enrolled.targets[0]
    digest = planner._target_digest(target)
    base_profile = profile(target)
    allocation = scheduling.ResourceVector.from_dict(base_profile["resource_cost"]).digest
    opp = planner.Opportunity.from_dict(opportunity(
        kind=kind, target=target, allocation=allocation))
    calls = []

    def actor(prompt, conflicts, snapshot):
        calls.append((kind, prompt, conflicts))
        return {
            "schema": planner.PROPOSAL_SCHEMA, "proposal_id": f"proposal-{kind}",
            "target_revision_digest": digest, "backend": "cpu",
            "parent_identity": {"target_ids": ["prod"], "target_revision": 1},
            "control_identity": dict(opp.claim_key.control_identity),
            "intervention_identity": dict(opp.claim_key.intervention_identity),
            "kind": kind, "mechanism_id": opp.mechanism_id,
            "estimand": opp.estimand, "metric": opp.metric,
            "metric_direction": opp.metric_direction,
            "effect_question": dict(opp.effect_question),
            "changed_factors": list(opp.changed_factors),
            "instrument": opp.instrument, "unit": opp.unit,
            "required_witnesses": list(opp.required_witnesses),
            "stage_class": opp.stage_class,
            "estimated_duration_seconds": opp.estimated_duration_seconds,
            "experiment_plan_digest": None, "native_artifact_sink_ref": "native:capture",
            "runtime_pair": None, "claim_key": opp.claim_key.to_dict(),
            "evidence_snapshot": snapshot.to_dict(),
        }

    _, engine = scheduler()
    result = planner.plan_iteration(
        resolved_campaign=enrolled,
        profiles={digest: profile(target, opportunities=[opp.to_dict()])},
        evidence_index=evidence.EvidenceIndex((), current_epoch="epoch-1"),
        runtime_anchors=prepared(enrolled), runtime_dimensions={},
        source_actor=actor if kind == "source" else lambda *_: pytest.fail("wrong actor"),
        build_actor=actor if kind == "build_recipe" else lambda *_: pytest.fail("wrong actor"),
        scheduler_engine=engine, experiment_plans={}, now=1,
        native_artifact_sink_ref="native:capture")
    assert len(calls) == 1
    assert result.stage_proposals[0].stage_class == "prerequisite"
    assert result.stage_proposals[0].frontier_id == digest
    assert result.dispatch.experiment_intent["experiment_plan_digest"] is None
    assert result.dispatch.execution_authorized is False
    if kind == "source":
        def forged_plan_actor(*args):
            payload = actor(*args)
            payload["experiment_plan_digest"] = "1" * 64
            return payload
        _, second_engine = scheduler()
        with pytest.raises(planner.PlanningRefused, match="cannot claim a final"):
            planner.plan_iteration(
                resolved_campaign=enrolled,
                profiles={digest: profile(target, opportunities=[opp.to_dict()])},
                evidence_index=evidence.EvidenceIndex((), current_epoch="epoch-1"),
                runtime_anchors=prepared(enrolled), runtime_dimensions={},
                source_actor=forged_plan_actor,
                build_actor=lambda *_: pytest.fail("wrong actor"),
                scheduler_engine=second_engine, experiment_plans={}, now=1,
                native_artifact_sink_ref="native:capture")


def test_missing_final_plan_is_retained_as_pending_and_never_dispatched():
    anchor = canonical_recipe()
    enrolled = campaign_for_recipe(anchor)
    target = enrolled.targets[0]
    digest = planner._target_digest(target)
    pair = planner.enumerate_runtime_dimensions(anchor, [dimension()])[0]
    _, engine = scheduler()
    result = planner.plan_iteration(
        resolved_campaign=enrolled, profiles={digest: profile(target, pair=pair)},
        evidence_index=evidence.EvidenceIndex((), current_epoch="epoch-1"),
        runtime_anchors=prepared(enrolled, {digest: runtime_anchor(target, anchor)}),
        runtime_dimensions={digest: [dimension()]},
        source_actor=lambda *_: pytest.fail("no source actor"),
        build_actor=lambda *_: pytest.fail("no build actor"), scheduler_engine=engine,
        experiment_plans={}, now=1, native_artifact_sink_ref="native:capture")
    assert not result.proposals and result.dispatch is None
    assert result.dispositions[0]["status"] == "pending_experiment_plan"


def test_projection_outage_blocks_reuse_support_not_fresh_runtime_hypothesis():
    anchor = canonical_recipe()
    enrolled = campaign_for_recipe(anchor)
    target = enrolled.targets[0]
    digest = planner._target_digest(target)
    pair = planner.enumerate_runtime_dimensions(anchor, [dimension()])[0]
    _, engine = scheduler()
    result = planner.plan_iteration(
        resolved_campaign=enrolled, profiles={digest: profile(target, pair=pair)},
        evidence_index=evidence.EvidenceIndex(
            (), current_epoch="epoch-1", projection_available=False),
        runtime_anchors=prepared(enrolled, {digest: runtime_anchor(target, anchor)}),
        runtime_dimensions={digest: [dimension()]},
        source_actor=lambda *_: pytest.fail("no source actor"),
        build_actor=lambda *_: pytest.fail("no build actor"), scheduler_engine=engine,
        experiment_plans={"opp-runtime_recipe:threads-4-8": experiment(pair, digest)},
        now=1, native_artifact_sink_ref="native:capture")
    assert result.selection.status == "selected"
    assert result.proposals[0].evidence_snapshot["retrieval_complete"] is False
    assert result.proposals[0].evidence_snapshot["supported_for_intended_use"] is False


def test_runtime_dimension_nested_values_are_immutable():
    source = dimension("env", {"key": "KNOB", "value": None},
                       {"key": "KNOB", "value": "1"}, "env")
    loaded = planner.RuntimeDimension.from_dict(source)
    source["candidate"]["value"] = "2"
    assert loaded.candidate["value"] == "1"
    with pytest.raises(TypeError):
        loaded.candidate["value"] = "3"


def test_validated_quarter_export_provenance_is_preserved_not_relabelled_full():
    recipe = canonical_recipe(instance_mode="quarter")
    enrolled = campaign_for_recipe(recipe)
    target = enrolled.targets[0]
    digest = planner._target_digest(target)
    rebound = prepared(enrolled, {digest: runtime_anchor(target, recipe)}).recipes[digest]
    assert dict(rebound.provenance)["instance_mode"] == "quarter"


@pytest.mark.parametrize(("campaign_kwargs", "wrong_recipe_pin"), [
    ({"model_digit": "2"}, False), ({"build_digit": "3"}, False),
    ({}, True), ({"context": 256}, False), ({"concurrency": 2}, False),
    ({"target_env": {"KNOB": "1"}}, False),
])
def test_runtime_anchor_must_match_every_enrolled_execution_binding(
        campaign_kwargs, wrong_recipe_pin):
    canonical = canonical_recipe()
    enrolled = campaign_for_recipe(canonical, **campaign_kwargs)
    target = enrolled.targets[0]
    digest = planner._target_digest(target)
    # The validated export still differs in one enrolled identity and must be refused.
    anchor_binding = runtime_anchor(target, canonical)
    if wrong_recipe_pin:
        anchor_binding["production_export"]["targets"][0]["artifacts"][-1]["sha256"] = "4" * 64
        anchor_binding["production_export"]["export_sha256"] = hashlib.sha256(json.dumps(
            {key: value for key, value in anchor_binding["production_export"].items()
             if key != "export_sha256"}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    with pytest.raises(planner.PlanningRefused):
        prepared(enrolled, {digest: anchor_binding})


def test_prepared_anchors_cannot_be_reused_for_changed_campaign():
    canonical = canonical_recipe()
    first = campaign_for_recipe(canonical)
    target = first.targets[0]
    digest = planner._target_digest(target)
    cached = prepared(first, {digest: runtime_anchor(target, canonical)})
    changed = resolved_campaign(
        campaign_id="unified-test", request_id="request-2",
        recipe_sha=target.execution.recipe.sha256)
    _, engine = scheduler()
    with pytest.raises(planner.PlanningRefused, match="differ from resolved campaign"):
        planner.plan_iteration(
            resolved_campaign=changed, profiles={},
            evidence_index=evidence.EvidenceIndex((), current_epoch="epoch-1"),
            runtime_anchors=cached, runtime_dimensions={},
            source_actor=lambda *_: pytest.fail("must not author"),
            build_actor=lambda *_: pytest.fail("must not build"),
            scheduler_engine=engine, experiment_plans={}, now=1,
            native_artifact_sink_ref="native:capture")

    # A moved resolved artifact retains the same source manifest digest, but is
    # still a different frozen campaign and cannot reuse the prepared capability.
    moved = resolved_campaign(
        build_digit="3", recipe_sha=target.execution.recipe.sha256)
    assert moved.manifest_digest == first.manifest_digest
    _, moved_engine = scheduler()
    with pytest.raises(planner.PlanningRefused, match="differ from resolved campaign"):
        planner.plan_iteration(
            resolved_campaign=moved, profiles={},
            evidence_index=evidence.EvidenceIndex((), current_epoch="epoch-1"),
            runtime_anchors=cached, runtime_dimensions={},
            source_actor=lambda *_: pytest.fail("must not author"),
            build_actor=lambda *_: pytest.fail("must not build"),
            scheduler_engine=moved_engine, experiment_plans={}, now=1,
            native_artifact_sink_ref="native:capture")


@pytest.mark.parametrize("field", ["target", "model", "workload", "mechanism"])
def test_unrelated_claim_scope_or_mechanism_never_reaches_runtime_adapter(field):
    canonical = canonical_recipe()
    enrolled = campaign_for_recipe(canonical)
    target = enrolled.targets[0]
    digest = planner._target_digest(target)
    pair = planner.enumerate_runtime_dimensions(canonical, [dimension()])[0]
    raw = opportunity(
        target=target, pair=pair,
        allocation=scheduling.ResourceVector.from_dict(profile(target)["resource_cost"]).digest)
    if field == "mechanism":
        raw["claim_key"]["mechanism_identity"]["id"] = "unrelated"
    else:
        raw["claim_key"]["target_scope"][field] = "unrelated"
    prof = profile(target, opportunities=[raw])
    _, engine = scheduler()
    result = planner.plan_iteration(
        resolved_campaign=enrolled, profiles={digest: prof},
        evidence_index=evidence.EvidenceIndex((), current_epoch="epoch-1"),
        runtime_anchors=prepared(enrolled, {digest: runtime_anchor(target, canonical)}),
        runtime_dimensions={digest: [dimension()]},
        source_actor=lambda *_: pytest.fail("must not author"),
        build_actor=lambda *_: pytest.fail("must not build"), scheduler_engine=engine,
        experiment_plans={}, now=1, native_artifact_sink_ref="native:capture")
    assert not result.proposals
    assert result.dispositions[0]["status"] == "claim_scope_mismatch"


def test_one_sealed_production_export_plans_distinct_cpu_and_gpu_frontiers(
        tmp_path, monkeypatch):
    from .production_enrollment import (manifest_from_export,
                                        registry_snapshot_from_export,
                                        resolve_exported_recipes)
    from .test_production_enrollment import (_campaign_config, _export, _policy,
                                             _seal_recipe_artifacts)

    cpu = _export(tmp_path / "cpu", compatible=True, backend="cpu")
    gpu = _export(tmp_path / "gpu", compatible=True, backend="gpu")
    gpu_target = gpu["targets"][0]
    gpu_target.update(target_id="gpu@8070", primary_role="gpu",
                      aliases=[], obligations=["gpu"])
    cpu["targets"].append(gpu_target)
    cpu["disposition"]["ready"] = 2
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _seal_recipe_artifacts(cpu, bundle)
    config = _campaign_config()
    config["resources"]["gpu_ids"] = ["ROCm0"]
    manifest = manifest_from_export(cpu, campaign_config=config)
    enrolled = campaign.resolve_manifest(
        manifest, registry_snapshot=registry_snapshot_from_export(cpu))
    resolved_rows = resolve_exported_recipes(cpu, environment_policy=_policy())
    recipes = {row["target_id"]: CanonicalResolvedRecipe.from_dict(row["resolved_recipe"])
               for row in resolved_rows["targets"]}
    profiles = {}
    anchors = {}
    dimensions = {}
    plans = {}
    for target in enrolled.targets:
        target_id = target.target_ids[0]
        recipe = recipes[target_id]
        digest = planner._target_digest(target)
        dim_id = f"threads-{target_id}"
        dim = dimension("threads", recipe.template.threads,
                        recipe.template.threads + 1, dim_id)
        pair = planner.enumerate_runtime_dimensions(recipe, [dim])[0]
        resource = scheduling.ResourceVector(
            1.0, ("ROCm0",) if target.execution.backend == "gpu" else (), 0)
        opp = opportunity(
            backend=target.execution.backend, pair=pair, target=target,
            allocation=resource.digest)
        opp["opportunity_id"] = f"opp-{target_id}"
        opp["runtime_dimension_ids"] = [dim_id]
        profiles[digest] = profile(
            target, backend=target.execution.backend, pair=pair, opportunities=[opp])
        anchors[digest] = {
            "schema": planner.ANCHOR_SCHEMA, "target_revision_digest": digest,
            "target_id": target_id, "production_export": cpu,
            "environment_policy": _policy(),
        }
        dimensions[digest] = [dim]
        plans[f"opp-{target_id}:{dim_id}"] = experiment(
            pair, digest, campaign_id=enrolled.campaign_id)
    vector = scheduling.ResourceVector(1.0, ("ROCm0",), 0)
    scheduler_config = scheduling.SchedulerConfig(
        config_id="both", max_stage_seconds=60, noncoverage_slots=0,
        reservation_slots={}, reservation_shares={}, campaign_attempt_cap=10,
        campaign_charged_seconds_cap=600, seed_attempt_cap=3,
        seed_charged_seconds_cap=180, capacity=vector, weights_source="fixed",
        apportionment_rule="coverage-first", adaptive_rule_id=None)
    engine = scheduling.SchedulerEngine(
        scheduler_config, scheduling.initial_state(scheduler_config, "both"))
    from . import production_enrollment as enrollment_module
    actual_resolver = enrollment_module.resolve_exported_recipes
    preparation_calls = 0

    def counted_resolver(*args, **kwargs):
        nonlocal preparation_calls
        preparation_calls += 1
        return actual_resolver(*args, **kwargs)

    monkeypatch.setattr(enrollment_module, "resolve_exported_recipes", counted_resolver)
    prepared_anchors = prepared(enrolled, anchors)
    assert preparation_calls == 1
    monkeypatch.setattr(enrollment_module, "load_export",
                        lambda *_args, **_kwargs: pytest.fail("hot path rescanned export"))
    monkeypatch.setattr(enrollment_module, "resolve_exported_recipes",
                        lambda *_args, **_kwargs: pytest.fail("hot path rederived recipes"))
    result = planner.plan_iteration(
        resolved_campaign=enrolled, profiles=profiles,
        evidence_index=evidence.EvidenceIndex((), current_epoch="epoch-1"),
        runtime_anchors=prepared_anchors, runtime_dimensions=dimensions,
        source_actor=lambda *_: pytest.fail("no source actor"),
        build_actor=lambda *_: pytest.fail("no build actor"), scheduler_engine=engine,
        experiment_plans=plans, now=1, native_artifact_sink_ref="native:capture")
    assert {row.backend for row in result.proposals} == {"cpu", "gpu"}, result.dispositions
    assert len({row.target_revision_digest for row in result.proposals}) == 2
    assert len({row.proposal_id for row in result.proposals}) == 2
    assert {row.frontier_id for row in result.stage_proposals} == {
        row.target_revision_digest for row in result.proposals}
    again = planner.plan_iteration(
        resolved_campaign=enrolled, profiles=profiles,
        evidence_index=evidence.EvidenceIndex((), current_epoch="epoch-1"),
        runtime_anchors=prepared_anchors, runtime_dimensions=dimensions,
        source_actor=lambda *_: pytest.fail("no source actor"),
        build_actor=lambda *_: pytest.fail("no build actor"), scheduler_engine=engine,
        experiment_plans=plans, now=2, native_artifact_sink_ref="native:capture")
    assert len(again.proposals) == 2
