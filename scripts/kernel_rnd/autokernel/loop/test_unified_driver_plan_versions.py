"""Public scheduler-to-materialization contracts for native plan versions.

V2 cases require the separately owned native-observation packet. No selected
proposal, Prepared record, or controller issuance state is rewritten by these tests.
"""
from contextlib import contextmanager
import time

import pytest

from . import campaign, campaign_control, experiment_plan as ep, measurement_capture as mc
from . import planned_serving as ps, scheduling, scoped_evidence, serving
from . import unified_driver as driver, unified_planner as planner, unified_worker as worker
from .test_experiment_plan import plan_dict
from .test_unified_driver import _prompt_manifest
from .test_unified_planner import (
    campaign_for_recipe, canonical_recipe, dimension, profile, runtime_anchor, scheduler,
)


V2 = pytest.mark.skipif(
    not hasattr(ep, "PLAN_SCHEMA_V2"),
    reason="requires the separately reviewed native-observation v2 dependency",
)


@contextmanager
def _stack(tmp_path, *, version, backend="cpu", instrument_id_override=None,
           wrong_model=False):
    recipe = canonical_recipe(backend=backend)
    campaign_row = campaign_for_recipe(recipe).to_dict()
    source_ref = "production-source:kernel:" + "a" * 40
    campaign_row["source_refs"] = {"kernel": source_ref}
    campaign_row["source_snapshot"]["kernel"]["ref"] = source_ref
    resolved = campaign.ResolvedCampaign.from_dict(campaign_row)
    target = resolved.targets[0]
    digest = planner._target_digest(target)
    anchors = planner.prepare_runtime_anchors(
        resolved, {digest: runtime_anchor(target, recipe)})
    pair = planner.enumerate_runtime_dimensions(anchors.recipes[digest], [dimension()])[0]
    scheduler_config, _ = scheduler(backend)
    scheduler_config = scheduling.SchedulerConfig.from_dict(
        scheduler_config.to_dict() | {"config_id": resolved.campaign_id})
    engine = scheduling.SchedulerEngine(
        scheduler_config, scheduling.initial_state(scheduler_config, resolved.campaign_id))
    store_path = tmp_path / "controller"
    store_path.mkdir(mode=0o700)
    loaded = None
    if version == 2:
        from . import observation_binding as ob
        store = mc.ArtifactStore(store_path / "unified-native-artifacts")
        try:
            loaded = ob.seal_loaded_instrument(
                store=store, measurement_callable=serving._measure_once,
                fence_clock=time.monotonic, serving_timer=time.time).to_dict()
        finally:
            store.close()
    identity_kwargs = {} if loaded is None else {"loaded_instrument": loaded}
    # Construct startup configuration in the requested version before planning.
    plan_row = {
        **plan_dict(instrument="serving", unit="process"),
        "schema": ep.PLAN_SCHEMA if version == 1 else ep.PLAN_SCHEMA_V2,
        "campaign_id": resolved.campaign_id, "target_revision": digest,
        "metric": "aggregate_tok_s", "changed_factors": ["threads"],
        "required_witnesses": ["native-capture-v1"],
        "anchor_identity": ps.arm_identity(pair.anchor.template, pair.anchor,
                                           **identity_kwargs),
        "candidate_identity": ps.arm_identity(pair.candidate.template, pair.candidate,
                                              **identity_kwargs),
        **({} if loaded is None else {"loaded_instrument": loaded}),
    }
    if wrong_model:
        plan_row["anchor_identity"]["model_digest"] = "f" * 64
    plan = ep.ExperimentPlan.from_dict(plan_row)
    instrument_id = ("planned-serving/v1" if loaded is None
                     else loaded["identity_sha256"])
    if instrument_id_override is not None:
        instrument_id = instrument_id_override
    with campaign_control.CampaignController(
            resolved, store_path, snapshot_version=3, scheduler_engine=engine,
            readiness_check=lambda: (True, None)) as controller:
        instance = driver.UnifiedCampaignDriver(
            resolved_campaign=resolved, controller=controller, scheduler_engine=engine,
            profiles={digest: profile(target, backend=backend, pair=pair)},
            evidence_index=scoped_evidence.EvidenceIndex((), current_epoch="epoch-1"),
            runtime_anchors=anchors, runtime_dimensions={digest: [dimension()]},
            experiment_plans={"opp-runtime_recipe:threads-4-8": plan},
            profile_requests={}, actor_identities={}, native_artifact_sink_ref="native:capture",
            execution_inputs={digest: driver.ExecutionInput(
                digest, _prompt_manifest(recipe), 30.0, 2.0, instrument_id)},
        )
        payload_digest = campaign_control.command_digest(
            operation="resume", payload={}, campaign_id=resolved.campaign_id,
            config_generation=1)
        controller.apply_command({
            "schema": campaign_control.COMMAND_SCHEMA,
            "campaign_id": resolved.campaign_id, "config_generation": 1,
            "request_id": "resume-version-test", "operation": "resume", "payload": {},
            "payload_digest": payload_digest, "expected_control_revision": 0,
        })
        yield instance, engine, plan, pair, loaded


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("version", [1, pytest.param(2, marks=V2)])
def test_public_scheduler_materializes_the_original_plan_version(tmp_path, backend, version):
    with _stack(tmp_path, version=version, backend=backend) as (instance, engine, plan,
                                                              pair, loaded):
        outcome = instance.tick(now=1.0)
        assert outcome.status == "intent_recorded"
        assert engine.operational_projection().body["issued_selection_digests"]
        prepared = instance.materialize_runtime(outcome)
        expected_schema = worker.PREPARED_SCHEMA if version == 1 else worker.PREPARED_SCHEMA_V2
        assert prepared.to_dict()["schema"] == expected_schema
        assert prepared.dispatch["selection"] == outcome.selection
        assert prepared.dispatch["proposal"]["experiment_plan_digest"] == plan.digest
        assert prepared.dispatch["experiment_intent"]["experiment_plan_digest"] == plan.digest
        assert prepared.plan.to_dict() == plan.to_dict()
        assert prepared.capture_context_base["instrument_id"] == (
            "planned-serving/v1" if loaded is None else loaded["identity_sha256"])
        for arm, recipe in (("anchor", pair.anchor), ("candidate", pair.candidate)):
            assert dict(prepared.capture_context_base["source_identities"][arm]) == {
                "source_revision": "a" * 40, "model_sha256": recipe.model.sha256,
                "build_sha256": recipe.executable.sha256,
                "recipe_hash": recipe.template.recipe_hash,
            }
        proposal = prepared.dispatch["proposal"]
        assert dict(proposal["control_identity"]) == dict(planner.serving_arm_identity(pair.anchor))
        assert dict(proposal["intervention_identity"]) == dict(
            planner.serving_arm_identity(pair.candidate))
        assert instance.materialize_runtime(outcome).to_dict() == prepared.to_dict()


@V2
def test_v2_materialization_refuses_conflicting_startup_instrument_pin(tmp_path):
    with _stack(tmp_path, version=2, instrument_id_override="planned-serving/v1") as stack:
        instance = stack[0]
        outcome = instance.tick(now=1.0)
        assert outcome.status == "intent_recorded"
        with pytest.raises(driver.DriverRefused, match="v2 loaded instrument pin"):
            instance.materialize_runtime(outcome)


@V2
def test_v2_plan_still_checks_exact_recipe_identity_before_scheduler_issue(tmp_path):
    with _stack(tmp_path, version=2, wrong_model=True) as stack:
        instance, engine = stack[:2]
        with pytest.raises(planner.PlanningRefused, match="proposal/arm identities"):
            instance.tick(now=1.0)
        assert engine.operational_projection().body["issued_selection_digests"] == ()
