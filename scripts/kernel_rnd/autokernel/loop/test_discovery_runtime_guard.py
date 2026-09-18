"""Installed discovery cannot borrow ordinary comparison launch authority.

Only configuration/observations/providers in these tests are fixtures. Scheduler,
catalog, Journal, materialization, startup and runtime paths are the real owners;
no worker or model is launched, and no A2 permit/event language is introduced.
"""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import discovery_screen, experiment_plan as ep, standalone_inputs as inputs
from . import unified_driver as driver, unified_planner as planner
from .test_driver_execution import _command
from .test_experiment_plan import plan_dict
from .test_feed_runtime import binding as feed_binding
from .test_standalone_inputs import FullHeldProvider
from .test_standalone_native_inputs import native_document
from .test_unified_driver_plan_versions import _stack


def _discovery(plan, marker="all"):
    row = plan.to_dict()
    if marker != "protocol_only":
        declared = plan_dict(phase="discovery", record_class="discovery_screen",
                             intended_use="nominate", n=3, paired=False)
        row.update({key: declared[key] for key in (
            "phase", "record_class", "intended_use", "stopping", "expected_units")})
        units = [unit for arm in ("anchor", "candidate")
                 for unit in row["expected_units"] if unit["arm"] == arm]
        for index, unit in enumerate(units):
            unit["order_index"] = index
        row["expected_units"] = units
    row["protocol_ref"] = ("fixture-other-discovery" if marker == "phase_only"
                           else discovery_screen.A2_PROTOCOL)
    return ep.ExperimentPlan.from_dict(row)


def _clone(seed, *, plans, profiles=None, executable_work_kinds=None):
    return driver.UnifiedCampaignDriver(
        resolved_campaign=seed.resolved, controller=seed.controller,
        scheduler_engine=seed.scheduler, profiles=profiles or seed.profiles,
        evidence_index=seed.evidence, runtime_anchors=seed.runtime_anchors,
        runtime_dimensions=seed.runtime_dimensions, experiment_plans=plans,
        profile_requests=seed.profile_requests, actor_identities=seed.actor_identities,
        native_artifact_sink_ref=seed.sink_ref, execution_inputs=seed.execution_inputs,
        executable_work_kinds=executable_work_kinds)


def _reason(plan):
    return f"target:{plan.target_revision}:discovery_unit:executor_unavailable"


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("marker", ["all", "phase_only", "protocol_only"])
def test_actual_tick_refuses_discovery_before_catalog_issue(tmp_path, version, marker):
    with _stack(tmp_path, version=version) as (seed, engine, plan, *_):
        declared = _discovery(plan, marker)
        key = next(iter(seed.experiment_plans))
        guarded = _clone(seed, plans={key: declared})
        before = engine.operational_projection().projection_digest
        outcome = guarded.tick(now=1)
        assert outcome.status == "waiting"
        assert outcome.reasons == (_reason(declared),)
        assert outcome.selection is None and outcome.transition_id is None
        assert engine.operational_projection().projection_digest == before
        assert guarded.controller.unified_driver_pending_intent() is None
        assert not any(entry.kind.startswith("UNIFIED_DRIVER_")
                       for entry in guarded.controller._journal.read_all())


@pytest.mark.parametrize("available", [True, False])
def test_unrelated_comparison_remains_schedulable(tmp_path, available):
    with _stack(tmp_path, version=2) as (seed, _engine, plan, *_):
        declared = _discovery(plan)
        key = next(iter(seed.experiment_plans))
        profiles = {target: value.to_dict() for target, value in seed.profiles.items()}
        profile = profiles[plan.target_revision]
        ordinary = copy.deepcopy(profile["opportunities"][0])
        ordinary["opportunity_id"] = "ordinary"
        profile["opportunities"].append(ordinary)
        guarded = _clone(seed, profiles=profiles, plans={
            key: declared, "ordinary:threads-4-8": plan},
            executable_work_kinds=({"runtime_comparison"} if available
                                   else {"profile_preparation"}))
        outcome = guarded.tick(now=1)
        if not available:
            assert outcome.status == "waiting"
            assert outcome.reasons == (
                _reason(declared),
                f"target:{plan.target_revision}:runtime_comparison:executor_unavailable")
            assert guarded.controller.unified_driver_pending_intent() is None
            return
        assert outcome.status == "intent_recorded"
        assert _reason(declared) in outcome.reasons
        assert outcome.selection["proposal"]["proposal_id"] == "ordinary:threads-4-8"
        record = guarded.controller.unified_driver_pending_intent()
        assert len(record["catalog"]["stage_proposals"]) == 1
        assert guarded.materialize_runtime(outcome).plan.digest == plan.digest


@pytest.mark.parametrize("marker", ["all", "phase_only", "protocol_only"])
def test_historical_misclassified_intent_refuses_before_binding(tmp_path, monkeypatch, marker):
    with _stack(tmp_path, version=2) as (seed, _engine, plan, *_):
        declared = _discovery(plan, marker)
        guarded = _clone(seed, plans={next(iter(seed.experiment_plans)): declared})
        # The existing closed catalog language permits these old records. Produce
        # one through the real planner/controller transaction, not private seeding
        # or a disabled guard; this models a previously issued misclassification.
        planning = planner.plan_iteration(
            resolved_campaign=guarded.resolved, profiles=guarded.profiles,
            evidence_index=guarded.evidence, runtime_anchors=guarded.runtime_anchors,
            runtime_dimensions=guarded.runtime_dimensions, source_actor=None,
            build_actor=None, scheduler_engine=guarded.scheduler,
            experiment_plans=guarded.experiment_plans, now=1,
            native_artifact_sink_ref=guarded.sink_ref, defer_actor_preparation=True,
            issue_selection=False, actor_identities=driver._thaw(guarded.actor_identities))
        assert len(planning.stage_proposals) == len(planning.proposals) == 1
        stage, proposal = planning.stage_proposals[0], planning.proposals[0]
        guarded.controller.refresh_unified_driver_readiness()
        ready = guarded.controller.unified_driver_readiness()
        catalog = driver.PlanningCatalog(
            guarded.campaign_digest, ready, ready["scheduler_projection_digest"], 1,
            (stage.to_dict(),), {stage.digest: {
                "kind": "runtime_comparison", "stage_plan_binding": "experiment_plan",
                "stage_plan_digest": declared.digest,
                "payload": {"proposal": proposal.to_dict(),
                            "experiment_plan": declared.to_dict()}}})
        guarded.controller.unified_driver_transaction(catalog.to_dict())
        original = guarded.controller.unified_driver_pending_intent()
        outcome = guarded.restore_issued_intent(original)

        def no_binding(**_kwargs):
            pytest.fail("discovery reached ordinary controller materialization binding")

        monkeypatch.setattr(guarded.controller, "unified_driver_materialization_binding", no_binding)
        with pytest.raises(driver.DriverRefused, match="discovery.*per-unit"):
            guarded.materialize_runtime(outcome)
        assert guarded.controller.unified_driver_pending_intent() == original


def test_installed_native_startup_tick_keeps_discovery_unavailable(tmp_path):
    document = native_document(tmp_path)
    key, raw = next(iter(document["driver_config"]["experiment_plans"].items()))
    declared = _discovery(ep.ExperimentPlan.from_dict(raw))
    document["driver_config"]["experiment_plans"][key] = declared.to_dict()
    document["manifest_digest"] = inputs._digest({
        key: value for key, value in document.items() if key != "manifest_digest"})
    materialized = inputs.materialize(inputs.StartupManifest.from_dict(document))
    feed = materialized.manifest.evidence_feed
    for path in (Path(feed.corpus_root), Path(feed.store_root), Path(feed.ledger_path).parent,
                 tmp_path / "containers"):
        path.mkdir(parents=True, exist_ok=True)

    class NoLaunchProvider(FullHeldProvider):
        def acquire(self, *_args, **_kwargs):
            pytest.fail("uninstalled discovery attempted provider acquisition")

    registry = inputs.ProviderRegistry({
        "fixture-lifecycle": inputs.ProviderBinding(
            lifecycle_provider=NoLaunchProvider(tmp_path / "containers")),
        "fixture-readiness": inputs.ProviderBinding(readiness_check=lambda: (True, None)),
    }, evidence_feeds={feed.binding_id: feed_binding()})
    factory = inputs.runtime_factory(materialized, registry)
    controller, runtime = factory(materialized.resolved, SimpleNamespace(
        store=document["driver_config"]["store_path"], config_generation=1,
        snapshot_version=3))
    try:
        _command(controller, "resume", "resume-discovery-guard")
        assert runtime.recover().status == "recovered"
        outcome = None
        for _ in range(32):
            outcome = runtime.tick()
            if (outcome.driver_outcome is not None
                    and _reason(declared) in outcome.driver_outcome["reasons"]):
                break
        assert outcome.status == "waiting"
        assert _reason(declared) in outcome.driver_outcome["reasons"]
        projection = controller.publish_snapshot()
        assert projection["unified"]["runtime"]["reason"] == outcome.reason
        assert projection["unified"]["runtime"]["publication_error"] is None
        assert projection["observed_state"] == "running"  # ACK is admission, not progress.
        assert projection["unified"]["runtime"]["installed_work_kinds"] == ["runtime_comparison"]
        assert controller.unified_driver_pending_intent() is None
        kinds = [entry.kind for entry in controller._journal.read_all()]
        assert "RETENTION_CATALOG_INSTALLED" in kinds
        assert not any(kind.startswith(("WORKER_", "UNIFIED_DRIVER_", "A2_")) for kind in kinds)
    finally:
        runtime.close()
        controller.close()
