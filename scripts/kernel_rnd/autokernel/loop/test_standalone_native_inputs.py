"""Closed standalone-v3 native evidence configuration and composition tests."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import time

import pytest

from . import campaign, campaign_control, experiment_plan, lifecycle_observation as lo
from . import measurement_capture as mc, native_model_preparation as nmp
from . import native_parent_receipt_replay as replay, native_parent_service
from . import native_scientific_witness as nsw, observation_binding as ob
from . import planned_serving, standalone_inputs as inputs, standalone_runtime
from . import production_enrollment, startup_factory, unified_driver, unified_planner
from . import search_window, serving, unified_worker, worker_lifecycle as wl
from ..execution import microbench
from .test_feed_runtime import binding as feed_binding, config as feed_config
from .test_driver_execution import _command
from .test_standalone_inputs import FullHeldProvider, _document
from .test_startup_factory import pin, request_for
from .test_unified_driver import _prompt_manifest
from .test_unified_planner import dimension, experiment, profile


def _budgets():
    return {"max_samples": 64, "max_pending_markers": 64, "max_processes": 8,
            "max_read_bytes": 1 << 20, "max_proc_entries": 128,
            "max_retained_bytes": 1 << 20, "max_map_entries": 128,
            "max_fd_entries": 128, "max_cpu_ids": 256, "max_numa_rows": 128,
            "max_dso_entries": 32, "phase_ack_timeout_s": 0.5,
            "join_timeout_s": 0.5, "max_probe_duration_s": 0.5}


def window_configuration(tmp_path: Path):
    root = tmp_path / "window"
    for name in ("claims", "proc", "sys", "storage"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return search_window.InstalledSearchWindowConfiguration(
        str(root / "claims"), str(root / "proc"), str(root / "sys"),
        str(root / "storage"), ("fixture",),
        microbench.HostStatePolicy(nominal_khz=3_000_000), 1,
        search_window.source_digest())


def native_document(tmp_path: Path, *, window=None):
    old = _document(tmp_path)
    cfg = replace(feed_config(tmp_path),
                  source_root=old["driver_config"]["store_path"] + "/journal")
    old.pop("evidence_index")
    old.pop("evidence_verifier_id")
    old.update(schema=inputs.FEED_MANIFEST_SCHEMA, evidence_feed=cfg.to_dict())
    old["manifest_digest"] = inputs._digest({
        key: value for key, value in old.items() if key != "manifest_digest"})
    parsed = inputs.materialize(inputs.StartupManifest.from_dict(old))
    root = Path(old["driver_config"]["store_path"]) / "unified-native-artifacts"
    selection = {"schema": inputs.SCIENTIFIC_SELECTION_SCHEMA,
                 "correctness": {"adapter_id": nsw.ADAPTER_ID, "max_units": 8},
                 "purpose": None, "contention": None, "residency": None}
    states, dsos, preparations = {}, {}, {}
    for target, anchor in parsed.inputs.runtime_anchors.recipes.items():
        recipes = [anchor]
        recipes.extend(arm for pair in unified_planner.enumerate_runtime_dimensions(
            anchor, parsed.inputs.runtime_dimensions.get(target, ()))
                       for arm in (pair.anchor, pair.candidate))
        target_rows = {}
        for recipe in recipes:
            states[recipe.execution_digest] = {
                "logical_cpus": sorted(lo.parse_cpu_list(recipe.template.cpu_list)),
                "numa_nodes": [0], "thp_mode": "madvise"}
            dsos[recipe.execution_digest] = ([] if recipe.backend == "cpu" else
                                             [item.to_dict() for item in recipe.dsos])
            identity = {"model_id": str(tmp_path / "model-inventory"),
                        "model_manifest": str(tmp_path / "model-manifest.json"),
                        "model_manifest_sha256": "8" * 64, "model_sha256": "9" * 64}
            body = {"schema": nmp.SPEC_SCHEMA, "target_revision_digest": target,
                    "recipe_execution_digest": recipe.execution_digest,
                    "entry_path": recipe.model.path, "entry_sha256": recipe.model.sha256,
                    "inventory_identity": identity}
            target_rows[recipe.execution_digest] = {
                **body, "preparation_digest": wl._digest(body)}
        preparations[target] = target_rows
    observation = {"requested_effective_states": states, "required_gpu_dsos": dsos,
                   "cadence_s": 0.01, "gap_limit_s": 0.05, "budgets": _budgets()}
    native = inputs.native_evidence_document(
        scientific_adapters=selection, model_preparations=preparations,
        artifact_root=str(root), observation_configuration=observation,
        search_window_configuration=window)
    _, _, reference, _, _ = inputs._native_evidence(native)
    document = copy.deepcopy(old)
    document["schema"] = inputs.NATIVE_MANIFEST_SCHEMA
    document["native_evidence"] = native
    document["driver_config"]["native_artifact_sink_ref"] = str(root)
    for key, raw in document["driver_config"]["experiment_plans"].items():
        plan = experiment_plan.ExperimentPlan.from_dict(raw)
        row = plan.to_dict() | {"schema": experiment_plan.PLAN_SCHEMA_V2,
                                "loaded_instrument": reference.to_dict()}
        pair = parsed.inputs.runtime_anchors.recipes[plan.target_revision]
        dimensions = parsed.inputs.runtime_dimensions[plan.target_revision]
        arms = unified_planner.enumerate_runtime_dimensions(pair, dimensions)[0]
        row["anchor_identity"] = planned_serving.arm_identity(
            arms.anchor.template, arms.anchor, loaded_instrument=reference.to_dict())
        row["candidate_identity"] = planned_serving.arm_identity(
            arms.candidate.template, arms.candidate, loaded_instrument=reference.to_dict())
        document["driver_config"]["experiment_plans"][key] = \
            experiment_plan.ExperimentPlan.from_dict(row).to_dict()
    for raw in document["driver_config"]["execution_inputs"].values():
        raw["instrument_id"] = reference.identity_sha256
    document["manifest_digest"] = inputs._digest({
        key: value for key, value in document.items() if key != "manifest_digest"})
    # The native sink is part of the resolved runtime snapshot. Rebuild this
    # synthetic profile's claim arms from that final configured snapshot.
    final = inputs.materialize(inputs.StartupManifest.from_dict(document))
    for target_digest, raw_profile in document["driver_config"]["profiles"].items():
        pairs = unified_planner.enumerate_runtime_dimensions(
            final.inputs.runtime_anchors.recipes[target_digest],
            final.inputs.runtime_dimensions[target_digest])
        by_id = {pair.dimension.dimension_id: pair for pair in pairs}
        for opportunity in raw_profile["opportunities"]:
            pair = by_id[opportunity["runtime_dimension_ids"][0]]
            opportunity["claim_key"]["control_identity"] = \
                unified_planner._thaw(unified_planner.serving_arm_identity(pair.anchor))
            opportunity["claim_key"]["intervention_identity"] = \
                unified_planner._thaw(unified_planner.serving_arm_identity(pair.candidate))
    document["manifest_digest"] = inputs._digest({
        key: value for key, value in document.items() if key != "manifest_digest"})
    return document


def test_v3_materialization_is_pure_and_binds_installed_source(tmp_path, monkeypatch):
    document = native_document(tmp_path)
    monkeypatch.setattr(mc.ArtifactStore, "__init__",
                        lambda *_args, **_kwargs: pytest.fail("dry materialization opened store"))
    materialized = inputs.materialize(inputs.StartupManifest.from_dict(document))
    assert materialized.inputs.native_evidence_configuration.schema.endswith(".v2")
    assert materialized.inputs.loaded_instrument_reference["identity_sha256"] == \
        next(iter(materialized.inputs.experiment_plans.values())).loaded_instrument[
            "identity_sha256"]
    changed = copy.deepcopy(document)
    changed["native_evidence"]["scientific_adapters"]["correctness"]["max_units"] -= 1
    changed["manifest_digest"] = inputs._digest({
        key: value for key, value in changed.items() if key != "manifest_digest"})
    with pytest.raises(inputs.StandaloneInputsRefused, match="installed adapters"):
        inputs.materialize(inputs.StartupManifest.from_dict(changed))


def test_v3_dry_run_labels_instrument_unpublished_without_store_io(
        tmp_path, monkeypatch, capsys):
    document = native_document(tmp_path)
    path = tmp_path / "startup-v3.json"
    path.write_text(json.dumps(document))
    monkeypatch.setattr(mc.ArtifactStore, "__init__",
                        lambda *_args, **_kwargs: pytest.fail("dry run opened artifact store"))
    assert unified_driver.main(["--config", str(path), "--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["schema"] == inputs.NATIVE_PREFLIGHT_SCHEMA
    assert report["native_instrument_runtime_status"] == "planned_unpublished"
    assert report["native_retention_catalog_status"] == "planned_unpublished"
    assert report["execution_authorized"] is False
    assert not Path(document["driver_config"]["store_path"]).exists()


def test_direct_compose_requires_runtime_instrument_then_reuses_one_adapter(tmp_path):
    document = native_document(tmp_path)
    materialized = inputs.materialize(inputs.StartupManifest.from_dict(document))
    store_path = Path(document["driver_config"]["store_path"])
    store_path.mkdir(mode=0o700)
    with campaign_control.CampaignController(
            materialized.resolved, store_path, snapshot_version=3,
            scheduler_engine=materialized.inputs.scheduler_engine,
            readiness_check=lambda: (True, None)) as controller:
        with pytest.raises(Exception, match="artifact does not exist"):
            standalone_runtime.StandaloneRuntime.compose(
                controller=controller, inputs=materialized.inputs)
        store = mc.ArtifactStore(materialized.inputs.native_artifact_root)
        try:
            identity = dict(materialized.inputs.loaded_instrument_identity)
            store.write(f"loaded-instrument:{identity['sha256']}", identity)
        finally:
            store.close()
        with pytest.raises(Exception, match="retention catalog is not durably installed"):
            standalone_runtime.StandaloneRuntime.compose(
                controller=controller, inputs=materialized.inputs)
        controller.install_native_retention_catalog(
            materialized.inputs.retention_catalog_seed,
            runtime_anchors=materialized.inputs.runtime_anchors,
            model_preparations=(
                materialized.inputs.native_evidence_configuration.model_preparations),
            runtime_recipes=materialized.inputs.retention_runtime_recipes,
            artifact_root=materialized.inputs.native_artifact_root)
        runtime = standalone_runtime.StandaloneRuntime.compose(
            controller=controller, inputs=materialized.inputs)
        try:
            assert runtime.executor._native_evidence_configuration is \
                materialized.inputs.native_evidence_configuration
            assert runtime.executor._native_evidence_configuration.scientific_adapters.correctness \
                is materialized.inputs.native_evidence_configuration.scientific_adapters.correctness
        finally:
            runtime.close()


def test_factory_v3_produces_matching_plan_and_execution_without_model_io(
        tmp_path, monkeypatch):
    request = request_for(tmp_path, include_candidate=False)
    envelope = json.loads(Path(request["resolved_export"]["path"]).read_text())
    resolved = campaign.ResolvedCampaign.from_dict(
        envelope["resolved_campaign"])
    target = resolved.targets[0]
    target_digest = unified_planner._target_digest(target)
    export = production_enrollment.load_export(json.loads(
        Path(request["production_export"]["path"]).read_text()))
    anchor = unified_planner.prepare_runtime_anchors(resolved, {target_digest: {
        "schema": unified_planner.ANCHOR_SCHEMA,
        "target_revision_digest": target_digest, "target_id": target.target_ids[0],
        "production_export": export,
        "environment_policy": request["environment_policy"]}}).recipes[target_digest]
    dim = dimension()
    pair = unified_planner.enumerate_runtime_dimensions(anchor, [dim])[0]
    selected = {"schema": inputs.SCIENTIFIC_SELECTION_SCHEMA,
                "correctness": {"adapter_id": nsw.ADAPTER_ID, "max_units": 8},
                "purpose": None, "contention": None, "residency": None}
    adapters = inputs._installed_scientific_adapters(selected)
    seed_store = mc.ArtifactStore(tmp_path / "seed-instrument")
    try:
        old_reference = ob.seal_loaded_instrument(
            store=seed_store, measurement_callable=serving._measure_once,
            fence_clock=time.monotonic, serving_timer=time.time,
            scientific_adapters=adapters)
    finally:
        seed_store.close()
    raw_plan = experiment(pair, target_digest, resolved.campaign_id)
    raw_plan.update(schema=experiment_plan.PLAN_SCHEMA_V2,
                    loaded_instrument=old_reference.to_dict(),
                    anchor_identity=planned_serving.arm_identity(
                        pair.anchor.template, pair.anchor,
                        loaded_instrument=old_reference.to_dict()),
                    candidate_identity=planned_serving.arm_identity(
                        pair.candidate.template, pair.candidate,
                        loaded_instrument=old_reference.to_dict()))
    plan = experiment_plan.ExperimentPlan.from_dict(raw_plan)
    profile_pin = pin(tmp_path / "profile.json", profile(target, pair=pair))
    plan_pin = pin(tmp_path / "plan.json", plan.to_dict())
    prompts_pin = pin(tmp_path / "prompts.json", _prompt_manifest(anchor).to_dict())
    preparations, states, dsos = {}, {}, {}
    for recipe in (pair.anchor, pair.candidate):
        identity = {"model_id": str(tmp_path / "model-inventory"),
                    "model_manifest": str(tmp_path / "model-manifest.json"),
                    "model_manifest_sha256": "8" * 64, "model_sha256": "9" * 64}
        body = {"schema": nmp.SPEC_SCHEMA, "target_revision_digest": target_digest,
                "recipe_execution_digest": recipe.execution_digest,
                "entry_path": recipe.model.path, "entry_sha256": recipe.model.sha256,
                "inventory_identity": identity}
        preparations[recipe.execution_digest] = {
            **body, "preparation_digest": wl._digest(body)}
        states[recipe.execution_digest] = {
            "logical_cpus": sorted(lo.parse_cpu_list(recipe.template.cpu_list)),
            "numa_nodes": [0], "thp_mode": "madvise"}
        dsos[recipe.execution_digest] = []
    request.update(schema=startup_factory.NATIVE_REQUEST_SCHEMA,
        native_artifact_sink_ref=str(Path(request["store_path"]) / "unified-native-artifacts"),
        target_defaults={"cpu": {"profile": profile_pin, "profile_request": None,
            "execution": {"prompt_manifest": prompts_pin, "max_stage_seconds": 30,
                          "teardown_seconds": 2},
            "runtime_dimensions": [dim]}},
        experiment_plans={"opp-runtime_recipe:threads-4-8": plan_pin},
        native_evidence={"scientific_adapters": selected,
            "model_preparations": {target_digest: preparations},
            "search_window_configuration": None,
            "observation_configuration": {"requested_effective_states": states,
                "required_gpu_dsos": dsos, "cadence_s": 0.01, "gap_limit_s": 0.05,
                "budgets": _budgets()}},
        evidence_feed=replace(feed_config(tmp_path),
            source_root=request["store_path"] + "/journal").to_dict())
    request.pop("evidence_index")
    request["providers"].pop("evidence_verifier")
    monkeypatch.setattr(nsw.NativeT0WitnessAdapter, "prepare_model_identity",
                        lambda *_args, **_kwargs: pytest.fail("factory performed model I/O"))
    out = tmp_path / "bundle"
    startup_factory.build_startup(request, output_dir=out)
    manifest = inputs.StartupManifest.from_dict(json.loads((out / "startup.json").read_text()))
    materialized = inputs.materialize(manifest)
    emitted = materialized.inputs.experiment_plans["opp-runtime_recipe:threads-4-8"]
    assert emitted.loaded_instrument["identity_sha256"] == \
        materialized.inputs.loaded_instrument_reference["identity_sha256"]
    assert materialized.inputs.execution_inputs[target_digest].instrument_id == \
        emitted.loaded_instrument["identity_sha256"]
    feed = materialized.manifest.evidence_feed
    for path in (Path(feed.corpus_root), Path(feed.store_root), Path(feed.ledger_path).parent,
                 tmp_path / "factory-containers"):
        path.mkdir(parents=True, exist_ok=True)
    registry = inputs.ProviderRegistry({
        "unavailable-explicit-lifecycle": inputs.ProviderBinding(
            lifecycle_provider=FullHeldProvider(tmp_path / "factory-containers")),
        "unavailable-explicit-readiness": inputs.ProviderBinding(
            readiness_check=lambda: (True, None)),
    }, evidence_feeds={feed.binding_id: feed_binding()})
    build = inputs.runtime_factory(materialized, registry)

    class Args:
        store = request["store_path"]
        config_generation = 1
        snapshot_version = 3

    controller, runtime = build(materialized.resolved, Args())
    try:
        assert runtime.executor._native_evidence_configuration is \
            materialized.inputs.native_evidence_configuration
        assert runtime.driver.experiment_plans[
            "opp-runtime_recipe:threads-4-8"].to_dict() == emitted.to_dict()
    finally:
        runtime.close()
        controller.close()


def test_runtime_factory_publishes_before_compose_and_retains_adapter_instance(tmp_path):
    document = native_document(tmp_path)
    materialized = inputs.materialize(inputs.StartupManifest.from_dict(document))
    feed = materialized.manifest.evidence_feed
    for path in (Path(feed.corpus_root), Path(feed.store_root), Path(feed.ledger_path).parent,
                 tmp_path / "containers"):
        path.mkdir(parents=True, exist_ok=True)
    registry = inputs.ProviderRegistry({
        "fixture-lifecycle": inputs.ProviderBinding(
            lifecycle_provider=FullHeldProvider(tmp_path / "containers")),
        "fixture-readiness": inputs.ProviderBinding(readiness_check=lambda: (True, None)),
    }, evidence_feeds={feed.binding_id: feed_binding()})
    factory = inputs.runtime_factory(materialized, registry)

    class Args:
        store = document["driver_config"]["store_path"]
        config_generation = 1
        snapshot_version = 3

    controller, runtime = factory(materialized.resolved, Args())
    try:
        configured = materialized.inputs.native_evidence_configuration
        assert runtime.executor._native_evidence_configuration is configured
        assert runtime.executor._native_evidence_configuration.scientific_adapters.correctness \
            is configured.scientific_adapters.correctness
        assert controller.native_retention_catalog_seed_digest() == \
            materialized.inputs.retention_catalog_seed.seed_digest
        assert [entry.kind for entry in controller._journal.read_all()].count(
            "RETENTION_CATALOG_INSTALLED") == 1
        store = mc.ArtifactStore(materialized.inputs.native_artifact_root)
        try:
            identity = dict(materialized.inputs.loaded_instrument_identity)
            assert store.verify(f"loaded-instrument:{identity['sha256']}",
                                unified_driver._thaw(identity)).verified
        finally:
            store.close()
    finally:
        runtime.close()
        controller.close()


def test_runtime_factory_installs_exact_nonnull_window_owner(tmp_path, monkeypatch):
    # This fixture constructs the parent service: declare its full admission
    # capacity before native_document seals the prospective instrument identity.
    budgets = _budgets()
    budgets["max_samples"] = lo.required_sample_capacity(
        max_duration_s=30 + 2, cadence_s=0.01,
        nonperiodic_samples=len(lo.PHASES) + 2)
    assert budgets["max_samples"] == 3209
    monkeypatch.setattr(__name__ + "._budgets", lambda: dict(budgets))
    window = window_configuration(tmp_path)
    document = native_document(tmp_path, window=window)
    materialized = inputs.materialize(inputs.StartupManifest.from_dict(document))
    feed = materialized.manifest.evidence_feed
    for path in (Path(feed.corpus_root), Path(feed.store_root), Path(feed.ledger_path).parent,
                 tmp_path / "window-containers"):
        path.mkdir(parents=True, exist_ok=True)
    registry = inputs.ProviderRegistry({
        "fixture-lifecycle": inputs.ProviderBinding(
            lifecycle_provider=FullHeldProvider(tmp_path / "window-containers")),
        "fixture-readiness": inputs.ProviderBinding(readiness_check=lambda: (True, None)),
    }, evidence_feeds={feed.binding_id: feed_binding()})
    factory = inputs.runtime_factory(materialized, registry)

    class Args:
        store = document["driver_config"]["store_path"]
        config_generation = 1
        snapshot_version = 3

    controller, runtime = factory(materialized.resolved, Args())
    service = None
    try:
        configured = materialized.inputs.native_evidence_configuration
        assert configured.search_window_configuration == window
        _command(controller, "resume", "window-startup-resume")
        outcome = runtime.driver.tick(now=1.0)
        prepared = runtime.driver.materialize_runtime(outcome)
        assert ob.validate_planned_sample_capacity(
            materialized.inputs.observation_configuration,
            max_stage_seconds=prepared.max_stage_seconds,
            teardown_seconds=prepared.teardown_seconds) == budgets["max_samples"]
        authority = unified_worker.ParentUnitEvidenceAuthority(
            max_records=len(prepared.plan.expected_units)
            * (3 + len(unified_worker.OBSERVATION_WINDOW_MARKERS)))
        issued = replay.IssuedNativeEvidenceRegistry(
            artifact_root=prepared.artifact_root,
            max_units=len(prepared.plan.expected_units))
        service = native_parent_service.NativeParentEvidenceService(
            authority, prepared, controller._worker_lifecycle,
            materialized.inputs.observation_configuration,
            registry=issued, factual_configuration=configured)
        assert service.factual_configuration is configured
        assert service.search_window_owner is not None
        assert service.search_window_owner.config is configured.search_window_configuration
        instrument = unified_planner._thaw(materialized.inputs.loaded_instrument_identity)
        assert instrument["used_constants"]["search_window_configuration"] == window.to_dict()
        assert instrument["used_constants"]["search_window_source"] == \
            ob._plain(search_window.source_identity())
    finally:
        if service is not None:
            service.stop_and_join()
        runtime.close()
        controller.close()
