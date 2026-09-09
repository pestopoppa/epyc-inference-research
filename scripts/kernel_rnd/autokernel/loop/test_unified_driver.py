from __future__ import annotations

import json
import copy
import hashlib
import threading

import pytest

from . import planned_serving, scheduling, scoped_evidence as evidence
from . import unified_driver as driver
from . import unified_planner as planner
from .test_unified_planner import (campaign_for_recipe, canonical_recipe, dimension,
                                   experiment, opportunity, prepared, profile,
                                   runtime_anchor, scheduler)


class FakeController:
    def __init__(self, campaign_id: str, config_digest: str, *, available: bool = True,
                 uncertain_once: bool = False, scheduler_engine=None):
        self.campaign_id = campaign_id
        self.config_digest = config_digest
        self.available = available
        self.uncertain_once = uncertain_once
        self.records = {}
        self.receipts = {}
        self.scheduler = scheduler_engine

    def unified_driver_readiness(self):
        return {"schema": driver.READINESS_SCHEMA, "campaign_id": self.campaign_id,
                "config_digest": self.config_digest, "config_generation": 1,
                "supervisor_incarnation": 1, "control_revision": 0,
                "admission_open": self.available, "provider_available": self.available,
                "reason": "ready" if self.available else "trusted provider unavailable",
                "scheduler_projection_digest": self.scheduler.operational_projection(
                ).projection_digest}

    def unified_driver_transaction(self, value):
        catalog_id = value["catalog_id"]
        prior = self.records.get(catalog_id)
        if prior is not None and prior != value:
            raise AssertionError("conflicting retry")
        if prior is not None:
            return self.receipts[catalog_id] | {"status": "duplicate"}
        assert value["scheduler_projection_digest"] == self.scheduler.operational_projection(
        ).projection_digest
        stages = [scheduling.StageProposal.from_dict(item)
                  for item in value["stage_proposals"]]
        preview = self.scheduler.preview_selection(stages, now=value["observed_at"])
        assert preview.selection.status == "selected"
        self.records[catalog_id] = value
        self.scheduler.apply_preview(preview)
        transition_id = driver._digest({"catalog_id": catalog_id,
                                        "selection": preview.selection.to_dict()})
        receipt = {"schema": driver.TRANSACTION_RECEIPT_SCHEMA,
                   "catalog_id": catalog_id, "transition_id": transition_id,
                   "status": "accepted", "selection": preview.selection.to_dict()}
        self.receipts[catalog_id] = receipt
        if self.uncertain_once:
            self.uncertain_once = False
            raise OSError("reply lost after append")
        return receipt


def _prompt_manifest(recipe):
    prompts = []
    for prompt_id in ("p1", "p2"):
        body = json.dumps({"cache_prompt": False, "n_predict": recipe.template.n_predict,
                           "prompt": prompt_id, "temperature": recipe.template.temperature,
                           "top_k": recipe.template.top_k, "top_p": recipe.template.top_p},
                          sort_keys=True, separators=(",", ":")).encode()
        prompts.append({"prompt_id": prompt_id, "prompt": prompt_id,
                        "n_predict": recipe.template.n_predict,
                        "temperature": recipe.template.temperature,
                        "top_p": recipe.template.top_p, "top_k": recipe.template.top_k,
                        "cache_prompt": False,
                        "request_digest": hashlib.sha256(body).hexdigest()})
    body = {"schema": planned_serving.PROMPT_SCHEMA, "version": "fixture-v1",
            "prompts": prompts}
    return planned_serving.FrozenPromptManifest.from_dict(
        {**body, "digest": driver._digest(body)})


def runtime_driver(*, controller=None, git_source=False, with_execution_input=True,
                   recipe=None):
    supplied_recipe = recipe is not None
    recipe = canonical_recipe() if recipe is None else recipe
    enrolled = (campaign_for_recipe(
        recipe, model_path=recipe.model.path, model_sha=recipe.model.sha256)
        if supplied_recipe else campaign_for_recipe(recipe))
    if git_source:
        revision = "a" * 40
        row = enrolled.to_dict()
        ref = f"production-source:kernel:{revision}"
        row["source_refs"] = {"kernel": ref}
        row["source_snapshot"]["kernel"]["ref"] = ref
        enrolled = driver.campaign.ResolvedCampaign.from_dict(row)
    target = enrolled.targets[0]
    digest = planner._target_digest(target)
    anchors = prepared(enrolled, {digest: runtime_anchor(target, recipe)})
    pair = planner.enumerate_runtime_dimensions(anchors.recipes[digest], [dimension()])[0]
    _, engine = scheduler()
    if controller is None:
        controller = FakeController(
            enrolled.campaign_id, driver.campaign_control.resolved_config_digest(enrolled),
            scheduler_engine=engine)
    elif controller.scheduler is None:
        controller.scheduler = engine
    instance = driver.UnifiedCampaignDriver(
        resolved_campaign=enrolled,
        controller=controller,
        scheduler_engine=engine,
        profiles={digest: profile(target, pair=pair)},
        evidence_index=evidence.EvidenceIndex((), current_epoch="epoch-1"),
        runtime_anchors=anchors,
        runtime_dimensions={digest: [dimension()]},
        experiment_plans={"opp-runtime_recipe:threads-4-8": experiment(pair, digest)},
        profile_requests={},
        actor_identities={"source": {"build": "1" * 64},
                          "build_recipe": {"build": "2" * 64}},
        native_artifact_sink_ref="native:capture",
        execution_inputs=({digest: driver.ExecutionInput(
            digest, _prompt_manifest(recipe), 30.0, 2.0, "planned-serving/v1")}
            if with_execution_input else {}))
    return instance, engine, enrolled, target, digest


def test_missing_provider_waits_before_scheduler_issues_selection():
    recipe = canonical_recipe()
    enrolled = campaign_for_recipe(recipe)
    instance, engine, _, _, _ = runtime_driver(
        controller=FakeController(
            enrolled.campaign_id, driver.campaign_control.resolved_config_digest(enrolled),
            available=False))
    before = engine.export_state()
    outcome = instance.tick(now=1)
    assert outcome.status == "waiting"
    assert outcome.reasons == ("trusted provider unavailable",)
    assert engine.export_state() == before


def test_runtime_tick_records_exact_intent_without_actor_or_execution():
    instance, engine, _, _, _ = runtime_driver()
    outcome = instance.tick(now=1)
    assert outcome.status == "intent_recorded"
    assert outcome.execution_authorized is False
    assert len(engine.export_state().issued_selection_digests) == 1
    record = next(iter(instance.controller.records.values()))
    work = next(iter(record["work_by_stage_digest"].values()))
    assert work["kind"] == "runtime_comparison"
    assert work["payload"]["proposal"]["experiment_plan_digest"] is not None


def test_source_actor_is_selected_and_budgeted_without_calling_actor():
    instance, _, enrolled, target, digest = runtime_driver()
    raw_profile = profile(target)
    allocation = scheduling.ResourceVector.from_dict(raw_profile["resource_cost"]).digest
    raw_profile["opportunities"] = [opportunity(
        kind="source", target=target, allocation=allocation)]
    instance.profiles = {digest: raw_profile}
    instance.runtime_dimensions = {}
    instance.experiment_plans = {}
    outcome = instance.tick(now=1)
    assert outcome.status == "intent_recorded"
    record = next(iter(instance.controller.records.values()))
    work = next(iter(record["work_by_stage_digest"].values()))
    assert work["kind"] == "actor_preparation"
    assert work["payload"]["actor_kind"] == "source"
    assert work["payload"]["proposal"]["experiment_plan_digest"] is None
    assert record["campaign_digest"] == planner._digest(enrolled.to_dict())


def test_missing_profile_is_real_scheduler_work_not_only_a_status():
    instance, engine, _, target, digest = runtime_driver()
    instance.profiles = {}
    cost = scheduling.ResourceVector(1.0, (), 0)
    stage = scheduling.StageProposal(
        proposal_id="profile:prod", submitted_at=1, backend="cpu",
        target_revision=digest, alias_identity=target.workload_signature,
        frontier_id=digest, production_frontier=True, seed_id=None,
        stage_class="prerequisite", estimated_duration_seconds=10,
        estimated_claims=cost, eligible=True, eligibility_ref="2" * 64,
        reservation_kind=None, full_region=True,
        compatibility_authority_refs=(), safe_chunking_declared=False)
    request = driver.ProfilePreparationRequest.from_dict({
        "schema": driver.PROFILE_REQUEST_SCHEMA,
        "target_revision_digest": digest, "stage_proposal": stage.to_dict(),
        "profile_contract": {"schema": driver.PROFILE_CONTRACT_SCHEMA,
                             "adapter_id": "fake-owned-profile-v1",
                             "adapter_digest": "9" * 64}})
    instance.profile_requests = {digest: request}
    outcome = instance.tick(now=1)
    assert outcome.status == "intent_recorded"
    assert len(engine.export_state().issued_selection_digests) == 1
    record = next(iter(instance.controller.records.values()))
    assert next(iter(record["work_by_stage_digest"].values()))[
        "kind"] == "profile_preparation"


def test_uncertain_append_requires_exact_retry_and_never_reselects():
    recipe = canonical_recipe()
    enrolled = campaign_for_recipe(recipe)
    controller = FakeController(
        enrolled.campaign_id, driver.campaign_control.resolved_config_digest(enrolled),
        uncertain_once=True)
    instance, engine, _, _, _ = runtime_driver(controller=controller)
    with pytest.raises(driver.DriverTransactionUncertain, match="exact retry"):
        instance.tick(now=1)
    issued = controller.scheduler.export_state().issued_selection_digests
    with pytest.raises(driver.DriverTransactionUncertain):
        instance.tick(now=2)
    outcome = instance.retry_pending()
    assert outcome.reasons == ("duplicate",)
    assert controller.scheduler.export_state().issued_selection_digests == issued


def test_stop_before_enumeration_does_not_mutate_scheduler_or_controller():
    instance, engine, _, _, _ = runtime_driver()
    before = engine.export_state()
    outcome = instance.tick(now=1, stop_requested=lambda: True)
    assert outcome.status == "stopped"
    assert engine.export_state() == before
    assert not instance.controller.records


def test_standalone_once_reuses_real_controller_and_waits_without_bridge(
        tmp_path, capsys):
    recipe = canonical_recipe()
    enrolled = campaign_for_recipe(recipe)
    resolved_path = tmp_path / "resolved.json"
    resolved_path.write_text(json.dumps(enrolled.to_dict()))
    scheduler_config, engine = scheduler()
    scheduler_config = scheduling.SchedulerConfig.from_dict(
        scheduler_config.to_dict() | {"config_id": enrolled.campaign_id})
    engine = scheduling.SchedulerEngine(
        scheduler_config, scheduling.initial_state(scheduler_config, enrolled.campaign_id))
    config = {
        "schema": driver.CONFIG_SCHEMA,
        "resolved_campaign_path": str(resolved_path.resolve()),
        "store_path": str((tmp_path / "controller").resolve()),
        "scheduler_config": scheduler_config.to_dict(),
        "scheduler_state": engine.export_state().to_dict(),
        "runtime_anchors": {}, "runtime_dimensions": {}, "profiles": {},
        "experiment_plans": {}, "profile_requests": {}, "execution_inputs": {},
        "native_artifact_sink_ref": "native:capture", "config_generation": 1,
    }
    config_path = tmp_path / "driver.json"
    config_path.write_text(json.dumps(config))
    assert driver.main(["--config", str(config_path), "--once"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "waiting"
    assert output["reasons"] == ["execution authority absent"]
    assert output["execution_authorized"] is False
    assert (tmp_path / "controller" / "journal").is_dir()


def test_config_loader_refuses_symlink_and_oversized_input(tmp_path):
    target = tmp_path / "target.json"
    target.write_text("{}")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(driver.DriverRefused, match="cannot load"):
        driver.load_config(link)
    oversized = tmp_path / "large.json"
    oversized.write_bytes(b" " * (4 * 1024 * 1024 + 1))
    with pytest.raises(driver.DriverRefused, match="identity/size"):
        driver.load_config(oversized)


def test_driver_copies_nested_profile_inputs_before_tick():
    instance, _, _, target, digest = runtime_driver()
    original = profile(target)
    instance.profiles = driver._freeze(driver._thaw({digest: original}))
    original["freshness"] = "stale"
    assert instance.profiles[digest]["freshness"] == "fresh"


def test_catalog_never_calls_preparation_contract_an_experiment_plan():
    instance, _, _, target, digest = runtime_driver()
    instance.profiles = {}
    cost = scheduling.ResourceVector(1.0, (), 0)
    stage = scheduling.StageProposal(
        proposal_id="profile:prod", submitted_at=1, backend="cpu",
        target_revision=digest, alias_identity=target.workload_signature,
        frontier_id=digest, production_frontier=True, seed_id=None,
        stage_class="prerequisite", estimated_duration_seconds=10,
        estimated_claims=cost, eligible=True, eligibility_ref="2" * 64,
        reservation_kind=None, full_region=True,
        compatibility_authority_refs=(), safe_chunking_declared=False)
    instance.profile_requests = {digest: driver.ProfilePreparationRequest.from_dict({
        "schema": driver.PROFILE_REQUEST_SCHEMA,
        "target_revision_digest": digest, "stage_proposal": stage.to_dict(),
        "profile_contract": {"schema": driver.PROFILE_CONTRACT_SCHEMA,
                             "adapter_id": "fake-owned-profile-v1",
                             "adapter_digest": "9" * 64}})}
    instance.tick(now=1)
    raw = dict(next(iter(instance.controller.records.values())))
    raw.pop("catalog_id")
    work = dict(raw["work_by_stage_digest"])
    key = next(iter(work))
    work[key] = dict(work[key]) | {"stage_plan_binding": "experiment_plan"}
    raw["work_by_stage_digest"] = work
    with pytest.raises(driver.DriverRefused, match="only runtime work"):
        driver.PlanningCatalog(**raw)


def test_real_controller_journals_before_apply_and_replays_exact_selection(tmp_path):
    instance, _, enrolled, _, _ = runtime_driver()
    base_config, _ = scheduler()
    config = scheduling.SchedulerConfig.from_dict(
        base_config.to_dict() | {"config_id": enrolled.campaign_id})
    initial = scheduling.initial_state(config, enrolled.campaign_id)
    engine = scheduling.SchedulerEngine(config, initial)
    store = tmp_path / "owned-controller"
    controller = driver.campaign_control.CampaignController(
        enrolled, store, snapshot_version=3, scheduler_engine=engine,
        readiness_check=lambda: (True, None))
    with controller:
        payload_digest = driver.campaign_control.command_digest(
            operation="resume", payload={}, campaign_id=enrolled.campaign_id,
            config_generation=1)
        controller.apply_command({
            "schema": driver.campaign_control.COMMAND_SCHEMA,
            "campaign_id": enrolled.campaign_id, "config_generation": 1,
            "request_id": "resume-driver", "operation": "resume", "payload": {},
            "payload_digest": payload_digest, "expected_control_revision": 0})
        instance.controller = controller
        outcome = instance.tick(now=1)
        assert outcome.status == "intent_recorded"
        selection = scheduling.Selection.from_dict(outcome.selection)
        assert selection.proposal is not None
        issued_catalog = copy.deepcopy(next(iter(controller._driver_issued.values()))["catalog"])
        pause_digest = driver.campaign_control.command_digest(
            operation="pause", payload={}, campaign_id=enrolled.campaign_id,
            config_generation=1)
        controller.apply_command({
            "schema": driver.campaign_control.COMMAND_SCHEMA,
            "campaign_id": enrolled.campaign_id, "config_generation": 1,
            "request_id": "pause-after-issue", "operation": "pause", "payload": {},
            "payload_digest": pause_digest, "expected_control_revision": 1})
        controller._cached_driver_readiness = (False, "provider expired")
        duplicate_issue = controller.unified_driver_transaction(issued_catalog)
        assert duplicate_issue["status"] == "duplicate"
        held = {
            "schema": scheduling.RECEIPT_SCHEMA, "receipt_id": "held-1",
            "proposal_id": selection.proposal.proposal_id,
            "backend": selection.proposal.backend,
            "stage_class": selection.proposal.stage_class,
            "started_at": 1.0, "ended_at": 2.0,
            "ownership_generation": 1, "allocation_generation": 1,
            "physical_claim_ids": ["cpu-region"],
            "physical_region_fraction": 0.5, "gpu_device_ids": [],
            "memory_reservation_bytes": 0, "affinity_cores": ["0"],
            "beneficiary_shares": {selection.proposal.proposal_id: 1.0},
        }
        controller.register_unified_settlement_validator(lambda value: value)
        settlement_request = {
            "schema": driver.campaign_control.DRIVER_SETTLEMENT_SCHEMA,
            "catalog_id": next(iter(controller._driver_issued)),
            "transition_id": outcome.transition_id,
            "selection": selection.to_dict(), "receipt": held,
            "outcome": "valid_comparison", "terminal_refs": ["native:arm-a", "native:arm-b"],
        }
        settled = controller.unified_driver_settle(settlement_request)
        assert settled["status"] == "accepted"
        assert engine.accounting_view().receipt_count == 1
        with pytest.raises(driver.DriverRefused, match="current exact issued"):
            instance.materialize_runtime(outcome)
        controller._driver_settlement_validator = None
        assert controller.unified_driver_settle(settlement_request)["status"] == "duplicate"
        assert engine.accounting_view().receipt_count == 1
        issued = engine.operational_projection().projection_digest
    replay_engine = scheduling.SchedulerEngine(config, initial)
    with driver.campaign_control.CampaignController(
            enrolled, store, snapshot_version=3, scheduler_engine=replay_engine,
            readiness_check=lambda: (True, None)) as replayed:
        assert replay_engine.operational_projection().projection_digest == issued
        assert len(replayed._driver_issued) == 1
        assert len(replayed._driver_settled) == 1
        assert replay_engine.accounting_view().receipt_count == 1


def test_selected_runtime_materializes_exact_bridge_record_from_startup_input(tmp_path):
    instance, _, enrolled, _, digest = runtime_driver(
        git_source=True, with_execution_input=True)
    base, _ = scheduler()
    config = scheduling.SchedulerConfig.from_dict(
        base.to_dict() | {"config_id": enrolled.campaign_id})
    engine = scheduling.SchedulerEngine(
        config, scheduling.initial_state(config, enrolled.campaign_id))
    with driver.campaign_control.CampaignController(
            enrolled, tmp_path / "materialize", snapshot_version=3,
            scheduler_engine=engine, readiness_check=lambda: (True, None)) as controller:
        payload_digest = driver.campaign_control.command_digest(
            operation="resume", payload={}, campaign_id=enrolled.campaign_id,
            config_generation=1)
        controller.apply_command({
            "schema": driver.campaign_control.COMMAND_SCHEMA,
            "campaign_id": enrolled.campaign_id, "config_generation": 1,
            "request_id": "resume-materialize", "operation": "resume", "payload": {},
            "payload_digest": payload_digest, "expected_control_revision": 0})
        instance.controller = controller
        outcome = instance.tick(now=1)
        prepared = instance.materialize_runtime(outcome)
        assert prepared.dispatch["selection"] == outcome.selection
        assert prepared.plan.digest == instance.experiment_plans[
            "opp-runtime_recipe:threads-4-8"].digest
        assert prepared.runtime_pair.to_dict() == driver._thaw(
            prepared.dispatch["proposal"]["runtime_pair"])
        assert prepared.capture_context_base["campaign_id"] == enrolled.campaign_id
        assert prepared.capture_context_base["supervisor_incarnation"] == 1
        assert prepared.capture_context_base["source_identities"]["anchor"][
            "source_revision"] == "a" * 40
        assert prepared.prompts.digest == instance.execution_inputs[digest].prompt_manifest.digest
        assert prepared.to_dict()["prepared_digest"] == prepared.prepared_digest
        assert prepared.artifact_root == tmp_path / "materialize" / "unified-native-artifacts"
    replay_engine = scheduling.SchedulerEngine(
        config, scheduling.initial_state(config, enrolled.campaign_id))
    with driver.campaign_control.CampaignController(
            enrolled, tmp_path / "materialize", snapshot_version=3,
            scheduler_engine=replay_engine) as reopened:
        instance.controller = reopened
        with pytest.raises(driver.DriverRefused, match="current exact issued"):
            instance.materialize_runtime(outcome)


def test_runtime_materialization_waits_without_target_execution_input(tmp_path):
    instance, engine, _enrolled, _, digest = runtime_driver(with_execution_input=False)
    outcome = instance.tick(now=1)
    assert outcome.status == "waiting"
    assert outcome.reasons == (f"target:{digest}:execution_input_missing",)
    assert engine.operational_projection().body["issued_selection_digests"] == ()


def test_runtime_materialization_refuses_if_input_is_removed_after_issue(tmp_path):
    instance, _, enrolled, _, _ = runtime_driver(git_source=True)
    base, _ = scheduler()
    config = scheduling.SchedulerConfig.from_dict(
        base.to_dict() | {"config_id": enrolled.campaign_id})
    engine = scheduling.SchedulerEngine(
        config, scheduling.initial_state(config, enrolled.campaign_id))
    with driver.campaign_control.CampaignController(
            enrolled, tmp_path / "missing-input", snapshot_version=3,
            scheduler_engine=engine, readiness_check=lambda: (True, None)) as controller:
        payload_digest = driver.campaign_control.command_digest(
            operation="resume", payload={}, campaign_id=enrolled.campaign_id,
            config_generation=1)
        controller.apply_command({
            "schema": driver.campaign_control.COMMAND_SCHEMA,
            "campaign_id": enrolled.campaign_id, "config_generation": 1,
            "request_id": "resume-missing-input", "operation": "resume", "payload": {},
            "payload_digest": payload_digest, "expected_control_revision": 0})
        instance.controller = controller
        outcome = instance.tick(now=1)
        instance.execution_inputs = {}
        with pytest.raises(driver.DriverRefused, match="waiting for frozen prompts"):
            instance.materialize_runtime(outcome)


def test_v3_snapshot_is_closed_bounded_and_fences_v2_reader(tmp_path):
    _instance, _engine, enrolled, _, _ = runtime_driver()
    base_config, _ = scheduler()
    config = scheduling.SchedulerConfig.from_dict(
        base_config.to_dict() | {"config_id": enrolled.campaign_id})
    initial = scheduling.initial_state(config, enrolled.campaign_id)
    engine = scheduling.SchedulerEngine(config, initial)
    store = tmp_path / "v3"
    with driver.campaign_control.CampaignController(
            enrolled, store, snapshot_version=3, scheduler_engine=engine) as controller:
        snapshot = controller.snapshot()
        assert snapshot["schema"] == driver.campaign_control.SNAPSHOT_SCHEMA_V3
        assert snapshot["unified"]["resources"]["status"] == "not_connected"
        assert snapshot["unified"]["targets"]["items_page_ref"] is None
        assert snapshot["unified"]["scheduler"]["capacity"] == engine.capacity.to_dict()
        driver.campaign_control.validate_snapshot_v3(snapshot)
    with pytest.raises(driver.campaign_control.ControlRefused, match="v3.*downgrade"):
        driver.campaign_control.CampaignController(
            enrolled, store, snapshot_version=2).__enter__()


def test_v3_nested_projection_rejects_forged_disconnected_values(tmp_path):
    _instance, _engine, enrolled, _, _ = runtime_driver()
    base, _ = scheduler()
    config = scheduling.SchedulerConfig.from_dict(
        base.to_dict() | {"config_id": enrolled.campaign_id})
    engine = scheduling.SchedulerEngine(
        config, scheduling.initial_state(config, enrolled.campaign_id))
    with driver.campaign_control.CampaignController(
            enrolled, tmp_path / "v3-malformed", snapshot_version=3,
            scheduler_engine=engine) as controller:
        source = controller.snapshot()
    mutations = [
        ("resources", "requested", {}), ("actors", "items", [{}]),
        ("evidence", "lag_seconds", float("nan")),
        ("candidate", "validated_identity", "label"),
        ("targets", "ready", True),
        ("scheduler", "pending_selection_digest", "not-a-digest"),
        ("scheduler", "capacity", {"asserted": True}),
    ]
    for section, field, value in mutations:
        malformed = copy.deepcopy(source)
        malformed["unified"][section][field] = value
        with pytest.raises(driver.campaign_control.ControlRefused):
            driver.campaign_control.validate_snapshot_v3(malformed)


def test_provider_refresh_does_not_hold_controller_mutex(tmp_path):
    _instance, _engine, enrolled, _, _ = runtime_driver()
    base, _ = scheduler()
    config = scheduling.SchedulerConfig.from_dict(
        base.to_dict() | {"config_id": enrolled.campaign_id})
    engine = scheduling.SchedulerEngine(
        config, scheduling.initial_state(config, enrolled.campaign_id))
    entered, release = threading.Event(), threading.Event()

    def blocking_readiness():
        entered.set()
        assert release.wait(2)
        return True, None

    with driver.campaign_control.CampaignController(
            enrolled, tmp_path / "readiness", snapshot_version=3,
            scheduler_engine=engine, readiness_check=blocking_readiness) as controller:
        thread = threading.Thread(target=controller.refresh_unified_driver_readiness)
        thread.start()
        assert entered.wait(1)
        assert controller.snapshot()["schema"] == driver.campaign_control.SNAPSHOT_SCHEMA_V3
        release.set()
        thread.join(2)
        assert not thread.is_alive()


def test_stale_readiness_callback_cannot_update_after_controller_close(tmp_path):
    _instance, _engine, enrolled, _, _ = runtime_driver()
    base, _ = scheduler()
    config = scheduling.SchedulerConfig.from_dict(
        base.to_dict() | {"config_id": enrolled.campaign_id})
    first_engine = scheduling.SchedulerEngine(
        config, scheduling.initial_state(config, enrolled.campaign_id))
    entered, release = threading.Event(), threading.Event()

    def blocking_readiness():
        entered.set()
        assert release.wait(2)
        return True, None

    store = tmp_path / "stale-readiness"
    first = driver.campaign_control.CampaignController(
        enrolled, store, snapshot_version=3, scheduler_engine=first_engine,
        readiness_check=blocking_readiness)
    first.__enter__()
    result = {}

    def refresh():
        try:
            result["value"] = first.refresh_unified_driver_readiness()
        except Exception as exc:  # expected stale-lifetime refusal
            result["error"] = exc

    thread = threading.Thread(target=refresh)
    thread.start()
    assert entered.wait(1)
    first.close()
    second_engine = scheduling.SchedulerEngine(
        config, scheduling.initial_state(config, enrolled.campaign_id))
    with driver.campaign_control.CampaignController(
            enrolled, store, snapshot_version=3,
            scheduler_engine=second_engine) as second:
        release.set()
        thread.join(2)
        assert not thread.is_alive()
        assert isinstance(result.get("error"), driver.campaign_control.ControlRefused)
        assert second.unified_driver_readiness()["provider_available"] is False


def test_stale_settlement_verifier_cannot_commit_after_controller_close(tmp_path):
    instance, _, enrolled, _, _ = runtime_driver()
    base, _ = scheduler()
    config = scheduling.SchedulerConfig.from_dict(
        base.to_dict() | {"config_id": enrolled.campaign_id})
    initial = scheduling.initial_state(config, enrolled.campaign_id)
    first_engine = scheduling.SchedulerEngine(config, initial)
    store = tmp_path / "stale-settlement"
    first = driver.campaign_control.CampaignController(
        enrolled, store, snapshot_version=3, scheduler_engine=first_engine,
        readiness_check=lambda: (True, None))
    first.__enter__()
    payload_digest = driver.campaign_control.command_digest(
        operation="resume", payload={}, campaign_id=enrolled.campaign_id,
        config_generation=1)
    first.apply_command({
        "schema": driver.campaign_control.COMMAND_SCHEMA,
        "campaign_id": enrolled.campaign_id, "config_generation": 1,
        "request_id": "resume-stale", "operation": "resume", "payload": {},
        "payload_digest": payload_digest, "expected_control_revision": 0})
    instance.controller = first
    issued = instance.tick(now=1)
    selection = scheduling.Selection.from_dict(issued.selection)
    assert selection.proposal is not None
    request = {
        "schema": driver.campaign_control.DRIVER_SETTLEMENT_SCHEMA,
        "catalog_id": next(iter(first._driver_issued)),
        "transition_id": issued.transition_id,
        "selection": selection.to_dict(),
        "receipt": {
            "schema": scheduling.RECEIPT_SCHEMA, "receipt_id": "held-stale",
            "proposal_id": selection.proposal.proposal_id,
            "backend": selection.proposal.backend,
            "stage_class": selection.proposal.stage_class,
            "started_at": 1.0, "ended_at": 2.0,
            "ownership_generation": 1, "allocation_generation": 1,
            "physical_claim_ids": ["cpu-region"],
            "physical_region_fraction": 0.5, "gpu_device_ids": [],
            "memory_reservation_bytes": 0, "affinity_cores": ["0"],
            "beneficiary_shares": {selection.proposal.proposal_id: 1.0},
        },
        "outcome": "valid_comparison", "terminal_refs": ["native:a", "native:b"],
    }
    entered, release = threading.Event(), threading.Event()

    def blocked(value):
        entered.set()
        assert release.wait(2)
        return value

    first.register_unified_settlement_validator(blocked)
    result = {}

    def settle():
        try:
            result["value"] = first.unified_driver_settle(request)
        except Exception as exc:  # expected stale-lifetime refusal
            result["error"] = exc

    thread = threading.Thread(target=settle)
    thread.start()
    assert entered.wait(1)
    first.close()
    replay_engine = scheduling.SchedulerEngine(config, initial)
    with driver.campaign_control.CampaignController(
            enrolled, store, snapshot_version=3,
            scheduler_engine=replay_engine) as current:
        release.set()
        thread.join(2)
        assert not thread.is_alive()
        assert isinstance(result.get("error"), driver.campaign_control.ControlRefused)
        assert current._driver_settled == {}
        assert replay_engine.accounting_view().receipt_count == 0


def test_unified_scheduler_must_bind_campaign_ids(tmp_path):
    _instance, engine, enrolled, _, _ = runtime_driver()
    with pytest.raises(driver.campaign_control.ControlRefused, match="exactly bind"):
        driver.campaign_control.CampaignController(
            enrolled, tmp_path / "foreign", snapshot_version=3,
            scheduler_engine=engine)
