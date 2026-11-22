from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time

import pytest

from .. import journal as journal_module
from . import (campaign_control, native_model_preparation, native_retention_catalog as catalog,
               scheduling, worker_lifecycle)
from . import candidate_transactions
from .test_candidate_manifest import _base, _state
from .test_candidate_transactions import FakeGitBackend
from .test_retention_consumer import _node, _policy, _view
from . import retention_consumer
from .test_unified_planner import (campaign_for_recipe, canonical_recipe, prepared,
                                   runtime_anchor, scheduler)
from .test_unified_driver import runtime_driver
from .test_worker_lifecycle import MockProvider


@pytest.fixture
def retention_tmp_path():
    path = Path(tempfile.mkdtemp(prefix="_native_catalog_", dir=Path(__file__).parent))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _seed(*, shared=False, manifest_path=None, artifact_root=None):
    recipe = canonical_recipe()
    resolved = campaign_for_recipe(recipe)
    if shared:
        target = resolved.targets[0]
        resolved = replace(resolved, targets=(target, replace(
            target, target_ids=("alias",), revision=target.revision + 1,
            spec_digests=("9" * 64,))))
    with tempfile.TemporaryDirectory(prefix="_seed_anchor_") as anchor_dir:
        anchor_tmp_path = Path(anchor_dir)
        anchors = prepared(resolved, {
            catalog._digest(target.to_dict()): runtime_anchor(target, recipe, anchor_tmp_path)
            for target in resolved.targets})
    preparations = {}
    for target_digest, recipe in anchors.recipes.items():
        inventory = {"model_id": recipe.model.path,
                     "model_manifest": str(manifest_path or "/models/manifest.json"),
                     "model_manifest_sha256": "7" * 64,
                     "model_sha256": "8" * 64}
        if manifest_path is not None:
            inventory["model_manifest_sha256"] = hashlib.sha256(
                manifest_path.read_bytes()).hexdigest()
            inventory["model_id"] = json.loads(
                manifest_path.read_text(encoding="utf-8"))["model_path"]
        body = {
            "schema": "epyc.autokernel.scheduled_model_preparation.v1",
            "target_revision_digest": target_digest,
            "recipe_execution_digest": recipe.execution_digest,
            "entry_path": recipe.model.path, "entry_sha256": recipe.model.sha256,
            "inventory_identity": inventory,
        }
        preparations[target_digest] = {recipe.execution_digest:
            native_model_preparation.ScheduledModelPreparation.from_dict(
                body | {"preparation_digest": worker_lifecycle._digest(body)})}
    return resolved, anchors, catalog.build_seed(
        resolved, anchors, config_digest=campaign_control.resolved_config_digest(resolved),
        model_preparations=preparations, artifact_root=artifact_root)


def test_seed_is_closed_and_install_event_round_trips():
    _resolved, _anchors, seed = _seed()
    event = catalog.make_install_event(seed, config_generation=2,
                                       supervisor_incarnation=3)
    assert catalog._plain(catalog.validate_install_event(event)) == event
    changed = dict(event)
    changed["seed"] = {**event["seed"], "unknown": True}
    with pytest.raises(catalog.NativeRetentionCatalogRefused,
                       match="seed fields differ"):
        catalog.validate_install_event(changed)
    changed_inventory = json.loads(json.dumps(event))
    changed_inventory["seed"]["model_inventories"][0]["unknown"] = True
    with pytest.raises(catalog.NativeRetentionCatalogRefused,
                       match="owning typed record"):
        catalog.validate_install_event(changed_inventory)


def test_shared_physical_artifacts_merge_target_owners_and_roots():
    _resolved, _anchors, seed = _seed(shared=True)
    model = next(item for item in seed.artifacts
                 if any(owner["role"] == "model"
                        for owner in item.provenance["owners"]))
    assert len(model.provenance["owners"]) == 2


def _recipe_path_alias(recipe, old="/build", new="/private-copy"):
    def changed(value):
        return value.replace(old, new) if isinstance(value, str) else value

    candidate = replace(
        recipe, build_dir=changed(recipe.build_dir),
        command_argv=tuple(changed(item) for item in recipe.command_argv),
        argv=tuple(changed(item) for item in recipe.argv),
        launch_env=tuple((key, changed(value)) for key, value in recipe.launch_env),
        relevant_environment=tuple(
            (key, changed(value)) for key, value in recipe.relevant_environment),
        readback_expectations=tuple(
            (key, changed(value)) for key, value in recipe.readback_expectations),
        executable=replace(recipe.executable, path=changed(recipe.executable.path)),
        dsos=tuple(replace(item, path=changed(item.path)) for item in recipe.dsos),
        runtime_binary_dir=changed(recipe.runtime_binary_dir),
        runtime_ld_paths=tuple(changed(item) for item in recipe.runtime_ld_paths),
        provenance=tuple((key, changed(value)) for key, value in recipe.provenance),
        snapshot_digest="0" * 64)
    candidate = replace(candidate, snapshot_digest=catalog._digest(candidate._snapshot_dict()))
    return type(recipe).from_dict(candidate.to_dict())


def test_snapshot_keyed_recipe_alias_retains_both_physical_executables():
    resolved, anchors, seed = _seed()
    target = next(iter(anchors.recipes))
    original = anchors.recipes[target]
    alias = _recipe_path_alias(original)
    assert alias.execution_digest == original.execution_digest
    assert alias.snapshot_digest != original.snapshot_digest
    preparations = _preparations(seed)
    rebuilt = catalog.build_seed(
        resolved, anchors, config_digest=seed.config_digest,
        model_preparations=preparations,
        runtime_recipes={target: {original.execution_digest: original}},
        runtime_recipe_snapshots={target: {
            original.snapshot_digest: original, alias.snapshot_digest: alias}})
    executable_paths = {item.path for item in rebuilt.artifacts
                        if any(owner.get("role") == "executable"
                               for owner in item.provenance.get("owners", ()))}
    assert executable_paths == {original.executable.path, alias.executable.path}
    assert {owner["recipe_snapshot_digest"] for item in rebuilt.artifacts
            for owner in item.provenance.get("owners", ())
            if owner.get("role") == "executable"} == {
                original.snapshot_digest, alias.snapshot_digest}
    with pytest.raises(catalog.NativeRetentionCatalogRefused,
                       match="snapshot key differs"):
        catalog.build_seed(
            resolved, anchors, config_digest=seed.config_digest,
            model_preparations=preparations,
            runtime_recipe_snapshots={target: {"f" * 64: alias}})


def test_missing_model_inventory_is_explicit_coverage_debt(tmp_path):
    recipe = canonical_recipe()
    resolved = campaign_for_recipe(recipe)
    anchors = prepared(resolved, {
        catalog._digest(target.to_dict()): runtime_anchor(target, recipe, tmp_path)
        for target in resolved.targets})
    seed = catalog.build_seed(
        resolved, anchors, config_digest=campaign_control.resolved_config_digest(resolved))
    assert any(item.endswith("model_inventory_missing")
               for item in seed.uncertain_scopes)


def test_config_digest_is_rederived_not_shape_checked(tmp_path):
    recipe = canonical_recipe()
    resolved = campaign_for_recipe(recipe)
    anchors = prepared(resolved, {
        catalog._digest(target.to_dict()): runtime_anchor(target, recipe, tmp_path)
        for target in resolved.targets})
    with pytest.raises(catalog.NativeRetentionCatalogRefused,
                       match="differs from resolved"):
        catalog.build_seed(resolved, anchors, config_digest="f" * 64)


def test_seed_refuses_mapping_that_only_looks_like_model_preparation():
    resolved, anchors, seed = _seed()
    preparations = _preparations(seed)
    target = next(iter(preparations))
    recipe = next(iter(preparations[target]))
    row = preparations[target][recipe]
    with pytest.raises(catalog.NativeRetentionCatalogRefused,
                       match="owning typed record"):
        catalog.build_seed(
            resolved, anchors, config_digest=seed.config_digest,
            model_preparations={target: {recipe: row | {"unknown": True}}})


def _controller(resolved, store, *, lifecycle_provider=None):
    config, _engine = scheduler()
    config = scheduling.SchedulerConfig.from_dict(
        config.to_dict() | {"config_id": resolved.campaign_id})
    engine = scheduling.SchedulerEngine(
        config, scheduling.initial_state(config, resolved.campaign_id))
    return campaign_control.CampaignController(
        resolved, store, snapshot_version=3, scheduler_engine=engine,
        lifecycle_provider=lifecycle_provider)


def test_controller_installs_exact_rederived_seed_and_replays(tmp_path):
    resolved, anchors, seed = _seed()
    store = tmp_path / "service"
    store.mkdir(mode=0o700)
    store.chmod(0o700)
    controller = _controller(resolved, store)
    controller.__enter__()
    try:
        row = controller.install_native_retention_catalog(
            seed, runtime_anchors=anchors,
            model_preparations={target: {recipe: item for recipe, item in recipes.items()}
                                for target, recipes in _preparations(seed).items()},
            artifact_root=None)
        assert row["seed_digest"] == seed.seed_digest
        assert controller.install_native_retention_catalog(
            seed, runtime_anchors=anchors, model_preparations=_preparations(seed),
            artifact_root=None) == row
    finally:
        controller.close()
    replayed = _controller(resolved, store)
    replayed.__enter__()
    try:
        assert replayed.native_retention_catalog_capture()["seed"].seed_digest \
               == seed.seed_digest
    finally:
        replayed.close()


def test_v2_controller_refuses_store_with_native_catalog_before_admission(tmp_path):
    resolved, anchors, seed = _seed()
    store = tmp_path / "service"
    with _controller(resolved, store) as controller:
        controller.install_native_retention_catalog(
            seed, runtime_anchors=anchors, model_preparations=_preparations(seed),
            artifact_root=None)
    with pytest.raises(campaign_control.ControlRefused, match="v3"):
        with campaign_control.CampaignController(resolved, store, snapshot_version=2):
            pass


def _preparations(seed):
    result = {}
    for row in seed.model_inventories:
        result.setdefault(row["target_revision_digest"], {})[
            row["recipe_execution_digest"]] = row
    return result


def test_controller_refuses_seed_not_rederived_from_owning_inputs(tmp_path):
    resolved, anchors, seed = _seed()
    changed = catalog.build_seed(
        resolved, anchors, config_digest=seed.config_digest,
        model_preparations=_preparations(seed), artifact_root=tmp_path / "other")
    with _controller(resolved, tmp_path / "service") as controller:
        with pytest.raises(campaign_control.ControlRefused, match="rederived"):
            controller.install_native_retention_catalog(
                changed, runtime_anchors=anchors, model_preparations=_preparations(seed),
                artifact_root=None)


def test_actual_candidate_owner_collects_outside_mutex_and_rechecks_frontier(
        tmp_path, monkeypatch):
    manifest_path = tmp_path / "model-inventory.json"
    manifest_path.write_text(json.dumps({
        "schema": "epyc.autokernel.model_identity.v1",
        "model_path": "/models",
        "files": [{"path": ("model.gguf" if index == 0 else
                              f"model-{index:05d}-of-00006.gguf"),
                   "sha256": ("a" * 64 if index == 0 else f"{index + 1:x}" * 64)}
                  for index in range(6)],
    }, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    resolved, anchors, seed = _seed(manifest_path=manifest_path)
    store = tmp_path / "service"
    with _controller(resolved, store) as controller:
        controller.install_native_retention_catalog(
            seed, runtime_anchors=anchors, model_preparations=_preparations(seed),
            artifact_root=None)
        manager = candidate_transactions.CandidateTransactions(
            controller, git_backend=FakeGitBackend())
        base = _base()
        manager.initialize(request_id="init", state=_state(base), manifest=base)
        calls = []

        def measured_directory(path):
            assert not controller._mutex._is_owned()
            calls.append(path)
            return "d" * 64

        monkeypatch.setattr(catalog.storage, "hash_tree_manifest", measured_directory)
        view = manager.retention_view()
        assert calls and not any(scope.endswith("model_inventory_unverified")
                                 for scope in view.uncertain_scopes)
        assert len([item for item in view.nodes
                    if item.artifact_id.startswith("native:model-shard:")]) == 6
        capture = controller.candidate_transaction(
            lambda context: controller._native_retention_catalog_capture_locked(
                manager._replay(context).state))
        prepared_view = catalog.build_native_view(capture)
    with _controller(resolved, store) as reopened:
        with pytest.raises(campaign_control.ControlRefused, match="changed"):
            reopened.validate_native_retention_frontier(prepared_view.frontier)


def test_inventory_reader_refuses_fifo_and_oversize_without_blocking(tmp_path):
    fifo = tmp_path / "inventory.fifo"
    fifo.parent.mkdir(exist_ok=True)
    fifo.touch()
    fifo.unlink()
    import os
    os.mkfifo(fifo)
    with pytest.raises(catalog.NativeRetentionCatalogRefused, match="regular file"):
        catalog._bounded_regular_bytes(fifo, 32, "model inventory")
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"x" * 33)
    with pytest.raises(catalog.NativeRetentionCatalogRefused, match="regular file"):
        catalog._bounded_regular_bytes(oversized, 32, "model inventory")


def test_prepared_job_binding_refuses_changed_policy_and_selected_prefix(
        retention_tmp_path):
    tmp_path = retention_tmp_path
    resolved, anchors, seed = _seed()
    store = tmp_path / "service"
    store.mkdir(mode=0o700)
    store.chmod(0o700)
    controller = _controller(resolved, store)
    controller.__enter__()
    try:
        controller.install_native_retention_catalog(
            seed, runtime_anchors=anchors, model_preparations=_preparations(seed),
            artifact_root=None)
        frontier = controller.native_retention_catalog_capture()["frontier"]
        view, _old = _view(tmp_path)
        second_path = tmp_path / "owned" / "second"
        second_path.mkdir(parents=True)
        (second_path / "artifact").write_text("second", encoding="utf-8")
        second = _node(second_path)
        second = replace(second, artifact_id="old-build-second",
                         expiry=replace(second.expiry,
                                        campaign_id=resolved.campaign_id))
        first = replace(view.nodes[-1], expiry=replace(
            view.nodes[-1].expiry, campaign_id=resolved.campaign_id))
        second_identity = retention_consumer.NativeArtifactIdentity(
            second.artifact_id, str(second_path), second.expiry.sha256,
            "experiment", view.identities[0].source)
        view = replace(
            view, snapshot_id=seed.seed_digest,
            generation=max(1, int(frontier[2]) + 1),
            nodes=view.nodes[:-1] + (first, second),
            identities=view.identities + (second_identity,))
        snapshot = retention_consumer.collect_native_snapshot(view)
        prepared = catalog.PreparedCatalogView(
            view, frontier, snapshot.snapshot_digest, catalog._VIEW_TOKEN)
        policy = _policy(tmp_path)
        job, previews = retention_consumer.prepare(view, policy, limit=1)
        assert job is not None and len(previews) == 1 \
            and len(job.plan.expirable_ids) == 2
        controller.bind_prepared_retention_job(job, prepared=prepared, policy=policy)
        with pytest.raises(campaign_control.ControlRefused, match="preparation"):
            controller.maintenance_admit(replace(job, policy_digest="f" * 64))
        with pytest.raises(campaign_control.ControlRefused, match="preparation"):
            controller.maintenance_admit(replace(
                job, selected_artifact_ids=(job.plan.expirable_ids[1],)))
    finally:
        controller.close()


def test_actual_issued_driver_intent_roots_selected_recipe_dependencies(
        tmp_path, monkeypatch):
    instance, _engine, resolved, _target, target_digest = runtime_driver()
    anchor = instance.runtime_anchors.recipes[target_digest]
    arm_pair = catalog.unified_planner.enumerate_runtime_dimensions(
        anchor, instance.runtime_dimensions[target_digest])[0]
    recipes = {item.execution_digest: item
               for item in (arm_pair.anchor, arm_pair.candidate)}
    recipe = anchor
    manifest_path = tmp_path / "inventory.json"
    manifest_path.write_text(json.dumps({
        "schema": "epyc.autokernel.model_identity.v1", "model_path": "/models",
        "files": [{"path": "model.gguf", "sha256": recipe.model.sha256}],
    }, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    preparations = {target_digest: {}}
    for runtime_recipe in recipes.values():
        body = {"schema": native_model_preparation.SPEC_SCHEMA,
                "target_revision_digest": target_digest,
                "recipe_execution_digest": runtime_recipe.execution_digest,
                "entry_path": runtime_recipe.model.path,
                "entry_sha256": runtime_recipe.model.sha256,
                "inventory_identity": {
                    "model_id": "/models", "model_manifest": str(manifest_path),
                    "model_manifest_sha256": hashlib.sha256(
                        manifest_path.read_bytes()).hexdigest(), "model_sha256": "8" * 64}}
        preparations[target_digest][runtime_recipe.execution_digest] = \
            native_model_preparation.ScheduledModelPreparation.from_dict(
                body | {"preparation_digest": worker_lifecycle._digest(body)})
    runtime_recipes = {target_digest: recipes}
    seed = catalog.build_seed(
        resolved, instance.runtime_anchors,
        config_digest=campaign_control.resolved_config_digest(resolved),
        model_preparations=preparations, runtime_recipes=runtime_recipes)
    containers = tmp_path / "containers"
    containers.mkdir()
    controller = _controller(
        resolved, tmp_path / "service", lifecycle_provider=MockProvider(containers))
    controller.readiness_check = lambda: (True, None)
    instance.scheduler_engine = controller._scheduler_engine
    with controller:
        controller.install_native_retention_catalog(
            seed, runtime_anchors=instance.runtime_anchors,
            model_preparations=preparations, runtime_recipes=runtime_recipes,
            artifact_root=None)
        payload_digest = campaign_control.command_digest(
            operation="resume", payload={}, campaign_id=resolved.campaign_id,
            config_generation=1)
        controller.apply_command({
            "schema": campaign_control.COMMAND_SCHEMA,
            "campaign_id": resolved.campaign_id, "config_generation": 1,
            "request_id": "resume-retention", "operation": "resume", "payload": {},
            "payload_digest": payload_digest, "expected_control_revision": 0})
        instance.controller = controller
        outcome = instance.tick(now=1)
        assert outcome.status == "intent_recorded"
        manager = candidate_transactions.CandidateTransactions(
            controller, git_backend=FakeGitBackend())
        base = _base()
        manager.initialize(request_id="init", state=_state(base), manifest=base)
        monkeypatch.setattr(catalog.storage, "hash_tree_manifest", lambda _path: "d" * 64)
        proposal = scheduling.StageProposal.from_dict(outcome.selection["proposal"])
        request = worker_lifecycle.StageRequest(
            "retention-live-worker", proposal.digest,
            f"driver:{outcome.transition_id}", "retention-live-stage", "sampling",
            (sys.executable, "-B", "-c", "import time; time.sleep(2)"),
            {"PATH": os.environ.get("PATH", "/usr/bin"),
             "PYTHONDONTWRITEBYTECODE": "1"}, tmp_path, "e" * 64, 3.0, 0.5,
            controller.control_revision)
        terminals, errors = [], []

        def run_worker():
            try:
                terminals.append(controller.run_worker_stage(request))
            except BaseException as exc:
                errors.append(exc)

        worker = threading.Thread(target=run_worker)
        worker.start()
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline and not (
                controller._worker_run_active and controller._active_worker_events
                and controller._active_worker_events[-1]["event"] == "WORKER_STAGE"):
            time.sleep(0.005)
        assert controller._worker_run_active \
            and controller._active_worker_events[-1]["event"] == "WORKER_STAGE"
        view = manager.retention_view()
        selected_recipe = next(item for item in seed.artifacts
            if any(owner.get("recipe_execution_digest") == anchor.execution_digest
                   for owner in item.provenance.get("owners", ())))
        assert selected_recipe.artifact_id in view.roots.launch_intents
        assert {owner["recipe_execution_digest"]
                for owner in selected_recipe.provenance["owners"]} == set(recipes)
        assert "active_worker_plan_dependency_join_unavailable" \
            not in view.uncertain_scopes
        worker.join(6)
        assert not worker.is_alive() and not errors and terminals[0].accepted
        results = controller._retention_owner_records["worker_result"].artifacts
        assert [item.provenance["owners"][0]["stage_id"] for item in results] \
            == ["retention-live-stage"]
        assert results[0].retention_class == "hash_and_provenance_only" \
            and results[0].root_kinds == ("retained_evidence",)
        kinds = [entry.kind for entry in controller._journal.read_all()]
        assert journal_module.KIND_WORKER_LIFECYCLE in kinds
        assert journal_module.KIND_WORKER_ACQUISITION in kinds


def _complete_controller(tmp_path, monkeypatch, *, artifact_root=True):
    """Fixture campaign whose every owner class can register a complete record."""
    manifest_path = tmp_path / "inventory.json"
    manifest_path.write_text(json.dumps({
        "schema": "epyc.autokernel.model_identity.v1", "model_path": "/models",
        "files": [{"path": "model.gguf", "sha256": canonical_recipe().model.sha256}],
    }, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    artifact_root = tmp_path / "artifacts" if artifact_root else None
    resolved, anchors, seed = _seed(manifest_path=manifest_path, artifact_root=artifact_root)
    monkeypatch.setattr(catalog.storage, "hash_tree_manifest", lambda _path: "d" * 64)
    controller = _controller(resolved, tmp_path / "service")
    controller.__enter__()
    controller.install_native_retention_catalog(
        seed, runtime_anchors=anchors, model_preparations=_preparations(seed),
        artifact_root=artifact_root)
    manager = candidate_transactions.CandidateTransactions(
        controller, git_backend=FakeGitBackend())
    return resolved, anchors, seed, controller, manager


def _plan(view):
    return catalog.retention.plan_retention(
        retention_consumer.collect_native_snapshot(view))


def test_owner_hooks_register_each_class_and_complete_catalog_admits_view_only(
        tmp_path, monkeypatch):
    resolved, anchors, seed, controller, manager = _complete_controller(
        tmp_path, monkeypatch)
    try:
        records = controller._retention_owner_records
        assert set(records) == set(catalog.OWNER_CLASSES)
        generation = (seed.seed_digest, controller.config_generation,
                      controller.supervisor_incarnation)
        assert all(item.generation == generation and not item.gaps
                   for item in records.values())
        assert records["physical_root"].artifacts[0].provenance["owners"][0]["path"] \
            == str(controller.store)
        before = records["candidate_manifest"].record_digest
        base = _base()
        manager.initialize(request_id="init", state=_state(base), manifest=base)
        manifest_ids = {item.path for item in records["candidate_manifest"].artifacts}
        assert records["candidate_manifest"].record_digest != before
        assert manifest_ids == {base.builds[0].executable.path,
                                base.builds[0].dsos[0].path}
        view = manager.retention_view()
        plan = _plan(view)
        assert view.uncertain_scopes == () and plan.complete and not plan.expirable_ids
        job, previews = retention_consumer.prepare(view, _policy(tmp_path))
        assert job is None and previews == ()
        with pytest.raises(retention_consumer.RetentionConsumerRefused, match="unavailable"):
            retention_consumer.MaintenanceOwner().held(lambda lease: lease)
    finally:
        controller.close()
    with _controller(resolved, tmp_path / "service") as reopened:
        reopened_manager = candidate_transactions.CandidateTransactions(
            reopened, git_backend=FakeGitBackend())
        assert reopened._retention_owner_records["candidate_manifest"].generation[2] \
            == reopened.supervisor_incarnation
        assert reopened_manager.retention_view().uncertain_scopes == ()


def test_missing_or_stale_owner_class_keeps_refusing(tmp_path, monkeypatch):
    _resolved, _anchors, seed, controller, manager = _complete_controller(
        tmp_path, monkeypatch)
    try:
        base = _base()
        manager.initialize(request_id="init", state=_state(base), manifest=base)
        for owner_class in catalog.OWNER_CLASSES:
            saved = controller._retention_owner_records.pop(owner_class)
            view = manager.retention_view()
            assert view.uncertain_scopes == (f"owner_coverage_missing:{owner_class}",)
            assert not _plan(view).complete
            controller._retention_owner_records[owner_class] = saved
        controller._retention_owner_records["evidence"] = catalog.derive_owner_record(
            "evidence", seed=seed,
            generation=(seed.seed_digest, controller.config_generation,
                        controller.supervisor_incarnation - 1))
        assert manager.retention_view().uncertain_scopes == (
            "owner_coverage_stale:evidence",)
        controller._register_retention_owners_locked("evidence")
        # A durable owner write without its hook leaves that class stale.
        controller._retention_worker_results[("unhooked", 1)] = {
            "worker_id": "unhooked", "worker_generation": 1, "lineage_id": "x",
            "stage_id": "y", "data": {"result_digest": "e" * 64}}
        assert manager.retention_view().uncertain_scopes == (
            "owner_coverage_stale:worker_result",)
    finally:
        controller.close()


def test_owner_gaps_refuse_without_artifact_store(tmp_path, monkeypatch):
    _resolved, _anchors, seed, controller, manager = _complete_controller(
        tmp_path, monkeypatch, artifact_root=False)
    try:
        base = _base()
        manager.initialize(request_id="init", state=_state(base), manifest=base)
        assert manager.retention_view().uncertain_scopes == (
            "owner_coverage_gap:physical_root:artifact_store_undeclared",)
    finally:
        controller.close()
    record = catalog.derive_owner_record(
        "evidence", seed=seed, generation=(seed.seed_digest, 1, 1),
        native_records={"m1": {"payload": {"artifact": {
            "locator": "objects/m1", "sha256": "c" * 64}}}})
    assert record.gaps == ("native_capture_artifact_root_missing",)


def test_caller_supplied_roots_or_membership_are_rejected(tmp_path, monkeypatch):
    _resolved, _anchors, seed, controller, manager = _complete_controller(
        tmp_path, monkeypatch)
    try:
        base = _base()
        manager.initialize(request_id="init", state=_state(base), manifest=base)
        record = controller._retention_owner_records["worker_result"]
        extra = catalog.CatalogArtifact(
            "caller:root", "evidence_record", None, "f" * 64, "file_sha256", (),
            "hash_and_provenance_only", ("production",), {"owners": [{"role": "caller"}]})
        with pytest.raises(catalog.NativeRetentionCatalogRefused, match="derivation"):
            catalog.OwnerRecord(*[getattr(record, name) for name in (
                "owner_class", "generation", "inputs_digest", "artifacts", "gaps",
                "record_digest")], object())
        with pytest.raises(catalog.NativeRetentionCatalogRefused, match="digest differs"):
            replace(record, artifacts=(extra,))
        values = (record.owner_class, record.generation, record.inputs_digest,
                  (extra,), ())
        forged = catalog.OwnerRecord(
            *values, catalog._digest(catalog._owner_body(*values)), catalog._OWNER_TOKEN)
        controller._retention_owner_records["worker_result"] = forged
        with pytest.raises(catalog.NativeRetentionCatalogRefused,
                           match="differs from owner state"):
            manager.retention_view()
        controller._retention_owner_records["worker_result"] = record
        capture = dict(controller.candidate_transaction(
            lambda context: controller._native_retention_catalog_capture_locked(
                manager._replay(context).state)))
        capture["owner_records"] = {**capture["owner_records"],
                                    "worker_result": record.body()}
        with pytest.raises(catalog.NativeRetentionCatalogRefused, match="owner-derived"):
            catalog.build_native_view(capture)
        assert manager.retention_view().uncertain_scopes == ()
    finally:
        controller.close()


def test_build_dso_and_runpath_dependencies_are_transitively_retained(
        tmp_path, monkeypatch):
    _resolved, _anchors, seed, controller, manager = _complete_controller(
        tmp_path, monkeypatch)
    try:
        base = _base()
        manager.initialize(request_id="init", state=_state(base), manifest=base)
        view = manager.retention_view()
        nodes = {item.artifact_id: item for item in view.nodes}
        build = base.builds[0]
        executable = catalog._build_artifact_id(seed, build.executable)
        dso = catalog._build_artifact_id(seed, build.dsos[0])
        runpath = catalog._directory_id(str(Path(build.dsos[0].path).parent))
        assert set(nodes[executable].dependencies) == {dso, runpath}
        assert nodes[runpath].dependencies and dso in nodes[runpath].dependencies
        plan = _plan(view)
        for artifact_id in (executable, dso, runpath):
            assert artifact_id in plan.retained_ids
            assert any(reason.startswith("integration_candidate:")
                       for reason in plan.retained_reasons[artifact_id])
        recipe_executable = next(item for item in seed.artifacts if any(
            owner.get("role") == "executable" for owner in item.provenance.get("owners", ())))
        recipe_runpaths = {item.artifact_id for item in seed.artifacts if any(
            owner.get("role") == "runpath" for owner in item.provenance.get("owners", ()))}
        assert recipe_runpaths <= set(nodes[recipe_executable.artifact_id].dependencies)
    finally:
        controller.close()
