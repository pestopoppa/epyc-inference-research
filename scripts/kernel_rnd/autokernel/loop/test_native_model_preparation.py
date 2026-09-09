from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from . import measurement_capture as mc
from . import lifecycle_observation as lo
from . import native_model_preparation as model_prep
from . import native_capture_control as nc
from . import native_parent_receipt_replay as replay
from . import native_parent_service as service
from . import native_scientific_witness as scientific
from . import observation_binding as ob
from . import worker_lifecycle as wl
from .test_driver_execution import _run_real_controller_child_v2_capture_and_restart
from .test_unified_planner import canonical_recipe


def _inventory(tmp_path: Path):
    root = tmp_path / "model"
    root.mkdir()
    files = []
    for index in range(6):
        path = root / f"part-{index}.gguf"
        path.write_bytes(f"fixture shard {index}".encode())
        files.append({"path": path.name,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    material = {"model_path": str(root), "files": files}
    manifest = tmp_path / "model-manifest.json"
    manifest.write_text(json.dumps({"schema": "epyc.autokernel.model_identity.v1",
                                    **material}), encoding="utf-8")
    identity = {"model_id": str(root), "model_manifest": str(manifest),
        "model_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(
            scientific.tensor_capture._canonical(material).encode()).hexdigest()}
    entry = root / files[0]["path"]
    return identity, entry, files[0]["sha256"]


def _spec(tmp_path: Path, *, target="1" * 64, execution="2" * 64):
    identity, entry, entry_sha = _inventory(tmp_path)
    body = {"schema": model_prep.SPEC_SCHEMA,
        "target_revision_digest": target, "recipe_execution_digest": execution,
        "entry_path": str(entry), "entry_sha256": entry_sha,
        "inventory_identity": identity}
    return model_prep.ScheduledModelPreparation.from_dict(
        {**body, "preparation_digest": wl._digest(body)})


class ClaimLifecycle:
    def __init__(self, claim):
        self.claim = ob._freeze(ob._plain(claim))
        self.calls = 0

    def describe_active_observation_claim(self, **kwargs):
        del kwargs
        self.calls += 1
        return self.claim


def _native_service(tmp_path: Path, spec, *, claim=None):
    claim = claim or {"schema": wl.ACTIVE_OBSERVATION_CLAIM_SCHEMA,
        "grant_id": "grant", "grant_generation": 1, "container_id": "container",
        "held_claim": {"logical_cpus": [0], "gpu_devices": []},
        "active_claim_ref": "claim:model-preparation"}
    claim = {**claim, "claim_digest": wl._digest(claim)}
    instance = object.__new__(service.NativeParentEvidenceService)
    instance.scientific_adapters = scientific.ParentScientificWitnessAdapters(
        correctness=scientific.NativeT0WitnessAdapter(max_units=2))
    instance.factual_configuration = service.NativeFactualEvidenceConfiguration(
        schema=service.FACTUAL_CONFIGURATION_SCHEMA_V2,
        scientific_adapters=instance.scientific_adapters,
        model_preparations={spec.target_revision_digest: {
            spec.recipe_execution_digest: spec}})
    instance.model_preparations = instance.factual_configuration.model_preparations
    instance.lifecycle = ClaimLifecycle(claim)
    instance._native_store = mc.ArtifactStore(tmp_path / "artifacts")
    instance._model_preparation_receipts = {}
    instance.prepared = SimpleNamespace(dispatch={"proposal": {
        "target_revision_digest": spec.target_revision_digest}})
    return instance, claim


def test_closed_spec_is_detached_and_digest_bound(tmp_path):
    spec = _spec(tmp_path)
    restored = model_prep.ScheduledModelPreparation.from_dict(spec.to_dict())
    assert restored.to_dict() == spec.to_dict()
    changed = spec.to_dict()
    changed["inventory_identity"]["model_sha256"] = "3" * 64
    with pytest.raises(model_prep.ModelPreparationRefused, match="digest"):
        model_prep.ScheduledModelPreparation.from_dict(changed)
    assert restored.inventory_identity["model_sha256"] != "3" * 64


def test_factual_configuration_v2_binds_target_and_detaches_nested_input(tmp_path):
    (tmp_path / "first").mkdir()
    (tmp_path / "second").mkdir()
    first = _spec(tmp_path / "first", target="1" * 64, execution="2" * 64)
    second = _spec(tmp_path / "second", target="3" * 64, execution="2" * 64)
    adapters = scientific.ParentScientificWitnessAdapters(
        correctness=scientific.NativeT0WitnessAdapter(max_units=2))
    supplied = {first.target_revision_digest: {first.recipe_execution_digest: first.to_dict()},
                second.target_revision_digest: {second.recipe_execution_digest: second.to_dict()}}
    configured = service.NativeFactualEvidenceConfiguration(
        schema=service.FACTUAL_CONFIGURATION_SCHEMA_V2,
        scientific_adapters=adapters, model_preparations=supplied)
    supplied[first.target_revision_digest][first.recipe_execution_digest]["entry_sha256"] = "f" * 64
    assert configured.preparation(first.target_revision_digest,
                                  first.recipe_execution_digest).entry_sha256 == first.entry_sha256
    assert configured.preparation(second.target_revision_digest,
                                  second.recipe_execution_digest).entry_path == second.entry_path
    assert configured.preparation("4" * 64, first.recipe_execution_digest) is None
    legacy = service.NativeFactualEvidenceConfiguration(
        scientific_adapters=adapters,
        model_preparations={first.recipe_execution_digest: first})
    assert legacy.preparation(first.target_revision_digest,
                              first.recipe_execution_digest) == first
    assert legacy.preparation(second.target_revision_digest,
                              first.recipe_execution_digest) is None


def test_full_six_shard_hash_finishes_under_same_live_claim_before_measure(tmp_path):
    spec = _spec(tmp_path)
    parent, claim = _native_service(tmp_path, spec)
    recipe = SimpleNamespace(execution_digest=spec.recipe_execution_digest,
        model=SimpleNamespace(path=spec.entry_path, sha256=spec.entry_sha256,
                              to_dict=lambda: {"path": spec.entry_path,
                                               "sha256": spec.entry_sha256}))
    unit = SimpleNamespace(unit_id="unit-1", process_id="process-1")
    fence = SimpleNamespace(valid_until=10_000_000.0)
    try:
        parent._before_observation_binding(
            start=object(), unit=unit, fence=fence, recipe=recipe, claim=claim)
        binding_ref = parent._model_preparation_receipts[recipe.execution_digest]
        binding = parent._native_store.read(binding_ref.locator, binding_ref.sha256)
        assert binding["preparation"] == spec.to_dict()
        receipt = mc.StoredArtifact(**binding["original_model_receipt"])
        body = parent._native_store.read(receipt.locator, receipt.sha256)
        assert len(body["manifest"]["files"]) == 6
        assert body["entry_sha256"] == spec.entry_sha256
        assert body["inventory_sha256"] == spec.inventory_identity["model_sha256"]
        assert parent.lifecycle.calls == 4
        # A second selected arm using the same model is metadata-only reuse and
        # still revalidates the live claim before reopening the original receipt.
        parent._before_observation_binding(
            start=object(), unit=unit, fence=fence, recipe=recipe, claim=claim)
        assert parent.lifecycle.calls == 6
    finally:
        parent._native_store.close()


@pytest.mark.parametrize("field", ["target", "execution", "path", "sha"])
def test_selected_target_recipe_and_entry_must_match_before_binding(tmp_path, field):
    spec = _spec(tmp_path)
    parent, claim = _native_service(tmp_path, spec)
    recipe = SimpleNamespace(execution_digest=spec.recipe_execution_digest,
        model=SimpleNamespace(path=spec.entry_path, sha256=spec.entry_sha256,
                              to_dict=lambda: {"path": spec.entry_path,
                                               "sha256": spec.entry_sha256}))
    if field == "target":
        parent.prepared.dispatch["proposal"]["target_revision_digest"] = "3" * 64
    elif field == "execution":
        recipe.execution_digest = "3" * 64
    elif field == "path":
        recipe.model.path = str(tmp_path / "other.gguf")
    else:
        recipe.model.sha256 = "3" * 64
    try:
        with pytest.raises(Exception, match="lacks scheduled|differs"):
            parent._before_observation_binding(start=object(),
                unit=SimpleNamespace(unit_id="u", process_id="p"),
                fence=SimpleNamespace(valid_until=10_000_000.0),
                recipe=recipe, claim=claim)
        assert not parent._model_preparation_receipts
    finally:
        parent._native_store.close()


def test_changed_live_claim_during_full_hash_refuses_without_receipt(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    parent, claim = _native_service(tmp_path, spec)
    original = scientific.tensor_capture._sha256_file

    def revoke_after_first(path):
        value = original(path)
        altered = ob._plain(parent.lifecycle.claim)
        altered["active_claim_ref"] = "claim:revoked"
        altered["claim_digest"] = wl._digest({
            key: item for key, item in altered.items() if key != "claim_digest"})
        parent.lifecycle.claim = ob._freeze(altered)
        return value

    monkeypatch.setattr(scientific.tensor_capture, "_sha256_file", revoke_after_first)
    recipe = SimpleNamespace(execution_digest=spec.recipe_execution_digest,
        model=SimpleNamespace(path=spec.entry_path, sha256=spec.entry_sha256,
                              to_dict=lambda: {"path": spec.entry_path,
                                               "sha256": spec.entry_sha256}))
    try:
        with pytest.raises(scientific.t0.ClaimNotHeld):
            parent._before_observation_binding(start=object(),
                unit=SimpleNamespace(unit_id="u", process_id="p"),
                fence=SimpleNamespace(valid_until=10_000_000.0),
                recipe=recipe, claim=claim)
        assert not parent._model_preparation_receipts
    finally:
        parent._native_store.close()


def test_blocked_hash_holds_no_lifecycle_lock_and_cannot_finish_after_claim_loss(
        tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    parent, claim = _native_service(tmp_path, spec)
    entered, release = threading.Event(), threading.Event()
    original = scientific.tensor_capture._sha256_file

    def blocked(path):
        entered.set()
        assert release.wait(2)
        return original(path)

    monkeypatch.setattr(scientific.tensor_capture, "_sha256_file", blocked)
    recipe = SimpleNamespace(execution_digest=spec.recipe_execution_digest,
        model=SimpleNamespace(path=spec.entry_path, sha256=spec.entry_sha256,
                              to_dict=lambda: {"path": spec.entry_path,
                                               "sha256": spec.entry_sha256}))
    errors = []

    def prepare():
        try:
            parent._before_observation_binding(start=object(),
                unit=SimpleNamespace(unit_id="u", process_id="p"),
                fence=SimpleNamespace(valid_until=10_000_000.0),
                recipe=recipe, claim=claim)
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=prepare)
    worker.start()
    try:
        assert entered.wait(1)
        # This revalidation runs while full-byte verification is blocked: the
        # preparation thread owns no lifecycle/controller mutex.
        assert ob._plain(parent.lifecycle.describe_active_observation_claim()) == claim
        altered = ob._plain(parent.lifecycle.claim)
        altered["active_claim_ref"] = "claim:expired"
        altered["claim_digest"] = wl._digest({
            key: item for key, item in altered.items() if key != "claim_digest"})
        parent.lifecycle.claim = ob._freeze(altered)
        release.set()
        worker.join(2)
        assert not worker.is_alive()
        assert len(errors) == 1 and isinstance(errors[0], scientific.t0.ClaimNotHeld)
        assert not parent._model_preparation_receipts
    finally:
        release.set()
        worker.join(2)
        parent._native_store.close()


def test_claim_loss_after_artifact_write_leaves_only_diagnostic_bytes(
        tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    parent, claim = _native_service(tmp_path, spec)
    original_write = parent._native_store.write

    def expire_after_write(namespace, body):
        artifact = original_write(namespace, body)
        if namespace.startswith("parent-verified-model:"):
            altered = ob._plain(parent.lifecycle.claim)
            altered["active_claim_ref"] = "claim:expired-after-write"
            altered["claim_digest"] = wl._digest({
                key: item for key, item in altered.items() if key != "claim_digest"})
            parent.lifecycle.claim = ob._freeze(altered)
        return artifact

    monkeypatch.setattr(parent._native_store, "write", expire_after_write)
    recipe = SimpleNamespace(execution_digest=spec.recipe_execution_digest,
        model=SimpleNamespace(path=spec.entry_path, sha256=spec.entry_sha256,
                              to_dict=lambda: {"path": spec.entry_path,
                                               "sha256": spec.entry_sha256}))
    adapter = parent.scientific_adapters.correctness
    try:
        with pytest.raises(scientific.t0.ClaimNotHeld):
            parent._before_observation_binding(start=object(),
                unit=SimpleNamespace(unit_id="u", process_id="p"),
                fence=SimpleNamespace(valid_until=10_000_000.0),
                recipe=recipe, claim=claim)
        assert not adapter._models
        assert not parent._model_preparation_receipts
        # The immutable CAS write may exist, but artifact presence cannot restore
        # issuance and no observation binding/measurement authority was returned.
        assert any(parent._native_store.root.iterdir())
    finally:
        parent._native_store.close()


def test_claim_loss_during_binding_publication_withholds_preparation(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    parent, claim = _native_service(tmp_path, spec)
    original_write = parent._native_store.write

    def expire_binding(namespace, body):
        artifact = original_write(namespace, body)
        if namespace.startswith("scheduled-model-preparation:"):
            altered = ob._plain(parent.lifecycle.claim)
            altered["active_claim_ref"] = "claim:expired-binding"
            altered["claim_digest"] = wl._digest({
                key: item for key, item in altered.items() if key != "claim_digest"})
            parent.lifecycle.claim = ob._freeze(altered)
        return artifact

    monkeypatch.setattr(parent._native_store, "write", expire_binding)
    recipe = SimpleNamespace(execution_digest=spec.recipe_execution_digest,
        model=SimpleNamespace(path=spec.entry_path, sha256=spec.entry_sha256,
                              to_dict=lambda: {"path": spec.entry_path,
                                               "sha256": spec.entry_sha256}))
    try:
        with pytest.raises(Exception, match="expired before observation binding"):
            parent._before_observation_binding(start=object(),
                unit=SimpleNamespace(unit_id="u", process_id="p"),
                fence=SimpleNamespace(valid_until=10_000_000.0),
                recipe=recipe, claim=claim)
        assert not parent._model_preparation_receipts
    finally:
        parent._native_store.close()


def test_actual_selected_child_prepares_complete_model_before_measurement(
        tmp_path, monkeypatch):
    identity, entry, entry_sha = _inventory(tmp_path)
    recipe = canonical_recipe(model_path=str(entry), model_sha256=entry_sha)
    adapter = scientific.NativeT0WitnessAdapter(max_units=4)
    adapters = scientific.ParentScientificWitnessAdapters(correctness=adapter)
    services, scopes = [], []
    validate_calls = []
    replayer = replay.NativeParentReceiptReplayer()
    injected = pytest.MonkeyPatch()
    original_validator_init = nc.NativeCaptureValidator.__init__

    def validator_init(self, *args, **kwargs):
        kwargs["parent_receipt_replayer"] = replayer
        original_validator_init(self, *args, **kwargs)

    injected.setattr(nc.NativeCaptureValidator, "__init__", validator_init)
    validate_code = scientific.tensor_capture.CaptureModelIdentity.validate.__code__

    def profile(_frame, event, _arg):
        if event == "call" and _frame.f_code is validate_code:
            validate_calls.append(_frame.f_locals["self"].to_dict())

    old_profile = threading.getprofile()
    threading.setprofile(profile)

    def producer_factory(authority, prepared, lifecycle, configuration):
        target = prepared.dispatch["proposal"]["target_revision_digest"]
        preparations = {}
        for selected in (prepared.runtime_pair.anchor, prepared.runtime_pair.candidate):
            body = {"schema": model_prep.SPEC_SCHEMA,
                "target_revision_digest": target,
                "recipe_execution_digest": selected.execution_digest,
                "entry_path": selected.model.path,
                "entry_sha256": selected.model.sha256,
                "inventory_identity": identity}
            preparations[selected.execution_digest] = {
                **body, "preparation_digest": wl._digest(body)}
        registry = replay.IssuedNativeEvidenceRegistry(
            artifact_root=prepared.artifact_root,
            max_units=len(prepared.plan.expected_units))
        scope = replayer.using(registry)
        scope.__enter__()
        scopes.append(scope)
        probe_root = tmp_path / "fixture-probe"
        probe = lo.FilesystemProbe(
            proc_root=probe_root / "proc", sysfs_cpu_root=probe_root / "cpu",
            boot_id_path=probe_root / "boot", cgroup_root=probe_root / "cgroup")
        producer = service.NativeParentEvidenceService(
            authority, prepared, lifecycle, configuration, registry=registry,
            runtime_probe=probe, scientific_adapters=adapters,
            model_preparations=preparations)
        services.append(producer)
        return producer

    try:
        _run_real_controller_child_v2_capture_and_restart(
            tmp_path, monkeypatch, producer_type=producer_factory, recipe=recipe,
            scientific_adapters=adapters)
    finally:
        for scope in reversed(scopes):
            scope.__exit__(None, None, None)
        threading.setprofile(old_profile)
        injected.undo()
    assert len(services) == 1
    parent = services[0]
    assert parent.stopped
    assert set(parent._model_preparation_receipts) == {
        parent.prepared.runtime_pair.anchor.execution_digest,
        parent.prepared.runtime_pair.candidate.execution_digest}
    # The selected thread recipes differ, but their exact model inventory is
    # verified once and reused only through the original parent issuance.
    assert validate_calls == [identity]
    assert len(adapter._models) == 1
