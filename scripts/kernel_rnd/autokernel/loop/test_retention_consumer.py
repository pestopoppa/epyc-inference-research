"""Fault and race tests for the held native retention consumer."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tempfile

import pytest

from .. import storage
from .. import journal as journal_module
from . import candidate_manifest as cm
from . import candidate_transactions as ct
from . import retention
from . import retention_consumer as rc
from . import resolved_recipe as rr
from .test_candidate_transactions import FakeGitBackend, _manager


NOW = datetime(2026, 9, 9, tzinfo=timezone.utc)
H = "a" * 64


@pytest.fixture
def native_tmp_path():
    """Storage intentionally refuses /tmp, so use a disposable owned repo peer."""
    path = Path(tempfile.mkdtemp(prefix="_retention_consumer_", dir=Path(__file__).parent))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _source(path: Path) -> cm.SourceIdentity:
    return cm.SourceIdentity("research", str(path), "sha1", "1" * 40, "2" * 40)


def _manifest(path: Path) -> cm.CandidateManifest:
    source = _source(path)
    build = cm.BuildIdentity(
        "build", "3" * 64, cm.source_set_digest((source,)),
        rr.ArtifactDigest("executable", str(path / "bin"), "4" * 64),
        (rr.ArtifactDigest("dso", str(path / "lib.so"), "5" * 64),))
    target = cm.CandidateTarget("target", "6" * 64, "cpu", build.execution_digest,
                                "7" * 64, "8" * 64, "9" * 64, None,
                                "b" * 64, True)
    return cm.CandidateManifest("candidate", "c" * 64, "d" * 64, (source,),
                                (build,), (target,), (), "e" * 64).validated()


def _state(manifest: cm.CandidateManifest) -> cm.CandidateState:
    return cm.CandidateState(manifest.production_ref_digest, manifest.manifest_digest,
                             None, 0, 0, 0, (), (), (), (), (), ()).validated()


def _node(path: Path, *, size: int | None = None) -> retention.ArtifactNode:
    digest = storage.hash_tree_manifest(path)
    return retention.ArtifactNode(
        "old-build", "build_dir", (), str(path), "expirable",
        storage.measure_usage(path).bytes_on_disk if size is None else size,
        retention.ExpiryDescriptor(
            "campaign", digest, "hash_and_provenance_only", "rejected_candidate_build_tree",
            "candidate rejected", "retention-v1", "maintenance",
            {"candidate_id": "old", "candidate_status": "rejected",
             "champion_status": "none", "evaluation_events_journaled": True}))


def _view(tmp_path: Path, *, generation: int = 1, extra_roots=(), uncertain=(),
          branch="experiment") -> tuple[rc.NativeRetentionView, Path]:
    old = tmp_path / "owned" / "old"
    old.mkdir(parents=True)
    (old / "artifact").write_text("obsolete", encoding="utf-8")
    manifest = _manifest(tmp_path / "repo")
    nodes = (
        retention.ArtifactNode("production", "candidate_artifact", (), None,
                               "permanent_in_repo", None, None),
        retention.ArtifactNode("rollback", "candidate_artifact", (), None,
                               "permanent_in_repo", None, None),
        retention.ArtifactNode("tip", "candidate_artifact", (), None,
                               "permanent_in_repo", None, None),
        _node(old),
    )
    roots = rc.NativeRoots(("production",), ("rollback",), launch_intents=extra_roots)
    identity = rc.NativeArtifactIdentity("old-build", str(old), nodes[-1].expiry.sha256,
                                         branch, _source(tmp_path / "repo"))
    view = rc.NativeRetentionView("native", generation, _state(manifest),
                                  (rc.ManifestArtifacts(manifest, ("tip",)),), nodes,
                                  roots, (identity,), uncertain)
    return view, old


class _Journal:
    def __init__(self, fail_state=None):
        self.rows = []
        self.fail_state = fail_state

    def append(self, kind, payload, *, campaign_id=None):
        assert kind == "TOMBSTONE" and campaign_id == "campaign"
        if payload["reclamation_state"] == self.fail_state:
            raise RuntimeError(f"fault at {self.fail_state}")
        self.rows.append(dict(payload))
        return f"ev-{len(self.rows)}"

    def native(self):
        result = []
        for index, row in enumerate(self.rows, 1):
            record = dict(row)
            record.pop("storage_class", None)
            record.pop("path", None)
            result.append(rc.NativeTombstone(f"ev-{index}", record))
        return tuple(result)


class _Owner(rc.MaintenanceOwner):
    def __init__(self, view, policy, journal, *, on_hold=None):
        self.view, self.policy, self.journal = view, policy, journal
        self.on_hold, self.calls = on_hold, 0

    def held(self, operation):
        self.calls += 1
        if self.on_hold:
            self.on_hold(self)
        return operation(rc.MaintenanceLease(self.view, self.policy, self.journal,
                                             "held-owner-receipt", self.journal.native()))


def _policy(tmp_path):
    return storage.StoragePolicy(1, owned_roots=(str(tmp_path / "owned"),))


def test_native_candidate_closure_dry_preview_then_held_expiry(native_tmp_path):
    tmp_path = native_tmp_path
    view, old = _view(tmp_path)
    job, previews = rc.prepare(view, _policy(tmp_path), now=NOW)
    assert job is not None and [item.state for item in previews] == ["DRY_RUN"]
    assert old.exists()
    journal = _Journal()
    result = rc.execute(job, owner=_Owner(view, _policy(tmp_path), journal), now=NOW)
    assert [item.state for item in result.outcomes] == ["RECLAIMED"]
    assert result.owner_receipt == "held-owner-receipt"
    assert not old.exists()
    assert [row["reclamation_state"] for row in journal.rows] == ["intent", "reclaimed"]


def test_dry_policy_is_limited_to_selected_batch(native_tmp_path, monkeypatch):
    view, _ = _view(native_tmp_path)
    calls = []
    real = storage.plan_expiry
    monkeypatch.setattr(storage, "plan_expiry",
                        lambda artifact, policy, **kwargs: (
                            calls.append(artifact.path) or real(artifact, policy, **kwargs)))
    job, previews = rc.prepare(view, _policy(native_tmp_path), limit=1, now=NOW)
    assert job is not None and len(previews) == len(calls) == 1


def test_actual_candidate_transactions_and_native_journal_reopen_recovery(native_tmp_path):
    view, old = _view(native_tmp_path)
    manifest = view.manifests[0].manifest
    backend = FakeGitBackend()
    (native_tmp_path / "service").mkdir(mode=0o700)
    (native_tmp_path / "service").chmod(0o700)  # parent worktree carries setgid
    controller, actual = _manager(native_tmp_path, backend=backend)
    try:
        expirable = view.nodes[-1]
        expirable = replace(
            expirable, expiry=replace(expirable.expiry,
                                       campaign_id=controller.resolved.campaign_id))
        view = replace(view, nodes=view.nodes[:-1] + (expirable,))
        actual.initialize(request_id="init", state=view.candidate_state, manifest=manifest)
        inspected = rc.inspect_candidate_state(actual)
        assert inspected == view.candidate_state
        object_root = controller.store / ct.OBJECT_DIR
        assert object_root.is_dir() and any(object_root.iterdir())
        job, previews = rc.prepare(view, _policy(native_tmp_path), now=NOW)
        assert job is not None
        intent_id = storage.JournalTombstoneSink(controller._journal).append(
            previews[0].tombstone)
        assert intent_id
        shutil.rmtree(old)
    finally:
        controller.close()

    reopened, replayed = _manager(native_tmp_path, backend=backend)
    try:
        replay_state = rc.inspect_candidate_state(replayed)
        entries = tuple(reopened._journal.read_all())
        native = []
        for entry in entries:
            if entry.kind != journal_module.KIND_TOMBSTONE:
                continue
            record = dict(entry.payload)
            record.pop("storage_class", None)
            record.pop("path", None)
            native.append(rc.NativeTombstone(entry.event_id, record))

        class NativeOwner(rc.MaintenanceOwner):
            def held(self, operation):
                current = replace(view, candidate_state=replay_state)
                return operation(rc.MaintenanceLease(
                    current, _policy(native_tmp_path), reopened._journal,
                    "native-held-receipt", tuple(native)))

        result = rc.execute(job, owner=NativeOwner(), now=NOW)
        assert result.outcomes[0].deleted is False
        states = [entry.payload["reclamation_state"]
                  for entry in reopened._journal.read_all()
                  if entry.kind == journal_module.KIND_TOMBSTONE]
        assert states == ["intent", "reclaimed"]
    finally:
        reopened.close()

    with pytest.raises(TypeError, match="CandidateTransactions"):
        rc.inspect_candidate_state({"state": view.candidate_state.to_dict()})


def test_default_unavailable_and_caller_json_has_no_authority(native_tmp_path):
    tmp_path = native_tmp_path
    view, _ = _view(tmp_path)
    job, _ = rc.prepare(view, _policy(tmp_path), now=NOW)
    with pytest.raises(rc.RetentionConsumerRefused, match="owner is unavailable"):
        rc.execute(job, now=NOW)
    with pytest.raises(TypeError, match="not caller JSON"):
        rc.collect_native_snapshot(view.__dict__ if hasattr(view, "__dict__") else {})
    with pytest.raises(TypeError, match="not caller JSON"):
        rc.execute(job.plan.to_dict(), owner=None)


def test_generation_race_and_second_owner_are_refused_before_bytes(native_tmp_path):
    tmp_path = native_tmp_path
    view, old = _view(tmp_path)
    job, _ = rc.prepare(view, _policy(tmp_path), now=NOW)
    newer = replace(view, generation=2)
    owner = _Owner(newer, _policy(tmp_path), _Journal())
    with pytest.raises(retention.RetentionRefused, match="no longer matches"):
        rc.execute(job, owner=owner, now=NOW)
    assert owner.calls == 1 and old.exists()


def test_sequential_callers_reuse_original_completion_without_duplicate_delete(native_tmp_path):
    view, old = _view(native_tmp_path)
    job, _ = rc.prepare(view, _policy(native_tmp_path), now=NOW)
    journal = _Journal()
    first = rc.execute(job, owner=_Owner(view, _policy(native_tmp_path), journal), now=NOW)
    second = rc.execute(job, owner=_Owner(view, _policy(native_tmp_path), journal), now=NOW)
    assert first.outcomes[0].deleted and not second.outcomes[0].deleted
    assert not old.exists()
    assert [row["reclamation_state"] for row in journal.rows] == ["intent", "reclaimed"]


def test_held_operation_builds_one_bounded_tombstone_index(native_tmp_path, monkeypatch):
    view, _ = _view(native_tmp_path)
    job, _ = rc.prepare(view, _policy(native_tmp_path), now=NOW)
    calls = []
    original = rc._tombstone_index
    monkeypatch.setattr(rc, "_tombstone_index",
                        lambda rows: (calls.append(len(rows)) or original(rows)))
    rc.execute(job, owner=_Owner(view, _policy(native_tmp_path), _Journal()), now=NOW)
    assert calls == [0]


def test_active_intent_and_ambiguous_root_withhold_expiry(native_tmp_path):
    tmp_path = native_tmp_path
    view, _ = _view(tmp_path, extra_roots=("old-build",))
    job, previews = rc.prepare(view, _policy(tmp_path), now=NOW)
    assert job is None and previews == ()
    uncertain = replace(view, uncertain_scopes=("worker dependencies unavailable",))
    with pytest.raises(retention.RetentionRefused, match="closure is unknown"):
        rc.prepare(uncertain, _policy(tmp_path), now=NOW)


def test_active_batch_projects_candidate_comparator_and_loo_roots(native_tmp_path):
    view, _ = _view(native_tmp_path)
    digest = view.manifests[0].manifest.manifest_digest
    batch = cm.ValidationBatch(
        "batch", digest, digest, "7" * 64, None, digest, 0, 0, ("keep",), ("row",),
        (cm.ValidationRowState("row", "pending", None, None),)).validated()
    state = replace(view.candidate_state, active_batches=(batch,)).validated()
    snapshot = rc.collect_native_snapshot(replace(view, candidate_state=state))
    kinds = {root.root_kind for root in snapshot.roots}
    assert {"pending_validation", "pending_comparator", "pending_loo"}.issubset(kinds)


def test_shared_ancestor_overlap_retains_candidate(native_tmp_path):
    tmp_path = native_tmp_path
    view, old = _view(tmp_path)
    child = old / "required-dso"
    retained = retention.ArtifactNode("shared-dso", "shared_dso_dir", (), str(child),
                                      "permanent_large", None, None)
    identity = rc.NativeArtifactIdentity("shared-dso", str(child), "f" * 64,
                                         "experiment", _source(tmp_path / "repo"))
    view = replace(view, nodes=view.nodes + (retained,), identities=view.identities + (identity,))
    job, previews = rc.prepare(view, _policy(tmp_path), now=NOW)
    assert job is None and previews == ()


def test_replaced_artifact_is_refused_by_content_identity(native_tmp_path):
    tmp_path = native_tmp_path
    view, old = _view(tmp_path)
    job, _ = rc.prepare(view, _policy(tmp_path), now=NOW)
    (old / "artifact").write_text("different", encoding="utf-8")
    # Preserve allocation size to prove the native hash check, not size alone, catches it.
    assert storage.measure_usage(old).bytes_on_disk == view.nodes[-1].declared_size_bytes
    with pytest.raises(rc.RetentionConsumerRefused, match="content changed"):
        rc.execute(job, owner=_Owner(view, _policy(tmp_path), _Journal()), now=NOW)
    assert old.exists()


def test_replay_requires_complete_descriptor_and_same_policy(native_tmp_path):
    view, old = _view(native_tmp_path)
    policy = _policy(native_tmp_path)
    job, previews = rc.prepare(view, policy, now=NOW)
    conflict = dict(previews[0].tombstone)
    conflict.update(reason="different reason", storage_class="expirable",
                    path=conflict["artifact_path"])
    journal = _Journal()
    journal.rows.append(conflict)
    with pytest.raises(rc.RetentionConsumerRefused, match="current descriptor"):
        rc.execute(job, owner=_Owner(view, policy, journal), now=NOW)
    assert old.exists()
    changed_policy = storage.StoragePolicy(2, owned_roots=policy.owned_roots)
    with pytest.raises(rc.RetentionConsumerRefused, match="policy differs"):
        rc.execute(job, owner=_Owner(view, changed_policy, _Journal()), now=NOW)
    assert old.exists()


@pytest.mark.parametrize("branch", ["production-consolidated-v10", "production-speech-v1"])
def test_protected_production_and_speech_branches_are_never_expirable(native_tmp_path, branch):
    tmp_path = native_tmp_path
    view, old = _view(tmp_path, branch=branch)
    with pytest.raises(rc.RetentionConsumerRefused, match="protected branch"):
        rc.prepare(view, _policy(tmp_path), now=NOW)
    assert old.exists()


def test_intent_append_fault_leaves_bytes(native_tmp_path):
    tmp_path = native_tmp_path
    view, old = _view(tmp_path)
    job, _ = rc.prepare(view, _policy(tmp_path), now=NOW)
    with pytest.raises(RuntimeError, match="intent"):
        rc.execute(job, owner=_Owner(view, _policy(tmp_path), _Journal("intent")), now=NOW)
    assert old.exists()


def test_cleanup_fault_records_failed_and_preserves_bytes(native_tmp_path, monkeypatch):
    tmp_path = native_tmp_path
    view, old = _view(tmp_path)
    job, _ = rc.prepare(view, _policy(tmp_path), now=NOW)
    journal = _Journal()
    monkeypatch.setattr(storage.shutil, "rmtree",
                        lambda path: (_ for _ in ()).throw(OSError("cleanup fault")))
    with pytest.raises(OSError, match="cleanup fault"):
        rc.execute(job, owner=_Owner(view, _policy(tmp_path), journal), now=NOW)
    assert old.exists()
    assert [row["reclamation_state"] for row in journal.rows] == ["intent", "failed"]


def test_completion_fault_restarts_original_intent_without_second_deletion(native_tmp_path):
    tmp_path = native_tmp_path
    view, old = _view(tmp_path)
    job, _ = rc.prepare(view, _policy(tmp_path), now=NOW)
    journal = _Journal("reclaimed")
    with pytest.raises(RuntimeError, match="reclaimed"):
        rc.execute(job, owner=_Owner(view, _policy(tmp_path), journal), now=NOW)
    assert not old.exists()
    assert [row["reclamation_state"] for row in journal.rows] == ["intent"]
    journal.fail_state = None
    result = rc.execute(job, owner=_Owner(view, _policy(tmp_path), journal), now=NOW)
    assert not result.outcomes[0].deleted
    assert [row["reclamation_state"] for row in journal.rows] == ["intent", "reclaimed"]
    assert result.outcomes[0].measured_size_bytes == journal.rows[0]["size_bytes"]
