"""Hermetic candidate transaction, recovery, immutable-object, and Git tests."""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import threading

import pytest

from .. import journal as journal_module
from . import candidate_manifest as cm
from . import candidate_transactions as transactions
from . import campaign_control as control
from . import kernel_mutation_guard
from . import measurement_capture as mc
from .test_campaign_control import _resolved
from .test_candidate_manifest import (
    _base, _batch, _loo_kwargs, _manifest, _next, _receipt, _row, _source, _state,
)


class FakeGitBackend:
    def __init__(self):
        self.refs: dict[tuple[str, str], str] = {}

    def plan(self, transaction_id, sources):
        return tuple(transactions.PreparedRef(
            item.repo_id, f"refs/autokernel/candidates/test/{transaction_id}/{item.repo_id}",
            item.commit, item.tree, item.object_format) for item in sources)

    def prepare(self, item):
        key = (item.repo_id, item.ref)
        old = self.refs.get(key)
        if old not in (None, item.commit):
            raise transactions.CandidateRecoveryRequired("prepared ref changed")
        self.refs[key] = item.commit

    def verify(self, item):
        if self.refs.get((item.repo_id, item.ref)) != item.commit:
            raise transactions.CandidateRecoveryRequired(
                f"restore exact prepared ref {item.repo_id}:{item.ref}")


def _manager(tmp_path, *, backend=None, fault_hook=None, verifiers=None):
    controller = control.CampaignController(_resolved(), tmp_path / "service")
    controller.__enter__()
    manager = transactions.CandidateTransactions(
        controller, git_backend=backend or FakeGitBackend(), fault_hook=fault_hook,
        verifiers=verifiers)
    return controller, manager


def _artifact_path(controller, kind, value):
    name, _encoded, _plain = mc.ArtifactStore._identity(
        f"candidate:{kind}", value)
    return controller.store / transactions.OBJECT_DIR / name


def test_initialize_is_journaled_before_pointer_and_replays_after_restart(tmp_path):
    backend = FakeGitBackend()
    controller, manager = _manager(tmp_path, backend=backend)
    base = _base()
    try:
        result = manager.initialize(request_id="init", state=_state(base), manifest=base)
        assert result["integration_tip"] == base.manifest_digest
        pointer = json.loads((controller.store / transactions.POINTER_FILE).read_text())
        assert pointer["integration_tip"] == base.manifest_digest
        assert [entry.payload["phase"] for entry in controller._journal.read_all()
                if entry.kind == journal_module.KIND_CANDIDATE_TRANSACTION] == [
                    "INTENT", "PREPARED", "COMMITTED"]
    finally:
        controller.close()
    with control.CampaignController(_resolved(), tmp_path / "service") as replayed:
        snapshot = transactions.CandidateTransactions(
            replayed, git_backend=backend).inspect()
        assert snapshot["state"]["integration_tip"] == base.manifest_digest
        assert snapshot["completed_transactions"] == ["init"]


def test_missing_or_corrupt_projection_is_ignored_and_explicit_retry_repairs(tmp_path):
    controller, manager = _manager(tmp_path)
    base = _base()
    try:
        first = manager.initialize(request_id="init", state=_state(base), manifest=base)
        pointer = controller.store / transactions.POINTER_FILE
        pointer.write_text("{corrupt", encoding="utf-8")
        assert manager.inspect()["state_digest"] == first["state_digest"]
        assert manager.initialize(request_id="init", state=_state(base), manifest=base) == first
        assert json.loads(pointer.read_text())["state_digest"] == first["state_digest"]
        pointer.unlink()
        assert manager.inspect()["initialized"] is True
    finally:
        controller.close()


def test_crash_after_intent_autonomously_finishes_its_unprepared_ref(tmp_path):
    backend = FakeGitBackend()
    crash = {"enabled": True}

    def fault(point):
        if crash["enabled"] and point == "after_intent":
            raise RuntimeError("crash after intent")

    controller, manager = _manager(tmp_path, backend=backend, fault_hook=fault)
    base = _base()
    try:
        with pytest.raises(RuntimeError, match="after intent"):
            manager.initialize(request_id="init", state=_state(base), manifest=base)
        crash["enabled"] = False
        result = manager.initialize(request_id="init", state=_state(base), manifest=base)
        assert result["integration_tip"] == base.manifest_digest
    finally:
        controller.close()


def test_source_ref_landed_without_completion_resumes_exact_same_request(tmp_path):
    backend = FakeGitBackend()
    fired = {"value": False}

    def fault(point):
        if point.startswith("after_ref:") and not fired["value"]:
            fired["value"] = True
            raise RuntimeError("crash after source ref")

    controller, manager = _manager(tmp_path, backend=backend, fault_hook=fault)
    base = _base()
    try:
        with pytest.raises(RuntimeError, match="source ref"):
            manager.initialize(request_id="init", state=_state(base), manifest=base)
        result = manager.initialize(request_id="init", state=_state(base), manifest=base)
        assert result["integration_tip"] == base.manifest_digest
    finally:
        controller.close()


def test_missing_dependency_after_durable_prepared_receipt_requires_repair(tmp_path):
    backend = FakeGitBackend()
    fired = False

    def fault(point):
        nonlocal fired
        if point == "after_prepared" and not fired:
            fired = True
            raise RuntimeError("crash after prepared receipt")

    controller, manager = _manager(tmp_path, backend=backend, fault_hook=fault)
    base = _base()
    try:
        with pytest.raises(RuntimeError, match="prepared receipt"):
            manager.initialize(request_id="init", state=_state(base), manifest=base)
        backend.refs.clear()
        with pytest.raises(transactions.CandidateRecoveryRequired,
                           match="restore exact prepared ref"):
            manager.initialize(request_id="init", state=_state(base), manifest=base)
        assert not (controller.store / transactions.POINTER_FILE).exists()
    finally:
        controller.close()


def test_half_prepared_cross_repo_intent_resumes_exact_remaining_refs(tmp_path):
    sources = []
    for repo_id, digit in (("a", "1"), ("b", "3")):
        source = _source(commit=digit * 40, tree=(str(int(digit) + 1)) * 40)
        source["repo_id"] = repo_id
        source["path"] = f"/repo/{repo_id}"
        sources.append(source)
    base = cm.CandidateManifest.from_dict(_manifest(sources=sources))
    backend = FakeGitBackend()

    def fault(point):
        if point == "after_ref:a":
            raise RuntimeError("crash after first repository")

    controller, manager = _manager(tmp_path, backend=backend, fault_hook=fault)
    try:
        with pytest.raises(RuntimeError, match="first repository"):
            manager.initialize(request_id="init", state=_state(base), manifest=base)
        assert len(backend.refs) == 1
        assert not (controller.store / transactions.POINTER_FILE).exists()
        result = manager.initialize(request_id="init", state=_state(base), manifest=base)
        assert result["integration_tip"] == base.manifest_digest
        assert len(backend.refs) == 2
    finally:
        controller.close()


def test_crash_after_commit_replays_and_repairs_missing_publication(tmp_path):
    backend = FakeGitBackend()
    fired = {"value": False}

    def fault(point):
        if point == "after_commit" and not fired["value"]:
            fired["value"] = True
            raise RuntimeError("crash after completion")

    controller, manager = _manager(tmp_path, backend=backend, fault_hook=fault)
    base = _base()
    try:
        with pytest.raises(RuntimeError, match="completion"):
            manager.initialize(request_id="init", state=_state(base), manifest=base)
        assert not (controller.store / transactions.POINTER_FILE).exists()
        result = manager.initialize(request_id="init", state=_state(base), manifest=base)
        assert result["integration_tip"] == base.manifest_digest
        assert (controller.store / transactions.POINTER_FILE).is_file()
    finally:
        controller.close()


def test_real_projection_write_failure_preserves_committed_result_for_retry(
        tmp_path, monkeypatch):
    backend = FakeGitBackend()
    controller, manager = _manager(tmp_path, backend=backend)
    base = _base()
    original = transactions.status.write_json
    monkeypatch.setattr(
        transactions.status, "write_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("projection fault")))
    try:
        with pytest.raises(OSError, match="projection fault"):
            manager.initialize(request_id="init", state=_state(base), manifest=base)
        assert any(entry.payload.get("phase") == "COMMITTED"
                   for entry in controller._journal.read_all()
                   if entry.kind == journal_module.KIND_CANDIDATE_TRANSACTION)
        monkeypatch.setattr(transactions.status, "write_json", original)
        result = manager.initialize(request_id="init", state=_state(base), manifest=base)
        assert result["integration_tip"] == base.manifest_digest
    finally:
        controller.close()


def test_torn_candidate_artifact_stage_publishes_nothing_and_retry_recovers(
        tmp_path, monkeypatch):
    backend = FakeGitBackend()
    armed = False
    real_write = os.write

    def torn_write(descriptor, value):
        nonlocal armed
        if armed:
            armed = False
            real_write(descriptor, value[:1])
            raise OSError("candidate artifact torn write")
        return real_write(descriptor, value)

    def fault(point):
        nonlocal armed
        if point == "after_intent":
            armed = True

    controller, manager = _manager(tmp_path, backend=backend, fault_hook=fault)
    base = _base()
    monkeypatch.setattr(os, "write", torn_write)
    try:
        with pytest.raises(transactions.CandidateRecoveryRequired, match="publication failed"):
            manager.initialize(request_id="init", state=_state(base), manifest=base)
        artifact_root = controller.store / transactions.OBJECT_DIR
        assert not list(artifact_root.iterdir())
        result = manager.initialize(request_id="init", state=_state(base), manifest=base)
        assert result["integration_tip"] == base.manifest_digest
    finally:
        controller.close()


def test_candidate_artifact_retry_recovers_post_link_interruption(
        tmp_path, monkeypatch):
    backend = FakeGitBackend()
    armed = False
    failed = False
    real_unlink = os.unlink

    def interrupted_unlink(path, *args, **kwargs):
        nonlocal failed
        if armed and not failed and str(path).endswith(".stage"):
            failed = True
            raise OSError("candidate artifact post-link interruption")
        return real_unlink(path, *args, **kwargs)

    def fault(point):
        nonlocal armed
        if point == "after_intent":
            armed = True

    controller, manager = _manager(tmp_path, backend=backend, fault_hook=fault)
    base = _base()
    monkeypatch.setattr(os, "unlink", interrupted_unlink)
    try:
        with pytest.raises(transactions.CandidateRecoveryRequired, match="publication failed"):
            manager.initialize(request_id="init", state=_state(base), manifest=base)
        result = manager.initialize(request_id="init", state=_state(base), manifest=base)
        assert result["integration_tip"] == base.manifest_digest
        assert not list((controller.store / transactions.OBJECT_DIR).glob("*.stage"))
    finally:
        controller.close()


@pytest.mark.parametrize("point, committed", [
    ("before_intent", False),
    ("before_projection", True),
])
def test_crash_boundary_before_intent_or_projection_recovers(
        tmp_path, point, committed):
    backend = FakeGitBackend()
    fired = {"value": False}

    def fault(current):
        if current == point and not fired["value"]:
            fired["value"] = True
            raise RuntimeError(f"crash at {point}")

    controller, manager = _manager(tmp_path, backend=backend, fault_hook=fault)
    base = _base()
    try:
        with pytest.raises(RuntimeError, match=point):
            manager.initialize(request_id="init", state=_state(base), manifest=base)
        events = [entry for entry in controller._journal.read_all()
                  if entry.kind == journal_module.KIND_CANDIDATE_TRANSACTION]
        assert any(entry.payload["phase"] == "COMMITTED" for entry in events) is committed
        result = manager.initialize(request_id="init", state=_state(base), manifest=base)
        assert result["integration_tip"] == base.manifest_digest
    finally:
        controller.close()


@pytest.mark.parametrize("point", [
    "before_objects", "after_objects", "before_ref:research", "after_prepared",
    "before_commit", "after_projection",
])
def test_each_prepare_and_publication_boundary_is_idempotently_recoverable(
        tmp_path, point):
    backend = FakeGitBackend()
    base = _base()
    for item in backend.plan("init", base.sources):
        backend.prepare(item)
    fired = {"value": False}

    def fault(current):
        if current == point and not fired["value"]:
            fired["value"] = True
            raise RuntimeError(f"crash at {point}")

    controller, manager = _manager(tmp_path, backend=backend, fault_hook=fault)
    try:
        with pytest.raises(RuntimeError, match=point):
            manager.initialize(request_id="init", state=_state(base), manifest=base)
        result = manager.initialize(request_id="init", state=_state(base), manifest=base)
        assert result["integration_tip"] == base.manifest_digest
    finally:
        controller.close()


def test_duplicate_request_is_idempotent_and_conflict_refuses(tmp_path):
    controller, manager = _manager(tmp_path)
    base = _base()
    try:
        first = manager.initialize(request_id="same", state=_state(base), manifest=base)
        assert manager.initialize(request_id="same", state=_state(base), manifest=base) == first
        changed = replace(_state(base), keeps_since_gate=1).validated()
        with pytest.raises(transactions.CandidateTransactionError, match="different semantics"):
            manager.initialize(request_id="same", state=changed, manifest=base)
    finally:
        controller.close()


def test_retry_of_old_transaction_never_rolls_projection_back(tmp_path):
    controller, manager = _manager(tmp_path)
    base, candidate = _base(), _next(_base(), 0)
    try:
        initial = manager.initialize(request_id="init", state=_state(base), manifest=base)
        manager.integrate(request_id="integrate", previous=base, candidate=candidate)
        assert manager.initialize(
            request_id="init", state=_state(base), manifest=base) == initial
        pointer = json.loads(
            (controller.store / transactions.POINTER_FILE).read_text(encoding="utf-8"))
        assert pointer["integration_tip"] == candidate.manifest_digest
    finally:
        controller.close()


def test_initialization_cannot_import_accumulated_or_validated_state(tmp_path):
    controller, manager = _manager(tmp_path)
    base = _base()
    try:
        for invalid in (
                replace(_state(base), validated_candidate=base.manifest_digest).validated(),
                replace(_state(base), keeps_since_gate=1).validated()):
            with pytest.raises(cm.TransitionError, match="empty unvalidated"):
                manager.initialize(request_id="init", state=invalid, manifest=base)
        assert not [entry for entry in controller._journal.read_all()
                    if entry.kind == journal_module.KIND_CANDIDATE_TRANSACTION]
    finally:
        controller.close()


def test_integration_parent_cas_and_frozen_production_ref(tmp_path):
    controller, manager = _manager(tmp_path)
    base, candidate = _base(), _next(_base(), 0)
    try:
        manager.initialize(request_id="init", state=_state(base), manifest=base)
        result = manager.integrate(request_id="integrate-1", previous=base,
                                   candidate=candidate)
        assert result["integration_tip"] == candidate.manifest_digest
        stale = _next(base, 1)
        with pytest.raises(cm.TransitionError, match="CAS"):
            manager.integrate(request_id="stale", previous=base, candidate=stale)
        raw = _next(candidate, 1).to_dict()
        raw["production_ref_digest"] = "0" * 64
        moved_production = cm.CandidateManifest.from_dict(raw)
        with pytest.raises(cm.TransitionError, match="production_ref"):
            manager.integrate(request_id="production", previous=candidate,
                              candidate=moved_production)
    finally:
        controller.close()


def test_batch_rows_completion_preserve_debt_and_optional_pending(tmp_path):
    controller, manager = _manager(tmp_path)
    base, candidate = _base(), _next(_base(), 0)
    required = _row("gpu", candidate, base)
    optional = _row("seed", candidate, base, required=False, row_kind="seed")
    batch, row_set = _batch(candidate, base, [required, optional])
    batch = replace(batch, accounted_keeps_since_gate=1).validated()
    try:
        manager.initialize(request_id="init", state=_state(base), manifest=base)
        manager.integrate(request_id="integrate", previous=base, candidate=candidate)
        manager.start_batch(request_id="start", batch=batch, row_set=row_set,
                            candidate=candidate, comparator=base)
        passed = cm.ValidationRowState("gpu", "passed", None,
                                       _receipt(batch, "gpu"))
        manager.record_row(request_id="row-gpu", batch_id=batch.batch_id,
                           row_set=row_set, row_state=passed)
        completed = replace(batch, rows=(passed, batch.rows[1])).validated()
        result = manager.complete_batch(request_id="complete", batch=completed)
        assert result["validation_debt"] == [batch.batch_id]
        snapshot = manager.inspect()["state"]
        optional_state = snapshot["completed_batches"][0]["rows"][1]
        assert optional_state["status"] == "pending"
    finally:
        controller.close()


def test_frozen_older_batch_completes_after_newer_integration(tmp_path):
    controller, manager = _manager(tmp_path)
    base, first = _base(), _next(_base(), 0)
    second = _next(first, 1)
    row = _row("gpu", first, base)
    batch, row_set = _batch(first, base, [row])
    batch = replace(batch, accounted_keeps_since_gate=1).validated()
    passed = cm.ValidationRowState("gpu", "passed", None, _receipt(batch, "gpu"))
    completed = replace(batch, rows=(passed,)).validated()
    try:
        manager.initialize(request_id="init", state=_state(base), manifest=base)
        manager.integrate(request_id="first", previous=base, candidate=first)
        manager.start_batch(request_id="start", batch=batch, row_set=row_set,
                            candidate=first, comparator=base)
        manager.integrate(request_id="second", previous=first, candidate=second)
        manager.record_row(request_id="row", batch_id=batch.batch_id,
                           row_set=row_set, row_state=passed)
        manager.complete_batch(request_id="complete", batch=completed)
        state = manager.inspect()["state"]
        assert state["integration_tip"] == second.manifest_digest
        assert state["completed_batches"][0]["candidate_manifest_digest"] == (
            first.manifest_digest)
        assert state["keeps_since_gate"] == 1
    finally:
        controller.close()


def test_cpu_pass_gpu_missing_cannot_complete_or_advance(tmp_path):
    controller, manager = _manager(tmp_path)
    raw = _base().to_dict()
    raw["targets"][0]["backend"] = "both"
    base = cm.CandidateManifest.from_dict(raw)
    candidate = _next(base, 0)
    cpu = _row("cpu", candidate, base, backend="cpu")
    gpu = _row("gpu", candidate, base, backend="gpu")
    batch, row_set = _batch(candidate, base, [cpu, gpu])
    batch = replace(batch, accounted_keeps_since_gate=1).validated()
    try:
        manager.initialize(request_id="init", state=_state(base), manifest=base)
        manager.integrate(request_id="integrate", previous=base, candidate=candidate)
        manager.start_batch(request_id="start", batch=batch, row_set=row_set,
                            candidate=candidate, comparator=base)
        passed = cm.ValidationRowState("cpu", "passed", None,
                                       _receipt(batch, "cpu"))
        manager.record_row(request_id="cpu", batch_id=batch.batch_id,
                           row_set=row_set, row_state=passed)
        partial = replace(batch, rows=(passed, batch.rows[1])).validated()
        with pytest.raises(cm.TransitionError, match="pending/running"):
            manager.complete_batch(request_id="complete", batch=partial)
    finally:
        controller.close()


def test_validated_pointer_requires_registered_verifier_and_loo(tmp_path):
    backend = FakeGitBackend()
    controller, manager = _manager(tmp_path, backend=backend)
    base, candidate = _base(), _next(_base(), 0)
    row = _row("gpu", candidate, base)
    batch, row_set = _batch(candidate, base, [row])
    batch = replace(batch, accounted_keeps_since_gate=1).validated()
    passed = cm.ValidationRowState("gpu", "passed", None, _receipt(batch, "gpu"))
    completed = replace(batch, rows=(passed,)).validated()
    try:
        manager.initialize(request_id="init", state=_state(base), manifest=base)
        manager.integrate(request_id="integrate", previous=base, candidate=candidate)
        manager.start_batch(request_id="start", batch=batch, row_set=row_set,
                            candidate=candidate, comparator=base)
        manager.record_row(request_id="row", batch_id=batch.batch_id,
                           row_set=row_set, row_state=passed)
        manager.complete_batch(request_id="complete", batch=completed)
        loo = _loo_kwargs(candidate, row_set)
        with pytest.raises(cm.TrustedVerificationRequired):
            manager.advance_validated(
                request_id="advance", verifier_id="trusted", batch=completed,
                row_set=row_set, candidate=candidate, comparator=base,
                loo_plans=loo["loo_plans"], loo_results=loo["loo_results"])
        verifier_calls = []

        def verifier(*_args):
            verifier_calls.append("row")
            return True

        def loo_verifier(*_args):
            verifier_calls.append("loo")
            return True

        trusted = transactions.CandidateTransactions(
            controller, git_backend=backend,
            verifiers={"trusted": (verifier, loo_verifier)})
        with pytest.raises(cm.TransitionError, match="LOO results are missing"):
            trusted.advance_validated(
                request_id="advance-missing-loo", verifier_id="trusted",
                batch=completed, row_set=row_set, candidate=candidate,
                comparator=base)
        verifier_calls.clear()
        result = trusted.advance_validated(
            request_id="advance", verifier_id="trusted", batch=completed,
            row_set=row_set, candidate=candidate, comparator=base,
            loo_plans=loo["loo_plans"], loo_results=loo["loo_results"])
        assert result["validated_candidate"] == candidate.manifest_digest
        assert verifier_calls == ["row", "loo"]
        trusted.inspect()
        assert verifier_calls == ["row", "loo"]
    finally:
        controller.close()
    with control.CampaignController(_resolved(), tmp_path / "service") as replayed:
        historical = transactions.CandidateTransactions(replayed)
        snapshot = historical.inspect()
        assert snapshot["state"]["validated_candidate"] == candidate.manifest_digest
        assert snapshot["historical_validation_receipt"]["decision"] == (
            "trusted_verifiers_accepted")
        assert snapshot["current_evidence_eligibility"] == (
            "requires_live_registered_verification")
        with pytest.raises(cm.TrustedVerificationRequired):
            historical.advance_validated(
                request_id="new-advance", verifier_id="trusted", batch=completed,
                row_set=row_set, candidate=candidate, comparator=base,
                loo_plans=loo["loo_plans"], loo_results=loo["loo_results"])


def test_historical_reconstruction_does_not_claim_current_artifact_availability(tmp_path):
    controller, manager = _manager(tmp_path)
    base = _base()
    try:
        manager.initialize(request_id="init", state=_state(base), manifest=base)
        artifact = _artifact_path(controller, "manifest", base.to_dict())
        artifact.unlink()
        assert manager.inspect()["state_digest"] is not None
        with pytest.raises(transactions.CandidateRecoveryRequired,
                           match="restore immutable candidate object"):
            manager.initialize(request_id="init", state=_state(base), manifest=base)
        assert not artifact.exists()
    finally:
        controller.close()

    second, manager = _manager(tmp_path / "hardlink")
    try:
        manager.initialize(request_id="init", state=_state(base), manifest=base)
        artifact = _artifact_path(second, "manifest", base.to_dict())
        peer = second.store / "peer-copy"
        os.link(artifact, peer)
        assert manager.inspect()["state_digest"] is not None
        with pytest.raises(transactions.CandidateRecoveryRequired,
                           match="restore immutable candidate object"):
            manager.initialize(request_id="init", state=_state(base), manifest=base)
    finally:
        second.close()


def test_symlinked_object_directory_and_tampered_content_refuse(tmp_path):
    backend = FakeGitBackend()
    controller, manager = _manager(tmp_path, backend=backend)
    base = _base()
    outside = tmp_path / "outside"
    outside.mkdir()
    (controller.store / transactions.OBJECT_DIR).symlink_to(
        outside, target_is_directory=True)
    try:
        with pytest.raises(transactions.CandidateRecoveryRequired,
                           match="artifact publication failed"):
            manager.initialize(request_id="init", state=_state(base), manifest=base)
    finally:
        controller.close()

    other, manager = _manager(tmp_path / "tampered")
    try:
        manager.initialize(request_id="init", state=_state(base), manifest=base)
        artifact = _artifact_path(other, "manifest", base.to_dict())
        artifact.write_text("{}\n", encoding="utf-8")
        assert manager.inspect()["state_digest"] is not None
    finally:
        other.close()


def test_historical_ref_absence_does_not_erase_state_but_exact_retry_checks_it(tmp_path):
    backend = FakeGitBackend()
    controller, manager = _manager(tmp_path, backend=backend)
    base = _base()
    try:
        manager.initialize(request_id="init", state=_state(base), manifest=base)
        key = next(iter(backend.refs))
        backend.refs.pop(key)
        assert manager.inspect()["state_digest"] is not None
        with pytest.raises(transactions.CandidateRecoveryRequired,
                           match="restore exact prepared ref"):
            manager.initialize(request_id="init", state=_state(base), manifest=base)
        backend.refs[key] = "f" * 40
        with pytest.raises(transactions.CandidateRecoveryRequired,
                           match="restore exact prepared ref"):
            manager.initialize(request_id="init", state=_state(base), manifest=base)
    finally:
        controller.close()


def test_uncertain_journal_append_poisons_controller_until_replay(tmp_path, monkeypatch):
    backend = FakeGitBackend()
    controller, manager = _manager(tmp_path, backend=backend)
    base, candidate = _base(), _next(_base(), 0)
    manager.initialize(request_id="init", state=_state(base), manifest=base)
    assert controller._journal is not None
    real_append = controller._journal.append

    def append_then_fault(*args, **kwargs):
        real_append(*args, **kwargs)
        raise OSError("uncertain append")

    monkeypatch.setattr(controller._journal, "append", append_then_fault)
    try:
        with pytest.raises(OSError, match="uncertain"):
            manager.integrate(request_id="integrate", previous=base, candidate=candidate)
        with pytest.raises(control.ControlRefused, match="poisoned"):
            manager.inspect()
    finally:
        controller.close()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _real_manifest(repo: Path) -> cm.CandidateManifest:
    base = _base()
    source = cm.SourceIdentity("research", str(repo), "sha1", _git(repo, "rev-parse", "HEAD"),
                               _git(repo, "rev-parse", "HEAD^{tree}"))
    build = replace(base.builds[0], source_set_digest=cm.source_set_digest((source,)))
    target = replace(base.targets[0], build_execution_digest=build.execution_digest)
    return replace(base, sources=(source,), builds=(build,), targets=(target,)).validated()


def test_real_git_backend_creates_only_owned_ref_and_never_moves_branch(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"],
                   check=True)
    (repo / "source.txt").write_text("source\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "source.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"], check=True)
    head = _git(repo, "rev-parse", "HEAD")
    base = _real_manifest(repo)
    controller = control.CampaignController(_resolved(), tmp_path / "service")
    controller.__enter__()
    backend = transactions.GitCandidateBackend(
        {"research": repo}, campaign_id=controller.resolved.campaign_id)
    try:
        transactions.CandidateTransactions(controller, git_backend=backend).initialize(
            request_id="init", state=_state(base), manifest=base)
        assert _git(repo, "rev-parse", "HEAD") == head
        refs = _git(repo, "for-each-ref", "--format=%(refname)",
                    transactions.OWNED_REF_PREFIX).splitlines()
        assert len(refs) == 1 and refs[0].startswith(transactions.OWNED_REF_PREFIX + "/")
        assert _git(repo, "rev-parse", refs[0]) == head
    finally:
        controller.close()


def test_real_git_backend_ref_cas_does_not_overwrite_conflicting_object(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email",
                    "test@example.invalid"], check=True)
    (repo / "source.txt").write_text("one\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "source.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "one"], check=True)
    base = _real_manifest(repo)
    backend = transactions.GitCandidateBackend({"research": repo}, campaign_id="campaign")
    planned = backend.plan("request", base.sources)[0]
    (repo / "source.txt").write_text("two\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "commit", "-qam", "two"], check=True)
    other = _git(repo, "rev-parse", "HEAD")
    subprocess.run(["git", "-C", str(repo), "update-ref", planned.ref, other], check=True)
    with pytest.raises(transactions.CandidateRecoveryRequired, match="changed"):
        backend.prepare(planned)
    assert _git(repo, "rev-parse", planned.ref) == other


@pytest.mark.parametrize("production_branch", [
    "production-consolidated-v999", "production-speech-v999",
])
def test_git_backend_refuses_unplanned_ref_and_production_branch(
        tmp_path, production_branch):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"],
                   check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email",
                    "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "--allow-empty", "-qm", "base"],
                   check=True)
    head = _git(repo, "rev-parse", "HEAD")
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    backend = transactions.GitCandidateBackend({"research": repo},
                                                campaign_id="campaign")
    arbitrary = transactions.PreparedRef(
        "research", "refs/heads/not-owned", head, tree, "sha1")
    with pytest.raises(transactions.CandidateTransactionError, match="not derived"):
        backend.prepare(arbitrary)
    assert subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "-q", "--verify", arbitrary.ref],
        capture_output=True, check=False).returncode != 0
    subprocess.run(["git", "-C", str(repo), "branch", "-m",
                    production_branch], check=True)
    with pytest.raises(transactions.CandidateTransactionError,
                       match="frozen production branch"):
        transactions.GitCandidateBackend({"research": repo}, campaign_id="campaign")


def test_frozen_kernel_roots_are_complete_and_exact_root_is_refused(
        tmp_path, monkeypatch):
    assert kernel_mutation_guard.FROZEN_PRODUCTION_ROOTS == frozenset({
        Path("/mnt/raid0/llm/llama.cpp"), Path("/mnt/raid0/llm/whisper.cpp"),
        Path("/mnt/raid0/llm/qwentts.cpp"),
    })
    repo = tmp_path / "synthetic-frozen"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c",
                    "user.email=test@example.invalid", "commit", "--allow-empty", "-qm",
                    "base"], check=True)
    monkeypatch.setattr(
        kernel_mutation_guard, "FROZEN_PRODUCTION_ROOTS", frozenset({repo.resolve()}))
    with pytest.raises(transactions.CandidateTransactionError,
                       match="canonical frozen production"):
        transactions.GitCandidateBackend({"research": repo}, campaign_id="campaign")


def test_experimental_linked_worktree_may_share_production_object_database(tmp_path):
    common = tmp_path / "common"
    common.mkdir()
    subprocess.run(["git", "init", "-q", str(common)], check=True)
    subprocess.run(["git", "-C", str(common), "config", "user.name", "Test"],
                   check=True)
    subprocess.run(["git", "-C", str(common), "config", "user.email",
                    "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(common), "commit", "--allow-empty", "-qm", "base"],
                   check=True)
    subprocess.run(["git", "-C", str(common), "branch", "-m",
                    "production-speech-v999"], check=True)
    linked = tmp_path / "experimental"
    subprocess.run(["git", "-C", str(common), "worktree", "add", "-qb",
                    "candidate-work", str(linked)], check=True)
    backend = transactions.GitCandidateBackend({"research": linked},
                                                campaign_id="campaign")
    assert backend.plan("request", _real_manifest(linked).sources)


def test_git_backend_rechecks_branch_guard_before_prepared_ref_cas(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"],
                   check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email",
                    "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "--allow-empty", "-qm", "base"],
                   check=True)
    manifest = _real_manifest(repo)
    backend = transactions.GitCandidateBackend({"research": repo},
                                                campaign_id="campaign")
    planned = backend.plan("request", manifest.sources)[0]
    subprocess.run(["git", "-C", str(repo), "branch", "-m",
                    "production-speech-v999"], check=True)

    with pytest.raises(transactions.CandidateRecoveryRequired,
                       match="became frozen"):
        backend.prepare(planned)
    assert subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "-q", "--verify", planned.ref],
        capture_output=True, check=False).returncode != 0


def test_git_backend_refuses_redirect_environment_and_replaced_git_directory(
        tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"],
                   check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email",
                    "test@example.invalid"], check=True)
    (repo / "source.txt").write_text("source\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "source.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"], check=True)
    manifest = _real_manifest(repo)
    backend = transactions.GitCandidateBackend({"research": repo},
                                                campaign_id="campaign")
    planned = backend.plan("request", manifest.sources)[0]
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "redirect"))
    with pytest.raises(transactions.CandidateRecoveryRequired,
                       match="redirection appeared"):
        backend.verify(planned)
    monkeypatch.delenv("GIT_DIR")
    (repo / ".git").rename(repo / ".git-retained")
    (repo / ".git").mkdir()
    with pytest.raises(transactions.CandidateRecoveryRequired,
                       match="Git identity changed"):
        backend.plan("request", manifest.sources)


def test_concurrent_same_parent_integrations_serialize_and_one_loses_cas(tmp_path):
    controller, manager = _manager(tmp_path)
    base = _base()
    candidates = (_next(base, 0), _next(base, 1))
    outcomes = []
    gate = threading.Barrier(3)

    def integrate(index):
        gate.wait()
        try:
            result = manager.integrate(request_id=f"integrate-{index}", previous=base,
                                       candidate=candidates[index])
        except cm.TransitionError as exc:
            outcomes.append(("refused", str(exc)))
        else:
            outcomes.append(("committed", result["integration_tip"]))

    try:
        manager.initialize(request_id="init", state=_state(base), manifest=base)
        workers = [threading.Thread(target=integrate, args=(index,))
                   for index in range(2)]
        for worker in workers:
            worker.start()
        gate.wait()
        for worker in workers:
            worker.join(2)
            assert not worker.is_alive()
        assert [kind for kind, _ in outcomes].count("committed") == 1
        assert [kind for kind, _ in outcomes].count("refused") == 1
        assert "CAS" in next(value for kind, value in outcomes if kind == "refused")
    finally:
        controller.close()


def test_two_managers_share_incremental_frontier_and_second_sees_first_cas(tmp_path):
    controller, first = _manager(tmp_path)
    second = transactions.CandidateTransactions(
        controller, git_backend=first.git_backend)
    base = _base()
    winner, stale = _next(base, 0), _next(base, 1)
    try:
        first.initialize(request_id="init", state=_state(base), manifest=base)
        first.integrate(request_id="winner", previous=base, candidate=winner)
        with pytest.raises(cm.TransitionError, match="CAS"):
            second.integrate(request_id="stale", previous=base, candidate=stale)
        assert second.inspect()["state"]["integration_tip"] == winner.manifest_digest
    finally:
        controller.close()


def test_candidate_capability_is_thread_bound_and_poison_is_rechecked(
        tmp_path, monkeypatch):
    controller, manager = _manager(tmp_path)
    base = _base()
    manager.initialize(request_id="init", state=_state(base), manifest=base)
    intent = next(entry.payload for entry in controller._journal.read_all()
                  if entry.kind == journal_module.KIND_CANDIDATE_TRANSACTION
                  and entry.payload["phase"] == "INTENT")
    foreign_errors = []

    def cross_thread(context):
        def append():
            try:
                context.append(
                    phase="INTENT", transaction_id="foreign",
                    operation=intent["operation"],
                    payload_digest=intent["payload_digest"], data=intent["data"])
            except Exception as exc:
                foreign_errors.append(exc)
        thread = threading.Thread(target=append)
        thread.start()
        thread.join(2)
        assert not thread.is_alive()

    try:
        controller.candidate_transaction(cross_thread)
        assert foreign_errors and isinstance(foreign_errors[0], control.ControlRefused)
        assert not any(entry.record_id == "foreign"
                       for entry in controller._journal.read_all())

        assert controller._journal is not None
        real_append = controller._journal.append
        calls = 0

        def uncertain_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                real_append(*args, **kwargs)
                raise OSError("uncertain append")
            return real_append(*args, **kwargs)

        monkeypatch.setattr(controller._journal, "append", uncertain_once)
        second_errors = []

        def catch_and_retry(context):
            for request_id in ("uncertain", "must-not-append"):
                try:
                    context.append(
                        phase="INTENT", transaction_id=request_id,
                        operation=intent["operation"],
                        payload_digest=intent["payload_digest"], data=intent["data"])
                except Exception as exc:
                    second_errors.append(exc)

        controller.candidate_transaction(catch_and_retry)
        assert isinstance(second_errors[0], OSError)
        assert isinstance(second_errors[1], control.ControlRefused)
        assert calls == 1
    finally:
        controller.close()


def test_controller_close_inside_candidate_callback_revokes_capability(tmp_path):
    controller, _manager_instance = _manager(tmp_path)
    refusal = []

    def close_then_append(context):
        controller.close()
        try:
            context.append(
                phase="INTENT", transaction_id="after-close", operation="init",
                payload_digest="0" * 64, data={})
        except Exception as exc:
            refusal.append(exc)

    controller.candidate_transaction(close_then_append)
    assert refusal and isinstance(refusal[0], control.ControlRefused)
    assert controller._journal is None


def test_candidate_projection_is_incremental_and_avoids_history_rescans(
        tmp_path, monkeypatch):
    backend = FakeGitBackend()
    verify_calls = 0
    real_verify = backend.verify

    def counted_verify(item):
        nonlocal verify_calls
        verify_calls += 1
        return real_verify(item)

    backend.verify = counted_verify
    controller, manager = _manager(tmp_path, backend=backend)
    base = _base()
    try:
        manager.initialize(request_id="init", state=_state(base), manifest=base)
        assert controller._journal is not None
        monkeypatch.setattr(
            controller._journal, "read_all",
            lambda: (_ for _ in ()).throw(AssertionError("whole WAL reread")))
        previous = base
        for index in range(6):
            candidate = _next(previous, index)
            manager.integrate(request_id=f"keep-{index}", previous=previous,
                              candidate=candidate)
            previous = candidate
        replay_calls = 0
        replay_apply = transactions.CandidateTransactions._replay_apply

        def counted_replay(self, *args, **kwargs):
            nonlocal replay_calls
            replay_calls += 1
            return replay_apply(self, *args, **kwargs)

        monkeypatch.setattr(
            transactions.CandidateTransactions, "_replay_apply", counted_replay)
        for _ in range(10):
            fresh = transactions.CandidateTransactions(
                controller, git_backend=backend)
            assert fresh.inspect()["state"]["integration_tip"] == previous.manifest_digest
        assert verify_calls == 14
        assert replay_calls == 1
    finally:
        controller.close()


def test_hot_operations_do_not_deepcopy_old_event_payloads(tmp_path, monkeypatch):
    controller, manager = _manager(tmp_path)
    base, candidate = _base(), _next(_base(), 0)
    try:
        initial = manager.initialize(request_id="old-init", state=_state(base), manifest=base)
        manager.inspect()  # advance the maintained projection to the append frontier
        original = control.copy.deepcopy
        copied_old_entries = 0

        def tracked(value, *args, **kwargs):
            nonlocal copied_old_entries
            if (isinstance(value, journal_module.JournalEntry)
                    and value.record_id == "old-init"):
                copied_old_entries += 1
            return original(value, *args, **kwargs)

        monkeypatch.setattr(control.copy, "deepcopy", tracked)
        assert manager.initialize(
            request_id="old-init", state=_state(base), manifest=base) == initial
        manager.integrate(request_id="new-keep", previous=base, candidate=candidate)
        for _ in range(4):
            transactions.CandidateTransactions(
                controller, git_backend=manager.git_backend).inspect()
        assert copied_old_entries == 0
    finally:
        controller.close()


def test_public_context_cannot_replace_projection_behind_journal(tmp_path):
    controller, manager = _manager(tmp_path)
    base = _base()
    try:
        manager.initialize(request_id="init", state=_state(base), manifest=base)
        assert manager.inspect()["initialized"] is True

        def attempt(context):
            forged = replace(context.projection_cache(), state=None)
            with pytest.raises(control.ControlRefused, match="trusted replayer"):
                context.update_projection_cache(forged)

        controller.candidate_transaction(attempt)
        assert manager.inspect()["initialized"] is True
    finally:
        controller.close()


def test_native_candidate_event_validator_returns_violations_not_type_errors():
    for payload in ([], {"schema": transactions.POINTER_SCHEMA},
                    {"schema": journal_module.CANDIDATE_TRANSACTION_SCHEMA,
                     "phase": [], "data": {}}):
        assert journal_module._validate_native_payload(
            journal_module.KIND_CANDIDATE_TRANSACTION, payload)


def test_native_candidate_validator_handles_malformed_nested_json(tmp_path):
    controller, manager = _manager(tmp_path)
    base = _base()
    try:
        manager.initialize(request_id="init", state=_state(base), manifest=base)
        rows = [json.loads(json.dumps(entry.payload))
                for entry in controller._journal.read_all()
                if entry.kind == journal_module.KIND_CANDIDATE_TRANSACTION]
    finally:
        controller.close()
    intent = next(row for row in rows if row["phase"] == "INTENT")
    committed = next(row for row in rows if row["phase"] == "COMMITTED")
    cases = []
    bad_repo = json.loads(json.dumps(intent))
    bad_repo["data"]["prepared_refs"][0]["repo_id"] = []
    cases.append(bad_repo)
    missing_operation_field = json.loads(json.dumps(intent))
    missing_operation_field["data"]["operation_payload"].pop("state")
    cases.append(missing_operation_field)
    bad_debt = json.loads(json.dumps(committed))
    bad_debt["data"]["result"]["validation_debt"] = [{}]
    cases.append(bad_debt)
    for payload in cases:
        assert journal_module._validate_native_payload(
            journal_module.KIND_CANDIDATE_TRANSACTION, payload)


def test_replay_refuses_structurally_valid_semantically_invalid_transition(tmp_path):
    controller, manager = _manager(tmp_path)
    base = _base()
    try:
        initial = manager.initialize(request_id="init", state=_state(base), manifest=base)
        operation_payload = {"batch": {}}
        objects = [{"kind": "batch", "digest": transactions._digest({})}]
        payload_digest = transactions._intent_digest(
            "complete_batch", initial["state_digest"], operation_payload, objects, [])
        state_row = manager.inspect()["state"]

        def append_invalid(context):
            context.append(
                phase="INTENT", transaction_id="invalid-completion",
                operation="complete_batch", payload_digest=payload_digest,
                data={"expected_state_digest": initial["state_digest"],
                      "operation_payload": operation_payload,
                      "prepared_objects": objects, "prepared_refs": []})
            context.append(
                phase="PREPARED", transaction_id="invalid-completion",
                operation="complete_batch", payload_digest=payload_digest,
                data={"prepared_objects": objects, "prepared_refs": []})
            receipt = manager._transition_receipt(
                "complete_batch", payload_digest, initial["state_digest"],
                state_row, operation_payload)
            context.append(
                phase="COMMITTED", transaction_id="invalid-completion",
                operation="complete_batch", payload_digest=payload_digest,
                data={"state": state_row, "state_digest": initial["state_digest"],
                      "result": initial, "transition_receipt": receipt})

        controller.candidate_transaction(append_invalid)
        with pytest.raises(transactions.CandidateRecoveryRequired,
                           match="replay payload is invalid"):
            manager.inspect()
    finally:
        controller.close()
