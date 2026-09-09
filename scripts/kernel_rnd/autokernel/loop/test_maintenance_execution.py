"""Disposable integration tests for bounded maintenance execution ownership."""
from __future__ import annotations

from dataclasses import replace
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time

import pytest

from .. import journal as journal_module, schemas, storage
from . import candidate_transactions as ct
from . import maintenance_execution as me
from . import retention_consumer as rc
from .test_campaign_control import _command
from .test_candidate_transactions import FakeGitBackend, _manager
from .test_retention_consumer import NOW, _policy, _view


@pytest.fixture
def native_root():
    path = Path(tempfile.mkdtemp(prefix="_maintenance_execution_", dir=Path(__file__).parent))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _native_tombstones(journal):
    result = []
    for entry in journal.read_all():
        if entry.kind != journal_module.KIND_TOMBSTONE:
            continue
        record = dict(entry.payload)
        record.pop("storage_class", None)
        record.pop("path", None)
        result.append(rc.NativeTombstone(entry.event_id, record))
    return tuple(result)


def _actual_token(controller, job):
    return me.ExclusionToken(
        "maintenance-native", controller.resolved.campaign_id,
        controller.config_digest, controller.config_generation,
        controller._maintenance_supervisor_id(), controller.supervisor_incarnation,
        job.plan.snapshot_id, job.plan.snapshot_generation, job.plan.snapshot_digest,
        job.plan.plan_digest, job.policy_digest, job.selected_artifact_ids,
        NOW.isoformat())


def _activate_actual_controller(controller, job):
    token = _actual_token(controller, job)
    with controller._mutex:
        controller._maintenance_append_locked(
            me.make_event("INTENT", token, occurred_at=NOW.isoformat()))
    return token


class _ControllerJournalBackend:
    def __init__(self, controller):
        self.controller = controller

    def revalidate(self, token, hold):
        return self.controller.maintenance_revalidate(token, hold)

    def append_tombstone(self, token, kind, payload, campaign_id):
        return self.controller.maintenance_append_tombstone(
            token, kind, payload, campaign_id)


class _ActualControllerBackend(me.ControllerMaintenanceBackend):
    """Fixture-only activation; fresh public catalog admission remains refused."""

    def __init__(self, controller, view, policy, *, stale_view=False):
        self.controller, self.view, self.policy = controller, view, policy
        self.stale_view = stale_view
        self.token = None

    def _lease(self):
        view = replace(self.view, generation=self.view.generation + 1) \
            if self.stale_view else self.view
        return rc.MaintenanceLease(
            view, self.policy, self.controller._journal,
            "actual-controller-execute", _native_tombstones(self.controller._journal))

    def admit(self, job):
        self.token = _activate_actual_controller(self.controller, job)
        return me.Admission(self.token, self._lease())

    def revalidate(self, token, hold):
        self.controller.maintenance_revalidate(token, hold)
        return self._lease()

    def append_tombstone(self, token, kind, payload, campaign_id):
        return self.controller.maintenance_append_tombstone(
            token, kind, payload, campaign_id)

    def io_complete(self, token, cost):
        self.controller.maintenance_io_complete(token, cost)

    def complete(self, token, accounting):
        self.controller.maintenance_complete(token, accounting)

    def abort(self, token, reason, receipt, hold=None):
        self.controller.maintenance_abort(token, reason, receipt, hold)

    def unresolved(self, token, reason):
        self.controller.maintenance_unresolved(token, reason)


class Provider(me.MaintenanceHoldProvider):
    def __init__(self):
        self.generation = 1
        self.revoked = False
        self.on_refresh = None
        self.calls = []
        self.fail_aborted_finish = False
        self.return_false_aborted = False
        self.aborted_reply = None
        self.return_false_complete = False
        self.acquire_error = False
        self.acquire_reply = None
        self.no_hold = False
        self.deadline = NOW.timestamp() + 86400
        self.finished_hold = None

    def _receipt(self, token):
        return me.HoldReceipt("fixture-provider", "hold-1", me._request_digest(token),
                              self.generation, 1, self.deadline, True,
                              self.revoked)

    def acquire(self, token):
        self.calls.append("acquire")
        if self.acquire_error:
            raise RuntimeError("provider reply lost after possible hold creation")
        if self.acquire_reply == "missing":
            return None
        if self.acquire_reply == "misbound":
            receipt = self._receipt(token)
            return me.HoldReceipt(
                receipt.provider_id, receipt.hold_id, "f" * 64,
                receipt.provider_generation, receipt.accounting_epoch,
                receipt.deadline, receipt.current, receipt.revoked)
        if self.no_hold:
            return me.NoHoldReceipt(
                "fixture-provider", me._request_digest(token), self.generation,
                NOW.isoformat(), "capacity refused")
        return self._receipt(token)

    def refresh(self, hold, token):
        self.calls.append("refresh")
        if self.on_refresh:
            self.on_refresh()
            self.on_refresh = None
        return self._receipt(token)

    def finish(self, hold, token, cost, disposition):
        self.calls.append(f"finish:{disposition}")
        self.finished_hold = hold
        if disposition == "aborted" and self.fail_aborted_finish:
            raise RuntimeError("provider settlement fault")
        if disposition == "aborted" and self.return_false_aborted:
            return False
        if disposition == "aborted" and self.aborted_reply == "missing":
            return None
        if disposition == "aborted" and self.aborted_reply == "misbound":
            return me.AccountingReceipt(
                hold.receipt_digest, "f" * 64, cost, disposition)
        if disposition == "complete" and self.return_false_complete:
            return False
        return me.AccountingReceipt(hold.receipt_digest, token.token_digest,
                                    cost, disposition)


class Backend(me.ControllerMaintenanceBackend):
    """Test implementation of the proposed short controller methods."""

    def __init__(self, controller, view, policy):
        self.controller, self.view, self.policy = controller, view, policy
        self.token = None
        self.state = "idle"
        self.events = []
        self.fail_completion_append = False
        self.fail_complete = False
        self.fail_unresolved = False
        self.recover = False

    def _lease(self):
        return rc.MaintenanceLease(
            self.view, self.policy, self.controller._journal,
            f"held:{self.token.token_digest}",
            _native_tombstones(self.controller._journal))

    def admit(self, job):
        with self.controller._mutex:
            self.controller._require_active_locked()
            if self.token is not None and not (self.recover and self.state == "unresolved"):
                raise me.MaintenanceExecutionRefused("maintenance exclusion is already owned")
            if self.token is None:
                self.token = me.ExclusionToken(
                    "maintenance-1", self.controller.resolved.campaign_id,
                    self.controller.config_digest, self.controller.config_generation,
                    "fixture-supervisor", self.controller.supervisor_incarnation,
                    job.plan.snapshot_id, job.plan.snapshot_generation,
                    job.plan.snapshot_digest, job.plan.plan_digest, job.policy_digest,
                    job.selected_artifact_ids, NOW.isoformat())
            self.state = "intent"
            self.events.append("INTENT")
            return me.Admission(self.token, self._lease())

    def revalidate(self, token, hold):
        with self.controller._mutex:
            self.controller._require_active_locked()
            if token != self.token or self.state not in {"intent", "provider_held", "io_complete"}:
                raise me.MaintenanceExecutionRefused("maintenance exclusion changed")
            if self.view.generation != token.snapshot_generation:
                raise me.MaintenanceExecutionRefused("native generation changed")
            if rc.collect_native_snapshot(self.view).snapshot_digest != token.snapshot_digest:
                raise me.MaintenanceExecutionRefused("native dependency/root snapshot changed")
            me._validate_hold(hold, token)
            if hold.deadline <= NOW.timestamp():
                raise me.MaintenanceExecutionRefused("maintenance hold deadline expired")
            self.state = "provider_held"
            self.events.append("MUTATION_REVALIDATED")
            return self._lease()

    def append_tombstone(self, token, kind, payload, campaign_id):
        with self.controller._mutex:
            if token != self.token:
                raise me.MaintenanceExecutionRefused("maintenance exclusion changed")
            if payload["reclamation_state"] == "reclaimed" and self.fail_completion_append:
                raise RuntimeError("completion append fault")
            return self.controller._journal.append(kind, payload, campaign_id=campaign_id)

    def io_complete(self, token, cost):
        with self.controller._mutex:
            assert token == self.token
            self.state = "io_complete"
            self.events.append(("IO_COMPLETE", cost))

    def complete(self, token, accounting):
        with self.controller._mutex:
            assert token == self.token
            if self.fail_complete:
                raise RuntimeError("controller completion append uncertain")
            self.events.append(("COMPLETED", accounting.receipt_digest))
            self.state, self.token = "complete", None

    def abort(self, token, reason, receipt, hold=None):
        with self.controller._mutex:
            if token == self.token:
                self.events.append(("ABORTED", reason, receipt, hold))
                self.state, self.token = "aborted", None

    def unresolved(self, token, reason):
        with self.controller._mutex:
            if token == self.token:
                if self.fail_unresolved:
                    raise RuntimeError("unresolved append fault")
                self.events.append(("UNRESOLVED", reason))
                self.state = "unresolved"


def _create_runtime(native_root):
    view, old = _view(native_root)
    service = native_root / "service"
    service.mkdir(mode=0o700)
    service.chmod(0o700)
    git_backend = FakeGitBackend()
    controller, candidates = _manager(native_root, backend=git_backend)
    manifest = view.manifests[0].manifest
    expirable = view.nodes[-1]
    expirable = replace(
        expirable, expiry=replace(expirable.expiry,
                                   campaign_id=controller.resolved.campaign_id))
    view = replace(view, nodes=view.nodes[:-1] + (expirable,))
    candidates.initialize(request_id="init", state=view.candidate_state, manifest=manifest)
    assert rc.inspect_candidate_state(candidates) == view.candidate_state
    assert any((controller.store / ct.OBJECT_DIR).iterdir())
    job, _ = rc.prepare(view, _policy(native_root), now=NOW)
    return controller, candidates, git_backend, view, old, job


@pytest.fixture
def runtime(native_root):
    values = _create_runtime(native_root)
    try:
        yield values
    finally:
        controller = values[0]
        if controller._entered and not controller._maintenance_state.owned:
            controller.close()


def _plain_view(view):
    return {
        "snapshot_id": view.snapshot_id, "generation": view.generation,
        "candidate_state": view.candidate_state.to_dict(),
        "manifests": [{"manifest": item.manifest.to_dict(),
                       "artifact_ids": list(item.artifact_ids)}
                      for item in view.manifests],
        "nodes": [item.to_dict() for item in view.nodes],
        "roots": {name: list(getattr(view.roots, name))
                  for name in view.roots.__dataclass_fields__},
        "identities": [{"artifact_id": item.artifact_id, "path": item.path,
                        "sha256": item.sha256, "branch": item.branch,
                        "source": item.source.to_dict()}
                       for item in view.identities],
        "uncertain_scopes": list(view.uncertain_scopes),
    }


def _native_view(value):
    return rc.NativeRetentionView(
        value["snapshot_id"], value["generation"],
        rc.cm.CandidateState.from_dict(value["candidate_state"]),
        tuple(rc.ManifestArtifacts(
            rc.cm.CandidateManifest.from_dict(item["manifest"]),
            tuple(item["artifact_ids"])) for item in value["manifests"]),
        tuple(rc.retention.ArtifactNode.from_dict(item) for item in value["nodes"]),
        rc.NativeRoots(**{key: tuple(items) for key, items in value["roots"].items()}),
        tuple(rc.NativeArtifactIdentity(
            item["artifact_id"], item["path"], item["sha256"], item["branch"],
            rc.cm.SourceIdentity.from_dict(item["source"]))
              for item in value["identities"]),
        tuple(value["uncertain_scopes"]))


def _crash_owner(root: str, scenario: str, connection) -> None:
    """Create durable owned state, report descriptors, then die without close."""
    controller, _candidates, git_backend, view, old, job = _create_runtime(Path(root))
    token = _actual_token(controller, job)
    hold = None
    if scenario in {"acquisition_reply_lost", "false_abort_receipt"}:
        _activate_actual_controller(controller, job)
        backend = _ActualControllerBackend(
            controller, view, _policy(old.parents[1]),
            stale_view=scenario == "false_abort_receipt")
        backend.token = token
        backend.admit = lambda _job: me.Admission(token, backend._lease())
        provider = Provider()
        provider.acquire_error = scenario == "acquisition_reply_lost"
        provider.return_false_aborted = scenario == "false_abort_receipt"
        try:
            me.execute(job, backend=backend, provider=provider, now=NOW)
        except BaseException:
            pass
    elif scenario == "legacy_intent":
        current = me.make_event("INTENT", token, occurred_at=NOW.isoformat())
        legacy = {key: value for key, value in current.items() if key != "abort_receipt"}
        legacy["schema"] = me.LEGACY_EVENT_SCHEMA
        controller._journal.append(journal_module.KIND_MAINTENANCE_EXECUTION, legacy)
    elif scenario == "byte_removal":
        _activate_actual_controller(controller, job)
        provider = Provider()
        hold = provider.acquire(token)
        controller.maintenance_revalidate(token, hold)
        proxy = me._JournalProxy(_ControllerJournalBackend(controller), provider, token, hold)

        class Owner(rc.MaintenanceOwner):
            def held(self, operation):
                return operation(rc.MaintenanceLease(
                    view, _policy(old.parents[1]), proxy, "crashing-owner", ()))

        original = storage.shutil.rmtree

        def removed_then_crash(path):
            original(path)
            raise KeyboardInterrupt("crash after byte removal")

        storage.shutil.rmtree = removed_then_crash
        try:
            rc.execute(job, owner=Owner(), now=NOW)
        except KeyboardInterrupt:
            pass
    else:
        raise AssertionError(f"unknown crash scenario {scenario}")
    connection.send({
        "refs": dict(git_backend.refs), "view": _plain_view(view),
        "job": {"plan": job.plan.to_dict(),
                "selected_artifact_ids": list(job.selected_artifact_ids),
                "policy_digest": job.policy_digest},
        "token": token.to_dict(),
        "hold": ({**hold.body(), "receipt_digest": hold.receipt_digest}
                 if hold is not None else None),
    })
    connection.close()
    os._exit(0)


def _crashed_runtime(native_root: Path, scenario: str):
    context = multiprocessing.get_context("fork")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_crash_owner,
                              args=(str(native_root), scenario, send))
    process.start()
    send.close()
    payload = receive.recv()
    receive.close()
    process.join(5)
    assert process.exitcode == 0
    backend = FakeGitBackend()
    backend.refs.update(payload["refs"])
    view = _native_view(payload["view"])
    job = rc.RetentionJob(
        rc.retention.RetentionPlan.from_dict(payload["job"]["plan"]),
        tuple(payload["job"]["selected_artifact_ids"]),
        payload["job"]["policy_digest"])
    return (backend, view, Path(view.identities[0].path), job,
            me.ExclusionToken.from_dict(payload["token"]),
            me.HoldReceipt.from_dict(payload["hold"])
            if payload["hold"] is not None else None)


def test_actual_controller_candidate_store_journal_and_exact_cost(runtime):
    controller, _candidates, _git_backend, view, old, job = runtime
    backend, provider = Backend(controller, view, _policy(old.parents[1])), Provider()
    result = me.execute(job, backend=backend, provider=provider, now=NOW)
    assert not old.exists() and backend.state == "complete"
    assert result.cost.artifact_count == result.cost.deleted_artifact_count == 1
    assert result.cost.reclaimed_bytes == result.cost.deleted_bytes_this_attempt
    assert result.cost.reclaimed_bytes > 0
    assert [entry.payload["reclamation_state"] for entry in controller._journal.read_all()
            if entry.kind == journal_module.KIND_TOMBSTONE] == ["intent", "reclaimed"]


def test_two_threads_exclude_competitor_and_control_stays_responsive(
        runtime, monkeypatch):
    controller, _candidates, _git_backend, view, old, job = runtime
    backend, provider = Backend(controller, view, _policy(old.parents[1])), Provider()
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    real_rmtree = shutil.rmtree

    def blocked(path):
        entered.set()
        assert release.wait(3)
        real_rmtree(path)

    monkeypatch.setattr("autokernel.storage.shutil.rmtree", blocked)
    errors = []

    def first():
        try:
            me.execute(job, backend=backend, provider=provider, now=NOW)
        except BaseException as exc:  # pragma: no cover - assertion reports it
            errors.append(exc)
        finally:
            finished.set()

    thread = threading.Thread(target=first)
    thread.start()
    assert entered.wait(3)
    with pytest.raises(me.MaintenanceExecutionRefused, match="already owned"):
        me.execute(job, backend=backend, provider=Provider(), now=NOW)
    snapshot = controller.snapshot()
    assert snapshot["campaign_id"] == controller.resolved.campaign_id
    drained = controller.apply_command(
        _command(controller.resolved, "drain-during-maintenance", "drain",
                 snapshot["control_revision"]))
    assert drained["accepted"] is True
    release.set()
    assert finished.wait(3)
    thread.join()
    assert errors == []


def test_generation_change_at_mutation_revalidation_aborts_before_bytes(runtime):
    controller, _candidates, _git_backend, view, old, job = runtime
    backend, provider = Backend(controller, view, _policy(old.parents[1])), Provider()
    provider.on_refresh = lambda: setattr(backend, "view", replace(view, generation=2))
    with pytest.raises(me.MaintenanceExecutionRefused, match="generation changed"):
        me.execute(job, backend=backend, provider=provider, now=NOW)
    assert old.exists() and backend.state == "aborted" and backend.token is None
    assert provider.finished_hold.provider_generation == 1


def test_dependency_root_change_at_revalidation_aborts_before_bytes(runtime):
    controller, _candidates, _git_backend, view, old, job = runtime
    backend, provider = Backend(controller, view, _policy(old.parents[1])), Provider()
    provider.on_refresh = lambda: setattr(
        backend, "view", replace(
            view, roots=replace(view.roots, launch_intents=("old-build",))))
    with pytest.raises(me.MaintenanceExecutionRefused, match="dependency/root"):
        me.execute(job, backend=backend, provider=provider, now=NOW)
    assert old.exists() and backend.state == "aborted" and backend.token is None


def test_stale_or_revoked_hold_aborts_without_tombstone(runtime):
    controller, _candidates, _git_backend, view, old, job = runtime
    backend, provider = Backend(controller, view, _policy(old.parents[1])), Provider()
    provider.revoked = True
    with pytest.raises(me.MaintenanceExecutionRefused, match="stale, revoked"):
        me.execute(job, backend=backend, provider=provider, now=NOW)
    assert old.exists() and backend.state == "unresolved"
    assert not [entry for entry in controller._journal.read_all()
                if entry.kind == journal_module.KIND_TOMBSTONE]


def test_changed_provider_generation_is_refused_before_tombstone_or_bytes(runtime):
    controller, _candidates, _git_backend, view, old, job = runtime
    backend, provider = Backend(controller, view, _policy(old.parents[1])), Provider()
    provider.on_refresh = lambda: setattr(provider, "generation", 2)
    with pytest.raises(me.MaintenanceExecutionRefused, match="generation changed"):
        me.execute(job, backend=backend, provider=provider, now=NOW)
    assert old.exists() and backend.state == "aborted" and backend.token is None
    assert not [entry for entry in controller._journal.read_all()
                if entry.kind == journal_module.KIND_TOMBSTONE]


def test_failed_pre_intent_provider_settlement_retains_exclusion(runtime):
    controller, _candidates, _git_backend, view, old, job = runtime
    backend, provider = Backend(controller, view, _policy(old.parents[1])), Provider()
    provider.fail_aborted_finish = True
    provider.on_refresh = lambda: setattr(backend, "view", replace(view, generation=2))
    with pytest.raises(me.MaintenanceExecutionRefused, match="generation changed"):
        me.execute(job, backend=backend, provider=provider, now=NOW)
    assert old.exists() and backend.state == "unresolved" and backend.token is not None
    assert provider.calls[-1] == "finish:aborted"
    assert controller.snapshot()["campaign_id"] == controller.resolved.campaign_id


def test_abort_settlement_uses_latest_valid_refreshed_hold(runtime):
    controller, _candidates, _git_backend, view, old, job = runtime
    backend, provider = Backend(controller, view, _policy(old.parents[1])), Provider()
    renewed_deadline = provider.deadline + 600

    def renew_then_stale():
        provider.deadline = renewed_deadline
        backend.view = replace(view, generation=2)

    provider.on_refresh = renew_then_stale
    with pytest.raises(me.MaintenanceExecutionRefused, match="generation changed"):
        me.execute(job, backend=backend, provider=provider, now=NOW)
    assert provider.finished_hold is not None
    assert provider.finished_hold.deadline == renewed_deadline
    assert backend.state == "aborted" and old.exists()


@pytest.mark.parametrize("reply", ["false", "missing", "misbound"])
def test_invalid_aborted_settlement_receipt_retains_exclusion(runtime, reply):
    controller, _candidates, _git_backend, view, old, job = runtime
    backend, provider = Backend(controller, view, _policy(old.parents[1])), Provider()
    provider.return_false_aborted = reply == "false"
    provider.aborted_reply = None if reply == "false" else reply
    provider.on_refresh = lambda: setattr(backend, "view", replace(view, generation=2))
    with pytest.raises(me.MaintenanceExecutionRefused, match="generation changed"):
        me.execute(job, backend=backend, provider=provider, now=NOW)
    assert old.exists() and backend.state == "unresolved" and backend.token is not None


@pytest.mark.parametrize("reply", ["exception", "missing", "misbound"])
def test_provider_acquisition_uncertainty_retains_unknown_ownership(runtime, reply):
    controller, _candidates, _git_backend, view, old, job = runtime
    backend, provider = Backend(controller, view, _policy(old.parents[1])), Provider()
    provider.acquire_error = reply == "exception"
    provider.acquire_reply = None if reply == "exception" else reply
    expected = "reply lost" if reply == "exception" else "[Hh]old"
    with pytest.raises(
            (RuntimeError, TypeError, me.MaintenanceExecutionRefused), match=expected):
        me.execute(job, backend=backend, provider=provider, now=NOW)
    assert old.exists() and backend.state == "unresolved" and backend.token is not None


def test_acquisition_error_remains_primary_when_unresolved_append_also_fails(runtime):
    controller, _candidates, _git_backend, view, old, job = runtime
    backend, provider = Backend(controller, view, _policy(old.parents[1])), Provider()
    backend.fail_unresolved = True
    provider.acquire_error = True
    with pytest.raises(RuntimeError, match="provider reply lost"):
        me.execute(job, backend=backend, provider=provider, now=NOW)
    assert old.exists() and backend.token is not None


def test_explicit_provider_no_hold_receipt_permits_abort(runtime):
    controller, _candidates, _git_backend, view, old, job = runtime
    backend, provider = Backend(controller, view, _policy(old.parents[1])), Provider()
    provider.no_hold = True
    with pytest.raises(me.MaintenanceExecutionRefused, match="explicitly refused"):
        me.execute(job, backend=backend, provider=provider, now=NOW)
    assert old.exists() and backend.state == "aborted" and backend.token is None


@pytest.mark.parametrize("failure", ["acquisition_reply_lost", "false_abort_receipt"])
def test_actual_controller_replay_keeps_uncertain_provider_ownership_gated(
        native_root, failure):
    git_backend, view, old, job, _token, _hold = _crashed_runtime(
        native_root, failure)
    reopened, reopened_candidates = _manager(native_root, backend=git_backend)
    try:
        assert old.exists()
        assert reopened._maintenance_state.phase == "UNRESOLVED"
        assert reopened._maintenance_state.owned
        assert rc.inspect_candidate_state(reopened_candidates) == view.candidate_state
        with pytest.raises(Exception, match="worker admission is fenced"):
            reopened.run_worker_stage(object())
        changed = replace(job, policy_digest="f" * 64)
        with pytest.raises(Exception, match="differs from unresolved"):
            reopened.maintenance_admit(changed)
    finally:
        with pytest.raises(Exception, match="maintenance ownership"):
            reopened.close()


def test_false_provider_completion_retains_named_unresolved_exclusion(runtime):
    controller, _candidates, _git_backend, view, old, job = runtime
    backend, provider = Backend(controller, view, _policy(old.parents[1])), Provider()
    provider.return_false_complete = True
    with pytest.raises(me.MaintenanceExecutionRefused, match="misbound"):
        me.execute(job, backend=backend, provider=provider, now=NOW)
    assert not old.exists() and backend.state == "unresolved" and backend.token is not None


def test_crash_after_intent_retains_visible_exclusion(runtime, monkeypatch):
    controller, _candidates, _git_backend, view, old, job = runtime
    backend, provider = Backend(controller, view, _policy(old.parents[1])), Provider()
    monkeypatch.setattr("autokernel.storage.shutil.rmtree",
                        lambda _path: (_ for _ in ()).throw(OSError("remove fault")))
    with pytest.raises(OSError, match="remove fault"):
        me.execute(job, backend=backend, provider=provider, now=NOW)
    assert old.exists() and backend.state == "unresolved" and backend.token is not None
    assert controller.snapshot()["campaign_id"] == controller.resolved.campaign_id


def test_crash_after_byte_removal_replays_native_intent_without_second_delete(runtime):
    controller, _candidates, git_backend, view, old, job = runtime
    backend, provider = Backend(controller, view, _policy(old.parents[1])), Provider()
    backend.fail_completion_append = True
    with pytest.raises(RuntimeError, match="completion append fault"):
        me.execute(job, backend=backend, provider=provider, now=NOW)
    assert not old.exists() and backend.state == "unresolved"
    assert [entry.payload["reclamation_state"] for entry in controller._journal.read_all()
            if entry.kind == journal_module.KIND_TOMBSTONE] == ["intent"]
    controller.close()
    reopened, reopened_candidates = _manager(old.parents[1], backend=git_backend)
    assert rc.inspect_candidate_state(reopened_candidates) == view.candidate_state
    backend.controller = reopened
    backend.fail_completion_append = False
    backend.recover = True
    try:
        result = me.execute(job, backend=backend, provider=Provider(), now=NOW)
        assert result.cost.reclaimed_bytes > 0
        assert result.cost.deleted_bytes_this_attempt == 0
        assert [entry.payload["reclamation_state"] for entry in reopened._journal.read_all()
                if entry.kind == journal_module.KIND_TOMBSTONE] == ["intent", "reclaimed"]
    finally:
        reopened.close()


def test_valid_provider_completion_then_controller_append_failure_stays_unresolved(runtime):
    controller, _candidates, _git_backend, view, old, job = runtime
    first = Backend(controller, view, _policy(old.parents[1]))
    me.execute(job, backend=first, provider=Provider(), now=NOW)
    assert not old.exists()
    retry = Backend(controller, view, _policy(old.parents[1]))
    retry.fail_complete = True
    with pytest.raises(RuntimeError, match="completion append uncertain"):
        me.execute(job, backend=retry, provider=Provider(), now=NOW)
    assert retry.state == "unresolved" and retry.token is not None
    assert not any(event == "ABORTED" for event in retry.events)


def test_default_core_and_provider_releases_are_unavailable(runtime):
    controller, candidates, _git_backend, _view_value, old, job = runtime
    with pytest.raises(me.MaintenanceExecutionRefused, match="controller maintenance"):
        me.execute(job, now=NOW)
    with pytest.raises(Exception, match="native retention catalog is unavailable"):
        controller.maintenance_admit(job)
    with pytest.raises(Exception, match="native retention catalog is unavailable"):
        candidates.retention_view()
    assert old.exists()


def test_closed_event_validator_and_transition_grammar(runtime):
    controller, _candidates, _git_backend, _view_value, _old, job = runtime
    token = _actual_token(controller, job)
    intent = me.make_event("INTENT", token, occurred_at=NOW.isoformat())
    with pytest.raises(me.MaintenanceExecutionRefused, match="missing/unknown"):
        me.validate_event({**intent, "unknown": None})
    with pytest.raises(ValueError, match="maintenance execution event"):
        controller._journal.append(
            journal_module.KIND_MAINTENANCE_EXECUTION,
            {**intent, "unknown": None})
    with pytest.raises(me.MaintenanceExecutionRefused, match="provider-owned receipt"):
        me.make_event("ABORTED", token, reason="no intent",
                      occurred_at=NOW.isoformat())
    legacy_abort = {
        key: value for key, value in intent.items() if key != "abort_receipt"
    }
    legacy_abort.update(schema=me.LEGACY_EVENT_SCHEMA, event="ABORTED",
                        reason="unverified legacy abort")
    with pytest.raises(me.MaintenanceExecutionRefused, match="lacks provider"):
        me.validate_event(legacy_abort)
    assert me.project_events([intent, intent]).phase == "INTENT"


def test_projection_rejects_misbound_first_provider_hold(runtime):
    controller, _candidates, _git_backend, _view_value, _old, job = runtime
    token = _actual_token(controller, job)
    intent = me.make_event("INTENT", token, occurred_at=NOW.isoformat())
    good = Provider()._receipt(token)
    misbound = replace(good, request_digest="f" * 64, receipt_digest="")
    held = me.make_event(
        "PROVIDER_HELD", token, hold=misbound, occurred_at=NOW.isoformat())

    with pytest.raises(me.MaintenanceExecutionRefused, match="misbound"):
        me.project_events([intent, held])


@pytest.mark.parametrize("terminal", ["COMPLETED", "ABORTED"])
def test_new_intent_after_terminal_clears_previous_hold_and_cost(runtime, terminal):
    controller, _candidates, _git_backend, _view_value, _old, job = runtime
    token = _actual_token(controller, job)
    hold = Provider()._receipt(token)
    intent = me.make_event("INTENT", token, occurred_at=NOW.isoformat())
    held = me.make_event(
        "PROVIDER_HELD", token, hold=hold, occurred_at=NOW.isoformat())
    if terminal == "COMPLETED":
        cost = me.MaintenanceCost(1, 2, 0, 0)
        io_complete = me.make_event(
            "IO_COMPLETE", token, hold=hold, cost=cost,
            occurred_at=NOW.isoformat())
        accounting = me.AccountingReceipt(
            hold.receipt_digest, token.token_digest, cost, "complete")
        terminal_event = me.make_event(
            "COMPLETED", token, hold=hold, cost=cost,
            accounting_receipt_digest=accounting.receipt_digest,
            occurred_at=NOW.isoformat())
        history = [intent, held, io_complete, terminal_event]
    else:
        cost = me.MaintenanceCost(0, 0, 0, 0)
        accounting = me.AccountingReceipt(
            hold.receipt_digest, token.token_digest, cost, "aborted")
        terminal_event = me.make_event(
            "ABORTED", token, hold=hold, cost=cost,
            accounting_receipt_digest=accounting.receipt_digest,
            abort_receipt=accounting, reason="cancelled",
            occurred_at=NOW.isoformat())
        history = [intent, held, terminal_event]
    next_token = replace(token, token_id="maintenance-next", token_digest="")
    history.append(me.make_event(
        "INTENT", next_token, occurred_at=NOW.isoformat()))

    projected = me.project_events(history)
    assert projected.phase == "INTENT"
    assert projected.token == next_token
    assert projected.hold is None
    assert projected.cost is None


def test_projection_rejects_changed_completion_cost_and_forged_accounting(runtime):
    controller, _candidates, _git_backend, _view_value, _old, job = runtime
    token = _actual_token(controller, job)
    hold = Provider()._receipt(token)
    original = me.MaintenanceCost(1, 2, 0, 0)
    changed = me.MaintenanceCost(1, 3, 0, 0)
    history = [
        me.make_event("INTENT", token, occurred_at=NOW.isoformat()),
        me.make_event(
            "PROVIDER_HELD", token, hold=hold, occurred_at=NOW.isoformat()),
        me.make_event(
            "IO_COMPLETE", token, hold=hold, cost=original,
            occurred_at=NOW.isoformat()),
    ]
    changed_accounting = me.AccountingReceipt(
        hold.receipt_digest, token.token_digest, changed, "complete")
    changed_completion = me.make_event(
        "COMPLETED", token, hold=hold, cost=changed,
        accounting_receipt_digest=changed_accounting.receipt_digest,
        occurred_at=NOW.isoformat())

    with pytest.raises(me.MaintenanceExecutionRefused, match="cost differs"):
        me.project_events([*history, changed_completion])
    with pytest.raises(me.MaintenanceExecutionRefused, match="digest is misbound"):
        me.make_event(
            "COMPLETED", token, hold=hold, cost=original,
            accounting_receipt_digest="f" * 64, occurred_at=NOW.isoformat())


def test_new_intent_after_terminal_requires_fresh_token_chain(runtime):
    controller, _candidates, _git_backend, _view_value, _old, job = runtime
    token = _actual_token(controller, job)
    refused = me.NoHoldReceipt(
        "fixture-provider", me._request_digest(token), 1,
        NOW.isoformat(), "capacity refused")
    history = [
        me.make_event("INTENT", token, occurred_at=NOW.isoformat()),
        me.make_event(
            "ABORTED", token, reason="capacity refused", abort_receipt=refused,
            occurred_at=NOW.isoformat()),
    ]
    same = me.make_event("INTENT", token, occurred_at=NOW.isoformat())
    arbitrary_predecessor = replace(
        token, token_id="maintenance-next",
        predecessor_token_digest="e" * 64, token_digest="")
    chained = me.make_event(
        "INTENT", arbitrary_predecessor, occurred_at=NOW.isoformat())

    with pytest.raises(me.MaintenanceExecutionRefused):
        me.project_events([*history, same])
    with pytest.raises(me.MaintenanceExecutionRefused):
        me.project_events([*history, chained])


def test_active_nonidentical_intent_requires_fresh_chained_token(runtime):
    controller, _candidates, _git_backend, _view_value, _old, job = runtime
    token = _actual_token(controller, job)
    intent = me.make_event("INTENT", token, occurred_at=NOW.isoformat())
    repeated = me.make_event(
        "INTENT", token, occurred_at="2026-08-03T12:00:01+00:00")

    with pytest.raises(me.MaintenanceExecutionRefused, match="fresh chained token"):
        me.project_events([intent, repeated])


def test_legacy_non_abort_event_preserves_shape_across_journal_reopen(native_root):
    git_backend, _view, _old, _job, _token, _hold = _crashed_runtime(
        native_root, "legacy_intent")
    reopened, _ = _manager(native_root, backend=git_backend)
    try:
        assert reopened._maintenance_state.phase == "UNRESOLVED"
        assert reopened._maintenance_state.owned
        legacy = reopened._maintenance_events[0]
        assert legacy["schema"] == me.LEGACY_EVENT_SCHEMA
        assert "abort_receipt" not in legacy
        assert me.validate_event(legacy) == legacy
    finally:
        with pytest.raises(Exception, match="maintenance ownership"):
            reopened.close()


def test_public_abort_requires_provider_evidence_and_replays_exact_no_hold(runtime):
    controller, _candidates, git_backend, _view, old, job = runtime
    token = _activate_actual_controller(controller, job)
    with pytest.raises(TypeError):
        controller.maintenance_abort(token, "receiptless")
    assert controller._maintenance_state.owned
    refusal = me.NoHoldReceipt(
        "fixture-provider", me._request_digest(token), 1,
        NOW.isoformat(), "capacity refused")
    controller.maintenance_abort(token, "capacity refused", refusal)
    controller.maintenance_abort(token, "capacity refused", refusal)
    assert not controller._maintenance_state.owned
    controller.close()
    reopened, _ = _manager(old.parents[1], backend=git_backend)
    try:
        assert reopened._maintenance_state.phase == "ABORTED"
        assert reopened._maintenance_state.abort_receipt_digest == refusal.receipt_digest
        assert not reopened._maintenance_state.owned
    finally:
        reopened.close()


def test_public_abort_validates_exact_accounting_evidence(runtime):
    controller, _candidates, _git_backend, _view, old, job = runtime
    token = _activate_actual_controller(controller, job)
    provider = Provider()
    hold = provider.acquire(token)
    controller.maintenance_revalidate(token, hold)
    zero = me.MaintenanceCost(0, 0, 0, 0)
    misbound = me.AccountingReceipt(hold.receipt_digest, "f" * 64, zero, "aborted")
    with pytest.raises(Exception, match="misbound"):
        controller.maintenance_abort(token, "stale view", misbound, hold)
    assert controller._maintenance_state.owned and old.exists()
    settled = provider.finish(hold, token, zero, "aborted")
    controller.maintenance_abort(token, "stale view", settled, hold)
    controller.maintenance_abort(token, "stale view", settled, hold)
    assert controller._maintenance_state.phase == "ABORTED"
    assert controller._maintenance_state.abort_receipt_digest == settled.receipt_digest
    assert not controller._maintenance_state.owned


def test_actual_controller_methods_complete_and_exact_retry(runtime):
    controller, _candidates, _git_backend, view, old, job = runtime
    token = _activate_actual_controller(controller, job)
    provider = Provider()
    hold = provider.acquire(token)
    controller.maintenance_revalidate(token, hold)
    controller.maintenance_revalidate(token, hold)
    cost = me.MaintenanceCost(1, storage.measure_usage(old).bytes_on_disk, 0, 0)
    controller.maintenance_io_complete(token, cost)
    controller.maintenance_io_complete(token, cost)
    accounting = provider.finish(hold, token, cost, "complete")
    controller.maintenance_complete(token, accounting)
    controller.maintenance_complete(token, accounting)
    assert controller._maintenance_state.phase == "COMPLETED"
    assert not controller._maintenance_state.owned
    assert view.generation == token.snapshot_generation


def test_uncertain_maintenance_append_poisons_actual_controller(runtime, monkeypatch):
    controller, _candidates, _git_backend, _view, _old, job = runtime
    token = _activate_actual_controller(controller, job)

    def uncertain(*_args, **_kwargs):
        raise OSError("fsync result unknown")

    monkeypatch.setattr(controller._journal, "append", uncertain)
    with pytest.raises(OSError, match="unknown"):
        controller.maintenance_unresolved(token, "provider release uncertain")
    with pytest.raises(Exception, match="poisoned; replay required"):
        controller.snapshot()


def test_actual_controller_exclusion_remains_responsive_during_blocked_disk(
        runtime, monkeypatch):
    controller, _candidates, _git_backend, view, old, job = runtime
    token = _activate_actual_controller(controller, job)
    provider = Provider()
    hold = provider.acquire(token)
    controller.maintenance_revalidate(token, hold)
    proxy = me._JournalProxy(
        _ControllerJournalBackend(controller), provider, token, hold)

    class Owner(rc.MaintenanceOwner):
        def held(self, operation):
            return operation(rc.MaintenanceLease(
                view, _policy(old.parents[1]), proxy, "actual-controller", ()))

    entered, release = threading.Event(), threading.Event()
    real_rmtree = shutil.rmtree

    def blocked(path):
        entered.set()
        assert release.wait(3)
        real_rmtree(path)

    monkeypatch.setattr("autokernel.storage.shutil.rmtree", blocked)
    result = []
    thread = threading.Thread(
        target=lambda: result.append(rc.execute(job, owner=Owner(), now=NOW)))
    thread.start()
    assert entered.wait(3)
    with pytest.raises(Exception, match="already owned"):
        controller.maintenance_admit(job)
    with pytest.raises(Exception, match="worker admission is fenced"):
        controller.run_worker_stage(object())
    with pytest.raises(Exception, match="maintenance ownership"):
        controller.close()
    assert controller._entered

    def mutate(context):
        operation_payload = {"state": {}, "manifest": {}}
        context.append(
            phase="INTENT", transaction_id="blocked-during-io", operation="init",
            payload_digest=ct._intent_digest(
                "init", None, operation_payload, (), ()),
            data={"expected_state_digest": None,
                  "operation_payload": operation_payload,
                  "prepared_objects": [], "prepared_refs": []})

    with pytest.raises(Exception, match="candidate mutation is fenced"):
        controller.candidate_transaction(mutate)
    snapshot = controller.snapshot()
    drained = controller.apply_command(_command(
        controller.resolved, "drain-while-maintenance", "drain",
        snapshot["control_revision"]))
    assert drained["accepted"] is True
    release.set()
    thread.join(3)
    assert not thread.is_alive() and len(result) == 1
    cost = me.MaintenanceCost.from_result(result[0])
    controller.maintenance_io_complete(token, cost)
    accounting = provider.finish(proxy.hold, token, cost, "complete")
    controller.maintenance_complete(token, accounting)


def test_shutdown_deadline_retains_maintenance_then_terminal_wakes_and_reopens(runtime):
    controller, _candidates, git_backend, _view, old, job = runtime
    token = _activate_actual_controller(controller, job)
    provider = Provider()
    hold = provider.acquire(token)
    controller.maintenance_revalidate(token, hold)
    controller.request_shutdown_drain()
    with pytest.raises(Exception, match="deadline expired; ownership retained"):
        controller.await_shutdown_drain(time.monotonic())
    with pytest.raises(Exception, match="maintenance ownership"):
        controller.close()

    zero = me.MaintenanceCost(0, 0, 0, 0)
    accounting = provider.finish(hold, token, zero, "aborted")
    controller.maintenance_abort(token, "shutdown before mutation", accounting, hold)
    drained = controller.await_shutdown_drain(time.monotonic() + 0.5)
    assert drained["completed"] is True and drained["observed_state"] == "drained"
    controller.close()

    reopened, _ = _manager(old.parents[1], backend=git_backend)
    try:
        assert reopened.request_shutdown_drain() == drained
        assert reopened.await_shutdown_drain(time.monotonic() + 0.5) == drained
    finally:
        reopened.close()


def test_exclusion_gates_worker_candidate_and_new_native_publication(runtime):
    controller, candidates, _git_backend, view, _old, job = runtime
    _activate_actual_controller(controller, job)
    assert candidates.inspect()["state"] == view.candidate_state.to_dict()
    with pytest.raises(Exception, match="native retention catalog is unavailable"):
        candidates.retention_view()
    with pytest.raises(Exception, match="worker admission is fenced"):
        controller.run_worker_stage(object())

    def mutate(context):
        operation_payload = {"state": {}, "manifest": {}}
        context.append(
            phase="INTENT", transaction_id="blocked", operation="init",
            payload_digest=ct._intent_digest(
                "init", None, operation_payload, (), ()),
            data={"expected_state_digest": None,
                  "operation_payload": operation_payload,
                  "prepared_objects": [], "prepared_refs": []})

    with pytest.raises(Exception, match="candidate mutation is fenced"):
        controller.candidate_transaction(mutate)

    payload = {"fixture": True}
    prior = {"existing": "capture"}
    token = object()
    with controller._mutex:
        controller._native_capabilities.add(token)
        controller._native_records["existing"] = prior
        controller._native_payload_digests["existing"] = schemas.content_hash(payload)
        assert controller._capture_native_locked(
            "existing", payload, token=token,
            lifetime_token=controller._lifetime_token,
            thread_id=threading.get_ident()) == prior
        with pytest.raises(Exception, match="dependency publication is fenced"):
            controller._capture_native_locked(
                "new", payload, token=token,
                lifetime_token=controller._lifetime_token,
                thread_id=threading.get_ident())


def test_restart_rebinds_unresolved_without_restoring_stale_provider_authority(
        native_root):
    git_backend, view, old, job, old_token, old_hold = _crashed_runtime(
        native_root, "byte_removal")
    reopened, reopened_candidates = _manager(native_root, backend=git_backend)
    try:
        assert not old.exists()
        assert reopened._maintenance_state.phase == "UNRESOLVED"
        assert reopened._maintenance_state.owned
        assert rc.inspect_candidate_state(reopened_candidates) == view.candidate_state
        assert reopened.snapshot()["campaign_id"] == reopened.resolved.campaign_id
        with pytest.raises(Exception, match="indexed owner|stale owner"):
            reopened.maintenance_revalidate(old_token, old_hold)

        fresh = reopened.maintenance_admit(job)
        assert fresh.predecessor_token_digest == old_token.token_digest
        with pytest.raises(me.MaintenanceExecutionRefused, match="misbound"):
            reopened.maintenance_revalidate(fresh, old_hold)
        provider = Provider()
        hold = provider.acquire(fresh)
        reopened.maintenance_revalidate(fresh, hold)
        recovery_proxy = me._JournalProxy(
            _ControllerJournalBackend(reopened), provider, fresh, hold)

        class RecoveryOwner(rc.MaintenanceOwner):
            def held(self, operation):
                return operation(rc.MaintenanceLease(
                    view, _policy(old.parents[1]), recovery_proxy,
                    "actual-controller-recovery",
                    _native_tombstones(reopened._journal)))

        result = rc.execute(job, owner=RecoveryOwner(), now=NOW)
        cost = me.MaintenanceCost.from_result(result)
        assert cost.reclaimed_bytes > 0 and cost.deleted_bytes_this_attempt == 0
        reopened.maintenance_io_complete(fresh, cost)
        accounting = provider.finish(recovery_proxy.hold, fresh, cost, "complete")
        reopened.maintenance_complete(fresh, accounting)
        assert reopened._maintenance_state.phase == "COMPLETED"
        assert [entry.payload["reclamation_state"]
                for entry in reopened._journal.read_all()
                if entry.kind == journal_module.KIND_TOMBSTONE] == ["intent", "reclaimed"]
    finally:
        reopened.close()
