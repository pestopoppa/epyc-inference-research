"""AKU-10g: broker hold/refresh/settlement interface over the actual controller.

``InMemoryMaintenanceBroker`` below is a FIXTURE ONLY. It is never registered in
``provider_factories`` and is not the OP-41 broker (AKU-11); it exists so the
interface, the concrete controller backend and replay can be exercised hermetically.
"""
from __future__ import annotations

from dataclasses import replace
import copy
import itertools
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time

import pytest

from .. import journal as journal_module
from . import maintenance_execution as me
from . import native_retention_catalog as catalog
from . import provider_factories
from . import retention_consumer as rc
from .test_maintenance_execution import _native_view, _plain_view
from .test_native_retention_catalog import _controller, _preparations, _seed
from .test_retention_consumer import NOW, _policy, _view

_REAL_RMTREE = shutil.rmtree
ZERO = me.MaintenanceCost(0, 0, 0, 0)


class InMemoryMaintenanceBroker(me.BrokerHoldProvider):
    """FIXTURE ONLY: in-memory stand-in for the OP-41 maintenance hold broker."""

    PROVIDER_ID = "fixture-in-memory-broker"

    def __init__(self, *, state=None, mutex=None, clock_start=1000, lease_s=3600.0):
        self.state = state or {"holds": {}, "ledger": [], "max_active": 0}
        self.mutex, self.lease_s = mutex, lease_s
        self._clock = itertools.count(clock_start, 250)
        self.lock = threading.Lock()
        self.calls, self.violations, self.gate, self.fail = [], [], {}, {}

    def _enter(self, name):
        self.calls.append(name)
        if self.mutex is not None:
            if self.mutex._is_owned() or not self.mutex.acquire(blocking=False):
                self.violations.append(name)
            else:
                self.mutex.release()
        if name in self.gate:
            entered, release = self.gate[name]
            entered.set()
            assert release.wait(10)
        if name in self.fail:
            raise self.fail.pop(name)

    def _receipt(self, hold_id):
        row = self.state["holds"][hold_id]
        return me.HoldReceipt(self.PROVIDER_ID, hold_id, row["request_digest"], 1, 1,
                              row["deadline"], True, False)

    def _issue(self, hold_id):
        receipt = self._receipt(hold_id)
        self.state["holds"][hold_id]["issued"].append(receipt.receipt_digest)
        return receipt

    def latest(self, hold_id):
        return self._receipt(hold_id)

    def acquire(self, token):
        self._enter("acquire")
        with self.lock:
            holds = self.state["holds"]
            hold_id = f"hold-{len(holds) + 1}"
            holds[hold_id] = {"request_digest": me._request_digest(token),
                              "token_digest": token.token_digest,
                              "acquired_ms": next(self._clock),
                              "deadline": time.time() + self.lease_s,
                              "refreshes": 0, "issued": [], "settlement": None}
            active = sum(row["settlement"] is None for row in holds.values())
            self.state["max_active"] = max(self.state["max_active"], active)
            return self._issue(hold_id)

    def refresh(self, hold, token):
        self._enter("refresh")
        with self.lock:
            row = self.state["holds"].get(hold.hold_id)
            if (row is None or row["settlement"] is not None
                    or row["request_digest"] != me._request_digest(token)):
                raise me.MaintenanceExecutionRefused("broker hold is unknown or settled")
            row["deadline"] = time.time() + self.lease_s
            row["refreshes"] += 1
            return self._issue(hold.hold_id)

    def _settle_locked(self, hold, token, cost, disposition):
        row = self.state["holds"].get(hold.hold_id)
        if (row is None or hold.receipt_digest not in row["issued"]
                or row["token_digest"] != token.token_digest):
            raise me.MaintenanceExecutionRefused("broker never issued this hold receipt")
        if row["settlement"] is not None:
            existing = me.HoldSettlement.from_dict(row["settlement"])
            if (existing.hold_receipt_digest, existing.cost, existing.disposition) \
                    == (hold.receipt_digest, cost, disposition):
                return existing
            raise me.MaintenanceExecutionRefused("broker hold already settled differently")
        settled_at = next(self._clock)
        settlement = me.HoldSettlement(
            self.PROVIDER_ID, hold.hold_id, hold.receipt_digest, token.token_digest, cost,
            disposition, me.AccountingReceipt(hold.receipt_digest, token.token_digest, cost,
                                              disposition).receipt_digest,
            row["acquired_ms"], settled_at, settled_at - row["acquired_ms"], row["refreshes"])
        row["settlement"] = settlement.to_dict()
        self.state["ledger"].append(settlement.to_dict())
        return settlement

    def settle(self, hold, token, cost, disposition):
        self._enter("settle")
        with self.lock:
            return self._settle_locked(hold, token, cost, disposition)

    def resolve(self, token, hold):
        self._enter("resolve")
        request = me._request_digest(token)
        with self.lock:
            for hold_id, row in self.state["holds"].items():
                if row["request_digest"] == request:
                    if row["settlement"] is not None:
                        return me.HoldSettlement.from_dict(row["settlement"])
                    return self._settle_locked(self._receipt(hold_id), token, ZERO, "aborted")
            return me.NoHoldReceipt(self.PROVIDER_ID, request, 1, NOW.isoformat(),
                                    "no hold exists for the unresolved request")


@pytest.fixture
def root():
    path = Path(tempfile.mkdtemp(prefix="_maintenance_broker_", dir=Path(__file__).parent))
    try:
        yield path
    finally:
        _REAL_RMTREE(path, ignore_errors=True)


def _runtime(path, seeded=None, *, artifact_root=True, bind=True):
    # A declared artifact store lets every owner class register a gap-free record.
    artifact_root = path / "artifacts" if artifact_root else None
    resolved, anchors, seed = seeded or _seed(artifact_root=artifact_root)
    store = path / "service"
    store.mkdir(mode=0o700)
    store.chmod(0o700)
    controller = _controller(resolved, store)
    controller.__enter__()
    controller.install_native_retention_catalog(
        seed, runtime_anchors=anchors, model_preparations=_preparations(seed),
        artifact_root=artifact_root)
    frontier = controller.native_retention_catalog_capture()["frontier"]
    view, old = _view(path)
    node = view.nodes[-1]
    node = replace(node, expiry=replace(node.expiry, campaign_id=resolved.campaign_id))
    view = replace(view, snapshot_id=seed.seed_digest,
                   generation=max(1, int(frontier[2]) + 1), nodes=view.nodes[:-1] + (node,))
    prepared = _prepared(view, frontier)
    policy = _policy(path)
    job, _ = rc.prepare(view, policy, limit=1, now=NOW)
    if bind:
        controller.bind_prepared_retention_job(job, prepared=prepared, policy=policy)
    return controller, prepared, policy, job, old


def _prepared(view, frontier):
    return catalog.PreparedCatalogView(
        view, frontier, rc.collect_native_snapshot(view).snapshot_digest, catalog._VIEW_TOKEN)


@pytest.fixture
def runtime(root):
    values = _runtime(root)
    try:
        yield values
    finally:
        if values[0]._entered and not values[0]._maintenance_state.owned:
            values[0].close()


def _events(controller):
    return [entry.payload for entry in controller._journal.read_all()
            if entry.kind == journal_module.KIND_MAINTENANCE_EXECUTION]


def _held_ms(rows):
    return sum(row["held_ms"] for row in rows)


def test_hold_refresh_completion_settles_exactly_once_with_held_time(runtime, root):
    controller, prepared, policy, job, old = runtime
    broker = InMemoryMaintenanceBroker(mutex=controller._mutex)
    backend = me.CampaignMaintenanceBackend(controller, prepared, policy, broker)
    result = me.execute(job, backend=backend, now=NOW)
    assert not old.exists() and broker.violations == []
    assert controller._maintenance_state.phase == "COMPLETED"
    assert broker.calls.count("acquire") == 1 and broker.calls.count("settle") == 1
    assert broker.calls.count("refresh") == 2  # one per native tombstone append
    [row] = broker.state["ledger"]
    settlement = me.HoldSettlement.from_dict(row)
    assert settlement.disposition == "complete" and settlement.cost == result.cost
    assert settlement.accounting_receipt_digest == result.accounting.receipt_digest
    assert settlement.refresh_count == 2
    assert settlement.held_ms == settlement.settled_at_ms - settlement.acquired_at_ms > 0
    completed = _events(controller)[-1]
    assert completed["schema"] == me.SETTLED_EVENT_SCHEMA
    assert completed["settlement"] == row
    assert _held_ms(event["settlement"] for event in _events(controller)
                    if event.get("settlement")) == _held_ms(broker.state["ledger"])
    # Exactly once: a repeat settlement and a repeat controller completion are no-ops.
    final_hold = broker.latest(settlement.hold_id)
    token = controller._maintenance_state.token
    assert broker.settle(final_hold, token, result.cost, "complete") == settlement
    with pytest.raises(me.MaintenanceExecutionRefused, match="settled differently"):
        broker.settle(final_hold, token, ZERO, "aborted")
    count = len(_events(controller))
    controller.maintenance_complete(token, result.accounting, settlement)
    assert len(_events(controller)) == count and len(broker.state["ledger"]) == 1
    resolved = controller.resolved
    controller.close()
    reopened = _controller(resolved, root / "service")
    reopened.__enter__()
    try:
        assert reopened._maintenance_state.phase == "COMPLETED"
        assert _events(reopened)[-1] == completed
    finally:
        reopened.close()


@pytest.mark.parametrize("failure", ["refresh_refused", "no_hold"])
def test_abort_receipt_settles_hold_or_records_explicit_refusal(runtime, failure):
    controller, prepared, policy, job, old = runtime
    broker = InMemoryMaintenanceBroker(mutex=controller._mutex)
    if failure == "refresh_refused":
        broker.fail["refresh"] = me.MaintenanceExecutionRefused("broker revoked refresh")
        expected = "revoked refresh"
    else:
        broker.acquire = lambda token: me.NoHoldReceipt(
            broker.PROVIDER_ID, me._request_digest(token), 1, NOW.isoformat(), "capacity")
        expected = "explicitly refused"
    backend = me.CampaignMaintenanceBackend(controller, prepared, policy, broker)
    with pytest.raises(me.MaintenanceExecutionRefused, match=expected):
        me.execute(job, backend=backend, now=NOW)
    state = controller._maintenance_state
    assert old.exists() and state.phase == "ABORTED" and not state.owned
    aborted = _events(controller)[-1]
    if failure == "no_hold":
        assert broker.state["ledger"] == [] and "settlement" not in aborted
        return
    [row] = broker.state["ledger"]
    assert row["disposition"] == "aborted" and row["cost"] == ZERO.to_dict()
    assert state.abort_receipt_digest == row["accounting_receipt_digest"]
    assert aborted["settlement"] == row and row["hold_receipt_digest"] == state.hold.receipt_digest


def test_refresh_deadline_exceeded_aborts_with_exact_accounting(runtime):
    controller, prepared, policy, job, old = runtime
    broker = InMemoryMaintenanceBroker(mutex=controller._mutex)
    entered, release = threading.Event(), threading.Event()
    broker.gate["refresh"] = (entered, release)
    bounded = me.BoundedHoldProvider(broker, mutex=controller._mutex, deadline_s=0.3)
    backend = me.CampaignMaintenanceBackend(controller, prepared, policy, bounded)
    try:
        with pytest.raises(me.ProviderDeadlineExceeded, match="refresh exceeded"):
            me.execute(job, backend=backend, now=NOW)
        assert entered.is_set() and old.exists()
        state = controller._maintenance_state
        assert state.phase == "ABORTED" and not state.owned
        [row] = broker.state["ledger"]
        assert row["disposition"] == "aborted" and row["cost"] == ZERO.to_dict()
        assert row["hold_receipt_digest"] == state.hold.receipt_digest
        assert row["held_ms"] == row["settled_at_ms"] - row["acquired_at_ms"]
        assert _events(controller)[-1]["settlement"] == row
    finally:
        release.set()
        for thread in threading.enumerate():
            if thread.name == "maintenance-refresh":
                thread.join(5)
    # The late refresh lost to the settlement: no refresh was ever granted.
    assert broker.state["holds"]["hold-1"]["refreshes"] == 0
    assert broker.violations == []


def test_provider_calls_are_refused_and_detected_under_the_mutex(runtime):
    controller, prepared, policy, job, _old = runtime
    broker = InMemoryMaintenanceBroker(mutex=controller._mutex)
    bounded = me.BoundedHoldProvider(broker, mutex=controller._mutex, deadline_s=1)
    token = me.ExclusionToken(
        "probe", "c", "a" * 64, 1, "s", 1, "snap", 1, "b" * 64, job.plan.plan_digest,
        job.policy_digest, job.selected_artifact_ids, NOW.isoformat())
    with controller._mutex:
        with pytest.raises(me.MaintenanceExecutionRefused, match="under controller mutex"):
            bounded.acquire(token)
        broker.acquire(token)  # an unwrapped call is recorded by the fixture checker
    assert broker.violations == ["acquire"] and broker.calls == ["acquire"]
    with pytest.raises(me.MaintenanceExecutionRefused, match="0..300s"):
        me.BoundedHoldProvider(me.MaintenanceHoldProvider(), mutex=controller._mutex,
                               deadline_s=1)


def test_default_controller_backend_and_factories_leave_execution_unavailable(runtime):
    controller, prepared, policy, job, old = runtime
    # AKU-11 registers only the activation-gated lifecycle provider; no maintenance broker.
    assert set(provider_factories.REGISTERED_FACTORIES) <= {"host_claim_v1"}
    with pytest.raises(ValueError, match="not registered"):
        provider_factories.construct("maintenance-broker")
    with pytest.raises(me.MaintenanceExecutionRefused, match="controller maintenance"):
        me.execute(job, now=NOW)
    backend = me.CampaignMaintenanceBackend(controller, prepared, policy)
    with pytest.raises(me.MaintenanceExecutionRefused, match="provider is unavailable"):
        me.execute(job, backend=backend, now=NOW)
    for provider in (None, me.BrokerHoldProvider(), me.MaintenanceHoldProvider()):
        with pytest.raises(me.MaintenanceExecutionRefused, match="provider is unavailable"):
            me.CampaignMaintenanceBackend(controller, prepared, policy, provider).admit(job)
    assert controller._maintenance_state.phase is None and old.exists()
    assert _events(controller) == []  # refused before INTENT
    for call in (lambda: me.BrokerHoldProvider().settle(None, None, ZERO, "aborted"),
                 lambda: me.BrokerHoldProvider().resolve(None, None)):
        with pytest.raises(me.MaintenanceExecutionRefused, match="broker is unavailable"):
            call()


def test_settlement_records_are_closed_and_bound(runtime):
    controller, _prepared, _policy, job, _old = runtime
    token = me.ExclusionToken(
        "maintenance-x", controller.resolved.campaign_id, controller.config_digest, 1,
        "sup", 1, job.plan.snapshot_id, job.plan.snapshot_generation,
        job.plan.snapshot_digest, job.plan.plan_digest, job.policy_digest,
        job.selected_artifact_ids, NOW.isoformat())
    intent = me.make_event("INTENT", token, occurred_at=NOW.isoformat())
    assert intent["schema"] == me.EVENT_SCHEMA and "settlement" not in intent
    hold = InMemoryMaintenanceBroker().acquire(token)
    cost = me.MaintenanceCost(1, 9, 1, 9)
    accounting = me.AccountingReceipt(hold.receipt_digest, token.token_digest, cost, "complete")
    good = me.HoldSettlement("p", "hold-1", hold.receipt_digest, token.token_digest, cost,
                             "complete", accounting.receipt_digest, 10, 30, 20, 0)
    with pytest.raises(me.MaintenanceExecutionRefused, match="interval"):
        replace(good, held_ms=19, receipt_digest="")
    with pytest.raises(me.MaintenanceExecutionRefused, match="accounting digest"):
        replace(good, cost=ZERO, receipt_digest="")
    completed = me.make_event("COMPLETED", token, hold=hold, cost=cost,
                              accounting_receipt_digest=accounting.receipt_digest,
                              occurred_at=NOW.isoformat(), settlement=good)
    assert me.validate_event(completed) == completed
    other = me.MaintenanceCost(1, 8, 1, 8)
    with pytest.raises(me.MaintenanceExecutionRefused, match="settlement is misbound"):
        me.make_event("IO_COMPLETE", token, hold=hold, cost=cost,
                      occurred_at=NOW.isoformat(), settlement=good)
    with pytest.raises(me.MaintenanceExecutionRefused, match="settlement is misbound"):
        me.make_event("COMPLETED", token, hold=hold, cost=other,
                      accounting_receipt_digest=me.AccountingReceipt(
                          hold.receipt_digest, token.token_digest, other,
                          "complete").receipt_digest,
                      occurred_at=NOW.isoformat(), settlement=good)
    with pytest.raises(me.MaintenanceExecutionRefused, match="recovery INTENT"):
        me.make_event("INTENT", token, occurred_at=NOW.isoformat(), settlement=good)
    with pytest.raises(me.MaintenanceExecutionRefused, match="missing/unknown"):
        me.validate_event({**intent, "settlement": None})  # v2 has no settlement field


def _crash_after_hold(path, seeded, journaled, connection):
    """Forked owner: take a broker hold, optionally journal it, then die unsettled."""
    controller, prepared, policy, job, _old = _runtime(Path(path), seeded)
    broker = InMemoryMaintenanceBroker(mutex=controller._mutex)
    backend = me.CampaignMaintenanceBackend(controller, prepared, policy, broker)
    token = backend.admit(job).token
    hold = broker.acquire(token)
    if journaled:
        backend.revalidate(token, hold)
    connection.send({"state": copy.deepcopy(broker.state), "view": _plain_view(prepared.view),
                     "token": token.to_dict(), "hold_digest": hold.receipt_digest})
    connection.close()
    os._exit(0)


@pytest.mark.parametrize("journaled", [True, False])
def test_crash_after_hold_before_receipt_replays_uncertain_and_resolves_first(
        root, journaled):
    seeded = _seed(artifact_root=root / "artifacts")
    context = multiprocessing.get_context("fork")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_crash_after_hold,
                              args=(str(root), seeded, journaled, send))
    process.start()
    send.close()
    payload = receive.recv()
    receive.close()
    process.join(10)
    assert process.exitcode == 0
    reopened = _controller(seeded[0], root / "service")
    reopened.__enter__()
    try:
        state = reopened._maintenance_state
        # Uncertain: neither success nor absence; the exclusion stays owned.
        assert state.phase == "UNRESOLVED" and state.owned
        assert (state.hold.receipt_digest if state.hold else None) == (
            payload["hold_digest"] if journaled else None)
        view = _native_view(payload["view"])
        prepared = _prepared(view, reopened.native_retention_catalog_capture()["frontier"])
        policy = _policy(root)
        job, _ = rc.prepare(view, policy, limit=1, now=NOW)
        with pytest.raises(Exception, match="needs its hold resolved"):
            reopened.maintenance_admit(job)
        broker = InMemoryMaintenanceBroker(state=payload["state"], mutex=reopened._mutex,
                                           clock_start=10_000)
        broker.fail["resolve"] = me.MaintenanceExecutionRefused("broker unreachable")
        backend = me.CampaignMaintenanceBackend(reopened, prepared, policy, broker)
        with pytest.raises(me.MaintenanceExecutionRefused, match="unreachable"):
            me.execute(job, backend=backend, now=NOW)
        assert "acquire" not in broker.calls and reopened._maintenance_state == state
        result = me.execute(job, backend=backend, now=NOW)
        assert reopened._maintenance_state.phase == "COMPLETED"
        assert broker.state["max_active"] == 1 and broker.violations == []
        prior, final = broker.state["ledger"]
        assert prior["disposition"] == "aborted" and prior["cost"] == ZERO.to_dict()
        assert prior["token_digest"] == payload["token"]["token_digest"]
        assert final["accounting_receipt_digest"] == result.accounting.receipt_digest
        settled = [event["settlement"] for event in _events(reopened)
                   if event.get("settlement")]
        assert settled == [prior, final]
        assert _held_ms(settled) == _held_ms(broker.state["ledger"])
    finally:
        if reopened._entered and not reopened._maintenance_state.owned:
            reopened.close()


# ---------------------------------------------------------------------------
# AKU-10d: fresh admission from complete owner coverage, and recovery ownership.


def test_fresh_admission_requires_complete_current_owner_coverage(runtime, root):
    controller, prepared, policy, job, old = runtime
    broker = InMemoryMaintenanceBroker(mutex=controller._mutex)
    backend = me.CampaignMaintenanceBackend(controller, prepared, policy, broker)
    records = controller._retention_owner_records
    # Missing class: even an exactly re-captured frontier cannot bind or admit.
    for owner_class in catalog.OWNER_CLASSES:
        saved = records.pop(owner_class)
        frontier = controller.native_retention_catalog_capture()["frontier"]
        with pytest.raises(Exception, match=f"coverage is incomplete: missing:{owner_class}"):
            controller.bind_prepared_retention_job(
                job, prepared=_prepared(prepared.view, frontier), policy=policy)
        with pytest.raises(Exception, match="preparation|coverage"):
            backend.admit(job)
        records[owner_class] = saved
    # Stale class: a durable owner write without its hook keeps the frontier but
    # differs from the owner state re-derived at admission.
    key = ("unhooked", 1)
    controller._retention_worker_results[key] = {
        "worker_id": "unhooked", "worker_generation": 1, "lineage_id": "x",
        "stage_id": "y", "data": {"result_digest": "e" * 64}}
    with pytest.raises(Exception, match="coverage is incomplete: stale:worker_result"):
        backend.admit(job)
    del controller._retention_worker_results[key]
    assert _events(controller) == [] and broker.calls == [] and old.exists()
    # Complete and current again: admission -> execution -> COMPLETED, bytes once.
    size = prepared.view.nodes[-1].declared_size_bytes
    result = me.execute(job, backend=backend, now=NOW)
    assert not old.exists() and controller._maintenance_state.phase == "COMPLETED"
    [row] = broker.state["ledger"]
    assert row["cost"]["reclaimed_bytes"] == result.cost.reclaimed_bytes == size > 0
    assert result.cost.deleted_bytes_this_attempt == size and not result.recovered


def test_owner_gap_without_artifact_store_refuses_binding(root):
    controller, prepared, policy, job, old = _runtime(root, artifact_root=False, bind=False)
    try:
        with pytest.raises(Exception, match="gap:physical_root:artifact_store_undeclared"):
            controller.bind_prepared_retention_job(job, prepared=prepared, policy=policy)
        broker = InMemoryMaintenanceBroker(mutex=controller._mutex)
        backend = me.CampaignMaintenanceBackend(controller, prepared, policy, broker)
        with pytest.raises(Exception, match="preparation"):
            me.execute(job, backend=backend, now=NOW)
        assert _events(controller) == [] and broker.calls == [] and old.exists()
    finally:
        controller.close()


def _crash_during_execute(path, seeded, stage, connection):
    """Forked owner: run execute and die at ``stage`` (io_complete / complete)."""
    controller, prepared, policy, job, _old = _runtime(Path(path), seeded)
    broker = InMemoryMaintenanceBroker(mutex=controller._mutex)
    backend = me.CampaignMaintenanceBackend(controller, prepared, policy, broker)

    def crash(*_args, **_kwargs):
        connection.send({"state": copy.deepcopy(broker.state),
                         "view": _plain_view(prepared.view),
                         "job": (job.plan.to_dict(), list(job.selected_artifact_ids),
                                 job.policy_digest),
                         "size": prepared.view.nodes[-1].declared_size_bytes})
        connection.close()
        os._exit(0)

    setattr(backend, stage, crash)
    me.execute(job, backend=backend, now=NOW)
    os._exit(3)


def _crashed(root, stage):
    seeded = _seed(artifact_root=root / "artifacts")
    context = multiprocessing.get_context("fork")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_crash_during_execute,
                              args=(str(root), seeded, stage, send))
    process.start()
    send.close()
    payload = receive.recv()
    receive.close()
    process.join(10)
    assert process.exitcode == 0
    reopened = _controller(seeded[0], root / "service")
    reopened.__enter__()
    view = _native_view(payload["view"])
    prepared = _prepared(view, reopened.native_retention_catalog_capture()["frontier"])
    policy = _policy(root)
    plan, selected, policy_digest = payload["job"]  # bytes are gone: no re-prepare
    job = rc.RetentionJob(rc.retention.RetentionPlan.from_dict(plan), tuple(selected),
                          policy_digest)
    broker = InMemoryMaintenanceBroker(state=payload["state"], mutex=reopened._mutex,
                                       clock_start=10_000)
    backend = me.CampaignMaintenanceBackend(reopened, prepared, policy, broker)
    return reopened, backend, broker, job, payload


def _reclaimed(rows):
    return sum(row["cost"]["reclaimed_bytes"] for row in rows)


def test_crash_mid_execution_resolves_then_recovers_without_second_delete(root):
    reopened, backend, broker, job, payload = _crashed(root, "io_complete")
    try:
        state = reopened._maintenance_state
        assert state.phase == "UNRESOLVED" and state.owned and state.hold is not None
        assert state.cost is None  # bytes removed, IO_COMPLETE never journaled
        result = me.execute(job, backend=backend, now=NOW)
        assert reopened._maintenance_state.phase == "COMPLETED" and not result.recovered
        assert broker.calls[:2] == ["resolve", "acquire"] and broker.violations == []
        assert broker.state["max_active"] == 1  # never two holds at once
        prior, final = broker.state["ledger"]
        assert prior["disposition"] == "aborted" and prior["cost"] == ZERO.to_dict()
        assert final["disposition"] == "complete"
        assert result.cost.deleted_bytes_this_attempt == 0  # replayed, not re-deleted
        assert _reclaimed(broker.state["ledger"]) == payload["size"] > 0
    finally:
        if not reopened._maintenance_state.owned:
            reopened.close()


def test_broker_settled_complete_recovers_to_completed_without_rerun(root):
    reopened, backend, broker, job, payload = _crashed(root, "complete")
    try:
        state = reopened._maintenance_state
        assert state.phase == "UNRESOLVED" and state.cost is not None
        [settled] = broker.state["ledger"]
        assert settled["disposition"] == "complete"
        result = me.execute(job, backend=backend, now=NOW)
        assert result.recovered and result.retention is None
        assert broker.calls == ["resolve"]  # no acquire, refresh or second settlement
        assert broker.state["ledger"] == [settled]
        assert result.cost.to_dict() == settled["cost"]
        assert result.accounting.receipt_digest == settled["accounting_receipt_digest"]
        events = _events(reopened)
        assert [event["event"] for event in events].count("COMPLETED") == 1
        assert events[-1]["settlement"] == settled
        assert _reclaimed(event["settlement"] for event in events
                          if event.get("settlement")) == payload["size"] > 0
        resolved = reopened.resolved
        reopened.close()
        again = _controller(resolved, root / "service")
        again.__enter__()
        try:
            assert again._maintenance_state.phase == "COMPLETED"
            assert not again._maintenance_state.owned
        finally:
            again.close()
    finally:
        if reopened._entered and not reopened._maintenance_state.owned:
            reopened.close()


def test_journaled_unresolved_completion_recovers_from_broker_settlement(runtime):
    controller, prepared, policy, job, old = runtime
    broker = InMemoryMaintenanceBroker(mutex=controller._mutex)
    backend = me.CampaignMaintenanceBackend(controller, prepared, policy, broker)

    def lost_append(*_args, **_kwargs):
        raise RuntimeError("COMPLETED append lost")

    backend.complete = lost_append
    with pytest.raises(RuntimeError, match="append lost"):
        me.execute(job, backend=backend, now=NOW)
    del backend.complete
    assert _events(controller)[-1]["event"] == "UNRESOLVED" and not old.exists()
    [settled] = broker.state["ledger"]
    calls = len(broker.calls)
    result = me.execute(job, backend=backend, now=NOW)
    assert result.recovered and broker.calls[calls:] == ["resolve"]
    assert broker.state["ledger"] == [settled]
    assert controller._maintenance_state.phase == "COMPLETED"
    token = controller._maintenance_state.token
    original = me.ExclusionToken.from_dict(_events(controller)[0]["token"])
    assert token.predecessor_token_digest == original.token_digest == settled["token_digest"]
    # A recovery INTENT can never carry a complete settlement (that would re-run).
    recovery = replace(original, token_id="r", predecessor_token_digest=original.token_digest,
                       token_digest="")
    with pytest.raises(me.MaintenanceExecutionRefused, match="aborted settlement"):
        me.make_event("INTENT", recovery, occurred_at=NOW.isoformat(),
                      settlement=me.HoldSettlement.from_dict(settled))


def test_uncertain_resolution_keeps_unresolved_and_refuses_admission(runtime):
    controller, prepared, policy, job, old = runtime
    broker = InMemoryMaintenanceBroker(mutex=controller._mutex)
    broker.fail["settle"] = me.MaintenanceExecutionRefused("settlement lost")
    backend = me.CampaignMaintenanceBackend(controller, prepared, policy, broker)
    with pytest.raises(me.MaintenanceExecutionRefused, match="settlement lost"):
        me.execute(job, backend=backend, now=NOW)
    state = controller._maintenance_state
    assert state.phase == "UNRESOLVED" and state.hold is not None and not old.exists()
    count = len(_events(controller))
    other = me.MaintenanceCost(1, 1, 1, 1)
    wrong_complete = me.HoldSettlement(
        broker.PROVIDER_ID, state.hold.hold_id, state.hold.receipt_digest,
        state.token.token_digest, other, "complete",
        me.AccountingReceipt(state.hold.receipt_digest, state.token.token_digest, other,
                             "complete").receipt_digest, 1, 2, 1, 0)
    absent = me.NoHoldReceipt(broker.PROVIDER_ID, me._request_digest(state.token), 1,
                              NOW.isoformat(), "claims absent while a hold is journaled")
    for reply, expected in ((None, "could not prove"), (absent, "does not resolve"),
                            (wrong_complete, "does not settle")):
        broker.resolve = lambda token, hold, reply=reply: reply
        with pytest.raises(Exception, match=expected):
            me.execute(job, backend=backend, now=NOW)
        assert controller._maintenance_state == state and len(_events(controller)) == count
    del broker.resolve
    entered, release = threading.Event(), threading.Event()
    broker.gate["resolve"] = (entered, release)
    bounded = me.BoundedHoldProvider(broker, mutex=controller._mutex, deadline_s=0.3)
    slow = me.CampaignMaintenanceBackend(controller, prepared, policy, bounded)
    try:
        with pytest.raises(me.ProviderDeadlineExceeded, match="resolve exceeded"):
            me.execute(job, backend=slow, now=NOW)
        assert controller._maintenance_state == state and len(_events(controller)) == count
        with pytest.raises(Exception, match="already owned|differs from unresolved"):
            controller.maintenance_admit(replace(job, policy_digest="f" * 64))
    finally:
        release.set()
        for thread in threading.enumerate():
            if thread.name == "maintenance-resolve":
                thread.join(5)
    assert "acquire" not in broker.calls[1:]  # no second hold while uncertain
    # The late resolve settled hold-1 aborted; recovery now proceeds exactly once.
    del broker.gate["resolve"]
    result = me.execute(job, backend=backend, now=NOW)
    assert controller._maintenance_state.phase == "COMPLETED" and not result.recovered
    prior, final = broker.state["ledger"]
    assert prior["disposition"] == "aborted" and final["disposition"] == "complete"
    assert broker.state["max_active"] == 1
    assert result.cost.deleted_bytes_this_attempt == 0
    assert _reclaimed(broker.state["ledger"]) == prepared.view.nodes[-1].declared_size_bytes
