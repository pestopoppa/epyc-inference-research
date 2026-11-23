"""AKU-10g2: BrokerHoldProvider contract for the lock-service broker (fake backend)."""
from __future__ import annotations

import hashlib
import itertools
import json
import os

import pytest

from . import host_claim_provider as hcp
from . import maintenance_broker as mb
from . import maintenance_execution as me
from . import provider_factories
from .test_maintenance_broker import ZERO, _events, _runtime, root  # noqa: F401
from .test_retention_consumer import NOW

SPEC = hcp.ClaimSpec("cpu", cpu_list="24-27")


class FakeLockService:
    """In-memory HostLockBackend: exclusive claims, observable loss, dead holders."""

    def __init__(self, boot="boot-1"):
        self.boot, self.held, self.alive, self.lost = boot, {}, {os.getpid(): 7}, set()
        self.gen = itertools.count(1)
        self.busy_with = None
        self.releases = []

    def acquire(self, spec, timeout_s):
        if self.busy_with or any(h is not None for h in self.held.values()):
            raise hcp.LockBusy("region held by another owner")
        n = next(self.gen)
        claim = f"{self.boot}:flock:1:{n}"
        handle = hcp.LockHandle(self.boot, os.getpid(), 7, (claim,), n, 0.25, (), ("24",), n)
        self.held[claim] = handle
        return handle

    def observe(self, handle):
        return "lost" if handle.token in self.lost else (
            "held" if self.held.get(handle.claim_ids[0]) else "unavailable")

    def release(self, handle):
        self.releases.append(handle.token)
        return self.held.pop(handle.claim_ids[0], None) is not None

    def owners(self, claim_ids):
        return tuple(1 for c in claim_ids if self.held.get(c))

    def holder_alive(self, pid, start_ticks):
        return self.alive.get(pid) == start_ticks

    def current_boot(self):
        return self.boot


def _broker(service=None, **kw):
    ticks = itertools.count(1000, 100)
    return mb.LockServiceMaintenanceBroker(service or FakeLockService(), SPEC,
                                           clock_ms=lambda: next(ticks), **kw)


def _token(job, controller=None):
    return me.ExclusionToken(
        "maintenance-x", "c", "a" * 64, 1, "sup", 1, job.plan.snapshot_id,
        job.plan.snapshot_generation, job.plan.snapshot_digest, job.plan.plan_digest,
        job.policy_digest, job.selected_artifact_ids, NOW.isoformat())


def test_full_execution_over_the_lock_service_settles_once_and_releases(runtime_):
    controller, prepared, policy, job, old = runtime_
    service = FakeLockService()
    broker = _broker(service)
    bounded = me.BoundedHoldProvider(broker, mutex=controller._mutex, deadline_s=5)
    backend = me.CampaignMaintenanceBackend(controller, prepared, policy, bounded)
    result = me.execute(job, backend=backend, now=NOW)
    assert not old.exists() and controller._maintenance_state.phase == "COMPLETED"
    [row] = [r for r in broker._holds.values()]
    s = row.settlement
    assert s.disposition == "complete" and s.cost == result.cost and s.refresh_count == 2
    assert s.held_ms == s.settled_at_ms - s.acquired_at_ms > 0
    assert service.held == {} and len(service.releases) == 1  # claim freed exactly once
    assert _events(controller)[-1]["settlement"] == s.to_dict()
    token = controller._maintenance_state.token
    assert broker.settle(row.receipt, token, result.cost, "complete") is s  # idempotent
    with pytest.raises(me.MaintenanceExecutionRefused, match="settled differently"):
        broker.settle(row.receipt, token, ZERO, "aborted")
    assert len(service.releases) == 1


def test_busy_claim_is_an_explicit_no_hold_and_aborts_untouched(runtime_):
    controller, prepared, policy, job, old = runtime_
    service = FakeLockService()
    service.busy_with = "other"
    backend = me.CampaignMaintenanceBackend(controller, prepared, policy, _broker(service))
    with pytest.raises(me.MaintenanceExecutionRefused, match="explicitly refused"):
        me.execute(job, backend=backend, now=NOW)
    assert old.exists() and controller._maintenance_state.phase == "ABORTED"


def test_lost_claim_refuses_refresh_and_uncertifies_completion():
    broker = _broker()
    service, token = broker.backend, _job_token()
    hold = broker.acquire(token)
    service.lost.add(1)
    with pytest.raises(me.MaintenanceExecutionRefused, match="no longer provably held"):
        broker.refresh(hold, token)
    with pytest.raises(me.MaintenanceExecutionRefused, match="uncertified"):
        broker.settle(hold, token, ZERO, "complete")
    aborted = broker.settle(hold, token, ZERO, "aborted")  # abort is always settleable
    assert aborted.disposition == "aborted" and service.held == {}


def _job_token():
    return _token(type("J", (), {
        "plan": type("P", (), {"snapshot_id": "s", "snapshot_generation": 1,
                               "snapshot_digest": "d" * 64, "plan_digest": "e" * 64}),
        "policy_digest": "f" * 64, "selected_artifact_ids": ("a",)}))


def test_one_hold_at_a_time_and_unknown_receipts_are_refused():
    broker = _broker()
    token = _job_token()
    hold = broker.acquire(token)
    assert isinstance(hold, me.HoldReceipt)
    assert isinstance(broker.acquire(token), me.NoHoldReceipt)
    forged = me.HoldReceipt(mb.PROVIDER_ID, hold.hold_id, hold.request_digest, 1, 1, 9e9,
                            True, False)
    with pytest.raises(me.MaintenanceExecutionRefused, match="never issued"):
        broker.settle(forged, token, ZERO, "aborted")
    with pytest.raises(me.MaintenanceExecutionRefused, match="unknown or settled"):
        broker.refresh(forged, token)
    fresh = broker.refresh(hold, token)
    assert fresh.receipt_digest != hold.receipt_digest
    settled = broker.settle(fresh, token, ZERO, "aborted")
    with pytest.raises(me.MaintenanceExecutionRefused, match="unknown or settled"):
        broker.refresh(fresh, token)
    assert broker.settle(fresh, token, ZERO, "aborted") == settled
    assert isinstance(broker.acquire(token), me.HoldReceipt)  # slot free again


def test_resolve_aborts_live_hold_and_proves_dead_holders_absent():
    service = FakeLockService()
    broker, token = _broker(service), _job_token()
    hold = broker.acquire(token)
    resolved = broker.resolve(token, hold)
    assert isinstance(resolved, me.HoldSettlement) and resolved.disposition == "aborted"
    assert broker.resolve(token, hold) == resolved and service.held == {}
    # A successor broker (new process) sees only the journaled hold and the lock table.
    dead = FakeLockService()
    h2 = _broker(dead).acquire(token)
    dead.alive.clear()
    dead.held.clear()
    absent = _broker(dead).resolve(token, h2)
    assert isinstance(absent, me.NoHoldReceipt) and "no flock remains" in absent.reason
    dead.alive[os.getpid()] = 7  # holder alive: ownership is not transferable
    with pytest.raises(me.MaintenanceExecutionRefused, match="holder is alive"):
        _broker(dead).resolve(token, h2)
    dead.alive.clear()
    dead.held[h2.hold_id.split("|")[-1]] = object()  # flock still owned by someone
    with pytest.raises(me.MaintenanceExecutionRefused, match="cannot prove"):
        _broker(dead).resolve(token, h2)
    dead.boot = "boot-2"  # reboot: kernel flocks are gone
    assert "reboot" in _broker(dead).resolve(token, h2).reason
    assert isinstance(_broker(dead).resolve(token, None), me.NoHoldReceipt)
    bogus = me.HoldReceipt(mb.PROVIDER_ID, "not-a-grant", h2.request_digest, 1, 1, 9e9,
                           True, False)
    with pytest.raises(me.MaintenanceExecutionRefused, match="not a host claim"):
        _broker(dead).resolve(token, bogus)


def test_bounds_are_closed():
    for kw in ({"lease_s": 0}, {"lease_s": 4000}, {"acquire_s": 0}, {"acquire_s": 61}):
        with pytest.raises(me.MaintenanceExecutionRefused, match="closed"):
            _broker(**kw)


def test_factory_is_registered_pinned_and_activation_gated(tmp_path, monkeypatch):
    entry = provider_factories.REGISTERED_FACTORIES["maintenance_broker_v1"]
    path, name = provider_factories.source("maintenance_broker_v1")
    assert (path.name, name) == ("maintenance_broker.py", "make_broker")
    assert entry[2] == hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.delenv(mb.ACTIVATION_ENV, raising=False)
    inert = provider_factories.construct("maintenance_broker_v1")
    # Names, not identity: the factory imports the loop under its absolute package name.
    assert type(inert).__name__ == "BrokerHoldProvider"  # unavailable; callers refuse it
    with pytest.raises(Exception, match="broker is unavailable"):
        inert.settle(None, None, ZERO, "aborted")
    base = {"schema": mb.ACTIVATION_SCHEMA, "provider_id": mb.PROVIDER_ID,
            "source_sha256": entry[2], "cpu_list": "24-27"}

    def write(record, mode=0o600):
        f = tmp_path / "act.json"
        f.write_text(json.dumps(record))
        f.chmod(mode)
        return {mb.ACTIVATION_ENV: str(f)}

    assert mb.activation(write(base))[1] == "active"
    assert mb.activation(write({**base, "source_sha256": "0" * 64}))[1] == "activation_source_differs"
    assert mb.activation(write(base, 0o666))[1] == "activation_record_untrusted"
    assert mb.activation(write({**base, "extra": "x"}))[1] == "activation_record_invalid"
    assert mb.activation({mb.ACTIVATION_ENV: "rel.json"})[1] != "active"
    monkeypatch.setenv(mb.ACTIVATION_ENV, write(base)[mb.ACTIVATION_ENV])
    live = provider_factories.construct("maintenance_broker_v1")
    assert type(live).__name__ == "LockServiceMaintenanceBroker"
    assert type(live.backend).__name__ == "OrchestratorHostLockBackend"  # never exercised here


@pytest.fixture
def runtime_(root):  # noqa: F811
    values = _runtime(root)
    try:
        yield values
    finally:
        if values[0]._entered and not values[0]._maintenance_state.owned:
            values[0].close()
