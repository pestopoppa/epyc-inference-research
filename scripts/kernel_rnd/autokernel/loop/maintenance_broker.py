"""AKU-10g2: the OP-41 maintenance broker over the host lock service.

One maintenance hold is one exclusive host claim (the same ``HostLockBackend`` that
``host_claim_provider`` uses: region lock role ``bench`` or the GPU flock). Hold
identity is the acquired grant id, so a successor can prove a dead holder's claim
released from the kernel lock table alone. ``settle`` is exactly-once and releases
the claim; ``resolve`` aborts a live hold or proves a dead one absent. OFF by default:
the registered factory returns the unavailable ``BrokerHoldProvider`` unless
``ACTIVATION_ENV`` names an operator record bound to this exact source digest.
"""
from __future__ import annotations

import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import stat
import threading
import time

PROVIDER_ID = "maintenance_broker_v1"
ACTIVATION_ENV = "AUTOKERNEL_MAINTENANCE_BROKER_ACTIVATION"
ACTIVATION_SCHEMA = "epyc.autokernel.maintenance_broker_activation.v1"
_FIELDS = {"schema", "provider_id", "source_sha256", "cpu_list"}


def _loop(name: str):
    # Exec'd by provider_factories without a package: use the absolute loop name.
    return importlib.import_module(f"{__package__ or 'scripts.kernel_rnd.autokernel.loop'}.{name}")


me = _loop("maintenance_execution")
hcp = _loop("host_claim_provider")


class _Hold:
    def __init__(self, handle, request, token_digest, acquired_ms, receipt):
        self.handle, self.request, self.token_digest = handle, request, token_digest
        self.acquired_ms, self.issued, self.refreshes = acquired_ms, {receipt.receipt_digest}, 0
        self.receipt, self.settlement = receipt, None


class LockServiceMaintenanceBroker(me.BrokerHoldProvider):
    """BrokerHoldProvider whose holds are real exclusive claims on the lock service."""

    PROVIDER_ID = PROVIDER_ID

    def __init__(self, backend, spec, *, lease_s: float = 600.0, acquire_s: float = 5.0,
                 clock_ms=lambda: time.monotonic_ns() // 1_000_000, wall=time.time):
        if not 0 < lease_s <= 3600 or not 0 < acquire_s <= 60:
            raise me.MaintenanceExecutionRefused("broker lease/acquire bounds are closed")
        self.backend, self.spec, self.lease_s, self.acquire_s = backend, spec, lease_s, acquire_s
        self._clock_ms, self._wall = clock_ms, wall
        self._lock, self._holds = threading.Lock(), {}

    def _no_hold(self, request, reason):
        return me.NoHoldReceipt(PROVIDER_ID, request, 1,
                                datetime.datetime.fromtimestamp(
                                    self._wall(), datetime.timezone.utc).isoformat(), reason)

    def _issue(self, row):
        handle = row.handle
        receipt = me.HoldReceipt(PROVIDER_ID, handle.grant_id, row.request, handle.generation, 1,
                                 self._wall() + self.lease_s, True, False)
        row.issued.add(receipt.receipt_digest)
        row.receipt = receipt
        return receipt

    def acquire(self, token):
        request = me._request_digest(token)
        with self._lock:  # one hold at a time; the claim itself is exclusive too
            if any(row.settlement is None for row in self._holds.values()):
                return self._no_hold(request, "a maintenance hold is already active")
        try:
            handle = self.backend.acquire(self.spec, self.acquire_s)
        except hcp.LockBusy as exc:
            return self._no_hold(request, f"host claim busy: {exc}")
        except hcp.HostClaimRefused as exc:
            raise me.MaintenanceExecutionRefused(f"host claim refused: {exc}") from exc
        try:
            with self._lock:
                row = _Hold(handle, request, token.token_digest, self._clock_ms(),
                            me.HoldReceipt(PROVIDER_ID, handle.grant_id, request,
                                           handle.generation, 1, self._wall() + self.lease_s,
                                           True, False))
                self._holds[handle.grant_id] = row
                return row.receipt
        except BaseException:
            self.backend.release(handle)
            raise

    def refresh(self, hold, token):
        with self._lock:
            row = self._holds.get(hold.hold_id)
            if (row is None or row.settlement is not None
                    or row.request != me._request_digest(token)
                    or hold.receipt_digest not in row.issued):
                raise me.MaintenanceExecutionRefused("broker hold is unknown or settled")
            if self.backend.observe(row.handle) != "held":
                raise me.MaintenanceExecutionRefused("host claim is no longer provably held")
            row.refreshes += 1
            return self._issue(row)

    def _settle_locked(self, row, hold, token, cost, disposition):
        if row.settlement is not None:
            same = row.settlement
            if (same.hold_receipt_digest, same.cost, same.disposition) == (
                    hold.receipt_digest, cost, disposition):
                return same
            raise me.MaintenanceExecutionRefused("broker hold already settled differently")
        if disposition == "complete" and self.backend.observe(row.handle) != "held":
            raise me.MaintenanceExecutionRefused("host claim was lost; completion is uncertified")
        settled = max(self._clock_ms(), row.acquired_ms)
        released = self.backend.release(row.handle)
        if not released and disposition == "complete":
            raise me.MaintenanceExecutionRefused("host claim did not survive to its release")
        row.settlement = me.HoldSettlement(
            PROVIDER_ID, hold.hold_id, hold.receipt_digest, token.token_digest, cost, disposition,
            me.AccountingReceipt(hold.receipt_digest, token.token_digest, cost,
                                 disposition).receipt_digest,
            row.acquired_ms, settled, settled - row.acquired_ms, row.refreshes)
        return row.settlement

    def settle(self, hold, token, cost, disposition):
        with self._lock:
            row = self._holds.get(hold.hold_id)
            if (row is None or hold.receipt_digest not in row.issued
                    or row.token_digest != token.token_digest):
                raise me.MaintenanceExecutionRefused("broker never issued this hold receipt")
            return self._settle_locked(row, hold, token, cost, disposition)

    def resolve(self, token, hold):
        request = me._request_digest(token)
        with self._lock:
            for row in self._holds.values():
                if row.request == request:
                    return row.settlement or self._settle_locked(
                        row, row.receipt, token, me.MaintenanceCost(0, 0, 0, 0), "aborted")
        if hold is None:
            return self._no_hold(request, "no hold was ever issued by this broker; a flock "
                                          "cannot outlive its holder")
        parsed = hcp.parse_grant_id(hold.hold_id)
        if parsed is None:
            raise me.MaintenanceExecutionRefused("hold identity is not a host claim identity")
        boot, pid, ticks, claim_ids = parsed
        if boot != self.backend.current_boot():
            return self._no_hold(request, "kernel flocks do not survive a reboot")
        if self.backend.holder_alive(pid, ticks):
            raise me.MaintenanceExecutionRefused("original holder is alive; hold is unresolved")
        owners = self.backend.owners(claim_ids)
        if owners is None or owners:
            raise me.MaintenanceExecutionRefused("lock table cannot prove the hold released")
        return self._no_hold(request, "holder exited and no flock remains on the claim")


def activation(environ=None):
    """The operator activation record, or (None, reason). Absent by default."""
    raw = (os.environ if environ is None else environ).get(ACTIVATION_ENV)
    if not raw:
        return None, "activation_absent"
    path = Path(raw)
    try:
        info = path.lstat()
        if (not path.is_absolute() or not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid() or info.st_mode & 0o022 or info.st_size > 65536):
            return None, "activation_record_untrusted"
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, "activation_record_unreadable"
    if (not isinstance(record, dict) or set(record) != _FIELDS
            or record["schema"] != ACTIVATION_SCHEMA or record["provider_id"] != PROVIDER_ID
            or not all(isinstance(value, str) and value for value in record.values())):
        return None, "activation_record_invalid"
    if record["source_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        return None, "activation_source_differs"
    return record, "active"


def make_broker():
    """Registered no-argument factory: the unavailable broker unless activated."""
    record, _reason = activation()
    if record is None:
        return me.BrokerHoldProvider()
    return LockServiceMaintenanceBroker(
        hcp.OrchestratorHostLockBackend(), hcp.ClaimSpec("cpu", cpu_list=record["cpu_list"]))
