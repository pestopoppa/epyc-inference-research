"""AKU-06l3: v2 child proof across a restart (current, reconciled, restored index)."""
from __future__ import annotations

import dataclasses

import pytest

from .. import journal as journal_module
from . import campaign_control, driver_execution, operation_child_proof as ocp
from . import worker_lifecycle as wl
from .settlement_test_support import SETTLE, settled_receipt
from .test_standalone_runtime import _crash_at, _payloads, _reopen_runtime, _runtime

SHA = "a" * 64


def _terminal(**changes):
    body = dict(worker_id="w", worker_generation=2, request_id="r", plan_digest=SHA,
                lineage_id="l", stage_id="s", grant_id="g", grant_generation=1,
                container_id="c", return_code=1, result_digest=None, accepted=False,
                reason="stale")
    return wl.TerminalWorker(**(body | changes))


class _Receipt:
    def __init__(self, receipt_id="h"):
        self.receipt_id = receipt_id

    def to_dict(self):
        return {"receipt_id": self.receipt_id}


class _Lifecycle:
    def __init__(self, current=None, refuse=False, owned=None):
        self.current, self.refuse, self.owned = current, refuse, owned or {}

    def terminal_for_request(self, **_key):
        if self.refuse:
            raise wl.LifecycleRefused("terminal lookup is ambiguous across worker generations")
        return self.current

    def trusted_held_claim_receipt(self, terminal):
        if terminal not in self.owned:
            raise wl.LifecycleRefused("terminal worker receipt is not owned by this lifecycle")
        return self.owned[terminal]


def _key(terminal):
    return (terminal.request_id, terminal.plan_digest, terminal.lineage_id, terminal.stage_id)


def test_proof_order_current_then_reconciled_then_restored_index():
    terminal, receipt = _terminal(), _Receipt()
    current = _Lifecycle(current=terminal, owned={terminal: receipt})
    assert ocp.child_proof(current, {}, terminal, SHA) == (terminal, receipt)
    reconciled = _Lifecycle(refuse=True, owned={terminal: receipt})
    assert ocp.child_proof(reconciled, {}, terminal, SHA) == (terminal, receipt)
    index = {_key(terminal): (terminal, receipt, None, None)}
    assert ocp.child_proof(_Lifecycle(refuse=True), index, terminal, SHA) == (terminal, receipt)
    bound = {_key(terminal): (terminal, receipt, SHA, None)}
    assert ocp.child_proof(_Lifecycle(), bound, terminal, SHA) == (terminal, receipt)
    assert ocp.child_proof(_Lifecycle(), {}, terminal, SHA) == (None, None)


def test_restored_index_refuses_mismatch_reuse_and_foreign_binding():
    terminal, receipt = _terminal(), _Receipt()
    with pytest.raises(wl.LifecycleRefused, match="ambiguous"):
        ocp.child_proof(_Lifecycle(refuse=True), {}, terminal, SHA)
    for entry, match in (((_terminal(worker_generation=3), receipt, None, None), "differs"),
                         ((terminal, receipt, "b" * 64, None), "another selection"),
                         (None, "reused")):
        with pytest.raises(ocp.ChildProofRefused, match=match):
            ocp.child_proof(_Lifecycle(refuse=True), {_key(terminal): entry}, terminal, SHA)


def test_duplicate_child_is_exact_or_refused():
    terminal, receipt = _terminal(), _Receipt()
    prior = {"request_id": "r", "receipt": receipt.to_dict(), "role": "stage",
             "terminal": ocp.terminal_body(terminal), "binding_digest": SHA, "digest": "d"}
    assert ocp.duplicate_child([], terminal, receipt, "stage", SHA) is None
    assert ocp.duplicate_child([prior], terminal, receipt, "stage", SHA) == {
        "status": "duplicate", "digest": "d", "role": "stage"}
    with pytest.raises(ocp.ChildProofRefused, match="differs"):
        ocp.duplicate_child([prior], terminal, _Receipt(), "stage", "b" * 64)


def test_restored_index_child_settles_v2_and_verifier_refuses_a_changed_proof(
        tmp_path, monkeypatch):
    runtime, controller, lifecycle, engine = _runtime(tmp_path, monkeypatch)
    runtime.recover()
    _crash_at(runtime, lifecycle, "OWNED_HELD_RECEIPT_RETAINED")
    transition_id = controller.unified_driver_pending_intent()["transition_id"]
    first = _reopen_runtime(runtime, controller, engine, monkeypatch, crashed=True)
    monkeypatch.setattr(first[1], SETTLE, lambda _value: (
        _ for _ in ()).throw(OSError("fixture crash before settlement")))
    assert first[0].recover().status == "recovery_required"
    current = _reopen_runtime(*first[:2], first[3], monkeypatch, crashed=True)
    controller = current[1]
    real_settle, checked = controller.unified_driver_settle_v2, []

    def verify_then_settle(request):
        # The child is durably accounted from the restored index; the registered
        # verifier accepts it, and refuses once the index proof no longer matches it.
        verify = current[0].executor._verify_operation_settlement
        assert request["schema"] == campaign_control.DRIVER_SETTLEMENT_SCHEMA_V2
        assert dict(verify(request)) == request
        index = controller._restored_held_accounting
        key, = [k for k, entry in index.items()
                if entry and entry[0].lineage_id == f"driver:{transition_id}"]
        entry = index[key]
        index[key] = (entry[0], dataclasses.replace(entry[1], receipt_id="changed"),
                      *entry[2:])
        with pytest.raises((driver_execution.DriverExecutionRefused,
                            campaign_control.ControlRefused)):
            verify(request)
        index[key] = entry
        checked.append(True)
        return real_settle(request)

    monkeypatch.setattr(controller, "unified_driver_settle_v2", verify_then_settle)
    try:
        assert controller._worker_lifecycle._terminals == {}  # only the restored index
        recovered = current[0].recover()
        assert recovered.status == "settled", recovered.reason
        assert checked == [True]
        settled, = _payloads(controller, journal_module.KIND_UNIFIED_DRIVER_SETTLED)
        held, = [row for row in _payloads(controller, journal_module.KIND_WORKER_LIFECYCLE)
                 if row["event"] == "OWNED_HELD_RECEIPT_RETAINED"]
        assert settled["schema"] == journal_module.UNIFIED_DRIVER_SETTLED_SCHEMA_V2
        assert settled_receipt(controller, settled) == held["data"]["receipt"]
    finally:
        current[0].close()
        current[1].close()
