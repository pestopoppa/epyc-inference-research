"""Hermetic tests for AKU-10i build enrollment (synthetic fixtures, no hardware)."""
from __future__ import annotations

import copy

import pytest

from .. import journal as journal_module
from . import build_enrollment as be
from . import candidate_manifest as cm
from . import campaign_control as control
from . import candidate_transactions as transactions
from .test_campaign_control import _resolved
from .test_candidate_manifest import _base, _next, _state
from .test_candidate_transactions import _manager


def _intent(candidate, request_id="integrate-1"):
    return {
        "schema": be.INTENT_SCHEMA, "event": "INTENT", "enrollment_id": "enroll-1",
        "campaign_id": "campaign", "integration_request_id": request_id,
        "candidate_manifest_digest": candidate.manifest_digest,
        "build_identity": candidate.builds[0].to_dict(),
        "recipe_snapshot_digest": "a" * 64,
        "source_refs": [item.to_dict() for item in candidate.sources],
        "expiry": {"source_valid_until": 1000.0, "build_valid_until": 2000.0,
                   "evaluation_valid_until": 3000.0},
        "clock_domain": "synthetic", "issued_at": 10.0, "occurred_at": "t0",
    }


def _activated(intent, **over):
    row = {"schema": be.ACTIVATED_SCHEMA, "event": "ACTIVATED",
           "enrollment_id": "enroll-1", "intent_digest": be.digest(intent),
           "integration_request_id": intent["integration_request_id"],
           "candidate_state_digest": "b" * 64,
           "evidence": {k: "c" * 64 for k in be.EVIDENCE_KEYS},
           "activated_at": 50.0, "occurred_at": "t1"}
    row.update(over)
    return row


def test_intent_activation_projection_and_expiry_status_only():
    intent = _intent(_next(_base(), 0))
    proj = be.project_records([intent, _activated(intent)])
    entry = proj["enroll-1"]
    assert entry["activated"] is not None
    assert be.expiry_status(entry, now=1500.0, clock_domain="synthetic") == {
        "source_valid_until": "expired", "build_valid_until": "valid",
        "evaluation_valid_until": "valid"}
    with pytest.raises(be.EnrollmentRefused, match="clock"):
        be.expiry_status(entry, now=1.0, clock_domain="other")


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(extra=1),
    lambda r: r.update(schema="x"),
    lambda r: r.update(recipe_snapshot_digest="zz"),
    lambda r: r["expiry"].update(build_valid_until=5.0),
    lambda r: r["expiry"].pop("source_valid_until"),
    lambda r: r.update(source_refs=[]),
    lambda r: r["build_identity"].update(source_set_digest="d" * 64),
])
def test_intent_is_closed(mutate):
    row = copy.deepcopy(_intent(_next(_base(), 0)))
    mutate(row)
    with pytest.raises(be.EnrollmentRefused):
        be.validate_record(row)


def test_activation_joins_exact_intent_and_closed_evidence():
    intent = _intent(_next(_base(), 0))
    for bad in (_activated(intent, intent_digest="e" * 64),
                _activated(intent, activated_at=1500.0),
                _activated(intent, integration_request_id="other")):
        with pytest.raises(be.EnrollmentRefused):
            be.project_records([intent, bad])
    missing = _activated(intent)
    missing["evidence"].pop("held_cost_ref")
    with pytest.raises(be.EnrollmentRefused, match="evidence"):
        be.validate_record(missing)
    with pytest.raises(be.EnrollmentRefused, match="lacks"):
        be.project_records([_activated(intent)])
    with pytest.raises(be.EnrollmentRefused, match="duplicated"):
        be.project_records([intent, intent])
    with pytest.raises(be.EnrollmentRefused, match="already"):
        be.project_records([intent, _activated(intent), _activated(intent)])


def test_enrolled_integration_persists_pointer_before_commit_and_replays(tmp_path):
    controller, manager = _manager(tmp_path)
    base, candidate = _base(), _next(_base(), 0)
    intent = _intent(candidate)
    try:
        manager.initialize(request_id="init", state=_state(base), manifest=base)
        result = manager.integrate(request_id="integrate-1", previous=base,
                                   candidate=candidate, enrollment=intent)
        assert result["enrollment"] == {"enrollment_id": "enroll-1",
                                        "intent_digest": be.digest(intent)}
        rows = [e.payload for e in controller._journal.read_all()
                if e.kind == journal_module.KIND_CANDIDATE_TRANSACTION
                and e.payload["transaction_id"] == "integrate-1"]
        assert [r["phase"] for r in rows] == ["INTENT", "PREPARED", "COMMITTED"]
        assert rows[0]["data"]["operation_payload"]["enrollment"] == intent
        assert {"kind": "enrollment", "digest": be.digest(intent)} in \
            rows[0]["data"]["prepared_objects"]
        for row in rows:
            assert not journal_module._validate_native_payload(
                journal_module.KIND_CANDIDATE_TRANSACTION, row)
    finally:
        controller.close()
    with control.CampaignController(_resolved(), tmp_path / "service") as again:
        snap = transactions.CandidateTransactions(
            again, git_backend=manager.git_backend).inspect()
        assert snap["state"]["integration_tip"] == candidate.manifest_digest


def test_enrollment_binding_refusals_and_legacy_payload_unchanged(tmp_path):
    controller, manager = _manager(tmp_path)
    base, candidate = _base(), _next(_base(), 0)
    try:
        manager.initialize(request_id="init", state=_state(base), manifest=base)
        for bad in (_intent(candidate, request_id="other"),
                    {**_intent(candidate), "candidate_manifest_digest": "f" * 64}):
            with pytest.raises(transactions.CandidateTransactionError,
                               match="enrollment"):
                manager.integrate(request_id="integrate-1", previous=base,
                                  candidate=candidate, enrollment=bad)
        result = manager.integrate(request_id="integrate-1", previous=base,
                                   candidate=candidate)  # ordinary path
        assert "enrollment" not in result
        intent_row = next(e.payload for e in controller._journal.read_all()
                          if e.kind == journal_module.KIND_CANDIDATE_TRANSACTION
                          and e.payload["transaction_id"] == "integrate-1"
                          and e.payload["phase"] == "INTENT")
        assert set(intent_row["data"]["operation_payload"]) == {
            "previous", "candidate", "threshold_signal", "integration_request_id"}
        assert {o["kind"] for o in intent_row["data"]["prepared_objects"]} == {"manifest"}
    finally:
        controller.close()


def test_journal_validator_refuses_malformed_enrollment_in_intent():
    row = {"schema": journal_module.CANDIDATE_TRANSACTION_SCHEMA, "phase": "INTENT",
           "campaign_id": "c", "config_generation": 1, "config_digest": "1" * 64,
           "supervisor_incarnation": 1, "transaction_id": "t", "operation": "integrate",
           "payload_digest": "2" * 64,
           "data": {"expected_state_digest": None, "prepared_objects": [],
                    "prepared_refs": [],
                    "operation_payload": {"previous": {}, "candidate": {},
                                          "threshold_signal": False,
                                          "integration_request_id": "t",
                                          "enrollment": {"event": "INTENT"}}}}
    out = journal_module._validate_native_payload(
        journal_module.KIND_CANDIDATE_TRANSACTION, row)
    assert any("enrollment" in item for item in out)
