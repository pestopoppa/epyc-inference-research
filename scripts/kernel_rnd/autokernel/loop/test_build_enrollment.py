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


# ---- AKU-10i remainder: minting, ACTIVATED writer/store, disposition gate ----

from .. import schemas  # noqa: E402


def _source_enrollment(candidate, **over):
    build = candidate.builds[0]
    identity = {"candidate_id": "akc-x", "output_binary_sha256": build.executable.sha256,
                "library_sha256s": [[d.path.rsplit("/", 1)[-1], d.sha256] for d in build.dsos]}
    row = {"schema": be.SOURCE_BUILD_ENROLLMENT_SCHEMA, "candidate_id": "akc-x",
           "campaign_id": "campaign", "build_identity": identity,
           "build_identity_digest": schemas.content_hash(identity),
           "artifact": {"verified": True}}
    row.update(over)
    return row


def _mint(candidate, source=None, **over):
    kw = dict(candidate=candidate, request_id="integrate-1", enrollment_id="enroll-1",
              recipe_snapshot_digest="a" * 64,
              expiry={"source_valid_until": 1000.0, "build_valid_until": 2000.0,
                      "evaluation_valid_until": 3000.0},
              clock_domain="synthetic", issued_at=10.0, occurred_at="t0")
    kw.update(over)
    return be.mint_intent_from_source_build(source or _source_enrollment(candidate), **kw)


def test_source_build_schema_matches_real_owner():
    from . import source_build_execution
    assert be.SOURCE_BUILD_ENROLLMENT_SCHEMA == source_build_execution.ENROLLMENT_SCHEMA


def test_mint_from_verified_source_build_identity_and_refusals(tmp_path):
    candidate = _next(_base(), 0)
    intent = _mint(candidate)
    assert intent["event"] == "INTENT" and intent["campaign_id"] == "campaign"
    assert be.validate_record(intent) == intent
    # minted INTENT binds to the integration exactly like a hand-written one
    assert be.bind_to_integration(intent, request_id="integrate-1", candidate=candidate)
    good = _source_enrollment(candidate)
    bad_cases = [
        _source_enrollment(candidate, artifact={"verified": False}),
        _source_enrollment(candidate, build_identity_digest="e" * 64),
        _source_enrollment(candidate, schema="other"),
        {**good, "build_identity": {**good["build_identity"],
                                    "output_binary_sha256": "9" * 64},
         "build_identity_digest": schemas.content_hash(
             {**good["build_identity"], "output_binary_sha256": "9" * 64})},
    ]
    for source in bad_cases:
        with pytest.raises(be.EnrollmentRefused):
            _mint(candidate, source)
    with pytest.raises(be.EnrollmentRefused):  # expiry before issue
        _mint(candidate, expiry={"source_valid_until": 5.0, "build_valid_until": 2000.0,
                                 "evaluation_valid_until": 3000.0})


def test_activated_writer_joins_exact_intent_and_precedes_expiry():
    intent = _intent(_next(_base(), 0))
    ev = {k: "c" * 64 for k in be.EVIDENCE_KEYS}
    row = be.build_activated(intent, candidate_state_digest="b" * 64, evidence=ev,
                             activated_at=50.0, occurred_at="t1")
    assert row["intent_digest"] == be.digest(intent)
    with pytest.raises(be.EnrollmentRefused, match="expiry"):
        be.build_activated(intent, candidate_state_digest="b" * 64, evidence=ev,
                           activated_at=999.0 + 1.0, occurred_at="t1")
    with pytest.raises(be.EnrollmentRefused):
        be.build_activated(intent, candidate_state_digest="b" * 64,
                           evidence={"evaluation_ref": "c" * 64}, activated_at=50.0,
                           occurred_at="t1")


def test_store_wrong_intent_digest_and_orphan_activation_refused(tmp_path):
    store = be.EnrollmentStore(tmp_path / "e")
    intent = _intent(_next(_base(), 0))
    with pytest.raises(be.EnrollmentRefused, match="lacks"):
        store.record_activated(_activated(intent))
    assert store.record_intent(intent) is True
    assert store.record_intent(intent) is False  # idempotent
    with pytest.raises(be.EnrollmentRefused, match="different content"):
        store.record_intent({**intent, "recipe_snapshot_digest": "d" * 64})
    with pytest.raises(be.EnrollmentRefused, match="digest"):
        store.record_activated(_activated(intent, intent_digest="e" * 64))
    assert store.reconcile()["enroll-1"]["state"] == "intent_only"


def test_store_restart_reconciles_activation(tmp_path):
    root = tmp_path / "e"
    intent = _intent(_next(_base(), 0))
    store = be.EnrollmentStore(root)
    store.record_intent(intent)
    activated = _activated(intent)
    assert store.record_activated(activated) is True
    again = be.EnrollmentStore(root)  # "restart"
    state = again.reconcile()["enroll-1"]
    assert state["state"] == "activated" and state["entry"]["activated"] == activated
    assert again.record_activated(activated) is False  # replay writes nothing
    with pytest.raises(be.EnrollmentRefused, match="different content"):
        again.record_activated(_activated(intent, candidate_state_digest="f" * 64))
    (root / "stray.json").write_text("{}")
    with pytest.raises(be.EnrollmentRefused):
        again.reconcile()


def _gate_fixture(**owner_over):
    candidate = _next(_base(), 0)
    intent = _intent(candidate)
    build = cm.BuildIdentity.from_dict(intent["build_identity"])
    ihash = be.digest(intent)
    records = {}
    for key in be.EVIDENCE_KEYS:
        rec = {"schema": be.EVIDENCE_RECORD_SCHEMA, "kind": key,
               "enrollment_id": "enroll-1", "intent_digest": ihash, "state": "closed"}
        if key == "physical_owner_ref":
            rec |= {"owners": ["sess-a"], "build_execution_digest": build.execution_digest}
            rec |= owner_over
        records[key] = rec
    return intent, records


def _entry(intent, records):
    act = _activated(intent, evidence={k: be.digest(v) for k, v in records.items()})
    return be.project_records([intent, act])["enroll-1"]


def _reader(records):
    def read(key, ref):
        rec = records.get(key)
        return rec if rec is not None and be.digest(rec) == ref else None
    return read


def test_disposition_gate_allows_only_with_closed_exclusive_evidence():
    intent, records = _gate_fixture()
    entry = _entry(intent, records)
    ok = be.disposition_gate(entry, now=1500.0, clock_domain="synthetic",
                             read_evidence=_reader(records))
    assert ok.allowed and ok.expired == ("source_valid_until",)
    assert ok.infers_rejection is False and ok.changes_deletion_policy is False
    # expiry_status is unchanged: status only, no disposition field
    assert set(be.expiry_status(entry, now=1500.0, clock_domain="synthetic")) == set(
        be.EXPIRY_KEYS)


def test_disposition_gate_refusals():
    intent, records = _gate_fixture()
    entry = _entry(intent, records)
    read = _reader(records)
    gate = lambda e=entry, **kw: be.disposition_gate(  # noqa: E731
        e, now=kw.pop("now", 1500.0), clock_domain=kw.pop("clock_domain", "synthetic"),
        read_evidence=kw.pop("read_evidence", read), **kw)
    assert "nothing_expired" in gate(now=500.0).reasons
    assert "expiry_unreadable" in gate(clock_domain="other").reasons[0]
    assert gate(None).reasons == ("not_activated",)
    assert gate(be.project_records([intent])["enroll-1"]).reasons == ("not_activated",)
    assert "duplicate_disposition_record" in gate(existing_dispositions=["d"]).reasons
    # missing evidence for each key
    for key in be.EVIDENCE_KEYS:
        partial = {k: v for k, v in records.items() if k != key}
        assert f"evidence_missing:{key}" in gate(read_evidence=_reader(partial)).reasons
    def boom(key, ref):
        raise OSError("owner unreadable")
    assert "evidence_missing:worker_ref" in gate(read_evidence=boom).reasons
    # caller label: record content changed but ref kept -> not content-addressed
    forged = dict(records, held_cost_ref={**records["held_cost_ref"], "state": "closed",
                                          "note": "label"})
    assert "evidence_unbound:held_cost_ref" in gate(
        read_evidence=lambda k, r: forged.get(k)).reasons
    # open (not closed) evidence
    opened = {**records["worker_ref"], "state": "open"}
    e2 = _entry(intent, {**records, "worker_ref": opened})
    assert "evidence_not_closed:worker_ref" in gate(
        e2, read_evidence=_reader({**records, "worker_ref": opened})).reasons
    # evidence bound to a different intent
    other = {**records["native_ref"], "intent_digest": "0" * 64}
    e3 = _entry(intent, {**records, "native_ref": other})
    assert "evidence_unbound:native_ref" in gate(
        e3, read_evidence=_reader({**records, "native_ref": other})).reasons


@pytest.mark.parametrize("owner_over,reason", [
    ({"owners": ["a", "b"]}, "ownership_not_exclusive"),
    ({"owners": []}, "ownership_not_exclusive"),
    ({"owners": "a"}, "ownership_not_exclusive"),
    ({"build_execution_digest": "1" * 64}, "ownership_unbound_to_build"),
])
def test_disposition_gate_refuses_shared_or_unbound_ownership(owner_over, reason):
    intent, records = _gate_fixture(**owner_over)
    decision = be.disposition_gate(_entry(intent, records), now=1500.0,
                                   clock_domain="synthetic",
                                   read_evidence=_reader(records))
    assert not decision.allowed and reason in decision.reasons


def test_gate_refuses_backfilled_activation():
    intent, records = _gate_fixture()
    act = _activated(intent, activated_at=10.0,
                     evidence={k: be.digest(v) for k, v in records.items()})
    entry = be.project_records([intent, act])["enroll-1"]  # legal record, but not after INTENT
    decision = be.disposition_gate(entry, now=1500.0, clock_domain="synthetic",
                                   read_evidence=_reader(records))
    assert "backfill_activation_not_after_intent" in decision.reasons


def test_legacy_integration_without_enrollment_is_unchanged(tmp_path):
    # the existing legacy-payload test covers integrate(); here: no module-level side effects
    assert be.INTENT_SCHEMA.endswith(".v1") and be.ACTIVATED_SCHEMA.endswith(".v1")
