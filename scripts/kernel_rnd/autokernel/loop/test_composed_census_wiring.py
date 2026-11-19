"""AKX-P3a: composed census wired into start_batch / complete_batch / advance_validated."""
from __future__ import annotations

from dataclasses import replace

import pytest

from . import candidate_manifest as cm
from . import candidate_transactions as transactions
from . import composed_census as cc
from . import test_candidate_manifest as t
from .test_candidate_transactions import _manager

FSHA = "9" * 64


def _setup():
    base = t._base()
    cand = t._next(base, 0)
    row = t._row("gpu", cand, base)
    rows = t._row_set([row])
    proto, _ = t._batch(cand, base, [row])
    passed = cm.ValidationRowState("gpu", "passed", None, t._receipt(proto, "gpu"))
    batch, _ = t._batch(cand, base, [row], states=[passed])
    fp = {"production_ref_digest": cand.production_ref_digest,
          "keep_ids": [k.keep_id for k in cand.keeps], "footprint_sha256": FSHA}
    census = {row.workload_digest: {"state": "OBSERVED", "classification": "TOUCHED",
                                    "footprint_sha256": FSHA}}
    return base, cand, row, rows, batch, {"footprint": fp, "census": census}


def _started(base, cand, rows, batch, composed=None):
    pending = replace(batch, rows=tuple(cm.ValidationRowState(r.row_id, "pending", None, None)
                                        for r in batch.rows)).validated()
    state = replace(t._state(base), integration_tip=cand.manifest_digest)
    return cm.start_batch(state, pending, rows, cand, base, composed=composed), pending


def test_start_refuses_footprint_not_against_production_ref_or_wrong_keeps():
    base, cand, _, rows, batch, ev = _setup()
    bad = {**ev, "footprint": {**ev["footprint"], "production_ref_digest": "1" * 64}}
    with pytest.raises(cm.TransitionError, match="composed census refused"):
        _started(base, cand, rows, batch, bad)
    bad = {**ev, "footprint": {**ev["footprint"], "keep_ids": ["only-one"]}}
    with pytest.raises(cm.TransitionError, match="exactly the candidate"):
        _started(base, cand, rows, batch, bad)
    # start does not demand a census or passed rows (they cannot exist at freeze)
    state, _ = _started(base, cand, rows, batch, {"footprint": ev["footprint"], "census": {}})
    assert len(state.active_batches) == 1


def test_legacy_transitions_without_evidence_are_unchanged():
    base, cand, _, rows, batch, _ = _setup()
    state, _ = _started(base, cand, rows, batch)
    assert state.active_batches


def test_complete_requires_census_for_passed_rows_but_allows_failed_rows():
    base, cand, row, rows, batch, ev = _setup()
    state, pending = _started(base, cand, rows, batch, ev)
    with pytest.raises(cm.TransitionError, match="no observed composed census"):
        cm.complete_batch(state, batch, composed={**ev, "census": {}}, row_set=rows,
                          candidate=cand)
    with pytest.raises(cm.TransitionError, match="needs the row set"):
        cm.complete_batch(state, batch, composed=ev)
    done = cm.complete_batch(state, batch, composed=ev, row_set=rows, candidate=cand)
    assert done.completed_batches
    failed = replace(batch, rows=(cm.ValidationRowState("gpu", "failed", "x", None),)).validated()
    state2, _ = _started(base, cand, rows, failed, ev)
    assert cm.complete_batch(state2, failed, composed={**ev, "census": {}}, row_set=rows,
                             candidate=cand).completed_batches


def test_advance_refuses_per_keep_inert_reuse_and_stale_census():
    base, cand, row, rows, batch, ev = _setup()
    state, _ = _started(base, cand, rows, batch, ev)
    state = cm.complete_batch(state, batch, composed=ev, row_set=rows, candidate=cand)
    kw = dict(verifier=lambda *_: True, **t._loo_kwargs(cand, rows))
    inert = {**ev, "census": {row.workload_digest: {
        "state": "OBSERVED", "classification": "INERT", "footprint_sha256": FSHA,
        "derived_from_keep": "k0"}}}
    with pytest.raises(cm.TransitionError, match="per-keep INERT reuse refused"):
        cm.advance_validated(state, batch, rows, cand, base, composed=inert, **kw)
    stale = {**ev, "census": {row.workload_digest: {
        "state": "OBSERVED", "classification": "TOUCHED", "footprint_sha256": "8" * 64}}}
    with pytest.raises(cm.TransitionError, match="not bound to the composed footprint"):
        cm.advance_validated(state, batch, rows, cand, base, composed=stale, **kw)
    ok = cm.advance_validated(state, batch, rows, cand, base, composed=ev, **kw)
    assert ok.validated_candidate == cand.manifest_digest


def test_transaction_manager_threads_composed_through_all_three_operations(tmp_path):
    base, cand, row, rows, batch, ev = _setup()
    controller, manager = _manager(tmp_path, verifiers=None)
    pending = replace(batch, accounted_keeps_since_gate=1, rows=(cm.ValidationRowState("gpu", "pending", None, None),)).validated()
    batch = replace(batch, accounted_keeps_since_gate=1).validated()
    try:
        manager.initialize(request_id="init", state=t._state(base), manifest=base)
        manager.integrate(request_id="integrate", previous=base, candidate=cand)
        with pytest.raises(cm.TransitionError, match="composed census refused"):
            manager.start_batch(request_id="start-bad", batch=pending, row_set=rows,
                                candidate=cand, comparator=base,
                                composed={**ev, "footprint": {**ev["footprint"],
                                                              "keep_ids": ["x"]}})
        manager.start_batch(request_id="start", batch=pending, row_set=rows,
                            candidate=cand, comparator=base, composed=ev)
        manager.record_row(request_id="row", batch_id=batch.batch_id, row_set=rows,
                           row_state=batch.rows[0])
        with pytest.raises(transactions.CandidateTransactionError, match="row set and candidate"):
            manager.complete_batch(request_id="c0", batch=batch, composed=ev)
        with pytest.raises(cm.TransitionError, match="no observed composed census"):
            manager.complete_batch(request_id="c1", batch=batch, composed={**ev, "census": {}},
                                   row_set=rows, candidate=cand)
        manager.complete_batch(request_id="c2", batch=batch, composed=ev, row_set=rows,
                               candidate=cand)
        # replay of the durable log re-derives the same state with the composed payloads
        assert manager.inspect()["state"]["completed_batches"]
    finally:
        controller.close()


def test_check_phases_are_validated():
    base, cand, _, rows, batch, ev = _setup()
    with pytest.raises(ValueError):
        cc.check(batch, rows, cand, ev["footprint"], ev["census"], phase="bogus")
    with pytest.raises(cc.ComposedCensusRefused):
        cc.check_evidence(batch, rows, cand, {"footprint": {}}, phase="start")


def test_census_record_binds_the_composed_footprint_and_refuses_per_keep_rows():
    row = {"classification": "INERT", "tier": "T1", "footprint_sha256": FSHA,
           "witnesses": {"shape_envelope_hash": "e" * 64}}
    rec = cc.census_record(row, FSHA)
    assert rec["state"] == "OBSERVED" and rec["witness"]["composed_footprint_sha256"] == FSHA
    assert rec["witness"]["shape_envelope_hash"] == "e" * 64
    with pytest.raises(cc.ComposedCensusRefused, match="not taken on the composed footprint"):
        cc.census_record({**row, "footprint_sha256": "8" * 64}, FSHA)
    with pytest.raises(cc.ComposedCensusRefused):
        cc.census_record({**row, "classification": "MAYBE"}, FSHA)
    touched = cc.census_record({"classification": "TOUCHED", "footprint_sha256": FSHA}, FSHA)
    assert "witness" not in touched
    unk = cc.census_record({"classification": "UNKNOWN", "footprint_sha256": FSHA,
                            "unknown_reason": "census_stale_or_unobserved"}, FSHA)
    assert unk["state"] == "UNOBSERVED"
