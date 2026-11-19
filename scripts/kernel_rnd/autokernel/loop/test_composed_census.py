from __future__ import annotations

import pytest

from . import candidate_manifest as cm
from . import composed_census as cc
from . import test_candidate_manifest as t

FSHA = "9" * 64


def _fixture(status="passed", census=None, keeps=2):
    comparator = t._base()
    candidate = comparator
    for i in range(keeps):
        candidate = t._next(candidate, i)
    row = t._row("gpu", candidate, comparator)
    proto, rows = t._batch(candidate, comparator, [row])
    state = (cm.ValidationRowState("gpu", "passed", None, t._receipt(proto, "gpu"))
             if status == "passed" else cm.ValidationRowState("gpu", status, "x" if status
             in ("failed",) else None, None))
    batch, _ = t._batch(candidate, comparator, [row], states=[state])
    footprint = {"production_ref_digest": candidate.production_ref_digest,
                 "keep_ids": [k.keep_id for k in candidate.keeps], "footprint_sha256": FSHA}
    good = {row.workload_digest: {"state": "OBSERVED", "classification": "TOUCHED",
                                  "footprint_sha256": FSHA}}
    return batch, rows, candidate, footprint, census if census is not None else good, row


def test_all_required_rows_passed_with_composed_census_advances():
    batch, rows, cand, fp, census, _ = _fixture()
    assert cc.check(batch, rows, cand, fp, census)["advance"] is True
    assert cc.require_advance(batch, rows, cand, fp, census)["refusals"] == []


@pytest.mark.parametrize("status", ["pending", "running", "failed"])
def test_pending_or_failed_required_row_blocks_the_batch(status):
    batch, rows, cand, fp, census, _ = _fixture(status)
    out = cc.check(batch, rows, cand, fp, census)
    assert out["advance"] is False and f"is {status}" in out["refusals"][0]
    with pytest.raises(cc.ComposedCensusRefused):
        cc.require_advance(batch, rows, cand, fp, census)


def test_missing_census_or_wrong_footprint_refuses():
    batch, rows, cand, fp, _, row = _fixture(census={})
    assert "no observed composed census" in cc.check(batch, rows, cand, fp, {})["refusals"][0]
    stale = {row.workload_digest: {"state": "OBSERVED", "classification": "TOUCHED",
                                   "footprint_sha256": "8" * 64}}
    assert not cc.check(batch, rows, cand, fp, stale)["advance"]
    assert not cc.check(batch, rows, cand, {**fp, "keep_ids": ["k0"]}, stale)["advance"]
    assert not cc.check(batch, rows, cand, {**fp, "production_ref_digest": "1" * 64},
                        stale)["advance"]


def test_per_keep_inert_reuse_is_refused():
    batch, rows, cand, fp, _, row = _fixture()
    wl = row.workload_digest
    # INERT inherited from one keep's screen (its footprint, no composed witness)
    reuse = {wl: {"state": "OBSERVED", "classification": "INERT", "footprint_sha256": FSHA,
                  "derived_from_keep": "k0"}}
    out = cc.check(batch, rows, cand, fp, reuse)
    assert not out["advance"] and "per-keep INERT reuse" in out["refusals"][-1]
    keep_sha = "7" * 64
    single = {wl: {"state": "OBSERVED", "classification": "INERT", "footprint_sha256": keep_sha}}
    out = cc.check(batch, rows, cand, fp, single, keep_footprints={"k0": keep_sha, "k1": "6" * 64})
    assert not out["advance"]
    # INERT with a witness bound to the composed footprint is admitted
    ok = {wl: {"state": "OBSERVED", "classification": "INERT", "footprint_sha256": FSHA,
               "witness": {"composed_footprint_sha256": FSHA}}}
    assert cc.check(batch, rows, cand, fp, ok)["advance"]
