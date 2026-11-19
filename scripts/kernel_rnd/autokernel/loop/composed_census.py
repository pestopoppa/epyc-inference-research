"""AKX-P3a (Spec D): composed census check for a ValidationBatch freeze.

A batch validates the COMPOSED candidate (every kept patch on one tree), not each keep alone.
This pure check binds the composed footprint (all kept patches vs ``production_ref``) and a
per-workload census of that composition to the batch's required full-workload rows, and
refuses to let the batch advance while any required row is pending/failed or lacks an
observed composed census. An INERT classification taken from a single keep's screen may not
stand in for the composed candidate: INERT needs a witness bound to the composed footprint.
"""
from __future__ import annotations

from typing import Any, Mapping

from . import candidate_manifest as cm

SCHEMA = "epyc.autokernel.composed_census.v1"
CLASSES = frozenset({"TOUCHED", "INERT", "UNKNOWN"})
PHASES = ("start", "complete", "advance")


class ComposedCensusRefused(ValueError):
    pass


def census_record(row: Mapping[str, Any], composed_footprint_sha256: str) -> dict[str, Any]:
    """The composed-census entry for one workload from a `blast_radius.classify` row of the
    COMPOSED footprint. INERT carries a witness bound to that composed footprint; a row
    classified from a single keep's footprint (its footprint_sha256 differs) is refused."""
    if row.get("footprint_sha256") != composed_footprint_sha256:
        raise ComposedCensusRefused("classification row was not taken on the composed footprint")
    cls = row.get("classification")
    if cls not in CLASSES:
        raise ComposedCensusRefused(f"unknown classification {cls!r}")
    observed = cls != "UNKNOWN" or row.get("unknown_reason") != "census_stale_or_unobserved"
    rec: dict[str, Any] = {"state": "OBSERVED" if observed else "UNOBSERVED",
                           "classification": cls,
                           "footprint_sha256": composed_footprint_sha256,
                           "tier": row.get("tier")}
    if cls == "INERT":
        rec["witness"] = {"composed_footprint_sha256": composed_footprint_sha256,
                          **dict(row.get("witnesses") or {})}
    return rec


def check(batch: cm.ValidationBatch, row_set: cm.RequiredRowSet, candidate: cm.CandidateManifest,
          footprint: Mapping[str, Any], census: Mapping[str, Mapping[str, Any]],
          keep_footprints: Mapping[str, str] | None = None,
          phase: str = "advance") -> dict[str, Any]:
    """Return ``{"advance": bool, "refusals": [...], ...}``; never raises on a refusal.

    footprint: {production_ref_digest, keep_ids, footprint_sha256} of the composed patches.
    census: {workload_digest: {state: OBSERVED, classification, footprint_sha256,
             witness?: {composed_footprint_sha256}, derived_from_keep?}}.
    keep_footprints: {keep_id: per-keep footprint sha} (to catch per-keep reuse).
    phase: ``start`` (batch freeze) checks the composed footprint only; ``complete`` adds the
    census of every row that passed; ``advance`` (default) also requires every required row
    passed and a census for each."""
    if phase not in PHASES:
        raise ValueError(f"unknown composed census phase {phase!r}")
    batch, row_set, candidate = batch.validated(), cm.RequiredRowSet.from_dict(row_set.to_dict()), \
        candidate.validated()
    out: list[str] = []
    fsha = footprint.get("footprint_sha256")
    if footprint.get("production_ref_digest") != candidate.production_ref_digest:
        out.append("composed footprint is not against the candidate's production_ref")
    if sorted(footprint.get("keep_ids", ())) != sorted(k.keep_id for k in candidate.keeps):
        out.append("composed footprint does not cover exactly the candidate's kept patches")
    if not isinstance(fsha, str) or not fsha:
        out.append("composed footprint has no digest")
    if phase == "start":
        return {"schema": SCHEMA, "batch_id": batch.batch_id, "advance": not out,
                "refusals": out, "phase": phase}
    single = set((keep_footprints or {}).values()) if len(candidate.keeps) > 1 else set()
    states = {s.row_id: s for s in batch.rows}
    rows = {r.row_id: r for r in row_set.rows}
    for row_id in batch.required_row_ids:
        state, row = states.get(row_id), rows.get(row_id)
        if state is None or row is None:
            out.append(f"row {row_id}: absent from the frozen batch/row set")
            continue
        if state.status != "passed":
            if phase == "advance":
                out.append(f"row {row_id}: required row is {state.status}")
            else:
                continue  # completing a failed/inconclusive batch is legal; it cannot advance
        rec = census.get(row.workload_digest)
        if not isinstance(rec, Mapping) or rec.get("state") != "OBSERVED" \
                or rec.get("classification") not in CLASSES:
            out.append(f"row {row_id}: no observed composed census for its workload")
            continue
        if rec.get("footprint_sha256") != fsha:
            out.append(f"row {row_id}: census is not bound to the composed footprint")
        if rec["classification"] == "INERT":
            witness = rec.get("witness")
            if rec.get("derived_from_keep") or rec.get("footprint_sha256") in single \
                    or not isinstance(witness, Mapping) \
                    or witness.get("composed_footprint_sha256") != fsha:
                out.append(f"row {row_id}: per-keep INERT reuse refused; INERT needs a "
                           "witness on the composed candidate")
    return {"schema": SCHEMA, "batch_id": batch.batch_id, "advance": not out,
            "refusals": out, "phase": phase}


def check_evidence(batch: cm.ValidationBatch, row_set: cm.RequiredRowSet,
                   candidate: cm.CandidateManifest, composed: Mapping[str, Any] | None,
                   *, phase: str) -> dict[str, Any]:
    """Run ``check`` on a transition's ``composed`` evidence mapping
    ``{footprint, census, keep_footprints?}``; a transition that carries no evidence is
    unchanged (opt-in until the AKX-P0d census producer feeds every launch)."""
    if composed is None:
        return {"schema": SCHEMA, "batch_id": batch.batch_id, "advance": True,
                "refusals": [], "phase": phase, "enforced": False}
    if not isinstance(composed, Mapping) or not isinstance(composed.get("footprint"), Mapping) \
            or not isinstance(composed.get("census"), Mapping):
        raise ComposedCensusRefused("composed evidence needs footprint and census mappings")
    result = check(batch, row_set, candidate, composed["footprint"], composed["census"],
                   composed.get("keep_footprints"), phase=phase)
    result["enforced"] = True
    if not result["advance"]:
        raise ComposedCensusRefused("; ".join(result["refusals"]))
    return result


def require_advance(*args: Any, **kwargs: Any) -> dict[str, Any]:
    result = check(*args, **kwargs)
    if not result["advance"]:
        raise ComposedCensusRefused("; ".join(result["refusals"]))
    return result


__all__ = ["ComposedCensusRefused", "PHASES", "SCHEMA", "check", "census_record", "check_evidence", "require_advance"]
