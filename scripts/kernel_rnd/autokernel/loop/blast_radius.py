"""AKX-P1e: the section 7.3 TOUCHED/INERT/UNKNOWN classifier for the keep screen.

Derives one row per workload from a patch footprint and that workload's census record,
and assembles the screen that `keep_screen.write_record` publishes. INERT needs a recorded
T0 or T1 witness; UNKNOWN always decides as TOUCHED. Pinned by the AKX-P1d controls in
`evaluator/test_blast_radius_controls.py`. Pure over its inputs: no build, no compute.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from ..evaluator import footprint as FP
from . import keep_screen as KS


@dataclass(frozen=True)
class WorkloadRecord:
    workload_id: str
    backend: str
    tensor_types: frozenset
    census_observed: bool = True
    census_stale: bool = False
    effective_env: Mapping[str, str] = field(default_factory=dict)
    is_author_target: bool = False
    # Recorded witnesses; None means "no such evidence recorded".
    t0_dso_anchor: Optional[Mapping[str, str]] = None
    t0_dso_candidate: Optional[Mapping[str, str]] = None
    t1_profiles: Optional[tuple] = None  # (anchor_sha, candidate_sha, envelope_sha)


def is_no_op(fp: FP.PatchFootprint) -> bool:
    """True iff every hunk is comment-only, so no compiled behaviour can differ."""
    return bool(fp.hunks) and all(h.kind == "comment" for h in fp.hunks)


def _live_hunks(fp: FP.PatchFootprint):
    return [h for h in fp.hunks if h.kind != "comment"]


def _static_types(fp: FP.PatchFootprint) -> frozenset:
    out: set = set()
    for site in fp.dispatch_sites:
        out.update(t.removeprefix("GGML_TYPE_") for t in site.discriminators["types"])
    return frozenset(out)


def exercised_knob_flips(fp: FP.PatchFootprint, wl: WorkloadRecord) -> list:
    """Knobs whose effective default changes for ``wl``.

    A knob that is read but whose default is unchanged is not an exercised change; a
    knob the workload sets explicitly pins its value, so a default flip cannot reach it.
    """
    out = []
    for k in fp.env_knobs:
        if k.default_before == k.default_after:
            continue
        if k.name in wl.effective_env:
            continue
        out.append(k.name)
    return out


def classify(fp: FP.PatchFootprint, wl: WorkloadRecord, *, keep_id: str,
             census_freshness_key: str, created_at: str,
             tool_versions: Mapping[str, str] | None = None) -> dict:
    """One ``blast_radius_row`` for ``wl`` (AKX program section 7.3). UNKNOWN decides as TOUCHED."""
    row: dict[str, Any] = {"schema": KS.ROW_SCHEMA, "keep_id": keep_id,
                           "parent": fp.production_base_commit,
                           "child": fp.candidate_source_commit,
                           "workload_id": wl.workload_id,
                           "census_freshness_key": census_freshness_key,
                           "footprint_sha256": fp.sha256(), "touched_detail": {},
                           "tool_versions": dict(tool_versions or {}), "created_at": created_at,
                           "witnesses": None, "tier": None, "unknown_reason": None}

    def done(cls: str, **kw: Any) -> dict:
        row.update(classification=cls,
                   decision_class="INERT" if cls == "INERT" else "TOUCHED", **kw)
        return row

    if wl.is_author_target:
        return done("TOUCHED", touched_detail={"reason": "author_target"})
    if not fp.hunks:
        raise FP.FootprintError("empty footprint refuses the screen")
    if wl.census_stale or not wl.census_observed:
        return done("UNKNOWN", unknown_reason="census_stale_or_unobserved")
    if wl.t0_dso_anchor and wl.t0_dso_anchor == wl.t0_dso_candidate:
        return done("INERT", tier="T0", witnesses={
            "dso_digests_anchor": dict(wl.t0_dso_anchor),
            "dso_digests_candidate": dict(wl.t0_dso_candidate)})
    if fp.change_class == "build":
        return done("TOUCHED", touched_detail={"reason": "build_variant"})
    flips = exercised_knob_flips(fp, wl)
    if flips:
        return done("TOUCHED", touched_detail={"knob_reads": flips})
    if fp.opaque_reasons:
        return done("UNKNOWN", unknown_reason=";".join(fp.opaque_reasons))
    if is_no_op(fp):
        # No compiled change, but section 7.3 only grants INERT with a witness.
        return done("UNKNOWN", unknown_reason="no_op_without_t0_witness")
    if fp.env_knobs:
        # Read site without a default change.  Unset knob: not an exercised change, so
        # INERT only through a T1 witness (clause iii); a set knob reaches the new code.
        if any(k.name in wl.effective_env for k in fp.env_knobs):
            return done("TOUCHED", touched_detail={
                "knob_reads": [k.name for k in fp.env_knobs]})
        if wl.t1_profiles:
            a, c, env = wl.t1_profiles
            return done("INERT", tier="T1", witnesses={
                "coverage_profile_sha256": [a, c], "shape_envelope_hash": env})
        return done("UNKNOWN", unknown_reason="knob_unset_without_t1_witness")
    live = _live_hunks(fp)
    types = _static_types(fp)
    type_gated = bool(types) and all(h.kind in ("predicate", "table") for h in live) 
    if type_gated and not (types & wl.tensor_types):
        if wl.t1_profiles:
            a, c, env = wl.t1_profiles
            return done("INERT", tier="T1", witnesses={
                "coverage_profile_sha256": [a, c], "shape_envelope_hash": env})
        return done("UNKNOWN", unknown_reason="type_disjoint_without_t1_witness")
    return done("TOUCHED", touched_detail={"executed_hunks": [h.hunk_id for h in live]})


def screen_keep(fp: FP.PatchFootprint, workloads: Sequence[WorkloadRecord], *, keep_id: str,
                census_freshness_keys: Mapping[str, str], created_at: str,
                tool_versions: Mapping[str, str] | None = None) -> tuple[dict, list[dict]]:
    """Classify every workload; return ``(screen, rows)`` for `keep_screen.write_record`.

    A workload with no census freshness key has no resolved census, so this refuses
    rather than publishing a row bound to nothing."""
    if not workloads or len({w.workload_id for w in workloads}) != len(workloads):
        raise ValueError("the workload set is empty or names a workload twice")
    missing = sorted(w.workload_id for w in workloads if w.workload_id not in census_freshness_keys)
    if missing:
        raise ValueError(f"no census freshness key for: {', '.join(missing)}")
    rows = [classify(fp, w, keep_id=keep_id, census_freshness_key=census_freshness_keys[w.workload_id],
                     created_at=created_at, tool_versions=tool_versions) for w in workloads]
    rows.sort(key=lambda row: row["workload_id"])
    by_class = lambda name: [r["workload_id"] for r in rows if r["classification"] == name]
    screen: dict[str, Any] = {"schema": KS.SCREEN_SCHEMA, "keep_id": keep_id,
                              "workload_ids": [r["workload_id"] for r in rows],
                              "rows_ref": [KS._digest(r) for r in rows],
                              "touched_ids": by_class("TOUCHED"), "inert_ids": by_class("INERT"),
                              "unknown_ids": by_class("UNKNOWN")}
    screen["screen_digest"] = KS._digest(screen)
    return screen, rows
