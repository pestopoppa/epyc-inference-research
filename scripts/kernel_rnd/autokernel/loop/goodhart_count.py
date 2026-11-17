"""S3-AKU-13: Goodharted-keep count at PROMOTE.

Beside each leave-one-out arm (R23-48) record that keep's bench effect at keep time.
A keep is Goodharted when the bundle WITHOUT it is at least as fast as the full bundle on
the serving surface or on any re-baselined surface: the keep won its screen but adds
nothing (or hurts) once composed. The count goes on the promotion record; INF-75 AKX-P3b
consumes it (intake-1378#record).
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

SCHEMA = "epyc.autokernel.goodhart_count.v1"


def _num(v, what):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"{what} must be a finite number, got {v!r}")
    return float(v)


def count_goodharted(arms: Sequence[Mapping], *, full_bundle: Mapping[str, float]) -> dict:
    """arms: [{"keep_id", "keep_time_effect", "loo": {surface: throughput_without_keep}}].

    `full_bundle` maps surface -> throughput of the full bundle (higher is better); every
    surface in an arm's `loo` must be present there (a surface with no full-bundle number
    cannot be compared, so it REFUSES rather than passing silently). Returns the count,
    the flagged keeps with the surfaces that convicted them, and the keep-time effect
    recorded beside each arm.
    """
    if not arms:
        raise ValueError("no leave-one-out arms")
    rows, flagged = [], []
    for arm in arms:
        kid = arm.get("keep_id")
        if not kid:
            raise ValueError("arm without keep_id")
        loo = arm.get("loo") or {}
        if not loo:
            raise ValueError(f"arm {kid}: no leave-one-out surfaces")
        bad = []
        for surface, v in loo.items():
            if surface not in full_bundle:
                raise ValueError(f"arm {kid}: surface {surface!r} has no full-bundle value")
            if _num(v, f"arm {kid} {surface}") >= _num(full_bundle[surface], f"full bundle {surface}"):
                bad.append(surface)
        row = {"keep_id": kid, "keep_time_effect": _num(arm.get("keep_time_effect"), f"arm {kid} keep_time_effect"),
               "goodharted": bool(bad), "surfaces": sorted(bad)}
        rows.append(row)
        if bad:
            flagged.append(kid)
    return {"schema": SCHEMA, "goodharted_count": len(flagged), "keeps": len(rows),
            "goodharted_keep_ids": flagged, "arms": rows}
