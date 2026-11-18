#!/usr/bin/env python3
"""R23-25: route a hypothesis to a measurement rung by MECHANISM FAMILY.

A screen-rung (small-model) verdict is informative only for mechanisms whose effect is
size-INDEPENDENT: removing redundant work, algebraic simplification, fewer dispatches.
For size-DEPENDENT families -- launch geometry, occupancy, wave/thread-block counts,
tiling -- the 1.5B screen verdict is uninformative and can be anti-informative (R23-5:
+17.26% on the screen shape became a decisive -1.46% on production shapes), so those go
straight to the confirm rung and accept the higher per-attempt cost.

This is pure classification over the hypothesis text. It never measures, never gates a
keep (the confirm gate in `rung_confirm.py` still owns promotion) and an UNCLASSIFIED
hypothesis keeps today's behaviour (screen first): a guess must not skip the cheap rung.
"""
from __future__ import annotations

from typing import Any, Mapping

SCREEN = "screen"
CONFIRM = "confirm"
UNCLASSIFIED = "unclassified"

#: (family, rung, needles). Order matters: size-dependent first, because a hypothesis that
#: mentions both "fuse" and "tile" must not be waved through the screen on the former.
FAMILIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("launch_geometry", CONFIRM, ("launch bound", "launch_bounds", "block size", "blocksize",
                                  "block dim", "grid size", "thread-block", "thread block",
                                  "threads per block", "nwarps", "num_warps")),
    ("occupancy", CONFIRM, ("occupancy", "waves per", "wave count", "wavefront count",
                            "register pressure", "vgpr", "lds usage", "lds budget")),
    ("tiling", CONFIRM, ("tile size", "tiling", "tile shape", "tile_n", "tile_m",
                         "mmq_x", "mmq_y", "rows per block", "rows_per_block",
                         "cols per block", "split-k", "split_k", "stream-k")),
    ("redundant_work", SCREEN, ("redundant", "recomput", "hoist", "loop-invariant",
                                "loop invariant", "dead store", "duplicate load")),
    ("algebraic", SCREEN, ("algebraic", "strength reduc", "constant fold", "precompute",
                           "simplif", "identity")),
    ("dispatch_count", SCREEN, ("fewer dispatch", "fewer launches", "fuse", "fusion",
                                "fused", "merge kernels", "single launch")),
)


def classify(hypothesis: Mapping[str, Any]) -> tuple[str, str]:
    """`(family, rung)` for a hypothesis-shaped mapping; `("unclassified", "screen")` if
    no family matches."""
    text = " ".join(str(hypothesis.get(k) or "").lower() for k in (
        "mechanism_id", "statement", "target_symbol", "mechanism_family", "falsifier"))
    for family, rung, needles in FAMILIES:
        if any(needle in text for needle in needles):
            return family, rung
    return UNCLASSIFIED, SCREEN


def rung_for(hypothesis: Mapping[str, Any]) -> str:
    return classify(hypothesis)[1]


def skips_screen(hypothesis: Mapping[str, Any]) -> bool:
    return rung_for(hypothesis) == CONFIRM


def route_record(hypothesis: Mapping[str, Any]) -> dict[str, str]:
    family, rung = classify(hypothesis)
    return {"family": family, "rung": rung,
            "reason": ("size-dependent mechanism: the screen-rung verdict is uninformative"
                       if rung == CONFIRM and family != UNCLASSIFIED else
                       "size-independent or unclassified: screen first")}
