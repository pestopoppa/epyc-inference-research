"""Production-shape verify surfaces for the 27B GPU recipe, and the transferability label
(AKX-ALL-LB2, intake-2386).

WHY A SIDECAR AND NOT A RECIPE FIELD.  The campaign recipe (`q27b-gpu-dflash2-np1.recipe.json`)
is a `serving.Recipe`: a frozen dataclass built with `cls(**d)` (an unknown key raises) whose
`recipe_hash` covers every field, so adding a field would both refuse to load and invalidate
every serving floor calibrated under that recipe. The surfaces are therefore declared HERE,
keyed by recipe family, and read by whatever labels a keep.

THE GAP.  The 27B GPU campaign measures ONE verify width: np 1 with `--spec-draft-n-max 7`,
i.e. ne11 = 8. Production (:8083) runs np 4 with the same n-max, so a target verify batch is
the SUM of the slots' rows -- up to 4 x 8 = 32 -- and slots drift out of step, so it is also
ragged (a slot that drafted nothing contributes 1 row). Matmul winners CHANGE with row count
(Locals: 8 vs 16-24 vs 25-32), and the anchor's own route flips at ne11 = 5 (MMVQ dp4a -> MMQ
int8-MFMA stream-k) and again at ne11 >= 9 on the K-quants. A keep measured only at N=8 says
nothing about N=16, N=32 or the ragged mix.

THE RULE.  A keep measured only at the anchor width N=8 is labelled `not-yet-transferable` and
cannot be promoted. It becomes `transferable` only when EVERY production surface has a
measurement and none regresses past the declared floor; a measured regression on any production
surface is `regresses-on-production-shape`, which also refuses promotion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

FAMILY = "q27b-gpu-dflash2"

NOT_YET_TRANSFERABLE = "not-yet-transferable"
REGRESSES = "regresses-on-production-shape"
TRANSFERABLE = "transferable"


@dataclass(frozen=True)
class Surface:
    surface_id: str
    kind: str                    # "uniform" | "ragged"
    slot_rows: tuple[int, ...]   # verify rows contributed by each slot
    anchor: bool                 # the width the campaign recipe measures today

    @property
    def ne11(self) -> int:
        """Total activation columns (src1 ne11) of the target verify MUL_MAT."""
        return sum(self.slot_rows)


#: `--spec-draft-n-max 7` => 8 rows per drafting slot (7 drafts + the bonus token).
SURFACES: tuple[Surface, ...] = (
    Surface("n8", "uniform", (8,), anchor=True),            # np 1: what the recipe measures today
    Surface("n16", "uniform", (8, 8), anchor=False),        # two drafting slots
    Surface("n32", "uniform", (8, 8, 8, 8), anchor=False),  # np 4, all slots drafting
    Surface("ragged_1_8_8_8", "ragged", (1, 8, 8, 8), anchor=False),  # ne11 25: one slot plain decode
)
BY_ID = {s.surface_id: s for s in SURFACES}


def production_surfaces() -> tuple[Surface, ...]:
    """Every surface the anchor-width measurement does not cover."""
    return tuple(s for s in SURFACES if not s.anchor)


@dataclass(frozen=True)
class Assessment:
    label: str
    missing: tuple[str, ...]
    regressed: tuple[str, ...]

    @property
    def promotable(self) -> bool:
        return self.label == TRANSFERABLE


def assess(measured: Mapping[str, Optional[float]], *, floor_pct: float) -> Assessment:
    """Label a keep from its per-surface relative effects (percent, higher = better).

    `measured` maps surface_id -> effect, or None / absent for "not measured". `floor_pct` is the
    predeclared noise floor for the metric (positive percent): an effect below `-floor_pct` on a
    production surface is a regression. An unknown surface id is a caller error, not ignored.
    """
    if floor_pct < 0:
        raise ValueError("floor_pct is a magnitude and must be >= 0")
    unknown = sorted(set(measured) - set(BY_ID))
    if unknown:
        raise ValueError(f"unknown surface id(s) {unknown}; declared: {sorted(BY_ID)}")
    prod = production_surfaces()
    missing = tuple(s.surface_id for s in prod if measured.get(s.surface_id) is None)
    regressed = tuple(s.surface_id for s in prod
                      if measured.get(s.surface_id) is not None and measured[s.surface_id] < -floor_pct)
    if missing:
        return Assessment(NOT_YET_TRANSFERABLE, missing, regressed)
    if regressed:
        return Assessment(REGRESSES, missing, regressed)
    return Assessment(TRANSFERABLE, missing, regressed)


def promotion_refusal(assessment: Assessment) -> Optional[str]:
    """The reason promotion is refused, or None."""
    if assessment.promotable:
        return None
    if assessment.label == NOT_YET_TRANSFERABLE:
        return ("keep is not-yet-transferable: no measurement on production surface(s) "
                f"{', '.join(assessment.missing)} (measured at the anchor width N=8 only, or partially)")
    return f"keep regresses on production surface(s) {', '.join(assessment.regressed)}"


__all__ = ["FAMILY", "Surface", "SURFACES", "BY_ID", "Assessment", "assess", "promotion_refusal",
           "production_surfaces", "NOT_YET_TRANSFERABLE", "REGRESSES", "TRANSFERABLE"]
