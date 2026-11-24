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
from pathlib import Path
from typing import Mapping, Optional

from . import sampling_axis

FAMILY = "q27b-gpu-dflash2"
#: Registry model whose roles share the :8083 server (architect_critic, coder_escalation, ...).
SERVED_MODEL = "Qwen3.8-27B-Q8_0"

NOT_YET_TRANSFERABLE = "not-yet-transferable"
REGRESSES = "regresses-on-production-shape"
TRANSFERABLE = "transferable"


@dataclass(frozen=True)
class Surface:
    surface_id: str
    kind: str                    # "uniform" | "ragged"
    slot_rows: tuple[int, ...]   # verify rows contributed by each slot
    anchor: bool                 # the width the campaign recipe measures today
    #: REQUIRED (AK-TEMP-AXIS-3): the served temperature this surface's traffic samples at.
    #: :8083 direct agent traffic is 1.0; the registry role defaults on the same server are
    #: `served_temperatures()`. A surface that does not say has no census key.
    temperature: float

    @property
    def ne11(self) -> int:
        """Total activation columns (src1 ne11) of the target verify MUL_MAT."""
        return sum(self.slot_rows)


T8083 = sampling_axis.DIRECT_8083_TEMPERATURE

#: `--spec-draft-n-max 7` => 8 rows per drafting slot (7 drafts + the bonus token).
SURFACES: tuple[Surface, ...] = (
    Surface("n8", "uniform", (8,), anchor=True, temperature=T8083),            # np 1: what the recipe measures today
    Surface("n16", "uniform", (8, 8), anchor=False, temperature=T8083),        # two drafting slots
    Surface("n32", "uniform", (8, 8, 8, 8), anchor=False, temperature=T8083),  # np 4, all slots drafting
    Surface("ragged_1_8_8_8", "ragged", (1, 8, 8, 8), anchor=False, temperature=T8083),  # ne11 25: one slot plain decode
)
BY_ID = {s.surface_id: s for s in SURFACES}


def served_temperatures(registry: "Mapping | Path | None" = None) -> tuple[float, ...]:
    """Every temperature the :8083 27B serves: its roles' registry defaults plus direct traffic."""
    return tuple(sorted({*sampling_axis.role_temperatures(registry, model_name=SERVED_MODEL).values(),
                         T8083}))


def census_keys(registry: "Mapping | Path | None" = None) -> tuple[tuple[str, float], ...]:
    """The verify ne11 / accepted-length census is keyed by (surface, temperature): accepted
    length, and so the rows a verify batch really carries, moves with the sampling regime."""
    return tuple((s.surface_id, t) for s in production_surfaces() for t in served_temperatures(registry))


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


def assess_by_temperature(measured: Mapping[tuple[str, float], Optional[float]], *, floor_pct: float,
                          registry: "Mapping | Path | None" = None) -> Assessment:
    """`assess` over the (surface, temperature) census: every production surface must be measured
    at EVERY served temperature. A surface key whose temperature is not served is a caller error."""
    keys = census_keys(registry)
    unknown = sorted(set(measured) - set(keys) - {(s.surface_id, s.temperature) for s in SURFACES})
    if unknown:
        raise ValueError(f"unknown (surface, temperature) key(s) {unknown}; declared: {sorted(keys)}")
    label = lambda k: f"{k[0]}@T{k[1]:g}"  # noqa: E731
    missing = tuple(label(k) for k in keys if measured.get(k) is None)
    regressed = tuple(label(k) for k in keys if measured.get(k) is not None and measured[k] < -floor_pct)
    verdict = NOT_YET_TRANSFERABLE if missing else REGRESSES if regressed else TRANSFERABLE
    return Assessment(verdict, missing, regressed)


def promotion_refusal(assessment: Assessment) -> Optional[str]:
    """The reason promotion is refused, or None."""
    if assessment.promotable:
        return None
    if assessment.label == NOT_YET_TRANSFERABLE:
        return ("keep is not-yet-transferable: no measurement on production surface(s) "
                f"{', '.join(assessment.missing)} (measured at the anchor width N=8 only, or partially)")
    return f"keep regresses on production surface(s) {', '.join(assessment.regressed)}"


#: AKX-ALL-LB2.1: whether a non-transferable assessment REFUSES a 27B PROMOTE. Default OFF until the operator
#: ratifies the flip (the AKX-ALL-31 NUMERICS_GATES path); off, the label is still recorded on every plan.
GATE_ENV = "AK_PRODUCTION_SURFACES_GATE"
GATE_DEFAULT = False


def gate_enabled() -> bool:
    import os
    value = os.environ.get(GATE_ENV)
    return GATE_DEFAULT if value is None else value == "1"


def keep_assessment(serving_row: Mapping) -> Optional[Assessment]:
    """The label for a serving A/B row of the 27B GPU family (row["recipe"] names the recipe), from its
    `production_surfaces` {surface_id: effect_pct}; None for any other recipe family."""
    if not str(serving_row.get("recipe") or "").startswith(FAMILY):
        return None
    floor = serving_row.get("noise_floor_pct")
    return assess(serving_row.get("production_surfaces") or {}, floor_pct=abs(float(floor or 0.0)))


__all__ = ["GATE_ENV", "gate_enabled", "keep_assessment", "FAMILY", "Surface", "SURFACES", "BY_ID", "Assessment", "assess", "promotion_refusal",
           "production_surfaces", "served_temperatures", "census_keys", "assess_by_temperature", "T8083", "SERVED_MODEL", "NOT_YET_TRANSFERABLE", "REGRESSES", "TRANSFERABLE"]
