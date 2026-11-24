"""AK-EVAL-NPLAN: pre-register a comparison's pair count from stored noise.

The smallest n in [n_min, n_max] whose calibrated floor `floor_at(n)` is below the
hypothesis's declared target effect (`expected_outcome.min_effect_fraction`), n_min
when no effect is declared. Chosen ONCE, before the first pair, and recorded as
`pairs_basis`; nothing here looks at outcomes, so N never moves after measuring
starts. An uncalibrated surface refuses: there is no floor to plan against.
"""
from __future__ import annotations

from typing import Callable

SCHEMA = "epyc.autokernel.pairs_basis.v1"
N_MIN = 5
N_MAX = 20


class Uncalibrated(ValueError):
    """No calibrated floor for this surface: pair planning refuses."""


def target_effect_pct(hypothesis) -> float | None:
    """The planner's declared minimum effect, in percent, or None."""
    expected = getattr(hypothesis, "expected_outcome", None)
    value = expected.get("min_effect_fraction") if isinstance(expected, dict) else None
    if type(value) not in (int, float) or value <= 0:
        return None
    return 100.0 * float(value)


def plan_pairs(floor_at: Callable[[int], float | None], target_pct: float | None, *,
               n_min: int = N_MIN, n_max: int = N_MAX) -> dict:
    if not 1 <= n_min <= n_max:
        raise ValueError("need 1 <= n_min <= n_max")
    floor_min = floor_at(n_min)
    if floor_min is None:
        raise Uncalibrated("surface has no calibrated floor; pairs cannot be planned")
    basis = {"schema": SCHEMA, "n_min": n_min, "n_max": n_max, "target_effect_pct": target_pct,
             "fixed_before_measurement": True}
    if target_pct is None:
        return {**basis, "pairs": n_min, "floor_pct": floor_min, "rule": "n_min_no_declared_effect"}
    for n in range(n_min, n_max + 1):
        floor = floor_at(n)
        if floor is not None and floor < target_pct:
            return {**basis, "pairs": n, "floor_pct": floor, "rule": "smallest_n_floor_below_target"}
    return {**basis, "pairs": n_max, "floor_pct": floor_at(n_max),
            "rule": "n_max_cap_target_below_floor"}
