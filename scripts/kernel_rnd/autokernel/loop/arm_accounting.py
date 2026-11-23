"""AKU-23: arm-seconds budgets and utilisation (held vs idle-while-claimed).

Pure accounting; no locks, no claims, no compute. A claim is HELD for `claimed_s`
(wall time between acquire and release). Of that, `held_s` is time the arm's work was
actually running; the rest is `idle_while_claimed_s` (spawn, model load, build waits,
gaps). Utilisation = held_s / claimed_s. Budgets are counted in ARM-SECONDS (claimed
time), never iterations: a CPU arm costs ~10x a GPU arm (reload + eviction).

Held time is only ever supplied by a source that observed it. A source that cannot
(counter unavailable, no intervals) yields `utilisation=None` with a reason; nothing
here substitutes wall time for held time -- `Comparison.device_seconds` includes spawn
and load, so it could never support an idle fraction.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping

SCHEMA = "epyc.autokernel.arm_utilisation.v1"
BUDGET_SCHEMA = "epyc.autokernel.arm_seconds_budget.v1"


def _nonneg(value, label):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or float(value) < 0):
        raise ValueError(f"{label} must be a finite number >= 0")
    return float(value)


def account(claimed_s, held_s, *, source: str, unavailable_reason: str | None = None) -> dict:
    """One arm's row. `held_s=None` (with a reason) records an unmeasured arm honestly."""
    claimed = _nonneg(claimed_s, "claimed_s")
    row = {"schema": SCHEMA, "arm_seconds_claimed": round(claimed, 3), "source": source,
           "arm_seconds_held": None, "arm_seconds_idle_while_claimed": None,
           "utilisation": None, "idle_fraction_while_claimed": None,
           "unavailable_reason": unavailable_reason}
    if held_s is None:
        row["unavailable_reason"] = unavailable_reason or "held time not observed"
        return row
    held = _nonneg(held_s, "held_s")
    if held > claimed * 1.01 + 0.5:  # a source cannot have been busy longer than the claim lasted
        row["unavailable_reason"] = f"held {held:.1f}s exceeds claimed {claimed:.1f}s"
        return row
    held = min(held, claimed)
    row.update(arm_seconds_held=round(held, 3), unavailable_reason=None,
               arm_seconds_idle_while_claimed=round(claimed - held, 3))
    if claimed > 0:
        row.update(utilisation=round(held / claimed, 4),
                   idle_fraction_while_claimed=round(1 - held / claimed, 4))
    return row


def merged_seconds(intervals: Iterable[tuple[float, float]], lo: float, hi: float) -> float:
    """Union length of [start, end] intervals clipped to [lo, hi]; overlaps count once."""
    clipped = sorted((max(lo, float(a)), min(hi, float(b))) for a, b in intervals
                     if float(b) > float(a))
    total, cursor = 0.0, lo
    for a, b in clipped:
        if b <= a:
            continue
        a = max(a, cursor)
        if b > a:
            total += b - a
            cursor = b
    return total


def from_intervals(claim_start: float, claim_end: float,
                   intervals: Iterable[tuple[float, float]], *, source: str = "work_intervals") -> dict:
    """Held = union of the arm's own work intervals within the claim (CPU arms, benches)."""
    if claim_end < claim_start:
        raise ValueError("claim_end precedes claim_start")
    return account(claim_end - claim_start,
                   merged_seconds(intervals, claim_start, claim_end), source=source)


def from_gfx_trace(claimed_s, trace: Mapping | None) -> dict:
    """GPU arm: held = `approx_gfx_activity_weighted_seconds` from the gpu_counter trace.

    That is weighted, device-wide GFX activity (not a binary busy duration), and the
    source label says so. An unavailable trace stays unavailable.
    """
    if not isinstance(trace, Mapping):
        return account(claimed_s, None, source="gfx_counter_weighted",
                       unavailable_reason="no gfx counter trace")
    seconds = trace.get("approx_gfx_activity_weighted_seconds")
    if seconds is None:
        return account(claimed_s, None, source="gfx_counter_weighted",
                       unavailable_reason=str(trace.get("unavailable_reason") or "trace reduced to nothing")[:256])
    return account(claimed_s, seconds, source="gfx_counter_weighted")


def from_cpu_work(claimed_s, intervals: Iterable[tuple[float, float]] | None = None,
                  claim_start: float | None = None) -> dict:
    """CPU arm: held = union of the arm's OBSERVED work intervals inside the claim.

    No interval source on the path (None) records `None` plus a reason, never wall time.
    """
    if intervals is None or claim_start is None:
        return account(claimed_s, None, source="cpu_work_intervals",
                       unavailable_reason="no observed CPU work-interval source on this path")
    return from_intervals(claim_start, claim_start + _nonneg(claimed_s, "claimed_s"), intervals,
                          source="cpu_work_intervals")


def wall_only(wall_s, source: str = "comparison_wall_only") -> dict:
    """A comparison row whose caller supplied no observed held-time source."""
    return account(wall_s, None, source=source, unavailable_reason=(
        "no held-time source observed for this comparison; device_seconds is wall time"))


def total(rows: Iterable[Mapping]) -> dict:
    """Roll arms up. Utilisation covers only arms whose held time was observed."""
    rows = list(rows)
    claimed = sum(r["arm_seconds_claimed"] for r in rows)
    seen = [r for r in rows if r.get("arm_seconds_held") is not None]
    sc, sh = sum(r["arm_seconds_claimed"] for r in seen), sum(r["arm_seconds_held"] for r in seen)
    return {"schema": SCHEMA, "arms": len(rows), "arms_measured": len(seen),
            "arm_seconds_claimed": round(claimed, 3),
            "arm_seconds_held": round(sh, 3) if seen else None,
            "utilisation": round(sh / sc, 4) if sc > 0 else None}


class ArmSecondsBudget:
    """Per-surface budgets in arm-seconds. `charge` records claimed time; `allows`
    answers whether a planned arm fits. Raises nothing on exhaustion -- the caller
    decides (refuse the arm, not the campaign)."""

    def __init__(self, budgets: Mapping[str, float]) -> None:
        if not budgets:
            raise ValueError("at least one surface budget is required")
        self._budget = {str(k): _nonneg(v, f"budget[{k}]") for k, v in budgets.items()}
        self._spent = dict.fromkeys(self._budget, 0.0)

    def charge(self, surface: str, row: Mapping) -> float:
        if surface not in self._budget:
            raise KeyError(f"no arm-seconds budget for surface {surface!r}")
        seconds = _nonneg(row["arm_seconds_claimed"], "arm_seconds_claimed")
        self._spent[surface] += seconds
        return self.remaining(surface)

    def remaining(self, surface: str) -> float:
        return max(0.0, self._budget[surface] - self._spent[surface])

    def allows(self, surface: str, planned_arm_s: float) -> bool:
        return _nonneg(planned_arm_s, "planned_arm_s") <= self.remaining(surface)

    def snapshot(self) -> dict:
        spent_all = sum(self._spent.values())
        return {"schema": BUDGET_SCHEMA, "unit": "arm_seconds_claimed",
                "surfaces": {k: {"budget": self._budget[k], "spent": round(self._spent[k], 3),
                                 "remaining": round(self.remaining(k), 3),
                                 "share_of_spend": round(self._spent[k] / spent_all, 4) if spent_all else None}
                             for k in self._budget}}
