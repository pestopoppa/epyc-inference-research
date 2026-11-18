#!/usr/bin/env python3
"""R23-48: leave-one-out arm per accumulated lever, run whenever the champion changes.

Levers interact through what they remove (INF-70: a +1.0% lever measured -1.39% after two
later levers removed the stall it paid for). The loop A/Bs each NEW keep against the tip
but never re-measures a PRIOR keep. On each champion-of-record advance this module plans
ONE arm per accumulated keep with that keep reverted (the build and the paired A/B run
elsewhere: `git revert --no-commit <keep>` in a detached worktree, incremental build,
paired A/B against the champion at the calibrated floor) and judges the result.

Pure planning and judgement, no git, no build, no measurement:

* `plan_arms` -> one arm per keep, bounded at n_keeps per promotion (not per iteration).
* `judge` -> `keep_confirmed` / `candidate_drop` / `inconclusive` from the arm's effect
  (champion-vs-reverted, positive = champion better) and its calibrated floor.
* `retraction_record` -> the `retracted_by_loo` row for `experiments.md`, carrying both
  sample vectors; never silently dropped.

The headline stays the assembled-stack A/B, never the product of solo deltas.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

SCHEMA = "epyc.autokernel.loo_arm.v1"
KEEP_CONFIRMED = "keep_confirmed"
CANDIDATE_DROP = "candidate_drop"
INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class Arm:
    keep: str                    # commit id of the lever being reverted
    champion: str                # champion-of-record commit the arm is measured against
    order: int                   # position in the accumulated keeps

    def arm_id(self) -> str:
        digest = hashlib.sha256(f"{self.champion}:{self.keep}".encode()).hexdigest()[:12]
        return f"loo-{self.keep[:9]}-{digest}"

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "arm_id": self.arm_id(), "revert": self.keep,
                "against": self.champion, "order": self.order,
                "build": "git revert --no-commit <keep> in a detached worktree; "
                         "incremental build via the object-digest path"}


def plan_arms(champion: str, keeps: Sequence[str], *, already_measured: Sequence[str] = ()
              ) -> list[Arm]:
    """One arm per accumulated keep for this champion; keeps already measured against this
    exact champion are skipped (cost is n_keeps per promotion, never repeated)."""
    if not champion:
        raise ValueError("champion commit required")
    seen = set(already_measured)
    arms, ids = [], set()
    for order, keep in enumerate(keeps):
        if not keep or keep in ids:
            raise ValueError(f"keeps must be non-empty and unique, got {keep!r}")
        ids.add(keep)
        arm = Arm(keep, champion, order)
        if arm.arm_id() not in seen:
            arms.append(arm)
    return arms


def judge(effect_pct: float | None, floor_pct: float | None) -> str:
    """`effect_pct` = champion minus reverted-arm, in percent (positive: the lever helps).

    The lever is CONFIRMED only when removing it costs more than the calibrated floor. A
    removal that is neutral-or-better (effect <= floor) is a candidate DROP (the box's
    rule). No calibrated floor, or no effect, is INCONCLUSIVE -- never a drop: a missing
    measurement is not evidence the lever is dead."""
    if effect_pct is None or floor_pct is None or floor_pct <= 0:
        return INCONCLUSIVE
    return KEEP_CONFIRMED if effect_pct > floor_pct else CANDIDATE_DROP


def judge_neutral_or_better(effect_pct: float | None, floor_pct: float | None) -> bool:
    """The drop trigger as a predicate: `judge(...) == CANDIDATE_DROP`."""
    return judge(effect_pct, floor_pct) == CANDIDATE_DROP


def retraction_record(arm: Arm, *, champion_samples: Sequence[float],
                      reverted_samples: Sequence[float], effect_pct: float,
                      floor_pct: float) -> dict[str, Any]:
    if judge_neutral_or_better(effect_pct, floor_pct) is False:
        raise ValueError("a retraction needs a calibrated floor and a neutral-or-better removal")
    if not champion_samples or not reverted_samples:
        raise ValueError("a retraction carries both sample vectors")
    return {"schema": SCHEMA, "status": "retracted_by_loo", "arm_id": arm.arm_id(),
            "keep": arm.keep, "champion": arm.champion, "effect_pct": effect_pct,
            "floor_pct": floor_pct, "champion_samples": list(champion_samples),
            "reverted_samples": list(reverted_samples), "verdict": judge(effect_pct, floor_pct),
            "note": "candidate DROP; recorded, never silently removed"}


def summarize(results: Mapping[str, str]) -> dict[str, Any]:
    drops = sorted(k for k, v in results.items() if v == CANDIDATE_DROP)
    return {"arms": len(results), "candidate_drops": drops,
            "inconclusive": sorted(k for k, v in results.items() if v == INCONCLUSIVE)}
