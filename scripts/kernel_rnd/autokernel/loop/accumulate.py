#!/usr/bin/env python3
"""Compound-then-gate: batch cheap bench keeps until their compounded gain clears the
serving floor, THEN spend the expensive serving gate once (R23-44, operator directive
2026-09-04: "collect llama-bench keeps until they compound to 2x-3x noise floor before
the llama-server final champion advancement gate").

WHY THIS EXISTS. R23-43 made the keep gate a llama-server measurement under the champion's
canonical recipe -- correct, because only serving performance matters -- but the serving
metric's noise floor is ~3.5% (per-request aggregate, R23-43 (2)), while an individual
bench keep is 1-3%. A per-keep serving gate therefore vetoes EVERY keep: each one is
smaller than the floor it must clear, so the loop can never advance. That is not the
serving gate being strict; it is a resolution mismatch.

THE FIX IS A TWO-TIER CHAMPION.
  * The ACCUMULATOR (a working champion) advances on every bench keep. The anchor tracks
    it, so successive keeps compound (each is measured marginal against the last), exactly
    as before. This is cheap: llama-bench only.
  * The CHAMPION OF RECORD -- the last serving-DEMONSTRATED commit, the one a promotion
    would ship -- advances ONLY when the accumulator's compounded bench gain over it
    reaches `fire_multiple` x the serving floor, and THEN only if the one serving gate
    fired at that point comes back decisive and positive.

So the serving gate runs RARELY (once per bundle, not once per keep) and only on a bundle
big enough that the ~3.5% floor can resolve it. `fire_multiple` defaults to 2.5 (the
midpoint of the operator's 2-3x), i.e. a bundle must compound past ~8.8% bench before the
serving gate is even attempted.

THE DIVERGENCE CASE IS THE POINT, NOT AN EDGE. 2026-09-04 proved bench can gain while
serving stays flat (dec-b4 +35% -> DFlash2 serving 0%). So a bundle CAN compound past the
threshold on bench and still fail the serving gate. When it does, the champion of record
does NOT move -- the operator's rule is that only a serving win advances it -- and the
bundle is recorded as a measured divergence. What happens to the accumulated commits then
is a policy choice the caller selects (`DivergenceAction`); this module computes the
decision and leaves the git/build mechanics to the loop.

This module is pure: it holds NO build directories and runs NO measurements. The loop
injects the compounded-bench number and the serving-gate row; `accumulate` decides.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import enum


class Decision(enum.Enum):
    """What to do after a bench keep lands on the accumulator."""
    ACCUMULATE = "accumulate"      #: below the fire threshold -- keep batching, no serving spend
    FIRE_SERVING = "fire_serving"  #: compounded gain cleared the threshold -- run the serving gate


class Outcome(enum.Enum):
    """What the serving gate said about the whole bundle."""
    PROMOTE = "promote"        #: decisive positive serving win -- advance the champion of record
    DIVERGED = "diverged"      #: bundle cleared bench but serving did not confirm -- champion holds


class DivergenceAction(enum.Enum):
    """What the loop does with the accumulated commits when a bundle diverges. The caller
    picks one; this module only NAMES the choice so the decision is explicit and logged."""
    ROLLBACK = "rollback"   #: reset the accumulator to the champion of record, discard the bundle
    HOLD = "hold"           #: keep the bundle on the accumulator, keep batching from here
    #: (a future BISECT action -- find which keeps transferred -- is deliberately not built
    #: yet: it costs O(log n) extra serving gates and only pays off once divergence is common.)


@dataclass(frozen=True)
class AccumulatorPolicy:
    """The thresholds. `fire_multiple` is the operator's 2-3x; `on_divergence` is the
    caller's choice for the divergence case (default ROLLBACK: the conservative reading of
    'only serving advances the champion' -- a bundle that did not transfer is not kept)."""
    fire_multiple: float = 2.5
    on_divergence: DivergenceAction = DivergenceAction.ROLLBACK

    def fire_threshold_pct(self, serving_floor_pct: float) -> float:
        return self.fire_multiple * serving_floor_pct


@dataclass
class Bundle:
    """The accumulator's state: the champion of record it builds on, its own tip, the
    keeps batched onto it, and the compounded bench gain of tip over champion-of-record
    (RE-MEASURED against the champion of record after each keep, never a product of
    marginal effects -- keeps interact, and the compounded number is what the serving gate
    will be asked to confirm)."""
    champion_of_record: str
    tip: str
    keeps: list = field(default_factory=list)
    compounded_bench_pct: float = 0.0

    def add_keep(self, mechanism_id: str, tip: str, compounded_bench_pct: float) -> None:
        self.keeps.append(mechanism_id)
        self.tip = tip
        self.compounded_bench_pct = compounded_bench_pct

    def is_empty(self) -> bool:
        return not self.keeps


def decide_after_keep(bundle: Bundle, serving_floor_pct: float,
                      policy: AccumulatorPolicy) -> Decision:
    """After a keep lands: does the compounded bench gain clear `fire_multiple` x floor?
    An uncalibrated floor (None) can never be cleared, so the loop keeps accumulating
    rather than firing a gate it cannot judge (fail-closed, same as R23-43)."""
    if serving_floor_pct is None:
        return Decision.ACCUMULATE
    if bundle.compounded_bench_pct >= policy.fire_threshold_pct(serving_floor_pct):
        return Decision.FIRE_SERVING
    return Decision.ACCUMULATE


def classify_serving(serving_row: dict) -> Outcome:
    """Read the serving A/B row (serving.compare output) for the whole bundle. PROMOTE
    only on a decisive, positive serving effect; anything else -- uncalibrated (decisive
    None), within-floor, or a regression -- is DIVERGED, and the champion of record holds.
    This is the same fail-closed grammar as R23-43's per-keep gate, applied to the bundle."""
    if not serving_row.get("decisive"):
        return Outcome.DIVERGED
    if serving_row.get("effect", 0.0) <= 0:
        return Outcome.DIVERGED
    return Outcome.PROMOTE


def resolve(bundle: Bundle, serving_row: dict, policy: AccumulatorPolicy) -> dict:
    """Fold the serving gate's verdict into a caller-actionable plan. Returns the outcome,
    the divergence action (only meaningful when DIVERGED), a one-line reason for the log,
    and the new champion of record the loop should record on a PROMOTE."""
    outcome = classify_serving(serving_row)
    if outcome is Outcome.PROMOTE:
        return {
            "outcome": outcome, "action": None,
            "new_champion_of_record": bundle.tip,
            "reason": (f"bundle of {len(bundle.keeps)} keep(s) (+{bundle.compounded_bench_pct:.2f}% "
                       f"bench) confirmed on serving: {serving_row.get('effect_pct', 0.0):+.3f}% "
                       f"(floor {serving_row.get('noise_floor_pct')}%) -> champion of record "
                       f"advances to {bundle.tip[:12]}")}
    return {
        "outcome": outcome, "action": policy.on_divergence,
        "new_champion_of_record": bundle.champion_of_record,
        "reason": (f"DIVERGENCE: bundle of {len(bundle.keeps)} keep(s) reached "
                   f"+{bundle.compounded_bench_pct:.2f}% bench but serving said "
                   f"{serving_row.get('effect_pct', 0.0):+.3f}% (decisive={serving_row.get('decisive')}, "
                   f"floor {serving_row.get('noise_floor_pct')}%); champion of record HOLDS at "
                   f"{bundle.champion_of_record[:12]}, action={policy.on_divergence.value}")}


__all__ = ["Decision", "Outcome", "DivergenceAction", "AccumulatorPolicy", "Bundle",
           "decide_after_keep", "classify_serving", "resolve"]
