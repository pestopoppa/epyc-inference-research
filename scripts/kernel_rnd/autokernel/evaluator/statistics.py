#!/usr/bin/env python3
"""statistics.py — the AK3 reducer P-AK-SEARCH-1's statistics clauses demand.

WHY THIS MODULE EXISTS
----------------------
`api.py` can refuse a verdict that its evidence does not imply, but it takes the
`EffectEstimate` — the e-value, its threshold, the MDE, the noise floor, the
block count — as an INPUT. Whoever produces that object is where the statistics
are actually decided, and every historical failure in this project happened
there, not in the verdict:

  * a threshold that was **supplied** rather than derived, so "significant" meant
    whatever the author typed;
  * a sample extended *"while it might still change"*, which is peeking, and
    which no amount of correct arithmetic afterwards repairs;
  * a magnitude summary (an LCB, a ratio, a bootstrap interval) used as the TEST,
    when the constitution's sanctioned instrument for a rate claim is an
    anytime-valid e-process and the word "LCB" appears nowhere in it;
  * an MDE computed after seeing the estimate, which is not an MDE;
  * the maximum over many candidates reported as the candidate's own effect,
    which is upward-biased by construction.

This module is the machinery that makes each of those unrepresentable rather
than merely discouraged. Governing instrument:
`epyc-root/measurement/protocols/kernel-research.md` (Annex K, **P-AK-SEARCH-1**,
RATIFIED 2026-08-03), sections *"Campaign calibration block — every threshold is
derived, none is supplied"* and *"Statistical requirements"*. Owning design:
`handoffs/active/autokernel-research-loop.md` §9.2 (AK-D15).

WHAT IS DERIVED, AND WHAT IS AN INPUT
-------------------------------------
Derived here, never accepted as a literal (the protocol's own list):

  1. the campaign noise floor `φ`          — `estimate_noise_floor`
  2. the minimum paired-block count `B_min` — `solve_calibration` step 4
  3. the e-process rejection thresholds     — `1/α_sel`, `1/α_conf`, from
     `max_candidates` and `confirmation_admission_count`
  4. the anchor-gate acceptance band        — `anchor_gate_band`

`api.CalibrationOutputs` is constructed by exactly one function in this
module — `solve_calibration` — and only when every condition the protocol states
has been evaluated and PASSED. There is no other way to obtain one from here.

Campaign INPUTS (precondition 8, fixed before the solve and held constant
through it): `calibration_block_count`, `contribution_floor`, `max_candidates`,
`confirmation_admission_count`, `max_blocks_per_candidate`, the stopping rule's
SHAPE, and the stratum split rule. These arrive in `api.CampaignControls`,
`StoppingRule` and `StratumSplitRule`; none of them is invented here.

CONSTRUCTION parameters (the one place a constant legitimately lives). The
protocol says: *"The e-process construction itself (its supermartingale or
betting form, its reducer, and its resampling method) is a property of the
evaluator bundle, fixed at the bundle hash; a campaign selects among
constructions the bundle already implements and records which one it selected."*
`EProcessConstruction` is that object. Its betting cap, its MDE power target and
its resampling counts live inside it, are hashed into its content hash, and
CANNOT be overridden by a campaign, a controller, or a keyword argument —
`select_construction()` returns a frozen registry member and refuses anything
else. A campaign records `e_process_construction_id`; it does not tune it.

STDLIB `statistics` IS DELIBERATELY NOT IMPORTED
------------------------------------------------
This file is named `statistics.py` inside a package. Importing the stdlib module
of the same name works today only because absolute imports win, and it would
break the moment anyone put `evaluator/` on `sys.path` — the exact failure the
package README documents for `resource`. Median, MAD and the quantile estimator
are therefore implemented here, which is wanted anyway: the quantile ESTIMATOR
is a recorded choice (`PERCENTILE_METHOD`), not an unstated library default.

NO INFERENCE, NO BENCHMARK, NO PROCESS, NO FILE. Every function here takes
already-measured samples and returns numbers, `schemas.Check`es, or a refusal.
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Sequence

from .. import schemas
from . import api

__all__ = [
    # identity
    "STATISTICS_MODULE_ID", "PERCENTILE_METHOD",
    # errors
    "StatisticsError", "CalibrationFailed", "InsufficientMaterial",
    "StoppingRuleMutated", "StoppingRuleViolation", "ReductionInadmissible",
    "ConstructionNotImplemented", "EValueNotRepresentable", "MaterialError",
    # constitutional floors
    "P_BENCH_1_REPS_RULE", "RepsFloor", "reps_floor_for_relative_effect",
    "OwningProtocolRepRule", "REP_RULE_FIXED", "REP_RULE_FLOOR",
    # robust reduction
    "median", "mad", "percentile", "min_samples_for_quantile",
    # measurement material
    "EFFECT_SCALE_RELATIVE", "EFFECT_SCALE_ABSOLUTE", "EFFECT_SCALES",
    "ORDER_ANCHOR_FIRST", "ORDER_CANDIDATE_FIRST", "ORDERS",
    "SEGMENT_BASE", "SEGMENT_EXTENSION", "SEGMENTS",
    "PairedBlock", "block_effect", "orient", "OrderSchedule", "derive_seed",
    # e-process
    "HYPOTHESIS_IMPROVEMENT", "HYPOTHESIS_NON_INFERIORITY", "HYPOTHESES",
    "EProcessConstruction", "CONSTRUCTIONS", "select_construction",
    "EProcessRun", "run_e_process", "null_boundary_for",
    # stopping rule
    "STOPPING_OUTCOMES", "AUTHORIZED_DECISIONS", "FUTILITY_UNREACHABLE_THRESHOLD",
    "BoundedExtension", "FutilityRule", "StoppingRule", "StoppingRuleCommitment",
    "StopDecision", "LookResult", "BlockRequest", "SequentialEvaluation",
    # calibration
    "NoiseFloor", "estimate_noise_floor", "neutral_control_consistency",
    "CrossingRate", "resampled_crossing_rate", "empirical_crossing_rate",
    "required_disjoint_windows", "MinimumDetectableEffect", "solve_mde",
    "AnchorGateBand", "anchor_gate_band", "anchor_gate_check",
    "CalibrationInputs", "CalibrationAttempt", "CalibrationSolve", "solve_calibration",
    # strata
    "RotationSchedule", "StratumSplitRule",
    # reduction
    "CampaignStatistics", "BlockReduction", "PairedBlockReducer",
    "DescriptiveLCB", "descriptive_lcb", "verify_reduction_reproducible",
]

#: Identity of this reducer implementation. It is part of what the runtime
#: source-label attestation (precondition 5) resolves, so it is versioned.
STATISTICS_MODULE_ID = "autokernel.evaluator.statistics/v1"


# =============================================================================
# Errors — every one is a refusal. None of them has a degraded-result branch.
# =============================================================================

class StatisticsError(api.EvaluatorError):
    """Base for every refusal in this module."""


class MaterialError(StatisticsError):
    """The measurement material is malformed. Not a finding about a candidate."""


class InsufficientMaterial(StatisticsError):
    """Not enough material to evaluate the thing at its own stated resolution.

    Raised rather than returning a number computed from too little data: a P95
    taken over 8 points is the maximum wearing a percentile's name, and a
    crossing rate of "0/12" cannot demonstrate a rate at or below 1/50.
    """


class CalibrationFailed(StatisticsError):
    """*"the calibration FAILS and the campaign does not start."*

    There is no partial calibration and no fallback ceiling, so this is raised
    only by `CalibrationSolve.require_accepted()`. `solve_calibration` itself
    RETURNS the failed solve, because the protocol requires both the failed and
    the accepted calibration to be retained in the manifest.
    """


class StoppingRuleMutated(StatisticsError):
    """*"Any post-hoc change to the stopping rule voids every affected record."*"""


class StoppingRuleViolation(StatisticsError):
    """A look, an extension, or a decision the pre-committed rule does not license.

    *"Extension follows the declared rule only … rather than unstructured
    continuation while the answer might still change."* The only way to obtain
    another block from `SequentialEvaluation` is to ask the rule for one; asking
    when the rule has terminated raises this.
    """


class ReductionInadmissible(StatisticsError):
    """The blocks cannot produce a conforming estimate. Carries the reduction.

    Deliberately an exception and NOT a `None` return: `api.TierDispatcher`
    treats `effect is None` as "this record is not a rate comparison" and then
    SKIPS the rate-only void conditions. Returning None on a strata violation
    would therefore suppress the very void the violation must raise. The full
    `BlockReduction` — including the failing `Check`s to feed into
    `api.WindowAttestations` — hangs off `.reduction` so the run is still
    journalable as INVALID with its reason.
    """

    def __init__(self, message: str, reduction: "BlockReduction") -> None:
        super().__init__(message)
        self.reduction = reduction


class ConstructionNotImplemented(StatisticsError):
    """*"a campaign selects among constructions the bundle already implements."*"""


class EValueNotRepresentable(StatisticsError):
    """The wealth process left the representable range of a float.

    The e-process is accumulated in LOG space precisely so this is unreachable
    for any block count a campaign ceiling permits. If it ever fires, the log
    e-value is still exact and is named in the message — the estimate is refused
    rather than reported with a silently clipped e-value.
    """


class EffectScaleError(MaterialError):
    """Absolute and relative effect scales were mixed, or the scale is unusable."""


# =============================================================================
# Constitutional constants, quoted — these are FLOORS, not calibration outputs
# =============================================================================

#: `bench-cpu.md:21-22`, quoted by P-AK-SEARCH-1 "Statistical requirements":
#: *"≥5 for ≥5% effects; **≥10 for ≤2% effects**; report median + MAD."*
#: `B_min` *"is floored by, and MUST NEVER fall below"* this rule. It is a
#: constitutional constant cited by the protocol, not a threshold this module
#: derives — the derived quantity is `B_min` itself, solved upward from here.
P_BENCH_1_REPS_RULE = (
    {"relative_effect_at_least": 0.05, "blocks": 5},
    {"relative_effect_at_most": 0.02, "blocks": 10},
)

REP_RULE_FLOOR = "floor"
REP_RULE_FIXED = "fixed"


@dataclass(frozen=True)
class RepsFloor:
    """The P-BENCH-1 reps floor for a campaign's contribution floor.

    `band` names which limb of the rule applied. The rule states two limbs and
    leaves (2%, 5%) UNDEFINED; this module does not interpolate a number the
    constitution does not state. It takes the stricter limb, and says so in
    `conservative` and `note` rather than silently — the protocol's own
    direction is that *"a calibration that would license fewer blocks than the
    owning protocol already requires is discarded, not applied."*
    """

    blocks: int
    band: str
    relative_effect: float
    citation: str
    conservative: bool
    note: str

    def to_dict(self) -> dict:
        return {"blocks": self.blocks, "band": self.band,
                "relative_effect": self.relative_effect, "citation": self.citation,
                "conservative": self.conservative, "note": self.note}


def reps_floor_for_relative_effect(relative_effect: float) -> RepsFloor:
    """Apply the P-BENCH-1 reps rule to a relative effect magnitude."""
    if isinstance(relative_effect, bool) or not isinstance(relative_effect, (int, float)):
        raise MaterialError(f"relative_effect must be a number, got {relative_effect!r}")
    if not math.isfinite(relative_effect) or relative_effect <= 0:
        raise MaterialError(
            f"relative_effect must be finite and strictly positive, got {relative_effect!r}; "
            "a campaign with a zero or unbounded contribution floor cannot derive a rep floor"
        )
    citation = "bench-cpu.md:21-22 (P-BENCH-1 reps rule), quoted by P-AK-SEARCH-1"
    if relative_effect >= 0.05:
        return RepsFloor(5, "at_or_above_5pct", float(relative_effect), citation, False,
                         "the rule states >=5 blocks for >=5% effects")
    if relative_effect <= 0.02:
        return RepsFloor(10, "at_or_below_2pct", float(relative_effect), citation, False,
                         "the rule states >=10 blocks for <=2% effects")
    return RepsFloor(
        10, "undefined_band_2pct_to_5pct", float(relative_effect), citation, True,
        "the P-BENCH-1 rule states no value strictly between 2% and 5%; this module takes "
        "the stricter limb (10) rather than interpolating a number the constitution does "
        "not state. The bound is reported, not silent.")


@dataclass(frozen=True)
class OwningProtocolRepRule:
    """A rep rule stated by the protocol that owns the cell's own phase.

    *"Where the cell's own phase is governed by a protocol that states a stricter
    or a fixed rep rule, that protocol's rule governs its own cells and this
    calibration never overrides it — in particular `bench-cpu.md:174-178`
    (P-BENCH-4, exactly five, no retry, replace, discard or pooling) is a fixed
    count, not a floor to be raised."*

    This is a declared campaign INPUT — the campaign names its cell's owning
    protocol. Nothing here maps cells to protocols on its own, because guessing
    which protocol owns a cell is how a fixed count silently becomes a floor.
    """

    protocol_id: str
    kind: str
    blocks: int
    citation: str

    def __post_init__(self) -> None:
        if not isinstance(self.protocol_id, str) or not self.protocol_id.strip():
            raise MaterialError("owning_rep_rule.protocol_id must be a non-empty string")
        if self.kind not in (REP_RULE_FLOOR, REP_RULE_FIXED):
            raise MaterialError(
                f"owning_rep_rule.kind: {self.kind!r} must be {REP_RULE_FLOOR!r} or "
                f"{REP_RULE_FIXED!r}; 'fixed' means the count may not be raised either")
        if isinstance(self.blocks, bool) or not isinstance(self.blocks, int) or self.blocks < 1:
            raise MaterialError("owning_rep_rule.blocks must be a positive int")
        if not isinstance(self.citation, str) or not self.citation.strip():
            raise MaterialError("owning_rep_rule.citation must name where the rule is stated")

    def to_dict(self) -> dict:
        return {"protocol_id": self.protocol_id, "kind": self.kind,
                "blocks": self.blocks, "citation": self.citation}


# =============================================================================
# Robust reduction — median + MAD, and a NAMED quantile estimator
# =============================================================================

#: The quantile estimator, recorded with every value it produces. Linear
#: interpolation between the two order statistics bracketing rank `q*(n-1)`
#: (Hyndman-Fan type 7). Named because "the 95th percentile" is not one number
#: until the estimator is stated.
PERCENTILE_METHOD = "linear_interpolation_type7"


def median(values: Sequence[float]) -> float:
    """Median. Raises on an empty sequence — there is no median of nothing."""
    xs = sorted(_finite_floats(values, "median"))
    n = len(xs)
    if n == 0:
        raise MaterialError("median() of an empty sequence")
    mid = n // 2
    if n % 2:
        return xs[mid]
    return (xs[mid - 1] + xs[mid]) / 2.0


def mad(values: Sequence[float]) -> float:
    """Median absolute deviation, unscaled — the form `bench-cpu.md:21-22` reports."""
    xs = _finite_floats(values, "mad")
    if not xs:
        raise MaterialError("mad() of an empty sequence")
    med = median(xs)
    return median([abs(x - med) for x in xs])


def percentile(values: Sequence[float], q: float) -> float:
    """The `q` quantile under `PERCENTILE_METHOD`. `q` in [0, 1]."""
    if isinstance(q, bool) or not isinstance(q, (int, float)) or not 0.0 <= q <= 1.0:
        raise MaterialError(f"percentile q must be in [0, 1], got {q!r}")
    xs = sorted(_finite_floats(values, "percentile"))
    n = len(xs)
    if n == 0:
        raise MaterialError("percentile() of an empty sequence")
    if n == 1:
        return xs[0]
    pos = (n - 1) * float(q)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[int(pos)]
    return xs[lo] + (pos - lo) * (xs[hi] - xs[lo])


def min_samples_for_quantile(q: float) -> int:
    """Smallest n at which the `q` quantile is not simply an extreme order statistic.

    `ceil(1/(1-q))` for an upper quantile. At n below this, "the 95th percentile"
    IS the maximum, and reporting it as a percentile overstates what was
    measured. Derived, not tabulated.
    """
    if isinstance(q, bool) or not isinstance(q, (int, float)) or not 0.0 < q < 1.0:
        raise MaterialError(f"min_samples_for_quantile q must be in (0, 1), got {q!r}")
    tail = min(q, 1.0 - q)
    return int(math.ceil(1.0 / tail))


#: How many offenders a reason string names before it elides. Reporting only,
#: never a limit on what is CHECKED — and the elision always states the total,
#: because "blocks [0,1,2,3,4,5,6,7]" reads as "eight blocks" when it means
#: "eight of fifty".
_REASON_LIST_LIMIT = 8


def _named(items: Sequence[Any], *, noun: str = "blocks") -> str:
    """`"blocks [0, 1, 2]"`, or `"8 of 50 blocks [...]"` when the list is elided."""
    shown = sorted(items)[:_REASON_LIST_LIMIT]
    if len(items) <= _REASON_LIST_LIMIT:
        return f"{noun} {shown}"
    return f"{len(shown)} of {len(items)} {noun} {shown} (list elided)"


def _finite_floats(values: Sequence[float], label: str) -> list:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise MaterialError(f"{label}: expected a sequence of numbers, got "
                            f"{type(values).__name__}")
    out = []
    for i, v in enumerate(values):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise MaterialError(f"{label}[{i}]: expected a finite number, got {v!r}")
        out.append(float(v))
    return out


# =============================================================================
# Deterministic seeding — every resampling in this module is reproducible
# =============================================================================

def _parse_instant(value: Any) -> Optional[datetime]:
    """An ISO-8601 timestamp WITH an offset, as an ordered instant, or `None`.

    Timestamps are never compared as strings in this module. Lexicographic order
    is not chronological order across two legal spellings of the same instant:
    `"2026-8-2T00:00:00+00:00"` sorts ABOVE `"2026-08-02T12:00:00+00:00"`, so a
    block measured half a day BEFORE lineage entry compares as "after" it and
    passes the winner's-curse gate. `None` (unparseable, or naive) is the third
    outcome — an instant that cannot be ordered is not an instant that is later.
    Same rule as `schemas._need_timestamp`: a naive stamp on a shared host is
    ambiguous across sessions.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return None if parsed.tzinfo is None else parsed


def derive_seed(campaign_seed: str, *parts: Any) -> int:
    """Derive a reproducible 128-bit seed from the committed campaign seed.

    *"the randomization seed derives from the campaign seed committed before the
    first candidate was measured, and is recorded."* Every resampling, every
    order draw and every bootstrap in this module goes through here, so a record
    naming its campaign seed and its purpose names the exact stream that ran.
    """
    if not isinstance(campaign_seed, str) or not campaign_seed.strip():
        raise MaterialError("campaign_seed must be a non-empty string committed at "
                            "campaign start")
    h = hashlib.blake2b(digest_size=16)
    h.update(campaign_seed.encode("utf-8"))
    for part in parts:
        h.update(b"\x1f")
        h.update(str(part).encode("utf-8"))
    return int.from_bytes(h.digest(), "big")


def _rng(campaign_seed: str, *parts: Any) -> random.Random:
    return random.Random(derive_seed(campaign_seed, *parts))


# =============================================================================
# Measurement material — paired blocks and order control
# =============================================================================

EFFECT_SCALE_RELATIVE = "relative"
EFFECT_SCALE_ABSOLUTE = "absolute"
EFFECT_SCALES = (EFFECT_SCALE_RELATIVE, EFFECT_SCALE_ABSOLUTE)

ORDER_ANCHOR_FIRST = "anchor_first"
ORDER_CANDIDATE_FIRST = "candidate_first"
ORDERS = (ORDER_ANCHOR_FIRST, ORDER_CANDIDATE_FIRST)

SEGMENT_BASE = "base"
SEGMENT_EXTENSION = "extension"
SEGMENTS = (SEGMENT_BASE, SEGMENT_EXTENSION)


@dataclass(frozen=True)
class PairedBlock:
    """One paired block: the anchor and the candidate, interleaved, in one block.

    *"Candidate and anchor are interleaved and order-randomized within every
    paired block … Blocked designs (candidate × n, then anchor × n) are
    forbidden — thermal and page-cache drift alias onto the arm effect."*

    A block therefore carries the ORDER it actually ran in, and `OrderSchedule`
    checks that order against the schedule the campaign seed determines. A caller
    cannot satisfy the check by declaring an order it did not run, but it also
    cannot pass a blocked design off as randomized, which is the failure mode the
    clause names.

    `unit_id` is the measurement-material unit (shape, seed) the block used. It
    is what the selection/confirmation split partitions, so it is required: a
    block with no unit identity cannot be shown to be in one stratum only.
    """

    block_index: int
    unit_id: str
    stratum: str
    order: str
    anchor_samples: tuple
    candidate_samples: tuple
    segment: str = SEGMENT_BASE
    extension_round: Optional[int] = None
    measured_at: Optional[str] = None

    def __post_init__(self) -> None:
        if isinstance(self.block_index, bool) or not isinstance(self.block_index, int) \
                or self.block_index < 0:
            raise MaterialError("block.block_index must be a non-negative int")
        if not isinstance(self.unit_id, str) or not self.unit_id.strip():
            raise MaterialError("block.unit_id must be a non-empty string naming the "
                                "measurement-material unit (shape/seed) the block used")
        if self.stratum not in api.STRATA:
            raise MaterialError(f"block.stratum: {self.stratum!r} is not one of "
                                f"{list(api.STRATA)}")
        if self.order not in ORDERS:
            raise MaterialError(f"block.order: {self.order!r} is not one of {list(ORDERS)}")
        if self.segment not in SEGMENTS:
            raise MaterialError(f"block.segment: {self.segment!r} is not one of "
                                f"{list(SEGMENTS)}")
        for name in ("anchor_samples", "candidate_samples"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or not values:
                raise MaterialError(
                    f"block.{name} must be a non-empty tuple; a block with one arm missing "
                    "is not a paired block")
            _finite_floats(values, f"block.{name}")
        if self.segment == SEGMENT_EXTENSION:
            if isinstance(self.extension_round, bool) or \
                    not isinstance(self.extension_round, int) or self.extension_round < 1:
                raise MaterialError(
                    "block.extension_round must be a positive int on an extension block; "
                    "an extension that cannot say which declared round it belongs to is "
                    "unstructured continuation")
        elif self.extension_round is not None:
            raise MaterialError("block.extension_round must be None on a base block")
        if self.measured_at is not None:
            if not isinstance(self.measured_at, str) or not self.measured_at.strip():
                raise MaterialError("block.measured_at must be a non-empty string or None")
            if _parse_instant(self.measured_at) is None:
                raise MaterialError(
                    f"block.measured_at {self.measured_at!r} is not an ISO-8601 timestamp "
                    "with a UTC offset. It is what orders confirmation evidence against "
                    "lineage entry, and an unorderable stamp on a shared host cannot do "
                    "that (same rule as schemas._need_timestamp).")

    def to_list(self) -> list:
        """Canonicalizable form — lists, never tuples (schemas rejects tuples)."""
        return [self.block_index, self.unit_id, self.stratum, self.order, self.segment,
                self.extension_round, self.measured_at,
                list(self.anchor_samples), list(self.candidate_samples)]

    def to_tuple(self) -> tuple:
        """Hashable raw-sample form for `api.EffectEstimate.raw_samples`."""
        return (self.block_index, self.unit_id, self.stratum, self.order, self.segment,
                self.extension_round, self.measured_at,
                tuple(float(v) for v in self.anchor_samples),
                tuple(float(v) for v in self.candidate_samples))


def block_effect(block: PairedBlock, *, scale: str) -> float:
    """The block's signed effect: candidate minus anchor, in the declared scale.

    Sign convention matches `api._resolve_effect`: the value is
    `candidate - anchor` in the metric's own units (or as a fraction of the
    anchor), so a positive value means "candidate higher", which `api` then
    reads against `metric_direction`. Orientation to "is this an improvement" is
    a separate step (`orient`), because the e-process needs the oriented form and
    the record needs the signed one.
    """
    if not isinstance(block, PairedBlock):
        raise MaterialError("block_effect() takes a PairedBlock")
    if scale not in EFFECT_SCALES:
        raise EffectScaleError(f"scale: {scale!r} is not one of {list(EFFECT_SCALES)}")
    anchor = median(block.anchor_samples)
    candidate = median(block.candidate_samples)
    if scale == EFFECT_SCALE_ABSOLUTE:
        return candidate - anchor
    if anchor <= 0.0:
        raise EffectScaleError(
            f"block {block.block_index}: relative effect needs a strictly positive anchor "
            f"median, got {anchor!r}. A relative scale over a non-positive denominator is "
            "not a percentage, and silently switching to an absolute scale here would mix "
            "two scales in one campaign.")
    return (candidate - anchor) / anchor


def orient(effect: float, metric_direction: str) -> float:
    """Return the effect oriented so that POSITIVE always means "candidate better"."""
    if metric_direction not in schemas.METRIC_DIRECTIONS:
        raise MaterialError(f"metric_direction: {metric_direction!r} is not one of "
                            f"{sorted(schemas.METRIC_DIRECTIONS)}")
    return float(effect) if metric_direction == "higher_better" else -float(effect)


@dataclass(frozen=True)
class OrderSchedule:
    """The order-randomization schedule, derived from the committed campaign seed.

    Prefix-stable on purpose: the order of block *i* depends on *i* alone (and on
    the campaign seed and the candidate), so adding extension blocks cannot
    retroactively change the schedule of the base blocks and turn a conforming
    run into a non-conforming one.

    *"A retry is a fresh reset in reversed order"* (`bench-cpu.md:48-49`) —
    `retry()` flips every element and nothing else. The base draw deliberately
    does NOT key on `attempt`: if a retry re-drew the schedule it would be a
    fresh randomization, not a reversal, and "reversed order on retry" would be
    unverifiable. `attempt` is therefore recorded provenance, and a second retry
    reverses back, which is what alternating reversal means.

    Extension pairs are *"fresh reversed-order pairs"*, so `order_for()` flips
    the base schedule for indices at or beyond the base block count.
    """

    campaign_seed: str
    candidate_id: str
    attempt: int
    base_blocks: int
    reversed_schedule: bool

    def __post_init__(self) -> None:
        if not isinstance(self.campaign_seed, str) or not self.campaign_seed.strip():
            raise MaterialError("order schedule needs the committed campaign seed")
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise MaterialError("order schedule needs a candidate id")
        for name in ("attempt", "base_blocks"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise MaterialError(f"order schedule {name} must be a non-negative int")
        if self.base_blocks < 1:
            raise MaterialError("order schedule base_blocks must be at least 1")
        if not isinstance(self.reversed_schedule, bool):
            raise MaterialError("order schedule reversed_schedule must be a bool")

    @classmethod
    def derive(cls, *, campaign_seed: str, candidate_id: str, base_blocks: int,
               attempt: int = 0) -> "OrderSchedule":
        return cls(campaign_seed=campaign_seed, candidate_id=candidate_id, attempt=attempt,
                   base_blocks=base_blocks, reversed_schedule=False)

    def retry(self) -> "OrderSchedule":
        """A retry is a fresh reset in reversed order."""
        return OrderSchedule(campaign_seed=self.campaign_seed, candidate_id=self.candidate_id,
                             attempt=self.attempt + 1, base_blocks=self.base_blocks,
                             reversed_schedule=not self.reversed_schedule)

    def _base_order(self, index: int) -> str:
        seed = derive_seed(self.campaign_seed, "order", self.candidate_id, index)
        return ORDER_ANCHOR_FIRST if seed % 2 == 0 else ORDER_CANDIDATE_FIRST

    @staticmethod
    def _flip(order: str) -> str:
        return ORDER_CANDIDATE_FIRST if order == ORDER_ANCHOR_FIRST else ORDER_ANCHOR_FIRST

    def order_for(self, index: int) -> str:
        """Required order for block `index` (0-based over base then extension)."""
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise MaterialError("order index must be a non-negative int")
        order = self._base_order(index % self.base_blocks)
        if index >= self.base_blocks:
            order = self._flip(order)         # extension = fresh REVERSED-order pairs
        if self.reversed_schedule:
            order = self._flip(order)
        return order

    def orders(self, count: int) -> tuple:
        return tuple(self.order_for(i) for i in range(count))

    def check_observed(self, blocks: Sequence[PairedBlock]) -> schemas.Check:
        """PASS/FAIL on order control, naming a blocked design when it sees one."""
        if not blocks:
            return schemas.Check(schemas.COULD_NOT_CHECK,
                                 ("no blocks were submitted; order control is unevaluable",))
        reasons = []
        for position, block in enumerate(blocks):
            expected = self.order_for(position)
            if block.order != expected:
                reasons.append(
                    f"block {position} ran {block.order!r} but the schedule derived from the "
                    f"committed campaign seed requires {expected!r}")
        observed = {b.order for b in blocks}
        expected_set = set(self.orders(len(blocks)))
        if len(observed) == 1 and len(expected_set) > 1:
            reasons.append(
                f"every block ran {next(iter(observed))!r}: this is a BLOCKED design "
                "(candidate x n, then anchor x n), which the protocol forbids because "
                "thermal and page-cache drift alias onto the arm effect")
        return schemas.Check(schemas.FAIL, tuple(reasons)) if reasons \
            else schemas.Check(schemas.PASS)

    def to_dict(self) -> dict:
        return {"candidate_id": self.candidate_id, "attempt": self.attempt,
                "base_blocks": self.base_blocks, "reversed_schedule": self.reversed_schedule}


# =============================================================================
# The e-process — anytime-valid, never an ad-hoc bound
# =============================================================================

HYPOTHESIS_IMPROVEMENT = "improvement"
HYPOTHESIS_NON_INFERIORITY = "non_inferiority"
HYPOTHESES = (HYPOTHESIS_IMPROVEMENT, HYPOTHESIS_NON_INFERIORITY)

_BETTING_PREDICTABLE = "predictable_grow_approximation"
_BETTING_FIXED = "fixed_fraction"
_STATISTIC_SIGN = "paired_block_effect_sign"


@dataclass(frozen=True)
class EProcessConstruction:
    """One e-process construction — a property of the bundle, fixed at its hash.

    *"The e-process construction itself (its supermartingale or betting form, its
    reducer, and its resampling method) is a property of the evaluator bundle,
    fixed at the bundle hash; a campaign selects among constructions the bundle
    already implements and records which one it selected."*

    Both implemented constructions use the SIGN of the per-block oriented effect
    against the null boundary, with TIES BROKEN AGAINST THE ALTERNATIVE:

        X_b = +1 if oriented_b > null_boundary else -1   in {-1, +1}
        W_0 = 1,  W_b = W_{b-1} * (1 + lambda_b * X_b),  lambda_b in [0, cap], cap < 1

    Under the null (the candidate is no better than the boundary, so
    `P(oriented > boundary) <= 1/2`), `E[X_b | past] = 2*P(> boundary) - 1 <= 0`,
    `lambda_b` is PREDICTABLE — computed from blocks strictly before *b* — and `W`
    is therefore a non-negative supermartingale. Ville's inequality gives
    `P(sup_b W_b >= 1/alpha) <= alpha` at ANY stopping time, which is what
    licenses inspecting the evidence every round. The statistic is
    distribution-free: it assumes only that blocks are exchangeable under the
    null, which is what order-randomized interleaving buys.

    THE TIE RULE IS LOAD-BEARING, not a rounding convenience. A three-valued
    `sign` in {-1, 0, +1} gives `E[X_b] = 2*P(> boundary) + P(= boundary) - 1`,
    which is STRICTLY POSITIVE whenever the boundary carries an atom and
    `P(> boundary) = 1/2` — a distribution squarely inside the null stated above.
    The wealth is then a SUBmartingale and Ville's bound does not hold: measured
    over 20000 null windows at `alpha=0.05`, a 40% tie rate crossed `1/alpha` at
    **67%**. Ties are not hypothetical here — an A/A control is the anchor
    measured against itself, a candidate that silently falls back to the anchor's
    code path is control 3, and both arms reduce through a median of a few
    quantized timings. An e-process whose validity depends on a continuity
    assumption that data can violate silently is the fail-open pattern this
    project bans, so the tie is charged to the candidate instead of being
    discarded.

    `mde_power_target` and the resampling counts live here, not in the campaign
    manifest, for the reason the protocol gives: the resampling method is part of
    the construction. `select_construction()` returns registry members only, so a
    campaign cannot tune them.
    """

    construction_id: str
    statistic: str
    betting_form: str
    lambda_cap: float
    lambda_init: float
    lambda_fixed: Optional[float]
    mde_power_target: float
    mde_resamples: int
    mde_max_doublings: int
    mde_search_tolerance: float
    crossing_rate_resamples: int
    band_resamples: int
    neutral_permutation_reps: int
    lcb_bootstrap_iterations: int
    description: str

    def __post_init__(self) -> None:
        if self.statistic != _STATISTIC_SIGN:
            raise ConstructionNotImplemented(
                f"statistic {self.statistic!r} is not implemented by this bundle")
        if self.betting_form not in (_BETTING_PREDICTABLE, _BETTING_FIXED):
            raise ConstructionNotImplemented(
                f"betting form {self.betting_form!r} is not implemented by this bundle")
        for name in ("lambda_cap", "lambda_init"):
            value = getattr(self, name)
            if not isinstance(value, float) or not 0.0 <= value < 1.0:
                raise ConstructionNotImplemented(
                    f"{name} must be a float in [0, 1): a betting fraction at or above 1 "
                    "can drive the wealth non-positive and destroys the supermartingale")
        if self.lambda_init > self.lambda_cap:
            raise ConstructionNotImplemented("lambda_init must not exceed lambda_cap")
        if self.betting_form == _BETTING_FIXED:
            if not isinstance(self.lambda_fixed, float) or not 0.0 < self.lambda_fixed < 1.0:
                raise ConstructionNotImplemented(
                    "a fixed-fraction construction needs lambda_fixed in (0, 1)")
        elif self.lambda_fixed is not None:
            raise ConstructionNotImplemented(
                "lambda_fixed belongs only to the fixed-fraction betting form")
        if not isinstance(self.mde_power_target, float) or not 0.0 < self.mde_power_target < 1.0:
            raise ConstructionNotImplemented("mde_power_target must be a float in (0, 1)")
        for name in ("mde_resamples", "mde_max_doublings", "crossing_rate_resamples",
                     "band_resamples", "neutral_permutation_reps",
                     "lcb_bootstrap_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConstructionNotImplemented(f"{name} must be a positive int")
        if not isinstance(self.mde_search_tolerance, float) \
                or not 0.0 < self.mde_search_tolerance < 1.0:
            raise ConstructionNotImplemented("mde_search_tolerance must be a float in (0, 1)")

    def to_dict(self) -> dict:
        return {
            "construction_id": self.construction_id, "statistic": self.statistic,
            "betting_form": self.betting_form, "lambda_cap": self.lambda_cap,
            "lambda_init": self.lambda_init, "lambda_fixed": self.lambda_fixed,
            "mde_power_target": self.mde_power_target, "mde_resamples": self.mde_resamples,
            "mde_max_doublings": self.mde_max_doublings,
            "mde_search_tolerance": self.mde_search_tolerance,
            "crossing_rate_resamples": self.crossing_rate_resamples,
            "band_resamples": self.band_resamples,
            "neutral_permutation_reps": self.neutral_permutation_reps,
            "lcb_bootstrap_iterations": self.lcb_bootstrap_iterations,
            "percentile_method": PERCENTILE_METHOD,
            "module": STATISTICS_MODULE_ID,
            "description": self.description,
        }

    def content_hash(self) -> str:
        """The construction's identity — what "fixed at the bundle hash" means here."""
        return schemas.content_hash(self.to_dict())

    def lambda_from_moments(self, count: int, total: float, total_sq: float) -> float:
        """THE definition of the predictable betting fraction, from past moments.

        Reads only blocks strictly before the one it bets on — that is the whole
        of what makes the wealth process a supermartingale, so it lives on the
        construction and is not a caller-supplied callback. Stated over moments
        rather than over the sample so the wealth loop can maintain it in O(1)
        per block without a second, drift-prone copy of the formula.
        """
        if self.betting_form == _BETTING_FIXED:
            return float(self.lambda_fixed)
        if count <= 0:
            return self.lambda_init
        mean = total / count
        var = max(total_sq / count - mean * mean, 0.0)
        denom = var + mean * mean
        if denom <= 0.0:
            value = self.lambda_cap if mean > 0.0 else 0.0
        else:
            value = mean / denom
        return min(max(value, 0.0), self.lambda_cap)

    def lambda_for(self, past: Sequence[float]) -> float:
        """The predictable betting fraction given the past signs. Same definition."""
        values = _finite_floats(past, "past")
        return self.lambda_from_moments(len(values), sum(values),
                                        sum(v * v for v in values))


#: The bundle's implemented constructions. A campaign selects by id.
CONSTRUCTIONS = {
    c.construction_id: c for c in (
        EProcessConstruction(
            construction_id="sign_martingale_predictable_lambda/v1",
            statistic=_STATISTIC_SIGN,
            betting_form=_BETTING_PREDICTABLE,
            lambda_cap=0.5,
            lambda_init=0.1,
            lambda_fixed=None,
            mde_power_target=0.8,
            mde_resamples=1000,
            mde_max_doublings=40,
            mde_search_tolerance=0.001,
            crossing_rate_resamples=2000,
            band_resamples=2000,
            neutral_permutation_reps=2000,
            lcb_bootstrap_iterations=1000,
            description=(
                "Sign test-martingale on the per-block oriented effect with a predictable, "
                "data-driven betting fraction (mean over second moment, clipped to the cap). "
                "Distribution-free; anytime-valid by Ville."),
        ),
        EProcessConstruction(
            construction_id="sign_martingale_fixed_lambda/v1",
            statistic=_STATISTIC_SIGN,
            betting_form=_BETTING_FIXED,
            lambda_cap=0.5,
            lambda_init=0.25,
            lambda_fixed=0.25,
            mde_power_target=0.8,
            mde_resamples=1000,
            mde_max_doublings=40,
            mde_search_tolerance=0.001,
            crossing_rate_resamples=2000,
            band_resamples=2000,
            neutral_permutation_reps=2000,
            lcb_bootstrap_iterations=1000,
            description=(
                "Sign test-martingale with a constant (trivially predictable) betting "
                "fraction. Same supermartingale, a different betting form: it accumulates "
                "more slowly against a large effect and degrades less against a small one."),
        ),
    )
}


#: The bundle's construction registry and `api.E_PROCESS_CONSTRUCTION_IDS` are
#: two spellings of one fact, and `api.CalibrationOutputs` refuses an id outside
#: the latter. `api` cannot ask this module what it implements (this module
#: imports `api`), so the two are reconciled by an IMPORT-TIME assertion rather
#: than by a convention: adding a construction here without adding its id there
#: is an `ImportError`, not a `CalibrationOutputs` that no reducer can honour.
_CONSTRUCTION_ID_DRIFT = (
    set(CONSTRUCTIONS) ^ set(api.E_PROCESS_CONSTRUCTION_IDS)
)
if _CONSTRUCTION_ID_DRIFT:  # pragma: no cover - import-time contract assertion
    raise ImportError(
        f"e-process construction registry drift: {sorted(_CONSTRUCTION_ID_DRIFT)} is in "
        f"exactly one of statistics.CONSTRUCTIONS ({sorted(CONSTRUCTIONS)}) and "
        f"api.E_PROCESS_CONSTRUCTION_IDS ({list(api.E_PROCESS_CONSTRUCTION_IDS)}). "
        "The construction is a property of the evaluator bundle, fixed at the bundle "
        "hash; two registries that disagree mean a campaign can record a selection the "
        "reducer cannot make, or make one the record cannot name."
    )


def select_construction(construction_id: str) -> EProcessConstruction:
    """Select among the constructions the bundle already implements. No tuning."""
    if not isinstance(construction_id, str):
        raise ConstructionNotImplemented(
            f"construction id must be a string, got {type(construction_id).__name__}")
    try:
        return CONSTRUCTIONS[construction_id]
    except KeyError:
        raise ConstructionNotImplemented(
            f"{construction_id!r} is not implemented by this evaluator bundle; implemented "
            f"constructions are {sorted(CONSTRUCTIONS)}. A campaign selects among the "
            "constructions the bundle implements and records which one it selected; it does "
            "not supply one."
        ) from None


def _require_bundle_construction(construction: Any, label: str) -> EProcessConstruction:
    """Refuse anything that is not the bundle's own construction of its own id.

    *"a campaign selects among constructions the bundle already implements and
    records which one it selected."* An object that carries a registry id while
    carrying different parameters records a selection the campaign did not make:
    the manifest says `sign_martingale_predictable_lambda/v1` and the wealth
    process bets at whatever cap the caller typed. Identity is checked by CONTENT
    HASH rather than by `is`, so a faithfully deserialized construction is
    admitted and a tuned one is not, which is the property that actually matters.
    """
    if not isinstance(construction, EProcessConstruction):
        raise ConstructionNotImplemented(f"{label} must be an EProcessConstruction")
    member = CONSTRUCTIONS.get(construction.construction_id)
    if member is None:
        raise ConstructionNotImplemented(
            f"{label}: {construction.construction_id!r} is not implemented by this "
            f"evaluator bundle; implemented constructions are {sorted(CONSTRUCTIONS)}")
    if member.content_hash() != construction.content_hash():
        raise ConstructionNotImplemented(
            f"{label}: the supplied construction carries the registry id "
            f"{construction.construction_id!r} but hashes to "
            f"{construction.content_hash()[:12]}, not the bundle's "
            f"{member.content_hash()[:12]}. The construction is a property of the "
            "evaluator bundle, fixed at the bundle hash; a campaign selects one and "
            "does not tune it.")
    return construction


def null_boundary_for(hypothesis: str, margin: float) -> float:
    """The oriented-effect boundary the null asserts the candidate does not beat."""
    if hypothesis not in HYPOTHESES:
        raise MaterialError(f"hypothesis: {hypothesis!r} is not one of {list(HYPOTHESES)}")
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) \
            or not math.isfinite(margin):
        raise MaterialError(f"margin must be a finite number, got {margin!r}")
    if hypothesis == HYPOTHESIS_IMPROVEMENT:
        if margin != 0:
            raise MaterialError(
                "an improvement e-process tests H0: oriented effect <= 0; a non-zero margin "
                "is a non-inferiority test and must say so")
        return 0.0
    if margin <= 0:
        raise MaterialError("a non-inferiority margin must be strictly positive")
    return -float(margin)


@dataclass(frozen=True)
class EProcessRun:
    """One e-process, evaluated block by block. Everything needed to replay it.

    `e_running_max` is what a record reports, because Ville bounds the SUPREMUM:
    a process that crossed at block 4 and fell back at block 9 rejected, and
    reporting only `e_final` would erase that. `log_e_running_max` is the exact
    quantity — `e_running_max` is its exponential and can, at extreme block
    counts, leave float range, in which case the estimate is REFUSED rather than
    clipped.
    """

    construction_id: str
    hypothesis: str
    margin: float
    null_boundary: float
    threshold: float
    log_threshold: float
    log_wealth: tuple
    lambdas: tuple
    signs: tuple
    log_e_final: float
    log_e_running_max: float
    first_crossing_block: Optional[int]

    @property
    def blocks(self) -> int:
        return len(self.log_wealth)

    @property
    def crossed(self) -> bool:
        return self.first_crossing_block is not None

    def _exp(self, value: float, label: str) -> float:
        try:
            out = math.exp(value)
        except OverflowError:
            raise EValueNotRepresentable(
                f"{label} = exp({value!r}) overflows a float. The log e-value is exact and "
                "is reported in `log_e_running_max`; the estimate is refused rather than "
                "reported with a clipped e-value.") from None
        if not math.isfinite(out):
            raise EValueNotRepresentable(
                f"{label} is not finite (log value {value!r}); refused rather than clipped.")
        return out

    @property
    def e_final(self) -> float:
        return self._exp(self.log_e_final, "e_final")

    @property
    def e_running_max(self) -> float:
        return self._exp(self.log_e_running_max, "e_running_max")

    def crossed_by(self, blocks: int) -> bool:
        """Did the process cross at or before `blocks` looks?"""
        if isinstance(blocks, bool) or not isinstance(blocks, int) or blocks < 0:
            raise MaterialError("blocks must be a non-negative int")
        return self.first_crossing_block is not None and self.first_crossing_block <= blocks

    def to_dict(self) -> dict:
        return {"construction_id": self.construction_id, "hypothesis": self.hypothesis,
                "margin": self.margin, "null_boundary": self.null_boundary,
                "threshold": self.threshold, "blocks": self.blocks,
                "log_e_final": self.log_e_final,
                "log_e_running_max": self.log_e_running_max,
                "first_crossing_block": self.first_crossing_block,
                "lambdas": list(self.lambdas), "signs": list(self.signs)}


def _log_wealth(oriented: Sequence[float], construction: EProcessConstruction,
                null_boundary: float) -> tuple:
    """Return (log_wealth_series, lambdas, signs). Pure; the whole e-process core.

    A block exactly ON the null boundary scores `-1`, not `0`. See
    `EProcessConstruction` — a three-valued sign makes `E[X_b] > 0` under the
    module's own stated null whenever the boundary carries an atom, which turns
    the wealth into a submartingale and voids Ville's bound. The tie is charged
    to the candidate: `E[X_b] = 2*P(> boundary) - 1 <= 0` holds with no
    continuity assumption at all.
    """
    log_w = 0.0
    count, total, total_sq = 0, 0.0, 0.0
    series, lambdas, signs = [], [], []
    for value in oriented:
        lam = construction.lambda_from_moments(count, total, total_sq)
        delta = value - null_boundary
        sign = 1.0 if delta > 0 else -1.0
        log_w += math.log1p(lam * sign)
        count += 1
        total += sign
        total_sq += sign * sign
        series.append(log_w)
        lambdas.append(lam)
        signs.append(sign)
    return tuple(series), tuple(lambdas), tuple(signs)


def run_e_process(oriented_effects: Sequence[float], *,
                  construction: EProcessConstruction,
                  hypothesis: str,
                  margin: float,
                  threshold: float) -> EProcessRun:
    """Evaluate the e-process over per-block oriented effects.

    `threshold` is `1/alpha` for the stratum and is passed in from
    `api.CalibrationOutputs.threshold_for()`. This function does not know how to
    produce a threshold, on purpose: the only place a threshold comes from is the
    calibration block.
    """
    if not isinstance(construction, EProcessConstruction):
        raise ConstructionNotImplemented("construction must be an EProcessConstruction")
    values = _finite_floats(oriented_effects, "oriented_effects")
    if not values:
        raise InsufficientMaterial("an e-process over zero blocks is not a measurement")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) \
            or not math.isfinite(threshold) or threshold <= 1.0:
        raise MaterialError(
            f"threshold must be a finite number greater than 1, got {threshold!r}; a "
            "threshold at or below 1 rejects on no evidence at all")
    boundary = null_boundary_for(hypothesis, margin)
    series, lambdas, signs = _log_wealth(values, construction, boundary)
    log_threshold = math.log(float(threshold))
    first = None
    for i, log_w in enumerate(series):
        if log_w >= log_threshold:
            first = i + 1
            break
    return EProcessRun(
        construction_id=construction.construction_id, hypothesis=hypothesis,
        margin=float(margin), null_boundary=boundary, threshold=float(threshold),
        log_threshold=log_threshold, log_wealth=series, lambdas=lambdas, signs=signs,
        log_e_final=series[-1], log_e_running_max=max(series), first_crossing_block=first)


# =============================================================================
# The pre-committed stopping rule
# =============================================================================

#: The complete outcome set. `futility_stop` is reachable only when a futility
#: rule is declared, and the decision table must then name it too.
STOPPING_OUTCOMES = (
    "evidence_threshold_crossed",
    "extension_exhausted",
    "block_ceiling_reached",
    "futility_stop",
)

#: The decisions a search record may trigger. This is the protocol's own
#: enumeration of what it authorizes ("rank, retain, abandon, branch, compose,
#: select the next experiment, request a readiness computation") and nothing
#: else. A stopping rule that declares "promote" or "deploy" as an outcome's
#: decision is a search record gating a decision denial 1 forbids, and it is
#: refused at rule-construction time rather than at the moment it is acted on.
AUTHORIZED_DECISIONS = (
    "rank_against_anchor",
    "retain",
    "abandon",
    "branch",
    "compose_into_champion_lineage",
    "select_next_experiment",
    "request_readiness_computation",
    "record_only",
)

#: The only implemented futility form: the threshold is unreachable within the
#: remaining declared budget even if every remaining block were maximally
#: favourable. Exact, parameter-free, and it cannot inflate the false-positive
#: rate because it only ever stops runs that were going to fail to cross.
FUTILITY_UNREACHABLE_THRESHOLD = "unreachable_threshold_within_declared_budget"


@dataclass(frozen=True)
class BoundedExtension:
    """*"a bounded number of fresh reversed-order pairs pooled to a pre-declared threshold."*

    Every field is a finite int and there is NO representation of an unbounded
    extension: `max_rounds` accepts neither `None`, nor a float, nor
    `math.inf`. `order` accepts only `"reversed"` and `pooled` only `True`,
    because `bench-cpu.md:85-86` — the manner the protocol names — is a fresh
    REVERSED-order pair POOLED to the pre-declared threshold, not a fresh
    independent test at a fresh threshold.

    This is the object that makes "extend while it might still change"
    unrepresentable.
    """

    max_rounds: int
    blocks_per_round: int
    order: str = "reversed"
    pooled: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.max_rounds, bool) or not isinstance(self.max_rounds, int):
            raise StoppingRuleViolation(
                f"extension.max_rounds must be an int, got {self.max_rounds!r}. There is no "
                "unbounded extension: 'extend while the answer might still change' is "
                "exactly the non-conforming pattern the protocol names.")
        if self.max_rounds < 0:
            raise StoppingRuleViolation("extension.max_rounds must be >= 0")
        if isinstance(self.blocks_per_round, bool) \
                or not isinstance(self.blocks_per_round, int) or self.blocks_per_round < 1:
            raise StoppingRuleViolation("extension.blocks_per_round must be a positive int")
        if self.order != "reversed":
            raise StoppingRuleViolation(
                "extension.order must be 'reversed': the declared manner is a fresh "
                "REVERSED-order pair (bench-cpu.md:48-49, 85-86)")
        if self.pooled is not True:
            raise StoppingRuleViolation(
                "extension.pooled must be True: the extension pools to the PRE-DECLARED "
                "threshold; re-testing the extension alone at a fresh threshold is a "
                "second look that the declared rule does not license")

    def to_dict(self) -> dict:
        return {"max_rounds": self.max_rounds, "blocks_per_round": self.blocks_per_round,
                "order": self.order, "pooled": self.pooled}


@dataclass(frozen=True)
class FutilityRule:
    """A pre-committed early stop. Optional; when absent it can never fire."""

    kind: str = FUTILITY_UNREACHABLE_THRESHOLD

    def __post_init__(self) -> None:
        if self.kind != FUTILITY_UNREACHABLE_THRESHOLD:
            raise StoppingRuleViolation(
                f"futility kind {self.kind!r} is not implemented. The only implemented form "
                "is exact and parameter-free; a 'stop when it looks unpromising' form would "
                "need a threshold the protocol does not derive, and an undeclared threshold "
                "in a stopping rule is a post-hoc rule change waiting to happen.")

    def to_dict(self) -> dict:
        return {"kind": self.kind}


@dataclass(frozen=True)
class StoppingRule:
    """The pre-committed stopping rule — a calibration INPUT, held constant.

    *"Declared at campaign start … name the table that is FINAL, the decision
    each outcome triggers, the `max_blocks_per_candidate` ceiling, and the
    bounded extension rule."* All four are required fields, and the decision
    table must name EXACTLY the reachable outcomes — no missing outcome (which
    would leave a result with no declared decision) and no extra one (which would
    declare a decision for an outcome the rule cannot produce).
    """

    rule_id: str
    final_table: str
    decisions: tuple                      # ((outcome, decision), ...)
    extension: BoundedExtension
    max_blocks_per_candidate: int
    futility: Optional[FutilityRule] = None

    def __post_init__(self) -> None:
        for name in ("rule_id", "final_table"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise StoppingRuleViolation(
                    f"stopping_rule.{name} is required: the rule must NAME the table that is "
                    "FINAL (MEASUREMENT_POLICY.md:59-61)")
        if not isinstance(self.extension, BoundedExtension):
            raise StoppingRuleViolation("stopping_rule.extension must be a BoundedExtension")
        if isinstance(self.max_blocks_per_candidate, bool) \
                or not isinstance(self.max_blocks_per_candidate, int) \
                or self.max_blocks_per_candidate < 1:
            raise StoppingRuleViolation(
                "stopping_rule.max_blocks_per_candidate must be a positive int (precondition 8)")
        if self.futility is not None and not isinstance(self.futility, FutilityRule):
            raise StoppingRuleViolation("stopping_rule.futility must be a FutilityRule or None")
        if not isinstance(self.decisions, tuple):
            raise StoppingRuleViolation(
                "stopping_rule.decisions must be a tuple of (outcome, decision) pairs")
        seen = {}
        for pair in self.decisions:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise StoppingRuleViolation(
                    "each stopping_rule.decisions entry must be an (outcome, decision) pair")
            outcome, decision = pair
            if outcome not in STOPPING_OUTCOMES:
                raise StoppingRuleViolation(
                    f"stopping_rule.decisions: {outcome!r} is not one of "
                    f"{list(STOPPING_OUTCOMES)}")
            if outcome in seen:
                raise StoppingRuleViolation(
                    f"stopping_rule.decisions: {outcome!r} declared twice")
            if decision not in AUTHORIZED_DECISIONS:
                raise StoppingRuleViolation(
                    f"stopping_rule.decisions[{outcome!r}] = {decision!r} is not one of "
                    f"{list(AUTHORIZED_DECISIONS)}. P-AK-SEARCH-1 denial 1: a search record "
                    "MUST NOT gate any keep / revert / deploy / promote / buy / close "
                    "decision.")
            seen[outcome] = decision
        required = {"evidence_threshold_crossed", "extension_exhausted",
                    "block_ceiling_reached"}
        if self.futility is not None:
            required.add("futility_stop")
        missing = sorted(required - set(seen))
        extra = sorted(set(seen) - required)
        if missing:
            raise StoppingRuleViolation(
                f"stopping_rule.decisions does not name the decision for {missing}; the rule "
                "must state the decision each outcome triggers")
        if extra:
            raise StoppingRuleViolation(
                f"stopping_rule.decisions names {extra}, which this rule cannot produce "
                "(futility is not declared)")

    def decision_for(self, outcome: str) -> str:
        for declared, decision in self.decisions:
            if declared == outcome:
                return decision
        raise StoppingRuleViolation(
            f"outcome {outcome!r} has no declared decision in rule {self.rule_id!r}")

    def max_total_blocks(self, b_min: int) -> int:
        """Total blocks the rule can ever license for one candidate at `b_min`.

        `b_min` ABOVE the ceiling is refused rather than clamped. Clamping
        returned a window SHORTER than the base segment it was supposed to
        contain, and every consumer of that window then failed with a message
        about internal replay bookkeeping — *"replay needs at least b_min=21
        blocks, got 20"* — instead of naming the finding, which is that the run
        used more blocks than its declared `max_blocks_per_candidate`. That is
        the over-extension the protocol most needs journaled: *"Extension follows
        the declared rule only … Any post-hoc change to the stopping rule voids
        every affected record."*
        """
        if isinstance(b_min, bool) or not isinstance(b_min, int) or b_min < 1:
            raise StoppingRuleViolation("b_min must be a positive int")
        if b_min > self.max_blocks_per_candidate:
            raise StoppingRuleViolation(
                f"{b_min} blocks exceeds the declared ceiling "
                f"max_blocks_per_candidate={self.max_blocks_per_candidate} of rule "
                f"{self.rule_id!r}; the rule cannot license a base segment larger than "
                "the ceiling, so there is no window to evaluate")
        planned = b_min + self.extension.max_rounds * self.extension.blocks_per_round
        return min(planned, self.max_blocks_per_candidate)

    def to_dict(self) -> dict:
        return {"rule_id": self.rule_id, "final_table": self.final_table,
                "decisions": [[o, d] for o, d in self.decisions],
                "extension": self.extension.to_dict(),
                "max_blocks_per_candidate": self.max_blocks_per_candidate,
                "futility": None if self.futility is None else self.futility.to_dict()}

    def content_hash(self) -> str:
        return schemas.content_hash(self.to_dict())


@dataclass(frozen=True)
class StoppingRuleCommitment:
    """The rule's hash, recorded at campaign start. The detector for a post-hoc change."""

    campaign_id: str
    rule_id: str
    rule_content_hash: str
    committed_at: str

    def __post_init__(self) -> None:
        for name in ("campaign_id", "rule_id", "committed_at"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise MaterialError(f"commitment.{name} must be a non-empty string")
        if not isinstance(self.rule_content_hash, str) \
                or len(self.rule_content_hash) != 64 \
                or any(c not in "0123456789abcdef" for c in self.rule_content_hash):
            raise MaterialError("commitment.rule_content_hash must be a lowercase sha256 hex")

    @classmethod
    def commit(cls, rule: StoppingRule, *, campaign_id: str,
               committed_at: str) -> "StoppingRuleCommitment":
        if not isinstance(rule, StoppingRule):
            raise MaterialError("commit() takes a StoppingRule")
        return cls(campaign_id=campaign_id, rule_id=rule.rule_id,
                   rule_content_hash=rule.content_hash(), committed_at=committed_at)

    def verify(self, rule: StoppingRule) -> schemas.Check:
        """PASS only when the rule is byte-for-byte the one that was committed."""
        if not isinstance(rule, StoppingRule):
            return schemas.Check(schemas.COULD_NOT_CHECK,
                                 ("no StoppingRule was supplied to verify against the "
                                  "commitment",))
        reasons = []
        if rule.rule_id != self.rule_id:
            reasons.append(f"rule id {rule.rule_id!r} is not the committed {self.rule_id!r}")
        actual = rule.content_hash()
        if actual != self.rule_content_hash:
            reasons.append(
                f"stopping-rule content hash {actual[:12]} does not match the hash committed "
                f"at campaign start ({self.rule_content_hash[:12]}); any post-hoc change to "
                "the stopping rule VOIDS every affected record")
        return schemas.Check(schemas.FAIL, tuple(reasons)) if reasons \
            else schemas.Check(schemas.PASS)

    def to_dict(self) -> dict:
        return {"campaign_id": self.campaign_id, "rule_id": self.rule_id,
                "rule_content_hash": self.rule_content_hash,
                "committed_at": self.committed_at}


@dataclass(frozen=True)
class LookResult:
    """One inspection of the evidence. Anytime-validity licenses this; it does
    not license changing the rule between looks."""

    blocks_submitted: int
    is_look: bool
    log_e_running_max: float
    log_threshold: float
    crossed: bool
    terminal: bool
    outcome: Optional[str]


@dataclass(frozen=True)
class BlockRequest:
    """The rule TELLING the caller what the next block must be."""

    block_index: int
    order: str
    segment: str
    extension_round: Optional[int]


@dataclass(frozen=True)
class StopDecision:
    """The terminal outcome and the decision the pre-committed rule declared for it."""

    outcome: str
    decision: str
    blocks_used: int
    log_e_running_max: float
    threshold: float
    crossed: bool
    rule_id: str
    final_table: str
    extension_rounds_used: int

    def to_dict(self) -> dict:
        return {"outcome": self.outcome, "decision": self.decision,
                "blocks_used": self.blocks_used,
                "log_e_running_max": self.log_e_running_max, "threshold": self.threshold,
                "crossed": self.crossed, "rule_id": self.rule_id,
                "final_table": self.final_table,
                "extension_rounds_used": self.extension_rounds_used}


def _replay(oriented: Sequence[float], *, rule: StoppingRule, b_min: int,
            construction: EProcessConstruction, null_boundary: float,
            log_threshold: float) -> tuple:
    """Replay the declared rule over a fixed sequence of oriented effects.

    Returns `(outcome, blocks_used, crossed, rounds_used)`. This is the SAME
    control flow `SequentialEvaluation` walks incrementally; the calibration
    block replays the rule over A/A material with the rule held fixed, and it
    must be the same rule or the calibrated error budget is about a different
    procedure than the one that runs.
    """
    total = rule.max_total_blocks(b_min)
    if len(oriented) < b_min:
        raise InsufficientMaterial(
            f"replay needs at least b_min={b_min} blocks, got {len(oriented)}")
    series, _lam, _sign = _log_wealth(oriented[:total], construction, null_boundary)
    k = b_min
    rounds = 0
    best_step = math.log1p(construction.lambda_cap)
    while True:
        prefix_max = max(series[:k])
        if prefix_max >= log_threshold:
            return "evidence_threshold_crossed", k, True, rounds
        if rule.futility is not None:
            remaining = total - k
            if series[k - 1] + remaining * best_step < log_threshold:
                return "futility_stop", k, False, rounds
        if rounds >= rule.extension.max_rounds:
            return "extension_exhausted", k, False, rounds
        nxt = k + rule.extension.blocks_per_round
        if nxt > rule.max_blocks_per_candidate:
            return "block_ceiling_reached", k, False, rounds
        if nxt > len(series):
            raise InsufficientMaterial(
                f"replay needs {nxt} blocks to run extension round {rounds + 1}, "
                f"only {len(series)} available")
        k = nxt
        rounds += 1


class SequentialEvaluation:
    """Drive one candidate's blocks through the pre-committed rule.

    The point of this class is what it does NOT expose. There is no "add another
    block" method a caller can call at will: `next_block_request()` is the only
    source of a next block, it raises once the rule has terminated, and it
    dictates the order and the segment. A caller that wants to keep going
    because the answer might still change gets `StoppingRuleViolation`, which is
    the protocol's own sentence turned into a control-flow fact.

    The commitment hash is re-verified on EVERY submission, so a rule mutated
    mid-run is caught at the next block rather than at the end of the campaign.
    """

    def __init__(self, *,
                 rule: StoppingRule,
                 commitment: StoppingRuleCommitment,
                 construction: EProcessConstruction,
                 b_min: int,
                 threshold: float,
                 hypothesis: str,
                 margin: float,
                 metric_direction: str,
                 effect_scale: str,
                 order_schedule: OrderSchedule) -> None:
        if not isinstance(rule, StoppingRule):
            raise StoppingRuleViolation("rule must be a StoppingRule")
        if not isinstance(commitment, StoppingRuleCommitment):
            raise StoppingRuleViolation("commitment must be a StoppingRuleCommitment")
        verified = commitment.verify(rule)
        if verified.outcome != schemas.PASS:
            raise StoppingRuleMutated("; ".join(verified.reasons))
        _require_bundle_construction(construction, "sequential_evaluation.construction")
        if isinstance(b_min, bool) or not isinstance(b_min, int) or b_min < 1:
            raise MaterialError("b_min must be a positive int from the calibration block")
        if b_min > rule.max_blocks_per_candidate:
            raise StoppingRuleViolation(
                f"b_min {b_min} exceeds max_blocks_per_candidate "
                f"{rule.max_blocks_per_candidate}")
        if not isinstance(order_schedule, OrderSchedule):
            raise MaterialError("order_schedule must be an OrderSchedule")
        if order_schedule.base_blocks != b_min:
            raise MaterialError(
                f"order schedule was derived for {order_schedule.base_blocks} base blocks "
                f"but b_min is {b_min}; the base segment is exactly b_min blocks")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) \
                or not math.isfinite(threshold) or threshold <= 1.0:
            raise MaterialError("threshold must come from CalibrationOutputs.threshold_for()")
        if effect_scale not in EFFECT_SCALES:
            raise EffectScaleError(f"effect_scale: {effect_scale!r} is not one of "
                                   f"{list(EFFECT_SCALES)}")
        self._rule = rule
        self._commitment = commitment
        self._construction = construction
        self._b_min = b_min
        self._threshold = float(threshold)
        self._log_threshold = math.log(float(threshold))
        self._hypothesis = hypothesis
        self._margin = float(margin)
        self._null_boundary = null_boundary_for(hypothesis, margin)
        self._direction = metric_direction
        self._scale = effect_scale
        self._schedule = order_schedule
        self._blocks: list = []
        self._oriented: list = []
        self._target = b_min
        self._rounds = 0
        self._terminal = False
        self._outcome: Optional[str] = None
        self._issued: Optional[BlockRequest] = None
        # Validates metric_direction eagerly.
        orient(0.0, metric_direction)

    # -- state ------------------------------------------------------------
    @property
    def blocks(self) -> tuple:
        return tuple(self._blocks)

    @property
    def terminal(self) -> bool:
        return self._terminal

    @property
    def outcome(self) -> Optional[str]:
        return self._outcome

    @property
    def extension_rounds_used(self) -> int:
        return self._rounds

    @property
    def b_min(self) -> int:
        return self._b_min

    def next_block_request(self) -> BlockRequest:
        """The next block the DECLARED RULE licenses. Raises once it has terminated."""
        if self._terminal:
            raise StoppingRuleViolation(
                f"rule {self._rule.rule_id!r} terminated with outcome {self._outcome!r} after "
                f"{len(self._blocks)} blocks. Extension follows the declared rule ONLY; "
                "continuing because the answer might still change is the non-conforming "
                "pattern the protocol names, and it voids the record.")
        index = len(self._blocks)
        if index >= self._b_min:
            segment = SEGMENT_EXTENSION
            round_no = self._rounds
        else:
            segment = SEGMENT_BASE
            round_no = None
        self._issued = BlockRequest(block_index=index, order=self._schedule.order_for(index),
                                    segment=segment, extension_round=round_no)
        return self._issued

    def submit_block(self, block: PairedBlock) -> LookResult:
        """Submit the block the rule asked for. Re-verifies the commitment first."""
        if self._terminal:
            raise StoppingRuleViolation(
                f"rule {self._rule.rule_id!r} already terminated with {self._outcome!r}; "
                "a block submitted after termination is an undeclared extra look")
        verified = self._commitment.verify(self._rule)
        if verified.outcome != schemas.PASS:
            raise StoppingRuleMutated("; ".join(verified.reasons))
        if not isinstance(block, PairedBlock):
            raise MaterialError("submit_block() takes a PairedBlock")
        if self._issued is None:
            raise StoppingRuleViolation(
                "no block was requested; call next_block_request() first. A block the rule "
                "did not ask for is a look the rule did not license.")
        request = self._issued
        mismatches = []
        if block.block_index != request.block_index:
            mismatches.append(f"index {block.block_index} != requested {request.block_index}")
        if block.order != request.order:
            mismatches.append(f"order {block.order!r} != requested {request.order!r}")
        if block.segment != request.segment:
            mismatches.append(f"segment {block.segment!r} != requested {request.segment!r}")
        if block.extension_round != request.extension_round:
            mismatches.append(
                f"extension_round {block.extension_round!r} != requested "
                f"{request.extension_round!r}")
        if mismatches:
            raise StoppingRuleViolation(
                f"submitted block does not match the block the rule requested: "
                f"{'; '.join(mismatches)}")
        self._issued = None
        self._blocks.append(block)
        self._oriented.append(orient(block_effect(block, scale=self._scale), self._direction))
        return self._evaluate_look()

    def _evaluate_look(self) -> LookResult:
        n = len(self._blocks)
        series, _lam, _sign = _log_wealth(self._oriented, self._construction,
                                          self._null_boundary)
        running_max = max(series)
        if n < self._target:
            return LookResult(blocks_submitted=n, is_look=False,
                              log_e_running_max=running_max,
                              log_threshold=self._log_threshold,
                              crossed=running_max >= self._log_threshold,
                              terminal=False, outcome=None)
        crossed = running_max >= self._log_threshold
        if crossed:
            return self._terminate("evidence_threshold_crossed", n, running_max, True)
        if self._rule.futility is not None:
            total = self._rule.max_total_blocks(self._b_min)
            remaining = total - n
            best = series[-1] + remaining * math.log1p(self._construction.lambda_cap)
            if best < self._log_threshold:
                return self._terminate("futility_stop", n, running_max, False)
        if self._rounds >= self._rule.extension.max_rounds:
            return self._terminate("extension_exhausted", n, running_max, False)
        nxt = n + self._rule.extension.blocks_per_round
        if nxt > self._rule.max_blocks_per_candidate:
            return self._terminate("block_ceiling_reached", n, running_max, False)
        self._target = nxt
        self._rounds += 1
        return LookResult(blocks_submitted=n, is_look=True, log_e_running_max=running_max,
                          log_threshold=self._log_threshold, crossed=False, terminal=False,
                          outcome=None)

    def _terminate(self, outcome: str, n: int, running_max: float,
                   crossed: bool) -> LookResult:
        self._terminal = True
        self._outcome = outcome
        self._issued = None
        return LookResult(blocks_submitted=n, is_look=True, log_e_running_max=running_max,
                          log_threshold=self._log_threshold, crossed=crossed, terminal=True,
                          outcome=outcome)

    def decide(self) -> StopDecision:
        """The declared decision for the realized outcome. Raises before termination."""
        if not self._terminal or self._outcome is None:
            raise StoppingRuleViolation(
                f"rule {self._rule.rule_id!r} has not terminated after "
                f"{len(self._blocks)} blocks; a decision taken at an undeclared look is "
                "peeking, not a decision")
        series, _lam, _sign = _log_wealth(self._oriented, self._construction,
                                          self._null_boundary)
        return StopDecision(
            outcome=self._outcome, decision=self._rule.decision_for(self._outcome),
            blocks_used=len(self._blocks), log_e_running_max=max(series),
            threshold=self._threshold,
            crossed=self._outcome == "evidence_threshold_crossed",
            rule_id=self._rule.rule_id, final_table=self._rule.final_table,
            extension_rounds_used=self._rounds)

    def e_process(self) -> EProcessRun:
        """The e-process as it stands. Reading it is a look; it changes nothing."""
        return run_e_process(self._oriented, construction=self._construction,
                             hypothesis=self._hypothesis, margin=self._margin,
                             threshold=self._threshold)


# =============================================================================
# Calibration output 1 — the campaign noise floor phi
# =============================================================================

_PHI_QUANTILE = 0.95          # the protocol's own "95th percentile of |effect|"
_BAND_MASS = 0.95             # the protocol's own "central 95%"


@dataclass(frozen=True)
class NoiseFloor:
    """Output 1. A property of the INSTRUMENT under this host state, not of a candidate."""

    value: float
    blocks: int
    quantile: float
    method: str
    declared_calibration_block_count: int
    neutral_check: schemas.Check

    def to_dict(self) -> dict:
        return {"value": self.value, "blocks": self.blocks, "quantile": self.quantile,
                "method": self.method,
                "declared_calibration_block_count": self.declared_calibration_block_count,
                "neutral_check": {"outcome": self.neutral_check.outcome,
                                  "reasons": list(self.neutral_check.reasons)}}


def estimate_noise_floor(aa_effects: Sequence[float], *,
                         calibration_block_count: int,
                         neutral_check: schemas.Check) -> NoiseFloor:
    """phi = the 95th percentile of the A/A `|effect|` distribution.

    *"where each A/A effect is computed by the same reducer, at the same block
    size, as a candidate effect"* — the caller passes per-block effects produced
    by `block_effect`, which is the same function the candidate reduction uses.

    Raises when there is less material than the campaign declared, and raises
    when there is too little material for a 95th percentile to be anything other
    than the maximum. A floor computed from 8 observations is the largest of 8
    observations, and using it would make every candidate look significant
    against a floor that is really an outlier.
    """
    values = _finite_floats(aa_effects, "aa_effects")
    if isinstance(calibration_block_count, bool) \
            or not isinstance(calibration_block_count, int) or calibration_block_count < 1:
        raise MaterialError("calibration_block_count must be a positive int (precondition 8)")
    if len(values) < calibration_block_count:
        raise InsufficientMaterial(
            f"the A/A control produced {len(values)} block effects but the campaign declared "
            f"calibration_block_count={calibration_block_count}; phi is estimated over AT "
            "LEAST the declared count")
    floor_n = min_samples_for_quantile(_PHI_QUANTILE)
    if len(values) < floor_n:
        raise InsufficientMaterial(
            f"a {_PHI_QUANTILE:.2f} quantile needs at least {floor_n} observations to be "
            f"anything other than the maximum; got {len(values)}")
    if not isinstance(neutral_check, schemas.Check):
        raise MaterialError("neutral_check must be a schemas.Check (the neutral control's "
                            "consistency check is part of output 1, not an afterthought)")
    magnitudes = [abs(v) for v in values]
    value = percentile(magnitudes, _PHI_QUANTILE)
    if value <= 0.0:
        raise InsufficientMaterial(
            "the A/A control produced a zero noise floor: every A/A block effect was exactly "
            "zero, which means the instrument did not vary and the floor is not measurable. "
            "A zero floor would admit every candidate.")
    return NoiseFloor(value=value, blocks=len(values), quantile=_PHI_QUANTILE,
                      method=PERCENTILE_METHOD,
                      declared_calibration_block_count=calibration_block_count,
                      neutral_check=neutral_check)


def neutral_control_consistency(neutral_effects: Sequence[float],
                                aa_effects: Sequence[float], *,
                                campaign_seed: str,
                                construction: EProcessConstruction) -> schemas.Check:
    """*"a neutral control materially exceeding the A/A floor FAILS the calibration."*

    "Materially" is resolved WITHOUT a new constant, by a PERMUTATION reference.
    Under the null that the two controls disperse alike, the pooled magnitudes
    are exchangeable, so the pool is repeatedly re-split into groups of the
    observed sizes and the neutral-sized group's own 95th percentile forms the
    reference. The observed neutral p95 is compared against the 95th percentile
    of that reference. Both 95s are the protocol's own; nothing is invented, and
    the test is exact under exchangeability.

    Two constructions were tried and REJECTED, and the reason is recorded here
    because both look right and neither is:

      * bootstrapping the A/A p95 at the A/A's own (larger) sample size — it
        compares a noisy estimate against a tight reference, and fails every
        campaign whose neutral control has fewer blocks than its A/A control,
        which is every campaign;
      * bootstrapping the A/A p95 at the NEUTRAL's sample size — the bootstrap
        cannot produce a value above the A/A sample's maximum, so at an extreme
        quantile its upper tail is truncated. Measured against fresh samples from
        the same generator it alarmed at ~27% instead of 5%.

    The permutation form measures at 3-5% against fresh samples. Note that this
    is the MARGINAL rate: conditional on one unusually tight A/A realization the
    rate is higher, which is inherent to comparing against a single realized
    control and is why a failed calibration is retained and re-runnable rather
    than fatal to the campaign's material.

    Exceeding the reference FAILS. The floor is never raised to accommodate the
    neutral control, which is exactly what the clause forbids.
    """
    aa = [abs(v) for v in _finite_floats(aa_effects, "aa_effects")]
    neutral = [abs(v) for v in _finite_floats(neutral_effects, "neutral_effects")]
    floor_n = min_samples_for_quantile(_PHI_QUANTILE)
    if len(aa) < floor_n or len(neutral) < floor_n:
        return schemas.Check(
            schemas.COULD_NOT_CHECK,
            (f"the consistency check compares two {_PHI_QUANTILE:.2f} quantiles and needs at "
             f"least {floor_n} observations of each; got A/A={len(aa)}, "
             f"neutral={len(neutral)}",))
    phi = percentile(aa, _PHI_QUANTILE)
    rng = _rng(campaign_seed, "neutral_consistency", construction.construction_id,
               len(aa), len(neutral))
    pool = aa + neutral
    k = len(neutral)
    draws = []
    for _ in range(construction.neutral_permutation_reps):
        rng.shuffle(pool)
        draws.append(percentile(pool[:k], _PHI_QUANTILE))
    upper = percentile(draws, _PHI_QUANTILE)
    neutral_p95 = percentile(neutral, _PHI_QUANTILE)
    if neutral_p95 > upper:
        return schemas.Check(schemas.FAIL, (
            f"the neutral control's |effect| p95 ({neutral_p95:.6g}) materially exceeds the "
            f"A/A floor: the permutation reference for a group of {k} has 95th percentile "
            f"{upper:.6g} (phi={phi:.6g} over {len(aa)} A/A blocks). The calibration FAILS "
            "rather than raising the floor.",))
    return schemas.Check(schemas.PASS, (
        f"neutral p95 {neutral_p95:.6g} <= the permutation reference p95 {upper:.6g} for a "
        f"group of {k} (phi={phi:.6g} over {len(aa)} A/A blocks)",))


# =============================================================================
# Crossing rates — condition (a) of output 2, and the step-5 validation
# =============================================================================

@dataclass(frozen=True)
class CrossingRate:
    """A realized crossing rate of the campaign's own rule, with its resolution."""

    rate: float
    crossings: int
    windows: int
    block_count: int
    window_length: int
    threshold: float
    method: str
    resolution: float
    seed: Optional[int]

    def to_dict(self) -> dict:
        return {"rate": self.rate, "crossings": self.crossings, "windows": self.windows,
                "block_count": self.block_count, "window_length": self.window_length,
                "threshold": self.threshold, "method": self.method,
                "resolution": self.resolution, "seed": self.seed}


def required_disjoint_windows(alpha: float) -> int:
    """How many independent windows a rate at or below `alpha` needs to be shown.

    `ceil(1/alpha)`. Fewer windows cannot distinguish "the rate is at most
    alpha" from "we did not look enough times to see one" — 0/12 does not
    demonstrate a rate at or below 1/50.
    """
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) \
            or not 0.0 < alpha < 1.0:
        raise MaterialError(f"alpha must be in (0, 1), got {alpha!r}")
    return int(math.ceil(1.0 / alpha))


def resampled_crossing_rate(aa_oriented: Sequence[float], *, block_count: int,
                            rule: StoppingRule, construction: EProcessConstruction,
                            hypothesis: str, margin: float, threshold: float,
                            campaign_seed: str, shift: float = 0.0) -> CrossingRate:
    """Condition (a): the rule's crossing rate over RESAMPLED A/A windows.

    *"the realized crossing rate of the campaign's own stopping rule, evaluated
    over resampled A/A windows with the rule held fixed"*. `shift` is 0 for the
    false-positive rate; `solve_mde` reuses this with a positive shift to get
    power, on the SAME resampled windows (common random numbers), which is what
    makes the MDE search stable.
    """
    values = _finite_floats(aa_oriented, "aa_oriented")
    if not values:
        raise InsufficientMaterial("no A/A material to resample")
    window_length = rule.max_total_blocks(block_count)
    boundary = null_boundary_for(hypothesis, margin)
    log_threshold = math.log(float(threshold))
    seed = derive_seed(campaign_seed, "crossing_rate", construction.construction_id,
                       block_count, rule.content_hash())
    rng = random.Random(seed)
    resamples = construction.crossing_rate_resamples
    crossings = 0
    for _ in range(resamples):
        window = rng.choices(values, k=window_length)
        if shift:
            window = [v + shift for v in window]
        outcome, _n, crossed, _r = _replay(
            window, rule=rule, b_min=block_count, construction=construction,
            null_boundary=boundary, log_threshold=log_threshold)
        del outcome
        if crossed:
            crossings += 1
    return CrossingRate(rate=crossings / resamples, crossings=crossings, windows=resamples,
                        block_count=block_count, window_length=window_length,
                        threshold=float(threshold), method="resampled_aa_windows",
                        resolution=1.0 / resamples, seed=seed)


def empirical_crossing_rate(aa_oriented: Sequence[float], *, block_count: int,
                            rule: StoppingRule, construction: EProcessConstruction,
                            hypothesis: str, margin: float,
                            threshold: float) -> CrossingRate:
    """Step 5: the empirical validation over DISJOINT A/A windows, no resampling.

    Step 4 solves against resampled windows; step 5 validates *"empirically once,
    at the solved `B_min`"*. Resampling reuses the same blocks and cannot detect
    material that is simply too thin, so the validation uses disjoint consecutive
    windows of the full rule length and reports how many it had.
    """
    values = _finite_floats(aa_oriented, "aa_oriented")
    window_length = rule.max_total_blocks(block_count)
    n_windows = len(values) // window_length
    if n_windows < 1:
        raise InsufficientMaterial(
            f"the empirical validation needs at least one disjoint window of "
            f"{window_length} blocks; the A/A control has {len(values)}")
    boundary = null_boundary_for(hypothesis, margin)
    log_threshold = math.log(float(threshold))
    crossings = 0
    for w in range(n_windows):
        window = values[w * window_length:(w + 1) * window_length]
        _outcome, _n, crossed, _r = _replay(
            window, rule=rule, b_min=block_count, construction=construction,
            null_boundary=boundary, log_threshold=log_threshold)
        if crossed:
            crossings += 1
    return CrossingRate(rate=crossings / n_windows, crossings=crossings, windows=n_windows,
                        block_count=block_count, window_length=window_length,
                        threshold=float(threshold), method="disjoint_aa_windows",
                        resolution=1.0 / n_windows, seed=None)


# =============================================================================
# The MDE — computed from the calibrated dispersion and the block count, and
# published WITH the estimate because it never depends on the estimate
# =============================================================================

@dataclass(frozen=True)
class MinimumDetectableEffect:
    """The smallest effect this procedure detects at `block_count`, at target power.

    *"The minimum detectable effect is computed from the calibrated dispersion
    and the realized block count and is written into the same record as the
    estimate, not afterwards."* Every argument to `solve_mde` is either
    calibration material or the block count. The candidate's own blocks are NOT
    an input, which is what makes "published WITH the result" true rather than
    merely asserted — see `test_statistics.py`, which computes the MDE from two
    different candidate datasets and asserts it is the same number.
    """

    value: float
    block_count: int
    window_length: int
    power_target: float
    achieved_power: float
    resamples: int
    search_tolerance: float
    bracket_low: float
    bracket_high: float
    method: str
    construction_id: str
    found: bool
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {"value": self.value, "block_count": self.block_count,
                "window_length": self.window_length, "power_target": self.power_target,
                "achieved_power": self.achieved_power, "resamples": self.resamples,
                "search_tolerance": self.search_tolerance,
                "bracket_low": self.bracket_low, "bracket_high": self.bracket_high,
                "method": self.method, "construction_id": self.construction_id,
                "found": self.found, "reason": self.reason}


def solve_mde(aa_oriented: Sequence[float], *, block_count: int, rule: StoppingRule,
              construction: EProcessConstruction, hypothesis: str, margin: float,
              threshold: float, campaign_seed: str) -> MinimumDetectableEffect:
    """Smallest shift whose MEASURED detection rate reaches the construction's power.

    Method: common-random-number resampling. One set of A/A windows is drawn per
    `(block_count, rule, construction)` and every candidate shift is evaluated on
    the SAME windows, so the power curve is stable and a bracketing search is
    meaningful. The bracket is grown by doubling from the A/A dispersion; the
    returned value is the upper bracket, whose power was MEASURED at or above
    the target, so the answer is never an extrapolation.

    `found=False` (with a reason) when no shift within the doubling budget
    reaches the target power at this block count. That is a real answer — this
    block count cannot detect the campaign's contribution floor — and condition
    (b) of output 2 fails on it. It is never silently replaced by a number.
    """
    values = _finite_floats(aa_oriented, "aa_oriented")
    if not values:
        raise InsufficientMaterial("no A/A material from which to derive an MDE")
    window_length = rule.max_total_blocks(block_count)
    boundary = null_boundary_for(hypothesis, margin)
    log_threshold = math.log(float(threshold))
    seed = derive_seed(campaign_seed, "mde", construction.construction_id, block_count,
                       rule.content_hash())
    rng = random.Random(seed)
    windows = [rng.choices(values, k=window_length)
               for _ in range(construction.mde_resamples)]

    def power(shift: float) -> float:
        hits = 0
        for window in windows:
            shifted = [v + shift for v in window]
            _o, _n, crossed, _r = _replay(shifted, rule=rule, b_min=block_count,
                                          construction=construction,
                                          null_boundary=boundary,
                                          log_threshold=log_threshold)
            if crossed:
                hits += 1
        return hits / len(windows)

    scale = mad(values)
    if scale <= 0.0:
        scale = percentile([abs(v) for v in values], _PHI_QUANTILE)
    if scale <= 0.0:
        raise InsufficientMaterial(
            "the A/A control has zero dispersion; an MDE cannot be derived from it")
    target = construction.mde_power_target
    high = scale
    achieved = power(high)
    doublings = 0
    while achieved < target:
        doublings += 1
        if doublings > construction.mde_max_doublings:
            return MinimumDetectableEffect(
                value=high, block_count=block_count, window_length=window_length,
                power_target=target, achieved_power=achieved,
                resamples=construction.mde_resamples,
                search_tolerance=construction.mde_search_tolerance,
                bracket_low=0.0, bracket_high=high, method="common_random_number_resampling",
                construction_id=construction.construction_id, found=False,
                reason=(f"no shift up to {high:.6g} reached power {target} at "
                        f"block_count={block_count} within "
                        f"{construction.mde_max_doublings} doublings"))
        high *= 2.0
        achieved = power(high)
    low = 0.0
    # Invariant: power(low) < target <= power(high). Return `high`, whose power
    # was measured, so a non-monotone power curve cannot return an unmet value.
    while (high - low) > construction.mde_search_tolerance * max(high, scale):
        mid = (low + high) / 2.0
        mid_power = power(mid)
        if mid_power >= target:
            high, achieved = mid, mid_power
        else:
            low = mid
    return MinimumDetectableEffect(
        value=high, block_count=block_count, window_length=window_length,
        power_target=target, achieved_power=achieved, resamples=construction.mde_resamples,
        search_tolerance=construction.mde_search_tolerance, bracket_low=low, bracket_high=high,
        method="common_random_number_resampling",
        construction_id=construction.construction_id, found=True)


# =============================================================================
# Calibration output 4 — the anchor-gate acceptance band
# =============================================================================

@dataclass(frozen=True)
class AnchorGateBand:
    """Output 4: the central 95% of the anchor cell's own calibration values at B_min."""

    low: float
    high: float
    mass: float
    b_min: int
    source_values: int
    resamples: int
    method: str

    def as_tuple(self) -> tuple:
        return (self.low, self.high)

    def to_dict(self) -> dict:
        return {"low": self.low, "high": self.high, "mass": self.mass, "b_min": self.b_min,
                "source_values": self.source_values, "resamples": self.resamples,
                "method": self.method}


def anchor_gate_band(anchor_values: Sequence[float], *, b_min: int,
                     construction: EProcessConstruction,
                     campaign_seed: str) -> AnchorGateBand:
    """The interval containing the central 95% of B_min-block anchor reductions.

    The reduction is the same one a run uses (the median), so the band is
    directly comparable to what the anchor gate measures at window open.
    """
    values = _finite_floats(anchor_values, "anchor_values")
    if isinstance(b_min, bool) or not isinstance(b_min, int) or b_min < 1:
        raise MaterialError("b_min must be a positive int")
    floor_n = min_samples_for_quantile((1.0 + _BAND_MASS) / 2.0)
    if len(values) < floor_n:
        raise InsufficientMaterial(
            f"a central {_BAND_MASS:.0%} band needs at least {floor_n} anchor calibration "
            f"values; got {len(values)}")
    rng = _rng(campaign_seed, "anchor_band", construction.construction_id, b_min)
    draws = [median(rng.choices(values, k=b_min)) for _ in range(construction.band_resamples)]
    tail = (1.0 - _BAND_MASS) / 2.0
    low = percentile(draws, tail)
    high = percentile(draws, 1.0 - tail)
    if not low < high:
        raise InsufficientMaterial(
            f"the anchor calibration values produced a degenerate band ({low}, {high}); a "
            "band with no width cannot gate anything")
    return AnchorGateBand(low=low, high=high, mass=_BAND_MASS, b_min=b_min,
                          source_values=len(values), resamples=construction.band_resamples,
                          method=f"bootstrap_median_of_{b_min}_blocks/{PERCENTILE_METHOD}")


def anchor_gate_check(observed_anchor_samples: Sequence[float], *,
                      band: Any, b_min: int) -> schemas.Check:
    """*"The anchor cell is measured FIRST in every window … Outside the band ⇒ VOID."*

    COULD_NOT_CHECK — not PASS — when the observation has fewer than `b_min`
    samples, because the band was computed at `b_min` and a gate evaluated at a
    different reduction size is a different gate.

    `band` SHOULD be the `AnchorGateBand` or the `api.CalibrationOutputs` that
    carries the size the band was calibrated at. A bare `(low, high)` pair does
    not, and then `b_min` is whatever the caller asserts — which is enough to
    defeat the paragraph above by declaring `b_min=1` and gating a single
    unreduced sample against a band bootstrapped over medians of `B_min`. When
    the band knows its own size, a caller-supplied `b_min` that disagrees is
    COULD_NOT_CHECK; when it does not, the PASS says so, so "this gate was
    evaluated at a caller-asserted reduction size" is visible in the record
    rather than being an unstated assumption.
    """
    if isinstance(b_min, bool) or not isinstance(b_min, int) or b_min < 1:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            f"the anchor gate needs the positive int reduction size the band was "
            f"calibrated at, got b_min={b_min!r}",))
    calibrated_at: Optional[int] = None
    if isinstance(band, AnchorGateBand):
        calibrated_at, band = band.b_min, band.as_tuple()
    elif isinstance(band, api.CalibrationOutputs):
        calibrated_at, band = band.b_min_blocks, tuple(band.anchor_gate_band)
    if band is None or not isinstance(band, Sequence) or len(band) != 2:
        return schemas.Check(schemas.COULD_NOT_CHECK,
                             ("no calibrated anchor-gate band was supplied",))
    if calibrated_at is not None and calibrated_at != b_min:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            f"the band was calibrated at b_min={calibrated_at} but the gate was asked to "
            f"evaluate at {b_min}; a gate evaluated at a different reduction size is a "
            "different gate",))
    try:
        low, high = float(band[0]), float(band[1])
    except (TypeError, ValueError) as exc:
        return schemas.Check(schemas.COULD_NOT_CHECK,
                             (f"the anchor-gate band is not a pair of numbers: {exc}",))
    if not (math.isfinite(low) and math.isfinite(high)) or not low < high:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            f"the anchor-gate band ({low}, {high}) is not a finite (low < high) interval; "
            "a band with no width cannot gate anything",))
    try:
        values = _finite_floats(observed_anchor_samples, "observed_anchor_samples")
    except MaterialError as exc:
        return schemas.Check(schemas.COULD_NOT_CHECK, (str(exc),))
    if not values:
        return schemas.Check(schemas.COULD_NOT_CHECK,
                             ("the anchor cell was not measured at window open; absence of a "
                              "comparison is not evidence of equivalence",))
    if len(values) < b_min:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            f"the anchor gate observed {len(values)} samples but the band was calibrated at "
            f"b_min={b_min}; a gate evaluated at a different reduction size is a different "
            "gate",))
    observed = median(values)
    if low <= observed <= high:
        provenance = (f"the band's own calibration size b_min={calibrated_at}"
                      if calibrated_at is not None else
                      f"a CALLER-ASSERTED reduction size b_min={b_min}; the bare "
                      "(low, high) pair does not carry the size it was calibrated at, so "
                      "pass the AnchorGateBand or the CalibrationOutputs to bind them")
        return schemas.Check(schemas.PASS,
                             (f"anchor cell {observed:.6g} inside calibrated band "
                              f"[{low:.6g}, {high:.6g}], evaluated at {provenance}",))
    return schemas.Check(schemas.FAIL, (
        f"anchor cell {observed:.6g} is outside the calibrated acceptance band "
        f"[{low:.6g}, {high:.6g}]; the window is VOID and may not be reported. A drifted "
        "anchor says nothing whatever about the candidate, so this is NOT a candidate "
        "failure.",))


# =============================================================================
# The calibration solve — the normative order, executed and recorded
# =============================================================================

@dataclass(frozen=True)
class CalibrationInputs:
    """Everything the calibration block consumes. All of it fixed before the solve."""

    backend: str
    phase: str
    cell_class: str
    campaign_seed: str
    controls: api.CampaignControls
    stopping_rule: StoppingRule
    construction: EProcessConstruction
    effect_scale: str
    metric_direction: str
    hypothesis: str
    margin: float
    aa_blocks: tuple
    neutral_blocks: tuple
    anchor_calibration_values: tuple
    samples_ref: str
    owning_rep_rule: Optional[OwningProtocolRepRule] = None
    max_tightening_rounds: int = 8

    def __post_init__(self) -> None:
        for name in ("backend", "phase", "cell_class", "campaign_seed", "samples_ref"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise MaterialError(f"calibration_inputs.{name} must be a non-empty string")
        if not isinstance(self.controls, api.CampaignControls):
            raise MaterialError("calibration_inputs.controls must be an api.CampaignControls")
        if not isinstance(self.stopping_rule, StoppingRule):
            raise MaterialError("calibration_inputs.stopping_rule must be a StoppingRule")
        if not isinstance(self.construction, EProcessConstruction):
            raise ConstructionNotImplemented(
                "calibration_inputs.construction must be an EProcessConstruction from "
                "CONSTRUCTIONS")
        if CONSTRUCTIONS.get(self.construction.construction_id) is not self.construction:
            raise ConstructionNotImplemented(
                f"{self.construction.construction_id!r} is not the registry member of that "
                "id; a campaign selects among the constructions the bundle implements, it "
                "does not supply a modified one")
        if self.effect_scale not in EFFECT_SCALES:
            raise EffectScaleError(f"effect_scale: {self.effect_scale!r} is not one of "
                                   f"{list(EFFECT_SCALES)}")
        if self.hypothesis not in HYPOTHESES:
            raise MaterialError(f"hypothesis: {self.hypothesis!r} is not one of "
                                f"{list(HYPOTHESES)}")
        null_boundary_for(self.hypothesis, self.margin)
        orient(0.0, self.metric_direction)
        for name in ("aa_blocks", "neutral_blocks"):
            blocks = getattr(self, name)
            if not isinstance(blocks, tuple) or not blocks:
                raise MaterialError(f"calibration_inputs.{name} must be a non-empty tuple of "
                                    "PairedBlock")
            for b in blocks:
                if not isinstance(b, PairedBlock):
                    raise MaterialError(f"calibration_inputs.{name} must contain PairedBlocks")
        if not isinstance(self.anchor_calibration_values, tuple) \
                or not self.anchor_calibration_values:
            raise MaterialError(
                "calibration_inputs.anchor_calibration_values must be a non-empty tuple; "
                "output 4 is the anchor cell's OWN calibration values and cannot be "
                "substituted")
        _finite_floats(self.anchor_calibration_values, "anchor_calibration_values")
        if self.owning_rep_rule is not None \
                and not isinstance(self.owning_rep_rule, OwningProtocolRepRule):
            raise MaterialError("calibration_inputs.owning_rep_rule must be an "
                                "OwningProtocolRepRule or None")
        if isinstance(self.max_tightening_rounds, bool) \
                or not isinstance(self.max_tightening_rounds, int) \
                or self.max_tightening_rounds < 1:
            raise MaterialError("calibration_inputs.max_tightening_rounds must be a "
                                "positive int")

    def aa_effects(self) -> tuple:
        return tuple(block_effect(b, scale=self.effect_scale) for b in self.aa_blocks)

    def neutral_effects(self) -> tuple:
        return tuple(block_effect(b, scale=self.effect_scale) for b in self.neutral_blocks)

    def aa_oriented(self) -> tuple:
        return tuple(orient(e, self.metric_direction) for e in self.aa_effects())

    def relative_contribution_floor(self) -> float:
        """The contribution floor expressed as a fraction, for the P-BENCH-1 rep rule."""
        floor = self.controls.contribution_floor
        if self.effect_scale == EFFECT_SCALE_RELATIVE:
            return float(floor)
        anchor = median(self.anchor_calibration_values)
        if anchor <= 0:
            raise EffectScaleError(
                "an absolute contribution floor needs a strictly positive anchor median to "
                "be expressed as a relative effect for the P-BENCH-1 reps rule")
        return float(floor) / anchor


@dataclass(frozen=True)
class CalibrationAttempt:
    """One pass of the solve. Retained whether it succeeded or failed."""

    attempt: int
    alpha_sel: float
    alpha_conf: float
    threshold_sel: float
    threshold_conf: float
    reps_floor: RepsFloor
    start_blocks: int
    b_min: Optional[int]
    noise_floor: Optional[NoiseFloor]
    mde: Optional[MinimumDetectableEffect]
    condition_a: Optional[CrossingRate]
    alpha_validation: schemas.Check
    band: Optional[AnchorGateBand]
    accepted: bool
    reasons: tuple
    solve_order_recorded: tuple = api.CALIBRATION_SOLVE_ORDER

    def to_dict(self) -> dict:
        return {
            "attempt": self.attempt, "alpha_sel": self.alpha_sel,
            "alpha_conf": self.alpha_conf, "threshold_sel": self.threshold_sel,
            "threshold_conf": self.threshold_conf, "reps_floor": self.reps_floor.to_dict(),
            "start_blocks": self.start_blocks, "b_min": self.b_min,
            "noise_floor": None if self.noise_floor is None else self.noise_floor.to_dict(),
            "mde": None if self.mde is None else self.mde.to_dict(),
            "condition_a": None if self.condition_a is None else self.condition_a.to_dict(),
            "alpha_validation": {"outcome": self.alpha_validation.outcome,
                                 "reasons": list(self.alpha_validation.reasons)},
            "band": None if self.band is None else self.band.to_dict(),
            "accepted": self.accepted, "reasons": list(self.reasons),
            "solve_order_recorded": list(self.solve_order_recorded),
        }


@dataclass(frozen=True)
class CalibrationSolve:
    """The complete calibration record: every attempt, and the accepted outputs if any.

    *"both the failed and the accepted calibration are retained in the
    manifest."* `attempts` is that retention. `outputs` is `None` unless the
    whole conjunction held — there is no partial calibration.
    """

    inputs_digest: dict
    attempts: tuple
    outputs: Optional[api.CalibrationOutputs]
    aa_effect_pool: tuple
    anchor_calibration_values: tuple
    reasons: tuple

    @property
    def accepted(self) -> bool:
        return self.outputs is not None

    def require_accepted(self) -> api.CalibrationOutputs:
        """*"A campaign that cannot complete its calibration block MUST NOT rank any candidate."*"""
        if self.outputs is None:
            raise CalibrationFailed(
                "the calibration block was not accepted, so no candidate may be ranked. "
                "Reasons: " + "; ".join(self.reasons))
        return self.outputs

    def to_dict(self) -> dict:
        return {
            "inputs": self.inputs_digest,
            "attempts": [a.to_dict() for a in self.attempts],
            "outputs": None if self.outputs is None else self.outputs.to_dict(),
            "aa_effect_pool": list(self.aa_effect_pool),
            "anchor_calibration_values": list(self.anchor_calibration_values),
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "solve_order": list(api.CALIBRATION_SOLVE_ORDER),
            "module": STATISTICS_MODULE_ID,
        }


def solve_calibration(inputs: CalibrationInputs) -> CalibrationSolve:
    """Execute the calibration block in the protocol's NORMATIVE solve order.

    1. inputs fixed first (the stopping rule's shape and its ceiling);
    2. `alpha_sel` from `max_candidates`, `alpha_conf` from `alpha_sel` and
       `confirmation_admission_count`;
    3. `phi` estimated from the A/A control;
    4. `B_min` solved upward from the constitutional floor until conditions (a)
       and (b) both hold;
    5. `alpha_sel` validated empirically once at the solved `B_min`, tightening
       and restarting at step 2 on failure;
    6. the anchor-gate band computed at the solved `B_min`.

    Returns the solve — accepted or not — because the manifest must retain both.
    Never returns a partially calibrated `api.CalibrationOutputs`.
    """
    if not isinstance(inputs, CalibrationInputs):
        raise MaterialError("solve_calibration() takes a CalibrationInputs")

    controls = inputs.controls
    rule = inputs.stopping_rule
    construction = inputs.construction
    reasons: list = []
    attempts: list = []

    # --- step 1: inputs are fixed first ---------------------------------
    if rule.max_blocks_per_candidate != controls.max_blocks_per_candidate:
        reasons.append(
            f"the stopping rule's ceiling ({rule.max_blocks_per_candidate}) is not the "
            f"declared max_blocks_per_candidate ({controls.max_blocks_per_candidate}); the "
            "ceiling is a campaign input held constant for the whole solve")

    aa_effects = inputs.aa_effects()
    aa_oriented = inputs.aa_oriented()
    neutral_effects = inputs.neutral_effects()

    digest = {
        "backend": inputs.backend, "phase": inputs.phase, "cell_class": inputs.cell_class,
        "effect_scale": inputs.effect_scale, "metric_direction": inputs.metric_direction,
        "hypothesis": inputs.hypothesis, "margin": float(inputs.margin),
        "controls": {
            "calibration_block_count": controls.calibration_block_count,
            "contribution_floor": controls.contribution_floor,
            "max_candidates": controls.max_candidates,
            "confirmation_admission_count": controls.confirmation_admission_count,
            "max_blocks_per_candidate": controls.max_blocks_per_candidate,
            "storage_floor_bytes_free": controls.storage_floor_bytes_free,
        },
        "stopping_rule": rule.to_dict(),
        "stopping_rule_content_hash": rule.content_hash(),
        "construction": construction.to_dict(),
        "construction_content_hash": construction.content_hash(),
        "aa_blocks": len(inputs.aa_blocks),
        "neutral_blocks": len(inputs.neutral_blocks),
        "anchor_calibration_values": len(inputs.anchor_calibration_values),
        "samples_ref": inputs.samples_ref,
        "owning_rep_rule": (None if inputs.owning_rep_rule is None
                            else inputs.owning_rep_rule.to_dict()),
    }

    if reasons:
        return CalibrationSolve(inputs_digest=digest, attempts=(), outputs=None,
                                aa_effect_pool=aa_effects,
                                anchor_calibration_values=inputs.anchor_calibration_values,
                                reasons=tuple(reasons))

    # The neutral-control consistency check belongs to output 1 and is a hard
    # calibration failure, not a floor adjustment.
    neutral_check = neutral_control_consistency(
        neutral_effects, aa_effects, campaign_seed=inputs.campaign_seed,
        construction=construction)

    try:
        floor_relative = inputs.relative_contribution_floor()
        reps_floor = reps_floor_for_relative_effect(floor_relative)
        noise = estimate_noise_floor(aa_effects,
                                     calibration_block_count=controls.calibration_block_count,
                                     neutral_check=neutral_check)
    except (InsufficientMaterial, MaterialError, EffectScaleError) as exc:
        reasons.append(str(exc))
        return CalibrationSolve(inputs_digest=digest, attempts=(), outputs=None,
                                aa_effect_pool=aa_effects,
                                anchor_calibration_values=inputs.anchor_calibration_values,
                                reasons=tuple(reasons))

    if neutral_check.outcome != schemas.PASS:
        attempt = CalibrationAttempt(
            attempt=0, alpha_sel=controls.alpha_sel_ceiling(),
            alpha_conf=controls.alpha_conf_ceiling(controls.alpha_sel_ceiling()),
            threshold_sel=1.0 / controls.alpha_sel_ceiling(),
            threshold_conf=1.0 / controls.alpha_conf_ceiling(controls.alpha_sel_ceiling()),
            reps_floor=reps_floor, start_blocks=reps_floor.blocks, b_min=None,
            noise_floor=noise, mde=None, condition_a=None,
            alpha_validation=schemas.Check(
                schemas.COULD_NOT_CHECK,
                ("not reached: the neutral-control consistency check did not pass",)),
            band=None, accepted=False,
            reasons=tuple(f"neutral control: {r}" for r in neutral_check.reasons))
        reasons.extend(attempt.reasons)
        return CalibrationSolve(inputs_digest=digest, attempts=(attempt,), outputs=None,
                                aa_effect_pool=aa_effects,
                                anchor_calibration_values=inputs.anchor_calibration_values,
                                reasons=tuple(reasons))

    # --- steps 2-6, with the tightening restart -------------------------
    alpha_sel = controls.alpha_sel_ceiling()
    for attempt_no in range(inputs.max_tightening_rounds):
        alpha_conf = controls.alpha_conf_ceiling(alpha_sel)
        threshold_sel = 1.0 / alpha_sel
        threshold_conf = 1.0 / alpha_conf
        attempt_reasons: list = []

        start = _start_blocks(reps_floor, inputs.owning_rep_rule)
        try:
            candidates = _b_min_candidates(start, controls.max_blocks_per_candidate,
                                           inputs.owning_rep_rule)
        except CalibrationFailed as exc:
            attempts.append(CalibrationAttempt(
                attempt=attempt_no, alpha_sel=alpha_sel, alpha_conf=alpha_conf,
                threshold_sel=threshold_sel, threshold_conf=threshold_conf,
                reps_floor=reps_floor, start_blocks=start, b_min=None, noise_floor=noise,
                mde=None, condition_a=None,
                alpha_validation=schemas.Check(
                    schemas.COULD_NOT_CHECK,
                    ("not reached: no admissible block count exists",)),
                band=None, accepted=False, reasons=(str(exc),)))
            reasons.append(str(exc))
            break
        solved_b_min = None
        solved_mde = None
        solved_rate = None
        last_reason = None
        for blocks in candidates:
            rate = resampled_crossing_rate(
                aa_oriented, block_count=blocks, rule=rule, construction=construction,
                hypothesis=inputs.hypothesis, margin=inputs.margin,
                threshold=threshold_sel, campaign_seed=inputs.campaign_seed)
            if rate.rate > alpha_sel:
                last_reason = (f"at {blocks} blocks the rule's resampled A/A crossing rate is "
                               f"{rate.rate:.6g} > alpha_sel {alpha_sel:.6g} (condition a)")
                continue
            mde = solve_mde(aa_oriented, block_count=blocks, rule=rule,
                            construction=construction, hypothesis=inputs.hypothesis,
                            margin=inputs.margin, threshold=threshold_sel,
                            campaign_seed=inputs.campaign_seed)
            if not mde.found or mde.value > controls.contribution_floor:
                last_reason = (
                    f"at {blocks} blocks the MDE is "
                    f"{'unreachable' if not mde.found else f'{mde.value:.6g}'}, which is not "
                    f"<= the declared contribution_floor "
                    f"{controls.contribution_floor:.6g} (condition b)")
                continue
            solved_b_min, solved_mde, solved_rate = blocks, mde, rate
            break

        if solved_b_min is None:
            attempt_reasons.append(
                f"no block count in {candidates[0]}..{candidates[-1]} satisfies both "
                f"conditions; last: {last_reason}")
            attempts.append(CalibrationAttempt(
                attempt=attempt_no, alpha_sel=alpha_sel, alpha_conf=alpha_conf,
                threshold_sel=threshold_sel, threshold_conf=threshold_conf,
                reps_floor=reps_floor, start_blocks=start, b_min=None, noise_floor=noise,
                mde=None, condition_a=None,
                alpha_validation=schemas.Check(schemas.COULD_NOT_CHECK,
                                               ("not reached: no B_min was solved",)),
                band=None, accepted=False, reasons=tuple(attempt_reasons)))
            reasons.extend(attempt_reasons)
            reasons.append(
                "the calibration FAILS and the campaign does not start; there is no partial "
                "calibration and no fallback ceiling")
            break

        # --- step 5: empirical validation, once, at the solved B_min ----
        needed = required_disjoint_windows(alpha_sel)
        try:
            empirical = empirical_crossing_rate(
                aa_oriented, block_count=solved_b_min, rule=rule, construction=construction,
                hypothesis=inputs.hypothesis, margin=inputs.margin, threshold=threshold_sel)
        except InsufficientMaterial as exc:
            validation = schemas.Check(schemas.COULD_NOT_CHECK, (str(exc),))
            empirical = None
        else:
            if empirical.windows < needed:
                validation = schemas.Check(schemas.COULD_NOT_CHECK, (
                    f"the empirical validation had {empirical.windows} disjoint A/A windows "
                    f"of {empirical.window_length} blocks, but demonstrating a rate at or "
                    f"below alpha_sel={alpha_sel:.6g} needs at least {needed}. Declare a "
                    f"larger calibration_block_count "
                    f"(>= {needed * empirical.window_length}).",))
            elif empirical.rate > alpha_sel:
                validation = schemas.Check(schemas.FAIL, (
                    f"the A/A control replayed through the campaign's own stopping rule "
                    f"crossed at {empirical.rate:.6g} > alpha_sel {alpha_sel:.6g} over "
                    f"{empirical.windows} disjoint windows",))
            else:
                validation = schemas.Check(schemas.PASS, (
                    f"empirical A/A crossing rate {empirical.rate:.6g} <= alpha_sel "
                    f"{alpha_sel:.6g} over {empirical.windows} disjoint windows",))

        if validation.outcome != schemas.PASS:
            attempt_reasons.extend(validation.reasons)
            attempts.append(CalibrationAttempt(
                attempt=attempt_no, alpha_sel=alpha_sel, alpha_conf=alpha_conf,
                threshold_sel=threshold_sel, threshold_conf=threshold_conf,
                reps_floor=reps_floor, start_blocks=start, b_min=solved_b_min,
                noise_floor=noise, mde=solved_mde, condition_a=solved_rate,
                alpha_validation=validation, band=None, accepted=False,
                reasons=tuple(attempt_reasons)))
            if validation.outcome == schemas.COULD_NOT_CHECK:
                reasons.extend(attempt_reasons)
                reasons.append(
                    "the empirical validation of alpha_sel could not be performed; a "
                    "calibration whose error budget was never validated is not an accepted "
                    "calibration")
                break
            alpha_sel = alpha_sel / 2.0
            continue

        # --- step 6: the anchor-gate band at the solved B_min -----------
        try:
            band = anchor_gate_band(inputs.anchor_calibration_values, b_min=solved_b_min,
                                    construction=construction,
                                    campaign_seed=inputs.campaign_seed)
        except (InsufficientMaterial, MaterialError) as exc:
            attempt_reasons.append(str(exc))
            attempts.append(CalibrationAttempt(
                attempt=attempt_no, alpha_sel=alpha_sel, alpha_conf=alpha_conf,
                threshold_sel=threshold_sel, threshold_conf=threshold_conf,
                reps_floor=reps_floor, start_blocks=start, b_min=solved_b_min,
                noise_floor=noise, mde=solved_mde, condition_a=solved_rate,
                alpha_validation=validation, band=None, accepted=False,
                reasons=tuple(attempt_reasons)))
            reasons.extend(attempt_reasons)
            break

        outputs = api.CalibrationOutputs(
            backend=inputs.backend, phase=inputs.phase, cell_class=inputs.cell_class,
            noise_floor_phi=noise.value, b_min_blocks=solved_b_min, alpha_sel=alpha_sel,
            alpha_conf=alpha_conf, anchor_gate_band=band.as_tuple(), accepted=True,
            solve_order_recorded=api.CALIBRATION_SOLVE_ORDER,
            samples_ref=inputs.samples_ref,
            e_process_construction_id=construction.construction_id)
        relations = outputs.check_against_controls(controls)
        if relations.outcome != schemas.PASS:
            attempt_reasons.extend(relations.reasons)
            attempts.append(CalibrationAttempt(
                attempt=attempt_no, alpha_sel=alpha_sel, alpha_conf=alpha_conf,
                threshold_sel=threshold_sel, threshold_conf=threshold_conf,
                reps_floor=reps_floor, start_blocks=start, b_min=solved_b_min,
                noise_floor=noise, mde=solved_mde, condition_a=solved_rate,
                alpha_validation=validation, band=band, accepted=False,
                reasons=tuple(attempt_reasons)))
            reasons.extend(attempt_reasons)
            break

        attempts.append(CalibrationAttempt(
            attempt=attempt_no, alpha_sel=alpha_sel, alpha_conf=alpha_conf,
            threshold_sel=threshold_sel, threshold_conf=threshold_conf,
            reps_floor=reps_floor, start_blocks=start, b_min=solved_b_min, noise_floor=noise,
            mde=solved_mde, condition_a=solved_rate, alpha_validation=validation, band=band,
            accepted=True, reasons=()))
        return CalibrationSolve(inputs_digest=digest, attempts=tuple(attempts),
                                outputs=outputs, aa_effect_pool=aa_effects,
                                anchor_calibration_values=inputs.anchor_calibration_values,
                                reasons=())
    else:
        reasons.append(
            f"alpha_sel was tightened {inputs.max_tightening_rounds} times and the empirical "
            "validation still did not pass; the calibration FAILS")

    return CalibrationSolve(inputs_digest=digest, attempts=tuple(attempts), outputs=None,
                            aa_effect_pool=aa_effects,
                            anchor_calibration_values=inputs.anchor_calibration_values,
                            reasons=tuple(reasons))


def _start_blocks(reps_floor: RepsFloor,
                  owning: Optional[OwningProtocolRepRule]) -> int:
    """The constitutional floor the upward solve starts from."""
    if owning is None:
        return reps_floor.blocks
    if owning.kind == REP_RULE_FIXED:
        return owning.blocks
    return max(reps_floor.blocks, owning.blocks)


def _b_min_candidates(start: int, ceiling: int,
                      owning: Optional[OwningProtocolRepRule]) -> tuple:
    if owning is not None and owning.kind == REP_RULE_FIXED:
        if owning.blocks > ceiling:
            raise CalibrationFailed(
                f"{owning.protocol_id} fixes the count at exactly {owning.blocks} blocks, "
                f"which exceeds the declared max_blocks_per_candidate {ceiling}")
        return (owning.blocks,)
    if start > ceiling:
        raise CalibrationFailed(
            f"the constitutional rep floor is {start} blocks, above the declared "
            f"max_blocks_per_candidate {ceiling}; the campaign cannot start")
    return tuple(range(start, ceiling + 1))


# =============================================================================
# Selection / confirmation split
# =============================================================================

@dataclass(frozen=True)
class RotationSchedule:
    """*"Confirmation shapes and control seeds rotate on the schedule declared in
    the evaluator bundle."* Declared, not chosen per campaign."""

    schedule_id: str
    period_campaigns: int
    declared_in: str = "evaluator_bundle"

    def __post_init__(self) -> None:
        if not isinstance(self.schedule_id, str) or not self.schedule_id.strip():
            raise MaterialError("rotation.schedule_id must be a non-empty string")
        if isinstance(self.period_campaigns, bool) \
                or not isinstance(self.period_campaigns, int) or self.period_campaigns < 1:
            raise MaterialError("rotation.period_campaigns must be a positive int")
        if self.declared_in != "evaluator_bundle":
            raise MaterialError(
                "the rotation schedule is declared in the evaluator bundle; a schedule "
                "declared anywhere the loop can write is one the loop can rotate away from")

    def epoch_for(self, campaign_ordinal: int) -> int:
        if isinstance(campaign_ordinal, bool) or not isinstance(campaign_ordinal, int) \
                or campaign_ordinal < 0:
            raise MaterialError("campaign_ordinal must be a non-negative int")
        return campaign_ordinal // self.period_campaigns

    def to_dict(self) -> dict:
        return {"schedule_id": self.schedule_id, "period_campaigns": self.period_campaigns,
                "declared_in": self.declared_in}


@dataclass(frozen=True)
class StratumSplitRule:
    """The disjoint selection/confirmation partition, keyed on the campaign seed.

    *"The measurement material — shapes, seeds, and blocks — is partitioned into
    disjoint selection and confirmation strata by a rule recorded in the campaign
    manifest and keyed on the campaign seed, before the first candidate is
    measured."*

    The reason is mechanical, not ceremonial: selecting the maximum over many
    candidates biases the selected estimate upward, so the evidence that promotes
    a candidate is structurally unfit to report how ready it is. `assign()` is
    the partition; `check_*` are the ways a record can violate it.
    """

    rule_id: str
    campaign_seed: str
    confirmation_fraction: float
    rotation: RotationSchedule
    campaign_ordinal: int = 0

    def __post_init__(self) -> None:
        for name in ("rule_id", "campaign_seed"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise MaterialError(f"split_rule.{name} must be a non-empty string")
        frac = self.confirmation_fraction
        if isinstance(frac, bool) or not isinstance(frac, (int, float)) \
                or not 0.0 < frac < 1.0:
            raise MaterialError(
                "split_rule.confirmation_fraction must be in (0, 1): a fraction of 0 leaves "
                "the readiness signal with no evidence, and 1 leaves selection with none")
        if not isinstance(self.rotation, RotationSchedule):
            raise MaterialError("split_rule.rotation must be a RotationSchedule")
        if isinstance(self.campaign_ordinal, bool) \
                or not isinstance(self.campaign_ordinal, int) or self.campaign_ordinal < 0:
            raise MaterialError("split_rule.campaign_ordinal must be a non-negative int")

    @property
    def epoch(self) -> int:
        return self.rotation.epoch_for(self.campaign_ordinal)

    def _key(self) -> bytes:
        return hashlib.sha256(
            f"{self.campaign_seed}:{self.rule_id}:{self.epoch}".encode("utf-8")).digest()

    def assign(self, unit_id: str) -> str:
        """Deterministic stratum for one measurement-material unit."""
        if not isinstance(unit_id, str) or not unit_id.strip():
            raise MaterialError("unit_id must be a non-empty string")
        digest = hashlib.blake2b(unit_id.encode("utf-8"), key=self._key(),
                                 digest_size=8).digest()
        u = int.from_bytes(digest, "big") / float(1 << 64)
        return api.STRATUM_CONFIRMATION if u < self.confirmation_fraction \
            else api.STRATUM_SELECTION

    def partition(self, unit_ids: Sequence[str]) -> dict:
        out = {api.STRATUM_SELECTION: [], api.STRATUM_CONFIRMATION: []}
        for unit in unit_ids:
            out[self.assign(unit)].append(unit)
        return {k: tuple(v) for k, v in out.items()}

    def check_blocks(self, blocks: Sequence[PairedBlock]) -> schemas.Check:
        """One stratum only, and the stratum the rule assigns. Anything else is a
        strata violation, which makes the record `INVALID`."""
        if not blocks:
            return schemas.Check(schemas.COULD_NOT_CHECK,
                                 ("no blocks; the stratum rule is unevaluable",))
        reasons = []
        strata = {b.stratum for b in blocks}
        if len(strata) > 1:
            reasons.append(
                f"blocks span {sorted(strata)}: no block may serve both strata, and a record "
                "mixing strata is INVALID")
        for b in blocks:
            expected = self.assign(b.unit_id)
            if b.stratum != expected:
                reasons.append(
                    f"block {b.block_index} on unit {b.unit_id!r} is labelled "
                    f"{b.stratum!r} but the recorded split rule assigns it {expected!r}")
        return schemas.Check(schemas.FAIL, tuple(reasons)) if reasons \
            else schemas.Check(schemas.PASS)

    def check_planner_context(self, unit_ids: Sequence[str]) -> schemas.Check:
        """*"The confirmation stratum's contents MUST NOT appear in planner context."*"""
        leaked = [u for u in unit_ids if self.assign(u) == api.STRATUM_CONFIRMATION]
        if leaked:
            return schemas.Check(schemas.FAIL, (
                f"confirmation-stratum {_named(leaked, noun='unit(s)')} appear in planner "
                "context",))
        return schemas.Check(schemas.PASS)

    def check_proposal_targets(self, unit_ids: Sequence[str]) -> schemas.Check:
        """*"a proposal that targets a confirmation shape is rejected before it
        consumes a window."*"""
        targeted = [u for u in unit_ids if self.assign(u) == api.STRATUM_CONFIRMATION]
        if targeted:
            return schemas.Check(schemas.FAIL, (
                f"the proposal targets confirmation-stratum "
                f"{_named(targeted, noun='shape(s)')}; it is rejected BEFORE it consumes a "
                "window",))
        return schemas.Check(schemas.PASS)

    def check_confirmation_admissible(self, blocks: Sequence[PairedBlock], *,
                                      lineage_entry_at: Optional[str]) -> schemas.Check:
        """Readiness evidence must be confirmation-stratum AND gathered AFTER entry.

        *"The readiness signal is computed ONLY from confirmation-stratum evidence
        gathered after the candidate entered the lineage."* Returns
        COULD_NOT_CHECK — never PASS — when the blocks do not carry the timestamps
        that would settle it.
        """
        if not blocks:
            return schemas.Check(schemas.COULD_NOT_CHECK, ("no blocks",))
        wrong = [b.block_index for b in blocks if b.stratum != api.STRATUM_CONFIRMATION]
        if wrong:
            return schemas.Check(schemas.FAIL, (
                f"{_named(wrong)} are not confirmation-stratum evidence; selection "
                "evidence is upward-biased by the very act of selection and MUST NOT report "
                "readiness",))
        if lineage_entry_at is None:
            return schemas.Check(schemas.COULD_NOT_CHECK, (
                "the candidate's lineage-entry time was not supplied, so 'gathered after the "
                "candidate entered the lineage' cannot be checked",))
        entry = _parse_instant(lineage_entry_at)
        if entry is None:
            return schemas.Check(schemas.COULD_NOT_CHECK, (
                f"the lineage-entry time {lineage_entry_at!r} is not an ISO-8601 timestamp "
                "with a UTC offset, so nothing can be ordered against it",))
        missing = [b.block_index for b in blocks if b.measured_at is None]
        if missing:
            return schemas.Check(schemas.COULD_NOT_CHECK, (
                f"{_named(missing)} carry no measured_at, so their ordering against "
                "lineage entry cannot be checked",))
        unparseable = [b.block_index for b in blocks
                       if _parse_instant(b.measured_at) is None]
        if unparseable:
            return schemas.Check(schemas.COULD_NOT_CHECK, (
                f"{_named(unparseable)} carry a measured_at that is not an ISO-8601 "
                "timestamp with a UTC offset; an instant that cannot be ordered is not an "
                "instant that is later than lineage entry",))
        early = [b.block_index for b in blocks
                 if _parse_instant(b.measured_at) <= entry]
        if early:
            return schemas.Check(schemas.FAIL, (
                f"{_named(early)} were measured at or before the candidate entered the "
                f"lineage ({lineage_entry_at}); that evidence is part of what selected it",))
        return schemas.Check(schemas.PASS)

    def to_dict(self) -> dict:
        return {"rule_id": self.rule_id, "confirmation_fraction": self.confirmation_fraction,
                "rotation": self.rotation.to_dict(),
                "campaign_ordinal": self.campaign_ordinal, "epoch": self.epoch}


# =============================================================================
# The descriptive LCB — carried BESIDE the e-value, never as the test
# =============================================================================

@dataclass(frozen=True)
class DescriptiveLCB:
    """A magnitude summary for a human reader. Explicitly not a test.

    *"An LCB MAY be carried beside the e-value as a labelled descriptive
    statistic — a magnitude summary for a human reader — provided the record
    carries the e-value and its threshold, the LCB is labelled `descriptive`, and
    no decision in the enumerated authority is taken on it."*

    Shape adapted from `scripts/benchmark/architect_bench_analyze.py:54`
    (`bootstrap_ci`), a seeded paired bootstrap over fixed samples. That
    function's own limitation is the reason this object cannot be the test: it
    addresses a FIXED sample and says nothing about sequential looks, and a
    controller inspecting its evidence every round takes sequential looks by
    construction.
    """

    value: float
    level: float
    iterations: int
    seed: int
    label: str = "descriptive"
    is_a_test: bool = False
    warning: str = ("descriptive only: a bootstrap lower bound over a fixed sample is not "
                    "valid under the sequential looks this loop takes, and MUST NOT rank, "
                    "retain, abandon, branch, compose or contribute to readiness")

    def to_dict(self) -> dict:
        return {"value": self.value, "level": self.level, "iterations": self.iterations,
                "seed": self.seed, "label": self.label, "is_a_test": self.is_a_test,
                "warning": self.warning}


def descriptive_lcb(block_effects: Sequence[float], *, campaign_seed: str,
                    candidate_id: str, construction: EProcessConstruction,
                    level: float = 0.95) -> DescriptiveLCB:
    """Seeded paired bootstrap lower bound on the median block effect. Descriptive."""
    values = _finite_floats(block_effects, "block_effects")
    if not values:
        raise InsufficientMaterial("a bootstrap over zero blocks is not a summary")
    if isinstance(level, bool) or not isinstance(level, (int, float)) \
            or not 0.0 < level < 1.0:
        raise MaterialError(f"level must be in (0, 1), got {level!r}")
    seed = derive_seed(campaign_seed, "lcb", candidate_id, construction.construction_id,
                       len(values))
    rng = random.Random(seed)
    n = len(values)
    draws = [median(rng.choices(values, k=n))
             for _ in range(construction.lcb_bootstrap_iterations)]
    return DescriptiveLCB(value=percentile(draws, 1.0 - level), level=level,
                          iterations=construction.lcb_bootstrap_iterations, seed=seed)


# =============================================================================
# The campaign's statistical state, and the reducer that plugs into api
# =============================================================================

@dataclass(frozen=True)
class CampaignStatistics:
    """Everything a reduction needs, all of it fixed before the first candidate.

    Constructing this object is where the campaign's statistical commitments are
    checked against each other: the stopping rule must be the committed one, the
    calibration must be accepted, and the construction must be the one the
    calibration recorded. A campaign that ran its calibration under one
    construction and its candidates under another has a threshold that was never
    validated for the procedure that produced the evidence.

    `aa_effect_pool` and `anchor_calibration_values` are carried because the
    protocol requires the calibration block's raw samples to be retained in the
    manifest, and because the MDE is *"computed from the calibrated dispersion
    and the realized block count"* — the dispersion has to still be there at
    reduction time or the MDE could only be computed afterwards, which is exactly
    what the clause forbids.
    """

    campaign_id: str
    campaign_seed: str
    effect_scale: str
    hypothesis: str
    margin: float
    stopping_rule: StoppingRule
    stopping_rule_commitment: StoppingRuleCommitment
    split_rule: StratumSplitRule
    construction: EProcessConstruction
    calibration: api.CalibrationOutputs
    aa_effect_pool: tuple
    anchor_calibration_values: tuple
    owning_rep_rule: Optional[OwningProtocolRepRule] = None

    def __post_init__(self) -> None:
        for name in ("campaign_id", "campaign_seed"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise MaterialError(f"campaign_statistics.{name} must be a non-empty string")
        if self.effect_scale not in EFFECT_SCALES:
            raise EffectScaleError(f"effect_scale: {self.effect_scale!r} is not one of "
                                   f"{list(EFFECT_SCALES)}")
        if self.hypothesis not in HYPOTHESES:
            raise MaterialError(f"hypothesis: {self.hypothesis!r} is not one of "
                                f"{list(HYPOTHESES)}")
        null_boundary_for(self.hypothesis, self.margin)
        if not isinstance(self.stopping_rule, StoppingRule):
            raise MaterialError("campaign_statistics.stopping_rule must be a StoppingRule")
        if not isinstance(self.stopping_rule_commitment, StoppingRuleCommitment):
            raise MaterialError("campaign_statistics.stopping_rule_commitment is required; a "
                                "rule with no commitment cannot be shown to be unmodified")
        verified = self.stopping_rule_commitment.verify(self.stopping_rule)
        if verified.outcome != schemas.PASS:
            raise StoppingRuleMutated("; ".join(verified.reasons))
        if not isinstance(self.split_rule, StratumSplitRule):
            raise MaterialError("campaign_statistics.split_rule must be a StratumSplitRule")
        if self.split_rule.campaign_seed != self.campaign_seed:
            raise MaterialError(
                "the stratum split rule is keyed on a different seed than the campaign; the "
                "split must be keyed on the campaign seed committed before the first "
                "candidate was measured")
        _require_bundle_construction(self.construction,
                                     "campaign_statistics.construction")
        if not isinstance(self.calibration, api.CalibrationOutputs):
            raise MaterialError("campaign_statistics.calibration must be an "
                                "api.CalibrationOutputs produced by solve_calibration()")
        if not self.calibration.accepted:
            raise CalibrationFailed(
                "the calibration block for this cell was not accepted; a campaign that "
                "cannot complete its calibration block MUST NOT rank any candidate")
        if self.calibration.e_process_construction_id != self.construction.construction_id:
            raise ConstructionNotImplemented(
                f"the calibration recorded construction "
                f"{self.calibration.e_process_construction_id!r} but the campaign is running "
                f"{self.construction.construction_id!r}; the thresholds were validated for "
                "the recorded construction and are not transferable")
        if self.stopping_rule.max_blocks_per_candidate < self.calibration.b_min_blocks:
            raise StoppingRuleViolation(
                "the stopping rule's ceiling is below the calibrated B_min")
        _finite_floats(self.aa_effect_pool, "aa_effect_pool")
        _finite_floats(self.anchor_calibration_values, "anchor_calibration_values")
        if not self.aa_effect_pool:
            raise MaterialError(
                "aa_effect_pool is empty: the calibration block's raw A/A samples must be "
                "retained, because the MDE is computed from the calibrated dispersion and "
                "must be published WITH the estimate, never afterwards")
        if self.owning_rep_rule is not None:
            if not isinstance(self.owning_rep_rule, OwningProtocolRepRule):
                raise MaterialError("owning_rep_rule must be an OwningProtocolRepRule or None")
            if self.owning_rep_rule.kind == REP_RULE_FIXED \
                    and self.calibration.b_min_blocks != self.owning_rep_rule.blocks:
                raise CalibrationFailed(
                    f"{self.owning_rep_rule.protocol_id} fixes the count at exactly "
                    f"{self.owning_rep_rule.blocks} blocks, but the calibration carries "
                    f"B_min={self.calibration.b_min_blocks}; a fixed count is not a floor to "
                    "be raised")

    @property
    def b_min(self) -> int:
        return self.calibration.b_min_blocks

    def threshold_for(self, stratum: str) -> float:
        return self.calibration.threshold_for(stratum)

    def order_schedule(self, candidate_id: str, *, attempt: int = 0) -> OrderSchedule:
        return OrderSchedule.derive(campaign_seed=self.campaign_seed,
                                    candidate_id=candidate_id,
                                    base_blocks=self.b_min, attempt=attempt)

    def sequential_evaluation(self, *, candidate_id: str, stratum: str,
                              metric_direction: str,
                              attempt: int = 0) -> SequentialEvaluation:
        return SequentialEvaluation(
            rule=self.stopping_rule, commitment=self.stopping_rule_commitment,
            construction=self.construction, b_min=self.b_min,
            threshold=self.threshold_for(stratum), hypothesis=self.hypothesis,
            margin=self.margin, metric_direction=metric_direction,
            effect_scale=self.effect_scale,
            order_schedule=self.order_schedule(candidate_id, attempt=attempt))

    def aa_oriented(self, metric_direction: str) -> tuple:
        return tuple(orient(e, metric_direction) for e in self.aa_effect_pool)

    def to_dict(self) -> dict:
        return {"campaign_id": self.campaign_id, "effect_scale": self.effect_scale,
                "hypothesis": self.hypothesis, "margin": float(self.margin),
                "stopping_rule": self.stopping_rule.to_dict(),
                "stopping_rule_commitment": self.stopping_rule_commitment.to_dict(),
                "split_rule": self.split_rule.to_dict(),
                "construction": self.construction.to_dict(),
                "construction_content_hash": self.construction.content_hash(),
                "calibration": self.calibration.to_dict(),
                "owning_rep_rule": (None if self.owning_rep_rule is None
                                    else self.owning_rep_rule.to_dict())}


@dataclass(frozen=True)
class BlockReduction:
    """Everything one reduction produced, admissible or not.

    `window_checks` maps directly onto the `api.WindowAttestations` fields this
    reduction is authoritative for, so an inadmissible reduction still produces
    the evidence that makes the record `INVALID` *with its reason* rather than
    silently absent.
    """

    candidate_id: str
    stratum: str
    metric: str
    metric_direction: str
    effect_scale: str
    blocks: tuple
    block_effects: tuple
    oriented_effects: tuple
    median_effect: float
    mad_effect: float
    e_process: EProcessRun
    mde: MinimumDetectableEffect
    noise_floor: float
    threshold: float
    checks: tuple
    admissible: schemas.Check
    raw_samples_ref: str
    lcb: DescriptiveLCB
    estimate: Optional[api.EffectEstimate]

    def check(self, name: str) -> schemas.Check:
        for declared, chk in self.checks:
            if declared == name:
                return chk
        raise KeyError(f"no check named {name!r}; checks are "
                       f"{[n for n, _ in self.checks]}")

    @property
    def window_checks(self) -> dict:
        """The `api.WindowAttestations` fields this reduction is authoritative for."""
        return {
            "strata": self.check("stratum_partition"),
            "rule_immutability": self.check("stopping_rule_unmodified"),
            "order_randomized": self.check("order_control"),
            "calibration": self.check("calibration_cell"),
        }

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id, "stratum": self.stratum,
            "metric": self.metric, "metric_direction": self.metric_direction,
            "effect_scale": self.effect_scale, "blocks": len(self.blocks),
            "block_effects": list(self.block_effects),
            "median_effect": self.median_effect, "mad_effect": self.mad_effect,
            "e_process": self.e_process.to_dict(), "mde": self.mde.to_dict(),
            "noise_floor": self.noise_floor, "threshold": self.threshold,
            "checks": [[name, {"outcome": chk.outcome, "reasons": list(chk.reasons)}]
                       for name, chk in self.checks],
            "admissible": {"outcome": self.admissible.outcome,
                           "reasons": list(self.admissible.reasons)},
            "raw_samples_ref": self.raw_samples_ref, "lcb_descriptive": self.lcb.to_dict(),
            "estimate": None if self.estimate is None else self.estimate.to_dict(),
            "module": STATISTICS_MODULE_ID,
        }


class PairedBlockReducer:
    """`api.EffectReducer` — turns paired blocks into a conforming `EffectEstimate`.

    Every quantity the protocol requires published WITH the estimate is computed
    here and put in the same object: the e-value, its calibrated threshold, the
    MDE at the realized block count, and the calibrated noise floor. None of them
    can be attached afterwards, because `api.EffectEstimate` requires all of them
    at construction.

    The MDE is a function of the CALIBRATION material and the block count only.
    The candidate's own blocks are not an input to it. That is what makes
    *"computed … and published WITH the result, not after seeing it"* a checkable
    property rather than a promise.
    """

    def __init__(self, campaign: CampaignStatistics) -> None:
        if not isinstance(campaign, CampaignStatistics):
            raise MaterialError("PairedBlockReducer takes a CampaignStatistics")
        self._campaign = campaign
        self._mde_cache: dict = {}

    @property
    def construction_id(self) -> str:
        return self._campaign.construction.construction_id

    @property
    def campaign(self) -> CampaignStatistics:
        return self._campaign

    def mde_for(self, block_count: int, *, stratum: str,
                metric_direction: str) -> MinimumDetectableEffect:
        """The MDE at `block_count`, from the calibrated dispersion. Memoized."""
        key = (block_count, stratum, metric_direction)
        cached = self._mde_cache.get(key)
        if cached is not None:
            return cached
        campaign = self._campaign
        mde = solve_mde(
            campaign.aa_oriented(metric_direction), block_count=block_count,
            rule=campaign.stopping_rule, construction=campaign.construction,
            hypothesis=campaign.hypothesis, margin=campaign.margin,
            threshold=campaign.threshold_for(stratum), campaign_seed=campaign.campaign_seed)
        self._mde_cache[key] = mde
        return mde

    def reduce(self, request: api.EvaluationRequest, blocks: Sequence[PairedBlock], *,
               raw_samples_ref: Optional[str] = None,
               attempt: int = 0) -> BlockReduction:
        """Reduce, checking every condition.

        A non-conforming RUN comes back as a `BlockReduction` whose `admissible`
        is FAIL and whose `estimate` is `None` — it is never raised, because a
        voided run is journaled with its reason and never silently discarded.

        Unusable MATERIAL still raises, because there is nothing to reduce: zero
        blocks, or blocks whose anchor arm medians to zero under a relative
        scale. Those refusals carry `.checks` — the same `(name, Check)` pairs a
        `BlockReduction` would have carried — so the run remains journalable as
        INVALID with its reason rather than surfacing as a bare traceback.
        """
        if not isinstance(request, api.EvaluationRequest):
            raise MaterialError("reduce() takes an api.EvaluationRequest")
        blocks = tuple(blocks)
        if not blocks:
            raise _with_checks(InsufficientMaterial(
                "reduce() over zero blocks: a rate comparison with no paired blocks is not a "
                "measurement. Pass no reducer at all for a record that is not a rate "
                "comparison."), [("block_count", schemas.Check(
                    schemas.FAIL, ("no paired blocks were submitted",)))])
        for b in blocks:
            if not isinstance(b, PairedBlock):
                raise MaterialError("every block must be a PairedBlock")
        campaign = self._campaign
        checks: list = []

        cal = campaign.calibration
        cell_reasons = []
        if (cal.backend, cal.phase, cal.cell_class) != (request.backend, request.phase,
                                                        request.cell_class):
            cell_reasons.append(
                f"calibration was solved for ({cal.backend}, {cal.phase}, {cal.cell_class}) "
                f"but the cell is ({request.backend}, {request.phase}, "
                f"{request.cell_class}); values calibrated under a different host state, "
                "backend, phase or cell class MUST NOT be reused")
        checks.append(("calibration_cell",
                       schemas.Check(schemas.FAIL, tuple(cell_reasons)) if cell_reasons
                       else schemas.Check(schemas.PASS)))

        checks.append(("stopping_rule_unmodified",
                       campaign.stopping_rule_commitment.verify(campaign.stopping_rule)))
        checks.append(("stratum_partition", campaign.split_rule.check_blocks(blocks)))

        schedule = campaign.order_schedule(request.candidate_id, attempt=attempt)
        checks.append(("order_control", schedule.check_observed(blocks)))

        stratum = blocks[0].stratum
        b_min = campaign.b_min
        if len(blocks) < b_min:
            count_check = schemas.Check(schemas.FAIL, (
                f"{len(blocks)} paired blocks is below the calibrated B_min={b_min}; "
                "search-grade requires B_min paired blocks under order-randomized "
                "interleaving",))
        elif len(blocks) > campaign.stopping_rule.max_blocks_per_candidate:
            count_check = schemas.Check(schemas.FAIL, (
                f"{len(blocks)} paired blocks exceeds the declared ceiling "
                f"max_blocks_per_candidate="
                f"{campaign.stopping_rule.max_blocks_per_candidate}; extension follows the "
                "declared rule only",))
        else:
            count_check = schemas.Check(schemas.PASS)
        checks.append(("block_count", count_check))

        checks.append(("extension_structure", _check_extension_structure(
            blocks, b_min=b_min, rule=campaign.stopping_rule)))

        checks.append(("raw_samples_present", _check_raw_samples(blocks)))
        checks.append(("block_identity", _check_block_identity(blocks)))

        try:
            effects = tuple(block_effect(b, scale=campaign.effect_scale) for b in blocks)
        except EffectScaleError as exc:
            checks.append(("effect_scale", schemas.Check(schemas.FAIL, (str(exc),))))
            effects = ()
        else:
            checks.append(("effect_scale", schemas.Check(schemas.PASS)))

        if not effects:
            raise _with_checks(MaterialError(
                "the blocks could not be reduced to effects at the campaign's declared "
                "scale; " + "; ".join(
                    r for name, chk in checks if name == "effect_scale"
                    for r in chk.reasons)), checks)

        oriented = tuple(orient(e, request.metric_direction) for e in effects)
        threshold = campaign.threshold_for(stratum)
        e_run = run_e_process(oriented, construction=campaign.construction,
                              hypothesis=campaign.hypothesis, margin=campaign.margin,
                              threshold=threshold)
        try:
            mde = self.mde_for(len(blocks), stratum=stratum,
                               metric_direction=request.metric_direction)
        except (StoppingRuleViolation, InsufficientMaterial) as exc:
            # An over-extended run is exactly the run that must be JOURNALED as
            # INVALID with its reason, so a block count the rule cannot license
            # is a not-found MDE on the record, never a traceback out of the
            # reducer. `found=False` forces the FAIL below and blocks the
            # estimate, so `value` is never read into an `api.EffectEstimate`.
            mde = MinimumDetectableEffect(
                value=0.0, block_count=len(blocks), window_length=0,
                power_target=campaign.construction.mde_power_target,
                achieved_power=0.0, resamples=campaign.construction.mde_resamples,
                search_tolerance=campaign.construction.mde_search_tolerance,
                bracket_low=0.0, bracket_high=0.0,
                method="common_random_number_resampling",
                construction_id=campaign.construction.construction_id,
                found=False, reason=str(exc))
        checks.append(("mde_derivable", schemas.Check(schemas.PASS) if mde.found
                       else schemas.Check(schemas.FAIL, (
                           f"no MDE could be derived at {len(blocks)} blocks: {mde.reason}. "
                           "A record without a published MDE is INVALID.",))))

        raw_list = [b.to_list() for b in blocks]
        ref = raw_samples_ref or f"sha256:{schemas.content_hash(raw_list)}"
        lcb = descriptive_lcb(effects, campaign_seed=campaign.campaign_seed,
                              candidate_id=request.candidate_id,
                              construction=campaign.construction)

        admissible = _combine_checks([chk for _n, chk in checks])
        estimate = None
        if admissible.outcome == schemas.PASS:
            try:
                estimate = api.EffectEstimate(
                    metric=request.metric, metric_direction=request.metric_direction,
                    value=median(effects), e_value=e_run.e_running_max, threshold=threshold,
                    mde=mde.value, noise_floor=cal.noise_floor_phi,
                    paired_blocks=len(blocks), stratum=stratum,
                    raw_samples=tuple(b.to_tuple() for b in blocks), raw_samples_ref=ref,
                    lcb_descriptive=lcb.value)
            except EValueNotRepresentable:
                raise
            except (ValueError, TypeError) as exc:
                checks.append(("estimate_wellformed",
                               schemas.Check(schemas.FAIL, (str(exc),))))
                admissible = schemas.Check(schemas.FAIL, (str(exc),))

        return BlockReduction(
            candidate_id=request.candidate_id, stratum=stratum, metric=request.metric,
            metric_direction=request.metric_direction, effect_scale=campaign.effect_scale,
            blocks=blocks, block_effects=effects, oriented_effects=oriented,
            median_effect=median(effects), mad_effect=mad(effects), e_process=e_run,
            mde=mde, noise_floor=cal.noise_floor_phi, threshold=threshold,
            checks=tuple(checks), admissible=admissible, raw_samples_ref=ref, lcb=lcb,
            estimate=estimate)

    def reduce_blocks(self, request: api.EvaluationRequest,
                      blocks: Sequence[Any]) -> Optional[api.EffectEstimate]:
        """The `api.EffectReducer` seam. Raises `ReductionInadmissible` on refusal.

        It deliberately does NOT return `None` for a non-conforming run:
        `api.TierDispatcher` reads `effect is None` as "not a rate comparison"
        and then skips the rate-only void conditions, so a `None` here would
        suppress the very void a strata or order-control violation must raise.
        The full reduction rides on the exception for journaling.
        """
        reduction = self.reduce(request, blocks)
        if reduction.estimate is None:
            failing = [f"{name}: {chk.outcome} {list(chk.reasons)}"
                       for name, chk in reduction.checks if chk.outcome != schemas.PASS]
            raise ReductionInadmissible(
                "the reduction is not search-grade and MUST NOT be reported as a rate "
                "comparison: " + "; ".join(failing), reduction)
        return reduction.estimate


def _check_raw_samples(blocks: Sequence[PairedBlock]) -> schemas.Check:
    """*"a record whose reduction cannot be recomputed from its raw samples is INVALID."*"""
    reasons = []
    for b in blocks:
        if not b.anchor_samples or not b.candidate_samples:
            reasons.append(f"block {b.block_index} carries no raw samples for one arm")
    return schemas.Check(schemas.FAIL, tuple(reasons)) if reasons \
        else schemas.Check(schemas.PASS)


def _check_block_identity(blocks: Sequence[PairedBlock]) -> schemas.Check:
    """Every submitted block is a DISTINCT block, at the position it claims.

    `SequentialEvaluation` issues block *i* and refuses anything whose
    `block_index` is not *i*, so a run driven through the rule always arrives
    here as 0..n-1. The `api.EffectReducer` seam has no such history, and without
    this check a caller could hand the same measured block to the reducer B_min
    times: the sign statistic is then IDENTICAL in every "block", the wealth
    grows at the betting cap in every step, and one measurement is reported as
    `blocks=B_min` paired blocks under order-randomized interleaving. Order
    control cannot catch it — every position's order can be made to match the
    schedule — so the count of independent blocks has to be checked as identity,
    not as `len()`.
    """
    reasons = []
    seen: dict = {}
    for position, b in enumerate(blocks):
        if b.block_index != position:
            reasons.append(
                f"the block at position {position} carries block_index {b.block_index}; "
                "the reduction's order control, its extension structure and its raw-sample "
                "reference are all positional, so a block must sit at the index it claims")
        first = seen.get(b.block_index)
        if first is not None:
            reasons.append(
                f"block_index {b.block_index} appears at positions {first} and {position}; "
                "B_min PAIRED BLOCKS means B_min distinct blocks, and a replayed block "
                "contributes its evidence again without having been measured again")
        else:
            seen[b.block_index] = position
    return schemas.Check(schemas.FAIL, tuple(reasons)) if reasons \
        else schemas.Check(schemas.PASS)


def _check_extension_structure(blocks: Sequence[PairedBlock], *, b_min: int,
                               rule: StoppingRule) -> schemas.Check:
    """Base blocks first, exactly `b_min` of them, then whole declared rounds."""
    reasons = []
    base = [b for b in blocks if b.segment == SEGMENT_BASE]
    ext = [b for b in blocks if b.segment == SEGMENT_EXTENSION]
    # The base segment is the FIRST min(len(blocks), B_min) blocks, always. The
    # guard used to read `len(blocks) > b_min`, which skipped the count entirely
    # for a submission of exactly B_min blocks — so B_min blocks carrying ZERO
    # base blocks, all labelled "extension round 1", passed. An extension round
    # that arrives before the base segment it extends is not an extension.
    expected_base = min(len(blocks), b_min)
    if len(base) != expected_base:
        reasons.append(
            f"{len(base)} base blocks with {len(blocks)} submitted; the base segment is "
            f"the first {expected_base} (B_min={b_min}) blocks and everything beyond it is "
            "a declared extension round")
    if base and blocks[:len(base)] != tuple(base):
        reasons.append("extension blocks are interleaved with base blocks; the extension is "
                       "a bounded number of FRESH reversed-order pairs after the base")
    rounds = sorted({b.extension_round for b in ext}) if ext else []
    if rounds:
        if rounds != list(range(1, len(rounds) + 1)):
            reasons.append(f"extension rounds {rounds} are not consecutive from 1")
        if len(rounds) > rule.extension.max_rounds:
            reasons.append(
                f"{len(rounds)} extension rounds exceeds the declared maximum "
                f"{rule.extension.max_rounds}; extension follows the declared rule only")
        for r in rounds:
            n = sum(1 for b in ext if b.extension_round == r)
            if n != rule.extension.blocks_per_round:
                reasons.append(
                    f"extension round {r} has {n} blocks, not the declared "
                    f"{rule.extension.blocks_per_round}")
    return schemas.Check(schemas.FAIL, tuple(reasons)) if reasons \
        else schemas.Check(schemas.PASS)


def _with_checks(error: StatisticsError, checks: Sequence) -> StatisticsError:
    """Attach the `(name, Check)` pairs computed so far to a material refusal.

    A refusal that carries no checks cannot be journaled as INVALID *with its
    reason*, and *"a voided run is journaled as INVALID with its reason, and is
    never silently discarded"*. `ReductionInadmissible` carries a whole
    `BlockReduction`; these refusals fire before one can be built (there are no
    effects, so there is no e-process and no MDE), so they carry the checks
    instead. Callers read `getattr(exc, "checks", ())`.
    """
    error.checks = tuple(checks)
    return error


def _combine_checks(checks: Sequence[schemas.Check]) -> schemas.Check:
    """FAIL dominates COULD_NOT_CHECK dominates PASS. Fail closed, but stay distinct."""
    reasons: list = []
    outcome = schemas.PASS
    for chk in checks:
        if chk.outcome == schemas.PASS:
            continue
        reasons.extend(chk.reasons)
        if chk.outcome == schemas.FAIL:
            outcome = schemas.FAIL
        elif outcome != schemas.FAIL:
            outcome = schemas.COULD_NOT_CHECK
    return schemas.Check(outcome, tuple(reasons))


def verify_reduction_reproducible(estimate: api.EffectEstimate,
                                  reducer: PairedBlockReducer,
                                  request: api.EvaluationRequest) -> schemas.Check:
    """*"raw samples from which the reduction is reproducible."*

    Rebuilds the blocks from the estimate's OWN `raw_samples` — not from the
    caller's block list — and recomputes the reduction. If the record's raw
    samples do not reproduce the record's numbers, the record is INVALID, and
    that has to be checkable from the record alone.

    Two independent checks, because either alone leaves a hole:

      1. the rebuilt samples must hash to the content-addressed
         `raw_samples_ref` the record carries; and
      2. the recomputed reduction must equal the recorded one, field by field.

    Check 2 alone would not catch every edit, and saying so matters: the
    reduction is deliberately ROBUST — a median over blocks and a sign-based
    e-process — so changing one sample in the upper half of the order statistics
    can leave every reported number identical. That is a virtue of the reducer
    and a hole in any tamper check built only on its outputs, which is what
    check 1 closes.
    """
    if not isinstance(estimate, api.EffectEstimate):
        return schemas.Check(schemas.COULD_NOT_CHECK,
                             ("no EffectEstimate was supplied",))
    if not isinstance(reducer, PairedBlockReducer):
        return schemas.Check(schemas.COULD_NOT_CHECK,
                             ("no PairedBlockReducer was supplied",))
    try:
        blocks = tuple(_block_from_tuple(raw) for raw in estimate.raw_samples)
    except (MaterialError, TypeError, ValueError) as exc:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            f"the record's raw samples could not be rebuilt into paired blocks: {exc}",))
    ref = estimate.raw_samples_ref
    if not ref.startswith("sha256:"):
        # Not a fail-open skip. Check 2 alone cannot detect an edited sample —
        # the docstring above says why — so a record whose `raw=` is not content
        # addressed is one whose samples CANNOT be shown to be the samples the
        # reduction was taken over. That is the third outcome, never a PASS.
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            f"the record carries raw={ref!r}, which is not a content-addressed "
            "'sha256:<hex>' reference, so the rebuilt samples cannot be checked against "
            "it. The reducer is deliberately ROBUST — editing one sample can leave every "
            "reported number identical — so recomputing the reduction is not by itself "
            "evidence that these are the samples it was taken over.",))
    actual = schemas.content_hash([b.to_list() for b in blocks])
    if actual != ref[len("sha256:"):]:
        return schemas.Check(schemas.FAIL, (
            f"the record's raw samples hash to {actual[:12]} but the record carries "
            f"raw={ref}; the samples are not the samples the reduction was taken over",))
    try:
        redone = reducer.reduce(request, blocks, raw_samples_ref=estimate.raw_samples_ref)
    except StatisticsError as exc:
        return schemas.Check(schemas.FAIL, (
            f"the reduction could not be recomputed from the record's raw samples: {exc}",))
    if redone.estimate is None:
        return schemas.Check(schemas.FAIL, (
            "recomputing from the record's raw samples did not produce an estimate: "
            + "; ".join(redone.admissible.reasons),))
    diffs = []
    for name in ("value", "e_value", "threshold", "mde", "noise_floor", "paired_blocks",
                 "stratum", "metric", "metric_direction"):
        was = getattr(estimate, name)
        now = getattr(redone.estimate, name)
        if was != now:
            diffs.append(f"{name}: recorded {was!r}, recomputed {now!r}")
    if diffs:
        return schemas.Check(schemas.FAIL, tuple(diffs))
    return schemas.Check(schemas.PASS, (
        f"the reduction recomputes exactly from {len(blocks)} raw paired blocks",))


def _block_from_tuple(raw: Any) -> PairedBlock:
    if not isinstance(raw, tuple) or len(raw) != 9:
        raise MaterialError(f"raw sample is not a PairedBlock tuple: {raw!r}")
    (index, unit_id, stratum, order, segment, extension_round, measured_at,
     anchor_samples, candidate_samples) = raw
    return PairedBlock(block_index=index, unit_id=unit_id, stratum=stratum, order=order,
                       segment=segment, extension_round=extension_round,
                       measured_at=measured_at, anchor_samples=tuple(anchor_samples),
                       candidate_samples=tuple(candidate_samples))
