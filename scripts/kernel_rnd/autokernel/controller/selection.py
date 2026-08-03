"""selection.py — target selection and proposal ranking (design §8.3, §8.4, §8.4.1).

WHY THIS MODULE EXISTS
----------------------
`state_machine` owns WHERE the loop is. This module owns WHAT the loop spends its
next window on, and it is the surface where a loop with full compute wastes it.
Five failures it is written against, each with a receipt in the owning design:

1. **A cheaper layer skipped on a hunch** (§8.3). The hierarchy — placement and
   launch config, dispatcher, autotuning, layout/repack, operator fusion, work
   scheduling, new kernel, scheduler architecture, alternate engine — is ordered
   by cost. *"A cheaper layer may be skipped only with an evidence receipt showing
   why it cannot explain the measured gap."* Here the receipt is ARITHMETIC, not
   prose: it names the measured gap, the measured ceiling of the layer it skips,
   and the events both rest on, and `check_layer_skip` refuses unless the ceiling
   is genuinely too small to explain the gap. A receipt bound to a commit that is
   no longer the anchor does not resolve, and does not license the skip (§19.3).
2. **A filtered proposal thrown away** (§8.4). AutoPilot dispatched 119 identical
   invalid actions whose rejection message named the exact fix, and none of it
   ever reached the planner. Every rejection here is journaled as
   `PROPOSAL_SKIPPED` with its reason codes and a fingerprint, is read back as
   next-round feedback, auto-blacklists on its SECOND occurrence, and trips
   `PLANNER_DEGRADED` on a declared run. The fingerprint is computed over
   STRUCTURAL facets only — never prose — because a fingerprint that includes the
   planner's own wording is one a reworder defeats on the next attempt.
3. **Metered drafting paid for before the cheap check** (§8.4). The reverse
   ordering cost roughly 38 draft-and-critique cycles that were paid for and then
   thrown away. `MeteredDraftGuard` is the only sanctioned way to reach a drafter:
   it demands a `PrescreenTicket` that only the deterministic screen issues, and
   it refuses a drafted proposal whose mechanism fingerprint diverged from the one
   that was screened — so screening a cheap idea and drafting an expensive one is
   not a route either.
4. **Deep work starved by arithmetic** (§8.4.1, AK-D31). EIG-first ranking
   systematically loses to low-variance incremental work, so architectural
   proposals compete only with each other out of a reserved arm, and an
   incremental proposal may never draw from that arm — not even when the general
   arm is empty.
5. **A dead region confused with a broken planner** (§8.4.1, §8.10). Falling yield
   WITH rising `PROPOSAL_SKIPPED` is `PLANNER_DEGRADED`, not EXPLORE. The decay
   floor, the trailing window and the minimum dwell are DERIVED from the harvest's
   own yield samples and re-derived on every read, so a supplied literal cannot
   survive construction — *"a supplied number here would decide the explore/exploit
   tradeoff by guess"*.

AUTHORITY
---------
The planner proposes; this module DISPOSES. Every rejection condition, the
hierarchy gate, the phase switch and the arm accounting are computed by
deterministic code from journaled records and measured receipts. The planner's
`expected_information_gain` is a RANKING input and never a gate: no value of it
admits a proposal, because admission is decided before ranking runs. The
planner's expected performance value is not consulted at all — value is recomputed
from the measured wall-share receipt or, in an architectural campaign, from the
predicted post-change profile against the measured one.

WHAT THIS MODULE IS NOT
-----------------------
It runs no inference, no benchmark and no build; it starts, stops and signals no
process; it calls no model — `MeteredDraftGuard` invokes a caller-supplied
drafter and is tested against a fake one. It writes nothing except journal
appends through `journal.Journal`.

Governing instrument: `measurement/protocols/kernel-research.md` (P-AK-SEARCH-1,
RATIFIED 2026-08-03). Two of its clauses bind here directly: *"a proposal that
targets a confirmation shape is rejected before it consumes a window"* and *"The
confirmation stratum's contents MUST NOT appear in planner context"* — which is
why `SelectionContext` holds confirmation shapes only as DIGESTS. Denial 6 (no
self-amendment) is why a proposal whose evaluation plan names a step the pinned
evaluator bundle does not implement is rejected rather than accommodated.

WHAT THIS MODULE DELIBERATELY DOES NOT DECIDE
---------------------------------------------
* **Affected-surface scope.** §6.4/invariant 18: the actor's declaration is a
  scored prediction, never a scope input. The correctness-oracle condition below
  is computed over the DECLARED target and is a pre-mutation necessary condition
  only; the derived-vs-traced reconciliation at T0 remains the authority, and a
  proposal admitted here can still fail there. `ScreenResult.oracle_coverage_basis`
  records which basis was used so the record cannot be misread as a scope verdict.
* **Whether a candidate banks.** §9.6 and the evaluator own that.
* **Stop disposition.** This module builds `state_machine.StopRequest`s; the
  machine validates the evidence and owns the transition.

Owning design: `epyc-root/handoffs/active/autokernel-research-loop.md` §8.3,
§8.3.1, §8.4, §8.4.0, §8.4.1, §9.5, §12, §19.2, §19.3, AK-D31..AK-D38.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Optional, Sequence

from .. import journal, schemas
from ..evaluator import statistics as ev_statistics
from . import fingerprint, planner, state_machine

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")      # mirrors schemas._COMMIT_RE

__all__ = [
    # errors
    "SelectionError", "DraftingRefused", "DraftedProposalDiverged",
    "CalibrationTampered", "LedgerEntryInadmissible", "InsufficientYieldMaterial",
    "SkipNotRecorded",
    # §8.3 hierarchy
    "HIERARCHY", "HIERARCHY_RANK", "LayerSkipReceipt", "check_layer_skip",
    # §19.2/§19.3 do-not-repeat ledger
    "LEDGER_CLASSES", "REJECTING_LEDGER_CLASSES", "RECEIPT_REQUIRED_CLASSES",
    "LEDGER_DIMENSIONS", "LedgerEntry", "LedgerMatch", "match_ledger",
    # fingerprinting
    "SELECTION_BLOCK_KEY", "mechanism_facets", "mechanism_fingerprint",
    "proposal_fingerprint",
    # §8.4 rejection vocabulary
    "REJECTION_CODES", "Rejection", "ScreenResult",
    "REJECT_SCHEMA_INVALID", "REJECT_CAMPAIGN_MISMATCH", "REJECT_MECHANISM_UNNAMED",
    "REJECT_REGIME_IDENTITY_INCOMPLETE", "REJECT_FINGERPRINT_BLACKLISTED",
    "REJECT_WALL_SHARE_CEILING", "REJECT_NO_CORRECTNESS_ORACLE",
    "REJECT_SHAPES_NOT_IN_REAL_GRAPH", "REJECT_REPEATS_RECEIPTED_NEGATIVE",
    "REJECT_BUDGET_EXCEEDED", "REJECT_CROSSES_UNOWNED_DOMAIN",
    "REJECT_REQUIRES_EVALUATOR_CHANGE", "REJECT_MULTIPLE_CONCEPTUAL_CHANGES",
    "REJECT_HIERARCHY_SKIP_UNRECEIPTED", "REJECT_FORBIDDEN_OBJECTIVE",
    "REJECT_TARGETS_CONFIRMATION_SHAPE", "REJECT_ARCHITECTURAL_ESCAPE_UNDECLARED",
    "REJECT_LINEAGE_STEP_OUT_OF_ORDER", "REJECT_SPIKE_MALFORMED",
    "REJECT_EIG_OUT_OF_RANGE", "REJECT_GAIN_UNDECLARED", "REJECT_UNVERIFIABLE",
    # §8.4.1 architectural campaigns and spikes
    "LineageStep", "ArchitecturalCampaign", "SpikeDeclaration",
    "spike_cost_regression",
    # context and screening
    "BUDGET_KEYS", "LANE_BUDGET_KEY", "BUDGET_CAP_BY_KEY",
    "BUDGET_KEYS_NOT_SCREENED", "SECONDS_PER_MINUTE", "budget_remaining_from_caps",
    "SelectionContext", "screen_proposal",
    "ProposalScreener",
    # prescreen / metered drafting
    "DraftBrief", "PrescreenTicket", "PrescreenOutcome", "prescreen",
    "MeteredDraftGuard",
    # skip memory and planner health
    "SkipRecord", "SkipHistory", "SkipFeedback", "read_skip_history",
    "planner_health_stop_request",
    # §8.4.1 phases
    "PHASE_HARVEST", "PHASE_EXPLORE", "PHASES", "YieldObservation",
    "YieldCalibration", "derive_yield_calibration", "PhaseDecision", "decide_phase",
    # ranking and arms
    "ARM_INCREMENTAL", "ARM_ARCHITECTURAL", "ArmBudget", "partition_budget",
    "RankedProposal", "rank_proposals", "SelectionDecision", "select_next",
]


# =============================================================================
# Errors — every one is a refusal, never a degraded result
# =============================================================================

class SelectionError(Exception):
    """Base for every refusal this module raises."""


class DraftingRefused(SelectionError):
    """Metered drafting was attempted without a valid prescreen ticket.

    §8.4: *"Cheap deterministic checks run before metered drafting, not after."*
    """


class DraftedProposalDiverged(SelectionError):
    """A drafter returned a proposal whose mechanism is not the one screened.

    Screening a cheap mechanism and drafting an expensive one would make the
    prescreen a formality; the ticket binds the fingerprint, so it cannot.
    """


class CalibrationTampered(SelectionError):
    """A `YieldCalibration` disagrees with its own derivation samples.

    §8.4.1: the decay floor and window are *"derived by the campaign calibration
    procedure, never supplied"*. A supplied literal is refused by recomputation,
    not by a comment asking callers not to supply one.
    """


class LedgerEntryInadmissible(SelectionError):
    """A do-not-repeat entry that §19.3 will not admit (missing receipt, missing
    regime identity, missing reopen predicate). A suppression that closes a
    research family carries a HIGHER bar than the win it blocks."""


class InsufficientYieldMaterial(SelectionError):
    """Too little harvest material to derive a floor and a window from."""


class SkipNotRecorded(SelectionError):
    """A rejection could not be journaled. §8.4 forbids a bare discard, so a
    screen whose `PROPOSAL_SKIPPED` append failed is a failure, not a silent
    rejection."""


# =============================================================================
# §8.3 — the selection hierarchy, cheapest first
# =============================================================================

#: §8.3, in order. The ORDER is the whole content: rank is cost, and a skip
#: forward is what needs a receipt.
HIERARCHY = (
    "placement_and_launch_config",
    "dispatcher",
    "autotuning",
    "layout_repack",
    "operator_fusion",
    "work_scheduling",
    "new_kernel",
    "scheduler_architecture",
    "alternate_engine",
)

HIERARCHY_RANK: Mapping[str, int] = {name: i for i, name in enumerate(HIERARCHY)}


@dataclass(frozen=True)
class LayerSkipReceipt:
    """Evidence that one cheaper layer cannot explain the measured gap (§8.3).

    The load-bearing field is `layer_ceiling`: the MEASURED most that this layer
    could contribute. `basis` is prose and is recorded, but the check is the
    comparison `layer_ceiling < measured_gap` — an argument a profile can refute.
    A receipt whose ceiling was never measured carries `layer_ceiling=None` and
    yields COULD_NOT_CHECK: not knowing is a third outcome, and it does not
    license the skip either.
    """

    layer: str
    measured_gap: float
    layer_ceiling: Optional[float]
    gap_receipt_id: str
    evidence_event_ids: tuple
    anchor_commit: str
    basis: str

    def __post_init__(self) -> None:
        if self.layer not in HIERARCHY_RANK:
            raise ValueError(f"layer: {self.layer!r} is not in the §8.3 hierarchy")
        if not isinstance(self.measured_gap, (int, float)) or isinstance(self.measured_gap, bool):
            raise TypeError("measured_gap must be a number")
        if not math.isfinite(float(self.measured_gap)) or float(self.measured_gap) <= 0.0:
            raise ValueError(
                "measured_gap must be finite and strictly positive; a receipt "
                "against a gap of zero explains nothing"
            )
        if self.layer_ceiling is not None:
            if not isinstance(self.layer_ceiling, (int, float)) or isinstance(
                self.layer_ceiling, bool
            ):
                raise TypeError("layer_ceiling must be a number or None")
            if not math.isfinite(float(self.layer_ceiling)) or float(self.layer_ceiling) < 0.0:
                raise ValueError("layer_ceiling must be finite and non-negative")
        for name in ("gap_receipt_id", "basis"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name}: required and non-empty")
        if not isinstance(self.evidence_event_ids, tuple) or not self.evidence_event_ids:
            raise ValueError(
                "evidence_event_ids: required, a non-empty tuple — §19.3 wants a "
                "receipt, not a confident sentence"
            )
        if not isinstance(self.anchor_commit, str) or not _COMMIT_RE.match(self.anchor_commit):
            raise ValueError("anchor_commit: required, a 40-char lowercase hex commit")

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "measured_gap": float(self.measured_gap),
            "layer_ceiling": None if self.layer_ceiling is None else float(self.layer_ceiling),
            "gap_receipt_id": self.gap_receipt_id,
            "evidence_event_ids": list(self.evidence_event_ids),
            "anchor_commit": self.anchor_commit,
            "basis": self.basis,
        }

    @staticmethod
    def from_dict(obj: Mapping[str, Any]) -> "LayerSkipReceipt":
        """Rebuild from the JSON form a proposal manifest carries.

        The manifest is a RECORD — it is journaled and must stay serialisable —
        so the receipt travels as a mapping and is validated on the way in, where
        a malformed one becomes a rejection rather than an exception three frames
        later.
        """
        if not isinstance(obj, Mapping):
            raise TypeError(f"receipt must be a mapping, got {type(obj).__name__}")
        events = obj.get("evidence_event_ids")
        if isinstance(events, (list, tuple)):
            events = tuple(events)
        return LayerSkipReceipt(
            layer=obj.get("layer"),
            measured_gap=obj.get("measured_gap"),
            layer_ceiling=obj.get("layer_ceiling"),
            gap_receipt_id=obj.get("gap_receipt_id"),
            evidence_event_ids=events,
            anchor_commit=obj.get("anchor_commit"),
            basis=obj.get("basis"),
        )


def check_layer_skip(
    target_layer: str,
    receipts: Sequence[LayerSkipReceipt],
    *,
    anchor_commit: str,
    known_event_ids: frozenset,
) -> schemas.Check:
    """§8.3's skip rule, enforced rather than documented.

    Every layer cheaper than `target_layer` needs its own receipt; the receipt
    must be bound to the CURRENT anchor commit (a receipt taken against a
    denominator that has since moved does not resolve — §19.3, AK-D22), its cited
    evidence must exist, and its measured ceiling must actually be too small to
    explain the measured gap.

    `known_event_ids` is required, not optional: a check that cannot resolve the
    evidence it was handed would be answering a different question.
    """
    if target_layer not in HIERARCHY_RANK:
        return schemas.Check(schemas.FAIL, (
            f"target_layer: {target_layer!r} is not in the §8.3 hierarchy {list(HIERARCHY)}",
        ))
    if not isinstance(known_event_ids, frozenset):
        raise TypeError("known_event_ids must be a frozenset of resolvable event ids")
    if not isinstance(anchor_commit, str) or not _COMMIT_RE.match(anchor_commit):
        raise ValueError("anchor_commit must be a 40-char lowercase hex commit")

    cheaper = [name for name in HIERARCHY if HIERARCHY_RANK[name] < HIERARCHY_RANK[target_layer]]
    if not cheaper:
        return schemas.Check(schemas.PASS)

    by_layer: dict = {}
    conflicting: set = set()
    for receipt in receipts:
        if not isinstance(receipt, LayerSkipReceipt):
            raise TypeError("receipts must all be LayerSkipReceipt")
        # A second receipt for a layer is not an update: last-wins would let a
        # ceiling that CAN explain the gap be overwritten by a convenient one in
        # the same submission, so a self-contradicting receipt set licenses
        # nothing.
        if receipt.layer in by_layer:
            conflicting.add(receipt.layer)
        by_layer[receipt.layer] = receipt

    failures: list = []
    unknowns: list = []
    for name in cheaper:
        receipt = by_layer.get(name)
        if receipt is None:
            failures.append(
                f"{name}: no evidence receipt; §8.3 permits skipping a cheaper layer "
                "ONLY with a receipt showing it cannot explain the measured gap"
            )
            continue
        if name in conflicting:
            failures.append(
                f"{name}: more than one receipt names this layer; a receipt set that "
                "contradicts itself does not license a skip, and taking the last one "
                "would let a submission overwrite its own failing ceiling"
            )
            continue
        if receipt.anchor_commit != anchor_commit:
            failures.append(
                f"{name}: receipt is bound to commit {receipt.anchor_commit[:12]} but the "
                f"campaign anchor is {anchor_commit[:12]}; a receipt whose binding no "
                "longer resolves does not license a skip (§19.3, AK-D22)"
            )
            continue
        missing = [e for e in receipt.evidence_event_ids if e not in known_event_ids]
        # The gap the arithmetic is against is evidence too. Without this, the
        # `gap_receipt_id` field is validated as a non-empty string and never
        # resolved, so both operands of `layer_ceiling < measured_gap` could be
        # written by the actor that wants the skip.
        if receipt.gap_receipt_id not in known_event_ids:
            missing.append(receipt.gap_receipt_id)
        if missing:
            failures.append(
                f"{name}: receipt cites event(s) {missing} that do not resolve in this "
                "journal; an unresolvable receipt is a confident sentence"
            )
            continue
        if receipt.layer_ceiling is None:
            unknowns.append(
                f"{name}: the layer's own ceiling was never measured, so whether it can "
                "explain the gap is unknown — inability to evaluate is not a licence"
            )
            continue
        if float(receipt.layer_ceiling) >= float(receipt.measured_gap):
            failures.append(
                f"{name}: measured ceiling {receipt.layer_ceiling} is not smaller than the "
                f"measured gap {receipt.measured_gap}, so this layer CAN explain the gap "
                "and must be tried before a more expensive one"
            )

    if failures:
        return schemas.Check(schemas.FAIL, tuple(failures + unknowns))
    if unknowns:
        return schemas.Check(schemas.COULD_NOT_CHECK, tuple(unknowns))
    return schemas.Check(schemas.PASS)


# =============================================================================
# §19.2 / §19.3 — the do-not-repeat and constraint ledger
# =============================================================================

LEDGER_CLASSES = (
    "HARD_CONSTRAINT", "MATCHED_NEGATIVE", "CONDITIONAL_NEGATIVE",
    "CONFOUNDED_RESULT", "SUPERSEDED_FACT", "LOW_VALUE",
)

#: Classes whose match REJECTS the proposal outright (§19.2 planner behaviour
#: column). `CONDITIONAL_NEGATIVE` excludes cells, `CONFOUNDED_RESULT` demands a
#: repaired experiment, `LOW_VALUE` deprioritizes — none of the three rejects.
REJECTING_LEDGER_CLASSES = frozenset({
    "HARD_CONSTRAINT", "MATCHED_NEGATIVE", "SUPERSEDED_FACT",
})

#: §19.3: every suppression that closes a family carries a source receipt, a
#: binding to the production commit it was verified against, and re-verification
#: on anchor move.
RECEIPT_REQUIRED_CLASSES = REJECTING_LEDGER_CLASSES

#: The regime dimensions a match may key on. §19.2: *"'Do not repeat' without
#: regime identity is dangerous because this project repeatedly observes sign
#: changes across architecture, substrate, batch, context, and quant."*
LEDGER_DIMENSIONS = frozenset({
    "backend", "phase", "regimes", "shapes", "models", "ops", "quant", "batch",
    "context", "change_class", "hierarchy_layer", "architecture", "substrate",
})


@dataclass(frozen=True)
class LedgerEntry:
    """One compiled do-not-repeat / constraint entry (§19.2).

    Construction refuses an entry §19.3 would not admit. That is deliberate: a
    machine-maintained suppression ledger is the most authoritative-looking
    artifact in the system and therefore the most dangerous one to leave
    unchecked, and a wrong suppression is invisible because nothing ever tests it
    again.
    """

    entry_id: str
    entry_class: str
    mechanism: str
    match_dimensions: Mapping[str, tuple]
    reopen_when: str
    receipt: Optional[str] = None
    anchor_commit: Optional[str] = None
    conflicted: bool = False

    def __post_init__(self) -> None:
        for name in ("entry_id", "mechanism", "reopen_when"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise LedgerEntryInadmissible(f"{name}: required and non-empty")
        if self.entry_class not in LEDGER_CLASSES:
            raise LedgerEntryInadmissible(
                f"entry_class: {self.entry_class!r} not in {list(LEDGER_CLASSES)}"
            )
        if not isinstance(self.match_dimensions, Mapping) or not self.match_dimensions:
            raise LedgerEntryInadmissible(
                "match_dimensions: required and non-empty — an entry without regime "
                "identity suppresses regimes it was never measured in (§19.2)"
            )
        for key, values in self.match_dimensions.items():
            if key not in LEDGER_DIMENSIONS:
                raise LedgerEntryInadmissible(
                    f"match_dimensions.{key}: not a declared dimension "
                    f"{sorted(LEDGER_DIMENSIONS)}"
                )
            if not isinstance(values, tuple) or not values:
                raise LedgerEntryInadmissible(
                    f"match_dimensions.{key}: required, a non-empty tuple of values"
                )
        if self.entry_class in RECEIPT_REQUIRED_CLASSES:
            if not isinstance(self.receipt, str) or not self.receipt.strip():
                raise LedgerEntryInadmissible(
                    f"receipt: required for {self.entry_class} — commit plus path plus "
                    "line, or an artifact hash, not a confident sentence (§19.3)"
                )
            if not isinstance(self.anchor_commit, str) or not _COMMIT_RE.match(
                self.anchor_commit
            ):
                raise LedgerEntryInadmissible(
                    f"anchor_commit: required for {self.entry_class} — §19.3 binds every "
                    "suppression to the production commit it was verified against"
                )

    def authoritative_against(self, anchor_commit: str) -> bool:
        """False when the entry is conflicted or its receipt binds elsewhere.

        §19.3: *"a suppression whose receipt no longer resolves reverts to
        `conflicted` rather than continuing to block."* A stale suppression that
        keeps blocking is how a research family closes silently.
        """
        if self.conflicted:
            return False
        if self.entry_class in RECEIPT_REQUIRED_CLASSES:
            return self.anchor_commit == anchor_commit
        return True


@dataclass(frozen=True)
class LedgerMatch:
    entry_id: str
    entry_class: str
    mechanism: str
    matched_dimensions: tuple
    rejects: bool
    reason: str


def match_ledger(
    facets: Mapping[str, Any],
    ledger: Sequence[LedgerEntry],
    *,
    anchor_commit: str,
    satisfied_reopen_predicates: frozenset,
) -> tuple:
    """Match a proposal's mechanism facets against the ledger.

    A `MATCHED_NEGATIVE` whose `reopen_when` predicate is newly satisfied does not
    reject (§19.2). Everything else that rejects, rejects — including an operator
    hypothesis, whose origin buys nothing (§8.4.0, AK-D38: *"being the operator's
    idea is not new evidence"*).
    """
    if not isinstance(satisfied_reopen_predicates, frozenset):
        raise TypeError("satisfied_reopen_predicates must be a frozenset")
    matches: list = []
    for entry in ledger:
        if not isinstance(entry, LedgerEntry):
            raise TypeError("ledger must contain LedgerEntry values")
        if entry.mechanism != facets.get("mechanism"):
            continue
        matched: list = []
        for dimension, values in sorted(entry.match_dimensions.items()):
            observed = _facet_values(facets, dimension)
            if observed is None:
                matched = []
                break
            # Both sides are canonicalised the same way: an entry saying `Q4_K`
            # and a proposal saying `Q4_K` must match whatever their containers,
            # and 1 must not match "1".
            if not (set(_canonical_items(values)) & observed):
                matched = []
                break
            matched.append(dimension)
        if not matched:
            continue
        if not entry.authoritative_against(anchor_commit):
            matches.append(LedgerMatch(
                entry_id=entry.entry_id, entry_class=entry.entry_class,
                mechanism=entry.mechanism, matched_dimensions=tuple(matched),
                rejects=False,
                reason=(
                    "entry is conflicted or its receipt binds to another anchor; a "
                    "suppression whose receipt no longer resolves does not block (§19.3)"
                ),
            ))
            continue
        rejects = entry.entry_class in REJECTING_LEDGER_CLASSES
        reason = _LEDGER_BEHAVIOUR[entry.entry_class]
        if rejects and entry.entry_class == "MATCHED_NEGATIVE" \
                and entry.reopen_when in satisfied_reopen_predicates:
            rejects = False
            reason = (
                f"reopen predicate {entry.reopen_when!r} is newly satisfied, so the "
                "matched negative does not block this round (§19.2)"
            )
        matches.append(LedgerMatch(
            entry_id=entry.entry_id, entry_class=entry.entry_class,
            mechanism=entry.mechanism, matched_dimensions=tuple(matched),
            rejects=rejects, reason=reason,
        ))
    return tuple(matches)


_LEDGER_BEHAVIOUR = {
    "HARD_CONSTRAINT": "hardware, policy, correctness or ownership prohibition (§19.2)",
    "MATCHED_NEGATIVE": (
        "mechanism falsified in a matching regime with adequate evidence and a "
        "receipt; rejected unless a reopen predicate is newly satisfied (§19.2)"
    ),
    "SUPERSEDED_FACT": (
        "current source or production behaviour invalidates the premise; regenerate "
        "from current source/profile rather than executing the stale proposal (§19.2)"
    ),
    "CONDITIONAL_NEGATIVE": "failed only for named cells; matched cells are excluded (§19.2)",
    "CONFOUNDED_RESULT": (
        "unusable because identity, placement, cache or baseline was wrong; its sign "
        "is not learned and a repaired experiment is required (§19.2)"
    ),
    "LOW_VALUE": "below the wall-share/effort threshold; deprioritized, not closed (§19.2)",
}


def _facet_values(facets: Mapping[str, Any], dimension: str) -> Optional[set]:
    """The proposal's declared values for one ledger dimension, or None.

    None means the proposal did not declare that dimension at all, which is not a
    match and not a pass: `screen_proposal` turns it into
    `REJECT_REGIME_IDENTITY_INCOMPLETE`, so a proposal cannot escape a receipted
    negative by declining to say which regime it is in.

    Facet COLLECTIONS are already canonical items (they are built by
    `_canonical_items`); facet SCALARS are raw and are canonicalised here, so
    both sides of every comparison are in one encoding.
    """
    direct = facets.get(dimension)
    if direct is not None:
        return set(direct) if isinstance(direct, (list, tuple)) else set(
            _canonical_items(direct)
        )
    identity = facets.get("regime_identity")
    if isinstance(identity, Mapping) and dimension in identity:
        values = identity[dimension]
        return set(values) if isinstance(values, (list, tuple)) else set(
            _canonical_items(values)
        )
    return None


# =============================================================================
# Mechanism fingerprinting — structural facets only
# =============================================================================

#: The AK4 controller-plane block inside a §7.2 proposal manifest. §7.2's schema
#: is AK1's and is NOT amended here; when it next versions, this block folds in
#: and `fingerprint.SELECTION_BLOCK_KEY` is the one place that changes.
#:
#: Re-exported, not redefined: `planner.py` fingerprints a REJECTED draft and this
#: module fingerprints a FILTERED one, both into the same journal field, so the
#: algorithm is owned by `fingerprint.py` and neither of us keeps a copy.
SELECTION_BLOCK_KEY = fingerprint.SELECTION_BLOCK_KEY
_selection_block = fingerprint.selection_block


_canonical_items = fingerprint.canonical_items


#: The structural identity of what a proposal proposes, and the digest taken over
#: it. Both live in `fingerprint.py` so that a skip recorded by the planner
#: adapter and a skip recorded by this screener are ONE concept in
#: `read_skip_history()`'s count — they were two, and §8.4's auto-blacklist at the
#: second occurrence therefore never fired.
mechanism_facets = fingerprint.mechanism_facets
mechanism_fingerprint = fingerprint.mechanism_fingerprint
proposal_fingerprint = fingerprint.proposal_fingerprint


# =============================================================================
# §8.4 rejection vocabulary
# =============================================================================

REJECT_SCHEMA_INVALID = "SCHEMA_INVALID"
REJECT_CAMPAIGN_MISMATCH = "CAMPAIGN_MISMATCH"
REJECT_MECHANISM_UNNAMED = "MECHANISM_UNNAMED"
REJECT_REGIME_IDENTITY_INCOMPLETE = "REGIME_IDENTITY_INCOMPLETE"
REJECT_FINGERPRINT_BLACKLISTED = "FINGERPRINT_BLACKLISTED"
REJECT_WALL_SHARE_CEILING = "WALL_SHARE_CEILING_EXCEEDED"
REJECT_NO_CORRECTNESS_ORACLE = "NO_CORRECTNESS_ORACLE"
REJECT_SHAPES_NOT_IN_REAL_GRAPH = "SHAPES_NOT_IN_REAL_GRAPH"
REJECT_REPEATS_RECEIPTED_NEGATIVE = "REPEATS_RECEIPTED_NEGATIVE"
REJECT_BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
REJECT_CROSSES_UNOWNED_DOMAIN = "CROSSES_UNOWNED_DOMAIN"
REJECT_REQUIRES_EVALUATOR_CHANGE = "REQUIRES_EVALUATOR_CHANGE"
REJECT_MULTIPLE_CONCEPTUAL_CHANGES = "MULTIPLE_CONCEPTUAL_CHANGES"
REJECT_HIERARCHY_SKIP_UNRECEIPTED = "CHEAPER_LAYER_SKIPPED_WITHOUT_RECEIPT"
REJECT_FORBIDDEN_OBJECTIVE = "FORBIDDEN_OBJECTIVE"
REJECT_TARGETS_CONFIRMATION_SHAPE = "TARGETS_CONFIRMATION_SHAPE"
REJECT_ARCHITECTURAL_ESCAPE_UNDECLARED = "ARCHITECTURAL_ESCAPE_UNDECLARED"
REJECT_LINEAGE_STEP_OUT_OF_ORDER = "LINEAGE_STEP_OUT_OF_ORDER"
REJECT_SPIKE_MALFORMED = "SPIKE_MALFORMED"
REJECT_EIG_OUT_OF_RANGE = "EIG_OUT_OF_RANGE"
REJECT_GAIN_UNDECLARED = "EXPECTED_GAIN_UNDECLARED"
REJECT_UNVERIFIABLE = "UNVERIFIABLE"

REJECTION_CODES = (
    REJECT_SCHEMA_INVALID, REJECT_CAMPAIGN_MISMATCH, REJECT_MECHANISM_UNNAMED,
    REJECT_REGIME_IDENTITY_INCOMPLETE, REJECT_FINGERPRINT_BLACKLISTED,
    REJECT_WALL_SHARE_CEILING, REJECT_NO_CORRECTNESS_ORACLE,
    REJECT_SHAPES_NOT_IN_REAL_GRAPH, REJECT_REPEATS_RECEIPTED_NEGATIVE,
    REJECT_BUDGET_EXCEEDED, REJECT_CROSSES_UNOWNED_DOMAIN,
    REJECT_REQUIRES_EVALUATOR_CHANGE, REJECT_MULTIPLE_CONCEPTUAL_CHANGES,
    REJECT_HIERARCHY_SKIP_UNRECEIPTED, REJECT_FORBIDDEN_OBJECTIVE,
    REJECT_TARGETS_CONFIRMATION_SHAPE, REJECT_ARCHITECTURAL_ESCAPE_UNDECLARED,
    REJECT_LINEAGE_STEP_OUT_OF_ORDER, REJECT_SPIKE_MALFORMED,
    REJECT_EIG_OUT_OF_RANGE, REJECT_GAIN_UNDECLARED, REJECT_UNVERIFIABLE,
)


@dataclass(frozen=True)
class Rejection:
    """One reason a proposal was filtered, in a form the next round can act on."""

    code: str
    reason: str
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.code not in REJECTION_CODES:
            raise ValueError(f"code: {self.code!r} not in {list(REJECTION_CODES)}")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason: required and non-empty — AutoPilot's 119 identical "
                             "invalid actions each carried the exact fix and it never "
                             "reached the planner")

    def to_dict(self) -> dict:
        return {"code": self.code, "reason": self.reason, "detail": dict(self.detail)}


# =============================================================================
# §8.4.1 — architectural campaigns, lineage steps, spikes
# =============================================================================

@dataclass(frozen=True)
class LineageStep:
    """One conceptual change inside a declared architectural lineage (§8.4.1,
    invariant 13). The rule is not waived; it binds per STEP."""

    index: int
    conceptual_change: str
    end_state_contribution: str

    def __post_init__(self) -> None:
        if not isinstance(self.index, int) or isinstance(self.index, bool) or self.index < 0:
            raise ValueError("index: required, a non-negative int")
        for name in ("conceptual_change", "end_state_contribution"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name}: required and non-empty")


@dataclass(frozen=True)
class ArchitecturalCampaign:
    """A declared architectural campaign (§8.4.1, AK-D31).

    Declaring one REPLACES three §8.4 rejection conditions; it waives none. The
    replacements are stricter, not looser: a predicted post-change profile can be
    wrong in a way the profiler can see, where a ceiling test cannot.
    """

    campaign_id: str
    end_state: str
    steps: tuple
    reserved_budget_fraction: float

    def __post_init__(self) -> None:
        for name in ("campaign_id", "end_state"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name}: required and non-empty — §8.4.1 requires a "
                                 "declared lineage WITH a stated end-state")
        if not isinstance(self.steps, tuple) or not self.steps:
            raise ValueError("steps: required, a non-empty tuple of LineageStep")
        for position, step in enumerate(self.steps):
            if not isinstance(step, LineageStep):
                raise TypeError("steps must all be LineageStep")
            if step.index != position:
                raise ValueError(
                    f"steps[{position}].index is {step.index}; steps are ordered and "
                    "their indices are their positions"
                )
        fraction = self.reserved_budget_fraction
        if not isinstance(fraction, (int, float)) or isinstance(fraction, bool):
            raise TypeError("reserved_budget_fraction must be a number")
        if not math.isfinite(float(fraction)) or not 0.0 < float(fraction) < 1.0:
            raise ValueError(
                "reserved_budget_fraction must be strictly inside (0, 1): a zero reserve "
                "is the starvation AK-D31 exists to prevent, and a full reserve is not a "
                "reserve"
            )


@dataclass(frozen=True)
class SpikeDeclaration:
    """A deliberately incomplete prototype that measures whether a mechanism is
    real (§8.4.1, AK-D33).

    It owes no anchor gate, no paired blocks, no e-process and no confirmation
    sample, because it emits a MECHANISM VERDICT and not a rate claim. It still
    owes a resource claim and a preflight, because it runs on shared hardware and
    contaminates its neighbours otherwise.
    """

    spike_id: str
    mechanism_question: str
    resource_lane: str
    claim_receipt: str
    preflight_ref: str
    expected_minutes: float

    def __post_init__(self) -> None:
        for name in ("spike_id", "mechanism_question", "claim_receipt", "preflight_ref"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{name}: required and non-empty — a spike is cheap, not unclaimed "
                    "(AK-D33: it still holds a claim and passes preflight)"
                )
        if self.resource_lane not in schemas.RESOURCE_LANES:
            raise ValueError(
                f"resource_lane: {self.resource_lane!r} not in "
                f"{sorted(schemas.RESOURCE_LANES)}"
            )
        if not isinstance(self.expected_minutes, (int, float)) or isinstance(
            self.expected_minutes, bool
        ):
            raise TypeError("expected_minutes must be a number")
        if not math.isfinite(float(self.expected_minutes)) or float(self.expected_minutes) <= 0:
            raise ValueError("expected_minutes must be finite and strictly positive")

    @staticmethod
    def from_dict(obj: Mapping[str, Any]) -> "SpikeDeclaration":
        if not isinstance(obj, Mapping):
            raise TypeError(f"spike must be a mapping, got {type(obj).__name__}")
        return SpikeDeclaration(
            spike_id=obj.get("spike_id"),
            mechanism_question=obj.get("mechanism_question"),
            resource_lane=obj.get("resource_lane"),
            claim_receipt=obj.get("claim_receipt"),
            preflight_ref=obj.get("preflight_ref"),
            expected_minutes=obj.get("expected_minutes"),
        )

    def to_dict(self) -> dict:
        return {
            "spike_id": self.spike_id,
            "mechanism_question": self.mechanism_question,
            "resource_lane": self.resource_lane,
            "claim_receipt": self.claim_receipt,
            "preflight_ref": self.preflight_ref,
            "expected_minutes": float(self.expected_minutes),
        }


#: A spike may not request these: each is what a RATE CLAIM owes, and a spike
#: does not make one (§8.4.1, AK-D33).
_RATE_CLAIM_TIERS = frozenset({"T1", "T1a", "T1b", "T1c", "T2", "T3", "T4"})


def spike_cost_regression(
    spike_minutes: Sequence[float], t1_minutes: Sequence[float]
) -> schemas.Check:
    """*"if a spike ever costs what a T1 costs, this mechanism has failed"* (§8.4.1).

    A MONITOR, not a per-proposal gate: it reports on the mechanism's health from
    realized costs. There is no supplied threshold — the comparison is the median
    realized spike cost against the CHEAPEST realized T1, both measured. With no
    T1 material the answer is COULD_NOT_CHECK, which is the honest one; a fresh
    campaign has nothing to compare against and pretending otherwise would either
    condemn or bless every spike by default.
    """
    spikes = [float(v) for v in spike_minutes if isinstance(v, (int, float))
              and not isinstance(v, bool) and math.isfinite(float(v))]
    t1s = [float(v) for v in t1_minutes if isinstance(v, (int, float))
           and not isinstance(v, bool) and math.isfinite(float(v))]
    if not spikes:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "no realized spike cost has been recorded yet",
        ))
    if not t1s:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "no realized T1 cost has been recorded yet, so 'as expensive as a T1' has "
            "no measured referent in this campaign",
        ))
    median_spike = ev_statistics.median(spikes)
    cheapest_t1 = min(t1s)
    if median_spike >= cheapest_t1:
        return schemas.Check(schemas.FAIL, (
            f"median realized spike cost {median_spike:.4g} min has reached the cheapest "
            f"realized T1 ({cheapest_t1:.4g} min); §8.4.1's cheap-by-construction "
            "property has failed and the loop will stop using spikes",
        ))
    return schemas.Check(schemas.PASS)


# =============================================================================
# The selection context — measured facts, never planner assertions
# =============================================================================

BUDGET_KEYS = (
    "wall_minutes", "gpu_minutes", "cpu_region_minutes", "storage_gb", "candidates",
)

#: §5.7: the exclusion source per lane, and therefore the budget a lane spends.
LANE_BUDGET_KEY = {"cpu": "cpu_region_minutes", "gpu": "gpu_minutes", "stack": "wall_minutes"}

#: The cap in `manifest["budgets"]` each `BUDGET_KEYS` entry is derived from, and
#: the factor that converts the cap's unit into this module's unit.
#:
#: THREE units meet here and nothing used to convert between them: §7.1's campaign
#: manifest declares HOURS (`max_gpu_hours`), a §7.2 proposal declares MINUTES
#: (`resource_request.expected_minutes`, which `_check_budget` compares against
#: `budget_remaining`), and `context.reduce_budget_ledger()` accumulates SECONDS
#: (`gpu_seconds`). A caller wiring `budget_remaining={"gpu_minutes":
#: manifest["budgets"]["max_gpu_hours"] - used}` — the obvious wiring — makes the
#: budget gate 60x too permissive, in the direction that overspends. So the
#: conversion is stated once, here, and `budget_remaining_from_caps()` is the only
#: sanctioned way across.
BUDGET_CAP_BY_KEY = {
    "wall_minutes": ("max_wall_hours", 60.0),
    "gpu_minutes": ("max_gpu_hours", 60.0),
    "cpu_region_minutes": ("max_cpu_region_hours", 60.0),
    "storage_gb": ("max_storage_gb", 1.0),
    "candidates": ("max_candidates", 1.0),
}

#: Declared in `manifest["budgets"]` and enforced by `guards.guard_budget`, NOT
#: here. A §7.2 proposal declares no token cost — `resource_request` carries only
#: `expected_minutes` and `expected_storage_gb` — so this screen cannot gate a
#: dimension the thing it screens never states. Named rather than omitted: an
#: undeclared budget dimension is an unbounded one wearing a different name.
BUDGET_KEYS_NOT_SCREENED = ("max_controller_tokens",)

SECONDS_PER_MINUTE = 60.0


def _non_negative(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{what}: must be a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{what}: must be finite; a budget the gate cannot compare "
                         "admits every proposal")
    if number < 0:
        raise ValueError(f"{what}: must not be negative; spend that reduces a total "
                         "is how a campaign spends the same hour twice")
    return number


def budget_remaining_from_caps(
    caps: Mapping[str, Any],
    *,
    wall_hours_used: float,
    gpu_seconds_used: float,
    cpu_region_seconds_used: float,
    storage_gb_used: float,
    candidates_used: int,
) -> dict:
    """Campaign caps minus realized spend, in THIS module's units (§7.1, §8.4).

    `caps` is `manifest["budgets"]` verbatim. Wall time is a host fact and comes
    in as hours; GPU and CPU-region spend are journal facts and come in as
    seconds, which is what `context.reduce_budget_ledger()` accumulates.

    A missing cap RAISES rather than defaulting: P-AK-SEARCH-1 precondition 8
    refuses to start a campaign that omits a budget dimension or declares it
    unbounded, and a converter that filled one in would undo that refusal two
    layers down. Remaining is floored at zero — an overspent campaign has no
    budget left, not a negative one, and `BUDGET_STOP` is `guards.guard_budget`'s
    call from the same caps, not this function's.
    """
    if not isinstance(caps, Mapping):
        raise TypeError("caps must be the manifest's `budgets` mapping")
    used = {
        "wall_minutes": _non_negative(wall_hours_used, "wall_hours_used") * 60.0,
        "gpu_minutes": _non_negative(gpu_seconds_used, "gpu_seconds_used")
        / SECONDS_PER_MINUTE,
        "cpu_region_minutes": _non_negative(
            cpu_region_seconds_used, "cpu_region_seconds_used") / SECONDS_PER_MINUTE,
        "storage_gb": _non_negative(storage_gb_used, "storage_gb_used"),
        "candidates": _non_negative(candidates_used, "candidates_used"),
    }
    remaining: dict = {}
    for key in BUDGET_KEYS:
        cap_name, factor = BUDGET_CAP_BY_KEY[key]
        if cap_name not in caps:
            raise ValueError(
                f"budgets.{cap_name}: required to derive {key!r}; a campaign that "
                "omits a budget dimension cannot derive its error budgets and MUST "
                "NOT start (P-AK-SEARCH-1 precondition 8)"
            )
        cap = _non_negative(caps[cap_name], f"budgets.{cap_name}") * factor
        remaining[key] = max(0.0, cap - used[key])
    return remaining


@dataclass(frozen=True)
class SelectionContext:
    """Everything the deterministic screen needs, and nothing the planner asserts.

    `confirmation_shape_digests` holds DIGESTS, never shapes. P-AK-SEARCH-1:
    *"The confirmation stratum's contents MUST NOT appear in planner context"* —
    yet *"a proposal that targets a confirmation shape is rejected before it
    consumes a window"*. Digests satisfy both: the controller can refuse the
    proposal without any context ever holding the shape.
    """

    campaign_id: str
    backend: str
    source_tree: str
    anchor_commit: str
    phase: str
    owned_domains: frozenset
    correctness_oracles: Mapping[str, str]
    real_graph_shape_digests: frozenset
    confirmation_shape_digests: frozenset
    wall_share_receipts: Mapping[str, float]
    measured_profile: Mapping[str, float]
    evaluator_steps: frozenset
    budget_remaining: Mapping[str, float]
    known_event_ids: frozenset
    ledger: tuple = ()
    satisfied_reopen_predicates: frozenset = frozenset()
    architectural: Optional[ArchitecturalCampaign] = None
    open_lineage_step: Optional[int] = None
    microkernel_only: bool = False

    def __post_init__(self) -> None:
        for name in ("campaign_id", "source_tree"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name}: required and non-empty")
        if self.backend not in schemas.BACKENDS:
            raise ValueError(f"backend: {self.backend!r} not in {sorted(schemas.BACKENDS)}")
        if not isinstance(self.anchor_commit, str) or not _COMMIT_RE.match(self.anchor_commit):
            raise ValueError("anchor_commit: required, a 40-char lowercase hex commit")
        if self.phase not in PHASES:
            raise ValueError(f"phase: {self.phase!r} not in {list(PHASES)}")
        for name in ("owned_domains", "real_graph_shape_digests",
                     "confirmation_shape_digests", "evaluator_steps", "known_event_ids",
                     "satisfied_reopen_predicates"):
            if not isinstance(getattr(self, name), frozenset):
                raise TypeError(f"{name} must be a frozenset")
        for name in ("correctness_oracles", "wall_share_receipts", "measured_profile",
                     "budget_remaining"):
            if not isinstance(getattr(self, name), Mapping):
                raise TypeError(f"{name} must be a mapping")
        missing = [k for k in BUDGET_KEYS if k not in self.budget_remaining]
        if missing:
            raise ValueError(
                f"budget_remaining: missing {missing}; a budget check against an "
                "undeclared budget is not a check (precondition 8's shape)"
            )
        if not isinstance(self.ledger, tuple):
            raise TypeError("ledger must be a tuple of LedgerEntry")
        if self.architectural is not None and not isinstance(
            self.architectural, ArchitecturalCampaign
        ):
            raise TypeError("architectural must be an ArchitecturalCampaign or None")
        if self.open_lineage_step is not None:
            if not isinstance(self.open_lineage_step, int) or isinstance(
                self.open_lineage_step, bool
            ):
                raise TypeError("open_lineage_step must be an int or None")


# =============================================================================
# The screen — every §8.4 rejection condition, before mutation
# =============================================================================

@dataclass(frozen=True)
class ScreenResult:
    """The whole verdict on one proposal. `admitted` is the only thing a caller
    acts on; everything else is what the record and the next round get."""

    proposal_id: str
    fingerprint: str
    admitted: bool
    rejections: tuple
    checks: Mapping[str, schemas.Check]
    ledger_matches: tuple
    excluded_cells: tuple
    information_gain: float
    performance_value: float
    arm: str
    tier_cost_rank: int
    oracle_coverage_basis: str
    journal_event_id: Optional[str] = None

    @property
    def codes(self) -> tuple:
        return tuple(r.code for r in self.rejections)

    def to_dict(self) -> dict:
        return {
            "proposal_id": self.proposal_id,
            "fingerprint": self.fingerprint,
            "admitted": self.admitted,
            "rejections": [r.to_dict() for r in self.rejections],
            "checks": {k: {"outcome": v.outcome, "reasons": list(v.reasons)}
                       for k, v in sorted(self.checks.items())},
            "ledger_matches": [
                {"entry_id": m.entry_id, "entry_class": m.entry_class,
                 "matched_dimensions": list(m.matched_dimensions), "rejects": m.rejects,
                 "reason": m.reason}
                for m in self.ledger_matches
            ],
            "excluded_cells": list(self.excluded_cells),
            "information_gain": self.information_gain,
            "performance_value": self.performance_value,
            "arm": self.arm,
            "tier_cost_rank": self.tier_cost_rank,
            "oracle_coverage_basis": self.oracle_coverage_basis,
        }


def _coerce_receipts(value: Any) -> tuple:
    """Accept the manifest's JSON form or already-built receipts.

    Returns `(receipts, errors)`. A malformed receipt does not raise here: it is
    one of the reasons the proposal is filtered, and the round that learns only
    its first defect comes back with its second.
    """
    if value is None:
        return (), ()
    if not isinstance(value, (list, tuple)):
        return (), (f"layer_skip_receipts must be a list, got {type(value).__name__}",)
    built: list = []
    errors: list = []
    for index, item in enumerate(value):
        if isinstance(item, LayerSkipReceipt):
            built.append(item)
            continue
        try:
            built.append(LayerSkipReceipt.from_dict(item))
        except (TypeError, ValueError) as exc:
            errors.append(f"layer_skip_receipts[{index}]: {exc}")
    return tuple(built), tuple(errors)


def _shape_digest(shape: Any) -> str:
    """One shape's identity. Used on both sides of the confirmation-stratum test,
    so a shape and its digest are never both needed in one place."""
    return schemas.content_hash(shape)


def screen_proposal(
    proposal: Mapping[str, Any],
    context: SelectionContext,
    *,
    blacklisted_fingerprints: frozenset,
) -> ScreenResult:
    """Every §8.4 rejection condition, evaluated BEFORE mutation.

    No condition short-circuits: a filtered proposal is journaled with ALL of its
    reasons, because the round that only learns its first defect comes back with
    its second. The three conditions §8.4.1 REPLACES inside a declared
    architectural campaign are replaced here, not skipped — the replacement is
    checked and can fail.

    `blacklisted_fingerprints` is a required argument rather than a context field
    so that it is impossible to screen against a stale copy: `ProposalScreener`
    re-reads it from the journal on every call.
    """
    if not isinstance(context, SelectionContext):
        raise TypeError("context must be a SelectionContext")
    if not isinstance(blacklisted_fingerprints, frozenset):
        raise TypeError("blacklisted_fingerprints must be a frozenset")
    if not isinstance(proposal, Mapping):
        raise TypeError("proposal must be a mapping")

    rejections: list = []
    checks: dict = {}
    excluded_cells: list = []
    block = _selection_block(proposal)
    facets = mechanism_facets(proposal)
    fingerprint = mechanism_fingerprint(facets)
    proposal_id = proposal.get("proposal_id")
    proposal_id = proposal_id if isinstance(proposal_id, str) and proposal_id else "<unidentified>"

    def reject(code: str, reason: str, detail: Optional[Mapping[str, Any]] = None) -> None:
        rejections.append(Rejection(code=code, reason=reason, detail=dict(detail or {})))

    # ---- 0. schema, identity, and the facts every later check reads ---------
    violations = schemas.validate_proposal(proposal)
    if violations:
        reject(REJECT_SCHEMA_INVALID,
               "proposal manifest is not a valid " + schemas.SCHEMA_PROPOSAL,
               {"violations": list(violations)})
    if proposal.get("campaign_id") != context.campaign_id:
        reject(REJECT_CAMPAIGN_MISMATCH,
               f"proposal names campaign {proposal.get('campaign_id')!r} but this context "
               f"is {context.campaign_id!r}; a record consumed by another campaign is "
               "forbidden by P-AK-SEARCH-1 denial 4",
               {"expected": context.campaign_id})

    mechanism = block.get("mechanism")
    if not isinstance(mechanism, str) or not mechanism.strip():
        reject(REJECT_MECHANISM_UNNAMED,
               f"{SELECTION_BLOCK_KEY}.mechanism: required — a proposal whose mechanism is "
               "unnamed cannot be matched against the do-not-repeat ledger, and an "
               "unmatched proposal is how a receipted negative gets repeated")

    # ---- 1. blacklist (§8.4: a repeated fingerprint auto-blacklists) --------
    if fingerprint in blacklisted_fingerprints:
        reject(REJECT_FINGERPRINT_BLACKLISTED,
               f"mechanism fingerprint {fingerprint[:12]} has already been filtered more "
               "than once in this campaign and is auto-blacklisted (§8.4)",
               {"fingerprint": fingerprint})

    # ---- 2. AK-D36/AK-D37: the objective, never the batch regime ------------
    objective_check = _check_objective(block, context)
    checks["objective"] = objective_check
    if objective_check.outcome != schemas.PASS:
        reject(REJECT_FORBIDDEN_OBJECTIVE, "; ".join(objective_check.reasons),
               {"decision_log": "AK-D36/AK-D37"})

    # ---- 3. confirmation stratum (P-AK-SEARCH-1, selection/confirmation) ----
    target = proposal.get("target") if isinstance(proposal.get("target"), Mapping) else {}
    raw_shapes = target.get("shapes")
    target_shapes = raw_shapes if isinstance(raw_shapes, (list, tuple)) else ()
    target_digests = {_shape_digest(s) for s in target_shapes}
    contaminating = sorted(target_digests & context.confirmation_shape_digests)
    if contaminating:
        reject(REJECT_TARGETS_CONFIRMATION_SHAPE,
               "proposal targets a confirmation-stratum shape; it is rejected before it "
               "consumes a window, because selection evidence that touches the "
               "confirmation stratum makes the readiness signal unfit to report",
               {"digests": contaminating})

    # ---- 4. hierarchy (§8.3) ------------------------------------------------
    layer = block.get("hierarchy_layer")
    receipts, receipt_errors = _coerce_receipts(block.get("layer_skip_receipts"))
    for message in receipt_errors:
        reject(REJECT_HIERARCHY_SKIP_UNRECEIPTED,
               f"malformed layer-skip receipt: {message}")
    if layer not in HIERARCHY_RANK:
        reject(REJECT_HIERARCHY_SKIP_UNRECEIPTED,
               f"{SELECTION_BLOCK_KEY}.hierarchy_layer: required, one of {list(HIERARCHY)}; "
               "selection follows the §8.3 hierarchy and a proposal that does not say "
               "where it sits cannot be shown to respect it")
        checks["hierarchy"] = schemas.Check(schemas.COULD_NOT_CHECK, (
            "no hierarchy layer declared",
        ))
    else:
        hierarchy_check = check_layer_skip(
            layer, receipts, anchor_commit=context.anchor_commit,
            known_event_ids=context.known_event_ids,
        )
        checks["hierarchy"] = hierarchy_check
        if hierarchy_check.outcome == schemas.FAIL:
            reject(REJECT_HIERARCHY_SKIP_UNRECEIPTED, "; ".join(hierarchy_check.reasons),
                   {"layer": layer})
        elif hierarchy_check.outcome == schemas.COULD_NOT_CHECK:
            reject(REJECT_UNVERIFIABLE,
                   "hierarchy skip could not be evaluated: " + "; ".join(hierarchy_check.reasons),
                   {"check": "hierarchy"})

    # ---- 5. wall-share ceiling, or the architectural replacement (§8.4.1) ---
    ceiling_check, performance_value = _check_gain_against_ceiling(
        proposal, block, context, reject
    )
    checks["wall_share"] = ceiling_check

    # ---- 6. correctness oracle coverage (§8.4) ------------------------------
    ops = [o for o in (target.get("ops") or []) if isinstance(o, str)]
    uncovered = sorted(o for o in ops if o not in context.correctness_oracles)
    if not ops:
        checks["correctness_oracle"] = schemas.Check(schemas.COULD_NOT_CHECK, (
            "target.ops is empty, so no affected path can be resolved to an oracle",
        ))
        reject(REJECT_NO_CORRECTNESS_ORACLE,
               "target.ops is empty: coverage cannot be established for a path that was "
               "never named, and §8.4 rejects a proposal no correctness oracle covers")
    elif uncovered:
        checks["correctness_oracle"] = schemas.Check(schemas.FAIL, tuple(
            f"{op}: no declared correctness oracle" for op in uncovered
        ))
        reject(REJECT_NO_CORRECTNESS_ORACLE,
               f"no correctness oracle covers {uncovered}; correctness is "
               "lexicographically prior to speed, so an uncovered path is not measurable",
               {"uncovered_ops": uncovered})
    else:
        checks["correctness_oracle"] = schemas.Check(schemas.PASS)

    # ---- 7. shapes occur in a real graph, or are prospective (§8.4.1) -------
    shapes_check = _check_shapes(block, context, target_digests, reject)
    checks["shapes"] = shapes_check

    # ---- 8. do-not-repeat ledger (§19.2, §19.3) ----------------------------
    identity = block.get("regime_identity")
    required_dimensions = sorted({
        dimension
        for entry in context.ledger
        if isinstance(entry, LedgerEntry) and entry.mechanism == mechanism
        for dimension in entry.match_dimensions
    })
    undeclared = [
        d for d in required_dimensions if _facet_values(facets, d) is None
    ]
    if undeclared:
        reject(REJECT_REGIME_IDENTITY_INCOMPLETE,
               f"the ledger holds entries for mechanism {mechanism!r} keyed on {undeclared}, "
               f"which this proposal does not declare in "
               f"{SELECTION_BLOCK_KEY}.regime_identity; a proposal cannot escape a "
               "receipted negative by declining to say which regime it is in",
               {"undeclared_dimensions": undeclared,
                "declared": sorted(identity) if isinstance(identity, Mapping) else []})
    matches = match_ledger(
        facets, context.ledger, anchor_commit=context.anchor_commit,
        satisfied_reopen_predicates=context.satisfied_reopen_predicates,
    )
    for match in matches:
        if match.rejects:
            reject(REJECT_REPEATS_RECEIPTED_NEGATIVE,
                   f"{match.entry_class} {match.entry_id!r} matches on "
                   f"{list(match.matched_dimensions)}: {match.reason}",
                   {"entry_id": match.entry_id, "entry_class": match.entry_class})
        elif match.entry_class == "CONDITIONAL_NEGATIVE":
            excluded_cells.extend(match.matched_dimensions)
        elif match.entry_class == "CONFOUNDED_RESULT":
            checks.setdefault("confounded", schemas.Check(schemas.COULD_NOT_CHECK, (
                f"{match.entry_id}: prior result is confounded; its sign is not learned "
                "and a repaired experiment is required before it is cited (§19.2)",
            )))

    # ---- 9. budget (§8.4) ---------------------------------------------------
    checks["budget"] = _check_budget(proposal, context, reject)

    # ---- 10. repo/release domain ownership (§8.4, AK-D9/AK-D23) ------------
    domains = [d for d in (block.get("domains") or []) if isinstance(d, str)]
    if not domains:
        reject(REJECT_CROSSES_UNOWNED_DOMAIN,
               f"{SELECTION_BLOCK_KEY}.domains: required and non-empty — a change whose "
               "repo/release domain is undeclared cannot be shown to stay inside the "
               "backend adapter's ownership")
    else:
        unowned = sorted(set(domains) - context.owned_domains)
        if unowned:
            reject(REJECT_CROSSES_UNOWNED_DOMAIN,
                   f"domains {unowned} are not owned by the {context.backend} adapter; a "
                   "scheduler or engine change routes to the stack-change gate instead "
                   "of the kernel-freeze path (AK-D9, AK-D23)",
                   {"unowned": unowned})

    # ---- 11. the evaluator is not modifiable (denial 6, invariant 17) ------
    plan = proposal.get("evaluation_plan")
    plan = plan if isinstance(plan, Mapping) else {}
    requested_steps = sorted({
        step
        for key in ("required_t0", "required_t1", "conditional_t2")
        for step in (plan.get(key) or [])
        if isinstance(step, str)
    })
    unimplemented = [s for s in requested_steps if s not in context.evaluator_steps]
    if unimplemented:
        reject(REJECT_REQUIRES_EVALUATOR_CHANGE,
               f"evaluation plan names step(s) {unimplemented} the pinned evaluator bundle "
               "does not implement; the controller RECORDS a coverage gap and does not "
               "patch the instrument (P-AK-SEARCH-1 denial 6, invariant 17)",
               {"unimplemented": unimplemented})

    # ---- 12. one conceptual change, or one per declared step (invariant 13) -
    checks["conceptual_scope"] = _check_conceptual_scope(block, context, reject)

    # ---- 13. spikes are cheap BY CONSTRUCTION (§8.4.1, AK-D33) -------------
    spike = block.get("spike")
    if spike is not None:
        checks["spike"] = _check_spike(spike, plan, context, reject)

    # ---- 14. ranking inputs -------------------------------------------------
    eig = proposal.get("expected_information_gain")
    if not isinstance(eig, (int, float)) or isinstance(eig, bool) or not math.isfinite(
        float(eig)
    ) or not 0.0 <= float(eig) <= 1.0:
        reject(REJECT_EIG_OUT_OF_RANGE,
               "expected_information_gain must be a finite number in [0, 1]; §8.4 ranks on "
               "it first, and an unbounded self-declared score would let the planner order "
               "its own queue",
               {"declared": eig})
        eig_value = 0.0
    else:
        eig_value = float(eig)

    is_architectural = _is_architectural(block, context)
    if spike is not None:
        # A spike emits a mechanism verdict, never a rate claim, so it cannot
        # carry performance value at all. It competes on information alone.
        performance_value = 0.0

    admitted = not rejections
    return ScreenResult(
        proposal_id=proposal_id,
        fingerprint=fingerprint,
        admitted=admitted,
        rejections=tuple(rejections),
        checks=dict(checks),
        ledger_matches=matches,
        excluded_cells=tuple(sorted(set(excluded_cells))),
        information_gain=eig_value,
        performance_value=performance_value,
        arm=ARM_ARCHITECTURAL if is_architectural else ARM_INCREMENTAL,
        tier_cost_rank=_tier_cost_rank(plan),
        oracle_coverage_basis="declared_target_ops",
    )


def _is_architectural(block: Mapping[str, Any], context: SelectionContext) -> bool:
    """True only when the campaign DECLARED an architectural lineage and this
    proposal declared a step of it. A proposal cannot self-declare into the
    reserved arm."""
    if context.architectural is None:
        return False
    return isinstance(block.get("lineage_step"), int) and not isinstance(
        block.get("lineage_step"), bool
    )


def _tier_cost_rank(plan: Mapping[str, Any]) -> int:
    """0 for a plan that stays inside the cheap tiers, 1 for one that reaches T2.

    §8.4.1's HARVEST policy is *"the cheap tiers dominate: many T1, few T2"*, and
    this integer is how that becomes an ordering rather than an aspiration.
    """
    conditional = plan.get("conditional_t2") or []
    return 1 if conditional else 0


def _check_objective(block: Mapping[str, Any], context: SelectionContext) -> schemas.Check:
    """AK-D36/AK-D37: the constraint is on the METRIC, never on the batch regime.

    Single-stream and batched prefill and decode are all legitimate directions and
    improvement is sought independently of batch count. What is refused is
    recruiting a whole-stack cross-engine ratio as a kernel objective, because
    that ratio is dominated by scheduling above 16 concurrent users and would
    spend a kernel campaign on a scheduler property.
    """
    objective = block.get("objective")
    if objective is None:
        # Absent means the campaign's own per-phase objective, which is the only
        # sanctioned one.
        return schemas.Check(schemas.PASS)
    if not isinstance(objective, Mapping):
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            f"{SELECTION_BLOCK_KEY}.objective is "
            f"{type(objective).__name__}, not a mapping",
        ))
    reasons: list = []
    kind = objective.get("kind")
    if kind == "cross_engine_whole_stack_ratio":
        reasons.append(
            "a whole-stack cross-engine throughput ratio is not a kernel objective "
            "(AK-D36): the headline arrives at 16-64 concurrent users and is continuous "
            "batching, PagedAttention and the scheduler — a serving_runtime question "
            "under AK-D9/AK-D23"
        )
    engine = objective.get("comparison_engine")
    if engine is not None and engine != "anchor":
        reasons.append(
            f"comparison_engine {engine!r}: every comparison in this protocol names the "
            "campaign's own immutable anchor; comparing against another engine makes the "
            "record a cross-stack ratio rather than a kernel search record "
            "(P-AK-SEARCH-1 precondition 4, AK-D36)"
        )
    if reasons:
        return schemas.Check(schemas.FAIL, tuple(reasons))
    return schemas.Check(schemas.PASS)


def _check_gain_against_ceiling(
    proposal: Mapping[str, Any],
    block: Mapping[str, Any],
    context: SelectionContext,
    reject: Callable[..., None],
) -> tuple:
    """§8.4's ceiling condition, or §8.4.1's replacement inside a declared campaign.

    Returns `(check, performance_value)`. The value is computed HERE, from the
    measured receipt or the predicted profile, and the planner's own estimate of
    its worth is never used: a self-declared value would order the queue by
    optimism.
    """
    mech = proposal.get("mechanism_prediction")
    mech = mech if isinstance(mech, Mapping) else {}
    gain = block.get("expected_end_to_end_gain")
    if not isinstance(gain, (int, float)) or isinstance(gain, bool) or not math.isfinite(
        float(gain)
    ) or float(gain) < 0.0:
        reject(REJECT_GAIN_UNDECLARED,
               f"{SELECTION_BLOCK_KEY}.expected_end_to_end_gain: required, a finite "
               "non-negative fraction — §8.4's first rejection condition compares it "
               "against the measured wall-share ceiling and cannot run without it",
               {"declared": gain})
        return schemas.Check(schemas.COULD_NOT_CHECK, ("no expected gain declared",)), 0.0
    gain = float(gain)

    architectural = _is_architectural(block, context)
    if architectural:
        profile = block.get("predicted_post_change_profile")
        check = _check_predicted_profile(profile, context)
        if check.outcome == schemas.FAIL:
            reject(REJECT_WALL_SHARE_CEILING,
                   "architectural replacement is not satisfied: " + "; ".join(check.reasons),
                   {"replacement": "predicted_post_change_profile"})
            return check, 0.0
        if check.outcome == schemas.COULD_NOT_CHECK:
            reject(REJECT_UNVERIFIABLE,
                   "predicted post-change profile could not be evaluated: "
                   + "; ".join(check.reasons),
                   {"check": "wall_share"})
            return check, 0.0
        value = 0.0
        for family, predicted in profile.items():
            current = float(context.measured_profile.get(family, 0.0))
            value += max(0.0, current - float(predicted))
        return check, value

    if block.get("predicted_post_change_profile") is not None \
            or block.get("prospective_shapes") is not None:
        reject(REJECT_ARCHITECTURAL_ESCAPE_UNDECLARED,
               "the proposal uses an §8.4.1 replacement (predicted post-change profile or "
               "prospective shapes) but no architectural campaign is declared for it; the "
               "three conditions are REPLACED inside a declared campaign, never waived "
               "outside one")

    receipt_id = mech.get("wall_share_receipt_id")
    if not isinstance(receipt_id, str) or receipt_id not in context.wall_share_receipts:
        check = schemas.Check(schemas.COULD_NOT_CHECK, (
            f"wall_share_receipt_id {receipt_id!r} does not resolve to a measured "
            "ceiling in this context, so the ceiling condition cannot be evaluated",
        ))
        reject(REJECT_UNVERIFIABLE, "; ".join(check.reasons), {"check": "wall_share"})
        return check, 0.0
    measured_ceiling = float(context.wall_share_receipts[receipt_id])
    declared_ceiling = mech.get("expected_wall_share_ceiling")
    reasons: list = []
    if isinstance(declared_ceiling, (int, float)) and not isinstance(declared_ceiling, bool):
        if float(declared_ceiling) > measured_ceiling:
            reasons.append(
                f"declared ceiling {float(declared_ceiling)} exceeds the measured ceiling "
                f"{measured_ceiling} on receipt {receipt_id}; the receipt is the ceiling"
            )
    if gain > measured_ceiling:
        fusion = _fusion_explanation(proposal, block)
        if fusion is None:
            reasons.append(
                f"expected end-to-end gain {gain} exceeds the measured wall-share ceiling "
                f"{measured_ceiling} with no fusion explanation; an optimization inside an "
                "op cannot return more than the op's own share of the wall"
            )
    if reasons:
        check = schemas.Check(schemas.FAIL, tuple(reasons))
        reject(REJECT_WALL_SHARE_CEILING, "; ".join(reasons),
               {"measured_ceiling": measured_ceiling, "expected_gain": gain})
        return check, 0.0
    return schemas.Check(schemas.PASS), min(gain, measured_ceiling)


def _fusion_explanation(
    proposal: Mapping[str, Any], block: Mapping[str, Any]
) -> Optional[str]:
    """§8.4's narrow escape: a fusion removes work between ops, so its return is
    not bounded by either op's share alone. It must be DECLARED, and the change
    class must actually be a fusion — a fusion explanation attached to a
    parameter tweak explains nothing."""
    explanation = block.get("fusion_explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        return None
    if proposal.get("change_class") != "fusion":
        return None
    return explanation


def _check_predicted_profile(
    profile: Any, context: SelectionContext
) -> schemas.Check:
    """§8.4.1's replacement for the ceiling test.

    *"the proposal states what the wall-share distribution BECOMES, per op family.
    This is strictly more falsifiable than a ceiling test — it can be wrong in a
    way the profiler can see."* So it must name every family the measured profile
    holds (a prediction that omits families is unfalsifiable where it is silent),
    the shares must be a distribution, and it must actually differ from today.
    """
    if not isinstance(profile, Mapping) or not profile:
        return schemas.Check(schemas.FAIL, (
            f"{SELECTION_BLOCK_KEY}.predicted_post_change_profile: required and non-empty "
            "inside an architectural campaign — it REPLACES the wall-share ceiling",
        ))
    if not context.measured_profile:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "no measured profile is available to compare the prediction against",
        ))
    reasons: list = []
    missing = sorted(set(context.measured_profile) - set(profile))
    if missing:
        reasons.append(
            f"prediction omits op famil(ies) {missing} present in the measured profile; a "
            "prediction is falsifiable only where it speaks"
        )
    total = 0.0
    for family, share in sorted(profile.items()):
        if not isinstance(share, (int, float)) or isinstance(share, bool) \
                or not math.isfinite(float(share)) or not 0.0 <= float(share) <= 1.0:
            reasons.append(f"{family}: predicted share must be a number in [0, 1]")
            continue
        total += float(share)
    if total > 1.0 + 1e-9:
        reasons.append(
            f"predicted shares sum to {total:.4f}; a wall-share distribution cannot exceed 1"
        )
    if not reasons:
        same = all(
            abs(float(profile[f]) - float(context.measured_profile[f])) <= 1e-12
            for f in context.measured_profile
        )
        if same:
            reasons.append(
                "the predicted profile is identical to the measured one, so it predicts "
                "nothing and no profiler run could refute it"
            )
    if reasons:
        return schemas.Check(schemas.FAIL, tuple(reasons))
    return schemas.Check(schemas.PASS)


def _check_shapes(
    block: Mapping[str, Any],
    context: SelectionContext,
    target_digests: set,
    reject: Callable[..., None],
) -> schemas.Check:
    """§8.4: target shapes must occur in a real graph, unless the campaign is
    explicitly microkernel-only — or, inside an architectural campaign, unless the
    shapes are PROSPECTIVE and the proposal declares the mechanism by which they
    come to occur and a way to observe that they did (§8.4.1)."""
    if not target_digests:
        reject(REJECT_SHAPES_NOT_IN_REAL_GRAPH,
               "target.shapes is empty: a proposal that names no shape cannot be shown to "
               "target one that occurs")
        return schemas.Check(schemas.COULD_NOT_CHECK, ("no target shapes declared",))
    unseen = sorted(target_digests - context.real_graph_shape_digests)
    if not unseen:
        return schemas.Check(schemas.PASS)
    if context.microkernel_only:
        return schemas.Check(schemas.PASS, ())
    prospective = block.get("prospective_shapes")
    if context.architectural is not None and _is_architectural(block, context) \
            and prospective is not None:
        check = _check_prospective_shapes(prospective, unseen)
        if check.outcome != schemas.PASS:
            reject(REJECT_SHAPES_NOT_IN_REAL_GRAPH,
                   "prospective-shape replacement is not satisfied: " + "; ".join(check.reasons),
                   {"unseen_digests": unseen})
        return check
    reject(REJECT_SHAPES_NOT_IN_REAL_GRAPH,
           f"{len(unseen)} target shape(s) do not occur in a captured real graph and the "
           "campaign is neither microkernel-only nor a declared architectural campaign "
           "with prospective shapes",
           {"unseen_digests": unseen})
    return schemas.Check(schemas.FAIL, (
        "target shapes do not occur in a captured real graph",
    ))


def _check_prospective_shapes(prospective: Any, unseen: Sequence[str]) -> schemas.Check:
    """A prospective shape is admissible *"when the proposal declares the mechanism
    by which they come to occur and a way to observe that they did"* (§8.4.1)."""
    if not isinstance(prospective, Sequence) or isinstance(prospective, (str, bytes)) \
            or not prospective:
        return schemas.Check(schemas.FAIL, (
            "prospective_shapes: required, a non-empty list of "
            "{shape_digest, mechanism, observation}",
        ))
    reasons: list = []
    declared: set = set()
    for index, entry in enumerate(prospective):
        if not isinstance(entry, Mapping):
            reasons.append(f"prospective_shapes[{index}]: must be a mapping")
            continue
        digest = entry.get("shape_digest")
        if not isinstance(digest, str) or not digest.strip():
            reasons.append(f"prospective_shapes[{index}].shape_digest: required")
        else:
            declared.add(digest)
        for key in ("mechanism", "observation"):
            value = entry.get(key)
            if not isinstance(value, str) or not value.strip():
                reasons.append(
                    f"prospective_shapes[{index}].{key}: required and non-empty — a shape "
                    "that does not yet occur is admissible only with the mechanism that "
                    "makes it occur AND a way to observe that it did"
                )
    uncovered = sorted(set(unseen) - declared)
    if uncovered:
        reasons.append(
            f"shape(s) {uncovered} occur in no real graph and are not declared prospective"
        )
    if reasons:
        return schemas.Check(schemas.FAIL, tuple(reasons))
    return schemas.Check(schemas.PASS)


def _check_budget(
    proposal: Mapping[str, Any], context: SelectionContext, reject: Callable[..., None]
) -> schemas.Check:
    request = proposal.get("resource_request")
    if not isinstance(request, Mapping):
        reject(REJECT_BUDGET_EXCEEDED,
               "resource_request: required — an unbudgeted proposal cannot be checked "
               "against the campaign's remaining budget")
        return schemas.Check(schemas.COULD_NOT_CHECK, ("no resource_request",))
    lane = request.get("lane")
    if lane not in LANE_BUDGET_KEY:
        reject(REJECT_BUDGET_EXCEEDED,
               f"resource_request.lane {lane!r} is not one of {sorted(LANE_BUDGET_KEY)}")
        return schemas.Check(schemas.COULD_NOT_CHECK, ("no resolvable lane",))
    minutes = request.get("expected_minutes")
    storage = request.get("expected_storage_gb")
    reasons: list = []
    if not isinstance(minutes, (int, float)) or isinstance(minutes, bool):
        reasons.append("resource_request.expected_minutes: required, a number")
    if not isinstance(storage, (int, float)) or isinstance(storage, bool):
        reasons.append("resource_request.expected_storage_gb: required, a number")
    if reasons:
        reject(REJECT_BUDGET_EXCEEDED, "; ".join(reasons))
        return schemas.Check(schemas.COULD_NOT_CHECK, tuple(reasons))

    lane_key = LANE_BUDGET_KEY[lane]
    over: list = []
    for key, wanted in ((lane_key, float(minutes)), ("wall_minutes", float(minutes)),
                        ("storage_gb", float(storage))):
        remaining = float(context.budget_remaining[key])
        if wanted > remaining:
            over.append(f"{key}: needs {wanted}, {remaining} remaining")
    if float(context.budget_remaining["candidates"]) < 1:
        over.append("candidates: the campaign's max_candidates budget is exhausted")
    if over:
        reject(REJECT_BUDGET_EXCEEDED,
               "resource or storage estimate exceeds the remaining budget: "
               + "; ".join(over),
               {"over": over})
        return schemas.Check(schemas.FAIL, tuple(over))
    return schemas.Check(schemas.PASS)


def _check_conceptual_scope(
    block: Mapping[str, Any], context: SelectionContext, reject: Callable[..., None]
) -> schemas.Check:
    """Invariant 13: one conceptual mutation per proposal — per STEP inside a
    declared architectural lineage.

    Outside a declared campaign the proposal must declare exactly one conceptual
    change. Inside one, it declares which step of the lineage it is, the step must
    exist, and steps are taken IN ORDER — a lineage whose steps can be taken in
    any order is not a lineage with an end-state, it is a bag of changes.
    """
    step = block.get("lineage_step")
    declared_count = block.get("conceptual_change_count")
    if context.architectural is None or step is None:
        if step is not None:
            reject(REJECT_ARCHITECTURAL_ESCAPE_UNDECLARED,
                   f"{SELECTION_BLOCK_KEY}.lineage_step is declared but this campaign has "
                   "no architectural lineage; the per-step scope rule is a REPLACEMENT "
                   "inside a declared campaign, not an escape outside one")
            return schemas.Check(schemas.FAIL, ("lineage step outside a declared campaign",))
        if declared_count is None:
            # Not a licence. Passing this check by DELETING the field it inspects
            # is exactly the shape §8.3's "inability to evaluate is not a licence"
            # refuses one section earlier.
            reject(REJECT_UNVERIFIABLE,
                   f"{SELECTION_BLOCK_KEY}.conceptual_change_count: required outside an "
                   "architectural campaign — a proposal that does not say how many "
                   "conceptual mutations it makes cannot be shown to make one, and "
                   "omitting the count would pass the check by removing what it reads "
                   "(invariant 13)",
                   {"check": "conceptual_scope"})
            return schemas.Check(schemas.COULD_NOT_CHECK, (
                "no conceptual_change_count declared",
            ))
        if not isinstance(declared_count, int) or isinstance(declared_count, bool) \
                or declared_count != 1:
            reject(REJECT_MULTIPLE_CONCEPTUAL_CHANGES,
                   f"{SELECTION_BLOCK_KEY}.conceptual_change_count is {declared_count!r}; "
                   "outside an architectural campaign a proposal is one conceptual "
                   "mutation, so that it stays falsifiable and revertible (invariant 13)")
            return schemas.Check(schemas.FAIL, ("more than one conceptual change",))
        return schemas.Check(schemas.PASS)

    campaign = context.architectural
    if not isinstance(step, int) or isinstance(step, bool) or not 0 <= step < len(campaign.steps):
        reject(REJECT_LINEAGE_STEP_OUT_OF_ORDER,
               f"lineage_step {step!r} is not a step of declared lineage "
               f"{campaign.campaign_id!r} (0..{len(campaign.steps) - 1})")
        return schemas.Check(schemas.FAIL, ("lineage step does not exist",))
    if context.open_lineage_step is not None and step != context.open_lineage_step:
        reject(REJECT_LINEAGE_STEP_OUT_OF_ORDER,
               f"lineage_step {step} was proposed while step {context.open_lineage_step} is "
               "the open one; one conceptual change per STEP means the steps are taken in "
               "the declared order toward the stated end-state",
               {"open_step": context.open_lineage_step})
        return schemas.Check(schemas.FAIL, ("lineage step out of order",))
    return schemas.Check(schemas.PASS)


def _check_spike(
    spike: Any, plan: Mapping[str, Any], context: SelectionContext,
    reject: Callable[..., None],
) -> schemas.Check:
    """§8.4.1/AK-D33: cheap BY CONSTRUCTION, and still claimed.

    The cheapness is structural, not a budget number: a spike that requests a rate
    tier is requesting paired blocks, an e-process and a confirmation sample, at
    which point it costs what a T1 costs and the mechanism has failed.
    """
    if not isinstance(spike, SpikeDeclaration):
        try:
            spike = SpikeDeclaration.from_dict(spike)
        except (TypeError, ValueError) as exc:
            reject(REJECT_SPIKE_MALFORMED,
                   f"{SELECTION_BLOCK_KEY}.spike is not a valid spike declaration: {exc}; a "
                   "spike is cheap, not unclaimed — it still holds a resource claim and "
                   "passes preflight (AK-D33)")
            return schemas.Check(schemas.FAIL, (str(exc),))
    reasons: list = []
    rate_tiers = sorted({
        step
        for key in ("required_t1", "conditional_t2")
        for step in (plan.get(key) or [])
        if isinstance(step, str)
    })
    if rate_tiers:
        reasons.append(
            f"spike evaluation plan requests rate cells {rate_tiers}; a spike emits a "
            "mechanism verdict and not a rate claim, so it owes no anchor gate, paired "
            "blocks, e-process or confirmation sample — and must not buy them either"
        )
    for key in ("required_t0",):
        for step in (plan.get(key) or []):
            if isinstance(step, str) and step in _RATE_CLAIM_TIERS:
                reasons.append(f"spike required_t0 names rate tier {step!r}")
    if spike.resource_lane == "gpu" and context.backend not in ("llama_gpu",):
        reasons.append(
            f"spike claims the gpu lane on backend {context.backend}, which does not "
            "serve it"
        )
    if reasons:
        reject(REJECT_SPIKE_MALFORMED, "; ".join(reasons), {"spike_id": spike.spike_id})
        return schemas.Check(schemas.FAIL, tuple(reasons))
    return schemas.Check(schemas.PASS)


# =============================================================================
# PROPOSAL_SKIPPED — the record, the blacklist, and the degraded signal
# =============================================================================

@dataclass(frozen=True)
class SkipRecord:
    event_id: str
    seq: int
    proposal_ref: str
    fingerprint: str
    reason: str
    codes: tuple
    stage: str


@dataclass(frozen=True)
class SkipHistory:
    """What the journal says about filtered proposals, folded deterministically.

    `blacklisted` is derived, never stored: §8.4's *"a repeated fingerprint
    auto-blacklists"* is a property of the record, and a blacklist held anywhere
    else is a cache that can disagree with it.
    """

    records: tuple
    counts: Mapping[str, int]
    blacklisted: frozenset
    trailing_run: int

    def feedback(self) -> tuple:
        """What §8.4 feeds into the next planning context. Deterministically
        ordered so two readers of one journal render the same brief."""
        by_fingerprint: dict = {}
        for record in self.records:
            slot = by_fingerprint.setdefault(record.fingerprint, {
                "fingerprint": record.fingerprint, "count": 0, "codes": set(),
                "last_reason": "", "last_event_id": "",
            })
            slot["count"] += 1
            slot["codes"].update(record.codes)
            slot["last_reason"] = record.reason
            slot["last_event_id"] = record.event_id
        rows = [
            SkipFeedback(
                fingerprint=slot["fingerprint"],
                count=slot["count"],
                codes=tuple(sorted(slot["codes"])),
                last_reason=slot["last_reason"],
                last_event_id=slot["last_event_id"],
                blacklisted=slot["fingerprint"] in self.blacklisted,
            )
            for slot in by_fingerprint.values()
        ]
        rows.sort(key=lambda r: (-r.count, r.fingerprint))
        return tuple(rows)


@dataclass(frozen=True)
class SkipFeedback:
    fingerprint: str
    count: int
    codes: tuple
    last_reason: str
    last_event_id: str
    blacklisted: bool


#: §8.4: *"a repeated fingerprint auto-blacklists"*. Repeated means seen twice.
#:
#: `planner.assess_repetition` is the PURE in-memory sibling of this fold and
#: encodes the same rule. It is not called from here because it also computes a
#: degradation verdict and demands a `degraded_run` policy value, which a
#: journal read has no business inventing — so the two are kept honest by
#: `test_selection.TestCrossModuleAgreement`, which fails if they ever disagree
#: about what "repeated" means. The journal-backed fold lives here because
#: `planner.py` deliberately holds no journal.
_BLACKLIST_AT = 2


def read_skip_history(journal_: journal.Journal, *, campaign_id: Optional[str] = None) -> SkipHistory:
    """Fold `PROPOSAL_SKIPPED` events into counts, a blacklist, and a trailing run.

    Reads the RECORD (`read_all`), not a derived view: `rebuild_views` folds no
    slot for `PROPOSAL_SKIPPED`, and inventing one here would create a second
    source of truth for the same events.
    """
    if not isinstance(journal_, journal.Journal):
        raise TypeError("journal_ must be a journal.Journal")
    records: list = []
    trailing = 0
    for entry in journal_.read_all():
        if entry.kind != journal.KIND_PROPOSAL_SKIPPED:
            if entry.kind in (journal.KIND_PROPOSAL_RECORDED, journal.KIND_CANDIDATE_RECORDED) \
                    and (campaign_id is None or entry.campaign_id in (None, campaign_id)):
                # An admitted proposal breaks the run: §8.10 distinguishes a
                # broken searcher from a slow one, and a run interrupted by real
                # work is not a run. It must be THIS campaign's work — a
                # neighbouring campaign's progress is not evidence that this
                # planner recovered, and letting it clear the run would silence
                # PLANNER_DEGRADED on a shared journal.
                trailing = 0
            continue
        if campaign_id is not None and entry.campaign_id not in (None, campaign_id):
            continue
        payload = entry.payload
        detail = payload.get("detail")
        detail = detail if isinstance(detail, Mapping) else {}
        records.append(SkipRecord(
            event_id=entry.event_id,
            seq=entry.seq,
            proposal_ref=str(payload.get("proposal_ref")),
            fingerprint=str(payload.get("fingerprint") or ""),
            reason=str(payload.get("reason")),
            codes=tuple(detail.get("reason_codes") or ()),
            stage=str(detail.get("stage") or "screen"),
        ))
        trailing += 1
    counts: dict = {}
    for record in records:
        if record.fingerprint:
            counts[record.fingerprint] = counts.get(record.fingerprint, 0) + 1
    blacklisted = frozenset(f for f, n in counts.items() if n >= _BLACKLIST_AT)
    return SkipHistory(
        records=tuple(records), counts=counts, blacklisted=blacklisted, trailing_run=trailing,
    )


def planner_health_stop_request(
    history: SkipHistory, *, stop_policy: Mapping[str, Any]
) -> Optional[state_machine.StopRequest]:
    """*"a run of them trips PLANNER_DEGRADED"* (§8.4, §8.10).

    The run length is a declared campaign INPUT, in the shape of P-AK-SEARCH-1
    precondition 8 — this module refuses to invent one, because a controller that
    picks its own tolerance for its own malfunction is grading itself. The
    returned object is a REQUEST: `ControllerStateMachine.dispose_stop_request`
    validates the evidence and owns the transition.
    """
    if not isinstance(history, SkipHistory):
        raise TypeError("history must be a SkipHistory")
    if not isinstance(stop_policy, Mapping):
        raise TypeError("stop_policy must be a mapping")
    limit = stop_policy.get("max_consecutive_proposal_skips")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError(
            "stop_policy.max_consecutive_proposal_skips: required, a positive int — a "
            "declared campaign input, not a threshold this module may choose for itself"
        )
    if history.trailing_run < limit:
        return None
    receipt = [r.event_id for r in history.records[-history.trailing_run:]]
    return state_machine.StopRequest(
        state=state_machine.PLANNER_DEGRADED,
        reason=(
            f"{history.trailing_run} consecutive filtered proposals with no admitted "
            f"proposal between them, against a declared limit of {limit}"
        ),
        detail={
            "signal": "consecutive_proposal_skipped",
            "receipt": receipt,
            "trailing_run": history.trailing_run,
            "declared_limit": limit,
            "repeated_fingerprints": sorted(history.blacklisted),
        },
        origin="controller",
    )


# =============================================================================
# Prescreen and metered drafting — cheap checks FIRST
# =============================================================================

@dataclass(frozen=True)
class DraftBrief:
    """What a cheap deterministic check can evaluate BEFORE a token is spent.

    Everything here is structural. Nothing here needs a model, and that is the
    point: §8.4's ordering rule exists because the reverse cost roughly 38
    draft-and-critique cycles that were paid for and then thrown away.
    """

    seed_id: str
    mechanism: str
    hierarchy_layer: str
    change_class: str
    campaign_kind: str
    regime_identity: Mapping[str, tuple]
    target_ops: tuple = ()
    target_regimes: tuple = ()
    target_shape_digests: tuple = ()
    domains: tuple = ()
    layer_skip_receipts: tuple = ()
    estimated_minutes: float = 0.0
    estimated_storage_gb: float = 0.0
    lane: str = "cpu"

    def __post_init__(self) -> None:
        for name in ("seed_id", "mechanism"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name}: required and non-empty")
        if self.hierarchy_layer not in HIERARCHY_RANK:
            raise ValueError(f"hierarchy_layer: {self.hierarchy_layer!r} is not in HIERARCHY")
        if self.change_class not in schemas.CHANGE_CLASSES:
            raise ValueError(f"change_class: {self.change_class!r} is not a declared class")
        if self.campaign_kind not in schemas.CAMPAIGN_KINDS:
            raise ValueError(f"campaign_kind: {self.campaign_kind!r} is not a declared kind")
        if self.lane not in LANE_BUDGET_KEY:
            raise ValueError(f"lane: {self.lane!r} is not one of {sorted(LANE_BUDGET_KEY)}")
        if not isinstance(self.regime_identity, Mapping):
            raise TypeError("regime_identity must be a mapping")
        if not isinstance(self.layer_skip_receipts, tuple) or any(
            not isinstance(r, LayerSkipReceipt) for r in self.layer_skip_receipts
        ):
            raise TypeError("layer_skip_receipts must be a tuple of LayerSkipReceipt")
        # `prescreen` compares these against the remaining budget with `>`. A NaN
        # loses every comparison, so an unvalidated cost is a seed the budget gate
        # cannot refuse — it would be admitted against a budget of zero.
        for name in ("estimated_minutes", "estimated_storage_gb"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) \
                    or not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(
                    f"{name}: required, a finite non-negative number — a cost the budget "
                    "check cannot compare is a cost it cannot refuse"
                )

    def facets(self) -> dict:
        """The facets the ledger and the blacklist key on. Deliberately the same
        shape `mechanism_facets` produces, so a brief and the proposal drafted
        from it live in one fingerprint space."""
        return {
            "mechanism": self.mechanism,
            "hierarchy_layer": self.hierarchy_layer,
            "change_class": self.change_class,
            "campaign_kind": self.campaign_kind,
            "ops": _canonical_items(self.target_ops),
            "regimes": _canonical_items(self.target_regimes),
            "regime_identity": {
                k: sorted(_canonical_items(v)) for k, v in sorted(self.regime_identity.items())
            },
        }


@dataclass(frozen=True)
class PrescreenTicket:
    """Proof that the cheap checks ran, and on WHICH mechanism.

    The ticket binds the brief's facet fingerprint. `MeteredDraftGuard` refuses a
    drafted proposal whose own mechanism facets do not match it, so a cheap idea
    cannot be screened and an expensive one drafted under its cover.
    """

    seed_id: str
    brief_fingerprint: str
    issued_at: str
    checks: Mapping[str, schemas.Check]


@dataclass(frozen=True)
class PrescreenOutcome:
    admitted: bool
    ticket: Optional[PrescreenTicket]
    rejections: tuple
    brief_fingerprint: str


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def prescreen(
    brief: DraftBrief,
    context: SelectionContext,
    *,
    blacklisted_fingerprints: frozenset,
    clock: Optional[Callable[[], str]] = None,
) -> PrescreenOutcome:
    """The cheap deterministic checks, run before any metered drafting (§8.4).

    Only checks that need no drafted manifest live here: the blacklist, the
    receipted-negative ledger, the §8.3 hierarchy receipts, budget headroom,
    domain ownership, the AK-D36 objective, and confirmation-stratum targeting.
    Everything that needs the drafted manifest stays in `screen_proposal`, which
    runs again afterwards — the prescreen narrows, it never substitutes.
    """
    if not isinstance(brief, DraftBrief):
        raise TypeError("brief must be a DraftBrief")
    if not isinstance(context, SelectionContext):
        raise TypeError("context must be a SelectionContext")
    if not isinstance(blacklisted_fingerprints, frozenset):
        raise TypeError("blacklisted_fingerprints must be a frozenset")
    facets = brief.facets()
    fingerprint = mechanism_fingerprint(facets)
    rejections: list = []
    checks: dict = {}

    if fingerprint in blacklisted_fingerprints:
        rejections.append(Rejection(
            code=REJECT_FINGERPRINT_BLACKLISTED,
            reason=(f"seed fingerprint {fingerprint[:12]} is auto-blacklisted; drafting it "
                    "again would pay for a rejection the record already holds"),
            detail={"fingerprint": fingerprint},
        ))

    hierarchy_check = check_layer_skip(
        brief.hierarchy_layer, brief.layer_skip_receipts,
        anchor_commit=context.anchor_commit, known_event_ids=context.known_event_ids,
    )
    checks["hierarchy"] = hierarchy_check
    if hierarchy_check.outcome != schemas.PASS:
        rejections.append(Rejection(
            code=REJECT_HIERARCHY_SKIP_UNRECEIPTED,
            reason=("cheaper hierarchy layers are unreceipted at seed time: "
                    + "; ".join(hierarchy_check.reasons)),
            detail={"layer": brief.hierarchy_layer},
        ))

    matches = match_ledger(
        facets, context.ledger, anchor_commit=context.anchor_commit,
        satisfied_reopen_predicates=context.satisfied_reopen_predicates,
    )
    for match in matches:
        if match.rejects:
            rejections.append(Rejection(
                code=REJECT_REPEATS_RECEIPTED_NEGATIVE,
                reason=f"{match.entry_class} {match.entry_id!r}: {match.reason}",
                detail={"entry_id": match.entry_id},
            ))

    lane_key = LANE_BUDGET_KEY[brief.lane]
    over = []
    if float(brief.estimated_minutes) > float(context.budget_remaining[lane_key]):
        over.append(f"{lane_key}: needs {brief.estimated_minutes}")
    if float(brief.estimated_storage_gb) > float(context.budget_remaining["storage_gb"]):
        over.append(f"storage_gb: needs {brief.estimated_storage_gb}")
    if float(context.budget_remaining["candidates"]) < 1:
        over.append("candidates: exhausted")
    if over:
        rejections.append(Rejection(
            code=REJECT_BUDGET_EXCEEDED,
            reason="seed cost exceeds the remaining budget: " + "; ".join(over),
            detail={"over": over},
        ))

    if brief.domains:
        unowned = sorted(set(brief.domains) - context.owned_domains)
        if unowned:
            rejections.append(Rejection(
                code=REJECT_CROSSES_UNOWNED_DOMAIN,
                reason=f"seed domains {unowned} are not owned by the {context.backend} adapter",
                detail={"unowned": unowned},
            ))

    contaminating = sorted(set(brief.target_shape_digests) & context.confirmation_shape_digests)
    if contaminating:
        rejections.append(Rejection(
            code=REJECT_TARGETS_CONFIRMATION_SHAPE,
            reason="seed targets a confirmation-stratum shape and is refused before it "
                   "consumes drafting budget, let alone a window",
            detail={"digests": contaminating},
        ))

    if rejections:
        return PrescreenOutcome(False, None, tuple(rejections), fingerprint)
    stamp = (clock or _iso_now)()
    return PrescreenOutcome(
        True,
        PrescreenTicket(seed_id=brief.seed_id, brief_fingerprint=fingerprint,
                        issued_at=stamp, checks=dict(checks)),
        (),
        fingerprint,
    )


class MeteredDraftGuard:
    """The ONLY sanctioned route to a drafter (§8.4 ordering rule).

    It is a guard rather than a convention because a convention is exactly what
    was violated the first time: cheap checks ran after drafting, and 38 cycles
    were paid for and thrown away. `draft()` refuses without a ticket, refuses a
    ticket for another seed, and refuses a drafted proposal whose mechanism
    diverged from the screened one.

    The drafter is a caller-supplied callable. Nothing here knows or cares
    whether it is a model, and the suite exercises it with a fake.
    """

    __slots__ = ("_drafter", "_calls")

    def __init__(self, drafter: Callable[[DraftBrief], Mapping[str, Any]]) -> None:
        if not callable(drafter):
            raise TypeError("drafter must be callable")
        self._drafter = drafter
        self._calls = 0

    @property
    def calls(self) -> int:
        """How many times metered drafting was actually entered. Asserted by the
        suite: a guard that lets one call through is not a guard."""
        return self._calls

    def draft(self, brief: DraftBrief, ticket: Optional[PrescreenTicket]) -> Mapping[str, Any]:
        if not isinstance(brief, DraftBrief):
            raise TypeError("brief must be a DraftBrief")
        if ticket is None:
            raise DraftingRefused(
                f"seed {brief.seed_id!r}: no prescreen ticket; cheap deterministic checks "
                "run BEFORE metered drafting, not after (§8.4)"
            )
        if not isinstance(ticket, PrescreenTicket):
            raise TypeError("ticket must be a PrescreenTicket")
        if ticket.seed_id != brief.seed_id:
            raise DraftingRefused(
                f"ticket was issued for seed {ticket.seed_id!r}, not {brief.seed_id!r}"
            )
        expected = mechanism_fingerprint(brief.facets())
        if ticket.brief_fingerprint != expected:
            raise DraftingRefused(
                "ticket does not match this brief's mechanism fingerprint; the brief was "
                "edited after it was screened"
            )
        self._calls += 1
        drafted = self._drafter(brief)
        if not isinstance(drafted, Mapping):
            raise TypeError("drafter must return a proposal mapping")
        drafted_facets = mechanism_facets(drafted)
        brief_facets = brief.facets()
        # Every facet the two shapes SHARE, not merely the labels. `target.ops`
        # and `target.regimes` are what the ledger, the oracle-coverage condition
        # and the wall-share receipt are all keyed on, so a drafter free to move
        # them could have the prescreen clear one target and the draft arrive
        # against another.
        for key in ("mechanism", "hierarchy_layer", "change_class", "campaign_kind",
                    "ops", "regimes"):
            if drafted_facets.get(key) != brief_facets.get(key):
                raise DraftedProposalDiverged(
                    f"drafted proposal's {key} is {drafted_facets.get(key)!r} but the "
                    f"screened brief declared {brief_facets.get(key)!r}; screening one "
                    "mechanism and drafting another would make the prescreen a formality"
                )
        # The draft may say MORE about its regime than the seed did; it may not
        # contradict what the seed was screened against.
        brief_identity = brief_facets.get("regime_identity") or {}
        drafted_identity = drafted_facets.get("regime_identity") or {}
        for dimension, values in sorted(brief_identity.items()):
            observed = drafted_identity.get(dimension)
            if observed is None or not set(values) <= set(observed):
                raise DraftedProposalDiverged(
                    f"drafted proposal's regime_identity.{dimension} is {observed!r} but "
                    f"the screened brief declared {list(values)!r}; the ledger match that "
                    "cleared this seed was computed against the seed's regime"
                )
        return drafted


# =============================================================================
# The screener — screen, and journal every rejection
# =============================================================================

class ProposalScreener:
    """Screens proposals against a live journal, and RECORDS every rejection.

    Two disciplines are structural here:

    * **the blacklist is re-read from the record on every call**, never held. It
      is the same shape invariant 19 demands of the control latch, for the same
      reason: a cached copy of a durable fact is a copy that can silently
      disagree with it.
    * **journal-then-return.** The `PROPOSAL_SKIPPED` append happens before the
      caller is told the proposal was filtered, and a failed append raises
      `SkipNotRecorded`. §8.4 forbids a bare discard; a rejection the caller
      learned about but the record did not is exactly that.
    """

    __slots__ = ("_journal", "_campaign_id")

    def __init__(self, journal_: journal.Journal, *, campaign_id: str) -> None:
        if not isinstance(journal_, journal.Journal):
            raise TypeError("journal_ must be a journal.Journal")
        if not isinstance(campaign_id, str) or not campaign_id.strip():
            raise ValueError("campaign_id: required and non-empty")
        self._journal = journal_
        self._campaign_id = campaign_id

    def blacklist(self) -> frozenset:
        return read_skip_history(self._journal, campaign_id=self._campaign_id).blacklisted

    def history(self) -> SkipHistory:
        return read_skip_history(self._journal, campaign_id=self._campaign_id)

    def screen(self, proposal: Mapping[str, Any], context: SelectionContext) -> ScreenResult:
        if context.campaign_id != self._campaign_id:
            raise ValueError(
                f"context is for campaign {context.campaign_id!r}, this screener is for "
                f"{self._campaign_id!r}; consumption is confined to the campaign that "
                "produced the record (P-AK-SEARCH-1 denial 4)"
            )
        result = screen_proposal(
            proposal, context, blacklisted_fingerprints=self.blacklist()
        )
        if result.admitted:
            return result
        entry = self._record_skip(result, stage="screen")
        return ScreenResult(
            proposal_id=result.proposal_id,
            fingerprint=result.fingerprint,
            admitted=False,
            rejections=result.rejections,
            checks=result.checks,
            ledger_matches=result.ledger_matches,
            excluded_cells=result.excluded_cells,
            information_gain=result.information_gain,
            performance_value=result.performance_value,
            arm=result.arm,
            tier_cost_rank=result.tier_cost_rank,
            oracle_coverage_basis=result.oracle_coverage_basis,
            journal_event_id=entry.event_id,
        )

    def record_prescreen_rejection(
        self, brief: DraftBrief, outcome: PrescreenOutcome
    ) -> journal.JournalEntry:
        """A seed refused before drafting is still a filtered proposal.

        Journaling it is what makes the prescreen visible to the next round —
        otherwise the cheapest rejections, which are the ones a planner most needs
        to learn from, would be the only invisible ones.
        """
        if outcome.admitted:
            raise ValueError("record_prescreen_rejection() takes a refused outcome")
        return self._append_skip(
            proposal_ref=brief.seed_id,
            fingerprint=outcome.brief_fingerprint,
            rejections=outcome.rejections,
            stage="prescreen",
        )

    def _record_skip(self, result: ScreenResult, *, stage: str) -> journal.JournalEntry:
        return self._append_skip(
            proposal_ref=result.proposal_id,
            fingerprint=result.fingerprint,
            rejections=result.rejections,
            stage=stage,
        )

    def _append_skip(
        self, *, proposal_ref: str, fingerprint: str, rejections: tuple, stage: str
    ) -> journal.JournalEntry:
        # Built through `planner.skip_payload` so AK4 has ONE `PROPOSAL_SKIPPED`
        # shape rather than one per module; it also canonical-json-checks the
        # payload here, where a non-serialisable rejection detail is a bug in the
        # screen, instead of at the journal append two frames later.
        payload = planner.skip_payload(
            proposal_ref=proposal_ref,
            reason="; ".join(f"[{r.code}] {r.reason}" for r in rejections),
            fingerprint=fingerprint,
            detail={
                "reason_codes": [r.code for r in rejections],
                "rejections": [r.to_dict() for r in rejections],
                "stage": stage,
            },
        )
        payload["campaign_id"] = self._campaign_id
        try:
            return self._journal.append(
                journal.KIND_PROPOSAL_SKIPPED, payload, campaign_id=self._campaign_id
            )
        except Exception as exc:  # the append is the record; a failure is not a rejection
            raise SkipNotRecorded(
                f"{proposal_ref}: PROPOSAL_SKIPPED could not be journaled "
                f"({type(exc).__name__}: {exc}); §8.4 forbids a bare discard, so the "
                "filtering did not happen either"
            ) from exc


# =============================================================================
# §8.4.1 — HARVEST and EXPLORE, switched on marginal yield
# =============================================================================

PHASE_HARVEST = "HARVEST"
PHASE_EXPLORE = "EXPLORE"
PHASES = (PHASE_HARVEST, PHASE_EXPLORE)

TRIGGER_DEEP_LEVER = "deep_lever_landed"
TRIGGER_ANCHOR_MOVE = "anchor_moved"
TRIGGER_YIELD_DECAY = "yield_decayed_below_derived_floor"
TRIGGER_DWELL = "minimum_dwell_not_elapsed"
TRIGGER_WINDOW = "trailing_window_incomplete"
TRIGGER_DEGRADED = "planner_degraded_not_explore"
TRIGGER_YIELD_HOLDING = "yield_still_above_floor"


@dataclass(frozen=True)
class YieldObservation:
    """One round's contribution to the phase signal.

    `receipt` is mandatory: the phase switch is a decision, and §8.10 wants the
    evidence that produced it to be nameable when it becomes a stop.
    """

    round_index: int
    banked_gain: float
    budget_spent: float
    proposal_skipped_count: int
    repeated_fingerprint_count: int
    receipt: str
    deep_lever_landed: bool = False
    anchor_moved: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.round_index, int) or isinstance(self.round_index, bool) \
                or self.round_index < 0:
            raise ValueError("round_index: required, a non-negative int")
        for name in ("banked_gain", "budget_spent"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) \
                    or not math.isfinite(float(value)):
                raise TypeError(f"{name}: required, a finite number")
        if float(self.banked_gain) < 0.0:
            raise ValueError("banked_gain must be non-negative")
        if float(self.budget_spent) <= 0.0:
            raise ValueError(
                "budget_spent must be strictly positive; marginal yield is gain PER UNIT "
                "of budget and a round that spent nothing has no yield"
            )
        for name in ("proposal_skipped_count", "repeated_fingerprint_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name}: required, a non-negative int")
        if not isinstance(self.receipt, str) or not self.receipt.strip():
            raise ValueError("receipt: required and non-empty")

    @property
    def marginal_yield(self) -> float:
        return float(self.banked_gain) / float(self.budget_spent)


@dataclass(frozen=True)
class YieldCalibration:
    """The DERIVED decay floor, trailing window, and minimum dwell (§8.4.1).

    *"Like every other threshold, the decay floor and window are derived by the
    campaign calibration procedure, never supplied — a supplied number here would
    decide the explore/exploit tradeoff by guess."*

    The derivation has no free parameter, which is what makes "never supplied"
    checkable rather than aspirational:

    * **floor** = the SMALLEST marginal yield the region produced while it was
      still worth harvesting. Falling below it means the region now yields less
      than it ever did during its own harvest — decay by the region's own
      standard, on the region's own scale.
    * **window** = the number of harvest samples the floor was derived from.
      Judging decay over a shorter window than the evidence that set the floor
      would compare a noisier estimate against a quieter one.
    * **dwell** = the window. A phase that can end on evidence shorter than the
      window able to end it is a phase that thrashes.

    Construction RECOMPUTES all three from `derivation_samples` and refuses a
    mismatch, so a hand-built object carrying a convenient floor does not exist.
    """

    floor: float
    window_rounds: int
    min_dwell_rounds: int
    derivation_samples: tuple
    derivation_id: str
    method: str = "min_of_early_harvest_marginal_yield/v1"

    def __post_init__(self) -> None:
        if not isinstance(self.derivation_id, str) or not self.derivation_id.strip():
            raise ValueError("derivation_id: required and non-empty")
        recomputed = _derive_from_samples(self.derivation_samples)
        mismatches = [
            f"{name}: carried {carried!r}, recomputed {expected!r}"
            for name, carried, expected in (
                ("floor", float(self.floor), recomputed["floor"]),
                ("window_rounds", int(self.window_rounds), recomputed["window_rounds"]),
                ("min_dwell_rounds", int(self.min_dwell_rounds),
                 recomputed["min_dwell_rounds"]),
            )
            if carried != expected
        ]
        if mismatches:
            raise CalibrationTampered(
                "the calibration disagrees with its own derivation samples ("
                + "; ".join(mismatches)
                + "); §8.4.1 derives the floor and window, it never accepts them"
            )

    def verify(self) -> schemas.Check:
        """Re-derive and report. Cheap, and called on every phase decision so a
        calibration mutated after construction cannot drive one."""
        try:
            recomputed = _derive_from_samples(self.derivation_samples)
        except InsufficientYieldMaterial as exc:
            return schemas.Check(schemas.COULD_NOT_CHECK, (str(exc),))
        if (float(self.floor), int(self.window_rounds), int(self.min_dwell_rounds)) != (
            recomputed["floor"], recomputed["window_rounds"], recomputed["min_dwell_rounds"]
        ):
            return schemas.Check(schemas.FAIL, (
                "calibration values do not match their derivation samples",
            ))
        return schemas.Check(schemas.PASS)


def _derive_from_samples(samples: Any) -> dict:
    if not isinstance(samples, tuple) or len(samples) < 2:
        raise InsufficientYieldMaterial(
            "derivation_samples: at least two harvest marginal yields are required — one "
            "sample is a point, and a window of one is not a window"
        )
    values = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, (int, float)) or isinstance(sample, bool) \
                or not math.isfinite(float(sample)):
            raise InsufficientYieldMaterial(
                f"derivation_samples[{index}]: must be a finite number"
            )
        if float(sample) <= 0.0:
            raise InsufficientYieldMaterial(
                f"derivation_samples[{index}] is {sample}; a non-positive marginal yield is "
                "not evidence of a region worth harvesting, and a floor derived from one "
                "could never be crossed"
            )
        values.append(float(sample))
    return {
        "floor": min(values),
        "window_rounds": len(values),
        "min_dwell_rounds": len(values),
    }


def derive_yield_calibration(
    harvest_samples: Sequence[float], *, derivation_id: str
) -> YieldCalibration:
    """Derive the floor, window and dwell from the harvest's own yield samples."""
    samples = tuple(harvest_samples)
    derived = _derive_from_samples(samples)
    return YieldCalibration(
        floor=derived["floor"],
        window_rounds=derived["window_rounds"],
        min_dwell_rounds=derived["min_dwell_rounds"],
        derivation_samples=samples,
        derivation_id=derivation_id,
    )


@dataclass(frozen=True)
class PhaseDecision:
    phase: str
    changed: bool
    trigger: str
    reason: str
    phase_started_round: int
    window: tuple
    stop_request: Optional[state_machine.StopRequest] = None


def decide_phase(
    *,
    current_phase: str,
    phase_started_round: int,
    observations: Sequence[YieldObservation],
    calibration: YieldCalibration,
) -> PhaseDecision:
    """§8.4.1's phase switch: marginal yield, never a fixed budget fraction.

    Order matters and is not arbitrary:

    1. **A deep lever landing or an anchor move enters HARVEST immediately.** A
       freshly opened region carries a cluster of adjacent wins that are cheap,
       high-probability and PERISHABLE — they rebase away at the next freeze — so
       the dwell that protects against thrash does not delay entry to the phase
       that strips them. This trigger is event-driven, not signal-driven, which is
       why it cannot oscillate.
    2. **Dwell and window before any conclusion.** A switch on less evidence than
       the window that defines it is the thrash the dwell exists to prevent.
    3. **PLANNER_DEGRADED before EXPLORE.** Falling yield WITH rising
       `PROPOSAL_SKIPPED` is a broken searcher, not a dead region (§8.10:
       conflating them once cost this project months of paid no-ops). Reading it
       as EXPLORE would send a broken planner off to do harder work.
    4. **EXPLORE only when the signal holds across the FULL window.**
    """
    if current_phase not in PHASES:
        raise ValueError(f"current_phase: {current_phase!r} not in {list(PHASES)}")
    if not isinstance(calibration, YieldCalibration):
        raise TypeError("calibration must be a YieldCalibration")
    verify = calibration.verify()
    if verify.outcome != schemas.PASS:
        raise CalibrationTampered(
            f"calibration {calibration.derivation_id!r} did not verify ({verify.outcome}): "
            + "; ".join(verify.reasons)
        )
    if not isinstance(phase_started_round, int) or isinstance(phase_started_round, bool):
        raise TypeError("phase_started_round must be an int")
    rows = list(observations)
    if not rows:
        raise ValueError(
            "observations: required and non-empty — a phase decision with no yield "
            "evidence is a guess"
        )
    for index, row in enumerate(rows):
        if not isinstance(row, YieldObservation):
            raise TypeError(f"observations[{index}] must be a YieldObservation")
        if index and row.round_index <= rows[index - 1].round_index:
            raise ValueError(
                "observations must be ordered by strictly increasing round_index"
            )

    latest = rows[-1]
    if latest.deep_lever_landed or latest.anchor_moved:
        trigger = TRIGGER_DEEP_LEVER if latest.deep_lever_landed else TRIGGER_ANCHOR_MOVE
        return PhaseDecision(
            phase=PHASE_HARVEST,
            changed=current_phase != PHASE_HARVEST,
            trigger=trigger,
            reason=(
                "a freshly opened region carries a cluster of adjacent wins that are "
                "cheap, high-probability and perishable; HARVEST strips them first (§8.4.1)"
            ),
            phase_started_round=latest.round_index,
            window=(latest,),
        )

    # Window BEFORE dwell, and both are checked. They ask different questions —
    # "is there enough evidence to conclude?" and "has this phase lasted long
    # enough to be allowed to end?" — and they are separately reachable: a phase
    # that started recently inside a long history satisfies the window and fails
    # the dwell.
    if len(rows) < calibration.window_rounds:
        return PhaseDecision(
            phase=current_phase, changed=False, trigger=TRIGGER_WINDOW,
            reason=(
                f"{len(rows)} observation(s) against a derived trailing window of "
                f"{calibration.window_rounds}; a switch on less evidence than the window "
                "that defines it is a switch on noise"
            ),
            phase_started_round=phase_started_round, window=tuple(rows),
        )
    in_phase = [r for r in rows if r.round_index >= phase_started_round]
    if len(in_phase) < calibration.min_dwell_rounds:
        return PhaseDecision(
            phase=current_phase, changed=False, trigger=TRIGGER_DWELL,
            reason=(
                f"{len(in_phase)} round(s) in {current_phase} against a derived minimum "
                f"dwell of {calibration.min_dwell_rounds}; a phase that can end on less "
                "evidence than the window able to end it thrashes"
            ),
            phase_started_round=phase_started_round, window=tuple(in_phase),
        )

    window = tuple(rows[-calibration.window_rounds:])
    yields = [row.marginal_yield for row in window]
    below_floor = all(y < calibration.floor for y in yields)

    skips = [row.proposal_skipped_count for row in window]
    fingerprints = [row.repeated_fingerprint_count for row in window]
    skips_rising = (
        all(b >= a for a, b in zip(skips, skips[1:])) and skips[-1] > skips[0]
    ) or any(fingerprints)
    # NOT `<`. A planner whose every proposal is filtered banks nothing, so its
    # marginal yield is a FLAT zero — the commonest shape of a broken searcher
    # produces no strict decrease at all, and a strict test would route exactly
    # that case to EXPLORE. §8.10's rule is that a searcher which is not
    # improving while its rejections climb is degraded, not that the yield must
    # have a downward slope.
    yield_not_improving = yields[-1] <= yields[0]

    if below_floor and skips_rising and yield_not_improving:
        receipt = [row.receipt for row in window]
        request = state_machine.StopRequest(
            state=state_machine.PLANNER_DEGRADED,
            reason=(
                "marginal yield decayed below the derived floor while PROPOSAL_SKIPPED "
                "was rising: the searcher is broken, not the region"
            ),
            detail={
                "signal": "falling_yield_with_rising_proposal_skipped",
                "receipt": receipt,
                "window_rounds": calibration.window_rounds,
                "floor": calibration.floor,
                "yields": yields,
                "proposal_skipped_counts": skips,
                "repeated_fingerprint_counts": fingerprints,
            },
            origin="controller",
        )
        return PhaseDecision(
            phase=current_phase, changed=False, trigger=TRIGGER_DEGRADED,
            reason=(
                "falling yield WITH rising PROPOSAL_SKIPPED is PLANNER_DEGRADED, not "
                "EXPLORE (§8.4.1, §8.10)"
            ),
            phase_started_round=phase_started_round, window=window, stop_request=request,
        )

    if below_floor and current_phase == PHASE_HARVEST:
        return PhaseDecision(
            phase=PHASE_EXPLORE, changed=True, trigger=TRIGGER_YIELD_DECAY,
            reason=(
                f"marginal yield was below the derived floor {calibration.floor:.6g} in "
                f"every round of the {calibration.window_rounds}-round trailing window; "
                "the region yields less than it ever did while harvesting"
            ),
            phase_started_round=latest.round_index, window=window,
        )
    return PhaseDecision(
        phase=current_phase, changed=False,
        trigger=TRIGGER_YIELD_HOLDING if not below_floor else TRIGGER_YIELD_DECAY,
        reason=(
            "the switch signal did not hold across the full window"
            if not below_floor else
            "already in EXPLORE; only a deep lever landing or an anchor move re-enters "
            "HARVEST"
        ),
        phase_started_round=phase_started_round, window=window,
    )


# =============================================================================
# §8.4 ranking and §8.4.1 reserved-budget arms
# =============================================================================

ARM_INCREMENTAL = "incremental"
ARM_ARCHITECTURAL = "architectural"


@dataclass(frozen=True)
class ArmBudget:
    """The two arms. `architectural` is RESERVED: an incremental proposal may not
    draw from it even when the general arm is empty (§8.4.1, AK-D31 — *"EIG-first
    ranking starves high-variance work by arithmetic"*)."""

    incremental_minutes: float
    architectural_minutes: float

    def __post_init__(self) -> None:
        # `select_next` compares a cost against these with `<=`; a NaN arm would
        # refuse everything and a negative one is not a budget. Both are silent.
        for name in ("incremental_minutes", "architectural_minutes"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) \
                    or not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name}: required, a finite non-negative number")

    def remaining(self, arm: str) -> float:
        if arm == ARM_INCREMENTAL:
            return self.incremental_minutes
        if arm == ARM_ARCHITECTURAL:
            return self.architectural_minutes
        raise ValueError(f"arm: {arm!r} not in {[ARM_INCREMENTAL, ARM_ARCHITECTURAL]}")


def partition_budget(
    total_minutes: float, architectural: Optional[ArchitecturalCampaign]
) -> ArmBudget:
    """Split the remaining budget into the general arm and the reserved arm.

    With no declared architectural campaign the reserve is zero — there is nothing
    to reserve it for, and reserving budget for a lineage nobody declared would
    silently shrink the general arm.
    """
    if not isinstance(total_minutes, (int, float)) or isinstance(total_minutes, bool):
        raise TypeError("total_minutes must be a number")
    total = float(total_minutes)
    if not math.isfinite(total) or total < 0.0:
        raise ValueError("total_minutes must be finite and non-negative")
    if architectural is None:
        return ArmBudget(incremental_minutes=total, architectural_minutes=0.0)
    reserved = total * float(architectural.reserved_budget_fraction)
    return ArmBudget(incremental_minutes=total - reserved, architectural_minutes=reserved)


@dataclass(frozen=True)
class RankedProposal:
    order: int
    proposal_id: str
    fingerprint: str
    arm: str
    priority_class: int
    information_gain: float
    performance_value: float
    tier_cost_rank: int


def rank_proposals(results: Sequence[ScreenResult], *, phase: str) -> tuple:
    """§8.4: expected information gain FIRST, then expected performance value.

    Two rules apply at once and neither is discarded. §8.4 orders WITHIN a
    priority class; §8.4.1's phase policy sets the classes — *"Incremental
    proposals take priority and the cheap tiers dominate"* in HARVEST,
    *"Architectural proposals and spikes take priority"* in EXPLORE. Ties break on
    the fingerprint, so two readers of one journal produce one order.

    Only ADMITTED results rank. A rejected proposal has no rank at all, which is
    the search-side shape of the correctness-precedence rule: a penalised rank is
    still a rank.
    """
    if phase not in PHASES:
        raise ValueError(f"phase: {phase!r} not in {list(PHASES)}")
    admitted = []
    for result in results:
        if not isinstance(result, ScreenResult):
            raise TypeError("results must all be ScreenResult")
        if result.admitted:
            admitted.append(result)

    def priority(result: ScreenResult) -> int:
        if phase == PHASE_HARVEST:
            base = 0 if result.arm == ARM_INCREMENTAL else 2
            return base + (1 if result.tier_cost_rank else 0)
        base = 0 if result.arm == ARM_ARCHITECTURAL else 2
        return base + (1 if result.tier_cost_rank else 0)

    ordered = sorted(
        admitted,
        key=lambda r: (
            priority(r), -r.information_gain, -r.performance_value, r.fingerprint
        ),
    )
    return tuple(
        RankedProposal(
            order=index,
            proposal_id=result.proposal_id,
            fingerprint=result.fingerprint,
            arm=result.arm,
            priority_class=priority(result),
            information_gain=result.information_gain,
            performance_value=result.performance_value,
            tier_cost_rank=result.tier_cost_rank,
        )
        for index, result in enumerate(ordered)
    )


@dataclass(frozen=True)
class SelectionDecision:
    """What SELECT_TARGET hands to PROPOSE. Deterministic and fully attributed."""

    chosen: Optional[RankedProposal]
    arm: Optional[str]
    reason: str
    ranked: tuple
    arm_budget: ArmBudget

    def transition_detail(self) -> dict:
        """The `detail` mapping for the machine's SELECT_TARGET -> PROPOSE
        transition. Built here so the record says why THIS proposal, not merely
        that one was chosen."""
        return {
            "chosen_proposal_id": None if self.chosen is None else self.chosen.proposal_id,
            "chosen_fingerprint": None if self.chosen is None else self.chosen.fingerprint,
            "arm": self.arm,
            "reason": self.reason,
            "ranked": [
                {"order": r.order, "proposal_id": r.proposal_id, "arm": r.arm,
                 "priority_class": r.priority_class,
                 "information_gain": r.information_gain,
                 "performance_value": r.performance_value}
                for r in self.ranked
            ],
            "arm_budget": {
                "incremental_minutes": self.arm_budget.incremental_minutes,
                "architectural_minutes": self.arm_budget.architectural_minutes,
            },
        }


def select_next(
    results: Sequence[ScreenResult],
    *,
    phase: str,
    arm_budget: ArmBudget,
    cost_minutes_by_proposal: Mapping[str, float],
) -> SelectionDecision:
    """Choose the next target, honouring the reserved arm (§8.4.1, AK-D31).

    The reserve is what makes it a reserve: an incremental proposal is refused
    when the general arm cannot pay for it, even if the architectural arm is full,
    and an architectural proposal remains selectable out of its own arm even when
    it ranks last globally. Without that, EIG-first ranking starves high-variance
    work by arithmetic, every round, forever.
    """
    if not isinstance(arm_budget, ArmBudget):
        raise TypeError("arm_budget must be an ArmBudget")
    if not isinstance(cost_minutes_by_proposal, Mapping):
        raise TypeError("cost_minutes_by_proposal must be a mapping")
    ranked = rank_proposals(results, phase=phase)
    for candidate in ranked:
        cost = cost_minutes_by_proposal.get(candidate.proposal_id)
        if cost is None:
            raise ValueError(
                f"cost_minutes_by_proposal is missing {candidate.proposal_id!r}; a "
                "selection that cannot price its candidate cannot honour an arm budget"
            )
        if float(cost) <= arm_budget.remaining(candidate.arm):
            return SelectionDecision(
                chosen=candidate, arm=candidate.arm,
                reason=(
                    f"rank {candidate.order} in the {phase} ordering; {candidate.arm} arm "
                    f"has {arm_budget.remaining(candidate.arm)} minutes and this costs "
                    f"{float(cost)}"
                ),
                ranked=ranked, arm_budget=arm_budget,
            )
    return SelectionDecision(
        chosen=None, arm=None,
        reason=(
            "no admitted proposal fits its own arm's remaining budget; the reserved "
            "architectural arm is not available to incremental work (AK-D31)"
            if ranked else
            "no proposal survived the §8.4 rejection conditions this round"
        ),
        ranked=ranked, arm_budget=arm_budget,
    )
