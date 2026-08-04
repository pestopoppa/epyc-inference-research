#!/usr/bin/env python3
"""readiness.py — the T2 lineage estimator and the advisory readiness signal.

WHY THIS MODULE EXISTS
----------------------
Three specific failures are what this file is shaped around. Each one is a thing
this project has already paid for, or a thing the constitution forbids twice over.

  * **A readiness figure that came out of a sentence.** §4 invariant 14: *"No
    estimated percentage by narration. Readiness is computed from records by a
    deterministic controller; the LLM may request, never declare."*
    `P-AK-SEARCH-1` authorization 5 says the same in the protocol's own voice — a
    readiness figure *"that originates in controller narrative rather than in
    records is `INVALID`"*. So every number this module reports is carried out of
    exactly one `api.EffectEstimate` that a reducer produced, together with the
    event id it came from. There is no code path here that computes a value no
    cell measured, because there is no arithmetic here at all
    (`audit_no_weighting_or_averaging()` proves that from this module's own AST).

  * **A cross-device composite.** AK-D12. The original production-weighted
    scalar across CPU, GPU, STT and TTS cells is withdrawn, and it was forbidden
    twice: `MEASUREMENT.md:83-84` makes a fold across P-BENCH-1, P-BENCH-PREFILL-1
    and P-GPU-1 cells *analysis, not a claim*, and `gpu-cross-device.md:106-111`
    forbids a reconstructed net outright — *"Measuring GPU gain and CPU loss
    separately and subtracting is FORBIDDEN."* The structural answer is that
    `compute_readiness()` takes **one backend** and refuses a cell belonging to
    any other. A composite is therefore not merely unwritten, it is unreachable:
    no function in this module ever sees two backends' measurements at once.
    `composite_readiness()` exists solely to be found by anyone reaching for one,
    and it raises.

  * **A signal that became a trigger.** AK-D3 demoted the `+25% point / +20%
    lower-bound` figure from an automatic release trigger to a readiness signal
    the loop REPORTS. `P-AK-SEARCH-1` denial 5 closes it from the other side: *"a
    readiness signal is not a freeze trigger."* So `ReadinessSignal.is_trigger`
    is `False`, `freeze_eligibility()` raises, and a **phase trade** — a small
    prefill regression buying a large decode gain — never resolves to
    `objective_met` here. It resolves to `operator_decision_required`, because
    §1.6 makes it *"an operator decision at freeze time, not a controller
    decision."*

WHAT IT COMPUTES
----------------
`§1.6`, the whole objective, per backend and per phase:

> At the **production-optimal** recipe for every protected cell, both **prefill**
> and **decode** throughput must be non-inferior to the production anchor, and at
> least one must improve.

Each phase is judged under its own protocol — P-BENCH-1 for decode,
P-BENCH-PREFILL-1 for prefill, P-GPU-1 for MI210 — so nothing crosses a protocol
boundary: `ObjectiveSpec.protocol_by_phase` binds the protocol to the phase, and a
cell citing a different protocol than its phase declares is refused
(`ProtocolBoundaryCrossed`).

`§9.7`, the T2 matrix, is checked rather than assumed:

  * runs on the **composed champion**, never by adding local percentages — every
    cell must name the combined candidate, and a member candidate's cell raises;
  * one or a few roles per affected architecture/regime (`COVERAGE_GAP`);
  * **stronger paired repetitions than T1** (`REPETITIONS_NOT_STRONGER_THAN_T1`);
  * broader dispatcher-boundary and non-target sentinels — a strict superset of
    T1's set (`SENTINEL_SET_NOT_BROADER`), and a regressing non-target sentinel
    blocks (`NON_TARGET_REGRESSION`);
  * **at least one co-resident PROTECTED cell for `llama_cpu`**
    (`CO_RESIDENT_CELL_ABSENT`). Production runs concurrent instances and CPU
    decode is bandwidth-bound **for a role the objective protects**, so a change
    can be neutral alone and harmful co-resident. This one is not negotiable by a
    caller and not closable by a sentinel: `CO_RESIDENT_REQUIRED_BACKENDS` is read
    by `check_matrix_coverage()` and a spec cannot switch it off, and a
    co-resident sentinel — a probe on a path nobody is protecting — does not
    discharge it;
  * capacity deltas — VRAM / RAM / context (`CAPACITY_DELTA_ABSENT`,
    `CAPACITY_REGRESSION`), whose required kinds the backend adapter must
    DECLARE; an undeclared requirement is `COULD_NOT_CHECK`, never a satisfied
    one;
  * cumulative mechanism confirmation (`MECHANISM_UNCONFIRMED`), per
    `P-AK-SEARCH-1-A1` clause 1: *"It got faster and I don't know why is a reason
    to keep measuring, not to land."*

Evidence admissibility follows the protocol's own strata clause: **confirmation
stratum only**, gathered **after** the candidate entered the lineage — cells,
capacity deltas and mechanism confirmations alike, since all three are read here
as statements about the composed champion. A selection-stratum cell raises,
because *"the evidence that promotes a candidate is structurally unfit to report
how ready it is."*

Every §9.7 requirement above counts only cells whose own verdict admits them to a
rank (`_rank_admissible`). A cell that failed a prior gate, was voided, never
resolved, carries no rate comparison or binds no anchor covers no architecture,
discharges no co-residency requirement and evidences no repetition strength:
*"a candidate failing any of them receives no speed rank at all — not a penalised
one"*, and each of those requirements asserts that the matrix learned something at
that cell.

THE PHASE FIGURE HAS THREE STATES AND NONE IS EXPRESSIBLE AS ANOTHER
--------------------------------------------------------------------
`_phase_figure` returns `None`, a `ParityFigure`, or a `ReadinessFigure`:

  * **nothing measured** — `None`. No admissible cell produced an effect at all.
  * **all at parity** — a `ParityFigure`, carrying the COUNT and the SENSITIVITY
    and no orderable value. §1.6's first half is NON-INFERIORITY, so a backend
    that genuinely did not regress produces cells at `no_detectable_difference`,
    and that is the most common HEALTHY outcome. Reporting it as `None` would
    render success as an absence; absences read as coverage gaps, and a coverage
    gap is what a later session closes by loosening the gate.
  * **orderable** — a `ReadinessFigure`, selected over the ORDERABLE cells only.

The exclusion predicate is the evaluator's own (`api.is_sub_floor_resolution`,
beside `_RANKABLE_RESOLUTIONS`), not a second copy: the evaluator already
withholds a speed rank from a sub-floor cell, and selecting one as "the weakest"
or "the best" IS a rank. In the MIXED case the figure discloses how many
protected cells it excluded and on which resolutions — an undisclosed exclusion
is how a figure becomes a lie without anyone writing one.

THE THREE OUTCOMES, AND WHY `FAIL` AND `COULD_NOT_CHECK` DIFFER ONLY IN THE REASON
---------------------------------------------------------------------------------
Every per-cell judgement is a `schemas.Check`, so *"inability to evaluate"* is a
third outcome and never a soft pass. For non-inferiority:

  * `PASS`   — the non-inferiority e-process crossed its calibrated threshold.
  * `FAIL`   — the e-process did not cross **and** the cell's own published MDE
               says the negative effect is detectable. This is reported as *"a
               detectable degradation with no non-inferiority evidence"*, which
               is a reason to withhold readiness. It is deliberately NOT dressed
               up as a test of inferiority: no such e-process was run, and this
               module does not invent one.
  * `COULD_NOT_CHECK` — anything else: below the floor, below the MDE (*"no
               detectable difference … is a result and a decision"*), voided,
               inconclusive, or simply not measured.

`FAIL` and `COULD_NOT_CHECK` have **identical** consequences for the objective —
both withhold it. They differ only in what the operator is told. That is what
keeps the `FAIL` branch from being a smuggled statistical construction: no
decision rides on the distinction.

WHERE THE REDUCTION HAPPENS
---------------------------
Not here. `evaluator/statistics.py` is the one reducer: it produces the paired
blocks, the e-process, the MDE at the realized block count, the calibrated floor
and the estimate, and `evaluator/api.py` computes the verdict from them. This
module reads `stats.EProcessRun` for the e-process identity (which hypothesis,
which margin, whether it crossed) and `api.Verdict` for correctness precedence,
and cross-checks that the two describe the same run. Writing a second reducer
here would produce a number whose provenance nobody could reconstruct, which is
the defect `MEASUREMENT.md:9-11` exists to prevent.

WHAT THIS MODULE IS NOT
-----------------------
It runs no inference, no benchmark and no build. It starts, stops and signals no
process. It writes no file. It performs no multiplication, division or
exponentiation of any kind, so a weighted scalar folding two cells — let alone
two backends — cannot be expressed in it. All four are proved from the module's
own AST by `audit_no_write_or_process_paths()` and
`audit_no_weighting_or_averaging()`, and asserted in `test_readiness.py`.

It also holds no release authority. T3 is a release gate and is not run here;
`freeze_eligibility()` exists only to raise.

Governing instruments: `epyc-root/measurement/protocols/kernel-research.md`
(Annex K, **P-AK-SEARCH-1**, RATIFIED 2026-08-03) — authorization 5 (advisory
readiness signal, *"computed by a deterministic reducer over journaled
records"*), denial 5 (*"a readiness signal is not a freeze trigger"*), denial 9
(no new instrument by composition), the selection/confirmation split, the
controls clause and its `HISTORICAL_REPLAY_UNAVAILABLE` marker — and
`P-AK-SEARCH-1-A1` clause 1 (mechanism plausibility).

Owning design: `epyc-root/handoffs/active/autokernel-research-loop.md` §1.2,
§1.6, §9.6, §9.7, §9.8, §4 invariants 14 and 15, §17 (AK-D3, AK-D4, AK-D12,
AK-D22), phase AK5.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Mapping, Optional, Sequence

from .. import schemas
from ..evaluator import api
from ..evaluator import statistics as stats

__all__ = [
    # identity
    "MODULE_ID", "TIER", "SIGNAL_CLASS", "OBJECTIVE_RULE",
    # errors
    "ReadinessError", "CrossBackendComposite", "ChampionMismatch", "StratumViolation",
    "ProtocolBoundaryCrossed", "CellInadmissible", "MatrixSpecInvalid",
    "TriggerAuthorityError", "CapabilityObjectiveInvalid", "StandingNotDerived",
    "CampaignMismatch", "ParityHasNoOrderableValue",
    # vocabularies
    "CELL_ROLES", "CELL_ROLE_PROTECTED", "CELL_ROLE_DISPATCHER_BOUNDARY",
    "CELL_ROLE_NON_TARGET", "STANDINGS", "STANDING_MET", "STANDING_NOT_MET",
    "STANDING_UNDETERMINED", "BLOCKERS", "CAPACITY_KINDS", "CONTROLS_MARKERS",
    "CONTROLS_COMPLETE", "CONTROLS_REPLAY_UNAVAILABLE", "IMPROVEMENT_QUANTIFIERS",
    "QUANTIFIER_BACKEND_WIDE", "QUANTIFIER_PER_PROTECTED_CELL",
    "T2_TRIGGER_CONDITIONS", "TRIGGER_OUTCOMES", "TRIGGER_RUN_T2", "TRIGGER_HOLD",
    "TRIGGER_COULD_NOT_EVALUATE", "CO_RESIDENT_REQUIRED_BACKENDS",
    "STACK_CHANGE_BACKEND",
    # inputs
    "ChampionLineage", "ObjectiveSpec", "PhaseTradeException", "T2MatrixSpec",
    "PhaseEvidence", "T2Cell", "CapacityDelta", "MechanismConfirmation",
    "CapabilityObjective", "ReferencePolicy",
    # outputs
    "CellStanding", "PhaseStanding", "MatrixCoverage", "PhaseTradeAssessment",
    "ReferenceComparison", "CapabilityStanding", "ReadinessFigure", "ParityFigure",
    "PHASE_FIGURE_TYPES", "ReadinessSignal",
    "ReadinessReport", "CrossBackendAnalysisView", "TriggerDecision",
    # functions
    "evaluate_t2_trigger", "check_matrix_coverage", "cell_standing", "phase_standing",
    "compute_readiness", "compute_readiness_report", "cross_backend_analysis_view",
    "composite_readiness", "freeze_eligibility", "render_readiness_line",
    "audit_no_write_or_process_paths", "audit_no_weighting_or_averaging",
]

# =============================================================================
# Identity
# =============================================================================

#: Versioned, because the readiness signal names the reducer that produced it and
#: a signal computed by a different estimator is a different signal.
MODULE_ID = "autokernel.release.readiness/v1"

#: The only tier whose records this module reads. T2 is a SEARCH tier under
#: `P-AK-SEARCH-1` (authorization 5 is what licenses a readiness computation at
#: all); T3 is a release gate and is not run, read, or implied here.
TIER = "T2"

#: What the signal IS, carried in every rendered line. Three separate denials in
#: one string, because each has been assumed away at least once in this project's
#: history: it is not a claim (`MEASUREMENT.md:9-11`), it is not a trigger
#: (AK-D3, denial 5), and it is advisory (§1.2).
SIGNAL_CLASS = "ADVISORY READINESS SIGNAL — NOT A CLAIM, NOT A TRIGGER"

#: The one objective rule §1.6 names. `schemas.OBJECTIVE_RULES` is the vocabulary;
#: this is the member this module implements, and adding another is a schema
#: version event rather than a branch here.
OBJECTIVE_RULE = "per_phase_non_inferiority_plus_improvement"


# =============================================================================
# Errors — every one is a refusal about MATERIAL, never a finding about a
# candidate. A finding about a candidate comes back as a Check.
# =============================================================================

class ReadinessError(Exception):
    """Base for every refusal in this module."""


class CrossBackendComposite(ReadinessError):
    """Someone tried to fold two backends into one number (AK-D12).

    Forbidden twice over: `MEASUREMENT.md:83-84` makes a cross-protocol fold
    ANALYSIS rather than a claim, and `gpu-cross-device.md:106-111` forbids a
    reconstructed net outright because it *"compounds both halves' noise … and
    measures the halves under conditions that do not co-occur."*
    """


class ChampionMismatch(ReadinessError):
    """A cell that is not the composed champion's own evidence (§9.7).

    *"Runs on the composed champion, never by adding local percentages."* A member
    candidate's cell in a champion matrix is the exact defect that sentence
    exists to prevent, so it is unusable material and refused at construction
    rather than quietly averaged in.
    """


class CampaignMismatch(ReadinessError):
    """A report labelled for one campaign carrying another campaign's signal.

    `P-AK-SEARCH-1` denial 4 confines consumption *"to the AutoKernel controller
    that produced the record, within the campaign that produced it"*, and a later
    campaign may use a prior record *"for hypothesis formation only — never to
    rank, bank, compose, or contribute to readiness"*, because a later campaign
    re-derives its own calibration and a reused record would be scored against a
    floor and a threshold it was never measured under.

    `ReadinessReport` emits ONE `campaign_id` over whatever signals it was handed,
    so an unchecked report is exactly the route by which a campaign-B signal
    acquires a campaign-A label — the reuse the denial forbids, performed by the
    reducer rather than by a person.
    """


class StratumViolation(ReadinessError):
    """Selection-stratum evidence offered to a readiness computation.

    *"The readiness signal is computed ONLY from confirmation-stratum evidence
    gathered after the candidate entered the lineage."* Selecting the maximum
    over many candidates biases the selected estimate upward, so the evidence
    that promotes a candidate is structurally unfit to report how ready it is.
    """


class ProtocolBoundaryCrossed(ReadinessError):
    """A cell judged under a protocol its phase does not declare (§1.6).

    *"Each phase is judged under its own protocol … so nothing crosses a protocol
    boundary."*
    """


class CellInadmissible(ReadinessError):
    """The cell's own material is malformed — not a verdict about the candidate."""


class MatrixSpecInvalid(ReadinessError):
    """The declared T2 matrix cannot be checked against, so nothing is checked."""


class TriggerAuthorityError(ReadinessError):
    """A release decision was requested from a signal that reports (AK-D3).

    `P-AK-SEARCH-1` denial 5: *"a readiness signal is not a freeze trigger."*
    Denial 7: no release activity, no freeze eligibility, no waiver judgement.
    """


class CapabilityObjectiveInvalid(ReadinessError):
    """A capability objective whose utility model was not fixed at campaign start.

    §9.8: *"the utility model was fixed at campaign start, not invented after
    observing the candidate."*
    """


class StandingNotDerived(ReadinessError):
    """A `ReadinessSignal` whose standing does not follow from its own evidence.

    §4 invariant 14: *"Readiness is computed from records by a deterministic
    controller; the LLM may request, never declare."* Every OTHER guarantee in
    this module is structural, but the standing itself was, until this error
    existed, a field a caller could simply set — `dataclasses.replace(signal,
    standing='objective_met', blockers=())` produced an object that rendered as
    met while its own phase standings said otherwise. `api.Verdict` solved the
    identical problem by re-deriving its status in `__post_init__` and raising
    `VerdictTampering`; this is the same lock on the same shape of hole.
    """


class ParityHasNoOrderableValue(ReadinessError):
    """Something asked a `ParityFigure` for a magnitude it does not have.

    A parity phase measured every protected cell and every estimate landed inside
    the campaign's own floor or MDE. There is no number to hand back, and the two
    ways of pretending otherwise are both worse than raising: returning `0.0`
    invents a measurement (sub-floor does not mean zero — it means the sign and
    the size are both unknown), and returning `None` turns the RESULT back into
    the ABSENCE that `ParityFigure` exists to stop it being read as.

    It is a raise rather than a missing attribute because `getattr(figure,
    "value", None)` is exactly how a caller silently reintroduces the `None`.
    """


# =============================================================================
# Controlled vocabularies
# =============================================================================

#: A cell the objective protects. §1.6 quantifies over exactly these.
CELL_ROLE_PROTECTED = "protected"
#: A sentinel around the dispatcher boundary (§9.7, §9.5 "mandatory sentinel").
CELL_ROLE_DISPATCHER_BOUNDARY = "dispatcher_boundary_sentinel"
#: A sentinel on a path the change was not supposed to touch (§9.7).
CELL_ROLE_NON_TARGET = "non_target_sentinel"

CELL_ROLES = (CELL_ROLE_PROTECTED, CELL_ROLE_DISPATCHER_BOUNDARY, CELL_ROLE_NON_TARGET)

#: Sentinel roles. They never contribute an improvement — a sentinel that got
#: faster is not why a lineage ships — but a regressing one blocks.
SENTINEL_ROLES = (CELL_ROLE_DISPATCHER_BOUNDARY, CELL_ROLE_NON_TARGET)

STANDING_MET = "objective_met"
STANDING_NOT_MET = "objective_not_met"
STANDING_UNDETERMINED = "undetermined"
STANDINGS = (STANDING_MET, STANDING_NOT_MET, STANDING_UNDETERMINED)

# --- blocking conditions ----------------------------------------------------
# Named, because §11 and the AK6 dashboard contract want "open blocking
# conditions" as a list an operator can read, not a boolean.

BLOCK_CELL_INVALID = "CELL_INVALID"
BLOCK_CELL_FAILED_PRIOR_GATE = "CELL_FAILED_PRIOR_GATE"
BLOCK_CELL_INCONCLUSIVE = "CELL_INCONCLUSIVE"
BLOCK_CELL_NOT_A_RATE_COMPARISON = "CELL_NOT_A_RATE_COMPARISON"
BLOCK_ANCHOR_ABSENT = "ANCHOR_ABSENT"
BLOCK_ANCHOR_MOVED = "ANCHOR_MOVED"
BLOCK_NON_INFERIORITY_EVIDENCE_ABSENT = "NON_INFERIORITY_EVIDENCE_ABSENT"
BLOCK_DETECTABLE_DEGRADATION = "DETECTABLE_DEGRADATION"
BLOCK_IMPROVEMENT_EVIDENCE_ABSENT = "IMPROVEMENT_EVIDENCE_ABSENT"
BLOCK_PHASE_NOT_MEASURED = "PHASE_NOT_MEASURED"
BLOCK_PROTECTED_CELL_ABSENT = "PROTECTED_CELL_ABSENT"
BLOCK_COVERAGE_GAP = "COVERAGE_GAP"
BLOCK_CO_RESIDENT_CELL_ABSENT = "CO_RESIDENT_CELL_ABSENT"
BLOCK_REPETITIONS_NOT_STRONGER_THAN_T1 = "REPETITIONS_NOT_STRONGER_THAN_T1"
BLOCK_SENTINEL_SET_NOT_BROADER = "SENTINEL_SET_NOT_BROADER"
BLOCK_NON_TARGET_REGRESSION = "NON_TARGET_REGRESSION"
BLOCK_CAPACITY_REQUIREMENT_UNDECLARED = "CAPACITY_REQUIREMENT_UNDECLARED"
BLOCK_CAPACITY_DELTA_ABSENT = "CAPACITY_DELTA_ABSENT"
BLOCK_CAPACITY_REGRESSION = "CAPACITY_REGRESSION"
BLOCK_MECHANISM_UNCONFIRMED = "MECHANISM_UNCONFIRMED"
BLOCK_CONFIRMATION_EVIDENCE_PREDATES_LINEAGE = "CONFIRMATION_EVIDENCE_PREDATES_LINEAGE"
BLOCK_PHASE_TRADE_DECISION_REQUIRED = "PHASE_TRADE_DECISION_REQUIRED"
BLOCK_CAPABILITY_UTILITY_MODEL_DRIFTED = "CAPABILITY_UTILITY_MODEL_DRIFTED"

BLOCKERS = (
    BLOCK_CELL_INVALID, BLOCK_CELL_FAILED_PRIOR_GATE, BLOCK_CELL_INCONCLUSIVE,
    BLOCK_CELL_NOT_A_RATE_COMPARISON, BLOCK_ANCHOR_ABSENT, BLOCK_ANCHOR_MOVED,
    BLOCK_NON_INFERIORITY_EVIDENCE_ABSENT, BLOCK_DETECTABLE_DEGRADATION,
    BLOCK_IMPROVEMENT_EVIDENCE_ABSENT, BLOCK_PHASE_NOT_MEASURED,
    BLOCK_PROTECTED_CELL_ABSENT, BLOCK_COVERAGE_GAP, BLOCK_CO_RESIDENT_CELL_ABSENT,
    BLOCK_REPETITIONS_NOT_STRONGER_THAN_T1, BLOCK_SENTINEL_SET_NOT_BROADER,
    BLOCK_NON_TARGET_REGRESSION, BLOCK_CAPACITY_REQUIREMENT_UNDECLARED,
    BLOCK_CAPACITY_DELTA_ABSENT, BLOCK_CAPACITY_REGRESSION,
    BLOCK_MECHANISM_UNCONFIRMED, BLOCK_CONFIRMATION_EVIDENCE_PREDATES_LINEAGE,
    BLOCK_PHASE_TRADE_DECISION_REQUIRED, BLOCK_CAPABILITY_UTILITY_MODEL_DRIFTED,
)

#: §9.7 "capacity (VRAM/RAM/context) deltas". All three are higher-better: more
#: free VRAM, more free RAM, more context. A candidate that costs capacity is a
#: candidate that may not fit the lineup at all, which is why capacity is a
#: banked axis in its own right (§9.6) rather than a footnote to throughput.
CAPACITY_VRAM = "vram_bytes_free"
CAPACITY_RAM = "ram_bytes_free"
CAPACITY_CONTEXT = "context_tokens"
CAPACITY_KINDS = (CAPACITY_VRAM, CAPACITY_RAM, CAPACITY_CONTEXT)

#: The controls marker the protocol requires a readiness signal to carry: *"the
#: readiness signal computed by such a campaign carries the same marker."*
CONTROLS_COMPLETE = "5/5"
CONTROLS_REPLAY_UNAVAILABLE = "4/5 (HISTORICAL_REPLAY_UNAVAILABLE)"
CONTROLS_MARKERS = (CONTROLS_COMPLETE, CONTROLS_REPLAY_UNAVAILABLE)

#: §9.7: *"at least one co-resident cell for `llama_cpu`"*. This is a property of
#: the backend, not of a campaign's taste, so it lives here and a `T2MatrixSpec`
#: can only ADD to it. Production runs concurrent instances and CPU decode is
#: bandwidth-bound, so a change can be neutral alone and harmful co-resident.
CO_RESIDENT_REQUIRED_BACKENDS = frozenset({"llama_cpu"})

# --- the improvement quantifier ---------------------------------------------
# §1.6 reads: "at the production-optimal recipe for every protected cell, both
# prefill and decode throughput must be non-inferior to the production anchor,
# and at least one must improve." The "for every protected cell" quantifier
# unambiguously scopes the NON-INFERIORITY half. Which quantifier scopes the
# IMPROVEMENT half is genuinely ambiguous in the sentence, and the two readings
# are very far apart:
#
#   * backend-wide      — some protected cell improves in some phase;
#   * per protected cell — every protected cell improves in at least one phase.
#
# The second would require a change targeting one architecture to also speed up
# every other protected role, which is a much stricter release bar. Guessing
# either way silently is exactly the class of defect this package exists to stop,
# so the campaign DECLARES which one it means and both are computed and reported.
QUANTIFIER_BACKEND_WIDE = "backend_wide"
QUANTIFIER_PER_PROTECTED_CELL = "per_protected_cell"
IMPROVEMENT_QUANTIFIERS = (QUANTIFIER_BACKEND_WIDE, QUANTIFIER_PER_PROTECTED_CELL)

# --- T2 trigger -------------------------------------------------------------
# §9.7: "Trigger when compatible winners have accumulated and interaction is the
# dominant uncertainty, when the readiness signal could plausibly change
# materially, or when a predeclared capability objective becomes runnable."
TRIGGER_WINNERS_ACCUMULATED = "compatible_winners_accumulated_interaction_dominant"
TRIGGER_READINESS_COULD_CHANGE = "readiness_could_change_materially"
TRIGGER_CAPABILITY_RUNNABLE = "capability_objective_runnable"
T2_TRIGGER_CONDITIONS = (
    TRIGGER_WINNERS_ACCUMULATED, TRIGGER_READINESS_COULD_CHANGE,
    TRIGGER_CAPABILITY_RUNNABLE,
)

TRIGGER_RUN_T2 = "run_t2"
TRIGGER_HOLD = "hold"
TRIGGER_COULD_NOT_EVALUATE = "could_not_evaluate"
TRIGGER_OUTCOMES = (TRIGGER_RUN_T2, TRIGGER_HOLD, TRIGGER_COULD_NOT_EVALUATE)


# =============================================================================
# Small validation helpers. None of them computes anything.
# =============================================================================

def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CellInadmissible(f"{label}: expected a non-empty string, got {value!r}")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CellInadmissible(f"{label}: expected a finite number, got {value!r}")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise CellInadmissible(f"{label}: expected a finite number, got {value!r}")
    return number


def _instant(value: Any, label: str) -> datetime:
    """An ISO-8601 timestamp WITH an offset, as an ordered instant.

    A naive timestamp on a shared host is ambiguous across sessions, and *"after
    the candidate entered the lineage"* is an ordering question — so a timestamp
    that cannot be ordered is refused rather than assumed to be UTC.
    """
    _text(value, label)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise CellInadmissible(f"{label}: {value!r} is not ISO-8601 ({exc})") from exc
    if parsed.tzinfo is None:
        raise CellInadmissible(
            f"{label}: {value!r} carries no timezone offset; 'gathered after the "
            "candidate entered the lineage' is an ordering question and a naive "
            "timestamp on a shared host cannot be ordered")
    return parsed


#: `serving_runtime` is in `schemas.BACKENDS` but does not release this way. It has
#: no source tree (absent from `SOURCE_TREE_BY_BACKEND`, so the champion-lineage
#: check silently skips) and no phase vocabulary (absent from `PHASES_BY_BACKEND`, so
#: the §1.6 conjunction check silently passes over any phase string). Both of this
#: module's structural guarantees therefore *degrade to nothing* on it rather than
#: refusing it — while still emitting a `standing: objective_met` line that reads
#: exactly like a kernel backend's. `plan.py`, `t3.py` and `packager.py` each refuse
#: it by name at their own door; readiness is the operator-facing one, and the
#: readiness line is what a freeze request cites (§11.1). Its path is §11.6.
STACK_CHANGE_BACKEND = "serving_runtime"


def _backend(value: Any, label: str) -> str:
    _text(value, label)
    if value not in schemas.BACKENDS:
        raise CellInadmissible(
            f"{label}: {value!r} is not one of {sorted(schemas.BACKENDS)}")
    if value == STACK_CHANGE_BACKEND:
        raise CellInadmissible(
            f"{label}: {STACK_CHANGE_BACKEND!r} has no source tree and no §1.6 phase "
            "vocabulary, so every structural guarantee in this module degrades to a "
            "no-op on it while the signal still renders as a kernel backend's. It "
            "travels the §11.6 three-gate stack-change path, which produces no kernel "
            "era and no readiness signal of this class; `plan.py`, `t3.py` and "
            "`packager.py` refuse it at their own doors for the same reason (§13.5)."
        )
    return value


_HEX_LOWER = frozenset("0123456789abcdef")


def _sha256(value: Any, label: str) -> str:
    """A 64-hex lowercase digest that is not a known filler.

    Spelled out rather than borrowed from `schemas`' private regex: a module that
    reaches into another module's underscore names is one rename away from an
    import error, and the shape of a digest is not a secret.
    """
    _text(value, label)
    if len(value) != 64 or not set(value) <= _HEX_LOWER:
        raise CellInadmissible(f"{label}: {value!r} is not a 64-hex lowercase digest")
    if schemas.is_placeholder_digest(value):
        raise CellInadmissible(
            f"{label}: {value!r} is a placeholder digest. A fabricated hash is "
            "indistinguishable from a measured one to every downstream reader, which is "
            "strictly worse than an absent value")
    return value


def _co_residency(value: Any, label: str) -> str:
    """`single` or `co_resident:<lineup_id>` — the vocabulary `schemas` validates.

    §9.7 needs the distinction to be readable, because *"at least one co-resident
    cell"* is the whole point of the llama_cpu requirement.
    """
    _text(value, label)
    if value == "single":
        return value
    if value.startswith("co_resident:") and value[len("co_resident:"):].strip():
        return value
    raise CellInadmissible(
        f"{label}: {value!r} must be 'single' or 'co_resident:<lineup_id>'")


def _phase_for(backend: str, value: Any, label: str) -> str:
    _text(value, label)
    declared = schemas.PHASES_BY_BACKEND.get(backend)
    if declared is not None and value not in declared:
        raise CellInadmissible(
            f"{label}: {value!r} is not one of {sorted(declared)} for backend "
            f"{backend!r}")
    return value


def _tuple_of(values: Any, label: str, klass: type) -> tuple:
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise CellInadmissible(f"{label}: expected a sequence of "
                               f"{klass.__name__}, got {type(values).__name__}")
    out = tuple(values)
    for item in out:
        if not isinstance(item, klass):
            raise CellInadmissible(
                f"{label}: every item must be a {klass.__name__}, got "
                f"{type(item).__name__}")
    return out


def _combine(checks: Sequence[schemas.Check]) -> schemas.Check:
    """FAIL dominates COULD_NOT_CHECK dominates PASS.

    FAIL dominates because one phase with a detectable degradation denies the
    objective whatever the other phases could not tell us; COULD_NOT_CHECK
    dominates PASS because an unevaluated conjunct is not a satisfied one.
    """
    reasons: list = []
    outcome = schemas.PASS
    for chk in checks:
        if not isinstance(chk, schemas.Check):
            raise CellInadmissible("every combined item must be a schemas.Check")
        if chk.outcome == schemas.FAIL:
            outcome = schemas.FAIL
        elif chk.outcome == schemas.COULD_NOT_CHECK and outcome != schemas.FAIL:
            outcome = schemas.COULD_NOT_CHECK
        if chk.outcome != schemas.PASS:
            reasons.extend(chk.reasons)
    return schemas.Check(outcome, tuple(reasons))


def _short(value: str) -> str:
    """The grammar's own 12-hex abbreviation. Display only; never an identity."""
    return value[:12]


#: The reason AK-D12 refuses, carried into every one-backend refusal so the three
#: doors say the same thing for the same reason.
_ONE_BACKEND_WHY = (
    "Records are comparable only within one backend and one instrument version; a "
    "cross-backend roll-up is a labelled analysis view and never gates (Annex K, "
    "'Comparison scope'; AK-D12)")


def _describe_cell(cell: "T2Cell") -> str:
    return f"cell {cell.cell_id!r}"


def _describe_capacity_delta(delta: "CapacityDelta") -> str:
    return f"capacity delta {delta.kind!r} (event {delta.event_id})"


def _require_one_backend(backend: str, records: Sequence, describe) -> None:
    """AK-D12 at a door, over EVERY record that names a backend.

    `compute_readiness()` held this and `check_matrix_coverage()` and
    `phase_standing()` did not, so the module's central structural claim — *"no
    function in this module ever sees two backends' measurements at once"* — was
    true of one caller rather than of the module. Both of the other two are
    exported, both fold the records they are handed into a single per-backend
    statement, and `phase_standing()` will happily select a `llama_gpu` decode cell
    as a `llama_cpu` phase's readiness figure, which is the reconstructed net
    `gpu-cross-device.md:106-111` forbids outright.

    A cell is not the only record here that names a backend. `CapacityDelta` does
    too, and `_check_capacity` reads it through a `delta.backend == spec.backend`
    FILTER: a `llama_gpu` VRAM regression handed to a `llama_cpu` matrix was
    silently dropped, the axis it regressed on read PASS, and no blocker recorded
    that a regression had been offered and discarded. A filter is not a refusal —
    it is a refusal that reports success — and the same record was simultaneously
    *gated on* by `_check_lineage_ordering`, which orders every delta it is given.
    One record, read by one sibling and ignored by the other, is the composition
    defect rather than the guard defect.

    One function rather than four copies of the refusal: copies drift, and a door
    whose refusal drifted is a door that stops refusing. `describe` names the
    record in the refusal; it does not decide whether to refuse.
    """
    for record in records:
        if record.backend != backend:
            raise CrossBackendComposite(
                f"{describe(record)} belongs to backend {record.backend!r}, not "
                f"{backend!r}. " + _ONE_BACKEND_WHY)


def _require_declared_protocol(objective: "ObjectiveSpec", cells: Sequence) -> None:
    """§1.6's per-phase protocol at a door. ONE definition, both public entries.

    `compute_readiness()` held this inline and `phase_standing()` did not, so the
    protocol boundary was closed for one caller and open for the other — the same
    shape as the backend hole, in the same two functions, half of it left behind.
    `phase_standing()` stamps `objective.protocol_for(phase)` onto the
    `PhaseStanding` and onto the `ReadinessFigure` it selects, WITHOUT asking the
    cell what it was measured under: a `llama_cpu` decode cell citing
    `P-BENCH-PREFILL-1` was judged as decode and its estimate came back labelled
    `P-BENCH-1`. That is a cross-protocol comparison wearing a within-protocol
    label, which `MEASUREMENT.md:83-84` makes analysis rather than a claim, and the
    label is the part a reader cannot check for themselves.
    """
    for cell in cells:
        declared = objective.protocol_by_phase.get(cell.phase)
        if declared is None:
            raise ProtocolBoundaryCrossed(
                f"cell {cell.cell_id!r} is in phase {cell.phase!r}, which this objective "
                f"does not declare {list(objective.phases)}")
        if cell.protocol_id != declared:
            raise ProtocolBoundaryCrossed(
                f"cell {cell.cell_id!r} cites protocol {cell.protocol_id!r} but phase "
                f"{cell.phase!r} is judged under {declared!r}; each phase is judged under "
                "its own protocol so nothing crosses a protocol boundary (§1.6), and a "
                "comparison across protocols is analysis rather than a claim "
                "(MEASUREMENT.md:83-84)")


def _require_composed_champion(champion: "ChampionLineage", cells: Sequence) -> None:
    """§9.7's composed champion at a door. Same reasoning as `_require_one_backend`.

    `check_matrix_coverage()` reads the champion for the mechanism, anchor and
    ordering checks, and then measured coverage, co-residency, repetitions and the
    sentinel set over whatever cells it was handed — including a member
    candidate's. That is *"adding local percentages"* wearing the composition's
    name at the one door that already knows both.
    """
    for cell in cells:
        if cell.candidate_id != champion.combined_candidate_id:
            raise ChampionMismatch(
                f"cell {cell.cell_id!r} is evidence of candidate {cell.candidate_id!r}, "
                f"not of the composed champion {champion.combined_candidate_id!r}. T2 "
                "runs on the composed champion, never by adding local percentages (§9.7); "
                "a member's own result is not the composition's result")


# =============================================================================
# Inputs — the champion, the objective, the declared matrix
# =============================================================================

@dataclass(frozen=True)
class ChampionLineage:
    """The composed champion a T2 round measures, and when it became one.

    §9.7 runs T2 *"on the composed champion, never by adding local percentages"*,
    and the protocol admits only confirmation evidence *"gathered after the
    candidate entered the lineage"*. Both need this object: the first needs the
    combined candidate's id to reject a member's cell, the second needs the
    moment the lineage was entered so an earlier measurement can be refused
    rather than silently counted.
    """

    combined_candidate_id: str
    source_tree: str
    anchor: api.AnchorIdentity
    entered_lineage_at: str
    member_candidate_ids: tuple = ()

    def __post_init__(self) -> None:
        _text(self.combined_candidate_id, "champion.combined_candidate_id")
        if not self.combined_candidate_id.startswith("akc-"):
            raise ChampionMismatch(
                f"champion.combined_candidate_id: {self.combined_candidate_id!r} must "
                "start with 'akc-'; the composed champion is a candidate that was "
                "re-measured as a whole (§8.9)")
        _text(self.source_tree, "champion.source_tree")
        if self.source_tree not in schemas.SOURCE_TREES:
            raise CellInadmissible(
                f"champion.source_tree: {self.source_tree!r} is not one of "
                f"{sorted(schemas.SOURCE_TREES)}")
        if not isinstance(self.anchor, api.AnchorIdentity):
            raise CellInadmissible("champion.anchor must be an api.AnchorIdentity")
        _instant(self.entered_lineage_at, "champion.entered_lineage_at")
        members = _tuple_of(self.member_candidate_ids, "champion.member_candidate_ids", str)
        for member in members:
            if not member.startswith("akc-"):
                raise ChampionMismatch(
                    f"champion.member_candidate_ids: {member!r} must start with 'akc-'")
            if member == self.combined_candidate_id:
                raise ChampionMismatch(
                    "champion.member_candidate_ids names the combined candidate itself; "
                    "the composition is re-measured as a whole and is not its own member")

    @property
    def entered_at(self) -> datetime:
        return _instant(self.entered_lineage_at, "champion.entered_lineage_at")

    def to_dict(self) -> dict:
        return {
            "combined_candidate_id": self.combined_candidate_id,
            "source_tree": self.source_tree,
            "anchor": self.anchor.to_dict(),
            "entered_lineage_at": self.entered_lineage_at,
            "member_candidate_ids": list(self.member_candidate_ids),
        }


@dataclass(frozen=True)
class PhaseTradeException:
    """A PRE-DECLARED phase trade (§1.6). Declaring it decides nothing.

    *"A phase trade — a small prefill regression buying a large decode gain — is
    permitted only as a pre-declared campaign exception naming the exact
    regression band, the exact expected gain, and the roles affected, and it is an
    operator decision at freeze time, not a controller decision."*

    So this object is read by `PhaseTradeAssessment` to report whether what was
    measured falls inside what was pre-declared. It never converts a
    `objective_not_met` standing into a met one — that conversion is the
    operator's, at freeze time, and it is not expressible here.
    """

    regressing_phase: str
    band: tuple
    expected_gain: float
    roles: tuple
    declared_at: str

    def __post_init__(self) -> None:
        _text(self.regressing_phase, "phase_trade.regressing_phase")
        if not isinstance(self.band, tuple) or len(self.band) != 2:
            raise CellInadmissible(
                "phase_trade.band must be a (low, high) tuple naming the EXACT "
                "regression band, in the phase's own metric and oriented scale")
        low, high = self.band
        _finite(low, "phase_trade.band[0]")
        _finite(high, "phase_trade.band[1]")
        if not low <= high:
            raise CellInadmissible(
                f"phase_trade.band: {self.band!r} is not ordered (low, high)")
        if high > 0:
            raise CellInadmissible(
                "phase_trade.band names a REGRESSION band, so its oriented bounds are "
                "at or below zero; a band whose upper bound is positive does not "
                "describe a regression")
        _finite(self.expected_gain, "phase_trade.expected_gain")
        if self.expected_gain <= 0:
            raise CellInadmissible(
                "phase_trade.expected_gain must be strictly positive: a trade with no "
                "expected gain is a regression with paperwork")
        roles = _tuple_of(self.roles, "phase_trade.roles", str)
        if not roles:
            raise CellInadmissible(
                "phase_trade.roles: the exception names the roles affected; an "
                "unscoped exception is not pre-declared, it is open-ended")
        _instant(self.declared_at, "phase_trade.declared_at")

    def to_dict(self) -> dict:
        return {"regressing_phase": self.regressing_phase, "band": list(self.band),
                "expected_gain": self.expected_gain, "roles": list(self.roles),
                "declared_at": self.declared_at}


@dataclass(frozen=True)
class ObjectiveSpec:
    """§1.6's objective for ONE backend. It never spans two (AK-D12).

    `protocol_by_phase` is what keeps prefill, decode and MI210 cells inside their
    own instruments: a cell citing a protocol its phase does not declare is
    refused, so P-BENCH-1 evidence can never be read as P-BENCH-PREFILL-1
    evidence. Two phases MAY share one protocol id — P-GPU-1 governs both GPU
    phases — which is why the check is "the phase's declared protocol", not
    "a protocol no other phase uses".
    """

    backend: str
    phases: tuple
    protocol_by_phase: Mapping[str, str]
    improvement_quantifier: str
    recipe_class: str = "production_optimal"
    rule: str = OBJECTIVE_RULE
    phase_trade_exception: Optional[PhaseTradeException] = None

    def __post_init__(self) -> None:
        _backend(self.backend, "objective.backend")
        phases = _tuple_of(self.phases, "objective.phases", str)
        if not phases:
            raise CellInadmissible("objective.phases: at least one phase is required")
        if len(set(phases)) != len(phases):
            raise CellInadmissible(f"objective.phases: {list(phases)} repeats a phase")
        for phase in phases:
            _phase_for(self.backend, phase, "objective.phases[]")
        if not isinstance(self.protocol_by_phase, Mapping):
            raise CellInadmissible("objective.protocol_by_phase must be a mapping")
        for phase in phases:
            protocol = self.protocol_by_phase.get(phase)
            if not isinstance(protocol, str) or not protocol.strip():
                raise CellInadmissible(
                    f"objective.protocol_by_phase[{phase!r}]: every declared phase needs "
                    "its own protocol id (MEASUREMENT.md:13); each phase is judged under "
                    "its own protocol so nothing crosses a protocol boundary (§1.6)")
        for phase in self.protocol_by_phase:
            if phase not in phases:
                raise CellInadmissible(
                    f"objective.protocol_by_phase[{phase!r}]: names a phase the objective "
                    "does not declare")
        if self.improvement_quantifier not in IMPROVEMENT_QUANTIFIERS:
            raise CellInadmissible(
                f"objective.improvement_quantifier: {self.improvement_quantifier!r} is "
                f"not one of {list(IMPROVEMENT_QUANTIFIERS)}. §1.6 does not disambiguate "
                "which quantifier scopes the improvement half, and the two readings are "
                "far apart, so the campaign declares it rather than this module guessing")
        if self.recipe_class not in schemas.RECIPE_CLASSES:
            raise CellInadmissible(
                f"objective.recipe_class: {self.recipe_class!r} is not one of "
                f"{sorted(schemas.RECIPE_CLASSES)}. Invariant 15: baseline/off-recipe "
                "cells are diagnostic and never veto or justify a release")
        if self.rule not in schemas.OBJECTIVE_RULES:
            raise CellInadmissible(
                f"objective.rule: {self.rule!r} is not one of "
                f"{sorted(schemas.OBJECTIVE_RULES)}")
        if self.phase_trade_exception is not None:
            if not isinstance(self.phase_trade_exception, PhaseTradeException):
                raise CellInadmissible(
                    "objective.phase_trade_exception must be a PhaseTradeException or None")
            if self.phase_trade_exception.regressing_phase not in phases:
                raise CellInadmissible(
                    f"objective.phase_trade_exception names phase "
                    f"{self.phase_trade_exception.regressing_phase!r}, which the objective "
                    f"does not declare {list(phases)}")

    def protocol_for(self, phase: str) -> str:
        protocol = self.protocol_by_phase.get(phase)
        if protocol is None:
            raise ProtocolBoundaryCrossed(
                f"phase {phase!r} has no declared protocol in this objective; a phase "
                "judged under no named protocol is not judged (MEASUREMENT.md:13)")
        return protocol

    def to_dict(self) -> dict:
        return {
            "backend": self.backend, "phases": list(self.phases),
            "protocol_by_phase": dict(self.protocol_by_phase),
            "improvement_quantifier": self.improvement_quantifier,
            "recipe_class": self.recipe_class, "rule": self.rule,
            "phase_trade_exception": (None if self.phase_trade_exception is None
                                      else self.phase_trade_exception.to_dict()),
        }


@dataclass(frozen=True)
class T2MatrixSpec:
    """The T2 matrix a backend adapter DECLARES, from compiled facts (§9.7, AK5).

    Every quantity here is a coverage requirement. None of them is a weight on a
    measurement: §1.6 withdrew the production-weighted composite, so
    `production_share` on a cell orders and selects the matrix and never
    multiplies an estimate. That is not a convention — this module contains no
    multiplication at all (`audit_no_weighting_or_averaging`).

    `required_capacity_kinds` has no default on purpose. An adapter that has not
    said which capacity axes it protects has not declared a satisfied
    requirement; it has declared nothing, and the coverage check says
    COULD_NOT_CHECK.
    """

    backend: str
    required_coverage: tuple
    t1_paired_blocks_by_phase: Mapping[str, int]
    t1_sentinel_ids: frozenset
    required_capacity_kinds: tuple
    effect_scale: str
    extra_co_resident_backends: frozenset = frozenset()

    def __post_init__(self) -> None:
        _backend(self.backend, "matrix.backend")
        coverage = _tuple_of(self.required_coverage, "matrix.required_coverage", tuple)
        if not coverage:
            raise MatrixSpecInvalid(
                "matrix.required_coverage is empty: §9.7 requires one or a few roles per "
                "AFFECTED architecture/regime, so a matrix that names no affected "
                "architecture cannot be checked for a coverage gap")
        for pair in coverage:
            if len(pair) != 2:
                raise MatrixSpecInvalid(
                    f"matrix.required_coverage: {pair!r} must be "
                    "(architecture_class, regime)")
            _text(pair[0], "matrix.required_coverage[].architecture_class")
            _text(pair[1], "matrix.required_coverage[].regime")
        if not isinstance(self.t1_paired_blocks_by_phase, Mapping) \
                or not self.t1_paired_blocks_by_phase:
            raise MatrixSpecInvalid(
                "matrix.t1_paired_blocks_by_phase must be a non-empty mapping: 'stronger "
                "paired repetitions than T1' is uncheckable without T1's count")
        for phase, blocks in self.t1_paired_blocks_by_phase.items():
            _phase_for(self.backend, phase, "matrix.t1_paired_blocks_by_phase key")
            if isinstance(blocks, bool) or not isinstance(blocks, int) or blocks < 1:
                raise MatrixSpecInvalid(
                    f"matrix.t1_paired_blocks_by_phase[{phase!r}]: {blocks!r} is not a "
                    "positive block count")
        if not isinstance(self.t1_sentinel_ids, (frozenset, set)):
            raise MatrixSpecInvalid("matrix.t1_sentinel_ids must be a set of cell ids")
        for sentinel in self.t1_sentinel_ids:
            _text(sentinel, "matrix.t1_sentinel_ids[]")
        kinds = _tuple_of(self.required_capacity_kinds,
                          "matrix.required_capacity_kinds", str)
        for kind in kinds:
            if kind not in CAPACITY_KINDS:
                raise MatrixSpecInvalid(
                    f"matrix.required_capacity_kinds: {kind!r} is not one of "
                    f"{list(CAPACITY_KINDS)}")
        if self.effect_scale not in stats.EFFECT_SCALES:
            raise MatrixSpecInvalid(
                f"matrix.effect_scale: {self.effect_scale!r} is not one of "
                f"{list(stats.EFFECT_SCALES)}")
        if not isinstance(self.extra_co_resident_backends, (frozenset, set)):
            raise MatrixSpecInvalid(
                "matrix.extra_co_resident_backends must be a set of backend names")
        for backend in self.extra_co_resident_backends:
            _backend(backend, "matrix.extra_co_resident_backends[]")

    @property
    def co_resident_required(self) -> bool:
        """§9.7's llama_cpu requirement, plus anything the adapter ADDS to it.

        A spec cannot subtract: `CO_RESIDENT_REQUIRED_BACKENDS` is consulted
        directly, so a campaign cannot declare its way out of the co-resident
        cell that exists because CPU decode is bandwidth-bound.
        """
        return (self.backend in CO_RESIDENT_REQUIRED_BACKENDS
                or self.backend in self.extra_co_resident_backends)

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "required_coverage": [list(pair) for pair in self.required_coverage],
            "t1_paired_blocks_by_phase": dict(self.t1_paired_blocks_by_phase),
            "t1_sentinel_ids": sorted(self.t1_sentinel_ids),
            "required_capacity_kinds": list(self.required_capacity_kinds),
            "effect_scale": self.effect_scale,
            "co_resident_required": self.co_resident_required,
        }


# =============================================================================
# Inputs — the measured evidence
# =============================================================================

@dataclass(frozen=True)
class PhaseEvidence:
    """One e-process statement about one cell, bound to the verdict it produced.

    Two objects, because two different questions need answering and neither
    answers the other:

      * `verdict` (`api.Verdict`) carries **correctness precedence**. A candidate
        failing a lexicographically prior gate *"receives no speed rank at all —
        not a penalised one"*, and that is decided by `api`, not here.
      * `e_process` (`stats.EProcessRun`) carries **which hypothesis was tested**.
        `api.EffectEstimate` records the e-value and its threshold but not the
        null they were computed against, and §1.6 needs two different statements
        — non-inferiority and improvement — which are two different nulls. Taking
        the caller's word for which one ran would make the objective
        unverifiable.

    The two are cross-checked against each other: same e-value, same threshold,
    same block count. A mismatch means they are not the same run, and a
    non-inferiority statement assembled from another run's e-process is exactly
    the shape of evidence nobody can reconstruct.
    """

    verdict: api.Verdict
    e_process: Optional[stats.EProcessRun] = None

    def __post_init__(self) -> None:
        if not isinstance(self.verdict, api.Verdict):
            raise CellInadmissible("evidence.verdict must be an api.Verdict")
        if self.verdict.tier != TIER:
            raise CellInadmissible(
                f"evidence.verdict.tier is {self.verdict.tier!r}; the readiness signal "
                f"reads {TIER} records — T1 evidence is search evidence for ranking and "
                "T3 is a release gate this module does not run")
        effect = self.verdict.effect
        if effect is None:
            if self.e_process is not None:
                raise CellInadmissible(
                    "evidence carries an e-process but its verdict carries no estimate; "
                    "an e-process with nothing reduced from it is not a rate comparison")
            return
        if self.e_process is None:
            raise CellInadmissible(
                "evidence carries a rate estimate but no e-process; the hypothesis the "
                "estimate was computed against is what §1.6 needs and the estimate does "
                "not record it")
        if not isinstance(self.e_process, stats.EProcessRun):
            raise CellInadmissible("evidence.e_process must be a stats.EProcessRun")
        mismatches: list = []
        if self.e_process.e_running_max != effect.e_value:
            mismatches.append(
                f"e-value {self.e_process.e_running_max!r} vs estimate "
                f"{effect.e_value!r}")
        if self.e_process.threshold != effect.threshold:
            mismatches.append(
                f"threshold {self.e_process.threshold!r} vs estimate "
                f"{effect.threshold!r}")
        if self.e_process.blocks != effect.paired_blocks:
            mismatches.append(
                f"blocks {self.e_process.blocks!r} vs estimate "
                f"{effect.paired_blocks!r}")
        if mismatches:
            raise CellInadmissible(
                "evidence.e_process and evidence.verdict.effect do not describe the same "
                "run (" + "; ".join(mismatches) + ")")

    @classmethod
    def from_reduction(cls, reduction: stats.BlockReduction,
                       verdict: api.Verdict) -> "PhaseEvidence":
        """Build from the reducer's own output — the path a real T2 round takes."""
        if not isinstance(reduction, stats.BlockReduction):
            raise CellInadmissible("from_reduction() takes a stats.BlockReduction")
        return cls(verdict=verdict, e_process=reduction.e_process)

    @property
    def hypothesis(self) -> Optional[str]:
        return None if self.e_process is None else self.e_process.hypothesis

    @property
    def crossed(self) -> bool:
        """Did the e-process reject its null? False when nothing was reduced."""
        return self.e_process is not None and self.e_process.crossed

    def to_dict(self) -> dict:
        return {
            "status": self.verdict.status,
            "effect_resolution": self.verdict.effect_resolution,
            "hypothesis": self.hypothesis,
            "crossed": self.crossed,
            "e_process": (None if self.e_process is None
                          else {"hypothesis": self.e_process.hypothesis,
                                "margin": self.e_process.margin,
                                "null_boundary": self.e_process.null_boundary,
                                "threshold": self.e_process.threshold,
                                "blocks": self.e_process.blocks,
                                "first_crossing_block":
                                    self.e_process.first_crossing_block}),
            "effect": (None if self.verdict.effect is None
                       else self.verdict.effect.to_dict()),
        }


@dataclass(frozen=True)
class T2Cell:
    """One measured cell of the T2 matrix, on the composed champion.

    Carries BOTH statements §1.6 needs, because they are two different e-processes
    over the same measured material:

      * `non_inferiority` — H0: oriented effect <= -margin. Required.
      * `improvement`     — H0: oriented effect <= 0. Optional, because only
        `at least one` phase has to improve, so most cells legitimately carry
        only the non-inferiority statement.

    When both are present they must be reductions of the SAME window: same raw
    samples, same block count, same stratum, same anchor. Two windows are two
    cells. `gpu-cross-device.md:106-111` is the precedent — a conjunction whose
    halves were measured under conditions that do not co-occur is a reconstructed
    net, and the objective is exactly such a conjunction.

    `production_share` is coverage material. It orders the matrix and reports
    which roles a cell protects. It never multiplies an estimate, and this module
    could not multiply it if it wanted to.
    """

    cell_id: str
    backend: str
    phase: str
    protocol_id: str
    cell_class: str
    role: str
    architecture_class: str
    regime: str
    recipe_class: str
    co_residency: str
    production_share: float
    candidate_id: str
    event_id: str
    measured_at: str
    non_inferiority: PhaseEvidence
    improvement: Optional[PhaseEvidence] = None
    protects_roles: tuple = ()

    def __post_init__(self) -> None:
        _text(self.cell_id, "cell.cell_id")
        _backend(self.backend, "cell.backend")
        _phase_for(self.backend, self.phase, "cell.phase")
        _text(self.protocol_id, "cell.protocol_id")
        _text(self.cell_class, "cell.cell_class")
        if self.role not in CELL_ROLES:
            raise CellInadmissible(
                f"cell.role: {self.role!r} is not one of {list(CELL_ROLES)}")
        _text(self.architecture_class, "cell.architecture_class")
        _text(self.regime, "cell.regime")
        if self.recipe_class not in schemas.RECIPE_CLASSES:
            raise CellInadmissible(
                f"cell.recipe_class: {self.recipe_class!r} is not one of "
                f"{sorted(schemas.RECIPE_CLASSES)}. Invariant 15: a baseline or "
                "off-recipe cell is diagnostic and never justifies or vetoes a release, "
                "so it is not admissible to the readiness signal at all")
        _co_residency(self.co_residency, "cell.co_residency")
        share = _finite(self.production_share, "cell.production_share")
        if share < 0 or share > 1:
            raise CellInadmissible(
                f"cell.production_share: {share!r} is outside [0, 1]")
        _text(self.candidate_id, "cell.candidate_id")
        if not self.candidate_id.startswith("akc-"):
            raise CellInadmissible(
                f"cell.candidate_id: {self.candidate_id!r} must start with 'akc-'")
        _text(self.event_id, "cell.event_id")
        _instant(self.measured_at, "cell.measured_at")
        _tuple_of(self.protects_roles, "cell.protects_roles", str)

        if not isinstance(self.non_inferiority, PhaseEvidence):
            raise CellInadmissible("cell.non_inferiority must be a PhaseEvidence")
        if self.non_inferiority.hypothesis is not None \
                and self.non_inferiority.hypothesis != stats.HYPOTHESIS_NON_INFERIORITY:
            raise CellInadmissible(
                f"cell.non_inferiority carries a "
                f"{self.non_inferiority.hypothesis!r} e-process; §1.6's non-inferiority "
                "half is a non-inferiority null and an improvement e-process does not "
                "test it. Substituting one test for another is what "
                "`P-AK-SEARCH-1` forbids of an LCB and forbids here for the same reason")
        if self.improvement is not None:
            if not isinstance(self.improvement, PhaseEvidence):
                raise CellInadmissible(
                    "cell.improvement must be a PhaseEvidence or None")
            if self.improvement.hypothesis is not None \
                    and self.improvement.hypothesis != stats.HYPOTHESIS_IMPROVEMENT:
                raise CellInadmissible(
                    f"cell.improvement carries a {self.improvement.hypothesis!r} "
                    "e-process; the improvement half of §1.6 is an improvement null")
            self._require_same_window()

        for label, evidence in self._statements():
            effect = evidence.verdict.effect
            if effect is None:
                continue
            if effect.stratum != api.STRATUM_CONFIRMATION:
                raise StratumViolation(
                    f"cell {self.cell_id!r} {label} evidence is "
                    f"{effect.stratum!r}-stratum; the readiness signal is computed ONLY "
                    "from confirmation-stratum evidence, because selecting the maximum "
                    "over many candidates biases the selected estimate upward and the "
                    "evidence that promotes a candidate is structurally unfit to report "
                    "how ready it is")

    def _statements(self) -> tuple:
        if self.improvement is None:
            return (("non_inferiority", self.non_inferiority),)
        return (("non_inferiority", self.non_inferiority),
                ("improvement", self.improvement))

    def _require_same_window(self) -> None:
        """The two statements must be reductions of ONE window. Everything the
        window fixes must agree; only what the HYPOTHESIS fixes may differ.

        The split matters, and getting it wrong in either direction is a defect:

          * `metric`, `metric_direction`, `noise_floor`, `raw_samples_ref`,
            `paired_blocks`, `stratum` and the estimate are fixed by the WINDOW.
            `statistics.BlockReduction` reads metric and direction off the one
            `EvaluationRequest` and takes `noise_floor=cal.noise_floor_phi`, a
            CAMPAIGN A/A calibration that knows nothing about which null was
            tested. Two statements that disagree about any of them are two
            windows, and a §1.6 conjunction over two windows is a reconstructed
            net. The noise floor especially: it is the boundary between "no win"
            and a magnitude, so halves that disagree about it let a cell be
            *inside* the campaign floor for one statement and orderable for the
            other — ranking noise through whichever door was left open.
          * `mde` and `threshold` and `e_value` are NOT bound, because
            `statistics.solve_mde` takes `hypothesis` and `margin`: the smallest
            detectable shift genuinely depends on where the null sits. Requiring
            one MDE for both halves would forbid a correctly computed pair.

        What the MDEs may NOT do is invert. `solve_mde` evaluates both halves on
        the SAME resampled A/A windows and differs only in
        `null_boundary_for()`: non-inferiority tests against `-margin`, improvement
        against `0`. A null the candidate has a `margin` head start on is reached
        by a SMALLER shift, so for one window the non-inferiority MDE is at or
        below the improvement MDE — measured on this construction the gap is the
        margin exactly. An inverted pair is not a coarser measurement, it is a
        pair `statistics.py` cannot emit, and it is the ONLY shape that makes a
        phase report `improved=PASS` beside a `ParityFigure` reading *"no
        detectable difference at any of them"* — one release line asserting both
        that a detectable improvement was resolved and that nothing was
        detectable. Refusing the pair closes that at its source instead of
        teaching every downstream reader which half to believe.
        """
        ni_effect = self.non_inferiority.verdict.effect
        imp_effect = self.improvement.verdict.effect
        if ni_effect is None or imp_effect is None:
            return
        mismatches: list = []
        if ni_effect.raw_samples_ref != imp_effect.raw_samples_ref:
            mismatches.append(
                f"raw samples {ni_effect.raw_samples_ref!r} vs "
                f"{imp_effect.raw_samples_ref!r}")
        if ni_effect.paired_blocks != imp_effect.paired_blocks:
            mismatches.append(
                f"blocks {ni_effect.paired_blocks} vs {imp_effect.paired_blocks}")
        if ni_effect.stratum != imp_effect.stratum:
            mismatches.append(f"stratum {ni_effect.stratum} vs {imp_effect.stratum}")
        if ni_effect.value != imp_effect.value:
            mismatches.append(f"estimate {ni_effect.value!r} vs {imp_effect.value!r}")
        if ni_effect.metric != imp_effect.metric:
            mismatches.append(f"metric {ni_effect.metric!r} vs {imp_effect.metric!r}")
        if ni_effect.metric_direction != imp_effect.metric_direction:
            mismatches.append(
                f"metric direction {ni_effect.metric_direction!r} vs "
                f"{imp_effect.metric_direction!r}; the two halves would orient one "
                "estimate two ways and disagree about its sign")
        if ni_effect.noise_floor != imp_effect.noise_floor:
            mismatches.append(
                f"campaign noise floor {ni_effect.noise_floor!r} vs "
                f"{imp_effect.noise_floor!r}; the floor is an A/A calibration of the "
                "window, not a property of the null, so two values for one window "
                "would let the same estimate be sub-floor for one statement and "
                "orderable for the other")
        if ni_effect.mde > imp_effect.mde:
            mismatches.append(
                f"the non-inferiority MDE {ni_effect.mde!r} is COARSER than the "
                f"improvement MDE {imp_effect.mde!r}; both are solved on the same A/A "
                "windows and non-inferiority tests against a null the candidate has a "
                "margin's head start on, so it cannot be the less sensitive of the "
                "two. An inverted pair is the one shape that makes a phase report "
                "improved=PASS beside a figure saying no cell showed a detectable "
                "difference")
        ni_anchor = self.non_inferiority.verdict.anchor
        imp_anchor = self.improvement.verdict.anchor
        if ni_anchor is not None and imp_anchor is not None:
            # FAIL only. `identity_matches` has three outcomes since 2026-08-04
            # (`AnchorIdentity.tool`), and this list is a list of things that
            # DISAGREE — every other entry above is a compared pair of values.
            # `!= PASS` swept the third outcome in with them and made an
            # unobserved tool name raise `CellInadmissible` saying the two halves
            # are "reductions of DIFFERENT windows", which is a stronger claim
            # than the comparator made. This follows
            # `correctness._refuse_replay_mismatch`'s stated rule: FAIL raises,
            # COULD_NOT_CHECK returns because nothing disagrees — and the
            # unobserved component still surfaces downstream, as
            # `_check_anchor_agreement`'s BLOCK_ANCHOR_ABSENT.
            if ni_anchor.identity_matches(imp_anchor).outcome == schemas.FAIL:
                mismatches.append("anchor identity")
        if mismatches:
            raise CellInadmissible(
                f"cell {self.cell_id!r}: the non-inferiority and improvement statements "
                "are reductions of DIFFERENT windows (" + "; ".join(mismatches) + "). "
                "§1.6's objective is a conjunction over one cell, and a conjunction "
                "whose halves were measured under conditions that do not co-occur is a "
                "reconstructed net (gpu-cross-device.md:106-111). Two windows are two "
                "cells")

    # -- projections used by the standing derivation ---------------------------

    @property
    def anchor(self) -> Optional[api.AnchorIdentity]:
        return self.non_inferiority.verdict.anchor

    @property
    def estimate(self) -> Optional[api.EffectEstimate]:
        return self.non_inferiority.verdict.effect

    @property
    def paired_blocks(self) -> Optional[int]:
        effect = self.estimate
        return None if effect is None else effect.paired_blocks

    @property
    def is_co_resident(self) -> bool:
        return self.co_residency.startswith("co_resident:")

    @property
    def measured_instant(self) -> datetime:
        return _instant(self.measured_at, "cell.measured_at")

    def oriented_effect(self) -> Optional[float]:
        """The cell's own estimate, oriented so POSITIVE means "candidate better".

        `stats.orient` does the orienting. This module does not know how to turn a
        lower-better metric around, on purpose: there is one place that knows, and
        a second one would eventually disagree with it about a sign.
        """
        effect = self.estimate
        if effect is None:
            return None
        return stats.orient(effect.value, effect.metric_direction)

    def to_dict(self) -> dict:
        return {
            "cell_id": self.cell_id, "backend": self.backend, "phase": self.phase,
            "protocol_id": self.protocol_id, "cell_class": self.cell_class,
            "role": self.role, "architecture_class": self.architecture_class,
            "regime": self.regime, "recipe_class": self.recipe_class,
            "co_residency": self.co_residency,
            "production_share": self.production_share,
            "candidate_id": self.candidate_id, "event_id": self.event_id,
            "measured_at": self.measured_at,
            "protects_roles": list(self.protects_roles),
            "non_inferiority": self.non_inferiority.to_dict(),
            "improvement": (None if self.improvement is None
                            else self.improvement.to_dict()),
        }


@dataclass(frozen=True)
class CapacityDelta:
    """One capacity axis, measured on the composed champion against the anchor.

    §9.7 makes capacity deltas part of the T2 matrix, and §9.6 makes capacity a
    banked axis in its own right: *"a single-signal noise gate discarded 8 of 11
    excluded trials that were in fact non-dominated"*, and on this host capacity
    is what makes the large models runnable at all. All three kinds are
    higher-better, so `delta` is already oriented: negative means capacity was
    lost.
    """

    kind: str
    backend: str
    delta: float
    event_id: str
    measured_at: str
    notes: tuple = ()

    def __post_init__(self) -> None:
        if self.kind not in CAPACITY_KINDS:
            raise CellInadmissible(
                f"capacity.kind: {self.kind!r} is not one of {list(CAPACITY_KINDS)}")
        _backend(self.backend, "capacity.backend")
        _finite(self.delta, "capacity.delta")
        _text(self.event_id, "capacity.event_id")
        _instant(self.measured_at, "capacity.measured_at")
        _tuple_of(self.notes, "capacity.notes", str)

    @property
    def regressed(self) -> bool:
        return self.delta < 0

    @property
    def measured_instant(self) -> datetime:
        return _instant(self.measured_at, "capacity.measured_at")

    def to_dict(self) -> dict:
        return {"kind": self.kind, "backend": self.backend, "delta": self.delta,
                "event_id": self.event_id, "measured_at": self.measured_at,
                "regressed": self.regressed, "notes": list(self.notes)}


@dataclass(frozen=True)
class MechanismConfirmation:
    """Cumulative mechanism confirmation for one member of the champion (§9.7).

    `P-AK-SEARCH-1-A1` clause 1: a banked candidate requires an explanation backed
    by bytes, FLOPs, counters, or a clean A/B — *"It got faster and I don't know
    why is a reason to keep measuring, not to land."* At T2 the question is
    CUMULATIVE: the mechanism each member predicted must still be observable on
    the composed champion, because composition is where two mechanisms cancel.

    `confirmed=False` is admissible ONLY with a recorded explanation, which is
    what the A1 clause's *"or a recorded explanation"* in §9.6 permits. An empty
    explanation with `confirmed=False` is refused rather than counted.

    `measured_at` is REQUIRED and has no default, because "cumulative confirmation
    on the composed champion" is an ordering statement before it is anything else:
    a receipt taken before the composition existed cannot be a receipt about the
    composition, however confidently it names it. Without a timestamp the class
    contradicted its own docstring — the member's local receipt it says *"does not
    carry forward"* was accepted as the cumulative one, and `_check_lineage_ordering`
    had nothing to order it by. An optional field would have left the same hole
    open to anyone who omitted it, so the ordering check can never be satisfied by
    withholding the fact it orders.
    """

    member_candidate_id: str
    predicted_mechanism: str
    confirmed: bool
    event_id: str
    measured_at: str
    explanation: str = ""

    def __post_init__(self) -> None:
        _text(self.member_candidate_id, "mechanism.member_candidate_id")
        if not self.member_candidate_id.startswith("akc-"):
            raise CellInadmissible(
                f"mechanism.member_candidate_id: {self.member_candidate_id!r} must start "
                "with 'akc-'")
        _text(self.predicted_mechanism, "mechanism.predicted_mechanism")
        if not isinstance(self.confirmed, bool):
            raise CellInadmissible("mechanism.confirmed must be a bool")
        _text(self.event_id, "mechanism.event_id")
        _instant(self.measured_at, "mechanism.measured_at")
        if not isinstance(self.explanation, str):
            raise CellInadmissible("mechanism.explanation must be a string")
        if not self.confirmed and not self.explanation.strip():
            raise CellInadmissible(
                f"mechanism for {self.member_candidate_id!r} is unconfirmed and carries "
                "no explanation; 'it got faster and I don't know why' is a reason to keep "
                "measuring, not to land (P-AK-SEARCH-1-A1 clause 1)")

    @property
    def measured_instant(self) -> datetime:
        return _instant(self.measured_at, "mechanism.measured_at")

    def to_dict(self) -> dict:
        return {"member_candidate_id": self.member_candidate_id,
                "predicted_mechanism": self.predicted_mechanism,
                "confirmed": self.confirmed, "event_id": self.event_id,
                "measured_at": self.measured_at,
                "explanation": self.explanation}


@dataclass(frozen=True)
class CapabilityObjective:
    """A workload that could not run before, entering readiness by §9.8's route.

    *"Some changes unlock a workload that previously could not run, so a
    throughput ratio is undefined. They enter the readiness signal only through a
    predeclared capability objective: the required model/role becomes runnable at
    the declared context/concurrency; correctness and quality floors pass;
    resource budget fits; and the utility model was fixed at campaign start, not
    invented after observing the candidate."*

    The last clause is the load-bearing one and the only one this module can
    check by itself, so it is checked by hash: the campaign manifest's
    campaign-start utility-model digest is compared with the one this objective
    carries, and a drift blocks the capability rather than admitting it.
    """

    objective_id: str
    backend: str
    utility_model_sha256: str
    declared_at: str
    runnable: schemas.Check
    correctness_floor: schemas.Check
    quality_floor: schemas.Check
    resource_budget: schemas.Check
    event_id: str

    def __post_init__(self) -> None:
        _text(self.objective_id, "capability.objective_id")
        _backend(self.backend, "capability.backend")
        try:
            _sha256(self.utility_model_sha256, "capability.utility_model_sha256")
        except CellInadmissible as exc:
            # Re-raised under this input's own error class so a caller catching
            # capability problems does not have to catch cell problems too. The
            # refusal itself is unchanged and its reason is carried verbatim.
            raise CapabilityObjectiveInvalid(str(exc)) from exc
        _instant(self.declared_at, "capability.declared_at")
        for name in ("runnable", "correctness_floor", "quality_floor",
                     "resource_budget"):
            value = getattr(self, name)
            if not isinstance(value, schemas.Check):
                raise CapabilityObjectiveInvalid(
                    f"capability.{name} must be a schemas.Check")
        _text(self.event_id, "capability.event_id")

    def to_dict(self) -> dict:
        return {"objective_id": self.objective_id, "backend": self.backend,
                "utility_model_sha256": self.utility_model_sha256,
                "declared_at": self.declared_at, "event_id": self.event_id,
                "runnable": self.runnable.outcome,
                "correctness_floor": self.correctness_floor.outcome,
                "quality_floor": self.quality_floor.outcome,
                "resource_budget": self.resource_budget.outcome}


@dataclass(frozen=True)
class ReferencePolicy:
    """The campaign's ADVISORY reference figures (§1.2, `readiness_reporting`).

    The `+25% point / +20% lower-bound` figure was demoted from an automatic
    release trigger to a readiness signal the loop reports (AK-D3), and the
    protocol closes the same door from its side: *"a readiness signal is not a
    freeze trigger."* So this object is a yardstick for a sentence an operator
    reads, and nothing in this module branches on whether the yardstick was met.

    `reference_lcb_gain` is the `+20%` half. `P-AK-SEARCH-1` permits an LCB
    *"beside the e-value as a labelled descriptive statistic"* only, so the
    comparison it feeds is labelled `descriptive` and no decision reads it.
    """

    reference_point_gain: float
    reference_lcb_gain: float

    def __post_init__(self) -> None:
        _finite(self.reference_point_gain, "reference.reference_point_gain")
        _finite(self.reference_lcb_gain, "reference.reference_lcb_gain")

    def to_dict(self) -> dict:
        return {"reference_point_gain": self.reference_point_gain,
                "reference_lcb_gain": self.reference_lcb_gain,
                "advisory": True, "is_trigger": False}


# =============================================================================
# Derived: per-cell standing
# =============================================================================

@dataclass(frozen=True)
class CellStanding:
    """What one cell says about §1.6's two halves. Two Checks, never one score."""

    cell_id: str
    backend: str
    phase: str
    role: str
    event_id: str
    non_inferiority: schemas.Check
    improvement: schemas.Check
    blockers: tuple
    oriented_effect: Optional[float]

    def __post_init__(self) -> None:
        for name in ("non_inferiority", "improvement"):
            if not isinstance(getattr(self, name), schemas.Check):
                raise CellInadmissible(f"standing.{name} must be a schemas.Check")
        for blocker in self.blockers:
            if blocker not in BLOCKERS:
                raise CellInadmissible(
                    f"standing.blockers: {blocker!r} is not a declared blocking "
                    f"condition {list(BLOCKERS)}")

    def to_dict(self) -> dict:
        return {
            "cell_id": self.cell_id, "backend": self.backend, "phase": self.phase,
            "role": self.role, "event_id": self.event_id,
            "non_inferiority": {"outcome": self.non_inferiority.outcome,
                                "reasons": list(self.non_inferiority.reasons)},
            "improvement": {"outcome": self.improvement.outcome,
                            "reasons": list(self.improvement.reasons)},
            "blockers": list(self.blockers),
            "oriented_effect": self.oriented_effect,
        }


def _verdict_gate(evidence: PhaseEvidence, label: str) -> Optional[tuple]:
    """Correctness precedence, applied before anything speed-shaped is read.

    Returns `(Check, blocker)` when the verdict denies a speed reading, or `None`
    when the verdict is a pass and the estimate may be read. *"A candidate failing
    any of them receives no speed rank at all — not a penalised one."*
    """
    status = evidence.verdict.status
    if status == api.STATUS_INVALID:
        return (schemas.Check(schemas.COULD_NOT_CHECK, (
            f"{label}: the record is INVALID — "
            f"{evidence.verdict.speed_rank_withheld_reason()}",)), BLOCK_CELL_INVALID)
    if status == api.STATUS_FAIL:
        return (schemas.Check(schemas.COULD_NOT_CHECK, (
            f"{label}: a lexicographically prior gate failed, so this cell has no speed "
            f"standing at all — {evidence.verdict.speed_rank_withheld_reason()}",)),
            BLOCK_CELL_FAILED_PRIOR_GATE)
    if status == api.STATUS_INCONCLUSIVE:
        return (schemas.Check(schemas.COULD_NOT_CHECK, (
            f"{label}: the record is INCONCLUSIVE — the experiment ran and did not "
            "resolve, which is not the same as a measurement that did not happen",)),
            BLOCK_CELL_INCONCLUSIVE)
    if evidence.verdict.effect is None:
        return (schemas.Check(schemas.COULD_NOT_CHECK, (
            f"{label}: the record carries no rate comparison, so it says nothing about "
            "non-inferiority or improvement",)), BLOCK_CELL_NOT_A_RATE_COMPARISON)
    if evidence.verdict.anchor is None:
        return (schemas.Check(schemas.COULD_NOT_CHECK, (
            f"{label}: no anchor is bound; absence of a comparison is not evidence of "
            "equivalence (P-AK-SEARCH-1 precondition 4)",)), BLOCK_ANCHOR_ABSENT)
    return None


def _inadmissibility_reason(cell: T2Cell) -> Optional[str]:
    """Why this cell's own verdict denies it a place in a matrix requirement.

    `None` means the cell counts. Anything else is the reason it does not, in the
    verdict's own words.

    §9.7's matrix requirements — coverage, co-residency, repetitions — are
    statements about what was MEASURED, and a cell whose correctness gate failed,
    whose window was voided, which never resolved, which carries no rate
    comparison, or which binds no anchor measured nothing that any of them can be
    satisfied by. *"A candidate failing any of them receives no speed rank at all
    — not a penalised one"*, and "this architecture is covered", "the co-resident
    requirement is met" and "T2 repeated more than T1" are all ranks in that sense:
    each one asserts the matrix learned something at that cell.

    The predicate is `_verdict_gate`, reused rather than restated — the same
    predicate `_phase_figure` selects under — because a second copy of *"what
    disqualifies a cell"* is a second copy that drifts, and the half that drifts is
    whichever one has fewer tests.
    """
    gated = _verdict_gate(cell.non_inferiority, f"cell {cell.cell_id!r}")
    if gated is None:
        return None
    check, _blocker = gated
    return "; ".join(check.reasons)


def _rank_admissible(cell: T2Cell) -> bool:
    """`True` when the cell's verdict permits it to count. See `_inadmissibility_reason`."""
    return _inadmissibility_reason(cell) is None


def _non_inferiority_check(cell: T2Cell) -> tuple:
    """§1.6's non-inferiority half for one cell. Returns `(Check, blockers)`."""
    evidence = cell.non_inferiority
    gated = _verdict_gate(evidence, "non-inferiority")
    if gated is not None:
        check, blocker = gated
        return check, (blocker,)

    effect = evidence.verdict.effect
    resolution = evidence.verdict.effect_resolution
    if evidence.crossed:
        note = ""
        if resolution == api.EFFECT_REGRESSION:
            note = (" The point estimate is oriented negative, so the cell is "
                    "non-inferior within the declared margin rather than faster.")
        return schemas.Check(schemas.PASS, (
            f"the non-inferiority e-process crossed at block "
            f"{evidence.e_process.first_crossing_block} with e={effect.e_value} against "
            f"threshold {effect.threshold}, rejecting H0: oriented effect <= "
            f"{evidence.e_process.null_boundary}.{note}",)), ()

    oriented = cell.oriented_effect()
    if resolution == api.EFFECT_EVIDENCE_BELOW_THRESHOLD and oriented is not None \
            and oriented < 0:
        return schemas.Check(schemas.FAIL, (
            f"a detectable degradation with no non-inferiority evidence: the oriented "
            f"effect is {oriented} with MDE {effect.mde} and floor {effect.noise_floor}, "
            f"and the non-inferiority e-process reached only e={effect.e_value} against "
            f"threshold {effect.threshold}. This is a reason to withhold readiness; it is "
            "not a test of inferiority, because no such e-process was run",)), \
            (BLOCK_DETECTABLE_DEGRADATION,)

    return schemas.Check(schemas.COULD_NOT_CHECK, (
        f"no non-inferiority evidence: resolution is {resolution!r} with e="
        f"{effect.e_value} against threshold {effect.threshold}, MDE {effect.mde} and "
        f"floor {effect.noise_floor}. |effect| below the MDE is 'no detectable "
        "difference', which is a result and a decision, and an estimate not exceeding the "
        "campaign floor MUST NOT be ranked, banked, or composed",)), \
        (BLOCK_NON_INFERIORITY_EVIDENCE_ABSENT,)


def _improvement_check(cell: T2Cell) -> tuple:
    """§1.6's improvement half for one cell. Returns `(Check, blockers)`."""
    evidence = cell.improvement
    if evidence is None:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "this cell carries no improvement e-process; §1.6 requires improvement "
            "somewhere, not everywhere, so a cell measured for non-inferiority alone is "
            "silent about improvement rather than negative",)), ()

    gated = _verdict_gate(evidence, "improvement")
    if gated is not None:
        check, blocker = gated
        return check, (blocker,)

    effect = evidence.verdict.effect
    resolution = evidence.verdict.effect_resolution
    if not evidence.verdict.speed_rank_admissible:
        return schemas.Check(schemas.FAIL, (
            f"no improvement: {evidence.verdict.speed_rank_withheld_reason()}",)), ()
    if evidence.crossed and resolution == api.EFFECT_IMPROVEMENT:
        return schemas.Check(schemas.PASS, (
            f"the improvement e-process crossed at block "
            f"{evidence.e_process.first_crossing_block} with e={effect.e_value} against "
            f"threshold {effect.threshold}, rejecting H0: oriented effect <= 0; the "
            f"estimate is {effect.value} with MDE {effect.mde} and floor "
            f"{effect.noise_floor}",)), ()
    return schemas.Check(schemas.FAIL, (
        f"no improvement: resolution is {resolution!r} with e={effect.e_value} against "
        f"threshold {effect.threshold}",)), ()


def cell_standing(cell: T2Cell) -> CellStanding:
    """Classify one cell against §1.6's two halves.

    Correctness first, always: a cell whose verdict failed a lexicographically
    prior gate gets no speed reading at all, and its blocker says which gate.
    """
    if not isinstance(cell, T2Cell):
        raise CellInadmissible("cell_standing() takes a T2Cell")
    ni_check, ni_blockers = _non_inferiority_check(cell)
    imp_check, imp_blockers = _improvement_check(cell)
    blockers: list = []
    for blocker in ni_blockers + imp_blockers:
        if blocker not in blockers:
            blockers.append(blocker)
    return CellStanding(
        cell_id=cell.cell_id, backend=cell.backend, phase=cell.phase, role=cell.role,
        event_id=cell.event_id, non_inferiority=ni_check, improvement=imp_check,
        blockers=tuple(blockers), oriented_effect=cell.oriented_effect())


# =============================================================================
# Derived: per-phase standing
# =============================================================================

def _resolution_of(cell: T2Cell) -> str:
    """The EVALUATOR's resolution for this cell's non-inferiority estimate.

    Read, never re-derived. `_resolve_effect` lives in `evaluator/api.py` and it
    is the one place that knows the floor/MDE/threshold ladder; a second
    classifier here would be a second copy of that arithmetic, and this module is
    supposed to contain no arithmetic at all.
    """
    return cell.non_inferiority.verdict.effect_resolution


def _resolution_census(cells: Sequence[T2Cell]) -> tuple:
    """How many measured cells landed on each effect resolution.

    Counting, not reducing: no estimate is folded into another, and no number a
    cell measured is altered. The census exists because an exclusion nobody can
    see is how a figure becomes a lie without anyone writing one — a selection
    over a subset must publish which cells were NOT in the subset and why, in the
    evaluator's own vocabulary and in its declared order.
    """
    census: list = []
    for resolution in api.EFFECT_RESOLUTIONS:
        matching = [cell for cell in cells if _resolution_of(cell) == resolution]
        if matching:
            census.append((resolution, len(matching)))
    return tuple(census)


def _census_text(census: Sequence[tuple]) -> str:
    return ", ".join(f"{resolution}:{count}" for resolution, count in census)


def _validate_census(census: Any, label: str) -> tuple:
    entries = _tuple_of(census, label, tuple)
    for entry in entries:
        if len(entry) != 2:
            raise CellInadmissible(
                f"{label}: every entry must be (resolution, count), got {entry!r}")
        resolution, count = entry
        if resolution not in api.EFFECT_RESOLUTIONS:
            raise CellInadmissible(
                f"{label}: {resolution!r} is not one of "
                f"{list(api.EFFECT_RESOLUTIONS)}")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise CellInadmissible(
                f"{label}: {resolution!r} has count {count!r}; a census entry with no "
                "cells behind it reports coverage nobody measured")
    return entries


def _require_selected_resolution(resolution: Any) -> str:
    """The resolution the HEADLINE number was read off, and it may not be parity.

    `ReadinessFigure.kind` says "weakest ORDERABLE protected cell", and until this
    field existed the word `orderable` was the figure's own assertion about
    itself: the `resolution_census` said which resolutions the phase contained
    but not which one supplied the number, so a reader looking at
    `{evidence_below_threshold: 1, improvement: 1}` could not tell whether the
    figure they were reading was the degradation or the win.

    It matters most for `evidence_below_threshold`, which IS admitted to the
    ordering here and for which `api` sets `speed_rank_admissible=False` — the
    estimate cleared the floor and the MDE (a detectable magnitude, and exactly
    what an operator must see) but its e-process fell short. Both facts now ride
    on the figure and onto the rendered line, so "the evaluator withheld a speed
    rank from the cell this number came from" is something the operator reads
    rather than something they would have to re-derive.

    The one thing it may NOT be is sub-floor: `_phase_figure` excludes those, so a
    figure claiming one is a figure built by something that bypassed the
    selection.
    """
    if not isinstance(resolution, str):
        raise CellInadmissible(
            f"figure.selected_effect_resolution: expected a str, got {resolution!r}")
    if api.is_sub_floor_resolution(resolution):
        raise CellInadmissible(
            f"figure.selected_effect_resolution: {resolution!r} is sub-floor, and the "
            "evaluator withholds a speed rank from such a cell. Selecting it as the "
            "weakest IS a rank; a phase whose cells are all sub-floor is a ParityFigure")
    return resolution


def _cell_count(value: Any, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CellInadmissible(
            f"{label}: expected an int >= {minimum}, got {value!r}")
    return value


@dataclass(frozen=True)
class ReadinessFigure:
    """The number the operator reads, carried WHOLE out of one cell's estimate.

    Invariant 14 gives readiness to a deterministic reducer over journaled
    records. The strongest available form of that is not "computed carefully" but
    "not computed at all": `value` is the oriented estimate of one named cell,
    from one named event, and there is no arithmetic in this module that could
    have produced any other number.

    The cell chosen is the **weakest ORDERABLE protected cell** of the phase.
    §1.6 quantifies non-inferiority over EVERY protected cell, so the binding
    constraint is the worst one — but a cell whose estimate never cleared the
    campaign's own floor or MDE has no place on the ordering at all
    (`api.is_sub_floor_resolution`), and selecting it as "the weakest" would be
    ranking noise. `best_cell_id` rides along so the operator sees the spread
    rather than only its floor. Neither is an average: an average over cells
    would be a composite of measurements taken on different roles, which is the
    same defect as a composite across backends, one scope down.

    THE WORD "ORDERABLE" IS LOAD-BEARING, AND SO IS THE DISCLOSURE THAT GOES WITH
    IT. Once the selection is over a subset, the figure covers fewer cells than
    the phase has, and an operator reading it as "the weakest protected cell"
    reads it as covering cells it does not cover. `protected_cell_count`,
    `orderable_cell_count`, `parity_cell_ids` and `resolution_census` are
    REQUIRED — no defaults — because an exclusion that a caller may omit is an
    exclusion that will be omitted, and an undisclosed exclusion is how a figure
    becomes a lie without anyone writing one.
    """

    backend: str
    phase: str
    protocol_id: str
    kind: str
    cell_id: str
    event_id: str
    value: float
    metric: str
    metric_direction: str
    e_value: float
    threshold: float
    mde: float
    noise_floor: float
    paired_blocks: int
    stratum: str
    lcb_descriptive: Optional[float]
    best_cell_id: Optional[str]
    best_value: Optional[float]
    protected_cell_count: int
    orderable_cell_count: int
    parity_cell_ids: tuple
    resolution_census: tuple
    selected_effect_resolution: str
    selected_speed_rank_admissible: bool
    reference: Optional["ReferenceComparison"] = None

    #: What `kind` says: the figure is a SELECTED cell, never a reduction — and
    #: the selection ran over the ORDERABLE cells, not over all of them.
    KIND_WEAKEST_ORDERABLE_PROTECTED_CELL: ClassVar[str] = \
        "weakest_orderable_protected_cell"

    def __post_init__(self) -> None:
        _backend(self.backend, "figure.backend")
        _text(self.phase, "figure.phase")
        _text(self.protocol_id, "figure.protocol_id")
        if self.kind != self.KIND_WEAKEST_ORDERABLE_PROTECTED_CELL:
            raise CellInadmissible(
                f"figure.kind: {self.kind!r} is not "
                f"{self.KIND_WEAKEST_ORDERABLE_PROTECTED_CELL!r}; this module reports "
                "selected cells and has no other kind of orderable figure to report")
        _text(self.cell_id, "figure.cell_id")
        _text(self.event_id, "figure.event_id")
        _finite(self.value, "figure.value")
        if self.stratum != api.STRATUM_CONFIRMATION:
            raise StratumViolation(
                f"figure.stratum: {self.stratum!r}; a readiness figure admits only "
                f"{api.STRATUM_CONFIRMATION!r} evidence")
        _cell_count(self.protected_cell_count, "figure.protected_cell_count", minimum=1)
        _cell_count(self.orderable_cell_count, "figure.orderable_cell_count", minimum=1)
        _tuple_of(self.parity_cell_ids, "figure.parity_cell_ids", str)
        _validate_census(self.resolution_census, "figure.resolution_census")
        covered = self.orderable_cell_count + len(self.parity_cell_ids)
        if covered > self.protected_cell_count:
            raise CellInadmissible(
                f"figure: {self.orderable_cell_count} orderable plus "
                f"{len(self.parity_cell_ids)} at parity is more cells than the "
                f"{self.protected_cell_count} protected cells the phase has; the "
                "disclosure describes a matrix nobody measured")
        _require_selected_resolution(self.selected_effect_resolution)
        if not isinstance(self.selected_speed_rank_admissible, bool):
            raise CellInadmissible(
                "figure.selected_speed_rank_admissible must be a bool; the "
                "evaluator's answer is not optional and not inferable")
        if self.selected_effect_resolution not in [
                resolution for resolution, _count in self.resolution_census]:
            raise CellInadmissible(
                f"figure.selected_effect_resolution "
                f"{self.selected_effect_resolution!r} does not appear in this "
                f"figure's own census {list(self.resolution_census)}; the headline "
                "number would be attributed to a resolution the phase did not measure")

    @property
    def parity_cell_count(self) -> int:
        """Protected cells excluded from the selection as sub-floor.

        Derived from the ids rather than stored beside them, so the count and the
        list cannot disagree about which cells the figure does not cover.
        """
        return len(self.parity_cell_ids)

    def observation_fields(self) -> dict:
        """The fields `controller.guards.ReadinessObservation` needs.

        Returned rather than constructed, because AK5 does not import AK4: the
        controller consumes the release plane, not the other way round. There is
        one series per (backend, phase) and never one series for a whole host —
        a plateau computed over a folded number would be a plateau of a quantity
        nobody measured.

        This is the ORDERABLE shape, and it is the only shape that carries a
        `readiness` key. `ParityFigure.observation_fields()` returns a mapping
        with no such key, so a parity round cannot be fed to the constructor that
        takes one.
        """
        return {"readiness": self.value, "source_event_id": self.event_id,
                "stratum": self.stratum}

    def to_dict(self) -> dict:
        return {
            "backend": self.backend, "phase": self.phase,
            "protocol_id": self.protocol_id, "kind": self.kind,
            "cell_id": self.cell_id, "event_id": self.event_id, "value": self.value,
            "metric": self.metric, "metric_direction": self.metric_direction,
            "e_value": self.e_value, "threshold": self.threshold, "mde": self.mde,
            "noise_floor": self.noise_floor, "paired_blocks": self.paired_blocks,
            "stratum": self.stratum, "lcb_descriptive": self.lcb_descriptive,
            "lcb_label": "descriptive", "best_cell_id": self.best_cell_id,
            "best_value": self.best_value,
            # The wire discriminator, and it is carried on BOTH sides on purpose.
            # `ParityFigure` publishes `orderable: false`, so a JSON reader will
            # branch on it — and a key present only on the negative side turns
            # `row.get("orderable", False)` into "no figure is ever orderable",
            # which silently empties the orderable set instead of failing. A
            # distinction that survives in Python and collapses in JSON is not a
            # distinction.
            "orderable": True,
            "protected_cell_count": self.protected_cell_count,
            "orderable_cell_count": self.orderable_cell_count,
            "parity_cell_count": self.parity_cell_count,
            "parity_cell_ids": list(self.parity_cell_ids),
            "resolution_census": [list(entry) for entry in self.resolution_census],
            "selected_effect_resolution": self.selected_effect_resolution,
            "selected_speed_rank_admissible": self.selected_speed_rank_admissible,
            "reference": None if self.reference is None else self.reference.to_dict(),
        }


@dataclass(frozen=True)
class ParityFigure:
    """Protected cells WERE measured and none of them carries an orderable effect.

    THIS IS A RESULT, NOT AN ABSENCE, and that is the entire reason it is a type
    of its own rather than `None`. §1.6's first half is NON-INFERIORITY, so a
    backend that genuinely did not regress produces cells at
    `no_detectable_difference` — *"a result and a decision, not a failed
    experiment"*. Reporting the most common HEALTHY outcome as `None` would
    render it as "no protected-cell figure"; an absence reads as a coverage gap,
    and a coverage gap is something a later session closes by loosening the gate.
    That would be a worse defect than ranking sub-floor cells, which is the defect
    this type exists to close. It is the same lesson as absent-versus-empty on the
    operator surface: make the states structurally distinct rather than
    overloading one representation.

    What it carries instead of a value is the COUNT and the SENSITIVITY — how
    many protected cells were measured, how many landed at parity, the coarsest
    MDE and the calibrated floor they were judged against, and the cell and event
    that sensitivity came from. That is what makes "at parity" a claim an
    operator can size: parity at ±1.8% and parity at ±18% are different results.

    `value` and `best_value` RAISE rather than being absent, because
    `getattr(figure, "value", None)` is precisely how a caller silently converts
    this result back into the absence it is not.
    """

    backend: str
    phase: str
    protocol_id: str
    kind: str
    protected_cell_count: int
    measured_cell_ids: tuple
    parity_cell_ids: tuple
    resolution_census: tuple
    mde: float
    noise_floor: float
    sensitivity_cell_id: str
    sensitivity_event_id: str
    metric: str
    metric_direction: str
    stratum: str
    reference: Optional["ReferenceComparison"] = None

    #: The only kind. Every measured protected cell resolved below the campaign's
    #: own floor or MDE. There is no second kind on purpose: a cell carrying a
    #: DETECTABLE magnitude whose e-process fell short of threshold
    #: (`evidence_below_threshold`) is NOT at parity and is not excluded from the
    #: ordering here, so a figure that mixed the two could not exist without
    #: someone first deciding to hide a measured degradation.
    KIND_ALL_PROTECTED_CELLS_AT_PARITY: ClassVar[str] = "all_protected_cells_at_parity"

    KINDS: ClassVar[tuple] = (KIND_ALL_PROTECTED_CELLS_AT_PARITY,)

    def __post_init__(self) -> None:
        _backend(self.backend, "parity_figure.backend")
        _text(self.phase, "parity_figure.phase")
        _text(self.protocol_id, "parity_figure.protocol_id")
        if self.kind not in self.KINDS:
            raise CellInadmissible(
                f"parity_figure.kind: {self.kind!r} is not one of {list(self.KINDS)}")
        _text(self.sensitivity_cell_id, "parity_figure.sensitivity_cell_id")
        _text(self.sensitivity_event_id, "parity_figure.sensitivity_event_id")
        _text(self.metric, "parity_figure.metric")
        _finite(self.mde, "parity_figure.mde")
        _finite(self.noise_floor, "parity_figure.noise_floor")
        if self.stratum != api.STRATUM_CONFIRMATION:
            raise StratumViolation(
                f"parity_figure.stratum: {self.stratum!r}; a readiness figure admits "
                f"only {api.STRATUM_CONFIRMATION!r} evidence")
        _cell_count(self.protected_cell_count, "parity_figure.protected_cell_count",
                    minimum=1)
        measured = _tuple_of(self.measured_cell_ids, "parity_figure.measured_cell_ids",
                             str)
        parity = _tuple_of(self.parity_cell_ids, "parity_figure.parity_cell_ids", str)
        if not measured:
            raise CellInadmissible(
                "parity_figure.measured_cell_ids is empty; a phase where nothing was "
                "measured is the NOTHING MEASURED state and is reported as no figure at "
                "all. A parity figure that names no cell would make an unmeasured phase "
                "read as a healthy one")
        if len(measured) > self.protected_cell_count:
            raise CellInadmissible(
                f"parity_figure: {len(measured)} measured cells against "
                f"{self.protected_cell_count} protected cells")
        for cell_id in parity:
            if cell_id not in measured:
                raise CellInadmissible(
                    f"parity_figure.parity_cell_ids: {cell_id!r} is not among the "
                    "measured cells")
        _validate_census(self.resolution_census, "parity_figure.resolution_census")
        # The kind is not a label a caller applies, it is a fact about the cells.
        # A figure that says "at parity" while holding a cell that measured a
        # detectable magnitude would dress an unresolved possible regression as a
        # clean non-inferior result — which is the exact misreport this type
        # exists to prevent, running in the other direction.
        if len(parity) != len(measured):
            raise CellInadmissible(
                f"parity_figure: {len(measured) - len(parity)} of {len(measured)} "
                "measured cell(s) are not at parity, so this phase is not "
                f"{self.KIND_ALL_PROTECTED_CELLS_AT_PARITY!r}. A cell whose estimate "
                "cleared the floor and the MDE is a DETECTABLE magnitude and belongs on "
                "the ordering; calling it parity would hide it")

    @property
    def measured_cell_count(self) -> int:
        return len(self.measured_cell_ids)

    @property
    def parity_cell_count(self) -> int:
        return len(self.parity_cell_ids)

    @property
    def sensitivity_bound(self) -> float:
        """The magnitude the BLINDEST measured cell could not have told from nothing.

        A cell is sub-floor for one of two reasons and they bind at different
        numbers: `below_noise_floor` says `|effect| <= floor`, and
        `no_detectable_difference` says `|effect| < mde`. So the size a parity
        claim actually rules out at a cell is the LARGER of that cell's two
        published numbers, and `_parity_figure` selects the cell where that is
        greatest. Quoting the MDE alone would understate the claim whenever the
        calibrated floor is coarser than the MDE — a noisy co-resident cell with
        a large phi can measure a five-percent swing, resolve `below_noise_floor`,
        and be reported under a two-percent MDE as though nothing above two
        percent had happened anywhere.

        A selection between two numbers of ONE named cell, not a pooled quantity:
        both `mde` and `noise_floor` are that cell's own published pair, from that
        cell's own event.
        """
        return max(self.mde, self.noise_floor)

    def could_have_detected(self, magnitude: float) -> bool:
        """Would a real effect of this size have been visible at the blindest cell?

        The one question that separates *"the backend did not move"* from *"the
        run was too coarse to see it move"* — two different facts with the same
        shape, and the whole reason a parity claim is unfalsifiable without its
        sensitivity beside it. False means this parity result cannot distinguish
        "no effect" from an effect of `magnitude`, and must not be read as the
        first.

        A question, not a gate: it answers about a magnitude the caller supplies
        and decides nothing here.
        """
        return _finite(magnitude, "magnitude") > self.sensitivity_bound

    @property
    def comparable_reference_gain(self) -> Optional[float]:
        """The magnitude the campaign is LOOKING FOR — when it is comparable at all.

        `None` is a real answer and not a missing one: a campaign with no
        reference policy has declared no target magnitude, and an ABSOLUTE-scale
        campaign cannot be measured against a percentage yardstick (the same
        category error `_compare_reference` refuses one layer up, and
        `stats.block_effect` refuses one layer down). In both cases the question
        *"could this run have seen what the search is looking for?"* has no
        answer, and inventing one would put a target nobody declared behind a
        power claim.

        This is the ONE place the gate is written. `_parity_power_clause` renders
        it for an operator and `observation_fields()` publishes it to the
        controller; neither restates the two conditions, so the rendered sentence
        and the stop rule cannot come to disagree about what the campaign wants.
        """
        reference = self.reference
        if reference is None:
            return None
        if reference.effect_scale != stats.EFFECT_SCALE_RELATIVE:
            return None
        return reference.reference_point_gain

    @property
    def value(self) -> float:
        raise ParityHasNoOrderableValue(
            f"{self.backend} {self.phase}: {self.parity_cell_count} of "
            f"{self.measured_cell_count} measured protected cell(s) resolved below the "
            f"campaign's own sensitivity (MDE {self.mde}, floor {self.noise_floor}) and "
            "none is orderable. There is no value to read: sub-floor does not mean zero, "
            "it means the sign and the size are both unknown")

    @property
    def best_value(self) -> float:
        raise ParityHasNoOrderableValue(
            f"{self.backend} {self.phase}: there is no best cell, because there is no "
            "ordering. Selecting a cell as the best IS a rank, and the evaluator "
            "withheld a speed rank from every one of these cells")

    def observation_fields(self) -> dict:
        """The fields the controller's series needs for a round with NO readiness.

        There is deliberately no `readiness` key, and that absence is the whole
        mechanism. `guards.ReadinessObservation` takes `readiness` as a required
        argument, so this mapping cannot construct one — a parity round is not
        merely discouraged from entering the plateau series as a magnitude, it is
        unable to. A boolean `at_parity` flag beside a number would have left the
        consumer free to not check it, and every "flag it and hope the consumer
        checks" design in this package has turned out to be a defect.

        WHAT MAKES THE ROUND USABLE ON THE OTHER SIDE. A round with no magnitude
        is still evidence — *"nothing above +/-b moved"* — but only against the
        bound `b` and only against a magnitude worth ruling out. So the mapping
        carries `sensitivity_bound` (the binding one, not the MDE alone: see
        `sensitivity_bound`) and `reference_gain`, and the controller's plateau
        rule decides with them rather than around them. Both are published rather
        than re-derived over there: a consumer that recomputed the bound from
        `mde` and `noise_floor` would be a second copy of a rule that lives here,
        and a consumer that had neither would be left with COULD_NOT_EVALUATE
        forever on the most common HEALTHY outcome — which is a stall wearing the
        costume of caution.

        `reference_gain` is always PRESENT and may be `None`. An absent key would
        make a producer that lost it indistinguishable from a campaign that
        declared no target, and the seam refuses keys it was not sent.
        """
        return {"protected_cells": self.protected_cell_count,
                "cells_at_parity": self.parity_cell_count,
                "mde": self.mde, "noise_floor": self.noise_floor,
                "sensitivity_bound": self.sensitivity_bound,
                "reference_gain": self.comparable_reference_gain,
                "source_event_id": self.sensitivity_event_id,
                "stratum": self.stratum}

    def to_dict(self) -> dict:
        return {
            "backend": self.backend, "phase": self.phase,
            "protocol_id": self.protocol_id, "kind": self.kind,
            "protected_cell_count": self.protected_cell_count,
            "measured_cell_count": self.measured_cell_count,
            "measured_cell_ids": list(self.measured_cell_ids),
            "parity_cell_count": self.parity_cell_count,
            "parity_cell_ids": list(self.parity_cell_ids),
            "resolution_census": [list(entry) for entry in self.resolution_census],
            "mde": self.mde, "noise_floor": self.noise_floor,
            "sensitivity_bound": self.sensitivity_bound,
            "sensitivity_cell_id": self.sensitivity_cell_id,
            "sensitivity_event_id": self.sensitivity_event_id,
            "metric": self.metric, "metric_direction": self.metric_direction,
            "stratum": self.stratum,
            # NO `value` key, and not a null one. `value` RAISES on this object
            # precisely because `getattr(figure, "value", None)` is how a caller
            # reintroduces the absence; `dict.get("value")` is the same move one
            # serialization out, and `to_dict()` is the form that survives into a
            # report where the type no longer does. Absent means the reader that
            # assumed a magnitude fails at the line that assumed it.
            "orderable": False, "no_orderable_value_reason": (
                f"{self.parity_cell_count} of {self.measured_cell_count} measured "
                f"protected cell(s) resolved below the campaign's own sensitivity "
                f"(MDE {self.mde}, floor {self.noise_floor}); sub-floor does not mean "
                "zero, it means the sign and the size are both unknown"),
            "reference": None if self.reference is None else self.reference.to_dict(),
        }


#: The two figure types a phase can carry. `None` is the third state and is not a
#: type: nothing was measured. Anything that stores a figure validates against
#: this tuple, so a third kind of figure cannot be introduced by a caller.
PHASE_FIGURE_TYPES = (ReadinessFigure, ParityFigure)


def _orderable_value(figure: Any) -> Optional[float]:
    """The figure's orderable magnitude, or `None` when it has none.

    The one place that turns the three states into the two answers a MAGNITUDE
    question can have. A `ParityFigure` raises on `.value`, so every caller that
    wants to ask "how big was it" must come through here and say what it means by
    "there was no number" — rather than discovering it as an AttributeError at
    the point where it is least expected.
    """
    if isinstance(figure, ReadinessFigure):
        return figure.value
    return None


@dataclass(frozen=True)
class ReferenceComparison:
    """The `+25% / +20%` yardstick, compared and reported. Never branched on.

    AK-D3 demoted this figure from a trigger to a signal, so both comparisons are
    ADVISORY and neither participates in the standing. `lcb_at_or_above` is
    additionally labelled `descriptive`, because `P-AK-SEARCH-1` permits an LCB
    only *"beside the e-value as a labelled descriptive statistic"* and forbids
    any decision in the enumerated authority being taken on it.

    Both comparisons are `COULD_NOT_CHECK` when the campaign's effect scale is
    absolute: a percentage yardstick against an absolute effect is a category
    error, and silently switching scales is what `stats.block_effect` refuses one
    layer down.
    """

    effect_scale: str
    reference_point_gain: float
    reference_lcb_gain: float
    observed_point: Optional[float]
    observed_lcb_descriptive: Optional[float]
    point_at_or_above: schemas.Check
    lcb_at_or_above: schemas.Check
    advisory: bool = True

    def __post_init__(self) -> None:
        if self.effect_scale not in stats.EFFECT_SCALES:
            raise CellInadmissible(
                f"reference.effect_scale: {self.effect_scale!r} is not one of "
                f"{list(stats.EFFECT_SCALES)}")
        for name in ("point_at_or_above", "lcb_at_or_above"):
            if not isinstance(getattr(self, name), schemas.Check):
                raise CellInadmissible(f"reference.{name} must be a schemas.Check")
        if self.advisory is not True:
            raise TriggerAuthorityError(
                "ReferenceComparison.advisory cannot be False: the +25%/+20% figure is a "
                "readiness signal the loop reports, not a trigger (AK-D3), and "
                "P-AK-SEARCH-1 denial 5 states that a readiness signal is not a freeze "
                "trigger")

    def to_dict(self) -> dict:
        return {
            "effect_scale": self.effect_scale,
            "reference_point_gain": self.reference_point_gain,
            "reference_lcb_gain": self.reference_lcb_gain,
            "observed_point": self.observed_point,
            "observed_lcb_descriptive": self.observed_lcb_descriptive,
            "point_at_or_above": {"outcome": self.point_at_or_above.outcome,
                                  "reasons": list(self.point_at_or_above.reasons)},
            "lcb_at_or_above": {"outcome": self.lcb_at_or_above.outcome,
                                "reasons": list(self.lcb_at_or_above.reasons),
                                "label": "descriptive"},
            "advisory": True, "is_trigger": False,
        }


def _compare_reference(figure_value: Optional[float], lcb: Optional[float], *,
                       policy: Optional[ReferencePolicy],
                       effect_scale: str,
                       no_value_reason: Optional[str] = None
                       ) -> Optional[ReferenceComparison]:
    """The advisory +25%/+20% comparison, or an honest refusal to make it.

    `no_value_reason` is how a PARITY phase gets an ANSWER rather than a verdict.
    Comparing "no orderable effect" to a percentage yardstick is a category error
    in exactly the way an absolute effect scale is, and it is the more dangerous
    of the two: a `FAIL` here would read as a regression against the reference,
    and no regression was measured. So both comparisons are COULD_NOT_CHECK and
    NEITHER can be anything else on this branch — it returns before any threshold
    is looked at.

    The observed point and the LCB are dropped rather than reported alongside.
    An LCB carried out of a sub-floor estimate is a plausible-looking number
    sitting next to a threshold, and something downstream would eventually
    compare the two.
    """
    if policy is None:
        return None
    if no_value_reason is not None:
        if figure_value is not None:
            raise CellInadmissible(
                "_compare_reference() was handed both an orderable value and a reason "
                "there is none; a comparison cannot be simultaneously answerable and "
                "unanswerable, and whichever of the two won would be silent about the "
                "other")
        unanswerable = schemas.Check(schemas.COULD_NOT_CHECK, (no_value_reason,))
        return ReferenceComparison(
            effect_scale=effect_scale,
            reference_point_gain=policy.reference_point_gain,
            reference_lcb_gain=policy.reference_lcb_gain,
            observed_point=None, observed_lcb_descriptive=None,
            point_at_or_above=unanswerable, lcb_at_or_above=unanswerable)
    if effect_scale != stats.EFFECT_SCALE_RELATIVE:
        unusable = schemas.Check(schemas.COULD_NOT_CHECK, (
            f"the campaign's effect scale is {effect_scale!r}; the reference gains are "
            "stated as fractions of the anchor and comparing them against an absolute "
            "effect is a category error",))
        return ReferenceComparison(
            effect_scale=effect_scale,
            reference_point_gain=policy.reference_point_gain,
            reference_lcb_gain=policy.reference_lcb_gain,
            observed_point=figure_value, observed_lcb_descriptive=lcb,
            point_at_or_above=unusable, lcb_at_or_above=unusable)

    if figure_value is None:
        point = schemas.Check(schemas.COULD_NOT_CHECK,
                              ("no orderable protected-cell figure was available",))
    elif figure_value >= policy.reference_point_gain:
        point = schemas.Check(schemas.PASS, (
            f"the weakest orderable protected cell is at {figure_value}, at or above "
            f"the campaign's advisory reference of {policy.reference_point_gain}",))
    else:
        point = schemas.Check(schemas.FAIL, (
            f"the weakest orderable protected cell is at {figure_value}, below the "
            f"campaign's advisory reference of {policy.reference_point_gain}",))

    if lcb is None:
        lcb_check = schemas.Check(schemas.COULD_NOT_CHECK, (
            "no descriptive LCB was carried beside the e-value",))
    elif lcb >= policy.reference_lcb_gain:
        lcb_check = schemas.Check(schemas.PASS, (
            f"descriptive: the LCB is {lcb}, at or above the advisory reference of "
            f"{policy.reference_lcb_gain}. No decision is taken on this",))
    else:
        lcb_check = schemas.Check(schemas.FAIL, (
            f"descriptive: the LCB is {lcb}, below the advisory reference of "
            f"{policy.reference_lcb_gain}. No decision is taken on this",))

    return ReferenceComparison(
        effect_scale=effect_scale,
        reference_point_gain=policy.reference_point_gain,
        reference_lcb_gain=policy.reference_lcb_gain,
        observed_point=figure_value, observed_lcb_descriptive=lcb,
        point_at_or_above=point, lcb_at_or_above=lcb_check)


@dataclass(frozen=True)
class PhaseStanding:
    """§1.6 for one phase of one backend, under that phase's own protocol.

    `figure` carries the phase's THREE-STATE answer, and the three states are
    three different things rather than one representation doing triple duty:

      * `None`            — NOTHING MEASURED. No admissible cell produced an
                            effect at all.
      * `ParityFigure`    — ALL AT PARITY. Cells were measured and none is
                            orderable. A result, with a count and a sensitivity.
      * `ReadinessFigure` — an ORDERABLE figure, selected over the orderable
                            cells only and disclosing which cells it excluded.
    """

    backend: str
    phase: str
    protocol_id: str
    cells: tuple
    non_inferior: schemas.Check
    improved: schemas.Check
    figure: Optional[Any]
    blockers: tuple

    def __post_init__(self) -> None:
        if self.figure is not None and not isinstance(self.figure, PHASE_FIGURE_TYPES):
            raise CellInadmissible(
                f"standing.figure: {type(self.figure).__name__} is not one of "
                f"{[klass.__name__ for klass in PHASE_FIGURE_TYPES]} and is not None; "
                "the phase figure has exactly three states and a fourth would be one "
                "nothing downstream knows how to read")

    def to_dict(self) -> dict:
        return {
            "backend": self.backend, "phase": self.phase,
            "protocol_id": self.protocol_id,
            "cells": [standing.to_dict() for standing in self.cells],
            "non_inferior": {"outcome": self.non_inferior.outcome,
                             "reasons": list(self.non_inferior.reasons)},
            "improved": {"outcome": self.improved.outcome,
                         "reasons": list(self.improved.reasons)},
            "figure": None if self.figure is None else self.figure.to_dict(),
            "blockers": list(self.blockers),
        }


def phase_standing(*, backend: str, phase: str, objective: ObjectiveSpec,
                   cells: Sequence[T2Cell],
                   reference: Optional[ReferencePolicy] = None,
                   effect_scale: str = stats.EFFECT_SCALE_RELATIVE) -> PhaseStanding:
    """Judge one phase, over the protected cells of that phase only.

    Sentinels do not enter here: a dispatcher-boundary or non-target sentinel is
    a guard on the change's blast radius, not a cell the objective protects, and
    counting a sentinel's improvement toward §1.6 would let a lineage ship on a
    speed-up somewhere nobody runs.

    A PUBLIC entry point, so it refuses a foreign-backend cell itself rather than
    trusting `compute_readiness()` to have done it: `llama_cpu` and `llama_gpu`
    share the phase names `prefill` and `decode`, so a GPU cell handed to a CPU
    phase is not filtered out by anything else here — it is judged, and it can be
    SELECTED as the phase's readiness figure.

    It refuses a cell that cites a protocol this phase is not judged under, for
    exactly the same reason and at exactly the same door. This function does not
    read `cell.protocol_id` anywhere: it STAMPS `objective.protocol_for(phase)`
    onto the `PhaseStanding` and onto the `ReadinessFigure`, so a decode cell
    measured under `P-BENCH-PREFILL-1` came back labelled `P-BENCH-1` — a
    cross-protocol comparison wearing a within-protocol label, which
    `MEASUREMENT.md:83-84` makes analysis rather than a claim. Backend and protocol
    are one hole with two halves, and closing one half is what leaves the other
    findable only by someone who reads both doors.
    """
    if not isinstance(objective, ObjectiveSpec):
        raise CellInadmissible("phase_standing() takes an ObjectiveSpec")
    if objective.backend != backend:
        raise CrossBackendComposite(
            f"phase_standing() was asked for backend {backend!r} with an objective for "
            f"{objective.backend!r}; a phase is judged inside one backend's objective")
    cells = _tuple_of(cells, "cells", T2Cell)
    _require_one_backend(backend, cells, _describe_cell)
    protocol_id = objective.protocol_for(phase)
    # The cells of THIS phase, and only those: `compute_readiness()` hands the whole
    # matrix to every phase in turn, so refusing a prefill cell here because the
    # objective under judgement is decode's would forbid this function's own caller.
    # The cells that are judged are the ones that must cite the protocol they are
    # judged under.
    of_phase = [cell for cell in cells if cell.phase == phase]
    _require_declared_protocol(objective, of_phase)
    protected = [cell for cell in of_phase if cell.role == CELL_ROLE_PROTECTED]
    blockers: list = []

    if not protected:
        absent = schemas.Check(schemas.COULD_NOT_CHECK, (
            f"phase {phase!r} has no protected cell in this T2 matrix; §1.6 quantifies "
            "over every protected cell, and a phase with none was not measured",))
        return PhaseStanding(
            backend=backend, phase=phase, protocol_id=protocol_id, cells=(),
            non_inferior=absent, improved=absent, figure=None,
            blockers=(BLOCK_PHASE_NOT_MEASURED, BLOCK_PROTECTED_CELL_ABSENT))

    standings = tuple(cell_standing(cell) for cell in protected)
    for standing in standings:
        for blocker in standing.blockers:
            if blocker not in blockers:
                blockers.append(blocker)

    non_inferior = _combine([standing.non_inferiority for standing in standings])

    improvement_checks = [standing.improvement for standing in standings]
    if any(chk.outcome == schemas.PASS for chk in improvement_checks):
        improved = schemas.Check(schemas.PASS, tuple(
            f"{standing.cell_id}: {reason}"
            for standing in standings if standing.improvement.outcome == schemas.PASS
            for reason in standing.improvement.reasons))
    elif all(chk.outcome == schemas.COULD_NOT_CHECK for chk in improvement_checks):
        # Deliberately NOT a blocker at phase scope. §1.6 requires improvement
        # SOMEWHERE, not everywhere, so a phase that carries only non-inferiority
        # evidence is silent about improvement rather than obstructing it. The
        # blocker belongs to the backend, where the declared quantifier decides.
        improved = schemas.Check(schemas.COULD_NOT_CHECK, tuple(
            reason for chk in improvement_checks for reason in chk.reasons))
    else:
        improved = schemas.Check(schemas.FAIL, tuple(
            f"{standing.cell_id}: {reason}" for standing in standings
            for reason in standing.improvement.reasons))

    figure = _phase_figure(backend=backend, phase=phase, protocol_id=protocol_id,
                           protected=protected, reference=reference,
                           effect_scale=effect_scale)
    return PhaseStanding(backend=backend, phase=phase, protocol_id=protocol_id,
                         cells=standings, non_inferior=non_inferior, improved=improved,
                         figure=figure, blockers=tuple(blockers))


def _phase_figure(*, backend: str, phase: str, protocol_id: str,
                  protected: Sequence[T2Cell],
                  reference: Optional[ReferencePolicy],
                  effect_scale: str) -> Optional[Any]:
    """The phase's figure, in one of THREE structurally distinct states.

    TWO INDEPENDENT REASONS A CELL MAY NOT BE SELECTED, and they are two because
    conflating them is what put this defect here:

      * **its VERDICT denies it a speed reading** — INVALID, a failed
        lexicographically prior gate, INCONCLUSIVE, no rate comparison, no
        anchor. The predicate is `_verdict_gate` via `_rank_admissible`, reused
        rather than restated. *"A candidate failing any of them receives no speed
        rank at all — not a penalised one."*
      * **its EFFECT RESOLUTION carries no ordering** — the estimate never
        cleared the campaign's own noise floor or MDE. The predicate is
        `api.is_sub_floor_resolution`, which is the evaluator's, not a second
        copy: `_RANKABLE_RESOLUTIONS` already says *"below the noise floor is not
        a small win; it is not a win"*, and the evaluator WITHHOLDS a speed rank
        from such a cell. Selecting one as "the weakest" or "the best" IS a rank,
        so this function may not do what the evaluator refused to do.

    `EFFECT_EVIDENCE_BELOW_THRESHOLD` is deliberately NOT excluded. Its estimate
    cleared both the floor and the MDE: it is a DETECTABLE magnitude whose
    e-process fell short, and it is exactly the number an operator needs to see.
    Dropping it would make a measured degradation invisible, which is the failure
    in the other direction and the one `_non_inferiority_check` already reports as
    *"a detectable degradation with no non-inferiority evidence"*. Sub-floor and
    below-threshold are unrankable for different reasons, and only one of them is
    parity.

    THE THREE STATES, and why none of them is expressible as another:

      1. **NOTHING MEASURED** — `None`. No admissible cell produced an effect at
         all. `phase_standing` carries the blockers saying why.
      2. **ALL AT PARITY** — a `ParityFigure`. Cells were measured and none is
         orderable. Under a NON-INFERIORITY objective this is the most common
         HEALTHY outcome, and reporting it as `None` would render success as an
         absence. Absences read as coverage gaps, and a coverage gap is what a
         later session closes by loosening the gate — which would make the cure
         worse than the disease.
      3. **ORDERABLE** — a `ReadinessFigure` selected over the orderable cells
         ONLY, disclosing how many protected cells it left out and why. In the
         MIXED case the figure is "the weakest ORDERABLE protected cell", not
         "the weakest protected cell", and an operator told the second while
         handed the first is reading a number as covering cells it does not
         cover.
    """
    admissible = [cell for cell in protected if _rank_admissible(cell)]
    measured = [cell for cell in admissible if cell.oriented_effect() is not None]
    if not measured:
        return None

    census = _resolution_census(measured)
    parity = [cell for cell in measured
              if api.is_sub_floor_resolution(_resolution_of(cell))]
    orderable = [cell for cell in measured
                 if not api.is_sub_floor_resolution(_resolution_of(cell))]
    if not orderable:
        return _parity_figure(
            backend=backend, phase=phase, protocol_id=protocol_id,
            protected_count=len(protected), measured=measured, parity=parity,
            census=census, reference=reference, effect_scale=effect_scale)

    weakest = min(orderable, key=lambda cell: cell.oriented_effect())
    strongest = max(orderable, key=lambda cell: cell.oriented_effect())
    effect = weakest.estimate
    comparison = _compare_reference(
        weakest.oriented_effect(), effect.lcb_descriptive,
        policy=reference, effect_scale=effect_scale)
    return ReadinessFigure(
        backend=backend, phase=phase, protocol_id=protocol_id,
        kind=ReadinessFigure.KIND_WEAKEST_ORDERABLE_PROTECTED_CELL,
        cell_id=weakest.cell_id, event_id=weakest.event_id,
        value=weakest.oriented_effect(), metric=effect.metric,
        metric_direction=effect.metric_direction, e_value=effect.e_value,
        threshold=effect.threshold, mde=effect.mde, noise_floor=effect.noise_floor,
        paired_blocks=effect.paired_blocks, stratum=effect.stratum,
        lcb_descriptive=effect.lcb_descriptive,
        best_cell_id=strongest.cell_id, best_value=strongest.oriented_effect(),
        protected_cell_count=len(protected),
        orderable_cell_count=len(orderable),
        parity_cell_ids=tuple(cell.cell_id for cell in parity),
        resolution_census=census,
        # Which resolution the HEADLINE number was read off, and the evaluator's
        # own answer for that cell. `evidence_below_threshold` is admitted here
        # (see above) and `api` withholds its speed rank, so the figure must say
        # so rather than let the word "orderable" in `kind` stand for it.
        selected_effect_resolution=_resolution_of(weakest),
        selected_speed_rank_admissible=(
            weakest.non_inferiority.verdict.speed_rank_admissible),
        reference=comparison)


def _parity_figure(*, backend: str, phase: str, protocol_id: str,
                   protected_count: int, measured: Sequence[T2Cell],
                   parity: Sequence[T2Cell], census: tuple,
                   reference: Optional[ReferencePolicy],
                   effect_scale: str) -> ParityFigure:
    """Report "measured, and nothing is orderable" as the result it is.

    The sensitivity reported is the COARSEST one, and coarsest is measured by the
    BINDING BOUND rather than by the MDE alone. A phase is only as sensitive as
    its blindest cell, and which number blinds a cell depends on why it is
    sub-floor: `below_noise_floor` bounds the magnitude by the calibrated phi,
    `no_detectable_difference` bounds it by the MDE. Ranking the cells by MDE and
    then reporting that cell's floor beside it answers two different questions
    with two different cells' numbers — and it under-reports the claim whenever
    some other cell's floor is the coarsest thing in the phase. That is the
    failure this selection exists to avoid: a co-resident cell with a large
    calibrated phi can measure a five-percent swing, resolve `below_noise_floor`,
    and be published under another cell's two-percent MDE as though nothing above
    two percent had happened anywhere in the phase. A parity claim quoted tighter
    than its evidence is exactly the unfalsifiable claim the sensitivity is
    carried to prevent.

    `mde` and `noise_floor` are therefore ONE cell's own published pair, from ONE
    named event, and `sensitivity_bound` is the larger of the two. It is a
    SELECTION (`max` over cells, then `max` between one cell's two published
    numbers), never a pooled or averaged sensitivity — this module contains no
    arithmetic over measurements and `audit_no_weighting_or_averaging()` proves
    it from the AST. The tie-break on `cell_id` is there so two runs over the same
    matrix name the same cell whatever order the caller supplied it in.
    """
    coarsest = max(measured, key=lambda cell: (max(cell.estimate.mde,
                                                   cell.estimate.noise_floor),
                                               cell.cell_id))
    effect = coarsest.estimate
    bound = max(effect.mde, effect.noise_floor)
    reason = (
        f"every one of the {len(measured)} measured protected cell(s) resolved below "
        f"the campaign's own sensitivity ({_census_text(census)}; nothing above "
        f"+/-{bound} would have been distinguishable at {coarsest.cell_id!r}, whose MDE "
        f"is {effect.mde} and calibrated floor {effect.noise_floor}). There is "
        "no orderable effect to place against a percentage yardstick, and 'no "
        "detectable difference' is a result, not a number below a threshold")
    comparison = _compare_reference(
        None, None, policy=reference, effect_scale=effect_scale,
        no_value_reason=reason)
    return ParityFigure(
        backend=backend, phase=phase, protocol_id=protocol_id,
        kind=ParityFigure.KIND_ALL_PROTECTED_CELLS_AT_PARITY,
        protected_cell_count=protected_count,
        measured_cell_ids=tuple(cell.cell_id for cell in measured),
        parity_cell_ids=tuple(cell.cell_id for cell in parity),
        resolution_census=census,
        mde=effect.mde, noise_floor=effect.noise_floor,
        sensitivity_cell_id=coarsest.cell_id,
        sensitivity_event_id=coarsest.event_id,
        metric=effect.metric, metric_direction=effect.metric_direction,
        stratum=effect.stratum, reference=comparison)


# =============================================================================
# Derived: matrix coverage
# =============================================================================

@dataclass(frozen=True)
class MatrixCoverage:
    """Whether the T2 matrix §9.7 describes was actually run."""

    backend: str
    coverage: schemas.Check
    repetitions: schemas.Check
    sentinels: schemas.Check
    co_resident: schemas.Check
    capacity: schemas.Check
    mechanism: schemas.Check
    non_target: schemas.Check
    lineage_ordering: schemas.Check
    anchor_agreement: schemas.Check
    overall: schemas.Check
    blockers: tuple

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "checks": {name: {"outcome": getattr(self, name).outcome,
                              "reasons": list(getattr(self, name).reasons)}
                       for name in ("coverage", "repetitions", "sentinels",
                                    "co_resident", "capacity", "mechanism",
                                    "non_target", "lineage_ordering",
                                    "anchor_agreement", "overall")},
            "blockers": list(self.blockers),
        }


def _check_coverage(spec: T2MatrixSpec, cells: Sequence[T2Cell]) -> schemas.Check:
    """§9.7 coverage, over cells that MEASURED something.

    A pair whose only protected cell failed a prior gate, was voided, never
    resolved, or binds no anchor is not covered by it: coverage asks whether the
    affected architecture/regime was exercised, and an inadmissible cell exercised
    nothing the matrix can read. Counting it made "covered" satisfiable by a cell
    the very next check refuses to give a speed reading.

    The two ways to be uncovered are reported apart, because they tell the
    operator different things: NOT MEASURED AT ALL is a FAIL (the matrix does not
    reach that architecture), while MEASURED AND UNUSABLE is COULD_NOT_CHECK (it
    reached it and the record cannot be read). Both block — `check_matrix_coverage`
    raises `COVERAGE_GAP` on anything that is not a PASS — so nothing rides on the
    distinction except what the operator is told.
    """
    protected = [cell for cell in cells if cell.role == CELL_ROLE_PROTECTED]
    admissible = [cell for cell in protected if _rank_admissible(cell)]
    covered = {(cell.architecture_class, cell.regime) for cell in admissible}
    attempted: dict = {}
    for cell in protected:
        reason = _inadmissibility_reason(cell)
        if reason is not None:
            attempted.setdefault((cell.architecture_class, cell.regime), []).append(
                f"cell {cell.cell_id!r}: {reason}")

    absent: list = []
    unusable: list = []
    for pair in spec.required_coverage:
        key = tuple(pair)
        if key in covered:
            continue
        blocked = attempted.get(key)
        if blocked:
            unusable.append(
                f"architecture/regime {key!r} carries protected cell(s) that measured "
                "nothing readable, so §9.7's 'one or a few roles per affected "
                "architecture/regime' is not evidenced by them — " + "; ".join(blocked))
        else:
            absent.append(
                f"no protected cell covers architecture/regime {key!r}; §9.7 requires "
                "one or a few roles per affected architecture/regime")
    if absent:
        return schemas.Check(schemas.FAIL, tuple(absent) + tuple(unusable))
    if unusable:
        return schemas.Check(schemas.COULD_NOT_CHECK, tuple(unusable))
    return schemas.Check(schemas.PASS, (
        f"every declared architecture/regime pair {[tuple(p) for p in spec.required_coverage]} "
        "carries at least one protected cell with a readable measurement",))


def _check_repetitions(spec: T2MatrixSpec, cells: Sequence[T2Cell]) -> schemas.Check:
    """§9.7's *"stronger paired repetitions than T1"*, over the WHOLE T2 matrix.

    Two things this used to get wrong, in opposite directions:

      * it examined protected cells only, so a sentinel re-run at fewer blocks than
        T1 left the requirement green. T2's sentinel set is a strict superset of
        T1's and it is part of the same matrix; a blast-radius check run weaker
        than T1's is precisely the case where "broader" bought nothing, and §9.7
        does not scope the repetition requirement to one role.
      * it read the block count off cells whose verdict denies them a speed
        reading, so an inadmissible cell's 16 blocks satisfied the requirement for
        a phase whose readable cells ran fewer. A block count is evidence about
        repetition strength only if the run it counts produced a reading.

    An inadmissible cell is therefore reported as UNEVALUABLE rather than skipped:
    skipping would let a matrix of entirely inadmissible cells report PASS, which
    is the requirement satisfied by deleting what it inspects.
    """
    reasons: list = []
    unknown: list = []
    compared = 0
    for cell in cells:
        label = f"cell {cell.cell_id!r} ({cell.role})"
        inadmissible = _inadmissibility_reason(cell)
        if inadmissible is not None:
            unknown.append(
                f"{label} has no rank-admissible measurement, so its block count is not "
                f"evidence that T2 repeated more strongly than T1 — {inadmissible}")
            continue
        t1_blocks = spec.t1_paired_blocks_by_phase.get(cell.phase)
        if t1_blocks is None:
            unknown.append(
                f"{label} is in phase {cell.phase!r}, for which the spec declares no T1 "
                "block count, so 'stronger than T1' is unevaluable")
            continue
        blocks = cell.paired_blocks
        if blocks is None:
            # Belt and braces: `_verdict_gate` already refuses a cell carrying no
            # rate comparison, so this is unreachable while that holds. It is kept
            # so a future change there degrades to UNEVALUABLE rather than to a
            # comparison against None.
            unknown.append(
                f"{label} carries no rate comparison, so its block count cannot be "
                "compared with T1's")
            continue
        compared += 1
        if blocks <= t1_blocks:
            reasons.append(
                f"{label} ran {blocks} paired blocks against T1's "
                f"{t1_blocks}; §9.7 requires STRONGER paired repetitions than T1, because "
                "T2 is what the readiness signal is reported from")
    if reasons:
        return schemas.Check(schemas.FAIL, tuple(reasons) + tuple(unknown))
    if unknown:
        return schemas.Check(schemas.COULD_NOT_CHECK, tuple(unknown))
    if not compared:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "no cell in this matrix carries a block count that could be compared with "
            "T1's, so 'stronger paired repetitions than T1' is unevaluable rather than "
            "satisfied; an empty matrix meets no requirement",))
    return schemas.Check(schemas.PASS, (
        f"{compared} rank-admissible cell(s) each ran more paired blocks than T1's count "
        "for their phase",))


def _check_sentinels(spec: T2MatrixSpec, cells: Sequence[T2Cell]) -> schemas.Check:
    t2_sentinels = {cell.cell_id for cell in cells if cell.role in SENTINEL_ROLES}
    t1_sentinels = set(spec.t1_sentinel_ids)
    missing = sorted(t1_sentinels - t2_sentinels)
    added = sorted(t2_sentinels - t1_sentinels)
    reasons: list = []
    if missing:
        reasons.append(
            f"T2 drops sentinels T1 carried: {missing}. §9.7 requires BROADER "
            "dispatcher-boundary and non-target sentinels, and a matrix that drops one "
            "is narrower where it matters most")
    if not added:
        reasons.append(
            "T2 adds no sentinel beyond T1's set; 'broader' is a strict superset, "
            "otherwise T2 re-runs T1's blast-radius check at greater cost and learns "
            "nothing new about it")
    if reasons:
        return schemas.Check(schemas.FAIL, tuple(reasons))
    return schemas.Check(schemas.PASS, (
        f"T2 carries T1's {len(t1_sentinels)} sentinel(s) and adds {len(added)}",))


#: The reason the co-residency requirement exists, carried into every refusal
#: because it is also the reason a sentinel cannot close it.
_CO_RESIDENT_WHY = (
    "Production runs concurrent instances and CPU decode is bandwidth-bound, so a "
    "change can be neutral alone and harmful co-resident; a single-instance matrix "
    "cannot see that at all")


def _check_co_resident(spec: T2MatrixSpec, cells: Sequence[T2Cell]) -> schemas.Check:
    """§9.7's *"at least one co-resident cell"*, closable only by a PROTECTED one.

    The requirement is not "the matrix contains the string `co_resident:`". It
    exists because CPU decode is bandwidth-bound under concurrency **for a role the
    objective protects**, so the cell that discharges it has to be a cell §1.6
    quantifies over, carrying a reading. A dispatcher-boundary or non-target
    sentinel run co-resident is a blast-radius probe on a path nobody is protecting
    — it cannot show the harm the requirement was written to catch — and an
    inadmissible protected cell shows nothing at all.

    Sentinel-only is a FAIL: the matrix ran co-resident work and still did not
    measure a protected role co-resident. Protected-but-unreadable is
    COULD_NOT_CHECK: it measured the right thing and the record cannot be read.
    Either way `CO_RESIDENT_CELL_ABSENT` blocks.
    """
    if not spec.co_resident_required:
        return schemas.Check(schemas.PASS, (
            f"backend {spec.backend!r} declares no co-residency requirement",))
    co_resident = [cell for cell in cells if cell.is_co_resident]
    protected = [cell for cell in co_resident if cell.role == CELL_ROLE_PROTECTED]
    readable = [cell for cell in protected if _rank_admissible(cell)]
    if readable:
        return schemas.Check(schemas.PASS, tuple(
            f"co-resident protected cell {cell.cell_id!r} at {cell.co_residency}"
            for cell in readable))
    if protected:
        return schemas.Check(schemas.COULD_NOT_CHECK, tuple(
            f"co-resident protected cell {cell.cell_id!r} measured nothing readable, so "
            f"the co-residency requirement is unevaluable rather than met — "
            f"{_inadmissibility_reason(cell)}" for cell in protected))
    if co_resident:
        return schemas.Check(schemas.FAIL, (
            f"the only co-resident cell(s) are sentinels "
            f"({sorted(cell.cell_id for cell in co_resident)}); backend "
            f"{spec.backend!r} requires at least one co-resident cell in a role the "
            f"objective PROTECTS. {_CO_RESIDENT_WHY}, and a sentinel is a guard on the "
            "change's blast radius rather than a role §1.6 quantifies over, so it cannot "
            "carry the requirement it was never measured to answer",))
    return schemas.Check(schemas.FAIL, (
        f"backend {spec.backend!r} requires AT LEAST ONE co-resident cell and the "
        f"matrix has none. {_CO_RESIDENT_WHY}",))


def _check_capacity(spec: T2MatrixSpec,
                    deltas: Sequence[CapacityDelta]) -> tuple:
    if not spec.required_capacity_kinds:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            f"backend {spec.backend!r} declares no required capacity kinds; §9.7 makes "
            "capacity (VRAM/RAM/context) deltas part of the T2 matrix, and an adapter "
            "that has not said which axes it protects has not declared a satisfied "
            "requirement",)), (BLOCK_CAPACITY_REQUIREMENT_UNDECLARED,)
    # EVERY delta for this backend, not one per kind. A `{kind: delta}` mapping
    # keeps the LAST record for each axis and silently discards the others, so a
    # measured RAM regression followed by a clean RAM re-measurement read as PASS
    # with no blocker at all — and the same two records in the other order read as
    # FAIL. A reducer whose answer depends on the order its evidence was handed to
    # it is not deterministic, and the direction it fails in is the one that hides
    # a regression. Capacity is a banked axis (§9.6): a regression on any record
    # of an axis is a regression on that axis.
    #
    # The backend comparison below is now belt-and-braces: `check_matrix_coverage`
    # REFUSES a foreign-backend delta at the door, so this list comprehension can no
    # longer be the thing that makes one disappear. It is kept so that a future
    # caller reaching this function directly degrades to dropping rather than to
    # folding two backends' capacity onto one axis.
    present = [delta for delta in deltas if delta.backend == spec.backend]
    measured_kinds = {delta.kind for delta in present}
    missing = [kind for kind in spec.required_capacity_kinds
               if kind not in measured_kinds]
    regressed = [delta for delta in present if delta.regressed]
    blockers: list = []
    reasons: list = []
    if regressed:
        blockers.append(BLOCK_CAPACITY_REGRESSION)
        for delta in regressed:
            reasons.append(
                f"capacity {delta.kind} moved by {delta.delta} against the anchor "
                f"(event {delta.event_id}); capacity is a banked axis in its own right "
                "and on this host it is what makes the large models runnable at all")
    if missing:
        blockers.append(BLOCK_CAPACITY_DELTA_ABSENT)
        for kind in missing:
            reasons.append(
                f"no {kind} delta was measured for backend {spec.backend!r}, which the "
                "matrix declares as required")
    if regressed:
        return schemas.Check(schemas.FAIL, tuple(reasons)), tuple(blockers)
    if missing:
        return schemas.Check(schemas.COULD_NOT_CHECK, tuple(reasons)), tuple(blockers)
    return schemas.Check(schemas.PASS, tuple(
        f"{delta.kind} delta {delta.delta} (event {delta.event_id})"
        for delta in present)), ()


def _check_mechanism(champion: ChampionLineage,
                     confirmations: Sequence[MechanismConfirmation]) -> tuple:
    # ALL confirmations per member, not the last one. A `{member: conf}` mapping
    # let a later `confirmed=True` record overwrite an earlier `confirmed=False`
    # one for the same member, so an unconfirmed mechanism — the exact thing
    # `P-AK-SEARCH-1-A1` clause 1 blocks on — vanished from the signal and the
    # standing read `objective_met` with no blocker. A member is confirmed only if
    # nothing on record says it is not; a contradiction is resolved against the
    # candidate, because *"it got faster and I don't know why is a reason to keep
    # measuring, not to land."*
    by_member: dict = {}
    for conf in confirmations:
        by_member.setdefault(conf.member_candidate_id, []).append(conf)
    reasons: list = []
    unknown: list = []
    for member in champion.member_candidate_ids:
        confs = by_member.get(member)
        if not confs:
            unknown.append(
                f"member {member!r} has no cumulative mechanism confirmation on the "
                "composed champion; composition is where two mechanisms cancel, so a "
                "member's local receipt does not carry forward")
            continue
        for conf in confs:
            if not conf.confirmed:
                reasons.append(
                    f"member {member!r} predicted {conf.predicted_mechanism!r} and it was "
                    f"not confirmed on the composed champion (event {conf.event_id}): "
                    f"{conf.explanation}")
    for member in by_member:
        if member not in champion.member_candidate_ids:
            unknown.append(
                f"mechanism confirmation names {member!r}, which is not a member of this "
                "champion lineage")
    if reasons:
        return schemas.Check(schemas.FAIL, tuple(reasons) + tuple(unknown)), \
            (BLOCK_MECHANISM_UNCONFIRMED,)
    if unknown:
        return schemas.Check(schemas.COULD_NOT_CHECK, tuple(unknown)), \
            (BLOCK_MECHANISM_UNCONFIRMED,)
    if not champion.member_candidate_ids:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            "the champion lineage names no member candidates, so there is no predicted "
            "mechanism to confirm cumulatively",)), (BLOCK_MECHANISM_UNCONFIRMED,)
    return schemas.Check(schemas.PASS, tuple(
        f"{member}: {conf.predicted_mechanism} confirmed (event {conf.event_id})"
        for member in champion.member_candidate_ids
        for conf in by_member[member])), ()


def _check_non_target(cells: Sequence[T2Cell]) -> tuple:
    sentinels = [cell for cell in cells if cell.role in SENTINEL_ROLES]
    reasons: list = []
    unknown: list = []
    for cell in sentinels:
        standing = cell_standing(cell)
        if standing.non_inferiority.outcome == schemas.FAIL:
            reasons.append(
                f"sentinel {cell.cell_id!r} ({cell.role}): "
                + "; ".join(standing.non_inferiority.reasons))
        elif standing.non_inferiority.outcome == schemas.COULD_NOT_CHECK:
            unknown.append(
                f"sentinel {cell.cell_id!r} ({cell.role}): "
                + "; ".join(standing.non_inferiority.reasons))
    if reasons:
        return schemas.Check(schemas.FAIL, tuple(reasons) + tuple(unknown)), \
            (BLOCK_NON_TARGET_REGRESSION,)
    if unknown:
        return schemas.Check(schemas.COULD_NOT_CHECK, tuple(unknown)), \
            (BLOCK_NON_TARGET_REGRESSION,)
    return schemas.Check(schemas.PASS, (
        f"{len(sentinels)} sentinel(s) are non-inferior",)), ()


def _check_lineage_ordering(champion: ChampionLineage,
                            cells: Sequence[T2Cell],
                            deltas: Sequence[CapacityDelta] = (),
                            confirmations: Sequence[MechanismConfirmation] = ()
                            ) -> tuple:
    """*"Gathered AFTER the candidate entered the lineage"* — over EVERY record.

    The clause is about evidence, not about throughput evidence. Capacity deltas
    and mechanism confirmations are T2 matrix requirements in their own right
    (§9.7), they are read here as statements about the composed champion, and both
    were exempt: a RAM delta timestamped before the champion existed satisfied
    `CAPACITY_DELTA_ABSENT`, and a mechanism receipt from before the composition
    satisfied the CUMULATIVE confirmation that `MechanismConfirmation`'s own
    docstring says a member's local receipt cannot carry forward.

    One check rather than three, because "after the lineage was entered" is one
    ordering over one instant, and three copies of it would eventually disagree
    about which records it covers.
    """
    entered = champion.entered_at
    early: list = []
    for cell in cells:
        if cell.measured_instant < entered:
            early.append(
                f"cell {cell.cell_id!r} was measured at {cell.measured_at}")
    for delta in deltas:
        if delta.measured_instant < entered:
            early.append(
                f"capacity delta {delta.kind!r} (event {delta.event_id}) was measured at "
                f"{delta.measured_at}")
    for conf in confirmations:
        if conf.measured_instant < entered:
            early.append(
                f"mechanism confirmation for {conf.member_candidate_id!r} (event "
                f"{conf.event_id}) was measured at {conf.measured_at}")
    if early:
        return schemas.Check(schemas.FAIL, tuple(
            f"{item}, before the candidate entered the lineage at "
            f"{champion.entered_lineage_at}; the readiness signal is computed only from "
            "confirmation-stratum evidence gathered AFTER the candidate entered the "
            "lineage, and a receipt that predates the composition is not a receipt about "
            "it" for item in early)), \
            (BLOCK_CONFIRMATION_EVIDENCE_PREDATES_LINEAGE,)
    return schemas.Check(schemas.PASS), ()


def _check_anchor_agreement(champion: ChampionLineage,
                            cells: Sequence[T2Cell]) -> tuple:
    reasons: list = []
    unknown: list = []
    for cell in cells:
        anchor = cell.anchor
        if anchor is None:
            unknown.append(
                f"cell {cell.cell_id!r} binds no anchor, so its ratio has no denominator")
            continue
        # THREE outcomes, not two. `identity_matches` gained a third on
        # 2026-08-04 (`AnchorIdentity.tool`: one side names the binary its digest
        # came off and the other does not). `!= PASS` was written when the
        # comparator answered only PASS or FAIL, and it files that third outcome
        # under BLOCK_ANCHOR_MOVED — a record asserting that the denominator was
        # REBUILT, on evidence that says only that one side did not say which
        # tool it was. `unknown`/BLOCK_ANCHOR_ABSENT is this function's own
        # existing bucket for exactly that, and it is where an unobserved
        # component belongs: a detected difference is a fact, an unobserved one
        # is not, and the two must not share a blocker name.
        match = champion.anchor.identity_matches(anchor)
        if match.outcome == schemas.FAIL:
            reasons.append(
                f"cell {cell.cell_id!r}: " + "; ".join(match.reasons)
                + ". A rebuilt anchor is a different anchor, and a comparison whose "
                "denominator no longer resolves is superseded, not reinterpreted "
                "(AK-D22)")
        elif match.outcome != schemas.PASS:
            unknown.append(
                f"cell {cell.cell_id!r}: " + "; ".join(match.reasons)
                + ". Nothing about this cell's anchor DISAGREES with the champion's; "
                "a component of the identity was not observed, which is not evidence "
                "that the anchor moved")
    if reasons:
        return schemas.Check(schemas.FAIL, tuple(reasons) + tuple(unknown)), \
            (BLOCK_ANCHOR_MOVED,)
    if unknown:
        return schemas.Check(schemas.COULD_NOT_CHECK, tuple(unknown)), \
            (BLOCK_ANCHOR_ABSENT,)
    return schemas.Check(schemas.PASS, (
        f"every cell names anchor {champion.anchor.short()}",)), ()


def check_matrix_coverage(*, spec: T2MatrixSpec, champion: ChampionLineage,
                          cells: Sequence[T2Cell],
                          capacity_deltas: Sequence[CapacityDelta] = (),
                          mechanisms: Sequence[MechanismConfirmation] = ()
                          ) -> MatrixCoverage:
    """Check the T2 matrix §9.7 describes against the matrix that was run.

    A PUBLIC entry point, so it holds the same two admissions AK-D12 and §9.7 need
    holding — one backend, one composed champion — rather than relying on
    `compute_readiness()` having held them upstream. A guarantee enforced one
    function deep is a guarantee for one caller: this function was directly
    callable with a `llama_gpu` cell in a `llama_cpu` matrix, and it counted it
    toward coverage, co-residency and the sentinel set.

    The one-backend admission covers CAPACITY DELTAS as well as cells, because a
    delta names a backend too and `_check_capacity` reads it through a filter
    rather than a refusal. A `llama_gpu` VRAM regression offered to a `llama_cpu`
    matrix was dropped with no blocker and the axis reported PASS — a
    success-shaped result over a record that says capacity was lost — while
    `_check_lineage_ordering` gated on the very same record. Refuse it at the door
    and both siblings agree about what they were handed.
    """
    if not isinstance(spec, T2MatrixSpec):
        raise MatrixSpecInvalid("check_matrix_coverage() takes a T2MatrixSpec")
    if not isinstance(champion, ChampionLineage):
        raise ChampionMismatch("check_matrix_coverage() takes a ChampionLineage")
    cells = _tuple_of(cells, "cells", T2Cell)
    deltas = _tuple_of(capacity_deltas, "capacity_deltas", CapacityDelta)
    confirmations = _tuple_of(mechanisms, "mechanisms", MechanismConfirmation)
    _require_one_backend(spec.backend, cells, _describe_cell)
    _require_one_backend(spec.backend, deltas, _describe_capacity_delta)
    _require_composed_champion(champion, cells)

    blockers: list = []

    coverage = _check_coverage(spec, cells)
    if coverage.outcome != schemas.PASS:
        blockers.append(BLOCK_COVERAGE_GAP)
    repetitions = _check_repetitions(spec, cells)
    if repetitions.outcome != schemas.PASS:
        blockers.append(BLOCK_REPETITIONS_NOT_STRONGER_THAN_T1)
    sentinels = _check_sentinels(spec, cells)
    if sentinels.outcome != schemas.PASS:
        blockers.append(BLOCK_SENTINEL_SET_NOT_BROADER)
    co_resident = _check_co_resident(spec, cells)
    if co_resident.outcome != schemas.PASS:
        blockers.append(BLOCK_CO_RESIDENT_CELL_ABSENT)
    capacity, capacity_blockers = _check_capacity(spec, deltas)
    mechanism, mechanism_blockers = _check_mechanism(champion, confirmations)
    non_target, non_target_blockers = _check_non_target(cells)
    ordering, ordering_blockers = _check_lineage_ordering(champion, cells, deltas,
                                                          confirmations)
    anchors, anchor_blockers = _check_anchor_agreement(champion, cells)

    for extra in (capacity_blockers, mechanism_blockers, non_target_blockers,
                  ordering_blockers, anchor_blockers):
        for blocker in extra:
            if blocker not in blockers:
                blockers.append(blocker)

    overall = _combine([coverage, repetitions, sentinels, co_resident, capacity,
                        mechanism, non_target, ordering, anchors])
    return MatrixCoverage(
        backend=spec.backend, coverage=coverage, repetitions=repetitions,
        sentinels=sentinels, co_resident=co_resident, capacity=capacity,
        mechanism=mechanism, non_target=non_target, lineage_ordering=ordering,
        anchor_agreement=anchors, overall=overall, blockers=tuple(blockers))


# =============================================================================
# Derived: the phase trade, which the operator decides
# =============================================================================

@dataclass(frozen=True)
class PhaseTradeAssessment:
    """Whether an observed trade matches what the campaign pre-declared.

    It is deliberately impossible for this object to convert a standing into
    `objective_met`. §1.6: a phase trade *"is an operator decision at freeze time,
    not a controller decision"*, and `P-AK-SEARCH-1` denial 5 forbids a readiness
    signal being cited as a reason a human-only write may happen automatically.
    So the strongest thing this can say is `within_predeclared_band`, which is a
    fact about the measurement and the manifest, plus
    `operator_decision_required=True`.
    """

    status: str
    regressing_phase: Optional[str]
    observed: Optional[float]
    band: Optional[tuple]
    expected_gain: Optional[float]
    observed_gain: Optional[float]
    roles: tuple
    reasons: tuple
    operator_decision_required: bool

    STATUS_NOT_APPLICABLE: ClassVar[str] = "not_applicable"
    STATUS_NOT_PREDECLARED: ClassVar[str] = "not_predeclared"
    STATUS_WITHIN_BAND: ClassVar[str] = "within_predeclared_band"
    STATUS_OUTSIDE_BAND: ClassVar[str] = "outside_predeclared_band"

    def to_dict(self) -> dict:
        return {
            "status": self.status, "regressing_phase": self.regressing_phase,
            "observed": self.observed,
            "band": None if self.band is None else list(self.band),
            "expected_gain": self.expected_gain, "observed_gain": self.observed_gain,
            "roles": list(self.roles), "reasons": list(self.reasons),
            "operator_decision_required": self.operator_decision_required,
        }


def _assess_phase_trade(objective: ObjectiveSpec,
                        phases: Sequence[PhaseStanding]) -> PhaseTradeAssessment:
    regressing = [standing for standing in phases
                  if standing.non_inferior.outcome == schemas.FAIL]
    if not regressing:
        return PhaseTradeAssessment(
            status=PhaseTradeAssessment.STATUS_NOT_APPLICABLE, regressing_phase=None,
            observed=None, band=None, expected_gain=None, observed_gain=None,
            roles=(), reasons=("no phase carries a detectable degradation",),
            operator_decision_required=False)

    names = [standing.phase for standing in regressing]
    exception = objective.phase_trade_exception
    if exception is None:
        return PhaseTradeAssessment(
            status=PhaseTradeAssessment.STATUS_NOT_PREDECLARED,
            regressing_phase=names[0], observed=None, band=None, expected_gain=None,
            observed_gain=None, roles=(),
            reasons=(f"phase(s) {names} carry a detectable degradation and the campaign "
                     "pre-declared no phase-trade exception; a trade discovered after "
                     "measuring is not an exception, it is a regression",),
            operator_decision_required=False)

    if len(regressing) > 1 or names[0] != exception.regressing_phase:
        return PhaseTradeAssessment(
            status=PhaseTradeAssessment.STATUS_OUTSIDE_BAND,
            regressing_phase=names[0], observed=None, band=exception.band,
            expected_gain=exception.expected_gain, observed_gain=None,
            roles=exception.roles,
            reasons=(f"the exception pre-declared phase "
                     f"{exception.regressing_phase!r} and the degradation is in {names}; "
                     "the exception names the exact regression band, phase and roles, so "
                     "it does not stretch to another phase",),
            operator_decision_required=True)

    standing = regressing[0]
    # `_orderable_value`, never `.figure.value`: a phase trade asks whether a
    # MAGNITUDE sits inside a pre-declared band, and "no orderable effect" is not
    # a magnitude outside it. A parity figure answers `None` here and the band
    # question is reported as unplaceable, which is what it is.
    observed = _orderable_value(standing.figure)
    low, high = exception.band
    gains = [_orderable_value(other.figure) for other in phases
             if other.phase != exception.regressing_phase and other.figure is not None
             and other.improved.outcome == schemas.PASS]
    gains = [gain for gain in gains if gain is not None]
    observed_gain = max(gains) if gains else None

    reasons: list = []
    inside = observed is not None and low <= observed <= high
    if observed is None:
        reasons.append(
            f"phase {standing.phase!r} carries a detectable degradation but no protected "
            "figure, so the observed regression cannot be placed inside the band")
    elif not inside:
        reasons.append(
            f"the observed regression {observed} is outside the pre-declared band "
            f"{exception.band}")
    else:
        reasons.append(
            f"the observed regression {observed} lies inside the pre-declared band "
            f"{exception.band}")
    gain_met = observed_gain is not None and observed_gain >= exception.expected_gain
    if observed_gain is None:
        reasons.append(
            "no other phase carries a confirmed improvement, so the gain the trade was "
            "declared to buy was not observed")
    elif not gain_met:
        reasons.append(
            f"the observed gain {observed_gain} is below the pre-declared expected gain "
            f"{exception.expected_gain}")
    else:
        reasons.append(
            f"the observed gain {observed_gain} meets the pre-declared expected gain "
            f"{exception.expected_gain}")
    reasons.append(
        "This assessment reports; it does not decide. A phase trade is an operator "
        "decision at freeze time, not a controller decision (§1.6), and a readiness "
        "signal is not a freeze trigger (P-AK-SEARCH-1 denial 5)")

    status = (PhaseTradeAssessment.STATUS_WITHIN_BAND if inside and gain_met
              else PhaseTradeAssessment.STATUS_OUTSIDE_BAND)
    return PhaseTradeAssessment(
        status=status, regressing_phase=standing.phase, observed=observed,
        band=exception.band, expected_gain=exception.expected_gain,
        observed_gain=observed_gain, roles=exception.roles, reasons=tuple(reasons),
        operator_decision_required=True)


# =============================================================================
# Derived: capability objectives (§9.8)
# =============================================================================

@dataclass(frozen=True)
class CapabilityStanding:
    """Whether a capability objective is admitted to the readiness signal."""

    objective_id: str
    backend: str
    admitted: schemas.Check
    utility_model_fixed_at_campaign_start: schemas.Check
    event_id: str

    def to_dict(self) -> dict:
        return {
            "objective_id": self.objective_id, "backend": self.backend,
            "event_id": self.event_id,
            "admitted": {"outcome": self.admitted.outcome,
                         "reasons": list(self.admitted.reasons)},
            "utility_model_fixed_at_campaign_start": {
                "outcome": self.utility_model_fixed_at_campaign_start.outcome,
                "reasons": list(self.utility_model_fixed_at_campaign_start.reasons)},
        }


def _capability_standing(objective: CapabilityObjective,
                         campaign_start_utility_model_sha256: Optional[str]
                         ) -> CapabilityStanding:
    if campaign_start_utility_model_sha256 is None:
        fixed = schemas.Check(schemas.COULD_NOT_CHECK, (
            "the campaign manifest's campaign-start utility-model digest was not "
            "supplied, so 'fixed at campaign start, not invented after observing the "
            "candidate' cannot be checked",))
    elif campaign_start_utility_model_sha256 == objective.utility_model_sha256:
        fixed = schemas.Check(schemas.PASS, (
            f"utility model {_short(objective.utility_model_sha256)} matches the "
            "campaign-start digest",))
    else:
        fixed = schemas.Check(schemas.FAIL, (
            f"utility model {_short(objective.utility_model_sha256)} does not match the "
            f"campaign-start digest "
            f"{_short(campaign_start_utility_model_sha256)}; §9.8 requires the utility "
            "model to be fixed at campaign start, not invented after observing the "
            "candidate",))
    admitted = _combine([fixed, objective.runnable, objective.correctness_floor,
                         objective.quality_floor, objective.resource_budget])
    return CapabilityStanding(
        objective_id=objective.objective_id, backend=objective.backend,
        admitted=admitted, utility_model_fixed_at_campaign_start=fixed,
        event_id=objective.event_id)


# =============================================================================
# The readiness signal
# =============================================================================

@dataclass(frozen=True)
class ReadinessSignal:
    """One backend's standing against §1.6. Never two (AK-D12).

    There is no scalar on this object, and there is no method that produces one.
    Every number it carries is a `ReadinessFigure`, which is one named cell's own
    estimate from one named event.
    """

    backend: str
    source_tree: str
    campaign_id: str
    champion_candidate_id: str
    anchor: api.AnchorIdentity
    objective: ObjectiveSpec
    phases: tuple
    matrix: MatrixCoverage
    phase_trade: PhaseTradeAssessment
    capabilities: tuple
    standing: str
    blockers: tuple
    controls_marker: str
    evaluator_bundle_sha256: str
    statistics_module_id: str
    computed_at: str
    improvement_backend_wide: schemas.Check
    improvement_per_protected_cell: schemas.Check

    #: AK-D3 and denial 5, as a field a reader can assert on.
    is_trigger: ClassVar[bool] = False
    signal_class: ClassVar[str] = SIGNAL_CLASS
    reducer_id: ClassVar[str] = MODULE_ID

    def __post_init__(self) -> None:
        if self.standing not in STANDINGS:
            raise CellInadmissible(
                f"signal.standing: {self.standing!r} is not one of {list(STANDINGS)}")
        if self.controls_marker not in CONTROLS_MARKERS:
            raise CellInadmissible(
                f"signal.controls_marker: {self.controls_marker!r} is not one of "
                f"{list(CONTROLS_MARKERS)}; the protocol requires the readiness signal "
                "computed by a four-control campaign to carry the same marker its "
                "records do, so it is never silently absent")
        for blocker in self.blockers:
            if blocker not in BLOCKERS:
                raise CellInadmissible(
                    f"signal.blockers: {blocker!r} is not a declared blocking condition "
                    f"{list(BLOCKERS)}")
        # Invariant 14, structurally: the standing is RE-DERIVED from this object's
        # own phase standings, matrix coverage, capability standings and phase-trade
        # assessment, and a disagreement raises. Without this, `standing` was a
        # field, and a field can be set: `dataclasses.replace(signal,
        # standing='objective_met', blockers=())` produced a signal that RENDERED as
        # met over phases that said otherwise. `api.Verdict.__post_init__` answers
        # the identical hole the identical way.
        if not isinstance(self.objective, ObjectiveSpec):
            raise CellInadmissible("signal.objective must be an ObjectiveSpec")
        if not isinstance(self.matrix, MatrixCoverage):
            raise CellInadmissible("signal.matrix must be a MatrixCoverage")
        if not isinstance(self.phase_trade, PhaseTradeAssessment):
            raise CellInadmissible("signal.phase_trade must be a PhaseTradeAssessment")
        _tuple_of(self.phases, "signal.phases", PhaseStanding)
        _tuple_of(self.capabilities, "signal.capabilities", CapabilityStanding)
        for name in ("improvement_backend_wide", "improvement_per_protected_cell"):
            if not isinstance(getattr(self, name), schemas.Check):
                raise CellInadmissible(f"signal.{name} must be a schemas.Check")
        derived_standing, derived_blockers = _derive_signal_state(
            objective=self.objective, phases=self.phases, matrix=self.matrix,
            capabilities=self.capabilities, phase_trade=self.phase_trade,
            improvement_backend_wide=self.improvement_backend_wide,
            improvement_per_protected_cell=self.improvement_per_protected_cell)
        if self.standing != derived_standing or tuple(self.blockers) != derived_blockers:
            raise StandingNotDerived(
                f"signal.standing/blockers do not follow from this signal's own "
                f"evidence: stored {self.standing!r} with {list(self.blockers)}, derived "
                f"{derived_standing!r} with {list(derived_blockers)}. Readiness is "
                "computed from records by a deterministic reducer; a standing that "
                "originates anywhere else is INVALID (§4 invariant 14, P-AK-SEARCH-1 "
                "authorization 5)")

    @property
    def figures(self) -> tuple:
        return tuple(standing.figure for standing in self.phases
                     if standing.figure is not None)

    def figure_for(self, phase: str) -> Optional[Any]:
        """The phase's figure: `None`, a `ParityFigure`, or a `ReadinessFigure`."""
        for standing in self.phases:
            if standing.phase == phase:
                return standing.figure
        return None

    def to_dict(self) -> dict:
        return {
            "reducer_id": self.reducer_id, "signal_class": self.signal_class,
            "is_trigger": self.is_trigger,
            "backend": self.backend, "source_tree": self.source_tree,
            "campaign_id": self.campaign_id,
            "champion_candidate_id": self.champion_candidate_id,
            "anchor": self.anchor.to_dict(), "objective": self.objective.to_dict(),
            "phases": [standing.to_dict() for standing in self.phases],
            "matrix": self.matrix.to_dict(),
            "phase_trade": self.phase_trade.to_dict(),
            "capabilities": [cap.to_dict() for cap in self.capabilities],
            "standing": self.standing, "blockers": list(self.blockers),
            "controls_marker": self.controls_marker,
            "evaluator_bundle_sha256": self.evaluator_bundle_sha256,
            "statistics_module_id": self.statistics_module_id,
            "computed_at": self.computed_at,
            "improvement_backend_wide": {
                "outcome": self.improvement_backend_wide.outcome,
                "reasons": list(self.improvement_backend_wide.reasons)},
            "improvement_per_protected_cell": {
                "outcome": self.improvement_per_protected_cell.outcome,
                "reasons": list(self.improvement_per_protected_cell.reasons)},
            "tier": TIER,
        }


def _derive_signal_state(*, objective: ObjectiveSpec, phases: Sequence[PhaseStanding],
                         matrix: MatrixCoverage,
                         capabilities: Sequence[CapabilityStanding],
                         phase_trade: PhaseTradeAssessment,
                         improvement_backend_wide: schemas.Check,
                         improvement_per_protected_cell: schemas.Check) -> tuple:
    """The ONE place a standing and its blocker list come from.

    Shared by `compute_readiness()`, which produces the signal, and by
    `ReadinessSignal.__post_init__`, which re-derives it and refuses to hold a
    standing that disagrees. Two callers of one derivation rather than one
    derivation and one assertion: an independent re-implementation would be a
    second reducer of the standing, and the two would eventually disagree about
    which one is the readiness signal.
    """
    improvement = (improvement_backend_wide
                   if objective.improvement_quantifier == QUANTIFIER_BACKEND_WIDE
                   else improvement_per_protected_cell)

    blockers: list = list(matrix.blockers)
    for standing in phases:
        for blocker in standing.blockers:
            if blocker not in blockers:
                blockers.append(blocker)
    if improvement.outcome == schemas.COULD_NOT_CHECK:
        # Under the DECLARED quantifier there is not enough improvement evidence
        # to answer §1.6's second half either way. That is a blocking condition on
        # the backend, not on any one phase.
        if BLOCK_IMPROVEMENT_EVIDENCE_ABSENT not in blockers:
            blockers.append(BLOCK_IMPROVEMENT_EVIDENCE_ABSENT)
    for capability in capabilities:
        if capability.utility_model_fixed_at_campaign_start.outcome != schemas.PASS:
            if BLOCK_CAPABILITY_UTILITY_MODEL_DRIFTED not in blockers:
                blockers.append(BLOCK_CAPABILITY_UTILITY_MODEL_DRIFTED)
    if phase_trade.operator_decision_required:
        if BLOCK_PHASE_TRADE_DECISION_REQUIRED not in blockers:
            blockers.append(BLOCK_PHASE_TRADE_DECISION_REQUIRED)

    overall = _combine([standing.non_inferior for standing in phases]
                       + [improvement, matrix.overall])
    if overall.outcome == schemas.PASS and not blockers:
        standing_value = STANDING_MET
    elif overall.outcome == schemas.FAIL:
        standing_value = STANDING_NOT_MET
    else:
        standing_value = STANDING_UNDETERMINED
    return standing_value, tuple(blockers)


def _improvement_quantifiers(phases: Sequence[PhaseStanding]) -> tuple:
    """Both readings of §1.6's improvement half, computed and reported.

    The declared one decides the standing; the other rides along so an operator
    reading the package can see what the alternative reading would have said.
    """
    improved_phases = [standing for standing in phases
                       if standing.improved.outcome == schemas.PASS]
    if improved_phases:
        backend_wide = schemas.Check(schemas.PASS, tuple(
            f"phase {standing.phase!r} improved" for standing in improved_phases))
    elif all(standing.improved.outcome == schemas.COULD_NOT_CHECK
             for standing in phases):
        backend_wide = schemas.Check(schemas.COULD_NOT_CHECK, (
            "no phase carries improvement evidence at all",))
    else:
        backend_wide = schemas.Check(schemas.FAIL, (
            "no phase improved; §1.6 requires at least one to improve, otherwise a "
            "release buys nothing",))

    by_cell: dict = {}
    for standing in phases:
        for cell in standing.cells:
            current = by_cell.get(cell.cell_id)
            outcome = cell.improvement.outcome
            if current is None or outcome == schemas.PASS:
                by_cell[cell.cell_id] = outcome
    if not by_cell:
        per_cell = schemas.Check(schemas.COULD_NOT_CHECK, (
            "no protected cell was measured, so the per-cell reading is unevaluable",))
    elif all(outcome == schemas.PASS for outcome in by_cell.values()):
        per_cell = schemas.Check(schemas.PASS, tuple(
            f"cell {cell_id!r} improves in at least one phase"
            for cell_id in sorted(by_cell)))
    elif any(outcome == schemas.FAIL for outcome in by_cell.values()):
        per_cell = schemas.Check(schemas.FAIL, tuple(
            f"cell {cell_id!r} does not improve in any phase"
            for cell_id, outcome in sorted(by_cell.items()) if outcome != schemas.PASS))
    else:
        per_cell = schemas.Check(schemas.COULD_NOT_CHECK, tuple(
            f"cell {cell_id!r} carries no improvement evidence"
            for cell_id, outcome in sorted(by_cell.items())
            if outcome == schemas.COULD_NOT_CHECK))
    return backend_wide, per_cell


def compute_readiness(*, backend: str, campaign_id: str, champion: ChampionLineage,
                      objective: ObjectiveSpec, spec: T2MatrixSpec,
                      cells: Sequence[T2Cell],
                      controls_marker: str,
                      evaluator_bundle_sha256: str,
                      computed_at: str,
                      capacity_deltas: Sequence[CapacityDelta] = (),
                      mechanisms: Sequence[MechanismConfirmation] = (),
                      capability_objectives: Sequence[CapabilityObjective] = (),
                      campaign_start_utility_model_sha256: Optional[str] = None,
                      reference: Optional[ReferencePolicy] = None) -> ReadinessSignal:
    """Compute ONE backend's readiness signal from journaled T2 records.

    Takes one backend and refuses a cell from any other. That is the structural
    form of AK-D12: a weighted scalar folding CPU and GPU cells is not merely
    forbidden here, it is unreachable, because no function in this module is ever
    handed two backends' measurements.

    Raises on unusable MATERIAL — a member candidate's cell, selection-stratum
    evidence, a cell citing a protocol its phase does not declare. Returns a
    signal with blockers for everything that is a RUN outcome, because a voided
    or inconclusive cell is evidence that has to be journaled and reported, not
    an exception in the caller's face.
    """
    _backend(backend, "backend")
    _text(campaign_id, "campaign_id")
    if not campaign_id.startswith("ak-"):
        raise CellInadmissible(f"campaign_id: {campaign_id!r} must start with 'ak-'")
    if not isinstance(champion, ChampionLineage):
        raise ChampionMismatch("compute_readiness() takes a ChampionLineage")
    if not isinstance(objective, ObjectiveSpec):
        raise CellInadmissible("compute_readiness() takes an ObjectiveSpec")
    if not isinstance(spec, T2MatrixSpec):
        raise MatrixSpecInvalid("compute_readiness() takes a T2MatrixSpec")
    if objective.backend != backend or spec.backend != backend:
        raise CrossBackendComposite(
            f"the objective names {objective.backend!r} and the matrix names "
            f"{spec.backend!r} for a readiness computation on {backend!r}. A readiness "
            "signal is computed for ONE backend: a scalar folding P-BENCH-1, "
            "P-BENCH-PREFILL-1 and P-GPU-1 cells is analysis and cannot gate "
            "(MEASUREMENT.md:83-84), and a reconstructed net is forbidden outright "
            "(gpu-cross-device.md:106-111)")
    expected_tree = schemas.SOURCE_TREE_BY_BACKEND.get(backend)
    if expected_tree is not None and champion.source_tree != expected_tree:
        raise ChampionMismatch(
            f"backend {backend!r} lives in source tree {expected_tree!r} but the champion "
            f"names {champion.source_tree!r}; champions are per SOURCE TREE (§1.5, "
            "AK-D11)")
    # Precondition 5's pinned bundle hash, validated as a DIGEST rather than as a
    # non-empty string. `_sha256` exists because *"a fabricated hash is
    # indistinguishable from a measured one to every downstream reader"*, and this
    # field is rendered into the operator's line as `eval=<12 hex>` — a placeholder
    # or a free-text label there looks exactly like a pinned evaluator.
    _sha256(evaluator_bundle_sha256, "evaluator_bundle_sha256")
    _instant(computed_at, "computed_at")

    cells = _tuple_of(cells, "cells", T2Cell)
    _require_one_backend(backend, cells, _describe_cell)
    _require_composed_champion(champion, cells)
    _require_declared_protocol(objective, cells)

    # §1.6 is a conjunction over BOTH phases the backend declares — *"both prefill
    # and decode throughput must be non-inferior … and at least one must improve"*
    # — so an objective naming a strict subset satisfies it by deleting a conjunct.
    # A decode-only objective on `llama_cpu` reached `objective_met` with an empty
    # blocker list and no record anywhere that prefill was never asked about.
    # Checked here, after the per-cell loop: the same misconfiguration usually
    # orphans a cell too, and "cell X is in a phase this objective does not
    # declare" names the specific record, which is the more actionable refusal.
    declared_phases = schemas.PHASES_BY_BACKEND.get(backend)
    if declared_phases is not None and set(objective.phases) != set(declared_phases):
        raise CellInadmissible(
            f"objective.phases {list(objective.phases)} is not the phase set backend "
            f"{backend!r} declares ({sorted(declared_phases)}). §1.6 quantifies over "
            "both prefill and decode; an objective that drops a phase does not meet a "
            "weaker objective, it reports on a different one, and the dropped phase "
            "leaves no blocker behind to say it was never measured")

    matrix = check_matrix_coverage(spec=spec, champion=champion, cells=cells,
                                   capacity_deltas=capacity_deltas,
                                   mechanisms=mechanisms)

    phases = tuple(
        phase_standing(backend=backend, phase=phase, objective=objective, cells=cells,
                       reference=reference, effect_scale=spec.effect_scale)
        for phase in objective.phases)

    backend_wide, per_cell = _improvement_quantifiers(phases)

    capabilities = tuple(
        _capability_standing(objective_item, campaign_start_utility_model_sha256)
        for objective_item in _tuple_of(capability_objectives, "capability_objectives",
                                        CapabilityObjective))
    for capability in capabilities:
        if capability.backend != backend:
            raise CrossBackendComposite(
                f"capability objective {capability.objective_id!r} names backend "
                f"{capability.backend!r}, not {backend!r}")

    phase_trade = _assess_phase_trade(objective, phases)

    standing_value, blockers = _derive_signal_state(
        objective=objective, phases=phases, matrix=matrix, capabilities=capabilities,
        phase_trade=phase_trade, improvement_backend_wide=backend_wide,
        improvement_per_protected_cell=per_cell)

    return ReadinessSignal(
        backend=backend, source_tree=champion.source_tree, campaign_id=campaign_id,
        champion_candidate_id=champion.combined_candidate_id, anchor=champion.anchor,
        objective=objective, phases=phases, matrix=matrix, phase_trade=phase_trade,
        capabilities=capabilities, standing=standing_value, blockers=blockers,
        controls_marker=controls_marker,
        evaluator_bundle_sha256=evaluator_bundle_sha256,
        statistics_module_id=stats.STATISTICS_MODULE_ID, computed_at=computed_at,
        improvement_backend_wide=backend_wide,
        improvement_per_protected_cell=per_cell)


# =============================================================================
# The report — several independent signals, and never a number over them
# =============================================================================

@dataclass(frozen=True)
class ReadinessReport:
    """Several per-backend signals, side by side, with nothing folded over them.

    A mapping and not a score. `MEASUREMENT.md:83-84` makes a fold across
    protocols analysis rather than a claim, and `gpu-cross-device.md:106-111`
    forbids a reconstructed net outright — so the report exposes the backends and
    stops.

    It carries ONE `campaign_id` over its signals, so it is also the one place a
    foreign campaign's signal can acquire this campaign's label. Every signal must
    therefore name this campaign: `P-AK-SEARCH-1` denial 4 confines consumption to
    the campaign that produced the record, and a later campaign re-derives its own
    calibration — so a campaign-B signal collected here would be reported against a
    floor and a threshold it was never measured under, by the reducer itself.
    """

    campaign_id: str
    computed_at: str
    signals: tuple

    def __post_init__(self) -> None:
        _text(self.campaign_id, "report.campaign_id")
        if not self.campaign_id.startswith("ak-"):
            raise CellInadmissible(
                f"report.campaign_id: {self.campaign_id!r} must start with 'ak-'; the "
                "report's own door holds the same shape `compute_readiness()` holds, "
                "otherwise the label over the signals is looser than the label on them")
        _instant(self.computed_at, "report.computed_at")
        signals = _tuple_of(self.signals, "report.signals", ReadinessSignal)
        seen: list = []
        for signal in signals:
            if signal.campaign_id != self.campaign_id:
                raise CampaignMismatch(
                    f"the report is labelled campaign {self.campaign_id!r} and carries a "
                    f"signal computed for campaign {signal.campaign_id!r} (backend "
                    f"{signal.backend!r}, champion {signal.champion_candidate_id!r}). "
                    "Consumption is confined to the campaign that produced the record, "
                    "and a later campaign may use a prior one for hypothesis formation "
                    "only — never to contribute to readiness — because a reused record "
                    "is scored against a floor and a threshold it was never measured "
                    "under (P-AK-SEARCH-1 denial 4)")
            if signal.backend in seen:
                raise CrossBackendComposite(
                    f"two readiness signals for backend {signal.backend!r}; a backend has "
                    "one standing, and two would invite a reader to combine them")
            seen.append(signal.backend)

    @property
    def backends(self) -> tuple:
        return tuple(signal.backend for signal in self.signals)

    def signal_for(self, backend: str) -> Optional[ReadinessSignal]:
        for signal in self.signals:
            if signal.backend == backend:
                return signal
        return None

    def to_dict(self) -> dict:
        return {"campaign_id": self.campaign_id, "computed_at": self.computed_at,
                "signals": [signal.to_dict() for signal in self.signals],
                "reducer_id": MODULE_ID, "signal_class": SIGNAL_CLASS,
                "is_trigger": False}


def compute_readiness_report(*, campaign_id: str, computed_at: str,
                             signals: Sequence[ReadinessSignal]) -> ReadinessReport:
    """Collect independently computed per-backend signals. It combines nothing."""
    return ReadinessReport(campaign_id=campaign_id, computed_at=computed_at,
                           signals=tuple(signals))


@dataclass(frozen=True)
class CrossBackendAnalysisView:
    """A labelled analysis view. It never gates, and it carries no aggregate.

    §1.6: *"cross-backend roll-ups may be reported to the operator as a labelled
    analysis view. They never gate."* Annex K says the same in its comparison
    scope: *"A single Annex K protocol id spanning several backends does NOT make
    a cross-backend comparison a within-protocol comparison."*

    Every row keeps its own protocol id, so a reader can see that the rows are not
    commensurable. `as_gate()` raises rather than returning anything.
    """

    label: str
    rows: tuple
    gates: bool = False

    LABEL: ClassVar[str] = "LABELLED ANALYSIS VIEW — CROSS-BACKEND, NEVER GATES"

    def __post_init__(self) -> None:
        if self.label != self.LABEL:
            raise CrossBackendComposite(
                f"a cross-backend view must carry the label {self.LABEL!r}; an unlabelled "
                "cross-backend comparison is what MEASUREMENT.md:83-84 forbids")
        if self.gates is not False:
            raise CrossBackendComposite(
                "a cross-backend view cannot gate. Cross-backend roll-ups are labelled "
                "analysis and never gate")

    def as_gate(self) -> None:
        raise CrossBackendComposite(
            "this is an analysis view; it cannot be read as a gate. A comparison across "
            "P-BENCH-1, P-BENCH-PREFILL-1 and P-GPU-1 cells is analysis, not a claim "
            "(MEASUREMENT.md:83-84), and the net between backends is measured directly or "
            "not at all (gpu-cross-device.md:106-111)")

    def to_dict(self) -> dict:
        return {"label": self.label, "gates": False, "rows": [dict(r) for r in self.rows]}


def cross_backend_analysis_view(report: ReadinessReport) -> CrossBackendAnalysisView:
    """Lay the per-backend standings side by side, labelled, with no aggregate."""
    if not isinstance(report, ReadinessReport):
        raise CellInadmissible("cross_backend_analysis_view() takes a ReadinessReport")
    rows: list = []
    for signal in report.signals:
        for standing in signal.phases:
            figure = standing.figure
            rows.append({
                "backend": signal.backend,
                "phase": standing.phase,
                "protocol_id": standing.protocol_id,
                "standing": signal.standing,
                "non_inferior": standing.non_inferior.outcome,
                "improved": standing.improved.outcome,
                # Three states, three sets of cells in the row. A parity figure has
                # no selected cell and no value, so it reports its own kind and its
                # own parity count rather than a `None` that reads like an
                # unmeasured phase.
                "figure_kind": None if figure is None else figure.kind,
                "figure_cell_id": (figure.cell_id
                                   if isinstance(figure, ReadinessFigure) else None),
                "figure_value": _orderable_value(figure),
                "figure_event_id": (
                    None if figure is None
                    else figure.event_id if isinstance(figure, ReadinessFigure)
                    else figure.sensitivity_event_id),
                "figure_parity_cell_count": (None if figure is None
                                             else figure.parity_cell_count),
                "blockers": list(standing.blockers),
                "not_commensurable_with_other_rows": True,
            })
    return CrossBackendAnalysisView(label=CrossBackendAnalysisView.LABEL,
                                    rows=tuple(rows))


def composite_readiness(*_args: Any, **_kwargs: Any) -> None:
    """A deliberate dead end, so the idea is found and refused rather than written.

    AK-D12: *"Objective is per-backend, per-phase non-inferiority plus improvement
    at production-optimal recipes"*, because *"a cross-device composite is
    forbidden by `MEASUREMENT.md:83-84` and `gpu-cross-device.md:106-111`."*
    Anyone grepping this package for a composite lands here and reads why there
    is not one.
    """
    raise CrossBackendComposite(
        "there is no composite readiness figure, and there is no way to compute one "
        "here. A weighted scalar folding CPU and GPU cells is forbidden twice over: "
        "MEASUREMENT.md:83-84 makes a cross-protocol comparison ANALYSIS rather than a "
        "claim, and gpu-cross-device.md:106-111 forbids a reconstructed net outright "
        "because it compounds both halves' noise and measures the halves under "
        "conditions that do not co-occur. Report per backend, per phase, each under its "
        "own protocol (§1.6, AK-D12)")


def freeze_eligibility(*_args: Any, **_kwargs: Any) -> None:
    """Another deliberate dead end. AutoKernel never freezes.

    `P-AK-SEARCH-1` denial 5: *"a readiness signal is not a freeze trigger."*
    Denial 7: *"No release activity. No T3 execution, no release verdict, no
    freeze eligibility, no waiver judgement, no sealing of a release candidate, no
    assembly of a production transaction."* §1.3: a kernel freeze crosses four
    human-only trust boundaries at once.
    """
    raise TriggerAuthorityError(
        "the readiness signal REPORTS; it does not trigger (AK-D3). A freeze crosses "
        "four human-only trust boundaries — the freeze itself, the era-registry rows, "
        "the AutoPilot baseline apply, and the pinned human-only path list "
        "(MEASUREMENT.md:140-142) — and P-AK-SEARCH-1 denial 5 forbids a search record "
        "being cited as a reason any of them may happen automatically. AutoKernel "
        "prepares a release package; a human executes it")


# =============================================================================
# Rendering — the sentence an operator reads
# =============================================================================

def _parity_power_clause(figure: "ParityFigure") -> str:
    """Say whether this parity result could have SEEN the effect being chased.

    "Nothing moved" and "the run was too coarse to see it move" are different
    facts with the same shape, and the parity line renders both as "at parity".
    The campaign's own advisory reference gain is the size the search is looking
    for, and it is already carried on the figure's `ReferenceComparison` — so the
    comparison that separates the two facts needs no new input, only saying.

    An operator handed "12/12 at parity" from a run that could not have resolved
    the gain it is hunting is being shown an underpowered round as a clean
    result, and the next session closes that "gap" by loosening something. The
    clause is a LABEL on the sensitivity already published; it decides nothing,
    and the reference comparison stays `COULD_NOT_CHECK` either way (`AK-D3`:
    the reference figure is advisory and never a trigger).

    The target comes from `figure.comparable_reference_gain`, which is also what
    crosses the seam into the controller's plateau rule. The sentence an operator
    reads and the rule that spends machine time are then answering the power
    question about the SAME number: a phase rendered "underpowered for this
    campaign" is exactly a phase the stop rule refuses to conclude anything from.
    """
    gain = figure.comparable_reference_gain
    if gain is None:
        return ""
    if figure.could_have_detected(gain):
        return (f". A gain of the campaign's advisory {gain} would have been visible at "
                f"this sensitivity, so parity here is a result about the candidate")
    return (f". UNDERPOWERED FOR THIS CAMPAIGN: a gain of the campaign's advisory "
            f"{gain} would NOT have been distinguishable at +/-"
            f"{figure.sensitivity_bound}, so this parity result cannot tell 'no effect' "
            f"from the effect the campaign is looking for. It is a statement about the "
            f"measurement, not about the candidate")


def render_readiness_line(signal: ReadinessSignal, phase: str) -> str:
    """One line per (backend, phase), in the search record's own grammar shape.

    It is deliberately shaped like `P-AK-SEARCH-1`'s record grammar — same anchor
    abbreviation, same e/threshold/MDE/floor quartet, same controls marker — and
    deliberately labelled `SIGNAL_CLASS` rather than `SEARCH RECORD, NOT A CLAIM`,
    because this is a derived signal over records rather than a record.

    THREE STATES, THREE VISIBLY DIFFERENT LINES. The parity line reads like the
    RESULT it is — *"12/12 protected cells at parity, MDE +/-0.018"* — and not
    like the gap it is not. An operator who cannot tell "we measured everything
    and nothing moved" from "we have no figure" will eventually be asked to fix
    the second and will fix the first by loosening something.
    """
    if not isinstance(signal, ReadinessSignal):
        raise CellInadmissible("render_readiness_line() takes a ReadinessSignal")
    standing = None
    for candidate in signal.phases:
        if candidate.phase == phase:
            standing = candidate
    if standing is None:
        raise CellInadmissible(
            f"the signal carries no standing for phase {phase!r}; it declares "
            f"{[s.phase for s in signal.phases]}")
    figure = standing.figure
    head = (f"{signal.backend} {phase} readiness: "
            f"non_inferior={standing.non_inferior.outcome} "
            f"improved={standing.improved.outcome} "
            f"standing={signal.standing}")
    if figure is None:
        body = "no protected-cell figure"
        evidence = f"stratum={api.STRATUM_CONFIRMATION}"
    elif isinstance(figure, ParityFigure):
        body = (f"{figure.parity_cell_count}/{figure.protected_cell_count} protected "
                f"cells at parity, nothing above +/-{figure.sensitivity_bound} "
                f"distinguishable — measured, no detectable difference at any of "
                f"them{_parity_power_clause(figure)}")
        evidence = (f"cells={figure.measured_cell_count}/"
                    f"{figure.protected_cell_count}, at_parity="
                    f"{figure.parity_cell_count}, "
                    f"sensitivity_bound={figure.sensitivity_bound}, "
                    f"MDE={figure.mde}, "
                    f"floor={figure.noise_floor}, "
                    f"sensitivity_cell={figure.sensitivity_cell_id}, "
                    f"resolutions={_census_text(figure.resolution_census)}, "
                    f"stratum={figure.stratum}, "
                    f"sensitivity_event={figure.sensitivity_event_id}")
    else:
        # The SELECTED cell's own resolution rides in the body, not the trailer.
        # `evidence_below_threshold` is admitted to the ordering and the evaluator
        # withholds its speed rank; a census reading
        # `{evidence_below_threshold:1, improvement:1}` in the trailer does not
        # tell an operator which of the two supplied the headline number.
        withheld = ("" if figure.selected_speed_rank_admissible
                    else ", evaluator withheld its speed rank")
        body = (f"weakest orderable protected cell {figure.cell_id} {figure.metric} "
                f"{figure.value} {figure.metric_direction} "
                f"[{figure.selected_effect_resolution}{withheld}] "
                f"(best {figure.best_cell_id} {figure.best_value}) over "
                f"{figure.orderable_cell_count}/{figure.protected_cell_count} "
                f"protected cells, {figure.parity_cell_count} at parity")
        evidence = (f"blocks={figure.paired_blocks}, e={figure.e_value}, "
                    f"thr={figure.threshold}, MDE={figure.mde}, "
                    f"floor={figure.noise_floor}, stratum={figure.stratum}, "
                    f"event={figure.event_id}, "
                    f"selected_resolution={figure.selected_effect_resolution}, "
                    f"selected_speed_rank_admissible="
                    f"{figure.selected_speed_rank_admissible}, "
                    f"resolutions={_census_text(figure.resolution_census)}, "
                    f"excluded_at_parity={list(figure.parity_cell_ids)}")
    trailer = (
        f"[{SIGNAL_CLASS}, reducer={MODULE_ID}, stats={signal.statistics_module_id}, "
        f"tier={TIER}, protocol={standing.protocol_id}, "
        f"vs anchor {signal.anchor.short()}, champion={signal.champion_candidate_id}, "
        f"campaign={signal.campaign_id}, {evidence}, "
        f"controls={signal.controls_marker}, "
        f"eval={_short(signal.evaluator_bundle_sha256)}, "
        f"blockers={list(signal.blockers)}, {signal.computed_at}]")
    return f"{head} — {body} {trailer}"


# =============================================================================
# T2 trigger (§9.7) — authorizes a MEASUREMENT window, never a release
# =============================================================================

@dataclass(frozen=True)
class TriggerDecision:
    """Whether to spend a T2 window. Not whether to release anything.

    §9.7's three trigger conditions, plus the §9.1 precondition that a T2 round
    runs on a champion that already passed T0/T1 as a WHOLE: *"Promote from T1 to
    T2 only after the full composed champion — not one isolated patch — passes
    T0/T1."*
    """

    outcome: str
    satisfied: tuple
    reasons: tuple
    precondition: schemas.Check

    def __post_init__(self) -> None:
        if self.outcome not in TRIGGER_OUTCOMES:
            raise CellInadmissible(
                f"trigger.outcome: {self.outcome!r} is not one of {list(TRIGGER_OUTCOMES)}")
        for condition in self.satisfied:
            if condition not in T2_TRIGGER_CONDITIONS:
                raise CellInadmissible(
                    f"trigger.satisfied: {condition!r} is not one of "
                    f"{list(T2_TRIGGER_CONDITIONS)}")

    def to_dict(self) -> dict:
        return {"outcome": self.outcome, "satisfied": list(self.satisfied),
                "reasons": list(self.reasons),
                "precondition": {"outcome": self.precondition.outcome,
                                 "reasons": list(self.precondition.reasons)},
                "authorizes": "one T2 measurement window",
                "is_trigger": False}


def evaluate_t2_trigger(*, composed_champion_passed_t0_t1: schemas.Check,
                        winners_accumulated_interaction_dominant: schemas.Check,
                        readiness_could_change_materially: schemas.Check,
                        capability_objective_runnable: schemas.Check) -> TriggerDecision:
    """§9.7's trigger, as an explicit disjunction over three declared conditions.

    Each condition arrives as a `Check` the controller computed from records; this
    function decides nothing about a release, and `TriggerDecision.to_dict()` says
    what it authorizes — one T2 measurement window.

    A COULD_NOT_CHECK precondition yields `could_not_evaluate`, not `hold`: "we
    cannot tell whether the champion is green" and "the champion is green and no
    condition fired" are different states, and collapsing them would spend or
    withhold a window for a reason nobody recorded.
    """
    named = (
        (TRIGGER_WINNERS_ACCUMULATED, winners_accumulated_interaction_dominant),
        (TRIGGER_READINESS_COULD_CHANGE, readiness_could_change_materially),
        (TRIGGER_CAPABILITY_RUNNABLE, capability_objective_runnable),
    )
    for _name, check in named:
        if not isinstance(check, schemas.Check):
            raise CellInadmissible("every trigger condition must be a schemas.Check")
    if not isinstance(composed_champion_passed_t0_t1, schemas.Check):
        raise CellInadmissible("composed_champion_passed_t0_t1 must be a schemas.Check")

    if composed_champion_passed_t0_t1.outcome == schemas.FAIL:
        return TriggerDecision(
            outcome=TRIGGER_HOLD, satisfied=(),
            reasons=("the composed champion has not passed T0/T1 as a whole: "
                     + "; ".join(composed_champion_passed_t0_t1.reasons)
                     + ". Promote from T1 to T2 only after the FULL composed champion — "
                     "not one isolated patch — passes T0/T1 (§9.1)",),
            precondition=composed_champion_passed_t0_t1)
    if composed_champion_passed_t0_t1.outcome == schemas.COULD_NOT_CHECK:
        return TriggerDecision(
            outcome=TRIGGER_COULD_NOT_EVALUATE, satisfied=(),
            reasons=("whether the composed champion passed T0/T1 could not be "
                     "determined: "
                     + "; ".join(composed_champion_passed_t0_t1.reasons),),
            precondition=composed_champion_passed_t0_t1)

    satisfied = tuple(name for name, check in named if check.outcome == schemas.PASS)
    if satisfied:
        reasons = tuple(
            f"{name}: " + "; ".join(check.reasons) for name, check in named
            if check.outcome == schemas.PASS)
        return TriggerDecision(outcome=TRIGGER_RUN_T2, satisfied=satisfied,
                               reasons=reasons,
                               precondition=composed_champion_passed_t0_t1)
    if all(check.outcome == schemas.COULD_NOT_CHECK for _name, check in named):
        return TriggerDecision(
            outcome=TRIGGER_COULD_NOT_EVALUATE, satisfied=(),
            reasons=tuple(f"{name}: " + "; ".join(check.reasons)
                          for name, check in named),
            precondition=composed_champion_passed_t0_t1)
    return TriggerDecision(
        outcome=TRIGGER_HOLD, satisfied=(),
        reasons=tuple(f"{name}: " + "; ".join(check.reasons) for name, check in named),
        precondition=composed_champion_passed_t0_t1)


# =============================================================================
# Self-audits — properties proved from this module's own AST
# =============================================================================

def _audits_this_module(source: str) -> bool:
    """Does `source` define THIS module's identity?

    An audit that returns PASS for source it was merely handed can be satisfied by
    handing it nothing: `audit_no_weighting_or_averaging("")` found no
    multiplication in the empty string and said so. The subject of these audits is
    `readiness.py`, so a clean bill of health is issued only for source that
    carries `MODULE_ID = "<this module's id>"` — the one line a stand-in cannot
    have by accident and cannot keep while being a different module. Findings are
    unaffected: a fabricated snippet that DOES contain a weighted average still
    FAILs, which is what the negative tests assert.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        if not isinstance(value, ast.Constant) or value.value != MODULE_ID:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "MODULE_ID":
                return True
    return False


_NOT_THIS_MODULE = (
    "the audited source does not define MODULE_ID = {module_id!r}, so it is not this "
    "module and a PASS over it would be a clean bill of health for something else. An "
    "audit that can be satisfied by deleting what it inspects audits nothing")


def audit_no_write_or_process_paths(source: Optional[str] = None) -> schemas.Check:
    """Reuse `api`'s AST audit on THIS module. No file, no process, no signal.

    Reusing rather than reimplementing is the point: a second copy of the
    forbidden-name list is a second copy that drifts, and the half that drifts is
    whichever one has fewer tests.

    A PASS is issued only for source that is this module (`_audits_this_module`);
    any other clean source is COULD_NOT_CHECK, because an unaudited module and an
    audited-clean one must not report the same outcome.
    """
    if source is None:
        try:
            source = Path(__file__).read_text(encoding="utf-8")
        except OSError as exc:
            return schemas.Check(schemas.COULD_NOT_CHECK,
                                 (f"could not read {__file__}: {exc}",))
    result = api.audit_no_write_or_process_paths(source)
    if result.outcome == schemas.PASS and not _audits_this_module(source):
        return schemas.Check(schemas.COULD_NOT_CHECK,
                             (_NOT_THIS_MODULE.format(module_id=MODULE_ID),))
    return result


#: Operators that are necessary to form a weighted average or any other
#: composite. Addition and subtraction are NOT here: string and tuple
#: concatenation use `+`, and neither of those can fold two measurements into a
#: scalar. Multiplication and division can, and their absence is what makes the
#: forbidden composite inexpressible rather than merely unwritten.
_FORBIDDEN_BINOPS = (ast.Mult, ast.Div, ast.FloorDiv, ast.Pow, ast.MatMult, ast.Mod)

#: Names that reduce a sequence of measurements to one number. `median` and
#: `mad` are included even though `evaluator/statistics.py` exports them: reducing
#: HERE would be a second reducer, which is exactly what this module must not be.
_FORBIDDEN_REDUCERS = frozenset({
    "sum", "mean", "fmean", "median", "mad", "average", "fsum", "prod", "geometric_mean",
    "harmonic_mean", "quantiles", "percentile",
})

#: Absolute imports that would bring a reducer into scope.
_FORBIDDEN_NUMERIC_IMPORTS = frozenset({"statistics", "numpy", "math"})


def audit_no_weighting_or_averaging(source: Optional[str] = None) -> schemas.Check:
    """Prove from this module's own AST that it cannot form a composite.

    AK-D12 forbids a cross-device composite, and §1.6 withdrew the
    production-weighted composite across CPU, GPU, STT and TTS cells. Prose cannot
    enforce that and neither can a review that happens once. A weighted scalar
    needs multiplication and division; a pooled figure needs a reducer. Neither
    appears in this file, so *"if you find yourself averaging across backends"* is
    a state this module cannot reach.

    It is also the machine-checked form of "do not write a second reducer":
    `evaluator/statistics.py` produces every estimate this module reads, and a
    call to `median` or `sum` here would be a competing reduction whose provenance
    nobody could reconstruct.

    COULD_NOT_CHECK when the source cannot be read or parsed — an unreadable
    module is not an audited one.
    """
    if source is None:
        try:
            source = Path(__file__).read_text(encoding="utf-8")
        except OSError as exc:
            return schemas.Check(schemas.COULD_NOT_CHECK,
                                 (f"could not read {__file__}: {exc}",))
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return schemas.Check(schemas.COULD_NOT_CHECK, (f"could not parse module: {exc}",))

    findings: list = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, _FORBIDDEN_BINOPS):
            findings.append(
                f"line {node.lineno}: uses {type(node.op).__name__}; a weighted or "
                "averaged figure over measurements is a composite")
        elif isinstance(node, ast.AugAssign) and isinstance(node.op, _FORBIDDEN_BINOPS):
            findings.append(
                f"line {node.lineno}: augmented {type(node.op).__name__}")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _FORBIDDEN_REDUCERS:
                findings.append(f"line {node.lineno}: calls {func.id}()")
            elif isinstance(func, ast.Attribute) and func.attr in _FORBIDDEN_REDUCERS:
                findings.append(f"line {node.lineno}: calls .{func.attr}()")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _FORBIDDEN_NUMERIC_IMPORTS:
                    findings.append(f"line {node.lineno}: imports {alias.name!r}")
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            root = (node.module or "").split(".")[0]
            if root in _FORBIDDEN_NUMERIC_IMPORTS:
                findings.append(f"line {node.lineno}: imports from {node.module!r}")

    if findings:
        return schemas.Check(schemas.FAIL, tuple(findings))
    if not _audits_this_module(source):
        return schemas.Check(schemas.COULD_NOT_CHECK,
                             (_NOT_THIS_MODULE.format(module_id=MODULE_ID),))
    return schemas.Check(schemas.PASS)
