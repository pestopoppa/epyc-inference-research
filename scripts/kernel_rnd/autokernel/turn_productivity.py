#!/usr/bin/env python3
"""Immutable refine-turn accounting and the AK-X-6 repair-only reducer.

This module consumes records; it does not run a planner, candidate, benchmark,
or selector.  AK-PT-1 needs the rescued/persistent distinction to survive into
the archive, and AK-X-6 needs a pre-committed statistical rule over that archive.
The rule below can only remove search authority: it labels a latest refine turn
``repair_only`` after two calibrated e-processes establish separation around the
pre-committed contribution floor.  It cannot retain, rank, promote, or deploy a
candidate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from . import schemas
from .evaluator import api, statistics

__all__ = [
    "PRODUCTIVITY_MODULE_ID",
    "RESCUED", "PERSISTENT", "FAILED", "REGRESSED", "TURN_CLASSES",
    "CONTINUE_REFINEMENT", "REPAIR_ONLY", "TURN_BUDGET_DECISIONS",
    "ProductivityError", "ArchiveError", "TurnBudgetRuleMutated",
    "TurnObservation", "ProductivityArchive", "ContributionFloor",
    "ProductivityCalibration", "TurnBudgetRule", "TurnBudgetCommitment",
    "TurnBudgetEvaluation", "evaluate_turn_budget",
]

PRODUCTIVITY_MODULE_ID = "autokernel.turn_productivity/v1"

RESCUED = "rescued"
PERSISTENT = "persistent"
FAILED = "failed"
REGRESSED = "regressed"
TURN_CLASSES = (RESCUED, PERSISTENT, FAILED, REGRESSED)

CONTINUE_REFINEMENT = "continue_refinement"
REPAIR_ONLY = "repair_only_no_search_advance"
TURN_BUDGET_DECISIONS = (CONTINUE_REFINEMENT, REPAIR_ONLY)


class ProductivityError(ValueError):
    """Base refusal for malformed or unbound productivity material."""


class ArchiveError(ProductivityError):
    """The append order or candidate-state chain is incoherent."""


class TurnBudgetRuleMutated(ProductivityError):
    """The evaluated rule is not the one committed before the archive."""


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProductivityError(f"{label} must be a non-empty string")
    return value


def _instant(value: Any, label: str) -> str:
    text = _text(value, label)
    candidate = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ProductivityError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ProductivityError(f"{label} must carry a UTC offset")
    return text


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 \
            or any(c not in "0123456789abcdef" for c in value):
        raise ProductivityError(f"{label} must be a lowercase sha256 hex")
    return value


@dataclass(frozen=True)
class TurnObservation:
    """One append-ordered ``(turn, task, correct?, speedup)`` observation.

    ``previous_correct`` is required because rescued and persistent cannot be
    reconstructed from the final correctness bit alone.  Incorrect candidates
    carry no performance number: T0 is lexicographically prior, so timing a
    wrong result would create exactly the rank the evaluator forbids.
    """

    sequence: int
    campaign_id: str
    proposal_id: str
    turn: int
    task_id: str
    previous_correct: bool
    correct: bool
    relative_speedup: Optional[float]
    evidence_ref: str
    measured_at: str

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) \
                or self.sequence < 0:
            raise ProductivityError("sequence must be a non-negative int")
        if isinstance(self.turn, bool) or not isinstance(self.turn, int) or self.turn < 1:
            raise ProductivityError("turn must be a positive int")
        for name in ("campaign_id", "proposal_id", "task_id", "evidence_ref"):
            _text(getattr(self, name), name)
        _instant(self.measured_at, "measured_at")
        if not isinstance(self.previous_correct, bool) or not isinstance(self.correct, bool):
            raise ProductivityError("previous_correct and correct must be bools")
        if self.correct:
            value = self.relative_speedup
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value) or value <= -1.0:
                raise ProductivityError(
                    "a correct turn needs finite relative_speedup > -1.0")
        elif self.relative_speedup is not None:
            raise ProductivityError(
                "an incorrect turn must not carry relative_speedup; T0 failures have no rank")

    @property
    def turn_class(self) -> str:
        if self.correct:
            return PERSISTENT if self.previous_correct else RESCUED
        return REGRESSED if self.previous_correct else FAILED

    def to_dict(self) -> dict:
        return {
            "sequence": self.sequence,
            "campaign_id": self.campaign_id,
            "proposal_id": self.proposal_id,
            "turn": self.turn,
            "task_id": self.task_id,
            "previous_correct": self.previous_correct,
            "correct": self.correct,
            "relative_speedup": self.relative_speedup,
            "turn_class": self.turn_class,
            "evidence_ref": self.evidence_ref,
            "measured_at": self.measured_at,
        }

    @property
    def content_hash(self) -> str:
        return schemas.content_hash(self.to_dict())


@dataclass(frozen=True)
class ProductivityArchive:
    """An immutable append-order view over per-turn observations."""

    campaign_id: str
    observations: tuple

    def __post_init__(self) -> None:
        _text(self.campaign_id, "archive.campaign_id")
        if not isinstance(self.observations, tuple):
            raise ArchiveError("archive.observations must be a tuple in append order")
        prior: dict[tuple[str, str], TurnObservation] = {}
        identities = set()
        for position, observation in enumerate(self.observations):
            if not isinstance(observation, TurnObservation):
                raise ArchiveError("archive entries must be TurnObservation objects")
            if observation.sequence != position:
                raise ArchiveError(
                    f"archive position {position} carries sequence {observation.sequence}; "
                    "the append order may not be rewritten")
            if observation.campaign_id != self.campaign_id:
                raise ArchiveError("an observation names another campaign")
            identity = (observation.proposal_id, observation.turn, observation.task_id)
            if identity in identities:
                raise ArchiveError(f"duplicate turn observation {identity!r}")
            identities.add(identity)
            chain_key = (observation.proposal_id, observation.task_id)
            previous = prior.get(chain_key)
            if previous is not None:
                if observation.turn != previous.turn + 1:
                    raise ArchiveError(
                        f"{chain_key!r} jumps from turn {previous.turn} to "
                        f"{observation.turn}")
                if observation.previous_correct != previous.correct:
                    raise ArchiveError(
                        f"{chain_key!r} turn {observation.turn} declares previous_correct="
                        f"{observation.previous_correct}, but turn {previous.turn} recorded "
                        f"correct={previous.correct}")
            prior[chain_key] = observation

    def by_class(self, turn_class: str) -> tuple:
        if turn_class not in TURN_CLASSES:
            raise ProductivityError(f"unknown turn class {turn_class!r}")
        return tuple(row for row in self.observations if row.turn_class == turn_class)

    @property
    def latest_turn(self) -> Optional[int]:
        return None if not self.observations else max(row.turn for row in self.observations)

    def to_dict(self) -> dict:
        return {
            "schema": "epyc.autokernel.turn_productivity_archive.v1",
            "module_id": PRODUCTIVITY_MODULE_ID,
            "campaign_id": self.campaign_id,
            "observations": [row.to_dict() for row in self.observations],
        }

    @property
    def content_hash(self) -> str:
        return schemas.content_hash(self.to_dict())


@dataclass(frozen=True)
class ContributionFloor:
    """The immutable campaign-start floor; never inferred from observed turns."""

    campaign_id: str
    relative_speedup: float
    rationale_ref: str
    committed_at: str

    def __post_init__(self) -> None:
        _text(self.campaign_id, "floor.campaign_id")
        _text(self.rationale_ref, "floor.rationale_ref")
        _instant(self.committed_at, "floor.committed_at")
        value = self.relative_speedup
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value) or value <= 0.0:
            raise ProductivityError("floor.relative_speedup must be finite and > 0")

    def to_dict(self) -> dict:
        return {
            "campaign_id": self.campaign_id,
            "relative_speedup": float(self.relative_speedup),
            "rationale_ref": self.rationale_ref,
            "committed_at": self.committed_at,
        }


@dataclass(frozen=True)
class ProductivityCalibration:
    """The threshold and construction copied from accepted campaign calibration."""

    stratum: str
    threshold: float
    construction_id: str
    calibration_sha256: str
    samples_ref: str

    def __post_init__(self) -> None:
        if self.stratum not in api.STRATA:
            raise ProductivityError(f"calibration.stratum must be one of {list(api.STRATA)}")
        if isinstance(self.threshold, bool) or not isinstance(self.threshold, (int, float)) \
                or not math.isfinite(self.threshold) or self.threshold <= 1.0:
            raise ProductivityError("calibration.threshold must be finite and > 1")
        statistics.select_construction(self.construction_id)
        _sha256(self.calibration_sha256, "calibration.calibration_sha256")
        _text(self.samples_ref, "calibration.samples_ref")

    @classmethod
    def from_outputs(cls, outputs: api.CalibrationOutputs, *,
                     stratum: str) -> "ProductivityCalibration":
        if not isinstance(outputs, api.CalibrationOutputs):
            raise ProductivityError("from_outputs requires api.CalibrationOutputs")
        if not outputs.accepted:
            raise ProductivityError("rejected campaign calibration cannot license a rule")
        return cls(
            stratum=stratum,
            threshold=outputs.threshold_for(stratum),
            construction_id=outputs.e_process_construction_id,
            calibration_sha256=schemas.content_hash(outputs.to_dict()),
            samples_ref=outputs.samples_ref,
        )

    @property
    def construction(self) -> statistics.EProcessConstruction:
        return statistics.select_construction(self.construction_id)

    def to_dict(self) -> dict:
        return {
            "stratum": self.stratum,
            "threshold": float(self.threshold),
            "construction_id": self.construction_id,
            "calibration_sha256": self.calibration_sha256,
            "samples_ref": self.samples_ref,
        }


@dataclass(frozen=True)
class TurnBudgetRule:
    """A one-way authority reduction bound to a floor and campaign calibration."""

    rule_id: str
    floor: ContributionFloor
    calibration: ProductivityCalibration

    def __post_init__(self) -> None:
        _text(self.rule_id, "rule.rule_id")
        if not isinstance(self.floor, ContributionFloor):
            raise ProductivityError("rule.floor must be a ContributionFloor")
        if not isinstance(self.calibration, ProductivityCalibration):
            raise ProductivityError(
                "rule.calibration must be a ProductivityCalibration")

    def to_dict(self) -> dict:
        return {
            "schema": "epyc.autokernel.turn_budget_rule.v1",
            "rule_id": self.rule_id,
            "floor": self.floor.to_dict(),
            "calibration": self.calibration.to_dict(),
            "authority": "repair_only_or_continue; never rank/retain/promote/deploy",
        }

    @property
    def content_hash(self) -> str:
        return schemas.content_hash(self.to_dict())


@dataclass(frozen=True)
class TurnBudgetCommitment:
    campaign_id: str
    rule_id: str
    rule_sha256: str
    committed_at: str

    def __post_init__(self) -> None:
        _text(self.campaign_id, "commitment.campaign_id")
        _text(self.rule_id, "commitment.rule_id")
        _sha256(self.rule_sha256, "commitment.rule_sha256")
        _instant(self.committed_at, "commitment.committed_at")

    @classmethod
    def commit(cls, rule: TurnBudgetRule, *,
               committed_at: str) -> "TurnBudgetCommitment":
        if not isinstance(rule, TurnBudgetRule):
            raise ProductivityError("commit requires TurnBudgetRule")
        return cls(campaign_id=rule.floor.campaign_id, rule_id=rule.rule_id,
                   rule_sha256=rule.content_hash, committed_at=committed_at)

    def verify(self, rule: TurnBudgetRule) -> schemas.Check:
        reasons = []
        if not isinstance(rule, TurnBudgetRule):
            return schemas.Check(schemas.COULD_NOT_CHECK, ("no TurnBudgetRule supplied",))
        if rule.floor.campaign_id != self.campaign_id:
            reasons.append("rule names another campaign")
        if rule.rule_id != self.rule_id:
            reasons.append("rule id differs from the campaign-start commitment")
        if rule.content_hash != self.rule_sha256:
            reasons.append("rule content differs from the campaign-start commitment")
        return schemas.Check(schemas.FAIL, tuple(reasons)) if reasons \
            else schemas.Check(schemas.PASS)


@dataclass(frozen=True)
class TurnBudgetEvaluation:
    decision: str
    latest_turn: Optional[int]
    latest_admitted_classes: tuple
    rescued_count: int
    persistent_count: int
    rescued_below_floor: Optional[statistics.EProcessRun]
    persistent_above_floor: Optional[statistics.EProcessRun]
    rule_sha256: str
    archive_sha256: str
    reasons: tuple

    def __post_init__(self) -> None:
        if self.decision not in TURN_BUDGET_DECISIONS:
            raise ProductivityError(f"unknown turn-budget decision {self.decision!r}")
        if self.decision == REPAIR_ONLY:
            if self.rescued_below_floor is None or not self.rescued_below_floor.crossed \
                    or self.persistent_above_floor is None \
                    or not self.persistent_above_floor.crossed:
                raise ProductivityError(
                    "repair_only requires both calibrated e-processes to cross")
            if not self.latest_admitted_classes \
                    or set(self.latest_admitted_classes) != {RESCUED}:
                raise ProductivityError("repair_only requires an only-rescued latest turn")

    def to_dict(self) -> dict:
        return {
            "schema": "epyc.autokernel.turn_budget_evaluation.v1",
            "decision": self.decision,
            "latest_turn": self.latest_turn,
            "latest_admitted_classes": list(self.latest_admitted_classes),
            "rescued_count": self.rescued_count,
            "persistent_count": self.persistent_count,
            "rescued_below_floor": None if self.rescued_below_floor is None
            else self.rescued_below_floor.to_dict(),
            "persistent_above_floor": None if self.persistent_above_floor is None
            else self.persistent_above_floor.to_dict(),
            "rule_sha256": self.rule_sha256,
            "archive_sha256": self.archive_sha256,
            "reasons": list(self.reasons),
        }


def _process(values: tuple, rule: TurnBudgetRule) -> Optional[statistics.EProcessRun]:
    if not values:
        return None
    return statistics.run_e_process(
        values,
        construction=rule.calibration.construction,
        hypothesis=statistics.HYPOTHESIS_IMPROVEMENT,
        margin=0.0,
        threshold=rule.calibration.threshold,
    )


def evaluate_turn_budget(archive: ProductivityArchive, *, rule: TurnBudgetRule,
                         commitment: TurnBudgetCommitment) -> TurnBudgetEvaluation:
    """Evaluate AK-X-6 without a point-comparison stopping branch.

    The first process tests that rescued speedups lie below the committed floor;
    the second tests that persistent speedups lie above it.  The latest turn must
    also admit only rescued candidates.  Thus ``repair_only`` means the two
    empirical distributions are separated by a boundary fixed before either was
    observed.  A median, mean, or latest point never licenses the decision.
    """
    if not isinstance(archive, ProductivityArchive):
        raise ProductivityError("archive must be a ProductivityArchive")
    if not isinstance(rule, TurnBudgetRule) or not isinstance(
            commitment, TurnBudgetCommitment):
        raise ProductivityError("rule and commitment must use their typed contracts")
    check = commitment.verify(rule)
    if check.outcome != schemas.PASS:
        raise TurnBudgetRuleMutated("; ".join(check.reasons))
    if archive.campaign_id != rule.floor.campaign_id:
        raise ArchiveError("archive and contribution floor name different campaigns")

    latest = archive.latest_turn
    latest_rows = () if latest is None else tuple(
        row for row in archive.observations if row.turn == latest)
    latest_admitted = tuple(
        row.turn_class for row in latest_rows if row.correct)
    rescued = archive.by_class(RESCUED)
    persistent = archive.by_class(PERSISTENT)
    floor = rule.floor.relative_speedup

    below = _process(tuple(floor - float(row.relative_speedup) for row in rescued), rule)
    above = _process(tuple(float(row.relative_speedup) - floor for row in persistent), rule)
    only_rescued = bool(latest_admitted) and set(latest_admitted) == {RESCUED}
    separated = below is not None and below.crossed and above is not None and above.crossed

    if only_rescued and separated:
        decision = REPAIR_ONLY
        reasons = (
            "the latest turn admitted only rescued candidates",
            "the calibrated e-process crossed for rescued speedups below the "
            "pre-committed contribution floor",
            "the calibrated e-process crossed for persistent speedups above the same floor; "
            "the distributions are separated around a boundary fixed before observation",
            "repair work may continue, but this turn does not advance search",
        )
    else:
        decision = CONTINUE_REFINEMENT
        reasons_list = []
        if not only_rescued:
            reasons_list.append(
                "the latest turn did not admit only rescued candidates")
        if below is None or not below.crossed:
            reasons_list.append(
                "rescued-below-floor evidence has not crossed its calibrated threshold")
        if above is None or not above.crossed:
            reasons_list.append(
                "persistent-above-floor evidence has not crossed its calibrated threshold")
        reasons_list.append(
            "insufficient evidence never shortens the refine budget")
        reasons = tuple(reasons_list)

    return TurnBudgetEvaluation(
        decision=decision,
        latest_turn=latest,
        latest_admitted_classes=latest_admitted,
        rescued_count=len(rescued),
        persistent_count=len(persistent),
        rescued_below_floor=below,
        persistent_above_floor=above,
        rule_sha256=rule.content_hash,
        archive_sha256=archive.content_hash,
        reasons=reasons,
    )
