"""Critic provenance helpers, split verbatim from loop.py (code-budget split)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class Hypothesis:
    mechanism_id: str
    statement: str
    falsifier: str
    target_surface: str
    target_symbol: str
    runtime_pair: Any = None
    #: Vidya claim ids the planner EXPLICITLY declared this hypothesis depends on, already
    #: validated against the claims its prompt presented (`belief_context.reliance`), and the
    #: planner-evidence receipt that recorded the presentation. Prompt exposure alone never
    #: lands here. Kept on the experiments row so a later correction of a claim can find
    #: every proposal that relied on it. Empty (and absent from `to_dict`) for every
    #: historical row.
    relies_on_claims: tuple[str, ...] = ()
    belief_receipt_id: str = ""
    #: Where the planner's proposal came from when it was NOT its own call's reply:
    #: "salvage_turn" (`actors.AgentPlanner._salvage_proposal`: the call spent its wall
    #: budget and one continuation of its session produced the proposal). Empty (and
    #: absent from `to_dict`) for every ordinary proposal and every historical row.
    planner_report_source: str = ""
    #: The declared runtime arm this hypothesis serves (`runtime_arms.DeclaredArmPlanner`),
    #: in memory only and never in `to_dict`. A declared arm's re-serves (a pending strict
    #: calibration's continuation) are governed by the arm ledger, so the DS41-C97
    #: runtime-treatment identity guard exempts it; planner proposals never set it.
    declared_runtime_arm: str = ""
    #: Degraded mode (`belief_context.DegradedWork`): "degraded" when this hypothesis was NOT
    #: planned against a verified ledger in this call but run from `--preseeded-hypotheses`
    #: while belief context was refused; with the seed's id, the ledger state it was seeded
    #: against (a digest) and its declared premises (claim ids). Empty for every planner
    #: proposal and every historical row (absent from `to_dict`).
    belief_mode: str = ""
    seeded_against: Any = None
    belief_premises: tuple[str, ...] = ()
    preseed_id: str = ""
    #: INF06-EXACT-LANE (`exact_lane`): "" (absent) or "tolerance" = today's route
    #: (correctness/quality gate, then speed); "exact" = token-ID identity gate instead of
    #: the quality gate. Declared by the author, verified by the exact judge. Omitted from
    #: `to_dict` unless set, so every existing row keeps its shape.
    lane: str = ""
    #: AK-PM-17: the planner's OPTIONAL declared expectation ({"direction", "min_effect_fraction"}),
    #: carried to the attempt row's `reflexion.expected_outcome` and linted there. Omitted from
    #: `to_dict` unless declared, so every existing row and identity digest keeps its shape.
    expected_outcome: Any = None
    signature_assessment: Any = None  # AK-BNSIG P2: bnsig_planner.assess row; in to_dict only when set

    def __post_init__(self) -> None:
        object.__setattr__(self, "relies_on_claims", tuple(self.relies_on_claims or ()))
        object.__setattr__(self, "belief_premises", tuple(self.belief_premises or ()))
        if self.runtime_pair is not None:
            from .unified_planner import RuntimeArmPair
            body = (self.runtime_pair.to_dict() if isinstance(self.runtime_pair, RuntimeArmPair)
                    else self.runtime_pair)
            object.__setattr__(self, "runtime_pair", RuntimeArmPair.from_dict(body))

    def to_dict(self) -> dict:
        row = {"mechanism_id": self.mechanism_id, "statement": self.statement,
                "falsifier": self.falsifier, "target_surface": self.target_surface,
                "target_symbol": self.target_symbol}
        if self.runtime_pair is not None:
            row["runtime_pair"] = self.runtime_pair.to_dict()
        if self.relies_on_claims:
            row["relies_on_claims"] = list(self.relies_on_claims)
        if self.belief_receipt_id:
            row["belief_receipt_id"] = self.belief_receipt_id
        if self.planner_report_source:
            row["planner_report_source"] = self.planner_report_source
        if self.belief_mode:
            row["belief_mode"] = self.belief_mode
            row["seeded_against"] = self.seeded_against
            row["belief_premises"] = list(self.belief_premises)
            row["preseed_id"] = self.preseed_id
        if self.lane:
            row["lane"] = self.lane
        if self.expected_outcome is not None:
            row["expected_outcome"] = self.expected_outcome
        if self.signature_assessment is not None:  # AK-BNSIG: journal + critic input; absent when not assessed
            row["signature_assessment"] = self.signature_assessment
        return row


@dataclass(frozen=True)
class Review:
    """A critic verdict. A rejection without a reason is a bug, so reason is required."""
    accepted: bool
    reason: str = ""
    validator_identity: str = ""
    validator_kind: str = ""
    independence: str = ""
    evidence_inspected: tuple[str, ...] = ()
    #: Critic pass 2 only: non-empty when the patch was rejected because the
    #: mechanism cannot be implemented without touching a region the admitted route
    #: refuses; names that route or rule. A `SCOPE:` reason prefix means the same
    #: (`classify_patch_rejection`).
    scope_rule: str = ""

    def __post_init__(self) -> None:
        if not self.accepted and not self.reason.strip():
            raise ValueError(
                "a critic rejection must carry a reason: the reason is what goes back "
                "to the planner, and a rejection with no destination is the defect "
                "that blinded 22 of 23 authoring failures")
        if self.validator_kind and self.validator_kind not in {
                "script", "oracle", "llm_critic"}:
            raise ValueError(f"unknown validator kind {self.validator_kind!r}")
        if self.independence and self.independence not in {
                "same_fleet", "same_family", "different_family", "non_model"}:
            raise ValueError(f"unknown validator independence {self.independence!r}")


class Critic(Protocol):
    def review_hypothesis(self, hypothesis: Hypothesis,
                          context: Mapping[str, Any]) -> Review: ...
    def review_patch(self, hypothesis: Hypothesis, paths: Sequence[str],
                     context: Mapping[str, Any]) -> Review: ...


def _critic_provenance(critic: Critic, review: Review, *, decision: str,
                       evidence: tuple[str, ...]) -> dict:
    """Project a critic verdict into the common provenance carrier."""
    identity = (review.validator_identity or
                f"{type(critic).__module__}.{type(critic).__qualname__}")
    kind = review.validator_kind or "script"
    independence = review.independence or "non_model"
    return {
        "decision": decision,
        "validator_identity": identity,
        "validator_kind": kind,
        "independence": independence,
        "evidence_inspected": list(review.evidence_inspected or evidence),
        "accepted": review.accepted,
        "reason": review.reason or None,
        "changed_subsequent_search": False,
    }
