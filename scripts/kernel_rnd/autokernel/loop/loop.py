#!/usr/bin/env python3
"""The discovery loop. Two critic passes, explicit loopbacks, independent budgets.

Normative specification: `docs/guides/agent-workflows/agent-loop-design.md` in
epyc-root. If this file and that block disagree, the block wins until it is
deliberately amended.

    planner probes freely
      -> CRITIC PASS 1 on the HYPOTHESIS   (no patch exists yet)
      -> planner writes the patch
      -> CRITIC PASS 2 on the DIFF         (before the build)
      -> build -> correctness -> A/B alternating, n>=5   <- the only GPU spend
      -> keep onto the champion branch, or a negative into experiments.md

THREE BUDGETS, NONE FEEDING ANOTHER. The old loop charged `critic_revise` to the same
3-strike counter as a real authoring failure, so a hypothesis could be retired for the
critic doing its job -- in v33 three turns retired `akh-v2-q5-type-specific-dequant`
without ever testing it.

EVERY REJECTION RETURNS ITS REASON to the actor that can act on it. The defect this
replaces filtered refusal reasons on a status the controller never wrote: 22 of 23
authoring failures returned nothing, and the planner re-derived rejected work blind.

The actors are INJECTED. This module never shells out to an LLM; it takes a `Planner`
and a `Critic` protocol, so the whole loop is testable without an API key, a GPU, or
a ROCm toolchain.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from . import archive, bench, gates

HYPOTHESIS_ROUNDS = 3
PATCH_ROUNDS = 2


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class Hypothesis:
    mechanism_id: str
    statement: str
    falsifier: str
    target_surface: str
    target_symbol: str

    def to_dict(self) -> dict:
        return {"mechanism_id": self.mechanism_id, "statement": self.statement,
                "falsifier": self.falsifier, "target_surface": self.target_surface,
                "target_symbol": self.target_symbol}


@dataclass(frozen=True)
class Review:
    """A critic verdict. A rejection without a reason is a bug, so reason is required."""
    accepted: bool
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.accepted and not self.reason.strip():
            raise ValueError(
                "a critic rejection must carry a reason: the reason is what goes back "
                "to the planner, and a rejection with no destination is the defect "
                "that blinded 22 of 23 authoring failures")


class Planner(Protocol):
    def propose(self, context: Mapping[str, Any]) -> Hypothesis: ...
    def author(self, hypothesis: Hypothesis,
               context: Mapping[str, Any]) -> tuple[str, ...]: ...


class Critic(Protocol):
    def review_hypothesis(self, hypothesis: Hypothesis,
                          context: Mapping[str, Any]) -> Review: ...
    def review_patch(self, hypothesis: Hypothesis, paths: Sequence[str],
                     context: Mapping[str, Any]) -> Review: ...


@dataclass
class Outcome:
    """One iteration's record. This, and a champion commit, are the only outputs."""
    status: str
    hypothesis: Hypothesis | None = None
    reasons: list[str] = field(default_factory=list)
    comparison: bench.Comparison | None = None
    gate_verdicts: list[gates.Verdict] = field(default_factory=list)
    champion_head: str | None = None

    def to_attempt(self) -> dict:
        row = {"status": self.status, "turn_recorded_at": _now()}
        if self.hypothesis is not None:
            row.update(self.hypothesis.to_dict())
        if self.reasons:
            row["reason"] = " | ".join(self.reasons)
        if self.comparison is not None:
            row["effect_fraction"] = self.comparison.effect
            row["comparison"] = self.comparison.to_dict()
        if self.gate_verdicts:
            row["gates"] = [verdict.to_dict() for verdict in self.gate_verdicts]
        if self.champion_head:
            row["champion_head"] = self.champion_head
        return row


def iterate(*, planner: Planner, critic: Critic,
            context: Mapping[str, Any],
            measure: Callable[[Hypothesis, Sequence[str]], bench.Comparison],
            gate: Callable[[Hypothesis, Sequence[str]], tuple[bool, list[gates.Verdict]]],
            commit: Callable[[Hypothesis, Sequence[str], bench.Comparison], str],
            hypothesis_rounds: int = HYPOTHESIS_ROUNDS,
            patch_rounds: int = PATCH_ROUNDS) -> Outcome:
    """One full turn. Pure control flow: every side effect is an injected callable."""
    working = dict(context)
    hypothesis_reasons: list[str] = []

    for _ in range(hypothesis_rounds):
        working["prior_hypothesis_rejections"] = list(hypothesis_reasons)
        hypothesis = planner.propose(working)

        # ---- CRITIC PASS 1: the hypothesis, before any patch exists ----------
        verdict = critic.review_hypothesis(hypothesis, working)
        if not verdict.accepted:
            # Verbatim, so the planner can answer the objection rather than guess.
            hypothesis_reasons.append(verdict.reason)
            continue

        patch_reasons: list[str] = []
        for _ in range(patch_rounds):
            working["prior_patch_rejections"] = list(patch_reasons)
            paths = planner.author(hypothesis, working)

            # ---- CRITIC PASS 2: the diff, BEFORE the build ------------------
            patch_verdict = critic.review_patch(hypothesis, paths, working)
            if not patch_verdict.accepted:
                # The hypothesis is untouched: a bad patch is not evidence against
                # the idea it was trying to implement.
                patch_reasons.append(patch_verdict.reason)
                continue

            passed, verdicts = gate(hypothesis, paths)
            if not passed:
                # Compile and correctness failures loop back the same way; the
                # toolchain's own message is the reason, so no critic is needed.
                patch_reasons.append(verdicts[-1].reason if verdicts else "gate refused")
                continue

            comparison = measure(hypothesis, paths)
            if comparison.decisive and comparison.effect > 0:
                head = commit(hypothesis, paths, comparison)
                return Outcome("kept", hypothesis, [], comparison, verdicts, head)
            # A null result IS a result. It is recorded with its mechanism and its
            # sample vector, because a loop whose record of failure is thinner than
            # its record of success teaches its planner to repeat the failures.
            return Outcome("measured_null", hypothesis,
                           [f"effect {comparison.effect * 100:+.3f}% did not clear the "
                            f"{comparison.noise_floor_pct}% noise floor"
                            if comparison.noise_floor_pct is not None
                            else "no noise floor declared"],
                           comparison, verdicts)

        # Patch budget spent. Control returns to the HYPOTHESIS loop, so the planner
        # may refine H knowing it could not be implemented cleanly.
        hypothesis_reasons.extend(patch_reasons)

    # Hypothesis budget spent. H is NOT retired: it re-enters the pool carrying its
    # rejection history, because the profile moves and what was unsupported this week
    # may be the hotspot next week.
    return Outcome("refused_at_formation", None, hypothesis_reasons)


def run(*, planner: Planner, critic: Critic, build_context: Callable[[], dict],
        measure: Callable[..., bench.Comparison],
        gate: Callable[..., tuple[bool, list[gates.Verdict]]],
        commit: Callable[..., str], store_root: Path, epoch: str,
        campaign_id: str, iterations: int) -> list[Outcome]:
    """Drive `iterate` and persist every outcome, kept or not."""
    outcomes = []
    for _ in range(iterations):
        outcome = iterate(planner=planner, critic=critic, context=build_context(),
                          measure=measure, gate=gate, commit=commit)
        archive.record(store_root, outcome.to_attempt(), epoch=epoch,
                       recorded_at=_now(), campaign_id=campaign_id)
        outcomes.append(outcome)
    return outcomes


__all__ = ["Critic", "HYPOTHESIS_ROUNDS", "Hypothesis", "Outcome", "PATCH_ROUNDS",
           "Planner", "Review", "iterate", "run"]
