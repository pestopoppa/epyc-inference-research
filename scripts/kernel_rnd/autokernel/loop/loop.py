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

from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from . import archive, bench, gates

HYPOTHESIS_ROUNDS = 3
PATCH_ROUNDS = 2

#: NOT an error and NOT a refusal. The run was told to stop while this candidate was
#: still forming, so the lane abandons rather than drawing further actor calls for a
#: result nobody will use (run 20: ~50 min of "draining" with the GPU at 0% while
#: seven lanes politely completed codex conversations). The planner's memory must
#: read this as "never attempted" -- neither a verdict on the mechanism nor a refusal
#: of it.
STOPPED_MID_FORMATION = (
    "run stop requested while this candidate was still forming; abandoned before "
    "the next actor call. No verdict on the mechanism — it was never attempted")


class RunAborted(RuntimeError):
    """The run stopped because iterations were failing systematically."""


#: The standing strategy, constraints and settled list. It is rendered into EVERY
#: actor bundle: it sat unread beside the loop for the whole of run 6 while the
#: planner proposed things its own "Already in v9" list names.
PROGRAM = Path(__file__).resolve().parent / "program.md"


class TailRefused(RuntimeError):
    """The serialized tail refused this candidate before it could be measured.

    Defined HERE so `iterate` can catch it at the point where the hypothesis is still
    in scope. Raised by the pool when the champion advanced under a lane, and the
    candidate must carry its hypothesis out with it: a formed-but-unmeasured candidate
    is a QUEUE ENTRY, not a loss, and the planner is supposed to reconsider it against
    the new champion. Returning None here is how it became a loss.
    """


class ActorTransient(RuntimeError):
    """The actor provider failed in a way worth retrying.

    Defined HERE, not in `actors`, so `iterate` can catch it without importing the
    concrete actor module (which imports this one). A transient must end an
    ITERATION, never the run: the superseded controller let provider faults escape as
    terminal, so a codex 401 on 2026-08-26 took down 284 attempts in 23 minutes.
    """


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


def _safe_step(hook):
    """Reporting must never kill the loop, and a heartbeat least of all."""
    if hook is None:
        return lambda _label: None

    def beat(label: str) -> None:
        try:
            hook(label)
        except Exception:      # noqa: BLE001
            pass
    return beat


def _null_reason(comparison: bench.Comparison) -> str:
    """Say WHY a measurement was not decisive: missing the bar and being vetoed for
    drift are different facts, and the planner must be able to tell them apart.

    A drift veto reported as "did not clear the floor" reads as a clean null -- the
    mechanism looks tested and unpromising when in fact it was never resolved. That
    invites the planner to abandon a live idea, which is the same class of error as a
    fabricated refusal.
    """
    if comparison.decisive is None:
        # Uncalibrated surface: recorded, UNDECIDABLE. Not a null -- the instrument
        # has no floor here, so "did not clear" would be a claim about a bar that
        # does not exist. The planner may keep proposing against it; keeps wait for
        # an A/A calibration campaign (`--calibrate-surface`).
        return (f"UNDECIDABLE — surface {comparison.surface} has no bootstrap-"
                f"calibrated noise floor; recorded (raw {comparison.effect * 100:+.3f}%),"
                f" decisive=None, keeps refused until --calibrate-surface writes one")
    if comparison.noise_floor_pct is None:
        return "no noise floor declared"
    if comparison.drifting:
        return (f"NOT RESOLVED — the instrument drifted during the run "
                f"(anchor {comparison.anchor_drift_pct:+.3f}%, candidate "
                f"{comparison.candidate_drift_pct:+.3f}%, floor "
                f"{comparison.noise_floor_pct:.3f}%). The raw effect was "
                f"{comparison.effect * 100:+.3f}%, but an arm that is still moving "
                f"resolves nothing. This mechanism is UNTESTED, not unpromising — "
                f"re-run it rather than abandoning it")
    return (f"effect {comparison.effect * 100:+.3f}% did not clear the "
            f"{comparison.noise_floor_pct:.3f}% noise floor")


def iterate(*, planner: Planner, critic: Critic,
            context: Mapping[str, Any],
            measure: Callable[[Hypothesis, Sequence[str]], bench.Comparison],
            gate: Callable[[Hypothesis, Sequence[str]], tuple[bool, list[gates.Verdict]]],
            commit: Callable[[Hypothesis, Sequence[str], bench.Comparison], str],
            hypothesis_rounds: int = HYPOTHESIS_ROUNDS,
            patch_rounds: int = PATCH_ROUNDS,
            on_step: Callable[[str], None] | None = None,
            tail_session: Callable[[], Any] = nullcontext,
            should_abandon: Callable[[], bool] | None = None) -> Outcome:
    """One full turn. Pure control flow: every side effect is an injected callable.

    `should_abandon` is the DRAIN TIER for a lane that does not hold the serialized
    tail: polled at every stage boundary of FORMATION (before each planner call,
    critic pass and authoring turn), and never inside the tail — a candidate that
    reaches the tail finishes build → oracle → A/B → commit exactly as before, so a
    stop can never kill a measurement mid-A/B.
    """
    working = dict(context)
    hypothesis_reasons: list[str] = []

    try:
        return _iterate(planner=planner, critic=critic, working=working,
                        hypothesis_reasons=hypothesis_reasons, measure=measure,
                        gate=gate, commit=commit,
                        hypothesis_rounds=hypothesis_rounds,
                        patch_rounds=patch_rounds, on_step=_safe_step(on_step),
                        tail_session=tail_session,
                        should_abandon=should_abandon or (lambda: False))
    except TailRefused as exc:
        # The candidate was formed and never measured. Carry the hypothesis: the
        # patch may well still help against the champion that displaced it, and the
        # planner is told to look at these FIRST.
        return Outcome("superseded", getattr(exc, "hypothesis", None), [str(exc)])
    except ActorTransient as exc:
        # The provider failed, not the science. This ends the ITERATION and is
        # recorded as such; the run continues, and a streak becomes visible in
        # experiments.md rather than taking the campaign down with it.
        return Outcome("planner_transient", None, [str(exc)])
    except bench.BenchFailed as exc:
        # The INSTRUMENT failed, not the science, and it gets the same treatment for
        # the same reason. Run 12 died on iteration 1 because `llama-bench` was
        # SIGKILLed (rc=-9) mid-measurement and BenchFailed escaped `iterate`: one
        # killed process ended a ten-iteration run that had already spent its profile
        # and held the device. `earlyoom` on this host ignores llama-server and NOT
        # llama-bench, so an external kill is a standing hazard rather than a freak.
        #
        # Recorded distinctly from a provider transient: "the benchmark could not be
        # taken" is a different fact from "the actor would not answer", and merging
        # them would hide an instrument failing behind an API being flaky.
        return Outcome("bench_failed", None, [str(exc)])


def _iterate(*, planner, critic, working, hypothesis_reasons, measure, gate, commit,
             hypothesis_rounds, patch_rounds, on_step=lambda _label: None,
             tail_session=nullcontext,
             should_abandon=lambda: False) -> Outcome:
    last_proposed: Hypothesis | None = None

    def stopped() -> Outcome:
        # Names whatever was in flight, exactly as a refusal row must.
        return Outcome("stopped_mid_formation", last_proposed,
                       [STOPPED_MID_FORMATION])

    for _ in range(hypothesis_rounds):
        # Polled BEFORE each actor call, never after: the whole point is that no
        # further multi-minute call is drawn once the run has been told to stop.
        if should_abandon():
            return stopped()
        working["prior_hypothesis_rejections"] = list(hypothesis_reasons)
        on_step("proposing a hypothesis")
        hypothesis = planner.propose(working)
        last_proposed = hypothesis

        # ---- CRITIC PASS 1: the hypothesis, before any patch exists ----------
        if should_abandon():
            return stopped()
        on_step("critic pass 1: reviewing the hypothesis")
        verdict = critic.review_hypothesis(hypothesis, working)
        if not verdict.accepted:
            # Verbatim, so the planner can answer the objection rather than guess.
            hypothesis_reasons.append(verdict.reason)
            continue

        patch_reasons: list[str] = []
        for _ in range(patch_rounds):
            if should_abandon():
                return stopped()
            working["prior_patch_rejections"] = list(patch_reasons)
            on_step("authoring the patch")
            paths = planner.author(hypothesis, working)

            # ---- CRITIC PASS 2: the diff, BEFORE the build ------------------
            if should_abandon():
                return stopped()
            on_step("critic pass 2: reviewing the diff")
            patch_verdict = critic.review_patch(hypothesis, paths, working)
            if not patch_verdict.accepted:
                # The hypothesis is untouched: a bad patch is not evidence against
                # the idea it was trying to implement.
                patch_reasons.append(patch_verdict.reason)
                continue

            on_step("building and gating")
            # Build, oracle, A/B and commit are ONE atomic step for this candidate.
            # Split across three separate acquisitions, a concurrent peer's keep
            # landing in a gap turns an ALREADY-MEASURED candidate into a stale one:
            # probed at 4 lanes, 20 A/B runs executed and only 5 recorded a
            # comparison -- 15 completed measurements thrown away after the device
            # time was spent. The same gap discards completed builds.
            #
            # Held per ATTEMPT, not per iteration: authoring sits between patch
            # rounds, and holding across it would serialize formation, which is the
            # 86% we are trying to overlap.
            # The refusal is raised by the session's __enter__ (the staleness check
            # happens under the lock), so it must be caught around the `with`, not
            # around the call that builds it.
            try:
                with tail_session():
                    passed, verdicts = gate(hypothesis, paths)
                    if not passed:
                        # Compile and correctness failures loop back the same way; the
                        # toolchain's own message is the reason, so no critic is needed.
                        patch_reasons.append(
                            verdicts[-1].reason if verdicts else "gate refused")
                        continue

                    on_step("measuring A/B on the device")
                    comparison = measure(hypothesis, paths)
                    if comparison.decisive and comparison.effect > 0:
                        head = commit(hypothesis, paths, comparison)
                        return Outcome("kept", hypothesis, [], comparison, verdicts, head)
            except TailRefused as exc:
                # Formed and never measured. Carry the hypothesis out so the planner
                # can reconsider it against the champion that displaced it.
                exc.hypothesis = hypothesis
                raise
            # A null result IS a result. It is recorded with its mechanism and its
            # sample vector, because a loop whose record of failure is thinner than
            # its record of success teaches its planner to repeat the failures.
            return Outcome("measured_null", hypothesis,
                           [_null_reason(comparison)],
                           comparison, verdicts)

        # Patch budget spent. Control returns to the HYPOTHESIS loop, so the planner
        # may refine H knowing it could not be implemented cleanly.
        hypothesis_reasons.extend(patch_reasons)

    # Hypothesis budget spent. H is NOT retired: it re-enters the pool carrying its
    # rejection history, because the profile moves and what was unsupported this week
    # may be the hotspot next week.
    # Carry the last hypothesis proposed, not None: a refusal row whose mechanism_id is
    # empty tells the operator that something was refused without saying WHAT, and it is
    # the row the dashboard shows most often. The refusal is of an idea; name the idea.
    return Outcome("refused_at_formation", last_proposed, hypothesis_reasons)


def run(*, planner: Planner, critic: Critic, build_context: Callable[[], dict],
        measure: Callable[..., bench.Comparison],
        gate: Callable[..., tuple[bool, list[gates.Verdict]]],
        commit: Callable[..., str], store_root: Path, epoch: str,
        campaign_id: str, iterations: int,
        on_iteration: Callable[[list["Outcome"]], None] | None = None
        ) -> list[Outcome]:
    """Drive `iterate` and persist every outcome, kept or not.

    TEST-ONLY SEAM since 2026-08-31: every production run drives `pipeline.run_pool`
    (via `run.py`, `--workers` >= 1 all pooled), and the sequential CLI path that
    called this was deleted. It survives because `test_loop`, `test_anchor` and
    `test_production` drive the real anchor-guard and headline composition through
    it with fakes — a driver those suites can hold still. It shrank to what those
    suites use: the consecutive-error BREAKER moved to `pipeline.run_pool`
    (`pipeline.MAX_CONSECUTIVE_ERRORS`) — the pool owns the real one, and this seam
    only contains single-iteration faults without counting them — and the
    `should_stop` / `reset` / `on_step` hooks went with the CLI path (the pool's
    equivalents are `Budget.should_stop` + `should_abandon`, `reset_to_champion`
    and the lane-labelled `on_step`).

    `on_iteration` fires after EVERY iteration, including refused and transient ones.
    Publishing status is not the loop's concern, but a loop that only reports when it
    succeeds is indistinguishable from one that is stuck -- which is what "STOPPED,
    authoring/build are event-silent by design" looked like on the old dashboard.
    """
    outcomes: list[Outcome] = []
    for _ in range(iterations):
        try:
            outcome = iterate(planner=planner, critic=critic,
                              context=build_context(), measure=measure, gate=gate,
                              commit=commit)
        except RunAborted:
            # NEVER an iteration fault, so it must not be laundered into one: the
            # blanket handler below would record it as `iteration_error` and draw the
            # next iteration. The promotion A/A guard refused the binary in the anchor
            # slot, so every later comparison would be against a champion that is not
            # the champion -- run 18 continued 6.5 hours and 114 candidates that way.
            raise
        except Exception as exc:      # noqa: BLE001 -- deliberate, see below
            # NOTHING that goes wrong in one iteration may end the run. Run 12 lost
            # ten iterations, a profile and a held device to a single SIGKILLed
            # benchmark; before that, a codex 401 took down 284 attempts. The
            # iteration is the unit of failure, and the run is what has to survive.
            # The traceback is recorded in full so containment cannot hide the fault;
            # the consecutive-failure BREAKER is the pool's (`pipeline.run_pool`).
            outcome = Outcome("iteration_error", None,
                              [f"{type(exc).__name__}: {exc}",
                               traceback.format_exc()[-1500:]])
        archive.record(store_root, outcome.to_attempt(), epoch=epoch,
                       recorded_at=_now(), campaign_id=campaign_id)
        outcomes.append(outcome)
        if on_iteration is not None:
            try:
                on_iteration(outcomes)
            except Exception:      # noqa: BLE001 - reporting must never kill the loop
                pass
    return outcomes


__all__ = ["ActorTransient", "TailRefused", "RunAborted", "Critic",
           "HYPOTHESIS_ROUNDS", "Hypothesis", "Outcome", "PATCH_ROUNDS",
           "Planner", "Review", "STOPPED_MID_FORMATION", "iterate", "run"]
