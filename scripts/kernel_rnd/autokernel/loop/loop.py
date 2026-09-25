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

import sys
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from . import bench, gates, integrity

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

#: The same stop, after this iteration had ALREADY disposed of candidates. DS41 run
#: 9c recorded a hoist that two authoring rounds implemented, the critic accepted
#: twice and `op_scope` refused twice as "never attempted", because the stop landed
#: on the NEXT planner call. Those candidates carry their own rows
#: (`CANDIDATE_DISPOSITIONS`); this row must not deny them.
STOPPED_AFTER_DISPOSALS = (
    "run stop requested while this iteration was still forming; abandoned before "
    "the next actor call. This iteration had already attempted and disposed of "
    "candidates (each has its own experiment row; any patch is retained under "
    "<store>/patches) — this row is no verdict on them")

#: One row per candidate abandoned INSIDE an iteration, recorded when it is
#: abandoned (`iterate(record_abandoned=...)`), never only folded into the
#: iteration's final outcome. Before this, a candidate refused at critic pass 2 or at
#: the pre-build gate survived only as a prompt string for the next round: when a
#: later round produced the iteration's outcome, the refused candidate -- its
#: reason, its gate and its patch pointer -- was in no experiment row at all.
CANDIDATE_DISPOSITIONS = ("hypothesis_rejected", "patch_rejected", "gate_refused",
                          "resume_rejected")

#: A resumed candidate (`iterate(resume=...)`) that failed re-validation: its patch no
#: longer applies, its bytes changed, its anchor moved, or a CURRENT gate refused it.
#: Recorded as a disposition (its own row, never an iteration) and its checkpoint is
#: consumed, so it is never retried in a loop; the iteration then draws fresh work.
RESUME_REJECTED = "resume_rejected"

#: The resumable checkpoint an outcome row carries (`resume_checkpoints`). The stop
#: path, a provider transient and every gate refusal write one, so work in flight at a
#: stop or refusal survives to the next launch instead of being re-derived: DS41 runs
#: 3-9c dropped ~300 actor-min and ~120 measure-min that way, and 0 of ~15 such
#: attempts ever reached a build. Consumer: `resume.py`.
CHECKPOINT_SCHEMA = "epyc.autokernel.resume_checkpoint.v1"


class RunAborted(RuntimeError):
    """The run stopped because iterations were failing systematically."""


class MeasurementInvalid(RuntimeError):
    """An original arm contradicted its instrument; not a candidate null.

    The serving owner supplies original raw facts after owned teardown. No measured
    comparison exists, and rescheduling must retain the candidate before lane reset.
    """

    def __init__(self, reason: str, record: dict):
        super().__init__(reason)
        self.record = record
        # Created by the original serving invocation only, never from stored JSON.
        self.reschedule = None


class MeasurementFailed(RuntimeError):
    """The instrument failed after collecting original facts; not a measured null."""

    def __init__(self, reason: str, record: dict | None = None):
        super().__init__(reason)
        self.record = {} if record is None else record


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


class ConfirmVetoed(RuntimeError):
    """The confirm rung refused to promote a screen keep (§5.3, D1-D6).

    Raised by the COMMIT callable when the two-rung gate is configured and the
    keep-candidate showed a decisive regression (or an uncalibrated surface) on the
    production-shaped confirm rung. Not an error and not a null: the screen
    measured a real positive, the confirm measured why it must not be committed --
    R23-5's +17.26%-at-b1 / -1.46%-at-b8 inversion is the class. The candidate is
    recorded as `keep_candidate` with the screen comparison; the full confirm
    record (both measurements) is in `<store>/confirm/`.
    """


class InteractionRegression(RuntimeError):
    """A kept source change reproduced a whole-bundle regression and was rolled back."""


class ActorTransient(RuntimeError):
    """The actor provider failed in a way worth retrying.

    Defined HERE, not in `actors`, so `iterate` can catch it without importing the
    concrete actor module (which imports this one). A transient must end an
    ITERATION, never the run: the superseded controller let provider faults escape as
    terminal, so a codex 401 on 2026-08-26 took down 284 attempts in 23 minutes.
    """


class AuthorReportMissing(ActorTransient):
    """The author's reply carried no usable `{"paths": [...]}` report (the concrete
    `actors.AuthorReplyMissing`). Only the author raises it. `iterate` then derives the
    report from the lane diff when it was given the lane and the base the loop reset
    it to for this draw (`author_lane`); with no diff, or no such lane, it ends the
    iteration as any provider transient does (`planner_transient`, with the author
    resume checkpoint). `failure_class` is the call's (`output_capped_empty`, ...)."""

    failure_class: str | None = None


#: `report_source` on a candidate whose paths were derived from the lane diff because
#: the author's reply carried no report (`AuthorReportMissing`).
REPORT_SOURCE_LANE_DIFF = "lane_diff"


class ActorStopped(ActorTransient):
    """A stop was asked while an actor call was in flight, and the actor was ended.

    NOT retryable. Before it existed a stop could not reach a planner call: the call
    IS the forming stage, so `should_abandon` was only polled after it returned, and
    the retry loop read the signal-killed actor (rc -15) as a provider transient and
    launched a new one (DS41 run 7, 2026-09-24 10:17). `iterate` records it as
    `stopped_mid_formation`, the outcome the stage-boundary poll already produces.
    """


@dataclass(frozen=True)
class Abstain:
    """A planner's truthful conclusion that this turn has no feasible answer."""

    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("a planner abstention must carry a reason")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "relies_on_claims", tuple(self.relies_on_claims or ()))
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


class Planner(Protocol):
    def propose(self, context: Mapping[str, Any]) -> Hypothesis | Abstain: ...
    def author(self, hypothesis: Hypothesis,
               context: Mapping[str, Any]) -> tuple[str, ...] | Abstain: ...


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
    invalid_measurement: dict | None = None
    instrument_failure: dict | None = None
    integrity_screen: dict | None = None
    attempt_identity: str | None = None
    exact_repeat_dispatch_count: int | None = None
    duplicate_of: str | None = None
    prior_effect: float | None = None
    prior_epoch: str | None = None
    refusal_gate: str | None = None
    candidate_diff_sha256: str | None = None
    mechanism_ablation: dict | None = None
    # Observe-only planner telemetry.  These fields describe the formation path
    # which produced the outcome; they do not alter either budget or selection.
    hypothesis_round: int = 0
    patch_round: int = 0
    prior_rejection_prompt: bool = False
    validator_provenance: list[dict] = field(default_factory=list)
    # Observed formation lineage only. Detached lanes are logical branches, not
    # additional Git champion refs; none of these fields selects work.
    spawn_parent: str | None = None
    branch_id: str | None = None
    width: int | None = None
    depth: int | None = None
    # Populated only after the append-only experiment store commits this outcome.
    # Kept out of to_attempt() so the receipt cannot recursively hash itself.
    journal_receipt: dict | None = None
    # Where this candidate's patch was retained (<store>/patches), set by the
    # owner that retained it; and the candidates this iteration disposed of before
    # its final outcome, each already recorded as its own row.
    retained_patch: dict | None = None
    abandoned_candidates: list[dict] = field(default_factory=list)
    # Resume lineage and checkpoints (`resume.py`). `resumed_from` names the
    # checkpoint (`<attempt_id>#<index>`) this candidate was resumed from;
    # `resume_checkpoints` is what a later launch needs to resume THIS one.
    resumed_from: str | None = None
    resume_stage: str | None = None
    resume_checkpoints: list[dict] = field(default_factory=list)
    # "lane_diff" when the candidate's paths were derived from the lane because the
    # author's reply carried no report; `author_report_recovery` says why and what.
    report_source: str | None = None
    author_report_recovery: dict | None = None

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
        if self.invalid_measurement is not None:
            row["invalid_measurement"] = self.invalid_measurement
        if self.instrument_failure is not None:
            row["instrument_failure"] = self.instrument_failure
        if self.integrity_screen is not None:
            row["integrity_screen"] = self.integrity_screen
        for key in ("attempt_identity", "exact_repeat_dispatch_count", "duplicate_of",
                    "prior_effect", "prior_epoch", "refusal_gate", "candidate_diff_sha256"):
            if getattr(self, key) is not None:
                row[key] = getattr(self, key)
        row.update({
            "hypothesis_round": self.hypothesis_round,
            "patch_round": self.patch_round,
            "prior_rejection_prompt": self.prior_rejection_prompt,
        })
        if self.validator_provenance:
            row["validator_provenance"] = self.validator_provenance
        if self.retained_patch is not None:
            row["retained_patch"] = self.retained_patch
        if self.abandoned_candidates:
            row["abandoned_candidates"] = self.abandoned_candidates
        if self.resumed_from is not None:
            row["resumed_from"] = self.resumed_from
            row["resume_stage"] = self.resume_stage
        if self.resume_checkpoints:
            row["resume_checkpoints"] = [dict(ck) for ck in self.resume_checkpoints]
        if self.report_source is not None:
            row["report_source"] = self.report_source
            row["author_report_recovery"] = self.author_report_recovery
        for key in ("spawn_parent", "branch_id", "width", "depth"):
            if getattr(self, key) is not None:
                row[key] = getattr(self, key)
        if self.hypothesis is not None:
            from . import claims
            split = claims.keep_claims(
                status=self.status,
                mechanism_id=self.hypothesis.mechanism_id,
                statement=self.hypothesis.statement,
                comparison=(self.comparison.to_dict()
                            if self.comparison is not None else None),
                gates=[verdict.to_dict() for verdict in self.gate_verdicts],
                ablation=self.mechanism_ablation)
            if split is not None:
                row["claims"] = split
        return row


#: Outcomes that end an iteration with accepted work still in flight. Their row
#: carries `resume_checkpoints` (the in-flight stage, and the latest gate-refused
#: patch) so a later launch resumes rather than re-derives it.
RESUMABLE_STATUSES = frozenset({"stopped_mid_formation", "planner_transient"})

#: Checkpoint fields the recording OWNER binds (never carried from an older row).
_OWNER_BOUND = frozenset({"anchor_commit", "epoch_sha256", "target"})


def gate_rules_fingerprint() -> str:
    """Digest of the deterministic gate rules a refusal was made under.

    A `gate_refused` checkpoint records it; `resume.py` treats a refusal by a rule
    gate as resumable only once this has changed (or was never recorded). Coarse on
    purpose: any change to `gates.py` re-opens the question, and the resumed
    candidate re-runs every current gate before any build or measurement.
    """
    import hashlib
    return hashlib.sha256(Path(gates.__file__).read_bytes()).hexdigest()


def _mark_report_source(outcome: "Outcome", progress: Mapping[str, Any]) -> None:
    """Stamp a candidate whose paths came from the lane diff, a lane-root reply file,
    or a normalized `<path>: <prose>` entry (this patch round)."""
    recovery = progress.get("report_recovery")
    if recovery is not None and outcome.hypothesis is not None \
            and outcome.report_source is None:
        outcome.report_source = recovery.get("report_source") or REPORT_SOURCE_LANE_DIFF
        outcome.author_report_recovery = dict(recovery)


def _recover_author_report(missing: AuthorReportMissing, author_lane, hypothesis,
                           before_tree: str | None,
                           progress: dict[str, Any]) -> tuple[str, ...]:
    """The author's paths, derived from the lane diff, or re-raise `missing`.

    Only for a lane the owner reset for this draw (`author_lane`). No diff (or no
    change by this call) re-raises the original transient unchanged: that is the
    refusal the loop always made. A lane that cannot stand in for the report re-raises
    it with the reason appended. Never invents a path: every one is in the diff."""
    if author_lane is None:
        raise missing
    worktree, base = Path(author_lane[0]), str(author_lane[1] or "")
    try:
        paths = integrity.lane_diff_report(worktree, base,
                                           target_surface=hypothesis.target_surface,
                                           before_tree=before_tree)
    except Exception as exc:      # noqa: BLE001 -- LaneDiffRefused, git faults alike
        raise _extended(missing, f"{missing}; lane diff not usable as the report: "
                                 f"{type(exc).__name__}: {exc}"[:2000]) from missing
    if not paths:
        raise missing
    progress["report_recovery"] = {
        "report_source": REPORT_SOURCE_LANE_DIFF, "base": base, "paths": list(paths),
        "failure_class": getattr(missing, "failure_class", None),
        "path_normalized": False, "reply_refusal": str(missing)[:500]}
    # Beside the author call's own metrics row (`actors.ACTOR_REPLY_DIR`).
    from . import actor_metrics
    actor_metrics.record_report_source(
        worktree.parent / actor_metrics.REPLY_DIR_NAME,
        {**progress["report_recovery"], "mechanism_id": hypothesis.mechanism_id,
         "workspace": str(worktree)})
    return tuple(paths)


def _note_author_report(paths, progress: dict[str, Any]) -> None:
    """Carry a non-default author report source onto the outcome: a reply read from a
    lane-root report file, or an entry normalized from `<path>: <prose>` (the author
    already wrote its `actor_report_source` row)."""
    report = getattr(paths, "report", None)
    if not isinstance(report, Mapping):
        return
    if report.get("path_normalized") or report.get("report_source") not in (None, "reply"):
        progress["report_recovery"] = dict(report)


def _extended(missing: AuthorReportMissing, message: str) -> AuthorReportMissing:
    """`missing` again (same type, same `failure_class`) with a longer message."""
    failure_class = getattr(missing, "failure_class", None)
    try:
        extended = type(missing)(message, failure_class=failure_class)
    except TypeError:
        extended = type(missing)(message)
    extended.failure_class = failure_class
    return extended


def _pending_checkpoints(progress: Mapping[str, Any]) -> list[dict]:
    pending: list[dict] = []
    for key in ("inflight", "build"):
        entry = progress.get(key)
        if entry is not None and entry not in pending:
            pending.append(dict(entry))
    return pending


def _disposal_summary(abandoned: Sequence[Mapping[str, Any]]) -> str:
    """One line naming what this iteration disposed of before it was stopped."""
    parts = [f"{row.get('status')} by {row.get('refusal_gate')} "
             f"(round {row.get('hypothesis_round')}.{row.get('patch_round')}): "
             f"{str(row.get('reason') or '')[:200]}" for row in abandoned]
    return f"disposed before the stop ({len(parts)}): " + "; ".join(parts)


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


def _gate_provenance(verdict: gates.Verdict) -> dict:
    kind = "oracle" if verdict.gate in {"correctness", "determinism"} else "script"
    return {
        "decision": f"gate:{verdict.gate}",
        "validator_identity": f"autokernel.gates.{verdict.gate}",
        "validator_kind": kind,
        "independence": "non_model",
        "evidence_inspected": ["candidate build artifact", verdict.gate],
        "accepted": verdict.passed,
        "reason": verdict.reason or None,
        "changed_subsequent_search": not verdict.passed,
    }


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
    if comparison.decisive and comparison.effect < 0:
        return (f"DECISIVE REGRESSION — effect {comparison.effect * 100:+.3f}% "
                f"exceeded the {comparison.noise_floor_pct:.3f}% noise floor; "
                "candidate rejected, champion unchanged")
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
            should_abandon: Callable[[], bool] | None = None,
            record_reschedule: Callable[[Outcome], bool] | None = None,
            accumulate_valid_positive: bool = False,
            validate_candidate: Callable[[Hypothesis, Sequence[str]], Any] | None = None,
            formation_guard=None, reserve_candidate=None,
            record_abandoned: Callable[[Outcome], None] | None = None,
            resume=None,
            author_lane: tuple[Path, str] | None = None
            ) -> Outcome:
    """One full turn. Pure control flow: every side effect is an injected callable.

    `should_abandon` is the DRAIN TIER for a lane that does not hold the serialized
    tail: polled at every stage boundary of FORMATION (before each planner call,
    critic pass and authoring turn), and never inside the tail — a candidate that
    reaches the tail finishes build → oracle → A/B → commit exactly as before, so a
    stop can never kill a measurement mid-A/B.

    `record_abandoned` receives one `CANDIDATE_DISPOSITIONS` outcome for every
    hypothesis or patch this turn abandons before its final outcome, at the moment
    it is abandoned (so a later stop, lane error or crash cannot erase it). The
    final outcome lists them again in `abandoned_candidates`.

    `resume` (a `resume.ResumePoint`) seeds ONE extra round, before the fresh
    hypothesis rounds, with work a previous launch already paid for: at stage
    "build" an accepted patch goes straight to the host checks, the CURRENT gates
    and the measurement (no planner, critic or author call); at stage "author" an
    accepted hypothesis resumes authoring with its original verdict and rejection
    history. Every re-validation failure is disposed as `resume_rejected` and the
    iteration continues with fresh work. A stop, provider transient or gate refusal
    leaves `resume_checkpoints` on its row so the next launch can resume it.

    `author_lane` is `(worktree, base)`: the lane the author edits and the commit the
    OWNER reset it to for this draw (`pipeline.run_pool`). Only with it, an author
    reply that carries no report (`AuthorReportMissing`) is answered from the lane diff
    (`integrity.lane_diff_report`, recorded `report_source: "lane_diff"`), and the
    normal gates judge that diff. Without it, or with no diff, the transient stands.
    """
    working = dict(context)
    abandoned: list[dict] = []
    progress: dict[str, Any] = {"inflight": None, "build": None, "report_recovery": None}
    hypothesis_reasons: list[str] = []
    round_telemetry = {
        "hypothesis_round": 0,
        "patch_round": 0,
        "prior_rejection_prompt": False,
    }
    validator_provenance: list[dict] = []

    def observed(outcome: Outcome) -> Outcome:
        outcome.hypothesis_round = int(round_telemetry["hypothesis_round"])
        outcome.patch_round = int(round_telemetry["patch_round"])
        outcome.prior_rejection_prompt = bool(
            round_telemetry["prior_rejection_prompt"])
        # An admissible A/B result is fed to campaign memory regardless of sign:
        # keeps move the anchor, while nulls/regressions close that exact attempt.
        # Record that causal use, rather than leaving every measurement validator
        # falsely marked as having no effect on subsequent search.
        for row in validator_provenance:
            if row.get("decision") == "measurement:paired_ab":
                row["changed_subsequent_search"] = outcome.status in {
                    "kept", "keep_candidate", "measured_null", "regression",
                    "confirm_vetoed"}
        outcome.validator_provenance = list(validator_provenance)
        if outcome.status == "stopped_mid_formation" and abandoned \
                and outcome.reasons[:1] == [STOPPED_MID_FORMATION]:
            outcome.reasons = [STOPPED_AFTER_DISPOSALS, *outcome.reasons[1:],
                               _disposal_summary(abandoned)]
        outcome.abandoned_candidates = list(abandoned)
        if resume is not None and outcome.hypothesis is not None \
                and outcome.hypothesis is resume.hypothesis:
            outcome.resumed_from = resume.checkpoint_id
            outcome.resume_stage = resume.stage
        if outcome.status in RESUMABLE_STATUSES and not outcome.resume_checkpoints:
            outcome.resume_checkpoints = _pending_checkpoints(progress)
        _mark_report_source(outcome, progress)
        return outcome

    try:
        return observed(_iterate(planner=planner, critic=critic, working=working,
                        hypothesis_reasons=hypothesis_reasons, measure=measure,
                        gate=gate, commit=commit,
                        hypothesis_rounds=hypothesis_rounds,
                        patch_rounds=patch_rounds, on_step=_safe_step(on_step),
                        tail_session=tail_session,
                        should_abandon=should_abandon or (lambda: False),
                        record_reschedule=record_reschedule,
                        accumulate_valid_positive=accumulate_valid_positive,
                        validate_candidate=validate_candidate or (lambda _h, _p: None),
                        formation_guard=formation_guard or (lambda _h, _c: None),
                        reserve_candidate=reserve_candidate,
                        round_telemetry=round_telemetry,
                        validator_provenance=validator_provenance,
                        record_abandoned=record_abandoned, abandoned=abandoned,
                        resume=resume, progress=progress, author_lane=author_lane))
    except TailRefused as exc:
        # The candidate was formed and never measured. Carry the hypothesis: the
        # patch may well still help against the champion that displaced it, and the
        # planner is told to look at these FIRST.
        return observed(Outcome("superseded", getattr(exc, "hypothesis", None), [str(exc)]))
    except ActorStopped as exc:
        # Raised outside the four formation calls (a wrapper planner): still a stop.
        return observed(Outcome("stopped_mid_formation", getattr(exc, "hypothesis", None),
                                [STOPPED_MID_FORMATION, str(exc)]))
    except ActorTransient as exc:
        # The provider failed, not the science. This ends the ITERATION and is
        # recorded as such; the run continues, and a streak becomes visible in
        # experiments.md rather than taking the campaign down with it.
        return observed(Outcome("planner_transient", None, [str(exc)]))
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
        return observed(Outcome("bench_failed", None, [str(exc)]))


def _iterate(*, planner, critic, working, hypothesis_reasons, measure, gate, commit,
             hypothesis_rounds, patch_rounds, on_step=lambda _label: None,
             tail_session=nullcontext,
             should_abandon=lambda: False, record_reschedule=None,
             accumulate_valid_positive=False,
             validate_candidate=lambda _hypothesis, _paths: None,
             formation_guard=lambda _hypothesis, _context: None,
             reserve_candidate=None, round_telemetry=None,
             validator_provenance=None, record_abandoned=None,
             abandoned=None, resume=None, progress=None, author_lane=None) -> Outcome:
    last_proposed: Hypothesis | None = None
    round_telemetry = round_telemetry if round_telemetry is not None else {}
    validator_provenance = (validator_provenance if validator_provenance is not None
                            else [])

    abandoned = abandoned if abandoned is not None else []
    progress = progress if progress is not None else {"inflight": None, "build": None}

    def is_resumed(hypothesis) -> bool:
        return resume is not None and hypothesis is not None \
            and hypothesis is resume.hypothesis

    def lineage(hypothesis) -> dict:
        if is_resumed(hypothesis):
            return {"resumed_from": resume.checkpoint_id,
                    "resume_depth": int(resume.checkpoint.get("resume_depth") or 0) + 1}
        return {"resumed_from": None, "resume_depth": 0}

    def latest_accepted(decision: str) -> dict | None:
        for row in reversed(validator_provenance):
            if row.get("decision") == decision and row.get("accepted"):
                return dict(row)
        return None

    def checkpoint(stage: str, hypothesis, **fields) -> dict:
        """What a later launch needs to resume this candidate at `stage`.

        The owner (`run.py`) binds the anchor, epoch and target it was formed
        against when it records the row; `resume.py` refuses a mismatch.
        """
        return {"schema": CHECKPOINT_SCHEMA, "stage": stage,
                "hypothesis": hypothesis.to_dict(),
                "critic_hypothesis": latest_accepted("critic:hypothesis"),
                "hypothesis_round": int(round_telemetry.get("hypothesis_round", 0)),
                "patch_round": int(round_telemetry.get("patch_round", 0)),
                **lineage(hypothesis), **fields}

    def dispose(hypothesis, status: str, reason: str | None, *, refusal_gate: str,
                verdicts=(), resume_checkpoint: dict | None = None) -> None:
        """Record one abandoned candidate NOW, with its reason and gate.

        A recording fault is carried on the iteration's final outcome rather than
        raised: raising here would turn one refused candidate into a lane error that
        discards the rest of the turn, which is the loss this exists to stop.
        """
        candidate = Outcome(status, hypothesis,
                            [reason or f"{refusal_gate} refused without a reason"],
                            gate_verdicts=list(verdicts), refusal_gate=refusal_gate)
        candidate.hypothesis_round = int(round_telemetry.get("hypothesis_round", 0))
        candidate.patch_round = int(round_telemetry.get("patch_round", 0))
        candidate.prior_rejection_prompt = bool(
            round_telemetry.get("prior_rejection_prompt", False))
        candidate.validator_provenance = [dict(row) for row in validator_provenance]
        if is_resumed(hypothesis):
            candidate.resumed_from = resume.checkpoint_id
            candidate.resume_stage = resume.stage
        if resume_checkpoint is not None:
            candidate.resume_checkpoints = [resume_checkpoint]
        _mark_report_source(candidate, progress)
        record_error = None
        if record_abandoned is not None:
            try:
                record_abandoned(candidate)
            except Exception as exc:      # noqa: BLE001 -- see docstring
                record_error = f"{type(exc).__name__}: {exc}"
                print(f"warning: abandoned-candidate record failed: {record_error}",
                      file=sys.stderr)
        if resume_checkpoint is not None:
            # The owner retained the diff while recording. A later stop row carries
            # this checkpoint too, so a failed disposal record cannot lose it.
            progress["build"] = {**resume_checkpoint, "retained_patch": (
                resume_checkpoint.get("retained_patch") or candidate.retained_patch)}
        summary = {
            "status": status, "refusal_gate": refusal_gate,
            "reason": candidate.reasons[0],
            "mechanism_id": getattr(hypothesis, "mechanism_id", None),
            "hypothesis_round": candidate.hypothesis_round,
            "patch_round": candidate.patch_round,
            "retained_patch": candidate.retained_patch,
            "attempt_identity": candidate.attempt_identity,
            "candidate_diff_sha256": candidate.candidate_diff_sha256,
        }
        if record_error is not None:
            summary["record_error"] = record_error
        if candidate.resumed_from is not None:
            summary["resumed_from"] = candidate.resumed_from
        abandoned.append(summary)

    def mark_search_changed(decision: str) -> None:
        for row in reversed(validator_provenance):
            if row["decision"] == decision and not row["accepted"]:
                row["changed_subsequent_search"] = True
                return

    def stopped() -> Outcome:
        # Names whatever was in flight, exactly as a refusal row must.
        return Outcome("stopped_mid_formation", last_proposed,
                       [STOPPED_MID_FORMATION])

    def actor_call(call, *args):
        # A stop asked DURING an actor call ends the call (the actor module TERMs its
        # child) and surfaces here as ActorStopped: the same outcome the boundary poll
        # gives, still naming whatever was in flight -- never a provider transient.
        try:
            return call(*args), None
        except ActorStopped:
            return None, stopped()

    # A resumed candidate is one EXTRA round ahead of the fresh ones: it replaces no
    # fresh round, so a rejected resume still leaves the iteration its full budget.
    schedule = ([resume] if resume is not None else []) + [None] * hypothesis_rounds
    for hypothesis_index, resumed in enumerate(schedule):
        # Polled BEFORE each actor call, never after: the whole point is that no
        # further multi-minute call is drawn once the run has been told to stop.
        if resumed is not None and getattr(resumed, "stale", None) is None:
            # Named before the poll, so a stop here still carries the claimed
            # checkpoint forward instead of consuming it.
            last_proposed = resumed.hypothesis
            progress["inflight"] = {
                **{key: value for key, value in resumed.checkpoint.items()
                   if key not in _OWNER_BOUND}, **lineage(resumed.hypothesis)}
        else:
            progress["inflight"] = None
        if should_abandon():
            return stopped()
        working["prior_hypothesis_rejections"] = list(hypothesis_reasons)
        round_telemetry["hypothesis_round"] = hypothesis_index + 1
        round_telemetry["patch_round"] = 0
        if hypothesis_reasons:
            round_telemetry["prior_rejection_prompt"] = True
            mark_search_changed("critic:hypothesis")
        if resumed is not None:
            hypothesis = resumed.hypothesis
            stale = getattr(resumed, "stale", None)
            if stale is not None:
                dispose(hypothesis, RESUME_REJECTED,
                        f"resume re-validation refused ({stale[0]}): {stale[1]}",
                        refusal_gate=f"resume:{stale[0]}")
                continue
            on_step(resumed.label)
            # The verdicts that admitted it are CARRIED, and marked so: no critic
            # ran in this launch, and the provenance must not claim one did.
            for row in resumed.provenance:
                validator_provenance.append({**row, "resumed_from": resumed.checkpoint_id,
                                             "changed_subsequent_search": False})
            repeat_reason = formation_guard(hypothesis, working)
            if repeat_reason:
                dispose(hypothesis, RESUME_REJECTED,
                        f"resume re-validation refused (do_not_repeat): {repeat_reason}",
                        refusal_gate="resume:do_not_repeat")
                continue
        else:
            on_step("proposing a hypothesis")
            hypothesis, halted = actor_call(planner.propose, working)
            if halted is not None:
                return halted
            if isinstance(hypothesis, Abstain):
                return Outcome("abstained", None, [hypothesis.reason])
            last_proposed = hypothesis
            repeat_reason = formation_guard(hypothesis, working)
            if repeat_reason:
                return Outcome("refused_at_formation", hypothesis, [repeat_reason],
                               refusal_gate="do_not_repeat")

        # ---- CRITIC PASS 1: the hypothesis, before any patch exists ----------
        if should_abandon():
            return stopped()
        # Only the installed canonical runtime options get the deterministic fast
        # path. A serialized pair is not itself installation or launch authority:
        # the exact current anchor/options must also be in this owner's context,
        # and the tail gate still rechecks ownership and correctness before launch.
        pair = hypothesis.runtime_pair
        prevalidated_runtime = (
            pair is not None and pair.anchor.backend in {"cpu", "gpu"}
            and pair.anchor.to_dict() == working.get("runtime_anchor")
            and pair.dimension.kind in {"threads", "cpu_list", "numa_policy", "env"}
            and (pair.dimension.kind != "env" or pair.dimension.candidate["key"]
                 in working.get("runtime_env_keys", ())))
        if resumed is not None:
            pass    # admitted by the carried verdict above
        elif prevalidated_runtime:
            on_step("prevalidated runtime option: deterministic checks, no critic call")
        else:
            on_step("critic pass 1: reviewing the hypothesis")
            verdict, halted = actor_call(critic.review_hypothesis, hypothesis, working)
            if halted is not None:
                return halted
            validator_provenance.append(_critic_provenance(
                critic, verdict, decision="critic:hypothesis",
                evidence=("hypothesis", "planner context")))
            if not verdict.accepted:
                # Verbatim, so the planner can answer the objection rather than guess.
                hypothesis_reasons.append(verdict.reason)
                dispose(hypothesis, "hypothesis_rejected", verdict.reason,
                        refusal_gate="critic:hypothesis")
                continue

        resumed_build = resumed is not None and resumed.stage == "build"
        materialized = False
        patch_reasons: list[str] = (list(resumed.prior_patch_rejections)
                                    if resumed is not None else [])
        round_count = (1 if hypothesis.runtime_pair is not None
                       else max(1, int(resumed.patch_rounds)) if resumed is not None
                       else patch_rounds)
        for patch_index in range(round_count):
            if hypothesis.runtime_pair is None and not resumed_build:
                # The in-flight checkpoint a stop or provider transient leaves behind:
                # an accepted hypothesis, its verdict and every patch rejection so far.
                progress["inflight"] = checkpoint(
                    "author", hypothesis, prior_patch_rejections=list(patch_reasons),
                    patch_rounds_remaining=round_count - patch_index)
            if should_abandon():
                return stopped()
            working["prior_patch_rejections"] = list(patch_reasons)
            round_telemetry["patch_round"] = patch_index + 1
            if patch_reasons:
                round_telemetry["prior_rejection_prompt"] = True
                mark_search_changed("critic:patch")
            paths = ()
            integrity_screen = None
            if hypothesis.runtime_pair is None:
                if resumed_build:
                    # No author call: restore the exact accepted bytes, re-verified
                    # (digest, anchor, clean apply) by the owner at this moment.
                    on_step("restoring the retained patch (no author call)")
                    try:
                        paths = tuple(resumed.materialize())
                        materialized = True
                    except Exception as exc:      # noqa: BLE001 -- a stale resume
                        dispose(hypothesis, RESUME_REJECTED,
                                f"resume re-validation refused: {exc}",
                                refusal_gate=f"resume:{getattr(exc, 'check', 'materialize')}")
                        break
                else:
                    on_step("authoring the patch")
                    progress["report_recovery"] = None
                    before_tree = None
                    if author_lane is not None:
                        try:
                            before_tree = integrity.candidate_tree(Path(author_lane[0]))
                        except Exception:      # noqa: BLE001 -- recovery then refuses
                            before_tree = None
                    try:
                        paths, halted = actor_call(planner.author, hypothesis, working)
                    except AuthorReportMissing as missing:
                        paths, halted = _recover_author_report(
                            missing, author_lane, hypothesis, before_tree, progress), None
                        on_step("authoring report derived from the lane diff")
                    else:
                        _note_author_report(paths, progress)
                    if halted is not None:
                        return halted
                    if isinstance(paths, Abstain):
                        return Outcome("abstained", hypothesis, [paths.reason])
                # A declared path list is a claim, not an isolation boundary.  The
                # injected host check resolves the full worktree before review/build.
                try:
                    checked = validate_candidate(hypothesis, paths)
                    integrity_screen = (checked.to_dict() if hasattr(checked, "to_dict")
                                        else checked)
                except integrity.IntegrityRefused as exc:
                    if resumed_build:
                        dispose(hypothesis, RESUME_REJECTED,
                                f"resume re-validation refused (integrity): {exc}",
                                refusal_gate="resume:integrity")
                        break
                    return Outcome(
                        "integrity_refused", hypothesis, [str(exc)],
                        integrity_screen={"refusal_class": exc.refusal_class})

            # ---- CRITIC PASS 2: the diff, BEFORE the build ------------------
            if should_abandon():
                return stopped()
            if hypothesis.runtime_pair is None and resumed_build:
                on_step("critic pass 2: carried verdict (resumed at build)")
            elif hypothesis.runtime_pair is None:
                on_step("critic pass 2: reviewing the diff")
                patch_verdict, halted = actor_call(critic.review_patch, hypothesis, paths,
                                                   working)
                if halted is not None:
                    return halted
                validator_provenance.append(_critic_provenance(
                    critic, patch_verdict, decision="critic:patch",
                    evidence=("candidate diff", "declared paths", "planner context")))
                if not patch_verdict.accepted:
                    # The hypothesis is untouched: a bad patch is not evidence against
                    # the idea it was trying to implement.
                    patch_reasons.append(patch_verdict.reason)
                    dispose(hypothesis, "patch_rejected", patch_verdict.reason,
                            refusal_gate="critic:patch")
                    continue
                # The critic is a verifier, not a second author. Re-run the host-owned
                # whole-tree validation after its call and before entering the build
                # tail. A critic that mutates even a declared file changes the tree
                # reviewed by the first integrity pass and hard-refuses the candidate.
                try:
                    post_critic = validate_candidate(hypothesis, paths)
                except integrity.IntegrityRefused as exc:
                    return Outcome(
                        "integrity_refused", hypothesis, [str(exc)],
                        integrity_screen={"refusal_class": exc.refusal_class})
                before_tree = ((integrity_screen or {}).get("measured_tree")
                               if isinstance(integrity_screen, Mapping) else None)
                after_tree = ((post_critic or {}).get("measured_tree")
                              if isinstance(post_critic, Mapping) else None)
                if before_tree is not None and after_tree != before_tree:
                    return Outcome(
                        "integrity_refused", hypothesis,
                        ["critic mutated the candidate worktree: "
                         f"tree changed {before_tree} -> {after_tree}"],
                        integrity_screen={"refusal_class": "critic_tree_mutation",
                                          "measured_tree": before_tree,
                                          "post_critic_tree": after_tree})

            on_step("checking runtime treatment and correctness" if hypothesis.runtime_pair
                    is not None else "building and gating")
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
                    if reserve_candidate is not None and hypothesis.runtime_pair is None:
                        try:
                            reserve_candidate(hypothesis, paths)
                        except Exception as exc:
                            if type(exc).__name__ != "DispatchRefused":
                                raise
                            return Outcome("refused_duplicate", hypothesis, [str(exc)],
                                           attempt_identity=getattr(
                                               exc, "attempt_identity", None),
                                           duplicate_of=getattr(exc, "duplicate_of", None),
                                           prior_effect=getattr(exc, "prior_effect", None),
                                           prior_epoch=getattr(exc, "prior_epoch", None),
                                           refusal_gate="exact_attempt_identity")
                    passed, verdicts = gate(hypothesis, paths)
                    validator_provenance.extend(_gate_provenance(v) for v in verdicts)
                    if not passed:
                        if hypothesis.runtime_pair is not None:
                            return Outcome("runtime_refused", hypothesis,
                                           [verdicts[-1].reason if verdicts else "gate refused"],
                                           gate_verdicts=verdicts)
                        # Compile and correctness failures loop back the same way; the
                        # toolchain's own message is the reason, so no critic is needed.
                        patch_reasons.append(
                            verdicts[-1].reason if verdicts else "gate refused")
                        refusing_gate = verdicts[-1].gate if verdicts else "gate"
                        if resumed_build:
                            # The CURRENT gates refused a resumed patch: the old
                            # verdict is never trusted, and a refusal is final.
                            dispose(hypothesis, RESUME_REJECTED,
                                    "resume re-validation refused (current gate "
                                    f"{refusing_gate}): {patch_reasons[-1]}",
                                    refusal_gate=refusing_gate, verdicts=verdicts)
                            continue
                        # Recorded as its own row, naming the deterministic rule: an
                        # accepted patch that never reaches the build must say which
                        # gate stopped it and where its diff is kept. It carries a
                        # build checkpoint: if that rule changes, the next launch can
                        # take this exact patch straight to the build (`resume.py`).
                        dispose(hypothesis, "gate_refused", patch_reasons[-1],
                                refusal_gate=refusing_gate, verdicts=verdicts,
                                resume_checkpoint=checkpoint(
                                    "build", hypothesis,
                                    critic_patch=latest_accepted("critic:patch"),
                                    refusal_gate=refusing_gate,
                                    refusal_reason=patch_reasons[-1],
                                    gate_rules_fingerprint=gate_rules_fingerprint(),
                                    retained_patch=None))
                        continue

                    on_step("measuring A/B on the device")
                    measure_original = lambda: measure(hypothesis, paths)
                    rescheduled = False
                    while True:
                        try:
                            comparison = measure_original()
                            validator_provenance.append({
                                "decision": "measurement:paired_ab",
                                "validator_identity": "autokernel.bench.paired_ab",
                                "validator_kind": "script",
                                "independence": "non_model",
                                "evidence_inspected": [
                                    "anchor samples", "candidate samples",
                                    "noise-floor calibration", "residency evidence"],
                                "accepted": bool(comparison.decisive),
                                "reason": None,
                                "changed_subsequent_search": False,
                            })
                            break
                        except MeasurementInvalid as exc:
                            invalid = Outcome("measurement_invalid", hypothesis, [str(exc)],
                                gate_verdicts=verdicts, invalid_measurement=exc.record)
                            if (rescheduled or exc.reschedule is None or record_reschedule is None
                                    or (should_abandon is not None and should_abandon())
                                    or not record_reschedule(invalid)):
                                return invalid
                            # Already archived and charged by the pool; no reset,
                            # reauthor, rebuild, or valid-arm repetition. The same
                            # serialized tail still owns the original candidate.
                            rescheduled = True
                            on_step("rescheduling original invalid CPU server arm (one bounded retry)")
                            if should_abandon is not None and should_abandon():
                                return Outcome("stopped_before_reschedule", hypothesis,
                                    ["STOP after invalid arm archival; no replacement server launched"],
                                    gate_verdicts=verdicts)
                            measure_original = exc.reschedule
                        except MeasurementFailed as exc:
                            return Outcome("bench_failed", hypothesis, [str(exc)],
                                gate_verdicts=verdicts, instrument_failure=exc.record)
                    # R23-44 compound-then-gate: an experimental serving source
                    # candidate may be smaller than the process-unit floor and still
                    # belong in the working accumulator.  It must still be a valid,
                    # calibrated, non-drifting positive observation.  Runtime recipe
                    # selection remains decisive-only: it has no source tree to
                    # compound and cannot be recovered by a later bundle gate.
                    accumulatable_positive = (
                        accumulate_valid_positive
                        and hypothesis.runtime_pair is None
                        and comparison.decisive is False
                        and comparison.noise_floor_pct is not None
                        and not comparison.drifting
                        and comparison.effect > 0)
                    if ((comparison.decisive and comparison.effect > 0)
                            or accumulatable_positive):
                        # A screen keep is a KEEP_CANDIDATE; with a confirm rung
                        # configured, `commit` measures it on the production shape
                        # and vetoes rather than committing (§5.3). Unconfigured,
                        # commit never raises ConfirmVetoed and this is the same
                        # single-rung keep as ever.
                        try:
                            head = commit(hypothesis, paths, comparison)
                        except ConfirmVetoed as veto:
                            return Outcome("keep_candidate", hypothesis,
                                           [str(veto)], comparison, verdicts,
                                           integrity_screen=integrity_screen)
                        except InteractionRegression as regression:
                            return Outcome("interaction_regression", hypothesis,
                                           [str(regression)], comparison, verdicts,
                                           integrity_screen=integrity_screen)
                        return Outcome("kept", hypothesis, [], comparison, verdicts, head,
                                       integrity_screen=integrity_screen)
            except MeasurementInvalid as exc:
                return Outcome("measurement_invalid", hypothesis, [str(exc)],
                               gate_verdicts=verdicts, invalid_measurement=exc.record)
            except MeasurementFailed as exc:
                return Outcome("bench_failed", hypothesis, [str(exc)],
                               gate_verdicts=verdicts, instrument_failure=exc.record)
            except TailRefused as exc:
                # Formed and never measured. Carry the hypothesis out so the planner
                # can reconsider it against the champion that displaced it.
                exc.hypothesis = hypothesis
                raise
            # A negative result IS a result. It is recorded with its mechanism and its
            # sample vector, because a loop whose record of failure is thinner than
            # its record of success teaches its planner to repeat the failures.
            return Outcome("runtime_observed" if hypothesis.runtime_pair is not None
                           else "regression" if comparison.decisive and comparison.effect < 0
                           else "measured_null", hypothesis,
                           ["Runtime observation retained; recipe selection requires original "
                            "strict anchor/instrument calibration and admission, not a source floor"
                            if hypothesis.runtime_pair is not None and comparison.decisive is None
                            else _null_reason(comparison)],
                           comparison, verdicts)

        # Patch budget spent. Control returns to the HYPOTHESIS loop, so the planner
        # may refine H knowing it could not be implemented cleanly.
        progress["inflight"] = None
        if materialized:
            # A rejected resume must not leave its bytes under the next fresh round.
            resumed.discard()
        hypothesis_reasons.extend(patch_reasons)

    # Hypothesis budget spent. H is NOT retired: it re-enters the pool carrying its
    # rejection history, because the profile moves and what was unsupported this week
    # may be the hotspot next week.
    # Carry the last hypothesis proposed, not None: a refusal row whose mechanism_id is
    # empty tells the operator that something was refused without saying WHAT, and it is
    # the row the dashboard shows most often. The refusal is of an idea; name the idea.
    return Outcome("refused_at_formation", last_proposed, hypothesis_reasons)


# `run`, the sequential driver, is GONE (R21-7, 2026-09-01). It outlived the
# sequential CLI path (deleted 2026-08-31) only as a test seam, and the suites that
# held it still — test_loop, test_anchor, test_production — now drive
# `pipeline.run_pool` as a one-lane pool, the same shape `run.py --workers 1`
# builds. Its `except RunAborted: raise` lives on as the pool lane's RunAborted
# handler; its blanket containment as the lane's `lane_error`; its per-outcome
# `archive.record` as `run.py`'s injected `record`. `iterate` is the whole of this
# module's control flow now, and the pool is its only driver.

__all__ = ["CANDIDATE_DISPOSITIONS", "CHECKPOINT_SCHEMA", "RESUMABLE_STATUSES",
           "RESUME_REJECTED", "STOPPED_AFTER_DISPOSALS", "gate_rules_fingerprint", "Abstain", "ActorStopped", "ActorTransient", "AuthorReportMissing", "REPORT_SOURCE_LANE_DIFF", "ConfirmVetoed", "InteractionRegression", "TailRefused", "RunAborted", "MeasurementInvalid", "MeasurementFailed", "Critic",
           "HYPOTHESIS_ROUNDS", "Hypothesis", "Outcome", "PATCH_ROUNDS",
           "Planner", "Review", "STOPPED_MID_FORMATION", "iterate"]
