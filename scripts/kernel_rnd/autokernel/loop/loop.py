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

import re
import sys
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from . import bench, gates, integrity

HYPOTHESIS_ROUNDS = 3
PATCH_ROUNDS = 2
#: Authoring attempts one ACCEPTED hypothesis gets across iterations, each of
#: `patch_rounds` rounds (`run.py --hypothesis-author-attempts`). A critic-accepted
#: idea is not evidence-free: DS41 run 10g threw `akm-q4k-x4t-avx512` back to the
#: planner after two patches failed on authoring defects (non-GCC intrinsics, wrong
#: arity, offset bugs), so the author's weakness retired the critic's verdict.
HYPOTHESIS_AUTHOR_ATTEMPTS = 3

#: Patch rounds ran out while the hypothesis stood accepted and the last rejection
#: was the AUTHOR's (a critic-pass-2 defect, or a compile/correctness gate). The row
#: carries an author checkpoint so the next draw re-authors it (`resume.py`), with
#: every patch rejection so far as author feedback. Not a verdict on the idea.
PATCH_ROUNDS_EXHAUSTED = "patch_rounds_exhausted"
#: Patch rounds ran out and the last rejection said the mechanism cannot be written
#: inside the admitted route (critic pass 2 tagged it `SCOPE`, or the `op_scope` rule
#: gate refused it). Resumable, but only once the scope rules change
#: (`scope_rules_fingerprint`): a widening re-admits it with no operator action.
SCOPE_BLOCKED = "scope_blocked"
#: The accepted hypothesis spent its authoring budget on author failures. Terminal,
#: no checkpoint; the reason names the budget and carries every patch rejection.
HYPOTHESIS_RETIRED = "hypothesis_retired"
#: The AUTHOR failed an accepted hypothesis without producing a checkable diff: it
#: abstained, returned no changed path, reported a change the lane does not hold, or
#: (best-of-N) every member did one of those. Operator rule 2026-09-26 (DS41 run 10h,
#: `akm-q4k-x4t-avx512` dropped as `abstained` a second time): an accepted hypothesis is
#: never discarded because only its AUTHORING failed. One authoring attempt is spent
#: and the row carries an author checkpoint whose feedback names every member's reason,
#: its ak-check result and its retained patch, so the next draw starts from them.
AUTHORING_FAILED = "authoring_failed"
#: The same, but every failure was the HARNESS's, not the author's: a provider
#: transient or timeout, a per-call budget, an output-capped final step (the edit was
#: truncated, so a "report_missing" is not the author's claim), an opencode store
#: error. No attempt is charged -- until `AUTHOR_HARNESS_FAILURE_CAP` of them in a row
#: for one hypothesis, whose last one IS charged, so a setup that fails every time
#: cannot re-author forever.
AUTHORING_HARNESS_FAILURE = "authoring_harness_failure"
#: Consecutive harness failures of one accepted hypothesis before the next is charged.
AUTHOR_HARNESS_FAILURE_CAP = 3
#: Critic-pass-1 transients (an empty reply, a provider error, a timeout) one PLANNER
#: hypothesis survives before it is dropped: each writes a "critic1" checkpoint the next
#: draw resumes at critic pass 1, before the planner is asked (DS41 run 10h batch 1:
#: `akm-verify-batch-solo-2rows`, 31 planner-minutes, lost to one empty critic reply).
CRITIC1_RETRIES = 3
#: Iteration outcomes whose row keeps an accepted hypothesis pending authoring.
PENDING_HYPOTHESIS_STATUSES = frozenset({PATCH_ROUNDS_EXHAUSTED, SCOPE_BLOCKED,
                                         AUTHORING_FAILED, AUTHORING_HARNESS_FAILURE})
#: Actor-call `failure_class` values (`actor_metrics`) that say the HARNESS ended the
#: author call: an output-capped empty final step, a per-call wall budget, opencode's
#: own store refusing a write, a refused tool permission that ended the session.
HARNESS_FAILURE_CLASSES = frozenset({"output_capped_empty", "budget_exhausted",
                                     "opencode_store_error", "permission_rejected"})
#: Author-member outcomes that are never the author's verdict (`bestof._Member.outcome`
#: vocabulary, shared by the single path): a provider transient, a contained error, a
#: call ended without a run stop.
HARNESS_OUTCOMES = frozenset({"transient", "error", "stopped"})
#: Most patch rejections an author checkpoint carries forward (newest kept).
MAX_CARRIED_PATCH_REJECTIONS = 8
#: How critic pass 2 tags a SCOPE rejection: the reason begins `SCOPE: ...` or
#: `SCOPE[<rule or route>]: ...`. Structured critics may set `Review.scope_rule`.
_SCOPE_TAG = re.compile(r"^\s*\**\s*SCOPE\s*(?:\[(?P<rule>[^\]]{1,400})\])?\s*\**\s*[:\-—]\s*",
                        re.IGNORECASE)

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

#: Checkpoint stages that restore retained patch BYTES into the lane instead of
#: calling the author. "build": a critic-accepted patch a gate refused (or a stop
#: caught), resumed at the host checks and the CURRENT gates. "critic2": an authored
#: patch whose critic pass 2 never returned a verdict (a critic transient or auth
#: failure, a stop, a lane_error, an author report-path failure over a real diff),
#: resumed AT critic pass 2 -- DS41 runs 10d/10e lost a 79- and a 116-line author
#: patch at that boundary, and every loss re-ran the author (~13-26 min).
PATCH_STAGES = frozenset({"build", "critic2"})


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


@dataclass(frozen=True)
class AuthoringFailure(Abstain):
    """A best-of-N round in which no author produced a diff and at least one failed on
    its own account (`bestof.AuthorPanel._no_diff`). `members` carries one
    `author_failure_record` per author, each classified "authoring" or "harness".
    An all-harness round raises the provider exception instead, with the same records
    on it as `author_failures`."""

    members: tuple = ()


def classify_author_failure(outcome: str, *, failure_class: str | None = None,
                            final_step_capped: bool | None = None,
                            timed_out: bool | None = None) -> dict:
    """Whose failure an author call that produced no checkable diff was.

    "harness"   -- a provider transient, contained error or ended call
                   (`HARNESS_OUTCOMES`); or a `HARNESS_FAILURE_CLASSES` failure class; or
                   a timeout; or a report that names a change the lane does not hold
                   while the call's FINAL step ended on the output cap (the edit was
                   truncated: DS41 run 10h a1-medium, 3 capped steps at 16,384).
    "authoring" -- the author's own answer: an abstention (an explicit reply, whatever
                   the step budget), an empty paths list, or a report of a change it
                   never made on an uncapped final step.
    """
    if outcome in HARNESS_OUTCOMES:
        return {"class": "harness",
                "harness_reason": failure_class or f"author call {outcome}"}
    if outcome != "abstained":
        if failure_class in HARNESS_FAILURE_CLASSES:
            return {"class": "harness", "harness_reason": failure_class}
        if timed_out:
            return {"class": "harness", "harness_reason": "author call timed out"}
        if final_step_capped:
            return {"class": "harness",
                    "harness_reason": "output-capped final step (the edit was truncated)"}
    return {"class": "authoring", "harness_reason": None}


def author_failure_record(*, label: str, outcome: str, reason: str,
                          thinking: str | None = None, evidence: Mapping[str, Any] | None = None,
                          patch: Mapping[str, Any] | None = None,
                          validation: Mapping[str, Any] | None = None,
                          panel_id: str | None = None) -> dict:
    """One author's failure, classified, with the evidence the next attempt needs.

    `evidence` is `actor_metrics.failure_evidence` of the call's metrics row
    (failure_class, final_step_capped, output_capped_steps, output_limit, timed_out,
    steps, decoded_tokens, ak_check). `patch` is the member's retained patch
    ({patch_file, patch_sha256, ...}) when its tree held one."""
    evidence = dict(evidence or {})
    failure_class = evidence.get("failure_class")
    if not failure_class and evidence.get("final_step_capped") and outcome != "diff" \
            and not (patch or {}).get("patch_sha256"):
        failure_class = "output_capped_empty"
    verdict = classify_author_failure(outcome, failure_class=failure_class,
                                      final_step_capped=evidence.get("final_step_capped"),
                                      timed_out=evidence.get("timed_out"))
    record = {"label": label, "thinking": thinking, "outcome": outcome,
              "reason": str(reason or "")[:600], **verdict,
              "failure_class": failure_class}
    for key in ("final_step_capped", "output_capped_steps", "output_limit", "context_limit",
                "timed_out", "steps", "decoded_tokens", "ak_check"):
        if evidence.get(key) is not None:
            record[key] = evidence[key]
    if isinstance(patch, Mapping) and patch.get("patch_sha256"):
        record["patch"] = {key: patch.get(key) for key in ("patch_file", "patch_sha256", "bytes")
                           if patch.get(key) is not None}
    if isinstance(validation, Mapping):
        record["validation"] = {key: validation.get(key)
                                for key in ("passed", "reason", "validator")}
    if panel_id:
        record["panel_id"] = panel_id
    return record


def _ak_check_text(summary: Mapping[str, Any] | None) -> str | None:
    """"ak-check: compile 1/5 pass, op-test 0/2 pass; last op-test 51/130" or None."""
    if not isinstance(summary, Mapping) or not summary.get("calls"):
        return None
    parts = []
    for mode, row in sorted((summary.get("by_mode") or {}).items()):
        if isinstance(row, Mapping):
            parts.append(f"{mode} {row.get('pass', 0)}/{row.get('calls', 0)} pass")
    text = "ak-check: " + (", ".join(parts) or f"{summary.get('calls')} call(s)")
    last = summary.get("last_op_test")
    if isinstance(last, Mapping) and last.get("total"):
        text += (f"; last op-test {str(last.get('status') or '').upper() or '?'} "
                 f"{last.get('passed')}/{last.get('total')}"
                 + (f" on {','.join(last.get('types') or ())}" if last.get("types") else ""))
    elif summary.get("last_status"):
        text += f"; last status {summary['last_status']}"
    return text


def author_failure_feedback(record: Mapping[str, Any]) -> str:
    """The author-facing line for one failure record (carried as
    `prior_patch_rejections`, rendered under "Your patch was rejected")."""
    who = str(record.get("label") or "author")
    if record.get("thinking"):
        who += f", thinking {record['thinking']}"
    harness = record.get("class") == "harness"
    head = (f"authoring harness failure, not charged [{who}] {record.get('outcome')}"
            if harness else f"authoring failed [{who}] {record.get('outcome')}")
    if harness and record.get("harness_reason"):
        head += f" ({record['harness_reason']}"
        if record.get("output_capped_steps"):
            head += (f"; {record['output_capped_steps']} step(s) hit the "
                     f"{record.get('output_limit') or '?'}-token output cap")
        head += ")"
    parts = [f"{head}: {str(record.get('reason') or '').strip()[:500]}"]
    checked = _ak_check_text(record.get("ak_check"))
    if checked:
        parts.append(checked)
    validation = record.get("validation")
    if isinstance(validation, Mapping) and validation.get("passed") is False:
        parts.append(f"validator: {str(validation.get('reason') or '')[:200]}")
    patch = record.get("patch")
    if isinstance(patch, Mapping) and patch.get("patch_file"):
        parts.append(f"its patch is retained at {patch['patch_file']}")
    return " — ".join(parts)


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
    # PATCH_ROUNDS_EXHAUSTED / SCOPE_BLOCKED / HYPOTHESIS_RETIRED: the accepted
    # hypothesis's authoring state (class, attempts used/budget/remaining, and for a
    # scope block the route and rule that blocked it).
    hypothesis_pending: dict | None = None
    # "lane_diff" when the candidate's paths were derived from the lane because the
    # author's reply carried no report; `author_report_recovery` says why and what.
    report_source: str | None = None
    author_report_recovery: dict | None = None
    # Best-of-N authoring (`bestof.AuthorPanel`): one record per panel round that
    # produced this candidate (per author: thinking mode, wall, steps, decoded tokens,
    # validator result, won/lost/cancelled, retained patch). Empty on the single path.
    author_panels: list[dict] = field(default_factory=list)

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
        if self.hypothesis_pending is not None:
            row["hypothesis_pending"] = dict(self.hypothesis_pending)
        if self.report_source is not None:
            row["report_source"] = self.report_source
            row["author_report_recovery"] = self.author_report_recovery
        if self.author_panels:
            row["author_panels"] = [dict(panel) for panel in self.author_panels]
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


def scope_rules_fingerprint() -> str:
    """Digest of what defines the ADMITTED ROUTES a scope rejection was made under.

    The routes are enforced by `gates.affected_op_scope` and described to every actor
    by `program.md`; a `scope_blocked` checkpoint records this and `resume.py` keeps
    it unresumable until it changes. Coarse on purpose, like `gate_rules_fingerprint`:
    any edit to either file re-opens the question, and the resumed hypothesis is
    re-authored and re-reviewed from scratch.
    """
    import hashlib
    digest = hashlib.sha256()
    for path in (Path(gates.__file__), PROGRAM):
        digest.update(path.name.encode() + b"\0")
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()


def classify_patch_rejection(reason: str, *, scope_rule: str = "",
                             source: str = "critic:patch") -> dict:
    """What a patch rejection says about the HYPOTHESIS, never about the idea's merit.

    "scope"     -- the mechanism cannot be written inside the admitted route: the
                   critic set `scope_rule`, or began its reason `SCOPE:` /
                   `SCOPE[<rule>]:`, or a RULE gate (`op_scope`) refused the diff.
    "authoring" -- everything else: the patch was wrong, the idea stands.
    """
    text = str(reason or "")
    rule = str(scope_rule or "").strip()
    if not rule:
        match = _SCOPE_TAG.match(text)
        if match is not None:
            rule = (match.group("rule") or "").strip() \
                or text[match.end():].strip().split(". ")[0][:400] or "unnamed scope rule"
    if not rule and source.startswith("gate:") and source.partition(":")[2] in _RULE_GATES:
        rule = text[:400] or f"{source} refused"
    if rule:
        return {"class": "scope", "rule": rule[:400], "source": source}
    return {"class": "authoring", "rule": None, "source": source}


#: Deterministic pre-build RULE gates (mirrors `resume.RULE_GATES`): a refusal by one
#: is a scope verdict of the rule, not an authoring defect of the patch.
_RULE_GATES = frozenset({"op_scope"})


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


def _lane_changed(author_lane, before_tree: str | None) -> bool:
    """True only when the author lane's full tree provably changed since `before_tree`."""
    if author_lane is None or before_tree is None:
        return False
    try:
        return integrity.candidate_tree(Path(author_lane[0])) != before_tree
    except Exception:      # noqa: BLE001 -- unprovable is "no"
        return False


#: Failure classes of a reply that came back EMPTY because the harness ended the
#: session (`actor_metrics`): worth one immediate retry of critic pass 1.
_EMPTY_REPLY_CLASSES = frozenset({"output_capped_empty", "permission_rejected"})


def _empty_reply(exc: BaseException) -> bool:
    """A transient whose reply was empty (not a provider error that will recur)."""
    return (getattr(exc, "failure_class", None) in _EMPTY_REPLY_CLASSES
            or "no final report" in str(exc))


def _critic1_spent(exc: BaseException, used: int) -> ActorTransient:
    """The transient that drops a planner hypothesis after `CRITIC1_RETRIES`."""
    spent = ActorTransient(f"critic pass 1 failed {used} times for this hypothesis "
                           f"(retry budget {CRITIC1_RETRIES} spent; not resumed): {exc}"[:2000])
    spent.failure_class = getattr(exc, "failure_class", None)
    return spent


def _call_log(author_lane) -> Path | None:
    """The author lane's actor-call metrics log (`actor_metrics`), or None."""
    if author_lane is None:
        return None
    from . import actor_metrics
    return Path(author_lane[0]).parent / actor_metrics.REPLY_DIR_NAME / actor_metrics.CALL_LOG_NAME


def _call_log_offset(author_lane) -> int:
    path = _call_log(author_lane)
    try:
        return path.stat().st_size if path is not None else 0
    except OSError:
        return 0


def _last_call_evidence(author_lane, offset: int) -> dict:
    """`actor_metrics.failure_evidence` of the last author metrics row this call wrote
    to the lane's log (past `offset`); {} when there is none. Evidence only."""
    path = _call_log(author_lane)
    if path is None:
        return {}
    from . import actor_metrics
    try:
        rows = actor_metrics.metric_rows_since(path, offset, role="author")
    except Exception:      # noqa: BLE001 -- evidence, never control
        return {}
    return actor_metrics.failure_evidence(rows[-1]) if rows else {}


def _failure_members(exc: BaseException, outcome: str, author_lane,
                     offset: int) -> list[dict]:
    """The failure records an author exception carries (a best-of-N panel attaches one
    per member as `author_failures`), else one record for the single author."""
    carried = getattr(exc, "author_failures", None)
    if carried:
        return [dict(item) for item in carried]
    evidence = _last_call_evidence(author_lane, offset)
    if getattr(exc, "failure_class", None) and not evidence.get("failure_class"):
        evidence["failure_class"] = getattr(exc, "failure_class")
    return [author_failure_record(label="author", outcome=outcome, reason=str(exc),
                                  evidence=evidence)]


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
    if getattr(missing, "author_failures", None):
        extended.author_failures = missing.author_failures     # a panel's member records
    return extended


def _panel_recorder(progress: dict[str, Any]):
    """Collect one `bestof.AuthorPanel` round's record: on the iteration's outcome
    (all rounds) and on any candidate this patch round disposes of (its own round)."""
    def record(panel: Mapping[str, Any]) -> None:
        progress["author_panel"] = dict(panel)
        progress.setdefault("author_panels", []).append(dict(panel))
    return record


def _pending_checkpoints(progress: Mapping[str, Any]) -> list[dict]:
    pending: list[dict] = []
    for key in ("inflight", "critic2", "build"):
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
            author_attempts: int = HYPOTHESIS_AUTHOR_ATTEMPTS,
            record_abandoned: Callable[[Outcome], None] | None = None,
            resume=None,
            next_resume: Callable[[], Any] | None = None,
            author_lane: tuple[Path, str] | None = None,
            iteration_scope=None,
            author_panel=None
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

    An ACCEPTED hypothesis whose patch rounds all end in rejection is not thrown back
    to the planner. The iteration ends as `patch_rounds_exhausted` (the last rejection
    was the author's) or `scope_blocked` (the last rejection said the mechanism cannot
    be written inside the admitted route), carrying an AUTHOR checkpoint with every
    patch rejection as author feedback, so the next draw re-authors it before the
    planner is asked (`resume.py`). `author_attempts` bounds that per hypothesis
    across iterations: each authoring attempt of `patch_rounds` rounds that ends on an
    author failure spends one, and the last one retires it (`hypothesis_retired`). A
    scope-blocked attempt spends none: it waits for the scope rules to change.

    An author call that produces NO checkable diff for an accepted hypothesis (it
    abstains, returns no path, reports a change the lane does not hold, or fails as a
    provider transient; for best-of-N, every member) is an AUTHORING failure, never a
    verdict on the idea: the iteration ends `authoring_failed` (one attempt spent,
    however many panel members failed) or `authoring_harness_failure` (every failure
    was the harness's -- a truncated final step, a timeout, a provider error -- none
    spent until `AUTHOR_HARNESS_FAILURE_CAP` in a row), with an author checkpoint whose
    feedback names each member's reason, ak-check result and retained patch. The budget
    spent retires it. Only the PLANNER's abstention (`propose`) ends as `abstained`.

    `resume` (a `resume.ResumePoint`) seeds ONE extra round, before the fresh
    hypothesis rounds, with work a previous launch already paid for: at stage
    "build" an accepted patch goes straight to the host checks, the CURRENT gates
    and the measurement (no planner, critic or author call); at stage "critic2" an
    authored patch whose critic pass 2 never answered is restored and reviewed by
    critic pass 2 for real, then gated and measured (no planner or author call; a
    rejection hands the author the reason if patch rounds remain); at stage "author"
    an accepted hypothesis resumes authoring with its original verdict and rejection
    history; at stage "critic1" a planner hypothesis whose critic pass 1 never answered
    is reviewed by critic pass 1 for real (no planner call). A fresh hypothesis is
    checkpointed at "critic1" before critic pass 1: an empty critic reply is retried once
    in the iteration, and a critic transient ends it `planner_transient` carrying that
    checkpoint, at most `CRITIC1_RETRIES` times per hypothesis. Every re-validation failure is disposed as `resume_rejected` and the
    iteration continues with the next queued checkpoint (`next_resume()`, drawn before
    any fresh round; each checkpoint at most once per iteration), else fresh work.
    A stop, provider transient or gate refusal leaves `resume_checkpoints` on its row
    so the next launch can resume it; an
    authored patch still awaiting critic pass 2 adds a "critic2" checkpoint (also
    handed to the pool on an escaping exception, for its `lane_error` row).

    `author_lane` is `(worktree, base)`: the lane the author edits and the commit the
    OWNER reset it to for this draw (`pipeline.run_pool`). Only with it, an author
    reply that carries no report (`AuthorReportMissing`) is answered from the lane diff
    (`integrity.lane_diff_report`, recorded `report_source: "lane_diff"`), and the
    normal gates judge that diff. Without it, or with no diff, the transient stands.

    `iteration_scope` (a `scratch.Scope`, opened by `pipeline.run_pool` on the lane's
    thread) is where THIS iteration's scratch is allocated: a best-of author worktree
    is `iteration_scope.worktree(repo, base, name)`, an ak-check build dir is
    `iteration_scope.dir("ak-check-build", name)`, and a subprocess gets a scoped TMPDIR
    from `iteration_scope.tmp_env()`. Check `registry.ensure_free(bytes)` first and
    degrade on False (best-of N->1, op-test->compile-only). Everything allocated here
    is released when the iteration ends; nothing else may create scratch
    (`scratch.py`, enforced by `test_scratch.py`). Injected callables running inside
    this call find the same scope with `scratch.current("iteration")`. None (a direct
    caller) falls back to the current thread's scope, else the registry's standing one.

    `author_panel` (`bestof.AuthorPanel`, best-of-N) replaces the one author call of
    each patch round: N authors race in scratch worktrees and the selected diff lands
    on `author_lane`, so everything after the call is the single path's. None (N=1)
    is the single-author path, byte for byte.
    """
    if iteration_scope is None:
        from . import scratch
        iteration_scope = scratch.current("iteration") or scratch.active_scope()
    working = dict(context)
    abandoned: list[dict] = []
    progress: dict[str, Any] = {"inflight": None, "critic2": None, "build": None,
                                "report_recovery": None, "resumed_active": False,
                                "author_panels": [], "author_panel": None,
                                # The resume in flight: `next_resume` replaces a refused one.
                                "resume": resume}
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
        if progress["author_panels"] and not outcome.author_panels:
            outcome.author_panels = [dict(panel) for panel in progress["author_panels"]]
        resumed = progress.get("resume")
        if resumed is not None and (
                (outcome.hypothesis is not None and outcome.hypothesis is resumed.hypothesis)
                or (outcome.hypothesis is None and progress.get("resumed_active"))):
            # The second clause: a provider transient (the critic's auth failure in
            # DS41 run 10e) ends the iteration with no hypothesis on the outcome while
            # the RESUMED candidate was in flight. Unattributed, the row named no
            # lineage and the claim was never settled (2738e95f...#0 stayed open).
            outcome.resumed_from = resumed.checkpoint_id
            outcome.resume_stage = resumed.stage
        if outcome.status in RESUMABLE_STATUSES and not outcome.resume_checkpoints:
            outcome.resume_checkpoints = _pending_checkpoints(progress)
        _mark_report_source(outcome, progress)
        return outcome

    try:
        return observed(_iterate(planner=planner, critic=critic, working=working,
                        hypothesis_reasons=hypothesis_reasons, measure=measure,
                        gate=gate, commit=commit,
                        hypothesis_rounds=hypothesis_rounds,
                        author_attempts=author_attempts,
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
                        resume=resume, next_resume=next_resume, progress=progress,
                        author_lane=author_lane,
                        author_panel=author_panel))
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
        # A critic-pass-1 transient carries the planner's hypothesis (and its row
        # the critic1 checkpoint): the row names what was in flight.
        return observed(Outcome("planner_transient", getattr(exc, "hypothesis", None),
                                [str(exc)]))
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
    except Exception as exc:
        # Contained by the pool as `lane_error`, which carries no hypothesis and
        # never read `progress`. An authored patch whose critic pass 2 had not
        # returned is the one piece of in-flight work worth carrying across that
        # containment: hand its checkpoint to the pool on the exception itself.
        pending = [dict(progress["critic2"])] if progress.get("critic2") else []
        if pending:
            try:
                exc.resume_checkpoints = pending
            except Exception:      # noqa: BLE001 -- an exception type without __dict__
                pass
        raise


def _iterate(*, planner, critic, working, hypothesis_reasons, measure, gate, commit,
             hypothesis_rounds, patch_rounds, on_step=lambda _label: None,
             tail_session=nullcontext,
             should_abandon=lambda: False, record_reschedule=None,
             accumulate_valid_positive=False,
             validate_candidate=lambda _hypothesis, _paths: None,
             formation_guard=lambda _hypothesis, _context: None,
             reserve_candidate=None, round_telemetry=None,
             author_attempts=HYPOTHESIS_AUTHOR_ATTEMPTS,
             validator_provenance=None, record_abandoned=None,
             abandoned=None, resume=None, next_resume=None, progress=None,
             author_lane=None, author_panel=None) -> Outcome:
    last_proposed: Hypothesis | None = None
    round_telemetry = round_telemetry if round_telemetry is not None else {}
    validator_provenance = (validator_provenance if validator_provenance is not None
                            else [])

    abandoned = abandoned if abandoned is not None else []
    progress = progress if progress is not None else {"inflight": None, "critic2": None,
                                                      "build": None}

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

    def attempts_used(hypothesis) -> int:
        """Authoring attempts this accepted hypothesis already spent (carried)."""
        if not is_resumed(hypothesis):
            return 0
        try:
            return max(0, int(resume.checkpoint.get("author_attempts_used") or 0))
        except (TypeError, ValueError):
            return 0

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
                "author_attempts_used": attempts_used(hypothesis),
                # A pending hypothesis interrupted mid-attempt keeps its budget (and so
                # its chain-depth allowance, `resume.depth_limit`) on the new checkpoint.
                **({"author_attempts_budget": resume.checkpoint["author_attempts_budget"]}
                   if is_resumed(hypothesis)
                   and resume.checkpoint.get("author_attempts_budget") is not None else {}),
                # So is its harness-failure streak (a stop is neither a success nor a
                # harness failure of the author call).
                **({"author_harness_failures": harness_streak(hypothesis)}
                   if harness_streak(hypothesis) else {}),
                **lineage(hypothesis), **fields}

    def exhausted(hypothesis, patch_reasons: list[str],
                  rejections: list[dict]) -> Outcome:
        """Every patch round of an ACCEPTED hypothesis ended in a rejection.

        The idea stands (critic pass 1 accepted it and nothing since judged it), so
        it stays pending at the author instead of going back to the planner.
        """
        last = rejections[-1]
        scope = last["class"] == "scope"
        budget = max(1, int(author_attempts))
        used = attempts_used(hypothesis) + (0 if scope else 1)
        carried = list(patch_reasons)[-MAX_CARRIED_PATCH_REJECTIONS:]
        state = {"class": SCOPE_BLOCKED if scope else PATCH_ROUNDS_EXHAUSTED,
                 "author_attempts_used": used, "author_attempts_budget": budget,
                 "author_attempts_remaining": max(0, budget - used),
                 "patch_rounds_per_attempt": int(patch_rounds),
                 "last_rejection": {key: last[key] for key in ("class", "source", "rule")}}
        rounds = f"{len(rejections)} patch round(s)"
        if not scope and used >= budget:
            state["class"] = HYPOTHESIS_RETIRED
            return Outcome(HYPOTHESIS_RETIRED, hypothesis, [
                f"retired: the accepted hypothesis spent its authoring budget "
                f"({used}/{budget} attempts of {int(patch_rounds)} patch rounds) on "
                f"author failures; the critic's acceptance of the idea was never "
                f"withdrawn", *carried], refusal_gate="author_attempts",
                hypothesis_pending=state)
        fields = {"prior_patch_rejections": carried,
                  "patch_rounds_remaining": int(patch_rounds),
                  "author_attempts_used": used, "author_attempts_budget": budget,
                  # Every round produced a diff: the harness streak is broken.
                  "author_harness_failures": 0}
        checkpoints = None
        if scope:
            state["scope_block"] = fields["scope_block"] = {
                "route": f"{hypothesis.target_surface}::{hypothesis.target_symbol}",
                "rule": last["rule"], "source": last["source"],
                "scope_rules_fingerprint": scope_rules_fingerprint()}
            summary = (f"scope_blocked: the accepted hypothesis cannot be written inside "
                       f"the admitted route ({last['source']}: {last['rule']}); kept "
                       f"pending until the scope rules change")
            if last["source"].startswith("gate:"):
                # A RULE gate refused a critic-ACCEPTED patch: its `gate_refused` row
                # already carries a build checkpoint that resumes the exact patch once
                # that rule changes (and outranks, then supersedes, any author sibling).
                # An author checkpoint here would be dead weight.
                checkpoints = []
                state["resumable_via"] = "gate_refused build checkpoint"
                summary += "; resumable through the gate_refused row's build checkpoint"
        else:
            summary = (f"patch_rounds_exhausted: {rounds} ended on author failures while "
                       f"the hypothesis stood accepted; pending re-authoring with that "
                       f"feedback ({budget - used} of {budget} attempts left)")
        return Outcome(state["class"], hypothesis, [summary, *carried],
                       refusal_gate=last["source"],
                       resume_checkpoints=(checkpoints if checkpoints is not None
                                           else [checkpoint("author", hypothesis, **fields)]),
                       hypothesis_pending=state)

    def critic1_used(hypothesis) -> int:
        """Critic-pass-1 transients this hypothesis already spent (a critic1 resume)."""
        if not is_resumed(hypothesis) or resume.stage != "critic1":
            return 0
        try:
            return max(0, int(resume.checkpoint.get("critic1_attempts_used") or 0))
        except (TypeError, ValueError):
            return 0

    def critic1_checkpoint(hypothesis, used: int, **extra) -> dict:
        """A planner hypothesis whose critic pass 1 has not answered: resumed AT critic
        pass 1 (no planner call), with every patch round still ahead of it."""
        return checkpoint("critic1", hypothesis, critic_hypothesis=None, patch_round=0,
                          critic1_attempts_used=int(used),
                          critic1_attempts_budget=CRITIC1_RETRIES,
                          patch_rounds_remaining=int(patch_rounds), **extra)

    def harness_streak(hypothesis) -> int:
        """Consecutive harness failures this accepted hypothesis carries (resumed)."""
        if not is_resumed(hypothesis):
            return 0
        try:
            return max(0, int(resume.checkpoint.get("author_harness_failures") or 0))
        except (TypeError, ValueError):
            return 0

    def authoring_failed(hypothesis, patch_reasons: list[str],
                         members: Sequence[Mapping[str, Any]]) -> Outcome:
        """The author produced no checkable diff for an ACCEPTED hypothesis.

        Operator rule 2026-09-26: only the AUTHORING failed, so the hypothesis stays
        pending at the author. A round with any genuine authoring failure spends ONE
        attempt (a best-of-N panel charges at most one, however many members failed);
        an all-harness round spends none until `AUTHOR_HARNESS_FAILURE_CAP` in a row.
        The budget spent retires it, exactly as `exhausted` does.
        """
        members = [dict(member) for member in members] or [author_failure_record(
            label="author", outcome="error", reason="author produced no diff")]
        harness = all(member.get("class") == "harness" for member in members)
        budget = max(1, int(author_attempts))
        streak = harness_streak(hypothesis) + 1 if harness else 0
        charged = not harness or streak >= AUTHOR_HARNESS_FAILURE_CAP
        used = attempts_used(hypothesis) + (1 if charged else 0)
        feedback = [author_failure_feedback(member) for member in members]
        if harness and charged:
            feedback.append(f"{streak} consecutive authoring harness failures "
                            f"(cap {AUTHOR_HARNESS_FAILURE_CAP}): this one is charged")
        carried = [*patch_reasons, *feedback][-MAX_CARRIED_PATCH_REJECTIONS:]
        status = AUTHORING_HARNESS_FAILURE if not charged else AUTHORING_FAILED
        state = {"class": status, "author_attempts_used": used,
                 "author_attempts_budget": budget,
                 "author_attempts_remaining": max(0, budget - used),
                 "patch_rounds_per_attempt": int(patch_rounds), "charged": charged,
                 "author_harness_failures": 0 if charged else streak,
                 "author_harness_failure_cap": AUTHOR_HARNESS_FAILURE_CAP,
                 "authoring_failures": members,
                 "last_rejection": {"class": "harness" if harness else "authoring",
                                    "source": "author", "rule": None}}
        who = ", ".join(f"{m.get('label')}: {m.get('outcome')} ({m.get('class')})"
                        for m in members)
        if charged and used >= budget:
            state["class"] = HYPOTHESIS_RETIRED
            return Outcome(HYPOTHESIS_RETIRED, hypothesis, [
                f"retired: the accepted hypothesis spent its authoring budget "
                f"({used}/{budget} attempts); the last attempt produced no diff ({who}); "
                f"the critic's acceptance of the idea was never withdrawn", *carried],
                refusal_gate="author_attempts", hypothesis_pending=state)
        if charged:
            summary = (f"authoring_failed: the author produced no diff for the accepted "
                       f"hypothesis ({who}); pending re-authoring with that feedback "
                       f"({budget - used} of {budget} attempts left)")
        else:
            summary = (f"authoring_harness_failure: no diff, and every failure was the "
                       f"harness's ({who}); no attempt charged ({streak}/"
                       f"{AUTHOR_HARNESS_FAILURE_CAP} consecutive), pending re-authoring")
        fields = {"prior_patch_rejections": carried,
                  "patch_rounds_remaining": int(patch_rounds),
                  "author_attempts_used": used, "author_attempts_budget": budget,
                  "author_harness_failures": state["author_harness_failures"],
                  "authoring_failures": members}
        return Outcome(status, hypothesis, [summary, *carried], refusal_gate="author",
                       resume_checkpoints=[checkpoint("author", hypothesis, **fields)],
                       hypothesis_pending=state)

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
        if progress.get("author_panel") is not None:
            candidate.author_panels = [dict(progress["author_panel"])]
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
    tried = {resume.checkpoint_id} if resume is not None else set()

    def draw_next_resume(index: int) -> None:
        """A refused resume yields to the NEXT queued checkpoint, not the planner.

        DS41 run 10i batch 0 queued two checkpoints; the first was refused at
        re-validation and the iteration fell through to a planner that abstained,
        wasting a one-iteration batch while an accepted hypothesis sat queued. Also an
        EXTRA round (no fresh round is replaced), and bounded: a checkpoint already
        tried in this iteration ends the draw.
        """
        nonlocal resume
        point = next_resume() if next_resume is not None else None
        if point is None or point.checkpoint_id in tried:
            return
        tried.add(point.checkpoint_id)
        # Rebound so lineage, carried budgets and the final outcome name THIS point.
        resume = progress["resume"] = point
        schedule.insert(index + 1, point)

    hypothesis_index = -1
    while hypothesis_index + 1 < len(schedule):
        hypothesis_index += 1
        resumed = schedule[hypothesis_index]
        # Polled BEFORE each actor call, never after: the whole point is that no
        # further multi-minute call is drawn once the run has been told to stop.
        progress["critic2"] = None
        progress["resumed_active"] = False
        progress["author_panel"] = None
        if resumed is not None and getattr(resumed, "stale", None) is None:
            # Named before the poll, so a stop here still carries the claimed
            # checkpoint forward instead of consuming it.
            last_proposed = resumed.hypothesis
            progress["resumed_active"] = True
            carried = {**{key: value for key, value in resumed.checkpoint.items()
                          if key not in _OWNER_BOUND}, **lineage(resumed.hypothesis)}
            if resumed.stage == "critic2":
                # Its retained pointer is carried as-is: the bytes are the same ones.
                progress["inflight"], progress["critic2"] = None, carried
            else:
                progress["inflight"] = carried
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
                draw_next_resume(hypothesis_index)
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
                draw_next_resume(hypothesis_index)
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
            if hypothesis.runtime_pair is None:
                # Formed and not yet judged: a stop before critic pass 1 answers keeps it.
                progress["inflight"] = critic1_checkpoint(hypothesis, 0)

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
        if resumed is not None and resumed.stage != "critic1":
            pass    # admitted by the carried verdict above
        elif prevalidated_runtime:
            on_step("prevalidated runtime option: deterministic checks, no critic call")
        else:
            # The planner's hypothesis is PAID FOR (DS41 run 10h batch 1: 31 minutes of
            # planning) before critic pass 1 answers: checkpointed first, so a critic
            # transient resumes it at critic pass 1 instead of discarding it.
            critic1 = hypothesis.runtime_pair is None
            if critic1:
                progress["inflight"] = critic1_checkpoint(hypothesis, critic1_used(hypothesis))
            on_step("critic pass 1: reviewing the hypothesis"
                    + (f" ({resumed.label})" if resumed is not None else ""))
            retried = False
            while True:
                try:
                    verdict, halted = actor_call(critic.review_hypothesis, hypothesis, working)
                    break
                except ActorStopped:
                    raise
                except ActorTransient as exc:
                    if not retried and _empty_reply(exc):
                        # One immediate retry for an EMPTY reply (a session a
                        # permission rejection or the output cap ended silently).
                        retried = True
                        on_step("critic pass 1: empty reply, retrying once")
                        if should_abandon():
                            return stopped()
                        continue
                    if critic1:
                        used = critic1_used(hypothesis) + 1
                        if used >= CRITIC1_RETRIES:
                            progress["inflight"] = None
                            spent = _critic1_spent(exc, used)
                            spent.hypothesis = hypothesis
                            raise spent from exc
                        progress["inflight"] = critic1_checkpoint(hypothesis, used,
                                                                  last_transient=str(exc))
                    try:
                        exc.hypothesis = hypothesis     # the row names what was in flight
                    except Exception:      # noqa: BLE001 -- an exception without __dict__
                        pass
                    raise
            progress["inflight"] = None
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
        #: The restored bytes are still the lane's only change (no author call since).
        restored_untouched = False
        patch_reasons: list[str] = (list(resumed.prior_patch_rejections)
                                    if resumed is not None else [])
        round_count = (1 if hypothesis.runtime_pair is not None
                       else max(1, int(resumed.patch_rounds)) if resumed is not None
                       else patch_rounds)
        #: One classification per round that ended in a patch rejection (critic pass
        #: 2 or a pre-build gate), and whether a RESUME was refused instead: only the
        #: former leaves an accepted hypothesis pending (`exhausted`).
        rejections: list[dict] = []
        resume_refused = False
        for patch_index in range(round_count):
            # Round 1 of a build or critic2 resume restores retained bytes instead of
            # authoring; a critic2 resume's later rounds author normally.
            restoring = (resumed is not None and resumed.stage in PATCH_STAGES
                         and patch_index == 0)
            if hypothesis.runtime_pair is None and not restoring:
                # The in-flight checkpoint a stop or provider transient leaves behind:
                # an accepted hypothesis, its verdict and every patch rejection so far.
                progress["inflight"] = checkpoint(
                    "author", hypothesis, prior_patch_rejections=list(patch_reasons),
                    patch_rounds_remaining=round_count - patch_index)
                progress["critic2"] = None
            if should_abandon():
                return stopped()
            working["prior_patch_rejections"] = list(patch_reasons)
            round_telemetry["patch_round"] = patch_index + 1
            if patch_reasons:
                round_telemetry["prior_rejection_prompt"] = True
                mark_search_changed("critic:patch")
            paths = ()
            integrity_screen = None
            progress["author_panel"] = None
            if hypothesis.runtime_pair is None:
                if restoring:
                    # No author call: restore the exact retained bytes, re-verified
                    # (digest, anchor, clean apply) by the owner at this moment.
                    on_step("restoring the retained patch (no author call)")
                    try:
                        paths = tuple(resumed.materialize())
                        materialized = restored_untouched = True
                    except Exception as exc:      # noqa: BLE001 -- a stale resume
                        dispose(hypothesis, RESUME_REJECTED,
                                f"resume re-validation refused: {exc}",
                                refusal_gate=f"resume:{getattr(exc, 'check', 'materialize')}")
                        resume_refused = True
                        break
                else:
                    on_step("authoring the patch")
                    progress["report_recovery"] = None
                    restored_untouched = False
                    before_tree = None
                    if author_lane is not None:
                        try:
                            before_tree = integrity.candidate_tree(Path(author_lane[0]))
                        except Exception:      # noqa: BLE001 -- recovery then refuses
                            before_tree = None

                    def authored_checkpoint() -> dict:
                        # The author produced a diff and critic pass 2 has not answered.
                        # The OWNER retains the lane diff when it records the row
                        # (`resume.retain_checkpoint_patches`) and drops this entry if
                        # the lane holds none. Rounds remaining count THIS round.
                        return checkpoint("critic2", hypothesis,
                                          prior_patch_rejections=list(patch_reasons),
                                          patch_rounds_remaining=round_count - patch_index,
                                          retained_patch=None)
                    author = planner.author
                    if author_panel is not None:
                        def author(h, ctx, _lane=author_lane, _solo=planner.author):
                            # Best-of-N: the selected diff lands on the lane, so the
                            # recovery below and every later stage read it as usual.
                            return author_panel(h, ctx, lane=_lane, solo=_solo,
                                                record=_panel_recorder(progress))
                    log_offset = _call_log_offset(author_lane)
                    try:
                        paths, halted = actor_call(author, hypothesis, working)
                    except AuthorReportMissing as missing:
                        try:
                            paths, halted = _recover_author_report(
                                missing, author_lane, hypothesis, before_tree, progress), None
                        except AuthorReportMissing as unrecovered:
                            # DS41 run 10d: the author edited the lane and its report
                            # was unusable. The edit may still be a patch worth a
                            # verdict -- but only an edit made BY THIS CALL: a lane
                            # still holding an earlier round's rejected patch is not.
                            if _lane_changed(author_lane, before_tree):
                                progress["critic2"] = authored_checkpoint()
                                raise
                            # No diff by this call: an AUTHORING failure of an accepted
                            # hypothesis (truncation-caused = the harness's).
                            return authoring_failed(hypothesis, patch_reasons, _failure_members(
                                unrecovered, "report_missing", author_lane, log_offset))
                        on_step("authoring report derived from the lane diff")
                    except ActorStopped:
                        raise
                    except ActorTransient as transient:
                        # Provider transient / timeout / per-call budget of the AUTHOR
                        # (every panel member's, for best-of-N): the harness failed.
                        return authoring_failed(hypothesis, patch_reasons, _failure_members(
                            transient, "transient", author_lane, log_offset))
                    else:
                        _note_author_report(paths, progress)
                    if halted is not None:
                        return halted
                    if isinstance(paths, Abstain):
                        # The AUTHOR abstained (or returned no changed path): only the
                        # authoring failed; the critic-accepted idea stays pending.
                        members = getattr(paths, "members", None) or [author_failure_record(
                            label="author", outcome="abstained", reason=paths.reason,
                            evidence=_last_call_evidence(author_lane, log_offset))]
                        return authoring_failed(hypothesis, patch_reasons, members)
                    progress["critic2"] = authored_checkpoint()
                # A declared path list is a claim, not an isolation boundary.  The
                # injected host check resolves the full worktree before review/build.
                try:
                    checked = validate_candidate(hypothesis, paths)
                    integrity_screen = (checked.to_dict() if hasattr(checked, "to_dict")
                                        else checked)
                except integrity.IntegrityRefused as exc:
                    if restoring:
                        progress["critic2"] = None
                        dispose(hypothesis, RESUME_REJECTED,
                                f"resume re-validation refused (integrity): {exc}",
                                refusal_gate="resume:integrity")
                        resume_refused = True
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
                # A verdict came back: the critic2 checkpoint is answered either way.
                progress["critic2"] = None
                validator_provenance.append(_critic_provenance(
                    critic, patch_verdict, decision="critic:patch",
                    evidence=("candidate diff", "declared paths", "planner context")))
                if not patch_verdict.accepted:
                    # The hypothesis is untouched: a bad patch is not evidence against
                    # the idea it was trying to implement.
                    patch_reasons.append(patch_verdict.reason)
                    rejections.append(classify_patch_rejection(
                        patch_verdict.reason,
                        scope_rule=getattr(patch_verdict, "scope_rule", "") or ""))
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
                            resume_refused = True
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
                        rejections.append(classify_patch_rejection(
                            patch_reasons[-1], source=f"gate:{refusing_gate}"))
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

        # Patch budget spent.
        progress["inflight"] = progress["critic2"] = None
        progress["resumed_active"] = False
        if materialized and restored_untouched:
            # A rejected resume must not leave its bytes under the next fresh round.
            # Once an author has edited on top of them (a critic2 resume whose restored
            # patch was rejected with rounds left), the lane is the author's, exactly
            # as after any fresh patch round.
            resumed.discard()
        if rejections and not resume_refused and hypothesis.runtime_pair is None:
            # The hypothesis is still ACCEPTED: every round was a verdict on a patch.
            # It stays pending at the author (or is retired on its attempt budget)
            # rather than being dropped for the planner's next proposal.
            return exhausted(hypothesis, patch_reasons, rejections)
        # A refused RESUME (stale bytes, moved anchor, a current gate): control
        # returns to the HYPOTHESIS loop, which takes the next queued checkpoint
        # first and only then draws fresh work.
        if resume_refused:
            draw_next_resume(hypothesis_index)
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

__all__ = ["AUTHORING_FAILED", "AUTHORING_HARNESS_FAILURE", "AUTHOR_HARNESS_FAILURE_CAP",
           "CRITIC1_RETRIES",
           "AuthoringFailure", "HARNESS_FAILURE_CLASSES", "HARNESS_OUTCOMES",
           "author_failure_feedback", "author_failure_record", "classify_author_failure",
           "CANDIDATE_DISPOSITIONS", "CHECKPOINT_SCHEMA", "PATCH_STAGES", "RESUMABLE_STATUSES",
           "HYPOTHESIS_AUTHOR_ATTEMPTS", "HYPOTHESIS_RETIRED", "PATCH_ROUNDS_EXHAUSTED",
           "PENDING_HYPOTHESIS_STATUSES", "SCOPE_BLOCKED", "classify_patch_rejection",
           "scope_rules_fingerprint",
           "RESUME_REJECTED", "STOPPED_AFTER_DISPOSALS", "gate_rules_fingerprint", "Abstain", "ActorStopped", "ActorTransient", "AuthorReportMissing", "REPORT_SOURCE_LANE_DIFF", "ConfirmVetoed", "InteractionRegression", "TailRefused", "RunAborted", "MeasurementInvalid", "MeasurementFailed", "Critic",
           "HYPOTHESIS_ROUNDS", "Hypothesis", "Outcome", "PATCH_ROUNDS",
           "Planner", "Review", "STOPPED_MID_FORMATION", "iterate"]
