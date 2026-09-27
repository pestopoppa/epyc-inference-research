"""ProactiveDelegator — orchestrates proactive delegation workflow.

Delegation modes (RD-10a)
-------------------------
Two delegation shapes are named explicitly (``DelegationMode``):

* **AS-TOOL** — a specialist *subtask*. The manager (this delegator) KEEPS
  control: the specialist's raw output flows through the ``output_extractor``
  seam and the delegator retains ownership of the iteration loop.
* **HANDOFF** — an ESCALATE decision. OWNERSHIP TRANSFERS to a higher-tier role,
  which receives an *input-filtered* slice of the feedback history (the
  ``input_filter``; its window is ``ReviewPlaneKnobs.input_filter_window``,
  formerly the hard-coded ``feedback_history[-3:]``).

Plan-reminder wire-point contract (RD-9)
----------------------------------------
The delegator exposes a call-site for a plan-reminder helper that the
``review_service`` agent owns. Contract:

* ``review_service`` MAY expose
  ``build_plan_reminder(objective: str, plan_steps: list, step_index: int) -> str | None``.
* The delegator calls it from the ``delegate()`` step loop every
  ``ReviewPlaneKnobs.reminder_cadence`` steps (default ``0`` → disabled). The
  returned reminder is prepended to subsequent specialist prompts — PREFERRED
  over re-review (intake-835).
* The call is resolved via ``getattr`` and wrapped so the helper's ABSENCE (the
  other agent has not landed it yet) or any error is a no-op. Neither agent
  blocks the other. Shadow-only; never blocks request completion.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, TYPE_CHECKING

from src.roles import Role, chain_name_to_role
from src.proactive_delegation.types import (
    AggregatedResult,
    ComplexitySignals,
    IterationContext,
    ReviewDecision,
    StepExecutionError,
    SubtaskResult,
    TaskComplexity,
)
from src.proactive_delegation.complexity import (
    ROLE_MAPPING,
    classify_task_complexity,
)
from src.proactive_delegation.review_service import (
    AggregationService,
    ArchitectReviewService,
)
from src.task_ir import canonicalize_task_ir

if TYPE_CHECKING:
    from src.registry_loader import RegistryLoader
    from src.llm_primitives import LLMPrimitives
    from orchestration.repl_memory.progress_logger import ProgressLogger

logger = logging.getLogger(__name__)

_MISSING = object()

# Ordinal rank for the complexity-gated review threshold (RD-10b). Reuses the
# existing TaskComplexity enum — a subtask is reviewed per-subtask only when its
# rank is >= review_trigger_complexity_threshold.
_COMPLEXITY_RANK: dict[TaskComplexity, int] = {
    TaskComplexity.TRIVIAL: 0,
    TaskComplexity.SIMPLE: 1,
    TaskComplexity.MODERATE: 2,
    TaskComplexity.COMPLEX: 3,
}


class DelegationMode(Enum):
    """The two delegation modes, named explicitly (RD-10a).

    See the module docstring for the full contract. ``AS_TOOL`` keeps control in
    the manager; ``HANDOFF`` transfers ownership on ESCALATE.
    """

    AS_TOOL = "as_tool"
    HANDOFF = "handoff"


def _identity_output_extractor(raw: str) -> str:
    """Default AS-TOOL output extractor: passthrough (behavior-preserving).

    RD-10a formalizes the specialist(as-tool)-output → manager boundary as a
    seam. The default is the identity function so the manager keeps exactly the
    bytes it does today; a caller may inject truncation/parse/normalization here.
    """
    return raw


@dataclass
class ReviewPlaneKnobs:
    """Control-plane tuning knobs (RD-11 / LB-5) consumed by the delegator loop.

    EVERY default preserves current behavior byte-for-byte:

    * ``per_subtask_review_enabled=True`` + ``review_trigger_complexity_threshold=0``
      → every subtask is reviewed (current behavior); the single final-aggregate
      review path in ``delegate()`` stays dormant.
    * ``input_filter_window=3`` → the HANDOFF input_filter reproduces
      ``feedback_history[-3:]`` exactly.
    * ``decision_cache_enabled=False`` → RD-10c sticky cache is off.
    * ``reminder_cadence=0`` → the RD-9 plan-reminder call-site never fires.
    * ``budget_shadow_logging_enabled=False`` → the LB-5 shadow budget hook is
      never invoked from the default path.

    The per-decision token/latency budgets are the LB-2 targets (PROPOSED /
    observation-grade). The authoritative declaration — bounds, dtype,
    restart_cost, provenance — lives in the guarded numeric-surface manifest at
    ``orchestration/review_plane_knobs.yaml`` (what the autopilot AP-1 consumes).
    """

    # RD-10b — complexity-gated per-subtask review placement
    per_subtask_review_enabled: bool = True
    review_trigger_complexity_threshold: int = 0  # 0=TRIVIAL → review all (inert)
    # RD-10a — HANDOFF input_filter window (was hard-coded [-3:])
    input_filter_window: int = 3
    # RD-10c — sticky decision cache
    decision_cache_enabled: bool = False
    # RD-9 — plan-reminder cadence (steps); 0 = disabled
    reminder_cadence: int = 0
    # RD-2 — near-band majority-of-k grading (declared; enforced in review_service)
    review_majority_k: int = 1
    # RA-6 — request_evidence follow-up rounds; 0 = disabled
    request_evidence_round_budget: int = 0
    # LB-2 — governed token multiplier vs single-model baseline (hard ceiling 2.0)
    review_token_multiplier: float = 2.0
    # LB-5 — shadow budget-violation logging (never blocks)
    budget_shadow_logging_enabled: bool = False
    # LB-2 per-decision budgets (ceilings; used only by the shadow budget hook)
    plan_review_token_budget: int = 350
    plan_review_latency_budget_ms: int = 22000
    candidate_review_token_budget: int = 300
    candidate_review_latency_budget_ms: int = 18000
    rubric_authoring_token_budget: int = 800
    rubric_authoring_latency_budget_ms: int = 45000
    rubric_grading_token_budget: int = 180
    rubric_grading_latency_budget_ms: int = 5000

    @classmethod
    def from_config(cls) -> "ReviewPlaneKnobs":
        """Merge point for the autopilot (AP-1).

        Reads per-field overrides from ``get_config().delegation`` when present.
        Until the autopilot lands those fields on the delegation config (or a
        dedicated ``review_plane`` section), every field is absent → the
        behavior-preserving defaults above apply. Any failure falls back to
        defaults so delegator construction never breaks.
        """
        knobs = cls()
        try:
            from src.config import get_config

            cfg = getattr(get_config(), "delegation", None)
        except Exception:  # pragma: no cover - defensive
            cfg = None
        if cfg is None:
            return knobs
        review_plane = getattr(cfg, "review_plane", None)
        for spec in fields(cls):
            value = _MISSING
            if review_plane is not None:
                value = getattr(review_plane, spec.name, _MISSING)
            if value is _MISSING:
                value = getattr(cfg, spec.name, _MISSING)
            if value is not _MISSING:
                setattr(knobs, spec.name, value)
        return knobs


class ProactiveDelegator:
    """Orchestrates proactive delegation workflow.

    Complexity-aware routing:
        TRIVIAL  -> Frontdoor answers directly (no delegation)
        SIMPLE   -> Frontdoor executes in REPL (no architect)
        MODERATE -> Frontdoor delegates to single specialist (no architect)
        COMPLEX  -> Architect generates TaskIR, multi-specialist workflow

    Only COMPLEX tasks invoke the expensive architect model.
    """

    def __init__(
        self,
        registry: "RegistryLoader",
        primitives: "LLMPrimitives",
        progress_logger: "ProgressLogger | None" = None,
        hybrid_router: Any | None = None,
        max_iterations: int = 3,
        max_total_iterations: int = 10,
        skip_complexity_check: bool = False,
        review_knobs: "ReviewPlaneKnobs | None" = None,
        output_extractor: "Callable[[str], str] | None" = None,
    ):
        """Initialize the delegator."""
        self.registry = registry
        self.primitives = primitives
        self.progress_logger = progress_logger
        self.hybrid_router = hybrid_router
        self.review_service = ArchitectReviewService(primitives)
        self.aggregation_service = AggregationService()
        self.skip_complexity_check = skip_complexity_check
        self.iteration_context = IterationContext(
            max_iterations=max_iterations,
            max_total_iterations=max_total_iterations,
        )
        # RD-10/RD-11 control-plane knobs. Default knobs preserve current behavior
        # (see ReviewPlaneKnobs); from_config() is the autopilot (AP-1) merge point.
        self._review_knobs = review_knobs or ReviewPlaneKnobs.from_config()
        # RD-10a AS-TOOL output-extraction seam (default = identity passthrough).
        self._output_extractor = output_extractor or _identity_output_extractor
        # RD-10c: honor the sticky-cache flag on the shared iteration context.
        self.iteration_context.decision_cache_enabled = (
            self._review_knobs.decision_cache_enabled
        )

    def _registry_has_role(self, role_name: str) -> bool:
        roles = getattr(self.registry, "roles", None)
        return isinstance(roles, dict) and role_name in roles

    def route_by_complexity(
        self,
        objective: str,
        task_ir: dict[str, Any] | None = None,
    ) -> tuple[TaskComplexity, str, ComplexitySignals, float]:
        """Determine delegation path based on task complexity + MemRL.

        Returns:
            (complexity, action, signals, confidence) where:
            - action: "direct", "repl", "specialist", or "architect"
            - signals.thinking_requested: True if should use architect-grade reasoning
            - confidence: 0.0-1.0 from MemRL (1.0 if no MemRL)
        """
        complexity, signals = classify_task_complexity(objective)
        confidence = 1.0

        if self.skip_complexity_check:
            return TaskComplexity.COMPLEX, "architect", signals, confidence

        # Consult MemRL HybridRouter if available
        if self.hybrid_router and task_ir:
            try:
                task_ir = canonicalize_task_ir(task_ir)
                # HybridRouter returns (roles, strategy) - "learned" or "rules"
                roles, strategy = self.hybrid_router.route(task_ir)

                # If learned routing suggests escalation to architect, upgrade complexity
                if strategy == "learned" and any("architect" in r for r in roles):
                    complexity = TaskComplexity.COMPLEX
                    confidence = 0.8  # High confidence from learned routing

                # If learned routing suggests deep reasoning
                elif strategy == "learned" and any("thinking" in r for r in roles):
                    signals.thinking_requested = True
                    confidence = 0.8

                # If learned routing suggests coder directly, may downgrade to MODERATE
                elif strategy == "learned" and any("coder" in r for r in roles):
                    if complexity == TaskComplexity.COMPLEX:
                        complexity = TaskComplexity.MODERATE
                        confidence = 0.7
            except Exception as e:
                logger.warning(f"HybridRouter query failed, using heuristics: {e}")
                confidence = 0.5

        action_map = {
            TaskComplexity.TRIVIAL: "direct",
            TaskComplexity.SIMPLE: "repl",
            TaskComplexity.MODERATE: "specialist",
            TaskComplexity.COMPLEX: "architect",
        }

        return complexity, action_map[complexity], signals, confidence

    def get_target_role(
        self,
        action: str,
        signals: ComplexitySignals,
    ) -> str:
        """Get the target role based on action and escalation flags."""
        # Retired thinking aliases fall through to architect_general.
        if signals.thinking_requested:
            return "architect_general"

        role_map = {
            "direct": "frontdoor",
            "repl": "frontdoor",
            "specialist": "coder_escalation",
            "architect": "architect_general",
        }
        return role_map.get(action, "frontdoor")

    def log_delegation_decision(
        self,
        task_id: str,
        complexity: TaskComplexity,
        action: str,
        confidence: float,
        difficulty_score: float = 0.0,
        difficulty_band: str = "",
    ) -> None:
        """Log delegation decision for MemRL Q-learning."""
        if self.progress_logger:
            self.progress_logger.log_delegation(
                task_id=task_id,
                complexity=complexity.value,
                action=action,
                confidence=confidence,
                difficulty_score=difficulty_score,
                difficulty_band=difficulty_band,
            )

    async def delegate(self, task_ir: dict[str, Any]) -> AggregatedResult:
        """Execute proactive delegation workflow."""
        task_id = task_ir.get("task_id", str(uuid.uuid4()))
        objective = task_ir.get("objective", "")

        # Log start
        if self.progress_logger:
            self.progress_logger.log_task_started(
                task_id=task_id,
                task_ir=task_ir,
                routing_decision=["proactive_delegation"],
                routing_strategy="proactive",
            )

        result = AggregatedResult(
            task_id=task_id,
            objective=objective,
        )

        # Extract subtasks from plan
        plan = task_ir.get("plan", {})
        steps = plan.get("steps", [])

        if not steps:
            logger.warning(f"No steps in TaskIR plan for task {task_id}")
            result.aggregated_output = "[ERROR: No subtasks in plan]"
            return result

        # Execute subtasks: wave-based if parallel_execution enabled, else sequential
        from src.features import features as _get_features

        plan_start = time.monotonic()
        used_parallel = False
        # RD-10b: set when any subtask was routed AROUND per-subtask review by the
        # complexity gate; if so, a single final-aggregate review runs below.
        # Default knobs never gate anything → stays False → no aggregate pass.
        skipped_any = False

        if _get_features().parallel_execution and len(steps) > 1:
            from src.parallel_step_executor import compute_waves, StepExecutor

            used_parallel = True
            waves = compute_waves(steps)
            max_concurrent = plan.get("parallelism", {}).get(
                "max_concurrent_steps",
                2,
            )
            executor = StepExecutor(
                primitives=self.primitives,
                review_service=self.review_service,
                iteration_context=self.iteration_context,
                hybrid_router=self.hybrid_router,
                max_burst_concurrent=max_concurrent,
                # RD-10b/RD-10c: parallel waves honor the same review gate + sticky
                # cache (default predicate reviews every step → behavior preserved).
                should_review=self._should_review_subtask,
            )
            subtask_results = await executor.execute_plan(
                task_ir,
                waves,
                ROLE_MAPPING,
            )
            if getattr(executor, "reviews_skipped", 0):
                skipped_any = True
            for sr in subtask_results:
                result.subtask_results.append(sr)
                if sr.role not in result.roles_used:
                    result.roles_used.append(sr.role)
        else:
            reminder = ""
            for step_index, step in enumerate(steps):
                # RD-9 plan-reminder call-site (default cadence 0 → no-op).
                reminder = self._maybe_plan_reminder(task_ir, steps, step_index, reminder)
                # RD-10b complexity-gated per-subtask review placement.
                review_enabled = self._should_review_subtask(step)
                if not review_enabled:
                    skipped_any = True
                subtask_result = await self._execute_with_review(
                    task_ir,
                    step,
                    review_enabled=review_enabled,
                    reminder=reminder,
                )
                result.subtask_results.append(subtask_result)
                if subtask_result.role not in result.roles_used:
                    result.roles_used.append(subtask_result.role)

        # Delegation telemetry
        for sr in result.subtask_results:
            result.delegation_events.append(
                {
                    "from_role": "proactive_delegation",
                    "to_role": sr.role,
                    "task_summary": sr.subtask_id,
                    "success": sr.success,
                    "elapsed_ms": round(sr.elapsed_seconds * 1000),
                    "tokens_generated": sr.tokens_used,
                }
            )

        plan_elapsed = time.monotonic() - plan_start

        # Critical path metrics (post-hoc observability)
        if used_parallel and len(result.subtask_results) > 1:
            try:
                from src.metrics.critical_path import compute_critical_path
                from src.parallel_step_executor import extract_step_timings

                timings = extract_step_timings(result.subtask_results, steps)
                cp_report = compute_critical_path(
                    timings,
                    wall_clock_seconds=plan_elapsed,
                )
                logger.info(
                    "Critical path: %.1fs (%d steps), parallelism ratio: %.2f, "
                    "total work: %.1fs, wall clock: %.1fs",
                    cp_report.critical_path_seconds,
                    len(cp_report.critical_path_steps),
                    cp_report.parallelism_ratio,
                    cp_report.total_work_seconds,
                    cp_report.wall_clock_seconds,
                )
            except Exception as e:
                logger.debug("Critical path computation skipped: %s", e)

        # Aggregate results
        result.aggregated_output = self.aggregation_service.aggregate(
            result.subtask_results,
            strategy="concatenate",
        )

        # RD-10b: single final-aggregate review when the complexity gate routed
        # one or more subtasks around per-subtask review. Dormant by default
        # (skipped_any stays False under default knobs → this block never runs).
        aggregate_rejected = False
        if skipped_any and self.review_service is not None:
            try:
                aggregate_review = self.review_service.review(
                    spec=task_ir,
                    subtask={"id": "__final_aggregate__", "action": "final aggregate review"},
                    output=result.aggregated_output,
                )
                self.iteration_context.record_iteration(
                    subtask_id="__final_aggregate__",
                    decision=aggregate_review.decision,
                    feedback=aggregate_review.feedback,
                )
                aggregate_rejected = aggregate_review.decision in (
                    ReviewDecision.REJECT,
                    ReviewDecision.REJECT_TO_EMPTY,
                )
            except Exception as e:
                logger.warning("Final-aggregate review failed: %s", e)

        result.all_approved = (
            all(r.success or getattr(r, 'partial', False) for r in result.subtask_results)
            and not aggregate_rejected
        )
        result.total_iterations = self.iteration_context.total_iterations

        # Log completion
        if self.progress_logger:
            self.progress_logger.log_task_completed(
                task_id=task_id,
                success=result.all_approved,
                details=f"Proactive delegation: {len(result.subtask_results)} subtasks, "
                f"{result.total_iterations} iterations",
            )

        return result

    async def _execute_with_review(
        self,
        task_ir: dict[str, Any],
        step: dict[str, Any],
        *,
        review_enabled: bool = True,
        reminder: str = "",
    ) -> SubtaskResult:
        """Execute a subtask in AS-TOOL mode (manager keeps control).

        ``review_enabled=False`` (RD-10b complexity gate) runs the specialist
        once and accepts the output without per-subtask review — the single
        final-aggregate review in ``delegate()`` covers it. ``reminder`` (RD-9)
        is prepended to the prompt when non-empty. The defaults
        (``review_enabled=True``, ``reminder=""``) reproduce the original
        review loop byte-for-byte.
        """
        subtask_id = step.get("id", f"S{uuid.uuid4().hex[:4]}")
        actor = step.get("actor", "worker")
        step.get("action", "")

        # Map actor to registry role.
        # Canonical roles resolve first, then generic chain names such as
        # worker/ingest/architect fall through the shared chain helper. The
        # remaining compatibility table only keeps non-standard spellings.
        canonical_actor = Role.from_string(actor)
        if canonical_actor is None:
            canonical_actor = chain_name_to_role(actor)

        if canonical_actor is not None and self._registry_has_role(canonical_actor.value):
            role = canonical_actor.value
        else:
            resolved_key = canonical_actor.value if canonical_actor is not None else actor
            role = ROLE_MAPPING.get(
                resolved_key,
                ROLE_MAPPING.get(actor, Role.WORKER_GENERAL.value),
            )

        # Build prompt for specialist (RD-9 reminder prepended when present).
        prompt = self._build_specialist_prompt(task_ir, step)
        if reminder:
            prompt = f"{reminder}\n\n{prompt}"

        # RD-10b: gated out of per-subtask review → single AS-TOOL call, accepted.
        if not review_enabled:
            try:
                raw = self.primitives.llm_call(prompt, role=role, n_tokens=1024)
            except Exception as e:
                exc = StepExecutionError(subtask_id, role, cause=e)
                logger.warning("%s", exc, exc_info=True)
                return SubtaskResult(
                    subtask_id=subtask_id,
                    role=role,
                    output="",
                    success=False,
                    error=str(exc),
                )
            return SubtaskResult(
                subtask_id=subtask_id,
                role=role,
                output=self._output_extractor(raw),
                success=True,
            )

        # Iteration loop
        current_output = ""
        feedback_history: list[str] = []
        # RD-10a HANDOFF input_filter: the feedback-history slice carried forward
        # (was hard-coded ``[-3:]``; window <= 0 → carry nothing).
        window = self._review_knobs.input_filter_window
        ctx = self.iteration_context

        while ctx.can_iterate(subtask_id):
            # Include feedback from previous iterations
            if feedback_history:
                sliced = feedback_history[-window:] if window > 0 else []
                prompt_with_feedback = (
                    prompt
                    + "\n\n## Previous Feedback\n"
                    + "\n".join(f"- {fb}" for fb in sliced)
                    + "\n\nAddress the feedback above."
                )
            else:
                prompt_with_feedback = prompt

            # Call specialist (AS-TOOL: raw output flows through the extractor seam)
            try:
                current_output = self._output_extractor(
                    self.primitives.llm_call(
                        prompt_with_feedback,
                        role=role,
                        n_tokens=1024,
                    )
                )
            except Exception as e:
                exc = StepExecutionError(subtask_id, role, cause=e)
                logger.warning("%s", exc, exc_info=True)
                return SubtaskResult(
                    subtask_id=subtask_id,
                    role=role,
                    output="",
                    success=False,
                    error=str(exc),
                )

            # RD-10c sticky decision cache: a cached APPROVE pattern skips re-review.
            signature = None
            if ctx.decision_cache_enabled:
                signature = ctx.subtask_signature(task_ir, step, current_output)
                if ctx.cached_decision(signature) == ReviewDecision.APPROVE:
                    return SubtaskResult(
                        subtask_id=subtask_id,
                        role=role,
                        output=current_output,
                        success=True,
                    )

            # Architect review. UTM-P1a.2: the delegator OWNS the review-fix loop, so
            # it genuinely knows which iteration this review is -- the count of
            # iterations already recorded for this step (0-based). The aggregate
            # review and the parallel executor run one review with no loop, so they
            # pass nothing and their ordinal stays NULL rather than an invented 0.
            review = self.review_service.review(
                spec=task_ir,
                subtask=step,
                output=current_output,
                turn_ordinal=ctx.subtask_iterations.get(subtask_id, 0),
            )

            # Record iteration
            self.iteration_context.record_iteration(
                subtask_id=subtask_id,
                decision=review.decision,
                feedback=review.feedback,
            )

            # Log escalation if needed. 2026-08-01: to_tier was hardcoded to
            # "architect_general" — a fifth restatement of the ladder, 25 lines above
            # the escalation it describes, that logged the wrong destination for every
            # role whose target is not architect_general. Ask the canonical ladder
            # (_escalate_role is pure, so calling it twice is free).
            if review.decision == ReviewDecision.ESCALATE and self.progress_logger:
                self.progress_logger.log_escalation(
                    task_id=task_ir.get("task_id", ""),
                    from_tier=role,
                    to_tier=self._escalate_role(role),
                    reason=f"Review escalation: {review.feedback}",
                )

            # Check decision
            if review.decision == ReviewDecision.APPROVE:
                # RD-10c: remember APPROVE so an equivalent pattern skips re-review.
                if signature is not None:
                    ctx.remember_decision(signature, review.decision)
                return SubtaskResult(
                    subtask_id=subtask_id,
                    role=role,
                    output=review.approved_output or current_output,
                    success=True,
                )
            elif review.decision == ReviewDecision.REJECT:
                return SubtaskResult(
                    subtask_id=subtask_id,
                    role=role,
                    output=current_output,
                    success=False,
                    error=f"Rejected: {review.feedback}",
                )
            elif review.decision == ReviewDecision.ESCALATE:
                # Escalate to higher-tier role
                role = self._escalate_role(role)

            # Add feedback for next iteration
            if review.feedback:
                feedback_history.append(review.feedback)

        # Max iterations reached
        return SubtaskResult(
            subtask_id=subtask_id,
            role=role,
            output=current_output,
            success=False,
            error=f"Max iterations ({self.iteration_context.max_iterations}) reached",
        )

    def _build_specialist_prompt(
        self,
        task_ir: dict[str, Any],
        step: dict[str, Any],
    ) -> str:
        """Build prompt for specialist from TaskIR and step."""
        objective = task_ir.get("objective", "")
        action = step.get("action", "")
        inputs = step.get("inputs", [])
        outputs = step.get("outputs", [])

        prompt_parts = [
            f"# Task: {action}",
            "",
            "## Overall Objective",
            objective,
            "",
        ]

        if inputs:
            prompt_parts.extend(
                [
                    "## Inputs",
                    "\n".join(f"- {i}" for i in inputs),
                    "",
                ]
            )

        if outputs:
            prompt_parts.extend(
                [
                    "## Expected Outputs",
                    "\n".join(f"- {o}" for o in outputs),
                    "",
                ]
            )

        prompt_parts.extend(
            [
                "## Instructions",
                "Complete the task above. Provide your output directly.",
            ]
        )

        return "\n".join(prompt_parts)

    def _escalate_role(self, current_role: str) -> str:
        """Get escalated role for current role, via the canonical ladder.

        2026-08-01: this held a private FOURTH copy of the escalation ladder. It had
        drifted into two dead hops — ``architect_general -> architect_general`` (a
        self-loop) and, after the W1 cutover, ``coder_escalation -> architect_general``
        (the same model in the same process on :8083, so escalating could only change
        the sampling seed). ``src.roles._ESCALATION_MAP``, reached through
        ``Role.escalates_to()``, is the single source of truth.

        Behaviour preserved from the private map:
          * returns a role-name ``str`` (the caller reassigns ``role`` and keeps
            dispatching, so ``None`` is not a legal return);
          * a string that is not a known role still falls back to
            ``architect_general``.
        Terminal rungs (``architect_general`` — terminal again since the 2026-09-27
        ARCHITECT SWAP; ``architect_critic`` was terminal 2026-08-01 → 2026-09-27)
        return the input unchanged — the old map expressed exactly that for the
        then-terminal ``architect_general``.
        """
        role = Role.from_string(current_role)
        if role is None:
            return Role.ARCHITECT_GENERAL.value
        target = role.escalates_to()
        if target is None:
            return role.value
        return target.value

    def _should_review_subtask(self, step: dict[str, Any]) -> bool:
        """Complexity-gated per-subtask review placement (RD-10b).

        Default knobs (``per_subtask_review_enabled=True``,
        ``review_trigger_complexity_threshold=0``) → always True → review every
        subtask (current behavior; the threshold short-circuits *before* any
        complexity classification, so the default path is unchanged). Raising the
        threshold reviews only subtasks whose complexity rank meets it; the rest
        are routed to the single final-aggregate review in ``delegate()``.
        """
        knobs = self._review_knobs
        if not knobs.per_subtask_review_enabled:
            return False
        if knobs.review_trigger_complexity_threshold <= 0:
            return True
        complexity, _signals = classify_task_complexity(step.get("action", ""))
        return _COMPLEXITY_RANK.get(complexity, 0) >= knobs.review_trigger_complexity_threshold

    def _maybe_plan_reminder(
        self,
        task_ir: dict[str, Any],
        steps: list[dict[str, Any]],
        step_index: int,
        current_reminder: str,
    ) -> str:
        """Plan-reminder call-site (RD-9). See the module docstring for the contract.

        Fires only when ``reminder_cadence > 0`` and the step boundary is hit.
        Resolves ``review_service.build_plan_reminder`` via ``getattr`` and tolerates
        its absence / any error (the review_service agent owns the helper) so neither
        agent blocks the other. Default cadence 0 → returns ``current_reminder``
        unchanged (no-op).
        """
        cadence = self._review_knobs.reminder_cadence
        if cadence <= 0 or step_index == 0 or step_index % cadence != 0:
            return current_reminder
        helper = getattr(self.review_service, "build_plan_reminder", None)
        if helper is None:
            return current_reminder
        try:
            reminder = helper(
                objective=task_ir.get("objective", ""),
                plan_steps=steps,
                step_index=step_index,
            )
        except Exception as e:  # pragma: no cover - defensive; never blocks
            logger.debug("plan-reminder helper failed: %s", e)
            return current_reminder
        return reminder or current_reminder
