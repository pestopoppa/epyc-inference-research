"""Quality gates and architect plan review for chat endpoints.

Extracted from chat.py during Phase 1 decomposition.
Contains: output quality detection and the plan review pipeline.

The MemRL-conditional answer review gate (``_should_review`` -> architect verdict
-> ``worker_general`` revision) and its RI-18 telemetry (the ``review_gate`` tap
event, the ``review_gate_score`` accessor, the verdict ``question_cap``) were
removed by RI-18c, applying RI-18's pre-registered DROP verdict (2026-10-01):
reviewing every answer nets -38.3 per 100 (12 fixed, 234 broken), and the
production ``review_low_q_threshold`` 0.6 gate fired 0 times in 579 items.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.config import get_config as _get_config
from src.constants import TASK_IR_OBJECTIVE_LEN
from src.roles import Role, chain_name_to_role
from src.roles import resolve_reviewer_role as _resolve_reviewer_role

log = logging.getLogger(__name__)


def _synthesized_plan_action(objective: str) -> str:
    """Describe the work to be reviewed, rather than echoing the user prompt.

    Chat requests normally arrive without an explicit multi-step plan. The
    review gate therefore creates a one-step plan for the architect. Passing
    the first 50 characters of the objective as that step's ``action`` made a
    request such as "You are an expert code verifier..." look like a plan that
    merely repeated the prompt. A ``drop`` verdict was then correct, but it
    rejected our synthetic plan rather than the user's task.
    """
    text = objective.lower()
    code_markers = ("code", "python", "function", "candidate solution", "assert ")
    if any(marker in text for marker in code_markers):
        return "Inspect the candidate code and verify expected behavior"
    return "Analyze the request and produce a directly supported answer"


if TYPE_CHECKING:
    from src.api.state import AppState
    from src.llm_primitives import LLMPrimitives
    from src.proactive_delegation.types import PlanReviewResult


def _detect_output_quality_issue(answer: str) -> str | None:
    """Detect quality issues in model output using text-based heuristics.

    Delegates to src.classifiers.quality_detector.detect_output_quality_issue.
    """
    from src.classifiers import detect_output_quality_issue

    return detect_output_quality_issue(answer)


# ── Architect Plan Review Gate ─────────────────────────────────────────────


def _needs_plan_review(
    task_ir: dict,
    routing_decision: list,
    state: "AppState",
) -> bool:
    """Determine whether the plan needs architect review before execution.

    Bypass conditions (skip review when any is true):
    1. TaskComplexity is TRIVIAL or SIMPLE
    2. TaskComplexity is COMPLEX (architect already owns plan)
    3. Single-step plan (no multi-step coordination to review)
    4. Architect is already the actor (no self-review)
    5. Phase B: Q-value >= 0.6 for task class
    6. Phase C: 90% skip (stochastic)
    7. Feature flag disabled (checked by caller)

    Args:
        task_ir: TaskIR dict with objective, task_type.
        routing_decision: List of roles selected for routing.
        state: Application state.

    Returns:
        True if architect plan review should run.
    """
    import random
    from src.proactive_delegation import classify_task_complexity, TaskComplexity

    objective = task_ir.get("objective", "")
    complexity, _signals = classify_task_complexity(objective)

    # Bypass 1+2: Only review MODERATE complexity
    if complexity != TaskComplexity.MODERATE:
        return False

    # Bypass 3: Single-step plans don't need coordination review
    plan = task_ir.get("plan", {})
    steps = plan.get("steps", [])
    if len(steps) <= 1:
        # No explicit plan steps yet — check routing for multi-role indication
        # For chat requests, routing_decision is typically 1 role, but plan
        # review is still useful if complexity is MODERATE
        pass  # Allow review for MODERATE tasks even without explicit steps

    # Bypass 4: Don't self-review architect
    if routing_decision and "architect" in str(routing_decision[0]):
        return False

    # Phase-dependent gating
    phase = state.plan_review_phase
    _chat_cfg = _get_config().chat

    # Bypass 6: Phase C — stochastic skip (default 90% skip rate)
    if phase == "C":
        if random.random() < _chat_cfg.plan_review_phase_c_skip_rate:
            return False

    # Bypass 5: Phase B — Q-value gating
    if phase == "B" and state.hybrid_router:
        try:
            retriever = state.hybrid_router.retriever
            results = retriever.retrieve_for_routing(task_ir)
            if results:
                avg_q = sum(r.q_value for r in results) / len(results)
                if avg_q >= _chat_cfg.review_skip_q_threshold:
                    return False
        except Exception as exc:
            log.debug("Plan review Q-value gating failed: %s", exc)

    return True


def _architect_plan_review(
    task_ir: dict,
    routing_decision: list,
    primitives: "LLMPrimitives",
    state: "AppState",
    task_id: str,
) -> "PlanReviewResult | None":
    """Execute architect plan review and return result.

    Non-blocking: returns None on timeout or error.

    Args:
        task_ir: TaskIR dict.
        routing_decision: Current routing decision.
        primitives: LLM primitives for calling architect.
        state: Application state.
        task_id: Current task ID.

    Returns:
        PlanReviewResult or None on failure.
    """
    from src.proactive_delegation import ArchitectReviewService

    objective = task_ir.get("objective", "")
    task_type = task_ir.get("task_type", "chat")

    # Construct plan steps from routing decision (minimal for chat requests)
    plan = task_ir.get("plan", {})
    plan_steps = plan.get("steps", [])

    # If no explicit plan, synthesize from routing_decision
    if not plan_steps and routing_decision:
        plan_steps = [
            {
                "id": f"S{i + 1}",
                "actor": str(role),
                "action": _synthesized_plan_action(objective),
                "outputs": [],
            }
            for i, role in enumerate(routing_decision)
        ]

    if not plan_steps:
        return None

    # RD-1/RD-5/TM-3: the reviewer role is resolved via the config-level binding
    # inside the service (default → architect_critic since ARCHSWAP-20260927, i.e. the
    # same MI210 27B that reviewed before the swap).
    # review_plan() emits an always-on shadow trace event regardless of whether the
    # plan_review feature acts — this is the DECOUPLED shadow-emission path: the
    # trace flows even though the plan_review flag itself requires memrl (features.py
    # validate()), because emission lives on the review_service seam, not the flag.
    # We anchor emitted rows to this task via session_id so decision_chain replay
    # (src/trace/query.py) can reconstruct task → plan → review → gate → outcome.
    review_service = ArchitectReviewService(primitives)
    result = review_service.review_plan(
        objective=objective,
        task_type=task_type,
        plan_steps=plan_steps,
        session_id=task_id,
    )

    if result:
        log.info(
            "Plan review: decision=%s score=%.2f feedback=%s",
            result.decision,
            result.score,
            result.feedback[:60],
        )

    return result


def _apply_plan_review(
    routing_decision: list,
    review: "PlanReviewResult",
) -> list:
    """Apply architect's plan review corrections to routing decision.

    Handles 'reroute' patches that change which specialist handles a step.

    Args:
        routing_decision: Current routing decision list.
        review: Architect's plan review result.

    Returns:
        Updated routing decision (may be unchanged if no reroute patches).
    """
    if not review.patches:
        return routing_decision

    updated = list(routing_decision)

    for patch in review.patches:
        op = patch.get("op", "")
        if op == "reroute":
            new_role = _normalize_plan_review_reroute_role(patch.get("v", ""))
            step_id = patch.get("step", "")
            if new_role:
                # Map step index to routing decision index
                # S1 → index 0, S2 → index 1, etc.
                try:
                    idx = int(step_id.replace("S", "")) - 1
                    if 0 <= idx < len(updated):
                        updated[idx] = new_role
                    elif idx == 0 and len(updated) >= 1:
                        updated[0] = new_role
                except (ValueError, IndexError):
                    # If step_id isn't parseable, reroute the first step
                    if updated:
                        updated[0] = new_role

    return updated


def _normalize_plan_review_reroute_role(value: object) -> str:
    """Return a canonical role for a plan-review reroute target, or "".

    Architect patches are model output, so the ``v`` field can contain free-form
    task guidance instead of a backend role. Letting that text overwrite
    ``routing_decision`` later turns it into ``force_role`` and fails at
    inference time with "No backend configured for role ...".
    """
    raw = str(value or "").strip()
    if not raw:
        return ""
    role = Role.from_string(raw) or chain_name_to_role(raw)
    if role is None:
        log.warning("Ignoring invalid plan-review reroute target: %r", raw[:120])
        return ""
    return str(role)


def _plan_review_should_abort(review: "PlanReviewResult | None") -> bool:
    """Return True only for an explicit execution-abort verdict.

    ``drop`` is a plan operation: discard the reviewed plan and proceed through
    the normal no-plan/default route. Treating it as a task rejection violated
    ``ArchitectReviewService.review_plan``'s non-blocking contract and turned a
    reviewer correctly spotting a bad synthesized plan into an HTTP 422 task
    failure. ``abort`` is reserved for a future separately-specified safety
    verdict; the current reviewer schema does not emit it.
    """
    return bool(review and str(review.decision).strip().lower() == "abort")


def _plan_review_abort_message(review: "PlanReviewResult") -> str:
    """Human-readable message for a terminal plan-review rejection."""
    feedback = str(review.feedback or "").strip()
    if feedback:
        return f"Plan rejected by architect review: {feedback}"
    return "Plan rejected by architect review."


def _store_plan_review_episode(
    state: "AppState",
    task_id: str,
    task_ir: dict,
    review: "PlanReviewResult",
) -> None:
    """Store plan review result as MemRL episode and progress log entry.

    Architect corrections become high-quality training signals:
    - review.score mapped to reward: score * 2 - 1 (0-1 → -1..+1)
    - Action: plan:{role1},{role2} (the routing decision)

    Args:
        state: Application state.
        task_id: Current task ID.
        task_ir: TaskIR dict.
        review: Architect's plan review result.
    """
    # Log to progress JSONL
    if state.progress_logger:
        from orchestration.repl_memory.progress_logger import ProgressEntry, EventType

        # RI-6: Structured review context replaces truncated feedback proxy.
        # Provides task_type, risk_band, key_claims for richer MemRL signal.
        _review_context = {
            "decision": review.decision,
            "score": review.score,
            "feedback": review.feedback[:200],
            "patches": review.patches[:5],
            "task_type": task_ir.get("task_type", "chat"),
            "risk_band": task_ir.get("risk_band", ""),
            "verification_focus": review.decision if review.decision != "ok" else "",
        }
        state.progress_logger.log(
            ProgressEntry(
                event_type=EventType.PLAN_REVIEWED,
                task_id=task_id,
                agent_role=str(_resolve_reviewer_role()),
                data=_review_context,
                outcome="success" if review.is_ok else "corrected",
            )
        )

    # Update plan review stats (thread-safe)
    stats = state.update_plan_review_stats(approved=review.is_ok)

    # Recompute phase
    state.plan_review_phase = _compute_plan_review_phase(stats)

    # Store as MemRL episode for Q-learning (expert demonstration)
    if state.q_scorer and state.hybrid_router:
        try:
            reward = review.score * 2 - 1  # Map 0-1 to -1..+1
            state.q_scorer.score_external_result(
                task_description=task_ir.get("objective", "")[:TASK_IR_OBJECTIVE_LEN],
                action=f"plan_review:{review.decision}",
                reward=reward,
                action_type="plan_review",
                context={
                    "task_type": task_ir.get("task_type", "chat"),
                    "review_decision": review.decision,
                    "review_feedback": review.feedback[:200],
                    "risk_band": task_ir.get("risk_band", ""),
                    "verification_focus": review.decision if review.decision != "ok" else "",
                    "source": "plan_review",
                },
            )
        except Exception as exc:
            log.debug("MemRL plan review score storage failed: %s", exc)


def _compute_plan_review_phase(stats: dict) -> str:
    """Compute current plan review phase from statistics.

    Phase A (bootstrap): < 50 reviews or low Q-values
    Phase B (supervised fade): mean Q >= 0.7, min Q >= 0.5
    Phase C (spot-check): min Q >= 0.7 and >= 100 reviews

    Args:
        stats: Plan review statistics dict.

    Returns:
        Phase string: "A", "B", or "C".
    """
    _pr = _get_config().chat

    total = stats.get("total_reviews", 0)
    if total < _pr.plan_review_phase_a_min:
        return "A"

    q_vals = stats.get("task_class_q_values", {})
    if not q_vals:
        return "A"

    values = list(q_vals.values())
    mean_q = sum(values) / len(values)
    min_q = min(values)

    if min_q >= _pr.plan_review_phase_c_min_q and total >= _pr.plan_review_phase_c_min_total:
        return "C"
    if mean_q >= _pr.plan_review_phase_b_mean_q and min_q >= _pr.plan_review_phase_b_min_q:
        return "B"
    return "A"
