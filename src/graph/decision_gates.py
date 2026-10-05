"""Decision gate helpers for orchestration graph nodes.

Implements binary state-transition decisions: escalation, retry, approval,
timeout skip, and end-result construction. Extracted from graph/helpers.py —
all callers continue to import from helpers via compatibility re-exports.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from pydantic_graph import End, GraphRunContext

from src.escalation import ErrorCategory
from src.graph.escalation_helpers import detect_role_cycle as _detect_role_cycle_impl
from src.graph.observability import _add_evidence
from src.graph.state import TaskDeps, TaskResult, TaskState
from src.graph.think_harder import _update_think_harder_stats
from src.roles import Role

log = logging.getLogger(__name__)

Ctx = GraphRunContext[TaskState, TaskDeps]


def _admit_early_abort_escalation(ctx: Ctx) -> bool:
    """Apply the shared budget and role-cycle gate for first-strike aborts.

    Early aborts intentionally bypass the retry-count threshold, but they may
    not bypass escalation admission controls. This predicate is side-effect
    free so both graph engines can preserve their existing fallthrough path
    when admission is denied.
    """
    state = ctx.state
    if state.escalation_count >= ctx.deps.config.max_escalations:
        return False
    return not _detect_role_cycle_impl(state.role_history)


def _is_infra_failure(artifacts: dict) -> bool:
    """True when this turn's ``error`` is an in-band backend/infra sentinel.

    ``_execute_turn`` (``graph/helpers.py::_backend_infra_sentinel``) tags
    ``artifacts["_infra_failure"]`` when the raw LLM-call output IS the
    orchestrator's own ``[ERROR: ...]`` string (a placement/admission timeout,
    a circuit-open refusal, a dead connection, ...) rather than genuine model
    text -- the backend denied the call before generation started. Every
    caller MUST check this FIRST, before ``_classify_error``/``_should_retry``/
    ``_should_escalate``, and end the turn immediately with ``error`` passed
    through VERBATIM (never wrapped as ``[FAILED: ...]``, which would erase
    the ``[ERROR: ...]`` prefix ``_annotate_error`` classifies into the right
    HTTP status). No retry, no escalation: the denial is a resource-contention
    fact about the host, not a capability gap a different role fixes by being
    "smarter" -- escalating just re-queues the same wait against the same
    held region.
    """
    return bool(artifacts.get("_infra_failure"))


def _should_escalate(
    ctx: Ctx,
    error_category: ErrorCategory,
    next_tier: Role | None,
) -> bool:
    """Determine if we should escalate (vs retry or fail)."""
    cfg = ctx.deps.config
    state = ctx.state

    # Format errors never escalate
    if error_category in cfg.no_escalate_categories:
        return False

    # Schema errors: escalate only after retries and on capability-gap signature.
    if error_category == ErrorCategory.SCHEMA:
        if next_tier is None:
            return False
        if state.escalation_count >= cfg.max_escalations:
            return False
        if state.consecutive_failures < cfg.max_retries:
            return False
        lower = (state.last_error or "").lower()
        parser_patterns = (
            "json decode",
            "expecting value",
            "unterminated string",
            "trailing comma",
            "invalid json",
            "parse error",
        )
        if any(p in lower for p in parser_patterns):
            return False
        capability_patterns = (
            "schema mismatch",
            "validation failed",
            "does not conform",
            "required property",
            "invalid type",
            "enum",
            "oneof",
            "anyof",
            "allof",
        )
        return any(p in lower for p in capability_patterns)

    # No target to escalate to
    if next_tier is None:
        return False

    # Max escalations reached
    if state.escalation_count >= cfg.max_escalations:
        return False

    # Cross-chain cycle detection: block A→B→A→B bouncing
    if _detect_role_cycle_impl(state.role_history):
        log.warning(
            "Escalation cycle detected, refusing escalation: %s",
            state.role_history[-6:],
        )
        return False

    # Retries exhausted → escalate
    return state.consecutive_failures >= cfg.max_retries


def _should_retry(ctx: Ctx, error_category: ErrorCategory) -> bool:
    """Determine if we should retry with the same role."""
    # Timeout retries are high-cost and commonly non-productive in current
    # infra path (e.g., repeated 90s frontdoor timeouts). Fail fast so
    # escalation/benchmark logic can move on.
    if error_category == ErrorCategory.TIMEOUT:
        return False
    cfg = ctx.deps.config
    return ctx.state.consecutive_failures < cfg.max_retries


def _check_approval_gate(
    ctx: Ctx,
    from_role: str,
    to_role: str,
    reason: str,
) -> bool:
    """Check approval gate before escalation. Returns True if approved."""
    from src.features import features as _get_features

    if not _get_features().approval_gates:
        return True

    from src.graph.approval_gate import request_approval_for_escalation, ApprovalDecision

    decision = request_approval_for_escalation(ctx, from_role, to_role, reason)
    return decision == ApprovalDecision.APPROVE


def _timeout_skip(ctx: Ctx, error_msg: str) -> bool:
    """Check if a timeout error should result in a SKIP (optional gate)."""
    # For now, check if the error mentions an optional gate
    cfg = ctx.deps.config
    for gate in cfg.optional_gates:
        if gate in error_msg.lower():
            return True
    return False


def _make_end_result(ctx: Ctx, answer: str, success: bool) -> End[TaskResult]:
    """Create an End node with a TaskResult."""
    repl = ctx.deps.repl
    tool_outputs = []
    tools_used = 0
    if repl and hasattr(repl, "artifacts"):
        tool_outputs = repl.artifacts.get("_tool_outputs", [])
    if repl and hasattr(repl, "_tool_invocations"):
        tools_used = repl._tool_invocations

    # Record outcome evidence
    _add_evidence(ctx, "success" if success else "failure", 0.5 if success else -0.5)
    _update_think_harder_stats(ctx)
    ws = ctx.state.workspace_state
    if isinstance(ws, dict):
        ws.setdefault("decisions", []).append(
            {
                "id": f"d{ctx.state.turns}",
                "text": answer[:180],
                "rationale": "success" if success else "failure",
            }
        )
        if len(ws["decisions"]) > 12:
            ws["decisions"] = ws["decisions"][-12:]
        ws["updated_at"] = datetime.now(timezone.utc).isoformat()

    return End(
        TaskResult(
            answer=answer,
            success=success,
            role_history=list(ctx.state.role_history),
            tool_outputs=tool_outputs,
            tools_used=tools_used,
            turns=ctx.state.turns,
            delegation_events=list(ctx.state.delegation_events),
        )
    )
