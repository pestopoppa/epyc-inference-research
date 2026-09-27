"""Halt-and-resume approval protocol (Lobster pattern).

Provides human approval gates at escalation boundaries and destructive
tool invocations. Graph halts, serializes state via resume token,
waits for approval, then resumes.

Only active when features().approval_gates is True.

Depends on:
- Phase 1A (side_effect_tracking) for destructive tool detection
- Phase 3B (resume_tokens) for state serialization

Halt triggers:
1. Escalation boundary — tier crossing (Worker→Coder, Coder→Architect)
2. Destructive tool — tool.destructive == True
3. High-cost invocation — architect-tier models (configurable threshold)

Usage:
    from src.graph.approval_gate import should_halt, HaltReason, AutoApproveCallback

    # In a graph node, before escalation:
    if should_halt(state, from_role, to_role, deps):
        halt = HaltState(reason=HaltReason.ESCALATION, ...)
        decision = deps.approval_callback.request_approval(halt)
        if decision == ApprovalDecision.REJECT:
            return End(TaskResult(success=False, answer="Rejected"))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger(__name__)


class HaltReason(str, Enum):
    """Why the graph is halting for approval."""

    ESCALATION = "escalation"  # Tier crossing
    DESTRUCTIVE_TOOL = "destructive_tool"  # Tool with destructive=True
    HIGH_COST = "high_cost"  # Architect-tier invocation
    INTERRUPT_CONDITION = "interrupt_condition"  # Generalized interrupt


class ApprovalDecision(str, Enum):
    """User's decision on a halt request."""

    APPROVE = "approve"
    REJECT = "reject"


@dataclass
class HaltState:
    """State captured when graph halts for approval."""

    reason: HaltReason
    from_role: str = ""
    to_role: str = ""
    resume_token: str = ""
    side_effects: list[str] = field(default_factory=list)
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ApprovalCallback(Protocol):
    """Protocol for approval decision providers.

    Implementations can be synchronous (CLI prompt), async (web UI),
    or auto-approve (preserves current behavior).
    """

    def request_approval(self, halt: HaltState) -> ApprovalDecision: ...


class AutoApproveCallback:
    """Default callback that auto-approves everything.

    Preserves current behavior when approval_gates is disabled or
    no callback is injected into TaskDeps.
    """

    def request_approval(self, halt: HaltState) -> ApprovalDecision:
        log.debug("Auto-approving: %s (%s → %s)", halt.reason, halt.from_role, halt.to_role)
        return ApprovalDecision.APPROVE


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STACK_PRIORS_PATH = PROJECT_ROOT / "orchestration" / "derived" / "stack_priors.yaml"

_HIGH_COST_MIN_MODEL_MEM_GB = 60.0
# Degraded fallback for the >= 60 GB model role. It read architect_general while
# the 2026-08-01 → 2026-09-27 lineup put the big CPU model on architect_critic (so
# the degraded path gated the wrong role); since the 2026-09-27 ARCHITECT SWAP
# architect_general IS the ~90 GB Flash-Next CPU :8074 process again.
_DEGRADED_HIGH_COST_ROLES = frozenset({"architect_general"})
_STACK_PRIOR_HIGH_COST_ROLES_CACHE: frozenset[str] | None = None


def high_cost_roles_from_stack_priors(
    stack_priors_path: Path = DEFAULT_STACK_PRIORS_PATH,
) -> frozenset[str] | None:
    """Return generated live roles that require high-cost approval gates."""
    try:
        from src.registry.stack_priors import live_stack_safe_non_stream_roles
    except Exception:
        return None

    try:
        return live_stack_safe_non_stream_roles(
            stack_priors_path,
            min_mem_gb=_HIGH_COST_MIN_MODEL_MEM_GB,
        )
    except Exception:
        log.debug("Failed to derive high-cost approval roles from stack priors", exc_info=True)
        return None


def high_cost_roles(
    stack_priors_path: Path = DEFAULT_STACK_PRIORS_PATH,
) -> frozenset[str]:
    """Return high-cost approval roles, falling back only in degraded mode."""
    global _STACK_PRIOR_HIGH_COST_ROLES_CACHE
    if stack_priors_path == DEFAULT_STACK_PRIORS_PATH:
        if _STACK_PRIOR_HIGH_COST_ROLES_CACHE is None:
            _STACK_PRIOR_HIGH_COST_ROLES_CACHE = high_cost_roles_from_stack_priors(
                stack_priors_path
            )
        derived = _STACK_PRIOR_HIGH_COST_ROLES_CACHE
    else:
        derived = high_cost_roles_from_stack_priors(stack_priors_path)
    return derived if derived is not None else _DEGRADED_HIGH_COST_ROLES


def should_halt(
    from_role: str,
    to_role: str,
    *,
    stack_priors_path: Path = DEFAULT_STACK_PRIORS_PATH,
) -> HaltReason | None:
    """Determine if a transition should trigger an approval gate.

    Args:
        from_role: Current role.
        to_role: Target role.
        stack_priors_path: Generated stack-priors artifact for live high-cost roles.

    Returns:
        HaltReason if approval needed, None otherwise.
    """
    from src.features import features as _get_features

    if not _get_features().approval_gates:
        return None

    # Escalation boundary (tier crossing)
    from src.roles import get_tier

    from_tier = get_tier(from_role)
    to_tier = get_tier(to_role)
    if from_tier != to_tier:
        return HaltReason.ESCALATION

    # High-cost invocation
    if to_role in high_cost_roles(stack_priors_path):
        return HaltReason.HIGH_COST

    return None


def request_approval_for_escalation(
    ctx: Any,  # GraphRunContext
    from_role: str,
    to_role: str,
    reason: str,
) -> ApprovalDecision:
    """Request approval for an escalation, returning the decision.

    If no callback or feature disabled, auto-approves.

    Args:
        ctx: GraphRunContext with state and deps.
        from_role: Current role.
        to_role: Target role after escalation.
        reason: Human-readable reason for escalation.

    Returns:
        ApprovalDecision.APPROVE or ApprovalDecision.REJECT.
    """
    halt_reason = should_halt(from_role, to_role)
    if halt_reason is None:
        return ApprovalDecision.APPROVE

    callback = getattr(ctx.deps, "approval_callback", None)
    if callback is None:
        return ApprovalDecision.APPROVE

    # Build resume token if available
    resume_token = ""
    from src.features import features as _get_features

    if _get_features().resume_tokens:
        try:
            from src.graph.resume_token import ResumeToken

            token = ResumeToken.from_state(ctx.state, type(ctx).__name__)
            resume_token = token.encode()
        except Exception:
            pass

    halt = HaltState(
        reason=halt_reason,
        from_role=from_role,
        to_role=to_role,
        resume_token=resume_token,
        description=reason,
    )

    ctx.state.pending_approval = halt
    decision = callback.request_approval(halt)
    ctx.state.pending_approval = None

    log.info(
        "Approval gate: %s → %s: %s (decision: %s)",
        from_role, to_role, halt_reason, decision,
    )
    return decision


# ---------------------------------------------------------------------------
# Generalized interrupts (LangGraph pre-migration)
# ---------------------------------------------------------------------------


@runtime_checkable
class InterruptCondition(Protocol):
    """Pluggable condition that can halt graph execution before REPL."""

    def should_interrupt(self, state: Any, artifacts: dict) -> str | None:
        """Return description string if should interrupt, None otherwise."""
        ...


def check_interrupt_conditions(
    conditions: list[InterruptCondition],
    state: Any,
    artifacts: dict,
) -> str | None:
    """Check all conditions, return first trigger description or None."""
    for cond in conditions:
        try:
            desc = cond.should_interrupt(state, artifacts)
            if desc is not None:
                return desc
        except Exception as exc:
            log.debug("Interrupt condition check failed: %s", exc)
    return None


def request_approval_for_interrupt(
    ctx: Any,
    description: str,
) -> ApprovalDecision:
    """Request approval for an interrupt condition."""
    callback = getattr(ctx.deps, "approval_callback", None)
    if callback is None:
        return ApprovalDecision.APPROVE

    resume_token = ""
    from src.features import features as _get_features

    if _get_features().resume_tokens:
        try:
            from src.graph.resume_token import ResumeToken

            token = ResumeToken.from_state(ctx.state, "interrupt")
            resume_token = token.encode()
        except Exception:
            pass

    halt = HaltState(
        reason=HaltReason.INTERRUPT_CONDITION,
        description=description,
        resume_token=resume_token,
        from_role=str(getattr(ctx.state, "current_role", "")),
    )
    ctx.state.pending_approval = halt
    decision = callback.request_approval(halt)
    ctx.state.pending_approval = None
    log.info("Interrupt gate: %s (decision: %s)", description, decision)
    return decision
