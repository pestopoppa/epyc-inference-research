"""Orchestration graph nodes — typed escalation flow.

Each node class maps to one or more orchestration roles. The ``run()``
return-type Union encodes which transitions are valid, enforced at type
check time.  Shared helpers live in ``src.graph.helpers``.

Bug fixes included in this migration:
- ``state.escalation_count`` is incremented on every escalation.
- ``deps.failure_graph.record_failure()`` is called on every error.
- ``deps.hypothesis_graph.add_evidence()`` is called on task outcomes.
- Hardcoded ``EscalationPolicy()`` fallbacks are eliminated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Union

from pydantic_graph import BaseNode, End

from src.escalation import ErrorCategory
from src.features import features as _get_features
from src.roles import Role

from src.graph.state import (
    TaskDeps,
    TaskResult,
    TaskState,
)
from src.graph.decision_gates import _admit_early_abort_escalation
from src.graph.langgraph.state import (
    APPEND_FIELDS,
    lg_to_task_state,
    task_state_to_lg,
)

# Import all helpers from the extracted module
from src.graph.helpers import (  # noqa: F401 — re-exported for backward compat
    Ctx,
    MAX_CONSECUTIVE_NUDGES,
    _add_evidence,
    _check_approval_gate,
    _check_budget_exceeded,
    _classify_error,
    _build_think_harder_config,
    _detect_role_cycle,
    _execute_turn,
    _extract_final_from_raw,
    _extract_prose_answer,
    _is_comment_only,
    _is_infra_failure,
    _log_escalation,
    _make_end_result,
    _maybe_compact_context,
    _record_failure,
    _record_mitigation,
    _repeated_nudge_failure,
    _rescue_from_last_output,
    _resolve_answer,
    _should_escalate,
    _should_retry,
    _should_think_harder,
    _timeout_skip,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LangGraph Phase 3: per-node dispatch helper
# ---------------------------------------------------------------------------

# Node name → (LG node function, pydantic_graph return type mapping)
# Lazy-imported to avoid circular imports at module level.
_LG_NODE_MAP: dict | None = None


def _get_lg_node_map():
    """Lazy-load LangGraph node functions to avoid circular imports."""
    global _LG_NODE_MAP
    if _LG_NODE_MAP is None:
        from src.graph.langgraph.nodes import (
            frontdoor_node,
            worker_node,
            coder_node,
            coder_escalation_node,
            ingest_node,
            architect_node,
        )
        _LG_NODE_MAP = {
            "frontdoor": frontdoor_node,
            "worker": worker_node,
            "coder": coder_node,
            "coder_escalation": coder_escalation_node,
            "ingest": ingest_node,
            "architect": architect_node,
        }
    return _LG_NODE_MAP


# Maps next_node string → pydantic_graph node class (populated after class defs)
_NEXT_NODE_TO_PG: dict[str, type] = {}


async def _run_via_langgraph(ctx, node_name: str):
    """Dispatch a single node execution to the LangGraph backend.

    Converts pydantic_graph ctx to LangGraph state, calls the LG node function,
    then maps the result back to a pydantic_graph return value (node instance or End).

    This allows per-node migration: the pydantic_graph outer loop remains in control
    while individual node logic can be swapped to LangGraph functions.
    """
    lg_func = _get_lg_node_map()[node_name]

    # Convert TaskState → LangGraph state dict
    lg_state = task_state_to_lg(ctx.state)

    # Save originals for append-reducer fields before LG call.
    # LG nodes return deltas (trimmed by state_update_delta), so we must
    # prepend the originals after lg_to_task_state overwrites them.
    _append_originals = {
        field: list(getattr(ctx.state, field, []))
        for field in APPEND_FIELDS
    }

    # Build RunnableConfig with deps in configurable
    config = {
        "configurable": {
            "deps": ctx.deps,
            "think_harder_min_expected_roi": getattr(ctx.state, "think_harder_min_expected_roi", 1.2),
            "think_harder_min_samples": getattr(ctx.state, "think_harder_min_samples", 3),
            "think_harder_cooldown_turns": getattr(ctx.state, "think_harder_cooldown_turns", 5),
            "think_harder_ema_alpha": getattr(ctx.state, "think_harder_ema_alpha", 0.3),
            "think_harder_min_marginal_utility": getattr(ctx.state, "think_harder_min_marginal_utility", 0.05),
        }
    }

    # Call LangGraph node function
    result = await lg_func(lg_state, config)

    # Apply state updates back to the pydantic_graph TaskState
    lg_to_task_state(result, ctx.state)

    # Restore append-reducer fields: prepend originals to the deltas
    # that lg_to_task_state set via setattr
    for field, original in _append_originals.items():
        delta = getattr(ctx.state, field, [])
        setattr(ctx.state, field, original + delta)

    # Map next_node → pydantic_graph return type
    next_node = result.get("next_node", "__end__")
    if next_node == "__end__":
        # Extract result from LG state
        lg_result = result.get("_result", {})
        return End(
            TaskResult(
                answer=lg_result.get("answer", ""),
                success=lg_result.get("success", False),
                role_history=lg_result.get("role_history", list(ctx.state.role_history)),
                turns=lg_result.get("turns", ctx.state.turns),
                delegation_events=lg_result.get("delegation_events", list(ctx.state.delegation_events)),
            )
        )

    # Return the appropriate pydantic_graph node instance
    pg_cls = _NEXT_NODE_TO_PG.get(next_node)
    if pg_cls is None:
        raise ValueError(f"Unknown next_node from LangGraph: {next_node!r}")
    return pg_cls()


def _record_escalation_role(state: TaskState, role: Role) -> None:
    """Record a role transition."""
    state.record_role(role)



# ── Node classes ───────────────────────────────────────────────────────


@dataclass
class FrontdoorNode(BaseNode[TaskState, TaskDeps, TaskResult]):
    """Entry node for unclassified/frontdoor requests.

    Escalates to CoderEscalationNode on failure (via escalation map: FRONTDOOR → CODER_ESCALATION).
    """

    async def run(
        self, ctx: Ctx
    ) -> Union["FrontdoorNode", "CoderEscalationNode", "WorkerNode", End[TaskResult]]:
        if _get_features().langgraph_frontdoor:
            return await _run_via_langgraph(ctx, "frontdoor")
        state = ctx.state

        if state.turns >= state.max_turns:
            rescued = _rescue_from_last_output(state.last_output)
            if rescued:
                log.info("Max-turns rescue (frontdoor): %r", rescued[:100])
                return _make_end_result(ctx, rescued, True)
            return _make_end_result(
                ctx, f"[Max turns ({state.max_turns}) reached]", False
            )

        budget_reason = _check_budget_exceeded(ctx)
        if budget_reason:
            rescued = _rescue_from_last_output(state.last_output)
            if rescued:
                log.info("Budget rescue (frontdoor): %s — %r", budget_reason, rescued[:100])
                return _make_end_result(ctx, rescued, True)
            return _make_end_result(ctx, f"[{budget_reason}]", False)

        output, error, is_final, artifacts = await _execute_turn(ctx, Role.FRONTDOOR)
        state.artifacts.update(artifacts)

        if is_final:
            tool_outputs = artifacts.get("_tool_outputs", [])
            answer = _resolve_answer(output, tool_outputs)
            return _make_end_result(ctx, answer, True)

        if error:
            if _is_infra_failure(artifacts):
                return _make_end_result(ctx, error, False)
            state.consecutive_failures += 1
            state.last_error = error
            state.last_output = output
            error_cat = _classify_error(error)
            _record_failure(ctx, error_cat, error)

            if error_cat == ErrorCategory.EARLY_ABORT and _admit_early_abort_escalation(ctx):
                # NIB2-80: budget-gated. Immediate escalation (skipping the retry
                # precondition _should_escalate enforces for other categories) is
                # deliberate; the max_escalations bound still applies. When the
                # budget is exhausted, fall through to the same
                # think-harder / _should_escalate / _should_retry / fail chain
                # every sibling site uses when its own gate refuses.
                state.escalation_count += 1
                state.consecutive_failures = 0
                from_role = str(state.current_role)
                state.record_role(Role.CODER_ESCALATION)
                _log_escalation(ctx, from_role, str(Role.CODER_ESCALATION), f"Early abort: {error[:100]}")
                return CoderEscalationNode()

            # Think-harder: same model with CoT + 2x tokens before escalating
            if _should_think_harder(ctx, error_cat):
                state.think_harder_attempted = True
                state.think_harder_config = _build_think_harder_config(state)
                log.info("Think-harder triggered at %s (attempt before escalation)", state.current_role)
                return FrontdoorNode()

            if _should_escalate(ctx, error_cat, Role.CODER_ESCALATION):
                # If think-harder was attempted and we still escalate, it failed
                if state.think_harder_attempted:
                    state.think_harder_succeeded = False
                state.escalation_count += 1
                state.consecutive_failures = 0
                state.think_harder_attempted = False  # Reset for next role
                from_role = str(state.current_role)
                state.record_role(Role.CODER_ESCALATION)
                _log_escalation(
                    ctx, from_role, str(Role.CODER_ESCALATION),
                    f"Escalating after {state.consecutive_failures} failures",
                )
                return CoderEscalationNode()

            if _should_retry(ctx, error_cat):
                return FrontdoorNode()

            return _make_end_result(ctx, f"[FAILED: {error}]", False)

        state.last_output = output
        # If think-harder was attempted and we got a successful turn, it worked
        if state.think_harder_attempted and state.think_harder_succeeded is None:
            state.think_harder_succeeded = True
        nudge = artifacts.get("_nudge")
        if nudge:
            state.consecutive_nudges += 1
            state.last_error = nudge
            if state.consecutive_nudges >= MAX_CONSECUTIVE_NUDGES:
                log.warning("Max nudges (%d) reached at %s, promoting to error", state.consecutive_nudges, state.current_role)
                state.consecutive_failures += 1
                state.consecutive_nudges = 0
                # Treat promoted nudge like a regular error — check escalation
                error_cat = _classify_error(nudge)
                if _should_escalate(ctx, error_cat, Role.CODER_ESCALATION):
                    state.escalation_count += 1
                    state.consecutive_failures = 0
                    from_role = str(state.current_role)
                    state.record_role(Role.CODER_ESCALATION)
                    _log_escalation(
                        ctx, from_role, str(Role.CODER_ESCALATION),
                        f"Escalating after {MAX_CONSECUTIVE_NUDGES} repeated nudges",
                    )
                    return CoderEscalationNode()
                return _make_end_result(
                    ctx, _repeated_nudge_failure(state.current_role, nudge), False
                )
        else:
            state.consecutive_failures = 0
            state.consecutive_nudges = 0
            state.last_error = ""
        return FrontdoorNode()


@dataclass
class WorkerNode(BaseNode[TaskState, TaskDeps, TaskResult]):
    """Worker node for all WORKER_* roles.

    Escalates to CoderEscalationNode on failure.
    """

    async def run(
        self, ctx: Ctx
    ) -> Union["WorkerNode", "CoderEscalationNode", End[TaskResult]]:
        if _get_features().langgraph_worker:
            return await _run_via_langgraph(ctx, "worker")
        state = ctx.state

        if state.turns >= state.max_turns:
            rescued = _rescue_from_last_output(state.last_output)
            if rescued:
                log.info("Max-turns rescue (worker): %r", rescued[:100])
                return _make_end_result(ctx, rescued, True)
            return _make_end_result(
                ctx, f"[Max turns ({state.max_turns}) reached]", False
            )

        budget_reason = _check_budget_exceeded(ctx)
        if budget_reason:
            rescued = _rescue_from_last_output(state.last_output)
            if rescued:
                log.info("Budget rescue (worker): %s — %r", budget_reason, rescued[:100])
                return _make_end_result(ctx, rescued, True)
            return _make_end_result(ctx, f"[{budget_reason}]", False)

        output, error, is_final, artifacts = await _execute_turn(ctx, state.current_role)
        state.artifacts.update(artifacts)

        if is_final:
            tool_outputs = artifacts.get("_tool_outputs", [])
            answer = _resolve_answer(output, tool_outputs)
            return _make_end_result(ctx, answer, True)

        if error:
            if _is_infra_failure(artifacts):
                return _make_end_result(ctx, error, False)
            state.consecutive_failures += 1
            state.last_error = error
            state.last_output = output
            error_cat = _classify_error(error)
            _record_failure(ctx, error_cat, error)

            if error_cat == ErrorCategory.EARLY_ABORT and _admit_early_abort_escalation(ctx):
                # NIB2-80: budget-gated — see FrontdoorNode for rationale.
                state.escalation_count += 1
                state.consecutive_failures = 0
                from_role = str(state.current_role)
                state.record_role(Role.CODER_ESCALATION)
                _log_escalation(ctx, from_role, str(Role.CODER_ESCALATION), f"Early abort: {error[:100]}")
                return CoderEscalationNode()

            if _should_think_harder(ctx, error_cat):
                state.think_harder_attempted = True
                state.think_harder_config = _build_think_harder_config(state)
                log.info("Think-harder triggered at %s", state.current_role)
                return WorkerNode()

            if _should_escalate(ctx, error_cat, Role.CODER_ESCALATION):
                if state.think_harder_attempted:
                    state.think_harder_succeeded = False
                state.escalation_count += 1
                state.consecutive_failures = 0
                state.think_harder_attempted = False
                from_role = str(state.current_role)
                state.record_role(Role.CODER_ESCALATION)
                _log_escalation(
                    ctx, from_role, str(Role.CODER_ESCALATION),
                    f"Escalating after {state.consecutive_failures} failures",
                )
                return CoderEscalationNode()

            if _should_retry(ctx, error_cat):
                return WorkerNode()

            return _make_end_result(ctx, f"[FAILED: {error}]", False)

        state.consecutive_failures = 0
        state.last_output = output
        if state.think_harder_attempted and state.think_harder_succeeded is None:
            state.think_harder_succeeded = True
        nudge = artifacts.get("_nudge")
        if nudge:
            state.consecutive_nudges += 1
            state.last_error = nudge
            if state.consecutive_nudges >= MAX_CONSECUTIVE_NUDGES:
                log.warning("Max nudges (%d) reached at %s, promoting to error", state.consecutive_nudges, state.current_role)
                state.consecutive_failures += 1
                state.consecutive_nudges = 0
                return _make_end_result(
                    ctx, _repeated_nudge_failure(state.current_role, nudge), False
                )
        else:
            state.consecutive_failures = 0
            state.consecutive_nudges = 0
            state.last_error = ""
        return WorkerNode()


@dataclass
class CoderNode(BaseNode[TaskState, TaskDeps, TaskResult]):
    """Coder node for THINKING_REASONING role.

    Escalates to ArchitectNode on failure.
    """

    async def run(
        self, ctx: Ctx
    ) -> Union["CoderNode", "ArchitectNode", End[TaskResult]]:
        if _get_features().langgraph_coder:
            return await _run_via_langgraph(ctx, "coder")
        state = ctx.state

        if state.turns >= state.max_turns:
            rescued = _rescue_from_last_output(state.last_output)
            if rescued:
                log.info("Max-turns rescue (coder): %r", rescued[:100])
                return _make_end_result(ctx, rescued, True)
            return _make_end_result(
                ctx, f"[Max turns ({state.max_turns}) reached]", False
            )

        budget_reason = _check_budget_exceeded(ctx)
        if budget_reason:
            rescued = _rescue_from_last_output(state.last_output)
            if rescued:
                log.info("Budget rescue (coder): %s — %r", budget_reason, rescued[:100])
                return _make_end_result(ctx, rescued, True)
            return _make_end_result(ctx, f"[{budget_reason}]", False)

        output, error, is_final, artifacts = await _execute_turn(ctx, state.current_role)
        state.artifacts.update(artifacts)

        # Check model-initiated escalation
        if artifacts.get("_escalation_requested"):
            artifacts.pop("_escalation_target", None)
            reason = artifacts.pop("_escalation_reason", "Model requested")
            artifacts.pop("_escalation_requested", None)

            state.escalation_count += 1
            state.consecutive_failures = 0
            from_role = str(state.current_role)
            _record_escalation_role(state, Role.ARCHITECT_GENERAL)
            _log_escalation(ctx, from_role, str(Role.ARCHITECT_GENERAL), f"Model-initiated: {reason}")
            return ArchitectNode()

        if is_final:
            tool_outputs = artifacts.get("_tool_outputs", [])
            answer = _resolve_answer(output, tool_outputs)
            # Record mitigation if we got here via escalation
            if state.escalation_count > 0:
                _record_mitigation(ctx, state.role_history[-2] if len(state.role_history) > 1 else "unknown", str(state.current_role))
            return _make_end_result(ctx, answer, True)

        if error:
            if _is_infra_failure(artifacts):
                return _make_end_result(ctx, error, False)
            state.consecutive_failures += 1
            state.last_error = error
            state.last_output = output
            error_cat = _classify_error(error)
            _record_failure(ctx, error_cat, error)

            if error_cat == ErrorCategory.EARLY_ABORT and _admit_early_abort_escalation(ctx):
                # NIB2-80: budget-gated — see FrontdoorNode for rationale.
                state.escalation_count += 1
                state.consecutive_failures = 0
                from_role = str(state.current_role)
                _record_escalation_role(state, Role.ARCHITECT_GENERAL)
                _log_escalation(ctx, from_role, str(Role.ARCHITECT_GENERAL), f"Early abort: {error[:100]}")
                return ArchitectNode()

            if _should_think_harder(ctx, error_cat):
                state.think_harder_attempted = True
                state.think_harder_config = _build_think_harder_config(state)
                log.info("Think-harder triggered at %s", state.current_role)
                return CoderNode()

            if _should_escalate(ctx, error_cat, Role.ARCHITECT_GENERAL):
                if state.think_harder_attempted:
                    state.think_harder_succeeded = False
                state.escalation_count += 1
                state.consecutive_failures = 0
                state.think_harder_attempted = False
                from_role = str(state.current_role)
                _record_escalation_role(state, Role.ARCHITECT_GENERAL)
                _log_escalation(
                    ctx, from_role, str(Role.ARCHITECT_GENERAL),
                    f"Escalating after {state.consecutive_failures} failures",
                )
                return ArchitectNode()

            if _should_retry(ctx, error_cat):
                return CoderNode()

            return _make_end_result(ctx, f"[FAILED: {error}]", False)

        state.consecutive_failures = 0
        state.last_output = output
        if state.think_harder_attempted and state.think_harder_succeeded is None:
            state.think_harder_succeeded = True
        nudge = artifacts.get("_nudge")
        if nudge:
            state.consecutive_nudges += 1
            state.last_error = nudge
            if state.consecutive_nudges >= MAX_CONSECUTIVE_NUDGES:
                log.warning("Max nudges (%d) reached at %s, promoting to error", state.consecutive_nudges, state.current_role)
                state.consecutive_failures += 1
                state.consecutive_nudges = 0
                return _make_end_result(
                    ctx, _repeated_nudge_failure(state.current_role, nudge), False
                )
        else:
            state.consecutive_failures = 0
            state.consecutive_nudges = 0
            state.last_error = ""
        return CoderNode()


@dataclass
class CoderEscalationNode(BaseNode[TaskState, TaskDeps, TaskResult]):
    """Escalation coder node for CODER_ESCALATION role.

    Escalates to ArchitectNode. The retired architect-coding role is
    retained as a compatibility alias, but its servers are intentionally dead.
    """

    async def run(
        self, ctx: Ctx
    ) -> Union["CoderEscalationNode", "ArchitectNode", End[TaskResult]]:
        if _get_features().langgraph_coder_escalation:
            return await _run_via_langgraph(ctx, "coder_escalation")
        state = ctx.state

        if state.turns >= state.max_turns:
            rescued = _rescue_from_last_output(state.last_output)
            if rescued:
                log.info("Max-turns rescue (coder_escalation): %r", rescued[:100])
                return _make_end_result(ctx, rescued, True)
            return _make_end_result(
                ctx, f"[Max turns ({state.max_turns}) reached]", False
            )

        budget_reason = _check_budget_exceeded(ctx)
        if budget_reason:
            rescued = _rescue_from_last_output(state.last_output)
            if rescued:
                log.info("Budget rescue (coder_escalation): %s — %r", budget_reason, rescued[:100])
                return _make_end_result(ctx, rescued, True)
            return _make_end_result(ctx, f"[{budget_reason}]", False)

        output, error, is_final, artifacts = await _execute_turn(ctx, Role.CODER_ESCALATION)
        state.artifacts.update(artifacts)

        if is_final:
            tool_outputs = artifacts.get("_tool_outputs", [])
            answer = _resolve_answer(output, tool_outputs)
            if state.escalation_count > 0:
                _record_mitigation(ctx, state.role_history[-2] if len(state.role_history) > 1 else "unknown", str(state.current_role))
            return _make_end_result(ctx, answer, True)

        if error:
            if _is_infra_failure(artifacts):
                return _make_end_result(ctx, error, False)
            state.consecutive_failures += 1
            state.last_error = error
            state.last_output = output
            error_cat = _classify_error(error)
            _record_failure(ctx, error_cat, error)

            if error_cat == ErrorCategory.EARLY_ABORT and _admit_early_abort_escalation(ctx):
                # NIB2-80: budget-gated — see FrontdoorNode for rationale.
                state.escalation_count += 1
                state.consecutive_failures = 0
                from_role = str(state.current_role)
                _record_escalation_role(state, Role.ARCHITECT_GENERAL)
                _log_escalation(ctx, from_role, str(Role.ARCHITECT_GENERAL), f"Early abort: {error[:100]}")
                return ArchitectNode()

            if _should_think_harder(ctx, error_cat):
                state.think_harder_attempted = True
                state.think_harder_config = _build_think_harder_config(state)
                log.info("Think-harder triggered at %s", state.current_role)
                return CoderEscalationNode()

            if _should_escalate(ctx, error_cat, Role.ARCHITECT_GENERAL):
                if state.think_harder_attempted:
                    state.think_harder_succeeded = False
                state.escalation_count += 1
                state.consecutive_failures = 0
                state.think_harder_attempted = False
                from_role = str(state.current_role)
                _record_escalation_role(state, Role.ARCHITECT_GENERAL)
                _log_escalation(
                    ctx, from_role, str(Role.ARCHITECT_GENERAL),
                    f"Escalating after {state.consecutive_failures} failures",
                )
                return ArchitectNode()

            if _should_retry(ctx, error_cat):
                return CoderEscalationNode()

            return _make_end_result(ctx, f"[FAILED: {error}]", False)

        state.consecutive_failures = 0
        state.last_output = output
        if state.think_harder_attempted and state.think_harder_succeeded is None:
            state.think_harder_succeeded = True
        nudge = artifacts.get("_nudge")
        if nudge:
            state.consecutive_nudges += 1
            state.last_error = nudge
            if state.consecutive_nudges >= MAX_CONSECUTIVE_NUDGES:
                log.warning("Max nudges (%d) reached at %s, promoting to error", state.consecutive_nudges, state.current_role)
                state.consecutive_failures += 1
                state.consecutive_nudges = 0
                return _make_end_result(
                    ctx, _repeated_nudge_failure(state.current_role, nudge), False
                )
        else:
            state.consecutive_failures = 0
            state.consecutive_nudges = 0
            state.last_error = ""
        return CoderEscalationNode()


@dataclass
class IngestNode(BaseNode[TaskState, TaskDeps, TaskResult]):
    """Ingest node for INGEST_LONG_CONTEXT role (SSM path, no spec).

    Escalates to ArchitectNode on failure.
    """

    async def run(
        self, ctx: Ctx
    ) -> Union["IngestNode", "ArchitectNode", End[TaskResult]]:
        if _get_features().langgraph_ingest:
            return await _run_via_langgraph(ctx, "ingest")
        state = ctx.state

        if state.turns >= state.max_turns:
            rescued = _rescue_from_last_output(state.last_output)
            if rescued:
                log.info("Max-turns rescue (ingest): %r", rescued[:100])
                return _make_end_result(ctx, rescued, True)
            return _make_end_result(
                ctx, f"[Max turns ({state.max_turns}) reached]", False
            )

        budget_reason = _check_budget_exceeded(ctx)
        if budget_reason:
            rescued = _rescue_from_last_output(state.last_output)
            if rescued:
                log.info("Budget rescue (ingest): %s — %r", budget_reason, rescued[:100])
                return _make_end_result(ctx, rescued, True)
            return _make_end_result(ctx, f"[{budget_reason}]", False)

        output, error, is_final, artifacts = await _execute_turn(
            ctx, Role.INGEST_LONG_CONTEXT
        )
        state.artifacts.update(artifacts)

        if is_final:
            tool_outputs = artifacts.get("_tool_outputs", [])
            answer = _resolve_answer(output, tool_outputs)
            return _make_end_result(ctx, answer, True)

        if error:
            if _is_infra_failure(artifacts):
                return _make_end_result(ctx, error, False)
            state.consecutive_failures += 1
            state.last_error = error
            state.last_output = output
            error_cat = _classify_error(error)
            _record_failure(ctx, error_cat, error)

            if error_cat == ErrorCategory.EARLY_ABORT and _admit_early_abort_escalation(ctx):
                # NIB2-80: budget-gated — see FrontdoorNode for rationale.
                state.escalation_count += 1
                state.consecutive_failures = 0
                from_role = str(state.current_role)
                _record_escalation_role(state, Role.ARCHITECT_GENERAL)
                _log_escalation(ctx, from_role, str(Role.ARCHITECT_GENERAL), f"Early abort: {error[:100]}")
                return ArchitectNode()

            if _should_think_harder(ctx, error_cat):
                state.think_harder_attempted = True
                state.think_harder_config = _build_think_harder_config(state)
                log.info("Think-harder triggered at %s", state.current_role)
                return IngestNode()

            if _should_escalate(ctx, error_cat, Role.ARCHITECT_GENERAL):
                if state.think_harder_attempted:
                    state.think_harder_succeeded = False
                state.escalation_count += 1
                state.consecutive_failures = 0
                state.think_harder_attempted = False
                from_role = str(state.current_role)
                _record_escalation_role(state, Role.ARCHITECT_GENERAL)
                _log_escalation(
                    ctx, from_role, str(Role.ARCHITECT_GENERAL),
                    f"Escalating after {state.consecutive_failures} failures",
                )
                return ArchitectNode()

            if _should_retry(ctx, error_cat):
                return IngestNode()

            return _make_end_result(ctx, f"[FAILED: {error}]", False)

        state.consecutive_failures = 0
        state.last_output = output
        if state.think_harder_attempted and state.think_harder_succeeded is None:
            state.think_harder_succeeded = True
        nudge = artifacts.get("_nudge")
        if nudge:
            state.consecutive_nudges += 1
            state.last_error = nudge
            if state.consecutive_nudges >= MAX_CONSECUTIVE_NUDGES:
                log.warning("Max nudges (%d) reached at %s, promoting to error", state.consecutive_nudges, state.current_role)
                state.consecutive_failures += 1
                state.consecutive_nudges = 0
                return _make_end_result(
                    ctx, _repeated_nudge_failure(state.current_role, nudge), False
                )
        else:
            state.consecutive_failures = 0
            state.consecutive_nudges = 0
            state.last_error = ""
        return IngestNode()


@dataclass
class ArchitectNode(BaseNode[TaskState, TaskDeps, TaskResult]):
    """Architect node for ARCHITECT_GENERAL role.

    Terminal — no further escalation. Falls back to EXPLORE on repeated failure.
    """

    async def run(
        self, ctx: Ctx
    ) -> Union["ArchitectNode", End[TaskResult]]:
        if _get_features().langgraph_architect:
            return await _run_via_langgraph(ctx, "architect")
        state = ctx.state

        if state.turns >= state.max_turns:
            rescued = _rescue_from_last_output(state.last_output)
            if rescued:
                log.info("Max-turns rescue (architect): %r", rescued[:100])
                return _make_end_result(ctx, rescued, True)
            return _make_end_result(
                ctx, f"[Max turns ({state.max_turns}) reached]", False
            )

        budget_reason = _check_budget_exceeded(ctx)
        if budget_reason:
            rescued = _rescue_from_last_output(state.last_output)
            if rescued:
                log.info("Budget rescue (architect): %s — %r", budget_reason, rescued[:100])
                return _make_end_result(ctx, rescued, True)
            return _make_end_result(ctx, f"[{budget_reason}]", False)

        output, error, is_final, artifacts = await _execute_turn(
            ctx, Role.ARCHITECT_GENERAL
        )
        state.artifacts.update(artifacts)

        if is_final:
            tool_outputs = artifacts.get("_tool_outputs", [])
            answer = _resolve_answer(output, tool_outputs)
            if state.escalation_count > 0:
                _record_mitigation(ctx, state.role_history[-2] if len(state.role_history) > 1 else "unknown", str(state.current_role))
            return _make_end_result(ctx, answer, True)

        if error:
            if _is_infra_failure(artifacts):
                return _make_end_result(ctx, error, False)
            state.consecutive_failures += 1
            state.last_error = error
            state.last_output = output
            error_cat = _classify_error(error)
            _record_failure(ctx, error_cat, error)

            if _should_think_harder(ctx, error_cat):
                state.think_harder_attempted = True
                state.think_harder_config = _build_think_harder_config(state)
                log.info("Think-harder triggered at %s (terminal role)", state.current_role)
                return ArchitectNode()

            if _should_retry(ctx, error_cat):
                return ArchitectNode()

            # Terminal — EXPLORE fallback
            _add_evidence(ctx, "explore_fallback", -0.3)
            return _make_end_result(
                ctx,
                f"[FAILED: Terminal role {state.current_role}: {error}]",
                False,
            )

        state.consecutive_failures = 0
        state.last_output = output
        if state.think_harder_attempted and state.think_harder_succeeded is None:
            state.think_harder_succeeded = True
        nudge = artifacts.get("_nudge")
        if nudge:
            state.consecutive_nudges += 1
            state.last_error = nudge
            if state.consecutive_nudges >= MAX_CONSECUTIVE_NUDGES:
                log.warning("Max nudges (%d) reached at %s, promoting to error", state.consecutive_nudges, state.current_role)
                state.consecutive_failures += 1
                state.consecutive_nudges = 0
                return _make_end_result(
                    ctx, _repeated_nudge_failure(state.current_role, nudge), False
                )
        else:
            state.consecutive_failures = 0
            state.consecutive_nudges = 0
            state.last_error = ""
        return ArchitectNode()

# ── LangGraph Phase 3: populate next_node → pydantic_graph class mapping ──

_NEXT_NODE_TO_PG.update({
    "frontdoor": FrontdoorNode,
    "worker": WorkerNode,
    "coder": CoderNode,
    "coder_escalation": CoderEscalationNode,
    "ingest": IngestNode,
    "architect": ArchitectNode,
})


# ── Node selection helper ──────────────────────────────────────────────

# Maps initial roles to their starting node class.
_ROLE_TO_NODE: dict[Role, type] = {
    Role.FRONTDOOR: FrontdoorNode,
    Role.WORKER_GENERAL: WorkerNode,
    Role.WORKER_MATH: WorkerNode,
    Role.WORKER_SUMMARIZE: WorkerNode,
    Role.WORKER_VISION: WorkerNode,
    Role.TOOLRUNNER: WorkerNode,
    Role.THINKING_REASONING: CoderNode,
    Role.CODER_ESCALATION: CoderEscalationNode,
    Role.INGEST_LONG_CONTEXT: IngestNode,
    Role.ARCHITECT_GENERAL: ArchitectNode,
}


def select_start_node(role: Role | str) -> BaseNode:
    """Select the graph start node class for a given role."""
    if isinstance(role, str):
        role = Role.from_string(role) or Role.FRONTDOOR

    node_cls = _ROLE_TO_NODE.get(role, FrontdoorNode)
    return node_cls()
