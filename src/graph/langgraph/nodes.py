"""LangGraph node functions — framework-agnostic wrappers around _execute_turn().

Each function mirrors a pydantic_graph node class from ``src.graph.nodes``.
They receive OrchestratorState, call the same ``_execute_turn()`` helper,
mutate state fields, and return a dict update with ``next_node`` set to
control conditional edges.

Node names used in edges:
    "frontdoor", "worker", "coder", "coder_escalation",
    "ingest", "architect", "__end__"
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

from langchain_core.runnables import RunnableConfig

from src.escalation import ErrorCategory
from src.roles import Role
from src.graph.state import TaskDeps, TaskState
from src.graph.decision_gates import _admit_early_abort_escalation
from src.graph.helpers import (
    MAX_CONSECUTIVE_NUDGES,
    _add_evidence,
    _build_think_harder_config,
    _check_budget_exceeded,
    _classify_error,
    _execute_turn,
    _is_infra_failure,
    _log_escalation,
    _make_end_result,
    _record_failure,
    _record_mitigation,
    _repeated_nudge_failure,
    _rescue_from_last_output,
    _resolve_answer,
    _should_escalate,
    _should_retry,
    _should_think_harder,
)
from src.graph.langgraph.state import (
    lg_to_task_state,
    snapshot_append_lengths,
    state_update_delta,
    task_state_to_lg,
)

log = logging.getLogger(__name__)

# Sentinel: when next_node is this value, the graph should terminate
END = "__end__"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_ctx(
    state_dict: dict[str, Any], config: RunnableConfig,
) -> tuple[Any, TaskState, TaskDeps, dict[str, int]]:
    """Build a pydantic_graph-compatible context from LangGraph state + config.

    Returns a (ctx, task_state, deps, snapshot_lengths) 4-tuple. The ``ctx``
    is a SimpleNamespace that duck-types ``GraphRunContext[TaskState, TaskDeps]``.
    ``snapshot_lengths`` captures the lengths of append-reducer list fields at
    entry so that ``_state_update`` can return only new elements (deltas).
    """
    deps: TaskDeps = config.get("configurable", {}).get("deps", TaskDeps())

    # Snapshot append-field lengths BEFORE reconstructing TaskState
    snap = snapshot_append_lengths(state_dict)

    # Reconstruct TaskState from LG state dict
    task_state = TaskState()
    lg_to_task_state(state_dict, task_state)

    # Restore config constants from configurable
    configurable = config.get("configurable", {})
    for attr in (
        "think_harder_min_expected_roi",
        "think_harder_min_samples",
        "think_harder_cooldown_turns",
        "think_harder_ema_alpha",
        "think_harder_min_marginal_utility",
    ):
        if attr in configurable:
            setattr(task_state, attr, configurable[attr])

    # Build ctx that matches Ctx = GraphRunContext[TaskState, TaskDeps]
    ctx = SimpleNamespace(state=task_state, deps=deps)
    return ctx, task_state, deps, snap


def _state_update(
    task_state: TaskState, next_node: str, snap: dict[str, int],
) -> dict[str, Any]:
    """Convert mutated TaskState back to a LangGraph state update dict.

    Trims append-reducer fields to deltas (new elements only) using the
    snapshot captured at node entry, preventing LangGraph's operator.add
    reducer from duplicating existing elements.
    """
    update = task_state_to_lg(task_state)
    state_update_delta(update, snap)
    update["next_node"] = next_node
    return update


def _record_escalation_role(state: TaskState, role: Role) -> None:
    """Record a role transition."""
    state.record_role(role)


def _handle_end(
    ctx, answer: str, success: bool, task_state: TaskState, snap: dict[str, int],
) -> dict[str, Any]:
    """Create end-of-graph state update.

    Calls ``_make_end_result`` for side effects (evidence recording, workspace
    update), then builds the terminal state dict with delta-trimmed append fields.
    """
    # _make_end_result produces side effects we need
    _make_end_result(ctx, answer, success)
    update = task_state_to_lg(task_state)
    state_update_delta(update, snap)
    update["next_node"] = END
    # Store result in state for extraction by the compiled graph
    update["_result"] = {
        "answer": answer,
        "success": success,
        "role_history": list(task_state.role_history),
        "turns": task_state.turns,
        "delegation_events": list(task_state.delegation_events),
    }
    return update


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


async def frontdoor_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    """Frontdoor node — entry point for unclassified requests.

    Escalates to coder_escalation on failure.
    """
    ctx, task_state, deps, snap = _build_ctx(state, config)

    if task_state.turns >= task_state.max_turns:
        rescued = _rescue_from_last_output(task_state.last_output)
        if rescued:
            return _handle_end(ctx, rescued, True, task_state, snap)
        return _handle_end(ctx, f"[Max turns ({task_state.max_turns}) reached]", False, task_state, snap)

    budget_reason = _check_budget_exceeded(ctx)
    if budget_reason:
        rescued = _rescue_from_last_output(task_state.last_output)
        if rescued:
            return _handle_end(ctx, rescued, True, task_state, snap)
        return _handle_end(ctx, f"[{budget_reason}]", False, task_state, snap)

    output, error, is_final, artifacts = await _execute_turn(ctx, Role.FRONTDOOR)
    task_state.artifacts.update(artifacts)

    if is_final:
        tool_outputs = artifacts.get("_tool_outputs", [])
        answer = _resolve_answer(output, tool_outputs)
        return _handle_end(ctx, answer, True, task_state, snap)

    if error:
        if _is_infra_failure(artifacts):
            return _handle_end(ctx, error, False, task_state, snap)
        task_state.consecutive_failures += 1
        task_state.last_error = error
        task_state.last_output = output
        error_cat = _classify_error(error)
        _record_failure(ctx, error_cat, error)

        if error_cat == ErrorCategory.EARLY_ABORT and _admit_early_abort_escalation(ctx):
            task_state.escalation_count += 1
            task_state.consecutive_failures = 0
            from_role = str(task_state.current_role)
            task_state.record_role(Role.CODER_ESCALATION)
            _log_escalation(ctx, from_role, str(Role.CODER_ESCALATION), f"Early abort: {error[:100]}")
            return _state_update(task_state, "coder_escalation", snap)

        if _should_think_harder(ctx, error_cat):
            task_state.think_harder_attempted = True
            task_state.think_harder_config = _build_think_harder_config(task_state)
            return _state_update(task_state, "frontdoor", snap)

        if _should_escalate(ctx, error_cat, Role.CODER_ESCALATION):
            if task_state.think_harder_attempted:
                task_state.think_harder_succeeded = False
            task_state.escalation_count += 1
            task_state.consecutive_failures = 0
            task_state.think_harder_attempted = False
            from_role = str(task_state.current_role)
            task_state.record_role(Role.CODER_ESCALATION)
            _log_escalation(ctx, from_role, str(Role.CODER_ESCALATION),
                            f"Escalating after {task_state.consecutive_failures} failures")
            return _state_update(task_state, "coder_escalation", snap)

        if _should_retry(ctx, error_cat):
            return _state_update(task_state, "frontdoor", snap)

        return _handle_end(ctx, f"[FAILED: {error}]", False, task_state, snap)

    task_state.last_output = output
    if task_state.think_harder_attempted and task_state.think_harder_succeeded is None:
        task_state.think_harder_succeeded = True
    nudge = artifacts.get("_nudge")
    if nudge:
        task_state.consecutive_nudges += 1
        task_state.last_error = nudge
        if task_state.consecutive_nudges >= MAX_CONSECUTIVE_NUDGES:
            task_state.consecutive_failures += 1
            task_state.consecutive_nudges = 0
            error_cat = _classify_error(nudge)
            if _should_escalate(ctx, error_cat, Role.CODER_ESCALATION):
                task_state.escalation_count += 1
                task_state.consecutive_failures = 0
                from_role = str(task_state.current_role)
                task_state.record_role(Role.CODER_ESCALATION)
                _log_escalation(ctx, from_role, str(Role.CODER_ESCALATION),
                                f"Escalating after {MAX_CONSECUTIVE_NUDGES} repeated nudges")
                return _state_update(task_state, "coder_escalation", snap)
            return _handle_end(
                ctx,
                _repeated_nudge_failure(task_state.current_role, nudge),
                False,
                task_state,
                snap,
            )
    else:
        task_state.consecutive_failures = 0
        task_state.consecutive_nudges = 0
        task_state.last_error = ""
    return _state_update(task_state, "frontdoor", snap)


async def worker_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    """Worker node for all WORKER_* roles. Escalates to coder_escalation."""
    ctx, task_state, deps, snap = _build_ctx(state, config)

    if task_state.turns >= task_state.max_turns:
        rescued = _rescue_from_last_output(task_state.last_output)
        if rescued:
            return _handle_end(ctx, rescued, True, task_state, snap)
        return _handle_end(ctx, f"[Max turns ({task_state.max_turns}) reached]", False, task_state, snap)

    budget_reason = _check_budget_exceeded(ctx)
    if budget_reason:
        rescued = _rescue_from_last_output(task_state.last_output)
        if rescued:
            return _handle_end(ctx, rescued, True, task_state, snap)
        return _handle_end(ctx, f"[{budget_reason}]", False, task_state, snap)

    output, error, is_final, artifacts = await _execute_turn(ctx, task_state.current_role)
    task_state.artifacts.update(artifacts)

    if is_final:
        tool_outputs = artifacts.get("_tool_outputs", [])
        answer = _resolve_answer(output, tool_outputs)
        return _handle_end(ctx, answer, True, task_state, snap)

    if error:
        if _is_infra_failure(artifacts):
            return _handle_end(ctx, error, False, task_state, snap)
        task_state.consecutive_failures += 1
        task_state.last_error = error
        task_state.last_output = output
        error_cat = _classify_error(error)
        _record_failure(ctx, error_cat, error)

        if error_cat == ErrorCategory.EARLY_ABORT and _admit_early_abort_escalation(ctx):
            task_state.escalation_count += 1
            task_state.consecutive_failures = 0
            from_role = str(task_state.current_role)
            task_state.record_role(Role.CODER_ESCALATION)
            _log_escalation(ctx, from_role, str(Role.CODER_ESCALATION), f"Early abort: {error[:100]}")
            return _state_update(task_state, "coder_escalation", snap)

        if _should_think_harder(ctx, error_cat):
            task_state.think_harder_attempted = True
            task_state.think_harder_config = _build_think_harder_config(task_state)
            return _state_update(task_state, "worker", snap)

        if _should_escalate(ctx, error_cat, Role.CODER_ESCALATION):
            if task_state.think_harder_attempted:
                task_state.think_harder_succeeded = False
            task_state.escalation_count += 1
            task_state.consecutive_failures = 0
            task_state.think_harder_attempted = False
            from_role = str(task_state.current_role)
            task_state.record_role(Role.CODER_ESCALATION)
            _log_escalation(ctx, from_role, str(Role.CODER_ESCALATION),
                            f"Escalating after {task_state.consecutive_failures} failures")
            return _state_update(task_state, "coder_escalation", snap)

        if _should_retry(ctx, error_cat):
            return _state_update(task_state, "worker", snap)

        return _handle_end(ctx, f"[FAILED: {error}]", False, task_state, snap)

    task_state.consecutive_failures = 0
    task_state.last_output = output
    if task_state.think_harder_attempted and task_state.think_harder_succeeded is None:
        task_state.think_harder_succeeded = True
    nudge = artifacts.get("_nudge")
    if nudge:
        task_state.consecutive_nudges += 1
        task_state.last_error = nudge
        if task_state.consecutive_nudges >= MAX_CONSECUTIVE_NUDGES:
            task_state.consecutive_failures += 1
            task_state.consecutive_nudges = 0
            return _handle_end(
                ctx,
                _repeated_nudge_failure(task_state.current_role, nudge),
                False,
                task_state,
                snap,
            )
    else:
        task_state.consecutive_failures = 0
        task_state.consecutive_nudges = 0
        task_state.last_error = ""
    return _state_update(task_state, "worker", snap)


async def coder_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    """Coder node for THINKING_REASONING. Escalates to architect."""
    ctx, task_state, deps, snap = _build_ctx(state, config)

    if task_state.turns >= task_state.max_turns:
        rescued = _rescue_from_last_output(task_state.last_output)
        if rescued:
            return _handle_end(ctx, rescued, True, task_state, snap)
        return _handle_end(ctx, f"[Max turns ({task_state.max_turns}) reached]", False, task_state, snap)

    budget_reason = _check_budget_exceeded(ctx)
    if budget_reason:
        rescued = _rescue_from_last_output(task_state.last_output)
        if rescued:
            return _handle_end(ctx, rescued, True, task_state, snap)
        return _handle_end(ctx, f"[{budget_reason}]", False, task_state, snap)

    output, error, is_final, artifacts = await _execute_turn(ctx, task_state.current_role)
    task_state.artifacts.update(artifacts)

    # Model-initiated escalation
    if artifacts.get("_escalation_requested"):
        artifacts.pop("_escalation_target", None)
        reason = artifacts.pop("_escalation_reason", "Model requested")
        artifacts.pop("_escalation_requested", None)
        task_state.escalation_count += 1
        task_state.consecutive_failures = 0
        from_role = str(task_state.current_role)
        _record_escalation_role(task_state, Role.ARCHITECT_GENERAL)
        _log_escalation(ctx, from_role, str(Role.ARCHITECT_GENERAL), f"Model-initiated: {reason}")
        return _state_update(task_state, "architect", snap)

    if is_final:
        tool_outputs = artifacts.get("_tool_outputs", [])
        answer = _resolve_answer(output, tool_outputs)
        if task_state.escalation_count > 0:
            _record_mitigation(ctx, task_state.role_history[-2] if len(task_state.role_history) > 1 else "unknown", str(task_state.current_role))
        return _handle_end(ctx, answer, True, task_state, snap)

    if error:
        if _is_infra_failure(artifacts):
            return _handle_end(ctx, error, False, task_state, snap)
        task_state.consecutive_failures += 1
        task_state.last_error = error
        task_state.last_output = output
        error_cat = _classify_error(error)
        _record_failure(ctx, error_cat, error)

        if error_cat == ErrorCategory.EARLY_ABORT and _admit_early_abort_escalation(ctx):
            task_state.escalation_count += 1
            task_state.consecutive_failures = 0
            from_role = str(task_state.current_role)
            _record_escalation_role(task_state, Role.ARCHITECT_GENERAL)
            _log_escalation(ctx, from_role, str(Role.ARCHITECT_GENERAL), f"Early abort: {error[:100]}")
            return _state_update(task_state, "architect", snap)

        if _should_think_harder(ctx, error_cat):
            task_state.think_harder_attempted = True
            task_state.think_harder_config = _build_think_harder_config(task_state)
            return _state_update(task_state, "coder", snap)

        if _should_escalate(ctx, error_cat, Role.ARCHITECT_GENERAL):
            if task_state.think_harder_attempted:
                task_state.think_harder_succeeded = False
            task_state.escalation_count += 1
            task_state.consecutive_failures = 0
            task_state.think_harder_attempted = False
            from_role = str(task_state.current_role)
            _record_escalation_role(task_state, Role.ARCHITECT_GENERAL)
            _log_escalation(ctx, from_role, str(Role.ARCHITECT_GENERAL),
                            f"Escalating after {task_state.consecutive_failures} failures")
            return _state_update(task_state, "architect", snap)

        if _should_retry(ctx, error_cat):
            return _state_update(task_state, "coder", snap)

        return _handle_end(ctx, f"[FAILED: {error}]", False, task_state, snap)

    task_state.consecutive_failures = 0
    task_state.last_output = output
    if task_state.think_harder_attempted and task_state.think_harder_succeeded is None:
        task_state.think_harder_succeeded = True
    nudge = artifacts.get("_nudge")
    if nudge:
        task_state.consecutive_nudges += 1
        task_state.last_error = nudge
        if task_state.consecutive_nudges >= MAX_CONSECUTIVE_NUDGES:
            task_state.consecutive_failures += 1
            task_state.consecutive_nudges = 0
            return _handle_end(
                ctx,
                _repeated_nudge_failure(task_state.current_role, nudge),
                False,
                task_state,
                snap,
            )
    else:
        task_state.consecutive_failures = 0
        task_state.consecutive_nudges = 0
        task_state.last_error = ""
    return _state_update(task_state, "coder", snap)


async def coder_escalation_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    """Coder escalation node. Escalates to live architect_general."""
    ctx, task_state, deps, snap = _build_ctx(state, config)

    if task_state.turns >= task_state.max_turns:
        rescued = _rescue_from_last_output(task_state.last_output)
        if rescued:
            return _handle_end(ctx, rescued, True, task_state, snap)
        return _handle_end(ctx, f"[Max turns ({task_state.max_turns}) reached]", False, task_state, snap)

    budget_reason = _check_budget_exceeded(ctx)
    if budget_reason:
        rescued = _rescue_from_last_output(task_state.last_output)
        if rescued:
            return _handle_end(ctx, rescued, True, task_state, snap)
        return _handle_end(ctx, f"[{budget_reason}]", False, task_state, snap)

    output, error, is_final, artifacts = await _execute_turn(ctx, Role.CODER_ESCALATION)
    task_state.artifacts.update(artifacts)

    if is_final:
        tool_outputs = artifacts.get("_tool_outputs", [])
        answer = _resolve_answer(output, tool_outputs)
        if task_state.escalation_count > 0:
            _record_mitigation(ctx, task_state.role_history[-2] if len(task_state.role_history) > 1 else "unknown", str(task_state.current_role))
        return _handle_end(ctx, answer, True, task_state, snap)

    if error:
        if _is_infra_failure(artifacts):
            return _handle_end(ctx, error, False, task_state, snap)
        task_state.consecutive_failures += 1
        task_state.last_error = error
        task_state.last_output = output
        error_cat = _classify_error(error)
        _record_failure(ctx, error_cat, error)

        if error_cat == ErrorCategory.EARLY_ABORT and _admit_early_abort_escalation(ctx):
            task_state.escalation_count += 1
            task_state.consecutive_failures = 0
            from_role = str(task_state.current_role)
            _record_escalation_role(task_state, Role.ARCHITECT_GENERAL)
            _log_escalation(ctx, from_role, str(Role.ARCHITECT_GENERAL), f"Early abort: {error[:100]}")
            return _state_update(task_state, "architect", snap)

        if _should_think_harder(ctx, error_cat):
            task_state.think_harder_attempted = True
            task_state.think_harder_config = _build_think_harder_config(task_state)
            return _state_update(task_state, "coder_escalation", snap)

        if _should_escalate(ctx, error_cat, Role.ARCHITECT_GENERAL):
            if task_state.think_harder_attempted:
                task_state.think_harder_succeeded = False
            task_state.escalation_count += 1
            task_state.consecutive_failures = 0
            task_state.think_harder_attempted = False
            from_role = str(task_state.current_role)
            _record_escalation_role(task_state, Role.ARCHITECT_GENERAL)
            _log_escalation(ctx, from_role, str(Role.ARCHITECT_GENERAL),
                            f"Escalating after {task_state.consecutive_failures} failures")
            return _state_update(task_state, "architect", snap)

        if _should_retry(ctx, error_cat):
            return _state_update(task_state, "coder_escalation", snap)

        return _handle_end(ctx, f"[FAILED: {error}]", False, task_state, snap)

    task_state.consecutive_failures = 0
    task_state.last_output = output
    if task_state.think_harder_attempted and task_state.think_harder_succeeded is None:
        task_state.think_harder_succeeded = True
    nudge = artifacts.get("_nudge")
    if nudge:
        task_state.consecutive_nudges += 1
        task_state.last_error = nudge
        if task_state.consecutive_nudges >= MAX_CONSECUTIVE_NUDGES:
            task_state.consecutive_failures += 1
            task_state.consecutive_nudges = 0
            return _handle_end(
                ctx,
                _repeated_nudge_failure(task_state.current_role, nudge),
                False,
                task_state,
                snap,
            )
    else:
        task_state.consecutive_failures = 0
        task_state.consecutive_nudges = 0
        task_state.last_error = ""
    return _state_update(task_state, "coder_escalation", snap)


async def ingest_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    """Ingest node for INGEST_LONG_CONTEXT. Escalates to architect."""
    ctx, task_state, deps, snap = _build_ctx(state, config)

    if task_state.turns >= task_state.max_turns:
        rescued = _rescue_from_last_output(task_state.last_output)
        if rescued:
            return _handle_end(ctx, rescued, True, task_state, snap)
        return _handle_end(ctx, f"[Max turns ({task_state.max_turns}) reached]", False, task_state, snap)

    budget_reason = _check_budget_exceeded(ctx)
    if budget_reason:
        rescued = _rescue_from_last_output(task_state.last_output)
        if rescued:
            return _handle_end(ctx, rescued, True, task_state, snap)
        return _handle_end(ctx, f"[{budget_reason}]", False, task_state, snap)

    output, error, is_final, artifacts = await _execute_turn(ctx, Role.INGEST_LONG_CONTEXT)
    task_state.artifacts.update(artifacts)

    if is_final:
        tool_outputs = artifacts.get("_tool_outputs", [])
        answer = _resolve_answer(output, tool_outputs)
        return _handle_end(ctx, answer, True, task_state, snap)

    if error:
        if _is_infra_failure(artifacts):
            return _handle_end(ctx, error, False, task_state, snap)
        task_state.consecutive_failures += 1
        task_state.last_error = error
        task_state.last_output = output
        error_cat = _classify_error(error)
        _record_failure(ctx, error_cat, error)

        if error_cat == ErrorCategory.EARLY_ABORT and _admit_early_abort_escalation(ctx):
            task_state.escalation_count += 1
            task_state.consecutive_failures = 0
            from_role = str(task_state.current_role)
            _record_escalation_role(task_state, Role.ARCHITECT_GENERAL)
            _log_escalation(ctx, from_role, str(Role.ARCHITECT_GENERAL), f"Early abort: {error[:100]}")
            return _state_update(task_state, "architect", snap)

        if _should_think_harder(ctx, error_cat):
            task_state.think_harder_attempted = True
            task_state.think_harder_config = _build_think_harder_config(task_state)
            return _state_update(task_state, "ingest", snap)

        if _should_escalate(ctx, error_cat, Role.ARCHITECT_GENERAL):
            if task_state.think_harder_attempted:
                task_state.think_harder_succeeded = False
            task_state.escalation_count += 1
            task_state.consecutive_failures = 0
            task_state.think_harder_attempted = False
            from_role = str(task_state.current_role)
            _record_escalation_role(task_state, Role.ARCHITECT_GENERAL)
            _log_escalation(ctx, from_role, str(Role.ARCHITECT_GENERAL),
                            f"Escalating after {task_state.consecutive_failures} failures")
            return _state_update(task_state, "architect", snap)

        if _should_retry(ctx, error_cat):
            return _state_update(task_state, "ingest", snap)

        return _handle_end(ctx, f"[FAILED: {error}]", False, task_state, snap)

    task_state.consecutive_failures = 0
    task_state.last_output = output
    if task_state.think_harder_attempted and task_state.think_harder_succeeded is None:
        task_state.think_harder_succeeded = True
    nudge = artifacts.get("_nudge")
    if nudge:
        task_state.consecutive_nudges += 1
        task_state.last_error = nudge
        if task_state.consecutive_nudges >= MAX_CONSECUTIVE_NUDGES:
            task_state.consecutive_failures += 1
            task_state.consecutive_nudges = 0
            return _handle_end(
                ctx,
                _repeated_nudge_failure(task_state.current_role, nudge),
                False,
                task_state,
                snap,
            )
    else:
        task_state.consecutive_failures = 0
        task_state.consecutive_nudges = 0
        task_state.last_error = ""
    return _state_update(task_state, "ingest", snap)


async def architect_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    """Architect node — terminal. No further escalation."""
    ctx, task_state, deps, snap = _build_ctx(state, config)

    if task_state.turns >= task_state.max_turns:
        rescued = _rescue_from_last_output(task_state.last_output)
        if rescued:
            return _handle_end(ctx, rescued, True, task_state, snap)
        return _handle_end(ctx, f"[Max turns ({task_state.max_turns}) reached]", False, task_state, snap)

    budget_reason = _check_budget_exceeded(ctx)
    if budget_reason:
        rescued = _rescue_from_last_output(task_state.last_output)
        if rescued:
            return _handle_end(ctx, rescued, True, task_state, snap)
        return _handle_end(ctx, f"[{budget_reason}]", False, task_state, snap)

    output, error, is_final, artifacts = await _execute_turn(ctx, Role.ARCHITECT_GENERAL)
    task_state.artifacts.update(artifacts)

    if is_final:
        tool_outputs = artifacts.get("_tool_outputs", [])
        answer = _resolve_answer(output, tool_outputs)
        if task_state.escalation_count > 0:
            _record_mitigation(ctx, task_state.role_history[-2] if len(task_state.role_history) > 1 else "unknown", str(task_state.current_role))
        return _handle_end(ctx, answer, True, task_state, snap)

    if error:
        if _is_infra_failure(artifacts):
            return _handle_end(ctx, error, False, task_state, snap)
        task_state.consecutive_failures += 1
        task_state.last_error = error
        task_state.last_output = output
        error_cat = _classify_error(error)
        _record_failure(ctx, error_cat, error)

        if _should_think_harder(ctx, error_cat):
            task_state.think_harder_attempted = True
            task_state.think_harder_config = _build_think_harder_config(task_state)
            return _state_update(task_state, "architect", snap)

        if _should_retry(ctx, error_cat):
            return _state_update(task_state, "architect", snap)

        _add_evidence(ctx, "explore_fallback", -0.3)
        return _handle_end(ctx, f"[FAILED: Terminal role {task_state.current_role}: {error}]", False, task_state, snap)

    task_state.consecutive_failures = 0
    task_state.last_output = output
    if task_state.think_harder_attempted and task_state.think_harder_succeeded is None:
        task_state.think_harder_succeeded = True
    nudge = artifacts.get("_nudge")
    if nudge:
        task_state.consecutive_nudges += 1
        task_state.last_error = nudge
        if task_state.consecutive_nudges >= MAX_CONSECUTIVE_NUDGES:
            task_state.consecutive_failures += 1
            task_state.consecutive_nudges = 0
            return _handle_end(
                ctx,
                _repeated_nudge_failure(task_state.current_role, nudge),
                False,
                task_state,
                snap,
            )
    else:
        task_state.consecutive_failures = 0
        task_state.consecutive_nudges = 0
        task_state.last_error = ""
    return _state_update(task_state, "architect", snap)


# ---------------------------------------------------------------------------
# Node name -> role mapping (for select_start_node equivalent)
# ---------------------------------------------------------------------------

# Every LIVE routable role must appear here. ``select_start_lg_node`` degrades an
# unmapped role to "frontdoor", which is correct for RETIRED roles (thinking_reasoning
# has no weights on disk) but silently wrong for a live one: the request starts on the
# frontdoor node instead of the node that dispatches its backend.
#
# 2026-08-01 W1 cutover additions — both were live, registry-declared, routable roles
# that the cutover added to the Role enum without adding here:
#   * architect_critic — then the terminal escalation rung (122B, :8074). Unmapped,
#     the whole boosted-reasoning ladder started at the frontdoor node.
#     2026-09-27 ARCHITECT SWAP: architect_critic is now the MI210 27B (:8083) and
#     architect_general (Flash-Next, :8074) is terminal. NOTE: "architect" runs
#     Role.ARCHITECT_GENERAL, so a request STARTED as architect_critic executes on
#     architect_general's process here (pre-existing; RI-21 left open).
#   * vision_escalation — alias on worker_vision's :8086 MI210 process.
# tests/unit/test_langgraph_migration.py derives this table's completeness from the
# live stack priors so the next role addition cannot repeat the omission.
ROLE_TO_LG_NODE: dict[str, str] = {
    "frontdoor": "frontdoor",
    "worker_general": "worker",
    "worker_math": "worker",
    "worker_summarize": "worker",
    "worker_vision": "worker",
    "vision_escalation": "worker",
    "toolrunner": "worker",
    "coder_escalation": "coder_escalation",
    "ingest_long_context": "ingest",
    "architect_general": "architect",
    "architect_critic": "architect",
}


def select_start_lg_node(role: str) -> str:
    """Map a role string to the LangGraph node name."""
    resolved = Role.from_string(role)
    if resolved is not None:
        return ROLE_TO_LG_NODE.get(str(resolved), "frontdoor")
    return ROLE_TO_LG_NODE.get(role, "frontdoor")
