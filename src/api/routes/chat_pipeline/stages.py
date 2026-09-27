"""Pipeline stages: shared helpers, mock mode, react mode, error annotation.

Large stage functions have been split into focused modules:
    vision_stage.py      — Stage 6: _execute_vision()
    delegation_stage.py  — Stage 7: _execute_delegated()
    proactive_stage.py   — Stage 7.5: _execute_proactive(), _parse_plan_steps()
    direct_stage.py      — Stage 9: _execute_direct()

This file retains:
    _quality_escalate()  — Shared quality-check-then-escalate helper
    _execute_mock()      — Stage 4: Mock mode
    _execute_react()     — Stage 8: ReAct tool loop
    _annotate_error()    — Error annotation
"""

from __future__ import annotations

import logging
import re
import time

from src.api.models import ChatRequest, ChatResponse
from src.api.services.memrl import score_completed_task
from src.features import features
from src.llm_primitives import LLMPrimitives
from src.roles import Role

from src.repl_environment import REPLEnvironment
from src.api.routes.chat_review import (
    _detect_output_quality_issue,
)
from src.api.routes.chat_utils import (
    RoutingResult,
    _truncate_looped_answer,
)
from src.api.routes.chat_pipeline.telemetry import (
    llm_completion_meta,
    work_completion_meta,
)
from src.api.structured_logging import task_extra

log = logging.getLogger(__name__)


# ── Shared helpers ──────────────────────────────────────────────────────


def _quality_escalate(
    answer: str,
    prompt: str,
    primitives: "LLMPrimitives",
    initial_role,
    *,
    allow_escalation: bool = True,
    escalation_role: Role = Role.CODER_ESCALATION,
) -> tuple[str, "Role"]:
    """Detect quality issue and escalate to coder_escalation if needed.

    Returns (answer, role) — either unchanged or with escalated answer and role.
    ``escalation_role`` is the role that re-answers; /chat always uses the
    default. /v1 (TE-1, ``x_escalation=architect_general``) pins the consultant
    here so the trigger is /chat's while the target is the one the caller named.
    """
    if not allow_escalation:
        return answer, initial_role
    if not (answer and not answer.startswith("[ERROR") and features().generation_monitor):
        return answer, initial_role
    quality_issue = _detect_output_quality_issue(answer)
    if not quality_issue:
        return answer, initial_role
    try:
        escalated = primitives.llm_call(
            prompt, role=str(escalation_role), n_tokens=2048, skip_suffix=True,
        )
        if escalated.strip():
            return escalated.strip(), escalation_role
    except Exception as exc:
        log.debug("Quality escalation failed: %s", exc)
    return answer, initial_role


# ── Stage 4: Mock mode ──────────────────────────────────────────────────


def _execute_mock(
    request: ChatRequest,
    routing: RoutingResult,
    state,
    start_time: float,
) -> ChatResponse:
    """Handle mock mode requests with simulated response."""
    turns = 1
    answer = f"[MOCK] Processed prompt: {request.prompt[:100]}..."

    if request.context:
        answer += f" (with {len(request.context)} chars of context)"

    elapsed = time.perf_counter() - start_time
    state.increment_request(mock_mode=True, turns=turns)

    if state.progress_logger:
        state.progress_logger.log_task_completed(
            task_id=routing.task_id,
            success=True,
            details=f"Mock response in {elapsed:.3f}s",
            completion_meta={
                "producer_role": "mock",
                "delegation_lineage": ["mock"],
                "final_answer_role": "mock",
            },
        )
        score_completed_task(
            state,
            routing.task_id,
            force_role=request.force_role,
            real_mode=request.real_mode,
        )

    return ChatResponse(
        answer=answer,
        turns=turns,
        tokens_used=0,
        # ETR-6: set explicitly, like vision_stage's unavailable response, so the two
        # "no decode" shapes serialize identically even under exclude_unset.
        tokens_generated=0,
        elapsed_seconds=elapsed,
        mock_mode=True,
        real_mode=False,
        cache_stats=None,
        mode="mock",
    )


# ── Stage 8: ReAct mode ─────────────────────────────────────────────────


def _execute_react(
    request: ChatRequest,
    routing: RoutingResult,
    primitives: LLMPrimitives,
    state,
    start_time: float,
    initial_role,
) -> ChatResponse | None:
    """Handle ReAct tool loop mode. Returns None if react fails (fall through)."""
    if not (request.real_mode):
        return None

    log.info(
        "ReAct mode for %s (prompt: %d chars)",
        initial_role,
        len(request.prompt),
        extra=task_extra(
            task_id=routing.task_id,
            role=str(initial_role),
            stage="execute",
            mode="react",
            prompt_len=len(request.prompt),
        ),
    )

    react_tools_used = 0
    react_tools_called: list[str] = []
    react_tool_timings: list[dict] = []
    _react_registry = state.tool_registry if hasattr(state, "tool_registry") else None
    try:
        # Derive tool context for cascading policy (WS-3: deny web for reasoning domains)
        from src.tool_policy import NO_WEB_TASK_TYPES

        _task_type = str(routing.task_ir.get("task_type", "chat"))
        _tool_context = {"no_web": True} if _task_type in NO_WEB_TASK_TYPES else {}

        # React mode unified into REPL with structured_mode=True
        react_repl = REPLEnvironment(
            context=f"{request.prompt}\n\nContext:\n{request.context}" if request.context else request.prompt,
            llm_primitives=primitives,
            tool_registry=_react_registry,
            role=str(initial_role),
            structured_mode=True,
            tool_context=_tool_context,
        )
        from src.prompt_builders import build_root_lm_prompt, extract_code_from_response, auto_wrap_final
        answer = ""
        react_last_output = ""
        react_last_error = ""
        for _turn in range(5):
            repl_state = react_repl.get_state()
            react_prompt = build_root_lm_prompt(
                state=repl_state,
                original_prompt=request.prompt,
                last_output=react_last_output,
                last_error=react_last_error,
                turn=_turn,
            )
            _final_re = re.compile(
                r"""FINAL\(\s*(?:'{3}(.+?)'{3}|"{3}(.+?)"{3}|["'](.+?)["']|(\S+?))\s*\)""",
                re.DOTALL,
            )
            _call_re = re.compile(
                r'CALL\s*\(\s*"[^"]+"\s*(?:,\s*\w+\s*=\s*(?:"[^"]*"|\'[^\']*\'|\d+|True|False|None))*\s*\)',
            )
            primitives._early_stop_check = lambda text: (
                bool(_final_re.search(text)) or bool(_call_re.search(text))
            )
            try:
                code = primitives.llm_call(
                    react_prompt,
                    role=str(initial_role),
                )
            finally:
                primitives._early_stop_check = None
            code = extract_code_from_response(code)
            code = auto_wrap_final(code)
            # Comment-only guard: skip execution if code is all comments
            if all(
                not ln.strip() or ln.strip().startswith("#")
                for ln in code.split("\n")
            ):
                react_last_error = (
                    "Your output was all comments — no executable code ran. "
                    "Write Python code that computes the answer and call FINAL(answer)."
                )
                react_last_output = ""
                continue
            result = react_repl.execute(code)
            if result.is_final:
                answer = result.final_answer or ""
                break
            react_last_output = result.output or ""
            react_last_error = result.error or ""
        else:
            answer = react_repl.get_state()
        react_tools_used = react_repl._tool_invocations
        # REQUEST-LOCAL invocation records captured at the _invoke_tool chokepoint
        # (src/repl_environment/context.py). Deliberately NOT
        # react_repl.tool_registry.get_invocation_log(): that registry is
        # process-global (one instance per API process, src/api/__init__.py) and its
        # ring is shared by every concurrently served request, so reading it here
        # wrote OTHER requests' tool calls into THIS request's durable record.
        # The records expose the same attribute interface as ToolInvocation
        # (tool_name/elapsed_ms/success/result), so the body below is unchanged.
        react_invocation_log = list(getattr(react_repl, "_invoked_tools", None) or [])
        if react_invocation_log:
            react_tool_timings = []
            for inv in react_invocation_log:
                _output_tokens = 0
                if inv.success and inv.result is not None:
                    if isinstance(inv.result, str):
                        _output_tokens = len(inv.result) // 4
                    elif isinstance(inv.result, dict):
                        _output_tokens = len(str(inv.result)) // 4
                react_tool_timings.append(
                    {"tool_name": inv.tool_name, "elapsed_ms": inv.elapsed_ms,
                     "success": inv.success, "output_tokens": _output_tokens}
                )
            react_tools_called = [inv.tool_name for inv in react_invocation_log]
        react_tools_used = max(
            react_tools_used,
            len(react_tools_called),
            len(react_tool_timings),
        )
        answer = answer.strip()
    except Exception as e:
        log.warning(
            "ReAct mode failed (%s), falling back to direct",
            e,
            extra=task_extra(
                task_id=routing.task_id,
                role=str(initial_role),
                stage="execute",
                mode="react",
                error_type=type(e).__name__,
            ),
        )
        return None

    if not answer:
        return None

    # Post-processing: truncation, quality check
    answer = _truncate_looped_answer(answer, request.prompt)

    answer, initial_role = _quality_escalate(
        answer,
        request.prompt,
        primitives,
        initial_role,
        allow_escalation=not bool(request.force_role),
    )

    elapsed = time.perf_counter() - start_time
    state.increment_request(mock_mode=False, turns=1)
    if state.progress_logger:
        state.progress_logger.log_task_completed(
            task_id=routing.task_id,
            success=True,
            details=f"ReAct mode ({initial_role}), {elapsed:.3f}s",
            completion_meta={
                "producer_role": str(initial_role),
                "delegation_lineage": [str(initial_role)],
                "final_answer_role": str(initial_role),
                **llm_completion_meta(primitives),
                # M-11a2b: answer + the tools the ReAct loop actually ran.
                **work_completion_meta(answer=answer, tool_calls=react_tool_timings),
            },
        )
        score_completed_task(
            state,
            routing.task_id,
            force_role=request.force_role,
            real_mode=request.real_mode,
        )

    cache_stats = primitives.get_cache_stats() if primitives._backends else None
    # Sum tool output tokens for effective throughput
    react_tool_output_tokens = sum(
        int(t.get("output_tokens", 0) or 0)
        for t in react_tool_timings
        if isinstance(t, dict)
    )
    return ChatResponse(
        answer=answer,
        turns=1,
        tokens_used=primitives.total_tokens_generated,
        elapsed_seconds=elapsed,
        mock_mode=False,
        real_mode=True,
        cache_stats=cache_stats,
        routed_to=str(initial_role),
        role_history=[str(initial_role)],
        routing_strategy="react",
        mode="react",
        tokens_generated=primitives.total_tokens_generated,
        tool_output_tokens=react_tool_output_tokens,
        formalization_applied=routing.formalization_applied,
        tools_used=react_tools_used,
        tools_called=react_tools_called,
        tool_timings=react_tool_timings,
        prompt_eval_ms=primitives.total_prompt_eval_ms,
        generation_ms=primitives.total_generation_ms,
        predicted_tps=primitives._last_predicted_tps,
        http_overhead_ms=primitives.total_http_overhead_ms,
        skills_retrieved=len(routing.skill_ids),
        skill_ids=routing.skill_ids,
    )


# ── Error annotation ────────────────────────────────────────────────────


def _annotate_error(response: ChatResponse) -> ChatResponse:
    """Detect error patterns in answer and set error_code/error_detail.

    Phase 1b KV cache bug mitigation: instead of silently returning HTTP 200
    with an error string, we set structured error fields that the endpoint
    wrapper can use to return appropriate HTTP status codes.
    """
    if not response.answer:
        return response

    answer = response.answer

    # Timeout / backend failure patterns
    if answer.startswith("[ERROR:") and (
        "timed out" in answer.lower() or "timeout" in answer.lower()
    ):
        response.error_code = 504
        response.error_detail = answer
    elif answer.startswith("[ERROR:") and (
        "circuit open" in answer.lower() or "unavailable" in answer.lower()
    ):
        response.error_code = 503
        response.error_detail = answer
    elif answer.startswith("[ERROR:") and (
        "admission" in answer.lower() or "queue full" in answer.lower()
    ):
        response.error_code = 429
        response.error_detail = answer
    elif answer.startswith("[ERROR:") and (
        "backend" in answer.lower() or "failed" in answer.lower()
    ):
        response.error_code = 502
        response.error_detail = answer
    elif answer.startswith("[ERROR:"):
        response.error_code = 500
        response.error_detail = answer
    elif answer.startswith("[FAILED:"):
        response.error_code = 500
        response.error_detail = answer

    return response


def _attach_budget_diagnostics(
    response: ChatResponse,
    primitives: LLMPrimitives | None,
) -> ChatResponse:
    """Attach request budget/deadline telemetry to API response diagnostics."""
    if primitives is None:
        return response
    try:
        response.budget_diagnostics = primitives.get_budget_diagnostics()
    except Exception:
        response.budget_diagnostics = {}
    return response
