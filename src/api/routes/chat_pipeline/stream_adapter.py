"""SSE streaming adapter — reuses pipeline stages for the streaming endpoint.

Provides generate_stream() which wraps pipeline stages 1-3 and 5 with SSE
event emission, eliminating the routing/preprocessing/init duplication between
_handle_chat() and chat_stream().

Gated behind features().unified_streaming flag.
"""

from __future__ import annotations

import logging
import time
from typing import AsyncGenerator

from src.api.models import ChatRequest
from src.api.services.memrl import failure_disposition_meta, score_completed_task
from src.escalation import (
    EscalationAction,
    EscalationContext,
    EscalationPolicy,
)
from src.llm_primitives import LLMPrimitives
from src.prompt_builders import (
    auto_wrap_final,
    build_escalation_prompt,
    build_root_lm_prompt,
    build_routing_context,
    classify_error,
    extract_code_from_response,
)
from src.prompt_builders.builder import build_corpus_context
from src.repl_environment import REPLEnvironment
from src.roles import Role
from src.runtime import routing_stage_timing
from src.sse_utils import (
    done_event,
    error_event,
    final_event,
    thinking_event,
    token_event,
    tool_end_event,
    tool_start_event,
    turn_end_event,
    turn_start_event,
)

from src.api.routes.chat_pipeline.routing import (
    _init_primitives,
    _plan_review_gate,
    _preprocess,
    _route_request,
)
from src.api.routes.chat_pipeline.telemetry import llm_completion_meta
from src.api.routes.chat_review import (
    _plan_review_abort_message,
    _plan_review_should_abort,
)
from src.api.routes.chat_utils import _resolve_answer

log = logging.getLogger(__name__)


# ── Mock-mode streaming ───────────────────────────────────────────────


async def _stream_mock(
    request: ChatRequest,
    routing,
    state,
    start_time: float,
) -> AsyncGenerator[dict, None]:
    """Yield SSE events for mock mode requests."""
    yield turn_start_event(turn=1, role=str(Role.FRONTDOOR))

    if request.thinking_budget > 0:
        for step in [
            "Analyzing the user's request...",
            f"Request type: {request.prompt[:30].split()[0] if request.prompt else 'unknown'}",
            "Determining appropriate response strategy...",
            "Preparing response...",
        ]:
            yield thinking_event(step)

    if request.permission_mode == "plan":
        analysis = f"[PLAN MODE] Would process: {request.prompt[:100]}..."
        yield token_event(analysis)
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        yield turn_end_event(tokens=len(analysis), elapsed_ms=elapsed_ms)
        if state.progress_logger:
            state.progress_logger.log_task_completed(
                routing.task_id,
                success=True,
                details="Plan mode",
                completion_meta={
                    "producer_role": "plan",
                    "delegation_lineage": ["plan"],
                    "final_answer_role": "plan",
                },
            )
            score_completed_task(
                state,
                routing.task_id,
                force_role=request.force_role,
                real_mode=request.real_mode,
            )
        yield done_event()
        return

    mock_response = f"[MOCK] Processed: {request.prompt[:50]}..."
    for char in mock_response:
        yield token_event(char)

    elapsed_ms = int((time.perf_counter() - start_time) * 1000)
    yield turn_end_event(tokens=len(mock_response), elapsed_ms=elapsed_ms)

    if state.progress_logger:
        state.progress_logger.log_task_completed(
            routing.task_id,
            success=True,
            details="Mock stream",
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
    yield done_event()


# ── Real-mode REPL streaming ──────────────────────────────────────────


async def _stream_repl(
    request: ChatRequest,
    routing,
    primitives: LLMPrimitives,
    state,
    start_time: float,
) -> AsyncGenerator[dict, None]:
    """Yield SSE events for real-mode REPL loop with escalation support."""
    task_id = routing.task_id

    combined_context = request.prompt
    if request.context:
        combined_context += f"\n\nContext:\n{request.context}"

    # Derive tool context for cascading policy (WS-3: deny web for reasoning domains)
    from src.tool_policy import NO_WEB_TASK_TYPES

    _task_type = str(routing.task_ir.get("task_type", "chat"))
    _tool_context = {"no_web": True} if _task_type in NO_WEB_TASK_TYPES else {}

    repl = REPLEnvironment(
        context=combined_context,
        llm_primitives=primitives,
        tool_registry=state.tool_registry,
        script_registry=state.script_registry,
        role=request.role or Role.FRONTDOOR,
        retriever=state.hybrid_router.retriever if state.hybrid_router else None,
        hybrid_router=state.hybrid_router,
        tool_context=_tool_context,
    )

    last_output = ""
    last_error = ""
    result = None
    prev_tool_count = 0

    current_role = request.role or Role.FRONTDOOR
    consecutive_failures = 0
    role_history = [current_role]
    escalation_prompt = ""

    for turn in range(request.max_turns):
        turn_start_time_inner = time.perf_counter()
        yield turn_start_event(turn=turn + 1, role=str(current_role))

        repl_state = repl.get_state()
        if escalation_prompt:
            root_prompt = escalation_prompt
            escalation_prompt = ""
        else:
            routing_ctx = ""
            corpus_ctx = ""
            if turn == 0:
                if state.hybrid_router:
                    routing_ctx = build_routing_context(
                        role=current_role,
                        hybrid_router=state.hybrid_router,
                        task_description=request.prompt,
                    )
                corpus_ctx = build_corpus_context(
                    role=current_role,
                    task_description=request.prompt,
                    task_id=task_id,
                    request_id=request.request_id,
                )
            root_prompt = build_root_lm_prompt(
                state=repl_state,
                original_prompt=request.prompt,
                last_output=last_output,
                last_error=last_error,
                turn=turn,
                routing_context=routing_ctx,
                corpus_context=corpus_ctx,
            )

        try:
            code = primitives.llm_call(root_prompt, role=current_role, n_tokens=1024)
        except Exception as e:
            if state.progress_logger:
                state.progress_logger.log_task_completed(
                    task_id,
                    success=False,
                    details=f"Root LM failed: {e}",
                    completion_meta={
                        "producer_role": str(current_role),
                        "delegation_lineage": [str(r) for r in role_history],
                        "final_answer_role": str(current_role),
                        **llm_completion_meta(primitives),
                        # The backend raised — the role never produced an answer
                        # to judge. A transport error is stamped and skipped; an
                        # application bug classifies as a task failure and is
                        # still scored.
                        **failure_disposition_meta(error=e),
                    },
                )
                score_completed_task(
                    state,
                    task_id,
                    force_role=request.force_role,
                    real_mode=request.real_mode,
                )
            yield error_event(f"Root LM call failed: {e}")
            yield done_event()
            return

        code = extract_code_from_response(code)
        code = auto_wrap_final(code)
        for line in code.split("\n"):
            yield token_event(line + "\n")

        result = repl.execute(code)

        # Emit tool events for any tool invocations in this turn.
        # REQUEST-LOCAL: repl._invoked_tools, not
        # repl.tool_registry.get_invocation_log() — the registry is process-global,
        # so the cursor-slice idiom below streamed OTHER concurrent requests' tool
        # events to this client (and, now that the registry ring is bounded,
        # eviction would make the cursor slice wrong). _invoked_tools is
        # append-only for the life of this one REPL, so the cursor is exact.
        inv_log = list(getattr(repl, "_invoked_tools", None) or [])
        for inv in inv_log[prev_tool_count:]:
            yield tool_start_event(inv.tool_name)
            yield tool_end_event(inv.tool_name, int(inv.elapsed_ms), inv.success)
        prev_tool_count = len(inv_log)

        # Model-initiated routing
        if repl.artifacts.get("_escalation_requested"):
            target = repl.artifacts.pop("_escalation_target", None)
            reason = repl.artifacts.pop("_escalation_reason", "Model requested")
            repl.artifacts.pop("_escalation_requested", None)

            new_role = None
            if target:
                resolved = Role.from_string(target)
                if resolved:
                    new_role = str(resolved)
            else:
                esc_ctx = EscalationContext(
                    current_role=current_role,
                    error_category="early_abort",
                    error_message=reason,
                    failure_count=1,
                    task_id=task_id,
                )
                esc_decision = EscalationPolicy().decide(esc_ctx)
                if esc_decision.should_escalate and esc_decision.target_role:
                    new_role = str(esc_decision.target_role)

            if new_role and new_role != current_role:
                turn_elapsed_ms = int(
                    (time.perf_counter() - turn_start_time_inner) * 1000
                )
                yield turn_end_event(tokens=len(code), elapsed_ms=turn_elapsed_ms)

                current_role = new_role
                role_history.append(current_role)
                consecutive_failures = 0
                escalation_prompt = build_escalation_prompt(
                    original_prompt=request.prompt,
                    state=repl_state,
                    failure_context=EscalationContext(
                        current_role=role_history[-2],
                        error_message=reason,
                        error_category="early_abort",
                        task_id=task_id,
                    ),
                    decision=EscalationPolicy().decide(
                        EscalationContext(
                            current_role=role_history[-2],
                            error_category="early_abort",
                            error_message=reason,
                            task_id=task_id,
                        )
                    ),
                )
                if state.progress_logger:
                    state.progress_logger.log_escalation(
                        task_id=task_id,
                        from_tier=role_history[-2],
                        to_tier=current_role,
                        reason=f"Model-initiated: {reason}",
                    )
                continue

        # Delegation logging
        if repl.artifacts.get("_delegations"):
            for deleg in repl.artifacts["_delegations"]:
                if state.progress_logger:
                    state.progress_logger.log_exploration(
                        task_id=task_id,
                        query=deleg.get("prompt_preview", ""),
                        strategy_used=f"delegate:{deleg.get('to_role', 'unknown')}",
                        success=deleg.get("success", False),
                    )
            repl.artifacts["_delegations"] = []

        turn_elapsed_ms = int((time.perf_counter() - turn_start_time_inner) * 1000)
        yield turn_end_event(tokens=len(code), elapsed_ms=turn_elapsed_ms)

        # Completion check
        if result.is_final:
            tool_outputs = repl.artifacts.get("_tool_outputs", [])
            stream_answer = _resolve_answer(result, tool_outputs=tool_outputs)

            yield final_event(stream_answer)
            break

        # Error handling with escalation
        if result.error:
            consecutive_failures += 1
            last_error = result.error
            last_output = result.output

            error_category = classify_error(result.error)
            esc_ctx = EscalationContext(
                current_role=current_role,
                error_message=result.error,
                error_category=error_category.value,
                failure_count=consecutive_failures,
                task_id=task_id,
            )
            decision = EscalationPolicy().decide(esc_ctx)

            if decision.should_think_harder and decision.config_override:
                # Same model, boosted config — prepend CoT, don't escalate
                cot_prefix = decision.config_override.get("cot_prefix", "")
                if cot_prefix:
                    escalation_prompt = cot_prefix + root_prompt
                log.info("Think-harder in stream: %s", decision.reason)
            elif decision.should_escalate and decision.target_role:
                current_role = str(decision.target_role)
                role_history.append(current_role)
                consecutive_failures = 0
                escalation_prompt = build_escalation_prompt(
                    original_prompt=request.prompt,
                    state=repl_state,
                    failure_context=esc_ctx,
                    decision=decision,
                )
                if state.progress_logger:
                    state.progress_logger.log_escalation(
                        task_id=task_id,
                        from_tier=role_history[-2],
                        to_tier=current_role,
                        reason=f"{decision.reason} (failures: {consecutive_failures})",
                    )
            elif decision.action == EscalationAction.FAIL:
                yield error_event(f"[FAILED: {decision.reason}]")
                break
        else:
            consecutive_failures = 0
            last_error = ""
            last_output = result.output

    # Log completion
    if state.progress_logger:
        success = result is not None and result.is_final
        role_info = (
            f", roles: {' -> '.join(str(r) for r in role_history)}"
            if len(role_history) > 1
            else ""
        )
        state.progress_logger.log_task_completed(
            task_id,
            success=success,
            details=f"Stream complete{role_info}",
            completion_meta={
                "producer_role": str(current_role),
                "delegation_lineage": [str(r) for r in role_history],
                "final_answer_role": str(current_role),
                **llm_completion_meta(primitives),
            },
        )
        score_completed_task(
            state,
            task_id,
            force_role=request.force_role,
            real_mode=request.real_mode,
        )
    yield done_event()


# ── Main entry point ──────────────────────────────────────────────────


async def generate_stream(
    request: ChatRequest,
    state,
) -> AsyncGenerator[dict, None]:
    """Yield SSE events using pipeline stages shared with _handle_chat().

    Reuses _route_request, _preprocess, _init_primitives, _plan_review_gate
    instead of reimplementing them inline. The REPL streaming loop remains
    separate because it must yield per-event (unlike the batch graph runner).
    """
    start_time = time.perf_counter()

    # Stage 1: Routing (shared with _handle_chat)
    routing = _route_request(
        request, state, routing_path=routing_stage_timing.PATH_UNIFIED_STREAM
    )

    # Stage 2: Preprocessing (shared with _handle_chat)
    _preprocess(request, state, routing)

    # Mock mode
    if routing.use_mock:
        async for event in _stream_mock(request, routing, state, start_time):
            yield event
        return

    # Stage 3: Backend init (shared with _handle_chat)
    try:
        primitives = _init_primitives(request, state)
    except Exception as e:
        if state.progress_logger:
            state.progress_logger.log_task_completed(
                routing.task_id,
                success=False,
                details=str(e),
                completion_meta={
                    "producer_role": "stream_init",
                    "delegation_lineage": ["stream_init"],
                    "final_answer_role": "stream_init",
                    # Backend init failed: no role ever ran, so there is no
                    # quality signal here at all.
                    **failure_disposition_meta(error=e),
                },
            )
            score_completed_task(
                state,
                routing.task_id,
                force_role=request.force_role,
                real_mode=request.real_mode,
            )
        yield error_event(str(e))
        yield done_event()
        return

    # Stage 5: Plan review gate (shared with _handle_chat)
    plan_review_result = _plan_review_gate(request, routing, primitives, state)
    if _plan_review_should_abort(plan_review_result):
        assert plan_review_result is not None
        answer = _plan_review_abort_message(plan_review_result)
        if state.progress_logger:
            state.progress_logger.log_task_completed(
                routing.task_id,
                success=False,
                details=answer,
                completion_meta={
                    "producer_role": "architect_general",
                    "delegation_lineage": ["architect_general"],
                    "final_answer_role": "architect_general",
                },
            )
            score_completed_task(
                state,
                routing.task_id,
                force_role=request.force_role,
                real_mode=request.real_mode,
            )
        yield error_event(answer)
        yield final_event(answer)
        yield done_event()
        return

    # Real-mode REPL streaming
    async for event in _stream_repl(request, routing, primitives, state, start_time):
        yield event
