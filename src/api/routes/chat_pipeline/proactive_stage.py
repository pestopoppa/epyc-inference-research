"""Pipeline stage 7.5: Proactive parallel delegation.

Decomposes COMPLEX tasks into parallel-executable steps via architect,
then delegates via ProactiveDelegator for wave-based execution.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import os
import re as _re
import time

from src.api.models import ChatRequest, ChatResponse
from src.api.routes.chat_pipeline.telemetry import (
    llm_completion_meta,
    work_completion_meta,
)
from src.constants import TASK_IR_OBJECTIVE_LEN
from src.api.routes.chat_utils import RoutingResult
from src.api.services.memrl import failure_disposition_meta, score_completed_task
from src.api.structured_logging import task_extra
from src.features import features
from src.llm_primitives import LLMPrimitives
from src.structured_output.repair import RepairResult, parse_with_repair, primitives_completer
from src.roles import resolve_planner_role
from src.task_ir import canonicalize_task_ir

log = logging.getLogger(__name__)


def _should_inline_plan_call_for_test(primitives: LLMPrimitives) -> bool:
    """Use inline call only in test/mocked contexts to avoid teardown hangs."""
    if os.getenv("PYTEST_CURRENT_TEST"):
        return True
    llm_call = getattr(primitives, "llm_call", None)
    return type(llm_call).__module__.startswith("unittest.mock")


def _parse_plan_steps(raw: str) -> list[dict]:
    """Parse architect JSON output into validated plan step dicts.

    Tolerant of markdown fences, trailing commas, and minor formatting issues.
    Returns empty list on parse failure (caller falls through to standard flow).
    """
    text = raw.strip()

    # Strip markdown code fences if present
    text = _re.sub(r"^```(?:json)?\s*", "", text)
    text = _re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Fix trailing commas before ] (common LLM quirk)
    text = _re.sub(r",\s*]", "]", text)

    try:
        steps = _json.loads(text)
    except _json.JSONDecodeError:
        return []

    if not isinstance(steps, list):
        return []

    # Validate each step has required fields
    valid_steps = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        if "id" not in step or "action" not in step:
            continue
        # Ensure defaults
        step.setdefault("actor", "worker")
        step.setdefault("depends_on", [])
        step.setdefault("outputs", [])
        valid_steps.append(step)

    return valid_steps


#: TD-21.20: the architect's own prompt contract already promises exactly
#: these three actors (`_TASK_DECOMPOSITION_FALLBACK` in
#: `src/prompt_builders/builder.py`: "actor: 'worker' for
#: exploration/summarization, 'coder' for code, 'architect' for design") --
#: this is "the code's existing valid set", not an invented one.
_PLAN_STEP_ACTORS: tuple[str, ...] = ("worker", "coder", "architect")

_PLAN_STEPS_REPAIR_SCHEMA: dict = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "action": {"type": "string"},
            "actor": {"type": "string", "enum": list(_PLAN_STEP_ACTORS)},
            "depends_on": {"type": "array", "items": {"type": "string"}},
            "outputs": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["id", "action"],
        "additionalProperties": False,
    },
}

_PLAN_STEPS_REPAIR_INSTRUCTION = (
    "The reply, given as the user message, is an architect's attempt to decompose a "
    "task into parallel-executable steps as a JSON array. Re-express the steps it "
    "already gives -- the same ids, actions, dependencies and outputs, never invented "
    "or improved -- as a JSON array matching the schema. `actor` must be exactly one "
    "of worker, coder, or architect; if a step names something else, pick whichever "
    "of those three its own wording is closest to. If the reply contains no "
    "extractable steps at all, return an empty array []."
)

_PLAN_STEPS_REPAIR_SITE = "chat_pipeline.proactive_stage.plan_steps"


def _finalize_plan_step(step: dict) -> dict | None:
    """Apply the same defaults `_parse_plan_steps` applies, to a step that
    came back from the repair turn instead of the fish. `None` when the step
    is not even a dict with the two truly required fields -- defensive only;
    the repair schema's `required`/`additionalProperties: False` already
    make this unreachable for a `"repaired"` result."""
    if not isinstance(step, dict) or "id" not in step or "action" not in step:
        return None
    step = dict(step)
    step.setdefault("actor", "worker")
    step.setdefault("depends_on", [])
    step.setdefault("outputs", [])
    return step


async def _decompose_plan_steps(
    raw: str,
    *,
    primitives: LLMPrimitives,
    task_id: str,
) -> list[dict]:
    """TD-21.20: fish first via the unchanged `_parse_plan_steps` (byte-
    identical happy path, 0 repair calls); on a miss, ONE repair turn back to
    the SAME planner role (`resolve_planner_role()`) instead of the old silent `[]`
    fall-through. Every outcome is counted under
    `STRUCTURED_OUTPUT_REPAIR_COUNTS` (site `_PLAN_STEPS_REPAIR_SITE`); a
    terminal repair failure is ALSO logged explicitly here (not left to the
    caller's generic "plan has 0 steps" fallback log), since that log alone
    cannot distinguish "architect declined to decompose" from "the plan was
    lost to formatting drift" -- exactly the audit's complaint about today's
    silent `[]`.
    """
    steps = _parse_plan_steps(raw)
    if steps:
        return steps

    # ARCHSWAP-20260927: review/plan work stays on the 27B — the repair turn goes back
    # to the SAME planner binding that produced the plan (resolve_planner_role).
    complete = primitives_completer(primitives, str(resolve_planner_role()))

    def _repair() -> RepairResult:
        return parse_with_repair(
            raw,
            schema=_PLAN_STEPS_REPAIR_SCHEMA,
            complete=complete,
            instruction=_PLAN_STEPS_REPAIR_INSTRUCTION,
            site=_PLAN_STEPS_REPAIR_SITE,
            # TD-21.34: `id`/`action`/`depends_on`/`outputs` must come from the
            # architect's own (malformed) decomposition -- an invented step id
            # or dependency would drive real parallel dispatch. `actor` is
            # exempt: the instruction explicitly allows a semantic
            # closest-match onto {worker, coder, architect} when the reply
            # names something else, so it is a classification, not a copy.
            require_evidence=True,
            evidence_exempt={"actor"},
        )

    if _should_inline_plan_call_for_test(primitives):
        result = _repair()
    else:
        # Keep the extra model I/O off the event loop, same as the primary
        # plan call above.
        result = await asyncio.to_thread(_repair)

    if result.status not in ("parsed", "repaired"):
        log.warning(
            "Proactive delegation: plan-step repair failed (status=%s, reason=%r); "
            "falling through to standard pipeline",
            result.status,
            result.reason,
            extra=task_extra(task_id=task_id, stage="execute", mode="proactive"),
        )
        return []

    return [s for s in (_finalize_plan_step(item) for item in result.value) if s is not None]


async def _execute_proactive(
    request: ChatRequest,
    routing: RoutingResult,
    primitives: LLMPrimitives,
    state,
    start_time: float,
) -> ChatResponse | None:
    """Proactive parallel delegation for COMPLEX tasks.

    When parallel_execution feature is enabled and the task is classified as
    COMPLEX, asks the architect to decompose it into parallel-executable steps,
    then delegates via ProactiveDelegator for wave-based parallel execution.

    Returns None to fall through to standard flow if:
    - Feature not enabled
    - Task not COMPLEX
    - Architect already selected (avoids double-entry with _execute_delegated)
    - Plan parsing fails or produces < 2 steps
    """
    if not (features().parallel_execution and request.real_mode):
        return None
    # INF-78 (OAB-1/OAB-7/OAB-8): a task_root-scoped request is ONE agentic call in a given
    # worktree (the AutoKernel planner/author/critic), and a context_bundle request asked
    # for the REPL loop over ITS bundle. This stage runs before mode selection, ignores
    # force_mode and is not scope-aware, so a prompt classified COMPLEX (a 35-75k-char
    # planner prompt easily is) would be answered by architect decomposition instead --
    # the caller's REPL loop, its bundle and OAB-8's scouts silently dropped. Both lanes
    # found this independently; the guard is the UNION (task_root OR context_bundle).
    if (getattr(request, "task_root", None) is not None
            or getattr(request, "context_bundle", None) is not None):
        return None

    from src.proactive_delegation import classify_task_complexity, TaskComplexity

    complexity, _signals = classify_task_complexity(request.prompt)
    if complexity != TaskComplexity.COMPLEX:
        return None

    # Avoid double-entry: if architect was already selected by routing, let
    # _execute_delegated() handle it (sequential TOON delegation path)
    initial_role = routing.routing_decision[0] if routing.routing_decision else "frontdoor"
    if str(initial_role) == "architect_general":
        return None

    log.info(
        "Proactive delegation: COMPLEX task detected, requesting plan from architect",
        extra=task_extra(task_id=routing.task_id, stage="execute", mode="proactive"),
    )

    # Ask the planner to decompose into parallel steps. ARCHSWAP-20260927:
    # review/plan work stays on the 27B (resolve_planner_role, default
    # architect_critic); only escalation targets architect_general.
    from src.prompt_builders import build_task_decomposition_prompt

    planner_role = str(resolve_planner_role())

    plan_prompt = build_task_decomposition_prompt(
        request.prompt,
        request.context or "",
    )

    try:
        if _should_inline_plan_call_for_test(primitives):
            plan_json_str = primitives.llm_call(
                plan_prompt,
                role=planner_role,
                n_tokens=256,
            )
        else:
            # Keep model I/O off the event loop in production runtime paths.
            plan_json_str = await asyncio.to_thread(
                primitives.llm_call,
                plan_prompt,
                role=planner_role,
                n_tokens=256,
            )
    except Exception as e:
        log.warning(
            "Proactive delegation: architect plan call failed: %s",
            e,
            extra=task_extra(
                task_id=routing.task_id,
                stage="execute",
                mode="proactive",
                error_type=type(e).__name__,
            ),
        )
        return None

    steps = await _decompose_plan_steps(plan_json_str, primitives=primitives, task_id=routing.task_id)
    if not steps or len(steps) < 2:
        log.info(
            "Proactive delegation: plan has %d steps (need >= 2), falling through",
            len(steps),
            extra=task_extra(task_id=routing.task_id, stage="execute", mode="proactive"),
        )
        return None

    # Build TaskIR from parsed steps
    task_ir = canonicalize_task_ir({
        "task_id": routing.task_id,
        "task_type": routing.task_ir.get("task_type", "chat"),
        "objective": request.prompt[:TASK_IR_OBJECTIVE_LEN],
        "plan": {"steps": steps},
        "context_preview": request.context or "",
    })

    from src.proactive_delegation import ProactiveDelegator

    delegator = ProactiveDelegator(
        registry=state.registry,
        primitives=primitives,
        progress_logger=state.progress_logger,
        hybrid_router=state.hybrid_router,
    )

    try:
        result = await delegator.delegate(task_ir)
    except Exception as e:
        log.warning(
            "Proactive delegation: execution failed: %s",
            e,
            extra=task_extra(
                task_id=routing.task_id,
                stage="execute",
                mode="proactive",
                error_type=type(e).__name__,
            ),
        )
        return None

    answer = result.aggregated_output.strip() if result.aggregated_output else ""
    if not answer:
        return None

    elapsed = time.perf_counter() - start_time
    n_subtasks = len(result.subtask_results)
    state.increment_request(mock_mode=False, turns=1 + n_subtasks)

    if state.progress_logger:
        state.progress_logger.log_task_completed(
            task_id=routing.task_id,
            success=result.all_approved,
            details=f"Proactive delegation: {n_subtasks} subtasks, {elapsed:.3f}s",
            completion_meta={
                "producer_role": "proactive_delegation",
                "delegation_lineage": result.roles_used or [str(resolve_planner_role())],
                "final_answer_role": (result.roles_used or [str(resolve_planner_role())])[-1],
                **llm_completion_meta(primitives),
                # M-11a2b
                **work_completion_meta(answer=answer),
                # `all_approved=False` on an in-band `[ERROR: ...]` / empty
                # answer is a backend fact, not a quality one.
                **failure_disposition_meta(
                    answer=answer,
                    tokens_generated=primitives.total_tokens_generated,
                ),
            },
        )
        score_completed_task(
            state,
            routing.task_id,
            force_role=request.force_role,
            real_mode=request.real_mode,
        )

    cache_stats = primitives.get_cache_stats() if primitives._backends else None
    delegation_events = getattr(result, "delegation_events", [])
    delegation_success = result.all_approved if delegation_events else None
    return ChatResponse(
        answer=answer,
        turns=1 + n_subtasks,
        tokens_used=primitives.total_tokens_generated,
        elapsed_seconds=elapsed,
        mock_mode=False,
        real_mode=True,
        cache_stats=cache_stats,
        routed_to="proactive_delegation",
        role_history=result.roles_used or [str(resolve_planner_role())],
        routing_strategy="proactive",
        mode="proactive",
        tokens_generated=primitives.total_tokens_generated,
        formalization_applied=routing.formalization_applied,
        tools_used=0,
        delegation_events=delegation_events,
        delegation_success=delegation_success,
        prompt_eval_ms=primitives.total_prompt_eval_ms,
        generation_ms=primitives.total_generation_ms,
        predicted_tps=primitives._last_predicted_tps,
        http_overhead_ms=primitives.total_http_overhead_ms,
        skills_retrieved=len(routing.skill_ids),
        skill_ids=routing.skill_ids,
    )
