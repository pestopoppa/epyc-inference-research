"""Chat endpoints for the orchestrator API.

Thin orchestrator module containing only endpoint functions and the
_handle_chat() / chat_stream() pipelines. All helper functions have been
extracted into focused modules during Phase 1 decomposition:

- chat_utils.py      — Constants + utility functions
- chat_vision.py     — Vision pipeline (OCR, VL routing, ReAct VL)
- chat_summarization.py — Two-stage/three-stage context processing
- chat_review.py     — Architect review, quality gates, plan review
- chat_delegation.py — Architect delegation (TOON parsing, multi-loop)
- chat_routing.py    — Intent classification, mode selection, routing
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncGenerator

log = logging.getLogger(__name__)

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from src.api.dependencies import dep_app_state
from src.api.models import ChatRequest, ChatResponse, RewardRequest
from src.api.state import AppState
from src.config import get_config
from src.constants import TASK_IR_OBJECTIVE_LEN
from src.repl_environment import knowledge_fence
from src.repl_environment import task_root as task_root_mod
from src.runtime import quiescence
from src.runtime import routing_stage_timing
from src.scheduling import contention_gate_capture, gate_observation
from src.delegation_reports import load_report
from src.task_ir import canonicalize_task_ir
from src.prompt_builders import (
    build_root_lm_prompt,
    build_routing_context,
    extract_code_from_response,
    auto_wrap_final,
    classify_error,
    build_escalation_prompt,
)
from src.prompt_builders.builder import build_corpus_context
from src.prompt_builders.resolver import prompt_dir_override
from src.api.services.memrl import (
    ensure_memrl_initialized,
    failure_disposition_meta,
    score_completed_task,
    store_external_reward,
)
from src.llm_primitives import LLMPrimitives
from src.repl_environment import REPLEnvironment
from src.escalation import (
    EscalationPolicy,
    EscalationContext,
    EscalationAction,
)
from src.roles import Role, architect_repl_allowed
from src.sse_utils import (
    create_sse_response,
    token_event,
    thinking_event,
    turn_start_event,
    turn_end_event,
    error_event,
    final_event,
    done_event,
    tool_start_event,
    tool_end_event,
)

# Decomposed modules (Phase 1 — extract-and-move, no behavior changes)
from src.api.routes.chat_utils import (
    _resolve_answer,
)
from src.api.routes.chat_review import (
    REVIEW_PATH_LEGACY_STREAM,
    _architect_verdict_with_status,
    _fast_revise,
    _plan_review_abort_message,
    _plan_review_should_abort,
    review_ms_since,
    evaluate_review_gate,
    record_review_gate,
)
from src.api.routes.chat_routing import (
    _select_mode,
)
from src.features import features

if TYPE_CHECKING:
    from src.api.routes.chat_utils import RoutingResult

# Phase 1b: Pipeline stage functions (extracted from _handle_chat)
from src.api.routes.chat_pipeline import (
    _route_request,
    _preprocess,
    _init_primitives,
    _execute_mock,
    _execute_vision,
    _execute_vision_multimodal,
    _plan_review_gate,
    _execute_proactive,
    _execute_delegated,
    _execute_react,
    _execute_direct,
    _execute_repl,
    _annotate_error,
    _attach_budget_diagnostics,
)
from src.api.routes.chat_pipeline.vision_stage import _vision_roles
from src.api.routes.chat_pipeline.script_interceptor import try_intercept
from src.api.routes.chat_pipeline.telemetry import llm_completion_meta


_PROMPT_OVERRIDE_BASE = (
    Path(__file__).resolve().parents[3] / "tmp" / "gepa_prompt_roots"
)


def _validated_prompt_dir_override(value: str | None) -> Path | None:
    if not value:
        return None
    try:
        root = Path(value).expanduser().resolve(strict=True)
    except OSError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"x_orchestrator_prompt_root does not exist: {value}",
        ) from exc
    if not root.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"x_orchestrator_prompt_root is not a directory: {root}",
        )
    base_value = os.environ.get("ORCHESTRATOR_PROMPT_ROOT_OVERRIDE_BASE")
    base = Path(base_value).expanduser() if base_value else _PROMPT_OVERRIDE_BASE
    base = base.resolve(strict=False)
    try:
        root.relative_to(base)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"x_orchestrator_prompt_root must be under {base}",
        ) from exc
    return root

# Factual question prefixes that the coding cheap-first model handles poorly.
# Prompts starting with these are routed past cheap-first to frontdoor/REPL
# which have better general knowledge and web-search access.
_FACTUAL_QUESTION_PREFIXES = (
    "who ", "who was ", "what was ", "what year", "in what year",
    "where was ", "in which ", "which country", "which city",
    "when did ", "when was ",
)

# Task types that require multi-step REPL execution and cannot be handled
# by the cheap-first single-shot path.
_REPL_ONLY_TASK_TYPES = frozenset({"coder", "agentic"})

router = APIRouter()


@router.get("/chat/delegation-report/{report_id}")
async def fetch_delegation_report(
    report_id: str,
    offset: int = Query(default=0, ge=0),
    max_chars: int = Query(default=2400, ge=64, le=12000),
):
    """Fetch persisted delegation report chunk by handle id."""
    payload = load_report(report_id, offset=offset, max_chars=max_chars)
    return payload


_UNFENCED_EVAL_REQUESTS = 0


def _note_unfenced_eval_request(request: ChatRequest) -> None:
    """AP-54: an eval rollout from a client that predates `eval_fence` runs
    UNFENCED. Say so in the log (first occurrence, then every 500th) rather than
    letting the gap pass silently. Behaviour is unchanged."""
    global _UNFENCED_EVAL_REQUESTS
    _UNFENCED_EVAL_REQUESTS += 1
    if _UNFENCED_EVAL_REQUESTS == 1 or _UNFENCED_EVAL_REQUESTS % 500 == 0:
        log.warning(
            "AP-54: eval_batch /chat request without eval_fence (request_id=%s, "
            "count=%d) — rollout is UNFENCED; the client predates the knowledge fence",
            request.request_id,
            _UNFENCED_EVAL_REQUESTS,
        )


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    http_request: Request,
    state: AppState = Depends(dep_app_state),
) -> ChatResponse:
    """Process a chat request through the orchestrator.

    Modes:
    - mock_mode=True (default): Returns simulated response, no real inference
    - real_mode=True: Uses RadixAttention caching with live llama-server instances
    - Neither: Uses legacy model server (if configured)

    The real_mode flag enables:
    - CachingBackend with prefix routing
    - Cache statistics in response
    - Full orchestration loop with Root LM (Phase 8)
    """
    from src.runtime.live_telemetry import emit_lifecycle_transition

    if not request.request_id:
        request.request_id = f"api-{uuid.uuid4().hex[:12]}"
    # EVL-42 1c-fix (d): ChatRequest.tools/tool_choice are deprecated and never
    # consumed on /chat; make a caller that relies on them visible, not silent.
    ignored_tool_fields = request.ignored_tool_fields()
    if ignored_tool_fields:
        log.warning(
            "ChatRequest %s: deprecated field(s) %s supplied but /chat does not consume "
            "native tool schemas; use /v1/chat/completions.",
            request.request_id,
            ",".join(ignored_tool_fields),
        )
    api_identity = {
        "request_id": request.request_id,
        "batch_id": request.batch_id,
        "role": request.force_role or request.role or None,
    }
    emit_lifecycle_transition(
        "queued",
        details={
            "handler": "/chat",
            "workload_class": request.workload_class,
            "request_priority": request.request_priority,
        },
        **api_identity,
    )
    emit_lifecycle_transition(
        "api_request_started",
        details={
            "handler": "/chat",
            "workload_class": request.workload_class,
            "request_priority": request.request_priority,
        },
        **api_identity,
    )
    handler_outcome = "failed"
    # Track active requests for idle-time Q-scoring (thread-safe)
    state.increment_active()
    cancel_event = threading.Event()

    async def _watch_disconnect() -> None:
        while not cancel_event.is_set():
            try:
                if await http_request.is_disconnected():
                    cancel_event.set()
                    return
            except Exception:
                # Best-effort disconnect tracking; ignore transient request state errors.
                return
            await asyncio.sleep(0.1)

    use_disconnect_watcher = bool(
        request.real_mode and "PYTEST_CURRENT_TEST" not in os.environ
    )
    watcher = asyncio.create_task(_watch_disconnect()) if use_disconnect_watcher else None
    prompt_dir = _validated_prompt_dir_override(request.x_orchestrator_prompt_root)
    # BRIDGE RESIDUAL 1 — collect the contention verdict for THIS request. Both
    # exits below (the error JSONResponse and the plain return) are downstream of
    # the single stamp, so neither can drop it.
    gate_observation.begin()
    # AP-54: per-request eval knowledge fence. No carrier (and no behaviour
    # change) unless the client sent `eval_fence`.
    fence_carrier = knowledge_fence.begin(request.eval_fence)
    if fence_carrier is None and request.workload_class == "eval_batch":
        _note_unfenced_eval_request(request)
    # INF-78 OAB-1: per-request task scope (ContextVar — concurrent requests never share it).
    # task_root was validated by ChatRequest; None installs nothing (production unchanged).
    # INF-78 OAB-3 (R2): suppress fire-and-forget work for this request; False is a no-op.
    # Both are installed INSIDE the try so the finally always clears them (and decrements
    # the active-request count) even if installation fails.
    task_scope = None
    quiescence_carrier = None
    try:
        try:
            task_scope = task_root_mod.begin_request_scope(
                request.task_root, request.edit_mode, request.read_roots
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        quiescence_carrier = quiescence.begin(
            request.quiescent_after,
            request_id=request.request_id or "",
            budget_s=request.timeout_s,
        )
        with prompt_dir_override(prompt_dir):
            response = await _handle_chat(request, state, cancel_event=cancel_event)
        response.contention_gate = gate_observation.snapshot()
        if fence_carrier is not None:
            response.eval_fence = fence_carrier.snapshot()
        if task_scope is not None:
            response.task_scope = task_scope.snapshot()
        if quiescence_carrier is not None:
            response.quiescence = quiescence_carrier.snapshot()
        # SC19 — persist the echoed verdict for the vidya belief kernel. Opt-in via
        # ORCHESTRATOR_CONTENTION_GATE_CAPTURE, never raises; the JSONL capture is the
        # durable bytes the adapter projects from. Both exits below (error JSONResponse
        # and the plain return) are downstream of this single stamp.
        contention_gate_capture.capture(response.contention_gate, request_id=request.request_id)
        # Return appropriate HTTP status instead of silent 200 OK on failure
        if response.error_code:
            handler_outcome = "failed"
            headers = {}
            if response.error_code == 503:
                headers["Retry-After"] = "30"
            return JSONResponse(
                status_code=response.error_code,
                content=response.model_dump(),
                headers=headers or None,
            )
        handler_outcome = "completed"
        return response
    except asyncio.CancelledError:
        handler_outcome = "disconnected"
        raise
    except TimeoutError:
        handler_outcome = "timeout"
        raise
    finally:
        # A carrier must never outlive its request — a leaked one would attribute
        # this request's gate verdict to the next caller on the same context.
        gate_observation.clear()
        knowledge_fence.clear()
        task_root_mod.clear_request_scope()
        quiescence.finish(quiescence_carrier)
        quiescence.clear()
        if cancel_event.is_set() and handler_outcome == "failed":
            handler_outcome = "disconnected"
        emit_lifecycle_transition(
            "completed" if handler_outcome == "completed" else "failed",
            details={
                "handler": "/chat",
                "outcome": handler_outcome,
                "workload_class": request.workload_class,
            },
            **api_identity,
        )
        emit_lifecycle_transition(
            "api_request_finished",
            details={
                "handler": "/chat",
                "outcome": handler_outcome,
                "client_disconnected": cancel_event.is_set(),
                "workload_class": request.workload_class,
            },
            **api_identity,
        )
        cancel_event.set()
        if watcher is not None:
            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watcher
        state.decrement_active()


@router.post("/chat/reward")
async def inject_reward(
    request: RewardRequest,
    state: AppState = Depends(dep_app_state),
):
    """Inject an external reward signal into MemRL.

    Used by orchestrator_eval.py to close the learning loop: after
    deterministic scoring of a benchmark answer, the score is fed back
    as a reward so the Q-scorer can learn from routing decisions.

    Supports precomputed embeddings via `embedding` field to avoid
    re-embedding the same task_description across multiple reward injections.
    """
    success = await asyncio.to_thread(
        store_external_reward,
        state,
        request.task_description,
        request.action,
        request.reward,
        request.context,
        request.embedding,
    )
    return {"success": success}


async def _try_cheap_first(
    request: ChatRequest,
    routing: "RoutingResult",
    primitives: "LLMPrimitives",
    state: AppState,
    start_time: float,
    initial_role,
    execution_mode: str,
) -> ChatResponse | None:
    """Speculative pre-filter: try answering with cheap 7B model first.

    Returns ChatResponse if the cheap answer passes quality gate,
    or None to fall through to the normal pipeline.

    The cheap attempt is a speculative optimization layer ABOVE the existing
    delegation/escalation chain. On failure, the 7B's output can be passed
    as context to the specialist, giving the expensive model a head start.
    """
    import logging

    log = logging.getLogger(__name__)

    cfg = get_config().chat
    if not cfg.try_cheap_first_enabled:
        _log_cheap_first_counter(
            state,
            routing.task_id,
            attempted=False,
            passed=None,
            reason="disabled",
            cheap_role=str(getattr(cfg, "try_cheap_first_role", "")),
            phase=str(getattr(cfg, "try_cheap_first_phase", "")),
        )
        return None

    # Skip for forced modes, vision, delegation, or already-cheap roles
    cheap_role = cfg.try_cheap_first_role
    if request.force_mode or request.force_role:
        _log_cheap_first_counter(
            state,
            routing.task_id,
            attempted=False,
            passed=None,
            reason="forced_request",
            cheap_role=cheap_role,
            phase=cfg.try_cheap_first_phase,
        )
        return None
    normalized_initial_role = Role.from_string(str(initial_role)) if initial_role is not None else None
    if normalized_initial_role == Role.INGEST_LONG_CONTEXT:
        # The long-context route is a capacity boundary, not merely a model
        # preference.  Sending the full prompt through the speculative cheap
        # worker first defeats that boundary and can strand an oversized
        # request on worker_general before ingest ever runs.
        _log_cheap_first_counter(
            state,
            routing.task_id,
            attempted=False,
            passed=None,
            reason="long_context_capacity_route",
            cheap_role=cheap_role,
            phase=cfg.try_cheap_first_phase,
        )
        return None
    if normalized_initial_role in {Role.WORKER_GENERAL, Role.WORKER_MATH, Role.WORKER_VISION}:
        _log_cheap_first_counter(
            state,
            routing.task_id,
            attempted=False,
            passed=None,
            reason="already_cheap_or_vision",
            cheap_role=cheap_role,
            phase=cfg.try_cheap_first_phase,
            initial_role=str(normalized_initial_role or initial_role),
        )
        return None  # Already cheap
    if execution_mode == "delegated":
        _log_cheap_first_counter(
            state,
            routing.task_id,
            attempted=False,
            passed=None,
            reason="delegated_mode",
            cheap_role=cheap_role,
            phase=cfg.try_cheap_first_phase,
        )
        return None

    # RI-2: Skip cheap-first when factual risk is high — cheap models are
    # unreliable on factual questions and incorrect answers are worse than slow.
    if routing.factual_risk_band == "high":
        _log_cheap_first_counter(
            state,
            routing.task_id,
            attempted=False,
            passed=None,
            reason="high_factual_risk",
            cheap_role=cheap_role,
            phase=cfg.try_cheap_first_phase,
            factual_risk_band=routing.factual_risk_band,
        )
        return None

    # Phase B/C: check Q-value before attempting
    if cfg.try_cheap_first_phase in ("B", "C"):
        if hasattr(state, "hybrid_router") and state.hybrid_router is not None:
            try:
                task_ir = {
                    "task_type": "chat",
                    "objective": request.prompt[:TASK_IR_OBJECTIVE_LEN],
                    "routing_preferences": request.routing_preferences,
                }
                task_ir = canonicalize_task_ir(task_ir)
                results = state.hybrid_router.retriever.retrieve_for_routing(task_ir)
                # Check if the cheap role Q-value is above threshold
                worker_q = 0.0
                for r in results:
                    if r.memory.metadata.get("role") == cheap_role:
                        worker_q = max(worker_q, r.q_value)
                if worker_q < cfg.try_cheap_first_q_threshold:
                    _log_cheap_first_counter(
                        state,
                        routing.task_id,
                        attempted=False,
                        passed=None,
                        reason="low_q_value",
                        cheap_role=cheap_role,
                        phase=cfg.try_cheap_first_phase,
                        worker_q=worker_q,
                        q_threshold=cfg.try_cheap_first_q_threshold,
                    )
                    return None  # MemRL says cheap won't work for this task class
            except Exception:
                pass  # Fall through to Phase A behavior

    # Attempt cheap answer
    prompt = request.prompt
    if request.context:
        prompt = f"{request.context}\n\n{request.prompt}"

    try:
        from src.api.routes.chat_utils import QWEN_STOP, apply_chat_template_for_role

        # 2026-05-23: skip template wrap when cheap_role routes through
        # /v1/chat/completions (server-side jinja handles templating).
        from src.chat_completions_roles import chat_completions_roles
        _cc_roles = chat_completions_roles()  # shared SoT (was a divergent inline default)
        if cheap_role in _cc_roles:
            templated_prompt = prompt
        else:
            # Apply per-role chat template — without this, gemma-family cheap
            # roles (worker_general post 2026-05-08 swap) see raw text instead
            # of the expected turn markers and return 0 tokens.
            templated_prompt = apply_chat_template_for_role(
                cheap_role,
                prompt,
                registry=getattr(state, "registry", None),
            )
        # QWEN_STOP only applies if the templated prompt is actually Qwen-
        # marked. For non-Qwen families, the marker won't appear in output;
        # for Qwen, leaving it in is safe.
        answer = primitives.llm_call(
            templated_prompt,
            role=cheap_role,
            n_tokens=cfg.try_cheap_first_max_tokens,
            skip_suffix=True,
            stop_sequences=["\n\n\n", QWEN_STOP],
        )
        answer = answer.strip()
    except Exception as e:
        log.debug("Cheap-first attempt failed: %s", e)
        _log_cheap_first_counter(
            state,
            routing.task_id,
            attempted=True,
            passed=False,
            reason="llm_exception",
            cheap_role=cheap_role,
            phase=cfg.try_cheap_first_phase,
            detail=type(e).__name__,
        )
        return None

    # Quality gate: reject empty, very short, or error-like answers
    if not answer or len(answer) < 20:
        _log_cheap_first_counter(
            state,
            routing.task_id,
            attempted=True,
            passed=False,
            reason="empty_or_short_answer",
            cheap_role=cheap_role,
            phase=cfg.try_cheap_first_phase,
            answer_chars=len(answer or ""),
        )
        return None
    if answer.startswith("[ERROR"):
        _log_cheap_first_counter(
            state,
            routing.task_id,
            attempted=True,
            passed=False,
            reason="error_answer",
            cheap_role=cheap_role,
            phase=cfg.try_cheap_first_phase,
            answer_chars=len(answer),
        )
        return None

    # Check for obvious quality issues
    from src.api.routes.chat_review import _detect_output_quality_issue

    quality_issue = _detect_output_quality_issue(answer)
    if quality_issue:
        log.debug("Cheap-first quality gate failed: %s", quality_issue)
        _log_cheap_first_counter(
            state,
            routing.task_id,
            attempted=True,
            passed=False,
            reason="quality_issue",
            cheap_role=cheap_role,
            phase=cfg.try_cheap_first_phase,
            answer_chars=len(answer),
            detail=str(quality_issue)[:160],
        )
        return None

    # Passed quality gate — return cheap answer
    elapsed = time.perf_counter() - start_time
    log.info(
        "Try-cheap-first PASSED: %s answered in %.1fs (phase %s)",
        cheap_role, elapsed, cfg.try_cheap_first_phase,
    )
    _log_cheap_first_counter(
        state,
        routing.task_id,
        attempted=True,
        passed=True,
        reason="passed",
        cheap_role=cheap_role,
        phase=cfg.try_cheap_first_phase,
        elapsed_seconds=elapsed,
        answer_chars=len(answer),
    )

    return ChatResponse(
        answer=answer,
        turns=1,
        elapsed_seconds=elapsed,
        mock_mode=False,
        real_mode=True,
        routed_to=cheap_role,
        role_history=[cheap_role],
        routing_strategy=f"cheap_first:{routing.routing_strategy}",
        generation_ms=elapsed * 1000,
        mode="direct",
        cheap_first_attempted=True,
        cheap_first_passed=True,
        skills_retrieved=len(routing.skill_ids),
        skill_ids=routing.skill_ids,
        tokens_generated=primitives.total_tokens_generated,
    )


def _log_cheap_first_counter(
    state: AppState,
    task_id: str,
    *,
    attempted: bool,
    passed: bool | None,
    reason: str,
    cheap_role: str = "",
    phase: str = "",
    initial_role: str = "",
    factual_risk_band: str = "",
    worker_q: float | None = None,
    q_threshold: float | None = None,
    elapsed_seconds: float | None = None,
    answer_chars: int | None = None,
    detail: str = "",
) -> None:
    """Append try-cheap-first denominator/accept/reject telemetry to progress JSONL."""
    progress_logger = getattr(state, "progress_logger", None)
    if not progress_logger:
        return
    try:
        from orchestration.repl_memory.progress_logger import EventType, ProgressEntry

        data = {
            "kind": "try_cheap_first",
            "cheap_first_attempted": attempted,
            "cheap_first_passed": passed,
            "reason": reason,
            "cheap_role": str(cheap_role or ""),
            "phase": str(phase or ""),
        }
        if initial_role:
            data["initial_role"] = initial_role
        if factual_risk_band:
            data["factual_risk_band"] = factual_risk_band
        if worker_q is not None:
            data["worker_q"] = round(float(worker_q), 4)
        if q_threshold is not None:
            data["q_threshold"] = round(float(q_threshold), 4)
        if elapsed_seconds is not None:
            data["elapsed_seconds"] = round(float(elapsed_seconds), 4)
        if answer_chars is not None:
            data["answer_chars"] = int(answer_chars)
        if detail:
            data["detail"] = detail
        progress_logger.log(
            ProgressEntry(
                event_type=EventType.ROUTING_FALLBACK,
                task_id=task_id,
                data=data,
            )
        )
    except Exception:
        log.debug("Cheap-first progress telemetry failed", exc_info=True)


async def _run_scout_stage(
    request: ChatRequest,
    primitives: Any,
    initial_role: Any,
    cancel_event: threading.Event | None,
    request_deadline_s: float,
    holder: dict[str, Any],
) -> ChatRequest:
    """INF-78 OAB-8: run the scouts and return the request the planner turn will see.

    The original request object is never mutated (a copy carries the augmented prompt), and
    every scout has returned before this does, so none can outlive the reply (R2)."""
    from src.api.routes.chat_pipeline.scout_stage import augment_prompt, run_scouts

    spec = request.scouts
    role = str(spec.role or request.force_role or initial_role)
    try:
        stage = await run_scouts(
            spec,
            scope=task_root_mod.request_scope(),
            role=role,
            primitives=primitives,
            cancel_event=cancel_event,
            request_deadline_s=request_deadline_s,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 -- scouts degrade to none; the planner still runs
        log.warning("scout stage failed for %s: %s", request.request_id, exc)
        holder["report"] = {"schema": "epyc.orchestrator.scouts.v1", "enabled": True,
                            "role": role, "error": f"{type(exc).__name__}: {exc}"[:500],
                            "launched": 0, "completed": 0, "scouts": []}
        return request
    holder["report"] = stage.report
    if not stage.block:
        return request
    return request.model_copy(update={"prompt": augment_prompt(request.prompt, stage.block)})


async def _handle_chat(
    request: ChatRequest,
    state: AppState,
    cancel_event: threading.Event | None = None,
) -> ChatResponse:
    """Thin dispatcher — routes through pipeline stages.

    Phase 1b restructure: each stage is a named function in chat_pipeline.py.
    The 1,091-line monolith is now ~80 lines of orchestration calling
    independently testable stage functions.

    Pipeline:
        0. try_intercept()      → zero-token local response (flag-gated)
        1. _route_request()     → RoutingResult (role, strategy, timeout)
        2. _preprocess()        → input formalization (mutates request)
        3. _execute_mock()      → ChatResponse (mock mode early return)
        4. _init_primitives()   → LLMPrimitives (backend setup)
        5. _plan_review_gate()  → optional routing adjustment
        6. _execute_vision()    → ChatResponse | None (vision early return)
        6.8 _run_scout_stage()  → INF-78 OAB-8 read-only scouts (only when request.scouts)
        7. Mode selection       → "direct" / "react" / "repl" / "delegated"
        8. Mode handler         → ChatResponse
        9. _annotate_error()    → set error_code/error_detail on failures
    """
    start_time = time.perf_counter()

    # Stage 0: zero-cost local resolution for deterministic prompts.
    if features().script_interception:
        interception = try_intercept(request.prompt)
        if interception.matched:
            elapsed = time.perf_counter() - start_time
            state.increment_request(mock_mode=False, turns=0)
            return ChatResponse(
                answer=interception.result or "",
                turns=0,
                tokens_used=0,
                elapsed_seconds=elapsed,
                mock_mode=False,
                real_mode=False,
                routed_to="local_script",
                role_history=["local_script"],
                routing_strategy=f"script_interception:{interception.pattern_name}",
                mode="script_interception",
                generation_ms=interception.elapsed_ms,
            )

    # Stage 1: Routing
    routing = _route_request(request, state)

    # Stage 2: Input formalization
    _preprocess(request, state, routing)

    # Stage 3: Mock mode (early return)
    if routing.use_mock:
        return _execute_mock(request, routing, state, start_time)

    # Stage 4: Initialize LLM backends
    primitives = _init_primitives(request, state)
    request_deadline_s = time.perf_counter() + max(1.0, float(routing.timeout_s))
    if request.client_deadline_unix_s is not None:
        # Clamp against client-observed wall-clock deadline to avoid server work
        # outliving client timeout under admission/concurrency queueing.
        client_remaining_s = request.client_deadline_unix_s - time.time()
        request_deadline_s = min(
            request_deadline_s,
            time.perf_counter() + max(0.0, client_remaining_s),
        )
    with primitives.request_context(
        cancel_check=cancel_event.is_set if cancel_event is not None else None,
        deadline_s=request_deadline_s,
        task_id=routing.task_id,
        session_id=request.session_id,
        request_id=request.request_id,
        trial_id=request.trial_id,
        batch_id=request.batch_id,
        priority=(
            "background"
            if str(request.request_priority).strip().lower() == "background"
            else "interactive"
        ),
        workload_class=request.workload_class,
        batch_placement_mode=request.batch_placement_mode,
        max_queue_wait_ms=request.max_queue_wait_ms,  # Phase B: cross-role gate budget
    ):
        def _attach_routing_telemetry(resp: ChatResponse) -> ChatResponse:
            """NIB2-35: stamp shadow routing signals onto the response so
            seeding_diagnostics.jsonl can persist them for offline analysis."""
            try:
                resp.factual_risk_score = float(getattr(routing, "factual_risk_score", 0.0) or 0.0)
                resp.factual_risk_band = str(getattr(routing, "factual_risk_band", "") or "")
                resp.difficulty_score = float(getattr(routing, "difficulty_score", 0.0) or 0.0)
                resp.difficulty_band = str(getattr(routing, "difficulty_band", "") or "")
                resp.xmas_meta = getattr(routing, "xmas_meta", None)
                # RI-16: per-request routing-stage latency (ms).
                stage_telemetry = routing_stage_timing.telemetry_for(routing.task_id)
                if stage_telemetry is not None:
                    resp.routing_stage_ms = stage_telemetry["stage_ms"]
            except Exception:
                # Best-effort telemetry attachment; never break the response.
                pass
            return resp

        # INF-78 OAB-8: the scout stage's provenance, stamped on every finalized response.
        scout_holder: dict[str, Any] = {}

        def _finalize(resp: ChatResponse) -> ChatResponse:
            if scout_holder.get("report") is not None:
                resp.scouts = scout_holder["report"]
            return _annotate_error(
                _attach_routing_telemetry(_attach_budget_diagnostics(resp, primitives))
            )

        # Stage 5: Architect plan review gate
        plan_review_result = _plan_review_gate(request, routing, primitives, state)
        if _plan_review_should_abort(plan_review_result):
            assert plan_review_result is not None
            answer = _plan_review_abort_message(plan_review_result)
            elapsed = time.perf_counter() - start_time
            state.increment_request(mock_mode=False, turns=1)
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
            return _finalize(
                ChatResponse(
                    answer=answer,
                    turns=1,
                    tokens_used=int(getattr(primitives, "total_tokens_generated", 0) or 0),
                    elapsed_seconds=elapsed,
                    mock_mode=False,
                    real_mode=request.real_mode,
                    routed_to="architect_general",
                    role_history=["architect_general"],
                    routing_strategy=f"plan_review_drop:{routing.routing_strategy}",
                    mode="plan_review_drop",
                    tokens_generated=int(getattr(primitives, "total_tokens_generated", 0) or 0),
                    formalization_applied=routing.formalization_applied,
                    generation_ms=float(getattr(primitives, "total_generation_ms", 0.0) or 0.0),
                    error_code=422,
                    error_detail=answer,
                    skills_retrieved=len(routing.skill_ids) if routing.skill_ids else None,
                    skill_ids=routing.skill_ids,
                )
            )

        # Stage 6: Vision pipeline (early return if image/file present)
        vision_result = await _execute_vision(request, routing, primitives, state, start_time)
        if vision_result is not None:
            return _finalize(vision_result)

        # Stage 6.5: Proactive delegation for COMPLEX tasks
        proactive_result = await _execute_proactive(
            request,
            routing,
            primitives,
            state,
            start_time,
        )
        if proactive_result is not None:
            return _finalize(proactive_result)

        # Stage 7: Mode selection
        initial_role = routing.routing_decision[0] if routing.routing_decision else Role.FRONTDOOR

        # Stage 6.8 (INF-78 OAB-8): orchestrator-run read-only scouts, concurrently, BEFORE
        # the planner turn; their labelled summaries head the planner's root context. Off
        # unless the request enables them — the default path never enters this branch.
        if request.scouts is not None and request.scouts.enabled and request.real_mode:
            request = await _run_scout_stage(
                request, primitives, initial_role, cancel_event, request_deadline_s,
                scout_holder,
            )

        vision_roles = _vision_roles()
        forced_mode = request.force_mode if request.force_mode in ("direct", "react", "repl", "delegated", "edit") else None

        # Vision-preprocessed requests need REPL document context unless an explicit
        # forced mode is set. For multi-file/document workflows we keep synthesis on
        # frontdoor; for pure image workflows preserve vision role if selected/forced.
        if routing.document_result is not None:
            execution_mode = forced_mode or "repl"
            if request.files:
                initial_role = Role.FRONTDOOR
            elif request.force_role in vision_roles:
                initial_role = request.force_role
            elif str(initial_role) not in vision_roles:
                initial_role = Role.FRONTDOOR
        elif forced_mode:
            execution_mode = forced_mode
        else:
            execution_mode = _select_mode(request.prompt, request.context or "", state)

        # INF-78 scoped exception (operator 2026-09-25): the architect may run REPL only
        # for task-scoped requests. Unscoped architect REPL keeps today's behaviour (the
        # 2026-09-24 ruling is caller-side) but is logged so a violation is visible.
        if execution_mode == "repl" and not architect_repl_allowed(
                initial_role, task_root=getattr(request, "task_root", None)):
            log.warning(
                "architect REPL without task_root (role=%s): outside the INF-78 scoped "
                "exception; operator ruling 2026-09-24 keeps the architect out of REPL",
                initial_role)

        # Stage 7.5: Vision multimodal handler
        # Text-only paths (_execute_direct, _execute_repl) discard image data.
        # When a vision role has image data, route through the VL handler instead.
        if str(initial_role) in vision_roles and (request.image_path or request.image_base64):
            vision_mm = await _execute_vision_multimodal(
                request, routing, primitives, state, start_time, initial_role, execution_mode,
            )
            if vision_mm is not None:
                return _finalize(vision_mm)

        # Stage 7.9: Try-cheap-first speculative pre-filter.
        # Attempts the task with the cheapest HOT model (7B, 44 t/s) before
        # routing to expensive specialists. On quality gate pass, returns the
        # cheap answer (2-3x faster). On fail, falls through to normal pipeline.
        # Bypass cheap-first for: factual recall questions (coding model hallucinates),
        # and task types that require multi-step REPL execution (coder, agentic).
        _task_type = routing.task_ir.get("task_type")
        _cheap_first_role = get_config().chat.try_cheap_first_role
        _xmas_enforced_non_cheap_role = (
            routing.xmas_meta
            and routing.xmas_meta.get("applied") is True
            and str(initial_role) != str(_cheap_first_role)
        )
        if _xmas_enforced_non_cheap_role:
            _log_cheap_first_counter(
                state,
                routing.task_id,
                attempted=False,
                passed=None,
                reason="xmas_enforce",
                cheap_role=_cheap_first_role,
                phase=get_config().chat.try_cheap_first_phase,
                detail="xmas_enforce_applied",
            )
            cheap_result = None
        elif request.prompt.lower().startswith(_FACTUAL_QUESTION_PREFIXES) or _task_type in _REPL_ONLY_TASK_TYPES:
            _log_cheap_first_counter(
                state,
                routing.task_id,
                attempted=False,
                passed=None,
                reason="pre_bypass",
                cheap_role=get_config().chat.try_cheap_first_role,
                phase=get_config().chat.try_cheap_first_phase,
                detail="factual_prefix_or_repl_only_task",
            )
            cheap_result = None
        else:
            cheap_result = await _try_cheap_first(
                request, routing, primitives, state, start_time, initial_role, execution_mode,
            )
        if cheap_result is not None:
            if cheap_result.answer and "<answer>" in cheap_result.answer and "</answer>" not in cheap_result.answer:
                cheap_result.answer += "</answer>"
            # Raw code answers (no <answer> tag) fall through to REPL — the cheap
            # model returned a function body without formatting (e.g. HumanEval/MBPP
            # intercepted before task_type bypass fires). REPL handles these correctly.
            if cheap_result.answer and "<answer>" not in cheap_result.answer:
                _log_cheap_first_counter(
                    state,
                    routing.task_id,
                    attempted=True,
                    passed=False,
                    reason="raw_answer_without_answer_tag",
                    cheap_role=cheap_result.routed_to,
                    phase=get_config().chat.try_cheap_first_phase,
                    answer_chars=len(cheap_result.answer),
                )
                cheap_result = None
        if cheap_result is not None:
            return _finalize(cheap_result)

        # Stage 8: Execute selected mode (with fallthrough on failure)

        # 8a: Delegated mode (architect → specialist)
        # _execute_delegated checks delegation_allowed internally and returns None if not allowed
        if execution_mode == "delegated":
            result = await asyncio.to_thread(
                _execute_delegated,
                request,
                routing,
                primitives,
                state,
                start_time,
                initial_role,
                execution_mode,
            )
            if result is not None:
                return _finalize(result)

        # 8b: ReAct tool loop mode
        if execution_mode == "react":
            result = await asyncio.to_thread(
                _execute_react,
                request,
                routing,
                primitives,
                state,
                start_time,
                initial_role,
            )
            if result is not None:
                return _finalize(result)

        # 8b2: Edit-transaction mode — one-shot full-file edit for ROUTINE file edits.
        # Diagnosis 2026-05-27 (handoffs/active/multi-file-coding-completion-capability.md):
        # the coding role solves these edits in ONE shot (one-shot ablation 5/5 on the same
        # tasks+verifiers) but cannot reliably navigate the multi-turn REPL read->edit->FINAL
        # loop. This bypasses that loop: assemble the workspace files -> ask the model ONCE for
        # the complete new files -> apply transactionally (snapshot/write/compile-check/rollback)
        # -> auto-finalize. The REPL stays for exploratory computation.
        #   SAFETY: requires BOTH the flag (ORCHESTRATOR_EDIT_TRANSACTION=1) AND a scoped
        #   task-root (ORCHESTRATOR_EDIT_ROOT pointing at the workspace) — without the scoped
        #   root, assemble_context would read/rewrite the whole orchestrator repo. An EXPLICIT
        #   force_mode="edit" with either precondition missing FAILS CLOSED (412) rather than
        #   silently using the REPL path this work exists to avoid. Default-off => no prod change.
        if execution_mode == "edit" and request.real_mode:
            from src.edit_transaction import run_edit_transaction, edit_transaction_enabled
            from src.repl_environment.task_root import task_root_active, get_task_root
            from src.chat_completions_roles import chat_completions_roles

            _et_scope = task_root_mod.request_scope()
            if _et_scope is not None and not _et_scope.can_write:
                # INF-78 OAB-1: a task_root-scoped request with edit_mode='none' cannot write.
                raise HTTPException(
                    status_code=412,
                    detail="force_mode='edit' requires edit_mode='direct' on a task_root-scoped request",
                )
            if not edit_transaction_enabled() or not task_root_active():
                missing = []
                if not edit_transaction_enabled():
                    missing.append("ORCHESTRATOR_EDIT_TRANSACTION=1")
                if not task_root_active():
                    missing.append("a scoped ORCHESTRATOR_EDIT_ROOT directory")
                raise HTTPException(
                    status_code=412,
                    detail="force_mode='edit' precondition not met: requires " + " and ".join(missing),
                )

            _et_cc_roles = chat_completions_roles()
            _captured: dict = {}

            def _edit_llm_call(edit_prompt: str) -> str:
                # Same contract as the direct branch: template orchestrator-side unless the role
                # routes through /v1/chat/completions (then llama-server --jinja applies it).
                p = edit_prompt
                if str(initial_role) not in _et_cc_roles:
                    from src.api.routes.chat_utils import apply_chat_template_for_role
                    p = apply_chat_template_for_role(
                        str(initial_role), p, registry=getattr(state, "registry", None),
                    )
                req2 = (request.model_copy(update={"prompt": p})
                        if hasattr(request, "model_copy") else request.copy(update={"prompt": p}))
                resp = _execute_direct(req2, routing, primitives, state, start_time, initial_role)
                _captured["resp"] = resp
                return resp.answer or ""

            review_enabled = bool(features().review_before_commit_consult)
            review_before_commit = None
            review_before_commit_gate = None
            if review_enabled:
                def _review_before_commit(review_context: str) -> tuple[dict, dict]:
                    from src.orchestration.consultation import consult

                    from src.roles import resolve_reviewer_role

                    # ARCHSWAP-20260927: review/plan work stays on the 27B. This is
                    # REVIEW work, so it follows the reviewer binding (default
                    # architect_critic = the MI210 27B the skill's 2 s max_queue_wait_ms
                    # was sized for), not the consultant role architect_general (now CPU
                    # Flash-Next). The skill is declared under architect_critic in
                    # orchestration/interaction_skills.yaml; a reviewer override must
                    # name a role that carries it.
                    return consult(
                        consultant_role=str(resolve_reviewer_role()),
                        requester_role=str(initial_role),
                        skill="review_before_commit",
                        context=review_context,
                        primitives=primitives,
                    )

                review_before_commit = _review_before_commit
                if bool(features().review_before_commit_targeted_gate):
                    from src.orchestration.review_consult_gate import review_before_commit_gate_from_context

                    def _targeted_review_gate(context: dict) -> dict:
                        context = dict(context)
                        context["signals"] = {
                            "difficulty_band": str(getattr(routing, "difficulty_band", "") or ""),
                            "factual_risk_band": str(getattr(routing, "factual_risk_band", "") or ""),
                            "factual_risk_score": float(getattr(routing, "factual_risk_score", 0.0) or 0.0),
                            "benchmark_class": str(getattr(request, "batch_id", "") or ""),
                            "latency_budget_remaining_s": max(0.0, request_deadline_s - time.perf_counter()),
                        }
                        decision = review_before_commit_gate_from_context(context)
                        return {"enabled": decision.enabled, "reasons": list(decision.reasons)}

                    review_before_commit_gate = _targeted_review_gate

            def _edit_finish_reason() -> str:
                # TD-21.21 (coordinator review of 56e33d49): reading the plain
                # `primitives._last_inference_meta` attribute here was UNSAFE --
                # `_init_primitives` (chat_pipeline/routing.py:487) reuses one shared
                # `state._real_primitives` instance across concurrent requests in the same
                # worker, so that attribute can hold ANOTHER request's finish reason, and a
                # truncated edit misread as a natural stop would WRITE a truncated file. Use
                # `get_last_inference_meta()` instead: a `contextvars.ContextVar` that
                # `asyncio.to_thread`/`asyncio.Task` isolate per call-context, so a concurrent
                # request's write against the SAME primitives instance can never be observed
                # here (see `src/llm_primitives/primitives.py` `_last_inference_meta_ctx`).
                # `None` (no call made yet in THIS context) maps to "" -- unknown,
                # fail-closed -- never to a stale or borrowed value.
                meta = primitives.get_last_inference_meta() or {}
                return str(meta.get("completion_reason") or "")

            edit_res, _raw = await asyncio.to_thread(
                run_edit_transaction,
                _edit_llm_call,
                request.prompt,
                get_task_root(),
                None,
                review_before_commit=review_before_commit,
                enable_review_before_commit=review_enabled,
                review_before_commit_gate=review_before_commit_gate,
                get_finish_reason=_edit_finish_reason,
            )
            answer = (
                edit_res.summary + (": " + ", ".join(edit_res.written) if edit_res.written else "")
                if edit_res.ok else f"[edit-transaction failed] {edit_res.error}"
            ) + (f"  (rejected unsafe paths: {edit_res.rejected})" if edit_res.rejected else "")
            base = _captured.get("resp")
            if base is not None:
                diagnostics = dict(getattr(base, "delegation_diagnostics", {}) or {})
                if edit_res.consult_events:
                    diagnostics["edit_transaction_consult_events"] = edit_res.consult_events
                upd = {"answer": answer, "mode": "edit", "delegation_diagnostics": diagnostics}
                base = (base.model_copy(update=upd)
                        if hasattr(base, "model_copy") else base.copy(update=upd))
                return _finalize(base)
            # The model was never called (e.g., scope exceeded caps -> fail-closed); surface the
            # error instead of silently falling through to the REPL path.
            raise HTTPException(status_code=422, detail=answer)

        # 8c: Direct LLM call mode
        if execution_mode == "direct" and request.real_mode:
            # Per-role chat-template application (replaces the prior hardcoded
            # Qwen3 ChatML wrap that broke silently after 2026-05-08's
            # worker_general swap to gemma-4-26B-A4B-it). The helper looks up
            # the role's model name from the registry and picks the right
            # turn markers — Qwen for Qwen-family, Gemma for Gemma, Llama3
            # headers for Llama3, pass-through for unknowns. Without this,
            # bare text is treated as continuation, not user query — causing
            # rambling self-correction on Qwen models OR zero-token output on
            # Gemma when the wrong markers are sent.
            # 2026-05-23: when this role routes through /v1/chat/completions
            # (env: ORCHESTRATOR_USE_CHAT_COMPLETIONS_ROLES), skip the
            # orchestrator-side template wrap entirely — llama-server's
            # --jinja flag will apply the GGUF's chat template server-side.
            # Pre-templating would inject our template markers as literal
            # user content, confusing the model.
            from src.chat_completions_roles import chat_completions_roles
            _cc_roles = chat_completions_roles()  # shared SoT (was a divergent inline default)
            if str(initial_role) not in _cc_roles:
                from src.api.routes.chat_utils import apply_chat_template_for_role
                request.prompt = apply_chat_template_for_role(
                    str(initial_role),
                    request.prompt,
                    registry=getattr(state, "registry", None),
                )
            _direct_result = await asyncio.to_thread(
                _execute_direct,
                request,
                routing,
                primitives,
                state,
                start_time,
                initial_role,
            )
            if _direct_result.answer:
                _direct_result.answer = _direct_result.answer.lstrip('\n ')
            return _finalize(_direct_result)

        # 8d: REPL orchestration mode (default fallback)
        _repl_result = await _execute_repl(request, routing, primitives, state, start_time, initial_role)
        if "<answer>" in _repl_result.answer and "</answer>" not in _repl_result.answer:
            _repl_result.answer += "</answer>"
        return _finalize(_repl_result)


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    state: AppState = Depends(dep_app_state),
) -> StreamingResponse:
    """SSE streaming endpoint with routing metadata.

    Streams events using standardized SSE format (via sse_utils):
    - turn_start: {type: "turn_start", turn: N, role: "..."}
    - thinking: {type: "thinking", content: "..."} (when thinking_budget > 0)
    - token: {type: "token", content: "..."}
    - tool: {type: "tool", name: "...", args: {...}, result: ...}
    - permission_request: {type: "permission_request", id: "...", tool: "...", args: {...}}
    - file: {type: "file", path: "...", content: "...", action: "create"|"modify"}
    - turn_end: {type: "turn_end", tokens: N, elapsed_ms: N}
    - error: {type: "error", message: "..."}
    - done: [DONE] when complete

    Parameters:
    - thinking_budget: Token budget for internal reasoning (0=disabled)
    - permission_mode: "normal", "auto-accept", or "plan"

    Note: Uses sse-starlette when available (via feature flag), otherwise
    falls back to manual SSE formatting for backward compatibility.
    """
    # INF-78 review fix F2: neither streaming path below installs a task scope or a
    # quiescence carrier (compare /chat's `begin_request_scope` / `quiescence.begin` in the
    # `try` block above) — a caller that sends task_root/read_roots/quiescent_after/edit_mode
    # here would have them silently ignored, so the model would run with the FULL, unscoped
    # filesystem surface while the caller believes it is confined. Refuse explicitly instead
    # of streaming an unscoped run under a scoped-looking request.
    if (
        request.task_root is not None
        or request.read_roots
        or request.quiescent_after
        or request.edit_mode != task_root_mod.EDIT_MODE_NONE
        or (request.scouts is not None and request.scouts.enabled)
    ):
        raise HTTPException(
            status_code=422,
            detail="task scope / quiescent_after / scouts not supported on /chat/stream",
        )
    # INF-78 OAB-7: the streaming paths never attach a context bundle either.
    if request.context_bundle is not None:
        raise HTTPException(
            status_code=422, detail="context_bundle not supported on /chat/stream"
        )
    # Unified streaming path — reuses pipeline stages from _handle_chat()
    if features().unified_streaming:
        from src.api.routes.chat_pipeline.stream_adapter import generate_stream

        return create_sse_response(generate_stream(request, state))

    # Legacy inline streaming path (preserved until unified_streaming is validated)
    # Generate task ID for MemRL tracking (outside generator for closure)
    task_id = f"stream-{uuid.uuid4().hex[:8]}"

    async def generate() -> AsyncGenerator[dict, None]:
        start_time = time.perf_counter()
        use_mock = request.mock_mode and not request.real_mode
        # RI-16: this path has no priors/route/mode stages; it times the turn-0
        # routing context and the review gate.
        routing_stage_timing.begin(task_id, routing_stage_timing.PATH_LEGACY_STREAM)

        # Construct task_ir and log start (MemRL integration)
        task_ir = {
            "task_type": "chat_stream",
            "objective": request.prompt[:TASK_IR_OBJECTIVE_LEN],
            "priority": "interactive",
            "routing_preferences": request.routing_preferences,
        }
        task_ir = canonicalize_task_ir(task_ir)
        if state.progress_logger:
            state.progress_logger.log_task_started(
                task_id=task_id,
                task_ir=task_ir,
                routing_decision=[request.role],
                routing_strategy="mock" if use_mock else "rules",
            )

        # Mock mode
        if use_mock:
            # Emit turn start
            yield turn_start_event(turn=1, role=str(Role.FRONTDOOR))

            # Emit thinking events if thinking_budget > 0 (Claude Code parity)
            if request.thinking_budget > 0:
                thinking_steps = [
                    "Analyzing the user's request...",
                    f"Request type: {request.prompt[:30].split()[0] if request.prompt else 'unknown'}",
                    "Determining appropriate response strategy...",
                    "Preparing response...",
                ]
                for step in thinking_steps:
                    yield thinking_event(step)

            # Check permission mode - in plan mode, only emit analysis
            if request.permission_mode == "plan":
                analysis = f"[PLAN MODE] Would process: {request.prompt[:100]}..."
                yield token_event(analysis)
                elapsed_ms = int((time.perf_counter() - start_time) * 1000)
                yield turn_end_event(tokens=len(analysis), elapsed_ms=elapsed_ms)
                # Log completion (MemRL)
                if state.progress_logger:
                    state.progress_logger.log_task_completed(
                        task_id, success=True, details="Plan mode"
                    )
                    score_completed_task(state, task_id)
                yield done_event()
                return

            # Simulate streaming tokens
            mock_response = f"[MOCK] Processed: {request.prompt[:50]}..."
            for char in mock_response:
                yield token_event(char)

            # Emit turn end
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            yield turn_end_event(tokens=len(mock_response), elapsed_ms=elapsed_ms)

            # Log completion (MemRL)
            if state.progress_logger:
                state.progress_logger.log_task_completed(
                    task_id, success=True, details="Mock stream"
                )
                score_completed_task(state, task_id)
            yield done_event()
            return

        # Real mode - initialize MemRL components on first real use (lazy loading)
        ensure_memrl_initialized(state)

        server_urls = request.server_urls or get_config().server_urls.as_dict()
        try:
            primitives = LLMPrimitives(
                mock_mode=False,
                server_urls=server_urls,
                registry=state.registry,
                health_tracker=state.health_tracker,
                admission_controller=getattr(state, "admission", None),
                server_urls_source="request" if request.server_urls else "config",
            )
        except Exception as e:
            # Log failure (MemRL). Backend construction failed, so NO role ever
            # served this request: an infra exception here carries zero quality
            # information and must not reach the reward writer (it scored -0.5 →
            # initial_q 0.25, under the 0.3 retrieval floor, evicting the memory
            # outright). An application bug still classifies as a task failure
            # and is still scored.
            if state.progress_logger:
                state.progress_logger.log_task_completed(
                    task_id,
                    success=False,
                    details=str(e),
                    completion_meta=failure_disposition_meta(error=e) or None,
                )
                score_completed_task(state, task_id)
            yield error_event(str(e))
            yield done_event()
            return

        # Create REPL
        combined_context = request.prompt
        if request.context:
            combined_context += f"\n\nContext:\n{request.context}"

        repl = REPLEnvironment(
            context=combined_context,
            llm_primitives=primitives,
            tool_registry=state.tool_registry,
            script_registry=state.script_registry,
            role=request.role or Role.FRONTDOOR,
            # MemRL components for model self-routing
            retriever=state.hybrid_router.retriever if state.hybrid_router else None,
            hybrid_router=state.hybrid_router,
        )

        # Root LM loop with streaming + escalation support
        last_output = ""
        last_error = ""
        result = None
        prev_tool_count = 0

        # Escalation tracking (parity with main endpoint)
        current_role = request.role or Role.FRONTDOOR
        consecutive_failures = 0
        role_history = [current_role]
        escalation_prompt = ""

        for turn in range(request.max_turns):
            turn_start_time_inner = time.perf_counter()

            # Emit turn start with current role
            yield turn_start_event(turn=turn + 1, role=str(current_role))

            # Get state and build prompt
            repl_state = repl.get_state()
            if escalation_prompt:
                root_prompt = escalation_prompt
                escalation_prompt = ""
            else:
                # Inject routing context on turn 0
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

            # Call Root LM with current role
            try:
                code = primitives.llm_call(root_prompt, role=current_role, n_tokens=1024)
            except Exception as e:
                # Log failure (MemRL)
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
                            # Transport failure → no answer to judge. An
                            # application bug is a task failure and still scores.
                            **failure_disposition_meta(error=e),
                        },
                    )
                    score_completed_task(state, task_id)
                yield error_event(f"Root LM call failed: {e}")
                yield done_event()
                return

            # Stream the generated code tokens
            code = extract_code_from_response(code)
            # Auto-wrap in FINAL() if code looks like a complete answer
            code = auto_wrap_final(code)
            for line in code.split("\n"):
                yield token_event(line + "\n")

            # Execute in REPL
            result = repl.execute(code)

            # Emit tool events for any tool invocations in this turn.
            # REQUEST-LOCAL: repl._invoked_tools, not
            # repl.tool_registry.get_invocation_log() — the registry is
            # process-global, so the cursor-slice idiom below streamed OTHER
            # concurrent requests' tool events to this client (and, now that the
            # registry ring is bounded, eviction would make the cursor slice wrong).
            # _invoked_tools is append-only for the life of this one REPL, so the
            # cursor is exact.
            inv_log = list(getattr(repl, "_invoked_tools", None) or [])
            for inv in inv_log[prev_tool_count:]:
                yield tool_start_event(inv.tool_name)
                yield tool_end_event(inv.tool_name, int(inv.elapsed_ms), inv.success)
            prev_tool_count = len(inv_log)

            # Check model-initiated routing artifacts
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
                    # Emit turn end before role switch
                    turn_elapsed_ms = int((time.perf_counter() - turn_start_time_inner) * 1000)
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
                    state.record_escalation(str(role_history[-2]), str(current_role))
                    continue  # Next turn with new role

            # Log delegation outcomes
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

            # Emit turn end
            turn_elapsed_ms = int((time.perf_counter() - turn_start_time_inner) * 1000)
            yield turn_end_event(tokens=len(code), elapsed_ms=turn_elapsed_ms)

            # Check for completion
            if result.is_final:
                tool_outputs = repl.artifacts.get("_tool_outputs", [])
                stream_answer = _resolve_answer(result, tool_outputs=tool_outputs)

                # MemRL-informed quality review gate (blocking, streaming parity)
                review_gate = evaluate_review_gate(state, current_role, stream_answer)
                verdict_status = None
                verdict_ms = None
                revision_applied = None
                revision_ms = None
                if review_gate.triggered:
                    _t_review = time.perf_counter()
                    verdict, verdict_status = _architect_verdict_with_status(
                        question=request.prompt,
                        answer=stream_answer,
                        primitives=primitives,
                    )
                    verdict_ms = review_ms_since(_t_review)
                    revision_applied = False
                    if verdict and verdict.upper().startswith("WRONG"):
                        corrections = (
                            verdict.split(":", 1)[1].strip() if ":" in verdict else verdict
                        )
                        _t_review = time.perf_counter()
                        revised = _fast_revise(
                            question=request.prompt,
                            original_answer=stream_answer,
                            corrections=corrections,
                            primitives=primitives,
                        )
                        revision_ms = review_ms_since(_t_review)
                        revision_applied = revised != stream_answer
                        stream_answer = revised
                record_review_gate(
                    task_id,
                    current_role,
                    gate=review_gate,
                    path=REVIEW_PATH_LEGACY_STREAM,
                    verdict_status=verdict_status,
                    revision_applied=revision_applied,
                    verdict_ms=verdict_ms,
                    revision_ms=revision_ms,
                )

                yield final_event(stream_answer)
                break

            # Handle errors with escalation policy
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
                    cot_prefix = decision.config_override.get("cot_prefix", "")
                    if cot_prefix:
                        escalation_prompt = cot_prefix + root_prompt
                    log.info("Think-harder in legacy stream: %s", decision.reason)
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
                    state.record_escalation(str(role_history[-2]), str(current_role))
                elif decision.action == EscalationAction.FAIL:
                    yield error_event(f"[FAILED: {decision.reason}]")
                    break
            else:
                consecutive_failures = 0
                last_error = ""
                last_output = result.output

        # Log completion (MemRL) - success if we got a final answer
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
            score_completed_task(state, task_id)
        yield done_event()

    # Use SSE utilities for response (handles sse-starlette vs manual fallback)
    return create_sse_response(generate())
