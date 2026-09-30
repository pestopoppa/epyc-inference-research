"""Pipeline stages 1-3 + 5: routing, preprocessing, backend init, plan review."""

from __future__ import annotations

import logging
import os
import time
import urllib.request
import uuid

from fastapi import HTTPException

from src.api.models import ChatRequest
from src.api.services.memrl import ensure_memrl_initialized
from src.config import get_config
from src.constants import TASK_IR_OBJECTIVE_LEN
from src.features import features
from src.llm_primitives import LLMPrimitives
from src.runtime import routing_stage_timing
from src.task_ir import canonicalize_task_ir

from src.api.routes.chat_pipeline.routing_decision import (
    _TIER_COST_WEIGHTS as _TIER_COST_WEIGHTS,  # noqa: F401 - compatibility re-export
    apply_failure_veto,
    apply_ingest_triviality_guard,
    assess_difficulty,
    assess_factual_risk,
    classify_trinity_role,
    derive_task_type_from_route,
    estimate_routing_cost,
    extract_skill_ids,
    fence_nonserving_route,
    log_routing_start,
    resolve_timeout,
    select_initial_route,
)
from src.api.routes.chat_review import (
    _apply_plan_review,
    _architect_plan_review,
    _needs_plan_review,
    _plan_review_should_abort,
    _store_plan_review_episode,
)
from src.api.routes.chat_routing import _classify_and_route
from src.api.routes.chat_routing import _heuristic_role_priors
from src.api.routes.chat_utils import (
    RoutingResult,
)
from src.api.structured_logging import task_extra

log = logging.getLogger(__name__)

_XMAS_MAX_EVIDENCE_LATENCY_RATIO = 1.10
_XMAS_MIN_SPEEDUP_FOR_TIE = 0.95
_EVAL_BATCH_FRONTDOOR_URL_ENV = "ORCHESTRATOR_EVAL_BATCH_FRONTDOOR_URL"
_EVAL_BATCH_FRONTDOOR_ROLES_ENV = "ORCHESTRATOR_EVAL_BATCH_FRONTDOOR_ROLES"
_EVAL_BATCH_FRONTDOOR_HEALTH_TIMEOUT_S = 0.5
_DEFAULT_EVAL_BATCH_FRONTDOOR_ROLES = (
    "frontdoor",
    "coder_escalation",
    "worker_summarize",
)


def _numeric_metric(metrics: dict, key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _xmas_quality(metrics: dict) -> float | None:
    correct = _numeric_metric(metrics, "correct")
    if correct is not None:
        return correct
    return _numeric_metric(metrics, "accuracy")


def _eval_batch_frontdoor_url() -> str | None:
    raw = os.environ.get(_EVAL_BATCH_FRONTDOOR_URL_ENV, "").strip()
    if raw:
        return raw.rstrip("/")
    try:
        from scripts.server.stack_manifest import PORT_MAP

        port = PORT_MAP.get("eval_batch_frontdoor")
        if isinstance(port, int) and port > 0:
            return f"http://localhost:{port}"
    except Exception:
        return None
    return None


def _eval_batch_frontdoor_roles() -> tuple[str, ...]:
    raw = os.environ.get(_EVAL_BATCH_FRONTDOOR_ROLES_ENV, "").strip()
    if not raw:
        return _DEFAULT_EVAL_BATCH_FRONTDOOR_ROLES
    return tuple(role.strip() for role in raw.split(",") if role.strip())


def _eval_batch_frontdoor_healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(
            f"{url.rstrip('/')}/health",
            timeout=_EVAL_BATCH_FRONTDOOR_HEALTH_TIMEOUT_S,
        ) as resp:
            return 200 <= int(getattr(resp, "status", 0)) < 300
    except Exception:
        return False


def _eval_batch_frontdoor_fits(request: ChatRequest) -> bool:
    """Whether the auxiliary lane can admit this request without truncation.

    The corrected lane has two slots over a 32K total context. Long EvalTower
    prompts must remain on the primary half fleet, whose per-slot context is
    much larger, rather than making the auxiliary look fast by rejecting or
    truncating them.
    """
    try:
        from scripts.server.stack_manifest import (
            LAUNCH_CONTEXT_TOKENS,
            resolve_slots,
        )

        slots = resolve_slots(
            "eval_batch_frontdoor", "eval_batch_frontdoor"
        ).slots
        per_slot_tokens = int(LAUNCH_CONTEXT_TOKENS["eval_batch_frontdoor"]) // max(
            1, int(slots)
        )
    except Exception:
        return False
    prompt_tokens = max(1, (len(str(request.prompt or "")) + 3) // 4)
    output_reserve = max(1, int(request.max_tokens or 4096))
    return prompt_tokens + output_reserve <= per_slot_tokens


def _server_urls_with_eval_batch_frontdoor(
    request: ChatRequest,
    server_urls: dict[str, str],
) -> tuple[dict[str, str], bool]:
    """Optionally route eval-batch traffic to the warm P-BENCH-3 lane.

    This is deliberately default-off and request-specific: normal primitives are
    cached/reused, while eval-batch rewrites create a fresh primitive map so an
    opt-in batch endpoint cannot leak into interactive traffic.
    """
    if request.server_urls or not request.real_mode:
        return server_urls, False
    if request.workload_class != "eval_batch":
        return server_urls, False
    if not features().eval_batch_serving:
        return server_urls, False
    if not _eval_batch_frontdoor_fits(request):
        log.info(
            "eval-batch request exceeds auxiliary per-slot context; keeping primary split fleet"
        )
        return server_urls, False

    url = _eval_batch_frontdoor_url()
    if not url:
        log.warning("eval_batch_serving enabled but no eval-batch frontdoor URL is configured")
        return server_urls, False
    if not _eval_batch_frontdoor_healthy(url):
        log.warning("eval_batch_serving enabled but eval-batch frontdoor is not healthy: %s", url)
        return server_urls, False

    rewritten = dict(server_urls)
    changed = False
    for role in _eval_batch_frontdoor_roles():
        if role in rewritten:
            rewritten[role] = url
            changed = True
    return (rewritten, changed) if changed else (server_urls, False)


def _xmas_latency_ratio(suggested: dict, incumbent: dict) -> float | None:
    suggested_latency = _numeric_metric(suggested, "wall_mean_s")
    incumbent_latency = _numeric_metric(incumbent, "wall_mean_s")
    if suggested_latency is None or incumbent_latency is None or incumbent_latency <= 0:
        return None
    return suggested_latency / incumbent_latency


def _xmas_evidence_allows_incumbent_replacement(
    xmas_meta: dict,
    previous_role: str,
    suggested_role: str,
) -> tuple[bool, str]:
    """Return whether cell evidence justifies replacing the incumbent route."""
    candidate_metrics = xmas_meta.get("candidate_metrics") or {}
    if not isinstance(candidate_metrics, dict):
        return False, "missing_candidate_metrics"
    suggested_metrics = candidate_metrics.get(suggested_role)
    if not isinstance(suggested_metrics, dict):
        return False, "suggested_role_not_evaluated"
    incumbent_metrics = candidate_metrics.get(previous_role)
    if not isinstance(incumbent_metrics, dict):
        return False, "incumbent_role_not_evaluated"

    suggested_quality = _xmas_quality(suggested_metrics)
    incumbent_quality = _xmas_quality(incumbent_metrics)
    if suggested_quality is None or incumbent_quality is None:
        return False, "missing_quality_evidence"

    latency_ratio = _xmas_latency_ratio(suggested_metrics, incumbent_metrics)
    if latency_ratio is not None and latency_ratio > _XMAS_MAX_EVIDENCE_LATENCY_RATIO:
        return False, "evidence_latency_regression"

    if suggested_quality > incumbent_quality:
        return True, "evidence_quality_lift"
    if suggested_quality < incumbent_quality:
        return False, "evidence_quality_regression"
    if latency_ratio is not None and latency_ratio < _XMAS_MIN_SPEEDUP_FOR_TIE:
        return True, "evidence_speed_lift"
    return False, "evidence_no_lift_over_incumbent"


def _apply_xmas_enforce_override(
    request: ChatRequest,
    routing_decision: list,
    routing_strategy: str,
    xmas_meta: dict | None,
) -> tuple[list, str]:
    """Apply a guarded X-MAS route override when metadata is enforce-ready."""
    if not xmas_meta or xmas_meta.get("mode") != "enforce":
        return routing_decision, routing_strategy
    if request.force_role:
        xmas_meta["applied"] = False
        xmas_meta["apply_reason"] = "forced_role"
        return routing_decision, routing_strategy
    if xmas_meta.get("winner_table_status") != "loaded":
        xmas_meta["applied"] = False
        xmas_meta["apply_reason"] = "winner_table_not_loaded"
        return routing_decision, routing_strategy
    if not xmas_meta.get("is_confident"):
        xmas_meta["applied"] = False
        xmas_meta["apply_reason"] = "low_confidence"
        return routing_decision, routing_strategy
    suggested_role = xmas_meta.get("suggested_role")
    if not suggested_role:
        xmas_meta["applied"] = False
        xmas_meta["apply_reason"] = "no_suggested_role"
        return routing_decision, routing_strategy

    previous_role = str(routing_decision[0]) if routing_decision else ""
    xmas_meta["previous_role"] = previous_role
    if suggested_role == previous_role:
        xmas_meta["applied"] = False
        xmas_meta["apply_reason"] = "already_selected"
        return routing_decision, routing_strategy
    evidence_allows, evidence_reason = _xmas_evidence_allows_incumbent_replacement(
        xmas_meta,
        previous_role,
        str(suggested_role),
    )
    xmas_meta["incumbent_role"] = previous_role
    xmas_meta["incumbent_policy"] = "evidence_lift_or_speedup"
    xmas_meta["incumbent_reason"] = evidence_reason
    if not evidence_allows:
        xmas_meta["applied"] = False
        xmas_meta["apply_reason"] = evidence_reason
        return routing_decision, routing_strategy

    xmas_meta["applied"] = True
    xmas_meta["apply_reason"] = evidence_reason
    return [str(suggested_role)], f"xmas_enforce:{routing_strategy}"


def _route_request(
    request: ChatRequest,
    state,
    routing_path: str = routing_stage_timing.PATH_CHAT,
) -> RoutingResult:
    """Determine routing decision, strategy, and task metadata.

    Produces a RoutingResult that captures all routing decisions made
    before execution begins. Includes failure graph veto and MemRL logging.

    RI-16: starts the request's routing-stage timing (``routing_path`` labels the
    pipeline); the stage timers are observability-only and never alter the route.
    """
    route_start = time.perf_counter()
    task_id = f"chat-{uuid.uuid4().hex[:8]}"
    routing_stage_timing.begin(task_id, routing_path)
    task_ir = canonicalize_task_ir(
        {
            "task_type": "chat",
            "objective": request.prompt[:TASK_IR_OBJECTIVE_LEN],
            "priority": "interactive",
            "context_preview": request.context or "",
            "routing_preferences": request.routing_preferences,
        }
    )

    use_mock = request.mock_mode and not request.real_mode
    has_image = bool(request.image_path or request.image_base64)
    with routing_stage_timing.timed("priors"):
        heuristic_priors = _heuristic_role_priors(
            request.prompt,
            request.context or "",
            has_image=has_image,
        )

    # Initialize MemRL early for real_mode to enable HybridRouter
    if request.real_mode and not use_mock:
        with routing_stage_timing.timed("memrl_init"):
            ensure_memrl_initialized(state)

    # Determine routing using HybridRouter if available, otherwise rules
    with routing_stage_timing.timed("route"):
        routing_decision, routing_strategy, skill_context = select_initial_route(
            request,
            state,
            task_ir,
            use_mock,
            heuristic_priors,
            _classify_and_route,
        )
    xmas_meta = None
    try:
        from src.classifiers.xmas_routing import build_xmas_routing_metadata

        with routing_stage_timing.timed("xmas"):
            xmas_meta = build_xmas_routing_metadata(
                request.prompt,
                request.context or "",
            )
        routing_decision, routing_strategy = _apply_xmas_enforce_override(
            request,
            routing_decision,
            routing_strategy,
            xmas_meta,
        )
    except Exception:
        xmas_meta = None

    if not use_mock:
        routing_decision, routing_strategy = fence_nonserving_route(
            routing_decision,
            routing_strategy,
        )

    role_for_signals = str(routing_decision[0]) if routing_decision else ""
    with routing_stage_timing.timed("factual_risk"):
        _factual_risk_score, _factual_risk_band = assess_factual_risk(
            request.prompt,
            role_for_signals,
            task_id,
        )

    # Failure graph veto — revert high-risk specialists to frontdoor
    # RI-5: Veto threshold modulated by factual-risk band.
    # High factual risk → lower veto threshold (more conservative routing).
    # Low factual risk → higher threshold (allow specialist attempts).
    with routing_stage_timing.timed("failure_veto"):
        routing_decision, routing_strategy = apply_failure_veto(
            state,
            routing_decision,
            routing_strategy,
            _factual_risk_band,
            task_id,
            has_image=has_image,
        )

    # Difficulty-signal scoring (shadow/enforce mode only — no-op when mode is "off")
    role_for_signals = str(routing_decision[0]) if routing_decision else ""
    with routing_stage_timing.timed("difficulty"):
        _difficulty_score, _difficulty_band = assess_difficulty(
            request.prompt,
            role_for_signals,
            task_id,
        )

    # Ingest-triviality guard (opt-in): keep trivially-easy short prompts off the
    # ingest_long_context 80B specialist. Reuses _difficulty_band; no-op when the
    # ingest_triviality_guard feature flag is off. Placed after the failure veto
    # and difficulty scoring so it sees the final pre-execution route.
    routing_decision, routing_strategy = apply_ingest_triviality_guard(
        request,
        routing_decision,
        routing_strategy,
        _difficulty_band,
        task_id,
    )

    # TR-3.2: Trinity tri-role classification (shadow mode). Always populates
    # `assigned_role` regardless of the ROLE_AWARE_ROUTING flag — TR-4 gates
    # acting on the role; TR-3.3 uses shadow telemetry to decide promotion.
    with routing_stage_timing.timed("trinity"):
        _assigned_role = classify_trinity_role(request, routing_decision, task_id)

    # Estimated cost (tier weight × prompt tokens / 1M — relative units for Pareto)
    _estimated_cost = estimate_routing_cost(request, state, routing_decision)

    sample_key = _factual_risk_sample_key(request, task_id)
    try:
        from src.classifiers.factual_risk import get_mode as _fr_get_mode

        _factual_risk_mode = _fr_get_mode(
            role=role_for_signals,
            sample_key=sample_key,
        )
    except Exception:
        _factual_risk_mode = ""

    # RI-16: the pre-execution routing decision is complete here; close its wall
    # timer so the routing_decision event below carries route_total.
    timing = routing_stage_timing.current()
    if timing is not None and timing.task_id == task_id:
        timing.record("route_total", (time.perf_counter() - route_start) * 1000.0)

    # Log task start (MemRL integration). This must happen after all shadow
    # signals are computed so TR-3.3/W7 telemetry is durable in progress JSONL.
    log_routing_start(
        request,
        state,
        task_id,
        task_ir,
        routing_decision,
        routing_strategy,
        heuristic_priors,
        _factual_risk_score,
        _factual_risk_band,
        _factual_risk_mode,
        _difficulty_score,
        _difficulty_band,
        _estimated_cost,
        _assigned_role,
        xmas_meta,
    )

    # Compute role-specific timeout
    timeout_s = resolve_timeout(request, routing_decision)

    # Detect tool requirement for forced tool use
    from src.api.routes.chat_routing import detect_tool_requirement

    tool_required, tool_hint = detect_tool_requirement(request.prompt)

    # Extract skill IDs from skill-augmented routing results
    skill_ids = extract_skill_ids(skill_context, state)

    # WS-3: Derive task_type from routed role so cascading tool policy can deny
    # web tools for reasoning domains. Without this, task_type stays "chat" and
    # NO_WEB_TASK_TYPES never matches.
    derive_task_type_from_route(task_ir, routing_decision)

    return RoutingResult(
        task_id=task_id,
        task_ir=task_ir,
        use_mock=use_mock,
        routing_decision=routing_decision,
        routing_strategy=routing_strategy,
        timeout_s=timeout_s,
        tool_required=tool_required,
        tool_hint=tool_hint,
        skill_context=skill_context,
        skill_ids=skill_ids,
        factual_risk_score=_factual_risk_score,
        factual_risk_band=_factual_risk_band,
        factual_risk_mode=_factual_risk_mode,
        difficulty_score=_difficulty_score,
        difficulty_band=_difficulty_band,
        estimated_cost=_estimated_cost,
        assigned_role=_assigned_role,
        xmas_meta=xmas_meta,
    )


# ── Stage 2: Preprocessing ──────────────────────────────────────────────


def _preprocess(request: ChatRequest, state, routing: RoutingResult) -> None:
    """Apply input formalization if enabled. Mutates request.context and routing."""
    if (
        features().input_formalizer
        and request.real_mode
        and not routing.use_mock
        and routing.routing_strategy not in ("mock",)
    ):
        from src.formalizer import should_formalize_input, formalize_prompt, inject_formalization

        should_fml, problem_hint = should_formalize_input(request.prompt)
        if should_fml:
            fml_result = formalize_prompt(request.prompt, problem_hint, state.registry)
            if fml_result.success:
                request.context = inject_formalization(
                    request.prompt, request.context or "", fml_result.ir_json
                )
                routing.formalization_applied = True
                log.info(
                    "Input formalization: %s (%.1fs, %s)",
                    problem_hint,
                    fml_result.elapsed_seconds,
                    fml_result.model_role,
                    extra=task_extra(
                        task_id=routing.task_id,
                        stage="preprocess",
                        latency_ms=fml_result.elapsed_seconds * 1000,
                    ),
                )


# ── Stage 3: Backend initialization ─────────────────────────────────────


def _init_primitives(request: ChatRequest, state) -> LLMPrimitives:
    """Initialize LLM backends for real inference.

    Reuses shared LLMPrimitives instance for connection pooling when possible.
    Raises HTTPException(503) if backends unavailable.
    """
    if request.real_mode:
        server_urls = request.server_urls or get_config().server_urls.as_dict()
        server_urls, eval_batch_urls = _server_urls_with_eval_batch_frontdoor(
            request,
            server_urls,
        )
        request_specific_urls = bool(request.server_urls) or eval_batch_urls

        if (
            hasattr(state, "_real_primitives")
            and state._real_primitives is not None
            and not request_specific_urls
        ):
            primitives = state._real_primitives
            primitives.reset_counters()
        else:
            try:
                primitives = LLMPrimitives(
                    mock_mode=False,
                    server_urls=server_urls,
                    registry=state.registry,
                    health_tracker=getattr(state, "health_tracker", None),
                    admission_controller=getattr(state, "admission", None),
                    num_slots=get_config().server.num_slots,
                    server_urls_source=("request" if request_specific_urls else "config"),
                )
                if not request_specific_urls:
                    state._real_primitives = primitives
            except Exception as e:
                raise HTTPException(
                    status_code=503,
                    detail=f"Failed to initialize real mode backends: {e}",
                )

        primitives.cache_prompt = request.cache_prompt

        if not primitives._backends:
            raise HTTPException(
                status_code=503,
                detail="No backends available. Ensure llama-server is running on configured ports.",
            )
    else:
        primitives = LLMPrimitives(mock_mode=False, registry=state.registry)
        if primitives.model_server is None:
            raise HTTPException(
                status_code=503,
                detail="Real inference not available: no model server configured",
            )

    return primitives


# ── Stage 5: Plan review gate ───────────────────────────────────────────


def _plan_review_gate(
    request: ChatRequest,
    routing: RoutingResult,
    primitives: LLMPrimitives,
    state,
) -> list | None:
    """Run architect plan review if applicable. Returns modified routing_decision or None."""
    # TD-5: observability-only typed-decision shadow next to the incumbent
    # route. `_route_request` decides the route before request-scoped
    # primitives exist, so this gate is the first point on BOTH pipeline paths
    # (chat.py and stream_adapter.py) where the pre-review decision and the
    # primitives coexist. It submits and returns; it never mutates routing,
    # never raises, and is skipped entirely — no import, no call — when off.
    from src.runtime import quiescence as _quiescence

    # INF-78 OAB-3 (R2): the shadow runs on a daemon executor after the reply; a
    # quiescent_after request skips it (suppress() is only consulted when the flag is on).
    if features().typed_decisions_shadow and not _quiescence.suppress("typed_decisions_shadow"):
        from src.typed_decisions.shadow import submit_route_shadow

        submit_route_shadow(
            primitives,
            prompt=request.prompt,
            context=request.context or "",
            incumbent_roles=routing.routing_decision,
            strategy=routing.routing_strategy,
            task_id=routing.task_id,
        )

    plan_review_result = None
    # RI-3: Force plan review when factual risk is high, regardless of complexity heuristics.
    # High-risk prompts need architect oversight to catch factual errors.
    _current_role = routing.routing_decision[0] if routing.routing_decision else ""
    factual_risk_mode = str(getattr(routing, "factual_risk_mode", "") or "")
    if not factual_risk_mode:
        from src.classifiers.factual_risk import get_mode as _fr_get_mode

        factual_risk_mode = _fr_get_mode(
            role=_current_role,
            sample_key=_factual_risk_sample_key(
                request,
                str(getattr(routing, "task_id", "") or ""),
            ),
        )
    risk_forced = routing.factual_risk_band == "high" and factual_risk_mode == "enforce"
    needs_review = _needs_plan_review(routing.task_ir, routing.routing_decision, state)
    if request.real_mode and features().plan_review and (needs_review or risk_forced):
        plan_review_result = _architect_plan_review(
            routing.task_ir, routing.routing_decision, primitives, state, routing.task_id
        )
        if plan_review_result and plan_review_result.decision == "drop":
            # A rejected plan falls back to the normal route. In particular,
            # synthesized one-step plans often only restate the objective; the
            # architect should be able to discard that scaffolding without
            # refusing the user's underlying task.
            routing.task_ir.pop("plan", None)
        elif (
            plan_review_result
            and plan_review_result.decision != "ok"
            and not _plan_review_should_abort(plan_review_result)
        ):
            routing.routing_decision = _apply_plan_review(
                routing.routing_decision, plan_review_result
            )
        if plan_review_result:
            _store_plan_review_episode(state, routing.task_id, routing.task_ir, plan_review_result)
    return plan_review_result


def _factual_risk_sample_key(request: ChatRequest, task_id: str) -> str:
    """Stable key for RI-10 canary arm assignment.

    Caller-provided request IDs let controlled canary accrual jobs choose a
    deterministic arm before dispatch. Ordinary traffic keeps the existing
    server task-id behavior.
    """
    request_id = str(getattr(request, "request_id", "") or "").strip()
    return request_id or str(task_id or "")
