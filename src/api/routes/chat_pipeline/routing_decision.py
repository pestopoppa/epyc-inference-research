"""Routing decision helpers for the chat pipeline."""

from __future__ import annotations

import logging
import math
from pathlib import Path

from src.api.models import ChatRequest
from src.api.routes.chat_routing import _classify_and_route
from src.api.routes.chat_utils import LONG_CONTEXT_CONFIG, role_timeout_for
from src.api.structured_logging import task_extra
from src.backends.context_limits import (
    CONSERVATIVE_CHARS_PER_TOKEN,
    ROUGH_CHARS_PER_TOKEN,
    context_overflow_roles,
    get_context_limit_resolver,
)
from src.features import features
from src.roles import Role
from src.runtime import routing_stage_timing

log = logging.getLogger(__name__)

_TIER_COST_WEIGHTS: dict[str, float] = {
    "A": 10.0,
    "B": 3.0,
    "C": 1.0,
    "D": 0.2,
}

_INGRESS_ROLE_ALIASES = frozenset({"worker_coder", "worker_code"})

# Upper bound on the image reference persisted into the routing-decision
# progress event. Real image paths are ~90 chars; the cap only fences a
# pathological caller from writing an unbounded string into every log line.
MAX_IMAGE_PATH_LEN = 512

_DEFAULT_STACK_PRIORS_PATH = (
    Path(__file__).resolve().parents[4] / "orchestration" / "derived" / "stack_priors.yaml"
)

# Declarative-source fallback ONLY. The authoritative set of vision-only roles
# is read from generated stack priors (launch mode == "vision") via
# ``vision_serving.vision_roles``; these two known VL-only role names are used
# solely when that artifact is unreadable (mirrors vision_serving's own
# LEGACY_VISION_ROLES). Do not treat this literal as the source of truth.
_FALLBACK_VISION_ONLY_ROLES = frozenset({"worker_vision", "vision_escalation"})


def _vision_only_roles() -> frozenset[str]:
    """Return roles that serve ONLY multimodal VL traffic (declarative source).

    A VL-only llama-server rejects a text-only ``/completion`` with HTTP 400
    (observed: a longbench TEXT question misrouted to ``worker_vision:8086``).
    Both directions of the modality fence key on this set: text requests are
    steered away from these roles, and image requests routed to them are exempt
    from the failure veto (they have no valid text fallback).
    """
    try:
        from src.api.routes.vision_serving import vision_roles as _stack_prior_vision_roles

        return _stack_prior_vision_roles(_DEFAULT_STACK_PRIORS_PATH)
    except Exception:
        return _FALLBACK_VISION_ONLY_ROLES


def _fence_text_off_vision_roles(
    routing_decision: list, routing_strategy: str
) -> tuple[list, str]:
    """Strip vision-only roles from a TEXT request's candidate route.

    The learned/hybrid router and the rules classifier can both emit a
    vision-only role for a text-only prompt; routing text there yields an HTTP
    400 from the VL server (blind/failed answers that poison the eval). Callers
    invoke this ONLY for requests with no image data and no explicit/forced role
    (forced/explicit routing is the caller's responsibility and bypasses the
    fence). If stripping empties the route, fall back to the frontdoor.
    """
    if not routing_decision:
        return routing_decision, routing_strategy
    vision_only = _vision_only_roles()
    filtered = [r for r in routing_decision if str(r) not in vision_only]
    if len(filtered) == len(routing_decision):
        return routing_decision, routing_strategy
    if not filtered:
        log.warning(
            "Modality fence: text request routed to vision-only %s → frontdoor",
            [str(r) for r in routing_decision],
        )
        return [str(Role.FRONTDOOR)], f"{routing_strategy}:vision_fenced"
    log.warning(
        "Modality fence: stripped vision-only role(s) from text route %s",
        [str(r) for r in routing_decision],
    )
    return filtered, f"{routing_strategy}:vision_fenced"


def normalize_ingress_role(role: object) -> object:
    """Normalize externally supplied role labels before config lookup."""
    if not isinstance(role, str):
        return role
    normalized = Role.from_string(role)
    if normalized is not None:
        return normalized
    if role in _INGRESS_ROLE_ALIASES:
        return Role.WORKER_GENERAL
    return role


def fence_nonserving_route(
    routing_decision: list, routing_strategy: str
) -> tuple[list, str]:
    """Fail closed when an automatic route names a non-serving action.

    Episodic control-plane actions and model-serving routes coexist in the
    same database. The learned router filters its own retrievals, but this
    pipeline boundary also protects against a future router or override
    returning an action that has no realized serving endpoint.
    """
    try:
        from src.registry.stack_priors import live_stack_role_ids

        live_roles = set(live_stack_role_ids())
    except Exception:
        live_roles = set()
    if not live_roles:
        # The generated serving contract is the only authority for routability.
        # If it is unavailable, fail closed to the always-present ingress lane
        # instead of reconstructing a stale allow-list from the broader Role
        # enum, which intentionally also contains historical and draft labels.
        live_roles = {Role.FRONTDOOR.value}

    canonical: list[str] = []
    for raw_role in routing_decision:
        role = Role.from_string(str(raw_role).strip())
        if role is None or role.value not in live_roles:
            log.error(
                "Non-serving route fenced: strategy=%s route=%s → frontdoor",
                routing_strategy,
                [str(candidate) for candidate in routing_decision],
            )
            return [str(Role.FRONTDOOR)], f"{routing_strategy}:nonserving_fenced"
        canonical.append(role.value)

    if not canonical:
        log.error("Empty route fenced: strategy=%s → frontdoor", routing_strategy)
        return [str(Role.FRONTDOOR)], f"{routing_strategy}:nonserving_fenced"
    return canonical, routing_strategy


def assess_factual_risk(prompt: str, role: str, task_id: str) -> tuple[float, str]:
    """Return factual-risk score and band, falling back to no-risk on failure."""
    try:
        from src.classifiers.factual_risk import assess_risk, get_configured_mode

        if get_configured_mode() == "off":
            return 0.0, ""
        result = assess_risk(prompt, role=role)
        log.info(
            "Factual risk: score=%.3f band=%s (raw=%.3f adj=%.1f)",
            result.adjusted_risk_score,
            result.risk_band,
            result.risk_score,
            result.role_adjustment,
            extra=task_extra(
                task_id=task_id,
                stage="routing",
                strategy="factual_risk",
            ),
        )
        return result.adjusted_risk_score, result.risk_band
    except Exception as exc:
        log.debug("Factual risk scoring skipped: %s", exc)
        return 0.0, ""


def _ceil_div(chars: int, chars_per_token: float) -> int:
    return int(math.ceil(max(0, chars) / chars_per_token))


def _long_context_threshold_tokens() -> int:
    """Specialist-routing threshold in tokens.

    ``threshold_tokens`` wins when configured; otherwise the character knob is
    converted at ROUGH_CHARS_PER_TOKEN (20,000 chars → 5,000 tokens).
    """
    explicit = LONG_CONTEXT_CONFIG.get("threshold_tokens")
    if isinstance(explicit, int) and not isinstance(explicit, bool) and explicit > 0:
        return explicit
    chars = max(1, int(LONG_CONTEXT_CONFIG.get("threshold_chars", 20_000)))
    return max(1, int(chars // ROUGH_CHARS_PER_TOKEN))


# Decode budget reserved for a request that states no max_tokens (the
# chat-completions backend path uses the same default).
DEFAULT_ROUTING_DECODE_RESERVE_TOKENS = 4096


def _needs_long_context(effective_chars: int, max_tokens: int | None) -> tuple[bool, str]:
    """(route to the long-context role?, human-readable reason for the log)."""
    prompt_tokens = _ceil_div(effective_chars, CONSERVATIVE_CHARS_PER_TOKEN)
    decode = max_tokens if isinstance(max_tokens, int) and max_tokens > 0 else DEFAULT_ROUTING_DECODE_RESERVE_TOKENS
    needed = prompt_tokens + decode
    frontdoor = str(Role.FRONTDOOR)
    try:
        limit = get_context_limit_resolver().limit_for_role(frontdoor)
    except Exception:
        log.debug("Frontdoor context limit lookup failed", exc_info=True)
        limit = None
    if limit is not None:
        fits = limit.fits(prompt_tokens, decode)
        return (not fits), (
            f"~{prompt_tokens} prompt + {decode} decode = {needed} tokens "
            f"{'fits' if fits else 'does not fit'} {frontdoor} per-request n_ctx "
            f"{limit.per_request_n_ctx} ({limit.source})"
        )
    threshold = _long_context_threshold_tokens()
    rough = _ceil_div(effective_chars, ROUGH_CHARS_PER_TOKEN)
    return rough > threshold, (
        f"{frontdoor} context limit unknown; conservative fallback: ~{rough} tokens "
        f"({effective_chars} chars) {'>' if rough > threshold else '<='} {threshold}-token threshold"
    )


def _long_context_target(effective_chars: int) -> str:
    """ingest_long_context, unless its live per-request context cannot hold the
    request and a configured larger-context role can (context_overflow_roles)."""
    default_role = str(Role.INGEST_LONG_CONTEXT)
    needed = _ceil_div(effective_chars, CONSERVATIVE_CHARS_PER_TOKEN)
    try:
        resolver = get_context_limit_resolver()
        limit = resolver.limit_for_role(default_role)
        if limit is None or limit.fits(needed):
            return default_role
        alternative = resolver.larger_context_role(
            needed, candidates=context_overflow_roles(), exclude={default_role}
        )
    except Exception:
        log.debug("Long-context capacity check failed; keeping %s", default_role, exc_info=True)
        return default_role
    if alternative is not None:
        alt_role, alt_limit = alternative
        log.warning(
            "Long-context request ~%d tokens exceeds %s per-request n_ctx %d (%s); "
            "routing to %s (n_ctx %d)",
            needed, default_role, limit.per_request_n_ctx, limit.source,
            alt_role, alt_limit.per_request_n_ctx,
        )
        return alt_role
    log.warning(
        "Long-context request ~%d tokens exceeds %s per-request n_ctx %d (%s) and no "
        "larger-context role is configured; the server will refuse it and the caller "
        "gets a typed ContextOverflowError",
        needed, default_role, limit.per_request_n_ctx, limit.source,
    )
    return default_role


def select_initial_route(
    request: ChatRequest,
    state,
    task_ir: dict,
    use_mock: bool,
    heuristic_priors: dict[str, float],
    classify_and_route=_classify_and_route,
) -> tuple[list, str, str]:
    """Select initial route before risk veto and telemetry enrichment."""
    skill_context = ""
    if use_mock:
        role = normalize_ingress_role(request.role) if request.role else Role.FRONTDOOR
        return [role], "mock", skill_context
    if request.force_role:
        return [normalize_ingress_role(request.force_role)], "forced", skill_context
    if request.role and request.role not in ("", "frontdoor"):
        return [normalize_ingress_role(request.role)], "explicit", skill_context
    if request.image_path or request.image_base64:
        return ["worker_vision"], "vision_input", skill_context
    # Capacity/specialist fence: learned routing only sees a bounded task-IR
    # objective and may therefore choose frontdoor for a request whose full
    # prompt is hundreds of thousands of characters.  That failure mode sends
    # long-context prefill to a latency-oriented worker and can outlive the
    # request budget.  Preserve explicit/forced and image routing above, but
    # keep ordinary oversized text on the role provisioned for it.
    #
    # 2026-09-24 (operator decision): route by the LIVE per-request limit. A
    # request goes to the long-context role only when its prompt + reserved
    # decode budget will not fit the frontdoor's live per-request n_ctx
    # (ContextLimitResolver: /props, registry fallback). Only when that limit
    # is unknown does the old conservative threshold (the autopilot knob
    # `chat.long_context_threshold_chars`, in tokens) decide.
    long_context_enabled = bool(LONG_CONTEXT_CONFIG.get("enabled", True))
    effective_chars = len(request.prompt or "") + len(request.context or "")
    if long_context_enabled:
        use_long, detail = _needs_long_context(effective_chars, request.max_tokens)
        if use_long:
            target = _long_context_target(effective_chars)
            log.info("Long-context routing guard → %s: %s", target, detail)
            return [target], "long_context_guard", skill_context
        if effective_chars > 0:
            log.debug("Long-context routing guard: stays on normal routing: %s", detail)
    # Below this point the request has NO image data and NO forced/explicit
    # role (those returned above). The learned/hybrid router and the rules
    # classifier can still emit a vision-only role for a text prompt, which a
    # VL server rejects with HTTP 400 — fence text traffic off those roles.
    if state.hybrid_router and request.real_mode:
        if hasattr(state.hybrid_router, "route_with_skills") and features().skillbank:
            routing_decision, routing_strategy, skill_context = (
                state.hybrid_router.route_with_skills(task_ir)
            )
            routing_decision, routing_strategy = _fence_text_off_vision_roles(
                routing_decision, routing_strategy
            )
            return routing_decision, routing_strategy, skill_context
        routing_decision, routing_strategy = state.hybrid_router.route(
            task_ir,
            priors=heuristic_priors,
        )
        routing_decision, routing_strategy = _fence_text_off_vision_roles(
            routing_decision, routing_strategy
        )
        return routing_decision, routing_strategy, skill_context

    classified_role, routing_strategy = classify_and_route(
        request.prompt,
        request.context or "",
        has_image=bool(request.image_path or request.image_base64),
    )
    routing_decision, routing_strategy = _fence_text_off_vision_roles(
        [classified_role], routing_strategy
    )
    return routing_decision, routing_strategy, skill_context


def apply_failure_veto(
    state,
    routing_decision: list,
    routing_strategy: str,
    factual_risk_band: str,
    task_id: str,
    has_image: bool = False,
) -> tuple[list, str]:
    """Apply failure-graph veto to risky specialist routes."""
    veto_thresholds = {"high": 0.3, "medium": 0.5, "low": 0.7, "": 0.5}
    if not (
        state.failure_graph
        and routing_decision
        and str(routing_decision[0]) != str(Role.FRONTDOOR)
        and routing_strategy not in ("mock", "forced")
    ):
        return routing_decision, routing_strategy

    # Modality fence (image direction): an image-carrying request routed to a
    # vision-only role has NO valid text fallback. Reverting it to the text
    # frontdoor makes the VL model answer BLIND (the same poisoning the vision
    # stage now fails visibly on) — worse than attempting the risky VL role.
    # Exempt it; if the VL backend is truly down the vision stage emits an
    # honest in-band vision_unavailable marker.
    if has_image and str(routing_decision[0]) in _vision_only_roles():
        log.debug(
            "Failure veto exempt: image request on vision-only role %s (no text fallback)",
            routing_decision[0],
            extra=task_extra(
                task_id=task_id,
                role=str(routing_decision[0]),
                stage="routing",
                strategy="vision_veto_exempt",
            ),
        )
        return routing_decision, routing_strategy

    try:
        risk = state.failure_graph.get_failure_risk(str(routing_decision[0]))
        veto_threshold = veto_thresholds.get(factual_risk_band, 0.5)
        if risk > veto_threshold:
            log.warning(
                "Failure veto: %s risk=%.2f > %.1f (factual_risk=%s), reverting to frontdoor",
                routing_decision[0],
                risk,
                veto_threshold,
                factual_risk_band or "none",
                extra=task_extra(
                    task_id=task_id,
                    role=str(routing_decision[0]),
                    stage="routing",
                    strategy="failure_vetoed",
                ),
            )
            return [str(Role.FRONTDOOR)], "failure_vetoed"
    except Exception as exc:
        log.debug("Failure risk veto check failed: %s", exc)
    return routing_decision, routing_strategy


# Ingest-triviality guard thresholds (chars; ~4 chars/token).
# Long-context payloads (big context) and genuinely hard prompts are never
# demoted — the guard only catches trivially-easy short prompts that the
# learned router (MemRL) leaks onto the 80B accuracy/long-context specialist.
_INGEST_GUARD_MAX_CONTEXT_CHARS = 2000   # above this we assume a real long-context task
_INGEST_GUARD_EASY_MAX_PROMPT_CHARS = 4000      # difficulty band == "easy"
_INGEST_GUARD_UNKNOWN_MAX_PROMPT_CHARS = 400    # difficulty signal off/unavailable: be strict


def apply_ingest_triviality_guard(
    request: ChatRequest,
    routing_decision: list,
    routing_strategy: str,
    difficulty_band: str,
    task_id: str,
) -> tuple[list, str]:
    """Demote trivially-easy short prompts off ``ingest_long_context``.

    ``ingest_long_context`` (Qwen3-Next-80B @ ~6.4 t/s) is a router-target
    accuracy/long-context specialist, but the learned MemRL router leaks some
    trivial short prompts (e.g. one-line arithmetic) onto it — paying a ~19×
    latency tax for work a cheap role answers identically. This guard redirects
    only *positively trivial* requests to ``worker_general``; it never touches
    long-context payloads or prompts the difficulty signal calls medium/hard,
    so short-but-hard reasoning (which ingest legitimately wins) is preserved.

    Opt-in: no-op unless the ``ingest_triviality_guard`` feature flag is on.
    Reuses the already-computed ``difficulty_band`` (no extra classifier call).
    """
    if not features().ingest_triviality_guard:
        return routing_decision, routing_strategy
    if not routing_decision or str(routing_decision[0]) != str(Role.INGEST_LONG_CONTEXT):
        return routing_decision, routing_strategy

    band = (difficulty_band or "").strip().lower()
    # Positive hard/medium evidence wins — never demote possibly-hard reasoning.
    if band in ("medium", "hard"):
        return routing_decision, routing_strategy

    context_chars = len(request.context or "")
    if context_chars > _INGEST_GUARD_MAX_CONTEXT_CHARS:
        return routing_decision, routing_strategy  # genuine long-context work

    prompt_chars = len(request.prompt or "")
    prompt_limit = (
        _INGEST_GUARD_EASY_MAX_PROMPT_CHARS
        if band == "easy"
        else _INGEST_GUARD_UNKNOWN_MAX_PROMPT_CHARS
    )
    if prompt_chars > prompt_limit:
        return routing_decision, routing_strategy

    target = str(Role.WORKER_GENERAL)
    log.info(
        "Ingest-triviality guard: ingest_long_context → %s "
        "(prompt_chars=%d, context_chars=%d, band=%s)",
        target,
        prompt_chars,
        context_chars,
        band or "none",
        extra=task_extra(
            task_id=task_id,
            role=target,
            stage="routing",
            strategy="ingest_triviality_guard",
        ),
    )
    return [target], f"{routing_strategy}:ingest_triviality_guard"


def assess_difficulty(prompt: str, role: str, task_id: str) -> tuple[float, str]:
    """Return difficulty score and band, falling back to no signal on failure."""
    try:
        from src.classifiers.difficulty_signal import assess_difficulty, get_mode as _ds_mode

        if _ds_mode() == "off":
            return 0.0, ""
        result = assess_difficulty(prompt, role=role)
        log.info(
            "Difficulty signal: score=%.3f band=%s",
            result.difficulty_score,
            result.difficulty_band,
            extra=task_extra(
                task_id=task_id,
                stage="routing",
                strategy="difficulty_signal",
            ),
        )
        return result.difficulty_score, result.difficulty_band
    except Exception as exc:
        log.debug("Difficulty signal scoring skipped: %s", exc)
        return 0.0, ""


def estimate_routing_cost(request: ChatRequest, state, routing_decision: list) -> float:
    """Estimate relative route cost for routing telemetry."""
    try:
        role_str = str(routing_decision[0]) if routing_decision else "frontdoor"
        if state.registry:
            role_cfg = state.registry.get_role(role_str)
            tier = getattr(role_cfg, "tier", "C") if role_cfg else "C"
        else:
            tier = "C"
        est_tokens = len(request.prompt) // 4 + len(request.context or "") // 4
        return _TIER_COST_WEIGHTS.get(tier, 1.0) * est_tokens / 1_000_000
    except Exception:
        return 0.0


def routing_meta(
    request: ChatRequest,
    state,
    routing_strategy: str,
    heuristic_priors: dict[str, float],
    factual_risk_score: float,
    factual_risk_band: str,
    factual_risk_mode: str,
    difficulty_score: float,
    difficulty_band: str,
    estimated_cost: float,
    assigned_role: str,
    xmas_meta: dict | None = None,
) -> dict:
    """Build progress-log routing metadata."""
    meta = {
        "decision_source": routing_strategy,
        "was_forced": bool(request.force_role),
        "heuristic_priors": {
            k: round(v, 4)
            for k, v in sorted(heuristic_priors.items(), key=lambda kv: -kv[1])[:4]
        },
        "factual_risk_score": round(factual_risk_score, 4),
        "factual_risk_band": factual_risk_band,
        "factual_risk_mode": factual_risk_mode,
        "difficulty_score": round(difficulty_score, 4),
        "difficulty_band": difficulty_band,
        "assigned_role": assigned_role,
        "estimated_cost": round(estimated_cost, 6),
    }
    if xmas_meta:
        meta["xmas"] = xmas_meta
    if state.llm_primitives and hasattr(state.llm_primitives, "get_stats"):
        try:
            stats = state.llm_primitives.get_stats()
            meta["active_requests"] = state.active_requests
            if "per_role" in stats:
                meta["queue_depth"] = {
                    role: rs.get("total_active", rs.get("round_robin_requests", 0))
                    for role, rs in stats["per_role"].items()
                    if isinstance(rs, dict)
                }
        except Exception:
            pass
    if state.registry and hasattr(state.registry, "roles"):
        try:
            meta["stack_state"] = {
                name: {
                    "model": str(getattr(cfg.model, "name", cfg.model))[:60],
                    "tier": cfg.tier,
                    "instances": getattr(cfg, "numa_instances", 1),
                }
                for name, cfg in state.registry.roles.items()
                if not name.startswith("draft_")
            }
        except Exception:
            pass
    if state.hybrid_router and hasattr(state.hybrid_router, "last_decision_meta"):
        try:
            router_meta = dict(state.hybrid_router.last_decision_meta or {})
            # Defensive remap: legacy episodic-memory seeds tagged actions as
            # `<role>:react` (a mode unified into REPL 2026-05-25). Normalize
            # here so the routing-decision event reflects current vocabulary
            # even if a stale seed slips through. The hybrid_router writes
            # last_decision_meta from ~7 inline sites; centralizing the
            # rewrite here is cheaper than patching each writer.
            ca = router_meta.get("chosen_action")
            if isinstance(ca, str) and ca.endswith(":react"):
                router_meta["chosen_action"] = ca[:-len(":react")] + ":repl"
            meta.update(router_meta)
        except Exception:
            pass
    # J10 (URE-1): persist uncertainty into the canonical progress event when
    # shadow logging is enabled. The sidecar remains for older ingest jobs, but
    # W7 requires the QScorer/replay path to see these fields directly.
    try:
        from src.features import features
        if features().ure_uncertainty_shadow_log:
            from src.uncertainty_shadow import (
                compute_routing_uncertainty,
                emit_uncertainty_shadow,
            )
            uncertainty = compute_routing_uncertainty(meta)
            meta["uncertainty_score"] = uncertainty["score"]
            meta["uncertainty_components"] = uncertainty["components"]
            meta["uncertainty_n_alternatives"] = uncertainty["n_alternatives"]
            emit_uncertainty_shadow(meta, request_id=getattr(request, "session_id", None))
    except Exception:
        pass

    # ── Vision capture ────────────────────────────────────────────────────
    # A request carrying image data must leave a DURABLE reference in the
    # progress JSONL. Until this, `image_path` was forwarded to the VL backend
    # but written to no progress event at all: every offline consumer (the
    # dashboard completed-tasks panel, replay, vl-suite triage) saw a vision
    # task as an ordinary text task with no way to tell which image produced
    # the answer. Verified against stored rows — `grep -c image_path` over
    # every logs/progress/*.jsonl returned 0, including for genuinely
    # successful `Vision multimodal` completions (e.g. chat-94e59d8f,
    # 2026-06-13, worker_vision).
    #
    # Written LAST so neither `meta.update(router_meta)` nor the uncertainty
    # block above can clobber or be perturbed by these keys.
    #
    # Only the PATH is stored — NEVER `image_base64`, whose payload is
    # megabytes per row and would balloon the progress log. A base64-only
    # request records the flag and the source so the panel can still label it.
    try:
        image_path = str(getattr(request, "image_path", "") or "").strip()
        if image_path:
            meta["image_path"] = image_path[:MAX_IMAGE_PATH_LEN]
            meta["has_image"] = True
            meta["image_source"] = "path"
        elif getattr(request, "image_base64", None):
            meta["has_image"] = True
            meta["image_source"] = "base64"
    except Exception:
        pass
    # Request-lifecycle join fields for the in-process live telemetry reducer.
    # These are observability-only and do not alter the routing decision.
    meta["request_id"] = getattr(request, "request_id", None)
    meta["batch_id"] = getattr(request, "batch_id", None)
    meta["workload_class"] = getattr(request, "workload_class", None)
    meta["request_priority"] = getattr(request, "request_priority", None)
    # RI-16: routing-stage latency snapshot (``routing_path`` + ``stage_ms``) for the
    # request being routed. Observability only; absent when no timing is active.
    stage_timing = routing_stage_timing.current()
    if stage_timing is not None:
        meta.update(stage_timing.telemetry())
    return meta


def log_routing_start(
    request: ChatRequest,
    state,
    task_id: str,
    task_ir: dict,
    routing_decision: list,
    routing_strategy: str,
    heuristic_priors: dict[str, float],
    factual_risk_score: float,
    factual_risk_band: str,
    factual_risk_mode: str,
    difficulty_score: float,
    difficulty_band: str,
    estimated_cost: float,
    assigned_role: str,
    xmas_meta: dict | None = None,
) -> None:
    """Log routing start to MemRL progress logger when available."""
    if not state.progress_logger:
        return
    state.progress_logger.log_task_started(
        task_id=task_id,
        task_ir=task_ir,
        routing_decision=routing_decision,
        routing_strategy=routing_strategy,
        routing_meta=routing_meta(
            request,
            state,
            routing_strategy,
            heuristic_priors,
            factual_risk_score,
            factual_risk_band,
            factual_risk_mode,
            difficulty_score,
            difficulty_band,
            estimated_cost,
            assigned_role,
            xmas_meta,
        ),
    )


def resolve_timeout(request: ChatRequest, routing_decision: list) -> int:
    """Compute role-specific timeout with request-level budget.

    Standard traffic clamps DOWN to the role SLA: a request-supplied
    ``timeout_s`` may only *shorten* the per-request budget below the role's
    interactive latency guarantee, never lengthen it. Interactive latency
    guarantees are therefore unchanged by this function.

    Exception — 2026-07-21 EV-11c incident. Eval-batch traffic
    (``workload_class == "eval_batch"``) that declares an explicit
    ``timeout_s`` may EXTEND its budget *beyond* the role SLA, because the
    request itself is declaring how long its work legitimately takes. The
    incident: 2,048-token MATH-tail generations served at 4-wide shared
    bandwidth need >60s, but the interactive worker SLA (60s) clamped every
    such call DOWN to 60s. Those doomed calls 504'd, tripped the production
    circuit breaker, and the breaker then served in-band ``[ERROR: ...]`` text
    as answers plus a silent role fallback. Honoring the eval batch's
    self-declared budget lets the long-but-legitimate call finish instead of
    being force-failed. Only self-declared ``eval_batch`` requests and
    ``task_root``-scoped agentic requests (INF-78 OAB-1, R3) can lengthen their
    budget; all other traffic keeps the exact DOWN-only ``min`` clamp.
    """
    role_str = str(routing_decision[0]) if routing_decision else str(Role.FRONTDOOR)
    timeout_s = role_timeout_for(role_str)
    if request.timeout_s is not None:
        if (
            str(getattr(request, "workload_class", "") or "") == "eval_batch"
            # INF-78 OAB-1 / R3: a task_root-scoped agentic call (the AutoKernel planner /
            # author, ~30-45 min for ~60 steps) self-declares its budget the same way.
            or getattr(request, "task_root", None)
        ):
            # Self-declared eval / scoped-agent budget: may EXTEND beyond the role SLA.
            timeout_s = max(1, int(request.timeout_s))
        else:
            # Interactive / all other traffic: DOWN-only clamp (unchanged).
            timeout_s = max(1, min(timeout_s, int(request.timeout_s)))
    return timeout_s


def extract_skill_ids(skill_context: str, state) -> list[str]:
    """Extract skill IDs from skill-augmented routing state."""
    if not (skill_context and hasattr(state, "hybrid_router") and state.hybrid_router):
        return []
    try:
        if hasattr(state.hybrid_router, "_last_skill_ids"):
            return list(state.hybrid_router._last_skill_ids)
    except Exception:
        pass
    return []


def derive_task_type_from_route(task_ir: dict, routing_decision: list) -> None:
    """Mutate task_ir task_type from routed role for tool-policy checks."""
    role_to_task_type = {
        "worker_math": "math",
        "coder_escalation": "coder",
        "coder_primary": "coder",
    }
    routed_role = str(routing_decision[0]) if routing_decision else ""
    derived_task_type = role_to_task_type.get(routed_role)
    if derived_task_type:
        task_ir["task_type"] = derived_task_type


def classify_trinity_role(request: ChatRequest, routing_decision: list, task_id: str) -> str:
    """Return shadow Trinity role classification."""
    try:
        from src.classifiers.role_classifier import classify_role as _classify_trinity_role

        result = _classify_trinity_role(
            request.prompt,
            routing_decision=routing_decision,
            force_role=request.force_role,
            thinking_budget=request.thinking_budget,
            context=request.context or "",
        )
        log.info(
            "Trinity role classified: role=%s reason=%s",
            result.role,
            result.reason,
            extra=task_extra(
                task_id=task_id,
                stage="routing",
                strategy="trinity_role_shadow",
            ),
        )
        return result.role
    except Exception as exc:
        log.debug("Trinity role classification skipped: %s", exc)
        return "worker"
