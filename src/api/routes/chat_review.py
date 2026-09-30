"""Architect review and quality gates for chat endpoints.

Extracted from chat.py during Phase 1 decomposition.
Contains: output quality detection, MemRL-conditional review gates,
architect verdict, fast revision, and plan review pipeline.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import Counter
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from src.config import get_config as _get_config
from src.constants import TASK_IR_OBJECTIVE_LEN
from src.roles import Role, chain_name_to_role
from src.roles import resolve_reviewer_role as _resolve_reviewer_role
from src.runtime import routing_stage_timing
from src.task_ir import canonicalize_task_ir
from src.prompt_builders import (
    build_review_verdict_prompt,
    build_revision_prompt,
)
from src.prompt_builders.review import REVIEW_VERDICT_QUESTION_CAP

log = logging.getLogger(__name__)


def _synthesized_plan_action(objective: str) -> str:
    """Describe the work to be reviewed, rather than echoing the user prompt.

    Chat requests normally arrive without an explicit multi-step plan. The
    review gate therefore creates a one-step plan for the architect. Passing
    the first 50 characters of the objective as that step's ``action`` made a
    request such as "You are an expert code verifier..." look like a plan that
    merely repeated the prompt. A ``drop`` verdict was then correct, but it
    rejected our synthetic plan rather than the user's task.
    """
    text = objective.lower()
    code_markers = ("code", "python", "function", "candidate solution", "assert ")
    if any(marker in text for marker in code_markers):
        return "Inspect the candidate code and verify expected behavior"
    return "Analyze the request and produce a directly supported answer"


if TYPE_CHECKING:
    from src.api.state import AppState
    from src.llm_primitives import LLMPrimitives
    from src.proactive_delegation.types import PlanReviewResult


def _detect_output_quality_issue(answer: str) -> str | None:
    """Detect quality issues in model output using text-based heuristics.

    Delegates to src.classifiers.quality_detector.detect_output_quality_issue.
    """
    from src.classifiers import detect_output_quality_issue

    return detect_output_quality_issue(answer)


def _should_review(state: "AppState", task_id: str, role: str, answer: str) -> bool:
    """MemRL-conditional: review only when confidence < threshold.

    Checks Q-values for the current role+task combination. If average
    Q-value is below 0.6, the role historically struggles with this
    task type and a brief architect review is warranted.

    Args:
        state: Application state with hybrid_router.
        task_id: Current task ID.
        role: The role that generated the answer.
        answer: The answer to potentially review.

    Returns:
        True if architect review should be triggered.
    """
    if not state.hybrid_router:
        return False
    if "architect" in str(role):
        return False  # Architects ARE the reviewer — don't self-review
    if len(answer) < 50:
        return False  # Trivial answers don't need review
    try:
        # Get Q-values for this role from MemRL
        retriever = state.hybrid_router.retriever
        task_ir = canonicalize_task_ir(
            {"task_type": "chat", "objective": answer[:TASK_IR_OBJECTIVE_LEN]}
        )
        results = retriever.retrieve_for_routing(task_ir)
        if not results:
            return False
        # Filter for current role
        role_results = [r for r in results if r.memory.action == str(role)]
        if not role_results:
            return False
        avg_q = sum(r.q_value for r in role_results) / len(role_results)
        return avg_q < _get_config().chat.review_low_q_threshold
    except Exception as exc:
        log.debug("Q-value review gate check failed: %s", exc)
        return False


# ── RI-18 C2: gate score accessor ─────────────────────────────────────────
#
# ``review_gate_score`` is ``_should_review`` with its inputs exposed: the same checks in
# the same order, the same retrieval, the same average. ``_should_review`` is left
# untouched on purpose (gitnexus rates it HIGH); the equivalence
# ``_should_review(...) == review_gate_score(...).triggered`` is pinned by
# ``tests/unit/test_review_gate_telemetry.py`` over a fixture grid. The review call
# sites use the accessor (one KNN, not two) so the RI-18 C1 telemetry can record
# ``avg_q``; the RI-18 driver uses it for the offline gate against a store snapshot.

GATE_SKIP_NO_ROUTER = "no_router"
GATE_SKIP_ARCHITECT = "architect_role"
GATE_SKIP_SHORT = "short"
GATE_SKIP_NO_RESULTS = "no_results"
GATE_SKIP_NO_ROLE_ROWS = "no_role_rows"
GATE_SKIP_ERROR = "error"
GATE_SCORED = "scored"
#: Answers shorter than this never reach the KNN (``_should_review``'s guard).
GATE_MIN_ANSWER_CHARS = 50


@dataclass(frozen=True)
class GateScore:
    """The review gate's inputs and decision for one answer.

    ``avg_q`` is set only when ``skip_reason == "scored"``: the mean Q of the answering
    role's rows among the retrieved memories. Every other reason is a structural
    no-review (or a caller-side precondition, ``caller:<reason>``).
    """

    avg_q: float | None
    n_results: int
    n_role_rows: int
    skip_reason: str
    threshold: float
    answer_chars: int
    gate_ms: float | None = None

    def fires_at(self, threshold: float) -> bool:
        """The gate's decision at ``threshold`` (``avg_q < threshold``)."""
        return self.avg_q is not None and self.avg_q < threshold

    @property
    def triggered(self) -> bool:
        """The production decision (the configured threshold)."""
        return self.fires_at(self.threshold)


def _review_threshold_or_nan() -> float:
    try:
        return float(_get_config().chat.review_low_q_threshold)
    except Exception:
        return math.nan


def review_gate_score(
    state: "AppState",
    role: str,
    answer: str,
    *,
    key_text: str | None = None,
) -> GateScore:
    """``_should_review``'s computation, returning its inputs as well as its decision.

    ``key_text`` replaces only the KNN key (default: the answer, as production does);
    the RI-18 driver uses it for the exploratory question-keyed variant. The short-answer
    guard always applies to ``answer``.
    """
    answer_chars = len(answer)

    def _skip(reason: str, n_results: int = 0, n_role_rows: int = 0) -> GateScore:
        return GateScore(
            avg_q=None,
            n_results=n_results,
            n_role_rows=n_role_rows,
            skip_reason=reason,
            threshold=_review_threshold_or_nan(),
            answer_chars=answer_chars,
        )

    if not state.hybrid_router:
        return _skip(GATE_SKIP_NO_ROUTER)
    if "architect" in str(role):
        return _skip(GATE_SKIP_ARCHITECT)
    if answer_chars < GATE_MIN_ANSWER_CHARS:
        return _skip(GATE_SKIP_SHORT)
    n_results = 0
    n_role_rows = 0
    try:
        retriever = state.hybrid_router.retriever
        key = answer if key_text is None else key_text
        task_ir = canonicalize_task_ir(
            {"task_type": "chat", "objective": key[:TASK_IR_OBJECTIVE_LEN]}
        )
        results = retriever.retrieve_for_routing(task_ir)
        if not results:
            return _skip(GATE_SKIP_NO_RESULTS)
        n_results = len(results)
        role_results = [r for r in results if r.memory.action == str(role)]
        n_role_rows = len(role_results)
        if not role_results:
            return _skip(GATE_SKIP_NO_ROLE_ROWS, n_results)
        avg_q = sum(r.q_value for r in role_results) / len(role_results)
        threshold = float(_get_config().chat.review_low_q_threshold)
        return GateScore(
            avg_q=avg_q,
            n_results=n_results,
            n_role_rows=n_role_rows,
            skip_reason=GATE_SCORED,
            threshold=threshold,
            answer_chars=answer_chars,
        )
    except Exception as exc:
        log.debug("Q-value review gate score failed: %s", exc)
        return _skip(GATE_SKIP_ERROR, n_results, n_role_rows)


def evaluate_review_gate(state: "AppState", role: str, answer: str) -> GateScore:
    """Score the gate at a review call site: timed into RI-16's ``review_gate`` stage.

    ``gate_ms`` carries the local wall time too, for paths without an RI-16 timer.
    """
    start = time.perf_counter()
    gate = routing_stage_timing.call_timed(
        "review_gate", review_gate_score, state, role, answer, accumulate=True
    )
    return replace(gate, gate_ms=(time.perf_counter() - start) * 1000.0)


def review_gate_skipped(reason: str, answer: str | None) -> GateScore:
    """A review site whose caller-side precondition failed before the gate ran."""
    return GateScore(
        avg_q=None,
        n_results=0,
        n_role_rows=0,
        skip_reason=f"caller:{reason}",
        threshold=_review_threshold_or_nan(),
        answer_chars=len(answer or ""),
    )


# RI-22: the verdict's three outcomes. ``unavailable`` = the call raised, came back empty,
# or produced text that starts with neither OK nor WRONG (e.g. a thinking block cut at the
# 80-token cap). Callers still keep the answer on ``unavailable`` (a review never blocks),
# but it is logged and counted as unavailable — never recorded as a silent OK.
VERDICT_OK = "ok"
VERDICT_WRONG = "wrong"
VERDICT_UNAVAILABLE = "unavailable"

_verdict_status_counts: Counter[str] = Counter()
_verdict_status_lock = threading.Lock()


def classify_verdict(text: str | None) -> str:
    """Classify raw verdict text as ``ok`` / ``wrong`` / ``unavailable`` (RI-22)."""
    stripped = (text or "").strip()
    if not stripped:
        return VERDICT_UNAVAILABLE
    upper = stripped.upper()
    if upper.startswith("OK"):
        return VERDICT_OK
    if upper.startswith("WRONG"):
        return VERDICT_WRONG
    return VERDICT_UNAVAILABLE


def verdict_status_counts() -> dict[str, int]:
    """Process-lifetime verdict outcome counts (``ok`` / ``wrong`` / ``unavailable``)."""
    with _verdict_status_lock:
        return dict(_verdict_status_counts)


def reset_verdict_status_counts() -> None:
    """Zero the verdict outcome counters (tests)."""
    with _verdict_status_lock:
        _verdict_status_counts.clear()


def _record_verdict_status(status: str, role: str, detail: str) -> None:
    with _verdict_status_lock:
        _verdict_status_counts[status] += 1
    if status == VERDICT_UNAVAILABLE:
        log.warning(
            "Review verdict UNAVAILABLE (role=%s): %s — answer kept unreviewed, "
            "counted as unavailable, not OK",
            role,
            detail,
        )


# ── RI-18 C1: persistent review-gate telemetry ────────────────────────────
#
# Before RI-18 a /chat review left no structured record: the verdict counter above is
# process memory, and the INFO log lines never reach ``orchestrator.log``. Every review
# call site now emits one ``review_gate`` event per gate evaluation — including a
# no-trigger evaluation and a caller-side skip — to the inference tap's structured
# events file. It is the write side of a belief-kernel source
# (``scripts/vidya/adapters/README.md`` in epyc-root). Emission is a no-op with the
# tap off and never raises.

REVIEW_GATE_TAP_EVENT = "review_gate"
REVIEW_GATE_SCHEMA = "review_gate/v1"

#: ``path`` values: where the review ran.
REVIEW_PATH_REPL = "repl"
REVIEW_PATH_DIRECT = "direct"
REVIEW_PATH_UNIFIED_STREAM = "unified_stream"
REVIEW_PATH_LEGACY_STREAM = "legacy_stream"
REVIEW_PATH_V1 = "v1"


def review_ms_since(start: float) -> float:
    """Milliseconds since ``start`` (a ``time.perf_counter()`` value)."""
    return (time.perf_counter() - start) * 1000.0


def _round_ms(value: float | None) -> float | None:
    return round(float(value), 3) if value is not None else None


def review_gate_event(
    task_id: str,
    role: str,
    *,
    gate: GateScore,
    path: str,
    verdict_status: str | None = None,
    reviewer_role: str | None = None,
    revision_applied: bool | None = None,
    verdict_ms: float | None = None,
    revision_ms: float | None = None,
) -> dict[str, Any]:
    """The ``review_gate`` event fields (pure; see :func:`record_review_gate`).

    ``reviewer_role`` defaults to the reviewer binding when a verdict ran (the /chat
    sites call the verdict with ``role=None``); /v1 passes the role it pinned.
    Timing reuses RI-16's request-scoped ``stage_ms`` (``review_gate``,
    ``review_verdict``) when this request is the timed one; otherwise the caller's local
    measurements. ``revision_ms`` is always local (RI-16 does not time the revision).
    """
    if reviewer_role is None and verdict_status is not None:
        try:
            reviewer_role = str(_resolve_reviewer_role())
        except Exception:
            reviewer_role = None
    stage = routing_stage_timing.telemetry_for(task_id)
    stage_ms = stage.get("stage_ms", {}) if stage else {}
    if stage_ms.get("review_gate") is not None:
        gate_ms = stage_ms.get("review_gate")
        timing_source = "stage_ms"
        if verdict_status is not None and stage_ms.get("review_verdict") is not None:
            verdict_ms = stage_ms.get("review_verdict")
    else:
        gate_ms = gate.gate_ms
        timing_source = "local"
    avg_q = gate.avg_q
    threshold = gate.threshold
    return {
        "schema": REVIEW_GATE_SCHEMA,
        "task_id": task_id,
        "path": path,
        "role": str(role),
        "triggered": gate.triggered,
        "avg_q": round(avg_q, 6) if avg_q is not None else None,
        "threshold": None if math.isnan(threshold) else threshold,
        "n_results": gate.n_results,
        "n_role_rows": gate.n_role_rows,
        "skip_reason": gate.skip_reason,
        "answer_chars": gate.answer_chars,
        "verdict_status": verdict_status,
        "reviewer_role": reviewer_role,
        "revision_applied": revision_applied,
        "gate_ms": _round_ms(gate_ms),
        "verdict_ms": _round_ms(verdict_ms),
        "revision_ms": _round_ms(revision_ms),
        "timing_source": timing_source,
    }


def record_review_gate(
    task_id: str,
    role: str,
    *,
    gate: GateScore,
    path: str,
    verdict_status: str | None = None,
    reviewer_role: str | None = None,
    revision_applied: bool | None = None,
    verdict_ms: float | None = None,
    revision_ms: float | None = None,
) -> bool:
    """Emit one ``review_gate`` tap event. Returns False (never raises) when not written.

    ``verdict_status`` is ``ok`` / ``wrong`` / ``unavailable`` (RI-22) when the gate
    triggered and the verdict ran, else None — an unavailable verdict is recorded as
    ``unavailable``, never as ``ok``. ``revision_applied`` is True only when the revision
    ran and returned text different from the original answer.
    """
    try:
        from src.runtime.inference_tap import emit_request_event

        fields = review_gate_event(
            task_id,
            role,
            gate=gate,
            path=path,
            verdict_status=verdict_status,
            reviewer_role=reviewer_role,
            revision_applied=revision_applied,
            verdict_ms=verdict_ms,
            revision_ms=revision_ms,
        )
        return emit_request_event(REVIEW_GATE_TAP_EVENT, **fields)
    except Exception as exc:  # telemetry must never affect the answer
        log.debug("review_gate tap event failed: %s", exc)
        return False


@routing_stage_timing.timed_stage("review_verdict", accumulate=True)
def _architect_verdict_with_status(
    question: str,
    answer: str,
    primitives: "LLMPrimitives",
    worker_digests: list[dict] | None = None,
    context_digest: str = "",
    role: str | None = None,
    *,
    question_cap: int = REVIEW_VERDICT_QUESTION_CAP,
) -> tuple[str | None, str]:
    """``_architect_verdict`` plus its RI-22 status (``ok`` / ``wrong`` / ``unavailable``).

    The returned verdict is exactly ``_architect_verdict``'s; the status is what telemetry
    records. RI-23: with ``thinking_roles_chat_lane`` on, the call runs thinking-OFF (the
    80-token cap is a verdict budget, not a reasoning budget) and without the registry
    ``system_prompt_suffix`` (``skip_suffix``), which reads as a second request after the
    verdict instruction. Flag off: the call is unchanged.

    ``question_cap`` is the verdict prompt's question truncation (RI-18 C3); the default
    is the production prompt.
    """
    from src.chat_completions_roles import thinking_off, thinking_roles_chat_lane_enabled

    prompt = build_review_verdict_prompt(
        question,
        answer,
        context_digest=context_digest,
        worker_digests=worker_digests,
        question_cap=question_cap,
    )
    verdict_role = role or str(_resolve_reviewer_role())
    call_kwargs: dict = {}
    if thinking_roles_chat_lane_enabled():
        call_kwargs["skip_suffix"] = True
    try:
        # ARCHSWAP-20260927: the verdict is REVIEW work, so it follows the reviewer
        # binding (default architect_critic, the MI210 27B) rather than the
        # consultant role architect_general (now CPU Flash-Next).
        with thinking_off():
            result = primitives.llm_call(
                prompt,
                role=verdict_role,
                n_tokens=80,  # Hard cap — verdict only
                **call_kwargs,
            )
        text = result.strip()
    except Exception as exc:
        log.debug("Architect verdict call failed: %s", exc)
        _record_verdict_status(VERDICT_UNAVAILABLE, verdict_role, f"call failed: {exc}")
        return None, VERDICT_UNAVAILABLE  # On error, don't block — return original answer
    status = classify_verdict(text)
    if status == VERDICT_UNAVAILABLE:
        _record_verdict_status(
            status,
            verdict_role,
            "empty verdict" if not text else f"unparseable verdict {text[:80]!r}",
        )
    else:
        _record_verdict_status(status, verdict_role, "")
    if text.upper().startswith("OK"):
        return None, status
    return text, status  # "WRONG: <corrections>" (or unparseable text: callers act on WRONG only)


def _architect_verdict(
    question: str,
    answer: str,
    primitives: "LLMPrimitives",
    worker_digests: list[dict] | None = None,
    context_digest: str = "",
    role: str | None = None,
) -> str | None:
    """Get architect's hyper-concise verdict on an answer.

    ``role`` defaults to the reviewer binding (``resolve_reviewer_role()``,
    ``architect_critic`` since ARCHSWAP-20260927: review/plan work stays on the 27B).
    An explicit ``role`` is for callers that PIN the consultant (the /v1 TE-1
    ``x_escalation=architect_general`` experiment arm).

    The architect emits ONLY a short verdict (~20-50 tokens at 6.75 t/s → ~6s).
    Returns None if OK, or "WRONG: <corrections>" if incorrect. A failed, empty or
    unparseable verdict also returns None / the raw text as before, but is logged and
    counted as ``unavailable`` (RI-22; see ``_architect_verdict_with_status``).

    Args:
        question: Original user question.
        answer: The answer to review.
        primitives: LLM primitives for inference.
        worker_digests: Optional TOON-encodable worker digests.
        context_digest: Optional compact context summary.

    Returns:
        None if answer is OK, or "WRONG: ..." string if corrections needed.
    """
    verdict, _status = _architect_verdict_with_status(
        question,
        answer,
        primitives,
        worker_digests=worker_digests,
        context_digest=context_digest,
        role=role,
    )
    return verdict


def _fast_revise(
    question: str,
    original_answer: str,
    corrections: str,
    primitives: "LLMPrimitives",
) -> str:
    """Fast worker expands architect's corrections into full answer.

    Uses worker_general (the live canonical worker role) — the fastest model
    in the stack after the worker-explore alias consolidation.
    7B is sufficient since the architect already specified exactly what to fix.

    Args:
        question: Original user question.
        original_answer: The answer to revise.
        corrections: Architect's correction notes.
        primitives: LLM primitives for inference.

    Returns:
        Revised answer, or original if revision fails.
    """
    prompt = build_revision_prompt(question, original_answer, corrections)
    try:
        result = primitives.llm_call(
            prompt,
            role=str(Role.WORKER_GENERAL),
            n_tokens=2000,
        )
        return result.strip() or original_answer
    except Exception as exc:
        log.debug("Fast revision failed: %s", exc)
        return original_answer  # Fallback to original on error


# ── Architect Plan Review Gate ─────────────────────────────────────────────


def _needs_plan_review(
    task_ir: dict,
    routing_decision: list,
    state: "AppState",
) -> bool:
    """Determine whether the plan needs architect review before execution.

    Bypass conditions (skip review when any is true):
    1. TaskComplexity is TRIVIAL or SIMPLE
    2. TaskComplexity is COMPLEX (architect already owns plan)
    3. Single-step plan (no multi-step coordination to review)
    4. Architect is already the actor (no self-review)
    5. Phase B: Q-value >= 0.6 for task class
    6. Phase C: 90% skip (stochastic)
    7. Feature flag disabled (checked by caller)

    Args:
        task_ir: TaskIR dict with objective, task_type.
        routing_decision: List of roles selected for routing.
        state: Application state.

    Returns:
        True if architect plan review should run.
    """
    import random
    from src.proactive_delegation import classify_task_complexity, TaskComplexity

    objective = task_ir.get("objective", "")
    complexity, _signals = classify_task_complexity(objective)

    # Bypass 1+2: Only review MODERATE complexity
    if complexity != TaskComplexity.MODERATE:
        return False

    # Bypass 3: Single-step plans don't need coordination review
    plan = task_ir.get("plan", {})
    steps = plan.get("steps", [])
    if len(steps) <= 1:
        # No explicit plan steps yet — check routing for multi-role indication
        # For chat requests, routing_decision is typically 1 role, but plan
        # review is still useful if complexity is MODERATE
        pass  # Allow review for MODERATE tasks even without explicit steps

    # Bypass 4: Don't self-review architect
    if routing_decision and "architect" in str(routing_decision[0]):
        return False

    # Phase-dependent gating
    phase = state.plan_review_phase
    _chat_cfg = _get_config().chat

    # Bypass 6: Phase C — stochastic skip (default 90% skip rate)
    if phase == "C":
        if random.random() < _chat_cfg.plan_review_phase_c_skip_rate:
            return False

    # Bypass 5: Phase B — Q-value gating
    if phase == "B" and state.hybrid_router:
        try:
            retriever = state.hybrid_router.retriever
            results = retriever.retrieve_for_routing(task_ir)
            if results:
                avg_q = sum(r.q_value for r in results) / len(results)
                if avg_q >= _chat_cfg.review_skip_q_threshold:
                    return False
        except Exception as exc:
            log.debug("Plan review Q-value gating failed: %s", exc)

    return True


def _architect_plan_review(
    task_ir: dict,
    routing_decision: list,
    primitives: "LLMPrimitives",
    state: "AppState",
    task_id: str,
) -> "PlanReviewResult | None":
    """Execute architect plan review and return result.

    Non-blocking: returns None on timeout or error.

    Args:
        task_ir: TaskIR dict.
        routing_decision: Current routing decision.
        primitives: LLM primitives for calling architect.
        state: Application state.
        task_id: Current task ID.

    Returns:
        PlanReviewResult or None on failure.
    """
    from src.proactive_delegation import ArchitectReviewService

    objective = task_ir.get("objective", "")
    task_type = task_ir.get("task_type", "chat")

    # Construct plan steps from routing decision (minimal for chat requests)
    plan = task_ir.get("plan", {})
    plan_steps = plan.get("steps", [])

    # If no explicit plan, synthesize from routing_decision
    if not plan_steps and routing_decision:
        plan_steps = [
            {
                "id": f"S{i + 1}",
                "actor": str(role),
                "action": _synthesized_plan_action(objective),
                "outputs": [],
            }
            for i, role in enumerate(routing_decision)
        ]

    if not plan_steps:
        return None

    # RD-1/RD-5/TM-3: the reviewer role is resolved via the config-level binding
    # inside the service (default → architect_critic since ARCHSWAP-20260927, i.e. the
    # same MI210 27B that reviewed before the swap).
    # review_plan() emits an always-on shadow trace event regardless of whether the
    # plan_review feature acts — this is the DECOUPLED shadow-emission path: the
    # trace flows even though the plan_review flag itself requires memrl (features.py
    # validate()), because emission lives on the review_service seam, not the flag.
    # We anchor emitted rows to this task via session_id so decision_chain replay
    # (src/trace/query.py) can reconstruct task → plan → review → gate → outcome.
    review_service = ArchitectReviewService(primitives)
    result = review_service.review_plan(
        objective=objective,
        task_type=task_type,
        plan_steps=plan_steps,
        session_id=task_id,
    )

    if result:
        log.info(
            "Plan review: decision=%s score=%.2f feedback=%s",
            result.decision,
            result.score,
            result.feedback[:60],
        )

    return result


def _apply_plan_review(
    routing_decision: list,
    review: "PlanReviewResult",
) -> list:
    """Apply architect's plan review corrections to routing decision.

    Handles 'reroute' patches that change which specialist handles a step.

    Args:
        routing_decision: Current routing decision list.
        review: Architect's plan review result.

    Returns:
        Updated routing decision (may be unchanged if no reroute patches).
    """
    if not review.patches:
        return routing_decision

    updated = list(routing_decision)

    for patch in review.patches:
        op = patch.get("op", "")
        if op == "reroute":
            new_role = _normalize_plan_review_reroute_role(patch.get("v", ""))
            step_id = patch.get("step", "")
            if new_role:
                # Map step index to routing decision index
                # S1 → index 0, S2 → index 1, etc.
                try:
                    idx = int(step_id.replace("S", "")) - 1
                    if 0 <= idx < len(updated):
                        updated[idx] = new_role
                    elif idx == 0 and len(updated) >= 1:
                        updated[0] = new_role
                except (ValueError, IndexError):
                    # If step_id isn't parseable, reroute the first step
                    if updated:
                        updated[0] = new_role

    return updated


def _normalize_plan_review_reroute_role(value: object) -> str:
    """Return a canonical role for a plan-review reroute target, or "".

    Architect patches are model output, so the ``v`` field can contain free-form
    task guidance instead of a backend role. Letting that text overwrite
    ``routing_decision`` later turns it into ``force_role`` and fails at
    inference time with "No backend configured for role ...".
    """
    raw = str(value or "").strip()
    if not raw:
        return ""
    role = Role.from_string(raw) or chain_name_to_role(raw)
    if role is None:
        log.warning("Ignoring invalid plan-review reroute target: %r", raw[:120])
        return ""
    return str(role)


def _plan_review_should_abort(review: "PlanReviewResult | None") -> bool:
    """Return True only for an explicit execution-abort verdict.

    ``drop`` is a plan operation: discard the reviewed plan and proceed through
    the normal no-plan/default route. Treating it as a task rejection violated
    ``ArchitectReviewService.review_plan``'s non-blocking contract and turned a
    reviewer correctly spotting a bad synthesized plan into an HTTP 422 task
    failure. ``abort`` is reserved for a future separately-specified safety
    verdict; the current reviewer schema does not emit it.
    """
    return bool(review and str(review.decision).strip().lower() == "abort")


def _plan_review_abort_message(review: "PlanReviewResult") -> str:
    """Human-readable message for a terminal plan-review rejection."""
    feedback = str(review.feedback or "").strip()
    if feedback:
        return f"Plan rejected by architect review: {feedback}"
    return "Plan rejected by architect review."


def _store_plan_review_episode(
    state: "AppState",
    task_id: str,
    task_ir: dict,
    review: "PlanReviewResult",
) -> None:
    """Store plan review result as MemRL episode and progress log entry.

    Architect corrections become high-quality training signals:
    - review.score mapped to reward: score * 2 - 1 (0-1 → -1..+1)
    - Action: plan:{role1},{role2} (the routing decision)

    Args:
        state: Application state.
        task_id: Current task ID.
        task_ir: TaskIR dict.
        review: Architect's plan review result.
    """
    # Log to progress JSONL
    if state.progress_logger:
        from orchestration.repl_memory.progress_logger import ProgressEntry, EventType

        # RI-6: Structured review context replaces truncated feedback proxy.
        # Provides task_type, risk_band, key_claims for richer MemRL signal.
        _review_context = {
            "decision": review.decision,
            "score": review.score,
            "feedback": review.feedback[:200],
            "patches": review.patches[:5],
            "task_type": task_ir.get("task_type", "chat"),
            "risk_band": task_ir.get("risk_band", ""),
            "verification_focus": review.decision if review.decision != "ok" else "",
        }
        state.progress_logger.log(
            ProgressEntry(
                event_type=EventType.PLAN_REVIEWED,
                task_id=task_id,
                agent_role=str(_resolve_reviewer_role()),
                data=_review_context,
                outcome="success" if review.is_ok else "corrected",
            )
        )

    # Update plan review stats (thread-safe)
    stats = state.update_plan_review_stats(approved=review.is_ok)

    # Recompute phase
    state.plan_review_phase = _compute_plan_review_phase(stats)

    # Store as MemRL episode for Q-learning (expert demonstration)
    if state.q_scorer and state.hybrid_router:
        try:
            reward = review.score * 2 - 1  # Map 0-1 to -1..+1
            state.q_scorer.score_external_result(
                task_description=task_ir.get("objective", "")[:TASK_IR_OBJECTIVE_LEN],
                action=f"plan_review:{review.decision}",
                reward=reward,
                action_type="plan_review",
                context={
                    "task_type": task_ir.get("task_type", "chat"),
                    "review_decision": review.decision,
                    "review_feedback": review.feedback[:200],
                    "risk_band": task_ir.get("risk_band", ""),
                    "verification_focus": review.decision if review.decision != "ok" else "",
                    "source": "plan_review",
                },
            )
        except Exception as exc:
            log.debug("MemRL plan review score storage failed: %s", exc)


def _compute_plan_review_phase(stats: dict) -> str:
    """Compute current plan review phase from statistics.

    Phase A (bootstrap): < 50 reviews or low Q-values
    Phase B (supervised fade): mean Q >= 0.7, min Q >= 0.5
    Phase C (spot-check): min Q >= 0.7 and >= 100 reviews

    Args:
        stats: Plan review statistics dict.

    Returns:
        Phase string: "A", "B", or "C".
    """
    _pr = _get_config().chat

    total = stats.get("total_reviews", 0)
    if total < _pr.plan_review_phase_a_min:
        return "A"

    q_vals = stats.get("task_class_q_values", {})
    if not q_vals:
        return "A"

    values = list(q_vals.values())
    mean_q = sum(values) / len(values)
    min_q = min(values)

    if min_q >= _pr.plan_review_phase_c_min_q and total >= _pr.plan_review_phase_c_min_total:
        return "C"
    if mean_q >= _pr.plan_review_phase_b_mean_q and min_q >= _pr.plan_review_phase_b_min_q:
        return "B"
    return "A"
