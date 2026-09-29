"""TE-1 (UFH-13) / HS-4 P4 subset — /v1 escalation parity with /chat.

Flag ``v1_escalation`` (default OFF) plus the per-request key ``x_escalation``
(``auto`` | ``off`` | ``architect_general``). Escalation is OPT-IN per request:
with the key absent nothing here runs and the route is byte-identical whether
the flag is on or off (golden-pinned both ways), so turning the flag on for an
experiment never changes other /v1 traffic. Only an explicit ``auto`` or
``architect_general`` escalates, and only with the flag on.

``auto`` keeps /chat's targets verbatim (quality escalation -> coder_escalation,
review verdict -> the reviewer binding, architect_critic since ARCHSWAP-20260927). ``architect_general`` keeps /chat's
TRIGGERS but pins every consultant call to that role: after the pending role
swap coder_escalation stays on the 27B while architect_general becomes
Flash-Next, so an arm that must measure Flash-Next as the consultant (UFH-13
A2) names it instead of inheriting frontdoor's default chain.

WHICH ESCALATION, AND WHY THESE TRIGGERS
========================================
This module adds no policy. It applies /chat's OWN post-answer escalation hooks
to a /v1 frontdoor answer, by calling the very functions /chat calls, in the
order /chat calls them:

* ``direct`` stage — client tool mode (the mode OpenCode uses) and
  ``x_disable_repl``. A client-mode backend call is one direct completion, so
  it gets /chat's direct-stage chain (``chat_pipeline/direct_stage.py``):

  1. ``quality_escalation`` — ``chat_pipeline.stages._quality_escalate``: when
     the ``generation_monitor`` flag is on and the quality detector flags the
     answer, the SAME prompt is re-answered by ``coder_escalation`` (``auto``;
     today an alias on architect_general's :8083 process, and it stays on the 27B
     after the role swap) or by the pinned consultant (``architect_general``),
     and that answer replaces it.
  2. ``review_gate`` — ``chat_review._should_review`` (MemRL: the answering
     role's mean Q-value for this answer is below
     ``chat.review_low_q_threshold``; never for an architect role or an answer
     under 50 chars) -> ``chat_review._architect_verdict`` (80-token verdict; the
     reviewer binding, ``architect_critic`` since ARCHSWAP-20260927, under ``auto``;
     the pinned consultant under ``x_escalation=architect_general``) -> on ``WRONG``, ``chat_review._fast_revise``
     (``worker_general`` rewrites the answer with the corrections; recorded as
     ``review_gate_revision``, which is NOT a consultant call).

* ``repl`` stage — the default REPL bridge. /chat's REPL stage runs only the
  review gate after the graph (``chat_pipeline/repl_executor.py``), so that is
  what applies here, to a FINAL answer only (/chat reviews a graph success).

NOT reproduced, and why (so the semantics stay identical rather than invented):

* failure-driven graph escalation (``FrontdoorNode`` -> ``CoderEscalationNode``
  -> ``ArchitectNode``: EARLY_ABORT, retries exhausted, promoted nudges) needs
  the graph's failure accounting over REPL execution errors. The /v1 REPL
  bridge is not the graph (it never feeds an error back), and in client tool
  mode tools run in the CLIENT, so the orchestrator never observes a tool
  failure. Porting /v1 onto the graph would change the no-escalation arm too.
* model-requested ``escalate()``: a REPL tool, absent in client mode, and
  ignored by ``FrontdoorNode`` in /chat's graph path (only ``CoderNode`` reads
  ``_escalation_requested``).
* ``chat_delegation`` is architect -> specialist delegation, not escalation.

/chat skips every hook when ``force_role`` is set; the /v1 analogue is a role
override (``x_force_role`` / ``x_force_model`` / ``x_orchestrator_role``), so an
eval pin (UFH-13 arm A0) never escalates.

TELEMETRY
=========
Every escalation call is recorded as a step with its trigger, from/to role, the
server URL the role resolves to, and llama-server's own timings for that call
(``prompt_ms`` + ``gen_ms``, the deltas of ``LLMPrimitives``' per-request
accumulators, which sum ``timings.prompt_ms`` / ``timings.predicted_ms``). A
step whose URL is ``architect_general``'s server is a consultant step, and
``consultant_device_seconds`` sums exactly those. While a step's call runs, the
primitives' trace keys carry ``escalation_trigger`` / ``escalation_from_role``
/ ``escalation_to_role``, so the call's own tap ``timings`` event is tagged
too. The per-request receipt goes to the tap as a ``v1_escalation`` event and
to ``x_orchestrator_metadata.escalation`` (with ``x_show_routing``).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from src.roles import Role

log = logging.getLogger(__name__)

CONSULTANT_ROLE = str(Role.ARCHITECT_GENERAL)
QUALITY_ESCALATION_ROLE = str(Role.CODER_ESCALATION)
REVISION_ROLE = str(Role.WORKER_GENERAL)

TRIGGER_QUALITY = "quality_escalation"
TRIGGER_REVIEW = "review_gate"
TRIGGER_REVISION = "review_gate_revision"

STAGE_DIRECT = "direct"
STAGE_REPL = "repl"

TAP_EVENT = "v1_escalation"

# x_escalation values that NAME the consultant (pin every escalation call to it).
PINNED_TARGETS = frozenset({CONSULTANT_ROLE})


def _role_name(role: Any) -> str:
    return role.value if isinstance(role, Role) else str(role)


def _number(owner: Any, name: str) -> float:
    value = getattr(owner, name, 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _counters(primitives: Any) -> dict[str, float]:
    return {
        "calls": _number(primitives, "total_calls"),
        "prompt_ms": _number(primitives, "total_prompt_eval_ms"),
        "gen_ms": _number(primitives, "total_generation_ms"),
        "tokens": _number(primitives, "total_tokens_generated"),
        "prompt_tokens": _number(primitives, "total_prompt_tokens_reported"),
    }


def _server_url(primitives: Any, role: str) -> str | None:
    urls = getattr(primitives, "server_urls", None)
    if not isinstance(urls, dict):
        return None
    url = urls.get(role)
    return url if isinstance(url, str) and url else None


def _server_ports(url: str | None) -> list[int]:
    if not url:
        return []
    from src.llm_primitives.backend import _url_str_ports

    return _url_str_ports(url)


def _model_id(primitives: Any, role: str) -> str | None:
    """The registry's model name for ``role`` (best effort; None when unknown)."""
    registry = getattr(primitives, "registry", None)
    getter = getattr(registry, "get_role", None)
    if not callable(getter):
        return None
    try:
        name = getattr(getattr(getter(role), "model", None), "name", None)
    except Exception:
        return None
    return name if isinstance(name, str) and name else None


@dataclass
class V1EscalationPlan:
    """What one /v1 request may do, decided once before any model call."""

    requested: str | None
    enabled: bool
    disabled_reason: str | None
    from_role: str
    # The consultant role the caller pinned (x_escalation=<role>), or None for
    # /chat's own targets (auto).
    target_role: str | None = None
    base_trace_keys: dict[str, Any] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)
    final_answer_role: str = ""
    consultant_url: str | None = None
    error: str | None = None

    @property
    def fired(self) -> bool:
        return any(step["calls"] > 0 for step in self.steps)

    @property
    def consultant_role(self) -> str:
        return self.target_role or CONSULTANT_ROLE

    @property
    def quality_escalation_role(self) -> str:
        return self.target_role or QUALITY_ESCALATION_ROLE

    @property
    def consultant_device_seconds(self) -> float:
        return round(sum(step["device_seconds"] for step in self.steps if step["consultant"]), 6)

    def receipt(self) -> dict[str, Any]:
        """The escalation receipt (metadata block and tap event body)."""
        return {
            "requested": self.requested,
            "enabled": self.enabled,
            "disabled_reason": self.disabled_reason,
            "fired": self.fired,
            "target": "pinned" if self.target_role else "chat_default",
            "target_role": self.target_role,
            "from_role": self.from_role,
            "final_answer_role": self.final_answer_role or self.from_role,
            "consultant_role": self.consultant_role,
            "consultant_url": self.consultant_url,
            "consultant_ports": _server_ports(self.consultant_url),
            "consultant_device_seconds": self.consultant_device_seconds,
            "steps": [dict(step) for step in self.steps],
            "error": self.error,
        }

    def usage_delta(self) -> tuple[int, int]:
        """(prompt_tokens, completion_tokens) the escalation calls added."""
        return (
            sum(step["prompt_tokens"] for step in self.steps),
            sum(step["tokens"] for step in self.steps),
        )


def plan_v1_escalation(
    *,
    flag_on: bool,
    requested: str | None,
    role: Any,
    role_override: bool,
    image_input: bool,
) -> V1EscalationPlan | None:
    """Decide eligibility. ``None`` = no key sent: touch nothing, flag on or off.

    Opt-in: only an explicit ``auto`` / ``architect_general`` can escalate, and
    only with the flag on. An explicit ``off`` gets a disabled receipt (the
    experiment's no-escalation arm proves it was off). Flag off + key sent is
    recorded as disabled (``flag_off``) so a run that believes it is escalating
    can see that it is not.
    """
    if requested is None:
        return None
    from_role = _role_name(role)
    reason: str | None = None
    if not flag_on:
        reason = "flag_off"
    elif requested == "off":
        reason = "x_escalation_off"
    elif role_override:
        reason = "role_override"
    elif from_role != str(Role.FRONTDOOR):
        reason = "not_frontdoor"
    elif image_input:
        reason = "image_input"
    return V1EscalationPlan(
        requested=requested,
        enabled=reason is None,
        disabled_reason=reason,
        from_role=from_role,
        target_role=requested if requested in PINNED_TARGETS else None,
        final_answer_role=from_role,
    )


@contextmanager
def _tagged_trace(
    plan: V1EscalationPlan, primitives: Any, trigger: str, from_role: str, to_role: str
) -> Iterator[None]:
    """Tag the escalation call's tap section; restore the request's own keys after."""
    setter = getattr(primitives, "set_request_trace_keys", None)
    if not callable(setter):
        yield
        return
    setter(
        {
            **plan.base_trace_keys,
            "escalation_trigger": trigger,
            "escalation_from_role": from_role,
            "escalation_to_role": to_role,
        }
    )
    try:
        yield
    finally:
        setter(dict(plan.base_trace_keys))


def _record_step(
    plan: V1EscalationPlan,
    primitives: Any,
    *,
    trigger: str,
    from_role: str,
    to_role: str,
    before: dict[str, float],
    outcome: str,
) -> dict[str, Any] | None:
    after = _counters(primitives)
    calls = int(after["calls"] - before["calls"])
    if calls <= 0:
        return None  # the hook decided not to call anything: no escalation
    prompt_ms = max(0.0, after["prompt_ms"] - before["prompt_ms"])
    gen_ms = max(0.0, after["gen_ms"] - before["gen_ms"])
    server_url = _server_url(primitives, to_role)
    last_meta = {}
    getter = getattr(primitives, "get_last_inference_meta", None)
    if callable(getter):
        meta = getter()
        last_meta = meta if isinstance(meta, dict) else {}
    step = {
        "trigger": trigger,
        "from_role": from_role,
        "to_role": to_role,
        "server_url": server_url,
        "ports": _server_ports(server_url),
        "model_id": _model_id(primitives, to_role),
        "consultant": bool(server_url) and server_url == plan.consultant_url,
        "calls": calls,
        "prompt_ms": round(prompt_ms, 3),
        "gen_ms": round(gen_ms, 3),
        "device_seconds": round((prompt_ms + gen_ms) / 1000.0, 6),
        "tokens": int(max(0.0, after["tokens"] - before["tokens"])),
        "prompt_tokens": int(max(0.0, after["prompt_tokens"] - before["prompt_tokens"])),
        "completion_reason": str(last_meta.get("completion_reason") or "") or None,
        "outcome": outcome,
    }
    plan.steps.append(step)
    return step


def escalate_answer(
    plan: V1EscalationPlan | None,
    *,
    stage: str,
    answer: str,
    question: str,
    direct_prompt: str,
    primitives: Any,
    state: Any,
    task_id: str,
) -> str:
    """Run /chat's post-answer escalation hooks on a /v1 answer; return the answer.

    ``stage`` is ``direct`` (client tool mode, ``x_disable_repl``) or ``repl``
    (a FINAL answer of the REPL bridge). A no-op unless ``plan.enabled``.
    """
    if plan is None or not plan.enabled or not answer:
        return answer
    try:
        return _escalate_answer(
            plan,
            stage=stage,
            answer=answer,
            question=question,
            direct_prompt=direct_prompt,
            primitives=primitives,
            state=state,
            task_id=task_id,
        )
    except Exception as exc:
        # /chat's hooks never block an answer (each helper already swallows its
        # own backend failure); anything else is recorded in the receipt, not hidden.
        log.warning(
            "v1 escalation failed (%s: %s); serving the unescalated answer", type(exc).__name__, exc
        )
        plan.error = f"{type(exc).__name__}: {exc}"
        plan.final_answer_role = plan.from_role
        return answer


def _escalate_answer(
    plan: V1EscalationPlan,
    *,
    stage: str,
    answer: str,
    question: str,
    direct_prompt: str,
    primitives: Any,
    state: Any,
    task_id: str,
) -> str:
    from src.api.routes import chat_review
    from src.api.routes.chat_pipeline import stages as chat_stages
    from src.api.services.memrl import ensure_memrl_initialized

    plan.consultant_url = _server_url(primitives, plan.consultant_role)
    role = plan.final_answer_role or plan.from_role

    if stage == STAGE_DIRECT:
        quality_role = plan.quality_escalation_role
        before = _counters(primitives)
        with _tagged_trace(plan, primitives, TRIGGER_QUALITY, role, quality_role):
            new_answer, new_role = chat_stages._quality_escalate(
                answer,
                direct_prompt,
                primitives,
                role,
                allow_escalation=True,
                escalation_role=Role(quality_role),
            )
        adopted = _role_name(new_role) != role
        _record_step(
            plan,
            primitives,
            trigger=TRIGGER_QUALITY,
            from_role=role,
            to_role=quality_role,
            before=before,
            outcome="adopted" if adopted else "not_adopted",
        )
        if adopted:
            answer, role = new_answer, _role_name(new_role)

    # The review gate reads MemRL Q-values; /chat initialises MemRL on its
    # routing path, /v1 does it here (idempotent; a no-op with memrl off).
    ensure_memrl_initialized(state)
    if (
        answer
        and not answer.startswith("[ERROR")
        and chat_review._should_review(state, task_id, role, answer)
    ):
        # ARCHSWAP-20260927: /chat's verdict goes to the REVIEWER binding
        # (resolve_reviewer_role(), default architect_critic = the 27B), so ``auto``
        # keeps that target verbatim. A pinned consultant (``x_escalation=
        # architect_general``, UFH-13 A2) pins this call too, as it pins every
        # consultant call, and the step records the role actually asked.
        from src.roles import resolve_reviewer_role

        verdict_role = plan.target_role or str(resolve_reviewer_role())
        before = _counters(primitives)
        with _tagged_trace(plan, primitives, TRIGGER_REVIEW, role, verdict_role):
            verdict, verdict_status = chat_review._architect_verdict_with_status(
                question=question,
                answer=answer,
                primitives=primitives,
                role=verdict_role,
            )
        wrong = bool(verdict) and verdict.upper().startswith("WRONG")
        review_step = _record_step(
            plan,
            primitives,
            trigger=TRIGGER_REVIEW,
            from_role=role,
            to_role=verdict_role,
            before=before,
            outcome="wrong" if wrong else "ok_or_unavailable",
        )
        if review_step is not None:
            # RI-22: split ok_or_unavailable. `outcome` keeps its established values
            # (the UFH-13 thesis report buckets on them); this names which one it was.
            review_step["verdict_status"] = verdict_status
        if wrong:
            corrections = verdict.split(":", 1)[1].strip() if ":" in verdict else verdict
            before = _counters(primitives)
            with _tagged_trace(plan, primitives, TRIGGER_REVISION, role, REVISION_ROLE):
                revised = chat_review._fast_revise(
                    question=question,
                    original_answer=answer,
                    corrections=corrections,
                    primitives=primitives,
                )
            changed = revised != answer
            _record_step(
                plan,
                primitives,
                trigger=TRIGGER_REVISION,
                from_role=role,
                to_role=REVISION_ROLE,
                before=before,
                outcome="revised" if changed else "kept_original",
            )
            if changed:
                answer, role = revised, REVISION_ROLE

    plan.final_answer_role = role
    return answer


def record_escalation(
    plan: V1EscalationPlan | None,
    *,
    chat_id: str,
    request_keys: dict[str, Any],
    primitives: Any,
) -> dict[str, Any] | None:
    """Emit the receipt to the tap; return it for the response metadata."""
    if plan is None:
        return None
    if plan.consultant_url is None and primitives is not None:
        plan.consultant_url = _server_url(primitives, plan.consultant_role)
    receipt = plan.receipt()
    if primitives is not None:
        # Whole-request llama-server time (frontdoor + every escalation call),
        # for the experiment's "frontdoor device-seconds" beside the consultant's.
        receipt["request_device_seconds"] = round(
            (
                _number(primitives, "total_prompt_eval_ms")
                + _number(primitives, "total_generation_ms")
            )
            / 1000.0,
            6,
        )
    try:
        from src.runtime.inference_tap import emit_request_event

        emit_request_event(TAP_EVENT, chat_id=chat_id, request_keys=dict(request_keys), **receipt)
    except Exception as exc:  # the tap must never affect the response
        log.debug("v1_escalation tap event failed: %s", exc)
    return receipt
