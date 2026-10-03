"""Draft, critique, fallback, and reconciliation for AutoPilot planning."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import peaf

from controller_io import (
    _append_planner_archive,
    _loads_json_payload,
    _open_planner_tap,
    action_repair_completer,
    extract_action_with_repair,
    extract_rationale_with_repair,
    validate_single_variable,
)
from planner_providers import (
    PlannerProvider,
    PlannerProviderResult,
    get_planner_provider,
)
from src.autopilot_core.action_identity import action_signature
from src.structured_output.repair import CompleteFn, parse_with_repair

log = logging.getLogger("autopilot")

KNOWN_ACTION_TYPES = {
    "seed_batch",
    "numeric_trial",
    "prompt_mutation",
    "gepa_optimize",
    "code_mutation",
    "structural_experiment",
    "consult_gate_probe",
    "structural_prune",
    "slot_compact",
    "train_routing_models",
    "distill_skillbank",
    "reset_memories",
    "deep_eval",
    "rollback",
    "distill_knowledge",
}

LOW_RISK_ACTIONS = {"seed_batch", "deep_eval", "distill_knowledge"}
MEDIUM_RISK_ACTIONS = {
    "numeric_trial",
    "consult_gate_probe",
    "slot_compact",
    "train_routing_models",
    "distill_skillbank",
}
HIGH_RISK_ACTIONS = KNOWN_ACTION_TYPES - LOW_RISK_ACTIONS - MEDIUM_RISK_ACTIONS
SAFE_FALLBACK_NUMERIC_SURFACE = "memrl_retrieval"
SAFE_FALLBACK_NUMERIC_PARAMS = {"memrl_retrieval.semantic_k": 15}
DETERMINISTIC_GUARD_PREFIX = "planner deterministic guard:"

# Actions safe to DISPATCH WITHOUT a critic verdict (degraded/uncritiqued mode).
# Intentionally EMPTY: when the binding critic is unavailable we PAUSE by default
# rather than run anything unreviewed. NOTE this is deliberately stricter than
# LOW_RISK_ACTIONS — seed_batch is NOT here, because low-risk seed looping was the
# exact failure the critic exists to catch (critic_reject_loop @708). Extend ONLY
# with genuinely observational, non-mutating, non-looping action types.
OBSERVATIONAL_ACTIONS: frozenset[str] = frozenset()
_CRITIQUE_CONTEXT_HEADINGS = (
    "### Evidence Power and Sequential Candidate Status",
    "### System Health",
    "### Action Availability",
    "### Blacklisted Configurations",
    "### Feature Flags",
    "### Last Non-Executing Action",
)
_CRITIQUE_SECTION_CHAR_LIMIT = 1600

# Claude planner drafts run a ~80KB prompt WITH tool access (Read/Grep/Glob) and
# take 60-228s in practice; the old hard 300s default intermittently KILLED a
# slow-but-valid draft → "empty response" → degraded (and, since 2026-06-07, a
# critic_unavailable pause that stalled the run, e.g. @711). Give generous
# headroom; operator-tunable via AUTOPILOT_PLANNER_TIMEOUT. (2026-06-09)
DEFAULT_PLANNER_TIMEOUT = int(os.environ.get("AUTOPILOT_PLANNER_TIMEOUT", "600"))
DEFAULT_SPEND_BREAKER_LOCAL_PRIMARY = "local_ingest"
DEFAULT_SPEND_BREAKER_LOCAL_CRITIC = "local_frontdoor"
DEFAULT_CODEX_CRITIC_FALLBACK = "claude"


@dataclass
class PlannerSettings:
    primary: str = "claude"
    critic: str = "codex"
    # 2026-06-04: default flipped shadow_critique -> draft_critique. The critic is
    # now BINDING: a valid revised_action is dispatched, and a reject routes to a
    # metric-bearing fallback (see _reconcile). Shadow mode logged 320 reject/revise
    # verdicts that were all ignored (264 dispatched anyway) while the run dead-locked.
    # Gated on the expanded _reconcile test matrix + the rejected-draft feedback
    # path in autopilot.py (a reject can no longer bypass blacklist/quota/skip).
    mode: str = "draft_critique"
    critique_policy: str = "medium_plus"
    circuit_failures: int = 2
    circuit_cooldown_s: float = 900.0


@dataclass
class PlannerCritique:
    decision: str = "approve"
    confidence: float = 0.0
    issues: list[str] = field(default_factory=list)
    revised_action: dict[str, Any] | None = None
    revised_rationale: dict[str, Any] | None = None
    raw_text: str = ""
    provider: str = ""
    parse_error: str = ""
    # TD-21.16: outcome of `extract_critique_with_repair` for the critic reply
    # that produced this object -- "parsed" (fished clean via `extract_critique`,
    # 0 repair calls), "repaired" (recovered via one constrained completion turn;
    # `decision`/`confidence`/`issues` are then exactly what the critic's own
    # reply stated, never invented -- see `require_evidence` on that turn), or
    # "failed" (unrepairable; this object carries the exact same fail-closed/
    # fail-open default `extract_critique` already produced before this
    # conversion). "" (the default) means no JSON-repair attempt was ever made
    # for this object at all -- the critic invoke failed outright before any
    # text existed to parse, or this is one of the synthetic placeholders built
    # directly (e.g. "critic circuit open"). Old `planner_archive.jsonl` rows
    # predate this field and are NOT backfilled: unlike TD-21.2/.3's action/
    # rationale fields (where "parsed" WAS the only state that ever existed
    # pre-repair), a missing critique already had non-parsed shapes
    # ("unavailable", the untouched "approve" default) before this field
    # existed, so treating a missing key as "parsed" would misrepresent
    # history -- readers must treat it as unknown, not "parsed".
    parse_status: str = ""
    repair_calls: int = 0


@dataclass
class PlannerDecision:
    action: dict[str, Any] | None
    rationale: dict[str, Any]
    session_id: str | None
    canonical_text: str
    draft_text: str
    draft_provider: str
    mode: str
    degraded: bool = False
    fallback_reason: str = ""
    critic_provider: str = ""
    critique: PlannerCritique | None = None
    predicted_objectives: dict[str, float] = field(default_factory=dict)
    # The planner's ORIGINAL parsed action, before _reconcile may substitute it
    # (revised_action in revise, or the metric fallback in reject). The main loop
    # uses this to record a critic-rejected/revised draft into the invalid-action
    # feedback + blacklist so a substituted draft cannot silently escape the
    # feedback loop (draft_critique authority change, 2026-06-04).
    draft_action: dict[str, Any] | None = None
    # True iff `action is None` AND every model we attempted was an AVAILABILITY
    # failure (timeout / empty / rc / not-found / exception, or circuit-open meant
    # we could not even attempt) — i.e. NO attempted model produced a usable
    # RESPONSE at all. False when some model responded ok but with bad/unparseable
    # content. The main loop uses this to pick deterministic-fallback vs pause when
    # both planner models are offline (cross-model failover, 2026-06-12).
    providers_unavailable: bool = False
    # Ordered provider attempt telemetry for planner_archive.jsonl. This is
    # observability only; dispatch semantics stay derived from action/critique.
    provider_trace: list[dict[str, Any]] = field(default_factory=list)
    deterministic_block_reason: str = ""
    # TD-21.2/.3: outcome of `extract_action_with_repair` / `extract_rationale_with_repair`
    # for the draft that ultimately supplied `action`/`rationale` -- "parsed" (fished
    # clean, 0 repair calls), "repaired" (recovered via one constrained completion),
    # or "failed" (unrepairable; action/rationale carries the same pre-repair default
    # as before this conversion). Persisted into planner_archive.jsonl by
    # `_archive_decision` so a repaired record is distinguishable from a clean parse.
    # Old journal rows predate this field entirely -- readers must treat a MISSING
    # field as "parsed", the only state that existed before TD-21.2/.3 landed.
    action_parse_status: str = "parsed"
    action_repair_calls: int = 0
    rationale_parse_status: str = "parsed"
    rationale_repair_calls: int = 0


ProviderFactory = Callable[[str], PlannerProvider]


def load_planner_settings_from_env() -> PlannerSettings:
    return PlannerSettings(
        primary=os.environ.get("AUTOPILOT_PLANNER_PRIMARY", "claude"),
        critic=os.environ.get("AUTOPILOT_PLANNER_CRITIC", "codex"),
        mode=os.environ.get("AUTOPILOT_PLANNER_MODE", "draft_critique"),
        critique_policy=os.environ.get(
            "AUTOPILOT_PLANNER_CRITIQUE_POLICY",
            "medium_plus",
        ),
        circuit_failures=_env_int("AUTOPILOT_PLANNER_CIRCUIT_FAILURES", 2),
        circuit_cooldown_s=_env_float("AUTOPILOT_PLANNER_CIRCUIT_COOLDOWN_S", 900.0),
    )


def _apply_planner_spend_breaker(
    settings: PlannerSettings,
    planner_state: dict[str, Any],
) -> tuple[PlannerSettings, bool]:
    if not _env_bool("AUTOPILOT_PLANNER_SPEND_BREAKER", False):
        planner_state.pop("_spend_breaker", None)
        return settings, False

    try:
        from scripts.economics.ledger import summarize_economics

        ledger = summarize_economics(days=_env_int("AUTOPILOT_PLANNER_SPEND_DAYS", 7))
    except Exception as exc:
        log.warning("Planner spend breaker unavailable: %s", exc)
        planner_state["_spend_breaker"] = {
            "active": False,
            "error": str(exc),
        }
        return settings, False

    review = ledger.review
    rules = ledger.rules
    active = bool(review.planner_spend_triggered)
    state = {
        "active": active,
        "projected_monthly_planner_spend_usd": round(
            review.projected_monthly_planner_spend_usd,
            6,
        ),
        "threshold_usd": round(rules.planner_monthly_spend_threshold_usd, 6),
        "planner_spend_usd": round(ledger.planner.total_usd, 6),
        "days": ledger.days,
    }
    planner_state["_spend_breaker"] = state
    if not active:
        return settings, False

    local_primary = _normalize_provider(
        os.environ.get(
            "AUTOPILOT_PLANNER_SPEND_BREAKER_PRIMARY",
            DEFAULT_SPEND_BREAKER_LOCAL_PRIMARY,
        )
    )
    spend_breaker_critic = os.environ.get("AUTOPILOT_PLANNER_SPEND_BREAKER_CRITIC")
    configured_critic = _normalize_provider(settings.critic)
    local_critic = _normalize_provider(
        spend_breaker_critic
        if spend_breaker_critic is not None
        else (
            configured_critic
            if _model_of(configured_critic) == "local"
            else DEFAULT_SPEND_BREAKER_LOCAL_CRITIC
        )
    )
    state["local_primary"] = local_primary
    state["local_critic"] = local_critic
    state["previous_primary"] = _normalize_provider(settings.primary)
    state["previous_critic"] = _normalize_provider(settings.critic)
    log.warning(
        "Planner spend breaker active: projected monthly planner spend $%.2f "
        "exceeds threshold $%.2f; using local providers %s/%s",
        review.projected_monthly_planner_spend_usd,
        rules.planner_monthly_spend_threshold_usd,
        local_primary,
        local_critic,
    )
    return (
        replace(
            settings,
            primary=local_primary,
            critic=local_critic,
        ),
        True,
    )


def _write_trial_planning_banner(trial_id: int | None) -> None:
    """Emit a big, scannable ``>>>> TRIAL N <<<<`` header into the planner tap.

    Best-effort and self-contained: opens its own append handle so it never
    disturbs the per-provider tap writes. The wide arrow rule makes the start of
    each trial's planning trivial to spot while scrolling the raw log or the
    dashboard planner panel.
    """
    if trial_id is None:
        return
    label = f"TRIAL {trial_id}"
    ts = datetime.now().isoformat(timespec="seconds")
    banner = f"\n{'>' * 24} {label} {'<' * 24}\n[{ts}] planning cycle start\n"
    tap = _open_planner_tap()
    if tap is None:
        return
    try:
        tap.write(banner)
        tap.flush()
    finally:
        try:
            tap.close()
        except Exception:
            pass


def plan_with_providers(
    prompt: str,
    *,
    session_id: str | None,
    timeout: int = DEFAULT_PLANNER_TIMEOUT,
    cwd: Path | str | None = None,
    planner_state: dict[str, Any] | None = None,
    stagnation_signal: str = "",
    settings: PlannerSettings | None = None,
    provider_factory: ProviderFactory = get_planner_provider,
    allowed_action_types: Iterable[str] | None = None,
    action_feedback_state: dict[str, Any] | None = None,
    trial_id: int | None = None,
    vidya_kvq_context: dict[str, Any] | None = None,
) -> PlannerDecision:
    """Draft a canonical planner action with optional secondary critique."""
    # One deterministic, unmistakable banner per live planning cycle. Tests and
    # helper calls omit trial_id, so they do not write to the shared planner tap.
    _write_trial_planning_banner(trial_id)
    settings = settings or load_planner_settings_from_env()
    planner_state = planner_state if planner_state is not None else {}
    action_feedback_state = action_feedback_state if action_feedback_state is not None else {}
    settings, spend_breaker_active = _apply_planner_spend_breaker(settings, planner_state)
    session_update = None
    allowed_actions = (
        frozenset(str(item) for item in allowed_action_types)
        if allowed_action_types is not None
        else None
    )

    primary_name = _normalize_provider(settings.primary)
    critic_name = _normalize_provider(settings.critic)
    # Compare MODELS, not provider names: the failover draft MUST target a
    # different underlying model. With PRIMARY=codex + CRITIC=codex_critic the
    # names differ but both resolve to the codex binary — a name-based fallback
    # would re-hit codex when codex is offline → "no usable draft action".
    fallback_name = _draft_fallback_provider_name(
        primary_name,
        critic_name,
        spend_breaker_active=spend_breaker_active,
    )
    allow_fallback = settings.mode.strip().lower() != "single"

    draft_provider_name = primary_name
    fallback_reason = ""
    primary_circuit_open_before_draft = _circuit_is_open(planner_state, primary_name)
    if allow_fallback and primary_circuit_open_before_draft:
        draft_provider_name = fallback_name
        fallback_reason = f"{primary_name} circuit open"

    draft_provider = provider_factory(draft_provider_name)
    draft_resume_id = session_id if draft_provider.supports_resume else None
    draft = draft_provider.invoke(
        prompt,
        role="draft",
        session_id=draft_resume_id,
        timeout=timeout,
        cwd=cwd,
    )
    if draft_provider.supports_resume:
        session_update = draft.session_id

    # Track whether ANY attempted model produced a usable RESPONSE (result.ok),
    # regardless of whether its content parsed. Distinguishes "both planner models
    # unavailable" (deterministic fallback / pause) from "models reachable but
    # drafted bad content" (keep the seed_batch default). (2026-06-12)
    any_response_ok = bool(draft.ok)
    provider_trace: list[dict[str, Any]] = []

    # TD-21.2/.3: the completer is only BUILT here (no network call yet -- it
    # only fires if `extract_action`/`extract_rationale` misses and a repair
    # turn is actually attempted). Built once per planning cycle and reused
    # across the primary draft, the fallback draft, and the rationale sidecar.
    repair_complete = action_repair_completer()

    draft_action_repair = extract_action_with_repair(draft.text, complete=repair_complete)
    action = draft_action_repair.value
    action_parse_status = draft_action_repair.status
    action_repair_calls = draft_action_repair.repair_calls
    deterministic_block_reason = ""
    draft_unusable = _draft_unusable_reason(
        draft,
        action,
        allowed_action_types=allowed_actions,
        action_feedback_state=action_feedback_state,
    )
    if _is_deterministic_action_block_reason(draft_unusable):
        deterministic_block_reason = draft_unusable
    provider_trace.append(
        _draft_provider_event(
            stage="draft_primary",
            result=draft,
            action=action,
            unusable_reason=draft_unusable,
            parse_status=action_parse_status,
            repair_calls=action_repair_calls,
        )
    )
    if draft_unusable:
        _mark_failure(planner_state, draft.provider, settings)
        if (
            allow_fallback
            and draft_provider_name != fallback_name
            and not _circuit_is_open(
                planner_state,
                fallback_name,
            )
        ):
            fallback_provider = provider_factory(fallback_name)
            fallback = fallback_provider.invoke(
                prompt,
                role="draft",
                session_id=session_update if fallback_provider.supports_resume else None,
                timeout=timeout,
                cwd=cwd,
            )
            any_response_ok = any_response_ok or bool(fallback.ok)
            fallback_action_repair = extract_action_with_repair(fallback.text, complete=repair_complete)
            fallback_action = fallback_action_repair.value
            fallback_unusable = _draft_unusable_reason(
                fallback,
                fallback_action,
                allowed_action_types=allowed_actions,
                action_feedback_state=action_feedback_state,
            )
            if _is_deterministic_action_block_reason(fallback_unusable):
                deterministic_block_reason = fallback_unusable
            provider_trace.append(
                _draft_provider_event(
                    stage="draft_fallback",
                    result=fallback,
                    action=fallback_action,
                    unusable_reason=fallback_unusable,
                    parse_status=fallback_action_repair.status,
                    repair_calls=fallback_action_repair.repair_calls,
                )
            )
            if not fallback_unusable:
                session_update = fallback.session_id if fallback_provider.supports_resume else None
                fallback_reason = (
                    fallback_reason or f"{draft.provider} draft failed: {draft_unusable}"
                )
                _mark_success(planner_state, fallback.provider)
                draft = fallback
                action = fallback_action
                action_parse_status = fallback_action_repair.status
                action_repair_calls = fallback_action_repair.repair_calls
                deterministic_block_reason = ""
            else:
                _mark_failure(planner_state, fallback.provider, settings)
        else:
            fallback_reason = fallback_reason or (
                f"{draft.provider} draft failed: {draft_unusable}"
            )
    else:
        _mark_success(planner_state, draft.provider)

    rationale_repair = extract_rationale_with_repair(draft.text, complete=repair_complete)
    rationale = rationale_repair.value
    rationale_parse_status = rationale_repair.status
    rationale_repair_calls = rationale_repair.repair_calls
    canonical_text = draft.text
    critique: PlannerCritique | None = None
    degraded = False
    # Snapshot the planner's ORIGINAL action before _reconcile can substitute it
    # (a dict copy so later mutation of `action` cannot alias it).
    draft_action = dict(action) if isinstance(action, dict) else action

    if not action:
        # providers_unavailable iff NO attempted model produced a usable response
        # at all (every attempt was an availability failure / circuit-open). If any
        # model responded ok but with bad content, this is a CONTENT failure → leave
        # it False (the loop keeps its seed_batch default). (2026-06-12)
        providers_unavailable = not any_response_ok
        no_action_reason = (
            "both planner models unavailable (no usable response from any attempted model)"
            if providers_unavailable
            else (fallback_reason or "no usable draft action")
        )
        decision = PlannerDecision(
            action=None,
            rationale=rationale,
            session_id=session_update,
            canonical_text=canonical_text,
            draft_text=draft.text,
            draft_provider=draft.provider,
            mode=settings.mode,
            degraded=True,
            fallback_reason=no_action_reason,
            predicted_objectives=peaf.extract_predicted_objectives(draft.text),
            providers_unavailable=providers_unavailable,
            provider_trace=provider_trace,
            deterministic_block_reason=deterministic_block_reason,
            action_parse_status=action_parse_status,
            action_repair_calls=action_repair_calls,
            rationale_parse_status=rationale_parse_status,
            rationale_repair_calls=rationale_repair_calls,
        )
        _archive_decision(decision, planner_state, trial_id=trial_id,
                          vidya_kvq_context=vidya_kvq_context)
        return decision

    if _should_critique(settings, action, stagnation_signal):
        active_critique = settings.mode.strip().lower() == "draft_critique"
        critique_provider_name = critic_name
        allow_primary_fallback_critique = False
        if draft.provider == critic_name:
            # A fallback draft from the configured critic provider is not
            # independently reviewed yet. Try the original primary as the
            # reviewer if it is available; otherwise keep fail-closed behavior.
            allow_primary_fallback_critique = (
                fallback_reason
                and primary_name != draft.provider
                and not primary_circuit_open_before_draft
            )
            if allow_primary_fallback_critique and (
                _model_of(primary_name) != "local"
                or spend_breaker_active
                or _distinct_local_roles(primary_name, draft.provider)
            ):
                critique_provider_name = primary_name
            else:
                degraded = True

        if not (degraded and critique is None):
            critic_fallback_name = _critic_fallback_provider_name(
                critique_provider_name,
                draft.provider,
            )
            if (
                _circuit_is_open(planner_state, critique_provider_name)
                and not allow_primary_fallback_critique
                and critic_fallback_name
                and not _circuit_is_open(planner_state, critic_fallback_name)
            ):
                critique_provider_name = critic_fallback_name
            elif (
                _circuit_is_open(planner_state, critique_provider_name)
                and not allow_primary_fallback_critique
            ):
                # Critic circuit is open (it failed repeatedly and is cooling down),
                # but the PRIMARY draft succeeded. Treat as critic-unavailable on a
                # TRUSTED draft: keep the draft and let the dispatch gate proceed for
                # low/medium-risk actions (pause only for HIGH risk) rather than
                # discarding a good primary draft for a stale seed_batch fallback.
                # (2026-06-10)
                degraded = True
                critique = PlannerCritique(
                    decision="unavailable",
                    provider=critique_provider_name,
                    parse_error="critic circuit open",
                )
            else:
                critic_provider = provider_factory(critique_provider_name)
                critique_prompt = build_critique_prompt(
                    prompt,
                    draft.text,
                    action,
                    rationale,
                    allowed_action_types=allowed_actions,
                )
                critique_result = critic_provider.invoke(
                    critique_prompt,
                    role="critique",
                    session_id=None,
                    timeout=timeout,
                    cwd=cwd,
                )
                if critique_result.ok:
                    # TD-21.16: `extract_critique` fish first (unchanged, 0 calls
                    # on a clean verdict); on a miss, ONE constrained repair turn
                    # via the SAME `repair_complete` the draft ACTION/RATIONALE
                    # already use, before ever spending the fallback-critic call
                    # below. A RECOVERED verdict's `parse_error` is empty, so it
                    # falls straight into the `else` branch below exactly like a
                    # clean parse -- no `_mark_failure`, no fallback-critic call,
                    # no circuit-breaker trip for a parse that was recoverable.
                    critique = extract_critique_with_repair(
                        critique_result.text, complete=repair_complete
                    )
                    critique.provider = critique_result.provider
                    provider_trace.append(
                        _critique_provider_event(
                            stage="critique_primary",
                            result=critique_result,
                            critique=critique,
                        )
                    )
                    if critique.parse_error:
                        # The critic invoke "succeeded" (nonzero text) and the TD-21.16
                        # repair turn above ALSO could not recover a valid verdict --
                        # e.g. Codex emitted prose or an error message instead of the
                        # json:autopilot_critique fence, and no evidence-checked repair
                        # was possible. This is an UNRECOVERABLE FAILED REVIEW, not a
                        # rejection of the draft. Mark failure (feeds circuit breaker)
                        # and degrade. In binding (draft_critique) mode flag the verdict
                        # "unavailable" and KEEP the trusted-primary draft: the dispatch
                        # gate then proceeds for low/medium-risk actions and pauses only
                        # for HIGH-risk ones. We deliberately do NOT force a `reject` →
                        # seed_batch fallback: a critic *failure* must not discard a good
                        # primary draft, and the stale-seed substitution is what
                        # re-triggered the critic_reject_loop halt @708. A genuine parsed
                        # `reject` (no parse_error) still routes to the safe fallback
                        # below. (2026-06-10; repair turn added 2026-09-24)
                        _mark_failure(planner_state, critique_result.provider, settings)
                        fallback_critique = _try_fallback_critic(
                            provider_factory=provider_factory,
                            fallback_name=critic_fallback_name,
                            failed_provider=critique_result.provider,
                            critique_prompt=critique_prompt,
                            timeout=timeout,
                            cwd=cwd,
                            planner_state=planner_state,
                            settings=settings,
                            provider_trace=provider_trace,
                            complete=repair_complete,
                        )
                        if fallback_critique and not fallback_critique.parse_error:
                            critique = fallback_critique
                            action, rationale, canonical_text = _reconcile(
                                action,
                                rationale,
                                draft.text,
                                critique,
                                active=active_critique,
                                allowed_action_types=allowed_actions,
                                planner_prompt=prompt,
                                action_feedback_state=action_feedback_state,
                            )
                        else:
                            degraded = True
                            if fallback_critique:
                                critique = fallback_critique
                            if active_critique:
                                critique.decision = "unavailable"
                            # keep draft action/rationale/canonical_text (no _reconcile)
                    else:
                        # Reached by BOTH a clean fish AND a TD-21.16 REPAIRED verdict
                        # (`critique.parse_status` distinguishes the two in the archive/
                        # provider_trace) -- a recovered parse is treated as a successful
                        # review, never a failure.
                        _mark_success(planner_state, critique_result.provider)
                        action, rationale, canonical_text = _reconcile(
                            action,
                            rationale,
                            draft.text,
                            critique,
                            active=active_critique,
                            allowed_action_types=allowed_actions,
                            planner_prompt=prompt,
                            action_feedback_state=action_feedback_state,
                        )
                else:
                    # The critic invoke FAILED outright (timeout / empty / nonzero rc).
                    # Same principle as the parse_error branch: a failed *review* must
                    # not discard the trusted-primary draft. Binding mode → verdict
                    # "unavailable" + KEEP the draft (gate proceeds for low/medium risk,
                    # pauses for HIGH risk). Shadow mode → non-binding, fail-open to the
                    # draft. Neither substitutes the stale seed_batch fallback. (2026-06-10)
                    _mark_failure(planner_state, critique_result.provider, settings)
                    provider_trace.append(
                        _critique_provider_event(
                            stage="critique_primary",
                            result=critique_result,
                            critique=None,
                        )
                    )
                    fallback_critique = _try_fallback_critic(
                        provider_factory=provider_factory,
                        fallback_name=critic_fallback_name,
                        failed_provider=critique_result.provider,
                        critique_prompt=critique_prompt,
                        timeout=timeout,
                        cwd=cwd,
                        planner_state=planner_state,
                        settings=settings,
                        provider_trace=provider_trace,
                        complete=repair_complete,
                    )
                    if fallback_critique and not fallback_critique.parse_error:
                        critique = fallback_critique
                        action, rationale, canonical_text = _reconcile(
                            action,
                            rationale,
                            draft.text,
                            critique,
                            active=active_critique,
                            allowed_action_types=allowed_actions,
                            planner_prompt=prompt,
                            action_feedback_state=action_feedback_state,
                        )
                    else:
                        degraded = True
                        critique = fallback_critique or PlannerCritique(
                            decision="unavailable" if active_critique else "approve",
                            raw_text=critique_result.text,
                            provider=critique_result.provider,
                            parse_error=critique_result.error or "critique failed",
                        )
                        if active_critique:
                            critique.decision = "unavailable"
                        # keep draft action/rationale/canonical_text (no _reconcile)

    decision = PlannerDecision(
        action=action,
        rationale=rationale,
        session_id=session_update,
        canonical_text=canonical_text,
        draft_text=draft.text,
        draft_provider=draft.provider,
        mode=settings.mode,
        degraded=degraded or bool(fallback_reason),
        fallback_reason=fallback_reason,
        critic_provider=critique.provider if critique else "",
        critique=critique,
        predicted_objectives=peaf.extract_predicted_objectives(canonical_text),
        draft_action=draft_action,
        provider_trace=provider_trace,
        action_parse_status=action_parse_status,
        action_repair_calls=action_repair_calls,
        rationale_parse_status=rationale_parse_status,
        rationale_repair_calls=rationale_repair_calls,
    )
    block_reason = _final_action_block_reason(
        decision.action,
        draft_action=decision.draft_action,
        critique=decision.critique,
        binding=decision.mode.strip().lower() == "draft_critique",
        allowed_action_types=allowed_actions,
        action_feedback_state=action_feedback_state,
    )
    if block_reason:
        decision.action = None
        decision.degraded = True
        decision.deterministic_block_reason = block_reason
        guard_reason = f"{DETERMINISTIC_GUARD_PREFIX} {block_reason}"
        decision.fallback_reason = (
            guard_reason
            if not decision.fallback_reason
            else f"{decision.fallback_reason}; {guard_reason}"
        )
    _archive_decision(decision, planner_state, trial_id=trial_id,
                      vidya_kvq_context=vidya_kvq_context)
    return decision


def uncritiqued_dispatch_block_reason(
    decision: PlannerDecision,
    *,
    is_blacklisted: bool = False,
    is_repeated: bool = False,
) -> str:
    """Pause reason if a DEGRADED decision must NOT dispatch its action.

    Two distinct degraded shapes, handled differently (2026-06-10):

    * NO critique object at all (`critique is None`): the PRIMARY failed to draft
      and the draft fell back to the critic provider (or the critic circuit opened
      before any draft existed). The action is uncritiqued AND not from the trusted
      primary → dispatch ONLY if explicitly observational (OBSERVATIONAL_ACTIONS,
      empty by default), else pause `critic_unavailable` for operator review.
      seed_batch is deliberately not observational (low-risk seed looping was the
      failure the critic catches — critic_reject_loop @708).

    * Critic verdict `"unavailable"`: the PRIMARY drafted fine but the binding
      critic could not render a verdict (timeout / empty / unparseable / circuit
      open). A failed *review* must not discard a good primary draft — but RISK
      CLASS ALONE IS NOT A SUFFICIENT GUARD (the @708 failure was "low-risk" seed
      looping). Tightened dispatch rule:
        - HIGH risk → pause.
        - seed_batch / passive low-risk actions (the loop-prone class) → pause,
          unless explicitly observational + one-shot (OBSERVATIONAL_ACTIONS).
        - MEDIUM-risk experiment → proceed ONLY IF novel and non-looping:
          not `is_blacklisted`, not `is_repeated` (a recurring invalid signature),
          and carrying a real `falsifier` hypothesis; else pause.
      `is_blacklisted` / `is_repeated` are supplied by the caller (which holds the
      blacklist + invalid-signature state). Shadow (non-binding) mode never blocks.

    Returns "" when a real verdict exists (approve/reject/revise, already
    reconciled), when not degraded, or when there is no dict action (the no-action
    path is handled separately by the caller).
    """
    if not decision.degraded:
        return ""
    action = decision.action
    if not isinstance(action, dict):
        return ""
    crit = decision.critique
    # Case A — no trusted primary draft (no critique object).
    if crit is None:
        if decision.fallback_reason and _model_of(decision.draft_provider) == "codex":
            # Operator-approved rollout mode: if the local drafter fails but Codex
            # produces a schema-valid fallback draft, dispatch it and learn from
            # the visible provider/fallback telemetry instead of pausing the loop.
            return ""
        if action.get("type") in OBSERVATIONAL_ACTIONS:
            return ""
        return "critic_unavailable"
    # Case B — trusted primary draft, but the binding critic could not review it.
    if crit.decision == "unavailable":
        # A fallback draft whose independent reviewer was also unavailable is not
        # a trusted-primary draft. Keep the stricter Case-A gate.
        if decision.fallback_reason:
            return "" if action.get("type") in OBSERVATIONAL_ACTIONS else "critic_unavailable"
        if decision.mode.strip().lower() != "draft_critique":
            return ""  # shadow / non-binding: advisory critic, never blocks
        atype = action.get("type", "")
        risk = action_risk(action)
        # HIGH risk → never auto-run an unreviewed structural/code/registry change.
        if risk == "high":
            return "critic_unavailable"
        # seed_batch / passive low-risk actions are the LOOP-PRONE class the critic
        # exists to catch (the @708 failure was "low-risk" seed looping). Risk class
        # alone is NOT enough — pause unless the action is EXPLICITLY observational
        # and one-shot (membership in OBSERVATIONAL_ACTIONS, empty by default).
        if atype in LOW_RISK_ACTIONS:
            return "" if atype in OBSERVATIONAL_ACTIONS else "critic_unavailable"
        # MEDIUM-risk experiment may proceed ONLY IF it is genuinely novel and not
        # loop-keeping: not blacklisted, not a repeated (recurring-invalid) action,
        # and carrying a real falsifiable hypothesis. Otherwise pause for review.
        if risk == "medium":
            if is_blacklisted or is_repeated:
                return "critic_unavailable"
            rationale = decision.rationale if isinstance(decision.rationale, dict) else {}
            if not str(rationale.get("falsifier", "")).strip():
                return "critic_unavailable"  # no hypothesis → treat as loop-keeping
            return ""
        # Any other class → pause (safe default).
        return "critic_unavailable"
    # A real verdict was already reconciled upstream → never blocks here.
    return ""


def build_critique_prompt(
    planner_prompt: str,
    draft_text: str,
    action: dict[str, Any],
    rationale: dict[str, Any],
    *,
    allowed_action_types: Iterable[str] | None = None,
) -> str:
    selected_context = _selected_critique_context(planner_prompt)
    action_type = str(action.get("type", ""))
    known_actions_text = ", ".join(sorted(KNOWN_ACTION_TYPES))
    if allowed_action_types is None:
        selectable_actions_text = "(not supplied; use the Action Availability section)"
    else:
        selectable_actions_text = ", ".join(sorted({str(item) for item in allowed_action_types}))
    return f"""\
You are the secondary AutoPilot planner reviewer.

Review the draft below. Do not invent an independent competing plan. Your job
is to find missing constraints, unsafe assumptions, weak attribution, missing
validation, stale context, or an action that violates the single-variable rule.

Your verdict is BINDING (draft_critique mode): a `reject` routes the action to a
metric-bearing fallback, and a `revise` with a valid `revised_action` REPLACES the draft.
Reject or revise (preferring a concrete `revised_action`) if the draft does any
of the following — the relevant evidence is in the selected context below:
  - re-proposes a feature flag whose dependencies are not all currently ON (see
    the "Feature Flags" section: live state + dependency rules). Prefer a
    `revised_action` that enables the missing dependency first (one flag/trial).
  - repeats the "Last Non-Executing Action" (the validator/dispatch already
    rejected it; re-proposing it burns a trial and will be auto-blacklisted).
  - matches a "Blacklisted Configurations" entry.
  - cites below-MDE or single-trial evidence as decisive without a reproduction
    plan; use the "Evidence Power" section to reject unmeasurable proposals.
Reject or revise a `numeric_trial` with missing or empty `"params"`. Planner
output must carry exactly one explicit parameter for the selected surface; the
Optuna empty-param sampler path is reserved for internal deterministic fallbacks,
not model-authored planner actions.
Do NOT manufacture host-noise / contention narratives when System Health is
nominal, and do NOT propose operator-domain actions (widening safety-gate
thresholds, baseline refresh) — those are outside the autopilot action space.
Do NOT reject solely because the shared worktree has unrelated dirty files.
The dispatcher has a target-path dirty fence: cite dirty state only when the
parsed action would write/stage that same target path or prompt directory.
Do NOT reject a draft by claiming its parsed action type is unrecognized,
non-standard, or not in the AutoPilot schema when it appears in the
authoritative action-type list below. If the selected context says a known
action is temporarily unavailable or cannot satisfy the current W8 replay
pressure, name that concrete availability/evidence reason instead.
`seed_batch` and `deep_eval` are valid known action types. They may be the wrong
choice for a replayable W8 candidate, but they are not schema errors.

Return JSON ONLY in this fenced block:

```json:autopilot_critique
{{"decision":"approve|revise|reject",
 "confidence":0.0,
 "issues":["short issue"],
 "revised_action":null,
 "revised_rationale":null}}
```

Use revise only when you can provide a better single canonical action. Use
reject only when the draft is unsafe or unsupported and no safe revision is
available.

## Selected Measurement and Constraint Context

{selected_context}

## Authoritative Action-Type Check

- parsed_action_type: `{action_type}`
- known_action_types: `{known_actions_text}`
- currently_selectable_action_types: `{selectable_actions_text}`

## Draft Text

{draft_text}

## Parsed Draft Action

{json.dumps(action, indent=2, sort_keys=True)}

## Parsed Draft Rationale

{json.dumps(rationale, indent=2, sort_keys=True)}
"""


def _critic_fallback_provider_name(
    critic_provider_name: str,
    draft_provider_name: str,
) -> str:
    configured = os.environ.get("AUTOPILOT_PLANNER_CRITIC_FALLBACK")
    if configured is None:
        if _model_of(critic_provider_name) != "codex":
            return ""
        configured = DEFAULT_CODEX_CRITIC_FALLBACK

    fallback = _normalize_provider(configured)
    if fallback in {"", "0", "false", "no", "none", "off"}:
        return ""
    if fallback == _normalize_provider(critic_provider_name):
        return ""
    if _model_of(fallback) == _model_of(draft_provider_name):
        return ""
    return fallback


def _try_fallback_critic(
    *,
    provider_factory: ProviderFactory,
    fallback_name: str,
    failed_provider: str,
    critique_prompt: str,
    timeout: int,
    cwd: Path | str | None,
    planner_state: dict[str, Any],
    settings: PlannerSettings,
    provider_trace: list[dict[str, Any]] | None = None,
    complete: CompleteFn,
) -> PlannerCritique | None:
    if not fallback_name or fallback_name == _normalize_provider(failed_provider):
        return None
    if _circuit_is_open(planner_state, fallback_name):
        return None

    fallback_provider = provider_factory(fallback_name)
    fallback_result = fallback_provider.invoke(
        critique_prompt,
        role="critique",
        session_id=None,
        timeout=timeout,
        cwd=cwd,
    )
    if not fallback_result.ok:
        _mark_failure(planner_state, fallback_result.provider, settings)
        if provider_trace is not None:
            provider_trace.append(
                _critique_provider_event(
                    stage="critique_fallback",
                    result=fallback_result,
                    critique=None,
                )
            )
        return PlannerCritique(
            decision="unavailable",
            raw_text=fallback_result.text,
            provider=fallback_result.provider,
            parse_error=(
                f"{failed_provider} critique failed; "
                f"{fallback_result.provider} fallback failed: "
                f"{fallback_result.error or 'critique failed'}"
            ),
        )

    # TD-21.16: same repair-before-failure shape as the primary critique call
    # above -- a malformed-but-repairable fallback verdict recovers via ONE
    # constrained completion turn and its `parse_error` comes back empty, so
    # it falls straight to `_mark_success` below like a clean parse; only an
    # UNREPAIRABLE fallback verdict still degrades to "unavailable".
    fallback_critique = extract_critique_with_repair(fallback_result.text, complete=complete)
    fallback_critique.provider = fallback_result.provider
    if provider_trace is not None:
        provider_trace.append(
            _critique_provider_event(
                stage="critique_fallback",
                result=fallback_result,
                critique=fallback_critique,
            )
        )
    if fallback_critique.parse_error:
        _mark_failure(planner_state, fallback_result.provider, settings)
        fallback_critique.decision = "unavailable"
        fallback_critique.parse_error = (
            f"{failed_provider} critique failed; "
            f"{fallback_result.provider} fallback unparseable: "
            f"{fallback_critique.parse_error}"
        )
        return fallback_critique

    _mark_success(planner_state, fallback_result.provider)
    return fallback_critique


def _selected_critique_context(planner_prompt: str) -> str:
    sections = [
        section
        for heading in _CRITIQUE_CONTEXT_HEADINGS
        if (section := _extract_markdown_section(planner_prompt, heading))
    ]
    if not sections:
        return (
            "(selected planner context unavailable; critique only the parsed draft and rationale)"
        )
    return "\n\n".join(sections)


def _extract_markdown_section(text: str, heading: str) -> str:
    start = text.find(heading)
    if start < 0:
        return ""
    tail = text[start:]
    next_heading = re.search(r"\n#{2,3}\s+", tail[len(heading) :])
    end = len(heading) + next_heading.start() if next_heading else len(tail)
    section = tail[:end].strip()
    if len(section) <= _CRITIQUE_SECTION_CHAR_LIMIT:
        return section
    return (
        section[:_CRITIQUE_SECTION_CHAR_LIMIT].rstrip() + "\n  ... [truncated for critic context]"
    )


def extract_critique(text: str) -> PlannerCritique:
    raw = text or ""
    if "```json:autopilot_critique" not in raw:
        return PlannerCritique(raw_text=raw, parse_error="autopilot_critique fence missing")
    data, error = _extract_json_payload(raw, "```json:autopilot_critique")
    if data is None:
        return PlannerCritique(raw_text=raw, parse_error=error or "no critique JSON")

    if "decision" not in data:
        return PlannerCritique(raw_text=raw, parse_error="critique decision missing")
    decision = str(data.get("decision", "")).strip().lower()
    if decision not in {"approve", "revise", "reject"}:
        return PlannerCritique(raw_text=raw, parse_error=f"invalid critique decision: {decision}")

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    issues_raw = data.get("issues", [])
    if isinstance(issues_raw, list):
        issues = [str(i) for i in issues_raw if str(i).strip()]
    elif issues_raw:
        issues = [str(issues_raw)]
    else:
        issues = []

    revised_action = data.get("revised_action")
    if not isinstance(revised_action, dict):
        revised_action = None
    revised_rationale = data.get("revised_rationale")
    if not isinstance(revised_rationale, dict):
        revised_rationale = None

    return PlannerCritique(
        decision=decision,
        confidence=confidence,
        issues=issues,
        revised_action=revised_action,
        revised_rationale=revised_rationale,
        raw_text=raw,
        parse_status="parsed",
    )


# --------------------------------------------------------------------------- TD-21.16 critic-verdict repair
#
# Same shape as TD-21.2/.3 in `controller_io.py`: fish first (`extract_critique`,
# unchanged); on a miss, ONE constrained completion turn back to a real
# llama-server via the SAME `action_repair_completer()` idiom (never the
# orchestrator's own :8000 `/v1`, which refuses `response_format` -- see
# `controller_io.py`'s TD-21.2/.3 comment block for why). The repair turn is
# available regardless of which provider (Claude CLI, Codex CLI, or a local
# HTTP provider) produced the unparseable critique: its only job is to
# re-express a reply that already exists, not to render a fresh review.

_CRITIC_REPAIR_SITE = "autopilot.planner_coordinator.extract_critique"

_CRITIC_REPAIR_INSTRUCTION = (
    "The reply given as the user message is an AutoPilot critic response that "
    "failed to parse as its intended fenced json:autopilot_critique verdict. "
    "Extract ONLY the verdict the reply intends -- `decision` (approve, "
    "revise, or reject), `confidence`, `issues`, and any "
    "`revised_action`/`revised_rationale` it proposes -- as exactly one JSON "
    "object matching the given schema. Copy the reply's own field values "
    "faithfully; do not invent a decision, confidence, issue, or revision it "
    "did not already give."
)


def autopilot_critique_schema() -> dict[str, Any]:
    """JSON schema for the fenced ``json:autopilot_critique`` verdict
    (TD-21.16), derived from exactly what `extract_critique` consumes after
    `_extract_json_payload`: a required `decision` enum (the same
    ``{approve, revise, reject}`` set `extract_critique` enforces), an
    optional `confidence` in ``[0, 1]``, an optional `issues` array of
    strings, and the two optional revision objects a `revise` verdict may
    carry. `additionalProperties: False` closes the object so a repair turn
    cannot smuggle in a field `extract_critique` would silently ignore
    anyway."""
    return {
        "type": "object",
        "properties": {
            "decision": {"enum": ["approve", "revise", "reject"]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "issues": {"type": "array", "items": {"type": "string"}},
            "revised_action": {"type": "object"},
            "revised_rationale": {"type": "object"},
        },
        "required": ["decision"],
        "additionalProperties": False,
    }


def extract_critique_with_repair(
    text: str,
    *,
    complete: CompleteFn,
    site: str = _CRITIC_REPAIR_SITE,
) -> PlannerCritique:
    """TD-21.16: `extract_critique` first (unchanged, 0 calls on a clean
    verdict); on a miss, ONE constrained completion turn against
    `autopilot_critique_schema()` before the caller's existing
    fallback-critic-provider / circuit-breaker path.

    Persisted-state / dispatch semantics (before/after this conversion):
    BEFORE, ANY unparseable critic reply (missing/garbled fence, missing or
    invalid `decision`) was a FAILED REVIEW: `_mark_failure` opened the
    critic provider's circuit-breaker counter and `_try_fallback_critic`
    re-prompted a full fallback critic call from zero -- one extra critic
    call per iteration (the handoff's "fallback provider" cost).
    AFTER, a malformed-but-repairable verdict is recovered via one
    constrained completion turn against the same local server the draft
    ACTION/RATIONALE repairs already use (`action_repair_completer()`); the
    recovered verdict's `parse_error` is empty, so the caller's existing
    `if critique.parse_error:` branch treats it EXACTLY like a clean parse --
    no `_mark_failure`, no fallback-critic call, no circuit-breaker trip. An
    UNREPAIRABLE verdict is unchanged from before: this function returns the
    SAME `PlannerCritique` `extract_critique` already produced (its
    `parse_error` names the ORIGINAL fish failure, not the repair attempt's),
    so the caller's fallback-critic-provider / circuit-breaker path fires
    exactly as it did pre-conversion -- an unrecoverable verdict is still a
    typed failure (`parse_status="failed"`), never a default verdict.

    `require_evidence=True` (same live-smoke finding TD-21.2/.3 apply):
    `confidence` is a number and `issues` are short strings -- the exact
    invented-leaf class the evidence check targets. `decision` is itself a
    short string too (not `const`-exempt, unlike an action's `type`
    discriminator), so it is ALSO evidence-checked: a repair cannot invent an
    approve/revise/reject the critic's own reply never stated in any form,
    which is the one field this conversion must never fabricate.
    """
    parsed = extract_critique(text)
    if not parsed.parse_error:
        return parsed

    result = parse_with_repair(
        text,
        schema=autopilot_critique_schema(),
        complete=complete,
        instruction=_CRITIC_REPAIR_INSTRUCTION,
        site=site,
        require_evidence=True,
    )
    if result.value is None:
        parsed.parse_status = "failed"
        parsed.repair_calls = result.repair_calls
        return parsed

    data = result.value
    decision = str(data.get("decision", "")).strip().lower()
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    issues_raw = data.get("issues", [])
    if isinstance(issues_raw, list):
        issues = [str(i) for i in issues_raw if str(i).strip()]
    else:
        issues = []
    revised_action = data.get("revised_action")
    revised_action = revised_action if isinstance(revised_action, dict) else None
    revised_rationale = data.get("revised_rationale")
    revised_rationale = revised_rationale if isinstance(revised_rationale, dict) else None

    return PlannerCritique(
        decision=decision,
        confidence=confidence,
        issues=issues,
        revised_action=revised_action,
        revised_rationale=revised_rationale,
        raw_text=text or "",
        parse_status="repaired",
        repair_calls=result.repair_calls,
    )


def action_risk(action: dict[str, Any]) -> str:
    action_type = action.get("type", "")
    if action_type == "seed_batch":
        try:
            n_questions = int(action.get("n_questions", 0))
        except (TypeError, ValueError):
            n_questions = 0
        return "medium" if n_questions > 50 else "low"
    if action_type in LOW_RISK_ACTIONS:
        return "low"
    if action_type in MEDIUM_RISK_ACTIONS:
        return "medium"
    if action_type in HIGH_RISK_ACTIONS:
        return "high"
    return "high"


def _should_critique(
    settings: PlannerSettings,
    action: dict[str, Any],
    stagnation_signal: str,
) -> bool:
    mode = settings.mode.strip().lower()
    if mode in {"single", "fallback"}:
        return False
    if mode not in {"draft_critique", "shadow_critique"}:
        return False

    policy = settings.critique_policy.strip().lower()
    risk = action_risk(action)
    stagnating = bool(stagnation_signal and stagnation_signal != "none")
    if policy == "always":
        return True
    if policy == "stagnation":
        return stagnating
    if policy == "high_risk":
        return risk == "high" or stagnating
    if policy == "medium_plus":
        return risk in {"medium", "high"} or stagnating
    return False


def _reconcile(
    action: dict[str, Any],
    rationale: dict[str, Any],
    draft_text: str,
    critique: PlannerCritique,
    *,
    active: bool,
    allowed_action_types: Iterable[str] | None = None,
    planner_prompt: str = "",
    action_feedback_state: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    if not active or critique.decision == "approve":
        return action, rationale, draft_text

    revised = critique.revised_action
    if (
        revised
        and _revision_validation_error(
            revised,
            allowed_action_types=allowed_action_types,
            planner_prompt=planner_prompt,
            action_feedback_state=action_feedback_state,
        )
        is None
    ):
        new_rationale = critique.revised_rationale or rationale
        return revised, new_rationale, _canonical_text_from_parts(revised, new_rationale)

    if critique.decision in {"reject", "revise"}:
        safe_action = _safe_numeric_fallback_action()
        safe_rationale = {
            "falsifier": "critic reject numeric fallback fails to produce replayable evidence",
            "rubric_scores": {},
            "critic_reject_numeric_fallback": True,
            "critic_reject_issues": list(critique.issues or []),
            "critic_reject_original_action": dict(action),
        }
        return (
            safe_action,
            safe_rationale,
            _canonical_text_from_parts(
                safe_action,
                safe_rationale,
            ),
        )

    return action, rationale, draft_text


def _revision_validation_error(
    action: dict[str, Any],
    *,
    allowed_action_types: Iterable[str] | None = None,
    planner_prompt: str = "",
    action_feedback_state: dict[str, Any] | None = None,
) -> str | None:
    schema_error = _action_validation_error(
        action,
        allowed_action_types=allowed_action_types,
    )
    if schema_error:
        return schema_error
    hard_block = _planner_action_hard_block_reason(
        action,
        action_feedback_state=action_feedback_state,
    )
    if hard_block:
        return hard_block
    return _structural_noop_revision_error(action, planner_prompt)


def _structural_noop_revision_error(
    action: dict[str, Any],
    planner_prompt: str,
) -> str | None:
    if action.get("type") != "structural_experiment":
        return None
    flags = action.get("flags")
    if not isinstance(flags, dict) or len(flags) != 1:
        return None
    flag, value = next(iter(flags.items()))
    if not isinstance(flag, str) or not isinstance(value, bool):
        return None
    live_state = _feature_flag_state_from_prompt(planner_prompt, flag)
    if live_state is None:
        return None
    if live_state is value:
        return (
            f"structural_experiment would not change live flag state: {flag}={str(value).lower()}"
        )
    return None


def _feature_flag_state_from_prompt(prompt: str, flag: str) -> bool | None:
    escaped = re.escape(flag)
    current_match = re.search(
        rf"\b{escaped}\b[^\n]{{0,120}}\(currently\s+(ON|OFF)\)",
        prompt,
        flags=re.IGNORECASE,
    )
    if current_match:
        return current_match.group(1).upper() == "ON"
    invalid_match = re.search(
        rf"would not change live flag state:\s*{escaped}\s*=\s*(true|false)",
        prompt,
        flags=re.IGNORECASE,
    )
    if invalid_match:
        return invalid_match.group(1).lower() == "true"
    return None


def _canonical_text_from_parts(
    action: dict[str, Any],
    rationale: dict[str, Any],
) -> str:
    return (
        "```json:autopilot_actions\n"
        f"{json.dumps(action, indent=2, sort_keys=True)}\n"
        "```\n\n"
        "```json:autopilot_rationale\n"
        f"{json.dumps(rationale, indent=2, sort_keys=True)}\n"
        "```\n"
    )


def _draft_is_usable(
    result: PlannerProviderResult,
    action: dict[str, Any] | None,
) -> bool:
    return not _draft_unusable_reason(result, action)


def _draft_unusable_reason(
    result: PlannerProviderResult,
    action: dict[str, Any] | None,
    *,
    allowed_action_types: Iterable[str] | None = None,
    action_feedback_state: dict[str, Any] | None = None,
) -> str:
    """Why a draft is unusable ("" if usable). Surfaces the EXACT schema-validation
    error (e.g. an out-of-range min_memories) rather than an opaque 'invalid action'
    — so the planner fallback log + the next trial's 'Last Non-Executing Action'
    feedback can self-correct instead of re-drafting the same hidden-cap violation
    (root cause of the #776 train_routing_models pause, 2026-06-11)."""
    if not result.ok:
        return result.error or "provider error / empty response"
    if action is None:
        return "no parseable json:autopilot_actions block"
    schema_error = _action_validation_error(
        action,
        allowed_action_types=allowed_action_types,
    )
    if schema_error:
        return schema_error
    return (
        _planner_action_hard_block_reason(
            action,
            action_feedback_state=action_feedback_state,
        )
        or ""
    )


def _action_validation_error(
    action: dict[str, Any] | None,
    *,
    allowed_action_types: Iterable[str] | None = None,
) -> str | None:
    if not action:
        return "missing action"
    action_type = action.get("type")
    if action_type not in KNOWN_ACTION_TYPES:
        return f"unknown action type: {action_type}"
    if allowed_action_types is not None:
        allowed = {str(item) for item in allowed_action_types}
        if action_type not in allowed:
            return f"action type currently unavailable: {action_type}"
    return validate_single_variable(action)


def _safe_numeric_fallback_action() -> dict[str, Any]:
    return {
        "type": "numeric_trial",
        "surface": SAFE_FALLBACK_NUMERIC_SURFACE,
        "params": dict(SAFE_FALLBACK_NUMERIC_PARAMS),
    }


def _numeric_surface_specs() -> dict[str, list[Any]]:
    try:
        from species.numeric_swarm import SURFACES
    except Exception:
        return {}
    if not isinstance(SURFACES, dict):
        return {}
    return {
        str(surface): list(specs)
        for surface, specs in SURFACES.items()
        if isinstance(surface, str) and isinstance(specs, list)
    }


def _numeric_param_label(spec: Any) -> str:
    name = str(getattr(spec, "name", "") or "")
    ptype = str(getattr(spec, "param_type", "float") or "float")
    low = getattr(spec, "low", "?")
    high = getattr(spec, "high", "?")
    return f"{name} ({ptype}, {low}-{high})"


def _numeric_param_validation_error(action: dict[str, Any]) -> str | None:
    if action.get("type") != "numeric_trial":
        return None
    params = action.get("params")
    if not isinstance(params, dict) or not params:
        return None
    if len(params) != 1:
        return None
    surface = str(action.get("surface") or "")
    specs_by_surface = _numeric_surface_specs()
    specs = specs_by_surface.get(surface)
    if not specs:
        return None

    key, value = next(iter(params.items()))
    key_s = str(key)
    full_names = {str(getattr(spec, "name", "") or ""): spec for spec in specs}
    short_names: dict[str, Any] = {}
    ambiguous: set[str] = set()
    for spec in specs:
        name = str(getattr(spec, "name", "") or "")
        if "." not in name:
            continue
        short = name.split(".", 1)[1]
        if short in short_names:
            ambiguous.add(short)
            short_names.pop(short, None)
        elif short not in ambiguous:
            short_names[short] = spec

    spec = full_names.get(key_s) or short_names.get(key_s)
    if spec is None:
        valid = ", ".join(_numeric_param_label(item) for item in specs)
        return (
            f"numeric_trial param {key_s!r} is not valid for surface {surface!r}; "
            f"valid params: {valid}. There is no numeric_trial "
            "tool_activation_threshold knob; tool-required routing is controlled "
            "by chat_routing.detect_tool_requirement(), not NumericSwarm."
        )

    ptype = str(getattr(spec, "param_type", "float") or "float")
    low = getattr(spec, "low", None)
    high = getattr(spec, "high", None)
    if ptype == "int":
        if not isinstance(value, int) or isinstance(value, bool):
            return f"numeric_trial param {key_s!r} must be int; got {value!r}"
    elif not isinstance(value, (int, float)) or isinstance(value, bool):
        return f"numeric_trial param {key_s!r} must be numeric; got {value!r}"

    if low is not None and value < low:
        return f"numeric_trial param {key_s!r} must be >= {low}; got {value!r}"
    if high is not None and value > high:
        return f"numeric_trial param {key_s!r} must be <= {high}; got {value!r}"
    return None


def _planner_action_hard_block_reason(
    action: dict[str, Any],
    *,
    action_feedback_state: dict[str, Any] | None = None,
) -> str | None:
    if not isinstance(action, dict):
        return "missing action"
    if action.get("type") == "numeric_trial":
        params = action.get("params")
        if not isinstance(params, dict) or not params:
            return (
                "numeric_trial planner output must include exactly one explicit "
                "params entry; empty params are reserved for internal sampler fallbacks"
            )
        param_error = _numeric_param_validation_error(action)
        if param_error:
            return param_error

    state = action_feedback_state if isinstance(action_feedback_state, dict) else {}
    signature = action_signature(action)
    rejected = state.get("critic_rejected_signatures")
    if isinstance(rejected, dict):
        record = rejected.get(signature)
        if isinstance(record, dict):
            trial = record.get("trial_id", "?")
            reason = str(record.get("reason") or "critic rejected this exact action")
            return (
                "exact action signature was previously critic-rejected at "
                f"trial {trial}: {reason[:220]}"
            )
    invalid_counts = state.get("invalid_signature_counts")
    if isinstance(invalid_counts, dict) and signature in invalid_counts:
        return (
            "exact action signature was previously invalid "
            f"({int(invalid_counts.get(signature) or 0)} recorded hit(s))"
        )
    return None


def _is_deterministic_action_block_reason(reason: str) -> bool:
    normalized = str(reason or "").strip().lower()
    return normalized.startswith(
        (
            "numeric_trial planner output must include",
            "exact action signature was previously",
            "action type currently unavailable:",
            "unknown action type:",
            "numeric_trial params must be an object",
            "numeric_trial sets ",
            "numeric_trial surface must be one of",
        )
    )


def _final_action_block_reason(
    action: dict[str, Any] | None,
    *,
    draft_action: dict[str, Any] | None,
    critique: PlannerCritique | None,
    binding: bool = False,
    allowed_action_types: Iterable[str] | None = None,
    action_feedback_state: dict[str, Any] | None = None,
) -> str | None:
    if not isinstance(action, dict):
        return None
    schema_error = _action_validation_error(
        action,
        allowed_action_types=allowed_action_types,
    )
    if schema_error:
        return schema_error
    hard_block = _planner_action_hard_block_reason(
        action,
        action_feedback_state=action_feedback_state,
    )
    if hard_block:
        return hard_block
    if (
        binding
        and critique is not None
        and critique.decision in {"revise", "reject"}
        and isinstance(draft_action, dict)
        and action == draft_action
    ):
        return (
            f"critique decision {critique.decision!r} left final action unchanged; "
            "revise/reject must produce a materially different dispatch action"
        )
    return None


def _extract_json_payload(
    text: str,
    marker: str,
) -> tuple[dict[str, Any] | None, str]:
    if marker in text:
        start = text.index(marker) + len(marker)
        try:
            end = text.index("```", start)
        except ValueError:
            return None, "unclosed fenced block"
        payload = text[start:end].strip()
    elif "```json" in text:
        start = text.index("```json") + len("```json")
        try:
            end = text.index("```", start)
        except ValueError:
            return None, "unclosed json block"
        payload = text[start:end].strip()
    else:
        payload = text.strip()

    try:
        data = _loads_json_payload(payload)
    except json.JSONDecodeError as exc:
        return None, str(exc)
    if not isinstance(data, dict):
        return None, "payload is not an object"
    return data, ""


def _circuit_is_open(planner_state: dict[str, Any], provider: str) -> bool:
    state = planner_state.get(provider, {})
    try:
        return float(state.get("circuit_open_until", 0.0)) > time.time()
    except (TypeError, ValueError):
        return False


def _mark_success(planner_state: dict[str, Any], provider: str) -> None:
    state = planner_state.setdefault(provider, {})
    state["last_success"] = time.time()
    state["failures"] = 0
    state["circuit_open_until"] = 0.0


def _mark_failure(
    planner_state: dict[str, Any],
    provider: str,
    settings: PlannerSettings,
) -> None:
    state = planner_state.setdefault(provider, {})
    failures = int(state.get("failures", 0)) + 1
    state["failures"] = failures
    state["last_failure"] = time.time()
    if failures >= settings.circuit_failures:
        state["circuit_open_until"] = time.time() + settings.circuit_cooldown_s


def _archive_decision(
    decision: PlannerDecision,
    planner_state: dict[str, Any],
    *,
    trial_id: int | None = None,
    vidya_kvq_context: dict[str, Any] | None = None,
) -> None:
    critique = decision.critique
    context = vidya_kvq_context or {}
    exposed = context.get("claim_ids") if context.get("status") == "available" else []
    exposed = [cid for cid in exposed if isinstance(cid, str)] if isinstance(exposed, list) else []
    declared = decision.rationale.get("vidya_claim_ids", [])
    declared = [cid for cid in declared if isinstance(cid, str)] if isinstance(declared, list) else []
    # A substituted action cannot inherit its draft's evidence declaration.
    if decision.action is None or (
        decision.draft_action != decision.action
        and not (critique and critique.revised_rationale)
    ):
        declared = []
    relied_on = sorted(set(declared) & set(exposed))
    _append_planner_archive(
        {
            "ts": time.time(),
            "type": "planner_coordinator",
            "trial_id": trial_id,
            "vidya_kvq": {
                "schema": "epyc.vidya.kvq_planner_receipt.v1",
                "status": context.get("status", "unavailable"),
                "reason": context.get("reason", "context not supplied"),
                "as_of": context.get("as_of"),
                "run": context.get("run"),
                "frontier": context.get("frontier"),
                "state_hash": context.get("state_hash"),
                "exposed_claim_ids": exposed,
                "declared_reliance_claim_ids": relied_on,
                "reliance_status": "planner_declared" if relied_on else "not_declared",
            },
            "mode": decision.mode,
            "draft_provider": decision.draft_provider,
            "critic_provider": decision.critic_provider,
            "degraded": decision.degraded,
            "fallback_reason": decision.fallback_reason,
            "action_type": (decision.action or {}).get("type"),
            "critique_decision": critique.decision if critique else "",
            "critique_confidence": critique.confidence if critique else 0.0,
            "critique_issues": critique.issues if critique else [],
            "draft_action": decision.draft_action,
            "final_action": decision.action,
            "deterministic_block_reason": decision.deterministic_block_reason,
            "provider_trace": decision.provider_trace,
            "planner_state": planner_state,
            # TD-21.2/.3: "parsed" (fished clean) / "repaired" (recovered via one
            # constrained completion) / "failed" (unrepairable; action/rationale
            # carries the same pre-repair default as before this conversion).
            # Absent on every row written before this field existed -- readers
            # must treat a missing key as "parsed", not backfill it.
            "action_parse_status": decision.action_parse_status,
            "action_repair_calls": decision.action_repair_calls,
            "rationale_parse_status": decision.rationale_parse_status,
            "rationale_repair_calls": decision.rationale_repair_calls,
            # TD-21.16: same "parsed"/"repaired"/"failed" vocabulary as the
            # action/rationale fields above, from `extract_critique_with_repair`
            # -- "" when no critique JSON was ever attempted (no critique object
            # at all, or the critic invoke failed outright before any text
            # existed to parse). UNLIKE the action/rationale fields, a missing
            # key on a row written before this field existed must NOT be read
            # as "parsed": pre-TD-21.16, an unparseable critique already had
            # other persisted shapes (`critique_decision="unavailable"`, or the
            # untouched "approve" dataclass default in shadow mode) — "parsed"
            # was never the only prior state, so absence here means unknown,
            # not "parsed". No backfill.
            "critic_parse_status": critique.parse_status if critique else "",
            "critic_repair_calls": critique.repair_calls if critique else 0,
        }
    )


def _clip(value: Any, limit: int = 300) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[:limit].rstrip() + "...[truncated]"


def _draft_provider_event(
    *,
    stage: str,
    result: PlannerProviderResult,
    action: dict[str, Any] | None,
    unusable_reason: str,
    parse_status: str = "",
    repair_calls: int = 0,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "role": result.role or "draft",
        "provider": result.provider,
        "ok": bool(result.ok),
        "text_chars": len(result.text or ""),
        "error": _clip(result.error),
        "parse_ok": action is not None,
        "action_type": (action or {}).get("type", ""),
        "unusable_reason": unusable_reason,
        # TD-21.2: "parsed" / "repaired" / "failed" from extract_action_with_repair;
        # "" for callers (none today) that don't pass one.
        "parse_status": parse_status,
        "repair_calls": repair_calls,
    }


def _critique_provider_event(
    *,
    stage: str,
    result: PlannerProviderResult,
    critique: PlannerCritique | None,
) -> dict[str, Any]:
    parse_error = critique.parse_error if critique else ""
    return {
        "stage": stage,
        "role": result.role or "critique",
        "provider": result.provider,
        "ok": bool(result.ok),
        "text_chars": len(result.text or ""),
        "error": _clip(result.error),
        "parse_ok": bool(result.ok and critique and not parse_error),
        "critique_decision": critique.decision if critique else "",
        "critique_confidence": critique.confidence if critique else 0.0,
        "parse_error": _clip(parse_error),
        # TD-21.16: "parsed" / "repaired" / "failed" from
        # `extract_critique_with_repair`; "" for a critique built without ever
        # attempting JSON repair (e.g. the invoke itself failed, or this is a
        # synthetic "critic circuit open" placeholder).
        "parse_status": critique.parse_status if critique else "",
        "repair_calls": critique.repair_calls if critique else 0,
    }


def _normalize_provider(name: str) -> str:
    return (name or "claude").strip().lower()


def _other_provider(name: str) -> str:
    return "codex" if name == "claude" else "claude"


def _distinct_local_roles(left: str, right: str) -> bool:
    return (
        _model_of(left) == "local"
        and _model_of(right) == "local"
        and _normalize_provider(left) != _normalize_provider(right)
    )


def _draft_fallback_provider_name(
    primary_name: str,
    critic_name: str,
    *,
    spend_breaker_active: bool,
) -> str:
    primary = _normalize_provider(primary_name)
    critic = _normalize_provider(critic_name)
    if spend_breaker_active:
        return critic if critic != primary else primary
    if _model_of(critic) != _model_of(primary):
        return critic
    if _distinct_local_roles(primary, critic):
        return critic
    return _other_provider(primary)


# Provider-role names that resolve to the same underlying MODEL binary. Mirrors
# the dispatch set in planner_providers.get_planner_provider: codex_critic /
# codex-critic / codex_reviewer / codex-reviewer all launch the codex binary.
_CODEX_ROLE_NAMES = {"codex", "codex_critic", "codex-critic", "codex_reviewer", "codex-reviewer"}
_LOCAL_ROLE_NAMES = {
    "local",
    "local_frontdoor",
    "frontdoor_local",
    "local_chat",
    "local_chat_planner",
    "chat_local",
    "local_worker",
    "local_worker_general",
    "worker_general_local",
    "local_ingest",
    "local_ingest_long_context",
    "ingest_local",
    "local_brief_frontdoor",
    "local_ingest_frontdoor",
    "local_two_stage",
    "local_brief_worker",
    "local_ingest_worker",
}


def _model_of(name: str) -> str:
    """Canonical underlying model for a provider-role name.

    Cross-model failover compares MODELS, not provider names. Historically the
    Codex aliases resolved to one model identity; the active sealed roster binds
    those aliases to distinct Sol and Terra model cells.
    """
    normalized = (name or "").strip().lower()
    if normalized == "claude":
        return "claude"
    if normalized in _CODEX_ROLE_NAMES:
        # Outside the temporary sealed roster, aliases retain their historical
        # same-model semantics. Under the active roster the aliases are two
        # genuinely distinct Codex model cells, so failover may use the other
        # staffed planner without touching Claude.
        active = os.environ.get("AUTOPILOT_PLANNER_ROSTER_POLICY_ACTIVE", "").strip().lower()
        if active in {"1", "true", "yes", "on"}:
            if normalized == "codex":
                model = os.environ.get("AUTOPILOT_CODEX_MODEL", "")
            else:
                model = os.environ.get("AUTOPILOT_CODEX_CRITIC_MODEL", "")
            if model:
                return f"codex:{model}"
        return "codex"
    if normalized in _LOCAL_ROLE_NAMES:
        return "local"
    return normalized


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}
