"""3-Way evaluation logic: _eval_single_config, evaluate_question_3way, ThreeWayResult.

Contains the core eval loop and the deduped ``_build_role_result`` helper
that replaces the previously copy-pasted RoleResult construction in
``_eval_single_config``'s first-attempt and retry paths.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from seeding_types import (
    ACTION_ARCHITECT,
    ACTION_SELF_DIRECT,
    ACTION_SELF_REPL,
    HEAVY_PORTS,
    ROLE_PORT,
    RoleResult,
    state,
)
from seeding_scoring import (
    _adaptive_timeout_s,
    _bump_timeout_from_observed,
    _classify_error,
    _is_coding_task,
    score_answer_deterministic,
)
from seeding_orchestrator import (
    _busy_heavy_ports,
    _call_orchestrator_with_slot_poll,
    _erase_slots,
    _force_erase_and_verify,
    _recover_heavy_ports_if_stuck,
    call_orchestrator_forced,
)
from seeding_infra import _wait_for_heavy_models_idle
from seeding_rewards import (
    compute_tool_value,
    score_delegation_chain,
    success_reward,
)

__all__ = [
    "ThreeWayResult",
    "_build_role_result",
    "_compute_3way_metadata",
    "_eval_single_config",
    "evaluate_question_3way",
]

logger = logging.getLogger("seed_specialist_routing")

_TAP_PATH = "/mnt/raid0/llm/tmp/inference_tap.log"
_REPL_TAP_PATH = "/mnt/raid0/llm/tmp/repl_tap.log"


def _log_delegation_diag(log_label: str, diag: dict[str, Any]) -> None:
    if not diag:
        return
    loops = diag.get("loops", "")
    break_reason = diag.get("break_reason", "")
    cap_reached = diag.get("cap_reached", False)
    repeated_edges = diag.get("repeated_edges", {}) or {}
    repeated_roles = diag.get("repeated_roles", {}) or {}
    infer_hops = diag.get("delegation_inference_hops")
    avg_prompt_ms = diag.get("avg_prompt_ms")
    avg_gen_ms = diag.get("avg_gen_ms")
    report_handles_count = diag.get("report_handles_count")
    report_handles = diag.get("report_handles", []) or []
    report_ids = [
        str(h.get("id", ""))
        for h in report_handles
        if isinstance(h, dict) and h.get("id")
    ]
    logger.info(
        "    [%s diag] loops=%s cap=%s break_reason=%s repeated_edges=%d repeated_roles=%d "
        "infer_hops=%s avg_prompt_ms=%s avg_gen_ms=%s report_handles=%s ids=%s",
        log_label,
        loops,
        cap_reached,
        break_reason or "none",
        len(repeated_edges),
        len(repeated_roles),
        infer_hops if infer_hops is not None else "n/a",
        avg_prompt_ms if avg_prompt_ms is not None else "n/a",
        avg_gen_ms if avg_gen_ms is not None else "n/a",
        report_handles_count if report_handles_count is not None else "n/a",
        ",".join(report_ids[:3]) if report_ids else "none",
    )


def _tap_size() -> int:
    """Current byte size of inference tap file (0 if missing)."""
    try:
        return os.path.getsize(_TAP_PATH)
    except OSError:
        return 0


def _repl_tap_size() -> int:
    """Current byte size of REPL tap file (0 if missing)."""
    try:
        return os.path.getsize(_REPL_TAP_PATH)
    except OSError:
        return 0


# ── ThreeWayResult dataclass ─────────────────────────────────────────


@dataclass
class ThreeWayResult:
    """Result from 3-way routing evaluation."""

    suite: str
    question_id: str
    prompt: str
    expected: str
    timestamp: str = ""
    role_results: dict[str, RoleResult] = field(default_factory=dict)
    rewards: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    rewards_injected: int = 0


# ── RoleResult builder (dedup helper) ────────────────────────────────


def _build_role_result(
    *,
    role: str,
    mode: str,
    resp: dict[str, Any],
    elapsed: float,
    expected: str,
    scoring_method: str,
    scoring_config: dict[str, Any],
) -> tuple[RoleResult, str]:
    """Build a RoleResult from orchestrator response, scoring the answer.

    Returns:
        (role_result, error_type) tuple.  error_type is one of
        "none", "infrastructure", or "task_failure".
    """
    answer = resp.get("answer", "")
    error = resp.get("error")
    error_type = _classify_error(error)

    if error_type == "infrastructure":
        passed = None
    elif error:
        passed = False
    else:
        passed = score_answer_deterministic(answer, expected, scoring_method, scoring_config)

    rr = RoleResult(
        role=role,
        mode=mode,
        answer=answer or "",
        passed=bool(passed) if passed is not None else False,
        elapsed_seconds=elapsed,
        error=error,
        error_type=error_type,
        tokens_generated=resp.get("tokens_generated", 0),
        tools_used=resp.get("tools_used", 0),
        tools_called=resp.get("tools_called", []),
        delegation_events=resp.get("delegation_events", []),
        delegation_diagnostics=resp.get("delegation_diagnostics", {}),
        tools_success=resp.get("tools_success"),
        delegation_success=resp.get("delegation_success"),
        routed_to=resp.get("routed_to", ""),
        role_history=resp.get("role_history", []),
        predicted_tps=resp.get("predicted_tps", 0.0),
        generation_ms=resp.get("generation_ms", 0.0),
        tokens_generated_estimate=resp.get("tokens_generated_estimate", 0),
        backend_task_id=resp.get("backend_task_id", 0),
        slot_progress_source=resp.get("slot_progress_source", ""),
        # Orchestrator intelligence diagnostics
        cost_dimensions=resp.get("cost_dimensions", {}),
        think_harder_attempted=resp.get("think_harder_attempted", False),
        think_harder_succeeded=resp.get("think_harder_succeeded"),
        cheap_first_attempted=resp.get("cheap_first_attempted", False),
        cheap_first_passed=resp.get("cheap_first_passed"),
        grammar_enforced=resp.get("grammar_enforced", False),
        parallel_tools_used=resp.get("parallel_tools_used", False),
        cache_affinity_bonus=resp.get("cache_affinity_bonus", 0.0),
        # SkillBank integration
        skills_retrieved=resp.get("skills_retrieved", 0),
        skill_ids=resp.get("skill_ids", []),
        # Timing fields
        cache_stats=resp.get("cache_stats"),
        prompt_eval_ms=resp.get("prompt_eval_ms", 0.0),
        http_overhead_ms=resp.get("http_overhead_ms", 0.0),
        # Context window management (C1/C3) and budget tracking (R1)
        budget_diagnostics=resp.get("budget_diagnostics", {}),
        session_persistence=resp.get("session_persistence", {}),
        tool_results_cleared=resp.get("tool_results_cleared", 0),
        compaction_triggered=resp.get("compaction_triggered", False),
        compaction_tokens_saved=resp.get("compaction_tokens_saved", 0),
        think_harder_expected_roi=resp.get("think_harder_expected_roi", 0.0),
    )
    return rr, error_type


# ── Core eval function ───────────────────────────────────────────────


def _eval_single_config(
    prompt: str,
    expected: str,
    scoring_method: str,
    scoring_config: dict,
    role: str,
    mode: str,
    url: str,
    timeout: int,
    client: "httpx.Client",
    allow_delegation: bool,
    image_path: str = "",
    log_label: str = "",
    format_fn=None,
) -> tuple[RoleResult, dict]:
    """Call the orchestrator, score, build RoleResult, and handle infra errors.

    Returns:
        (role_result, raw_response_dict) tuple.
    """
    port = ROLE_PORT.get(role, 0)
    did_recover_precheck = False

    # Proactive slot erase: clear ALL slots (including idle ones) to flush
    # stale KV cache from previous questions and prevent cross-question
    # context contamination (e.g. USACO text leaking into GPQA answers).
    if port > 0:
        _erase_slots(port, all_slots=True)

    if port in HEAVY_PORTS:
        idle_wait_cap = max(30, min(120, int(timeout // 2) if timeout else 120))
        _wait_for_heavy_models_idle(max_wait=idle_wait_cap)

        busy_ports = _busy_heavy_ports(timeout_s=2.0)
        if busy_ports:
            for bp in busy_ports:
                _erase_slots(bp)
            time.sleep(1.0)
            still_busy = _busy_heavy_ports(timeout_s=2.0)
            if still_busy:
                did_recover_precheck = _recover_heavy_ports_if_stuck(url, still_busy)

    logger.info(f"  → {log_label} ({role}:{mode}, timeout={timeout}s)...")
    tap_before = _tap_size()
    repl_tap_before = _repl_tap_size()
    resp, elapsed, slot_progress = _call_orchestrator_with_slot_poll(
        prompt=prompt,
        force_role=role,
        force_mode=mode,
        url=url,
        timeout=timeout,
        image_path=image_path,
        cache_prompt=False,
        client=client,
        allow_delegation=allow_delegation,
        log_label=log_label,
        poll_port=port,
        session_id=state.session_id,
        scoring_method=scoring_method,
    )

    # Merge slot progress into response
    max_decoded = int(slot_progress.get("max_decoded", 0) or 0)
    if max_decoded > 0:
        resp["tokens_generated_estimate"] = max_decoded
    resp["backend_task_id"] = int(slot_progress.get("task_id", 0) or 0)
    resp["slot_progress_source"] = str(slot_progress.get("source", "") or "")

    rr, error_type = _build_role_result(
        role=role, mode=mode, resp=resp, elapsed=elapsed,
        expected=expected, scoring_method=scoring_method,
        scoring_config=scoring_config,
    )
    _log_delegation_diag(log_label, rr.delegation_diagnostics)
    tap_after = _tap_size()
    rr.tap_offset_bytes = tap_before
    rr.tap_length_bytes = tap_after - tap_before
    repl_tap_after = _repl_tap_size()
    rr.repl_tap_offset_bytes = repl_tap_before
    rr.repl_tap_length_bytes = repl_tap_after - repl_tap_before

    # Retry logic for zero-token infra failures on heavy paths
    if error_type == "infrastructure" and resp.get("tokens_generated", 0) == 0:
        error_msg = (resp.get("error") or "").lower()
        is_timeout = "timed out" in error_msg or "timeout" in error_msg or "readtimeout" in error_msg
        target_port = ROLE_PORT.get(role, 0)
        if target_port:
            _force_erase_and_verify(target_port)
        if is_timeout:
            # Don't retry timeouts — server was processing but too slow;
            # retrying with the same budget just doubles elapsed time.
            logger.info(f"  [skip-retry] {log_label} timeout — not retrying")
        elif port in HEAVY_PORTS and not did_recover_precheck:
            busy_now = _busy_heavy_ports(timeout_s=2.0)
            if _recover_heavy_ports_if_stuck(url, busy_now):
                logger.info(f"  [retry] {log_label} retry after recovery")
                resp2, elapsed2, slot_progress2 = _call_orchestrator_with_slot_poll(
                    prompt=prompt,
                    force_role=role,
                    force_mode=mode,
                    url=url,
                    timeout=timeout,
                    image_path=image_path,
                    cache_prompt=False,
                    client=client,
                    allow_delegation=allow_delegation,
                    log_label=log_label,
                    poll_port=port,
                    session_id=state.session_id,
                    scoring_method=scoring_method,
                )
                max_decoded2 = int(slot_progress2.get("max_decoded", 0) or 0)
                max_decoded = max(max_decoded, max_decoded2)
                if max_decoded > 0:
                    resp2["tokens_generated_estimate"] = max_decoded
                resp2["backend_task_id"] = int(slot_progress2.get("task_id", 0) or 0) or int(
                    slot_progress.get("task_id", 0) or 0
                )
                resp2["slot_progress_source"] = (
                    str(slot_progress2.get("source", "") or "")
                    or str(slot_progress.get("source", "") or "")
                )

                rr, error_type = _build_role_result(
                    role=role, mode=mode, resp=resp2,
                    elapsed=elapsed + elapsed2,
                    expected=expected, scoring_method=scoring_method,
                    scoring_config=scoring_config,
                )
                _log_delegation_diag(f"{log_label}:retry", rr.delegation_diagnostics)
                tap_after = _tap_size()
                rr.tap_offset_bytes = tap_before
                rr.tap_length_bytes = tap_after - tap_before
                repl_tap_after = _repl_tap_size()
                rr.repl_tap_offset_bytes = repl_tap_before
                rr.repl_tap_length_bytes = repl_tap_after - repl_tap_before
                resp = resp2

    if format_fn is not None:
        final_error = rr.error
        final_passed = rr.passed
        for line in format_fn(log_label, final_passed, final_error, rr.elapsed_seconds, resp,
                              infra=(error_type == "infrastructure")):
            logger.info(line)

    return rr, resp


# ── 3-Way metadata computation ───────────────────────────────────────


def _compute_3way_metadata(
    role_results: dict[str, RoleResult],
    arch_results: dict[str, dict[str, Any]],
    prompt: str,
    suite: str,
    passed_direct: bool,
    passed_repl: bool,
    self_role: str,
    self_direct_mode: str,
    self_repl_mode: str,
    arch_mode: str,
) -> dict[str, Any]:
    """Compute metadata dict (tool value, cost metrics, architect eval)."""
    metadata = compute_tool_value(passed_direct, passed_repl)
    metadata["suite"] = suite
    metadata["cache_disabled"] = True

    # Determine best architect (prefer generation_ms over elapsed)
    best_arch = None
    for ar, res in arch_results.items():
        if res["passed"] is True:
            if best_arch is None:
                best_arch = ar
            else:
                cur_time = res.get("generation_ms") or (res["elapsed_seconds"] * 1000)
                best_time = arch_results[best_arch].get("generation_ms") or (
                    arch_results[best_arch]["elapsed_seconds"] * 1000
                )
                if cur_time < best_time:
                    best_arch = ar
    if best_arch is None:
        for ar, res in arch_results.items():
            if res["passed"] is not None:
                best_arch = ar
                break

    metadata["architect_eval"] = {
        "general": arch_results.get("architect_general"),
        "coding": arch_results.get("architect_coding"),
        "best": best_arch,
        "heuristic_would_pick": "architect_coding" if _is_coding_task(prompt) else "architect_general",
    }
    metadata["architect_role"] = best_arch or ""

    direct_key = f"{self_role}:{self_direct_mode}"
    repl_key = f"{self_role}:{self_repl_mode}"
    metadata["cost_metrics"] = {
        ACTION_SELF_DIRECT: {
            "elapsed_seconds": role_results[direct_key].elapsed_seconds,
            "tokens_generated": role_results[direct_key].tokens_generated,
            "tokens_generated_estimate": role_results[direct_key].tokens_generated_estimate,
            "predicted_tps": role_results[direct_key].predicted_tps,
            "generation_ms": role_results[direct_key].generation_ms,
            "backend_task_id": role_results[direct_key].backend_task_id,
            "slot_progress_source": role_results[direct_key].slot_progress_source,
        },
        ACTION_SELF_REPL: {
            "elapsed_seconds": role_results[repl_key].elapsed_seconds,
            "tokens_generated": role_results[repl_key].tokens_generated,
            "tokens_generated_estimate": role_results[repl_key].tokens_generated_estimate,
            "predicted_tps": role_results[repl_key].predicted_tps,
            "generation_ms": role_results[repl_key].generation_ms,
            "tools_used": role_results[repl_key].tools_used,
            "backend_task_id": role_results[repl_key].backend_task_id,
            "slot_progress_source": role_results[repl_key].slot_progress_source,
        },
    }
    if best_arch:
        arch_key = f"{best_arch}:{arch_mode}"
        metadata["cost_metrics"][ACTION_ARCHITECT] = {
            "elapsed_seconds": arch_results[best_arch]["elapsed_seconds"],
            "tokens_generated": role_results[arch_key].tokens_generated,
            "tokens_generated_estimate": role_results[arch_key].tokens_generated_estimate,
            "predicted_tps": role_results[arch_key].predicted_tps,
            "generation_ms": role_results[arch_key].generation_ms,
            "role_history": role_results[arch_key].role_history,
            "backend_task_id": role_results[arch_key].backend_task_id,
            "slot_progress_source": role_results[arch_key].slot_progress_source,
        }

    infra_flags = [
        rr.error_type == "infrastructure"
        for rr in role_results.values()
        if rr is not None
    ]
    metadata["all_infra"] = bool(infra_flags) and all(infra_flags)

    return metadata


# ── 3-Way evaluation ─────────────────────────────────────────────────


def evaluate_question_3way(
    prompt_info: dict,
    url: str,
    timeout: int,
    client: "httpx.Client",
    dry_run: bool = False,
    cooldown_s: float = 0.0,
) -> tuple[dict[str, RoleResult], dict[str, float], dict[str, Any]]:
    """Evaluate one question across the 3-way routing matrix.

    Test configurations:
    1. SELF:direct - Frontdoor without tools (direct mode)
    2. SELF:repl - Frontdoor with tools, delegation disabled
    3. ARCHITECT - Architect with full delegation freedom

    WORKER is scored indirectly via delegation chains.
    """
    prompt = prompt_info["prompt"]
    expected = prompt_info.get("expected", "")
    scoring_method = prompt_info.get("scoring_method", "exact_match")
    scoring_config = prompt_info.get("scoring_config", {})
    suite = prompt_info["suite"]
    image_path = prompt_info.get("image_path", "")
    is_vl = bool(image_path)

    role_results: dict[str, RoleResult] = {}

    # Flush KV cache from ALL model servers between questions to prevent
    # cross-question context contamination.  Without this, idle slots
    # retain stale KV state from the previous question (e.g. USACO text
    # leaking into GPQA organic chemistry answers).
    for _port in set(ROLE_PORT.values()):
        if _port > 0:
            _erase_slots(_port, all_slots=True)

    from eval_log_format import (
        format_self_direct, format_self_repl, format_architect_result,
        format_reward_skip, format_all_infra_skip,
    )

    # ── VL-aware role mapping ──
    if is_vl:
        self_role = "worker_vision"
        self_direct_mode = "direct"
        self_repl_mode = "repl"
        arch_role = "vision_escalation"
    else:
        self_role = "frontdoor"
        self_direct_mode = "direct"
        self_repl_mode = "repl"
        arch_role = None

    # ── Configuration 1: SELF:direct ──
    timeout_direct = _adaptive_timeout_s(
        role=self_role,
        mode=self_direct_mode,
        prompt=prompt,
        is_vl=is_vl,
        hard_timeout_s=timeout,
    )
    rr_direct, _ = _eval_single_config(
        prompt, expected, scoring_method, scoring_config,
        role=self_role, mode=self_direct_mode,
        url=url, timeout=timeout_direct, client=client,
        allow_delegation=False, image_path=image_path,
        log_label=ACTION_SELF_DIRECT, format_fn=format_self_direct,
    )
    role_results[f"{self_role}:{self_direct_mode}"] = rr_direct
    if cooldown_s > 0:
        time.sleep(cooldown_s)

    # ── Cleanup: ensure port is idle before next strategy ──
    if rr_direct.error_type == "infrastructure" or rr_direct.error:
        direct_port = ROLE_PORT.get(self_role, 0)
        if direct_port in HEAVY_PORTS:
            _force_erase_and_verify(direct_port)

    # ── Configuration 2: SELF:repl ──
    timeout_repl = _adaptive_timeout_s(
        role=self_role,
        mode=self_repl_mode,
        prompt=prompt,
        is_vl=is_vl,
        hard_timeout_s=timeout,
    )
    if rr_direct.error_type != "infrastructure":
        timeout_repl = _bump_timeout_from_observed(
            current_s=timeout_repl,
            observed_s=rr_direct.elapsed_seconds,
            factor=2.2 if not is_vl else 1.8,
            slack_s=30,
            hard_timeout_s=timeout,
            role_cap_s=300 if not is_vl else 260,
        )
    rr_repl, _ = _eval_single_config(
        prompt, expected, scoring_method, scoring_config,
        role=self_role, mode=self_repl_mode,
        url=url, timeout=timeout_repl, client=client,
        allow_delegation=False, image_path=image_path,
        log_label=ACTION_SELF_REPL, format_fn=format_self_repl,
    )
    role_results[f"{self_role}:{self_repl_mode}"] = rr_repl
    if cooldown_s > 0:
        time.sleep(cooldown_s)

    # ── Configuration 3: ARCHITECT (dual evaluation) ──
    if is_vl:
        arch_roles_to_eval = [arch_role]
    else:
        arch_roles_to_eval = ["architect_general", "architect_coding"]

    arch_results: dict[str, dict[str, Any]] = {}
    arch_mode = "direct" if is_vl else "delegated"

    for ar in arch_roles_to_eval:
        timeout_arch = _adaptive_timeout_s(
            role=ar,
            mode=arch_mode,
            prompt=prompt,
            is_vl=is_vl,
            hard_timeout_s=timeout,
        )
        observed_base = rr_direct.elapsed_seconds
        if rr_repl.error_type != "infrastructure" and rr_repl.elapsed_seconds > observed_base:
            observed_base = rr_repl.elapsed_seconds
        timeout_arch = _bump_timeout_from_observed(
            current_s=timeout_arch,
            observed_s=observed_base,
            factor=4.0 if not is_vl else 2.5,
            slack_s=60 if not is_vl else 40,
            hard_timeout_s=timeout,
            role_cap_s=max(540, int(timeout)) if not is_vl else 360,
        )
        rr_arch, resp_arch = _eval_single_config(
            prompt, expected, scoring_method, scoring_config,
            role=ar, mode=arch_mode,
            url=url, timeout=timeout_arch, client=client,
            allow_delegation=not is_vl, image_path=image_path,
            log_label=ACTION_ARCHITECT,
            format_fn=lambda label, passed, error, elapsed, resp, infra=False: format_architect_result(label, passed, error, elapsed, resp),
        )
        role_results[f"{ar}:{arch_mode}"] = rr_arch
        if cooldown_s > 0:
            time.sleep(cooldown_s)

        passed_arch = None if rr_arch.error_type == "infrastructure" else rr_arch.passed
        arch_results[ar] = {
            "passed": passed_arch,
            "elapsed_seconds": rr_arch.elapsed_seconds,
            "tokens_generated": rr_arch.tokens_generated,
            "predicted_tps": rr_arch.predicted_tps,
            "generation_ms": rr_arch.generation_ms,
            "tools_used": rr_arch.tools_used,
            "tools_called": rr_arch.tools_called,
            "role_history": rr_arch.role_history,
            "error": rr_arch.error,
            "error_type": rr_arch.error_type,
        }

    # ── Compute 3-way rewards (binary for faithful P(success)) ──
    passed_direct = rr_direct.passed
    passed_repl = rr_repl.passed
    error_type_direct = rr_direct.error_type
    error_type_repl = rr_repl.error_type

    rewards: dict[str, float] = {}

    if error_type_direct == "infrastructure":
        for line in format_reward_skip(ACTION_SELF_DIRECT):
            logger.info(line)
    else:
        rewards[ACTION_SELF_DIRECT] = success_reward(passed_direct)

    if error_type_repl == "infrastructure":
        for line in format_reward_skip(ACTION_SELF_REPL):
            logger.info(line)
    else:
        rewards[ACTION_SELF_REPL] = success_reward(passed_repl)

    valid_results = {k: v for k, v in arch_results.items() if v["passed"] is not None}
    if valid_results:
        passed_arch_any = any(v["passed"] for v in valid_results.values())
        rewards[ACTION_ARCHITECT] = success_reward(passed_arch_any)
    else:
        for line in format_all_infra_skip(ACTION_ARCHITECT):
            logger.info(line)

    worker_rewards = score_delegation_chain(role_results)
    rewards.update(worker_rewards)

    # ── Metadata ──
    metadata = _compute_3way_metadata(
        role_results, arch_results, prompt, suite,
        passed_direct, passed_repl,
        self_role, self_direct_mode, self_repl_mode, arch_mode,
    )

    for action, reward in sorted(rewards.items()):
        logger.info(f"    reward[{action}] = {reward:.1f}")

    return role_results, rewards, metadata
