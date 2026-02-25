"""Legacy comparative evaluation path (deprecated).

Kept for backwards compatibility. New seeding should use --3way mode
with binary rewards for faithful P(success) estimation.

Contains: _build_role_mode_combos, _deduplicate_roles, _modes_for_role,
evaluate_question, run_batch, print_batch_summary, print_stats.
"""

from __future__ import annotations

import logging
import time
import warnings
from datetime import datetime, timezone
from typing import Any

from seeding_types import (
    ARCHITECT_MODES,
    ARCHITECT_ROLES,
    ComparativeResult,
    HEAVY_PORTS,
    HealthCheckError,
    ROLE_PORT,
    RoleResult,
    VISION_MODES,
    VISION_ROLES,
    state,
)
from seeding_scoring import score_answer_deterministic
from seeding_orchestrator import _erase_slots, call_orchestrator_forced
from seeding_checkpoint import (
    _prompt_hash,
    append_checkpoint,
    load_checkpoint,
    load_seen_questions,
    record_seen,
)
from seeding_infra import (
    MAX_RECOVERY_ATTEMPTS,
    _attempt_recovery,
    _check_server_health,
    _wait_for_heavy_models_idle,
)
from seeding_rewards import (
    _inject_escalation_chains_http,
    _inject_rewards_http,
    compute_comparative_rewards,
    detect_escalation_chains,
)

__all__ = [
    "_build_role_mode_combos",
    "_deduplicate_roles",
    "_modes_for_role",
    "evaluate_question",
    "print_batch_summary",
    "print_stats",
    "run_batch",
]

logger = logging.getLogger("seed_specialist_routing")


# ── Combo building ───────────────────────────────────────────────────


def _build_role_mode_combos(
    roles: list[str],
    modes: list[str],
) -> list[tuple[str, str]]:
    """Build (role, mode) combinations with two invariants:

    1. MODE-FIRST: Cycle through modes before roles, so consecutive calls
       hit different backend servers (natural cooldown for each server).
    2. HEAVY SEPARATION: Heavy model combos (architects, ingest) are never
       adjacent. Light combos are interleaved between them so light work
       runs while heavy servers cool down.

    The idle-wait in evaluate_question() enforces that heavy models are
    actually idle before any request, but good ordering reduces idle-wait
    time by doing useful light work in the gaps.
    """
    all_modes = list(modes)
    for m in sorted(ARCHITECT_MODES):
        if m not in all_modes:
            all_modes.append(m)

    light: list[tuple[str, str]] = []
    heavy: list[tuple[str, str]] = []

    for mode in all_modes:
        for role in roles:
            port = ROLE_PORT.get(role, 0)
            is_heavy = port in HEAVY_PORTS
            if role in ARCHITECT_ROLES:
                if mode in ARCHITECT_MODES:
                    heavy.append((role, mode))
            elif role in VISION_ROLES:
                if mode in VISION_MODES.get(role, {"direct"}):
                    (heavy if is_heavy else light).append((role, mode))
            else:
                if mode in modes:
                    (heavy if is_heavy else light).append((role, mode))

    # Interleave: spread heavy combos evenly across the light sequence.
    if not heavy:
        return light
    if not light:
        return heavy

    result: list[tuple[str, str]] = []
    gap = max(1, len(light) // len(heavy))
    heavy_iter = iter(heavy)
    next_heavy = next(heavy_iter, None)

    for i, combo in enumerate(light):
        result.append(combo)
        if next_heavy is not None and (i + 1) % gap == 0:
            result.append(next_heavy)
            next_heavy = next(heavy_iter, None)

    if next_heavy is not None:
        result.append(next_heavy)
    for h in heavy_iter:
        result.append(h)

    return result


def _deduplicate_roles(
    roles: list[str],
    server_urls: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Remove roles that share the same backend URL."""
    from src.config import get_config as _get_cfg
    urls = server_urls or _get_cfg().server_urls.as_dict()
    seen: dict[str, str] = {}
    unique: list[str] = []
    aliases: dict[str, str] = {}

    for role in roles:
        url = urls.get(role, "")
        if url and url in seen:
            aliases[role] = seen[url]
        else:
            if url:
                seen[url] = role
            unique.append(role)

    return unique, aliases


def _modes_for_role(role: str, modes: list[str]) -> list[str]:
    """Return the effective mode list for a role."""
    if role in ARCHITECT_ROLES:
        return sorted(ARCHITECT_MODES)
    if role in VISION_ROLES:
        return sorted(VISION_MODES.get(role, {"direct"}))
    return list(modes)


# ── Main evaluation loop ──────────────────────────────────────────────


def evaluate_question(
    prompt_info: dict,
    combos: list[tuple[str, str]],
    alias_map: dict[str, str],
    modes: list[str],
    url: str,
    timeout: int,
    client: "httpx.Client",
    skip_cache: bool = False,
    cooldown: float = 0.0,
    dry_run: bool = False,
    escalation_chains: bool = False,
) -> ComparativeResult | None:
    """Evaluate one question across all role x mode combos.

    Returns ComparativeResult or None if shutdown requested.
    """
    suite = prompt_info["suite"]
    qid = prompt_info["id"]
    prompt = prompt_info["prompt"]
    expected = prompt_info.get("expected", "")
    scoring_method = prompt_info.get("scoring_method", "exact_match")
    scoring_config = prompt_info.get("scoring_config", {})
    image_path = prompt_info.get("image_path", "")
    dataset_source = prompt_info.get("dataset_source", "yaml")

    is_vl = bool(image_path)
    if is_vl:
        active_combos = [
            (r, m) for r, m in combos
            if r in VISION_ROLES or r == "frontdoor"
        ]
    else:
        active_combos = [
            (r, m) for r, m in combos
            if r not in VISION_ROLES
        ]

    role_results: dict[str, RoleResult] = {}
    cache_prompt_val = False if skip_cache else None

    SLOW_ROLES = {"architect_general", "architect_coding"}
    SLOW_ROLE_TIMEOUT = max(timeout, 300)

    for combo_idx, (role, mode) in enumerate(active_combos):
        if state.shutdown:
            return None

        target_port = ROLE_PORT.get(role, 0)
        if target_port in HEAVY_PORTS:
            _wait_for_heavy_models_idle()

        key = f"{role}:{mode}"
        if target_port in HEAVY_PORTS:
            logger.info(f"  → {key} (heavy model, expect 30-120s)...")
        role_timeout = SLOW_ROLE_TIMEOUT if role in SLOW_ROLES else timeout
        q_start = time.perf_counter()
        response = call_orchestrator_forced(
            prompt, role, mode, url, role_timeout,
            image_path=image_path, cache_prompt=cache_prompt_val,
            client=client,
        )
        q_elapsed = time.perf_counter() - q_start

        if cooldown > 0 and combo_idx < len(active_combos) - 1:
            time.sleep(cooldown)

        answer = response.get("answer", "")
        error = response.get("error")
        tokens_generated = response.get("tokens_generated", 0)
        tools_used = response.get("tools_used", 0)
        tools_called = response.get("tools_called", [])
        routed_to = response.get("routed_to", "")
        role_history = response.get("role_history", [])
        routing_strategy = response.get("routing_strategy", "")
        turns = response.get("turns", 0)
        tokens_used = response.get("tokens_used", 0)
        formalization_applied = response.get("formalization_applied", False)
        cache_stats = response.get("cache_stats")
        predicted_tps = response.get("predicted_tps", 0.0)
        generation_ms = response.get("generation_ms", 0.0)
        prompt_eval_ms = response.get("prompt_eval_ms", 0.0)
        http_overhead_ms = response.get("http_overhead_ms", 0.0)

        if error:
            passed = False
            if target_port in HEAVY_PORTS and tokens_generated == 0:
                _erase_slots(target_port)
        else:
            passed = score_answer_deterministic(answer, expected, scoring_method, scoring_config)

        role_results[key] = RoleResult(
            role=role,
            mode=mode,
            answer=answer or "",
            passed=passed,
            elapsed_seconds=q_elapsed,
            error=error,
            tokens_generated=tokens_generated,
            tools_used=tools_used,
            tools_called=tools_called,
            routed_to=routed_to,
            role_history=role_history,
            routing_strategy=routing_strategy,
            turns=turns,
            tokens_used=tokens_used,
            formalization_applied=formalization_applied,
            cache_stats=cache_stats,
            predicted_tps=predicted_tps,
            generation_ms=generation_ms,
            prompt_eval_ms=prompt_eval_ms,
            http_overhead_ms=http_overhead_ms,
        )

        status = "PASS" if passed else ("ERROR" if error else "FAIL")
        display_tps = predicted_tps if predicted_tps > 0 else 0

        parts = [f"  {key:30s} → {status} ({q_elapsed:.1f}s"]
        if display_tps > 0:
            parts.append(f", {display_tps:.1f} t/s")
        parts.append(f", {tokens_generated} tok)")
        logger.info("".join(parts))

        indent = "  " + " " * 30 + "   "

        if role_history and len(role_history) > 1:
            logger.info(f"{indent}chain: {' → '.join(role_history)}")

        if tools_used > 0:
            if tools_called:
                deduped = []
                for t in tools_called:
                    if not deduped or deduped[-1] != t:
                        deduped.append(t)
                tool_str = ", ".join(deduped)
            else:
                tool_str = "?"
            logger.info(f"{indent}tools({tools_used}): {tool_str}")

        timing_parts = []
        if generation_ms > 0:
            timing_parts.append(f"gen={generation_ms/1000:.1f}s")
        if prompt_eval_ms > 0:
            timing_parts.append(f"prompt={prompt_eval_ms/1000:.1f}s")
        if formalization_applied:
            timing_parts.append("formalized")
        if timing_parts:
            logger.info(f"{indent}{', '.join(timing_parts)}")

    # Compute comparative rewards (baseline is frontdoor:direct)
    rewards = compute_comparative_rewards(role_results, baseline_key="frontdoor:direct")

    # Clone rewards and results to aliased (deduplicated) roles
    for alias, canonical in alias_map.items():
        for mode in _modes_for_role(alias, modes):
            canonical_key = f"{canonical}:{mode}"
            alias_key = f"{alias}:{mode}"
            if canonical_key in rewards:
                rewards[alias_key] = rewards[canonical_key]
            if canonical_key in role_results:
                role_results[alias_key] = role_results[canonical_key]

    # Inject rewards immediately (per-question, not batched)
    rewards_injected = 0
    if not dry_run:
        comp_for_inject = ComparativeResult(
            suite=suite, question_id=qid, prompt=prompt[:200],
            expected=expected[:200], rewards=rewards,
        )
        rewards_injected = _inject_rewards_http(comp_for_inject, url, client)

    # Escalation chains: detect cheap-fail -> expensive-pass patterns
    escalation_data: list[dict[str, Any]] = []
    if escalation_chains and not dry_run:
        escalation_data = detect_escalation_chains(role_results)
        if escalation_data:
            comp_for_esc = ComparativeResult(
                suite=suite, question_id=qid, prompt=prompt[:200],
                expected=expected[:200], rewards=rewards,
            )
            esc_injected = _inject_escalation_chains_http(
                comp_for_esc, escalation_data, url, client,
            )
            rewards_injected += esc_injected
            for chain in escalation_data:
                logger.info(
                    f"    escalation: {chain['from_role']}:{chain['from_mode']} → "
                    f"{chain['to_role']}:{chain['to_mode']} "
                    f"reward={chain['reward']:+.2f}"
                )

    # Log rewards
    for key, reward in sorted(rewards.items()):
        alias_tag = ""
        role_part = key.split(":")[0]
        if role_part in alias_map:
            alias_tag = f" (={alias_map[role_part]})"
        logger.info(f"    reward[{key}] = {reward:+.2f}{alias_tag}")

    return ComparativeResult(
        suite=suite,
        question_id=qid,
        prompt=prompt[:200],
        expected=expected[:200],
        dataset_source=dataset_source,
        prompt_hash=_prompt_hash(prompt),
        timestamp=datetime.now(timezone.utc).isoformat(),
        role_results=role_results,
        rewards=rewards,
        rewards_injected=rewards_injected,
    )


def run_batch(
    suites: list[str],
    roles: list[str],
    modes: list[str],
    sample_per_suite: int,
    seed: int,
    url: str,
    timeout: int,
    session_id: str,
    dry_run: bool = False,
    skip_cache: bool = False,
    cooldown: float = 0.0,
    no_dedup: bool = False,
    escalation_chains: bool = False,
    use_pool: bool = True,
    debugger: Any = None,
) -> list[ComparativeResult]:
    """Run one evaluation batch: sample, evaluate per-question, checkpoint.

    DEPRECATED: Use run_batch_3way() with --3way flag for new seeding.
    This legacy mode uses cost-weighted comparative rewards which conflate
    P(success) with cost. The 3-way mode uses binary rewards for faithful
    probability estimation.
    """
    # Import here to avoid circular dependency at module level
    from seed_specialist_routing import sample_unseen_questions

    warnings.warn(
        "Legacy comparative seeding is deprecated. Use --3way for binary rewards "
        "and faithful P(success) estimation.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Deduplicate roles
    alias_map: dict[str, str] = {}
    if not no_dedup:
        unique_roles, alias_map = _deduplicate_roles(roles)
        if alias_map:
            for alias, canonical in sorted(alias_map.items()):
                from src.config import get_config as _get_cfg2
                canon_url = _get_cfg2().server_urls.as_dict().get(canonical, "?")
                logger.info(f"Dedup: {alias} → {canonical} (same backend {canon_url})")
        roles_to_test = unique_roles
    else:
        roles_to_test = list(roles)

    combos = _build_role_mode_combos(roles_to_test, modes)
    combo_keys = [f"{r}:{m}" for r, m in combos]

    if not _check_server_health(url):
        raise HealthCheckError(f"API unreachable: {url}")

    completed = load_checkpoint(session_id)
    completed_ids = {r.question_id for r in completed}
    seen = load_seen_questions()
    logger.info(f"Checkpoint: {len(completed)} completed, {len(seen)} total seen")

    questions = sample_unseen_questions(
        suites, sample_per_suite, seen, seed,
        use_pool=use_pool, allow_reseen=debugger is not None,
    )
    questions = [q for q in questions if q["id"] not in completed_ids]

    if not questions:
        logger.info("No unseen questions available. Try a different seed or suite.")
        return completed

    vl_count = sum(1 for p in questions if p.get("image_path"))
    text_count = len(questions) - vl_count

    logger.info(f"\n{'='*60}")
    logger.info(f"Session: {session_id}")
    logger.info(f"Batch: {len(questions)} questions ({text_count} text, {vl_count} VL)")
    logger.info(f"Combos: {len(combos)} ({', '.join(combo_keys)})")
    heavy_count = sum(1 for r, m in combos if ROLE_PORT.get(r, 0) in HEAVY_PORTS)
    if heavy_count:
        logger.info(f"  Heavy combos per question: {heavy_count} (30-120s each)")
    logger.info(f"Seed: {seed}  Rewards: {'off' if dry_run else 'on'}")
    logger.info(f"{'='*60}\n")

    SLOW_ROLE_TIMEOUT = max(timeout, 300)
    import httpx as _httpx
    _client = _httpx.Client(timeout=SLOW_ROLE_TIMEOUT)

    new_results: list[ComparativeResult] = []
    consecutive_zero_success = 0

    try:
        for i, prompt_info in enumerate(questions):
            if state.shutdown:
                logger.info(f"\n[Stopped after {i} questions]")
                break

            qid = prompt_info["id"]
            suite = prompt_info["suite"]
            is_vl = bool(prompt_info.get("image_path"))
            logger.info(f"[{i+1}/{len(questions)}] {suite}/{qid} ({'VL' if is_vl else 'text'})")

            result = evaluate_question(
                prompt_info, combos, alias_map, modes,
                url, timeout, _client,
                skip_cache=skip_cache, cooldown=cooldown, dry_run=dry_run,
                escalation_chains=escalation_chains,
            )

            if result is None:
                break

            append_checkpoint(session_id, result)
            if result.rewards_injected > 0:
                record_seen(result.question_id, result.suite, session_id)
            new_results.append(result)

            any_success = any(rr.error is None for rr in result.role_results.values())
            if any_success:
                consecutive_zero_success = 0
            else:
                consecutive_zero_success += 1
                if consecutive_zero_success >= 3:
                    logger.error(
                        f"Aborting: {consecutive_zero_success} consecutive questions "
                        f"with zero successful combos — server appears dead"
                    )
                    break
    finally:
        _client.close()

    all_results = completed + new_results
    return all_results


# ── Stats / summary ───────────────────────────────────────────────────


def print_batch_summary(
    results: list[ComparativeResult],
    roles: list[str],
    modes: list[str],
    alias_map: dict[str, str] | None = None,
) -> None:
    """Print summary of results."""
    alias_map = alias_map or {}
    combos = _build_role_mode_combos(roles, modes)
    combo_keys = [f"{r}:{m}" for r, m in combos]

    key_stats: dict[str, dict[str, Any]] = {
        k: {"pass": 0, "fail": 0, "error": 0,
            "total_tokens": 0, "total_elapsed": 0.0,
            "samples": 0, "predicted_tps_sum": 0.0, "predicted_tps_count": 0,
            "total_reward": 0.0}
        for k in combo_keys
    }

    for comp in results:
        for key, rr in comp.role_results.items():
            if key not in key_stats:
                continue
            key_stats[key]["samples"] += 1
            key_stats[key]["total_tokens"] += rr.tokens_generated
            key_stats[key]["total_elapsed"] += rr.elapsed_seconds
            if rr.predicted_tps > 0:
                key_stats[key]["predicted_tps_sum"] += rr.predicted_tps
                key_stats[key]["predicted_tps_count"] += 1
            if rr.error:
                key_stats[key]["error"] += 1
            elif rr.passed:
                key_stats[key]["pass"] += 1
            else:
                key_stats[key]["fail"] += 1
        for key, reward in comp.rewards.items():
            if key in key_stats:
                key_stats[key]["total_reward"] += reward

    print(f"\n{'='*100}")
    print("COMPARATIVE EVALUATION SUMMARY")
    print(f"{'='*100}")
    print(f"Questions: {len(results)}")
    if alias_map:
        dedup_strs = [f"{a} → {c}" for a, c in sorted(alias_map.items())]
        print(f"Deduplicated: {', '.join(dedup_strs)}")

    print(f"\n{'Role:Mode':30s} {'Pass':>5s} {'Fail':>5s} {'Err':>4s} {'Acc%':>6s} {'Avg t/s':>8s} {'Reward':>8s}")
    print("-" * 75)
    for key in combo_keys:
        s = key_stats[key]
        total = s["pass"] + s["fail"]
        acc = s["pass"] / total * 100 if total > 0 else 0
        if s["predicted_tps_count"] > 0:
            avg_tps = s["predicted_tps_sum"] / s["predicted_tps_count"]
        else:
            avg_tps = s["total_tokens"] / s["total_elapsed"] if s["total_elapsed"] > 0 else 0
        role_part = key.split(":")[0]
        alias_tag = f" (={alias_map[role_part]})" if role_part in alias_map else ""
        print(
            f"{key:30s} {s['pass']:5d} {s['fail']:5d} {s['error']:4d} "
            f"{acc:5.1f}% {avg_tps:7.1f} {s['total_reward']:+7.1f}"
            f"{alias_tag}"
        )

    total_injected = sum(r.rewards_injected for r in results)
    print(f"\nRewards injected: {total_injected}")


def print_stats():
    """Aggregate stats across all seeding sessions."""
    from seeding_types import EVAL_DIR

    if not EVAL_DIR.exists():
        print("No evaluation data found.")
        return

    sessions: dict[str, list[ComparativeResult]] = {}
    for path in sorted(EVAL_DIR.glob("seeding_*.jsonl")):
        sid = path.stem
        results = load_checkpoint(sid)
        if results:
            sessions[sid] = results

    if not sessions:
        print("No seeding sessions found.")
        return

    print(f"\n{'='*60}")
    print("ALL SEEDING SESSIONS")
    print(f"{'='*60}")

    total_questions = 0
    all_combo_stats: dict[str, dict[str, int]] = {}

    for sid, results in sessions.items():
        total_questions += len(results)
        ts = results[0].timestamp[:10] if results and results[0].timestamp else "?"
        print(f"  {sid:45s} {len(results):4d} questions  {ts}")

        for comp in results:
            for key, rr in comp.role_results.items():
                if key not in all_combo_stats:
                    all_combo_stats[key] = {"pass": 0, "fail": 0, "error": 0, "total": 0}
                all_combo_stats[key]["total"] += 1
                if rr.error:
                    all_combo_stats[key]["error"] += 1
                elif rr.passed:
                    all_combo_stats[key]["pass"] += 1
                else:
                    all_combo_stats[key]["fail"] += 1

    print(f"\nTotal questions: {total_questions}")
    print(f"Sessions: {len(sessions)}")

    seen = load_seen_questions()
    print(f"Unique questions seen: {len(seen)}")

    if all_combo_stats:
        print(f"\nAggregate accuracy by role x mode:")
        print(f"  {'Role:Mode':30s} {'Pass':>5s} {'Fail':>5s} {'Err':>4s} {'Acc%':>6s} {'N':>5s} {'>=3?':>4s}")
        print("  " + "-" * 60)
        for key in sorted(all_combo_stats.keys()):
            s = all_combo_stats[key]
            total = s["pass"] + s["fail"]
            acc = s["pass"] / total * 100 if total > 0 else 0
            confident = "YES" if s["total"] >= 3 else "no"
            print(
                f"  {key:30s} {s['pass']:5d} {s['fail']:5d} {s['error']:4d} "
                f"{acc:5.1f}% {s['total']:5d} {confident:>4s}"
            )

    covered = sum(1 for s in all_combo_stats.values() if s["total"] >= 3)
    total_combos = len(all_combo_stats)
    print(f"\nMemRL coverage: {covered}/{total_combos} combos have >=3 observations")
