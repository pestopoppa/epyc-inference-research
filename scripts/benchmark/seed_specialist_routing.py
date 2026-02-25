#!/usr/bin/env python3
"""MemRL Episodic Seeding for 3-Way Routing.

THE canonical evaluation script for training frontdoor probability estimation.
Runs each question through multiple configurations, scores deterministically,
and injects rewards so MemRL learns optimal routing decisions.

## Preferred Mode: 3-Way Routing (--3way)

Uses binary rewards for faithful P(success|action) estimation:
  - SELF:direct  → Frontdoor without tools
  - SELF:repl    → Frontdoor with tools, delegation disabled
  - ARCHITECT    → Architect with full delegation freedom
  - WORKER       → Scored via delegation chain attribution

Binary reward: 1.0 for pass, 0.0 for fail. Cost stored in metadata for Optuna.
TD learning with α=0.1 converges Q-values to empirical success rates.

Usage (3-way mode - RECOMMENDED):
    # THE command. Binary rewards for faithful probability estimation.
    python scripts/benchmark/seed_specialist_routing.py \\
      --3way --suites all --sample-size 10

    # Dry run (no reward injection)
    python scripts/benchmark/seed_specialist_routing.py \\
      --3way --dry-run --suites thinking --sample-size 3

## Legacy Mode: Comparative Rewards

Uses cost-weighted comparative rewards (deprecated for new seeding):
  specialist correct & frontdoor wrong → +1.0 (specialist clearly better)
  specialist wrong & frontdoor right   → -0.5 (specialist worse)
  both correct                         → 0.5 - λ*max(0, cost_ratio-1) (cost-aware)
  both wrong                           → -0.3 (neither helps)

Usage (legacy mode):
    # Continuous comparative seeding (legacy)
    python scripts/benchmark/seed_specialist_routing.py \\
      --continuous --suites all --sample-size 10 --cooldown 2.0 --preflight

    # Quick stats from all sessions
    python scripts/benchmark/seed_specialist_routing.py --stats
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Bootstrap: needed before seeding_types can be imported. Same value as seeding_types.PROJECT_ROOT.
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "benchmark"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Silence noisy HTTP client logs (httpx, httpcore, urllib3)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


# ── Re-exports from extracted modules (test backwards compatibility) ──
# Tests import these symbols from this file; keep them accessible here.

from seeding_types import (  # noqa: E402, F401
    ARCHITECT_MODES,
    ARCHITECT_ROLES,
    ComparativeResult,
    DEBUG_PROMPTS_DIR,
    DEFAULT_MODES,
    DEFAULT_ORCHESTRATOR_URL,
    DEFAULT_ROLES,
    DEFAULT_SUITES,
    DEFAULT_TIMEOUT,
    ESCALATION_REWARD,
    EVAL_DIR,
    HEAVY_PORTS,
    HealthCheckError,
    MODEL_PORTS,
    ROLE_COST_TIER,
    ROLE_PORT,
    RoleResult,
    SEEN_FILE,
    VISION_MODES,
    VISION_ROLES,
    state,
)
from seeding_types import (  # noqa: E402 (additional imports for 3-way routing)
    ACTION_SELF_DIRECT,
    ACTION_SELF_REPL,
    ACTION_ARCHITECT,
    ACTION_WORKER,
    THREE_WAY_ACTIONS,
)
from seeding_rewards import (  # noqa: E402, F401
    DEFAULT_BASELINE_TPS,
    _inject_escalation_chains_http,
    _inject_rewards_http,
    compute_comparative_rewards,
    detect_escalation_chains,
    success_reward,
    compute_3way_rewards,
    score_delegation_chain,
    compute_tool_value,
)
from seeding_infra import (  # noqa: E402, F401
    MAX_RECOVERY_ATTEMPTS,
    _attempt_recovery,
    _check_server_health,
    _wait_for_heavy_models_idle,
    run_preflight,
)
from seeding_checkpoint import (  # noqa: E402, F401
    _atomic_append,
    _prompt_hash,
    append_checkpoint,
    _checkpoint_3way,
    checkpoint_result,
    load_checkpoint,
    load_seen_questions,
    record_seen,
)
from seeding_scoring import (  # noqa: E402, F401
    INFRA_PATTERNS,
    _adaptive_timeout_s,
    _bump_timeout_from_observed,
    _classify_error,
    _is_coding_task,
    score_answer_deterministic,
)
from seeding_orchestrator import (  # noqa: E402, F401
    _SLOT_ERASE_CAPABILITY,
    _busy_heavy_ports,
    _call_orchestrator_with_slot_poll,
    _erase_slots,
    _force_erase_and_verify,
    _normalize_tool_telemetry,
    _read_slot_progress,
    _recover_heavy_ports_if_stuck,
    call_orchestrator_forced,
)
from seeding_injection import (  # noqa: E402, F401
    EMBEDDER_PORTS,
    _get_reward_executor,
    _inject_3way_rewards_http,
    _inject_single_reward,
    _precompute_embedding,
)
from seeding_eval import (  # noqa: E402, F401
    ThreeWayResult,
    _build_role_result,
    _compute_3way_metadata,
    _eval_single_config,
    evaluate_question_3way,
)
from seeding_legacy import (  # noqa: E402, F401
    _build_role_mode_combos,
    _deduplicate_roles,
    _modes_for_role,
    evaluate_question,
    print_batch_summary,
    print_stats,
    run_batch,
)


# ── Signal handlers ──────────────────────────────────────────────────


def _handle_sigint(sig, frame):
    if state.shutdown:
        state.close_poll_client()
        sys.exit(1)
    state.shutdown = True
    logger.info("[SIGINT] Finishing current question, then stopping...")


def _handle_sigterm(sig, frame):
    state.shutdown = True
    state.close_poll_client()


signal.signal(signal.SIGINT, _handle_sigint)
signal.signal(signal.SIGTERM, _handle_sigterm)


# ── Question sampling ─────────────────────────────────────────────────


def _load_from_dataset_adapter(
    suite_name: str, sample_count: int, seed: int,
) -> list[dict]:
    """Sample questions from HF dataset adapters."""
    try:
        from dataset_adapters import get_adapter, ADAPTER_SUITES
    except ImportError:
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from dataset_adapters import get_adapter, ADAPTER_SUITES
        except ImportError:
            return []

    if suite_name not in ADAPTER_SUITES:
        return []

    adapter = get_adapter(suite_name)
    if adapter is None:
        return []

    prompts = adapter.sample(n=sample_count, seed=seed)
    if prompts:
        logger.info(f"  [{suite_name}] Sampled {len(prompts)} from "
                     f"{adapter.total_available} HF dataset questions (seed={seed})")
    return prompts


def _load_from_yaml(
    suite_name: str, sample_count: int, seed: int,
) -> list[dict]:
    """Fall back to static YAML debug prompts."""
    try:
        import yaml
    except ImportError:
        return []

    yaml_path = DEBUG_PROMPTS_DIR / f"{suite_name}.yaml"
    if not yaml_path.exists():
        return []

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    questions = data.get("questions", [])
    if not questions:
        return []

    rng = random.Random(seed)
    n = min(sample_count, len(questions))
    sampled = rng.sample(questions, n)
    logger.info(f"  [{suite_name}] Sampled {n}/{len(questions)} from YAML (seed={seed})")

    result = []
    for q in sampled:
        result.append({
            "id": q["id"],
            "suite": suite_name,
            "prompt": q["prompt"].strip(),
            "context": "",
            "expected": q.get("expected", ""),
            "image_path": q.get("image_path", ""),
            "tier": q.get("tier", 1),
            "scoring_method": q.get("scoring_method", "exact_match"),
            "scoring_config": q.get("scoring_config", {}),
            "dataset_source": "yaml",
        })
    return result


def sample_unseen_questions(
    suites: list[str],
    sample_per_suite: int,
    seen: set[str],
    seed: int,
    *,
    use_pool: bool = True,
    allow_reseen: bool = False,
) -> list[dict]:
    """Sample questions not in the seen set, interleaved across suites.

    If use_pool=True (default), tries the pre-extracted question pool first
    (~100ms). Falls back to HF dataset adapters, then YAML.

    If ``allow_reseen`` (debug mode), backfills with seen questions when a
    suite is exhausted.  Normal mode skips exhausted suites.

    Returns questions interleaved by suite (round-robin) so the orchestrator
    sees diverse question types early rather than processing one suite at a time.
    """
    suite_names = DEFAULT_SUITES if suites == ["all"] else suites

    # Try the pre-extracted pool first
    if use_pool:
        try:
            from question_pool import POOL_FILE, build_pool, load_pool, sample_from_pool

            if not POOL_FILE.exists():
                logger.info("Question pool not found — building automatically (one-time)...")
                build_pool()

            pool = load_pool()
            if pool:
                result = sample_from_pool(
                    pool, suite_names, sample_per_suite, seed, seen,
                    allow_reseen=allow_reseen,
                )
                if result:
                    logger.info(f"Sampled {len(result)} questions from pool (fast path)")
                    return result
                logger.info("Pool returned no results — falling back to adapters")
        except Exception as e:
            logger.warning(f"Pool loading failed ({e}) — falling back to adapters")

    per_suite: list[list[dict]] = []

    for suite_name in suite_names:
        oversample = sample_per_suite * 20

        prompts = _load_from_dataset_adapter(suite_name, oversample, seed)
        if not prompts:
            prompts = _load_from_yaml(suite_name, oversample, seed)

        fresh = [p for p in prompts if p["id"] not in seen]
        if len(fresh) < len(prompts):
            filtered = len(prompts) - len(fresh)
            logger.info(f"  [{suite_name}] Filtered {filtered} previously seen questions")

        per_suite.append(fresh[:sample_per_suite])

    # Interleave: round-robin across suites
    all_prompts: list[dict] = []
    max_len = max((len(s) for s in per_suite), default=0)
    for i in range(max_len):
        for suite_questions in per_suite:
            if i < len(suite_questions):
                all_prompts.append(suite_questions[i])

    return all_prompts


# ── 3-Way Batch Runner ───────────────────────────────────────────────


def run_batch_3way(
    suites: list[str],
    sample_per_suite: int,
    seed: int,
    url: str,
    timeout: int,
    session_id: str,
    dry_run: bool = False,
    cooldown: float = 0.0,
    on_progress: "Callable[[int, int, str, str], None] | None" = None,
    use_pool: bool = True,
    debugger: "ClaudeDebugger | None" = None,
    outcome_tracker: Any = None,
) -> list[ThreeWayResult]:
    """Run one 3-way evaluation batch.

    Each question is tested through:
    1. SELF:direct (frontdoor, no tools)
    2. SELF:repl (frontdoor, tools, no delegation)
    3. ARCHITECT (architect with full delegation)

    Binary rewards injected for faithful probability estimation.

    Args:
        cooldown: Seconds to sleep between SELF:direct/SELF:repl/ARCHITECT calls.
        on_progress: Optional callback ``(idx, total, suite, qid)`` called
            at the start of each question.  Used by the TUI to update the
            status bar.
        debugger: Optional ClaudeDebugger for pipeline monitoring.
    """
    import httpx as _httpx

    # Health check
    if not _check_server_health(url):
        raise HealthCheckError(f"API unreachable: {url}")

    # Load seen questions
    seen = load_seen_questions()
    logger.info(f"Previously seen questions: {len(seen)}")

    # Sample unseen questions (debug mode backfills with seen when exhausted)
    questions = sample_unseen_questions(
        suites, sample_per_suite, seen, seed,
        use_pool=use_pool, allow_reseen=debugger is not None,
    )
    if not questions:
        logger.info("No unseen questions available.")
        return []

    logger.info(f"\n{'='*60}")
    logger.info(f"3-Way Routing Evaluation: {len(questions)} questions")
    logger.info(f"Session: {session_id}")
    logger.info(f"Actions: {', '.join(THREE_WAY_ACTIONS)}")
    logger.info(f"Seed: {seed}  Rewards: {'off' if dry_run else 'on'}")
    logger.info(f"Cooldown: {cooldown:.1f}s between strategy calls")
    logger.info("Cache control: cache_prompt=False for all 3-way eval calls (fair timing)")
    logger.info(f"{'='*60}\n")

    _client = _httpx.Client(timeout=max(timeout, 300))
    results: list[ThreeWayResult] = []

    # Lookup tables for post-fix retry regression suite
    prompt_info_by_qid: dict[tuple[str, str], dict] = {
        (pi["suite"], pi["id"]): pi for pi in questions
    }
    passing_by_suite: dict[str, list[dict]] = {}  # suite → passing prompt_infos

    try:
        for i, prompt_info in enumerate(questions):
            if state.shutdown:
                logger.info(f"\n[Stopped after {i} questions]")
                break

            qid = prompt_info["id"]
            suite = prompt_info["suite"]
            logger.info(f"[{i+1}/{len(questions)}] {suite}/{qid}")

            if on_progress is not None:
                on_progress(i + 1, len(questions), suite, qid, prompt_info.get("prompt", ""))

            # Run 3-way evaluation
            role_results, rewards, metadata = evaluate_question_3way(
                prompt_info, url, timeout, _client, dry_run, cooldown_s=cooldown,
            )
            if metadata.get("all_infra"):
                logger.warning("  All roles failed due to infra. Attempting recovery...")
                recovered = _attempt_recovery(url)
                if recovered:
                    logger.info("  Recovery successful — continuing")
                else:
                    logger.warning("  Recovery failed — sleeping 30s before continuing")
                    time.sleep(30)

            # Inject rewards
            rewards_injected = 0
            if not dry_run:
                rewards_injected = _inject_3way_rewards_http(
                    prompt_info["prompt"][:200],
                    suite,
                    qid,
                    rewards,
                    metadata,
                    url,
                    _client,
                )
                logger.info(f"  Injected {rewards_injected} rewards")

            # Record skill outcomes for evolution tracking
            if outcome_tracker is not None:
                for config_key, rr in role_results.items():
                    if rr.skill_ids:
                        for skill_id in rr.skill_ids:
                            outcome_tracker.record_outcome(
                                skill_id, f"{suite}/{qid}", success=rr.passed,
                            )

            result = ThreeWayResult(
                suite=suite,
                question_id=qid,
                prompt=prompt_info["prompt"][:200],
                expected=prompt_info.get("expected", "")[:200],
                timestamp=datetime.now(timezone.utc).isoformat(),
                role_results=role_results,
                rewards=rewards,
                metadata=metadata,
                rewards_injected=rewards_injected,
            )
            results.append(result)

            # Checkpoint immediately
            checkpoint_result(session_id, result)

            # Always mark seen after checkpoint — re-evaluating wastes compute
            # and injects duplicate/conflicting rewards regardless of injection
            record_seen(qid, suite, session_id)

            # Track passing questions for regression testing
            if any(rr.passed for rr in role_results.values()):
                passing_by_suite.setdefault(suite, []).append(prompt_info)

            # Pipeline debugger: build diagnostics for each role result
            if debugger is not None:
                try:
                    from src.pipeline_monitor.diagnostic import build_diagnostic, append_diagnostic

                    for config_key, rr in role_results.items():
                        diag = build_diagnostic(
                            question_id=f"{suite}/{qid}",
                            suite=suite,
                            config=config_key,
                            role=rr.role,
                            mode=rr.mode,
                            passed=rr.passed,
                            answer=rr.answer,
                            expected=prompt_info.get("expected", ""),
                            scoring_method=prompt_info.get("scoring_method", "exact_match"),
                            error=rr.error,
                            error_type=rr.error_type,
                            tokens_generated=rr.tokens_generated,
                            elapsed_s=rr.elapsed_seconds,
                            role_history=rr.role_history,
                            delegation_events=rr.delegation_events,
                            delegation_diagnostics=rr.delegation_diagnostics,
                            tools_used=rr.tools_used,
                            tools_called=rr.tools_called,
                            tap_offset_bytes=rr.tap_offset_bytes,
                            tap_length_bytes=rr.tap_length_bytes,
                            repl_tap_offset_bytes=rr.repl_tap_offset_bytes,
                            repl_tap_length_bytes=rr.repl_tap_length_bytes,
                            cost_dimensions=rr.cost_dimensions,
                            think_harder_attempted=rr.think_harder_attempted,
                            think_harder_succeeded=rr.think_harder_succeeded,
                            cheap_first_attempted=rr.cheap_first_attempted,
                            cheap_first_passed=rr.cheap_first_passed,
                            grammar_enforced=rr.grammar_enforced,
                            parallel_tools_used=rr.parallel_tools_used,
                            cache_affinity_bonus=rr.cache_affinity_bonus,
                            # SkillBank retrieval data
                            skills_retrieved=rr.skills_retrieved,
                            skill_types=[sid.split("_")[0] for sid in rr.skill_ids] if rr.skill_ids else [],
                            skill_context_tokens=0,
                            # Context window management and budget tracking
                            budget_diagnostics=rr.budget_diagnostics,
                            tool_results_cleared=rr.tool_results_cleared,
                            compaction_triggered=rr.compaction_triggered,
                            compaction_tokens_saved=rr.compaction_tokens_saved,
                            think_harder_expected_roi=rr.think_harder_expected_roi,
                        )
                        append_diagnostic(diag)
                        debugger.add_diagnostic(diag)
                    debugger.end_question()

                    # ── Post-fix mini regression suite ──────────────
                    failed_retries, affected_suites = debugger.pop_retries()
                    if failed_retries:
                        logger.info(
                            f"  [RETRY] Post-fix regression suite: "
                            f"{len(failed_retries)} verify + generalize + regress"
                        )

                        # Health-check: poll API readiness after hot-restart
                        for _hc in range(10):
                            if _check_server_health(url):
                                break
                            time.sleep(1)

                        retry_questions: list[tuple[dict, str]] = []  # (prompt_info, tag)

                        # 1. VERIFY: the exact failed questions
                        for s, q in failed_retries:
                            pi = prompt_info_by_qid.get((s, q))
                            if pi:
                                retry_questions.append((pi, "verify"))

                        # 2. GENERALIZE: 2 fresh unseen questions per affected suite
                        seen_now = load_seen_questions()
                        fresh = sample_unseen_questions(
                            list(affected_suites), 2, seen_now,
                            seed=int(time.time()),
                            use_pool=True, allow_reseen=True,
                        )
                        for fpi in fresh:
                            fkey = (fpi["suite"], fpi["id"])
                            if fkey not in prompt_info_by_qid:
                                prompt_info_by_qid[fkey] = fpi
                            retry_questions.append((fpi, "generalize"))

                        # 3. REGRESS: up to 2 previously-passing per affected suite
                        import random as _retry_rng
                        for rs in affected_suites:
                            candidates = passing_by_suite.get(rs, [])
                            regress_sample = _retry_rng.sample(
                                candidates, min(2, len(candidates)),
                            )
                            for rpi in regress_sample:
                                retry_questions.append((rpi, "regress"))

                        # Run the mini regression suite
                        retry_batch_id = debugger.batch_count
                        for rpi, tag in retry_questions:
                            if state.shutdown:
                                break
                            logger.info(
                                f"  [RETRY:{tag.upper()}] "
                                f"{rpi['suite']}/{rpi['id']}"
                            )

                            rr_retry, rew_retry, meta_retry = evaluate_question_3way(
                                rpi, url, timeout, _client, dry_run,
                            )
                            meta_retry["is_retry"] = True
                            meta_retry["retry_tag"] = tag
                            meta_retry["retry_batch_id"] = retry_batch_id

                            # Inject rewards (post-fix signal)
                            ri_retry = 0
                            if not dry_run:
                                ri_retry = _inject_3way_rewards_http(
                                    rpi["prompt"][:200], rpi["suite"],
                                    rpi["id"], rew_retry, meta_retry,
                                    url, _client,
                                )

                            retry_result = ThreeWayResult(
                                suite=rpi["suite"],
                                question_id=rpi["id"],
                                prompt=rpi["prompt"][:200],
                                expected=rpi.get("expected", "")[:200],
                                timestamp=datetime.now(timezone.utc).isoformat(),
                                role_results=rr_retry,
                                rewards=rew_retry,
                                metadata=meta_retry,
                                rewards_injected=ri_retry,
                            )
                            results.append(retry_result)
                            checkpoint_result(session_id, retry_result)

                            # Track passing retries for future regression
                            if any(r.passed for r in rr_retry.values()):
                                passing_by_suite.setdefault(
                                    rpi["suite"], [],
                                ).append(rpi)

                            # Feed diagnostics back to debugger
                            for ck, rr_diag in rr_retry.items():
                                retry_diag = build_diagnostic(
                                    question_id=f"{rpi['suite']}/{rpi['id']}",
                                    suite=rpi["suite"],
                                    config=ck,
                                    role=rr_diag.role,
                                    mode=rr_diag.mode,
                                    passed=rr_diag.passed,
                                    answer=rr_diag.answer,
                                    expected=rpi.get("expected", ""),
                                    scoring_method=rpi.get(
                                        "scoring_method", "exact_match",
                                    ),
                                    error=rr_diag.error,
                                    error_type=rr_diag.error_type,
                                    tokens_generated=rr_diag.tokens_generated,
                                    elapsed_s=rr_diag.elapsed_seconds,
                                    role_history=rr_diag.role_history,
                                    delegation_events=rr_diag.delegation_events,
                                    delegation_diagnostics=rr_diag.delegation_diagnostics,
                                    tools_used=rr_diag.tools_used,
                                    tools_called=rr_diag.tools_called,
                                    tap_offset_bytes=rr_diag.tap_offset_bytes,
                                    tap_length_bytes=rr_diag.tap_length_bytes,
                                    repl_tap_offset_bytes=rr_diag.repl_tap_offset_bytes,
                                    repl_tap_length_bytes=rr_diag.repl_tap_length_bytes,
                                    cost_dimensions=rr_diag.cost_dimensions,
                                    think_harder_attempted=rr_diag.think_harder_attempted,
                                    think_harder_succeeded=rr_diag.think_harder_succeeded,
                                    cheap_first_attempted=rr_diag.cheap_first_attempted,
                                    cheap_first_passed=rr_diag.cheap_first_passed,
                                    grammar_enforced=rr_diag.grammar_enforced,
                                    parallel_tools_used=rr_diag.parallel_tools_used,
                                    cache_affinity_bonus=rr_diag.cache_affinity_bonus,
                                    skills_retrieved=rr_diag.skills_retrieved,
                                    skill_types=[sid.split("_")[0] for sid in rr_diag.skill_ids] if rr_diag.skill_ids else [],
                                    skill_context_tokens=0,
                                    budget_diagnostics=rr_diag.budget_diagnostics,
                                    tool_results_cleared=rr_diag.tool_results_cleared,
                                    compaction_triggered=rr_diag.compaction_triggered,
                                    compaction_tokens_saved=rr_diag.compaction_tokens_saved,
                                    think_harder_expected_roi=rr_diag.think_harder_expected_roi,
                                )
                                retry_diag["is_retry"] = True
                                retry_diag["retry_tag"] = tag
                                append_diagnostic(retry_diag)
                                debugger.add_diagnostic(retry_diag)
                            debugger.end_question()

                        logger.info(
                            f"  [RETRY] Suite complete: "
                            f"{len(retry_questions)} questions evaluated"
                        )

                except Exception as e:
                    logger.warning(f"[DEBUG] Debugger error (non-fatal): {e}")

    finally:
        _client.close()
        if debugger is not None:
            debugger.flush()

    return results


def print_3way_summary(results: list[ThreeWayResult]) -> None:
    """Print summary of 3-way evaluation results."""
    if not results:
        print("No results to summarize.")
        return

    action_stats: dict[str, dict[str, Any]] = {
        a: {"pass": 0, "fail": 0, "error": 0, "total_reward": 0.0, "n": 0}
        for a in THREE_WAY_ACTIONS
    }

    tool_value_stats = {"helped": 0, "neutral": 0, "hurt": 0}

    for result in results:
        for action, reward in result.rewards.items():
            if action in action_stats:
                action_stats[action]["n"] += 1
                action_stats[action]["total_reward"] += reward
                if reward >= 0.5:
                    action_stats[action]["pass"] += 1
                else:
                    action_stats[action]["fail"] += 1

        meta = result.metadata
        if meta.get("tools_helped"):
            tool_value_stats["helped"] += 1
        elif meta.get("tools_hurt"):
            tool_value_stats["hurt"] += 1
        else:
            tool_value_stats["neutral"] += 1

    print(f"\n{'='*70}")
    print("3-WAY ROUTING EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"Questions: {len(results)}")

    print(f"\n{'Action':20s} {'Pass':>5s} {'Fail':>5s} {'Acc%':>7s} {'Q-bar':>7s}")
    print("-" * 50)
    for action in THREE_WAY_ACTIONS:
        s = action_stats[action]
        total = s["pass"] + s["fail"]
        acc = s["pass"] / total * 100 if total > 0 else 0
        avg_q = s["total_reward"] / s["n"] if s["n"] > 0 else 0.5
        print(f"{action:20s} {s['pass']:5d} {s['fail']:5d} {acc:6.1f}% {avg_q:6.3f}")

    print(f"\nTool Value (SELF:direct vs SELF:repl):")
    print(f"  Tools helped: {tool_value_stats['helped']}")
    print(f"  Tools neutral: {tool_value_stats['neutral']}")
    print(f"  Tools hurt: {tool_value_stats['hurt']}")

    total_injected = sum(r.rewards_injected for r in results)
    print(f"\nRewards injected: {total_injected}")


# ── CLI ───────────────────────────────────────────────────────────────


def _build_retrieval_config_from_args(args) -> "RetrievalConfig":
    """Build RetrievalConfig with optional CLI overrides for replay/debug tuning."""
    from orchestration.repl_memory.retriever import RetrievalConfig

    overrides: dict[str, Any] = {}
    for key in (
        "cost_lambda",
        "confidence_threshold",
        "confidence_estimator",
        "confidence_trim_ratio",
        "confidence_min_neighbors",
        "warm_probability_hit",
        "warm_probability_miss",
        "warm_cost_fallback_s",
        "cold_cost_fallback_s",
        "calibrated_confidence_threshold",
        "conformal_margin",
        "risk_control_enabled",
        "risk_budget_id",
        "risk_gate_min_samples",
        "risk_abstain_target_role",
        "risk_gate_rollout_ratio",
        "risk_gate_kill_switch",
        "risk_budget_guardrail_min_events",
        "risk_budget_guardrail_max_abstain_rate",
        "prior_strength",
    ):
        val = getattr(args, key, None)
        if val is not None:
            overrides[key] = val
    return RetrievalConfig(**overrides)


_PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    "baseline": {
        "cooldown": 0.0,
        "timeout": None,
        "env": {},
    },
    "infra-stable": {
        "cooldown": 2.0,
        "timeout": None,
        "env": {
            "ORCHESTRATOR_DEFERRED_TOOL_RESULTS": "1",
            "ORCHESTRATOR_INFERENCE_LOCK_TIMEOUT_EXCLUSIVE_S": "45",
            "ORCHESTRATOR_INFERENCE_LOCK_TIMEOUT_SHARED_S": "45",
            "ORCHESTRATOR_UVICORN_WORKERS": "1",
        },
    },
}


def _apply_profile(args: argparse.Namespace) -> None:
    profile = _PROFILE_PRESETS.get(args.profile, _PROFILE_PRESETS["baseline"])

    for key, value in profile.get("env", {}).items():
        os.environ.setdefault(key, str(value))

    if args.cooldown is None:
        args.cooldown = float(profile.get("cooldown", 0.0))
    if args.timeout is None:
        timeout_default = profile.get("timeout")
        args.timeout = int(timeout_default) if timeout_default is not None else int(DEFAULT_TIMEOUT)

    logger.info(
        "Seeding profile=%s cooldown=%.1fs timeout=%ss deferred_tool_results=%s "
        "lock_timeout_exclusive_s=%s lock_timeout_shared_s=%s uvicorn_workers=%s",
        args.profile,
        args.cooldown,
        args.timeout,
        os.environ.get("ORCHESTRATOR_DEFERRED_TOOL_RESULTS", "0"),
        os.environ.get("ORCHESTRATOR_INFERENCE_LOCK_TIMEOUT_EXCLUSIVE_S", ""),
        os.environ.get("ORCHESTRATOR_INFERENCE_LOCK_TIMEOUT_SHARED_S", ""),
        os.environ.get("ORCHESTRATOR_UVICORN_WORKERS", ""),
    )


def main():
    parser = argparse.ArgumentParser(
        description="MemRL Episodic Seeding for 3-Way Routing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (3-way mode - RECOMMENDED):
  # THE command. Run continuously for days:
  %(prog)s --3way --continuous --suites all --sample-size 10 --preflight

  # One-shot batch:
  %(prog)s --3way --suites thinking coder --sample-size 5

  # Dry run (no reward injection):
  %(prog)s --3way --dry-run --suites thinking --sample-size 3

Examples (legacy mode - DEPRECATED):
  # Quick stats from all sessions:
  %(prog)s --stats
""",
    )
    parser.add_argument(
        "--suites", nargs="+", default=DEFAULT_SUITES,
        help=f"Suites to evaluate (default: {' '.join(DEFAULT_SUITES)})",
    )
    parser.add_argument(
        "--roles", nargs="+", default=DEFAULT_ROLES,
        help=f"Roles to compare (default: {' '.join(DEFAULT_ROLES)})",
    )
    parser.add_argument(
        "--modes", nargs="+", default=DEFAULT_MODES,
        help=f"Execution modes to test (default: {' '.join(DEFAULT_MODES)}). "
        "Architect roles always use direct+delegated.",
    )
    parser.add_argument(
        "--profile",
        choices=tuple(_PROFILE_PRESETS.keys()),
        default="infra-stable",
        help="Runtime tuning preset (default: infra-stable).",
    )
    parser.add_argument(
        "--sample-size", type=int, default=10,
        help="Questions per suite per batch (default: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (default: timestamp)",
    )
    parser.add_argument(
        "--url", default=DEFAULT_ORCHESTRATOR_URL,
        help=f"Orchestrator URL (default: {DEFAULT_ORCHESTRATOR_URL})",
    )
    parser.add_argument(
        "--timeout", type=int, default=None,
        help=f"Request timeout in seconds (default from profile; fallback {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Score only, don't inject rewards",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file (default: auto-generated). Only for one-shot mode.",
    )
    parser.add_argument(
        "--skip-cache", action="store_true",
        help="Disable KV cache reuse (cache_prompt=False).",
    )
    parser.add_argument(
        "--cooldown", type=float, default=None,
        help="Seconds between strategy calls (default from profile).",
    )
    parser.add_argument(
        "--no-dedup", action="store_true",
        help="Disable URL-based role deduplication.",
    )
    # Continuous mode
    parser.add_argument(
        "--continuous", action="store_true",
        help="Run batches continuously until Ctrl+C. Checkpoints per question.",
    )
    parser.add_argument(
        "--continuous-interval", type=int, default=30,
        help="Seconds between continuous batches (default: 30)",
    )
    parser.add_argument(
        "--resume", default=None,
        help="Resume a specific session ID (e.g. seeding_20260201_143022)",
    )
    # Preflight & stats
    parser.add_argument(
        "--preflight", action="store_true",
        help="Run health checks and smoke test before starting.",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Show aggregate stats from all sessions, then exit.",
    )
    parser.add_argument(
        "--no-escalation-chains", action="store_true",
        help="Disable escalation chain reward detection (enabled by default). "
        "Escalation chains detect when cheap models fail but expensive models "
        "pass on the same question, and inject escalation rewards into MemRL.",
    )
    # Phase 4: 3-way routing mode
    parser.add_argument(
        "--3way", action="store_true", dest="three_way",
        help="Use 3-way routing evaluation (SELF:direct, SELF:repl, ARCHITECT). "
        "Binary rewards for faithful probability estimation. "
        "Ignores --roles and --modes flags.",
    )
    parser.add_argument(
        "--tui", action="store_true",
        help="Rich split-screen TUI (requires --3way)",
    )
    # Question pool options
    parser.add_argument(
        "--rebuild-pool", action="store_true",
        help="Rebuild the pre-extracted question pool from all adapters, then exit.",
    )
    parser.add_argument(
        "--no-pool", action="store_true",
        help="Bypass the question pool and load from HF adapters directly (slow).",
    )
    # Claude-in-the-loop debugging
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable Claude-in-the-loop debugging (requires --3way).",
    )
    parser.add_argument(
        "--debug-batch-size", type=int, default=5,
        help="Diagnostic records per Claude invocation (default: 5).",
    )
    parser.add_argument(
        "--debug-threshold", type=float, default=0.3,
        help="Anomaly score threshold for diagnostics (default: 0.3).",
    )
    parser.add_argument(
        "--debug-auto-commit", action="store_true",
        help="Auto-commit each debug batch (easy git revert).",
    )
    parser.add_argument(
        "--debug-dry-run", action="store_true",
        help="Compute diagnostics and log what would be sent to Claude, but don't invoke.",
    )
    parser.add_argument(
        "--debug-replay", action="store_true",
        help="Include MemRL replay context in debug prompts (routing accuracy, Q-convergence).",
    )
    parser.add_argument("--cost-lambda", type=float, default=None, dest="cost_lambda")
    parser.add_argument("--confidence-threshold", type=float, default=None, dest="confidence_threshold")
    parser.add_argument(
        "--confidence-estimator",
        choices=("median", "trimmed_mean"),
        default=None,
        dest="confidence_estimator",
    )
    parser.add_argument("--confidence-trim-ratio", type=float, default=None, dest="confidence_trim_ratio")
    parser.add_argument("--confidence-min-neighbors", type=int, default=None, dest="confidence_min_neighbors")
    parser.add_argument("--warm-probability-hit", type=float, default=None, dest="warm_probability_hit")
    parser.add_argument("--warm-probability-miss", type=float, default=None, dest="warm_probability_miss")
    parser.add_argument("--warm-cost-fallback-s", type=float, default=None, dest="warm_cost_fallback_s")
    parser.add_argument("--cold-cost-fallback-s", type=float, default=None, dest="cold_cost_fallback_s")
    parser.add_argument(
        "--calibrated-confidence-threshold",
        type=float,
        default=None,
        dest="calibrated_confidence_threshold",
    )
    parser.add_argument("--conformal-margin", type=float, default=None, dest="conformal_margin")
    parser.add_argument(
        "--risk-control-enabled",
        action="store_true",
        dest="risk_control_enabled",
        help="Enable calibrated confidence threshold for conformal abstain/escalate behavior.",
    )
    parser.add_argument("--risk-budget-id", type=str, default=None, dest="risk_budget_id")
    parser.add_argument("--risk-gate-min-samples", type=int, default=None, dest="risk_gate_min_samples")
    parser.add_argument(
        "--risk-abstain-target-role",
        type=str,
        default=None,
        dest="risk_abstain_target_role",
    )
    parser.add_argument(
        "--risk-gate-rollout-ratio",
        type=float,
        default=None,
        dest="risk_gate_rollout_ratio",
    )
    parser.add_argument(
        "--risk-gate-kill-switch",
        action="store_true",
        dest="risk_gate_kill_switch",
    )
    parser.add_argument(
        "--risk-budget-guardrail-min-events",
        type=int,
        default=None,
        dest="risk_budget_guardrail_min_events",
    )
    parser.add_argument(
        "--risk-budget-guardrail-max-abstain-rate",
        type=float,
        default=None,
        dest="risk_budget_guardrail_max_abstain_rate",
    )
    parser.add_argument("--prior-strength", type=float, default=None, dest="prior_strength")
    parser.add_argument(
        "--evolve", action="store_true",
        help="Run skill evolution cycle after seeding (requires ORCHESTRATOR_SKILLBANK=1).",
    )

    args = parser.parse_args()
    _apply_profile(args)

    # Rebuild pool — build and exit
    if args.rebuild_pool:
        from question_pool import build_pool
        import time as _time
        t0 = _time.monotonic()
        stats = build_pool()
        elapsed = _time.monotonic() - t0
        total = sum(stats.values())
        print(f"Pool rebuilt in {elapsed:.1f}s: {total} questions across {len(stats)} suites")
        for suite, count in sorted(stats.items(), key=lambda x: -x[1]):
            print(f"  {suite:25s} {count:>6,d}")
        return

    # Stats mode — just print and exit
    if args.stats:
        print_stats()
        return

    # Preflight
    if args.preflight:
        if not run_preflight(args.url):
            sys.exit(1)

    # Session ID
    if args.resume:
        session_id = args.resume
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"seeding_{ts}" if not args.three_way else f"3way_{ts}"

    # Publish session_id for cross-request persistence (Phase 3 checkpoints)
    state.session_id = session_id

    # Default seed
    base_seed = args.seed if args.seed is not None else int(time.time())
    replay_retrieval_config = _build_retrieval_config_from_args(args)
    replay_retrieval_overrides = {
        k: getattr(replay_retrieval_config, k)
        for k in (
            "cost_lambda",
            "confidence_threshold",
            "confidence_estimator",
            "confidence_trim_ratio",
            "confidence_min_neighbors",
            "warm_probability_hit",
            "warm_probability_miss",
            "warm_cost_fallback_s",
            "cold_cost_fallback_s",
            "calibrated_confidence_threshold",
            "conformal_margin",
            "risk_control_enabled",
            "risk_budget_id",
            "risk_gate_min_samples",
            "risk_abstain_target_role",
            "risk_gate_rollout_ratio",
            "risk_gate_kill_switch",
            "risk_budget_guardrail_min_events",
            "risk_budget_guardrail_max_abstain_rate",
            "prior_strength",
        )
    }

    # ── Claude-in-the-loop debugger setup ──
    if args.debug_dry_run or args.debug_auto_commit:
        args.debug = True
    _debugger = None
    if args.debug:
        if not args.three_way:
            logger.warning("--debug requires --3way. Ignoring.")
        else:
            from src.pipeline_monitor.claude_debugger import ClaudeDebugger
            _debugger = ClaudeDebugger(
                project_root=PROJECT_ROOT,
                batch_size=args.debug_batch_size,
                anomaly_threshold=args.debug_threshold,
                auto_commit=args.debug_auto_commit,
                dry_run=args.debug_dry_run,
                replay_context=args.debug_replay,
                retrieval_overrides=replay_retrieval_overrides,
            )
            logger.info(f"[DEBUG] Claude-in-the-loop debugger enabled "
                        f"(batch_size={args.debug_batch_size}, "
                        f"threshold={args.debug_threshold})")

    # ── SkillBank OutcomeTracker setup ──
    _outcome_tracker = None
    if args.evolve:
        try:
            from orchestration.repl_memory.skill_evolution import OutcomeTracker
            _outcome_tracker = OutcomeTracker()
            logger.info("[SKILLBANK] OutcomeTracker enabled for --evolve")
        except Exception as _ot_err:
            logger.error("[SKILLBANK] OutcomeTracker init failed: %s", _ot_err)
            logger.error("[SKILLBANK] --evolve requires OutcomeTracker. Fix the error above.")

    # Interval (batches) between periodic evolve/replay runs in continuous mode
    _POST_BATCH_HOOK_INTERVAL = 10

    def _run_post_batch_hooks(batch_num: int) -> None:
        """Run --evolve and --debug-replay hooks periodically during continuous mode."""
        if batch_num % _POST_BATCH_HOOK_INTERVAL != 0:
            return

        if args.debug_replay:
            try:
                from orchestration.repl_memory.replay.trajectory import TrajectoryExtractor
                from orchestration.repl_memory.replay.engine import ReplayEngine
                from orchestration.repl_memory.q_scorer import ScoringConfig
                from orchestration.repl_memory.progress_logger import ProgressReader

                logger.info("[REPLAY] Periodic replay evaluation (batch %d)...", batch_num)
                reader = ProgressReader()
                extractor = TrajectoryExtractor(reader)
                trajectories = extractor.extract_complete(days=14, max_trajectories=1000)
                if trajectories:
                    engine = ReplayEngine()
                    metrics = engine.run_with_metrics(
                        replay_retrieval_config, ScoringConfig(), trajectories,
                        f"periodic_batch_{batch_num}",
                    )
                    logger.info(
                        "[REPLAY] batch=%d trajectories=%d accuracy=%.1f%% avg_reward=%.3f",
                        batch_num, metrics.num_trajectories,
                        metrics.routing_accuracy * 100, metrics.avg_reward,
                    )
                else:
                    logger.info("[REPLAY] No complete trajectories found.")
            except Exception as e:
                logger.warning("[REPLAY] Periodic replay failed (non-fatal): %s", e)

        if args.evolve and _outcome_tracker is not None:
            try:
                from orchestration.repl_memory.skill_evolution import EvolutionMonitor
                from orchestration.repl_memory.skill_bank import SkillBank

                skill_db = Path("orchestration/repl_memory/sessions/skills.db")
                if skill_db.exists():
                    sb = SkillBank(db_path=skill_db)
                    monitor = EvolutionMonitor(sb)
                    report = monitor.run_evolution_cycle(outcome_tracker=_outcome_tracker)
                    logger.info(
                        "[EVOLVE] batch=%d evaluated=%d promoted=%d decayed=%d deprecated=%d",
                        batch_num, report.skills_evaluated, report.skills_promoted,
                        report.skills_decayed, report.skills_deprecated,
                    )
                    sb.close()
            except Exception as e:
                logger.warning("[EVOLVE] Periodic evolution failed (non-fatal): %s", e)

    # ── Phase 4: 3-Way Routing Mode ──
    if args.three_way:
        from contextlib import nullcontext

        if args.tui:
            from seeding_tui import SeedingTUI
            tui = SeedingTUI(session_id=session_id)
            tui_ctx = tui
        else:
            tui = None
            tui_ctx = nullcontext()

        def _on_progress(idx: int, total: int, suite: str, qid: str, question: str = "") -> None:
            if tui is not None:
                tui.update_progress(idx, total, suite, qid, question=question)

        with tui_ctx:
            if args.continuous:
                batch = 0
                consecutive_failures = 0
                all_results: list[ThreeWayResult] = []
                logger.info(f"Starting continuous 3-way evaluation: session={session_id}")
                logger.info(f"  Ctrl+C to stop gracefully (finishes current question)")

                while not state.shutdown:
                    if not _check_server_health(args.url):
                        consecutive_failures += 1
                        if consecutive_failures > MAX_RECOVERY_ATTEMPTS:
                            logger.error(f"API unrecoverable after {MAX_RECOVERY_ATTEMPTS} attempts.")
                            break
                        backoff = min(30 * (2 ** (consecutive_failures - 1)), 600)
                        logger.warning(f"API down. Attempting recovery...")
                        recovered = _attempt_recovery(args.url)
                        if recovered:
                            logger.info("Recovery successful — resuming")
                            consecutive_failures = 0
                            continue
                        logger.warning(f"Recovery failed — sleeping {backoff}s")
                        for _ in range(backoff):
                            if state.shutdown:
                                break
                            time.sleep(1)
                        continue
                    consecutive_failures = 0

                    batch += 1
                    batch_seed = base_seed + batch
                    logger.info(f"\n[3-Way Batch {batch}, seed={batch_seed}]")

                    try:
                        results = run_batch_3way(
                            suites=args.suites,
                            sample_per_suite=args.sample_size,
                            seed=batch_seed,
                            url=args.url,
                            timeout=args.timeout,
                            session_id=session_id,
                            dry_run=args.dry_run,
                            cooldown=args.cooldown,
                            on_progress=_on_progress,
                            use_pool=not args.no_pool,
                            debugger=_debugger,
                            outcome_tracker=_outcome_tracker,
                        )
                        all_results.extend(results)

                        # Run evolve/replay hooks periodically
                        _run_post_batch_hooks(batch)

                        if not results:
                            logger.info("No unseen questions. Waiting 60s...")
                            for _ in range(60):
                                if state.shutdown:
                                    break
                                time.sleep(1)

                    except Exception as e:
                        logger.error(f"Batch failed: {e}")
                        time.sleep(10)

            else:
                # One-shot 3-way mode
                logger.info(f"Starting 3-way routing evaluation: session={session_id}")
                results = run_batch_3way(
                    suites=args.suites,
                    sample_per_suite=args.sample_size,
                    seed=base_seed,
                    url=args.url,
                    timeout=args.timeout,
                    session_id=session_id,
                    dry_run=args.dry_run,
                    cooldown=args.cooldown,
                    on_progress=_on_progress,
                    use_pool=not args.no_pool,
                    debugger=_debugger,
                    outcome_tracker=_outcome_tracker,
                )

        # Summary printed AFTER TUI context exits (normal terminal restored)
        if args.continuous:
            print_3way_summary(all_results)
        else:
            print_3way_summary(results)

        # End-of-run replay evaluation (if --debug-replay enabled)
        if args.debug_replay:
            try:
                from orchestration.repl_memory.replay.trajectory import TrajectoryExtractor
                from orchestration.repl_memory.replay.engine import ReplayEngine
                from orchestration.repl_memory.q_scorer import ScoringConfig
                from orchestration.repl_memory.progress_logger import ProgressReader

                logger.info("[REPLAY] Running post-seeding replay evaluation...")
                reader = ProgressReader()
                extractor = TrajectoryExtractor(reader)
                trajectories = extractor.extract_complete(days=14, max_trajectories=1000)
                if trajectories:
                    # Try skill-aware replay if SkillBank available
                    skill_metrics = None
                    try:
                        from orchestration.repl_memory.replay.skill_replay import (
                            SkillAwareReplayEngine, SkillBankConfig,
                        )
                        from orchestration.repl_memory.skill_bank import SkillBank

                        skill_db = Path("orchestration/repl_memory/sessions/skills.db")
                        if skill_db.exists():
                            sb = SkillBank(db_path=skill_db)
                            engine = SkillAwareReplayEngine(skill_bank=sb)
                            skill_metrics = engine.run_with_skill_metrics(
                                replay_retrieval_config, ScoringConfig(), SkillBankConfig(),
                                trajectories, "post_seeding",
                            )
                            metrics = skill_metrics.base_metrics
                    except ImportError:
                        pass

                    if skill_metrics is None:
                        engine = ReplayEngine()
                        metrics = engine.run_with_metrics(
                            replay_retrieval_config, ScoringConfig(), trajectories, "post_seeding",
                        )

                    print(f"\n{'='*70}")
                    print("REPLAY EVALUATION (default config, last 14 days)")
                    print(f"{'='*70}")
                    print(f"  Trajectories:      {metrics.num_trajectories}")
                    print(f"  Routing accuracy:  {metrics.routing_accuracy:.1%}")
                    for t, a in metrics.routing_accuracy_by_type.items():
                        print(f"    {t:20s} {a:.1%}")
                    print(f"  Cumulative reward: {metrics.cumulative_reward:.1f}")
                    print(f"  Avg reward:        {metrics.avg_reward:.3f}")
                    print(f"  Q convergence:     step {metrics.q_convergence_step}")
                    print(f"  Replay duration:   {metrics.replay_duration_seconds:.2f}s")
                    print(f"  Escalation:        prec={metrics.escalation_precision:.0%} "
                          f"recall={metrics.escalation_recall:.0%}")
                    print(
                        "  Calibration:       "
                        f"ECE={metrics.ece_global:.3f} "
                        f"Brier={metrics.brier_global:.3f} "
                        f"coverage={metrics.conformal_coverage:.1%} "
                        f"risk={metrics.conformal_risk:.1%}"
                    )

                    # Print skill metrics if available
                    if skill_metrics:
                        print(f"  Skills retrieved:  {skill_metrics.total_skills_retrieved}")
                        print(f"  Skill coverage:    {skill_metrics.skill_coverage:.1%}")
                        print(f"  Avg skills/step:   {skill_metrics.avg_skills_per_step:.1f}")
                else:
                    logger.info("[REPLAY] No complete trajectories found in last 14 days.")
            except Exception as e:
                logger.warning(f"[REPLAY] Replay evaluation failed (non-fatal): {e}")

        # End-of-run skill evolution (if --evolve enabled)
        if args.evolve:
            if _outcome_tracker is None:
                logger.error("[EVOLVE] --evolve passed but OutcomeTracker is None! "
                             "Evolution skipped. Check init errors above.")
        if args.evolve and _outcome_tracker is not None:
            try:
                from orchestration.repl_memory.skill_evolution import EvolutionMonitor
                from orchestration.repl_memory.skill_bank import SkillBank

                skill_db = Path("orchestration/repl_memory/sessions/skills.db")
                if skill_db.exists():
                    sb = SkillBank(db_path=skill_db)
                    monitor = EvolutionMonitor(sb)
                    report = monitor.run_evolution_cycle(outcome_tracker=_outcome_tracker)
                    print(f"\n{'='*70}")
                    print("SKILL EVOLUTION REPORT")
                    print(f"{'='*70}")
                    print(f"  Evaluated:    {report.skills_evaluated}")
                    print(f"  Promoted:     {report.skills_promoted}")
                    print(f"  Decayed:      {report.skills_decayed}")
                    print(f"  Deprecated:   {report.skills_deprecated}")
                    if report.redistillation_candidates:
                        print(f"  Redistill:    {', '.join(report.redistillation_candidates)}")
                    sb.close()
                else:
                    logger.info("[EVOLVE] No skills.db found — skipping evolution cycle.")
            except Exception as e:
                logger.warning(f"[EVOLVE] Evolution cycle failed: {e}")

        return

    # Compute alias_map for summary display
    alias_map: dict[str, str] = {}
    if not args.no_dedup:
        _, alias_map = _deduplicate_roles(args.roles)

    if args.continuous:
        # ── Continuous mode ──
        batch = 0
        consecutive_failures = 0
        logger.info(f"Starting continuous evaluation: session={session_id}")
        logger.info(f"  Ctrl+C to stop gracefully (finishes current question)")

        while not state.shutdown:
            if not _check_server_health(args.url):
                consecutive_failures += 1
                if consecutive_failures > MAX_RECOVERY_ATTEMPTS:
                    logger.error(
                        f"API unrecoverable after {MAX_RECOVERY_ATTEMPTS} attempts. Exiting."
                    )
                    break
                backoff = min(30 * (2 ** (consecutive_failures - 1)), 600)
                logger.warning(
                    f"API down (attempt {consecutive_failures}/{MAX_RECOVERY_ATTEMPTS}). "
                    f"Attempting recovery..."
                )
                recovered = _attempt_recovery(args.url)
                if recovered:
                    logger.info("  Recovery successful — resuming evaluation")
                    consecutive_failures = 0
                    continue
                logger.warning(f"  Recovery failed — sleeping {backoff}s before retry")
                for _ in range(backoff):
                    if state.shutdown:
                        break
                    time.sleep(1)
                continue
            consecutive_failures = 0

            batch += 1
            batch_seed = base_seed + batch
            logger.info(f"\n[Batch {batch}, seed={batch_seed}]")

            try:
                results = run_batch(
                    suites=args.suites,
                    roles=args.roles,
                    modes=args.modes,
                    sample_per_suite=args.sample_size,
                    seed=batch_seed,
                    url=args.url,
                    timeout=args.timeout,
                    session_id=session_id,
                    dry_run=args.dry_run,
                    skip_cache=args.skip_cache,
                    cooldown=args.cooldown,
                    no_dedup=args.no_dedup,
                    escalation_chains=not args.no_escalation_chains,
                    use_pool=not args.no_pool,
                )
            except HealthCheckError:
                logger.warning("API died during batch — will attempt recovery")
                continue

            if results:
                print_batch_summary(results, args.roles, args.modes, alias_map=alias_map)

            if state.shutdown:
                break

            logger.info(f"\n[Sleeping {args.continuous_interval}s before next batch...]")
            for _ in range(args.continuous_interval):
                if state.shutdown:
                    break
                time.sleep(1)

        logger.info(f"\nSession complete: {session_id}")
        logger.info(f"  Run --stats to see aggregate results")

    else:
        # ── One-shot mode (original behavior with HF datasets) ──
        import json
        from dataclasses import asdict

        results = run_batch(
            suites=args.suites,
            roles=args.roles,
            modes=args.modes,
            sample_per_suite=args.sample_size,
            seed=base_seed,
            url=args.url,
            timeout=args.timeout,
            session_id=session_id,
            dry_run=args.dry_run,
            skip_cache=args.skip_cache,
            cooldown=args.cooldown,
            no_dedup=args.no_dedup,
            escalation_chains=not args.no_escalation_chains,
            use_pool=not args.no_pool,
        )

        if results:
            print_batch_summary(results, args.roles, args.modes, alias_map=alias_map)

        # Save JSON output (legacy format for backwards compat)
        output_path = args.output
        if output_path is None:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_path = str(
                PROJECT_ROOT / "benchmarks" / "results" / "orchestrator"
                / f"seeding_{ts}.json"
            )

        all_combos = _build_role_mode_combos(args.roles, args.modes)
        output_data = {
            "config": {
                "suites": args.suites,
                "roles": args.roles,
                "modes": args.modes,
                "combos": [f"{r}:{m}" for r, m in all_combos],
                "sample_size": args.sample_size,
                "seed": base_seed,
                "dry_run": args.dry_run,
                "dedup": not args.no_dedup,
                "alias_map": alias_map,
            },
            "results": [asdict(r) for r in results],
            "timestamp": datetime.utcnow().isoformat(),
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to: {output_path}")
        logger.info(f"JSONL checkpoint: {EVAL_DIR / f'{session_id}.jsonl'}")


if __name__ == "__main__":
    import signal

    _crash_log = Path("/mnt/raid0/llm/claude/logs/seeding_crash.log")
    _signal_exit_code = {"value": None}

    def _append_crash_marker(kind: str, details: str) -> None:
        try:
            with open(_crash_log, "a") as _f:
                from datetime import datetime as _dt
                _f.write(f"\n{'='*60}\n{_dt.now().isoformat()} {kind}\n{'='*60}\n")
                _f.write(details.rstrip() + "\n")
        except Exception:
            pass

    def _handle_signal(sig_num: int, _frame) -> None:
        sig_name = signal.Signals(sig_num).name
        _signal_exit_code["value"] = 128 + sig_num
        _msg = f"Received {sig_name} ({sig_num}). External termination requested."
        logger.error(_msg)
        _append_crash_marker("TERMINATED", _msg)
        raise SystemExit(128 + sig_num)

    for _sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        try:
            signal.signal(_sig, _handle_signal)
        except Exception:
            # Not all signals are available on all platforms.
            pass

    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception:
        import traceback as _tb
        _crash_msg = _tb.format_exc()
        logger.error("Fatal crash:\n%s", _crash_msg)
        try:
            with open(_crash_log, "a") as _f:
                from datetime import datetime as _dt
                _f.write(f"\n{'='*60}\n{_dt.now().isoformat()} CRASH\n{'='*60}\n")
                _f.write(_crash_msg)
        except Exception:
            pass
        raise
    finally:
        state.close_poll_client()
