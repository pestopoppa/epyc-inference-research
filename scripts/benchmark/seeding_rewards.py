"""Reward computation, escalation chain detection, and reward injection.

Imports only seeding_types — no other project modules.
"""

from __future__ import annotations

from typing import Any

from seeding_types import (
    ComparativeResult,
    ESCALATION_REWARD,
    ROLE_COST_TIER,
    RoleResult,
    # Phase 4: 3-way routing
    ACTION_SELF_DIRECT,
    ACTION_SELF_REPL,
    ACTION_ARCHITECT,
    ACTION_WORKER,
    THREE_WAY_COST_TIER,
)

__all__ = [
    "DEFAULT_BASELINE_TPS",
    "compute_comparative_rewards",
    "detect_escalation_chains",
    # Phase 4: Binary rewards for faithful probability estimation
    "success_reward",
    "compute_3way_rewards",
    "score_delegation_chain",
    "compute_tool_value",
]

# Default per-role optimized tokens/second from production benchmarks.
# Update these when swapping models in the orchestrator stack.
DEFAULT_BASELINE_TPS: dict[str, float] = {
    "frontdoor": 18.3,
    "coder_escalation": 18.3,
    "coder_escalation": 39.44,
    "architect_general": 6.75,
    "architect_coding": 10.3,
    "ingest_long_context": 6.29,
    "worker_explore": 27.88,
    "worker_math": 48.5,
    "worker_vision": 15.28,
    "vision_escalation": 27.6,
}


def compute_comparative_rewards(
    role_results: dict[str, RoleResult],
    baseline_key: str = "frontdoor:direct",
    cost_config: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Compute comparative rewards relative to the baseline.

    Reward scheme (xRouter-style, correctness-gated cost penalty):
      specialist correct & baseline wrong -> +1.0 (clear specialist win)
      specialist wrong & baseline right   -> -0.5 (specialist worse)
      both correct -> 0.5 - lambda * max(0, cost_ratio - 1.0)  (cost-aware)
      both wrong   -> -0.3 (neither helps)
    """
    cost_config = cost_config or {}
    lam = cost_config.get("lambda", 0.15)
    baseline_tps = cost_config.get("baseline_tps_by_role", DEFAULT_BASELINE_TPS)

    rewards: dict[str, float] = {}
    baseline = role_results.get(baseline_key)
    if baseline is None:
        for key, result in role_results.items():
            rewards[key] = 1.0 if result.passed else 0.0
        return rewards

    baseline_passed = baseline.passed

    for key, result in role_results.items():
        if key == baseline_key:
            rewards[key] = 1.0 if result.passed else 0.0
        elif result.passed and not baseline_passed:
            rewards[key] = 1.0
        elif not result.passed and baseline_passed:
            rewards[key] = -0.5
        elif result.passed and baseline_passed:
            base = 0.5
            role_tps = baseline_tps.get(result.role, 0)
            gen_elapsed = result.generation_ms / 1000.0 if result.generation_ms > 0 else 0
            actual_elapsed = gen_elapsed if gen_elapsed > 0 else result.elapsed_seconds
            if (role_tps > 0 and result.tokens_generated > 0
                    and actual_elapsed > 0):
                expected = result.tokens_generated / role_tps
                cost_ratio = actual_elapsed / expected
                cost_penalty = lam * max(0.0, cost_ratio - 1.0)
                rewards[key] = max(0.1, base - cost_penalty)
            else:
                rewards[key] = 0.3
        else:
            rewards[key] = -0.3

    return rewards


def detect_escalation_chains(
    role_results: dict[str, RoleResult],
) -> list[dict[str, Any]]:
    """Detect cases where a cheap model fails but a more expensive one passes.

    Returns list of escalation chain dicts:
      {"from_role": "worker_explore", "from_mode": "direct",
       "to_role": "coder_escalation", "to_mode": "direct",
       "action": "escalate:worker_explore->coder_escalation",
       "reward": 0.8}
    """
    chains: list[dict[str, Any]] = []
    entries = []
    for key, rr in role_results.items():
        role, mode = key.split(":", 1)
        tier = ROLE_COST_TIER.get(role, 99)
        entries.append((tier, role, mode, rr))

    entries.sort(key=lambda x: x[0])

    # For each failed cheap role, find the cheapest passing expensive role
    for i, (tier_i, role_i, mode_i, rr_i) in enumerate(entries):
        if rr_i.passed or rr_i.error:
            continue  # Only look at failures (not errors)
        for j in range(i + 1, len(entries)):
            tier_j, role_j, mode_j, rr_j = entries[j]
            if tier_j <= tier_i:
                continue
            if rr_j.passed:
                chains.append({
                    "from_role": role_i,
                    "from_mode": mode_i,
                    "to_role": role_j,
                    "to_mode": mode_j,
                    "action": f"escalate:{role_i}->{role_j}",
                    "reward": ESCALATION_REWARD,
                })
                break  # Only the cheapest passing escalation target

    return chains


def _inject_escalation_chains_http(
    comp: ComparativeResult,
    chains: list[dict[str, Any]],
    url: str,
    client: "Any",
) -> int:
    """Inject escalation chain rewards via HTTP API.

    Returns number of rewards successfully injected.
    """
    injected = 0
    for chain in chains:
        try:
            resp = client.post(
                f"{url}/chat/reward",
                json={
                    "task_description": comp.prompt[:200],
                    "action": chain["action"],
                    "reward": chain["reward"],
                    "context": {
                        "task_type": comp.suite,
                        "source": "escalation_chain",
                        "question_id": comp.question_id,
                        "action_type": "escalation",
                        "from_role": chain["from_role"],
                        "to_role": chain["to_role"],
                    },
                },
                timeout=10,
            )
            if resp.status_code == 200:
                injected += 1
        except Exception as e:
            continue
    return injected


def _inject_rewards_http(
    comp: ComparativeResult,
    url: str,
    client: "Any",
) -> int:
    """Inject comparative rewards for one question via HTTP API.

    Returns number of rewards successfully injected.
    """
    injected = 0
    for action_key, reward in comp.rewards.items():
        try:
            resp = client.post(
                f"{url}/chat/reward",
                json={
                    "task_description": comp.prompt[:200],
                    "action": action_key,
                    "reward": reward,
                    "context": {
                        "task_type": comp.suite,
                        "source": "comparative_seeding",
                        "question_id": comp.question_id,
                        "comparative": True,
                    },
                },
                timeout=10,
            )
            if resp.status_code == 200:
                injected += 1
        except Exception as e:
            continue
    return injected


# ── Phase 4: Binary rewards for faithful probability estimation ──────


def success_reward(passed: bool) -> float:
    """Binary reward for faithful probability estimation.

    Q-values should converge to P(success|action).
    With binary rewards and TD learning (α=0.1):
        new_q = old_q + α(reward - old_q)
    converges to empirical success rate.

    Args:
        passed: Whether the task succeeded.

    Returns:
        1.0 for success, 0.0 for failure.
    """
    return 1.0 if passed else 0.0


def compute_3way_rewards(
    results: dict[str, RoleResult],
) -> dict[str, float]:
    """Compute binary rewards for 3-way routing evaluation.

    Maps role results to 3-way action categories and computes
    binary rewards. Cost is NOT included — Q-values represent
    pure P(success).

    Args:
        results: Dict mapping action_key (e.g. "frontdoor:direct") to RoleResult.

    Returns:
        Dict mapping 3-way action key to binary reward.
    """
    rewards: dict[str, float] = {}

    # SELF:direct — frontdoor/worker_vision without tools
    direct_keys = [k for k in results if k in ("frontdoor:direct", "worker_vision:direct")]
    if direct_keys:
        rewards[ACTION_SELF_DIRECT] = success_reward(results[direct_keys[0]].passed)

    # SELF:repl — frontdoor/worker_vision with tools (no delegation)
    # Backward-compatible: older sessions may still have worker_vision:react.
    repl_keys = [k for k in results if k in ("frontdoor:repl", "worker_vision:repl", "worker_vision:react")]
    if repl_keys:
        rewards[ACTION_SELF_REPL] = success_reward(results[repl_keys[0]].passed)

    # ARCHITECT — architect_coding, architect_general, or vision_escalation
    architect_keys = [k for k in results if k.startswith(("architect_coding", "architect_general", "vision_escalation"))]
    if architect_keys:
        # Take best architect result (they have delegation freedom)
        best_architect = max(architect_keys, key=lambda k: int(results[k].passed))
        rewards[ACTION_ARCHITECT] = success_reward(results[best_architect].passed)

    return rewards


def score_delegation_chain(
    results: dict[str, RoleResult],
) -> dict[str, float]:
    """Score WORKER based on delegation chain outcomes.

    Workers are "glorified tools" — they're triggered by SELF:repl or ARCHITECT
    delegation. When delegation occurs, we also inject a WORKER reward.

    Args:
        results: Dict mapping action_key to RoleResult.

    Returns:
        Dict with WORKER reward if delegation occurred.
    """
    rewards: dict[str, float] = {}

    # Check SELF:repl for delegation (frontdoor:repl for text, worker_vision:repl for VL).
    # Keep worker_vision:react for backward compatibility with older logs.
    repl_keys = [k for k in results if k in ("frontdoor:repl", "worker_vision:repl", "worker_vision:react")]
    for key in repl_keys:
        rr = results[key]
        if getattr(rr, "error_type", "none") == "infrastructure":
            continue
        if _has_delegation(rr):
            if rr.delegation_success is not None:
                rewards[ACTION_WORKER] = success_reward(rr.delegation_success)
            else:
                rewards[ACTION_WORKER] = success_reward(rr.passed)

    # Check ARCHITECT for delegation
    architect_keys = [k for k in results if k.startswith(("architect_coding", "architect_general", "vision_escalation"))]
    for key in architect_keys:
        rr = results[key]
        if getattr(rr, "error_type", "none") == "infrastructure":
            continue
        if _has_delegation(rr):
            score = (
                success_reward(rr.delegation_success)
                if rr.delegation_success is not None
                else success_reward(rr.passed)
            )
            if ACTION_WORKER in rewards:
                rewards[ACTION_WORKER] = max(rewards[ACTION_WORKER], score)
            else:
                rewards[ACTION_WORKER] = score

    return rewards


def _has_delegation(rr: RoleResult) -> bool:
    """Check if a RoleResult involved delegation to a worker."""
    if getattr(rr, "error_type", "none") == "infrastructure":
        return False
    if getattr(rr, "delegation_events", None):
        return True
    # Check tools_called for delegation indicators
    if rr.tools_called:
        delegation_tools = {"delegate", "delegate_to_worker", "spawn_worker"}
        for tool in rr.tools_called:
            if any(dt in tool.lower() for dt in delegation_tools):
                return True

    # Check role_history for worker involvement
    if rr.role_history and len(rr.role_history) > 1:
        worker_roles = {"worker_explore", "worker_math", "worker_vision", "worker_summarize"}
        for role in rr.role_history:
            if any(wr in role.lower() for wr in worker_roles):
                return True

    return False


def compute_tool_value(
    direct_passed: bool,
    repl_passed: bool,
) -> dict[str, Any]:
    """Compute tool value signal comparing SELF:direct vs SELF:repl.

    This is stored as metadata, not separate rewards.

    Args:
        direct_passed: Whether SELF:direct succeeded.
        repl_passed: Whether SELF:repl succeeded.

    Returns:
        Dict with tool value analysis.
    """
    return {
        "tools_helped": repl_passed and not direct_passed,
        "tools_neutral": repl_passed == direct_passed,
        "tools_hurt": direct_passed and not repl_passed,
        "tool_advantage": int(repl_passed) - int(direct_passed),
    }
