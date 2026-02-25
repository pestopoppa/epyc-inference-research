"""Embedder precompute and reward injection via HTTP.

Handles fire-and-forget async reward injection using a background
ThreadPoolExecutor, and embedding precomputation via the BGE pool.
"""

from __future__ import annotations

import concurrent.futures
import logging
from typing import Any

__all__ = [
    "EMBEDDER_PORTS",
    "_get_reward_executor",
    "_inject_3way_rewards_http",
    "_inject_single_reward",
    "_precompute_embedding",
]

logger = logging.getLogger("seed_specialist_routing")

# Embedder server ports for precomputing embeddings
EMBEDDER_PORTS = [8090, 8091, 8092, 8093, 8094, 8095]

# Background executor for async reward injection (fire-and-forget)
_reward_executor: concurrent.futures.ThreadPoolExecutor | None = None


def _get_reward_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Get or create the background executor for reward injection."""
    global _reward_executor
    if _reward_executor is None:
        _reward_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="reward_inject"
        )
    return _reward_executor


def _precompute_embedding(
    task_description: str,
    client: "httpx.Client",
) -> list[float] | None:
    """Precompute embedding for task_description using embedder servers.

    Tries each embedder port until one succeeds. Returns None on failure
    (caller will fall back to letting the API compute the embedding).

    Args:
        task_description: Text to embed (will be truncated to 200 chars).
        client: HTTP client for requests.

    Returns:
        List of float embeddings, or None on failure.
    """
    # Build the text the same way q_scorer does
    text = f"type:chat | objective:{task_description[:200]}"

    for port in EMBEDDER_PORTS:
        try:
            resp = client.post(
                f"http://127.0.0.1:{port}/embedding",
                json={"content": text},
                timeout=5.0,
            )
            if resp.status_code != 200:
                continue

            data = resp.json()
            # Parse llama-server response format
            if "embedding" in data:
                embedding_data = data["embedding"]
                if isinstance(embedding_data[0], list):
                    return embedding_data[0]
                return embedding_data
            elif "data" in data and len(data["data"]) > 0:
                return data["data"][0]["embedding"]
        except Exception:
            continue

    logger.debug("All embedder servers failed, will let API compute embedding")
    return None


def _inject_single_reward(
    url: str,
    payload: dict[str, Any],
    action_key: str,
) -> bool:
    """Inject a single reward (runs in background thread)."""
    import httpx
    try:
        with httpx.Client() as client:
            resp = client.post(f"{url}/chat/reward", json=payload, timeout=30)
            if resp.status_code == 200:
                return True
            else:
                logger.debug(f"Reward injection failed for {action_key}: HTTP {resp.status_code}")
                return False
    except Exception as e:
        logger.debug(f"Reward injection error for {action_key}: {e}")
        return False


def _inject_3way_rewards_http(
    prompt: str,
    suite: str,
    question_id: str,
    rewards: dict[str, float],
    metadata: dict[str, Any],
    url: str,
    client: "httpx.Client",
) -> int:
    """Inject 3-way rewards via HTTP API (async, non-blocking).

    Q-values receive binary rewards for faithful probability estimation.
    Cost metrics are stored in context for later Optuna threshold optimization.

    Precomputes the embedding once and reuses it for all reward injections.
    Submissions are fire-and-forget to avoid blocking the eval loop.

    Returns number of rewards submitted (not necessarily injected yet).
    """
    cost_metrics = metadata.get("cost_metrics", {})

    # Precompute embedding once for all reward injections (same task_description)
    embedding = _precompute_embedding(prompt[:200], client)

    executor = _get_reward_executor()
    submitted = 0

    for action_key, reward in rewards.items():
        # Build context with cost metrics for this specific action
        action_cost = cost_metrics.get(action_key, {})
        tokens_generated = int(action_cost.get("tokens_generated", 0) or 0)
        tokens_estimate = int(action_cost.get("tokens_generated_estimate", 0) or 0)

        context = {
            "task_type": suite,
            "source": "3way_eval",
            "question_id": question_id,
            "action_type": "routing",
            # Tool value metadata (flat scalars)
            "tools_helped": metadata.get("tools_helped", False),
            "tools_neutral": metadata.get("tools_neutral", False),
            "tools_hurt": metadata.get("tools_hurt", False),
            "tool_advantage": metadata.get("tool_advantage", 0),
            # Cost metrics for this action (for Optuna later)
            "elapsed_seconds": action_cost.get("elapsed_seconds", 0.0),
            "tokens_generated": tokens_generated,
            "tokens_generated_estimate": tokens_estimate,
            "tokens_generated_effective": (
                tokens_generated if tokens_generated > 0 else tokens_estimate
            ),
            "predicted_tps": action_cost.get("predicted_tps", 0.0),
            "generation_ms": action_cost.get("generation_ms", 0.0),
            "tools_used": action_cost.get("tools_used", 0),
            "backend_task_id": action_cost.get("backend_task_id", 0),
            "slot_progress_source": action_cost.get("slot_progress_source", ""),
        }

        payload = {
            "task_description": prompt[:200],
            "action": action_key,
            "reward": reward,
            "context": context,
        }
        # Include precomputed embedding if available
        if embedding is not None:
            payload["embedding"] = embedding

        # Fire-and-forget: submit to background executor
        executor.submit(_inject_single_reward, url, payload, action_key)
        submitted += 1

    return submitted
