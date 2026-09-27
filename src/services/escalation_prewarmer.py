"""Speculative prefill pre-warming for architect model servers.

When classify_task_complexity() returns COMPLEX at request ingestion, fires
a non-blocking n_predict=0, cache_prompt=true request to the architect server
with the system prompt prefix. This warms the KV cache before escalation
actually happens.

Saving: ~500 tokens of system prompt prefix at 1.2 t/s = 417ms per architect
escalation that hits the pre-warmed slot.

Risk: Medium. Pre-warm occupies a slot. We check /slots before pre-warming
to avoid evicting hot slots.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from src.registry.stack_priors import (
    live_stack_role_records,
    stack_prior_endpoint_port,
    stack_prior_serving,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STACK_PRIORS_PATH = PROJECT_ROOT / "orchestration" / "derived" / "stack_priors.yaml"

# Explicit degraded fallbacks. Normal operation derives architect endpoint and
# model hint from generated stack priors so model/port swaps update prewarming
# without touching this module.
# 2026-09-27 ARCHITECT SWAP (operator-decided): architect_general = Qwen3.8-Flash-Next
# on the full CPU instance (:8074); architect_critic = Qwen3.8-27B Q8 on the MI210
# (:8083). The prewarm target is the ROLE the graph escalates to (architect_general),
# so it now prefills the CPU :8074 process. (The 8083 -> "Qwen3.5-122B-A10B" hint that
# stood here had been stale since the 2026-08-01 W1 cutover.)
_DEGRADED_ARCHITECT_PORTS = {
    "architect_general": 8074,
    "architect_critic": 8083,
}

_DEGRADED_ARCHITECT_PORT_MODEL_HINT = {
    8074: "Qwen3.8-Flash-Next-UD-IQ4_XS",
    8083: "Qwen3.8-27B-Q8_0",
}

# System prompt prefix used across architect roles (warm this into KV cache)
ARCHITECT_SYSTEM_PREFIX = (
    "You are a senior software architect. Analyze the task, identify components, "
    "assess complexity, and provide a structured response. Your role is to reason "
    "about architecture, design patterns, and system-level concerns.\n\n"
)


def architect_ports_from_stack_priors(
    stack_priors_path: Path = DEFAULT_STACK_PRIORS_PATH,
) -> dict[str, int]:
    """Return live architect role -> endpoint port from generated stack priors."""
    ports: dict[str, int] = {}
    for role, record in live_stack_role_records(stack_priors_path).items():
        if not role.startswith("architect_"):
            continue
        try:
            port = stack_prior_endpoint_port(stack_prior_serving(record))
        except ValueError:
            logger.debug("Skipping malformed architect endpoint in stack priors for %s", role)
            port = None
        if port is not None:
            ports[role] = port
    return dict(sorted(ports.items()))


def architect_port_for_role(
    role: str = "architect_general",
    stack_priors_path: Path = DEFAULT_STACK_PRIORS_PATH,
) -> int:
    """Return the architect prewarm port, falling back only in degraded mode."""
    ports = architect_ports_from_stack_priors(stack_priors_path)
    return ports.get(role) or _DEGRADED_ARCHITECT_PORTS.get(role, 8083)


def architect_port_model_hints_from_stack_priors(
    stack_priors_path: Path = DEFAULT_STACK_PRIORS_PATH,
) -> dict[int, str]:
    """Return live architect endpoint port -> display/model hint from stack priors."""
    hints: dict[int, str] = {}
    for role, record in live_stack_role_records(stack_priors_path).items():
        if not role.startswith("architect_"):
            continue
        try:
            port = stack_prior_endpoint_port(stack_prior_serving(record))
        except ValueError:
            logger.debug("Skipping malformed architect endpoint in stack priors for %s", role)
            continue
        if port is None:
            continue
        display_name = record.get("display_name")
        model_id = record.get("model_id")
        hint = display_name if isinstance(display_name, str) and display_name else model_id
        if isinstance(hint, str) and hint:
            hints[port] = hint
    return dict(sorted(hints.items()))


def architect_model_hint_for_port(
    port: int,
    stack_priors_path: Path = DEFAULT_STACK_PRIORS_PATH,
) -> str:
    """Return model hint for an architect port, falling back only in degraded mode."""
    return (
        architect_port_model_hints_from_stack_priors(stack_priors_path).get(port)
        or _DEGRADED_ARCHITECT_PORT_MODEL_HINT.get(port, "")
    )


def __getattr__(name: str) -> Any:
    """Preserve legacy architect prewarm constants without stale snapshots."""
    if name == "ARCHITECT_PORTS":
        return dict(_DEGRADED_ARCHITECT_PORTS)
    if name == "ARCHITECT_PORT_MODEL_HINT":
        return dict(_DEGRADED_ARCHITECT_PORT_MODEL_HINT)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class EscalationPrewarmer:
    """Speculatively pre-warms architect KV cache for complex tasks.

    Usage:
        prewarmer = EscalationPrewarmer()
        # At turn 1, fire and forget:
        asyncio.create_task(prewarmer.prewarm_if_complex(objective, complexity))
    """

    def __init__(
        self,
        timeout: float = 10.0,
        stack_priors_path: Path = DEFAULT_STACK_PRIORS_PATH,
    ):
        self._timeout = timeout
        self._stack_priors_path = stack_priors_path
        self._prewarm_count = 0
        self._prewarm_hits = 0  # Incremented externally when prewarm slot is used
        self._prewarm_by_port: dict[int, int] = {}
        self._prewarm_hits_by_role: dict[str, int] = {}
        self._lock = threading.Lock()

    async def prewarm_if_complex(
        self,
        objective: str,
        complexity_level: str,
        target_port: int | None = None,
    ) -> bool:
        """Pre-warm architect slot if task is complex.

        Args:
            objective: Task description.
            complexity_level: Result of classify_task_complexity() (as string).
            target_port: Override architect port (default: architect_general).

        Returns:
            True if pre-warm was sent successfully.
        """
        if complexity_level not in ("COMPLEX",):
            return False

        port = target_port or architect_port_for_role(
            "architect_general",
            self._stack_priors_path,
        )

        # Check if slot is available before pre-warming
        slot_available = await self._check_slot_available(port)
        if not slot_available:
            logger.debug("Pre-warm skipped: no idle slot on port %d", port)
            return False

        # Send non-blocking prefill request
        return await self._send_prewarm(port, objective)

    async def _check_slot_available(self, port: int) -> bool:
        """Check /slots endpoint for an idle slot."""
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"http://localhost:{port}/slots",
                    timeout=self._timeout,
                )
                if resp.status_code != 200:
                    return False
                data = resp.json()
                # llama-server returns a list for -np>1, a single dict for -np=1
                slots = data if isinstance(data, list) else [data]
                # Check is_processing (modern llama-server) or state==0 (legacy)
                idle_slots = [
                    s for s in slots
                    if not s.get("is_processing", True) or s.get("state") == 0
                ]
                return len(idle_slots) > 0
        except Exception as e:
            logger.debug("Slot check failed for port %d: %s", port, e)
            return False

    async def _send_prewarm(self, port: int, objective: str) -> bool:
        """Send a n_predict=0, cache_prompt=true request to warm KV cache."""
        # Build the prefix that will be shared with the actual escalation.
        # Apply the same chat template the real escalation will use so the
        # KV cache prefix matches end-to-end. Falls back to raw text on
        # unknown ports (defensive — same shape as the prior behavior).
        from src.api.routes.chat_utils import apply_chat_template_for_model
        body = ARCHITECT_SYSTEM_PREFIX + objective[:2000]
        model_hint = architect_model_hint_for_port(port, self._stack_priors_path)
        prewarm_prompt = apply_chat_template_for_model(model_hint, body) if model_hint else body

        payload = {
            "prompt": prewarm_prompt,
            "n_predict": 0,  # No generation, just prefill
            "cache_prompt": True,  # Cache the KV
        }

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"http://localhost:{port}/completion",
                    json=payload,
                    timeout=self._timeout,
                )
                if resp.status_code == 200:
                    with self._lock:
                        self._prewarm_count += 1
                        self._prewarm_by_port[port] = self._prewarm_by_port.get(port, 0) + 1
                    logger.info(
                        "Pre-warmed architect slot on port %d (prompt: %d chars)",
                        port,
                        len(prewarm_prompt),
                    )
                    return True
                logger.debug("Pre-warm failed: HTTP %d", resp.status_code)
                return False
        except Exception as e:
            logger.debug("Pre-warm request failed for port %d: %s", port, e)
            return False

    def record_prewarm_hit(self, role: str) -> None:
        """Record that execution actually escalated to an architect role."""
        role_key = str(role or "").strip() or "unknown"
        with self._lock:
            self._prewarm_hits += 1
            self._prewarm_hits_by_role[role_key] = (
                self._prewarm_hits_by_role.get(role_key, 0) + 1
            )

    def get_stats(self) -> dict[str, Any]:
        """Get pre-warming statistics."""
        with self._lock:
            prewarm_count = self._prewarm_count
            prewarm_hits = self._prewarm_hits
            prewarm_by_port = dict(self._prewarm_by_port)
            prewarm_hits_by_role = dict(self._prewarm_hits_by_role)
        return {
            "prewarm_count": prewarm_count,
            "prewarm_hits": prewarm_hits,
            "prewarm_by_port": prewarm_by_port,
            "prewarm_hits_by_role": prewarm_hits_by_role,
            "hit_rate": (
                prewarm_hits / prewarm_count
                if prewarm_count > 0
                else 0.0
            ),
        }


_shared_prewarmer: EscalationPrewarmer | None = None
_shared_lock = threading.Lock()


def get_shared_prewarmer() -> EscalationPrewarmer:
    """Return the process-wide prewarmer used by orchestration helpers."""
    global _shared_prewarmer
    if _shared_prewarmer is None:
        with _shared_lock:
            if _shared_prewarmer is None:
                _shared_prewarmer = EscalationPrewarmer()
    return _shared_prewarmer
