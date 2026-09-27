"""The CPU region claim for model calls that bypass ``LLMPrimitives``.

Two callers talk to a llama-server directly instead of through
``LLMPrimitives._real_call_single``: the escalation prewarmer
(``src/services/escalation_prewarmer.py``, an ``n_predict=0`` prefill) and the
OAB-8 scouts (``src/api/routes/chat_pipeline/scout_stage.py``). The normal call
path claims the target instance's CPU regions before every direct-backend call
(``src/llm_primitives/inference.py``, the ``_per_region_on`` branch):

    resolved = topology_instance_for_port(port)
    lock_role, lock_idx = resolved or (role, 0)
    with cpu_region_lock_for_instance(lock_role, lock_idx, cancel_check=...,
                                      deadline_s=..., request_tag=...):

Without the same claim, a direct call to a CPU-resident server contends unlocked
with AutoKernel CPU measurement windows and with every locked caller on the same
cores (ARCHSWAP 2026-09-27, item A-3). This module resolves the target exactly
the way the normal path does and hands back that same claim; it adds no new
locking mechanism.

Scope:
- A target with no CPU regions (a GPU server's HT-only host lane, an embedder)
  gets no claim: ``cpu_region_lock`` treats an empty region set as a no-op, and
  ``resolve_claim_target`` returns ``None`` for it so callers can tell the cases
  apart and report them.
- With ``ORCHESTRATOR_PER_REGION_LOCKS`` off the normal path uses the legacy
  global ``inference_lock``. Its timeout erases the HOLDER's slots, which an
  optimisation must never trigger, so no direct caller takes it: in legacy mode
  ``resolve_claim_target`` returns ``None`` and the direct callers keep their
  pre-existing behaviour. Production runs with the flag on
  (``orchestrator_stack.py`` defaults it to 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ContextManager, Optional


@dataclass(frozen=True)
class RegionClaimTarget:
    """The physical instance a direct call lands on, and its CPU regions."""

    lock_role: str
    instance_idx: int
    regions: frozenset[str]
    port: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "lock_role": self.lock_role,
            "instance_idx": self.instance_idx,
            "regions": sorted(self.regions),
            "port": self.port,
        }


def per_region_locks_enabled() -> bool:
    """The normal path's own flag check (one definition, not a copy)."""
    from src.llm_primitives.inference import _per_region_locks_enabled

    return _per_region_locks_enabled()


def resolve_claim_target(
    role: str, url: str | None = None, *, port: int | None = None
) -> RegionClaimTarget | None:
    """Resolve the instance the normal path would lock for ``role`` at ``url``/``port``.

    Returns ``None`` when no claim applies: the per-region flag is off, there is no
    endpoint, or the instance owns no CPU regions (GPU / HT-only lane)."""
    if not per_region_locks_enabled():
        return None
    if port is None:
        if not url:
            return None
        from src.llm_primitives.inference import _extract_port

        port = _extract_port(url)
    from src.runtime.instance_topology import get_instance_regions, topology_instance_for_port

    resolved = topology_instance_for_port(port or 0)
    lock_role, lock_idx = resolved or (role, 0)
    regions = get_instance_regions().get((lock_role, lock_idx), frozenset())
    if not regions:
        return None
    return RegionClaimTarget(
        lock_role=str(lock_role), instance_idx=int(lock_idx), regions=frozenset(regions), port=port
    )


def region_claim(
    target: RegionClaimTarget,
    *,
    timeout_s: Optional[float] = None,
    deadline_s: Optional[float] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    request_tag: Optional[str] = None,
) -> ContextManager[Any]:
    """The same exclusive ``cpu_region_lock_for_instance`` claim the normal path takes.

    Raises ``CpuRegionLockTimeout`` on entry when the claim is contended past the
    budget (``timeout_s`` / ``deadline_s``) or ``cancel_check`` trips."""
    from src.runtime.cpu_region_lock import cpu_region_lock_for_instance

    return cpu_region_lock_for_instance(
        target.lock_role,
        target.instance_idx,
        timeout_s=timeout_s,
        deadline_s=deadline_s,
        cancel_check=cancel_check,
        request_tag=request_tag,
    )


__all__ = ["RegionClaimTarget", "per_region_locks_enabled", "region_claim", "resolve_claim_target"]
