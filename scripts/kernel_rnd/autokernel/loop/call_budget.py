"""OAB-32: a default per-call wall budget for every role and every backend kind.

OAB-23 bounded the opencode planner/author only (`ActorSeat.*_budget_s`, 0 = none). A
critic call, or a call on codex/claude/orchestrator, ran to the hard `timeout_s`, which
the loop reads as a TRANSIENT and retries. A budget instead ends the call through the
stop path and records `budget_exhausted`, an abstention that `_with_backoff` never
retries. This module is the pure policy: the default budget sits under the hard
timeout, a configured budget is never loosened, and a local backend that would run
with no wall bound is refused rather than started.

The decoded-token cap is table-only here: it is enforceable where the backend takes an
output limit (opencode `limit.output`, set from `DEFAULT_OUTPUT_TOKENS`), and for the
other kinds the wall budget is the only enforced bound.
"""
from __future__ import annotations

import os

#: Fraction of the hard per-call timeout a call may spend before it is cut as
#: `budget_exhausted` (the remainder is the grace for the stop path and salvage).
DEFAULT_WALL_FRACTION = 0.9
MIN_WALL_S = 30.0
DISABLE_ENV = "AK_ACTOR_DEFAULT_BUDGET"        # "off" restores the historical behaviour

ROLES = ("planner", "author", "critic")
KINDS = ("codex", "claude", "opencode", "orchestrator")
#: Backends the loop serves from this host: an unbounded call here holds a local
#: server slot and a region claim, so it must never run without a wall bound.
LOCAL_KINDS = frozenset({"opencode", "orchestrator"})
#: Default decoded-token cap per role (opencode `limit.output`; see
#: `actor_opencode_config.DEFAULT_PLANNER_OUTPUT_LIMIT` / OAB-34).
DEFAULT_OUTPUT_TOKENS = {"planner": 32_000, "author": 16_384, "critic": 16_384}


class UnboundedBackendRefused(ValueError):
    """A local backend would run with no wall budget."""


def enabled(env: dict[str, str] | None = None) -> bool:
    return (env if env is not None else os.environ).get(DISABLE_ENV, "").strip().lower() \
        not in ("off", "0", "false", "no")


def default_wall_s(timeout_s: float) -> float:
    """The default budget under a hard timeout: 90 %, never below MIN_WALL_S unless the
    timeout itself is lower (then half of it), and 0 when there is no hard timeout."""
    if not timeout_s or timeout_s <= 0:
        return 0.0
    if timeout_s <= MIN_WALL_S:
        return float(timeout_s) / 2
    return max(MIN_WALL_S, float(timeout_s) * DEFAULT_WALL_FRACTION)


def effective_wall_s(configured: float | None, timeout_s: float, *,
                     env: dict[str, str] | None = None) -> float | None:
    """The wall budget to apply: a configured one wins unchanged; else the default when
    enabled; None means "no budget" (historical path)."""
    if configured:
        return float(configured)
    if not enabled(env):
        return None
    wall = default_wall_s(timeout_s)
    return wall or None


def require_bounded(kind: str, role: str, wall_s: float | None) -> None:
    """Fail closed: a local backend with no wall budget refuses to run."""
    if kind in LOCAL_KINDS and not wall_s:
        raise UnboundedBackendRefused(
            f"{kind} {role} call has no wall budget (timeout_s={wall_s!r}); a local backend "
            "never runs unbounded")
