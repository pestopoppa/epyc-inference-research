"""Actor budget extraction helpers, split verbatim from actors.py (code-budget split)."""
from __future__ import annotations
import json
from typing import Any, Mapping


def _budgets_of(env: Mapping[str, str] | None, budget_s: float | None,
                budget_exhausted: bool, opencode_stats: Mapping[str, Any] | None
                ) -> dict[str, Any] | None:
    """OAB-22/23 on the metrics row: the limits the per-call config applied (from
    `SEAT_ENV_BUDGETS`), the wall budget, whether it ran out, and -- when the export was
    readable -- whether compaction ran and how many steps hit the output cap. None for a
    call that applied none of them (historical rows unchanged)."""
    applied: dict[str, Any] = {}
    raw = (env or {}).get(SEAT_ENV_BUDGETS)
    if raw:
        try:
            applied = dict(json.loads(raw))
        except (ValueError, TypeError) as exc:
            applied = {"error": f"{type(exc).__name__}: {exc}"[:200]}
    if not applied and not budget_s:
        return None
    totals = (opencode_stats or {}).get("totals") or None
    return {
        "context_limit": applied.get("context_limit", 0),
        "output_limit": applied.get("output_limit", 0),
        "concise": bool(applied.get("concise", False)),
        # OAB-24: the reasoning switch the call's config applied ("default" = none).
        "thinking": applied.get("thinking", "default"),
        "action_rule": bool(applied.get("action_rule", False)),
        # Planner reasoning-history drop: only on rows whose call applied it.
        **({"reasoning_history": applied["reasoning_history"]}
           if "reasoning_history" in applied else {}),
        **({"error": applied["error"]} if "error" in applied else {}),
        "budget_s": budget_s or 0,
        "budget_exhausted": budget_exhausted,
        # None when the export was unreadable: unknown, never "did not happen".
        "compacted": (totals.get("compactions", 0) > 0) if totals else None,
        "compactions": totals.get("compactions") if totals else None,
        "output_capped_steps": totals.get("output_capped_steps") if totals else None,
        "context_max_tokens": (opencode_stats or {}).get("context_max_tokens"),
    }


#: OAB-22/23: the limits a call's per-call config applied, as JSON, for the metrics row.
SEAT_ENV_BUDGETS = "AK_ACTOR_SEAT_BUDGETS"
