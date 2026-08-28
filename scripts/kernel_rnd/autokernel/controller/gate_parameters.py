#!/usr/bin/env python3
"""Every decision parameter, recorded beside the result it decided.

WHY THIS EXISTS
---------------
The acceptance gate MOVED, and that is what produced every `keep` in the project's
history. From the campaign records:

    r29  anchor_drift 0.1527   drift_bound 0.0308   keep=False   blocks=10
    r30  anchor_drift 0.1879   drift_bound 0.0308   keep=False   blocks=10
    r31  anchor_drift 0.1406   drift_bound 0.0308   keep=False   blocks=10
    r32  anchor_drift 0.2010   drift_bound 0.0308   keep=False   blocks=10
    r37  anchor_drift 0.1104   drift_bound 0.0308   keep=False   blocks=15
    ---- drift_bound 0.0308 -> 0.1850  (6.0x) ----
    r38  anchor_drift 0.0708   drift_bound 0.1850   keep=TRUE    blocks=15
    r39  anchor_drift 0.0927   drift_bound 0.1850   keep=TRUE    blocks=15
    r40  anchor_drift 0.1100   drift_bound 0.1850   keep=TRUE    blocks=15
    r41  anchor_drift 0.1585   drift_bound 0.1850   keep=TRUE    blocks=15
    r43  anchor_drift 0.1504   drift_bound 0.1850   keep=TRUE    blocks=15

`r41` and `r43` were accepted at drift EXCEEDING `r29`'s, which had been rejected.
The gate moved, not the candidates. Nothing in 107,707 lines of tests caught it,
because no test and no record ever compared a run's thresholds to the previous run's.

So: snapshot the parameters on every result, and diff them against the last one. A
widening is then visible in the record the day it happens, next to the results it
changed, rather than reconstructable only by an audit months later.

This module DECIDES NOTHING. It reports. Changing a threshold stays a legitimate act
-- calibration genuinely moves -- it just stops being a silent one.
"""
from __future__ import annotations

from typing import Any, Mapping

GATE_SCHEMA = "epyc.autokernel.gate_parameters.v1"

#: Every parameter that can change whether a candidate is accepted. A parameter that
#: gates a decision and is not listed here is invisible to the diff, which is the
#: defect this module exists to close -- so add to this list when you add a gate.
GATE_KEYS: tuple[str, ...] = (
    "nomination_threshold",
    "continuation_floor_pct",
    "nomination_floor_pct",
    "min_replication_effect_pct",
    "required_replications",
    "max_replication_spread_pct",
    "max_distinct_candidates",
    "sign_policy",
    "conflict_policy",
    "terminal_rule",
    "effect_unit",
    "metric",
)

#: Loosening these admits candidates that would previously have been refused. A
#: change here is reported as a WIDENING, which is the direction that produced every
#: historical keep.
_HIGHER_IS_LOOSER = frozenset({"max_replication_spread_pct", "max_distinct_candidates"})
_LOWER_IS_LOOSER = frozenset({
    "nomination_threshold", "continuation_floor_pct", "nomination_floor_pct",
    "min_replication_effect_pct", "required_replications",
})


def snapshot(*, nomination_threshold: float,
             decision_policy: Mapping[str, Any] | None) -> dict[str, Any]:
    """The gate, as it stood for one decision."""
    values: dict[str, Any] = {"nomination_threshold": float(nomination_threshold)}
    if decision_policy is not None:
        for key in GATE_KEYS:
            if key == "nomination_threshold":
                continue
            if key in decision_policy:
                values[key] = decision_policy[key]
    return {"schema": GATE_SCHEMA, "values": values}


def _direction(key: str, before: Any, after: Any) -> str:
    if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
        return "changed"
    if isinstance(before, bool) or isinstance(after, bool):
        return "changed"
    if after == before:
        return "unchanged"
    looser = ((key in _LOWER_IS_LOOSER and after < before)
              or (key in _HIGHER_IS_LOOSER and after > before))
    return "WIDENED" if looser else "tightened"


def diff(previous: Mapping[str, Any] | None,
         current: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Parameter changes between two consecutive decisions.

    A widening is labelled as such and carries its magnitude, because "6.0x" is the
    number that makes an r38 keep legible next to an r29 refusal.
    """
    if not isinstance(current, Mapping):
        return []
    # No predecessor is not the same as a predecessor with no parameters. The first
    # decision of a campaign has nothing to have moved AWAY from, so it reports no
    # change; an actual empty predecessor would report every parameter as introduced.
    if previous is None:
        return []
    current_values = current.get("values") if "values" in current else current
    previous_values = (previous.get("values") if "values" in previous else previous)
    if not isinstance(current_values, Mapping):
        return []
    if not isinstance(previous_values, Mapping):
        previous_values = {}

    changes: list[dict[str, Any]] = []
    for key in sorted(set(current_values) | set(previous_values)):
        before = previous_values.get(key)
        after = current_values.get(key)
        if before == after:
            continue
        change: dict[str, Any] = {
            "parameter": key, "before": before, "after": after,
            "direction": _direction(key, before, after),
        }
        if (isinstance(before, (int, float)) and not isinstance(before, bool)
                and isinstance(after, (int, float)) and not isinstance(after, bool)
                and before):
            change["ratio"] = after / before
        changes.append(change)
    return changes


def widenings(changes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [change for change in changes if change.get("direction") == "WIDENED"]


__all__ = ["GATE_KEYS", "GATE_SCHEMA", "diff", "snapshot", "widenings"]
