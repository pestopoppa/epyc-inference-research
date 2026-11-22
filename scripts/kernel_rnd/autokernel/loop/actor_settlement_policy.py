"""AKU-06d2: explicit settlement policy for an actor-only operation (no build child).

A source-advice ``actor_preparation`` operation runs planner and critic children and
never a build. When the final invocation of each of those roles is accepted, the
operation produced its advice and settles ``abstained`` (settled work, no measurement
or build claim) instead of ``failed``. ``prerequisite`` stays reserved for a
source build with an accepted build child and its enrollment reference.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

ACTOR_ONLY_OPERATION = "actor_preparation"
ACTOR_ONLY_OUTCOME = "abstained"
ACTOR_ONLY_ROLES = ("planner", "critic")


class ActorSettlementRefused(ValueError):
    pass


def eligible(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Actor-only chain whose planner and critic each finally accepted."""
    if not rows or any(row.get("operation") != ACTOR_ONLY_OPERATION for row in rows):
        return False
    last = {row["role"]: row for row in rows}
    return (tuple(last) == ACTOR_ONLY_ROLES
            and all(row["disposition"] == "accepted" for row in last.values()))


def check(rows, denial_ref, *, enrollment_ref: Any = None, uncertain: int = 0) -> None:
    if denial_ref is not None or enrollment_ref is not None or uncertain or not eligible(rows):
        raise ActorSettlementRefused(
            "abstained settles only an actor-only operation with accepted planner and critic")


def outcome_for(rows: Sequence[Mapping[str, Any]]) -> str:
    """The policy outcome a producer should request for this actor-only prefix."""
    return ACTOR_ONLY_OUTCOME if eligible(rows) else "failed"
