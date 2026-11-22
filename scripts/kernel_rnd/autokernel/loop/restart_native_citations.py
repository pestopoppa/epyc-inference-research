"""AKU-07h3: a restart settlement cites native records the first run already wrote.

A crash after native ingestion but before driver settlement leaves durable native
records for the transition. Restart settles the ORIGINAL transition as ``failed``
(never a measurement), but the settlement must not hide those records: each one is
cited as a diagnostic ``native-retained:<measurement_id>`` reference. The ids are
found from the controller's durable native index by the transition's lineage id.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

PREFIX = "native-retained:"


def _lineage(payload: Any) -> Any:
    if not isinstance(payload, Mapping):
        return None
    carrier = payload.get("carrier")
    if isinstance(carrier, Mapping) and "lineage_id" in carrier:
        return carrier["lineage_id"]
    return payload.get("lineage_id")


def retained_ids(controller, transition_id: str) -> tuple[str, ...]:
    """Sorted measurement ids of durable native records bound to this transition."""
    lineage = f"driver:{transition_id}"
    with controller._mutex:
        records = dict(controller._native_records)
    return tuple(sorted(measurement_id for measurement_id, entry in records.items()
                        if _lineage(getattr(entry, "payload", None)) == lineage))


def refs(controller, attempt) -> list[str]:
    """Diagnostic citations for a recovered failed attempt; none for a live failure."""
    if not getattr(attempt, "recovered", False):
        return []
    return [PREFIX + item for item in retained_ids(controller, attempt.transition_id)]


def is_retained_ref(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(PREFIX) and len(value) > len(PREFIX)
