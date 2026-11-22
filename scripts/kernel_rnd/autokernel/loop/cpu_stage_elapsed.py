"""CPU stage elapsed-time validation, split verbatim from scheduling.py (code-budget split)."""
from __future__ import annotations
import math
from typing import Any, Mapping


class SchedulingRefused(ValueError):
    """An input or transition cannot be proven within this accounting model."""


def _exact(value: Any, fields: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchedulingRefused(f"{label} must be an object")
    keys = tuple(value.keys())
    if any(not isinstance(key, str) for key in keys):
        raise SchedulingRefused(f"{label} keys must be strings")
    actual = set(keys)
    if actual != fields:
        raise SchedulingRefused(
            f"{label} fields differ: missing={sorted(fields - actual)}, "
            f"unknown={sorted(actual - fields)}")
    return value


def _number(value: Any, label: str, *, minimum: float = 0.0,
            maximum: float | None = None) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or float(value) < minimum
            or (maximum is not None and float(value) > maximum)):
        suffix = f" and <= {maximum}" if maximum is not None else ""
        raise SchedulingRefused(f"{label} must be finite and >= {minimum}{suffix}")
    return float(value)


def validate_cpu_stage_elapsed(value):
    """Original completed CPU scope clock frame; never a physical claim receipt."""
    row = _exact(value, frozenset({"context_id", "domain", "started_at", "ended_at", "completed"}),
                 "CPU stage elapsed")
    domain = _exact(row["domain"], frozenset({"kind", "clock", "pid", "boot_id",
        "process_start_ticks", "error"}), "CPU stage domain")
    context = row["context_id"]
    if (not isinstance(context, str) or len(context) != 64
            or any(char not in "0123456789abcdef" for char in context)
            or row["completed"] is not True or domain["kind"] != "direct_loop"
            or domain["clock"] != "monotonic" or domain["error"] is not None
            or type(domain["pid"]) is not int or domain["pid"] <= 0
            or type(domain["process_start_ticks"]) is not int or domain["process_start_ticks"] <= 0
            or not isinstance(domain["boot_id"], str) or not domain["boot_id"]):
        raise SchedulingRefused("CPU original elapsed scope is unavailable")
    start = _number(row["started_at"], "CPU stage elapsed start")
    end = _number(row["ended_at"], "CPU stage elapsed end")
    if end <= start:
        raise SchedulingRefused("CPU stage elapsed interval must be positive")
    return dict(row, domain=dict(domain), started_at=start, ended_at=end)
