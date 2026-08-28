#!/usr/bin/env python3
"""What fraction of a held GPU claim the loop actually used.

WHY THIS EXISTS
---------------
Across its entire life the loop held the MI210 for **1.403 hours** in total (122
claims, mean hold 41.4 s) while spending **29.0 hours** compiling, single-threaded,
on a 192-thread host. The device was idle for 95.4% of the campaign's wall-clock,
and the campaign held session-level ownership of it for weeks.

Nobody reported that, because the loop reported iterations and receipts. It had no
number for "am I using the thing I am holding". This module is that number, and it
belongs on the same row as every result, so the condition is visible without an
audit.

It measures the apparatus, never a kernel. `under_measurement_load` on the sampled
device state is the busy signal; a claim held while that stays false is a claim that
should have been released.
"""
from __future__ import annotations

from typing import Any, Mapping

UTILIZATION_SCHEMA = "epyc.autokernel.gpu_utilization.v1"


def _seconds_between(start: str | None, end: str | None) -> float | None:
    from datetime import datetime
    if not isinstance(start, str) or not isinstance(end, str):
        return None
    try:
        began = datetime.fromisoformat(start.replace("Z", "+00:00"))
        ended = datetime.fromisoformat(end.replace("Z", "+00:00"))
    except ValueError:
        return None
    seconds = (ended - began).total_seconds()
    return seconds if seconds >= 0 else None


def from_sampling(sampling: Mapping[str, Any] | None, *,
                  claim_acquired_at: str | None = None,
                  window_ended_at: str | None = None) -> dict[str, Any]:
    """Utilization for one screen, from its own device trace.

    `device_seconds_under_load` is sample-count-derived: the sampler runs at a fixed
    cadence and refuses a trace whose gaps exceed twice the declared interval, so
    counting busy samples and multiplying by the interval is sound rather than a
    convenient approximation.
    """
    payload: dict[str, Any] = {"schema": UTILIZATION_SCHEMA}
    samples = (sampling or {}).get("samples")
    interval = (sampling or {}).get("interval_s")
    duration = (sampling or {}).get("duration_s")

    if isinstance(samples, list) and samples and isinstance(interval, (int, float)):
        under_load = sum(1 for row in samples
                         if isinstance(row, Mapping) and row.get("under_measurement_load"))
        payload["device_samples"] = len(samples)
        payload["device_samples_under_load"] = under_load
        payload["device_seconds_sampled"] = (
            float(duration) if isinstance(duration, (int, float)) else
            len(samples) * float(interval))
        payload["device_seconds_under_load"] = under_load * float(interval)
        payload["sampled_busy_fraction"] = under_load / len(samples)
    else:
        # An absent trace is reported as absent, never as zero utilization -- a
        # missing measurement is not evidence of an idle device.
        payload["device_samples"] = 0
        payload["device_seconds_sampled"] = None
        payload["device_seconds_under_load"] = None
        payload["sampled_busy_fraction"] = None

    held = _seconds_between(claim_acquired_at, window_ended_at)
    payload["claim_held_s"] = held
    busy = payload.get("device_seconds_under_load")
    if held and held > 0 and isinstance(busy, (int, float)):
        payload["gpu_seconds_idle_while_claimed"] = max(0.0, held - busy)
        payload["idle_fraction_while_claimed"] = max(0.0, 1.0 - busy / held)
    else:
        payload["gpu_seconds_idle_while_claimed"] = None
        payload["idle_fraction_while_claimed"] = None
    return payload


def summarise(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Campaign-level utilization, for the weekly scoreboard.

    The lifetime figure that made the condition legible was a ratio -- 1.4 hours of
    GPU held against 29.0 hours of compiling -- so this reports totals, not an
    average of fractions, which would weight a 40-second claim the same as an hour.
    """
    held = [row.get("claim_held_s") for row in rows]
    busy = [row.get("device_seconds_under_load") for row in rows]
    total_held = sum(value for value in held if isinstance(value, (int, float)))
    total_busy = sum(value for value in busy if isinstance(value, (int, float)))
    return {
        "schema": UTILIZATION_SCHEMA,
        "screens": len(rows),
        "total_claim_held_s": total_held,
        "total_device_seconds_under_load": total_busy,
        "total_gpu_seconds_idle_while_claimed": max(0.0, total_held - total_busy),
        "idle_fraction_while_claimed": (
            max(0.0, 1.0 - total_busy / total_held) if total_held > 0 else None),
    }


__all__ = ["UTILIZATION_SCHEMA", "from_sampling", "summarise"]
