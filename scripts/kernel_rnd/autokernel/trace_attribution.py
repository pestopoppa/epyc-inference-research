"""GPU-active union versus summed kernel time, plus a cross-run subtraction lint (AK-PROF-ATTR-1, intake-2562#record).

Overlapping kernel work is not additive: on a trace where kernels overlap (multiple streams, graph replay), the sum of
kernel durations exceeds the time the device was busy. This module reports, per trace:

- ``sum_ns``: summed kernel durations (what ``kernel_wall_share`` style tables divide by today);
- ``active_union_ns``: length of the union of [start, end) intervals, i.e. device-active time;
- ``overlap_ns``: ``sum_ns - active_union_ns`` (0 on a strictly serial trace);
- ``span_ns``: first start to last end of the trace window;
- ``idle_gap_ns``: ``span_ns - active_union_ns``, the in-window time with no kernel running (launch gaps, host stalls).

On a single device the active union is the length of the device's critical path through the captured window.

It also refuses a derived "unattributed" figure built by subtracting a GPU-time total from an engine/benchmark wall time
that came from a different run. Different profiler and benchmark runs do not share a window, so the difference is not
attributable. ``unattributed_ns`` only computes when both sides carry the same ``run_id``.

The module is diagnostic and additive. It does not change ``profile_report`` output or any report hash.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


class CrossRunSubtractionError(ValueError):
    """Raised when an unattributed-time figure would subtract quantities measured in different runs."""


@dataclass(frozen=True)
class TraceAttribution:
    run_id: str
    n_dispatches: int
    sum_ns: int
    active_union_ns: int
    overlap_ns: int
    span_ns: int
    idle_gap_ns: int

    def as_dict(self) -> dict[str, object]:
        return {"run_id": self.run_id, "n_dispatches": self.n_dispatches, "sum_ns": self.sum_ns,
                "active_union_ns": self.active_union_ns, "overlap_ns": self.overlap_ns, "span_ns": self.span_ns,
                "idle_gap_ns": self.idle_gap_ns, "additive": self.overlap_ns == 0}


def _intervals(dispatches: Iterable[object]) -> list[tuple[int, int]]:
    out = []
    for d in dispatches:
        if isinstance(d, Mapping):
            s, e = int(d["start_ns"]), int(d["end_ns"])
        else:
            s, e = int(getattr(d, "start_ns")), int(getattr(d, "end_ns"))
        if e < s:
            raise ValueError(f"dispatch ends before it starts: [{s}, {e})")
        out.append((s, e))
    return out


def union_length(intervals: Sequence[tuple[int, int]]) -> int:
    total = 0
    cur_s = cur_e = None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        elif e > cur_e:
            cur_e = e
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def attribute(dispatches: Iterable[object], run_id: str) -> TraceAttribution:
    """Summed duration, active union and span for one trace (dispatches: ProfileDispatch or mappings with start/end ns)."""
    iv = _intervals(dispatches)
    if not iv:
        return TraceAttribution(run_id, 0, 0, 0, 0, 0, 0)
    total = sum(e - s for s, e in iv)
    union = union_length(iv)
    span = max(e for _, e in iv) - min(s for s, _ in iv)
    return TraceAttribution(run_id, len(iv), total, union, total - union, span, span - union)


def unattributed_ns(engine_wall_ns: int, engine_run_id: str, trace: TraceAttribution) -> int:
    """Engine wall time minus GPU-active union, only when both come from the SAME run; otherwise refuse."""
    if not engine_run_id or engine_run_id != trace.run_id:
        raise CrossRunSubtractionError(
            f"refusing unattributed time: engine run {engine_run_id!r} != trace run {trace.run_id!r}; "
            "do not subtract different profiler and benchmark runs")
    return int(engine_wall_ns) - trace.active_union_ns
