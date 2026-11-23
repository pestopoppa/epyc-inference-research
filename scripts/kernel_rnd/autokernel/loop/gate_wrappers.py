"""One list of gate features for every path that runs the loop (AK-TRIM-GATE-PLUMBING).

`GateWrappers` carries each optional gate feature once; run.py builds it, pool.drive and
pipeline.run_pool pass it through unchanged, and `loop.iterate` applies `layers()` to its
gate. A new feature is a field here plus one entry in `layers()`, not a kwarg per module.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

from . import exact_lane, isa_claims


@dataclass(frozen=True)
class GateWrappers:
    #: INF06-EXACT-LANE: `make_exact_judge(worker) -> judge`; the lane dispatch is always on.
    make_exact_judge: Callable | None = None
    #: C5-ISA-1: `make_isa_check(worker) -> check`, composed after the gate (GPU only).
    make_isa_check: Callable | None = None
    #: AK-AMDAHL-PRUNE: `tail_demote(hypothesis) -> bool` orders the serialized tail.
    tail_demote: Callable | None = None

    def layers(self, worker=None) -> tuple[Callable, ...]:
        """`gate -> gate` wrappers in application order (first applied is innermost)."""
        judge, check = (make(worker) if make else None
                        for make in (self.make_exact_judge, self.make_isa_check))
        return (partial(exact_lane.lane_gate, exact_judge=judge, on_mismatch="reject"),
                *([partial(isa_claims.wrap_gate, check=check)] if check else ()))
