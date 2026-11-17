"""S3-AKU-04: 2-D (effect, spread) keep frontier.

A keep record carries BOTH coordinates, and retention keeps every non-dominated
candidate instead of the single best scalar. "Reduces variance" is a legitimate keep
(program.md section 3.3), so a candidate that is slightly slower but much tighter must not
be discarded by a one-number comparison against a faster, noisier one.

Dominance is NOT reimplemented here: it is the orchestrator's single authoritative
`pareto_math.dominates` (max-objectives), loaded by explicit path like the other
cross-repo primitives. Objectives are (effect, -spread): larger effect is better,
smaller spread is better. Spread is any non-negative dispersion measure the caller
uses consistently (CI half-width in effect units, IQR, ...).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
from typing import Iterable, Sequence

SCHEMA = "epyc.autokernel.keep_frontier.v1"


def _load_dominates():
    root = Path(os.environ.get("EPYC_ORCHESTRATOR_ROOT",
                               "/mnt/raid0/llm/epyc-orchestrator")).resolve()
    if not (root / "src" / "autopilot_core" / "pareto_math.py").is_file():
        raise RuntimeError("authoritative pareto_math.py not found; set "
                           "EPYC_ORCHESTRATOR_ROOT to the orchestrator checkout")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.autopilot_core.pareto_math import dominates  # pylint: disable=import-outside-toplevel
    return dominates


@dataclass(frozen=True)
class KeepPoint:
    candidate_id: str
    effect: float
    spread: float

    def objectives(self) -> tuple[float, float]:
        return (self.effect, -self.spread)

    def record(self) -> dict:
        """The two coordinates as they ride on a keep record."""
        return {"schema": SCHEMA, "candidate_id": self.candidate_id,
                "effect": self.effect, "spread": self.spread}


def make_point(candidate_id: str, effect, spread) -> KeepPoint:
    """Validate and build a point. A missing/NaN/negative coordinate REFUSES: a keep
    without a spread cannot be placed on the frontier, and guessing one would let an
    unmeasured candidate dominate a measured one."""
    for name, v in (("effect", effect), ("spread", spread)):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"keep {candidate_id!r}: {name} must be a finite number, got {v!r}")
    if spread < 0:
        raise ValueError(f"keep {candidate_id!r}: spread must be >= 0, got {spread!r}")
    return KeepPoint(str(candidate_id), float(effect), float(spread))


def frontier(points: Iterable[KeepPoint]) -> list[KeepPoint]:
    """Non-dominated subset, input order preserved. Exact duplicates (same effect and
    spread) do not dominate each other and are all retained; ids must be unique."""
    pts = list(points)
    ids = [p.candidate_id for p in pts]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate candidate_id in keep frontier input")
    dominates = _load_dominates()
    return [p for p in pts
            if not any(dominates(q.objectives(), p.objectives()) for q in pts if q is not p)]


def retain(existing: Sequence[KeepPoint], new: KeepPoint) -> tuple[list[KeepPoint], list[KeepPoint]]:
    """Add one candidate. Returns (frontier_after, evicted). A candidate dominated by a
    member is not admitted (it is returned in neither list)."""
    before = list(existing)
    after = frontier([*before, new])
    evicted = [p for p in before if p not in after]
    return after, evicted
