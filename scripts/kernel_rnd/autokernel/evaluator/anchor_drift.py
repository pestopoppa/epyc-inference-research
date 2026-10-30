"""Unchanged arithmetic over the original consecutive A/A observations."""
from __future__ import annotations

from typing import Sequence


def adjacent_relative_steps(series: Sequence[float]) -> tuple:
    """The relative change between each pair of CONSECUTIVE runs.

    Adjacent, not against the first run, because the quantity a paired design
    is exposed to is ONE step: within a block the anchor and the candidate are
    neighbours in time, so the drift the pairing fails to difference out is the
    drift between two adjacent measurements, not the drift across the run.
    """
    values = [float(v) for v in series]
    if len(values) < 2:
        raise ValueError("a drift bound needs at least two consecutive observations")
    return tuple((values[i + 1] - values[i]) / values[i] for i in range(len(values) - 1))

def drift_bound_from(series: Sequence[float]) -> float:
    """The largest single-step relative move an A/A of IDENTICAL code produced.

    This is the number a candidate's median relative gain has to beat. It is
    computed here from the recorded series rather than typed, so re-running the
    A/A and updating `AA_*_RUNS` updates the rule — and so that no one can
    quietly loosen the bound without changing a measurement.
    """
    return max(abs(step) for step in adjacent_relative_steps(series))
