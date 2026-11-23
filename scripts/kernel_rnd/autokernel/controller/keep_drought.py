#!/usr/bin/env python3
"""R24-7 keep-drought trigger: pure detection, advisory only, never a gate.

The trigger decides WHEN a clean-context diagnostician call would be worth one
actor call. It reads measured candidates in recorded order and fires when K
consecutive measured candidates fail to keep (K is pre-registered here) or the
window-best effect stays inside the floor. It never blocks, ranks or admits a
candidate, and it makes no actor call: the actor call, and the plan artifact it
writes into `hypotheses/inbox/`, stay behind the R24-7 exit criterion, which is
a RETRODICTION on the real run-18 and run-24 stores (`retrodict`). Until that
passes on real rows this module is detection plus the pre-registered contract.

Exit criterion (from the box): fire within the first 20 non-keeps of the drought
and name the anchor defect or the surface anti-correlation. `retrodict` checks
the first half mechanically; the naming half needs the diagnostician and the
real stores, and is recorded as OPEN in the result, never faked.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA = "epyc.autokernel.keep_drought.v1"

#: ADVISORY ONLY (AutoKernel ruling, R24-7). Nothing here may return a blocking verdict,
#: refuse or reorder a candidate, or alter a gate decision. A firing is a hint that a
#: diagnostician call may be worth making; the rows passed in are never mutated and the
#: result carries no verdict field a gate could read. `ADVISORY_ONLY` is asserted by tests.
ADVISORY_ONLY = True
#: Result keys a gate would read; none of these may ever appear on a Fired/retrodict result.
FORBIDDEN_VERDICT_KEYS = frozenset({"block", "blocked", "blocking", "verdict", "gate",
                                    "admit", "refuse", "reject", "deny", "halt", "stop"})

#: Pre-registered K. Chosen below the 20-non-keep retrodiction bound with margin.
DEFAULT_K = 10
RETRODICTION_BOUND = 20

#: Statuses that are a measurement of a candidate (mirrors experiments.py:527).
MEASURED = ("kept", "measured_null", "regression", "keep_candidate")

#: The diagnosis checklist the plan artifact must be written against.
CHECKLIST = ("repetition", "wrong_bottleneck", "instrument_or_anchor_broken",
             "surface_anti_correlated_with_production", "exhausted_rung")

PLAN_REQUIRED_KEYS = ("schema", "trigger", "diagnosis", "ranked_actions", "do_not_try")


@dataclass(frozen=True)
class Fired:
    index: int            # position in the MEASURED subsequence (0-based)
    non_keeps: int        # non-keeps seen since the last keep, inclusive
    reason: str           # "consecutive_non_keeps" | "window_best_inside_floor"
    advisory: bool = True  # never a gate


def _measured(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [r for r in rows if r.get("status") in MEASURED]


def _effect_pct(row: Mapping[str, Any]) -> float | None:
    value = row.get("effect_fraction")
    return None if value is None else float(value) * 100.0


def detect(rows: Sequence[Mapping[str, Any]], *, k: int = DEFAULT_K) -> Fired | None:
    """First firing point over rows in recorded order, or None.

    A keep resets the drought. Rows with an unknown effect count toward the
    consecutive-non-keep rule but never toward the window-best rule (a missing
    measurement is not evidence of a null).
    """
    if k < 2:
        raise ValueError("k must be >= 2")
    run = 0
    window: list[tuple[float | None, float | None]] = []   # (|effect_pct|, floor_pct)
    for index, row in enumerate(_measured(rows)):
        if row.get("status") == "kept":
            run, window = 0, []
            continue
        run += 1
        effect = _effect_pct(row)
        window.append((None if effect is None else abs(effect), row.get("floor_pct")))
        window = window[-k:]
        if run >= k:
            return Fired(index, run, "consecutive_non_keeps")
    return None


def detect_window_best(rows: Sequence[Mapping[str, Any]], *, k: int = DEFAULT_K) -> Fired | None:
    """Second trigger: the best effect among the last K measured stays inside the floor."""
    if k < 2:
        raise ValueError("k must be >= 2")
    window: list[tuple[float, float]] = []
    non_keeps = 0
    for index, row in enumerate(_measured(rows)):
        if row.get("status") == "kept":
            window, non_keeps = [], 0
            continue
        non_keeps += 1
        effect, floor = _effect_pct(row), row.get("floor_pct")
        if effect is None or floor is None:
            window = []           # an unmeasured step breaks the evidence window
            continue
        window = (window + [(effect, float(floor))])[-k:]
        if len(window) == k and max(e for e, _ in window) <= min(f for _, f in window):
            return Fired(index, non_keeps, "window_best_inside_floor")
    return None


def first_firing(rows: Sequence[Mapping[str, Any]], *, k: int = DEFAULT_K) -> Fired | None:
    hits = [h for h in (detect(rows, k=k), detect_window_best(rows, k=k)) if h]
    return min(hits, key=lambda h: h.index) if hits else None


def retrodict(rows: Sequence[Mapping[str, Any]], *, k: int = DEFAULT_K,
              bound: int = RETRODICTION_BOUND) -> dict[str, Any]:
    """Mechanical half of the exit criterion over a recorded run."""
    hit = first_firing(rows, k=k)
    measured = _measured(rows)
    keeps = sum(1 for r in measured if r.get("status") == "kept")
    return {
        "schema": SCHEMA, "k": k, "bound": bound,
        "measured": len(measured), "keeps": keeps,
        "fired": hit is not None,
        "fired_at_non_keeps": None if hit is None else hit.non_keeps,
        "reason": None if hit is None else hit.reason,
        "within_bound": hit is not None and hit.non_keeps <= bound,
        # The naming half needs the diagnostician on the real stores.
        "names_defect": "OPEN: requires the real run-18/run-24 stores and the actor call",
    }


def load_rows(db_path: Path | str) -> list[dict[str, Any]]:
    """Read measured rows in recorded order from an experiments.db, read-only."""
    uri = f"file:{Path(db_path)}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        cursor = connection.execute(
            "SELECT e.status, e.effect_fraction, s.floor_pct, e.mechanism_id, e.attempt_id "
            "FROM experiments e LEFT JOIN same_code_measurements s USING (attempt_id) "
            "ORDER BY e.recorded_at, e.rowid")
        return [{"status": a, "effect_fraction": b, "floor_pct": c,
                 "mechanism_id": d, "attempt_id": e} for a, b, c, d, e in cursor]
    finally:
        connection.close()


def validate_plan(plan: Mapping[str, Any]) -> list[str]:
    """Defects in a diagnostician plan artifact; empty list means well-formed."""
    defects = [f"missing key: {key}" for key in PLAN_REQUIRED_KEYS if key not in plan]
    diagnosis = plan.get("diagnosis")
    if isinstance(diagnosis, Mapping):
        for name in diagnosis:
            if name not in CHECKLIST:
                defects.append(f"diagnosis outside checklist: {name}")
        if not diagnosis:
            defects.append("diagnosis empty")
    elif "diagnosis" in plan:
        defects.append("diagnosis must be a mapping over the checklist")
    for entry in plan.get("do_not_try") or []:
        if not (isinstance(entry, Mapping) and entry.get("experiment_id")):
            defects.append("do_not_try entry without experiment id")
    if plan.get("gate") not in (None, False):
        defects.append("plan is advisory and must not carry a gate")
    return defects
