"""RSL-R3 OP-PERF SCREEN core: a pure, served-weighted op-level perf screen. Never a gate.

A *shape class* ({op_family, dims: {name: [values]}}) is intersected with the lane's served shapes
(`served_shape_cases.lane_served_shapes`, passed IN by the integrator) and crossed with the served
n-set to form the cells. Each cell is measured as paired anchor/candidate samples with its OWN paired
floor (percent, same unit as `bench.Comparison.noise_floor_pct`). The decision follows the loop's
measurement-sufficiency rules:

  * per cell the check is FUTILITY-only: a cell that regresses beyond its own floor is MARKED; a cell
    never proves a win by itself;
  * the verdict is on the weights-weighted AGGREGATE (weight = calls_per_round x share), and any marked
    cell rejects the screen (a one-cell win cannot pay for a regression elsewhere);
  * fail-closed: a weighted cell without a valid measurement (too few pairs, non-finite sample, no
    floor, no weight) rejects as UNMEASURED, never as a silent skip.

Direction: samples are TIME per call (us/run, lower is better), like `test-backend-ops perf`.
`effect_pct` = speedup = median(anchor) / median(candidate) - 1, so POSITIVE is better and a
regression is `effect_pct < -floor_pct`. The output is always `screening_only=True`: it routes work
to the serving A/B and the gates and decides nothing there. This module wires nothing.
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Iterable, Mapping, Sequence

SCHEMA = "epyc.autokernel.op_perf_screen.v1"
STAGE = "op-perf-screen"
#: Same pairing floor the serving A/B uses (`bench.MIN_PAIRS`); duplicated so the module stays pure.
MIN_PAIRS = 5

PASS, REJECT, UNMEASURED = "pass", "reject", "unmeasured"
_DIM_NAMES = ("k", "m", "n_mats", "n_used")


def weight(calls_per_round: float, share: float) -> float:
    """A cell's decision weight: how many times a round calls the op times its share of the round."""
    w = float(calls_per_round) * float(share)
    if not math.isfinite(w) or w <= 0:
        raise ValueError(f"weight must be finite and positive, got calls_per_round={calls_per_round!r} share={share!r}")
    return w


def cell_key(shape_name: str, n: int) -> str:
    return f"{shape_name}@n{int(n)}"


def _attr(shape: Any, name: str) -> Any:
    return shape[name] if isinstance(shape, Mapping) else getattr(shape, name)


def grid(shape_class: Mapping[str, Any], served_shapes: Iterable[Any], n_set: Iterable[int]) -> list[dict]:
    """The cells: served shapes of `op_family` whose dims all fall in the class, x each n in `n_set`.

    `served_shapes` are `ServedShape`s (or mappings with name/op/k/m/n_mats/n_used). The class can only
    NARROW what the lane serves, never add a shape the GGUF does not have. An unknown dim name or an
    empty n-set is a defect (ValueError), not an empty grid; a class that matches nothing is `[]`, and
    `decide([])` rejects it.
    """
    family = str(shape_class["op_family"])
    dims = {str(name): {int(v) for v in values} for name, values in dict(shape_class.get("dims") or {}).items()}
    unknown = sorted(set(dims) - set(_DIM_NAMES))
    if unknown:
        raise ValueError(f"unknown shape-class dims {unknown}; known: {list(_DIM_NAMES)}")
    ns = sorted({int(n) for n in n_set})
    if not ns or any(n < 1 for n in ns):
        raise ValueError("n_set must be a non-empty set of positive widths")
    cells, seen = [], set()
    for shape in served_shapes:
        if str(_attr(shape, "op")) != family or any(int(_attr(shape, name)) not in allowed for name, allowed in dims.items()):
            continue
        name = str(_attr(shape, "name"))
        if name in seen:
            continue
        seen.add(name)
        for n in ns:
            cells.append({"key": cell_key(name, n), "shape": name, "op": family, "n": n, "dims": {d: int(_attr(shape, d)) for d in _DIM_NAMES}})
    return sorted(cells, key=lambda c: (c["shape"], c["n"]))


def _samples(values: Any) -> list[float] | None:
    try:
        out = [float(v) for v in values]
    except (TypeError, ValueError):
        return None
    return out if out and all(math.isfinite(v) and v > 0 for v in out) else None


def _cell_row(cell: Mapping[str, Any], w: float | None) -> dict:
    """One cell's verdict: ok / regressed / unmeasured (with the reason)."""
    row: dict[str, Any] = {"key": cell.get("key"), "weight": w}

    def bad(reason: str) -> dict:
        return {**row, "status": UNMEASURED, "reason": reason, "effect_pct": None, "floor_pct": None}

    if w is None or not math.isfinite(w) or w <= 0:
        return bad("no positive census weight (unmeasured n is non-deciding)")
    a, c = _samples(cell.get("anchor")), _samples(cell.get("candidate"))
    if a is None or c is None:
        return bad("anchor/candidate samples missing, non-finite or non-positive")
    if len(a) != len(c):
        return bad(f"unpaired samples ({len(a)} anchor vs {len(c)} candidate)")
    if len(a) < MIN_PAIRS:
        return bad(f"{len(a)} pairs < {MIN_PAIRS}")
    floor = cell.get("floor_pct")
    if not isinstance(floor, (int, float)) or isinstance(floor, bool) or not math.isfinite(floor) or floor <= 0:
        return bad("no paired floor for this cell")
    effect = (statistics.median(a) / statistics.median(c) - 1.0) * 100.0
    status = "regressed" if effect < -float(floor) else "ok"
    return {**row, "status": status, "effect_pct": effect, "floor_pct": float(floor), "pairs": len(a)}


def decide(cells: Sequence[Mapping[str, Any]], weights: Mapping[str, float]) -> dict:
    """Screen decision over `cells` (dicts: key, anchor[], candidate[], floor_pct) and `weights`
    ({key: calls_per_round x share}). `screen` is pass / reject / unmeasured; `screening_only` is
    always True. Keys are the union of both inputs, so a weighted key with no cell (or a cell with no
    weight) is UNMEASURED: nothing decides on a subset of the served mass."""
    by_key: dict[Any, Mapping[str, Any]] = {}
    for cell in cells:
        if cell.get("key") in by_key:
            raise ValueError(f"duplicate cell key {cell.get('key')!r}")
        by_key[cell.get("key")] = cell
    rows = [_cell_row(by_key.get(key, {"key": key}), weights.get(key)) for key in sorted(set(by_key) | set(weights), key=str)]
    unmeasured = [r["key"] for r in rows if r["status"] == UNMEASURED]
    regressed = [r["key"] for r in rows if r["status"] == "regressed"]
    out: dict[str, Any] = {"schema": SCHEMA, "screening_only": True, "cells": rows, "regressed": regressed,
                           "unmeasured": unmeasured, "aggregate_effect_pct": None, "aggregate_floor_pct": None}
    if not rows:
        return {**out, "screen": UNMEASURED, "reason": "no cells (empty grid or class matched no served shape)"}
    if unmeasured:
        return {**out, "screen": UNMEASURED,
                "reason": f"{len(unmeasured)}/{len(rows)} cells unmeasured: {unmeasured[0]} ({next(r['reason'] for r in rows if r['key'] == unmeasured[0])})"}
    total = sum(r["weight"] for r in rows)
    agg = sum(r["weight"] * r["effect_pct"] for r in rows) / total
    agg_floor = sum(r["weight"] * r["floor_pct"] for r in rows) / total
    out.update(aggregate_effect_pct=agg, aggregate_floor_pct=agg_floor, weight_total=total)
    if regressed:
        return {**out, "screen": REJECT,
                "reason": f"{len(regressed)} cell(s) regressed beyond their own floor: {regressed[0]} "
                          f"({next(r['effect_pct'] for r in rows if r['key'] == regressed[0]):+.3f}%); weighted aggregate {agg:+.3f}% does not offset a regression"}
    if agg <= agg_floor:
        return {**out, "screen": REJECT,
                "reason": f"weighted aggregate {agg:+.3f}% does not clear the weighted floor {agg_floor:.3f}%"}
    return {**out, "screen": PASS,
            "reason": f"weighted aggregate {agg:+.3f}% clears the weighted floor {agg_floor:.3f}% over {len(rows)} cells, none regressed"}


def polar_result(decision: Mapping[str, Any], base: Mapping[str, Any] | None = None) -> dict:
    """The loop's polar RESULT (stage_bodies convention): `verdict` names the answer, `polar` is the one
    PASS/FAIL token that `gate_eval` reads. UNMEASURED is a harness futility, so it carries NO polar
    token (as `stage_timing` does for too few pairs): it must never read as a candidate negative.
    `base` is the integrator's `stage_bodies._base(...)` dict (row/candidate/shas/sampling)."""
    result = dict(base) if base is not None else {"schema": SCHEMA, "stage": STAGE, "complete": True}
    result.update(stage=STAGE, screening_only=True, op_perf_screen=dict(decision))
    screen, reason = decision["screen"], str(decision.get("reason", ""))[:160]
    if screen == PASS:
        result.update(verdict="PASS_OP_PERF_SCREEN", polar="PASS")
    elif screen == REJECT:
        kind = "REGRESSION" if decision.get("regressed") else "NO_GAIN"
        result.update(verdict=f"FAIL_OP_PERF_{kind} ({reason})", polar="FAIL")
    else:
        result.update(verdict=f"FUTILITY_HARNESS_INSTRUMENT (op-perf screen {reason})")
        result.pop("polar", None)
    return result
