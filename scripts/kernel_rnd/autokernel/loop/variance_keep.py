"""U3-SEED / S3-AKU-04: variance-only keep rule for RUNTIME_CONFIG arms (OP-AKU-U3).

Pure module (stdlib only, no I/O except `load_input`). Schema `epyc.autokernel.variance_keep.v1`.

DATA. n paired SESSION UNITS: unit i is one (anchor, candidate) pair of separate-process
launches measured back to back, throughput values > 0. The session unit is what is
resampled; the pairing is never broken.

ESTIMAND. Downside semideviation of an arm, in the reporting-only `serving._spread`
grammar: deviation is taken from the arm's OWN median (the same reference `_sorted_devs`,
`floor_pct` and `_spread` use), relative to that median, and only launches BELOW the median
count:  sd_down(arm) = sqrt(mean_i(max(0, (med - x_i) / med)^2)).  The statistic is
theta = ln(sd_down(candidate) / sd_down(anchor)); theta < 0 means the candidate's downside
tail is tighter. Zero semideviations are clamped at SEMIDEV_EPS (1e-9, fractional) so the log
is defined; an anchor whose point semideviation is 0 has no tail to reduce and is REFUSED.

TEST. Deterministic percentile paired bootstrap (random.Random(seed), B >= 10000 draws,
resampling session units with replacement; the median is recomputed inside every draw).
KEEP_VARIANCE requires the one-sided 95% upper bound (95th percentile of theta*) < 0.

THROUGHPUT GUARD. Per-unit effect e_i = 100 * (cand_i / anchor_i - 1); the point effect is
the mean over units; its one-sided 95% lower confidence bound is the 5th percentile of the
bootstrap mean (same draws). Guard passes iff LCB > -floor_pct, where `floor_pct` (> 0, the
applicable SESSION-unit floor, percent) is an INPUT, never hardcoded here; `floor_unit` must
be "session" or the call refuses (`floor_unit_mismatch`).

NOT CHANGED. p95 deviation stays reporting-only. n < 14 refuses (`insufficient_n`).
MULTIPLICITY: exactly one variance test per candidate; no alpha spending beyond that
(recorded in the verdict). This verdict is ADDITIVE: it never overrides or vetoes an
effect-based decision; callers record it beside that decision.
"""
from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "epyc.autokernel.variance_keep.v1"
INPUT_SCHEMA = "epyc.autokernel.variance_input.v1"
KEEP_VARIANCE = "KEEP_VARIANCE"
REFUSE = "REFUSE"
MIN_N = 14
DRAWS = 10000
SEED = 20261118
LEVEL = 0.95
SEMIDEV_EPS = 1e-9
SESSION_UNIT = "session"
MULTIPLICITY = "one_variance_test_per_candidate;no_alpha_spending"
METHOD = "paired_session_unit_bootstrap.log_ratio_downside_semidev_vs_own_median.v1"

R_INSUFFICIENT_N = "insufficient_n"
R_BAD_INPUT = "bad_input"
R_FLOOR_UNIT = "floor_unit_mismatch"
R_BAD_FLOOR = "bad_floor"
R_DEGENERATE_ANCHOR = "degenerate_anchor_spread"
R_NOT_TIGHTER = "variance_not_reduced"
R_THROUGHPUT = "throughput_regression"


def semideviation(values: Sequence[float]) -> float:
    """Downside semideviation relative to the arm's own median (fractional units)."""
    med = statistics.median(values)
    return math.sqrt(sum(max(0.0, (med - v) / med) ** 2 for v in values) / len(values))


def _log_ratio(cand: Sequence[float], anchor: Sequence[float]) -> float:
    return math.log(max(semideviation(cand), SEMIDEV_EPS) / max(semideviation(anchor), SEMIDEV_EPS))


def _pct(sorted_vals: Sequence[float], q: float) -> float:
    return sorted_vals[min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))]


def _refuse(reasons: list[str], **extra: Any) -> dict:
    return {"schema": SCHEMA, "verdict": REFUSE, "reasons": reasons, "method": METHOD,
            "multiplicity": MULTIPLICITY, "authority": "additive_never_overrides_effect_decision",
            **extra}


def evaluate(units: Sequence[Mapping[str, float]], *, floor_pct: Any, floor_unit: str,
             seed: int = SEED, draws: int = DRAWS) -> dict:
    """Typed verdict: KEEP_VARIANCE or REFUSE with reasons. Never raises on data defects."""
    if type(draws) is not int or draws < DRAWS:
        return _refuse([R_BAD_INPUT], detail=f"draws must be an int >= {DRAWS}")
    try:
        anchor = [float(u["anchor"]) for u in units]
        cand = [float(u["candidate"]) for u in units]
    except (KeyError, TypeError, ValueError):
        return _refuse([R_BAD_INPUT], detail="each unit needs numeric anchor and candidate")
    if any(not math.isfinite(v) or v <= 0 for v in anchor + cand):
        return _refuse([R_BAD_INPUT], detail="launch values must be finite and > 0")
    n = len(anchor)
    reasons: list[str] = []
    if n < MIN_N:
        reasons.append(R_INSUFFICIENT_N)
    if floor_unit != SESSION_UNIT:
        reasons.append(R_FLOOR_UNIT)
    if (isinstance(floor_pct, bool) or not isinstance(floor_pct, (int, float))
            or not math.isfinite(floor_pct) or floor_pct <= 0):
        reasons.append(R_BAD_FLOOR)
    if reasons:
        return _refuse(reasons, n_units=n, min_n=MIN_N)
    if semideviation(anchor) == 0.0:
        return _refuse([R_DEGENERATE_ANCHOR], n_units=n, min_n=MIN_N)

    rng = random.Random(seed)
    thetas, effects = [], []
    for _ in range(draws):
        idx = [rng.randrange(n) for _ in range(n)]
        a = [anchor[i] for i in idx]
        c = [cand[i] for i in idx]
        thetas.append(_log_ratio(c, a))
        effects.append(sum(100.0 * (c[i] / a[i] - 1.0) for i in range(n)) / n)
    thetas.sort()
    effects.sort()
    theta_hat = _log_ratio(cand, anchor)
    upper = _pct(thetas, LEVEL)
    effect_hat = sum(100.0 * (cand[i] / anchor[i] - 1.0) for i in range(n)) / n
    lcb = _pct(effects, 1.0 - LEVEL)
    tighter = upper < 0.0
    guard = lcb > -float(floor_pct)
    out_reasons = []
    if not tighter:
        out_reasons.append(R_NOT_TIGHTER)
    if not guard:
        out_reasons.append(R_THROUGHPUT)
    return {"schema": SCHEMA, "verdict": KEEP_VARIANCE if not out_reasons else REFUSE,
            "reasons": out_reasons, "method": METHOD, "multiplicity": MULTIPLICITY,
            "authority": "additive_never_overrides_effect_decision",
            "n_units": n, "min_n": MIN_N, "seed": seed, "draws": draws, "level": LEVEL,
            "semidev_anchor": semideviation(anchor), "semidev_candidate": semideviation(cand),
            "log_ratio": theta_hat, "log_ratio_upper95": upper,
            "effect_pct": effect_hat, "effect_pct_lcb95": lcb,
            "floor_pct": float(floor_pct), "floor_unit": floor_unit,
            "p95_dev": "reporting_only_not_read"}


def load_input(path: Path | str) -> dict:
    """Read `epyc.autokernel.variance_input.v1`: {schema, candidate_id, floor_pct,
    floor_unit, units:[{anchor, candidate}, ...]}. Closed: unknown keys or a wrong schema raise."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(data, dict) or data.get("schema") != INPUT_SCHEMA
            or set(data) != {"schema", "candidate_id", "floor_pct", "floor_unit", "units"}):
        raise ValueError(f"not a closed {INPUT_SCHEMA} record")
    return data


def keep_record_fields(path: Path | str, frontier_path: Path | str) -> dict:
    """Evaluate the supplied variance input, put (effect, spread) on the keep record and
    apply keep_frontier.retain against the persisted frontier. Spread = candidate downside
    semideviation in percent. Never raises: a defect becomes a typed REFUSE/error field, and
    the effect-based keep stands either way."""
    from . import keep_frontier as kf  # local: keep this module's core free of the orchestrator load
    try:
        data = load_input(path)
        verdict = evaluate(data["units"], floor_pct=data["floor_pct"], floor_unit=data["floor_unit"])
    except Exception as exc:  # noqa: BLE001 - recorded, never veto a committed keep
        return {"variance_keep": _refuse([R_BAD_INPUT], detail=f"{type(exc).__name__}: {exc}")}
    out: dict = {"variance_keep": verdict}
    if "effect_pct" not in verdict:
        return out
    try:
        point = kf.make_point(str(data["candidate_id"]), verdict["effect_pct"],
                              100.0 * verdict["semidev_candidate"])
        fp = Path(frontier_path)
        existing = []
        if fp.is_file():
            existing = [kf.make_point(r["candidate_id"], r["effect"], r["spread"])
                        for r in json.loads(fp.read_text(encoding="utf-8"))["points"]]
        existing = [p for p in existing if p.candidate_id != point.candidate_id]
        after, evicted = kf.retain(existing, point)
        fp.parent.mkdir(parents=True, exist_ok=True)
        tmp = fp.with_suffix(".tmp")
        tmp.write_text(json.dumps({"schema": kf.SCHEMA, "points": [p.record() for p in after]},
                                  sort_keys=True), encoding="utf-8")
        tmp.replace(fp)
        out["keep_frontier"] = {**point.record(), "admitted": point in after,
                                "evicted": [p.candidate_id for p in evicted],
                                "frontier_size": len(after)}
    except Exception as exc:  # noqa: BLE001
        out["keep_frontier"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out
