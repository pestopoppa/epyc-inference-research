"""KEEP-TAIL-1: p95 timing inflation P beside every GPU A/B, widened floor F.

    F = max(0.02, sqrt(2) * P)

P is the p95 inflation of a single arm's timing (p95/median - 1, per-arm
median-normalized samples pooled, worst cell).  Independent arms double the
variance, hence sqrt(2).  Reporting P and F is harness work; ENFORCING F as a
keep gate changes the C7 A/B contract and needs the champion owner's agreement,
so `apply_rule` returns the legacy and the widened verdict side by side and
callers choose which one gates (default: legacy gates, widened is info).
"""
import math

FLOOR_MIN = 0.02


def percentile(xs, q):
    """Linear-interpolated percentile, q in [0,1].  n small => tends to the max."""
    xs = sorted(xs)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def p95_inflation(*arm_samples):
    """P from within-arm scatter (or A/A arms).  Each argument is one arm's
    timing samples; each is normalized by its own median, pooled, and P =
    p95(pooled) - 1 (>=0).  None when fewer than 3 pooled samples."""
    pooled = []
    for s in arm_samples:
        s = [x for x in s if x]
        if len(s) < 2:
            continue
        m = _median(s)
        pooled += [x / m for x in s]
    if len(pooled) < 3:
        return None
    return max(0.0, percentile(pooled, 0.95) - 1.0)


def widened_floor(p, floor_min=FLOOR_MIN):
    if p is None:
        return floor_min
    return max(floor_min, math.sqrt(2.0) * p)


def cells_tail(cells):
    """cells as built by decode_metrics.cells_from_visits.  Returns
    {"P_per_cell": {pid: P}, "P": worst, "F": widened floor}."""
    per = {pid: p95_inflation(c["a"], c["b"]) for pid, c in cells.items()}
    vals = [p for p in per.values() if p is not None]
    p = max(vals) if vals else None
    return {"P_per_cell": per, "P": p, "F": widened_floor(p)}


def apply_rule(cells, geo_min=0.995, cell_min=0.98, ratio_key="mean_ratio", floor=None, min_gain=None):
    """Apply a pre-registered 'geomean >= geo_min and no cell < cell_min' rule.

    Legacy verdict uses the thresholds as written.  Widened verdict replaces
    each tolerance t = 1 - threshold by max(t, F).  If min_gain is given (a
    keep-by-gain rule), a gain below F is refused as inside the tail.
    Returns {"legacy", "widened", "P", "F", "geomean", "clauses_*", "refusal"}.
    """
    tail = cells_tail(cells)
    F = tail["F"] if floor is None else floor
    ratios = {pid: c[ratio_key] for pid, c in cells.items()}
    prod = 1.0
    for r in ratios.values():
        prod *= r
    geo = prod ** (1.0 / len(ratios)) if ratios else None

    def clauses(gt, ct):
        out = []
        if geo is None:
            return ["no cells"]
        if geo < 1.0 - gt:
            out.append(f"geomean {geo:.4f} < {1.0 - gt:.4f}")
        out += [f"{p} {r:.4f} < {1.0 - ct:.4f}" for p, r in ratios.items() if r < 1.0 - ct]
        return out

    leg = clauses(1.0 - geo_min, 1.0 - cell_min)
    wid = clauses(max(1.0 - geo_min, F), max(1.0 - cell_min, F))
    refusal = None
    if min_gain is not None and geo is not None and min_gain <= geo - 1.0 < F:
        refusal = f"gain {geo - 1.0:.4f} inside tail floor F={F:.4f}: keep refused"
    return {"P": tail["P"], "P_per_cell": tail["P_per_cell"], "F": F, "geomean": geo,
            "legacy": "PASS" if not leg else "FAIL", "clauses_legacy": leg,
            "widened": ("PASS" if not wid else "FAIL") if not refusal else "REFUSED_INSIDE_TAIL",
            "clauses_widened": wid, "refusal": refusal,
            "verdict_changed": ("PASS" if not leg else "FAIL") != ("PASS" if not wid else "FAIL")}
