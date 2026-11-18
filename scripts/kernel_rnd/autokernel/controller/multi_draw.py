#!/usr/bin/env python3
"""R24-10 randomized multi-draw arm for near-tie ops (rider on R22-5).

Selection and near-tie ops (ARGMAX, TOP_K, sampler paths) can pass a fixed or
3-draw oracle and still fail at a small per-draw rate p: a k-draw check misses a
p failure with probability (1-p)^k (about 95.6 % at p=0.015, k=3). This module is
the pure arithmetic and aggregation for the arm: seed schedule, miss probability,
draws needed for a detection target, and a mismatch-rate summary with a
conservative upper bound. The run itself goes through the oracle's existing
`--suite-seed`/`--repeat-suite` surface; `run_draws` takes the per-draw verdict
as a callable, so no process is started here.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, Mapping

SCHEMA = "epyc.autokernel.multi_draw.v1"
DEFAULT_DRAWS = 100
#: Ops whose correctness is a selection or near-tie decision.
NEAR_TIE_OPS = frozenset({"ARGMAX", "TOP_K", "ARGSORT", "TOPK_MOE", "SOFT_MAX"})


def miss_probability(p: float, draws: int) -> float:
    """Chance that `draws` independent draws all pass despite per-draw failure rate p."""
    if not 0.0 <= p <= 1.0 or draws < 0:
        raise ValueError("p must be in [0, 1] and draws >= 0")
    return (1.0 - p) ** draws


def draws_needed(p: float, detect: float = 0.95) -> int:
    """Smallest draw count that detects a rate-p failure with probability >= detect."""
    if not 0.0 < p <= 1.0 or not 0.0 < detect < 1.0:
        raise ValueError("p in (0, 1], detect in (0, 1)")
    if p == 1.0:
        return 1
    return math.ceil(math.log(1.0 - detect) / math.log(1.0 - p))


def seed_schedule(base: int, draws: int) -> list[int]:
    """Deterministic distinct u64 suite seeds (splitmix64 of base + i)."""
    mask = (1 << 64) - 1
    out = []
    for i in range(draws):
        z = (base + (i + 1) * 0x9E3779B97F4A7C15) & mask
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        out.append(z ^ (z >> 31))
    return out


def upper_bound(failures: int, draws: int, confidence: float = 0.95) -> float:
    """One-sided upper bound on the per-draw failure rate (exact for 0 failures)."""
    if draws <= 0 or not 0 <= failures <= draws:
        raise ValueError("need draws > 0 and 0 <= failures <= draws")
    if failures == 0:
        return 1.0 - (1.0 - confidence) ** (1.0 / draws)       # "rule of three" form
    phat = failures / draws
    z = 1.6448536269514722 if abs(confidence - 0.95) < 1e-9 else _z(confidence)
    denom = 1 + z * z / draws
    centre = phat + z * z / (2 * draws)
    margin = z * math.sqrt(phat * (1 - phat) / draws + z * z / (4 * draws * draws))
    return min(1.0, (centre + margin) / denom)


def _z(confidence: float) -> float:
    lo, hi = 0.0, 8.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < confidence:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def run_draws(op: str, case: str, run_one: Callable[[int], bool], *, base_seed: int,
              draws: int = DEFAULT_DRAWS) -> dict:
    """Run `draws` fresh-input checks for one boundary case; return the mismatch record.

    `run_one(seed) -> bool` is True when the draw matched the reference. An exception
    or non-bool is a refused draw (counted separately, never as a pass).
    """
    if op not in NEAR_TIE_OPS:
        raise ValueError(f"{op} is not a registered near-tie op")
    seeds = seed_schedule(base_seed, draws)
    mismatches, refused = [], 0
    for seed in seeds:
        try:
            ok = run_one(seed)
        except Exception:
            refused += 1
            continue
        if ok is True:
            continue
        if ok is False:
            mismatches.append(seed)
        else:
            refused += 1
    answered = draws - refused
    return {
        "schema": SCHEMA, "op": op, "case": case, "draws": draws, "answered": answered,
        "refused": refused, "mismatches": len(mismatches), "mismatch_seeds": mismatches[:20],
        "mismatch_rate": (len(mismatches) / answered) if answered else None,
        "rate_upper_95": upper_bound(len(mismatches), answered) if answered else None,
        "passes": answered > 0 and refused == 0 and not mismatches,
    }


def summarize(records: Iterable[Mapping[str, object]]) -> dict:
    records = list(records)
    return {"schema": SCHEMA, "cases": len(records),
            "failed_cases": [r["case"] for r in records if not r["passes"]],
            "passes": bool(records) and all(r["passes"] for r in records)}
