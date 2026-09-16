"""Paired contrasts for OCC-1 (stdlib only).

Recall is paired per question (same qid, same chunk, arm vs text). Primary recall metric is
SQuAD F1 (higher = better); EM is secondary with an exact McNemar test.
"""

from __future__ import annotations

import math
import random


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def se_mean(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n * (n - 1)))


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar p on discordant counts b (arm-only right), c (text-only right)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_bootstrap_ci(deltas: list[float], iters: int = 10000, seed: int = 0,
                        alpha: float = 0.05, cluster: list[int] | None = None) -> tuple[float, float]:
    """Percentile CI of the mean paired delta. With `cluster` (chunk ids), resamples whole chunks,
    since questions sharing a carrier are not independent."""
    rng = random.Random(seed)
    if not deltas:
        return (float("nan"), float("nan"))
    if cluster is None:
        groups = [[d] for d in deltas]
    else:
        by: dict[int, list[float]] = {}
        for d, g in zip(deltas, cluster):
            by.setdefault(g, []).append(d)
        groups = list(by.values())
    stats = []
    for _ in range(iters):
        tot = n = 0.0
        for _ in range(len(groups)):
            g = groups[rng.randrange(len(groups))]
            tot += sum(g)
            n += len(g)
        stats.append(tot / n)
    stats.sort()
    lo = stats[int(math.floor(alpha / 2 * iters))]
    hi = stats[min(iters - 1, int(math.ceil((1 - alpha / 2) * iters)) - 1)]
    return lo, hi


def verdict(ratio: float, ci_lo: float, max_ratio: float, ni_margin: float) -> str:
    """Pre-registered OCC-1 decision for one image arm vs text.

    POSITIVE         billed-token ratio <= max_ratio AND F1-delta CI lower bound > -ni_margin
    NOT_NONINFERIOR  ratio <= max_ratio but the CI admits a recall loss beyond the margin
    NEGATIVE_COST    ratio > max_ratio (not worth it whatever the recall)
    """
    if ratio > max_ratio:
        return "NEGATIVE_COST"
    if ci_lo > -ni_margin:
        return "POSITIVE"
    return "NOT_NONINFERIOR"
