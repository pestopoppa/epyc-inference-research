"""Statistical admission contract for an AutoKernel performance headline.

The unit is one independent process launch. The design uses the measured
between-launch SD from INF-70 RETEST-1 (2.793%), a stated 3.0% minimum
detectable effect, two-sided alpha=.05, and 80% power. The conventional
two-independent-sample normal approximation is

    n/arm = ceil(2 * ((z_(1-alpha/2) + z_power) * sd / MDE) ** 2)

which gives 14 launches per arm / 28 total. The MDE is the campaign's existing
3% operational screening scale, made explicit rather than inferred from a
caller's convenient pair count.
"""
from __future__ import annotations

import math
import random
import statistics
from typing import Sequence

UNIT = "process"
BETWEEN_LAUNCH_SD_PCT = 2.793
MDE_PCT = 3.0
ALPHA = 0.05
POWER = 0.80
Z_TWO_SIDED_95 = 1.959963984540054
Z_POWER_80 = 0.8416212335729143
MIN_LAUNCHES_PER_ARM = math.ceil(
    2.0 * ((Z_TWO_SIDED_95 + Z_POWER_80) * BETWEEN_LAUNCH_SD_PCT / MDE_PCT) ** 2)
MIN_LAUNCHES = 2 * MIN_LAUNCHES_PER_ARM
CI_METHOD = "paired_bootstrap_median_ratio_95_seed_20260915_draws_20000"


class HeadlineInadmissible(ValueError):
    """The samples cannot support an absolute performance headline."""


def contract() -> dict:
    return {"unit": UNIT, "n": MIN_LAUNCHES,
            "n_per_arm": MIN_LAUNCHES_PER_ARM, "mde_pct": MDE_PCT,
            "between_launch_sd_pct": BETWEEN_LAUNCH_SD_PCT,
            "alpha": ALPHA, "power": POWER, "sidedness": "two-sided",
            "planning_method": "normal_two_independent_samples_equal_n",
            "ci_level": 0.95, "ci_method": CI_METHOD}


def confidence_interval(anchor: Sequence[float], candidate: Sequence[float]) -> dict:
    """Deterministic paired-bootstrap 95% interval for the headline's median ratio."""
    a, c = list(anchor), list(candidate)
    if len(a) < MIN_LAUNCHES_PER_ARM or len(c) < MIN_LAUNCHES_PER_ARM:
        raise HeadlineInadmissible(
            f"headline REFUSED: {len(a) + len(c)} independent {UNIT} launches "
            f"({len(a)} anchor, {len(c)} candidate) is below N={MIN_LAUNCHES} "
            f"({MIN_LAUNCHES_PER_ARM} per arm), sized for MDE={MDE_PCT:.1f}% from "
            f"between-launch sd={BETWEEN_LAUNCH_SD_PCT:.3f}% at two-sided "
            f"alpha={ALPHA:.2f}, power={POWER:.2f}")
    if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in a + c):
        raise HeadlineInadmissible("headline REFUSED: CI requires finite positive launch rates")
    # Alternation makes each ordinal an A/B block. Resample whole blocks so the CI
    # preserves that drift control and targets the exact median/median headline.
    pairs = min(len(a), len(c))
    rng = random.Random(20260915)
    effects = []
    for _ in range(20_000):
        idx = [rng.randrange(pairs) for _ in range(pairs)]
        effects.append(statistics.median(c[i] for i in idx)
                       / statistics.median(a[i] for i in idx) - 1.0)
    effects.sort()
    return {"unit": UNIT, "level": 0.95, "method": CI_METHOD,
            "lower_effect_fraction": effects[int(0.025 * (len(effects) - 1))],
            "upper_effect_fraction": effects[int(0.975 * (len(effects) - 1))]}


__all__ = ["BETWEEN_LAUNCH_SD_PCT", "CI_METHOD", "HeadlineInadmissible",
           "MDE_PCT", "MIN_LAUNCHES", "MIN_LAUNCHES_PER_ARM", "UNIT",
           "confidence_interval", "contract"]
