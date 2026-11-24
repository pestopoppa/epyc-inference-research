"""AK-EVAL-FUT: futility-only interim looks for the paired A/B instruments.

After each COMPLETE pair k >= min_pairs the comparison may stop as futile when

    mean(per-pair log-ratio %) + z * sigma_pair / sqrt(k) < bar_pct

i.e. even an optimistic z-sigma bound on the running effect cannot reach the bar.
A futile stop is a NULL, never a keep: `bench.Comparison.decisive` and the serving
row's `decisive` are False with `stop_reason: "futility"`, and `loop.iterate`
refuses it on the accumulate path too. Stop-on-answer (an early KEEP) does not
exist here: formal-gate cells stay futility-only until OP-ABSEQ / OP-PQP /
OP-ALPHAWIN are decided (measurement-sufficiency.md).

sigma_pair comes from the surface's stored A/A calibration -- the same source as
the floor (`bench.floor_rows` / the matched serving floor record). Defaults were
chosen on the retrospective replay (audit 2026-11-24 §P4, test_futility.py):
serving z=2.5 (0 DS41/Q38FN winners stopped), GPU bench z=3.5 (z=2.5 and 3.0
stopped 3 of 42 above-floor GPU-era winners; 3.5 stopped none).
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

STOPPING = "fixed_pairs+futility_only.v1"
SCHEMA = "epyc.autokernel.futility_look.v1"
Z_SERVING = 2.5
Z_BENCH = 3.5
MIN_PAIRS = 2
#: p95 of |N(0, sigma)| is 1.96 sigma: a single-pair p95 floor row converts to sigma.
P95_Z = 1.96


def log_ratios_pct(anchor, candidate) -> list[float]:
    """Per complete pair: 100 * ln(candidate / anchor)."""
    return [100.0 * math.log(c / a) for a, c in zip(anchor, candidate) if a > 0 and c > 0]


def sigma_from_aa_samples(anchor, candidate) -> float | None:
    """sigma_pair of a stored A/A calibration (its per-pair log-ratio sd)."""
    ratios = log_ratios_pct(anchor or (), candidate or ())
    return statistics.stdev(ratios) if len(ratios) >= 3 else None


def sigma_from_single_pair_floor(floor1_pct) -> float | None:
    """sigma_pair from a k=1 p95 floor row (`bench.floor_rows(...)[1]`)."""
    return None if not floor1_pct or floor1_pct <= 0 else floor1_pct / P95_Z


def look(ratios, *, sigma_pair_pct: float, bar_pct: float, z: float,
         min_pairs: int = MIN_PAIRS) -> dict:
    """One interim look over the complete pairs so far. Pure."""
    k = len(ratios)
    if k < min_pairs:
        return {"decision": "continue", "k": k, "bound_pct": None}
    bound = statistics.fmean(ratios) + z * sigma_pair_pct / math.sqrt(k)
    return {"decision": "futile" if bound < bar_pct else "continue", "k": k,
            "bound_pct": round(bound, 4)}


@dataclass(frozen=True)
class Rule:
    """A pre-registered futility rule; recorded on the measurement plan."""
    sigma_pair_pct: float
    bar_pct: float
    z: float
    min_pairs: int = MIN_PAIRS
    sigma_source: str = "aa_calibration"

    def __post_init__(self):
        if not (self.sigma_pair_pct > 0 and self.z > 0 and self.min_pairs >= 2
                and self.bar_pct is not None):
            raise ValueError("futility rule needs sigma>0, z>0, min_pairs>=2 and a bar")

    def plan(self) -> dict:
        return {"schema": SCHEMA, "stop_on_answer": False, "z": self.z,
                "min_pairs": self.min_pairs, "bar_pct": self.bar_pct,
                "sigma_pair_pct": round(self.sigma_pair_pct, 6),
                "sigma_source": self.sigma_source}

    def __call__(self, anchor, candidate) -> dict:
        return look(log_ratios_pct(anchor, candidate), sigma_pair_pct=self.sigma_pair_pct,
                    bar_pct=self.bar_pct, z=self.z, min_pairs=self.min_pairs)


def rule_or_none(*, z, min_pairs, bar_pct, sigma_pair_pct, sigma_source) -> Rule | None:
    """The run's rule, or None (fixed_pairs) when off, uncalibrated or sigma-less."""
    if z is None or bar_pct is None or not sigma_pair_pct:
        return None
    return Rule(sigma_pair_pct, bar_pct, z, min_pairs, sigma_source)


def replay(comparisons, *, sigma_pair_pct: float, z: float, min_pairs: int = MIN_PAIRS) -> dict:
    """Retrospective replay over recorded per-pair samples.

    `comparisons`: iterables of {"floor", "effect_pct", "log_ratios_pct"}. A winner is a
    recorded effect above its floor; `winners_stopped` must be 0 for a usable rule.
    """
    total = saved = stopped = winners_stopped = 0
    for row in comparisons:
        ratios = row["log_ratios_pct"]
        total += len(ratios)
        for k in range(min_pairs, len(ratios)):
            if look(ratios[:k], sigma_pair_pct=sigma_pair_pct, bar_pct=row["floor"],
                    z=z, min_pairs=min_pairs)["decision"] == "futile":
                saved += len(ratios) - k
                stopped += 1
                winners_stopped += row["effect_pct"] > row["floor"]
                break
    return {"pairs": total, "saved": saved, "saved_frac": saved / total if total else 0.0,
            "stopped": stopped, "winners_stopped": winners_stopped}
