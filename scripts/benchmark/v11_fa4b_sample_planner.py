#!/usr/bin/env python3
"""Prospective FA-4b sample-size sensitivity, not a quality decision.

Projects independent new paired questions from each historical cell's empirical
discordance rates. The quality statistic and percentile indices are the
original paired question-bootstrap rule. This model cannot estimate uncapping.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

BOOTSTRAP_ITERS = 10_000
CELLS = ("q8_0:off", "q8_0:on", "f16:off", "f16:on")


def nonnegative_int(x):
    return type(x) is int and x >= 0


def inspect_cell(cell: dict) -> tuple[int, int, int, dict]:
    n = cell.get("n_decidable_pairs")
    counts = cell.get("mcnemar", {})
    on = counts.get("on_only")
    off = counts.get("off_only")
    if (not nonnegative_int(n) or n != 200 or not nonnegative_int(on) or not nonnegative_int(off)
            or on + off > n or cell.get("n_scored_observed_per_arm") != {"ON": n, "OFF": n}):
        raise ValueError("complete original 200-pair discordance counts required")
    delta = cell.get("delta_on_minus_off")
    if not isinstance(delta, (int, float)) or not math.isfinite(delta) or abs(delta - (on - off) / n) > 1e-12:
        raise ValueError("discordance and delta disagree")
    termination = cell.get("termination", {})
    capped = {}
    for arm in ("ON", "OFF"):
        term = termination.get(arm, {}).get("score", {})
        cap, unknown = term.get("truncated"), term.get("unknown")
        if (not nonnegative_int(cap) or not nonnegative_int(unknown) or cap + unknown > n):
            raise ValueError("missing or invalid original cap accounting")
        capped[arm] = {"capped": cap, "termination_unknown": unknown, "observed": n}
    return n, on, off, capped


def wilson(successes: int, total: int) -> list[float]:
    p = successes / total
    z = 1.959963984540054
    den = 1 + z*z/total
    centre = (p + z*z/(2*total)) / den
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / den
    return [max(0.0, centre-radius), min(1.0, centre+radius)]


def simulate_cell(n: int, on: int, off: int, target: int, outer: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    future = rng.multinomial(target, [on/n, off/n, 1-(on+off)/n], size=outer)
    successes = 0
    for plus, minus, _ in future:
        # A question's paired difference is +1, -1 or 0; multinomial counts
        # are exactly the paired question resampling distribution.
        draws = rng.multinomial(target, [plus/target, minus/target, 1-(plus+minus)/target],
                                size=BOOTSTRAP_ITERS)
        values = np.sort((draws[:, 0] - draws[:, 1]) / target)
        if values[250] >= -0.02 and values[9750] <= 0.02:
            successes += 1
    return {"sample_pairs": target, "simulations": outer, "bootstrap_draws_per_simulation": BOOTSTRAP_ITERS,
            "inside_margin_simulations": successes, "fraction": successes/outer,
            "monte_carlo_wilson_95": wilson(successes, outer)}


def analyze(report: dict, sizes: list[int], outer: int, seed: int) -> dict:
    if report.get("schema") != "v11-fa4b-quality.m2" or set(report.get("cells", {})) != set(CELLS):
        raise ValueError("complete M2 report required")
    if not sizes or any(not nonnegative_int(n) or n < 200 or n > 100_000 for n in sizes) or len(set(sizes)) != len(sizes):
        raise ValueError("distinct sample sizes between 200 and 100000 required")
    if type(outer) is not int or not 1 <= outer <= 2000 or type(seed) is not int or seed < 0:
        raise ValueError("invalid simulation count or seed")
    cells = {}
    for index, name in enumerate(CELLS):
        n, plus, minus, capped = inspect_cell(report["cells"][name])
        cells[name] = {"historical_pairs": n, "on_only": plus, "off_only": minus,
                       "capped_responses": capped,
                       "size_sensitivity": [simulate_cell(n, plus, minus, size, outer, seed + index*100_000 + i)
                                            for i, size in enumerate(sizes)]}
    return {"schema": "v11-fa4b.sample-planning.v1", "input_manifest_sha256": report.get("manifest_sha256"),
            "seed": seed, "cells": cells,
            "scope": "Empirical capped-run discordance point estimates held fixed; Monte Carlo interval covers simulation error only. No uncertainty for true rates, corpus sampling, cap effects, quality, power guarantee, protocol amendment, or promotion. Zero observed discordance does not prove zero true discordance. Uncapped quality and runtime are unknown."}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", required=True, type=Path)
    p.add_argument("--report-sha256", required=True)
    p.add_argument("--sizes", default="200,400,800,1600")
    p.add_argument("--simulations", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    raw = args.report.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != args.report_sha256:
        raise ValueError("source report digest mismatch")
    result = analyze(json.loads(raw), [int(x) for x in args.sizes.split(",")], args.simulations, args.seed)
    result["source_report_sha256"] = digest
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
