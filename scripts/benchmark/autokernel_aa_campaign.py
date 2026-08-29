#!/usr/bin/env python3
"""A/A campaign: calibrate the instrument against itself, under three conditions.

    python3 scripts/benchmark/autokernel_aa_campaign.py --out <dir>

Anchor against ANCHOR -- the same binary on both arms -- so the true effect is exactly
zero and every structure in the output is the instrument, not a kernel. Settles four
open questions in one GPU window:

  H1  Is the drift GATE misfiring, or is the machine drifting?
      A median-of-halves contrast on 9 samples has a null SD near the bar it is
      compared against. Here we measure the drift distribution directly and compare it
      against the permutation null of its own samples.

  H2a Does host load from the preceding -j64 build bleed into the measurement?
      Condition POST_BUILD runs immediately after a build; SETTLED runs after an idle
      wait. If the mean rank trend differs, the host is in the measurement.

  H2b Is it device settling rather than host load?
      Condition PREHEATED discards six warm-up pairs instead of one. If drift collapses
      there but not in POST_BUILD, it is the device, not the host. These two are
      collinear by construction -- both monotone in position -- so only this paired
      contrast can separate them.

  H5  Is the effect floor itself calibrated?
      `MEASURED_FLOOR_PCT` was derived from exhaustive subsets of ONE fixed 20-pair
      sample. Subsets of a fixed sample cannot exceed that sample's observed tail, so
      the construction systematically understates p95. Here the floor is re-derived by
      BOOTSTRAP over fresh pairs.

Non-promotable screening under `P-AK-SEARCH-1`. Holds the mi210_0 claim, proves
residency on every invocation, and records the achieved clock.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import random
import statistics as st
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kernel_rnd"))

from autokernel.controller import build_recipe                      # noqa: E402
from autokernel.loop import bench, claim, gates                     # noqa: E402

PAIRS = 20
SETTLE_S = 300


def spearman(values):
    return bench.trend_rho(values)


def permutation_null(values, *, samples: int = 20000, seed: int = 20260829):
    """Distribution of |drift_pct| under random reorderings of these same values.

    If the observed drift sits inside this null, the arm did not drift -- the STATISTIC
    moved. This is the H1 test, and it needs no model of the machine at all.
    """
    rng = random.Random(seed)
    pool = list(values)
    out = []
    for _ in range(samples):
        rng.shuffle(pool)
        out.append(abs(bench.drift_pct(pool)))
    out.sort()
    return out


def one_condition(name: str, anchor: Path, model: Path, *, pairs: int, warmup: int,
                  floor: float, surface_pp: int, surface_tg: int) -> dict:
    print(f"\n=== {name}: {pairs} A/A pairs, {warmup} warm-up pair(s) discarded")
    started = time.monotonic()
    comparison = bench.compare(
        bench.Arm("anchor-a", anchor), bench.Arm("anchor-b", anchor),
        model, pp=surface_pp, tg=surface_tg, pairs=pairs,
        noise_floor_pct=floor, warmup_pairs=warmup)
    body = comparison.to_dict()
    body["condition"] = name
    body["warmup_pairs"] = warmup
    body["elapsed_s"] = round(time.monotonic() - started, 1)
    a, c = comparison.anchor_samples, comparison.candidate_samples
    print(f"    effect {comparison.effect * 100:+.3f}%  (TRUE effect is 0 by construction)")
    print(f"    drift  a {bench.drift_pct(a):+.3f}%  b {bench.drift_pct(c):+.3f}%")
    print(f"    rho    a {spearman(a):+.3f}  b {spearman(c):+.3f}")
    print(f"    clock  {body['residency'].get('sclk_min_mhz')}-"
          f"{body['residency'].get('sclk_max_mhz')} MHz  "
          f"stable={body['residency'].get('clock_stable')}")
    return body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-build", type=Path,
                        default=Path("/mnt/raid0/llm/tmp/build-anchor-j64"))
    parser.add_argument("--worktree", type=Path,
                        default=Path("/mnt/raid0/llm/tmp/ak-loop-tree"))
    parser.add_argument("--model", type=Path,
                        default=Path("/mnt/raid0/llm/models/"
                                     "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"))
    parser.add_argument("--surface", choices=("pp512", "tg128"), default="tg128")
    parser.add_argument("--pairs", type=int, default=PAIRS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    pp, tg = (512, 0) if args.surface == "pp512" else (0, 128)
    floor = bench.MEASURED_FLOOR_PCT[args.surface].get(9, 1.175)
    anchor = args.anchor_build / "bin" / "llama-bench"
    conditions = []

    with claim.hold() as receipt:
        print(f"claim     held on {receipt['device_id']}")

        # SETTLED — the reference condition. Idle first so neither the host nor the
        # device carries anything in from whatever ran before.
        print(f"\nsettling {SETTLE_S}s before the reference condition ...")
        time.sleep(SETTLE_S)
        conditions.append(one_condition(
            "SETTLED", anchor, args.model, pairs=args.pairs, warmup=1,
            floor=floor, surface_pp=pp, surface_tg=tg))

        # PREHEATED — same as SETTLED but six warm-up pairs instead of one. Differs
        # from SETTLED only in device warm-up, so it isolates H2b.
        conditions.append(one_condition(
            "PREHEATED", anchor, args.model, pairs=args.pairs, warmup=6,
            floor=floor, surface_pp=pp, surface_tg=tg))

        # POST_BUILD — a real -j64 build immediately before measuring, which is what
        # the loop actually does every iteration. Differs from SETTLED only in host
        # load, so it isolates H2a.
        print("\nrunning a -j64 build to load the host (this is what an iteration does)")
        verdict = gates.compiles(
            args.worktree, Path("/mnt/raid0/llm/tmp/build-aa-loadgen"),
            cmake_defines=tuple(build_recipe.HOUSE_GPU_RECIPE.cmake_defines()),
            jobs=64, cpu_list="96-183")
        print(f"    build {verdict.gate} passed={verdict.passed}")
        conditions.append(one_condition(
            "POST_BUILD", anchor, args.model, pairs=args.pairs, warmup=1,
            floor=floor, surface_pp=pp, surface_tg=tg))

    # ---- analysis -----------------------------------------------------------
    print("\n" + "=" * 72)
    report = {"schema": "epyc.autokernel.aa_campaign.v1",
              "authority": "instrument_characterisation_not_a_claim",
              "surface": args.surface, "pairs": args.pairs,
              "conditions": conditions}

    for body in conditions:
        arms = {"a": body["anchor_samples"], "b": body["candidate_samples"]}
        body["analysis"] = {}
        for label, values in arms.items():
            observed = abs(bench.drift_pct(values))
            null = permutation_null(values)
            over = sum(1 for x in null if x >= observed) / len(null)
            body["analysis"][label] = {
                "drift_pct": bench.drift_pct(values),
                "trend_rho": spearman(values),
                "permutation_p": round(over, 4),
                "null_p95_drift_pct": round(null[int(0.95 * (len(null) - 1))], 3),
                "cv_pct": round(100.0 * st.pstdev(values) / st.mean(values), 3),
            }
            print(f"  {body['condition']:>10} arm {label}: drift "
                  f"{bench.drift_pct(values):+.3f}%  rho {spearman(values):+.3f}  "
                  f"perm-p {over:.3f}  null-p95 "
                  f"{null[int(0.95 * (len(null) - 1))]:.3f}%  CV "
                  f"{100.0 * st.pstdev(values) / st.mean(values):.2f}%")

    # H5 — a fresh floor, bootstrapped over pairs rather than re-subset from one sample.
    settled = conditions[0]
    a, b = settled["anchor_samples"], settled["candidate_samples"]
    rng = random.Random(20260829)
    for k in (5, 9, 20):
        effects = []
        for _ in range(20000):
            idx = [rng.randrange(len(a)) for _ in range(k)]
            effects.append(abs(st.median([b[i] for i in idx])
                               / st.median([a[i] for i in idx]) - 1.0) * 100.0)
        effects.sort()
        p95 = effects[int(0.95 * (len(effects) - 1))]
        report.setdefault("bootstrap_floor_pct", {})[str(k)] = round(p95, 3)
        enforced = bench.MEASURED_FLOOR_PCT[args.surface].get(k)
        flag = "" if enforced is None or enforced >= p95 else "  <-- ENFORCED IS TOO LOW"
        print(f"  bootstrap floor at {k:>2} pairs: p95 {p95:.3f}%   "
              f"(current table: {enforced}){flag}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "aa-campaign.json").write_text(json.dumps(report, indent=2),
                                               encoding="utf-8")
    print(f"\nwrote {args.out / 'aa-campaign.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
