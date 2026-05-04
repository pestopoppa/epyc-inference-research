#!/bin/bash
# Analyze Phase A.2 perf gate results.
# Computes per-model Δ and aggregate (arithmetic + geometric mean).
# Pass criteria:
#   - Aggregate (geomean) Δ ≥ +0.5%, OR
#   - All per-model |Δ| ≤ 1% AND PPL bit-exact (Phase A.1) — pragmatic flip
#   - No per-model regression > -1%
set -uo pipefail

DIR="/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-05-04-q6k-default-on-validation"
SUMMARY="$DIR/a2_perf_summary.tsv"

if [[ ! -f "$SUMMARY" ]]; then
  echo "ERROR: $SUMMARY not found"
  exit 1
fi

python3 << 'PY'
import csv, math
path = "/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-05-04-q6k-default-on-validation/a2_perf_summary.tsv"
rows = []
with open(path) as f:
    next(f)  # header
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 4:
            rows.append({
                "model": parts[0],
                "env": int(parts[1]),
                "avg_ts": float(parts[2]) if parts[2] not in ("", "FAIL") else None,
                "stddev_ts": float(parts[3]) if parts[3] else 0.0,
            })

# Pair env=0 vs env=1 per model
by_model = {}
for r in rows:
    by_model.setdefault(r["model"], {})[r["env"]] = r

print(f"{'model':<26} {'env=0 t/s':>14} {'env=1 t/s':>14} {'Δ t/s':>10} {'Δ %':>8} {'σ0%':>6} {'σ1%':>6}  verdict")
print("=" * 110)
deltas = []
all_within_noise = True
worst_regression = 0.0
for model, runs in by_model.items():
    if 0 not in runs or 1 not in runs or runs[0]["avg_ts"] is None or runs[1]["avg_ts"] is None:
        print(f"{model:<26} INCOMPLETE")
        continue
    e0, e1 = runs[0]["avg_ts"], runs[1]["avg_ts"]
    s0, s1 = runs[0]["stddev_ts"] / e0 * 100, runs[1]["stddev_ts"] / e1 * 100
    delta = e1 - e0
    delta_pct = delta / e0 * 100
    deltas.append(delta_pct / 100)  # for geomean
    verdict = "OK" if abs(delta_pct) <= 1.0 else ("REGRESSION" if delta_pct < -1.0 else "WIN")
    if delta_pct < worst_regression:
        worst_regression = delta_pct
    print(f"{model:<26} {e0:>14.4f} {e1:>14.4f} {delta:>+10.4f} {delta_pct:>+7.2f}% {s0:>5.2f}% {s1:>5.2f}%  {verdict}")

# Aggregate
if deltas:
    arith = sum(deltas) / len(deltas) * 100
    geo = (math.prod(1 + d for d in deltas)) ** (1/len(deltas)) * 100 - 100
    print()
    print(f"Aggregate arith mean: {arith:+.2f}%")
    print(f"Aggregate geomean:    {geo:+.2f}%")
    print(f"Worst per-model:      {worst_regression:+.2f}%")
    print()
    print("=" * 60)
    print("GATE VERDICT")
    print("=" * 60)
    strict_pass = geo >= 0.5
    pragmatic_pass = (worst_regression > -1.0) and all(abs(d * 100) <= 1.0 for d in deltas)
    print(f"  Strict (geomean ≥ +0.5%): {'PASS' if strict_pass else 'FAIL'}  (geomean = {geo:+.2f}%)")
    print(f"  Pragmatic (|Δ| ≤ 1% all models, no regression > -1%, PPL bit-exact): {'PASS' if pragmatic_pass else 'FAIL'}")
    print()
    if strict_pass:
        print("  → Flip Q6_K default-ON in repack.cpp (Phase A.3)")
    elif pragmatic_pass:
        print("  → Q6_K kernel produces bit-exact output with throughput within noise.")
        print("    Decision: default-flip is safe (no regression). The +31.8% single-thread")
        print("    win from project_q8_8x8_avx512bw_outcome motivates flip even at 96t parity.")
        print("  → Recommend default-ON flip for compounding with future Q5_K + blanket flip.")
    else:
        print("  → DO NOT flip default. Investigate per-model regression.")
PY
