# Q6_K AVX-512BW 8x8 Kernel — Validation Findings (2026-05-04)

## Summary

Q6_K AVX-512BW 8x8 kernel (commit `529fcbd6a`, Session 17 2026-04-27, in `production-consolidated-v5`)
validated against the scalar generic across 5-model production lineup.

**Verdict: KEEP DEFAULT-OFF.** Bit-exact correctness confirmed; at 96t multi-thread the kernel is
BW-saturated and shows -0.28% aggregate / -1.01% worst-case regression vs scalar fallback.
The default-on flip is NOT justified.

## Tripwire

```
Coder-30B Q4_K_M tg32 r=5: 47.86 ± 0.36 t/s (cold-boot canonical band 47-49 t/s) ✅
```

## Phase A.1 — PPL bit-exact gate (PASS)

| Model | env=0 PPL | env=1 PPL | Δ |
|---|---|---|---|
| Coder-30B-A3B Q4_K_M | 8.2622 ± 0.27495 | 8.2622 ± 0.27495 | 0.0000 |
| gemma-4-31B-it Q4_K_M | 4359.7047 ± 353.16114 | 4359.7047 ± 353.16114 | 0.0000 |
| SuperGemma4-31B Q4_K_M | 19.6921 ± 0.86338 | 19.6921 ± 0.86338 | 0.0000 |
| Qwen3-Next-80B-A3B Q4_K_M | 4.1565 ± 0.10725 | 4.1565 ± 0.10725 | 0.0000 |
| REAP-246B-A35B Q4_K_M | 8.1396 ± 0.24168 | 8.1396 ± 0.24168 | 0.0000 |

5/5 bit-exact. Q6_K kernel produces identical output as scalar generic.

(gemma-4-31B's 4359.7 PPL is high vs typical ~6 because gemma-4 chat tuning shifts the WikiText
distribution dramatically — the relevant fact here is bit-exact between env states, which holds.)

## Phase A.2 — perf gate at 96t (FAIL strict, FAIL pragmatic)

llama-bench tg32 r=5 under canonical recipe (`numactl --interleave=all -- taskset -c 0-95`,
OMP env stack, --mmap 0, -fa 1).

| Model | env=0 t/s | env=1 t/s | Δ % | σ0% | σ1% |
|---|---|---|---|---|---|
| Coder-30B-A3B Q4_K_M | 47.366 ± 0.109 | 47.421 ± 0.231 | +0.12% | 0.23% | 0.49% |
| gemma-4-31B-it Q4_K_M | 6.849 ± 0.056 | 6.839 ± 0.068 | -0.13% | 0.81% | 0.99% |
| SuperGemma4-31B Q4_K_M | 6.937 ± 0.051 | 6.938 ± 0.024 | +0.02% | 0.73% | 0.34% |
| Qwen3-Next-80B-A3B Q4_K_M | 21.322 ± 0.129 | 21.238 ± 0.113 | -0.40% | 0.61% | 0.53% |
| **REAP-246B-A35B Q4_K_M** | **6.150 ± 0.040** | **6.087 ± 0.089** | **-1.01%** | **0.65%** | **1.47%** |

**Aggregate (geomean): -0.28%. Worst per-model: -1.01% (REAP-246B).**

Strict gate (≥ +0.5% geomean): FAIL.
Pragmatic gate (|Δ| ≤ 1% all models, PPL bit-exact): FAIL on REAP at -1.01%.

## Decision

**Keep Q6_K env-gated default-OFF.** The +31.8% single-thread win documented in
`project_q8_8x8_avx512bw_outcome` is preserved via opt-in `GGML_Q6_K_8X8_AVX=1` for
low-thread workloads. At our production 96t serving regime, the kernel is BW-saturated
and adds no value — confirming the existing memory's "+1-3% at 12-96t (BW-saturated)" finding.

## Implications for blanket Q{5,6,8}_K default-on flip

The `cpu-shape-specialized-gemv-decode.md:123` recommendation to flip the blanket default
ON once Q5_K and Q6_K both have AVX-512BW kernels was predicated on the assumption that
the kernels would deliver compounding benefit. **This data falsifies that assumption** at
our 96t production regime. The blanket flip is no longer recommended without new evidence.

Phase B (Q5_K body) and Phase C (blanket flip) in `qkernel-q5q6-default-on-flip.md` are
**de-prioritized** based on this finding. Phase D (Q4_K_M-direct ukernel) was already
gated on Phase C trigger and remains deferred.

## What this rules out vs what stays open

**Ruled out for current 96t single-instance regime**:
- Q6_K AVX-512BW kernel as a default-on perf lever
- Compounding case for blanket Q{5,6,8}_K flip

**Still open**:
- Q6_K kernel for low-thread / 1-thread workloads (env opt-in, +31.8% at 1t per Session 17)
- Q5_K body for completeness if low-thread workloads emerge
- Different attribution: per-thread BW contention, sync primitive overhead, MoE-Spec
  algorithmic spec-dec — all tracked elsewhere

## Bundle contents

- `00-tripwire-coder30b.json` — canonical recipe tripwire (47.86 ± 0.36 t/s)
- `a1-*.log` — 10 PPL runs (5 models × 2 env states)
- `a1_ppl_summary.tsv` — Phase A.1 results
- `a2-*.json` — 10 llama-bench JSON outputs
- `a2_perf_summary.tsv` — Phase A.2 results
- `a2_progress.log` — Phase A.2 timeline
- `analyze_a2.sh` — gate analyzer
- `findings.md` — this file
