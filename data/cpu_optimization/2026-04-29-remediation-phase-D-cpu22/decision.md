# Phase D — CPU22 work-stealing re-test under canonical

**Date**: 2026-04-29
**Verdict**: **CLOSURE CONFIRMED — gate not flipped**. The original 2026-04-28 measurement was already done with the full canonical OMP stack, so this Phase is just verification. Absolute numbers are ~10% lower (cold-boot baseline gap) but disposition is identical.

## Result

5-rep canonical (`OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active GGML_EP_WORK_STEALING=0|1 numactl --interleave=all -- taskset -c 0-95 llama-bench -t 96 -fa 1 --mmap 0 -p 0 -n 64 -r 5`):

| Model | env=0 | env=1 | Δ now | Δ original (2026-04-28) | Disposition |
|---|---|---|---|---|---|
| Coder-30B Q4_K_M | 47.41 ± 0.13 | 46.99 ± 0.35 | **-0.89%** | -2.3% | Negative (smaller) |
| Next-80B Q4_K_M | 22.54 ± 0.01 | 22.58 ± 0.04 | **+0.18%** | -0.3% (NS) | Noise |
| REAP-246B Q4_K_M | 6.26 ± 0.03 | 6.24 ± 0.01 | **-0.32%** | -0.8% (NS) | Noise |

Gate (≥10% on at least 2 of 3): **NOT MET** on any model.

## Why this Phase didn't flip (and wasn't expected to)

Original CPU22 README (2026-04-28) explicitly documents:

> Proper canonical: `OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active taskset -c 0-95 numactl --interleave=all -t 96 -fa 1 -p 0 -n 64 -mmp 0 -r 5`

The OMP env stack was already applied in the original measurement. Unlike Phases A/B/C which had broken-OMP baselines and may have been poisoned, Phase D was a clean measurement from the start. Re-running just confirms the result.

## Cold-boot baseline drift

Absolute throughputs in this re-test are ~10% lower than original:
- Coder: 53.12 → 47.41 (-10.8%)
- Next-80B: 23.36 → 22.54 (-3.5%)
- REAP-246B: 6.64 → 6.26 (-5.7%)

This matches the pattern documented in `feedback_canonical_baseline_protocol.md`: cold-boot canonical reaches 90% of warmed-state references. The 2026-04-28 measurement happened after extended uptime (page cache warmed, NUMA balance migrations settled); today's re-test is post-2026-04-29-reboot at ~1 hour of uptime.

This drift is COMMON-MODE — it affects env=0 and env=1 equally, so the relative comparison (the actual gate criterion) is preserved. The closure stands.

## Closure framing (UNCHANGED from original)

> CPU22 work-stealing prototype empirically fails the binding ≥10% gate on 3 sync-bound MoE models tested. PPL bit-exact verified. Track closes honestly via test.

The original framing was already correct. Phase D verification confirms.

## Files

- `run_phaseD.sh` — measurement script
- `phaseD_master.log` — master log
- `bench_*_steal0.log`, `bench_*_steal1.log` — per-model bench logs
- `decision.md` — this document
