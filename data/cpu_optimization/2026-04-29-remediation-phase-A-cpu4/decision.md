# Phase A — CPU4 op-coalesced barriers re-test under canonical

**Date**: 2026-04-29
**Verdict**: **NEUTRAL** (was: DECISIVE NEGATIVE −19.7% under broken OMP env). Gate criterion ≥10% gain still not met → disposition stays "don't enable for v5". Original closure framing materially wrong.

## Result

5-rep canonical (`OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active numactl --interleave=all -- taskset -c 0-95 llama-bench -t 96 -fa 1 --mmap 0 -p 0 -n 64 -r 5`) on Qwen3-Coder-30B-A3B Q4_K_M tg64:

| Arm | t/s ± std |
|---|---|
| COALESCE=0 (control) | 46.96 ± 0.16 |
| COALESCE=1 (treatment) | 47.05 ± 0.16 |
| COALESCE=0 (recheck, drift control) | 47.00 ± 0.09 |

**Δ c1 vs c0 = +0.19%** (within noise — drift between c0 measurements is comparable).

## Why the original measurement was wrong

Original Phase 1 script (`run_full.sh` from `2026-04-29-cpu4-op-coalesced-barriers-phase1/`) was missing THREE pieces of the canonical recipe:
- No `OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active`
- No `numactl --interleave=all`
- No `--mmap 0`

Under the broken-OMP regime (post-this-session diagnosis), the baseline (`COALESCE=0`) was running at degraded ~13-25 t/s with high variance. Adding barrier coalescing changed the barrier path enough to interact asymmetrically with the broken-env's sleep-wake latency, producing the -19.7% reading. With proper OMP env, the barrier path is well-behaved in both arms and coalescing is essentially a no-op for throughput on this workload.

## Disposition

- **Code stays in tree**: env-gated `GGML_BARRIER_COALESCE=1`, default OFF, MUL_MAT/MUL_MAT_ID excluded from allowlist.
- **MUL_MAT wdata race finding stands** — that's a correctness discovery independent of perf measurement. Documented in `wdata-aware-mul-mat-coalescing-design.md` handoff.
- **Original "DECISIVE NEGATIVE" framing was wrong**. Update closure to "NEUTRAL — gate not met, but no regression".
- **Future**: expanding the coalesce allowlist (e.g., adding ROPE+ATTENTION pairs) is now a cleaner exploration — the conservative allowlist is at least a no-op, not a footgun.

## Files

- `run_phaseA.sh` — measurement script with full canonical recipe
- `bench_coder_c0.log`, `bench_coder_c1.log`, `bench_coder_recheck_c0.log` — bench logs
- `phaseA_master.log` — full run output
- `decision.md` — this document
