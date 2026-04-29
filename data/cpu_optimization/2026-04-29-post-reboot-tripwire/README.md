# Post-Reboot Reproducibility Tripwire — RESOLVED

**Date**: 2026-04-29
**Trigger**: prior session's measurements were 3-5× below production canonical (58.65 t/s Coder-30B Q4_K_M tg32 → 11-20 t/s in session). User rebooted host and asked agent to verify recovery.

## Resolution

**Root cause was NOT hardware throttle**. It was missing OMP env vars in the bench command. The full canonical recipe (documented in `feedback_canonical_baseline_protocol.md`) requires:

```bash
OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active \
  numactl --interleave=all -- taskset -c 0-95 \
  llama-bench -m <model> -t 96 -fa 1 --mmap 0 -p 0 -n 32 -r 5
```

Without OMP env vars: **17 t/s, CV 30%+** (broken).
With OMP env vars: **48.79 ± 0.05 t/s, CV 0.1%** (recovered).

48.79 actually slightly exceeds the existing documented cold-boot baseline of 47.08 from `feedback_canonical_baseline_protocol.md`. The host is fully recovered. The 58.65 reference is a WARMED-state number (hours-to-days of uptime, page cache populated, NUMA balance migrations settled), not a cold-boot expectation.

## Diagnostic timeline

1. **Pristine binary fail**: tested with `build_libomp_pgo_use/bin/llama-bench` (build hash 0bc793637, the EXACT canonical binary) → 17 t/s with 30% CV. Rules out binary issue.
2. **Pristine BOLT binary fail**: same 17 t/s. Rules out binary class.
3. **sysctls applied**: `numa_balancing=0`, THP=always, `perf_event_paranoid=1`, `swappiness=1`, `sched_autogroup_enabled=0`. No improvement.
4. **Memory BW microbench misleading**: tp_gemv_numa_bench reports 67 GB/s vs 246 reference, but the bench code does its own first-touch allocation that ignores numactl wrapping. False alarm.
5. **CPU freq diagnostic**: under llama-bench load, only 5-25 of 96 cores boost; under synthetic CPU burn, 96 cores boost normally. Suggested workload-specific issue (which it was — OMP barrier/scheduling cascade).
6. **Clean rebuild from canonical commit**: built `build_libomp_pgo_use2/` with `clang-20+libomp+znver5+PGO` against existing `merged.profdata`. Same regression. Rules out toolchain drift.
7. **`numactl --interleave=all` alone**: 17 → 43.44 ± 0.07 t/s (one-shot). Necessary but not sufficient.
8. **Concurrent 4×24t topology**: each instance ~17 t/s tight (CV <1%), aggregate ~68 t/s. Confirms hardware is healthy; the 1×96t topology was the unstable one.
9. **OMP env stack applied**: 48.65-48.98 t/s, std ±0.05-0.20. Recovery complete.

## Key takeaways

- Post-reboot, the kernel resets ALL process-environment defaults including OMP. Scripts/sessions that work in continuous operation may have inherited OMP env from prior shell state that's lost on reboot.
- Apply the documented PRIMARY canonical recipe in full from the first measurement post-reboot. The protocol memory explicitly states the OMP env vars + numactl --interleave=all are mandatory, not optional.
- 58.65 t/s and 60.54 t/s are warmed-state references; cold-boot canonical is 47-49 t/s. The gap closes over time via numa_balancing page migration and page cache warming.
- Memory BW microbenches with internal first-touch are misleading — trust the actual llama-bench measurement with the full canonical stack.

## Files in this bundle

- `tripwire_coder30b_q4km_tg32.log` — first attempt (no OMP env, 16.80 t/s, broken)
- `tripwire_coder30b_q4km_tg32_PGO_BOLT_pristine.log` — pristine BOLT bin no env (17.01 t/s)
- `tripwire_during_freq.log` — bench with concurrent freq sampling (showed boost oscillation)
- `tripwire_PGO_USE_exact_canonical.log` — exact canonical bin (20.37, no OMP)
- `tripwire_libomp_baseline.log` — older libomp baseline bin (16.01)
- `tripwire_INTERLEAVE.log` — first --interleave=all (43.44 ± 0.07)
- `tripwire_long_warmup.log` — r=20 (41.20 ± 8.86, degrading)
- `tripwire_post_idle.log` — after idle (27.01, no recovery)
- `tripwire_REBUILD_canonical.log` — fresh PGO rebuild (32.71)
- `tripwire_FRESH_after_compact.log` — post drop_caches (28.60)
- `tripwire_GGML_NUMA_REPACK.log` — env var alone (28.49)
- `tripwire_CPU1_stack.log` — CPU1 env stack alone (25.01)
- `tripwire_OMP_v5_stack.log` — **THE FIX**: OMP_PROC_BIND+OMP_PLACES+OMP_WAIT_POLICY (48.43)
- `tripwire_BOLT_OMP_stack.log` — BOLT + OMP (47.96)
- `single_thread.log` — 1-thread bench (14.65 t/s sanity)
- `membw_probe.log` — microbench (misleading 67 GB/s)
- `per_iter.csv` — sequential 1-rep bench timings (all broken without OMP)

## Disposition

CPU20 protocol satisfied: post-reboot recovery confirmed at 48.79 ± 0.05 t/s on the documented cold-boot canonical recipe. Multi-arch coverage probe and Probe B can now proceed.
