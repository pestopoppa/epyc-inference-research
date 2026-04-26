# CPU24-deeper Attribution Infra Scripts

Scripts to drill deeper into the "sync overhead = 96% of parallelism loss" finding from CPU24-narrow (perf-stat counter analysis 2026-04-26 evening). Each script targets a different attribution dimension.

## Prerequisites

- `numa_balancing=0` and `THP=always` (re-apply per session — sysctl drift is documented)
- All scripts run on REAP-246B Q4_K_M proper canonical (`numactl --interleave=all -t 96 -fa 1`)
- **Run scripts SEQUENTIALLY**, never concurrently — parallel benches contend for cores/DRAM and invalidate measurements (this was demonstrated 2026-04-26: concurrent CPU21 sweep + perf record on REAP gave 23 t/s on Coder vs alone 42 t/s)
- `perf_event_paranoid=4` requires sudo for perf access

## Execution order (recommended)

1. **`01_perfrecord_hotfunc.sh`** (~2 min wall) — system-wide hot-function profile during decode. Confirms or refutes which symbols dominate (libgomp barriers? matmul kernels? quantize?). REQUIRES sudo.
2. **`02_perCCD_perfstat.sh`** (~2 min wall) — per-CCD counter signatures. Tests if specific CCDs are hot-spotted (load imbalance) vs uniform (sync wait).
3. **`03_thread_imbalance.sh`** (~1 min wall) — pidstat thread CPU% histogram. If sync-bound, expect bimodal distribution (some threads at 100%, others stalling). No sudo required.
4. **`04_stall_attribution.sh`** (~3 min wall) — Zen frontend/backend/dispatch-token stall classification. REQUIRES sudo.

## What each script answers

### Script 01 — Hot-function profile

**Question**: where is REAP-246B spending its decode cycles?

**Hypothesis if sync-bound**: top symbols are libgomp `gomp_team_barrier_wait`, `__kmp_*` (libomp), or `pthread_*` synchronization primitives — NOT compute kernels.

**Hypothesis if compute-bound**: top symbols are `ggml_vec_dot_q4_K_q8_K`, `ggml_gemv_q4_K_8x8_q8_K`, `ggml_compute_forward_mul_mat`.

### Script 02 — Per-CCD counter signatures

**Question**: do all 4 NUMA quarters work uniformly, or is one bottlenecked?

**Hypothesis if uniform sync**: all 4 quarters show similar IPC, similar local/remote fill ratios.

**Hypothesis if load imbalance**: significant variance in IPC across quarters (e.g. one quarter at IPC 0.6, another at 0.2).

### Script 03 — Thread CPU% histogram

**Question**: are all 96 threads compute-busy or do some sit at barriers?

**Hypothesis if sync-bound**: histogram bimodal — N threads at >95%, 96-N threads in 25-75% range (waiting at barriers).

**Hypothesis if compute-bound**: histogram tight at 95%+ across all threads.

### Script 04 — Stall attribution

**Question**: of the 0.61 cycles/instruction wasted (1 - IPC=0.39), where do they go?

- High `ls_locks.spec_lock_*` → spinlock cmpxchg loops (sync overhead)
- High `de_dis_dispatch_token_stalls0.*` → backend resource exhaustion (compute pressure)
- High `de_no_dispatch_per_slot.no_ops_from_frontend` → frontend starvation
- High `de_no_dispatch_per_slot.backend_stalls` → backend wait

## Convergent vs divergent results

If scripts 01+04 both point to sync overhead AND scripts 02+03 show uniform thread/CCD work distribution → **conclusion: sync stalls dominate, action = CPU21 OpenMP runtime tuning + CPU22 dynamic balancing**.

If scripts 02+03 show high imbalance → **conclusion: load imbalance dominates, action = CPU22 dynamic balancing is highest priority**.

If scripts 01 shows compute kernels at top → **conclusion: actually compute-bound at IPC 0.39 (memory-stall-bound), action = focus on prefetching / DRAM latency hiding**.

## Outputs

- `perfrecord/perf.data` + `perf_symbols.txt` + `perf_callgraph.txt` (script 01)
- `perCCD/node{0..3}_cpu{lo}-{hi}.log` (script 02)
- `thread_imbalance/pidstat.log` + `thread_histogram.txt` (script 03)
- `stalls/{stalls,backend,frontend}.log` (script 04)

All scripts overwrite existing outputs. Move/archive between runs if needed.

## Already collected (CPU24-narrow, 2026-04-26 evening)

- `reap_canonical_perfstat.log` — REAP @ 96t baseline counters
- `q8_canonical_perfstat.log` — Q8_0 @ 96t comparison
- `reap_singlethread.log` — REAP @ 1t = 1.41 t/s (scaling efficiency 4.27×)

These establish the high-level picture: sync overhead ≈ 96% of parallelism loss; not BW-saturated. The deeper scripts above test which sync mechanism dominates and where to apply mitigations.
