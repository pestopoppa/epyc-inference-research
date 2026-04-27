# CPU21 — OpenMP Runtime + Scheduling Matrix (artifact bundle)

**Track**: CPU21 — OpenMP Runtime And Scheduling Matrix ([handoff](../../../../../workspace/handoffs/active/cpu-openmp-runtime-scheduling-matrix.md))
**Run date**: 2026-04-26 evening
**Backfill date**: 2026-04-27 evening (this README + system-state.txt + process-pre/post.txt + ld_debug.log + results.csv + decision.md added retroactively per CPU20 artifact-bundle-backfill policy)

## Scope of what was actually run

This is a **partial submatrix** of the full CPU21 promised matrix. The full matrix (per the handoff) is:
- runtime: libgomp + libomp
- schedule: static, dynamic, guided
- chunk: 1, 4, 8, 16
- affinity: `OMP_PROC_BIND={false,close,spread}` × `OMP_PLACES={cores,threads}` (+ `master` as a control)
- wait policy: active, passive

**What ran on 2026-04-26**:
- runtime: libgomp ONLY (libomp not installed)
- schedule: static, dynamic, guided
- chunk: 1, 4 ONLY (chunks 8, 16 not run)
- affinity: full perm set (close/spread × cores/threads + master_cores + false + baseline_no_omp)
- wait policy: active, passive

**Honest closure scope**: "libgomp affinity submatrix exhausted (universal +3-8% deployable stack from `OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active`); libomp + chunks 8/16 PENDING in remediation Phase 2.1".

## Commands run

Model: Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf (sync-bound class proxy)
Binary: `/mnt/raid0/llm/llama.cpp-experimental/build/bin/llama-bench` at HEAD `8cb04da9d` (per pre-NUMA_MIRROR session log; current HEAD is 29a69599a but NUMA_MIRROR commits are MIRROR-flag-gated and don't affect the default build path).

### Phase A — Affinity matrix (default schedule, default wait policy)

Wrapper: `numactl --interleave=all --physcpubind=0-95 ./bin/llama-bench -m <coder30b-q4km> -t 96 -fa 1 -p 0 -n 32 -r 3 --mmap 0`. `drop_caches` between runs.

| Run | OMP_PROC_BIND | OMP_PLACES | Log |
|-----|---------------|------------|-----|
| baseline | (unset) | (unset) | `A_baseline_no_omp.log` |
| close + cores | close | cores | `A_proc_close_cores.log` |
| close + threads | close | threads | `A_proc_close_threads.log` |
| spread + cores | spread | cores | `A_proc_spread_cores.log` |
| spread + threads | spread | threads | `A_proc_spread_threads.log` |
| master + cores | master | cores | `A_proc_master_cores.log` (HUNG, killed at 6 min) |
| false | false | (unset) | `A_proc_false.log` |

### Phase B — Schedule × chunk (default affinity)

Wrapper: same as Phase A with addition of `OMP_SCHEDULE=<sched>,<chunk>`.

| Run | OMP_SCHEDULE | Log |
|-----|--------------|-----|
| static, 1 | static,1 | `B_static_chunk1.log` |
| static, 4 | static,4 | `B_static_chunk4.log` |
| dynamic, 1 | dynamic,1 | `B_dynamic_chunk1.log` |
| dynamic, 4 | dynamic,4 | `B_dynamic_chunk4.log` |
| guided, 1 | guided,1 | `B_guided_chunk1.log` |
| guided, 4 | guided,4 | `B_guided_chunk4.log` |

NOT RUN: chunks 8 and 16 across all three schedule policies (Phase 2.1 of remediation).

### Phase C — Wait policy (default affinity, default schedule)

| Run | OMP_WAIT_POLICY | Log |
|-----|-----------------|-----|
| active | active | `C_active.log` |
| passive | passive | `C_passive.log` |

### Cross-model verification

Follow-up sweep (cpu21_followup.sh) on REAP-246B Q4_K_M and Qwen3.6-35B Q8_0 with the combined best stack (`OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active`). See `followup/` subdirectory.

NOT RUN: dense/hybrid Qwen3.5/3.6-27B (Phase 2.6 of remediation closes this gap).
NOT RUN: libomp comparison on any model (Phase 2.1 of remediation).

## Files in this bundle

| File | Purpose | Source |
|---|---|---|
| `A_*.log`, `B_*.log`, `C_*.log` | raw llama-bench stdout per Phase A/B/C run | original 2026-04-26 evening run |
| `SUMMARY.md` | per-phase t/s table + best-stack recommendation | original 2026-04-26 evening run |
| `cpu21_followup.sh` | shell script for the cross-model verification sweep | original 2026-04-26 evening run |
| `followup/` | cross-model sweep raw logs | original 2026-04-26 evening run |
| `system-state.txt` | numactl + numa_balancing + THP + governor + SMT + uptime + free + hugepages | backfilled 2026-04-27 evening (current snapshot; system has not drifted from run-time state per spot-check) |
| `process-pre.txt` | pgrep snapshot showing no llama-* processes before run | backfilled 2026-04-27 evening (current snapshot used as proxy; the original 2026-04-26 evening run is closed and processes long since terminated) |
| `process-post.txt` | pgrep snapshot showing no llama-* processes after run | backfilled 2026-04-27 evening (same caveat as process-pre.txt) |
| `ld_debug.log` | LD_DEBUG=libs trace of one smoke command on the default-flags build | backfilled 2026-04-27 evening (smoke run on current `build/` binary; relevant to confirm linker resolves to experimental build, not v4 production) |
| `results.csv` | tabulated mean ± std per Phase | backfilled 2026-04-27 evening from existing `*.log` files |
| `decision.md` | explicit pass/fail/partial verdict | backfilled 2026-04-27 evening |

## Backfill caveat

system-state.txt + process-pre/post.txt + ld_debug.log are captured at backfill time (2026-04-27 evening), not at original-run time (2026-04-26 evening). The substantive system properties (NUMA topology, governor, THP, numa_balancing, SMT) have not changed between those times per project memory `feedback_numa_balancing_self_reset.md` (sysctl drifts back to `numa_balancing=1` on its own; current value is 0 confirmed in system-state.txt). The retroactive snapshot is acceptable evidence per the Artifact-bundle backfill policy in `cpu-benchmark-rigor-and-revalidation.md`.

ld_debug.log captures the linker-resolution path on the current default `build/` binary, confirming `/mnt/raid0/llm/llama.cpp-experimental/build/bin` is searched first via the binary's RUNPATH, not the system path that has v4 production builds. This is the same behavior the original run had.
