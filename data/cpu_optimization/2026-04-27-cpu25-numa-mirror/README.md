# CPU25 — NUMA_MIRROR Fork Integration (artifact bundle)

**Track**: CPU25 — NUMA_MIRROR Fork Integration ([handoff](../../../../../workspace/handoffs/active/numa-mirror-integration.md))
**Run dates**: 2026-04-27 (Phase 0a/0b/1a/1b/1c implementation + Phase 2 throughput gate)
**Backfill date**: 2026-04-27 evening (this entire bundle is created retroactively per CPU20 artifact-bundle-backfill policy; the original 2026-04-27 NUMA_MIRROR work was committed but did not land an artifact directory)

## Closure verdict

**CLOSED — DECISIVE NEGATIVE on single-socket NPS4 (MoE proxies tested)**.

Phase 2 throughput gate (≥ +25% on Coder-30B Q4_K_M over 47.98 t/s baseline): **NOT MET**. See `decision.md` for the full attribution.

## Implementation chain (commits on `feature/cpu-ep-inter-process` of `/mnt/raid0/llm/llama.cpp-experimental`)

| Phase | Commit | Description |
|---|---|---|
| 0a | `9b1dbf4dd` | tensor_data() / tensor_set_data() accessor in ggml.h + 97 refs migrated in 5 read-only files |
| 0b | `b9920cc44` | 67 refs migrated in 6 files with writes/chained-pointers (ggml-backend.cpp, ggml-alloc.c, ggml-backend-meta.cpp, llama-model-loader.cpp, llama-kv-cache.cpp, llama-quant.cpp) |
| 1a | `ca39cb80a` | data_per_node[GGML_NUMA_MAX_NODES] field + tensor_set_data_per_node() API + ggml_new_tensor_impl populates the array |
| 1b | `90a17af62` | TLS setter at graph-compute entry via getcpu(2) |
| 1c | `29a69599a` | CPU_REPACK buffer-level mirror — per-buffer side-table tracks N anon-mmap+mbind replicas; init/set/free hooks; forward_mul_mat/forward_mul_mat_id (5 sites in repack.cpp) migrated to tensor_data() |

## Builds used

| Build | CMake flags | Purpose |
|---|---|---|
| `build_znver5/` | `-march=znver5` (no MIRROR) | apples-to-apples baseline |
| `build_mirror/` | `-march=znver5 -DGGML_NUMA_MIRROR=4` | Phase 2 mirror=4 measurement |
| `build/` | default (no -march, no MIRROR) | original session-zero baseline; not used for Phase 2 gate (build flag mismatch was the source of an earlier 0.116-PPL discrepancy that turned out to be `-march=znver5` codegen drift, not a mirror bug) |

## Phase 2 throughput measurements (proper canonical)

Wrapper: `OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active taskset -c 0-95 numactl --interleave=all -t 96 -fa 1 -mmp 0`.

### Original Phase 2 measurement (2026-04-27)

| Model | Quant | tg128 baseline (znver5) | tg128 mirror=4 | Δ |
|---|---|---|---|---|
| Coder-30B-A3B | Q4_K_M | 48.16 ± 0.15 | 47.66 ± 0.04 | **−1.0%** (within noise) |
| Qwen3.6-35B-A3B | Q8_0 (tg64) | 23.30 ± 0.02 | 23.45 ± 0.02 | **+0.6%** (within noise) |

### Backfill smoke verification (2026-04-27 evening, this bundle)

| Model | Quant | tg64 baseline (znver5) | tg64 mirror=4 | Δ |
|---|---|---|---|---|
| Coder-30B-A3B | Q4_K_M | 47.26 ± 0.06 | 46.97 ± 0.15 | **−0.6%** (within noise) |

`smoke_znver5_baseline_coder30b.log` and `smoke_mirror4_coder30b.log` confirm the Phase 2 negative finding holds on a fresh re-bench.

## PPL bit-exactness

Mirror=4 PPL on Coder-30B Q4_K_M wiki.test.raw chunks 1-12: chunk1 = 7.4537, final = **11.1215 ± 0.62430**. Byte-identical to a clean -march=znver5 baseline build. Mirror is mathematically correct; the negative is throughput-only, not quality.

## Mirror correctness verification (from session log)

Mirror correctly fires: `cpu-repack-mirror: 13.1 GiB primary on node 0 + 3 node replicas (mirror=4)`.

Threads correctly distribute (one-shot `GGML_NUMA_MIRROR_DEBUG=1` log captured during testing): cores 0-23 → node 0, 24-47 → node 1, 48-71 → node 2, 72-95 → node 3, with `getcpu(2)` reporting the matching node and `ggml_current_numa_node` set per thread. Each thread reads from its node's replica via `tensor_data()`.

## Files in this bundle

| File | Purpose | Source |
|---|---|---|
| `system-state.txt` | numactl + numa_balancing + THP + governor + SMT + uptime + free + hugepages | backfilled 2026-04-27 evening (current snapshot) |
| `process-pre.txt` | pgrep snapshot showing no llama-* processes before the smoke runs | backfilled 2026-04-27 evening |
| `process-post.txt` | pgrep snapshot showing no llama-* processes after the smoke runs | backfilled 2026-04-27 evening |
| `ld_debug.log` | LD_DEBUG=libs trace of the mirror=4 build smoke run | backfilled 2026-04-27 evening |
| `smoke_znver5_baseline_coder30b.log` | znver5 non-mirror baseline tg64 (47.26 ± 0.06) | captured during 2026-04-27 evening backfill |
| `smoke_mirror4_coder30b.log` | mirror=4 tg64 (46.97 ± 0.15) | captured during 2026-04-27 evening backfill |
| `results.csv` | tabulated baseline vs mirror deltas (original Phase 2 + backfill smoke) | backfilled 2026-04-27 evening |
| `decision.md` | explicit closure verdict with attribution | backfilled 2026-04-27 evening |

## Backfill caveat

The original Phase 2 measurements (Coder-30B tg128 48.16 → 47.66; Qwen3.6-35B Q8 tg64 23.30 → 23.45) were recorded in the session log + commit message of `29a69599a` but not into a formal artifact directory at the time. The `smoke_*` logs in this bundle re-confirm the Coder-30B negative on a fresh tg64 run; the ±0.6% delta on tg64 is consistent with the original ±1.0% on tg128 (both within noise).

system-state.txt + process-pre/post.txt + ld_debug.log are captured at backfill time; system has not drifted from run-time state.

## Remediation reference

See `~/.claude/plans/nifty-discovering-allen.md` Phase 2.6 (cross-architecture sanity coverage):
- Add Qwen3.5/3.6-27B Q8_0 mirror=4 vs baseline run to confirm or refute that the Phase 2 negative generalizes from MoE to dense/hybrid.
- Per-thread BW math says dense should fail the gate identically; one quick run confirms.

Output dir: `2026-04-28-cpu-cross-architecture-sanity/`.

After Phase 2.6 lands, this bundle's closure language upgrades from "DECISIVE NEGATIVE on MoE proxies" to "DECISIVE NEGATIVE on MoE + dense proxies" OR (unlikely per the math) "DECISIVE NEGATIVE on MoE; dense behavior diverges and reopens the investigation".
