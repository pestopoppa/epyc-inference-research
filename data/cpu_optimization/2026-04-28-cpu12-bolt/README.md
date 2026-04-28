# CPU12 — LLVM BOLT post-link optimization on PGO build (artifact bundle)

**Track**: CPU12 BOLT/FDO ([cpu-inference-optimization-index.md](../../../../../workspace/handoffs/active/cpu-inference-optimization-index.md))
**Run date**: 2026-04-28
**Purpose**: Empirical follow-up to CPU11 PGO. BOLT (Binary Optimization and Layout Tool, LLVM 20) is the natural next compiler-level lever — it does code layout based on real perf-counter LBR samples, after PGO has already optimized the build. This bundle measures BOLT on top of the libomp+PGO build across all four production model classes, with two profile strategies: (1) coder-only single-model profile, (2) merged 4-model profile.

## Headline finding

**BOLT is workload-sensitive, not universally positive.** The merged 4-model profile is the safest choice (best compromise), but cross-model variance is real:

| Model | Class | PGO baseline | + BOLT (coder profile) | + BOLT (merged 4-model) |
|---|---|---|---|---|
| Qwen3-Coder-30B-A3B Q4_K_M | sync-bound MoE | 59.27 ± 0.57 (3-rep) / 58.65 ± 0.24 (5-rep) | 60.48 ± 0.32 (+2.0%) | **60.54 ± 0.20 (+2.1%)** |
| Qwen3.6-35B-A3B Q8_0 | BW-bound frontdoor MoE | 27.59 ± 0.06 | 26.94 ± 0.06 (−2.4%) | 27.27 ± 0.05 (−1.2%) |
| Qwen3-Coder-REAP-246B-A35B Q4_K_M | DRAM-bound large MoE | 6.91 ± 0.02 | 7.00 ± 0.02 (+1.3%) | 6.90 ± 0.02 (−0.1%) |
| Qwen3.5-27B Q8_0 | dense/hybrid SSM-Dense | 5.76 ± 0.01 | 5.74 ± 0.01 (−0.3%) | 5.71 ± 0.01 (−0.9%) |

Recommendation: **per-role BOLT only on Coder-30B-A3B-Instruct**, where it adds +2.1% on top of PGO and reaches the **60.54 t/s** ceiling. For other roles, ship the PGO build without BOLT.

## Method

BOLT pipeline:
1. **PGO use build** (`build_libomp_pgo_use/`, hash `0bc793637`) is the BOLT input. Built with clang-20 + `-Wl,--emit-relocs` so `libggml-cpu.so.0` retains relocation info for BOLT to consume.
2. **Profile collection** (per model class): `taskset -c 0-95 perf record -e cycles:u -j any,u -o perf.data -- llama-bench -m <model> -t 96 -fa 1 -mmp 0 -p 0 -n 32 -r 1` — captures ~30 sec of LBR-sampled cycles during tg32. One perf.data per model class:
   - `perf.data` — Coder-30B Q4_K_M (498 MB)
   - `perf_q8.data` — Qwen3.6-35B Q8_0 (471 MB)
   - `perf_reap.data` — Qwen3-Coder-REAP-246B Q4_K_M (2.36 GB)
   - `perf_dense.data` — Qwen3.5-27B Q8_0 (1.09 GB)
3. **perf2bolt conversion** (per profile): `perf2bolt-20 -p perf.data -o libggml-cpu.so.0.fdata libggml-cpu.so.0` → 4 .fdata files (~150-230 KB each).
4. **fdata merge**: `merge-fdata-20 *.fdata > libggml-cpu.so.0.merged.fdata` → combined 4-model profile.
5. **BOLT rewrite** (single + merged): `llvm-bolt-20 libggml-cpu.so.0 -o libggml-cpu.so.0.bolt -data <fdata> -reorder-blocks=ext-tsp -reorder-functions=hfsort+ -split-functions -split-all-cold -dyno-stats` → optimized .so.
6. **Measurement**: same llama-bench harness, 5-rep at proper canonical for the merged-profile binary, 3-rep for the single-profile binary. PPL bit-exactness gate on Coder-30B.

## PPL bit-exactness

Coder-30B Q4_K_M chunks 1-12: BOLT-optimized build PPL = **11.1146 ± 0.62405** — byte-identical to PGO baseline. BOLT only changes code layout (block reordering, function reordering, hot/cold split); does not modify instruction encoding or fp ordering. Quality preserved.

## Why BOLT is workload-sensitive (unlike PGO)

PGO is collected once during compile time and applied to the whole binary; the optimizer can simulate the full optimization space. BOLT operates at link time on already-finalized machine code, with much less freedom. Its main wins are:
- **Block layout** (move hot blocks together → fewer i-cache misses, fewer mispredicted branches)
- **Function layout** (HFSort+ groups callee/caller pairs, shrinks i-TLB footprint)
- **Cold split** (move cold paths to a separate .so segment → hot section becomes denser)

When the BOLT profile matches the workload, all three wins fire. When it doesn't:
- Q8 frontdoor BW-saturates differently (its `mul_mat_id` hot path uses different inner kernels — Q8_0 vs Q4_K). Reordering for Q4_K paths actively *hurts* its i-cache behavior.
- REAP-246B's ~5x larger active-param footprint means most i-cache misses are already structural; reordering offers minimal improvement.
- Dense doesn't trigger `mul_mat_id` at all — most of the BOLT profile is irrelevant to it.

The merged 4-model profile partially compensates, but doesn't eliminate the cross-model penalty: **+2.1% gain on Coder vs −1.2% loss on Q8 means net negative if a workload is Q8-dominated.**

## Compounding stack (Coder-30B Q4_K_M tg32)

| Build | t/s | vs prior step | vs gcc+libgomp+no-march |
|---|---|---|---|
| gcc + libgomp + no-march (original `build/`) | 48.28 | reference | reference |
| gcc + libgomp + `-march=znver5` | 50.06 | +3.7% codegen | +3.7% |
| clang + libomp + `-march=znver5` | 56.84 | +13.5% runtime | +17.7% |
| clang + libomp + `-march=znver5` + PGO | 58.65 | +3.2% codegen | +21.5% |
| clang + libomp + `-march=znver5` + PGO + BOLT (merged) | **60.54** | +3.2% layout | **+25.4%** |

Total compounded gain over original v4 production binary on Coder-30B: **+25.4%, reaching 60.54 t/s**.

## Deployable runtime profile (REVISED)

Universal runtime stack stays: `OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active`.

**v5 cherry-pick recommendation**: ship **clang+libomp+znver5+PGO** as the universal production binary. **Per-role BOLT** as opt-in for Coder-30B-A3B-Instruct ONLY (where the gain is +2.1% to 60.54 t/s). Do NOT ship a single BOLT binary universally; the cross-model variance creates net regressions on Q8/dense workloads.

If a Coder-30B-only role is deployed (e.g., the agentic coding worker), use the BOLT-merged binary. If the role multiplexes across model classes, stick with PGO-only.

Build environment additions for BOLT (one-time):
- `apt install linux-tools-common linux-tools-generic` (for perf with LBR)
- LLVM 20 already installed (`llvm-bolt-20`, `perf2bolt-20`, `merge-fdata-20`)
- `-Wl,--emit-relocs` linker flag in the PGO use build

Total: ~10-20 min profile-and-rewrite cycle per model class.

## Files

| File | Purpose |
|---|---|
| `coder30b_pgo_baseline.log`, `q8_pgo.log`, `reap_pgo.log`, `dense_pgo.log` | PGO baselines for the 4 model classes |
| `coder30b_pgo_bolt.log`, `q8_pgo_bolt.log`, `reap_pgo_bolt.log`, `dense_pgo_bolt.log` | BOLT with coder-only profile |
| `coder30b_pgo_bolt_merged.log`, `q8_pgo_bolt_merged.log`, `reap_pgo_bolt_merged.log`, `dense_pgo_bolt_merged.log` | BOLT with merged 4-model profile (recommended config) |
| `coder30b_pgo_bolt_ppl.log` | PPL chunks 1-12 bit-exactness gate |
| `libggml-cpu.so.0.fdata` | Coder-only fdata profile |
| `libggml-cpu.so.0.q8.fdata`, `.reap.fdata`, `.dense.fdata` | per-class fdata profiles |
| `libggml-cpu.so.0.merged.fdata` | merged 4-model fdata (recommended) |
| `system-state.txt`, `process-pre.txt`, `process-post.txt`, `ld_debug.log` | CPU20 protocol files |
| `results.csv` | tabulated results |
| `decision.md` | verdict + per-role recommendation |

(Note: `perf*.data` files are gitignored — recreate from baseline binary if needed for re-runs.)
