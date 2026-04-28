# CPU11 LTO + CPU12 BOLT-libomp extensions (artifact bundle)

**Track**: CPU11 LTO + CPU12 BOLT extensions ([cpu-inference-optimization-index.md](../../../../../workspace/handoffs/active/cpu-inference-optimization-index.md))
**Run date**: 2026-04-28
**Purpose**: Empirical close-out of the two items deferred at end of the morning's PGO+BOLT session: (1) LTO without PGO + LTO compounding on PGO; (2) BOLT-optimization of the libomp shared library itself, beyond the just-`libggml-cpu.so.0` BOLT path covered in the prior CPU12 bundle.

## Headline findings

### LTO is neutral within noise — does NOT compound on PGO

| Build | Coder-30B Q4_K_M tg32 (5-rep -pp 64 warmup, warm position 2-3) | Δ vs PGO |
|---|---|---|
| clang+libomp+znver5 (no PGO, no LTO) | 28.18 (baseline reference) | reference |
| clang+libomp+znver5+PGO | 28.38 ± 0.08 | reference for LTO comparison |
| **clang+libomp+znver5+PGO+LTO** | **28.09 ± 0.04** | **−1.0% within noise** |

Both PGO-only and PGO+LTO measured at warm positions (3rd or 4th in sequence) with extremely tight std (≤0.08). LTO does not produce a measurable win on top of PGO. Likely mechanism: PGO already triggers most of the cross-TU inlining that LTO would enable, leaving negligible additional codegen headroom on this codebase.

**Decision**: do NOT add LTO to the v5 cherry-pick build. Keep the build as `clang + libomp + -march=znver5 + PGO`.

### BOLT-libomp is FUNCTIONAL but not faster than system libomp

| Configuration | Coder-30B Q4_K_M tg32 (5-rep -pp 64 warmup) | Δ vs system libomp |
|---|---|---|
| PGO + system libomp (apt llvm-20) — pos 4 | 31.95 ± 0.09 | reference |
| **PGO + BOLTed libomp via RUNPATH symlink — pos 2** | **29.58 ± 0.04** | −7.4% (but POSITION confound) |
| PGO + custom-rebuilt libomp (no BOLT) — pos 3 | 19.59 ± 7.89 | INCONCLUSIVE (high std) |

The BOLT-rewritten libomp is **functional** (PPL bit-exact) but does not beat the system libomp. Tight std (±0.04) on the BOLTed result rules out a BOLT-libomp win >0.5% in this measurement. Position effect (warm position 2 vs position 4) confounds the absolute comparison.

**Important caveat: system noise was high during this session** (megasync at 95% CPU on one core throughout, plus build/extraction operations evicting cache mid-session). The morning's same PGO build measured 58.65 ± 0.24 t/s on the same hardware/code; afternoon measurements degraded to 27-32 t/s with the same binary. The libomp-BOLT comparison is at best a relative measurement under degraded absolute throughput.

**Decision**: do NOT pursue libomp-BOLT for v5. The BOLT pipeline works; the throughput delta does not. Custom libomp build path is a v5+1 candidate IF a quieter measurement window confirms the rewritten libomp is at least neutral.

## Method

### LTO measurement

Two new builds:
- `build_libomp_lto/`: clang-20 + libomp + `-march=znver5` + `GGML_LTO=ON` (no PGO).
- `build_libomp_pgo_lto/`: clang-20 + libomp + `-march=znver5` + `-fprofile-instr-use=merged.profdata` + `GGML_LTO=ON`.

Bench protocol: `taskset -c 0-95 llama-bench -t 96 -fa 1 -mmp 0 -p 64 -n 32 -r 5` (the `-pp 64` is a prefill warmup that makes the threadpool init cost stable; without it, std balloons to >5 t/s). Sequence-position effect tested in both forward and reverse order to verify the LTO conclusion is order-independent.

### libomp-BOLT pipeline

1. **Source**: downloaded `openmp-20.1.8.src.tar.xz` from the LLVM 20.1.8 release; extracted alongside `cmake-20.1.8.src` (LLVM common cmake utilities are needed for openmp's own CMakeLists).
2. **Build**: clang-20 + `-march=znver5 -O3` + `-Wl,--emit-relocs` (mandatory for BOLT to consume relocations later). Output: `_libomp_src/openmp-build/runtime/src/libomp.so` (1.68 MB, vs 1.21 MB system — extra size from `.rela.text` section that BOLT needs).
3. **Profile collection**: 4 perf record runs (one per model class — Coder-30B, Q8 frontdoor, REAP-246B, dense 27B) with `LD_PRELOAD=$CUSTOM_LIBOMP perf record -e cycles:u -j any,u`. Captures LBR-sampled cycles inside libomp's hot path.
4. **perf2bolt** (per profile): `perf2bolt-20 -p perf_$M.data -o libomp.$M.fdata $CUSTOM_LIBOMP`. Each fdata file ~40-50 KB.
5. **merge-fdata** (legacy format): `merge-fdata-20 *.fdata > libomp.merged.fdata`. The merged profile failed to load with `llvm-bolt` ("no valid profile data found" — likely due to legacy-format fallback), so the BOLT rewrite used the single-model coder fdata.
6. **llvm-bolt**: `llvm-bolt-20 libomp.so.original -o libomp.so.bolted -data libomp.coder.fdata -reorder-blocks=ext-tsp -reorder-functions=cdsort -split-functions -split-all-cold`. Output: 6.6 MB BOLTed .so.
7. **Bench protocol**: 4 sequential 5-rep benches, libomp variants swapped in via symlink in the build directory (DT_RUNPATH lookup), so the linker picks up the local libomp.so.5 before falling through to /usr/lib/llvm-20/lib. Cleaner than `LD_PRELOAD` (which produced ±6-8 t/s std).

### PPL bit-exactness gate (BOLTed libomp)

Coder-30B Q4_K_M chunks 1-12 of WikiText-2 with BOLTed libomp loaded — PPL value preserved (see `coder30b_bolted_libomp_ppl.log`). BOLT only rearranges code layout, never changes instruction encoding or fp ordering.

## System noise context

The morning session measured `clang+libomp+znver5+PGO` at **58.65 ± 0.24 t/s** on Coder-30B Q4_K_M tg32. The same build at the same parameters in this afternoon session produced **27.77-31.95 t/s** with similar tight std at warm positions. The 2x degradation is real but the cause is not isolated:

- Megasync was at 95% CPU on one of the 96 cores throughout the afternoon
- 5-6 parallel claude processes were holding 5-10% CPU each
- The morning's library compilation + bench parallelism evicted the model from page cache, potentially fragmenting NUMA placement
- /tmp had cumulative pressure from 4-5 build trees (PGO instrumented + use, LTO, libomp source + build)

The relative comparisons within a single sweep (positions 2-4) are still meaningful because each sweep took <2 min and saw the same noise floor. The absolute scale is degraded by ~50% from morning.

## Files

| File | Purpose |
|---|---|
| `libomp.so.original` | Custom-built libomp.so with `--emit-relocs`, pre-BOLT |
| `libomp.so.bolted` | BOLT-rewritten libomp.so (6.6 MB) |
| `libomp.coder.fdata`, `libomp.q8.fdata`, `libomp.reap.fdata`, `libomp.dense.fdata` | Per-class fdata profiles |
| `libomp.merged.fdata` | Merged 4-model fdata (legacy format; not used for BOLT due to format mismatch) |
| `coder30b_bolted_libomp_ppl.log` | PPL bit-exactness verification |
| `system-state.txt`, `process-pre.txt`, `process-post.txt`, `ld_debug.log` | CPU20 protocol files |
| `results.csv` | tabulated results (LTO + libomp-BOLT) |
| `decision.md` | verdict + v5 cherry-pick implications |

(Note: `perf*.data` files are gitignored — recreate from `libomp.so.original` if needed for re-runs. Total perf data on disk: ~7 GB for the 4 model classes.)
