# CPU12 BOLT — Decision

**Verdict**: **PER-ROLE OPT-IN ONLY for Coder-30B-A3B-Instruct.** BOLT adds **+2.1% on top of PGO on Coder-30B (60.54 t/s)** but is workload-sensitive — the same BOLT binary regresses Q8 frontdoor (−1.2%) and dense 27B (−0.9%) under the merged 4-model profile. Not safe for universal v5 cherry-pick. Recommend shipping a per-role BOLT binary on the dedicated Coder role only; ship the PGO binary (sibling track CPU11) as the universal v5 production base.

## Headline result

| Model | PGO baseline | + BOLT (merged 4-model profile) | Δ |
|---|---|---|---|
| Qwen3-Coder-30B-A3B Q4_K_M | 58.65 ± 0.24 (libomp_baseline 5-rep) / 59.27 ± 0.57 (3-rep this bundle) | **60.54 ± 0.20** | **+3.2% / +2.1%** |
| Qwen3.6-35B-A3B Q8_0 | 27.59 ± 0.06 | 27.27 ± 0.05 | −1.2% |
| Qwen3-Coder-REAP-246B-A35B Q4_K_M | 6.91 ± 0.02 | 6.90 ± 0.02 | −0.1% |
| Qwen3.5-27B Q8_0 | 5.76 ± 0.01 | 5.71 ± 0.01 | −0.9% |

## Why BOLT is per-role, not universal

PGO is collected during compile time and applied across the whole optimization space — universal-positive across all 4 model classes. BOLT operates at link time, only on machine-code layout (block reorder, function reorder, hot/cold split). Its win depends on whether the LBR profile matches the runtime workload:
- **Coder-30B Q4_K_M** matches the dominant profile sample weight → +2.1% gain (i-cache locality wins the most because per-thread tiles are thin and barrier-coupled).
- **Q8 frontdoor** uses different inner kernels (Q8_0 vs Q4_K dot loops); reordering for Q4_K paths penalizes its i-cache → −1.2%.
- **REAP-246B** is BW-saturated; layout wins are negligible against the structural DRAM stalls → near-neutral.
- **Dense 27B** doesn't even hit `mul_mat_id`; most of the BOLT-optimized layout is dead-code from its perspective → −0.9%.

The merged 4-model profile (best compromise) leaves Coder positive but doesn't recover the others. Per-role profiles (single-model fdata) push Coder to +2.0% but make Q8/dense worse, not better.

## Compounding stack (Coder-30B Q4_K_M tg32)

| Build | t/s | Δ vs prior |
|---|---|---|
| gcc + libgomp + no-march | 48.28 | reference (v4 production) |
| + `-march=znver5` | 50.06 | +3.7% codegen |
| + libomp runtime | 56.84 | +13.5% runtime |
| + PGO codegen | 58.65 | +3.2% codegen |
| + BOLT layout (merged 4-model) | **60.54** | +3.2% layout |

**Total compounded gain over original v4 production binary on Coder-30B: +25.4%, reaching 60.54 t/s.**

## Quality

PPL bit-exact on Coder-30B Q4_K_M wiki.test chunks 1-12: 11.1146 ± 0.62405 — byte-identical to the PGO pre-BOLT build. BOLT does not modify instruction encoding or fp ordering. Quality preserved.

## v5 cherry-pick implications

**Two-binary deployment strategy**:
1. **Universal binary (all roles except dedicated Coder)**: clang + libomp + `-march=znver5` + PGO. Shipped as `llama-server` for the worker, frontdoor, REAP, and dense roles. +1.3% to +6.6% over the gcc+libgomp baseline; PPL bit-exact; no runtime config changes.
2. **Coder-30B-A3B role binary**: PGO + BOLT (merged 4-model profile). Shipped as `llama-server-coder` (or symlinked at the role level). Adds +2.1% on top of PGO → 60.54 t/s. Requires per-deploy LBR profile collection if hardware/codebase change substantially.

If the orchestrator multiplexes Coder-30B with other models on a single binary (e.g., dynamic role swap on the same server), do NOT use BOLT — the cross-model penalty erases the Coder gain.

Build environment additions for BOLT (one-time, beyond CPU11 PGO requirements):
- `linux-tools-common linux-tools-generic` (perf with `-j any,u` LBR)
- `-Wl,--emit-relocs` linker flag in the PGO use build (already enabled in `build_libomp_pgo_use/`)
- `llvm-bolt-20`, `perf2bolt-20`, `merge-fdata-20` (already installed via `llvm-20`)

Profile collection cycle: ~30 sec perf record + ~30 sec perf2bolt + ~10 sec llvm-bolt rewrite per model class. Merged profile generation: `merge-fdata-20 *.fdata > merged.fdata`.

## Closure scope

**Closed**:
- BOLT uplift measured across all 4 production model classes (this bundle)
- Cross-model variance characterized (Coder +2.1%, Q8 −1.2%, REAP −0.1%, dense −0.9% under merged profile)
- PPL bit-exactness verified
- Per-role deployment recommendation explicit

**NOT closed (out-of-scope)**:
- Whether BOLT yields more on a Coder-only training corpus larger than the current 30 sec of perf samples
- BOLT on the libomp shared library itself (this bundle only optimizes `libggml-cpu.so.0`)
- Combined BOLT-on-PGO vs PGO-on-BOLT (we only tested the canonical PGO→BOLT order)

## Remediation reference

`~/.claude/plans/nifty-discovering-allen.md` Phase 2.1 followup. CPU12 was queued for v5+1; this bundle moves it to "executed, results landed; per-role only" status.
