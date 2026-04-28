# MAB Tree-Shape Selector — Phase 0 falsification probe

**Date**: 2026-04-29
**Phase**: 0 (falsification — single fixed paper-shape vs linear baseline)
**Branch**: `feature/cpu-ep-inter-process` HEAD `0c8d05597`
**Build**: v5 PGO at `/mnt/raid0/llm/llama.cpp-experimental/build_v5_pgo_use/`
**Source**: intake-491 (Mamba Drafters, EMNLP'25 Findings) §3.2 + `mab-tree-shape-selector.md`

## Hypothesis

The paper reports MAB tree-shape selector achieving sequential 112.69 → MAB-optimized 128.21 t/s (+13.7%) on Pythia-6.9B at temp=0. Phase 0 tests whether the underlying tree-spec mechanism translates to CPU heap-spec under v5 PGO build on Coder-30B Q4_K_M and REAP-246B Q4_K_M.

## Phase 0 GO/NO-GO

GO iff: ≥0% on at least one of (Coder, REAP) at p_split=0.05 vs p_split=0 baseline, BOTH pp32 forward-pass AND end-to-end llama-server.

## Result: NO-GO

End-to-end Coder regresses -18% mean (high variance), REAP +1.4% within noise. Tree at temp=0 produces BIT-IDENTICAL outputs to linear (verifier collapses to greedy path) but adds wasted compute on non-greedy branches.

See `decision.md` for full verdict + scoped closure language.

## Methodology

- 5-rep proper canonical: `taskset -c 0-95 -t 96 -fa 1 --mmap 0` + `numactl --interleave=all`
- pp32 baseline via llama-bench (target forward-pass; identical regardless of p_split)
- End-to-end via llama-server: 3 prompts × 3 reps × 2 configs (linear vs tree) × 2 models = ~12 cells. **rep0 of every cell missing** due to server warmup race despite 60s post-`/health=ok` sleep — preserved 2 reps per cell.
- p_split passed via env var `LLAMA_ARG_DRAFT_P_SPLIT` (CLI flag is restricted to LLAMA_EXAMPLE_SPECULATIVE only; llama-server rejects --draft-p-split).
- Megasync at ~110% on 1 core during measurement window (consistent noise floor across all cells).

## Files

- `decision.md` — NO-GO verdict + closure scope
- `system-state.txt` — environment snapshot
- `process-pre.txt`, `process-post.txt` — pgrep snapshots
- `ld_debug.txt` — LD_DEBUG=files capture
- `results.csv` — per-cell measurements
- `pp32_coder.log`, `pp32_reap.log` — pp32 forward-pass baselines
- `srv_*.log` — llama-server stdouts (rep0 timeouts visible here)
- `comp_*_rep[12].json` — per-completion timings + accept rates (rep1 + rep2 only; rep0 dropped)
