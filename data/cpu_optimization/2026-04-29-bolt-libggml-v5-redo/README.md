# BOLT-libggml v5 redo — Phase 3 #1

**Date**: 2026-04-29
**Source**: morning's CPU12 BOLT-libggml +2.1% Coder result (pre-v5 build)
**Hypothesis**: BOLT-libggml on v5 PGO build delivers +2-5% on Coder via hot-block reordering
**Build**: v5 PGO at `/mnt/raid0/llm/llama.cpp-experimental/build_v5_pgo_use/`

## Methodology improvements over earlier BOLT attempt

- 60s perf record per model (vs ~10s earlier)
- 4 model classes (Coder + Q8 + REAP + dense)
- Total ~330 MB perf.data captured

## Result: NOT DEPLOYABLE

BOLT-vs-PGO comparisons on v5 PGO are noise-band; no consistent directional signal:
- Coder pp32: variance too high (PGO baseline 213→152 trial-to-trial, ~30% spread)
- Coder pp64 (alternated 3 trials): -13%, -10%, +27% — net noise
- REAP pp32 BOLT vs PGO: +0.2% (parity, expected per morning's workload-sensitivity finding)

**Note on BOLT function coverage**: even with 60s × 4 model perf records, BOLT-INFO consistently reports 4-5% function coverage with warnings "estimated to optimize 1.4-6.9x more samples needed". The hot-path is structurally narrow (mul_mat / GEMM kernels); 95% of the binary is cold. This is a structural property of the workload, not a sample-density issue.

See `decision.md` for full numbers + verdict.
