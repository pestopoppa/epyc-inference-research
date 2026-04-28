# MoE-Spec Phase 1 prototype — measurement bundle

**Date**: 2026-04-28
**Branch**: `feature/cpu-ep-inter-process` HEAD `0bc793637` + uncommitted MoE-Spec patch
**Build**: `/mnt/raid0/llm/llama.cpp-experimental/build/` (gcc + libgomp, GGML_NATIVE=ON, GGML_OPENMP=ON, CMAKE_BUILD_TYPE=Release)
**Purpose**: Phase 1 falsification probe — does MoE-Spec (arXiv:2602.16052) produce a measurable verification-batch throughput gain on CPU at quality-acceptable budgets?

## Files

| File | Purpose |
|---|---|
| `decision.md` | Full Phase 1 verdict + tradeoff analysis |
| `results.csv` | Tabulated mean/std/Δ across all (model, B, prompt) combinations |
| `system-state.txt` | numactl topology + numa_balancing + THP + governor + load |
| `process-pre.txt` / `process-post.txt` | pgrep snapshots before/after benchmarks |
| `ld_debug.log` | LD_DEBUG=libs smoke run confirming library identity |
| `coder30b_pp{32,64}_B{0,32,64,96}{,_run2}.log` | Coder-30B raw bench logs |
| `reap246b_pp32_B{0,20,40,60,80}{,_run2,_gateoff}.log` | REAP-246B raw bench logs |
| `reap246b_ppl_B{0,20,40,60}.log` | REAP-246B 3-chunk PPL diagnostic for quality gate |

## Commands run

```bash
# Build (rebuild after MoE-Spec patch)
cd /mnt/raid0/llm/llama.cpp-experimental/build
cmake --build . --target llama-server llama-bench llama-perplexity -j 96

# Throughput sweep — Coder-30B Q4_K_M
export LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build/bin
M=/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf
for B in 0 96 64 32; do
  for P in 32 64; do
    LLAMA_ARG_MOE_SPEC_BUDGET=$B numactl --interleave=all -- \
      taskset -c 0-95 ./bin/llama-bench -m $M -p $P -n 0 -t 96 -fa 1 --mmap 0 -r 5
  done
done

# Throughput sweep — REAP-246B Q4_K_M (n_expert=80)
M=/mnt/raid0/llm/models/Qwen3-Coder-REAP-246B-A35B-Q4_K_M.gguf
for B in 0 60 40 20 80; do  # B=80 = gate-off (B>=n_expert)
  LLAMA_ARG_MOE_SPEC_BUDGET=$B numactl --interleave=all -- \
    taskset -c 0-95 ./bin/llama-bench -m $M -p 32 -n 0 -t 96 -fa 1 --mmap 0 -r 5
done

# PPL diagnostic — Coder-30B
for B in 0 128 96 64; do
  ./bin/llama-perplexity -m $M_CODER -f /mnt/raid0/llm/data/wiki.test.raw \
    --chunks 3 -t 96 -fa 1 --moe-spec-budget $B
done

# PPL diagnostic — REAP-246B
for B in 0 60 40 20; do
  ./bin/llama-perplexity -m $M_REAP -f /mnt/raid0/llm/data/wiki.test.raw \
    --chunks 3 -t 96 -fa 1 --moe-spec-budget $B
done
```

## Headline result

**WIN — Phase 1 mechanism gate MET on both target models.**

- Coder-30B Q4_K_M pp32: +7.3% at B=64 (50% of n_expert)
- REAP-246B Q4_K_M pp32: +15.2% at B=40 (50% of n_expert)
- B≥n_expert path produces byte-identical baseline output (gate-skip confirmed empirically)

See `decision.md` for full verdict, quality tradeoff analysis, and Phase 2 deferred items.

## Caveats

1. **pp32/pp64 throughput is the verification-batch shape**, not end-to-end spec-dec throughput. Production gain depends additionally on (acceptance-rate × verification-share). Phase 2 measures end-to-end.
2. **Forward-pass PPL drifts at B<n_expert** (Coder-30B B=64: +6.7% chunk-3 drift; REAP B=40: +23%). In spec-dec mode the verifier rejects mismatches making end-to-end output bit-exact, but acceptance rate likely drops. Paper claims 1.4% average acceptance reduction. Phase 2 measures.
3. **Build is gcc+libgomp**, NOT the v5 production clang+libomp+znver5+PGO target. Mechanism gain may compound or shrink under PGO. Phase 2 should re-validate.
4. **Existing `cparams.moe_n_expert_override`** (per-token K reduction; production uses `--override-kv qwen3moe.expert_used_count=int:4` for some Coder configs) was NOT combined in this sweep. Interaction with MoE-Spec budget is unmeasured.
5. **First-run B=0 on REAP-246B was a noisy outlier** (35.64 ± 5.77 — high std). Re-run gave 45.23 ± 0.99. Stable baseline = 45.23. The Δ vs B=0 in `results.csv` for REAP rows uses the corrected baseline.
