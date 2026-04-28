# Slot-Promotion Phase 1.0 — Empirical confirmation on hybrid Delta Net

**Date**: 2026-04-29
**Target**: Qwen3.6-35B-A3B-Q8_0 (qwen35moe arch = hybrid Delta Net)
**Drafter**: Qwen3-1.7B-Q8_0 (vocab-compatible)
**Build**: v5 PGO at `/mnt/raid0/llm/llama.cpp-experimental/build_v5_pgo_use/`
**Gate test**: ≥30% acceptance + ≥0% throughput vs p_split=0 linear baseline

## Result: GATE MET

| Shape | t/s | accept% | Δ |
|---|---|---|---|
| linear (p_split=0) | 6.80 ± 0.20 | 100.0% | reference |
| tree (p_split=0.05) | 6.88 ± 0.30 | 100.0% | +1.2% (within noise) |

DySpec heap-spec runs to completion on hybrid Delta Net via existing `llama_memory_seq_cp` infrastructure. The 6 closed SSM-hybrid handoffs' "spec-dec is dead on Delta Net hybrids" assumption is falsified empirically.

## What's NOT yet tested

- DFlash-style NUMA-parallel candidate verification (Phase 1.1, the actually-new mechanism intake-490 advocates) — ~50-100 LOC server-side scheduler change, gated on this Phase 1.0 GO
- Production-realistic acceptance rate on harder coding prompts (current 100% is greedy-temp + drafter-aligned artifact)
- Bit-exact verification at higher temperature

## Caveats

- Phase 1.0 originally targeted Qwen3.5-35B-A3B-MTP-Q4_K_M but that file has `ssm_conv1d` tensors our v5 PGO build doesn't recognize. Substituted Qwen3.6-35B-A3B-Q8_0 (same architecture handler). Q8 vs Q4 doesn't change the structural conclusion (does heap-spec work on hybrid?).
- 100% acceptance rate is greedy-temp + drafter-alignment artifact, not realistic production behavior.
- rep0 of each (linear, tree) cell lost to server warmup race; preserved 2/3 reps per config.
- Megasync noise floor (~100% on 1 core) during measurement window.
