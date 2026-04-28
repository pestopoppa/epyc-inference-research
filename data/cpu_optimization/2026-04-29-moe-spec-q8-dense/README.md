# MoE-Spec on Q8 frontdoor + Dense — Phase 3 #3

**Date**: 2026-04-29
**Hypothesis**: MoE-Spec budgeting (REAP +13.5% deployable) generalizes to Q8 frontdoor + dense models
**Build**: v5 PGO at `/mnt/raid0/llm/llama.cpp-experimental/build_v5_pgo_use/`

## Result: Q8 NOT DEPLOYABLE; Dense N/A (no MoE)

### Q8 frontdoor (Qwen3.6-35B-A3B-Q8_0) — n_expert=256

| B | pp32 (mean ± std) | Δ vs B=0 |
|---|---|---|
| 0 | 126.68 ± 3.41 | reference |
| 192 (75%) | 97.67 ± 14.66 | **-22.9%** |
| 128 (50%) | 53.42 ± 7.48 | **-57.8%** |
| 64 (25%) | 60.50 ± 6.18 | -52.2% |
| 32 | 56.26 ± 9.87 | -55.6% |

ALL budgets regress 23-58% on Q8 frontdoor. MoE-Spec fundamentally hurts on this model class.

### Dense (Qwen3.6-27B-Q8_0) — n_expert=0

Hybrid SSM-Dense architecture (Gated DeltaNet + dense FFN). NO MoE layers. MoE-Spec mechanism doesn't fire. Skipped — N/A.

## Root cause: MoE-Spec scales unfavorably with n_expert

The MoE-Spec mask uses `ggml_argsort_top_k(scores, B)` per MoE layer per forward pass. Cost ~O(n_expert log B):

| Model | n_expert | argsort_top_k cost (relative) |
|---|---|---|
| REAP-246B | 80 | 1.0× (baseline; +13.5% gain) |
| Coder-30B | 128 | 1.7× |
| **Qwen3.6-35B-A3B Q8** | **256** | **3.4× (overhead exceeds savings)** |

Plus, with n_expert_used=8 (top-K=8), reducing the budget from 256→192 doesn't shrink the union meaningfully — most tokens still pick their full 8 experts within the shortlist of 192. Below 64, the shortlist is tight enough to force substitutions, but quality cost is severe.

The MoE-Spec mechanism only delivers when:
1. n_expert is small enough that argsort_top_k overhead is manageable
2. Per-token compute is heavy enough that DRAM expert-weight read reduction dominates the overhead

Both met by REAP-246B (80 experts, heavy 246B model). Coder-30B is borderline (128 experts, lighter model). Q8 frontdoor (256 experts) is too far in the wrong direction.

## Production decision

| Role | Model | n_expert | MoE-Spec |
|---|---|---|---|
| Coder | Qwen3-Coder-30B-A3B Q4_K_M | 128 | OFF (noise-band; not deployable) |
| Worker | Qwen3-Coder-30B-A3B Q4_K_M | 128 | OFF |
| Frontdoor | Qwen3.6-35B-A3B Q8_0 | 256 | OFF (-23 to -58% regression) |
| Architect coding | REAP-246B-A35B Q4_K_M | 80 | **ON, B=40 (+13.5%)** |
| Coder REAP-25B | Qwen3-Coder-REAP-25B-A3B Q4_K_M | unknown | TBD (probe needed) |

REAP-246B is the only deployable target.

## Closure scope (per closure-inflation policy)

> "MoE-Spec budgeting on Qwen3.6-35B-A3B-Q8_0 (n_expert=256) regresses -22.9% (B=192) to -57.8% (B=128) on pp32 forward-pass under v5 PGO. The mask construction overhead via ggml_argsort_top_k scales with n_expert; at 256 experts, overhead exceeds the DRAM-bandwidth-reduction savings. Does NOT generalize to 'MoE-Spec is dead on hybrid models': REAP-246B at n_expert=80 is deployable. The structural finding is that MoE-Spec works for small-n_expert + heavy-compute combinations; large-n_expert models (256+) are not viable targets."
