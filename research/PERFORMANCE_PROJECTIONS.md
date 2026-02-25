# Orchestrator Performance Projections

**Created**: 2026-01-13
**Last Updated**: 2026-01-13
**Status**: Living document - update as research proceeds

---

## Benchmark Baseline (as of 2026-01-13)

Total benchmark data collected:
- **9,476,363 completion tokens** generated
- **232 hours** of generation time
- **11.4 t/s** average across all models/configs

> **Note**: This average is NOT representative of production performance. It includes:
> - Baseline tests (no optimization)
> - Large models (235B, 480B) at slow speeds
> - Thinking models with long chain-of-thought
> - Experimental/failed configurations

---

## Current Production Performance

| Role | Model | Optimization | Speed | Source |
|------|-------|--------------|-------|--------|
| Front Door (Tier A) | Qwen3-Coder-30B-A3B | MoE-6 | 18.3 t/s | model_registry.yaml |
| Coder Primary (Tier B1) | Qwen3-Coder-30B-A3B | MoE-6 | 18.3 t/s | model_registry.yaml |
| Coder Escalation (Tier B1) | Qwen3-Coder-53B-A3B | MoE-4 | 30.4 t/s | model_registry.yaml |
| Ingestion (Tier B2) | Qwen3-Next-80B-A3B (SSM) | MoE-2 | 11.55 t/s | model_registry.yaml |
| Architect General (Tier B3) | Qwen3-235B-A22B | MoE-4 | 6.75 t/s | model_registry.yaml |
| Architect Coding (Tier B4) | Qwen3-Coder-480B-A35B | MoE-3 | 10.3 t/s | model_registry.yaml |

**Weighted Average (typical usage)**: ~20 t/s
- ~70% interactions at Front Door/Coder (18-30 t/s)
- ~20% worker tasks (15-25 t/s)
- ~10% architect escalations (6-10 t/s)

---

## Research Pipeline Optimizations

### 1. Paged Attention (PR #18747)

| Status | Expected Impact | Methodology |
|--------|-----------------|-------------|
| Production (cherry-picked) | **+76%** on 70B+ models | Measured on 70B+ models with block_size=64 |

**Applies to**:
- Architect General (235B): 6.75 → ~11.9 t/s
- Architect Coding (480B): 10.3 → ~18.1 t/s

**Estimation Method**: Direct measurement on 70B+ models, documented in PR #18747. Block prefetching optimization measured at +76% generation speedup for memory-bound large models.

### 2. RadixAttention Prefix Cache

| Status | Expected Impact | Methodology |
|--------|-----------------|-------------|
| Complete (46/46 tests) | **40-60%** prefill reduction | 80% cache hit rate verified |

**Applies to**: All models via llama-server

**Estimation Method**:
- Cache hit rate measured at 80% on RLM workloads
- Prefill time reduction proportional to cached prefix length
- 40-60% range based on SGLang RadixAttention paper results on similar workloads

### 3. RLM/REPL Paradigm

| Status | Expected Impact | Methodology |
|--------|-----------------|-------------|
| 80% aligned with paper | Eliminates context rot | Quality improvement, not speed |

**Estimation Method**: Based on arXiv:2512.24601 - OOLONG benchmark showed +34 points (114% improvement) with RLM vs direct processing. Speed benefit comes from avoiding re-processing via prefix caching.

### 4. Vanilla TTT (Test-Time Training)

| Status | Expected Impact | Methodology |
|--------|-----------------|-------------|
| Ready for implementation | TBD | Pending Phase 1 benchmarks |

**Target**: SSM models (Qwen3-Next-80B-A3B) that cannot use speculation

**GO/NO-GO Criteria** (from handoff):
- TTT overhead (1K tokens): Target <500ms, NO-GO >10s
- Memory usage (1.5B F32): Target <25GB, NO-GO >60GB
- Quality: Any improvement = GO, Degradation = NO-GO

---

## Projected Performance (With All Optimizations)

### Projection Methodology

1. **PagedAttn**: Apply +76% to models >= 70GB (measured)
2. **RadixCache**: Apply ~20% improvement (conservative, 80% hit × 40-60% prefill reduction)
3. **TTT**: Not included until Phase 1 GO decision

### Projected Speeds

| Role | Current | + PagedAttn | + RadixCache | Combined |
|------|---------|-------------|--------------|----------|
| Front Door (30B) | 18.3 t/s | 18.3 t/s* | ~22 t/s | **~22 t/s** |
| Coder Primary (30B) | 18.3 t/s | 18.3 t/s* | ~22 t/s | **~22 t/s** |
| Architect General (235B) | 6.75 t/s | ~11.9 t/s | ~14.3 t/s | **~14 t/s** |
| Architect Coding (480B) | 10.3 t/s | ~18.1 t/s | ~21.7 t/s | **~20 t/s** |
| Ingestion SSM (80B) | 11.55 t/s | ~20.3 t/s | ~24.4 t/s | **~14 t/s**** |

*Models <39GB don't benefit from PagedAttn
**SSM may not fully benefit from RadixCache - rollback behavior unknown

### Projected Weighted Average

| Scenario | Weighted Average | vs Benchmark Baseline |
|----------|------------------|----------------------|
| Current production | ~20 t/s | 1.75x |
| + PagedAttn | ~24 t/s | 2.1x |
| + RadixCache | ~28 t/s | 2.5x |
| + TTT (if successful) | ~32 t/s | 2.8x |

---

## Update Checklist

When updating this document:

- [ ] Verify source data from `orchestration/model_registry.yaml`
- [ ] Check benchmark results in `benchmarks/results/reviews/summary.csv`
- [ ] Update "Last Updated" date
- [ ] Note measurement conditions (context length, batch size, etc.)
- [ ] Mark projections as "VERIFIED" when measured, "PROJECTED" when estimated

---

## References

| Document | Contains |
|----------|----------|
| `orchestration/model_registry.yaml` | Current production speeds |
| `benchmarks/results/reviews/summary.csv` | Benchmark data (9.5M tokens) |
| `handoffs/active/vanilla-ttt-feasibility.md` | TTT implementation plan |
| `research/radix_attention_handoff.md` | RadixCache implementation |
| `research/rlm_analysis.md` | RLM/REPL analysis |
| CLAUDE.md (paged_attention section) | PagedAttn configuration |

---

## Revision History

| Date | Change | Author |
|------|--------|--------|
| 2026-01-13 | Initial creation with baseline + projections | Claude |
