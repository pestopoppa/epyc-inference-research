# K-Value Optimization Analysis

**Date**: 2026-01-16
**Status**: Analysis complete, empirical testing pending

## Summary

Analyzed 76 model+draft combinations across K={4,8,16,24}. Found that **24% of models** (18) show increasing performance at K=24 and would benefit from testing K=32 and K=48.

## Key Findings

### Distribution of Optimal K Values

| Category | Count | Percentage |
|----------|-------|------------|
| K=24 optimal (still increasing) | 18 | 24% |
| K<=8 optimal | 47 | 62% |
| K=16 optimal | 11 | 14% |

### Pattern: Model Size Determines Optimal K

**Large dense models (70B+)** - ALL show monotonically increasing performance:
- The target model inference is slow (2-5 t/s baseline)
- Draft overhead is negligible relative to target inference time
- Higher K = more tokens per slow target call = better performance

**MoE models** - ALL peak at K=8:
- Target inference already fast due to sparse activation
- Draft overhead dominates quickly at higher K

**Small models (<32B)** - Peak at K=4 or K=8:
- Target inference already fast
- High K causes draft overhead to exceed gains

## Data: Models Where K=24 Is Best (Should Test K>24)

```
general_qwen2_5_7b + coder_0_5b:     K4=22.8 → K24=46.6 t/s  (2x gain from K4→K24!)
worker_summarize + qwen25_coder:    K4=5.9  → K24=21.3 t/s  (3.6x gain!)
ingest_qwen2_5_coder_32b + qwen25:  K4=9.5  → K24=17.0 t/s
Llama-3.1-70B + PARD-1B-Q8:         K4=4.7  → K24=10.3 t/s
Qwen2.5-Math-72B + qwen25:          K4=4.1  → K24=6.9 t/s
Qwen2.5-72B + qwen25:               K4=3.0  → K24=5.7 t/s
```

## Data: MoE Models (K=8 is Optimal)

```
Qwen3-Coder-30B + 0.75B:  K8=31.4 → K16=23.9 → K24=14.0 (DECREASING)
coder_primary + 0.75B:    K8=29.7 → K16=26.0 → K24=22.3 (DECREASING)
ingest_qwen3_coder_30b:   K8=31.9 → K16=27.0 → K24=24.4 (DECREASING)
```

## Recommendations

### For Future Benchmarks

1. **70B+ dense models**: Add K=32 and K=48 to test range
2. **MoE models**: Only test K={4,8} - higher K wastes time
3. **Small dense (<32B)**: Only test K={4,8,16}

### Priority Models for K>24 Testing

When model slot is available, test these first:

1. `general_qwen2_5_7b + coder_0_5b` (K=32, K=48)
2. `worker_summarize + qwen25_coder` (K=32, K=48)
3. `ingest_qwen2_5_coder_32b + qwen25_coder` (K=32)
4. `Llama-3.1-70B + PARD-1B` (K=32)

### Test Command Template

```bash
timeout 120 numactl --interleave=all /mnt/raid0/llm/llama.cpp/build/bin/llama-speculative \
  -m /path/to/target.gguf \
  -md /path/to/draft.gguf \
  --draft-max 32 -t 96 -n 128 --no-display-prompt \
  -p "Test prompt"
```

## Theoretical Basis

Optimal K = f(target_inference_time / draft_inference_time)

When target model is slow (large dense models), higher K amortizes the target inference cost over more speculated tokens. When target model is fast (MoE, small models), the draft model overhead dominates at high K.

## Open Questions

1. Does llama.cpp support K>48? (No documented limit)
2. Is there a theoretical upper bound based on acceptance rate?
3. Should we implement adaptive K search in the benchmark system?

## Status Update (2026-01-16)

**K=32 and K=48 now included in benchmark scripts:**
- `run_draft_discovery.sh` updated: `K_VALUES_FULL=(4 8 16 24 32 48)`
- These values will be tested automatically in full mode

## Next Steps

- [x] Add K=32/48 to benchmark scripts
- [ ] Run full draft discovery to collect K=32/48 data
- [ ] Consider implementing model-class-based K ranges in benchmark config
