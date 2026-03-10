# SpecExec Verification Profile — EPYC 9655 Empirical Results

**Date**: 2026-03-10
**Hardware**: AMD EPYC 9655 (96 cores, 2 NUMA nodes), 768 GB DDR5-5600
**Software**: llama.cpp (custom fork, build 8208, commit 6e49ca1ae)
**Feeds into**: HSD (hierarchical self-speculation), tree-speculation-numa-drafting

## Thesis

> On bandwidth-bound hardware, verifying N tokens costs approximately the same as verifying 1 token (SpecExec, arXiv:2406.02532).

If confirmed on EPYC 9655, this implies that tree-structured speculation with hundreds of nodes would have near-zero verification overhead vs linear speculation, unlocking 5-9x additional speedup beyond current production settings.

## Phase 1 — Batch Verification Latency Curve

**Method**: `llama-bench -p <batch_sizes> -n 0` across 5 target models, 10 batch sizes (1-512), 3 repetitions, two NUMA modes (distribute, isolate).

**Script**: `scripts/benchmark/profile_verification_cost.sh phase1`
**Data**: `data/specexec/phase1_<model>_<numa>.csv`
**Plots**: `data/specexec/plots/phase1_verification_latency.png`

### Results — Processing Time (ms per batch, NUMA distribute)

```
┌───────────────────────────────┬──────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
│ Model                         │ Size │   1    │   2    │   4    │   8    │  16    │  32    │  64    │  128   │  256   │  512   │ 64/1x  │
├───────────────────────────────┼──────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤
│ Qwen3.5-27B-Q4_K_M           │ 16GB │  240.5 │  451.8 │  496.4 │  487.7 │  524.5 │  666.8 │  974.7 │ 1635.1 │ 3238.2 │ 6128.5 │  4.05x │
│ Qwen2.5-Coder-32B-Q4_K_M     │ 20GB │  142.0 │  233.6 │  224.8 │  293.3 │  297.2 │  525.8 │  704.1 │ 1184.0 │ 2107.3 │ 3966.5 │  4.96x │
│ Qwen3.5-9B-Q4_K_M            │ 5.3GB│   82.2 │  162.1 │  180.6 │  172.7 │  211.6 │  293.7 │  360.5 │  556.5 │ 1055.3 │ 2019.6 │  4.39x │
│ Qwen2.5-7B-Instruct-f16      │ 15GB │  151.3 │  179.7 │  184.1 │  181.6 │  191.3 │  216.6 │  255.2 │  439.8 │  772.2 │ 1347.3 │  1.69x │
│ Qwen3.5-0.8B-Q8_0            │ 775MB│   94.6 │   52.8 │   60.0 │   53.0 │   58.5 │   68.4 │   91.8 │  144.2 │  260.8 │  500.6 │  0.97x │
└───────────────────────────────┴──────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘
```

**64/1x** = ratio of batch-64 time to batch-1 time. If SpecExec thesis holds, this should be close to 1.0.

### Interpretation

The SpecExec thesis of near-flat verification cost is **not confirmed** for most models on this hardware:

- **Qwen2.5-7B-f16 (15 GB)**: Best case — 1.69x at N=64. Near-flat up to ~32 tokens, then linear growth. This f16 model is the most bandwidth-bound (15 GB weights, no quantization compute overhead).
- **Qwen3.5-0.8B-Q8_0 (775 MB)**: 0.97x at N=64 — actually *faster* for batches than single tokens. This tiny model is fully compute-bound; batch processing amortizes fixed overhead.
- **27B, 32B, 9B Q4_K_M models**: 4-5x at N=64. Verification cost scales roughly linearly with batch size. The dequantization compute from Q4_K_M adds per-token cost that scales with batch size.

**Key insight**: The Q4_K_M quantization adds significant per-token compute that prevents the bandwidth-bound regime the SpecExec paper assumes (which targets GPU HBM bandwidth). On CPU with DDR5, dequantization cost is non-trivial. The f16 model (no dequant) shows the flattest curve.

### NUMA Impact

```
┌───────────────────────────────┬──────────────────────────────────────────────────────┐
│ Model                         │ Isolate vs Distribute overhead                       │
│                               │   N=1       N=16      N=64     N=256     N=512       │
├───────────────────────────────┼──────────────────────────────────────────────────────┤
│ Qwen3.5-27B-Q4_K_M           │  +75.6%     +6.0%     +8.2%    +13.5%    +17.4%     │
│ Qwen2.5-Coder-32B-Q4_K_M     │  +88.6%     -1.7%    +19.0%    +23.7%    +30.6%     │
│ Qwen3.5-9B-Q4_K_M            │  +93.7%     -6.0%     -3.0%     +8.0%    +12.1%     │
│ Qwen2.5-7B-Instruct-f16      │  +29.9%     +2.9%     +0.4%     +5.0%    +11.2%     │
│ Qwen3.5-0.8B-Q8_0            │  +24.5%   +132.4%     -7.0%     +1.3%     +1.2%     │
└───────────────────────────────┴──────────────────────────────────────────────────────┘
```

NUMA `distribute` is dramatically better for single-token processing (75-94% faster for large models). The gap narrows at larger batch sizes. For small models (0.8B), NUMA mode has minimal impact. **Production should always use `--numa distribute`.**

## Phase 2 — Draft Model Cost Profiling

**Method**: `llama-bench -p 0 -n 128` for each draft model, NUMA distribute, 3 repetitions.

**Script**: `scripts/benchmark/profile_verification_cost.sh phase2`
**Data**: `data/specexec/phase2_draft_costs.csv`
**Plots**: `data/specexec/plots/phase2_draft_costs.png`

### Results

```
┌──────────────────────────────────────┬──────┬──────────┬────────────┐
│ Draft Model                          │ Size │ Gen t/s  │ ms/token   │
├──────────────────────────────────────┼──────┼──────────┼────────────┤
│ Qwen2.5-Coder-0.5B-Q8_0             │ 507M │    185.4 │       5.39 │
│ Qwen3-Coder-0.75B-Q4_0              │ 448M │    181.0 │       5.52 │
│ Qwen3-0.6B-Q8_0                     │ 768M │    129.0 │       7.75 │
│ Qwen2.5-0.5B-Instruct-f16           │ 949M │    106.7 │       9.37 │
│ Gemma-3-1B-IT-Q8_0                  │ 1.0G │    100.0 │      10.00 │
│ DeepSeek-R1-Distill-Qwen-1.5B-Q8_0  │ 1.8G │     63.9 │      15.64 │
│ Qwen3.5-0.8B-Q4_0                   │ 484M │     51.7 │      19.33 │
│ Llama-3.2-1B-Instruct-f16           │ 2.4G │     48.9 │      20.45 │
│ Qwen3.5-0.8B-Q8_0                   │ 775M │     44.2 │      22.64 │
└──────────────────────────────────────┴──────┴──────────┴────────────┘
```

**Surprise finding**: Qwen3.5-0.8B is the **slowest** draft model despite being small (44-52 t/s). The Qwen3.5 architecture (752M actual params) has higher per-token overhead than Qwen2.5 or Qwen3 models of similar size. The fastest drafters are Qwen2.5-Coder-0.5B (185 t/s) and Qwen3-Coder-0.75B (181 t/s).

### Critical Ratios (T_draft / T_target_verify_1)

The ratio tells us: how many draft tokens can we generate in the time it takes to verify one token on the target?

```
┌──────────────────────────────────────┬───────────────────────────────┬────────┬───────┐
│ Draft                                │ Target                        │ Ratio  │ Max K │
├──────────────────────────────────────┼───────────────────────────────┼────────┼───────┤
│ Qwen2.5-Coder-0.5B-Q8_0             │ Qwen2.5-Coder-32B-Q4_K_M     │ 0.0380 │    26 │
│ Qwen2.5-0.5B-Instruct-f16           │ Qwen2.5-7B-Instruct-f16      │ 0.0619 │    16 │
│ Qwen3-Coder-0.75B-Q4_0              │ Qwen3.5-9B-Q4_K_M            │ 0.0672 │    14 │
│ Qwen3.5-0.8B-Q8_0                   │ Qwen3.5-27B-Q4_K_M           │ 0.0941 │    10 │
│ Qwen3.5-0.8B-Q8_0                   │ Qwen3.5-9B-Q4_K_M            │ 0.2754 │     3 │
└──────────────────────────────────────┴───────────────────────────────┴────────┴───────┘
```

**Max K** = floor(1/ratio) — maximum draft tokens affordable before drafting cost exceeds one target verification. Higher is better.

The Qwen2.5-Coder pair (0.5B → 32B) has the best ratio: 26 draft tokens per verification cycle. The Qwen3.5 pairs are penalized by the slow 0.8B drafter.

## Phase 3 — Large-K Linear Speculation

**Method**: `llama-server` with target + draft model, `--draft-max K` for K in {16, 32, 64, 128, 256}. 20 prompts per config from question_pool (coder + thinking suites), temperature=0, max_tokens=512.

**Script**: `scripts/benchmark/bench_largek_speculation.sh`
**Data**: `data/specexec/phase3_<pair>_k<K>.csv`
**Plots**: `data/specexec/plots/phase3_largek_throughput.png`

### Results

```
┌──────────────────────────┬─────┬──────────┬────────────┐
│ Pair                     │  K  │ Avg t/s  │ Accept %   │
├──────────────────────────┼─────┼──────────┼────────────┤
│ Qwen2.5-7B + 0.5B       │  16 │    42.0  │     91%    │
│                          │  32 │    43.5  │     89%    │
│                          │  64 │    43.0  │     88%    │
│                          │ 128 │    42.7  │     87%    │
│                          │ 256 │    43.3  │     86%    │
├──────────────────────────┼─────┼──────────┼────────────┤
│ Qwen2.5-Coder-32B + 0.5B│  16 │    16.8  │     74%    │
│                          │  32 │    17.1  │     73%    │
│                          │  64 │    16.8  │     72%    │
│                          │ 128 │    17.2  │     72%    │
│                          │ 256 │    17.1  │     72%    │
├──────────────────────────┼─────┼──────────┼────────────┤
│ Qwen3.5-27B + 0.8B      │  16 │     6.9  │     73%    │
│                          │  32 │     6.4  │     70%    │
│                          │  64 │     6.7  │     67%    │
│                          │ 128 │     6.3  │     67%    │
│                          │ 256 │     5.2  │     57%    │
├──────────────────────────┼─────┼──────────┼────────────┤
│ Qwen3.5-9B + 0.8B       │  16 │    11.9  │     78%    │
│                          │  32 │    11.7  │     75%    │
│                          │  64 │    11.8  │     76%    │
│                          │ 128 │    11.9  │     77%    │
│                          │ 256 │    11.7  │     74%    │
└──────────────────────────┴─────┴──────────┴────────────┘
```

### Interpretation

**Throughput is flat from K=16 to K=256 for all pairs.** Increasing `--draft-max` beyond 16 provides zero benefit with linear speculation:

- **Qwen2.5-7B + 0.5B**: Excellent pair — 42-43 t/s regardless of K, 86-91% acceptance. The high acceptance rate means most tokens are already accepted at K=16.
- **Qwen2.5-Coder-32B + 0.5B**: Solid 17 t/s, flat. Note: ~50% of prompts got only 2 tokens with 0% acceptance (draft mismatch on first token for code completion prompts).
- **Qwen3.5-27B + 0.8B**: Only pair showing degradation — drops from 6.9 to 5.2 t/s at K=256 as acceptance rate falls from 73% to 57%. The slow drafter (22.6 ms/tok) wastes time generating tokens that get rejected.
- **Qwen3.5-9B + 0.8B**: Remarkably stable 11.7-11.9 t/s across all K values.

**Root cause of flat throughput**: With linear speculation, acceptance rate decays geometrically with sequence length. At K=16 with 75% per-token acceptance, expected accepted tokens = ~4. At K=256, still ~4 because probability of accepting beyond ~20 tokens is negligible. The extra draft tokens are wasted compute.

This is exactly why tree speculation matters — instead of one long sequence, a tree can explore multiple short branches, each with high acceptance probability.

## Phase 4 — HSD Integration Design

### Current Verification Path in llama.cpp

1. **`common_sampler_sample_and_accept_n()`** (`common/sampling.cpp:521-548`):
   - Iterates linearly: samples target logits at each position, accepts if matches draft
   - Stops at first mismatch — returns accepted tokens + one resampled token
   - **Linear-only**: no tree structure support

2. **Call site** (`server-context.cpp:2891`):
   - Draft tokens from `slot.draft` (populated by draft model)
   - Result determines how many tokens to accept

### HSD Modification Points

| Component | Change | LOC estimate |
|-----------|--------|-------------|
| `common_sampler_sample_and_accept_n` | Add tree-aware verification: accept along branches, resample at branch points | ~80-120 |
| `server-context.cpp` (call site) | Pass tree structure instead of linear draft sequence | ~30-50 |
| New: `common_sampler_sample_tree()` | Tree verification: DFS/BFS over candidate tree, verify each branch | ~100-150 |
| New: tree data structure | `struct draft_tree { vector<node> nodes; }` with parent pointers | ~50 |

**Total estimated**: ~260-370 LOC

### Expected Gain (Bounded by Phase 1-3 Data)

Phase 1 shows verification cost grows ~4-5x from N=1 to N=64 for Q4_K_M models — **not** near-flat. This means tree verification of 64 nodes costs ~4-5x a single token, not ~1x as SpecExec predicts. However:

- A tree of 64 candidates with branching factor 2-4 would accept ~8-12 tokens per verification (vs ~4 for linear K=64)
- Even with 4-5x verification cost, the 2-3x higher acceptance yield may net positive
- The f16 model (1.69x at N=64) would benefit most from tree speculation

**Estimated net gain**: 1.5-2.5x throughput improvement over current linear K=16, primarily for f16 and larger Q4_K_M targets.

### Risk Assessment

- **Low risk**: No changes to model weights or inference semantics
- **Medium complexity**: Tree KV-cache management in llama.cpp requires careful index mapping
- **Dependency**: Requires llama.cpp tree attention support (not yet implemented upstream)
- **Mitigation**: Can prototype with linear multi-path (run top-2 branches sequentially) before full tree

## Conclusions

1. **SpecExec thesis partially refuted on EPYC 9655**: Verification cost scales ~4-5x from N=1 to N=64 for Q4_K_M models. Only the f16 model (Qwen2.5-7B) shows near-flat behavior (1.69x at N=64). The dequantization compute overhead of Q4_K_M quantization prevents the pure bandwidth-bound regime SpecExec assumes.

2. **Inflection point**: Models >5 GB in f16 format are bandwidth-bound (flat verification). Q4_K_M models of any size show linear verification scaling due to dequant compute. Sub-1 GB models are fully compute-bound.

3. **Optimal K for linear speculation is 16**: Increasing `--draft-max` from 16 to 256 provides zero throughput benefit (and slight degradation for some pairs). The acceptance rate decay of linear sequences neutralizes any verification cost savings.

4. **Tree speculation is worth pursuing but with tempered expectations**: Expected 1.5-2.5x gain over linear K=16, not the 5-9x originally hypothesized. The primary benefit comes from higher acceptance yield per verification cycle, not from near-free verification. **Prioritize f16 targets** where verification is genuinely near-flat.

5. **Draft model selection matters more than K**: The Qwen2.5 pairs (0.5B draft at 185 t/s, 91% acceptance) dramatically outperform Qwen3.5 pairs (0.8B draft at 44 t/s, 73% acceptance). Investing in faster/better-matched drafters yields more gain than tree speculation.

## Reproduction

```bash
# Phase 1+2 (~8 minutes total)
cd /mnt/raid0/llm/epyc-inference-research
bash scripts/benchmark/profile_verification_cost.sh

# Phase 3 (~3 hours total)
bash scripts/benchmark/bench_largek_speculation.sh

# Generate plots
python scripts/benchmark/plot_verification_profile.py
```
