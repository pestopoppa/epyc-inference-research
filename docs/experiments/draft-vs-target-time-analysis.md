# Draft vs Target Time Analysis — Production Spec Decode Breakdown

**Date**: 2026-03-29
**Hardware**: AMD EPYC 9655 (96 cores, 2 NUMA nodes), 1.1 TB DDR5
**Software**: llama.cpp (custom fork), llama-server with speculative decoding
**Data source**: `data/tree_speculation/server_sweep_*.csv` + server logs `logs_*/pair*_psplit0.log`
**Related**: [specexec-verification-profile.md](specexec-verification-profile.md), [RESULTS.md](../reference/benchmarks/RESULTS.md)

## Question

For each production spec decode pair, what percentage of inference time is spent drafting vs target verification? If drafting were instantaneous, what is the limiting throughput?

## Method

llama.cpp tracks cumulative `t_draft_us` (microseconds in draft generation) via `common_speculative_print_stats()`, logged as `statistics draft: dur(b,g,a) = begin, generate, accept ms`. The CSV benchmark data records end-to-end `time_ms` per request.

For each model pair:
1. Extract incremental `t_draft_generate` between successive log entries (cumulative → per-request)
2. Compute `t_target = time_ms_csv - t_draft_generate` per request
3. Sum across all 4 benchmark prompts (256 or 128 output tokens each)

The "begin" and "accept" durations for the draft model are negligible (<0.1ms total), so `t_draft ≈ t_draft_generate`.

## Results

### Time Breakdown by Model Pair

| # | Target Model | Draft Model | Draft % | Target % | Draft t/s | Avg Spec t/s | Accept % |
|---|--------------|-------------|---------|----------|-----------|--------------|----------|
| 1 | Qwen3-Coder-30B-A3B Q4KM | 0.75B-Q4 | **26.6%** | 73.4% | 148 | 35.4 | 56% |
| 2 | Qwen2.5-7B f16 (ref) | 0.5B-f16 | **24.2%** | 75.8% | 124 | 33.0 | 79% |
| 3 | Qwen3.5-122B-A10B Q4KM | 0.8B-Q8 | **17.4%** | 82.6% | 38 | 5.9 | 51% |
| 4 | Qwen2.5-Coder-32B Q4KM | 0.5B-f16 | **15.0%** | 85.0% | 121 | 19.8 | 76% |
| 5 | Qwen3-235B-A22B Q4KM | 0.6B-Q8 | **10.9%** | 89.1% | 81 | 8.8 | 59% |
| 6 | Qwen3-Coder-480B Q4KM | 0.75B-Q4 | **5.2%** | 94.8% | 110 | 5.4 | 63% |

### Instantaneous Draft Limit (target-only throughput)

| Target Model | Current t/s | Limit t/s (draft=0) | Gain | Production Role |
|--------------|------------|---------------------|------|-----------------|
| Qwen3-Coder-30B-A3B Q4KM | 35.4 | **48.3** | +36% | worker |
| Qwen2.5-7B f16 | 33.0 | **43.6** | +32% | (reference) |
| Qwen3.5-122B-A10B Q4KM | 5.9 | **7.2** | +21% | architect_general |
| Qwen2.5-Coder-32B Q4KM | 19.8 | **23.3** | +18% | coder_escalation |
| Qwen3-235B-A22B Q4KM | 8.8 | **9.8** | +12% | (not deployed) |
| Qwen3-Coder-480B Q4KM | 5.4 | **5.7** | +6% | architect_coding (replaced by REAP-246B) |

### Models NOT using spec decode

| Model | Role | Reason | Throughput |
|-------|------|--------|-----------|
| Qwen3.5-35B-A3B Q4KM | frontdoor | SSM hybrid — spec incompatible | 12.7 t/s (moe6) |
| Qwen3-Next-80B-A3B Q4KM | ingest | SSM hybrid — spec incompatible | ~12 t/s |

## Key Findings

### 1. Drafting is never the bottleneck

Draft models run at 80–150 t/s regardless of target size. As targets get larger and slower, draft time shrinks to a negligible fraction. The 480B target spends 95% of time in verification; making the draft infinitely fast would only yield +6%.

### 2. Inverse relationship: model size vs draft fraction

```
Draft time fraction ≈ (draft_tokens_per_round / draft_tps) / total_round_time
                    ≈ K / draft_tps / (K / draft_tps + t_verify)
```

Since `t_verify` scales with model size but `K / draft_tps` is roughly constant (~0.2s for K=24 at 120 t/s), the draft fraction shrinks as models get larger:

- 30B target: draft ~27%, target ~73%
- 32B target: draft ~15%, target ~85%
- 122B target: draft ~17%, target ~83%
- 235B target: draft ~11%, target ~89%
- 480B target: draft ~5%, target ~95%

### 3. Draft model bandwidth contention

The 0.8B-Q8 draft alongside the 122B target runs at only 38 t/s (vs 120–150 for the same class of draft alongside smaller targets). The 69GB target model saturates memory bandwidth, starving the draft model. This suggests co-scheduling optimizations (NUMA pinning of draft to a different node) could help the 122B pair specifically.

### 4. Maximum theoretical return on draft speed improvements

Even if a hypothetical "perfect drafter" (instantaneous, 100% acceptance) existed:

| Model | Theoretical max (instant draft + perfect acceptance) | vs current |
|-------|-----------------------------------------------------|-----------|
| 30B worker | K × target_solo_tps ≈ 8 × 12 = 96 t/s | 2.7× current |
| 32B coder | K × target_solo_tps ≈ 24 × 5.8 = 139 t/s | 7× current |
| 122B architect | limited by SSM sequential verify | ~1.2× current |

Dense models have the most room because verification is truly parallel (batch of N ≈ cost of 1). Hybrid SSM models can't batch verification — each token must traverse recurrent layers sequentially.

### 5. Implication for optimization priorities

The dominant speed lever for each model class:

| Model Class | Best lever | Potential | Draft speed lever |
|-------------|-----------|-----------|-------------------|
| Small dense (7B-32B) | NUMA 4×48t | 4× | +18-32% (moderate) |
| Large MoE (235B-480B) | Expert reduction | +58-87% | +6-12% (marginal) |
| Hybrid SSM (35B-122B) | Expert reduction | +42-45% | N/A (spec incompatible) |

**Conclusion**: Investing in faster draft models yields diminishing returns. The verification pass is the wall. NUMA parallelism, expert reduction, and architecture improvements (dense vs hybrid) are higher-leverage.

## Raw Data

### Pair 5: Qwen2.5-Coder-32B Q4KM + 0.5B-f16 (from `logs_20260311_104941`)

```
Cumulative draft stats:
  After warmup:  87.877ms,   9 tokens
  After req 0:  1003.647ms, 126 tokens (CSV: 4458ms, 128t, 105/112 accept)
  After req 1:  2015.647ms, 249 tokens (CSV: 7595ms, 128t, 80/121 accept)
  After req 2:  2992.724ms, 362 tokens (CSV: 7441ms, 128t, 80/111 accept)
  After req 3:  3972.111ms, 480 tokens (CSV: 6336ms, 128t, 90/115 accept)

  Total draft gen: 3884ms, Total end-to-end: 25830ms → 15.0% draft
```

### Pair 6: Qwen3.5-122B-A10B Q4KM + 0.8B-Q8 (from `logs_20260311_104941`)

```
Cumulative draft stats:
  After warmup:  1423.335ms,  54 tokens
  After req 0:  4915.830ms, 185 tokens (CSV: 23415ms, 128t, 65/131 accept)
  After req 1:  9057.269ms, 345 tokens (CSV: 23175ms, 128t, 70/154 accept)
  After req 2: 12403.335ms, 472 tokens (CSV: 21600ms, 128t, 71/127 accept)
  After req 3: 16460.315ms, 630 tokens (CSV: 18349ms, 128t, 86/158 accept)

  Total draft gen: 15037ms, Total end-to-end: 86539ms → 17.4% draft
```

### Pair 8: Qwen3-235B-A22B Q4KM + 0.6B-Q8 (from `logs_20260315_000913`)

```
Cumulative draft stats:
  After warmup:    333.053ms,   27 tokens
  After req 0:   3537.002ms,  293 tokens (CSV: 29617ms, 256t, 154/266 accept)
  After req 1:   6650.977ms,  538 tokens (CSV: 29486ms, 256t, 147/242 accept)
  After req 2:   9698.601ms,  766 tokens (CSV: 31138ms, 256t, 134/226 accept)
  After req 3:  13072.136ms, 1056 tokens (CSV: 26634ms, 256t, 172/289 accept)

  Total draft gen: 12739ms, Total end-to-end: 116875ms → 10.9% draft
```

### Pair 9: Qwen3-Coder-480B Q4KM + 0.75B-Q4 (from `logs_20260318_032701`)

```
Cumulative draft stats:
  After warmup:     73.204ms,    8 tokens
  After req 0:   2513.944ms,  317 tokens (CSV: 35993ms, 256t, 206/302 accept)
  After req 1:   4880.430ms,  557 tokens (CSV: 51948ms, 256t, 135/238 accept)
  After req 2:   7389.026ms,  834 tokens (CSV: 50321ms, 256t, 159/275 accept)
  After req 3:   9896.225ms, 1091 tokens (CSV: 49922ms, 256t, 143/254 accept)

  Total draft gen: 9823ms, Total end-to-end: 188184ms → 5.2% draft
```

### Pair 15: Qwen3-Coder-30B-A3B Q4KM + 0.75B-Q4 (from `logs_20260318_035407`)

```
Cumulative draft stats:
  After warmup:     55.572ms,    8 tokens
  After req 0:   1933.942ms,  313 tokens (CSV: 5694ms, 256t, 201/284 accept)
  After req 1:   3736.968ms,  570 tokens (CSV: 8046ms, 256t, 130/257 accept)
  After req 2:   5727.531ms,  856 tokens (CSV: 7147ms, 256t, 173/280 accept)
  After req 3:   7764.548ms, 1146 tokens (CSV: 8043ms, 256t, 138/285 accept)

  Total draft gen: 7709ms, Total end-to-end: 28930ms → 26.6% draft
```

### Pair 1: Qwen2.5-7B f16 + 0.5B-f16 (from `logs_20260311_014340`, reference)

```
Cumulative draft stats:
  After warmup:     90.644ms,    9 tokens
  After req 0:   2012.886ms,  271 tokens (CSV: 4770ms, 256t, 222/259 accept)
  After req 1:   3869.308ms,  483 tokens (CSV: 10170ms, 256t, 149/212 accept)
  After req 2:   5688.009ms,  711 tokens (CSV: 7394ms, 256t, 185/223 accept)
  After req 3:   7582.822ms,  936 tokens (CSV: 8664ms, 256t, 170/225 accept)

  Total draft gen: 7492ms, Total end-to-end: 30998ms → 24.2% draft
```
