# Model Sizing Guide

This guide helps you determine which models fit your hardware and how to allocate resources across roles. Configurations reflect the production stack as of 2026-05 (Qwen3.5 / Qwen3.6 frontdoor era, post-monorepo-split).

## Table of Contents

- [Assess Your Hardware](#assess-your-hardware)
- [Model Size Estimation](#model-size-estimation)
- [Memory Budgeting](#memory-budgeting)
- [Recommended Configurations](#recommended-configurations)
- [Performance Expectations](#performance-expectations)
- [Acceleration Methods](#acceleration-methods)

## Assess Your Hardware

> The script and reference numbers below are tuned to the production EPYC 9655 deployment (192-core, 1.1 TB RAM, NPS4, NUMA). Adjust paths and thresholds for other systems.

### Quick Assessment Script

```bash
#!/bin/bash
echo "=== System Assessment ==="

RAM_GB=$(free -g | awk '/Mem:/ {print $2}')
echo "RAM: ${RAM_GB} GB"

CORES=$(nproc)
echo "CPU Cores: ${CORES}"

AVX512=$(grep -o 'avx512[a-z]*' /proc/cpuinfo 2>/dev/null | sort -u | wc -l)
if [ "$AVX512" -gt 0 ]; then
    echo "AVX-512: Supported ($(grep -o 'avx512[a-z]*' /proc/cpuinfo | sort -u | tr '\n' ' '))"
else
    echo "AVX-512: Not supported"
fi

NUMA_NODES=$(lscpu | grep "NUMA node(s)" | awk '{print $NF}')
echo "NUMA Nodes: ${NUMA_NODES:-1}"

echo "=== Storage ==="
df -h /mnt/raid0 2>/dev/null || df -h / | tail -1

echo "=== NUMA Balancing ==="
cat /proc/sys/kernel/numa_balancing  # MUST be 0; kernel resets to 1 (feedback_numa_balancing_self_reset)

echo ""
echo "=== Recommendations ==="
if [ "$RAM_GB" -lt 32 ]; then
    echo "Tier: DEV ONLY - Use mock mode or 0.5B-1.5B models"
elif [ "$RAM_GB" -lt 64 ]; then
    echo "Tier: MINIMAL - One 7B model for all roles"
elif [ "$RAM_GB" -lt 128 ]; then
    echo "Tier: BASIC - Frontdoor + workers, no architects"
elif [ "$RAM_GB" -lt 256 ]; then
    echo "Tier: STANDARD - HOT tier + one architect"
elif [ "$RAM_GB" -lt 512 ]; then
    echo "Tier: PRODUCTION - HOT + architect_general (Qwen3.5-122B fits)"
else
    echo "Tier: FULL - All tiers including 480B architect_coding"
fi
```

### Understanding Your Resources

| Resource | How to Check | Why It Matters |
|----------|--------------|----------------|
| **RAM** | `free -g` | Models load entirely into RAM (mmap + mlock for HOT) |
| **CPU Cores** | `nproc` | Affects decode throughput; production uses 96t per single-NUMA-node instance |
| **AVX-512** | `grep avx512 /proc/cpuinfo` | Required for quantized GEMV ukernels (Zen 5 VPMADDUBSW 2/cycle, see `project_zen5_vnni_vs_maddubs`) |
| **NUMA** | `numactl --hardware` | NPS4 is the production BIOS setting; pinning + interleave required |
| **Storage** | `df -h` | Models are 1–300 GB each |
| **CPU governor** | `cpupower frequency-info` | Verify before trusting any bench (`feedback_host_throttle_check`) |

### NUMA: The Production Constraint

The production EPYC 9655 runs **NPS4** (4 NUMA nodes per socket). Each node has 6 memory channels. RAID0 NVMes are split across nodes 2+3, so I/O-heavy work needs `numactl --interleave=2,3` (`project_raid_numa_split_nps4`).

Key sizing implications:

- Single-instance peak (96t-single-NUMA-node) ≈ 49 t/s for 30B-A3B Q4_K_M.
- Production "concurrent split" mode runs 4×48t instances per model, aggregating to ~150 t/s for 7B / Qwen3-Coder-30B (`project_concurrent_split_throughput`).
- For NUMA multi-instance, **always** use `--mlock` with `numactl --membind` — never bare `taskset` (`feedback_mmap_numa_sharing`).

## Model Size Estimation

### Quantization Impact

GGUF quantization determines model memory footprint:

| Quant | Bits/Weight | Multiplier | Quality vs F16 |
|-------|-------------|------------|----------------|
| F16 | 16 | 2.0× | 100% (baseline) |
| Q8_0 | 8 | 1.0× | ~99.9% |
| Q6_K | 6 | 0.75× | ~99.5% |
| Q5_K_M | 5 | 0.625× | ~99% |
| Q4_K_M | 4 | 0.5× | ~98% (production default) |
| Q3_K_M | 3 | 0.375× | ~95% |
| Q2_K | 2 | 0.25× | ~90% |

### Size Formula

```
Model Size (GB) ≈ Parameters (B) × Quant Multiplier × 1.1
```

The 1.1 factor accounts for metadata + KV cache overhead. For SSM+MoE hybrids subtract a few % (Qwen3.5 fits in less RAM than a pure-MoE of the same total-param count).

**Examples (verified against `model_registry.yaml` 2026-05):**

| Model | Params | Quant | Calculation | Measured |
|-------|--------|-------|-------------|----------|
| Qwen2.5-7B | 7B | Q4_K_M | 7 × 0.5 × 1.1 | ~4 GB |
| Qwen3-Coder-30B-A3B | 30B (3B active) | Q4_K_M | 30 × 0.5 × 1.1 | 16 GB (production worker) |
| Qwen2.5-Coder-32B | 32B | Q4_K_M | 32 × 0.5 × 1.1 | 20 GB |
| Qwen3.5-35B-A3B-UD | 35B (3B active, SSM+MoE) | Q4_K_M (moe6) | — | 19 GB |
| Qwen3.5-122B-A10B | 122B (10B active, MoE) | Q4_K_M | — | 69 GB |
| Qwen3-Next-80B-A3B | 80B (3B active, SSM) | Q4_K_M | 80 × 0.5 × 1.1 | 46 GB |
| Qwen3-Coder-REAP-246B-A35B | 246B (35B active, pruned) | Q4_K_M | — | ~139 GB |
| Qwen3-235B-A22B | 235B (22B active) | Q4_K_M | 235 × 0.5 × 1.1 | ~134 GB |
| Qwen3-Coder-480B-A35B | 480B (35B active) | Q4_K_M | 480 × 0.5 × 1.1 | ~271 GB |

### MoE Architectures

MoE (Mixture of Experts) models are named like `Model-TotalB-ActiveB`:

- **Total params** determine RAM footprint.
- **Active params** determine decode latency.

Three flavors matter for sizing:

1. **Pure MoE** (Qwen3-Coder-30B, Qwen3-235B, Qwen3-Coder-480B, REAP-25B). Compatible with speculative decoding. Expert reduction (`moe4`, `moe6`) trades quality for speed.
2. **SSM+MoE hybrid** (Qwen3.5-35B, Qwen3.5-122B). Speculative decoding net-negative (SSM checkpoint overhead). `--lookup` segfaults after a few prompts (PR #13194) — disabled in production. Use `moe6`-style expert reduction.
3. **Pure SSM** (Qwen3-Next-80B). NO speculative decoding (SSM requires consecutive positions). Expert reduction OK.

Reducing experts to fewer than 3 typically collapses quality. Production frontdoor runs `moe6` (reduced from GGUF default 8); REAP variants are permanently 25–50% pruned at conversion time.

## Memory Budgeting

### RAM Allocation

| Component | Typical Usage | Notes |
|-----------|---------------|-------|
| **OS + Services** | 8–16 GB | System overhead |
| **llama.cpp per-server overhead** | 2–4 GB | Per llama-server process |
| **KV Cache** | 1–8 GB per instance | Scales with context + ubatch |
| **CPU_REPACK heap (NUMA)** | Mbind required | `feedback_repack_buffer_numa_mbind` |
| **Safety buffer** | 10–20% | Prevent OOM and OOM-killer wakeups |
| **Model weights** | Varies | Primary usage |

### Budget Calculator

```python
def calculate_model_budget(total_ram_gb: int) -> dict:
    """Calculate how much RAM is available for models on the EPYC deployment."""
    os_overhead = 16
    safety_buffer = total_ram_gb * 0.15
    available = total_ram_gb - os_overhead - safety_buffer

    return {
        "total_ram": total_ram_gb,
        "os_overhead": os_overhead,
        "safety_buffer": safety_buffer,
        "available_for_models": available,
        "hot_tier_budget": available * 0.15,  # always-resident HOT roles
        "warm_tier_budget": available * 0.85,  # mmap-preloaded, loaded on demand
    }
```

**Example budgets:**

| Total RAM | Available for Models | HOT Budget | WARM Budget |
|-----------|---------------------|------------|-------------|
| 64 GB | 38 GB | 6 GB | 32 GB |
| 128 GB | 93 GB | 14 GB | 79 GB |
| 256 GB | 202 GB | 30 GB | 172 GB |
| 512 GB | 420 GB | 63 GB | 357 GB |
| 1024 GB | 854 GB | 128 GB | 726 GB |

## Recommended Configurations

These configurations track the 2026-05 production registry. The HOT tier grew when `architect_general` was promoted to HOT after the 2026-05-07 swap to Qwen3.5-122B-A10B (69 GB).

### Minimal (64 GB RAM)

**Model Budget:** ~38 GB

| Role | Model | Size | Notes |
|------|-------|------|-------|
| All roles | Qwen2.5-7B-Instruct Q4_K_M | 4 GB | Single model serves everything |

**Limitations:** No architect escalation, no parallel workers (one model), basic quality.

### Basic (128 GB RAM)

**Model Budget:** ~93 GB

| Role | Model | Size |
|------|-------|------|
| frontdoor | Qwen3.5-35B-A3B-UD Q4_K_M (moe6) | 19 GB |
| coder_escalation | Qwen2.5-Coder-32B Q4_K_M | 20 GB |
| worker | Qwen3-Coder-30B-A3B Q4_K_M (spec) | 16 GB |
| worker_vision | Qwen2.5-VL-7B Q4_K_M | 5 GB |
| draft | Qwen3-Coder DRAFT 0.75B Q4_0 (vocab-transplant) | 0.5 GB |
| **Total HOT** | | **~61 GB** |

**Note:** Cannot fit architect_general (69 GB). For Qwen3.x frontdoor + escalation, ensure `enable_thinking=false` is passed.

### Standard (256 GB RAM)

**Model Budget:** ~202 GB

| Role | Model | Size |
|------|-------|------|
| frontdoor | Qwen3.5-35B-A3B-UD Q4_K_M (moe6) | 19 GB |
| coder_escalation | Qwen2.5-Coder-32B Q4_K_M | 20 GB |
| worker | Qwen3-Coder-30B-A3B Q4_K_M (spec) | 16 GB |
| worker_vision | Qwen2.5-VL-7B Q4_K_M | 5 GB |
| architect_general | Qwen3.5-122B-A10B Q4_K_M | 69 GB |
| draft + utilities | — | ~5 GB |
| **Total HOT** | | **~134 GB** |

**WARM available:** ~68 GB for ingest_long_context (46 GB) on demand.

### Production (512 GB RAM) — Current Deployment

**Model Budget:** ~420 GB

| Tier | Role | Model | Size |
|------|------|-------|------|
| HOT | frontdoor | Qwen3.5-35B-A3B-UD Q4_K_M (moe6, 4×48t NUMA) | 19 GB |
| HOT | coder_escalation | Qwen2.5-Coder-32B Q4_K_M | 20 GB |
| HOT | worker | Qwen3-Coder-30B-A3B Q4_K_M | 16 GB |
| HOT | worker_vision | Qwen2.5-VL-7B Q4_K_M | 5 GB |
| HOT | architect_general | Qwen3.5-122B-A10B Q4_K_M | 69 GB |
| WARM | ingest_long_context | Qwen3-Next-80B-A3B Q4_K_M | 46 GB |
| HOT | drafts + embeddings + OCR + ColBERT | — | ~15 GB |
| **Total** | | | **~190 GB** |

**Remaining:** ~230 GB headroom for architect_coding (271 GB borderline) or candidate models (gemma-4 MTP variants, REAP architects).

### Full (1 TB+ RAM) — Production EPYC 9655

All roles populated:

| Tier | Role | Model | Size |
|------|------|-------|------|
| HOT | (as Production above) | — | ~190 GB |
| WARM | architect_coding | Qwen3-Coder-480B-A35B Q4_K_M | ~271 GB |
| WARM | ingest_long_context | Qwen3-Next-80B-A3B Q4_K_M | 46 GB |
| **Total resident peak** | | | **~507 GB** |

Alternative architect candidates:

- **Qwen3.6-35B-A3B-Q8_0** as drop-in alt frontdoor (PPL gate passed 2026-05).
- **gemma-4-26B-A4B (MTP, ik_llama.cpp PR #1744)** as `worker_general` — production swap recorded 2026-05-08 (+18pp tool_compliance, +36% tps, 76.5 t/s solo). Requires `KMP_BLOCKTIME=10` in launch env.
- **Qwen3-Coder-REAP-246B-A35B** (139 GB) — 50%-pruned alternative to 480B architect_coding (lower latency, lower RAM, slight quality tradeoff).

## Performance Expectations

Benchmarks below are sourced from `model_registry.yaml` (sweep-verified 2026-03-21 and Probe B 2026-05-04). All numbers assume canonical baseline: `taskset -c 0-95 -t 96 -fa 1` (no OMP env vars unless noted).

### Single-Instance Throughput (96t NUMA)

| Model | Quant | t/s | Source |
|-------|-------|-----|--------|
| Qwen2.5-7B (f16, +spec) | f16 | 39.1 | sweep 2026-03-21 (4×48t agg: ~156) |
| Qwen3-Coder-30B-A3B Q4_K_M (spec) | Q4_K_M | 39.1 | sweep 2026-03-21 |
| Qwen3.5-35B-A3B-UD (moe6) | Q4_K_M | 12.7 | 2026-03-24 (4×48t agg: ~50.8) |
| Qwen2.5-Coder-32B (spec, tree) | Q4_K_M | 10.8 | sweep 2026-03-21 (192t ref: 12.2) |
| Qwen3.5-122B-A10B | Q4_K_M | 12.19 | Probe B 2026-05-04 (n=5) |
| Qwen3-Coder-480B-A35B (spec, linear) | Q4_K_M | 7.0 | sweep 2026-03-21 (192t ref: 7.1) |
| Qwen3-Next-80B-A3B (SSM, moe4) | Q4_K_M | 6.3 | measured 2026-01-26 (10K-context summarization) |
| Qwen3-Coder-REAP-246B-A35B | Q4_K_M | 6.25 | tg32 r=5 audit |
| gemma-4-31B-it | Q4_K_M | 7.11 | tg64 r=5 audit |

Numbers below are *omitted* (no current authoritative measurement in scope):

- Sub-1.5B draft model peak throughput.
- 14B / 70B dense models (not in current production stack).

### Acceleration Methods

| Method | Typical Speedup | Best For | Notes |
|--------|----------------|----------|-------|
| Speculative Decoding | 3–11× | Pure MoE + dense with compatible drafts | Net-negative on Qwen3.5 hybrids |
| Prompt Lookup | 1.5–12× | Tasks with repeated context (corpus retrieval) | Disabled on Qwen3.5 hybrid frontdoor (lookup segfault) |
| MoE Expert Reduction | 1.5–2× | MoE / hybrid models | moe6 is production default for Qwen3.5-35B |
| MTP (Multi-Token Prediction) | model-specific | gemma-4 family via ik_llama.cpp PR #1744 | Requires `KMP_BLOCKTIME=10`; SIGKILL needed if FA assertion wedges |
| Combined spec + lookup | 5–12× | coder_escalation, architect_coding | Sweep-verified params per role |

For role-specific tuned values, consult `model_registry.yaml` directly. Hand-rolled "general" defaults are usually wrong by 2026 — every production role has been sweep-tuned.

### Quality vs Speed Tradeoffs

| Priority | Recommendation |
|----------|----------------|
| **Max Quality** | Larger models, full experts, Q6_K or Q8_0 |
| **Balanced** | Production defaults (Q4_K_M, moe6 for hybrids, sweep-tuned spec params) |
| **Max Speed** | Smaller models, aggressive MoE reduction, +spec +lookup +corpus_retrieval |

## Choosing Models for Your Hardware

### Decision Tree

```
RAM < 32 GB?
  → Use mock mode only

RAM < 64 GB?
  → Single 7B model for all roles

RAM < 128 GB?
  → Frontdoor (Qwen3.5-35B moe6) + escalation + worker
  → No architects

RAM < 256 GB?
  → Full HOT minus architect_general
  → One WARM model (ingest OR architect_general on demand)

RAM < 512 GB?
  → Full HOT including architect_general (Qwen3.5-122B, 69 GB)
  → ingest WARM

RAM >= 1 TB?
  → Full deployment including architect_coding (480B WARM, 271 GB)
```

### Scaling Down

When constrained, prioritize:

1. **frontdoor** (always needed) — Qwen3.5-35B-A3B-UD Q4_K_M moe6, 19 GB.
2. **worker** (parallel tasks) — Qwen3-Coder-30B-A3B Q4_K_M, 16 GB.
3. **coder_escalation** (quality fallback for code) — Qwen2.5-Coder-32B Q4_K_M + spec, 20 GB.
4. **ingest_long_context** (document handling) — Qwen3-Next-80B Q4_K_M, 46 GB.
5. **architect_general** (complex decisions) — Qwen3.5-122B-A10B Q4_K_M, 69 GB.
6. **architect_coding** (hardest code problems) — Qwen3-Coder-480B Q4_K_M, 271 GB (or REAP-246B at 139 GB).

### Scaling Up

With excess RAM:

1. Use larger quantizations (Q6_K, Q8_0) where the quality gap matters.
2. Add more `worker_pool` workers (heterogeneous pool, see `worker_pool` section in registry).
3. Increase KV cache for longer context (default 65536 server-side).
4. Keep WARM models pre-loaded with `--mlock` + `numactl --membind` for NUMA-pinned residency.

## See Also

- [Model Manifest](../MODEL_MANIFEST.md) — Current per-role assignments + ports.
- [Benchmarking Guide](benchmarking-guide.md) — How to measure new models and feed numbers back into the registry.
- [Chapter 06: Benchmarking Framework](../chapters/06-benchmarking-framework.md) — Methodology and suite definitions.
- [Chapter 08: Cost-Aware Rewards](../chapters/08-cost-aware-rewards.md) — How baseline_tps per role gets consumed by the routing reward.
