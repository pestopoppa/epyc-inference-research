# Model Sizing Guide

This guide helps you determine which models fit your hardware and how to allocate resources across roles.

## Table of Contents

- [Assess Your Hardware](#assess-your-hardware)
- [Model Size Estimation](#model-size-estimation)
- [Memory Budgeting](#memory-budgeting)
- [Recommended Configurations](#recommended-configurations)
- [Performance Expectations](#performance-expectations)

## Assess Your Hardware

### Quick Assessment Script

Run this to assess your system:

```bash
#!/bin/bash
echo "=== System Assessment ==="

# RAM
RAM_GB=$(free -g | awk '/Mem:/ {print $2}')
echo "RAM: ${RAM_GB} GB"

# CPU
CORES=$(nproc)
echo "CPU Cores: ${CORES}"

# AVX-512
AVX512=$(grep -o 'avx512[a-z]*' /proc/cpuinfo 2>/dev/null | sort -u | wc -l)
if [ "$AVX512" -gt 0 ]; then
    echo "AVX-512: Supported ($(grep -o 'avx512[a-z]*' /proc/cpuinfo | sort -u | tr '\n' ' '))"
else
    echo "AVX-512: Not supported"
fi

# NUMA nodes
NUMA_NODES=$(lscpu | grep "NUMA node(s)" | awk '{print $NF}')
echo "NUMA Nodes: ${NUMA_NODES:-1}"

# Storage
echo "=== Storage ==="
df -h /mnt/raid0 2>/dev/null || df -h / | tail -1

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
    echo "Tier: PRODUCTION - HOT + architect_general"
else
    echo "Tier: FULL - All tiers including 480B architect"
fi
```

### Understanding Your Resources

| Resource | How to Check | Why It Matters |
|----------|--------------|----------------|
| **RAM** | `free -g` | Models load entirely into RAM |
| **CPU Cores** | `nproc` | More threads = faster inference |
| **AVX-512** | `grep avx512 /proc/cpuinfo` | 2-4x faster inference |
| **NUMA** | `numactl --hardware` | Affects memory bandwidth |
| **Storage** | `df -h` | Models are 1-300GB each |

## Model Size Estimation

### Quantization Impact

GGUF quantization determines model memory footprint:

| Quant | Bits/Weight | Multiplier | Quality vs F16 |
|-------|-------------|------------|----------------|
| F16 | 16 | 2.0x | 100% (baseline) |
| Q8_0 | 8 | 1.0x | ~99.9% |
| Q6_K | 6 | 0.75x | ~99.5% |
| Q5_K_M | 5 | 0.625x | ~99% |
| Q4_K_M | 4 | 0.5x | ~98% |
| Q3_K_M | 3 | 0.375x | ~95% |
| Q2_K | 2 | 0.25x | ~90% |

### Size Formula

```
Model Size (GB) ≈ Parameters (B) × Quant Multiplier × 1.1
```

The 1.1 factor accounts for metadata and KV cache overhead.

**Examples:**

| Model | Params | Quant | Calculation | Size |
|-------|--------|-------|-------------|------|
| Qwen2.5-7B | 7B | Q4_K_M | 7 × 0.5 × 1.1 | ~4 GB |
| Qwen2.5-32B | 32B | Q4_K_M | 32 × 0.5 × 1.1 | ~18 GB |
| Qwen3-235B | 235B | Q4_K_M | 235 × 0.5 × 1.1 | ~130 GB |
| Qwen3-480B | 480B | Q4_K_M | 480 × 0.5 × 1.1 | ~265 GB |

### MoE Models

MoE (Mixture of Experts) models are named like `Model-TotalB-ActiveB`:
- **Total params**: Full model size
- **Active params**: What runs per token (affects speed)

Example: `Qwen3-235B-A22B` = 235B total, 22B active per token

**MoE benefits:**
- Quality of large model
- Speed closer to active param count
- Memory = total param count

## Memory Budgeting

### RAM Allocation

| Component | Typical Usage | Notes |
|-----------|---------------|-------|
| **OS + Services** | 8-16 GB | System overhead |
| **llama.cpp overhead** | 2-4 GB | Per server instance |
| **KV Cache** | 1-8 GB per model | Scales with context |
| **Model weights** | Varies | Primary usage |
| **Safety buffer** | 10-20% | Prevent OOM |

### Budget Calculator

```python
def calculate_model_budget(total_ram_gb: int) -> dict:
    """Calculate how much RAM is available for models."""
    os_overhead = 16
    safety_buffer = total_ram_gb * 0.15
    available = total_ram_gb - os_overhead - safety_buffer

    return {
        "total_ram": total_ram_gb,
        "os_overhead": os_overhead,
        "safety_buffer": safety_buffer,
        "available_for_models": available,
        "hot_tier_budget": available * 0.1,  # ~10% for always-resident
        "warm_tier_budget": available * 0.9,  # ~90% for on-demand
    }
```

**Example budgets:**

| Total RAM | Available for Models | HOT Budget | WARM Budget |
|-----------|---------------------|------------|-------------|
| 64 GB | 38 GB | 4 GB | 34 GB |
| 128 GB | 93 GB | 9 GB | 84 GB |
| 256 GB | 202 GB | 20 GB | 182 GB |
| 512 GB | 420 GB | 42 GB | 378 GB |
| 1024 GB | 854 GB | 85 GB | 769 GB |

## Recommended Configurations

### Minimal (64GB RAM)

**Model Budget:** ~38GB

| Role | Model | Size | Notes |
|------|-------|------|-------|
| All roles | Qwen2.5-7B-Instruct Q4 | 4 GB | Single model serves everything |

**Limitations:**
- No architect escalation
- No parallel workers (same model)
- Basic quality

### Basic (128GB RAM)

**Model Budget:** ~93GB

| Role | Model | Size |
|------|-------|------|
| frontdoor | Qwen3-Coder-30B Q4 | 18 GB |
| coder_escalation | Qwen2.5-Coder-32B Q4 | 20 GB |
| workers | Qwen2.5-7B Q4 | 4 GB |
| draft | Qwen2.5-Coder-0.5B Q8 | 0.5 GB |
| **Total** | | **~43 GB** |

**Remaining:** ~50GB for KV cache and warm models

### Standard (256GB RAM)

**Model Budget:** ~202GB

| Role | Model | Size |
|------|-------|------|
| frontdoor | Qwen3-Coder-30B Q4 | 18 GB |
| coder_escalation | Qwen2.5-Coder-32B Q4 | 20 GB |
| ingest | Qwen3-Next-80B Q4 | 46 GB |
| workers | Qwen2.5-7B Q4 | 4 GB |
| worker_vision | Qwen2.5-VL-7B Q4 | 5 GB |
| draft | Qwen2.5-Coder-0.5B Q8 | 0.5 GB |
| **Total HOT** | | **~48 GB** |

**WARM available:** ~154GB for one architect when needed

### Production (512GB RAM)

**Model Budget:** ~420GB

| Tier | Role | Model | Size |
|------|------|-------|------|
| HOT | frontdoor | Qwen3-Coder-30B Q4 | 18 GB |
| HOT | coder_escalation | Qwen2.5-Coder-32B Q4 | 20 GB |
| HOT | workers | Qwen2.5-7B f16 + 0.5B | 16 GB |
| HOT | worker_vision | Qwen2.5-VL-7B Q4 | 5 GB |
| WARM | architect_general | Qwen3-235B Q4 | 133 GB |
| WARM | ingest | Qwen3-Next-80B Q4 | 46 GB |
| **Total** | | | **~238 GB** |

**Remaining:** ~182GB for architect_coding or alternative models

### Full (1TB+ RAM)

All roles populated, including Qwen3-Coder-480B for architect_coding (~271GB).

## Performance Expectations

### Inference Speed by Model Size

With AVX-512 (AMD EPYC or Intel Xeon):

| Model Size | Quant | Expected t/s | Notes |
|------------|-------|--------------|-------|
| 0.5B | Q8 | 80-120 | Draft models |
| 1.5B | Q4 | 50-70 | Fast workers |
| 7B | Q4 | 25-40 | Workers |
| 14B | Q4 | 15-25 | |
| 32B | Q4 | 8-15 | |
| 70B | Q4 | 4-8 | |
| 235B MoE | Q4 | 5-7 | With MoE4 reduction |
| 480B MoE | Q4 | 8-12 | With MoE3 reduction |

### Acceleration Methods

| Method | Typical Speedup | Best For |
|--------|----------------|----------|
| Speculative Decoding | 3-11x | Dense models with compatible drafts |
| Prompt Lookup | 1.5-12x | Tasks with repeated context |
| MoE Expert Reduction | 1.5-2x | MoE models only |
| Combined Spec+Lookup | 5-12x | Best results |

### Quality vs Speed Tradeoffs

| Priority | Recommendation |
|----------|----------------|
| **Max Quality** | Larger models, no MoE reduction, Q6_K |
| **Balanced** | Production defaults (MoE4, Q4_K_M) |
| **Max Speed** | Smaller models, aggressive MoE reduction, spec decode |

## Choosing Models for Your Hardware

### Decision Tree

```
RAM < 32GB?
  → Use mock mode only

RAM < 64GB?
  → Single 7B model for all roles

RAM < 128GB?
  → Frontdoor + workers only
  → No architects

RAM < 256GB?
  → Full HOT tier
  → One WARM model (ingest OR architect)

RAM < 512GB?
  → Full HOT tier
  → architect_general + ingest

RAM >= 512GB?
  → Full deployment
  → All roles populated
```

### Scaling Down

When constrained, prioritize:

1. **frontdoor** (always needed)
2. **workers** (parallel tasks)
3. **coder_escalation** (quality fallback)
4. **ingest** (document handling)
5. **architect_general** (complex decisions)
6. **architect_coding** (hardest problems)

### Scaling Up

With excess RAM:

1. Use larger quantizations (Q6_K, Q8_0)
2. Add more worker instances
3. Increase KV cache for longer context
4. Keep WARM models pre-loaded
