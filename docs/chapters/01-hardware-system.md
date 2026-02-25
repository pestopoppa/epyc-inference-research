# Chapter 01: Hardware System

## Introduction

This project optimizes LLM inference on AMD's EPYC 9655 "Turin" processor. The system was chosen for its massive memory capacity (1.13 TB) and high memory bandwidth (~460 GB/s across 12 channels), which are critical bottlenecks for large language model inference on CPU.

## Hardware Specifications

Everything about this build revolves around memory — how much of it we have, and how fast we can read it. With 1.13 TB of DDR5 across 12 channels, we can hold models up to ~500B parameters in RAM and stream weights at ~460 GB/s. The CPU itself brings true 512-bit AVX-512 (not Intel's double-pumped variant), giving genuine 2x vector width for the matrix math that dominates inference.

<details>
<summary>Full specifications table</summary>

| Component | Specification |
|-----------|---------------|
| CPU | AMD EPYC 9655 "Turin" (Zen 5 architecture) |
| Cores/Threads | 96 cores / 192 threads |
| RAM | 1.13 TB DDR5-5600 ECC (12 channels) |
| Memory Bandwidth | ~460 GB/s theoretical |
| Storage | 2× Solidigm P44 Pro 2TB NVMe in RAID0 |
| OS Drive | 120GB SSD (system only) |
| Architecture | True 512-bit AVX-512 (not double-pumped like Intel) |

</details>

<details>
<summary>Why this hardware matters</summary>

**Memory Capacity**: At 1.13TB, we can load models up to ~500B parameters at Q4_K_M quantization entirely in RAM. This eliminates disk I/O bottlenecks that plague smaller systems.

**Memory Bandwidth**: The 12-channel DDR5 configuration provides approximately 460 GB/s of bandwidth. Since LLM inference is memory-bound during generation (reading weights for each token), this bandwidth directly determines maximum throughput.

**AVX-512**: Zen 5 implements true 512-bit AVX-512 units, unlike Intel's double-pumped approach. This provides genuine 2x vector width for SIMD operations in matrix multiplications.

</details>

## Runtime Optimizations

Getting the hardware to actually deliver its theoretical bandwidth requires careful tuning. The wrong thread settings can cut performance in half, and ignoring NUMA topology leaves bandwidth on the table. Three settings matter most: pinning OpenMP to one thread (llama.cpp handles its own parallelism), interleaving memory across all 12 channels, and using only physical cores.

<details>
<summary>Critical environment settings</summary>

<details>
<summary>Code: environment variables and launch flags</summary>

```bash
# Prevent nested parallelism (severely degrades performance)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# NUMA interleaving for balanced memory access across 12 channels
numactl --interleave=all <command>

# Use physical cores only (hyperthreading hurts inference)
-t 96

# Enable Transparent Huge Pages
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled

# Set CPU governor to performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

</details>

**OMP_NUM_THREADS=1**: llama.cpp handles threading internally. OpenMP trying to parallelize on top of llama.cpp's threading causes thread contention and can reduce performance by 50% or more.

**numactl --interleave=all**: With 12 memory channels, NUMA effects are significant. Interleaving distributes data across all channels, maximizing bandwidth utilization.

**96 threads (physical cores only)**: Hyperthreading provides no benefit for compute-bound LLM inference and can actually hurt due to cache contention.

</details>

## Baseline Performance

Before any optimization tricks, here's what raw token generation looks like on this hardware. The key takeaway: generation is the bottleneck, not prompt processing. Each generated token requires reading the entire model's weights, making it purely memory-bandwidth bound — which is exactly why speculative decoding (amortizing multiple tokens per read) delivers such dramatic speedups.

<details>
<summary>Baseline speed measurements</summary>

| Model | Size (GGUF) | Prompt Processing | Token Generation |
|-------|-------------|-------------------|------------------|
| Qwen2.5-Coder-32B Q4_K_M | 19GB | 69.05 t/s | 2.89 t/s |
| Qwen2.5-72B Q4_K_M | 42GB | ~50 t/s | ~1.8 t/s |
| Qwen3-235B-A22B Q4_K_M | 131GB | ~30 t/s | ~3.6 t/s |
| Qwen3-Coder-480B Q4_K_M | 271GB | 34.66 t/s | 3.06 t/s |

</details>

## Storage Architecture

The system uses a split storage design to keep the tiny 120GB OS drive from being overwhelmed by model files and caches. All LLM-related data lives on the 4TB RAID0 array at `/mnt/raid0/`. Writing large files to the OS drive has caused system crashes in the past — this is the single most important rule of the project.

<details>
<summary>Storage layout details</summary>

- **OS Drive (120GB SSD)**: System files only. DO NOT write LLM data here.
- **RAID0 Array (/mnt/raid0/)**: 4TB striped array for all models, caches, and project files.

**Critical Rule**: All LLM-related files must reside on `/mnt/raid0/`. Writing large files to the OS drive causes system instability.

</details>

<details>
<summary>References</summary>

### Hardware Documentation

1. AMD Corporation. (2024). *AMD EPYC 9655 Processor Specifications*. https://www.amd.com/en/products/cpu/amd-epyc-9655

2. AMD Corporation. (2024). *AMD Zen 5 Architecture White Paper*. https://www.amd.com/en/technologies/zen-architecture

### Software and Implementation

3. Gerganov, G., et al. (2024). *llama.cpp: LLM inference in C/C++*. GitHub. https://github.com/ggml-org/llama.cpp

4. llama.cpp Contributors. (2024). *CPU Performance Discussion: EPYC and Threadripper*. GitHub Discussions. https://github.com/ggml-org/llama.cpp/discussions/4167

### System Optimization

5. Drepper, U. (2007). *What Every Programmer Should Know About Memory*. Red Hat, Inc. https://people.freebsd.org/~lstewart/articles/cpumemory.pdf

6. Linux Kernel Documentation. *Transparent Hugepages*. https://www.kernel.org/doc/Documentation/vm/transhuge.txt

7. Linux Kernel Documentation. *NUMA Memory Policy*. https://www.kernel.org/doc/Documentation/admin-guide/mm/numa_memory_policy.rst

</details>

---

*Next: [Chapter 02: Runtime Environment](02-runtime-environment.md)*
