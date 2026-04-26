# Compounding Verification Matrix — 2026-04-26 (post-revert NPS4)

**Goal**: verify that CPU1, CPU2, and CPU15 (EP) wins compound multiplicatively against a properly-distributed baseline. **Result**: levers do NOT compound — most prior wins were sub-baseline artifacts.

## Method

All runs:
- `taskset -c 0-95 -t 96 -fa 1 -p 0 -n 32`
- Branch `feature/cpu-ep-inter-process` HEAD `8cb04da9d`
- `LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build/bin:/opt/AMD/aocc-compiler-5.0.0/lib`
- NPS4, numa_balancing=0, THP=always, governor=performance
- `drop_caches` before each run (cold-cache)
- `r=3` initially; `r=5` for stability re-runs where variance was high

**Two canonical configs compared**:
- "Proper" cold-cache: `numactl --interleave=all --physcpubind=0-95 ... --mmap 0`
- "Plain" warmed reference: `taskset -c 0-95 ... ` (mmap=1 default; reference is steady-state-after-warming)

## Baseline comparison (cold-cache canonical vs warmed mmap=1 reference)

| Model | Quant | Proper cold canonical | Warmed mmap=1 ref | Δ proper vs warmed |
|---|---|---|---|---|
| Qwen3-Coder-30B-A3B | Q4_K_M | 42.27 ± 0.39 | 43.57 ± 0.10 | −3% (~equivalent) |
| **Qwen3.6-35B-A3B** | **Q8_0** | **20.81 ± 0.02** | **14.63 ± 0.01** | **+44%** |
| Qwen3-Next-80B-A3B | Q4_K_M | 20.51 ± 0.02 | 23.25 ± 0.08 | −12% |
| Qwen3-Coder-REAP-246B-A35B | Q4_K_M | 5.94 ± 0.01 (r=5) | 6.85 ± 0.01 | −13% |
| **gemma-4-26B-A4B-it** | **Q4_K_M** | **34.69 ± 0.07** | **25.01 ± 0.08** | **+39%** |

**Model-specific picture**: proper canonical is dramatically better for Q8_0 and gemma-4-26B (the models with hybrid attention + heavy CPU_REPACK paths); roughly even for Coder-30B; worse for Next-80B and REAP-246B (where warmed mmap=1 settles into reasonable per-node distribution over time).

## Compounding deltas (vs proper canonical)

### Coder-30B Q4_K_M

| Config | t/s | Δ |
|---|---|---|
| Proper canonical | 42.27 ± 0.39 | 0% |
| + CPU1 3-flag (`CCD_POOLS + CCD_WORK_DIST + BARRIER_LOCAL_BETWEEN_OPS`) | 42.53 ± 0.11 | **+0.6%** (noise) |

Original claim: "CPU1 3-flag = +1.8% on Coder-30B" (warmed mmap=1 baseline). Re-measured against proper baseline: noise.

### Qwen3.6-35B-A3B Q8_0

| Config | t/s | Δ |
|---|---|---|
| Proper canonical (auto-mbind ON) | 20.81 ± 0.02 | 0% (reference) |
| + CPU1 3-flag stack | 21.16 ± 0.02 | **+1.7%** (noise) |
| + EP frontdoor stack (`GGML_EP_*`) | 21.15 ± 0.04 | **+1.6%** (noise) |
| + EP + 3-flag | 20.89 ± 0.04 | +0.4% (regression vs +EP alone) |
| Auto-mbind kill-switch off (`GGML_NUMA_REPACK_INTERLEAVE=0`) | 20.85 ± 0.01 (r=5) | **0%** (redundant with --interleave=all) |
| Plain canonical mmap=1 + taskset, auto-mbind ON, cold | 7.85 ± 0.01 | −62% |
| Plain canonical mmap=1 + taskset, auto-mbind OFF, cold | 7.85 ± 0.01 | −62% (auto-mbind also irrelevant on cold mmap=1) |

**Key finding**: the historical "EP frontdoor +17% production win" was 14.63 → 17.18 measured against the **warmed mmap=1 reference**. Against the proper cold canonical (20.81), EP delivers +1.6% — within noise. EP machinery was largely fixing the bad first-touch placement of mmap=1; `--mmap 0 + --interleave=all` fixes it more cleanly with no code.

**Auto-mbind on CPU_REPACK is redundant with `--interleave=all`** — no measurable difference with kill-switch on/off when --interleave is active. The +6% claim for auto-mbind was against a different baseline state.

### REAP-246B Q4_K_M (>150B class)

| Config | t/s | Δ |
|---|---|---|
| Proper canonical | 5.94 ± 0.01 (r=5) | 0% |
| + EP frontdoor stack | 5.92 ± 0.01 (r=5) | **0%** (neutral) |

**Key finding**: the historical "EP regresses −47% on REAP-246B" was 3.65 (warmed mmap=1 baseline 6.85, with EP applied). Against the proper cold canonical (5.94), EP is **neutral** (5.92). The catastrophic >150B regression was a baseline artifact.

This re-frames the entire CPU24 attribution work — there is no measurable EP regression on >150B to attribute when measured properly. CPU24 is now optional: the open question is "why is REAP-246B 5.94 t/s and what's the ceiling" rather than "why does EP regress".

## Strategic implications

1. **Most "wins" were sub-baseline artifacts**: CPU1 3-flag (+1.8% → +0.6%), EP frontdoor (+17% → +1.6%), auto-mbind (+6% → 0%), REAP EP regression (−47% → 0%) all collapse when measured against a properly-distributed baseline.
2. **The biggest practical win is the canonical config itself**: `--mmap 0 + --interleave=all` gives +44% on Q8_0 and +39% on gemma-26B vs the historical reference. **Production deployment of this config alone captures more than all the optimization code combined for those models**.
3. **CPU1 / CPU2 auto-mbind / EP machinery deliver ≤2% on the proper baseline**: still bit-correct, still useful for research, but not a meaningful production win.
4. **Model-specific picture**: Next-80B and REAP-246B prefer the warmed mmap=1 path. Production deployment should be model-aware.

## Files

- `A_coder_canonical.log` — Coder-30B canonical
- `B_coder_3flag.log` — Coder-30B + 3-flag
- `C_q8_canonical.log` — Q8_0 canonical (initial r=3)
- `D_q8_3flag.log` — Q8_0 + 3-flag
- `E_q8_EP.log` — Q8_0 + EP
- `F_q8_EP_3flag.log` — Q8_0 + EP + 3-flag
- `G_next80b_canonical.log` — Next-80B canonical
- `H_reap246b_canonical.log` — REAP canonical (r=3, σ=0.24)
- `H2_reap246b_canonical_r5.log` — REAP canonical (r=5 stable)
- `I_reap246b_EP.log` — REAP + EP (r=3, σ=0.97)
- `I2_reap246b_EP_r5.log` — REAP + EP (r=5 stable)
- `J_gemma_canonical.log` — Gemma-26B canonical
- `K_q8_killswitch_off.log` — Q8_0 + kill-switch off (r=3, σ=9.56)
- `K2_q8_killswitch_off_r5.log` — Q8_0 + kill-switch off (r=5 stable)
- `L_q8_taskset_mbindON.log` — Q8_0 mmap=1 + taskset + auto-mbind ON, cold
- `M_q8_taskset_mbindOFF.log` — Q8_0 mmap=1 + taskset + auto-mbind OFF, cold
- `V1_q8_plain_canonical_cold.log` — Q8_0 plain canonical cold (verification)
- `V2_q8_cold_canonical_repeat.log` — Q8_0 cold canonical sanity repeat
