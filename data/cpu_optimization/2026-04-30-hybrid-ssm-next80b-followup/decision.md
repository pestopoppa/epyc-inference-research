# Hybrid SSM Follow-up — Qwen3-Next-80B

**Date**: 2026-04-30
**Verdict**: **+8.9% pp512 finding is Nemotron-specific, does NOT generalize to MoE Hybrid SSM (Qwen3-Next-80B-A3B)**. Effect on Next-80B is +1.7% pp512 c3 — direction same, magnitude 5× smaller. Recommendation: narrow the Hybrid SSM c3 deployment recommendation to "dense Hybrid SSM only".

## Method

Same canonical recipe as Probe B, n=5, single model (Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf), 3 configs × 3 shapes = 9 cells.

## Result

| Shape | c0 baseline | c2 mbind-off | c3 CPU1+mbind-off |
|---|---|---|---|
| tg64 | 22.37 ± 0.01 | 22.35 (-0.1%) | 22.46 (+0.4%) |
| pp512 | 282.58 ± 0.62 | 281.02 (-0.6%) | **287.36 (+1.7%)** |
| pp2048 | 274.95 ± 0.67 | 280.15 (+1.9%) | 277.35 (+0.9%) |

## Comparison to Nemotron-9B (Probe B)

| Shape | Nemotron-9B c3 | Next-80B c3 | Generalization |
|---|---|---|---|
| tg64 | +2.1% | +0.4% | Direction same, 5× smaller |
| pp512 | **+8.9%** | **+1.7%** | Direction same, 5× smaller |
| pp2048 | +3.7% | +0.9% | Direction same, 4× smaller |

**Pattern**: c3 effect on Next-80B is uniformly ~5× smaller than on Nemotron-9B across all shapes. Direction is preserved (positive), but magnitude is materially attenuated.

## Why the difference

Architectural:
- **Nemotron-9B-v2**: dense Mamba2 SSM + interleaved attention, 9B params, no MoE.
- **Qwen3-Next-80B-A3B**: hybrid SSM + MoE, 80B total params, 3B active per token via `mul_mat_id`.

Hypothesis: the MoE expert dispatcher (`mul_mat_id`) has per-token-varying memory access patterns — different experts touched per token — which defeats CPU1's CCD-aware partitioning. CCD-pinning works when each thread reads a predictable slice of weight memory; with MoE, each thread's working set varies per token, making CCD locality worse than the standard interleave=all baseline.

Additionally, Next-80B-A3B's baseline pp512 is already high (282 t/s vs Nemotron's 317 t/s — note Nemotron's higher baseline despite smaller model size, suggesting Nemotron's prefill is bandwidth-bound and Next-80B-A3B's is compute-bound at the MoE dispatcher).

## Implications

The earlier deployment recommendation for Hybrid SSM should narrow:

**REVISED recommendation**:

| Hybrid SSM sub-class | Recommendation |
|---|---|
| **Dense Hybrid SSM** (Nemotron-9B-v2 type) | c3 (CPU1+mbind-off) opt-in for prefill-heavy workload (+8.9% pp512) |
| **MoE Hybrid SSM** (Qwen3-Next-80B-A3B type) | Default v5 stack (c3 effect is +1.7% — within practical noise floor for production deployment decisions) |

The "Hybrid SSM benefits from c3" generalization was overstated — it's actually a "dense-arch + barrier-heavy + bandwidth-bound prefill" pattern. The MoE structure of Next-80B-A3B places it closer to MoE Q4_K_M Coder/REAP behavior (where CPU1 effects are small — +1.8% on Coder per CPU21).

## Files

- `run_followup.sh` — measurement script
- `master.log` — full run log
- `followup_<shape>_<config>.log` — 9 per-cell bench logs
- `decision.md` — this document
