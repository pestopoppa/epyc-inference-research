# A resident-grid layer kernel with global-atomic mailboxes may recover the within-step gap on gfx90a (2026-11-19)

Sources: research intake 2026-11-19.
- intake-2697 (FlyDSL blog, anchored): HIP graph replay leaves dependent kernels serialized on the GPU; gains
  shrink as batch grows; an FFN-only layer test was near parity.
- intake-2703 (FlyDSL `kernels/monokernel` @ 31f389b): resident-grid schedule, tagged-pair mailbox,
  phase-overlaid LDS arena; the shared GEMM is gfx950-only.

Grade on arrival: `design_prior`. Gate class: futility-only. Root task: AK-RESGRID-1, gated on AK-GAP-1 (a).

## AK-H-RESGRID-1 — a persistent fused sub-block kernel on gfx90a beats the graph-replayed kernel chain

gfx90a substitutes:
- global-memory epoch flags with atomics and `__threadfence` (two slots per region) in place of the gfx950
  tagged mailbox;
- buffer_load plus ds_write in place of async-to-LDS;
- MFMA 16x16x16 f16, or the existing Q8_0 mmvq dot path, in place of `mma=16x16x32`.

**Falsifier:** AK-GAP-1 (a) is below 5% of the step, or the fused-pair op A/B is below 1.10x.
**Expected served-model gain [inferred]:** at most about half of the (a) share on Qwen3.8-27B decode. The
upstream 1.64x included cross-GPU overlap that a single MI210 lacks. So >= 5% e2e only if (a) >= 10%.

## Compounded-gain check (2026-11-20, operator rule: sub-3% levers judged on lever + what they unlock)
- AK-H-RESGRID-1: box AK-RESGRID-1 exists (gated on AK-GAP-1 (a)); no unlock beyond its own share; KEPT.
