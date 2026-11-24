# Seeds: post-Stage-4 kernel triage, roofline-parity fold levers (2026-11-24)
Source: autokernel session triage `/mnt/raid0/llm/tmp/ak-kernel-triage-20261124/TRIAGE.md`; roofline-parity table in epyc-root `docs/design/mi210-served-kernel-bandwidth-census-20261122.md` @ 801631795; root boxes in `handoffs/active/autokernel-loop.md` (AK-RF-*) and `kernel-champion-and-promotion.md` (CH-19). Grade: design_prior. Gate class: futility-only. Base: champion C21 `98ff8a42a`.
Common KEEP gates: GPU W2 floor 0.84%; CPU seeds use the quarter A/A floor of the row's region; 27B KLD <= 0.003 if numerics change; A3B GPU paths are tracked as served (operator 2026-11-24). Every GPU seed first re-reads its cell from the AK-RF-C21-CENSUS-REFRESH rows; if the C21 share is < 5% of step, drop without building.

## AK-RF-27B-TAIL-FUSE (proposed) -- 27B verify add+rms_norm epilogue fusion and graph-capture widening
Mechanism: the verify tail (eltwise/norm/copy/rope/conv, ~1000 launches, 8.6 ms/fwd, plus 5.67 ms idle gaps) runs launch-bound at ~35% roofline-proxy vs 85% fused-tail reference (parity #4, score 7.3, uncovered). Fuse residual add + rms_norm (+ adjacent scale) into one launch per layer, and capture the remaining small launches in one HIP graph per verify width. Expected gain: 1.5-4 ms/fwd (inferred, 3-8% of round).
**Falsifier:** tail ms/verify on the 27B DF2 np1 trace; < 1 ms/fwd saved or e2e gain <= 0.84% -> drop. Bit-identical outputs required for the add+norm fusion. Effort M.

## AK-RF-A3B-ROUTER-FUSE (proposed) -- fused router GEMV + top-k for A3B MTP verify (n>=2)
Mechanism: router mul_mat_f F32 [2048,256] measured 18 GB/s, 4.6 ms/step (C14, pre MMVF cap knob 79650eeab). One per-layer kernel computes the 256 logits and the top-k select in registers/LDS, absorbing AK-BW-A3B-VERIFY-TOPK's staged chain (softmax/argsort/get_rows, 1.9 ms/step). Expected gain: residual-dependent, <= 4.5 ms/step ceiling.
**Falsifier:** C21 residual router+chain share < 5% of step -> drop unbuilt; else op us at n=3,4,8 < 2x incumbent, or A3B MTP e2e <= floor -> drop. Expert ids must match the reference exactly. Effort M.

## AK-RF-A3B-Q8_1-FOLD (proposed) -- quantize src1 once per activation on the A3B graph
Mechanism: quantize_q8_1 is 3.1 ms/step (9.1%, C14). Router, shared expert and MMID gate/up all read the same normed activation; quantize once and share, fusing into the norm epilogue. Reuse AK-BW-27B-ACTQUANT-FUSE's implementation once it lands. Expected gain: <= 2-3 ms/step (inferred).
**Falsifier:** quantize ms/step saved < 0.5 or e2e <= floor -> drop. Bit-identical. Effort S (after actquant).

## AK-RF-A3B-BINBCAST-FOLD (proposed) -- fold k_bin_bcast / scale into producing epilogues on A3B
Mechanism: ~520 launches x 4.6 us = 2.4 ms/step (7.1%, C14). Fold the router-weight scale into the expert-sum and the gate scalars into the GDN / shared-expert outputs. Composes with the CPU GGML_MOE_WSUM_FOLD idea (AK-CPU-C22-LEVERS) on the GPU side. Expected gain: <= 2 ms/step.
**Falsifier:** launch count per step drops < 30% or e2e <= floor -> drop. Bit-identical. Effort M.

## AK-RF-A3B-GV-ATTN (proposed) -- apply the 27B gv_fattn split/LDS-cut tuning to the A3B 10-layer f16-KV shape
Mechanism: A3B FA/gv is ~1.4 ms/step (4.1%) at a GB/s borrowed from 27B (130-235 GB/s). Measure the A3B head geometry first, then reuse the GGML_CUDA_FA_GQA_VERIFY_SPLITS auto-tune and LDS cut from AK-GVFATTN-LONGCTX / the gvfattn-served fix. Expected gain: <= 1 ms/step at 8-16k, more at long ctx.
**Falsifier:** A3B gv us/call at 8k and 32k not >= 1.2x incumbent -> drop. Wait for the gvfattn-served fix to land. Effort S (reuse).

## CH-19-A3B (proposed, low prior) -- fused MoE up+gate+SwiGLU on A3B Q8_0 CPU
Mechanism: one pass over gate and up expert rows and the SwiGLU, so the intermediate never round-trips. Prior is weak: the A3B CPU exps_gate_up is already at 90% of the quarter ceiling (census CPU 32-37; AK-CPU-A3B-MMID NO_LEVER), and the Q38FN IQ3_S variant was slower. The gain can only come from the activation write plus one fewer sync.
**Falsifier:** A3B CPU MTP step ms at the quarter shape; gain <= quarter A/A floor -> drop and record a do-not-repeat. Effort M.
