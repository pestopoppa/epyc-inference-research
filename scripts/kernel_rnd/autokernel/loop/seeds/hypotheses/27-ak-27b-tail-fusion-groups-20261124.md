# Seed: 27B verify elementwise-tail fusion, untried groups G2/G3/G5-G8 (2026-11-24)
<!-- journal-key: {"op_family": "launch-bound-tail", "device": "gpu", "shape_class": {"n": [1, 8], "group": [2, 8]}, "touches": [{"path": "27b-df2-mi210", "family": "launch-bound-tail", "fraction": 0.48}]} -->
Source: AK-RF-27B-TAIL-FUSE closed NO_KEEP by its own +1% CI-lower floor (root baba3def5). Classification:
`/mnt/raid0/llm/tmp/ak-27b-tail-classify-20261124/FINDINGS.md` (classify.py / classify.json in that directory).
Grade: design_prior. Gate class: futility-only. Base: champion C21 `98ff8a42a`. Device: MI210, 27B Q8_0 + DFlash2, verify widths 1-8.

## AK-RF-27B-TAIL-FUSE-G2-G8 (proposed) -- fuse the untried launch-floor groups of the 27B verify tail
Mechanism: the true eltwise/norm tail is ~5.2 ms and ~880 launches per verify (13% of verify GPU time); 60% of
kernels sit on the MI210 ~4.5 us per-dispatch floor, so a fusion saves about one floor per absorbed launch. There
are no idle gaps inside verify, so graph-capture widening is out. Launch-count-fusion ceiling ~3.0 ms/verify
(~6% of a ~47 ms round). Measured so far: G1 (residual add + rms_norm + mul) +1.1% [+0.6, +1.7] ms/round;
G4 (split-K zero-fill) not significant; G1+G4 +0.6%. Untried, in launch order of expected value:
- G3 GDN gates: unary 512x1 + k_bin_bcast add/mul 24x10 -> one kernel (192 launches, 0.88 ms)
- G7 GDN state gather: k_get_rows_float x2 -> one gather (97 launches, 0.75 ms)
- G2 SwiGLU: unary_gated_op into the up/gate GEMV epilogue (121 launches, 0.55 ms)
- G6 attention prologue: q/k rms_norm + rope_multi + k_set_rows + cpy (110 launches, 0.53 ms)
- G5 GDN per-head output norm into the GDN output (46 launches, 0.21 ms)
- G8 conv: already fused (ak_h15_conv_fused_kernel); only a merge with G7 is left
Expected gain (inferred upper bound): untried share of the ceiling ~ (3.0 - 0.5 G1) / 5.2 ms = fraction 0.48 of
the tail surface; realistic 1-2 ms/verify for G2+G3+G7.
**Falsifier:** per-group tail ms/verify on the 27B DF2 np1 trace saves < 1 launch-floor per absorbed launch, or the
e2e ms/round gain's CI lower bound <= the GPU W2 floor (0.84%) -> drop that group. Bit-identical outputs required.
Effort M per group.
Journal key: `group` is the G-number above. A negative for the closed AK-RF-27B-TAIL-FUSE must be keyed
`group: 1` and `group: 4` (the measured groups) so it never suppresses this seed's `group: [2, 8]`.
