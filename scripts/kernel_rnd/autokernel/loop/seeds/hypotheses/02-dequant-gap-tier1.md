# The dequant gap on MI210: ranked levers, each with its decisive experiment

Source: `handoffs/active/mi210-q8-dequant-gemv-roofline.md` (epyc-root), "Levers, ranked".
Note the handoff's own 2026-07-04 correction at its head: the Tier-1 *premise* was wrong and
async-prefetch was reframed as the lever — then async-prefetch was itself measured
net-negative (see the negatives already in memory). Read the levers, not the framing.

Roofline context: Q8 decode sits at ~47% of achievable bandwidth against an fp16 ceiling of
~62%. The gap is described as kernel-addressable.

## Tier 1 — the dequant gap (highest confidence)

**L1 — kill the `quantize_q8_1` activation-requant overhead.** Measured at **5.68% of decode**.
The GEMV requants activations to Q8_1 every step; fuse it into the GEMV prologue, or cache it.
*Decisive experiment:* rocprof before/after.
**Falsifier:** `quantize_q8_1` share does not fall to ~0, or decode does not rise ~5%.

**L2 — fused dequant-in-GEMV / int8-native MMQ path for batch-1.** `mul_mat_vec_q` currently
dequantizes Q8 blocks and then accumulates in fp. Port the CPU **iqk** approach —
weight-block-outer, dequant-under-load, reference
`ggml/src/ggml-cpu/iqk/iqk_gemm_legacy_quants.cpp:302-330` — to HIP, or hide the dequant under
the HBM weight load. *Decisive experiment:* single-stream `llama-bench -n 128`, Q8 against the
fp16 62% ceiling.
**Falsifier:** Q8 decode fails to reach ~55-62% of roofline (~34-38 t/s on the 27B), or
PPL/output moves.

## Tier 2 — the batch-1 latency wall (harder, MLP-bound)

**L3 — async weight prefetch / double-buffering in the GEMV.** Software-pipeline the next
weight tile's `buffer_load` under current compute (Little's law: more requests in flight ->
higher achieved BW). *Decisive experiment:* rocprof MemUnitStalled + achieved-BW before/after.
**Falsifier:** achieved BW does not rise from ~62% toward ~70%.
**Before proposing this, read the `akm-hist-q8-prefetch` record in memory: this mechanism was
built and measured net-negative on gfx90a (dense −12%/−18%).** Proposing it again requires
saying what is different, not restating the Little's-law argument.

**L4 — persistent / megakernel decode.** CUDA-proven, no ROCm version — greenfield and
high-effort. Fuse decode into one launch with pipelined weight streaming (the Hazy/Mirage ~78%
single-dispatch result). *Scope first, do not build blind:* estimate the launch-overhead
fraction via rocprof gap-analysis before committing.
**Falsifier (inferred here, not stated in the handoff):** gap-analysis shows inter-dispatch
gaps are a small fraction of decode wall time, leaving nothing for fusion to reclaim.

**L5 — weight swizzle / layout for gfx90a.** Verify MMVQ HBM access is coalesced to the
64-wide wavefront and the 128B cache line. *Decisive experiment:* rocprof L2 / MemUnit efficiency.
**Falsifier (inferred here, not stated in the handoff):** MMVQ HBM access is already near-ideally
coalesced, so a layout change has no headroom to recover.

## Tier 3 — bytes per token (orthogonal; can exceed BW-% gains)

**L7 — sub-4-bit weights (IQ2/TQ1/TQ2) with an efficient CDNA2 dequant.** Fewer bytes/token ->
higher absolute t/s at the same roofline percentage. Gated on the Tier-1 dequant kernel and an
eval-parity/PPL check. Connects directly to the occupancy-knee hypotheses in seed 01.
**Falsifier (inferred here, not stated in the handoff):** sub-4-bit decode fails to exceed Q8
in absolute t/s despite fewer bytes/token -- which is also what seed 01's occupancy cliff
predicts, so a null here is evidence FOR AK-H-QL-1, not merely a dead lever.

**L8 — KV-quant for single-stream LONG context** (q8-KV, `-fa 1`). Distinct from the aggregate
case, where it was dead: at batch-1 long context the KV read adds to bytes/token.
*Decisive experiment:* 27B decode at 64k ctx, q8-KV vs f16-KV.
**Falsifier (inferred here, not stated in the handoff):** q8-KV decode at 64k does not beat
f16-KV outside the measured noise band.

Deliberately omitted from this seed: the handoff's Tier-3 lever 6 (n-gram speculation) and its
"do this FIRST" ranking. It is a serving-configuration change, not a kernel patch, so it is
outside what this loop measures — and our own record retracts the headline ngram result as a
warm-context self-copy artifact.
