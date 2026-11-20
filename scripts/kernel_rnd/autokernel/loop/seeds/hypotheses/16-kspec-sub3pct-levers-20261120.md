# Sub-3% kernel/perf levers from intake-kspec-20261120, seeded for the AK loop (2026-11-20)

Source: Stage-4 filing of research intake intake-kspec-20261120 (epyc-root commit `43e2a69ec`, plan
`/mnt/raid0/llm/tmp/intake-kspec-20261120-plan-r2a.md`). Operator directive 2026-11-20: boxes with expected
gain below 3% are not hand-prepped; they are seeded here for the AK loop to explore when it runs. The AK loop
launch HOLD stays in place; seeding is not launching. Grade on arrival: `design_prior`. Gate class:
futility-only. Production stays v10; champion is C14 `eee3da838` (do not re-propose anything already in C14/C16).

Serving floors for any keep: CPU W5 4.15%, GPU W2 0.84% (end-to-end, same-window A/A, interleaved).
Where numerics change: 27B KLD <= 0.003, |dPPL| <= 0.5%, Q38FN top-1 >= 93%. All gains below are `inferred`
or `derived` as marked; none is measured.

## AK-H-Q6KHEAD-1 — Q6_K lm_head call cost on CPU: is an AVX-512 Q6_K GEMV worth writing?
Root box: Q6K-HEAD-CALLCOST (`handoffs/active/cpu-decode-roofline-program.md`). Source: intake-2718#record.
**Mechanism:** split the Q6_K result_output call at t48 into per-thread compute vs straggler/dispatch cost
(thr_max/thr_mean/thr_min vs wall). If instruction-bound dequant, an AVX-512 Q6_K GEMV lowers head cost.
**Target:** Qwen3.8-Flash-Next UD-IQ4_XS, shared Q8_0 MTP head, CPU t48, d1k and d32k (trunk and draft copies).
**Expected gain [inferred]:** up to ~2.0% of the Q38FN step if dequant-bound; 0 if per-call cost dominates
(prior evidence points there). Confidence low.
**Falsifier:** thr_max/thr_mean < 1.1 and wall within 5% of bytes/read-BW => per-call cost, close the lever.
**Gate:** reuse existing units first (`/mnt/raid0/llm/tmp/q38fn-cpu-perop-profile-20261118-r2/out/servers/d1k-L1.prof.json`);
a GEMV patch only opens on a dequant-bound verdict; then CPU W5 4.15% end-to-end plus bit-exact vs current kernel
(numerics unchanged, so KLD gates not triggered).
**Code:** C14 `eee3da838` (`/mnt/raid0/llm/worktrees/llama-c14-pin-20261119`), GGML_CPU_PROF build; head borrowed in
`src/models/qwen4exp.cpp` L582-584; head sizing in cpu-decode-roofline-program.md L386, L393.

## AK-H-IQ4NLZMM-1 — widen the IQ4_NL repack GEMV from 256-bit AVX2 LUT to 512-bit vpshufb + vpdpbusd
Root box: IQ4NL-GEMV-ZMM (`handoffs/active/cpu-decode-roofline-program.md`). Source: intake-2718#record.
**Mechanism:** `ggml_gemv_iq4_nl_8x8_q8_0` uses an AVX2 LUT path; a 512-bit vpshufb dequant plus vpdpbusd dot on Zen 5.
**Target:** Q38FN IQ4_NL expert-down tensors, CPU decode, only if XFER-4b keeps IQ4_NL on the repack path.
**Expected gain [derived]:** ~0.8% of step. Confidence low. Below the CPU 4.15% W5 floor on its own, so it is
only worth a keep as part of a composite; the loop should weigh it as a composite ingredient, not a standalone keep.
**Falsifier:** XFER-4b moves IQ4_NL off repack, or the micro A/B on expert-down shapes is < 1.05x.
**Gate:** bit-exact unit test vs the AVX2 GEMV on expert-down shapes, strings proof of the new symbol, decode micro A/B,
then CPU W5 4.15% end-to-end (numerics bit-exact, KLD gates not triggered).
**Code:** C14 `ggml/src/ggml-cpu/arch/x86/repack.cpp` (`ggml_gemv_iq4_nl_8x8_q8_0`); experimental branch from fresh C14.

## AK-H-DF2FLOOR-1 — fuse the 27B DFlash2 drafter tiny-kernel floor into a few HIP kernels
Root box: DFLASH2-DRAFT-FLOOR-FUSE (`handoffs/active/autokernel-loop.md`). Source: intake-2725#record.
**Mechanism:** 488 of 560 dispatches per round are under 6 us (2.14 ms, 3.6% of the round); fuse the per-round conv
taps (`build_dflash2_conv`, 20 calls x 2 taps) and the selector lattice loop into a few HIP kernels behind a
default-off knob, after a zero-compute attribution step (GGML_SCHED_DEBUG node dump matched to the rocprofv3 order).
**Target:** Qwen3.8-27B Q8_0 + DFlash2 Q8_0, MI210, served recipe.
**Expected gain [inferred]:** upper +3.4% of the 27B DFlash2 round (448 x 4.38 us = 1.96 ms of 59.3 ms), realistic
+1.5% to +2.5% (profiler inflates tiny kernels). Confidence low.
**Falsifier:** attribution shows < 270 conv+selector dispatches per round, or fused subgraph < 1.10x at op level.
**Gate:** NMSE of fused vs unfused subgraph, greedy token identity, then end-to-end decode A/B above the GPU W2
0.84% floor (5 interleaved reps); fused subgraph >= 1.30x and e2e above 0.84% carries to the champion gate.
**Code:** C14 `src/models/dflash.cpp` build_dflash2_conv L775 (calls L1146/1203/1216/1229), selector loop L1312-1378;
trace input `/mnt/raid0/llm/tmp/ak-gap-1-trace-gpu-20261119/out/trace/27B-dflash2/k_kernel_trace.csv`.

## AK-H-CLUSTVOCAB-GPU-1 — training-free clustered vocab projection for the GPU draft heads (A3B MTP, 27B DFlash2)
Root box: DRAFT-HEAD-APPROX-RECALL (`handoffs/active/speculative-decoding.md`); the box's CPU Q38FN arm is >= 3% and
stays a hand-prepped task, this seed covers only the GPU-side arms. Sources: intake-2727, intake-2740, intake-2725.
**Mechanism:** two-stage approximate draft head: 2048-cluster centroid router (or truncated-SVD pre-score), exact
logits on the top 32 clusters / top C rows; removes most draft-head MACs. The drafter only proposes, so verify
guarantees target-distribution output; the cost is acceptance, not correctness.
**Target:** A3B MTP head and 27B DFlash2 head on MI210; head is 1.4-2.6% of the round (round splits: 27B drafter 14.8%
of 59.3 ms, A3B 20.0% of 22.6 ms).
**Expected gain [derived]:** GPU side A3B +0.5..+2.6%, 27B -0.3..+1.4% before acceptance loss. Confidence low-medium.
**Falsifier:** the box's offline recall@1 gate fails on the A3B/27B heads (recall@1 < 99.5% overall or < 98% in the
low-margin quartile at >= 79% MACs removed), or the extra router launch costs more than the saved head MACs.
**Gate:** offline recall first (DRAFT-HEAD-APPROX-RECALL result), then end-to-end tok/s above GPU W2 0.84% with
acceptance length reported; target output unchanged by construction (greedy token identity vs full head under verify).
**Code:** A3B/27B heads via act_dump capture (research `123ec088e`, matcher on the MUL_MAT whose weight is
output.weight); drafter graphs in C14 `src/models/dflash.cpp` and the MTP graph builders.
