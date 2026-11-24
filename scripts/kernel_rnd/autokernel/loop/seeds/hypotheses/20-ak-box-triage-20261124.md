# Seeds: AK box triage leftovers, op-level and single-knob levers (2026-11-24)

Source: autokernel session, DN triage 2026-11-24 (`/mnt/raid0/llm/tmp/ak-box-triage-20261124/ROI.md`). Root boxes named per seed (epyc-root `handoffs/active/autokernel-loop.md`, `autokernel-akx-program.md`, `speculative-decoding.md`). Grade: `design_prior`. Gate class: futility-only. Base: champion C21 `98ff8a42a`.
Common KEEP gates: GPU W2 floor 0.84%; CPU seeds use the quarter A/A floor of the row's region (ak-cpu-aa-calibration-q-a3b / -q-c18); 27B KLD <= 0.003 if numerics change; bit-identical where stated.

## AK-H-Q8IP-PERMUTE-R4 (proposed) -- elide or fuse q8ip_permute on DFlash2 draft segments
Mechanism: q8ip_permute runs on every DFlash2 draft segment (0.43 ms/segment); fuse it into the producer, or cache the permuted layout across segments. Expected gain: <= 0.43 ms/segment on the 27B drafter; r2 measured NEUTRAL, and r3 was inconclusive because only the control arm diverged.
**Falsifier:** paired drafter ABBA on C21 with the candidate-only modal-hash identity rule; no gain > 0.84% -> drop and close the root box NEUTRAL. Effort S.

## AK-H-I8GEMM-PC-TILE (proposed) -- retune per-channel int8 GEMM tiles at 27B prefill shapes
Mechanism: the direct per-channel port was 0.821x the incumbent (ak-asm-i8gemm-perchannel-gpu-20261122). Search the tile M/N/K, waves/WG and LDS staging on gfx90a. Expected gain: prefill only, and 27B prefill is not the served bottleneck.
**Falsifier:** op time at 27B prefill shapes; stop at the first config search that cannot reach >= 1.10x the incumbent. Effort M.

## AK-H-DET-SCRATCH (proposed) -- scratch + fixed-order fixup for GGML_DETERMINISTIC reductions
Mechanism: replace the 2.4x ticket chain with per-WG scratch partials and one fixed-order fixup kernel (q4_K and f32 paths). Expected gain: only for roles that need bitwise determinism; no served speed change.
**Falsifier:** overhead vs non-deterministic >= 1.3x, or any bitwise mismatch across 5 reruns -> drop. Effort M.

## AK-H-CPU-TB-SPLIT (proposed) -- decouple -tb from -t on the served A3B CPU frontdoor
Mechanism: threads_batch as an independent runtime arm (for example -t 24 -tb 48 at the quarter shape); prefill scales with threads further than decode does. Prior: +21-29% shallow prefill on Q38FN at -t 48 -tb 96.
**Falsifier:** quarter-shape pp512/pp2048 and tg128 on A3B Q8_0; pp gain <= floor, or tg regression > floor -> drop. Effort S (knob).

## AK-H-Q38FN-IDX-FUSE (proposed) -- fused indexer score op (dot128 + ReLU + head-sum) on CPU
Mechanism: one fused op replaces three graph ops in the Q38FN lightning indexer. Ceiling 0.65-0.77 ms/step at d32k (AK-Q38FN-CPU-INDEXER-DEPTH: an accounting artefact, ~0.6-0.8 ms/step). Q38FN is served on GPU, so this is CPU-lane only.
**Falsifier:** bit-identical output plus ms/step at d32k quarter; gain <= floor -> drop. Effort M.

## AK-H-Q38FN-TOPK-EXACT (proposed) -- parallel CPU TOP_K matching partial_sort exactly
Mechanism: a parallel TOP_K that returns exactly the partial_sort set and order (the failed row saw -33 ms/step at 32k, but the order was not exact).
**Falsifier:** any set/order mismatch on the captured indexer scores, or no ms/step gain > floor at 32k -> drop. Effort M.

## AK-H-Q38FN-DRAFT-FA-W (proposed) -- draft FA window W8192 and adaptive W4096A40K
Mechanism: bound the MTP draft attention window at long context (fork aaec7962a).
**Falsifier:** 32k acceptance < 0.713, or t/s gain <= floor -> drop. Effort S (knob).

## AK-H-DF2-LAT (proposed) -- zero-training lattice-headroom probe of the DFlash2 selector
Mechanism: replay logged draft lattices offline and measure the accepted length an oracle path selector would get versus the current selector (intake-2561#record).
**Falsifier:** oracle headroom < 5% accepted length -> drop; no timing claim. Effort S (offline).

## AK-H-SPEC-CONF (proposed) -- DFlash2 proxy-confidence reliability
Mechanism: log selector top-1 probability against realized acceptance at np4; if it is reliable, it gates a cross-request verify-budget arm.
**Falsifier:** AUROC < 0.7 -> drop the verify-budget arm. Effort S.
