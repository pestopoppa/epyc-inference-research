# Seeds from intake-lpr-20261122 (H12 sparse QSA band and union, Q8_0 two-stream one_chunk)

Source: epyc-root research intake intake-2792, intake-2795, intake-2808, intake-2818, intake-2819 (Stage 2, 2026-11-22;
dives in epyc-root `research/deep-dives/intake-lpr-20261122/`). Grade: `design_prior`. Gate class: futility-only.
Filed here rather than as handoff boxes on operator instruction (Stage 4, 2026-11-22). Cite the entries with `#record`;
no number below is a measurement on our hardware.

## AK-H12-BAND (proposed) -- measure the unmeasured QSA sparse ratio band 4-8 on gfx90a
AK-ASM-H12 (qsa-sparse.cu:454-600 @ eee3da8) dispatches fused sparse QSA only at n_kv/width >= 8
(GGML_CUDA_QSA_SPARSE_MIN_KV_RATIO), set post hoc from a loss at ratio 4 and a win at 16; ratios 5-8 were never measured.
Upstream #32741 tunes a CDNA (gfx908) break-even that admits sparse gather above a selected fraction of 1/4 when
nq x n_kv_max is small, and upstream #29298 makes the gate token-count-aware (min(nq, cap)).
**Question:** does H12 win at ratio 5, 6 or 8 for 1, 4, 8 and 16 query tokens (Q38FN QSA shape, hd256, gqa12)?
**Expected:** >= 3% op time vs an A/A noise of ~1%, op-level test-backend-ops perf, OFF/OFF A/A floor, 3 repeats.
**Stop rules:** stop-on-answer when every cell has a paired median; keep ratio 8 if the ratio-6 result is < 1.03x at
either nq 1 or 16. **Falsifier:** the ratio-5 and ratio-6 cells both lose at nq 1 and nq 16 -- close the band.

## AK-H12-UNION (proposed) -- share gathered K/V rows across a verify batch
H12 runs one workgroup per (token, kv head) and shares nothing across tokens (qsa-sparse.cu:248 @ eee3da8); it loses
0.78x at ratio 4 x 16 tokens. Upstream #28770 builds one union index list per group of queries so each union row is
read once for every query-head column in the tile.
**Step 1 (zero kernel work):** from a Q38FN top-k dump at verify batches of 2/4/8/16 tokens and ratio 4/8/16, compute
the union fraction |union of the batch's top-k lists| / (tokens x top_k) per kv head.
**Step 2 (only if step 1 passes):** gather each union row once into a tokens x G (G=12) MFMA tile, mask per token; screen
vs H12 at cells 2/4/8/16 tokens x ratio 4/8/16. A top-k bitmap fits in LDS (4 KB at 32k context).
**Expected:** KEEP if geomean >= 1.10 vs H12 and no cell < 0.97.
**Falsifier:** the union fraction exceeds 0.6 at 16 tokens, ratio 4 -- no shared rows to exploit; close without a kernel.
Note: upstream's 8 x 8 tile wastes a quarter of the head columns at GQA 12; do not copy its tile shape.

## AK-CPU-Q8-2STREAM (proposed) -- upstream #30846 two-address-stream one_chunk on Zen 5
Upstream #30846 (merged 2026-11-21, 660be5506) feeds the two weight rows of an nrc==2 vec_dot from the first and second
half of the chunk instead of adjacent rows, so the hardware prefetcher sees two streams. Grace (ARM) S_TG gains were
+2.6% (bandwidth-saturated, 64 threads) to +30% (2 threads) at -npl 2; never measured on x86. On our champion only Q8_0
under AVX2/AVX-512 has nrows=2 and reaches mul_mat_one_chunk below the iqk Q8_0 threshold (ggml-cpu.c:308-312,
:1688 @ eee3da8); at n=1 it is a no-op. Our Q8_0 nrc==2 branch (ggml-cpu.c:1414-1418) collides with the upstream rewrite.
**Question:** does the two-stream effect exist on Zen 5 at verify widths?
**Bench:** one-matrix microbench, 27B-shaped ffn tensor, 24 threads on one quarter (q0-q2), ne11 in {2,4,8,16},
adjacent-pair vs split-stream, hot, bit-identical output.
**Stop rules:** stop-on-answer when 4 ne11 x 3 reps are done; only a pass opens an e2e 27B DFlash2 CPU verify row.
**Falsifier:** gain < 1.5% at ne11 2..8 at the realistic thread count -- close.
