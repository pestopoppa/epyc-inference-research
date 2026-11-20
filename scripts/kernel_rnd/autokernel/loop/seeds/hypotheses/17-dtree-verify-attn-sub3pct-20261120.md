# Seeds from intake-dtree-verify-20261120 (tree-verify attention, sub-3% levers)

Source: research intake intake-2747..2751 + intake-2705 (Stage 2, 2026-11-20). Grade: `design_prior`. Gate class: futility-only.
Cost model: 27B 2.39 ms per added verify row; wvsk16 GEMM floor 2.0 ms/row (rows 8->16); A3B MMVF knob=8 1.28 ms/row.
Attention plus the rest therefore caps at ~0.39 ms/row on the 27B. The task-box lever (DTREE-ATTN-ROUTE) and its share
probe (DTREE-ATTN-SHARE) live in epyc-root `handoffs/active/speculative-decoding.md` under AK-DTREE. Do not duplicate
DTREE-0 wvsk16 or DTREE-0-Q8MF (GEMM side).

All seeds wait for DTREE-ATTN-SHARE: if FA < 0.10 ms per added row on both models, close every seed here.

Compounded-gain checks below (2026-11-20, annotate only: DTREE-ATTN-ROUTE and its ranking belong to the verify-cost driver). Tree value model as in seed 16: (1+a)/(1+dW*s/R)-1, R 52.5 ms, a +12% W12 / +17% W16 (intake-2705#record). Removing the WHOLE 0.39 ms/row attention residual moves s 2.01 -> 1.62: W12 -0.3%, W16 -6.2%, i.e. still no tree. The tree only pays once the GEMM side (DTREE-0-Q8MF) brings s near 1.5, and clears +3% only at s <= ~1.15; so attention levers carry at most a small share of that unlock.

## AK-DTREE-TAILMASK (proposed) -- skip mask reads over the committed prefix
llama.cpp FA loads the dense f16 mask for every KV tile; for a tree verify all prefix columns are visible and only the last W are
masked. Add a "mask covers tail W only" flag to fattn-tile so k < n_kv-W skips the mask load/add.
Expected: <= ~0.02-0.04 ms/row (5-10% of the attention residual), < 1% of a 27B verify.
**Falsifier:** op-level FA at n_q 16, n_kv 32k improves < 2% vs the dense-mask path. Effort S.
**Compounded-gain check: KEPT.** <= 0.04 ms/row cannot move the tree across breakeven alone; compounded ~= standalone < 1%.

## AK-DTREE-PB-OCC (proposed) -- parallel_blocks occupancy at n_q 12-16
Tile FA splits KV by parallel_blocks (fattn-common.cuh:1259/1283 @36cd3af). Check blocks_num vs 104 CUs for the verify shape and
raise the split when wave efficiency < 95%.
Expected: 0 to ~0.39 ms/row bound, likely < 3%.
**Falsifier:** FA time vs KV length is already at >= 95% wave efficiency, or a larger split changes FA < 2%. Effort S.
**Compounded-gain check: KEPT.** Best case 0.39 ms/row leaves s at 1.62 (tree W12 -0.3%); only after Q8MF lands near s 1.6-1.7 would it tip the tree positive, and then the tree adds ~+0.5..+2% at W12. Compounded < 3%; also largely the same bandwidth DTREE-ATTN-ROUTE targets.

## AK-DTREE-ANCESTOR-REG (proposed) -- in-register uint32 ancestor mask (Infernix)
Replace the f16 tail mask with one uint32 per query row applied to the MFMA 16x16x16 accumulators (v_bfe/shift per lane), the
mask arithmetic compiled out for chain. Functional parity with DTREE-ATTN-ROUTE's design; standalone gain < 0.1 ms per verify.
**Falsifier:** A/B vs the f16 mask path at 16 rows, 64k: delta < 2% -> drop. Effort S.
**Compounded-gain check: KEPT.** Functional piece of DTREE-ATTN-ROUTE; < 0.1 ms per verify, no independent unlock; compounded ~= standalone.

## AK-DTREE-CPU-KMASK (proposed) -- Zen 5 AVX-512 ancestor k-mask for CPU tree verify
CPU FA (ggml-cpu ops.cpp ~8916 @ffc1bac82) reads an f16 mask per KV position. Build an AVX-512 __mmask16 from the per-row ancestor
word (vpsrlvd + vptestmd) for the last W columns and skip the mask over the prefix.
Expected: < 1% of a CPU verify (CPU decode is weight-bandwidth bound).
**Falsifier:** CPU FA op time at n_q 16, n_kv 8k improves < 2%. Effort S.
**Compounded-gain check: KEPT.** No CPU tree-verify consumer exists (AK-DTREE is GPU-only); unlocks nothing; compounded = standalone < 1%.

Declined (dive-overturned, do not propose): PR #30370 GCN sparse FA (GCN4/5 only, dense prefix never qualifies); SGLang
verify_splitkv and Hydragen/DeFT prefix sharing (llama.cpp HIP FA already splits KV and reads the prefix once per verify).
