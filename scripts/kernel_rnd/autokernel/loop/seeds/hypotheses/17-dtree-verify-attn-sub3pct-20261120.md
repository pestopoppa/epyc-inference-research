# Seeds from intake-dtree-verify-20261120 (tree-verify attention, sub-3% levers)

Source: research intake intake-2747..2751 + intake-2705 (Stage 2, 2026-11-20). Grade: `design_prior`. Gate class: futility-only.
Cost model: 27B 2.39 ms per added verify row; wvsk16 GEMM floor 2.0 ms/row (rows 8->16); A3B MMVF knob=8 1.28 ms/row.
Attention plus the rest therefore caps at ~0.39 ms/row on the 27B. The task-box lever (DTREE-ATTN-ROUTE) and its share
probe (DTREE-ATTN-SHARE) live in epyc-root `handoffs/active/speculative-decoding.md` under AK-DTREE. Do not duplicate
DTREE-0 wvsk16 or DTREE-0-Q8MF (GEMM side).

All seeds wait for DTREE-ATTN-SHARE: if FA < 0.10 ms per added row on both models, close every seed here.

## AK-DTREE-TAILMASK (proposed) -- skip mask reads over the committed prefix
llama.cpp FA loads the dense f16 mask for every KV tile; for a tree verify all prefix columns are visible and only the last W are
masked. Add a "mask covers tail W only" flag to fattn-tile so k < n_kv-W skips the mask load/add.
Expected: <= ~0.02-0.04 ms/row (5-10% of the attention residual), < 1% of a 27B verify.
**Falsifier:** op-level FA at n_q 16, n_kv 32k improves < 2% vs the dense-mask path. Effort S.

## AK-DTREE-PB-OCC (proposed) -- parallel_blocks occupancy at n_q 12-16
Tile FA splits KV by parallel_blocks (fattn-common.cuh:1259/1283 @36cd3af). Check blocks_num vs 104 CUs for the verify shape and
raise the split when wave efficiency < 95%.
Expected: 0 to ~0.39 ms/row bound, likely < 3%.
**Falsifier:** FA time vs KV length is already at >= 95% wave efficiency, or a larger split changes FA < 2%. Effort S.

## AK-DTREE-ANCESTOR-REG (proposed) -- in-register uint32 ancestor mask (Infernix)
Replace the f16 tail mask with one uint32 per query row applied to the MFMA 16x16x16 accumulators (v_bfe/shift per lane), the
mask arithmetic compiled out for chain. Functional parity with DTREE-ATTN-ROUTE's design; standalone gain < 0.1 ms per verify.
**Falsifier:** A/B vs the f16 mask path at 16 rows, 64k: delta < 2% -> drop. Effort S.

## AK-DTREE-CPU-KMASK (proposed) -- Zen 5 AVX-512 ancestor k-mask for CPU tree verify
CPU FA (ggml-cpu ops.cpp ~8916 @ffc1bac82) reads an f16 mask per KV position. Build an AVX-512 __mmask16 from the per-row ancestor
word (vpsrlvd + vptestmd) for the last W columns and skip the mask over the prefix.
Expected: < 1% of a CPU verify (CPU decode is weight-bandwidth bound).
**Falsifier:** CPU FA op time at n_q 16, n_kv 8k improves < 2%. Effort S.

Declined (dive-overturned, do not propose): PR #30370 GCN sparse FA (GCN4/5 only, dense prefix never qualifies); SGLang
verify_splitkv and Hydragen/DeFT prefix sharing (llama.cpp HIP FA already splits KV and reads the prefix once per verify).
