# NEW-11 — iqk coverage of the ISTA-DASLab GSQ-RCO Qwen3.8-Flash-Next builds

Date 2026-10-21. Task: `iqk-iquant-enablement.md` NEW-11 (intake-2206). Text/metadata only: **no weights downloaded.**

## Method and provenance
- Repo `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` pinned at sha `ed59f92082b1e93c0e96d60a8b11aab089b52f09`.
- Per-tensor **types** from the four published `tensor-allocation/*.rco-allocation.txt` files (1223 tensors each, shard 1).
- The allocation files carry no shapes, so **byte sizes** come from an HTTP `Range: bytes=0-16777215` read of each shard-1 GGUF (header + KV + tensor-info table only; 16 MiB, ends before any tensor data; held in scratch only). Sizes = dims x ggml block geometry; cross-checked against tensor-offset deltas (0 mismatches in all four) and against HF API shard sizes (parsed tensor bytes + ~11 MB header = API size, e.g. Q2_0 37.613 GB + header vs 37.624 GB). Shard 2 header read for the Q2_0 build only: one tensor, `per_layer_token_embd`, IQ4_NL, 28.80 GB; the other three builds' shard 2 has byte-identical size (28,800,138,432) and the card says IQ4_NL, so it is carried over, not independently parsed.
- Side result: the stored ggml type id of the Q2_0 tensors parsed as **42 = `GGML_TYPE_Q2_0`** (the layout our v10 `ggml.h:432` defines; block = fp16 scale + 64 x 2-bit = 18 B / 64 weights). This closes the "Q2_0 layout unverifiable without a header" gap in intake-2206.
- iqk classification from `ggml/src/ggml-cpu/iqk/iqk_dispatch.cpp` @ `ffc1bac82` (v10): `iqk_typeA_supported` = Q4_K, Q5_K, Q6_K, Q8_0 (gated by `iqk_q8_0_enabled()`), Q4_0, Q5_0, Q4_1, Q5_1, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS. **Explicitly not iqk:** Q2_K/Q3_K (static_assert), Q2_0, IQ4_NL, IQ1_M (stub), IQ1_S, float types. Extra shape rule: `iqk_mmid_shape_supported` — IQ3_XXS MoE (routed experts) is iqk **only when n_tokens == 1** (decode); batched/prefill MMID falls to the native kernel.
- Reproduce: `gsq_rco_iqk_coverage.py` (header parser) -> `gsq_rco_coverage.json` (this directory).

## Routed-expert type mix (share of routed-expert bytes)
| Build | Routed GB | iqk types | Non-iqk types |
|---|---|---|---|
| Q2_0 | 33.97 | none | Q2_0 100% |
| IQ2_XS | 35.45 | IQ2_S 51.5%, IQ2_XXS 13.4% | Q2_0 31.9%, IQ1_M 3.1% |
| IQ3_XXS | 42.91 | IQ3_S 21.8%, IQ2_S 12.5%, IQ2_XS 11.3%, IQ2_XXS 9.1%, IQ3_XXS 9.0% (decode only) | IQ4_NL 19.8%, Q2_0 16.5% |
| IQ3_S | 50.29 | IQ3_XXS 21.7% (decode only), IQ2_S 21.4%, IQ3_S 14.3%, IQ4_XS 1.8% | IQ4_NL 36.6%, Q2_0 4.2% |

`ffn_down_exps` rows are 640 wide (not a multiple of 256), so they are confined to Q2_0 / IQ4_NL by construction: every build has a non-iqk floor on the down projection.

## Fraction of bytes on non-iqk kernels
"Decode" counts IQ3_XXS experts as iqk (n_tokens == 1); "prefill" counts them as native. Quant-matmul bytes exclude the ~1.5 GB of F32/F16/BF16 tensors (483 BF16 tensors, indexer/hyper-connection class), which sit on non-iqk float paths in every build and are listed separately.

| Build | Shard-1 GB | Routed experts non-iqk (decode / prefill) | All quant-matmul bytes non-iqk (decode / prefill) | Float GB | Whole file incl. 28.8 GB shard 2, decode view |
|---|---|---|---|---|---|
| Q2_0 | 37.61 | **100.0% / 100.0%** | 96.4% / 96.4% | 1.49 | 98.0% |
| IQ2_XS | 39.22 | 35.0% / 35.0% | 33.1% / 33.1% | 1.54 | 62.9% |
| IQ3_XXS | 47.03 | 36.3% / 45.3% | 34.3% / 42.8% | 1.54 | 60.6% |
| IQ3_S | 54.81 | 40.8% / **62.5%** | 38.6% / 59.1% | 1.54 | 60.9% |

Non-routed quant bytes (attention/SSM/dense/shared-expert/embedding) are only 2.8-3.5 GB per build and almost all K-quant/IQ4_XS; the Q2_0 build is the one with a non-iqk piece there (Q3_K 0.77 GB, which v10 explicitly rejects from iqk).

Shard 2 (28.8 GB, `per_layer_token_embd`, IQ4_NL, identical in all builds) is a per-token **row lookup (get_rows), not a matmul**, and is lazily mmapped, so its "non-iqk" status costs no kernel throughput; the shard-1 columns are the meaningful ones. The whole-file column exists only because the task asked for bytes.

Reference: the served UD-IQ4_XS file is essentially IQ4_XS/K-quant, i.e. ~all matmul bytes on iqk.

## What this means
1. **Q2_0 build: the entire routed-expert stack (34 GB, 100%) has no iqk kernel**, plus Q3_K dense weights that v10 rejects from iqk. The card's 367 t/s prefill / 93 t/s decode headline cannot transfer: our path would run the native ggml Q2_0 kernel exactly where the bytes are. Precedent: a ternary "Q2_0" failed to load on experimental-v7; ISTA's uses upstream id 42 so that failure mode may not recur (unproven).
2. **IQ2_XS build**: 32% of experts on Q2_0 (down projections) and 3% on IQ1_M (stubbed); 35% non-iqk, no prefill penalty.
3. **IQ3_XXS / IQ3_S builds** lean on IQ4_NL for `ffn_down_exps` (20% / 37% of routed bytes) with a Q2_0 floor (16% / 4%). Because v10 restricts IQ3_XXS MMID to single-token calls, the IQ3_S build's prefill non-iqk rises to 62.5% (vs 40.8% decode).
4. The iqk gain was measured prefill-dominant (decode ~0% on BW-bound Q8_0; iqk-iquant-enablement.md section 5), so for a prefill-heavy lane the **prefill** column is the one that matters.

## Recommendation for the Q3 (NEW-10) download decision
- **Do not download the Q2_0 build** (100% routed on native). **Deprioritise IQ2_XS** (card task-avg 89.2 vs 93.1 BF16, still 35% non-iqk).
- **If a download is approved, take IQ3_XXS first** (75.8 GB total on disk; 36% decode / 45% prefill routed non-iqk; card task-avg 92.57), then IQ3_S (83.6 GB; card 93.26) **only if** IQ3_XXS clears load smoke and PPL/KLD, because IQ3_S has the worst prefill coverage (62.5% routed non-iqk) and sits within 4 GB of the served 88 GB UD-IQ4_XS, so its size advantage is thin. The 28.8 GB shard 2 is shared by every build, so a second build costs only its shard 1 (+47/+55 GB) if shard 2 is kept.
- Set NEW-10's expectation as a **quality-per-byte** test (GSQ allocation vs UD-IQ4_XS), not a speed win; this table is the per-layer fallback prediction to check against. The card's `--lazy-mode` does not exist in v10 (`--tensor-read-lazy`, mapping unverified).
- The disk/download approval itself stays the operator's, per NEW-10; NEW-11 is complete.
