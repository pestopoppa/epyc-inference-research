# DS41-W5a: NMSE spot checks for the 1017 MXFP4 build (2026-10-21)

## What W5a should compare

The W5 gate asks for "NMSE spot checks". The LOW-BIT numerics contract (operator, 2026-10-06) is ppl
plus op-level NMSE plus coherence. That gives two separate questions:

1. **Weight fidelity**: GGUF dequant against the HF 1017 source tensors. This asks whether antirez's
   conversion lost anything.
2. **Served-kernel fidelity**: the CPU expert kernel the 1017 server actually runs, against an
   f32-dequant reference. This is the W5a checkbox text: served-shape MUL_MAT_ID, cap 5e-4.

## 1. Weight fidelity: DONE (no server, about 50 MB of HTTP range reads)

Source: HF `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e1088993b9b82de08f59de5be5a34e42170`. The
safetensors headers and tensor bytes were read by range request (`sthdr.py`). GGUF offsets come from
`gguf_tensors.json` (`gguf_types.py`, header-only). The comparison is `weight_fidelity.py`, and the
results are in `weight_fidelity.json`.

- **The experts are natively MXFP4.** HF stores each expert's `w1`/`w2`/`w3` as `I8` packed E2M1, two
  codes per byte with the low nibble first, plus an `F8_E8M0` scale per 32 input elements. That is
  OCP MXFP4 exactly. In 4 of 4 checked expert matrices (L3 e0 gate and up, L3 e7 down, L30 e200 gate;
  11.8M weights each), the GGUF E2M1 codes and E8M0 scales are **bit-identical** to the source, so
  **NMSE = 0.0** with max abs diff 0. The conversion is a lossless repack, so NMSE against the source
  is zero by construction for every routed-expert tensor.
- **Q8_0 non-experts** (attention `q_a`/`kv`, shared-expert gate; L3 and L30). The source is FP8 E4M3
  with an E8M0 scale per 32x32 block. NMSE against source is **2.92e-5 to 2.99e-5** on all 4 tensors,
  about 17x under the 5e-4 cap. That is the expected cost of moving E4M3 onto a linear int8 grid.
- **F16 tensors** (`hc_attn_fn`, `hc_ffn_fn` at L3; the source is F32). NMSE is **3.9e-8 to 4.6e-8**.

## 2. Served-kernel fidelity: the existing evidence is VACUOUS. Harness built; it runs in the booked slot.

The AK served-shape calibration
(`autokernel/campaigns/ak-ds41-1017-cpu-decode-20261021/store/served_shape/calibration-20261021T154353Z.json`)
reports **NMSE = 0.0** for all 28 MXFP4 cases, and that number proves nothing about the served path:

- `test-backend-ops` allocates weights in the plain CPU buffer (`ggml_backend_alloc_ctx_tensors`), so
  it never uses the `CPU_REPACK` extra buffer.
- The 1017 server repacks the experts (`ggml_repack_get_optimal_repack_type`: MXFP4 on AVX2 with
  `ne[1] % 8 == 0` gives `mxfp4_8x8_q8_0`). That repack is the ~276 GiB of private RAM per server, and
  `forward_mul_mat_id` then calls `ggml_gemv_mxfp4_8x8_q8_0` once per routed row.
- `use_ref` only disables fusion, iqk and FA tiling. iqk does not engage for MXFP4 here, so both arms
  run the same generic kernel, and 0.0 follows by construction.

`mxfp4_repack_nmse.cpp` (driven by `run_w5a_kernel.sh`) closes the gap:

- It loads REAL expert slices from the 1017 GGUF: 8 experts each from L3 gate, L3 down, L20 up and L39
  down.
- It uploads them into `CPU_REPACK` (the served path) and into a plain buffer (the generic path).
- It runs MUL_MAT_ID with n_used = 6 of 8 at n = 1, 2, 3, 4, 5, 8, 32, 128 and 512 tokens.
- It compares both outputs against a double-precision f32-dequant reference, with cap 5e-4.
- It also asserts `repacked == true` (`tensor->extra` is set), so the check cannot be vacuous.

It is step W5A of the slot driver (`artifacts/ds41_w6_dspark_1017_20261021/w6.2/driver.sh`,
2026-10-22T22:00Z, q0+q1, build role). It needs no server and about 0.5 GB of RAM. Results land in
`w5a_kernel.jsonl` through `analyze_post.py`. The expected magnitude is Q8_0 activation-quantisation
noise, about 1e-5.
