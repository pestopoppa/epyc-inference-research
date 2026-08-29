# The Tensile GEMM on prefill is OUR dispatch decision, not a vendor wall

Read this before proposing anything on the `pp512` surface. An earlier note in this
program described the ~51% rocBLAS/Tensile share as "not ours to patch". That was
wrong, and the correction is the most actionable thing in this file.

## What actually happens

`ggml_cuda_should_use_mmq()` — **`ggml/src/ggml-cuda/mmq.cu:240`, in our tree** —
decides, per matmul, between MMQ (our quantized kernels) and the
dequantize → convert → rocBLAS/Tensile GEMM path. Traced for the contracted case
(gfx90a = CDNA2, `Q4_K`, `pp512` ⇒ `ne11 = 512`, `n_experts = 0`), against
`HOUSE_GPU_RECIPE`, which does **not** define `GGML_HIP_NO_MMQ_MFMA`:

| line | test | result |
|---|---|---|
| :247 | `Q4_K` in the supported switch | `mmq_supported = true` |
| :281 | `turing_mma_available` | false (AMD) |
| :332 (common.cuh) | `amd_mfma_available` = `IS_CDNA(cc)` | **true** |
| :299 | `IS_CDNA3` | false |
| :303 | `n_experts > 64 \|\| ne11 <= 128` | false (0, 512) |
| :306 | type in {Q4_0,Q4_1,Q5_0,Q5_1} | false |
| **:309** | **`ne11 <= 256 && (Q4_K \|\| Q5_K)`** | **false — 512 > 256** |
| :312 | | **`return false`** → dequant + Tensile |

So the entire vendor share exists because **one hardcoded threshold on line 309 says
256 and our prompt is 512.** Not a library boundary. A constant.

## AK-H-MMQ-1 — the CDNA2 `ne11 <= 256` cutoff for Q4_K/Q5_K is mis-set for gfx90a

Raising it (or removing it) routes `pp512` through `mul_mat_q` and deletes three
kernels from the profile at once: `dequantize_block_q4_K` (15.06%),
`convert_unary` (9.32%) and the Tensile GEMMs (51.33%) — **~76% of measured device
time** — replacing them with our own quantized matmul.

**Falsifier:** with the threshold raised past 512, `pp512` throughput fails to rise
outside the 0.973% floor over ≥5 alternating pairs. That is a complete answer either
way: a null says the cutoff is correctly placed for CDNA2 and the Tensile path
genuinely wins at this shape, which is worth knowing and is *not* currently
established anywhere in our record.

**Why this is the right first experiment on this surface.** It is a one-line change,
so critic pass 2 is trivial and the diff cannot creep. It does not touch the numerics
of any kernel — it only selects which existing, already-correct kernel runs — so it
sidesteps the failure that killed 7 of 10 run-9 iterations, where every patch to the
dequant path broke `MUL_MAT`. And its effect size is bounded below by nothing: it
moves three quarters of the surface.

**Scope note, so it is not confused with a settled item.** The settled list records
`MMQ_MFMA` OFF as a *build flag* decision (+23.09% on the 0.5B toy, +0.50% on the 27B).
This is a different question: not whether the MFMA MMQ kernels are enabled, but whether
the **dispatcher** hands them work at `ne11 = 512`. `MMQ_MFMA` being unhelpful on the
27B does not tell us where the CDNA2 crossover sits.

**Adjacent, cheaper probe if you want the answer before patching:** build once with
`-DGGML_CUDA_FORCE_MMQ=ON` (line :288 returns true unconditionally) and bench it against
the anchor. That answers "is MMQ better here" without editing the heuristic at all, and
it is a legitimate build-recipe arm rather than a source patch.
