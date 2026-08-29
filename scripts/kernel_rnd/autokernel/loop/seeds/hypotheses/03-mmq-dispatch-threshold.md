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

---

## ANSWERED 2026-08-29 — AK-H-MMQ-1 is a NULL. Do not re-propose it.

Measured directly as a build arm, which needed no source patch: `GGML_CUDA_FORCE_MMQ=ON`
against the anchor, same model, same surface.

| | |
|---|---|
| effect | **+0.105%** on pp512 |
| pairs | 9 alternating, 1 warm-up pair discarded |
| estimator | median_over_median, one statistic on both arms |
| floor | 0.973% |
| decisive | **no — inside the floor** |
| drift | anchor +0.282%, candidate +0.213% (both under the floor) |
| residency | 18/18 invocations resident, peak 1.88 GB VRAM |
| correctness | `test-backend-ops -o MUL_MAT -b ROCm0` **passed** on the MMQ build |

**MMQ and the dequantize→convert→Tensile path are indistinguishable at `ne11 = 512` on
gfx90a.** Raising or removing the `ne11 <= 256` cutoff at `mmq.cu:309` would buy nothing
measurable. The cutoff is not mis-set; it simply does not matter for throughput at this
shape. This is a bounded null, not an absence of data: any true effect is smaller than
0.973%.

Note MMQ is also *correct* here — the oracle passed — so the cutoff is not protecting
against a numerical problem either.

**A first reading of this experiment said −1.469% and "MMQ LOSES".** That run had no
warm-up: the candidate arm climbed +4.324% across five pairs while the anchor stayed flat,
and the per-pair effect marched −4.491% → −0.037%. The headline was first-use cost. The
sign flipped once the arm was allowed to settle. Recorded here because the failure mode is
more transferable than the result: **an arm that is still warming produces a decisive-looking
effect in whichever direction it is warming.**
