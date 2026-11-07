# MI210 asm/HIP port: open and proposed seeds after the 2026-11-06 port night

Source: epyc-root `handoffs/active/autokernel-loop.md` AK-ASM-* boxes, the results summary in
`progress/2026-11/2026-11-06-main-inf22.md`, and `artifacts/analysis/mi300-asm-port-inventory-20261105.md`.
Measured KEEP/DROP receipts are in `loop/program.md` -> *MI210 port results 2026-11-06*. Nothing below
re-opens a decided arm: H1/H1b, H7, H2a, H4, H6, H9, H9B, H10, H11, H12, H13, H14, H15, H5-PORT hd256
and T0 are decided. Grade on arrival: `design_prior` -- none of these is measured.
Baseline for every seed: champion C6 store (plus the AK-ASM-FOLD candidate once it exists), HIP graphs on.
Pitfalls to design around: *gfx90a measurement pitfalls* in `program.md` (op-only KEEPs need a real-model
witness at WARN plus PPL).

## AK-ASM-H3 -- C6 Q8_0 GEMV prefetch confirmation A/B (low priority, likely closeable)

**CLOSED 2026-11-07: measured DROP 0.9465 (n 5/5), `ak-asm-t0-h3-gate-20261105`; see program.md MI210 table.**

`GGML_CUDA_Q8_PREFETCH=1` vs 0 at idle decode, Qwen3.8-27B Q8_0, tg128. The same async weight prefetch
mechanism is already a measured negative: `akm-hist-q8-prefetch` (-12%/-18% on 27B dense, 2026-07-11).
Run only as a 5-pair confirmation on C6 if a free GPU slot exists. Do not author new prefetch code.
**Falsifier:** median on/off < 1.03 over 5 interleaved valid pairs (expected, per akm-hist-q8-prefetch).

## AK-ASM-H8 -- fused MoE expert GEMM (sorted tiles + fused gate/up + GLU) for A3B (Q38FN cells only if a hybrid deployment is adopted)

**STATUS 2026-11-07: H8 r4 FAIL 1.0981; successor H8b KEEP op-level 1.2905 on C7 (`ak-asm-h8b-gate-r2-20261106`). Open caveats -> seed 09 AK-ASM-H8c.**

Open: r1 crashed in correctness (MUL_MAT_ID rc=-6); r3 (store cb973fc37) produced perf cells but no
verdict. Arms: technique port into `mmid.cu` (current), CK 2-stage MoE (sorting unsupported on gfx90a per
the AITER step-1 trial), per-channel int8 requant (quality gate). Est. A3B/Q38FN prefill +10-20%,
verify +5-10% [inferred]. Q38FN is CPU-served in production (does not fit VRAM); its cells are informational.
**Falsifier:** geomean C6/new < 1.10 over A3B/Q38FN MUL_MAT_ID cells at M >= 512, any M <= 8 cell slower
than 0.97x, or MUL_MAT_ID correctness failure.

## AK-ASM-T1-SPILL -- retargeted asm families on the spill path, timed vs C6

782/782 non-fp8 gfx942 `.co` assemble for gfx90a and most families pass the MI210 harness; the spill-path
harness (r3) and the AITER-vs-C6 timing row were interrupted. Time each family that passes the harness
(PA bf16/fp16 hd128, fmoe int8 g1u1/g1u0, i8gemm, bf16gemm fp32-out) against the C6 op at production shapes.
Only families that beat C6 by >= 1.10x at a production shape become integration seeds.
**Falsifier:** no family >= 1.10x vs C6 at any production shape, or harness correctness failure.

## AK-ASM-H16 (proposed) -- MFMA-based skinny Q8_0 GEMM for M 12..16 (27B verify)

H1b (wvSplitK GEMV) and H1 v1-v3 both lose at M >= 12 on 27B large shapes; C6 MMQ sits at ~31% BW there.
Build an MFMA 16x16x16 f16 tile kernel that dequantizes Q8_0 in registers and streams weights once for all
M rows (bytes-bound, not compute-bound: int8 MFMA = f16 rate, so use f16). Expected: 27B DFlash verify
step at M 12..16 1.3-1.6x on the large GEMMs if it reaches ~60% BW (927 GB/s seen in H1b at M <= 11).
**Falsifier:** geomean C6/new < 1.10 at 27B shapes M in {12,16}, or NMSE outside test-backend-ops tolerance.
Effort M.

## AK-ASM-H17 (proposed) -- profile the combined DFlash2 verify step after the fold

With H1b + H10 + H5-PORT folded, re-take a rocprofv3 kernel-time share of one 27B DFlash2 verify step at
8k/32k/64k. The next seed is whichever family holds >= 15% of step time and sits under 50% of its roofline.
This is a measurement seed (no kernel): it decides H16 vs further attention work.
**Falsifier:** no family >= 15% of step time below 50% of roofline (the verify step is balanced; stop).
Effort S.

## AK-ASM-H18 (proposed) -- hd256 CK FMHA under HIP graphs at long context (after H5-PORT-R4)

H5-PORT measured +3% pp2048 only. FA share grows with context: extend to pp8k/pp32k and verify at 32k/64k,
with graph capture (6148b1333). Expected +10-20% prefill at >= 32k [inferred from the 3.71x op ratio and
the FA share at long context].
**Falsifier:** pp32k gain < 3% vs C6, or graph-captured run fails/regresses.
Effort S (measurement on an existing store).

## AK-ASM-H19 (proposed) -- A3B on GPU: zero graph splits (MANDATORY objective)

Operator policy 2026-11-06: Qwen3.8-27B and Qwen3.6-35B-A3B each run entirely on CPU or entirely on GPU.
Zero backend graph splits (CPU fallbacks) on their GPU decode and prefill path is a MANDATORY objective,
not an optimisation. A3B on MI210 (Q8_0, ~37 GB) fits. Enumerate every CPU fallback / graph split on the A3B
GPU graph, decode AND prefill (`GGML_SCHED_DEBUG` / GGML_SCHED split count, per token and per ubatch), and give
each offending op a GPU kernel using the H13 recipe (GPU-side op coverage; H13 took Q38FN splits 294 -> 100).
Repeat until splits == 0. The same zero-split requirement applies to Qwen3.8-27B on GPU: run the census on its
decode and prefill graphs and close every split the same way.
**Falsifier / done:** split count 0 (decode and prefill) with a `test-backend-ops` correctness pass for every
new kernel. Any remaining split is a blocker to be closed, not a tolerated cost; a decode-speed gain is
reported but is not the done criterion.
Effort S-M per split class.

## AK-ASM-H20 (proposed) -- prefill GEMM: shape-tuned rocBLAS / hipBLASLt selection for 27B and A3B

Every authored int8 / fused-dequant arm lost to dequant + rocBLAS (H4 0.48x, H9 0.52x, H9B 0.64x). The
remaining prefill lever is tier 2 of the targeting order: exact-shape hipBLASLt / rocBLAS algorithm
selection for the 27B / A3B prefill GEMMs (n 512/2048), plus fusing the f16 convert into the previous op's
epilogue (convert is ~15% of prefill). Expected 27B pp2048 +5-10% [inferred].
**Falsifier:** best tuned algorithm within 3% of the default at every production shape.
Effort S-M.

## AK-ASM-H21 (proposed) -- QSA + MoE sparse path for Q38FN prefill at long context (if a hybrid deployment is adopted)

Q38FN does not fit 64 GB VRAM and is a fully CPU model in production; this seed applies only if a hybrid
CPU+GPU deployment is adopted (see H22). H12 was measured at kv=32k op level. Extend the fused QSA kernel to the prefill (n_q >= 512) path and time
pp32k/pp80k on Q38FN end to end. Expected +10-25% long-context prefill [inferred: dense masked FA reads all KV].
**Falsifier:** pp32k gain < 5%, or output mismatch beyond FA tolerance.
Effort M.

## AK-ASM-H22 (proposed, INFORMATIONAL) -- Q38FN hybrid CPU+GPU investigation

Q38FN does not fit 64 GB VRAM; in production it is fully CPU-served. Hybrid CPU+GPU is uncommitted and worth
measuring only to inform the next stack decision; here graph splits are structural (CPU experts), not a defect.
Measure (a) the per-token cost of the structural CPU-expert round trips under `--n-cpu-moe` (sync + PCIe per
MoE layer), with and without HIP graphs; (b) a GPU-assists-prefill-only mode vs CPU-only prefill.
**Output:** a decision table (t/s decode, pp t/s, per-split cost) vs the CPU-only production baseline.
No champion claim, no promotion gate; informational only.
Effort M.
