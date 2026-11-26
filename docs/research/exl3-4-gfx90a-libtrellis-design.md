# EXL3-4 libtrellis design for gfx90a

This proposal pins libtrellis to 8e0e016e0cd13f153380bde2bec01add08eae85e and the standalone harness to epyc-inference-research dd090a874c22be58e09b18fc1fc86da7b46b1d85. Local line references are relative to the standalone source directory. [Harness source](https://github.com/pestopoppa/epyc-inference-research/tree/dd090a874c22be58e09b18fc1fc86da7b46b1d85/scripts/kernel_rnd/exl3_gfx90a).

## Offset decode

MUL1 h_grid masks the four product bytes with 0xFCFCFCFC. Each masked byte divided by four is a digit in [0, 63]. The digit sum s is at most 252, and 126 centers the integer grid as v = s − 126. Adding 4s to the FP16 bit pattern 0x6400 gives 1024 + 4s = 4v + 1528. Centered 4v spans [−504, 504] and remains exact after the FP16 subtraction of 1528. [Offset decode](https://github.com/nari-labs/nari-llama.cpp/blob/8e0e016e0cd13f153380bde2bec01add08eae85e/ggml/src/ggml-cuda/libtrellis/src/common.cuh#L89-L109), [centered decode](https://github.com/nari-labs/nari-llama.cpp/blob/8e0e016e0cd13f153380bde2bec01add08eae85e/ggml/src/ggml-cuda/libtrellis/src/common.cuh#L165-L175).

For the proposed offset arm, let q be the staged activation after scaling by 2^e and rounding to FP16. For each token and K stage, sum those same rounded halves in FP32. Correct the FP32 MFMA result as (dot(4v + 1528, q) − 1528 × sum(q)) × 2^-e / 4 before accumulating across stages. The centered arm supplies 4v to MFMA and applies the same 2^-e / 4 epilogue. Compare the arms using identical rounded inputs to isolate cancellation and accumulation error. [GEMV correction](https://github.com/nari-labs/nari-llama.cpp/blob/8e0e016e0cd13f153380bde2bec01add08eae85e/ggml/src/ggml-cuda/libtrellis/src/gemv.cu#L212-L222), [GEMM epilogue](https://github.com/nari-labs/nari-llama.cpp/blob/8e0e016e0cd13f153380bde2bec01add08eae85e/ggml/src/ggml-cuda/libtrellis/src/gemm.cu#L7-L16).

Current MUL1 sums unmasked bytes and applies the half-rounded FMA decode in kernels.hip:19–29. Preserve that legacy route and MCG. Select the proposed offset path using a separate profile and format identity after portable codebook and packing equivalence is established, or after explicit conversion and a new artifact identity. Size the converted layout from its own stride before dispatch.

## Activation preparation

Current prepare_activations multiplies x by suh before normalized H128, rounds to half, and stores a float at [route][padded_input]. Padded positions use zero. The current MFMA constructs its half fragment on load (kernels.hip:62–74,100–114). The proposed preparation preserves those transform and padding rules, then stages halves in native B-fragment order for a 16-token tile. Qualify each new profile against its portable oracle at scale one first. Then qualify one power-of-two exponent per token and 128-input segment, with its reciprocal and FP32 sum of the staged halves available for each of the eight K=16 stages. CUDA libtrellis lays out its B fragment in a separate prologue. [Prologue source](https://github.com/nari-labs/nari-llama.cpp/blob/8e0e016e0cd13f153380bde2bec01add08eae85e/ggml/src/ggml-cuda/libtrellis/src/gemm.cu#L60-L113).

## Grouped launch

The proposed 256-thread workgroup assigns four wave64s to four output tiles for the same 16 tokens. Multiple matrices may share a prepared input stage when token rows, input extents, transform, input scale values and scaling policy match. Keep codebook/format, packed stride, output bounds, output scales and bias per matrix. Split groups when preparation differs. Current grouped eligibility uses rate_x2 and codebook and the MFMA loop launches per expert (kernels.hip:246–275). Upstream tq_gemv_group shares a raw input pointer while retaining member-specific preparation and workspace allocations. [Grouped source](https://github.com/nari-labs/nari-llama.cpp/blob/8e0e016e0cd13f153380bde2bec01add08eae85e/ggml/src/ggml-cuda/libtrellis/src/gemv.cu#L677-L806).

## LDS budget

The current hipMalloc arena is device global memory. bind() maps its fields inside that allocation, and the current MFMA kernel launches with zero dynamic LDS. See test_runtime.hip:46–48,149–151 and kernels.hip:185–203,272. MI210 provides 64 KiB LDS per compute unit (CU), shared by resident workgroups. [AMD MI210 specifications](https://rocm.docs.amd.com/en/docs-6.3.2/reference/gpu-arch-specs.html), [AMD LDS hardware](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html).

Proposed staging uses two stages, K=16 and 16 tokens, four packed weight tiles, with 256-byte-aligned regions. The local layout capacity is 32*b bytes per 16×16 tile at b bits per weight (8*b uint32 words). Gate any converted artifact against its actual footprint.

| Region | Bytes per stage at b=8 | Bytes for two stages |
|---|---:|---:|
| Packed code tiles | 4 × 256 = 1024 | 2048 |
| Shared FP16 activation stage | 16 × 16 × 2 = 512 | 1024 |
| Sixteen FP32 sums and sixteen FP32 reciprocals, padded to 256 B | 256 | 512 |
| Total | 1792 | 3584 |

Total LDS is 3584 bytes per workgroup, below 65536 bytes per CU. Synchronize before consumption and reuse of each stage. Keep decoded fragments and accumulators in registers. Store the result in global memory for the existing output transform. Compiled LDS and register usage, bank conflicts and occupancy need qualification.

## Wave64 MFMA

Use __builtin_amdgcn_mfma_f32_16x16x16f16 with FP16 input and FP32 accumulation. For lane L, define r = L % 16 and g = floor(L / 16). Every lane owns four register indices j = 0..3 and uses k = 4*g+j within the K tile. The operands and result are A[r,k], B[k,r] and C[4*g+j,r]. Current harness operands are activation A and packed weight B with route-major output (kernels.hip:93–114,138–142). Proposed operands are weight A and activation B. Map the transposed output C[output, token] back to token-major storage before the existing epilogue.

A single CUDA warp32 owns one mma.sync.m16n8k16 tile. The libtrellis GEMM uses eight output tiles per CTA with one per warp. The proposed CDNA2 fragment order follows the wave64 mapping above. [AMD CDNA2 MFMA builtin and register-layout reference](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/latest/compiler-builtins/cdna/cdna2-dense-mfma-builtins.html), [AMD instruction calculator](https://github.com/ROCm/amd_matrix_instruction_calculator), [libtrellis tiling](https://github.com/nari-labs/nari-llama.cpp/blob/8e0e016e0cd13f153380bde2bec01add08eae85e/ggml/src/ggml-cuda/libtrellis/src/gemm.cu#L9-L14).

## Harness coverage

At the pinned source, test_runtime.hip:63–80 defines exhaustive procedural-value and packed-state tests and four 256-cell MFMA lane-map patterns. Lines 149–164 and 269–280 define full-operator comparisons, output-tail guards, fixed-order repeat checks and named grouped dispatch counters. The pinned validation records report hip_runtime_executed=false. Device qualification remains open.

The offset sum correction, native B-fragment preparation and grouped shared-input reuse are proposals with coverage still open. Add parity controls for offset versus centered operands with identical rounded halves, a lane probe for the swapped operand and store mapping, and split/reuse tests with differing input scales and transforms. Qualify the final compiled stage size and synchronization before performance claims.

**Review provenance:** the A3B CPU draft and GPU thinking correction both failed independent source checks. Main prepared and reviewed this replacement against the pinned implementation and primary documentation. No kernel, serving configuration, device qualification or model measurement changed.
