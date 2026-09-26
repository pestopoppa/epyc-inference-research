# Standalone experimental EXL3 CPU operators

EXL3-2 implementation seam, based on research commit `300951de`. No llama.cpp
integration, model inference, or production kernel change. The code is independent
of Torch and ggml. EXL3-1 owns canonical artifacts; this C++ API is an in-memory
experimental consumer, not a new serialization format.

## Numerical contracts

`reconstruct()` materializes row-major `[input, output]` weights using native
MUL1 or MCG fp16 codebook values. It uses ascending-index FP32 FMA for each H128
product, normalization `float32(1/sqrt(128))`, then the four explicit fp16 RNE
rounds: left H, multiply `suh`, right H, multiply `svh`. `prefill()` passes those
weights to a caller-supplied existing GEMM provider and adds FP32 bias afterward.
The test provider is SciPy's existing LP64 OpenBLAS `scipy_cblas_sgemm`; generic
LP64 `cblas_sgemm` is also accepted. Provider accumulation order can differ from
the deterministic ascending-FMA output oracle, so provider output is tested
with `abs(error) <= 2e-6*(1+abs(reference))`.

`dense()` is the **fused numerical path**: input scaling and FP32 H128, codebook
decode/GEMV, FP32 output H128/scaling, crop, optional FP32 bias. It intentionally
does not materialize weights or round the four transform stages to fp16. Its
output proposition is distinct from materialized reconstruction. ISA outputs
must match the scalar fused oracle exactly under the supplied compiler flags.
The fixed real-fixture input has a predeclared relative-L2 envelope of 0.003
against materialized output. This is an operator fixture envelope, not a model
quality guarantee or a universal bound for arbitrary ill-conditioned inputs.

`mul1_q8()` is a second, explicitly approximate path. It symmetrically quantizes
the transformed activation into [-127,127] and uses the donor's affine
`(bytesum(state * 0x83dcd12d)-510)*fp16(0x1eee)` codebook approximation. BW uses
vector byte sums and integer products; VNNI uses `vpdpbusd`; VBMI uses byte
permutation to replicate signed activations before `vpdpbusd`. Int64 output
accumulators avoid the donor's large-K int32 overflow limit. All ISA versions
must match the scalar Q8 oracle exactly. The real-fixture envelope versus the
materialized oracle is relative L2 < 0.025. MCG has its own vector multiply,
mask/xor, half conversion/add/round and FP32 vector dot path; it is never routed
through the MUL1 affine identity.

Inputs must use RNE, disjoint x/y buffers, finite activations/scales/bias, and
ordinary finite arithmetic ranges. Caller buffers must actually hold their
specified row/stride extents. The API rejects unsupported ISAs, invalid K,
geometry, payload lengths, strides and expert/input-row IDs before output writes.
`indexed()` validates every selected route before computing, supports duplicate
expert IDs, and requires common logical expert geometry. Each route writes one
output row and consumes its own expert's scales. Empty routing is a no-op for
valid nonempty input capacity. The GEMM callback must honor its extents and not
throw after writing output.

## Layout and dispatch

`Matrix` carries integer K1-K8, logical dimensions, explicit padded dimensions
(multiples of 128), native or band8 layout, packed u16 words, scales and optional
bias. Input tails are zero-filled **before** H128; padded output participates in
the complete H128 before crop. Partial tiles are carried in padded storage.
Native indexing is `(kt*tiles_n+nt)*16*K`; band8 is
`((nt/8)*tiles_k*8+kt*8+nt%8)*16*K`. Exact payload extent is required; no sentinel
words or readable overrun are assumed. Repack is lossless storage permutation.

The band8 representation is temporary derived storage, not canonical evidence.
This API deliberately has no artifact-digest identity; the EXL3-1 consumer must
bind the validated native artifact digest, format/layout version and this
repack recipe before retaining/publishing any derived cache. Do not infer a
canonical artifact identity from a C++ `Matrix` or the test binary fixture.

`indexed(..., grouped_by_k=true)` selects a K-specialized entry outside the route
hot loop, iterating K buckets; `false` reads runtime K per expert. Both preserve
route output order. The microbenchmark compares complete operator boundaries,
including validation/activation transforms, and is a cache-resident candidate
observation, not inference throughput. No many-row packed GEMM optimization is
implemented: decode accepts 1-4 rows; prefill reconstructs and delegates to the
existing GEMM provider. Any reconstruction cache needs profiling and artifact
identity work first.

## Tests

From the repository root, with an installed LP64 CBLAS library:

```bash
EXL3_BLAS_LIBRARY=/path/to/libopenblas.so experiments/exl3_cpu/run_tests.sh
EXL3_SANITIZE=1 EXL3_BLAS_LIBRARY=/path/to/libopenblas.so experiments/exl3_cpu/run_tests.sh
```

If the running Python has SciPy installed, its bundled LP64 library is discovered
automatically. `EXL3_BUILD_DIR` chooses a retained build directory; otherwise a
fresh temporary directory is used. Builds use C++17, baseline x86-64, function
ISA multiversioning, warnings as errors, and disabled implicit FP contraction.
Unsupported ISA tiers are reported; an explicit request is rejected. The
current EPYC diagnostic run executes scalar/BW/VNNI/VBMI. No unsupported tier is
silently treated as if it ran.

Coverage includes both codebooks, K1-K8, 1-4 rows, both layouts, logical tails,
padded extents, nontrivial strides, filled output guards, zero input, invalid
routes/shapes, bias, and seven-row prefill against OpenBLAS. Scalar codebook
values exhaust all 65,536 states against independent fp16 LUTs. Packed-tile tests
use exact allocations, including under ASan/UBSan.

The three **mandatory real** fixtures are small 128x128 tile submatrices from
MUL1 K3/K4 and MCG K4 expert payloads; they are not whole experts. The committed
manifest records donor paths, model branch/revision, slice and hashes. Missing,
truncated or corrupt fixtures fail; tests never fetch, regenerate, or skip them.
`make_fixtures.py` is an explicit refresh utility, using independent Python
big-integer extraction, NumPy codebooks and explicit FP32 FMA emulation for the
canonical oracle. Frozen MCG LUT SHA-256 is
`95383563929baa9d1ae5de0be22490284382673620bdd85929feea7bdbef04a9`.

## Donors and evidence scope

Arithmetic/layout were checked against the retained exllamav3 v1.4.6 donor
`499890c75` and the newer EXL3 surface donor under
`tmp/intake-jev-exl3/exp-exl3/repo1/engines/tools/exl3hf_surface.py`. This
implementation is independently written. The old CPU donor's Q8 affine formula
is not substituted for the CUDA codebook's single fp16-rounding rule.

The direct shell test is a **diagnostic**, not a measurement constitution
attestation or acceptance receipt. Acceptance must run after the native
EXL3 verifier writer is live, binding proposition, canonical fixture/artifact,
read-set and checker digests. Timing (`--bench`) likewise requires the native
measurement writer and applicable region ownership before it is used as evidence.
No performance selection or EXL3-2 gate closure follows from raw stdout alone.

GitNexus status on the base was current; upstream-impact queries failed because
the graph WAL did not match its database. No graph-risk classification is
claimed. This directory is additive and has no existing tracked callers.

## Prospective verification

After the shared EXL3-1 files are integrated, build with `run_tests.sh --build-only`
and pass the retained binary to `run_evidence.py`. The wrapper requires an
explicit writer SHA-256, binary, BLAS library, the committed
`fixtures/canonical_bindings.json`, and a fresh output directory. Canonical
artifact paths are relative to `--canonical-root` (defaults to this repository).
It reopens all canonical tensor bytes, compares packed/scales and transformed
weight hashes to the C++ fixture, snapshots the checker and complete listed read
set before execution, rejects drift, and seals/projections-checks the resulting
native row. The wrapper requires all four ISA tiers for this suite's full
proposition; unsupported hardware cannot receive a misleading full-suite pass.

The immutable pinned writer is
`8caeb33dbb12986fadc385afe25d22bd791b036253c736f9527e67a55f85e268`.
MUL1 fixture bytes were checked by bounded HTTP range requests against HF commits
`69e33439ae950f17bcbe95c98f117d80f759ab6d` (K3) and
`55a732e0c4c3d4614bc42b68493bb930d9b02c0a` (K4); exact offsets and digests are in
`fixtures/mul1_k{3,4}.provenance.json`. The shared canonical manifests and fixture
references own the identities; the C++ blobs bind those identities as derivatives.
MCG K4's corresponding bounded comparison covered 33,284 bytes (trellis slice,
scales and marker) against `0xSero/GLM-5.3-Flash-EXL3-Q4` commit
`99cccdf0e8741715662c383828a9ea601990c125`, shard
`layers/layer-03-part-0.safetensors`; `fixtures/mcg_k4.provenance.json` records it.
