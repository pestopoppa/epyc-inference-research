# EXL3 portable contract v1

This is a project-owned experimental artifact/oracle/evidence boundary. It runs
on standard Python without Torch, GPU allocation, model loading, or inference.
No source-available ShapleyMCG implementation is included or imported.

## Native packed view

`contract.load(manifest.json)` returns `(manifest, payload_bytes)`. Planning
validates **every matrix's** metadata before reading any tensor. Binding verifies
all sizes and SHA-256 digests before reconstructed outputs are allocated. The
canonical manifest hash is SHA-256 of UTF-8 sorted compact JSON, excluding only
`artifact_sha256`; each referenced native tensor also has its own byte digest.
Duplicate JSON keys, unknown fields, ambiguous roles/domains, conflicting K and
rate, bad paths, changed bytes, unsupported layouts and oversized requests refuse.
The portable admission limit is 1,048,576 reconstructed elements across all
matrices, 256 activation rows, and 1,048,576 output elements.

Each Q/K/V/O/gate/up/down/head logical matrix owns an ID, role, logical and padded
`[input, output]` shape, private trellis/suh/svh/optional bias descriptors,
codebook, source digest, and explicit dense or local/global expert domain.
Fused QKV/gate-up naming is not inferred. A caller must split logical descriptors
without repacking the tensor data. Native tensor files are read without layout
conversion. Backend-derived files must bind the canonical manifest digest through
`epyc.exl3.repack.v1`; `bind_repack` refuses another input identity.

The native trellis shape is `[padded_input/16, padded_output/16, 16*K]` little
endian uint16, equivalently `8*K` little endian uint32 per 16×16 tile. Tiles are
input-block-major, then output-block-major. In each uint32 the circular bit stream
walks MSB first. State `i` is the 16-bit window starting at
`((i+257)*K-16) mod (256*K)`. Matrix coordinate `(r,c)` selects state
`((c%8)*4+(r%8)//2)*8+r%2+(r//8)*2+(c//8)*4`.

V1 admits integer K1–K8, `rate_x2=2*K`, and exactly `mul1` or `mcg`. Half-bit rates,
legacy LCG, unknown scale conventions, and unknown geometry fail before tensor
allocation. Adding them requires a new validated contract, not an implicit cast.

## Reconstruction and operator arithmetic

MCG multiplies a uint16 state by `0xCBAC1FED` modulo 2^32, applies
`(product & 0x8FFF8FFF) XOR 0x3B603B60`, interprets each 16-bit half as binary16,
and rounds their sum to binary16 RNE. This re-derives the three instruction path:
integer multiply, Boolean LUT, half addition.

MUL1 multiplies by `0x83DCD12D` modulo 2^32, sums four bytes, interprets
`0x6400+sum` as binary16, and performs one binary16 FMA with multiplier bits
`0x1EEE` and addend bits `0xC931`, rounding once to binary16 RNE. The historical
`(sum-510)*constant` Q8 affine path is an approximation and cannot prove exact
reconstruction.

Let Q be the reconstructed raw `[input,output]` matrix and H the Sylvester H128
with coefficients `FP32(±1/sqrt(128))`. Each H matrix product uses ascending-index
FP32 FMA accumulation. The materialized reference is:

```
L = half(H @ Q)
L = half(L * suh[:, None])
W = half(L @ H)
W = half(W * svh[None, :])
y = ordered_fp32_fma(x @ W) + bias
```

All `half` operations are binary16 RNE. H blocks cover full padded dimensions.
Inputs are implicitly zero-padded; output columns are cropped **after** complete
weight transforms. Bias is added in FP32. `materialized_weight_fp32_fma_v1` names
this proposition. A runtime that transforms activations before a raw dot, or
normalizes after a butterfly, has different rounding. It needs its own named
path and numerical envelope; bit-exact parity must not be claimed across them.

## Fixtures and independent checks

The committed fixtures cover MUL1/MCG × K1–K8, a 125×123 logical matrix within
128×128 padding, three activation rows, nonzero bias, packed states, raw tile,
full raw/transformed hashes, transform vectors and complete operator outputs.
`fixtures.py` independently walks single bits, scatters lane/fragment coordinates,
implements the LUT bitwise, decodes IEEE half values arithmetically, and uses a
separate direct normalized-matrix interpreter to create golden outputs. It does
not call `oracle.py`.

Three real 128×128 native tensor slices are mandatory and committed:

| Fixture | Immutable model revision |
|---|---|
| real-mul1-k3 | turboderp/Qwen3.8-Flash-Next-exl3 @ 69e33439ae950f17bcbe95c98f117d80f759ab6d |
| real-mul1-k4 | turboderp/Qwen3.8-Flash-Next-exl3 @ 55a732e0c4c3d4614bc42b68493bb930d9b02c0a |
| real-mcg-k4 | 0xSero/GLM-5.3-Flash-EXL3-Q4 @ 99cccdf0e8741715662c383828a9ea601990c125 |

They preserve input tiles 0:8 and output tiles 0:8, plus the corresponding
Hadamard-block transform vectors. They test real-weight operators on those
submatrices; they do not establish complete model behavior. MUL1 source receipts
record byte-exact immutable HTTP range comparisons of 31,232/41,472 total bytes
(trellis row slabs plus both scales). MCG independently rechecks 33,284 remote bytes (native trellis prefix, both
scale vectors and the codebook marker) against the same immutable model revision.
A asserted revision without byte-verification evidence is rejected. `import_real.py` requires the independent CPU fixture's full raw and
transformed hashes to match the portable golden before importing. No missing
real fixture can silently skip: `require_real_suite` refuses G1.

`check_donor.py` executes the external Apache-2.0 revision
`3753c33b0b70737a60a8859f4c4ad0b136a0ae22` of vllm-exl3 as a second state oracle,
checks its license at that revision, and records the exact read set. It does not
vendor donor implementation bytes into this repository.

## Prospective evidence

`evidence.write(output_directory, fields)` is the single prospective writer for
`epyc.exl3.measurement.v1` and `epyc.exl3.verifier.v1`. The caller supplies native
facts. It authors producer identity/hash, locator-derived row identity, the
attestation and self-hash. Existing row IDs cannot be overwritten.

Measurement locators are run × arm × backend × operator × shape × metric. Each
row carries one directed metric, units, raw vector, arithmetic-mean aggregation,
scored repetitions/basis, category, timezone date, protocol eligibility/ID,
comparator, and model/artifact/source/binary/library/toolchain/hardware/residency
identities. An ineligible record must have an empty protocol ID.

Verifier locators are run × fixture × backend/path × decided proposition. Rows
bind exact checker bytes, fixture and read-set digests, verdict, and the exact
claim/proposition identity. `validate` reopens checker/read-set/attestation bytes.
A failed verdict remains a failed verifier; it is never changed into a pass.
Both schemas have experimental/no-promotion authority and no production flag.

The root adapter `scripts/vidya/adapters/exl3.py` pins the reviewed producer hash,
reopens native evidence, projects the existing `ClaimTuple`, and delegates all
grading to its existing class ladder. CLI source names are `exl3-measurement` and
`exl3-verifier`. No older records are backfilled. Example native dictionaries are
in `test_exl3.py`; the full checker writer is `verify.py`.

```
python3 -m unittest scripts.kernel_rnd.exl3.test_exl3 -v
python3 -m scripts.kernel_rnd.exl3.verify --output /durable/new-exl3-run
python3 -m scripts.kernel_rnd.exl3.check_donor --repo /path/to/vllm-exl3 --output /durable/new-donor-run
```

Receipts name absolute read-set paths and are immutable; moving the source or
changing checker bytes requires a fresh verification run, not rewriting old rows.
No throughput, quality, teacher-logit or full-model conclusion is emitted. The
25-window final quality panel remains untouched; future development must use
nonfinal windows and keep quality receipts separate.
