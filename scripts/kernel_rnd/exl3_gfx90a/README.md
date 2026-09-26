# Standalone EXL3 on gfx90a

This experimental module implements EXL3-3/4/5 operator primitives without a model
loader, inference server, or changes to the frozen production kernel. GPU runtime
qualification is **open**. The committed validation record distinguishes compiled
code and host checks from device execution.

## Numerical and storage contracts

The canonical artifact is `epyc.exl3.artifact.v1`, defined by sibling
`scripts/kernel_rnd/exl3/contract.py`. `fixtures.py` validates it before creating a
small test transport whose header binds the canonical digest, source, model and
projection role. Native packed tensor bytes are copied unchanged. The transport
has no serving or model format authority.

Native tiles are input-block-major, then output-block-major. Each 16×16 tile has
`8*K` little-endian uint32 words. The circular stream advances MSB first; a weight
uses its trailing 16-bit state. The storage permutation is inverted into logical
input/output coordinates before assigning work to 64 lanes. Both codebooks round
their procedural result to binary16: MUL1 uses the exact half FMA constants and
MCG uses its masked/XOR half-add construction. The portable harness independently
extracts individual bits and enumerates the permutation.

Per-matrix `rate_x2=2*K` remains device-visible. Version 1 supports integer K1–K8;
odd per-matrix rates are an explicit `unsupported_rate` result before allocation.
A model can have a fractional average bit rate through mixed integer-K matrices.
Each local expert has a pointer-table entry, explicit global ID, exact codebook,
source digest, role, logical extent and padded extent. Count/sort tables have E+1
entries; compute tables have exactly E. Nonlocal routes carry -1 and never read
entry E.

Two numerical propositions are deliberately separate:

* `materialized_weight_fp32_fma_v1`: explicitly reconstruct normalized H128 matrix
  products in ascending FP32 FMA order, round each transform and scale to FP16,
  then compute an ascending FP32 dot. The temporary full weight matrices exist
  only when `Config.materialized_reference=true` is planned. Default planning
  reserves no dense weight materialization.
* `fused_activation_transform`: multiply input by `suh`, normalized H128, round
  to FP16, multiply directly from native packed weights, then normalized output
  H128 and `svh`; add FP32 bias after cropping and multiply by route weight.
  Its reference follows that operation order. The real-fixture harness also
  reports its maximum difference from the materialized proposition, without
  claiming the two are bit-exact.

Logical input padding is zero before all transforms; output cropping is after
all transforms. Every device K loop receives `padded_inputs`, a multiple of 16.
Synthetic checks poison unused output guard elements and cover logical dimensions
15/17, 31/33 and 127/129. Packed padded coordinates are part of the transform and
are never arbitrarily discarded.

## Compute and dispatch

`Compute::wave64_gemv` assigns each lane an output column and keeps the input
reduction in a fixed ascending order. `K=0` uses device metadata; K1–K8 have
specialized instantiations. No CUDA warp32 geometry is reused.

`Compute::mfma_prefill` is a separate entry point. It decodes native packed B
fragments directly into registers and executes
`__builtin_amdgcn_mfma_f32_16x16x16f16`. Inputs are FP16 and accumulators FP32.
For lane L, A uses row L%16 and K elements 4*(L/16)+j; B uses column L%16 and
the same K elements. Accumulator j owns row 4*(L/16)+j and column L%16. The mapping
is derived from AMD's [matrix-core documentation](https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/)
and [CDNA2 calculator](https://github.com/ROCm/amd_matrix_instruction_calculator).
Four nontrivial 256-cell checks are compiled; their device results are still open.
MFMA is never selected automatically as a performance winner.

`capability()` is the sole policy truth table:

| Condition in order | Named result |
|---|---|
| Invalid rate, codebook or layout | corresponding unsupported result |
| No routed rows | `empty` |
| Unified policy | `unified` |
| BC or extension absent under grouped policy | `batched_fallback` |
| Tokens or expert rows below 4 | `small_row_fallback` |
| Fewer than four active experts in a rate/codebook bucket | `single_expert_fallback` |
| Remaining grouped case | `grouped` |
| Capture requests a fallback | `capture_fallback_refused` |

BC and extension flags are caller-declared availability facts; they do not import
an external extension. Fallbacks execute the runtime-K kernel. Grouped buckets
execute specialized kernels. Host launch counters prove which named path was
enqueued and full output checks decide correctness. Actual HIP stream capture is
currently refused for every path because route uploads are host-driven.

Routes must remain token-major, then routing-k. Expert sorting only builds an
indirection array. Deterministic gather sums routing-k in order in FP32; atomic
gather is separately selected and labeled. The standalone library never silently
changes reduction or compute modes.

## Lifecycle and allocation

1. `plan()` validates identities, dimensions, codebooks, expert domains, exact
   packed byte lengths, token/route/prefill/scheduler capacities and aggregate
   arena bytes. E≤256, routes≤1024 and the default arena cap is 512 MiB.
2. The caller allocates all source tensors and one 256-byte-aligned device arena.
   `bind()` validates device and byte extents, maps stable views, uploads the
   pointer table, and optionally materializes reference weights. Plan/source
   buffers must remain alive until the binding stream has completed.
3. `schedule()` prepares bounded host arrays. `run()` validates identities and
   routes, checks device/extent/alias/capture boundaries, and enqueues kernels.
   Caller-owned schedule storage must remain alive until that stream completes.
   The runtime object has no malloc/new/hipMalloc/hipFree imports; the build checks
   this. HIP runtime internals are outside that source-level allocation claim.
4. The caller synchronizes and owns teardown. The harness checks every hipFree
   return and resets its own device context after all allocations leave scope.

The runtime identity binds all capacities, source/codebook/role fields, pointer
identities, matrix extents, arena offsets and bound arena/stream/device views.
Post-bind mutation refuses. Device launch errors are checked individually.

## Reproduction

Compilation never acquires or touches a GPU:

```bash
python3 scripts/kernel_rnd/exl3_gfx90a/build.py \
  --output /mnt/raid0/llm/tmp/exl3-gfx90a-build
python3 scripts/kernel_rnd/exl3_gfx90a/verify_host.py \
  --contract-root /path/to/integrated/research \
  --build /mnt/raid0/llm/tmp/exl3-gfx90a-build \
  --output /mnt/raid0/llm/tmp/exl3-host-unique-run
```

The build pins ROCm 6.2, `gfx90a:xnack-:sramecc+`, code object v5, strict FP32
math and wave64. It records compiler hash/version, all flags/source/object hashes,
disassembly, wave size and kernel register/LDS usage. ROCm's bundled output is
explicitly unbundled before disassembly. HBM traffic and achieved occupancy remain
`not_measured` until device profiling.

GPU execution uses `claimed_run.py` with one explicit authority mode:

* Delegated runs require an ACTIVE bus resource lease and a qualified, enabled
  GPU provider. Disabled/null providers refuse before device allocation.
* `--owner-run --holder inference --task-id TASK_ID` uses the existing Inference
  Main authority in root `agents/inference-main.md`. It requires the canonical
  roster to name `inference` as the unique eligible GPU owner and records
  `mode=inference_owner`, `lease=null`, the task identity, roster and policy hashes.
  This mode neither grants nor represents a delegated lease.

Both modes acquire the same canonical `gpu_device_claim`, recheck authority while
holding it, pass its descriptor to the child, reject foreign KFD co-residency, set
an isolated HIP library path and require in-window KFD/VRAM evidence. The binary
independently validates its inherited claim descriptor and the provider lock's
device/inode and exclusion. A lone environment claim ID has no authority.

```bash
python3 scripts/kernel_rnd/exl3_gfx90a/claimed_run.py \
  --contract-root /path/to/integrated/research \
  --build /mnt/raid0/llm/tmp/exl3-gfx90a-build \
  --output /mnt/raid0/llm/tmp/exl3-device-unique-run \
  --lease-id ACTIVE_LEASE_ID --holder OWNING_SESSION
```

An authority-only owner check acquires no claim and launches no GPU work:

```bash
python3 scripts/kernel_rnd/exl3_gfx90a/claimed_run.py \
  --contract-root /path/to/integrated/research \
  --build /mnt/raid0/llm/tmp/exl3-gfx90a-build \
  --output /mnt/raid0/llm/tmp/exl3-owner-unique-run \
  --owner-run --holder inference --task-id EXL3-3 --preflight-only
python3 scripts/kernel_rnd/exl3_gfx90a/test_claimed_run.py
```

Authority-only success does not establish physical availability or residency.
The owner-mode build is refreshed in
`/mnt/raid0/llm/tmp/exl3-gfx90a-owner-build-20260926`; its committed receipt is
`validation/build-owner-run-20260926.json`. It binds the current launcher and
README. Later source changes require another manifest refresh: source drift is
refused before any GPU launch, even when the kernel sources did not change.

Use `--fixture path/to/canonical/manifest.json` for each real MUL1 K3/K4 and MCG K4
fixture. Use `--microbench` separately for concentrated/spread/random routes,
unified/grouped policy and GEMV/MFMA; each timing row carries its route histogram,
seven raw event intervals, and source/scratch identity. Those records are
protocol-ineligible observations. There is no tokens/sec or model throughput
claim. Native verifier and measurement rows use the EXL3-1 writer and projection.

## Gate accounting

| Gate | Implemented and checked without GPU | Still requires authorized, physically claimed runtime |
|---|---|---|
| G3 | Exact target build/disassembly; host procedural/canonical real fixture references; compiled native packed wave64 kernels | Exhaustive device states/lane maps; real MUL1/MCG GPU↔portable↔CPU output parity; poisoned-tail/device teardown checks |
| G4 | Separate 16×16×16 FP16/FP32 MFMA path; register decode and transform references; emitted MFMA instruction and resource metadata | Full 256-cell device lane map; prefill output parity; measured shape regime, launch/latency/occupancy/HBM observations |
| G5 | 5,120 truth cells; E=256 sentinel separation; plan/schedule bounds, source identity, native arena cap; all fallback tests compiled | Every named GPU path/fallback output and counter; device capacity/alias/capture guard checks; deterministic/atomic routing results and microbenchmarks |

No gate is marked complete from compilation or host checks alone. The delegated
provider remains disabled/null. The explicit owner authority check passes, but
the 2026-09-26 read-only KFD census found existing processes and the shared
co-residency check refused. No physical claim or GPU workload was attempted for
this correction; see `validation/owner-run-20260926.json`.
