# E8M0 cross-backend conformance matrix

**Instrument:** `CONFORMANCE-VECTORS-1` (ratified 2026-08-03, `epyc-root/MEASUREMENT.md`).
**Vectors:** `conformance/vectors/e8m0_*.json` · **Consumer:** `conformance/test_e8m0_vectors.py`
**Generated:** 2026-08-03.

## The reading rule, before the table

**A row is `VERIFIED` only if a test in this repo actually consumes the vectors against that
backend.** Everything else is an observation from reading source and is marked `ASSERTED`. This
distinction is the whole instrument: committed vectors nobody runs look exactly like coverage, and
that is the failure mode being prevented — the divergence below survived for months precisely
because nothing compared the backends.

## Contracts

| Contract | 0xFF decodes to | Status | Checked by |
|---|---|---|---|
| `e8m0_mx_spec` — OCP MX v1.0 | `0x7fc00000` NaN | **VERIFIED** | `test_contract_matches_reference` |
| `e8m0_ggml_full` — `ggml_e8m0_to_fp32` | `0x7f800000` +Inf | **VERIFIED** | `test_contract_matches_reference` |
| `e8m0_ggml_half` — `ggml_e8m0_to_fp32_half` | `0x7f000000` 2^127, finite | **VERIFIED** | `test_contract_matches_reference` |

"VERIFIED" here means the **reference decoder** matches its pinned vectors bit-exactly. It does not
mean any compiled backend has been executed — see the next table.

## Backends

| Backend | Expected contract | Status | Why |
|---|---|---|---|
| CPU (MXFP4 live path) | `e8m0_ggml_half` | `ASSERTED` | read from `ggml/src/ggml-impl.h:477`; no harness links the real decoder |
| CPU (`ggml_e8m0_to_fp32`) | `e8m0_ggml_full` | `ASSERTED` | **has zero call sites in the tree** — reachable only if something starts calling it |
| HIP / ROCm (our MI210) | `e8m0_ggml_full` | `ASSERTED` | `ggml-cuda/common.cuh:814-822`; `CUDART_VERSION` undefined on this build, so the fallback is taken |
| CUDA ≥ 12.8 | `e8m0_mx_spec` | `ASSERTED` | uses `__nv_cvt_e8m0_to_bf16raw`; we do not build this path |
| Metal / SYCL / Vulkan / OpenCL | `e8m0_ggml_full` | `ASSERTED` | read from source; none is a path we build or serve |

**No backend row is VERIFIED.** Moving any of them requires a harness that links the real decoder and
feeds it the vectors — follow-on work, deliberately not claimed here.

## Why the divergence is documented-divergent rather than a defect

`validate_e_e8m0` (`ggml/src/ggml-quants.c:5366`) **rejects `0xFF` at load**, wired for MXFP4 and
called from `llama-model-loader.cpp`. A GGUF carrying `0xFF` is refused outright, so the three-way
divergence is **unreachable in practice** — `0xFF` is treated as reserved, which is what the MX spec
intends.

That is exactly the situation the dual-contract design exists for: the ggml behaviour is recorded as
its own contract rather than as a bug against the spec, and a backend cannot satisfy one contract by
breaking another.

## What the consumer checks beyond value equality

Three structural properties, because a vector set can pass every value check while testing nothing:

- **`test_coverage_is_edge_weighted`** — the edges must still be present. They are the only place the
  three contracts disagree; losing them silently would leave a green suite over a vacuous test.
- **`test_the_contracts_actually_disagree`** — asserts three *distinct* answers at `0xFF`. If a future
  edit collapsed the contracts onto one behaviour, every other test would still pass while the
  instrument quietly stopped discriminating anything.
- **`test_contracts_agree_away_from_the_edge`** — the converse. Divergence anywhere other than `0xFF`
  means either a decoder changed or a vector is wrong, and both need a human.

## Known limitation, recorded at adoption

These vectors are hand-written and will drift from the implementations they describe — the same
failure mode they exist to document. The `VERIFIED`/`ASSERTED` column is what makes that drift
visible rather than silent. Nothing here re-reads the frozen tree automatically; a source change to
any decoder will not be caught until someone re-runs the derivation.

## Regenerating

```bash
python3 conformance/generate_e8m0_vectors.py                       # derive vectors
uv run --with pytest python -m pytest conformance/ -q              # consume them
```

The generator derives each contract from its **stated rule**, not by transcribing the implementation
it describes. Transcribing would produce vectors that can never fail.
