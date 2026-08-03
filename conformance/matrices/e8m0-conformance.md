# E8M0 cross-backend conformance matrix

**Instrument:** `CONFORMANCE-VECTORS-1` (ratified 2026-08-03, `epyc-root/MEASUREMENT.md`).
**Vectors:** `conformance/vectors/` · **Consumers:** `conformance/test_e8m0_vectors.py`,
`conformance/test_e8m0_reachability.py` · **Harnesses:** `conformance/harness/`
**Generated:** 2026-08-03.

## The finding

**Two LIVE decode paths give different answers for the same byte.**

| Path | Composition | `0xFF` decodes to |
|---|---|---|
| CPU MXFP4 | `GGML_E8M0_TO_FP32_HALF(e)` — fused half | `0x7f000000` = 2^127, **finite** |
| GPU MXFP4 (HIP) | `ggml_cuda_e8m0_to_fp32(e) * 0.5f` at each call site | `+Inf * 0.5` = **+Inf** |
| OCP MX spec | — | NaN |

They agree on every other code. The disagreement is structural: `+Inf × 0.5` is still `+Inf`, while
the fused half yields a finite value. Both paths are live and heavily called.

## Classification: LATENT DEFECT — and the first justification was wrong

**RETRACTED 2026-08-03.** An earlier version of this matrix called the divergence
*"documented-divergent, not a defect"*, justified by `validate_e_e8m0` rejecting `0xFF` at load.
**That justification is false.** The gate runs only under `check_tensors`, which
**defaults to `false`** (`common/common.h:585`) and is **passed by none of our launchers**. The
loader does not protect us.

**What actually bounds the risk is much weaker: we serve no MXFP4 model** — none in the registry,
none on disk. So the path is never exercised. That holds only until an MXFP4 model is adopted, at
which point CPU and GPU would silently disagree on any `0xFF` scale byte, with no error raised.

This is the distinction worth keeping: *"the loader rejects it"* would be a property of the code.
*"we don't use the format"* is a property of the current fleet, and fleets change.

## Reachability sentinels — the claim is now checked, not asserted

`test_e8m0_reachability.py` fails the moment a precondition stops holding:

| Sentinel | Fails when | Consequence |
|---|---|---|
| `test_no_mxfp4_model_is_served` | an MXFP4 model enters the registry or lands on disk | **the divergence goes live** — the one that matters |
| `test_cpu_nonhalf_decoder_still_has_no_call_sites` | `ggml_e8m0_to_fp32` gains a caller | a **third** live answer appears |
| `test_cpu_live_path_still_uses_the_half_decoder` | the CPU MXFP4 path stops using `_half` | vectors need re-deriving |
| `test_gpu_live_path_still_composes_full_times_half` | GPU call sites stop applying `*0.5f` | the divergence may have closed; re-analyse |
| `test_check_tensors_is_still_off_by_default…` | upstream flips the default to `true` | **good news** — the gate would then protect us |

## Contracts

| Contract | `0xFF` | Status | Checked by |
|---|---|---|---|
| `e8m0_mx_spec` | `0x7fc00000` NaN | **VERIFIED** | `test_contract_matches_reference` |
| `e8m0_ggml_full` | `0x7f800000` +Inf | **VERIFIED** | `test_contract_matches_reference` |
| `e8m0_ggml_half` | `0x7f000000` 2^127 | **VERIFIED** | `test_contract_matches_reference` |

## Backends

| Backend | Expected | Status | Evidence |
|---|---|---|---|
| **CPU `_half`** (MXFP4 live path) | `e8m0_ggml_half` | **VERIFIED** | `harness/e8m0_cpu_harness.c` **executed**; includes the frozen `ggml-impl.h` and calls the real decoder |
| **CPU `ggml_e8m0_to_fp32`** | `e8m0_ggml_full` | **VERIFIED** (dead code) | same harness; zero call sites, so verified but unreached |
| **HIP / ROCm (MI210)** | `e8m0_ggml_full` | **VERIFIED** | `harness/e8m0_hip_harness.hip` **executed on gfx90a**; `cudart_version_defined: false` confirmed *by execution*, not by reading the `#if` |
| CUDA ≥ 12.8 | `e8m0_mx_spec` | `ASSERTED` | we do not build this path |
| Metal / SYCL / Vulkan / OpenCL | `e8m0_ggml_full` | `ASSERTED` | read from source; none is a path we build or serve |

**Three rows moved ASSERTED → VERIFIED** by executing the real decoders. The HIP row matters most:
which branch of `#if CUDART_VERSION >= 12080` is taken was previously *inferred* from source, and is
now *observed* on the card.

## Remediation options (operator decision, not taken here)

1. **Pass `--check-tensors` in production launchers.** Restores the loader gate. Cost: validation
   time at model load, on every model, for a defect that is currently unreachable.
2. **Report upstream.** Two live decoders disagreeing at a reserved byte is an upstream bug, not
   ours; production is FROZEN and we would not patch it locally regardless.
3. **Do nothing, and rely on the sentinels.** Cheapest, and defensible *because* the sentinel fires
   before an MXFP4 model can be served — the risk is bounded by a check rather than by memory.

Recommendation: **3 now, 1 at the moment an MXFP4 model is proposed** — the sentinel makes that
moment impossible to miss, which is what the first version of this document did not.

## Regenerating and re-verifying

```bash
python3 conformance/generate_e8m0_vectors.py                    # derive vectors from stated rules
bash    conformance/harness/run_backend_conformance.sh          # build + execute real decoders
uv run --with pytest python -m pytest conformance/ -q           # 15 tests
```

The generator derives each contract from its **stated rule**, never by transcribing the
implementation it describes — transcription produces vectors that can never fail. The harnesses do
the opposite: they **call the real code**, so they track the frozen tree rather than restating it.
