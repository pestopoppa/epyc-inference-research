# A10 — IQ2_XXS attribution on the real 122B UD-IQ2_M (MI210, 2026-08-12)

Row: `handoffs/active/mi210-q8-dequant-gemv-roofline.md:254`. Classification: **OBSERVATION**
per MEASUREMENT.md. Two captures, both zero source change and zero commit.

## Run 1 — governed, PREFILL ONLY

`run_autokernel_rocprofv1_attribution.py`, receipt `status: passed`, no `rocprofv2`, no seed flags.
Receipt: `/mnt/raid0/llm/autokernel/probes/iq2xxs-rocprofv1-attribution-20260812T1302Z`.
Binary: clean frozen-v9 `0db32c06e` (`build_number` 10125). Residency proven during: VRAM 57→58%
(~37 GB), GPU 99–100%, external sampler. 25,888 dispatches, 5,586 ms kernel time, prefill 732.8 t/s.

| share | count | kernel |
|---|---|---|
| 40.01% | 752 | `mul_mat_q<(ggml_type)16, 64, false>` — **IQ2_XXS**, MMQ batched path |
| 18.85% | 376 | `mul_mat_q<(ggml_type)18, …>` — IQ3_XXS |
| 11.04% | 288 | `gated_delta_net_cuda<128, …>` |
| 8.85% | 576 | `Cijk_…MT64x32x64_MI16x16x16x1` (rocBLAS/Tensile MFMA) |

**Defect found: the governed runner cannot capture decode at all.** `bench_command()` hardcodes
`-n 0` (line 69), so every governed rocprof-v1 attribution to date — including the K28 attribution
this path was validated on — is **prefill-only**. This run yielded 2 `mul_mat_vec_q` dispatches
(warmup). A10's roofline question is a *batch-1 decode* question, so run 1 does not answer it.

## Run 2 — decode, NON-GOVERNED (no receipt, no durable identity)

`llama-bench -p 0 -n 128 -r 1 -ngl 99 -fa on -t 8`, driven under the same rocprof-v1 invocation
copied from `profile_command()`. `/mnt/raid0/llm/autokernel/probes/iq2xxs-decode-nongoverned-20260812T1306Z`.
Residency: VRAM 57%/56% sampled during. **245,246 dispatches, 3,158 ms kernel time, 39.40 t/s decode.**
`-t 8` deliberately (mainA held four CPU regions); with `-ngl 99` host threads are near-irrelevant
to a GPU-resident decode. Recorded deviation.

| share | count | median | kernel |
|---|---|---|---|
| **16.42%** | 6,063 | **85.60 µs** | `mul_mat_vec_q<(ggml_type)16, 1, true, false>` — **IQ2_XXS** |
| 15.40% | 6,063 | 79.68 µs | `mul_mat_vec_q<(ggml_type)18, 1, false, false>` — IQ3_XXS |
| 13.52% | 13,674 | — | `mul_mat_vec_q<(ggml_type)13, …>` — Q5_K |
| 6.32% | 10,965 | — | `mul_mat_vec_q<(ggml_type)14, …>` — Q6_K |
| 6.31% | 45,021 | — | `quantize_q8_1` |
| 5.04% | 21,672 | — | `mul_mat_vec_f<float, float, 1, 128, …>` |

### The correction this run forced

An earlier note today concluded "IQ2_XXS MMVQ is not occupancy-limited" from the synthetic smoke's
`Arch_VGPR=64` → 8 waves/SIMD. **That kernel is not the one production runs.**

| | synthetic smoke | production decode |
|---|---|---|
| instantiation | `<(ggml_type)16, 1, **false**, false>` | `<(ggml_type)16, 1, **true**, false>` (mm_ids/MoE) |
| `Arch_VGPR` | 64 | **80** |
| waves/SIMD | 512/64 = **8** (max) | 512/80 = **6** (75% of max) |
| `Scratch_Per_Workitem` | 0 | 0 |
| `LDS_Per_Workgroup` | 512 B | 512 B |

So IQ2_XXS decode **is** modestly occupancy-limited — 6 of 8 waves — while still showing zero
register spill from the codebook gather. A synthetic op capture is a different kernel from the
production one until the template arguments are compared.

### What the timing distribution says

Per-dispatch duration is extraordinarily tight: median 85,601 ns, min 83,520, max 88,801 — a 6%
spread across 6,063 dispatches at a single grid size (1,048,576). That is the signature of a
throughput-limited kernel running at a hard resource ceiling, not one with variable stalls.

Bandwidth, **estimate with a stated assumption** (not a measurement): at 39.40 t/s with ~10B active
params of a 124.6B-param model at 2.7 bpw, per-token active-weight traffic is ~3.4 GB, giving
~133 GB/s against the MI210's 1,638 GB/s peak — **~8%**, consistent with the ~10.3% figure the row
cites. The assumption is the active-expert byte count; it is not derived from the trace and should
be confirmed before being used to gate anything.

## Bearing on the row's question

The row asks whether MoE-IQ2 at ~10% of bandwidth is fundable or an architectural floor. This does
not settle it, but it removes two candidate explanations and adds one:
- **Not dequant compute.** `vec_dot_iq2_xxs_q8_1` (`vecdotq.cuh:990`) is int8-native via
  `ggml_cuda_dp4a`; there is no float per-element dequant, same as Q8_0 (`a8afd338` L3).
- **Not register spill.** Scratch is 0 in both variants.
- **Partly occupancy**, at 6/8 waves — a ceiling of ~33% more waves if VGPR pressure in the mm_ids
  variant can be reduced. That is the one concrete, testable lever this capture surfaces.

## Open, and deliberately not closed here

A governed *counter-level* differential still has no path: rocprof-v1's `SQ_WAVES`,
`SQ_BUSY_CYCLES`, `SQ_INSTS_VMEM_RD`, `SQ_INSTS_VALU_INT32`, `SQ_INSTS_SALU` all read exactly 0 on
this box, and `rocprofv2` segfaults on IQ2_XXS. Run 2 is non-governed. Two follow-ups worth filing:
extend the governed runner past its hardcoded `-n 0`, and re-run K28's attribution knowing it
measured prefill.
