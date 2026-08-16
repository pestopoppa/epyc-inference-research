# Handoff: MI210 single-stream roofline — Q8 dequant-GEMV kernel + batch-1 latency levers

**Status**: OPEN — Q4_K inside-kernel attribution and IQ2 governed replay remain live. **Created**: 2026-07-04 (Fable5 window-2 MI210 campaign follow-on).
**Owner tree**: `/mnt/raid0/llm/llama.cpp-experimental`; all work starts from current frozen production v9 in an isolated experimental branch. The old `upstream-mtp-verify`, production-v6, and `build-hip/` instructions below are historical only.
**Substrate**: MI210 gfx90a/CDNA2, 64 GB HBM2e, ROCm 6.2. Current runs follow P-GPU-1, exclusive device claims, numeric in-window sampling, and the governed INF-48 profiler runners; pre-P-GPU-1 observations below retain their historical authority only.
**Context doc**: `handoffs/active/fable5-window2-findings-05b-mi210-inference-architecture.md` §1/§9.

## UPDATE 2026-07-04 — measured results REFRAME this handoff (Tier-1 premise is WRONG; async-prefetch is the lever)

Kernel-thread `a8afd338` executed the levers below. Corrections, most important first:

1. **Tier-1 lever 2 (fused dequant-in-GEMV) is a NON-TASK.** The Q8_0 GEMV is ALREADY int8-native — `vec_dot_q8_0_q8_1` (`vecdotq.cuh:797`) uses `ggml_cuda_dp4a` integer MAC + one fp scale per 32-block, i.e. the CPU iqk "dequant-under-load" pattern is already implemented on GPU. There is **no per-element fp dequant to hide.** The **47→62% gap is achieved-bandwidth / occupancy, NOT dequant-compute.** Do not port iqk to "hide dequant" — there is nothing to hide.
2. **The real lever is Tier-2 lever 3 (async weight-prefetch / memory-level-parallelism).** Down-payment already LANDED: **nwarps 2→4 for batch-1 Q8_0 on CDNA2 = +4.6% (28.99→30.32 t/s)**, `test-backend-ops MUL_MAT` 1103/1103, committed **`5dc116130`** (fork `upstream-mtp-verify`). Async prefetch (`raw.buffer.load.lds` LDS double-buffer) is now in build+measure — exact gfx90a intrinsic design in [mi210-batch1-latency-wall-greenfield.md](../completed/mi210-batch1-latency-wall-greenfield.md).
3. **Tier-1 lever 1 (`quantize_q8_1` requant) = DEFER.** Measured **3.37%** of decode (this build; 5.68% on the older mi210-hip build). Every GEMV re-quantizes its own activation (call-count == `mul_mat_vec_q` count); localized prologue-fusion is *counterproductive* (would requant the activation `gridDim.x`≈thousands of times). Only fix is graph-level (q/k/v activation caching, or fuse into the preceding RMSNorm), ceiling ~1.5–3% with correctness risk on the 82%-of-decode hot path → not worth it now.
4. **Tier-3 lever 6 (n-gram / prompt-lookup GPU spec) = MEASURED NEGATIVE.** All variants regress on the 27B: plain 28.4 → ngram-simple 27.7 (best) → ngram-cache 26.1 (worst); context-only acceptance **~15%**, below break-even. A zero-cost n-gram does NOT beat plain single-stream here — a real trained drafter (draft-mtp/eagle3) remains the path. (Corpus-static ngram, CPL-4b, is separate but must clear a HIGH acceptance bar to beat this negative.)
5. **rocprof note**: v1 aborts at init on this build (PDL/graph kernels) — use **rocprofv2** with the `pmc:` prefix, and add `/usr/lib/x86_64-linux-gnu` to `LD_LIBRARY_PATH` (`libpciaccess.so.0`).

Net: the Tier-1 dequant framing is superseded. The live single-stream track is **nwarps (done +4.6%) → async-prefetch → SoA-repack**, toward the fp16 62% ceiling. Levers below are kept for the record; read them through this update.

## L3-MoE / L15 quantized-MMQ bet — scope said GO, **BUILD FALSIFIED IT: L3-MoE occupancy rewrite = NO-GO** (2026-07-04)

**Phase-1 result (measure/build-first worked as intended):** the compact-LDS rewrite was BUILT + gated + correct — LDS halved 49→25 KB, LDS-limited residency 1→2 WG/CU, accumulator untouched; `MUL_MAT 1103/1103`, `MUL_MAT_ID 789/789`, PPL indistinguishable — but occupancy stayed **FLAT (3.07→3.07)** and aggregate moved **+1.6% B=32 / −12% B=64**. **The Phase-0 LDS-limit premise is FALSIFIED with data:** at B=32 Q8-MMQ is **grid-limited** (grid = 53248/512 = exactly 104 WGs = 1 WG/CU → no second workgroup to place in the freed LDS), NOT LDS-limited. bf16's aggregate win is its **native-MFMA / zero-dequant compute**, not an occupancy edge Q8 could match by freeing LDS. Raising Q8-MMQ aggregate needs MORE workgroups → **stream-K K-splitting** (a bigger separate bet w/ fixup-kernel overhead; the gated compact-LDS kernel is the ready substrate, saved `campaign/mmq-compact-lds-NEGATIVE.patch`) — or smaller `mmq_y` (forbidden: halves the accumulator; campaign already measured mmid-forcing negative). **Settled: bf16-for-aggregate (B≥16–24), Q8 only for HBM-capacity.** **L15 (sub-4-bit capacity unlock) is INDEPENDENT of this occupancy result and still viable.** The optimistic scope below is kept for the record — read it through this result.

The aggregate-MoE #1 kernel lever, scoped + de-risked into a fundable GO with a concrete mechanism + a measured target.
- **Mechanism (de-risked):** Q8 `mul_mat_q`'s occupancy limiter is the **dequant-staging LDS (45–49 KB > half of 64 KB) + Arch-VGPR (128)**, which caps residency at exactly **1 WG/CU** (grid = 104 WGs) → ~2.5 waves resident → memory unit under-driven at **32%** vs bf16's 48% (the 1.5× maps onto the measured +32% gap). **NOT the MFMA accumulator tile** — bf16 wins with the SAME ~128 Acc-VGPR, just smaller LDS (12.8–25.6 KB) + tiny Arch-VGPR (16). So the "shrink the MFMA tile / lose efficiency" risk is OFF the table; the lever is halving the **staging** footprint only.
- **Target** = bf16's own measured profile: LDS 49→~24 KB, Arch-VGPR 128→~64 → **2 WG/CU** → MemUnitBusy 32→~48% → occ 2.55→~3.2. **Payoff: Q8 aggregate B=32 from 563 t/s toward bf16's 744 — Q8 aggregate-competitive at HALF the HBM.** Residual: at B=128 bf16 also wins on native-MFMA FLOPs, so occupancy alone won't fully erase the top-end; "competitive at half HBM" is the residency-bet goal + achievable. Risk = more dequant recompute / less N-reuse if staging cut too far.
- **L15 sub-4-bit — path ALREADY EXISTS + numerically correct on CDNA2** (not MMVQ-only): IQ2_XXS/XS/S, IQ3, IQ1, Q2_K, Q3_K all MMQ-supported (`mmq.cu:267`) with MFMA tile mappings; CDNA2 gate routes them to MMQ when `n_experts>64 OR ne11≤128`. Live-confirmed (HEAD 7c28056b7): `MUL_MAT 1103/1103` + **`MUL_MAT_ID 789/789, 0 FAIL`** across q8_0/q2_K/iq2/iq3/iq1/iq4. **Missing = a quantized GGUF + the SAME occupancy rewrite** (IQ codebook dequant is MORE LDS-hungry → needs it more than Q8). **Capacity: Qwen3.5-122B-A10B @ IQ2 ≈ 38 GB FITS → fully GPU-resident (the big residency-bet unlock); GLM-5.2 ~238 GB does NOT fit** even at IQ2. **STATUS 2026-07-04: Qwen3.5-122B-A10B UD-IQ2_M (40.4 GB, unsloth) is DOWNLOADING** (in flight ~11%+; obviates the "quantize an IQ2 proxy" step in the build sequence) → on completion, measure PPL + aggregate vs the existing UD-Q4_K_M benchmarks. L15 measurement PENDING.
- **Shared kernel family (one investment, two payoffs):** `mul_mat_q_case<TYPE>` → single `mul_mat_q` device kernel; the LDS-staging/accumulator/occupancy body is SHARED, only `load_tiles` (per-type dequant) differs. **L3-MoE (the shared-body rewrite) is the prerequisite for L15 being fast.**
- **Build sequence:** Phase 0 (sub-4-bit MMQ correctness ✅; remaining = quantize an IQ2 proxy + operator-gated IQ2-vs-Q8 bench) → **Phase 1 (shared LDS-staging/occupancy rewrite, IN PROGRESS)** → Phase 2 (apply to IQ2/IQ3 + retune codebook dequant + quantize 122B→IQ2).
- **Key files:** `ggml/src/ggml-cuda/mmq.cu` (267–340), `mmq.cuh` (MMA tile map 239–264, LDS `MMQ_MMA_TILE_X_K_*` 219–225). Prior CSVs: `campaign/moe-agg/prof/pmc_{q8,bf16}/`.

## UPDATE 2026-08-03 — the gap is ONE handoff but TWO mechanisms; splitting them (research-intake Stage-2b)

_Via `/research-intake` Stage-2b dive 2b-C. This **refines** the 2026-07-04 update above, it does not
contradict it — the recomputed ladder reproduces the existing fable5 series to within 1 pp._

**The full quant ladder, recomputed on one consistent GiB basis** (spec-BW denominator; see the calibration
caveat below), with `mul_mat_vec_q` measured at **77.8% of decode time**:

| Rung | Bandwidth attainment |
|---|---|
| fp16 (llama.cpp) | **62.6%** |
| fp16 (vLLM-ROCm, same device) | **69.2%** |
| Q8_0 | 50.2% |
| Q4_K | 35.1% |
| MoE-bf16 | 34.4% |
| MoE-Q8 (frontdoor) | 21.3% |
| **MoE-IQ2 (architect, `Qwen3.5-122B-A10B UD-IQ2_M`)** | **10.3%** |
| Qwen3-Next-80B i1-IQ2 | 3.3% |

**The reading that matters: 62–69% is already reached on this device, so the memory system is not the
limiter.** The collapse is entirely down the quant ladder. For reference, DGX Spark GB10 reaches **77–80%
at Q4_K_M dense across five models** on the same engine — NVIDIA's quant sag is 5–10 pp; ours is 27 pp.

**Why this handoff now carries two items instead of one.** The 2026-07-04 update's finding —
*"the 47→62% gap is achieved-bandwidth / occupancy, NOT dequant-compute. Do not port iqk to hide dequant"* —
is **correct, and it is correct for Q8_0**, whose GEMV is already int8-native. But **Q4_K 35→50 is a
different mechanism** (k-quant superblock unpack, not occupancy), and treating the two as one gap has
already produced one wrong verdict. They are filed separately in the checklist below.

**Corrected IQ2_XXS characterization** (verified read-only against our tree, supersedes the
"four *dependent* random reads plus *a* sign-table read" framing): per 32 quantized weights there are
**four INDEPENDENT data-dependent gathers** into a 256-entry / 2 KiB grid — all four indices arrive in a
**single 8-byte `memcpy`** (`ggml-quants.c:2500`), so no address depends on another lookup's result, ILP ≈ 4,
**not a pointer chase** — plus **FOUR** sign-table gathers (`ksigns_iq2xs[…]` is inside the `l` loop), each
grid entry expanding to 8 values. The 4+4 ratio holds across every SIMD path including our production iqk
(`iqk_gemm_iquants.cpp:176-181`, signs `:86-90`), AVX2, and CUDA.

**Calibration — MEASURED 2026-08-03, and the correction is smaller than estimated.**
`epyc-inference-research` `328b768d`, receipt
`data/mi210-achievable-bandwidth/20260803T124401Z/receipt.json`
(SHA-256 `0aab9c7e135929e72fd3a5c2498eb807dc16d0f80b773f063e1df3524df7b4d3`).

**Achievable = 1433.3 GB/s, i.e. 87.5% of the 1638 GB/s datasheet peak** — high for HBM2e. Triad 1371.1;
per-kernel medians copy 1412.5 / mul 1413.9 / add 1362.3 / triad 1362.1, p20–p80 within ~1.2%; 10 warmup,
50 timed, correctness PASS (max abs err 0.0). **Correction factor 1.143×, not the 1.17–1.26× estimated
here before the run — the prior ~1.3–1.4 TB/s guess was low.**

| Rung | % of spec (1638) | % of **achievable** (1433.3) |
|---|---|---|
| fp16 | 62.6 | **71.5** |
| fp16, vLLM-ROCm | 69.2 | **79.1** |
| Q8_0 | 50.2 | 57.4 |
| Q4_K | 35.1 | 40.1 |
| MoE-bf16 | 34.4 | 39.3 |
| MoE-Q8 (frontdoor) | 21.3 | 24.3 |
| **MoE-IQ2 (architect)** | **10.3** | **11.8** |
| Qwen3-Next-80B i1-IQ2 | 3.3 | 3.8 |

**⚠ THE BASIS WARNING, and it is the most important line on this page.** Converting *our* numbers to an
achievable basis while leaving *someone else's* on a spec basis makes the gap look smaller without it
being smaller. **The AMD-vs-NVIDIA comparison must stay spec-to-spec** — our fp16 62.6 against DGX Spark
GB10's 77–80, both against datasheet — **until somebody measures GB10's achievable bandwidth.** Our
71.5%-of-achievable set beside their 77–80%-of-spec is a mixed-basis comparison and is not a reading.
This measurement makes our own numbers physically meaningful; **it does not narrow the NVIDIA gap at all.**
That is precisely the error found this session in AMD's own KB, where a per-OAM TFLOPS figure was divided
by a per-GCD bandwidth to give a ridge point off by 2×.

Use the **achievable** column for headroom and campaign sizing (it is the real roof); use the **spec**
column for any cross-vendor comparison, and say which one you used — a utilisation quoted without its
denominator is not a number.

**The E8M0 divergence is instrumented, and its first classification was WRONG** ✅ 2026-08-03.
`CONFORMANCE-VECTORS-1` ships at `epyc-inference-research/conformance/` (research `33f7076b`): three
contracts, harnesses that **execute the real CPU and HIP decoders** (27 values, gfx90a, all matching),
and 15 tests. Three backend rows are now **VERIFIED** rather than asserted — including the HIP row,
where which branch of `#if CUDART_VERSION >= 12080` is taken was previously *inferred* from source and
is now *observed* on the card.

**RETRACTED: "documented-divergent, not a defect."** That rested on `validate_e_e8m0` rejecting
`0xFF` at load — but that gate runs only under `check_tensors`, which **defaults to false** and is
passed by none of our launchers. And it is not dead code: **both paths are live.** CPU MXFP4 uses
fused `_half`; GPU MXFP4 uses `full(e) * 0.5f`. They agree everywhere except `0xFF`, where
`+Inf × 0.5` is still `+Inf` while the fused half gives a finite 2^127.

**What actually bounds it is weaker: we serve no MXFP4 model** — a property of the fleet, not the
code. So the claim is now *checked*: a sentinel fails the moment an MXFP4 model enters the registry
or lands on disk, carrying the remediation options. Recommendation is to rely on the sentinel now and
pass `--check-tensors` the moment an MXFP4 model is proposed.

**Free incidental finding in our own CPU tree, filed here because it is the same IQ2 sign path:**
`iqk_gemm_iquants.cpp` contains an AVX512-VPOPCNTDQ routine deriving IQ2 signs arithmetically with **zero
table reads**, but its dispatch at `:184` is guarded by `#if defined z_HAVE_FANCY_SIMD && …`. **The `z_`
prefix makes the macro undefinable, so the branch is permanently dead** (same at `:109`, `:571`) and the
table lookup always runs — **including on our Zen 5 host, which has VPOPCNTDQ.** Inherited verbatim from
ik_llama.cpp via `fec061dea`; not introduced by us. Experimental branch only — production is frozen.

## Objective

Raise **single-stream** GPU decode throughput for the qwen35/Q8 family toward the bandwidth roofline. Measured today: Q8 decode ~47% roofline (766 GB/s), fp16 ~62%, and rocprof attributes single-stream decode ~78% to `mul_mat_vec_q` (the Q8 weight GEMV) + 5.68% to `quantize_q8_1` (activation requant). Two gaps: **47→62% = Q8 dequant cost** (kernel-addressable); **62→100% = batch-1 latency wall** (memory-level-parallelism, harder). This handoff attacks both.

## Why this matters / transfer

- The already-landed MMVQ→MMQ dispatch fix (`de447119f`, `mmvq.cu:323` CDNA2 `case GGML_TYPE_Q8_0: return ne11<=1;`) gave +17% (27B) / +31.7% (gemma-31B) on the **MTP-verify** batch. This handoff is the **plain single-token** decode, the dominant path.
- Transfers to every GPU-resident Q8 role (the deferred residency bet): frontdoor, and any dense-Q8 model.
- The dequant kernel also unlocks **sub-4-bit** viability (bytes/token axis, below).

## Levers, ranked (each ships with its decisive experiment)

### Tier 1 — the dequant gap (47→62%, kernel-addressable, highest confidence)
1. **Kill the `quantize_q8_1` activation-requant overhead** (measured 5.68% of decode). The GEMV requants activations to Q8_1 every step; investigate fusing it into the GEMV prologue or caching. *Experiment*: rocprof before/after; acceptance = quantize_q8_1 share → ~0, decode +5%.
2. **Fused dequant-in-GEMV / int8-native MMQ path for batch-1.** Current `mul_mat_vec_q` dequantizes Q8 blocks then does fp accumulate. Port the CPU **iqk** approach (weight-block-outer, dequant-under-load; ref `ggml/src/ggml-cpu/iqk/iqk_gemm_legacy_quants.cpp:302-330`) to HIP, or hide dequant under the HBM weight load. *Experiment*: single-stream `llama-bench -n 128` Q8 vs the fp16 62% ceiling; acceptance = Q8 decode → ~55-62% roofline (~34-38 t/s on 27B, +15-30%), PPL/output unchanged.

### Tier 2 — the batch-1 latency wall (62→~80%, harder, MLP-bound)
3. **Async weight prefetch / double-buffering** in the GEMV — software-pipeline the next weight tile's `buffer_load` under current compute (Little's law: more requests in flight → higher achieved BW). *Experiment*: rocprof MemUnitStalled + achieved-BW before/after; acceptance = achieved BW 62%→~70%.
4. **Persistent/megakernel decode** (CUDA-proven, NO ROCm version — greenfield, high-effort). Fuse decode into one launch, weights streamed with pipelining; the Hazy/Mirage ~78% single-dispatch result. *Scope first, don't build blind* — estimate launch-overhead fraction via rocprof gap-analysis before committing.
5. **Weight swizzle/layout for gfx90a** — verify MMVQ HBM access is coalesced to the 64-wide wavefront + 128B cache line. *Experiment*: rocprof L2/MemUnit efficiency; a layout change is only worth it if coalescing is measurably poor.

### Tier 3 — sidestep / bytes-per-token (orthogonal, can exceed BW-% gains)
6. **n-gram / prompt-lookup speculation on GPU** (`--spec-type ngram-simple|ngram-cache|ngram-map-k`, already in v6, UNTESTED on GPU). Zero-cost draft → converts batch-1 into batch-N verify → more MLP + amortized weight read, at high acceptance for code/JSON/repetitive output. *Experiment (cheap, no kernel)*: run `--spec-type ngram-*` on the 27B over a code/structured prompt set, measure decode t/s + acceptance vs plain. **Do this FIRST — it's the cheapest single-stream win and needs no kernel.**
7. **Sub-4-bit weights (IQ2/TQ1/TQ2) + efficient CDNA2 dequant** — fewer bytes/token → higher absolute t/s even at same roofline-%. Gated on Tier-1 dequant kernel + an eval-parity/PPL quality check. *Experiment*: IQ2/TQ quant of the 27B, decode t/s + PPL vs Q8.
8. **KV-quant for single-stream LONG context** (q8-KV, `-fa 1`) — distinct from the aggregate case (where it was dead): at batch-1 long context the KV read adds to bytes/token. *Experiment*: 27B decode at 64k ctx, q8-KV vs f16-KV.

## Start-here sequence for the executor
1. **Lever 6 (ngram-spec) FIRST** — cheapest, no kernel, may bank a single-stream win in an hour.
2. **Lever 1 (quantize_q8_1)** — small, high-confidence kernel win.
3. **Lever 2 (fused dequant-GEMV)** — the main event; port the iqk pattern to HIP.
4. Then Tier-2 (prefetch/megakernel) only if Tier-1 leaves the 62% gap and it's worth the effort.

## Correctness / measurement discipline
- Every kernel change: `test-backend-ops` parity + PPL/greedy-output unchanged (the MMVQ fix was numerically-valid-not-bit-exact; document the same for any new kernel).
- Pair every speed number with a correctness check. Label OBSERVATION until a P-GPU-1 protocol exists.
- Do NOT commit to production-consolidated-v6; experimental only, operator-gated for promotion.

## Key files
`ggml/src/ggml-cuda/mmvq.cu` (dispatch + `mul_mat_vec_q`), `ggml/src/ggml-cuda/quantize.cu` (`quantize_q8_1`), `ggml/src/ggml-cuda/ggml-cuda.cu:2554-2617` (mul_mat dispatch), CPU reference for fused approach: `ggml/src/ggml-cpu/iqk/iqk_gemm_legacy_quants.cpp:302-330`. Profiling artifacts: `/mnt/raid0/llm/tmp/mi210-build/campaign/{prof,finish}/`.

## Progress checklist

- [x] nwarps 2->4 (+4.6%, committed 5dc116130) ✅
- [x] async weight-prefetch raw.buffer.load.lds (+3.3%, committed 7c28056b7) ✅
- [x] L3-MoE compact-LDS occupancy rewrite (BUILT + FALSIFIED, NO-GO) ✅
- [x] L15 sub-4-bit: quantize 122B->IQ2 proxy + operator-gated IQ2-vs-Q8 bench ✅ **2026-07-29 — the "PENDING" predates the run.**
  Both halves landed 2026-07-05: 122B UD-IQ2_M GPU-resident at 43.7 t/s single / 148.7 agg @B32, PPL 5.02
  (`progress/2026-07/2026-07-05-mi210-capability-kernel-rnd.md:23-24`), and the IQ2-vs-Q4 paired eval returned Δ0.0pp,
  McNemar p=1.000 (`progress/2026-07/2026-07-05-mi210-residency-and-cot-reframe.md:12`). Certified by read-certification (`auditor`).
- [ ] SoA-repack lever (only if coalescing measured poor - currently deemed healthy)
- [x] Generic stream-K K-splitting for Q8-MMQ aggregate is already present and therefore not a new
  lever ✅ 2026-08-11 — the current CDNA MMQ path is persistent stream-K, and the governed WGM
  experiment exercised that exact launch. Retain only a distinctly justified higher-persistent-grid
  residual (for example `2*nsm`) behind fresh grid/occupancy evidence; do not reopen “add stream-K.”
- [x] Recompute the full quant ladder on one consistent basis (fp16 62.6 / Q8_0 50.2 / Q4_K 35.1 / MoE-Q8 21.3 / MoE-IQ2 10.3) ✅ 2026-08-03 — via /research-intake Stage-2b; reproduces the fable5 series to within 1 pp
- [x] Correct the IQ2_XXS access characterization: four INDEPENDENT gathers (single 8-byte memcpy, ILP≈4) plus FOUR sign-table gathers, not a dependent pointer chase with one sign read ✅ 2026-08-03
- [ ] **Item A — Q8_0 50→62 rung (achieved-bandwidth / occupancy).** There is no per-element fp dequant to hide; do NOT reopen an iqk port for this rung. The governed frozen-v9 replay now supersedes async-prefetch as a presumed win: its 20-block median was only **+0.936%**, below the 2% floor, with one negative block (`NOT_REPRODUCED`; receipt SHA-256 `7b173cafcccb8a99319bf93a80fd13a2e94a400afab2bf03355363f9521ab17f`). Reopen only from fresh cache-line/MLP profiling evidence, not the 2026-07-04 patch prior.
- [x] **Item B — attribute the Q4_K superblock-unpack mechanism before authoring.** ✅ 2026-08-11 —
  the representative-shape mechanism-counter route below passed. It supports a materially larger
  instruction burden than the same-bit Q4_0 control, but not the former +38–43% single-lever premise.
  - **2026-08-11 bounded op probe:** the hash-bound single-process C4 reports at
    `/mnt/raid0/llm/autokernel/probes/c4-{q4k,q8}-op-singleproc-20260811T12{10,15}Z/report.json`
    hold `m=16,n=1,k=256` and the dispatch sequence fixed. Q4_K versus Q8_0 `mul_mat_vec_q`
    share is 41.95% versus 40.92%, and per-dispatch duration is ~5.72 versus ~5.50 µs. That small
    synthetic surface does **not** explain the 35→50 roofline rung and cannot see unpack work inside
    the fused kernel. Item B therefore remains open for a representative-shape counter/source-timer
    probe; the current result prevents mistaking a complete kernel-family table for the requested
    inside-kernel attribution.
  - [x] **Run the representative-shape single-pass PMC differential and resolve what is identifiable.**
    ✅ 2026-08-11 — clean frozen-v9 `test-backend-ops` ran four balanced blocks × five exact
    dispatches at model-derived `m=17408,n=1,k=5120`. A single rocprofv2 pass collected
    `SQ_WAVES`, `SQ_INSTS_VALU`, and `SQ_INSTS_VALU_INT32`; every counter was exactly invariant
    within and across blocks. Q4_K and the same-bit Q4_0 control both launched 34,816 waves, but
    Q4_K carried **+112.5 VALU instructions/wave**, **+35 INT32 instructions/wave**, and a
    **+11.751%** median dispatch duration (77,600 vs 69,440 ns). Q8_0 launched twice the waves, so
    it remains the quant-ladder control rather than the closest unpack control. The fused dispatch
    makes an exact inside-kernel wall share unidentifiable; the admissible result is differential
    mechanism evidence, not a fabricated share. Receipt:
    `/mnt/raid0/llm/autokernel/probes/inf37-q4k-unpack-v9-20260811-r7/receipt.json`, SHA-256
    `1e34339c1c986413c4eeb1b56ba3202c8763d08df45aba1c0580917c888f5e47`; research
    `70374c43` (promoted via `ac88d75a`). Four rocprofv2 exit-139 attempts were retained and only
    transport-retried under the predeclared two-attempt ceiling; no parsed counter failure was
    retried. **The former +38–43% single-lever expectation is not supported by this matched
    diagnostic.** Before authoring, use the exact instruction delta to define a surgical unpack
    change and test whether it can recover part of the observed ~10.5% Q4_K-to-Q4_0 headroom.
  - [x] **Author and test one surgical Q4_K unpack hypothesis.** ✅ 2026-08-11 — the candidate removed
    the two lane-local Q8 partial-sum `dp4a` operations per `QR4_K` iteration and consumed the
    already-stored `block_q8_1.ds.y` sum. It failed **5/5** representative Q4_K correctness cases
    (relative errors 0.729–0.977 versus the 0.0005 limit), while frozen v9 passed 5/5 under a separate
    released MI210 claim. The reason is structural: `ds.y` covers all 32 block elements, whereas each
    MMVQ lane needs a distinct 8-element slice selected by `iqs`. Receipt SHA-256
    `c8c055ff43f022ae4c61e3142b0278c15a807476db03aa29d16a50b6dbb25eea`; the one-file diagnostic
    remains uncommitted and has no performance or promotion authority.
  - [x] **Split the remaining +35 INT32/wave burden before a second Q4_K source candidate.** ✅
    2026-08-11 — exact gfx90a disassembly of the measured
    `mul_mat_vec_q<Q4_K,1,false,false>` specialization contains four extra
    `v_dot4c_i32_i8` sites for lane-local Q8 sums. At `k=5120`, Q4_K has 20 superblocks and each
    lane executes the vecdot five times, attributing **20/35 INT32 instructions/wave (57.1%)** to
    the required subgroup sums. The residual **15/35 (42.9%)** covers six-bit scale/min unpack plus
    Q4_K-specific packed-nibble address/control; it is not honestly pure unpack yet. Static receipt:
    `/mnt/raid0/llm/autokernel/probes/inf37-q4k-isa-attribution-20260811/receipt.json`, SHA-256
    `01458d64fcd9d2fab0bdb883a619f0904ab3aa3c28d23f3ef8a3fc881517860c`.
  - [x] **Test a correctness-preserving branchless six-bit scale/min decoder.** ✅ 2026-08-11 — the
    one-file experimental candidate leaves both lane-local Q8 subgroup sums untouched and replaces
    only the divergent `j < 2` scale/min extraction. The rebuilt gfx90a backend passed all five exact
    representative `m=17408,n=1,k=5120` Q4_K correctness repetitions. Static ISA size remained 1,452
    bytes while the specialization lost three `s_cbranch_execz` and two `s_branch` sites. A balanced
    two-control/two-candidate diagnostic then found **69,840 vs 78,080.5 ns median (-10.554%)**, despite
    the candidate executing **236.5 vs 216.5 VALU/wave (+9.238%)** and **87 vs 78 INT32/wave
    (+11.538%)**. This is directional evidence for reduced exec-mask/control-flow cost, not an
    instruction-count win. The source is uncommitted, so the result is explicitly diagnostic-only:
    `/mnt/raid0/llm/autokernel/probes/inf37-q4k-branchless-scales-20260811/diagnostic-paired-r3/receipt.json`,
    SHA-256 `de4241bd26b77f5dac7df746d165034b67e6f8105133daf0359142a97dd35d5d`.
  - [ ] **Commit only after explicit experimental-tree approval, then clean-replay the branchless
    decoder through the governed paired runner.** The clean candidate must reproduce correctness and
    timing before any promotion or model-level claim; the dirty diagnostic cannot satisfy this gate.
- [ ] **Item C — architect MoE-IQ2 at 10.3%**, our worst rung by 2× and a production-serving model. Attach the kill-criterion first (below) — this is a probe, not yet a funded kernel
- [ ] Kill-criterion probe for item C: on gfx906 an optimised community fork **and** vLLM independently converge on ~10% bandwidth for MoE batch-1 — the same rung as ours. Two stacks hitting one wall means this may be an **architectural floor**; establish cheaply whether it is before funding a kernel
  - **2026-08-11 tool-boundary result:** the exact main tensor type in the 122B UD-IQ2_M file is
    IQ2_XXS (94 tensors; alongside IQ3_XXS/Q5_K/Q6_K). Ten unprofiled seeded warm-up repetitions pass,
    but `rocprofv2` exits 139 during the IQ2_XXS active trace even after moving all repetitions into
    one backend process. Durable failure receipt:
    `/mnt/raid0/llm/autokernel/probes/c4-iq2xxs-op-singleproc-20260811T1220Z/receipt.json`,
    SHA-256 `fdf355ebc933f3cf20def077cdcc7b998c0072295c00c97b977ad32358c284e2`.
    This is not evidence of an architectural floor. Keep the kill-criterion open and switch the next
    probe to a non-`rocprofv2` device timer/counter path rather than retrying the same crash.
  - [x] **IQ2 tool-boundary follow-up:** capture the same seeded IQ2_XXS shape through a
    non-`rocprofv2` device-timer/counter path and retain the failed receipt as the negative control.
    ✅ 2026-08-12 — done by a different route than the row assumed, so read the caveats. The
    non-`rocprofv2` path is `rocprof` v1 device timestamps, which needs **no seed flags**, so OP-11
    was never on this critical path (OP-11 subsequently DECLINED, Option B). Governed run on the real
    122B UD-IQ2_M passed: `/mnt/raid0/llm/autokernel/probes/iq2xxs-rocprofv1-attribution-20260812T1302Z`,
    residency proven during (VRAM 57→58%, GPU 99–100%). Full analysis:
    [`artifacts/gpu-aux-baselines/a10_iq2_decode_attribution_20260812.md`](../../artifacts/gpu-aux-baselines/a10_iq2_decode_attribution_20260812.md).
    **Three caveats that matter more than the capture:** (1) the governed runner hardcodes `-n 0`, so
    that run is **prefill-only** — the decode GEMV numbers come from a **non-governed** companion run
    (`iq2xxs-decode-nongoverned-20260812T1306Z`); fix filed as RVP-C2-11 with a verified patch.
    (2) IQ2_XXS decode is **16.42%** of decode kernel time, median 85.60 µs across 6,063 dispatches.
    (3) The earlier "not occupancy-limited" reading was from the **synthetic** smoke's
    `<…,false,false>` variant at `Arch_VGPR=64`; production MoE decode runs `<…,true,false>` at
    `Arch_VGPR=80` → **6 waves/SIMD, 75% of max**. Scratch 0 in both, so no spill.
  - [x] **IQ2 VGPR-pressure lever — ISA read done, zero GPU time** ✅ 2026-08-12. Read statically from
    the shipped `libggml-hip.so.0.16.0` (clean frozen-v9 `0db32c06e`). Full table:
    [`artifacts/gpu-aux-baselines/a10_iq2_vgpr_lever_20260812.md`](../../artifacts/gpu-aux-baselines/a10_iq2_vgpr_lever_20260812.md).
    True counts are **63** (`<…,false,false>`, 8 waves) and **78** (`<…,true,false>`, 6 waves) — `rocprof`
    had reported 64/80, which is the allocation-granularity rounding, not the true count. **The mm_ids
    wrapper is NOT the cost:** Q8_0 runs the identical wrapper in **25** VGPRs, so the 78 is IQ2_XXS's own
    dequant state. **But "codebook ⇒ expensive" is also wrong** — IQ4_NL is 38 and IQ1_S 42, while block
    quant Q3_K is the table's worst at 88; pressure tracks per-block state-machine complexity, not the
    lookup. **IQ4_XS sits at exactly 64 on this same wrapper**, so the 8-wave target is proven reachable
    and IQ2_XXS is 14 registers over. Constraint: spill is 0 everywhere, so there is no slack — any
    reduction attempt must report `vgpr_spill_count` beside occupancy or it is not evidence.
  - [x] **Disassembly diff `<IQ2_XXS,1,true,false>` vs `<IQ4_XS,1,true,false>`** ✅ 2026-08-12, zero GPU.
    1002 instructions vs 565; `v_*` 758 vs 334 while scalar (210/203) and LDS (16/16) are near-identical
    and IQ2_XXS issues **fewer** global loads (3 vs 5) — **the codebook gather is not the cost, for the
    third independent time**. The excess is sign unpacking: `unpack_ksigns`/`__vcmpne4`/`__vsub4` lowers
    to ~272 sub-word ops (`v_or_b32_sdwa` 96, `v_lshlrev_b16` 96, `v_sub_i16` 64, `v_bfe_i32` 46), while
    IQ4_XS does its byte manipulation with **48 `v_perm_b32`** that IQ2_XXS never uses.
  - [ ] **Re-express the IQ2_XXS sign expansion using `v_perm_b32`** (byte permute) instead of 16-bit
    shift/or/sub chains, in `vecdotq.cuh`. IQ4_XS is a working in-tree example of the idiom on the
    identical wrapper landing at exactly 64 VGPR = 8 waves. **EXPERIMENTAL BRANCH ONLY** — production
    `production-consolidated-v9` is frozen. Report `vgpr_spill_count` beside occupancy, and note that
    decode is bandwidth-bound so this is necessary-not-sufficient for throughput.
  - [ ] **Re-run the decode attribution under governance** once RVP-C2-11's `--gen-tokens` patch
    lands, so the 16.42% decode figure carries a durable receipt instead of a non-governed probe.
    The Omniperf 2.0.1 / `rocprof` v1 fallback is now durable in research, but its first governed run
    correctly failed compatibility before profiling: clean frozen-v9 `test-backend-ops` at
    `0db32c06` does not implement `--suite-seed` or `--repeat-suite`. The failed receipt retains the
    exact command, clean source/binary/profiler identities, one device sample, and claim
    acquisition/release:
    `/mnt/raid0/llm/autokernel/probes/inf37-iq2xxs-omniperf-v1-20260811/receipt.json`, SHA-256
    `2054a31b5f9104bcd3437b250833de6086a7dded8533e3cc9182bc9a79222510`. A prior manual smoke at
    `/mnt/raid0/llm/autokernel/probes/omniperf-iq2xxs-v1-smoke-20260811T1238Z` produced 260 dispatch
    rows and proves rocprof-v1 reachability, but it has no governed receipt and is non-evidence.
    Keep this parent open until OP-11 permits a durable seeded producer and the runner records a
    passing matched capture.
  - [x] **Build a fail-closed Omniperf-v1 fallback runner for the IQ2 profiler boundary.** ✅ 2026-08-11 —
    it binds clean exact source/tool/Python identities, requires seeded repeated correctness before
    SQ/TCC collection, holds the MI210 claim, samples device state, and writes failure receipts.
- [x] Investigate the permanently-dead `z_HAVE_FANCY_SIMD` AVX512-VPOPCNTDQ IQ2 sign path on an EXPERIMENTAL branch only (production kernel is frozen; this is a CPU-side finding filed here for mechanism adjacency) ✅ 2026-08-11
  - Globally reviving the historical branch was correctly rejected: a governed ten-block A/B at the
    exact IQ2_XXS `m=4096,k=14336` shapes improved `n=1` by **+5.753%** median but regressed `n=512`
    by **-9.511%** median. The kill switch therefore encoded a real prompt-processing tradeoff, not a
    typo. Global-candidate receipt:
    `/mnt/raid0/llm/autokernel/probes/inf37-iq2-fancy-simd-ab-v9-20260811-r4/receipt.json`, SHA-256
    `242cb61b122b39324316d020d1a2a4bc4be4c17ec3008a66f5ecaf7a2a7c2a91`.
  - A one-row-only template dispatch preserves the arithmetic VPOPCNT sign decoder exclusively for
    `kernels[0]` while every multi-row kernel keeps the table decoder. Native correctness passed
    **44/44** supported IQ2_XXS matmul cases plus the full quantization-function suite; the AVX2-only
    fallback compiled. In the fresh governed replay, `n=1` improved **+5.733%** median across all ten
    blocks (range **+5.325% to +6.027%**) while `n=512` returned to parity at **+0.020%** median
    (range **-0.117% to +0.219%**). Receipt:
    `/mnt/raid0/llm/autokernel/probes/inf37-iq2-fancy-simd-ab-v9-20260811-r5/receipt.json`, SHA-256
    `12dc4d95a8b208f97ce8c82ab7917f4e6aa28872a90c5fc85f15b72f07fa73ea`; candidate diff SHA-256
    `c24892485af0bddedc641b4ae764302a3c7dc070ed2d765c8e820c01f680b470` against frozen v9
    `0db32c06e3e550065b78311a6031ef3dd2c4f27c`.
- [ ] With OP-12 approval, commit the one-file IQ2_XXS one-row dispatch and run matched model-level TG/PP confirmation before any promotion claim.
- [x] **Settle the gfx90a low-precision datapath question at ISA level, so it stops being re-derived.** ✅ 2026-08-15 — zero GPU, zero inference. All 39 MFMA mnemonics extracted from the shipped clang and assembled under `llvm-mc` for gfx908/gfx90a/gfx942, discriminating "instruction not supported on this GPU" from "invalid operand"; re-run against a second toolchain, identical; AMD's matrix-instruction-calculator agrees on the count. **gfx90a has 27 MFMA instructions, and FP16 and INT8 have IDENTICAL shape sets** `{4x4x4, 16x16x4, 16x16x16, 32x32x4, 32x32x8}` — so int8 MFMA is **1× fp16, not 2×** (both 181.0 at peak), and routing a quantized GEMM through int8 MFMA on this card has **zero peak-rate advantage**. No fp8/bf8 (the `fp8-insts` target feature is absent), no `xf32`, no sparse `v_smfmac_*`. **INT4 has no matrix path on ANY CDNA generation** — zero `i4` MFMA mnemonics exist in the compiler; INT4 on gfx90a is VALU dot-product only (`v_dot8_i32_i4`). Two consequences already carried elsewhere: MFMA *decode* kernels are worth exactly zero here (decode sits 31–113× below the ridge — the `mfma-decode-kernels-are-worth-zero` HARD_CONSTRAINT row in `autokernel-research-loop.md` is the durable form), and the low-quant gap this handoff has been chasing is **VALU bit-unpacking and occupancy**, exactly what the disassembly diff above found, not a missing matrix path. Compiled into `wiki/hardware-optimization.md` (2026-08-15) with vendor citations and a new benchmarking trap: Composable Kernel does **not refuse** fp8 on gfx90a, it silently lowers each fp8 MFMA into 8 sequential fp32 ops, so a "CK fp8 GEMM on MI210" measures fp32 emulation, not a datapath.
- [ ] **TRIPWIRE — re-open the low-quant VALU cost the moment an IQ-format role becomes GPU-resident.** Filed 2026-08-15 at operator emphasis: the finding above is a **LATENT exposure, not a non-problem**. Nothing in production is exposed to the VALU bit-unpacking penalty *today* — no IQ-format role is currently GPU-resident, so the 14-VGPR overshoot and the 6-vs-8 waves/SIMD occupancy loss cost us nothing in the live stack. That flips the day any IQ-quant role is promoted onto the MI210 (the live candidates are tracked in `architect-model-selection-bench.md` and the IQ2 GPU-residency tasks in this handoff). **Reopen condition**: any registry change that makes a `IQ1_*`/`IQ2_*`/`IQ3_*`/`IQ4_*` GGUF resident on ROCm0 in a serving role. On that event the `v_perm_b32` re-expression task above stops being a speculative micro-optimization and becomes a measured production cost — re-price it then, and do **not** record the finding as "not a problem" in the meantime, which is how a latent exposure gets closed by mistake.
