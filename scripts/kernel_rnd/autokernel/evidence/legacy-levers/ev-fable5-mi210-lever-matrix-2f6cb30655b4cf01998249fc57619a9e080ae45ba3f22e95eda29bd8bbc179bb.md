# findings-05c — MI210 gfx90a Speed Campaign: Lever × Model-Category Isolation

> **Measurement discipline.** Every throughput number below is an **OBSERVATION** per MEASUREMENT.md — no `P-GPU-1` protocol exists, so **no cell is decision-gating**. Evidence tags: **[M]** a number exists in-source with a paired correctness/garble check · **[H]** reasoned, un-run · **[U]** queued, never run. Every dead/negative verdict carries its **regime qualifier** — there is no unqualified "X is dead" anywhere in this document. All verify-pass corrections from the isolation audit have been applied to the cells (the changed cells are flagged **⚙︎corrected** and expanded in §1.2).

> **⚠︎ `a8afd338` reconciliation (measured 2026-07-04, AFTER this matrix was built — supersedes several §3/§5 items).** This matrix's evidence was harvested before the Q8-dequant/MFMA kernel thread landed. Its measured results flip the TOP of the dense-Q8 execution plan:
> - **L12 n-gram / prompt-lookup GPU spec = MEASURED NEGATIVE** (was "[U] do-first" in §3.1 #1 and §5 #1). Every variant regresses on 27B-Q8: plain 28.4 → best ngram-simple 27.7; context-only acceptance ~15% << break-even. **Drop it from "do-first"** — a trained drafter (MTP/EAGLE3) remains the spec path. Corpus-static (L13) must clear a ~40–50% acceptance bar to beat this.
> - **L3 fused dequant-in-GEMV = NON-TASK, not the "main event"** (retires §3.1 #3 and §5 #2). The Q8_0 GEMV is already int8-native (`vec_dot_q8_0_q8_1` = `dp4a` + one fp scale/32-block); there is **no per-element dequant to hide**. The 47→62% gap is **BW/occupancy, not dequant-compute** — this mechanistically corrects the 2026-07-02 "quantized-dequant-artifact" framing carried in the L3 / Axis-C cells: the roofline-% *decrease* with quant is real, but it is lower compute/byte → more latency-bound, not dequant cycles.
> - **L4 async weight-prefetch → the #1 dense-Q8 single-stream lever** (was #4), now MEASURED [M]. Two landed stackable wins: **nwarps 2→4 = +4.6%** (→30.32 t/s, commit `5dc116130`), then **`raw.buffer.load.lds` LDS double-buffer = +3.3%** (30.20→31.20, output byte-identical, `test-backend-ops` 1103/1103, rocprofv2 MemUnitStalled −62% = mechanism confirmed, commit `7c28056b7`, runtime-gated `GGML_CUDA_Q8_PREFETCH` default-off). **Cap:** the prefetch covers only ~half the Q8 GEMV dispatches — the fused-SwiGLU FFN up/gate matmuls (the larger ~127 KB ones) are excluded; **extending to the fused path was tested and FALSIFIED** — coverage did double (to 100% of Q8 GEMV dispatches), but throughput *regressed* (−1.8% at full occupancy / −13% naive): the large FFN GEMVs are already wave-pipelined, so the per-iter `s_waitcnt`+barrier cost exceeds the stall-reduction (patch saved, not committed). **+3.3% is the CDNA2 ceiling for LDS-prefetch.** SoA-repack (L6) also NOT warranted — coalescing measured healthy.
> - **L2 quantize_q8_1 = DEFER** — measured **3.37%** (not 5.68%); graph-level fix only (activation caching / RMSNorm-fuse), correctness risk on the 82%-of-decode hot path.
> - **L7 MFMA prefill = DEFER (measured gate failed)** — prefill GEMM is already rocBLAS/Tensile MFMA + memory-bound (VALUBusy 3.55% / MemUnitBusy 78.5%); high-batch VALUBusy 16.8%. Both fail the "high VALUBusy + idle matrix cores" gate. Orthogonal levers instead: prefill skip Q8→f16 convert (~15%); fuse the high-batch norm tail (43% of B=128 time).
>
> **Reprioritized dense-Q8 order:** async-prefetch (nwarps=4 done → `raw.buffer.load.lds` in-flight → SoA-repack) → then the harder GDN-aggregate (L20) + MoE-mmid (L1-MoE) levers. Everything else in this matrix — the arch×substrate circumstance isolation (§2), all [H]/[U] hypotheses, the taxonomy gaps (§4) — stands unaffected. Full results: [../../progress/2026-07/2026-07-03-mi210-qwen36-27b-speed-campaign.md](../../progress/2026-07/2026-07-03-mi210-qwen36-27b-speed-campaign.md) + the dequant/MFMA handoffs.
>
> **SINGLE-STREAM DENSE-Q8 IS CLOSED (2026-07-04 Pass-2 diagnostic).** The megakernel (L5) is measured NOT worth building: HIP graphs (which kill ALL host-launch) buy only **+5.9%**, and decode is memory-latency-bound at ~50% roofline — the 62→100% gap is the batch-1 MLP floor, not a launch/grid-drain bubble (the trace's 64% "gap" was a profiler artifact: ~10µs × ~1860 kernels/token). Levers banked: MMVQ +17.4%, nwarps +4.6%, prefetch +3.3%; everything else ruled out with data. **Next phase = aggregate/MoE regime** (§3.3 + the L1-MoE mmid / L16 bf16-aggregate / L20 GDN-aggregate levers). One cheap non-megakernel note if ever revisited: the hybrid's ~1860 kernels/token is high → a *targeted* per-SSM-block elementwise/norm/gate fusion (not a full megakernel) is the only launch-side lever left.

## ⭐ v7-CANDIDATE SPEC SHEET (2026-07-06, VERIFIED + protocol-pinned) — all Q8+F16 models, optimal experimental kernel

Branch `experimental-v7-candidate` (fresh v6+iqk + MMQ/nwarps/prefetch/bf16-state + tree-draft, `46f876c12`), `GGML_CUDA_Q8_PREFETCH=1`. **Protocol (pinned):** production temp+seed42, fixed prompt, server model-identity VERIFIED, reps confirmed byte-identical at fixed temp. Every spec-dec number tagged `(temp, α)`; **α = ACCEPTANCE = the SPEED dial** — MTP/NEXTN is distribution-lossless so there is NO quality tradeoff (coherence PASS all temps; byte-differs from plain only via MMQ FP-drift). OBSERVATION.

### Single-stream (−fa 0) — DEPLOYABLE = production temp 0.1–0.3
| model | best config | **deployable t/s** | vs plain | temp→α curve (t0 / t0.2 / t0.6) |
|---|---|---|---|---|
| Qwen3.6-27B (dense-GDN) | **MTP/NEXTN** | **~41** (α .68 @t0.2) | +31% (plain 31.7) | 42.6(.69) / 41.4(.68) / 39.8(.64) — monotone↓ |
| gemma-4-31B (pure-dense) | **MTP** (ext head) | **~30** (α .45 @t0.2) | +15% (plain ~26) | 31.3(.49) / 30.1(.45) / **44.3(.84)** — NON-monotone, peaks t0.6 |
| Qwen3.6-35B-A3B (MoE **FRONTDOOR**) | **PLAIN (no MTP)** | **~101** | MTP LOSES (90<101) | MTP hurts every temp (MoE-GDN) |
| Qwen3.6-27B **F16**-proxy | MTP | ~31 | — | lever-2 proxy (t0.2, α~.6) |

- **temp→α is MODEL-SPECIFIC** (27B monotone↓, gemma peaks at t0.6, MoE hurts) — NO universal "temp-0 inflates" rule; ALWAYS report `(temp, α)`. **This model-specific curve × prompt-dependent α was the ENTIRE cause of the "non-reproducible" bounce** — gemma's old "45.7" was a temp-0.6/α-.84 figure (deployable t0.2 = ~30); 27B's old "40.4" ≈ verified 41.4@t0.2. Fixed (prompt,temp,seed) → byte-identical reps [[feedback_compare_against_top_optimized_spec]] [[feedback_production_sampling_seed_not_temp0]].
- **SUPERSEDES the earlier "gemma SS 45.7 / 27B SS 40.4 (MTP+MMQ)" line** — those conflated temperature/prompt. Never quote a high-temp α figure as the deployable spec.

### Aggregate (`llama-batched-bench` S_TG, plain — spec-dec degrades to ~0 at batch)
| model (optimal config) | B=1 | B=8 | B=16 | **B=32** |
|---|---|---|---|---|
| gemma-4-31B Q8 (−fa0; no GDN opt) | 27.1 | 104.0 | 174.3 | **245.9** |
| Qwen3.6-27B Q8 (−fa0, bf16-state) | 31.4 | 103.6 | 157.8 | **198.8** |
| Qwen3.6-35B-A3B Q8 (**FRONTDOOR**, −fa1, bf16-state) | 94.0 | 228.1 | 286.2 | **408.3** |
| Qwen3.6-27B **F16**-proxy (−fa0, bf16-state) | 19.2 | 72.6 | 109.3 | **141.2** |

- **bf16-state (L20) ON/OFF @B32 — regression PASS:** 27B 165.5→**198.8 (+20.1%)**, 35B-A3B 346.9→**408.3 (+17.7%)** (matches campaign +21.5%/+17.7%) — GPU opts survived the fresh-pull onto v6+iqk.
- **B=1-of-aggregate ≠ single-stream top:** it's plain at the aggregate FA config (e.g. 35B-A3B B=1 94.0 is −fa1; single-stream −fa0 plain = 101). Use the single-stream table above for SS.
- **Q4 note:** gemma Q4_K_M agg B32 = 272 raw, but Q4 is capacity-only on gfx90a (dequant-bound 33% vs Q8 47% roofline); F16 aggregate costs ~30% (141 vs 199) for precision.

Categories (columns): **D-Q8** Dense-Q8 · **D-16** Dense-fp16/bf16 · **MoE** MoE-on-GPU-decode · **GDN** GDN/SSM-hybrid (dense-FFN + MoE-FFN sub-variants) · **DiT** Diffusion/DiT · **Aux** Auxiliary (embed/rerank/vision). Verdict glyphs: **Y** applies · **C** conditional · **N** no.

---

## 1. Lever × model-category matrix

### 1.1 Compact grid (applies · evidence)

| # | Lever | D-Q8 | D-16 | MoE | GDN | DiT | Aux |
|---|-------|------|------|-----|-----|-----|-----|
| L1 | MMVQ→MMQ verify-dispatch fix | **Y**[M] +17–32% w/verify | **N**[M] no dequant path | **C**[M/U]⚙︎ landed flat; mmid untested | **C**[M] dense-FFN +17% / MoE-FFN flat | **N**[H] large-M ∈ MMQ | **N**[H] no verify batch |
| L2 | quantize_q8_1 requant kill | **Y**[H] ~+5.7% ceil | **N**[H] no requant | **C**[H] expert-path, frac unquant | **C**[H] Q8 prologue | **N**[H] amortized GEMM | **N**[H] f16 none / prefill-amort |
| L3 | Fused dequant-in-GEMV | **Y**[H]⚙︎ ~+19% (not +32) | **N**[M] already 62% | **C**[H] expert 25→35% | **C**[H] dense-FFN ~+19% | **N**[H] GEMV-only | **C**[H]⚙︎ VL decode tail only |
| L4 | Async weight prefetch / LDS DB | **Y**[H] +~13%→70%BW | **Y**[H] primary lever | **C**[H] dynamic-route discount | **C**[H] latency wall | **N**[H] compute-bound | **C**[H]⚙︎ VL decode tail |
| L5 | Persistent / megakernel decode | **C**[H] if residual bubble | **C**[H] if residual bubble | **C**[H] extra MoE kernels | **C**[H] if residual bubble | **N**[H] GEMM self-amort | **C**[H]⚙︎ VL decode tail |
| L6 | Weight swizzle / SoA Q8_0 repack | **Y**[H] coalesce; unmeas | **N**[H] 2B contiguous | **C**[H] per-expert blocks | **C**[H] shared w/L3 | **N**[H] wants MFMA-swizzle | **N**[H] M>1 / f16 |
| L7 | MFMA compute-bound paths | **C**[H] prefill/hi-batch | **C**[H] prefill; bf16 native | **C**[H] hi-batch GEMM | **C**[M-killed/H] decode killed, prefill open | **C**[H] **headline** denoise | **C**[H] ViT/BGE prefill |
| L8 | MTP / NEXTN self-spec | **Y**[M] +15.6% | **Y**[M-proxy] **+60.2%** on F16 (Q8→F16 dequant, α 66.9%≈Q8) = 4× the Q8 sign; abs 31<Q8 | **N**[M] −12% (head-indep) | **C**[M] dense-FFN win / MoE-FFN −12% | **N**[H] non-causal | **N**[H] no AR loop |
| L9 | −md double-load fix | **Y**[M] unlocks +15.6% | **Y**[M] MTP prereq | **N**[M] unlocks net-neg path | **Y**[M] NEXTN prereq | **N**[H] no AR head | **N**[H] no self-spec |
| L10 | EAGLE-3 draft head | **C**[M-hyb/U-pure] 25<33.6 | **C**[M-hyb/U-pure] | **C**[U]⚙︎ MoE untested (num=dense-hyb) | **N**+carve[M] slower-than-MTP on hyb; open dense/CPU | **N**[H] non-AR | **N**[H] no AR target |
| L11 | Tree-draft (DySpec) | **C**[U] pure-dense re-test | **C**[M]⚙︎ +15.8% f16; −53/−66 hyb-CPU | **C**[M-adj/U]⚙︎ GPU untested | **C**[M] −53/−66 CPU; GPU re-test | **N**[M-adj]⚙︎ no AR loop | **N**[M-adj]⚙︎ no decode loop |
| L12 | N-gram / prompt-lookup | **Y**[U] **do-first** | **Y**[U] **do-first** | **C**[U] MoE-verify headwind | **C**[U] dense-FFN win/MoE-FFN neg | **N**[U] no AR pos | **N**[H] no token stream |
| L13 | Corpus-static n-gram (−lcs) | **C**[U] code-corpus | **Y**[U] code-corpus | **C**[U] headwind + vocab-lock | **C**[U] FFN split | **N**[U] non-text | **N**[H] no decode |
| L14 | KV-quant (q8 K/V + −fa 1) | **C**[H]⚙︎ short=hyp-dead / long-SS alive | **C**[U/H]⚙︎ source-alive dense | **C**[M] dead frontdoor / long-SS alive | **C**[M] dead frontdoor / long-SS alive | **N**[H] non-causal | **N**[H] KV never re-read |
| L15 | Sub-4-bit (IQ2/TQ1/TQ2) | **C**[U] exits Q8 | **N**[U] exits category | **C**[U] **the** use case | **C**[U] large MoE-GDN | **C**[H]⚙︎ HBM-capacity fallback | **C**[H] ~neutral thru |
| L16 | bf16-vs-Q8 crossover | **C**[H] Q8 single / bf16 batch | **Y**[M-proxy] 744>561@B32 | **C**[M] Q8 96.6 / bf16 744@B32 | **C**[H] doubtful (latency-bound) | **C**[H] bf16 for batched denoise | **C**[H]⚙︎ VL prefill not decode-tail |
| L17 | vLLM serving fabric | **N**[M] arch-blocked gfx90a | **Y-stock**[M] +11/+24 Qwen3-8B | **N**[M] can't load MoE | **N**[M] GDN Triton blocker | **N**[U] no diff path | **C**[H] stock BGE loadable |
| L18 | GPU-draft / CPU-target spec | **N**[U] GPU-resident | **N**[U] GPU-resident | **C**[U] overflow-to-CPU MoE | **C**[U] CPU-target 122B | **C**[U] DFlash drafter role | **N**[H] no verify loop |
| L19 | Op-offload prefill (−ot/−n-cpu-moe) | **N**[U] no experts | **N**[U] no experts | **N**[U] GPU-only violation | **N**[U] GPU-only violation | **N**[H] dense, no experts | **N**[H] self-defeating |
| L20 | GDN occupancy + RS-traffic/layout | **C**[H] hybrid member only | **C**[H] hybrid member only | **C**[H] GDN-MoE subclass only | **Y**[M]⚙︎ occ NO-GO; **bf16-state GO +21.5%@B32** (`496e2f098`, drift+isolation PASS) | **N**[H] no recurrence | **N**[H] no recurrence |
| L21 | Q4_K dequant side bet | **N**[H] Q8-path already | **N**[H] no dequant | **C**[H] expert Q4_K path | **C**[H]⚙︎ ~+43%→~47 t/s | **C**[H] HBM fallback | **C**[H] VL prefill only |
| L22 | FA decode gating (−fa0 dec/−fa1 pre) | **Y**[M] −fa0 29.4>28.8 | **Y**[M]⚙︎ −fa0=138/166 (swap fixed) | **Y**[M]⚙︎ frontdoor MEASURED: −fa0 SS/−fa1 aggB≥16, MARGINAL (+2.8%@B32, GDN-suppressed) | **Y**[M] −fa0 29.4>28.8 | **C**[H] non-causal→−fa1 branch | **C**[H] prefill→−fa1 branch |

### 1.2 Expanded cells (corrections applied + load-bearing qualifiers)

- **L3 D-Q8 ⚙︎corrected (headline magnitude).** The 15 pp Q8-vs-fp16 roofline gap is **[M]**; the recoverable estimate is **~+19%** (source brackets Q8 52% deployed → 62% fp16 ceiling; 62/52). **+32% is only the optimistic kernel-level 47%→62% bound** (rocprof B1 −fa 0 endpoint) and is **not** the headline. Caveat: the 62% fp16 ceiling was measured on **stock Qwen3-8B fp16**, not our 27B — a further reason to hold this at **[H]**.
- **L3 Aux ⚙︎corrected N→C.** N/A for embed/rerank (single forward pass) **and** for the prefill/encode hot path — but a VL model autoregressively generates its answer, and that **Q4_K VL single-stream decode tail is an M=1 `mul_mat_vec_q` GEMV where this lever fires**. Verdict: conditional (VL decode tail, minor, untested).
- **L4 / L5 Aux ⚙︎corrected N→C.** Same reasoning: the async-prefetch batch-1 latency-wall target and the megakernel batch-1..8 decode-bubble both **exist in the VL decode tail** (batch-1 GEMV). Conditional (VL tail, minor); still hedged (external CDNA3/4, unbuilt on CDNA2).
- **L8 D-16 — now [M-proxy] MEASURED (2026-07-06), hypothesis CONFIRMED and then some.** Qwen3.6-27B Q8→F16 dequant proxy (values preserved so α≈Q8's, isolating the kernel-path): **plain F16 19.37 → MTP F16 31.03 t/s = +60.2%**, α 66.9% (170/254), mean-accept 3.00/3, per-position (0.871, 0.647, 0.482). This is **~4× the Q8 +15.6%** — the "plausibly stronger" hypothesis was right because F16 is 2 bytes/param (more BW-bound), so each accepted draft token avoids a fatter weight read. **BUT abs throughput F16-MTP 31.0 < Q8-MTP 33.6 < Q8-MTP+MMQ 40.4** — F16's doubled weight bytes dominate, so F16 is NOT a throughput choice; it is a precision choice that MTP makes far cheaper. Proxy α is Q8-valued; a TRUE-fp16 GGUF (download the official Qwen3.6-27B fp16 HF + `convert_hf_to_gguf --outtype f16`, ~54 GB) is needed to confirm the real α (likely ≥ proxy, more accurate draft) and to measure whether fp16 QUALITY justifies the throughput cost. The old +44% Q4 proxy stands separately.
- **L10 MoE ⚙︎corrected.** The 25.0 vs 33.6 t/s EAGLE-vs-MTP pair is measured on the **Qwen3.6-27B DENSE hybrid test vehicle** (its MTP baseline is 33.6; the MoE frontdoor MTP is ~86). For the MoE category (gemma-26B-A4B, 35B-A3B) EAGLE-3 is **[U] untested**, confidence low. Dead-verdict stays regime-scoped ("slower-than-MTP on qwen35-hybrid-GPU; open + SOTA-plausible for dense transformers / CPU").
- **L10 GDN carve-out.** On qwen35-GDN-hybrid-GPU EAGLE-3 (community head, n_max=3) is strictly dominated (25.0 < MTP 33.6 **and** < plain 29.06, accept 2.34<2.99) → "no-go here." **Not refuted** for dense-transformer targets (where EAGLE-3 is the vLLM/SGLang SOTA) or CPU targets. Capability finding: PRISM-EAGLE3 head loads + survives 900 tok on the fork (upstream #24541 crash does **not** reproduce).
- **L11 D-16 ⚙︎corrected number.** Corrected hybrid/CPU-era throughput range is **−53% to −66%** (not −22..−66). The **−22** was the frozen-multipath **acceptance** delta (−12..−22 **pp**), not a throughput figure. The in-category positive **+15.8% on 32B-f16 DENSE** stands.
- **L11 MoE / DiT / Aux ⚙︎corrected orphan −22%.** The "−22% MoE@48t" point is a **CPU/v2-era, out-of-evidence** number (traces to the tree-speculation handoff, not findings-05b). Cite the supported **−53/−66%** (CPU-hybrid, GPU-retest-warranted); treat −22% as either the acceptance-pp figure or an un-listed CPU number. MoE-on-GPU tree-draft itself = **[U]** (GPU state-clone flips to ~0.1 ms, but the expert-verify overhead headwind remains).
- **L14 D-Q8 ⚙︎corrected mislabeled/wrong-regime.** There is **no dense-Q8 KV-quant number**. The "±1.5% noise / saves ~0.94 GB / VRAM-not-binding" figure is measured on the **MoE-GDN frontdoor at long-ctx (80k) 128-way aggregate**. Correct framing: dead is **[M] only** on the weight-dominated MoE frontdoor; for dense-Q8 short-ctx/aggregate it is a **[H]** (weights still dominate); **single-stream long-ctx** (KV rivals weights; the hybrid's ~1/4 full-global unbounded-KV layers grow fast) is the **alive/[H]** regime. **[SCOPED 2026-07-05 → DEFER (marginal, rider-only): NO dedicated GPU run.** Only the qwen35 ~1/4 full-global attn layers @ single-stream 32-64k are alive; GDN keeps KV O(1), gemma SWA bounds it, aggregate is weight-dominated (3/4 resident classes ~0 payoff), and CPU precedent shows the dequant cast COSTS throughput (+9% wall/−30% gen). No deployed role needs it. Run only as a cheap ~2-4h RIDER on a future dense-full-global long-ctx role to close the [U] with data; recipe scoped. Not a speed lever — a max-context/VRAM characterization.]**
- **L14 D-16 ⚙︎corrected over-broad-dead + wrong-regime.** Downgrade "measured dead short-ctx/aggregate" to **[U] / source-hypothesized-alive**. The lone measurement is MoE-Q8 80k 128-way aggregate with VRAM **not** binding — non-transferable to dense. The same source explicitly hypothesizes KV-quant **alive for "a dense model, or single-stream long-context where KV rivals the weights."** For bf16 (~50 GB weights, ~55 GB @B32 on 64 GB) VRAM **binds sooner**, making aggregate KV-quant a plausible slot-capacity lever, not dead.
- **L16 Aux ⚙︎corrected wrong-regime.** The measured crossover is **bf16-wins-only-at-aggregate-B=32** (compute-bound). The row had mapped it onto the Q4_K VL **decode tail** — a low-batch, memory-bound regime where **Q8** wins. Correct auxiliary analog of the compute-bound-batched regime is **VL prefill / concurrent batched BGE encode**, not the decode tail. For f16 BGE the point is moot (already native matrix-core).
- **L20 GDN ⚙︎corrected wrong-regime scaling.** The single "3.4x B1→B32" is the **MoE-FFN frontdoor** (qwen35moe 35B-A3B). The **dense-FFN 27B** test vehicle scales **~5.8x** (29.4→165.8 @B32). Restate by sub-variant: **dense-GDN ~5.8x, MoE-GDN ~3.4x** — both GDN-suppressed vs non-GDN counterparts (gemma-MoE 5.9x, gemma-dense 8.6x), so the "GDN-is-the-ceiling" mechanism holds for both; only the flat 3.4x was mis-scoped.
- **L21 GDN ⚙︎corrected number.** "+55% → ~45 t/s" is untraceable/self-inconsistent. From the measured Q4_K_M baseline **32.88 t/s / 33% roofline** to Q8's **47% / 766 GB/s** efficiency, the BW ratio is 47/33 = **~+43% → ~47 t/s**. (The source's +55% is off a **~29 t/s** baseline, and the ~45 t/s endpoint alone implies only +37% — neither reconciles with the other.) Keep **[H]**, quality-gated (PPL/eval-parity), kernel un-authored.
- **L22 D-16 ⚙︎corrected data swap.** The aggregate pair was written backwards. Correct: **−fa 0 = 138/166** t/s (higher/better) vs **−fa 1 = 135.5/162.2** @B16/32 — consistent with the (correct) "FA never helps DECODE here" direction.
- **L22 MoE ⚙︎corrected attribution.** The 28.8/135.5/162.2 vs 29.4/138/166 numbers are the **DENSE 27B vehicle** (B=1 ~29 is the dense 27B's **plain FA-isolation baseline** — MTP/MMQ held OFF to isolate the FA variable; it is NOT the dense achieved throughput, which is **40.4 t/s = MTP + MMVQ→MMQ, +37%** [`de447119f`]. Contrast is only for attribution: dense-B1 ~29-plain vs MoE frontdoor-B1 ~97-plain). Keep applies:Y and "−fa 0 for decode" (gfx90a FA-is-prefill-only is a substrate property, low-risk across archs), but the **MoE-frontdoor FA-decode A/B is now [M] MEASURED (2026-07-06, 35B-A3B-Q8, MI210)** — and the prior "~97" estimate was dead-on (llama-bench tg128 **99.64 t/s -fa0** vs 94.68 -fa1, +5.2% single-stream decode). Aggregate (batched-bench S_TG): −fa1 wins B≥16 (B32 **342.3** vs 333.0, **+2.8%**; B8 ~tie; B1 −fa0 95.8 vs 94.5). So the MoE crossover (−fa0 single-stream / −fa1 aggregate B≥16) HOLDS on the frontdoor but is MARGINAL — GDN suppresses the attention fraction FA optimizes, so the frontdoor's +2.8%@B32 is far below gemma-26B pure-MoE's +16%@B32. Coherence PASS (Rayleigh-scattering reasoning, 98.5 t/s). NOTE: single-model GPU characterization; the frontdoor is deployed on CPU and the full stack does not fit 64 GB.
- **L15 DiT ⚙︎corrected N→C.** On a 64 GB GPU-compute-only card, a DiT too large to fit at Q4/bf16 **requires** sub-4-bit to run at all → applies in the capacity regime. Throughput hypothesis unchanged (net-neutral-to-negative for compute-bound denoise: dequant VALU added, no BW relief).

---

## 2. Circumstantial-dimension isolation (the core ask)

For each axis: the levers that **flip sign or magnitude**, and why.

### Axis A — Batch size (B=1 single-stream ↔ B=8/32/128 aggregate)
- **bf16-vs-Q8 (L16) FLIPS SIGN.** Q8 wins single-stream (96.6 vs 73.1 t/s, fewer bytes to move on a BW-bound path); bf16 wins @B=32 (744 vs 561). **Mechanism:** at high batch the GEMM turns **compute-bound**, and bf16 runs native on CDNA2 matrix cores with **nothing to dequant-amortize**, while Q8 pays a per-batch dequant tax.
- **MFMA (L7) FLIPS APPLICABILITY.** MfmaUtil ≈ 0% at B=1 decode (matrix cores idle → killed-by-profile) → **alive** at high-batch expert/prefill GEMM (VALUBusy climbs). Same profile class that killed GDN-MFMA-decode.
- **All spec-dec levers (L8/L10/L11/L12/L13) are a single-stream story.** Their gain **degrades toward zero as batch rises** — at B=32 the slots already saturate compute, so a batched verify competes with real work instead of filling an idle pipe. MTP is fundamentally a B=1 lever.
- **GDN occupancy (L20) GROWS with batch.** GDN share of decode goes **2%@B1 → 19.5%@B32** (abs ×39), so the recurrent-state ceiling is invisible at B=1 and dominant in aggregate.
- **MMVQ→MMQ (L1)** targets exactly the small verify batch (ne11≈4) that a draft creates in the single-stream+spec regime — inert at pure B=1 GEMV and above the small-batch dispatch boundary at large B.

### Axis B — Context length (short ↔ 64k+)
- **KV-quant (L14) FLIPS SIGN.** Dead for short-ctx/aggregate (weights dominate bytes/token; KV savings are noise, VRAM not binding) → **alive** for single-stream long-ctx where **KV read rivals weight bytes**. **Mechanism:** KV bytes scale with ctx; at 64k the per-step KV read approaches the weight read. **Arch-modulated:** qwen35's ~1/4 full-global **unbounded**-KV layers drive −22.3% decode 1k→64k, so its KV-quant relevance rises faster than gemma **SWA**'s −7.9% (bounded KV).
- **MFMA (L7) + FA-prefill (L22) GAIN MAGNITUDE.** Long prompts shift work into **prefill/TTFT**, where VALUBusy ~50% leaves matrix-core headroom and the **−fa 1 prefill branch** engages (it only ever costs decode).
- **Cross-arch re-flip (taxonomy gap):** under **MLA** (GLM-4.7-Flash) KV is latent-compressed, which **re-flips the KV-quant math again** — the L14 verdict measured on GQA does **not** transfer to MLA.

### Axis C — Quant level (fp16/bf16 ↔ Q8 ↔ Q4_K ↔ sub-4-bit IQ/Q2/TQ)
- **MMVQ→MMQ (L1) FLIPS MAGNITUDE:** Q8 **+17–32%**, Q4_K **+5.8%**, fp16 **N/A** (no dequant to amortize), IQ **untested/different codebook kernel**.
- **Fused dequant-in-GEMV (L3) target moves:** Q8 **47→62%** (~+19%), Q4_K **33→47%** (~+43%), fp16 **nothing to fuse** (already at 62%).
- **bf16-batched crossover (L16):** only **no-dequant** weights (bf16) amortize into compute at high batch — the flip is quant-gated, not just batch-gated.
- **quantize_q8_1 requant (L2):** exists **only** on the quantized (Q8/Q4_K) activation path; **zero surface** for fp16/bf16.
- **HBM-capacity binding:** bf16 ≈ 2× bytes → VRAM binds sooner (bf16 27B ~55 GB @B32 fits 64 GB; 80B/122B bf16 do **not**); **sub-4-bit (L15)** is the axis that decides whether a large MoE is a GPU candidate at all.
- **Q4_K side bet (L21)** exists only on the Q4_K MMQ dequant path — inert for Q8/fp16.

### Axis D — Arch (dense ↔ MoE-separate-dispatch ↔ GDN-fused ↔ diffusion) — **the master axis**
- **MTP / self-spec (L8) FLIPS SIGN.** **+15.6% dense** (BW-bound plain decode absorbs the cheap resident-hidden-state verify) → **−12% MoE-on-GPU** (plain MoE decode already fast — reads only ~active-expert bytes off 1.6 TB/s — so draft+verify overhead isn't repaid; **head-quant-independent**, proven with both Q8 and f16 heads at ~62% accept) **[RE-MEASURED 2026-07-05 — temp-dependent, ~NEUTRAL at the production regime.** MTP-on-GPU-MoE delta vs output temperature (35B-A3B, seed 42, experimental build): **temp 0 (greedy) +6.5%** (accept 0.79) · **temp 0.2 (PRODUCTION — registry intent 0.1–0.3 + greedy fallback) −1.6%** (accept 0.63) · **temp 0.6 −6.8%** (accept 0.57). Production runs low-temp so the operative number is **~neutral (−1.6%, within single-sample noise)**. `de447119f` (+17.4% MTP-verify MMQ) **neutralized the old −12% penalty** → MTP is a **WASH** on GPU-MoE at production temp: not worth enabling as a speed lever, but no longer a reason to avoid it. Three-way flip-flop root cause = measured arbitrary temps before the deployed config; lesson [[feedback_production_sampling_seed_not_temp0]].] → **N/A diffusion** (non-causal, no AR token). GDN-hybrid **splits by FFN sub-variant**: dense-FFN win / MoE-FFN lose.
- **MMVQ→MMQ (L1) FLIPS PATH.** Dense weights route `mul_mat_vec_q` (the fix's target); **MoE experts route a separate `get_mmvq_mmid_max_batch` (mmid) dispatch the fix never touches** → landed fix is flat on the frontdoor; the mmid analog is the untested MoE lever. Diffusion is GEMM-side entirely.
- **Tree-draft (L11):** dead on hybrid/MoE-at-48t (CPU/v2-era −53/−66%, recurrent state-clone term) → **+15.8% on pure-dense-f16**; guarded off `!has_recurrent` so it **won't even run** on GDN today.
- **EAGLE-3 (L10):** slower-than-MTP on qwen35-GDN-hybrid-GPU; **open/SOTA-plausible for dense**; **untested for MoE**; N/A diffusion.
- **GDN occupancy (L20):** qwen35-GDN-**exclusive** — zero mechanism for pure-dense, gemma-MoE, or diffusion.
- **MFMA (L7):** **dead for GDN decode** (MfmaUtil 0%) but the **headline** for diffusion denoise (large dense non-causal GEMM = classic matrix-core workload).
- **KV/attention kernel identity** (GQA vs GDN-recurrent vs SWA-capped vs full-global) is what drives the L14 context-flip above; it is an arch property, not a tunable.

### Secondary axes (campaign-load-bearing, under-weighted because the campaign centered on one dense 27B vehicle)
- **Substrate GPU↔CPU:** re-flips **MTP** (GPU-MoE lose vs **CPU-MoE win**, both BW-bound-plain-decode logic inverted), the **tree-draft state-clone term** (450 MB @ CPU-BW killed it vs 149 MB @ 1.6 TB/s ≈ 0.1 ms on GPU → flips), and **MMVQ→MMQ** (CDNA2-specific; CPU iqk already amortizes → no transfer).
- **Prompt type (code/JSON/structured ↔ prose):** the entire value of **L12/L13 (n-gram)** lives here — high acceptance on repetitive tokens, ~0 on free prose.
- **Expert-activation fraction / expert count (A3B/A4B ~8-of-N ↔ 256-of-8 ultra-sparse):** flips the cost of the expert-gather/scatter kernel and the **relevance of the mmid dispatch (L1-MoE)** — far more valuable at 256 experts than at A3B.
- **HBM-capacity binding (fits 64 GB ↔ spills):** gates the bf16-batched win, the KV-quant value (dead when VRAM slack), and **which category is even a GPU candidate** (122B ~69 GB / 230B / 480B don't fit → size-gated by quant).

---

## 3. Per-category ranked execution plan (run when a8afd338 frees the GPU)

Ordered highest-ROI-first. Each item: **one decisive experiment** → **acceptance bar**. All bars pair a speed delta with a correctness/garble or PPL check.

### 3.1 Dense-Q8 (dominant deployed single-stream target)
1. **N-gram / prompt-lookup (L12) — do FIRST, zero kernel.** Run 27B-Q8 `--spec-type ngram-cache` over a code/JSON/repetitive prompt set vs plain. **Bar:** >+5% t/s on structured with no garble; ~0 on prose is acceptable (gate: don't enable for prose). Bonus: its verify batch is exactly what the landed **MMVQ→MMQ** fix amortizes.
2. **quantize_q8_1 requant kill (L2) — low effort.** Fuse/cache the requant into the GEMV prologue; rocprof-confirm the 5.68% slice is gone. **Bar:** +3–5% single-stream, PPL unchanged.
3. **Fused dequant-in-GEMV (L3) — main event, high effort.** Port the CPU iqk weight-block-outer pattern to HIP; measure achieved BW. **Bar:** ~52→~62% roofline (**~+19%**), bit-exact or PPL parity. (Do the SoA repack **L6** jointly.)
4. **Async weight prefetch / LDS double-buffer (L4) — medium-high; 2026-07-04 update flags it dominant over megakernel.** Two-LDS-buffer via `raw.buffer.load.lds`, VMCNT-scheduled. **Bar:** 62→~70% achieved BW (+~12%), PPL unchanged.
5. **Weight swizzle (L6) — gate first.** Read `TCC_EA_RDREQ_32B` sub-line ratio; repack only if amplification is high. **Bar:** measurable drop in sub-128B reads → BW gain.
6. **Persistent/megakernel (L5) — only if** the wall-budget (`GGML_HIP_GRAPHS=ON` vs `0`) shows a large residual device-side bubble. **Bar:** residual bubble > prefetch's remaining headroom before authoring.

### 3.2 Dense-fp16/bf16
1. **Async prefetch (L4) — THE primary lever.** No dequant gap to close first, so the **entire 62→~70% residual is prefetch-addressable**. **Bar:** +~12%, PPL unchanged.
2. **N-gram (L12) — cheap, precision-agnostic.** Structured-prompt A/B. **Bar:** >+5% structured.
3. **MTP on fp16 (L8) — [U].** 27B fp16 no-`md` self-spec vs plain. **Bar:** ≥+15% (match/beat the Q8 dense sign; hypothesis says stronger).
4. **Tree-draft on pure-dense (L11).** gemma-4-31B bf16 (no recurrence, so it actually runs). **Bar:** reproduce the +15.8% sign seen on 32B-f16.
5. **bf16-for-batched selection (L16).** Deploy bf16 for the high-concurrency fan-out role, Q8 for single-stream (crossover confirm). **Bar:** bf16 > Q8 aggregate @ target batch, fits HBM.
6. **MFMA prefill (L7)** — TTFT for long prompts; gate on VALUBusy high / MfmaUtil low.

### 3.3 MoE-on-GPU-decode

> **⚙︎ MoE-aggregate characterization (2026-07-04, gemma-26B-A4B) updates this section — measured:** (1) **L1-MoE mmid dispatch = NEGATIVE, not a lever** (item 1 below FALSIFIED — forcing experts to MMQ at low batch inverts the dense result, B2 −30% / B4 −21% / B8 −10.5%; MMVQ_moe is correct for sparse low-batch, default threshold=8 already optimal; zero surface at B≥16). (2) **NEW zero-code aggregate WIN — FA-decode**: unlike dense-27B, `-fa 1` WINS for MoE aggregate (B≥8): B32 +16–18%, B128 +30–43%, peak bf16-fa1@B128 = **1548 t/s** (resolves L22). Deploy -fa1 for B≥8, -fa0 single-stream. (3) **Real bottleneck = Q8-MMQ GEMM (61% @B32) + f16 attention (18%), NOT gather/scatter (~3%)** → the #1 kernel lever is **L3-MoE Q8-MMQ fused-dequant efficiency** (`quantize_mmq_q8_1` + per-tile dequant tax makes bf16 beat Q8 +32%); feasibility probe VERDICT: **NOT low-hanging** — a structural occupancy/streaming inefficiency (Q8-MMQ occupancy 2.61 vs bf16 3.22, register-pressure-limited; requant round-trip only ~2–5%, int8-MFMA underutilized not saturated). MMQ occupancy/tiling rewrite — **BUILT + FALSIFIED (NO-GO, 2026-07-04)**: the compact-LDS rewrite worked mechanically (LDS 49→25 KB, residency 1→2 WG/CU, correct) but occupancy stayed FLAT — at B=32 the kernel is **grid-limited (104 WGs = 1/CU), NOT LDS-limited**; bf16 wins on native-MFMA, not occupancy. Aggregate +1.6% B=32 / −12% B=64. **[CORRECTED 2026-07-05: stream-K is ALREADY the LIVE MMQ path on CDNA2](mmq.cu `use_stream_k=true` for CDNA2; the 104-WG grid = `nsm` persistent blocks = stream-K working as designed, one balanced block/CU, NOT a naive tiling to fix). It PRODUCED this very aggregate baseline. 'stream-K as a bigger separate bet' was a factual error. The only untested residual = raise the persistent grid `nsm→k·nsm` (2 WG/CU) + the saved compact-LDS patch (~2-line change, expected +0-10%, IQ2/capacity slot only), gated on a zero-build read of the captured pmc CSVs — see below.]** **Settled: bf16 is the aggregate answer; Q8 for HBM-capacity only.** **L15 sub-4-bit = MEASURED VIABLE (2026-07-05): the 122B architect runs FULLY GPU-RESIDENT at IQ2** (47 GB, 43.7 single / 148.7 aggregate @B32 with bf16-state +16.4%, IQ2 PPL 5.02 healthy — no collapse), **eval-parity PASSED judge-free (2026-07-05): IQ2 ≈ Q4** — 212-Q deterministic paired eval, IQ2 163/212 = Q4 163/212 (Δ0.0pp, p=1.000, symmetric noise). *Correction:* the "93%" was the **35B coder**, not the 122B (Q4-122B architect = 2.57/3 = **85.67%**); the LLM-judge weighted-rubric architect gate (70 Qs) is deferred — needs a cross-family judge, not GPU-only.** Detail: [mi210-q8-dequant-gemv-roofline.md](mi210-q8-dequant-gemv-roofline.md). (4) bf16↔Q8 crossover ≈ **B=16–24**, HBM-fit-gated. Deployment wins → [moe-aggregate-deployment-wins-brief.md](moe-aggregate-deployment-wins-brief.md).
1. **mmid dispatch threshold analog (L1-MoE) — [U], the deferred production-relevant fix.** Apply a `get_mmvq_mmid_max_batch` threshold; measure gemma-26B-A4B step-rate **and** acceptance. **Bar:** net e2e > 0 (step-rate gain must survive MMQ non-bit-exact acceptance perturbation, which measured −6 pp on the frontdoor and dominated to ~−5% e2e).
2. **bf16-for-aggregate selection (L16) — zero-code, measured.** Deploy bf16 for high-concurrency (744 vs 561 @B32), Q8 for single-stream (96.6 vs 73.1). **Bar:** already measured — wire the role→precision selection.
3. **Fused dequant on the expert MMVQ path (L3-MoE) — high effort.** MoE Q8 sits at only **25% roofline** (vs bf16 35.7%), a bigger gap than dense. **Bar:** 25→~35%, PPL parity.
4. **Sub-4-bit for oversized MoE capacity (L15) — [U].** GLM-5.2 UD-IQ2 ~238 GB / 122B architect. Gated on a CDNA2 sub-4-bit dequant kernel + PPL gate. **Bar:** fits + eval-parity.
5. **N-gram on MoE (L12) — cheap probe only.** **Bar:** structured-output acceptance high enough to offset the MoE-verify headwind (net >0); otherwise shelve.
6. **Do NOT invest in MTP/EAGLE/tree for GPU-MoE decode** — MTP measured **−12%** (regime: MoE-on-GPU decode, head-quant-independent). Regime-qualified: MTP is +15% for dense and a win on CPU-MoE; EAGLE untested here.

### 3.4 GDN/SSM-hybrid
1. **(dense-FFN) MTP already +15.6%, stacked with MMVQ→MMQ = +37% (→40.4 t/s).** Confirm the deployed config takes the no-`md` self-spec path.
2. **(dense-FFN) N-gram on structured (L12) — do-first probe.**
3. **GDN occupancy + recurrent-state traffic/layout (L20) — SCOPED 2026-07-04.** **Occupancy build = NO-GO (structural):** `gated_delta_net_cuda` is 32/256 VGPR, 0 LDS, grid 49152 (472× CUs) → theoretical occupancy is ALREADY 100%; the 42% is pure memory-latency (MemUnitBusy 65%, 266 GB/s), not a relievable resource limiter — same mis-spec class as L3-MoE, caught at scope (no wasted build). **The one real lever = bf16 recurrent state — BUILT + GO (`496e2f098`, 2026-07-04): +21.5% aggregate @B32** (162.8→197.8; BEAT the +11% projection — bf16 halves the state gather+scatter too, so the whole ~32%-of-decode recurrent machinery benefits; L2 hit 47.8→59.9%, VALUBusy 15.7→56%). **Gates PASS:** drift PPL +0.0035% (~500× under CI) + 512-tok coherent; isolation `test-backend-ops` 1103/1103 + gemma-MoE byte-identical. Runtime-gated `GGML_CUDA_GDN_STATE_BF16` default-off. **Wires all 3 GDN-hybrids → the DEPLOYED frontdoor 35B-A3B + architect 122B inherit it** (frontdoor 35B-A3B CONFIRMED +17.7% @B32, byte-identical; architect 122B CONFIRMED +16.4% @B32 (all 3 GDN-hybrid sizes)) — a real deployed-role aggregate lever, NOT just the test vehicle. B=1 neutral (high-batch only). Occupancy was never the lever — precision/traffic was.
4. **Tree-draft GPU re-test on the hybrid (L11).** GPU state-clone is ~0.1 ms (149 MB @1.6 TB/s) vs the CPU term that caused −53/−66%. **Bar:** flip from negative toward positive; still watch recurrent-COMPUTE scaling.
5. **Q4_K dequant efficiency (L21).** **Bar:** ~+43% → ~47 t/s at half the weight bytes, PPL/eval-parity.
6. **KV-quant single-stream long-ctx (L14).** Exercise the ~1/4 full-global unbounded-KV layers at 64k single-stream. **Bar:** decode t/s gain as KV rivals weight bytes (the alive regime the aggregate no-help finding never covered).
7. **(MoE-FFN) Do NOT enable MTP** — measured −12% for MoE-FFN GDN-hybrid on GPU.

### 3.5 Diffusion/DiT
> **Blocker:** no diffusion serving path exists on ROCm/HIP — every cell is [H]/[U]. All items gated on standing up a pipeline first.
1. **Stand up a HIP/diffusers-ROCm DiT serving path (prerequisite).**
2. **MFMA denoise kernel (L7) — headline.** rocprofv2 the denoise GEMMs first. **Bar:** high VALUBusy + low MfmaUtil confirmed, then MFMA routing yields measurable prefill/denoise throughput gain.
3. **bf16 for batched denoise (L16).** Diffusion is definitionally compute-bound/batched → bf16 preference. **Bar:** bf16 ≥ Q8 for the batched denoise, fits HBM.
4. **−fa 1 prefill branch (L22)** for non-causal full-block attention.
5. **DFlash GPU-hosted block-diffusion drafter (L18) — creative role.** Not diffusion-as-target; a denoise drafter feeding an AR target. Gated on the MFMA denoise kernel + a trained DFlash head. (Flagged per research-intake — do not dismiss.)
6. **Sub-4-bit (L15)** only as an HBM-capacity fallback for a DiT too large for bf16/Q4.

### 3.6 Auxiliary (embed/rerank/vision)
> **Blocker:** no auxiliary model was ever benchmarked on this substrate — all [H]. First task is to create the missing baseline.
1. **Benchmark a BGE encode + a VL vision-encoder prefill on the card.** Establish the absent baseline (this is the gating deliverable).
2. **MFMA prefill/encode (L7) — gate on rocprofv2.** f16 BGE may already be MFMA-saturated (rocBLAS/hipBLASLt) → possibly no win; the **Q4_K VL MMQ prefill** path is the more plausible MFMA-idle target. **Bar:** high VALUBusy + low MfmaUtil on that model's GEMMs before building.
3. **vLLM for stock BGE-BERT (L17).** vLLM has native pooling/embedding on ROCm and BGE is stock BERT (loadable, unlike our custom archs). **Bar:** match/beat the fork's embedder throughput; weigh a second serving fabric for one role.
4. **−fa 1 prefill branch (L22)** — marginal for BGE ctx=512 bidirectional, larger for VL long visual-token prefill.
5. **bf16 already deployed for BGE (0.64 GB, native matrix-core)** — nothing to move; confirm no residency pressure.

---

## 4. Gap list — queued measurements (checklist)

**[U] = runnable now (flag flip / config), needs GPU only.** Highest-ROI first.
- [x] **N-gram / prompt-lookup on GPU** (L12) — **SUPERSEDED, do not run** ✅ 2026-08-12 (`mainB`).
  Already MEASURED and NEGATIVE at `a8afd338` (2026-07-04) — recorded in **this file's own line ~6**
  reconciliation block: *"Every variant regresses on 27B-Q8: plain 28.4 → best ngram-simple 27.7;
  context-only acceptance ~15% << break-even. Drop it from do-first."* The arms this row names
  (`ngram-simple|ngram-cache|ngram-map-k`) are the arms that were measured. Re-running would consume
  a multi-hour GPU window to reproduce a known regression.
  **Two independent reasons it could not have run as written anyway**: (a) no fp16 27B model exists on
  disk, so the "27B-Q8 + fp16" matrix has no fp16 arm; (b) the row still carries the *"do-first"*
  designation that the same reconciliation explicitly retired.
  **The live successor is L13** (corpus-static n-gram), which must clear a ~40–50% acceptance bar to
  beat this; the trained-drafter path (MTP/EAGLE3) remains the spec route.
  *Recorded because the checkbox and the retraction lived 193 lines apart in one file for 39 days.*
- [x] **MTP on fp16/bf16** (L8) ✅ 2026-07-06 — MTP **+60.2%** on F16 (Qwen3.6-27B, Q8→F16 dequant proxy, α 66.9%≈Q8, mean-accept 3.00/3) vs Q8's +15.6% → BW-bound hypothesis CONFIRMED (F16 2 bytes/param → each accepted draft avoids a fatter weight read). **Caveat:** F16 abs 31.0 t/s MTP stays BELOW Q8 40.4 (MTP+MMQ) — F16 is a precision choice, not a throughput one; MTP just makes it far more affordable. TRUE-fp16 download pending to confirm real α + measure quality headroom (proxy α is Q8-valued).
- [ ] **KV-quant single-stream long-ctx** (L14) — dense-Q8 **and** GDN full-global layers at 64k single-stream (the alive regime never measured).
- [x] **FA-decode A/B on the MoE frontdoor** (L22) ✅ 2026-07-06 — MEASURED (35B-A3B-Q8): −fa0 single-stream (99.64>94.68, +5.2%) / −fa1 aggregate B≥16 (342.3>333.0 @B32, +2.8%). MoE crossover holds but MARGINAL (GDN suppresses the attention fraction FA targets; gemma-26B pure-MoE was +16%). Coherence PASS.
- [x] **Tree-draft GPU re-test** (L11) ✅ 2026-07-06 — DySpec Phase 1a ported + validated on the v7-candidate (engine bit-identical to linear draft) then **SHELVED**: external-drafter tree is DOMINATED by embedded MTP on every stack target (qwen-27B MTP 41.9 vs tree ~18 < plain 31), and GLM-5.2 also ships an MTP head → no MTP-less niche remains. Native-GLM-MTP forward-graph port is the better future lever (gated on GLM-5.2 runnability). See [tree-draft-forward-port-plan.md](tree-draft-forward-port-plan.md).
- [ ] **EAGLE-3 on pure-dense** (L10) — needs a trained head + relax `target_layer_ids_n==3`; also gemma-MoE untested.
- [x] **Corpus-static n-gram (−lcs)** (L13) — build chunk-and-merge bigram cache from a few-GB code slice, vocab-locked; then A/B.
  ✅ **CLOSED 2026-07-29 (read-certification, `auditor`) — retired by the operator, and the input no longer exists.**
  `handoffs/completed/corpus-augmented-prompt-lookup-revalidation.md:179-187`: CPL-4b *"Retired by operator decision after CPL-4
  … corpus has been deleted"*, and CPL-5 records the operator *"declined the optional static n-gram experiment before reclaiming
  disk"* (651 GB). Same mechanism (`llama-lookup-create -lcs`, chunk-and-merge). This row cannot be run as written.
- [x] **Sub-4-bit capacity/throughput** (L15) — only after a CDNA2 sub-4-bit dequant kernel exists; PPL/eval-parity gate.
  ✅ **CLOSED 2026-07-29 (read-certification, `auditor`) — both stated gates were met on 2026-07-05.** The precondition was already
  satisfied (IQ1/IQ2/IQ3/Q2_K/Q3_K are MMQ-supported, `MUL_MAT_ID 789/789`). Capacity/throughput:
  `progress/2026-07/2026-07-05-mi210-capability-kernel-rnd.md:23-24` — 122B UD-IQ2_M fully GPU-resident, 47/64 GB, 43.7 t/s single /
  148.7 agg @B32, PPL 5.02. Eval-parity gate: `progress/2026-07/2026-07-05-mi210-residency-and-cot-reframe.md:12` — 212-question
  deterministic **paired** eval, judge-free, IQ2 163/212 = Q4 163/212, Δ0.0pp, McNemar p=1.000.
- [ ] **GPU-draft / CPU-target spec** (L18) — for overflow-to-CPU MoE (122B, GLM-5.2 IQ2) and CPU-target GDN roles.

**[H] = needs a build, then measure.**
- [ ] **quantize_q8_1 requant kill** (L2) — low effort, do early.
- [x] **Fused dequant-in-GEMV** (L3) ✅ 2026-07-14 — RESOLVED NON-TASK (a8afd338 reconciliation): Q8_0 GEMV is already int8-native (`vec_dot_q8_0_q8_1` = dp4a + one fp scale/32-block), no per-element dequant to hide; the 47→62% gap is BW/occupancy, not dequant-compute.
- [x] **Async weight prefetch / LDS double-buffer** (L4) ✅ 2026-07-14 — MEASURED + landed: nwarps 2→4 +4.6% (`5dc116130`), `raw.buffer.load.lds` LDS double-buffer +3.3% (`7c28056b7`, MemUnitStalled −62%). +3.3% is the CDNA2 LDS-prefetch ceiling; fused-FFN extension FALSIFIED.
- [ ] **Weight swizzle / SoA Q8_0 repack** (L6) — gate on `TCC_EA_RDREQ_32B`; do jointly with L3.
- [x] **Persistent / megakernel decode** (L5) ✅ 2026-07-14 — RESOLVED NEGATIVE (2026-07-04 Pass-2): NOT worth building. HIP graphs (kill ALL host-launch) buy only +5.9%; decode is memory-latency-bound at ~50% roofline, so the 62→100% gap is the batch-1 MLP floor, not a launch bubble. Single-stream dense-Q8 CLOSED.
- [ ] **MFMA compute-bound kernels** (L7) — prefill / DiT denoise / ViT encoder / high-batch expert GEMM; profile-gated (high VALUBusy + low MfmaUtil).
- [x] **GDN occupancy + recurrent-state traffic/layout** (L20) ✅ 2026-07-14 — bf16-state BUILT + GO (`496e2f098`): +21.5%@B32 (drift+isolation PASS), deployed on frontdoor + architect. Occupancy sub-lever was NO-GO; bf16-state is the win.
- [ ] **Q4_K dequant efficiency** (L21) — ~+43% → ~47 t/s single-stream, quality-gated.
- [x] **mmid MoE dispatch threshold** (L1-MoE) — the deferred production-relevant MoE analog; most valuable at 256-expert sparsity.
  ✅ **CLOSED 2026-07-29 (read-certification, `auditor`) — built, run, and falsified.**
  `progress/2026-07/2026-07-03-mi210-qwen36-27b-speed-campaign.md:131`: *"L1-MoE mmid-dispatch lever = MEASURED NEGATIVE (do not
  build)"* — forcing experts to MMQ inverts the dense fix (B2 −30%, B4 −21%, B8 −10.5%; controls B≥16 ±0.4%), and
  `get_mmvq_mmid_max_batch_cdna` default=8 is already optimal. This document's own line 160 records the same falsification.
  **Residual moved, not lost:** the *256-expert sparsity* clause was never measured and belongs to the ultra-sparse MoE row below.
- [ ] **Diffusion serving path + all DiT levers** (§3.5) — no path exists yet.
- [ ] **Auxiliary baselines (BGE encode, VL prefill)** (§3.6) — no aux model ever benchmarked; the entire Aux column is currently deductive.

**Taxonomy gaps (categories the 6-bucket matrix does not yet isolate — flag before trusting a transferred verdict):**
- [ ] **Dense-Q4_K / K-quant** — the registry **default** quant (Q4_K 291 hits), a distinct 33%-roofline dequant regime where L1 barely applies (+5.8% vs +17–32% Q8). Treated as a lever, is really a category.
- [ ] **Pure-MoE vs GDN-hybrid-MoE split** inside "MoE-on-GPU-decode" — the campaign's own sharpest distinction (spec-dec viable + 5.9–8.6x scaling for pure-MoE vs net-negative + 3.4x for GDN-hybrid-MoE).
- [ ] **Sub-4-bit / IQ codebook quant** — architecturally different dequant kernel from K-quant block dequant; the bytes/token axis deciding 64 GB fit.
- [ ] **MLA-attention MoE** (GLM-4.7-Flash frontdoor candidate, deepseek2) — latent-compressed KV **flips the L14 KV-quant math**; spec-dec-forbidden.
- [ ] **Mamba2-hybrid** (Nemotron-Cascade-2 frontdoor candidate) — float32 SSM state (~2× traffic), 88% recurrent; must **not** inherit delta-net GDN verdicts (`--mamba-ssm-dtype` is its own recurrent-traffic knob).
- [ ] **Lightning / linear-attention hybrid** (Ring, bailingmoe_linear) — a third distinct linear kernel, neither delta-net nor Mamba2.
- [ ] **VL split** (ViT vision-**encoder** prefill = compute-bound MFMA target; VL decoder = full MoE/dense decode; spec-dec mmproj-forbidden) — currently hidden inside "Auxiliary(vision)."
  - [ ] **VL vision-encoder prefill baseline — the unrun half of A3.** 2026-08-12 ran the BGE encode half only (13 GPU points, `artifacts/gpu-aux-baselines/a3_bge_mi210_20260812.jsonl`, commit `182f30b6`): bge-m3 Q8_0 and f16 land within noise at every sequence point, so **the quant lever is inert on encode** and buys memory, not speed. The VL prefill half was never run and is the compute-bound MFMA regime where L16's bf16-vs-Q8 crossover is actually testable — the row's own ⚙︎corrected note says VL prefill, not the decode tail, is the correct auxiliary analog. Needs a GPU window; nothing blocks it.
- [ ] **Ultra-sparse MoE** (MiniMax-M2, 256-of-8) — where expert-gather/scatter and the mmid dispatch behave very differently than at A3B/A4B.
- [ ] **ASR / speech encoder-decoder** (Whisper large-v3-turbo) — conv frontend + bidirectional encoder + cross-attention decoder fits none of the 6 buckets.

---

## 5. Headline

**Highest-leverage moves across all categories (skeptical, ROI-first):**
1. **N-gram / prompt-lookup on GPU (L12) — the cheapest un-run lever.** Zero kernel, runtime flag already in v6, "do-first," and its verify batch **synergizes with the landed MMVQ→MMQ fix**. Regime: positive on code/JSON/structured for dense & dense-FFN-GDN; ~0 on prose (don't enable there); headwind for MoE-on-GPU (probe only). Un-run on the 27B despite being flagged do-first.
2. **Fused dequant-in-GEMV (L3) — the main-event dense kernel.** Closes the **measured** Q8 47→62% roofline gap (**~+19%**, corrected down from the +32% kernel-bound), the single biggest single-stream dense lever; share the SoA repack (L6) once.
3. **Async weight prefetch / LDS double-buffer (L4) — the batch-1 latency-wall lever.** 62→~70% achieved BW; **purest for fp16/bf16** (no dequant gap to close first). The 2026-07-04 update flags it as dominating megakernel on effort-adjusted return since HIP graphs are already on.
4. **bf16-for-aggregate / Q8-for-single-stream selection (L16) — zero-code, already measured.** 744 vs 561 t/s @B=32 (bf16 batched) and 96.6 vs 73.1 single-stream (Q8) — a deploy-time role→precision routing win, HBM-fit-gated.
5. **mmid MoE dispatch analog (L1-MoE) — the deferred production-relevant MoE fix.** The banked dense MMVQ→MMQ win never touches experts; the mmid threshold is the untested MoE lever, most valuable at ultra-sparse (256-expert) MoE.

**The single most important circumstance-flip to remember:**
**Self-spec (MTP/EAGLE/tree) sign is set jointly by ARCH × SUBSTRATE — never carry a spec-dec verdict across the dense↔MoE or GPU↔CPU boundary.** MTP is **+15.6% on GPU-dense/BW-bound**, **−12% on MoE-on-GPU decode** (measured **head-quant-independent** — it is MoE-verify overhead on already-fast plain MoE decode, not head dtype), and **flips back to a win on CPU-MoE**. GDN-hybrid resolves the same flip *within a single arch family* by FFN sub-variant (dense-FFN win / MoE-FFN lose). Every "spec-dec is dead" statement is only true with its `{arch, substrate, batch}` tag attached.