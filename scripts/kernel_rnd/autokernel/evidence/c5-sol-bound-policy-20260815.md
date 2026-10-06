# Agentic ROCm Kernel Authoring — MI210 Verify+Profile Harness

**Status**: active investigation — hardware present; P-GPU-1 ratified. **Corrected 2026-08-10 (operator): P-GPU-1 governs the CLASS OF CLAIM a result may carry, not permission to run — the human boundary is freeze / cutover / promotion.** Benching or profiling a *live server* is still owned by whoever owns that inference. Every "operator-approved GPU runs" phrase below predates this correction; read it as claim-class, not permission.
**Next action (2026-08-12)**: preserve immutable r19 while it runs, then validate its terminal receipt
chain before any comparison. Never resume r4-r18 or aggregate a partial attempt into the panel.
The full 8/8 panel continues to refuse only on the unavailable exact ARGUS source release.
**Created**: 2026-06-03 (via /research-intake deep-dive of the LLM-kernel-generation cluster) · **Updated**: 2026-08-12 (r19 live after copy-time staged-input binding)
**Categories**: hardware_optimization, agent_architecture, autonomous_research, tool_implementation, training_distillation
**Hardware gate — SATISFIED 2026-07-02**: AMD MI210 Instinct (CDNA2 / gfx90a, 64 GB) is racked and the llama.cpp HIP build is verified on gfx90a (`progress/2026-07/2026-07-02-mi210.md`; memory `project_mi210_gpu_inference`). This program is now **ACTIVE**, priority **MEDIUM** — it is an *optimization*, **not a production blocker**: llama.cpp-HIP already serves ~910 tok/s @32-way as-is (2026-07-02 obs). First step = reproduce **GEAK-eval** (intake-674, arXiv 2507.23194) on gfx90a — compile+correctness+timing round-trip — as the sanity gate. **Scoping caveat (adversarially verified 2026-07-03; AMENDED 2026-08-03, see §"GEAK scoping — amended")**: GEAK **v4** retains first-class gfx90a knowledge, though all published *evaluation* is gfx942; **AgentKernelArena (679) / robust-kbench (668) are gfx942/CDNA3-listed** and must be treated as ports, not drop-in reproductions. All GPU runs remain operator-approved measurements per MEASUREMENT.md (write P-GPU-1 first). [was: "expected ~July 2026; nothing executes until the card racks" — stale after 2026-07-02 install] [was: "close the measured quantized-MMQ-dequant roofline gap: ~33% Q4_K / ~47% Q8 at batch-1" — **re-targeted 2026-08-03**, that is half the prize; see §"Program re-target"]
**Priority**: MEDIUM (activates on MI210; prep proceeds now)
**Workstream**: Inference Acceleration / GPU · **Parent index**: [`inference-research-index.md`](inference-research-index.md)
**Full reasoning + evidence**: [`research/deep-dives/agentic-rocm-kernel-authoring-geak-synthesis.md`](../../research/deep-dives/agentic-rocm-kernel-authoring-geak-synthesis.md) ← the durable narrative; this handoff is the operational summary.
**Related**:
- [`rocm-verify-profile-backend.md`](rocm-verify-profile-backend.md) — **child**: the ROCm verify/profile/benchmark backend this loop drives
- [`gpu-drafter-mi200-investigation.md`](gpu-drafter-mi200-investigation.md) — MI210-gated; consumes the kernels this loop produces
- [`gpu-acceleration-path.md`](gpu-acceleration-path.md) — the ROCm kernel-library hand-port path this automates
- [`llama-cpp-dsa-contribution.md`](llama-cpp-dsa-contribution.md) / [`cpu-shape-specialized-gemv-decode.md`](cpu-shape-specialized-gemv-decode.md) — the hand-HIP endgame + the CPU ukernel loop this rhymes with

---

## Objective
Stand up an **agentic, train-free kernel-authoring loop for the installed MI210** — drive a strong coding agent through generate → compile → verify → profile → refine to produce and tune **HIP/Triton kernels** for the EPYC stack, replacing the manual hipify-and-hand-tune path. **We cannot retrain a kernel model on one MI210, but we can run a train-free verify+profile loop** — and AMD has already open-sourced most of the substrate (GEAK/Apex/AgentKernelArena), demonstrated on gfx90a.

## Current Decision Snapshot (2026-06-03)
The path *today* (supersedes any earlier "EvoEngineer/CudaForge-first" framing):
1. **Backend = adopt AMD-native code.** **GEAK-eval (674, MIT)** is the primary benchmark/oracle/timing substrate (C1/C2/C3/C5), reproduced on our gfx90a; **Apex (675, MIT)** supplies the E2E deploy harness + Magpie scorer; **AgentKernelArena (679, Apache-2.0)** supplies a second substrate + the controller-A/B shell. Net-new to us: **C4 + C6** (see child handoff).
2. **Controller-A/B = register adapters, don't build a harness.** AgentKernelArena (679) already ships Claude Code / Codex / Cursor / GEAK adapters with a `@register_agent` pattern — **register our controllers (Claude+Codex actor-critic, EvoEngineer, KernelFoundry, K-Search, Xe-Forge, GEAK) as adapters and A/B them on gfx90a.** It compares whole agents at task level, complementing each controller's inner loop.
3. **Agent backend = Claude+Codex actor-critic** (reuse the autopilot planner's infra); local coder role is the self-hosted fallback. `opensource_only` governs deployed services, not build-time tooling — the authored kernel is the artifact, not the LLM. Empirically favored: CudaForge's best result was a cross-model coder/judge split; AgentKernelArena's best results are Claude Code / Cursor / Codex.
4. **Triton first (on-ramp), HIP second (endgame).** GEAK-eval (Triton, gfx90a-proven) → then the HIP arm via GEAK-HIP patterns (678) + AgentKernelArena's Torch2HIP suite (679) + our own HIP oracle. Pairs with `llama-cpp-dsa-contribution.md`.
5. **Differentiators we own: C6 (anti-reward-hacking) + C4 (gfx90a profiler-metric).** Both now have an AutoKernel implementation. Paired `rocprofv2` is the op-level Q4_K/Q8_0 path; direct timestamp-only `rocprof` v1 is the governed whole-model fallback and completed K28 attribution. IQ2_XXS still requires the seeded Omniperf fallback after OP-11 gives its producer a durable identity.

**Why this is the decision (one paragraph):** the entire cluster was NVIDIA-bound at the toolchain, so a from-scratch ROCm backend looked like the long pole — until GEAK/Apex/AgentKernelArena turned out to be AMD-native, permissively licensed, and (for GEAK-v1) demonstrated on **gfx90a, the MI210's exact ISA family**. That predicts *compile compatibility* on our card (not performance — single-GCD bandwidth, ROCm version, autotune space, and harness details still need reproduction), which shrinks the program to "adopt + reproduce + add C4/C6." Full reasoning, alternatives, and the rejected paths: see the [deep dive](../../research/deep-dives/agentic-rocm-kernel-authoring-geak-synthesis.md).

## Evidence — grouped by role (intake-660…679; details in the deep dive + `intake_index.yaml`)
| Role | Entries | Use |
|------|---------|-----|
| **AMD-native substrate (adopt)** | GEAK 674 (gfx90a-proven, MIT), Apex 675 (MIT), AgentKernelArena 679 (Apache-2.0) | C1/C2/C3/C5 backend + Magpie scorer + controller-A/B arena. **674 carries the only gfx90a *evaluation*; 675/679 are gfx942-listed/eval'd. But GEAK v4 carries first-class gfx90a *knowledge* — see §"GEAK scoping — amended".** |
| **AMD-native, patterns-only (gfx942-only)** | GEAK-v2 677, GEAK-HIP 678 | 677 → C4 Profiler-Analyzer (try first) + QD upgrades; 678 → HIP-arm loop (out-optimized a human engineer) |
| **Controller candidates** | EvoEngineer 666 (lead), KernelFoundry 669 (hw-awareness layer), Xe-Forge 672 (linear archetype), K-Search 673 (world-model tree, MoE-strong), GEAK 674 (first to stand up) | register as AgentKernelArena adapters; A/B on gfx90a |
| **C4 (profiler-metric) sources** | GEAK-v2 677 (raw rocprof→LLM, try first), Xe-Forge 672 (static KB), CudaForge 662 (formal selection) | the standing research risk |
| **C6 (anti-hacking) sources** | robust-kbench 668 (exploit classes, Apache-2.0), AgentKernelArena 679 (unseen-shape generalization) | the two complementary halves; our differentiator |
| **Eval philosophy** | KernelBench 664 (`fast_p`), FastKernels 671 (vendor baseline + whole-model gate) | weight end-to-end over isolated-op |
| **RL lessons (no training)** | CUDA Agent 660, CUDA-L1 661, Kevin 663 | reward design + anti-hack gates + multi-turn |
| **Optional offline later** | ConCuR 665, TritonForge 676 | SFT data-curation for a local HIP-specialized small model |
| **Not our path** | KernelCraft 670 (AIE-ML NPU/Peano, not ROCm) | harvested only: tool vocabulary + thinking-budget + ICL findings |

## gfx90a caveat (applies to every AMD number above)
Same `gfx90a` ISA **predicts compile compatibility, not performance equivalence.** GEAK-v1's MI250X results should *build and run* on the MI210 (wavefront=64/MFMA/LDS identical), but single-GCD bandwidth, ROCm version, autotune space, and harness details require reproduction. GEAK-v2 / GEAK-HIP / AgentKernelArena publish **gfx942/CDNA3 numbers only**, so their *numbers* carry even less. **All AMD numbers are vendor-reported until reproduced on our own gfx90a.** [was: "a coverage regression vs v1" — **retired 2026-08-03**: for GEAK proper this is unpublished coverage, not removed coverage; see below.]

## GEAK scoping — amended 2026-08-03 (research-intake Stage-2b; this IS the mandated freshness sweep)

_The prior caveat **understated** the dependency. The README's narrowing is a deployment/evaluation
statement, not a capability removal._

- README (verbatim, confirmed): *"GEAK targets AMD Instinct MI GPUs (CDNA, e.g. gfx942 / gfx950; the
  on-box card is auto-detected)"*. **All published evaluation is gfx942 — there are no gfx90a numbers.**
- **But the tree says otherwise.** `perf_knowledge/hardware/cdna2_mi200/` ships **four files**
  (`arch.md`, `matrix_core.md`, `memory.md`, `occupancy.md`), all `gens: [gfx90a]`, all
  `updated: 2026-06-08`, titled *"CDNA2 / MI250X / MI210 (gfx90a)"* — **our exact card, named**.
  `perf_knowledge/index/capability_index.yaml` carries **40 `gens:` entries including gfx90a**
  (vs 214 gfx942 / 225 gfx950 / 20 gfx908). `perf_knowledge/README.md` scopes CDNA1 → CDNA4.
- **No arch gating found** — the card is auto-detected; there is no allowlist rejecting gfx90a.
- **Correct formulation to carry forward:** *GEAK v4 carries first-class gfx90a hardware knowledge and 40
  gfx90a-applicable capability entries, but publishes zero gfx90a numbers.* **Pin `v1.0.0 @ 4ffba15a`**
  (Apache-2.0, still pinnable) for the paper's MI250X evidence. v1→v4 spans eight tags, 2025-08-01 →
  2026-07-22; repo active (pushed 2026-08-03).
- Caveat on the v1 evidence itself: **v1's release notes name only MI300X** — the gfx90a claim traces to
  the v1 **paper** (arXiv 2507.23194), not the release.
- This cuts *with* the grain of GEAK's own consumption contract, which states the KB is
  *"reference material… not decisions"* and that consumers must *"decide by on-box measurement"* —
  i.e. AMD is telling us to measure on our own card, which is exactly the standing unchecked sanity gate.
- **The KB is not error-free** — its `memory.md` ridge-point figure is off by 2× (per-GCD vs per-OAM); see
  [`mi210-mfma-compute-bound-paths.md`](../completed/mi210-mfma-compute-bound-paths.md).
- **Bonus, unexploited:** GEAK ships `languages/` docs on **hipkittens · tilelang · mojo · cutlass ·
  flydsl** and a `landscape/` section on multi-backend libs, DSLs, AI kernel agents, autotuning and
  "AMD SOTA 2026" — a **vendor-maintained survey of the exact taxonomy our Stage-2 dive built by hand**,
  available as an external check.

## Program re-target — aim at the fp16 rung, not the Q8 rung (2026-08-03)

The old objective closed *"~33% Q4_K / ~47% Q8"*. **That is half the prize.** The fp16 rung — **62.6%
attainment — is demonstrated on our own device**, and vLLM-ROCm reaches **69.2%** on the same silicon;
DGX Spark GB10 reaches **77–80% at Q4_K_M dense across five models** on the same engine. The full ladder
and its calibration caveat live in
[`mi210-q8-dequant-gemv-roofline.md`](mi210-q8-dequant-gemv-roofline.md).

**Banded ceiling, for sizing campaigns** (bands, not point estimates; confidence stated):

| Lever | Band | Confidence |
|---|---|---|
| **K1 Q4_K → Q8 rung** | **+38–43%** | HIGH |
| **K5 batched elementwise/norm fusion** | **0 — CLOSED_NO_GO** | CERTAIN for frozen-v9 target selection: strict B=64/128 norm + activation + elementwise share is 1.837% / 1.490%, far below the predeclared 20% floor; the old 43% bucket conflated gather/recurrent/copy work |
| K2 Q4_K → fp16 rung | +60–80% | MEDIUM |
| K3 MoE expert-gather | ~2.0× | MEDIUM |
| K4 architect IQ2 | +2–3× | MED-LOW — attach the gfx906 kill-criterion first |
| K6 fp16 batch-1 | +11% | HIGH |
| K7 HIP graphs | +5.9% | banked already |
| K8 LDS prefetch | ~0 remaining | CDNA2 ceiling |
| **K9 MFMA decode kernels** | **0 — DO NOT BUILD** | CERTAIN (arithmetic, not counters) |
| K10 prefill | +20–30% | MEDIUM |
| **K11 closing the vLLM gap** | **not a kernel program** | see AK-D36 in `autokernel-research-loop.md` §17 |
| **K12 matching Blackwell prefill** | **unreachable** | 5.6× int8 silicon deficit |

**Prefill is mid-pack, not an AMD software problem.** MI210 converts 19–29% of matrix peak; A100 22.8%,
RTX PRO 6000 22–44%, H100 15.3%, MI300X 12.3%. **Prefill kernel *quality* is not the gap; prefill
*silicon* is.**

## New index-backed leads — 2026-08-03 (research-intake Stage-2b)

- **ARGUS (arXiv `2604.18616`) — register as a controller candidate.** Agentic GPU kernel optimisation
  reaching **99–104% of hand-optimised assembly on AMD MI300X** for GEMM / FlashAttention / MoE;
  2–1543× over prior agentic systems; 100% KernelBench L1, 90% L2. **Absent from all six MI210/autokernel
  handoffs.** Directly on-point for the controller A/B — and, being MI300X, it is CDNA3 evidence that
  still tells us what an agentic loop achieves against *AMD's own* hand-tuned assembly.
- **HipKittens fragment-layout identity — a free compositional result.** HK's `rt_base` is **bit-identical**
  to `ggml/src/ggml-cuda/mma.cuh`'s `tile<16,16>` in our frozen v8 (`:127,144` — `get_i = tid%16`,
  `get_j = 4*(tid/16)+l`, `ne=4`). Every HK technique composes onto our existing fragments with **zero
  layout re-derivation**. The arch-independent lessons are filed in
  [`mi210-mfma-compute-bound-paths.md`](../completed/mi210-mfma-compute-bound-paths.md); do **not** vendor the framework.
- **A live gfx90a build arm already exists** on HK's `cdna3` branch (`GPU_TARGET=CDNA2` →
  `-DKITTENS_CDNA2 --offload-arch=gfx90a`, `tests/unit/Makefile:39-40`). Across all 67 headers the library
  uses exactly **six** `__builtin_amdgcn_*` intrinsics and **exactly one is unavailable on gfx90a**
  (`mfma_f32_16x16x32_fp8_fp8`, `mma.cuh:47` — non-template static inline, so it needs an `#if` guard).
  That arm carries a **~3000-test correctness harness** that would run on our silicon. This is what makes
  the *decline to port* a decision about economics rather than capability.
- **A runnable LDS bank/phase solver** — `analysis/paper_experiments/phases/*/{bank_solver.py,
  phase_solver.py, kernel.cpp}`, a 45-line kernel over rocprofv3 PMC counters, ~40 min GPU on gfx90a.
  **Do not assume the CDNA3 answer (64 banks / 2 phases of 32 lanes) transfers** — whether gfx90a is 32 or
  64 banks decides whether HK's `>>7 <<3` swizzle constants transfer at all. Blocked on profiler tooling
  (below). ~~Still blocked as of 2026-08-12 (mainB)~~ — **RESOLVED, and this line is the mis-stamp.**
  The solver **ran on 2026-08-11** via a rocprofv2-compatible counter path (receipt SHA-256
  `ae1d833c…`, 372 bank / 6,048 phase dispatches, 32 banks + 8 phase cliques). `rocprofv3` really is
  absent — that half of the 2026-08-12 check stands — but it was never required: **this line names an
  instrument the run did not use.** Nothing was blocked. Corrected 2026-08-12 (mainB).

- **`rocm-flash-attn` as an enabling path**, re-assessed: an adaptation layer with no kernel code of its
  own, but genuine tested code with honest defect annotations. Judged on whether it improves performance,
  not on whether it contains kernels.
- **Two operational levers from AMD's own ROCm llama.cpp blog:** hipBLASLt grouped-GEMM plus tuning
  (**+29%**) and ~10× fewer `hipMemcpyAsync` calls.
- **llama.cpp issue #19984 — an LLVM loop-unroll regression in ROCm 7+ costing 3.7–5× on prefill**,
  workaround `-mllvm --amdgpu-unroll-threshold-local=600`. We are on ROCm 6.2 so this does not bite today;
  it belongs on the build-flag checklist **before any ROCm upgrade**.

**Profiler tooling — RESOLVED FOR THE CURRENT C4 AND LDS SURFACES.** `rocprofv2`, `rocprof` and
`rocm-bandwidth-test` are available, version-matched to ROCm 6.2.0-66, side-loaded by extraction
rather than installed so nothing in the shared `/opt/rocm` bind mount changed:
`source /mnt/raid0/llm/tools/rocm-profilers-6.2/env.sh`. **The gfx90a counter taxonomy is proven** —
465 counters across 12 blocks, enumerated on our own card, including every counter this program
already cites. Details, per-block collection limits, and the two path quirks:
[`rocm-verify-profile-backend.md`](rocm-verify-profile-backend.md). This unblocks C4's existing
`rocprofv2` path. The LDS solver was also successfully adapted to rocprofv2; its r4 receipt binds the
exact commands, tool and binary hashes, 372 bank dispatches and 6,048 phase dispatches. `rocprofv3`
remains absent but is not a blocker for that completed surface. The governed IQ2 Omniperf fallback
remains separately gated by the OP-11 producer identity.

- [x] Resolve the side-loaded ROCm 6.2 profiler-tool availability for C4. ✅ 2026-08-12 — live
  version checks confirm `rocprof`/`rocprofv2` at ROCProfiler 2.0 and a runnable argument parser for
  `rocm-bandwidth-test`. These tools are intentionally absent from the default PATH and `/opt/rocm`.
- [x] **Establish whether the LDS bank/phase counters are reachable through `rocprofv2`.** ✅
  2026-08-12 — reconciled against the already-terminal r4 empirical receipt: rocprofv2 captured 372
  bank and 6,048 phase dispatches and the solver derived 32 banks plus eight phase cliques. Receipt
  SHA-256 `ae1d833c704bdae9a78767d0fc0b927298d6d1dfdb31a0ea11c34058dc525987`.

## Open questions (decided ones live in the deep dive §5)
- Which controller wins on gfx90a? Unknown until the AgentKernelArena A/B runs on the MI210 with EPYC ops.
- Does GEAK-eval's published MI250X speedup reproduce on the single-GCD MI210? The substrate-level
  compile/correctness/timing round-trip now passes on physical gfx90a; the published task suite and
  matched controller-performance comparison remain separate empirical questions.
- Does C4's cheapest path give a usable signal on CDNA2? **Answered 2026-08-11: yes at op level,
  not as a whole-model `rocprofv2` capture on this host.** The deterministic report resolves the
  Q4_K/Q8_0 fill → requantize → matvec sequence and emits 1%-floor wall shares. Whole-model Qwen
  prefill and IQ2_XXS op captures reproducibly exit 139 inside the profiler; these are tool-scope
  limits, not zero-work readings. AutoKernel consumes only the deterministic report, never raw
  profiler text.
- Will AMD publish gfx90a numbers / arXiv papers for GEAK-v2 / GEAK-HIP / AgentKernelArena? → Freshness Appendix in the deep dive.

## Reporting / maintenance instructions
- After any work: update the **Current Decision Snapshot** here + the deep dive; log progress in `progress/YYYY-MM/`.
- **At every audit of this handoff, run the GEAK-family freshness sweep** in the deep dive §9 (GEAK repo pin/tag drift; missing gfx90a evidence for 677/678/679; AgentKernelArena leaderboard + GEAK-vs-general A/B; new AMD-native siblings on `AMD-AIG-AIMA`/`AMD-AGI`).
- **Done this session:** GEAK repo state recorded (HEAD `c8bfc19`, tags →`v4.8.3.3`, branches GEAK-v2/GEAK-HIP); AgentKernelArena ingested (intake-679). **Next intake candidates** if they appear: a GEAK-v2 arXiv, a GEAK-HIP open benchmark, the AgentKernelArena leaderboard.

## Research Intake Update — 2026-07-08: KernelBench (rec-007)

**Source**: KernelBench (**intake-664**, arXiv:2502.10517)

> **⚠ CORRECTION 2026-08-10 — this attribution is wrong, and was already corrected elsewhere.**
> The identical line in [`mi210-speed-campaign-summary.md`](../completed/mi210-speed-campaign-summary.md)
> carries a verified 2026-07-22 correction that never propagated here: this is a **three-way
> conflation**. The real **KernelBench** is Stanford ScalingIntelligence **arXiv:2502.10517**
> (kernel *generation*, metric `fast_p`), confirmed via intake-660/661. `arXiv:2606.20128` is a
> separate seeded-fuzzing paper, and `intake-797` was never either of them — it was
> "Externalization in LLM Agents" (now merged into intake-418). The 9/9 seeded-fuzzing finding
> below belongs to the 2606.20128 paper. Surfaced while auditing references to merged intake ids.

**Corrected disposition**: the seeded-fuzzing result belongs to a separate paper and maps to **C2
(correctness)**, not C3 (timing/reward). Its useful design rules—stateful-op coverage, adversarial and
non-power-of-two extents, absolute-duration gates, speed-of-light rejection, and anti-reward-hacking
checks—are already implemented natively in AutoKernel's C2/C6 surfaces. KernelBench itself is an
isolated PyTorch operator suite, not a baseline corpus for current llama.cpp HIP kernels.

- [x] **AK-RB-1** — adopt RE-Bench's SCORING PROTOCOL for gfx90a kernel-agent evaluation, not the
      benchmark itself (intake-1072, filed 2026-08-10). Transferable: log-time scoring of
      behaviour-preserving optimization, 0 = starting state / 1 = a strong reference solution, and
      time-budget curves (2h/8h/32h) rather than pass-fail. NOT worth standing up as-is — only 1 of 7
      environments is a kernel task, it is Triton on H100, and porting to gfx90a invalidates the
      published human and model anchors that are the reason to use it. ✅ 2026-08-11 — implemented
      reference-normalized, deliberately unclipped log-time scores; behavior failures withhold reward;
      matched best-so-far curves emit at 2h/8h/32h in
      `scripts/kernel_rnd/autokernel/evaluator/rebench_scoring.py`.
- [x] **AK-KB-1** — audit KernelBench for the GEAK-eval **C2** correctness surface ✅ 2026-08-11 —
      declined the incompatible task artifacts (current first-party HIP is gfx942/gfx950 and ROCm 7.1+),
      while retaining the already-native correctness and anti-hacking design rules above. The live
      gfx90a GEAK/AgentKernelArena round-trip is the compatible C2/C5 substrate.
- [x] **AK-KB-2** — establish the correct baseline disposition for current llama.cpp HIP kernels
      ✅ 2026-08-11 — retired the category error: exact-surface llama.cpp baselines and historical C5
      replay are authoritative; translating them into isolated PyTorch KernelBench tasks would change
      the unit under test and cannot establish a production-kernel baseline.

## Progress checklist

- [x] Reproduce the GEAK/AgentKernelArena compile+correctness+timing round-trip on gfx90a MI210
  (first sanity gate) ✅ 2026-08-11 — the exact-pinned adapter refused gfx90a spoofing, compiled the
  live add-kernel arm under Torch 2.5.1+ROCm 6.2 / Triton 3.1.0, passed correctness 3/3 and timing
  harness 5/5 on the physical MI210, and released its device claim. Receipts:
  `/mnt/raid0/llm/autokernel/probes/inf03-geak-arena-gfx90a-preflight-20260811/receipt.json`
  (SHA-256 `256a4c60a416828a1299a35d8399609c3c5ad2541272ab41de40dd00f2f297a4`) and
  `/mnt/raid0/llm/autokernel/probes/inf03-geak-arena-add-roundtrip-20260811/receipt.json`
  (SHA-256 `aee866ee3ebd2fe88b37185f4226c3c20a5b439c60a40fac57ef1bb42898be8c`). This closes compatibility,
  not the published-suite speedup reproduction.
- [x] Write P-GPU-1 measurement protocol before any GPU runs ✅ 2026-07-29 — human amendment ratified the canonical MI210 GPU protocol in [`MEASUREMENT.md`](../../MEASUREMENT.md#p-gpu-1--mi210-gpu-canonical-throughput-ratified-2026-07-19) on 2026-07-19; this closes protocol authoring only, not any GPU run or decision claim.
- [ ] Run the matched controller-authoring A/B on gfx90a across the registered Claude+Codex,
  EvoEngineer, KernelFoundry, K-Search, Xe-Forge, GEAK, ARGUS, and baseline arms. Registration,
  C4 prompt-hygiene binding, and the three-argument `@register_agent` bridge are complete in
  research commit `48350b24`; the full-panel implementation now refuses honestly at 6/8, and the
  matched eight-arm comparison waits only on the two unavailable licensed source releases.
  - [x] **Build the governed eight-arm campaign driver and audit executable coverage.** ✅ 2026-08-11
    — baseline plus Claude/Codex actor-critic, EvoEngineer, KernelFoundry, K-Search, Xe-Forge,
    GEAK-v1, and ARGUS are serialized on one MI210 at exact 2h/8h/32h checkpoints with source,
    entrypoint, executable, model, configuration, and driver hashes. The physical audit correctly
    refused before launch because only **1/8** arms was executable at the initial audit; baseline is
    explicitly a zero-hour non-authoring control. Audit receipt:
    `/mnt/raid0/llm/autokernel/probes/inf03-controller-ab-audit-20260811/receipt.json`, SHA-256
    `3152e2fa97b52d9f0b91fb43449375adec66128189fddf253b0772fadfcf59c4`; research
    `429b59a1` (promoted via `ee54c144`).
  - [x] **Provision the internal Claude-planner/Codex-actor arm.** ✅ 2026-08-11 — the exact
    three-argument Arena callable and stdin executable pin Claude Opus 5/high as planner+critic and
    GPT-5.6-Codex/high as actor, enforce exact 2h/8h/32h checkpoints plus captured process-group
    termination, confine all writes to one candidate inside the Arena workspace, and hash-bind the
    CLIs, prompts, transcripts, candidate, artifacts, and final receipt. The clean physical audit now
    admits exactly **2/8** arms (baseline plus this controller); the other six refusal rows remain
    verbatim. Implementation `dcdc2311`, binding `084b8be1`, research main `9947f805`; 409
    controller tests and 17 AutoKernel README tests pass. Neither CLI was invoked during
    implementation or audit.
  - [x] **Build the governed Arena cell runner.** ✅ 2026-08-11 — research `36ed94a8` and
    `cffd3b02` provide direct belief receipts, process-group teardown, workspace confinement, and
    a no-inference test surface for upstream controllers; the original validation passed 129 tests
    plus eight subtests.
  - [x] **Port and bind K-Search without replacing its search policy.** ✅ 2026-08-11 — research
    `866855a3`, `832f6833`, and `2ceb2878` retain the exact upstream source pin
    `53c8fab9a5e8fab2c86610d24fbec5067f90e115` and put the governed adapter on the fixed campaign path.
  - [x] **Port and bind GEAK-v1 without replacing its evolutionary loop.** ✅ 2026-08-11 —
    research `38b8e07e` and `60cae5bf` retain exact upstream source pin
    `4ffba15a55f250816598b4e27eb56ca40a699cea`, safe cleanup, and direction-preserving receipts on
    the fixed campaign path.
  - [x] **Port, bind, and repair Xe-Forge.** ✅ 2026-08-11 — research `d76fe4b9`, `af5d026a`,
    `1bd5ec75`, and wrap repair `e31cc8c8` retain exact source pin
    `4dcb5080b0f56d0b655ec8c8c9509b8e3ba0382c`; the real-upstream regression proves AMD-only
    prompts, refuses fabricated shapes, and closes the executor cleanly.
  - [x] **Port KernelFoundry and preserve its governed MAP-Elites/QD behavior.** ✅ 2026-08-11 —
    research `7654383c` plus wrap repair `e31cc8c8` retain exact upstream source pin
    `1c053e02383d12937f144923bcc1faa82fa7788f`, activate 159 missing Triton regexes, and prove a real
    inherited 2x2 fixture with two occupied cells, four programs, two QD transitions, and one
    direction without model, GPU, or inference.
    The selected AutoKernel suite passes 538 tests plus 202 subtests at the repaired tip.
  - [x] **Bind KernelFoundry and produce the final no-execution full-panel audit.** ✅ 2026-08-11 —
    research `58c4e332` through `26ad6178` binds all six available arms and emits a clean physical
    gfx90a refusal at **6/8** before any controller or GPU command. Receipt
    `/mnt/raid0/llm/autokernel/probes/inf03-final-audits-20260811-Kpg5wU/full-eight-arm-refusal.json`
    carries receipt SHA-256 `3f67b750c99dccbbe45f5c0043c8aa11973c3e014ab3610bb117786f60a79f7f`
    and file SHA-256 `b432fcb802797136444b510618966489529147aac60d73209b0c1ee946231b1d`.
  - [x] **Add and audit the separately governed available-source panel.** ✅ 2026-08-11 — the same
    pinned task, evaluator, identities, and 2h/8h/32h budgets are ready at **6/6**. Receipt
    `/mnt/raid0/llm/autokernel/probes/inf03-final-audits-20260811-Kpg5wU/available-source-six-arm.json`
    carries receipt SHA-256 `d812b69ce380613bd854dd0d15206c09981899abac11b10234a8f98bb02482b8`
    and file SHA-256 `88101db4f28f909f220acb3bf906f488fe8b4f8e7c307821d325a5542fd4627d`.
    Its authority is availability-conditioned and diagnostic only: it cannot imply an eight-arm
    result, rank partial full-panel evidence, or authorize promotion. No inference ran in the audit.
  - [x] **Run a real KernelFoundry diagnostic smoke and repair the launch boundary it exercised.**
    ✅ 2026-08-11 — v1 found that copied task workspaces lacked the immutable repository import root;
    v2 passed import and two real GPT-5.6 Sol/high calls, then found concurrent branches racing in
    the shared Arena evaluator. Research `f8569112` and `8afd016c` repair both boundaries. V3 passed
    compilation, correctness, and all 4/4 baseline plus 4/4 optimized timing cases under a cleanly
    released MI210 claim. Receipt
    `/mnt/raid0/llm/autokernel/probes/inf03-kernelfoundry-real-smoke-v3-20260811-Mg6Fl9/smoke-receipt.json`
    carries receipt SHA-256 `cd61675e83040b196a92aa85f2c0bd951f34912bef10a37e1b07f41864f52276`
    and file SHA-256 `9b47fefcb2744392923745385053aa6ee9a8a959102a17acb7eaea079a1be5b1`.
    Its `0.9986680991832163` average speedup is diagnostic and non-rankable, not a campaign result.
  - [x] **Run one-iteration diagnostic smokes for K-Search, Xe-Forge, and GEAK-v1.**
    ✅ 2026-08-11 — all three terminal runs passed centralized compilation and correctness plus
    4/4 baseline and 4/4 optimized timing cases, then released their MI210 claims. K-Search receipt
    `/mnt/raid0/llm/autokernel/probes/inf03-k-search-real-smoke-20260811-tQf7zZ/smoke-receipt.json`
    has receipt/file SHA-256 `6eea9028399635083a6aed7a4d0101aa106cc4393e4225dd768c6f27c23e7704` /
    `74f49b472dcb6b2eed1cef66e706e9471ea67f71c94de7a8ba046e3bcd7520b7`; Xe-Forge receipt
    `/mnt/raid0/llm/autokernel/probes/inf03-xe-forge-real-smoke-20260811-NIffwN/smoke-receipt.json`
    has `a53ef172b42d3fbb6008902a865eac9c884d181a6cc0fc0cc981f9e8aad1ccae` /
    `e82917d9f1f751520317700520f33b7bf62dd372749ef18ce3d822ba5b2806ea`; GEAK-v1 receipt
    `/mnt/raid0/llm/autokernel/probes/inf03-geak-v1-real-smoke-v3-20260811-RJ4UYN/smoke-receipt.json`
    has `0deef125d026625055e77a63270c444101fd96ce5f6f9b2fce433e47b509a229` /
    `a4c30029a38011cfc7ca0b59be1eb826463cc336322db57e7d6b2cdea19d7487`.
    Their near-1.0 speedups are non-rankable smoke telemetry, not comparative evidence.
  - [x] **Complete the Claude/Codex actor-critic diagnostic smoke through its confined actor path.**
    ✅ 2026-08-11 — v1–v4 exposed fenced-JSON parsing, contained absolute-path handling, nested
    sandbox, and Docker-stdin defects. Research `84e2f948`, `dd0daedd`, `da677443`, and `22e60940`
    repair them while keeping the actor in a digest-pinned, read-only-root container with one
    writable workspace bind, dropped capabilities, no-new-privileges, exact teardown, and ephemeral
    auth. V5 completed planner, actor, and critic and reached centralized evaluation, but the worker
    inherited `/usr/bin/python3` without pytest. Its apparent correctness failure and zero timing
    cases are therefore an evaluator-runtime defect, not candidate evidence. Receipt
    `/mnt/raid0/llm/autokernel/probes/inf03-actor-critic-real-smoke-v5-20260811-2Oo2cJ/smoke-receipt.json`
    has receipt/file SHA-256 `dcb2da77bf691c670b705149bfaec0bf6e8062594cd4297ac3247037da5937fb` /
    `e890240fa4e3e5134c2975f77930bf5a611e3db285dea2f01a9560d5b699d0d3`.
    The cleanly released 457.02 s MI210 claim retained 1,829 samples. Research `a57feba0` then pinned
    the ROCm evaluator Python/package identity and made mismatch a refusal. V6 passed compilation,
    correctness, and 4/4 baseline plus 4/4 optimized timing cases, reporting diagnostic speedup
    `0.9936027407797491`. Receipt
    `/mnt/raid0/llm/autokernel/probes/inf03-actor-critic-real-smoke-v6-20260811-3hAJir/smoke-receipt.json`
    has receipt/file SHA-256 `5961eef441e487e787310d3bea9d4e57693a8f7a621dff1cc39190d48d952ef9` /
    `816ecc2d60ee48e8b0be3c6fb05ff1d562d7283697f1def9499ce6d0a98a916c`.
    The nested controller receipt has SHA-256
    `fa11ee8162fb6da877358a0c26c67d58d84c75b6c284fe9ee53540bb3673e315`.
    Its cleanly released 158.72 s claim retained 635 samples; the result remains non-rankable.
  - [x] **Refresh the campaign pins to the validated controller trees and repeat both static audits.**
    ✅ 2026-08-11 — clean detached research `6233cd42` binds the actor pin
    `a57feba0`/`e223a9cd26184aff5df7e24ceafebd53dab7cb1ab712a92c4bf82ca7712d9213` and GEAK pin
    `743f59df`/`7418493b4322f1be0df76be21686a6e0d48f43f27e0198a0db55b915f27516a7`.
    The full-panel receipt
    `/mnt/raid0/llm/autokernel/probes/inf03-final-audits-v2-20260811-v6/full-eight-arm-refusal.json`
    has receipt/file SHA-256 `4a13f7d0ba91c2610efae4e51bcf7e0be8661d07657f74fecf9a5ee0a4dab3af` /
    `199e4e129daf561f42d59750a1c2e157da340f433c0fec3abf19cf7c1bd91195` and refuses
    at 6/8 only for external EvoEngineer and ARGUS. The available-source receipt
    `/mnt/raid0/llm/autokernel/probes/inf03-final-audits-v2-20260811-v6/available-source-six-arm.json`
    has `cf8b03df355a8124a1dd668293c1d7d6e839c9f176aebdcd082f073f92ba0581` /
    `625280fb92b678b8a2a24f15f9a87484a2409c0dcb94e32a777e613639f327ed` and is
    ready at 6/6. Neither audit ran a controller or GPU command.
  - [x] **Bind a representative four-task EPYC Arena campaign instead of an add-only proxy.** ✅
    2026-08-12 — research `99fe3014` hash-pins the existing add control plus real
    `attention_llama`, `moe_gemm`, and `dequantize_matmul` task files under the same MI210/gfx90a,
    elapsed-wall-time, 2h/8h/32h, and one-GPU-cell contract. The available six controller arms keep
    their exact implementation/source/license hashes; EvoEngineer and ARGUS remain explicitly
    unavailable rather than receiving namesake substitutes. This is static campaign authority only;
    no controller, model, or GPU command ran.
  - [x] **Re-audit the available-source panel against the current integrated research source.** ✅
    2026-08-12 — the static audit at research `5c8714a1` reports `status=ready`, **6/6** executable
    arms, all four representative tasks ready, a physical `gfx90a` MI210 identity, and the fixed
    elapsed-wall-time 2h/8h/32h budget. It preserves the separate 8-arm refusal for unavailable
    EvoEngineer and ARGUS sources and grants no ranking or promotion authority. Receipt
    `/mnt/raid0/llm/autokernel/probes/inf03-available-source-reaudit-20260812-r1/receipt.json` carries
    self-hash `7c670931628b17a1491d7f25642681f2ee17d996848c0b09032518ad58e08d5a` and file SHA-256
    `e86fbe4a4fb3f45d28d0a9ca7a62be97e021bdf710b5e2f4d7b2287644f0a623`. No controller,
    inference, build, or GPU command ran.
  - [x] **Make the available-source campaign restart-safe before spending the 2h/8h/32h
    budgets.** ✅ 2026-08-12 — research `971e3a00` verifies immutable audit/config/source identity,
    atomically records completed cells and checkpoints, resumes only exact hash-matched terminal
    work, and rejects partial, tampered, in-flight, or claim-leaking state. Focused campaign tests
    passed **139/139**. Resume does not make partial results rankable; the terminal aggregate remains
    the only campaign result.
  - [x] **Repair Arena workspace containment and guaranteed device-claim release before restarting
    INF-03.** ✅ 2026-08-12 — Research `b84c809f` uses collision-bound dot-free cell components,
    rejects dot-bearing ancestor paths, compares exact post-worker cell/sibling manifests on success
    and failure, and defers TERM/INT across claim acquisition so teardown journals the release. The
    focused runner/claim suite passed 68/68 after promotion. The invalid r1 attempt had exposed a
    dotted-path escape: the task's
    `__file__.replace('.', '_')` wrote `test_add_kernel_py.pt` into an adjacent sibling cell tree.
    Reject writes outside the exact workspace after every evaluator/controller step, normalize cell
    paths safely, and release/journal the device claim from a SIGTERM-safe `finally` path. Add
    regression tests for dotted task ids, external writes, and exact-signal cleanup before r2. The
    r2 baseline then completed in the dot-free hash-bound cell without creating an escaped sibling;
    its first controller checkpoint started only after that live containment regression passed.
  - [x] **Enforce the exact planner/critic response schemas at the provider boundary before spending
    another INF-03 checkpoint.** ✅ 2026-08-12 — r2 is immutable and invalid: its baseline completed
    safely, but the first Claude planner returned structurally valid JSON with extra fields that the
    strict four-field parser correctly refused. Research `298bab0c`, pin advances `98f7658f` and
    `6f599ed0`, plus the fresh r3 audit now bind provider-enforced planner and critic schemas without
    weakening the parser. Focused actor/critic tests passed **21/21** and the controller slice passed
    **540/540**. The r3 audit is ready at **6/6** over four tasks with authority still limited to an
    availability-conditioned diagnostic; self-hash
    `b8d966ee7581f2a2fe4ee7e8c2eb7e314514bc4f888f22635dfa23d9ec4ce6dc`.
  - [x] **Stop INF-03 r4 and preserve it only as invalid diagnostic history after auditing its
    intermediate feedback path.** ✅ 2026-08-12 — the captured campaign processes were terminated
    by exact PID and confirmed dead; no MI210 claim remained. The immutable partial root is
    `/mnt/raid0/llm/autokernel/campaigns/inf03-available-source-six-arm-20260812-r4`, bound to manifest
    self-hash `941fece21400362f41772682ec2cf8e3162b08488183a767dce276209ec61eaf`. It retains **5/64
    checkpoints**, **2/24 cells**, **10/10 released** vendor/final windows, 215 samples, and 52.588
    claimed GPU-seconds, but it is not resumable, rankable, aggregatable, or valid controller-comparison
    evidence. KernelFoundry's receipt records 64 intermediate centralized evaluations; the evaluator
    log records 64/64 correctness failures. Those calls ran from the deliberately GPU-blind controller
    environment outside the only two durable claim windows, so the controller received invalid search
    feedback even though its final checkpoint receipt was structurally complete.
  - [x] **Broker every intermediate controller evaluation through the parent worker's exact-PID
    boundary.** ✅ 2026-08-12 — integrated research `70a0d254` + `de7214ee` make the parent own an
    authenticated AF_UNIX broker and every intermediate/final evaluation. The broker verifies the exact
    registered peer PID/start ticks, accepts bounded candidate bytes, and the parent evaluates each
    request in a fresh GPU-capable deny-network subprocess under its own serialized MI210 claim. Neither
    controller nor candidate receives a claim credential. Request/result/baseline, sampler, claim,
    subprocess, activation, and teardown receipts are rehashed during durable restore.
  - [x] **Enforce controller/candidate OS isolation before a decision-bearing INF-03 pilot.** ✅
    2026-08-12 — research `9480a1b3`, `9a17e638`, and `de7214ee` wire every non-baseline Arena cell
    through an exact-read/exec controller sandbox with one inherited peer-bound broker socket, no ROCm
    device access, no campaign-sibling access, cgroup-owned descendants, and durable activation/teardown
    evidence. Candidate evaluation runs separately in a fresh deny-network evaluator sandbox; the parent
    alone owns the claim, sampler, and subprocess. The integrated focused warning-strict suite passed
    **95/95**, the pinned EvoEngineer suite **11/11**, and the pinned controller suite **572/572**.
  - [x] **Exercise the governed one-task/K-Search pilot through its first real Codex request and retain
    every fail-closed attempt.** ✅ 2026-08-12 — research branch
    `integrate/inf03-isolation-merge-20260812` at `569a8c42` added the explicit
    `epyc.autokernel.arena_diagnostic_pilot.v1` authority, a pilot-only one-round K-Search override,
    exact task-source discovery, narrow controller import/runtime/ELF/Git/null-device grants, and a
    workspace-local ephemeral Codex home that is scrubbed during teardown. The immutable probe ladder is:

    | Attempt | First fail-closed boundary | Disposition |
    |---|---|---|
    | r1 | task contract omitted `source_file_path` | fixed by deterministic instruction-task source discovery |
    | r2-r3 | sandbox could not import the controller module at the declared package root | fixed by the exact `scripts/` import root and below-root module identity |
    | r4-r5 | Python extension closure was incomplete (`libffi.so.8`) | fixed by exact extension/ELF dependency admission |
    | r6-r7 | the selected Python environment could not import `yaml` / the pinned Arena evaluator | fixed by binding the governed K-Search Python and exact venv package root |
    | r8 | exact source verification could not start `git` | fixed by admitting the exact executable and ELF closure |
    | r9 | Git could not open `/dev/null` | fixed by admitting only the null device; KFD/render remained denied |
    | r10 | ambient system/global Git configuration paths were denied | fixed by `GIT_CONFIG_NOSYSTEM=1` and `GIT_CONFIG_GLOBAL=/dev/null` |
    | r11 | Codex could not initialize under the read-only host `CODEX_HOME` | fixed with an ephemeral, workspace-local auth/config projection that is scrubbed on exit |
    | r12 | the real Codex client started, attempted the response request, retried five times for about 60 seconds, then failed with `Operation not permitted` while sending to the responses endpoint | fixed by binding the exact sandboxed Codex launcher |
    | r13 | the terminal controller path passed, but a stale controller null-device validator rejected its governed `/dev/null` grant | immutable invalid validator attempt; fixed in research `fe33954e` |
    | r14 | the controller path passed again, but a stale evaluator validator rejected the deny-network evaluator's exact device set | immutable invalid validator attempt; fixed in research `aa2020ab` |
    | r15 | one task / one K-Search round completed through six `gpt-5.6-sol:high` calls, one brokered intermediate evaluation, and the final evaluator | terminal compatibility-only receipt; no campaign/ranking/belief/promotion/release authority |

    R1-r14 remain immutable, non-rankable engineering history. R15 is a terminal compatibility pilot,
    not a campaign result: its final ratio `1.0019039030` and brokered intermediate ratio
    `1.0033912745` are diagnostic telemetry only. The terminal receipt is
    `/mnt/raid0/llm/autokernel/probes/inf03-mi210-isolated-k-search-pilot-20260812-r15/diagnostic-pilot-receipt.json`
    (self-hash `3425cf579a9d6fb06f5ed76b480ae652eaed3b3685eec9116299d253528a8771`,
    file SHA-256 `eb53de389450ba6ef800528a0da6f54c8508eabc39b4357f0cd9beff64aacefa`).
  - [x] **Resolve the r12 sandboxed Codex transport denial with least additional authority, then rerun
    the identical one-task/K-Search diagnostic pilot.** ✅ 2026-08-12 — research main `15c61ed5`
    binds the exact sandboxed Codex launcher (`45b82bbc`) and carries the r13/r14 validator repairs
    (`fe33954e`, `aa2020ab`). R15 completed six model calls, non-null brokered feedback, exact
    controller/evaluator activation, three matched claim/release and sampler windows (11/21/21
    samples), scrubbed ephemeral CLI state, empty/removed cgroups, and a terminal receipt. Controller
    access remained `/dev/null` only; both evaluators used deny-network sandboxes with exact GPU
    devices. Full merged validation passed 578 tests plus 202 subtests; footprint/readme checks passed
    106 plus 6 tests.
  - [x] **Project the diagnostic-pilot v1 contract and pre-receipt failure posture into the Kernel-R&D
    dashboard.** ✅ 2026-08-12 — the hub now reads the newest terminal pilot receipt, verifies its
    self-hash and explicit no-authority constraints, and renders PASS, broker/model/evaluation/window/
    teardown facts under a loud no-campaign-authority boundary.
  - [x] **Retract the reported all-tree discovery parser collision after reproducing its exact test
    context.** ✅ 2026-08-12 — `test_live_controls.py` intentionally invokes mutually exclusive
    `--evaluate-existing` and `--execute` arguments inside `assertRaises(SystemExit)`; the argparse
    stderr is the expected assertion surface, not an import-time collection abort. Discovery with both
    strings in `sys.argv` collected **3,927** tests, and an AST sweep found zero module-scope
    `parse_args` calls in AutoKernel. No parser repair is warranted.
  - [x] **Trace and repair one reproducible Claude/Landlock startup failure outside the campaign.**
    ✅ 2026-08-12 — local, unpushed research commit `916cdc92` adds ten fixed volatile read paths,
    stages only Claude credentials/config with identity hashes, scrubs the
    staged config, keeps `/dev/urandom` read-only, and excludes volatile reads from stable identities.
    The bounded standalone Claude probe returned structured `{"ok": true}`, left an empty/removed
    cgroup, and retained activation/result/teardown evidence under
    `/mnt/raid0/llm/autokernel/probes/inf03-claude-sandbox-runtime-20260812-r1`. Focused validation passed
    **59/59**, footprint/README validation passed **106/106**, and the pinned full AutoKernel suite ran
    **5,653** tests with one expected failure and no unexpected failure. Those counts were observed in
    the live session but lack a retained transcript; the probe receipts are the durable runtime evidence.
    This proves the bounded probe, not full actor-critic campaign compatibility.
  - [x] **Preserve r5 and r6 as immutable non-rankable engineering attempts after the exact campaign
    cell falsified the standalone compatibility inference.** ✅ 2026-08-12 — both attempts completed
    only the starting-state baseline, then the first Claude/Codex 2h planner exited `-11` with empty
    stdout/stderr. R6 included all ten new runtime paths and used source `916cdc92`, yet reproduced r5;
    its controller cgroup was verified empty and removed. Neither attempt has an aggregate, controller
    ranking, belief update, proposal-bank entry, champion, or release authority.
  - [x] **Trace r6's exact campaign-versus-standalone differential and repair the process-bound model
    sandbox boundary.** ✅ 2026-08-12 — `/proc/self/*` Landlock rules were bound to the controller PID,
    so its forked Claude child still received `EACCES`. Research `43ba6263` leaves the controller in a
    deny-network broker-only sandbox, launches each read-only model client in its own outbound sandbox
    so `/proc/self/*` binds to the actual CLI PID, preserves the digest-pinned single-writable-bind
    Docker boundary for the Codex actor, and journals model/evaluation receipts through the parent.
    Targeted validation passed **77/77**; the pinned full AutoKernel suite passed **5,655** tests with
    one expected failure.
  - [x] **Refresh the actor pin and preserve r7's pre-execution refusal.** ✅ 2026-08-12 — research
    `a3fcea3c` pins the repaired actor entrypoint. R7 correctly refused the stale prior pin before any
    controller or GPU command; its audit file SHA-256 is
    `45c4ae3dbb563dfebd20715238407b304cdbdd98f96dfc841cae310d2dd017b0` and its receipt self-hash is
    `0f0b9000b4a2647e68db493c751769f4f845db4bce6d089a0aa15d3e10b7cfab`.
  - [x] **Preserve r8 as an immutable partial, non-rankable implementation diagnostic.** ✅ 2026-08-12
    — the 7/7 audit passed and the starting-state baseline completed, but the first actor cell exited
    before Claude inference because the broker-backed controller unnecessarily constructed
    `ArenaWorkspaceEvaluator`, importing unavailable `yaml` and the pinned vendor evaluator inside the
    broker-only Python environment. The controller cgroup teardown is hash-bound, verified empty, and
    removed. R8 has no aggregate, ranking, belief, proposal-bank, champion, promotion, or release
    authority.
  - [x] **Remove the broker-backed controller's unnecessary Arena evaluator import, prove the exact
    Claude cell under the campaign sandbox, then launch a fresh governed 7/7 attempt.** ✅ 2026-08-12
    — research `6270ebc1` keeps vendor evaluation in the parent and `cd1303b9` refreshes the actor
    pin. R9 then exposed a separate broker transport defect: `socket.sendall()` selected denied
    `sendto`, so it stopped before Claude with a verified-empty, removed controller cgroup. Research
    `eb00c411` writes framed requests directly on the inherited descriptor. R10 consequently banked
    the starting-state evaluation and completed the first real Claude planner request, then correctly
    stopped before actor execution when its mutation guard counted the controller's staged session and
    config as task source. The ephemeral state was scrubbed. Research `2525f2f7` separates semantic
    task files from model state and `7b20b5d6` refreshes the manifest pin. R11 completed the first full
    real planner → digest-pinned Docker Codex actor → centralized GPU evaluation → Claude critic
    loop, but was stopped as an immutable non-rankable partial after repeated mechanisms exposed the
    missing feedback seam.
  - [x] **Close r11 without interpreting it and prove measured critic feedback alters the next live
    proposal.** ✅ 2026-08-12 — r12 iteration one compiled and passed 4/4 correctness at
    `average_speedup=0.9967805648538064`; its critic selected `revise`. Iteration two's planner cited
    that exact measurement, forbade the rejected unmasked-fastpath mechanism, corrected the infeasible
    autotune suggestion from the fixed launcher contract, and proposed a distinct shared-offset/
    vectorized-streaming mechanism. It also compiled and passed 4/4 correctness at
    `average_speedup=1.0059084616458236`; the critic selected `accept` while acknowledging that the
    ratio remained inside noise. This proves feedback propagation, not a kernel speedup.
  - [x] **Move host actor-runtime attestation out of the confined controller and pin the repair.** ✅
    2026-08-12 — after all six model calls and three brokered evaluations, r12 finalization failed by
    reopening `/usr/bin/docker` inside Landlock. Research `4b516e9f` makes the parent model-receipt
    chain authoritative for host runtime identity and validates the actual flat model-receipt layout;
    `f316bf31` pins the source. The full AutoKernel suite passed 5,660 tests with one expected failure.
  - [x] **Close INF-03 seven-arm r13 as an immutable partial and identify its confined-runtime failure.**
    ✅ 2026-08-12 — the baseline completed and planner model receipt ordinal `0001` persisted, then
    Claude failed exactly `EACCES: permission denied, mkdir '/mnt/raid0/llm/tmp/claude-1000'` before a
    completed planner response. All captured PIDs are dead and every controller/model cgroup was verified
    empty and removed. R13 has no controller cell receipt or aggregate and is permanently non-rankable.
  - [x] **Repair the confined Claude runtime temp path.** ✅ 2026-08-12 — research `268115ad` refuses
    inherited host scratch for direct model clients, creates a fresh call-scoped directory under the
    governed workspace, binds the relative path plus `ambient_host_temp_inherited=false` into the model
    sandbox receipt, and verifies deletion after every call. The campaign manifest still pins its parent.
  - [x] **Pin `268115ad` and launch a fresh seven-arm attempt.** ✅ 2026-08-12 — research
    `446a1c31` isolates each brokered model call and `152ed0d9` pins that runtime into the campaign.
    Seven-arm r14 audited `ready` at 7/7 on observed clean source `152ed0d9`. A timestamped live
    checkpoint found seven successful, non-timeout model receipts (five Claude and two Codex) and three
    complete parent-owned evaluator windows. Every Claude receipt recorded
    `ambient_host_temp_inherited=false` plus a call-scoped relative temp path, and its captured cgroup was
    verified empty and removed. This proves the r13 capability repair across repeated real calls; it does
    not make the live campaign terminal or rankable.
  - [x] **Close seven-arm r14 as an invalid source-identity attempt.** ✅ 2026-08-12 — after nine
    successful model calls and four correct evaluator windows, the outer checkpoint guard raised
    `task or controller source identity changed during checkpoint execution`. Its audit had bound clean
    source `152ed0d9`, but implementation work advanced that same worktree to `03f9ae69` while the cell
    ran. The actor cell therefore emitted no admitted cell receipt or panel aggregate. All eight device
    claims have released receipts; 14 captured PIDs are absent; and all 12 recorded controller/model/
    evaluator cgroups are verified empty and removed. R14 is permanently invalid and non-rankable.
  - [x] **Close seven-arm r15 as a transient structured-output failure with complete teardown.** ✅
    2026-08-12 — r15 completed the baseline plus four compilation/correctness-4/4 controller-feedback
    windows, then Claude call ordinal `0010` returned non-timeout exit `1` with exact subtype
    `error_max_structured_output_retries` and terminal reason `structured_output_retry_exhausted` after
    five provider-internal attempts. No actor-cell receipt or aggregate was admitted, so r15 is
    permanently non-rankable. All **7/7** device claims released, all **13** captured PIDs were absent,
    every recorded cgroup was empty and removed, and the immutable source worktree remained clean at
    `03f9ae69`.
  - [x] **Add one bounded, receipt-visible retry for exact structured-output exhaustion and refresh the
    campaign pin.** ✅ 2026-08-12 — research `537163d5` retries exactly once only for the recognized
    non-timeout Claude wrapper failure, journals the first and retry attempts separately, and leaves
    arbitrary non-zero exits and timeouts fail-closed. Research `eb1de388` refreshes the actor-critic
    campaign identity to those bytes. The admitted Python 3.12 environment passed the full AutoKernel
    suite: **5,682 tests with one expected failure**. Both commits are pushed and on research `main`.
  - [x] **Preserve seven-arm r16 as a pre-claim source-identity refusal.** ✅ 2026-08-12 — its static
    audit observed the new actor entrypoint but the campaign still expected the pre-repair digest, so it
    refused before controller inference, GPU execution, or device-claim acquisition. The corrected pin
    is `eb1de388`; r16 carries no empirical or comparison authority.
  - [ ] **Drive retry-hardened seven-arm comparison to a terminal aggregate and validate the complete
    receipt chain.** R18 is terminal-noncomplete and non-rankable after the parent rejected mutable
    host Claude-config drift following the staged copy. No actor-cell receipt or aggregate was admitted.
    Fresh r19 is live from clean immutable research `0b1fdbe9`; its ready 7/7 audit, completed baseline,
    and active first actor cell grant no comparison authority. Preserve diagnostic checkpoints, but never
    aggregate, rank, bank, promote, release, or resume r4-r18.
    - [x] **Patch the EvoEngineer evaluator-constructor call and launch a fresh immutable successor.**
      ✅ 2026-08-12 — research `381bc55e` wires governed `source_paths` through EvoEngineer, GEAK,
      K-Search, KernelFoundry, and Xe-Forge; `11bccd41` refreshes affected entrypoint pins. Research
      `b0d6f79f` adds prospective intermediate evaluator belief rows and strict feedback-only reading.
      Fresh r18 was sealed at `17b9208d`; its later terminal disposition is recorded below.
    - [x] **Close seven-arm r18 as a staged-host-input identity failure without upgrading its measured
      partial.** ✅ 2026-08-12 — r18 completed the baseline and full Claude/Codex 2h loop, including
      four compile/correctness-4/4 candidate evaluations, but the outer parent re-hashed mutable host
      `~/.claude/.claude.json` after its bytes had already been copied into the scrubbed workspace.
      That host file changed during the multi-minute cell, so admission failed after controller exit.
      No actor-cell receipt or aggregate was admitted; r18 remains permanently non-rankable and is
      recorded in the dashboard disposition overlay under exact attempt id `21754d23…`.
    - [x] **Bind staged controller inputs at copy time and launch a clean immutable successor.** ✅
      2026-08-12 — research `0b1fdbe9` verifies the source hash immediately before copying, verifies
      the staged bytes, emits `epyc.autokernel.controller_staged_inputs.v1`, and validates the receipt
      instead of reopening a later-mutated host credential/config file. The controller suite passed
      **596 tests**. R19 is live from clean exact source `0b1fdbe966f338a8ac840c9961eb718528af1a0f`
      at `/mnt/raid0/llm/autokernel/campaigns/inf03-available-source-seven-arm-r19-20260812`; its audit
      is ready 7/7, baseline is complete, and the first actor cell is active.
    - [x] **Specify the safe controller-overlap architecture before enabling it.** ✅ 2026-08-12 —
      research `dfe265a1` defines immutable schedule order, isolated deterministic lanes, serial
      per-lane checkpoints, exact MI210 claims around evaluator windows, bounded claim waits, an
      inherited attempt lease, PID/start-tick cancellation, manifest-v3 validation, truthful observed
      overlap, and a required overlap A/A noise gate. This is a design/verification contract only:
      overlap remains disabled, and r19 intentionally retains the proven sequential runner.
    - [ ] **Implement and validate deterministic controller overlap for a post-r19 successor.** Close
      every `dfe265a1` gate: manifest-v3 lane/claim bindings, inherited attempt lease, PID/start-tick
      cancellation, lane confinement, contention/restart regressions, truthful observed concurrency,
      disjoint CPU placement, and a governed overlap A/A inside its predeclared noise bound. Do not
      change r19 execution semantics or make concurrent results rankable before that matrix passes.
  - [x] **Narrow INF-03 MI210 claims to the centralized evaluator's actual GPU windows.** ✅
    2026-08-12 — research `e6c7aab6` and the r4 manifest bind
    `controller_deliberation_holds_no_gpu_claim=true`, a GPU-blind controller environment, claims
    only around vendor/final measurement windows, exact complete-checkpoint resume, and signal-safe
    release. The audit observed no KFD PID or active device claim while KernelFoundry deliberated;
    other governed GPU work may use those intervals.
  - [x] **Obtain and execute the exact licensed EvoEngineer source through the governed adapter.** ✅
    2026-08-12 — research `f42b3540`, `7e915831`, and `a81c2692` pin MIT source commit
    `1649715a975b9022c84b5279c88aaef0b73b28dc`, preserve the historical EvoEngineer-Full loop and its
    four ordered fallback semantics, and exercise the exact upstream `EvoEngineer.run()` for 10
    generations / 40 samples with fixture model/evaluator bridges only. The fresh physical static audit
    at `/mnt/raid0/llm/autokernel/probes/inf03-isolation-available-source-reaudit-20260812-r3/receipt.json`
    is ready at **7/7** over four tasks with zero controller/GPU commands (self-hash `5fa93695…`, file
    SHA-256 `082d977f…`).
  - [ ] Obtain the exact licensed ARGUS source release, then port its real controller policy through the
    same governed adapter contract; a namesake substitute is not admissible. Until then the separate
    full 8/8 panel must refuse, while the 7/7 available-source panel remains diagnostic-only.
- [x] Build C4 gfx90a profiler-metric analyzer (GEAK-v2 raw-rocprof path first) ✅ 2026-08-11 —
  `profile_report.py` implements the deterministic paired mapping/formal report; the live single-process
  Q4_K/Q8_0 captures prove it on gfx90a. `profile_context.py` binds the report hash into a priced
  AutoKernel discovery block and a framework-neutral `c4_evaluator_observation.v1` seam for GEAK /
  AgentKernelArena adapters. The seam is diagnostic-only and cannot write a verdict or rank.
- [x] Build C6 anti-reward-hacking layer (robust-kbench + AgentKernelArena unseen-shape) ✅ 2026-08-11
  — the immutable external scorer, search-tree monitor, unseen/hostile-shape cases, stream/thread
  escape checks, planted red-team corpus, prompt disclosure guard, and ranked-set short-circuit cases
  are implemented and covered by the AutoKernel suite. Live candidate sensitivity remains governed by
  the producer-dependent RVP-C2 rows, not by reopening the C6 implementation task.
- [x] **Compile the initial EPYC C3/C5 exact-suite seam.** ✅ 2026-08-11 — research `4320d83a`
  (main `5dfd14ad`)
  emits a hash-bound three-case plan/receipt contract for attention `k228`, MoE `k175`, and native
  Q4_K dequant, with honest vendor floors, correctness/integrity precedence, and captured-workload
  whole-model exit. Apex owns only the Python/Triton overlay cases; dequant correctly uses the
  experimental-binary adapter. This closes the offline integration seam, not the real MI210/Apex
  evidence campaign.
- [x] **Add selected-entry-only Apex runtime validation without guessing C5 mappings.** ✅ 2026-08-11
  — research `934b2a8d` bypasses Apex's unrelated-repository global validation bug only for an exact,
  reviewed registry row; all source/toolchain/model/capture identities remain fail-closed. The
  subsequent hash-bound mapping audit (`a9779163`, main `4fd985c2`) classifies both seeds honestly: `k228` is adjacent
  to AITER MLA prefill but has no pinned gfx90a HSA image or proved LSE/ABI contract, while composite
  `k175` requires missing component-graph/multi-trace support. Both now return typed structural
  mismatches rather than a similar-looking single kernel.
- [x] **Harden the governed C3/C5 real-capture seam and bind its observation window.** ✅ 2026-08-12
  — research `7b3cc6f4`, `a865bd7c`, and `300647bd` bind live-produced gfx90a inventory, KFD
  producer ancestry, exact sampler-window overlap, immutable acquire/release claim slices, frozen
  source paths, and selected-entry-only Apex admission. Focused validation passed 27 tests and the
  evaluator regression passed 1,579 tests. This closes the capture mechanism, not the model-specific
  tensor evidence.
- [ ] **Provide the exact k228 model-tensor capture hook manifest and run the governed capture.** The
  hook must execute the real model surface and emit the source/model/process/window identities the
  C3/C5 provider validates; a HyRA reference tensor or synthetic fixture cannot substitute.
- [ ] **Provide the exact k175 ordered multi-trace capture hook manifest and run the governed capture.**
  The hook must preserve the router → top8/count/rank → dispatch → routed/shared experts →
  weighted-undispatch graph; a single similar-looking kernel cannot satisfy the composite case.
- [x] **GEAK-family freshness sweep completed ✅ 2026-08-03** (research-intake Stage-2b): GEAK v4 retains a current gfx90a KB (`perf_knowledge/hardware/cdna2_mi200/`, `updated: 2026-06-08`) and 40 gfx90a capability entries; all published evaluation remains gfx942. Scoping caveat amended above; "677/678/679 are a coverage regression" **retired** — for GEAK proper it is unpublished coverage. `v1.0.0 @ 4ffba15a` pinned.
- [x] Re-target the program objective from the Q8 rung to the fp16 rung, with a banded per-lever ceiling (K1–K12) ✅ 2026-08-03
- [x] Register **ARGUS** (arXiv 2604.18616) as a controller candidate alongside EvoEngineer /
  KernelFoundry / K-Search / Xe-Forge / GEAK ✅ 2026-08-11 — `arena_adapter.py` registers the
  `argus` controller id under the same exact three-argument bridge and prompt-hygiene contract as
  the other arms. This records registration only; the cited MI300X result remains vendor evidence.
- [x] Read GEAK's `landscape/` + `languages/` sections as an external check on our seven-school
  taxonomy ✅ 2026-08-11 — current GEAK `5107c7e4` showed that “seven schools” was never a durable,
  enumerated fact. The corrected comparison is seven *abstraction families*, recorded in the
  [deep-dive sweep](../../research/deep-dives/agentic-rocm-kernel-authoring-geak-synthesis.md#sweep-2026-08-11--landscape--languages-taxonomy-check-closed).
  The live gfx90a ladder is Triton → TileLang → HIP/C++ → MFMA/ISA when measurement justifies each
  descent, with CK/CK-Tile/rocWMMA as baselines. FlyDSL is a valuable new vocabulary source but GEAK
  scopes it to gfx942/gfx950, so it is not a current MI210 execution arm.
- [x] Resolve the profiler-tooling blocker on the host (rocprofv2/rocprof/omniperf/rocm-bandwidth-test) — C4 and the LDS bank/phase solver are both gated on it ✅ 2026-08-12 (mainB)
  — **re-verified live, not inherited from the 2026-08-03 note.** Through
  `source /mnt/raid0/llm/tools/rocm-profilers-6.2/env.sh`: `rocprof` and `rocprofv2` both report
  `ROCm 6.2.0-66 / ROCProfiler 2.0`, `rocm-bandwidth-test` responds to argument parsing, and
  `omniperf` resolves on PATH but **does not run** — it exits on missing Python deps
  (`astunparse==1.6.2`, `colorlover`, `dash>=1.12.0`), exactly as line ~138 predicted, so its
  deliberate deferral stands and is accurate. **Note for anyone re-checking this**: none of these
  are on the default `PATH` and none are under `/opt/rocm` — that is by design (extraction, so the
  shared bind mount is untouched). Probing `PATH` or `/opt/rocm` returns a confident *absent* for
  all five. Source the env or you are measuring the wrong universe.
- [x] Run HipKittens' LDS bank/phase solver method on gfx90a ✅ 2026-08-11 — 372 bank and 6,048 phase
  dispatches across three repetitions measured **32 LDS banks and eight phase cliques of eight lanes**.
  HipKittens' CDNA3 64-bank/two-phase swizzle topology therefore does not transfer to gfx90a;
  `swizzle_transfer_class=retune_required`. Receipt:
  `/mnt/raid0/llm/autokernel/probes/inf03-lds-gfx90a-20260811-r4/receipt.json`, SHA-256
  `ae1d833c704bdae9a78767d0fc0b927298d6d1dfdb31a0ea11c34058dc525987`.
- [x] Add `-mllvm --amdgpu-unroll-threshold-local=600` to the build-flag checklist as a
  **precondition of any ROCm 7+ upgrade** ✅ 2026-08-11 — the new
  [ROCm upgrade checklist](../../docs/runbooks/rocm-upgrade-checklist.md) requires Linux builds to pass
  the option through `CMAKE_HIP_FLAGS`, proves it reached HIP compile commands, and retains it unless
  an exact-toolchain matched A/B shows it unnecessary. This does not alter frozen v9 or ROCm 6.2.


> **MERGE NOTE 2026-08-12 (coordinator).** The two sides of this section disagreed on FACT,
> not formatting, and I preserved both rather than picking. The 2026-08-11 side records the
> LDS bank/phase solver as RUN with a receipt (32 banks, 8 phase cliques, SHA-256
> `ae1d833c704bdae9a78767d0fc0b927298d6d1dfdb31a0ea11c34058dc525987`); the 2026-08-12 side
> records `rocprofv3` as ABSENT and questions whether the solver's counters are reachable
> at all. If the receipt is real the question is answered; if the receipt came from a
> different path than the solver row describes, the row is mis-stamped. **OWNERS: verify
> and correct — I did not adjudicate this.** The profiler row below is the NEWER
> 2026-08-12 re-verified-live text, deliberately preferred over the 2026-08-11 inherited note.
- [x] **`rocprofv3` absence vs the LDS solver — settled, the row was mis-stamped** ✅ 2026-08-12 (mainB).
  Both halves were true and they were never in conflict. `rocprofv3` IS absent (the side-load is a ROCm
  6.2 extraction topping out at ROCProfiler 2.0) — but the solver did **not** need it: it ran
  2026-08-11 over a **rocprofv2-compatible counter path**, receipt SHA-256 `ae1d833c…`, 372 bank and
  6,048 phase dispatches, deriving 32 banks and eight phase cliques. So the ~40 min GPU window is
  already spent and its result is banked; what was wrong is the *description* at line ~126, which
  names `rocprofv3` as the instrument. **The run is real; the row named the wrong tool.**
  Recorded because the failure mode generalises: a row that names its instrument can go stale against
  a run that used a different one, and then a correct check of the instrument returns a false BLOCKED.
- [x] **GEAK-family freshness sweep completed ✅ 2026-07-29**: refreshed the deep-dive appendix from AMD's current AgentKernelArena/GEAKv3 reports. AKA now publishes 214 tasks and a 44-task MI300X comparison; this is vendor/CDNA3 evidence only and does not close the MI210/gfx90a reproduction gap. [Freshness appendix](../../research/deep-dives/agentic-rocm-kernel-authoring-geak-synthesis.md#9-freshness-appendix-sweep-at-each-handoff-audit--when-the-mi210-racks).


## Auto-kernel revival — research-intake integration 2026-07-22 (C5 seed corpus + FP8 authoring target)
_Via /research-intake Stage-2 (intake-884 HyRA MI210 re-check). The loop AUTHORS gfx90a kernels — these are task specs, not port-or-decline artifacts._
- [x] Seed C5 with the HyRA sol_execbench kernels as reference specs the loop authors gfx90a kernels FROM (maps onto this handoff's "seed EPYC ops: attention / MoE-dispatch / dequant"): 5 bf16/fp16 Triton kernels directly (k138 mamba, k145 hyena, k154 chunk-gated-delta linear-attn, k175 MoE dispatch, k228 MLA paged-prefill); CUDA-lib-bound winners (k215 GEMM/flashinfer, k225/k227 GQA/MLA) as reference targets to re-author. All Hopper-autotuned + NVIDIA-only-attested -> loop re-authors + re-attests on gfx90a (wavefront-64/MFMA/LDS) ✅ 2026-08-11 — `c5_seed_corpus.{json,py}` pins all eight artifacts to Hyra-results commit `26ebfbe7`, separates the direct and CUDA-bound sets, excludes Hopper scores/latencies from authoring context, and lets Arena tasks select exact rows through `c5_seed_ids`.
- [x] Add FP8 as an authoring datatype target: gfx90a has no native FP8 MFMA, but an FP8-weight -> bf16-MFMA upcast GEMV is buildable and potentially bandwidth-valuable for memory-bound decode (half the weight bytes of bf16; compute upcast to bf16) — an experimental-kernel investigation, NOT a hard wall. NVFP4 microscaling (per HyRA k185) is the harder end; scope after the FP8-upcast path ✅ 2026-08-11 — `datatype_targets.py` exposes a non-numeric FP8-weight/software-upcast contract to Arena tasks, requires an independent decoder, exact-shape baseline, upcast-cost attribution and whole-model gate, forbids batch-one MFMA assumptions, and mechanically defers NVFP4 until the FP8 path has a terminal result.

---

## Loop engineering — controller experiments and prompt hygiene (research-intake Stage-3, 2026-08-10)

_Via `/research-intake`. Source: an operator-supplied field note on running a coding-agent research
loop against a batched linear-algebra kernel to competitive results._

**Provenance, stated once and carried by every item below.** The note's loop-engineering claims —
that raising the planner's reasoning-effort knob lengthens search rather than deepening any single
answer, and that a proximate numeric target changes what the agent proposes — have **zero primary
backing**: the accompanying repository does not contain the loop, the transcripts, or any ablation.
They are `[unverified]` **hypotheses** and no number from them may be quoted as evidence anywhere.

**Operator steering, verbatim (2026-08-10):** *"zero primary backing doesn't mean we can't reason
about the potential usefulness of this information from first principles."* That is the governing
instruction for this section. The unverified contract governs **citation**, not **consideration** —
so each hypothesis below is evaluated on our own stack and paired with the cheapest experiment that
would settle it here. Nothing is blocked: frontier Anthropic/GPT models are available to our
planners (operator, 2026-08-10 — we are not restricted to local models), and every experiment in this
section costs **zero GPU and zero local inference**.

- [x] **AK-PL-1 — Prompt-leak guard on the assembled authoring prompt.** Grep the fully-rendered
  planner/actor prompt for `test-backend-ops`, `max_nmse_err` and `init_tensor_uniform`. An agent that
  can read the tolerance it must clear is being graded by a rubric it holds.
  **Test the guard against the COMPLIANT path too** — a guard that also rejects the legitimate prompt
  is a guard that will be disabled the first time it fires
  (`feedback_guard_must_not_forbid_its_own_idiom`). Pairs with RVP-C6-7 in
  [`rocm-verify-profile-backend.md`](rocm-verify-profile-backend.md), which filters the *diagnostics*
  half of the same channel. ✅ 2026-08-11 — `controller/authoring_contract.py` scans the fully rendered
  prompt for all three named internals plus the exact `ERR = value > tolerance` disclosure shape;
  tests plant every leak and prove an ordinary compliant authoring prompt remains admissible.
- [x] **AK-PT-1 — Per-turn productivity accounting (this handoff owns it; emitted at
  `autokernel-research-loop.md` §8.8).** Record `(turn, task, correct?, speedup)` per refine turn, and
  split **rescued** (a turn that fixed a previously-failing candidate) from **persistent** (a turn
  that improved an already-correct one). We currently measure **neither**. The reason the split is
  load-bearing: mean speedup falls across turns as a **composition** effect — later turns admit
  candidates that earlier turns could not fix, which are systematically worse — so a declining mean is
  not evidence that refinement stopped working, and reading it that way would truncate the loop
  exactly where it is still paying. ✅ 2026-08-11 — AutoKernel `turn_productivity.py` implements an
  immutable append-order archive with state-continuity checks; rescued/persistent/failed/regressed is
  mechanically derived from the correctness transition, incorrect turns cannot carry a speed rank,
  and every observation retains its evidence reference and content hash. AK-X-6 is its first consumer.
- [x] **AK-LE-0 — Implement the static experiment contracts and selected-task-safe scaffold for
  AK-LE-1/2/3.** ✅ 2026-08-12 — research main `51742ebd` integrates the effort/persistence,
  proximate-target, direct-implement, split-implement, and split-exploit arm contracts. Every scaffold
  consumes the same independently SHA-256-bound selected-task artifact; planner-only `PROPOSE … do
  not implement it yet` text is excluded. The accepted full AutoKernel suite passed **5,464 tests
  with one expected failure**. This closes static wiring only; AK-LE-1/2/3 remain open until their
  budget-matched model runs produce empirical results.
- [x] **Add the governed execution bridge for the AK-LE-1/2 planner panel.** ✅ 2026-08-12 —
  research `98f642cd` compiles exact `ExperimentContract` model/quant/effort cells, resolves and
  hashes the Claude/Codex CLI, executes each cell in a captured read-only process group, and seals
  the exact prompt, argv, stdout, stderr, last message, timing, and strict parsed observation. It
  has observe-only authority and cannot mutate a campaign, rank, select a champion, or release.
  Focused runner tests passed **16/16**, the controller slice passed **487/487**, and the integrated
  canonical AutoKernel suite exited zero.
- [x] **AK-LE-1 — Reasoning-effort × search-persistence experiment.** Hold champion, retrieval context
  and PROPOSE prompt fixed; sweep only the planner's effort knob. Measure **search** outcomes, not
  answer quality: count of novel non-duplicate hypotheses, count of explicit "this is already
  optimized" terminations, and how many proposals survive the pre-filter.
  - **Why this is not already covered.** [`reasoning-effort-levels.md`](reasoning-effort-levels.md) is
    a **stub**, and its entire measured evidence (the A4 / GPQA-Diamond ladder, +32.0 pp at 4.0×
    tokens) is about *answer accuracy on a fixed question*. Nothing in it measures whether more effort
    makes an agent keep looking. That is a different dependent variable.
  - **Pre-register the direction before running.** Our own observed planner failure is the *opposite*
    one — halting after a single critic "revise" — so the sign must be predicted in advance rather
    than read off the result. Its per-model invariant applies here too: effort is a property of a
    (model, quant) pair and is never inherited. ✅ 2026-08-12 — the corrected r3 panel completed all
    **8/8** predeclared Claude Opus 5 / GPT-5.6 Sol × high/xhigh × control/target cells, then the
    source-pinned structural reducer emitted one planner receipt and **32** prospective belief rows.
    Higher effort did not increase novel/surviving counts in either control arm (Claude `6→6`, Codex
    `3→3`) and all four xhigh cells took longer. This is a bounded null result for the predeclared
    control contrast, not a model-wide effort claim: there is one observation per cell and no
    scaffold or campaign-ranking authority. Evidence:
    `/mnt/raid0/llm/autokernel/campaigns/ak-le-planner-20260812-r3/planner-reduction.json`, file
    SHA-256 `c24122893acdd7cf10f042a447d7daa41aeb7d77161dcda8b5b5bf5344b57791`.
- [x] **AK-LE-2 — `proximate_target` arm, run as one extra arm on AK-LE-1.** Tests the effort × target
  interaction jointly rather than as a second study. **Specify it as a rendered planner-context line
  only — never a manifest field a gate can read** — regenerated each round and scoped **per cell**.
  AK-D3 demoted a percentage figure from trigger to readiness signal for sound statistical reasons
  (threshold peeking, winner's-curse inflation); a target that any gate can read re-introduces exactly
  that, one layer up. The rendered-context form gives the planner the steer without giving the
  evaluator the number. ✅ 2026-08-12 — r3 rendered the decode target only as planner context and
  reduced the complete matched panel. Claude produced `6` surviving hypotheses in every arm; Codex
  target arms produced fewer than their matched controls (`3→1` at high, `3→2` at xhigh). No cell
  emitted an already-optimized termination. The honest result is null for Claude and mixed/adverse
  for Codex on this single panel; it does not establish a transferable proximate-target benefit.
- [x] **AK-LE-3 — Split *implement* from *exploit* as two prompt roles on the SAME model first,
  budget-matched in wall-hours, before any multi-model A/B.** The note's four-role split
  (explore / critique / implement / exploit) is a runnable design, but the delta from our current
  two-role planner may be small, and a multi-model comparison would confound scaffold with model.
  **Vary scaffold and model independently** — published evidence has identical models inverting rank
  under different scaffolds.
  - **What is already reusable (inventory, operator steering: "what could be repurposed of our current
    design").** AutoPilot already runs a two-role draft + critique planner with a **binding** critic —
    that is `explore` + `critique` in place, and it is the half that usually costs most to build.
    [`architect-model-selection-bench.md`](architect-model-selection-bench.md) already selects a model
    per role, but on **quality**, never on throughput — the per-role choice machinery exists and its
    objective is the missing piece. AgentKernelArena's `@register_agent` adapter pattern (Decision
    Snapshot item 2) is the substrate for registering the arms. So the increment is roles 3 and 4 plus
    a throughput-aware selection objective, not a new controller.
  - [x] **Implement a governed same-model direct-implement versus implement-then-exploit
    authoring/evaluation seam before running this panel.** ✅ 2026-08-12 — research `c4434810` and
    `230721b6` compile the exact two-model × two-scaffold panel, use the same writable Codex actor and
    trusted Arena evaluator in every cell, isolate disposable baseline/candidate worktrees, bind
    runtime/image/import identities, serialize MI210 claims, and clean up exact owned worktrees on
    every exit. The focused integrated slice passed **29/29** and the controller suite passed
    **512/512**. The following terminal campaign closes the parent experiment.
  - [x] **Run and reduce the governed same-model scaffold panel.** ✅ 2026-08-12 — all **4/4**
    gpt-5.6-sol/terra × direct/split cells compiled, passed all four correctness cases, retained four
    valid baseline and four optimized timing cases, and released their MI210 claims. Average speedup
    was Sol direct `0.9957477195`, Sol split `1.3930213301`, Terra direct `1.0021830618`, and Terra
    split `0.9935661224`. The split benefit is therefore task- and model-specific, not a cross-model
    scaffold default. The panel has diagnostic observation authority only: it cannot rank a campaign,
    choose a champion, or authorize a release. Receipt:
    `/mnt/raid0/llm/autokernel/campaigns/ak-le-3-scaffold-20260812-r1/panel/panel.json`, self-hash
    `6e764cca1df8cf347d43c48bca70654f1cec4dddbe967e7e80a3705c3d71ff32`, file SHA-256
    `5af0307ffd823e093d06d26497a536e414090d72e3dee73bf3c33f4ea853e666`.
- [x] **AK-LE-4 — Context discipline: a priced context budget and a reversible compaction protocol.**
  (a) A per-round context-budget table with an explicit **never-bulk-read** rule, so the loop cannot
  spend its window on a file it will not use. (b) A research-log compaction step that writes a
  **what-was-kept / what-was-dropped header** — and, critically, **a git recovery recipe**. The source
  note describes the compaction and omits the recovery half; compaction is only safe *because* it is
  reversible, and a protocol that drops the reversibility keeps the risk and discards the mitigation.
  ✅ 2026-08-11 — the authoring contract prices every context item and round, enforces per-item and
  total caps, rejects duplicate/bulk reads, prints the budget table and literal never-bulk-read rule,
  and emits kept/dropped compaction headers with an exact `git -C <repo> show <commit>:<path>` recovery.
- [x] **AK-LE-5 — Anti-fabrication clause for externally-sourced numbers.** Our invariants
  (`autokernel-research-loop.md` §4) bind the loop's own measurements to a protocol, an anchor and a
  receipt. **Nothing currently covers a number the loop claims came from outside itself** — a vendor
  figure, a paper result, a leaderboard score quoted into a proposal. Require an external number to
  carry a source, a retrieval date or commit, and a normalisation to roofline utilisation (§8.3.1)
  before it may appear in a proposal at all; otherwise it is not admissible even as `design_prior`.
  The source campaign logged this exact failure three times in its own write-up, which is the only
  part of that document that is self-evidencing. ✅ 2026-08-11 — proposal-v3 now requires a validated
  `external_numbers` vector (empty is explicit). Every entry binds source, retrieval date or commit,
  exact quant, denominator basis and normalized roofline utilization; the validator independently
  re-derives utilization and refuses mixed unit/quant/basis denominators.

**Declined here (recorded so it is not re-derived):** adopting the note's evaluation cadence. It rests
on elastic external accelerator capacity and a free third-party adversarial verifier; we have one
MI210 and a verifier we own. **The orchestration is reproducible almost fully; the cadence is not** —
and per AK-LN-1/AK-LN-4 in [`autokernel-research-loop.md`](autokernel-research-loop.md) §21, our
answer to cadence is concurrent screening lanes, not more devices.

## Research Intake Update — 2026-08-15 (SOL-ExecBench as a gfx90a correctness oracle; intake-1102)

The Stage-2 dive settled the C5 provider question, and the answer is more useful than the headline
suggests. **State it in two parts or it misleads.** The ROCm port carries *no measured constants* for
gfx90a — it is gfx950-only (MI350X/MI355X), pinned to ROCm 7.2 against our 6.2. **That is not the same
as unusable here.** Its compile path is genuinely architecture-agnostic (the gencode helper falls back
to reading `gcnArchName` from torch and would emit `--offload-arch=gfx90a`, and a `LOCAL` hardware
target exists for exactly this case), so **compilation and correctness checking run on our MI210
today**. Only SOL *scoring* needs a port, because T_SOL, T_b and tolerances are per-part measured
quantities.

All eight C5 seeds resolve to real definition/workload/reference/tolerance/bound records — 193
workloads, all scoreable, zero deferred, all bf16/fp16 and therefore gfx90a-viable (the FP8 and NVFP4
deferrals do not touch our seeds).

- [ ] **C5-3 — Use the port as a gfx90a CORRECTNESS ORACLE now, ahead of any bound port.** Target
      `LOCAL` hardware, compile through the existing arch-agnostic path, and run the 10 fresh-input
      correctness rounds against the live-executed references. This is the C3/C5 provider validation
      the handoff already says a HyRA reference tensor or synthetic fixture cannot substitute for — and
      it needs no measured constants, so it is available immediately. AutoKernel is explicitly in scope
      to author and screen gfx90a kernels against it.
- [ ] **C5-4 — Classify the eight seeds by bound quality, and stop quoting `S` where it is meaningless.**
      Median headroom (T_b/T_SOL) per seed: **k215 6.8× (ok)**; k145 16.8×, k175 17.0×, k138 33.4×
      (loose); **k154 506×, k227 3,690×, k225 5,710×, k228 36,837× (vacuous)**. Above ~100× the score
      collapses toward T_b/(T_b+T_k) and carries no roofline content — it degenerates into plain
      speedup-over-PyTorch, the exact framing the benchmark exists to replace, and the repo says so
      itself. **Treat the four vacuous seeds as correctness-only oracles plus a PyTorch-relative speed
      number; never quote their `S` as a speed-of-light figure.** k228 is named in the source as a
      problem where a correct kernel beat the bound, and k227's correction took its bound to 8 cycles —
      "correct, and vacuous". Note k175 additionally sits on the still-defective `declared_traffic` tier.
- [ ] **C5-5 — Persist the kNNN ↔ problem-slug mapping into `c5_seed_corpus.json`.** All eight resolve
      uniquely with matching ordinals (k138→`L2__044_mamba…`, k145→`L2__051_…hyena…`,
      k154→`L2__060_chunk_gated_delta_rule…`, k175→`L2__081_moe_sparse_expert_dispatch`,
      k215→`FlashInfer-Bench__006_gemm_n2048_k4096`, k225→`FlashInfer-Bench__016_gqa_ragged_prefill…`,
      k227→`FlashInfer-Bench__018_mla_paged_decode…`, k228→`FlashInfer-Bench__019_mla_paged_prefill…`),
      with workload counts reproducing exactly (16/16/16/16/29/15/47/38 = 193). **The kNNN ids appear
      nowhere in either repository** — they are HyRA's — so this join is ours to maintain and should
      stop being re-derived. Add it as a `sol_execbench_problem_id` field.
      **Premise re-checked at wrap-up 2026-08-15 — the task is real but the gap is one level lower than
      written.** A `c5_seed_corpus.json` carrying all eight seeds *with* a populated `"slug"` field
      already exists (schema `epyc.autokernel.c5_seed_corpus.v1`), but only inside two throwaway
      verification snapshots — `tmp/wrap-research-router-20260812/` and
      `tmp/verify-provider-a54-20260812/scripts/kernel_rnd/autokernel/`. `git ls-files` finds **no**
      `c5_seed_corpus.json` in epyc-root or epyc-orchestrator, and the live orchestrator tree has no
      copy at all. So the mapping is not missing, it is *unpersisted*: land the file in the orchestrator
      tree under version control first, then add `sol_execbench_problem_id` beside the existing `slug`.

### OPERATOR DECISION — port the SOL bound constants to gfx90a?

This is the one item that needs MI210 GPU time and therefore a region claim, so it is presented rather
than filed. The port is **bounded and signposted** — the source's two refusals name their own remedy,
and its own MI355X runbook is effectively the checklist: add an MI210 `Part` entry (spec-sheet peak
clock + power cap), measure `F_LOCK` on gfx90a (they ship the task for it), add a measured `gfx90a`
`LLC_BYTES` (do NOT fall back to torch's reported L2 — it reports the per-XCD L2 and would undersize the
cache flush ~64×, making every memory-bound kernel look faster than it is), regenerate the SOLAR arch
YAML with the parametric generator, then re-measure T_b and tolerances.

**Scope it by bound quality, not by seed count.** Because four of eight seeds are vacuous, the honest
target is **k215 alone at 29 workloads** for a bound worth optimizing against, or ~77 workloads if
k138/k145/k175 are included at loose quality. Not 193, and certainly not the full 3,717.

| Option | Buys | Costs |
|---|---|---|
| **A. No bound port** | Correctness oracle (C5-3) + PyTorch-relative speed. Zero GPU time. | No speed-of-light measure on any seed. |
| **B. Port k215 only** | One genuine SOL-scored seed at 6.8× headroom. | F_LOCK + LLC measurement + T_b/tolerance re-measure over 29 workloads. |
| **C. Port k215 + the three loose seeds** | Four scored seeds, three of them loose. | ~77 workloads; the loose three yield weak roofline content. |
| **D. Full eight-seed port** | — | 193 workloads, four of which produce vacuous scores by construction. Not recommended. |

**Recommendation: B**, after C5-3 has shown the correctness oracle works end-to-end on gfx90a. A is a
legitimate stopping point if GPU time is scarce — the correctness half is where the C3/C5 provider gap
actually was.
