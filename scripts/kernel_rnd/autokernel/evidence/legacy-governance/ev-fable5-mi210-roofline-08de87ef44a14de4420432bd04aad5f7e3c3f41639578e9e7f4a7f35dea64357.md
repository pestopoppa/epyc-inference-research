# Window-2 findings 05 — Intake/deep-dive sweep + kernel roofline gap (2026-07-03, operator-directed)

**Two operator questions**: (A) did we miss important tasks in the research-intake / deep-dive corpus — especially stale **DGX-gated** dismissals now unblocked by the MI210? (B) roofline gap for the **CPU kernel (v6+iqk)** and the **MI210** — highest-ROI avenues to close it, and can the dequant slowdown be compensated (even Q8/fp16 caps ~60%)? Plus a standing instruction to **apply** the reprioritization to the live backlog, not just propose it (§7).

**Method**: 8 read-only sweep agents over **769 intakes + 125 deep-dives** (compact-extract → deep-read candidates → fold-check vs 119 active handoffs) + 2 roofline agents (CPU, MI210) + 3 adversarial verifiers + 1 kernel synthesis. 14 agents, 0 errors, ~2.09M tokens. Every number below is an **OBSERVATION** per MEASUREMENT.md (no protocol-id; throttle-suspect 28-day CPU host / single-run contended MI210 host) — usable for hypotheses, never to gate a keep/revert/deploy decision.

---

## 1. Intake sweep — the pattern is one stale gate, repeated

**The DGX-Spark gating is stale across a whole cluster of handoffs**, for one reason: the project was *considering* a DGX Spark (GB10, unified memory) and bought a discrete **MI210 (gfx90a, 64 GB HBM2e, PCIe4)** instead (2026-07-02). Handoffs written before then still read "expected ~July 2026 / nothing runs until the card racks / GPU-gated / deferred to DGX Spark." The sweep found **22 distinct missed-tasks** (16 MI210-triggered, 4 both-triggers, 1 evidence-plane, 1 standalone), **95 dead-confirmed**, **135 already-covered**.

### Applied this session (the fired-gate un-stalings)
| Handoff | Was | Now |
|---|---|---|
| `agentic-rocm-kernel-authoring.md` | "expected ~July 2026; nothing executes until the card racks" | **ACTIVE (MEDIUM)**; first step = GEAK-eval on gfx90a; only GEAK-v1 is gfx90a-proven (AgentKernelArena/robust-kbench are gfx942-listed) |
| `rocm-verify-profile-backend.md` | same gate | **ACTIVE (MEDIUM)**; pytorch-triton-rocm install + rocprof-compute + BitBLAS/TileLang (gfx90a-reachable, **not** AITER=gfx942-only) |
| `gpu-acceleration-path.md` | plan-of-record = DGX-Spark unified-memory hybrid | **retargeted** onto MI210 (discrete → real PCIe boundary); the "DGX Spark Target" section marked superseded reference; MI210 probes named |
| `ernie-image-turbo-evaluation.md` | "Spark/GPU next big lever when available" | GPU rebench **unblocked** (ROCm/HIP DiT bench) |
| `engram-conditional-memory.md` | Track-B Phase-0 proxy on a "rented H100" | **local MI210** (no cloud spend; gfx90a training-viability [unverified]) |
| `frontier-f3-data-flywheel.md` | W3 "HW-GATED — do not start before the MI210 card" | HW gate **cleared**; binding gates now = 100 trusted labels + gfx90a training-viability smoke [unverified] |
| `agent-world-env-synthesis.md` | Phase 2 "deferred to DGX Spark" | GPU present (MI210); Phase 2 stays downstream-gated on Phase 1 + a GRPO-viability smoke; scope as small-model repro (1 card ≪ 8×B200) |
| `pipeline-integration-index.md` ×3 | P0.5 ERNIE "when available"; P4 doc-to-LoRA "reopen with GPU"; L30 "DGX-GATED (no training GPU)" UniRL | ERNIE unblocked; doc-to-LoRA **stays closed** (GPU conjunct opened but the REPL-need conjunct did not); UniRL downgraded to training-viability-[unverified] gate |
| `master-handoff-index.md` §F + §B2-F3 + §A0 | "HW-GATED — all gated per operator instruction" | **ordered GPU queue** (G0 α-log → G1 P-GPU-1 → G2 op-smoke → G3 Gate-R) + strategic-priorities block surfaced above the N-rows |
| `speculative-decoding-mtp-refresh.md` | DFlash only cited as a "not-viable / same deployment wall" comparison | DFlash **promoted** to explicit MI210/HIP α-gated candidate (the wall was a *CPU* wall) |
| `learned-routing-controller.md` | ES cluster router-scoped only | recorded the **non-router ES sliver** (uncovered) + its 2 hard gates |

### The one genuinely never-folded *capability*: gradient-free Evolution-Strategies fine-tuning
`intake-564` (ESSA) / `intake-563` (ES-at-Scale, pop=30) / `intake-532` (EGGROLL) / `intake-565` (the off-task-KL guardrail). This is the **only training family that needs no autograd/flash-attn** — pure forward passes — so it sidesteps the "gfx90a training-viability [unverified]" gate that blocks every gradient-based fine-tune (F3-W3 QLoRA, agent-world GRPO). The project has **no fine-tuning path today**; this is the cheapest one to stand up. **Adversarial verify demoted it from a new P1**: it is mostly already owned (router-scoped) in `learned-routing-controller.md:822-831` with a deliberate no-separate-handoff decision; the uncovered sliver is a **non-router target** (small verifier/drafter LoRA-SVD). Two hard gates, confirmed: (1) the fitness oracle must be a **held-out eval slice, not the live authority eval-tower** (wiring the tower as an ES oracle is an operator-only, human-amendment-only change + a Goodhart risk); (2) **no LoRA-SVD→GGUF reconstruction path exists** in our llama.cpp+GGUF stack — that tooling is the real first task. Folded as a note into `learned-routing-controller.md`, not a new handoff.

### P2/P3 folds queued (owners assigned in the sweep)
`intake-310` `-ot exps=CPU`/`--n-cpu-moe` hybrid-MoE-offload probe (→ gpu-acceleration-path) · BitBLAS/TileLang gfx90a low-bit GEMM as the dequant-gap lever (`intake-497`/tilelang → rocm-verify) · `intake-460` Splitwise GPU-prefill/CPU-decode with KV handoff (→ findings-02 Family D) · `intake-729` Ornith-1.0-35B-A3B MIT agentic-coder as a decode-only worker A/B candidate (→ capability-registry) · cross-model-LoRA cluster `intake-764..771` (lightweight watch vs completed doc-to-lora) · `intake-576` Nemotron torch-ROCm self-spec [unverified] · `intake-757` Dockerless execution-free patch-verifier as a coder gating signal.

### Adversarial corrections absorbed
- **Both P1s demoted.** GEAK program → MEDIUM (an *optimization* to close the roofline gap, not a production blocker — llama.cpp-HIP already serves ~910 tok/s @32-way as-is). ES → not a new P1 (mostly covered).
- **AgentKernelArena (679) / robust-kbench (668) are gfx942/CDNA3-listed**, not gfx90a-proven; only GEAK-v1 (`intake-674`) is the gfx90a sanity substrate. "Reproduce GEAK-eval + AgentKernelArena" conflated a clean sanity gate with an exploratory port.
- Every GPU-number run remains an **operator-approved measurement** (the whole 2026-07-02 MI210 session ran under explicit operator GO); an agent stands the gate up, it does not autonomously run it.

## 2. Dead-confirmed — do not revive (highlights of 95)
Vulkan on gfx90a (falsified on this exact card — no ICD) · AITER/MORI/DeepEP on gfx90a (gfx942/gfx950-only, `intake-759`) · vLLM for `gemma4`/`qwen35` (0.10.1 predates both archs) · anything needing >64 GB HBM in one card (122B architect UD-Q4 = 78 GB; Kimi/GLM-5.2 1.6T/754B) · DGX-unified-memory-specific plays (the "no PCIe transfer" premise is false on a discrete card) · TiDAR-B (checkpoint-gated, not hardware) · the DAR-3/SPO+/Package-I freeze (regret-gated at 0.00%, not hardware — run the replay, don't reopen).

---

## 3. CPU roofline (production-consolidated-v6 + iqk)

**Where we are** (OBSERVATIONS, `iqk-port.md` FULL-STACK table): frontdoor Qwen3.6-35B-A3B Q8_0 ≈ **26.4 t/s = 13.8%** of the 460 GB/s STREAM roofline; worker gemma-4-26B-A4B Q4_K_M ≈ **48.5 t/s = 21–22%**. Bottom-up active-bytes/token: frontdoor 2.27 B active × 1.0625 B/param = **2.41 GB/tok → 63.6 GB/s**; worker ~3.2 B active × ~0.62 = **~2.0 GB/tok → ~96–102 GB/s**.

**But STREAM ≠ achievable GEMV.** A dense reference (Qwen2.5-Coder-32B) reaches only ~41–44% of STREAM (≈190–200 GB/s); MoE loses further to op-count / small-N, so the *realistic single-stream MoE-GEMV ceiling is ~25–30%* (115–140 GB/s). Against that: the **worker is already ~74–78% of its realistic ceiling** (little single-stream headroom left), while the **frontdoor is only ~49%** (a real single-stream gap). Correcting K1: neither role is "BW-tapped-out" — both sit on a **barrier/op-count-bound plateau below the ~40% dense-GEMV ceiling** (phase-0 profile: 45% of Q4_K decode cycles are libomp barrier at 96t; halving the barrier → +22%).

**No post-iqk *canonical* decode number exists yet** — the clean post-reboot llama-bench is still pending (`v6-iqk-promotion.md` Phase J). Closest datum: eval-parity throughput byproduct **38.46 (iqk-on) vs 27.78 (iqk-off) t/s** on worker_general, N=206 AA-Omniscience — an eval-run observation, not the canonical recipe. iqk is **neutral on Q8_0 decode** (its wins are prefill + Q4 decode), so it is *not* the frontdoor's decode lever.

**Highest-ROI CPU avenue**: **frontdoor Q8_0 barrier-count reduction via operator/graph fusion** (fuse expert gate+up, fuse the attn QKV cluster) — owner `cpu-shape-specialized-gemv-decode.md` (its Session-15 barrier profile is the landing zone). Est +10–15% decode; one fusion cluster already measured +2.6%. Cheapest test: llama-bench tg128 frontdoor Q8_0, fusion flag on/off, same window. Secondary: iqk prefill src1-fusion (232→~252 pp512) + widen the iqk MoE hook to the larger-MoE experts (122B-A10B / 80B) — **prefill/TTFT only, decode neutral** — owner `iqk-port.md` (already its deferred secondary wins). **Deprioritize** worker decode micro-kernels (near the plateau) and any new SIMD ukernel (BW-bound; roofline says the ALU is waiting on DRAM).

---

## 4. MI210 roofline + the direct answer on dequant

**Two distinct gaps** (consistent GiB/GiB basis; the log's GiB-over-decimal-TB mixing slightly deflates the quantized rungs):
- **GAP-A — dequant** (batch-1): Q4_K **34%** → Q8_0 **50%** → fp16 **62.5%**. ~**28 pp** lost to Q4_K dequant, ~**12 pp** to Q8_0. Mechanism: MMQ dequant throughput + VGPR pressure; MFMA idle at batch-1 GEMV; no tuned int8/fp8 path on gfx90a (OCP fp8 needs ROCm ≥6.3; we run 6.2).
- **GAP-B — batch-1 latency** (fp16 still only 62% llama / 69% vLLM): latency-bound GEMV, too few waves-in-flight to hide HBM latency. **62% is already near the practical batch-1 ceiling** — above vLLM-on-H100's ~50% and near Hazy's ~78% single-dispatch ceiling; the recoverable slice is small/speculative and **no ROCm megakernel exists** (Hazy/Mirage are CUDA-only).

**The reframe that answers the operator's "even Q8/fp16 only gets ~60%":** that 60% is a **batch-1 artifact**. Weights are read once per forward step and reused across the batch, so at batch-32 the weight-BW utilisation *falls to ~28%* while token throughput *rises 14.6×* (62.45 → 909.8 t/s). Under concurrent production serving the card **leaves the weight-BW-bound edge and climbs the compute/occupancy roof** — "% of weight-BW roofline" is the wrong axis at batch>1. Production already delivers ~910 tok/s @32-way; the batch-1 60% is not a loss under our actual workload.

**Can the dequant slowdown be compensated? YES — and the cheapest, highest-recovery method is a config change, not a kernel.**
1. **Batched/concurrent serving (`-np 8–32`)** — free, dominant, for *throughput* roles. The MMQ weight tile is dequantized once and reused across B columns (per-token dequant ÷B) **and MFMA engages** on the dequantized tiles at batch>1 (exactly what `GGML_HIP_MMQ_MFMA` is for). Expected: GAP-A (~28 pp Q4, ~12 pp Q8) largely closes for throughput roles — already realized at fp16 (62→910 t/s). **A custom dequant kernel is unnecessary for frontdoor/worker-style serving.**
2. **Custom gfx90a Q4_K dequant-GEMV kernel** — only worth it for a **batch-1 *latency* role** (e.g. a GPU drafter / interactive path). Author via the GEAK/agentic-rocm loop (LDS-staged dequant tiles, better VGPR allocation for occupancy, CDNA2 dual-issue). Expected: ~half the ~28 pp Q4 gap → Q4 33% toward ~45–50%. Effort M.
3. **Not** via AITER (gfx90a unsupported), **not** via fp8 (needs ROCm ≥6.3), **not** via MFMA at batch-1 (GEMV leaves matrix cores idle).

**Two caveats that make measurement mandatory before any kernel work**: (1) the ~910 batched number is **fp16 Qwen3-8B — the *quantized* batched sweep was never measured on MI210**, so "batching closes GAP-A for quantized" is sound reasoning, not a measured MI210 result; (2) **MoE batches worse than dense** — distinct tokens hit distinct experts, so expert weight traffic grows with batch (frontdoor/worker will amortize less cleanly than the dense 8B).

---

## 5. Smallest decisive measurements (kernel) — operator-runnable, ranked
1. **MI210 quantized `-np {1,2,4,8,16,32}` sweep** on Q4_K gemma4-31B + Q8_0 Qwen3.6-27B — does batching close GAP-A for a *quantized MoE*? This forks the entire "build a dequant kernel?" question. (config only; one attested window)
2. **rocprof the batch-1 Q4_K MMQ kernel** — confirm it is VALU/issue-bound (not HBM-saturated) *before* authoring any kernel; if HBM-saturated, a dequant kernel buys nothing.
3. **CPU frontdoor Q8_0 gate+up fusion A/B** (llama-bench tg128, same window) — the top CPU decode lever.
4. **The pending canonical post-reboot CPU decode bench** (`v6-iqk-promotion.md` Phase J) — the first *decision-gating* post-iqk number; everything above it is currently an observation.
5. **Quantized-vs-quantized vLLM head-to-head** — the one gap the fp16 test left open (is llama.cpp's quantized gfx90a path behind vLLM's, or does the fp16 62%/69% gap not reproduce under quant?).

## 6. Ranked cross-substrate kernel actions (the ROI answer)
1. **MI210 — serve quantized BATCHED (`-np 8–32`)** instead of authoring a kernel. Effort XS (config). Proof: measurement #1. Owner: partial (batched-decode harness in `batched-decode-measurement.md`; MI210 quantized-batched sweep is only an open bullet in the mi210 progress note — needs an owner).
2. **CPU — frontdoor Q8_0 barrier-count reduction (op/graph fusion).** Est +10–15%. Effort L-M. Owner: `cpu-shape-specialized-gemv-decode.md`.
3. **MI210 — tuned gfx90a Q4_K dequant-GEMV kernel via GEAK**, *batch-1 latency roles only* (GPU drafter). ~half the 28 pp Q4 gap. Effort M; deprioritized while batching covers throughput. Owner: `agentic-rocm-kernel-authoring.md`.
4. **CPU — iqk secondary prefill wins** (src1-fusion + larger-MoE hook). Prefill/TTFT only. Effort M. Owner: `iqk-port.md`.
5. **MI210 — HIP-graph capture** to remove launch bubbles (GAP-B, batch-1 latency). Effort M. Owner: none (mi210 progress open item).
- **Deprioritize**: CPU worker decode micro-kernels (near plateau) + a full ROCm megakernel port (CUDA-only references; batching dominates the ROI).

## 7. Reporting — applied changes
All edits above are on branch `spec-dec-mtp-refresh-2026-06-22` and are **uncommitted** (harness policy: commit only on operator request; no codex agent is currently running, so the shared-tree ride-along hazard is low right now). Files touched: the 11 handoffs/indices in §1 + this findings doc + a progress note. **Not yet folded** (P3 tail, low-cost, listed for a follow-up pass): agent-world deep-body DGX references (headline fixed; Phase-2 body still names DGX in ~6 places), `intake-460` Splitwise as a tracked Family-D experiment, the Ornith-1.0 decode-swap A/B row, and a cross-model-LoRA watch stub — none is load-bearing.

## 8. Self-critique
- **Every throughput number is an OBSERVATION** — throttle-suspect 28-day CPU host, single-run contended MI210 host, no protocol-id. None may gate a keep/revert/deploy decision (MEASUREMENT.md). The one decision-gating CPU number (canonical post-iqk bench) does not exist yet.
- **The "batching closes the quantized dequant gap" claim is architecturally reasoned, not measured on MI210** (fp16-only), and MoE batches worse than dense — so measurement #1 is a genuine go/no-go, not a formality.
- **The CPU realistic-ceiling (~25–30%)** is an estimate extrapolated from a dense reference, not an MoE-measured bound — the frontdoor "~49% of ceiling" headroom rides on it.
- **The sweep is index/intake-grounded, not full-handoff-grounded** — a deciding fact wrong *in an intake entry* propagates; the applied un-stalings were each verified against the owning handoff's actual gate line, but the P3 folds were not.
- **What would most change the recommendations**: measurement #1 showing quantized MoE *not* amortizing under batch (→ the gfx90a dequant kernel jumps to high-ROI for throughput too, not just latency); the canonical CPU bench showing iqk decode-neutral on the worker as well (→ the CPU story becomes prefill-only + the frontdoor-fusion lever is the *only* decode lever).

## Progress checklist

- [x] Intake-sweep findings deliverable produced (edits applied across 11 handoffs) ✅
