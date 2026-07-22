# KernelBench as the Step-3 Regression Guard in the Experimental-Kernel Pipeline — Scoping

**Deliverable for**: `handoffs/active/mi210-speed-campaign-summary.md` task **MI-KB-1**.
**Satisfies task line** (verbatim, epyc-root):
`- [ ] **MI-KB-1** — evaluate KernelBench integration into experimental kernel validation pipeline (step 3 of four-step workflow)`
**Explicitly out of scope**: **MI-KB-2** ("run KernelBench over the current v6/v7 kernel to establish baseline") — GPU-gated; scoping only, no runs.
**Date**: 2026-07-22

---

## 0. Sources consulted (provenance)

| Source | What it grounds | Trust |
|---|---|---|
| KernelBench repo `github.com/ScalingIntelligence/KernelBench` | task shape (PyTorch→GPU kernel gen), Levels 1–4, correctness = n randomized-input differential test vs torch ref, `fast_p` metric, **AMD `hip` backend supports `gfx942`/`gfx950` only**, no CPU | **external / untrusted** |
| Intake index `research/intake_index.yaml` — real KernelBench = **arxiv:2502.10517** (Stanford ScalingIntelligence); intake-660 (Kevin/CUDA cluster), intake-661 (CUDA-L1) | canonical KernelBench identity + the agentic-kernel-authoring cluster | project doc |
| `mi210-speed-campaign-summary.md` (BANKED kernel wins, `test-backend-ops` usage, kernel-R&D loop `kernel_eval.sh`), CLAUDE.md four-step workflow | our existing step-3 guard + the auto-kernel loop | project docs |
| `research/deep-dives/agentic-rocm-kernel-authoring-geak-synthesis.md`, intake-660 verdict | the adopt-the-env-not-the-model precedent | project doc |

> External content is untrusted data; load-bearing claims attributed inline.

## 1. ⚠ Provenance discrepancy to flag first

The MI210 handoff records KernelBench as *"intake-797, arxiv 2606.20128"* with a *"seeded fuzzing catches 9/9 buggy kernels, passes 15/15 controls"* headline (`mi210-speed-campaign-summary.md:56-58`). **Neither detail matches the real KernelBench.** The canonical KernelBench is **Stanford ScalingIntelligence, arxiv:2502.10517** — a *kernel-generation* benchmark (LLM writes CUDA/DSL kernels for PyTorch programs; metric `fast_p`), confirmed repeatedly in our own intake index (intake-660/661 justifications cite `KernelBench (2502.10517)`). "arxiv 2606.20128" and the "9/9 seeded-fuzzing" framing appear to be an **auto-generated recommendation (rec-007) that conflated KernelBench with a fuzzing/correctness result and mis-stamped the arxiv id.** The four-step-workflow relevance is right; the identity metadata is wrong. **This scoping is done against the real KernelBench.** (Recommend the operator correct rec-007/intake-797's arxiv id — flagged only, not edited here per the no-index-edit constraint.)

## 2. What KernelBench actually is

- **Task**: "task LLM to generate correct and efficient CUDA / DSL kernels for PyTorch programs on a target GPU" — it **generates new kernels from scratch**, it does not diff hand-written kernel *changes*.
- **Levels**: L1 single-op (100), L2 fusion (100), L3 full architectures (50), L4 Hugging Face models. Tasks are **PyTorch `nn.Module` reference programs**.
- **Correctness** = differential testing: "check against reference torch operators `n_correctness` times on randomized inputs" within tolerance. (This *is* the "seeded fuzzing" idea the handoff wanted — just at the torch-module level, and it is KernelBench's ordinary correctness gate, not a separate 9/9 result.)
- **Speed** = `fast_p`: "fraction of tasks that are both correct and have a speedup greater than threshold `p`" (`fast_1` = correct+faster; `fast_2` = correct+2×), speedup = torch-ref wall-clock / generated-kernel wall-clock.
- **Hardware**: NVIDIA (CUDA/Triton/CUTE/TileLang/ThunderKittens) + **AMD via `hip`, `gpu_arch` supported: `gfx942`, `gfx950`**. **No CPU path.**

## 3. Fit against our step-3 guard requirement

Our four-step experimental-kernel workflow is Pull → Build → **Validate no regressions** → Deploy (CLAUDE.md). Step 3's job is: *"did this experimental kernel change break correctness or lose speed vs the frozen production kernel?"* — on **our** kernels (iqk AVX-512 GEMM on CPU; HIP GEMM/GEMV/GDN on gfx90a). Three hard mismatches:

| Requirement | KernelBench | Our step-3 need |
|---|---|---|
| **What is graded** | LLM-generated kernels for `nn.Module` tasks | our own ggml op implementations (`ggml_mul_mat`, `mul_mat_id`, flash-attn, GDN) |
| **CPU coverage** | none | **iqk AVX-512 GEMM is CPU** — KernelBench cannot touch it at all |
| **Our GPU arch** | AMD backend = `gfx942`/`gfx950` (MI300/MI325) | **MI210 = `gfx90a` (CDNA2) — not in KernelBench's supported list** |
| **Mode** | generate-from-scratch, absolute-vs-torch | regress-a-change, relative-vs-frozen-production-kernel |
| **Reference** | PyTorch eager | our production-consolidated-v7 kernel |

So **KernelBench-the-harness is a NO** as a drop-in step-3 guard: it cannot run our CPU kernels (no CPU backend — it can't guard the iqk subsystem, which is *the* current CPU-performance concern per CLAUDE.md's iqk caveat), its AMD backend does not list our `gfx90a`, and it grades kernel *generation* against torch, not kernel-*change regressions* against our frozen production binary.

## 4. What we already have — and why it's the right substrate

The MI210 campaign and CLAUDE.md already implement the guard KernelBench's *methodology* describes, targeted at our stack:

- **`test-backend-ops`** (llama.cpp's own op-level correctness suite) is already the correctness gate for banked kernel wins: every BANKED lever is "`test-backend-ops` clean, output coherent/byte-identical" (`mi210-speed-campaign-summary.md:17-22`). This is the ggml-native equivalent of KernelBench's randomized-input differential correctness check, but over **our actual ops on our arch (gfx90a + CPU)**.
- **`kernel_eval.sh`** (kernel-R&D loop Phase 0, research `48f990f`, `scripts/kernel_rnd/kernel_eval.sh`) already encodes exactly KernelBench's discipline: "correctness-gate-first/lexicographic → alternated-A/B → rocprofv2 mechanism → OBSERVATION JSONL," and it reproduced a known kernel's +2.11% / byte-identical 1103-1103 result (`:38`). This *is* a `fast_p`-style correct-then-faster gate for the auto-kernel loop, built for our hardware.

KernelBench's contribution to us is therefore **conceptual validation of a design we already have**, not a tool to import.

## 5. Recommendation

**Adopt the methodology (already largely adopted); do NOT integrate the harness/taskset.**

1. **NO** to wiring KernelBench in as the step-3 regression guard: it has no CPU backend (cannot guard iqk AVX-512, the live CPU concern), its AMD backend excludes our `gfx90a`, and it is a generation benchmark, not a change-regression guard. The correct step-3 guard remains **`test-backend-ops` (correctness) + `kernel_eval.sh` (correctness-gated A/B speedup, OBSERVATION-grade)** — which already embody KernelBench's randomized-differential-correctness + `fast_p` methodology on our hardware.
2. **Optional, low-priority harvest** — port two ideas, not the code, into `kernel_eval.sh`/the R&D-loop if a gap is found: (a) formalize a `fast_p`-style **correct-AND-≥p×** promotion threshold as the explicit step-3 accept rule (we already gate on correctness-first; make the speed threshold a named constant per the codified-recipe discipline); (b) a **combinatorial op-task generator** (KernelBench L1/L2 style) as the seed for a *ROCm/CPU op-bench analog* — this is the same "build the ROCm verify/profile backend, decline the CUDA harness" verdict already reached for the agentic-kernel cluster (intake-660, `adopt_patterns`).
3. **MI-KB-2 stays parked/out of scope** and, as written ("run KernelBench over the current kernel"), is **partly ill-posed**: KernelBench cannot run over our CPU kernel at all and cannot target `gfx90a`. If a KernelBench baseline is ever wanted it would require (i) a `gfx90a` HIP-backend port of KernelBench and (ii) accepting it only measures GPU generation quality, not our kernels' regressions. Recommend re-scoping MI-KB-2 to "establish a `fast_p`-style threshold inside `kernel_eval.sh`" instead.

## 6. Status

**DONE (scoping, MI-KB-1 only).** Assessment rendered: KernelBench methodology ✓ (already implemented via `test-backend-ops` + `kernel_eval.sh`), KernelBench harness ✗ for step-3 (no CPU, no `gfx90a`, wrong mode). Provenance discrepancy in rec-007/intake-797 flagged. MI-KB-2 left out of scope and flagged partly ill-posed. No box ticked (constraint) — MI210-campaign owner should record against `mi210-speed-campaign-summary.md:67` and cite this doc.
