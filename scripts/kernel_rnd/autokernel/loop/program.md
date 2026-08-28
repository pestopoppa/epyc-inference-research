# AutoKernel loop — strategy

The loop's shape is normative and lives in
`docs/guides/agent-workflows/agent-loop-design.md` (epyc-root). This file is the
strategy the loop runs *inside* that shape: what to attack, what the hardware makes
possible, and what has already been settled.

---

## The loop

```
when the champion changes (at most weekly otherwise):
    rocprofv3 the champion → ranked hotspots

each iteration, planner works in a worktree with the full toolbox:
    reads   champion · experiments.md · hotspots · hypotheses/inbox/
    probes  FREELY — llama-bench, rocprofv3, test-backend-ops -o OP --perf,
            llvm-objdump for VGPR/occupancy, env-flag sweeps. Nothing gated.
    forms   hypothesis H, backed by evidence it gathered itself

CRITIC PASS 1 on H, before any patch exists     · budget 3 rounds
    reject → reason returned VERBATIM; planner refines or regenerates
CRITIC PASS 2 on the committed diff, before the build   · budget 2 rounds
    reject → reason returned VERBATIM; H untouched, planner rewrites the patch

    build → test-backend-ops → A/B alternating, n≥5   ← the only GPU spend
    keep → commit onto the champion branch
    else → negative, with mechanism and sample vector, into experiments.md
```

---

## What the instrument can actually resolve

Measured 2026-08-28, n=20 alternating A/A pairs, residency proven on 80/80
invocations (`artifacts/autokernel-aa-noise-floor/`):

| pairs | prefill p95 \|effect\| | decode p95 \|effect\| |
|---|---|---|
| 1 | 2.175% | 3.452% |
| 5 | 0.753% | 1.848% |
| 9 | 0.442% | 1.502% |
| 20 | 0.182% | 1.175% |

**4 of 20 pure-noise decode pairs already exceeded a 3% bar.** So: `n≥5`, and never
claim an effect smaller than the floor for the pair count used. Decode has heavier
tails and converges slowly; prefill is the cheaper surface to detect on.

## The workload

`DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf` — n_embd 1536 (divisible by the
256-element K-quant superblock), **Q4_K ×169, Q6_K ×29**. That is production's
dispatch path at ~1 GB.

The superseded workload was `Qwen2.5-Coder-0.5B-Q4_K_M.gguf`, which despite its name
is **Q5_0 ×132, Q4_K ×12**: n_embd 896 is not divisible by 256, so llama.cpp fell
back silently and a month of screening measured a kernel production never dispatches.
`workload_contract.verify_workload` now refuses that class outright. **Never trust a
filename; census the tensor table.**

## The build recipe

`controller/build_recipe.HOUSE_GPU_RECIPE`, versioned, with every flag naming its
production counterpart. A flag that diverges without a stated reason is refused at
construction. `GGML_HIP_ROCWMMA_FATTN=ON` is not optional on gfx90a: the CMake
default OFF produces non-finite values at longer sequence lengths under `-fa on`, and
a short smoke test hides it because prompt length is the discriminator.

---

## Where to attack

Generate against the **live** profile of the current champion, not a frozen list.
Every accepted patch moves the distribution.

Standing seeds, carried because they are expensive to rediscover rather than because
they are ranked:

- **IQ4_XS / Q5_K and the 64-VGPR occupancy knee.** Every rung ≤64 VGPR (8 waves)
  decodes ≥90 t/s; both rungs above it (6 waves) decode ≤83 t/s *while 27–46%
  smaller*. IQ4_XS sits exactly on the boundary and is the fastest rung. Any IQ2/IQ3
  lever must carry an explicit VGPR target — a reduction landing at 70 buys nothing.
- **`iqk_gemm_1bit.cpp` and `iqk_flash_attn.cpp`** — vendored, audited clean
  2026-07-29, omitted from CMake, never staged.
- **The hypothesis inbox** (`<store>/inbox/*.md`, re-read every iteration): hypotheses
  arrive asynchronously — from the operator, or harvested from the backlog by
  `python3 -m autokernel.loop.seed`. Dropping a file in never blocks the loop.

## Measured gfx90a facts — check a mechanism against these before proposing it

These are receipts, not priors. A hypothesis that contradicts one is dead on arrival.

- **gfx90a LDS is 32 banks with 8 phase cliques** — *not* the CDNA3 answer (64 banks,
  2 phases of 32 lanes). Solved on our own silicon 2026-08-11 (372 bank / 6,048 phase
  dispatches; receipt SHA-256 `ae1d833c…`). Any swizzle constant lifted from CDNA3
  work — HipKittens' `>>7 <<3` among them — **does not transfer unexamined**.
- **`__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8` does not exist on gfx90a.** Of the six
  `__builtin_amdgcn_*` intrinsics HipKittens uses across 67 headers, exactly this one is
  unavailable. Any fp8-MFMA mechanism is unimplementable here, not merely unmeasured.
- **Our MFMA fragment layout is already the standard one.** `ggml/src/ggml-cuda/mma.cuh`
  `tile<16,16>` (`:127,144` — `get_i = tid%16`, `get_j = 4*(tid/16)+l`, `ne=4`) is
  **bit-identical** to HipKittens' `rt_base`. Fragment-level techniques from that
  literature compose onto our existing tiles with **zero layout re-derivation** — this
  is an enabling fact, and it means "we would have to re-derive the layout" is not a
  reason to decline one.
  (Source for all three: `handoffs/active/agentic-rocm-kernel-authoring.md`.)

## What the correctness gate does NOT catch

`test-backend-ops` is the gate, and it is necessary rather than sufficient. A real
corruption has already passed a green test suite here: `b8ad9d292` incidentally
reclassified Q2_K/Q3_K activations `Q8_2_X4` → `Q8_K`, which engaged never-validated
iqk kquant kernels and corrupted Hy3 output. It was **caught on live output at 20:47Z,
not by tests** (fixed same day by `1977a5d78` plus static_asserts).

So: a patch that changes an activation quant type, a type-traits row, or a dispatch
table entry is in a known-dangerous class whose failure mode the gate cannot see. Say
so in the hypothesis, and prefer a mechanism that leaves type selection alone.

## Not this loop's surface — do not propose these

The backlog is full of measured levers that are NOT kernel-source patches. They are
real work; they are simply not what this loop can build, gate and A/B:

- **Serving and sampling flags** — spec-dec draft depth, `--spec-type ngram-*`,
  `-ctk/-ctv` KV quant, batch/ubatch geometry. Config, not source.
- **Drafter work** — MTP/EAGLE3 heads, tree-draft, drafter fine-tuning for acceptance
  rate. A separate track with its own handoffs.
- **Lossy model surgery** — vocabulary pruning, layer/depth removal. Out of scope by
  the ratified lossless-only rule, and both contributed to a 15/40-point degradation.
- **KT / trellis quants** — CPU-only, so the MI210 cannot participate at all. Also
  dominated on the arithmetic that motivated them: `block_iq2_kt` is 68 B per 256
  weights = **2.125 bpw** against **IQ2_XXS at 2.0625 bpw**, so it is *larger* than
  what GLM-5.2 already uses and saves zero bandwidth while adding per-weight work. A
  quality-at-equal-bpw play mis-framed as a speed play.

## Settled — do not re-open without new evidence

- **`MMQ_MFMA` OFF.** +23.09% on the 0.5B toy, **+0.50%** on Qwen3.8-27B. Real where
  it was taken, worth nothing where the fleet runs.
- **`ubatch 512→1024`.** A NULL ARM: llama.cpp clamps `n_ubatch = min(n_batch,
  n_ubatch)`, so both arms ran at 512 on one identical binary. The +46.9% was a
  bimodal sample whose median landed on the fast mode.
- **ngram 2.8×.** Retracted — a warm-context self-copy artifact; −17.4% on 122B-IQ2.
- **Already in v9:** `GGML_IQK` (since v8), MMQ `a6b4b5263`, HIP graphs (upstream
  default ON).

## Authority

This package **screens**. It never promotes. Every result is non-promotable by
construction, and promotion is `docs/reference/kernel-freeze-runbook.md` — seven
steps, ~100 lines, and it shipped v7, v8 and v9.

Under `P-AK-SEARCH-1` denial 4, prior records inform **hypothesis formation** only;
`experiments.ExperimentStore.recall(ranking_authorized=False)` is the default until
the operator amends it (decision D1,
`handoffs/active/autokernel-rebuild-program.md`). Cross-epoch records are returned but
marked `stale_epoch` — the fact that a mechanism was tried is formation; its *number*
is not comparable.
