# CPU25 — Decision

**Verdict**: **CLOSED — DECISIVE NEGATIVE on single-socket NPS4 (MoE proxies tested)**.

## What was decided

Per-NUMA-node weight replication (NUMA_MIRROR) does NOT deliver throughput on this hardware. Phase 2 throughput gate (≥ +25% on Coder-30B Q4_K_M over 47.98 t/s baseline) **NOT MET**:

- Coder-30B Q4_K_M tg128: 48.16 → 47.66 (−1.0%, within noise)
- Qwen3.6-35B Q8_0 tg64: 23.30 → 23.45 (+0.6%, within noise)
- Backfill re-confirmation Coder-30B Q4_K_M tg64: 47.26 → 46.97 (−0.6%, within noise)

PPL is bit-exact (11.1215 ± 0.62430 on Coder-30B Q4_K_M wiki chunks 1-12, byte-identical to znver5 baseline). The implementation is mathematically correct and the mirror correctly fires + threads correctly distribute across nodes; the negative is throughput-only.

## Root cause

Single-socket NPS4 EPYC 9655 is **DRAM-channel-bound**, not fabric-bound, at 96-thread saturation:

- Total DRAM bandwidth: 460 GB/s (12 channels × DDR5-6000)
- Per-thread share at 96t: 460 / 96 = 4.79 GB/s/thread
- With mirror, each NPS4 node has 3 channels (115.2 GB/s) and 24 threads → 115.2 / 24 = 4.79 GB/s/thread — IDENTICAL

Mirroring shifts cross-NUMA reads to local reads, which would help only if the **fabric** were the binding constraint. CPU24's perf-record at 96t showed compute kernels stalled on memory loads at 4.79 GB/s/thread, but that measurement could not distinguish fabric-stall from DRAM-channel-stall. Phase 1c cleanly rules out the fabric-stall hypothesis on this hardware.

The vproxy-tools fork's reported gains (+62% QwQ-32B FP16, +34% DeepSeek-R1 671B Q8) were on **two-socket** 2× EPYC 9275F configurations where cross-SOCKET fabric IS the binding constraint.

## What was NOT decided (gates that remain open)

- **Dense/hybrid (Qwen3.5/3.6-27B) coverage** (peer review finding #11): per-thread BW math says dense should fail the +25% gate identically, but empirical confirmation is missing. Phase 2.6 of remediation closes this gap.

That is the ONLY remaining gap; the measurement on the two MoE proxies is honest and complete.

## Closure scope

**Closed**: Phase 2 throughput gate fails on Coder-30B Q4_K_M MoE and Qwen3.6-35B Q8_0 MoE on single-socket NPS4. Hardware is DRAM-channel-bound. Production stack should NOT enable `GGML_NUMA_MIRROR`.

**NOT closed (in spirit, narrow)**: dense/hybrid generalization. Closes after Phase 2.6.

**Reopen rule**: only if (a) deployment shifts to a 2-socket configuration, OR (b) Phase 2.6 dense run finds an unexpected dense-only win.

## Implementation preserved as zero-overhead infrastructure

The accessor migration (Phase 0a/0b: `tensor_data()` / `tensor_set_data()` replacing 164 raw `tensor->data` reads across 11 files) is preserved in the codebase. In default builds (no `GGML_NUMA_MIRROR`), `tensor_data()` compiles to direct field access — pure no-op.

The Phase 1c CPU_REPACK buffer-mirror code in `ggml/src/ggml-cpu/repack.cpp` is preserved compile-flag-gated for future hardware where fabric IS the binding constraint (2-socket EPYC, multi-package, GPU-CPU NUMA).

## Implications for the broader CPU optimization picture

The original framing of "this was the largest concrete throughput lever from CPU24's analysis; if even per-NUMA-node weight replication doesn't deliver, the software-level CPU optimization runway on this hardware is materially exhausted" was **over-broad** — closure inflation per peer review.

**Narrowed framing**: the per-NUMA-node-weight-replication and per-CCD-mbind levers are falsified for single-socket NPS4. Multiple software levers remain open:

- libomp completion (CPU21 Phase 2.1)
- CPU22 work-stealing prototype (Phase 3)
- CPU23 interference + 5-model coverage (Phase 2.2)
- MoE-Spec algorithmic spec-dec verification budgeting
- ZenDNN, PGO/LTO, BOLT/FDO, prefill optimizations, parallel-slot benchmarking (CPU6/11/12/13/14, future-track triage in Phase 4)

See `feedback_closure_inflation.md` memory for the recurring failure mode this remediation addresses.

## Remediation reference

See `~/.claude/plans/nifty-discovering-allen.md` Phase 2.6 for the dense/hybrid sanity probe.
