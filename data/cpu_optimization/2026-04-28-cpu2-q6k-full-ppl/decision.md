# Q6_K AVX-512BW Full 32-chunk PPL Gate — Decision

**Verdict**: **PASSED**.

## What was decided

The Q6_K AVX-512BW 8x8 SIMD kernel (`gemv_q6_K_8x8_q8_K_avx512bw` in `ggml/src/ggml-cpu/arch/x86/repack.cpp`, ~95 lines of intrinsics, env-gated `GGML_Q6_K_8X8_AVX=1`) is **mathematically correct on the full WikiText-2 32-chunk PPL gate**:

- **Coder-30B Q4_K_M**: PPL = 8.2622 ± 0.27495 in both env=0 (generic) and env=1 (SIMD). All 32 chunks byte-identical between the two runs.
- **REAP-246B Q4_K_M**: PPL = 8.1396 ± 0.24168 in both env=0 and env=1. All 32 chunks byte-identical.

This closes the cleanup item from CPU2 Session 17 ("Full 32-chunk WikiText-2 PPL gate is a follow-up validation step before flipping the env default; the 3-chunk match is sufficient to confirm there's no catastrophic correctness bug"). The 3-chunk match held; the 32-chunk match also holds.

## Closure scope

**Closed**:
- Q6_K AVX-512BW SIMD kernel correctness on Coder-30B Q4_K_M and REAP-246B Q4_K_M at the full 32-chunk WikiText-2 PPL gate.
- The env flag `GGML_Q6_K_8X8_AVX` graduates from "default-OFF until full PPL gate" to "production-ready opt-in".

**NOT in scope of this gate** (intentionally):
- Throughput delta on dense/hybrid (Qwen3.5/3.6-27B Q8_0 — note: Q6_K kernel doesn't fire on Q8_0 weights; dense throughput delta for CPU2 is the Q8_0 SIMD + prefetch combination, covered separately in Phase 2.6).
- Throughput retest on Coder-30B / REAP-246B (the Session 17 / Session 18 measurements stand: +0.4-0.5% SIMD-alone, +0.7% with T1 prefetch on Coder-30B; both within the BW-bound ceiling identified by CPU24).

## Followup actions

1. **`cpu-kernel-env-flags-inventory.md`**: flip `GGML_Q6_K_8X8_AVX` row from "default-off pending PPL gate" to "production-ready opt-in (PPL bit-exact on Coder-30B + REAP-246B 32-chunk WikiText-2)". Mark v5 cherry-pick candidate.
2. **`cpu-shape-specialized-gemv-decode.md`**: remove the "full 32-chunk PPL pending" caveat; reference this artifact bundle.
3. The Q6_K T1 prefetch (`+0.7%` on Coder-30B from Session 18) shares the same code path; it's already validated bit-exact at 3 chunks and the same 32-chunk PPL gate covers both since `GGML_Q6_K_8X8_AVX=1` engages both.

## Remediation reference

This is Phase 2.4 of `~/.claude/plans/nifty-discovering-allen.md`. Phase 2.6 will add Qwen3.5/3.6-27B Q8_0 throughput delta for CPU2 SIMD + prefetch (Q8_0 path, not Q6_K) closing the cross-architecture coverage gap (peer review finding #11).
