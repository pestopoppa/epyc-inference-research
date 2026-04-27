# Q6_K AVX-512BW SIMD — Full 32-chunk WikiText-2 PPL Gate

**Track**: CPU2 — Shape-specialized GEMV decode (Q6_K kernel; [handoff](../../../../../workspace/handoffs/active/cpu-shape-specialized-gemv-decode.md))
**Run date**: 2026-04-28 (remediation Phase 2.4)
**Purpose**: cleanup item from CPU2 Session 17. Earlier session validated bit-exact PPL on a 3-chunk WikiText-2 probe; the handoff explicitly listed "Full 32-chunk WikiText-2 PPL gate" as a follow-up validation step before flipping the env default. This bundle delivers that gate.

## Verdict

**PASSES**. Q6_K AVX-512BW 8x8 SIMD kernel produces byte-identical PPL to the generic reference at 32 chunks on both Coder-30B Q4_K_M and REAP-246B Q4_K_M.

| Model | Quant | env=0 (generic) PPL | env=1 (SIMD) PPL | Match |
|---|---|---|---|---|
| Qwen3-Coder-30B-A3B | Q4_K_M | **8.2622 ± 0.27495** | **8.2622 ± 0.27495** | ✅ bit-exact, all 32 chunks identical |
| Qwen3-Coder-REAP-246B-A35B | Q4_K_M | **8.1396 ± 0.24168** | **8.1396 ± 0.24168** | ✅ bit-exact, all 32 chunks identical |

The kernel is **production-ready**. The env flag `GGML_Q6_K_8X8_AVX=1` should be marked production-ready opt-in in `cpu-kernel-env-flags-inventory.md`.

## Commands run

Binary: `/mnt/raid0/llm/llama.cpp-experimental/build/bin/llama-perplexity` at HEAD `29a69599a` (default-flags build; Q6_K SIMD body and dispatcher land in this build because `GGML_Q6_K_8X8_AVX` is a runtime env, not a compile flag).

Wrapper: `LD_LIBRARY_PATH=$EXP/build/bin GGML_Q6_K_8X8_AVX={0,1} taskset -c 0-95 numactl --interleave=all`.

```bash
# Coder-30B Q4_K_M
./bin/llama-perplexity -m Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf \
  -f /mnt/raid0/llm/data/wiki.test.raw -t 96 -fa 1 --chunks 32 --no-mmap

# REAP-246B Q4_K_M
./bin/llama-perplexity -m Qwen3-Coder-REAP-246B-A35B-Q4_K_M.gguf \
  -f /mnt/raid0/llm/data/wiki.test.raw -t 96 -fa 1 --chunks 32 --no-mmap
```

Two runs per model — one with `GGML_Q6_K_8X8_AVX=0` (forces generic Q6_K vec-dot), one with `GGML_Q6_K_8X8_AVX=1` (engages the AVX-512BW 8x8 SIMD kernel).

## Files in this bundle

| File | Purpose |
|---|---|
| `coder30b_env0_ppl32.log` | Coder-30B Q4_K_M, GGML_Q6_K_8X8_AVX=0, full perplexity output |
| `coder30b_env1_ppl32.log` | Coder-30B Q4_K_M, GGML_Q6_K_8X8_AVX=1 |
| `reap246b_env0_ppl32.log` | REAP-246B Q4_K_M, GGML_Q6_K_8X8_AVX=0 |
| `reap246b_env1_ppl32.log` | REAP-246B Q4_K_M, GGML_Q6_K_8X8_AVX=1 |
| `system-state.txt` | numactl + numa_balancing + THP + governor + SMT + uptime + free + hugepages |
| `process-pre.txt`, `process-post.txt` | pgrep snapshots |
| `ld_debug.log` | LD_DEBUG=libs trace of one smoke command on the default-flags build |
| `results.csv` | tabulated chunk-by-chunk + final PPL per model × env |
| `decision.md` | explicit verdict |

## Followup actions

1. Flip `GGML_Q6_K_8X8_AVX` to "production-ready opt-in" in `cpu-kernel-env-flags-inventory.md`.
2. Update `cpu-shape-specialized-gemv-decode.md` to remove the "full 32-chunk PPL pending" caveat for Q6_K SIMD body.
3. Phase 2.6 will add Qwen3.5/3.6-27B Q8_0 throughput delta for the CPU2 Q8_0 SIMD + prefetch (CPU2 Q8_0 has its own pre-existing 32-chunk PPL bit-exact validation; Phase 2.6 is throughput-only on dense).

Note: the Q4_K T1 prefetch revert from Session 18 (-4% on Coder-30B Q4_K_M) is unaffected by this gate; that decision was about throughput, not correctness.
