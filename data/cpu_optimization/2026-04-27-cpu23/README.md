# CPU23 — Context-Regime Coverage Matrix (artifact bundle)

**Track**: CPU23 — Context-Regime Coverage Matrix ([handoff](../../../../../workspace/handoffs/active/cpu-context-regime-coverage.md))
**Run date**: 2026-04-27
**Backfill date**: 2026-04-27 evening (this README + system-state.txt + process-pre/post.txt + ld_debug.log + results.csv + decision.md added retroactively per CPU20 artifact-bundle-backfill policy)

## Scope of what was actually run

This is a **partial probe**. The handoff's Required matrix (line 14-26) requires:
- 4 regimes per target model class: 2K, 8K, 32K, **long-prompt-mid-stream interference**
- 4 metrics per regime: generation t/s, **TTFT**, **decode stall fraction**, **per-iteration latency variance**
- 5 production models in the target set (Qwen3.6-35B Q8_0, Coder-30B Q4_K_M, Qwen3-Next-80B Q4_K_M, REAP-246B Q4_K_M, gemma-26B Q4_K_M)

**What ran on 2026-04-27**:
- 3 of 4 regimes: 2K, 8K, 32K (NO long-prompt-mid-stream interference scenario)
- 1 of 4 metrics: pp+tg32 throughput only (NO TTFT, NO decode stall fraction, NO per-iteration latency variance)
- 2 of 5 models: Coder-30B Q4_K_M (sync-bound MoE proxy), Qwen3.6-35B Q8_0 (BW-bound MoE proxy). NOT measured: Qwen3-Next-80B, REAP-246B, gemma-26B, dense/hybrid

**Honest closure scope**: "Partial methodology probe complete (3 regimes × 1 metric × 2 MoE proxies). Gate explicitly NOT met. Phase 2.2 of remediation fills missing interference scenario + 4 metrics on the 2 proxies + adds Qwen3.5/3.6-27B dense as 3rd proxy."

## Commands run

Binary: `/mnt/raid0/llm/llama.cpp-experimental/build/bin/llama-bench` at HEAD `8cb04da9d`.
Wrapper: `OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active numactl --interleave=all -t 96 -fa 1 -mmp 0 -r 3` (proper canonical).

### Per-context plain throughput (`-n 32` default decode mode)

| Model | Context | Log |
|---|---|---|
| Coder-30B Q4_K_M | 2K | `Coder_2K.log` |
| Coder-30B Q4_K_M | 8K | `Coder_8K.log` |
| Coder-30B Q4_K_M | 32K | `Coder_32K.log` |
| Qwen3.6-35B Q8_0 | 2K | `Q8_2K.log` |
| Qwen3.6-35B Q8_0 | 8K | `Q8_8K.log` |
| Qwen3.6-35B Q8_0 | 32K | `Q8_32K.log` |

### Combined prefill+decode (`-pg pp,tg` mode)

| Model | Context | Log |
|---|---|---|
| Coder-30B Q4_K_M | 2K | `Coder_2K_pg32.log` |
| Coder-30B Q4_K_M | 8K | `Coder_8K_pg32.log` |
| Coder-30B Q4_K_M | 32K | `Coder_32K_pg32.log` |
| Qwen3.6-35B Q8_0 | 2K | `Q8_2K_pg32.log` |
| Qwen3.6-35B Q8_0 | 8K | `Q8_8K_pg32.log` |
| Qwen3.6-35B Q8_0 | 32K | `Q8_32K_pg32.log` |

### NOT run (Phase 2.2 of remediation)

- Long-prompt-mid-stream interference scenario (handoff regime 4): `llama-server --parallel 2` config sending 2K decode while concurrent 32K prefill in flight on a different slot.
- TTFT, decode stall fraction, per-iteration latency variance metrics for the 2K/8K/32K runs already collected.
- Qwen3.5/3.6-27B (dense/hybrid) across the same regimes (closes finding #11 cross-architecture coverage gap).

## Files in this bundle

| File | Purpose | Source |
|---|---|---|
| `Coder_*.log`, `Q8_*.log`, `*_pg32.log` | raw llama-bench stdout per regime | original 2026-04-27 |
| `SUMMARY.md` | per-regime t/s table + commentary | original 2026-04-27 (note: contains "no follow-up actions" wording that has been corrected in the handoff per remediation Phase 1) |
| `system-state.txt` | numactl + numa_balancing + THP + governor + SMT + uptime + free + hugepages | backfilled 2026-04-27 evening (current snapshot) |
| `process-pre.txt` | pgrep snapshot showing no llama-* processes before run | backfilled 2026-04-27 evening |
| `process-post.txt` | pgrep snapshot showing no llama-* processes after run | backfilled 2026-04-27 evening |
| `ld_debug.log` | LD_DEBUG=libs trace of one smoke command on the default-flags build | backfilled 2026-04-27 evening |
| `results.csv` | tabulated per-regime t/s | backfilled 2026-04-27 evening from `*_pg32.log` files |
| `decision.md` | explicit pass/fail/partial verdict | backfilled 2026-04-27 evening |

## Backfill caveat

system-state.txt + process-pre/post.txt + ld_debug.log captured at backfill time. The Artifact-bundle backfill policy in `cpu-benchmark-rigor-and-revalidation.md` accepts this for already-claimed-closed tracks where the system has not drifted.

## Remediation reference

See `~/.claude/plans/nifty-discovering-allen.md` Phase 2.2:
- Long-prompt-mid-stream interference scenario via `llama-server --parallel 2`.
- TTFT, decode stall fraction, per-iteration latency variance metrics.
- Qwen3.5/3.6-27B as 3rd proxy.

Output dir: `2026-04-28-cpu23-interference-metrics/`.
