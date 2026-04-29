# Multi-Arch Coverage Probe — Clean Re-Run (PARTIAL, ABORTED 2026-04-29)

> **⚠️ DATA INTEGRITY WARNING**: this bundle's measurements were taken on a host with degraded power state — 38 of 96 cores stuck at base 1998 MHz under load instead of expected 2800-3000 MHz all-core boost. Coder-30B Q4_K_M tg32 reproduced at 11-20 t/s vs production canonical 58.65 t/s (3-5× regression). User rebooted the host to restore. **Re-run this entire matrix on post-reboot host before trusting any number here.**

## Purpose

Re-run of the multi-arch coverage matrix (`../2026-04-29-multi-arch-coverage/`) at n=15 with per-cell pgrep guards, after first-pass n=5 + n=30 replication showed evidence of concurrent-agent contention from 3 other claude sessions on the host.

## Method

Same matrix as Probe A: 3 architecture classes × 4 flag configs × n=15 reps. Per-cell pgrep guard aborts if foreign llama process appears mid-run. 5-second sleep between cells to stabilize.

## Partial results landed before abort

| Model | Config | tg64 t/s (n=15) | Notes |
|---|---|---|---|
| Nemotron-9B Q8 | c0 baseline | 6.75 ± 0.25 | |
| Nemotron-9B Q8 | c1 (CPU1 stack) | 7.22 ± 0.16 | +6.96% vs c0 |
| Nemotron-9B Q8 | c2 (CPU2 mbind off) | 7.02 ± 0.61 | +4.00% vs c0 (high CV) |
| Nemotron-9B Q8 | c3 (CPU1+CPU2off) | 7.24 ± 0.08 | +7.26% vs c0 |
| Qwen3.6-27B Q8 (Qwen3.5 hybrid SSM 27B per ggml) | c0 baseline | 1.68 ± 0.02 | |
| Qwen3.6-27B Q8 | c1 (CPU1 stack) | 1.64 ± 0.05 | -2.38% vs c0 |
| Qwen3.6-27B Q8 | c2 (CPU2 mbind off) | 1.67 ± 0.05 | -0.60% vs c0 |
| Qwen3.6-27B Q8 | c3 (CPU1+CPU2off) | running... | aborted |
| gemma-4-31B Q4_K_M | all 4 cells | (not run) | aborted |

## Why aborted

User flagged that the absolute t/s numbers are way below production canonical (Coder-30B 58.65 → 11-20 in this session). Investigation revealed host-level CPU freq throttle (38/96 cores stuck at 1998 MHz under load). Decision: abort and reboot host.

## Disposition

- All numbers above are PROVISIONAL — the relative ratios (e.g., Nemotron c1 +6.96%) may approximately hold under common-mode throttle, but absolute t/s is unreliable.
- Full matrix re-run required on post-reboot host with reproducibility-tripwire check (Coder-30B tg32 must reproduce ~58.65 t/s before trusting subsequent measurements).

## Files

- `run_clean.sh` — run script with per-cell pgrep guard
- `clean_master.log` — master log up to abort
- `clean_*.log` — per-cell bench logs (8 of 12 cells completed)
- (no decision.md — closure deferred to post-reboot re-run)
