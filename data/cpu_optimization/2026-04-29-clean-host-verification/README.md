# Clean Host Verification — Coder-30B Q4_K_M tg32 Reproducibility Tripwire

## Purpose

Single-bench reproducibility check after discovering session-wide measurements at 3-5× below production canonical. Per `feedback_host_throttle_check`, run a Coder-30B Q4_K_M tg32 5-rep canonical and compare to morning-canonical baseline of 58.65 t/s.

## Result — DID NOT REPRODUCE

| Run | Result |
|---|---|
| Production canonical (kernel-env-flags-inventory, 2026-04-28) | 58.65 t/s (PGO) / 60.54 (with BOLT) |
| This session, my libggml-cpu.so rebuild | **14.87 ± 1.12 t/s** — 4× too slow |
| This session, with `.pgo-only` validated baseline binary | **20.68 ± 9.44 t/s** — 3× too slow, very high CV |
| This session, my rebuild after binary swap-back | **16.19 ± 6.35 t/s** — 4× too slow |

The `.pgo-only` baseline binary (md5 834fb8ced77fc1c771496e6c1a58f02f, sha-stamp matching production canonical) ALSO did not reproduce. So my rebuild was NOT the cause.

## Diagnosis — host-level performance throttle

Tested CPU freq scaling under 96-thread CPU burn:
- 38 of 96 cores stuck at 1998 MHz (base) under all-core load
- Only ~50 cores hit expected 2800-2870 MHz all-core boost
- Governor: `performance` (not the cause)
- No thermal-throttle dmesg events checked

Most likely root cause: thermal/power hysteresis from sustained 2-day uptime + heavy benchmark load earlier in session. Possibly compounded by megasync (83% on 1 core, sustained 2 days), Firefox active web content, and multi-day kernel scheduler drift.

User's decision: reboot the host to restore production-canonical state.

## Implication for the entire session's measurements

ALL bench numbers measured this session are 3-5× lower than they should be. RELATIVE comparisons (A vs B with same arms) likely still hold under common-mode throttle. ABSOLUTE numbers definitely do not. See:

- `../2026-04-29-multi-arch-coverage/` — provisional, contaminated AND throttled
- `../2026-04-29-multi-arch-coverage-rerun/` — partial, throttled
- `../2026-04-29-cpu4-op-coalesced-barriers-phase1/` — closure relies on relative ratio (still likely valid)
- `../2026-04-30-state-sync-cost-probe/` (slot-promotion canonical 3×2) — closure relies on K=4/K=1 ratio (still likely valid)
- `../2026-04-30-divergent-tree-sweep/` — closure relies on aux-win-rate (independent of absolute throughput)
- `../2026-04-30-slot-promotion-dense-target/` — closure relies on K=4/K=1 ratio (still likely valid)
- `../2026-04-29-mab-phase-0-prime-prime-replication/` — closure relies on linear/tree paired ratio (still likely valid)

## Files

- bench output captured inline in commit message; no separate run script preserved (single-shot diagnostic)
- This README

## Next-session todo

1. Reboot host (user has decided to do this).
2. Run a fresh Coder-30B Q4_K_M tg32 -r 5 canonical FIRST. Expected: 56-60 t/s. If reproduces, host is clean.
3. If clean, re-run multi-arch coverage matrix from scratch.
4. (If absolute numbers matter for some closure) re-verify slot-promotion + MAB closures' headline numbers under clean conditions.
