# Phase B — Slot-promotion canonical 3×2 re-test under canonical

**Date**: 2026-04-29
**Verdict**: **CLOSURE CONFIRMED — gate not flipped**. K=4/K=1 ratio under canonical = 0.45 (worse than original 0.651). Structural finding stands: canonical prompts produce single-leaf trees, dispatcher engages 0 times, K=4 collapses to 24t/quarter vs K=1's 96t.

## Result

OMP env stack DID lift the K=1 baseline materially (11.40 → 15.15 t/s, +33%) but K=4 stayed flat around 6.82 t/s. Gate (≥1.3×) requires K=4 ≥ 19.7 t/s under canonical baseline; actual = 6.82.

| Config | n=6 mean t/s | vs original |
|---|---|---|
| K=1 (numa-q=1) | **15.15** | +33% (OMP env recovery) |
| K=4 (numa-q=4) | **6.82** | -8% (within noise) |
| K=4/K=1 ratio | **0.450** | original 0.651 (gate worse) |

K-parallel verify hits: **0 / 6** k4 reps (same as original). `numa_alt_paths` empty on every round because canonical prompts under temperature=0.0 produce single-leaf speculation trees.

## Why the gate didn't flip

The OMP env stack helps the 96-thread K=1 path (proper barrier behavior across NUMA). But the K=4 path's bottleneck isn't OMP — it's the structural property that the dispatcher's K-parallel block never engages on canonical prompts. With no alt-paths, K=4 mode degenerates to 4 ctxs each running at 24 threads (one NUMA quarter), losing the full-machine bandwidth that K=1 enjoys. OMP env stack doesn't change that ratio.

If anything, proper OMP makes the gap WIDER because K=1's 96t scales better with proper OMP env than K=4's 24t/ctx does.

## Closure framing (UNCHANGED from original)

> Phase 1.1 dispatcher v1 (per-ctx sample-and-accept reducer + sequential pre-decode aux state sync + parallel aux decode + winner-state commit) on Qwen3.6-35B-A3B Q8 hybrid Delta Net + Qwen3-1.7B Q8 drafter at v5 PGO build delivers parity-or-worse vs K=1 baseline on the canonical 3-prompt × 2-rep workload. The dispatcher functions correctly. Canonical prompts under temperature=0.0 produce single-leaf speculation trees — `numa_alt_paths` empty on every round. Gate not met on this workload + config.
>
> Does NOT generalize to "K-parallel verify is dead on hybrid" or to "NUMA-parallel verify is dead". Different (workload, K, p_split, temperature, drafter pair) configurations remain unevaluated.

The original scoped framing was already correct. Re-test under canonical OMP recipe REINFORCES the finding (gap wider, not narrower).

## Files

- `run_phaseB.sh` — measurement script with full canonical recipe
- `phaseB_master.log` — master log
- `srv_k1.log`, `srv_k4.log` — per-config server logs
- `comp_k1_p*_r*.json`, `comp_k4_p*_r*.json` — per-prompt completion JSONs
- `decision.md` — this document
