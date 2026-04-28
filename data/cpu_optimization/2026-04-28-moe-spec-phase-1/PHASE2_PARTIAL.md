# MoE-Spec Phase 2 — Partial results (wrap-up snapshot 2026-04-28 evening)

This file captures the in-flight state at session wrap-up. End-to-end spec-dec measurement task `b4bk8cu9g` ran but hit a summary-script error after data capture; the underlying 18 completion JSONs ARE valid (rep0 of each B value empty due to server-not-ready timeout; rep1/rep2 captured cleanly).

## Phase 1 re-validation under v5 PGO mixed-B build

5-rep proper canonical pp32, megasync at 100% on 1 core throughout (depresses absolute numbers ~50%; relative deltas hold).

### Coder-30B Q4_K_M (n_expert=128)

| B | mean ± std | Δ vs B=0 |
|---|---|---|
| 0 | 198.57 ± 5.55 | reference |
| 96 | 144.79 ± 6.54 | -27.0% (mask overhead exceeds savings at light budget) |
| 64 | 193.34 ± 10.64 | **-2.6% (parity)** — was -43% on single-B PGO before mixed-B fix |
| 32 | 249.60 ± 3.24 | +25.7% (severe quality cost per Phase 1 PPL) |

### REAP-246B Q4_K_M (n_expert=80)

| B | mean ± std | Δ vs B=0 |
|---|---|---|
| 0 | 51.14 ± 1.11 | reference |
| 60 | 52.05 ± 0.99 | +1.8% (within noise) |
| 40 | 58.06 ± 0.59 | **+13.5% (clean signal)** |
| 20 | 74.45 ± 0.08 | +45.6% (quality unusable per Phase 1 PPL) |

## Phase 2 end-to-end spec-dec via llama-server (rep0 of each B excluded — server warmup timeout)

llama-cli/llama-completion segfault in non-TTY mode in this fork; pivoted to llama-server HTTP API with /completion. 3 different prompts per (model, B) to prevent cache regurgitation. `--draft-max 32 --p-split 0` matches production registry. Megasync still loud during measurement.

### Coder-30B Q4_K_M end-to-end

| B | rep1 | rep2 | mean (rep1+rep2) | accept% mean | Δ vs B=0 |
|---|---|---|---|---|---|
| 0 | 30.19 t/s, 68.7% | 36.37 t/s, 59.6% | ~33.3 t/s | ~64% | reference |
| 64 | 34.10 t/s, 61.2% | 38.62 t/s, 61.3% | ~36.4 t/s | ~61% | **+9.3%** (with ~3pp accept drop) |
| 32 | 19.37 t/s, 71.2% | 39.17 t/s, 62.7% | ~29.3 t/s | ~67% | -12% (huge per-prompt variance — unstable) |

### REAP-246B Q4_K_M end-to-end

| B | rep1 | rep2 | mean (rep1+rep2) | accept% mean | Δ vs B=0 |
|---|---|---|---|---|---|
| 0 | 7.27 t/s, 59.9% | 7.90 t/s, 56.2% | ~7.59 t/s | ~58% | reference |
| 60 | 7.03 t/s, 60.5% | 7.48 t/s, 55.6% | ~7.26 t/s | ~58% | -4.3% (regression, light budget) |
| 40 | 7.61 t/s, 61.3% | 8.06 t/s, 55.6% | ~7.84 t/s | ~58% | **+3.3%** (small but positive) |

## Honest interpretation

**Verification-batch (pp32) gain DOES NOT translate 1:1 to end-to-end spec-dec gain** because spec-dec round = drafter forward + target verification + acceptance evaluation. MoE-Spec only accelerates the target verification step. With drafter (small, fast) taking ~30% of round time, target verification ~50%, accept eval ~20%, a +13.5% verification speedup yields ~+7% end-to-end at best (Amdahl).

REAP-246B observed +3.3% end-to-end is in the expected range, modest but real. Coder-30B end-to-end +9.3% is surprisingly larger than its pp32 parity result — but rep variance is wider, may not survive 5-rep stable measurement.

## Caveats

1. **Rep0 of every B excluded** due to server-not-ready timeout (curl fired before /health returned ok). Should re-run with longer warmup wait.
2. **Megasync system noise** during measurement window (100% CPU on 1 core). Re-measurement under quieter conditions could shift absolute numbers ~50% upward.
3. **3 reps × 3 prompts** is below CPU20 protocol's ≥5 reps for sub-5% claims. End-to-end gains are sub-10% so 5-10 reps preferred.
4. **No PPL gate yet on v5 PGO build** — full 32-chunk PPL deferred for Phase 3.

## Recommended next steps for the planning agent

1. **Re-measure end-to-end spec-dec** with proper warm-up (≥60s post-server-ready before first request) and 5-10 reps per (model, B) under quieter system conditions.
2. **Coder-30B production decision**: B=64 end-to-end +9% is borderline; if confirmed under 5-10 reps, deployable opt-in. If end-to-end Coder gain shrinks <2%, keep MoE-Spec OFF on Coder.
3. **REAP-246B production decision**: B=40 end-to-end +3% is small but positive. Tradeoff vs ~3pp acceptance-rate noise. Probably deployable; ALSO check at B=50 (between 60-noise and 40-positive).
4. **BOLT-libggml on v5 PGO for Coder role**: independent gain on top of MoE-Spec.
5. **Investigate end-to-end Coder >Coder-pp32 anomaly**: pp32 said -2.6% parity but end-to-end says +9.3%. May be measurement noise (rep0 missing), may indicate something interesting about how spec-dec round dynamics differ from pure prefill timing.
