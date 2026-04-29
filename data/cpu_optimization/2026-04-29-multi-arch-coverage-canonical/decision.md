# Multi-Arch Coverage Matrix — Canonical Re-Run

**Date**: 2026-04-29 evening (post-reboot, post-OMP-fix)
**Verdict**: previous "Probe A" first-pass (2026-04-29 morning, broken-OMP) headlines were almost entirely contamination. Real picture is much smaller-magnitude with different signs.

## Method

3 archs × 4 configs × n=15 reps under FULL canonical recipe:
```
OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active
numactl --interleave=all -- taskset -c 0-95
llama-bench -t 96 -fa 1 --mmap 0 -p 0 -n 64 -r 15
```

Per-cell pgrep guard, sequential cells, model order: smallest first.

## Result

| Model | c0 baseline | c1 CPU1 stack | c2 CPU2-off | c3 CPU1+CPU2-off |
|---|---|---|---|---|
| Nemotron-9B Q8 (Hybrid SSM) | 12.94 ± 0.04 | 13.07 (+1.0%) | **13.17 (+1.78%)** | 13.09 (+1.16%) |
| Qwen3.6-27B Q8 (Dense Q8)   |  4.28 ± 0.06 |  4.08 (-4.7%) |  4.14 (-3.3%) |  4.21 (-1.6%) |
| gemma-4-31B Q4 (Dense Q4)   |  6.40 ± 0.41 |  6.44 (+0.6%) | **6.65 (+3.9%)** | 6.63 (+3.6%) |

> **Note**: gemma c0's std is high (±0.41 = 6.4% CV) suggesting baseline drift; Probe B re-tested with n=5 and tighter std (±0.01) showed gemma c2/c3 are within noise. The "+3.9% on gemma c2" here may be partially baseline-drift artifact rather than a real treatment effect.

## Comparison to original first-pass (2026-04-29 morning, BROKEN OMP env)

| Cell | Original first-pass | Canonical re-run | Comment |
|---|---|---|---|
| Nemotron c3 | **+13.43%** | +1.16% | Original was 11× too positive (broken OMP + concurrent agents + host throttle interaction) |
| Nemotron c1 | +2.93% | +1.0% | Direction same, magnitude similar |
| Nemotron c2 | -0.62% | +1.78% | Direction flipped |
| Qwen3.6-27B c1 | +1.79% | -4.7% | Direction flipped |
| gemma c1 | **-12.36%** | +0.6% | Original was 20× too negative — major contamination |
| gemma c2 | -11.01% | +3.9% | Direction flipped |
| gemma c3 | -11.69% | +3.6% | Direction flipped |

The original "Probe A" measurement was poisoned by THREE compounding contaminations: broken OMP env (this session's discovery), 3 concurrent claude sessions on the host, and the host throttle that triggered the user reboot. Almost every cell's effect was either inflated or sign-flipped.

## Per-arch interpretation

### Hybrid SSM (Nemotron-9B Q8)
All three treatment configs show small positive effects (+1.0% to +1.8%). c2 (mbind-off) alone is the strongest in tg64. Confirmed by Probe B with stronger prefill effect (see [`workload-shape-canonical/decision.md`](../2026-04-29-workload-shape-canonical/decision.md)).

### Dense Q8 (Qwen3.6-27B Q8)
ALL configs are NEGATIVE (-1.6% to -4.7%). CPU1 stack alone is the most negative. Dense Q8 architecture does not benefit from these CPU1/CPU2 levers under canonical recipe — likely because Dense Q8 already saturates DRAM channels in the standard --interleave=all path, and CPU1's CCD-aware partitioning fragments the tile distribution without compensating gain.

### Dense Q4 (gemma-4-31B Q4_K_M)
Multi-arch shows c2/c3 around +3.6-3.9% but Probe B reduces this to ~0% under tighter measurement. Likely no real signal at canonical recipe.

## Headline conclusion

**No clear "ship CPU1 stack universally" signal.** Per-arch picture:
- Hybrid SSM: small +1-2% wins from c2 (mbind-off) — possibly worth shipping for hybrid-SSM workload
- Dense Q8: all configs HURT — do NOT enable CPU1 or mbind-off for Dense Q8 production
- Dense Q4: configs are within noise (per Probe B verification) — leave defaults

The previous "ship CPU1 + mbind=off everywhere" recommendation from the broken-OMP first-pass is invalidated. The actual deployment recommendation is **arch-conditional**:

| Arch class | Recommendation |
|---|---|
| Hybrid SSM (Nemotron, Qwen3-Next?) | Consider c2 (mbind-off) opt-in for small +1-2% tg + larger +8.9% pp512 (Probe B) |
| Dense Q8 (Qwen3.6-27B) | Default v5 (no CPU1, mbind ON). All probed configs negative. |
| Dense Q4_K_M (gemma) | Default v5. Configs neutral within noise. |
| MoE Q4_K_M (Coder-30B, REAP-246B) | Defer to existing closures (CPU1 +1.8% Coder per CPU21; CPU2 mbind +6% on Q8 MoE per CPU2 work). |

## Files

- `run_canonical.sh` — measurement script
- `master.log` — full run log
- `canon_<model>_<config>.log` — per-cell bench logs (12 cells)
- `decision.md` — this document
