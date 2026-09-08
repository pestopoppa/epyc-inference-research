# Champion maximum performance — Qwen3.8-27B-Q8_0 on MI210, 2026-09-08

**Build `ef81196d5`** (GPU tip `bff30cebe` + CPU `champion3 9c4f73e29`, FOLD-2 passed), DFlash2
drafter, canonical recipe `qwen3.8-27b-q8-gpu-dflash2-np4` with **only `np` varied**.
**3 launches per point; unit is the LAUNCH.** All residency `proven`, VRAM 33.1-43.1 GB.

| slots | aggregate tok/s | per slot | p95 dev over 3 launches | runs |
|---:|---:|---:|---:|---|
| 1 | **79.25** | 79.25 | 0.44% | 79.2, 79.2, 79.6 |
| 2 | 109.41 | 54.70 | 1.60% | 109.4, 108.9, 111.2 |
| 4 | 167.76 | 41.94 | 3.33% | 167.8, 162.2, 169.7 |
| 8 | **179.12** | 22.39 | 1.82% | 182.4, 178.2, 179.1 |

**Headline: 79.25 tok/s for a single user, 179.12 tok/s aggregate at peak concurrency.**

**Operating point is np=4**: it delivers 94% of peak aggregate while every user still sees ~42 tok/s.
Going 4 -> 8 buys +7% total throughput for nearly half the per-user rate.

**Dispersion grows with concurrency** (0.44% at np=1 to 3.33% at np=4), so a single reading at np=4
is far less trustworthy than one at np=1. Always quote a launch-unit spread.

## Two things this is NOT

1. **Not comparable to `llama-bench` tg128.** That harness cannot do speculative decoding at all and
   reads ~31 tok/s on this model. It is valid for A/B kernel comparison and is **never** a serving
   rate. Quoting it as one understated the system by 2.5x (error made and corrected 2026-09-08).
2. **Not a champion-vs-production ratio.** The v9 freeze
   (`epyc-root artifacts/operator/ratify_v9_final_freeze_20260811.json`) records
   `qwen36_27b_q8_dflash: lane_ineligible_acceptance_below_floor` and `dflash_lineup_enabled: false`
   — production was frozen with the DFlash drafter compiled in but **not enabled** for the 27B, and
   for Qwen3.**6**, not 3.8. There is no like-for-like production baseline because production never
   ran this lane; the advantage here is partly a **capability**, not a speed delta. Any such headline
   must measure production at ITS own best rather than force it through this recipe.

Raw: `sweep.json`, `sweep.log`, and `sweep.py` as executed.
