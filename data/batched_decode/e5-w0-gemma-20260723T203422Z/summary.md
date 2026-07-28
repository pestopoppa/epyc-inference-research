# E5 NUMA x np sweep summary (P-BENCH-3)

> E5 K=1 cells are NOT byte-comparable to E1 rows (different -c convention: E5 uses -c max(8192, per_stream_ctx*np), E1 used fixed -c 32768) — direction cross-check only.

## Cells

| cell | config | np | tasks/h raw | tasks/h trimmed | p50 ms | p95 ms | TTFT p50 ms | accept | kvu | grade | error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4_26b_a4b_q4km_mtp-C1-np1-scout | C1 | 1 | 1515.42 | 1513.30 | 2363 | 3011 | 0 | 0.846 | off | degraded |  |
| gemma4_26b_a4b_q4km_mtp-C1-np16-scout | C1 | 16 | 2879.48 | 1693.58 | 18103 | 26273 | 0 | 0.851 | off | degraded |  |
| gemma4_26b_a4b_q4km_mtp-C1-np2-scout | C1 | 2 | 2018.16 | 1963.12 | 3508 | 4687 | 0 | 0.848 | off | degraded |  |
| gemma4_26b_a4b_q4km_mtp-C1-np32-scout | C1 | 32 | 2854.45 | 0.00 | 32193 | 41698 | 0 | 0.850 | off | degraded |  |
| gemma4_26b_a4b_q4km_mtp-C1-np4-scout | C1 | 4 | 2127.16 | 1985.61 | 6734 | 8155 | 0 | 0.850 | off | degraded |  |
| gemma4_26b_a4b_q4km_mtp-C1-np8-scout | C1 | 8 | 2474.62 | 2122.61 | 11224 | 13022 | 0 | 0.857 | off | degraded |  |
| gemma4_26b_a4b_q4km_mtp-C3-np1-scout | C3 | 1 | 3246.33 | 3063.48 | 4227 | 6181 | 0 | 0.857 | off | degraded |  |
| gemma4_26b_a4b_q4km_mtp-C3-np2-scout | C3 | 2 | 4128.81 | 3329.00 | 6553 | 9805 | 0 | 0.863 | off | degraded |  |
| gemma4_26b_a4b_q4km_mtp-C3-np4-scout | C3 | 4 | 4447.90 | 2716.51 | 11680 | 16113 | 0 | 0.861 | off | degraded |  |
| gemma4_26b_a4b_q4km_mtp-C3-np8-scout | C3 | 8 | 5076.47 | 0.00 | 19277 | 20377 | 0 | 0.863 | off | degraded |  |

## R1 — iso-T crossover

- whole-machine T=8: C1b@4 vs C3@2: insufficient_data
- whole-machine T=16: C1b@8 vs C3@4: insufficient_data
- whole-machine T=32: C1b@16 vs C3@8: insufficient_data
- half-machine T=16: C1@16 vs C2@8: insufficient_data
- half-machine T=32: C1@32 vs C2@16: insufficient_data
- K* roofline flip: None

## R2 — lane reality


## R3 — eval-lane pricing

- status: refused
- reason: no --current-arm-baseline supplied; refusing to price the eval lane until a FRESH current-arm baseline row (v7 + core_v2 + WP-12 fleet layer) is measured

## R4 — model-keyed capability rows


## Degraded cells (garbage gate)

- gemma4_26b_a4b_q4km_mtp-C1-np1-scout: speed demoted to observation
- gemma4_26b_a4b_q4km_mtp-C1-np16-scout: speed demoted to observation
- gemma4_26b_a4b_q4km_mtp-C1-np2-scout: speed demoted to observation
- gemma4_26b_a4b_q4km_mtp-C1-np32-scout: speed demoted to observation
- gemma4_26b_a4b_q4km_mtp-C1-np4-scout: speed demoted to observation
- gemma4_26b_a4b_q4km_mtp-C1-np8-scout: speed demoted to observation
- gemma4_26b_a4b_q4km_mtp-C3-np1-scout: speed demoted to observation
- gemma4_26b_a4b_q4km_mtp-C3-np2-scout: speed demoted to observation
- gemma4_26b_a4b_q4km_mtp-C3-np4-scout: speed demoted to observation
- gemma4_26b_a4b_q4km_mtp-C3-np8-scout: speed demoted to observation
