# E5 NUMA x np sweep summary (P-BENCH-3)

> E5 K=1 cells are NOT byte-comparable to E1 rows (different -c convention: E5 uses -c max(8192, per_stream_ctx*np), E1 used fixed -c 32768) — direction cross-check only.

## Cells

| cell | config | np | tasks/h raw | tasks/h trimmed | p50 ms | p95 ms | TTFT p50 ms | accept | kvu | grade | error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen36_27b_q8-C1-np1-scout-full-e1parity | C1 | 1 | 393.79 | 391.17 | 9546 | 12934 | 2551 | 0.731 | off | degraded |  |
| qwen36_27b_q8-C1-np1-scout-full | C1 | 1 | 574.17 | 574.94 | 4866 | 11755 | 2692 | 0.822 | off | obs |  |
| qwen36_27b_q8-C1-np1-scout | C1 | 1 | 751.48 | 751.91 | 3884 | 9018 | 2425 | 0.822 | off | obs |  |
| qwen36_27b_q8-C1-np16-scout | C1 | 16 | 780.93 | 383.63 | 58603 | 142873 | 18707 | 0.563 | off | obs |  |
| qwen36_27b_q8-C1-np2-scout | C1 | 2 | 807.92 | 794.35 | 7144 | 17602 | 3997 | 0.809 | off | obs |  |
| qwen36_27b_q8-C1-np32-scout | C1 | 32 | 810.29 | 0.00 | 111870 | 190164 | 47957 | 0.603 | off | obs |  |
| qwen36_27b_q8-C1-np4-scout | C1 | 4 | 848.72 | 763.15 | 14544 | 35640 | 5298 | 0.740 | off | obs |  |
| qwen36_27b_q8-C1-np8-scout-full | C1 | 8 | 611.35 | 452.56 | 38245 | 96526 | 12639 | 0.592 | off | obs |  |
| qwen36_27b_q8-C1-np8-scout | C1 | 8 | 814.31 | 625.05 | 32807 | 69677 | 10151 | 0.652 | off | obs |  |
| qwen36_27b_q8-C1b-np1-scout | C1b | 1 | 835.63 | 842.55 | 6880 | 17280 | 3737 | 0.827 | off | obs |  |
| qwen36_27b_q8-C1b-np16-scout | C1b | 16 | 983.73 | 0.00 | 84963 | 156063 | 55742 | 0.657 | off | obs |  |
| qwen36_27b_q8-C1b-np2-scout | C1b | 2 | 938.98 | 903.87 | 14526 | 35046 | 6553 | 0.810 | off | obs |  |
| qwen36_27b_q8-C1b-np4-scout | C1b | 4 | 965.27 | 754.21 | 25684 | 56435 | 11098 | 0.711 | off | obs |  |
| qwen36_27b_q8-C1b-np8-scout | C1b | 8 | 1008.61 | 502.47 | 51031 | 117399 | 18490 | 0.620 | off | obs |  |
| qwen36_27b_q8-C2-np1-scout | C2 | 1 | 818.69 | 809.15 | 6461 | 20636 | 4020 | 0.825 | off | obs |  |
| qwen36_27b_q8-C2-np16-scout | C2 | 16 | 926.10 | 0.00 | 91709 | 164653 | 67691 | 0.637 | off | obs |  |
| qwen36_27b_q8-C2-np2-scout | C2 | 2 | 1008.89 | 927.64 | 12606 | 26945 | 6216 | 0.818 | off | obs |  |
| qwen36_27b_q8-C2-np4-scout | C2 | 4 | 1085.25 | 829.47 | 21939 | 50978 | 10763 | 0.715 | off | obs |  |
| qwen36_27b_q8-C2-np8-scout | C2 | 8 | 1026.61 | 501.26 | 45611 | 110027 | 20961 | 0.657 | off | obs |  |
| qwen36_27b_q8-C3-np1-scout | C3 | 1 | 1225.38 | 1116.97 | 8726 | 22897 | 5969 | 0.827 | off | obs |  |
| qwen36_27b_q8-C3-np2-scout | C3 | 2 | 1415.17 | 1085.15 | 18337 | 39211 | 9505 | 0.812 | off | obs |  |
| qwen36_27b_q8-C3-np4-scout | C3 | 4 | 1407.22 | 675.53 | 29943 | 77383 | 15780 | 0.695 | off | obs |  |
| qwen36_27b_q8-C3-np8-scout | C3 | 8 | 1333.53 | 135.72 | 59821 | 114690 | 43135 | 0.676 | off | obs |  |

## R1 — iso-T crossover

- whole-machine T=8: C1b@4 vs C3@2: winner -> qwen36_27b_q8-C3-np2-scout
- whole-machine T=16: C1b@8 vs C3@4: winner -> qwen36_27b_q8-C3-np4-scout
- whole-machine T=32: C1b@16 vs C3@8: winner -> qwen36_27b_q8-C1b-np16-scout [MIXED METRIC BASIS — caveated]
- half-machine T=16: C1@16 vs C2@8: winner -> qwen36_27b_q8-C2-np8-scout
- half-machine T=32: C1@32 vs C2@16: winner -> qwen36_27b_q8-C2-np16-scout
- K* roofline flip: 32

## Scout paired probes

- dense C1 shape qwen36_27b_q8-C1-np1-scout-full vs qwen36_27b_q8-C1-np1-scout: ok winner=half0
- dense C1 shape qwen36_27b_q8-C1-np8-scout-full vs qwen36_27b_q8-C1-np8-scout: ok winner=half0

## R2 — lane reality

- qwen36_27b_q8+Q8_0: lanes_real=False

## R3 — eval-lane pricing

- status: refused
- reason: no --current-arm-baseline supplied; refusing to price the eval lane until a FRESH current-arm baseline row (v7 + core_v2 + WP-12 fleet layer) is measured

## R4 — model-keyed capability rows

- qwen36_27b_q8+Q8_0: C3@np1 (1116.969 tasks/h)

## Degraded cells (garbage gate)

- qwen36_27b_q8-C1-np1-scout-full-e1parity: speed demoted to observation
