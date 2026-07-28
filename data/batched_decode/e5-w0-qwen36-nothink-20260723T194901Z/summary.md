# E5 NUMA x np sweep summary (P-BENCH-3)

> E5 K=1 cells are NOT byte-comparable to E1 rows (different -c convention: E5 uses -c max(8192, per_stream_ctx*np), E1 used fixed -c 32768) — direction cross-check only.

## Cells

| cell | config | np | tasks/h raw | tasks/h trimmed | p50 ms | p95 ms | TTFT p50 ms | accept | kvu | grade | error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen36_q8_0-C1-np1-scout | C1 | 1 | 1359.06 | 1368.13 | 1615 | 4986 | 1070 | 0.837 | off | obs |  |
| qwen36_q8_0-C1-np16-scout-kvu | C1 | 16 | 1733.61 | 794.93 | 23333 | 64786 | 9676 | 0.674 | on | obs |  |
| qwen36_q8_0-C1-np16-scout | C1 | 16 | 1528.48 | 729.97 | 23270 | 77671 | 10417 | 0.603 | off | obs |  |
| qwen36_q8_0-C1-np2-scout | C1 | 2 | 1505.17 | 1485.98 | 3076 | 9298 | 1800 | 0.813 | off | obs |  |
| qwen36_q8_0-C1-np32-scout | C1 | 32 | 1584.69 | 0.00 | 50002 | 97085 | 32454 | 0.591 | off | obs |  |
| qwen36_q8_0-C1-np4-scout | C1 | 4 | 1548.72 | 1389.97 | 5514 | 20401 | 2834 | 0.742 | off | obs |  |
| qwen36_q8_0-C1-np8-scout | C1 | 8 | 1609.97 | 1261.12 | 11915 | 39212 | 5966 | 0.704 | off | obs |  |
| qwen36_q8_0-C1b-np1-scout | C1b | 1 | 988.45 | 984.23 | 4153 | 15375 | 2665 | 0.837 | off | obs |  |
| qwen36_q8_0-C1b-np16-scout | C1b | 16 | 1153.99 | 0.00 | 64029 | 134089 | 36860 | 0.638 | off | obs |  |
| qwen36_q8_0-C1b-np2-scout | C1b | 2 | 1120.30 | 1024.68 | 8258 | 29655 | 4736 | 0.828 | off | obs |  |
| qwen36_q8_0-C1b-np4-scout | C1b | 4 | 1109.28 | 830.77 | 16695 | 55651 | 9520 | 0.735 | off | obs |  |
| qwen36_q8_0-C1b-np8-scout | C1b | 8 | 1197.76 | 584.16 | 34934 | 98422 | 15937 | 0.655 | off | obs |  |
| qwen36_q8_0-C2-np1-scout | C2 | 1 | 1539.69 | 1548.55 | 3410 | 10905 | 1647 | 0.833 | off | obs |  |
| qwen36_q8_0-C2-np16-scout | C2 | 16 | 1706.73 | 0.00 | 41574 | 89799 | 27943 | 0.668 | off | obs |  |
| qwen36_q8_0-C2-np2-scout | C2 | 2 | 1728.68 | 1622.67 | 5812 | 19107 | 2889 | 0.814 | off | obs |  |
| qwen36_q8_0-C2-np4-scout | C2 | 4 | 1826.47 | 1517.00 | 11010 | 28903 | 5598 | 0.771 | off | obs |  |
| qwen36_q8_0-C2-np8-scout | C2 | 8 | 1717.95 | 868.93 | 27150 | 75098 | 11030 | 0.591 | off | obs |  |
| qwen36_q8_0-C3-np1-scout | C3 | 1 | 1681.74 | 1517.37 | 4797 | 18275 | 2933 | 0.837 | off | obs |  |
| qwen36_q8_0-C3-np2-scout | C3 | 2 | 1908.68 | 1504.57 | 12097 | 32991 | 5949 | 0.809 | off | obs |  |
| qwen36_q8_0-C3-np4-scout | C3 | 4 | 2027.87 | 1057.70 | 21242 | 61797 | 10755 | 0.715 | off | obs |  |
| qwen36_q8_0-C3-np8-scout | C3 | 8 | 1774.43 | 0.00 | 50135 | 82223 | 23395 | 0.632 | off | obs |  |

## R1 — iso-T crossover

- whole-machine T=8: C1b@4 vs C3@2: winner -> qwen36_q8_0-C3-np2-scout
- whole-machine T=16: C1b@8 vs C3@4: winner -> qwen36_q8_0-C3-np4-scout
- whole-machine T=32: C1b@16 vs C3@8: winner -> qwen36_q8_0-C3-np8-scout
- half-machine T=16: C1@16 vs C2@8: winner -> qwen36_q8_0-C2-np8-scout
- half-machine T=32: C1@32 vs C2@16: tie (margin < 10%: tie — prefer status-quo quarters (C3))
- K* roofline flip: None

## Scout paired probes

- kvu probe qwen36_q8_0-C1-np16-scout-kvu vs qwen36_q8_0-C1-np16-scout: ok delta=0.089 escalate=True

## R2 — lane reality

- qwen36_q8_0+Q8_0: lanes_real=True

## R3 — eval-lane pricing

- status: refused
- reason: no --current-arm-baseline supplied; refusing to price the eval lane until a FRESH current-arm baseline row (v7 + core_v2 + WP-12 fleet layer) is measured

## R4 — model-keyed capability rows

- qwen36_q8_0+Q8_0: C2@np2 (1622.674 tasks/h)
