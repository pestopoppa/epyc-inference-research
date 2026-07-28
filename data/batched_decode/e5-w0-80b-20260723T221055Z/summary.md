# E5 NUMA x np sweep summary (P-BENCH-3)

> E5 K=1 cells are NOT byte-comparable to E1 rows (different -c convention: E5 uses -c max(8192, per_stream_ctx*np), E1 used fixed -c 32768) — direction cross-check only.

## Cells

| cell | config | np | tasks/h raw | tasks/h trimmed | p50 ms | p95 ms | TTFT p50 ms | accept | kvu | grade | error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3_next_80b-C1-np1-scout | C1 | 1 | 1162.18 | 1175.32 | 1544 | 6315 | 1103 | - | off | obs |  |
| qwen3_next_80b-C1-np16-scout | C1 | 16 | 1700.04 | 929.90 | 17493 | 67372 | 8820 | - | off | obs |  |
| qwen3_next_80b-C1-np2-scout | C1 | 2 | 1434.34 | 1449.15 | 2148 | 11208 | 1426 | - | off | obs |  |
| qwen3_next_80b-C1-np32-scout | C1 | 32 | 1556.16 | 0.00 | 51212 | 99004 | 18244 | - | off | obs |  |
| qwen3_next_80b-C1-np4-scout | C1 | 4 | 1549.75 | 1421.43 | 4013 | 19909 | 1689 | - | off | obs |  |
| qwen3_next_80b-C1-np8-scout | C1 | 8 | 1669.43 | 1318.38 | 7890 | 36960 | 3227 | - | off | obs |  |
| qwen3_next_80b-C1b-np1-scout | C1b | 1 | 1376.52 | 1370.16 | 2291 | 15578 | 1513 | - | off | obs |  |
| qwen3_next_80b-C1b-np16-scout | C1b | 16 | 1904.62 | 0.00 | 43771 | 81234 | 21485 | - | off | obs |  |
| qwen3_next_80b-C1b-np2-scout | C1b | 2 | 1659.45 | 1495.52 | 4444 | 21428 | 2057 | - | off | obs |  |
| qwen3_next_80b-C1b-np4-scout | C1b | 4 | 1886.03 | 1546.74 | 7996 | 35662 | 3281 | - | off | obs |  |
| qwen3_next_80b-C1b-np8-scout | C1b | 8 | 2013.43 | 1198.19 | 14793 | 73172 | 10503 | - | off | obs |  |
| qwen3_next_80b-C3-np1-scout | C3 | 1 | 1896.87 | 1795.27 | 4624 | 17486 | 2306 | - | off | obs |  |
| qwen3_next_80b-C3-np2-scout | C3 | 2 | 2274.68 | 1873.43 | 7512 | 22696 | 3260 | - | off | obs |  |
| qwen3_next_80b-C3-np4-scout | C3 | 4 | 2520.09 | 1627.69 | 13335 | 53936 | 5964 | - | off | obs |  |
| qwen3_next_80b-C3-np8-scout | C3 | 8 | 2385.68 | 0.00 | 29671 | 62329 | 16440 | - | off | obs |  |

## R1 — iso-T crossover

- whole-machine T=8: C1b@4 vs C3@2: winner -> qwen3_next_80b-C3-np2-scout
- whole-machine T=16: C1b@8 vs C3@4: winner -> qwen3_next_80b-C3-np4-scout
- whole-machine T=32: C1b@16 vs C3@8: winner -> qwen3_next_80b-C3-np8-scout
- half-machine T=16: C1@16 vs C2@8: insufficient_data
- half-machine T=32: C1@32 vs C2@16: insufficient_data
- K* roofline flip: None

## R2 — lane reality

- qwen3_next_80b+Q4_K_M: lanes_real=True

## R3 — eval-lane pricing

- status: refused
- reason: no --current-arm-baseline supplied; refusing to price the eval lane until a FRESH current-arm baseline row (v7 + core_v2 + WP-12 fleet layer) is measured

## R4 — model-keyed capability rows

- qwen3_next_80b+Q4_K_M: C3@np8 (2385.676 tasks/h)
