# NUMA-DECOUPLE-1 offline re-grade (2026-11-21)

Re-grade of the 5 E5 Stage-B cells banked with `empty_trimmed_window` (INF-45 signed decouple),
computed offline from existing run dirs (`requests.jsonl` + `cells.jsonl`); no server, no inference.
Reproduce: `python3 regrade.py <epyc-inference-research>/data/batched_decode regrade.json`.
`regrade.json` holds the per-cell table. Banked run dirs are not modified; trimmed numbers recompute
identically (all 0.0), primary tok/s unchanged (read from the same row; note the primary is on the
`wallclock_fallback` basis corpus-wide).

| cell | old | new | trimmed_window_empty | primary tok/s |
|---|---|---|---|---|
| qwen36_q8_0-C1-np32 | False | True | True | 8.819 |
| qwen36_q8_0-C2-np16 | False | True | True | 14.047 |
| gemma4_26b_a4b_q4km_mtp-C1-np32 | False | True | True | 64.136 |
| qwen3_next_80b-C1-np32 | False | True | True | 10.407 |
| qwen3_next_80b-C1b-np16 | False | True | True | 10.793 |

Caveat: `run_overrides_active` is not persisted per row; assumed False (no override blocker in banked rows).
