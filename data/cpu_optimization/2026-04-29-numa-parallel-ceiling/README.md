# 4-NUMA Aggregate Ceiling Probe — Phase 2 gain potential test

**Date**: 2026-04-29
**Purpose**: Measure aggregate ceiling for slot-promotion Phase 2 (NUMA-parallel candidate verify) WITHOUT implementing the orchestration. Establishes upper bound on Phase 2 gain.

## Result: 6.10× aggregate over single-instance — GATE MET WITH HEADROOM

| Configuration | pp32 t/s |
|---|---|
| Single × 96t | 68.22 ± 29.07 |
| 1 quarter solo (24t) | 113.93 ± 15.81 |
| 4 quarters parallel (24t each) | Q0=115, Q1=101, Q2=98, Q3=102 |
| Aggregate | **416.48** |
| Ratio | **6.10×** |

Phase 2 gate is ≥1.3×. Probe shows 6.10× available. Phase 2 implementation is structurally justified.

## Bonus finding (independently relevant)

1 quarter at 24t (113.93 t/s) runs ~1.7× faster than full machine at 96t (68.22 t/s) on Qwen3.6-35B-A3B Q8. Model is over-threaded at 96; the existing 4×48t NUMA orchestrator may be leaving aggregate throughput on the table vs 4×24t. NOT TESTED here; separately actionable.

## Files

- `single_t96.log` — single-instance 96-thread baseline
- `quarter0_solo.log` — single quarter solo (no parallel contention)
- `quarter[0-3]_parallel.log` — 4 quarters launched simultaneously
