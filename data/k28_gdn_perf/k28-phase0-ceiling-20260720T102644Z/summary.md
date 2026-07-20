# K28 Phase 0 Ceiling Model

Op rerun: `/mnt/raid0/llm/epyc-inference-research/data/k28_gdn_perf/k28-phase0-op-rerun-20260720T102526Z`
Full-model source: `/mnt/raid0/llm/epyc-inference-research/data/k28_gdn_perf/k28-fused-vs-graph-qwen36-35b-summary-20260720.json`

| n_prompt | full prompt t/s | full s | GDN share est. | full gain @2x op | @3x | @4x | @5x | method |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 64 | 706.40 | 0.091 | 5.06% | 2.53% | 3.38% | 3.80% | 4.05% | measured |
| 256 | 1643.63 | 0.156 | 12.03% | 6.02% | 8.02% | 9.03% | 9.63% | measured |
| 2048 | 2100.06 | 0.975 | 15.31% | 7.65% | 10.21% | 11.48% | 12.25% | linear_extrapolated_from_1024 |
| 8192 | 1995.07 | 4.106 | 14.54% | 7.27% | 9.70% | 10.91% | 11.63% | linear_extrapolated_from_1024 |

Profiler tools (`rocprofv2`, `rocprof`, `omniperf`) were not available in this environment, so this is modeled attribution, not direct trace attribution.

Recommendation: do not delay v7 promotion for K28 Phase 1. K28 remains a plausible post-promotion/default-off kernel project, but Phase 1 should wait for direct profiler availability or a throwaway prototype that proves a materially higher full-model ceiling.
