# E1 P-BENCH-3 Clean Sweep Note

Run ID: `e1-pbench3-clean-20260703T1912Z`
Protocol: `P-BENCH-3`
Date: 2026-07-03

This artifact is decision-grade for the `qwen36_q8_0` A3B model only.

Completed rows:

- `qwen36_q8_0` at `-np 1,2,4,8,16`
- 43/43 successful requests at every completed `-np` level
- host-health warnings: none
- `kernel.numa_balancing=0`, governors `performance`, no pre-existing llama processes

The original manifest also included dense control `qwen36_27b_q8`, but that
control was stopped before any completed summary row. The dense `-np 1` cell
was diagnostic only: live logs showed roughly `0.59` generated tok/s on long
responses, making the full five-level dense control impractical for this
window. Do not use this directory as a complete dense-control E1 result.

Key A3B result from `summary.csv`: best task throughput was `-np 16` at
846.72 tasks/hour, but `-np 2`, `-np 8`, and `-np 16` were all in the same
throughput band while tail latency rose sharply with `-np`.
