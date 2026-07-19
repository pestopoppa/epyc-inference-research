# OP-2 Canonical Bench Window Evidence — 2026-07-19 live5

Run root: `data/op2_canonical_bench_window/op2-canonical-bench-window-20260719-live5`

Prepared bundle: `data/op2_canonical_bench_window/op2-20260719-live-nothink`

Operator approval ref: `operator-directive-2026-07-19-op2-ungated`

## Verdict

- Live v6+iqk preflight: PASS.
- Live role/garbage smoke: PASS, 6/6 exact `OP2_READY`.
- Clean-host canonical CPU sentinel: PASS.
- Canonical frontdoor Q8 tg128 bench: COMPLETE.
- B1 barrier-fusion A/B: skipped, no current immutable on/off binary pair.
- B4 DSA-D3 profile: closed no-go from prior D3.1 profile, Lightning Indexer 1.08% cycle samples.

## Live v6+iqk Evidence

Preflight artifact: `preflight/live_stack_preflight.json`

- `overall=PASS`
- `live_affinity_verified=true`
- `health_ok=true`
- `topology_hash=acf01b15781654d5904e8d4c0aa4aec3987d2def4b7a9546eeb2df8ee095774c`
- `registry_hash=a185591d35134009ebab23706b28140d322ccb2d84478b2dec3909bcd8f9ee4c`
- `contention_matrix_fresh=false`, recorded observation-only for this OP-2 run.

Perf preflight artifact: `preflight/perf_counter_preflight.json`

- `status=ok`
- Canonical AMD events present and smoke probe passed.

Process gate artifact: `live-v6/process_blockers.json`

- `blocker_n=0`

Role smoke artifact: `live-v6/role_smoke_aggregate.json`

- `row_n=6`
- `pass_n=6`
- `fail_n=0`
- `all_pass=true`

| Role | Prompt t/s | Decode t/s | Completion tokens | Draft accepted |
|---|---:|---:|---:|---:|
| frontdoor | 43.26 | 35.48 | 4 | 4/4 |
| worker_general | 105.08 | 30.99 | 5 | 4/4 |
| architect_general | 18.74 | 16.59 | 4 | 4/4 |
| ingest_long_context | 53.76 | 27.76 | 4 | n/a |
| worker_vision | 26.49 | 36.04 | 4 | n/a |
| vision_escalation | 28.10 | 39.98 | 4 | n/a |

The record-only CPU-clean preflight before live smokes reported `status=blocked` because the live `llama-server` processes were intentionally up. That artifact is retained as host/process context, not as the clean-host decision gate.

## Clean Canonical CPU Bench

The generated workflow stopped the six live OP-2 roles before the strict canonical CPU phase. Post-run process check found no `llama-server`, `llama-bench`, `bench_canonical`, `operator_next_commands`, or AutoPilot process.

Strict clean sentinel artifact: `canonical-v6/cpu_clean_sentinel.json`

- `status=ok`
- `host_warnings=[]`
- `process blockers=0`
- Sentinel `avg_ts=19.1828`

Canonical bench artifact: `canonical-v6/frontdoor_q8_tg128.results.json`

- Protocol: `bench_canonical.sh`, P-BENCH-1 canonical CPU decode.
- Binary: `/mnt/raid0/llm/llama.cpp/build/bin/llama-bench`
- Build: `build_commit=91745611f`, `build_number=9774`
- Backend: `CPU`
- Model: `Qwen3.6-35B-A3B-MTP-Q8_0.gguf`
- Shape: `n_prompt=0`, `n_gen=128`, `reps=10`, `n_threads=96`
- KV: `type_k=f16`, `type_v=f16`
- Result: `avg_ts=12.442712`, `stddev_ts=0.010877`

Canonical stderr records `GGML_IQK=1` and the iqk activation lines:

- `ik_llama GEMM kernels engaged`
- `MoE mul_mat_id via ik kernels`

This tg128 bench is the canonical raw CPU decode measurement. It is not a stack-serving MTP/NEXTN throughput matrix row.
