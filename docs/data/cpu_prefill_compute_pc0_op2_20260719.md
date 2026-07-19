# PC-0 OP-2 CPU Prefill-Compute Profile - 2026-07-19

Status: OP-2 quiet-window profile evidence for the first PC-0 cell. This run
does not edit, rebuild, or promote the production kernel. It uses the frozen
production-v6 `bench_canonical.sh` path as an execution source and records the
result for CPU prefill-compute target selection.

Run root:
`/mnt/raid0/llm/epyc-inference-research/data/cpu_prefill_compute/pc0-op2-20260719T225343Z`

## Identity

| Field | Value |
|---|---|
| Kernel worktree | `/mnt/raid0/llm/llama.cpp` |
| Branch at post-check | `production-consolidated-v6` |
| Tree HEAD at post-check | `91a8424ea` |
| Binary | `/mnt/raid0/llm/llama.cpp/build/bin/llama-bench` |
| Binary-reported build commit | `91745611f` |
| Binary-reported build number | `9774` |
| Model | `Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf` |
| Model type | `qwen35moe 122B.A10B Q4_K - Medium` |
| Backend reported by bench | `CPU`; `gpu_info` empty |
| Threads | `96`, `taskset -c 0-95`, `numactl --interleave=all` |
| Runtime flags | `GGML_IQK=1`, f16 KV, flash-attn on, mmap off |
| GPU post-check | no KFD PIDs, ROCm GPU idle |

## Throughput

| Cell | Reps | Result |
|---|---:|---|
| `pp8192/n0` under `perf stat` | 3 | `112.730698 t/s` mean, stddev `1.234638` |
| `tg1` tail under `perf stat` | 3 | `4.989817 t/s` mean, stddev `0.013130` |
| `pp8192/n0` under `perf record` | 1 | `115.618125 t/s` |
| `tg1` tail under `perf record` | 1 | `4.811150 t/s` |

## Perf Stat

| Counter | Value |
|---|---:|
| Vector MAC | `4,409,716,259,822` |
| Vector all | `7,690,028,606,794` |
| Scalar all | `832,417,883,252` |
| Demand DRAM fills | `19,540,343,650` |
| HW prefetch DRAM fills | `45,486,971,154` |
| Cycles | `84,718,920,597,550` |
| Instructions | `92,224,771,750,541` |
| IPC | `1.09` |
| CPU utilization | `68.597 CPUs` |
| Elapsed | `410.573992780s` |

## Perf Record / Report

`perf record` captured `10837.601 MB` / `1,348,833` samples. The bounded
`perf report --stdio --no-children -g none` row had `0` lost samples and
`1M` cycle samples.

DSO summary from the bounded report:

| Shared object | Overhead |
|---|---:|
| `(deleted)` main `llama-bench` mapping | `49.57%` |
| `libggml-cpu.so.0.15.2` | `46.47%` |
| `libggml-base.so.0.15.2` | `1.41%` |
| `[unknown]` | `1.80%` |
| `libc.so.6` | `0.68%` |

Visible `libggml-cpu` hot symbols include Q4_K/Q5_K `mul_mat_qX_K_q8_2_X4_T`,
Q8 helper matmul, and `tinyBLAS<16,...>::gemm_bloc<4, 6>`. The report is
partially symbolized: the largest main-binary mapping appears as `(deleted)`
addresses, and kernel symbols are restricted by `kptr_restrict`. This makes the
profile sufficient for the PC-0 premise verdict but insufficient to select a
precise implementation target by itself.

## Interpretation

The PC-0 premise survives the quiet-window profile. The 8K prefill cell is
materially different from the CPU decode roofline: IPC is `1.09` here rather
than the decode reference `0.17`, vector work is substantial, and nearly half
of cycle samples resolve into `libggml-cpu` rather than a pure DRAM-stall
signature.

This closes the first PC-0 profile cell as positive. It does not yet identify a
safe kernel edit target. Before coding a prefill-specific kernel, run or derive
a cleaner symbolized report for the unresolved main-binary mapping, or use an
equivalent address-resolution pass against the exact sampled binary.

## Artifact Policy

This report is the committed durable artifact. The raw run-root files are kept
local because the repository PII pre-commit hook intentionally blocks long
unseparated digit runs, and perf/bench outputs contain legitimate large hardware
counters and model parameter counts that match that detector. Do not bypass the
hook. If raw perf evidence needs to be pushed, first add a sanctioned sanitized
artifact format or a narrow allow-list for benchmark counter files.

Local raw artifacts:

- `dryrun/architect_p8192_n1.dryrun.stderr`
- `perf-stat/architect_p8192_n1.results.json`
- `perf-stat/architect_p8192_n1.perf_stat.txt`
- `perf-record/architect_p8192_n1.record_results.json`
- `perf-record/architect_p8192_n1.record_stderr.txt`
- `reports/architect_p8192_n1.perf_report.no_children.txt`
- `reports/architect_p8192_n1.perf_report.dso.txt`

Do not commit `perf-record/architect_p8192_n1.perf.data`; it is an 11GB local
scratch profile.
