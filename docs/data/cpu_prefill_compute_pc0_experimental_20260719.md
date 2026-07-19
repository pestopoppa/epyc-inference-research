# PC-0 CPU Prefill-Compute Experimental Profile - 2026-07-19

Status: observation-grade profile artifact. This is not a `P-BENCH-1` or OP-2
decision row. It used the experimental CPU build only and did not touch production
v6.

Run root:
`/mnt/raid0/llm/epyc-inference-research/data/cpu_prefill_compute/b7-pc0-prefill-experimental-20260719T083513Z-codex/b7-pc0-prefill`

## Identity

| Field | Value |
|---|---|
| Source worktree | `/mnt/raid0/llm/llama.cpp-experimental` |
| Source HEAD at dryrun | `6ad45fa3f` |
| Binary | `/mnt/raid0/llm/llama.cpp-experimental/build-k24-cpu/bin/llama-bench` |
| Binary-reported build commit | `9882c2c69` |
| Libraries | `libllama*` / `libggml*` resolved from `build-k24-cpu/bin`; `libgomp` from LLVM 20 |
| Model | `Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf` |
| Device | CPU only: `-dev none -ngl 0 -nopo 1 -nkvo 1` |
| Threads | `96`, `taskset -c 0-95`, `numactl --interleave=all` |
| Runtime flags | `GGML_IQK=1`, f16 KV, flash-attn on, mmap off |

## Throughput

| Cell | Reps | Result |
|---|---:|---|
| `pp8192/n0` | 3 | `121.963712 t/s` mean, stddev `0.057502` |
| `tg1` tail | 3 | `5.739871 t/s` mean, stddev `0.016634` |
| `pp8192/n0` under `perf record` | 1 | `122.233563 t/s` |
| `tg1` under `perf record` | 1 | `5.355556 t/s` |

## Perf Stat

| Counter | Value |
|---|---:|
| Vector MAC | `8,456,982,072,716` |
| Vector all | `22,643,196,192,923` |
| Scalar all | `885,723,783,653` |
| Demand DRAM fills | `14,794,057,632` |
| HW prefetch DRAM fills | `36,609,497,262` |
| Cycles | `78,832,270,658,790` |
| Instructions | `116,070,023,346,414` |
| IPC | `1.47` |
| CPU utilization | `92.660 CPUs` |
| Elapsed | `279.535805496s` |

## Perf Report Highlights

`perf record` captured `10.323 GiB` / `1,284,822` samples with zero lost
samples. Top children from `architect_p8192_n1.perf_report.txt`:

| Area | Children |
|---|---:|
| `ggml_graph_compute_thread` | `98.54%` |
| `GOMP_barrier` / OpenMP barrier path | `43.12%` |
| `ggml_iqk_try_mul_mat_id` | `22.16%` |
| `iqk_mul_mat_moe` | `18.93%` |
| `ggml_compute_forward_mul_mat` | `16.52%` |
| `llamafile_sgemm` | `14.75%` |
| `ggml_compute_forward_flash_attn_ext` | `5.88%` |
| `ggml_compute_forward_gated_delta_net` | `1.74%` |

## Interpretation

The prefill shape is materially different from the decode roofline. IPC is far
healthier (`1.47` here versus the decode reference `0.17`), vector work is
substantial, and the profile is dominated by OpenMP/barrier overhead plus large
matrix paths rather than a pure DRAM-stall signature.

The immediate kernel/design implication is not a blind new GEMV kernel. The
positive next lever is barrier-count/operator fusion and prefill graph fusion,
especially around qwen35 MoE/GDN prefill hot paths. This is still observation
evidence: OP-2 or another measurement-grade quiet window must rerun or
retro-certify before using the number as a promotion/decision gate.
