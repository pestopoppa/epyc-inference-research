# PC-4i qwen35moe scheduler split trace

Date: 2026-07-20

## Scope

Attribute the PC-4h OpenMP barrier-heavy profile to scheduler split boundaries
before attempting another qwen35moe prefill prototype. PC-4h showed
`GOMP_barrier` / `__kmpc_barrier` at `43.95%` children while router/top-k/
weights symbols were too small to justify a router prototype.

## Artifacts

- Valid run root:
  `data/cpu_prefill_compute/pc4i-qwen35moe-sched-split-trace-20260720T013749Z/`
- Parsed summary:
  `data/cpu_prefill_compute/pc4i-qwen35moe-sched-split-trace-20260720T013749Z/summary.json`
- Strict postflight:
  `data/cpu_prefill_compute/pc4i-qwen35moe-sched-split-trace-20260720T013749Z/post_process_check_strict.txt`
- Stale-DSO guardrail run:
  `data/cpu_prefill_compute/pc4i-qwen35moe-sched-split-trace-20260720T013242Z/`

## Command shape

Model:
`/mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf`

Binary:
`/mnt/raid0/llm/llama.cpp-experimental/build-k24-cpu/bin/llama-bench`

Key flags/env:

- `LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-k24-cpu/bin`
- `GGML_IQK=1`
- `GGML_SCHED_TRACE_SPLITS=2`
- `HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1`
- `taskset -c 0-95 numactl --interleave=all`
- `-t 96 -p 8192 -n 1 -r 1 -fa 1 -mmp 0 -dev none -ngl 0 -nopo 1 -nkvo 1 -o json`

The explicit `LD_LIBRARY_PATH` is required in this shell: the ambient
`LD_LIBRARY_PATH` resolves unpinned experimental executables against production
v6 libraries under `/mnt/raid0/llm/llama.cpp/build/bin`. The first PC-4i run
therefore produced throughput but no scheduler trace because it did not load the
instrumented experimental `libggml-base.so`. The valid run records `ldd.txt`
showing all `libllama*` / `libggml*` DSOs resolved from
`llama.cpp-experimental/build-k24-cpu/bin`.

## Result

The valid run exited `0`.

| Shape | Result |
|---|---:|
| `pp8192/n0` | `98.610457 t/s` |
| `tg1` | `4.897132 t/s` |

Execution metadata:

| Field | Value |
|---|---:|
| Wall time | `2:57.48` |
| CPU utilization | `7628%` |
| Max RSS | `77037420 KB` |
| Trace plan rows | `45` |
| Trace compute rows | `34` |
| Median split compute time | `5171460 us` |

Scheduler trace:

| Split | Backend | Range | Nodes | Inputs | First | Last |
|---:|---|---|---:|---:|---|---|
| `0` | `CPU` | `[0,4471)` | `4471` | `0` | `model.input_embed/GET_ROWS` | `result_output/MUL_MAT` |

IQK activation was present:

- `[iqk] ACTIVE: MoE mul_mat_id via ik kernels (type=12 n_as=256)`
- `[iqk] ACTIVE: ik_llama GEMM kernels engaged (first mul_mat type=14 ne00=3072)`

## Decision

PC-4i closes the scheduler-split attribution step. The qwen35moe CPU-only
`p8192/n1` workload is not split-boundary-bound: every observed graph is a
single CPU scheduler split with zero cross-backend inputs. A scheduler-level
split/copy prototype has no current target.

The next PC-4 step is CPU-backend/node-level barrier attribution inside that
single split. Specifically, instrument or profile `ggml_graph_compute_thread`
and adjacent CPU backend execution so barrier counts and wall time can be mapped
to graph-node/operator classes before another fusion prototype. Do not change
router/top-k, view/add expansion, `mul_mat_id` math, or scheduler split logic
from the current PC-4 evidence.
