# PC-4b qwen35moe prefill graph-node trace

Date: 2026-07-19

## Scope

Trace qwen35/qwen35moe graph construction for the CPU prefill-compute PC-4
track before implementing any default-off fusion. This run is post-candidate
research: the validated v7 promotion candidate remains `6ad45fa3ff` / binary
`10098`; this trace used later experimental build `12a292f0c` / binary `10099`.

## Artifacts

- Initial non-verbose trace attempt:
  `data/cpu_prefill_compute/pc4b-qwen35-trace-20260719T234651Z/`
- Valid verbose trace:
  `data/cpu_prefill_compute/pc4b-qwen35-trace-verbose-20260719T235218Z/`
- Compact summary:
  `data/cpu_prefill_compute/pc4b-qwen35-trace-verbose-20260719T235218Z/reports/trace_summary.json`

The first run completed cleanly but produced no per-layer lines because
`llama-bench` installs a null log callback unless `-v` is passed. The rerun used
the same CPU-only shape plus `-v` and emitted the intended trace.

## Command shape

Model:
`/mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf`

Binary:
`/mnt/raid0/llm/llama.cpp-experimental/build-k24-cpu/bin/llama-bench`

Key flags/env:

- `LLAMA_QWEN35_PREFILL_TRACE=1`
- `LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-k24-cpu/bin:/usr/lib/llvm-20/lib`
- `GGML_IQK=1`
- `taskset -c 0-95 numactl --interleave=all`
- `-t 96 -fa 1 -mmp 0 -p 8192 -n 1 -r 1 -dev none -ngl 0 -nopo 1 -nkvo 1 -o json -v`

Preflight `ldd` resolved `libllama.so.0` and `libggml-cpu.so.0` from the
experimental build directory, not production v6.

## Result

The verbose run exited `0` and cleanup found no `llama-bench`, `llama-server`,
`perf`, `rocprof`, AutoPilot, or KFD PIDs.

Measured throughput in the trace run:

| Shape | Result |
|---|---:|
| `pp8192/n0` | `112.082350 t/s` |
| `tg1` | `4.311924 t/s` |
| Max RSS | `77043932 KiB` |

Trace summary:

| Item | Result |
|---|---:|
| Graph builds traced | `45` |
| Final graph-node count | `4471` every build |
| Layer trace lines | `2160` |
| Recurrent `linear_attn` layer-0 delta | `92` |
| Recurrent `linear_attn` nonzero-layer delta | `99` |
| `full_attn` layer delta | `75` |

Interpretation: the high graph-node/dispatch surface is the recurrent
`linear_attn` qwen35moe path, not full attention. The current layer-level trace
is enough to reject a full-attention-first implementation, but it is still too
coarse to safely patch a fusion target inside the recurrent layer.

## Decision

PC-4b closes as a trace/target-selection step. The next safe work is PC-4c:
add a deeper default-off sublayer trace inside the recurrent `linear_attn` path
to break down GDN, SSM, shared expert, routed expert, norm, and residual
islands. Only then pick a default-off implementation. PC-4 remains open until an
exact-output/profile-guarded implementation reduces both libomp spin/pause and
wall time on repeated `p8192/n1`.
