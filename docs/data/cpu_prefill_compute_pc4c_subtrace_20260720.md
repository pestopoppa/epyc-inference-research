# PC-4c qwen35moe recurrent prefill subphase trace

Date: 2026-07-20

## Scope

Run the default-off level-2 qwen35/qwen35moe prefill trace added after PC-4b
so PC-4 can choose a concrete implementation target from subphase evidence
rather than from layer-level graph-node totals.

This is post-candidate kernel research. The v7 promotion candidate remains
frozen at `6ad45fa3ff` / binary `10098`; this run used the later experimental
CPU build from `llama.cpp-experimental` with local, uncommitted trace-only
instrumentation.

## Artifacts

- Run root:
  `data/cpu_prefill_compute/pc4c-qwen35-subtrace-20260720T001959Z/`
- Compact parsed summary:
  `data/cpu_prefill_compute/pc4c-qwen35-subtrace-20260720T001959Z/reports/trace_summary.json`
- Raw stderr/time trace:
  `data/cpu_prefill_compute/pc4c-qwen35-subtrace-20260720T001959Z/trace/architect_p8192_n1.trace_stderr_time.txt`

## Command shape

Model:
`/mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf`

Binary:
`/mnt/raid0/llm/llama.cpp-experimental/build-k24-cpu/bin/llama-bench`

Key flags/env:

- `LLAMA_QWEN35_PREFILL_TRACE=2`
- `LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-k24-cpu/bin:/usr/lib/llvm-20/lib`
- `GGML_IQK=1`
- `HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1`
- `taskset -c 0-95 numactl --interleave=all`
- `-t 96 -fa 1 -mmp 0 -p 8192 -n 1 -r 1 -dev none -ngl 0 -nopo 1 -nkvo 1 -o json -v`

Preflight `ldd` resolved `libllama.so.0` and `libggml-cpu.so.0` from the
experimental build directory, not production v6. The run was CPU-only: cleanup
found no residual `llama-bench`, `llama-server`, `perf`, `rocprof`, AutoPilot,
or KFD GPU PIDs.

## Result

The run exited `0`.

| Shape | Result |
|---|---:|
| `pp8192/n0` | `118.040030 t/s` |
| `tg1` | `5.339296 t/s` |
| Max RSS | `77038696 KiB` |
| Wall time | `2:29.13` |

Trace coverage:

| Item | Result |
|---|---:|
| Graph builds traced | `45` |
| Final graph-node count | `4471` every build |
| Phase lines | `2160` |
| Subphase lines | `24300` |
| Recurrent linear-attention layers per build | `36` |
| Full-attention layers per build | `12` |

Subphase deltas, median per traced subphase line:

| Subphase | Count | Median delta | Unique deltas |
|---|---:|---:|---|
| `linear_attn_total` | `1620` | `53` | `53,55` |
| `ffn_total` | `2160` | `40` | `40` |
| `full_attn_total` | `540` | `29` | `29` |
| `conv_state` | `1620` | `15` | `15,17` |
| `gated_delta_net` | `1620` | `13` | `13` |
| `ssm_state` | `1620` | `8` | `8` |
| `gated_norm` | `1620` | `6` | `6` |
| `linear_proj` | `1620` | `5` | `5` |
| `conv_qkv_norm` | `1620` | `3` | `3` |
| `linear_out` | `1620` | `2` | `2` |

Layer-level phase totals:

| Phase | Count | Median delta | Unique deltas |
|---|---:|---:|---|
| `linear_attn` | `1620` | `99` | `99,101` |
| `full_attn` | `540` | `75` | `75` |

## Interpretation

PC-4b correctly identified recurrent `linear_attn` as the largest
attention-side graph-node island, but PC-4c shows it is not the only dispatch
pressure source. Each recurrent layer contributes about `53` graph nodes inside
`linear_attn_total`; every layer, including full-attention layers, also
contributes a stable `40` graph-node `ffn_total` island. The first prototype
should therefore optimize recurrent prefill only if a follow-up timing profile
confirms those nodes drive wall time or OpenMP barrier spin; otherwise the
same-input MoE/FFN island may be the better barrier-count target.

Within recurrent `linear_attn_total`, the largest sub-islands are
`conv_state` (`15` nodes), `gated_delta_net` (`13`), and `ssm_state` (`8`).
Small norm/residual/output islands are visible but are not large enough to be
the first implementation target without profile evidence.

## Decision

PC-4c closes the trace-disambiguation step. The next safe step is PC-4d:
run a bounded profile or implement a default-off prototype against the
recurrent `conv_state`/`gated_delta_net`/`ssm_state` island only if the profile
confirms it reduces libomp spin and wall time. If profile evidence instead
keeps `ffn_total` dominant, route PC-4d to same-input MoE/FFN graph fusion.
