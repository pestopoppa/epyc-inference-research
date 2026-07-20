# PC-4k qwen35moe CONCAT dim0 row-partition probe

Date: 2026-07-20

Status: keep-candidate, default-off. Not promotion-ready.

Run root:
`/mnt/raid0/llm/epyc-inference-research/data/cpu_prefill_compute/pc4k-qwen35moe-concat-dim0-rows-ab-20260720T021127Z/`

Sanitized summary:
`/mnt/raid0/llm/epyc-inference-research/data/cpu_prefill_compute/pc4k-qwen35moe-concat-dim0-rows-ab-20260720T021127Z/summary.json`

## Probe

PC-4j mapped qwen35moe CPU prefill barrier time to `CONCAT` / `conv_input-*`
inside shared `llm_build_delta_net_base::build_conv_state()`. The target graph
shape concatenates recurrent convolution state with transposed prompt activations
on dim0, while `ne2 == 1` for the tested single-sequence `p8192/n1` shape.

The current CPU `CONCAT` implementation assigns `n_threads` tasks but partitions
the copy loop only across `i2`. For this shape, only thread 0 receives useful
work and the rest wait at the post-node barrier.

The PC-4k probe adds an experimental-only opt-in path:

```text
GGML_CPU_CONCAT_DIM0_ROWS=1
```

When enabled and `ggml_concat(..., dim=0)` is used, the CPU path partitions work
across flattened `(i1, i2, i3)` rows instead of only `i2`. Dim>0 keeps the
existing kernel.

## Validation

Experimental tree only:
`/mnt/raid0/llm/llama.cpp-experimental`

Validated commands/results:

- `cmake --build build-k24-cpu --target llama-bench llama-simple -j 16` passed.
- `ctest --test-dir build-k24-cpu -R '^test-llama-archs$' --output-on-failure`
  passed.
- `cmake --build build-k24-cpu --target test-backend-ops -j 16` passed.
- `test-backend-ops test -o CONCAT -b CPU -j 1` passed with env unset:
  `195/195`.
- `GGML_CPU_CONCAT_DIM0_ROWS=1 test-backend-ops test -o CONCAT -b CPU -j 1`
  passed: `195/195`.
- `test-recurrent-state-rollback` passed with env unset after rebuilding the
  stale binary.
- `GGML_CPU_CONCAT_DIM0_ROWS=1 test-recurrent-state-rollback` passed.
- Direct generated `qwen35moe-moe.gguf` recurrent rollback passed with env off.
- Direct generated `qwen35moe-moe.gguf` recurrent rollback passed with env on.

Test coverage added a transposed-src dim0 CONCAT backend case for F32, F16, and
BF16 to mirror the `build_conv_state()` source shape.

Important run guard: commands pinned
`LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-k24-cpu/bin`,
because the ambient shell path contains the production-v6 build directory.

## A/B Evidence

Model:
`Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf`

Common command shape:
CPU-only, `GGML_IQK=1`, `-t 96`, `-p 8192`, `-n 1`, `-fa 1`, `-mmp 0`,
`-dev none`, `-ngl 0`, `-nopo 1`, `-nkvo 1`. IQK activation lines were present
in every arm. Postflight found no active benchmark/server processes and no KFD
GPU PIDs.

### Clean no-trace A/B

`llama-bench -r 2`, no CPU trace instrumentation.

| Arm | pp8192 | tg1 | Wall | Max RSS |
| --- | ---: | ---: | ---: | ---: |
| default | `97.492993 t/s` | `4.648280 t/s` | `4:24.73` | `77036888 KB` |
| `GGML_CPU_CONCAT_DIM0_ROWS=1` | `100.641523 t/s` | `4.743434 t/s` | `4:12.34` | `77036992 KB` |

Deltas:

- pp8192: `+3.2295%`
- tg1: `+2.0471%`

### Traced attribution A/B

`llama-bench -r 1` with `GGML_CPU_TRACE_GRAPH=1` and
`GGML_CPU_TRACE_GRAPH_TOPK=16`.

| Metric | Default | `GGML_CPU_CONCAT_DIM0_ROWS=1` | Delta |
| --- | ---: | ---: | ---: |
| pp8192 | `99.432725 t/s` | `103.385572 t/s` | `+3.9754%` |
| tg1 | `4.835903 t/s` | `4.730467 t/s` | `-2.1803%` |
| median barrier/compute ratio | `0.6915` | `0.5603` | `-18.9661%` |
| median barrier time | `201249411 us` | `171007051 us` | `-15.0273%` |
| median compute time | `291034484 us` | `305180705.5 us` | `+4.8607%` |

The target `CONCAT` attribution changed from the dominant barrier row to a
minor row:

| Target op | Default | `GGML_CPU_CONCAT_DIM0_ROWS=1` | Delta |
| --- | ---: | ---: | ---: |
| `CONCAT` barrier sum | `2196940708 us` | `17871828 us` | `-99.1865%` |
| `CONCAT` rank-1 appearances | `32/34` | `0/13 top16 rows` | reduced |
| `CONCAT` compute sum | `23114250 us` | `41401514 us` | `+79.1168%` |

## Decision

PC-4k proves the immediate barrier diagnosis and keeps the dim0 row-partition
path as a default-off candidate. It is not broad enough for a default flip or a
promotion patch yet.

Next gate:

- repeat clean qwen35moe `p8192/n1` A/B to bound noise;
- add wider shape coverage, especially non-single `ne2` and generated-token
  smoke;
- keep recurrent rollback and transposed-src CONCAT tests in the validation set;
- only then decide whether the opt-in path should be carried forward, narrowed,
  or retired.
