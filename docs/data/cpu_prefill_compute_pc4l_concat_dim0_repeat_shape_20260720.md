# PC-4l qwen35moe CONCAT dim0 repeat/shape gate

Date: 2026-07-20

Status: complete. Verdict: carry forward as an experimental, default-off
candidate. Not default-on and not promotion-ready.

Run root:
`/mnt/raid0/llm/epyc-inference-research/data/cpu_prefill_compute/pc4l-qwen35moe-concat-dim0-repeat-20260720T023955Z/`

Sanitized summary:
`/mnt/raid0/llm/epyc-inference-research/data/cpu_prefill_compute/pc4l-qwen35moe-concat-dim0-repeat-20260720T023955Z/summary.json`

## Scope

PC-4k showed that `GGML_CPU_CONCAT_DIM0_ROWS=1` removes the
`conv_input-*` dim0 `CONCAT` barrier in qwen35moe `build_conv_state()`. PC-4l
tested whether that result survives a repeat, a generated-token smoke, and a
multi-sequence path.

All runs used the experimental v7 worktree only:
`/mnt/raid0/llm/llama.cpp-experimental`.

Common environment:

- `LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-k24-cpu/bin`
- `GGML_IQK=1`
- `HIP_VISIBLE_DEVICES=-1`
- `ROCR_VISIBLE_DEVICES=-1`
- `taskset -c 0-95 numactl --interleave=all`
- `-dev none`, `-ngl 0`, `-nopo 1`, `-nkvo 1`

IQK activation lines were present in every arm. Postflight found no active
`llama-bench`, `llama-batched-bench`, `llama-server`, or KFD GPU PIDs.

## Repeat A/B

Command shape: `llama-bench -p 8192 -n 1 -r 3`.

| Arm | pp8192 | tg1 | Wall |
| --- | ---: | ---: | ---: |
| default | `95.531624 t/s` | `4.678276 t/s` | `5:55.26` |
| `GGML_CPU_CONCAT_DIM0_ROWS=1` | `104.210589 t/s` | `4.778680 t/s` | `5:27.92` |

Deltas:

- pp8192: `+9.0849%`
- tg1: `+2.1462%`

## Generated-Token Smoke

Command shape: `llama-bench -pg 8192,16 -r 2`.

Note: this `llama-bench` mode also emitted default `pp512` and `tg128` rows
because `-pg` does not clear the default `-p` / `-n` lists.

| Row | Default | `GGML_CPU_CONCAT_DIM0_ROWS=1` | Delta |
| --- | ---: | ---: | ---: |
| pp8192+tg16 | `88.838786 t/s` | `93.782587 t/s` | `+5.5649%` |
| tg128 | `4.641114 t/s` | `4.654019 t/s` | `+0.2781%` |
| pp512 | `95.853716 t/s` | `113.058759 t/s` | `+17.9493%` |

## Multi-Sequence Smoke

Built `llama-batched-bench` in the experimental build tree and ran:
`-npp 2048 -ntg 1 -npl 2 --output-format jsonl`.

This exercises `pl=2` / `n_seq_max=2` rather than the single-sequence
`llama-bench` path.

| Metric | Default | `GGML_CPU_CONCAT_DIM0_ROWS=1` | Delta |
| --- | ---: | ---: | ---: |
| speed_pp | `169.369247 t/s` | `261.157013 t/s` | `+54.1939%` |
| speed_tg | `13.472096 t/s` | `13.896030 t/s` | `+3.1468%` |
| total speed | `168.418091 t/s` | `258.908661 t/s` | `+53.7297%` |
| t_pp | `24.183847 s` | `15.684051 s` | `-35.1466%` |
| total time | `24.332302 s` | `15.827976 s` | `-34.9508%` |

## Decision

The PC-4l repeat/shape gate is positive. The dim0 row-partition path should be
carried forward as a default-off experimental candidate.

It still must not become default-on or promotion code from this evidence alone.
Next gate:

- source-hardening/code-review the `GGML_CPU_CONCAT_DIM0_ROWS=1` path;
- expand backend correctness coverage beyond F32/F16/BF16 transposed dim0 and
  existing CONCAT cases;
- keep recurrent rollback tests in the validation set;
- decide whether this remains an env-gated path, becomes a narrower
  qwen35/qwen3next-specific path, or is retired before any production candidate.
