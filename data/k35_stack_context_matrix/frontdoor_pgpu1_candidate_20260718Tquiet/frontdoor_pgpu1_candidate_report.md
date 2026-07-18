# Frontdoor P-GPU-1 Candidate Observation

Date: 2026-07-18

Status: observation-grade. `P-GPU-1` is still marked deferred in `MEASUREMENT.md`, so this artifact does not by itself close Gate R as a decision-grade measurement.

## Protocol Shape

- Runner: `scripts/benchmark/k35_stack_context_matrix_runner.py`
- Output directory: `data/k35_stack_context_matrix/frontdoor_pgpu1_candidate_20260718Tquiet/`
- Model: `/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf`
- Binary: `/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server`
- Version: `10088 (d1e5a20eb)`, experimental worktree clean
- Device: `ROCm0: AMD Instinct MI210 (65520 MiB, 65416 MiB free)`
- Shape: nominal 8192 context, actual `6214` prompt tokens, `1024` completion tokens per rep
- Reps: `n=5` sequential fresh-server reps per arm
- Host policy: quiet host; `guard_state.json` recorded no process blockers; `summary.json` recorded no cleanup blockers
- Artifact fields: exact commands, guard state, device listing, process memory samples, ROCm snapshots, raw responses, per-rep results, and cleanup proof

## Results

| Arm | Decode median | MAD | Prompt median | Speedup vs CPU | Draft acceptance |
|---|---:|---:|---:|---:|---:|
| CPU no-spec | `17.10 t/s` | `0.06` | `192.67 t/s` | `1.00x` | n/a |
| MI210 no-spec | `95.39 t/s` | `0.24` | `2036.85 t/s` | `5.58x` | n/a |
| MI210 native MTP | `119.69 t/s` | `0.35` | `1947.57 t/s` | `7.00x` | `3835/3835` |

All `15/15` cells completed `1024` generated tokens, passed the `512` token floor, and cleaned up with dead server PIDs.

## Interpretation

This is strong frontdoor residency evidence for the longer repetitive output shape. MI210 no-spec beats the same-window CPU re-anchor by `5.58x`, and native MTP beats CPU by `7.00x`. Unlike the short Stage-2 prompt pack, this run shows native MTP is useful when the frontdoor is producing long repetitive structured text: `119.69 t/s` vs `95.39 t/s` no-spec, with `100%` accepted draft tokens.

Do not generalize this to all frontdoor traffic without task-class quality and acceptance monitoring. The result supports promoting a real P-GPU-1 ratification/update and using this runner shape for Gate R closeout.
