# K11 CPU No-Spec Single-Slot Token Trace Control

Artifact: `data/k11_gemma4_determinism/k11_natural_freeform_orig_q4_cpu_nospec_np1_explicit_greedy_trace_20260720T114120Z`

## Result

- Runs OK: `10/10`
- Unique output hashes: `1`
- Task passed: `False`
- Finish reasons: `{'stop': 10}`
- Word counts: `[156, 156, 156, 156, 156, 156, 156, 156, 156, 156]`
- Mean decode: `21.124 t/s`

## Interpretation

CPU no-spec single-slot is deterministic on the same prompt and tracing shape. Paired GPU no-spec single-slot diverged, so the natural-prose defect is GPU/backend-path dependent for this model and prompt.
