# K35 Worker-General Context Curve - 2026-07-18

Command:

```bash
python3 scripts/benchmark/k35_stack_context_matrix_runner.py \
  --execute \
  --only worker_general_cpu_composed_spec \
  --context 2048 --context 8192 --context 14000 \
  --max-tokens 1024 --min-completion-tokens 512 --reps 1 \
  --output-dir data/k35_stack_context_matrix/worker_general_context_curve_20260718Tcodex
```

All three fresh-server cells completed 1024 generated tokens and cleaned up. The
lane is the production-shaped CPU Gemma4 worker with composed
`ngram-mod,draft-mtp`, assistant v6 Q8 head, q8 KV, reasoning off, and
`--spec-draft-n-max 2`.

| Nominal context | Prompt tokens | Completion tokens | Prompt t/s | Decode t/s | Draft accepted |
|---:|---:|---:|---:|---:|---:|
| 2048 | 135 | 1024 | 199.59 | 175.75 | 996 / 1173 |
| 8192 | 6215 | 1024 | 246.03 | 110.03 | 996 / 1173 |
| 14000 | 12024 | 1024 | 233.55 | 97.57 | 996 / 1173 |

Interpretation:

- The optimized worker lane is fast at short context but shows a real decode
  slope with context depth: 175.75 t/s at 2K, 110.03 t/s at 8K, and 97.57 t/s
  at 14K on this 1024-token structured-output prompt.
- Draft acceptance is stable across contexts at 996/1173 accepted tokens
  (84.91%), so the context-depth slowdown is not caused by a collapse in
  drafter acceptance.
- This gives the worker row needed for the stack throughput-vs-context table;
  broader deployment conclusions still depend on the ratified K35/P-GPU
  measurement framing.

Cleanup:

- `summary.json` reports `cleanup_process_blockers: []`.
- ROCm snapshots show no GPU use for this CPU-only lane; post-run process checks
  found no leftover `llama-server` or AutoPilot process.
