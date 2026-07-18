# K35 Ingest Long-Context Curve - 2026-07-18

Command:

```bash
python3 scripts/benchmark/k35_stack_context_matrix_runner.py \
  --execute \
  --only ingest_long_context_cpu_default_experts \
  --context 2048 --context 8192 --context 32768 \
  --max-tokens 1024 --min-completion-tokens 512 --reps 1 \
  --output-dir data/k35_stack_context_matrix/ingest_long_context_curve_20260718Tcodex
```

All three fresh-server cells completed 1024 generated tokens and cleaned up.
The lane is the production-shaped CPU Qwen3-Next ingest-long-context path with
default expert count, q4_0 KV, flash attention, jinja, mlock, and no
speculative decoding.

| Nominal context | Prompt tokens | Completion tokens | Prompt t/s | Decode t/s | Draft accepted |
|---:|---:|---:|---:|---:|---:|
| 2048 | 128 | 1024 | 150.96 | 20.52 | 0 / 0 |
| 8192 | 6208 | 1024 | 172.45 | 15.93 | 0 / 0 |
| 32768 | 30785 | 1024 | 96.71 | 9.72 | 0 / 0 |

Interpretation:

- The ingest lane shows a steep decode slope with context depth: 20.52 t/s at
  2K, 15.93 t/s at 8K, and 9.72 t/s at 32K on this 1024-token
  structured-output prompt.
- Prefill also tapers materially by the 32K cell, from the short-context
  150.96 t/s row to 96.71 t/s over 30,785 prompt tokens.
- This is the current optimized baseline row for ingest-long-context in the K35
  stack table. It does not exercise native MTP, n-gram, sparse DSA, or GPU
  offload.

Cleanup:

- `summary.json` reports `cleanup_process_blockers: []`.
- ROCm snapshots show no GPU use for this CPU-only lane; post-run process checks
  found no leftover `llama-server`, K35 runner, AutoPilot, or KFD process.
