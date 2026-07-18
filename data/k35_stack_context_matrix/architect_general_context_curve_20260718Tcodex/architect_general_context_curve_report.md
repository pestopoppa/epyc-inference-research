# K35 Architect-General Context Curve - 2026-07-18

Command:

```bash
python3 scripts/benchmark/k35_stack_context_matrix_runner.py \
  --execute \
  --only architect_general_cpu_native_mtp \
  --context 2048 --context 8192 \
  --max-tokens 1024 --min-completion-tokens 512 --reps 1 \
  --output-dir data/k35_stack_context_matrix/architect_general_context_curve_20260718Tcodex
```

Both fresh-server cells completed 1024 generated tokens and cleaned up. The
lane is the production-shaped Qwen3.5-122B architect CPU path with native
NEXTN/draft-MTP, q4_0/f16 KV, jinja, mlock, and thinking disabled.

| Nominal context | Prompt tokens | Completion tokens | Prompt t/s | Decode t/s | Draft accepted |
|---:|---:|---:|---:|---:|---:|
| 2048 | 134 | 1024 | 89.69 | 23.89 | 818 / 820 |
| 8192 | 6214 | 1024 | 143.02 | 20.72 | 818 / 819 |

Interpretation:

- The architect lane is much slower than frontdoor/worker, but native MTP
  acceptance is effectively saturated on this structured-output prompt.
- Decode falls modestly from 23.89 t/s at 2K to 20.72 t/s at 8K.
- The runner skips 14K/32K for this scenario because the production-shaped
  architect config uses two slots under a 16K context cap.

Cleanup:

- `summary.json` reports `cleanup_process_blockers: []`.
- Post-run checks found no leftover `llama-server`, K35 runner, AutoPilot, or
  KFD process.
