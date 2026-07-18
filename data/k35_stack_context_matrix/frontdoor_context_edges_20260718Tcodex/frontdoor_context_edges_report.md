# K35 Frontdoor Context-Edge Observation - 2026-07-18

Command:

```bash
python3 scripts/benchmark/k35_stack_context_matrix_runner.py \
  --execute \
  --only frontdoor_cpu_no_spec \
  --only frontdoor_gpu_resident_no_spec \
  --only frontdoor_gpu_native_mtp \
  --context 2048 --context 32768 \
  --max-tokens 1024 --min-completion-tokens 512 --reps 1 \
  --output-dir data/k35_stack_context_matrix/frontdoor_context_edges_20260718Tcodex
```

All six fresh-server cells completed 1024 generated tokens and cleaned up. The
runner recorded guard state, commands, per-cell raw responses, memory samples,
ROCm snapshots, and post-run process-blocker proof.

| Scenario | Nominal context | Prompt tokens | Completion tokens | Prompt t/s | Decode t/s | Draft accepted |
|---|---:|---:|---:|---:|---:|---:|
| CPU no-spec | 2048 | 134 | 1024 | 182.23 | 21.63 | 0 / 0 |
| CPU no-spec | 32768 | 30791 | 1024 | 114.20 | 10.15 | 0 / 0 |
| MI210 no-spec | 2048 | 134 | 1024 | 674.96 | 101.52 | 0 / 0 |
| MI210 no-spec | 32768 | 30791 | 1024 | 1765.07 | 78.14 | 0 / 0 |
| MI210 native MTP | 2048 | 134 | 1024 | 592.39 | 123.55 | 767 / 767 |
| MI210 native MTP | 32768 | 30791 | 1024 | 1681.27 | 105.17 | 767 / 767 |

Interpretation:

- This extends the earlier 8K `frontdoor_pgpu1_candidate_20260718Tquiet`
  observation to short and deep context edges.
- On this long repetitive structured-output prompt, MI210 native MTP is the
  fastest frontdoor lane at every measured context: 123.55 t/s at 2K, 119.69
  t/s at 8K in the prior n=5 run, and 105.17 t/s at 32K.
- Decode at 32K remains materially above the CPU 2K no-spec row and roughly
  10.4x the CPU 32K no-spec row.
- This is still observation-grade until `P-GPU-1` is ratified or these artifacts
  are explicitly retro-certified under the ratified protocol.

Cleanup:

- `summary.json` reports `cleanup_process_blockers: []`.
- Post-run checks showed no `llama-server`, K35 runner, AutoPilot, or KFD PIDs;
  `rocm-smi` reported GPU use 0% and VRAM allocated 0%.
