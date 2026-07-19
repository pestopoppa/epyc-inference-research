# AXA-2 Qwen3.5 122B IQ2_M MI210 Prefill Sizing

Artifact: `/mnt/raid0/llm/epyc-inference-research/data/gpu-mi210/axa2-qwen35-122b-iq2m-prefill-sizing-20260719T060039Z`
Exit code: `15` (SIGTERM after partial run)

Completed rows from `partial_rows.tsv`:

| n_prompt | n_gen | avg t/s | KV | device |
|---:|---:|---:|---|---|
| 2048 | 0 | 342.056712 | q4_0/f16 | ROCm0 |
| 8192 | 0 | 135.559948 | q4_0/f16 | ROCm0 |
| 16384 | 0 | 76.519188 | q4_0/f16 | ROCm0 |

Missing rows: `32768` prefill and `tg512` decode did not complete before stop. The raw `llama_bench_stdout.json` is intentionally partial JSON; use `summary.json` or `partial_rows.tsv` for the salvaged observation rows.

Cleanup: post_processes_empty=True, post_kfd_scan_empty=True, post_rocm_smi_no_kfd=True
