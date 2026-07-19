# AXA-2 32K Prefill Variant: f16/q4_0 b1024/ub256 Rerun

Artifact: /mnt/raid0/llm/epyc-inference-research/data/gpu-mi210/axa2_32k_prefill_qwen35_122b_v1_f16k_q4v_b1024_ub256_rerun_20260719T070336Z
Stop reason: gpu_zero_6_polls_no_stdout_after_180s
Exit code: 143

| elapsed_s | gpu_use_pct | vram_alloc_pct | stdout_bytes |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 31 | 0 | 60 | 0 |
| 61 | 0 | 60 | 0 |
| 91 | 0 | 60 | 0 |
| 122 | 0 | 60 | 0 |
| 152 | 0 | 60 | 0 |
| 182 | 0 | 60 | 0 |
| 213 | 0 | 60 | 0 |
| 243 | 0 | 60 | 0 |
| 273 | 0 | 60 | 0 |
| 303 | 0 | 60 | 0 |
| 334 | 0 | 60 | 0 |

Rows:

Cleanup no KFD: true

Independent cleanup verification: no exact `llama-bench`, `llama-server`, or `llama-cli` process remained; ROCm reported no KFD PIDs. The generated `post_process_matches` field in `summary.json` includes the wrapper command text and should not be treated as a live llama process.
