# AXA-2 32K Prefill Variant: q4_0/f16 no-warmup b1024/ub256

Artifact: /mnt/raid0/llm/epyc-inference-research/data/gpu-mi210/axa2_32k_prefill_qwen35_122b_v1_q4k_f16v_no_warmup_b1024_ub256_20260719T072934Z
Stop reason: manual_stop_cpu_fallback_zero_gpu_no_stdout_after_182s
Exit code: 143

| elapsed_s | gpu_use_pct | vram_alloc_pct | stdout_bytes |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 32 | 0 | 60 | 0 |
| 62 | 0 | 60 | 0 |
| 92 | 0 | 60 | 0 |
| 122 | 0 | 60 | 0 |
| 152 | 0 | 60 | 0 |
| 182 | 0 | 60 | 0 |

Rows:

CPU fallback signal: 0% GPU / 60% VRAM / 0 stdout bytes at 182s; manual stop to avoid burning the host on unsupported mixed-flash path.
Cleanup no KFD: true
