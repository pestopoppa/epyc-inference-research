# AXA-2 32K Prefill Variant: q4_0/f16 b1024/ub256

Artifact: `/mnt/raid0/llm/epyc-inference-research/data/gpu-mi210/axa2_32k_prefill_qwen35_122b_v1_q4k_f16v_b1024_ub256_20260719T064333Z`
Status: manual watchdog stop; no completed `llama-bench` row.

Command:

```bash
env ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin:/opt/rocm/lib /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-bench -m /mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-IQ2_M/Qwen3.5-122B-A10B-UD-IQ2_M.gguf -p 32768 -n 0 -t 32 -b 1024 -ub 256 -ctk q4_0 -ctv f16 -fa on -r 1 --progress
```

Watch summary:

| elapsed_s | gpu_use_pct | vram_alloc_pct | stdout_bytes |
|---:|---:|---:|---:|
| 30 | 0 | 60 | 0 |
| 60 | 100 | 60 | 0 |
| 90 | 0 | 60 | 0 |
| 120 | 0 | 60 | 0 |
| 150 | 0 | 60 | 0 |
| 180 | 0 | 60 | 0 |

Stop reason: `gpu_zero_4_polls_no_stdout`.
Cleanup: post_rocm_smi_no_kfd=True.
