# AXA-2 32K Prefill Variant: f16/f16 b1024/ub256

Artifact: `/mnt/raid0/llm/epyc-inference-research/data/gpu-mi210/axa2_32k_prefill_qwen35_122b_v1_f16kv_b1024_ub256_20260719T065143Z`
Stop reason: `completed`
Exit code: `0`

| elapsed_s | gpu_use_pct | vram_alloc_pct | stdout_bytes |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 31 | 100 | 60 | 0 |
| 61 | 100 | 60 | 0 |
| 91 | 100 | 60 | 0 |
| 121 | 100 | 60 | 0 |

Rows:

```json
[
  {
    "build_commit": "6a8dd5ea6",
    "build_number": 10097,
    "cpu_info": "AMD EPYC 9655 96-Core Processor",
    "gpu_info": "AMD Instinct MI210",
    "backends": "ROCm",
    "model_filename": "/mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-IQ2_M/Qwen3.5-122B-A10B-UD-IQ2_M.gguf",
    "model_type": "qwen35moe 122B.A10B IQ2_M - 2.7 bpw",
    "model_size": 40366862336,
    "model_n_params": 124635206144,
    "n_batch": 1024,
    "n_ubatch": 256,
    "n_threads": 32,
    "cpu_mask": "0x0",
    "cpu_strict": false,
    "poll": 50,
    "type_k": "f16",
    "type_v": "f16",
    "n_gpu_layers": 99,
    "n_cpu_moe": 0,
    "split_mode": "layer",
    "main_gpu": 0,
    "no_kv_offload": false,
    "flash_attn": 1,
    "devices": "ROCm0",
    "tensor_split": "0.00",
    "tensor_buft_overrides": "none",
    "use_mmap": true,
    "use_direct_io": false,
    "embeddings": false,
    "no_op_offload": 0,
    "no_host": false,
    "fit_target": 0,
    "fit_min_ctx": 0,
    "n_prompt": 32768,
    "n_gen": 0,
    "n_depth": 0,
    "test_time": "2026-07-19T06:51:49Z",
    "avg_ns": 66967467453,
    "stddev_ns": 0,
    "avg_ts": 489.312218,
    "stddev_ts": 0.0,
    "samples_ns": [
      66967467453
    ],
    "samples_ts": [
      489.312
    ]
  }
]
```

Cleanup: post_processes_empty=True, post_rocm_smi_no_kfd=True
