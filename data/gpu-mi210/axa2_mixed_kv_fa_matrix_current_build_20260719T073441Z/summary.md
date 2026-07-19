# AXA-2 Mixed-KV Flash Matrix: Current Build

Artifact: /mnt/raid0/llm/epyc-inference-research/data/gpu-mi210/axa2_mixed_kv_fa_matrix_current_build_20260719T073441Z
Stop reason: completed
Exit code: 1

| ctk | ctv | fa | pp | avg_t/s | note |
|---|---|---:|---:|---:|---|
| q4_0 | f16 | 1 | 4096 | 372.700671 | |
| q4_0 | f16 | -1 | 4096 | 564.271867 | |
| q4_0 | f16 | 0 | 4096 | 567.229698 | |
| f16 | q4_0 | 1 | 512 | 454.66935 | |

Telemetry tail:
```
elapsed_s	gpu_use_pct	vram_alloc_pct	stdout_bytes
0	0	0	30
15	0	59	30
31	0	0	3246
```
Cleanup no KFD: true
