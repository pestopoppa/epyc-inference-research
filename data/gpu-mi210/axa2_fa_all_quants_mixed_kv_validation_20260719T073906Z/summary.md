# AXA-2 FA_ALL_QUANTS Mixed-KV Validation

Artifact: /mnt/raid0/llm/epyc-inference-research/data/gpu-mi210/axa2_fa_all_quants_mixed_kv_validation_20260719T073906Z
Stop reason: completed
Exit code: 0
FA_ALL_QUANTS: True

| ctk | ctv | fa | pp | avg_t/s | note |
|---|---|---:|---:|---:|---|
| f16 | f16 | 1 | 512 | 442.703164 | |
| q4_0 | q4_0 | 1 | 512 | 427.327762 | |
| q4_0 | f16 | 1 | 512 | 440.975915 | |
| f16 | q4_0 | 1 | 1 | 16.487315 | |
| q4_0 | f16 | 1 | 32768 | 415.30593 | |

Telemetry tail:
```
elapsed_s	gpu_use_pct	vram_alloc_pct	stdout_bytes
0	0	0	17
15	0	58	2139
31	5	58	4276
47	100	60	4276
63	100	60	4276
79	100	60	4276
95	100	60	4276
111	0	58	5342
```
Cleanup no KFD: true
