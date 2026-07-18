# Qwen3.6 27B MTP Q4_K_M v7 MI210 context benchmark

Artifact: /mnt/raid0/llm/epyc-inference-research/data/gpu-mi210/qwen36-27b-mtp-q4km-v7-context-20260718T223647Z

Rows:
- pp=512 tg=0: avg_ts=781.197774, stddev_ts=16.613609, samples=[769.45, 792.945]
- pp=0 tg=128: avg_ts=34.800303, stddev_ts=0.085804, samples=[34.861, 34.7396]
- pp=2048 tg=512: avg_ts=146.39687, stddev_ts=0.244542, samples=[146.57, 146.224]
- pp=8192 tg=512: avg_ts=327.542879, stddev_ts=2.180242, samples=[329.085, 326.001]
- pp=32768 tg=512: avg_ts=471.503662, stddev_ts=0.583264, samples=[471.916, 471.091]

Caveats:
- llama-bench exposes no MTP/speculative selector; this measures base llama-bench throughput for the MTP GGUF artifact.
- Paired `-pg` rows are combined prompt+512 generation throughput, not isolated decode.
- Concurrent CPU-only Qwable benchmark may have used CPU/memory resources but no KFD/GPU process was active preflight.
