# Qwen3.6 27B MTP f16-upcast v7 MI210 context benchmark

Artifact: /mnt/raid0/llm/epyc-inference-research/data/gpu-mi210/qwen36-27b-mtp-f16-upcast-v7-context-20260718T224253Z

Rows:
- pp=512 tg=0: avg_ts=930.090936, stddev_ts=0.0, samples=[930.091]
- pp=0 tg=128: avg_ts=19.339837, stddev_ts=0.0, samples=[19.3398]
- pp=2048 tg=512: avg_ts=87.66574, stddev_ts=0.0, samples=[87.6657]
- pp=8192 tg=512: avg_ts=226.275162, stddev_ts=0.0, samples=[226.275]
- pp=32768 tg=512: avg_ts=412.264802, stddev_ts=0.0, samples=[412.265]

Caveats:
- llama-bench exposes no MTP/speculative selector; this measures base llama-bench throughput for the MTP GGUF artifact.
- Paired `-pg` rows are combined prompt+512 generation throughput, not isolated decode.
- Concurrent CPU-only Qwable benchmark may have used CPU/memory resources but no KFD/GPU process was active preflight.
