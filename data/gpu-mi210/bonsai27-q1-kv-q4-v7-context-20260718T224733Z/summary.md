# Bonsai-27B Q1_0 v7 MI210 KV-q4 context benchmark

Artifact: /mnt/raid0/llm/epyc-inference-research/data/gpu-mi210/bonsai27-q1-kv-q4-v7-context-20260718T224733Z

Rows:
- pp=512 tg=0: avg_ts=772.014825, stddev_ts=30.799268, samples=[750.236, 793.793], type_k=q4_0, type_v=q4_0
- pp=0 tg=128: avg_ts=11.113175, stddev_ts=0.044117, samples=[11.082, 11.1444], type_k=q4_0, type_v=q4_0
- pp=2048 tg=512: avg_ts=52.615406, stddev_ts=0.023134, samples=[52.6318, 52.599], type_k=q4_0, type_v=q4_0
- pp=8192 tg=512: avg_ts=151.107401, stddev_ts=0.093234, samples=[151.173, 151.041], type_k=q4_0, type_v=q4_0
- pp=32768 tg=512: avg_ts=330.059677, stddev_ts=0.295461, samples=[330.269, 329.851], type_k=q4_0, type_v=q4_0

Caveats:
- Speed-only observation with q4_0 KV cache; no task-quality or retention claim.
- Paired `-pg` rows are combined prompt+512 generation throughput, not isolated decode.
- Concurrent CPU-only Qwable benchmark may have used CPU/memory resources but no KFD/GPU process was active preflight.
