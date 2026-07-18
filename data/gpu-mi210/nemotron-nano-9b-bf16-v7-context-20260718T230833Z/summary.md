# Nemotron Nano 9B BF16 v7 MI210 context benchmark

Artifact: /mnt/raid0/llm/epyc-inference-research/data/gpu-mi210/nemotron-nano-9b-bf16-v7-context-20260718T230833Z

Rows:
- pp=512 tg=0: avg_ts=2250.667076, stddev_ts=73.487875, samples=[2198.7, 2302.63], type_k=f16, type_v=f16
- pp=0 tg=128: avg_ts=59.644627, stddev_ts=0.044544, samples=[59.6131, 59.6761], type_k=f16, type_v=f16
- pp=2048 tg=512: avg_ts=269.222998, stddev_ts=0.177785, samples=[269.349, 269.097], type_k=f16, type_v=f16
- pp=8192 tg=512: avg_ts=679.663268, stddev_ts=4.122361, samples=[682.578, 676.748], type_k=f16, type_v=f16
- pp=32768 tg=512: avg_ts=1223.900232, stddev_ts=1.128344, samples=[1224.7, 1223.1], type_k=f16, type_v=f16

Caveats:
- Speed-only llama-bench observation; no task-quality or role admission claim.
- Paired `-pg` rows are combined prompt+512 generation throughput, not isolated decode.
- llama-bench exposes no Nemotron serving prompt/template or reasoning controls; use server quality artifacts for admission decisions.
