# Nemotron Nano 9B Q8_0 v7 MI210 q8-KV context benchmark

Artifact: /mnt/raid0/llm/epyc-inference-research/data/gpu-mi210/nemotron-nano-9b-q8-kvq8-v7-context-20260718T231115Z

Rows:
- pp=512 tg=0: avg_ts=2042.478727, stddev_ts=40.157887, samples=[2014.08, 2070.87], type_k=q8_0, type_v=q8_0
- pp=0 tg=128: avg_ts=82.975179, stddev_ts=0.107908, samples=[82.8989, 83.0515], type_k=q8_0, type_v=q8_0
- pp=2048 tg=512: avg_ts=355.611807, stddev_ts=0.225358, samples=[355.771, 355.452], type_k=q8_0, type_v=q8_0
- pp=8192 tg=512: avg_ts=826.756473, stddev_ts=0.212854, samples=[826.907, 826.606], type_k=q8_0, type_v=q8_0
- pp=32768 tg=512: avg_ts=1261.532514, stddev_ts=2.287416, samples=[1263.15, 1259.92], type_k=q8_0, type_v=q8_0

Caveats:
- Speed-only llama-bench observation with q8_0/q8_0 KV cache.
- Paired `-pg` rows are combined prompt+512 generation throughput, not isolated decode.
- Role/admission decisions still depend on server quality artifacts.
