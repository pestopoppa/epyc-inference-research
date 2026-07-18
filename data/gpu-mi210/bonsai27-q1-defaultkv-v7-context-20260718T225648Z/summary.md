# Bonsai-27B Q1_0 v7 MI210 default-KV context benchmark

Artifact: /mnt/raid0/llm/epyc-inference-research/data/gpu-mi210/bonsai27-q1-defaultkv-v7-context-20260718T225648Z

Rows:
- pp=512 tg=0: avg_ts=780.906553, stddev_ts=23.108986, samples=[764.566, 797.247], type_k=f16, type_v=f16
- pp=0 tg=128: avg_ts=11.153425, stddev_ts=0.002766, samples=[11.1515, 11.1554], type_k=f16, type_v=f16
- pp=2048 tg=512: avg_ts=52.71504, stddev_ts=0.067504, samples=[52.6673, 52.7628], type_k=f16, type_v=f16
- pp=8192 tg=512: avg_ts=152.198285, stddev_ts=0.00911, samples=[152.192, 152.205], type_k=f16, type_v=f16
- pp=32768 tg=512: avg_ts=334.844484, stddev_ts=0.392475, samples=[334.567, 335.122], type_k=f16, type_v=f16

Caveats:
- Speed-only observation with default KV cache; no task-quality or retention claim.
- Paired `-pg` rows are combined prompt+512 generation throughput, not isolated decode.
- Concurrent CPU-only Qwable benchmark may have used CPU/memory resources but no KFD/GPU process was active preflight except its hidden-device process with 0 VRAM.
