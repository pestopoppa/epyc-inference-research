# Nemotron Nano 9B Q8_0 v7 MI210 context benchmark

Artifact: /mnt/raid0/llm/epyc-inference-research/data/gpu-mi210/nemotron-nano-9b-q8-v7-context-20260718T230602Z

Rows:
- pp=512 tg=0: avg_ts=2052.901773, stddev_ts=42.035275, samples=[2023.18, 2082.63], type_k=f16, type_v=f16
- pp=0 tg=128: avg_ts=83.302708, stddev_ts=0.118706, samples=[83.2188, 83.3866], type_k=f16, type_v=f16
- pp=2048 tg=512: avg_ts=358.140989, stddev_ts=0.214575, samples=[358.293, 357.989], type_k=f16, type_v=f16
- pp=8192 tg=512: avg_ts=837.590811, stddev_ts=0.178065, samples=[837.717, 837.465], type_k=f16, type_v=f16
- pp=32768 tg=512: avg_ts=1279.524608, stddev_ts=1.689037, samples=[1280.72, 1278.33], type_k=f16, type_v=f16

Caveats:
- Speed-only llama-bench observation; no task-quality or role admission claim.
- Paired `-pg` rows are combined prompt+512 generation throughput, not isolated decode.
- llama-bench exposes no Nemotron serving prompt/template or reasoning controls; use server quality artifacts for admission decisions.
