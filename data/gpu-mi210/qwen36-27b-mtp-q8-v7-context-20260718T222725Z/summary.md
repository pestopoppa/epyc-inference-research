# Qwen3.6 27B MTP Q8 v7 MI210 context benchmark

Artifact: /mnt/raid0/llm/epyc-inference-research/data/gpu-mi210/qwen36-27b-mtp-q8-v7-context-20260718T222725Z
Command:
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin:/opt/rocm/lib /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-bench -m /mnt/raid0/llm/models/Qwen3.6-27B-MTP-Q8_0.gguf -dev ROCm0 -ngl 99 -fa on -pg 2048,512 -pg 8192,512 -pg 32768,512 -r 2 -o json --progress

Experimental source commit:
cf051d3e18c7d8d898581f42a468602c7f6bade0

Exit code / duration seconds:
0
311

Metrics (n_prompt, n_gen, avg_ts tok/s, stddev_ts, samples_ts):
- pp=512 tg=0: avg_ts=839.720441, stddev_ts=32.776731, samples=[816.544, 862.897]
- pp=0 tg=128: avg_ts=30.849266, stddev_ts=0.010980, samples=[30.8415, 30.857]
- pp=2048 tg=512: avg_ts=134.703575, stddev_ts=0.027041, samples=[134.723, 134.684]
- pp=8192 tg=512: avg_ts=311.898449, stddev_ts=0.036870, samples=[311.925, 311.872]
- pp=32768 tg=512: avg_ts=477.888607, stddev_ts=1.393851, samples=[478.874, 476.903]

Caveats:
- llama-bench help exposes no MTP/speculative selector; this run did not fake MTP and measures base llama-bench throughput for the MTP GGUF artifact.
- -pg emitted baseline prompt-only/generation-only rows plus the requested paired pp/tg context rows.
- LD_LIBRARY_PATH was overridden to /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin:/opt/rocm/lib because the inherited environment would otherwise bind production v6 shared libraries.

Cleanup:
- final_rocm_smi_showpids.txt reports no KFD PIDs currently running.
- The MI210 benchmark PID 4083533 is gone.
- final_pgrep_llama.txt is not empty because another agent has an unrelated CPU-only llama-bench run active under build-k24-cpu; this task did not kill or modify that process.
