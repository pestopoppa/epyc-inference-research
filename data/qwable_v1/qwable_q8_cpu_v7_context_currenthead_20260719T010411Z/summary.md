# Qwable Q8 CPU v7 Context Current Head

Artifact: `/mnt/raid0/llm/epyc-inference-research/data/qwable_v1/qwable_q8_cpu_v7_context_currenthead_20260719T010411Z`
Exit code: `0`
Experimental head: `ed4091266d286045510e498ceb059c209a65aff9`

## Rows

- p2048 / n0: `142.729602` tok/s, threads `16`, devices `none`, ngl `0`, build `ed4091266`
- p8192 / n0: `118.585662` tok/s, threads `16`, devices `none`, ngl `0`, build `ed4091266`
- p32768 / n0: `94.735757` tok/s, threads `16`, devices `none`, ngl `0`, build `ed4091266`
- p0 / n512: `10.742701` tok/s, threads `16`, devices `none`, ngl `0`, build `ed4091266`

## Command

```bash
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin GGML_IQK=1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-bench -m /mnt/raid0/llm/models/Qwable-v1-GGUF/Qwable-v1.Q8_0.gguf -dev none -ngl 0 -fa off -t 16 -p 2048,8192,32768 -n 512 -r 1 -o json --progress
```
