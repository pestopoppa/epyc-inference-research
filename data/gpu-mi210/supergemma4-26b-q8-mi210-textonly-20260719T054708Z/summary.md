# SuperGemma4-26B MI210 text-only llama-bench

- Status: `ok`; exit code: `0`
- Model: `/mnt/raid0/llm/models/supergemma4-26b-abliterated-multimodal-8bit/supergemma4-26b-abliterated-multimodal-Q8_0.gguf`
- Binary: `/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-bench`
- Experimental branch/head: `experimental-v7-refresh-20260716` / `6a8dd5ea680860f790f98136cc370e400c3dabe3`
- Workload: requested `pp2048/tg1024`, one repetition, `ROCm0`, `ngl=99`, text-only
- Requested row: `n_prompt=2048`, `n_gen=1024`, `avg_ts=238.131863 t/s`, `stddev_ts=0.0`, backend `ROCm`
- The JSON also contains llama-bench standalone `pp512` and `tg128` records emitted by the same invocation.

Cleanup verification: `/mnt/raid0/llm/epyc-inference-research/data/gpu-mi210/supergemma4-26b-q8-mi210-textonly-20260719T054708Z/post_rocm_smi.txt` shows 0% use, 0% VRAM, and no KFD PIDs; `/mnt/raid0/llm/epyc-inference-research/data/gpu-mi210/supergemma4-26b-q8-mi210-textonly-20260719T054708Z/post_processes.txt` contains no llama process.
