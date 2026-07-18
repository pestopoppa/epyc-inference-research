# Qwable Q8 CPU v7 context run2

Artifact: `/mnt/raid0/llm/epyc-inference-research/data/qwable_v1/qwable_q8_cpu_v7_context_run2_20260718T224956Z`

Command: `env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin GGML_IQK=1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-bench -m /mnt/raid0/llm/models/Qwable-v1-GGUF/Qwable-v1.Q8_0.gguf -dev none -ngl 0 -fa off -t 16 -p 2048,8192,32768 -n 512 -r 1 -o json --progress`

Exit code: `0`; duration: `883` seconds.

Source commit: `cf051d3e18c7d8d898581f42a468602c7f6bade0`

Binary: `/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-bench`

Metrics:

- pp2048 tg0: 145.233612 t/s, threads=16, backend=ROCm, devices=none, ngl=0, fa=0
- pp8192 tg0: 121.325412 t/s, threads=16, backend=ROCm, devices=none, ngl=0, fa=0
- pp32768 tg0: 99.107122 t/s, threads=16, backend=ROCm, devices=none, ngl=0, fa=0
- pp0 tg512: 10.838663 t/s, threads=16, backend=ROCm, devices=none, ngl=0, fa=0

Caveats:

- speed-only llama-bench observation, not a quality/admission gate
- single repetition, so stddev is zero by construction
- build/bin default probe failed with unresolved symbol; used current experimental build-hip binary with pinned LD_LIBRARY_PATH
- HIP/ROCR devices hidden and llama-bench argv used -dev none -ngl 0 -fa off; JSON reports devices=none and n_gpu_layers=0
- post/final checks include unrelated GPU llama-bench processes from other sessions
