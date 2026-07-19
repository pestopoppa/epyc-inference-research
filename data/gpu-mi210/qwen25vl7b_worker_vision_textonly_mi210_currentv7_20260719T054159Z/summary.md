# Qwen2.5-VL-7B Worker Vision Text-Only MI210 Benchmark

Artifact: `/mnt/raid0/llm/epyc-inference-research/data/gpu-mi210/qwen25vl7b_worker_vision_textonly_mi210_currentv7_20260719T054159Z`

Exit code: `0`

| phase | prompt tokens | generation tokens | avg t/s | stddev t/s | device | build |
|---|---:|---:|---:|---:|---|---|
| prompt | 2048 | 0 | 3474.966969 | 0.0 | ROCm0 | 6a8dd5ea6 |
| decode | 0 | 1024 | 123.296409 | 0.0 | ROCm0 | 6a8dd5ea6 |

One bounded `llama-bench` invocation, one repetition, current experimental HIP build, `-ngl 99`, no mmproj, `OMP_NUM_THREADS=8`.

Cleanup verification: no `llama-bench` or `llama-server` executable processes; `rocm-smi` reports MI210 VRAM allocation `0%`.
