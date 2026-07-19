# Qwen3-VL-8B Text-Only MI210 GPU Benchmark

Artifact: `/mnt/raid0/llm/epyc-inference-research/data/gpu-mi210/qwen3vl8b_textonly_mi210_currentv7_20260719T053640Z`

Exit code: `0`

| prompt tokens | generation tokens | avg t/s | stddev t/s | devices | build |
|---:|---:|---:|---:|---|---|
| 2048 | 0 | 2275.956946 | 0.0 | ROCm0 | 6a8dd5ea6 |
| 0 | 1024 | 105.99538 | 0.0 | ROCm0 | 6a8dd5ea6 |

This is a single bounded `llama-bench` row using the experimental HIP build, `-dev ROCm0`, `-ngl 99`, no mmproj, and n=1024 generation.
