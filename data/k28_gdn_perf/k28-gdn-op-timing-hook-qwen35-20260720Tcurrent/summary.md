# K28 GDN Direct Timing Hook — Qwen3.6-35B-A3B Q8

Date: 2026-07-20

Kernel: `/mnt/raid0/llm/llama.cpp-experimental`, branch `experimental-v7-refresh-20260716`, commit `93d945885`, build number `10100`.

Patch under test: default-off `GGML_CUDA_GDN_TIMING=1` HIP-event timing around `GGML_OP_GATED_DELTA_NET`. The hook requires `GGML_CUDA_DISABLE_GRAPHS=1` so it does not synchronize inside graph capture.

Validation:

- `cmake --build build-hip --target test-backend-ops -j 32` passed.
- `GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_GDN_TIMING=1 test-backend-ops test -o GATED_DELTA_NET -b ROCm0 -j 8` passed `38/38`.

Full-model command:

```bash
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin \
  GGML_CUDA_DISABLE_GRAPHS=1 \
  GGML_CUDA_GDN_TIMING=1 \
  /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-bench -v \
  -m /mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf \
  -ngl 99 -dev ROCm0 -fa on -p 2048,8192 -n 1 -r 1 -o json
```

Log: `qwen35_p2048_p8192_verbose.log`.

| Cell | Prompt t/s | Prompt wall | GDN measured | GDN share |
|---|---:|---:|---:|---:|
| p2048/n0 | 2073.75 | 987.58 ms | 152.63 ms | 15.45% |
| p8192/n0 | 1975.94 | 4145.87 ms | 606.90 ms | 14.64% |

Interpretation: direct HIP-event timing validates the earlier Phase-0 ceiling model rather than raising it. A hypothetical 4x GDN op kernel maps to about 11.59% p2048 and 10.98% p8192 full-model prompt gain. K28 remains plausible as a default-off/post-promotion fused recurrence project, but this evidence does not justify delaying frozen v7 promotion for Phase 1.
