# StreamingLLM Floor Sweep

Mode: `execute`
Model: `/mnt/raid0/llm/models/Qwen3-1.7B-Q8_0.gguf`
Binary: `/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-completion`
Context/tokens: `384` / `768`

| Arm | sink | window | rc | decode t/s | ratio | quality | marker | alpha | numbered |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|
| baseline_context_shift | 0 | 0 | 0 | 31.17 | 1.000 | fail | no | 31 | 31 |
| streaming_sink8_window128 | 8 | 128 | 0 | 31.50 | 1.011 | fail | no | 31 | 31 |
| streaming_sink16_window192 | 16 | 192 | 0 | 31.28 | 1.004 | fail | no | 31 | 31 |
| streaming_sink32_window256 | 32 | 256 | 0 | 30.86 | 0.990 | fail | no | 31 | 31 |

Admission decision:
{
  "admit_cluster": false,
  "reason": "no_streaming_cluster_passed_quality_and_speed_floor",
  "candidate_rows": []
}

This artifact is observation-grade only. It is not a production serving or P-GPU-1 claim.
