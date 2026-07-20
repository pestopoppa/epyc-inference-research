# K11 GPU Backend-Path A/B

Artifact set: `data/k11_gemma4_determinism/k11_gpu_backend_path_ab_20260720T1204Z`

## Verdict

K11 natural-prose nondeterminism persists on MI210 no-spec single-slot with pre-sampling probabilities captured. CPU no-spec single-slot is deterministic on the same prompt, while GPU diverges with graphs on, graphs off, and flash-attention off. The remaining root cause is therefore broader GPU backend numerical nondeterminism or logits handoff, not external-head MTP, multi-slot scheduling, HIP graph replay, or flash-attention alone.

The invalid first attempt `k11_natural_freeform_orig_q4_gpu_nospec_np1_presampling_trace_20260720T115054Z` is not admitted: it used the default compact JSON prompt instead of the natural 160-word prompt. It was moved out of the repo to `/tmp/k11_invalid_default_prompt_20260720T115054Z`.

## Results

| Arm | Unique hashes | Mean decode t/s | First divergence | Branches | Notes |
|---|---:|---:|---:|---|---|
| GPU graphs on, FA on, pre-sampling | 10 | 72.52 | 16 | ` a` x4; ` serving` x6 | top-1/top-2 logprob gap range `0.0587-0.5555` |
| GPU graphs off, FA on, pre-sampling | 10 | 66.48 | 7 | ` a` x2; ` maintaining` x8 | `GGML_CUDA_DISABLE_GRAPHS=1`; graph replay is not the fix |
| GPU graphs on, FA off, pre-sampling | 9 | 79.61 | 16 | ` a` x5; ` serving` x5 | `-fa off`; flash-attention is not the fix |

All three valid GPU arms returned `finish_reason=stop` for `10/10` runs and wrote pre-sampling `top_logprobs`.

## Evidence Paths

- `data/k11_gemma4_determinism/k11_natural_freeform_orig_q4_gpu_nospec_np1_presampling_trace_corrected_20260720T115334Z/summary.json`
- `data/k11_gemma4_determinism/k11_natural_freeform_orig_q4_gpu_nospec_np1_presampling_trace_graphs_off_20260720T115655Z/summary.json`
- `data/k11_gemma4_determinism/k11_natural_freeform_orig_q4_gpu_nospec_np1_presampling_trace_fa_off_20260720T120133Z/summary.json`

## Next

Inspect GPU backend matmul/reduction determinism and logits copy/handoff. Do not spend more K11 time repeating MTP, slot-count, graph, or flash-attention toggles unless the new run changes the actual GPU math path and records that command surface.
