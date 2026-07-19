# GLM-5.2 native-MTP A/B - 2026-07-19

## Scope

Paired CPU-only GLM-5.2 UD-IQ2_M long-context serving A/B on experimental v7
`experimental-v7-refresh-20260716`, intended to close the retained GLM native-MTP
acceptance/throughput gate after the GLM reviewer route-away verdict.

Artifact root:
`data/glm52_native_mtp_ab/glm52-native-mtp-ab-20260719T185837Z/`

Runner:
`scripts/benchmark/glm52_dsa_probe_runner.py`

Common shape:

- Stage: `long_context_dsa_probe`
- Context: `4096`
- Prompt floor: `2500` tokens
- Generated-token cap: `512`
- Completion floor: `384`
- `glm-dsa.attention.indexer.top_k=4096`
- Server trace logs enabled

## Result

| Arm | Status | Prompt tokens | Completion tokens | Prompt t/s | Decode t/s | Streaming chunks | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| `--spec-type none` | `ok` | 2931 | 512 | 22.56 | 2.49 | 514 | Baseline served to the length cap; the output entered reasoning text and did not prove strict content quality. |
| `--spec-type draft-mtp` | `failed_completion_floor` | 2931 | 0 | n/a | n/a | 0 | GLM same-model MTP context initialized, then the streamed request returned no chunks, no usage, and no timings. |

## Failure Signature

The draft-MTP server log confirms the intended path was reached:

- `common_speculative_init_result: creating MTP draft context`
- `common_specu: adding speculative implementation 'draft-mtp'`
- `n_max=3, n_min=0, p_min=0.00, n_embd=6144, backend_sampling=1`
- `speculative decoding context initialized`
- request accepted with `task.n_tokens = 2931`

After that point the draft-MTP run produced no decode checkpoints, no prompt-eval final timing,
no response chunks, and no API usage object. The corresponding no-spec run on the same prompt
completed normally. This is therefore a functional GLM native-MTP serving failure, not a measured
alpha or throughput win.

## Decision

Native GLM-MTP is not releasable from the current v7 scaffold. The remaining GLM acceleration
work is a repair task for the GLM/DeepSeek32 `DECODER_MTP` serving path and speculative lifecycle.
It should not be treated as a closed performance win or promoted from the earlier one-token /
eight-token scaffold smokes.

If the operator keeps GLM acceleration coupled to v7, v7 remains blocked on this repair. If the
operator decouples GLM acceleration from the v7 release after the reviewer route-away verdict, this
failed gate becomes post-promotion research work rather than a release blocker.
