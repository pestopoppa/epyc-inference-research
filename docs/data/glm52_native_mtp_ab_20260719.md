# GLM-5.2 native-MTP A/B - 2026-07-19

## Scope

Paired CPU-only GLM-5.2 UD-IQ2_M long-context serving A/B on experimental v7
`experimental-v7-refresh-20260716`, intended to close the retained GLM native-MTP
acceptance/throughput gate after the GLM reviewer route-away verdict.

Initial A/B artifact root:
`data/glm52_native_mtp_ab/glm52-native-mtp-ab-20260719T185837Z/`

Repair validation artifact:
`data/glm52_native_mtp_ab/glm52-native-mtp-draft-long-repair-20260719T195037Z/`

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
| `--spec-type draft-mtp` after DeepSeek32/GLM-DSA NEXTN row-selection repair | `ok` | 2919 | 512 | 22.77 | 5.33 | 515 | Matched long-context retry served to length cap with `draft_n=403`, `draft_n_accepted=376`, alpha `0.933`, mean accepted length `3.79`. |

The repaired arm is a `2.14x` decode-speed improvement over the no-spec baseline on this
long-output serving shape (`5.33 / 2.49`). This is an acceleration/serving result only: the
prompt intentionally asked for a long repeated sequence, and the GLM output still entered
reasoning text, so this row does not change the separate reviewer-quality verdict.

## Initial Failure And Repair Signature

The draft-MTP server log confirms the intended path was reached:

- `common_speculative_init_result: creating MTP draft context`
- `common_specu: adding speculative implementation 'draft-mtp'`
- `n_max=3, n_min=0, p_min=0.00, n_embd=6144, backend_sampling=1`
- `speculative decoding context initialized`
- request accepted with `task.n_tokens = 2931`

After that point the draft-MTP run produced no decode checkpoints, no prompt-eval final timing,
no response chunks, and no API usage object. The corresponding no-spec run on the same prompt
completed normally.

The repair build changed the shared DeepSeek32/GLM-DSA main graph to preserve full token rows
for `res->t_h_nextn` when target-side unmasked NEXTN extraction is enabled, then apply
`inp_out_ids` row selection before logits. That matches the established Qwen35/Gemma4/Cohere2MoE
NEXTN contract and prevents prompt batches with no output rows from exposing a zero/small hidden
state to the MTP draft context. The repaired matched run then completed normally:

- `completion_token_count=512`, completion floor `384` passed.
- Prompt eval `2931` tokens at `22.77 t/s`.
- Decode `512` tokens at `5.33 t/s`.
- Streaming chunks `515`, first chunk at `2026-07-19T19:52:56.509717+00:00`.
- Draft acceptance `376/403` tokens, alpha `0.933`; acceptance per position `(0.970, 0.926, 0.889)`.

## Source Hardening

The experimental-v7 source now includes a no-inference regression assertion for this contract.
`test-llama-archs` builds a GLM-DSA+single-NextN synthetic graph, enables target-side unmasked
NEXTN extraction, and asserts that `h_nextn` retains all token rows before `result_norm` applies
output-row selection. Focused validation passed:

- `cmake --build build-hip --target test-llama-archs -j 32`
- `test-llama-archs --arch glm-dsa -s 12345`
- `test-llama-archs --arch deepseek32 -s 12345`

## Decision

Native GLM-MTP is now functionally repaired on the matched long-context serving gate and shows a
measured decode-speed win on this long repeated-output shape. It is no longer correct to describe
B6 as a zero-chunk serving failure.

This does **not** admit GLM as the production patch reviewer. The decision-grade C-CRAB P-REV-1
reviewer gate remains failed, and the RM-2.fast slate did not produce a clean small-model
replacement. The repaired MTP path is therefore a v7/GLM acceleration result, while production
reviewer routing remains a separate quality/control-plane decision.
