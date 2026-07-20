# K11 Natural Free-Form Explicit-Greedy Token Trace

Artifact: `data/k11_gemma4_determinism/k11_natural_freeform_orig_q4_nospec_explicit_greedy_trace_20260720T113425Z`

## Result

- Spec type: `none`
- Slots: `4`
- Runs OK: `10/10`
- Unique output hashes: `9`
- Finish reasons: `{'stop': 10}`
- Word counts: `[160, 154, 157, 157, 156, 153, 159, 153, 158, 156]`
- Mean decode: `69.748 t/s`
- Mean draft acceptance: `0.0000`

## First Divergence

Common prefix length: `7` tokens. Prefix: `deterministic cleanup is a critical component of`

| token id | token | count |
|---:|---|---:|
| 16977 | ` maintaining` | 9 |
| 496 | ` a` | 1 |

Returned probabilities are post-sampling view values and often `1.0` at the divergence point, so this trace localizes the branch but does not expose the pre-sampling margin.

## Interpretation

no-spec also diverges, so target/no-spec serving path is implicated; external-head MTP is not required
