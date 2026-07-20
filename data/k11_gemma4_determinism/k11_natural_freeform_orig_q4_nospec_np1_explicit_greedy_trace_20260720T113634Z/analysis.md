# K11 Natural Free-Form Explicit-Greedy Token Trace

Artifact: `data/k11_gemma4_determinism/k11_natural_freeform_orig_q4_nospec_np1_explicit_greedy_trace_20260720T113634Z`

## Result

- Spec type: `none`
- Slots: `1`
- Runs OK: `10/10`
- Unique output hashes: `10`
- Finish reasons: `{'stop': 10}`
- Word counts: `[152, 156, 160, 155, 152, 155, 157, 152, 153, 155]`
- Mean decode: `69.435 t/s`
- Mean draft acceptance: `0.0000`

## First Divergence

Common prefix length: `16` tokens. Prefix: `deterministic cleanup is a critical component of maintaining a stable and efficient inference service. when`

| token id | token | count |
|---:|---|---:|
| 12284 | ` serving` | 5 |
| 496 | ` a` | 5 |

Returned probabilities are post-sampling view values and often `1.0` at the divergence point, so this trace localizes the branch but does not expose the pre-sampling margin.

## Interpretation

single-slot no-spec also diverges, so multi-slot scheduling is not required to reproduce the defect
