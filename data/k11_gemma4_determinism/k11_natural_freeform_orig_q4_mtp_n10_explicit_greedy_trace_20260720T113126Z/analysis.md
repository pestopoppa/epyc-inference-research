# K11 Natural Free-Form Explicit-Greedy Token Trace

Artifact: `data/k11_gemma4_determinism/k11_natural_freeform_orig_q4_mtp_n10_explicit_greedy_trace_20260720T113126Z`

## Result

- Spec type: `draft-mtp`
- Slots: `4`
- Runs OK: `10/10`
- Unique output hashes: `10`
- Finish reasons: `{'stop': 10}`
- Word counts: `[156, 156, 155, 154, 156, 155, 155, 152, 158, 155]`
- Mean decode: `89.808 t/s`
- Mean draft acceptance: `0.4882`

## First Divergence

Common prefix length: `7` tokens. Prefix: `deterministic cleanup is a critical component of`

| token id | token | count |
|---:|---|---:|
| 16977 | ` maintaining` | 9 |
| 496 | ` a` | 1 |

Returned probabilities are post-sampling view values and often `1.0` at the divergence point, so this trace localizes the branch but does not expose the pre-sampling margin.

## Interpretation

MTP diverges; paired no-spec artifacts reproduce the defect, so MTP is not required
