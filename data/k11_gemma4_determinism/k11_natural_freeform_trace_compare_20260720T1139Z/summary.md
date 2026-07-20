# K11 Natural Free-Form Trace Compare

K11 natural-prose nondeterminism reproduces on GPU MTP/no-spec/np1 but CPU no-spec np1 is deterministic; defect is GPU/backend-path dependent, not external-head MTP or multi-slot scheduling alone

| device | spec | slots | unique hashes | task pass | word counts | common prefix len | branch counts | mean decode t/s |
|---|---|---:|---:|---|---|---:|---|---:|
| MI210 | draft-mtp | 4 | 10 | False | `[156, 156, 155, 154, 156, 155, 155, 152, 158, 155]` | 7 | ` maintaining x9,  a x1` | 89.81 |
| MI210 | none | 4 | 9 | False | `[160, 154, 157, 157, 156, 153, 159, 153, 158, 156]` | 7 | ` maintaining x9,  a x1` | 69.75 |
| MI210 | none | 1 | 10 | False | `[152, 156, 160, 155, 152, 155, 157, 152, 153, 155]` | 16 | ` serving x5,  a x5` | 69.44 |
| CPU | none | 1 | 1 | False | `[156, 156, 156, 156, 156, 156, 156, 156, 156, 156]` | 182 | `none` | 21.12 |

GPU MTP/no-spec traces diverge early; CPU no-spec single-slot stays byte-identical. External-head MTP and multi-slot scheduling are insufficient explanations.
