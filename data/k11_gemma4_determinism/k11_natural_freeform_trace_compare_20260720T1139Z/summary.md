# K11 Natural Free-Form Trace Compare

K11 natural-prose nondeterminism reproduces in MTP, no-spec np4, and no-spec np1; external-head MTP and multi-slot scheduling are not required

| spec | slots | unique hashes | word counts | common prefix len | branch counts | mean decode t/s |
|---|---:|---:|---|---:|---|---:|
| draft-mtp | 4 | 10 | `[156, 156, 155, 154, 156, 155, 155, 152, 158, 155]` | 7 | ` maintaining x9,  a x1` | 89.81 |
| none | 4 | 9 | `[160, 154, 157, 157, 156, 153, 159, 153, 158, 156]` | 7 | ` maintaining x9,  a x1` | 69.75 |
| none | 1 | 10 | `[152, 156, 160, 155, 152, 155, 157, 152, 153, 155]` | 16 | ` serving x5,  a x5` | 69.44 |

All three traces branch after the same 7-token prefix: `deterministic cleanup is a critical component of`. This makes late stop handling, external-head MTP, and multi-slot scheduling insufficient explanations by themselves.
