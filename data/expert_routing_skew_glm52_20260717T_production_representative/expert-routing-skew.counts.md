# Expert Routing Counts

- Artifact: `/mnt/raid0/llm/tmp/expert-routing-skew-glm52-20260717T-production-representative/expert-routing-skew.imatrix.gguf`
- Tensor kind: `ffn_down_exps`
- Layers: `75`
- Experts: `256`
- Total selections: `19123200`
- Classification: `near_uniform_global` / `weak_layer_local_skew` (hypothesis_only)
- Caveat: Corpus representativeness and sample size are not decision-grade; repeat on workload prompts before offload/REAP gates.

## Aggregate

| Metric | Value |
|---|---:|
| Nonzero experts | 256 |
| Normalized entropy | 0.9987 |
| Gini | 0.0664 |
| top_1 share | 0.0054 |
| top_4 share | 0.0212 |
| top_8 share | 0.0412 |
| top_16 share | 0.0793 |
| top_32 share | 0.1519 |
| top_64 share | 0.2896 |
| top_128 share | 0.5458 |

## Layer Distribution

| Metric | Value |
|---|---:|
| top_32 share min | 0.2133 |
| top_32 share median | 0.3919 |
| top_32 share max | 0.4574 |
| nonzero experts min | 256 |
| nonzero experts median | 256 |
| nonzero experts max | 256 |

## Strongest Layer-Local Skew

| Layer | Nonzero | top_8 | top_16 | top_32 | Entropy |
|---:|---:|---:|---:|---:|---:|
| 33 | 256 | 0.2191 | 0.3377 | 0.4574 | 0.9120 |
| 20 | 256 | 0.2132 | 0.3235 | 0.4531 | 0.9081 |
| 43 | 256 | 0.2404 | 0.3289 | 0.4521 | 0.9037 |
| 39 | 256 | 0.2481 | 0.3305 | 0.4515 | 0.9011 |
| 41 | 256 | 0.2273 | 0.3258 | 0.4437 | 0.9111 |
| 32 | 256 | 0.1949 | 0.3061 | 0.4406 | 0.9159 |
| 50 | 256 | 0.2398 | 0.3234 | 0.4399 | 0.9049 |
| 17 | 256 | 0.1682 | 0.2840 | 0.4397 | 0.9221 |
| 22 | 256 | 0.2257 | 0.3159 | 0.4362 | 0.9162 |
| 57 | 256 | 0.2095 | 0.3083 | 0.4307 | 0.9146 |
| 19 | 256 | 0.2044 | 0.3030 | 0.4303 | 0.9165 |
| 42 | 256 | 0.2159 | 0.3053 | 0.4279 | 0.9147 |
