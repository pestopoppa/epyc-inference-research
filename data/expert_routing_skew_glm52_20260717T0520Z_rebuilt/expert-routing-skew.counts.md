# Expert Routing Counts

- Artifact: `/mnt/raid0/llm/tmp/expert-routing-skew-glm52-20260717T0520Z-rebuilt/expert-routing-skew.imatrix.gguf`
- Tensor kind: `ffn_down_exps`
- Layers: `75`
- Experts: `256`
- Total selections: `691200`
- Classification: `near_uniform_global` / `moderate_layer_local_skew` (hypothesis_only)
- Caveat: Corpus representativeness and sample size are not decision-grade; repeat on workload prompts before offload/REAP gates.

## Aggregate

| Metric | Value |
|---|---:|
| Nonzero experts | 256 |
| Normalized entropy | 0.9962 |
| Gini | 0.1159 |
| top_1 share | 0.0064 |
| top_4 share | 0.0248 |
| top_8 share | 0.0480 |
| top_16 share | 0.0909 |
| top_32 share | 0.1710 |
| top_64 share | 0.3188 |
| top_128 share | 0.5821 |

## Layer Distribution

| Metric | Value |
|---|---:|
| top_32 share min | 0.3587 |
| top_32 share median | 0.5562 |
| top_32 share max | 0.7048 |
| nonzero experts min | 203 |
| nonzero experts median | 236 |
| nonzero experts max | 250 |

## Strongest Layer-Local Skew

| Layer | Nonzero | top_8 | top_16 | top_32 | Entropy |
|---:|---:|---:|---:|---:|---:|
| 32 | 223 | 0.3687 | 0.5640 | 0.7048 | 0.7730 |
| 33 | 228 | 0.3178 | 0.4745 | 0.6789 | 0.7891 |
| 31 | 213 | 0.3150 | 0.4874 | 0.6694 | 0.7890 |
| 19 | 217 | 0.3580 | 0.5058 | 0.6687 | 0.7846 |
| 38 | 222 | 0.2899 | 0.4586 | 0.6679 | 0.7977 |
| 17 | 219 | 0.3334 | 0.4736 | 0.6635 | 0.7924 |
| 15 | 203 | 0.3262 | 0.4727 | 0.6611 | 0.7863 |
| 42 | 235 | 0.3373 | 0.4891 | 0.6516 | 0.8054 |
| 30 | 221 | 0.3015 | 0.4460 | 0.6374 | 0.8047 |
| 22 | 222 | 0.2522 | 0.4079 | 0.6261 | 0.8195 |
| 18 | 220 | 0.3054 | 0.4439 | 0.6216 | 0.8081 |
| 39 | 225 | 0.3035 | 0.4650 | 0.6175 | 0.8137 |
