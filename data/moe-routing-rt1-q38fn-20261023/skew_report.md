# RT-5 expert-skew report (Qwen3.8-Flash-Next UD-IQ4_XS, qwen4exp 512/top-10; PROXY for the RT-5 frontdoor target)

arch `qwen4exp`, experts [512], top-k [10], layers 48; gate = top-10% share >= 0.6.

| workload | phase | rows | top-10% in-sample | top-10% held-out | top-10% null | top-25% in-sample | top-25% null | layers>=gate | H/ln E | eff. experts | gini | never |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| code | prefill | 262144 | 0.5763 [0.333-0.707] | 0.5761 | 0.1040 | 0.8217 | 0.2544 | 18/48 | 0.8298 | 181.5 | 0.724 | 0.9 |
| code | decode | 2975 | 0.6041 [0.377-0.727] | 0.5934 | 0.1252 | 0.8367 | 0.2919 | 25/48 | 0.8159 | 166.8 | 0.745 | 55.1 |
| prose | prefill | 262144 | 0.4639 [0.316-0.583] | 0.4635 | 0.1040 | 0.7301 | 0.2544 | 0/48 | 0.8731 | 234.1 | 0.646 | 3.6 |
| prose | decode | 2373 | 0.4753 [0.324-0.653] | 0.4316 | 0.1281 | 0.7271 | 0.2969 | 3/48 | 0.8684 | 229.6 | 0.647 | 56.0 |
| reasoning | prefill | 262144 | 0.4775 [0.315-0.650] | 0.4772 | 0.1040 | 0.7512 | 0.2544 | 3/48 | 0.8644 | 224.1 | 0.666 | 1.7 |
| reasoning | decode | 2843 | 0.4810 [0.312-0.638] | 0.4635 | 0.1259 | 0.7569 | 0.2929 | 6/48 | 0.8609 | 220.3 | 0.671 | 69.4 |

## Cross-workload overlap (top-10% expert sets, mean over layers)

* prefill: within-domain even/odd ceiling Jaccard 0.9465770116090001, random-set chance 0.0535
  * code|prose: Jaccard 0.0238 (min 0.000, max 0.106), Spearman -0.1225 (min -0.341)
  * code|reasoning: Jaccard 0.0831 (min 0.010, max 0.224), Spearman 0.3088 (min 0.052)
  * prose|reasoning: Jaccard 0.0599 (min 0.000, max 0.238), Spearman 0.1002 (min -0.371)
* decode: within-domain even/odd ceiling Jaccard 0.6344611659831076, random-set chance 0.0535
  * code|prose: Jaccard 0.0392 (min 0.010, max 0.130), Spearman -0.0010 (min -0.175)
  * code|reasoning: Jaccard 0.1227 (min 0.020, max 0.316), Spearman 0.2772 (min 0.017)
  * prose|reasoning: Jaccard 0.0387 (min 0.000, max 0.195), Spearman 0.1769 (min -0.103)

## RT-5.1 load-balance statistics (mean over layers; null = uniform router with the same number of rows)

| workload | phase | MaxVio stream | null | MinVio stream | null | inactive stream | null | MaxVio per-512-chunk | null | inactive per-512-chunk | null |
|---|---|---|---|---|---|---|---|---|---|---|---|
| code | prefill | 19.173 (worst layer 30.78) | 0.043 | 0.999 | 0.041 | 0.002 | 0.000 | 28.429 | 1.109 | 0.475 | 0.000 |
| code | decode | 20.030 (worst layer 34.76) | 0.403 | 1.000 | 0.358 | 0.108 | 0.000 |  |  |  |  |
| prose | prefill | 15.913 (worst layer 43.71) | 0.043 | 1.000 | 0.041 | 0.007 | 0.000 | 30.920 | 1.109 | 0.421 | 0.000 |
| prose | decode | 16.333 (worst layer 35.49) | 0.474 | 1.000 | 0.420 | 0.109 | 0.000 |  |  |  |  |
| reasoning | prefill | 12.914 (worst layer 22.22) | 0.043 | 1.000 | 0.041 | 0.003 | 0.000 | 18.488 | 1.109 | 0.410 | 0.000 |
| reasoning | decode | 12.870 (worst layer 27.02) | 0.428 | 1.000 | 0.404 | 0.135 | 0.000 |  |  |  |  |

## RT-5 gate

Rule: top-10% share >= 0.6 on >= 2 workloads.

* prefill: workloads meeting the gate by layer-mean share = [] -> gate met: **False**
* decode: workloads meeting the gate by layer-mean share = ['code'] -> gate met: **False**
