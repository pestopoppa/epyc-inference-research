# Expert locality (Qwen3.8-Flash-Next UD-IQ4_XS, qwen4exp 512/top-10)

## Adjacent-token reuse IR_t (mean over layers; ratio to chance K/E)

| workload | phase | IR mean | chance | ratio | layer min..max IR |
|---|---|---|---|---|---|
| code | prefill | 0.3585 | 0.0195 | 18.4x | 0.085..0.471 |
| code | decode | 0.3402 | 0.0195 | 17.4x | 0.064..0.472 |
| prose | prefill | 0.3885 | 0.0195 | 19.9x | 0.089..0.593 |
| prose | decode | 0.3861 | 0.0195 | 19.8x | 0.082..0.548 |
| reasoning | prefill | 0.3286 | 0.0195 | 16.8x | 0.070..0.507 |
| reasoning | decode | 0.3274 | 0.0195 | 16.8x | 0.060..0.530 |

## LRU hit rate vs cache size (experts per layer; mean over the sampled layers; within-window)

| workload | phase | k=8 | k=16 | k=32 | k=64 | k=128 | k=256 | compulsory miss | median dist |
|---|---|---|---|---|---|---|---|---|---|
| code | prefill | 0.310 | 0.400 | 0.499 | 0.681 | 0.878 | 0.944 | 0.052 | 30 |
| code | decode | 0.311 | 0.397 | 0.496 | 0.672 | 0.871 | 0.967 | 0.014 | 34 |
| prose | prefill | 0.328 | 0.413 | 0.519 | 0.646 | 0.803 | 0.922 | 0.062 | 28 |
| prose | decode | 0.323 | 0.408 | 0.510 | 0.644 | 0.810 | 0.951 | 0.016 | 33 |
| reasoning | prefill | 0.266 | 0.355 | 0.471 | 0.622 | 0.801 | 0.922 | 0.061 | 36 |
| reasoning | decode | 0.270 | 0.359 | 0.475 | 0.635 | 0.827 | 0.962 | 0.012 | 38 |

## Co-activation (mean over layers)

| workload | excess fraction | noise floor | top-1% pair mass | noise floor | max lift (exp>=5) | frac pairs lift>=2 |
|---|---|---|---|---|---|---|
| code | 0.4787 | 0.0228 | 0.5482 | 0.3657 | 112.2 | 0.1430 |
| prose | 0.4724 | 0.0263 | 0.4198 | 0.2495 | 121.4 | 0.1251 |
| reasoning | 0.5075 | 0.0249 | 0.4534 | 0.2590 | 121.0 | 0.1311 |
