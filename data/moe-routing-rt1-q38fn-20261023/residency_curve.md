# RT-7 residency curve (Qwen3.8-Flash-Next UD-IQ4_XS qwen4exp 512/top-10; PROXY for DS41)

arch `qwen4exp`, 512 experts, top-k [10], 48 layers. Static residency learned on A (prefill), greedy cross-layer allocation, scored on B. k rescaled to the same fraction of the pool as 384-expert DS41: [52, 71, 84].

## B = prefill

| B | learned on | h(39) | h(53) | h(63) | h(52) | h(71) | h(84) | h(128) | h(256) |
|---|---|---|---|---|---|---|---|---|---|
| code | code_even->code_odd(prefill) | 0.497 | 0.583 | 0.632 | 0.577 | 0.667 | 0.715 | 0.824 | 0.954 |
| code | prose | 0.044 | 0.060 | 0.070 | 0.059 | 0.077 | 0.092 | 0.140 | 0.326 |
| code | reasoning | 0.110 | 0.151 | 0.183 | 0.148 | 0.203 | 0.239 | 0.376 | 0.750 |
| code | *oracle in-sample (bound)* | 0.498 | 0.584 | 0.633 | 0.578 | 0.668 | 0.716 | 0.824 | 0.954 |
| code | *chance k/E* | 0.076 | 0.104 | 0.123 | 0.102 | 0.139 | 0.164 | 0.250 | 0.500 |
| prose | code | 0.033 | 0.049 | 0.062 | 0.048 | 0.072 | 0.091 | 0.163 | 0.402 |
| prose | prose_even->prose_odd(prefill) | 0.396 | 0.469 | 0.515 | 0.464 | 0.548 | 0.598 | 0.732 | 0.932 |
| prose | reasoning | 0.063 | 0.094 | 0.114 | 0.091 | 0.131 | 0.161 | 0.257 | 0.509 |
| prose | *oracle in-sample (bound)* | 0.397 | 0.470 | 0.516 | 0.465 | 0.549 | 0.599 | 0.733 | 0.932 |
| prose | *chance k/E* | 0.076 | 0.104 | 0.123 | 0.102 | 0.139 | 0.164 | 0.250 | 0.500 |
| reasoning | code | 0.130 | 0.162 | 0.183 | 0.159 | 0.201 | 0.230 | 0.320 | 0.587 |
| reasoning | prose | 0.066 | 0.100 | 0.125 | 0.097 | 0.145 | 0.175 | 0.272 | 0.489 |
| reasoning | reasoning_even->reasoning_odd(prefill) | 0.409 | 0.485 | 0.533 | 0.480 | 0.568 | 0.618 | 0.754 | 0.948 |
| reasoning | *oracle in-sample (bound)* | 0.408 | 0.485 | 0.532 | 0.480 | 0.567 | 0.618 | 0.754 | 0.949 |
| reasoning | *chance k/E* | 0.076 | 0.104 | 0.123 | 0.102 | 0.139 | 0.164 | 0.250 | 0.500 |

## B = decode

| B | learned on | h(39) | h(53) | h(63) | h(52) | h(71) | h(84) | h(128) | h(256) |
|---|---|---|---|---|---|---|---|---|---|
| code | code(same-domain prefill->decode) | 0.492 | 0.568 | 0.613 | 0.563 | 0.646 | 0.691 | 0.795 | 0.940 |
| code | prose | 0.053 | 0.071 | 0.081 | 0.070 | 0.089 | 0.103 | 0.156 | 0.340 |
| code | reasoning | 0.115 | 0.160 | 0.193 | 0.156 | 0.216 | 0.254 | 0.395 | 0.753 |
| code | *oracle in-sample (bound)* | 0.529 | 0.612 | 0.660 | 0.606 | 0.693 | 0.738 | 0.840 | 0.961 |
| code | *chance k/E* | 0.076 | 0.104 | 0.123 | 0.102 | 0.139 | 0.164 | 0.250 | 0.500 |
| prose | code | 0.039 | 0.057 | 0.073 | 0.056 | 0.085 | 0.106 | 0.187 | 0.448 |
| prose | prose(same-domain prefill->decode) | 0.332 | 0.395 | 0.434 | 0.391 | 0.464 | 0.508 | 0.636 | 0.857 |
| prose | reasoning | 0.063 | 0.093 | 0.113 | 0.090 | 0.129 | 0.157 | 0.250 | 0.507 |
| prose | *oracle in-sample (bound)* | 0.412 | 0.481 | 0.525 | 0.477 | 0.556 | 0.603 | 0.730 | 0.931 |
| prose | *chance k/E* | 0.076 | 0.104 | 0.123 | 0.102 | 0.139 | 0.164 | 0.250 | 0.500 |
| reasoning | code | 0.132 | 0.163 | 0.185 | 0.161 | 0.204 | 0.234 | 0.327 | 0.584 |
| reasoning | prose | 0.060 | 0.089 | 0.112 | 0.087 | 0.130 | 0.159 | 0.255 | 0.483 |
| reasoning | reasoning(same-domain prefill->decode) | 0.353 | 0.428 | 0.474 | 0.423 | 0.509 | 0.560 | 0.698 | 0.923 |
| reasoning | *oracle in-sample (bound)* | 0.410 | 0.488 | 0.536 | 0.483 | 0.572 | 0.624 | 0.760 | 0.952 |
| reasoning | *chance k/E* | 0.076 | 0.104 | 0.123 | 0.102 | 0.139 | 0.164 | 0.250 | 0.500 |

## Stationarity (top-n expert set, consecutive 1024-row blocks)

| workload | phase | n | blocks | consecutive Jaccard | random-set | half-split ceiling | next-block coverage | oracle coverage |
|---|---|---|---|---|---|---|---|---|
| code | prefill | 53 | 256 | 0.663 | 0.055 | 0.828 | 0.677 | 0.730 |
| code | prefill | 71 | 256 | 0.664 | 0.075 | 0.810 | 0.745 | 0.799 |
| code | decode | 53 | 2 | 0.375 | 0.055 | 0.808 | 0.474 | 0.632 |
| code | decode | 71 | 2 | 0.403 | 0.075 | 0.794 | 0.549 | 0.710 |
| prose | prefill | 53 | 256 | 0.539 | 0.055 | 0.734 | 0.556 | 0.638 |
| prose | prefill | 71 | 256 | 0.561 | 0.075 | 0.737 | 0.627 | 0.711 |
| prose | decode | 53 | 2 | 0.407 | 0.055 | 0.714 | 0.395 | 0.512 |
| prose | decode | 71 | 2 | 0.451 | 0.075 | 0.712 | 0.466 | 0.588 |
| reasoning | prefill | 53 | 256 | 0.524 | 0.055 | 0.739 | 0.487 | 0.566 |
| reasoning | prefill | 71 | 256 | 0.553 | 0.075 | 0.757 | 0.569 | 0.651 |
| reasoning | decode | 53 | 2 | 0.369 | 0.055 | 0.718 | 0.380 | 0.502 |
| reasoning | decode | 71 | 2 | 0.409 | 0.075 | 0.728 | 0.457 | 0.587 |

## RT-7 gate (applied literally; PROXY model)

* k=53 prefill: same-domain held-out >= 0.30 on ['code', 'prose', 'reasoning']; worst-case-source >= 0.30 on []; mean-over-sources >= 0.30 on []; worst-case < 0.23 on ['code', 'prose', 'reasoning']
* k=53 decode: same-domain held-out >= 0.30 on ['code', 'prose', 'reasoning']; worst-case-source >= 0.30 on []; mean-over-sources >= 0.30 on []; worst-case < 0.23 on ['code', 'prose', 'reasoning']
* k=71 prefill: same-domain held-out >= 0.30 on ['code', 'prose', 'reasoning']; worst-case-source >= 0.30 on []; mean-over-sources >= 0.30 on []; worst-case < 0.23 on ['code', 'prose', 'reasoning']
* k=71 decode: same-domain held-out >= 0.30 on ['code', 'prose', 'reasoning']; worst-case-source >= 0.30 on []; mean-over-sources >= 0.30 on []; worst-case < 0.23 on ['code', 'prose', 'reasoning']
