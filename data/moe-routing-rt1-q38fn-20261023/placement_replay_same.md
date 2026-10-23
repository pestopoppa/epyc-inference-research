# RT-6 offline expert-placement replay (Qwen3.8-Flash-Next UD-IQ4_XS qwen4exp 512/top-10; PROXY for DS41 + frontdoor) (calibration: same-domain)

Per-step max-bin-load / mean-bin-load (lower is better, 1.0 perfect), mean over steps and layers, [95% CI over windows]. Trigger: 1.65 / best >= 1.10 on >= 2 workloads at T in {1,3}. Per-step arms (oracle) use <= 1000 evenly spaced steps.

## B4/decode/T1

| arm | code | prose | reasoning |
|---|---|---|---|
| contiguous | 1.675 [1.669-1.682] | 1.669 [1.663-1.675] | 1.672 [1.666-1.678] |
| deepinfer_R0 | 1.530 [1.523-1.539] | 1.586 [1.579-1.594] | 1.555 [1.548-1.563] |
| deepinfer_R32 | 1.496 [1.486-1.508] (+32 copies/layer) | 1.553 [1.543-1.565] (+32 copies/layer) | 1.543 [1.534-1.554] (+32 copies/layer) |
| deepinfer_R32@random | 1.547 [1.541-1.554] (+32 copies/layer) | 1.585 [1.579-1.592] (+32 copies/layer) | 1.563 [1.556-1.571] (+32 copies/layer) |
| eplb_R0 | 1.617 [1.608-1.625] | 1.648 [1.641-1.655] | 1.648 [1.644-1.652] |
| eplb_R32 | 1.368 [1.351-1.389] (+32 copies/layer) | 1.451 [1.432-1.470] (+32 copies/layer) | 1.433 [1.416-1.456] (+32 copies/layer) |
| eplb_R32@random | 1.659 [1.657-1.661] (+32 copies/layer) | 1.672 [1.668-1.676] (+32 copies/layer) | 1.661 [1.658-1.664] (+32 copies/layer) |
| integrality_floor | 1.200 | 1.200 | 1.200 |
| lp_floor@deepinfer_R32 | 1.496 [1.486-1.508] | 1.553 [1.543-1.565] | 1.543 [1.534-1.554] |
| lp_floor@eplb_R32 | 1.296 [1.270-1.324] | 1.415 [1.392-1.438] | 1.389 [1.366-1.417] |
| moonep_floor | 1.200 (1.3 copies/step) | 1.200 (1.3 copies/step) | 1.200 (1.3 copies/step) |
| null_random | 1.672 [1.672-1.673] | 1.673 [1.672-1.674] | 1.673 [1.672-1.674] |
| null_uniform_mc | 1.678 | 1.678 | 1.678 |
| oracle_R32 | 1.200 [1.200-1.200] (1.3 copies/step) | 1.200 [1.200-1.200] (1.3 copies/step) | 1.200 [1.200-1.200] (1.3 copies/step) |
| *trigger ratio* | realizable 1.65/1.530 = **1.078** (deepinfer_R0); idealized-dispatch 1.206 (eplb_R32); bound 1.200 (oracle_R32) -> 1.375 | realizable 1.65/1.585 = **1.041** (deepinfer_R32@random); idealized-dispatch 1.137 (eplb_R32); bound 1.200 (oracle_R32) -> 1.375 | realizable 1.65/1.555 = **1.061** (deepinfer_R0); idealized-dispatch 1.151 (eplb_R32); bound 1.200 (oracle_R32) -> 1.375 |

## B4/decode/T3

| arm | code | prose | reasoning |
|---|---|---|---|
| contiguous | 1.495 [1.487-1.502] | 1.500 [1.491-1.508] | 1.488 [1.481-1.496] |
| deepinfer_R0 | 1.349 [1.344-1.354] | 1.416 [1.407-1.424] | 1.378 [1.373-1.383] |
| deepinfer_R32 | 1.149 [1.139-1.161] (+32 copies/layer) | 1.243 [1.230-1.257] (+32 copies/layer) | 1.208 [1.193-1.226] (+32 copies/layer) |
| deepinfer_R32@random | 1.370 [1.367-1.374] (+32 copies/layer) | 1.407 [1.401-1.413] (+32 copies/layer) | 1.387 [1.383-1.392] (+32 copies/layer) |
| eplb_R0 | 1.426 [1.419-1.432] | 1.479 [1.470-1.487] | 1.458 [1.452-1.464] |
| eplb_R32 | 1.175 [1.160-1.190] (+32 copies/layer) | 1.278 [1.259-1.296] (+32 copies/layer) | 1.244 [1.227-1.266] (+32 copies/layer) |
| eplb_R32@random | 1.433 [1.427-1.438] (+32 copies/layer) | 1.475 [1.467-1.483] (+32 copies/layer) | 1.448 [1.445-1.451] (+32 copies/layer) |
| integrality_floor | 1.067 | 1.067 | 1.067 |
| lp_floor@deepinfer_R32 | 1.105 [1.092-1.121] | 1.219 [1.204-1.237] | 1.178 [1.160-1.200] |
| lp_floor@eplb_R32 | 1.135 [1.117-1.154] | 1.256 [1.234-1.277] | 1.218 [1.198-1.243] |
| moonep_floor | 1.067 (2.3 copies/step) | 1.067 (2.2 copies/step) | 1.067 (2.2 copies/step) |
| null_random | 1.487 [1.477-1.495] | 1.506 [1.499-1.511] | 1.487 [1.483-1.490] |
| null_uniform_mc | 1.384 | 1.384 | 1.384 |
| oracle_R32 | 1.067 [1.067-1.067] (2.2 copies/step) | 1.067 [1.067-1.067] (2.1 copies/step) | 1.067 [1.067-1.067] (2.2 copies/step) |
| *trigger ratio* | realizable 1.65/1.349 = **1.223** (deepinfer_R0); idealized-dispatch 1.436 (deepinfer_R32); bound 1.067 (oracle_R32) -> 1.547 | realizable 1.65/1.407 = **1.173** (deepinfer_R32@random); idealized-dispatch 1.328 (deepinfer_R32); bound 1.067 (oracle_R32) -> 1.547 | realizable 1.65/1.378 = **1.197** (deepinfer_R0); idealized-dispatch 1.366 (deepinfer_R32); bound 1.067 (oracle_R32) -> 1.547 |

## B4/prefill/T512

| arm | code | prose | reasoning |
|---|---|---|---|
| contiguous | 1.283 [1.275-1.291] | 1.271 [1.265-1.277] | 1.196 [1.192-1.200] |
| deepinfer_R0 | 1.071 [1.067-1.074] | 1.093 [1.090-1.096] | 1.066 [1.063-1.068] |
| deepinfer_R32 | 1.008 [1.007-1.009] (+32 copies/layer) | 1.013 [1.011-1.015] (+32 copies/layer) | 1.005 [1.005-1.007] (+32 copies/layer) |
| deepinfer_R32@random | 1.060 [1.058-1.063] (+32 copies/layer) | 1.076 [1.073-1.078] (+32 copies/layer) | 1.055 [1.053-1.058] (+32 copies/layer) |
| eplb_R0 | 1.170 [1.162-1.179] | 1.200 [1.195-1.205] | 1.122 [1.117-1.128] |
| eplb_R32 | 1.026 [1.018-1.036] (+32 copies/layer) | 1.031 [1.026-1.038] (+32 copies/layer) | 1.011 [1.009-1.014] (+32 copies/layer) |
| eplb_R32@random | 1.143 [1.134-1.153] (+32 copies/layer) | 1.160 [1.154-1.166] (+32 copies/layer) | 1.100 [1.096-1.105] (+32 copies/layer) |
| integrality_floor | 1.000 | 1.000 | 1.000 |
| lp_floor@deepinfer_R32 | 1.001 [1.001-1.002] | 1.005 [1.003-1.007] | 1.001 [1.000-1.002] |
| lp_floor@eplb_R32 | 1.019 [1.010-1.029] | 1.025 [1.019-1.032] | 1.005 [1.003-1.008] |
| moonep_floor | 1.000 (4.0 copies/step) | 1.000 (3.8 copies/step) | 1.000 (4.1 copies/step) |
| null_random | 1.276 [1.269-1.284] | 1.265 [1.261-1.270] | 1.196 [1.194-1.199] |
| null_uniform_mc | 1.029 | 1.029 | 1.029 |
| oracle_R32 | 1.000 [1.000-1.000] (4.3 copies/step) | 1.000 [1.000-1.000] (4.0 copies/step) | 1.000 [1.000-1.000] (4.5 copies/step) |
| *trigger ratio* | realizable 1.65/1.060 = **1.556** (deepinfer_R32@random); idealized-dispatch 1.636 (deepinfer_R32); bound 1.000 (oracle_R32) -> 1.650 | realizable 1.65/1.076 = **1.534** (deepinfer_R32@random); idealized-dispatch 1.629 (deepinfer_R32); bound 1.000 (oracle_R32) -> 1.650 | realizable 1.65/1.055 = **1.563** (deepinfer_R32@random); idealized-dispatch 1.641 (deepinfer_R32); bound 1.000 (oracle_R32) -> 1.650 |

## B12/decode/T1

| arm | code | prose | reasoning |
|---|---|---|---|
| contiguous | 3.080 [3.070-3.089] | 3.042 [3.030-3.056] | 3.040 [3.029-3.049] |
| deepinfer_R0 | 2.631 [2.609-2.659] | 2.786 [2.760-2.814] | 2.700 [2.681-2.722] |
| deepinfer_R32 | 2.596 [2.570-2.628] (+32 copies/layer) | 2.740 [2.716-2.765] (+32 copies/layer) | 2.668 [2.642-2.698] (+32 copies/layer) |
| deepinfer_R32@random | 2.689 [2.670-2.711] (+32 copies/layer) | 2.797 [2.780-2.815] (+32 copies/layer) | 2.714 [2.694-2.737] (+32 copies/layer) |
| eplb_R0 | 2.904 [2.886-2.922] | 2.969 [2.957-2.983] | 2.972 [2.958-2.987] |
| eplb_R32 | 2.523 [2.488-2.562] (+32 copies/layer) | 2.693 [2.665-2.723] (+32 copies/layer) | 2.674 [2.643-2.713] (+32 copies/layer) |
| eplb_R32@random | 2.974 [2.967-2.981] (+32 copies/layer) | 3.016 [3.007-3.025] (+32 copies/layer) | 3.009 [3.002-3.016] (+32 copies/layer) |
| integrality_floor | 1.200 | 1.200 | 1.200 |
| lp_floor@deepinfer_R32 | 2.596 [2.570-2.628] | 2.740 [2.716-2.765] | 2.668 [2.642-2.698] |
| lp_floor@eplb_R32 | 2.344 [2.287-2.407] | 2.612 [2.571-2.654] | 2.571 [2.527-2.626] |
| moonep_floor | 1.200 (3.0 copies/step) | 1.200 (3.0 copies/step) | 1.200 (3.0 copies/step) |
| null_random | 3.052 [3.049-3.055] | 3.043 [3.041-3.045] | 3.051 [3.049-3.052] |
| null_uniform_mc | 3.077 | 3.077 | 3.077 |
| oracle_R32 | 1.200 [1.200-1.200] (3.0 copies/step) | 1.200 [1.200-1.200] (3.0 copies/step) | 1.200 [1.200-1.200] (3.0 copies/step) |
| *trigger ratio* | realizable 1.65/2.631 = **0.627** (deepinfer_R0); idealized-dispatch 0.654 (eplb_R32); bound 1.200 (oracle_R32) -> 1.375 | realizable 1.65/2.786 = **0.592** (deepinfer_R0); idealized-dispatch 0.613 (eplb_R32); bound 1.200 (oracle_R32) -> 1.375 | realizable 1.65/2.700 = **0.611** (deepinfer_R0); idealized-dispatch 0.619 (deepinfer_R32); bound 1.200 (oracle_R32) -> 1.375 |

## B12/decode/T3

| arm | code | prose | reasoning |
|---|---|---|---|
| contiguous | 2.528 [2.497-2.554] | 2.533 [2.508-2.554] | 2.474 [2.456-2.491] |
| deepinfer_R0 | 2.077 [2.058-2.095] | 2.275 [2.251-2.301] | 2.165 [2.148-2.184] |
| deepinfer_R32 | 1.759 [1.731-1.789] (+32 copies/layer) | 2.016 [1.981-2.050] (+32 copies/layer) | 1.910 [1.879-1.945] (+32 copies/layer) |
| deepinfer_R32@random | 2.114 [2.101-2.127] (+32 copies/layer) | 2.259 [2.237-2.281] (+32 copies/layer) | 2.170 [2.157-2.184] (+32 copies/layer) |
| eplb_R0 | 2.319 [2.293-2.342] | 2.458 [2.436-2.477] | 2.394 [2.378-2.413] |
| eplb_R32 | 1.865 [1.830-1.900] (+32 copies/layer) | 2.127 [2.087-2.164] (+32 copies/layer) | 2.049 [2.011-2.091] (+32 copies/layer) |
| eplb_R32@random | 2.289 [2.271-2.306] (+32 copies/layer) | 2.430 [2.409-2.447] (+32 copies/layer) | 2.365 [2.353-2.377] (+32 copies/layer) |
| integrality_floor | 1.200 | 1.200 | 1.200 |
| lp_floor@deepinfer_R32 | 1.682 [1.646-1.721] | 1.984 [1.945-2.023] | 1.869 [1.833-1.910] |
| lp_floor@eplb_R32 | 1.801 [1.758-1.844] | 2.097 [2.053-2.139] | 2.012 [1.969-2.060] |
| moonep_floor | 1.200 (4.3 copies/step) | 1.200 (4.2 copies/step) | 1.200 (4.2 copies/step) |
| null_random | 2.494 [2.463-2.523] | 2.546 [2.523-2.564] | 2.479 [2.469-2.488] |
| null_uniform_mc | 2.138 | 2.138 | 2.138 |
| oracle_R32 | 1.200 [1.200-1.200] (4.3 copies/step) | 1.200 [1.200-1.200] (4.2 copies/step) | 1.200 [1.200-1.200] (4.2 copies/step) |
| *trigger ratio* | realizable 1.65/2.077 = **0.794** (deepinfer_R0); idealized-dispatch 0.938 (deepinfer_R32); bound 1.200 (oracle_R32) -> 1.375 | realizable 1.65/2.259 = **0.730** (deepinfer_R32@random); idealized-dispatch 0.818 (deepinfer_R32); bound 1.200 (oracle_R32) -> 1.375 | realizable 1.65/2.165 = **0.762** (deepinfer_R0); idealized-dispatch 0.864 (deepinfer_R32); bound 1.200 (oracle_R32) -> 1.375 |

## B12/prefill/T512

| arm | code | prose | reasoning |
|---|---|---|---|
| contiguous | 1.874 [1.858-1.895] | 1.819 [1.803-1.836] | 1.573 [1.563-1.582] |
| deepinfer_R0 | 1.226 [1.216-1.235] | 1.317 [1.307-1.326] | 1.212 [1.204-1.220] |
| deepinfer_R32 | 1.092 [1.086-1.098] (+32 copies/layer) | 1.145 [1.137-1.156] (+32 copies/layer) | 1.075 [1.070-1.081] (+32 copies/layer) |
| deepinfer_R32@random | 1.194 [1.186-1.202] (+32 copies/layer) | 1.255 [1.245-1.265] (+32 copies/layer) | 1.180 [1.173-1.188] (+32 copies/layer) |
| eplb_R0 | 1.525 [1.495-1.562] | 1.617 [1.601-1.634] | 1.353 [1.337-1.370] |
| eplb_R32 | 1.225 [1.195-1.262] (+32 copies/layer) | 1.289 [1.268-1.310] (+32 copies/layer) | 1.135 [1.120-1.151] (+32 copies/layer) |
| eplb_R32@random | 1.445 [1.419-1.477] (+32 copies/layer) | 1.509 [1.490-1.526] (+32 copies/layer) | 1.311 [1.294-1.327] (+32 copies/layer) |
| integrality_floor | 1.001 | 1.001 | 1.001 |
| lp_floor@deepinfer_R32 | 1.039 [1.031-1.048] | 1.113 [1.103-1.125] | 1.036 [1.030-1.044] |
| lp_floor@eplb_R32 | 1.175 [1.141-1.215] | 1.260 [1.237-1.284] | 1.097 [1.080-1.115] |
| moonep_floor | 1.001 (11.7 copies/step) | 1.001 (11.3 copies/step) | 1.001 (11.7 copies/step) |
| null_random | 1.842 [1.824-1.865] | 1.824 [1.809-1.839] | 1.597 [1.589-1.605] |
| null_uniform_mc | 1.080 | 1.080 | 1.080 |
| oracle_R32 | 1.001 [1.001-1.001] (13.0 copies/step) | 1.001 [1.001-1.001] (12.5 copies/step) | 1.001 [1.001-1.001] (12.9 copies/step) |
| *trigger ratio* | realizable 1.65/1.194 = **1.382** (deepinfer_R32@random); idealized-dispatch 1.512 (deepinfer_R32); bound 1.001 (oracle_R32) -> 1.649 | realizable 1.65/1.255 = **1.315** (deepinfer_R32@random); idealized-dispatch 1.440 (deepinfer_R32); bound 1.001 (oracle_R32) -> 1.649 | realizable 1.65/1.180 = **1.399** (deepinfer_R32@random); idealized-dispatch 1.535 (deepinfer_R32); bound 1.001 (oracle_R32) -> 1.649 |

## DI-081 trigger

* B4/decode/T1: realizable static arms meet 1.10 on []; with idealized least-loaded dispatch on ['code', 'prose', 'reasoning']; bounds meet 1.10 on ['code', 'prose', 'reasoning']
* B4/decode/T3: realizable static arms meet 1.10 on ['code', 'prose', 'reasoning']; with idealized least-loaded dispatch on ['code', 'prose', 'reasoning']; bounds meet 1.10 on ['code', 'prose', 'reasoning']
* B12/decode/T1: realizable static arms meet 1.10 on []; with idealized least-loaded dispatch on []; bounds meet 1.10 on ['code', 'prose', 'reasoning']
* B12/decode/T3: realizable static arms meet 1.10 on []; with idealized least-loaded dispatch on []; bounds meet 1.10 on ['code', 'prose', 'reasoning']
