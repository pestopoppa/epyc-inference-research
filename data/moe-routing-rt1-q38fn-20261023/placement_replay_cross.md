# RT-6 offline expert-placement replay (Qwen3.8-Flash-Next UD-IQ4_XS qwen4exp 512/top-10; PROXY for DS41 + frontdoor) (calibration: cross-domain)

Per-step max-bin-load / mean-bin-load (lower is better, 1.0 perfect), mean over steps and layers, [95% CI over windows]. Trigger: 1.65 / best >= 1.10 on >= 2 workloads at T in {1,3}. Per-step arms (oracle) use <= 1000 evenly spaced steps.

## B4/decode/T1

| arm | code | prose | reasoning |
|---|---|---|---|
| contiguous | 1.675 [1.669-1.682] | 1.669 [1.663-1.675] | 1.672 [1.666-1.678] |
| deepinfer_R0 | 1.631 [1.626-1.635] | 1.645 [1.641-1.650] | 1.629 [1.623-1.634] |
| deepinfer_R32 | 1.638 [1.634-1.642] (+32 copies/layer) | 1.643 [1.636-1.648] (+32 copies/layer) | 1.632 [1.626-1.637] (+32 copies/layer) |
| deepinfer_R32@random | 1.639 [1.635-1.644] (+32 copies/layer) | 1.643 [1.637-1.649] (+32 copies/layer) | 1.636 [1.630-1.640] (+32 copies/layer) |
| eplb_R0 | 1.679 [1.675-1.683] | 1.671 [1.667-1.675] | 1.663 [1.659-1.668] |
| eplb_R32 | 1.624 [1.619-1.629] (+32 copies/layer) | 1.648 [1.643-1.654] (+32 copies/layer) | 1.599 [1.584-1.614] (+32 copies/layer) |
| eplb_R32@random | 1.677 [1.671-1.682] (+32 copies/layer) | 1.671 [1.667-1.676] (+32 copies/layer) | 1.671 [1.668-1.674] (+32 copies/layer) |
| integrality_floor | 1.200 | 1.200 | 1.200 |
| lp_floor@deepinfer_R32 | 1.638 [1.634-1.642] | 1.643 [1.636-1.648] | 1.632 [1.626-1.637] |
| lp_floor@eplb_R32 | 1.618 [1.613-1.623] | 1.646 [1.641-1.651] | 1.590 [1.572-1.606] |
| moonep_floor | 1.200 (1.3 copies/step) | 1.200 (1.3 copies/step) | 1.200 (1.3 copies/step) |
| null_random | 1.672 [1.672-1.673] | 1.673 [1.672-1.674] | 1.673 [1.672-1.674] |
| null_uniform_mc | 1.678 | 1.678 | 1.678 |
| oracle_R32 | 1.200 [1.200-1.200] (1.3 copies/step) | 1.200 [1.200-1.200] (1.3 copies/step) | 1.200 [1.200-1.200] (1.3 copies/step) |
| *trigger ratio* | realizable 1.65/1.631 = **1.012** (deepinfer_R0); idealized-dispatch 1.016 (eplb_R32); bound 1.200 (oracle_R32) -> 1.375 | realizable 1.65/1.643 = **1.004** (deepinfer_R32@random); idealized-dispatch 1.005 (deepinfer_R32); bound 1.200 (oracle_R32) -> 1.375 | realizable 1.65/1.629 = **1.013** (deepinfer_R0); idealized-dispatch 1.032 (eplb_R32); bound 1.200 (oracle_R32) -> 1.375 |

## B4/decode/T3

| arm | code | prose | reasoning |
|---|---|---|---|
| contiguous | 1.495 [1.487-1.502] | 1.500 [1.491-1.508] | 1.488 [1.481-1.496] |
| deepinfer_R0 | 1.440 [1.434-1.446] | 1.488 [1.478-1.497] | 1.450 [1.444-1.456] |
| deepinfer_R32 | 1.377 [1.371-1.383] (+32 copies/layer) | 1.454 [1.441-1.466] (+32 copies/layer) | 1.380 [1.365-1.393] (+32 copies/layer) |
| deepinfer_R32@random | 1.429 [1.423-1.435] (+32 copies/layer) | 1.484 [1.475-1.493] (+32 copies/layer) | 1.446 [1.440-1.451] (+32 copies/layer) |
| eplb_R0 | 1.493 [1.483-1.502] | 1.505 [1.497-1.513] | 1.477 [1.472-1.483] |
| eplb_R32 | 1.443 [1.431-1.455] (+32 copies/layer) | 1.480 [1.470-1.489] (+32 copies/layer) | 1.414 [1.396-1.430] (+32 copies/layer) |
| eplb_R32@random | 1.492 [1.479-1.505] (+32 copies/layer) | 1.501 [1.493-1.510] (+32 copies/layer) | 1.482 [1.477-1.486] (+32 copies/layer) |
| integrality_floor | 1.067 | 1.067 | 1.067 |
| lp_floor@deepinfer_R32 | 1.373 [1.366-1.379] | 1.452 [1.439-1.465] | 1.374 [1.357-1.388] |
| lp_floor@eplb_R32 | 1.440 [1.428-1.452] | 1.479 [1.469-1.488] | 1.409 [1.390-1.426] |
| moonep_floor | 1.067 (2.3 copies/step) | 1.067 (2.2 copies/step) | 1.067 (2.2 copies/step) |
| null_random | 1.487 [1.477-1.495] | 1.506 [1.499-1.511] | 1.487 [1.483-1.490] |
| null_uniform_mc | 1.384 | 1.384 | 1.384 |
| oracle_R32 | 1.067 [1.067-1.067] (2.2 copies/step) | 1.067 [1.067-1.067] (2.1 copies/step) | 1.067 [1.067-1.067] (2.2 copies/step) |
| *trigger ratio* | realizable 1.65/1.429 = **1.154** (deepinfer_R32@random); idealized-dispatch 1.198 (deepinfer_R32); bound 1.067 (oracle_R32) -> 1.547 | realizable 1.65/1.484 = **1.112** (deepinfer_R32@random); idealized-dispatch 1.135 (deepinfer_R32); bound 1.067 (oracle_R32) -> 1.547 | realizable 1.65/1.446 = **1.141** (deepinfer_R32@random); idealized-dispatch 1.196 (deepinfer_R32); bound 1.067 (oracle_R32) -> 1.547 |

## B4/prefill/T512

| arm | code | prose | reasoning |
|---|---|---|---|
| contiguous | 1.283 [1.275-1.291] | 1.271 [1.265-1.277] | 1.196 [1.192-1.200] |
| deepinfer_R0 | 1.227 [1.217-1.239] | 1.265 [1.260-1.270] | 1.173 [1.170-1.176] |
| deepinfer_R32 | 1.185 [1.177-1.196] (+32 copies/layer) | 1.237 [1.231-1.242] (+32 copies/layer) | 1.117 [1.112-1.121] (+32 copies/layer) |
| deepinfer_R32@random | 1.227 [1.220-1.236] (+32 copies/layer) | 1.255 [1.250-1.261] (+32 copies/layer) | 1.167 [1.164-1.169] (+32 copies/layer) |
| eplb_R0 | 1.277 [1.270-1.283] | 1.277 [1.272-1.282] | 1.188 [1.184-1.191] |
| eplb_R32 | 1.224 [1.216-1.232] (+32 copies/layer) | 1.243 [1.238-1.248] (+32 copies/layer) | 1.128 [1.123-1.133] (+32 copies/layer) |
| eplb_R32@random | 1.270 [1.263-1.277] (+32 copies/layer) | 1.262 [1.258-1.267] (+32 copies/layer) | 1.192 [1.189-1.196] (+32 copies/layer) |
| integrality_floor | 1.000 | 1.000 | 1.000 |
| lp_floor@deepinfer_R32 | 1.185 [1.177-1.196] | 1.237 [1.231-1.242] | 1.116 [1.112-1.121] |
| lp_floor@eplb_R32 | 1.223 [1.216-1.232] | 1.243 [1.238-1.248] | 1.128 [1.123-1.133] |
| moonep_floor | 1.000 (4.0 copies/step) | 1.000 (3.8 copies/step) | 1.000 (4.1 copies/step) |
| null_random | 1.276 [1.269-1.284] | 1.265 [1.261-1.270] | 1.196 [1.194-1.199] |
| null_uniform_mc | 1.029 | 1.029 | 1.029 |
| oracle_R32 | 1.000 [1.000-1.000] (4.3 copies/step) | 1.000 [1.000-1.000] (4.0 copies/step) | 1.000 [1.000-1.000] (4.5 copies/step) |
| *trigger ratio* | realizable 1.65/1.227 = **1.345** (deepinfer_R32@random); idealized-dispatch 1.392 (deepinfer_R32); bound 1.000 (oracle_R32) -> 1.650 | realizable 1.65/1.255 = **1.314** (deepinfer_R32@random); idealized-dispatch 1.334 (deepinfer_R32); bound 1.000 (oracle_R32) -> 1.650 | realizable 1.65/1.167 = **1.414** (deepinfer_R32@random); idealized-dispatch 1.478 (deepinfer_R32); bound 1.000 (oracle_R32) -> 1.650 |

## B12/decode/T1

| arm | code | prose | reasoning |
|---|---|---|---|
| contiguous | 3.080 [3.070-3.089] | 3.042 [3.030-3.056] | 3.040 [3.029-3.049] |
| deepinfer_R0 | 2.931 [2.922-2.940] | 2.962 [2.948-2.977] | 2.907 [2.887-2.923] |
| deepinfer_R32 | 2.944 [2.935-2.952] (+32 copies/layer) | 2.959 [2.944-2.972] (+32 copies/layer) | 2.918 [2.899-2.933] (+32 copies/layer) |
| deepinfer_R32@random | 2.948 [2.939-2.955] (+32 copies/layer) | 2.962 [2.948-2.975] (+32 copies/layer) | 2.922 [2.905-2.937] (+32 copies/layer) |
| eplb_R0 | 3.028 [3.020-3.034] | 3.062 [3.050-3.074] | 3.041 [3.033-3.048] |
| eplb_R32 | 2.959 [2.950-2.969] (+32 copies/layer) | 3.003 [2.991-3.016] (+32 copies/layer) | 2.944 [2.920-2.965] (+32 copies/layer) |
| eplb_R32@random | 3.039 [3.033-3.045] (+32 copies/layer) | 3.038 [3.027-3.051] (+32 copies/layer) | 3.047 [3.040-3.053] (+32 copies/layer) |
| integrality_floor | 1.200 | 1.200 | 1.200 |
| lp_floor@deepinfer_R32 | 2.944 [2.935-2.952] | 2.959 [2.944-2.972] | 2.918 [2.899-2.933] |
| lp_floor@eplb_R32 | 2.948 [2.938-2.958] | 3.000 [2.987-3.012] | 2.928 [2.899-2.953] |
| moonep_floor | 1.200 (3.0 copies/step) | 1.200 (3.0 copies/step) | 1.200 (3.0 copies/step) |
| null_random | 3.052 [3.049-3.055] | 3.043 [3.041-3.045] | 3.051 [3.049-3.052] |
| null_uniform_mc | 3.077 | 3.077 | 3.077 |
| oracle_R32 | 1.200 [1.200-1.200] (3.0 copies/step) | 1.200 [1.200-1.200] (3.0 copies/step) | 1.200 [1.200-1.200] (3.0 copies/step) |
| *trigger ratio* | realizable 1.65/2.931 = **0.563** (deepinfer_R0); idealized-dispatch 0.560 (deepinfer_R32); bound 1.200 (oracle_R32) -> 1.375 | realizable 1.65/2.962 = **0.557** (deepinfer_R0); idealized-dispatch 0.558 (deepinfer_R32); bound 1.200 (oracle_R32) -> 1.375 | realizable 1.65/2.907 = **0.568** (deepinfer_R0); idealized-dispatch 0.565 (deepinfer_R32); bound 1.200 (oracle_R32) -> 1.375 |

## B12/decode/T3

| arm | code | prose | reasoning |
|---|---|---|---|
| contiguous | 2.528 [2.497-2.554] | 2.533 [2.508-2.554] | 2.474 [2.456-2.491] |
| deepinfer_R0 | 2.378 [2.354-2.399] | 2.485 [2.460-2.510] | 2.389 [2.371-2.405] |
| deepinfer_R32 | 2.308 [2.291-2.321] (+32 copies/layer) | 2.442 [2.417-2.468] (+32 copies/layer) | 2.275 [2.237-2.309] (+32 copies/layer) |
| deepinfer_R32@random | 2.382 [2.365-2.395] (+32 copies/layer) | 2.480 [2.458-2.503] (+32 copies/layer) | 2.377 [2.354-2.396] (+32 copies/layer) |
| eplb_R0 | 2.478 [2.447-2.506] | 2.565 [2.538-2.587] | 2.476 [2.463-2.488] |
| eplb_R32 | 2.399 [2.369-2.428] (+32 copies/layer) | 2.502 [2.480-2.525] (+32 copies/layer) | 2.381 [2.345-2.410] (+32 copies/layer) |
| eplb_R32@random | 2.469 [2.438-2.498] (+32 copies/layer) | 2.533 [2.514-2.555] (+32 copies/layer) | 2.474 [2.454-2.489] (+32 copies/layer) |
| integrality_floor | 1.200 | 1.200 | 1.200 |
| lp_floor@deepinfer_R32 | 2.304 [2.288-2.317] | 2.440 [2.415-2.466] | 2.268 [2.228-2.303] |
| lp_floor@eplb_R32 | 2.395 [2.365-2.423] | 2.500 [2.478-2.523] | 2.375 [2.338-2.405] |
| moonep_floor | 1.200 (4.3 copies/step) | 1.200 (4.2 copies/step) | 1.200 (4.2 copies/step) |
| null_random | 2.494 [2.463-2.523] | 2.546 [2.523-2.564] | 2.479 [2.469-2.488] |
| null_uniform_mc | 2.138 | 2.138 | 2.138 |
| oracle_R32 | 1.200 [1.200-1.200] (4.3 copies/step) | 1.200 [1.200-1.200] (4.2 copies/step) | 1.200 [1.200-1.200] (4.2 copies/step) |
| *trigger ratio* | realizable 1.65/2.378 = **0.694** (deepinfer_R0); idealized-dispatch 0.715 (deepinfer_R32); bound 1.200 (oracle_R32) -> 1.375 | realizable 1.65/2.480 = **0.665** (deepinfer_R32@random); idealized-dispatch 0.676 (deepinfer_R32); bound 1.200 (oracle_R32) -> 1.375 | realizable 1.65/2.377 = **0.694** (deepinfer_R32@random); idealized-dispatch 0.725 (deepinfer_R32); bound 1.200 (oracle_R32) -> 1.375 |

## B12/prefill/T512

| arm | code | prose | reasoning |
|---|---|---|---|
| contiguous | 1.874 [1.858-1.895] | 1.819 [1.803-1.836] | 1.573 [1.563-1.582] |
| deepinfer_R0 | 1.698 [1.677-1.722] | 1.813 [1.795-1.832] | 1.562 [1.554-1.571] |
| deepinfer_R32 | 1.654 [1.630-1.684] (+32 copies/layer) | 1.796 [1.780-1.813] (+32 copies/layer) | 1.458 [1.449-1.468] (+32 copies/layer) |
| deepinfer_R32@random | 1.701 [1.677-1.729] (+32 copies/layer) | 1.818 [1.803-1.835] (+32 copies/layer) | 1.513 [1.505-1.522] (+32 copies/layer) |
| eplb_R0 | 1.815 [1.794-1.838] | 1.830 [1.814-1.847] | 1.603 [1.594-1.612] |
| eplb_R32 | 1.756 [1.737-1.779] (+32 copies/layer) | 1.774 [1.760-1.790] (+32 copies/layer) | 1.523 [1.513-1.533] (+32 copies/layer) |
| eplb_R32@random | 1.809 [1.792-1.829] (+32 copies/layer) | 1.797 [1.782-1.812] (+32 copies/layer) | 1.593 [1.583-1.602] (+32 copies/layer) |
| integrality_floor | 1.001 | 1.001 | 1.001 |
| lp_floor@deepinfer_R32 | 1.654 [1.630-1.684] | 1.796 [1.780-1.813] | 1.458 [1.449-1.468] |
| lp_floor@eplb_R32 | 1.756 [1.737-1.779] | 1.774 [1.760-1.790] | 1.523 [1.512-1.532] |
| moonep_floor | 1.001 (11.7 copies/step) | 1.001 (11.3 copies/step) | 1.001 (11.7 copies/step) |
| null_random | 1.842 [1.824-1.865] | 1.824 [1.809-1.839] | 1.597 [1.589-1.605] |
| null_uniform_mc | 1.080 | 1.080 | 1.080 |
| oracle_R32 | 1.001 [1.001-1.001] (13.0 copies/step) | 1.001 [1.001-1.001] (12.5 copies/step) | 1.001 [1.001-1.001] (12.9 copies/step) |
| *trigger ratio* | realizable 1.65/1.698 = **0.972** (deepinfer_R0); idealized-dispatch 0.997 (deepinfer_R32); bound 1.001 (oracle_R32) -> 1.649 | realizable 1.65/1.797 = **0.918** (eplb_R32@random); idealized-dispatch 0.930 (eplb_R32); bound 1.001 (oracle_R32) -> 1.649 | realizable 1.65/1.513 = **1.091** (deepinfer_R32@random); idealized-dispatch 1.131 (deepinfer_R32); bound 1.001 (oracle_R32) -> 1.649 |

## DI-081 trigger

* B4/decode/T1: realizable static arms meet 1.10 on []; with idealized least-loaded dispatch on []; bounds meet 1.10 on ['code', 'prose', 'reasoning']
* B4/decode/T3: realizable static arms meet 1.10 on ['code', 'prose', 'reasoning']; with idealized least-loaded dispatch on ['code', 'prose', 'reasoning']; bounds meet 1.10 on ['code', 'prose', 'reasoning']
* B12/decode/T1: realizable static arms meet 1.10 on []; with idealized least-loaded dispatch on []; bounds meet 1.10 on ['code', 'prose', 'reasoning']
* B12/decode/T3: realizable static arms meet 1.10 on []; with idealized least-loaded dispatch on []; bounds meet 1.10 on ['code', 'prose', 'reasoning']
