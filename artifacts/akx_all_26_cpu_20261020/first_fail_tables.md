# AKX-ALL-26 CPU first-fail-k tables

cells=4761 groups=529 A/A-unclean cells=0/4113 missing groups=0 err cells=0
controls: {"ctl-positive-width-dependent": "PFFFFFFFF", "ctl-negative-width-independent": "PPPPPPPPP", "ctl-bitflip-injection": "FFFFFFFFF"} controls_ok=True

Legend: PASS = bit-equal for k=1..9; F@N = first width N whose batched bytes differ from the serial ones; UNSUP = cannot be constructed.

### CONTROL  route=`cpu`  nth=24
| type | ctl-bitflip-injection | ctl-negative-width-independent | ctl-positive-width-dependent |
|---|---|---|---|
| F32 | F@1 | PASS | F@2 |

### FLASH_ATTN_EXT  route=`cpu`  nth=1
| type | 2048 | 7168 | 16384 | 65536 |
|---|---|---|---|---|
| Q8_0 | PASS | PASS | PASS | PASS |
| F16 | PASS | PASS | PASS | PASS |
| Q4_0 | PASS | PASS | PASS | PASS |

### FLASH_ATTN_EXT  route=`cpu`  nth=24
| type | 2048 | 7168 | 16384 | 65536 |
|---|---|---|---|---|
| Q8_0 | PASS | PASS | PASS | PASS |
| F16 | F@2 | F@2 | F@2 | F@2 |
| Q4_0 | PASS | PASS | PASS | PASS |

### GATED_DELTA_NET  route=`cpu`  nth=1
| type | A3B-Hv32-scalar | Q38-27B-Hv48-kda | Q38-27B-Hv48-scalar |
|---|---|---|---|
| F32 | PASS | PASS | PASS |

### GATED_DELTA_NET  route=`cpu`  nth=24
| type | A3B-Hv32-scalar | Q38-27B-Hv48-kda | Q38-27B-Hv48-scalar |
|---|---|---|---|
| F32 | PASS | PASS | PASS |

### MUL_MAT  route=`cpu`  nth=1
| type | M2048 | M512 |
|---|---|---|
| Q8_0 | F@2 | F@2 |
| Q4_K | PASS | PASS |
| Q3_K | PASS | PASS |
| Q6_K | PASS | PASS |
| IQ2_XXS | F@2 | F@2 |
| IQ2_XS | F@2 | F@2 |
| IQ3_XXS | F@2 | F@2 |
| IQ3_S | F@2 | F@2 |
| IQ4_XS | PASS | PASS |

### MUL_MAT  route=`cpu`  nth=24
| type | D27-17408 | D27-5120 | D27-6144 | F2560 | F640 | M2048 | M512 |
|---|---|---|---|---|---|---|---|
| Q8_0 | F@2 | F@2 | F@2 | F@2 | F@2 | F@2 | F@2 |
| Q4_K | PASS | PASS | PASS | PASS | UNSUP | PASS | PASS |
| Q3_K | PASS | PASS | PASS | PASS | UNSUP | PASS | PASS |
| Q6_K | PASS | PASS | PASS | PASS | UNSUP | PASS | PASS |
| IQ2_XXS | F@2 | F@2 | F@2 | F@2 | UNSUP | F@2 | F@2 |
| IQ2_XS | F@2 | F@2 | F@2 | F@2 | UNSUP | F@2 | F@2 |
| IQ3_XXS | F@2 | F@2 | F@2 | F@2 | UNSUP | F@2 | F@2 |
| IQ3_S | F@2 | F@2 | F@2 | F@2 | UNSUP | F@2 | F@2 |
| IQ4_XS | PASS | PASS | PASS | PASS | UNSUP | PASS | PASS |

### MUL_MAT  route=`cpu-native`  nth=24
| type | D27-17408 | D27-5120 | D27-6144 | F2560 | F640 | M2048 | M512 |
|---|---|---|---|---|---|---|---|
| Q8_0 | F@2 | F@2 | F@2 | F@2 | F@2 | F@2 | F@2 |
| Q4_K | PASS | PASS | PASS | PASS | UNSUP | PASS | PASS |
| Q3_K | PASS | PASS | PASS | PASS | UNSUP | PASS | PASS |
| Q6_K | PASS | PASS | PASS | PASS | UNSUP | PASS | PASS |
| IQ2_XXS | PASS | PASS | PASS | PASS | UNSUP | PASS | PASS |
| IQ2_XS | PASS | PASS | PASS | PASS | UNSUP | PASS | PASS |
| IQ3_XXS | PASS | PASS | PASS | PASS | UNSUP | PASS | PASS |
| IQ3_S | PASS | PASS | PASS | PASS | UNSUP | PASS | PASS |
| IQ4_XS | PASS | PASS | PASS | PASS | UNSUP | PASS | PASS |

### MUL_MAT  route=`cpu[GGML_IQK=1,GGML_IQK_Q8_0=1]`  nth=24
| type | D27-17408 | D27-5120 | D27-6144 | F2560 | F640 | M2048 | M512 |
|---|---|---|---|---|---|---|---|
| Q8_0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS |

### MUL_MAT  route=`cpu[GGML_IQK=1,GGML_ROWEXACT_N=9]`  nth=24
| type | D27-17408 | D27-5120 | D27-6144 | F2560 | F640 | M2048 | M512 |
|---|---|---|---|---|---|---|---|
| Q8_0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| Q4_K | PASS | PASS | PASS | PASS | UNSUP | PASS | PASS |
| Q3_K | PASS | PASS | PASS | PASS | UNSUP | PASS | PASS |
| Q6_K | PASS | PASS | PASS | PASS | UNSUP | PASS | PASS |
| IQ2_XXS | PASS | PASS | PASS | PASS | UNSUP | PASS | PASS |
| IQ2_XS | PASS | PASS | PASS | PASS | UNSUP | PASS | PASS |
| IQ3_XXS | PASS | PASS | PASS | PASS | UNSUP | PASS | PASS |
| IQ3_S | PASS | PASS | PASS | PASS | UNSUP | PASS | PASS |
| IQ4_XS | PASS | PASS | PASS | PASS | UNSUP | PASS | PASS |

### MUL_MAT_ID  route=`cpu`  nth=1  ids-mode=disjoint
| type | M2048 | M512 |
|---|---|---|
| Q8_0 | PASS | PASS |
| Q4_K | PASS | PASS |
| Q3_K | PASS | PASS |
| Q6_K | PASS | PASS |
| IQ2_XXS | PASS | PASS |
| IQ2_XS | PASS | PASS |
| IQ3_XXS | F@2 | F@2 |
| IQ3_S | PASS | PASS |
| IQ4_XS | PASS | PASS |

### MUL_MAT_ID  route=`cpu`  nth=1  ids-mode=rand
| type | M2048 | M512 |
|---|---|---|
| Q8_0 | PASS | PASS |
| Q4_K | PASS | PASS |
| Q3_K | PASS | PASS |
| Q6_K | PASS | PASS |
| IQ2_XXS | F@5 | F@5 |
| IQ2_XS | F@5 | F@5 |
| IQ3_XXS | F@2 | F@2 |
| IQ3_S | F@5 | F@5 |
| IQ4_XS | PASS | PASS |

### MUL_MAT_ID  route=`cpu`  nth=1  ids-mode=shared
| type | M2048 | M512 |
|---|---|---|
| Q8_0 | PASS | PASS |
| Q4_K | PASS | PASS |
| Q3_K | PASS | PASS |
| Q6_K | PASS | PASS |
| IQ2_XXS | F@2 | F@2 |
| IQ2_XS | F@2 | F@2 |
| IQ3_XXS | F@2 | F@2 |
| IQ3_S | F@2 | F@2 |
| IQ4_XS | PASS | PASS |

### MUL_MAT_ID  route=`cpu`  nth=24  ids-mode=disjoint
| type | F2560 | F640 | M2048 | M512 |
|---|---|---|---|---|
| Q8_0 | PASS | PASS | PASS | PASS |
| Q4_K | PASS | UNSUP | PASS | PASS |
| Q3_K | PASS | UNSUP | PASS | PASS |
| Q6_K | PASS | UNSUP | PASS | PASS |
| IQ2_XXS | PASS | UNSUP | PASS | PASS |
| IQ2_XS | PASS | UNSUP | PASS | PASS |
| IQ3_XXS | F@2 | UNSUP | F@2 | F@2 |
| IQ3_S | PASS | UNSUP | PASS | PASS |
| IQ4_XS | PASS | UNSUP | PASS | PASS |

### MUL_MAT_ID  route=`cpu`  nth=24  ids-mode=rand
| type | F2560 | F640 | M2048 | M512 |
|---|---|---|---|---|
| Q8_0 | PASS | PASS | PASS | PASS |
| Q4_K | PASS | UNSUP | PASS | PASS |
| Q3_K | PASS | UNSUP | PASS | PASS |
| Q6_K | PASS | UNSUP | PASS | PASS |
| IQ2_XXS | F@5 | UNSUP | F@5 | F@5 |
| IQ2_XS | F@5 | UNSUP | F@5 | F@5 |
| IQ3_XXS | F@2 | UNSUP | F@2 | F@2 |
| IQ3_S | F@5 | UNSUP | F@5 | F@5 |
| IQ4_XS | PASS | UNSUP | PASS | PASS |

### MUL_MAT_ID  route=`cpu`  nth=24  ids-mode=shared
| type | F2560 | F640 | M2048 | M512 |
|---|---|---|---|---|
| Q8_0 | PASS | PASS | PASS | PASS |
| Q4_K | PASS | UNSUP | PASS | PASS |
| Q3_K | PASS | UNSUP | PASS | PASS |
| Q6_K | PASS | UNSUP | PASS | PASS |
| IQ2_XXS | F@2 | UNSUP | F@2 | F@2 |
| IQ2_XS | F@2 | UNSUP | F@2 | F@2 |
| IQ3_XXS | F@2 | UNSUP | F@2 | F@2 |
| IQ3_S | F@2 | UNSUP | F@2 | F@2 |
| IQ4_XS | PASS | UNSUP | PASS | PASS |

### MUL_MAT_ID  route=`cpu-native`  nth=24  ids-mode=disjoint
| type | F2560 | F640 | M2048 | M512 |
|---|---|---|---|---|
| Q8_0 | PASS | PASS | PASS | PASS |
| Q4_K | PASS | UNSUP | PASS | PASS |
| Q3_K | PASS | UNSUP | PASS | PASS |
| Q6_K | PASS | UNSUP | PASS | PASS |
| IQ2_XXS | PASS | UNSUP | PASS | PASS |
| IQ2_XS | PASS | UNSUP | PASS | PASS |
| IQ3_XXS | PASS | UNSUP | PASS | PASS |
| IQ3_S | PASS | UNSUP | PASS | PASS |
| IQ4_XS | PASS | UNSUP | PASS | PASS |

### MUL_MAT_ID  route=`cpu-native`  nth=24  ids-mode=rand
| type | F2560 | F640 | M2048 | M512 |
|---|---|---|---|---|
| Q8_0 | PASS | PASS | PASS | PASS |
| Q4_K | PASS | UNSUP | PASS | PASS |
| Q3_K | PASS | UNSUP | PASS | PASS |
| Q6_K | PASS | UNSUP | PASS | PASS |
| IQ2_XXS | PASS | UNSUP | PASS | PASS |
| IQ2_XS | PASS | UNSUP | PASS | PASS |
| IQ3_XXS | PASS | UNSUP | PASS | PASS |
| IQ3_S | PASS | UNSUP | PASS | PASS |
| IQ4_XS | PASS | UNSUP | PASS | PASS |

### MUL_MAT_ID  route=`cpu-native`  nth=24  ids-mode=shared
| type | F2560 | F640 | M2048 | M512 |
|---|---|---|---|---|
| Q8_0 | PASS | PASS | PASS | PASS |
| Q4_K | PASS | UNSUP | PASS | PASS |
| Q3_K | PASS | UNSUP | PASS | PASS |
| Q6_K | PASS | UNSUP | PASS | PASS |
| IQ2_XXS | PASS | UNSUP | PASS | PASS |
| IQ2_XS | PASS | UNSUP | PASS | PASS |
| IQ3_XXS | PASS | UNSUP | PASS | PASS |
| IQ3_S | PASS | UNSUP | PASS | PASS |
| IQ4_XS | PASS | UNSUP | PASS | PASS |

### MUL_MAT_ID  route=`cpu[GGML_IQK=1,GGML_IQK_Q8_0=1]`  nth=24  ids-mode=disjoint
| type | F2560 | F640 | M2048 | M512 |
|---|---|---|---|---|
| Q8_0 | PASS | PASS | PASS | PASS |

### MUL_MAT_ID  route=`cpu[GGML_IQK=1,GGML_IQK_Q8_0=1]`  nth=24  ids-mode=rand
| type | F2560 | F640 | M2048 | M512 |
|---|---|---|---|---|
| Q8_0 | PASS | PASS | PASS | PASS |

### MUL_MAT_ID  route=`cpu[GGML_IQK=1,GGML_IQK_Q8_0=1]`  nth=24  ids-mode=shared
| type | F2560 | F640 | M2048 | M512 |
|---|---|---|---|---|
| Q8_0 | PASS | PASS | PASS | PASS |

## Prediction scoring

| id | source | selector | predicted | result | measured |
|---|---|---|---|---|---|
| CPU-MM-Q8_0-ALL | frozen | cpu nth24 MUL_MAT Q8_0 | 2 | confirmed | D27-5120:PFFFFFFFF; D27-17408:PFFFFFFFF; D27-6144:PFFFFFFFF; M2048:PFFFFFFFF; M512:PFFFFFFFF; F2560:PFFFFFFFF; F640:PFFFFFFFF |
| CPU-MM-Q4_K-ALL | frozen | cpu nth24 MUL_MAT Q4_K | PASS | confirmed | D27-5120:PPPPPPPPP; D27-17408:PPPPPPPPP; D27-6144:PPPPPPPPP; M2048:PPPPPPPPP; M512:PPPPPPPPP; F2560:PPPPPPPPP |
| CPU-MM-Q3_K-ALL | frozen | cpu nth24 MUL_MAT Q3_K | PASS | confirmed | D27-5120:PPPPPPPPP; D27-17408:PPPPPPPPP; D27-6144:PPPPPPPPP; M2048:PPPPPPPPP; M512:PPPPPPPPP; F2560:PPPPPPPPP |
| CPU-MM-Q6_K-ALL | frozen | cpu nth24 MUL_MAT Q6_K | PASS | confirmed | D27-5120:PPPPPPPPP; D27-17408:PPPPPPPPP; D27-6144:PPPPPPPPP; M2048:PPPPPPPPP; M512:PPPPPPPPP; F2560:PPPPPPPPP |
| CPU-MM-IQ2_XXS-ALL | frozen | cpu nth24 MUL_MAT IQ2_XXS | 2 | confirmed | D27-5120:PFFFFFFFF; D27-17408:PFFFFFFFF; D27-6144:PFFFFFFFF; M2048:PFFFFFFFF; M512:PFFFFFFFF; F2560:PFFFFFFFF |
| CPU-MM-IQ2_XS-ALL | frozen | cpu nth24 MUL_MAT IQ2_XS | 2 | confirmed | D27-5120:PFFFFFFFF; D27-17408:PFFFFFFFF; D27-6144:PFFFFFFFF; M2048:PFFFFFFFF; M512:PFFFFFFFF; F2560:PFFFFFFFF |
| CPU-MM-IQ3_XXS-ALL | frozen | cpu nth24 MUL_MAT IQ3_XXS | 2 | confirmed | D27-5120:PFFFFFFFF; D27-17408:PFFFFFFFF; D27-6144:PFFFFFFFF; M2048:PFFFFFFFF; M512:PFFFFFFFF; F2560:PFFFFFFFF |
| CPU-MM-IQ3_S-ALL | frozen | cpu nth24 MUL_MAT IQ3_S | 2 | confirmed | D27-5120:PFFFFFFFF; D27-17408:PFFFFFFFF; D27-6144:PFFFFFFFF; M2048:PFFFFFFFF; M512:PFFFFFFFF; F2560:PFFFFFFFF |
| CPU-MM-IQ4_XS-ALL | frozen | cpu nth24 MUL_MAT IQ4_XS | PASS | confirmed | D27-5120:PPPPPPPPP; D27-17408:PPPPPPPPP; D27-6144:PPPPPPPPP; M2048:PPPPPPPPP; M512:PPPPPPPPP; F2560:PPPPPPPPP |
| CPUN-MM-Q8_0-ALL | frozen | cpu-native nth24 MUL_MAT Q8_0 | 2 | confirmed | D27-5120:PFFFFFFFF; D27-17408:PFFFFFFFF; D27-6144:PFFFFFFFF; M2048:PFFFFFFFF; M512:PFFFFFFFF; F2560:PFFFFFFFF; F640:PFFFFFFFF |
| CPUN-MM-Q4_K-ALL | frozen | cpu-native nth24 MUL_MAT Q4_K | PASS | confirmed | D27-5120:PPPPPPPPP; D27-17408:PPPPPPPPP; D27-6144:PPPPPPPPP; M2048:PPPPPPPPP; M512:PPPPPPPPP; F2560:PPPPPPPPP |
| CPUN-MM-Q3_K-ALL | frozen | cpu-native nth24 MUL_MAT Q3_K | PASS | confirmed | D27-5120:PPPPPPPPP; D27-17408:PPPPPPPPP; D27-6144:PPPPPPPPP; M2048:PPPPPPPPP; M512:PPPPPPPPP; F2560:PPPPPPPPP |
| CPUN-MM-Q6_K-ALL | frozen | cpu-native nth24 MUL_MAT Q6_K | PASS | confirmed | D27-5120:PPPPPPPPP; D27-17408:PPPPPPPPP; D27-6144:PPPPPPPPP; M2048:PPPPPPPPP; M512:PPPPPPPPP; F2560:PPPPPPPPP |
| CPUN-MM-IQ2_XXS-ALL | frozen | cpu-native nth24 MUL_MAT IQ2_XXS | PASS | confirmed | D27-5120:PPPPPPPPP; D27-17408:PPPPPPPPP; D27-6144:PPPPPPPPP; M2048:PPPPPPPPP; M512:PPPPPPPPP; F2560:PPPPPPPPP |
| CPUN-MM-IQ2_XS-ALL | frozen | cpu-native nth24 MUL_MAT IQ2_XS | PASS | confirmed | D27-5120:PPPPPPPPP; D27-17408:PPPPPPPPP; D27-6144:PPPPPPPPP; M2048:PPPPPPPPP; M512:PPPPPPPPP; F2560:PPPPPPPPP |
| CPUN-MM-IQ3_XXS-ALL | frozen | cpu-native nth24 MUL_MAT IQ3_XXS | PASS | confirmed | D27-5120:PPPPPPPPP; D27-17408:PPPPPPPPP; D27-6144:PPPPPPPPP; M2048:PPPPPPPPP; M512:PPPPPPPPP; F2560:PPPPPPPPP |
| CPUN-MM-IQ3_S-ALL | frozen | cpu-native nth24 MUL_MAT IQ3_S | PASS | confirmed | D27-5120:PPPPPPPPP; D27-17408:PPPPPPPPP; D27-6144:PPPPPPPPP; M2048:PPPPPPPPP; M512:PPPPPPPPP; F2560:PPPPPPPPP |
| CPUN-MM-IQ4_XS-ALL | frozen | cpu-native nth24 MUL_MAT IQ4_XS | PASS | confirmed | D27-5120:PPPPPPPPP; D27-17408:PPPPPPPPP; D27-6144:PPPPPPPPP; M2048:PPPPPPPPP; M512:PPPPPPPPP; F2560:PPPPPPPPP |
| CPUQ8-MM-Q8_0-ALL | frozen | cpu[GGML_IQK=1,GGML_IQK_Q8_0=1] nth24 MUL_MAT Q8_0 | PASS | confirmed | D27-5120:PPPPPPPPP; D27-17408:PPPPPPPPP; D27-6144:PPPPPPPPP; M2048:PPPPPPPPP; M512:PPPPPPPPP; F2560:PPPPPPPPP; F640:PPPPPPPPP |
| CPU-MMID-Q8_0-ALL | frozen | cpu nth24 MUL_MAT_ID Q8_0 rand | PASS | confirmed | M2048/rand:PPPPPPPPP; M512/rand:PPPPPPPPP; F2560/rand:PPPPPPPPP; F640/rand:PPPPPPPPP |
| CPU-MMID-Q4_K-ALL | frozen | cpu nth24 MUL_MAT_ID Q4_K rand | PASS | confirmed | M2048/rand:PPPPPPPPP; M512/rand:PPPPPPPPP; F2560/rand:PPPPPPPPP |
| CPU-MMID-Q3_K-ALL | frozen | cpu nth24 MUL_MAT_ID Q3_K rand | PASS | confirmed | M2048/rand:PPPPPPPPP; M512/rand:PPPPPPPPP; F2560/rand:PPPPPPPPP |
| CPU-MMID-Q6_K-ALL | frozen | cpu nth24 MUL_MAT_ID Q6_K rand | PASS | confirmed | M2048/rand:PPPPPPPPP; M512/rand:PPPPPPPPP; F2560/rand:PPPPPPPPP |
| CPU-MMID-IQ2_XXS-ALL | frozen | cpu nth24 MUL_MAT_ID IQ2_XXS rand | 3 | falsified | M2048/rand:PPPPFFFFF; M512/rand:PPPPFFFFF; F2560/rand:PPPPFFFFF |
| CPU-MMID-IQ2_XS-ALL | frozen | cpu nth24 MUL_MAT_ID IQ2_XS rand | 3 | falsified | M2048/rand:PPPPFFFFF; M512/rand:PPPPFFFFF; F2560/rand:PPPPFFFFF |
| CPU-MMID-IQ3_XXS-ALL | frozen | cpu nth24 MUL_MAT_ID IQ3_XXS rand | 2 | confirmed | M2048/rand:PFFFFFFFF; M512/rand:PFFFFFFFF; F2560/rand:PFFFFFFFF |
| CPU-MMID-IQ3_S-ALL | frozen | cpu nth24 MUL_MAT_ID IQ3_S rand | 3 | falsified | M2048/rand:PPPPFFFFF; M512/rand:PPPPFFFFF; F2560/rand:PPPPFFFFF |
| CPU-MMID-IQ4_XS-ALL | frozen | cpu nth24 MUL_MAT_ID IQ4_XS rand | PASS | confirmed | M2048/rand:PPPPPPPPP; M512/rand:PPPPPPPPP; F2560/rand:PPPPPPPPP |
| CPUN-MMID-Q8_0-ALL | frozen | cpu-native nth24 MUL_MAT_ID Q8_0 rand | PASS | confirmed | M2048/rand:PPPPPPPPP; M512/rand:PPPPPPPPP; F2560/rand:PPPPPPPPP; F640/rand:PPPPPPPPP |
| CPUN-MMID-Q4_K-ALL | frozen | cpu-native nth24 MUL_MAT_ID Q4_K rand | PASS | confirmed | M2048/rand:PPPPPPPPP; M512/rand:PPPPPPPPP; F2560/rand:PPPPPPPPP |
| CPUN-MMID-Q3_K-ALL | frozen | cpu-native nth24 MUL_MAT_ID Q3_K rand | PASS | confirmed | M2048/rand:PPPPPPPPP; M512/rand:PPPPPPPPP; F2560/rand:PPPPPPPPP |
| CPUN-MMID-Q6_K-ALL | frozen | cpu-native nth24 MUL_MAT_ID Q6_K rand | PASS | confirmed | M2048/rand:PPPPPPPPP; M512/rand:PPPPPPPPP; F2560/rand:PPPPPPPPP |
| CPUN-MMID-IQ2_XXS-ALL | frozen | cpu-native nth24 MUL_MAT_ID IQ2_XXS rand | PASS | confirmed | M2048/rand:PPPPPPPPP; M512/rand:PPPPPPPPP; F2560/rand:PPPPPPPPP |
| CPUN-MMID-IQ2_XS-ALL | frozen | cpu-native nth24 MUL_MAT_ID IQ2_XS rand | PASS | confirmed | M2048/rand:PPPPPPPPP; M512/rand:PPPPPPPPP; F2560/rand:PPPPPPPPP |
| CPUN-MMID-IQ3_XXS-ALL | frozen | cpu-native nth24 MUL_MAT_ID IQ3_XXS rand | PASS | confirmed | M2048/rand:PPPPPPPPP; M512/rand:PPPPPPPPP; F2560/rand:PPPPPPPPP |
| CPUN-MMID-IQ3_S-ALL | frozen | cpu-native nth24 MUL_MAT_ID IQ3_S rand | PASS | confirmed | M2048/rand:PPPPPPPPP; M512/rand:PPPPPPPPP; F2560/rand:PPPPPPPPP |
| CPUN-MMID-IQ4_XS-ALL | frozen | cpu-native nth24 MUL_MAT_ID IQ4_XS rand | PASS | confirmed | M2048/rand:PPPPPPPPP; M512/rand:PPPPPPPPP; F2560/rand:PPPPPPPPP |
| CPU-MM-Q4_K-F640 | frozen | cpu nth24 MUL_MAT Q4_K F640 | UNSUPPORTED | confirmed | F640:UUUUUUUUU |
| CPU-MMID-Q4_K-F640 | frozen | cpu nth24 MUL_MAT_ID Q4_K F640 | UNSUPPORTED | confirmed | F640/rand:UUUUUUUUU; F640/shared:UUUUUUUUU; F640/disjoint:UUUUUUUUU |
| CPU-MM-Q3_K-F640 | frozen | cpu nth24 MUL_MAT Q3_K F640 | UNSUPPORTED | confirmed | F640:UUUUUUUUU |
| CPU-MMID-Q3_K-F640 | frozen | cpu nth24 MUL_MAT_ID Q3_K F640 | UNSUPPORTED | confirmed | F640/rand:UUUUUUUUU; F640/shared:UUUUUUUUU; F640/disjoint:UUUUUUUUU |
| CPU-MM-Q6_K-F640 | frozen | cpu nth24 MUL_MAT Q6_K F640 | UNSUPPORTED | confirmed | F640:UUUUUUUUU |
| CPU-MMID-Q6_K-F640 | frozen | cpu nth24 MUL_MAT_ID Q6_K F640 | UNSUPPORTED | confirmed | F640/rand:UUUUUUUUU; F640/shared:UUUUUUUUU; F640/disjoint:UUUUUUUUU |
| CPU-MM-IQ2_XXS-F640 | frozen | cpu nth24 MUL_MAT IQ2_XXS F640 | UNSUPPORTED | confirmed | F640:UUUUUUUUU |
| CPU-MMID-IQ2_XXS-F640 | frozen | cpu nth24 MUL_MAT_ID IQ2_XXS F640 | UNSUPPORTED | confirmed | F640/rand:UUUUUUUUU; F640/shared:UUUUUUUUU; F640/disjoint:UUUUUUUUU |
| CPU-MM-IQ2_XS-F640 | frozen | cpu nth24 MUL_MAT IQ2_XS F640 | UNSUPPORTED | confirmed | F640:UUUUUUUUU |
| CPU-MMID-IQ2_XS-F640 | frozen | cpu nth24 MUL_MAT_ID IQ2_XS F640 | UNSUPPORTED | confirmed | F640/rand:UUUUUUUUU; F640/shared:UUUUUUUUU; F640/disjoint:UUUUUUUUU |
| CPU-MM-IQ3_XXS-F640 | frozen | cpu nth24 MUL_MAT IQ3_XXS F640 | UNSUPPORTED | confirmed | F640:UUUUUUUUU |
| CPU-MMID-IQ3_XXS-F640 | frozen | cpu nth24 MUL_MAT_ID IQ3_XXS F640 | UNSUPPORTED | confirmed | F640/rand:UUUUUUUUU; F640/shared:UUUUUUUUU; F640/disjoint:UUUUUUUUU |
| CPU-MM-IQ3_S-F640 | frozen | cpu nth24 MUL_MAT IQ3_S F640 | UNSUPPORTED | confirmed | F640:UUUUUUUUU |
| CPU-MMID-IQ3_S-F640 | frozen | cpu nth24 MUL_MAT_ID IQ3_S F640 | UNSUPPORTED | confirmed | F640/rand:UUUUUUUUU; F640/shared:UUUUUUUUU; F640/disjoint:UUUUUUUUU |
| CPU-MM-IQ4_XS-F640 | frozen | cpu nth24 MUL_MAT IQ4_XS F640 | UNSUPPORTED | confirmed | F640:UUUUUUUUU |
| CPU-MMID-IQ4_XS-F640 | frozen | cpu nth24 MUL_MAT_ID IQ4_XS F640 | UNSUPPORTED | confirmed | F640/rand:UUUUUUUUU; F640/shared:UUUUUUUUU; F640/disjoint:UUUUUUUUU |
| CPU-FA-F16-2048 | frozen | cpu nth24 FLASH_ATTN_EXT F16 2048 | 2 | confirmed | 2048:PFFFFFFFF |
| CPU-FA-F16-7168 | frozen | cpu nth24 FLASH_ATTN_EXT F16 7168 | 2 | confirmed | 7168:PFFFFFFFF |
| CPU-FA-F16-16384 | frozen | cpu nth24 FLASH_ATTN_EXT F16 16384 | 2 | confirmed | 16384:PFFFFFFFF |
| CPU-FA-F16-65536 | frozen | cpu nth24 FLASH_ATTN_EXT F16 65536 | 2 | confirmed | 65536:PFFFFFFFF |
| CPU-FA-Q8_0-2048 | frozen | cpu nth24 FLASH_ATTN_EXT Q8_0 2048 | PASS | confirmed | 2048:PPPPPPPPP |
| CPU-FA-Q8_0-7168 | frozen | cpu nth24 FLASH_ATTN_EXT Q8_0 7168 | PASS | confirmed | 7168:PPPPPPPPP |
| CPU-FA-Q8_0-16384 | frozen | cpu nth24 FLASH_ATTN_EXT Q8_0 16384 | PASS | confirmed | 16384:PPPPPPPPP |
| CPU-FA-Q8_0-65536 | frozen | cpu nth24 FLASH_ATTN_EXT Q8_0 65536 | PASS | confirmed | 65536:PPPPPPPPP |
| CPU-FA-Q4_0-2048 | frozen | cpu nth24 FLASH_ATTN_EXT Q4_0 2048 | PASS | confirmed | 2048:PPPPPPPPP |
| CPU-FA-Q4_0-7168 | frozen | cpu nth24 FLASH_ATTN_EXT Q4_0 7168 | PASS | confirmed | 7168:PPPPPPPPP |
| CPU-FA-Q4_0-16384 | frozen | cpu nth24 FLASH_ATTN_EXT Q4_0 16384 | PASS | confirmed | 16384:PPPPPPPPP |
| CPU-FA-Q4_0-65536 | frozen | cpu nth24 FLASH_ATTN_EXT Q4_0 65536 | PASS | confirmed | 65536:PPPPPPPPP |
| CPU-GDN-F32 | frozen | cpu nth24 GATED_DELTA_NET F32 | PASS | confirmed | Q38-27B-Hv48-scalar:PPPPPPPPP; A3B-Hv32-scalar:PPPPPPPPP |
| ADD-CPU-MMID-IQ2_XXS-shared | addendum | cpu nth24 MUL_MAT_ID IQ2_XXS shared | 2 | confirmed | M2048/shared:PFFFFFFFF; M512/shared:PFFFFFFFF; F2560/shared:PFFFFFFFF |
| ADD-CPU-MMID-IQ2_XXS-disjoint | addendum | cpu nth24 MUL_MAT_ID IQ2_XXS disjoint | PASS | confirmed | M2048/disjoint:PPPPPPPPP; M512/disjoint:PPPPPPPPP; F2560/disjoint:PPPPPPPPP |
| ADD-CPU-MMID-IQ2_XS-shared | addendum | cpu nth24 MUL_MAT_ID IQ2_XS shared | 2 | confirmed | M2048/shared:PFFFFFFFF; M512/shared:PFFFFFFFF; F2560/shared:PFFFFFFFF |
| ADD-CPU-MMID-IQ2_XS-disjoint | addendum | cpu nth24 MUL_MAT_ID IQ2_XS disjoint | PASS | confirmed | M2048/disjoint:PPPPPPPPP; M512/disjoint:PPPPPPPPP; F2560/disjoint:PPPPPPPPP |
| ADD-CPU-MMID-IQ3_S-shared | addendum | cpu nth24 MUL_MAT_ID IQ3_S shared | 2 | confirmed | M2048/shared:PFFFFFFFF; M512/shared:PFFFFFFFF; F2560/shared:PFFFFFFFF |
| ADD-CPU-MMID-IQ3_S-disjoint | addendum | cpu nth24 MUL_MAT_ID IQ3_S disjoint | PASS | confirmed | M2048/disjoint:PPPPPPPPP; M512/disjoint:PPPPPPPPP; F2560/disjoint:PPPPPPPPP |
| ADD-CPU-MMID-IQ3_XXS-shared | addendum | cpu nth24 MUL_MAT_ID IQ3_XXS shared | 2 | confirmed | M2048/shared:PFFFFFFFF; M512/shared:PFFFFFFFF; F2560/shared:PFFFFFFFF |
| ADD-CPU-MMID-IQ3_XXS-disjoint | addendum | cpu nth24 MUL_MAT_ID IQ3_XXS disjoint | 2 | confirmed | M2048/disjoint:PFFFFFFFF; M512/disjoint:PFFFFFFFF; F2560/disjoint:PFFFFFFFF |
| ADD-CPU-MMID-Q8_0-shared | addendum | cpu nth24 MUL_MAT_ID Q8_0 shared | PASS | confirmed | M2048/shared:PPPPPPPPP; M512/shared:PPPPPPPPP; F2560/shared:PPPPPPPPP; F640/shared:PPPPPPPPP |
| ADD-CPU-MMID-Q8_0-disjoint | addendum | cpu nth24 MUL_MAT_ID Q8_0 disjoint | PASS | confirmed | M2048/disjoint:PPPPPPPPP; M512/disjoint:PPPPPPPPP; F2560/disjoint:PPPPPPPPP; F640/disjoint:PPPPPPPPP |
| ADD-CPU-MMID-Q4_K-shared | addendum | cpu nth24 MUL_MAT_ID Q4_K shared | PASS | confirmed | M2048/shared:PPPPPPPPP; M512/shared:PPPPPPPPP; F2560/shared:PPPPPPPPP |
| ADD-CPU-MMID-Q4_K-disjoint | addendum | cpu nth24 MUL_MAT_ID Q4_K disjoint | PASS | confirmed | M2048/disjoint:PPPPPPPPP; M512/disjoint:PPPPPPPPP; F2560/disjoint:PPPPPPPPP |
| ADD-CPU-MMID-Q3_K-shared | addendum | cpu nth24 MUL_MAT_ID Q3_K shared | PASS | confirmed | M2048/shared:PPPPPPPPP; M512/shared:PPPPPPPPP; F2560/shared:PPPPPPPPP |
| ADD-CPU-MMID-Q3_K-disjoint | addendum | cpu nth24 MUL_MAT_ID Q3_K disjoint | PASS | confirmed | M2048/disjoint:PPPPPPPPP; M512/disjoint:PPPPPPPPP; F2560/disjoint:PPPPPPPPP |
| ADD-CPU-MMID-Q6_K-shared | addendum | cpu nth24 MUL_MAT_ID Q6_K shared | PASS | confirmed | M2048/shared:PPPPPPPPP; M512/shared:PPPPPPPPP; F2560/shared:PPPPPPPPP |
| ADD-CPU-MMID-Q6_K-disjoint | addendum | cpu nth24 MUL_MAT_ID Q6_K disjoint | PASS | confirmed | M2048/disjoint:PPPPPPPPP; M512/disjoint:PPPPPPPPP; F2560/disjoint:PPPPPPPPP |
| ADD-CPU-MMID-IQ4_XS-shared | addendum | cpu nth24 MUL_MAT_ID IQ4_XS shared | PASS | confirmed | M2048/shared:PPPPPPPPP; M512/shared:PPPPPPPPP; F2560/shared:PPPPPPPPP |
| ADD-CPU-MMID-IQ4_XS-disjoint | addendum | cpu nth24 MUL_MAT_ID IQ4_XS disjoint | PASS | confirmed | M2048/disjoint:PPPPPPPPP; M512/disjoint:PPPPPPPPP; F2560/disjoint:PPPPPPPPP |
| ADD-CPU-FA-F16-nth1 | addendum | cpu nth1 FLASH_ATTN_EXT F16 | PASS | confirmed | 2048:PPPPPPPPP; 7168:PPPPPPPPP; 16384:PPPPPPPPP; 65536:PPPPPPPPP |
| ADD-CPU-FA-Q8_0-nth1 | addendum | cpu nth1 FLASH_ATTN_EXT Q8_0 | PASS | confirmed | 2048:PPPPPPPPP; 7168:PPPPPPPPP; 16384:PPPPPPPPP; 65536:PPPPPPPPP |
| ADD-CPU-FA-Q4_0-nth1 | addendum | cpu nth1 FLASH_ATTN_EXT Q4_0 | PASS | confirmed | 2048:PPPPPPPPP; 7168:PPPPPPPPP; 16384:PPPPPPPPP; 65536:PPPPPPPPP |
| ADD-CPU-GDN-all-shapes-nth24 | addendum | cpu nth24 GATED_DELTA_NET F32 | PASS | confirmed | Q38-27B-Hv48-scalar:PPPPPPPPP; A3B-Hv32-scalar:PPPPPPPPP; Q38-27B-Hv48-kda:PPPPPPPPP |
| ADD-CPU-GDN-all-shapes-nth1 | addendum | cpu nth1 GATED_DELTA_NET F32 | PASS | confirmed | Q38-27B-Hv48-scalar:PPPPPPPPP; A3B-Hv32-scalar:PPPPPPPPP; Q38-27B-Hv48-kda:PPPPPPPPP |
| ADD-CPU-MM-Q8_0-nth1 | addendum | cpu nth1 MUL_MAT Q8_0 | 2 | confirmed | M2048:PFFFFFFFF; M512:PFFFFFFFF |
| ADD-CPU-MM-IQ2_XXS-nth1 | addendum | cpu nth1 MUL_MAT IQ2_XXS | 2 | confirmed | M2048:PFFFFFFFF; M512:PFFFFFFFF |
| ADD-CPU-MM-IQ2_XS-nth1 | addendum | cpu nth1 MUL_MAT IQ2_XS | 2 | confirmed | M2048:PFFFFFFFF; M512:PFFFFFFFF |
| ADD-CPU-MM-IQ3_XXS-nth1 | addendum | cpu nth1 MUL_MAT IQ3_XXS | 2 | confirmed | M2048:PFFFFFFFF; M512:PFFFFFFFF |
| ADD-CPU-MM-IQ3_S-nth1 | addendum | cpu nth1 MUL_MAT IQ3_S | 2 | confirmed | M2048:PFFFFFFFF; M512:PFFFFFFFF |
| ADD-CPU-MM-Q8_0-rowexact9 | addendum | cpu[GGML_IQK=1,GGML_ROWEXACT_N=9] nth24 MUL_MAT Q8_0 | PASS | confirmed | D27-5120:PPPPPPPPP; D27-17408:PPPPPPPPP; D27-6144:PPPPPPPPP; M2048:PPPPPPPPP; M512:PPPPPPPPP; F2560:PPPPPPPPP; F640:PPPPPPPPP |
| ADD-CPU-MM-Q4_K-rowexact9 | addendum | cpu[GGML_IQK=1,GGML_ROWEXACT_N=9] nth24 MUL_MAT Q4_K | PASS | confirmed | D27-5120:PPPPPPPPP; D27-17408:PPPPPPPPP; D27-6144:PPPPPPPPP; M2048:PPPPPPPPP; M512:PPPPPPPPP; F2560:PPPPPPPPP |
| ADD-CPU-MM-Q3_K-rowexact9 | addendum | cpu[GGML_IQK=1,GGML_ROWEXACT_N=9] nth24 MUL_MAT Q3_K | PASS | confirmed | D27-5120:PPPPPPPPP; D27-17408:PPPPPPPPP; D27-6144:PPPPPPPPP; M2048:PPPPPPPPP; M512:PPPPPPPPP; F2560:PPPPPPPPP |
| ADD-CPU-MM-Q6_K-rowexact9 | addendum | cpu[GGML_IQK=1,GGML_ROWEXACT_N=9] nth24 MUL_MAT Q6_K | PASS | confirmed | D27-5120:PPPPPPPPP; D27-17408:PPPPPPPPP; D27-6144:PPPPPPPPP; M2048:PPPPPPPPP; M512:PPPPPPPPP; F2560:PPPPPPPPP |
| ADD-CPU-MM-IQ2_XXS-rowexact9 | addendum | cpu[GGML_IQK=1,GGML_ROWEXACT_N=9] nth24 MUL_MAT IQ2_XXS | PASS | confirmed | D27-5120:PPPPPPPPP; D27-17408:PPPPPPPPP; D27-6144:PPPPPPPPP; M2048:PPPPPPPPP; M512:PPPPPPPPP; F2560:PPPPPPPPP |
| ADD-CPU-MM-IQ2_XS-rowexact9 | addendum | cpu[GGML_IQK=1,GGML_ROWEXACT_N=9] nth24 MUL_MAT IQ2_XS | PASS | confirmed | D27-5120:PPPPPPPPP; D27-17408:PPPPPPPPP; D27-6144:PPPPPPPPP; M2048:PPPPPPPPP; M512:PPPPPPPPP; F2560:PPPPPPPPP |
| ADD-CPU-MM-IQ3_XXS-rowexact9 | addendum | cpu[GGML_IQK=1,GGML_ROWEXACT_N=9] nth24 MUL_MAT IQ3_XXS | PASS | confirmed | D27-5120:PPPPPPPPP; D27-17408:PPPPPPPPP; D27-6144:PPPPPPPPP; M2048:PPPPPPPPP; M512:PPPPPPPPP; F2560:PPPPPPPPP |
| ADD-CPU-MM-IQ3_S-rowexact9 | addendum | cpu[GGML_IQK=1,GGML_ROWEXACT_N=9] nth24 MUL_MAT IQ3_S | PASS | confirmed | D27-5120:PPPPPPPPP; D27-17408:PPPPPPPPP; D27-6144:PPPPPPPPP; M2048:PPPPPPPPP; M512:PPPPPPPPP; F2560:PPPPPPPPP |
| ADD-CPU-MM-IQ4_XS-rowexact9 | addendum | cpu[GGML_IQK=1,GGML_ROWEXACT_N=9] nth24 MUL_MAT IQ4_XS | PASS | confirmed | D27-5120:PPPPPPPPP; D27-17408:PPPPPPPPP; D27-6144:PPPPPPPPP; M2048:PPPPPPPPP; M512:PPPPPPPPP; F2560:PPPPPPPPP |
| ADD-CTL-positive | addendum | cpu nth24 CONTROL F32 ctl-positive-width-dependent | 2 | confirmed | ctl-positive-width-dependent:PFFFFFFFF |
| ADD-CTL-negative | addendum | cpu nth24 CONTROL F32 ctl-negative-width-independent | PASS | confirmed | ctl-negative-width-independent:PPPPPPPPP |
| ADD-CTL-bitflip | addendum | cpu nth24 CONTROL F32 ctl-bitflip-injection | 1 | confirmed | ctl-bitflip-injection:FFFFFFFFF |
