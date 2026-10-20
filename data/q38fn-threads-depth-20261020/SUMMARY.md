# Q38FN served thread x depth sweep (2026-10-20)

Binary cpu-20261004-b0ba1d427, UD-IQ4_XS + shared MTP head (n-max 4, p-min 0.5), production env/OMP stack, taskset 0-95 + interleave=all, f16 KV, greedy. Context: wikitext2_test via slot restore. Higher t/s = better.

## Prefill at depth (4096-token tail appended at depth, t/s)

| depth | -t48 -tb48 (current) | -t48 -tb96 | -t96 -tb96 |
|---|---|---|---|
| 512 | 163.6 | 197.6 | 211.3 |
| 8192 | 161.8 | 164.5 | 169.9 |
| 32768 | 116.9 | 116.4 | 113.3 |
| 65536 | 93.7 | 94.3 | 92.0 |
| 131072 | 72.4 | 72.7 | 71.3 |

## Decode at depth with MTP (256 greedy tokens, t/s, mean of reps)

| depth | -t48 -tb48 (current) | -t48 -tb96 | -t96 -tb96 |
|---|---|---|---|
| 512 | 32.8 | 36.9 | 36.0 |
| 8192 | 34.9 | 35.3 | 37.2 |
| 32768 | 29.7 | 29.6 | 29.0 |
| 65536 | 25.1 | 25.1 | 19.7 |
| 131072 | 15.9 | 16.9 | 17.0 |

## Draft acceptance (accepted/drafted) and greedy-output identity

| depth | -t48 -tb48 (current) | -t48 -tb96 | -t96 -tb96 |
|---|---|---|---|
| 512 | 0.66 / 2 distinct | 0.70 / 3 distinct | 0.66 / 2 distinct |
| 8192 | 0.79 / 2 distinct | 0.80 / 4 distinct | 0.79 / 2 distinct |
| 32768 | 0.78 / 1 distinct | 0.77 / 1 distinct | 0.78 / 1 distinct |
| 65536 | 0.78 / 1 distinct | 0.78 / 1 distinct | 0.78 / 1 distinct |
| 131072 | 0.61 / 1 distinct | 0.60 / 1 distinct | 0.60 / 1 distinct |

## Drift control (arm B run first and last)

| depth | B0 | B1 |
|---|---|---|
| 512 | 38.9 | 34.9 |
| 8192 | 33.9 | 36.7 |
| 32768 | 28.0 | 31.2 |
| 65536 | 24.4 | 25.9 |
| 131072 | — | — |
