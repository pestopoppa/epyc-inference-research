| model                          |       size |     params | backend    | ngl | threads |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --: | --------------: | -------------------: |
| lfm2 1.2B Q4_K - Medium        | 694.76 MiB |     1.17 B | ROCm       |  99 |       8 |   0 |           pp512 |   14001.55 ± 3057.87 |
| lfm2 1.2B Q4_K - Medium        | 694.76 MiB |     1.17 B | ROCm       |  99 |       8 |   0 |           tg128 |        419.95 ± 1.81 |
| lfm2 1.2B Q4_K - Medium        | 694.76 MiB |     1.17 B | ROCm       |  99 |       8 |   0 |           tg512 |        418.21 ± 1.38 |
| lfm2 1.2B Q4_K - Medium        | 694.76 MiB |     1.17 B | ROCm       |  99 |       8 |   1 |           pp512 |    16479.19 ± 480.06 |
| lfm2 1.2B Q4_K - Medium        | 694.76 MiB |     1.17 B | ROCm       |  99 |       8 |   1 |           tg128 |        430.34 ± 0.78 |
| lfm2 1.2B Q4_K - Medium        | 694.76 MiB |     1.17 B | ROCm       |  99 |       8 |   1 |           tg512 |        432.37 ± 0.44 |

build: 0db32c06e (10125)
