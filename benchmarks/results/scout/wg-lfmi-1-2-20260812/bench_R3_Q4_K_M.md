| model                          |       size |     params | backend    | ngl | threads |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --: | --------------: | -------------------: |
| lfm2 1.2B Q4_K - Medium        | 694.76 MiB |     1.17 B | ROCm       |  99 |       8 |   0 |           pp512 |   13779.66 ± 2163.45 |
| lfm2 1.2B Q4_K - Medium        | 694.76 MiB |     1.17 B | ROCm       |  99 |       8 |   0 |           tg128 |        410.54 ± 4.39 |
| lfm2 1.2B Q4_K - Medium        | 694.76 MiB |     1.17 B | ROCm       |  99 |       8 |   0 |           tg512 |        407.64 ± 1.71 |
| lfm2 1.2B Q4_K - Medium        | 694.76 MiB |     1.17 B | ROCm       |  99 |       8 |   1 |           pp512 |    15217.28 ± 693.91 |
| lfm2 1.2B Q4_K - Medium        | 694.76 MiB |     1.17 B | ROCm       |  99 |       8 |   1 |           tg128 |        424.57 ± 7.68 |
| lfm2 1.2B Q4_K - Medium        | 694.76 MiB |     1.17 B | ROCm       |  99 |       8 |   1 |           tg512 |        437.54 ± 1.65 |

build: 0db32c06e (10125)
