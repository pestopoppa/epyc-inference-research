| model                          |       size |     params | backend    | ngl | threads |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --: | --------------: | -------------------: |
| nemotron_h_moe 75B.A9B IQ2_XXS - 2.0625 bpw |  20.50 GiB |    74.04 B | ROCm       |  99 |       8 |   1 |           pp512 |        899.17 ± 2.40 |
| nemotron_h_moe 75B.A9B IQ2_XXS - 2.0625 bpw |  20.50 GiB |    74.04 B | ROCm       |  99 |       8 |   1 |           tg128 |         53.44 ± 0.29 |

build: 59c1b915f (10348)
