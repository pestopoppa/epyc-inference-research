=== Validating canonical recipe ===
=== Canonical bench command ===
Binary:    /mnt/raid0/llm/llama.cpp/build/bin/llama-bench
Env:       LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp/build/bin:/usr/lib/llvm-20/lib OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active OMP_DYNAMIC=false GGML_IQK=1
Cmd:       taskset -c 0-95 numactl --interleave=all /mnt/raid0/llm/llama.cpp/build/bin/llama-bench -t 96 -fa 1 -mmp 0 -m /mnt/raid0/llm/models/GLM-5.2-UD-IQ2_M/UD-IQ2_M/GLM-5.2-UD-IQ2_M-00001-of-00006.gguf -p 0 -n 128 -r 10 -o md -dev none -ngl 0 --no-op-offload 1 -o json -oe md
=================================
| model                          |       size |     params | backend    | threads |  fa | dev          | mmap | nopo |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | ------------ | ---: | ---: | --------------: | -------------------: |
[iqk] ACTIVE: ik_llama GEMM kernels engaged (first mul_mat type=13 ne00=6144)
| glm-dsa 744B.A40B IQ2_M - 2.7 bpw | 222.18 GiB |   753.86 B | CPU        |      96 |   1 | none         |    0 |    1 |           tg128 |          4.14 ± 0.00 |

build: 6ad45fa3f (10098)
