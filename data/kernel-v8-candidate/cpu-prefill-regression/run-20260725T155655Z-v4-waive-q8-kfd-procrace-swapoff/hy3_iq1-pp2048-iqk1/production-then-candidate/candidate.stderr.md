=== Validating canonical recipe ===
=== Canonical bench command ===
Binary:    /mnt/raid0/llm/llama.cpp-experimental/build-v8-cpu/bin/llama-bench
Env:       LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-v8-cpu/bin:/usr/lib/llvm-20/lib OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active OMP_DYNAMIC=false GGML_IQK=1
Cmd:       taskset -c 0-95 numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-v8-cpu/bin/llama-bench -t 96 -fa 1 -mmp 0 -m /mnt/raid0/llm/models/hy3-angelslim/Hy3-IQ1_M-mtp.gguf -p 2048 -n 0 -r 10 -o md -dev none -ngl 0 --no-op-offload 1 -o json -oe md
=================================
| model                          |       size |     params | backend    | threads |  fa | dev          | mmap | nopo |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | ------------ | ---: | ---: | --------------: | -------------------: |
[iqk] ACTIVE: ik_llama GEMM kernels engaged (first mul_mat type=13 activation=99 ne00=8192)
[iqk] ACTIVE: MoE mul_mat_id via ik kernels (type=18 activation=15 n_as=192)
[iqk] ACTIVE: ik_llama GEMM kernels engaged (first mul_mat type=14 activation=99 ne00=1536)
[iqk] ACTIVE: MoE mul_mat_id via ik kernels (type=16 activation=15 n_as=192)
| hy_v3 ?B IQ1_M - 1.75 bpw      |  85.45 GiB |   298.79 B | CPU        |      96 |   1 | none         |    0 |    1 |          pp2048 |        100.08 ± 0.24 |

build: 67a433bf4 (10107)
