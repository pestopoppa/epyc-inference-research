=== Validating canonical recipe ===
=== Canonical bench command ===
Binary:    /mnt/raid0/llm/llama.cpp-experimental/build-v8-cpu/bin/llama-bench
Env:       LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-v8-cpu/bin:/usr/lib/llvm-20/lib OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active OMP_DYNAMIC=false GGML_IQK=1
Cmd:       taskset -c 0-95 numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-v8-cpu/bin/llama-bench -t 96 -fa 1 -mmp 0 -m /mnt/raid0/llm/models/Qwen3-Next-80B-A3B-Instruct.i1-IQ2_M.gguf -p 2048 -n 0 -r 10 -o md -dev none -ngl 0 --no-op-offload 1 -o json -oe md
=================================
| model                          |       size |     params | backend    | threads |  fa | dev          | mmap | nopo |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | ------------ | ---: | ---: | --------------: | -------------------: |
[iqk] ACTIVE: ik_llama GEMM kernels engaged (first mul_mat type=22 activation=15 ne00=2048)
[iqk] ACTIVE: MoE mul_mat_id via ik kernels (type=22 activation=15 n_as=512)
[iqk] ACTIVE: MoE mul_mat_id via ik kernels (type=21 activation=15 n_as=512)
[iqk] ACTIVE: ik_llama GEMM kernels engaged (first mul_mat type=21 activation=15 ne00=512)
[iqk] ACTIVE: ik_llama GEMM kernels engaged (first mul_mat type=12 activation=99 ne00=2048)
[iqk] ACTIVE: ik_llama GEMM kernels engaged (first mul_mat type=13 activation=99 ne00=2048)
| qwen3next 80B.A3B IQ2_M - 2.7 bpw |  24.26 GiB |    79.67 B | CPU        |      96 |   1 | none         |    0 |    1 |          pp2048 |        182.60 ± 0.86 |

build: 67a433bf4 (10107)
