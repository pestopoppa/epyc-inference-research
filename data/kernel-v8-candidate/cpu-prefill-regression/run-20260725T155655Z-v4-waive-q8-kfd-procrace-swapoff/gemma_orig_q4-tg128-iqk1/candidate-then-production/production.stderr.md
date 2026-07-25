=== Validating canonical recipe ===
=== Canonical bench command ===
Binary:    /mnt/raid0/llm/llama.cpp/build/bin/llama-bench
Env:       LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp/build/bin:/usr/lib/llvm-20/lib OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active OMP_DYNAMIC=false GGML_IQK=1
Cmd:       taskset -c 0-95 numactl --interleave=all /mnt/raid0/llm/llama.cpp/build/bin/llama-bench -t 96 -fa 1 -mmp 0 -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf -p 0 -n 128 -r 10 -o md -dev none -ngl 0 --no-op-offload 1 -o json -oe md
=================================
| model                          |       size |     params | backend    | threads |  fa | dev          | mmap | nopo |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | ------------ | ---: | ---: | --------------: | -------------------: |
[iqk] ACTIVE: ik_llama GEMM kernels engaged (first mul_mat type=12 ne00=2816)
[iqk] ACTIVE: MoE mul_mat_id via ik kernels (type=12 n_as=128)
| gemma4 26B.A4B Q4_K - Medium   |  15.63 GiB |    25.23 B | CPU        |      96 |   1 | none         |    0 |    1 |           tg128 |         27.49 ± 0.04 |

build: 6ad45fa3f (10098)
