=== Validating canonical recipe ===
=== Canonical bench command ===
Binary:    /mnt/raid0/llm/llama.cpp-experimental/build-v8-cpu/bin/llama-bench
Env:       LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-v8-cpu/bin:/usr/lib/llvm-20/lib OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active OMP_DYNAMIC=false GGML_IQK=0
Cmd:       taskset -c 0-95 numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-v8-cpu/bin/llama-bench -t 96 -fa 1 -mmp 0 -m /mnt/raid0/llm/models/Qwen3-Next-80B-A3B-Instruct.i1-IQ2_M.gguf -p 0 -n 128 -r 10 -o md -dev none -ngl 0 --no-op-offload 1 -o json -oe md
=================================
| model                          |       size |     params | backend    | threads |  fa | dev          | mmap | nopo |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | ------------ | ---: | ---: | --------------: | -------------------: |
| qwen3next 80B.A3B IQ2_M - 2.7 bpw |  24.26 GiB |    79.67 B | CPU        |      96 |   1 | none         |    0 |    1 |           tg128 |         22.86 ± 0.02 |

build: 67a433bf4 (10107)
