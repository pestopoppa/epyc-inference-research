=== Validating canonical recipe ===
=== Canonical bench command ===
Binary:    /mnt/raid0/llm/llama.cpp/build/bin/llama-bench
Env:       LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp/build/bin:/usr/lib/llvm-20/lib OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active OMP_DYNAMIC=false GGML_IQK=0
Cmd:       taskset -c 0-95 numactl --interleave=all /mnt/raid0/llm/llama.cpp/build/bin/llama-bench -t 96 -fa 1 -mmp 0 -m /mnt/raid0/llm/models/hy3-angelslim/Hy3-IQ1_M-mtp.gguf -p 0 -n 128 -r 10 -o md -dev none -ngl 0 --no-op-offload 1 -o json -oe md
=================================
| model                          |       size |     params | backend    | threads |  fa | dev          | mmap | nopo |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | ------------ | ---: | ---: | --------------: | -------------------: |
| hy_v3 ?B IQ1_M - 1.75 bpw      |  85.45 GiB |   298.79 B | CPU        |      96 |   1 | none         |    0 |    1 |           tg128 |          7.91 ± 0.01 |

build: 6ad45fa3f (10098)
