#!/bin/bash
# R21-10 determinism probe: build 14ba02627f56 twice, fresh dirs, identical inputs.
# Mirrors gates.compiles (loop/gates.py:42) but throttled to cores 8-63, -j24.
set -euo pipefail
SRC=/mnt/raid0/llm/tmp/r2110-src
DEFINES=(-DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx90a
         -DGGML_HIP_ROCWMMA_FATTN=ON -DGGML_NATIVE=ON)
for tag in a b; do
  B=/mnt/raid0/llm/tmp/r2110-build-$tag
  rm -rf "$B"
  taskset -c 8-63 cmake -S "$SRC" -B "$B" "${DEFINES[@]}" > "$B.configure.log" 2>&1
  taskset -c 8-63 cmake --build "$B" -j 24 --target llama-bench --target test-backend-ops \
      > "$B.build.log" 2>&1
  echo "build $tag done rc=$?"
done
echo ALL-DONE
