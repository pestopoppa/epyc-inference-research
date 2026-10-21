#!/bin/bash
#maxjob_s=5400
# CPU build of experimental/intake-fixes-20261020 (SPEC-RNG.1, DS41-B3.1, CH-19, DSA-IDX.1 test targets).
# Recipe = native-openmp-gcc15-cpu-v1 (same as the champion store build) in a scratch dir, NOT the kernel store.
set -euo pipefail
JOB=50-intakefix-build
SRC=/mnt/raid0/llm/worktrees/intake-fixes-20261020
B=/mnt/raid0/llm/tmp/intake-fixes-20261020/build-cpu
OUT=/mnt/raid0/llm/tmp/intake-fixes-20261020/queue-results/$JOB
mkdir -p "$OUT" "$B"
[ -d "$SRC/.git" ] || [ -f "$SRC/.git" ] || { echo "DEP_MISSING $SRC"; exit 3; }
[ -z "$(git -C "$SRC" status --porcelain --untracked-files=no)" ] || { echo "FAIL: $SRC dirty"; echo "RESULT $JOB pass=0 fail=1"; exit 1; }
echo "src=$(git -C "$SRC" rev-parse HEAD) start=$(date -u +%FT%TZ)" | tee "$OUT/build.log"
cmake -S "$SRC" -B "$B" -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=OFF -DGGML_NATIVE=ON -DGGML_OPENMP=ON \
    -DCMAKE_C_COMPILER=/usr/bin/gcc-15 -DCMAKE_CXX_COMPILER=/usr/bin/g++-15 -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF \
    -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON '-DCMAKE_INSTALL_RPATH=$ORIGIN' \
    >> "$OUT/build.log" 2>&1
rc=0
cmake --build "$B" -j "$(nproc)" --target \
    test-spec-rng-poskey test-dsv41-engram-gate test-moe-fused-upgate test-dsa-sparse-attn-oracle test-iqk-ser test-sampling \
    >> "$OUT/build.log" 2>&1 || rc=$?
echo "end=$(date -u +%FT%TZ) rc=$rc" | tee -a "$OUT/build.log"
n=0; for t in test-spec-rng-poskey test-dsv41-engram-gate test-moe-fused-upgate test-dsa-sparse-attn-oracle test-iqk-ser test-sampling; do
    [ -x "$B/bin/$t" ] && n=$((n+1)) || echo "MISSING binary $t" | tee -a "$OUT/build.log"; done
echo "RESULT $JOB pass=$n fail=$((6-n))"
[ "$rc" -eq 0 ] && [ "$n" -eq 6 ]
