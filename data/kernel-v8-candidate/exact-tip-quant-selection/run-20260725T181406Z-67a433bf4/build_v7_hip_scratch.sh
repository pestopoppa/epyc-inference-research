#!/bin/bash
set -euo pipefail

readonly ROOT="/mnt/raid0/llm/epyc-inference-research/data/kernel-v8-candidate/exact-tip-quant-selection/run-20260725T181406Z-67a433bf4"
readonly V7_REPO="/mnt/raid0/llm/llama.cpp"
readonly V7_HEAD="6ad45fa3ff6718c07c000061dbc6e29c1771f6e3"
readonly SCRATCH="/mnt/raid0/llm/tmp/v8-exact-tip-v7-hip-quant-selection-20260725T181406Z"
readonly BUILD="$SCRATCH/build-hip-quant-selection"
readonly OUT="$ROOT/v7-hip-scratch"
readonly CLEAN_PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/rocm/bin"

mkdir -p "$OUT"
printf '%s\n' "$V7_HEAD" > "$OUT/requested_source_head.txt"
printf '%s\n' "$SCRATCH" > "$OUT/scratch_worktree_path.txt"
git -C "$V7_REPO" worktree add --detach "$SCRATCH" "$V7_HEAD" > "$OUT/worktree_add.stdout.txt" 2> "$OUT/worktree_add.stderr.txt"
cleanup() {
    git -C "$V7_REPO" worktree remove --force "$SCRATCH" > "$OUT/worktree_remove.stdout.txt" 2> "$OUT/worktree_remove.stderr.txt" || true
}
trap cleanup EXIT

git -C "$SCRATCH" rev-parse HEAD > "$OUT/source_head.txt"
cmake -S "$SCRATCH" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS=gfx90a \
    -DGGML_NATIVE=ON \
    -DGGML_HIP_ROCWMMA_FATTN=ON \
    -DLLAMA_BUILD_TESTS=ON \
    -DBUILD_TESTING=ON \
    > "$OUT/configure.stdout.txt" 2> "$OUT/configure.stderr.txt"
cp "$BUILD/CMakeCache.txt" "$OUT/CMakeCache.txt"
cmake --build "$BUILD" --target test-quant-type-selection --parallel 32 > "$OUT/build.stdout.txt" 2> "$OUT/build.stderr.txt"
readonly BINARY="$BUILD/bin/test-quant-type-selection"
sha256sum "$BINARY" > "$OUT/binary.sha256"
printf '%q ' env -i "PATH=$CLEAN_PATH" LANG=C LC_ALL=C "$BINARY" > "$OUT/command.txt"
printf '\n' >> "$OUT/command.txt"
set +e
env -i "PATH=$CLEAN_PATH" LANG=C LC_ALL=C "$BINARY" > "$OUT/stdout.txt" 2> "$OUT/stderr.txt"
status=$?
set -e
printf '%s\n' "$status" > "$OUT/exit_code.txt"
