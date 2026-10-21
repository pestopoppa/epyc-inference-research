#!/bin/bash
# DS41-W5a part 2: build + run the CPU_REPACK MXFP4 MUL_MAT_ID NMSE harness against the port build used by
# W5/W6 (llama.cpp-experimental-ds41-20261005 @ 3376147f8, build-cpu). No llama-server; ~0.5 GB RAM.
# usage: run_w5a_kernel.sh <nthreads>    (run inside the slot's build-role region-lock claim, under taskset)
set -euo pipefail
NT=${1:-48}
W=/mnt/raid0/llm/tmp/ds41-w5ab-w62-20261021/w5a
T=/mnt/raid0/llm/llama.cpp-experimental-ds41-20261005
BIN=$T/build-cpu/bin
OUT=$W/kernel
mkdir -p "$OUT"
D=/mnt/raid0/llm/models/antirez/deepseek-v4.1-flash-gguf
CASES=(
  "blk.3.ffn_gate_exps.weight:$D/DeepSeek-V4.1-Flash-1017-MXFP4-experts-00001-of-00009.gguf:24730394336:5120:2304:384:1"
  "blk.3.ffn_down_exps.weight:$D/DeepSeek-V4.1-Flash-1017-MXFP4-experts-00001-of-00009.gguf:29568425696:2304:5120:384:0"
  "blk.20.ffn_up_exps.weight:$D/DeepSeek-V4.1-Flash-1017-MXFP4-experts-00004-of-00009.gguf:4998643936:5120:2304:384:1"
  "blk.39.ffn_down_exps.weight:$D/DeepSeek-V4.1-Flash-1017-MXFP4-experts-00007-of-00009.gguf:192:2304:5120:384:0"
)
{
  echo "tree_head=$(git -C "$T" rev-parse HEAD)"
  echo "tree_dirty=$(git -C "$T" status --porcelain --untracked-files=no | wc -l)"
  sha256sum "$BIN/libggml-cpu.so" "$BIN/libggml-base.so" "$BIN/libggml.so"
  echo "libggml-cpu mtime=$(stat -c %y "$BIN/libggml-cpu.so")"
  echo "mxfp4_8x8 symbols=$(strings "$BIN/libggml-cpu.so" | grep -c mxfp4_8x8 || true)"
} > "$OUT/provenance.txt"
g++ -O3 -march=native -fopenmp -std=c++17 -I"$T/ggml/include" "$W/mxfp4_repack_nmse.cpp" \
    -L"$BIN" -lggml -lggml-base -lggml-cpu -Wl,-rpath,"$BIN" -o "$OUT/mxfp4_repack_nmse" 2> "$OUT/build.log"
export LD_LIBRARY_PATH=$BIN GGML_IQK=1 OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active OMP_DYNAMIC=false
hrc=0
"$OUT/mxfp4_repack_nmse" "$NT" "$OUT/w5a_kernel.jsonl" "${CASES[@]}" 2> "$OUT/run.log" || hrc=$?
# Re-run the weight-fidelity comparison inside the claim so its belief row carries an honest region
# (the 2026-10-21 run held no claim: the host was fully claimed by kvu12-redo). ~50 MB HTTP range reads.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 timeout 600 /mnt/raid0/llm/epyc-inference-research/.venv/bin/python -I \
  "$W/weight_fidelity.py" "$W/index.json" "$W/gguf_tensors.json" "$OUT/weight_fidelity_slot.json" > "$OUT/weight_fidelity_slot.log" 2>&1 \
  || echo "weight_fidelity slot rerun failed rc=$? (non-fatal; the 2026-10-21 result stands)" >> "$OUT/run.log"
exit "$hrc"
