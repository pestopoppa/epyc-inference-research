#!/bin/bash
# MMQ J-cap GPU window (INF-06 AK-MMQ-H10): correctness, then timing A/B.
#   A = production v10 kernel store (ffc1bac82, build 10303)
#   B = experimental/mmq-jcap-20260927 (36e83c0f2, build 10306)
# RUNS ON THE MI210 AND USES HOST THREADS. Run it only in a CPU window: no CPU measurement
# may run at the same time (pinned GPU host threads still raise the CPU A/A floor).
# Preconditions, which the main session provides: the production GPU servers are drained,
# so the MI210 has at least 40 GB of free VRAM, and nobody else holds the device lock.
set -euo pipefail
[ "${MMQ_JCAP_WINDOW:-}" = 1 ] || { echo "refusing: set MMQ_JCAP_WINDOW=1 inside a CPU window" >&2; exit 2; }

A_BIN=/mnt/raid0/llm/kernels/production/gpu
B_BIN=/mnt/raid0/llm/llama.cpp-experimental-mmq-jcap-20260927/build-hip-jcap/bin
VERIFY=/mnt/raid0/llm/epyc-inference-research/scripts/utils/verify_ggml_linkage.sh
M27=/mnt/raid0/llm/models/Qwen3.8-27B-Q8_0.gguf                    # dense Q8_0, production GPU model
M35=/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf            # MoE Q8_0, 256 experts: always MMQ
MQ2=/mnt/raid0/llm/models/QuantFactory/Qwen2.5-Coder-1.5B-GGUF/Qwen2.5-Coder-1.5B.Q2_K.gguf  # Q2_K cap 64 -> 32
MQ4=/mnt/raid0/llm/models/Qwen3-Coder-Instruct-DRAFT-0.75B-32k-Q4_0.gguf  # Q4_0: J changes only where ne01 % 128 != 0 (near-null)
ROUNDS=${ROUNDS:-5}
CPUS=184-191   # the codified MI210 host-thread mask (SMT siblings)
OUT=/mnt/raid0/llm/tmp/mmq-jcap-ab-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
exec 9>/mnt/raid0/llm/tmp/gpu_device.mi210_0.lock
flock -n 9 || { echo "MI210 device lock is held; not running" >&2; exit 3; }

gpu() {  # gpu <libdir> <cmd...>: same env for both arms; only the library directory differs
  local lib=$1; shift
  env -u HSA_OVERRIDE_GFX_VERSION LD_LIBRARY_PATH="$lib:/opt/rocm/lib" GGML_IQK=1 \
      ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
      taskset -c "$CPUS" "$@"
}
smi() { rocm-smi --showclocks --showpower --showtemp --showmeminfo vram > "$OUT/smi.$1.txt" 2>&1 || true; }

# 0. Linkage receipts (P-GPU-1). Arm A's test harness is B's binary with A's libraries.
env -u LD_LIBRARY_PATH bash "$VERIFY" "$B_BIN/llama-bench" > "$OUT/linkage.B.llama-bench.txt" 2>&1
env -u LD_LIBRARY_PATH bash "$VERIFY" "$A_BIN/llama-bench" > "$OUT/linkage.A.llama-bench.txt" 2>&1
LD_LIBRARY_PATH="$A_BIN:/opt/rocm/lib" bash "$VERIFY" "$B_BIN/test-backend-ops" "$A_BIN" > "$OUT/linkage.A.tbo.txt" 2>&1
sha256sum "$A_BIN/libggml-hip.so.0.16.0" "$B_BIN/libggml-hip.so.0.16.0" > "$OUT/libggml-hip.sha256"

# 1. Correctness. MUL_MAT and MUL_MAT_ID on ROCm0 against the CPU reference, including the
#    new J-window cases. Arm A is the control: a failure that shows up in both arms is
#    not the cap's. Any B failure stops the window, because timing a wrong kernel is pointless.
TYPES='q8_0|q5_0|q4_0|q1_0|q2_K|mxfp4|iq4_nl|iq4_xs|iq2_xxs|iq3_xxs|iq3_s'
smi pre-correctness
gpu "$A_BIN" "$B_BIN/test-backend-ops" test -o MUL_MAT,MUL_MAT_ID -b ROCm0 > "$OUT/tbo_test.A.log" 2>&1 || true
gpu "$B_BIN" "$B_BIN/test-backend-ops" test -o MUL_MAT,MUL_MAT_ID -b ROCm0 > "$OUT/tbo_test.B.log" 2>&1 || { echo "B correctness FAIL: see $OUT/tbo_test.B.log" >&2; exit 4; }
grep -q "Device 0: AMD Instinct MI210\|ROCm0" "$OUT/tbo_test.B.log" || { echo "no MI210 in the B log: not a HIP run" >&2; exit 5; }
smi post-correctness

# 2. Kernel-level timing ABAB: the only arm that covers IQ4_XS, IQ4_NL, MXFP4, Q5_0 and Q1_0.
#    No such model is on disk. Same harness binary; only the kernel library differs.
PERF_RE="type_a=($TYPES),type_b=f32,(n_mats=32,.*m=1792,n=(384|768)|m=(4096|4000),n=(48|64|96|128)),k="
for r in $(seq 1 "$ROUNDS"); do
  for arm in A B; do
    lib=$A_BIN; [ "$arm" = B ] && lib=$B_BIN
    smi "perf.r$r.$arm.pre"
    gpu "$lib" "$B_BIN/test-backend-ops" perf -o MUL_MAT,MUL_MAT_ID -b ROCm0 -p "$PERF_RE" --output csv \
        > "$OUT/tbo_perf.r$r.$arm.csv" 2> "$OUT/tbo_perf.r$r.$arm.err"
  done
done

# 3. Model-level timing ABAB with llama-bench, using the codified GPU builder shape
#    (recipes.py _build_llama_bench gpu=True). -ub/-b 2048 match production, so ne11 = p.
#    Dense Q8_0 p: 5 and 48 and 96 use the same J in both arms (null controls); 64, 112 and 128
#    are affected (v10 J=64 with spills, cap J=32x2 / J=48x3); 256 goes to rocBLAS (control).
bench() {  # bench <arm> <round> <tag> <model> <p-list>
  local lib=$A_BIN; [ "$1" = B ] && lib=$B_BIN
  smi "bench.$3.r$2.$1.pre"
  gpu "$lib" "$lib/llama-bench" -m "$4" -t 8 -fa 1 -mmp 0 -ngl 99 -dev ROCm0 \
      -p "$5" -n 0 -ub 2048 -b 2048 -r 5 --autokernel-harden 42 -o jsonl \
      > "$OUT/bench.$3.r$2.$1.jsonl" 2> "$OUT/bench.$3.r$2.$1.err"
}
for r in $(seq 1 "$ROUNDS"); do
  for arm in A B; do bench $arm $r q8dense "$M27" 5,48,64,96,112,128,256; done
  for arm in A B; do bench $arm $r q8moe   "$M35" 5,64,128,512,2048; done
  for arm in A B; do bench $arm $r q2k     "$MQ2" 32,48,64,128; done
  for arm in A B; do bench $arm $r q4null  "$MQ4" 64,128,512; done
done

# 4. Speed+correctness pairing: a greedy completion per arm for each model, with a prompt of
#    about 100 tokens so that prefill lands in the changed J window. Byte-identical output
#    is the expectation, but not a guarantee: a different tile count changes the stream-k
#    fixup summation order. Report the first differing token for any diff; never aggregate.
PROMPT="$OUT/prompt.txt"
cat > "$PROMPT" <<'TXT'
Below is a short technical note. Read it and then continue the numbered list with the next three items, one sentence each.
The MI210 accelerator has 104 compute units, 64 GB of HBM2e memory and a unified register file of 512 vector registers per lane per SIMD.
1. A matrix multiplication kernel stages tiles of both operands in shared memory before issuing matrix-core instructions.
2. When a kernel needs more registers than the hardware provides, the compiler spills values to private scratch memory.
3.
TXT
for m in "$M27" "$M35"; do
  tag=$(basename "$m" .gguf)
  for arm in A B; do
    lib=$A_BIN; [ "$arm" = B ] && lib=$B_BIN
    gpu "$lib" "$lib/llama-completion" -m "$m" -f "$PROMPT" -n 64 --temp 0 -s 42 -no-cnv \
        --no-display-prompt -t 8 -ngl 99 -dev ROCm0 -fa on -c 4096 -ub 2048 -b 2048 \
        -ctk f16 -ctv f16 > "$OUT/pair.$tag.$arm.txt" 2> "$OUT/pair.$tag.$arm.err"
  done
  if cmp -s "$OUT/pair.$tag.A.txt" "$OUT/pair.$tag.B.txt"; then echo "PAIR $tag: byte-identical";
  else echo "PAIR $tag: DIFFERS (first differing byte: $(cmp "$OUT/pair.$tag.A.txt" "$OUT/pair.$tag.B.txt" | head -1))"; fi
done | tee "$OUT/pairing.txt"
echo "done: $OUT"
