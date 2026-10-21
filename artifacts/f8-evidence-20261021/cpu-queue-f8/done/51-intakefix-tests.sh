#!/bin/bash
#maxjob_s=900
# Unit/oracle tests for the four intake-fix commits. Tiny synthetic fixtures; NO model weights loaded.
# DS41-B3.1: the gate is covered by a small-tensor graph test; the full T1-smoke (decode logits bit-identical
# on the real DS41 model) is NOT run here because it needs the 80+ GB model and a timing-grade host.
set -euo pipefail
JOB=51-intakefix-tests
B=/mnt/raid0/llm/tmp/intake-fixes-20261020/build-cpu/bin
OUT=/mnt/raid0/llm/tmp/intake-fixes-20261020/queue-results/$JOB
mkdir -p "$OUT"
for t in test-spec-rng-poskey test-dsv41-engram-gate test-moe-fused-upgate test-dsa-sparse-attn-oracle test-iqk-ser test-sampling; do
    [ -x "$B/$t" ] || { echo "DEP_MISSING $B/$t"; exit 3; }
done
export LD_LIBRARY_PATH="$B"
pass=0; fail=0
ok()  { echo "PASS $1"; pass=$((pass+1)); }
bad() { echo "FAIL $1"; fail=$((fail+1)); }
run() { # run <name> <cmd...>  (env prefix via `env`)
    local n=$1; shift
    if "$@" > "$OUT/$n.out" 2> "$OUT/$n.err"; then ok "$n"; else bad "$n (rc=$?, see $OUT/$n.err)"; fi
}

# DSA-IDX.1: sparse reference == dense masked oracle, bit-for-bit
run dsa-idx1-oracle "$B/test-dsa-sparse-attn-oracle"
# SPEC-RNG.1: np1==np4, depth0==depth3 at sampler level (test sets LLAMA_RNG_POSKEY=1 itself); default path unchanged
run spec-rng1-poskey "$B/test-spec-rng-poskey"
run spec-rng1-default-sampling-regression "$B/test-sampling"
# DS41-B3.1: exact gate at +0.0, bit-identical elsewhere
run ds41-b31-gate "$B/test-dsv41-engram-gate"
# CH-19: fused MoE up+gate+SwiGLU. ref = unfused (flag 0), then flag 1 must engage (stderr marker) and match within NMSE 1e-5.
export GGML_IQK=1 GGML_IQK_Q8_0=1
REF="$OUT/ch19-ref.bin"
run ch19-write-unfused env GGML_CPU_FUSED_UPGATE=0 "$B/test-moe-fused-upgate" write "$REF"
if grep -q "fused MoE up+gate" "$OUT/ch19-write-unfused.err"; then bad "ch19-flag0-must-not-fuse"; else ok "ch19-flag0-must-not-fuse"; fi
run ch19-cmp-fused env GGML_CPU_FUSED_UPGATE=1 "$B/test-moe-fused-upgate" cmp "$REF"
if grep -q "fused MoE up+gate" "$OUT/ch19-cmp-fused.err"; then ok "ch19-fused-path-engaged"; else bad "ch19-fused-path-engaged (marker missing: fusion fell through, NMSE above is then vacuous)"; fi
run ch19-ctest-smoke env GGML_CPU_FUSED_UPGATE=1 "$B/test-moe-fused-upgate"
# regression: existing iqk MoE SER test with the new flag on and off
run ch19-iqk-ser-flag0 env GGML_CPU_FUSED_UPGATE=0 "$B/test-iqk-ser"
run ch19-iqk-ser-flag1 env GGML_CPU_FUSED_UPGATE=1 "$B/test-iqk-ser"
cat "$OUT"/*.out | grep -E "^RESULT|nmse=" || true
echo "RESULT $JOB pass=$pass fail=$fail"
[ "$fail" -eq 0 ]
