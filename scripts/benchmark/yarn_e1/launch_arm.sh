#!/bin/bash
# INF-59 E1: launch ONE scratch single-slot Qwen3.8-27B instance on :18083 for arm A0 | A1 | A2.
#   A0  native RoPE, -c 262144                       (paired anchor)
#   A1  YaRN f2 to 524288                            (candidate)
#   A2  context_length override 524288, NO rope flags (negative control: raw extrapolation)
# Base = the production :8083 environment and argv (logs/server_launches/8083.json, argv b865820d) with:
# -np 1, no drafter, --cache-ram 0 (no host prompt cache, so no idle-slot purge and no stale-geometry
# entries), a scratch --slot-save-path, --port 18083. Binary = the v10 production GPU KERNEL STORE.
# PRECONDITION: production :8083 is STOPPED (a parked :8083 still holds ~59 GiB of VRAM).
set -euo pipefail
ARM="${1:?usage: launch_arm.sh A0|A0Q4|A1|A2}"
D=${YARN_E1_HOME:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
STORE=/mnt/raid0/llm/kernels/production/gpu
EXPECT_BIN_DIR=/mnt/raid0/llm/kernels/builds/gpu-20260921-ffc1bac82/bin
MODEL=/mnt/raid0/llm/models/Qwen3.8-27B-Q8_0.gguf
TEMPLATE=/mnt/raid0/llm/models/chat-templates/epyc-qwen3x-v1-terse.jinja
PORT=18083
LOG=$D/results/$ARM.server.log
PIDF=$D/results/$ARM.server.pid
mkdir -p "$D/results/kv_slots_scratch"

BIN_DIR=$(readlink -f "$STORE")
if [ "$BIN_DIR" != "$EXPECT_BIN_DIR" ]; then
  echo "REFUSING: $STORE -> $BIN_DIR, expected $EXPECT_BIN_DIR (v10 ffc1bac82). Re-derive before running."; exit 2
fi
BIN=$BIN_DIR/llama-server
"$BIN" --version 2>&1 | grep -q 'version: 10303' || { echo "REFUSING: $BIN is not build 10303"; exit 2; }
if ss -ltn 'sport = :8083' | grep -q LISTEN; then echo "REFUSING: :8083 still listening (stop architect_critic first)"; exit 2; fi
if ss -ltn "sport = :$PORT" | grep -q LISTEN; then echo "REFUSING: :$PORT already in use"; exit 2; fi
echo "KFD processes now: $(ls /sys/class/kfd/kfd/proc/ | tr '\n' ' ')  card VRAM used: $(( $(cat /sys/class/drm/card2/device/mem_info_vram_used) / 1048576 )) MiB"

PROOF_FLAGS_TEXT=$(python3 "$D/yarn_proof.py" --log-flags)
mapfile -t PROOF_FLAGS <<< "$PROOF_FLAGS_TEXT"
COMMON=(-m "$MODEL" --host 127.0.0.1 --port "$PORT" -np 1 -t 8 -ub 2048 --flash-attn on --jinja
        -ctk q8_0 -ctv q8_0 --no-mmap -ngl all --cache-ram 0 --chat-template-file "$TEMPLATE"
        --device ROCm0 --slot-save-path "$D/results/kv_slots_scratch" "${PROOF_FLAGS[@]}")   # -lv 4: print_info/llama_context proof lines
case "$ARM" in
  A0) EXTRA=(-c 262144) ;;
  # 1M option (a) pre-test: q4_0 K+V (later -ctk/-ctv win over the COMMON q8_0); native RoPE
  A0Q4) EXTRA=(-c 262144 -ctk q4_0 -ctv q4_0) ;;
  # The operator-specified CLI flags, PLUS the same YaRN parameters as GGUF metadata overrides. The
  # overrides make print_info show `rope scaling = yarn`, `freq_scale_train = 0.5`,
  # `n_ctx_orig_yarn = 262144` (with CLI flags alone it prints `linear`, `1`, `524288`: source_findings
  # H-7). Both routes set identical values (freq_scale 0.5, orig 262144, ext_factor 1, attn_factor 1).
  A1) EXTRA=(-c 524288 --rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144
             --override-kv "qwen35.context_length=int:524288,qwen35.rope.scaling.type=str:yarn,qwen35.rope.scaling.factor=float:2,qwen35.rope.scaling.original_context_length=int:262144") ;;
  A2) EXTRA=(-c 524288 --override-kv "qwen35.context_length=int:524288") ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac

echo "=== launch $(date -u +%FT%TZ) arm $ARM" >> "$LOG"
nohup env -u LLAMA_SERVER_SLOTS_DEBUG -u GGML_CUDA_GDN_STATE_BF16 -u GGML_CUDA_FA_MASK_SKIP \
  LD_LIBRARY_PATH="$BIN_DIR:/opt/rocm/lib:/usr/lib/llvm-20/lib:/opt/AMD/aocc-compiler-5.0.0/lib:/opt/rocm/lib" \
  HIP_PATH=/opt/rocm ROCM_PATH=/opt/rocm OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active \
  OMP_DYNAMIC=false GGML_IQK=1 \
  numactl --membind=3 -- taskset -c 184-191 "$BIN" "${COMMON[@]}" "${EXTRA[@]}" >> "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$PIDF"
echo "scratch $ARM PID $PID (log $LOG)"

# 1 Hz KFD sampler from load onward; it exits by itself when the server PID dies.
nohup /usr/bin/python3 "$D/kfd_sampler.py" --pid "$PID" --out "$D/results/kfd_$ARM.jsonl" \
  > "$D/results/kfd_$ARM.sampler.log" 2>&1 &
echo "$!" > "$D/results/kfd_$ARM.sampler.pid"

for i in $(seq 1 300); do       # 27 GB --no-mmap load: allow up to 10 min
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
  if ! ps -p "$PID" >/dev/null; then echo "server PID $PID died during load; tail of log:"; tail -30 "$LOG"; exit 1; fi
  sleep 2
done
curl -sf "http://127.0.0.1:$PORT/health" && echo
echo "--- proof lines (this launch only)"
awk -v m="=== launch" 'index($0,m)==1{buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}' "$LOG" | grep -E \
  "Using metadata override|n_ctx_train|rope scaling|freq_scale_train|n_ctx_orig_yarn|llama_context: (n_ctx|n_ctx_seq|freq_scale|freq_base|kv_unified) |llama_kv_cache: size|n_ctx_slot|capping|training context overflow|attn_rot" || true
echo "--- residency / linkage (read-only)"
tr '\0' ' ' < "/proc/$PID/cmdline"; echo
grep -o '/[^ ]*libggml[^ ]*\.so[^ ]*' "/proc/$PID/maps" | sort -u || echo 'no libggml in maps yet'
KFD=/sys/class/kfd/kfd/proc/$PID
if [ -d "$KFD" ]; then echo "KFD vram MiB: $(( $(cat $KFD/vram_* | paste -sd+ | bc) / 1048576 ))"; else echo "NOT a KFD process (no GPU residency!)"; exit 1; fi
LD_LIBRARY_PATH="$BIN_DIR:/opt/rocm/lib:/usr/lib/llvm-20/lib:/opt/AMD/aocc-compiler-5.0.0/lib:/opt/rocm/lib" \
  /mnt/raid0/llm/epyc-inference-research/scripts/utils/verify_ggml_linkage.sh "$BIN" "$(dirname "$BIN_DIR")" \
  || echo "LINKAGE CHECK REPORTED A PROBLEM: do not run the arm"
