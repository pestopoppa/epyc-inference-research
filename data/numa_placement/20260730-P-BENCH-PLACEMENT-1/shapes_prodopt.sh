#!/bin/bash
# Shape x concurrency, every role on its PRODUCTION acceleration recipe.
# Replaces the earlier sweep, which used spec-dec-off baselines for three of the
# four roles and omitted the half shape for three of them.
#
# Recipes (from orchestration/model_registry.yaml):
#   qwen36_q8_0        draft-mtp self-draft, n_max 4
#   worker_general     draft-mtp + separate draft assistant-v6-Q8_0, n_max 2,
#                      threads-draft 16, ub 512, p-min 0.0
#   qwen35_122b_q4km   draft-mtp self-draft, n_max 4
#   ingest_long_context acceleration {type: none} -- off IS its recipe
#
# Shapes: full = 0-95 interleave=all; half = 2 instances, each 2 NPS4 nodes,
# interleaved over its own pair; quarter = 4 instances, one node each, membind.
# All --no-mmap so every instance owns node-local weights.
LS=/mnt/raid0/llm/llama.cpp/build/bin/llama-server
OUT=/mnt/raid0/llm/tmp/shapes_prodopt_results.txt
export GGML_IQK=1 OMP_DYNAMIC=false OMP_PLACES=cores OMP_PROC_BIND=spread OMP_WAIT_POLICY=active KMP_BLOCKTIME=10
: > "$OUT"

G=/mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf
GD=/mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf
Q122=/mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf
Q35=/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf
Q80=/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf

GEMMA_SPEC=(-md "$GD" --spec-type draft-mtp --spec-draft-n-max 2 --draft-p-min 0.0 --threads-draft 16)
SELF_SPEC=(--spec-type draft-mtp --spec-draft-n-max 4 --device-draft none)

sweep () { # label model ub shape np  [spec args...]
  local LABEL="$1" MODEL="$2" UB="$3" SHAPE="$4" NP="$5"; shift 5
  local -a CPUS=() POL=(); local T=0 TH=0
  case "$SHAPE" in
    full)    CPUS=("0-95");                                   POL=("--interleave=all"); TH=96 ;;
    half)    CPUS=("0-47,96-143" "48-95,144-191");            POL=("--interleave=0,1" "--interleave=2,3"); TH=48 ;;
    quarter) CPUS=("0-23,96-119" "24-47,120-143" "48-71,144-167" "72-95,168-191")
             POL=("--membind=0" "--membind=1" "--membind=2" "--membind=3"); TH=24 ;;
  esac
  local N=${#CPUS[@]}; T=$(( N * NP ))
  local CTX=$(( 4096 * NP )); [ "$CTX" -lt 8192 ] && CTX=8192
  echo "=== $LABEL | $SHAPE | inst=$N np=$NP T=$T -t $TH ===" | tee -a "$OUT"
  sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
  local -a SP=()
  for i in $(seq 0 $((N-1))); do
    local SL=/mnt/raid0/llm/tmp/sp_${LABEL}_${SHAPE}_np${NP}_i${i}.log; : > "$SL"
    taskset -c "${CPUS[$i]}" numactl ${POL[$i]} "$LS" -m "$MODEL" \
      --host 127.0.0.1 --port $((19930 + i)) -np "$NP" -c "$CTX" -t "$TH" -ub "$UB" \
      -ctk q8_0 -ctv q8_0 --flash-attn on --jinja --no-mmap --device none --log-colors off \
      "$@" >> "$SL" 2>&1 &
    SP[$i]=$!
  done
  local UP=0
  for t in $(seq 1 500); do
    UP=1; for i in $(seq 0 $((N-1))); do
      grep -q "model loaded" /mnt/raid0/llm/tmp/sp_${LABEL}_${SHAPE}_np${NP}_i${i}.log || UP=0; done
    [ "$UP" = "1" ] && break; sleep 5
  done
  if [ "$UP" != "1" ]; then echo "    NOT ALL LOADED" | tee -a "$OUT"
  else
    local RP=""
    for i in $(seq 0 $((N-1))); do for r in $(seq 1 "$NP"); do
      curl -s --max-time 1800 "http://127.0.0.1:$((19930 + i))/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d '{"messages":[{"role":"user","content":"Write a Python function returning the first n Fibonacci numbers, then explain it in three sentences."}],"max_tokens":256,"temperature":0.3,"seed":42}' \
        -o /dev/null 2>/dev/null & RP="$RP $!"
    done; done
    for p in $RP; do wait "$p"; done
    python3 - "$LABEL" "$SHAPE" "$NP" "$N" <<'PY' | tee -a "$OUT"
import re, sys, glob, statistics as st
lab, shape, np_, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
rates, acc = [], []
for i in range(n):
    t = open(f"/mnt/raid0/llm/tmp/sp_{lab}_{shape}_np{np_}_i{i}.log").read()
    rates += [float(m.group(1)) for l in t.splitlines()
              if "eval time =" in l and "prompt eval" not in l
              for m in [re.search(r"([\d.]+) tokens per second", l)] if m]
    acc += [float(x) for x in re.findall(r"draft acceptance\s*=\s*([\d.]+)", t)]
if rates:
    print(f"    per-stream median={st.median(rates):.2f}  aggregate={sum(rates):.2f} tok/s  n={len(rates)}"
          + (f"  accept={sum(acc)/len(acc):.3f}" if acc else "  spec=off"))
else:
    print("    NO TIMINGS")
PY
  fi
  for i in $(seq 0 $((N-1))); do kill -TERM "${SP[$i]}" 2>/dev/null; done; sleep 10
  for i in $(seq 0 $((N-1))); do kill -9 "${SP[$i]}" 2>/dev/null; done; sleep 4
}

for NP in 1 2 4; do sweep gemma "$G"    512  full    $NP "${GEMMA_SPEC[@]}"; done
for NP in 1 2 4; do sweep gemma "$G"    512  half    $NP "${GEMMA_SPEC[@]}"; done
for NP in 1 2;   do sweep gemma "$G"    512  quarter $NP "${GEMMA_SPEC[@]}"; done
for NP in 1 2 4; do sweep q35   "$Q35"  8192 full    $NP "${SELF_SPEC[@]}"; done
for NP in 1 2 4; do sweep q35   "$Q35"  8192 half    $NP "${SELF_SPEC[@]}"; done
for NP in 1 2;   do sweep q35   "$Q35"  8192 quarter $NP "${SELF_SPEC[@]}"; done
for NP in 1 2 4; do sweep q122  "$Q122" 8192 full    $NP "${SELF_SPEC[@]}"; done
for NP in 1 2;   do sweep q122  "$Q122" 8192 half    $NP "${SELF_SPEC[@]}"; done
for NP in 1 2 4; do sweep q80   "$Q80"  8192 half    $NP; done
echo "=== SHAPES_PRODOPT DONE ===" | tee -a "$OUT"
