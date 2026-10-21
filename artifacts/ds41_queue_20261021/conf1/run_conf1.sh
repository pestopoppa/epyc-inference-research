#!/bin/bash
# DS41-CONF.1 INNER (inside the launcher's build-role claim, own process group).
# Phase 0 (once): if the logging build is missing, BUILD it from the worktree (branch experimental/ds41-conf1-logging-20261021,
#   commit b428d7392 on champion dd3d9721c) inside this claim, then prove the knob is compiled in (strings) -- a null from a
#   knob that is not in the binary is not evidence. The patch was authored WITHOUT a compile proof; a build failure exits 6.
# Phase 1 MAIN: ONE 1017 target + 1017 DSpark drafter server, recipe block width 2, LLAMA_DSPARK_CONF_LOG=main.jsonl, prompts A.
# Phase 2 HOLDOUT (a later, separate server process, prompts B, holdout.jsonl). Sequential, one DS41 server at a time.
# Then conf_reliability.py. Resumable per phase (DONE markers). COMPLETE when main+holdout done and >=500 rounds in each.
# Env: FILLER_END FILLER_CPUS FILLER_THREADS RUNDIR STOPFILE
set -uo pipefail
HERE=$(dirname "$(readlink -f "$0")")
LIB=${LIB:-$HERE/../lib}
. "$LIB/ds41_guard.sh"; . "$LIB/srv.sh"
WT=/mnt/raid0/llm/worktrees/ds41-conf1-20261021
BIN=$WT/build-conf1/bin
M1017=/mnt/raid0/llm/models/antirez/deepseek-v4.1-flash-gguf/DeepSeek-V4.1-Flash-1017-MXFP4-experts-00001-of-00009.gguf
DRAFT=/mnt/raid0/llm/models/deepseek-ai/DeepSeek-V4.1-Flash-1017-DSpark.gguf
PY=${PY:-/mnt/raid0/llm/epyc-inference-research/.venv/bin/python}
PORT=18995
NT=${FILLER_THREADS:?}; CPUS=${FILLER_CPUS:?}
R=$RUNDIR/results; mkdir -p "$R"
log() { echo "$(date -u +%FT%TZ) [conf1] $*"; }
trap 'kill_srv' EXIT
trap 'kill_srv; exit 143' TERM INT

# ---- phase 0: build
if [ ! -x "$BIN/llama-server" ]; then
  [ "$(time_left)" -lt 2400 ] && { log "SKIP build: $(time_left)s left < 2400"; exit 3; }
  [ "$(git -C "$WT" rev-parse --short HEAD)" = b428d7392 ] || { echo "worktree HEAD is not the CONF.1 patch commit"; exit 6; }
  log "building conf1 binary (-j $NT)"
  export CC=${CC:-gcc} CXX=${CXX:-g++}
  ( cd "$WT" && cmake -S . -B build-conf1 -DCMAKE_BUILD_TYPE=Release -DGGML_CPU_PROF=OFF -DGGML_IQK=ON -DGGML_NATIVE=ON -DLLAMA_CURL=OFF \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_INSTALL_RPATH='$ORIGIN' > "$R/build_configure.log" 2>&1 \
    && taskset -c "$CPUS" cmake --build build-conf1 -j "$NT" --target llama-server > "$R/build.log" 2>&1 ) || { log "BUILD FAILED (see $R/build.log)"; exit 6; }
fi
check_bin "$BIN" || exit 6
strings "$BIN"/libllama*.so* "$BIN/llama-server" 2>/dev/null | grep -q LLAMA_DSPARK_CONF_LOG || { echo "knob LLAMA_DSPARK_CONF_LOG not compiled into $BIN"; exit 6; }
log "binary ok: $(git -C "$WT" rev-parse --short HEAD) knob present"

# ---- prompts: A = the 16 W6 prompts (main); B = 16 different prompts (holdout, later window)
PA=$R/prompts_A.json; PB=$R/prompts_B.json
"$PY" -I - "$PA" "$PB" <<'PYEOF'
import json, sys
A = ["The history of the printing press begins in the fifteenth century, when", "def quicksort(arr):\n    \"\"\"Sort a list using quicksort.\"\"\"\n",
 "Photosynthesis is the process by which plants convert light energy into chemical energy. The process has two main stages:", "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,",
 "Q: A train travels 120 km in 1.5 hours. What is its average speed?\nA: Let's think step by step.", "| Country | Capital | Population (millions) |\n|---|---|---|\n| France | Paris | 68 |\n",
 "The French Revolution was a period of political and societal change in France that began in 1789 and", "import numpy as np\n\ndef softmax(x):\n",
 "Dear hiring manager,\n\nI am writing to apply for the position of", "In mathematics, a prime number is a natural number greater than 1 that",
 "The recipe for a classic margherita pizza: first, prepare the dough by", "SELECT customer_id, COUNT(*) AS orders\nFROM orders\n",
 "Once upon a time, in a small village at the edge of a great forest,", "The main differences between TCP and UDP are", "Monday, Tuesday, Wednesday, Thursday,", "#include <stdio.h>\n\nint main(void) {\n"]
B = ["Explain how a hash table handles collisions, with an example in Python:\n", "The causes of the First World War include", "fn main() {\n    let v: Vec<i32> = (1..=10).collect();\n",
 "Summarize the plot of Romeo and Juliet in five sentences.\n", "Translate to French: The weather is lovely today and we are going to the park.\n", "A: What is the derivative of x^3 * sin(x)?\nB:",
 "Steps to configure a reverse proxy with nginx:\n1.", "The mitochondrion is often called the powerhouse of the cell because", "class LinkedList:\n    def __init__(self):\n",
 "Write a short product description for a stainless steel water bottle.\n", "In 1969, the Apollo 11 mission", "{\"name\": \"Alice\", \"orders\": [", "Compare and contrast supervised and unsupervised learning:\n",
 "Bash one-liner to find the ten largest files under a directory:\n", "Quarterly revenue grew 12% year over year, driven by", "The three laws of thermodynamics state that"]
json.dump(A, open(sys.argv[1], "w")); json.dump(B, open(sys.argv[2], "w"))
PYEOF

phase() { # name prompts_file seedbase
  local name=$1 pf=$2 seed=$3 D=$R/$1
  [ -e "$D/DONE" ] && return 0
  stopped && { log "STOP file"; return 3; }
  [ "$(time_left)" -lt 1500 ] && { log "SKIP $name: $(time_left)s left < 1500"; return 3; }
  ds41_guard_or_refuse "conf1 $name"; ds41_memgate "conf1 $name"
  ss -ltn "sport = :$PORT" | grep -q LISTEN && { echo "port $PORT in use"; exit 4; }
  rm -rf "$D"; mkdir -p "$D"
  export LD_LIBRARY_PATH=$BIN OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active OMP_DYNAMIC=false GGML_IQK=1
  LLAMA_DSPARK_CONF_LOG=$D/$name.jsonl LLAMA_DSPARK_CONF_PMIN_LOG=0 \
  taskset -c "$CPUS" "$BIN/llama-server" --no-webui -t "$NT" -tb "$NT" -np 1 -c 4096 --host 127.0.0.1 --port "$PORT" -m "$M1017" -fa on \
     -md "$DRAFT" --spec-type draft-dspark --spec-draft-n-max 2 > "$D/server.log" 2>&1 &
  SRV_PID=$!; echo "$SRV_PID" > "$D/server.pid"
  wait_ready "$PORT" 1500 || { log "not ready"; kill_srv; return 7; }
  "$PY" -I - "$PORT" "$D" "$pf" "$seed" $(( $(time_left) - 90 )) <<'PYEOF'
import json, sys, time, urllib.request
port, d, pf, seed, budget = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
t0 = time.time(); prompts = json.load(open(pf)); n = 0
for rep in range(4):                      # repeat the prompt list with new seeds until the window or 4 passes end
    for i, p in enumerate(prompts):
        if time.time() - t0 > budget: print("budget reached"); sys.exit(75)
        body = {"prompt": p, "n_predict": 256, "temperature": 0.7, "top_p": 1.0, "top_k": 0, "seed": seed + 100 * rep + i, "cache_prompt": False, "speculative.n_max": 2}
        r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800))
        n += 1; t = r.get("timings", {})
        print(rep, i, t.get("draft_n"), t.get("draft_n_accepted"), flush=True)
        json.dump({"requests": n}, open(d + "/progress.json", "w"))
PYEOF
  local rc=$?
  kill_srv
  local rounds; rounds=$(wc -l < "$D/$name.jsonl" 2>/dev/null || echo 0)
  log "$name: client rc=$rc, rounds logged=$rounds"
  [ "$rounds" -ge 500 ] && { date -u +%FT%TZ > "$D/DONE"; return 0; }
  return 3
}
phase main "$PA" 1234;    rc1=$?
[ $rc1 = 0 ] && { phase holdout "$PB" 91234; rc2=$?; } || rc2=3
if [ -e "$R/main/DONE" ]; then
  args=(--log "$R/main/main.jsonl" --out "$R/conf_reliability.json" --binary-id "dd3d9721c+b428d7392" --trunk "$M1017" --drafter "$DRAFT")
  [ -e "$R/holdout/DONE" ] && args+=(--holdout-log "$R/holdout/holdout.jsonl")
  "$PY" -I "$HERE/conf_reliability.py" "${args[@]}" > "$R/conf_reliability.txt" 2>&1 || log "conf_reliability failed (see conf_reliability.txt)"
fi
[ -e "$R/main/DONE" ] && [ -e "$R/holdout/DONE" ] && { date -u +%FT%TZ > "$RUNDIR/COMPLETE"; log "all done"; }
[ "$rc1" = 4 ] && exit 4
exit 0
