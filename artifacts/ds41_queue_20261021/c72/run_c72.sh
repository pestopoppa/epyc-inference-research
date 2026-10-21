#!/bin/bash
# DS41-C7.2 INNER (runs inside the launcher's build-role region-lock claim, in its own process group).
# For ctx in 8192 32768 131072: ONE profiled 1017 target-only server per ctx (sequential, never two), exact-N-token
# id prompt, GGML_CPU_PROF_SKIP = prefill graph count, 64 decode steps, graceful TERM so the at-exit dump lands,
# then group_nodes.py. Resumable per ctx (ctx_<N>/DONE). Env from the launcher: FILLER_END FILLER_CPUS FILLER_THREADS
# RUNDIR STOPFILE. Writes $RUNDIR/COMPLETE when all three ctx are DONE and grouped.
set -uo pipefail
HERE=$(dirname "$(readlink -f "$0")")
LIB=${LIB:-$HERE/../lib}
. "$LIB/ds41_guard.sh"; . "$LIB/srv.sh"
BIN=${C72_BIN:-/mnt/raid0/llm/llama.cpp-experimental-ds41-20261005/build-cpu-prof/bin}   # DS41-C8 instrumented sibling (GGML_CPU_PROF=ON)
TGT=${C72_TGT:-/mnt/raid0/llm/models/antirez/deepseek-v4.1-flash-gguf/DeepSeek-V4.1-Flash-1017-MXFP4-experts-00001-of-00009.gguf}
TOKJSON=${C72_TOKENIZER:-/mnt/raid0/llm/models/deepseek-ai/DeepSeek-V4.1-Flash/tokenizer.json}
CORPUS=${C72_CORPUS:-/mnt/raid0/llm/tmp/ds41-queue-20261021/corpus/docs}   # frozen copy of epyc-root docs/*.md at origin/main
PY=${PY:-/mnt/raid0/llm/epyc-inference-research/.venv/bin/python}
PORT=${C72_PORT:-18991}
UB=512; BATCH=2048; ND=64; SKIP_PAD=2
CTXS=${C72_CTXS:-"8192 32768 131072"}
OUT=$RUNDIR/results; mkdir -p "$OUT"
NT=${FILLER_THREADS:?}; CPUS=${FILLER_CPUS:?}
trap 'kill_srv' EXIT
trap 'kill_srv; exit 143' TERM INT
log() { echo "$(date -u +%FT%TZ) [c72] $*"; }
check_bin "$BIN" || exit 6
for s in "$LIB/../c72/group_nodes.py" "$LIB/../c72/make_prompt.py"; do [ -s "$s" ] || { echo "missing $s"; exit 6; }; done
# a null from an instrument that is not compiled in is not evidence: prove the profiler is in THIS binary
strings "$BIN"/libggml-cpu.so.0.* | grep -q GGML_CPU_PROF_THREADS || { echo "profiler not compiled into $BIN"; exit 6; }
# crude load-time estimates (s) used only to decide whether a ctx fits; refined from the measured prefill rate
est_for() { # ctx
  local c=$1
  if [ -s "$OUT/rate_tok_s" ]; then awk -v c="$c" -v r="$(cat "$OUT/rate_tok_s")" 'BEGIN{ f=(c>32768)?1.35:1.0; printf "%d", 150 + 120 + c/r*f + 64/3 }'
  else case $c in 8192) echo 700;; 32768) echo 2000;; *) echo 6500;; esac; fi
}
ALLDONE=1
for N in $CTXS; do
  D=$OUT/ctx_$N
  [ -e "$D/DONE" ] && continue
  ALLDONE=0
  stopped && { log "STOP file: no more ctx"; break; }
  need=$(est_for "$N")
  if [ "$(time_left)" -lt $(( need + 90 )) ]; then log "SKIP ctx=$N: $(time_left)s left < need ${need}s+90 (resume on the next grant)"; continue; fi
  ds41_guard_or_refuse "c72 ctx=$N"; ds41_memgate "c72 ctx=$N"
  ss -ltn "sport = :$PORT" | grep -q LISTEN && { echo "port $PORT in use"; exit 4; }
  rm -rf "$D"; mkdir -p "$D"
  "$PY" -I "$LIB/../c72/make_prompt.py" "$TOKJSON" "$CORPUS" "$N" "$D/prompt.json" > "$D/prompt.meta" || { log "prompt build failed"; exit 5; }
  SKIP=$(( (N + UB - 1) / UB + SKIP_PAD ))
  CTXSZ=$(( N + 256 ))
  log "ctx=$N skip=$SKIP -t $NT cpus=$CPUS need~${need}s left=$(time_left)s"
  export LD_LIBRARY_PATH=$BIN OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active OMP_DYNAMIC=false GGML_IQK=1
  GGML_CPU_PROF=1 GGML_CPU_PROF_THREADS=1 GGML_CPU_PROF_SKIP=$SKIP GGML_CPU_PROF_JSON_FILE=$D/ggml-cpu-prof.json \
  LLAMA_HOST_PROF_JSON_FILE=$D/llama-host-prof.json \
  taskset -c "$CPUS" "$BIN/llama-server" --no-webui --no-warmup -t "$NT" -tb "$NT" -np 1 -c "$CTXSZ" -b $BATCH -ub $UB --cache-ram 0 \
     --host 127.0.0.1 --port "$PORT" -m "$TGT" -fa on > "$D/server.log" 2>&1 &
  SRV_PID=$!; echo "$SRV_PID" > "$D/server.pid"
  wait_ready "$PORT" 1500 || { log "not ready"; kill_srv; exit 7; }
  grep Cpus_allowed_list /proc/"$SRV_PID"/status > "$D/affinity.txt"
  # request: ids prompt, greedy, exactly ND decode steps (ignore_eos). timeout = time left - 120 s for teardown.
  t0=$(date +%s)
  "$PY" -I - "$PORT" "$D" "$ND" $(( $(time_left) - 120 )) <<'PYEOF'
import sys, json, urllib.request, time
port, d, nd, tmo = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
ids = json.load(open(d + "/prompt.json"))["ids"]
body = {"prompt": ids, "n_predict": nd, "temperature": 0.0, "top_k": 1, "seed": 1, "ignore_eos": True, "cache_prompt": False}
req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
t = time.time()
r = json.load(urllib.request.urlopen(req, timeout=max(tmo, 60)))
tm = r.get("timings", {})
json.dump({"wall_s": time.time() - t, "tokens_evaluated": r.get("tokens_evaluated"), "timings": tm, "predicted_n": tm.get("predicted_n"),
           "prompt_n": tm.get("prompt_n"), "prompt_ms": tm.get("prompt_ms")}, open(d + "/request.json", "w"), indent=1)
print("request ok", r.get("tokens_evaluated"), tm.get("prompt_ms"), tm.get("predicted_per_second"))
PYEOF
  rrc=$?
  kill_srv      # graceful TERM first: the profiler dumps at exit
  if [ $rrc -ne 0 ] || [ ! -s "$D/request.json" ]; then log "request failed rc=$rrc"; exit 8; fi
  if [ ! -s "$D/ggml-cpu-prof.json" ]; then log "no ggml-cpu-prof.json dump (server did not exit gracefully?)"; exit 9; fi
  "$PY" -I - "$D" "$N" "$ND" "$SKIP" "$UB" "$NT" "$BIN" "$TGT" <<'PYEOF'
import sys, json
d, n, nd, skip, ub, nt, binp, tgt = sys.argv[1:9]
rq = json.load(open(d + "/request.json"))
pm = rq.get("prompt_ms")
json.dump({"n_tokens": int(n), "n_decode": int(nd), "skip": int(skip), "n_ubatch": int(ub), "threads": int(nt), "binary": binp, "target": tgt,
           "prefill_s": (pm / 1000.0) if pm else None, "tokens_evaluated": rq.get("tokens_evaluated"), "prompt_n": rq.get("prompt_n")},
          open(d + "/meta.json", "w"), indent=1)
PYEOF
  "$PY" -I -c "import json,sys;m=json.load(open('$D/meta.json'));print(m['n_tokens']/m['prefill_s'] if m.get('prefill_s') else '')" > "$OUT/rate_tok_s.tmp" 2>/dev/null
  [ -s "$OUT/rate_tok_s.tmp" ] && mv -f "$OUT/rate_tok_s.tmp" "$OUT/rate_tok_s"
  date -u +%FT%TZ > "$D/DONE"; log "ctx=$N DONE in $(( $(date +%s) - t0 ))s"
done
# group whatever is DONE so partial grants still produce a usable table
args=(); for N in $CTXS; do [ -e "$OUT/ctx_$N/DONE" ] && args+=(--ctx "$N=$OUT/ctx_$N"); done
if [ ${#args[@]} -gt 0 ]; then
  "$PY" -I "$LIB/../c72/group_nodes.py" "${args[@]}" --out "$OUT/c72_result.json" --md "$OUT/c72_table.md" || log "group_nodes failed"
fi
done_all=1; for N in $CTXS; do [ -e "$OUT/ctx_$N/DONE" ] || done_all=0; done
[ $done_all = 1 ] && { date -u +%FT%TZ > "$RUNDIR/COMPLETE"; log "all ctx done"; }
exit 0
