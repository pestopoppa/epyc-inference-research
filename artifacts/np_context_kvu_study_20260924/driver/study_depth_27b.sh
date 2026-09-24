#!/bin/bash
# G1b: MTP draft depth 4/6 vs 8 under unified KV (production 27B). Built from the kvu headroom driver.
# np x generation-length surface, DEDICATED vs UNIFIED KV, for the production Qwen3.8-27B Q8_0 (:8083 model).
# Successor to np_context_study_20260723 (TB-6, v7 kernel, pre-BIOS). Operator request 2026-09-24:
# relative effect of --kv-unified vs split slots at the SAME total -c, on the v10 production binary and the
# production recipe (q8_0 KV, draft-mtp n_max 8, terse template, -b/-ub 2048 per K4).
# Also: O-2 (-ctkd/-ctvd q8_0) A/B, and a >98k single-prompt functional check (unified accepts, split refuses).
# GPU ONLY. Requires :8083 stopped (VRAM). Host threads pinned 184-191 (SMT siblings; see memory: they still
# contend with CPU measurement — no CPU campaign may run concurrently).
set -uo pipefail
ART=/mnt/raid0/llm/epyc-inference-research/artifacts/np_context_kvu_study_20260924
RES=/mnt/raid0/llm/epyc-inference-research
BIN=/mnt/raid0/llm/kernels/production/gpu/llama-server
LIBDIR=/mnt/raid0/llm/kernels/production/gpu
PORT=18072
CORES=184-191
MODEL=/mnt/raid0/llm/models/Qwen3.8-27B-Q8_0.gguf
TEMPLATE=/mnt/raid0/llm/models/chat-templates/epyc-qwen3x-v1-terse.jinja
PIN=/mnt/raid0/llm/epyc-inference-research/artifacts/architect-bench-gpu-20260720/questions_olympiadbench_hard.json
LABEL=q38_27b_q8_depth
HEADROOM=1024   # per-request prompt room; v1 (c=L*np) ran every pool full: split truncated, unified hit the MTP bug
SUMMARY="$ART/$LABEL/summary.tsv"
mkdir -p "$ART/$LABEL"
[ -f "$SUMMARY" ] || printf 'arm\tnp\tL\tc\tagg_decode\tperreq_med\terrors\tvram_gib\tn_ctx_slot\tkv_unified\tdraft_accept\textra\n' > "$SUMMARY"

launch() {  # launch <dir> <extra flags...>
  local d="$1"; shift; mkdir -p "$d"
  nohup taskset -c "$CORES" env LD_LIBRARY_PATH="$LIBDIR" ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=1 "$BIN" -m "$MODEL" --host 127.0.0.1 --port "$PORT" --metrics --slots --jinja \
    --chat-template-file "$TEMPLATE" --reasoning off --device ROCm0 -ngl all -fa on --no-mmap \
    -t 8 -tb 8 -b 2048 -ub 2048 -ctk q8_0 -ctv q8_0 --spec-type draft-mtp --spec-draft-n-max 8 \
    "$@" > "$d/server.stdout" 2> "$d/server.stderr" &
  echo $! > "$d/server.pid"
  printf '%s ' "$BIN" -m "$MODEL" "$@" > "$d/server_command.txt"
}
wait_healthy() {  # wait_healthy <dir>
  local d="$1" pid; pid=$(cat "$d/server.pid"); local deadline=$(( $(date +%s) + 600 ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    ps -p "$pid" >/dev/null 2>&1 || { echo DIED; return 1; }
    curl -sf "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -qi ok && { echo HEALTHY; return 0; }
    sleep 3
  done; echo TIMEOUT; return 1
}
stop_server() {  # stop_server <dir>
  local d="$1" pid; pid=$(cat "$d/server.pid" 2>/dev/null || echo ""); [ -z "$pid" ] && return 0
  kill -TERM "$pid" 2>/dev/null; sleep 8
  ps -p "$pid" >/dev/null 2>&1 && { kill -9 "$pid" 2>/dev/null; sleep 5; }
  ps -p "$pid" >/dev/null 2>&1 && { echo "KILL_FAILED $pid"; return 1; }; return 0
}
facts() {  # facts <dir> -> "n_ctx_slot kv_unified vram_gib"
  local d="$1"
  local nslot kvu vram
  nslot=$(grep -oP 'n_ctx_slot\s*=\s*\K[0-9]+|n_ctx_seq\s*=\s*\K[0-9]+' "$d/server.stderr" | tail -1)
  kvu=$(grep -oP "kv_unified\s*=\s*'\K[a-z]+" "$d/server.stderr" | tail -1)
  vram=$(( $(rocm-smi --showmeminfo vram 2>/dev/null | grep -oiP 'used memory.*?:\s*\K[0-9]+' | head -1) / 1073741824 ))
  echo "${nslot:-NA} ${kvu:-NA} ${vram}"
}
draft_accept() { grep -oP 'draft acceptance\s*=\s*\K[0-9.]+' "$1/server.stderr" | awk '{s+=$1;n++} END{if(n)printf "%.3f",s/n; else print "NA"}'; }

cell() {  # cell <arm> <np> <L> [extra flags...]   arm = split|unified|unified_o2|split_o2
  local arm="$1" NP="$2" L="$3"; shift 3
  local c=$(( (L + HEADROOM) * NP )); local d="$ART/$LABEL/${arm}/np${NP}_L${L}"
  if [ -f "$d/done" ]; then return; fi
  rm -rf "$d"; mkdir -p "$d"
  local kvflag="--no-kv-unified"; case "$arm" in unified*) kvflag="--kv-unified";; esac
  launch "$d" -np "$NP" -c "$c" "$kvflag" "$@"
  local st; st=$(wait_healthy "$d")
  if [ "$st" != HEALTHY ]; then
    printf '%s\t%s\t%s\t%s\tSERVER_%s\t\t\t\t\t\t\t%s\n' "$arm" "$NP" "$L" "$c" "$st" "$*" >> "$SUMMARY"
    tail -3 "$d/server.stderr"; stop_server "$d"; touch "$d/done"; return
  fi
  read -r nslot kvu vram <<< "$(facts "$d")"
  if [ "$vram" -gt 62 ]; then
    printf '%s\t%s\t%s\t%s\tSKIP_VRAM\t\t\t%s\t%s\t%s\t\t%s\n' "$arm" "$NP" "$L" "$c" "$vram" "$nslot" "$kvu" "$*" >> "$SUMMARY"
    stop_server "$d"; touch "$d/done"; return
  fi
  ( cd "$RES" && HF_HOME=/mnt/raid0/llm/cache/huggingface RUNNER_REQUEST_TIMEOUT_S=5400 \
    uv run python scripts/benchmark/v7_quality_gate_runner.py \
      --port "$PORT" --host 127.0.0.1 --suites olympiadbench_hard --n 155 --limit "$NP" --seed 42 \
      --max-tokens "$L" --repeats 1 --concurrency "$NP" \
      --temperature 0.6 --top-p 0.95 --top-k 20 --no-enable-thinking --endpoint chat \
      --arm "${LABEL}_${arm}_np${NP}_L${L}" --binary x --models "$MODEL" --questions-in "$PIN" \
      --per-question-out "$d/pq.jsonl" --output "$d/r.json" > "$d/out" 2> "$d/err" ) || true
  local da; da=$(draft_accept "$d")
  stop_server "$d"
  python3 - "$d" "$arm" "$NP" "$L" "$c" "$vram" "$nslot" "$kvu" "$da" "$*" >> "$SUMMARY" <<'EOF'
import json,statistics,sys
d,arm,np_,L,c,vram,nslot,kvu,da,extra=sys.argv[1:]
try:
    r=json.load(open(d+"/r.json"))["suites"][0]; tp=r.get("throughput",{})
    pq=[json.loads(l) for l in open(d+"/pq.jsonl")]
    dec=[x["decode_tok_s"] for x in pq if x.get("decode_tok_s")]
    print("\t".join([arm,np_,L,c,"%.1f"%tp.get("aggregate_decode_tok_s",0),"%.1f"%(statistics.median(dec) if dec else 0),
                     str(r.get("errors",0)),vram,nslot,kvu,da,extra]))
except Exception as e:
    print("\t".join([arm,np_,L,c,"PARSE_FAIL",repr(e)[:60],"",vram,nslot,kvu,da,extra]))
EOF
  touch "$d/done"
  tail -1 "$SUMMARY"
}

longprompt() {  # longprompt <arm> : one ~120k-token prompt into -np 2 -c 196608
  local arm="$1"; local d="$ART/$LABEL/longprompt_${arm}"
  if [ -f "$d/done" ]; then return; fi
  rm -rf "$d"; mkdir -p "$d"
  local kvflag="--no-kv-unified"; [ "$arm" = unified ] && kvflag="--kv-unified"
  launch "$d" -np 2 -c 196608 "$kvflag"
  [ "$(wait_healthy "$d")" = HEALTHY ] || { echo "longprompt $arm: server failed"; stop_server "$d"; touch "$d/done"; return; }
  python3 - "$PORT" "$d" <<'EOF' > "$d/result.json"
import json,sys,time,urllib.request,urllib.error
port,d=sys.argv[1],sys.argv[2]
para=("The quick brown fox jumps over the lazy dog while the orchestrator schedules kernels across NUMA nodes. ")*1
text=para*6200   # ~120k tokens
body={"messages":[{"role":"user","content":text+"\nReply with the single word OK."}],"max_tokens":16,"temperature":0}
t=time.time()
req=urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
try:
    with urllib.request.urlopen(req,timeout=3600) as r:
        out=json.loads(r.read()); print(json.dumps({"ok":True,"wall_s":round(time.time()-t,1),"usage":out.get("usage"),"timings":out.get("timings"),"content":out["choices"][0]["message"]["content"][:40]}))
except urllib.error.HTTPError as e:
    print(json.dumps({"ok":False,"status":e.code,"wall_s":round(time.time()-t,1),"body":e.read().decode()[:400]}))
except Exception as e:
    print(json.dumps({"ok":False,"error":repr(e)[:300]}))
EOF
  read -r nslot kvu vram <<< "$(facts "$d")"
  stop_server "$d"
  echo "longprompt $arm: n_ctx_slot=$nslot kv_unified=$kvu vram=${vram}G -> $(cat "$d/result.json")" | tee -a "$ART/$LABEL/longprompt.txt"
  touch "$d/done"
}


loadonly() {  # loadonly <depth> : production shape -np 2 -c 196608 --kv-unified, measure resident GPU memory only
  local D="$1"; local d="$ART/$LABEL/loadonly_d${D}"
  if [ -f "$d/done" ]; then return; fi
  rm -rf "$d"; mkdir -p "$d"
  launch "$d" -np 2 -c 196608 --kv-unified --spec-draft-n-max "$D"
  if [ "$(wait_healthy "$d")" = HEALTHY ]; then
    sleep 5; read -r nslot kvu vram <<< "$(facts "$d")"
    local mib; mib=$(( $(rocm-smi --showmeminfo vram 2>/dev/null | grep -oiP 'used memory.*?:\s*\K[0-9]+' | head -1) / 1048576 ))
    echo "loadonly depth=$D: n_ctx_slot=$nslot kv_unified=$kvu vram_mib=$mib" | tee -a "$ART/$LABEL/loadonly.txt"
  else
    echo "loadonly depth=$D: server failed" | tee -a "$ART/$LABEL/loadonly.txt"
  fi
  stop_server "$d"; touch "$d/done"
}

echo "=== start $(date -u +%FT%TZ) ==="
loadonly 8
loadonly 4
for L in 2048 8192; do
  cell unified_d4 2 "$L" --spec-draft-n-max 4
  cell unified_d6 2 "$L" --spec-draft-n-max 6
done
cell unified_d4 4 8192 --spec-draft-n-max 4
echo "=== done $(date -u +%FT%TZ) ==="
