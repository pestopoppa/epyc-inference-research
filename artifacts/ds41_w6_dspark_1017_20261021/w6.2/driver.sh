#!/bin/bash
# DS41 W5a/W5b/W6.2 chained driver for the booked slot 2026-10-22T22:00Z -> 2026-10-23T00:30Z (q0+q1, 0-47, -t 48,
# build role). Starts detached:  setsid nohup driver.sh </dev/null >> driver.log 2>&1 &
#   1. waits (shell sleep, zero tokens) until BOTH now >= START and the DS41 AK floor run's end flag
#      /mnt/raid0/llm/tmp/compute-flags/DS41_AK_FLOOR_ENDED exists and is fresh (mtime >= FLAG_MIN);
#   2. takes ONE build-role region-lock claim on 0-47 (acquire bounded so it can never start after HARD);
#   3. runs inner.sh (W5A, W5B, W6.2 pairs incl. greedy) inside it;
#   4. always writes the end flag /mnt/raid0/llm/tmp/compute-flags/DS41_W62_ENDED (rc, time, reason).
# DRY_RUN=1: validate everything and print the plan; no wait, no claim, no server, no flags.
# STOP file: /mnt/raid0/llm/tmp/ds41-w5ab-w62-20261021/STOP (honoured while waiting and between steps).
set -uo pipefail
W=/mnt/raid0/llm/tmp/ds41-w5ab-w62-20261021
FLAGDIR=/mnt/raid0/llm/tmp/compute-flags
TL=${DS41_W62_TL:-$FLAGDIR/DS41_W62_TIMELINE.txt}
ENDF=${DS41_W62_ENDF:-$FLAGDIR/DS41_W62_ENDED}
AKF=${DS41_W62_AKF:-$FLAGDIR/DS41_AK_FLOOR_ENDED}
START=$(date -u -d "${DS41_W62_START:-2026-10-22T22:00:00Z}" +%s)
HARD=$(date -u -d "${DS41_W62_HARD:-2026-10-23T00:30:00Z}" +%s)
FLAG_MIN=$(date -u -d "${DS41_W62_FLAG_MIN:-2026-10-22T00:00:00Z}" +%s)
RL=/mnt/raid0/llm/epyc-orchestrator/scripts/region-lock
PY=/mnt/raid0/llm/epyc-inference-research/.venv/bin/python
iso() { date -u -d @"$1" +%FT%TZ; }

if [ "${DRY_RUN:-0}" = 1 ]; then
  out=$W/dryrun.txt; : > "$out"
  say() { echo "$*" | tee -a "$out"; }
  fail=0
  chk() { if eval "$2" >/dev/null 2>&1; then say "OK   $1"; else say "FAIL $1"; fail=1; fi; }
  say "DRY RUN $(date -u +%FT%TZ)  start=$(iso "$START") hard=$(iso "$HARD") flag=$AKF (fresh if mtime>=$(iso "$FLAG_MIN"))"
  for s in driver.sh inner.sh memgate.sh w62/arm.sh w5a/run_w5a_kernel.sh; do chk "bash -n $s" "bash -n $W/$s"; chk "exec bit $s" "test -x $W/$s"; done
  chk "py_compile probes.py client.py analyze_post.py" "$PY -m py_compile $W/w5b/probes.py $W/w62/client.py $W/analyze_post.py"
  chk "probe prompts build (W5B_DRY)" "W5B_DRY=1 $PY -I $W/w5b/probes.py 0-47 48 18981 $W/w5b/dry"
  for f in /mnt/raid0/llm/llama.cpp-experimental-ds41-20261005/build-cpu/bin/llama-server \
           /mnt/raid0/llm/llama.cpp-experimental-ds41-20261005/build-cpu/bin/libggml-cpu.so \
           /mnt/raid0/llm/llama.cpp-experimental-ds41-20261005/ggml/include/ggml-backend.h \
           /mnt/raid0/llm/models/deepseek-ai/DeepSeek-V4.1-Flash-1017-DSpark.gguf \
           /mnt/raid0/llm/models/deepseek-ai/DeepSeek-V4.1-Flash-DSpark.gguf \
           /mnt/raid0/llm/models/chat-templates/deepseek-v41-flash-epyc-v1.jinja \
           $W/w5a/mxfp4_repack_nmse.cpp $W/w5b/prompts_len.jsonl $RL; do chk "exists $f" "test -e $f"; done
  for i in 1 2 3 4 5 6 7 8 9; do chk "shard $i" "test -s /mnt/raid0/llm/models/antirez/deepseek-v4.1-flash-gguf/DeepSeek-V4.1-Flash-1017-MXFP4-experts-0000$i-of-00009.gguf"; done
  for p in 18981 18982 18983; do chk "port $p free now" "! curl -sf http://127.0.0.1:$p/health"; done
  chk "g++ present" "command -v g++"
  say "memgate (informational, no wait): $($W/memgate.sh DRY 0)"
  say "AK floor flag now: $(ls -l --time-style=+%FT%TZ "$AKF" 2>/dev/null || echo absent)"
  say "region-lock status: $($RL status 2>&1 | tr '\n' ';')"
  say "PLAN: wait until max($(iso "$START"), fresh $AKF); region-lock run --cpu-list 0-47 --role build --tag ds41-w62-slot --timeout-s <to HARD-15min> -- inner.sh"
  say "PLAN: W5A (<=15 min) -> W5B one target-only server (<=20 min) -> p3 A,B (+greedy) -> p4 B,A -> p5 A,B -> p6 B,A if it fits; every step bounded to HARD"
  say "DRY RESULT: $([ $fail = 0 ] && echo PASS || echo FAIL)"
  exit $fail
fi

echo $$ > "$W/driver.pid"
reason="unknown"; rc=1
finish() {
  echo "rc=$rc time=$(date -u +%FT%TZ) reason=$reason driver_pid=$$" > "$ENDF"
  echo "$(date -u +%FT%TZ) DRIVER END rc=$rc reason=$reason" >> "$TL"
}
trap 'reason="signalled"; rc=143; exit 143' TERM INT
trap finish EXIT
echo "$(date -u +%FT%TZ) DRIVER armed pid=$$ start=$(iso "$START") hard=$(iso "$HARD") waits_on=$AKF" >> "$TL"

while :; do
  now=$(date +%s)
  if [ -e "$W/STOP" ]; then reason="STOP file before start"; rc=3; exit 3; fi
  if [ "$now" -ge $(( HARD - 1200 )) ]; then reason="no start: <20 min before HARD (flag $( [ -e "$AKF" ] && echo present || echo absent))"; rc=124; exit 124; fi
  if [ "$now" -ge "$START" ] && [ -e "$AKF" ] && [ "$(stat -c %Y "$AKF")" -ge "$FLAG_MIN" ]; then break; fi
  sleep 60
done
echo "$(date -u +%FT%TZ) GATE open: now>=START and fresh $AKF ($(cat "$AKF" 2>/dev/null | head -c 200))" >> "$TL"

acq=$(( HARD - 900 - $(date +%s) ))
"$RL" run --cpu-list 0-47 --role build --tag ds41-w62-slot --timeout-s "$acq" -- "$W/inner.sh"
rc=$?
case $rc in
  0) reason="completed";;
  75) reason="region-lock acquire timeout";;
  *) reason="inner rc=$rc";;
esac
exit $rc
