#!/bin/bash
# DS41 W5a/W5b/W6.2 slot body. Runs INSIDE the driver's single build-role region-lock claim on 0-47.
# Steps (strictly sequential; at most one DS41 server alive at any time):
#   W5A  CPU_REPACK MXFP4 MUL_MAT_ID NMSE harness (no server)
#   W5B  one target-only 1017 server, W5b probe set
#   W62  pairs p3..p6, ABBA order (p3 A,B / p4 B,A / p5 A,B / p6 B,A), seeds 1000*p+234+prompt;
#        p3 arms also run the full greedy text-identity pass (greedymode 1) on the same server.
#        p3-p5 are the target (>=3 new complete pairs); p6 only if the measured pair time fits before HARD.
# Every step is bounded by `timeout` to the hard stop; arm.sh traps TERM and kills its own server PID.
set -uo pipefail
W=/mnt/raid0/llm/tmp/ds41-w5ab-w62-20261021
TL=${DS41_W62_TL:-/mnt/raid0/llm/tmp/compute-flags/DS41_W62_TIMELINE.txt}
HARD=$(date -u -d "${DS41_W62_HARD:-2026-10-23T00:30:00Z}" +%s)
CPUS=0-47; NT=48
DA=/mnt/raid0/llm/models/deepseek-ai/DeepSeek-V4.1-Flash-1017-DSpark.gguf
DB=/mnt/raid0/llm/models/deepseek-ai/DeepSeek-V4.1-Flash-DSpark.gguf
PY=/mnt/raid0/llm/epyc-inference-research/.venv/bin/python
tl() { echo "$(date -u +%FT%TZ) $*" | tee -a "$TL"; }
left() { echo $(( HARD - $(date +%s) )); }
stopped() { [ -e "$W/STOP" ]; }
# run_step <name> <budget_s> <cmd...> : bounded by min(budget, time to HARD - 60 s)
run_step() {
  local name=$1 budget=$2; shift 2
  local l; l=$(( $(left) - 60 ))
  [ "$l" -lt "$budget" ] && budget=$l
  if [ "$budget" -lt 120 ]; then tl "SKIP $name (only ${l}s before HARD)"; return 124; fi
  if stopped; then tl "SKIP $name (STOP file)"; return 3; fi
  tl "START $name budget=${budget}s"
  local t0; t0=$(date +%s)
  timeout --signal=TERM --kill-after=150 "$budget" "$@"
  local rc=$?
  tl "END $name rc=$rc dur=$(( $(date +%s) - t0 ))s"
  return $rc
}

tl "INNER start pid=$$ cpus=$CPUS -t $NT hard=$(date -u -d @"$HARD" +%FT%TZ)"
run_step W5A 900 taskset -c "$CPUS" "$W/w5a/run_w5a_kernel.sh" "$NT"

if "$W/memgate.sh" W5B 600 | tee -a "$TL"; then
  run_step W5B 1200 "$PY" -I "$W/w5b/probes.py" "$CPUS" "$NT" 18981 "$W/w5b/run"
else
  tl "SKIP W5B (memgate)"
fi

# pair p: order, greedymode
declare -A ORDER=( [3]="A B" [4]="B A" [5]="A B" [6]="B A" )
declare -A GM=( [3]=1 [4]=0 [5]=0 [6]=0 )
est_pair=0
for p in 3 4 5 6; do
  if stopped; then tl "STOP file: no more pairs"; break; fi
  need=$(( est_pair > 0 ? est_pair * 115 / 100 : 2400 ))
  if [ "$(left)" -lt "$need" ]; then tl "SKIP pair p$p: $(left)s left < need ${need}s"; break; fi
  t0=$(date +%s); prc=0
  for arm in ${ORDER[$p]}; do
    if [ "$arm" = A ]; then D=$DA; port=18982; else D=$DB; port=18983; fi
    run_step "W62_p${p}${arm}" 2700 "$W/w62/arm.sh" "p${p}${arm}" "$D" "$port" $(( 1000 * p + 234 )) "${GM[$p]}" "$CPUS" "$NT" \
      >> "$W/w62/arms.log" 2>&1 || prc=1
  done
  dur=$(( $(date +%s) - t0 ))
  tl "PAIR p$p done rc=$prc dur=${dur}s"
  # p3 carries the greedy pass; later pairs are stoch-only, so estimate from them once measured
  if [ "$p" = 3 ]; then est_pair=$(( dur * 80 / 100 )); else est_pair=$dur; fi
  [ "$prc" = 0 ] || tl "pair p$p had a failed arm (excluded by the analysis if incomplete)"
done
tl "INNER end"
