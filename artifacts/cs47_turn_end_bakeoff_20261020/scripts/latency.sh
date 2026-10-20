#!/bin/bash
# CS-47 TIMING LEG - run ONLY after the operator's 00:30-06:30Z full-host window ends. Not run on 2026-10-20.
# usage: latency.sh --deadline 2026-10-21T10:00:00Z [--repeats 30] [--regions q0,q1,q2,q3] [--pin4 48-51] [--pin1 48]
# Takes ONE exclusive bench claim (region-lock, role bench) for the whole sweep; exits 75 if the claim is not granted within --claim-wait s.
# Expected runtime ~70-90 min (3 parakeet variants x {4 cores,1 core} x A/A pair, + smartturn + silero). Never contacts any production port.
set -euo pipefail
T=/mnt/raid0/llm/tmp/cs-47-20261020
RL=/workspace/repos/epyc-orchestrator/scripts/region-lock
DEADLINE=""; REPEATS=30; REGIONS="q0,q1,q2,q3"; PIN4="48-51"; PIN1="48"; CLAIM_WAIT=900
while [ $# -gt 0 ]; do case "$1" in
  --deadline) DEADLINE="$2"; shift 2;; --repeats) REPEATS="$2"; shift 2;; --regions) REGIONS="$2"; shift 2;;
  --pin4) PIN4="$2"; shift 2;; --pin1) PIN1="$2"; shift 2;; --claim-wait) CLAIM_WAIT="$2"; shift 2;;
  *) echo "unknown arg $1" >&2; exit 2;; esac; done
[ -n "$DEADLINE" ] || { echo "--deadline ISO-UTC required" >&2; exit 2; }
DEADLINE_EPOCH=$(date -u -d "$DEADLINE" +%s)
[ "$(date -u +%s)" -lt "$DEADLINE_EPOCH" ] || { echo "deadline already passed" >&2; exit 3; }
if [ "${CS47_INNER:-0}" != 1 ]; then
  # outer: acquire the exclusive bench claim, then re-enter as inner
  export CS47_INNER=1
  exec "$RL" run --regions "$REGIONS" --role bench --timeout-s "$CLAIM_WAIT" --tag cs-47-latency -- \
       timeout --signal=TERM $((DEADLINE_EPOCH - $(date -u +%s))) "$0" --deadline "$DEADLINE" --repeats "$REPEATS" --regions "$REGIONS" --pin4 "$PIN4" --pin1 "$PIN1"
fi
OUT=$T/latency_out; mkdir -p "$OUT"
PYV=$T/venv/bin/python; PYO=/mnt/raid0/llm/epyc-orchestrator/.venv/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ORT_NUM_THREADS=1 HF_HUB_OFFLINE=1
{ echo "# config echo"; date -u; uname -a; echo "regions=$REGIONS pin4=$PIN4 pin1=$PIN1 repeats=$REPEATS deadline=$DEADLINE";
  "$RL" status; sha256sum $T/dl/smart-turn-v3.2-cpu.onnx; python3 - <<'P'
import glob
for f in sorted(glob.glob("/sys/devices/system/cpu/cpu48/cpufreq/scaling_cur_freq")): print(f, open(f).read().strip())
P
} > "$OUT/config_echo.txt" 2>&1
run() { # pinlabel cpus arm variant aa
  local plabel=$1 cpus=$2 arm=$3 var=$4 aa=$5 py=$PYO
  [ "$arm" = smartturn ] && py=$PYV
  [ "$(date -u +%s)" -lt "$DEADLINE_EPOCH" ] || { echo "deadline reached, stopping" >&2; return 0; }
  taskset -c "$cpus" "$py" -I $T/lat_worker.py --arm "$arm" --variant "$var" --repeats "$REPEATS" --pin-label "$plabel" --aa-pass "$aa" \
     --deadline-epoch "$DEADLINE_EPOCH" --out "$OUT/lat_${arm}_${var//,/_}_${plabel}_${aa}.json" >> "$OUT/latency.log" 2>&1
}
for AA in A1 A2; do
  for PL in "pin4:$PIN4" "pin1:$PIN1"; do
    P=${PL%%:*}; C=${PL#*:}
    run $P $C silero x $AA
    run $P $C smartturn x $AA
    for V in fp32_fp32,160,decoder fp32_fp32,80,decoder fp32_selfdec,160,decoder; do run $P $C parakeet $V $AA; done
  done
done
echo "latency legs finished $(date -u)" | tee -a "$OUT/latency.log"
