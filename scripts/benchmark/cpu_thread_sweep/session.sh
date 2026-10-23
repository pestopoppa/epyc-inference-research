#!/bin/bash
# R23-64: ONE LAUNCH = one llama-server process = one arm run of the 24-prompt production mix.
#
# ADAPTED FROM the INF-70 RETEST-1 session2_t.sh (the t-parameterised session script). Deliberate
# differences, each a correction (the measured recipe itself is unchanged -- it now comes from
# scripts/lib/qwen38_flash_next_recipe.py via sweep_lib.py, never retyped here):
#   1. CHAMP / TRUNK / MTPH / THREADS / CPU list are parameters, not constants. CHAMP is the CURRENT
#      champion build (the old script pinned cpu-20260921-ffc1bac82); TRUNK is the production UD-IQ4_XS
#      3-shard set (the old IQ4_XS-uniform file was retired 2026-10-20).
#   2. `-t` is set PER ARM, and the taskset mask is per arm (0-95, or 0-47 for the optional 6-CCD arm).
#   3. knobs.py is NOT called. The 141e4bcf5 champion has no knob page (GGML_KNOB_FILE is in no
#      library: STRINGS-PROVEN by knob-proof, re-proved per run). NO_KNOBS=1 is the only supported
#      mode; NO_KNOBS=0 refuses, it does not fall back.
#   4. samplers are scripts/utils/foreign_load.py (in this repo) + the vendored host_load.py; the
#      screens are the vendored screen.py with the arm's own -t as OWN_THREADS.
#   5. the server runs under `env -i` with an explicit LD_LIBRARY_PATH (the build's own bin dir only),
#      launched as taskset -c <list> numactl --interleave=all.
#   6. a hard-stop watchdog kills the server by CAPTURED PID (start-time checked against PID reuse),
#      then the exit path TERM -> KILL -> verify dead. Nothing here ever matches a process by name.
#   7. the health probe must prove OUR pid owns the port, not just that something answers.
#   8. a launch is complete only when client.py renames <label>.rows.jsonl.partial; a killed launch
#      leaves no rows.jsonl, so resume re-runs it.
#
#  usage: session.sh <label> <threads> <cpu-list>
#  env (required): CHAMP=<build>/bin   OUT=<runs dir>
#  env (optional): TRUNK MTPH (default: recipe)  PORT=18499  CTX=8192  UBATCH(unset)  EVICT_GIB=40  TRIES=3
#                  EXPECT_VERSION="10412 (141e4bcf5)"  NO_KNOBS=1  PY  FILLER_PIDFILE  PREFLIGHT_ONLY=1 (proofs only: no eviction, no server)
#                  HARD_STOP_EPOCH (epoch seconds, e.g. FILLER_END)  LAUNCH_MAX_S=720  HEALTH_MAX_S=480  CLIENT_MAX_S=600
#  exit: 0 done/skip | 3 placement | 5x refused-before-load | 6x server/proof failure | 70-74 proof failures
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)                       # the checkout this file lives in (scripts/ above it)
LABEL=${1:?label}; THREADS=${2:?threads}; CPUS=${3:?cpu-list}
CHAMP=${CHAMP:?CHAMP (build bin dir) required}; OUT=${OUT:?OUT (runs dir) required}
PY=${PY:-/usr/bin/python3 -E -s}
LIB="$PY $HERE/sweep_lib.py"
PORT=${PORT:-18499}; CTX=${CTX:-8192}; EVICT_GIB=${EVICT_GIB:-40}; TRIES=${TRIES:-3}
EXPECT_VERSION=${EXPECT_VERSION:-"10412 (141e4bcf5)"}; NO_KNOBS=${NO_KNOBS:-1}
LAUNCH_MAX_S=${LAUNCH_MAX_S:-720}; HEALTH_MAX_S=${HEALTH_MAX_S:-480}; CLIENT_MAX_S=${CLIENT_MAX_S:-600}
TRUNK=${TRUNK:-$($LIB files | head -1)}                  # shard 1 of the split set
MTPH=${MTPH:-$($LIB files | tail -1)}
mkdir -p "$OUT"
T0=$(date +%s)
say(){ echo "$(date -u +%T) +$(( $(date +%s)-T0 ))s [$LABEL] $*" | tee -a "$OUT/$LABEL.timeline"; }

[ -f "$OUT/$LABEL.rows.jsonl" ] && { echo "[$LABEL] rows.jsonl present, SKIP"; exit 0; }
[ "$NO_KNOBS" = 1 ] || { say "FATAL NO_KNOBS=$NO_KNOBS: knob-page mode needs knobs.py, which this harness does not vendor"; exit 70; }

SPID=""; SSTART=""; CPID=""; HPID=""; WDPID=""
# pid_ours PID START: the pid still carries the start time we captured (PID-reuse guard).
pid_ours(){ [ -n "$1" ] && [ -r "/proc/$1/stat" ] && [ "$(awk '{print $22}' "/proc/$1/stat" 2>/dev/null)" = "$2" ]; }
stop_server(){
  [ -n "$SPID" ] || return 0
  if pid_ours "$SPID" "$SSTART"; then
    kill -TERM "$SPID" 2>/dev/null; local j=0
    while pid_ours "$SPID" "$SSTART" && [ $j -lt 120 ]; do sleep 0.5; j=$((j+1)); done
    pid_ours "$SPID" "$SSTART" && { say "SIGTERM not honoured -> SIGKILL"; kill -KILL "$SPID" 2>/dev/null; sleep 3; }
  fi
  if pid_ours "$SPID" "$SSTART"; then say "FATAL server pid $SPID STILL ALIVE"; else say "server pid $SPID dead (verified)"; fi
  SPID=""
}
# arm_watchdog: hard stop on the CAPTURED server pid at min(T0+LAUNCH_MAX_S, HARD_STOP_EPOCH).
arm_watchdog(){
  [ -n "$WDPID" ] && kill -TERM "$WDPID" 2>/dev/null
  local hard=$(( T0 + LAUNCH_MAX_S )) pid=$SPID st=$SSTART
  [ -n "${HARD_STOP_EPOCH:-}" ] && [ "$HARD_STOP_EPOCH" -lt "$hard" ] && hard=$HARD_STOP_EPOCH
  ( s=$(( hard - $(date +%s) )); [ "$s" -lt 1 ] && s=1
    sleep "$s" & sl=$!; trap 'kill "$sl" 2>/dev/null; exit 0' TERM; wait "$sl"
    echo "$(date -u +%T) [$LABEL] WATCHDOG: hard stop reached, killing server pid=$pid"
    if pid_ours "$pid" "$st"; then kill -TERM "$pid" 2>/dev/null; sleep 20; pid_ours "$pid" "$st" && kill -KILL "$pid" 2>/dev/null; fi
  ) >> "$OUT/$LABEL.timeline" 2>&1 &
  WDPID=$!
}
cleanup(){
  [ -n "$CPID" ]  && kill -TERM "$CPID"  2>/dev/null
  [ -n "$HPID" ]  && kill -TERM "$HPID"  2>/dev/null
  [ -n "$WDPID" ] && kill -TERM "$WDPID" 2>/dev/null
  stop_server
}
trap cleanup EXIT
die(){ say "LAUNCH $LABEL end rc=$1 held $(( $(date +%s)-T0 ))s"; exit "$1"; }

# ---- proofs that need no server --------------------------------------------------------------
VER=$(LD_LIBRARY_PATH="$CHAMP" "$CHAMP/llama-server" --version 2>&1 | head -2 | tr '\n' ' ')
say "label=$LABEL threads=$THREADS cpus=$CPUS bin=$CHAMP ctx=$CTX port=$PORT version: $VER"
case "$VER" in *"$EXPECT_VERSION"*) ;; *) say "FATAL binary is not build '$EXPECT_VERSION'"; die 71;; esac
if [ ! -s "$OUT/knob_proof.json" ]; then
  $LIB knob-proof --bindir "$CHAMP" --out "$OUT/knob_proof.json" > "$OUT/knob_proof.txt" 2>&1 \
    || { say "FATAL knob proof refused: $(tail -1 "$OUT/knob_proof.txt")"; die 71; }
fi
say "knob proof: $(head -c 400 "$OUT/knob_proof.txt" 2>/dev/null | tr '\n' ' ')"
if [ ! -s "$OUT/linkage.txt" ]; then
  LD_LIBRARY_PATH="$CHAMP" bash "$ROOT/scripts/utils/verify_ggml_linkage.sh" "$CHAMP/llama-server" "$CHAMP" > "$OUT/linkage.txt" 2>&1 \
    || { say "FATAL ggml linkage proof failed ($OUT/linkage.txt)"; rm -f "$OUT/linkage.txt"; die 74; }
fi
$LIB serve-files --bindir "$CHAMP" --threads "$THREADS" --cpus "$CPUS" --port "$PORT" --ctx "$CTX" \
    --trunk "$TRUNK" --head "$MTPH" ${UBATCH:+--ubatch "$UBATCH"} --out-prefix "$OUT/$LABEL" \
  || { say "FATAL recipe refused to build the serve command"; die 72; }
mapfile -t ARGV < "$OUT/$LABEL.argv"; mapfile -t ENVV < "$OUT/$LABEL.env"
say "argv: ${ARGV[*]}"
mapfile -t EVICT_FILES < <($LIB files --trunk "$TRUNK" --head "$MTPH")
# PREFLIGHT_ONLY=1: stop here. Everything above (version, knob proof, linkage proof, recipe argv/env) is
# proven without loading a model, evicting a cache or starting a server -- the job's precondition check.
[ "${PREFLIGHT_ONLY:-0}" = 1 ] && { say "PREFLIGHT_ONLY: proofs passed, no server started"; trap - EXIT; exit 0; }

# ---- load, with eviction + placement proof ---------------------------------------------------
for try in $(seq 1 "$TRIES"); do
  { $PY "$HERE/pagecache_drop.py" "${EVICT_FILES[@]}"
    $PY "$ROOT/scripts/utils/numa_evict.py" --target-gib "$EVICT_GIB"; echo "numa_evict rc=$?"; } > "$OUT/$LABEL.evict.$try" 2>&1
  say "try $try evict: $(grep -E '^TOTAL|numa_evict rc|DROP' "$OUT/$LABEL.evict.$try" | tail -2 | tr '\n' ' ')| free/node GiB: $(numactl -H | grep free | awk '{printf "%d ", $4/1024}')"
  ss -ltn "sport = :$PORT" | grep -q LISTEN && { say "FATAL port $PORT busy"; die 64; }
  env -i "${ENVV[@]}" "${ARGV[@]}" > "$OUT/$LABEL.srv.log" 2>&1 &
  SPID=$!; SSTART=$(awk '{print $22}' "/proc/$SPID/stat" 2>/dev/null)
  [ -n "${FILLER_PIDFILE:-}" ] && echo "$SPID" >> "$FILLER_PIDFILE"
  say "try $try server pid=$SPID"
  arm_watchdog                                              # re-armed per server: it must watch THIS pid
  i=0; until curl -sf --max-time 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    sleep 2; i=$((i+1))
    kill -0 "$SPID" 2>/dev/null || { say "FATAL exited during load: $(grep -aiE 'error|invalid|unrecognized' "$OUT/$LABEL.srv.log" | tail -3 | tr '\n' ' ')"; die 61; }
    [ $((i*2)) -ge "$HEALTH_MAX_S" ] && { say "FATAL health timeout"; die 62; }
  done
  say "health OK after $((i*2))s"
  ss -ltnp "sport = :$PORT" 2>/dev/null | grep -q "pid=$SPID," || { say "FATAL :$PORT is not owned by our pid $SPID"; die 73; }
  # linkage: the libggml-cpu the PROCESS mapped must live under the build's own bin dir
  LGG=$(grep -oE "/[^ ]*libggml-cpu\.so[^ ]*" "/proc/$SPID/maps" | sort -u | head -1)
  case "$(readlink -f "$LGG")" in "$(readlink -f "$CHAMP")"/*) say "linkage OK $LGG";; *) say "FATAL linkage '$LGG' not under $CHAMP"; die 65;; esac
  # knob page must NOT be mapped (NO_KNOBS=1): a mapped page would mean the wrong binary
  grep -q "knobs.page" "/proc/$SPID/maps" && { say "FATAL a knob page is mapped but NO_KNOBS=1"; die 69; }
  # live env readback: the knobs the launch relied on are in the PROCESS, not just in the command
  tr '\0' '\n' < "/proc/$SPID/environ" | grep -E '^(GGML_|OMP_|LLAMA_|LD_LIBRARY_PATH)' | sort > "$OUT/$LABEL.environ"
  for kv in GGML_NOHUGEPAGE_PROCESS=1 GGML_IQK=1 GGML_FUSED_DECODE_OFF=1 GGML_FA_SPLIT_KV=0 OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active OMP_DYNAMIC=false; do
    grep -qx "$kv" "$OUT/$LABEL.environ" || { say "FATAL live environ lacks $kv"; die 68; }
  done
  # live affinity: the mask the kernel enforces, not the one we asked for
  AFF=$(awk '/^Cpus_allowed_list:/{print $2}' "/proc/$SPID/status")
  [ "$AFF" = "$CPUS" ] || { say "FATAL live Cpus_allowed_list=$AFF != $CPUS"; die 68; }
  # CHAMP-2 THP shim: THP_enabled 0 = PR_SET_THP_DISABLE in force. Fail closed, once per launch.
  THPEN=$(awk '/^THP_enabled:/{print $2}' "/proc/$SPID/status" 2>/dev/null)
  say "THP_enabled=$THPEN (0 = shim ON)"
  [ "$THPEN" = 0 ] || { say "FATAL shim requested but THP_enabled=$THPEN"; die 67; }
  numastat -p "$SPID" > "$OUT/$LABEL.load.numastat" 2>&1
  PL=$(awk '$1=="Total"{printf "%.2f %.2f %.2f %.2f",$2/1024,$3/1024,$4/1024,$5/1024}' "$OUT/$LABEL.load.numastat")
  say "try $try placement GiB n0..n3: $PL"
  if $PY - "$PL" <<'PY'
import sys
v = [float(x) for x in sys.argv[1].split()]
m = sum(v) / len(v)
d = max(abs(x - m) / m for x in v)
print(f"placement max dev = {d*100:.1f}% (limit 15%)")
sys.exit(0 if d <= 0.15 else 3)
PY
  then say "PLACEMENT OK"; break; fi
  say "PLACEMENT SKEWED on try $try -> re-evicting"; stop_server
  [ "$try" = "$TRIES" ] && { say "FATAL placement never converged"; die 3; }
done
LOADED=$(( $(date +%s)-T0 ))

# ---- the measured arm ------------------------------------------------------------------------
rm -f "$OUT/$LABEL.foreign.jsonl" "$OUT/$LABEL.host.jsonl"
$PY "$ROOT/scripts/utils/foreign_load.py" --label "$LABEL" --server-pid "$SPID" --out "$OUT/$LABEL.foreign.jsonl" \
    --interval 10 --bench-cpus 0-95 >> "$OUT/$LABEL.sampler.log" 2>&1 & CPID=$!
$PY "$HERE/host_load.py" --label "$LABEL" --out "$OUT/$LABEL.host.jsonl" --interval 10 >> "$OUT/$LABEL.sampler.log" 2>&1 & HPID=$!
A0=$(date +%s)
timeout --foreground -k 20 "$CLIENT_MAX_S" $PY "$HERE/client.py" "$LABEL" "$PORT" --out "$OUT" > "$OUT/$LABEL.client.log" 2>&1
CRC=$?
kill -TERM "$CPID" 2>/dev/null; wait "$CPID" 2>/dev/null; CPID=""
kill -TERM "$HPID" 2>/dev/null; wait "$HPID" 2>/dev/null; HPID=""
say "arm done in $(( $(date +%s)-A0 ))s client rc=$CRC  $(tail -1 "$OUT/$LABEL.client.log")"
say "contention: $($PY "$HERE/screen.py" one "$OUT/$LABEL.foreign.jsonl" 2>&1 | tail -1)"
say "host: $($PY "$HERE/screen.py" host "$OUT/$LABEL.host.jsonl" --own-threads "$THREADS" 2>&1 | tail -1)"
numastat -p "$SPID" > "$OUT/$LABEL.end.numastat" 2>&1
THPEND=$(awk '/^THP_enabled:/{print $2}' "/proc/$SPID/status" 2>/dev/null)
$LIB meta --out "$OUT/$LABEL.meta.json" label="$LABEL" threads="$THREADS" cpus="$CPUS" champ="$CHAMP" version="$VER" \
    trunk="$TRUNK" mtp_head="$MTPH" ctx="$CTX" port="$PORT" pid="$SPID" load_s="$LOADED" client_rc="$CRC" \
    thp_enabled_end="$THPEND" live_affinity="$AFF" placement_load_gib="$PL" started="$(date -u -d "@$T0" +%FT%TZ)" >/dev/null
stop_server
[ "$CRC" = 0 ] && [ -f "$OUT/$LABEL.rows.jsonl" ] || { say "FATAL launch incomplete (client rc=$CRC)"; die 60; }
die 0
