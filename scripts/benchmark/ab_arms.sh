#!/bin/bash
# GPU champion A/B arm runner, committed copy (restored 2026-10-20 as ab-probe-restore).
# Derived from the 2026-10-04 FOLD COPY (epyc-root artifacts/champion-fold-kvu19-20261004/receipt-ab/ab_arms_fold.sh), itself
# a copy of the workspace-8d /mnt/raid0/llm/tmp/gpu-champion-ab/ab_arms.sh (V6R-4d.2). Changes vs the fold copy, nothing else:
#   * HERE is this script's own directory (ab_probe.py sits beside it), not the scratch dir that was lost;
#   * the P arm may be any store build via P_BUILD / P_TIP / P_BUILDNO (default: production/gpu = v10 ffc1bac82, as before);
#     with P_BUILD set the production-link expectation is skipped, because P is then the champion, not production;
#   * the HIP library file name is HIPLIB (default libggml-hip.so.0.16.0, as before).
# The C arm comes from CAND_TIP / CAND_BUILD / CAND_BUILDNO (fold-copy change, kept).
# GPU champion A/B arm runner (workspace-8d, 2026-09-29). Called by vision_ab.sh and w8083_bundle.sh;
# not meant to be run by hand, though it can be (it never touches a production port).
#
# Runs a SEQUENCE of arms (e.g. "P C P C") on ONE spare port, one llama-server at a time:
#   P = production v10 GPU  = readlink -f /mnt/raid0/llm/kernels/production/gpu   (10303, ffc1bac82)
#   C = champion HIP build  = /mnt/raid0/llm/kernels/builds/gpu-20260929-90c12df42/bin (10308, 90c12df42)
# Per arm: launch with production's argv (only --port and --slot-save-path changed), its OWN
# LD_LIBRARY_PATH (the arm's bin dir first, then production's non-ggml entries), host threads pinned
# to ARM_CPUSET (default 72-79 = NUMA node 3 inside 48-87, mirroring production's --membind=3; never
# 88-95 / 184-191) -> prove: exe + sha256, verify_ggml_linkage.sh, the GPU device in the startup log,
# /proc/<pid>/maps shows libggml-hip from the arm's own dir, thread placement (below), /props
# build_info names the arm's commit -> residency sampler (KFD pid list + per-pid VRAM from
# /sys/class/kfd, total VRAM + gpu_busy from sysfs, every 2 s) runs DURING the arm -> ab_probe.py ->
# TERM/KILL/verify the captured PID -> optional llama-bench with the same env/placement (its own
# residency samples).
#
# LIVE-CHECK SEMANTICS (hardened 2026-09-29 after the :8083 window failed 3x on over-strict checks;
# every check is an `ab_probe.py <check>` subcommand that prints what it SAW and is testable read-only
# against a production server -- see selftest/live_checks.sh):
#   device     = the pid holds >= 1 GB VRAM in KFD (/sys/class/kfd). Startup-log device lines are
#                RECORDED, not required: v10 at verbosity 3 prints none (only a sampler warning).
#   placement  = every thread inside ARM_CPUSET EXCEPT recognized ROCm runtime threads, which are
#                reported with name/CPUs/wchan. On this host ROCr's async-event thread (wchan
#                kfd_wait_on_events) always gets its affinity reset (0-191) -- seen in all four live GPU
#                processes -- and all threads are named after the executable, so it is recognized by
#                wait channel. Checked at verify (dies) and again AFTER the probe, when the OMP compute
#                pool exists (it is created at the first compute, not at load) -> meta.affinity_post,
#                gated in the summary.
#   /props     = build_info names the commit, model_path is the -m argument, modalities.vision for the
#                vision kind. speculative.* in /props is per-request DEFAULTS ('none' on production
#                :8083 despite --spec-type draft-mtp): recorded, never gated.
#   watcher    = --watch-mode strict: window_gate.py --mid-run (window must stay "open").
#                --watch-mode paused: for a run inside a PEER-PAUSED loop (window reads closed/exited
#                for the whole run -- strict mode would abort within 5 s): abort when the loop comes
#                back (state/phase change vs the start snapshot, loop_holds_claim, cpuset overlap).
#
# USAGE
#   ab_arms.sh --kind vision|mtp --run DIR --args-file F --model GGUF [--order "P C P C"]
#              [--llama-bench "ARGS"] [--watch-window] [--deadline-epoch N] [--est-arm-s N]
#              [--probe-args "ARGS"] [--watch-mode strict|paused] [--dry-run]
# EXIT 0 all arms ran; 1 error; 3 aborted by the window watcher; 4 deadline cut the sequence short
#      (only if it left no P or no C arm).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROD_LINK="/mnt/raid0/llm/kernels/production/gpu"
PROD_EXPECT_DIR="/mnt/raid0/llm/kernels/builds/gpu-20260921-ffc1bac82/bin"
P_BUILD="${P_BUILD:-}"                                   # store build dir for the P arm (default: production/gpu)
P_TIP="${P_TIP:-ffc1bac82}"; P_BUILDNO="${P_BUILDNO:-10303}"
HIPLIB="${HIPLIB:-libggml-hip.so.0.16.0}"
CAND_TIP="${CAND_TIP:-1bceceb0514d423ad220888313eafcddc30e485e}"     # FOLD COPY: candidate from env (default: fold 1bceceb05)
CAND_BUILD="${CAND_BUILD:-/mnt/raid0/llm/kernels/builds/gpu-20261004-1bceceb05}"
CAND_BUILDNO="${CAND_BUILDNO:-10311}"
CHAMP_DIR="$CAND_BUILD/bin"
LINKAGE="/mnt/raid0/llm/epyc-inference-research/scripts/utils/verify_ggml_linkage.sh"
GATE="/mnt/raid0/llm/tmp/champion-sidecar/window_gate.py"
WINDOW_FILE="${AB_WINDOW_FILE:-/mnt/raid0/llm/autokernel/cpu-window.json}"   # override: selftest only
PORT="${AB_PORT:-8197}"
ARM_CPUSET="${ARM_CPUSET:-72-79}"
ARM_MEMBIND="${ARM_MEMBIND:-3}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-600}"
SAMPLE_S="${SAMPLE_S:-2}"

KIND=""; RUN=""; ARGS_FILE=""; MODEL=""; ORDER="P C P C"; LB_ARGS=""; WATCH=0; DEADLINE=0
EST_ARM_S=120; PROBE_ARGS=""; DRY=0; WATCH_MODE=strict
while [ $# -gt 0 ]; do
  case "$1" in
    --kind) KIND="$2"; shift;;
    --run) RUN="$2"; shift;;
    --args-file) ARGS_FILE="$2"; shift;;
    --model) MODEL="$2"; shift;;
    --order) ORDER="$2"; shift;;
    --llama-bench) LB_ARGS="$2"; shift;;
    --watch-window) WATCH=1;;
    --deadline-epoch) DEADLINE="$2"; shift;;
    --est-arm-s) EST_ARM_S="$2"; shift;;
    --probe-args) PROBE_ARGS="$2"; shift;;
    --watch-mode) WATCH_MODE="$2"; shift;;
    --dry-run) DRY=1;;
    *) echo "ab_arms: unknown argument $1" >&2; exit 1;;
  esac
  shift
done
[ -n "$KIND" ] && [ -n "$RUN" ] && [ -n "$ARGS_FILE" ] && [ -n "$MODEL" ] || { echo "ab_arms: missing args" >&2; exit 1; }
case "$WATCH_MODE" in strict|paused) ;; *) echo "ab_arms: --watch-mode must be strict|paused" >&2; exit 1;; esac
mkdir -p "$RUN"
LOG="$RUN/ab_arms.log"

ts() { date -u +%H:%M:%SZ; }
say() { echo "[ab $(ts)] $*" | tee -a "$LOG"; }
die() { say "ERROR: $*"; exit 1; }
show() { printf '%q ' "$@"; echo; }

# ── placement sanity: ARM_CPUSET must lie inside 48-87 (DS41: no 88-95; not 184-191) ───────────
python3 - "$ARM_CPUSET" <<'EOF' || die "ARM_CPUSET $ARM_CPUSET is not inside 48-87"
import sys
s = set()
for p in sys.argv[1].split(","):
    a, _, b = p.partition("-"); s.update(range(int(a), int(b or a) + 1))
sys.exit(0 if s and s <= set(range(48, 88)) else 1)
EOF

mapfile -d '' SERVER_ARGS < "$ARGS_FILE"
[ "${#SERVER_ARGS[@]}" -gt 4 ] || die "args file $ARGS_FILE is empty"
SERVER_MODEL=""; SLOT_DIR=""
for ((i = 0; i + 1 < ${#SERVER_ARGS[@]}; i++)); do
  case "${SERVER_ARGS[$i]}" in
    -m|--model) SERVER_MODEL="${SERVER_ARGS[$((i + 1))]}";;
    --slot-save-path) SLOT_DIR="${SERVER_ARGS[$((i + 1))]}";;
  esac
done
[ -n "$SERVER_MODEL" ] || die "args file $ARGS_FILE has no -m"
PROPS_VISION=(); [ "$KIND" = vision ] && PROPS_VISION=(--vision)
listening() { local o; o="$(ss -ltnH "( sport = :$1 )" 2>/dev/null || true)"; [ -n "$o" ]; }

arm_dir() {
  case "$1" in
    P) if [ -n "$P_BUILD" ]; then echo "$P_BUILD/bin"; else readlink -f "$PROD_LINK"; fi;;
    C) echo "$CHAMP_DIR";;
  esac
}
arm_commit() { case "$1" in P) echo "${P_TIP:0:9}";; C) echo "${CAND_TIP:0:7}";; esac; }
arm_build() { case "$1" in P) echo "$P_BUILDNO";; C) echo "$CAND_BUILDNO";; esac; }

arm_env() { # prints nothing; fills ENV_ARR for arm $1
  local dir; dir="$(arm_dir "$1")"
  ENV_ARR=(env -i "HOME=${HOME:-/home/node}" "PATH=/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin" "LANG=C.UTF-8" "TZ=UTC"
    "HIP_PATH=/opt/rocm" "ROCM_PATH=/opt/rocm"
    "LD_LIBRARY_PATH=$dir:/opt/rocm/lib:/usr/lib/llvm-20/lib:/opt/AMD/aocc-compiler-5.0.0/lib:/opt/rocm/lib"
    OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active OMP_DYNAMIC=false KMP_BLOCKTIME=10 GGML_IQK=1)
}
NUMA=(numactl "--membind=$ARM_MEMBIND" -- taskset -c "$ARM_CPUSET")

CARD=""
for c in /sys/class/drm/card*/device; do
  [ -f "$c/mem_info_vram_used" ] && { CARD="$c"; break; }
done
[ -n "$CARD" ] || die "no amdgpu sysfs card with mem_info_vram_used"

# ── preflight per build: the store link, sha256 vs SHA256SUMS, linkage under the arm's own env ───
[ -n "$P_BUILD" ] || [ "$(readlink -f "$PROD_LINK")" = "$PROD_EXPECT_DIR" ] \
  || die "kernels/production/gpu resolves to $(readlink -f "$PROD_LINK"), expected $PROD_EXPECT_DIR"
declare -A BIN_SHA LB_SHA HIP_SHA
for A in P C; do
  case " $ORDER " in *" $A "*) ;; *) continue;; esac   # FOLD COPY: only arms that run
  d="$(arm_dir "$A")"; sums="$(dirname "$d")/SHA256SUMS"
  for f in llama-server llama-bench "$HIPLIB"; do
    want="$(awk -v f="bin/$f" -v g="./$f" '$2==f||$2==g{print $1}' "$sums")"   # FOLD COPY: store builds list ./<file>
    got="$(sha256sum "$d/$f" | awk '{print $1}')"
    [ -n "$want" ] && [ "$want" = "$got" ] || die "arm $A: $f sha256 $got != SHA256SUMS $want"
    case "$f" in llama-server) BIN_SHA[$A]="$got";; llama-bench) LB_SHA[$A]="$got";; *) HIP_SHA[$A]="$got";; esac
  done
  arm_env "$A"
  out="$("${ENV_ARR[@]}" bash "$LINKAGE" "$d/llama-server" "$d" 2>&1)" || { echo "$out" >> "$LOG"; die "arm $A: verify_ggml_linkage.sh failed"; }
  echo "$out" > "$RUN/linkage-$A.txt"
  say "arm $A: $d  llama-server ${BIN_SHA[$A]:0:12}  linkage $(echo "$out" | grep -m1 '^PASS' || echo '?')"
done

# ── process helpers: only PIDs this script captured ──────────────────────────────────────────
ARM_PID=""; BENCH_PID=""; SAMPLER_PID=""; WATCH_PID=""; PROBE_PID=""
stop_pid() { # pid label
  local pid="$1" label="$2" i
  [ -n "$pid" ] && [ -d "/proc/$pid" ] || return 0
  kill -TERM "$pid" 2>/dev/null || true
  for i in $(seq 1 30); do ps -p "$pid" >/dev/null 2>&1 || break; sleep 1; done
  if ps -p "$pid" >/dev/null 2>&1; then
    say "$label PID $pid survived SIGTERM 30 s; SIGKILL"
    kill -KILL "$pid" 2>/dev/null || true
    for i in $(seq 1 15); do ps -p "$pid" >/dev/null 2>&1 || break; sleep 1; done
  fi
  ps -p "$pid" >/dev/null 2>&1 && { say "FATAL: $label PID $pid STILL alive after SIGKILL"; return 1; }
  say "$label PID $pid verified dead"
}
stop_quiet() { local p="$1"; [ -n "$p" ] && kill "$p" 2>/dev/null || true; [ -n "$p" ] && wait "$p" 2>/dev/null || true; }
cleanup() {
  local rc=$?
  stop_quiet "$WATCH_PID"; WATCH_PID=""
  stop_quiet "$PROBE_PID"; PROBE_PID=""
  stop_pid "$BENCH_PID" "llama-bench" || rc=1; BENCH_PID=""
  stop_pid "$ARM_PID" "arm server" || rc=1; ARM_PID=""
  stop_quiet "$SAMPLER_PID"; SAMPLER_PID=""
  if listening "$PORT"; then say "WARNING: port $PORT still has a listener"; fi
  if [ ! -f "$RUN/result.json" ] && ls "$RUN"/arm-*.meta.json >/dev/null 2>&1; then
    say "partial run: summarizing the arms that completed"
    python3 "$HERE/ab_probe.py" summarize --run "$RUN" --kind "$KIND" 2>&1 | tee -a "$LOG" || true
  fi
  exit "$rc"
}
on_term() { say "ABORT: $(cat "$RUN/ABORT" 2>/dev/null || echo signal)"; exit 3; }
if [ "$DRY" -eq 0 ]; then trap cleanup EXIT; trap on_term TERM INT; fi

int_or0() { case "$1" in ''|*[!0-9]*) echo 0;; *) echo "$1";; esac; }
sampler() { # pid outfile -- runs in background until killed
  # set +e: a failed read (process exiting, EBUSY on gpu_busy_percent) must not silently kill the
  # sampler -- a dead sampler reads as "not resident" and fails the arm on a phantom
  set +e +o pipefail
  local pid="$1" out="$2" kfd pv v used busy
  while :; do
    kfd="$(ls /sys/class/kfd/kfd/proc 2>/dev/null | tr '\n' ' ')"
    pv=0
    if [ -d "/sys/class/kfd/kfd/proc/$pid" ]; then
      for v in /sys/class/kfd/kfd/proc/"$pid"/vram_*; do pv=$(( pv + $(int_or0 "$(cat "$v" 2>/dev/null)") )); done
    fi
    used="$(int_or0 "$(cat "$CARD/mem_info_vram_used" 2>/dev/null)")"
    busy="$(int_or0 "$(cat "$CARD/gpu_busy_percent" 2>/dev/null)")"
    printf '{"t":"%s","pid":%s,"kfd_pids":[%s],"pid_vram_bytes":%s,"vram_used_bytes":%s,"gpu_busy_pct":%s}\n' \
      "$(date -u +%FT%TZ)" "$pid" "$(for k in $kfd; do printf '"%s",' "$k"; done | sed 's/,$//')" \
      "$pv" "$used" "$busy" >> "$out"
    sleep "$SAMPLE_S"
  done
}

watch_check() { # prints the verdict; rc 0 = keep going
  if [ "$WATCH_MODE" = paused ]; then
    python3 "$HERE/ab_probe.py" window-watch --baseline "$RUN/window_watch_baseline.json" --window "$WINDOW_FILE" --cpuset "$ARM_CPUSET" 2>&1
  else
    python3 "$GATE" --mid-run --need-s 0 --cpuset "$ARM_CPUSET" --window "$WINDOW_FILE" 2>&1
  fi
}
watcher() { # aborts the run the moment the AutoKernel window leaves "open" (strict) / the loop returns (paused)
  set +e +o pipefail
  local main="$1" out rc unreadable=0
  while :; do
    out="$(watch_check)"; rc=$?
    # rc 2 = window file unreadable (e.g. caught mid-rewrite): act on two consecutive, not one
    if [ "$rc" -eq 2 ]; then unreadable=$((unreadable + 1)); else unreadable=0; fi
    [ "$rc" -eq 0 ] || { [ "$rc" -eq 2 ] && [ "$unreadable" -lt 2 ]; } || {
      echo "window watcher ($WATCH_MODE) refused (rc=$rc): $out" > "$RUN/ABORT"
      [ -n "$(cat "$RUN/arm.pid" 2>/dev/null)" ] && kill -TERM "$(cat "$RUN/arm.pid")" 2>/dev/null || true
      kill -TERM "$main" 2>/dev/null || true
      exit 0
    }
    sleep 5
  done
}

wait_healthy() { # pid log
  local pid="$1" log="$2" t0 exe want h i
  t0=$(date +%s); want="$(readlink -f "$BIN")"; exe=""
  # the exec chain setsid -> env -> numactl -> taskset -> llama-server keeps ONE pid; poll until the
  # pid IS the server (or died: a bad argument exits in < 1 s and must be reported as that, not as
  # "wrong exe" -- 2026-09-29 07:05Z, the missing slot dir)
  for i in $(seq 1 50); do
    if [ ! -d "/proc/$pid" ]; then
      tail -30 "$log" >> "$LOG"
      die "arm PID $pid exited during startup; server log says: $(grep -av '^\s*$' "$log" | head -3 | tr '\n' ' ' | cut -c1-300)"
    fi
    exe="$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)"
    [ "$exe" = "$want" ] && break
    sleep 0.2
  done
  [ "$exe" = "$want" ] || { tail -20 "$log" >> "$LOG"; die "PID $pid exe is '$exe' after 10 s, expected $want"; }
  while :; do
    [ -d "/proc/$pid" ] || { tail -30 "$log" >> "$LOG"; die "arm PID $pid died during load; log tail: $(tail -3 "$log" | tr '\n' ' ' | cut -c1-300)"; }
    [ -f "$RUN/ABORT" ] && exit 3
    h="$(curl -s -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null || true)"   # no pipe: grep -q + pipefail = SIGPIPE races
    if [[ "$h" == *'"ok"'* ]]; then
      LOAD_S=$(( $(date +%s) - t0 )); return 0
    fi
    if [ $(( $(date +%s) - t0 )) -ge "$HEALTH_TIMEOUT_S" ]; then
      tail -30 "$log" >> "$LOG"; die "health timeout ${HEALTH_TIMEOUT_S}s; last /health reply: ${h:-<none>}"
    fi
    sleep 2
  done
}

check() { # name outfile -- cmd...: run an ab_probe.py check, keep its JSON, echo its summary; rc = check rc
  local name="$1" out="$2" rc=0; shift 2
  "$@" > "$out" 2>> "$LOG" || rc=$?
  CHECK_SUMMARY="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("summary", "?"))' "$out" 2>/dev/null \
    || head -c 300 "$out")" || CHECK_SUMMARY="<no summary>"
  return "$rc"
}

verify_arm() { # arm pid log tag -> writes meta json
  local A="$1" pid="$2" log="$3" tag="$4" dir maps bad rc
  dir="$(arm_dir "$A")"
  # 1. GPU device: KFD VRAM of the pid (log lines recorded only)
  rc=0; check devlog "$RUN/arm-$tag.device.json" python3 "$HERE/ab_probe.py" devlog --pid "$pid" --log "$log" || rc=$?
  say "arm $tag device: $CHECK_SUMMARY"
  [ "$rc" -eq 0 ] || die "arm $tag: no GPU device evidence: $CHECK_SUMMARY"
  # 2. own ggml/llama libs mapped
  maps="$(grep -E 'libggml|libllama|libmtmd' "/proc/$pid/maps" | awk '{print $6}' | sort -u || true)"
  bad="$(printf '%s\n' "$maps" | grep . | grep -v "^$dir/" || true)"
  [ -z "$bad" ] || die "arm $tag: maps a ggml/llama lib outside $dir: $bad"
  [[ "$maps" == *"$dir/libggml-hip.so"* ]] || die "arm $tag: libggml-hip.so from $dir is NOT mapped (not a HIP run); mapped: $(echo $maps)"
  printf '%s\n' "$maps" > "$RUN/arm-$tag.maps.txt"
  # 3. placement (pre-probe; the compute pool may not exist yet -- re-checked after the probe)
  rc=0; check affinity "$RUN/arm-$tag.affinity-pre.json" python3 "$HERE/ab_probe.py" affinity --pid "$pid" --cpuset "$ARM_CPUSET" || rc=$?
  say "arm $tag placement (pre-probe): $CHECK_SUMMARY"
  [ "$rc" -eq 0 ] || die "arm $tag: thread placement: $CHECK_SUMMARY"
  # 4. /props
  rc=0; check props "$RUN/arm-$tag.props.json" python3 "$HERE/ab_probe.py" props --port "$PORT" \
    --commit "$(arm_commit "$A")" --model "$SERVER_MODEL" "${PROPS_VISION[@]}" || rc=$?
  say "arm $tag /props: $CHECK_SUMMARY"
  [ "$rc" -eq 0 ] || die "arm $tag: /props: $CHECK_SUMMARY"
  # 5. meta (python reads every value from files/argv: nothing is spliced into source text)
  python3 - "$RUN" "$tag" "$A" "$pid" "$PORT" "$dir" "${BIN_SHA[$A]}" "${HIP_SHA[$A]}" "$(arm_commit "$A")" \
    "$(arm_build "$A")" "$LOAD_S" "$ARM_CPUSET" "$ARM_MEMBIND" "$log" "$ARM_T0" <<'PYEOF'
import json, sys
(run, tag, arm, pid, port, d, bsha, hsha, commit, build, load_s, cpuset, membind, log, t0) = sys.argv[1:]
def j(name):
    try: return json.load(open(f"{run}/arm-{tag}.{name}.json"))
    except (OSError, ValueError): return None
dev, props, aff = j("device"), j("props"), j("affinity-pre")
maps = open(f"{run}/arm-{tag}.maps.txt").read().split()
omp = []
try: omp = [l.split()[-1] for l in open(f"/proc/{pid}/maps") if "libomp" in l or "libgomp" in l][:1]
except OSError: pass
json.dump({
  "tag": tag, "arm": arm, "pid": int(pid), "port": int(port), "bin_dir": d,
  "binary_sha256": bsha, "libggml_hip_sha256": hsha, "expected_commit": commit, "expected_build": build,
  "props_build_info": (props or {}).get("build_info"), "props": props, "load_s": int(load_s),
  "cpuset": cpuset, "membind": membind,
  "cmdline": open(f"/proc/{pid}/cmdline", "rb").read().decode().split("\0")[:-1],
  "device_evidence": dev, "startup_log_device_lines": (dev or {}).get("startup_log_device_lines", []),
  "runtime_maps_ok": True, "mapped_libs": maps, "omp_lib": omp,
  "affinity_pre": aff, "affinity_post": None,
  "log": log, "started_epoch": int(t0),
}, open(f"{run}/arm-{tag}.meta.json", "w"), indent=1)
PYEOF
  say "arm $tag verified: GPU (KFD VRAM), own libggml-hip mapped, placement OK, build_info=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("build_info"))' "$RUN/arm-$tag.props.json"), load ${LOAD_S}s"
}

affinity_post() { # arm-tag pid: placement AFTER the probe (compute pool exists now); recorded + gated in summary
  local tag="$1" pid="$2" rc=0
  check affinity "$RUN/arm-$tag.affinity-post.json" python3 "$HERE/ab_probe.py" affinity --pid "$pid" --cpuset "$ARM_CPUSET" || rc=$?
  say "arm $tag placement (post-probe): $CHECK_SUMMARY"
  python3 - "$RUN/arm-$tag.meta.json" "$RUN/arm-$tag.affinity-post.json" <<'PYEOF' || say "WARNING: could not fold affinity_post into meta"
import json, sys
m = json.load(open(sys.argv[1]))
try: m["affinity_post"] = json.load(open(sys.argv[2]))
except (OSError, ValueError) as e: m["affinity_post"] = {"ok": False, "summary": f"affinity check produced no JSON: {e}"}
json.dump(m, open(sys.argv[1], "w"), indent=1)
PYEOF
  return 0
}

run_llama_bench() { # arm tag
  local A="$1" tag="$2" dir out
  dir="$(arm_dir "$A")"
  arm_env "$A"
  # shellcheck disable=SC2206
  local lb=("$dir/llama-bench" -m "$MODEL" $LB_ARGS -o json)
  out="$RUN/arm-$tag.llama-bench.raw.json"
  if [ "$DRY" -eq 1 ]; then echo "[dry-run] $(show "${ENV_ARR[@]}" "${NUMA[@]}" "${lb[@]}") > $out"; return 0; fi
  "${ENV_ARR[@]}" "${NUMA[@]}" "${lb[@]}" > "$out" 2> "$RUN/arm-$tag.llama-bench.log" < /dev/null &
  BENCH_PID=$!
  local bp="$BENCH_PID"
  echo "$BENCH_PID" > "$RUN/arm.pid"
  sampler "$BENCH_PID" "$RUN/arm-$tag.llama-bench.residency.jsonl" & SAMPLER_PID=$!
  local rc=0; wait "$BENCH_PID" || rc=$?
  BENCH_PID=""; stop_quiet "$SAMPLER_PID"; SAMPLER_PID=""
  [ "$rc" -eq 0 ] || die "arm $tag llama-bench rc=$rc (see $RUN/arm-$tag.llama-bench.log)"
  grep -aqE 'ROCm|MI210' "$RUN/arm-$tag.llama-bench.log" "$out" || die "arm $tag llama-bench: no ROCm0 device in its output"
  python3 - "$out" "$RUN/arm-$tag.llama-bench.json" "$tag" <<'EOF'
import json, sys
rows = json.load(open(sys.argv[1]))
pp = [r["avg_ts"] for r in rows if r.get("n_prompt", 0) > 0 and r.get("n_gen", 0) == 0]
tg = [r["avg_ts"] for r in rows if r.get("n_gen", 0) > 0 and r.get("n_prompt", 0) == 0]
dev = sorted({r.get("devices") or r.get("gpu_info") or "" for r in rows})
json.dump({"pp": pp[0] if pp else None, "tg": tg[0] if tg else None, "devices": dev,
           "build_commit": rows[0].get("build_commit") if rows else None, "rows": rows},
          open(sys.argv[2], "w"), indent=1)
print(f"llama-bench {sys.argv[3]}: pp={pp} tg={tg} devices={dev}")
EOF
  python3 - "$RUN/arm-$tag.meta.json" "$bp" <<'EOF'
import json, sys
p = sys.argv[1]; m = json.load(open(p)); m["bench_pid"] = int(sys.argv[2])
json.dump(m, open(p, "w"), indent=1)
EOF
}

# ── the sequence ────────────────────────────────────────────────────────────────────────────
if [ "$WATCH" -eq 1 ]; then
  cp "$WINDOW_FILE" "$RUN/window_watch_baseline.json" 2>/dev/null \
    || { [ "$WATCH_MODE" = strict ] || die "paused watch mode needs a window-file baseline"; }
  wout="$(watch_check)" && wrc=0 || wrc=$?
  say "window watcher ($WATCH_MODE) at start: rc=$wrc $(echo "$wout" | head -1)"
  if [ "$DRY" -eq 1 ]; then
    echo "[dry-run] window watcher ($WATCH_MODE): every 5 s $([ "$WATCH_MODE" = paused ] && show python3 "$HERE/ab_probe.py" window-watch --baseline "$RUN/window_watch_baseline.json" --cpuset "$ARM_CPUSET" || show python3 "$GATE" --mid-run --need-s 0 --cpuset "$ARM_CPUSET") ; on refusal -> ABORT + TERM the arm and this runner"
    [ "$wrc" -eq 0 ] || echo "[dry-run] NOTE: the $WATCH_MODE watcher REFUSES right now -> a real run would abort within 5 s"
  else
    [ "$wrc" -eq 0 ] || die "window watcher ($WATCH_MODE) refuses before the first arm: $wout"
    watcher "$$" & WATCH_PID=$!
  fi
fi
say "sequence '$ORDER' kind=$KIND port=$PORT cpuset=$ARM_CPUSET membind=$ARM_MEMBIND llama-bench='${LB_ARGS:-off}' model=$SERVER_MODEL slot_dir=${SLOT_DIR:-none}"
idx=0; done_P=0; done_C=0
for A in $ORDER; do
  idx=$((idx + 1)); tag="${idx}${A}"
  if [ "$DEADLINE" -gt 0 ] && [ $(( $(date +%s) + EST_ARM_S )) -gt "$DEADLINE" ]; then
    say "deadline: skipping arm $tag and the rest (est ${EST_ARM_S}s per arm)"; break
  fi
  BIN="$(arm_dir "$A")/llama-server"
  arm_env "$A"
  alog="$RUN/arm-$tag.server.log"
  if [ "$DRY" -eq 1 ]; then
    echo "[dry-run] arm $tag launch: setsid $(show "${ENV_ARR[@]}" "${NUMA[@]}" "$BIN" "${SERVER_ARGS[@]}") > $alog"
    echo "[dry-run] arm $tag: wait for exe==$BIN then /health; verify: ab_probe.py devlog (KFD VRAM >= 1 GB; log lines recorded), own libggml-hip in /proc/<pid>/maps, ab_probe.py affinity --cpuset $ARM_CPUSET (ROCm runtime threads reported, not failed), ab_probe.py props --commit $(arm_commit "$A") --model $SERVER_MODEL ${PROPS_VISION[*]}; residency sampler every ${SAMPLE_S}s"
    echo "[dry-run] arm $tag probe: $(show python3 "$HERE/ab_probe.py" probe "$KIND" --port "$PORT" --out "$RUN/arm-$tag.probe.json") $PROBE_ARGS"
    echo "[dry-run] arm $tag post-probe: ab_probe.py affinity again (OMP pool now exists) -> meta.affinity_post"
    echo "[dry-run] arm $tag stop: kill -TERM <captured pid>; 30 s; kill -KILL; ps -p verify"
    [ -n "$LB_ARGS" ] && run_llama_bench "$A" "$tag"
    continue
  fi
  [ -f "$RUN/ABORT" ] && exit 3
  if listening "$PORT"; then die "port $PORT already has a listener: $(ss -ltnpH "( sport = :$PORT )" 2>/dev/null || true)"; fi
  [ -z "$SLOT_DIR" ] || mkdir -p "$SLOT_DIR"   # llama-server refuses a --slot-save-path that is not a directory
  ARM_T0=$(date +%s)
  say "arm $tag: launching $BIN"
  setsid "${ENV_ARR[@]}" "${NUMA[@]}" "$BIN" "${SERVER_ARGS[@]}" > "$alog" 2>&1 < /dev/null &
  ARM_PID=$!
  echo "$ARM_PID" > "$RUN/arm.pid"
  sampler "$ARM_PID" "$RUN/arm-$tag.residency.jsonl" & SAMPLER_PID=$!
  LOAD_S=0
  wait_healthy "$ARM_PID" "$alog"
  verify_arm "$A" "$ARM_PID" "$alog" "$tag"
  # shellcheck disable=SC2086
  python3 "$HERE/ab_probe.py" probe "$KIND" --port "$PORT" --out "$RUN/arm-$tag.probe.json" $PROBE_ARGS \
    >> "$LOG" 2>&1 < /dev/null &
  PROBE_PID=$!
  prc=0; wait "$PROBE_PID" || prc=$?; PROBE_PID=""
  tail -1 "$LOG"
  affinity_post "$tag" "$ARM_PID"
  stop_pid "$ARM_PID" "arm $tag server"; ARM_PID=""; : > "$RUN/arm.pid"
  stop_quiet "$SAMPLER_PID"; SAMPLER_PID=""
  [ "$prc" -eq 0 ] || die "arm $tag probe failed (rc=$prc); see $RUN/arm-$tag.probe.json"
  if [ -n "$LB_ARGS" ]; then run_llama_bench "$A" "$tag"; fi
  case "$A" in P) done_P=$((done_P + 1));; C) done_C=$((done_C + 1));; esac
  say "arm $tag done ($(( $(date +%s) - ARM_T0 )) s)"
done

if [ "$DRY" -eq 1 ]; then echo "[dry-run] summarize: $(show python3 "$HERE/ab_probe.py" summarize --run "$RUN" --kind "$KIND")"; exit 0; fi
python3 "$HERE/ab_probe.py" summarize --run "$RUN" --kind "$KIND" | tee -a "$LOG"
[ "$done_P" -ge 1 ] && [ "$done_C" -ge 1 ] || exit 4
exit 0
