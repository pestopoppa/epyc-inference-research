#!/bin/bash
# C46: two -j64 builds of the DS41 anchor source, CPU recipe, aborting if the CPU window leaves `open`.
set -euo pipefail
SRC=/mnt/raid0/llm/llama.cpp-experimental-fastload-ds41-20260925
WANT=c0ef3961327b4036e7e5950ffba7487df51e4c25
WIN=/mnt/raid0/llm/autokernel/cpu-window.json
ROOT=/mnt/raid0/llm/tmp/c46-repro
CPUS=0-39,96-135
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
winstate() { python3 -c "import json;print(json.load(open('$WIN'))['state'])"; }
check_src() {
  local h; h=$(git -C "$SRC" rev-parse HEAD)
  [[ "$h" == "$WANT" ]] || { log "SRC HEAD moved to $h"; exit 3; }
  [[ -z "$(git -C "$SRC" status --porcelain --untracked-files=no)" ]] || { log "SRC dirty"; exit 3; }
}
run_guarded() {  # run "$@" in its own process group; kill it if the window leaves open
  setsid "$@" & local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    s=$(winstate || echo unknown)
    if [[ "$s" != "open" ]]; then
      log "window state=$s -> aborting pgid $pid"; kill -TERM -- -"$pid" 2>/dev/null || true
      sleep 5; kill -KILL -- -"$pid" 2>/dev/null || true; wait "$pid" || true; exit 4
    fi
    sleep 5
  done
  wait "$pid"
}
margin=$(python3 -c "
import json,datetime as D
d=json.load(open('$WIN'))
c=D.datetime.fromisoformat(d['est_close_at'].replace('Z','+00:00'))
print(d['state'], int((c-D.datetime.now(D.timezone.utc)).total_seconds()))")
log "window at start: $margin"
read st secs <<<"$margin"
[[ "$st" == open && "$secs" -ge 1500 ]] || { log "window not open with >=25min margin; not starting"; exit 5; }
for tag in a b; do
  B=$ROOT/j64-$tag
  check_src
  log "configure $B"
  run_guarded taskset -c $CPUS cmake -S "$SRC" -B "$B" -DCMAKE_BUILD_TYPE=Release \
    -DGGML_HIP=OFF -DGGML_NATIVE=ON -DGGML_OPENMP=ON \
    -DCMAKE_C_COMPILER=/usr/bin/gcc-15 -DCMAKE_CXX_COMPILER=/usr/bin/g++-15 > "$ROOT/configure-$tag.log" 2>&1
  log "build $B -j64"
  t0=$(date +%s)
  run_guarded taskset -c $CPUS cmake --build "$B" -j 64 --target llama-bench --target test-backend-ops \
    --target llama-cli --target llama-server > "$ROOT/build-$tag.log" 2>&1
  log "build $tag done in $(( $(date +%s) - t0 ))s"
  check_src
done
log "ALL DONE"
