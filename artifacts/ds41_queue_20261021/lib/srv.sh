#!/bin/bash
# Sourced by the INNER scripts. Server lifecycle for ONE DS41-class llama-server at a time (captured PID only).
# Needs: RUNDIR, FILLER_END (epoch), ds41_guard.sh sourced.
SRV_PID=""
time_left() { echo $(( ${FILLER_END:-0} - $(date -u +%s) )); }
stopped() { [ -e "${STOPFILE:-/nonexistent}" ]; }
kill_srv() {   # captured PID only; TERM -> KILL; verify dead
  [ -n "${SRV_PID:-}" ] || return 0
  kill -TERM "$SRV_PID" 2>/dev/null || true
  for _ in $(seq 1 90); do kill -0 "$SRV_PID" 2>/dev/null || { echo "server $SRV_PID dead (verified) $(date -u +%T)"; SRV_PID=""; return 0; }; sleep 1; done
  kill -KILL "$SRV_PID" 2>/dev/null || true; sleep 3
  if kill -0 "$SRV_PID" 2>/dev/null; then echo "STILL ALIVE server $SRV_PID"; else echo "server $SRV_PID dead after KILL (verified)"; fi
  SRV_PID=""
}
wait_ready() { # $1 port, $2 max seconds : ready only when OUR pid owns the port (a foreign server also answers /health)
  local i
  for ((i=0;i<$2;i+=5)); do
    kill -0 "$SRV_PID" 2>/dev/null || { echo "server died during load"; return 1; }
    if curl -sf -m 3 "http://127.0.0.1:$1/health" >/dev/null 2>&1; then
      ss -ltnpH "sport = :$1" 2>/dev/null | grep -q "pid=$SRV_PID," && return 0
      echo "port $1 answered /health but is not owned by pid $SRV_PID (foreign server)"; return 1
    fi
    [ "$(time_left)" -le 60 ] && { echo "out of time while loading"; return 1; }
    sleep 5
  done; echo "server not ready in $2 s"; return 1
}
# prove the binary is the one reasoned about (CLAUDE.md: build artifacts) : $1 bin dir
check_bin() {
  local B=$1
  [ -x "$B/llama-server" ] || { echo "no llama-server at $B"; return 6; }
  if ldd "$B/llama-server" 2>/dev/null | grep -E 'libllama\.so|libggml' | grep -v "$B" >/dev/null; then echo "foreign ggml/llama linkage"; return 6; fi
  grep -qaF deepseek41 "$B/libllama.so.0" || { echo "binary lacks deepseek41"; return 6; }
  echo "BIN $B libllama $(stat -c '%y' "$B/libllama.so.0")"
}
