#!/bin/bash
set -euo pipefail
# ── Kill seeding script and all related processes ────────────────────
#
# Stops: seeding script, TUI, orchestrator API (uvicorn :8000),
#         and erases active inference slots — WITHOUT killing llama-servers.
#
# Usage:
#   scripts/benchmark/kill_seeding.sh          # normal teardown
#   scripts/benchmark/kill_seeding.sh --force   # SIGKILL everything

FORCE=false
[[ "${1:-}" == "--force" ]] && FORCE=true

HEAVY_PORTS=(8080 8081 8083 8084 8085 8087)
ALL_MODEL_PORTS=(8080 8081 8082 8083 8084 8085 8086 8087)

killed_something=false

# ── 1. Kill seeding script (parent) ─────────────────────────────────
# The TUI uses curses threads that ignore SIGTERM, so we must:
#   SIGTERM → wait up to 3s → verify → SIGKILL if still alive → reap zombies
echo "==> Looking for seeding script..."
seed_pids=$(pgrep -f 'seed_specialist_routing' 2>/dev/null || true)
if [[ -n "$seed_pids" ]]; then
  for pid in $seed_pids; do
    echo "    Sending SIGTERM to seeding script PID $pid"
    if $FORCE; then
      kill -9 "$pid" 2>/dev/null || true
    else
      kill "$pid" 2>/dev/null || true
    fi
  done
  killed_something=true

  # Wait up to 3s for graceful shutdown, then escalate to SIGKILL
  if ! $FORCE; then
    echo "    Waiting for graceful shutdown (max 3s)..."
    for i in 1 2 3; do
      sleep 1
      remaining=$(pgrep -f 'seed_specialist_routing' 2>/dev/null || true)
      if [[ -z "$remaining" ]]; then
        echo "    Graceful shutdown succeeded"
        break
      fi
      if [[ "$i" -eq 3 ]]; then
        echo "    SIGTERM not enough (TUI curses threads) — escalating to SIGKILL"
        for rpid in $remaining; do
          kill -9 "$rpid" 2>/dev/null || true
        done
        sleep 1
      fi
    done
  else
    sleep 1
  fi

  # Reap zombie children left by the TUI
  zombie_pids=$(ps -eo pid,ppid,stat,comm 2>/dev/null | awk '$3 ~ /^Z/ && $4 == "python" {print $1}' || true)
  if [[ -n "$zombie_pids" ]]; then
    for zpid in $zombie_pids; do
      echo "    Reaping zombie PID $zpid"
      kill -9 "$zpid" 2>/dev/null || true
    done
  fi
fi

# ── 2. Kill orchestrator API (uvicorn on :8000) ─────────────────────
echo "==> Looking for orchestrator API (uvicorn :8000)..."
api_pids=$(pgrep -f 'uvicorn.*src.api:app' 2>/dev/null || true)
if [[ -n "$api_pids" ]]; then
  # Kill entire process group (uvicorn parent + multiprocessing workers)
  for pid in $api_pids; do
    pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    if [[ -n "$pgid" && "$pgid" != "0" ]]; then
      echo "    Killing uvicorn process group (PGID $pgid)"
      if $FORCE; then
        kill -9 -"$pgid" 2>/dev/null || true
      else kill -"$pgid" 2>/dev/null || true; fi
    else
      echo "    Killing uvicorn PID $pid"
      if $FORCE; then
        kill -9 "$pid" 2>/dev/null || true
      else kill "$pid" 2>/dev/null || true; fi
    fi
  done
  killed_something=true
  sleep 1
fi

# Also catch orphaned multiprocessing workers from uvicorn
orphan_pids=$(pgrep -f 'multiprocessing.spawn.*spawn_main' 2>/dev/null || true)
if [[ -n "$orphan_pids" ]]; then
  for pid in $orphan_pids; do
    echo "    Killing orphaned worker PID $pid"
    kill -9 "$pid" 2>/dev/null || true
  done
  killed_something=true
  sleep 1
fi

# ── 3. Verify port 8000 is free ─────────────────────────────────────
if ss -tnlp 2>/dev/null | grep -q ':8000 '; then
  echo "    Port 8000 still occupied — force-killing with fuser"
  fuser -k 8000/tcp 2>/dev/null || true
  sleep 1
fi

# ── 4. Erase active inference slots (don't kill servers) ─────────────
echo "==> Erasing active inference slots..."
for port in "${ALL_MODEL_PORTS[@]}"; do
  # Quick check: is this port even up?
  if ! curl -sf --max-time 1 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    continue
  fi

  slots_json=$(curl -sf --max-time 2 "http://127.0.0.1:$port/slots" 2>/dev/null || echo "[]")
  processing=$(echo "$slots_json" | python3 -c "
import sys, json
try:
    slots = json.load(sys.stdin)
    for s in slots:
        if s.get('is_processing'):
            print(s.get('id', 0))
except: pass
" 2>/dev/null)

  if [[ -n "$processing" ]]; then
    for slot_id in $processing; do
      echo "    Port $port slot $slot_id: erasing..."
      # Try POST first, fall back to GET — with short timeout
      curl -sf --max-time 3 -X POST \
        "http://127.0.0.1:$port/slots/${slot_id}?action=erase" \
        >/dev/null 2>&1 ||
        curl -sf --max-time 3 \
          "http://127.0.0.1:$port/slots/${slot_id}?action=erase" \
          >/dev/null 2>&1 ||
        echo "    Port $port slot $slot_id: erase timed out (server may need restart)"
    done
  fi
done

# ── 5. Verify: check all slots are idle ──────────────────────────────
echo "==> Verifying all slots idle..."
any_stuck=false
for port in "${ALL_MODEL_PORTS[@]}"; do
  if ! curl -sf --max-time 1 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    continue
  fi
  still_processing=$(curl -sf --max-time 2 "http://127.0.0.1:$port/slots" 2>/dev/null |
    python3 -c "
import sys, json
try:
    slots = json.load(sys.stdin)
    stuck = [s['id'] for s in slots if s.get('is_processing')]
    if stuck: print(f'port $port: slots {stuck} still processing')
except: pass
" 2>/dev/null)
  if [[ -n "$still_processing" ]]; then
    echo "    WARNING: $still_processing"
    any_stuck=true
  fi
done

if $any_stuck; then
  echo ""
  echo "WARNING: Some slots are still processing. The erase call may have"
  echo "timed out. You can force-restart those servers with:"
  echo "    python3 scripts/server/orchestrator_stack.py restart --port <PORT>"
  echo "Or re-run with: scripts/benchmark/kill_seeding.sh --force"
fi

# ── Summary ──────────────────────────────────────────────────────────
echo ""
if $killed_something; then
  echo "Done. Seeding script, TUI, and orchestrator API stopped. Model servers untouched."
else
  echo "Nothing to kill — no seeding processes found."
fi
