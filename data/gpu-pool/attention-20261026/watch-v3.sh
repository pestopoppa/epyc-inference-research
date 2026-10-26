#!/bin/bash
set -euo pipefail
cd /mnt/raid0/llm/tmp/codex-ni-20261026-gpu-sampler-exit-recovery/v3/attention
while :; do
  if [[ -e JOB_ENDED || -e FAILED || -e SUCCESS ]]; then
    date -u +%FT%TZ > WATCH-RESULT-READY
    exit 0
  fi
  if [[ -r launcher.pid ]] && ! ps -o pid= -p "$(cat launcher.pid)" >/dev/null; then
    date -u +%FT%TZ > WATCH-LAUNCHER-ENDED
    exit 0
  fi
  sleep 10
done
