#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

apply=0
if [[ "${1:-}" == "--apply" ]]; then
  apply=1
elif [[ $# -gt 0 ]]; then
  echo "usage: $0 [--apply]" >&2
  exit 2
fi

pid_files=(
  "data/cpu_optimization/2026-05-04-q6k-default-on-validation/00-tripwire.pid"
  "data/cpu_optimization/2026-05-04-qwen35-122b-arch-probe/phase2.pid"
  "data/cpu_optimization/2026-05-04-qwen35-122b-arch-probe/probe.pid"
  "data/cpu_optimization/2026-05-04-reap246b-arch-probe/probe.pid"
)

echo "== epyc-inference-research emergency cleanup =="
echo "mode: $([[ "$apply" -eq 1 ]] && echo apply || echo dry-run)"

removed=0
kept=0
for path in "${pid_files[@]}"; do
  if [[ ! -f "$path" ]]; then
    continue
  fi
  pid="$(tr -cd '0-9' < "$path")"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "live pid file kept: $path pid=$pid"
    kept=$((kept + 1))
    continue
  fi
  if [[ "$apply" -eq 1 ]]; then
    rm -f -- "$path"
    echo "removed stale pid file: $path"
  else
    echo "would remove stale pid file: $path"
  fi
  removed=$((removed + 1))
done

echo "summary: stale=$removed live=$kept"
