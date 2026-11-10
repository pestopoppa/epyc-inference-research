#!/bin/bash
# AGG-0.4 launcher skeleton. Keeper-run only, under normal grants (one CPU quarter,
# then one MI210 window). Default action is the zero-compute preflight.
#   launch_phase0.sh                 -> preflight only
#   launch_phase0.sh run <out_dir> --phase cpu|gpu-audit|gpu-answers|score [run_phase0 args]
#   (keeper jobs: /mnt/raid0/llm/tmp/agg0-arms-{cpu,gpu}-20261110; they launch the private servers)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${AGG_PY:-python3}"
export ORCHESTRATOR_FEATURE_AGG_LAYER=1

mode="${1:-preflight}"
case "$mode" in
  preflight)
    exec "$PY" -I "$HERE/run_phase0.py" --set-dir "$HERE" --preflight ;;
  run)
    out="${2:?usage: launch_phase0.sh run <out_dir> --phase PHASE [args]}"
    shift 2
    exec "$PY" -I "$HERE/run_phase0.py" --set-dir "$HERE" --out "$out" "$@" ;;
  *)
    echo "usage: $0 [preflight|run <out_dir>]" >&2; exit 64 ;;
esac
