#!/bin/bash
set -euo pipefail
# Drop-in PID resolver for job.sh; the caller owns parent_pid and its probe window.
if [[ ${1:-} == --help ]]; then
  printf 'usage: probe_child.sh CAPTURED_PARENT_PID\nPrint the real llama-bench descendant PID, or exit1 without sampling a wrapper.\n'
  exit 0
fi
tools_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$tools_dir/lb_pid.sh"
lb_pid "${1:?captured parent PID required}"
