#!/bin/bash
set -euo pipefail
readonly tools_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
readonly base=${V11_REFRESH_STATE:?reviewed isolated state root required}
[[ -f "$base/MAIN-APPROVED-BUILD-v3" && -f "$base/synthetic/COMPLETE" ]]
export FILLER_STATE_DIR="$base/state-build-v3"
export FILLERS_MD="$base/OWN-FILLERS-BUILD-v3.md"
exec /bin/bash /mnt/raid0/llm/tmp/fillers-3c4ca599/filler_run8.sh \
 --id codex-ni-v11-086-slow2-build-20261025 --class cpu --role build --gpu-quiet none \
 --min-window 120 --stop "$base/BUILD-STOP-v3" --complete-cmd "test -e $base/build-logs-v3/COMPLETE" \
 --cmd "/usr/bin/taskset -c \"\$FILLER_CPUS\" /bin/bash $tools_dir/build-bench-only.sh" --workdir "$base"
