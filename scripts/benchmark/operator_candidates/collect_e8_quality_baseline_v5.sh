#!/bin/bash
# Tokenless v5 evidence collection. Never writes AutoPilot state or receipts.
set -euo pipefail

ORCH="/mnt/raid0/llm/epyc-orchestrator"
SOURCE_ROOT="${E8_V5_SOURCE_ROOT:-$ORCH}"
PYTHON="$ORCH/.venv/bin/python"
RUNNER="$SOURCE_ROOT/scripts/benchmark/run_e8_quality_baseline_v5.py"
BASE_RUNNER="$SOURCE_ROOT/scripts/benchmark/run_e8_quality_baseline_reseed.py"
LOCK="/mnt/raid0/llm/tmp/e8-quality-baseline-v5-collect.lock"

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}
sha() { sha256sum -- "$1" | awk '{print $1}'; }
[[ $# -eq 2 && "$1" == "--output-dir" ]] ||
  fail 'usage: collect_e8_quality_baseline_v5.sh --output-dir NEW_DIRECTORY'
OUTPUT="$2"
[[ "$OUTPUT" = /* && ! -e "$OUTPUT" ]] || fail 'output must be a new absolute path'
[[ -x "$PYTHON" && -f "$RUNNER" ]] || fail 'canonical orchestrator v5 prerequisite is missing'
[[ "$(readlink -f -- "$PYTHON")" == "$(readlink -f -- "$ORCH/.venv/bin/python")" ]] ||
  fail 'orchestrator venv identity differs'
[[ "${E8_V5_COLLECT_WRAPPER_SHA256:-}" =~ ^[0-9a-f]{64}$ && "$(sha "$0")" == "$E8_V5_COLLECT_WRAPPER_SHA256" ]] ||
  fail 'collection wrapper differs from the externally reviewed hash'
[[ "${E8_V5_RUNNER_SHA256:-}" =~ ^[0-9a-f]{64}$ && "$(sha "$RUNNER")" == "$E8_V5_RUNNER_SHA256" ]] ||
  fail 'v5 runner differs from the externally reviewed hash'
[[ "${E8_V5_BASE_RUNNER_SHA256:-}" =~ ^[0-9a-f]{64}$ && "$(sha "$BASE_RUNNER")" == "$E8_V5_BASE_RUNNER_SHA256" ]] ||
  fail 'v4 base runner differs from the externally reviewed hash'
[[ "${E8_V5_ORCHESTRATOR_HEAD:-}" =~ ^[0-9a-f]{40}$ ]] ||
  fail 'E8_V5_ORCHESTRATOR_HEAD must externally pin the clean source commit'
[[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" == "$E8_V5_ORCHESTRATOR_HEAD" ]] ||
  fail 'canonical orchestrator HEAD differs from the reviewed source pin'
[[ -z "$(git -C "$SOURCE_ROOT" status --porcelain)" ]] ||
  fail 'canonical orchestrator tracked worktree is not clean'

exec 9>"$LOCK"
flock -n 9 || fail 'another v5 collection owns the lock'
exec env -u E8_BASELINE_APPLY_TOKEN PYTHONOPTIMIZE=0 "$PYTHON" "$RUNNER" \
  --collect-candidate --output-dir "$OUTPUT"
