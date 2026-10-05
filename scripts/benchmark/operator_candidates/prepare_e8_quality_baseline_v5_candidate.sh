#!/bin/bash
# Proposed human-owned v5 evidence-validator amendment. Read-only.
set -euo pipefail

ORCH="/mnt/raid0/llm/epyc-orchestrator"
SOURCE_ROOT="${E8_V5_SOURCE_ROOT:-$ORCH}"
PYTHON="$ORCH/.venv/bin/python"
VALIDATOR="$SOURCE_ROOT/scripts/benchmark/validate_e8_quality_baseline_v5.py"
PRODUCER="$SOURCE_ROOT/scripts/benchmark/terminalize_e8_quality_baseline_source.py"
RUNNER="$SOURCE_ROOT/scripts/benchmark/run_e8_quality_baseline_v5.py"
BASE_RUNNER="$SOURCE_ROOT/scripts/benchmark/run_e8_quality_baseline_reseed.py"
RESUME_RUNNER="$SOURCE_ROOT/scripts/benchmark/resume_e8_quality_baseline_v5.py"
RECOVERY_RUNNER="$SOURCE_ROOT/scripts/benchmark/recover_e8_quality_baseline_v5_partial_r2.py"
FINALIZER_RUNNER="$SOURCE_ROOT/scripts/benchmark/finalize_e8_quality_baseline_v5_recovery_r2.py"
SUCCESSOR_RUNNER="$SOURCE_ROOT/scripts/benchmark/prepare_e8_quality_baseline_v5_partial_r2_successor.py"
RACE_RETRY_RUNNER="$SOURCE_ROOT/scripts/benchmark/prepare_e8_quality_baseline_v5_partial_r2_race_retry.py"
MIXED_TAIL_REPAIR_RUNNER="$SOURCE_ROOT/scripts/benchmark/prepare_e8_quality_baseline_v5_partial_r2_mixed_tail_repair.py"
TERMINALIZER_RUNNER="$SOURCE_ROOT/scripts/benchmark/terminalize_e8_quality_baseline_v5_partial_r2_successor.py"
FINAL_C1_RETRY_RUNNER="$SOURCE_ROOT/scripts/benchmark/final_c1_retry.py"
FINAL_C1_VALIDATOR="$SOURCE_ROOT/scripts/benchmark/final_c1_validator.py"

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}
sha() { sha256sum -- "$1" | awk '{print $1}'; }
[[ $# -eq 2 && "$1" == "--validate-evidence" ]] ||
  fail 'usage: prepare_e8_quality_baseline_v5_candidate.sh --validate-evidence EVIDENCE'
[[ -x "$PYTHON" && -f "$VALIDATOR" && -f "$PRODUCER" && -f "$RUNNER" && -f "$BASE_RUNNER" && -f "$RESUME_RUNNER" && -f "$RECOVERY_RUNNER" && -f "$FINALIZER_RUNNER" && -f "$SUCCESSOR_RUNNER" && -f "$RACE_RETRY_RUNNER" && -f "$MIXED_TAIL_REPAIR_RUNNER" && -f "$TERMINALIZER_RUNNER" && -f "$FINAL_C1_RETRY_RUNNER" && -f "$FINAL_C1_VALIDATOR" ]] ||
  fail 'v5 composite validator prerequisite is missing'
[[ "$(readlink -f -- "$PYTHON")" == "$(readlink -f -- "$ORCH/.venv/bin/python")" ]] ||
  fail 'canonical orchestrator venv identity differs'
[[ "${E8_V5_ORCHESTRATOR_HEAD:-}" =~ ^[0-9a-f]{40}$ && "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" == "$E8_V5_ORCHESTRATOR_HEAD" ]] ||
  fail 'reviewed source HEAD differs from the supplied source pin'
[[ "${E8_V5_VALIDATOR_SHA256:-}" =~ ^[0-9a-f]{64}$ && "$(sha "$0")" == "$E8_V5_VALIDATOR_SHA256" ]] ||
  fail 'v5 validator wrapper differs from the externally reviewed hash'
for binding in \
  "E8_V5_PRODUCER_SHA256:$PRODUCER" \
  "E8_V5_RUNNER_SHA256:$RUNNER" \
  "E8_V5_BASE_RUNNER_SHA256:$BASE_RUNNER" \
  "E8_V5_RESUME_RUNNER_SHA256:$RESUME_RUNNER" \
  "E8_V5_RECOVERY_RUNNER_SHA256:$RECOVERY_RUNNER" \
  "E8_V5_FINALIZER_RUNNER_SHA256:$FINALIZER_RUNNER" \
  "E8_V5_SUCCESSOR_RUNNER_SHA256:$SUCCESSOR_RUNNER" \
  "E8_V5_RACE_RETRY_RUNNER_SHA256:$RACE_RETRY_RUNNER" \
  "E8_V5_MIXED_TAIL_REPAIR_RUNNER_SHA256:$MIXED_TAIL_REPAIR_RUNNER" \
  "E8_V5_FINAL_C1_RETRY_RUNNER_SHA256:$FINAL_C1_RETRY_RUNNER" \
  "E8_V5_FINAL_C1_VALIDATOR_SHA256:$FINAL_C1_VALIDATOR" \
  "E8_V5_VALIDATOR_PY_SHA256:$VALIDATOR"; do
  name="${binding%%:*}"
  path="${binding#*:}"
  expected="${!name:-}"
  [[ "$expected" =~ ^[0-9a-f]{64}$ && "$(sha "$path")" == "$expected" ]] ||
    fail "reviewed composite artifact pin differs: $name"
done
terminalizer_args=()
if [[ -n "${E8_V5_TERMINALIZER_RUNNER_SHA256:-}" ]]; then
  [[ "$E8_V5_TERMINALIZER_RUNNER_SHA256" =~ ^[0-9a-f]{64}$ && "$(sha "$TERMINALIZER_RUNNER")" == "$E8_V5_TERMINALIZER_RUNNER_SHA256" ]] ||
    fail 'reviewed composite artifact pin differs: E8_V5_TERMINALIZER_RUNNER_SHA256'
  terminalizer_args=(--expected-terminalizer-runner-sha256 "$E8_V5_TERMINALIZER_RUNNER_SHA256")
fi

PYTHONOPTIMIZE=0 "$PYTHON" "$VALIDATOR" \
  --evidence "$2" \
  --expected-runner-sha256 "$E8_V5_RUNNER_SHA256" \
  --expected-base-runner-sha256 "$E8_V5_BASE_RUNNER_SHA256" \
  --expected-resume-runner-sha256 "$E8_V5_RESUME_RUNNER_SHA256" \
  --expected-recovery-runner-sha256 "$E8_V5_RECOVERY_RUNNER_SHA256" \
  --expected-finalizer-runner-sha256 "$E8_V5_FINALIZER_RUNNER_SHA256" \
  --expected-successor-runner-sha256 "$E8_V5_SUCCESSOR_RUNNER_SHA256" \
  --expected-race-retry-runner-sha256 "$E8_V5_RACE_RETRY_RUNNER_SHA256" \
  --expected-mixed-tail-repair-runner-sha256 "$E8_V5_MIXED_TAIL_REPAIR_RUNNER_SHA256" \
  --expected-final-c1-retry-runner-sha256 "$E8_V5_FINAL_C1_RETRY_RUNNER_SHA256" \
  --expected-final-c1-validator-sha256 "$E8_V5_FINAL_C1_VALIDATOR_SHA256" \
  "${terminalizer_args[@]}"
