#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CASE_SCRIPT="${ROOT}/docs/data/model_admission_smoke_commands_20260716.sh"
OUT_BASE="${OUT_BASE:-/mnt/raid0/llm/tmp/model-admission-smoke-$(date -u +%Y%m%dT%H%M%SZ)}"
SMOKE_TIMEOUT_SECONDS="${SMOKE_TIMEOUT_SECONDS:-180}"
SMOKE_KILL_AFTER_SECONDS="${SMOKE_KILL_AFTER_SECONDS:-10}"
SMOKE_LOG_BYTES="${SMOKE_LOG_BYTES:-1048576}"

QUEUE=(
  hy3_cpu_smoke
  qwable_iq4xs_cpu_v7
  qwable_iq4xs_mi210_v7
  qwable_q8_mi210_v7
  qwen3_4b_thinking_cpu_v7
  qwen3_4b_thinking_mi210_v7
  qwen25_coder14_cpu_v7
  qwen25_coder14_mi210_v7
  qwen35_9b_mtp_cpu_v7
  qwen35_9b_mtp_mi210_v7
  minicpm_q4_cpu_text_v7
  minicpm_q4_mi210_text_v7
  deepseek_v4_flash_cpu_v7
)

STOPPED_CASES=(
  bonsai_q1_cpu
  bonsai_q1_mi210_v7
  bonsai_dspark_cpu_v7
  bonsai_dspark_mi210_v7
  ternary_q2_0_cpu_v7
  ternary_q2_0_mi210_v7
  ternary_bonsai_dspark_cpu_v7
  ternary_bonsai_dspark_mi210_v7
  bonsai_8b_cpu_v7
  bonsai_8b_mi210_v7
  qwen3_vl8_cpu_text_v7
  qwen3_vl8_mi210_text_v7
  nemotron_nano_9b_q8_cpu_v7
  nemotron_nano_9b_q8_mi210_v7
  nemotron_diff14_q8_cpu_v7
  nemotron_diff14_q8_mi210_v7
)

usage() {
  cat <<'USAGE'
usage:
  run_model_admission_smoke_queue.sh --list
  run_model_admission_smoke_queue.sh --list-stopped
  run_model_admission_smoke_queue.sh --run [--from CASE] [--only CASE ...] [--out DIR] [--allow-glm-download] [--quality-fix-ref REF|--loader-fix-ref REF|--protocol-fix-ref REF|--artifact-fix-ref REF]

Runs the 2026-07-16 model admission smoke queue with bounded per-case
stdout/stderr capture. Each case runs under timeout(1) and live log caps, so a
bad direct-CLI smoke cannot produce unbounded logs or survive after timeout.

By default, model-load cases still refuse to run while the GLM-5.2 HF writer is
active because docs/data/model_admission_smoke_commands_20260716.sh enforces
that guard. Use --allow-glm-download only for deliberately light non-GLM smoke
churn; resulting speed/quality numbers are admission observations, not
decision-grade measurements.

Stopped cases are quality/protocol/loader blocked and are not part of the
default queue. They can run only via --only plus a concrete fix reference. Every
future model probe must append a row to:
  /mnt/raid0/llm/epyc-root/docs/reference/model-probe-scoreboard.md
USAGE
}

list_queue() {
  local i=1
  for case_name in "${QUEUE[@]}"; do
    printf '%02d %s\n' "${i}" "${case_name}"
    i=$((i + 1))
  done
}

list_stopped_cases() {
  local i=1
  for case_name in "${STOPPED_CASES[@]}"; do
    printf '%02d %s\n' "${i}" "${case_name}"
    i=$((i + 1))
  done
}

is_stopped_case() {
  local needle="$1"
  local case_name
  for case_name in "${STOPPED_CASES[@]}"; do
    [[ "${case_name}" == "${needle}" ]] && return 0
  done
  return 1
}

is_known_case() {
  local needle="$1"
  case_index "${needle}" >/dev/null && return 0
  is_stopped_case "${needle}"
}

case_index() {
  local needle="$1"
  local i=0
  for case_name in "${QUEUE[@]}"; do
    if [[ "${case_name}" == "${needle}" ]]; then
      printf '%s\n' "${i}"
      return 0
    fi
    i=$((i + 1))
  done
  return 1
}

RUN=0
FROM=""
ONLY_CASES=()
ALLOW_GLM=0
REOPEN_FIX_REF=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --list)
      list_queue
      exit 0
      ;;
    --list-stopped)
      list_stopped_cases
      exit 0
      ;;
    --run)
      RUN=1
      shift
      ;;
    --from)
      FROM="${2:-}"
      shift 2
      ;;
    --only)
      ONLY_CASES+=("${2:-}")
      shift 2
      ;;
    --out)
      OUT_BASE="${2:-}"
      shift 2
      ;;
    --allow-glm-download)
      ALLOW_GLM=1
      shift
      ;;
    --quality-fix-ref|--loader-fix-ref|--protocol-fix-ref|--artifact-fix-ref|--reopen-fix-ref)
      if [[ -z "${2:-}" ]]; then
        echo "FATAL: $1 requires a non-empty reference" >&2
        exit 64
      fi
      REOPEN_FIX_REF="${1#--}:${2}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 64
      ;;
  esac
done

if [[ "${RUN}" -ne 1 ]]; then
  usage
  echo
  list_queue
  exit 0
fi

[[ -x "${CASE_SCRIPT}" ]] || {
  echo "FATAL: case script is not executable: ${CASE_SCRIPT}" >&2
  exit 70
}

mkdir -p "${OUT_BASE}"

START=0
if [[ -n "${FROM}" ]]; then
  START="$(case_index "${FROM}")" || {
    echo "FATAL: unknown --from case: ${FROM}" >&2
    list_queue >&2
    exit 64
  }
fi

if [[ -n "${FROM}" && "${#ONLY_CASES[@]}" -gt 0 ]]; then
  echo "FATAL: --from cannot be combined with --only; use one queue-shaping mode." >&2
  exit 64
fi

if [[ "${#ONLY_CASES[@]}" -gt 0 ]]; then
  for only_case in "${ONLY_CASES[@]}"; do
    is_known_case "${only_case}" || {
      echo "FATAL: unknown --only case: ${only_case}" >&2
      list_queue >&2
      echo "Stopped cases:" >&2
      list_stopped_cases >&2
      exit 64
    }
    if is_stopped_case "${only_case}" && [[ -z "${REOPEN_FIX_REF}" ]]; then
      echo "FATAL: ${only_case} is stopped because it is quality/protocol/loader blocked." >&2
      echo "Provide --quality-fix-ref, --loader-fix-ref, --protocol-fix-ref, or --artifact-fix-ref only after a concrete fix lands." >&2
      echo "Do not run speed-only reruns for stopped candidates." >&2
      exit 75
    fi
  done
fi

SUMMARY="${OUT_BASE}/summary.tsv"
printf 'case\tstatus\texit_code\tstdout\tstderr\ttimeout_s\tlog_cap_bytes\treopen_fix_ref\n' > "${SUMMARY}"

ITER_CASES=("${QUEUE[@]}")
if [[ "${#ONLY_CASES[@]}" -gt 0 ]]; then
  ITER_CASES=("${ONLY_CASES[@]}")
fi

for idx in "${!ITER_CASES[@]}"; do
  case_name="${ITER_CASES[$idx]}"
  if [[ "${#ONLY_CASES[@]}" -eq 0 ]]; then
    [[ "${idx}" -ge "${START}" ]] || continue
  fi

  stdout="${OUT_BASE}/${case_name}.stdout"
  stderr="${OUT_BASE}/${case_name}.stderr"
  echo "== ${case_name} =="
  set +e
  ALLOW_GLM_DOWNLOAD="${ALLOW_GLM}" \
    MODEL_PROBE_REOPEN_FIX_REF="${REOPEN_FIX_REF}" \
    timeout -k "${SMOKE_KILL_AFTER_SECONDS}s" "${SMOKE_TIMEOUT_SECONDS}s" \
    "${CASE_SCRIPT}" "${case_name}" \
    > >(head -c "${SMOKE_LOG_BYTES}" >"${stdout}") \
    2> >(head -c "${SMOKE_LOG_BYTES}" >"${stderr}")
  code=$?
  set -e

  status="pass"
  if [[ "${code}" -ne 0 ]]; then
    status="fail"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${case_name}" "${status}" "${code}" "${stdout}" "${stderr}" \
    "${SMOKE_TIMEOUT_SECONDS}" "${SMOKE_LOG_BYTES}" "${REOPEN_FIX_REF}" >> "${SUMMARY}"

  if [[ "${code}" -ne 0 ]]; then
    echo "case failed: ${case_name} exit=${code}" >&2
    echo "summary: ${SUMMARY}" >&2
    exit "${code}"
  fi
done

echo "summary: ${SUMMARY}"
