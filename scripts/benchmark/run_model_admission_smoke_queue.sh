#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CASE_SCRIPT="${ROOT}/docs/data/model_admission_smoke_commands_20260716.sh"
OUT_BASE="${OUT_BASE:-/mnt/raid0/llm/tmp/model-admission-smoke-$(date -u +%Y%m%dT%H%M%SZ)}"

QUEUE=(
  hy3_cpu_smoke
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
  qwen3_vl8_cpu_text_v7
  qwen3_vl8_mi210_text_v7
  deepseek_v4_flash_cpu_v7
)

usage() {
  cat <<'USAGE'
usage:
  run_model_admission_smoke_queue.sh --list
  run_model_admission_smoke_queue.sh --run [--from CASE] [--only CASE] [--out DIR]

Runs the 2026-07-16 model admission smoke queue with per-case stdout/stderr
capture. Model-load cases still refuse to run while the GLM-5.2 HF writer is
active because docs/data/model_admission_smoke_commands_20260716.sh enforces
that guard.
USAGE
}

list_queue() {
  local i=1
  for case_name in "${QUEUE[@]}"; do
    printf '%02d %s\n' "${i}" "${case_name}"
    i=$((i + 1))
  done
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
ONLY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --list)
      list_queue
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
      ONLY="${2:-}"
      shift 2
      ;;
    --out)
      OUT_BASE="${2:-}"
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

if [[ -n "${ONLY}" ]]; then
  case_index "${ONLY}" >/dev/null || {
    echo "FATAL: unknown --only case: ${ONLY}" >&2
    list_queue >&2
    exit 64
  }
fi

SUMMARY="${OUT_BASE}/summary.tsv"
printf 'case\tstatus\texit_code\tstdout\tstderr\n' > "${SUMMARY}"

for idx in "${!QUEUE[@]}"; do
  case_name="${QUEUE[$idx]}"
  [[ "${idx}" -ge "${START}" ]] || continue
  if [[ -n "${ONLY}" && "${case_name}" != "${ONLY}" ]]; then
    continue
  fi

  stdout="${OUT_BASE}/${case_name}.stdout"
  stderr="${OUT_BASE}/${case_name}.stderr"
  echo "== ${case_name} =="
  set +e
  "${CASE_SCRIPT}" "${case_name}" >"${stdout}" 2>"${stderr}"
  code=$?
  set -e

  status="pass"
  if [[ "${code}" -ne 0 ]]; then
    status="fail"
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' "${case_name}" "${status}" "${code}" "${stdout}" "${stderr}" >> "${SUMMARY}"

  if [[ "${code}" -ne 0 ]]; then
    echo "case failed: ${case_name} exit=${code}" >&2
    echo "summary: ${SUMMARY}" >&2
    exit "${code}"
  fi
done

echo "summary: ${SUMMARY}"
