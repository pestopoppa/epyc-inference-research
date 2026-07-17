#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_ROOT="/mnt/raid0/llm/tmp"
EXPERIMENTAL_ROOT="/mnt/raid0/llm/llama.cpp-experimental"
EXPERIMENTAL_BIN_DIR="${EXPERIMENTAL_ROOT}/build-hip/bin"
EXPERIMENTAL_GGUF_PY="${EXPERIMENTAL_ROOT}/gguf-py"
DEFAULT_BINARY="${EXPERIMENTAL_BIN_DIR}/llama-imatrix"
PRODUCTION_BINARY="/mnt/raid0/llm/llama.cpp/build-hip/bin/llama-imatrix"
GLM_PATTERN='hf download unsloth/GLM-5.2-GGUF'
EXPERT_COUNT_EXTRACTOR="${SCRIPT_DIR}/extract_imatrix_expert_counts.py"

EXECUTE=0
SHOW_STATISTICS=0
ALLOW_GLM_DOWNLOAD=0
INPUT_ARTIFACT=""
MODEL=""
CORPUS_FILE=""
OUTPUT_ARTIFACT=""
OUT_ROOT=""
BINARY_PATH="${DEFAULT_BINARY}"
CTX_SIZE=128
BATCH_SIZE=128
UBATCH_SIZE=128
CHUNKS=-1
PARSE_SPECIAL=1

usage() {
  cat <<'USAGE'
usage:
  expert_routing_skew_profile.sh [--show-statistics --artifact PATH --model PATH]
  expert_routing_skew_profile.sh [--execute --artifact PATH --model PATH]
  expert_routing_skew_profile.sh [--execute --model PATH [--corpus-file PATH]]
                                 [--output-artifact PATH] [--out-root DIR]

Defaults to dry-run mode. The dry-run prints the exact command plan and keeps
all artifacts under /mnt/raid0/llm/tmp/. Execution is opt-in via --execute.

Options:
  --artifact PATH            Existing imatrix artifact for statistics mode.
  --corpus-file PATH         Text corpus to use for execute mode.
  --output-artifact PATH     Output imatrix artifact path for execute mode.
  --out-root DIR             Output directory under /mnt/raid0/llm/tmp/.
  --binary PATH              Override the llama-imatrix binary.
  --ctx-size N               Context size for execute mode.
  --batch-size N             Batch size for execute mode.
  --ubatch-size N            Ubatch size for execute mode.
  --chunks N                 Max chunks to process (-1 = all).
  --allow-glm-download       Override the GLM-5.2 download guard.
  --show-statistics          Statistics-only mode for an existing artifact.
  --execute                  Run calibration first, then statistics.
  -h, --help                 Show this help.
USAGE
}

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

die() {
  printf 'FATAL: %s\n' "$*" >&2
  exit 1
}

require_tmp_path() {
  local path="$1"
  case "$path" in
    "${TMP_ROOT}"/*) ;;
    *)
      die "path must live under ${TMP_ROOT}: ${path}"
      ;;
  esac
}

resolve_binary() {
  if [[ ! -x "${BINARY_PATH}" ]]; then
    die "llama-imatrix not found or not executable: ${BINARY_PATH}"
  fi

  local resolved
  resolved="$(realpath "${BINARY_PATH}")"
  case "${resolved}" in
    "${EXPERIMENTAL_ROOT}"/*) ;;
    "${PRODUCTION_BINARY}")
      die "refusing to use production v6 binary: ${resolved}"
      ;;
    *)
      die "refusing non-experimental llama-imatrix binary: ${resolved}"
      ;;
  esac

  BINARY_PATH="${resolved}"
}

require_no_glm_download() {
  if [[ "${ALLOW_GLM_DOWNLOAD}" -eq 1 ]]; then
    return 0
  fi
  if pgrep -af "${GLM_PATTERN}" >/dev/null; then
    printf 'FATAL: GLM-5.2 download is active; rerun with --allow-glm-download only if you accept cache contention.\n' >&2
    pgrep -af "${GLM_PATTERN}" >&2 || true
    exit 75
  fi
}

make_out_root() {
  if [[ -z "${OUT_ROOT}" ]]; then
    OUT_ROOT="${TMP_ROOT}/expert-routing-skew-$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  require_tmp_path "${OUT_ROOT}"
  mkdir -p "${OUT_ROOT}"
}

generate_default_corpus() {
  local corpus_path="$1"
  cat >"${corpus_path}" <<'EOF'
Route the query through sparse experts.
Record per-layer expert hit frequency for the quiet-window MI210 gate.
Prefer short, technical sentences so tokenization stays deterministic.
Measure whether the routing pattern is Zipfian or near-uniform.
EOF
  local i
  for i in {1..48}; do
    cat <<'EOF'
The calibration corpus should stay small, repetitive, and workload-like.
It should exercise expert selection without turning into a long benchmark.
EOF
  done >>"${corpus_path}"
}

stats_command() {
  local artifact="$1"
  local stats_out="$2"
  local model="$3"
  printf 'LD_LIBRARY_PATH=%q %q -m %q --show-statistics --in-file %q > %q 2>&1\n' \
    "${EXPERIMENTAL_BIN_DIR}" \
    "${BINARY_PATH}" \
    "${model}" \
    "${artifact}" \
    "${stats_out}"
}

artifact_stem() {
  local artifact="$1"
  local stem
  stem="$(basename "${artifact%.*}")"
  printf '%s' "${stem%.imatrix}"
}

counts_command() {
  local artifact="$1"
  local counts_json="$2"
  local counts_md="$3"
  [[ -x "${EXPERT_COUNT_EXTRACTOR}" ]] || die "expert-count extractor not executable: ${EXPERT_COUNT_EXTRACTOR}"
  printf 'PYTHONPATH=%q uv run --with numpy python %q --artifact %q --output-json %q --output-md %q\n' \
    "${EXPERIMENTAL_GGUF_PY}" \
    "${EXPERT_COUNT_EXTRACTOR}" \
    "${artifact}" \
    "${counts_json}" \
    "${counts_md}"
}

execute_command() {
  local corpus="$1"
  local artifact="$2"
  local chunks_arg=""
  if [[ "${CHUNKS}" -ge 0 ]]; then
    chunks_arg="$(printf -- '--chunks %q' "${CHUNKS}")"
  fi
  printf 'LD_LIBRARY_PATH=%q %q -m %q -f %q -o %q --ctx-size %q --batch-size %q --ubatch-size %q --no-ppl %s %s\n' \
    "${EXPERIMENTAL_BIN_DIR}" \
    "${BINARY_PATH}" \
    "${MODEL}" \
    "${corpus}" \
    "${artifact}" \
    "${CTX_SIZE}" \
    "${BATCH_SIZE}" \
    "${UBATCH_SIZE}" \
    "${chunks_arg}" \
    "$( [[ "${PARSE_SPECIAL}" -eq 1 ]] && printf '%s' '--parse-special' || printf '%s' '' )"
}

run_shell_command() {
  local command="$1"
  bash -lc "set -euo pipefail; ${command}"
}

INPUT_ARTIFACT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --artifact)
      INPUT_ARTIFACT="${2:-}"
      shift 2
      ;;
    --corpus-file)
      CORPUS_FILE="${2:-}"
      shift 2
      ;;
    --output-artifact)
      OUTPUT_ARTIFACT="${2:-}"
      shift 2
      ;;
    --out-root)
      OUT_ROOT="${2:-}"
      shift 2
      ;;
    --binary)
      BINARY_PATH="${2:-}"
      shift 2
      ;;
    --model)
      MODEL="${2:-}"
      shift 2
      ;;
    --ctx-size)
      CTX_SIZE="${2:-}"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="${2:-}"
      shift 2
      ;;
    --ubatch-size)
      UBATCH_SIZE="${2:-}"
      shift 2
      ;;
    --chunks)
      CHUNKS="${2:-}"
      shift 2
      ;;
    --allow-glm-download)
      ALLOW_GLM_DOWNLOAD=1
      shift
      ;;
    --show-statistics)
      SHOW_STATISTICS=1
      shift
      ;;
    --execute)
      EXECUTE=1
      shift
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

if [[ "${SHOW_STATISTICS}" -ne 1 && -z "${INPUT_ARTIFACT}" && -z "${MODEL}" ]]; then
  usage
  printf '\nExamples:\n' >&2
  printf '  %s --show-statistics --artifact /mnt/raid0/llm/tmp/example.imatrix.gguf --model /mnt/raid0/llm/models/example.gguf\n' "$0" >&2
  printf '  %s --execute --model /mnt/raid0/llm/models/example.gguf\n' "$0" >&2
  exit 0
fi

resolve_binary
require_no_glm_download
make_out_root

if [[ -n "${MODEL}" && -z "${INPUT_ARTIFACT}" && "${SHOW_STATISTICS}" -eq 0 ]]; then
  [[ -f "${MODEL}" ]] || die "model not found: ${MODEL}"
  if [[ -n "${INPUT_ARTIFACT}" || "${SHOW_STATISTICS}" -eq 1 ]]; then
    die "--execute cannot be combined with --artifact or --show-statistics"
  fi

  local_corpus="${CORPUS_FILE}"
  if [[ -z "${local_corpus}" ]]; then
    local_corpus="${OUT_ROOT}/expert-routing-skew.corpus.txt"
    generate_default_corpus "${local_corpus}"
  fi
  [[ -f "${local_corpus}" ]] || die "corpus file not found: ${local_corpus}"

  if [[ -z "${OUTPUT_ARTIFACT}" ]]; then
    OUTPUT_ARTIFACT="${OUT_ROOT}/expert-routing-skew.imatrix.gguf"
  fi
  require_tmp_path "${OUTPUT_ARTIFACT}"
  mkdir -p "$(dirname "${OUTPUT_ARTIFACT}")"
  stats_out="${OUT_ROOT}/expert-routing-skew.stats.txt"
  counts_json="${OUT_ROOT}/$(artifact_stem "${OUTPUT_ARTIFACT}").counts.json"
  counts_md="${OUT_ROOT}/$(artifact_stem "${OUTPUT_ARTIFACT}").counts.md"

  log "dry-run=$([[ "${EXECUTE}" -eq 1 ]] && printf 'false' || printf 'true')"
  log "binary=${BINARY_PATH}"
  log "corpus=${local_corpus}"
  log "artifact=${OUTPUT_ARTIFACT}"
  log "stats=${stats_out}"
  log "counts_json=${counts_json}"
  log "counts_md=${counts_md}"
  log "command:"
  execute_cmd=$(execute_command "${local_corpus}" "${OUTPUT_ARTIFACT}")
  printf '%s\n' "${execute_cmd}"
  log "follow-up:"
  stats_cmd=$(stats_command "${OUTPUT_ARTIFACT}" "${stats_out}" "${MODEL}")
  printf '%s\n' "${stats_cmd}"
  counts_cmd=$(counts_command "${OUTPUT_ARTIFACT}" "${counts_json}" "${counts_md}")
  printf '%s\n' "${counts_cmd}"

  if [[ "${EXECUTE}" -eq 1 ]]; then
    log "running calibration"
    run_shell_command "${execute_cmd}" >"${OUT_ROOT}/expert-routing-skew.execute.log" 2>&1
    log "running statistics"
    run_shell_command "${stats_cmd}"
    log "extracting expert counts"
    run_shell_command "${counts_cmd}"
    log "done"
  fi
  exit 0
fi

if [[ "${SHOW_STATISTICS}" -eq 1 || -n "${INPUT_ARTIFACT}" ]]; then
  [[ -n "${INPUT_ARTIFACT}" ]] || die "--show-statistics requires --artifact PATH"
  [[ -f "${INPUT_ARTIFACT}" ]] || die "artifact not found: ${INPUT_ARTIFACT}"
  [[ -n "${MODEL}" ]] || die "--show-statistics requires --model PATH"
  [[ -f "${MODEL}" ]] || die "model not found: ${MODEL}"
  stats_out="${OUT_ROOT}/$(basename "${INPUT_ARTIFACT%.*}").stats.txt"
  counts_json="${OUT_ROOT}/$(artifact_stem "${INPUT_ARTIFACT}").counts.json"
  counts_md="${OUT_ROOT}/$(artifact_stem "${INPUT_ARTIFACT}").counts.md"

  log "dry-run=$([[ "${EXECUTE}" -eq 1 ]] && printf 'false' || printf 'true')"
  log "binary=${BINARY_PATH}"
  log "artifact=${INPUT_ARTIFACT}"
  log "model=${MODEL}"
  log "stats=${stats_out}"
  log "counts_json=${counts_json}"
  log "counts_md=${counts_md}"
  log "command:"
  stats_cmd=$(stats_command "${INPUT_ARTIFACT}" "${stats_out}" "${MODEL}")
  printf '%s\n' "${stats_cmd}"
  counts_cmd=$(counts_command "${INPUT_ARTIFACT}" "${counts_json}" "${counts_md}")
  printf '%s\n' "${counts_cmd}"
  if [[ "${EXECUTE}" -eq 1 ]]; then
    log "running statistics"
    run_shell_command "${stats_cmd}"
    log "extracting expert counts"
    run_shell_command "${counts_cmd}"
    log "done"
  fi
  exit 0
fi
