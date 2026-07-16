#!/usr/bin/env bash
set -euo pipefail

# Quiet-window candidate-model smoke commands staged on 2026-07-16.
# This file is opt-in: pass one case name. It does not run the whole queue.
# Keep AutoPilot and production stack orchestration stopped while running these.

PROD_LLAMA=/mnt/raid0/llm/llama.cpp/build/bin/llama-cli
V7_BIN_DIR=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin
V7_LLAMA="${V7_BIN_DIR}/llama-cli"
HY3_DIR=/mnt/raid0/llm/models/hy3-angelslim
HY3_BUILD=/mnt/raid0/llm/tmp/llama.cpp-hyv3-20260716

# Production v6 is immutable: this script must never build or modify production
# v6. Candidate smokes use the relinked v7 build-hip binary path above.

v7_cpu_llama() {
  LD_LIBRARY_PATH="${V7_BIN_DIR}:${LD_LIBRARY_PATH:-}" \
  HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= \
  "${V7_LLAMA}" --device none --device-draft none \
    --simple-io --no-warmup --single-turn --no-display-prompt --color off "$@"
}

v7_mi210_llama() {
  LD_LIBRARY_PATH="${V7_BIN_DIR}:${LD_LIBRARY_PATH:-}" HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
  "${V7_LLAMA}" --device ROCm0 \
    --simple-io --no-warmup --single-turn --no-display-prompt --color off "$@"
}

require_no_glm_download() {
  if pgrep -af "hf download unsloth/GLM-5.2-GGUF" >/dev/null; then
    if [[ "${ALLOW_GLM_DOWNLOAD:-0}" == "1" ]]; then
      echo "WARNING: running non-GLM smoke while GLM-5.2 HF download is active (ALLOW_GLM_DOWNLOAD=1)." >&2
      return 0
    fi
    echo "Refusing to run smoke while GLM-5.2 HF download is active." >&2
    pgrep -af "hf download unsloth/GLM-5.2-GGUF" >&2
    exit 75
  fi
}

glm_status() {
  local root=/mnt/raid0/llm/models/GLM-5.2-UD-IQ2_M
  local finalized incomplete avail
  du -sh "${root}" || true
  pgrep -af "hf download unsloth/GLM-5.2-GGUF" || true
  finalized=$(find "${root}" -type f -name '*.gguf' ! -path '*/.cache/*' | wc -l)
  incomplete=$(find "${root}" -type f \( -name '*.incomplete' -o -name '*.lock' \) | wc -l)
  avail=$(df -h /mnt/raid0/llm/models | awk 'NR==2{print $4}')
  printf 'finalized_ggufs=%s incomplete_or_lock=%s avail=%s\n' "${finalized}" "${incomplete}" "${avail}"
  echo "finalized shards:"
  find "${root}" -type f -name '*.gguf' ! -path '*/.cache/*' \
    -printf '%s %TY-%Tm-%TdT%TH:%TM:%TS %p\n' | sort -nr
  echo "largest cache/incomplete markers:"
  find "${root}" -type f \( -name '*.incomplete' -o -name '*.lock' \) \
    -printf '%s %TY-%Tm-%TdT%TH:%TM:%TS %p\n' | sort -nr | head -20
}

registry_gap_status() {
  for dir in \
    /mnt/raid0/llm/models/deepseek-v4-flash \
    /mnt/raid0/llm/models/MiniCPM-o-4_5-gguf \
    /mnt/raid0/llm/models/qwen2.5-coder-14b-base \
    /mnt/raid0/llm/models/Qwen3.5-9B-MTP-GGUF \
    /mnt/raid0/llm/models/Qwen3-VL-8B-Instruct-GGUF \
    /mnt/raid0/llm/models/Qwen3-4B-Thinking-2507-GGUF
  do
    [ -d "${dir}" ] || continue
    printf '\n== %s ==\n' "${dir}"
    du -sh "${dir}"
    find "${dir}" -maxdepth 4 -type f -name '*.gguf' -printf '%s %p\n' | sort -nr | head -30
    locks=$(find "${dir}" \( -name '*.incomplete' -o -name '*.lock' \) | wc -l)
    printf 'cache_markers=%s\n' "${locks}"
  done
}

bonsai_q1_cpu() {
  require_no_glm_download
  v7_cpu_llama \
    -m /mnt/raid0/llm/models/bonsai-27b/Bonsai-27B-Q1_0.gguf \
    -ngl 0 -t "${THREADS:-96}" -c 2048 -n 64 \
    -p 'Return exactly: ok'
}

bonsai_q1_mi210_v7() {
  require_no_glm_download
  v7_mi210_llama \
    -m /mnt/raid0/llm/models/bonsai-27b/Bonsai-27B-Q1_0.gguf \
    -ngl 99 -c 2048 -n 64 \
    -p 'Return exactly: ok'
}

bonsai_dspark_cpu_v7() {
  require_no_glm_download
  v7_cpu_llama \
    -m /mnt/raid0/llm/models/bonsai-27b/Bonsai-27B-dspark-Q4_1.gguf \
    -ngl 0 -t "${THREADS:-96}" -c 2048 -n 64 \
    -p 'Return exactly: ok'
}

bonsai_dspark_mi210_v7() {
  require_no_glm_download
  v7_mi210_llama \
    -m /mnt/raid0/llm/models/bonsai-27b/Bonsai-27B-dspark-Q4_1.gguf \
    -ngl 99 -c 2048 -n 64 \
    -p 'Return exactly: ok'
}

ternary_q2_0_mi210_v7() {
  require_no_glm_download
  v7_mi210_llama \
    -m /mnt/raid0/llm/models/ternary-bonsai-27b/Ternary-Bonsai-27B-Q2_0.gguf \
    -ngl 99 -c 2048 -n 64 \
    -p 'Return exactly: ok'
}

ternary_q2_0_cpu_v7() {
  require_no_glm_download
  v7_cpu_llama \
    -m /mnt/raid0/llm/models/ternary-bonsai-27b/Ternary-Bonsai-27B-Q2_0.gguf \
    -ngl 0 -t "${THREADS:-96}" -c 2048 -n 64 \
    -p 'Return exactly: ok'
}

ternary_bonsai_dspark_cpu_v7() {
  require_no_glm_download
  v7_cpu_llama \
    -m /mnt/raid0/llm/models/ternary-bonsai-27b/Ternary-Bonsai-27B-dspark-Q4_1.gguf \
    -ngl 0 -t "${THREADS:-96}" -c 2048 -n 64 \
    -p 'Return exactly: ok'
}

ternary_bonsai_dspark_mi210_v7() {
  require_no_glm_download
  v7_mi210_llama \
    -m /mnt/raid0/llm/models/ternary-bonsai-27b/Ternary-Bonsai-27B-dspark-Q4_1.gguf \
    -ngl 99 -c 2048 -n 64 \
    -p 'Return exactly: ok'
}

bonsai_8b_cpu_v7() {
  require_no_glm_download
  v7_cpu_llama \
    -m /mnt/raid0/llm/models/Bonsai-8B.gguf \
    -ngl 0 -t "${THREADS:-48}" -c 2048 -n 64 \
    -p 'Return exactly: ok'
}

bonsai_8b_mi210_v7() {
  require_no_glm_download
  v7_mi210_llama \
    -m /mnt/raid0/llm/models/Bonsai-8B.gguf \
    -ngl 99 -c 2048 -n 64 \
    -p 'Return exactly: ok'
}

qwable_iq4xs_mi210_v7() {
  require_no_glm_download
  v7_mi210_llama \
    -m /mnt/raid0/llm/models/Qwable-v1-GGUF/Qwable-v1.IQ4_XS.gguf \
    -ngl 99 -c 2048 -n 32 --reasoning off --reasoning-budget 0 \
    -p 'Answer in one sentence: what is the purpose of a proof assistant?'
}

qwable_iq4xs_cpu_v7() {
  require_no_glm_download
  v7_cpu_llama \
    -m /mnt/raid0/llm/models/Qwable-v1-GGUF/Qwable-v1.IQ4_XS.gguf \
    -ngl 0 -t "${THREADS:-8}" -c 1024 -n 8 --reasoning off --reasoning-budget 0 \
    -p 'Answer in one sentence: what is the purpose of a proof assistant?'
}

qwable_q8_mi210_v7() {
  require_no_glm_download
  v7_mi210_llama \
    -m /mnt/raid0/llm/models/Qwable-v1-GGUF/Qwable-v1.Q8_0.gguf \
    -ngl 99 -c 2048 -n 32 --reasoning off --reasoning-budget 0 \
    -p 'Answer in one sentence: what is the purpose of a proof assistant?'
}

hy3_build_cpu_runtime() {
  require_no_glm_download
  cd "${HY3_DIR}"
  CUDA=0 SERVER=1 JOBS="${JOBS:-32}" bash setup_hyv3_llama.sh "${HY3_BUILD}"
}

hy3_cpu_smoke() {
  require_no_glm_download
  v7_cpu_llama \
    -m "${HY3_DIR}/Hy3-IQ1_M-mtp.gguf" \
    -ngl 0 -t "${THREADS:-8}" -c 512 -n 8 -cnv --reasoning off --reasoning-budget 0 \
    -p 'Return exactly: ok'
}

deepseek_v4_flash_cpu_v7() {
  require_no_glm_download
  v7_cpu_llama \
    -m /mnt/raid0/llm/models/deepseek-v4-flash/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf \
    -ngl 0 -t "${THREADS:-96}" -c 1024 -n 16 \
    -p 'Return exactly: ok'
}

minicpm_q4_cpu_text_v7() {
  require_no_glm_download
  v7_cpu_llama \
    -m /mnt/raid0/llm/models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf \
    -ngl 0 -t "${THREADS:-48}" -c 2048 -n 64 \
    -p 'Return exactly: ok'
}

minicpm_q4_mi210_text_v7() {
  require_no_glm_download
  v7_mi210_llama \
    -m /mnt/raid0/llm/models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf \
    -ngl 99 -c 2048 -n 64 \
    -p 'Return exactly: ok'
}

qwen25_coder14_cpu_v7() {
  require_no_glm_download
  v7_cpu_llama \
    -m /mnt/raid0/llm/models/qwen2.5-coder-14b-base/Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf \
    -ngl 0 -t "${THREADS:-48}" -c 2048 -n 96 \
    -p 'Write a C++ function named add that returns the sum of two ints.'
}

qwen25_coder14_mi210_v7() {
  require_no_glm_download
  v7_mi210_llama \
    -m /mnt/raid0/llm/models/qwen2.5-coder-14b-base/Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf \
    -ngl 99 -c 2048 -n 96 \
    -p 'Write a C++ function named add that returns the sum of two ints.'
}

qwen35_9b_mtp_cpu_v7() {
  require_no_glm_download
  v7_cpu_llama \
    -m /mnt/raid0/llm/models/Qwen3.5-9B-MTP-GGUF/Qwen3.5-9B-Q4_K_M.gguf \
    -ngl 0 -t "${THREADS:-48}" -c 2048 -n 96 \
    --spec-type draft-mtp --spec-draft-n-max 2 \
    -p 'Return exactly: ok'
}

qwen35_9b_mtp_mi210_v7() {
  require_no_glm_download
  v7_mi210_llama \
    -m /mnt/raid0/llm/models/Qwen3.5-9B-MTP-GGUF/Qwen3.5-9B-Q4_K_M.gguf \
    -ngl 99 -c 2048 -n 96 \
    --spec-type draft-mtp --spec-draft-n-max 2 \
    -p 'Return exactly: ok'
}

qwen3_vl8_cpu_text_v7() {
  require_no_glm_download
  v7_cpu_llama \
    -m /mnt/raid0/llm/models/Qwen3-VL-8B-Instruct-GGUF/Qwen3VL-8B-Instruct-Q4_K_M.gguf \
    --mmproj /mnt/raid0/llm/models/Qwen3-VL-8B-Instruct-GGUF/mmproj-Qwen3VL-8B-Instruct-F16.gguf \
    -ngl 0 -t "${THREADS:-48}" -c 2048 -n 64 \
    -p 'Return exactly: ok'
}

qwen3_vl8_mi210_text_v7() {
  require_no_glm_download
  v7_mi210_llama \
    -m /mnt/raid0/llm/models/Qwen3-VL-8B-Instruct-GGUF/Qwen3VL-8B-Instruct-Q4_K_M.gguf \
    --mmproj /mnt/raid0/llm/models/Qwen3-VL-8B-Instruct-GGUF/mmproj-Qwen3VL-8B-Instruct-F16.gguf \
    -ngl 99 -c 2048 -n 64 \
    -p 'Return exactly: ok'
}

qwen3_4b_thinking_cpu_v7() {
  require_no_glm_download
  v7_cpu_llama \
    -m /mnt/raid0/llm/models/Qwen3-4B-Thinking-2507-GGUF/Qwen3-4B-Thinking-2507-Q8_0.gguf \
    -ngl 0 -t "${THREADS:-48}" -c 2048 -n 96 --reasoning off \
    -p 'Answer in one sentence: what is a proof assistant?'
}

qwen3_4b_thinking_mi210_v7() {
  require_no_glm_download
  v7_mi210_llama \
    -m /mnt/raid0/llm/models/Qwen3-4B-Thinking-2507-GGUF/Qwen3-4B-Thinking-2507-Q8_0.gguf \
    -ngl 99 -c 2048 -n 96 --reasoning off \
    -p 'Answer in one sentence: what is a proof assistant?'
}

case "${1:-}" in
  glm_status) glm_status ;;
  registry_gap_status) registry_gap_status ;;
  bonsai_q1_cpu) bonsai_q1_cpu ;;
  bonsai_q1_mi210_v7) bonsai_q1_mi210_v7 ;;
  bonsai_dspark_cpu_v7) bonsai_dspark_cpu_v7 ;;
  bonsai_dspark_mi210_v7) bonsai_dspark_mi210_v7 ;;
  ternary_q2_0_cpu_v7) ternary_q2_0_cpu_v7 ;;
  ternary_q2_0_mi210_v7) ternary_q2_0_mi210_v7 ;;
  ternary_bonsai_dspark_cpu_v7) ternary_bonsai_dspark_cpu_v7 ;;
  ternary_bonsai_dspark_mi210_v7) ternary_bonsai_dspark_mi210_v7 ;;
  bonsai_8b_cpu_v7) bonsai_8b_cpu_v7 ;;
  bonsai_8b_mi210_v7) bonsai_8b_mi210_v7 ;;
  qwable_iq4xs_cpu_v7) qwable_iq4xs_cpu_v7 ;;
  qwable_iq4xs_mi210_v7) qwable_iq4xs_mi210_v7 ;;
  qwable_q8_mi210_v7) qwable_q8_mi210_v7 ;;
  hy3_build_cpu_runtime) hy3_build_cpu_runtime ;;
  hy3_cpu_smoke) hy3_cpu_smoke ;;
  deepseek_v4_flash_cpu_v7) deepseek_v4_flash_cpu_v7 ;;
  minicpm_q4_cpu_text_v7) minicpm_q4_cpu_text_v7 ;;
  minicpm_q4_mi210_text_v7) minicpm_q4_mi210_text_v7 ;;
  qwen25_coder14_cpu_v7) qwen25_coder14_cpu_v7 ;;
  qwen25_coder14_mi210_v7) qwen25_coder14_mi210_v7 ;;
  qwen35_9b_mtp_cpu_v7) qwen35_9b_mtp_cpu_v7 ;;
  qwen35_9b_mtp_mi210_v7) qwen35_9b_mtp_mi210_v7 ;;
  qwen3_vl8_cpu_text_v7) qwen3_vl8_cpu_text_v7 ;;
  qwen3_vl8_mi210_text_v7) qwen3_vl8_mi210_text_v7 ;;
  qwen3_4b_thinking_cpu_v7) qwen3_4b_thinking_cpu_v7 ;;
  qwen3_4b_thinking_mi210_v7) qwen3_4b_thinking_mi210_v7 ;;
  *)
    echo "usage: $0 {glm_status|registry_gap_status|bonsai_q1_cpu|bonsai_q1_mi210_v7|bonsai_dspark_cpu_v7|bonsai_dspark_mi210_v7|ternary_q2_0_cpu_v7|ternary_q2_0_mi210_v7|ternary_bonsai_dspark_cpu_v7|ternary_bonsai_dspark_mi210_v7|bonsai_8b_cpu_v7|bonsai_8b_mi210_v7|qwable_iq4xs_cpu_v7|qwable_iq4xs_mi210_v7|qwable_q8_mi210_v7|hy3_build_cpu_runtime|hy3_cpu_smoke|deepseek_v4_flash_cpu_v7|minicpm_q4_cpu_text_v7|minicpm_q4_mi210_text_v7|qwen25_coder14_cpu_v7|qwen25_coder14_mi210_v7|qwen35_9b_mtp_cpu_v7|qwen35_9b_mtp_mi210_v7|qwen3_vl8_cpu_text_v7|qwen3_vl8_mi210_text_v7|qwen3_4b_thinking_cpu_v7|qwen3_4b_thinking_mi210_v7}" >&2
    exit 64
    ;;
esac
