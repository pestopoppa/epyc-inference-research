#!/bin/bash
# OCC-1 reader launch: Qwen3-VL-30B-A3B-Instruct Q4_K_M + F16 mmproj on the MI210 (ROCm0),
# champion llama.cpp ef81196d5 (build-fold). GPU ONLY: all layers and the projector offloaded.
#
# Flags mirror the production worker_vision recipe (epyc-orchestrator launch_manifest.yaml
# `vision.worker` + the 2026-08-02 KV A/B, data/vision_kv_q8_ab_20260802/) except:
#   -c 16384   the largest OCC-1 request is ~9.3k prompt + 1024 output tokens (saves ~2.4 GiB KV)
#   --port: a free TEST port (default 18431), never a production port. :8090 is the production
#           embedder; check orchestration/launch_manifest.yaml and `ss -ltn` before overriding.
#   --jinja off (as in the KV A/B), binary = champion build-fold-ef81196d5
#
# Usage: launch_reader.sh [--port N]      (or OCC1_PORT=N; --port wins)
#
# The caller (the GPU runner) owns this process: this script only execs llama-server in the
# foreground. Run it under nohup / a pane and capture its PID yourself; kill only that PID.
set -euo pipefail

BIN_DIR=/mnt/raid0/llm/tmp/build-fold-ef81196d5/bin
MODEL=/mnt/raid0/llm/models/lmstudio-community/Qwen3-VL-30B-A3B-Instruct-GGUF/Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf
MMPROJ=/mnt/raid0/llm/models/lmstudio-community/Qwen3-VL-30B-A3B-Instruct-GGUF/mmproj-Qwen3-VL-30B-A3B-Instruct-F16.gguf
PORT="${OCC1_PORT:-18431}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="${2:?--port needs a value}"; shift 2 ;;
    --port=*) PORT="${1#--port=}"; shift ;;
    *) echo "unknown argument: $1 (usage: $0 [--port N])" >&2; exit 2 ;;
  esac
done
if ! [[ "${PORT}" =~ ^[0-9]+$ ]] || (( PORT < 1024 || PORT > 65535 )); then
  echo "bad port: ${PORT}" >&2; exit 2
fi
LISTENING="$(ss -ltnH "sport = :${PORT}" 2>/dev/null || true)"
if [[ -n "${LISTENING}" ]]; then
  echo "port ${PORT} is already listening; pick another with --port" >&2; exit 1
fi
HOST_CPUS="${OCC1_HOST_CPUS:-184-191}"   # GPU host threads = SMT siblings

export LD_LIBRARY_PATH="${BIN_DIR}:/opt/rocm/lib"
RESEARCH="$(cd "$(dirname "$0")/../../.." && pwd)"
"${RESEARCH}/scripts/utils/verify_ggml_linkage.sh" "${BIN_DIR}/llama-server" "${BIN_DIR}" >&2

exec taskset -c "${HOST_CPUS}" "${BIN_DIR}/llama-server" \
  -m "${MODEL}" --mmproj "${MMPROJ}" \
  --host 127.0.0.1 --port "${PORT}" \
  -np 1 -c 16384 -t 8 \
  --flash-attn on --device ROCm0 -ngl 999 \
  -ctk q8_0 -ctv q8_0 \
  --image-min-tokens 1024 --cache-ram 0 \
  --log-colors off
