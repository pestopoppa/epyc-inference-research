#!/bin/bash
# Start sd-server (stable-diffusion.cpp native, ggml backend).
#
# Invoked by orchestrator_stack.py's start_sd_server(); not intended as a
# user-facing entry point. Replaces the prior ComfyUI launcher (sd.cpp
# delivers ~1.7-3.4× CPU speedup over the ComfyUI-GGUF + PyTorch path
# because Q8_0 weights stay packed and ggml's native quantized GEMM
# kernels skip the per-layer dequant-to-BF16 step).
#
# Usage: ./start_sd_server.sh [--port N] [--listen IP]

set -euo pipefail

SD_BIN="${SD_BIN:-/mnt/raid0/llm/stable-diffusion.cpp/build/bin/sd-server}"
PORT="${SD_SERVER_PORT:-8190}"
LISTEN="${SD_SERVER_LISTEN:-127.0.0.1}"

# ERNIE-Image-Turbo Q8 stack (matches the model files Phase 1 wired up).
DIFFUSION_MODEL="${DIFFUSION_MODEL:-/mnt/raid0/llm/models/diffusion/ernie-image-turbo-gguf/ernie-image-turbo-Q8_0.gguf}"
VAE="${VAE:-/mnt/raid0/llm/models/diffusion/ernie-image-turbo-comfy/vae/flux2-vae.safetensors}"
LLM="${LLM:-/mnt/raid0/llm/models/diffusion/ernie-image-turbo-comfy/text_encoders/ministral-3-3b.safetensors}"

THREADS="${SD_THREADS:-96}"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --port)
      PORT="$2"
      shift 2
      ;;
    --listen)
      LISTEN="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 [--port N] [--listen IP]" >&2
      exit 1
      ;;
  esac
done

if [[ ! -x "$SD_BIN" ]]; then
  echo "ERROR: sd-server binary not found or not executable: $SD_BIN" >&2
  echo "  Build it: cd /mnt/raid0/llm/stable-diffusion.cpp && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j" >&2
  exit 2
fi

for f in "$DIFFUSION_MODEL" "$VAE" "$LLM"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: model file not found: $f" >&2
    exit 3
  fi
done

# Canonical CPU baseline: numactl --interleave=all + full-host thread span.
# Flags:
#   --diffusion-fa            flash attention in the DiT (default-on)
#   --diffusion-conv-direct   ggml_conv2d_direct in the DiT (quality-neutral, may help cache behavior)
#   --vae-conv-direct         ggml_conv2d_direct in the VAE (quality-neutral, target ~30% of wall-clock today)
exec numactl --interleave=all -- "$SD_BIN" \
  --diffusion-model "$DIFFUSION_MODEL" \
  --vae "$VAE" \
  --llm "$LLM" \
  -t "$THREADS" \
  --diffusion-fa \
  --diffusion-conv-direct \
  --vae-conv-direct \
  --listen-ip "$LISTEN" \
  --listen-port "$PORT"
