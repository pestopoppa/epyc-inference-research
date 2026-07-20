#!/usr/bin/env bash
set -euo pipefail

# DR-0 dry-run bundle. These are future live-run templates; this script does not run them.
# Production v6 is intentionally absent. Use only llama.cpp-experimental.
pgrep -af 'llama-bench|llama-server|llama-cli|llama-mtmd-cli' || true
rocm-smi --showpids || true

# arm: cpu_high_quant_verifier_baseline
# purpose: High-quant CPU verifier baseline for the selected task classes
LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin OMP_NUM_THREADS=1 PATH=/usr/bin:/bin:/opt/rocm/bin /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf --host 127.0.0.1 --port 19730 -np 1 -c 8192 -t 96 -ub 1024 --metrics --slots --jinja --reasoning off --device none -ngl 0 --spec-type none

pgrep -af 'llama-bench|llama-server|llama-cli|llama-mtmd-cli' || true
rocm-smi --showpids || true

# arm: mi210_aggressive_drafter_alone_k2
# purpose: MI210 resident aggressive same-family artifact, measured without CPU verifier
HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin OMP_NUM_THREADS=1 PATH=/usr/bin:/bin:/opt/rocm/bin ROCR_VISIBLE_DEVICES=0 /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-IQ2_M/Qwen3.5-122B-A10B-UD-IQ2_M.gguf --host 127.0.0.1 --port 19731 -np 1 -c 8192 -t 32 -ub 1024 --metrics --slots --jinja --reasoning off --device ROCm0 -ngl all -ctk q4_0 -ctv f16 -fa on --spec-type ngram-mod,draft-mtp --spec-draft-n-max 2

pgrep -af 'llama-bench|llama-server|llama-cli|llama-mtmd-cli' || true
rocm-smi --showpids || true

# arm: quant_asymmetric_combined_k2
# purpose: CPU 122B Q4 verifier with MI210 122B IQ2 MTP drafter, only valid when F/H are observable
HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin OMP_NUM_THREADS=1 PATH=/usr/bin:/bin:/opt/rocm/bin ROCR_VISIBLE_DEVICES=0 /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf -md /mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-IQ2_M/Qwen3.5-122B-A10B-UD-IQ2_M.gguf --host 127.0.0.1 --port 19732 -np 1 -c 8192 -t 96 -ub 1024 --metrics --slots --jinja --reasoning off --device none -ngl 0 --spec-type draft-mtp --spec-draft-device ROCm0 --spec-draft-ngl all --spec-draft-n-max 2

pgrep -af 'llama-bench|llama-server|llama-cli|llama-mtmd-cli' || true
rocm-smi --showpids || true
