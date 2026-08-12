#!/bin/bash
# Shared GPU environment for WG-LFMI-1/2. Prepends the HIP tree so libggml-hip.so
# is the one dlopened; the ambient LD_LIBRARY_PATH is known-contaminated with the
# CPU build/bin (INC-20260731-ggml-linkage-silent-cpu-fallback).
export HIPBIN=/mnt/raid0/llm/llama.cpp/build-hip/bin
export LD_LIBRARY_PATH="$HIPBIN:/opt/rocm/lib"
export GPU_LANE="184-191"       # orchestration/stack_topology.yaml:220, declared GPU host lane
export OUT=/workspace/tmp/wg-lfmi
export M4=/mnt/raid0/llm/models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf
export M8=/mnt/raid0/llm/models/LFM2.5-1.2B-Instruct-Q8_0.gguf
ulimit -c 0
