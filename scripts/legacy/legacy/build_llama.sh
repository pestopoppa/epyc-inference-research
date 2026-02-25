#!/bin/bash

# Optimized Build Script for llama.cpp from outside the source directory
# Supports HugePages, mlock, OpenBLAS, and EPYC CPU tuning

set -e

# === CONFIGURATION ===
LLAMA_DIR="${1:-$(pwd)/llama.cpp}" # Pass path as first arg or default to ./llama.cpp
BUILD_DIR="$LLAMA_DIR/build"

# === VALIDATION ===
if [ ! -d "$LLAMA_DIR" ] || [ ! -f "$LLAMA_DIR/CMakeLists.txt" ]; then
  echo "[ERROR] $LLAMA_DIR is not a valid llama.cpp directory"
  exit 1
fi

echo "[INFO] Building llama.cpp from: $LLAMA_DIR"

# === CLEAN BUILD DIRECTORY ===
if [ -d "$BUILD_DIR" ]; then
  echo "[INFO] Cleaning previous build..."
  rm -rf "$BUILD_DIR"
fi

# === CONFIGURE ===
echo "[INFO] Configuring build..."
cmake -B "$BUILD_DIR" -S "$LLAMA_DIR" \
  -DLLAMA_CURL=ON \
  -DGGML_BLAS=ON \
  -DGGML_BLAS_VENDOR=OpenBLAS \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_NATIVE=ON \
  -DLLAMA_ACCELERATE=ON \
  -DLLAMA_HUGE_PAGES=ON \
  -DLLAMA_MLOCK=ON

# === COMPILE ===
echo "[INFO] Building targets..."
cmake --build "$BUILD_DIR" --config Release -j --clean-first \
  --target llama-quantize llama-cli llama-gguf-split

echo "[DONE] Build complete. Binaries are in: $BUILD_DIR/bin"
