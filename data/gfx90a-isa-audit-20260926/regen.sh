#!/bin/bash
# Regenerates every file in this directory from EXISTING binaries (read-only).
# No build, no GPU: ELF parsing plus ROCm 6.2 llvm-readelf/llvm-objdump only.
# Pinned off the DS41 measurement cores (0-47/0-95).
set -euo pipefail
cd "$(dirname "$0")"
TOOL=../../scripts/kernel_rnd/gfx90a_isa_audit.py
V10=/mnt/raid0/llm/kernels/production/gpu/libggml-hip.so.0.16.0
V9=/mnt/raid0/llm/kernels/builds/gpu-20260810-0db32c06e/bin/libggml-hip.so.0.16.0
SCOPE=mmq,mmf,fattn_mma,fattn_wmma
RUN="taskset -c 72-79 python3 $TOOL"
sha256sum "$V10" "$V9" > binaries.sha256
# Governed scope (INF03-REGAUDIT-1): MMQ, MMA-FA, rocWMMA FA, mul_mat_f.
$RUN audit "$V10" --families "$SCOPE" --label v10-production-consolidated --json audit_v10_scope.json --table table_v10_scope.txt > /dev/null
$RUN audit "$V9" --families "$SCOPE" --label v9-rollback-anchor --json audit_v9_scope.json > /dev/null
# Whole-library census (summary only; the full JSON is ~15 MB and regenerable).
$RUN audit "$V10" --families all --label v10-production-consolidated --json /dev/null > summary_v10_all.txt
# Gate demonstration: v10 (candidate) against the v9 rollback anchor (baseline).
$RUN diff audit_v9_scope.json audit_v10_scope.json --json diff_v9_to_v10.json > diff_v9_to_v10.txt || true
$RUN table audit_v10_scope.json --families mmq --sort spill --limit 60 > top_mmq_spill.txt
$RUN table audit_v10_scope.json --families fattn_mma --sort spill --limit 60 > top_fattn_mma_spill.txt
$RUN table audit_v10_scope.json --sort acc --limit 60 > top_accvgpr_copy_tax.txt
$RUN table audit_v10_scope.json --families fattn_wmma --sort acc > top_fattn_wmma_acc.txt
$RUN table audit_v10_scope.json --families mmf --sort acc --limit 60 > top_mmf_acc.txt
gzip -n -9 -f audit_v10_scope.json audit_v9_scope.json
