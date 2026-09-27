#!/bin/bash
# Regenerates every file in this directory from EXISTING binaries (read-only).
# No build, no GPU: ELF parsing plus ROCm 6.2 llvm-readelf/llvm-objdump only.
# Pinned off the DS41 measurement cores (0-47/0-95).
set -euo pipefail
cd "$(dirname "$0")"
TOOL=../../scripts/kernel_rnd/gfx90a_isa_audit.py
V10=/mnt/raid0/llm/kernels/production/gpu/libggml-hip.so.0.16.0
CAND=/mnt/raid0/llm/llama.cpp-experimental-mmq-jcap-20260927/build-hip-jcap/bin/libggml-hip.so.0.16.0
V10_COMMIT=ffc1bac82eeca6f9099e1ccd9ba49703c460a115
CAND_COMMIT=cb28e8bd29b4765e2ab3e21c68155afad8c7d343
RUN="nice -n 19 taskset -c 96-103 python3 $TOOL"
sha256sum "$V10" "$CAND" > binaries.sha256
$RUN audit "$V10" --families mmq --jobs 8 --label v10-production-consolidated \
    --category BASELINE --source-commit "$V10_COMMIT" \
    --flags "production-consolidated-v10 GPU recipe: HIP gfx90a Release -O3, GGML_HIP_ROCWMMA_FATTN=ON GGML_HIP_MMQ_MFMA=ON GGML_HIP_GRAPHS=ON GGML_HIP_NO_VMM=ON, ROCm 6.2 clang 18" \
    --json audit_v10_mmq_baseline.json --table table_v10_mmq_baseline.txt --sort name > /dev/null
$RUN audit "$CAND" --families mmq --jobs 8 --label mmq-jcap-candidate \
    --category CANDIDATE --source-commit "$CAND_COMMIT" \
    --flags "production-consolidated-v10 GPU recipe (HIP compile flags identical to kernels/builds/gpu-20260921-ffc1bac82) + per-type CDNA MMQ J cap" \
    --json audit_jcap_mmq_candidate.json --table table_jcap_mmq_candidate.txt --sort name > /dev/null
# Accept gate: candidate vs incumbent. Exit 1 means at least one FAIL.
set +e
$RUN diff audit_v10_mmq_baseline.json audit_jcap_mmq_candidate.json --families mmq \
    --json diff_v10_to_jcap_mmq.json > diff_v10_to_jcap_mmq.txt
echo "$?" > diff_exit_code.txt
set -e
$RUN table audit_jcap_mmq_candidate.json --families mmq --sort spill --limit 40 > top_mmq_spill_jcap.txt
gzip -n -9 -f audit_v10_mmq_baseline.json audit_jcap_mmq_candidate.json
