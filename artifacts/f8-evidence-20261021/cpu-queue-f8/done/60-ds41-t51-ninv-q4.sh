#!/bin/bash
#maxjob_s=5400
# DS41-T5.1 n-invariance audit of the verify path (CPU), Sept Q4_K target. Harness lib/ds41_ninv.cpp: ONE single-file compile into the parity dir
# against the DS41 build libs (no tree/build-dir mutation); libllama API only; syntax-checked, never run -> exit 7 on harness failure.
# Q4_K expert kernel differs from MXFP4 (1017): MoE-gemv ranking is Q4-specific; attention/compressor/indexer rows carry over.
# 1017 rerun: held/60-ds41-t51-ninv-1017.sh.
# Resources: 483 GiB shared mapping, <30 GiB anon, threads=$(nproc); load-dominated (5 contexts x 96-token prefill + <=6 decodes). needs_full_host=0. No server.
set -euo pipefail
. /mnt/raid0/llm/tmp/ds41-parity-20261021/lib/common.sh
J=ds41-t51-ninv-q4; OUT=$PAR/$J; mkdir -p $OUT
ds41_check_bin
bash $PAR/lib/build_ninv.sh
$PY -I -c "print(' '.join(f'Line {j}: the committee reviewed item {j*3} and recorded its decision.' for j in range(30)))" > $OUT/prompt.txt
rc=0; $PAR/bin/ds41_ninv -m "${MODEL:-$Q4}" --prompt-file $OUT/prompt.txt -t "$(nproc)" --prefix 96 --ns 1,2,4,6 -o $OUT/ninv.tsv > $OUT/run.log 2>&1 || rc=$?
[ $rc = 0 ] || { echo "RESULT $J HARNESS_FAILED rc=$rc (see $OUT/run.log)"; exit 7; }
$PY -I $PAR/lib/rank_ninv.py $OUT/ninv.tsv > $OUT/ranked.txt
echo "RESULT $J era=sept-q4 $(tr '\n' ' ' < $OUT/ranked.txt)"
