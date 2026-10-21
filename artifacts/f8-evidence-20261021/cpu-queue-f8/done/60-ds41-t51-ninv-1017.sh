#!/bin/bash
#maxjob_s=5400
# DS41-T5.1 n-invariance audit of the verify path (CPU), Sept Q4_K target. Harness lib/ds41_ninv.cpp: ONE single-file compile into the parity dir
# against the DS41 build libs (no tree/build-dir mutation); libllama API only; syntax-checked, never run -> exit 7 on harness failure.
# 1017 MXFP4 experts (CPU_REPACK mxfp4_8x8_q8_0): the authoritative MoE-gemv n-invariance list for the new era.
# GUARDED: exit 3 DEP_MISSING until the 1017 download is verified. Move to pending/ only after that (a pending job exiting 3 is consumed, not requeued).
# Resources: 483 GiB shared mapping, <30 GiB anon, threads=$(nproc); load-dominated (5 contexts x 96-token prefill + <=6 decodes). needs_full_host=0. No server.
set -euo pipefail
. /mnt/raid0/llm/tmp/ds41-parity-20261021/lib/common.sh
J=ds41-t51-ninv-1017; OUT=$PAR/$J; mkdir -p $OUT
ds41_need_1017; ds41_check_bin
bash $PAR/lib/build_ninv.sh
$PY -I -c "print(' '.join(f'Line {j}: the committee reviewed item {j*3} and recorded its decision.' for j in range(30)))" > $OUT/prompt.txt
rc=0; $PAR/bin/ds41_ninv -m "$M1017" --prompt-file $OUT/prompt.txt -t "$(nproc)" --prefix 96 --ns 1,2,4,6 -o $OUT/ninv.tsv > $OUT/run.log 2>&1 || rc=$?
[ $rc = 0 ] || { echo "RESULT $J HARNESS_FAILED rc=$rc (see $OUT/run.log)"; exit 7; }
$PY -I $PAR/lib/rank_ninv.py $OUT/ninv.tsv > $OUT/ranked.txt
echo "RESULT $J era=1017-mxfp4 $(tr '\n' ' ' < $OUT/ranked.txt)"
