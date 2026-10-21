#!/bin/bash
#maxjob_s=5400
# DS41-T5.1.1 n-invariance audit v2 (harness lib/ds41_ninv.cpp marker ninv_v2_inputs_matched): a node is n-variant only if its output row differs AND
# every comparable input is bit-identical to the single-row reference (strict); incomparable-input nodes are cand, differing-input nodes are prop (not attributed).
# Rel diff uses a floor of 1e-6*max|ref slice| (max_abs reported too). Sept Q4 arm always; 1017 arm only when the 1017 download is verified (else skipped with a note).
# The binary is (re)built here ONLY if bin/ds41_ninv lacks the v2 marker (single-file g++ into the parity dir; this job owns q0 while it runs).
# Resources: 483 GiB shared mapping, <40 GiB anon (input hashing adds copies), threads=$(nproc). needs_full_host=0. No server.
set -euo pipefail
. /mnt/raid0/llm/tmp/ds41-parity-20261021/lib/common.sh
J=ds41-t51b; OUT=$PAR/$J; mkdir -p $OUT
ds41_check_bin
grep -qaF ninv_v2_inputs_matched $PAR/bin/ds41_ninv 2>/dev/null || bash $PAR/lib/build_ninv.sh
grep -qaF ninv_v2_inputs_matched $PAR/bin/ds41_ninv || { echo "RESULT $J BUILD_FAILED (no v2 marker)"; exit 7; }
$PY -I -c "print(' '.join(f'Line {j}: the committee reviewed item {j*3} and recorded its decision.' for j in range(30)))" > $OUT/prompt.txt
arm() { # label model
  local label=$1 model=$2 rc=0
  mkdir -p $OUT/$label
  $PAR/bin/ds41_ninv -m "$model" --prompt-file $OUT/prompt.txt -t "$(nproc)" --prefix 96 --ns 1,2,4,6 -o $OUT/$label/ninv.tsv > $OUT/$label/run.log 2>&1 || rc=$?
  [ $rc = 0 ] || { echo "RESULT $J $label HARNESS_FAILED rc=$rc (see $OUT/$label/run.log)"; return 7; }
  $PY -I $PAR/lib/rank_ninv.py $OUT/$label/ninv.tsv > $OUT/$label/ranked.txt
  echo "RESULT $J era=$label $(tr '\n' ' ' < $OUT/$label/ranked.txt)"
}
rc_all=0
arm sept-q4 "${MODEL:-$Q4}" || rc_all=7
if ( ds41_need_1017 ) >/dev/null 2>&1; then arm 1017-mxfp4 "$M1017" || rc_all=7
else echo "RESULT $J era=1017-mxfp4 SKIPPED (1017 weights not verified; rerun this job's 1017 arm later)"; fi
exit $rc_all
