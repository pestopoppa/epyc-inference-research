#!/bin/bash
#maxjob_s=7200
# DS41-T4.1 golden exactness driver, Sept Q4 + Sept DSpark (the only matching target/drafter pair; 1017 needs the W6 drafter -> not runnable on 1017).
# Era: sept-q4. Plain (no drafter) vs DSpark batched (LLAMA_SPEC_EXACT=batched-greedy-inexact) vs DSpark serial control.
# 8 prompts x ctx{512,2048} x 32 greedy rows, zero tolerance on token id + logprob + top-5. Observation grade: unit=sequence, noise floor not yet named.
# Resources: 483 GiB shared page-cache mapping of Q4 (+10 GiB drafter), ~20 GiB anon; threads=$(nproc) (one quarter; est ~3-4 t/s plain decode,
#  ~40 t/s prefill). Est. runtime 90-110 min incl. 3 server starts -> near the cap. needs_full_host=0. Ports 18961-18963.
set -euo pipefail
. /mnt/raid0/llm/tmp/ds41-parity-20261021/lib/common.sh
J=ds41-t41-golden-exactness; OUT=$PAR/$J; mkdir -p $OUT; trap kill_srv EXIT
ds41_check_bin
run() { # label port env-assign extra-args...
  local label=$1 port=$2 envv=$3; shift 3
  env $envv "$BIN/llama-server" -m "$Q4" --host 127.0.0.1 --port $port -t "$(nproc)" -tb "$(nproc)" -np 1 -c 4608 -fa on --no-warmup --no-webui --cache-ram 0 -s 1 "$@" > $OUT/server-$label.log 2>&1 &
  SRV_PID=$!
  wait_ready $port 2400
  $PY -I $PAR/lib/t41.py gen $port $OUT/rows-$label.json $label | tee $OUT/gen-$label.log
  kill_srv
}
run plain 18961 LLAMA_SPEC_EXACT=off
run dspark_batched 18962 LLAMA_SPEC_EXACT=batched-greedy-inexact --spec-type draft-dspark -md "$DSPARK" --spec-draft-n-max 3
$PY -I $PAR/lib/t41.py diff $OUT/rows-plain.json $OUT/rows-dspark_batched.json $OUT/diff-plain-vs-batched.json | tee $OUT/summary.txt
run dspark_serial 18963 LLAMA_SPEC_EXACT=serial --spec-type draft-dspark -md "$DSPARK" --spec-draft-n-max 3
$PY -I $PAR/lib/t41.py diff $OUT/rows-plain.json $OUT/rows-dspark_serial.json $OUT/diff-plain-vs-serial.json | tee -a $OUT/summary.txt
echo "RESULT $J era=sept-q4 $(tr '\n' ' ' < $OUT/summary.txt)"
