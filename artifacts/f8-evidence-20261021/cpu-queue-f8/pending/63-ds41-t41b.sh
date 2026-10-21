#!/bin/bash
#maxjob_s=5400
# DS41-T4.1.1 golden exactness rerun, Sept Q4 + Sept DSpark. Same arms as job 60-ds41-t41-golden-exactness (plain / dspark_batched / dspark_serial).
# Root cause of the empty serial rows: server-context.cpp sample_and_accept_dspark_serial() re-decodes ctx_tgt one token at a time and
#  carries TODO(TD-1d.2) "n_probs stays empty for tokens accepted through this path" => draft-accepted tokens get no logprob/top-5 on the serial arm.
# Harness (lib/t41.py v2): every row records probs_available; the diff compares logprob/top-5 ONLY when both arms returned them, otherwise token ids only and the
#  logprob comparison is reported NOT AVAILABLE / PARTIAL with counts (empty lists are never compared as data; logprob 0.0 of an empty row is recorded as None).
# Resources: 483 GiB shared mapping of Q4 (+10 GiB drafter), ~20 GiB anon; threads=$(nproc); est. 25-40 min (job 60 took 25 min). needs_full_host=0. Ports 18966-18968.
set -euo pipefail
. /mnt/raid0/llm/tmp/ds41-parity-20261021/lib/common.sh
J=ds41-t41b; OUT=$PAR/$J; mkdir -p $OUT; trap kill_srv EXIT
ds41_check_bin
run() { # label port env-assign extra-args...
  local label=$1 port=$2 envv=$3; shift 3
  env $envv "$BIN/llama-server" -m "$Q4" --host 127.0.0.1 --port $port -t "$(nproc)" -tb "$(nproc)" -np 1 -c 4608 -fa on --no-warmup --no-webui --cache-ram 0 -s 1 "$@" > $OUT/server-$label.log 2>&1 &
  SRV_PID=$!
  wait_ready $port 2400
  $PY -I $PAR/lib/t41.py gen $port $OUT/rows-$label.json $label | tee $OUT/gen-$label.log
  kill_srv
}
run plain 18966 LLAMA_SPEC_EXACT=off
run dspark_batched 18967 LLAMA_SPEC_EXACT=batched-greedy-inexact --spec-type draft-dspark -md "$DSPARK" --spec-draft-n-max 3
$PY -I $PAR/lib/t41.py diff $OUT/rows-plain.json $OUT/rows-dspark_batched.json $OUT/diff-plain-vs-batched.json | tee $OUT/summary.txt
run dspark_serial 18968 LLAMA_SPEC_EXACT=serial --spec-type draft-dspark -md "$DSPARK" --spec-draft-n-max 3
$PY -I $PAR/lib/t41.py diff $OUT/rows-plain.json $OUT/rows-dspark_serial.json $OUT/diff-plain-vs-serial.json | tee -a $OUT/summary.txt
echo "RESULT $J era=sept-q4 $(tr '\n' ' ' < $OUT/summary.txt)"
