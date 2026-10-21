#!/bin/bash
#maxjob_s=5400
# KPF-17 guard-removed variants must FAIL, Qwen3.5-0.8B; needs 10-kpf17-build
set -euo pipefail
JOB=21-kpf17-mutants-tiny
WT=/mnt/raid0/llm/worktrees/kv-prefix-fork-p1r2-20261020
S=/mnt/raid0/llm/tmp/kv-prefix-fork/p1r2-20261020
OUT=$S/queue-results/$JOB
B=$WT/build-cpu/bin
PY=/workspace/repos/epyc-orchestrator/.venv/bin/python3
M=/mnt/raid0/llm/models
HYB=$M/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf
ATT=$M/Qwen2.5-0.5B-Instruct-f16.gguf
SWA=$M/gemma-3-1b-it-Q8_0.gguf
FRONT=$M/Qwen3.6-35B-A3B-MTP-Q8_0.gguf
NT=$(nproc)
mkdir -p "$OUT"
export LD_LIBRARY_PATH=$B
PASS=0; FAIL=0
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$OUT/job.log"; }
# kill only OUR OWN descendants (pytest-spawned llama-server children), by captured parent PID, never by name
kill_tree() {
  local p c
  for p in "$@"; do
    for c in $(ps -o pid= --ppid "$p" 2>/dev/null); do kill_tree "$c"; done
    kill "$p" 2>/dev/null || true
  done
}
on_exit() {
  local me=$$ c
  for c in $(ps -o pid= --ppid "$me" 2>/dev/null); do kill_tree "$c"; done
  sleep 2
  for c in $(ps -o pid= --ppid "$me" 2>/dev/null); do kill -9 "$c" 2>/dev/null || true; done
}
trap on_exit EXIT
dep_check() {
  local ok=$S/queue-results/10-kpf17-build/build.ok
  if [ ! -f "$ok" ]; then echo "DEP_MISSING: $ok (job 10-kpf17-build did not succeed)"; echo "RESULT $JOB pass=0 fail=0 DEP_MISSING"; exit 3; fi
  local want have
  want=$(awk '/^llama_server_sha256=/{sub(/^[^=]*=/,"");print}' "$ok"); have=$(sha256sum "$B/llama-server" | cut -d' ' -f1)
  if [ "$want" != "$have" ]; then echo "DEP_MISSING: build.ok sha $want != binary $have (build changed/stale)"; echo "RESULT $JOB pass=0 fail=0 DEP_MISSING"; exit 3; fi
  [ -x "$B/test-kpf-swa-checkpoint" ] || { echo "DEP_MISSING: test-kpf-swa-checkpoint"; echo "RESULT $JOB pass=0 fail=0 DEP_MISSING"; exit 3; }
  grep -qaF LLAMA_KPF_MUTATE "$B/libllama-server-impl.so" || { echo "DEP_MISSING: mutation hooks not compiled into the binary"; echo "RESULT $JOB pass=0 fail=0 DEP_MISSING"; exit 3; }
  log "provenance: branch $(git -C $WT rev-parse HEAD) llama-server sha256 $have"
}
note() { # label status detail
  echo "$1 $2 $3" >> "$OUT/steps.txt"; log "STEP $1 $2 $3"
  case "$2" in OK) PASS=$((PASS+1));; FAIL) FAIL=$((FAIL+1));; esac
}
# run_pt <label> <model> <expect pass|fail> <mutate or -> <spec type or -> <dft_full 0|1> <port> <kexpr>
run_pt() {
  local label=$1 model=$2 expect=$3 mut=$4 spec=$5 dft=$6 port=$7 kexpr=$8
  local lg=$OUT/$label.log sd=$OUT/slots-$label
  rm -rf "$sd"; mkdir -p "$sd"
  local rc=0
  local -a xenv=()
  [ "$mut" != "-" ] && xenv+=("LLAMA_KPF_MUTATE=$mut")
  [ "$spec" != "-" ] && xenv+=("KPF_SPEC_TYPE=$spec" "KPF_SPEC_N_MAX=4")
  ( cd "$WT/tools/server/tests" && \
    env LLAMA_SERVER_BIN_PATH="$B/llama-server" LLAMA_TEST_FORK_MODEL="$model" PORT=$port LLAMA_TEST_THREADS=$NT \
        PYTHONPATH=$S/pystub KPF_SLOT_SAVE_DIR="$sd" KPF_RECORD="$OUT/$label.records.jsonl" KPF_DFT_FULL=$dft \
        "${xenv[@]}" \
        "$PY" -m pytest -rA -p no:cacheprovider unit/test_slot_fork.py -k "$kexpr" ) > "$lg" 2>&1 || rc=$?
  local np nf ns
  np=$(grep -c '^PASSED' "$lg" || true); nf=$(grep -cE '^(FAILED|ERROR)' "$lg" || true); ns=$(grep -c '^SKIPPED' "$lg" || true)
  rm -rf "$sd"
  if [ "$expect" = pass ]; then
    if [ "$nf" -eq 0 ] && [ "$np" -ge 1 ]; then note "$label" OK "passed=$np skipped=$ns"
    elif [ "$nf" -eq 0 ]; then note "$label" SKIP "all skipped=$ns (model/mode does not exercise these cases)"
    else note "$label" FAIL "passed=$np failed=$nf skipped=$ns rc=$rc"; fi
  else
    if [ "$nf" -ge 1 ]; then note "$label" OK "mutation $mut DETECTED (failed=$nf passed=$np)"
    elif [ "$np" -ge 1 ]; then note "$label" FAIL "MUTATION_NOT_DETECTED mut=$mut passed=$np"
    else note "$label" SKIP "mutation run skipped all (skipped=$ns)"; fi
  fi
}
# run_cmd <label> <expect pass|fail> <cmd...>   (C++ tests)
run_cmd() {
  local label=$1 expect=$2; shift 2
  local rc=0
  "$@" > "$OUT/$label.log" 2>&1 || rc=$?
  if [ "$expect" = pass ]; then
    [ $rc -eq 0 ] && note "$label" OK "rc=0" || note "$label" FAIL "rc=$rc"
  else
    [ $rc -ne 0 ] && note "$label" OK "mutation DETECTED rc=$rc" || note "$label" FAIL "MUTATION_NOT_DETECTED"
  fi
}
finish() {
  echo "RESULT $JOB pass=$PASS fail=$FAIL" | tee -a "$OUT/job.log"
  [ "$FAIL" -eq 0 ]
}
dep_check
PORT=18984
# every case must FAIL with its guard removed (LLAMA_KPF_MUTATE on the server). SKIP = the case did not run.
run_pt mut-17.1-1-ckpt_ntokens_skew         "$HYB" fail ckpt_ntokens_skew - 0 $PORT "kpf171_pinned_ntokens_equal_requested"
run_pt mut-17.1-2-ckpt_ntokens_skew         "$HYB" fail ckpt_ntokens_skew - 0 $PORT "kpf171_offset_oracle_under_concurrency"
run_pt mut-17.1-2-skew+no_posmax_guard      "$HYB" fail ckpt_ntokens_skew,no_posmax_guard - 0 $PORT "kpf171_offset_oracle_under_concurrency"
run_pt mut-17.1-4-fork_no_lcp               "$HYB" fail fork_no_lcp - 0 $PORT "kpf171_key_check_negative_control"
run_pt mut-17.2-1-fork_skip_ckpt_load       "$HYB" fail fork_skip_ckpt_load - 0 $PORT "kpf172_source_diverges_below_junction"
run_pt mut-17.2-2-fork_skip_ckpt_load       "$HYB" fail fork_skip_ckpt_load - 0 $PORT "kpf172_fork_same_iteration_as_source_erase"
run_pt mut-17.2-3-fork_clobber_src          "$HYB" fail fork_clobber_src - 0 $PORT "kpf172_fork_from_decoding_source_no_leak"
run_pt mut-17.2-4-skew+no_posmax_guard      "$HYB" fail ckpt_ntokens_skew,no_posmax_guard - 0 $PORT "kpf172_fork_from_checkpoint_of_current_step"
run_pt mut-17.3-pin_cap_off                 "$HYB" fail pin_cap_off - 0 $PORT "kpf173_pinned_cap_per_slot"
finish
