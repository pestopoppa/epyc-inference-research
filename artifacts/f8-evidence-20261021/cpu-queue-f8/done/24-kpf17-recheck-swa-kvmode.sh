#!/bin/bash
#maxjob_s=3600
# re-check of jobs 22+23 after the KPF-17 conformance fixes (f3a3bb4b1 1f9b2a8bf 1fd7e2e12); needs 10-kpf17-build (build.ok refreshed)
set -euo pipefail
JOB=24-kpf17-recheck-swa-kvmode
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
PORT=18986
# --- KPF-17.4 C++ (job 23 swa-cpp-*): matched control + measured floor + byte-exact restore
run_cmd swa-cpp-normal pass "$B/test-kpf-swa-checkpoint" -m "$SWA" -t "$NT"
KPF_SWA_MUTATE=use_recorded_range run_cmd swa-cpp-mut-use_recorded_range fail "$B/test-kpf-swa-checkpoint" -m "$SWA" -t "$NT"
# non-SWA control: prints the batch-boundary floor and SKIPs (rc 0)
run_cmd swa-cpp-floor-nonswa pass "$B/test-kpf-swa-checkpoint" -m "$ATT" -t "$NT"
grep -h -E '^KPF174|floor' "$OUT"/swa-cpp-*.log > "$OUT/kpf174_recorded_vs_real_pos_min.txt" || true
# --- KPF-17.4 server + the 17.1/17.2/17.3 cases on the SWA model (job 23 swa-server-*)
run_pt swa-server-fork "$SWA" pass - - 0 $PORT "kpf174"
run_pt swa-server-mut-skew "$SWA" fail ckpt_ntokens_skew - 0 $PORT "kpf174"
run_pt swa-server-171-172-173 "$SWA" pass - - 0 $PORT "kpf171 or kpf172 or kpf173"
# --- kv fork mode (job 22): abort-then-retry now aborts mid-prefill; the kv-mode pos_max guard is unreachable alone,
# proven with the abort_keeps_tail fault injection
run_pt kvmode-abort-then-retry "$ATT" pass - - 0 $PORT "kpf171_abort_then_retry"
run_pt kvmode-all-new-cases "$ATT" pass - - 0 $PORT "kpf17"
run_pt kvmode-inj-abort_keeps_tail "$ATT" pass abort_keeps_tail - 0 $PORT "kpf171_abort_then_retry"
run_pt kvmode-mut-abort_keeps_tail+no_posmax_guard "$ATT" fail abort_keeps_tail,no_posmax_guard - 0 $PORT "kpf171_abort_then_retry"
# documented as undetectable by construction (no reachable state violates the guard): expected to PASS
run_pt kvmode-no_posmax_guard-alone "$ATT" pass no_posmax_guard - 0 $PORT "kpf171_abort_then_retry"
finish
