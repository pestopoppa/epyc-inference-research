#!/bin/bash
#maxjob_s=5400
# KPF-17 vacuity-audit re-check (assert_forked, no stale /slots waits): every changed case honest + its mutant on tiny hybrid, gemma SWA and (kv cases) Qwen2.5-0.5B. Python-only change: build.ok (defc5a094fcd) still valid; needs 10-kpf17-build
set -euo pipefail
JOB=26-kpf17-vacuity-recheck
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
grep -qF "def assert_forked" "$WT/tools/server/tests/unit/test_slot_fork.py" || { echo "DEP_MISSING: test_slot_fork.py lacks the vacuity-audit commit"; echo "RESULT $JOB pass=0 fail=0 DEP_MISSING"; exit 3; }
PORT=18985
K_BUSY="test_fork_from_busy_source"
K_P2="test_p2_epoch_hash_ladder_and_ids"
K_ABORT="kpf171_abort_then_retry"
K_KEY="kpf171_key_check_negative_control"
K_SAME="kpf172_fork_same_iteration_as_source_erase"
K_STEP="kpf172_fork_from_checkpoint_of_current_step"
K_174="kpf174_swa_pinned_junction_fork_equals_fresh"
# --- honest: every changed case must PASS (and now provably fork)
for m in HYB SWA; do
  model=${!m}
  run_pt "honest-busy-$m"  "$model" pass - - 0 $PORT "$K_BUSY"
  run_pt "honest-p2-$m"    "$model" pass - - 0 $PORT "$K_P2"
  run_pt "honest-abort-$m" "$model" pass - - 0 $PORT "$K_ABORT"
  run_pt "honest-key-$m"   "$model" pass - - 0 $PORT "$K_KEY"
  run_pt "honest-same-$m"  "$model" pass - - 0 $PORT "$K_SAME"
  run_pt "honest-step-$m"  "$model" pass - - 0 $PORT "$K_STEP"
done
run_pt honest-174-SWA   "$SWA" pass - - 0 $PORT "$K_174"
run_pt honest-busy-ATT  "$ATT" pass - - 0 $PORT "$K_BUSY"
run_pt honest-p2-ATT    "$ATT" pass - - 0 $PORT "$K_P2"
run_pt honest-abort-ATT "$ATT" pass - - 0 $PORT "$K_ABORT"
# --- mutants: must FAIL
for m in HYB SWA; do
  model=${!m}
  run_pt "mut-busy-fork_skip_ckpt_load-$m"  "$model" fail fork_skip_ckpt_load - 0 $PORT "$K_BUSY"
  run_pt "mut-key-fork_no_lcp-$m"           "$model" fail fork_no_lcp - 0 $PORT "$K_KEY"
  run_pt "mut-same-fork_skip_ckpt_load-$m"  "$model" fail fork_skip_ckpt_load - 0 $PORT "$K_SAME"
  run_pt "mut-step-skew+no_posmax_guard-$m" "$model" fail ckpt_ntokens_skew,no_posmax_guard - 0 $PORT "$K_STEP"
done
run_pt mut-174-fork_skip_ckpt_load-SWA "$SWA" fail fork_skip_ckpt_load - 0 $PORT "$K_174"
run_pt mut-174-ckpt_ntokens_skew-SWA   "$SWA" fail ckpt_ntokens_skew - 0 $PORT "$K_174"
# kv mode abort: the injection alone passes (guard skips the source), injection + guard removed fails
run_pt inj-abort-abort_keeps_tail-ATT                 "$ATT" pass abort_keeps_tail - 0 $PORT "$K_ABORT"
run_pt mut-abort-abort_keeps_tail+no_posmax_guard-ATT "$ATT" fail abort_keeps_tail,no_posmax_guard - 0 $PORT "$K_ABORT"
# evidence: the fork lines and per-round prints of the timing-dependent cases
for f in "$OUT"/honest-*.log; do
  { echo "== $(basename "$f")"; grep -ahE 'do_slot_fork|^delay |^abort@|^source vs unforked|greedy flip' "$f" | cut -c1-200 | head -12; } >> "$OUT/evidence.txt" || true
done
finish
