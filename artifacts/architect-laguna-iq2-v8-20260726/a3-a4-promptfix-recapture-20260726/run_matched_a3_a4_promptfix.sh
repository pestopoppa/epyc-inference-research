#!/usr/bin/env bash
# Raw capture only. A3 then A4, never concurrent, after the 27B continuation.
set -euo pipefail

REPO=/mnt/raid0/llm/epyc-inference-research
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONTINUATION_ROOT="$REPO/artifacts/architect-27b-finetunes-v8-20260726/live-20260726T1750Z/continuation-27b-v8"
LAGUNA_RUN="$REPO/artifacts/architect-laguna-iq2-v8-20260726/scorer-artifact-rescore-20260726/clean-full40-promptfix-20260726/run-20260726T220759Z"
RUNNER_SOURCE="$LAGUNA_RUN/runner_source.py"
WATCHDOG_SOURCE="$LAGUNA_RUN/capture_integrity_watchdog.py"
SERVER=/mnt/raid0/llm/llama.cpp/build-hip/bin/llama-server
A3_MODEL=/mnt/raid0/llm/models/Qwen3.6-27B-MTP-Q8_0.gguf
A4_MODEL=/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf
PORT=18093
CORES=184-191
QUESTION_SHA=4b03ad7703bbf2dbaa1eb91b3313cc3cab2892672db87f6242ffd1d489e76375
RUNNER_SHA=79721927e95293d070aba294bf422a24b1182dde07310d461d9e3ddaf6c84b0e
WATCHDOG_SHA=f4bd45b9617ca880a92be506d741038df65d457f0923f07bc3db7091a7303055
SERVER_SHA=112c560f1c978c584a9899539851348a0ce1e05cde458061c281758aff066882
A3_MODEL_SHA=9408dcb356cc061a05c139e5647cbde0698ff980c6a69f7fc214e9989f86cfa8
A4_MODEL_SHA=93dd505d5b4d3f6adcef8c3b6b35465f7537379893f80b87b9ddc2baa62ca557
INITIAL_STATUS_TIMEOUT_S=300
SERVER_PID=""

die() { printf 'matched-a3-a4: %s\n' "$*" >&2; return 1; }
port_busy() { lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; }
wait_exit() { local pid=$1 deadline=$((SECONDS + 30)); while kill -0 "$pid" 2>/dev/null; do (( SECONDS < deadline )) || return 1; sleep 1; done; }
stop_server() { [[ -n ${SERVER_PID:-} ]] && kill -0 "$SERVER_PID" 2>/dev/null || return 0; kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true; wait_exit "$SERVER_PID" || { kill -KILL -- "-$SERVER_PID" 2>/dev/null || kill -KILL "$SERVER_PID" 2>/dev/null || true; wait_exit "$SERVER_PID"; }; ! ps -p "$SERVER_PID" >/dev/null 2>&1 || die "owned server survived cleanup"; SERVER_PID=""; }
cleanup() { local rc=$?; set +e; stop_server; if [[ -n ${RUN_DIR:-} ]]; then { printf 'exit_rc=%s\n' "$rc"; lsof -nP -iTCP:"$PORT" -sTCP:LISTEN 2>&1 || true; lsof /dev/kfd 2>&1 || true; } >"$RUN_DIR/cleanup_proof.txt"; fi; exit "$rc"; }
wait_health() { local dir=$1 deadline=$((SECONDS + 180)); while (( SECONDS < deadline )); do curl -fsS --max-time 5 "http://127.0.0.1:$PORT/health" >"$dir/health.json" 2>/dev/null && return; kill -0 "$SERVER_PID" 2>/dev/null || die "server exited before health"; sleep 2; done; die "health timeout"; }
wait_for_live_status() {
    local status_path=$1 runner_pid=$2 deadline=$((SECONDS + INITIAL_STATUS_TIMEOUT_S))
    while (( SECONDS < deadline )); do
        if [[ -s $status_path ]] && jq -e 'type == "object"' "$status_path" >/dev/null 2>&1; then
            return 0
        fi
        kill -0 "$runner_pid" 2>/dev/null || die "runner exited before publishing live status"
        sleep 2
    done
    die "runner did not publish live status within ${INITIAL_STATUS_TIMEOUT_S}s"
}
require_clean_gpu() { local pids; pids=$(lsof -t /dev/kfd 2>/dev/null || true); [[ -z $pids ]] || die "GPU/KFD is owned by PID(s) $pids"; }
verify_package_static() { python3 "$HERE/prepare_matched_questions.py" >/dev/null; [[ $(sha256sum "$RUNNER_SOURCE" | awk '{print $1}') == "$RUNNER_SHA" ]] || die "v4 runner SHA mismatch"; [[ $(sha256sum "$WATCHDOG_SOURCE" | awk '{print $1}') == "$WATCHDOG_SHA" ]] || die "watchdog SHA mismatch"; [[ $(sha256sum "$HERE/questions_pinned_40.json" | awk '{print $1}') == "$QUESTION_SHA" ]] || die "question SHA mismatch"; [[ -x $SERVER ]] || die "frozen v8 server is unavailable"; [[ -s $A3_MODEL ]] || die "A3 MTP model is unavailable"; [[ -s $A4_MODEL ]] || die "A4 MTP model is unavailable"; }
verify_execution_hashes() { [[ $(sha256sum "$SERVER" | awk '{print $1}') == "$SERVER_SHA" ]] || die "frozen v8 HIP binary SHA mismatch"; [[ $(sha256sum "$A3_MODEL" | awk '{print $1}') == "$A3_MODEL_SHA" ]] || die "A3 MTP model SHA mismatch"; [[ $(sha256sum "$A4_MODEL" | awk '{print $1}') == "$A4_MODEL_SHA" ]] || die "A4 MTP model SHA mismatch"; }
preflight() { python3 "$HERE/validate_27b_continuation.py" "$CONTINUATION_ROOT" >/dev/null; verify_package_static; verify_execution_hashes; port_busy && die "port $PORT is occupied"; require_clean_gpu; }
shell_lifecycle_self_test() {
    sleep 60 & local fake_runner=$!
    (exit 7) & local fake_watchdog=$!
    local rc
    set +e; await_runner_and_watchdog "$fake_runner" "$fake_watchdog"; rc=$?; set -e
    (( rc == 7 )) || die "self-test expected watchdog failure 7, got $rc"
    ! ps -p "$fake_runner" >/dev/null 2>&1 || die "self-test runner survived watchdog failure"
    (exit 0) & fake_runner=$!
    (sleep 0.1; exit 6) & fake_watchdog=$!
    set +e; await_runner_and_watchdog "$fake_runner" "$fake_watchdog"; rc=$?; set -e
    (( rc == 6 )) || die "self-test expected late watchdog failure 6, got $rc"
}
self_test() { verify_package_static; [[ $(jq length "$HERE/expected_question_ids.json") == 40 ]]; [[ $(jq -r '.sampling.max_tokens' "$HERE/prepared_manifest.json") == 3072 ]]; shell_lifecycle_self_test; printf 'matched-a3-a4 self-test: PASS (no inference)\n'; }
await_runner_and_watchdog() {
    local runner=$1 watchdog=$2 runner_done=false watchdog_done=false runner_rc=0 watchdog_rc=0
    while [[ $runner_done == false || $watchdog_done == false ]]; do
        if [[ $watchdog_done == false ]] && ! kill -0 "$watchdog" 2>/dev/null; then
            if wait "$watchdog"; then watchdog_rc=0; else watchdog_rc=$?; fi
            watchdog_done=true
            (( watchdog_rc == 0 )) || { [[ $runner_done == true ]] || kill -TERM "$runner" 2>/dev/null || true; }
        fi
        if [[ $runner_done == false ]] && ! kill -0 "$runner" 2>/dev/null; then
            if wait "$runner"; then runner_rc=0; else runner_rc=$?; fi
            runner_done=true
            (( runner_rc == 0 )) || { [[ $watchdog_done == true ]] || kill -TERM "$watchdog" 2>/dev/null || true; }
        fi
        [[ $runner_done == true && $watchdog_done == true ]] || sleep 1
    done
    (( watchdog_rc == 0 )) || return "$watchdog_rc"
    return "$runner_rc"
}
run_arm() {
    local label=$1 model=$2 arm=$3
    RUN_DIR="$HERE/run-$(date -u +%Y%m%dT%H%M%SZ)-$label"; mkdir -p "$RUN_DIR"
    cp "$HERE/questions_pinned_40.json" "$RUN_DIR/questions_pinned_40.json"; cp "$HERE/expected_question_ids.json" "$RUN_DIR/expected_question_ids.json"; cp "$RUNNER_SOURCE" "$RUN_DIR/runner_source.py"; cp "$WATCHDOG_SOURCE" "$RUN_DIR/capture_integrity_watchdog.py"
    python3 - "$RUN_DIR/provenance.json" "$label" "$arm" "$model" "$SERVER" "$QUESTION_SHA" "$RUNNER_SHA" "$WATCHDOG_SHA" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
out, label, arm, model, server, question_sha, runner_sha, watchdog_sha = sys.argv[1:]
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''): h.update(block)
    return h.hexdigest()
Path(out).write_text(json.dumps({'schema':'a3_a4_matched_promptfix_capture.v1','timestamp_utc':datetime.now(timezone.utc).isoformat(),'label':label,'arm':arm,'kernel':'production-consolidated-v8','kernel_head':'67a433bf45a8a091d83b4ea0b32ff0735fd51800','model':model,'model_sha256':sha(model),'binary':server,'binary_sha256':sha(server),'question_sha256':question_sha,'runner_source_sha256':runner_sha,'watchdog_source_sha256':watchdog_sha,'capture_only':True}, indent=2, sort_keys=True)+'\n')
PY
    local -a server=("$SERVER" -m "$model" --host 127.0.0.1 --port "$PORT" --metrics --slots --jinja --reasoning off --device ROCm0 -ngl all -fa on -np 1 -c 49152 -t 8 -tb 8 -b 2048 -ub 2048 -ctk f16 -ctv f16 --spec-type draft-mtp --spec-draft-n-max 4)
    printf '%q ' env GGML_IQK=1 LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp/build-hip/bin taskset -c "$CORES" "${server[@]}" >"$RUN_DIR/server.argv"; printf '\n' >>"$RUN_DIR/server.argv"
    setsid env GGML_IQK=1 LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp/build-hip/bin taskset -c "$CORES" "${server[@]}" >"$RUN_DIR/server.stdout" 2>"$RUN_DIR/server.stderr" & SERVER_PID=$!
    { printf 'server_pid=%s\n' "$SERVER_PID"; ps -p "$SERVER_PID" -o pid,ppid,lstart,args; } >"$RUN_DIR/server_process_start.txt"
    wait_health "$RUN_DIR"
    local -a runner=("$REPO/.venv/bin/python3" -B "$RUN_DIR/runner_source.py" --host 127.0.0.1 --port "$PORT" --suites swebench_oracle --n 40 --limit 40 --seed 42 --max-tokens 3072 --repeats 1 --concurrency 1 --temperature 0.6 --top-p 0.95 --top-k 20 --no-enable-thinking --endpoint chat --kernel production-consolidated-v8 --arm "$arm" --binary "$SERVER" --models "$model" --questions-in "$RUN_DIR/questions_pinned_40.json" --per-question-out "$RUN_DIR/pq.jsonl" --output "$RUN_DIR/runner.json")
    printf '%q ' env PYTHONPATH="$REPO/scripts/benchmark" RUNNER_REQUEST_TIMEOUT_S=3600 taskset -c "$CORES" "${runner[@]}" >"$RUN_DIR/runner.argv"; printf '\n' >>"$RUN_DIR/runner.argv"
    env PYTHONPATH="$REPO/scripts/benchmark" RUNNER_REQUEST_TIMEOUT_S=3600 taskset -c "$CORES" "${runner[@]}" >"$RUN_DIR/runner.stdout" 2>"$RUN_DIR/runner.stderr" & local runner_pid=$!
    if ! wait_for_live_status "$RUN_DIR/pq.live-status.json" "$runner_pid"; then
        kill -TERM "$runner_pid" 2>/dev/null || true
        wait "$runner_pid" 2>/dev/null || true
        die "$label initial live-status gate failed"
    fi
    "$REPO/.venv/bin/python3" -B "$RUN_DIR/capture_integrity_watchdog.py" --watch --poll-interval-s 5 --startup-grace-s "$INITIAL_STATUS_TIMEOUT_S" --stale-timeout-s 3600 --request-error-threshold 1 "$RUN_DIR/pq.live-status.json" >"$RUN_DIR/watchdog.stdout" 2>"$RUN_DIR/watchdog.stderr" & local watchdog_pid=$!
    { printf 'runner_pid=%s\nwatchdog_pid=%s\n' "$runner_pid" "$watchdog_pid"; ps -p "$runner_pid,$watchdog_pid" -o pid,ppid,lstart,args; } >"$RUN_DIR/runner_watchdog_process_start.txt"
    await_runner_and_watchdog "$runner_pid" "$watchdog_pid" || die "$label runner or watchdog failed"
    python3 "$HERE/validate_matched_capture.py" "$RUN_DIR" "$arm" >"$RUN_DIR/capture.validation.json"
    stop_server
    RUN_DIR=""
}
case ${1:-} in
    --self-test) [[ $# -eq 1 ]] || die 'usage: --self-test|--preflight|--execute'; self_test ;;
    --preflight) [[ $# -eq 1 ]] || die 'usage: --self-test|--preflight|--execute'; preflight; printf 'matched-a3-a4 preflight: READY (no inference)\n' ;;
    --execute) [[ $# -eq 1 ]] || die 'usage: --self-test|--preflight|--execute'; preflight; trap cleanup EXIT INT TERM; run_arm A3_27B_dense "$A3_MODEL" A3_27B_dense_v8_matched_laguna_promptfix_3072; run_arm A4_35B_A3B "$A4_MODEL" A4_35B_A3B_v8_matched_laguna_promptfix_3072; date -u +%Y-%m-%dT%H:%M:%SZ >"$HERE/a3_a4_matched_promptfix.complete"; trap - EXIT ;;
    *) die 'usage: --self-test|--preflight|--execute' ;;
esac
