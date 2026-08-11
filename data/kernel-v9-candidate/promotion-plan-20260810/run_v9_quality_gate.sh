#!/bin/bash
set -euo pipefail

RESEARCH=/workspace/worktrees/v9-promotion-research
SELF=$(realpath "${BASH_SOURCE[0]}")
PLAN_ROOT="$RESEARCH/data/kernel-v9-candidate/promotion-plan-20260810"
PLAN="$PLAN_ROOT/b1-v9-plan/plan.json"
OUTPUT="$PLAN_ROOT/v9-quality-run"
BASELINE=/mnt/raid0/llm/epyc-inference-research/data/kernel-v8-candidate/quality-gate/run-20260725T204443Z-fullcontract-both-mode
QUESTIONS="$BASELINE/questions.json"
RUNNER="$RESEARCH/scripts/benchmark/v7_quality_gate_runner.py"
COMPARE="$RESEARCH/scripts/benchmark/v7_quality_gate_compare.py"
REGION_LOCK=/workspace/repos/epyc-orchestrator/scripts/region-lock
BINARY=/mnt/raid0/llm/llama.cpp-experimental/build-v9-cpu/bin/llama-server
EXPECTED_SHA256=0aadef69f2b75a1bf5a839a22ed88a5e2f895e5dc492acaae058354690ea9b05
EXPECTED_VERSION='version: 10125 (0db32c06e)'
active_pid=

mkdir -p "$OUTPUT"

[[ $(sha256sum -- "$BINARY" | awk '{print $1}') == "$EXPECTED_SHA256" ]]
[[ $("$BINARY" --version 2>&1 | sed -n '1p') == "$EXPECTED_VERSION" ]]
[[ $(sha256sum -- "$QUESTIONS" | awk '{print $1}') == 1532906b4a754673937027e73e2023d8eee7ed5d08f084c207a60ac81460adb1 ]]

cleanup_active() {
    local pid=${active_pid:-}
    [[ -n $pid ]] || return 0
    if ps -p "$pid" >/dev/null 2>&1; then
        kill -TERM "$pid"
        for _ in $(seq 1 60); do
            ps -p "$pid" >/dev/null 2>&1 || break
            sleep 1
        done
        if ps -p "$pid" >/dev/null 2>&1; then
            kill -KILL "$pid"
        fi
        wait "$pid" 2>/dev/null || true
    fi
    if ps -p "$pid" >/dev/null 2>&1; then
        return 1
    fi
    active_pid=
}
trap cleanup_active EXIT
trap 'cleanup_active; exit 130' INT
trap 'cleanup_active; exit 143' TERM

run_role() {
    local scenario=$1 label=$2 model_label=$3 concurrency=$4 baseline_result=$5
    local port report result rows log
    local -a server_argv

    port=$(jq -r --arg scenario "$scenario" \
        '.cells[] | select(.scenario == $scenario and .rep == 1) | .port' "$PLAN")
    mapfile -t server_argv < <(jq -r --arg scenario "$scenario" \
        '.cells[] | select(.scenario == $scenario and .rep == 1) | .server_argv[]' "$PLAN")
    [[ ${#server_argv[@]} -gt 0 && $port =~ ^[0-9]+$ ]]

    result="$OUTPUT/$label.json"
    rows="$OUTPUT/$label.per-question.jsonl"
    log="$OUTPUT/$label.server.log"
    report="$OUTPUT/$label-vs-v8.md"
    [[ ! -e "$result" && ! -e "$rows" && ! -e "$report" ]]

    "${server_argv[@]}" >"$log" 2>&1 &
    active_pid=$!

    python3 "$RUNNER" \
        --port "$port" \
        --output "$result" \
        --per-question-out "$rows" \
        --suites mmlu_pro gpqa \
        --n 200 \
        --seed 42 \
        --stratify \
        --max-tokens 64 \
        --endpoint chat \
        --kernel experimental-v9-dspark-promotion \
        --binary "$BINARY" \
        --models "$model_label" \
        --questions-in "$QUESTIONS" \
        --concurrency "$concurrency" \
        --arm "$label"

    python3 "$COMPARE" \
        --baseline "$baseline_result" \
        --candidate "$result" \
        --output "$report" \
        --regression-threshold 0.05 \
        --min-n 195

    cleanup_active
}

run_all() {
    run_role \
        v9_worker_general_cpu_native_mtp \
        v9-candidate-worker-general-full \
        'worker_general gemma q4 + drafter q8' \
        4 \
        "$BASELINE/v8-production-worker-general.json"
    run_role \
        v9_architect_critic_cpu_native_mtp \
        v9-candidate-architect-critic-full \
        'architect_critic Qwen3.5-122B-A10B-UD-Q4_K_M MTP q4/f16' \
        1 \
        "$BASELINE/v8-production-architect-general.json"
}

if [[ ${1:-} == --inner ]]; then
    run_all
else
    [[ $# -eq 0 ]]
    "$REGION_LOCK" run \
        --regions q0,q1,q2,q3 \
        --role bench \
        --tag v9-promotion-quality-20260810 \
        -- bash "$SELF" --inner
fi
