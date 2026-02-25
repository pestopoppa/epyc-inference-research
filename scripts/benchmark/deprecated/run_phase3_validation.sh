#!/bin/bash
set -euo pipefail

# =============================================================================
# Phase 3 Validation: Specialist Routing Benchmarking
# =============================================================================
#
# Runs the 6-step live validation pipeline to determine whether specialist
# routing improves orchestrator quality.
#
# Prerequisites:
#   - HOT tier backends running on ports 8080, 8081, 8082, 8083, 8090
#   - Orchestrator API on port 8000
#
# Usage:
#   bash scripts/benchmark/run_phase3_validation.sh            # Full pipeline
#   bash scripts/benchmark/run_phase3_validation.sh --step 2   # Resume from step 2
#   bash scripts/benchmark/run_phase3_validation.sh --dry-run  # Print commands only
# =============================================================================

PYTHON=/home/daniele/miniforge3/envs/pace-env/bin/python3
PROJECT_ROOT=/mnt/raid0/llm/claude
RESULTS_DIR="${PROJECT_ROOT}/benchmarks/results/orchestrator"
LOG_DIR="/mnt/raid0/llm/claude/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PHASE3_LOG="${LOG_DIR}/phase3_${TIMESTAMP}.log"

# Defaults
START_STEP=0
DRY_RUN=false
PIPELINE_SEED=""

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --step)
      START_STEP="$2"
      shift 2
      ;;
    --seed)
      PIPELINE_SEED="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h | --help)
      echo "Usage: $0 [--step N] [--seed N] [--dry-run]"
      echo "  --step N    Resume from step N (0-7)"
      echo "              0=health, 1=baseline, 2=seeding, 3=policy, 4=learning,"
      echo "              5=regression, 5b/6=plan-review, 7=kill-switch"
      echo "  --seed N    RNG seed (default: random, logged for reproducibility)"
      echo "  --dry-run   Print commands without executing"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

# Generate random seed if not specified, for question variety across runs
if [[ -z "$PIPELINE_SEED" ]]; then
  PIPELINE_SEED=$((RANDOM * RANDOM % 1000000))
fi

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

# =============================================================================
# Helpers
# =============================================================================

log() {
  echo "[$(date '+%H:%M:%S')] $*" | tee -a "${PHASE3_LOG}"
}

gate_fail() {
  log "GATE FAILED: $*"
  log "Pipeline halted. Check ${PHASE3_LOG} for details."
  exit 1
}

run_cmd() {
  if $DRY_RUN; then
    log "[DRY-RUN] $*"
  else
    log "Running: $*"
    "$@" 2>&1 | tee -a "${PHASE3_LOG}"
    return "${PIPESTATUS[0]}"
  fi
}

restart_api() {
  # Args: env vars to set (e.g. "SPECIALIST_ROUTING=1 MEMRL=1")
  local env_vars="${1:-}"
  log "Restarting orchestrator API with: ${env_vars:-default env}"

  if $DRY_RUN; then
    log "[DRY-RUN] Would restart API"
    return 0
  fi

  # Kill existing uvicorn
  local pids
  pids=$(lsof -t -i:8000 2>/dev/null || true)
  if [[ -n "$pids" ]]; then
    for pid in $pids; do
      kill -9 "$pid" 2>/dev/null || true
      log "  Killed PID $pid"
    done
    sleep 2
  fi

  # Launch with env vars
  local api_log="${LOG_DIR}/orchestrator_phase3.log"
  (cd "${PROJECT_ROOT}" &&
    env ${env_vars} \
      HF_HOME=/mnt/raid0/llm/cache/huggingface \
      TMPDIR=/mnt/raid0/llm/tmp \
      ${PYTHON} -m uvicorn src.api:app \
      --host 127.0.0.1 --port 8000 \
      >"${api_log}" 2>&1) &

  # Wait for health
  log "  Waiting for API health..."
  local deadline=$((SECONDS + 60))
  while [[ $SECONDS -lt $deadline ]]; do
    if curl -s --connect-timeout 2 http://localhost:8000/health | grep -q '"status":"ok"'; then
      log "  API ready"
      return 0
    fi
    sleep 1
  done

  log "  API failed to start. Check ${api_log}"
  return 1
}

check_port() {
  local port=$1
  curl -s --connect-timeout 3 "http://localhost:${port}/health" | grep -q '"status"' 2>/dev/null
}

# =============================================================================
# Step 0: Stack Health Check
# =============================================================================

step0_health_check() {
  log "=========================================="
  log "STEP 0: Stack Health Check"
  log "=========================================="

  local all_ok=true
  local health_results="/tmp/phase3_health_$$"
  mkdir -p "$health_results"
  for port in 8080 8081 8082 8083 8086 8087 8090 9001; do
    (if check_port "$port"; then
      echo "OK" >"${health_results}/${port}"
    else
      echo "DOWN" >"${health_results}/${port}"
    fi) &
  done
  wait
  for port in 8080 8081 8082 8083 8086 8087 8090 9001; do
    local status
    status=$(cat "${health_results}/${port}" 2>/dev/null || echo "DOWN")
    log "  Port ${port}: ${status}"
    if [[ "$status" != "OK" ]]; then
      all_ok=false
    fi
  done
  rm -rf "$health_results"

  if ! $all_ok; then
    gate_fail "Not all backends healthy. Start HOT tier first."
  fi

  # Check API
  if ! check_port 8000; then
    log "  API port 8000 not running, starting..."
    restart_api "ORCHESTRATOR_SPECIALIST_ROUTING=0 ORCHESTRATOR_MEMRL=0"
  fi

  # Smoke test
  log "  Smoke test: 2+2..."
  local smoke_result
  smoke_result=$(curl -s -X POST http://localhost:8000/chat \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"What is 2+2? Answer with just the number.","real_mode":true}' \
    --connect-timeout 10 --max-time 120 2>/dev/null)

  local answer
  answer=$(echo "$smoke_result" | ${PYTHON} -c "import sys,json; print(json.load(sys.stdin).get('answer','')[:100])" 2>/dev/null)
  local routed_to
  routed_to=$(echo "$smoke_result" | ${PYTHON} -c "import sys,json; print(json.load(sys.stdin).get('routed_to','unknown'))" 2>/dev/null)

  if [[ -z "$answer" ]]; then
    gate_fail "Smoke test returned empty answer"
  fi

  log "  Smoke test OK: routed_to=${routed_to}, answer=${answer:0:50}"
  log "STEP 0 PASSED"
}

# =============================================================================
# Step 1: Reproducible Baseline (routing OFF)
# =============================================================================

step1_baseline() {
  log "=========================================="
  log "STEP 1: Reproducible Baseline (seed=${PIPELINE_SEED})"
  log "=========================================="

  restart_api "ORCHESTRATOR_SPECIALIST_ROUTING=0 ORCHESTRATOR_MEMRL=0"

  local output="${RESULTS_DIR}/phase3_baseline_seed${PIPELINE_SEED}.json"
  run_cmd "${PYTHON} ${PROJECT_ROOT}/scripts/benchmark/compare_orchestrator_direct.py \
    --debug --suite all \
    --debug-seed ${PIPELINE_SEED} --debug-sample 10 \
    --output ${output}"

  if $DRY_RUN; then
    log "STEP 1 [DRY-RUN] DONE"
    return 0
  fi

  # Gate: verify output exists and has results
  if [[ ! -f "$output" ]]; then
    gate_fail "Step 1 output missing: ${output}"
  fi

  # Extract and log accuracy per suite
  log "Step 1 Results:"
  ${PYTHON} -c "
import json, sys
with open('${output}') as f:
    data = json.load(f)
results = data.get('results', [])
suites = {}
for r in results:
    s = r.get('suite', '?')
    if s not in suites:
        suites[s] = {'total': 0, 'pass': 0, 'routed': {}}
    suites[s]['total'] += 1
    if r.get('debug_score'):
        suites[s]['pass'] += 1
    rt = r.get('orchestrator_routed_to', 'unknown')
    suites[s]['routed'][rt] = suites[s]['routed'].get(rt, 0) + 1

print('Suite                  Pass/Total  Accuracy  Routing')
print('-' * 65)
for s in sorted(suites):
    st = suites[s]
    pct = st['pass']/st['total']*100 if st['total'] else 0
    routes = ', '.join(f'{k}:{v}' for k,v in st['routed'].items())
    print(f'{s:22s} {st[\"pass\"]:2d}/{st[\"total\"]:2d}     {pct:5.1f}%    {routes}')

total_pass = sum(st['pass'] for st in suites.values())
total = sum(st['total'] for st in suites.values())
overall = total_pass/total*100 if total else 0
print(f\"{'OVERALL':22s} {total_pass:2d}/{total:2d}     {overall:5.1f}%\")
" 2>&1 | tee -a "${PHASE3_LOG}"

  log "STEP 1 PASSED — Baseline saved to ${output}"
}

# =============================================================================
# Step 2: Comparative Specialist Seeding
# =============================================================================

step2_seeding() {
  log "=========================================="
  log "STEP 2: Comparative Specialist Seeding"
  log "=========================================="
  # NOTE: Roles sharing the same backend URL (e.g. frontdoor and coder_escalation
  # on localhost:8080) are automatically deduplicated by seed_specialist_routing.py.
  # Rewards for the deduplicated role are cloned from the canonical role.
  # Use --no-dedup to disable this behavior for debugging.

  restart_api "ORCHESTRATOR_SPECIALIST_ROUTING=1 ORCHESTRATOR_ARCHITECT_DELEGATION=1 ORCHESTRATOR_MEMRL=1"

  local output="${RESULTS_DIR}/seeding_live_seed${PIPELINE_SEED}.json"
  run_cmd "${PYTHON} ${PROJECT_ROOT}/scripts/benchmark/seed_specialist_routing.py \
    --suites thinking general math agentic coder instruction_precision vl \
    --roles frontdoor coder_escalation coder_escalation architect_general architect_coding worker_vision vision_escalation \
    --sample-size 10 --seed ${PIPELINE_SEED} \
    --output ${output}"

  if $DRY_RUN; then
    log "STEP 2 [DRY-RUN] DONE"
    return 0
  fi

  if [[ ! -f "$output" ]]; then
    gate_fail "Step 2 output missing: ${output}"
  fi

  # Gate: check if any specialist outperforms frontdoor
  local specialist_wins
  specialist_wins=$(${PYTHON} -c "
import json
with open('${output}') as f:
    data = json.load(f)
wins = data.get('specialist_wins', data.get('summary', {}).get('specialist_wins', 0))
print(wins)
" 2>/dev/null || echo "0")

  log "  Specialist wins: ${specialist_wins}"

  # Decision point: if frontdoor wins everything, stop
  # (We continue anyway but log the warning)
  if [[ "$specialist_wins" == "0" ]]; then
    log "  WARNING: Frontdoor wins every suite. Specialist routing may add no value."
    log "  Continuing pipeline to confirm..."
  fi

  log "STEP 2 PASSED — Seeding saved to ${output}"
}

# =============================================================================
# Step 3: Analyze Learned Policy
# =============================================================================

step3_analyze_policy() {
  log "=========================================="
  log "STEP 3: Analyze Learned Policy"
  log "=========================================="

  local output="${RESULTS_DIR}/phase3_policy_analysis.json"
  run_cmd "${PYTHON} ${PROJECT_ROOT}/scripts/benchmark/analyze_routing_policy.py \
    --min-samples 3 --format json \
    --output ${output}"

  if $DRY_RUN; then
    log "STEP 3 [DRY-RUN] DONE"
    return 0
  fi

  # Print the learned routing table
  ${PYTHON} -c "
import json
try:
    with open('${output}') as f:
        data = json.load(f)
    routing = data.get('routing_table', data.get('per_suite', {}))
    specialist_advantage = 0
    print('Suite                  Best Action            Q-value  Frontdoor Q  Delta')
    print('-' * 80)
    for suite, info in sorted(routing.items()):
        if isinstance(info, dict):
            best = info.get('best_action', info.get('best_role', '?'))
            q = info.get('best_q', info.get('q_value', 0))
            fq = info.get('frontdoor_q', 0)
            delta = q - fq
            if delta > 0.1 and best != 'frontdoor':
                specialist_advantage += 1
            print(f'{suite:22s} {best:22s} {q:7.3f}    {fq:7.3f}    {delta:+.3f}')
    print(f'\nSuites with specialist advantage (delta > 0.1): {specialist_advantage}')
except Exception as e:
    print(f'Policy analysis incomplete: {e}')
" 2>&1 | tee -a "${PHASE3_LOG}"

  log "STEP 3 PASSED — Policy saved to ${output}"
}

# =============================================================================
# Step 4: Learning Loop (5 iterations)
# =============================================================================

step4_learning_loop() {
  log "=========================================="
  log "STEP 4: Learning Loop (5 iterations)"
  log "=========================================="

  # API already running with SPECIALIST_ROUTING=1 ARCHITECT_DELEGATION=1 MEMRL=1
  # from step 2 — skip redundant restart (identical env vars)

  local output="${RESULTS_DIR}/memrl_phase3_loop.json"
  run_cmd "${PYTHON} ${PROJECT_ROOT}/scripts/benchmark/memrl_learning_loop.py \
    --iterations 5 --sample-size 10 --seed $((PIPELINE_SEED + 58)) \
    --suites thinking general math agentic coder instruction_precision vl \
    --regression-check \
    --output ${output}"

  local exit_code=$?

  if $DRY_RUN; then
    log "STEP 4 [DRY-RUN] DONE"
    return 0
  fi

  if [[ $exit_code -ne 0 ]]; then
    log "  WARNING: Learning loop exited with code ${exit_code}"
    log "  This may indicate a regression halt. Checking output..."
  fi

  if [[ -f "$output" ]]; then
    ${PYTHON} -c "
import json
with open('${output}') as f:
    data = json.load(f)
iters = data.get('iterations', [])
print(f'Iterations completed: {len(iters)}')
for i, it in enumerate(iters):
    acc = it.get('accuracy', it.get('debug_accuracy', '?'))
    print(f'  Iter {i+1}: accuracy={acc}')
" 2>&1 | tee -a "${PHASE3_LOG}"
  fi

  log "STEP 4 DONE (exit_code=${exit_code}) — Output: ${output}"
}

# =============================================================================
# Step 5: Final Regression Gate
# =============================================================================

step5_regression_gate() {
  log "=========================================="
  log "STEP 5: Final Regression Gate"
  log "=========================================="

  # API already running with SPECIALIST_ROUTING=1 ARCHITECT_DELEGATION=1 MEMRL=1
  # from step 2 — skip redundant restart (identical env vars)

  local output="${RESULTS_DIR}/phase3_specialist_seed${PIPELINE_SEED}.json"
  run_cmd "${PYTHON} ${PROJECT_ROOT}/scripts/benchmark/compare_orchestrator_direct.py \
    --debug --suite all --debug-seed ${PIPELINE_SEED} --debug-sample 10 \
    --regression-gate \
    --output ${output}"

  local exit_code=$?

  if $DRY_RUN; then
    log "STEP 5 [DRY-RUN] DONE"
    return 0
  fi

  if [[ $exit_code -ne 0 ]]; then
    log "  REGRESSION DETECTED (exit code ${exit_code})"
    log "  Specialist routing SHOULD NOT be shipped."
  fi

  # Compare with baseline
  local baseline="${RESULTS_DIR}/phase3_baseline_seed${PIPELINE_SEED}.json"
  if [[ -f "$output" ]] && [[ -f "$baseline" ]]; then
    ${PYTHON} -c "
import json

with open('${baseline}') as f:
    base = json.load(f)
with open('${output}') as f:
    spec = json.load(f)

def suite_accuracy(data):
    suites = {}
    for r in data.get('results', []):
        s = r.get('suite', '?')
        if s not in suites:
            suites[s] = {'total': 0, 'pass': 0}
        suites[s]['total'] += 1
        if r.get('debug_score'):
            suites[s]['pass'] += 1
    return suites

base_suites = suite_accuracy(base)
spec_suites = suite_accuracy(spec)

# Count specialist routing
spec_routing = {}
for r in spec.get('results', []):
    rt = r.get('orchestrator_routed_to', 'unknown')
    spec_routing[rt] = spec_routing.get(rt, 0) + 1
specialist_pct = sum(v for k,v in spec_routing.items() if k != 'frontdoor') / max(1, sum(spec_routing.values())) * 100

print('A/B COMPARISON: Baseline vs Specialist')
print('=' * 70)
print(f'{\"Suite\":22s} {\"Baseline\":>10s} {\"Specialist\":>12s} {\"Delta\":>8s}')
print('-' * 70)

total_base = total_spec = 0
for s in sorted(set(list(base_suites.keys()) + list(spec_suites.keys()))):
    bp = base_suites.get(s, {}).get('pass', 0)
    bt = base_suites.get(s, {}).get('total', 0)
    sp = spec_suites.get(s, {}).get('pass', 0)
    st = spec_suites.get(s, {}).get('total', 0)
    ba = bp/bt*100 if bt else 0
    sa = sp/st*100 if st else 0
    delta = sa - ba
    total_base += bp; total_spec += sp
    marker = ' <<' if delta < -5 else (' >>' if delta > 5 else '')
    print(f'{s:22s} {bp:2d}/{bt:2d} ({ba:5.1f}%) {sp:2d}/{st:2d} ({sa:5.1f}%) {delta:+6.1f}%{marker}')

bt = sum(s['total'] for s in base_suites.values())
st = sum(s['total'] for s in spec_suites.values())
ba = total_base/bt*100 if bt else 0
sa = total_spec/st*100 if st else 0
print(f'{\"OVERALL\":22s} {total_base:2d}/{bt:2d} ({ba:5.1f}%) {total_spec:2d}/{st:2d} ({sa:5.1f}%) {sa-ba:+6.1f}%')
print(f'\nSpecialist routing used: {specialist_pct:.0f}%')
print(f'Routing distribution: {spec_routing}')

# Decision matrix
delta = sa - ba
if delta >= 5 and specialist_pct >= 30:
    print('\nDECISION: SHIP — Flip specialist_routing default to True in features.py')
elif delta >= 0 and specialist_pct >= 30:
    print('\nDECISION: CONDITIONAL — Enable with monitoring')
elif delta >= 0 and specialist_pct < 30:
    print('\nDECISION: MARGINAL — Specialist routing rarely activates')
else:
    print('\nDECISION: KEEP OFF — Quality regression detected')
" 2>&1 | tee -a "${PHASE3_LOG}"
  fi

  log "STEP 5 DONE (exit_code=${exit_code}) — Output: ${output}"
}

# =============================================================================
# Step 5b: Plan Review A/B Test (architect-in-the-loop)
# =============================================================================

step5b_plan_review() {
  log "=========================================="
  log "STEP 5b: Plan Review A/B Test"
  log "=========================================="

  # Same seed suite but with PLAN_REVIEW enabled alongside SPECIALIST_ROUTING.
  # Compares: accuracy delta vs step 5, correction rate, convergence signal.
  # Hot-toggle plan_review via /config instead of restarting the API.
  log "  Enabling plan_review via /config..."
  if ! $DRY_RUN; then
    curl -sf -X POST http://localhost:8000/config \
      -H 'Content-Type: application/json' \
      -d '{"plan_review": true}' >/dev/null
  else
    log "[DRY-RUN] Would POST /config {plan_review: true}"
  fi

  local output="${RESULTS_DIR}/phase3_plan_review_seed${PIPELINE_SEED}.json"
  run_cmd "${PYTHON} ${PROJECT_ROOT}/scripts/benchmark/compare_orchestrator_direct.py \
    --debug --suite all --debug-seed ${PIPELINE_SEED} --debug-sample 10 \
    --regression-gate \
    --output ${output}"

  local exit_code=$?

  if $DRY_RUN; then
    log "STEP 5b [DRY-RUN] DONE"
    return 0
  fi

  if [[ $exit_code -ne 0 ]]; then
    log "  Plan review regression detected (exit code ${exit_code})"
    log "  Plan review may add latency without quality gain."
  fi

  # Compare with step 5 (specialist-only) and step 1 (baseline)
  local step5_output="${RESULTS_DIR}/phase3_specialist_seed${PIPELINE_SEED}.json"
  local baseline="${RESULTS_DIR}/phase3_baseline_seed${PIPELINE_SEED}.json"
  if [[ -f "$output" ]] && [[ -f "$baseline" ]]; then
    ${PYTHON} -c "
import json

with open('${baseline}') as f:
    base = json.load(f)
with open('${output}') as f:
    pr = json.load(f)

# Load step 5 if available
step5 = None
try:
    with open('${step5_output}') as f:
        step5 = json.load(f)
except FileNotFoundError:
    pass

def count_pass(data):
    return sum(1 for r in data.get('results', []) if r.get('debug_score'))

def count_total(data):
    return len(data.get('results', []))

base_pass = count_pass(base)
pr_pass = count_pass(pr)
base_total = count_total(base)
pr_total = count_total(pr)

print('PLAN REVIEW A/B COMPARISON')
print('=' * 60)
print(f'Baseline (no routing):    {base_pass}/{base_total} ({base_pass/max(1,base_total)*100:.1f}%)')

if step5:
    s5_pass = count_pass(step5)
    s5_total = count_total(step5)
    print(f'Specialist only (step 5): {s5_pass}/{s5_total} ({s5_pass/max(1,s5_total)*100:.1f}%)')
    delta_vs_s5 = pr_pass - s5_pass
    print(f'Plan review (step 5b):    {pr_pass}/{pr_total} ({pr_pass/max(1,pr_total)*100:.1f}%)  delta vs step5: {delta_vs_s5:+d}')
else:
    print(f'Plan review (step 5b):    {pr_pass}/{pr_total} ({pr_pass/max(1,pr_total)*100:.1f}%)')

delta_vs_base = pr_pass - base_pass
print(f'\nDelta vs baseline: {delta_vs_base:+d}')

# Check progress logs for plan review stats
try:
    import glob, os
    log_dir = '${PROJECT_ROOT}/logs/progress'
    today = __import__('datetime').date.today().isoformat()
    log_file = os.path.join(log_dir, f'{today}.jsonl')
    reviewed = 0
    corrected = 0
    if os.path.exists(log_file):
        with open(log_file) as f:
            for line in f:
                if 'plan_reviewed' in line:
                    entry = json.loads(line)
                    reviewed += 1
                    if entry.get('outcome') == 'corrected':
                        corrected += 1
    if reviewed > 0:
        print(f'\nPlan reviews: {reviewed} total, {corrected} corrected ({corrected/reviewed*100:.0f}% correction rate)')
    else:
        print('\nNo plan_reviewed events found in progress logs')
except Exception as e:
    print(f'\nCould not read plan review stats: {e}')

# Decision
if step5:
    if delta_vs_s5 > 0:
        print('\nDECISION: PLAN REVIEW HELPS — Enable plan_review alongside specialist_routing')
    elif delta_vs_s5 == 0:
        print('\nDECISION: NEUTRAL — Plan review adds latency without accuracy change. Check correction rate for MemRL value.')
    else:
        print('\nDECISION: PLAN REVIEW HURTS — Keep plan_review OFF')
else:
    if delta_vs_base > 0:
        print('\nDECISION: PLAN REVIEW HELPS vs baseline — Enable plan_review')
    else:
        print('\nDECISION: INCONCLUSIVE — No step 5 comparison available')
" 2>&1 | tee -a "${PHASE3_LOG}"
  fi

  log "STEP 5b DONE (exit_code=${exit_code}) — Output: ${output}"
}

# =============================================================================
# Step 6: Kill Switch Test
# =============================================================================

step6_kill_switch() {
  log "=========================================="
  log "STEP 6: Kill Switch Test"
  log "=========================================="

  # Hot-toggle routing off via /config instead of restarting the API.
  log "  Disabling specialist_routing, architect_delegation, plan_review via /config..."
  if ! $DRY_RUN; then
    curl -sf -X POST http://localhost:8000/config \
      -H 'Content-Type: application/json' \
      -d '{"specialist_routing": false, "architect_delegation": false, "plan_review": false}' >/dev/null
  else
    log "[DRY-RUN] Would POST /config {specialist_routing: false, architect_delegation: false, plan_review: false}"
  fi

  local output="${RESULTS_DIR}/phase3_killswitch_seed${PIPELINE_SEED}.json"
  # Kill-switch only checks routing returns to frontdoor (binary property).
  # 5 per suite (35 total) is sufficient.
  run_cmd "${PYTHON} ${PROJECT_ROOT}/scripts/benchmark/compare_orchestrator_direct.py \
    --debug --suite all \
    --debug-seed ${PIPELINE_SEED} --debug-sample 5 \
    --output ${output}"

  if $DRY_RUN; then
    log "STEP 6 [DRY-RUN] DONE"
    return 0
  fi

  if [[ ! -f "$output" ]]; then
    gate_fail "Step 6 output missing: ${output}"
  fi

  # Verify all routing = frontdoor
  local non_frontdoor
  non_frontdoor=$(${PYTHON} -c "
import json
with open('${output}') as f:
    data = json.load(f)
non_fd = sum(1 for r in data.get('results', []) if r.get('orchestrator_routed_to', '') != 'frontdoor')
print(non_fd)
" 2>/dev/null || echo "-1")

  if [[ "$non_frontdoor" != "0" ]]; then
    log "  WARNING: ${non_frontdoor} questions NOT routed to frontdoor with routing OFF"
  else
    log "  Kill switch verified: all questions routed to frontdoor"
  fi

  # Compare with Step 1 baseline
  local baseline="${RESULTS_DIR}/phase3_baseline_seed${PIPELINE_SEED}.json"
  if [[ -f "$output" ]] && [[ -f "$baseline" ]]; then
    ${PYTHON} -c "
import json
with open('${baseline}') as f:
    base = json.load(f)
with open('${output}') as f:
    kill = json.load(f)
base_pass = sum(1 for r in base.get('results',[]) if r.get('debug_score'))
kill_pass = sum(1 for r in kill.get('results',[]) if r.get('debug_score'))
base_total = len(base.get('results',[]))
kill_total = len(kill.get('results',[]))
print(f'Baseline: {base_pass}/{base_total}  Kill switch: {kill_pass}/{kill_total}  Delta: {kill_pass - base_pass}')
delta = abs(kill_pass - base_pass)
if delta <= 2:
    print('PASS: Kill switch returns to baseline (within noise)')
else:
    print(f'WARNING: {delta} question delta — investigate')
" 2>&1 | tee -a "${PHASE3_LOG}"
  fi

  log "STEP 6 PASSED — Output: ${output}"
}

# =============================================================================
# Main Pipeline
# =============================================================================

log "=========================================="
log "PHASE 3 VALIDATION PIPELINE"
log "=========================================="
log "Start step: ${START_STEP}"
log "Seed: ${PIPELINE_SEED}"
log "Dry run: ${DRY_RUN}"
log "Timestamp: ${TIMESTAMP}"
log "Log: ${PHASE3_LOG}"
log "=========================================="

pipeline_start=$SECONDS

[[ $START_STEP -le 0 ]] && step0_health_check
[[ $START_STEP -le 1 ]] && step1_baseline
[[ $START_STEP -le 2 ]] && step2_seeding
[[ $START_STEP -le 3 ]] && step3_analyze_policy
[[ $START_STEP -le 4 ]] && step4_learning_loop
[[ $START_STEP -le 5 ]] && step5_regression_gate
[[ $START_STEP -le 6 ]] && step5b_plan_review
[[ $START_STEP -le 7 ]] && step6_kill_switch

elapsed=$((SECONDS - pipeline_start))
log "=========================================="
log "PHASE 3 PIPELINE COMPLETE"
log "Total elapsed: $((elapsed / 3600))h $(((elapsed % 3600) / 60))m $((elapsed % 60))s"
log "Results: ${RESULTS_DIR}/phase3_*"
log "Full log: ${PHASE3_LOG}"
log "=========================================="
