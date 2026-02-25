#!/usr/bin/env bash
set -u
set -o pipefail

# Run perf-parallel-tools validation tasks consecutively.
#
# Covers:
# 1) WS2 concurrent inference sweep (dry-run + live)
# 2) WS3A id_slot/prefix bypass ON/OFF probes via /chat
# 3) WS3B escalation compression A/B (seed_specialist_routing)
# 4) WS3C prewarm evidence via orchestrator logs + role_history probes
#
# Usage:
#   bash scripts/benchmark/run_perf_parallel_tools_validation.sh
#   bash scripts/benchmark/run_perf_parallel_tools_validation.sh --include-architects
#   bash scripts/benchmark/run_perf_parallel_tools_validation.sh --fail-fast
#   bash scripts/benchmark/run_perf_parallel_tools_validation.sh --resume-from ws3b --ws3b-start-seed 902

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 1

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="benchmarks/results/eval/perf_parallel_tools_validation_${TIMESTAMP}"
API_URL="http://127.0.0.1:8000"
INCLUDE_ARCHITECTS=0
FAIL_FAST=0
RESUME_FROM="ws2" # ws2|ws3a|ws3b|ws3c

# Sweep defaults
SWEEP_N_WARMUP=2
SWEEP_N_MEASURED=5
SWEEP_N_PREDICT=128

# Compression A/B defaults
AB_SUITE="coder"
AB_TIMEOUT=120
AB_SEEDS=(901 902 903)
WS3B_CMD_TIMEOUT_S=420
WS3B_PREFLIGHT_ONCE=1
WS3B_NO_PREFLIGHT=0
WS3B_START_SEED=""
WS3B_MODE="both" # off|on|both

# /chat probes
CHAT_TIMEOUT_S=150

mkdir -p "$OUT_DIR"
MASTER_LOG="$OUT_DIR/run.log"
SUMMARY_JSON="$OUT_DIR/summary.json"
SUMMARY_MD="$OUT_DIR/SUMMARY.md"

FAILED_STEPS=()

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$MASTER_LOG"
}

usage() {
  cat <<USAGE
Usage: $0 [options]

Options:
  --out-dir PATH           Output directory (default: $OUT_DIR)
  --api-url URL            API URL (default: $API_URL)
  --include-architects     Include architect_general in WS2 sweep live run
  --fail-fast              Stop on first failed step (default: continue)
  --resume-from STAGE      Resume from stage: ws2|ws3a|ws3b|ws3c (default: $RESUME_FROM)
  --ws3b-timeout-sec N     Wrapper timeout for each ws3b seed run (default: $WS3B_CMD_TIMEOUT_S)
  --ws3b-start-seed N      Skip ws3b seeds before N (default: none)
  --ws3b-mode MODE         Run ws3b mode: off|on|both (default: $WS3B_MODE)
  --ws3b-preflight-each    Run ws3b with --preflight for every seed (default: first seed only)
  --ws3b-no-preflight      Do not pass --preflight to ws3b seed runs
  -h, --help               Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir)
      OUT_DIR="$2"
      MASTER_LOG="$OUT_DIR/run.log"
      SUMMARY_JSON="$OUT_DIR/summary.json"
      SUMMARY_MD="$OUT_DIR/SUMMARY.md"
      mkdir -p "$OUT_DIR"
      shift 2
      ;;
    --api-url)
      API_URL="$2"
      shift 2
      ;;
    --include-architects)
      INCLUDE_ARCHITECTS=1
      shift
      ;;
    --fail-fast)
      FAIL_FAST=1
      shift
      ;;
    --resume-from)
      RESUME_FROM="$2"
      shift 2
      ;;
    --ws3b-timeout-sec)
      WS3B_CMD_TIMEOUT_S="$2"
      shift 2
      ;;
    --ws3b-start-seed)
      WS3B_START_SEED="$2"
      shift 2
      ;;
    --ws3b-mode)
      WS3B_MODE="$2"
      shift 2
      ;;
    --ws3b-preflight-each)
      WS3B_PREFLIGHT_ONCE=0
      shift
      ;;
    --ws3b-no-preflight)
      WS3B_NO_PREFLIGHT=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 2
      ;;
  esac
done

stage_rank() {
  case "$1" in
    ws2) echo 1 ;;
    ws3a) echo 2 ;;
    ws3b) echo 3 ;;
    ws3c) echo 4 ;;
    *) echo 0 ;;
  esac
}

if [[ "$(stage_rank "$RESUME_FROM")" -eq 0 ]]; then
  echo "Invalid --resume-from: $RESUME_FROM (expected ws2|ws3a|ws3b|ws3c)" >&2
  exit 2
fi
if [[ "$WS3B_MODE" != "off" && "$WS3B_MODE" != "on" && "$WS3B_MODE" != "both" ]]; then
  echo "Invalid --ws3b-mode: $WS3B_MODE (expected off|on|both)" >&2
  exit 2
fi

should_run_stage() {
  local stage="$1"
  [[ "$(stage_rank "$stage")" -ge "$(stage_rank "$RESUME_FROM")" ]]
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 2
  fi
}

require_cmd python3
require_cmd curl
require_cmd timeout
require_cmd grep

wait_for_health() {
  local max_wait="${1:-60}"
  local waited=0
  while ((waited < max_wait)); do
    if curl -sS --max-time 2 "$API_URL/health" >/dev/null; then
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done
  return 1
}

run_step() {
  local name="$1"
  shift
  log "STEP START: $name"
  {
    echo "---- $name ----"
    "$@"
  } >>"$MASTER_LOG" 2>&1
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    log "STEP PASS: $name"
  else
    log "STEP FAIL($rc): $name"
    FAILED_STEPS+=("$name:$rc")
    if [[ $FAIL_FAST -eq 1 ]]; then
      log "Fail-fast enabled. Stopping."
      exit "$rc"
    fi
  fi
}

reload_orchestrator_api() {
  local -a env_kv=("$@")
  run_step "reload_orchestrator_api(${env_kv[*]:-default})" \
    env "${env_kv[@]}" python3 scripts/server/orchestrator_stack.py reload orchestrator
  run_step "wait_for_api_health" wait_for_health 90
}

run_chat_probe() {
  local name="$1"
  local prompt="$2"
  local out_json="$3"
  run_step "chat_probe_${name}" \
    timeout "$CHAT_TIMEOUT_S" python3 - "$API_URL" "$prompt" "$out_json" <<'PY'
import json
import sys
import urllib.request

api_url, prompt, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
payload = {
    "prompt": prompt,
    "real_mode": True,
    "mock_mode": False,
    "timeout_s": 120,
}
req = urllib.request.Request(
    f"{api_url}/chat",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=140) as resp:
    body = resp.read().decode("utf-8")
data = json.loads(body)
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
print(json.dumps({
    "elapsed_seconds": data.get("elapsed_seconds"),
    "mode": data.get("mode"),
    "routed_to": data.get("routed_to"),
    "role_history": data.get("role_history"),
    "error_code": data.get("error_code"),
}, indent=2))
PY
}

extract_cache_diag() {
  local in_json="$1"
  local out_json="$2"
  python3 - "$in_json" "$out_json" <<'PY'
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
with open(src, "r", encoding="utf-8") as f:
    data = json.load(f)

cache = data.get("cache_stats") or {}

def walk(node, out):
    if isinstance(node, dict):
        for k, v in node.items():
            lk = str(k).lower()
            if lk in {
                "frontdoor_repl_bypass_enabled",
                "frontdoor_repl_bypass_count",
                "router_hit_rate",
                "router_total_routes",
                "backend_hit_rate",
                "token_savings_pct",
            }:
                out[k] = v
            walk(v, out)
    elif isinstance(node, list):
        for x in node:
            walk(x, out)

out = {}
walk(cache, out)
with open(dst, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
PY
}

count_prewarm_lines() {
  local log_file="logs/orchestrator.log"
  if [[ ! -f "$log_file" ]]; then
    echo 0
    return
  fi
  grep -c "Pre-warmed architect slot on port" "$log_file" || true
}

run_ws3b_mode() {
  local mode="$1" # off|on
  local esc_flag="0"
  if [[ "$mode" == "on" ]]; then
    esc_flag="1"
  fi

  local preflight_done=0
  for seed in "${AB_SEEDS[@]}"; do
    if [[ -n "$WS3B_START_SEED" ]] && ((seed < WS3B_START_SEED)); then
      continue
    fi

    local -a cmd=(
      timeout "$WS3B_CMD_TIMEOUT_S"
      env "ESCALATION_COMPRESSION=$esc_flag" "ORCHESTRATOR_ESCALATION_COMPRESSION=$esc_flag"
      python3 scripts/benchmark/seed_specialist_routing.py
      --3way --suites "$AB_SUITE" --sample-size 1 --no-pool --seed "$seed"
      --timeout "$AB_TIMEOUT"
    )
    if [[ $WS3B_NO_PREFLIGHT -eq 0 ]] && [[ $WS3B_PREFLIGHT_ONCE -eq 0 || $preflight_done -eq 0 ]]; then
      cmd+=(--preflight)
      preflight_done=1
    fi

    run_step "ws3b_ab_${mode}_seed_${seed}" "${cmd[@]}"
  done
}

log "Output directory: $OUT_DIR"
log "API URL: $API_URL"
log "Resume from stage: $RESUME_FROM"

# -----------------------------------------------------------------------------
# Phase 0/1: WS2
# -----------------------------------------------------------------------------
if should_run_stage "ws2"; then
  run_step "api_health_initial" wait_for_health 30

  run_step "ws2_sweep_dry_run" \
    python3 scripts/benchmark/concurrent_inference_sweep.py --dry-run

  SWEEP_ROLES="frontdoor,coder,worker,fast_worker"
  if [[ $INCLUDE_ARCHITECTS -eq 1 ]]; then
    SWEEP_ROLES="$SWEEP_ROLES,architect_general"
  fi

  run_step "ws2_sweep_live" \
    python3 scripts/benchmark/concurrent_inference_sweep.py \
    --roles "$SWEEP_ROLES" \
    --n-warmup "$SWEEP_N_WARMUP" \
    --n-measured "$SWEEP_N_MEASURED" \
    --n-predict "$SWEEP_N_PREDICT" \
    --yes
fi

# -----------------------------------------------------------------------------
# Phase 2: WS3A
# -----------------------------------------------------------------------------
if should_run_stage "ws3a"; then
  reload_orchestrator_api \
    PREFIX_CACHE_BYPASS_FRONTDOOR_REPL=1 \
    ORCHESTRATOR_PREFIX_CACHE_BYPASS_FRONTDOOR_REPL=1

  run_chat_probe "ws3a_bypass_on_1" \
    "Give 3 concise steps to optimize Python file I/O for a codebase scan." \
    "$OUT_DIR/ws3a_bypass_on_1.json"
  run_chat_probe "ws3a_bypass_on_2" \
    "List likely bottlenecks in a recursive directory hash utility and how to profile them." \
    "$OUT_DIR/ws3a_bypass_on_2.json"
  run_step "ws3a_extract_bypass_on_1" extract_cache_diag "$OUT_DIR/ws3a_bypass_on_1.json" "$OUT_DIR/ws3a_bypass_on_1.cache_diag.json"
  run_step "ws3a_extract_bypass_on_2" extract_cache_diag "$OUT_DIR/ws3a_bypass_on_2.json" "$OUT_DIR/ws3a_bypass_on_2.cache_diag.json"

  reload_orchestrator_api \
    PREFIX_CACHE_BYPASS_FRONTDOOR_REPL=0 \
    ORCHESTRATOR_PREFIX_CACHE_BYPASS_FRONTDOOR_REPL=0

  run_chat_probe "ws3a_bypass_off_1" \
    "Give 3 concise steps to optimize Python file I/O for a codebase scan." \
    "$OUT_DIR/ws3a_bypass_off_1.json"
  run_chat_probe "ws3a_bypass_off_2" \
    "List likely bottlenecks in a recursive directory hash utility and how to profile them." \
    "$OUT_DIR/ws3a_bypass_off_2.json"
  run_step "ws3a_extract_bypass_off_1" extract_cache_diag "$OUT_DIR/ws3a_bypass_off_1.json" "$OUT_DIR/ws3a_bypass_off_1.cache_diag.json"
  run_step "ws3a_extract_bypass_off_2" extract_cache_diag "$OUT_DIR/ws3a_bypass_off_2.json" "$OUT_DIR/ws3a_bypass_off_2.cache_diag.json"
fi

# -----------------------------------------------------------------------------
# Phase 3: WS3B
# -----------------------------------------------------------------------------
if should_run_stage "ws3b"; then
  if [[ "$WS3B_MODE" == "off" || "$WS3B_MODE" == "both" ]]; then
    run_ws3b_mode "off"
  fi
  if [[ "$WS3B_MODE" == "on" || "$WS3B_MODE" == "both" ]]; then
    run_ws3b_mode "on"
  fi
fi

# -----------------------------------------------------------------------------
# Phase 4: WS3C
# -----------------------------------------------------------------------------
PREWARM_BEFORE=0
PREWARM_AFTER=0
PREWARM_DELTA=0
if should_run_stage "ws3c"; then
  reload_orchestrator_api \
    FRONTDOOR_TRACE=1 \
    ORCHESTRATOR_FRONTDOOR_TRACE=1 \
    DELEGATION_TRACE=1 \
    ORCHESTRATOR_DELEGATION_TRACE=1

  PREWARM_BEFORE="$(count_prewarm_lines)"
  log "WS3C prewarm log count before probes: $PREWARM_BEFORE"

  run_chat_probe "ws3c_complex_1" \
    "Design a robust migration plan for splitting a monolithic orchestration service into role-based workers with rollback, metrics, and failure isolation." \
    "$OUT_DIR/ws3c_complex_1.json"
  run_chat_probe "ws3c_complex_2" \
    "Propose an architecture and staged rollout for cache-aware delegation with diagnostics, contention controls, and regression guardrails across multiple LLM roles." \
    "$OUT_DIR/ws3c_complex_2.json"
  run_chat_probe "ws3c_complex_3" \
    "Create a detailed refactor strategy for a large orchestration API to support parallel tools, request budgets, and specialist escalation without deadlocks." \
    "$OUT_DIR/ws3c_complex_3.json"

  PREWARM_AFTER="$(count_prewarm_lines)"
  log "WS3C prewarm log count after probes: $PREWARM_AFTER"
  PREWARM_DELTA=$((PREWARM_AFTER - PREWARM_BEFORE))
  if ((PREWARM_DELTA < 0)); then
    PREWARM_DELTA=0
  fi
fi

# -----------------------------------------------------------------------------
# Final summary artifacts
# -----------------------------------------------------------------------------
python3 - "$OUT_DIR" "$SUMMARY_JSON" "$PREWARM_BEFORE" "$PREWARM_AFTER" "$PREWARM_DELTA" <<'PY'
import json
import os
import sys
from glob import glob

out_dir, summary_json, pre_before, pre_after, pre_delta = sys.argv[1:]

def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

cache_diags = {}
for p in sorted(glob(os.path.join(out_dir, "*.cache_diag.json"))):
    cache_diags[os.path.basename(p)] = read_json(p)

chat_role_histories = {}
for p in sorted(glob(os.path.join(out_dir, "ws3*.json"))):
    if p.endswith(".cache_diag.json"):
        continue
    data = read_json(p) or {}
    chat_role_histories[os.path.basename(p)] = {
        "role_history": data.get("role_history", []),
        "routed_to": data.get("routed_to"),
        "mode": data.get("mode"),
        "error_code": data.get("error_code"),
        "elapsed_seconds": data.get("elapsed_seconds"),
    }

summary = {
    "out_dir": out_dir,
    "ws3a_cache_diags": cache_diags,
    "ws3c_prewarm_log_counts": {
        "before": int(pre_before),
        "after": int(pre_after),
        "delta": int(pre_delta),
    },
    "ws3_chat_probes": chat_role_histories,
}

with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
PY

{
  echo "# Perf Parallel Tools Validation Summary"
  echo
  echo "- Output dir: \`$OUT_DIR\`"
  echo "- Master log: \`$MASTER_LOG\`"
  echo "- Summary JSON: \`$SUMMARY_JSON\`"
  echo
  if [[ ${#FAILED_STEPS[@]} -eq 0 ]]; then
    echo "## Status"
    echo
    echo "- All steps completed without command-level failures."
  else
    echo "## Status"
    echo
    echo "- Some steps failed:"
    for s in "${FAILED_STEPS[@]}"; do
      echo "  - \`$s\`"
    done
  fi
  echo
  echo "## Notes"
  echo
  echo "- WS2 sweep CSV + summary JSON are generated by \`concurrent_inference_sweep.py\` under \`benchmarks/results/eval/\`."
  echo "- WS3A bypass ON/OFF cache diagnostics are under \`$OUT_DIR/*.cache_diag.json\`."
  echo "- WS3C evidence uses orchestrator log line counts for \`Pre-warmed architect slot on port\`."
  echo "- WS3B A/B runs are logged in \`$MASTER_LOG\` (search for \`ws3b_ab_\`)."
} >"$SUMMARY_MD"

if [[ ${#FAILED_STEPS[@]} -eq 0 ]]; then
  log "Run complete: all steps passed."
  exit 0
fi

log "Run complete: some steps failed."
exit 1
