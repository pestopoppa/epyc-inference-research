#!/bin/bash
# DS-E1 production KV-size measurement harness.
#
# This is a clean-window runner for the Dynamic Stack Phase-E packet. It writes
# to data/dynamic_stack/**/kv* so epyc-orchestrator's DS-E1 packet can discover
# the artifact. By default it only prints the planned matrix; pass --execute to
# launch llama-server instances and collect measurements.

set -euo pipefail

RESEARCH_ROOT="/mnt/raid0/llm/epyc-inference-research"
LLAMA_SERVER="${LLAMA_SERVER:-/mnt/raid0/llm/llama.cpp/build/bin/llama-server}"
TIMESTAMP="${DS_E1_KV_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESEARCH_ROOT}/data/dynamic_stack/ds_e1_kv_measurements_${TIMESTAMP}}"
RESULTS_FILE="${OUTPUT_DIR}/kv_measurements.csv"
LOG_DIR="${OUTPUT_DIR}/logs"
PORT="${PORT:-8194}"
THREADS="${THREADS:-96}"
UBATCH="${UBATCH:-8192}"
N_PREDICT="${N_PREDICT:-32}"
EXECUTE=0
ROLE_FILTER=""
CTX_FILTER=""
ALLOW_ACTIVE_AUTOPILOT=0
ALLOW_LIVE_LLAMA=0
WRITE_PLAN=0

# role|model_id|model_path|max_ctx
TARGETS=(
  "frontdoor|qwen3.6-35b-a3b-q8_0|/mnt/raid0/llm/models/Qwen_Qwen3.6-35B-A3B-Q8_0.gguf|32768"
  "ingest_long_context|qwen3-next-80b-a3b-q4_k_m|/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf|32768"
  "worker_general|gemma4-26b-a4b-q4_k_m|/mnt/raid0/llm/models/gemma-4-26B-A4B-it-Q4_K_M.gguf|16384"
  "architect_general|qwen3.5-122b-a10b-q4_k_m|/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3.5-122B-A10B-GGUF/Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf|16384"
)
CONTEXTS=(2048 8192 32768)

usage() {
  cat <<'EOF'
Usage: ds_e1_kv_measurements.sh [--execute] [--role ROLE] [--ctx TOKENS] [--write-plan]

Default mode is dry-run. The execute mode starts one production-configured
llama-server at a time, measures RSS and server-reported KV allocation, and
writes:
  data/dynamic_stack/ds_e1_kv_measurements_<timestamp>/kv_measurements.csv

Execute mode fails closed when AutoPilot or existing llama-server processes are
live. Override only for an intentional coordinated window:
  --allow-active-autopilot
  --allow-live-llama

Environment overrides:
  LLAMA_SERVER, OUTPUT_DIR, PORT, THREADS, UBATCH, N_PREDICT

Planning:
  --write-plan writes measurement_plan.json and run_clean_window.sh under
  OUTPUT_DIR without starting inference. The runner preserves the timestamp,
  output path, role/context filters, and measurement knobs for the clean window.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute) EXECUTE=1; shift ;;
    --role) ROLE_FILTER="$2"; shift 2 ;;
    --ctx) CTX_FILTER="$2"; shift 2 ;;
    --allow-active-autopilot) ALLOW_ACTIVE_AUTOPILOT=1; shift ;;
    --allow-live-llama) ALLOW_LIVE_LLAMA=1; shift ;;
    --write-plan) WRITE_PLAN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

port_in_use() {
  python3 - "$PORT" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(0.25)
try:
    sys.exit(0 if sock.connect_ex(("127.0.0.1", port)) == 0 else 1)
finally:
    sock.close()
PY
}

clean_window_preflight() {
  local blockers=()
  local autopilot_pids=""
  local llama_pids=""

  if [[ "$ALLOW_ACTIVE_AUTOPILOT" -ne 1 ]]; then
    autopilot_pids=$(pgrep -af "scripts/autopilot/autopilot.py start" || true)
    if [[ -n "$autopilot_pids" ]]; then
      blockers+=("active AutoPilot process(es): ${autopilot_pids//$'\n'/; }")
    fi
  fi

  if [[ "$ALLOW_LIVE_LLAMA" -ne 1 ]]; then
    llama_pids=$(pgrep -a -x "llama-server" || true)
    if [[ -n "$llama_pids" ]]; then
      blockers+=("live llama-server process(es): ${llama_pids//$'\n'/; }")
    fi
  fi

  if port_in_use; then
    blockers+=("measurement port ${PORT} is already accepting connections")
  fi

  if [[ "${#blockers[@]}" -gt 0 ]]; then
    echo "Refusing DS-E1 KV execute mode: clean-window preflight failed." >&2
    printf '  - %s\n' "${blockers[@]}" >&2
    echo "Stop or coordinate the live workload, or pass the explicit --allow-* override for an intentional contaminated run." >&2
    exit 3
  fi
}

generate_prompt() {
  local target_tokens=$1
  python3 - "$target_tokens" <<'PY'
import sys
target_tokens = int(sys.argv[1])
seed = "Measure production KV residency for a long-context routing workload.\n\n"
filler = (
    "This synthetic paragraph preserves deterministic token pressure while "
    "avoiding external dependencies. It describes routing, factual-risk review, "
    "dynamic stack scheduling, and context residency under production flags. "
)
target_chars = max(512, int(target_tokens * 3.5))
text = seed + (filler * ((target_chars // len(filler)) + 1))
print(text[:target_chars] + "\n\nReturn exactly one sentence.")
PY
}

wait_for_server() {
  local port=$1
  local elapsed=0
  while true; do
    if curl -fsS "http://127.0.0.1:${port}/health" 2>/dev/null | grep -q '"status":"ok"'; then
      return 0
    fi
    sleep 2
    elapsed=$((elapsed + 2))
    if [[ "$elapsed" -ge 600 ]]; then
      return 1
    fi
  done
}

rss_mb() {
  local pid=$1
  local rss_kb
  rss_kb=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ')
  python3 - "${rss_kb:-0}" <<'PY'
import sys
rss = int(sys.argv[1] or 0)
print(f"{rss / 1024:.1f}")
PY
}

stop_server() {
  local pid=$1
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  sleep 3
}

write_plan_artifacts() {
  local plan_file="${OUTPUT_DIR}/measurement_plan.json"
  local runner_file="${OUTPUT_DIR}/run_clean_window.sh"
  mkdir -p "$OUTPUT_DIR"
  python3 - "$plan_file" "$runner_file" "$RESEARCH_ROOT" "$TIMESTAMP" "$OUTPUT_DIR" \
    "$RESULTS_FILE" "$LLAMA_SERVER" "$PORT" "$THREADS" "$UBATCH" "$N_PREDICT" \
    "$ROLE_FILTER" "$CTX_FILTER" "${planned_rows[@]}" <<'PY'
import json
from pathlib import Path
import shlex
import sys

(
    plan_file,
    runner_file,
    research_root,
    timestamp,
    output_dir,
    results_file,
    llama_server,
    port,
    threads,
    ubatch,
    n_predict,
    role_filter,
    ctx_filter,
    *rows,
) = sys.argv[1:]

parsed_rows = []
for raw in rows:
    role, model_id, model_path, max_ctx, ctx = raw.split("|", 4)
    parsed_rows.append({
        "role": role,
        "model_id": model_id,
        "model_path": model_path,
        "context_length": int(ctx),
        "max_context": int(max_ctx),
    })

execute_parts = [
    "DS_E1_KV_TIMESTAMP=" + shlex.quote(timestamp),
    "OUTPUT_DIR=" + shlex.quote(output_dir),
    "LLAMA_SERVER=" + shlex.quote(llama_server),
    "PORT=" + shlex.quote(port),
    "THREADS=" + shlex.quote(threads),
    "UBATCH=" + shlex.quote(ubatch),
    "N_PREDICT=" + shlex.quote(n_predict),
    "bash",
    "scripts/benchmark/ds_e1_kv_measurements.sh",
    "--execute",
]
if role_filter:
    execute_parts.extend(["--role", shlex.quote(role_filter)])
if ctx_filter:
    execute_parts.extend(["--ctx", shlex.quote(ctx_filter)])
execute_command = " ".join(execute_parts)

plan = {
    "schema": "ds_e1_kv_measurement_plan.v1",
    "timestamp": timestamp,
    "output_dir": output_dir,
    "results_file": results_file,
    "execute_command": f"cd {shlex.quote(research_root)} && {execute_command}",
    "role_filter": role_filter,
    "ctx_filter": int(ctx_filter) if ctx_filter else None,
    "measurement_knobs": {
        "llama_server": llama_server,
        "port": int(port),
        "threads": int(threads),
        "ubatch": int(ubatch),
        "n_predict": int(n_predict),
    },
    "rows": parsed_rows,
    "clean_window_required": True,
    "contamination_overrides_excluded": [
        "--allow-active-autopilot",
        "--allow-live-llama",
    ],
}
Path(plan_file).write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
runner = "\n".join([
    "#!/bin/bash",
    "set -euo pipefail",
    f"cd {shlex.quote(research_root)}",
    execute_command,
    "",
])
Path(runner_file).write_text(runner, encoding="utf-8")
Path(runner_file).chmod(0o755)
PY
  echo "Wrote DS-E1 plan artifacts:"
  printf '  plan: %s\n' "$plan_file"
  printf '  runner: %s\n' "$runner_file"
}

run_prefill() {
  local prompt=$1
  curl -fsS --max-time 900 "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"ds-e1-kv\",
      \"messages\": [{\"role\": \"user\", \"content\": $(printf '%s' "$prompt" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')}],
      \"max_tokens\": ${N_PREDICT},
      \"temperature\": 0,
      \"stream\": false
    }" | python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)
    usage = data.get("usage", {})
    timings = data.get("timings", {})
    prompt_tps = float(timings.get("prompt_per_second", 0) or 0)
    print(",".join([
        str(usage.get("prompt_tokens", 0)),
        f"{prompt_tps:.2f}",
    ]))
except Exception:
    print("0,0.00")
'
}

planned_rows=()
for target in "${TARGETS[@]}"; do
  IFS='|' read -r role model_id model_path max_ctx <<< "$target"
  [[ -n "$ROLE_FILTER" && "$role" != "$ROLE_FILTER" ]] && continue
  for ctx in "${CONTEXTS[@]}"; do
    [[ -n "$CTX_FILTER" && "$ctx" != "$CTX_FILTER" ]] && continue
    if [[ "$ctx" -gt "$max_ctx" ]]; then
      continue
    fi
    planned_rows+=("${role}|${model_id}|${model_path}|${max_ctx}|${ctx}")
  done
done

if [[ "${#planned_rows[@]}" -eq 0 ]]; then
  echo "No DS-E1 KV measurement rows selected." >&2
  exit 1
fi

echo "DS-E1 KV measurement matrix (${#planned_rows[@]} rows)"
printf '  output: %s\n' "$RESULTS_FILE"
printf '  mode: %s\n' "$([[ "$EXECUTE" -eq 1 ]] && echo execute || echo dry-run)"
for row in "${planned_rows[@]}"; do
  IFS='|' read -r role model_id model_path max_ctx ctx <<< "$row"
  printf '  - role=%s model=%s ctx=%s max_ctx=%s\n' "$role" "$model_id" "$ctx" "$max_ctx"
done

if [[ "$WRITE_PLAN" -eq 1 ]]; then
  write_plan_artifacts
fi

if [[ "$EXECUTE" -ne 1 ]]; then
  echo
  echo "Dry-run only. Re-run with --execute in a clean window to collect measurements."
  exit 0
fi

clean_window_preflight

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
echo "role,model_id,model_path,context_length,max_context,ctk,ctv,hadamard,status,rss_load_mb,rss_after_prefill_mb,server_kv_size_mb,prompt_tokens,prompt_tps,log_file,notes" > "$RESULTS_FILE"

for row in "${planned_rows[@]}"; do
  IFS='|' read -r role model_id model_path max_ctx ctx <<< "$row"
  log_file="${LOG_DIR}/${role}_ctx${ctx}.log"
  notes=""
  if [[ ! -f "$model_path" ]]; then
    notes="model_path_missing"
    echo "$role,$model_id,$model_path,$ctx,$max_ctx,q4_0,f16,yes,missing_model,0,0,0,0,0,$log_file,$notes" >> "$RESULTS_FILE"
    continue
  fi

  server_args=(
    "$LLAMA_SERVER"
    -m "$model_path"
    -t "$THREADS"
    -np 1
    --port "$PORT"
    -ngl 0
    --flash-attn
    on
    -c "$ctx"
    -ub "$UBATCH"
    -ctk q4_0
    -ctv f16
    --kv-hadamard
  )

  echo "Measuring $role ctx=$ctx"
  numactl --interleave=all "${server_args[@]}" > "$log_file" 2>&1 &
  pid=$!
  if ! wait_for_server "$PORT"; then
    stop_server "$pid"
    echo "$role,$model_id,$model_path,$ctx,$max_ctx,q4_0,f16,yes,start_failed,0,0,0,0,0,$log_file,server_health_timeout" >> "$RESULTS_FILE"
    continue
  fi

  rss_load=$(rss_mb "$pid")
  kv_mb=$(grep -oP 'llama_kv_cache: size =\s*\K[0-9.]+' "$log_file" 2>/dev/null | tail -1 || true)
  [[ -z "$kv_mb" ]] && kv_mb=$(grep -oP 'KV buffer size\s*=\s*\K[0-9.]+' "$log_file" 2>/dev/null | tail -1 || true)
  [[ -z "$kv_mb" ]] && kv_mb="0"

  prompt=$(generate_prompt $((ctx * 3 / 4)))
  prefill=$(run_prefill "$prompt" || echo "0,0.00")
  prompt_tokens=$(echo "$prefill" | cut -d, -f1)
  prompt_tps=$(echo "$prefill" | cut -d, -f2)
  rss_after=$(rss_mb "$pid")

  echo "$role,$model_id,$model_path,$ctx,$max_ctx,q4_0,f16,yes,ok,$rss_load,$rss_after,$kv_mb,$prompt_tokens,$prompt_tps,$log_file,$notes" >> "$RESULTS_FILE"
  stop_server "$pid"
done

echo "Done: $RESULTS_FILE"
