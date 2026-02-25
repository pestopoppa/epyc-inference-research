#!/bin/bash
# Agentic/Tool-Use Model Quality Rubric Test Script
# Runs all T1/T2/T3 tool-calling and agentic questions
#
# Single Config Mode: Pass config as 4th param
# Multi Config Mode: Omit 4th param, uses shared lib to determine configs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Configuration
MODEL="${1:-}"
MODEL_NAME="${2:-unknown}"
MODEL_ARCH="${3:-dense}" # dense, qwen3moe
CONFIG_PARAM="${4:-}"    # Optional: specific config to run (single config mode)
OUTPUT_DIR="/mnt/raid0/llm/tmp/agentic_rubric_results"
LLAMA_COMPLETION="/mnt/raid0/llm/llama.cpp/build/bin/llama-completion"
TIMEOUT=120

# Source shared libraries
if [[ -f "$SCRIPT_DIR/lib/optimization_configs.sh" ]]; then
  source "$SCRIPT_DIR/lib/optimization_configs.sh"
fi

if [[ -z "$MODEL" ]]; then
  echo "Usage: $0 <model.gguf> <model_name> [arch] [config]"
  echo ""
  echo "Architecture types:"
  echo "  dense       - Standard dense model (no MoE optimization)"
  echo "  qwen3moe    - Qwen3-MoE model (tests baseline + MoE reduction)"
  echo ""
  echo "Single config mode: Specify config as 4th param"
  echo "Multi config mode: Omit 4th param, let shared lib determine configs"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Determine configurations to test
declare -a CONFIGS
if [[ -n "$CONFIG_PARAM" ]]; then
  # Single config mode
  CONFIGS=("$CONFIG_PARAM")
  echo "[SINGLE CONFIG MODE] Running only: $CONFIG_PARAM"
else
  # Multi-config mode - use shared library if available
  if type setup_configs &>/dev/null; then
    setup_configs "$MODEL_NAME" "$MODEL_ARCH"
  else
    # Fallback
    CONFIGS=("baseline")
    case "$MODEL_ARCH" in
      qwen3moe | qwen3next)
        for exp in 2 4 6 8; do
          CONFIGS+=("moe${exp}")
        done
        ;;
    esac
  fi
fi

echo "=============================================="
echo "Agentic/Tool-Use Model Quality Rubric"
echo "Model: $MODEL_NAME"
echo "Model file: $MODEL"
echo "Architecture: $MODEL_ARCH"
echo "Configurations to test: ${CONFIGS[*]}"
echo "Date: $(date)"
echo "=============================================="

# Function to run a test and extract timing
run_test() {
  local test_name="$1"
  local prompt="$2"
  local config="$3"
  local output_file="$OUTPUT_DIR/${MODEL_NAME}_${config}_${test_name}.txt"

  echo ""
  echo "--- Running $test_name ($config) ---"

  # DRY RUN: Skip actual model invocation but iterate through all tests
  if [[ "${BENCHMARK_DRY_RUN:-false}" == "true" ]]; then
    echo "[DRY RUN] Would run: $MODEL_NAME | $config | $test_name"
    cat >"$output_file" <<'DRYRUN'
DRY_RUN: test placeholder
llama_print_timings:        eval time =    1000.00 ms /   100 tokens (   10.00 ms per token,   0.00 tokens per second)
DRYRUN
    echo "Speed: 0.00 tokens per second (dry run)"
    return 0
  fi

  # Compute MoE override if needed
  local moe_override=""
  if [[ "$config" =~ ^moe([0-9]+) ]]; then
    if type get_moe_override &>/dev/null; then
      moe_override=$(get_moe_override "$config" "$MODEL_ARCH")
    else
      local exp="${BASH_REMATCH[1]}"
      case "$MODEL_ARCH" in
        qwen3moe | qwen3next) moe_override="--override-kv qwen3moe.expert_used_count=int:$exp" ;;
      esac
    fi
  fi

  # Write prompt to temp file
  echo "$prompt" >"/mnt/raid0/llm/tmp/agentic_prompt.txt"

  # Run model and capture output
  timeout "$TIMEOUT" OMP_NUM_THREADS=1 numactl --interleave=all \
    "$LLAMA_COMPLETION" \
    -m "$MODEL" \
    -t 96 -n 512 --temp 0.2 \
    $moe_override \
    -f "/mnt/raid0/llm/tmp/agentic_prompt.txt" \
    >"$output_file" 2>&1 || true

  # Extract timing
  local speed
  speed=$(grep "eval time" "$output_file" | grep -oP '\d+\.\d+ tokens per second' | tail -1 || echo "N/A")

  echo "Speed: $speed"
  echo "Output saved to: $output_file"

  # Show the answer
  echo "--- Answer ---"
  grep -v "^llama\|^load\|^print_info\|^common\|^sampler\|^generate\|^system_info\|^main:" "$output_file" | tail -30 | head -25
}

# Run tests for each configuration
for CONFIG in "${CONFIGS[@]}"; do
  echo ""
  echo "##############################################"
  echo "# Configuration: $CONFIG"
  if [[ "$CONFIG" =~ ^moe ]] && type get_moe_override &>/dev/null; then
    echo "# Override: $(get_moe_override "$CONFIG" "$MODEL_ARCH")"
  fi
  echo "##############################################"

  # T1: Baseline questions
  echo ""
  echo "========== TIER 1 (Baseline) =========="

  run_test "t1_q1_single_tool" \
    'You have access to a tool:
{
  "name": "get_weather",
  "parameters": {
    "city": "string (required)",
    "units": "string (optional, celsius or fahrenheit, default celsius)"
  }
}

User asks: "What is the weather in Tokyo?"
Generate the tool call as JSON. Output only the JSON, nothing else.' "$CONFIG"

  run_test "t1_q2_multi_param" \
    'Tool available:
{
  "name": "search_files",
  "parameters": {
    "pattern": "string (required) - glob pattern",
    "directory": "string (required) - path to search",
    "max_results": "integer (optional, default 10)"
  }
}

User asks: "Find all Python files in /src with max 5 results"
Generate the tool call as JSON. Output only the JSON.' "$CONFIG"

  run_test "t1_q3_choose_tool" \
    'Available tools:
1. read_file: {"path": "string"} - Read file contents
2. write_file: {"path": "string", "content": "string"} - Write to file
3. list_directory: {"path": "string"} - List directory contents

User asks: "Show me what is in the config folder"
Which tool and what parameters? Output only the JSON tool call.' "$CONFIG"

  # T2: Medium-Hard questions
  echo ""
  echo "========== TIER 2 (Medium-Hard) =========="

  run_test "t2_q1_sequential" \
    'Tools:
- read_file: {"path": "string"}
- grep_search: {"pattern": "string", "path": "string"}

User asks: "Find where ERROR is logged in /var/log/app.log, then show me that file"

Generate the tool calls in order as a JSON array. Output only the JSON array.' "$CONFIG"

  run_test "t2_q2_error_handling" \
    'You called: {"name": "get_user", "parameters": {"id": 123}}

Tool returned: {"error": "User not found", "code": 404}

Tools available:
- get_user: {"id": "integer"}
- search_users: {"query": "string"} - Search by name/email
- create_user: {"name": "string", "email": "string"}

What should you do next? Either output a tool call as JSON, or a message to the user asking for clarification.' "$CONFIG"

  run_test "t2_q3_nested_params" \
    'Tool:
{
  "name": "create_task",
  "parameters": {
    "title": "string (required)",
    "assignees": "array of strings (required)",
    "metadata": {
      "priority": "string (low/medium/high)",
      "tags": "array of strings",
      "due_date": "string (ISO date)"
    }
  }
}

User: "Create a high priority task called Fix login bug for Alice and Bob, tagged as bug and urgent, due 2024-03-15"

Output only the JSON tool call.' "$CONFIG"

  # T3: Hard questions
  echo ""
  echo "========== TIER 3 (Hard) =========="

  run_test "t3_q1_ambiguous" \
    'Tools:
- send_email: {"to": "string", "subject": "string", "body": "string"}
- send_slack: {"channel": "string", "message": "string"}
- create_ticket: {"title": "string", "description": "string", "assignee": "string"}

User: "Let the team know about the outage"

This is ambiguous. Either ask a clarifying question, OR if you must act, choose the most appropriate tool and explain why in one sentence, then output the JSON.' "$CONFIG"

  run_test "t3_q2_error_chain" \
    'Conversation so far:

You: {"name": "deploy", "parameters": {"env": "prod", "version": "1.2.3"}}
Tool: {"error": "Deployment blocked: failing health check on staging"}

You: {"name": "get_health", "parameters": {"env": "staging"}}
Tool: {"status": "unhealthy", "failing_checks": ["database_connection"]}

You: {"name": "check_database", "parameters": {"env": "staging"}}
Tool: {"status": "connection_refused", "host": "db-staging.internal", "port": 5432}

What is your next action? Available tools:
- restart_service: {"service": "string", "env": "string"}
- get_logs: {"service": "string", "env": "string", "lines": "integer"}
- notify_oncall: {"message": "string", "severity": "string"}
- check_dns: {"hostname": "string"}

Explain your reasoning briefly, then output the JSON tool call.' "$CONFIG"

  run_test "t3_q3_schema_edge" \
    'Tool schema:
{
  "name": "query_api",
  "parameters": {
    "endpoint": "string (required) - must start with /",
    "method": "string (required) - GET/POST/PUT/DELETE",
    "body": "object (required for POST/PUT, must be null for GET/DELETE)",
    "headers": "object (optional)"
  }
}

User: "GET the users endpoint with an auth header Bearer token123"

Generate the correct tool call. Output only the JSON.' "$CONFIG"

  run_test "t3_q4_orchestration" \
    'You need to deploy a hotfix. Tools available:
- git_checkout: {"branch": "string"}
- run_tests: {"suite": "string"} - returns pass/fail
- build_image: {"tag": "string"}
- deploy: {"env": "string", "image": "string"}
- notify_slack: {"channel": "string", "message": "string"}
- rollback: {"env": "string", "to_version": "string"}

Current state: main branch, last deploy was v1.2.2, hotfix is on branch hotfix/auth-fix

Create a deployment plan as ordered tool calls. Include what to do if tests fail or deploy fails.
Output as JSON with structure: {"steps": [...], "on_test_fail": [...], "on_deploy_fail": [...]}' "$CONFIG"

  # Configuration Summary
  echo ""
  echo "=============================================="
  echo "CONFIG $CONFIG COMPLETE"
  echo "=============================================="
  echo "Speed summary for $MODEL_NAME ($CONFIG):"

  max_speed=0 q_count=0
  for f in "$OUTPUT_DIR"/${MODEL_NAME}_${CONFIG}_*.txt; do
    if [[ -f "$f" ]]; then
      ((q_count++)) || true
      test_name=$(basename "$f" .txt | sed "s/${MODEL_NAME}_${CONFIG}_//")
      speed=$(grep "eval time" "$f" 2>/dev/null | grep -oP '\d+\.\d+(?= tokens per second)' | tail -1 || echo "0")
      echo "  $test_name: $speed"
      [[ -n "$speed" ]] && (($(echo "$speed > $max_speed" | bc -l 2>/dev/null || echo 0))) && max_speed="$speed"
    fi
  done

  # Add discovery info for spec_k configs: draft_model,K=N,T=X
  discovery_info="-"
  if [[ "$CONFIG" =~ ^spec_k([0-9]+) ]]; then
    k_val="${BASH_REMATCH[1]}"
    draft_name=$(basename "${DRAFT_MODEL_PATH:-unknown}" .gguf | cut -c1-15)
    discovery_info="${draft_name},K=${k_val},T=0.2"
  fi
  printf "%-10s %-20s %-8s %2d %8s %-30s\n" "agentic" "${MODEL_NAME:0:20}" "$CONFIG" "$q_count" "${max_speed}" "$discovery_info" >>/mnt/raid0/llm/tmp/benchmark_completions.log
done

# Final Summary
echo ""
echo "=============================================="
echo "AGENTIC RUBRIC TEST COMPLETE - ALL CONFIGURATIONS"
echo "=============================================="
echo "Results saved to: $OUTPUT_DIR"
