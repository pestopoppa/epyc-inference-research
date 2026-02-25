#!/bin/bash
# Formalizer Model Evaluation Script
#
# Evaluates formalizer models on their ability to convert vague task descriptions
# into well-structured FormalizationIR JSON.
#
# Usage:
#   ./bench_formalizers.sh --model <model_path> --prompts <prompts_dir> [--output <output_dir>]
#
# Example:
#   ./bench_formalizers.sh \
#     --model /mnt/raid0/llm/models/xLAM-2-1B-FC-r.Q4_K_M.gguf \
#     --prompts benchmarks/prompts/v1/formalizer/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source environment library for path variables
# shellcheck source=../lib/env.sh
source "${SCRIPT_DIR}/../lib/env.sh"

# Configuration (use unique var name to avoid conflict with env.sh LOG_DIR)
LLAMA_BIN="${LLAMA_BIN:-${LLAMA_CPP_BIN}}"
THREADS="${THREADS:-96}"
FORMALIZER_LOG_DIR="${FORMALIZER_LOG_DIR:-${PROJECT_ROOT}/logs/formalizer_eval}"
SCHEMA_PATH="${PROJECT_ROOT}/orchestration/formalization_ir.schema.json"

# Parse arguments
MODEL=""
PROMPTS_DIR=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --prompts)
      PROMPTS_DIR="$2"
      shift 2
      ;;
    --output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Validate arguments
if [[ -z "$MODEL" ]]; then
  echo "Error: --model is required"
  exit 1
fi

if [[ -z "$PROMPTS_DIR" ]]; then
  echo "Error: --prompts is required"
  exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  OUTPUT_DIR="${FORMALIZER_LOG_DIR}/run_${TIMESTAMP}"
fi

mkdir -p "$OUTPUT_DIR"

# Extract model name for reporting
MODEL_NAME=$(basename "$MODEL" .gguf)

echo "Formalizer Evaluation"
echo "===================="
echo ""
echo "Model: $MODEL_NAME"
echo "Prompts: $PROMPTS_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# System prompt for formalizer
SYSTEM_PROMPT='You are a task formalizer. Convert the given task description into a FormalizationIR JSON object.

The JSON must include:
- problem_id: A unique identifier for this formalization
- original_task: The original task description
- formal_specification: Object containing problem_type and relevant fields
- acceptance_criteria: Array of testable criteria

For tool_orchestration problems, include tool_sequence with steps.
For architecture problems, include architecture with components and interfaces.
For proof/verification, include preconditions, postconditions, and invariants.

Output ONLY valid JSON, no explanations.'

# Results tracking
RESULTS_CSV="${OUTPUT_DIR}/results.csv"
echo "test_case,category,parsable,completeness,schema_valid,time_sec,tokens_per_sec" >"$RESULTS_CSV"

# Function to evaluate a single prompt
evaluate_prompt() {
  local prompt_file="$1"
  local ground_truth_file="$2"
  local category="$3"
  local test_name
  test_name=$(basename "$prompt_file" .txt)

  echo -n "  Evaluating $test_name..."

  local prompt_content
  prompt_content=$(cat "$prompt_file")

  local output_file="${OUTPUT_DIR}/${category}_${test_name}_output.json"
  local temp_prompt="${OUTPUT_DIR}/temp_prompt.txt"

  # Create prompt with system instruction
  cat >"$temp_prompt" <<EOF
$SYSTEM_PROMPT

Task to formalize:
$prompt_content

FormalizationIR JSON:
EOF

  # Run inference
  local start_time
  start_time=$(date +%s.%N)

  if ! "$LLAMA_BIN/llama-cli" \
    -m "$MODEL" \
    -t "$THREADS" \
    -n 2048 \
    --temp 0 \
    -f "$temp_prompt" \
    --no-display-prompt \
    --simple-io \
    2>/dev/null >"$output_file"; then
    echo " INFERENCE FAILED"
    echo "$test_name,$category,0,0,0,0,0" >>"$RESULTS_CSV"
    return
  fi

  local end_time
  end_time=$(date +%s.%N)
  local duration
  duration=$(echo "$end_time - $start_time" | bc)

  # Extract JSON from output (may contain extra text)
  local json_output
  json_output=$(grep -oP '\{.*\}' "$output_file" | head -1 || echo "")

  # Score: Parsability (0 or 1)
  local parsable=0
  if python3 -c "import json; json.loads('$json_output')" 2>/dev/null; then
    parsable=1
  fi

  # Score: Schema validity (0 or 1)
  local schema_valid=0
  if [[ $parsable -eq 1 ]]; then
    if python3 -c "
import json
import jsonschema
schema = json.load(open('$SCHEMA_PATH'))
data = json.loads('$json_output')
try:
    jsonschema.validate(data, schema)
    print('valid')
except:
    pass
" 2>/dev/null | grep -q "valid"; then
      schema_valid=1
    fi
  fi

  # Score: Completeness (0.0 - 1.0)
  local completeness=0
  if [[ $parsable -eq 1 ]] && [[ -f "$ground_truth_file" ]]; then
    completeness=$(
      python3 <<'PYTHON'
import json
import sys

try:
    output = json.loads('''$json_output''')
    ground_truth = json.load(open("$ground_truth_file"))

    # Check required fields
    required = ["problem_id", "original_task", "formal_specification", "acceptance_criteria"]
    fields_present = sum(1 for f in required if f in output)

    # Check problem_type specific fields
    if output.get("formal_specification", {}).get("problem_type") == "tool_orchestration":
        if "tool_sequence" in output.get("formal_specification", {}):
            fields_present += 1
    elif output.get("formal_specification", {}).get("problem_type") == "architecture":
        if "architecture" in output.get("formal_specification", {}):
            fields_present += 1
    elif output.get("formal_specification", {}).get("problem_type") == "proof":
        if "preconditions" in output.get("formal_specification", {}):
            fields_present += 1

    max_fields = 5  # 4 required + 1 type-specific
    completeness = fields_present / max_fields
    print(f"{completeness:.2f}")
except Exception as e:
    print("0.00")
PYTHON
    )
  fi

  # Calculate tokens/sec (approximate)
  local token_count
  token_count=$(wc -w <"$output_file")
  local tps
  tps=$(echo "scale=2; $token_count / $duration" | bc)

  echo " parsable=$parsable complete=$completeness valid=$schema_valid ${tps}t/s"
  echo "$test_name,$category,$parsable,$completeness,$schema_valid,$duration,$tps" >>"$RESULTS_CSV"

  rm -f "$temp_prompt"
}

# Evaluate tool formalization prompts
echo "Tool Formalization Tests:"
for prompt in "$PROMPTS_DIR/tool_formalization/"*.txt; do
  if [[ -f "$prompt" ]]; then
    test_name=$(basename "$prompt" .txt)
    ground_truth="$PROMPTS_DIR/ground_truth/${test_name}.json"
    evaluate_prompt "$prompt" "$ground_truth" "tool"
  fi
done

echo ""

# Evaluate architecture formalization prompts
echo "Architecture Formalization Tests:"
for prompt in "$PROMPTS_DIR/architecture_formalization/"*.txt; do
  if [[ -f "$prompt" ]]; then
    test_name=$(basename "$prompt" .txt)
    ground_truth="$PROMPTS_DIR/ground_truth/arch_${test_name}.json"
    evaluate_prompt "$prompt" "$ground_truth" "arch"
  fi
done

echo ""

# Evaluate verification formalization prompts
echo "Verification Formalization Tests:"
for prompt in "$PROMPTS_DIR/verification_formalization/"*.txt; do
  if [[ -f "$prompt" ]]; then
    test_name=$(basename "$prompt" .txt)
    ground_truth="$PROMPTS_DIR/ground_truth/verif_${test_name}.json"
    evaluate_prompt "$prompt" "$ground_truth" "verif"
  fi
done

echo ""
echo "===================="
echo "Results Summary"
echo "===================="

# Calculate summary statistics
python3 <<PYTHON
import csv
import sys

with open("$RESULTS_CSV") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

if not rows:
    print("No results to summarize")
    sys.exit(0)

total = len(rows)
parsable = sum(1 for r in rows if int(r["parsable"]) == 1)
schema_valid = sum(1 for r in rows if int(r["schema_valid"]) == 1)
avg_completeness = sum(float(r["completeness"]) for r in rows) / total
avg_tps = sum(float(r["tokens_per_sec"]) for r in rows) / total

print(f"Total tests:      {total}")
print(f"Parsable JSON:    {parsable}/{total} ({100*parsable/total:.1f}%)")
print(f"Schema valid:     {schema_valid}/{total} ({100*schema_valid/total:.1f}%)")
print(f"Avg completeness: {avg_completeness:.2f}")
print(f"Avg tokens/sec:   {avg_tps:.1f}")

# Per-category breakdown
categories = set(r["category"] for r in rows)
print("")
print("Per-category breakdown:")
for cat in sorted(categories):
    cat_rows = [r for r in rows if r["category"] == cat]
    cat_parsable = sum(1 for r in cat_rows if int(r["parsable"]) == 1)
    cat_complete = sum(float(r["completeness"]) for r in cat_rows) / len(cat_rows)
    print(f"  {cat}: {cat_parsable}/{len(cat_rows)} parsable, {cat_complete:.2f} avg completeness")
PYTHON

echo ""
echo "Full results: $RESULTS_CSV"
echo "Outputs: $OUTPUT_DIR/"
