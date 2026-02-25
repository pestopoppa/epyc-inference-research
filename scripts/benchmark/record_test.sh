#!/bin/bash
# record_test.sh — Record test results and update research report
# Usage: ./record_test.sh --model PATH --method METHOD --result "t/s" [--notes "..."]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source environment library for path variables
# shellcheck source=../lib/env.sh
source "${SCRIPT_DIR}/../lib/env.sh"

source "${PROJECT_ROOT}/scripts/utils/agent_log.sh" 2>/dev/null || true

LOGS_DIR="${LOG_DIR}"
TESTED_FILE="$LOGS_DIR/tested_models.json"
RESEARCH_REPORT="$LOGS_DIR/research_report.md"
TEMPLATE="$SCRIPT_DIR/research_report_template.md"

# Parse arguments
MODEL_PATH=""
METHOD=""
RESULT=""
NOTES=""
ACCEPTANCE=""
CONFIG=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL_PATH="$2"
      shift 2
      ;;
    --method)
      METHOD="$2"
      shift 2
      ;;
    --result)
      RESULT="$2"
      shift 2
      ;;
    --notes)
      NOTES="$2"
      shift 2
      ;;
    --acceptance)
      ACCEPTANCE="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [ -z "$MODEL_PATH" ] || [ -z "$METHOD" ] || [ -z "$RESULT" ]; then
  echo "Usage: $0 --model PATH --method METHOD --result 't/s' [--notes '...'] [--acceptance '%'] [--config '...']"
  exit 1
fi

TIMESTAMP=$(date -Iseconds)
DATE=$(date +%Y-%m-%d)
MODEL_NAME=$(basename "$MODEL_PATH")

agent_task_start "Record test result" "Model: $MODEL_NAME, Result: $RESULT t/s"

# ============================================
# 1. UPDATE TESTED MODELS JSON
# ============================================

# Initialize if doesn't exist
if [ ! -f "$TESTED_FILE" ]; then
  echo '{"tested": []}' >"$TESTED_FILE"
fi

# Add new test record
TEMP_FILE=$(mktemp)
jq --arg path "$MODEL_PATH" \
  --arg name "$MODEL_NAME" \
  --arg method "$METHOD" \
  --arg result "$RESULT" \
  --arg date "$DATE" \
  --arg ts "$TIMESTAMP" \
  --arg notes "$NOTES" \
  --arg acceptance "$ACCEPTANCE" \
  --arg config "$CONFIG" \
  '.tested += [{
     "path": $path,
     "name": $name,
     "method": $method,
     "result_tps": $result,
     "date": $date,
     "timestamp": $ts,
     "notes": $notes,
     "acceptance": $acceptance,
     "config": $config
   }]' "$TESTED_FILE" >"$TEMP_FILE" && mv "$TEMP_FILE" "$TESTED_FILE"

echo "✓ Recorded test in $TESTED_FILE"
agent_observe "tested_model" "$MODEL_PATH"
agent_observe "test_result" "$RESULT t/s ($METHOD)"

# ============================================
# 2. UPDATE RESEARCH REPORT
# ============================================

# Initialize report from template if doesn't exist
if [ ! -f "$RESEARCH_REPORT" ]; then
  if [ -f "$TEMPLATE" ]; then
    cp "$TEMPLATE" "$RESEARCH_REPORT"
    echo "✓ Initialized research report from template"
  else
    echo "WARNING: No template found, creating minimal report"
    cat >"$RESEARCH_REPORT" <<'EOF'
# LLM Inference Optimization Research Report

## Executive Summary
Research in progress.

## Tested Models

| Model | Format | Method | t/s | Acceptance | Date | Notes |
|-------|--------|--------|-----|------------|------|-------|

## Findings
(To be updated)
EOF
  fi
fi

# Update timestamp in report
sed -i "s/\*\*Last Updated:\*\* .*/\*\*Last Updated:\*\* $TIMESTAMP/" "$RESEARCH_REPORT"

# Determine format from path
if [[ "$MODEL_PATH" == *.gguf ]]; then
  FORMAT="GGUF"
else
  FORMAT="HF"
fi

# Add result to Tested Models table
# Find the table and append row
NEW_ROW="| $MODEL_NAME | $FORMAT | $METHOD | $RESULT | ${ACCEPTANCE:-N/A} | $DATE | ${NOTES:-} |"

# Insert after table header (look for the header row pattern)
if grep -q "^| Model | Format |" "$RESEARCH_REPORT"; then
  # Find line number of divider after header
  LINE_NUM=$(grep -n "^|----" "$RESEARCH_REPORT" | head -1 | cut -d: -f1)
  if [ -n "$LINE_NUM" ]; then
    sed -i "${LINE_NUM}a\\${NEW_ROW}" "$RESEARCH_REPORT"
    echo "✓ Added result to research report table"
  fi
else
  echo "WARNING: Could not find results table in research report"
fi

# ============================================
# 3. LOG TO AUDIT
# ============================================

agent_observe "research_report_updated" "$RESEARCH_REPORT"
agent_task_end "Record test result" "success"

echo ""
echo "Test recorded:"
echo "  Model: $MODEL_NAME"
echo "  Method: $METHOD"
echo "  Result: $RESULT t/s"
echo "  Acceptance: ${ACCEPTANCE:-N/A}"
echo "  Notes: ${NOTES:-None}"
echo ""
echo "Files updated:"
echo "  - $TESTED_FILE"
echo "  - $RESEARCH_REPORT"
