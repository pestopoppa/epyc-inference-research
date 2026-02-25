#!/bin/bash
# report_update_workflow.sh - Automated research report update pipeline
#
# Usage:
#   ./report_update_workflow.sh --benchmark RESULTS_CSV --section "Benchmark Results"
#   ./report_update_workflow.sh --track Track1 --status "5.9x speedup validated"
#   ./report_update_workflow.sh --summary "Run this after major milestones"
#
# This script:
# 1. Collects data from multiple sources
# 2. Validates data consistency
# 3. Invokes @research-writer to update report
# 4. Logs the update in agent audit
# 5. Creates a summary of changes

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source environment library for path variables
# shellcheck source=../lib/env.sh
source "${SCRIPT_DIR}/../lib/env.sh"

LOGS_DIR="${LOG_DIR}"
REPORT_FILE="${PROJECT_ROOT}/docs/reference/benchmarks/RESULTS.md"
AUDIT_LOG="${LOG_DIR}/agent_audit.log"
TEMP_DATA="/tmp/report_update_$$"

# Source logging if available
if [[ -f "${SCRIPT_DIR}/agent_log.sh" ]]; then
  source "${SCRIPT_DIR}/agent_log.sh"
else
  # Fallback
  agent_task_start() { echo "[TASK] $1"; }
  agent_task_end() { :; }
  agent_observe() { :; }
fi

# Ensure directories exist
mkdir -p "$LOGS_DIR" "$TEMP_DATA"

# ============================================
# OPTION 1: Benchmark Results Update
# ============================================

update_benchmark_results() {
  local csv_file="$1"

  agent_task_start "Update benchmark results" "Processing $csv_file"

  # Validate CSV exists and has data
  if [[ ! -f "$csv_file" ]]; then
    echo "ERROR: Benchmark file not found: $csv_file"
    agent_task_end "Update benchmark results" "failure"
    exit 1
  fi

  local line_count

  line_count=$(wc -l <"$csv_file")
  if [[ $line_count -lt 2 ]]; then
    echo "ERROR: Benchmark file has no data"
    agent_task_end "Update benchmark results" "failure"
    exit 1
  fi

  # Extract summary stats
  echo "Benchmark Summary:" >"$TEMP_DATA/benchmark_summary.txt"
  echo "File: $csv_file" >>"$TEMP_DATA/benchmark_summary.txt"
  echo "Lines: $line_count" >>"$TEMP_DATA/benchmark_summary.txt"
  echo "Date: $(date -Iseconds)" >>"$TEMP_DATA/benchmark_summary.txt"
  echo "" >>"$TEMP_DATA/benchmark_summary.txt"
  echo "Column Headers:" >>"$TEMP_DATA/benchmark_summary.txt"
  head -1 "$csv_file" >>"$TEMP_DATA/benchmark_summary.txt"
  echo "" >>"$TEMP_DATA/benchmark_summary.txt"
  echo "Sample Results (last 5 lines):" >>"$TEMP_DATA/benchmark_summary.txt"
  tail -5 "$csv_file" >>"$TEMP_DATA/benchmark_summary.txt"

  cat "$TEMP_DATA/benchmark_summary.txt"

  echo ""
  echo "=== NEXT STEP ==="
  echo "@research-writer please update research_report.md with these benchmark results:"
  echo ""
  cat "$TEMP_DATA/benchmark_summary.txt"
  echo ""
  echo "Include:"
  echo "  1. New entry in 'Benchmark Results' table"
  echo "  2. Analysis of speedup trend across K values"
  echo "  3. Acceptance rate observations"
  echo "  4. Optimal K recommendation for this model"
  echo "  5. Update 'Last Updated' timestamp"

  agent_task_end "Update benchmark results" "success"
}

# ============================================
# OPTION 2: Track Status Update
# ============================================

update_track_status() {
  local track="$1"
  local status="$2"
  local details="${3:-}"

  agent_task_start "Update track status" "$track: $status"

  echo "Track Update Request:"
  echo "  Track: $track"
  echo "  Status: $status"
  echo "  Details: ${details:-none}"
  echo "  Timestamp: $(date -Iseconds)"

  echo ""
  echo "=== NEXT STEP ==="
  echo "@research-writer update the status of $track in research_report.md:"
  echo ""
  echo "Track: $track"
  echo "New Status: $status"
  echo "Details: $details"
  echo ""
  echo "Update the corresponding Track section with:"
  echo "  1. Current status (e.g., Production, In Progress, Blocked)"
  echo "  2. Key metrics (speedup, acceptance rate if applicable)"
  echo "  3. Validation details (which models tested, command used)"
  echo "  4. Next milestone or blocker"
  echo "  5. Any relevant findings from testing"

  agent_task_end "Update track status" "success"
}

# ============================================
# OPTION 3: Full Report Summary
# ============================================

generate_full_summary() {
  agent_task_start "Generate full report summary" "Collecting all data sources"

  echo "=== FULL REPORT SUMMARY ==="
  echo ""

  # 1. Collect benchmark files
  echo "--- Benchmark Files ---"
  find "$LOGS_DIR" -name "zen5_benchmark_*.csv" -type f | while read -r f; do
    echo "File: $(basename "$f")"
    echo "  Size: $(wc -l <"$f") lines"
    echo "  Date: $(stat -c %y "$f" | cut -d' ' -f1-2)"
  done
  echo ""

  # 2. Tested models summary
  echo "--- Tested Models ---"
  if [[ -f "$LOGS_DIR/tested_models.json" ]]; then
    echo "File: tested_models.json"
    jq '.tested | length' "$LOGS_DIR/tested_models.json" | xargs echo "  Total tested:"
    jq '.tested[] | "\(.name): \(.result_tps) t/s (\(.method))"' "$LOGS_DIR/tested_models.json" | head -10
    echo "  [... more in tested_models.json]"
  else
    echo "  (No tested models file yet)"
  fi
  echo ""

  # 3. Agent activity summary
  echo "--- Agent Activity ---"
  if [[ -f "$AUDIT_LOG" ]]; then
    echo "Audit log: $AUDIT_LOG"
    echo "  Total entries: $(wc -l <"$AUDIT_LOG")"
    echo "  Sessions:"
    grep '"cat":"SESSION_START"' "$AUDIT_LOG" | jq -r '.ts' | tail -5
    echo "  Recent tasks:"
    grep '"cat":"TASK_START"' "$AUDIT_LOG" | tail -5 | jq -r '.msg'
  else
    echo "  (No audit log yet)"
  fi
  echo ""

  # 4. Report status
  echo "--- Report Status ---"
  if [[ -f "$REPORT_FILE" ]]; then
    echo "File: $REPORT_FILE"
    echo "  Size: $(wc -l <"$REPORT_FILE") lines"
    echo "  Sections:"
    grep "^## " "$REPORT_FILE" | head -10
    echo "  [... more sections]"
  else
    echo "  (Report not yet created)"
  fi
  echo ""

  echo "=== NEXT STEP ==="
  echo "@research-writer please generate a comprehensive research report update."
  echo "Use the summary above as a starting point. Include:"
  echo ""
  echo "1. Executive Summary (updated with latest results)"
  echo "2. System Configuration (unchanged unless upgraded)"
  echo "3. Tested Models (from tested_models.json)"
  echo "4. Benchmark Results (from CSV files)"
  echo "5. Track Status (from agent activity and manual input)"
  echo "6. Key Findings (synthesized from results)"
  echo "7. Future Work (based on current blockers/priorities)"
  echo "8. Literature References (add any new papers cited)"
  echo ""
  echo "Ensure consistency with CLAUDE.md track status (Track 1: [OK], Track 2: [OK], etc.)"

  agent_task_end "Generate full report summary" "success"
}

# ============================================
# OPTION 4: Validate Report
# ============================================

validate_report() {
  agent_task_start "Validate research report" "Checking consistency"

  if [[ ! -f "$REPORT_FILE" ]]; then
    echo "ERROR: Report file not found: $REPORT_FILE"
    agent_task_end "Validate research report" "failure"
    exit 1
  fi

  local errors=0

  echo "Validating research_report.md..."
  echo ""

  # Check 1: Required sections
  echo "Check 1: Required sections"
  local required_sections=(
    "Executive Summary"
    "System Configuration"
    "Tested Models"
    "Benchmark Results"
    "Key Findings"
    "Future Work"
    "Literature References"
  )

  for section in "${required_sections[@]}"; do
    if grep -q "^## $section" "$REPORT_FILE"; then
      echo "  [OK] $section"
    else
      echo "  [FAIL] $section (MISSING)"
      ((errors++))
    fi
  done
  echo ""

  # Check 2: Track status consistency
  echo "Check 2: Track status consistency"
  local track_regex="Track [0-9]+"
  if grep -q "$track_regex" "$REPORT_FILE"; then
    echo "  [OK] Track references found"
  else
    echo "  [FAIL] No track references (check if intentional)"
    ((errors++))
  fi
  echo ""

  # Check 3: Speedup values sanity
  echo "Check 3: Speedup values sanity"
  if grep -qE "[0-9]+\.[0-9]+x" "$REPORT_FILE"; then
    echo "  [OK] Speedup values present"
    grep -o "[0-9]\+\.[0-9]\+x" "$REPORT_FILE" | sort -u | sed 's/^/     /'
  else
    echo "  [FAIL] No speedup values found"
    ((errors++))
  fi
  echo ""

  # Check 4: Timestamp freshness
  echo "Check 4: Timestamp freshness"
  if grep -q "Last Updated:" "$REPORT_FILE"; then
    echo "  [OK] Last Updated field present"
    grep "Last Updated:" "$REPORT_FILE" | head -1 | sed 's/^/     /'
  else
    echo "  [FAIL] No Last Updated timestamp"
    ((errors++))
  fi
  echo ""

  if [[ $errors -eq 0 ]]; then
    echo "[OK] Report validation PASSED"
    agent_task_end "Validate research report" "success"
    return 0
  else
    echo "[FAIL] Report validation FAILED ($errors issues found)"
    echo ""
    echo "=== NEXT STEP ==="
    echo "@research-writer please fix the issues above in research_report.md"
    agent_task_end "Validate research report" "failure"
    return 1
  fi
}

# ============================================
# OPTION 5: Show Report Contents
# ============================================

show_report() {
  if [[ ! -f "$REPORT_FILE" ]]; then
    echo "Report not found: $REPORT_FILE"
    return 1
  fi

  echo "=== RESEARCH REPORT CONTENTS ==="
  echo ""
  head -100 "$REPORT_FILE"
  echo ""
  echo "[... truncated ...]"
  echo ""
  echo "Full report: $REPORT_FILE"
}

# ============================================
# Main
# ============================================

main() {
  local cmd="${1:-}"

  case "$cmd" in
    --benchmark)
      if [[ -z "${2:-}" ]]; then
        echo "Usage: $0 --benchmark CSV_FILE"
        exit 1
      fi
      update_benchmark_results "$2"
      ;;
    --track)
      if [[ -z "${2:-}" ]] || [[ -z "${3:-}" ]]; then
        echo "Usage: $0 --track TRACK_NAME STATUS [details]"
        exit 1
      fi
      update_track_status "$2" "$3" "${4:-}"
      ;;
    --summary)
      generate_full_summary
      ;;
    --validate)
      validate_report
      ;;
    --show)
      show_report
      ;;
    *)
      echo "Research Report Update Workflow"
      echo ""
      echo "Usage: $0 <command> [args]"
      echo ""
      echo "Commands:"
      echo "  --benchmark FILE.csv       Update benchmark results"
      echo "  --track TRACK STATUS [msg] Update track status"
      echo "  --summary                  Generate full summary for report update"
      echo "  --validate                 Validate report consistency"
      echo "  --show                     Show current report"
      echo ""
      echo "Examples:"
      echo "  $0 --benchmark /mnt/raid0/llm/LOGS/zen5_benchmark_20251215_143022.csv"
      echo "  $0 --track 'Track 1' '[OK] Production' '5.9x speedup validated'"
      echo "  $0 --summary"
      echo "  $0 --validate"
      exit 1
      ;;
  esac
}

main "$@"
