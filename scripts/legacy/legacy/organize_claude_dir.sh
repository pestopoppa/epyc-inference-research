#!/bin/bash
# organize_claude_dir.sh - Reorganize /mnt/raid0/llm/epyc-inference-research into clean structure
# Usage: bash organize_claude_dir.sh [--dry-run]

set -euo pipefail

CLAUDE_DIR="/mnt/raid0/llm/epyc-inference-research"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  echo "🔍 DRY RUN MODE - No files will be moved"
  echo ""
fi

cd "$CLAUDE_DIR"

echo "=============================================="
echo "Organizing $CLAUDE_DIR"
echo "=============================================="
echo ""

# Create target directory structure
DIRS=(
  "agents"        # Agent definitions
  "docs"          # Documentation and guides
  "scripts/utils" # Utility scripts
  "scripts/benchmark"# Benchmarking scripts
  "scripts/session" # Session management
  "reports"         # Research reports and findings
  "backups"         # Old backups
  "archive"         # Deprecated/old files
)

echo "Creating directory structure..."
for dir in "${DIRS[@]}"; do
  if [ $DRY_RUN -eq 0 ]; then
    mkdir -p "$dir"
  fi
  echo "  ✓ $dir"
done
echo ""

# Helper function to move files
move_file() {
  local src="$1"
  local dest="$2"

  if [ ! -e "$src" ]; then
    return
  fi

  if [ $DRY_RUN -eq 1 ]; then
    echo "  [DRY RUN] $src → $dest"
  else
    # Don't overwrite if destination exists
    if [ -e "$dest" ]; then
      echo "  ⚠️  SKIP: $src (destination exists)"
    else
      mv "$src" "$dest"
      echo "  ✓ $src → $dest"
    fi
  fi
}

echo "--- Organizing Agent Definitions ---"
move_file "sysadmin.md" "agents/"
move_file "build-engineer.md" "agents/"
move_file "benchmark-analyst.md" "agents/"
move_file "model-engineer.md" "agents/"
move_file "research-engineer.md" "agents/"
move_file "safety-reviewer.md" "agents/"
echo ""

echo "--- Organizing Documentation ---"
move_file "CLAUDE.md" "docs/"
move_file "SYSTEM_PROMPT_GUIDE.md" "docs/"
move_file "OPENING_PROMPT.md" "docs/"
move_file "dynamic_speculative_depth.md" "docs/"
move_file "research_report_template.md" "docs/"
move_file "track2_revised_approach_strategy.docx" "docs/"
echo ""

echo "--- Organizing Research Reports ---"
move_file "SPECULATIVE_DECODING_REPORT.md" "reports/"
move_file "speculative_decoding_research.md" "reports/"
move_file "speculative_decoding_results.md" "reports/"
move_file "RECOVERY_ACTION_PLAN.md" "reports/"
echo ""

echo "--- Organizing Utility Scripts ---"
move_file "agent_log.sh" "scripts/utils/"
move_file "agent_log_analyze.sh" "scripts/utils/"
move_file "health_check.sh" "scripts/utils/"
move_file "monitor_storage.sh" "scripts/utils/"
move_file "emergency_cleanup.sh" "scripts/utils/"
move_file "claude_safe_start.sh" "scripts/utils/"
move_file "system_audit.sh" "scripts/utils/"
echo ""

echo "--- Organizing Benchmark Scripts ---"
move_file "bench_zen5.sh" "scripts/benchmark/"
move_file "run_inference.sh" "scripts/benchmark/"
move_file "record_test.sh" "scripts/benchmark/"
echo ""

echo "--- Organizing Session Scripts ---"
move_file "session_init.sh" "scripts/session/"
echo ""

echo "--- Moving Backups ---"
move_file "backup-20251213-203226" "backups/"
move_file "backup-20251213-205449" "backups/"
echo ""

echo "--- Moving Temporary/Cache Directories ---"
# Only move if they're in the root of claude dir
if [ -d "tmp" ] && [ $DRY_RUN -eq 0 ]; then
  # Check if tmp has contents
  if [ "$(ls -A tmp 2>/dev/null)" ]; then
    echo "  ⚠️  SKIP: tmp/ (has contents - manually review)"
  else
    rmdir tmp
    echo "  ✓ Removed empty tmp/"
  fi
elif [ -d "tmp" ]; then
  echo "  [DRY RUN] Would check tmp/ for contents"
fi

if [ -d "cache" ] && [ $DRY_RUN -eq 0 ]; then
  if [ "$(ls -A cache 2>/dev/null)" ]; then
    echo "  ⚠️  SKIP: cache/ (has contents - manually review)"
  else
    rmdir cache
    echo "  ✓ Removed empty cache/"
  fi
elif [ -d "cache" ]; then
  echo "  [DRY RUN] Would check cache/ for contents"
fi
echo ""

echo "--- Archive Old State/Config ---"
# These might be from old Claude sessions
if [ -d "state" ]; then
  move_file "state" "archive/"
fi
if [ -d "config" ]; then
  move_file "config" "archive/"
fi
if [ -d "share" ]; then
  move_file "share" "archive/"
fi
if [ -d "logs" ]; then
  # Only if it's different from /mnt/raid0/llm/LOGS
  if [ $DRY_RUN -eq 1 ]; then
    echo "  [DRY RUN] Would check if logs/ should be archived"
  else
    echo "  ⚠️  MANUAL: Review logs/ - may want to merge with /mnt/raid0/llm/LOGS"
  fi
fi
echo ""

echo "=============================================="
echo "Organization Complete!"
echo "=============================================="
echo ""

if [ $DRY_RUN -eq 1 ]; then
  echo "This was a DRY RUN. To actually perform the reorganization:"
  echo "  bash organize_claude_dir.sh"
  echo ""
else
  echo "Final directory structure:"
  tree -L 2 -d "$CLAUDE_DIR" 2>/dev/null || find "$CLAUDE_DIR" -maxdepth 2 -type d | sort
  echo ""

  echo "📋 Post-Organization Tasks:"
  echo ""
  echo "1. Update symlinks/references:"
  echo "   - /mnt/raid0/llm/UTILS/agent_log.sh should point to scripts/utils/agent_log.sh"
  echo "   - Update any scripts that source agent_log.sh"
  echo ""
  echo "2. Review these directories manually:"
  echo "   - tmp/ - May have session data"
  echo "   - cache/ - May have cached models/data"
  echo "   - logs/ - Merge with /mnt/raid0/llm/LOGS if needed"
  echo "   - archive/ - Delete if truly obsolete"
  echo ""
  echo "3. Update CLAUDE.md paths to reflect new structure"
  echo ""
  echo "4. Create README.md in each directory explaining its purpose"
fi
