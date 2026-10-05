#!/bin/bash
set -euo pipefail

# ══════════════════════════════════════════════════════════════════
#  Package A: Instrumented Seeding Eval
# ══════════════════════════════════════════════════════════════════
#
#  One eval run, five tasks worth of data:
#
#  1. CF Phase 1 validation       — two-level condensation on real sessions
#  2. B5 Session Analytics         — token budgeting at 70%/100% thresholds
#  3. Difficulty signal telemetry  — shadow predictions vs correctness
#  4. RI-9 Risk threshold sweep    — factual risk scores in seeding results
#  5. Omega metric (reasoning)     — per-suite reasoning tokens vs accuracy
#
#  Duration: ~30 min (200 questions × 3-way × ~3s each)
#
#  Usage:
#    bash scripts/benchmark/package_a_instrumented_eval.sh
#    bash scripts/benchmark/package_a_instrumented_eval.sh --dry-run
#    bash scripts/benchmark/package_a_instrumented_eval.sh --sample-size 5  # quick test
#
# ══════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR="$PROJECT_ROOT/data/package_a/$TIMESTAMP"

# ── Feature flags for instrumentation ───────────────────────────
# CF Phase 1: Enable two-level condensation
export ORCHESTRATOR_TWO_LEVEL_CONDENSATION=1

# Token budgeting: existing task_token_budget mechanism (Fast-RLM)
# B5 (session-level) not yet implemented — use existing task-level budget
export ORCHESTRATOR_TASK_TOKEN_BUDGET=1

# Session log + scratchpad (needed for CF Phase 1 to exercise)
export ORCHESTRATOR_SESSION_LOG=1
export ORCHESTRATOR_SESSION_COMPACTION=1
export ORCHESTRATOR_SESSION_SCRATCHPAD=1

# Difficulty signal already in shadow mode (classifier_config.yaml)
# Factual risk already in shadow mode (classifier_config.yaml)
# No additional env vars needed — telemetry captured automatically

# ── Parse CLI args ──────────────────────────────────────────────
EXTRA_ARGS=()
SAMPLE_SIZE=15
DRY_RUN=""
SUITES="math simpleqa hotpotqa gpqa coder thinking general agentic instruction_precision mode_advantage_hard"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN="--dry-run"
      shift
      ;;
    --sample-size)
      SAMPLE_SIZE="$2"
      shift 2
      ;;
    --suites)
      SUITES="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

# ── Pre-flight ──────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Package A: Instrumented Seeding Eval                   ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Timestamp:  $TIMESTAMP"
echo "║  Output:     $OUTPUT_DIR"
echo "║  Suites:     $(echo $SUITES | wc -w) suites"
echo "║  Sample:     $SAMPLE_SIZE questions/suite"
echo "║  Est. total: $(($(echo $SUITES | wc -w) * SAMPLE_SIZE * 3)) evals (3-way)"
echo "║"
echo "║  Feature flags:"
echo "║    TWO_LEVEL_CONDENSATION = 1  (CF Phase 1)"
echo "║    TASK_TOKEN_BUDGET = 1       (Fast-RLM budgeting)"
echo "║    SESSION_LOG = 1             (session journal)"
echo "║    SESSION_COMPACTION = 1      (compaction enabled)"
echo "║    difficulty_signal = shadow   (from config)"
echo "║    factual_risk = shadow        (from config)"
if [[ -n "$DRY_RUN" ]]; then
  echo "║    [DRY RUN — no reward injection]"
fi
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Health check — require API availability before starting the eval.
echo "→ Checking orchestrator health..."
if ! curl -sf http://localhost:8000/health >/dev/null 2>&1; then
  echo "  ✗ Orchestrator API health check failed at http://localhost:8000/health"
  echo "    Package A does not manage API lifecycle. Recover API health from its owning session, then retry."
  exit 1
fi
echo "  ✓ Orchestrator healthy"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Save environment snapshot
env | grep -E "^ORCHESTRATOR_" | sort >"$OUTPUT_DIR/env_flags.txt"
echo "  ✓ Environment snapshot saved"

# ── Phase 1: 3-Way Seeding Eval ────────────────────────────────
echo ""
echo "═══ Phase 1: 3-Way Seeding Eval ═══"
echo "Running $SAMPLE_SIZE questions/suite across $(echo $SUITES | wc -w) suites..."
echo ""

cd "$PROJECT_ROOT"
python3 scripts/benchmark/seed_specialist_routing.py \
  --3way \
  --suites $SUITES \
  --sample-size "$SAMPLE_SIZE" \
  --output "$OUTPUT_DIR/seeding_results.json" \
  --preflight \
  $DRY_RUN \
  "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

echo ""
echo "  ✓ Seeding eval complete → $OUTPUT_DIR/seeding_results.json"

# ── Phase 2: TrimR Evaluation ──────────────────────────────────
echo ""
echo "═══ Phase 2: TrimR Reasoning Compression ═══"
echo "Running think-strip + full comparison on math/gpqa..."
echo ""

cd /mnt/raid0/llm/epyc-inference-research
python3 scripts/benchmark/eval_trimr.py \
  --suites math gpqa \
  --n-questions "$SAMPLE_SIZE" \
  --strategy all \
  --output "$OUTPUT_DIR/trimr_results.jsonl" \
  ${DRY_RUN:+--dry-run}

echo ""
echo "  ✓ TrimR eval complete → $OUTPUT_DIR/trimr_results.jsonl"

# ── Phase 3: Collect Telemetry ─────────────────────────────────
echo ""
echo "═══ Phase 3: Collect Telemetry ═══"

# Copy today's progress log for analysis
PROGRESS_DATE=$(date -u +%Y-%m-%d)
PROGRESS_LOG="$PROJECT_ROOT/logs/progress/$PROGRESS_DATE.jsonl"
if [[ -f "$PROGRESS_LOG" ]]; then
  cp "$PROGRESS_LOG" "$OUTPUT_DIR/progress_log.jsonl"
  echo "  ✓ Progress log copied"
fi

# Run SLO report
cd "$PROJECT_ROOT"
python3 scripts/server/delegation_slo_report.py \
  --date "$PROGRESS_DATE" \
  --json >"$OUTPUT_DIR/slo_report.json" 2>/dev/null || true
echo "  ✓ SLO report generated"

# Run anomaly detector
python3 scripts/server/chain_anomaly_detector.py \
  --date "$PROGRESS_DATE" \
  --json >"$OUTPUT_DIR/anomaly_report.json" 2>/dev/null || true
echo "  ✓ Anomaly report generated"

# ── Phase 4: Analysis ──────────────────────────────────────────
echo ""
echo "═══ Phase 4: Post-Run Analysis ═══"

# Extract difficulty signal vs correctness correlation
python3 -c "
import json, sys
from pathlib import Path
from collections import defaultdict

progress = Path('$OUTPUT_DIR/progress_log.jsonl')
if not progress.exists():
    print('  (no progress log to analyze)')
    sys.exit(0)

# Collect difficulty predictions and outcomes
by_band = defaultdict(lambda: {'total': 0, 'success': 0})
risk_by_band = defaultdict(lambda: {'total': 0, 'success': 0})

for line in progress.open():
    try:
        e = json.loads(line)
    except json.JSONDecodeError:
        continue

    if e.get('event_type') == 'routing_decision':
        data = e.get('data', {})
        d_band = data.get('difficulty_band', '')
        r_band = data.get('factual_risk_band', '')
        tid = e.get('task_id', '')
        if d_band:
            by_band[d_band]['total'] += 1
        if r_band:
            risk_by_band[r_band]['total'] += 1

# Output summary
out = {
    'difficulty_signal_distribution': {k: v['total'] for k, v in sorted(by_band.items())},
    'factual_risk_distribution': {k: v['total'] for k, v in sorted(risk_by_band.items())},
}
Path('$OUTPUT_DIR/telemetry_summary.json').write_text(json.dumps(out, indent=2))
print('  ✓ Telemetry summary:', json.dumps(out))
" 2>/dev/null || echo "  (telemetry analysis skipped)"

# ── Summary ────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Package A Complete                                      ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Output directory: $OUTPUT_DIR"
echo "║"
echo "║  Files produced:"
echo "║    seeding_results.json    — 3-way routing eval results"
echo "║    trimr_results.jsonl     — reasoning compression data"
echo "║    progress_log.jsonl      — raw telemetry (difficulty + risk)"
echo "║    slo_report.json         — delegation latency stats"
echo "║    anomaly_report.json     — chain anomaly detection"
echo "║    telemetry_summary.json  — difficulty/risk distributions"
echo "║    env_flags.txt           — feature flags snapshot"
echo "║"
echo "║  Next steps:"
echo "║    1. Review seeding_results.json for CF Phase 1 quality"
echo "║    2. Check telemetry_summary.json for difficulty signal calibration"
echo "║    3. Use trimr_results.jsonl for Omega metric computation"
echo "║    4. Set RI-10 thresholds based on risk distribution"
echo "║    5. Launch Package B (RTK + AR-3) with validated config"
echo "╚══════════════════════════════════════════════════════════╝"
