#!/bin/bash
# Long Context Model Quality Rubric Test Script
# Tests information retrieval and synthesis across long contexts
#
# Single Config Mode: Pass config as 4th param
# Multi Config Mode: Omit 4th param, uses shared lib to determine configs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Configuration
MODEL="${1:-}"
MODEL_NAME="${2:-unknown}"
MODEL_ARCH="${3:-dense}"
CONFIG_PARAM="${4:-}"
OUTPUT_DIR="/mnt/raid0/llm/tmp/long_context_rubric_results"
LLAMA_COMPLETION="/mnt/raid0/llm/llama.cpp/build/bin/llama-completion"
TIMEOUT=300 # Long context needs more time

# Source shared libraries
if [[ -f "$SCRIPT_DIR/lib/optimization_configs.sh" ]]; then
  source "$SCRIPT_DIR/lib/optimization_configs.sh"
fi

if [[ -z "$MODEL" ]]; then
  echo "Usage: $0 <model.gguf> <model_name> [arch] [config]"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Determine configurations
declare -a CONFIGS
if [[ -n "$CONFIG_PARAM" ]]; then
  CONFIGS=("$CONFIG_PARAM")
  echo "[SINGLE CONFIG MODE] Running only: $CONFIG_PARAM"
else
  if type setup_configs &>/dev/null; then
    setup_configs "$MODEL_NAME" "$MODEL_ARCH"
  else
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
echo "Long Context Model Quality Rubric"
echo "Model: $MODEL_NAME"
echo "Architecture: $MODEL_ARCH"
echo "Date: $(date)"
echo "=============================================="

# Generate filler content (technical documentation style)
generate_filler() {
  local tokens="$1"
  local lines=$((tokens / 10)) # ~10 tokens per line

  for i in $(seq 1 $lines); do
    case $((i % 5)) in
      0) echo "The system processes requests through the main handler which validates input parameters." ;;
      1) echo "Configuration options include timeout settings, retry policies, and connection pooling." ;;
      2) echo "Error handling follows the standard pattern with graceful degradation enabled." ;;
      3) echo "Logging is configured at DEBUG level for development and INFO for production." ;;
      4) echo "Performance metrics are collected every 30 seconds and aggregated hourly." ;;
    esac
  done
}

# Generate context with needle
generate_context_with_needle() {
  local total_tokens="$1"
  local needle="$2"
  local position="$3" # early, middle, deep, very_deep

  local needle_line
  case "$position" in
    early) needle_line=$((total_tokens / 10 / 10)) ;;
    middle) needle_line=$((total_tokens / 10 / 2)) ;;
    deep) needle_line=$((total_tokens * 75 / 100 / 10)) ;;
    very_deep) needle_line=$((total_tokens * 85 / 100 / 10)) ;;
    *) needle_line=$((total_tokens / 10 / 2)) ;;
  esac

  local total_lines

  total_lines=$((total_tokens / 10))

  for i in $(seq 1 $total_lines); do
    if [[ $i -eq $needle_line ]]; then
      echo "$needle"
    else
      case $((i % 7)) in
        0) echo "Section $((i / 50 + 1)): The application layer handles all business logic processing." ;;
        1) echo "Database connections are managed through a connection pool with max size 100." ;;
        2) echo "API endpoints follow RESTful conventions with JSON request/response bodies." ;;
        3) echo "Authentication uses JWT tokens with 24-hour expiration and refresh capability." ;;
        4) echo "Caching is implemented at multiple layers including CDN, application, and database." ;;
        5) echo "Monitoring alerts are configured for error rates exceeding 1% threshold." ;;
        6) echo "Deployment follows blue-green strategy with automated rollback on failures." ;;
      esac
    fi
  done
}

# Generate meeting notes content
generate_meeting_notes() {
  cat <<'MEETING_EOF'
# Team Meeting Notes - Q4 Planning
Date: 2024-03-15
Attendees: Alice, Bob, Carol, Dave, Eve

## Agenda
1. Q3 Review
2. Q4 Goals
3. Resource Allocation
4. Action Items

## Discussion

### Q3 Review
Alice presented the Q3 metrics. Revenue was up 15% compared to Q2. Customer satisfaction scores improved from 4.2 to 4.5. However, there were concerns about the deployment delays.

Bob noted that the infrastructure team faced challenges with the cloud migration. Three major incidents occurred in September, each lasting more than 4 hours.

### Q4 Goals
The team agreed on the following goals:

**Decision 1**: Migrate remaining services to Kubernetes by end of November.

**Decision 2**: Implement automated testing pipeline with 80% coverage target.

**Decision 3**: Launch the new dashboard feature by December 15th.

Carol raised concerns about timeline feasibility. After discussion:

**Decision 4**: Hire two additional engineers to support the migration effort.

### Resource Allocation
Dave will lead the Kubernetes migration. Eve will own the testing pipeline. Alice will coordinate the dashboard launch.

### Action Items
1. **Bob**: Create detailed migration plan by March 20
2. **Carol**: Review and approve cloud budget by March 18
3. **Dave**: Set up Kubernetes staging environment by March 25
4. **Eve**: Research testing frameworks and present options by March 22
5. **Alice**: Finalize dashboard requirements with stakeholders by March 19

## Next Meeting
March 22, 2024 at 2:00 PM

---
Additional notes from sidebar conversations...

The team discussed potential risks including vendor lock-in and skill gaps. Training budget was approved for Kubernetes certification for 5 team members.

Performance benchmarks from Q3 showed API latency averaging 145ms, which is within acceptable range but could be improved.

Security audit scheduled for April will require documentation updates from each team lead.
MEETING_EOF
}

# Generate Python project content
generate_python_project() {
  cat <<'PROJECT_EOF'
# ===== FILE: config.py =====
"""Configuration management module."""

import os
import json
from typing import Dict, Any

DEFAULT_CONFIG = {
    "database": {
        "host": "localhost",
        "port": 5432,
        "name": "myapp"
    },
    "cache": {
        "enabled": True,
        "ttl": 3600
    },
    "logging": {
        "level": "INFO"
    }
}

def load_config(path: str = None) -> Dict[str, Any]:
    """Load configuration from file or use defaults."""
    if path and os.path.exists(path):
        with open(path, 'r') as f:
            user_config = json.load(f)
            # BUG: This doesn't deep merge, overwrites nested dicts entirely
            return {**DEFAULT_CONFIG, **user_config}
    return DEFAULT_CONFIG.copy()

def get_database_url(config: Dict[str, Any]) -> str:
    """Build database URL from config."""
    db = config["database"]
    return f"postgresql://{db['host']}:{db['port']}/{db['name']}"

# ===== FILE: utils.py =====
"""Utility functions for the application."""

import re
import hashlib
from typing import List, Optional

def parse_config_value(value: str) -> any:
    """Parse a config value string into appropriate type."""
    if value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value

def validate_email(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def hash_password(password: str, salt: str = "") -> str:
    """Hash a password with optional salt."""
    combined = password + salt
    return hashlib.sha256(combined.encode()).hexdigest()

# ===== FILE: database.py =====
"""Database connection and query management."""

from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

class DatabaseConnection:
    def __init__(self, url: str):
        self.url = url
        self.connected = False
        self._connection = None

    def connect(self) -> bool:
        """Establish database connection."""
        try:
            # Simulated connection
            self._connection = {"url": self.url}
            self.connected = True
            logger.info(f"Connected to database: {self.url}")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def query(self, sql: str, params: tuple = None) -> List[Dict]:
        """Execute a query and return results."""
        if not self.connected:
            raise RuntimeError("Not connected to database")
        logger.debug(f"Executing query: {sql}")
        # Simulated query execution
        return []

# ===== FILE: main.py =====
"""Main application entry point."""

import sys
import logging
from config import load_config, get_database_url
from database import DatabaseConnection
from utils import validate_email, hash_password

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_app(config_path: str = None):
    """Initialize the application with configuration."""
    logger.info("Starting application initialization")

    # Load configuration
    config = load_config(config_path)
    logger.info(f"Loaded config with cache TTL: {config['cache']['ttl']}")

    # Connect to database
    db_url = get_database_url(config)
    db = DatabaseConnection(db_url)

    if not db.connect():
        logger.error("Failed to connect to database")
        sys.exit(1)

    return config, db

def main():
    """Main entry point."""
    config, db = initialize_app()
    logger.info("Application started successfully")

    # Example usage
    if validate_email("test@example.com"):
        hashed = hash_password("secret123")
        logger.info(f"Password hash: {hashed[:16]}...")

if __name__ == "__main__":
    main()
PROJECT_EOF
}

# Run test function
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

  echo "$prompt" >"/mnt/raid0/llm/tmp/long_context_prompt.txt"

  timeout "$TIMEOUT" OMP_NUM_THREADS=1 numactl --interleave=all \
    "$LLAMA_COMPLETION" \
    -m "$MODEL" \
    -t 96 -n 1024 --temp 0.2 \
    $moe_override \
    -f "/mnt/raid0/llm/tmp/long_context_prompt.txt" \
    >"$output_file" 2>&1 || true

  local speed

  speed=$(grep "eval time" "$output_file" | grep -oP '\d+\.\d+ tokens per second' | tail -1 || echo "N/A")
  echo "Speed: $speed"

  echo "--- Answer ---"
  grep -v "^llama\|^load\|^print_info\|^common\|^sampler\|^generate\|^system_info\|^main:" "$output_file" | tail -20 | head -15
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

  # T1: Short Context (4K)
  echo ""
  echo "========== TIER 1 (4K Context) =========="

  # T1-Q1: Information Retrieval
  CONTEXT=$(generate_context_with_needle 4000 "The maximum concurrent connections allowed is 2847." "middle")
  run_test "t1_q1_retrieval" "$CONTEXT

Based on the documentation above, what is the exact maximum number of concurrent connections allowed?

Answer with just the number." "$CONFIG"

  # T1-Q2: Summary
  CONTEXT=$(generate_meeting_notes)
  run_test "t1_q2_summary" "$CONTEXT

Based on the meeting notes above:
1. List all decisions made
2. List all action items with their assignees

Be comprehensive." "$CONFIG"

  # T2: Medium Context (8-16K)
  echo ""
  echo "========== TIER 2 (8-16K Context) =========="

  # T2-Q1: Needle in Haystack
  CONTEXT=$(generate_context_with_needle 12000 "# CRITICAL_CONFIG: api_key = 'sk-proj-7x9mK2nP4qR8sT1uV3wY5zA'" "deep")
  run_test "t2_q1_needle" "$CONTEXT

Review the content above and find any hardcoded API keys, secrets, or credentials.
List exactly what you find with the exact values." "$CONFIG"

  # T2-Q2: Multi-File Analysis
  CONTEXT=$(generate_python_project)
  run_test "t2_q2_multi_file" "$CONTEXT

Analyze the Python project above:
1. Trace the data flow from config loading in config.py through to main.py
2. Identify any bugs or issues in the code

Be specific about file names and function names." "$CONFIG"

  # T3: Long Context (32K+) - Only if model likely supports it
  echo ""
  echo "========== TIER 3 (32K+ Context) =========="
  echo "Note: T3 tests require large context window support"

  # T3-Q1: Deep Needle
  CONTEXT=$(generate_context_with_needle 35000 "[2024-03-15 03:47:22] CRITICAL: Server node-7 experienced fatal memory corruption" "very_deep")
  run_test "t3_q1_deep_needle" "$CONTEXT

Review the logs above. Find the CRITICAL error.
What server was affected and at what exact time?" "$CONFIG"

  # Summary
  echo ""
  echo "=============================================="
  echo "CONFIG $CONFIG COMPLETE"
  echo "=============================================="

  max_speed=0 q_count=0
  for f in "$OUTPUT_DIR"/${MODEL_NAME}_${CONFIG}_*.txt; do
    if [[ -f "$f" ]]; then
      ((q_count++)) || true
      speed=$(grep "eval time" "$f" 2>/dev/null | grep -oP '\d+\.\d+(?= tokens per second)' | tail -1 || echo "0")
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
  printf "%-10s %-20s %-8s %2d %8s %-30s\n" "longctx" "${MODEL_NAME:0:20}" "$CONFIG" "$q_count" "${max_speed}" "$discovery_info" >>/mnt/raid0/llm/tmp/benchmark_completions.log
done

echo ""
echo "=============================================="
echo "LONG CONTEXT RUBRIC COMPLETE"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
