#!/bin/bash
# N5 frontdoor drafter retest harness.
#
# Default mode is no-inference: verify the retest prerequisites and emit a
# reproducible clean-window command package. Pass --execute only in a
# coordinated window with a llama.cpp worktree checked out at the required
# sequence-capacity fix.

set -euo pipefail

RESEARCH_ROOT="/mnt/raid0/llm/epyc-inference-research"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/mnt/raid0/llm/llama.cpp}"
LLAMA_SERVER="${LLAMA_SERVER:-${LLAMA_CPP_DIR}/build/bin/llama-server}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-a6c793fc6}"
SAFETY_COMMIT="${SAFETY_COMMIT:-53e9a6550}"
TARGET_MODEL="${TARGET_MODEL:-/mnt/raid0/llm/models/Qwen_Qwen3.6-35B-A3B-Q8_0.gguf}"
DRAFT_MODEL="${DRAFT_MODEL:-/mnt/raid0/llm/scratch/n5/Qwen3.5-0.8B-Q8_0.frontdoor-specials.gguf}"
COMPAT_CHECK="${COMPAT_CHECK:-${RESEARCH_ROOT}/scripts/utils/check_draft_compatibility.py}"
if [[ -z "${PYTHON_BIN:-}" && -x "${RESEARCH_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${RESEARCH_ROOT}/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${RESEARCH_ROOT}/data/specdec_frontdoor_alpha/n5_retest_$(date -u +%Y%m%dT%H%M%SZ)}"
PORT="${PORT:-19087}"
THREADS="${THREADS:-96}"
CONTEXT="${CONTEXT:-8192}"
UBATCH="${UBATCH:-8192}"
N_PREDICT="${N_PREDICT:-96}"
DRAFT_MAX="${DRAFT_MAX:-1}"
EXECUTE=0
STRICT=0
ALLOW_ACTIVE_AUTOPILOT=0
ALLOW_LIVE_LLAMA=0

usage() {
  cat <<'EOF'
Usage: n5_frontdoor_drafter_retest.sh [--strict] [--execute]

Default mode writes a no-inference preflight package under:
  data/specdec_frontdoor_alpha/n5_retest_<timestamp>/

Execution is intentionally fail-closed if AutoPilot, live llama-server
processes, port conflicts, commit mismatches, missing models, or missing
compatibility checks are detected.

Environment overrides:
  LLAMA_CPP_DIR, LLAMA_SERVER, EXPECTED_COMMIT, SAFETY_COMMIT
  TARGET_MODEL, DRAFT_MODEL, PYTHON_BIN, OUTPUT_DIR, PORT, THREADS
  CONTEXT, UBATCH, N_PREDICT, DRAFT_MAX

Flags:
  --strict                  Exit nonzero if any preflight item is blocked.
  --execute                 Launch the smoke after strict preflight passes.
  --allow-active-autopilot  Permit execution while AutoPilot is active.
  --allow-live-llama        Permit execution while llama-server processes live.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --strict) STRICT=1; shift ;;
    --execute) EXECUTE=1; STRICT=1; shift ;;
    --allow-active-autopilot) ALLOW_ACTIVE_AUTOPILOT=1; shift ;;
    --allow-live-llama) ALLOW_LIVE_LLAMA=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

mkdir -p "$OUTPUT_DIR"
PREFLIGHT_JSON="${OUTPUT_DIR}/preflight.json"
COMMANDS_SH="${OUTPUT_DIR}/commands.sh"
RESPONSE_JSON="${OUTPUT_DIR}/response.json"
SERVER_LOG="${OUTPUT_DIR}/llama-server.log"
COMPAT_LOG="${OUTPUT_DIR}/compatibility.log"
RESULTS_CSV="${OUTPUT_DIR}/n5_frontdoor_qwen35_aligned.csv"

status="ready"
blockers=()

add_blocker() {
  blockers+=("$1")
  status="blocked"
}

git_head() {
  git -C "$LLAMA_CPP_DIR" rev-parse --short HEAD 2>/dev/null || true
}

commit_timestamp() {
  git -C "$LLAMA_CPP_DIR" show -s --format=%ct "$EXPECTED_COMMIT" 2>/dev/null || echo 0
}

commit_present() {
  git -C "$LLAMA_CPP_DIR" rev-parse --verify --quiet "$1^{commit}" >/dev/null
}

is_ancestor() {
  git -C "$LLAMA_CPP_DIR" merge-base --is-ancestor "$1" "$2" >/dev/null 2>&1
}

port_in_use() {
  if command -v ss >/dev/null 2>&1; then
    ss -ltn "sport = :${PORT}" | grep -q ":${PORT}"
    return
  fi
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

active_autopilot() {
  pgrep -af "scripts/autopilot/autopilot.py start" >/dev/null 2>&1
}

live_llama_servers() {
  pgrep -a -x "llama-server" >/dev/null 2>&1
}

wait_for_server() {
  local elapsed=0
  while true; do
    if curl -fsS "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q '"status":"ok"'; then
      return 0
    fi
    sleep 2
    elapsed=$((elapsed + 2))
    if [[ "$elapsed" -ge 900 ]]; then
      return 1
    fi
  done
}

json_array() {
  python3 - "$@" <<'PY'
import json
import sys

print(json.dumps(sys.argv[1:]))
PY
}

server_cmd_json() {
  python3 - "$LLAMA_CPP_DIR" "$LLAMA_SERVER" "$TARGET_MODEL" "$DRAFT_MODEL" "$PORT" "$THREADS" "$CONTEXT" "$UBATCH" "$DRAFT_MAX" "$N_PREDICT" <<'PY'
import json
import shlex
import sys

(
    llama_cpp_dir,
    llama_server,
    target,
    draft,
    port,
    threads,
    context,
    ubatch,
    draft_max,
    _n_predict,
) = sys.argv[1:]
cmd = [
    "env",
    f"LD_LIBRARY_PATH={llama_cpp_dir}/build/bin",
    "numactl",
    "--interleave=all",
    llama_server,
    "-m",
    target,
    "-md",
    draft,
    "--draft-max",
    draft_max,
    "--draft-p-split",
    "0.05",
    "-t",
    threads,
    "-np",
    "1",
    "-c",
    context,
    "-ub",
    ubatch,
    "-ngl",
    "0",
    "--port",
    port,
    "--metrics",
    "--slots",
    "--jinja",
    "--reasoning",
    "auto",
    "-fa",
    "on",
    "-ctk",
    "q8_0",
    "-ctv",
    "q8_0",
]
print(json.dumps({"argv": cmd, "shell": shlex.join(cmd)}))
PY
}

run_compatibility_check() {
  "$PYTHON_BIN" "$COMPAT_CHECK" "$DRAFT_MODEL" "$TARGET_MODEL" >"$COMPAT_LOG" 2>&1
}

write_commands() {
  local server_shell
  server_shell=$(server_cmd_json | python3 -c 'import json,sys; print(json.load(sys.stdin)["shell"])')
  cat >"$COMMANDS_SH" <<EOF
#!/bin/bash
set -euo pipefail

# Generated by n5_frontdoor_drafter_retest.sh.
# Run only in a coordinated clean window after the strict preflight is ready.

export LLAMA_CPP_DIR=${LLAMA_CPP_DIR@Q}
export LLAMA_SERVER=${LLAMA_SERVER@Q}
export TARGET_MODEL=${TARGET_MODEL@Q}
export DRAFT_MODEL=${DRAFT_MODEL@Q}
export PYTHON_BIN=${PYTHON_BIN@Q}
export OUTPUT_DIR=${OUTPUT_DIR@Q}
export PORT=${PORT@Q}
export THREADS=${THREADS@Q}
export CONTEXT=${CONTEXT@Q}
export UBATCH=${UBATCH@Q}
export N_PREDICT=${N_PREDICT@Q}
export DRAFT_MAX=${DRAFT_MAX@Q}

${RESEARCH_ROOT@Q}/scripts/benchmark/n5_frontdoor_drafter_retest.sh --execute

# Server launch command used internally by --execute:
# $server_shell >${SERVER_LOG@Q} 2>&1
EOF
  chmod +x "$COMMANDS_SH"
}

if [[ ! -d "$LLAMA_CPP_DIR/.git" ]]; then
  add_blocker "llama.cpp git tree not found at ${LLAMA_CPP_DIR}"
else
  if ! commit_present "$EXPECTED_COMMIT"; then
    add_blocker "expected commit ${EXPECTED_COMMIT} is not present in ${LLAMA_CPP_DIR}"
  fi
  if ! commit_present "$SAFETY_COMMIT"; then
    add_blocker "safety commit ${SAFETY_COMMIT} is not present in ${LLAMA_CPP_DIR}"
  elif commit_present "$EXPECTED_COMMIT" && ! is_ancestor "$SAFETY_COMMIT" "$EXPECTED_COMMIT"; then
    add_blocker "safety commit ${SAFETY_COMMIT} is not an ancestor of expected commit ${EXPECTED_COMMIT}"
  fi
  current_head=$(git_head)
  if [[ "$current_head" != "$EXPECTED_COMMIT" ]]; then
    add_blocker "active llama.cpp HEAD is ${current_head:-unknown}, expected ${EXPECTED_COMMIT}; use an isolated worktree/build for the retest"
  fi
fi

if [[ ! -x "$LLAMA_SERVER" ]]; then
  add_blocker "llama-server binary not executable: ${LLAMA_SERVER}"
else
  built_at=$(stat -c %Y "$LLAMA_SERVER")
  expected_at=$(commit_timestamp)
  if [[ "$expected_at" -gt 0 && "$built_at" -lt "$expected_at" ]]; then
    add_blocker "llama-server binary predates expected commit ${EXPECTED_COMMIT}; rebuild before retest"
  fi
fi
command -v numactl >/dev/null 2>&1 || add_blocker "numactl is not available"

[[ -f "$TARGET_MODEL" ]] || add_blocker "target model missing: ${TARGET_MODEL}"
[[ -f "$DRAFT_MODEL" ]] || add_blocker "draft model missing: ${DRAFT_MODEL}"
[[ -f "$COMPAT_CHECK" ]] || add_blocker "compatibility checker missing: ${COMPAT_CHECK}"

if [[ -f "$TARGET_MODEL" && -f "$DRAFT_MODEL" && -f "$COMPAT_CHECK" ]]; then
  if ! "$PYTHON_BIN" -c 'import gguf' >/dev/null 2>&1; then
    {
      echo "ERROR: Python package 'gguf' is not installed for ${PYTHON_BIN}."
      echo "Install the benchmark extras or rerun with PYTHON_BIN pointing to a python that can import gguf."
      echo "Suggested repo-local setup: cd ${RESEARCH_ROOT} && uv sync --extra benchmark"
    } >"$COMPAT_LOG"
    add_blocker "compatibility checker dependency missing: ${PYTHON_BIN} cannot import gguf"
  elif ! run_compatibility_check; then
    add_blocker "draft/target compatibility check failed; see ${COMPAT_LOG}"
  fi
fi

if port_in_use; then
  add_blocker "port ${PORT} is already accepting connections"
fi
if [[ "$ALLOW_ACTIVE_AUTOPILOT" -ne 1 ]] && active_autopilot; then
  add_blocker "active AutoPilot process detected"
fi
if [[ "$ALLOW_LIVE_LLAMA" -ne 1 ]] && live_llama_servers; then
  add_blocker "live llama-server process detected"
fi

write_commands

server_cmd=$(server_cmd_json)
execution_mode="dry_run"
if [[ "$EXECUTE" -eq 1 ]]; then
  execution_mode="execute"
fi
python3 - "$PREFLIGHT_JSON" "$status" "$execution_mode" "$(json_array "${blockers[@]}")" "$LLAMA_CPP_DIR" "$(git_head)" "$EXPECTED_COMMIT" "$SAFETY_COMMIT" "$LLAMA_SERVER" "$TARGET_MODEL" "$DRAFT_MODEL" "$PYTHON_BIN" "$PORT" "$THREADS" "$CONTEXT" "$UBATCH" "$N_PREDICT" "$DRAFT_MAX" "$COMMANDS_SH" "$COMPAT_LOG" "$server_cmd" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    out,
    status,
    execution_mode,
    blockers_json,
    llama_cpp_dir,
    llama_head,
    expected_commit,
    safety_commit,
    llama_server,
    target,
    draft,
    python_bin,
    port,
    threads,
    context,
    ubatch,
    n_predict,
    draft_max,
    commands_sh,
    compat_log,
    server_cmd_json,
) = sys.argv[1:]

payload = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "status": status,
    "execution_mode": execution_mode,
    "blockers": json.loads(blockers_json),
    "purpose": "N5 qwen35-compatible frontdoor drafter alpha retest preflight",
    "acceptance_contract": "Only runs that reach draft/verify and emit draft_n/draft_n_accepted are alpha evidence.",
    "llama_cpp": {
        "dir": llama_cpp_dir,
        "head": llama_head or None,
        "expected_commit": expected_commit,
        "required_safety_commit": safety_commit,
    },
    "binary": {
        "llama_server": llama_server,
        "ld_library_path_prefix": str(Path(llama_cpp_dir) / "build" / "bin"),
    },
    "models": {"target": target, "draft": draft},
    "python": {"compatibility_python": python_bin},
    "runtime": {
        "port": int(port),
        "threads": int(threads),
        "context": int(context),
        "ubatch": int(ubatch),
        "n_predict": int(n_predict),
        "draft_max": int(draft_max),
    },
    "artifacts": {
        "commands_sh": commands_sh,
        "compatibility_log": compat_log,
    },
    "server_command": json.loads(server_cmd_json),
}
with open(out, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

echo "N5 frontdoor drafter retest preflight: ${status}"
echo "mode: ${execution_mode}"
echo "purpose: N5 qwen35-compatible frontdoor drafter alpha retest preflight"
echo "preflight: ${PREFLIGHT_JSON}"
echo "commands:  ${COMMANDS_SH}"
if [[ "${#blockers[@]}" -gt 0 ]]; then
  printf '  - %s\n' "${blockers[@]}"
fi
if [[ "$EXECUTE" -ne 1 ]]; then
  echo "Dry-run only. No inference was launched."
  echo "Review ${PREFLIGHT_JSON} and ${COMMANDS_SH} for the clean-window launch package."
  echo "Re-run with --strict --execute in a coordinated clean window to launch the smoke."
fi

if [[ "$STRICT" -eq 1 && "$status" != "ready" ]]; then
  exit 3
fi

if [[ "$EXECUTE" -ne 1 ]]; then
  exit 0
fi

echo "Launching N5 smoke on port ${PORT}..."
server_shell=$(server_cmd_json | python3 -c 'import json,sys; print(json.load(sys.stdin)["shell"])')
eval "$server_shell" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

if ! wait_for_server; then
  echo "ERROR: server did not become healthy; see ${SERVER_LOG}" >&2
  exit 4
fi

prompt="Summarize why tokenizer-aligned speculative decoding must report draft acceptance before it can gate production routing."
start_ms=$(date +%s%N | cut -b1-13)
curl -fsS --max-time 900 "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"n5-frontdoor-drafter\",
    \"messages\": [{\"role\": \"user\", \"content\": $(printf '%s' "$prompt" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')}],
    \"max_tokens\": ${N_PREDICT},
    \"temperature\": 0,
    \"stream\": false
  }" >"$RESPONSE_JSON"
end_ms=$(date +%s%N | cut -b1-13)
elapsed_ms=$((end_ms - start_ms))

python3 - "$RESPONSE_JSON" "$RESULTS_CSV" "$elapsed_ms" <<'PY'
import csv
import json
import sys

response_path, csv_path, elapsed_ms = sys.argv[1:]
with open(response_path, "r", encoding="utf-8") as handle:
    data = json.load(handle)
usage = data.get("usage", {})
timings = data.get("timings", {})
completion_tokens = int(usage.get("completion_tokens", 0) or 0)
draft_accepted = int(timings.get("draft_n_accepted", 0) or 0)
draft_total = int(timings.get("draft_n", 0) or 0)
elapsed = int(elapsed_ms)
tps = completion_tokens * 1000 / elapsed if elapsed > 0 else 0.0
acceptance = draft_accepted / draft_total if draft_total > 0 else 0.0
with open(csv_path, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "completion_tokens",
            "elapsed_ms",
            "tokens_per_sec",
            "draft_accepted",
            "draft_total",
            "acceptance_rate",
            "decision_grade",
        ],
    )
    writer.writeheader()
    writer.writerow(
        {
            "completion_tokens": completion_tokens,
            "elapsed_ms": elapsed,
            "tokens_per_sec": f"{tps:.4f}",
            "draft_accepted": draft_accepted,
            "draft_total": draft_total,
            "acceptance_rate": f"{acceptance:.6f}",
            "decision_grade": "true" if draft_total > 0 else "false",
        }
    )
if draft_total <= 0:
    raise SystemExit("smoke completed without draft tokens; not alpha evidence")
PY

echo "N5 smoke complete: ${RESULTS_CSV}"
