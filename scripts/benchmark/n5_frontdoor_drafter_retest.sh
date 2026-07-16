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
LLAMA_LIB_DIR="${LLAMA_LIB_DIR:-$(dirname "$LLAMA_SERVER")}"
PRODUCTION_LLAMA_CPP_DIR="${PRODUCTION_LLAMA_CPP_DIR:-/mnt/raid0/llm/llama.cpp}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-a6c793fc6}"
SAFETY_COMMIT="${SAFETY_COMMIT:-53e9a6550}"
SAFETY_COMMIT_MODE="${SAFETY_COMMIT_MODE:-ancestor}"
SAFETY_AUDIT_REF="${SAFETY_AUDIT_REF:-}"
TARGET_MODEL="${TARGET_MODEL:-/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf}"
DRAFT_MODEL="${DRAFT_MODEL:-/mnt/raid0/llm/scratch/n5/Qwen3.5-0.8B-Q8_0.frontdoor-mtp-specials.gguf}"
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
  LLAMA_CPP_DIR, LLAMA_SERVER, LLAMA_LIB_DIR, EXPECTED_COMMIT, SAFETY_COMMIT
  SAFETY_COMMIT_MODE=ancestor|semantic_audit, SAFETY_AUDIT_REF
  TARGET_MODEL, DRAFT_MODEL, PYTHON_BIN, OUTPUT_DIR, PORT, THREADS
  CONTEXT, UBATCH, N_PREDICT, DRAFT_MAX
  PRODUCTION_LLAMA_CPP_DIR (guard only; production trees are blocked)

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
RESULTS_JSONL="${OUTPUT_DIR}/n5_frontdoor_qwen35_aligned.jsonl"
SUMMARY_JSON="${OUTPUT_DIR}/summary.json"
N_PROMPTS="${N_PROMPTS:-8}"
MIN_DRAFT_RATIO="${MIN_DRAFT_RATIO:-0.25}"
REQUIRED_SPEC_TOKENS=(
  "--spec-type"
  "draft-tree"
  "draft-mtp"
  "--spec-draft-n-max"
  "--spec-draft-p-split"
  "-md"
)
missing_spec_tokens=()

status="ready"
blockers=()

add_blocker() {
  blockers+=("$1")
  status="blocked"
}

git_head() {
  git -C "$LLAMA_CPP_DIR" rev-parse --short HEAD 2>/dev/null || true
}

git_dir_present() {
  git -C "$LLAMA_CPP_DIR" rev-parse --git-dir >/dev/null 2>&1
}

binary_version() {
  LD_LIBRARY_PATH="${LLAMA_LIB_DIR}:${LD_LIBRARY_PATH:-}" "$LLAMA_SERVER" --version 2>&1 | head -1 || true
}

binary_help() {
  LD_LIBRARY_PATH="${LLAMA_LIB_DIR}:${LD_LIBRARY_PATH:-}" "$LLAMA_SERVER" --help 2>&1 || true
}

check_spec_flag_surface() {
  local help_text="$1"
  local token

  missing_spec_tokens=()
  for token in "${REQUIRED_SPEC_TOKENS[@]}"; do
    if [[ "$help_text" != *"$token"* ]]; then
      missing_spec_tokens+=("$token")
    fi
  done

  [[ "${#missing_spec_tokens[@]}" -eq 0 ]]
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

wait_for_port_release() {
  local timeout="${1:-60}"
  local elapsed=0
  while port_in_use; do
    sleep 1
    elapsed=$((elapsed + 1))
    if [[ "$elapsed" -ge "$timeout" ]]; then
      return 1
    fi
  done
}

port_listener_pids() {
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -tiTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null || true
  fi
}

kill_port_listeners() {
  local signal="$1"
  local pid
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    [[ "$pid" != "$$" ]] || continue
    kill "-${signal}" "$pid" 2>/dev/null || true
  done < <(port_listener_pids)
}

active_autopilot() {
  pgrep -af "scripts/autopilot/autopilot.py start" >/dev/null 2>&1
}

live_llama_servers() {
  pgrep -a -x "llama-server" >/dev/null 2>&1
}

wait_for_server() {
  local pid="${1:-}"
  local elapsed=0
  while true; do
    if curl -fsS "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q '"status":"ok"'; then
      return 0
    fi
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      return 1
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
  local arm="${1:-n5_spec_on}"
  python3 - "$arm" "$LLAMA_CPP_DIR" "$LLAMA_LIB_DIR" "$LLAMA_SERVER" "$TARGET_MODEL" "$DRAFT_MODEL" "$PORT" "$THREADS" "$CONTEXT" "$UBATCH" "$DRAFT_MAX" "$N_PREDICT" <<'PY'
import json
import shlex
import sys

(
    arm,
    llama_cpp_dir,
    llama_lib_dir,
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
    f"LD_LIBRARY_PATH={llama_lib_dir}",
    "numactl",
    "--interleave=all",
    llama_server,
    "-m",
    target,
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
if arm == "n5_spec_on":
    cmd.extend([
        "-md",
        draft,
        "--spec-type",
        "draft-tree",
        "--spec-draft-n-max",
        draft_max,
        "--spec-draft-p-split",
        "0.05",
    ])
elif arm == "positive_mtp":
    cmd.extend([
        "--spec-type",
        "draft-mtp",
        "--spec-draft-n-max",
        draft_max,
    ])
elif arm == "spec_off":
    cmd.extend(["--spec-type", "none"])
else:
    raise SystemExit(f"unknown N5 server arm: {arm}")
print(json.dumps({"arm": arm, "argv": cmd, "shell": shlex.join(cmd)}))
PY
}

server_cmds_json() {
  python3 - "$LLAMA_CPP_DIR" "$LLAMA_LIB_DIR" "$LLAMA_SERVER" "$TARGET_MODEL" "$DRAFT_MODEL" "$PORT" "$THREADS" "$CONTEXT" "$UBATCH" "$DRAFT_MAX" "$N_PREDICT" <<'PY'
import json
import shlex
import subprocess
import sys

arms = ["positive_mtp", "spec_off", "n5_spec_on"]
out = {}
for arm in arms:
    raw = subprocess.check_output(
        [
            sys.executable,
            "-c",
            r'''
import json, shlex, sys
arm, llama_cpp_dir, llama_lib_dir, llama_server, target, draft, port, threads, context, ubatch, draft_max, _n_predict = sys.argv[1:]
cmd = [
    "env", f"LD_LIBRARY_PATH={llama_lib_dir}", "numactl", "--interleave=all",
    llama_server, "-m", target, "-t", threads, "-np", "1", "-c", context, "-ub", ubatch,
    "-ngl", "0", "--port", port, "--metrics", "--slots", "--jinja", "--reasoning", "auto",
    "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",
]
if arm == "n5_spec_on":
    cmd.extend(["-md", draft, "--spec-type", "draft-tree", "--spec-draft-n-max", draft_max, "--spec-draft-p-split", "0.05"])
elif arm == "positive_mtp":
    cmd.extend(["--spec-type", "draft-mtp", "--spec-draft-n-max", draft_max])
elif arm == "spec_off":
    cmd.extend(["--spec-type", "none"])
else:
    raise SystemExit(f"unknown N5 server arm: {arm}")
print(json.dumps({"arm": arm, "argv": cmd, "shell": shlex.join(cmd)}))
''',
            arm,
            *sys.argv[1:],
        ],
        text=True,
    )
    out[arm] = json.loads(raw)
print(json.dumps(out, sort_keys=True))
PY
}

run_compatibility_check() {
  "$PYTHON_BIN" "$COMPAT_CHECK" \
    --strict \
    --expect-bos 248044 \
    --expect-eos 248046 \
    --expect-pad 248055 \
    "$DRAFT_MODEL" "$TARGET_MODEL" >"$COMPAT_LOG" 2>&1
}

write_commands() {
  local server_shell
  server_shell=$(server_cmd_json n5_spec_on | python3 -c 'import json,sys; print(json.load(sys.stdin)["shell"])')
  cat >"$COMMANDS_SH" <<EOF
#!/bin/bash
set -euo pipefail

# Generated by n5_frontdoor_drafter_retest.sh.
# Run only in a coordinated clean window after the strict preflight is ready.

export LLAMA_CPP_DIR=${LLAMA_CPP_DIR@Q}
export LLAMA_SERVER=${LLAMA_SERVER@Q}
export LLAMA_LIB_DIR=${LLAMA_LIB_DIR@Q}
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
export N_PROMPTS=${N_PROMPTS@Q}
export MIN_DRAFT_RATIO=${MIN_DRAFT_RATIO@Q}

${RESEARCH_ROOT@Q}/scripts/benchmark/n5_frontdoor_drafter_retest.sh --execute

# N5 external-draft server launch command used internally by --execute:
# $server_shell >${SERVER_LOG@Q} 2>&1
EOF
  chmod +x "$COMMANDS_SH"
}

if ! git_dir_present; then
  add_blocker "llama.cpp git tree not found at ${LLAMA_CPP_DIR}"
else
  if [[ "$(readlink -f "$LLAMA_CPP_DIR")" == "$(readlink -f "$PRODUCTION_LLAMA_CPP_DIR")" ]]; then
    add_blocker "production llama.cpp tree selected at ${LLAMA_CPP_DIR}; use an isolated experimental worktree/build for the retest"
  fi
  if ! commit_present "$EXPECTED_COMMIT"; then
    add_blocker "expected commit ${EXPECTED_COMMIT} is not present in ${LLAMA_CPP_DIR}"
  fi
  case "$SAFETY_COMMIT_MODE" in
    ancestor)
      if ! commit_present "$SAFETY_COMMIT"; then
        add_blocker "safety commit ${SAFETY_COMMIT} is not present in ${LLAMA_CPP_DIR}"
      elif commit_present "$EXPECTED_COMMIT" && ! is_ancestor "$SAFETY_COMMIT" "$EXPECTED_COMMIT"; then
        add_blocker "safety commit ${SAFETY_COMMIT} is not an ancestor of expected commit ${EXPECTED_COMMIT}"
      fi
      ;;
    semantic_audit)
      if [[ -z "$SAFETY_AUDIT_REF" ]]; then
        add_blocker "SAFETY_COMMIT_MODE=semantic_audit requires SAFETY_AUDIT_REF"
      fi
      ;;
    *)
      add_blocker "unknown SAFETY_COMMIT_MODE=${SAFETY_COMMIT_MODE}; expected ancestor or semantic_audit"
      ;;
  esac
  current_head=$(git_head)
  if [[ "$current_head" != "$EXPECTED_COMMIT" ]]; then
    add_blocker "active llama.cpp HEAD is ${current_head:-unknown}, expected ${EXPECTED_COMMIT}; use an isolated worktree/build for the retest"
  fi
fi

if [[ ! -x "$LLAMA_SERVER" ]]; then
  add_blocker "llama-server binary not executable: ${LLAMA_SERVER}"
else
  version=$(binary_version)
  if [[ -z "$version" ]]; then
    add_blocker "llama-server --version produced no output; cannot verify binary provenance"
  elif [[ "$version" != *"$EXPECTED_COMMIT"* && "$version" != *"${EXPECTED_COMMIT:0:8}"* ]]; then
    add_blocker "llama-server --version does not report expected commit ${EXPECTED_COMMIT}: ${version}"
  fi
  built_at=$(stat -c %Y "$LLAMA_SERVER")
  expected_at=$(commit_timestamp)
  if [[ "$expected_at" -gt 0 && "$built_at" -lt "$expected_at" ]]; then
    add_blocker "llama-server binary predates expected commit ${EXPECTED_COMMIT}; rebuild before retest"
  fi
  if ! check_spec_flag_surface "$(binary_help)"; then
    add_blocker "llama-server speculative flag surface is missing required N5/v7 tokens: ${missing_spec_tokens[*]}"
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

server_cmds=$(server_cmds_json)
execution_mode="dry_run"
if [[ "$EXECUTE" -eq 1 ]]; then
  execution_mode="execute"
fi
python3 - "$PREFLIGHT_JSON" "$status" "$execution_mode" "$(json_array "${blockers[@]}")" "$LLAMA_CPP_DIR" "$(git_head)" "$EXPECTED_COMMIT" "$SAFETY_COMMIT" "$SAFETY_COMMIT_MODE" "$SAFETY_AUDIT_REF" "$LLAMA_SERVER" "$LLAMA_LIB_DIR" "$TARGET_MODEL" "$DRAFT_MODEL" "$PYTHON_BIN" "$PORT" "$THREADS" "$CONTEXT" "$UBATCH" "$N_PREDICT" "$DRAFT_MAX" "$N_PROMPTS" "$MIN_DRAFT_RATIO" "$COMMANDS_SH" "$COMPAT_LOG" "$RESULTS_CSV" "$RESULTS_JSONL" "$SUMMARY_JSON" "$server_cmds" "$(json_array "${REQUIRED_SPEC_TOKENS[@]}")" "$(json_array "${missing_spec_tokens[@]}")" <<'PY'
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
    safety_commit_mode,
    safety_audit_ref,
    llama_server,
    llama_lib_dir,
    target,
    draft,
    python_bin,
    port,
    threads,
    context,
    ubatch,
    n_predict,
    draft_max,
    n_prompts,
    min_draft_ratio,
    commands_sh,
    compat_log,
    results_csv,
    results_jsonl,
    summary_json,
    server_cmds_json,
    required_spec_tokens_json,
    missing_spec_tokens_json,
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
        "safety_commit_mode": safety_commit_mode,
        "safety_audit_ref": safety_audit_ref or None,
    },
    "binary": {
        "llama_server": llama_server,
        "ld_library_path_prefix": llama_lib_dir,
        "required_spec_tokens": json.loads(required_spec_tokens_json),
        "missing_spec_tokens": json.loads(missing_spec_tokens_json),
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
        "n_prompts": int(n_prompts),
        "min_draft_ratio": float(min_draft_ratio),
    },
    "artifacts": {
        "commands_sh": commands_sh,
        "compatibility_log": compat_log,
        "results_csv": results_csv,
        "results_jsonl": results_jsonl,
        "summary_json": summary_json,
    },
    "server_commands": json.loads(server_cmds_json),
    "required_arms": ["positive_mtp", "spec_off", "n5_spec_on"],
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

run_arm() {
  local arm="$1"
  local arm_log="${OUTPUT_DIR}/${arm}.llama-server.log"
  local server_shell
  server_shell=$(server_cmd_json "$arm" | python3 -c 'import json,sys; print(json.load(sys.stdin)["shell"])')
  if port_in_use; then
    echo "ERROR: port ${PORT} is still in use before launching ${arm}" >&2
    exit 5
  fi
  echo "Launching ${arm} on port ${PORT}..."
  eval "$server_shell" >"$arm_log" 2>&1 &
  SERVER_PID=$!

  cleanup_arm() {
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      kill "$SERVER_PID" 2>/dev/null || true
      wait "$SERVER_PID" 2>/dev/null || true
    fi
    if port_in_use; then
      kill_port_listeners TERM
      wait_for_port_release 20 || true
    fi
    if port_in_use; then
      kill_port_listeners KILL
      wait_for_port_release 20 || true
    fi
  }
  trap cleanup_arm EXIT

  sleep 1
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: ${arm} server exited during startup; see ${arm_log}" >&2
    cleanup_arm
    trap - EXIT
    exit 4
  fi

  if ! wait_for_server "$SERVER_PID"; then
    echo "ERROR: ${arm} server did not become healthy; see ${arm_log}" >&2
    cleanup_arm
    trap - EXIT
    exit 4
  fi

  python3 - "$arm" "$PORT" "$N_PREDICT" "$N_PROMPTS" "$RESULTS_CSV" "$RESULTS_JSONL" "$SUMMARY_JSON" "$arm_log" "$MIN_DRAFT_RATIO" <<'PY'
import csv
import json
import math
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

(
    arm,
    port,
    n_predict,
    n_prompts,
    csv_path,
    jsonl_path,
    summary_path,
    server_log_path,
    min_draft_ratio,
) = sys.argv[1:]
port = int(port)
n_predict = int(n_predict)
n_prompts = int(n_prompts)
min_draft_ratio = float(min_draft_ratio)

prompts = [
    "Summarize why tokenizer-aligned speculative decoding must report draft acceptance before it can gate production routing.",
    "Write a compact Python function that validates a JSON object has keys action and rationale.",
    "Explain why a spec-off control arm is necessary when measuring draft-token acceptance.",
    "Return a short checklist for detecting fallback paths in llama-server speculative decoding logs.",
    "Describe how token-weighted acceptance differs from prompt-averaged acceptance.",
    "Write three concise risks of using a stale frontdoor model path in a benchmark harness.",
    "Explain why a positive control should prove draft_n telemetry before an experimental arm runs.",
    "Produce a two-sentence operator note about keeping production kernels immutable.",
][:n_prompts]

fields = [
    "created_at",
    "arm",
    "prompt_idx",
    "status",
    "completion_tokens",
    "elapsed_ms",
    "tokens_per_sec",
    "draft_accepted",
    "draft_total",
    "acceptance_rate",
    "taxonomy",
    "error",
]

def post_completion(prompt: str) -> tuple[dict, int]:
    payload = {
        "model": "n5-frontdoor-drafter",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n_predict,
        "temperature": 0,
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=900) as response:
        data = json.loads(response.read().decode("utf-8"))
    elapsed_ms = int((time.monotonic() - start) * 1000)
    return data, elapsed_ms

def classify(log_text: str, draft_total: int) -> str:
    lower = log_text.lower()
    if arm == "spec_off":
        return "spec_off_control"
    if draft_total > 0:
        return "drafted_ok"
    if "decode" in lower and ("fail" in lower or "error" in lower or "negative" in lower):
        return "decode_failed_fallback"
    if "spec" not in lower or "draft" not in lower:
        return "no_spec_enabled"
    return "no_draft_tokens"

csv_file = Path(csv_path)
jsonl_file = Path(jsonl_path)
csv_file.parent.mkdir(parents=True, exist_ok=True)
write_header = not csv_file.exists() or csv_file.stat().st_size == 0
rows = []
log_text = Path(server_log_path).read_text(errors="replace") if Path(server_log_path).exists() else ""

with csv_file.open("a", newline="", encoding="utf-8") as csv_handle, jsonl_file.open("a", encoding="utf-8") as jsonl_handle:
    writer = csv.DictWriter(csv_handle, fieldnames=fields)
    if write_header:
        writer.writeheader()
    for idx, prompt in enumerate(prompts):
        error = ""
        status = "ok"
        completion_tokens = 0
        draft_accepted = 0
        draft_total = 0
        elapsed_ms = 0
        try:
            data, elapsed_ms = post_completion(prompt)
            usage = data.get("usage", {})
            timings = data.get("timings", {})
            completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            draft_accepted = int(timings.get("draft_n_accepted", 0) or 0)
            draft_total = int(timings.get("draft_n", 0) or 0)
        except Exception as exc:  # noqa: BLE001 - persisted in result taxonomy
            status = "error"
            error = str(exc)
        taxonomy = classify(log_text, draft_total)
        elapsed = max(elapsed_ms, 1)
        acceptance = draft_accepted / draft_total if draft_total > 0 else 0.0
        row = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "arm": arm,
            "prompt_idx": idx,
            "status": status,
            "completion_tokens": completion_tokens,
            "elapsed_ms": elapsed_ms,
            "tokens_per_sec": f"{completion_tokens * 1000 / elapsed:.4f}",
            "draft_accepted": draft_accepted,
            "draft_total": draft_total,
            "acceptance_rate": f"{acceptance:.6f}",
            "taxonomy": taxonomy,
            "error": error,
        }
        writer.writerow(row)
        jsonl_handle.write(json.dumps(row, sort_keys=True) + "\n")
        jsonl_handle.flush()
        rows.append(row)

def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    phat = successes / total
    denom = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))

total_completion = sum(int(row["completion_tokens"]) for row in rows)
total_draft = sum(int(row["draft_total"]) for row in rows)
total_accepted = sum(int(row["draft_accepted"]) for row in rows)
statuses_ok = all(row["status"] == "ok" for row in rows)
taxonomy_counts: dict[str, int] = {}
for row in rows:
    taxonomy_counts[row["taxonomy"]] = taxonomy_counts.get(row["taxonomy"], 0) + 1
acceptance = total_accepted / total_draft if total_draft > 0 else 0.0
ci_low, ci_high = wilson(total_accepted, total_draft)
min_total_completion = math.ceil(len(prompts) * n_predict * 0.9)
min_total_draft = math.ceil(total_completion * min_draft_ratio)

if arm == "spec_off":
    decision_grade = statuses_ok and total_draft == 0 and total_completion >= min_total_completion
elif arm == "positive_mtp":
    decision_grade = statuses_ok and total_draft >= min_total_draft and "drafted_ok" in taxonomy_counts
else:
    decision_grade = statuses_ok and total_draft >= min_total_draft and "drafted_ok" in taxonomy_counts and total_completion >= min_total_completion

summary_path_obj = Path(summary_path)
summary = {}
if summary_path_obj.exists():
    try:
        summary = json.loads(summary_path_obj.read_text())
    except json.JSONDecodeError:
        summary = {}
summary.setdefault("created_at", datetime.now(timezone.utc).isoformat())
summary.setdefault("arms", {})
summary["arms"][arm] = {
    "prompts": len(rows),
    "status_ok": statuses_ok,
    "completion_tokens": total_completion,
    "draft_accepted": total_accepted,
    "draft_total": total_draft,
    "acceptance_rate": acceptance,
    "acceptance_wilson95": [ci_low, ci_high],
    "taxonomy_counts": taxonomy_counts,
    "min_total_completion_tokens": min_total_completion,
    "min_total_draft_tokens": min_total_draft,
    "decision_grade": decision_grade,
}
required = {"positive_mtp", "spec_off", "n5_spec_on"}
have = set(summary["arms"])
summary["decision_grade"] = required.issubset(have) and all(summary["arms"][name]["decision_grade"] for name in required)
summary["required_arms"] = sorted(required)
summary_path_obj.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

if not decision_grade:
    raise SystemExit(f"{arm} did not satisfy decision-grade evidence requirements; see {summary_path}")
PY

  cleanup_arm
  if ! wait_for_port_release 30; then
    echo "ERROR: port ${PORT} did not close after ${arm}; refusing to reuse stale server" >&2
    trap - EXIT
    exit 5
  fi
  trap - EXIT
}

rm -f "$RESULTS_CSV" "$RESULTS_JSONL" "$SUMMARY_JSON"
for arm in positive_mtp spec_off n5_spec_on; do
  run_arm "$arm"
done

echo "N5 evidence run complete: ${SUMMARY_JSON}"
