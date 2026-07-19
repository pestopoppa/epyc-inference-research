#!/usr/bin/env python3
"""Prepare an OP-2 quiet-window run bundle without running inference.

This helper turns the narrowed OP-2 package into an artifact directory that
records current state and the exact operator commands for the remaining gates:
live v6+iqk role/garbage verification and the clean canonical CPU decode bench.
It intentionally does not start/stop servers, AutoPilot, perf, ROCm tools, or
llama binaries.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable


SCHEMA = "epyc.op2_quiet_window_prep.v1"
PGPU1_CERTIFICATION_NOTE = (
    "P-GPU-1 is ratified for production-named MI210 GPU claims only: experimental, "
    "candidate, or fork GPU rows remain observation-grade until promoted to a "
    "production-named kernel or strict retro-certification applies."
)

DEFAULT_ROOT = Path("/mnt/raid0/llm/epyc-root")
DEFAULT_RESEARCH = Path("/mnt/raid0/llm/epyc-inference-research")
DEFAULT_ORCHESTRATOR = Path("/mnt/raid0/llm/epyc-orchestrator")
DEFAULT_PROD_LLAMA = Path("/mnt/raid0/llm/llama.cpp")
DEFAULT_EXP_LLAMA = Path("/mnt/raid0/llm/llama.cpp-experimental")
DEFAULT_OUTPUT_BASE = DEFAULT_RESEARCH / "data" / "op2_canonical_bench_window"
DEFAULT_FRONTDOOR_Q8 = Path("/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf")

RUN_SUBDIRS = (
    "approvals",
    "preflight",
    "attestations",
    "live-v6",
    "canonical-v6",
    "b1-barrier-fusion",
    "b4-dsa-d3",
    "routing",
)

Runner = Callable[..., subprocess.CompletedProcess[str]]


def utc_now() -> datetime:
    return datetime.now(UTC)


def utc_stamp() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ")


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def run_capture(
    argv: list[str],
    *,
    cwd: Path | None = None,
    runner: Runner = subprocess.run,
    timeout: float = 20.0,
) -> dict[str, Any]:
    try:
        proc = runner(argv, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "argv": argv,
            "cwd": str(cwd) if cwd else None,
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
        }
    return {
        "argv": argv,
        "cwd": str(cwd) if cwd else None,
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def git_capture(repo: Path, argv: list[str], *, runner: Runner = subprocess.run) -> dict[str, Any]:
    return run_capture(["git", *argv], cwd=repo, runner=runner, timeout=20)


def git_state(repo: Path, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    state = {
        "path": str(repo),
        "exists": repo.exists(),
        "branch": None,
        "head": None,
        "upstream": None,
        "tracked_dirty_lines": None,
        "commands": {},
    }
    if not repo.exists():
        return state

    commands = {
        "branch": git_capture(repo, ["branch", "--show-current"], runner=runner),
        "head": git_capture(repo, ["rev-parse", "HEAD"], runner=runner),
        "upstream": git_capture(repo, ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], runner=runner),
        "tracked_status": git_capture(repo, ["status", "--porcelain=v1", "--untracked-files=no"], runner=runner),
    }
    state["commands"] = commands
    if commands["branch"]["ok"]:
        state["branch"] = commands["branch"]["stdout"].strip()
    if commands["head"]["ok"]:
        state["head"] = commands["head"]["stdout"].strip()
    if commands["upstream"]["ok"]:
        state["upstream"] = commands["upstream"]["stdout"].strip()
    if commands["tracked_status"]["ok"]:
        status = commands["tracked_status"]["stdout"].splitlines()
        state["tracked_dirty_lines"] = len([line for line in status if line.strip()])
    return state


def collect_host_state(*, runner: Runner = subprocess.run) -> dict[str, Any]:
    return {
        "captured_at": utc_now().isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "kernel_knobs": {
            "numa_balancing": read_text(Path("/proc/sys/kernel/numa_balancing")),
            "perf_event_paranoid": read_text(Path("/proc/sys/kernel/perf_event_paranoid")),
            "thp_enabled": read_text(Path("/sys/kernel/mm/transparent_hugepage/enabled")),
            "thp_defrag": read_text(Path("/sys/kernel/mm/transparent_hugepage/defrag")),
        },
        "commands": {
            "uptime": run_capture(["uptime"], runner=runner),
            "free_h": run_capture(["free", "-h"], runner=runner),
            "uname_a": run_capture(["uname", "-a"], runner=runner),
        },
    }


def parse_matching_processes(stdout: str, *, current_pid: int | None = None) -> list[dict[str, Any]]:
    process_names = {"llama-server", "llama-bench", "llama-cli", "uvicorn", "rocprof", "rocprofv2", "perf"}
    arg_markers = ("autopilot", "orchestrator_stack")
    matches: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=2)
        if len(parts) < 3:
            continue
        pid_text, comm, args = parts
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if current_pid is not None and pid == current_pid:
            continue
        lower_args = args.lower()
        if comm in process_names or any(marker in lower_args for marker in arg_markers):
            matches.append({"pid": pid, "comm": comm, "args": args})
    return matches


def collect_process_state(*, runner: Runner = subprocess.run) -> dict[str, Any]:
    proc = run_capture(["ps", "-eo", "pid=,comm=,args="], runner=runner, timeout=10)
    return {
        "captured_at": utc_now().isoformat(),
        "ps_ok": proc["ok"],
        "matching_processes": parse_matching_processes(proc["stdout"], current_pid=os.getpid()) if proc["ok"] else [],
        "ps_stderr": proc["stderr"],
    }


def measurement_status(measurement_path: Path) -> dict[str, Any]:
    text = read_text(measurement_path) or ""
    p_gpu_line = ""
    for line in text.splitlines():
        if "P-GPU-1" in line:
            p_gpu_line = line.strip()
            break
    stale_reason_note = ""
    if "hardware not acquired" in p_gpu_line:
        stale_reason_note = (
            "Raw MEASUREMENT line still carries the pre-MI210 defer reason; "
            "treat only the deferred/unratified status as current until the human "
            "MEASUREMENT amendment updates P-GPU-1."
        )
    return {
        "path": str(measurement_path),
        "exists": measurement_path.exists(),
        "p_gpu_1_line": p_gpu_line,
        "p_gpu_1_deferred": "DEFERRED" in p_gpu_line,
        "p_gpu_1_line_note": stale_reason_note,
        "p_gpu_1_certification_note": PGPU1_CERTIFICATION_NOTE,
        "authoritative": True,
    }


def stage_plan() -> dict[str, Any]:
    return {
        "remaining_payload": [
            {
                "stage": "live_v6_iqk_role_garbage_verification",
                "status": "operator_window_required",
                "protocol": "P-SMOKE-1 unless a stronger runner stamps otherwise",
                "starts_inference": True,
                "production_v6_edits_allowed": False,
            },
            {
                "stage": "clean_canonical_cpu_decode_bench",
                "status": "operator_window_required",
                "protocol": "P-BENCH-1 via bench_canonical.sh/canonical_recipe.py",
                "starts_inference": True,
                "production_v6_edits_allowed": False,
            },
        ],
        "skipped_or_closed": [
            {
                "stage": "b1_barrier_fusion_ab",
                "status": "skipped_not_staged",
                "reason": "no current v7 barrier-fusion flag or immutable binary pair",
            },
            {
                "stage": "b4_dsa_d3_profile",
                "status": "closed_no_go",
                "reason": "D3.1 profile found Lightning Indexer at only 1.08% of cycle samples",
            },
        ],
    }


def build_manifest(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    run_root = Path(args.output_dir).expanduser().resolve()
    repos = {
        "root": Path(args.root_repo),
        "research": Path(args.research_repo),
        "orchestrator": Path(args.orchestrator_repo),
        "production_llama": Path(args.production_llama_repo),
        "experimental_llama": Path(args.experimental_llama_repo),
    }
    return {
        "schema": SCHEMA,
        "generated_at": utc_now().isoformat(),
        "run_id": args.run_id,
        "run_root": str(run_root),
        "status": "prepared_no_inference",
        "operator_approval_ref": args.operator_approval_ref,
        "quiet_window_required": True,
        "autopilot_restart_authorized": False,
        "production_v6_touch_authorized": False,
        "stage_plan": stage_plan(),
        "measurement": measurement_status(Path(args.measurement_path)),
        "operator_execution": operator_execution_manifest(args),
        "host_state": collect_host_state(runner=runner),
        "process_state": collect_process_state(runner=runner),
        "repo_state": {name: git_state(path, runner=runner) for name, path in repos.items()},
        "artifacts": {
            "stage_plan": str(run_root / "stage_plan.json"),
            "operator_next_commands": str(run_root / "operator_next_commands.sh"),
            "summary": str(run_root / "summary.md"),
        },
    }


def operator_execution_manifest(args: argparse.Namespace) -> dict[str, Any]:
    if args.execution_output_dir:
        return {
            "mode": "static",
            "default_run_root": str(Path(args.execution_output_dir).expanduser().resolve()),
            "output_base": None,
        }
    return {
        "mode": "dynamic_timestamped",
        "default_run_root": "${OP2_EXECUTION_BASE}/${OP2_RUN_ID}",
        "output_base": str(DEFAULT_OUTPUT_BASE),
    }


def operator_run_root_preamble(execution_output_dir: str | None) -> str:
    if execution_output_dir:
        execution_root = Path(execution_output_dir).expanduser().resolve()
        return f'export OP2_RUN_ROOT="${{OP2_RUN_ROOT:-{execution_root}}}"'
    return "\n".join(
        [
            ': "${OP2_RUN_ID:=op2-canonical-bench-window-$(date -u +%Y%m%dT%H%M%SZ)}"',
            f'export OP2_EXECUTION_BASE="${{OP2_EXECUTION_BASE:-{DEFAULT_OUTPUT_BASE}}}"',
            'export OP2_RUN_ROOT="${OP2_RUN_ROOT:-${OP2_EXECUTION_BASE}/${OP2_RUN_ID}}"',
        ]
    )


def build_operator_commands(frontdoor_model: Path, execution_output_dir: str | None) -> str:
    run_root_preamble = operator_run_root_preamble(execution_output_dir)
    return f"""#!/usr/bin/env bash
set -euo pipefail

# OP-2 operator quiet-window commands.
# This file is generated by op2_quiet_window_prep.py and is intentionally not
# executed by that helper. Run it only inside an approved quiet window.
# P-GPU-1 caveat: {PGPU1_CERTIFICATION_NOTE}
# These commands collect live-v6/CPU evidence; they do not certify experimental-v7 GPU rows.

{run_root_preamble}
export FRONTDOOR_Q8="${{FRONTDOOR_Q8:-{frontdoor_model}}}"
mkdir -p "$OP2_RUN_ROOT"/{{approvals,preflight,attestations,live-v6,canonical-v6,b1-barrier-fusion,b4-dsa-d3,routing}}

cd /mnt/raid0/llm/epyc-orchestrator
python scripts/server/preflight_gate.py \\
  --require-servers \\
  --output-dir "$OP2_RUN_ROOT/attestations" \\
  --json | tee "$OP2_RUN_ROOT/preflight/live_stack_preflight.json"

cd /mnt/raid0/llm/epyc-inference-research
python3 scripts/benchmark/perf_counter_preflight.py \\
  --probe \\
  --strict \\
  --output-json "$OP2_RUN_ROOT/preflight/perf_counter_preflight.json" \\
  --output-md "$OP2_RUN_ROOT/preflight/perf_counter_preflight.md"

python3 scripts/benchmark/cpu_bench_clean_preflight.py \\
  --output-json "$OP2_RUN_ROOT/preflight/cpu_clean_record_only.json" \\
  --strict

cd /mnt/raid0/llm/epyc-orchestrator
python scripts/server/orchestrator_stack.py status \\
  | tee "$OP2_RUN_ROOT/live-v6/orchestrator_stack_status.txt"

ps -eo pid,lstart,comm,args \\
  | rg 'llama-server|uvicorn|autopilot|perf|rocprof' \\
  > "$OP2_RUN_ROOT/live-v6/process_snapshot.txt" || true

python3 - "$OP2_RUN_ROOT/live-v6/process_snapshot.txt" "$OP2_RUN_ROOT/live-v6/process_blockers.json" <<'PY'
import json
import sys
from pathlib import Path

snapshot_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
allowed = {{"llama-server", "uvicorn"}}
blocked_basenames = {{"llama-bench", "llama-cli", "perf", "rocprof", "rocprofv2"}}
blockers = []
for raw in snapshot_path.read_text(encoding="utf-8").splitlines():
    parts = raw.split(maxsplit=4)
    if len(parts) < 5:
        continue
    pid, _weekday, _month, _day, rest = parts
    fields = rest.split(maxsplit=4)
    if len(fields) < 5:
        continue
    _time, _year, comm, args = fields[0], fields[1], fields[2], fields[3] if len(fields) == 4 else fields[4]
    lower_args = raw.lower()
    reason = None
    if comm in blocked_basenames:
        reason = f"blocked process {{comm}}"
    elif "autopilot" in lower_args:
        reason = "blocked AutoPilot process"
    elif comm == "perf" and (" stat " in f" {{lower_args}} " or " record " in f" {{lower_args}} "):
        reason = "blocked perf profiler"
    elif "rocprof" in lower_args:
        reason = "blocked ROCm profiler"
    if reason is not None:
        blockers.append({{"pid": pid, "comm": comm, "reason": reason, "line": raw}})

payload = {{
    "schema": "epyc.op2.quiet_window_process_blockers.v1",
    "allowed_processes": sorted(allowed),
    "blocker_n": len(blockers),
    "blockers": blockers,
}}
out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
if blockers:
    print(json.dumps(payload, sort_keys=True))
    raise SystemExit(74)
print(json.dumps(payload, sort_keys=True))
PY

for pid in $(pgrep -f '/mnt/raid0/llm/llama.cpp/build/bin/llama-server' || true); do
  mkdir -p "$OP2_RUN_ROOT/live-v6/pid-${{pid}}"
  ps -p "$pid" -o pid,lstart,etime,comm,args > "$OP2_RUN_ROOT/live-v6/pid-${{pid}}/ps.txt"
  tr '\\0' '\\n' < "/proc/${{pid}}/environ" \\
    | rg '^(GGML_IQK|LD_LIBRARY_PATH|OMP_|KMP_)=' \\
    > "$OP2_RUN_ROOT/live-v6/pid-${{pid}}/environ.filtered.txt" || true
  rg -a 'llama.cpp/build|libllama|libggml' "/proc/${{pid}}/maps" \\
    > "$OP2_RUN_ROOT/live-v6/pid-${{pid}}/maps.llama.txt" || true
done

cat > "$OP2_RUN_ROOT/live-v6/role_smoke_ports.tsv" <<'EOF'
frontdoor 8070
worker_general 8072
architect_general 8083
ingest_long_context 8085
worker_vision 8086
vision_escalation 8087
EOF

cat > "$OP2_RUN_ROOT/live-v6/role_smoke_request.json" <<'EOF'
{{"model":"local","messages":[{{"role":"user","content":"Return exactly: OP2_READY"}}],"temperature":0,"seed":42,"max_tokens":32}}
EOF

cat > "$OP2_RUN_ROOT/live-v6/role_smoke_check.py" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

role, port, response_path, meta_path, out_path = sys.argv[1:]
response_text = Path(response_path).read_text(encoding="utf-8")
meta_text = Path(meta_path).read_text(encoding="utf-8")
try:
    response = json.loads(response_text)
except json.JSONDecodeError as exc:
    response = None
    content = ""
    parse_error = str(exc)
else:
    choice = (response.get("choices") or [{{}}])[0]
    message = choice.get("message") or {{}}
    content = message.get("content") or choice.get("text") or ""
    parse_error = None
try:
    curl_meta = json.loads(meta_text)
except json.JSONDecodeError:
    curl_meta = dict(raw=meta_text.strip())
ok = content.strip() == "OP2_READY"
out = dict(
    role=role,
    port=int(port),
    ok=ok,
    expected="OP2_READY",
    content=content,
    content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    response_parse_error=parse_error,
    usage=(response or {{}}).get("usage"),
    timings=(response or {{}}).get("timings"),
    curl=curl_meta,
)
Path(out_path).write_text(json.dumps(out, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(json.dumps(out, sort_keys=True))
PY

: > "$OP2_RUN_ROOT/live-v6/role_smoke_summary.jsonl"
while read -r role port; do
  role_dir="$OP2_RUN_ROOT/live-v6/role-${{role}}"
  mkdir -p "$role_dir"
  cp "$OP2_RUN_ROOT/live-v6/role_smoke_request.json" "$role_dir/request.json"
  if curl -sS --max-time 180 \\
      -H 'Content-Type: application/json' \\
      -o "$role_dir/response.json" \\
      -w '{{"http_code":%{{http_code}},"time_total":%{{time_total}},"remote_ip":"%{{remote_ip}}","remote_port":%{{remote_port}}}}\\n' \\
      "http://127.0.0.1:${{port}}/v1/chat/completions" \\
      -d @"$role_dir/request.json" \\
      > "$role_dir/curl_meta.json"; then
    python3 "$OP2_RUN_ROOT/live-v6/role_smoke_check.py" \\
      "$role" "$port" "$role_dir/response.json" "$role_dir/curl_meta.json" "$role_dir/check.json" \\
      >> "$OP2_RUN_ROOT/live-v6/role_smoke_summary.jsonl"
  else
    rc=$?
    printf '{{"role":"%s","port":%s,"ok":false,"curl_exit":%s}}\\n' "$role" "$port" "$rc" \\
      | tee "$role_dir/check.json" >> "$OP2_RUN_ROOT/live-v6/role_smoke_summary.jsonl"
  fi
done < "$OP2_RUN_ROOT/live-v6/role_smoke_ports.tsv"

python3 - "$OP2_RUN_ROOT/live-v6/role_smoke_summary.jsonl" "$OP2_RUN_ROOT/live-v6/role_smoke_aggregate.json" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
aggregate_path = Path(sys.argv[2])
rows = [json.loads(line) for line in summary_path.read_text(encoding="utf-8").splitlines() if line.strip()]
aggregate = dict(
    schema="epyc.op2.live_v6_role_smoke.v1",
    row_n=len(rows),
    pass_n=sum(1 for row in rows if row.get("ok") is True),
    fail_n=sum(1 for row in rows if row.get("ok") is not True),
    roles=[row.get("role") for row in rows],
    all_pass=all(row.get("ok") is True for row in rows) if rows else False,
)
aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(json.dumps(aggregate, sort_keys=True))
PY

cd /mnt/raid0/llm/epyc-inference-research
python3 scripts/benchmark/cpu_bench_clean_preflight.py \\
  --run-sentinel \\
  --output-json "$OP2_RUN_ROOT/canonical-v6/cpu_clean_sentinel.json" \\
  --strict

./scripts/benchmark/bench_canonical.sh \\
  -m "$FRONTDOOR_Q8" \\
  -p 0 \\
  -n 128 \\
  -r 10 \\
  --dry-run \\
  -- -o json \\
  > "$OP2_RUN_ROOT/canonical-v6/frontdoor_q8_tg128.dryrun.txt" 2>&1

./scripts/benchmark/bench_canonical.sh \\
  -m "$FRONTDOOR_Q8" \\
  -p 0 \\
  -n 128 \\
  -r 10 \\
  -- -o json \\
  > "$OP2_RUN_ROOT/canonical-v6/frontdoor_q8_tg128.results.json" \\
  2> "$OP2_RUN_ROOT/canonical-v6/frontdoor_q8_tg128.stderr.txt"

cat > "$OP2_RUN_ROOT/b1-barrier-fusion/status.json" <<'EOF'
{{"stage":"b1_barrier_fusion_ab","status":"skipped_not_staged","reason":"no current v7 barrier-fusion flag or immutable binary pair"}}
EOF

cat > "$OP2_RUN_ROOT/b4-dsa-d3/status.json" <<'EOF'
{{"stage":"b4_dsa_d3_profile","status":"closed_no_go","reason":"D3.1 profile found Lightning Indexer at only 1.08% of cycle samples"}}
EOF
"""


def render_summary(manifest: dict[str, Any]) -> str:
    stage_rows = []
    for row in manifest["stage_plan"]["remaining_payload"]:
        stage_rows.append(f"| {row['stage']} | {row['status']} | {row['protocol']} |")
    skipped_rows = []
    for row in manifest["stage_plan"]["skipped_or_closed"]:
        skipped_rows.append(f"| {row['stage']} | {row['status']} | {row['reason']} |")
    process_count = len(manifest["process_state"].get("matching_processes", []))
    measurement = manifest["measurement"]
    return "\n".join(
        [
            "# OP-2 Quiet-Window Prep",
            "",
            f"- Schema: `{manifest['schema']}`",
            f"- Generated: `{manifest['generated_at']}`",
            f"- Run id: `{manifest['run_id']}`",
            f"- Run root: `{manifest['run_root']}`",
            "- Status: `prepared_no_inference`",
            f"- Raw P-GPU-1 MEASUREMENT line: `{measurement.get('p_gpu_1_line', '')}`",
            f"- Raw-line caveat: `{measurement.get('p_gpu_1_line_note', '')}`",
            f"- P-GPU-1 certification caveat: `{measurement.get('p_gpu_1_certification_note', '')}`",
            f"- Matching live process lines at prep time: `{process_count}`",
            "",
            "## Remaining Payload",
            "",
            "| Stage | Status | Protocol |",
            "|---|---|---|",
            *stage_rows,
            "",
            "## Skipped Or Closed",
            "",
            "| Stage | Status | Reason |",
            "|---|---|---|",
            *skipped_rows,
            "",
            "Run `operator_next_commands.sh` only inside an approved quiet window.",
            "",
        ]
    )


def write_bundle(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    run_root = Path(args.output_dir).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    for subdir in RUN_SUBDIRS:
        (run_root / subdir).mkdir(exist_ok=True)

    manifest = build_manifest(args, runner=runner)
    (run_root / "manifest.json").write_text(canonical_json(manifest), encoding="utf-8")
    (run_root / "stage_plan.json").write_text(canonical_json(manifest["stage_plan"]), encoding="utf-8")
    commands = build_operator_commands(Path(args.frontdoor_model), args.execution_output_dir)
    commands_path = run_root / "operator_next_commands.sh"
    commands_path.write_text(commands, encoding="utf-8")
    commands_path.chmod(0o755)
    (run_root / "summary.md").write_text(render_summary(manifest), encoding="utf-8")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_run_id = f"op2-canonical-bench-window-{utc_stamp()}"
    parser.add_argument("--run-id", default=default_run_id)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_BASE / default_run_id))
    parser.add_argument("--operator-approval-ref", default="MISSING_OPERATOR_APPROVAL")
    parser.add_argument(
        "--execution-output-dir",
        default="",
        help=(
            "Optional static artifact directory for the operator execution script. "
            "If omitted, the generated script creates a fresh timestamped run under "
            f"{DEFAULT_OUTPUT_BASE}."
        ),
    )
    parser.add_argument("--frontdoor-model", default=str(DEFAULT_FRONTDOOR_Q8))
    parser.add_argument("--measurement-path", default="/workspace/MEASUREMENT.md")
    parser.add_argument("--root-repo", default=str(DEFAULT_ROOT))
    parser.add_argument("--research-repo", default=str(DEFAULT_RESEARCH))
    parser.add_argument("--orchestrator-repo", default=str(DEFAULT_ORCHESTRATOR))
    parser.add_argument("--production-llama-repo", default=str(DEFAULT_PROD_LLAMA))
    parser.add_argument("--experimental-llama-repo", default=str(DEFAULT_EXP_LLAMA))
    parser.add_argument("--print-summary", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = write_bundle(args)
    if args.print_summary:
        print(render_summary(manifest))
    else:
        print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
