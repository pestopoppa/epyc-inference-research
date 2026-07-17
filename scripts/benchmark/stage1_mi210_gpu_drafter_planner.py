#!/usr/bin/env python3
"""Dry-run-first Stage-1 MI210 GPU-drafter end-to-end planner.

This planner never launches inference. It emits a reproducible planning bundle
for the post-GLM Stage-1 frontdoor external GPU-drafter A/B:

  - manifest.json
  - commands.sh
  - gate_summary.json

The command templates are pinned to the experimental v7 HIP build and refuse
production v6 llama-server paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import shlex
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_ROOT = SCRIPT_DIR.parent.parent

EXPERIMENTAL_ROOT = Path("/mnt/raid0/llm/llama.cpp-experimental")
EXPERIMENTAL_BIN_DIR = EXPERIMENTAL_ROOT / "build-hip" / "bin"
EXPERIMENTAL_SERVER = EXPERIMENTAL_BIN_DIR / "llama-server"
PRODUCTION_ROOT = Path("/mnt/raid0/llm/llama.cpp")
PRODUCTION_SERVER = PRODUCTION_ROOT / "build-hip" / "bin" / "llama-server"

AUTOPILOT_PATTERN = "scripts/autopilot/autopilot.py start"
DEFAULT_TARGET_MODEL = Path("/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf")
DEFAULT_DRAFT_MODEL = Path("/mnt/raid0/llm/scratch/n5/Qwen3.5-0.8B-Q8_0.frontdoor-mtp-specials.gguf")
DEFAULT_OUTPUT_DIR = (
    RESEARCH_ROOT
    / "data"
    / "specdec_frontdoor_alpha"
    / f"stage1_mi210_gpu_drafter_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
)
DEFAULT_BASELINE_PORT = 19187
DEFAULT_STAGE1_PORT = 19188
DEFAULT_THREADS = 96
DEFAULT_CONTEXT = 8192
DEFAULT_UBATCH = 8192
DEFAULT_DRAFT_MAX = 1
DEFAULT_SPEC_P_SPLIT = 0.05
DEFAULT_MAX_TOKENS = 256
DEFAULT_REQUEST_TIMEOUT_S = 300
DEFAULT_STARTUP_TIMEOUT_S = 300
DEFAULT_MIN_COMPLETION_RATIO = 0.70
PASS_SPEEDUP_THRESHOLD = 1.3
N5_PREREQUISITE_COMMAND = (
    f"{RESEARCH_ROOT}/scripts/benchmark/n5_frontdoor_drafter_retest.sh --strict"
)
DEFAULT_N5_SUMMARY_GLOB = "n5_retest_v7_execute_*/summary.json"
DEFAULT_PROMPT_PACK = [
    "Write exactly 12 numbered checklist items about validating speculative decoding telemetry. Each item must be one sentence of 12 to 18 words.",
    "Write exactly 10 compact JSONL records about frontdoor routing experiments, one per line, with keys id, risk, and mitigation.",
    "Write exactly 8 Python comments describing safe benchmark harness cleanup. Each comment must mention a distinct failure mode.",
    "Write exactly 12 short operator notes about immutable production kernels and experimental v7 validation. Use one sentence per note.",
]


@dataclass(frozen=True)
class GuardState:
    quiet_host_blockers: list[str]

    @property
    def quiet_host_ready(self) -> bool:
        return not self.quiet_host_blockers


@dataclass(frozen=True)
class ArmSpec:
    name: str
    purpose: str
    speculative: bool


BASELINE_ARM = ArmSpec(
    name="baseline_cpu_target_no_spec",
    purpose="CPU target only baseline with speculation disabled",
    speculative=False,
)
STAGE1_ARM = ArmSpec(
    name="stage1_cpu_target_mi210_external_drafter",
    purpose="CPU target plus MI210 external drafter Stage-1 candidate",
    speculative=True,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _probe_pattern(
    pattern: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[str]:
    result = runner(["pgrep", "-af", pattern], capture_output=True, text=True)
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _probe_process_basename(
    name: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[str]:
    result = runner(["ps", "-eo", "pid=,args="], capture_output=True, text=True)
    if result.returncode != 0:
        return []
    matches: list[str] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(maxsplit=1)
        if len(parts) != 2:
            continue
        _pid, args = parts
        argv0 = args.split(maxsplit=1)[0]
        if Path(argv0).name == name:
            matches.append(stripped)
    return matches


def collect_guard_state(
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> GuardState:
    blockers: list[str] = []

    autopilot_matches = _probe_pattern(AUTOPILOT_PATTERN, runner=runner)
    if autopilot_matches:
        blockers.append(
            f"quiet host guard blocked by AutoPilot pattern {AUTOPILOT_PATTERN!r}: {autopilot_matches[0]}"
        )

    llama_server_matches = _probe_process_basename("llama-server", runner=runner)
    if llama_server_matches:
        blockers.append(
            f"quiet host guard blocked by process basename 'llama-server': {llama_server_matches[0]}"
        )

    return GuardState(quiet_host_blockers=blockers)


def validate_experimental_server(binary: Path) -> Path:
    resolved = binary.expanduser().resolve()
    production_resolved = PRODUCTION_SERVER.resolve()
    production_root_resolved = PRODUCTION_ROOT.resolve()
    experimental_root_resolved = EXPERIMENTAL_ROOT.resolve()

    if resolved == production_resolved or production_root_resolved in resolved.parents:
        raise ValueError(f"refusing production v6 llama-server path: {resolved}")
    if resolved != EXPERIMENTAL_SERVER.resolve() and experimental_root_resolved not in resolved.parents:
        raise ValueError(f"refusing non-experimental llama-server path: {resolved}")
    return resolved


def _base_argv(binary: Path) -> list[str]:
    return [
        "env",
        f"LD_LIBRARY_PATH={EXPERIMENTAL_BIN_DIR}",
        "OMP_NUM_THREADS=1",
        "numactl",
        "--interleave=all",
        str(binary),
    ]


def _server_argv(
    *,
    binary: Path,
    target_model: Path,
    port: int,
    draft_model: Path | None = None,
    speculative: bool = False,
) -> list[str]:
    argv = _base_argv(binary)
    argv.extend(
        [
            "-m",
            str(target_model),
            "-t",
            str(DEFAULT_THREADS),
            "-np",
            "1",
            "-c",
            str(DEFAULT_CONTEXT),
            "-ub",
            str(DEFAULT_UBATCH),
            "-ngl",
            "0",
            "--device",
            "none",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
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
    )
    if speculative:
        if draft_model is None:
            raise ValueError("speculative Stage-1 plan requires a draft model path")
        argv.extend(
            [
                "-md",
                str(draft_model),
                "--spec-type",
                "draft-tree",
                "--spec-draft-n-max",
                str(DEFAULT_DRAFT_MAX),
                "--spec-draft-p-split",
                str(DEFAULT_SPEC_P_SPLIT),
                "--spec-draft-device",
                "ROCm0",
                "--spec-draft-ngl",
                "99",
            ]
        )
    else:
        argv.extend(["--spec-type", "none"])
    return argv


def _render_shell(argv: list[str]) -> str:
    return shlex.join(argv)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pick_ephemeral_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def is_pid_alive(pid: int) -> bool:
    probe = subprocess.run(
        ["ps", "-p", str(pid), "-o", "pid="],
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.returncode == 0 and probe.stdout.strip() == str(pid)


def port_is_listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def wait_for_health(port: int, timeout_s: int, pid: int | None = None) -> None:
    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if pid is not None and not is_pid_alive(pid):
            raise RuntimeError(f"server pid {pid} exited before health check on port {port}")
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = resp.read().decode("utf-8", errors="replace").strip().lower()
            if "ok" in body:
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"server on port {port} did not become healthy within {timeout_s}s")


def terminate_server(proc: subprocess.Popen[str], port: int) -> None:
    pid = proc.pid
    if pid is None:
        return
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        pgid = None

    def send(sig: int) -> None:
        if pgid is not None:
            try:
                os.killpg(pgid, sig)
                return
            except ProcessLookupError:
                return
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            return

    if proc.poll() is None:
        send(signal.SIGTERM)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and proc.poll() is None:
            time.sleep(0.2)

    if proc.poll() is None:
        send(signal.SIGKILL)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and proc.poll() is None:
            time.sleep(0.2)

    if proc.poll() is None:
        raise RuntimeError(f"failed to terminate server pid {pid}")
    if is_pid_alive(pid):
        raise RuntimeError(f"server pid {pid} still appears alive after termination")
    if port_is_listening(port):
        raise RuntimeError(f"server port {port} is still listening after pid {pid} terminated")

    log_handle = getattr(proc, "_stage1_log_handle", None)
    if log_handle is not None:
        log_handle.close()


def launch_server(argv: list[str], log_path: Path) -> subprocess.Popen[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            argv,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    except Exception:
        log_handle.close()
        raise
    proc._stage1_log_handle = log_handle  # type: ignore[attr-defined]
    return proc


def query_chat(
    *,
    port: int,
    prompt: str,
    max_tokens: int,
    timeout_s: int,
) -> tuple[dict[str, Any], str, float]:
    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "seed": 42,
        "stream": False,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=canonical_json(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    elapsed_s = time.monotonic() - started
    return json.loads(raw), raw, elapsed_s


def _numeric(mapping: dict[str, Any], key: str) -> float:
    value = mapping.get(key)
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _integer(mapping: dict[str, Any], key: str) -> int:
    return int(_numeric(mapping, key))


def _message_text(response: dict[str, Any]) -> dict[str, str]:
    choices = response.get("choices", [])
    choice = choices[0] if choices else {}
    message = choice.get("message", {}) if isinstance(choice, dict) else {}

    def coerce(value: Any) -> str:
        if isinstance(value, list):
            return "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in value
            )
        return str(value or "")

    return {
        "content": coerce(message.get("content", "")),
        "reasoning_content": coerce(message.get("reasoning_content", "")),
    }


def _server_version(binary: Path) -> str:
    result = subprocess.run(
        [str(binary), "--version"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    return (result.stdout + result.stderr).strip()


def latest_n5_summary_path() -> Path | None:
    root = RESEARCH_ROOT / "data" / "specdec_frontdoor_alpha"
    matches = sorted(root.glob(DEFAULT_N5_SUMMARY_GLOB), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def validate_n5_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    n5_arm = data.get("arms", {}).get("n5_spec_on", {})
    if not data.get("decision_grade"):
        raise ValueError(f"N5 summary is not decision_grade: {path}")
    if not n5_arm.get("status_ok"):
        raise ValueError(f"N5 n5_spec_on arm is not status_ok: {path}")
    if int(n5_arm.get("draft_accepted") or 0) <= 0:
        raise ValueError(f"N5 n5_spec_on arm has no accepted draft tokens: {path}")
    return data


def collect_process_snapshot() -> list[str]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,stat=,pcpu=,pmem=,etime=,args="],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    needles = ("llama", "autopilot", "benchmark", "glm52", "imatrix", "hf download")
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if any(needle in line for needle in needles)
    ]


def classify_stage1_response(record: dict[str, Any], log_text: str) -> str:
    if record.get("status") != "ok":
        return "error"
    lowered = log_text.lower()
    if "fallback" in lowered or "decode fail" in lowered:
        return "decode_failed_fallback"
    draft_n = int(record.get("draft_n") or 0)
    draft_accepted = int(record.get("draft_n_accepted") or 0)
    if draft_n > 0 and draft_accepted > 0:
        return "drafted_ok"
    if "--spec-type" not in " ".join(record.get("server_argv", [])) or "spec-type none" in lowered:
        return "no_spec_enabled"
    return "no_draft_tokens"


def summarize_arm(records: list[dict[str, Any]], log_text: str, *, speculative: bool) -> dict[str, Any]:
    ok_records = [record for record in records if record.get("status") == "ok"]
    prompt_ms = sum(_numeric(record.get("timings", {}), "prompt_ms") for record in ok_records)
    predicted_ms = sum(_numeric(record.get("timings", {}), "predicted_ms") for record in ok_records)
    prompt_n = sum(_integer(record.get("timings", {}), "prompt_n") for record in ok_records)
    predicted_n = sum(_integer(record.get("timings", {}), "predicted_n") for record in ok_records)
    request_duration_s = sum(float(record.get("request_duration_s") or 0.0) for record in ok_records)
    draft_n = sum(int(record.get("draft_n") or 0) for record in ok_records)
    draft_accepted = sum(int(record.get("draft_n_accepted") or 0) for record in ok_records)
    taxonomy_counts: dict[str, int] = {}
    for record in records:
        taxonomy = classify_stage1_response(record, log_text) if speculative else "spec_off_control"
        taxonomy_counts[taxonomy] = taxonomy_counts.get(taxonomy, 0) + 1
    return {
        "status_ok": len(ok_records) == len(records),
        "prompts": len(records),
        "ok_prompts": len(ok_records),
        "prompt_tokens": prompt_n,
        "completion_tokens": predicted_n,
        "prompt_ms": prompt_ms,
        "predicted_ms": predicted_ms,
        "prompt_per_second": (prompt_n * 1000.0 / prompt_ms) if prompt_ms > 0 else 0.0,
        "predicted_per_second": (predicted_n * 1000.0 / predicted_ms) if predicted_ms > 0 else 0.0,
        "request_duration_s": request_duration_s,
        "wall_tokens_per_second": (predicted_n / request_duration_s) if request_duration_s > 0 else 0.0,
        "draft_n": draft_n,
        "draft_n_accepted": draft_accepted,
        "acceptance_rate": (draft_accepted / draft_n) if draft_n > 0 else 0.0,
        "taxonomy_counts": taxonomy_counts,
    }


def build_execute_plan(args: argparse.Namespace, binary: Path, baseline_port: int, stage1_port: int) -> dict[str, Any]:
    return {
        "schema": "stage1_mi210_gpu_drafter_execute_plan.v1",
        "created_at": _utc_now(),
        "mode": "execute",
        "binary": str(binary),
        "server_version": _server_version(binary),
        "target_model": str(args.target_model.resolve()),
        "draft_model": str(args.draft_model.resolve()),
        "max_tokens": args.max_tokens,
        "min_completion_ratio": args.min_completion_ratio,
        "request_timeout_s": args.request_timeout,
        "startup_timeout_s": args.startup_timeout,
        "prompts": args.prompts,
        "arms": [
            {
                "name": BASELINE_ARM.name,
                "purpose": BASELINE_ARM.purpose,
                "port": baseline_port,
                "argv": _server_argv(
                    binary=binary,
                    target_model=args.target_model.resolve(),
                    port=baseline_port,
                ),
            },
            {
                "name": STAGE1_ARM.name,
                "purpose": STAGE1_ARM.purpose,
                "port": stage1_port,
                "argv": _server_argv(
                    binary=binary,
                    target_model=args.target_model.resolve(),
                    draft_model=args.draft_model.resolve(),
                    port=stage1_port,
                    speculative=True,
                ),
            },
        ],
    }


def build_manifest(
    args: argparse.Namespace,
    guards: GuardState,
    binary: Path,
) -> dict[str, Any]:
    baseline_argv = _server_argv(
        binary=binary,
        target_model=args.target_model.resolve(),
        port=args.baseline_port,
    )
    stage1_argv = _server_argv(
        binary=binary,
        target_model=args.target_model.resolve(),
        draft_model=args.draft_model.resolve(),
        port=args.stage1_port,
        speculative=True,
    )

    gate_status = "blocked" if guards.quiet_host_blockers else "ready"
    return {
        "schema": "stage1_mi210_gpu_drafter_plan.v1",
        "created_at": _utc_now(),
        "mode": "dry_run",
        "phase": "post_glm_frontdoor_external_gpu_drafter_ab",
        "binary": {
            "path": str(binary),
            "ld_library_path": str(EXPERIMENTAL_BIN_DIR),
            "production_refused": str(PRODUCTION_SERVER),
        },
        "prerequisites": {
            "glm_completion_required": True,
            "quiet_host_required": True,
            "n5_frontdoor_retest": {
                "required": True,
                "command": N5_PREREQUISITE_COMMAND,
                "purpose": "strict preflight before the Stage-1 external drafter A/B window",
            },
        },
        "guards": {
            "quiet_host": {
                "ready": guards.quiet_host_ready,
                "blockers": guards.quiet_host_blockers,
            }
        },
        "gate": {
            "gate_id": "stage1_frontdoor_external_gpu_drafter_ab",
            "title": "Stage-1 frontdoor external GPU drafter A/B",
            "status": gate_status,
            "dry_run_only": True,
            "pass_metadata": {
                "speedup_gte": PASS_SPEEDUP_THRESHOLD,
                "usable_acceptance_evidence_required": True,
                "usable_acceptance_evidence_examples": [
                    "summary artifact or server metrics show speculative acceptance rather than permanent fallback",
                    "accepted speculative tokens or acceptance ratio is present and non-zero",
                    "baseline and Stage-1 runs use the same target model and prompt pack",
                ],
            },
            "kill_metadata": {
                "kill_if": [
                    f"speedup < {PASS_SPEEDUP_THRESHOLD}",
                    "usable speculative acceptance evidence is missing",
                    "quiet-host blockers are present during the planned window",
                    "N5 strict retest prerequisite is not clean",
                ]
            },
            "acceptance_artifacts": [
                "manifest.json",
                "commands.sh",
                "gate_summary.json",
                "baseline summary/metrics",
                "stage1 summary/metrics",
                "acceptance evidence excerpt",
            ],
            "server_command_templates": [
                {
                    "arm": "baseline_cpu_target_no_spec",
                    "argv": baseline_argv,
                    "shell": _render_shell(baseline_argv),
                    "purpose": "CPU target only baseline with speculation disabled",
                },
                {
                    "arm": "stage1_cpu_target_mi210_external_drafter",
                    "argv": stage1_argv,
                    "shell": _render_shell(stage1_argv),
                    "purpose": "CPU target plus MI210 external drafter Stage-1 candidate",
                },
            ],
        },
    }


def build_gate_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    gate = manifest["gate"]
    guards = manifest["guards"]["quiet_host"]
    return {
        "gate_id": gate["gate_id"],
        "title": gate["title"],
        "phase": manifest["phase"],
        "status": gate["status"],
        "dry_run_only": True,
        "next_window": "after GLM completes and the host is quiet",
        "prerequisite_command": manifest["prerequisites"]["n5_frontdoor_retest"]["command"],
        "quiet_host_ready": guards["ready"],
        "quiet_host_blockers": guards["blockers"],
        "pass_speedup_gte": gate["pass_metadata"]["speedup_gte"],
        "acceptance_evidence_required": gate["pass_metadata"]["usable_acceptance_evidence_required"],
    }


def render_commands(manifest: dict[str, Any]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Dry-run package only. Review and execute manually in a post-GLM quiet window.",
        f'export LD_LIBRARY_PATH="{manifest["binary"]["ld_library_path"]}"',
        f'# pinned experimental binary: {manifest["binary"]["path"]}',
        f'# prerequisite strict harness: {manifest["prerequisites"]["n5_frontdoor_retest"]["command"]}',
        "",
    ]
    for command in manifest["gate"]["server_command_templates"]:
        lines.append(f'# arm: {command["arm"]}')
        lines.append(f'# purpose: {command["purpose"]}')
        lines.append(command["shell"])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_artifacts(output_dir: Path, manifest: dict[str, Any], gate_summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "commands.sh").write_text(render_commands(manifest), encoding="utf-8")
    (output_dir / "gate_summary.json").write_text(
        json.dumps(gate_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_arm(
    *,
    arm: ArmSpec,
    argv: list[str],
    port: int,
    output_dir: Path,
    prompts: list[str],
    max_tokens: int,
    request_timeout: int,
    startup_timeout: int,
) -> dict[str, Any]:
    short_name = "stage1" if arm.speculative else "baseline"
    log_path = output_dir / f"{short_name}.server.log"
    raw_path = output_dir / f"{short_name}.raw.json"
    result_path = output_dir / f"{short_name}.result.json"
    proc: subprocess.Popen[str] | None = None
    records: list[dict[str, Any]] = []
    started_at = _utc_now()
    startup_elapsed_s = 0.0
    try:
        started = time.monotonic()
        proc = launch_server(argv, log_path)
        wait_for_health(port, startup_timeout, pid=proc.pid)
        startup_elapsed_s = time.monotonic() - started
        for index, prompt in enumerate(prompts, start=1):
            response, raw_response, request_duration_s = query_chat(
                port=port,
                prompt=prompt,
                max_tokens=max_tokens,
                timeout_s=request_timeout,
            )
            message = _message_text(response)
            timings = response.get("timings", {})
            usage = response.get("usage", {})
            semantic_output = {
                "content": message["content"],
                "reasoning_content": message["reasoning_content"],
            }
            record = {
                "status": "ok",
                "prompt_index": index,
                "prompt": prompt,
                "response_sha256": sha256_text(canonical_json(response)),
                "output_sha256": sha256_text(canonical_json(semantic_output)),
                "content": message["content"],
                "reasoning_content": message["reasoning_content"],
                "usage": usage,
                "timings": timings,
                "draft_n": _integer(timings, "draft_n"),
                "draft_n_accepted": _integer(timings, "draft_n_accepted"),
                "request_duration_s": request_duration_s,
                "raw_response": raw_response,
            }
            records.append(record)
    except Exception as exc:
        records.append(
            {
                "status": "error",
                "error": str(exc),
                "prompt_index": len(records) + 1,
            }
        )
    finally:
        cleanup_error = None
        if proc is not None:
            try:
                terminate_server(proc, port)
            except Exception as exc:  # pragma: no cover - exercised by integration failures
                cleanup_error = str(exc)
        log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        summary = summarize_arm(records, log_text, speculative=arm.speculative)
        result = {
            "arm": arm.name,
            "purpose": arm.purpose,
            "started_at": started_at,
            "finished_at": _utc_now(),
            "port": port,
            "server_pid": proc.pid if proc is not None else None,
            "server_argv": argv,
            "startup_elapsed_s": startup_elapsed_s,
            "server_log": str(log_path),
            "raw_response_path": str(raw_path),
            "summary": summary,
            "records": [
                {key: value for key, value in record.items() if key != "raw_response"}
                for record in records
            ],
        }
        if cleanup_error:
            result["cleanup_error"] = cleanup_error
            result["summary"]["status_ok"] = False
        _write_json(
            raw_path,
            {
                "arm": arm.name,
                "responses": [
                    {
                        "prompt_index": record.get("prompt_index"),
                        "raw_response": record.get("raw_response"),
                    }
                    for record in records
                    if "raw_response" in record
                ],
            },
        )
        _write_json(result_path, result)
        if cleanup_error:
            raise RuntimeError(cleanup_error)
        return result


def run_execute(args: argparse.Namespace, binary: Path, guards: GuardState) -> dict[str, Any]:
    if not guards.quiet_host_ready:
        raise RuntimeError(f"quiet host blockers present: {guards.quiet_host_blockers}")
    if not args.target_model.exists():
        raise FileNotFoundError(f"target model not found: {args.target_model}")
    if not args.draft_model.exists():
        raise FileNotFoundError(f"draft model not found: {args.draft_model}")

    n5_summary_path = args.n5_summary or latest_n5_summary_path()
    if n5_summary_path is None:
        raise FileNotFoundError("no N5 summary found under data/specdec_frontdoor_alpha")
    n5_summary = validate_n5_summary(n5_summary_path)

    baseline_port = pick_ephemeral_port()
    stage1_port = pick_ephemeral_port()
    while stage1_port == baseline_port:
        stage1_port = pick_ephemeral_port()

    execute_plan = build_execute_plan(args, binary, baseline_port, stage1_port)
    _write_json(args.output_dir / "execute_plan.json", execute_plan)

    baseline_argv = execute_plan["arms"][0]["argv"]
    stage1_argv = execute_plan["arms"][1]["argv"]
    baseline_result = run_arm(
        arm=BASELINE_ARM,
        argv=baseline_argv,
        port=baseline_port,
        output_dir=args.output_dir,
        prompts=args.prompts,
        max_tokens=args.max_tokens,
        request_timeout=args.request_timeout,
        startup_timeout=args.startup_timeout,
    )
    stage1_result = run_arm(
        arm=STAGE1_ARM,
        argv=stage1_argv,
        port=stage1_port,
        output_dir=args.output_dir,
        prompts=args.prompts,
        max_tokens=args.max_tokens,
        request_timeout=args.request_timeout,
        startup_timeout=args.startup_timeout,
    )

    baseline_summary = baseline_result["summary"]
    stage1_summary = stage1_result["summary"]
    baseline_decode_tps = float(baseline_summary["predicted_per_second"])
    stage1_decode_tps = float(stage1_summary["predicted_per_second"])
    decode_speedup = (stage1_decode_tps / baseline_decode_tps) if baseline_decode_tps > 0 else 0.0
    baseline_wall_tps = float(baseline_summary["wall_tokens_per_second"])
    stage1_wall_tps = float(stage1_summary["wall_tokens_per_second"])
    wall_speedup = (stage1_wall_tps / baseline_wall_tps) if baseline_wall_tps > 0 else 0.0
    taxonomy = stage1_summary.get("taxonomy_counts", {})
    has_usable_draft = (
        int(stage1_summary.get("draft_n") or 0) > 0
        and int(stage1_summary.get("draft_n_accepted") or 0) > 0
        and int(taxonomy.get("drafted_ok") or 0) == len(args.prompts)
    )
    min_completion_tokens = int(len(args.prompts) * args.max_tokens * args.min_completion_ratio)
    enough_completion = (
        int(baseline_summary.get("completion_tokens") or 0) >= min_completion_tokens
        and int(stage1_summary.get("completion_tokens") or 0) >= min_completion_tokens
    )
    pass_gate = (
        bool(baseline_summary.get("status_ok"))
        and bool(stage1_summary.get("status_ok"))
        and enough_completion
        and decode_speedup >= PASS_SPEEDUP_THRESHOLD
        and has_usable_draft
    )
    summary = {
        "schema": "stage1_mi210_gpu_drafter_result.v1",
        "created_at": _utc_now(),
        "mode": "execute",
        "verdict": "pass" if pass_gate else "fail",
        "decision_grade": pass_gate,
        "pass_speedup_gte": PASS_SPEEDUP_THRESHOLD,
        "min_completion_tokens_per_arm": min_completion_tokens,
        "enough_completion": enough_completion,
        "decode_speedup": decode_speedup,
        "wall_speedup": wall_speedup,
        "usable_draft": has_usable_draft,
        "n5_summary": {
            "path": str(n5_summary_path),
            "decision_grade": n5_summary.get("decision_grade"),
            "n5_spec_on": n5_summary.get("arms", {}).get("n5_spec_on", {}),
        },
        "quiet_host": {
            "ready": guards.quiet_host_ready,
            "blockers": guards.quiet_host_blockers,
            "process_snapshot": collect_process_snapshot(),
        },
        "binary": execute_plan["binary"],
        "server_version": execute_plan["server_version"],
        "target_model": execute_plan["target_model"],
        "draft_model": execute_plan["draft_model"],
        "prompts": len(args.prompts),
        "max_tokens": args.max_tokens,
        "baseline": baseline_summary,
        "stage1": stage1_summary,
        "artifacts": {
            "manifest": str(args.output_dir / "manifest.json"),
            "commands": str(args.output_dir / "commands.sh"),
            "gate_summary": str(args.output_dir / "gate_summary.json"),
            "execute_plan": str(args.output_dir / "execute_plan.json"),
            "baseline_log": baseline_result["server_log"],
            "stage1_log": stage1_result["server_log"],
            "baseline_raw": baseline_result["raw_response_path"],
            "stage1_raw": stage1_result["raw_response_path"],
            "baseline_result": str(args.output_dir / "baseline.result.json"),
            "stage1_result": str(args.output_dir / "stage1.result.json"),
        },
    }
    _write_json(args.output_dir / "summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run-first Stage-1 MI210 GPU-drafter planner")
    parser.add_argument("--execute", action="store_true", help="Launch sequential fresh-server baseline/Stage-1 A/B")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for planner artifacts")
    parser.add_argument("--binary", type=Path, default=EXPERIMENTAL_SERVER, help="Pinned experimental llama-server binary")
    parser.add_argument("--target-model", type=Path, default=DEFAULT_TARGET_MODEL, help="Target model path")
    parser.add_argument("--draft-model", type=Path, default=DEFAULT_DRAFT_MODEL, help="External draft model path")
    parser.add_argument("--baseline-port", type=int, default=DEFAULT_BASELINE_PORT, help="Port used by the baseline template")
    parser.add_argument("--stage1-port", type=int, default=DEFAULT_STAGE1_PORT, help="Port used by the Stage-1 template")
    parser.add_argument("--n5-summary", type=Path, default=None, help="Decision-grade N5 summary JSON prerequisite")
    parser.add_argument("--prompt", action="append", dest="prompts", help="Prompt to include in the execute prompt pack")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max completion tokens per prompt")
    parser.add_argument(
        "--min-completion-ratio",
        type=float,
        default=DEFAULT_MIN_COMPLETION_RATIO,
        help="Minimum generated-token ratio per arm for decision-grade speed evidence",
    )
    parser.add_argument("--request-timeout", type=int, default=DEFAULT_REQUEST_TIMEOUT_S, help="HTTP request timeout per prompt")
    parser.add_argument("--startup-timeout", type=int, default=DEFAULT_STARTUP_TIMEOUT_S, help="Server startup timeout per arm")
    args = parser.parse_args(argv)
    if args.prompts is None:
        args.prompts = list(DEFAULT_PROMPT_PACK)
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if not (0 < args.min_completion_ratio <= 1):
        parser.error("--min-completion-ratio must be in (0, 1]")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    binary = validate_experimental_server(args.binary)
    guards = collect_guard_state()
    manifest = build_manifest(args, guards, binary)
    gate_summary = build_gate_summary(manifest)
    write_artifacts(args.output_dir, manifest, gate_summary)

    print("Stage-1 MI210 GPU-drafter planner")
    print("mode: dry_run")
    print(f"output_dir: {args.output_dir}")
    print(f"binary: {binary}")
    print(f"manifest: {args.output_dir / 'manifest.json'}")
    print(f"commands: {args.output_dir / 'commands.sh'}")
    print(f"gate_summary: {args.output_dir / 'gate_summary.json'}")
    print(f"quiet_host_ready: {guards.quiet_host_ready}")
    if not args.execute:
        print("Dry run only. No inference was launched.")
        return 0

    print("mode: execute")
    print(f"prompts: {len(args.prompts)}")
    print(f"max_tokens: {args.max_tokens}")
    summary = run_execute(args, binary, guards)
    print(f"execute_plan: {args.output_dir / 'execute_plan.json'}")
    print(f"summary: {args.output_dir / 'summary.json'}")
    print(f"verdict: {summary['verdict']}")
    print(f"decode_speedup: {summary['decode_speedup']:.3f}x")
    print(f"wall_speedup: {summary['wall_speedup']:.3f}x")
    return 0 if summary["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
