#!/usr/bin/env python3
"""Dry-run-first Bonsai-27B Q1_0 quality gate planner.

This planner turns the documented "Bonsai Q1 quality gate before role claim"
into an executable plan without loading any model. It writes a manifest and a
companion shell script that pin the experimental v7 build-hip tree only.

The gate is intentionally narrow:
  - model: Bonsai-27B-Q1_0.gguf
  - prompts: small deterministic prompt-obedience probes
  - command templates: CPU-only and MI210 experimental-v7 llama-cli invocations
  - default mode: dry-run only

No production v6 path is used as a default fallback.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_ROOT = SCRIPT_DIR.parent.parent
EXPERIMENTAL_ROOT = Path("/mnt/raid0/llm/llama.cpp-experimental")
EXPERIMENTAL_BIN_DIR = EXPERIMENTAL_ROOT / "build-hip" / "bin"
EXPERIMENTAL_LLAMA_CLI = EXPERIMENTAL_BIN_DIR / "llama-cli"
EXPERIMENTAL_LD_LIBRARY_PATH = str(EXPERIMENTAL_BIN_DIR)

MODEL_PATH = Path("/mnt/raid0/llm/models/bonsai-27b/Bonsai-27B-Q1_0.gguf")
DEFAULT_THREADS = 96
DEFAULT_CONTEXT = 2048
DEFAULT_MAX_TOKENS = 64
DEFAULT_CPU_NGL = 0
DEFAULT_MI210_NGL = 99
DEFAULT_TIMEOUT_S = 300
GLM_PATTERN = "hf download unsloth/GLM-5.2-GGUF"
AUTOPILOT_PATTERN = "scripts/autopilot/autopilot.py start"

PROBES = [
    {
        "id": "exact_ok",
        "prompt": "Return exactly: ok",
        "expected": "exact ok, no reasoning preamble",
        "max_tokens": 64,
    },
    {
        "id": "strict_json",
        "prompt": 'Return exactly this minified JSON and nothing else: {"status":"ok","model":"bonsai"}',
        "expected": 'exact minified JSON {"status":"ok","model":"bonsai"}, no markdown',
        "max_tokens": 96,
    },
    {
        "id": "simple_math",
        "prompt": "Answer with only the integer result: 37 + 58",
        "expected": "exact integer 95",
        "max_tokens": 32,
    },
    {
        "id": "short_instruction",
        "prompt": "In exactly six lowercase words, describe why benchmarks need held-out tests.",
        "expected": "six lowercase words, no punctuation-heavy explanation",
        "max_tokens": 48,
    },
]


class GuardState:
    def __init__(self, quiet_host_blockers: list[str], glm_download_active: bool, glm_download_blockers: list[str]):
        self.quiet_host_blockers = quiet_host_blockers
        self.glm_download_active = glm_download_active
        self.glm_download_blockers = glm_download_blockers

    @property
    def quiet_host_ready(self) -> bool:
        return not self.quiet_host_blockers


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _probe_pattern(pattern: str, runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> list[str]:
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
    quiet_host_blockers: list[str] = []
    autopilot_matches = _probe_pattern(AUTOPILOT_PATTERN, runner=runner)
    if autopilot_matches:
        quiet_host_blockers.append(
            f"quiet host guard blocked by pattern {AUTOPILOT_PATTERN!r}: {autopilot_matches[0]}"
        )
    llama_server_matches = _probe_process_basename("llama-server", runner=runner)
    if llama_server_matches:
        quiet_host_blockers.append(
            f"quiet host guard blocked by process basename 'llama-server': {llama_server_matches[0]}"
        )

    glm_matches = _probe_pattern(GLM_PATTERN, runner=runner)
    glm_download_active = bool(glm_matches)
    glm_download_blockers = []
    if glm_download_active:
        glm_download_blockers.append(
            "GLM HF writer is active; repeat the gate in a quieter window before any role claim"
        )
        glm_download_blockers.extend(f"GLM match: {line}" for line in glm_matches)

    return GuardState(
        quiet_host_blockers=quiet_host_blockers,
        glm_download_active=glm_download_active,
        glm_download_blockers=glm_download_blockers,
    )


def _validated_experimental_binary() -> Path:
    resolved = EXPERIMENTAL_LLAMA_CLI.resolve()
    production = Path("/mnt/raid0/llm/llama.cpp/build-hip/bin/llama-cli").resolve()
    if resolved == production:
        raise RuntimeError("refusing production v6 llama-cli binary")
    if EXPERIMENTAL_ROOT not in resolved.parents and resolved.parent != EXPERIMENTAL_BIN_DIR:
        raise RuntimeError(f"refusing non-experimental binary: {resolved}")
    return resolved


def _base_prefix() -> list[str]:
    return [
        "env",
        "-i",
        "PATH=/usr/bin:/bin",
        f"LD_LIBRARY_PATH={EXPERIMENTAL_LD_LIBRARY_PATH}",
        "OMP_NUM_THREADS=1",
        "numactl",
        "--interleave=all",
        str(_validated_experimental_binary()),
    ]


def _cpu_command(probe: dict[str, Any]) -> list[str]:
    return _base_prefix() + [
        "--device",
        "none",
        "--device-draft",
        "none",
        "--simple-io",
        "--no-warmup",
        "--single-turn",
        "--no-display-prompt",
        "--no-show-timings",
        "--color",
        "off",
        "-no-cnv",
        "--reasoning",
        "off",
        "--reasoning-budget",
        "0",
        "--temp",
        "0",
        "--seed",
        "1",
        "-m",
        str(MODEL_PATH),
        "-t",
        str(DEFAULT_THREADS),
        "-c",
        str(DEFAULT_CONTEXT),
        "-ngl",
        str(DEFAULT_CPU_NGL),
        "-n",
        str(probe.get("max_tokens", DEFAULT_MAX_TOKENS)),
        "-p",
        str(probe["prompt"]),
    ]


def _mi210_command(probe: dict[str, Any]) -> list[str]:
    return _base_prefix() + [
        "--device",
        "ROCm0",
        "--simple-io",
        "--no-warmup",
        "--single-turn",
        "--no-display-prompt",
        "--no-show-timings",
        "--color",
        "off",
        "-no-cnv",
        "--reasoning",
        "off",
        "--reasoning-budget",
        "0",
        "--temp",
        "0",
        "--seed",
        "1",
        "-m",
        str(MODEL_PATH),
        "-t",
        str(DEFAULT_THREADS),
        "-c",
        str(DEFAULT_CONTEXT),
        "-ngl",
        str(DEFAULT_MI210_NGL),
        "-n",
        str(probe.get("max_tokens", DEFAULT_MAX_TOKENS)),
        "-p",
        str(probe["prompt"]),
    ]


def _template_shell(argv: list[str]) -> str:
    return shlex.join(argv)


def _command_templates() -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    for probe in PROBES:
        cpu_argv = _cpu_command(probe)
        mi210_argv = _mi210_command(probe)
        commands.extend([
            {
                "arm": f"bonsai_q1_cpu_{probe['id']}",
                "device": "none",
                "role": "cpu_quality_baseline",
                "probe_id": probe["id"],
                "prompt": probe["prompt"],
                "expected": probe["expected"],
                "argv": cpu_argv,
                "shell": _template_shell(cpu_argv),
            },
            {
                "arm": f"bonsai_q1_mi210_{probe['id']}",
                "device": "ROCm0",
                "role": "gpu_quality_probe",
                "probe_id": probe["id"],
                "prompt": probe["prompt"],
                "expected": probe["expected"],
                "argv": mi210_argv,
                "shell": _template_shell(mi210_argv),
            },
        ])
    return commands


def build_manifest(guards: GuardState, *, execute: bool = False) -> dict[str, Any]:
    blockers = list(guards.quiet_host_blockers)
    blockers.extend(guards.glm_download_blockers)

    if not MODEL_PATH.is_file():
        blockers.append(f"missing Bonsai Q1_0 model artifact: {MODEL_PATH}")
    if not EXPERIMENTAL_LLAMA_CLI.is_file():
        blockers.append(f"missing experimental v7 llama-cli: {EXPERIMENTAL_LLAMA_CLI}")

    command_templates = _command_templates()
    status = "ready" if not blockers else "blocked"

    return {
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": "execute" if execute else "dry_run",
            "dry_run_only": not execute,
            "supports_execute": True,
            "experimental_root": str(EXPERIMENTAL_ROOT),
            "experimental_binary": str(EXPERIMENTAL_LLAMA_CLI),
            "experimental_ld_library_path": EXPERIMENTAL_LD_LIBRARY_PATH,
            "model_path": str(MODEL_PATH),
            "probe_count": len(PROBES),
            "role_claim_policy": (
                "Do not claim Bonsai-27B Q1_0 as role-ready until every CPU and MI210 "
                "probe satisfies its expected output without reasoning preambles, markdown, "
                "or extra explanatory text."
            ),
        },
        "guards": {
            "quiet_host": {
                "required": True,
                "blockers": guards.quiet_host_blockers,
                "ready": guards.quiet_host_ready,
            },
            "glm_download": {
                "required": True,
                "active": guards.glm_download_active,
                "blockers": guards.glm_download_blockers,
            },
        },
        "gate": {
            "gate_id": "bonsai_q1_role_claim_gate",
            "title": "Bonsai-27B Q1_0 quality/prompting gate",
            "status": status,
            "dry_run_only": not execute,
            "exact_command_known": True,
            "model_path": str(MODEL_PATH),
            "blockers": blockers,
            "acceptance_rule": (
                "Every command template must satisfy its probe-specific expected output "
                "and must not emit markdown, extra prose, or a reasoning preamble."
            ),
            "probes": PROBES,
            "evidence_fields": [
                "model path",
                "experimental binary path",
                "probe id",
                "prompt text",
                "expected output rule",
                "quiet host state",
                "command",
                "stdout transcript",
                "role-claim decision",
            ],
            "command_templates": command_templates,
            "notes": [
                "Dry-run mode does not load the model or launch inference.",
                "The commands are pinned to /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin.",
                "Command templates force completion mode (-no-cnv), disable reasoning, and suppress prompt/timing output so stdout is the generated text under test.",
                "The exact-ok prompt mirrors the staged Bonsai smoke commands; the additional probes test minimal instruction following before any role claim.",
            ],
        },
    }


def render_commands(manifest: dict[str, Any]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f'export LD_LIBRARY_PATH="{EXPERIMENTAL_LD_LIBRARY_PATH}"',
        f'# experimental binary: {EXPERIMENTAL_LLAMA_CLI}',
        "",
    ]
    gate = manifest["gate"]
    lines.append(f'# gate: {gate["gate_id"]}')
    for command in gate["command_templates"]:
        lines.append(f'# arm: {command["arm"]}')
        lines.append(command["shell"])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_artifacts(output_dir: Path, manifest: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "gate.json").write_text(
        json.dumps(manifest["gate"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "commands.sh").write_text(render_commands(manifest), encoding="utf-8")


def extract_generated_text(stdout: str, prompt: str | None = None) -> str:
    text = stdout.replace("\r\n", "\n")
    if prompt:
        marker = f"> {prompt}\n"
        if marker in text:
            text = text.split(marker, 1)[1]
    if "\n\nExiting..." in text:
        text = text.split("\n\nExiting...", 1)[0]
    return text


def normalize_stdout(stdout: str, prompt: str | None = None) -> str:
    if prompt is not None:
        stdout = extract_generated_text(stdout, prompt)
    return stdout.strip()


def evaluate_probe(probe_id: str, stdout: str, prompt: str | None = None) -> dict[str, Any]:
    text = normalize_stdout(stdout, prompt)
    if probe_id == "exact_ok":
        passed = text == "ok"
        reason = "exact ok" if passed else "stdout was not exactly ok"
    elif probe_id == "strict_json":
        expected = '{"status":"ok","model":"bonsai"}'
        passed = text == expected
        reason = "exact minified JSON" if passed else "stdout did not match exact minified JSON"
    elif probe_id == "simple_math":
        passed = text == "95"
        reason = "exact integer" if passed else "stdout was not exactly 95"
    elif probe_id == "short_instruction":
        words = text.split()
        passed = len(words) == 6 and all(re.fullmatch(r"[a-z]+", word) for word in words)
        reason = "six lowercase words" if passed else "stdout was not exactly six lowercase words"
    else:
        passed = False
        reason = f"unknown probe id: {probe_id}"
    return {
        "passed": passed,
        "reason": reason,
        "normalized_stdout": text,
        "generated_text": text,
    }


def run_arm(command: dict[str, Any], output_dir: Path, timeout_s: int) -> dict[str, Any]:
    arm = command["arm"]
    arm_dir = output_dir / "arms" / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = arm_dir / "stdout.txt"
    stderr_path = arm_dir / "stderr.txt"
    result_path = arm_dir / "result.json"

    record: dict[str, Any] = {
        "arm": arm,
        "probe_id": command["probe_id"],
        "device": command["device"],
        "role": command["role"],
        "prompt": command["prompt"],
        "expected": command["expected"],
        "command": command["shell"],
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "timeout_s": timeout_s,
    }
    try:
        completed = subprocess.run(
            command["argv"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        acceptance = evaluate_probe(command["probe_id"], completed.stdout, command["prompt"])
        record.update(
            {
                "returncode": completed.returncode,
                "status": "pass" if completed.returncode == 0 and acceptance["passed"] else "fail",
                "acceptance": acceptance,
            }
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        record.update(
            {
                "returncode": None,
                "status": "timeout",
                "acceptance": {
                    "passed": False,
                    "reason": "timeout",
                    "normalized_stdout": normalize_stdout(stdout, command["prompt"]),
                    "generated_text": normalize_stdout(stdout, command["prompt"]),
                },
            }
        )
    result_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return record


def run_execute(output_dir: Path, manifest: dict[str, Any], selected: list[str] | None, timeout_s: int) -> dict[str, Any]:
    commands = manifest["gate"]["command_templates"]
    if selected:
        selected_set = set(selected)
        commands = [command for command in commands if command["arm"] in selected_set]
    records = [run_arm(command, output_dir, timeout_s) for command in commands]
    passed = sum(1 for record in records if record["status"] == "pass")
    failed = sum(1 for record in records if record["status"] == "fail")
    timed_out = sum(1 for record in records if record["status"] == "timeout")
    summary = {
        "schema": "bonsai_q1_quality_gate_execute.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if passed == len(records) else "fail",
        "passed": passed,
        "failed": failed,
        "timed_out": timed_out,
        "total": len(records),
        "records": records,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run-first Bonsai Q1_0 quality gate planner")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESEARCH_ROOT / "data" / "bonsai_q1_quality_gate" / _utc_stamp(),
        help="Directory for manifest.json, gate.json, and commands.sh",
    )
    parser.add_argument("--execute", action="store_true", help="Run the generated command templates after writing the plan")
    parser.add_argument("--only", action="append", help="Arm name to execute. May be repeated. Defaults to all arms.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S, help="Per-arm timeout in seconds")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    guards = collect_guard_state()
    manifest = build_manifest(guards, execute=args.execute)
    write_artifacts(args.output_dir, manifest)

    print("Bonsai Q1_0 quality gate planner")
    print(f"mode: {'execute' if args.execute else 'dry_run'}")
    print(f"output_dir: {args.output_dir}")
    print(f"experimental_binary: {EXPERIMENTAL_LLAMA_CLI}")
    print(f"experimental_ld_library_path: {EXPERIMENTAL_LD_LIBRARY_PATH}")
    print(f"quiet_host_ready: {guards.quiet_host_ready}")
    print(f"glm_download_active: {guards.glm_download_active}")
    print(f"plan_file: {args.output_dir / 'manifest.json'}")
    print(f"commands_file: {args.output_dir / 'commands.sh'}")
    if not args.execute:
        print("Dry run only. No inference was launched.")
        return 0
    if manifest["gate"]["blockers"]:
        print("FATAL: execute blocked by guards:", file=sys.stderr)
        for blocker in manifest["gate"]["blockers"]:
            print(f"- {blocker}", file=sys.stderr)
        return 75

    known_arms = {command["arm"] for command in manifest["gate"]["command_templates"]}
    unknown = sorted(set(args.only or []) - known_arms)
    if unknown:
        print(f"FATAL: unknown --only arm(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    summary = run_execute(args.output_dir, manifest, args.only, args.timeout)
    print(f"summary_file: {args.output_dir / 'summary.json'}")
    print(f"passed: {summary['passed']}/{summary['total']}")
    print(f"status: {summary['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
