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
import json
import shlex
import subprocess
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
PASS_SPEEDUP_THRESHOLD = 1.3
N5_PREREQUISITE_COMMAND = (
    f"{RESEARCH_ROOT}/scripts/benchmark/n5_frontdoor_drafter_retest.sh --strict"
)


@dataclass(frozen=True)
class GuardState:
    quiet_host_blockers: list[str]

    @property
    def quiet_host_ready(self) -> bool:
        return not self.quiet_host_blockers


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
        "-i",
        "PATH=/usr/bin:/bin",
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run-first Stage-1 MI210 GPU-drafter planner")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for planner artifacts")
    parser.add_argument("--binary", type=Path, default=EXPERIMENTAL_SERVER, help="Pinned experimental llama-server binary")
    parser.add_argument("--target-model", type=Path, default=DEFAULT_TARGET_MODEL, help="Target model path")
    parser.add_argument("--draft-model", type=Path, default=DEFAULT_DRAFT_MODEL, help="External draft model path")
    parser.add_argument("--baseline-port", type=int, default=DEFAULT_BASELINE_PORT, help="Port used by the baseline template")
    parser.add_argument("--stage1-port", type=int, default=DEFAULT_STAGE1_PORT, help="Port used by the Stage-1 template")
    return parser.parse_args(argv)


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
    print("Dry run only. No inference was launched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
