#!/usr/bin/env python3
"""Dry-run-first MI210 strategy gate planner.

This planner emits JSON plans for the MI210 strategy gates without launching
inference by default. The dry-run path is the primary contract:

  - `frontdoor_residency_p_gpu1`
  - `hybrid_moe_offload_cpu_experts`
  - `ngram_mtp_quality_monitoring_stub`

The commands are pinned to the experimental v7 build tree and never fall back
to the production v6 tree.
"""

from __future__ import annotations

import argparse
import json
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
EXPERIMENTAL_SERVER = EXPERIMENTAL_BIN_DIR / "llama-server"
EXPERIMENTAL_LD_LIBRARY_PATH = str(EXPERIMENTAL_BIN_DIR)
GLM_PATTERN = "hf download unsloth/GLM-5.2-GGUF"
AUTOPILOT_PATTERN = "scripts/autopilot/autopilot.py start"

DEFAULT_PORT = 18080
DEFAULT_THREADS = 96
DEFAULT_UBATCH = 512
DEFAULT_CONTEXT = 8192
DEFAULT_N_GPU_LAYERS = 99
DEFAULT_N_CPU_MOE = 32

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
            "GLM HF writer is active; pass --allow-glm-download only if cache contention is acceptable"
        )
        glm_download_blockers.extend(f"GLM match: {line}" for line in glm_matches)

    return GuardState(
        quiet_host_blockers=quiet_host_blockers,
        glm_download_active=glm_download_active,
        glm_download_blockers=glm_download_blockers,
    )


def _validated_experimental_server() -> Path:
    resolved = EXPERIMENTAL_SERVER.resolve()
    production = Path("/mnt/raid0/llm/llama.cpp/build-hip/bin/llama-server").resolve()
    if resolved == production:
        raise RuntimeError("refusing production v6 llama-server binary")
    if EXPERIMENTAL_ROOT not in resolved.parents and resolved.parent != EXPERIMENTAL_BIN_DIR:
        raise RuntimeError(f"refusing non-experimental binary: {resolved}")
    return resolved


def _base_argv() -> list[str]:
    return [
        "env",
        "-i",
        "PATH=/usr/bin:/bin",
        f"LD_LIBRARY_PATH={EXPERIMENTAL_LD_LIBRARY_PATH}",
        "OMP_NUM_THREADS=1",
        "numactl",
        "--interleave=all",
        str(_validated_experimental_server()),
    ]


def _server_argv(model: Path, port: int, ngl: int, extra: list[str] | None = None) -> list[str]:
    argv = _base_argv()
    argv.extend(
        [
            "-m",
            str(model),
            "-t",
            str(DEFAULT_THREADS),
            "-ub",
            str(DEFAULT_UBATCH),
            "-c",
            str(DEFAULT_CONTEXT),
            "-ngl",
            str(ngl),
            "-np",
            "1",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--metrics",
        ]
    )
    if extra:
        argv.extend(extra)
    return argv


def _template_shell(argv: list[str]) -> str:
    return shlex.join(argv)


def _gate_blockers(*groups: list[str]) -> list[str]:
    blockers: list[str] = []
    for group in groups:
        blockers.extend(group)
    return blockers


def _frontdoor_gate(model: Path | None, guards: GuardState) -> dict[str, Any]:
    blockers = _gate_blockers(guards.quiet_host_blockers, guards.glm_download_blockers)
    if model is None:
        blockers.append("frontdoor residency plan needs an explicit model path before execution")

    return {
        "gate_id": "frontdoor_residency_p_gpu1",
        "title": "Frontdoor residency / P-GPU-1",
        "status": "dry_run_only",
        "dry_run_only": True,
        "exact_command_known": False,
        "model_path": str(model) if model is not None else None,
        "guards": {
            "quiet_host_required": True,
            "glm_download_guard": True,
        },
        "blockers": blockers,
        "evidence_fields": [
            "model path",
            "stack stopped",
            "GPU idle",
            "command",
            "prompt set",
            "throughput",
            "quality pass/fail",
            "residency VRAM",
            "rollback note",
        ],
        "command_templates": [],
        "notes": [
            "Plan only: exact production command is not known yet.",
            "Use this plan as the evidence checklist for Gate R under P-GPU-1.",
        ],
    }


def _hybrid_gate(model: Path | None, skew_evidence: Path | None, force_no_skew_evidence: bool, guards: GuardState) -> dict[str, Any]:
    blockers = _gate_blockers(guards.quiet_host_blockers, guards.glm_download_blockers)
    if model is None:
        blockers.append("hybrid MoE offload requires --model PATH")

    skew_runner = SCRIPT_DIR / "expert_routing_skew_profile.sh"
    skew_evidence_present = bool(skew_evidence and skew_evidence.exists())
    if not skew_evidence_present and not force_no_skew_evidence:
        blockers.append(
            f"expert-routing-skew profile evidence missing; run {skew_runner} first or pass --force-no-skew-evidence"
        )

    command_templates: list[dict[str, Any]] = []
    if model is not None:
        mi210_argv = _server_argv(
            model,
            DEFAULT_PORT + 1,
            DEFAULT_N_GPU_LAYERS,
            extra=[
                "--device",
                "ROCm0",
                "-ot",
                "exps=CPU",
                "--n-cpu-moe",
                str(DEFAULT_N_CPU_MOE),
            ],
        )
        cpu_only_argv = _server_argv(
            model,
            DEFAULT_PORT + 2,
            0,
            extra=[
                "--device",
                "none",
                "--n-cpu-moe",
                str(DEFAULT_N_CPU_MOE),
            ],
        )
        command_templates = [
            {
                "arm": "mi210_hybrid_offload",
                "argv": mi210_argv,
                "shell": _template_shell(mi210_argv),
            },
            {
                "arm": "cpu_only_baseline",
                "argv": cpu_only_argv,
                "shell": _template_shell(cpu_only_argv),
            },
        ]

    status = "ready" if not blockers else "blocked"
    return {
        "gate_id": "hybrid_moe_offload_cpu_experts",
        "title": "Hybrid MoE offload / CPU experts",
        "status": status,
        "dry_run_only": False,
        "exact_command_known": True,
        "model_path": str(model) if model is not None else None,
        "guards": {
            "quiet_host_required": True,
            "glm_download_guard": True,
        },
        "blockers": blockers,
        "prerequisite_evidence": {
            "expert_routing_skew_profile": {
                "runner": str(skew_runner),
                "artifact": str(skew_evidence) if skew_evidence is not None else None,
                "present": skew_evidence_present,
                "required": True,
            }
        },
        "evidence_fields": [
            "model path",
            "skew profile evidence",
            "stack stopped",
            "GPU idle",
            "command",
            "throughput",
            "quality pass/fail",
            "rollback note",
        ],
        "command_templates": command_templates,
        "notes": [
            "Compare MI210 expert offload against the CPU-only baseline.",
            "The expert-routing-skew runner is a prerequisite and is not duplicated here.",
        ],
    }


def _ngram_gate(guards: GuardState) -> dict[str, Any]:
    blockers = _gate_blockers(guards.quiet_host_blockers, guards.glm_download_blockers)
    return {
        "gate_id": "ngram_mtp_quality_monitoring_stub",
        "title": "n-gram + MTP quality monitoring stub",
        "status": "dry_run_only",
        "dry_run_only": True,
        "exact_command_known": False,
        "guards": {
            "quiet_host_required": True,
            "glm_download_guard": True,
        },
        "blockers": blockers,
        "evidence_fields": [
            "prompt set",
            "command",
            "throughput",
            "quality pass/fail",
            "rollback note",
        ],
        "command_templates": [],
        "notes": [
            "This is a placeholder for the future quality-monitoring harness.",
            "Do not treat the gate as executable until a real monitoring command is defined.",
        ],
    }


def build_manifest(
    args: argparse.Namespace,
    guards: GuardState,
) -> dict[str, Any]:
    model = args.model
    skew_evidence = args.skew_evidence
    if skew_evidence is not None:
        skew_evidence = skew_evidence.resolve()

    gates = [
        _frontdoor_gate(model, guards),
        _hybrid_gate(model, skew_evidence, args.force_no_skew_evidence, guards),
        _ngram_gate(guards),
    ]

    return {
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": "execute" if args.execute else "dry_run",
            "experimental_root": str(EXPERIMENTAL_ROOT),
            "experimental_binary": str(EXPERIMENTAL_SERVER),
            "experimental_ld_library_path": EXPERIMENTAL_LD_LIBRARY_PATH,
            "quiet_host_required": True,
            "glm_download_pattern": GLM_PATTERN,
            "allow_glm_download": bool(args.allow_glm_download),
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
                "allowed": bool(args.allow_glm_download),
            },
        },
        "gates": gates,
    }


def render_commands(manifest: dict[str, Any]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f'export LD_LIBRARY_PATH="{EXPERIMENTAL_LD_LIBRARY_PATH}"',
        f'# experimental binary: {EXPERIMENTAL_SERVER}',
        "",
    ]
    for gate in manifest["gates"]:
        lines.append(f'# gate: {gate["gate_id"]}')
        if not gate["command_templates"]:
            lines.append(f'# {gate["title"]}: dry-run-only')
        for command in gate["command_templates"]:
            lines.append(f'# arm: {command["arm"]}')
            lines.append(command["shell"])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_artifacts(output_dir: Path, manifest: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    for gate in manifest["gates"]:
        (output_dir / f'{gate["gate_id"]}.json').write_text(json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "commands.sh").write_text(render_commands(manifest), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run-first MI210 strategy gate planner")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESEARCH_ROOT / "data" / "gpu-mi210" / f"mi210_strategy_gates_{_utc_stamp()}",
        help="Directory for manifest.json, per-gate JSON plans, and commands.sh",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Model path used by the hybrid MoE offload plan and, if provided, the frontdoor plan",
    )
    parser.add_argument(
        "--skew-evidence",
        type=Path,
        default=None,
        help="Artifact path from the expert-routing-skew profile runner",
    )
    parser.add_argument(
        "--force-no-skew-evidence",
        action="store_true",
        help="Allow the hybrid MoE plan to be emitted without skew evidence",
    )
    parser.add_argument(
        "--allow-glm-download",
        action="store_true",
        help="Permit execute mode while the GLM-5.2 HF writer is active",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Attempt to run executable gate commands after emitting plans",
    )
    return parser.parse_args(argv)


def _execution_blockers(manifest: dict[str, Any], args: argparse.Namespace) -> list[str]:
    blockers: list[str] = []
    guards = manifest["guards"]
    if guards["glm_download"]["active"] and not args.allow_glm_download:
        blockers.append("GLM HF writer is active")
    if not guards["quiet_host"]["ready"]:
        blockers.extend(guards["quiet_host"]["blockers"])
    for gate in manifest["gates"]:
        if gate["status"] == "blocked":
            for blocker in gate["blockers"]:
                if args.allow_glm_download and "GLM" in blocker:
                    continue
                blockers.append(blocker)
    return blockers


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    guards = collect_guard_state()
    manifest = build_manifest(args, guards)
    write_artifacts(args.output_dir, manifest)

    print("MI210 strategy gate planner")
    print(f"mode: {'execute' if args.execute else 'dry_run'}")
    print(f"output_dir: {args.output_dir}")
    print(f"experimental_binary: {EXPERIMENTAL_SERVER}")
    print(f"experimental_ld_library_path: {EXPERIMENTAL_LD_LIBRARY_PATH}")
    print(f"quiet_host_ready: {guards.quiet_host_ready}")
    print(f"glm_download_active: {guards.glm_download_active}")
    print(f"plan_file: {args.output_dir / 'manifest.json'}")
    print(f"commands_file: {args.output_dir / 'commands.sh'}")

    if not args.execute:
        print("Dry run only. No inference was launched.")
        return 0

    blockers = _execution_blockers(manifest, args)
    if blockers:
        print("Execution refused:")
        for blocker in blockers:
            print(f"  - {blocker}", file=sys.stderr)
        if guards.glm_download_active and not args.allow_glm_download:
            return 75
        return 1

    executable_gates = [gate for gate in manifest["gates"] if gate["command_templates"]]
    if not executable_gates:
        print("Execution refused: no executable command templates are defined yet.", file=sys.stderr)
        return 1

    for gate in executable_gates:
        for command in gate["command_templates"]:
            print(f"[execute] {gate['gate_id']} / {command['arm']}")
            subprocess.run(command["argv"], check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
