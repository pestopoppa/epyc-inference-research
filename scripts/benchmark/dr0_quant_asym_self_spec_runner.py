#!/usr/bin/env python3
"""DR-0 quant-asymmetric self-spec runner scaffold.

Dry-run mode only emits an execution bundle. It does not launch inference.
The future live runner is intentionally shaped around fresh sequential servers
per arm so DR-0 can account for identity, commands, metrics, quality, cleanup,
and F/H observability without depending on root docs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_ROOT = SCRIPT_DIR.parent.parent

EXPERIMENTAL_ROOT = Path("/mnt/raid0/llm/llama.cpp-experimental")
EXPERIMENTAL_BIN_DIR = EXPERIMENTAL_ROOT / "build-hip" / "bin"
EXPERIMENTAL_SERVER = EXPERIMENTAL_BIN_DIR / "llama-server"
PRODUCTION_ROOT = Path("/mnt/raid0/llm/llama.cpp")

DEFAULT_OUTPUT_DIR = (
    RESEARCH_ROOT
    / "data"
    / "dr0_quant_asym_self_spec"
    / f"dr0_quant_asym_self_spec_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
)
DEFAULT_CPU_VERIFIER_MODEL = Path(
    "/mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-Q4_K_M/"
    "Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf"
)
DEFAULT_MI210_DRAFTER_MODEL = Path(
    "/mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-IQ2_M/"
    "Qwen3.5-122B-A10B-UD-IQ2_M.gguf"
)
DEFAULT_BASE_PORT = 19730
DEFAULT_CONTEXT = 8192
DEFAULT_THREADS = 96
DEFAULT_UBATCH = 1024
DEFAULT_MAX_TOKENS = 256
DEFAULT_SPEC_DRAFT_N_MAX = 2
DEFAULT_K_VALUES = [1, 2, 4]
FH_NOT_OBSERVABLE = "not_observable_without_engine_telemetry"

TASK_CLASSES = [
    {
        "id": "repetitive_structured_generation",
        "quality_sanity": "strict repeated/structured output count",
        "prompt": (
            "Write exactly 64 lines. Each line must be valid JSON with keys index and status. "
            "Use status value READY on every line."
        ),
    },
    {
        "id": "bounded_architect_reviewer_json_decision",
        "quality_sanity": "single valid JSON decision object with required keys",
        "prompt": (
            "Return only JSON with keys decision, confidence, and rationale. Decide whether a "
            "default-off benchmark gate should run when cleanup telemetry is missing."
        ),
    },
    {
        "id": "short_code_review_no_bug_control",
        "quality_sanity": "states no blocking bug and cites the invariant",
        "prompt": (
            "Review this function for bugs: def add(a, b): return a + b. Return three concise "
            "bullets and do not invent missing context."
        ),
    },
    {
        "id": "exact_format_strict_instruction",
        "quality_sanity": "exact prefix and item count",
        "prompt": (
            "Write exactly five lines. Each line must start with DR0- and contain exactly six words."
        ),
    },
]

RECENT_OBSERVATIONS = [
    {
        "artifact": (
            "data/model_admission_throughput/"
            "qwen35_122b_iq2m_long_mtp_ab_local_20260719T013555Z/summary.json"
        ),
        "summary": "long repeated output: no-spec 37.87 t/s, draft-mtp 60.65 t/s, 511/511 accepted",
    },
    {
        "artifact": (
            "data/model_admission_throughput/"
            "qwen35_122b_iq2m_long_ngram_mtp_local_20260719T013814Z/summary.json"
        ),
        "summary": (
            "long repeated output: ngram-mod,draft-mtp 287.09 t/s, 746/746 accepted, "
            "quality_sanity=true"
        ),
    },
    {
        "artifact": (
            "data/model_admission_throughput/"
            "qwen35_122b_iq2m_mixed_ngram_mtp_ab_local_20260719T013943Z/summary.json"
        ),
        "summary": "mixed 3-prompt slice: no-spec 41.85 t/s, composed 50.77 t/s, 3/3 quality",
    },
    {
        "artifact": (
            "data/model_admission_throughput/"
            "qwen35_122b_iq2m_ngram_mtp_broad_20260719T014335Z/summary.json"
        ),
        "summary": "broad 8-prompt slice: composed mean 80.77 t/s, 1166/1440 accepted, 5/8 quality",
    },
]


@dataclass(frozen=True)
class Arm:
    id: str
    role: str
    device: str
    spec_type: str
    purpose: str


ARMS = [
    Arm(
        id="cpu_high_quant_verifier_baseline",
        role="verifier_baseline",
        device="cpu",
        spec_type="none",
        purpose="High-quant CPU verifier baseline for the selected task classes",
    ),
    Arm(
        id="mi210_aggressive_drafter_alone",
        role="drafter_alone",
        device="mi210",
        spec_type="ngram-mod,draft-mtp",
        purpose="MI210 resident aggressive same-family artifact, measured without CPU verifier",
    ),
    Arm(
        id="quant_asymmetric_combined",
        role="combined",
        device="cpu_plus_mi210",
        spec_type="draft-mtp",
        purpose="CPU 122B Q4 verifier with MI210 122B IQ2 MTP drafter, only valid when F/H are observable",
    ),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def render_shell(argv: list[str], env: dict[str, str] | None = None) -> str:
    if not env:
        return shlex.join(argv)
    return " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(env.items())) + " " + shlex.join(argv)


def validate_experimental_binary(binary: Path) -> Path:
    resolved = binary.expanduser().resolve()
    production_root = PRODUCTION_ROOT.resolve()
    experimental_root = EXPERIMENTAL_ROOT.resolve()
    if resolved == production_root or production_root in resolved.parents:
        raise ValueError(f"refusing production v6 path: {resolved}")
    if resolved != EXPERIMENTAL_SERVER.resolve() and experimental_root not in resolved.parents:
        raise ValueError(f"refusing non-experimental llama-server path: {resolved}")
    return resolved


def safe_stat(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError as exc:
        return {"path": str(path), "exists": False, "error": str(exc)}
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def git_identity(root: Path) -> dict[str, Any]:
    def run_git(args: list[str]) -> str | None:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    return {
        "root": str(root),
        "head": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["branch", "--show-current"]),
        "tracked_status_short": run_git(["status", "--short", "--untracked-files=no"]),
        "untracked_status": "omitted_from_manifest",
    }


def server_version(binary: Path) -> str | None:
    if not binary.exists():
        return None
    result = subprocess.run(
        [str(binary), "--version"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    return (result.stdout + result.stderr).strip()


def arm_env(arm: Arm) -> dict[str, str]:
    env = {
        "LD_LIBRARY_PATH": str(EXPERIMENTAL_BIN_DIR),
        "OMP_NUM_THREADS": "1",
        "PATH": "/usr/bin:/bin:/opt/rocm/bin",
    }
    if "mi210" in arm.device:
        env["HIP_VISIBLE_DEVICES"] = "0"
        env["ROCR_VISIBLE_DEVICES"] = "0"
    return env


def arm_argv(args: argparse.Namespace, arm: Arm, port: int) -> list[str]:
    if arm.role == "verifier_baseline":
        return [
            str(args.binary),
            "-m",
            str(args.cpu_verifier_model),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "-np",
            "1",
            "-c",
            str(args.context),
            "-t",
            str(args.threads),
            "-ub",
            str(args.ubatch),
            "--metrics",
            "--slots",
            "--jinja",
            "--reasoning",
            "auto",
            "--device",
            "none",
            "-ngl",
            "0",
            "--spec-type",
            "none",
        ]
    if arm.role == "drafter_alone":
        return [
            str(args.binary),
            "-m",
            str(args.mi210_drafter_model),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "-np",
            "1",
            "-c",
            str(args.context),
            "-t",
            "32",
            "-ub",
            str(args.ubatch),
            "--metrics",
            "--slots",
            "--jinja",
            "--reasoning",
            "off",
            "--device",
            "ROCm0",
            "-ngl",
            "all",
            "-ctk",
            "q4_0",
            "-ctv",
            "f16",
            "-fa",
            "on",
            "--spec-type",
            "ngram-mod,draft-mtp",
            "--spec-draft-n-max",
            str(args.spec_draft_n_max),
        ]
    return [
        str(args.binary),
        "-m",
        str(args.cpu_verifier_model),
        "-md",
        str(args.mi210_drafter_model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-np",
        "1",
        "-c",
        str(args.context),
        "-t",
        str(args.threads),
        "-ub",
        str(args.ubatch),
        "--metrics",
        "--slots",
        "--jinja",
        "--reasoning",
        "auto",
        "--device",
        "none",
        "-ngl",
        "0",
        "--spec-type",
        "draft-mtp",
        "--spec-draft-device",
        "ROCm0",
        "--spec-draft-ngl",
        "all",
        "--spec-draft-n-max",
        str(args.spec_draft_n_max),
    ]


def empty_quality_result(task_class: str) -> dict[str, Any]:
    return {
        "task_class": task_class,
        "status": "not_run",
        "pass": None,
        "checker": None,
        "details": None,
    }


def empty_arm_metrics(arm: Arm, task_ids: list[str]) -> dict[str, Any]:
    return {
        "arm": arm.id,
        "status": "not_run",
        "fresh_server": True,
        "prompt_tokens": None,
        "generated_tokens": None,
        "draft_tokens": None,
        "accepted_draft_tokens": None,
        "alpha": None,
        "wall_time_s": None,
        "prompt_time_s": None,
        "decode_time_s": None,
        "prompt_tps": None,
        "decode_tps": None,
        "load_wall_clock_s": None,
        "quality_results": [empty_quality_result(task_id) for task_id in task_ids],
    }


def fh_accounting_stub() -> dict[str, Any]:
    return {
        "rule": "E(alpha,K) > F(K)+H(K)",
        "k_values": None,
        "E_alpha_K": {
            "status": "not_run",
            "value": None,
            "unit": "seconds_saved_or_tokens_equivalent",
        },
        "F_K": {
            "status": FH_NOT_OBSERVABLE,
            "value": None,
            "unit": None,
            "required_engine_telemetry": [
                "per-speculation-window verifier token-evaluation time for proposed draft tokens",
                "accepted vs rejected draft-token verifier work separated from normal target decode",
                "per-K verifier batch shape/cycle counts",
            ],
        },
        "H_K": {
            "status": FH_NOT_OBSERVABLE,
            "value": None,
            "unit": None,
            "required_engine_telemetry": [
                "per-speculation-window scheduler/coordination overhead",
                "draft proposal latency separated from verifier compute",
                "HTTP/request framing overhead excluded or separately timed from engine decode",
            ],
        },
        "observable_from_llama_server_response_today": [
            "prompt_tokens",
            "generated_tokens",
            "draft_tokens",
            "accepted_draft_tokens",
            "alpha",
            "wall_time_s",
            "prompt_time_s",
            "decode_time_s",
            "prompt_tps",
            "decode_tps",
            "quality_results",
        ],
        "accounting_verdict": "not_evaluable_until_F_and_H_are_observable",
    }


def build_summary_skeleton(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    task_ids = [task["id"] for task in manifest["task_classes"]]
    fh_accounting = fh_accounting_stub()
    fh_accounting["k_values"] = args.k
    return {
        "schema": "epyc.dr0_quant_asym_self_spec.summary.v1",
        "created_at": utc_now(),
        "mode": "dry_run",
        "decision_grade": False,
        "observation_grade": False,
        "dry_run_only": True,
        "run_id": manifest["run_id"],
        "artifact_dir": str(args.output_dir),
        "arms": {arm.id: empty_arm_metrics(arm, task_ids) for arm in ARMS},
        "quality_gate": {
            "required": "quality sanity passes on every included row",
            "status": "not_run",
            "pass_count": None,
            "total_count": None,
        },
        "fh_accounting": fh_accounting,
        "cleanup_proof": {
            "status": "not_run",
            "pre_process_snapshot": None,
            "post_process_snapshot": None,
            "pre_rocm_smi_showpids": None,
            "post_rocm_smi_showpids": None,
            "no_llama_process_leak": None,
            "no_kfd_pid_leak": None,
        },
        "recent_observation_inputs": RECENT_OBSERVATIONS,
        "artifacts": {
            "manifest": str(args.output_dir / "manifest.json"),
            "commands": str(args.output_dir / "commands.sh"),
            "summary": str(args.output_dir / "summary.json"),
        },
    }


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    run_id = args.output_dir.name
    command_templates = []
    for index, arm in enumerate(ARMS):
        port = args.base_port + index
        argv = arm_argv(args, arm, port)
        env = arm_env(arm)
        command_templates.append(
            {
                "arm": arm.id,
                "role": arm.role,
                "purpose": arm.purpose,
                "fresh_server_required": True,
                "port": port,
                "env": env,
                "argv": argv,
                "shell": render_shell(argv, env),
                "expected_artifacts": [
                    f"{arm.id}.server.log",
                    f"{arm.id}.responses.jsonl",
                    f"{arm.id}.metrics.json",
                    f"{arm.id}.quality.json",
                    f"{arm.id}.cleanup.json",
                ],
            }
        )

    return {
        "schema": "epyc.dr0_quant_asym_self_spec.manifest.v1",
        "created_at": utc_now(),
        "run_id": run_id,
        "mode": "dry_run",
        "dry_run_only": True,
        "scope": "DR-0 quant-asymmetric self-spec accounting runner scaffold",
        "source_run_sheet": (
            "/mnt/raid0/llm/epyc-root/docs/reference/"
            "mi210-axa-dr0-run-sheets-2026-07-19.md"
        ),
        "identity": {
            "research": git_identity(RESEARCH_ROOT),
            "llama_cpp_experimental": git_identity(EXPERIMENTAL_ROOT),
            "server_binary": {
                **safe_stat(args.binary),
                "version": server_version(args.binary),
                "production_v6_refused": str(PRODUCTION_ROOT),
            },
            "models": {
                "cpu_verifier": safe_stat(args.cpu_verifier_model),
                "mi210_drafter": safe_stat(args.mi210_drafter_model),
            },
        },
        "task_classes": TASK_CLASSES,
        "parameters": {
            "context": args.context,
            "threads": args.threads,
            "ubatch": args.ubatch,
            "max_tokens": args.max_tokens,
            "spec_draft_n_max": args.spec_draft_n_max,
            "k_values": args.k,
        },
        "arms": [
            {
                "id": arm.id,
                "role": arm.role,
                "device": arm.device,
                "spec_type": arm.spec_type,
                "purpose": arm.purpose,
            }
            for arm in ARMS
        ],
        "command_templates": command_templates,
        "required_live_artifacts": [
            "manifest.json",
            "commands.sh",
            "summary.json",
            "per-arm server command/env/identity",
            "per-arm metrics: prompt/generated/draft/accepted tokens, alpha, wall/prompt/decode times",
            "per-arm quality results for every included task class",
            "pre/post pgrep process snapshots",
            "pre/post rocm-smi --showpids cleanup proof",
            "F/H accounting fields with observable values or machine-readable not-observable markers",
        ],
        "recent_observation_inputs": RECENT_OBSERVATIONS,
    }


def render_commands(manifest: dict[str, Any]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# DR-0 dry-run bundle. These are future live-run templates; this script does not run them.",
        "# Production v6 is intentionally absent. Use only llama.cpp-experimental.",
        "pgrep -af 'llama-bench|llama-server|llama-cli|llama-mtmd-cli' || true",
        "rocm-smi --showpids || true",
        "",
    ]
    for command in manifest["command_templates"]:
        lines.append(f'# arm: {command["arm"]}')
        lines.append(f'# purpose: {command["purpose"]}')
        lines.append(command["shell"])
        lines.append("")
        lines.append("pgrep -af 'llama-bench|llama-server|llama-cli|llama-mtmd-cli' || true")
        lines.append("rocm-smi --showpids || true")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_artifacts(args: argparse.Namespace, manifest: dict[str, Any], summary: dict[str, Any]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    commands_path = args.output_dir / "commands.sh"
    commands_path.write_text(render_commands(manifest), encoding="utf-8")
    commands_path.chmod(0o755)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DR-0 quant-asymmetric self-spec accounting runner scaffold"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--binary", type=Path, default=EXPERIMENTAL_SERVER)
    parser.add_argument("--cpu-verifier-model", type=Path, default=DEFAULT_CPU_VERIFIER_MODEL)
    parser.add_argument("--mi210-drafter-model", type=Path, default=DEFAULT_MI210_DRAFTER_MODEL)
    parser.add_argument("--base-port", type=int, default=DEFAULT_BASE_PORT)
    parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--ubatch", type=int, default=DEFAULT_UBATCH)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--spec-draft-n-max", type=int, default=DEFAULT_SPEC_DRAFT_N_MAX)
    parser.add_argument("--k", type=int, action="append", default=None, help="K value for F/H accounting")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Reserved for the live runner; currently refuses to launch inference",
    )
    args = parser.parse_args(argv)
    args.k = args.k or list(DEFAULT_K_VALUES)
    args.binary = validate_experimental_binary(args.binary)
    args.cpu_verifier_model = args.cpu_verifier_model.expanduser()
    args.mi210_drafter_model = args.mi210_drafter_model.expanduser()
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.execute:
        raise SystemExit(
            "--execute is not implemented in this DR-0 scaffold; dry-run bundle generation only"
        )
    manifest = build_manifest(args)
    summary = build_summary_skeleton(args, manifest)
    write_artifacts(args, manifest, summary)
    print(json.dumps({"status": "dry_run_written", "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
