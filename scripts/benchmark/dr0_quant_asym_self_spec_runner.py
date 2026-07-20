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
import os
import signal
import shlex
import socket
import subprocess
import time
import urllib.request
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
DEFAULT_STARTUP_TIMEOUT_S = 900
DEFAULT_REQUEST_TIMEOUT_S = 600
DEFAULT_SEED = 42
DEFAULT_TEMPERATURE = 0.0
FH_NOT_OBSERVABLE = "not_observable_without_engine_telemetry"
PROCESS_PATTERN = "llama-bench|llama-server|llama-cli|llama-mtmd-cli"

TASK_CLASSES = [
    {
        "id": "repetitive_structured_generation",
        "quality_sanity": "strict repeated/structured output count",
        "prompt": (
            "Write exactly 64 lines. Each line must be valid JSON with keys index and status. "
            "Use index values 0 through 63 in order. Use status value READY on every line. "
            "Do not write any text before or after the JSON lines."
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
            "Review this function for bugs: def add(a, b): return a + b. Return exactly three "
            "concise bullets. State that there is no blocking bug and cite the addition invariant. "
            "Do not invent missing context."
        ),
    },
    {
        "id": "exact_format_strict_instruction",
        "quality_sanity": "exact prefix and item count",
        "prompt": (
            "Write exactly five lines. Each line must start with DR0- and contain exactly six words. "
            "Do not write any text before or after the five lines."
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


@dataclass(frozen=True)
class ArmVariant:
    id: str
    arm: Arm
    k: int | None


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


def execution_variants(args: argparse.Namespace) -> list[ArmVariant]:
    variants: list[ArmVariant] = []
    for arm in ARMS:
        if arm.role in {"drafter_alone", "combined"}:
            for k_value in args.k:
                variants.append(ArmVariant(id=f"{arm.id}_k{k_value}", arm=arm, k=k_value))
        else:
            variants.append(ArmVariant(id=arm.id, arm=arm, k=None))
    return variants


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


def pick_ephemeral_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def process_snapshot() -> dict[str, Any]:
    result = subprocess.run(
        ["pgrep", "-af", PROCESS_PATTERN],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [
        line
        for line in result.stdout.splitlines()
        if line.strip()
        and "pgrep -af" not in line
        and "/usr/local/bin/earlyoom" not in line
    ]
    return {
        "command": ["pgrep", "-af", PROCESS_PATTERN],
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "lines": lines,
    }


def rocm_smi_showpids() -> dict[str, Any]:
    result = subprocess.run(
        ["rocm-smi", "--showpids"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    return {
        "command": ["rocm-smi", "--showpids"],
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "kfd_pids_observed": "No KFD PIDs" not in result.stdout,
    }


def snapshot_pid_set(snapshot: dict[str, Any]) -> set[str]:
    pids: set[str] = set()
    for line in snapshot.get("lines", []):
        parts = str(line).split(maxsplit=1)
        if parts and parts[0].isdigit():
            pids.add(parts[0])
    return pids


def port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def merged_env(env: dict[str, str]) -> dict[str, str]:
    merged = os.environ.copy()
    existing_ld = merged.get("LD_LIBRARY_PATH", "")
    merged.update(env)
    if env.get("LD_LIBRARY_PATH") and existing_ld:
        merged["LD_LIBRARY_PATH"] = f"{env['LD_LIBRARY_PATH']}:{existing_ld}"
    return merged


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


def arm_argv(
    args: argparse.Namespace,
    arm: Arm,
    port: int,
    spec_draft_n_max: int | None = None,
) -> list[str]:
    draft_n_max = spec_draft_n_max if spec_draft_n_max is not None else args.spec_draft_n_max
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
            "off",
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
            str(draft_n_max),
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
        "off",
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
        str(draft_n_max),
    ]


def empty_quality_result(task_class: str) -> dict[str, Any]:
    return {
        "task_class": task_class,
        "status": "not_run",
        "pass": None,
        "checker": None,
        "details": None,
    }


def empty_arm_metrics(arm_id: str, task_ids: list[str]) -> dict[str, Any]:
    return {
        "arm": arm_id,
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
        "request_count": None,
        "error_count": None,
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
    variants = execution_variants(args)
    return {
        "schema": "epyc.dr0_quant_asym_self_spec.summary.v1",
        "created_at": utc_now(),
        "mode": "execute" if args.execute else "dry_run",
        "decision_grade": False,
        "observation_grade": False,
        "dry_run_only": not args.execute,
        "run_id": manifest["run_id"],
        "artifact_dir": str(args.output_dir),
        "arms": {variant.id: empty_arm_metrics(variant.id, task_ids) for variant in variants},
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
    variants = execution_variants(args)
    for index, variant in enumerate(variants):
        arm = variant.arm
        port = args.base_port + index
        argv = arm_argv(args, arm, port, spec_draft_n_max=variant.k)
        env = arm_env(arm)
        command_templates.append(
            {
                "arm": variant.id,
                "base_arm": arm.id,
                "role": arm.role,
                "purpose": arm.purpose,
                "k": variant.k,
                "fresh_server_required": True,
                "port": port,
                "env": env,
                "argv": argv,
                "shell": render_shell(argv, env),
                "expected_artifacts": [
                    f"{variant.id}.server.log",
                    f"{variant.id}.responses.jsonl",
                    f"{variant.id}.metrics.json",
                    f"{variant.id}.quality.json",
                    f"{variant.id}.cleanup.json",
                ],
            }
        )

    return {
        "schema": "epyc.dr0_quant_asym_self_spec.manifest.v1",
        "created_at": utc_now(),
        "run_id": run_id,
        "mode": "execute" if args.execute else "dry_run",
        "dry_run_only": not args.execute,
        "scope": "DR-0 quant-asymmetric self-spec accounting runner",
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
            "seed": args.seed,
            "temperature": args.temperature,
            "startup_timeout_s": args.startup_timeout,
            "request_timeout_s": args.request_timeout,
            "fixed_ports": args.fixed_ports,
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
        "execution_variants": [
            {
                "id": variant.id,
                "base_arm": variant.arm.id,
                "role": variant.arm.role,
                "device": variant.arm.device,
                "spec_type": variant.arm.spec_type,
                "k": variant.k,
            }
            for variant in variants
        ],
        "command_templates": command_templates,
        "live_port_policy": (
            "execute uses ephemeral ports by default; per-arm *.command.json files are the "
            "source of truth for actual live ports unless --fixed-ports is supplied"
        ),
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


def is_pid_alive(pid: int) -> bool:
    probe = subprocess.run(
        ["ps", "-p", str(pid), "-o", "pid="],
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.returncode == 0 and probe.stdout.strip() == str(pid)


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
                if resp.status == 200 and ("ok" in body or body == ""):
                    return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"server on port {port} did not become healthy within {timeout_s}s")


def launch_server(argv: list[str], env: dict[str, str], log_path: Path) -> subprocess.Popen[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            argv,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=merged_env(env),
            start_new_session=True,
        )
    except Exception:
        log_handle.close()
        raise
    proc._dr0_log_handle = log_handle  # type: ignore[attr-defined]
    return proc


def terminate_server(proc: subprocess.Popen[str]) -> dict[str, Any]:
    pid = proc.pid
    cleanup: dict[str, Any] = {
        "pid": pid,
        "terminated": False,
        "sigterm_sent": False,
        "sigkill_sent": False,
        "pid_alive_after": None,
        "returncode": None,
    }
    if pid is None:
        return cleanup
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
            pass

    if proc.poll() is None:
        send(signal.SIGTERM)
        cleanup["sigterm_sent"] = True
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and proc.poll() is None:
            time.sleep(0.2)

    if proc.poll() is None:
        send(signal.SIGKILL)
        cleanup["sigkill_sent"] = True
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and proc.poll() is None:
            time.sleep(0.2)

    cleanup["returncode"] = proc.poll()
    cleanup["pid_alive_after"] = is_pid_alive(pid)
    cleanup["terminated"] = cleanup["returncode"] is not None and not cleanup["pid_alive_after"]
    log_handle = getattr(proc, "_dr0_log_handle", None)
    if log_handle is not None:
        log_handle.close()
    if not cleanup["terminated"]:
        raise RuntimeError(f"failed to terminate server pid {pid}")
    return cleanup


def query_chat(
    port: int,
    prompt: str,
    max_tokens: int,
    temperature: float,
    seed: int,
    timeout_s: int,
) -> tuple[dict[str, Any], str, float]:
    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        "top_k": 1,
        "seed": seed,
        "stream": False,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=canonical_json(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    wall_s = time.perf_counter() - start
    return json.loads(raw), raw, wall_s


def response_content(response: dict[str, Any]) -> str:
    choices = response.get("choices", [])
    choice = choices[0] if choices else {}
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    content = message.get("content", "")
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content or "")


def response_reasoning_content(response: dict[str, Any]) -> str:
    choices = response.get("choices", [])
    choice = choices[0] if choices else {}
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    reasoning = message.get("reasoning_content", "")
    if isinstance(reasoning, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in reasoning)
    return str(reasoning or "")


def score_quality(task: dict[str, Any], content: str) -> dict[str, Any]:
    task_id = task["id"]
    stripped = content.strip()
    if task_id == "repetitive_structured_generation":
        lines = [line for line in stripped.splitlines() if line.strip()]
        parsed: list[dict[str, Any]] = []
        errors: list[str] = []
        for line_no, line in enumerate(lines, start=1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_no}: {exc}")
                continue
            if not isinstance(item, dict):
                errors.append(f"line {line_no}: not an object")
                continue
            parsed.append(item)
        indexes = [item.get("index") for item in parsed]
        statuses = [item.get("status") for item in parsed]
        keys_ok = all(set(item) == {"index", "status"} for item in parsed)
        indexes_ok = sorted(indexes) == list(range(64))
        passed = (
            len(lines) == 64
            and len(parsed) == 64
            and not errors
            and keys_ok
            and statuses == ["READY"] * 64
            and indexes_ok
        )
        return {
            "task_class": task_id,
            "status": "checked",
            "pass": passed,
            "checker": task["quality_sanity"],
            "details": {
                "line_count": len(lines),
                "json_object_count": len(parsed),
                "error_count": len(errors),
                "errors": errors[:5],
                "keys_exact": keys_ok,
                "all_status_ready": statuses == ["READY"] * len(statuses),
                "indexes_unique_0_to_63": indexes_ok,
            },
        }
    if task_id == "bounded_architect_reviewer_json_decision":
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            return {
                "task_class": task_id,
                "status": "checked",
                "pass": False,
                "checker": task["quality_sanity"],
                "details": {"json_ok": False, "error": str(exc)},
            }
        required = {"decision", "confidence", "rationale"}
        decision = parsed.get("decision") if isinstance(parsed, dict) else None
        confidence = parsed.get("confidence") if isinstance(parsed, dict) else None
        confidence_ok = isinstance(confidence, (int, float)) and 0.0 <= float(confidence) <= 1.0
        decision_ok = isinstance(decision, bool) or (
            isinstance(decision, str) and bool(decision.strip())
        )
        passed = (
            isinstance(parsed, dict)
            and set(parsed) == required
            and decision_ok
            and confidence_ok
            and isinstance(parsed.get("rationale"), str)
            and bool(parsed.get("rationale", "").strip())
        )
        return {
            "task_class": task_id,
            "status": "checked",
            "pass": passed,
            "checker": task["quality_sanity"],
            "details": {
                "json_ok": isinstance(parsed, dict),
                "keys": sorted(parsed) if isinstance(parsed, dict) else None,
                "keys_exact": set(parsed) == required if isinstance(parsed, dict) else False,
                "decision_ok": decision_ok,
                "confidence_ok": confidence_ok,
            },
        }
    if task_id == "short_code_review_no_bug_control":
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        lowered = stripped.lower()
        bullet_like = all(line.startswith(("-", "*")) or line[:2].isdigit() for line in lines)
        no_bug = "no blocking bug" in lowered or "no bug" in lowered or "no issue" in lowered
        invariant = "return a + b" in lowered or "addition" in lowered or "adds" in lowered
        passed = len(lines) == 3 and bullet_like and no_bug and invariant
        return {
            "task_class": task_id,
            "status": "checked",
            "pass": passed,
            "checker": task["quality_sanity"],
            "details": {
                "line_count": len(lines),
                "bullet_like": bullet_like,
                "states_no_bug": no_bug,
                "cites_invariant": invariant,
            },
        }
    if task_id == "exact_format_strict_instruction":
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        starts_ok = [line.startswith("DR0-") for line in lines]
        word_counts = [len(line.split()) for line in lines]
        passed = len(lines) == 5 and all(starts_ok) and word_counts == [6] * 5
        return {
            "task_class": task_id,
            "status": "checked",
            "pass": passed,
            "checker": task["quality_sanity"],
            "details": {
                "line_count": len(lines),
                "starts_ok": starts_ok,
                "word_counts": word_counts,
            },
        }
    raise ValueError(f"unknown task class: {task_id}")


def number_or_none(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def row_from_response(
    variant: ArmVariant,
    task: dict[str, Any],
    response: dict[str, Any],
    raw_response_path: Path,
    wall_s: float,
) -> dict[str, Any]:
    timings = response.get("timings", {}) if isinstance(response.get("timings"), dict) else {}
    usage = response.get("usage", {}) if isinstance(response.get("usage"), dict) else {}
    choices = response.get("choices", [])
    choice = choices[0] if choices else {}
    finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
    content = response_content(response)
    reasoning_content = response_reasoning_content(response)
    quality = score_quality(task, content)
    prompt_tokens = number_or_none(usage.get("prompt_tokens")) or number_or_none(timings.get("prompt_n"))
    generated_tokens = number_or_none(usage.get("completion_tokens")) or number_or_none(timings.get("predicted_n"))
    draft_tokens = int(number_or_none(timings.get("draft_n")) or 0)
    accepted_draft_tokens = int(number_or_none(timings.get("draft_n_accepted")) or 0)
    return {
        "arm": variant.id,
        "base_arm": variant.arm.id,
        "k": variant.k,
        "task_class": task["id"],
        "status": "ok",
        "wall_time_s": wall_s,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "draft_tokens": draft_tokens,
        "accepted_draft_tokens": accepted_draft_tokens,
        "alpha": accepted_draft_tokens / draft_tokens if draft_tokens else None,
        "prompt_time_s": (number_or_none(timings.get("prompt_ms")) or 0) / 1000.0,
        "decode_time_s": (number_or_none(timings.get("predicted_ms")) or 0) / 1000.0,
        "prompt_tps": number_or_none(timings.get("prompt_per_second")),
        "decode_tps": number_or_none(timings.get("predicted_per_second")),
        "finish_reason": finish_reason,
        "quality": quality,
        "content_len": len(content),
        "reasoning_content_len": len(reasoning_content),
        "content_sha256": sha256_text(content),
        "reasoning_content_sha256": sha256_text(reasoning_content),
        "response_sha256": sha256_text(canonical_json(response)),
        "raw_response_path": str(raw_response_path),
    }


def aggregate_arm_rows(variant: ArmVariant, rows: list[dict[str, Any]], load_wall_clock_s: float | None) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    error_rows = [row for row in rows if row.get("status") != "ok"]
    prompt_tokens = sum(int(row.get("prompt_tokens") or 0) for row in ok_rows)
    generated_tokens = sum(int(row.get("generated_tokens") or 0) for row in ok_rows)
    draft_tokens = sum(int(row.get("draft_tokens") or 0) for row in ok_rows)
    accepted_draft_tokens = sum(int(row.get("accepted_draft_tokens") or 0) for row in ok_rows)
    wall_time_s = sum(float(row.get("wall_time_s") or 0.0) for row in ok_rows)
    prompt_time_s = sum(float(row.get("prompt_time_s") or 0.0) for row in ok_rows)
    decode_time_s = sum(float(row.get("decode_time_s") or 0.0) for row in ok_rows)
    quality_results = [row.get("quality") for row in ok_rows if row.get("quality") is not None]
    quality_results.extend(
        {
            "task_class": row.get("task_class"),
            "status": "error",
            "pass": False,
            "checker": None,
            "details": {"error": row.get("error")},
        }
        for row in error_rows
    )
    return {
        "arm": variant.id,
        "base_arm": variant.arm.id,
        "status": "ok" if ok_rows and not error_rows else ("partial" if ok_rows else "error"),
        "fresh_server": True,
        "k": variant.k,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "draft_tokens": draft_tokens,
        "accepted_draft_tokens": accepted_draft_tokens,
        "alpha": accepted_draft_tokens / draft_tokens if draft_tokens else None,
        "wall_time_s": wall_time_s,
        "prompt_time_s": prompt_time_s,
        "decode_time_s": decode_time_s,
        "prompt_tps": prompt_tokens / prompt_time_s if prompt_time_s > 0 else None,
        "decode_tps": generated_tokens / decode_time_s if decode_time_s > 0 else None,
        "load_wall_clock_s": load_wall_clock_s,
        "request_count": len(rows),
        "error_count": len(error_rows),
        "quality_results": quality_results,
        "task_results": [
            {
                "task_class": row.get("task_class"),
                "status": row.get("status"),
                "finish_reason": row.get("finish_reason"),
                "content_sha256": row.get("content_sha256"),
                "content_len": row.get("content_len"),
                "reasoning_content_len": row.get("reasoning_content_len"),
                "quality_pass": row.get("quality", {}).get("pass")
                if isinstance(row.get("quality"), dict)
                else None,
            }
            for row in rows
        ],
    }


def fetch_metrics(port: int, timeout_s: int = 10) -> dict[str, Any]:
    url = f"http://127.0.0.1:{port}/metrics"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        return {"status": "ok", "url": url, "body": body}
    except Exception as exc:
        return {"status": "error", "url": url, "error": str(exc)}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def run_arm_variant(args: argparse.Namespace, variant: ArmVariant, port: int) -> dict[str, Any]:
    env = arm_env(variant.arm)
    argv = arm_argv(args, variant.arm, port, spec_draft_n_max=variant.k)
    log_path = args.output_dir / f"{variant.id}.server.log"
    responses_path = args.output_dir / f"{variant.id}.responses.jsonl"
    quality_path = args.output_dir / f"{variant.id}.quality.json"
    metrics_path = args.output_dir / f"{variant.id}.metrics.json"
    cleanup_path = args.output_dir / f"{variant.id}.cleanup.json"
    command_path = args.output_dir / f"{variant.id}.command.json"
    proc: subprocess.Popen[str] | None = None
    rows: list[dict[str, Any]] = []
    cleanup: dict[str, Any] = {"status": "not_started"}
    load_wall_clock_s: float | None = None
    command_path.write_text(
        json.dumps(
            {
                "arm": variant.id,
                "base_arm": variant.arm.id,
                "k": variant.k,
                "port": port,
                "env": env,
                "argv": argv,
                "shell": render_shell(argv, env),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        launch_start = time.perf_counter()
        proc = launch_server(argv, env, log_path)
        wait_for_health(port, args.startup_timeout, pid=proc.pid)
        load_wall_clock_s = time.perf_counter() - launch_start
        for task_index, task in enumerate(TASK_CLASSES):
            raw_response_path = args.output_dir / f"{variant.id}.{task['id']}.raw.json"
            try:
                response, raw_response, wall_s = query_chat(
                    port=port,
                    prompt=task["prompt"],
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    seed=args.seed + task_index,
                    timeout_s=args.request_timeout,
                )
                raw_response_path.write_text(raw_response, encoding="utf-8")
                rows.append(row_from_response(variant, task, response, raw_response_path, wall_s))
            except Exception as exc:
                rows.append(
                    {
                        "arm": variant.id,
                        "base_arm": variant.arm.id,
                        "k": variant.k,
                        "task_class": task["id"],
                        "status": "error",
                        "error": str(exc),
                    }
                )
        metrics_path.write_text(
            json.dumps(fetch_metrics(port), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        rows.append(
            {
                "arm": variant.id,
                "base_arm": variant.arm.id,
                "k": variant.k,
                "task_class": "server_startup",
                "status": "error",
                "error": str(exc),
            }
        )
    finally:
        if proc is not None:
            try:
                cleanup = terminate_server(proc)
                cleanup["port_open_after"] = port_is_open(port)
                cleanup["terminated"] = cleanup["terminated"] and not cleanup["port_open_after"]
                cleanup["status"] = "ok"
            except Exception as exc:
                cleanup = {
                    "status": "error",
                    "error": str(exc),
                    "pid": proc.pid,
                    "pid_alive_after": is_pid_alive(proc.pid) if proc.pid else None,
                    "port_open_after": port_is_open(port),
                }
        cleanup_path.write_text(json.dumps(cleanup, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(responses_path, rows)
    aggregate = aggregate_arm_rows(variant, rows, load_wall_clock_s)
    quality_path.write_text(
        json.dumps(aggregate["quality_results"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    aggregate["artifacts"] = {
        "command": str(command_path),
        "server_log": str(log_path),
        "responses": str(responses_path),
        "quality": str(quality_path),
        "metrics": str(metrics_path),
        "cleanup": str(cleanup_path),
    }
    aggregate["cleanup"] = cleanup
    return aggregate


def build_coarse_economics(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline = arms.get("cpu_high_quant_verifier_baseline", {})
    baseline_hashes = {
        row.get("task_class"): row.get("content_sha256")
        for row in baseline.get("task_results", [])
        if row.get("task_class")
    }
    rows: list[dict[str, Any]] = []
    baseline_decode_tps = baseline.get("decode_tps")
    baseline_decode_time_s = baseline.get("decode_time_s")
    for arm_id, metrics in sorted(arms.items()):
        if not arm_id.startswith("quant_asymmetric_combined_"):
            continue
        decode_tps = metrics.get("decode_tps")
        decode_time_s = metrics.get("decode_time_s")
        rows.append(
            {
                "arm": arm_id,
                "k": metrics.get("k"),
                "baseline_decode_tps": baseline_decode_tps,
                "combined_decode_tps": decode_tps,
                "decode_tps_ratio_vs_baseline": (
                    decode_tps / baseline_decode_tps
                    if isinstance(decode_tps, (int, float))
                    and isinstance(baseline_decode_tps, (int, float))
                    and baseline_decode_tps > 0
                    else None
                ),
                "coarse_decode_seconds_saved_vs_baseline": (
                    baseline_decode_time_s - decode_time_s
                    if isinstance(baseline_decode_time_s, (int, float))
                    and isinstance(decode_time_s, (int, float))
                    else None
                ),
                "target_output_match_vs_baseline": {
                    row.get("task_class"): row.get("content_sha256")
                    == baseline_hashes.get(row.get("task_class"))
                    for row in metrics.get("task_results", [])
                    if row.get("task_class") in baseline_hashes
                },
            }
        )
    return {
        "status": "coarse_speed_delta_only",
        "warning": "F(K) and H(K) are still not separately observable from current llama-server timings",
        "rows": rows,
    }


def validate_live_inputs(args: argparse.Namespace) -> None:
    missing = [
        str(path)
        for path in (args.binary, args.cpu_verifier_model, args.mi210_drafter_model)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("missing live-run input(s): " + ", ".join(missing))


def ensure_quiet_preflight(args: argparse.Namespace, pre_process: dict[str, Any], pre_rocm: dict[str, Any]) -> None:
    if args.allow_existing_processes:
        return
    blockers: list[str] = []
    if pre_process.get("lines"):
        blockers.append("existing llama-family process(es): " + "; ".join(pre_process["lines"][:5]))
    if pre_rocm.get("kfd_pids_observed"):
        blockers.append("existing ROCm KFD process(es)")
    if blockers:
        raise RuntimeError(
            "refusing contaminated DR-0 execute preflight; pass --allow-existing-processes "
            "only for a deliberately non-clean run: "
            + " | ".join(blockers)
        )


def run_execute(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    validate_live_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary_skeleton(args, manifest)
    pre_process = process_snapshot()
    pre_rocm = rocm_smi_showpids()
    ensure_quiet_preflight(args, pre_process, pre_rocm)
    launched_summaries: dict[str, dict[str, Any]] = {}
    for index, variant in enumerate(execution_variants(args)):
        port = args.base_port + index if args.fixed_ports else pick_ephemeral_port()
        print(f"DR-0 {variant.id}: launch port={port} k={variant.k}", flush=True)
        launched_summaries[variant.id] = run_arm_variant(args, variant, port)
        status = launched_summaries[variant.id].get("status")
        decode_tps = launched_summaries[variant.id].get("decode_tps")
        alpha = launched_summaries[variant.id].get("alpha")
        print(
            f"DR-0 {variant.id}: status={status} decode_tps={decode_tps} alpha={alpha}",
            flush=True,
        )
    post_process = process_snapshot()
    post_rocm = rocm_smi_showpids()
    summary["arms"] = launched_summaries
    quality_results = [
        quality
        for arm_summary in launched_summaries.values()
        for quality in arm_summary.get("quality_results", [])
    ]
    total_quality = len(quality_results)
    pass_quality = sum(1 for quality in quality_results if quality.get("pass") is True)
    summary["quality_gate"] = {
        "required": "quality sanity passes on every included row",
        "status": "pass" if total_quality and pass_quality == total_quality else "fail",
        "pass_count": pass_quality,
        "total_count": total_quality,
    }
    all_cleanup_ok = all(
        arm_summary.get("cleanup", {}).get("status") == "ok"
        and arm_summary.get("cleanup", {}).get("terminated") is True
        and arm_summary.get("cleanup", {}).get("port_open_after") is False
        for arm_summary in launched_summaries.values()
    )
    pre_pids = snapshot_pid_set(pre_process)
    post_pids = snapshot_pid_set(post_process)
    new_post_pids = sorted(post_pids - pre_pids)
    no_llama_process_leak = not new_post_pids if args.allow_existing_processes else not post_process.get("lines")
    no_kfd_pid_leak = (
        pre_rocm.get("kfd_pids_observed") == post_rocm.get("kfd_pids_observed")
        if args.allow_existing_processes
        else not post_rocm.get("kfd_pids_observed")
    )
    cleanup_pass = all_cleanup_ok and no_llama_process_leak and no_kfd_pid_leak
    summary["cleanup_proof"] = {
        "status": "pass" if cleanup_pass else "fail",
        "pre_process_snapshot": pre_process,
        "post_process_snapshot": post_process,
        "pre_rocm_smi_showpids": pre_rocm,
        "post_rocm_smi_showpids": post_rocm,
        "new_post_process_pids": new_post_pids,
        "no_llama_process_leak": no_llama_process_leak,
        "no_kfd_pid_leak": no_kfd_pid_leak,
    }
    summary["fh_accounting"]["E_alpha_K"] = {
        "status": "coarse_observed_speed_delta_only",
        "value": build_coarse_economics(launched_summaries),
        "unit": "aggregate_decode_seconds_and_tps_ratio",
    }
    summary["fh_accounting"]["accounting_verdict"] = "not_decision_grade_until_F_K_and_H_K_are_separately_observable"
    summary["decision_grade"] = False
    summary["observation_grade"] = (
        summary["quality_gate"]["status"] == "pass"
        and summary["cleanup_proof"]["status"] == "pass"
        and all(arm_summary.get("status") == "ok" for arm_summary in launched_summaries.values())
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DR-0 quant-asymmetric self-spec accounting runner"
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
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--startup-timeout", type=int, default=DEFAULT_STARTUP_TIMEOUT_S)
    parser.add_argument("--request-timeout", type=int, default=DEFAULT_REQUEST_TIMEOUT_S)
    parser.add_argument("--spec-draft-n-max", type=int, default=DEFAULT_SPEC_DRAFT_N_MAX)
    parser.add_argument("--k", type=int, action="append", default=None, help="K value for F/H accounting")
    parser.add_argument(
        "--fixed-ports",
        action="store_true",
        help="Use base-port+index for live execution instead of safer ephemeral ports",
    )
    parser.add_argument(
        "--allow-existing-processes",
        action="store_true",
        help="Allow execute mode to run with existing llama/KFD processes and mark cleanup relative to preflight",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Launch fresh sequential llama-server instances and run the DR-0 task slice",
    )
    args = parser.parse_args(argv)
    args.k = args.k or list(DEFAULT_K_VALUES)
    args.binary = validate_experimental_binary(args.binary)
    args.cpu_verifier_model = args.cpu_verifier_model.expanduser()
    args.mi210_drafter_model = args.mi210_drafter_model.expanduser()
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_manifest(args)
    summary = run_execute(args, manifest) if args.execute else build_summary_skeleton(args, manifest)
    write_artifacts(args, manifest, summary)
    print(
        json.dumps(
            {
                "status": "execute_complete" if args.execute else "dry_run_written",
                "output_dir": str(args.output_dir),
                "decision_grade": summary["decision_grade"],
                "quality_status": summary["quality_gate"]["status"],
                "cleanup_status": summary["cleanup_proof"]["status"],
            },
            sort_keys=True,
        )
    )
    if args.execute and summary["cleanup_proof"]["status"] != "pass":
        return 2
    if args.execute and any(arm.get("status") == "error" for arm in summary["arms"].values()):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
