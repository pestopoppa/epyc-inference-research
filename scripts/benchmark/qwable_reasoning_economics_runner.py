#!/usr/bin/env python3
"""Qwable reasoning-economics runner.

Defaults to a dry-run plan. The dry-run writes a JSON plan plus a companion
command file for the experimental v7 build. Pass --execute to run the selected
smoke arm(s); by default this remains the first, minimal IQ4 arm.

The runner is intentionally narrow:
  - the llama-server binary is pinned to /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin
  - LD_LIBRARY_PATH is sanitized and pinned to that same experimental bin dir
  - production v6 is never used as a default fallback
  - GLM download contention blocks execute mode unless --allow-glm-download is set
  - the concrete arms cover standalone, CPU baseline, scaffold, and verifier-selector stubs
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_ROOT = SCRIPT_DIR.parent.parent
EXPERIMENTAL_ROOT = Path("/mnt/raid0/llm/llama.cpp-experimental")
EXPERIMENTAL_BIN_DIR = EXPERIMENTAL_ROOT / "build-hip" / "bin"
SERVER_BIN = EXPERIMENTAL_BIN_DIR / "llama-server"
SERVER_LIB_DIR = EXPERIMENTAL_BIN_DIR

MODEL_DIR = Path("/mnt/raid0/llm/models/Qwable-v1-GGUF")
MODEL_IQ4_XS = MODEL_DIR / "Qwable-v1.IQ4_XS.gguf"
MODEL_Q8_0 = MODEL_DIR / "Qwable-v1.Q8_0.gguf"

GLM_PATTERN = "hf download unsloth/GLM-5.2-GGUF"
DEFAULT_OUTPUT_DIR = RESEARCH_ROOT / "data" / "qwable_reasoning_economics" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

DEFAULT_THREADS = 96
DEFAULT_CONTEXT = 8192
DEFAULT_MAX_TOKENS = 64
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = 42
DEFAULT_PORT_BASE = 18700
DEFAULT_REQUEST_TIMEOUT_S = 120
DEFAULT_STARTUP_TIMEOUT_S = 180

SANITIZED_ENV = {
    "HOME": "/tmp",
    "LD_LIBRARY_PATH": str(SERVER_LIB_DIR),
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PATH": "/usr/bin:/bin",
}


@dataclasses.dataclass(frozen=True)
class ArmSpec:
    name: str
    model_path: Path
    device: str
    ngl: int
    role: str
    prompt: str
    resource_class: str
    residency_note: str
    co_residency_policy: str
    beneficiary_policy: str
    response_format: dict[str, Any] | None = None
    json_schema: dict[str, Any] | None = None


ARMS: tuple[ArmSpec, ...] = (
    ArmSpec(
        name="standalone_iq4_gpu",
        model_path=MODEL_IQ4_XS,
        device="ROCm0",
        ngl=99,
        role="standalone_reasoner",
        prompt=(
            "Return a compact JSON object with keys arm, quant, and role using values "
            '"standalone_iq4_gpu", "IQ4_XS", and "reasoner".'
        ),
        resource_class="gpu_iq4",
        residency_note="IQ4 can co-reside more plausibly than Q8.",
        co_residency_policy="co-resident-plausible",
        beneficiary_policy="can share the host with a smaller beneficiary more plausibly than Q8",
    ),
    ArmSpec(
        name="standalone_q8_gpu",
        model_path=MODEL_Q8_0,
        device="ROCm0",
        ngl=99,
        role="standalone_reasoner",
        prompt=(
            "Return a compact JSON object with keys arm, quant, and role using values "
            '"standalone_q8_gpu", "Q8_0", and "reasoner".'
        ),
        resource_class="gpu_q8",
        residency_note="Q8 is the heavier standalone arm.",
        co_residency_policy="sequential_only",
        beneficiary_policy="smaller-beneficiary-only",
    ),
    ArmSpec(
        name="strict_iq4_json_gpu",
        model_path=MODEL_IQ4_XS,
        device="ROCm0",
        ngl=99,
        role="strict_json_reasoner",
        prompt=(
            'Return exactly this minified JSON and no markdown: '
            '{"arm":"strict_iq4_json_gpu","quant":"IQ4_XS","role":"reasoner"}'
        ),
        resource_class="gpu_iq4_strict_json",
        residency_note="Prompt/template strict-output probe without sampler grammar.",
        co_residency_policy="co-resident-plausible",
        beneficiary_policy="can share the host with a smaller beneficiary more plausibly than Q8",
    ),
    ArmSpec(
        name="strict_iq4_schema_gpu",
        model_path=MODEL_IQ4_XS,
        device="ROCm0",
        ngl=99,
        role="schema_json_reasoner",
        prompt=(
            "Return a JSON object for the Qwable schema gate with arm, quant, and role."
        ),
        resource_class="gpu_iq4_schema_json",
        residency_note="Sampler/schema constrained strict-output probe after K22 grammar-prefill fix.",
        co_residency_policy="co-resident-plausible",
        beneficiary_policy="can share the host with a smaller beneficiary more plausibly than Q8",
        json_schema={
            "type": "object",
            "properties": {
                "arm": {"type": "string", "enum": ["strict_iq4_schema_gpu"]},
                "quant": {"type": "string", "enum": ["IQ4_XS"]},
                "role": {"type": "string", "enum": ["reasoner"]},
            },
            "required": ["arm", "quant", "role"],
            "additionalProperties": False,
        },
    ),
    ArmSpec(
        name="cpu_iq4_baseline",
        model_path=MODEL_IQ4_XS,
        device="none",
        ngl=0,
        role="cpu_baseline",
        prompt=(
            "Return a compact JSON object with keys arm, quant, and role using values "
            '"cpu_iq4_baseline", "IQ4_XS", and "baseline".'
        ),
        resource_class="cpu_iq4",
        residency_note="CPU baseline establishes the lower-bound economics floor.",
        co_residency_policy="baseline_only",
        beneficiary_policy="not_a_co_residency_candidate",
    ),
    ArmSpec(
        name="scaffold_then_beneficiary_stub",
        model_path=MODEL_IQ4_XS,
        device="ROCm0",
        ngl=99,
        role="scaffold_generator_stub",
        prompt=(
            "Draft a minimal scaffold plan, then tag the beneficiary path as a stub. "
            "Return compact JSON with keys arm, scaffold, and beneficiary."
        ),
        resource_class="hybrid_stub_iq4",
        residency_note="Stub arm for the eventual scaffold-plus-beneficiary hybrid characterization.",
        co_residency_policy="co-resident-plausible",
        beneficiary_policy="smaller-beneficiary-preferred",
    ),
    ArmSpec(
        name="verifier_selector_stub",
        model_path=MODEL_IQ4_XS,
        device="none",
        ngl=0,
        role="verifier_selector_stub",
        prompt=(
            "Draft a verifier-selector stub and return compact JSON with keys arm, "
            "verifier, and selector."
        ),
        resource_class="cpu_selector_stub",
        residency_note="Stub arm for the selector/verifier economics path.",
        co_residency_policy="small-footprint-preferred",
        beneficiary_policy="smaller-beneficiary-only",
    ),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwable reasoning-economics runner")
    parser.add_argument("--execute", action="store_true", help="Run selected smoke arm(s) after writing the plan")
    parser.add_argument(
        "--only",
        action="append",
        choices=[arm.name for arm in ARMS],
        help="Arm name to execute. May be repeated. Defaults to the first IQ4 smoke.",
    )
    parser.add_argument(
        "--allow-glm-download",
        action="store_true",
        help="Override the GLM download guard for execute mode",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for plan/results/logs",
    )
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS, help="CPU threads for llama-server")
    parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT, help="Context size")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Completion token budget")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic request seed")
    parser.add_argument("--port-base", type=int, default=DEFAULT_PORT_BASE, help="Base port for exact command plans")
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT_S,
        help="HTTP request timeout in seconds",
    )
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=DEFAULT_STARTUP_TIMEOUT_S,
        help="Server health timeout in seconds",
    )
    return parser.parse_args(argv)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def glm_download_active() -> bool:
    probe = subprocess.run(
        ["pgrep", "-af", GLM_PATTERN],
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.returncode == 0 and bool(probe.stdout.strip())


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


def query_chat(port: int, payload: dict[str, Any], timeout_s: int) -> tuple[dict[str, Any], str]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=canonical_json(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw), raw


def terminate_server(proc: subprocess.Popen[str]) -> None:
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
            pass

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


def arm_port(base: int, index: int) -> int:
    return base + (index * 10)


def shell_prefix() -> list[str]:
    return ["env", "-i", *[f"{key}={value}" for key, value in SANITIZED_ENV.items()]]


def launch_argv(arm: ArmSpec, port: int, args: argparse.Namespace) -> list[str]:
    return [
        str(SERVER_BIN),
        "-m",
        str(arm.model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--device",
        arm.device,
        "-ngl",
        str(arm.ngl),
        "-t",
        str(args.threads),
        "-c",
        str(args.context),
        "-fa",
        "on",
        "-rea",
        "off",
    ]


def launch_command_string(arm: ArmSpec, port: int, args: argparse.Namespace) -> str:
    argv = [
        *shell_prefix(),
        "numactl",
        "--interleave=all",
        *launch_argv(arm, port, args),
    ]
    return " ".join(shlex.quote(part) for part in argv)


def smoke_payload(arm: ArmSpec, args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": arm.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": 1.0,
        "top_k": 1,
        "seed": args.seed,
        "stream": False,
    }
    if arm.response_format is not None:
        payload["response_format"] = arm.response_format
    if arm.json_schema is not None:
        payload["json_schema"] = arm.json_schema
    return payload


def selected_arm_indices(args: argparse.Namespace) -> list[int]:
    if not args.only:
        return [0]
    wanted = set(args.only)
    return [index for index, arm in enumerate(ARMS) if arm.name in wanted]


def smoke_command_string(port: int, arm: ArmSpec, args: argparse.Namespace) -> str:
    payload = canonical_json(smoke_payload(arm, args))
    argv = [
        *shell_prefix(),
        "curl",
        "-fsS",
        f"http://127.0.0.1:{port}/v1/chat/completions",
        "-H",
        "Content-Type: application/json",
        "--data",
        payload,
    ]
    return " ".join(shlex.quote(part) for part in argv)


def cleanup_command_string(pid_var: str = "SERVER_PID") -> str:
    return f"if kill -0 ${pid_var} 2>/dev/null; then kill ${pid_var}; wait ${pid_var} 2>/dev/null || true; fi"


def build_arm_plan(arm: ArmSpec, index: int, args: argparse.Namespace) -> dict[str, Any]:
    port = arm_port(args.port_base, index)
    return {
        "name": arm.name,
        "role": arm.role,
        "model_path": str(arm.model_path),
        "device": arm.device,
        "ngl": arm.ngl,
        "port": port,
        "resource_class": arm.resource_class,
        "resource_notes": {
            "residency": arm.residency_note,
            "co_residency_policy": arm.co_residency_policy,
            "beneficiary_policy": arm.beneficiary_policy,
        },
        "response_format": arm.response_format,
        "json_schema": arm.json_schema,
        "commands": {
            "launch": launch_command_string(arm, port, args),
            "smoke": smoke_command_string(port, arm, args),
            "cleanup": cleanup_command_string(),
        },
    }


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": "qwable_reasoning_economics_plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "execute" if args.execute else "dry_run",
        "experimental_root": str(EXPERIMENTAL_ROOT),
        "server_bin": str(SERVER_BIN),
        "ld_library_path": str(SERVER_LIB_DIR),
        "sanitized_env": SANITIZED_ENV,
        "glm_guard": {
            "pattern": GLM_PATTERN,
            "active": glm_download_active(),
            "blocked_in_execute": True,
            "allow_override_flag": "--allow-glm-download",
        },
        "model_paths": {
            "iq4_xs": str(MODEL_IQ4_XS),
            "q8_0": str(MODEL_Q8_0),
        },
        "execution": {
            "execute_mode_is_minimal": True,
            "first_smoke_arm": ARMS[0].name,
            "selected_smoke_arms": [ARMS[index].name for index in selected_arm_indices(args)],
            "request": {
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "seed": args.seed,
                "request_timeout_s": args.request_timeout,
                "startup_timeout_s": args.startup_timeout,
            },
        },
        "resource_summary": {
            "iq4_xs": "can co-reside more plausibly",
            "q8_0": "sequential or smaller-beneficiary only",
        },
        "arms": [build_arm_plan(arm, index, args) for index, arm in enumerate(ARMS)],
    }


def render_commands(plan: dict[str, Any], output_dir: Path) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f'# Qwable reasoning-economics dry-run plan generated at {plan["created_at"]}',
        f'# plan.json: {output_dir / "plan.json"}',
        "",
    ]
    for arm in plan["arms"]:
        lines.extend(
            [
                f'# arm: {arm["name"]}',
                f'# resource_class: {arm["resource_class"]}',
                arm["commands"]["launch"],
                arm["commands"]["smoke"],
                arm["commands"]["cleanup"],
                "",
            ]
        )
    return "\n".join(lines)


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
            env=SANITIZED_ENV,
        )
    except Exception:
        log_handle.close()
        raise
    proc._qwable_log_handle = log_handle  # type: ignore[attr-defined]
    return proc


def parse_content_json(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if not stripped:
        return {"content_json_mode": "empty", "content_json": None}
    try:
        return {"content_json_mode": "strict", "content_json": json.loads(stripped)}
    except json.JSONDecodeError:
        pass

    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            fenced = "\n".join(lines[1:-1]).strip()
            try:
                return {"content_json_mode": "fenced", "content_json": json.loads(fenced)}
            except json.JSONDecodeError:
                pass
    return {"content_json_mode": "non_json", "content_json": None}


def response_summary(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices") or []
    first = choices[0] if choices else {}
    message = first.get("message") if isinstance(first, dict) else {}
    content = message.get("content", "") if isinstance(message, dict) else ""
    summary = {
        "finish_reason": first.get("finish_reason") if isinstance(first, dict) else None,
        "usage": response.get("usage"),
        "timings": response.get("timings"),
        "content": content,
    }
    summary.update(parse_content_json(content))
    return summary


def run_smoke_arm(args: argparse.Namespace, output_dir: Path, plan: dict[str, Any], arm_index: int) -> dict[str, Any]:
    arm = ARMS[arm_index]
    port = plan["arms"][arm_index]["port"]
    log_path = output_dir / "logs" / f"{arm.name}.server.log"
    raw_response_path = output_dir / "responses" / f"{arm.name}.raw.json"
    result_path = output_dir / "results" / f"{arm.name}.json"
    for directory in (log_path.parent, raw_response_path.parent, result_path.parent):
        directory.mkdir(parents=True, exist_ok=True)

    proc: subprocess.Popen[str] | None = None
    record: dict[str, Any] = {
        "arm": arm.name,
        "port": port,
        "command": plan["arms"][arm_index]["commands"]["launch"],
        "smoke_command": plan["arms"][arm_index]["commands"]["smoke"],
        "model_path": str(arm.model_path),
        "device": arm.device,
        "ngl": arm.ngl,
        "role": arm.role,
    }
    try:
        proc = launch_server(launch_argv(arm, port, args), log_path)
        record["server_pid"] = proc.pid
        wait_for_health(port, args.startup_timeout, pid=proc.pid)
        response, raw_response = query_chat(port=port, payload=smoke_payload(arm, args), timeout_s=args.request_timeout)
        raw_response_path.write_text(raw_response, encoding="utf-8")
        response_hash = sha256_text(canonical_json(response))
        record.update(
            {
                "status": "ok",
                "response_sha256": response_hash,
                "response_path": str(raw_response_path),
                "response_summary": response_summary(response),
            }
        )
        result_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        return record
    except Exception as exc:
        record.update({"status": "error", "error": str(exc)})
        result_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        raise
    finally:
        if proc is not None:
            try:
                terminate_server(proc)
            finally:
                log_handle = getattr(proc, "_qwable_log_handle", None)
                if log_handle is not None:
                    log_handle.close()


def run_first_smoke(args: argparse.Namespace, output_dir: Path, plan: dict[str, Any]) -> dict[str, Any]:
    return run_smoke_arm(args, output_dir, plan, 0)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not str(SERVER_BIN).startswith(str(EXPERIMENTAL_BIN_DIR)):
        raise RuntimeError(f"refusing non-experimental server binary: {SERVER_BIN}")

    if args.execute and not args.allow_glm_download and glm_download_active():
        print("FATAL: GLM-5.2 download is active; rerun with --allow-glm-download only if you accept contention.", file=sys.stderr)
        return 75

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (output_dir / "responses").mkdir(parents=True, exist_ok=True)
    (output_dir / "results").mkdir(parents=True, exist_ok=True)

    plan = build_plan(args)
    (output_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "commands.sh").write_text(render_commands(plan, output_dir), encoding="utf-8")

    print("Qwable reasoning-economics runner")
    print(f"mode: {'execute' if args.execute else 'dry_run'}")
    print(f"output_dir: {output_dir}")
    print(f"server_bin: {SERVER_BIN}")
    print(f"ld_library_path: {SERVER_LIB_DIR}")
    print(f"iq4_xs_model: {MODEL_IQ4_XS}")
    print(f"q8_0_model: {MODEL_Q8_0}")
    print(f"glm_active: {plan['glm_guard']['active']}")
    print(f"first_smoke_arm: {ARMS[0].name}")
    print(f"selected_smoke_arms: {', '.join(plan['execution']['selected_smoke_arms'])}")
    print("resource_note_iq4: can co-reside more plausibly")
    print("resource_note_q8: sequential or smaller-beneficiary only")

    if not args.execute:
        print("Dry run only. No server will be launched.")
        print(f"Plan written to {output_dir / 'plan.json'}")
        print(f"Commands written to {output_dir / 'commands.sh'}")
        return 0

    print("Executing selected smoke arm(s).")
    try:
        records = [run_smoke_arm(args, output_dir, plan, index) for index in selected_arm_indices(args)]
    except Exception as exc:
        print(f"Execute mode failed: {exc}", file=sys.stderr)
        return 1

    summary = {
        "schema": "qwable_reasoning_economics_execute.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "execute",
        "first_smoke": records[0] if records else None,
        "smokes": records,
        "plan_path": str(output_dir / "plan.json"),
        "commands_path": str(output_dir / "commands.sh"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Summary written to {output_dir / 'summary.json'}")
    print("Selected-smoke execution complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
