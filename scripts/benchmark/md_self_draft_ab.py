#!/usr/bin/env python3
"""A/B embedded NEXTN self-draft against redundant same-file ``-md``.

This is a quiet-window harness for the CPU MTP bug fixed on 2026-07-03:
Qwen NEXTN roles should use ``--spec-type draft-mtp`` without ``-md`` when the
draft is embedded in the target GGUF. Passing ``-md <same file>`` forces a
second full-model draft path. The harness launches a throwaway llama-server on
one port, runs the same completion prompt for both arms, then writes a compact
JSON/Markdown report.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

from scripts.benchmark.mtp_acceptance_report import parse_log
from src.registry.kernel_paths import server_binary

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = Path("/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf")
DEFAULT_PROMPT = (
    "Solve this precisely and show the final numeric answer only after ####.\n"
    "A 3-phase power system has line voltage 13.8 kV, power factor 0.85 lagging, "
    "and real power 2.5 MW per phase. Calculate the magnitude of the line current "
    "in amperes."
)


@dataclass(frozen=True)
class Arm:
    name: str
    include_same_file_md: bool


@dataclass
class MemorySample:
    rss_kib: int | None = None
    pss_kib: int | None = None


@dataclass
class CompletionRun:
    repetition: int
    latency_s: float
    predicted_tokens_per_second: float | None
    predicted_tokens: int | None
    prompt_tokens: int | None
    response_chars: int
    error: str | None = None


@dataclass
class ArmResult:
    name: str
    include_same_file_md: bool
    command: list[str]
    log_path: str
    pid: int | None
    load_memory: MemorySample
    post_run_memory: MemorySample
    runs: list[CompletionRun]
    acceptance: dict[str, Any]
    startup_error: str | None = None


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def default_output_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return PROJECT_ROOT / "orchestration" / "reports" / f"md_self_draft_ab_{stamp}"


def _append_optional(cmd: list[str], flag: str, value: str | int | float | None) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def build_server_command(args: argparse.Namespace, arm: Arm) -> list[str]:
    cmd = [
        str(args.binary),
        "-m",
        str(args.model),
    ]
    if arm.include_same_file_md:
        cmd.extend(["-md", str(args.model)])
    cmd.extend(
        [
            "--host",
            args.host,
            "--port",
            str(args.port),
            "-np",
            str(args.slots),
            "-c",
            str(args.context_tokens),
            "-t",
            str(args.threads),
            "-ub",
            str(args.ubatch),
        ]
    )
    if args.flash_attn:
        cmd.extend(["--flash-attn", "on"])
    if args.jinja:
        cmd.append("--jinja")
    if args.kv_type_k and args.kv_type_v:
        cmd.extend(["-ctk", args.kv_type_k, "-ctv", args.kv_type_v])
    if args.mlock:
        cmd.append("--mlock")
    if args.no_mmap:
        cmd.append("--no-mmap")
    cmd.extend(["--spec-type", args.spec_type, "--spec-draft-n-max", str(args.draft_max)])
    _append_optional(cmd, "--draft-p-min", args.draft_p_min)
    _append_optional(cmd, "--threads-draft", args.threads_draft)
    cmd.extend(args.server_arg or [])
    return cmd


def port_is_open(host: str, port: int, timeout_s: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_s)
        return sock.connect_ex((host, port)) == 0


def wait_for_health(host: str, port: int, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    url = f"http://{host}:{port}/health"
    last_error: str | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if 200 <= response.status < 300:
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = str(exc)
        time.sleep(1)
    raise TimeoutError(f"server did not become healthy on {host}:{port}: {last_error}")


def read_memory(pid: int | None) -> MemorySample:
    if pid is None:
        return MemorySample()
    smaps = Path(f"/proc/{pid}/smaps_rollup")
    rss: int | None = None
    pss: int | None = None
    if smaps.exists():
        for line in smaps.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("Rss:"):
                rss = int(line.split()[1])
            elif line.startswith("Pss:"):
                pss = int(line.split()[1])
        return MemorySample(rss_kib=rss, pss_kib=pss)

    status = Path(f"/proc/{pid}/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1])
                break
    return MemorySample(rss_kib=rss, pss_kib=pss)


def _post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _timing_value(timings: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = timings.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _timing_int(timings: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = timings.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    return None


def run_completion(args: argparse.Namespace, repetition: int) -> CompletionRun:
    payload = {
        "prompt": args.prompt,
        "n_predict": args.n_predict,
        "temperature": args.temperature,
        "seed": args.seed,
        "cache_prompt": False,
        "stream": False,
    }
    start = time.perf_counter()
    try:
        response = _post_json(
            f"http://{args.host}:{args.port}/completion",
            payload,
            timeout_s=args.request_timeout,
        )
    except Exception as exc:  # noqa: BLE001 - record the local server failure in the artifact.
        return CompletionRun(
            repetition=repetition,
            latency_s=time.perf_counter() - start,
            predicted_tokens_per_second=None,
            predicted_tokens=None,
            prompt_tokens=None,
            response_chars=0,
            error=str(exc),
        )

    timings = response.get("timings")
    if not isinstance(timings, dict):
        timings = {}
    content = response.get("content") or response.get("response") or ""
    return CompletionRun(
        repetition=repetition,
        latency_s=time.perf_counter() - start,
        predicted_tokens_per_second=_timing_value(
            timings,
            "predicted_per_second",
            "predicted_tps",
            "tokens_per_second",
        ),
        predicted_tokens=_timing_int(timings, "predicted_n", "completion_tokens"),
        prompt_tokens=_timing_int(timings, "prompt_n", "prompt_tokens"),
        response_chars=len(str(content)),
    )


def terminate_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
        return
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


def run_arm(args: argparse.Namespace, arm: Arm, output_dir: Path) -> ArmResult:
    if port_is_open(args.host, args.port):
        raise RuntimeError(f"refusing to use occupied port {args.host}:{args.port}")

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{arm.name}.log"
    command = build_server_command(args, arm)
    env = os.environ.copy()
    env.setdefault("GGML_IQK", "1")
    for item in args.env or []:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"--env must be KEY=VALUE, got {item!r}")
        env[key] = value

    proc: subprocess.Popen[str] | None = None
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# command: {' '.join(command)}\n")
        log.flush()
        try:
            proc = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            wait_for_health(args.host, args.port, args.startup_timeout)
            load_memory = read_memory(proc.pid)
            for _ in range(args.warmups):
                run_completion(args, repetition=-1)
            runs = [run_completion(args, repetition=i + 1) for i in range(args.repetitions)]
            post_run_memory = read_memory(proc.pid)
        except Exception as exc:  # noqa: BLE001 - surface harness failures as report data.
            return ArmResult(
                name=arm.name,
                include_same_file_md=arm.include_same_file_md,
                command=command,
                log_path=str(log_path),
                pid=proc.pid if proc else None,
                load_memory=read_memory(proc.pid if proc else None),
                post_run_memory=read_memory(proc.pid if proc else None),
                runs=[],
                acceptance={},
                startup_error=str(exc),
            )
        finally:
            if proc is not None:
                terminate_process(proc)

    evidence = parse_log(log_path)
    acceptance = {
        "task_line_count": evidence.task_line_count,
        "cumulative_line_count": evidence.cumulative_line_count,
        "task_generated_tokens": evidence.task_generated_tokens,
        "task_accepted_tokens": evidence.task_accepted_tokens,
        "task_token_acceptance_rate": evidence.task_token_acceptance_rate,
        "latest_cumulative": asdict(evidence.latest_cumulative)
        if evidence.latest_cumulative is not None
        else None,
        "no_spec_implementation": evidence.no_spec_implementation,
    }
    return ArmResult(
        name=arm.name,
        include_same_file_md=arm.include_same_file_md,
        command=command,
        log_path=str(log_path),
        pid=proc.pid if proc else None,
        load_memory=load_memory,
        post_run_memory=post_run_memory,
        runs=runs,
        acceptance=acceptance,
    )


def _valid_speeds(result: ArmResult) -> list[float]:
    return [
        run.predicted_tokens_per_second
        for run in result.runs
        if run.error is None and run.predicted_tokens_per_second is not None
    ]


def _summarize_arm(result: ArmResult) -> dict[str, Any]:
    speeds = _valid_speeds(result)
    errors = [run.error for run in result.runs if run.error]
    return {
        "name": result.name,
        "include_same_file_md": result.include_same_file_md,
        "successful_runs": len(speeds),
        "failed_runs": len(errors),
        "mean_tps": mean(speeds) if speeds else None,
        "median_tps": median(speeds) if speeds else None,
        "load_rss_mib": result.load_memory.rss_kib / 1024 if result.load_memory.rss_kib is not None else None,
        "load_pss_mib": result.load_memory.pss_kib / 1024 if result.load_memory.pss_kib is not None else None,
        "post_run_rss_mib": result.post_run_memory.rss_kib / 1024
        if result.post_run_memory.rss_kib is not None
        else None,
        "post_run_pss_mib": result.post_run_memory.pss_kib / 1024
        if result.post_run_memory.pss_kib is not None
        else None,
        "errors": errors,
        "startup_error": result.startup_error,
        "acceptance": result.acceptance,
    }


def build_report(args: argparse.Namespace, results: list[ArmResult]) -> dict[str, Any]:
    summaries = {result.name: _summarize_arm(result) for result in results}
    same = summaries.get("same_file_md", {})
    embedded = summaries.get("embedded_self_draft", {})
    same_tps = same.get("median_tps")
    embedded_tps = embedded.get("median_tps")
    speedup_ratio = (
        embedded_tps / same_tps
        if isinstance(same_tps, (int, float))
        and isinstance(embedded_tps, (int, float))
        and same_tps > 0
        else None
    )
    same_pss = same.get("load_pss_mib")
    embedded_pss = embedded.get("load_pss_mib")
    pss_delta_mib = (
        embedded_pss - same_pss
        if isinstance(same_pss, (int, float)) and isinstance(embedded_pss, (int, float))
        else None
    )
    if speedup_ratio is None:
        decision = "inconclusive_missing_speed"
    elif speedup_ratio >= args.min_speedup_ratio:
        decision = "embedded_self_draft_faster"
    elif speedup_ratio >= 1.0:
        decision = "embedded_self_draft_not_slower"
    else:
        decision = "embedded_self_draft_slower"
    return {
        "schema_version": "md_self_draft_ab.v1",
        "generated_at": utc_now(),
        "decision": decision,
        "speedup_ratio_median_tps_embedded_over_same_file_md": speedup_ratio,
        "pss_delta_mib_embedded_minus_same_file_md": pss_delta_mib,
        "config": {
            "binary": str(args.binary),
            "model": str(args.model),
            "host": args.host,
            "port": args.port,
            "repetitions": args.repetitions,
            "warmups": args.warmups,
            "n_predict": args.n_predict,
            "seed": args.seed,
            "temperature": args.temperature,
            "context_tokens": args.context_tokens,
            "threads": args.threads,
            "ubatch": args.ubatch,
            "slots": args.slots,
            "spec_type": args.spec_type,
            "draft_max": args.draft_max,
            "mlock": args.mlock,
            "no_mmap": args.no_mmap,
            "min_speedup_ratio": args.min_speedup_ratio,
        },
        "arm_summaries": summaries,
        "arms": [asdict(result) for result in results],
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# CPU Embedded NEXTN Self-Draft A/B",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Decision: `{report['decision']}`",
        "- Speedup ratio, median t/s embedded/no-`-md` over same-file `-md`: "
        f"`{report['speedup_ratio_median_tps_embedded_over_same_file_md']}`",
        "- PSS delta MiB, embedded minus same-file `-md`: "
        f"`{report['pss_delta_mib_embedded_minus_same_file_md']}`",
        "",
        "## Arms",
        "",
        "| Arm | Same-file `-md` | Runs | Median t/s | Mean t/s | Load PSS MiB | Acceptance lines | Error |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for summary in report["arm_summaries"].values():
        acceptance = summary.get("acceptance") or {}
        lines.append(
            "| {name} | {md} | {runs} | {median} | {mean} | {pss} | {lines_count} | {error} |".format(
                name=summary["name"],
                md="yes" if summary["include_same_file_md"] else "no",
                runs=summary["successful_runs"],
                median=summary["median_tps"],
                mean=summary["mean_tps"],
                pss=summary["load_pss_mib"],
                lines_count=(acceptance.get("task_line_count") or 0)
                + (acceptance.get("cumulative_line_count") or 0),
                error=summary.get("startup_error") or ", ".join(summary.get("errors") or []) or "-",
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This harness uses `/completion`, not `/chat`, and launches throwaway local servers.",
            "- The default quiet-window guard refuses to run while AutoPilot appears active.",
            "- Decision-grade publication still depends on the measurement protocol in `/workspace/MEASUREMENT.md`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    return args.prompt or DEFAULT_PROMPT


def autopilot_quiet(project_root: Path) -> tuple[bool, dict[str, Any]]:
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/autopilot/phase_health_report.py",
        "--require-current-code",
        "--json",
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            text=True,
            capture_output=True,
            check=False,
            timeout=45,
        )
    except Exception as exc:  # noqa: BLE001 - quiet-window guard should fail closed.
        return False, {"error": str(exc), "command": cmd}
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return False, {
            "returncode": result.returncode,
            "stdout": result.stdout[-1000:],
            "stderr": result.stderr[-1000:],
        }
    if not payload.get("pid"):
        return True, payload
    phase = str(payload.get("phase") or "")
    action_type = payload.get("action_type")
    quiet = phase in {"idle", "paused", "stopped", "complete", "completed"} and not action_type
    return quiet, payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18070)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--n-predict", type=int, default=192)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--context-tokens", type=int, default=32768)
    parser.add_argument("--threads", type=int, default=96)
    parser.add_argument("--ubatch", type=int, default=8192)
    parser.add_argument("--slots", type=int, default=1)
    parser.add_argument("--kv-type-k", default="q8_0")
    parser.add_argument("--kv-type-v", default="q8_0")
    parser.add_argument("--spec-type", default="draft-mtp")
    parser.add_argument("--draft-max", type=int, default=4)
    parser.add_argument("--draft-p-min", type=float, default=None)
    parser.add_argument("--threads-draft", type=int, default=None)
    parser.add_argument("--request-timeout", type=float, default=900)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--min-speedup-ratio", type=float, default=1.05)
    parser.add_argument("--server-arg", action="append", default=[])
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--mlock", action="store_true")
    parser.add_argument("--no-mmap", action="store_true")
    parser.add_argument("--no-flash-attn", dest="flash_attn", action="store_false")
    parser.set_defaults(flash_attn=True)
    parser.add_argument("--no-jinja", dest="jinja", action="store_false")
    parser.set_defaults(jinja=True)
    parser.add_argument("--skip-autopilot-idle-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.binary is None:
        args.binary = server_binary("cpu")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.prompt = read_prompt(args)
    output_dir = args.output_dir or default_output_dir()
    arms = [
        Arm("same_file_md", include_same_file_md=True),
        Arm("embedded_self_draft", include_same_file_md=False),
    ]
    if args.dry_run:
        payload = {
            "output_dir": str(output_dir),
            "commands": {arm.name: build_server_command(args, arm) for arm in arms},
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if not args.skip_autopilot_idle_check:
        quiet, status = autopilot_quiet(PROJECT_ROOT)
        if not quiet:
            print(
                json.dumps(
                    {
                        "error": "autopilot_not_idle",
                        "status": status,
                        "hint": "pause AutoPilot or pass --skip-autopilot-idle-check for an intentional measured window",
                    },
                    indent=2,
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            return 2

    results = [run_arm(args, arm, output_dir) for arm in arms]
    report = build_report(args, results)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    md_path = output_dir / "summary.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({"summary_json": str(json_path), "summary_md": str(md_path), "decision": report["decision"]}))
    return 1 if report["decision"] == "embedded_self_draft_slower" else 0


if __name__ == "__main__":
    raise SystemExit(main())
