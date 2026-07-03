#!/usr/bin/env python3
"""P-BENCH-3 single-server -np sweep for batched/slot decode.

This harness launches one llama-server instance per cell, sends a fixed
question batch with bounded client concurrency, and records tasks/hour plus
per-stream latency. It intentionally avoids the orchestrator role path so E1
can test the serving primitive before E2 tests the eval driver.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import signal
import socket
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO_ROOT / "orchestration" / "model_registry.yaml"
DEFAULT_QUESTION_POOL = REPO_ROOT / "benchmarks" / "prompts" / "question_pool.jsonl"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "batched_decode"
DEFAULT_LLAMA_SERVER = Path("/mnt/raid0/llm/llama.cpp/build/bin/llama-server")
DEFAULT_MODEL_KEYS = ("qwen36_q8_0", "qwen36_27b_q8")
DEFAULT_NP_LEVELS = (1, 2, 4, 8, 16)
MAX_DECISION_GRADE_UPTIME_SECONDS = 7 * 24 * 60 * 60
DEFAULT_ENV = {
    "OMP_PROC_BIND": "spread",
    "OMP_PLACES": "cores",
    "OMP_WAIT_POLICY": "active",
    "OMP_DYNAMIC": "false",
    "KMP_BLOCKTIME": "10",
}
LLVM20_LIBDIR = Path("/usr/lib/llvm-20/lib")
REQUIRED_NUMA_BALANCING = "0"


@dataclass(frozen=True)
class ModelSpec:
    label: str
    path: Path
    registry_key: str | None = None
    quant: str | None = None
    architecture: str | None = None


@dataclass(frozen=True)
class PromptSpec:
    qid: str
    suite: str
    tier: int | None
    prompt: str


@dataclass
class RequestResult:
    model: str
    np_level: int
    request_index: int
    qid: str
    suite: str
    success: bool
    latency_ms: float
    predicted_tokens: int
    prompt_tokens: int
    predicted_tps: float
    http_status: int | None = None
    error: str = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_csv_ints(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = math.ceil(q * len(ordered)) - 1
    return ordered[max(0, min(idx, len(ordered) - 1))]


def read_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load model registry defaults")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise RuntimeError(f"registry did not parse as a mapping: {path}")
    return data


def registry_entry(registry: dict[str, Any], key: str) -> dict[str, Any] | None:
    entry = registry.get(key)
    if isinstance(entry, dict):
        return entry
    for section_name in ("roles", "models", "model_catalog"):
        section = registry.get(section_name)
        if isinstance(section, dict):
            entry = section.get(key)
            if isinstance(entry, dict):
                return entry
    return None


def load_model_specs(
    registry_path: Path,
    model_keys: list[str],
    model_overrides: list[str],
) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    if model_keys:
        registry = read_yaml(registry_path)
        for key in model_keys:
            entry = registry_entry(registry, key)
            if entry is None:
                raise RuntimeError(f"registry key not found or invalid: {key}")
            model = entry.get("model")
            if not isinstance(model, dict):
                raise RuntimeError(f"registry key has no model mapping: {key}")
            raw_path = model.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                raise RuntimeError(f"registry key has no model.path: {key}")
            specs.append(
                ModelSpec(
                    label=key,
                    path=Path(raw_path),
                    registry_key=key,
                    quant=str(model.get("quant") or "") or None,
                    architecture=str(model.get("architecture") or "") or None,
                )
            )

    for raw in model_overrides:
        if "=" in raw:
            label, path = raw.split("=", 1)
        elif ":" in raw:
            label, path = raw.split(":", 1)
        else:
            raise argparse.ArgumentTypeError(
                "--model must use label=/path/to/model.gguf or label:/path/to/model.gguf"
            )
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise argparse.ArgumentTypeError("--model label and path must be non-empty")
        specs.append(ModelSpec(label=label, path=Path(path)))

    if not specs:
        raise RuntimeError("no models selected")

    for spec in specs:
        if not spec.path.exists():
            raise FileNotFoundError(f"{spec.label}: model file not found: {spec.path}")
    return specs


def load_prompt_batch(
    path: Path,
    *,
    limit: int,
    seed: int,
    tier: int | None,
    suites: set[str],
) -> list[PromptSpec]:
    prompts: list[PromptSpec] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("__pool_metadata__"):
                continue
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                continue
            if row.get("image_path"):
                continue
            row_tier = row.get("tier")
            if tier is not None and row_tier != tier:
                continue
            suite = str(row.get("suite") or "unknown")
            if suites and suite not in suites:
                continue
            context = row.get("context")
            full_prompt = f"{context}\n\n{prompt}" if isinstance(context, str) and context else prompt
            prompts.append(
                PromptSpec(
                    qid=str(row.get("id") or f"row-{len(prompts)}"),
                    suite=suite,
                    tier=row_tier if isinstance(row_tier, int) else None,
                    prompt=full_prompt,
                )
            )
    if len(prompts) < limit:
        raise RuntimeError(f"only {len(prompts)} prompts matched filters, need {limit}")
    rng = random.Random(seed)
    selected = rng.sample(prompts, limit)
    return selected


def run_capture(cmd: list[str], timeout: float = 10.0) -> str:
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return proc.stdout.strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def collect_attestation() -> dict[str, Any]:
    governors = sorted(
        {
            path.read_text(encoding="utf-8").strip()
            for path in Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_governor")
            if path.exists()
        }
    )
    numa_balancing_path = Path("/proc/sys/kernel/numa_balancing")
    uptime_seconds = None
    try:
        uptime_seconds = float(Path("/proc/uptime").read_text(encoding="utf-8").split()[0])
    except Exception:
        pass
    return {
        "created_at": utc_now(),
        "host": socket.gethostname(),
        "kernel": run_capture(["uname", "-a"]),
        "lscpu_summary": run_capture(["lscpu"]),
        "loadavg": Path("/proc/loadavg").read_text(encoding="utf-8").strip(),
        "uptime_seconds": uptime_seconds,
        "numa_balancing": (
            numa_balancing_path.read_text(encoding="utf-8").strip()
            if numa_balancing_path.exists()
            else None
        ),
        "scaling_governors": governors,
        "meminfo_head": "\n".join(Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()[:8]),
        "existing_llama_processes": find_llama_processes(),
    }


def host_health_warnings(attestation: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    uptime = attestation.get("uptime_seconds")
    if isinstance(uptime, (int, float)) and uptime > MAX_DECISION_GRADE_UPTIME_SECONDS:
        warnings.append(
            "uptime exceeds 1 week; MEASUREMENT.md P-BENCH-1/P-BENCH-3 policy requires reboot "
            "before decision-grade claims"
        )
    numa_balancing = attestation.get("numa_balancing")
    if numa_balancing != REQUIRED_NUMA_BALANCING:
        warnings.append(
            f"kernel.numa_balancing={numa_balancing!r}; expected {REQUIRED_NUMA_BALANCING!r} "
            "for canonical NUMA-interleave CPU benchmarking"
        )
    existing = attestation.get("existing_llama_processes")
    if existing:
        warnings.append("existing llama processes present during attestation")
    return warnings


def find_llama_processes() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    ps = run_capture(["ps", "-eo", "pid=,args="])
    for line in ps.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid, _, args = stripped.partition(" ")
        if "earlyoom" in args:
            continue
        if re.search(r"(^|/)(llama-server|llama-bench|llama-cli)(\s|$)", args):
            rows.append({"pid": pid, "args": args})
    return rows


def ensure_clean_runtime() -> None:
    existing = find_llama_processes()
    if existing:
        formatted = "\n".join(f"  {row['pid']} {row['args']}" for row in existing)
        raise RuntimeError(f"existing llama processes would contaminate P-BENCH-3:\n{formatted}")


def http_json(method: str, url: str, payload: dict[str, Any] | None, timeout: float) -> tuple[int, dict[str, Any]]:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        if not raw:
            return resp.status, {}
        return resp.status, json.loads(raw.decode("utf-8"))


def wait_for_health(port: int, timeout_s: float, proc: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited before health check passed, rc={proc.returncode}")
        try:
            status, _ = http_json("GET", f"http://127.0.0.1:{port}/health", None, timeout=5.0)
            if status == 200:
                return
            last_error = f"HTTP {status}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(2.0)
    raise TimeoutError(f"server on port {port} did not become healthy: {last_error}")


def build_env(role_label: str) -> dict[str, str]:
    env = os.environ.copy()
    ld_parts = [part for part in env.get("LD_LIBRARY_PATH", "").split(":") if part]
    required = [str(LLVM20_LIBDIR), str(DEFAULT_LLAMA_SERVER.parent)]
    env["LD_LIBRARY_PATH"] = ":".join(required + [part for part in ld_parts if part not in required])
    env.update(DEFAULT_ENV)
    if "27b" in role_label.lower():
        # Mirrors the orchestrator arch-class note: dense Q8 stays on default v5.
        pass
    return env


def build_server_command(
    *,
    binary: Path,
    model: ModelSpec,
    port: int,
    np_level: int,
    threads: int,
    context_size: int,
    ubatch_size: int,
    kv_type: str,
    flash_attn: bool,
    jinja: bool,
    mlock: bool,
    extra_args: list[str],
) -> list[str]:
    cmd = [
        "numactl",
        "--interleave=all",
        str(binary),
        "-m",
        str(model.path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-np",
        str(np_level),
        "-c",
        str(context_size),
        "-t",
        str(threads),
        "-ub",
        str(ubatch_size),
        "-ctk",
        kv_type,
        "-ctv",
        kv_type,
        "--log-colors",
        "off",
    ]
    if flash_attn:
        cmd.extend(["--flash-attn", "on"])
    if jinja:
        cmd.append("--jinja")
    if mlock:
        cmd.append("--mlock")
    cmd.extend(extra_args)
    return cmd


def start_server(cmd: list[str], env: dict[str, str], log_path: Path) -> subprocess.Popen[str]:
    log_fh = log_path.open("w", encoding="utf-8")
    try:
        return subprocess.Popen(
            cmd,
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    except Exception:
        log_fh.close()
        raise


def stop_server(proc: subprocess.Popen[str], timeout_s: float = 30.0) -> dict[str, Any]:
    result: dict[str, Any] = {
        "pid": proc.pid,
        "signal": None,
        "returncode": None,
        "killed": False,
        "ps_verified_dead": False,
    }
    if proc.poll() is None:
        result["signal"] = "SIGTERM"
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            result["signal"] = "SIGKILL"
            result["killed"] = True
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=timeout_s)
    result["returncode"] = proc.returncode
    if proc.poll() is None:
        raise RuntimeError(f"server pid {proc.pid} still alive after stop")
    ps = subprocess.run(
        ["ps", "-p", str(proc.pid)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    result["ps_verified_dead"] = ps.returncode != 0
    if not result["ps_verified_dead"]:
        raise RuntimeError(f"server pid {proc.pid} still visible after stop:\n{ps.stdout}")
    return result


def send_completion(
    *,
    port: int,
    prompt: PromptSpec,
    request_index: int,
    model_label: str,
    np_level: int,
    n_predict: int,
    timeout_s: float,
) -> RequestResult:
    payload = {
        "prompt": prompt.prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "cache_prompt": False,
        "stream": False,
    }
    start = time.perf_counter()
    try:
        status, data = http_json(
            "POST",
            f"http://127.0.0.1:{port}/completion",
            payload,
            timeout=timeout_s,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        timings = data.get("timings", {}) if isinstance(data, dict) else {}
        return RequestResult(
            model=model_label,
            np_level=np_level,
            request_index=request_index,
            qid=prompt.qid,
            suite=prompt.suite,
            success=status == 200,
            latency_ms=latency_ms,
            predicted_tokens=int(timings.get("predicted_n") or 0),
            prompt_tokens=int(timings.get("prompt_n") or 0),
            predicted_tps=float(timings.get("predicted_per_second") or 0.0),
            http_status=status,
            error="" if status == 200 else f"HTTP {status}",
        )
    except urllib.error.HTTPError as exc:
        latency_ms = (time.perf_counter() - start) * 1000.0
        return RequestResult(
            model=model_label,
            np_level=np_level,
            request_index=request_index,
            qid=prompt.qid,
            suite=prompt.suite,
            success=False,
            latency_ms=latency_ms,
            predicted_tokens=0,
            prompt_tokens=0,
            predicted_tps=0.0,
            http_status=exc.code,
            error=f"HTTP {exc.code}",
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000.0
        return RequestResult(
            model=model_label,
            np_level=np_level,
            request_index=request_index,
            qid=prompt.qid,
            suite=prompt.suite,
            success=False,
            latency_ms=latency_ms,
            predicted_tokens=0,
            prompt_tokens=0,
            predicted_tps=0.0,
            error=str(exc),
        )


def measure_stream_ttft(port: int, prompt: PromptSpec, timeout_s: float) -> float:
    payload = {
        "prompt": prompt.prompt,
        "n_predict": 1,
        "temperature": 0.0,
        "cache_prompt": False,
        "stream": True,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if line.startswith("data:"):
                return (time.perf_counter() - start) * 1000.0
    return 0.0


def run_prompt_batch(
    *,
    port: int,
    model: ModelSpec,
    np_level: int,
    prompts: list[PromptSpec],
    n_predict: int,
    request_timeout_s: float,
    requests_path: Path,
) -> tuple[list[RequestResult], float]:
    results: list[RequestResult] = []
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=np_level) as pool:
        futures = [
            pool.submit(
                send_completion,
                port=port,
                prompt=prompt,
                request_index=index,
                model_label=model.label,
                np_level=np_level,
                n_predict=n_predict,
                timeout_s=request_timeout_s,
            )
            for index, prompt in enumerate(prompts)
        ]
        with requests_path.open("a", encoding="utf-8") as fh:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                fh.write(json.dumps(asdict(result), sort_keys=True) + "\n")
                fh.flush()
    wall_s = time.perf_counter() - start
    results.sort(key=lambda item: item.request_index)
    return results, wall_s


def summarize_cell(
    *,
    model: ModelSpec,
    np_level: int,
    results: list[RequestResult],
    wall_s: float,
    ttft_ms: float,
    server_pid: int,
    server_command: list[str],
) -> dict[str, Any]:
    successes = [result for result in results if result.success]
    latencies = [result.latency_ms for result in successes]
    per_tps = [result.predicted_tps for result in successes if result.predicted_tps > 0]
    total_predicted = sum(result.predicted_tokens for result in successes)
    total_prompt = sum(result.prompt_tokens for result in successes)
    total = len(results)
    success_count = len(successes)
    return {
        "timestamp": utc_now(),
        "protocol_id": "P-BENCH-3",
        "model": model.label,
        "registry_key": model.registry_key,
        "model_path": str(model.path),
        "quant": model.quant,
        "architecture": model.architecture,
        "np": np_level,
        "server_pid": server_pid,
        "server_command": server_command,
        "total_count": total,
        "success_count": success_count,
        "error_rate": ((total - success_count) / total) if total else 1.0,
        "wall_seconds": wall_s,
        "tasks_per_hour": (success_count / wall_s * 3600.0) if wall_s > 0 else 0.0,
        "aggregate_predicted_tps": (total_predicted / wall_s) if wall_s > 0 else 0.0,
        "predicted_tokens_total": total_predicted,
        "prompt_tokens_total": total_prompt,
        "per_request_tps_mean": statistics.mean(per_tps) if per_tps else 0.0,
        "per_request_tps_stdev": statistics.stdev(per_tps) if len(per_tps) > 1 else 0.0,
        "p50_latency_ms": percentile(latencies, 0.50),
        "p95_latency_ms": percentile(latencies, 0.95),
        "ttft_ms": ttft_ms,
    }


def write_csv_row(path: Path, row: dict[str, Any]) -> None:
    fields = [
        "timestamp",
        "protocol_id",
        "model",
        "registry_key",
        "np",
        "success_count",
        "total_count",
        "error_rate",
        "wall_seconds",
        "tasks_per_hour",
        "aggregate_predicted_tps",
        "predicted_tokens_total",
        "prompt_tokens_total",
        "per_request_tps_mean",
        "per_request_tps_stdev",
        "p50_latency_ms",
        "p95_latency_ms",
        "ttft_ms",
        "model_path",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field) for field in fields})


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def run_cell(
    *,
    model: ModelSpec,
    np_level: int,
    prompts: list[PromptSpec],
    args: argparse.Namespace,
    output_dir: Path,
    cell_index: int,
) -> dict[str, Any]:
    port = args.port_base + cell_index
    log_path = output_dir / "logs" / f"{model.label}-np{np_level}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_server_command(
        binary=args.llama_server,
        model=model,
        port=port,
        np_level=np_level,
        threads=args.threads,
        context_size=args.context_size,
        ubatch_size=args.ubatch_size,
        kv_type=args.kv_type,
        flash_attn=not args.no_flash_attn,
        jinja=not args.no_jinja,
        mlock=not args.no_mlock,
        extra_args=args.extra_server_arg,
    )
    if args.dry_run:
        return {
            "timestamp": utc_now(),
            "protocol_id": "P-BENCH-3",
            "model": model.label,
            "registry_key": model.registry_key,
            "model_path": str(model.path),
            "np": np_level,
            "server_command": command,
            "dry_run": True,
        }

    proc: subprocess.Popen[str] | None = None
    stop_result: dict[str, Any] | None = None
    try:
        proc = start_server(command, build_env(model.label), log_path)
        wait_for_health(port, args.startup_timeout, proc)
        warmup_prompts = prompts[: args.warmup_prompts]
        if warmup_prompts:
            warmup_path = output_dir / "warmup_requests.jsonl"
            run_prompt_batch(
                port=port,
                model=model,
                np_level=np_level,
                prompts=warmup_prompts,
                n_predict=min(args.n_predict, args.warmup_n_predict),
                request_timeout_s=args.request_timeout,
                requests_path=warmup_path,
            )
        ttft_ms = measure_stream_ttft(port, prompts[0], args.request_timeout) if args.measure_ttft else 0.0
        requests_path = output_dir / "requests.jsonl"
        results, wall_s = run_prompt_batch(
            port=port,
            model=model,
            np_level=np_level,
            prompts=prompts,
            n_predict=args.n_predict,
            request_timeout_s=args.request_timeout,
            requests_path=requests_path,
        )
        row = summarize_cell(
            model=model,
            np_level=np_level,
            results=results,
            wall_s=wall_s,
            ttft_ms=ttft_ms,
            server_pid=proc.pid,
            server_command=command,
        )
        row["server_log"] = str(log_path)
        return row
    finally:
        if proc is not None:
            stop_result = stop_server(proc)
            write_jsonl(
                output_dir / "events.jsonl",
                {
                    "timestamp": utc_now(),
                    "event": "server_stopped",
                    "model": model.label,
                    "np": np_level,
                    "port": port,
                    "result": stop_result,
                },
            )


def build_recommendations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("dry_run"):
            continue
        by_model.setdefault(str(row["model"]), []).append(row)
    recommendations: list[dict[str, Any]] = []
    for model, model_rows in sorted(by_model.items()):
        successful = [row for row in model_rows if row.get("success_count", 0) > 0]
        if not successful:
            recommendations.append({"model": model, "status": "no_successful_cells"})
            continue
        best = max(successful, key=lambda row: row.get("tasks_per_hour", 0.0))
        best_rate = float(best.get("tasks_per_hour") or 0.0)
        saturation = min(
            (
                row
                for row in successful
                if best_rate > 0 and float(row.get("tasks_per_hour") or 0.0) >= 0.95 * best_rate
            ),
            key=lambda row: int(row["np"]),
        )
        recommendations.append(
            {
                "model": model,
                "status": "ok",
                "best_np": best["np"],
                "best_tasks_per_hour": round(best_rate, 3),
                "saturation_np_95pct": saturation["np"],
                "saturation_tasks_per_hour": round(float(saturation.get("tasks_per_hour") or 0.0), 3),
                "best_p95_latency_ms": round(float(best.get("p95_latency_ms") or 0.0), 1),
                "saturation_p95_latency_ms": round(float(saturation.get("p95_latency_ms") or 0.0), 1),
            }
        )
    return recommendations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=f"e1-pbench3-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--model-key", action="append", default=[])
    parser.add_argument("--model", action="append", default=[], help="label=/abs/model.gguf override; can repeat")
    parser.add_argument("--question-pool", type=Path, default=DEFAULT_QUESTION_POOL)
    parser.add_argument("--prompt-limit", type=int, default=43)
    parser.add_argument("--prompt-seed", type=int, default=42)
    parser.add_argument("--tier", type=int, default=1)
    parser.add_argument("--suites", default="", help="optional comma-separated suite filter")
    parser.add_argument("--np-levels", type=parse_csv_ints, default=list(DEFAULT_NP_LEVELS))
    parser.add_argument("--llama-server", type=Path, default=DEFAULT_LLAMA_SERVER)
    parser.add_argument("--threads", type=int, default=96)
    parser.add_argument("--context-size", type=int, default=32768)
    parser.add_argument("--ubatch-size", type=int, default=8192)
    parser.add_argument("--kv-type", default="q8_0")
    parser.add_argument("--n-predict", type=int, default=256)
    parser.add_argument("--warmup-prompts", type=int, default=2)
    parser.add_argument("--warmup-n-predict", type=int, default=32)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--port-base", type=int, default=18070)
    parser.add_argument("--extra-server-arg", action="append", default=[])
    parser.add_argument("--no-flash-attn", action="store_true")
    parser.add_argument("--no-jinja", action="store_true")
    parser.add_argument("--no-mlock", action="store_true")
    parser.add_argument("--no-ttft", dest="measure_ttft", action="store_false")
    parser.set_defaults(measure_ttft=True)
    parser.add_argument("--skip-clean-check", action="store_true")
    parser.add_argument(
        "--allow-host-health-warning",
        action="store_true",
        help="run despite host-health warnings; manifest marks the run non-decision-grade",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.prompt_limit <= 0:
        parser.error("--prompt-limit must be positive")
    if args.n_predict <= 0:
        parser.error("--n-predict must be positive")
    if args.warmup_prompts < 0:
        parser.error("--warmup-prompts must be non-negative")
    if not args.model_key and not args.model:
        args.model_key = list(DEFAULT_MODEL_KEYS)
    return args


def main() -> int:
    args = parse_args()
    if not args.llama_server.exists():
        raise FileNotFoundError(f"llama-server binary not found: {args.llama_server}")
    if not args.skip_clean_check and not args.dry_run:
        ensure_clean_runtime()

    suites = {suite.strip() for suite in args.suites.split(",") if suite.strip()}
    models = load_model_specs(args.registry, args.model_key, args.model)
    prompts = load_prompt_batch(
        args.question_pool,
        limit=args.prompt_limit,
        seed=args.prompt_seed,
        tier=args.tier,
        suites=suites,
    )
    attestation = collect_attestation()
    health_warnings = host_health_warnings(attestation)
    decision_grade = not health_warnings
    if health_warnings and not (args.allow_host_health_warning or args.dry_run):
        formatted = "\n".join(f"- {warning}" for warning in health_warnings)
        raise RuntimeError(
            "host-health preconditions failed; refusing decision-grade P-BENCH-3 run.\n"
            f"{formatted}\n"
            "Use --allow-host-health-warning only for exploratory/non-gating data."
        )

    output_dir = args.output_root / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = output_dir / "selected_prompts.jsonl"
    with prompt_path.open("w", encoding="utf-8") as fh:
        for prompt in prompts:
            fh.write(json.dumps(asdict(prompt), sort_keys=True) + "\n")

    manifest = {
        "run_id": args.run_id,
        "created_at": utc_now(),
        "protocol_id": "P-BENCH-3",
        "output_dir": str(output_dir),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key not in {"extra_server_arg"}
        }
        | {"extra_server_arg": args.extra_server_arg},
        "models": [asdict(model) | {"path": str(model.path)} for model in models],
        "prompt_batch": {
            "path": str(prompt_path),
            "source": str(args.question_pool),
            "limit": args.prompt_limit,
            "seed": args.prompt_seed,
            "tier": args.tier,
            "suites": sorted(suites),
            "qids": [prompt.qid for prompt in prompts],
        },
        "attestation": attestation,
        "host_health_warnings": health_warnings,
        "decision_grade": decision_grade,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    summary_rows: list[dict[str, Any]] = []
    cell_index = 0
    for model in models:
        for np_level in args.np_levels:
            print(f"[{utc_now()}] model={model.label} np={np_level}", flush=True)
            try:
                row = run_cell(
                    model=model,
                    np_level=np_level,
                    prompts=prompts,
                    args=args,
                    output_dir=output_dir,
                    cell_index=cell_index,
                )
            except Exception as exc:
                row = {
                    "timestamp": utc_now(),
                    "protocol_id": "P-BENCH-3",
                    "model": model.label,
                    "registry_key": model.registry_key,
                    "model_path": str(model.path),
                    "quant": model.quant,
                    "architecture": model.architecture,
                    "np": np_level,
                    "total_count": args.prompt_limit,
                    "success_count": 0,
                    "error_rate": 1.0,
                    "wall_seconds": 0.0,
                    "tasks_per_hour": 0.0,
                    "aggregate_predicted_tps": 0.0,
                    "predicted_tokens_total": 0,
                    "prompt_tokens_total": 0,
                    "per_request_tps_mean": 0.0,
                    "per_request_tps_stdev": 0.0,
                    "p50_latency_ms": 0.0,
                    "p95_latency_ms": 0.0,
                    "ttft_ms": 0.0,
                    "cell_error": str(exc),
                }
                write_jsonl(
                    output_dir / "events.jsonl",
                    {
                        "timestamp": utc_now(),
                        "event": "cell_failed",
                        "model": model.label,
                        "np": np_level,
                        "error": str(exc),
                    },
                )
            cell_index += 1
            summary_rows.append(row)
            write_jsonl(output_dir / "cells.jsonl", row)
            write_csv_row(output_dir / "summary.csv", row)
            if args.dry_run:
                print("  dry-run command:", " ".join(str(part) for part in row["server_command"]), flush=True)
            else:
                print(
                    "  tasks/hour={:.2f} agg_tps={:.2f} p95={:.0f}ms err={:.1%}".format(
                        float(row.get("tasks_per_hour") or 0.0),
                        float(row.get("aggregate_predicted_tps") or 0.0),
                        float(row.get("p95_latency_ms") or 0.0),
                        float(row.get("error_rate") or 0.0),
                    ),
                    flush=True,
                )

    recommendations = {
        "created_at": utc_now(),
        "protocol_id": "P-BENCH-3",
        "summary_csv": str(output_dir / "summary.csv"),
        "recommendations": build_recommendations(summary_rows),
    }
    (output_dir / "recommendations.json").write_text(
        json.dumps(recommendations, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
