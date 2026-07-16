#!/usr/bin/env python3
"""GLM-5.2 DSA probe runner.

Defaults to dry-run / preflight mode. The runner only stages commands unless
--execute is provided, and execution is refused until the GLM shard directory
is clean and all non-cache shards are present.

The plan is intentionally narrow:
  1. shard integrity inventory (size/path/count only; no hashing)
  2. short load/decode smoke
  3. long-context DSA probe
  4. KV-length scaling with a fixed indexer_top_k

The default binary points at the experimental v7 tree and the runtime
environment is sanitized with a pinned LD_LIBRARY_PATH. Production paths are
rejected by default.
"""

from __future__ import annotations

import argparse
import dataclasses
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

MODEL_DIR = Path("/mnt/raid0/llm/models/GLM-5.2-UD-IQ2_M")
DEFAULT_BINARY = Path("/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server")
PRODUCTION_ROOT = Path("/mnt/raid0/llm/llama.cpp")
DEFAULT_LIBRARY_PATH = DEFAULT_BINARY.parent

SCHEMA = "glm52_dsa_probe_plan.v1"
REQUIRED_NON_CACHE_SHARDS = 6
DEFAULT_THREADS = 96
DEFAULT_UBATCH = 512
DEFAULT_MAX_TOKENS = 32
DEFAULT_SEED = 42
DEFAULT_TEMPERATURE = 0.0
DEFAULT_INDEXER_TOP_K = 32
DEFAULT_SHORT_CONTEXT = 4096
DEFAULT_LONG_CONTEXT = 32768
DEFAULT_KV_CONTEXTS = (4096, 8192, 16384, 32768)
DEFAULT_PORT_BASE = 19100
INDEXER_TOP_K_OVERRIDE_KEY = "glm-dsa.attention.indexer.top_k"

PROMPT_RESERVE_TOKENS = 192
FILLER_TEXT = (
    "The GLM DSA probe keeps the context deterministic while the runner checks "
    "shard integrity, load behavior, and KV scaling under a fixed indexer "
    "configuration. "
)


@dataclasses.dataclass(frozen=True)
class ShardRecord:
    path: str
    size: int


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def _is_under_production_root(path: Path) -> bool:
    try:
        return path.resolve().is_relative_to(PRODUCTION_ROOT.resolve())
    except FileNotFoundError:
        return False


def resolve_binary(binary: Path) -> Path:
    resolved = binary.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"llama binary not found: {resolved}")
    if not os.access(resolved, os.X_OK):
        raise PermissionError(f"llama binary is not executable: {resolved}")
    if _is_under_production_root(resolved):
        raise ValueError(f"refusing production llama binary: {resolved}")
    return resolved


def resolve_library_path(binary: Path, library_path: Path | None = None) -> Path:
    path = library_path if library_path is not None else binary.parent
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"library path not found: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"library path is not a directory: {resolved}")
    return resolved


def is_blocker_file(path: Path) -> bool:
    return path.name.endswith(".incomplete") or path.name.endswith(".lock")


def collect_inventory(model_dir: Path) -> dict[str, Any]:
    resolved_dir = model_dir.expanduser().resolve()
    shards: list[ShardRecord] = []
    blockers: list[str] = []
    blocker_records: list[ShardRecord] = []

    if resolved_dir.exists():
        for entry in sorted(resolved_dir.rglob("*")):
            if not entry.is_file():
                continue
            rel = entry.relative_to(resolved_dir).as_posix()
            size = entry.stat().st_size
            if is_blocker_file(entry):
                blockers.append(rel)
                blocker_records.append(ShardRecord(path=rel, size=size))
            if ".cache" not in entry.parts and entry.suffix == ".gguf":
                shards.append(ShardRecord(path=rel, size=size))

    shard_records = [dataclasses.asdict(record) for record in shards]
    blocker_payload = [dataclasses.asdict(record) for record in blocker_records]
    total_bytes = sum(record.size for record in shards)
    shard_count = len(shards)
    status = "ready" if shard_count == REQUIRED_NON_CACHE_SHARDS and not blockers else "blocked"
    primary_shard = str((resolved_dir / shards[0].path).resolve()) if shards else None

    reasons: list[str] = []
    if not resolved_dir.exists():
        reasons.append(f"model directory missing: {resolved_dir}")
    if shard_count != REQUIRED_NON_CACHE_SHARDS:
        reasons.append(
            f"expected {REQUIRED_NON_CACHE_SHARDS} non-cache gguf shards, found {shard_count}"
        )
    if blockers:
        reasons.append(f"found blocker files: {', '.join(blockers)}")

    return {
        "model_dir": str(resolved_dir),
        "status": status,
        "required_non_cache_shards": REQUIRED_NON_CACHE_SHARDS,
        "non_cache_shard_count": shard_count,
        "non_cache_shards": shard_records,
        "primary_shard": primary_shard,
        "blocker_files": blocker_payload,
        "total_shard_bytes": total_bytes,
        "refusal_reasons": reasons,
    }


def build_server_command(
    *,
    binary: Path,
    library_path: Path,
    model_path: Path,
    port: int,
    context_length: int,
    threads: int,
    ubatch: int,
    indexer_top_k: int,
    extra_args: list[str] | None = None,
) -> list[str]:
    command = [
        "env",
        "-i",
        "PATH=/usr/bin:/bin",
        f"LD_LIBRARY_PATH={library_path}",
        "OMP_NUM_THREADS=1",
        "numactl",
        "--interleave=all",
        str(binary),
        "-m",
        str(model_path),
        "--override-kv",
        f"{INDEXER_TOP_K_OVERRIDE_KEY}=int:{indexer_top_k}",
        "--device",
        "none",
        "-ngl",
        "0",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-c",
        str(context_length),
        "-t",
        str(threads),
        "-ub",
        str(ubatch),
        "--log-disable",
    ]
    if extra_args:
        command.extend(extra_args)
    return command


def _build_prompt(task_line: str, context_length: int) -> str:
    target_chars = max(0, (context_length - PROMPT_RESERVE_TOKENS) * 4)
    repeats = max(1, target_chars // len(FILLER_TEXT) + 1)
    filler = (FILLER_TEXT * repeats)[:target_chars]
    return (
        f"{filler}\n\n--- TASK ---\n"
        f"{task_line}\n"
        "Answer with a single short sentence."
    )


def _prompt_spec(kind: str, context_length: int, task_line: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "context_length": context_length,
        "task_line": task_line,
    }


def _server_spec(
    *,
    binary: Path,
    library_path: Path,
    model_path: Path,
    port: int,
    context_length: int,
    threads: int,
    ubatch: int,
    indexer_top_k: int,
) -> dict[str, Any]:
    command = build_server_command(
        binary=binary,
        library_path=library_path,
        model_path=model_path,
        port=port,
        context_length=context_length,
        threads=threads,
        ubatch=ubatch,
        indexer_top_k=indexer_top_k,
    )
    return {
        "server_command": command,
        "server_command_shell": shlex.join(command),
        "port": port,
        "context_length": context_length,
        "threads": threads,
        "ubatch": ubatch,
        "indexer_top_k": indexer_top_k,
        "indexer_top_k_override": f"{INDEXER_TOP_K_OVERRIDE_KEY}=int:{indexer_top_k}",
    }


def build_plan(args: argparse.Namespace, inventory: dict[str, Any], binary: Path, library_path: Path) -> dict[str, Any]:
    short_task = "Return READY after the short load/decode smoke."
    long_task = "Return READY after the long-context DSA probe."
    scaling_task = "Return READY for the KV-length scaling checkpoint."

    primary_shard = Path(inventory["primary_shard"]) if inventory["primary_shard"] else args.model_dir

    stages: list[dict[str, Any]] = [
        {
            "name": "shard_integrity_inventory",
            "kind": "inventory",
            "status": inventory["status"],
            "hashing": "disabled",
            "summary": {
                "required_non_cache_shards": REQUIRED_NON_CACHE_SHARDS,
                "non_cache_shard_count": inventory["non_cache_shard_count"],
                "total_shard_bytes": inventory["total_shard_bytes"],
            },
            "blockers": inventory["blocker_files"],
        },
        {
            "name": "short_load_decode_smoke",
            "kind": "load_decode_smoke",
            "status": "ready" if inventory["status"] == "ready" else "blocked",
            "prompt": _prompt_spec("short_load_decode_smoke", args.short_context, short_task),
            "server": _server_spec(
                binary=binary,
                library_path=library_path,
                model_path=primary_shard,
                port=args.port_base + 1,
                context_length=args.short_context,
                threads=args.threads,
                ubatch=args.ubatch,
                indexer_top_k=args.indexer_top_k,
            ),
            "request": {
                "endpoint": "/v1/chat/completions",
                "max_tokens": args.max_tokens,
                "seed": args.seed,
                "temperature": args.temperature,
            },
        },
        {
            "name": "long_context_dsa_probe",
            "kind": "long_context_probe",
            "status": "ready" if inventory["status"] == "ready" else "blocked",
            "prompt": _prompt_spec("long_context_dsa_probe", args.long_context, long_task),
            "server": _server_spec(
                binary=binary,
                library_path=library_path,
                model_path=primary_shard,
                port=args.port_base + 2,
                context_length=args.long_context,
                threads=args.threads,
                ubatch=args.ubatch,
                indexer_top_k=args.indexer_top_k,
            ),
            "request": {
                "endpoint": "/v1/chat/completions",
                "max_tokens": args.max_tokens,
                "seed": args.seed,
                "temperature": args.temperature,
            },
        },
        {
            "name": "kv_length_scaling",
            "kind": "kv_length_scaling",
            "status": "ready" if inventory["status"] == "ready" else "blocked",
            "fixed_indexer_top_k": args.indexer_top_k,
            "series": [
                {
                    "context_length": context_length,
                    "prompt": _prompt_spec("kv_length_scaling", context_length, scaling_task),
                    "server": _server_spec(
                        binary=binary,
                        library_path=library_path,
                        model_path=primary_shard,
                        port=args.port_base + 10 + idx,
                        context_length=context_length,
                        threads=args.threads,
                        ubatch=args.ubatch,
                        indexer_top_k=args.indexer_top_k,
                    ),
                    "request": {
                        "endpoint": "/v1/chat/completions",
                        "max_tokens": args.max_tokens,
                        "seed": args.seed,
                        "temperature": args.temperature,
                    },
                }
                for idx, context_length in enumerate(args.kv_contexts)
            ],
        },
    ]

    plan = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "execute" if args.execute else "dry-run",
        "model_dir": str(args.model_dir.resolve()),
        "model_path": str(primary_shard),
        "binary": str(binary),
        "library_path": str(library_path),
        "execution_allowed": inventory["status"] == "ready",
        "refusal_reasons": inventory["refusal_reasons"],
        "inventory": inventory,
        "stages": stages,
        "execution": None,
    }
    return plan


def write_plan(output_path: Path, plan: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_canonical_json(plan) + "\n", encoding="utf-8")


def wait_for_health(port: int, timeout_s: int = 180) -> None:
    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = resp.read().decode("utf-8", errors="replace").strip().lower()
                if "ok" in body:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"server on port {port} did not become healthy within {timeout_s}s")


def call_completion(port: int, prompt: str, max_tokens: int, temperature: float, seed: int) -> dict[str, Any]:
    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "stream": False,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def launch_server(command: list[str]) -> subprocess.Popen[str]:
    return subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )


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
            return

    if proc.poll() is None:
        send(signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            send(signal.SIGKILL)
            proc.wait(timeout=20)

    ps = subprocess.run(
        ["ps", "-p", str(pid), "-o", "pid="],
        check=False,
        capture_output=True,
        text=True,
    )
    if ps.returncode == 0 and ps.stdout.strip() == str(pid):
        raise RuntimeError(f"server pid {pid} still visible after stop")


def run_stage(stage: dict[str, Any]) -> dict[str, Any]:
    server_command = stage["server"]["server_command"]
    prompt = _build_prompt(stage["prompt"]["task_line"], stage["prompt"]["context_length"])
    proc = launch_server(server_command)
    try:
        wait_for_health(stage["server"]["port"])
        response = call_completion(
            stage["server"]["port"],
            prompt,
            stage["request"]["max_tokens"],
            stage["request"]["temperature"],
            stage["request"]["seed"],
        )
        return {
            "name": stage["name"],
            "status": "ok",
            "port": stage["server"]["port"],
            "context_length": stage["server"]["context_length"],
            "prompt_kind": stage["prompt"]["kind"],
            "usage": response.get("usage", {}),
        }
    finally:
        terminate_server(proc)


def run_execution(plan: dict[str, Any]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for stage in plan["stages"]:
        if stage["status"] != "ready":
            results.append({"name": stage["name"], "status": "skipped", "reason": "preflight blocked"})
            continue
        if stage["kind"] == "inventory":
            results.append({"name": stage["name"], "status": "checked"})
            continue
        if stage["kind"] == "kv_length_scaling":
            series_results = []
            for entry in stage["series"]:
                series_results.append(run_stage({
                    "name": stage["name"],
                    "prompt": entry["prompt"],
                    "server": entry["server"],
                    "request": entry["request"],
                }))
            results.append({
                "name": stage["name"],
                "status": "ok",
                "fixed_indexer_top_k": stage["fixed_indexer_top_k"],
                "series": series_results,
            })
            continue
        results.append(run_stage(stage))
    return {
        "status": "ok",
        "stages": results,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GLM-5.2 DSA probe runner")
    parser.add_argument("--execute", action="store_true", help="Execute the staged probe after preflight")
    parser.add_argument(
        "--output",
        type=Path,
        default=RESEARCH_ROOT / "data" / "glm52_dsa_probe" / _utc_timestamp() / "plan.json",
        help="Plan JSON output path",
    )
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR, help="GLM shard directory")
    parser.add_argument(
        "--binary",
        type=Path,
        default=DEFAULT_BINARY,
        help="llama-server binary (experimental v7 default)",
    )
    parser.add_argument(
        "--library-path",
        type=Path,
        default=None,
        help="Pinned LD_LIBRARY_PATH directory (defaults to the binary directory)",
    )
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS, help="llama-server threads")
    parser.add_argument("--ubatch", type=int, default=DEFAULT_UBATCH, help="llama-server ubatch")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Request max_tokens")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Request seed")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Request temperature")
    parser.add_argument("--indexer-top-k", type=int, default=DEFAULT_INDEXER_TOP_K, help="Fixed DSA indexer_top_k")
    parser.add_argument(
        "--short-context",
        type=int,
        default=DEFAULT_SHORT_CONTEXT,
        help="Context length for the short load/decode smoke",
    )
    parser.add_argument(
        "--long-context",
        type=int,
        default=DEFAULT_LONG_CONTEXT,
        help="Context length for the long-context DSA probe",
    )
    parser.add_argument(
        "--kv-contexts",
        nargs="+",
        type=int,
        default=list(DEFAULT_KV_CONTEXTS),
        help="Context lengths for the KV-length scaling series",
    )
    parser.add_argument(
        "--port-base",
        type=int,
        default=DEFAULT_PORT_BASE,
        help="Base port for staged server commands",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    binary = resolve_binary(args.binary)
    library_path = resolve_library_path(binary, args.library_path)
    inventory = collect_inventory(args.model_dir)
    plan = build_plan(args, inventory, binary, library_path)
    write_plan(args.output, plan)

    if not args.execute:
        print(f"[glm52] mode: dry-run")
        print(f"[glm52] plan: {args.output}")
        if plan["execution_allowed"]:
            print("[glm52] preflight: ready for execute mode")
        else:
            print("[glm52] preflight: blocked")
            for reason in plan["refusal_reasons"]:
                print(f"[glm52]   - {reason}")
        return 0

    if not plan["execution_allowed"]:
        print("[glm52] refusing execute mode: preflight blocked", file=sys.stderr)
        for reason in plan["refusal_reasons"]:
            print(f"[glm52]   - {reason}", file=sys.stderr)
        return 3

    execution = run_execution(plan)
    plan["execution"] = execution
    write_plan(args.output, plan)
    print(f"[glm52] execute complete: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
