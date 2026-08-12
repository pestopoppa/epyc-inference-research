#!/usr/bin/env python3
"""Capture clean-source whole-model kernel attribution with rocprof v1.

This producer exists for workloads that crash rocprofv2.  It is diagnostic
only: it binds a clean source commit, binary, model, profiler and linkage;
holds the MI210 claim; samples device state; and reports device-timestamp wall
share without turning that attribution into a performance verdict.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark.capture_autokernel_c4_profile import (
    artifact_inventory,
    assert_source_identity,
)
from scripts.benchmark.autokernel_claimed_sampling import (
    error_payload,
    stop_sampler_and_release,
)
from scripts.benchmark.run_autokernel_gpu_factorial import (
    sha256_file,
    terminate_owned,
    write_json_atomic,
)
from scripts.kernel_rnd.autokernel import storage
from scripts.kernel_rnd.autokernel.execution import device_sampler
from scripts.kernel_rnd.autokernel.resource import device_claim


SCHEMA = "epyc.autokernel.rocprofv1_attribution.v1"
PROFILER_NAME = "rocprof-v1/device-timestamps"
_PROMPT_RE = re.compile(r"[1-9][0-9]*")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def prompt_tokens(value: str) -> tuple[int, ...]:
    fields = value.split(",")
    if not fields or any(not _PROMPT_RE.fullmatch(field) for field in fields):
        raise argparse.ArgumentTypeError("prompt tokens must be comma-separated positive integers")
    parsed = tuple(int(field) for field in fields)
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("prompt token values must be unique")
    return parsed


def workload_phase(gen_tokens: int) -> str:
    if gen_tokens < 0:
        raise ValueError("generation tokens must be non-negative")
    return "prefill" if gen_tokens == 0 else "prefill+decode"


def bench_command(binary: Path, model: Path, *, tokens: int,
                  repetitions: int, gen_tokens: int = 0) -> tuple[str, ...]:
    workload_phase(gen_tokens)
    return (
        str(binary), "-m", str(model), "-p", str(tokens),
        "-n", str(gen_tokens),
        "-r", str(repetitions), "-ngl", "99", "-fa", "on", "-o", "jsonl",
    )


def profiler_environment(binary: Path, args: argparse.Namespace) -> dict[str, str]:
    prefix = Path(args.profiler_prefix).resolve()
    root = Path(args.profiler_root).resolve()
    env = os.environ.copy()
    env.update({
        "ROCM_PATH": "/opt/rocm",
        "ROCP_METRICS": str(prefix / "lib" / "rocprofiler" / "metrics.xml"),
        "GGML_CUDA_DISABLE_GRAPHS": "1",
        "PATH": f"{prefix / 'bin'}:/opt/rocm/bin:{env['PATH']}",
        "LD_LIBRARY_PATH": (
            f"{prefix / 'lib'}:{root / 'usr/lib/x86_64-linux-gnu'}:"
            f"{binary.parent}:/opt/rocm/lib"),
    })
    return env


def profile_command(binary: Path, model: Path, *, tokens: int,
                    repetitions: int, profiler: Path, input_file: Path,
                    output_file: Path, gen_tokens: int = 0) -> tuple[str, ...]:
    return (
        str(profiler), "--tool-version", "1", "--timestamp", "on",
        "--ctx-wait", "on", "--heartbeat", "30", "-i", str(input_file),
        "-o", str(output_file),
        *bench_command(binary, model, tokens=tokens, repetitions=repetitions,
                       gen_tokens=gen_tokens),
    )


def run_owned(command: tuple[str, ...], *, env: dict[str, str],
              timeout_s: float) -> tuple[int, str, str, float]:
    started = time.monotonic()
    process = subprocess.Popen(
        command, env=env, stdin=subprocess.DEVNULL, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except BaseException:
        if process.poll() is None:
            terminate_owned(process)
        raise
    return process.returncode, stdout, stderr, time.monotonic() - started


def parse_bench_result(stdout: str, stderr: str, *, tokens: int,
                       repetitions: int) -> dict[str, Any]:
    rows = []
    for line in (stdout + "\n" + stderr).splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("n_prompt") == tokens:
            rows.append(value)
    if len(rows) != 1:
        raise RuntimeError(
            f"expected one llama-bench JSON row for p{tokens}, observed {len(rows)}")
    row = rows[0]
    if row.get("backends") != "ROCm" or row.get("gpu_info") != "AMD Instinct MI210":
        raise RuntimeError("llama-bench did not execute on ROCm / AMD Instinct MI210")
    if row.get("flash_attn") != 1 or row.get("n_gpu_layers") != 99:
        raise RuntimeError("llama-bench did not retain full GPU offload and flash-attention=on")
    if len(row.get("samples_ns", ())) != repetitions:
        raise RuntimeError("llama-bench did not retain every requested repetition")
    return row


def kernel_family(name: str) -> str:
    if "gated_delta_net" in name:
        return "gated_delta_net"
    if "mul_mat_q" in name:
        return "mul_mat_q"
    if "mul_mat_vec" in name:
        return "mul_mat_vec"
    if "quantize" in name:
        return "quantize"
    if "flash_attn" in name or "fattn" in name:
        return "flash_attention"
    if "rms_norm" in name:
        return "rms_norm"
    if "cpy" in name or "copy" in name:
        return "copy"
    if "fillBuffer" in name:
        return "buffer_fill"
    return "other"


def summarize_timestamps(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        rows = list(reader)
    aliases = {
        "dispatch": ("Dispatch_ID", "Index"),
        "kernel": ("Kernel_Name", "KernelName"),
        "gpu": ("GPU_ID", "gpu-id"),
        "start": ("Start_Timestamp", "BeginNs"),
        "end": ("End_Timestamp", "EndNs"),
    }
    selected = {
        key: next((field for field in candidates if field in fields), None)
        for key, candidates in aliases.items()
    }
    missing = sorted(key for key, value in selected.items() if value is None)
    if missing:
        raise RuntimeError(f"rocprof timestamp table is missing semantic fields: {missing}")
    if not rows:
        raise RuntimeError("rocprof timestamp table is empty")

    grouped: dict[str, list[float]] = defaultdict(list)
    first_start = None
    last_end = None
    for row in rows:
        try:
            start = float(row[selected["start"]])
            end = float(row[selected["end"]])
        except ValueError as exc:
            raise RuntimeError("rocprof emitted a non-numeric device timestamp") from exc
        if end < start:
            raise RuntimeError("rocprof emitted an inverted device timestamp")
        grouped[kernel_family(row[selected["kernel"]])].append(end - start)
        first_start = start if first_start is None else min(first_start, start)
        last_end = end if last_end is None else max(last_end, end)

    total = sum(sum(values) for values in grouped.values())
    gdn = grouped.get("gated_delta_net", [])
    if not gdn:
        raise RuntimeError("rocprof timestamp table has no gated_delta_net dispatch")
    families = []
    for family, values in sorted(grouped.items(), key=lambda item: -sum(item[1])):
        duration = sum(values)
        families.append({
            "family": family,
            "dispatches": len(values),
            "device_duration_ns_total": duration,
            "device_duration_ns_median": statistics.median(values),
            "summed_kernel_time_share": duration / total if total else 0.0,
        })
    return {
        "dispatches": len(rows),
        "summed_kernel_time_ns": total,
        "device_span_ns": (last_end - first_start) if first_start is not None else 0.0,
        "gated_delta_net_share": sum(gdn) / total if total else 0.0,
        "families": families,
    }


def linkage_identity(binary: Path, *, env: dict[str, str]) -> dict[str, Any]:
    completed = subprocess.run(
        ("ldd", str(binary)), env=env, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30.0)
    ggml = []
    for line in completed.stdout.splitlines():
        if "libggml" not in line and "libllama" not in line:
            continue
        target = line.split("=>", 1)[1].strip().split(" ", 1)[0] if "=>" in line else ""
        if not target or not Path(target).is_relative_to(binary.parent):
            raise RuntimeError(f"binary resolves a llama/ggml DSO outside its build: {line.strip()}")
        ggml.append(target)
    if not ggml:
        raise RuntimeError("ldd exposed no llama/ggml linkage")
    return {"resolved_libraries": ggml, "ldd_stdout": completed.stdout}


def identity(binary: Path, model: Path, source_root: Path,
             args: argparse.Namespace, *, env: dict[str, str]) -> dict[str, Any]:
    profiler = Path(args.profiler_prefix).resolve() / "bin" / "rocprof"
    for path, label in ((binary, "binary"), (model, "model"), (profiler, "rocprof v1")):
        if not path.is_file():
            raise RuntimeError(f"{label} is unavailable: {path}")
    commit = assert_source_identity(source_root, args.source_commit)
    return {
        "source_root": str(source_root),
        "source_commit": commit,
        "binary": str(binary),
        "binary_sha256": sha256_file(binary),
        "model": str(model),
        "model_size": model.stat().st_size,
        "model_sha256": sha256_file(model),
        "profiler_name": PROFILER_NAME,
        "profiler": str(profiler),
        "profiler_sha256": sha256_file(profiler),
        "linkage": linkage_identity(binary, env=env),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    binary = Path(args.binary).resolve()
    model = Path(args.model).resolve()
    source_root = Path(args.source_root).resolve()
    output_dir = Path(storage.assert_not_scratch(
        args.output_dir, what="rocprof v1 attribution evidence directory"))
    if output_dir.exists():
        raise RuntimeError(f"rocprof v1 attribution output already exists: {output_dir}")
    env = profiler_environment(binary, args)
    tool_identity = identity(binary, model, source_root, args, env=env)
    output_dir.mkdir(parents=True)
    input_file = output_dir / "timestamps.txt"
    input_file.write_text("pmc:\n\ngpu:\nrange:\nkernel:\n", encoding="utf-8")
    profiler = Path(tool_identity["profiler"])

    claim = device_claim.acquire_device_claim(
        "mi210_0", purpose="AutoKernel K28 rocprof-v1 GDN attribution",
        campaign_id=args.campaign_id,
        journal=device_claim.ClaimJournal(args.claim_journal),
        holder_label="run_autokernel_rocprofv1_attribution.py",
        timeout_s=args.claim_timeout_s,
        max_hold_s=args.timeout_s * (len(args.prompt_tokens) + 1) + 300.0)
    opened = claim.receipt().to_dict()
    sampler = None
    sampling_receipt = None
    captured_error: BaseException | None = None
    results = []
    captured_profiles = []
    started_at = utc_now()
    started = time.monotonic()
    released = None
    teardown_errors: tuple[BaseException, ...] = ()
    try:
        sampler = device_sampler.RocmSmiSampler(
            device_index=0, interval_s=0.250).start()
        rc, stdout, stderr, duration = run_owned(
            bench_command(binary, model, tokens=args.preflight_tokens, repetitions=1),
            env=env, timeout_s=args.timeout_s)
        (output_dir / "preflight.stdout.jsonl").write_text(stdout, encoding="utf-8")
        (output_dir / "preflight.stderr.txt").write_text(stderr, encoding="utf-8")
        if rc != 0:
            raise RuntimeError(f"llama-bench preflight exited {rc}")
        preflight = parse_bench_result(stdout, stderr, tokens=args.preflight_tokens, repetitions=1)
        preflight["duration_s"] = duration

        for tokens in args.prompt_tokens:
            raw = output_dir / f"p{tokens}.timestamps.csv"
            command = profile_command(
                binary, model, tokens=tokens, repetitions=args.repetitions,
                profiler=profiler, input_file=input_file, output_file=raw,
                gen_tokens=args.gen_tokens)
            rc, stdout, stderr, duration = run_owned(
                command, env=env, timeout_s=args.timeout_s)
            (output_dir / f"p{tokens}.stdout.jsonl").write_text(stdout, encoding="utf-8")
            (output_dir / f"p{tokens}.stderr.txt").write_text(stderr, encoding="utf-8")
            if rc != 0:
                raise RuntimeError(f"rocprof v1 p{tokens} exited {rc}")
            bench = parse_bench_result(
                stdout, stderr, tokens=tokens, repetitions=args.repetitions)
            captured_profiles.append({
                "prompt_tokens": tokens,
                "command": list(command),
                "duration_s": duration,
                "bench": bench,
                "timestamp_csv": str(raw),
                "timestamp_csv_sha256": sha256_file(raw),
            })
    except BaseException as exc:
        captured_error = exc
        preflight = locals().get("preflight")
    finally:
        sampling_receipt, released_receipt, teardown_errors = stop_sampler_and_release(
            sampler=sampler, claim=claim)
        released = released_receipt.to_dict() if released_receipt is not None else None
    if teardown_errors and captured_error is None:
        captured_error = teardown_errors[0]
    if captured_error is None:
        try:
            for profile in captured_profiles:
                profile["attribution"] = summarize_timestamps(
                    Path(profile["timestamp_csv"]))
                results.append(profile)
        except BaseException as exc:
            captured_error = exc
    belief_measurements = []
    if captured_error is None:
        for profile in results:
            tokens = profile["prompt_tokens"]
            share = profile["attribution"]["gated_delta_net_share"]
            belief_measurements.append({
                "measurement_id": f"gdn_share_p{tokens}",
                "metric": "gated_delta_net_summed_kernel_time_share",
                "value": share,
                "unit": "fraction",
                "metric_direction": "lower_better",
                "category": "BASELINE",
                "reps": args.repetitions,
                "reps_basis": "scored:llama-bench prompt repetitions",
                "claim": (
                    f"Qwen3.6-35B-A3B Q8 gfx90a p{tokens} gated-delta-net "
                    f"summed kernel-time share is {share:.12f}"),
                "extra": {"prompt_tokens": tokens},
            })
    payload = {
        "schema": SCHEMA,
        "status": "failed" if captured_error is not None else "passed",
        "authority": "diagnostic_only",
        "campaign_id": args.campaign_id,
        "started_at": started_at,
        "ended_at": utc_now(),
        "duration_s": time.monotonic() - started,
        "identity": tool_identity,
        "workload": {
            "prompt_tokens": list(args.prompt_tokens),
            "gen_tokens": args.gen_tokens,
            "phase": workload_phase(args.gen_tokens),
            "preflight_tokens": args.preflight_tokens,
            "repetitions": args.repetitions,
            "graphs_disabled": True,
        },
        "preflight": preflight,
        "profiles": results,
        "belief_measurements": belief_measurements,
        "captured_profiles": captured_profiles if captured_error is not None else None,
        "device_claim_open": opened,
        "device_claim_released": released,
        "device_sampling": sampling_receipt.to_dict() if sampling_receipt is not None else None,
        "teardown_errors": error_payload(teardown_errors),
        "error": None if captured_error is None else {
            "type": type(captured_error).__name__,
            "message": str(captured_error),
        },
    }
    payload["artifacts"] = artifact_inventory(output_dir)
    write_json_atomic(output_dir / "receipt.json", payload)
    if captured_error is not None:
        raise RuntimeError(
            f"rocprof v1 attribution failed; durable receipt: {output_dir / 'receipt.json'}") from captured_error
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-root", required=True)
    result.add_argument("--source-commit")
    result.add_argument("--binary", required=True)
    result.add_argument("--model", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--campaign-id", default="k28-rocprofv1-attribution-20260811")
    result.add_argument("--prompt-tokens", type=prompt_tokens, default=(2048, 8192, 32768))
    result.add_argument(
        "--gen-tokens", type=int, default=0,
        help="tokens to generate (llama-bench -n). Default 0 = PREFILL ONLY. "
             "Set >0 to reach the batch-1 decode MMVQ/GEMV path; a decode "
             "question answered with the default silently measures prefill.")
    result.add_argument("--preflight-tokens", type=int, default=32)
    result.add_argument("--repetitions", type=int, default=1)
    result.add_argument("--profiler-root", default="/mnt/raid0/llm/tools/rocm-profilers-6.2")
    result.add_argument("--profiler-prefix", default="/mnt/raid0/llm/tools/rocm-profilers-6.2/opt/rocm-6.2.0")
    result.add_argument("--claim-journal", default="/mnt/raid0/llm/ak-claims/device.jsonl")
    result.add_argument("--claim-timeout-s", type=float, default=0.0)
    result.add_argument("--timeout-s", type=float, default=1800.0)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.repetitions < 1 or args.preflight_tokens < 1 or args.gen_tokens < 0:
        raise RuntimeError(
            "repetitions and preflight tokens must be positive; "
            "generation tokens must be non-negative")
    payload = run(args)
    print(json.dumps({
        "receipt": str(Path(args.output_dir) / "receipt.json"),
        "status": payload["status"],
        "profiles": len(payload["profiles"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
