#!/usr/bin/env python3
"""Capture a current gfx90a wall-share map for AutoKernel hypothesis G15.

G15 is a target-selection hypothesis, not a performance claim: the batched
decode elementwise/norm tail must account for at least 20% of summed device
kernel time before AutoKernel spends an authoring campaign on fusion.  This
runner binds a clean source, binary, model, profiler and device trace and emits
the exact kernel/family clusters that support that decision.
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
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark.autokernel_claimed_sampling import (
    error_payload,
    stop_sampler_and_release,
)
from scripts.benchmark.capture_autokernel_c4_profile import (
    artifact_inventory,
    assert_source_identity,
)
from scripts.benchmark.run_autokernel_gpu_factorial import (
    sha256_file,
    terminate_owned,
    write_json_atomic,
)
from scripts.kernel_rnd.autokernel import storage
from scripts.kernel_rnd.autokernel.execution import device_sampler
from scripts.kernel_rnd.autokernel.resource import device_claim


SCHEMA = "epyc.autokernel.g15_profile.v1"
PROFILER_NAME = "rocprof-v1/device-timestamps"
HYPOTHESIS_ID = "akh-g15-elementwise-fusion"
TAXONOMY_ID = "epyc.autokernel.g15_kernel_taxonomy.v1"
MIN_TARGET_SHARE = 0.20
TARGET_FAMILIES = frozenset(("norm", "activation", "elementwise"))
ADJACENT_FAMILIES = TARGET_FAMILIES | {"copy_convert"}
_POSITIVE_INT_RE = re.compile(r"[1-9][0-9]*")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def positive_ints(value: str) -> tuple[int, ...]:
    fields = value.split(",")
    if not fields or any(not _POSITIVE_INT_RE.fullmatch(field) for field in fields):
        raise argparse.ArgumentTypeError("values must be comma-separated positive integers")
    parsed = tuple(int(field) for field in fields)
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("values must be unique")
    return parsed


def bench_command(binary: Path, model: Path, *, parallel: int,
                  prompt_tokens: int, generation_tokens: int,
                  context: int, batch: int, ubatch: int) -> tuple[str, ...]:
    return (
        str(binary), "-m", str(model), "-ngl", "99", "-fa", "off",
        "-c", str(context), "-b", str(batch), "-ub", str(ubatch),
        "-npp", str(prompt_tokens), "-ntg", str(generation_tokens),
        "-npl", str(parallel), "--output-format", "jsonl",
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


def profile_command(bench: tuple[str, ...], *, profiler: Path,
                    input_file: Path, output_file: Path) -> tuple[str, ...]:
    return (
        str(profiler), "--tool-version", "1", "--timestamp", "on",
        "--ctx-wait", "on", "--heartbeat", "30", "-i", str(input_file),
        "-o", str(output_file), *bench,
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


def parse_bench_result(stdout: str, stderr: str, *, parallel: int,
                       prompt_tokens: int, generation_tokens: int) -> dict[str, Any]:
    rows = []
    for line in (stdout + "\n" + stderr).splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (isinstance(value, dict) and value.get("pl") == parallel
                and value.get("pp") == prompt_tokens
                and value.get("tg") == generation_tokens):
            rows.append(value)
    if len(rows) != 1:
        raise RuntimeError(
            f"expected one batched-bench row for pl={parallel}, observed {len(rows)}")
    row = rows[0]
    if row.get("n_gpu_layers") != 99 or row.get("flash_attn") != 0:
        raise RuntimeError("batched-bench did not retain full GPU offload / flash-attention=off")
    for field in ("t_tg", "speed_tg"):
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise RuntimeError(f"batched-bench emitted invalid {field}")
    return row


def kernel_family(name: str) -> str:
    text = name.casefold()
    if "gated_delta_net" in text or "ssm_conv" in text:
        return "recurrent"
    if any(token in text for token in (
            "mul_mat", "gemm", "gemv", "rocblas", "tensile", "mfma")) \
            or text.startswith("cijk_"):
        return "matrix"
    if "flash_attn" in text or "fattn" in text or "soft_max" in text:
        return "attention"
    if "norm" in text:
        return "norm"
    if any(token in text for token in (
            "silu", "gelu", "relu", "sigmoid", "softplus", "tanh")):
        return "activation"
    if "get_rows" in text or "set_rows" in text:
        return "gather_scatter"
    if "quantize" in text or "dequantize" in text:
        return "quantization"
    if "concat" in text or "repeat" in text or "contiguous" in text:
        return "layout"
    if any(token in text for token in ("cpy", "copy", "convert")):
        return "copy_convert"
    if "rope" in text:
        return "position"
    if any(token in text for token in (
            "op_add(", "op_mul(", "add_f32", "mul_f32", "scale_", "clamp",
            "unary_", "arange", "argsort", "k_acc")):
        return "elementwise"
    if "fillbuffer" in text or "memset" in text:
        return "buffer_fill"
    return "other"


def _semantic_fields(fields: Iterable[str]) -> dict[str, str]:
    available = set(fields)
    aliases = {
        "kernel": ("Kernel_Name", "KernelName"),
        "start": ("Start_Timestamp", "BeginNs"),
        "end": ("End_Timestamp", "EndNs"),
    }
    selected = {
        key: next((field for field in names if field in available), None)
        for key, names in aliases.items()
    }
    missing = sorted(key for key, value in selected.items() if value is None)
    if missing:
        raise RuntimeError(f"rocprof timestamp table is missing semantic fields: {missing}")
    return {key: value for key, value in selected.items() if value is not None}


def _clusters(rows: list[dict[str, Any]], total: float,
              allowed: frozenset[str] | set[str]) -> list[dict[str, Any]]:
    clusters: Counter[tuple[tuple[str, str], ...]] = Counter()
    durations: defaultdict[tuple[tuple[str, str], ...], float] = defaultdict(float)
    current: list[dict[str, Any]] = []

    def flush() -> None:
        if not current:
            return
        sequence = tuple((row["family"], row["kernel"]) for row in current)
        clusters[sequence] += 1
        durations[sequence] += sum(row["duration_ns"] for row in current)
        current.clear()

    for row in rows:
        if row["family"] in allowed:
            current.append(row)
        else:
            flush()
    flush()
    return [
        {
            "family_sequence": [family for family, _kernel in sequence],
            "kernel_sequence": [kernel for _family, kernel in sequence],
            "occurrences": clusters[sequence],
            "device_duration_ns_total": durations[sequence],
            "summed_kernel_time_share": durations[sequence] / total if total else 0.0,
        }
        for sequence in sorted(
            clusters, key=lambda item: (-durations[item], item))
    ]


def summarize_timestamps(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        selected = _semantic_fields(reader.fieldnames or ())
        raw = list(reader)
    if not raw:
        raise RuntimeError("rocprof timestamp table is empty")

    rows = []
    by_family: defaultdict[str, list[float]] = defaultdict(list)
    by_kernel: defaultdict[str, list[float]] = defaultdict(list)
    first_start = None
    last_end = None
    for source in raw:
        try:
            start = float(source[selected["start"]])
            end = float(source[selected["end"]])
        except ValueError as exc:
            raise RuntimeError("rocprof emitted a non-numeric device timestamp") from exc
        if end < start:
            raise RuntimeError("rocprof emitted an inverted device timestamp")
        name = source[selected["kernel"]]
        family = kernel_family(name)
        duration = end - start
        rows.append({"start_ns": start, "kernel": name, "family": family,
                     "duration_ns": duration})
        by_family[family].append(duration)
        by_kernel[name].append(duration)
        first_start = start if first_start is None else min(first_start, start)
        last_end = end if last_end is None else max(last_end, end)
    rows.sort(key=lambda row: (row["start_ns"], row["kernel"]))
    total = sum(sum(values) for values in by_family.values())
    target_duration = sum(
        sum(by_family.get(family, ())) for family in TARGET_FAMILIES)
    adjacent_duration = sum(
        sum(by_family.get(family, ())) for family in ADJACENT_FAMILIES)

    def table(grouped: dict[str, list[float]], key: str) -> list[dict[str, Any]]:
        result = []
        for label, values in sorted(grouped.items(), key=lambda item: (-sum(item[1]), item[0])):
            duration = sum(values)
            result.append({
                key: label, "dispatches": len(values),
                "device_duration_ns_total": duration,
                "device_duration_ns_median": statistics.median(values),
                "summed_kernel_time_share": duration / total if total else 0.0,
            })
        return result

    return {
        "dispatches": len(rows),
        "taxonomy_id": TAXONOMY_ID,
        "summed_kernel_time_ns": total,
        "device_span_ns": (last_end - first_start) if first_start is not None else 0.0,
        "target_families": sorted(TARGET_FAMILIES),
        "elementwise_norm_target_share": target_duration / total if total else 0.0,
        "adjacent_fusion_surface_families": sorted(ADJACENT_FAMILIES),
        "adjacent_fusion_surface_share": adjacent_duration / total if total else 0.0,
        "family_table": table(by_family, "family"),
        "kernel_table": table(by_kernel, "kernel"),
        "target_cluster_table": _clusters(rows, total, TARGET_FAMILIES),
        "adjacent_cluster_table": _clusters(rows, total, ADJACENT_FAMILIES),
    }


def hypothesis_result(share: float, *, parallel: int) -> dict[str, Any]:
    return {
        "hypothesis_id": HYPOTHESIS_ID,
        "question": (
            f"is the B={parallel} elementwise/norm tail large enough "
            "to fund fusion authoring?"),
        "minimum_target_share": MIN_TARGET_SHARE,
        "observed_target_share": share,
        "verdict": (
            "READY_PROFILE_SELECTED" if share >= MIN_TARGET_SHARE
            else "FALSIFIED_PROFILE_TARGET"),
        "authority": "target_selection_only",
    }


def linkage_identity(binary: Path, *, env: dict[str, str]) -> dict[str, Any]:
    completed = subprocess.run(
        ("ldd", str(binary)), env=env, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30.0)
    libraries = []
    for line in completed.stdout.splitlines():
        if "libggml" not in line and "libllama" not in line:
            continue
        target = line.split("=>", 1)[1].strip().split(" ", 1)[0] if "=>" in line else ""
        if not target or not Path(target).is_relative_to(binary.parent):
            raise RuntimeError(f"binary resolves a llama/ggml DSO outside its build: {line.strip()}")
        libraries.append(target)
    if not libraries:
        raise RuntimeError("ldd exposed no llama/ggml linkage")
    return {"resolved_libraries": libraries, "ldd_stdout": completed.stdout}


def identity(binary: Path, model: Path, source_root: Path,
             args: argparse.Namespace, *, env: dict[str, str]) -> dict[str, Any]:
    profiler = Path(args.profiler_prefix).resolve() / "bin" / "rocprof"
    for path, label in ((binary, "binary"), (model, "model"), (profiler, "rocprof v1")):
        if not path.is_file():
            raise RuntimeError(f"{label} is unavailable: {path}")
    commit = assert_source_identity(source_root, args.source_commit)
    return {
        "source_root": str(source_root), "source_commit": commit,
        "binary": str(binary), "binary_sha256": sha256_file(binary),
        "model": str(model), "model_size": model.stat().st_size,
        "model_sha256": sha256_file(model),
        "profiler_name": PROFILER_NAME, "profiler": str(profiler),
        "profiler_sha256": sha256_file(profiler),
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "linkage": linkage_identity(binary, env=env),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    binary = Path(args.binary).resolve()
    model = Path(args.model).resolve()
    source_root = Path(args.source_root).resolve()
    output_dir = Path(storage.assert_not_scratch(
        args.output_dir, what="G15 profile evidence directory"))
    if output_dir.exists():
        raise RuntimeError(f"G15 output already exists: {output_dir}")
    env = profiler_environment(binary, args)
    tool_identity = identity(binary, model, source_root, args, env=env)
    output_dir.mkdir(parents=True)
    input_file = output_dir / "timestamps.txt"
    input_file.write_text("pmc:\n\ngpu:\nrange:\nkernel:\n", encoding="utf-8")
    profiler = Path(tool_identity["profiler"])

    claim = device_claim.acquire_device_claim(
        "mi210_0", purpose="AutoKernel G15 B=128 wall-share selection",
        campaign_id=args.campaign_id,
        journal=device_claim.ClaimJournal(args.claim_journal),
        holder_label="run_autokernel_g15_profile.py",
        timeout_s=args.claim_timeout_s,
        max_hold_s=args.timeout_s * (len(args.parallel) + 1) + 300.0)
    opened = claim.receipt().to_dict()
    sampler = None
    sampling_receipt = None
    captured_error: BaseException | None = None
    captured_profiles = []
    profiles = []
    started_at = utc_now()
    started = time.monotonic()
    released = None
    teardown_errors: tuple[BaseException, ...] = ()
    preflight = None
    try:
        sampler = device_sampler.RocmSmiSampler(device_index=0, interval_s=0.250).start()
        warmup_command = bench_command(
            binary, model, parallel=args.parallel[0], prompt_tokens=args.prompt_tokens,
            generation_tokens=args.generation_tokens, context=args.context,
            batch=args.batch, ubatch=args.ubatch)
        rc, stdout, stderr, duration = run_owned(
            warmup_command, env=env, timeout_s=args.timeout_s)
        (output_dir / "warmup.stdout.jsonl").write_text(stdout, encoding="utf-8")
        (output_dir / "warmup.stderr.txt").write_text(stderr, encoding="utf-8")
        if rc != 0:
            raise RuntimeError(f"G15 warmup exited {rc}")
        preflight = parse_bench_result(
            stdout, stderr, parallel=args.parallel[0],
            prompt_tokens=args.prompt_tokens, generation_tokens=args.generation_tokens)
        preflight["duration_s"] = duration

        for parallel in args.parallel:
            raw = output_dir / f"b{parallel}.timestamps.csv"
            bench = bench_command(
                binary, model, parallel=parallel, prompt_tokens=args.prompt_tokens,
                generation_tokens=args.generation_tokens, context=args.context,
                batch=args.batch, ubatch=args.ubatch)
            command = profile_command(
                bench, profiler=profiler, input_file=input_file, output_file=raw)
            rc, stdout, stderr, duration = run_owned(
                command, env=env, timeout_s=args.timeout_s)
            (output_dir / f"b{parallel}.stdout.jsonl").write_text(stdout, encoding="utf-8")
            (output_dir / f"b{parallel}.stderr.txt").write_text(stderr, encoding="utf-8")
            if rc != 0:
                raise RuntimeError(f"rocprof v1 B={parallel} exited {rc}")
            result = parse_bench_result(
                stdout, stderr, parallel=parallel,
                prompt_tokens=args.prompt_tokens, generation_tokens=args.generation_tokens)
            captured_profiles.append({
                "parallel": parallel, "command": list(command), "duration_s": duration,
                "bench": result, "timestamp_csv": str(raw),
                "timestamp_csv_sha256": sha256_file(raw),
            })
    except BaseException as exc:
        captured_error = exc
    finally:
        sampling_receipt, released_receipt, teardown_errors = stop_sampler_and_release(
            sampler=sampler, claim=claim)
        released = released_receipt.to_dict() if released_receipt is not None else None
    if teardown_errors and captured_error is None:
        captured_error = teardown_errors[0]
    if captured_error is None:
        try:
            for captured in captured_profiles:
                profile = dict(captured)
                attribution = summarize_timestamps(Path(profile["timestamp_csv"]))
                profile["attribution"] = attribution
                profile["hypothesis"] = hypothesis_result(
                    attribution["elementwise_norm_target_share"],
                    parallel=profile["parallel"])
                profiles.append(profile)
        except BaseException as exc:
            captured_error = exc

    payload = {
        "schema": SCHEMA,
        "status": "failed" if captured_error is not None else "passed",
        "authority": "diagnostic_target_selection_only",
        "campaign_id": args.campaign_id,
        "started_at": started_at, "ended_at": utc_now(),
        "duration_s": time.monotonic() - started,
        "identity": tool_identity,
        "workload": {
            "parallel": list(args.parallel), "prompt_tokens": args.prompt_tokens,
            "generation_tokens": args.generation_tokens, "context": args.context,
            "batch": args.batch, "ubatch": args.ubatch,
            "graphs_disabled": True, "flash_attention": False,
        },
        "preflight": preflight,
        "profiles": profiles,
        "captured_profiles": captured_profiles if captured_error is not None else None,
        "device_claim_open": opened, "device_claim_released": released,
        "device_sampling": (
            sampling_receipt.to_dict() if sampling_receipt is not None else None),
        "teardown_errors": error_payload(teardown_errors),
        "error": None if captured_error is None else {
            "type": type(captured_error).__name__, "message": str(captured_error)},
    }
    payload["artifacts"] = artifact_inventory(output_dir)
    write_json_atomic(output_dir / "receipt.json", payload)
    if captured_error is not None:
        raise RuntimeError(
            f"G15 profile failed; durable receipt: {output_dir / 'receipt.json'}") from captured_error
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-root", required=True)
    result.add_argument("--source-commit")
    result.add_argument("--binary", required=True)
    result.add_argument("--model", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--campaign-id", default="inf36-g15-profile-20260811")
    result.add_argument("--parallel", type=positive_ints, default=(64, 128))
    result.add_argument("--prompt-tokens", type=int, default=128)
    result.add_argument("--generation-tokens", type=int, default=128)
    result.add_argument("--context", type=int, default=32768)
    result.add_argument("--batch", type=int, default=2048)
    result.add_argument("--ubatch", type=int, default=512)
    result.add_argument(
        "--profiler-root", default="/mnt/raid0/llm/tools/rocm-profilers-6.2")
    result.add_argument(
        "--profiler-prefix",
        default="/mnt/raid0/llm/tools/rocm-profilers-6.2/opt/rocm-6.2.0")
    result.add_argument(
        "--claim-journal", default="/mnt/raid0/llm/ak-claims/device.jsonl")
    result.add_argument("--claim-timeout-s", type=float, default=0.0)
    result.add_argument("--timeout-s", type=float, default=1800.0)
    return result


def main() -> int:
    args = parser().parse_args()
    numeric = (
        args.prompt_tokens, args.generation_tokens, args.context,
        args.batch, args.ubatch)
    if any(value < 1 for value in numeric):
        raise SystemExit("workload dimensions must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
