#!/usr/bin/env python3
"""Capture and render the paired AutoKernel C4 rocprofv2 profile."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark.run_autokernel_gpu_factorial import (
    sha256_file,
    terminate_owned,
    write_json_atomic,
)
from scripts.kernel_rnd.autokernel import prior_art, profile_report, storage
from scripts.kernel_rnd.autokernel.execution import device_sampler
from scripts.kernel_rnd.autokernel.resource import device_claim


SCHEMA = "epyc.autokernel.c4_profile_capture.v1"
PROFILER_ROOT = Path("/mnt/raid0/llm/tools/rocm-profilers-6.2")
PROFILER_PREFIX = PROFILER_ROOT / "opt" / "rocm-6.2.0"
_QUANT_TYPE_RE = re.compile(r"[A-Za-z0-9_]+")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def source_commit(source_root: Path) -> str:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=source_root, check=True,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.strip()


def profiler_environment(binary: Path, *, graphs_disabled: bool) -> dict[str, str]:
    env = os.environ.copy()
    env["ROCM_PATH"] = "/opt/rocm"
    env["PATH"] = f"{PROFILER_PREFIX / 'bin'}:/opt/rocm/bin:{env['PATH']}"
    profiler_libraries = (
        f"{PROFILER_PREFIX / 'lib'}:{PROFILER_ROOT / 'usr/lib/x86_64-linux-gnu'}")
    env["LD_LIBRARY_PATH"] = f"{profiler_libraries}:{binary.parent}:/opt/rocm/lib"
    env["ROCP_METRICS"] = str(PROFILER_PREFIX / "lib/rocprofiler/metrics.xml")
    if graphs_disabled:
        env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
    else:
        env.pop("GGML_CUDA_DISABLE_GRAPHS", None)
    return env


def bench_command(binary: Path, model: Path | None, *, repetitions: int,
                  args: argparse.Namespace) -> tuple[str, ...]:
    if args.workload_kind in ("q4k-op", "quant-op"):
        quant_type = "q4_K" if args.workload_kind == "q4k-op" else args.quant_type
        if not isinstance(quant_type, str) or not _QUANT_TYPE_RE.fullmatch(quant_type):
            raise RuntimeError(
                "--quant-type must contain only letters, digits, and underscores")
        case_pattern = (
            rf"^type_a={quant_type},type_b=f32,m={args.op_m},"
            rf"n={args.op_n},k={args.op_k}.*$")
        base = (
            str(binary), "test", "-o", "MUL_MAT", "-b", "ROCm0",
            "-p", case_pattern,
            "--suite-seed", str(args.suite_seed),
            "--repeat-suite", str(repetitions), "--output", "csv",
        )
        return base
    if model is None:
        raise RuntimeError("llama-prefill C4 workload requires --model")
    return (
        str(binary), "-m", str(model), "-p", str(args.prompt_tokens),
        "-n", "0", "-r", str(repetitions), "-ngl", "99", "-fa", "on",
        "-o", "jsonl",
    )


def run_owned(command: tuple[str, ...], *, env: dict[str, str], stdout: Path,
              stderr: Path, timeout_s: float) -> float:
    started = time.monotonic()
    with stdout.open("wb") as stdout_handle, stderr.open("wb") as stderr_handle:
        process = subprocess.Popen(
            command, env=env, stdin=subprocess.DEVNULL, stdout=stdout_handle,
            stderr=stderr_handle, start_new_session=True)
        try:
            returncode = process.wait(timeout=timeout_s)
        except BaseException:
            if process.poll() is None:
                terminate_owned(process)
            raise
    if returncode != 0:
        tail = stderr.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"command exited {returncode}: {tail!r}")
    return time.monotonic() - started


def validate_bench_output(path: Path, *, repetitions: int,
                          args: argparse.Namespace) -> dict:
    if args.workload_kind in ("q4k-op", "quant-op"):
        text = path.read_text(encoding="utf-8")
        header = '"backend_name","op_name","op_params"'
        if text.count(header) != 1:
            raise RuntimeError(
                f"C4 op process emitted {text.count(header)} suite headers, expected one")
        rows = sum(1 for line in text.splitlines() if line.startswith('"ROCm0","MUL_MAT"'))
        if rows < repetitions:
            quant_type = (
                "q4_K" if args.workload_kind == "q4k-op" else args.quant_type)
            raise RuntimeError(f"C4 op loop emitted no tested {quant_type} rows")
        return {"suite_invocations": 1, "test_repetitions": repetitions,
                "tested_rows": rows}
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"llama-bench emitted {len(lines)} result rows, expected one")
    row = json.loads(lines[0])
    if row.get("backends") != "ROCm" or row.get("gpu_info") != "AMD Instinct MI210":
        raise RuntimeError("C4 workload did not execute on ROCm0 / MI210")
    if row.get("flash_attn") != 1:
        raise RuntimeError("C4 workload did not preserve explicit flash-attention=on")
    if not isinstance(row.get("samples_ns"), list) or len(row["samples_ns"]) != repetitions:
        raise RuntimeError("C4 workload did not retain every timing repetition")
    return row


def capture_role(*, role: str, binary: Path, model: Path | None, args: argparse.Namespace,
                 output_dir: Path) -> dict:
    graphs_disabled = role == "mapping"
    env = profiler_environment(binary, graphs_disabled=graphs_disabled)
    warmup_stdout = output_dir / f"{role}.warmup.stdout.jsonl"
    warmup_stderr = output_dir / f"{role}.warmup.stderr.txt"
    warmup_command = bench_command(
        binary, model, repetitions=profile_report.DEFAULT_WARMUP_STEPS, args=args)
    warmup_duration = run_owned(
        warmup_command, env=env, stdout=warmup_stdout, stderr=warmup_stderr,
        timeout_s=args.arm_timeout_s)
    warmup_row = validate_bench_output(
        warmup_stdout, repetitions=profile_report.DEFAULT_WARMUP_STEPS, args=args)

    raw_dir = output_dir / f"{role}.raw"
    raw_dir.mkdir()
    active_command = bench_command(
        binary, model, repetitions=profile_report.DEFAULT_ACTIVE_STEPS, args=args)
    profile_command = (
        str(PROFILER_PREFIX / "bin" / "rocprofv2"), "--kernel-trace",
        "--plugin", "file", "-d", str(raw_dir), "-o", role,
        *active_command,
    )
    active_stdout = output_dir / f"{role}.active.stdout.jsonl"
    active_stderr = output_dir / f"{role}.active.stderr.txt"
    active_duration = run_owned(
        profile_command, env=env, stdout=active_stdout, stderr=active_stderr,
        timeout_s=args.arm_timeout_s)
    active_row = validate_bench_output(
        active_stdout, repetitions=profile_report.DEFAULT_ACTIVE_STEPS, args=args)
    profile_path = raw_dir / f"results_{role}.csv"
    if not profile_path.is_file():
        raise RuntimeError(f"rocprofv2 did not emit {profile_path}")
    profile_hash = sha256_file(profile_path)
    receipt = prior_art.ProfileReceipt(
        corpus_id=f"c4-{args.campaign_id}-{role}",
        workload_id=args.workload_id,
        profile_path=str(profile_path),
        profile_sha256=profile_hash,
        source_commit=args.source_commit,
    )
    dispatches = prior_art.load_rocprof_dispatches(profile_path, receipt)
    return {
        "role": role,
        "attribution_mode": "graphs_disabled" if graphs_disabled
                            else profile_report.FORMAL_MODE,
        "warmup_steps": profile_report.DEFAULT_WARMUP_STEPS,
        "active_steps": profile_report.DEFAULT_ACTIVE_STEPS,
        "warmup_command": list(warmup_command),
        "profile_command": list(profile_command),
        "warmup_duration_s": warmup_duration,
        "active_duration_s": active_duration,
        "warmup_result": warmup_row,
        "active_result": active_row,
        "profile_path": str(profile_path),
        "profile_sha256": profile_hash,
        "dispatches": len(dispatches),
        "receipt": receipt,
    }


def manifest_for(mapping: dict, formal: dict, *, args: argparse.Namespace,
                 catalogue_hash: str) -> dict:
    def capture(value: dict) -> dict:
        receipt = value["receipt"]
        return {
            "role": value["role"],
            "stage": args.stage,
            "attribution_mode": value["attribution_mode"],
            "warmup_steps": value["warmup_steps"],
            "active_steps": value["active_steps"],
            "receipt": {
                "corpus_id": receipt.corpus_id,
                "workload_id": receipt.workload_id,
                "profile_path": receipt.profile_path,
                "profile_sha256": receipt.profile_sha256,
                "source_commit": receipt.source_commit,
            },
        }

    if args.workload_kind in ("q4k-op", "quant-op"):
        quant_type = "q4_K" if args.workload_kind == "q4k-op" else args.quant_type
        architecture_blocks = [{
            "block_id": f"{quant_type.casefold()}-op-requantized-matvec",
            "kernel_families": [
                "__amd_rocclr_fillBufferAligned", "quantize_q8_1", "mul_mat_vec_q"],
            "source_paths": [
                "tests/test-backend-ops.cpp",
                "ggml/src/ggml-cuda/quantize.cu",
                "ggml/src/ggml-cuda/mmvq.cu",
            ],
        }]
    else:
        architecture_blocks = [{
            "block_id": "qwen2-prefill-quantized-matmul",
            "kernel_families": ["quantize_q8_1", "mul_mat_q"],
            "source_paths": ["src/models/qwen2.cpp"],
        }]
    return {
        "comparison_id": args.campaign_id,
        "mapping": capture(mapping),
        "formal": capture(formal),
        "source_catalog_sha256": catalogue_hash,
        "cumulative_floor": profile_report.CUMULATIVE_FLOOR,
        "catalogue_scope": "kernel_and_host",
        "host_catalog_sha256": catalogue_hash,
        "patterns": [{
            "pattern_id": "q8-requant-overlap",
            "table": "overlap",
            "kernel_keywords": ["quantize_q8_1"],
            "match_mode": "all",
            "source_symbols": ["quantize_q8_1"],
            "source_paths": ["ggml/src/ggml-cuda/quantize.cu"],
            "reader_should_conclude": "inspect producer-consumer overlap",
        }, {
            "pattern_id": "q8-requant-matmul-fuse",
            "table": "fuse",
            "kernel_keywords": ["quantize_q8_1", "mul_mat"],
            "match_mode": "all",
            "source_symbols": ["quantize_q8_1", "ggml_cuda_mul_mat"],
            "source_paths": [
                "ggml/src/ggml-cuda/quantize.cu",
                "ggml/src/ggml-cuda/mmq.cu",
            ],
            "reader_should_conclude": "profile a fused or requantization-free alternative",
        }],
        "architecture_blocks": architecture_blocks,
        "profilers": [{
            "name": "rocprofv2", "state": "available",
            "gfx90a_state": "supported", "evidence": "this paired live capture",
        }, {
            "name": "rocprof_v1", "state": "unavailable",
            "gfx90a_state": "unsupported", "evidence": "SQ/TA counters read zero on host",
        }, {
            "name": "omniperf", "state": "fallback",
            "gfx90a_state": "unchecked", "evidence": "Python environment incomplete",
        }, {
            "name": "rpd", "state": "unchecked",
            "gfx90a_state": "unchecked", "evidence": "only MI300X/gfx942 source evidence",
        }],
    }


def serializable_capture(value: dict) -> dict:
    rendered = dict(value)
    receipt = rendered.pop("receipt")
    rendered["receipt"] = {
        "corpus_id": receipt.corpus_id,
        "workload_id": receipt.workload_id,
        "profile_path": receipt.profile_path,
        "profile_sha256": receipt.profile_sha256,
        "source_commit": receipt.source_commit,
    }
    return rendered


def artifact_inventory(output_dir: Path) -> list[dict[str, object]]:
    """Hash every durable capture artifact already written below ``output_dir``."""
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "receipt.json":
            rows.append({
                "path": str(path.relative_to(output_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    return rows


def run(args: argparse.Namespace) -> dict:
    binary = Path(args.binary).resolve()
    model = Path(args.model).resolve() if args.model else None
    source_root = Path(args.source_root).resolve()
    catalogue = REPO_ROOT / "scripts/kernel_rnd/autokernel/prior_art_catalogue.json"
    required_paths = [(binary, "binary"), (catalogue, "catalogue")]
    if model is not None:
        required_paths.append((model, "model"))
    for path, label in required_paths:
        if not path.is_file():
            raise RuntimeError(f"C4 {label} does not exist: {path}")
    if args.source_commit is None:
        args.source_commit = source_commit(source_root)
    if args.stage is None:
        args.stage = (
            "decode" if args.workload_kind in ("q4k-op", "quant-op")
            else "prefill")
    output_dir = Path(storage.assert_not_scratch(
        args.output_dir, what="C4 paired profile evidence directory"))
    output_dir.mkdir(parents=True, exist_ok=False)
    claim = device_claim.acquire_device_claim(
        "mi210_0", purpose=f"AutoKernel C4 paired {args.stage} profile",
        campaign_id=args.campaign_id,
        journal=device_claim.ClaimJournal(args.claim_journal),
        holder_label="capture_autokernel_c4_profile.py",
        timeout_s=args.claim_timeout_s,
        max_hold_s=4 * args.arm_timeout_s + 300.0)
    opened = claim.receipt().to_dict()
    sampler = None
    sampling_receipt = None
    captured_error: BaseException | None = None
    mapping = None
    formal = None
    started_at = utc_now()
    started_mono = time.monotonic()
    try:
        sampler = device_sampler.RocmSmiSampler(device_index=0, interval_s=0.250).start()
        mapping = capture_role(
            role="mapping", binary=binary, model=model, args=args,
            output_dir=output_dir)
        formal = capture_role(
            role="formal", binary=binary, model=model, args=args,
            output_dir=output_dir)
    except BaseException as exc:
        captured_error = exc
    finally:
        if sampler is not None:
            sampling_receipt = sampler.stop()
        released = claim.release().to_dict()
    if sampling_receipt is None:
        raise RuntimeError("C4 capture completed without device sampling")
    if captured_error is not None:
        receipt_path = output_dir / "receipt.json"
        failure = {
            "schema": SCHEMA,
            "status": "failed",
            "campaign_id": args.campaign_id,
            "started_at": started_at,
            "ended_at": utc_now(),
            "duration_s": time.monotonic() - started_mono,
            "source_root": str(source_root),
            "source_commit": args.source_commit,
            "binary": str(binary),
            "binary_sha256": sha256_file(binary),
            "model": str(model) if model is not None else None,
            "model_sha256": sha256_file(model) if model is not None else None,
            "workload_kind": args.workload_kind,
            "stage": args.stage,
            "quant_type": args.quant_type,
            "error": {
                "type": type(captured_error).__name__,
                "message": str(captured_error),
            },
            "artifacts": artifact_inventory(output_dir),
            "device_claim_open": opened,
            "device_claim_released": released,
            "device_sampling": sampling_receipt.to_dict(),
        }
        write_json_atomic(receipt_path, failure)
        raise RuntimeError(
            f"C4 capture failed; durable receipt: {receipt_path}") from captured_error
    assert mapping is not None and formal is not None
    catalogue_hash = sha256_file(catalogue)
    manifest_payload = manifest_for(
        mapping, formal, args=args, catalogue_hash=catalogue_hash)
    manifest = profile_report.ReportManifest.from_dict(manifest_payload)
    report = profile_report.run_profile_report(
        mapping["profile_path"], formal["profile_path"], manifest).as_dict()
    write_json_atomic(output_dir / "manifest.json", manifest_payload)
    write_json_atomic(output_dir / "report.json", report)
    payload = {
        "schema": SCHEMA,
        "campaign_id": args.campaign_id,
        "started_at": started_at,
        "ended_at": utc_now(),
        "duration_s": time.monotonic() - started_mono,
        "source_root": str(source_root),
        "source_commit": args.source_commit,
        "measurement_hardening": False,
        "measurement_hardening_reason": (
            "C4 profiles the production dispatch path; hardened A/B duplicates the graph"),
        "binary": str(binary),
        "binary_sha256": sha256_file(binary),
        "model": str(model) if model is not None else None,
        "model_sha256": sha256_file(model) if model is not None else None,
        "workload_kind": args.workload_kind,
        "mapping": serializable_capture(mapping),
        "formal": serializable_capture(formal),
        "report": str(output_dir / "report.json"),
        "report_sha256": sha256_file(output_dir / "report.json"),
        "device_claim_open": opened,
        "device_claim_released": released,
        "device_sampling": sampling_receipt.to_dict(),
    }
    write_json_atomic(output_dir / "receipt.json", payload)
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--binary", required=True)
    result.add_argument("--model")
    result.add_argument(
        "--workload-kind", choices=("llama-prefill", "q4k-op", "quant-op"),
        default="llama-prefill")
    result.add_argument("--stage", choices=profile_report.STAGES)
    result.add_argument("--quant-type", default="q4_K")
    result.add_argument("--op-m", type=int, default=16)
    result.add_argument("--op-n", type=int, default=1)
    result.add_argument("--op-k", type=int, default=256)
    result.add_argument("--source-root", required=True)
    result.add_argument("--source-commit")
    result.add_argument("--output-dir", required=True)
    result.add_argument("--campaign-id", default="autokernel-c4-prefill-20260811")
    result.add_argument("--workload-id", default="qwen25-coder-0.5b-q4k-p512")
    result.add_argument("--claim-journal", default="/mnt/raid0/llm/ak-claims/device.jsonl")
    result.add_argument("--claim-timeout-s", type=float, default=0.0)
    result.add_argument("--arm-timeout-s", type=float, default=300.0)
    result.add_argument("--prompt-tokens", type=int, default=512)
    result.add_argument("--suite-seed", type=int, default=4711)
    return result


def main() -> int:
    args = parser().parse_args()
    payload = run(args)
    print(json.dumps({
        "receipt": str(Path(args.output_dir) / "receipt.json"),
        "report": payload["report"],
        "mapping_dispatches": payload["mapping"]["dispatches"],
        "formal_dispatches": payload["formal"]["dispatches"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
