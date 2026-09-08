#!/usr/bin/env python3
"""Fast non-promotable MI210 discovery over one factor and workload frame."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from fractions import Fraction
import hashlib
import json
import math
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Callable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.kernel_rnd.autokernel import schemas, storage
from scripts.kernel_rnd.autokernel.execution import (
    cpu_region_claim, device_sampler, inference_window)
from scripts.kernel_rnd.autokernel.resource import device_claim
from scripts.benchmark import autokernel_gpu_discovery_beliefs as gpu_beliefs
from scripts.benchmark import autokernel_progression
from scripts.kernel_rnd.autokernel.controller import split_runtime_verifier
from scripts.kernel_rnd.autokernel.controller import gpu_load_admission


SCHEMA_BANK = "epyc.autokernel.gpu_screening_baseline.v2"
SCHEMA_RESULT = "epyc.autokernel.gpu_candidate_only_screen.v2"
SCHEMA_LIVE_GOVERNANCE = "epyc.autokernel.gpu_discovery_live_governance.v1"
SCHEMA_PROCESS_RECEIPT = "epyc.autokernel.gpu_discovery_process_receipt.v1"
SCHEMA_OUTPUT_REFUSAL = "epyc.autokernel.gpu_discovery_output_refusal.v1"
SCHEMA_CORRECTNESS_DIVERGENCE = (
    "epyc.autokernel.gpu_candidate_correctness_divergence.v1")
SCHEMA_TIMED_OUTPUT_INFRASTRUCTURE = (
    "epyc.autokernel.timed_output_infrastructure_ambiguity.v1")
SOURCE_COMMIT = "0db32c06e3e550065b78311a6031ef3dd2c4f27c"
READY_CONTINUE_INSTRUMENT_COMMIT = "5bbcc5498e4732162356953b7be96a53073a6706"
READY_CONTINUE_CONTRACT_SHA256 = "1411f5e81c1b0b3db6952523922c672d88a78aaff5945865c9ccc2b4fc5fd99f"
CPU_LIST = "184-191"
DEVICE_ID = "mi210_0"
DEFAULT_HOST_BANDWIDTH_BYTES_S = 400 * 1000 * 1000 * 1000
DEFAULT_HOST_TRANSFER_FRACTION = 0.01
VRAM_USED = Path("/sys/class/drm/card2/device/mem_info_vram_used")
KFD_PROCS = Path("/sys/class/kfd/kfd/proc")
MODEL_CALL_WINDOW = inference_window.InferenceCallWindow(timeout_s=600.0)

_HEX64_RE = re.compile(r"^[0-9a-f]{16}$")
_ADDRESS_RE = re.compile(r"^0x[0-9a-f]+$")
_DECIMAL6_RE = re.compile(r"^[0-9]+\.[0-9]{6}$")
_MAX_PROCESS_OUTPUT_BYTES = 8 * 1024 * 1024


class MeasurementOutputRefusal(RuntimeError):
    """A completed process whose bounded output fails the measurement contract."""

    stage = "measurement_output"
    disposition = "measurement_output_refused"
    scientific_budget_spent = False

    def __init__(self, message: str, *, receipt_path: str,
                 receipt_sha256: str) -> None:
        super().__init__(message)
        if (not receipt_path or not isinstance(receipt_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", receipt_sha256) is None):
            raise RuntimeError("measurement output refusal lacks a sealed receipt")
        self.receipt_path = receipt_path
        self.receipt_sha256 = receipt_sha256


class CandidateCorrectnessDivergence(RuntimeError):
    """A fully validated same-input candidate whose outputs differ from anchor."""

    stage = "correctness"
    disposition = "correctness_falsified"
    scientific_budget_spent = True

    def __init__(self, message: str, *, receipt_path: str,
                 receipt_sha256: str, result_sha256: str,
                 operation_key: str) -> None:
        super().__init__(message)
        if (not receipt_path or not isinstance(receipt_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", receipt_sha256) is None):
            raise RuntimeError(
                "candidate correctness divergence lacks a sealed receipt")
        self.receipt_path = receipt_path
        self.receipt_sha256 = receipt_sha256
        if (not isinstance(result_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", result_sha256) is None):
            raise RuntimeError(
                "candidate correctness divergence lacks its native result hash")
        self.result_sha256 = result_sha256
        if (not isinstance(operation_key, str)
                or re.fullmatch(r"[0-9a-f]{64}", operation_key) is None):
            raise RuntimeError(
                "candidate correctness divergence lacks its operation identity")
        self.operation_key = operation_key


class TimedOutputInfrastructureAmbiguity(RuntimeError):
    """Per-arm integrity failure requiring a fresh operation epoch."""

    stage = "measurement_integrity"
    disposition = "infrastructure_ambiguity"
    scientific_budget_spent = False

    def __init__(self, message: str, *, receipt_path: str,
                 receipt_sha256: str, operation_key: str) -> None:
        super().__init__(message)
        if (not receipt_path or not isinstance(receipt_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", receipt_sha256) is None
                or not isinstance(operation_key, str)
                or re.fullmatch(r"[0-9a-f]{64}", operation_key) is None):
            raise RuntimeError(
                "timed-output infrastructure ambiguity lacks sealed authority")
        self.receipt_path = receipt_path
        self.receipt_sha256 = receipt_sha256
        self.operation_key = operation_key


class _CrossArmOutputDivergence(RuntimeError):
    """Internal marker raised only after both arm receipts validate completely."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    if len(payload) > _MAX_PROCESS_OUTPUT_BYTES:
        raise RuntimeError("GPU discovery process output exceeds the sealed bound")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


class _NativeOutputError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _reject_json_constant(value: str) -> Any:
    raise _NativeOutputError(
        "nonfinite_json_number", f"GPU discovery output contains {value}")


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _NativeOutputError(
                "duplicate_json_key", f"GPU discovery output repeats JSON key {key}")
        result[key] = value
    return result


def _plain_json(value: Any) -> Any:
    if isinstance(value, Decimal):
        converted = float(value)
        if not value.is_finite() or not math.isfinite(converted):
            raise _NativeOutputError(
                "nonfinite_json_number",
                "GPU discovery output contains an out-of-range JSON number")
        return converted
    if isinstance(value, list):
        return [_plain_json(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain_json(item) for key, item in value.items()}
    return value


def _fraction_decimal(value: Decimal) -> Fraction:
    if not value.is_finite():
        raise _NativeOutputError(
            "nonfinite_native_metric", "GPU discovery native metric is non-finite")
    return Fraction(value)


def _defaultfloat_quantum(value: Fraction) -> Decimal:
    """Return the C++ defaultfloat, precision-6 decimal quantum for value."""
    with localcontext() as context:
        context.prec = 80
        rendered = Decimal(value.numerator) / Decimal(value.denominator)
        if not rendered.is_finite() or rendered <= 0:
            raise _NativeOutputError(
                "invalid_native_metric", "GPU discovery native metric is not positive")
        return Decimal(1).scaleb(rendered.adjusted() - 5)


def _parse_native_measurement(
        stdout: bytes, *, repetitions: int,
        tokens_per_repetition: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Parse one llama-bench JSONL row and prove its decimal provenance.

    ``avg_ts`` is printed with ``std::to_string`` (six places), while
    ``samples_ts`` uses the stream default (six significant digits).  Neither
    rounded carrier is an exact arithmetic authority.  The integer
    ``samples_ns`` vector is, so all rewarded samples are rederived from it and
    the two decimal projections are checked only against their declared
    rounding intervals.
    """
    if (not isinstance(stdout, bytes) or not stdout
            or len(stdout) > _MAX_PROCESS_OUTPUT_BYTES):
        raise _NativeOutputError(
            "output_size", "GPU discovery stdout is empty or exceeds the sealed bound")
    if not stdout.endswith(b"\n") or stdout.count(b"\n") != 1 or b"\r" in stdout:
        raise _NativeOutputError(
            "jsonl_framing", "GPU discovery stdout is not exactly one newline-terminated JSONL row")
    try:
        text = stdout[:-1].decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise _NativeOutputError(
            "stdout_utf8", "GPU discovery stdout is not strict UTF-8") from exc
    try:
        parsed = json.loads(
            text, parse_float=Decimal, parse_int=int,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_object_without_duplicates)
    except _NativeOutputError:
        raise
    except (json.JSONDecodeError, InvalidOperation, ValueError) as exc:
        raise _NativeOutputError(
            "json_parse", "GPU discovery stdout is not one complete JSON object") from exc
    if not isinstance(parsed, dict):
        raise _NativeOutputError(
            "json_shape", "GPU discovery stdout row is not an object")
    if (isinstance(tokens_per_repetition, bool)
            or not isinstance(tokens_per_repetition, int)
            or tokens_per_repetition <= 0):
        raise _NativeOutputError(
            "token_count", "GPU discovery token count is invalid")
    samples_ns = parsed.get("samples_ns")
    avg_ns = parsed.get("avg_ns")
    samples_ts = parsed.get("samples_ts")
    avg_ts = parsed.get("avg_ts")
    if (not isinstance(samples_ns, list) or len(samples_ns) != repetitions
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value <= 0 for value in samples_ns)):
        raise _NativeOutputError(
            "samples_ns", f"GPU discovery invocation requires exactly {repetitions} positive integer samples_ns")
    if (isinstance(avg_ns, bool) or not isinstance(avg_ns, int)
            or avg_ns != sum(samples_ns) // repetitions):
        raise _NativeOutputError(
            "avg_ns", "GPU discovery avg_ns does not rederive from samples_ns")
    if (not isinstance(samples_ts, list) or len(samples_ts) != repetitions
            or any(isinstance(value, bool)
                   or not isinstance(value, (int, Decimal))
                   or (isinstance(value, Decimal) and not value.is_finite())
                   or value <= 0 for value in samples_ts)):
        raise _NativeOutputError(
            "samples_ts", f"GPU discovery invocation requires exactly {repetitions} finite samples_ts")
    if (not isinstance(avg_ts, Decimal) or not avg_ts.is_finite() or avg_ts <= 0
            or _DECIMAL6_RE.fullmatch(str(avg_ts)) is None):
        raise _NativeOutputError(
            "avg_ts_format", "GPU discovery avg_ts lacks its exact six-place decimal provenance")

    exact_samples = [Fraction(1_000_000_000 * tokens_per_repetition, elapsed)
                     for elapsed in samples_ns]
    exact_average = sum(exact_samples, Fraction(0, 1)) / repetitions
    if abs(_fraction_decimal(avg_ts) - exact_average) > Fraction(1, 2_000_000):
        raise _NativeOutputError(
            "avg_ts_rounding", "GPU discovery avg_ts is outside its six-place rounding interval")
    reported_decimals = [value if isinstance(value, Decimal) else Decimal(value)
                         for value in samples_ts]
    for reported, exact in zip(reported_decimals, exact_samples):
        quantum = _defaultfloat_quantum(exact)
        if abs(_fraction_decimal(reported) - exact) > _fraction_decimal(quantum) / 2:
            raise _NativeOutputError(
                "samples_ts_rounding",
                "GPU discovery samples_ts is outside its precision-6 rounding interval")
    rederived_samples = [float(value) for value in exact_samples]
    rederived_average = float(exact_average)
    if (not math.isfinite(rederived_average) or rederived_average <= 0
            or any(not math.isfinite(value) or value <= 0
                   for value in rederived_samples)):
        raise _NativeOutputError(
            "native_metric_range",
            "GPU discovery integer timings rederive outside the finite metric range")

    row = _plain_json(parsed)
    diagnostic_body = {
        "schema": "epyc.autokernel.native_llama_bench_diagnostic.v2",
        "integer_timing_authority": "samples_ns",
        "tokens_per_repetition": tokens_per_repetition,
        "avg_ns": avg_ns,
        "samples_ns": list(samples_ns),
        "reported_avg_ts_decimal": str(avg_ts),
        "reported_samples_ts_decimal": [str(value) for value in reported_decimals],
        "rederived_samples_ts": rederived_samples,
        "rederived_avg_ts": rederived_average,
        "rounding_contract": {
            "avg_ts": "std_to_string_fixed_6_places",
            "samples_ts": "cpp_defaultfloat_precision_6_significant_digits",
        },
    }
    return row, {**diagnostic_body,
                 "receipt_sha256": schemas.content_hash(diagnostic_body)}


def _process_capture_identity(
        *, argv: tuple[str, ...], env: Mapping[str, str], binary: Path,
        loader_dir: Path, model: Path, seed: int, repetitions: int,
        runtime_graphs: str, runtime_arm: str | None,
        process_context: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        "argv": list(argv),
        "environment": dict(env),
        "reward_binary": str(binary),
        "reward_binary_sha256": sha256_file(binary),
        "hip_library": str(loader_dir / "libggml-hip.so"),
        "hip_library_sha256": sha256_file(loader_dir / "libggml-hip.so"),
        "model": str(model.resolve()),
        "model_sha256": sha256_file(model),
        "seed": seed,
        "repetitions": repetitions,
        "runtime_graphs": runtime_graphs,
        "runtime_arm": runtime_arm,
        "process_context": (None if process_context is None
                            else dict(process_context)),
    }


def _capture_file(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(f"GPU discovery {label} capture is unsafe") from exc
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
                or before.st_nlink != 1 or stat.S_IMODE(before.st_mode) & 0o077
                or before.st_size > _MAX_PROCESS_OUTPUT_BYTES):
            raise RuntimeError(f"GPU discovery {label} capture is unsafe")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError(f"GPU discovery {label} capture was truncated")
            chunks.append(chunk); remaining -= len(chunk)
        after = os.fstat(fd)
        if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
            raise RuntimeError(f"GPU discovery {label} capture changed while reading")
        payload = b"".join(chunks)
        return payload, {
            "path": str(path.resolve()), "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "device": before.st_dev, "inode": before.st_ino,
        }
    finally:
        os.close(fd)


def _seal_process_capture(
        root: Path, *, identity: Mapping[str, Any], returncode: int,
        stdout: bytes, stderr: bytes, residency: list[dict[str, Any]],
        runtime_maps_identity: Mapping[str, Any] | None,
        readiness_witness: Mapping[str, Any] | None,
        elapsed_s: float, teardown: Mapping[str, Any],
        resource_context: Mapping[str, Any] | None) -> dict[str, Any]:
    if root.exists() or root.is_symlink():
        raise RuntimeError("GPU discovery process receipt already exists")
    root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = root.with_name(f".{root.name}.tmp-{os.getpid()}")
    staging.mkdir(mode=0o700)
    try:
        stdout_stored = stdout[:_MAX_PROCESS_OUTPUT_BYTES]
        stderr_stored = stderr[:_MAX_PROCESS_OUTPUT_BYTES]
        _atomic_bytes(staging / "stdout.bin", stdout_stored)
        _atomic_bytes(staging / "stderr.bin", stderr_stored)
        stdout_binding = {
            "path": "stdout.bin", "observed_size": len(stdout),
            "observed_sha256": hashlib.sha256(stdout).hexdigest(),
            "stored_size": len(stdout_stored),
            "stored_sha256": hashlib.sha256(stdout_stored).hexdigest(),
            "truncated": len(stdout) > len(stdout_stored)}
        stderr_binding = {
            "path": "stderr.bin", "observed_size": len(stderr),
            "observed_sha256": hashlib.sha256(stderr).hexdigest(),
            "stored_size": len(stderr_stored),
            "stored_sha256": hashlib.sha256(stderr_stored).hexdigest(),
            "truncated": len(stderr) > len(stderr_stored)}
        body = {
            "schema": SCHEMA_PROCESS_RECEIPT,
            "status": "process_complete",
            "identity": dict(identity),
            "returncode": returncode,
            "stdout": stdout_binding,
            "stderr": stderr_binding,
            "residency": residency,
            "runtime_maps_identity": (None if runtime_maps_identity is None
                                      else dict(runtime_maps_identity)),
            "load_readiness_witness": (None if readiness_witness is None
                                       else dict(readiness_witness)),
            "supervisor_elapsed_s": elapsed_s,
            "teardown": dict(teardown),
            "resource_context": (None if resource_context is None
                                 else dict(resource_context)),
            "output_bound_bytes": _MAX_PROCESS_OUTPUT_BYTES,
        }
        atomic_json(staging / "receipt.json", {
            **body, "receipt_sha256": schemas.content_hash(body)})
        os.chmod(staging / "receipt.json", 0o600)
        os.replace(staging, root)
    except BaseException:
        # A pre-rename staging directory is never reusable evidence.  Keep it
        # visible for ambiguity/forensics rather than deleting completed bytes.
        raise
    return _load_process_capture(root, identity=identity)


def _load_process_capture(root: Path, *, identity: Mapping[str, Any]) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("GPU discovery process receipt root is unsafe")
    root_stat = root.stat()
    if (root_stat.st_uid != os.getuid() or stat.S_IMODE(root_stat.st_mode) != 0o700
            or root_stat.st_nlink != 2):
        raise RuntimeError("GPU discovery process receipt root is unsafe")
    if {entry.name for entry in root.iterdir()} != {
            "stdout.bin", "stderr.bin", "receipt.json"}:
        raise RuntimeError("GPU discovery process receipt closure changed")
    stdout, stdout_binding = _capture_file(root / "stdout.bin", "stdout")
    stderr, stderr_binding = _capture_file(root / "stderr.bin", "stderr")
    receipt_path = root / "receipt.json"
    receipt_bytes, receipt_binding = _capture_file(receipt_path, "receipt")
    try:
        receipt = json.loads(receipt_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("GPU discovery process receipt is malformed") from exc
    unsigned = {key: value for key, value in receipt.items()
                if key != "receipt_sha256"}
    if (receipt.get("schema") != SCHEMA_PROCESS_RECEIPT
            or receipt.get("status") != "process_complete"
            or receipt.get("identity") != dict(identity)
            or receipt.get("receipt_sha256") != schemas.content_hash(unsigned)
            or isinstance(receipt.get("returncode"), bool)
            or not isinstance(receipt.get("returncode"), int)
            or not isinstance(receipt.get("residency"), list)
            or (receipt["returncode"] == 0 and not receipt["residency"])
            or any(not isinstance(sample, Mapping)
                   for sample in receipt["residency"])
            or isinstance(receipt.get("supervisor_elapsed_s"), bool)
            or not isinstance(receipt.get("supervisor_elapsed_s"), (int, float))
            or not math.isfinite(float(receipt["supervisor_elapsed_s"]))
            or receipt["supervisor_elapsed_s"] < 0
            or not isinstance(receipt.get("teardown"), Mapping)
            or receipt.get("output_bound_bytes") != _MAX_PROCESS_OUTPUT_BYTES):
        raise RuntimeError("GPU discovery process receipt identity changed")
    for label, binding in (("stdout", stdout_binding),
                           ("stderr", stderr_binding)):
        declared = receipt.get(label)
        if (not isinstance(declared, Mapping)
                or set(declared) != {"path", "observed_size", "observed_sha256",
                                     "stored_size", "stored_sha256", "truncated"}
                or declared.get("path") != f"{label}.bin"
                or declared.get("stored_size") != binding["size"]
                or declared.get("stored_sha256") != binding["sha256"]
                or not isinstance(declared.get("observed_size"), int)
                or declared["observed_size"] < declared["stored_size"]
                or re.fullmatch(r"[0-9a-f]{64}", str(
                    declared.get("observed_sha256"))) is None
                or declared.get("truncated") is not (
                    declared["observed_size"] > declared["stored_size"])
                or (declared["truncated"] is False
                    and declared["observed_sha256"] != declared["stored_sha256"])):
            raise RuntimeError(
                f"GPU discovery {label} process binding changed")
    resource_context = receipt.get("resource_context")
    process_context = identity.get("process_context")
    if isinstance(process_context, Mapping) and "campaign_id" in process_context:
        if (not isinstance(resource_context, Mapping)
                or resource_context.get("device_claim_mode") not in {
                    "borrowed_outer_reservation", "direct_device_claim"}
                or not isinstance(resource_context.get("device_claim_open"), Mapping)):
            raise RuntimeError("GPU discovery process receipt lacks its claim window")
        try:
            opened = device_claim.ClaimReceipt.from_dict(
                resource_context["device_claim_open"])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "GPU discovery process receipt claim window is malformed") from exc
        if (opened.released_at is not None
                or opened.campaign_id != process_context.get("campaign_id")):
            raise RuntimeError("GPU discovery process receipt claim window changed")
    return {
        "receipt": receipt,
        "receipt_path": str(receipt_path.resolve()),
        "receipt_file_sha256": receipt_binding["sha256"],
        "stdout": stdout, "stderr": stderr,
    }


def _seal_output_refusal(
        root: Path, *, capture: Mapping[str, Any], code: str,
        message: str) -> MeasurementOutputRefusal:
    path = root.with_name(f"{root.name}-refusal.json")
    diagnostic = _output_refusal_diagnostic(capture)
    body = {
        "schema": SCHEMA_OUTPUT_REFUSAL,
        "status": "measurement_output_refused",
        "scientific_budget_spent": False,
        "process_receipt_path": capture["receipt_path"],
        "process_receipt_file_sha256": capture["receipt_file_sha256"],
        "reason_code": code,
        "reason_sha256": hashlib.sha256(message.encode()).hexdigest(),
        "diagnostic": diagnostic,
    }
    value = {**body, "receipt_sha256": schemas.content_hash(body)}
    if path.exists() or path.is_symlink():
        refusal_bytes, refusal_binding = _capture_file(path, "output refusal")
        if json.loads(refusal_bytes) != value:
            raise RuntimeError("GPU discovery output refusal changed on reopen")
    else:
        atomic_json(path, value)
        os.chmod(path, 0o600)
        refusal_bytes, refusal_binding = _capture_file(path, "output refusal")
    return MeasurementOutputRefusal(
        message, receipt_path=str(path.resolve()),
        receipt_sha256=refusal_binding["sha256"])


def _seal_timed_output_infrastructure_ambiguity(
        root: Path, *, capture: Mapping[str, Any], code: str,
        message: str) -> TimedOutputInfrastructureAmbiguity:
    receipt = capture.get("receipt")
    identity = receipt.get("identity") if isinstance(receipt, Mapping) else None
    context = (identity.get("process_context")
               if isinstance(identity, Mapping) else None)
    operation_key = (context.get("operation_key")
                     if isinstance(context, Mapping) else None)
    if (not isinstance(operation_key, str)
            or re.fullmatch(r"[0-9a-f]{64}", operation_key) is None):
        raise RuntimeError(
            "timed-output infrastructure ambiguity lacks operation identity")
    path = root.with_name(f"{root.name}-infrastructure-ambiguity.json")
    body = {
        "schema": SCHEMA_TIMED_OUTPUT_INFRASTRUCTURE,
        "status": "infrastructure_ambiguity",
        "stage": "measurement_integrity",
        "scientific_budget_spent": False,
        "candidate_disposition": False,
        "requires_fresh_operation": True,
        "operation_key": operation_key,
        "process_receipt_path": capture["receipt_path"],
        "process_receipt_file_sha256": capture["receipt_file_sha256"],
        "reason_code": code,
        "reason_sha256": hashlib.sha256(message.encode()).hexdigest(),
        "diagnostic": _output_refusal_diagnostic(capture),
    }
    value = {**body, "receipt_sha256": schemas.content_hash(body)}
    if path.exists() or path.is_symlink():
        payload, binding = _capture_file(path, "timed-output infrastructure ambiguity")
        if json.loads(payload) != value:
            raise RuntimeError(
                "timed-output infrastructure ambiguity changed on reopen")
    else:
        atomic_json(path, value)
        os.chmod(path, 0o600)
        payload, binding = _capture_file(path, "timed-output infrastructure ambiguity")
        if json.loads(payload) != value:
            raise RuntimeError(
                "timed-output infrastructure ambiguity changed after sealing")
    return TimedOutputInfrastructureAmbiguity(
        message, receipt_path=str(path.resolve()),
        receipt_sha256=binding["sha256"], operation_key=operation_key)


def _directory_namespace_identity(path: Path, label: str) -> dict[str, Any]:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(f"{label} is unavailable") from exc
    try:
        info = os.fstat(fd)
        pathname = path.lstat()
        parent = path.parent.stat()
        if (path.is_symlink() or not stat.S_ISDIR(info.st_mode)
                or (info.st_dev, info.st_ino) !=
                   (pathname.st_dev, pathname.st_ino)
                or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) & 0o022
                or info.st_nlink < 2):
            raise RuntimeError(
                f"{label} is not a trusted operation directory")
        return {
            "path": str(path), "type": "directory",
            "device": info.st_dev, "inode": info.st_ino,
            "uid": info.st_uid, "mode": stat.S_IMODE(info.st_mode),
            "nlink": info.st_nlink,
            "parent_device": parent.st_dev, "parent_inode": parent.st_ino,
        }
    finally:
        os.close(fd)


def _operation_namespace(
        *, operations_root: Path, output_root: Path, operation_key: str,
        repetition: int, runtime_graphs: str) -> dict[str, Any]:
    if (not isinstance(operation_key, str)
            or re.fullmatch(r"[0-9a-f]{64}", operation_key) is None
            or isinstance(repetition, bool) or repetition not in {1, 2}
            or runtime_graphs not in {"off", "on"}
            or not operations_root.is_absolute()
            or not output_root.is_absolute()
            or operations_root.resolve() != operations_root
            or output_root.resolve(strict=False) != output_root):
        raise RuntimeError("runner operation namespace is malformed or aliased")
    stage = ("measurement-graphs-off" if runtime_graphs == "off"
             else "target-runtime-graphs-on")
    operation_dir = operations_root / operation_key
    runner_dir = operation_dir / "runner"
    repetition_dir = runner_dir / f"s{repetition}"
    expected_output = repetition_dir / stage
    if output_root != expected_output:
        raise RuntimeError(
            "runner output does not belong to its exact operation namespace")
    directories = [operations_root, operation_dir, runner_dir, repetition_dir]
    identities = [
        _directory_namespace_identity(path, label)
        for path, label in zip(
            directories,
            ("operations root", "operation root", "runner root",
             "runner repetition root"), strict=True)]
    output_identity = (
        _directory_namespace_identity(output_root, "runner stage root")
        if output_root.exists() or output_root.is_symlink() else None)
    return {
        "schema": "epyc.autokernel.gpu_runner_operation_namespace.v1",
        "operation_key": operation_key,
        "repetition": repetition,
        "runtime_graphs": runtime_graphs,
        "stage": stage,
        "output_root": str(output_root),
        "directories": identities,
        "output_identity": output_identity,
    }


def _revalidate_operation_namespace(
        namespace: Mapping[str, Any], *, output_root: Path,
        operation_key: str, runtime_graphs: str) -> None:
    if (not isinstance(namespace, Mapping)
            or set(namespace) != {
                "schema", "operation_key", "repetition", "runtime_graphs",
                "stage", "output_root", "directories", "output_identity"}
            or namespace.get("schema") !=
               "epyc.autokernel.gpu_runner_operation_namespace.v1"
            or namespace.get("operation_key") != operation_key
            or namespace.get("runtime_graphs") != runtime_graphs
            or namespace.get("output_root") != str(output_root)
            or not isinstance(namespace.get("directories"), list)
            or len(namespace["directories"]) != 4):
        raise RuntimeError("sealed runner operation namespace changed")
    directories = namespace["directories"]
    operations_root = Path(str(directories[0].get("path", "")))
    current = _operation_namespace(
        operations_root=operations_root, output_root=output_root,
        operation_key=operation_key,
        repetition=namespace.get("repetition"), runtime_graphs=runtime_graphs)
    sealed_output = namespace.get("output_identity")
    current_output = current.get("output_identity")
    if (not isinstance(sealed_output, Mapping)
            or not isinstance(current_output, Mapping)):
        raise RuntimeError("sealed runner stage leaf identity is absent")
    stable_directory_keys = {
        "path", "type", "device", "inode", "uid", "mode",
        "parent_device", "parent_inode"}
    identity_keys = stable_directory_keys | {"nlink"}

    def validate_directory(
            sealed: Mapping[str, Any], observed: Mapping[str, Any],
            label: str) -> None:
        # A directory's link count legitimately grows when a governed child
        # directory is created.  Its original count is a lower bound, while
        # the inode and every other authority field remain immutable.
        if (set(sealed) != identity_keys or set(observed) != identity_keys
                or {key: sealed.get(key) for key in stable_directory_keys}
                   != {key: observed.get(key)
                       for key in stable_directory_keys}
                or isinstance(sealed.get("nlink"), bool)
                or not isinstance(sealed.get("nlink"), int)
                or sealed["nlink"] < 2
                or isinstance(observed.get("nlink"), bool)
                or not isinstance(observed.get("nlink"), int)
                or observed["nlink"] < sealed["nlink"]):
            raise RuntimeError(f"sealed runner {label} identity changed")

    validate_directory(sealed_output, current_output, "stage leaf")
    for index, (sealed_directory, current_directory) in enumerate(zip(
            namespace["directories"], current["directories"], strict=True)):
        if (not isinstance(sealed_directory, Mapping)
                or not isinstance(current_directory, Mapping)):
            raise RuntimeError(
                "sealed runner operation namespace identity changed")
        validate_directory(
            sealed_directory, current_directory,
            f"operation directory {index}")
    stable_namespace_keys = {
        "schema", "operation_key", "repetition", "runtime_graphs", "stage",
        "output_root"}
    if ({key: namespace.get(key) for key in stable_namespace_keys}
            != {key: current.get(key) for key in stable_namespace_keys}):
        raise RuntimeError("sealed runner operation namespace identity changed")


def _seal_candidate_correctness_divergence(
        output_root: Path, *, anchor: Mapping[str, Any],
        candidate: Mapping[str, Any], runtime_graphs: str,
        campaign_id: str, operation_key: str,
        operation_namespace: Mapping[str, Any],
        anchor_identity: Mapping[str, Any],
        candidate_identity: Mapping[str, Any]
        ) -> CandidateCorrectnessDivergence:
    """Seal a scientific rejection without copying raw output hashes forward."""
    if runtime_graphs != "off":
        raise RuntimeError(
            "candidate timed-output divergence is only defined for graphs-off")
    if (not isinstance(operation_key, str)
            or re.fullmatch(r"[0-9a-f]{64}", operation_key) is None):
        raise RuntimeError(
            "candidate correctness divergence lacks operation identity")
    _revalidate_operation_namespace(
        operation_namespace, output_root=output_root,
        operation_key=operation_key, runtime_graphs=runtime_graphs)
    operation_namespace_sha256 = schemas.content_hash(operation_namespace)
    semantics = []
    processes = []
    preflight_sha256 = None
    for label, run in (("anchor", anchor), ("candidate", candidate)):
        current = run.get("timed_output_semantics")
        supervisor = run.get("supervisor")
        if (not isinstance(current, Mapping)
                or not isinstance(supervisor, Mapping)):
            raise RuntimeError(
                f"{label} correctness divergence lacks its validated process binding")
        receipt_path = supervisor.get("process_receipt_path")
        receipt_sha256 = supervisor.get("process_receipt_file_sha256")
        if (not isinstance(receipt_path, str) or not receipt_path
                or not isinstance(receipt_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", receipt_sha256) is None):
            raise RuntimeError(
                f"{label} correctness divergence process binding is malformed")
        expected_receipt = (output_root / f"process-{label}" /
                            "receipt.json").resolve()
        if Path(receipt_path) != expected_receipt:
            raise RuntimeError(
                f"{label} correctness divergence escaped its operation namespace")
        receipt_bytes, receipt_binding = _capture_file(
            Path(receipt_path), f"{label} correctness divergence process receipt")
        if receipt_binding["sha256"] != receipt_sha256:
            raise RuntimeError(
                f"{label} correctness divergence process receipt changed")
        # Require syntactically intact JSON without projecting its private raw
        # carriers into this dashboard-facing terminal.
        receipt = json.loads(receipt_bytes)
        if not isinstance(receipt, Mapping):
            raise RuntimeError(
                f"{label} correctness divergence process receipt is malformed")
        identity = receipt.get("identity")
        context = (identity.get("process_context")
                   if isinstance(identity, Mapping) else None)
        current_preflight = (context.get("preflight_sha256")
                             if isinstance(context, Mapping) else None)
        if (not isinstance(context, Mapping)
                or context.get("campaign_id") != campaign_id
                or context.get("operation_key") != operation_key
                or context.get("operation_namespace_sha256") !=
                   operation_namespace_sha256
                or context.get("arm") != label
                or context.get("runtime_graphs") != runtime_graphs
                or not isinstance(current_preflight, str)
                or re.fullmatch(r"[0-9a-f]{64}", current_preflight) is None
                or preflight_sha256 is not None
                and current_preflight != preflight_sha256):
            raise RuntimeError(
                f"{label} correctness divergence process context changed")
        preflight_sha256 = current_preflight
        semantics.append(current)
        processes.append({
            "arm": label,
            "process_receipt_path": receipt_path,
            "process_receipt_file_sha256": receipt_sha256,
        })
    anchor_semantics, candidate_semantics = semantics
    inputs = anchor_semantics["input_hashes"]
    anchor_outputs = anchor_semantics["output_hashes"]
    candidate_outputs = candidate_semantics["output_hashes"]
    mismatch_count = sum(
        left != right for left, right in zip(anchor_outputs, candidate_outputs))
    if (not isinstance(inputs, list) or not isinstance(anchor_outputs, list)
            or not isinstance(candidate_outputs, list)
            or not inputs or len(anchor_outputs) != len(inputs)
            or len(candidate_outputs) != len(inputs) or mismatch_count <= 0):
        raise RuntimeError(
            "candidate correctness divergence lacks a nonempty matched bank")
    message = (
        "candidate timed outputs differ bitwise from the sealed anchor")
    body = {
        "schema": SCHEMA_CORRECTNESS_DIVERGENCE,
        "status": "correctness_falsified",
        "classification": "screened_out",
        "stage": "correctness",
        "scientific_budget_spent": True,
        "candidate_rejected": True,
        "promotion_claim": False,
        "campaign_id": campaign_id,
        "operation_key": operation_key,
        "operation_namespace_sha256": operation_namespace_sha256,
        "preflight_sha256": preflight_sha256,
        "anchor_build_identity_sha256": schemas.content_hash(anchor_identity),
        "candidate_build_identity_sha256": schemas.content_hash(candidate_identity),
        "runtime_graphs": runtime_graphs,
        "target_runtime_executed": False,
        "reason_code": "cross_arm_timed_output_divergence",
        "reason_sha256": hashlib.sha256(message.encode()).hexdigest(),
        "repetitions": len(inputs),
        "differing_repetitions": mismatch_count,
        # Hash the vectors as one opaque bank.  The exact member hashes remain
        # solely in the mode-0600 native process captures named below.
        "matched_input_bank_sha256": schemas.content_hash(inputs),
        "anchor_output_bank_sha256": schemas.content_hash(anchor_outputs),
        "candidate_output_bank_sha256": schemas.content_hash(candidate_outputs),
        "process_receipts": processes,
    }
    value = {**body, "receipt_sha256": schemas.content_hash(body)}
    path = output_root / "correctness-divergence.json"
    if path.exists() or path.is_symlink():
        payload, binding = _capture_file(path, "correctness divergence")
        if json.loads(payload) != value:
            raise RuntimeError(
                "candidate correctness divergence changed on reopen")
    else:
        atomic_json(path, value)
        os.chmod(path, 0o600)
        payload, binding = _capture_file(path, "correctness divergence")
        if json.loads(payload) != value:
            raise RuntimeError(
                "candidate correctness divergence changed after sealing")
    return CandidateCorrectnessDivergence(
        message, receipt_path=str(path.resolve()),
        receipt_sha256=binding["sha256"],
        result_sha256=value["receipt_sha256"],
        operation_key=operation_key)


def _output_refusal_diagnostic(capture: Mapping[str, Any]) -> dict[str, Any]:
    """Project bounded, secret-free native timing facts from refused output.

    Raw stdout and stderr remain in their mode-0600 process receipt.  The
    refusal itself carries only the workload identity, timing carriers,
    independently rederived values when the integer authority permits it, and
    hashes/sizes needed to locate the exact forensic bytes.
    """
    receipt = capture.get("receipt")
    identity = receipt.get("identity") if isinstance(receipt, Mapping) else None
    context = (identity.get("process_context")
               if isinstance(identity, Mapping) else None)
    context = context if isinstance(context, Mapping) else {}
    repetitions = (identity.get("repetitions")
                   if isinstance(identity, Mapping) else None)
    tokens = context.get("tokens_per_repetition")
    native_fields: dict[str, Any] = {
        "avg_ns": None,
        "samples_ns": None,
        "avg_ts_decimal": None,
        "samples_ts_decimal": None,
    }
    rederived: dict[str, Any] = {
        "samples_ts": None,
        "avg_ts": None,
    }
    available = False
    stdout = capture.get("stdout")
    if isinstance(stdout, bytes) and stdout.endswith(b"\n") \
            and stdout.count(b"\n") == 1 and b"\r" not in stdout:
        try:
            parsed = json.loads(
                stdout[:-1].decode("utf-8", errors="strict"),
                parse_float=Decimal, parse_int=int,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_object_without_duplicates)
        except (UnicodeDecodeError, json.JSONDecodeError, InvalidOperation,
                ValueError, _NativeOutputError):
            parsed = None
        if isinstance(parsed, Mapping):
            avg_ns = parsed.get("avg_ns")
            samples_ns = parsed.get("samples_ns")
            avg_ts = parsed.get("avg_ts")
            samples_ts = parsed.get("samples_ts")
            if isinstance(avg_ns, int) and not isinstance(avg_ns, bool):
                native_fields["avg_ns"] = avg_ns
            if (isinstance(samples_ns, list)
                    and isinstance(repetitions, int)
                    and not isinstance(repetitions, bool)
                    and len(samples_ns) == repetitions
                    and all(isinstance(value, int)
                            and not isinstance(value, bool)
                            and value > 0 for value in samples_ns)):
                native_fields["samples_ns"] = list(samples_ns)
            if isinstance(avg_ts, (int, Decimal)) \
                    and not isinstance(avg_ts, bool) \
                    and (not isinstance(avg_ts, Decimal)
                         or avg_ts.is_finite()):
                native_fields["avg_ts_decimal"] = str(avg_ts)
            if (isinstance(samples_ts, list)
                    and isinstance(repetitions, int)
                    and not isinstance(repetitions, bool)
                    and len(samples_ts) == repetitions
                    and all(isinstance(value, (int, Decimal))
                            and not isinstance(value, bool)
                            and (not isinstance(value, Decimal)
                                 or value.is_finite())
                            for value in samples_ts)):
                native_fields["samples_ts_decimal"] = [
                    str(value) for value in samples_ts]
            available = any(value is not None
                            for value in native_fields.values())
            if (native_fields["samples_ns"] is not None
                    and isinstance(tokens, int) and not isinstance(tokens, bool)
                    and tokens > 0):
                exact = [Fraction(1_000_000_000 * tokens, elapsed)
                         for elapsed in native_fields["samples_ns"]]
                average = sum(exact, Fraction(0, 1)) / len(exact)
                rederived = {
                    "samples_ts": [float(value) for value in exact],
                    "avg_ts": float(average),
                }
    stdout_binding = (receipt.get("stdout")
                      if isinstance(receipt, Mapping) else None)
    stderr_binding = (receipt.get("stderr")
                      if isinstance(receipt, Mapping) else None)

    def public_binding(binding: Any) -> dict[str, Any] | None:
        if not isinstance(binding, Mapping):
            return None
        return {key: binding.get(key) for key in (
            "observed_size", "observed_sha256", "stored_size",
            "stored_sha256", "truncated")}

    return {
        "schema": "epyc.autokernel.measurement_output_refusal_diagnostic.v1",
        "diagnostic_available": available,
        "measurement_identity": {
            "campaign_id": context.get("campaign_id"),
            "arm": context.get("arm"),
            "workload": context.get("workload"),
            "metric": context.get("metric"),
            "runtime_graphs": context.get("runtime_graphs"),
            "prompt_tokens": context.get("prompt_tokens"),
            "generation_tokens": context.get("generation_tokens"),
            "tokens_per_repetition": tokens,
            "repetitions": repetitions,
            "preflight_sha256": context.get("preflight_sha256"),
        },
        "native_fields": native_fields,
        "rederived": rederived,
        "stdout": public_binding(stdout_binding),
        "stderr": public_binding(stderr_binding),
    }


def _split_exact(value: Any, *, field: str, count: int) -> list[str]:
    if not isinstance(value, str):
        raise RuntimeError(f"governed instrument {field} is not a string")
    items = value.split(",") if value else []
    if len(items) != count:
        raise RuntimeError(
            f"governed instrument {field} must contain exactly {count} entries")
    return items


def _validate_timed_output_semantics(row: Mapping[str, Any], *, repetitions: int,
                                     seed: int, tokens_per_repetition: int,
                                     serialization_env: Mapping[str, str]) -> dict[str, Any]:
    """Validate every semantic-integrity field emitted by the sealed 81bf instrument.

    HIP serialization makes the instrument's first host interval synchronous.
    The protected score is therefore the slower member of each semantically
    identical pair, so moving work from one invocation to the other cannot earn
    reward.
    """
    exact_integrity_env = {
        "AMD_SERIALIZE_KERNEL": "3", "AMD_SERIALIZE_COPY": "3",
        "GGML_CUDA_DISABLE_GRAPHS": "1",
    }
    if serialization_env != exact_integrity_env:
        raise RuntimeError("rewarded process lacks exact serialized graphs-off integrity environment")
    if (isinstance(tokens_per_repetition, bool)
            or not isinstance(tokens_per_repetition, int) or tokens_per_repetition <= 0):
        raise RuntimeError("timed-output semantics require a positive token count")
    required_true = (
        "autokernel_hardened", "autokernel_output_invariant",
        "autokernel_hybrid_ab_complete", "autokernel_thread_set_stable",
        "autokernel_escape_checks_complete",
    )
    if any(row.get(field) is not True for field in required_true):
        raise RuntimeError("governed instrument semantic-integrity flags are incomplete")
    working_set = row.get("autokernel_input_working_set_bytes")
    if isinstance(working_set, bool) or not isinstance(working_set, int) or working_set <= 0:
        raise RuntimeError("governed instrument input working set is invalid")
    if row.get("autokernel_device_sync_mode") != "hip_full_device":
        raise RuntimeError("governed instrument ranked member lacks full-device synchronization")

    input_hashes = _split_exact(
        row.get("autokernel_input_hashes"), field="input hashes", count=repetitions)
    if any(_HEX64_RE.fullmatch(value) is None for value in input_hashes) \
            or len(set(input_hashes)) != repetitions:
        raise RuntimeError("governed instrument input hashes are malformed or reused")

    output_pairs = _split_exact(
        row.get("autokernel_output_hashes"), field="output hash pairs", count=repetitions)
    output_hashes: list[str] = []
    for pair in output_pairs:
        members = pair.split("/")
        if (len(members) != 2 or any(_HEX64_RE.fullmatch(value) is None for value in members)
                or members[0] != members[1]):
            raise RuntimeError("governed instrument paired output hashes are not bitwise invariant")
        output_hashes.append(members[0])

    for field in ("autokernel_input_addresses", "autokernel_context_addresses"):
        pairs = _split_exact(row.get(field), field=field, count=repetitions)
        flattened: list[str] = []
        for pair in pairs:
            members = pair.split("/")
            if len(members) != 2 or any(_ADDRESS_RE.fullmatch(value) is None for value in members):
                raise RuntimeError(f"governed instrument {field} is malformed")
            flattened.extend(members)
        if len(set(flattened)) != 2 * repetitions:
            raise RuntimeError(f"governed instrument {field} did not rotate every pair member")

    first_samples_raw = _split_exact(
        row.get("autokernel_unsynchronized_samples_ns"),
        field="serialized first-member samples", count=repetitions)
    if any(not value.isdigit() or int(value) <= 0 for value in first_samples_raw):
        raise RuntimeError("governed instrument serialized first-member timings are malformed")
    first_samples_ns = [int(value) for value in first_samples_raw]
    second_samples_ns = row.get("samples_ns")
    if (not isinstance(second_samples_ns, list)
            or len(second_samples_ns) != repetitions
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value <= 0 for value in second_samples_ns)):
        raise RuntimeError("governed instrument ranked integer timings are malformed")
    protected_samples_ns = [max(first, second)
                            for first, second in zip(first_samples_ns, second_samples_ns)]
    protected_samples_ts = [1e9 * tokens_per_repetition / value
                            for value in protected_samples_ns]
    thread_sets = _split_exact(
        row.get("autokernel_thread_set_hashes"), field="thread-set hashes", count=repetitions)
    for item in thread_sets:
        members = item.split("/")
        if (len(members) != 4 or any(_HEX64_RE.fullmatch(value) is None for value in members)
                or len(set(members)) != 1):
            raise RuntimeError("governed instrument thread-set hashes are unstable")

    body = {
        "schema": "epyc.autokernel.timed_output_semantics.v1",
        "instrument_commit": READY_CONTINUE_INSTRUMENT_COMMIT,
        "seed": seed,
        "repetitions": repetitions,
        "tokens_per_repetition": tokens_per_repetition,
        "input_hashes": input_hashes,
        "output_hashes": output_hashes,
        "within_pair_bitwise_equal": True,
        "ranked_member_device_sync": "hip_full_device",
        "serialization_env": dict(serialization_env),
        "first_samples_ns": first_samples_ns,
        "second_samples_ns": second_samples_ns,
        "protected_samples_ns": protected_samples_ns,
        "protected_samples_ts": protected_samples_ts,
        "anti_shift_witness": "hip_serialized_pair_max",
        "reward_admissible": True,
    }
    return {**body, "receipt_sha256": schemas.content_hash(body)}


def _validate_cross_arm_timed_outputs(anchor: Mapping[str, Any],
                                      candidate: Mapping[str, Any]) -> dict[str, Any]:
    anchor_semantics = anchor.get("timed_output_semantics")
    candidate_semantics = candidate.get("timed_output_semantics")
    if not isinstance(anchor_semantics, Mapping) or not isinstance(candidate_semantics, Mapping):
        raise RuntimeError("matched arms lack sealed timed-output semantic receipts")
    exact_env = {
        "AMD_SERIALIZE_KERNEL": "3", "AMD_SERIALIZE_COPY": "3",
        "GGML_CUDA_DISABLE_GRAPHS": "1",
    }
    for label, semantics in (("anchor", anchor_semantics),
                             ("candidate", candidate_semantics)):
        receipt_sha256 = semantics.get("receipt_sha256")
        unsigned = {key: value for key, value in semantics.items()
                    if key != "receipt_sha256"}
        if (semantics.get("schema") != "epyc.autokernel.timed_output_semantics.v1"
                or semantics.get("instrument_commit") != READY_CONTINUE_INSTRUMENT_COMMIT
                or semantics.get("serialization_env") != exact_env
                or semantics.get("anti_shift_witness") != "hip_serialized_pair_max"
                or semantics.get("reward_admissible") is not True
                or not isinstance(semantics.get("repetitions"), int)
                or semantics.get("repetitions", 0) <= 0
                or receipt_sha256 != schemas.content_hash(unsigned)):
            raise RuntimeError(f"{label} timed-output semantic receipt is invalid")
        repetitions = semantics["repetitions"]
        tokens = semantics.get("tokens_per_repetition")
        first = semantics.get("first_samples_ns")
        second = semantics.get("second_samples_ns")
        protected = semantics.get("protected_samples_ns")
        protected_ts = semantics.get("protected_samples_ts")
        inputs = semantics.get("input_hashes")
        outputs = semantics.get("output_hashes")
        if (isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0
                or not all(isinstance(values, list) and len(values) == repetitions
                           for values in (first, second, protected, protected_ts,
                                          inputs, outputs))
                or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                       for values in (first, second, protected) for value in values)
                or protected != [max(a, b) for a, b in zip(first, second)]
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isclose(float(value), 1e9 * tokens / elapsed,
                                           rel_tol=1e-12, abs_tol=1e-12)
                       for value, elapsed in zip(protected_ts, protected))
                or len(set(inputs)) != repetitions):
            raise RuntimeError(f"{label} timed-output semantic receipt is internally inconsistent")
    if anchor_semantics.get("seed") != candidate_semantics.get("seed"):
        raise RuntimeError("matched arms did not use the same hidden input seed")
    if anchor_semantics.get("repetitions") != candidate_semantics.get("repetitions"):
        raise RuntimeError("matched arms have different timed-output repetition counts")
    if anchor_semantics.get("input_hashes") != candidate_semantics.get("input_hashes"):
        raise RuntimeError("matched arms did not execute the same hidden input bank")
    if anchor_semantics.get("output_hashes") != candidate_semantics.get("output_hashes"):
        raise _CrossArmOutputDivergence(
            "candidate timed outputs differ bitwise from the sealed anchor")
    if (anchor_semantics.get("reward_admissible") is not True
            or candidate_semantics.get("reward_admissible") is not True):
        raise RuntimeError(
            "timed-output semantics lack a complete anti-shift witness for reward admission")
    body = {
        "schema": "epyc.autokernel.cross_arm_timed_output_oracle.v1",
        "seed": anchor_semantics["seed"],
        "repetitions": anchor_semantics["repetitions"],
        "input_hashes": anchor_semantics["input_hashes"],
        "output_hashes": anchor_semantics["output_hashes"],
        "bitwise_equal": True,
        "anti_shift_witness": "hip_serialized_pair_max",
    }
    return {**body, "receipt_sha256": schemas.content_hash(body)}


def _validate_graphs_on_output_semantics(
        row: Mapping[str, Any], *, repetitions: int, seed: int) -> dict[str, Any]:
    """Prove graphs-on native timing inspected every hardened output."""
    required_true = (
        "autokernel_hardened", "autokernel_output_invariant",
        "autokernel_hybrid_ab_complete", "autokernel_thread_set_stable",
        "autokernel_escape_checks_complete")
    if any(row.get(field) is not True for field in required_true):
        raise RuntimeError("graphs-on output-integrity flags are incomplete")
    input_hashes = _split_exact(
        row.get("autokernel_input_hashes"), field="input hashes",
        count=repetitions)
    if (any(_HEX64_RE.fullmatch(value) is None for value in input_hashes)
            or len(set(input_hashes)) != repetitions):
        raise RuntimeError("graphs-on input contents are malformed or reused")
    output_pairs = _split_exact(
        row.get("autokernel_output_hashes"), field="output hash pairs",
        count=repetitions)
    output_hashes: list[str] = []
    for pair in output_pairs:
        members = pair.split("/")
        if (len(members) != 2
                or any(_HEX64_RE.fullmatch(value) is None for value in members)
                or members[0] != members[1]):
            raise RuntimeError("graphs-on timed outputs are not bitwise invariant")
        output_hashes.append(members[0])
    addresses: dict[str, list[str]] = {}
    for field in ("autokernel_input_addresses", "autokernel_context_addresses"):
        pairs = _split_exact(row.get(field), field=field, count=repetitions)
        flattened: list[str] = []
        for pair in pairs:
            members = pair.split("/")
            if (len(members) != 2
                    or any(_ADDRESS_RE.fullmatch(value) is None for value in members)):
                raise RuntimeError(f"graphs-on {field} is malformed")
            flattened.extend(members)
        if len(set(flattened)) != 2 * repetitions:
            raise RuntimeError(f"graphs-on {field} reused an address")
        addresses[field] = flattened
    samples = row.get("samples_ts")
    if (not isinstance(samples, list) or len(samples) != repetitions
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(float(value)) or float(value) <= 0
                   for value in samples)):
        raise RuntimeError("graphs-on native samples are malformed")
    body = {
        "schema": "epyc.autokernel.graphs_on_output_integrity.v1",
        "instrument_commit": READY_CONTINUE_INSTRUMENT_COMMIT,
        "seed": seed, "repetitions": repetitions,
        "input_hashes": input_hashes, "output_hashes": output_hashes,
        "input_addresses": addresses["autokernel_input_addresses"],
        "context_addresses": addresses["autokernel_context_addresses"],
        "unique_content_per_repetition": True,
        "unique_addresses_per_pair_member": True,
        "within_repetition_bitwise_equal": True,
        "graph_environment": {"GGML_CUDA_DISABLE_GRAPHS": None},
        "reward_admissible": True,
    }
    return {**body, "receipt_sha256": schemas.content_hash(body)}


def _validate_cross_arm_graphs_on_outputs(
        anchor: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for label, arm in (("anchor", anchor), ("candidate", candidate)):
        semantics = arm.get("graphs_on_output_integrity")
        if not isinstance(semantics, Mapping):
            raise RuntimeError(f"{label} lacks graphs-on output-integrity receipt")
        unsigned = {key: value for key, value in semantics.items()
                    if key != "receipt_sha256"}
        if (semantics.get("schema") !=
                "epyc.autokernel.graphs_on_output_integrity.v1"
                or semantics.get("instrument_commit") !=
                READY_CONTINUE_INSTRUMENT_COMMIT
                or semantics.get("graph_environment") != {
                    "GGML_CUDA_DISABLE_GRAPHS": None}
                or semantics.get("reward_admissible") is not True
                or semantics.get("receipt_sha256") != schemas.content_hash(unsigned)):
            raise RuntimeError(f"{label} graphs-on output receipt is invalid")
        rows.append(semantics)
    anchor_semantics, candidate_semantics = rows
    for field in ("seed", "repetitions", "input_hashes", "output_hashes"):
        if anchor_semantics.get(field) != candidate_semantics.get(field):
            raise RuntimeError(f"graphs-on cross-arm {field} differs")
    body = {
        "schema": "epyc.autokernel.cross_arm_graphs_on_output_oracle.v1",
        "seed": anchor_semantics["seed"],
        "repetitions": anchor_semantics["repetitions"],
        "input_hashes": anchor_semantics["input_hashes"],
        "output_hashes": anchor_semantics["output_hashes"],
        "cross_arm_bitwise_equal": True,
        "graph_environment": {"GGML_CUDA_DISABLE_GRAPHS": None},
        "reward_admissible": True,
    }
    return {**body, "receipt_sha256": schemas.content_hash(body)}


def cache_bool(build: Path, name: str) -> bool:
    prefix = f"{name}:BOOL="
    for line in (build / "CMakeCache.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            value = line[len(prefix):]
            if value in {"ON", "OFF"}:
                return value == "ON"
    raise RuntimeError(f"{build} does not declare {name}:BOOL=ON|OFF")


def artifacts(build: Path) -> dict:
    binary = build / "bin" / "llama-bench"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"llama-bench is not executable: {binary}")
    libraries = {
        path.name: sha256_file(path)
        for path in sorted((build / "bin").glob("*.so*")) if path.is_file()
    }
    return {"binary": str(binary.resolve()), "binary_sha256": sha256_file(binary),
            "libraries": libraries}


def build_identity(build: Path) -> dict:
    source_commit = SOURCE_COMMIT
    try:
        resolved = subprocess.run(
            ["git", "-C", str(build), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True).stdout.strip()
        if len(resolved) == 40:
            source_commit = resolved
    except (OSError, subprocess.CalledProcessError):
        pass
    identity = {
        "source_commit": source_commit,
        "hip_graphs": cache_bool(build, "GGML_HIP_GRAPHS"),
        "rocwmma_fattn": cache_bool(build, "GGML_HIP_ROCWMMA_FATTN"),
        "mmq_mfma": cache_bool(build, "GGML_HIP_MMQ_MFMA"),
        "artifacts": artifacts(build),
    }
    return identity


def _sealed_source_build_identity(
        args: argparse.Namespace, *, arm: str, build: Path,
        observed: dict) -> dict:
    """Bind a source-patch arm to the builder's already-verified identity.

    Source builds are intentionally torn down to non-git runtime snapshots, so
    ``git rev-parse`` cannot recover their distinct commits here.  The static
    deployment installs this private in-process carrier from ``GpuSourceBuild``;
    there is deliberately no corresponding CLI option.  Revalidate every live
    artifact which survives teardown before using the sealed source identity.
    """
    raw = getattr(args, f"_sealed_{arm}_source_build_identity", None)
    required = {
        "source_commit", "source_sha256", "binary_sha256",
        "hip_library_sha256", "config_sha256", "linkage_sha256",
    }
    if not isinstance(raw, dict) or set(raw) != required:
        raise RuntimeError(
            f"source patch {arm} arm lacks its sealed builder identity")
    if (not isinstance(raw["source_commit"], str)
            or len(raw["source_commit"]) != 40
            or any(ch not in "0123456789abcdef" for ch in raw["source_commit"])
            or any(not isinstance(raw[key], str) or len(raw[key]) != 64
                   or any(ch not in "0123456789abcdef" for ch in raw[key])
                   for key in required - {"source_commit"})):
        raise RuntimeError(f"source patch {arm} builder identity is malformed")
    cache = build / "CMakeCache.txt"
    binary = build / "bin" / "llama-bench"
    hip = build / "bin" / "libggml-hip.so"
    try:
        hip_resolved = hip.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(
            f"source patch {arm} HIP artifact cannot be resolved") from exc
    live = {
        "config_sha256": sha256_file(cache),
        "binary_sha256": sha256_file(binary),
        "hip_library_sha256": sha256_file(hip_resolved),
    }
    if any(raw[key] != value for key, value in live.items()):
        raise RuntimeError(
            f"source patch {arm} live artifact differs from sealed builder identity")
    return {**observed, **raw}


def _kfd_pids() -> tuple[int, ...]:
    try:
        return tuple(sorted(int(path.name) for path in KFD_PROCS.iterdir()
                            if path.name.isdigit()))
    except OSError as exc:
        raise RuntimeError(f"KFD process inventory unreadable: {exc}") from exc


def _start_ticks(pid: int, *, proc_root: Path = Path("/proc")) -> int:
    try:
        tail = (proc_root / str(pid) / "stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        return int(tail[19])  # proc stat field 22 after pid+comm
    except (OSError, IndexError, ValueError) as exc:
        raise RuntimeError("captured GPU child start ticks are unavailable") from exc


def _runtime_maps_identity(*, runtime_root: Path, arm: str, model: Path,
                           kfd_pid: int, proc_root: Path = Path("/proc"),
                           boot_id_path: Path = Path("/proc/sys/kernel/random/boot_id")) -> dict:
    """Prove actual loader mapping while the governed arm is resident."""
    try:
        manifest = split_runtime_verifier.verify_split_runtime(runtime_root)
        maps = (proc_root / str(kfd_pid) / "maps").read_text(encoding="utf-8")
        identity = split_runtime_verifier.verify_runtime_maps(
            manifest, arm=arm, maps_text=maps, model_path=model,
            model_sha256=sha256_file(model), device_id=DEVICE_ID, kfd_pid=kfd_pid,
            boot_id=boot_id_path.read_text(encoding="utf-8").strip(),
            process_start_ticks=_start_ticks(kfd_pid, proc_root=proc_root))
    except (OSError, split_runtime_verifier.SplitRuntimeError) as exc:
        raise RuntimeError(f"runtime loader-map proof refused: {exc}") from exc
    return identity.to_dict()


@dataclass(frozen=True)
class LoadReadinessPolicy:
    """Typed authority for releasing a serialized cold-load window.

    A cold serialized arm may not release the shared CPU window merely because
    a child started or because VRAM happened to rise.  This policy binds the
    exact split-runtime closure, model and arm that must be witnessed in the
    child's maps while it owns KFD residency.  There is intentionally no
    fallback for a normal build without a split-runtime maps authority.
    """

    schema: str
    runtime_root: Path
    runtime_manifest_sha256: str
    runtime_arm: str
    model_path: Path
    model_sha256: str
    device_id: str
    policy_sha256: str

    @classmethod
    def from_split_runtime(cls, *, runtime_root: Path, runtime_arm: str,
                           model: Path, device_id: str = DEVICE_ID
                           ) -> "LoadReadinessPolicy":
        if runtime_arm not in {"anchor", "candidate"}:
            raise RuntimeError("serialized cold load requires an exact runtime arm")
        root = runtime_root.resolve(strict=True)
        model_path = model.resolve(strict=True)
        manifest = split_runtime_verifier.verify_split_runtime(root)
        body = {
            "schema": "epyc.autokernel.gpu_load_readiness_policy.v1",
            "runtime_root": str(root),
            "runtime_manifest_sha256": manifest.manifest_sha256,
            "runtime_arm": runtime_arm,
            "model_path": str(model_path),
            "model_sha256": sha256_file(model_path),
            "device_id": device_id,
        }
        return cls(
            schema=body["schema"], runtime_root=root,
            runtime_manifest_sha256=body["runtime_manifest_sha256"],
            runtime_arm=runtime_arm, model_path=model_path,
            model_sha256=body["model_sha256"], device_id=device_id,
            policy_sha256=schemas.content_hash(body))

    def __post_init__(self) -> None:
        body = self.to_dict(include_hash=False)
        if (self.schema != "epyc.autokernel.gpu_load_readiness_policy.v1"
                or self.runtime_arm not in {"anchor", "candidate"}
                or self.device_id != DEVICE_ID
                or not self.runtime_root.is_absolute() or not self.model_path.is_absolute()
                or len(self.runtime_manifest_sha256) != 64
                or len(self.model_sha256) != 64
                or self.policy_sha256 != schemas.content_hash(body)):
            raise RuntimeError("serialized load readiness policy is malformed")

    def to_dict(self, *, include_hash: bool = True) -> dict[str, str]:
        body = {
            "schema": self.schema,
            "runtime_root": str(self.runtime_root),
            "runtime_manifest_sha256": self.runtime_manifest_sha256,
            "runtime_arm": self.runtime_arm,
            "model_path": str(self.model_path),
            "model_sha256": self.model_sha256,
            "device_id": self.device_id,
        }
        if include_hash:
            body["policy_sha256"] = self.policy_sha256
        return body

    def validate_witness(self, witness: Mapping[str, Any]) -> None:
        if not isinstance(witness, Mapping):
            raise RuntimeError("serialized load readiness witness is absent")
        expected = {
            "schema": split_runtime_verifier.MAPS_SCHEMA,
            "runtime_manifest_sha256": self.runtime_manifest_sha256,
            "arm": self.runtime_arm,
            "model_path": str(self.model_path),
            "model_sha256": self.model_sha256,
            "device_id": self.device_id,
        }
        observed = {key: witness.get(key) for key in expected}
        if observed != expected or not isinstance(witness.get("identity_sha256"), str):
            raise RuntimeError("serialized load readiness witness does not bind the sealed runtime/model")


@dataclass(frozen=True)
class ReadyContinueHandshake:
    """One sealed, opt-in pre-measurement barrier for the governed instrument."""

    schema: str
    decision_sha256: str
    readiness_policy_sha256: str
    arm: str
    seed: int
    repetitions: int
    token: str
    ready_path: Path
    continue_path: Path

    @classmethod
    def create(cls, *, root: Path, decision: Mapping[str, Any],
               policy: LoadReadinessPolicy, arm: str, seed: int,
               repetitions: int,
               resume_existing: bool = False) -> "ReadyContinueHandshake":
        if (not isinstance(decision.get("decision_sha256"), str)
                or len(decision["decision_sha256"]) != 64
                or arm != policy.runtime_arm or repetitions < 1):
            raise RuntimeError("ready/continue handshake lacks sealed decision authority")
        target = root.resolve()
        if not target.is_absolute() or target.is_symlink():
            raise RuntimeError("ready/continue handshake root is unsafe")
        if resume_existing and target.exists() and not target.is_symlink():
            if not target.is_dir() or tuple(target.iterdir()):
                raise RuntimeError(
                    "ready/continue resume root is not an empty real directory")
        else:
            target.mkdir(mode=0o700, parents=True, exist_ok=False)
        os.chmod(target, 0o700)
        root_stat = target.lstat()
        if (not stat.S_ISDIR(root_stat.st_mode) or root_stat.st_uid != os.geteuid()
                or stat.S_IMODE(root_stat.st_mode) != 0o700):
            raise RuntimeError("ready/continue handshake root ownership is unsafe")
        marker = {
            "schema": "epyc.autokernel.ready_continue.v1",
            "decision_sha256": decision["decision_sha256"],
            "readiness_policy_sha256": policy.policy_sha256,
            "arm": arm, "seed": seed, "repetitions": repetitions,
        }
        token = schemas.content_hash(marker)
        return cls(schema=marker["schema"], decision_sha256=marker["decision_sha256"],
                   readiness_policy_sha256=marker["readiness_policy_sha256"],
                   arm=arm, seed=seed, repetitions=repetitions, token=token,
                   ready_path=target / "ready", continue_path=target / "continue")

    def __post_init__(self) -> None:
        if (self.schema != "epyc.autokernel.ready_continue.v1" or self.arm not in {"anchor", "candidate"}
                or self.seed < 0 or self.repetitions < 1 or len(self.token) != 64
                or len(self.decision_sha256) != 64 or len(self.readiness_policy_sha256) != 64
                or not self.ready_path.is_absolute() or not self.continue_path.is_absolute()
                or self.ready_path.parent != self.continue_path.parent):
            raise RuntimeError("ready/continue handshake is malformed")

    def argv(self) -> tuple[str, ...]:
        return ("--autokernel-ready-file", str(self.ready_path),
                "--autokernel-continue-file", str(self.continue_path),
                "--autokernel-ready-token", self.token,
                "--autokernel-ready-timeout-ms", "600000")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "decision_sha256": self.decision_sha256,
                "readiness_policy_sha256": self.readiness_policy_sha256,
                "arm": self.arm, "seed": self.seed, "repetitions": self.repetitions,
                "token": self.token, "ready_path": str(self.ready_path),
                "continue_path": str(self.continue_path)}

    def validate_ready(self, *, pid: int) -> dict[str, Any]:
        try:
            file_stat = self.ready_path.lstat()
            raw = self.ready_path.read_text(encoding="ascii")
        except OSError as exc:
            raise RuntimeError("governed instrument ready receipt is unavailable") from exc
        if (self.ready_path.is_symlink() or not stat.S_ISREG(file_stat.st_mode)
                or file_stat.st_uid != os.geteuid() or file_stat.st_nlink != 1
                or stat.S_IMODE(file_stat.st_mode) != 0o600 or file_stat.st_size > 512):
            raise RuntimeError("governed instrument ready receipt is unsafe")
        fields = raw.split()
        expected = [self.schema, str(pid), str(self.seed), str(self.repetitions), self.token]
        if fields != expected or raw != " ".join(expected) + "\n":
            raise RuntimeError("governed instrument ready receipt does not bind PID/seed/repetitions/token")
        return {"schema": self.schema, "pid": pid, "seed": self.seed,
                "repetitions": self.repetitions, "token": self.token,
                "ready_path": str(self.ready_path), "continue_path": str(self.continue_path)}

    def continue_after_release(self) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(self.continue_path, flags, 0o600)
        except OSError as exc:
            raise RuntimeError("cannot create governed instrument continue receipt") from exc
        try:
            payload = (self.token + "\n").encode("ascii")
            if os.write(fd, payload) != len(payload):
                raise RuntimeError("governed instrument continue receipt write was incomplete")
            os.fsync(fd)
            file_stat = os.fstat(fd)
            if (not stat.S_ISREG(file_stat.st_mode) or file_stat.st_uid != os.geteuid()
                    or file_stat.st_nlink != 1 or stat.S_IMODE(file_stat.st_mode) != 0o600):
                raise RuntimeError("governed instrument continue receipt ownership is unsafe")
        finally:
            os.close(fd)

    def cleanup(self) -> dict[str, bool]:
        result = {"ready_removed": False, "continue_removed": False}
        for key, path in (("ready_removed", self.ready_path),
                          ("continue_removed", self.continue_path)):
            try:
                if path.exists():
                    file_stat = path.lstat()
                    if (path.is_symlink() or not stat.S_ISREG(file_stat.st_mode)
                            or file_stat.st_uid != os.geteuid() or file_stat.st_nlink != 1):
                        raise RuntimeError("governed instrument handshake marker changed ownership")
                    path.unlink()
                    result[key] = True
            except OSError as exc:
                raise RuntimeError(f"governed instrument handshake cleanup failed: {path}") from exc
        return result


class BandwidthDutyCycleBudget:
    """Sealed cold-load host-transfer budget; size alone never decides overlap."""
    def __init__(self, *, host_bandwidth_bytes_per_s: float,
                 rolling_interval_s: float, budget_fraction: float) -> None:
        if (host_bandwidth_bytes_per_s <= 0 or rolling_interval_s <= 0
                or not 0 < budget_fraction <= 1):
            raise RuntimeError("host-transfer duty-cycle budget is invalid")
        self.host_bandwidth_bytes_per_s = float(host_bandwidth_bytes_per_s)
        self.rolling_interval_s = float(rolling_interval_s)
        self.budget_fraction = float(budget_fraction)

    def admit(self, *, cold_load_host_bytes: int, observed_at_s: float,
              prior_cold_load_bytes: int = 0) -> dict:
        if (isinstance(cold_load_host_bytes, bool) or cold_load_host_bytes < 1
                or isinstance(prior_cold_load_bytes, bool) or prior_cold_load_bytes < 0
                or observed_at_s < 0):
            raise RuntimeError("host-transfer load observation is invalid")
        budget = self.host_bandwidth_bytes_per_s * self.rolling_interval_s * self.budget_fraction
        rolling = prior_cold_load_bytes + cold_load_host_bytes
        return {"schema": "epyc.autokernel.host_transfer_budget.v1",
                "cold_load_host_bytes": cold_load_host_bytes,
                "rolling_interval_s": self.rolling_interval_s,
                "host_bandwidth_bytes_per_s": self.host_bandwidth_bytes_per_s,
                "budget_fraction": self.budget_fraction,
                "rolling_cold_load_bytes": rolling, "budget_bytes": budget,
                "transfer_ratio": rolling / (self.host_bandwidth_bytes_per_s * self.rolling_interval_s),
                "observed_at_s": observed_at_s, "admitted": rolling <= budget}


@dataclass(frozen=True)
class SiteLoadProfile:
    """Reviewed, workload-specific cold-overlap authority; never planner input."""
    policy_version: str
    model_sha256: str
    model_path: str
    model_bytes: int
    workload: str
    calls_per_arm: int
    device_id: str
    worst_case_cold_loads: int
    budget: BandwidthDutyCycleBudget

    def decide(self, *, model: Path, workload: str, calls: int, device_id: str,
               observed_headroom: bool) -> dict:
        actual_sha = sha256_file(model)
        exact = (actual_sha == self.model_sha256 and str(model.resolve()) == self.model_path
                 and model.stat().st_size == self.model_bytes and workload == self.workload
                 and calls == self.calls_per_arm and device_id == self.device_id)
        transfer = self.budget.admit(cold_load_host_bytes=model.stat().st_size,
            observed_at_s=time.monotonic(), prior_cold_load_bytes=model.stat().st_size
            * max(0, self.worst_case_cold_loads - 1))
        if exact and observed_headroom and transfer["admitted"]:
            return {**transfer, "policy_version": self.policy_version, "mode": "cold_overlap",
                    "reason": "exact reviewed site load profile", "lock_interval": None,
                    "residency_transition": "cold_load_required"}
        return {**transfer, "policy_version": self.policy_version, "mode": "cold_serialized",
                "reason": "profile mismatch/missing headroom/transfer budget", "lock_interval": "load_only",
                "residency_transition": "cold_load_required"}


SITE_LOAD_PROFILES = {
    "mi210-qwen05b-tg128-18-v1": SiteLoadProfile(
        policy_version="mi210-qwen05b-tg128-18-v1",
        model_sha256="f175ecace8c24336cbf9e22bd71ea032a16492bd264a3caab6dfa4cafe80ddd3",
        model_path="/mnt/raid0/llm/models/lmstudio-community/Qwen2.5-Coder-0.5B-GGUF/Qwen2.5-Coder-0.5B-Q4_K_M.gguf",
        model_bytes=397807840, workload="decode_tg128", calls_per_arm=9,
        device_id=DEVICE_ID, worst_case_cold_loads=18,
        budget=BandwidthDutyCycleBudget(host_bandwidth_bytes_per_s=DEFAULT_HOST_BANDWIDTH_BYTES_S,
                                        rolling_interval_s=60.0, budget_fraction=.01)),
}


def decide_load_mode(*, hot_resident: bool, residency_identity_matches: bool,
                     host_observation_available: bool, transfer: dict,
                     dedicated_window_available: bool,
                     policy_version: str = "site-host-transfer-v1") -> dict:
    """Fail-closed three-mode load admission; planner text cannot select it."""
    if not policy_version or not isinstance(transfer, dict):
        raise RuntimeError("load admission policy input is malformed")
    if hot_resident:
        if not residency_identity_matches:
            raise RuntimeError("hot resident declaration lacks exact model/runtime/residency identity")
        return {"mode": "hot_resident", "policy_version": policy_version,
                "reason": "exact resident model/runtime identity", "cpu_window_required": False}
    if host_observation_available and transfer.get("admitted") is True:
        return {"mode": "cold_overlap", "policy_version": policy_version,
                "reason": "sealed host-transfer policy admitted declared cold load", "cpu_window_required": False}
    if dedicated_window_available:
        return {"mode": "cold_serialized", "policy_version": policy_version,
                "reason": "unknown/over-budget host transfer serialized for load only", "cpu_window_required": True}
    raise RuntimeError("cold GPU load cannot be admitted: no host observation budget or dedicated window")


def host_transfer_admission(*, bytes_per_cold_load: int, cold_loads: int,
                            interval_s: float, host_bandwidth_bytes_s: float,
                            conservative_fraction: float,
                            site_policy_allows_overlap: bool = True,
                            observed_headroom: bool = True,
                            hot_resident: bool = False,
                            resident_identity: str | None = None,
                            expected_identity: str | None = None) -> dict:
    """Compatibility/public policy entry point with an explicit three-mode outcome."""
    budget = BandwidthDutyCycleBudget(host_bandwidth_bytes_per_s=host_bandwidth_bytes_s,
                                      rolling_interval_s=interval_s,
                                      budget_fraction=conservative_fraction)
    transfer = budget.admit(cold_load_host_bytes=bytes_per_cold_load, observed_at_s=time.monotonic(),
                            prior_cold_load_bytes=bytes_per_cold_load * max(0, cold_loads - 1))
    exact_hot = hot_resident and resident_identity is not None and resident_identity == expected_identity
    if exact_hot:
        decision = {"mode": "hot_resident", "reason": "exact resident identity", "lock_interval": None,
                    "residency_transition": "reused"}
    elif site_policy_allows_overlap and observed_headroom and transfer["admitted"]:
        decision = {"mode": "cold_overlap", "reason": "site policy/headroom/duty-cycle admitted",
                    "lock_interval": None, "residency_transition": "cold_load_required"}
    else:
        decision = {"mode": "cold_serialized", "reason": "missing/over-budget/unsafe overlap observation",
                    "lock_interval": "load_only", "residency_transition": "cold_load_required"}
    return {**transfer, "policy_version": "site-host-transfer-v1", "inputs": {
        "site_policy_allows_overlap": site_policy_allows_overlap,
        "observed_headroom": observed_headroom, "hot_resident": hot_resident,
        "resident_identity": resident_identity, "expected_identity": expected_identity}, **decision}


def invoke(*, build: Path, model: Path, seed: int, baseline_vram: int,
           flash_attention: bool, campaign_id: str,
           expected_source_commit: str = SOURCE_COMMIT,
           prompt_tokens: int = 512, generation_tokens: int = 0,
           cpu_journal: cpu_region_claim.RegionClaimJournal,
           threads: int = 8, batch: int = 512, ubatch: int = 512,
           mmap: bool = True, no_op_offload: bool = False,
           split_mode: str = "layer", no_kv_offload: bool = False,
           poll: int = 50, inference_window_lock: Path | None = None,
           reward_binary: Path | None = None, hip_library_dir: Path | None = None,
           common_loader_dir: Path | None = None,
           runtime_arm: str | None = None,
           host_transfer_interval_s: float = 60.0,
           host_bandwidth_bytes_s: float = DEFAULT_HOST_BANDWIDTH_BYTES_S,
           host_transfer_fraction: float = DEFAULT_HOST_TRANSFER_FRACTION,
           cold_loads_in_interval: int = 1,
           sealed_load_decision: dict | None = None,
           repetitions: int = 1,
           timed_output_oracle: bool = False,
           runtime_graphs: str = "inherit",
           load_readiness_policy: LoadReadinessPolicy | None = None,
           ready_continue_handshake: ReadyContinueHandshake | None = None,
           process_factory: Callable[..., Any] | None = None,
           kfd_pid_provider: Callable[[], tuple[int, ...]] | None = None,
           vram_reader: Callable[[], int] | None = None,
           pgid_provider: Callable[[int], int] | None = None,
           sleep: Callable[[float], None] | None = None,
           supervisor_root: Path | None = None,
           process_receipt_root: Path | None = None,
           process_context: Mapping[str, Any] | None = None,
           process_resource_context: Mapping[str, Any] | None = None,
           after_process_checkpoint: Callable[[Path], None] | None = None) -> dict:
    """Run one cold load and all sealed repetitions for one discovery arm."""
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise RuntimeError("GPU discovery repetitions must be a positive integer")
    if runtime_graphs not in {"inherit", "off", "on"}:
        raise RuntimeError("runtime graph mode must be inherit, off, or on")
    if (not isinstance(sealed_load_decision, dict)
            or sealed_load_decision.get("mode") not in {"cold_overlap", "cold_serialized"}):
        raise RuntimeError("nonpersistent runner requires a sealed cold load decision")
    if sealed_load_decision["mode"] == "cold_overlap":
        model_bytes = model.stat().st_size
        if sealed_load_decision is None:
            raise RuntimeError("GPU overlap requires a preflight-sealed site load decision")
        transfer = sealed_load_decision
        if transfer["mode"] != "cold_overlap":
            raise RuntimeError("GPU cold load was not admitted for overlap; use serialized load mode")
        claims = cpu_region_claim.inspect_region_claims()
        concurrent = []
        for region, entries in (claims.get("regions") or {}).items():
            for entry in entries:
                if not entry.get("held"):
                    continue
                concurrent.append({
                    "region": region, "role": entry.get("role"),
                    "holder_pids": entry.get("holder_pids") or [],
                    "attribution": entry.get("attribution"),
                })
        overlap_coverage = {
            "schema": "epyc.autokernel.discovery_cpu_overlap.v1",
            "cpu_overlap_policy": "allowed_discovery_noise",
            "cpu_exclusivity": False, "borrowed": False,
            "model_size_bytes": model_bytes,
            "host_transfer": transfer,
            "load_mode": "cold",
            "concurrent_claims": concurrent,
            "promotion_claim": False,
        }
        result = _invoke_locked(
            build=build, model=model, seed=seed, baseline_vram=baseline_vram,
            expected_source_commit=expected_source_commit,
            flash_attention=flash_attention, prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens, threads=threads, ubatch=ubatch,
            batch=batch, mmap=mmap, no_op_offload=no_op_offload,
            split_mode=split_mode, no_kv_offload=no_kv_offload, poll=poll,
            reward_binary=reward_binary, hip_library_dir=hip_library_dir, common_loader_dir=common_loader_dir,
            runtime_arm=runtime_arm, repetitions=repetitions,
            timed_output_oracle=timed_output_oracle,
            runtime_graphs=runtime_graphs,
            process_factory=process_factory, kfd_pid_provider=kfd_pid_provider,
            vram_reader=vram_reader, pgid_provider=pgid_provider, sleep=sleep,
            supervisor_root=supervisor_root,
            process_receipt_root=process_receipt_root,
            process_context=process_context,
            process_resource_context={
                **({} if process_resource_context is None else
                   dict(process_resource_context)),
                "cpu_coverage": overlap_coverage,
                "inference_call_window": None,
            },
            after_process_checkpoint=after_process_checkpoint)
        # Deliberately no shared CPU inference-window lock here.  The sealed
        # admission receipt, not model size or caller flags, admits this noise.
        result["inference_call_window"] = None
        carried = result.get("supervisor", {}).get(
            "process_resource_context")
        result["cpu_coverage"] = (
            carried["cpu_coverage"]
            if isinstance(carried, Mapping)
            and isinstance(carried.get("cpu_coverage"), Mapping)
            else overlap_coverage)
        result["load_admission_decision"] = sealed_load_decision
        result["load_readiness_transition"] = {
            "schema": "epyc.autokernel.gpu_load_readiness_transition.v1",
            "status": "not_required_cold_overlap",
            "lock_released_before_measurement": True,
        }
        return result
    # JSONL is emitted only after llama-bench completes its repetitions.  It
    # cannot prove a point *before* the first timed sample, so absent an
    # explicit instrument ready/continue barrier we conservatively retain the
    # serialized lock for the complete one-load batched process.
    if load_readiness_policy is not None and (
            load_readiness_policy.model_path != model.resolve()
            or load_readiness_policy.model_sha256 != sha256_file(model)
            or load_readiness_policy.runtime_arm != runtime_arm):
        raise RuntimeError("serialized load readiness policy does not bind this arm/model")
    if ready_continue_handshake is not None and (
            load_readiness_policy is None
            or ready_continue_handshake.arm != runtime_arm
            or ready_continue_handshake.seed != seed
            or ready_continue_handshake.repetitions != repetitions
            or ready_continue_handshake.readiness_policy_sha256 != load_readiness_policy.policy_sha256):
        raise RuntimeError("ready/continue handshake does not bind this serialized arm")
    window = (inference_window.InferenceCallWindow(inference_window_lock, timeout_s=600.0)
              if inference_window_lock is not None else MODEL_CALL_WINDOW)
    configured_lease = window.acquire()
    owned_claim = None
    coverage: Any = None
    coverage_receipt: dict[str, Any] | None = None
    transition: dict[str, Any] = {
        "schema": "epyc.autokernel.gpu_load_readiness_transition.v1",
        "status": "instrument_barrier_unavailable_held_through_process",
        "lock_released_before_measurement": False,
        "required_instrument_capability": "autokernel-ready-continue-v1",
        "readiness_policy": (None if load_readiness_policy is None
                             else load_readiness_policy.to_dict()),
    }
    invocation_resource_context: dict[str, Any] = {}

    def release_for_ready(witness: Mapping[str, Any]) -> None:
        nonlocal owned_claim, transition
        if getattr(coverage, "borrowed", False):
            coverage.validate()
        elif owned_claim is not None:
            owned_claim.release()
            owned_claim = None
        configured_lease.release()
        transition = {
            "schema": "epyc.autokernel.gpu_load_readiness_transition.v1",
            "status": "ready_witnessed_lock_released_before_continue",
            "lock_released_before_measurement": True,
            "lock_path": str(configured_lease.path),
            "waited_s": configured_lease.waited_s,
            "witness": dict(witness),
            "readiness_policy": load_readiness_policy.to_dict(),
            "handshake": ready_continue_handshake.to_dict(),
        }
        invocation_resource_context["load_readiness_transition"] = transition

    try:
        try:
            owned_claim = cpu_region_claim.acquire_cpu_region_claim(
                CPU_LIST, purpose="AutoKernel GPU cold-load helper window",
                campaign_id=campaign_id, journal=cpu_journal,
                role="autokernel-gpu-discovery", timeout_s=0, max_hold_s=300)
            coverage = owned_claim
            coverage_receipt = {
                "schema": "epyc.autokernel.owned_cpu_coverage.v1",
                "borrowed": False,
                "claim": owned_claim.receipt().to_dict(),
            }
        except cpu_region_claim.CpuRegionClaimTimeout:
            coverage = inference_window.borrow_windowed_cpu_coverage(CPU_LIST)
            coverage_receipt = coverage.to_dict()
        invocation_resource_context.update({
            **({} if process_resource_context is None else
               dict(process_resource_context)),
            "cpu_coverage": coverage_receipt,
            "inference_call_window": {
                "schema": "epyc.autokernel.inference_call_window.v1",
                "lock_path": str(configured_lease.path),
                "waited_s": configured_lease.waited_s,
                "scope": "one_load_and_all_batched_measurements_no_ready_barrier",
            },
            "load_readiness_transition": transition,
        })
        result = _invoke_locked(
            build=build, model=model, seed=seed, baseline_vram=baseline_vram,
            expected_source_commit=expected_source_commit,
            flash_attention=flash_attention, prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens, threads=threads, ubatch=ubatch,
            batch=batch, mmap=mmap, no_op_offload=no_op_offload,
            split_mode=split_mode, no_kv_offload=no_kv_offload, poll=poll,
            reward_binary=reward_binary, hip_library_dir=hip_library_dir,
            common_loader_dir=common_loader_dir, runtime_arm=runtime_arm,
            repetitions=repetitions, readiness_policy=load_readiness_policy,
            timed_output_oracle=timed_output_oracle,
            runtime_graphs=runtime_graphs,
            ready_continue_handshake=ready_continue_handshake,
            on_load_ready=(release_for_ready if ready_continue_handshake is not None else None),
            process_factory=process_factory,
            kfd_pid_provider=kfd_pid_provider, vram_reader=vram_reader,
            pgid_provider=pgid_provider, sleep=sleep,
            supervisor_root=supervisor_root,
            process_receipt_root=process_receipt_root,
            process_context=process_context,
            process_resource_context=invocation_resource_context,
            after_process_checkpoint=after_process_checkpoint)
        if getattr(coverage, "borrowed", False) and configured_lease.held:
            coverage.validate()
    finally:
        if owned_claim is not None:
            try:
                owned_claim.release()
            finally:
                owned_claim = None
        configured_lease.release()
        if ready_continue_handshake is not None:
            cleanup = ready_continue_handshake.cleanup()
            transition = {**transition, "handshake_cleanup": cleanup}
    carried = result.get("supervisor", {}).get("process_resource_context")
    carried_window = (carried.get("inference_call_window")
                      if isinstance(carried, Mapping) else None)
    result["inference_call_window"] = {
        "schema": "epyc.autokernel.inference_call_window.v1",
        "lock_path": (carried_window.get("lock_path")
                      if isinstance(carried_window, Mapping)
                      else str(configured_lease.path)),
        "waited_s": (carried_window.get("waited_s")
                     if isinstance(carried_window, Mapping)
                     else configured_lease.waited_s),
        "scope": "one_load_and_all_batched_measurements_no_ready_barrier",
        "released": configured_lease.held is False,
    }
    result["cpu_coverage"] = (
        carried["cpu_coverage"]
        if isinstance(carried, Mapping)
        and isinstance(carried.get("cpu_coverage"), Mapping)
        else coverage_receipt or {})
    result["load_mode"] = "cold_serialized"
    result["site_load_decision"] = sealed_load_decision
    result["load_admission_decision"] = sealed_load_decision
    result["load_readiness_transition"] = (
        carried["load_readiness_transition"]
        if isinstance(carried, Mapping)
        and isinstance(carried.get("load_readiness_transition"), Mapping)
        else transition)
    return result


def _validate_completed_invocation(
        *, stdout: bytes, stderr: bytes, returncode: int,
        build: Path, model: Path, seed: int, baseline_vram: int,
        flash_attention: bool, prompt_tokens: int,
        expected_source_commit: str | None, generation_tokens: int,
        threads: int, ubatch: int, batch: int, mmap: bool,
        no_op_offload: bool, split_mode: str, no_kv_offload: bool, poll: int,
        binary: Path, loader_dir: Path, common_dir: Path,
        repetitions: int, timed_output_oracle: bool, runtime_graphs: str,
        runtime_arm: str | None, max_runtime_s: float,
        serialization_env: Mapping[str, str], argv: tuple[str, ...],
        env: Mapping[str, str], residency: list[dict[str, Any]],
        maps_identity: Mapping[str, Any] | None,
        readiness_witness: Mapping[str, Any] | None,
        ready_continue_handshake: ReadyContinueHandshake | None,
        elapsed_s: float, teardown: Mapping[str, Any]) -> dict[str, Any]:
    if returncode != 0:
        raise _NativeOutputError(
            "process_exit_nonzero",
            f"GPU discovery invocation exited {returncode}")
    row, native_diagnostic = _parse_native_measurement(
        stdout, repetitions=repetitions,
        tokens_per_repetition=prompt_tokens + generation_tokens)
    exact_samples = native_diagnostic["rederived_samples_ts"]
    metric = native_diagnostic["rederived_avg_ts"]
    if row.get("backends") != "ROCm" or row.get("gpu_info") != "AMD Instinct MI210":
        raise _NativeOutputError(
            "backend_identity", "GPU discovery invocation did not report MI210 ROCm execution")
    reported_commit = str(row.get("build_commit", ""))
    if expected_source_commit is not None and (
            len(reported_commit) < 7
            or not expected_source_commit.startswith(reported_commit)):
        raise _NativeOutputError(
            "source_identity", "GPU discovery binary does not report the sealed source commit")
    expected_flash = 1 if flash_attention else 0
    if (row.get("n_prompt") != prompt_tokens
            or row.get("n_gen") != generation_tokens
            or row.get("flash_attn") != expected_flash):
        raise _NativeOutputError(
            "workload_frame", "GPU discovery result differs from the sealed workload frame")
    expected = {
        "n_threads": threads, "n_batch": batch, "n_ubatch": ubatch,
        "use_mmap": mmap, "no_op_offload": 1 if no_op_offload else 0,
        "split_mode": split_mode, "no_kv_offload": no_kv_offload,
        "poll": poll,
    }
    mismatched = {key: (row.get(key), value) for key, value in expected.items()
                  if row.get(key) != value}
    if mismatched:
        raise _NativeOutputError(
            "runtime_config",
            f"GPU discovery result differs from sealed runtime config: {mismatched}")
    if not any(sample["owned_kfd_pids"] for sample in residency):
        raise _NativeOutputError(
            "kfd_residency", "GPU discovery window has no owned KFD residency sample")
    if max(sample["vram_used_bytes"] for sample in residency) <= baseline_vram:
        raise _NativeOutputError(
            "vram_residency", "GPU discovery window has no positive VRAM residency delta")
    if runtime_arm is not None and maps_identity is None:
        raise _NativeOutputError(
            "runtime_maps", "GPU discovery window lacks sealed runtime loader-map identity")
    if ready_continue_handshake is not None and readiness_witness is None:
        raise _NativeOutputError(
            "readiness", "governed instrument exited without ready before measurement")
    try:
        timed_output_semantics = (
            _validate_timed_output_semantics(
                row, repetitions=repetitions, seed=seed,
                tokens_per_repetition=prompt_tokens + generation_tokens,
                serialization_env={key: env[key] for key in serialization_env})
            if timed_output_oracle else None)
        graphs_on_output_integrity = (
            _validate_graphs_on_output_semantics(
                row, repetitions=repetitions, seed=seed)
            if runtime_graphs == "on" else None)
    except RuntimeError as exc:
        raise _NativeOutputError(
            "timed_output_semantics", str(exc)) from exc
    protected_samples = (timed_output_semantics["protected_samples_ts"]
                         if timed_output_semantics is not None
                         else exact_samples)
    protected_metric = (1e9 * (prompt_tokens + generation_tokens) * repetitions
                        / sum(timed_output_semantics["protected_samples_ns"])
                        if timed_output_semantics is not None else metric)
    metric_contract = ({
        "schema": "epyc.autokernel.serialized_pair_max_metric.v1",
        "scope": "integrity_discovery_only",
        "production_throughput_authority": False,
        "graph_mode": "disabled_for_integrity",
        "scored_sample": "min(first_tokens_per_s,second_tokens_per_s)",
        "serialization_env": dict(serialization_env),
    } if timed_output_semantics is not None else {
        "schema": "epyc.autokernel.native_llama_bench_metric.v1",
        "scope": ("target_runtime_graphs_on_direction"
                  if runtime_graphs == "on" else "legacy_nonpromotable_discovery"),
        "production_throughput_authority": runtime_graphs == "on",
        **({"graph_mode": runtime_graphs,
            **({"graph_environment": {"GGML_CUDA_DISABLE_GRAPHS": "1"}}
               if runtime_graphs == "off" else {
                   "graph_environment": {"GGML_CUDA_DISABLE_GRAPHS": None}})}
           if runtime_graphs != "inherit" else {}),
    })
    native_diagnostic = {
        **native_diagnostic,
        "reward_authority": timed_output_semantics is None,
        "production_throughput_authority": False,
    }
    unsigned_diagnostic = {key: value for key, value in native_diagnostic.items()
                           if key != "receipt_sha256"}
    native_diagnostic["receipt_sha256"] = schemas.content_hash(unsigned_diagnostic)
    return {
        "argv": list(argv),
        "env": {"LD_LIBRARY_PATH": env["LD_LIBRARY_PATH"],
                **dict(serialization_env)},
        "reward_binary": str(binary), "reward_binary_sha256": sha256_file(binary),
        "hip_library": str(loader_dir / "libggml-hip.so"),
        "hip_library_sha256": sha256_file(loader_dir / "libggml-hip.so"),
        "common_loader_dir": str(common_dir),
        "metric": protected_metric, "samples": protected_samples,
        "metric_contract": metric_contract,
        "native_metric_diagnostic": native_diagnostic,
        "sample_count": repetitions, "seed": seed, "raw_row": row,
        "stderr_tail": stderr.decode("utf-8", errors="replace")[-2000:],
        "residency": residency,
        "supervisor": {
            "deadline_s": float(max_runtime_s), "elapsed_s": elapsed_s,
            "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
            "teardown": dict(teardown), "temporary_output_cleaned": True,
        },
        "runtime_maps_identity": maps_identity,
        "load_readiness_witness": readiness_witness,
        **({"timed_output_semantics": timed_output_semantics}
           if timed_output_semantics is not None else {}),
        **({"graphs_on_output_integrity": graphs_on_output_integrity}
           if graphs_on_output_integrity is not None else {}),
        "hip_residency_proved": True,
    }


def _invoke_locked(*, build: Path, model: Path, seed: int, baseline_vram: int,
                   flash_attention: bool, prompt_tokens: int = 512,
                   expected_source_commit: str = SOURCE_COMMIT,
                   generation_tokens: int = 0, threads: int = 8, ubatch: int = 512,
                   batch: int = 512, mmap: bool = True,
                   no_op_offload: bool = False, split_mode: str = "layer",
                   no_kv_offload: bool = False, poll: int = 50,
                   reward_binary: Path | None = None, hip_library_dir: Path | None = None,
                   common_loader_dir: Path | None = None, runtime_arm: str | None = None,
                   repetitions: int = 1,
                   timed_output_oracle: bool = False,
                   runtime_graphs: str = "inherit",
                   readiness_policy: LoadReadinessPolicy | None = None,
                   ready_continue_handshake: ReadyContinueHandshake | None = None,
                   on_load_ready: Callable[[Mapping[str, Any]], None] | None = None,
                   process_factory: Callable[..., Any] | None = None,
                   kfd_pid_provider: Callable[[], tuple[int, ...]] | None = None,
                   vram_reader: Callable[[], int] | None = None,
                   pgid_provider: Callable[[int], int] | None = None,
                   sleep: Callable[[float], None] | None = None,
                   max_runtime_s: float = 1800.0,
                   supervisor_root: Path | None = None,
                   process_receipt_root: Path | None = None,
                   process_context: Mapping[str, Any] | None = None,
                   process_resource_context: Mapping[str, Any] | None = None,
                   after_process_checkpoint: Callable[[Path], None] | None = None) -> dict:
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise RuntimeError("GPU discovery repetitions must be a positive integer")
    if (runtime_graphs not in {"inherit", "off", "on"}
            or timed_output_oracle and runtime_graphs not in {"inherit", "off"}):
        raise RuntimeError("timed-output oracle requires graphs off")
    if (isinstance(max_runtime_s, bool) or not isinstance(max_runtime_s, (int, float))
            or not math.isfinite(float(max_runtime_s)) or not 1 <= max_runtime_s <= 3600):
        raise RuntimeError("GPU discovery supervisor deadline is outside reviewed bounds")
    if readiness_policy is not None and (
            runtime_arm != readiness_policy.runtime_arm
            or model.resolve() != readiness_policy.model_path
            or sha256_file(model) != readiness_policy.model_sha256
            or common_loader_dir is None or hip_library_dir is None):
        raise RuntimeError("serialized readiness policy lacks its exact runtime/model closure")
    if (ready_continue_handshake is None) != (on_load_ready is None):
        raise RuntimeError("ready/continue handshake and release callback must be paired")
    if ready_continue_handshake is not None and readiness_policy is None:
        raise RuntimeError("ready/continue handshake requires a typed readiness policy")
    binary = (reward_binary or build / "bin" / "llama-bench").resolve()
    loader_dir = (hip_library_dir or build / "bin").resolve()
    common_dir = (common_loader_dir or binary.parent).resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError("sealed reward executable is not executable")
    if not loader_dir.is_dir() or not (loader_dir / "libggml-hip.so").is_file():
        raise RuntimeError("sealed HIP loader directory lacks libggml-hip.so")
    argv = ("/usr/bin/taskset", "-c", CPU_LIST, "/usr/bin/numactl", "--interleave=all", str(binary),
            "-m", str(model), "-p", str(prompt_tokens), "-n", str(generation_tokens),
            "-r", str(repetitions), "-ngl", "99",
            "-fa", "on" if flash_attention else "off",
            "-t", str(threads), "-b", str(batch), "-ub", str(ubatch),
            "-mmp", "1" if mmap else "0",
            "-nopo", "1" if no_op_offload else "0", "-sm", split_mode,
            "-nkvo", "1" if no_kv_offload else "0",
            "--poll", str(poll),
            "--autokernel-harden", str(seed),
            *(ready_continue_handshake.argv() if ready_continue_handshake else ()),
            "-o", "jsonl")
    if not common_dir.is_dir():
        raise RuntimeError("sealed common reward loader directory is absent")
    if not isinstance(timed_output_oracle, bool):
        raise RuntimeError("timed-output oracle capability must be boolean")
    serialization_env = ({
        "AMD_SERIALIZE_KERNEL": "3", "AMD_SERIALIZE_COPY": "3",
        "GGML_CUDA_DISABLE_GRAPHS": "1",
    }
                         if timed_output_oracle else (
                             {"GGML_CUDA_DISABLE_GRAPHS": "1"}
                             if runtime_graphs == "off" else {}))
    env = {"PATH": "/usr/bin:/bin",
           "LD_LIBRARY_PATH": f"{loader_dir}:{common_dir}:/opt/rocm/lib",
           **serialization_env}
    capture_identity = _process_capture_identity(
        argv=argv, env=env, binary=binary, loader_dir=loader_dir,
        model=model, seed=seed, repetitions=repetitions,
        runtime_graphs=runtime_graphs, runtime_arm=runtime_arm,
        process_context=process_context)

    def validate_capture(capture: Mapping[str, Any], *, reused: bool) -> dict[str, Any]:
        receipt = capture["receipt"]
        try:
            if (receipt.get("stdout", {}).get("truncated") is True
                    or receipt.get("stderr", {}).get("truncated") is True):
                raise _NativeOutputError(
                    "process_output_truncated",
                    "GPU discovery process output exceeded its durable raw-byte bound")
            result = _validate_completed_invocation(
                stdout=capture["stdout"], stderr=capture["stderr"],
                returncode=receipt["returncode"], build=build, model=model,
                seed=seed, baseline_vram=baseline_vram,
                flash_attention=flash_attention, prompt_tokens=prompt_tokens,
                expected_source_commit=expected_source_commit,
                generation_tokens=generation_tokens, threads=threads,
                ubatch=ubatch, batch=batch, mmap=mmap,
                no_op_offload=no_op_offload, split_mode=split_mode,
                no_kv_offload=no_kv_offload, poll=poll, binary=binary,
                loader_dir=loader_dir, common_dir=common_dir,
                repetitions=repetitions, timed_output_oracle=timed_output_oracle,
                runtime_graphs=runtime_graphs, runtime_arm=runtime_arm,
                max_runtime_s=max_runtime_s, serialization_env=serialization_env,
                argv=argv, env=env, residency=receipt["residency"],
                maps_identity=receipt["runtime_maps_identity"],
                readiness_witness=receipt["load_readiness_witness"],
                ready_continue_handshake=ready_continue_handshake,
                elapsed_s=receipt["supervisor_elapsed_s"],
                teardown=receipt["teardown"])
        except _NativeOutputError as exc:
            if process_receipt_root is None:
                raise RuntimeError(str(exc)) from exc
            if exc.code == "timed_output_semantics":
                raise _seal_timed_output_infrastructure_ambiguity(
                    process_receipt_root, capture=capture,
                    code=exc.code, message=str(exc)) from exc
            raise _seal_output_refusal(
                process_receipt_root, capture=capture,
                code=exc.code, message=str(exc)) from exc
        refusal_path = (None if process_receipt_root is None else
                        process_receipt_root.with_name(
                            f"{process_receipt_root.name}-refusal.json"))
        if refusal_path is not None and (
                refusal_path.exists() or refusal_path.is_symlink()):
            raise RuntimeError(
                "GPU discovery output refusal no longer rederives from captured bytes")
        result["supervisor"]["process_receipt_path"] = capture["receipt_path"]
        result["supervisor"]["process_receipt_file_sha256"] = capture[
            "receipt_file_sha256"]
        result["supervisor"]["process_reused"] = reused
        result["supervisor"]["process_resource_context"] = receipt.get(
            "resource_context")
        return result

    if process_receipt_root is not None and (
            process_receipt_root.exists() or process_receipt_root.is_symlink()):
        capture = _load_process_capture(
            process_receipt_root, identity=capture_identity)
        return validate_capture(capture, reused=True)
    factory = subprocess.Popen if process_factory is None else process_factory
    kfd_provider = _kfd_pids if kfd_pid_provider is None else kfd_pid_provider
    pgid = os.getpgid if pgid_provider is None else pgid_provider
    pause = time.sleep if sleep is None else sleep
    def read_vram() -> int:
        if vram_reader is not None:
            return vram_reader()
        try:
            return int(VRAM_USED.read_text(encoding="utf-8").strip())
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"VRAM residency counter unreadable: {exc}") from exc

    # Real children write to regular files, not PIPEs: llama-bench may emit
    # enough diagnostics to fill a pipe while the supervisor is sampling KFD,
    # which would deadlock the process and retain the shared inference lock.
    if supervisor_root is not None:
        supervisor_root = supervisor_root.resolve()
        supervisor_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        root_stat = supervisor_root.lstat()
        if (supervisor_root.is_symlink() or root_stat.st_uid != os.getuid()
                or stat.S_IMODE(root_stat.st_mode) != 0o700):
            raise RuntimeError("GPU supervisor output root is not private operation authority")
    output_context = tempfile.TemporaryDirectory(
        prefix="arm-", dir=None if supervisor_root is None else supervisor_root)
    output_root = Path(output_context.name)
    stdout_path, stderr_path = output_root / "stdout", output_root / "stderr"
    stdout_handle = stdout_path.open("w+", encoding="utf-8")
    stderr_handle = stderr_path.open("w+", encoding="utf-8")
    os.chmod(stdout_path, 0o600); os.chmod(stderr_path, 0o600)
    real_process = process_factory is None
    try:
        process = factory(argv, env=env, stdin=subprocess.DEVNULL,
                          stdout=(stdout_handle if real_process else subprocess.PIPE),
                          stderr=(stderr_handle if real_process else subprocess.PIPE),
                          text=True, start_new_session=True)
    except BaseException:
        stdout_handle.close(); stderr_handle.close(); output_context.cleanup()
        raise
    samples = []
    maps_identity = None
    readiness_witness = None
    supervisor_started = time.monotonic()
    teardown: dict[str, Any] = {"required": False, "term_sent": False,
                                "kill_sent": False, "death_proved": False}
    stdout = stderr = ""
    durable_capture: dict[str, Any] | None = None
    captured_owned: set[int] = set()
    def stop_child() -> None:
        teardown["required"] = True
        if process.poll() is None:
            try:
                if real_process:
                    if os.getpgid(process.pid) != process.pid:
                        raise RuntimeError("GPU discovery child does not own its sealed process group")
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                teardown["term_sent"] = True
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if real_process:
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                teardown["kill_sent"] = True
                process.wait(timeout=10)
        if process.returncode is None:
            raise RuntimeError("GPU discovery child remained alive after TERM/KILL teardown")
        remaining = captured_owned.intersection(kfd_provider())
        if remaining:
            raise RuntimeError(f"GPU discovery owned KFD descendants survived teardown: {sorted(remaining)}")
        if real_process:
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError("GPU discovery process group survived teardown")
        teardown["death_proved"] = True
    try:
      while process.poll() is None:
        if time.monotonic() - supervisor_started > max_runtime_s:
            raise RuntimeError("GPU discovery supervisor deadline exceeded")
        kfd = kfd_provider()
        try:
            vram = read_vram()
        except BaseException:
            raise
        owned = []
        foreign = []
        for pid in kfd:
            try:
                (owned if pgid(pid) == process.pid else foreign).append(pid)
            except (ProcessLookupError, PermissionError):
                continue
        if foreign:
            raise RuntimeError(f"foreign KFD inference overlapped discovery: {foreign}")
        captured_owned.update(owned)
        samples.append({"offset_s": time.monotonic(), "kfd_pids": list(kfd),
                        "owned_kfd_pids": owned, "vram_used_bytes": vram})
        if (maps_identity is None and runtime_arm is not None and owned
                and vram > baseline_vram and common_loader_dir is not None
                and hip_library_dir is not None):
            maps_identity = _runtime_maps_identity(runtime_root=common_loader_dir.parent,
                arm=runtime_arm, model=model, kfd_pid=owned[0])
        if ready_continue_handshake is not None and readiness_witness is None:
            if len(owned) == 1 and vram > baseline_vram and maps_identity is not None:
                assert readiness_policy is not None and on_load_ready is not None
                readiness_policy.validate_witness(maps_identity)
                ready = ready_continue_handshake.validate_ready(pid=process.pid)
                readiness_witness = {
                    "ready": ready, "owned_kfd_pids": list(owned),
                    "vram_used_bytes": vram, "baseline_vram_bytes": baseline_vram,
                    "runtime_maps_identity": maps_identity,
                    "sample_offset_s": samples[-1]["offset_s"],
                }
                # Ordering is the contract: lock/claim release is complete
                # before the token that permits the first timed sample exists.
                on_load_ready(readiness_witness)
                ready_continue_handshake.continue_after_release()
        pause(0.05)
      if real_process:
          stdout_handle.flush(); stderr_handle.flush()
          for artifact in (stdout_path, stderr_path):
              output_stat = artifact.lstat()
              if (artifact.is_symlink() or output_stat.st_uid != os.getuid()
                      or output_stat.st_nlink != 1
                      or stat.S_IMODE(output_stat.st_mode) & 0o077):
                  raise RuntimeError("GPU supervisor output carrier is unsafe")
          stdout_handle.seek(0); stderr_handle.seek(0)
          stdout, stderr = stdout_handle.read(), stderr_handle.read()
      else:
          stdout, stderr = process.communicate(timeout=10)
      stdout_bytes = stdout.encode("utf-8")
      stderr_bytes = stderr.encode("utf-8")
      if process_receipt_root is not None:
          durable_capture = _seal_process_capture(
              process_receipt_root, identity=capture_identity,
              returncode=process.returncode, stdout=stdout_bytes,
              stderr=stderr_bytes, residency=samples,
              runtime_maps_identity=maps_identity,
              readiness_witness=readiness_witness,
              elapsed_s=time.monotonic() - supervisor_started,
              teardown=teardown, resource_context=process_resource_context)
    except BaseException as original:
        try:
            stop_child()
        except BaseException as teardown_error:
            raise RuntimeError(
                f"GPU discovery teardown failed after {type(original).__name__}: {teardown_error}") from original
        raise
    finally:
        stdout_handle.close(); stderr_handle.close()
        output_context.cleanup()
    if (durable_capture is not None
            and process_receipt_root is not None
            and after_process_checkpoint is not None):
        after_process_checkpoint(process_receipt_root)
    if durable_capture is None:
        # Unit-level callers may exercise the supervisor without a durable
        # operation root.  Governed runner calls always provide one.
        ephemeral_receipt = {
            "returncode": process.returncode,
            "residency": samples,
            "runtime_maps_identity": maps_identity,
            "load_readiness_witness": readiness_witness,
            "supervisor_elapsed_s": time.monotonic() - supervisor_started,
            "teardown": teardown,
        }
        durable_capture = {
            "receipt": ephemeral_receipt,
            "receipt_path": "",
            "receipt_file_sha256": "",
            "stdout": stdout.encode("utf-8"),
            "stderr": stderr.encode("utf-8"),
        }
    return validate_capture(durable_capture, reused=False)

def factor_spec(*, factor: str, anchor_build: Path, candidate_build: Path,
                anchor_identity: dict, candidate_identity: dict) -> dict:
    """Validate and describe the only difference admitted by this screen."""
    if factor == "mmq_mfma":
        if not anchor_identity["rocwmma_fattn"] or not candidate_identity["rocwmma_fattn"]:
            raise RuntimeError("both MMQ arms must keep ROCWMMA_FATTN=ON")
        if anchor_identity["mmq_mfma"] is not True or candidate_identity["mmq_mfma"] is not False:
            raise RuntimeError("sole factor must be GGML_HIP_MMQ_MFMA ON->OFF")
        return {
            "name": "GGML_HIP_MMQ_MFMA", "anchor": "ON", "candidate": "OFF",
            "anchor_flash_attention": True, "candidate_flash_attention": True,
        }
    if factor == "flash_attention":
        if anchor_build != candidate_build:
            raise RuntimeError("flash_attention screen requires one identical build path for both arms")
        if anchor_identity != candidate_identity:
            raise RuntimeError("flash_attention screen requires identical sealed build identities")
        if not anchor_identity["rocwmma_fattn"] or anchor_identity["mmq_mfma"]:
            raise RuntimeError("flash_attention screen requires the r1m0 build (ROCWMMA ON, MMQ MFMA OFF)")
        return {
            "name": "flash_attention", "anchor": "OFF", "candidate": "ON",
            "anchor_flash_attention": False, "candidate_flash_attention": True,
        }
    if factor == "rocwmma_fattn":
        if anchor_identity["mmq_mfma"] or candidate_identity["mmq_mfma"]:
            raise RuntimeError("both ROCWMMA arms must keep MMQ_MFMA=OFF")
        if anchor_identity["rocwmma_fattn"] is not False or candidate_identity["rocwmma_fattn"] is not True:
            raise RuntimeError("sole factor must be GGML_HIP_ROCWMMA_FATTN OFF->ON")
        return {
            "name": "GGML_HIP_ROCWMMA_FATTN", "anchor": "OFF", "candidate": "ON",
            "anchor_flash_attention": True, "candidate_flash_attention": True,
        }
    if factor == "hip_graphs":
        if anchor_identity["rocwmma_fattn"] != candidate_identity["rocwmma_fattn"]:
            raise RuntimeError("HIP graphs arms must keep ROCWMMA_FATTN identical")
        if anchor_identity["mmq_mfma"] != candidate_identity["mmq_mfma"]:
            raise RuntimeError("HIP graphs arms must keep MMQ_MFMA identical")
        if anchor_identity["hip_graphs"] is not True or candidate_identity["hip_graphs"] is not False:
            raise RuntimeError("sole factor must be GGML_HIP_GRAPHS ON->OFF")
        return {
            "name": "GGML_HIP_GRAPHS", "anchor": "ON", "candidate": "OFF",
            "anchor_flash_attention": True, "candidate_flash_attention": True,
        }
    if factor == "source_patch":
        for key in ("hip_graphs", "rocwmma_fattn", "mmq_mfma"):
            if anchor_identity[key] != candidate_identity[key]:
                raise RuntimeError(
                    f"source patch arms must keep {key} compile setting identical")
        if anchor_identity["source_commit"] == candidate_identity["source_commit"]:
            raise RuntimeError("source patch arms must have distinct source commits")
        return {
            "name": "source_patch",
            "anchor": anchor_identity["source_commit"][:12],
            "candidate": candidate_identity["source_commit"][:12],
            "anchor_flash_attention": True,
            "candidate_flash_attention": True,
        }
    if factor in {"helper_threads", "helper_threads_12", "helper_threads_16",
                  "helper_threads_24", "batch", "batch_up", "ubatch", "ubatch_up",
                  "mmap", "op_offload", "split_row", "kv_offload", "poll_zero"}:
        if anchor_build != candidate_build or anchor_identity != candidate_identity:
            raise RuntimeError(f"{factor} screen requires one identical sealed build")
        configs = {
            "helper_threads": ("gpu_helper_threads", 8, 4),
            "helper_threads_12": ("gpu_helper_threads", 8, 12),
            "helper_threads_16": ("gpu_helper_threads", 8, 16),
            "helper_threads_24": ("gpu_helper_threads", 8, 24),
            "batch": ("batch_size", 512, 256),
            "batch_up": ("batch_size", 512, 1024),
            "ubatch": ("ubatch_size", 512, 256),
            "ubatch_up": ("ubatch_size", 512, 1024),
            "mmap": ("mmap", "ON", "OFF"),
            "op_offload": ("op_offload", "ON", "OFF"),
            "split_row": ("split_mode", "layer", "row"),
            "kv_offload": ("kv_offload", "ON", "OFF"),
            "poll_zero": ("gpu_poll", 50, 0),
        }
        name, anchor, candidate = configs[factor]
        result = {"name": name, "anchor": anchor, "candidate": candidate,
                "anchor_flash_attention": True, "candidate_flash_attention": True,
                "anchor_threads": 8,
                "candidate_threads": (4 if factor == "helper_threads" else
                                      12 if factor == "helper_threads_12" else
                                      16 if factor == "helper_threads_16" else 8),
                "anchor_batch": 512,
                "candidate_batch": (256 if factor == "batch" else
                                    1024 if factor == "batch_up" else 512),
                "anchor_ubatch": 512,
                "candidate_ubatch": (256 if factor == "ubatch" else
                                     1024 if factor == "ubatch_up" else 512),
                "anchor_mmap": True, "candidate_mmap": False if factor == "mmap" else True}
        if factor in {"batch", "batch_up", "ubatch", "ubatch_up"}:
            # llama.cpp clamps the micro-batch to the batch (src/llama-context.cpp:265):
            #   cparams.n_ubatch = std::min(cparams.n_batch,
            #                               params.n_ubatch == 0 ? params.n_batch : params.n_ubatch)
            # so a candidate ubatch ABOVE the candidate batch is silently clamped back to
            # the anchor value and the screen degenerates into an A/A comparison on one
            # identical binary -- which still reports a median effect, because run-to-run
            # noise is not zero. Measured: ak-gpu-ubatch-up-screen-20260813-s3 passed
            # `-b 512 -ub 1024`, so BOTH arms ran at an effective ubatch of 512, and the
            # screen reported +46.9% from a bimodal sample whose median happened to land
            # on the fast mode (its `batch_up` sibling, equally null, reported +0.59%).
            # Refuse loudly here rather than emit a number that reads as a win.
            anchor_eff = min(result["anchor_batch"], result["anchor_ubatch"])
            cand_eff = min(result["candidate_batch"], result["candidate_ubatch"])
            if (result["anchor_batch"], anchor_eff) == (result["candidate_batch"], cand_eff):
                raise RuntimeError(
                    f"{factor} screen is a null arm: anchor (b={result['anchor_batch']}, "
                    f"ub={result['anchor_ubatch']} -> effective {anchor_eff}) and candidate "
                    f"(b={result['candidate_batch']}, ub={result['candidate_ubatch']} -> "
                    f"effective {cand_eff}) are the same effective configuration after "
                    "llama.cpp's ubatch<=batch clamp; raise candidate_batch with "
                    "candidate_ubatch or drop the factor")
        if factor == "op_offload":
            result["anchor_no_op_offload"] = False
            result["candidate_no_op_offload"] = True
        if factor == "split_row":
            result["anchor_split_mode"] = "layer"
            result["candidate_split_mode"] = "row"
        if factor == "kv_offload":
            result["anchor_no_kv_offload"] = False
            result["candidate_no_kv_offload"] = True
        if factor == "helper_threads_24":
            result["candidate_threads"] = 24
        if factor == "poll_zero":
            result["anchor_poll"] = 50
            result["candidate_poll"] = 0
        return result
    raise RuntimeError(f"unsupported GPU discovery factor: {factor}")


def preflight(args: argparse.Namespace) -> dict:
    model = Path(args.model).resolve()
    anchor_build = Path(args.anchor_build).resolve()
    candidate_build = Path(args.candidate_build).resolve()
    order = tuple(getattr(args, "arm_order_schedule", "anchor,candidate").split(","))
    order_seed = getattr(args, "arm_order_seed_sha256", "0" * 64)
    if (set(order) != {"anchor", "candidate"} or len(order) != 2
            or not isinstance(order_seed, str) or len(order_seed) != 64
            or any(ch not in "0123456789abcdef" for ch in order_seed)):
        raise RuntimeError("GPU discovery arm-order authority is malformed")
    if not model.is_file():
        raise RuntimeError(f"model does not exist: {model}")
    model_size_bytes = model.stat().st_size
    # Admission is computed once by the sealed deployment lease.  This runner
    # is deliberately only a consumer: it never accepts a CLI profile, a size
    # threshold, or fabricated host headroom as an overlap authority.
    transfer = getattr(args, "load_admission_decision", None)
    if not isinstance(transfer, dict):
        raise RuntimeError("GPU discovery runner requires a sealed load-admission decision")
    try:
        gpu_load_admission.validate_decision_receipt(
            transfer,
            expected_policy_version=getattr(args, "load_admission_policy_version"),
            expected_policy_sha256=getattr(args, "load_admission_policy_sha256"),
            expected_policy_file_sha256=getattr(args, "load_admission_policy_file_sha256"),
            expected_effective_context_sha256=getattr(args, "load_admission_effective_context_sha256"))
    except gpu_load_admission.AdmissionPolicyError as exc:
        raise RuntimeError(f"sealed load-admission decision refused: {exc}") from exc
    request = transfer.get("request")
    if (not isinstance(request, dict) or request.get("model_path") != str(model)
            or request.get("model_sha256") != sha256_file(model)
            or request.get("model_bytes") != model_size_bytes
            or request.get("workload") != args.workload
            or request.get("calls_per_arm") != args.calls
            or request.get("device_id") != getattr(args, "device_id", DEVICE_ID)):
        raise RuntimeError("sealed load-admission decision does not bind this runner frame")
    if getattr(args, "device_id", DEVICE_ID) != DEVICE_ID:
        raise RuntimeError("GPU discovery device must be the admitted MI210")
    configured_lock = getattr(args, "inference_window_lock", None)
    lock = (Path(configured_lock) if configured_lock else MODEL_CALL_WINDOW.path).resolve()
    if lock.is_symlink() or not lock.parent.is_dir():
        raise RuntimeError("configured inference-window lock is unsafe")
    anchor_identity = build_identity(anchor_build)
    candidate_identity = build_identity(candidate_build)
    operation_key = getattr(args, "_operation_key", None)
    operation_namespace = None
    if args.factor == "source_patch":
        if (not isinstance(operation_key, str)
                or re.fullmatch(r"[0-9a-f]{64}", operation_key) is None):
            raise RuntimeError(
                "source patch runner lacks its private operation identity")
        operations_root_value = getattr(args, "_operations_root", None)
        repetition = getattr(args, "_operation_repetition", None)
        if not isinstance(operations_root_value, str):
            raise RuntimeError(
                "source patch runner lacks its private operations root")
        operation_namespace = _operation_namespace(
            operations_root=Path(operations_root_value),
            output_root=Path(args.output_dir), operation_key=operation_key,
            repetition=repetition, runtime_graphs=args.runtime_graphs)
        anchor_identity = _sealed_source_build_identity(
            args, arm="anchor", build=anchor_build,
            observed=anchor_identity)
        candidate_identity = _sealed_source_build_identity(
            args, arm="candidate", build=candidate_build,
            observed=candidate_identity)
    factor = factor_spec(
        factor=args.factor, anchor_build=anchor_build, candidate_build=candidate_build,
        anchor_identity=anchor_identity, candidate_identity=candidate_identity)
    prompt_tokens, generation_tokens, recipe, metric = (
        (512, 0, "pp512-ngl99", "prefill_tokens_per_s")
        if args.workload == "prefill_pp512"
        else (0, 128, "tg128-ngl99", "decode_tokens_per_s"))
    runtime_arms = None
    if args.factor == "source_patch":
        if not all(getattr(args, key, None) for key in
                   ("measurement_binary", "common_loader_dir", "anchor_loader_dir", "candidate_loader_dir")):
            raise RuntimeError("source patch requires a sealed shared reward runtime closure")
        measurement = Path(args.measurement_binary).resolve()
        anchor_loader = Path(args.anchor_loader_dir).resolve()
        candidate_loader = Path(args.candidate_loader_dir).resolve()
        common_loader = Path(args.common_loader_dir).resolve()
        def hip_object(path: Path) -> tuple[Path, str]:
            link = path / "libggml-hip.so.0"
            if (not (path / "libggml-hip.so").is_symlink() or not link.is_symlink()
                    or (path / "libggml-hip.so").resolve(strict=True) != link.resolve(strict=True)):
                raise RuntimeError("source patch HIP runtime lacks an exact .so/.so.0 topology")
            resolved = link.resolve(strict=True)
            if resolved.parent != path or resolved.is_symlink() or not resolved.is_file():
                raise RuntimeError("source patch HIP SONAME resolves outside its arm runtime")
            return resolved, sha256_file(resolved)
        if (not measurement.is_file() or not os.access(measurement, os.X_OK) or not common_loader.is_dir()
                or not all(path.is_dir() for path in (anchor_loader, candidate_loader))):
            raise RuntimeError("source patch runtime closure is incomplete")
        shared_sha = sha256_file(measurement)
        _anchor_hip_object, anchor_hip = hip_object(anchor_loader)
        _candidate_hip_object, candidate_hip = hip_object(candidate_loader)
        if anchor_hip == candidate_hip:
            raise RuntimeError("source patch runtime closure requires distinct HIP DSOs")
        if (shared_sha != anchor_identity["binary_sha256"]
                or shared_sha != candidate_identity["binary_sha256"]
                or anchor_hip != anchor_identity["hip_library_sha256"]
                or candidate_hip != candidate_identity["hip_library_sha256"]):
            raise RuntimeError(
                "source patch reward runtime differs from sealed builder artifacts")
        runtime_arms = {"measurement_binary": str(measurement),
                        "measurement_binary_sha256": shared_sha,
                        "anchor_loader_dir": str(anchor_loader),
                        "candidate_loader_dir": str(candidate_loader),
                        "common_loader_dir": str(common_loader),
                        "anchor_hip_sha256": anchor_hip,
                        "candidate_hip_sha256": candidate_hip,
                        "reward_closure": "shared_anchor_binary_per_arm_hip_dso"}
    requested_handshake = getattr(args, "instrument_ready_continue_v1", False)
    instrument_commit = getattr(args, "instrument_ready_continue_commit", None)
    contract_sha256 = getattr(args, "instrument_ready_continue_contract_sha256", None)
    if requested_handshake and (
            not isinstance(instrument_commit, str)
            or instrument_commit != READY_CONTINUE_INSTRUMENT_COMMIT
            or contract_sha256 != READY_CONTINUE_CONTRACT_SHA256
            or runtime_arms is None
            or anchor_identity["source_commit"] != READY_CONTINUE_INSTRUMENT_COMMIT):
        raise RuntimeError(
            "ready/continue requires the sealed 81bf32f11 instrument, exact contract, "
            "and instrument-derived anchor")
    runtime_graphs = getattr(args, "runtime_graphs", "off")
    if runtime_graphs not in {"off", "on"}:
        raise RuntimeError("runtime graph mode must be off or on")
    timed_output_oracle = runtime_arms is not None and runtime_graphs == "off"
    if timed_output_oracle and anchor_identity["source_commit"] != READY_CONTINUE_INSTRUMENT_COMMIT:
        raise RuntimeError(
            "shared source-discovery reward requires the exact sealed 81bf32f11 "
            "timed-output instrument")
    return {
        "schema": "epyc.autokernel.gpu_discovery_preflight.v1",
        "campaign_id": args.campaign_id,
        **({"operation_key": operation_key}
           if args.factor == "source_patch" else {}),
        **({"operation_namespace": operation_namespace}
           if args.factor == "source_patch" else {}),
        "authority": "nonpromotable_candidate_only_discovery",
        "model": str(model),
        "model_sha256": sha256_file(model),
        "model_size_bytes": model_size_bytes,
        "host_transfer": transfer,
        "device_id": DEVICE_ID,
        "inference_window_lock": str(lock),
        "cpu_overlap_policy": ("allowed_discovery_noise" if transfer["mode"] == "cold_overlap"
                               else "cold_serialized_load_window"),
        "promotion_claim": False,
        "non_promotable": True,
        "anchor_build": str(anchor_build),
        "candidate_build": str(candidate_build),
        "anchor_identity": anchor_identity,
        "candidate_identity": candidate_identity,
        "sole_factor": {key: factor[key] for key in ("name", "anchor", "candidate")},
        "anchor_flash_attention": factor["anchor_flash_attention"],
        "candidate_flash_attention": factor["candidate_flash_attention"],
        "anchor_threads": factor.get("anchor_threads", 8),
        "candidate_threads": factor.get("candidate_threads", 8),
        "anchor_batch": factor.get("anchor_batch", 512),
        "candidate_batch": factor.get("candidate_batch", 512),
        "anchor_ubatch": factor.get("anchor_ubatch", 512),
        "candidate_ubatch": factor.get("candidate_ubatch", 512),
        "anchor_mmap": factor.get("anchor_mmap", True),
        "candidate_mmap": factor.get("candidate_mmap", True),
        "anchor_no_op_offload": factor.get("anchor_no_op_offload", False),
        "candidate_no_op_offload": factor.get("candidate_no_op_offload", False),
        "anchor_split_mode": factor.get("anchor_split_mode", "layer"),
        "candidate_split_mode": factor.get("candidate_split_mode", "layer"),
        "anchor_no_kv_offload": factor.get("anchor_no_kv_offload", False),
        "candidate_no_kv_offload": factor.get("candidate_no_kv_offload", False),
        "anchor_poll": factor.get("anchor_poll", 50),
        "candidate_poll": factor.get("candidate_poll", 50),
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "frame": recipe,
        "metric": metric,
        "runtime_graphs": runtime_graphs,
        "metric_contract": ({
            "schema": "epyc.autokernel.serialized_pair_max_metric.v1",
            "scope": "integrity_discovery_only",
            "production_throughput_authority": False,
            "graph_mode": "disabled_for_integrity",
            "scored_sample": "min(first_tokens_per_s,second_tokens_per_s)",
            "serialization_env": {
                "AMD_SERIALIZE_KERNEL": "3", "AMD_SERIALIZE_COPY": "3",
                "GGML_CUDA_DISABLE_GRAPHS": "1"},
        } if timed_output_oracle else {
            "schema": "epyc.autokernel.native_llama_bench_metric.v1",
            "scope": ("target_runtime_graphs_on_direction"
                      if runtime_graphs == "on" else "legacy_nonpromotable_discovery"),
            "production_throughput_authority": runtime_graphs == "on",
            "graph_mode": runtime_graphs,
            **({"graph_environment": {"GGML_CUDA_DISABLE_GRAPHS": "1"}}
               if runtime_graphs == "off" else {
                   "graph_environment": {"GGML_CUDA_DISABLE_GRAPHS": None}}),
        }),
        "invocations": {"anchor": args.calls, "candidate": args.calls},
        "arm_order_schedule": list(order),
        "arm_order_seed_sha256": order_seed,
        "inference_executed": False,
        "runtime_arms": runtime_arms,
        "serialized_readiness": {
            "required": transfer["mode"] == "cold_serialized",
            "proof": "owned_kfd+positive_vram+exact_split_runtime_maps",
            "available": runtime_arms is not None,
            "ready_continue": {"enabled": bool(requested_handshake),
                               "instrument_commit": instrument_commit,
                               "contract_source_sha256": contract_sha256},
        },
        "timed_output_oracle": {
            "enabled": timed_output_oracle,
            "instrument_commit": (READY_CONTINUE_INSTRUMENT_COMMIT
                                  if timed_output_oracle else None),
            "authority": "sealed_81bf_64bit_output_hash_contract",
            "serialization_env": ({
                "AMD_SERIALIZE_KERNEL": "3", "AMD_SERIALIZE_COPY": "3",
                "GGML_CUDA_DISABLE_GRAPHS": "1"}
                if timed_output_oracle else {}),
        },
    }


def _readiness_policy_for_arm(*, sealed: Mapping[str, Any], arm: str,
                              model: Path) -> LoadReadinessPolicy | None:
    """Materialize the only authority permitted to release a cold-load lock."""
    runtime_arms = sealed.get("runtime_arms")
    if runtime_arms is None:
        return None
    if not isinstance(runtime_arms, Mapping):
        raise RuntimeError("sealed shared runtime closure is malformed")
    common = runtime_arms.get("common_loader_dir")
    if not isinstance(common, str):
        raise RuntimeError("sealed shared runtime closure lacks its common loader")
    return LoadReadinessPolicy.from_split_runtime(
        runtime_root=Path(common).resolve().parent, runtime_arm=arm, model=model)


def _prepare_runner_output(root: Path, sealed: Mapping[str, Any]) -> bool:
    """Create a fresh stage root or validate the exact resumable closure."""
    if not root.exists() and not root.is_symlink():
        root.mkdir(parents=True, mode=0o700)
        namespace = sealed.get("operation_namespace")
        if isinstance(namespace, Mapping):
            if not isinstance(sealed, dict):
                raise RuntimeError(
                    "runner preflight cannot bind its stage leaf identity")
            namespace = dict(namespace)
            if namespace.get("output_root") != str(root):
                raise RuntimeError(
                    "runner stage leaf differs from its preflight namespace")
            directories = namespace.get("directories")
            if not isinstance(directories, list) or not directories:
                raise RuntimeError(
                    "runner preflight lacks its operations-root identity")
            sealed["operation_namespace"] = _operation_namespace(
                operations_root=Path(str(directories[0].get("path", ""))),
                output_root=root,
                operation_key=str(namespace.get("operation_key", "")),
                repetition=namespace.get("repetition"),
                runtime_graphs=str(namespace.get("runtime_graphs", "")))
        atomic_json(root / "preflight.json", dict(sealed))
        os.chmod(root / "preflight.json", 0o600)
        return False
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("GPU discovery output is not a real directory")
    root_stat = root.stat()
    if (root_stat.st_uid != os.getuid()
            or stat.S_IMODE(root_stat.st_mode) & 0o077):
        raise RuntimeError("GPU discovery output is not private")
    allowed = {
        "preflight.json", "live-governance.json",
        "ready-continue-anchor", "ready-continue-candidate",
        "supervisor-anchor", "supervisor-candidate",
        "process-anchor", "process-candidate",
        "process-anchor-refusal.json", "process-candidate-refusal.json",
        "process-anchor-infrastructure-ambiguity.json",
        "process-candidate-infrastructure-ambiguity.json",
        "correctness-divergence.json",
    }
    entries = tuple(root.iterdir())
    if any(entry.name not in allowed or entry.is_symlink()
           or entry.stat().st_uid != os.getuid()
           or (entry.is_file() and (
               not stat.S_ISREG(entry.stat().st_mode)
               or entry.stat().st_nlink != 1))
           or (entry.is_dir() and not stat.S_ISDIR(entry.stat().st_mode))
           for entry in entries):
        raise RuntimeError("GPU discovery resumable output closure changed")
    try:
        preflight_bytes, _binding = _capture_file(
            root / "preflight.json", "resumable preflight")
        existing = json.loads(preflight_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("GPU discovery resumable preflight is unavailable") from exc
    if not isinstance(existing, dict):
        raise RuntimeError("GPU discovery resumable preflight identity changed")
    supplied = dict(sealed)
    stored_namespace = existing.pop("operation_namespace", None)
    supplied_namespace = supplied.pop("operation_namespace", None)
    if existing != supplied:
        raise RuntimeError("GPU discovery resumable preflight identity changed")
    try:
        if (not isinstance(stored_namespace, Mapping)
                or not isinstance(supplied_namespace, Mapping)):
            raise RuntimeError("runner operation namespace is absent")
        _revalidate_operation_namespace(
            stored_namespace, output_root=root,
            operation_key=str(stored_namespace.get("operation_key", "")),
            runtime_graphs=str(stored_namespace.get("runtime_graphs", "")))
        _revalidate_operation_namespace(
            supplied_namespace, output_root=root,
            operation_key=str(supplied_namespace.get("operation_key", "")),
            runtime_graphs=str(supplied_namespace.get("runtime_graphs", "")))
    except RuntimeError as exc:
        raise RuntimeError(
            "GPU discovery resumable preflight identity changed") from exc
    if not isinstance(sealed, dict):
        raise RuntimeError("GPU discovery resumable preflight cannot be restored")
    # All later process and terminal receipts keep the original namespace
    # hash.  The freshly sampled namespace is only a revalidation witness.
    sealed["operation_namespace"] = dict(stored_namespace)
    return True


def validate_resumable_output(root: Path, *, graph_mode: str) -> bool:
    """Conservatively recognize a process-complete, result-incomplete stage."""
    try:
        if root.is_symlink() or not root.is_dir() or graph_mode not in {"off", "on"}:
            return False
        preflight_path = root / "preflight.json"
        if preflight_path.is_symlink() or not preflight_path.is_file():
            return False
        preflight_bytes, _preflight_binding = _capture_file(
            preflight_path, "resumable preflight")
        preflight = json.loads(preflight_bytes)
        if (not isinstance(preflight, dict)
                or preflight.get("runtime_graphs") != graph_mode):
            return False
        completed = 0
        for arm in preflight.get("arm_order_schedule", []):
            receipt_root = root / f"process-{arm}"
            if not receipt_root.exists() and not receipt_root.is_symlink():
                break
            # Identity is embedded and self-hashed.  Full current-argv
            # validation occurs on re-entry before reuse.
            receipt_path = receipt_root / "receipt.json"
            receipt_bytes, _receipt_binding = _capture_file(
                receipt_path, "resumable process receipt")
            raw = json.loads(receipt_bytes)
            unsigned = {key: value for key, value in raw.items()
                        if key != "receipt_sha256"}
            identity = raw.get("identity")
            if (raw.get("schema") != SCHEMA_PROCESS_RECEIPT
                    or raw.get("receipt_sha256") != schemas.content_hash(unsigned)
                    or not isinstance(identity, Mapping)
                    or identity.get("runtime_graphs") != graph_mode
                    or identity.get("runtime_arm") != arm):
                return False
            _load_process_capture(receipt_root, identity=identity)
            completed += 1
        return completed >= 1 and not (root / "result.json").exists()
    except (OSError, ValueError, TypeError, KeyError, RuntimeError,
            json.JSONDecodeError):
        return False


def _invocation_seed(*, base_seed: int, repetitions: int, arm: str,
                     timed_output_oracle_enabled: bool,
                     runtime_graphs: str) -> int:
    """Select an arm seed without contradicting a cross-arm output oracle."""
    if arm not in {"anchor", "candidate"}:
        raise RuntimeError("runner arm is invalid")
    if runtime_graphs not in {"off", "on"}:
        raise RuntimeError("runner graph mode is invalid")
    same_input_required = timed_output_oracle_enabled or runtime_graphs == "on"
    return (base_seed if same_input_required or arm == "anchor"
            else base_seed + repetitions)


def run(args: argparse.Namespace) -> dict:
    sealed = preflight(args)
    started_at = utc_now()
    out = Path(storage.assert_not_scratch(args.output_dir, what="GPU discovery output"))
    resumed = _prepare_runner_output(out, sealed)
    if sealed.get("sole_factor", {}).get("name") == "source_patch":
        _revalidate_operation_namespace(
            sealed["operation_namespace"], output_root=out,
            operation_key=sealed["operation_key"],
            runtime_graphs=sealed["runtime_graphs"])
    model = Path(sealed["model"])
    anchor_build = Path(sealed["anchor_build"])
    candidate_build = Path(sealed["candidate_build"])
    anchor_identity = sealed["anchor_identity"]
    candidate_identity = sealed["candidate_identity"]
    sole_factor = sealed["sole_factor"]
    anchor_readiness = _readiness_policy_for_arm(
        sealed=sealed, arm="anchor", model=model)
    candidate_readiness = _readiness_policy_for_arm(
        sealed=sealed, arm="candidate", model=model)
    handshake_enabled = bool(sealed["serialized_readiness"]["ready_continue"]["enabled"])
    timed_output_oracle_enabled = bool(sealed.get("timed_output_oracle", {}).get("enabled"))
    anchor_handshake = (ReadyContinueHandshake.create(
        root=out / "ready-continue-anchor", decision=sealed["host_transfer"],
        policy=anchor_readiness, arm="anchor", seed=args.seed, repetitions=args.calls,
        resume_existing=resumed)
        if handshake_enabled and anchor_readiness is not None else None)
    candidate_handshake = (ReadyContinueHandshake.create(
        root=out / "ready-continue-candidate", decision=sealed["host_transfer"],
        policy=candidate_readiness, arm="candidate", seed=args.seed,
        repetitions=args.calls, resume_existing=resumed)
        if handshake_enabled and candidate_readiness is not None else None)
    if _kfd_pids():
        raise RuntimeError("MI210 already has KFD users")
    baseline_vram = int(VRAM_USED.read_text(encoding="utf-8").strip())
    purpose = ("AutoKernel GPU candidate-only discovery "
               f"{sole_factor['name']} {sole_factor['anchor']}->{sole_factor['candidate']}")
    cpu_journal = cpu_region_claim.RegionClaimJournal(args.cpu_claim_journal)
    gpu_journal = device_claim.ClaimJournal(args.device_claim_journal)
    claim_acquirer = getattr(args, "_device_claim_acquirer",
                             device_claim.acquire_device_claim)
    if not callable(claim_acquirer):
        raise RuntimeError("device claim acquirer is not callable")
    gpu = None
    sampler = None
    live_governance = None
    borrowed_phase_end = None
    live_governance_path = out / "live-governance.json"
    try:
        gpu = claim_acquirer(
            DEVICE_ID, purpose=purpose, campaign_id=args.campaign_id,
            journal=gpu_journal, timeout_s=0, max_hold_s=300)
        claim_mode = ("borrowed_outer_reservation"
                      if getattr(gpu, "borrowed_outer_reservation", False)
                      else "direct_device_claim")
        live_governance = {
            "schema": SCHEMA_LIVE_GOVERNANCE,
            "status": "active",
            "campaign_id": args.campaign_id,
            "runner_pid": os.getpid(),
            "authority": "nonpromotable_candidate_only_discovery",
            "cpu_overlap_policy": sealed["cpu_overlap_policy"],
            "model": sealed["model"],
            "model_sha256": sealed["model_sha256"],
            "model_size_bytes": sealed["model_size_bytes"],
            "site_load_decision": sealed["host_transfer"],
            "promotion_claim": False,
            "non_promotable": True,
            "preflight_sha256": schemas.content_hash(sealed),
            "device_claim_open": gpu.receipt().to_dict(),
            "device_claim_mode": claim_mode,
            "started_at": started_at,
        }
        atomic_json(live_governance_path, live_governance)
        os.chmod(live_governance_path, 0o600)
        sampler = device_sampler.RocmSmiSampler(device_index=0, interval_s=0.250).start()
        arm_order = tuple(sealed.get("arm_order_schedule", [
            "anchor", "candidate"]))
        if set(arm_order) != {"anchor", "candidate"} or len(arm_order) != 2:
            raise RuntimeError("arm order must contain anchor and candidate exactly once")

        def run_arm(arm: str) -> list[dict]:
            if sealed.get("sole_factor", {}).get("name") == "source_patch":
                _revalidate_operation_namespace(
                    sealed["operation_namespace"], output_root=out,
                    operation_key=sealed["operation_key"],
                    runtime_graphs=sealed["runtime_graphs"])
            anchor = arm == "anchor"
            prefix = "anchor" if anchor else "candidate"
            identity = anchor_identity if anchor else candidate_identity
            readiness = anchor_readiness if anchor else candidate_readiness
            handshake = anchor_handshake if anchor else candidate_handshake
            return [invoke(
                build=anchor_build if anchor else candidate_build,
                model=model, seed=_invocation_seed(
                    base_seed=args.seed, repetitions=args.calls, arm=arm,
                    timed_output_oracle_enabled=timed_output_oracle_enabled,
                    runtime_graphs=str(sealed.get("runtime_graphs", "off"))),
                expected_source_commit=(None if sealed["runtime_arms"]
                                        else identity["source_commit"]),
                baseline_vram=baseline_vram,
                flash_attention=sealed[f"{prefix}_flash_attention"],
                prompt_tokens=sealed["prompt_tokens"],
                generation_tokens=sealed["generation_tokens"],
                threads=sealed[f"{prefix}_threads"],
                ubatch=sealed[f"{prefix}_ubatch"], batch=sealed[f"{prefix}_batch"],
                mmap=sealed[f"{prefix}_mmap"],
                no_op_offload=sealed[f"{prefix}_no_op_offload"],
                split_mode=sealed[f"{prefix}_split_mode"],
                no_kv_offload=sealed[f"{prefix}_no_kv_offload"],
                poll=sealed[f"{prefix}_poll"],
                campaign_id=args.campaign_id, cpu_journal=cpu_journal,
                sealed_load_decision=sealed["host_transfer"],
                inference_window_lock=Path(sealed["inference_window_lock"]),
                reward_binary=(Path(sealed["runtime_arms"]["measurement_binary"])
                               if sealed["runtime_arms"] else None),
                hip_library_dir=(Path(sealed["runtime_arms"][f"{prefix}_loader_dir"])
                                 if sealed["runtime_arms"] else None),
                common_loader_dir=(Path(sealed["runtime_arms"]["common_loader_dir"])
                                   if sealed["runtime_arms"] else None),
                runtime_arm=(arm if sealed["runtime_arms"] else None),
                repetitions=args.calls,
                timed_output_oracle=timed_output_oracle_enabled,
                runtime_graphs=str(sealed.get("runtime_graphs", "off")),
                load_readiness_policy=readiness,
                ready_continue_handshake=handshake,
                supervisor_root=out / f"supervisor-{arm}",
                process_receipt_root=out / f"process-{arm}",
                process_context={
                    "campaign_id": args.campaign_id,
                    **({"operation_key": sealed["operation_key"]}
                       if "operation_key" in sealed else {}),
                    **({"operation_namespace_sha256": schemas.content_hash(
                            sealed["operation_namespace"])}
                       if "operation_namespace" in sealed else {}),
                    "preflight_sha256": schemas.content_hash(sealed),
                    "arm": arm,
                    "workload": getattr(args, "workload", sealed["frame"]),
                    "metric": sealed["metric"],
                    "runtime_graphs": str(sealed.get("runtime_graphs", "off")),
                    "prompt_tokens": sealed["prompt_tokens"],
                    "generation_tokens": sealed["generation_tokens"],
                    "tokens_per_repetition": (
                        sealed["prompt_tokens"] + sealed["generation_tokens"]),
                },
                process_resource_context={
                    "device_claim_open": gpu.receipt().to_dict(),
                    "device_claim_mode": claim_mode,
                },
                after_process_checkpoint=getattr(
                    args, "_after_process_checkpoint", None))]

        arm_runs = {arm: run_arm(arm) for arm in arm_order}
        anchor_runs = arm_runs["anchor"]
        candidate_runs = arm_runs["candidate"]
        timed_output_oracle = None
        graphs_on_output_oracle = None
        if timed_output_oracle_enabled:
            try:
                timed_output_oracle = _validate_cross_arm_timed_outputs(
                    anchor_runs[0], candidate_runs[0])
            except _CrossArmOutputDivergence:
                raise _seal_candidate_correctness_divergence(
                    out, anchor=anchor_runs[0], candidate=candidate_runs[0],
                    runtime_graphs=str(sealed.get("runtime_graphs", "off")),
                    campaign_id=args.campaign_id,
                    operation_key=str(sealed.get("operation_key", "")),
                    operation_namespace=sealed.get("operation_namespace", {}),
                    anchor_identity=anchor_identity,
                    candidate_identity=candidate_identity)
        if sealed.get("runtime_graphs") == "on":
            graphs_on_output_oracle = _validate_cross_arm_graphs_on_outputs(
                anchor_runs[0], candidate_runs[0])
        bank_body = {
            "schema": SCHEMA_BANK, "campaign_id": args.campaign_id,
            "status": "complete", "started_at": started_at, "ended_at": utc_now(),
            "authority": "nonpromotable_candidate_only_discovery",
            "frame": {"backend": "llama_gpu", "recipe": sealed["frame"],
                      "metric": sealed["metric"], "metric_direction": "higher_better",
                      "metric_contract": sealed.get("metric_contract", {
                          "schema": "epyc.autokernel.native_llama_bench_metric.v1",
                          "scope": "legacy_nonpromotable_discovery",
                          "production_throughput_authority": False,
                      }),
                      "n_prompt": sealed["prompt_tokens"],
                      "n_gen": sealed["generation_tokens"],
                      "model": str(model), "model_sha256": sha256_file(model),
                      "source_commit": candidate_identity["source_commit"], "cpu_list": CPU_LIST,
                      "device": "AMD Instinct MI210", "architecture": "gfx90a"},
            "sole_factor": sole_factor,
            "anchor_invocations": args.calls,
            "anchor_identity": anchor_identity,
            "candidate_identity": candidate_identity,
            "anchor_processes": 1,
            "arm_order_schedule": list(arm_order),
            "arm_order_seed_sha256": sealed.get("arm_order_seed_sha256", "0" * 64),
            "anchor_samples": [sample for run in anchor_runs for sample in run["samples"]],
            "anchor_runs": anchor_runs,
            **({"timed_output_oracle": timed_output_oracle}
               if timed_output_oracle is not None else {}),
            **({"graphs_on_output_oracle": graphs_on_output_oracle}
               if graphs_on_output_oracle is not None else {}),
        }
        bank = gpu_beliefs.attach_baseline_beliefs(
            bank_body, producer_path=Path(__file__).resolve())
        atomic_json(out / "baseline-bank.json", bank)
        center = (float(anchor_runs[0]["metric"])
                  if bank["frame"]["metric_contract"]["schema"] ==
                  "epyc.autokernel.serialized_pair_max_metric.v1"
                  else sum(bank["anchor_samples"]) / len(bank["anchor_samples"]))
        values = [sample for run in candidate_runs for sample in run["samples"]]
        effects = [(value - center) / center for value in values]
        numeric = sampler.stop().to_dict()
        sampler = None
        if claim_mode == "borrowed_outer_reservation":
            borrowed_phase_end = gpu.release()
            if hasattr(borrowed_phase_end, "to_dict"):
                borrowed_phase_end = borrowed_phase_end.to_dict()
            if (not isinstance(borrowed_phase_end, Mapping)
                    or borrowed_phase_end.get("schema") !=
                    "epyc.autokernel.borrowed_device_claim_phase.v1"
                    or borrowed_phase_end.get("mode") != "borrowed_outer_reservation"
                    or borrowed_phase_end.get("outer_claim_id") !=
                    gpu.receipt().to_dict().get("claim_id")
                    or borrowed_phase_end.get("physical_release") is not False
                    or "released_at" in borrowed_phase_end):
                raise RuntimeError("borrowed throughput phase end is malformed")
        result_body = {
            "schema": SCHEMA_RESULT, "campaign_id": args.campaign_id,
            "status": "complete", "started_at": started_at, "ended_at": utc_now(),
            "authority": "nonpromotable_candidate_only_discovery",
            "state": "decided", "ok": True, "non_promotable": True,
            "nomination": "top_k_candidate_only_not_a_keep",
            "baseline_sha256": bank["baseline_sha256"],
            "anchor_invocations": args.calls, "candidate_invocations": args.calls,
            "anchor_processes": 1, "candidate_processes": 1,
            "arm_order_schedule": list(arm_order),
            "arm_order_seed_sha256": sealed.get("arm_order_seed_sha256", "0" * 64),
            "baseline_center": center, "candidate_samples": values,
            "relative_effects": effects, "median_relative": median(effects),
            "host_noise_policy": "ordinary_host_activity_recorded_not_blocking",
            "cpu_overlap_policy": sealed["cpu_overlap_policy"],
            "model_size_bytes": model.stat().st_size,
            "site_load_decision": sealed["host_transfer"],
            "promotion_claim": False,
            "runtime_graphs": sealed.get("runtime_graphs", "off"),
            "frame": bank["frame"], "sole_factor": bank["sole_factor"],
            "candidate_identity": bank["candidate_identity"],
            "candidate_runs": candidate_runs, "device_sampling": numeric,
            **({"timed_output_oracle": timed_output_oracle}
               if timed_output_oracle is not None else {}),
            **({"graphs_on_output_oracle": graphs_on_output_oracle}
               if graphs_on_output_oracle is not None else {}),
            "hip_residency_proved": all(run["hip_residency_proved"]
                                         for run in anchor_runs + candidate_runs),
            "cpu_coverage_windows": [
                run["cpu_coverage"] for run in anchor_runs + candidate_runs],
            "device_claim_open": gpu.receipt().to_dict(),
            "device_claim_mode": claim_mode,
            **({"device_claim_borrowed_phase_end": dict(borrowed_phase_end)}
               if borrowed_phase_end is not None else {}),
        }
        result = gpu_beliefs.attach_result_beliefs(
            result_body, bank=bank, producer_path=Path(__file__).resolve())
        atomic_json(out / "result.json", result)
        # A derived operator view, kept separate from the strict terminal
        # campaign contract.  The immutable result above is already durable, so
        # an export failure must not erase or reclassify the measurement.
        try:
            autokernel_progression.export_progression()
        except Exception as exc:
            print(f"WARNING: GPU result is durable but progression export failed: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return result
    finally:
        primary_active = sys.exc_info()[0] is not None
        sampler_error: BaseException | None = None
        if sampler is not None:
            try:
                sampler.stop()
            except BaseException as exc:
                sampler_error = exc
        if gpu is not None:
            ended = (borrowed_phase_end
                     if borrowed_phase_end is not None else gpu.release())
            if hasattr(ended, "to_dict"):
                ended = ended.to_dict()
            if not isinstance(ended, Mapping):
                raise RuntimeError("device claim end did not return a typed receipt")
            if live_governance is not None:
                terminal = {
                    **live_governance,
                    "ended_at": utc_now(),
                }
                if claim_mode == "borrowed_outer_reservation":
                    terminal.update(
                        status="borrowed_phase_ended",
                        device_claim_borrowed_phase_end=dict(ended))
                else:
                    terminal.update(
                        status="released", device_claim_released=dict(ended))
                atomic_json(live_governance_path, terminal)
                os.chmod(live_governance_path, 0o600)
        if sampler_error is not None and not primary_active:
            raise sampler_error


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--anchor-build", required=True)
    result.add_argument("--candidate-build", required=True)
    result.add_argument("--model", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--campaign-id", required=True)
    result.add_argument("--factor", choices=("mmq_mfma", "flash_attention", "rocwmma_fattn",
                                             "source_patch",
                                             "hip_graphs", "helper_threads", "helper_threads_12",
                                             "helper_threads_16", "helper_threads_24", "batch",
                                             "batch_up", "ubatch", "ubatch_up", "mmap",
                                             "op_offload", "split_row", "kv_offload", "poll_zero"),
                        default="mmq_mfma")
    result.add_argument("--preflight-only", action="store_true")
    result.add_argument("--preflight-output")
    result.add_argument("--seed", type=int, default=8613)
    result.add_argument("--calls", type=int, choices=(3, 5, 9), default=3,
                        help="fresh invocations per arm (discovery evidence only)")
    result.add_argument("--arm-order-schedule",
                        choices=("anchor,candidate", "candidate,anchor"),
                        default="anchor,candidate")
    result.add_argument("--arm-order-seed-sha256", default="0" * 64)
    result.add_argument("--workload", choices=("prefill_pp512", "decode_tg128"),
                        default="prefill_pp512")
    result.add_argument("--inference-window-lock")
    result.add_argument("--device-id", default=DEVICE_ID)
    result.add_argument("--measurement-binary")
    result.add_argument("--common-loader-dir")
    result.add_argument("--anchor-loader-dir")
    result.add_argument("--candidate-loader-dir")
    result.add_argument("--instrument-ready-continue-v1", action="store_true")
    result.add_argument("--instrument-ready-continue-commit")
    result.add_argument("--instrument-ready-continue-contract-sha256")
    result.add_argument("--runtime-graphs", choices=("off", "on"), default="off")
    # These four paths/digests are emitted only by the sealed deployment
    # runner binding.  Keep the path carriers separate from the typed fields
    # consumed by ``preflight``: in-process governed callers install the
    # already-validated decision object, while the standalone CLI hydrates the
    # same object from these exact byte-bound carriers before crossing the
    # execution boundary.
    result.add_argument("--load-admission-decision",
                        dest="load_admission_decision_path")
    result.add_argument("--load-admission-policy",
                        dest="load_admission_policy_path")
    result.add_argument("--load-admission-policy-sha256",
                        dest="load_admission_policy_file_sha256")
    result.add_argument("--effective-context-sha256",
                        dest="load_admission_effective_context_sha256")
    result.add_argument("--cpu-claim-journal", default="/mnt/raid0/llm/ak-claims/region.jsonl")
    result.add_argument("--device-claim-journal", default="/mnt/raid0/llm/ak-claims/device.jsonl")
    return result


def _hydrate_cli_load_admission(args: argparse.Namespace) -> argparse.Namespace:
    """Turn the exact CLI carriers into the typed preflight authority.

    The runner never derives an admission decision.  It only reloads and
    recursively validates the lease-owned receipt and its policy bytes.
    """
    decision_raw = getattr(args, "load_admission_decision_path", None)
    policy_raw = getattr(args, "load_admission_policy_path", None)
    policy_file_sha = getattr(args, "load_admission_policy_file_sha256", None)
    effective_sha = getattr(args, "load_admission_effective_context_sha256", None)
    if not all(isinstance(value, str) and value for value in (
            decision_raw, policy_raw, policy_file_sha, effective_sha)):
        raise RuntimeError(
            "GPU discovery runner requires the complete sealed load-admission CLI frame")
    decision_path = Path(decision_raw)
    if (not decision_path.is_absolute() or decision_path.is_symlink()
            or decision_path.resolve(strict=True) != decision_path
            or not decision_path.is_file()):
        raise RuntimeError("load-admission decision carrier is unsafe")
    try:
        raw = decision_path.read_bytes()
        if len(raw) > gpu_load_admission.MAX_POLICY_BYTES:
            raise RuntimeError("load-admission decision carrier is oversized")
        decision = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("load-admission decision carrier is unreadable") from exc
    if not isinstance(decision, dict):
        raise RuntimeError("load-admission decision carrier is not an object")
    try:
        corpus = gpu_load_admission.load_policy_corpus(
            Path(policy_raw), expected_file_sha256=policy_file_sha)
        gpu_load_admission.validate_decision_receipt(
            decision, expected_policy_version=corpus.version,
            expected_policy_sha256=corpus.policy_sha256,
            expected_policy_file_sha256=corpus.file_sha256,
            expected_effective_context_sha256=effective_sha)
    except gpu_load_admission.AdmissionPolicyError as exc:
        raise RuntimeError(f"sealed load-admission CLI frame refused: {exc}") from exc
    args.load_admission_decision = decision
    args.load_admission_policy_version = corpus.version
    args.load_admission_policy_sha256 = corpus.policy_sha256
    args.load_admission_policy_file_sha256 = corpus.file_sha256
    args.load_admission_effective_context_sha256 = effective_sha
    return args


def main() -> int:
    try:
        args = _hydrate_cli_load_admission(parser().parse_args())
        payload = preflight(args) if args.preflight_only else run(args)
        if args.preflight_output:
            atomic_json(Path(args.preflight_output), payload)
    except Exception as exc:
        print(f"GPU discovery REFUSED: {type(exc).__name__}: {exc}", file=os.sys.stderr)
        return 1
    if args.preflight_only:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps({key: payload[key] for key in (
            "state", "baseline_center", "candidate_samples", "median_relative",
            "hip_residency_proved", "result_sha256")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
