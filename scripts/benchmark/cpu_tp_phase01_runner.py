#!/usr/bin/env python3
"""Hardware-free preflight and evidence tooling for CPU NUMA TP Phase 0/1.

This module deliberately has no benchmark launcher.  It can attest the local
host, production v9, the mechanism model, the region-lock state, and the
counter-tool surface; parse already-captured perf/uProf numeric output; and
materialize deterministic schedules.  Any request for execution is refused
unless an external human-ratification receipt binds the exact protocol,
runner, and schema hashes, and is still refused by this validate-only version.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
import stat
import struct
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL_ID = "P-BENCH-NUMA-TP-1"
PREFLIGHT_SCHEMA = "epyc.cpu_tp.phase01_preflight.v1"
PANEL_SCHEMA = "epyc.cpu_tp.phase01_counter_panel.v1"
RATIFICATION_SCHEMA = "epyc.measurement.protocol_ratification.v1"
N25_SCHEMA = "epyc.cpu_tp.n25_attestation.v1"
SCHEMA_DIR = Path(__file__).with_name("cpu_tp_phase01_schemas")

EXPECTED_PRODUCTION_BRANCH = "production-consolidated-v9"
EXPECTED_PRODUCTION_COMMIT = "0db32c06e3e550065b78311a6031ef3dd2c4f27c"
EXPECTED_BINARY_VERSION = 10125
EXPECTED_MODEL_NAME = "Qwen2.5-Coder-32B-Instruct-Q8_0.gguf"
EXPECTED_MODEL_BYTES = 34_820_885_184
DEFAULT_PRODUCTION_ROOT = Path("/mnt/raid0/llm/llama.cpp")
DEFAULT_LLAMA_SERVER = DEFAULT_PRODUCTION_ROOT / "build/bin/llama-server"
DEFAULT_MODEL = Path(
    "/mnt/raid0/llm/models/lmstudio-community/"
    "Qwen2.5-Coder-32B-Instruct-GGUF/"
    "Qwen2.5-Coder-32B-Instruct-Q8_0.gguf"
)
DEFAULT_REGION_LOCK = Path("/workspace/repos/epyc-orchestrator/scripts/region-lock")
DEFAULT_N25_HANDOFF = Path(
    "/workspace/handoffs/active/numa-topology-cutover-resume-20260730.md"
)
UPTIME_LIMIT_SECONDS = 7 * 24 * 60 * 60
CAMPAIGN_SEED = 2026082001

PERF_PANELS: dict[str, tuple[str, ...]] = {
    "C0": ("cycles", "instructions", "cache-references", "cache-misses"),
    "C1": (
        "ls_dmnd_fills_from_sys.dram_io_all",
        "ls_hw_pf_dc_fills.dram_io_all",
        "cycles",
        "instructions",
    ),
    "C2": (
        "ls_dmnd_fills_from_sys.dram_io_near",
        "ls_dmnd_fills_from_sys.dram_io_far",
        "cycles",
        "instructions",
    ),
}
OPTIONAL_PERF_PANEL = (
    "fp_ops_retired_by_type.vector_mac",
    "fp_ops_retired_by_type.vector_all",
    "fp_ops_retired_by_type.scalar_all",
)
UPROF_GROUPS = ("memory", "ipc", "pipeline_util", "dc", "l3", "ccm_bw")

MODEL_KEYS = (
    "general.architecture",
    "qwen2.block_count",
    "qwen2.embedding_length",
    "qwen2.attention.head_count",
    "qwen2.attention.head_count_kv",
    "qwen2.feed_forward_length",
)
EXPECTED_MODEL_METADATA: dict[str, Any] = {
    "general.architecture": "qwen2",
    "qwen2.block_count": 64,
    "qwen2.embedding_length": 5120,
    "qwen2.attention.head_count": 40,
    "qwen2.attention.head_count_kv": 8,
    "qwen2.feed_forward_length": 27648,
}

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_VERSION = re.compile(r"version:\s*(\d+)\s*\(([0-9a-f]+)\)")


class CpuTpError(RuntimeError):
    """A fail-closed validation or evidence error."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise CpuTpError(f"not a one-link regular file: {path}")
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
        def identity(item: os.stat_result) -> tuple[int, int, int, int, int]:
            return (
                item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns,
                item.st_nlink,
            )
        if identity(before) != identity(after) or total != after.st_size:
            raise CpuTpError(f"file changed while hashing: {path}")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def file_identity(path: Path, *, hash_bytes: bool = True) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    facts = resolved.stat()
    if not stat.S_ISREG(facts.st_mode) or facts.st_nlink != 1:
        raise CpuTpError(f"authority file is not a one-link regular file: {path}")
    result: dict[str, Any] = {
        "path": str(resolved),
        "size": facts.st_size,
        "uid": facts.st_uid,
        "mode": stat.S_IMODE(facts.st_mode),
        "device": facts.st_dev,
        "inode": facts.st_ino,
    }
    if hash_bytes:
        result["sha256"] = sha256_file(resolved)
    return result


def self_hashed(body: Mapping[str, Any]) -> dict[str, Any]:
    if "receipt_sha256" in body:
        raise CpuTpError("receipt body already contains receipt_sha256")
    result = dict(body)
    result["receipt_sha256"] = content_sha256(result)
    return result


def write_receipt(path: Path, body: Mapping[str, Any]) -> dict[str, Any]:
    payload = self_hashed(body)
    raw = canonical_bytes(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise CpuTpError("receipt write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return payload


def read_canonical_receipt(path: Path, *, expected_schema: str) -> dict[str, Any]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8", "strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CpuTpError(f"receipt is not JSON: {path}") from exc
    if not isinstance(value, dict) or value.get("schema") != expected_schema:
        raise CpuTpError(f"receipt schema mismatch: {path}")
    if raw != canonical_bytes(value) + b"\n":
        raise CpuTpError(f"receipt is not canonically encoded: {path}")
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if value.get("receipt_sha256") != content_sha256(unsigned):
        raise CpuTpError(f"receipt self-hash mismatch: {path}")
    return value


def run_capture(
    argv: Sequence[str], *, timeout: float = 20.0,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(argv), capture_output=True, text=True, check=False,
            timeout=timeout, env=dict(env) if env is not None else None,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "argv": list(argv), "returncode": None, "ok": False,
            "stdout": "", "stderr": str(exc),
        }
    return {
        "argv": list(argv), "returncode": completed.returncode,
        "ok": completed.returncode == 0, "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _compact_command(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "argv": result["argv"], "returncode": result["returncode"],
        "ok": result["ok"],
        "stdout_sha256": hashlib.sha256(str(result["stdout"]).encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(str(result["stderr"]).encode()).hexdigest(),
        "stdout": result["stdout"], "stderr": result["stderr"],
    }


def _read_exact(handle: io.BufferedReader, count: int) -> bytes:
    raw = handle.read(count)
    if len(raw) != count:
        raise CpuTpError("GGUF header is truncated")
    return raw


def _u32(handle: io.BufferedReader) -> int:
    return struct.unpack("<I", _read_exact(handle, 4))[0]


def _u64(handle: io.BufferedReader) -> int:
    return struct.unpack("<Q", _read_exact(handle, 8))[0]


def _gguf_string(handle: io.BufferedReader) -> str:
    length = _u64(handle)
    if length > 16 * 1024 * 1024:
        raise CpuTpError("GGUF string exceeds metadata ceiling")
    return _read_exact(handle, length).decode("utf-8", "strict")


_GGUF_SCALAR = {0: (1, "B"), 1: (1, "b"), 2: (2, "H"), 3: (2, "h"),
                4: (4, "I"), 5: (4, "i"), 6: (4, "f"), 7: (1, "?"),
                10: (8, "Q"), 11: (8, "q"), 12: (8, "d")}


def _gguf_value(handle: io.BufferedReader, value_type: int, *, retain: bool) -> Any:
    if value_type == 8:
        value = _gguf_string(handle)
        return value if retain else None
    if value_type == 9:
        element_type = _u32(handle)
        count = _u64(handle)
        if count > 10_000_000:
            raise CpuTpError("GGUF metadata array exceeds element ceiling")
        values = [_gguf_value(handle, element_type, retain=retain) for _ in range(count)]
        return values if retain else None
    descriptor = _GGUF_SCALAR.get(value_type)
    if descriptor is None:
        raise CpuTpError(f"unsupported GGUF metadata type {value_type}")
    size, format_char = descriptor
    value = struct.unpack("<" + format_char, _read_exact(handle, size))[0]
    return value if retain else None


def read_model_metadata(path: Path) -> dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise CpuTpError("model must be a one-link regular GGUF")
            magic = _read_exact(handle, 4)
            if magic != b"GGUF":
                raise CpuTpError("model does not have GGUF magic")
            version = _u32(handle)
            if version not in (2, 3):
                raise CpuTpError(f"unsupported GGUF version {version}")
            tensor_count, kv_count = _u64(handle), _u64(handle)
            if kv_count > 1_000_000:
                raise CpuTpError("GGUF metadata count exceeds ceiling")
            retained: dict[str, Any] = {}
            for _ in range(kv_count):
                key = _gguf_string(handle)
                value_type = _u32(handle)
                retain = key in MODEL_KEYS
                value = _gguf_value(handle, value_type, retain=retain)
                if retain:
                    retained[key] = value
            after = os.fstat(descriptor)
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                    after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                raise CpuTpError("model changed while reading metadata")
    finally:
        os.close(descriptor)
    return {
        "gguf_version": version, "tensor_count": tensor_count,
        "kv_count": kv_count, "selected": retained,
    }


def model_attestation(path: Path, *, hash_model: bool) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    try:
        identity = file_identity(path, hash_bytes=hash_model)
        metadata = read_model_metadata(path)
    except (OSError, CpuTpError) as exc:
        return {"path": str(path), "available": False, "error": str(exc)}, [
            "primary_model_unavailable"
        ]
    if path.name != EXPECTED_MODEL_NAME:
        reasons.append("primary_model_name_mismatch")
    if identity["size"] != EXPECTED_MODEL_BYTES:
        reasons.append("primary_model_size_mismatch")
    if not hash_model:
        reasons.append("primary_model_sha256_not_collected")
    if metadata["selected"] != EXPECTED_MODEL_METADATA:
        reasons.append("primary_model_metadata_mismatch")
    dimensions = {key: metadata["selected"].get(key) for key in MODEL_KEYS[1:]}
    divisible = all(isinstance(value, int) and value % 4 == 0 for value in dimensions.values())
    quant_aligned = all(
        isinstance(dimensions.get(key), int) and dimensions[key] // 4 % 32 == 0
        for key in ("qwen2.embedding_length", "qwen2.feed_forward_length")
    )
    if not divisible or not quant_aligned:
        reasons.append("primary_model_tp4_alignment_failed")
    return {
        "available": True, "identity": identity, "metadata": metadata,
        "tp_degree": 4, "dimensions_divisible": divisible,
        "local_quant_dimensions_block32_aligned": quant_aligned,
    }, reasons


def parse_lscpu_summary(text: str) -> dict[str, str]:
    try:
        payload = json.loads(text)
        rows = payload["lscpu"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise CpuTpError("lscpu summary is malformed") from exc
    result: dict[str, str] = {}
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("field"), str):
            result[row["field"].rstrip(":")] = str(row.get("data", ""))
    return result


def parse_lscpu_extended(text: str) -> list[dict[str, Any]]:
    try:
        rows = json.loads(text)["cpus"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise CpuTpError("lscpu extended topology is malformed") from exc
    if not isinstance(rows, list) or not rows:
        raise CpuTpError("lscpu extended topology has no CPUs")
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"cpu", "node", "socket", "core", "online"}:
            raise CpuTpError("lscpu extended row schema mismatch")
        if not all(isinstance(row[key], int) for key in ("cpu", "node", "socket", "core")):
            raise CpuTpError("lscpu extended numeric field is malformed")
        if row["online"] is not True:
            raise CpuTpError("offline CPU present in fixed topology")
        normalized.append(dict(row))
    if len({row["cpu"] for row in normalized}) != len(normalized):
        raise CpuTpError("duplicate logical CPU in topology")
    return sorted(normalized, key=lambda item: item["cpu"])


def topology_attestation(summary_text: str, extended_text: str, numactl_text: str) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    summary = parse_lscpu_summary(summary_text)
    cpus = parse_lscpu_extended(extended_text)
    cores: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in cpus:
        cores.setdefault((row["socket"], row["core"]), []).append(row)
    physical = [min(rows, key=lambda row: row["cpu"]) for rows in cores.values()]
    physical_by_node: dict[str, list[int]] = {}
    logical_by_node: dict[str, list[int]] = {}
    for row in cpus:
        logical_by_node.setdefault(str(row["node"]), []).append(row["cpu"])
    for row in physical:
        physical_by_node.setdefault(str(row["node"]), []).append(row["cpu"])
    expected = {str(node): list(range(node * 24, (node + 1) * 24)) for node in range(4)}
    checks = {
        "vendor": summary.get("Vendor ID") == "AuthenticAMD",
        "model": summary.get("Model name") == "AMD EPYC 9655 96-Core Processor",
        "family": summary.get("CPU family") == "26",
        "model_id": summary.get("Model") == "2",
        "sockets": summary.get("Socket(s)") == "1",
        "cores": len(cores) == 96,
        "logical_cpus": len(cpus) == 192,
        "nodes": sorted(logical_by_node) == ["0", "1", "2", "3"],
        "physical_rank_masks": physical_by_node == expected,
        "numactl_nps4": "available: 4 nodes (0-3)" in numactl_text,
    }
    reasons.extend(f"topology_{key}_mismatch" for key, ok in checks.items() if not ok)
    body = {
        "summary": summary, "logical_cpus": cpus,
        "physical_cpus_by_node": physical_by_node,
        "logical_cpus_by_node": logical_by_node,
        "expected_physical_cpus_by_node": expected,
        "checks": checks,
        "raw_sha256": {
            "lscpu_summary": hashlib.sha256(summary_text.encode()).hexdigest(),
            "lscpu_extended": hashlib.sha256(extended_text.encode()).hexdigest(),
            "numactl_hardware": hashlib.sha256(numactl_text.encode()).hexdigest(),
        },
    }
    body["topology_sha256"] = content_sha256(body)
    return body, reasons


def parse_region_lock_status(text: str) -> dict[str, Any]:
    rows: dict[str, str] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 2 and re.fullmatch(r"q[0-3]", fields[0]):
            if fields[0] in rows:
                raise CpuTpError("duplicate region-lock row")
            rows[fields[0]] = fields[1]
    if set(rows) != {"q0", "q1", "q2", "q3"}:
        raise CpuTpError("region-lock status does not name q0-q3 exactly")
    return {"regions": rows, "all_free": all(value == "free" for value in rows.values())}


def _number(text: str, *, label: str, positive: bool = False) -> float:
    cleaned = text.strip().replace("%", "")
    if not cleaned or cleaned.startswith("<"):
        raise CpuTpError(f"{label} is not numeric: {text!r}")
    try:
        value = float(cleaned)
    except ValueError as exc:
        raise CpuTpError(f"{label} is not numeric: {text!r}") from exc
    if not math.isfinite(value) or (positive and value <= 0):
        raise CpuTpError(f"{label} is not a finite positive value: {text!r}")
    return value


def parse_perf_stat(
    text: str, expected_events: Iterable[str], *, delimiter: str = ";",
    minimum_running_ratio: float = 0.90,
) -> dict[str, Any]:
    expected = tuple(expected_events)
    if not expected or len(set(expected)) != len(expected):
        raise CpuTpError("perf expected-event set is empty or duplicated")
    rows: dict[str, dict[str, Any]] = {}
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    for fields in reader:
        if not fields or not any(item.strip() for item in fields):
            continue
        if fields[0].lstrip().startswith("#"):
            continue
        if len(fields) < 5:
            raise CpuTpError("perf stat row has fewer than five CSV fields")
        value_text, unit, event, enabled_text, running_text = [item.strip() for item in fields[:5]]
        if event not in expected:
            raise CpuTpError(f"unexpected perf event {event!r}")
        if event in rows:
            raise CpuTpError(f"duplicate perf event {event!r}")
        value = _number(value_text, label=f"perf {event} count")
        if value < 0:
            raise CpuTpError(f"perf {event} count is negative")
        enabled = _number(enabled_text, label=f"perf {event} enabled time", positive=True)
        running_value = _number(running_text, label=f"perf {event} running ratio", positive=True)
        ratio = running_value / 100.0 if running_value > 1.0 else running_value
        if ratio > 1.0 or ratio < minimum_running_ratio:
            raise CpuTpError(f"perf {event} running/enabled ratio {ratio:.6f} is invalid")
        rows[event] = {
            "value": value, "unit": unit, "time_enabled": enabled,
            "time_running_ratio": ratio,
        }
    missing = [event for event in expected if event not in rows]
    if missing:
        raise CpuTpError(f"perf stat output misses events: {missing}")
    return {"events": rows, "minimum_running_ratio": minimum_running_ratio}


def _normalized_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def parse_uprof_pcm(
    text: str, expected_groups: Iterable[str], *, delimiter: str = ",",
) -> dict[str, Any]:
    """Parse the strict long-form PCM export retained by this harness.

    uProf versions vary in their presentation files.  The collector must retain
    those raw files and emit one lossless long-form CSV with this exact header:
    ``metric_group,metric,scope,scope_id,value,unit,duration_seconds``.
    No metric may be renamed: ``metric_group`` is one of the names listed by the
    installed AMDuProfPcm binary, while ``metric`` preserves its output label.
    """
    expected = tuple(expected_groups)
    if not expected or len(set(expected)) != len(expected):
        raise CpuTpError("uProf expected metric-group set is empty or duplicated")
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    required = (
        "metric_group", "metric", "scope", "scope_id", "value", "unit",
        "duration_seconds",
    )
    if reader.fieldnames is None or tuple(_normalized_header(item) for item in reader.fieldnames) != required:
        raise CpuTpError("uProf PCM long-form header is not exact")
    groups: dict[str, list[dict[str, Any]]] = {name: [] for name in expected}
    for line_number, raw in enumerate(reader, 2):
        if None in raw:
            raise CpuTpError(f"uProf row has extra CSV fields on line {line_number}")
        row = {_normalized_header(str(key)): str(value).strip() for key, value in raw.items()}
        group = row["metric_group"]
        if group not in groups:
            raise CpuTpError(f"unexpected uProf metric group {group!r} on line {line_number}")
        if not row["metric"] or not row["scope"] or not row["scope_id"] or not row["unit"]:
            raise CpuTpError(f"uProf identity field is empty on line {line_number}")
        value = _number(row["value"], label=f"uProf value line {line_number}")
        duration = _number(
            row["duration_seconds"], label=f"uProf duration line {line_number}",
            positive=True,
        )
        groups[group].append({
            "metric": row["metric"], "scope": row["scope"],
            "scope_id": row["scope_id"], "value": value,
            "unit": row["unit"], "duration_seconds": duration,
        })
    missing = [name for name, rows in groups.items() if not rows]
    if missing:
        raise CpuTpError(f"uProf PCM output misses metric groups: {missing}")
    for name, rows in groups.items():
        identities = {(row["metric"], row["scope"], row["scope_id"]) for row in rows}
        if len(identities) != len(rows):
            raise CpuTpError(f"uProf PCM group {name} has duplicate metric/scope rows")
    return {"metric_groups": groups}


def hashed_order(items: Sequence[str], *, seed: int, namespace: str) -> list[str]:
    return sorted(
        items,
        key=lambda item: hashlib.sha256(f"{seed}:{namespace}:{item}".encode()).hexdigest(),
    )


def stopping_rules(seed: int = CAMPAIGN_SEED) -> dict[str, Any]:
    h_blocks = [
        {"block": block, "arms": hashed_order(("H0", "H1"), seed=seed,
                                                namespace=f"phase0:{block}")}
        for block in range(1, 11)
    ]
    panels = ["U0", "U1", "U2", "C0", "C1", "C2"]
    panel_runs = [f"{panel}:rep-{rep:02d}" for panel in panels for rep in range(1, 6)]
    panel_order = hashed_order(panel_runs, seed=seed, namespace="phase0-panels")
    algorithms = hashed_order(
        ("central-reduce-broadcast", "binary-tree", "reduce-scatter-all-gather"),
        seed=seed, namespace="phase1-latin-base",
    )
    phase1: list[dict[str, Any]] = []
    for sample in range(1, 31):
        rotation = (sample - 1) % len(algorithms)
        order = algorithms[rotation:] + algorithms[:rotation]
        phase1.append({"sample": sample, "algorithms": order})
    later = [
        {"block": block, "arms": hashed_order(("A", "B", "C"), seed=seed,
                                                namespace=f"later:{block}")}
        for block in range(1, 31)
    ]
    body = {
        "schema": "epyc.cpu_tp.phase01_stopping_rule.v1",
        "seed": seed,
        "phase0": {
            "paired_blocks": 10, "schedule": h_blocks,
            "profile_repetitions_per_panel": 5, "panel_schedule": panel_order,
            "bootstrap_resamples": 100000, "no_early_stop": True,
            "no_extension": True,
        },
        "phase1": {
            "warmups_per_algorithm": 5, "samples_per_algorithm": 30,
            "allreduce_calls_per_sample": 128, "elements": 5120,
            "transport": "FP32", "latin_square_schedule": phase1,
            "pass_upper_bound_lte": 0.10, "fail_lower_bound_gt": 0.15,
            "otherwise": "INCONCLUSIVE_STOP", "no_early_stop": True,
            "no_extension": True,
        },
        "later": {
            "paired_abc_blocks": 30, "schedule": later,
            "bootstrap_resamples": 100000, "no_early_stop": True,
            "no_extension": True,
        },
    }
    body["stopping_rule_sha256"] = content_sha256(body)
    return body


def schema_attestation() -> dict[str, Any]:
    names = (
        "cpu_tp_phase01_preflight.schema.json",
        "cpu_tp_phase01_counter_panel.schema.json",
        "cpu_tp_phase01_ratification.schema.json",
        "cpu_tp_phase01_n25_attestation.schema.json",
        "cpu_tp_phase01_stopping_rule.schema.json",
    )
    rows = {name: file_identity(SCHEMA_DIR / name) for name in names}
    return {"files": rows, "schema_manifest_sha256": content_sha256(rows)}


def protocol_attestation(path: Path) -> dict[str, Any]:
    identity = file_identity(path)
    text = path.read_text(encoding="utf-8")
    return {
        **identity, "protocol_id": PROTOCOL_ID,
        "draft_marker_present": "NOT RATIFIED" in text,
    }


def verify_ratification(
    path: Path, expected_file_sha256: str, *, protocol: Mapping[str, Any],
    runner: Mapping[str, Any], schemas: Mapping[str, Any],
) -> dict[str, Any]:
    if not _HEX64.fullmatch(expected_file_sha256):
        raise CpuTpError("ratification receipt file SHA-256 is required")
    if sha256_file(path) != expected_file_sha256:
        raise CpuTpError("ratification receipt file SHA-256 mismatch")
    receipt = read_canonical_receipt(path, expected_schema=RATIFICATION_SCHEMA)
    exact = {
        "schema", "status", "protocol_id", "protocol_sha256", "runner_sha256",
        "schema_manifest_sha256", "ratified_at", "ratified_by", "receipt_sha256",
    }
    if set(receipt) != exact:
        raise CpuTpError("ratification receipt key set is not exact")
    if receipt["status"] != "ratified" or receipt["protocol_id"] != PROTOCOL_ID:
        raise CpuTpError("protocol is not human-ratified")
    if (
        receipt["protocol_sha256"] != protocol["sha256"]
        or receipt["runner_sha256"] != runner["sha256"]
        or receipt["schema_manifest_sha256"] != schemas["schema_manifest_sha256"]
    ):
        raise CpuTpError("ratification receipt does not bind this runner/protocol/schema set")
    if not isinstance(receipt["ratified_by"], str) or not receipt["ratified_by"].strip():
        raise CpuTpError("ratification receipt lacks a human identity")
    return receipt


def n25_attestation(path: Path | None, expected_file_sha256: str | None,
                    topology_sha256: str) -> tuple[dict[str, Any], list[str]]:
    if path is None or expected_file_sha256 is None:
        return {"status": "absent"}, ["n25_landed_reloaded_attestation_absent"]
    if not _HEX64.fullmatch(expected_file_sha256) or sha256_file(path) != expected_file_sha256:
        return {"status": "invalid", "path": str(path)}, ["n25_attestation_hash_mismatch"]
    try:
        receipt = read_canonical_receipt(path, expected_schema=N25_SCHEMA)
    except CpuTpError as exc:
        return {"status": "invalid", "path": str(path), "error": str(exc)}, [
            "n25_attestation_invalid"
        ]
    required = {
        "schema", "status", "topology_sha256", "root_commit", "orchestrator_commit",
        "research_commit", "reloaded_at", "ratified_by", "receipt_sha256",
    }
    reasons: list[str] = []
    if set(receipt) != required:
        reasons.append("n25_attestation_key_set_mismatch")
    if receipt.get("status") != "landed_reloaded":
        reasons.append("n25_not_landed_reloaded")
    if receipt.get("topology_sha256") != topology_sha256:
        reasons.append("n25_topology_hash_mismatch")
    for key in ("root_commit", "orchestrator_commit", "research_commit"):
        if not isinstance(receipt.get(key), str) or not re.fullmatch(r"[0-9a-f]{40}", receipt[key]):
            reasons.append(f"n25_{key}_malformed")
    return {"status": "valid" if not reasons else "invalid", "receipt": receipt}, reasons


def production_attestation(root: Path, binary: Path) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    head = run_capture(("git", "-C", str(root), "rev-parse", "HEAD"))
    branch = run_capture(("git", "-C", str(root), "branch", "--show-current"))
    status = run_capture(("git", "-C", str(root), "status", "--porcelain"))
    library_path = f"{binary.parent}:/opt/AMD/aocc-compiler-5.0.0/lib:/opt/rocm/lib"
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = library_path
    version = run_capture((str(binary), "--version"), env=env)
    ldd = run_capture(("ldd", str(binary)))
    for name, result in (("head", head), ("branch", branch), ("status", status),
                         ("version", version), ("ldd", ldd)):
        if not result["ok"]:
            reasons.append(f"production_{name}_probe_failed")
    if head["stdout"].strip() != EXPECTED_PRODUCTION_COMMIT:
        reasons.append("production_commit_mismatch")
    if branch["stdout"].strip() != EXPECTED_PRODUCTION_BRANCH:
        reasons.append("production_branch_mismatch")
    if status["stdout"].strip():
        reasons.append("production_worktree_dirty")
    match = _VERSION.search(version["stdout"] + version["stderr"])
    if not match or int(match.group(1)) != EXPECTED_BINARY_VERSION \
            or not EXPECTED_PRODUCTION_COMMIT.startswith(match.group(2)):
        reasons.append("production_binary_version_mismatch")
    libraries: dict[str, Any] = {}
    if ldd["ok"]:
        for line in ldd["stdout"].splitlines():
            match_path = re.search(r"=>\s+(/\S+)", line)
            if match_path:
                path = Path(match_path.group(1))
                if path.parent == binary.parent and path.exists():
                    libraries[path.name] = file_identity(path)
    if not libraries:
        reasons.append("production_local_library_closure_empty")
    try:
        binary_identity = file_identity(binary)
    except (OSError, CpuTpError) as exc:
        binary_identity = {"path": str(binary), "error": str(exc)}
        reasons.append("production_binary_identity_unavailable")
    return {
        "root": str(root), "expected_branch": EXPECTED_PRODUCTION_BRANCH,
        "expected_commit": EXPECTED_PRODUCTION_COMMIT,
        "expected_version": EXPECTED_BINARY_VERSION,
        "binary": binary_identity, "libraries": libraries,
        "library_path": library_path,
        "commands": {name: _compact_command(result) for name, result in (
            ("head", head), ("branch", branch), ("status", status),
            ("version", version), ("ldd", ldd))},
    }, reasons


def tool_attestation(perf: Path | None, uprof: Path | None) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    output: dict[str, Any] = {}
    required = {
        "lscpu": shutil.which("lscpu"), "numactl": shutil.which("numactl"),
        "numastat": shutil.which("numastat"),
        "perf": str(perf) if perf else shutil.which("perf"),
        "AMDuProfPcm": str(uprof) if uprof else shutil.which("AMDuProfPcm"),
    }
    for name, value in required.items():
        if not value:
            output[name] = {"available": False}
            reasons.append(f"tool_{name}_absent")
            continue
        try:
            output[name] = {"available": True, "identity": file_identity(Path(value))}
        except (OSError, CpuTpError) as exc:
            output[name] = {"available": False, "path": value, "error": str(exc)}
            reasons.append(f"tool_{name}_identity_invalid")
    if output.get("perf", {}).get("available"):
        for label, argv in (
            ("version", (required["perf"], "version")),
            ("list", (required["perf"], "list", "--no-desc")),
        ):
            result = run_capture(tuple(str(item) for item in argv), timeout=30)
            output["perf"][label] = _compact_command(result)
            if not result["ok"]:
                reasons.append(f"perf_{label}_failed")
        listed = output["perf"].get("list", {}).get("stdout", "")
        aliases = sorted({event for panel in PERF_PANELS.values() for event in panel})
        missing = [event for event in aliases if re.search(
            rf"(?<![A-Za-z0-9_.]){re.escape(event)}(?![A-Za-z0-9_.])", listed
        ) is None]
        output["perf"]["required_aliases"] = aliases
        output["perf"]["missing_aliases"] = missing
        if missing:
            reasons.append("perf_required_aliases_missing")
    if output.get("AMDuProfPcm", {}).get("available"):
        probes = {}
        for label, option in (("version", "-v"), ("help", "-h"),
                              ("topology", "-n"), ("metrics", "-l")):
            result = run_capture((required["AMDuProfPcm"], option), timeout=30)
            probes[label] = _compact_command(result)
            if not result["ok"]:
                reasons.append(f"uprof_{label}_failed")
        output["AMDuProfPcm"]["probes"] = probes
        listing = probes.get("metrics", {}).get("stdout", "") + probes.get("help", {}).get("stdout", "")
        missing = [name for name in UPROF_GROUPS if re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", listing
        ) is None]
        output["AMDuProfPcm"]["required_metric_groups"] = list(UPROF_GROUPS)
        output["AMDuProfPcm"]["missing_metric_groups"] = missing
        if missing:
            reasons.append("uprof_required_metric_groups_missing")
    return output, reasons


def collect_preflight(args: argparse.Namespace) -> dict[str, Any]:
    reasons: list[str] = []
    generated_at = datetime.now(UTC).isoformat()
    runner = file_identity(Path(__file__).resolve())
    schemas = schema_attestation()
    protocol = protocol_attestation(args.protocol_file)

    lscpu_summary = run_capture(("lscpu", "--json"))
    lscpu_extended = run_capture(("lscpu", "--json", "--extended=CPU,NODE,SOCKET,CORE,ONLINE"))
    numactl = run_capture(("numactl", "--hardware"))
    if not all(item["ok"] for item in (lscpu_summary, lscpu_extended, numactl)):
        raise CpuTpError("topology command failed; cannot form an attestation")
    topology, topology_reasons = topology_attestation(
        lscpu_summary["stdout"], lscpu_extended["stdout"], numactl["stdout"]
    )
    reasons.extend(topology_reasons)

    production, production_reasons = production_attestation(
        args.production_root, args.llama_server
    )
    reasons.extend(production_reasons)
    model, model_reasons = model_attestation(args.model, hash_model=args.hash_model)
    reasons.extend(model_reasons)

    lock_result = run_capture((str(args.region_lock), "status"))
    try:
        lock = parse_region_lock_status(lock_result["stdout"]) if lock_result["ok"] else {
            "error": lock_result["stderr"]
        }
    except CpuTpError as exc:
        lock = {"error": str(exc)}
    if not lock_result["ok"] or lock.get("all_free") is not True:
        reasons.append("region_lock_q0_q3_not_free")
    # A free status is only eligibility. validate-only never acquires authority.
    reasons.append("region_lock_q0_q3_not_held_by_validate_only")

    try:
        uptime_seconds = float(Path("/proc/uptime").read_text().split()[0])
    except (OSError, ValueError, IndexError) as exc:
        raise CpuTpError("cannot read host uptime") from exc
    if uptime_seconds >= UPTIME_LIMIT_SECONDS:
        reasons.append("uptime_at_least_one_week_reboot_required")

    tools, tool_reasons = tool_attestation(args.perf, args.uprof)
    reasons.extend(tool_reasons)
    n25, n25_reasons = n25_attestation(
        args.n25_attestation, args.n25_attestation_sha256,
        topology["topology_sha256"],
    )
    reasons.extend(n25_reasons)
    try:
        n25_handoff = file_identity(args.n25_handoff)
    except (OSError, CpuTpError) as exc:
        n25_handoff = {"path": str(args.n25_handoff), "error": str(exc)}
        reasons.append("n25_handoff_identity_unavailable")

    ratification: dict[str, Any]
    if args.ratification_receipt and args.ratification_receipt_sha256:
        try:
            receipt = verify_ratification(
                args.ratification_receipt, args.ratification_receipt_sha256,
                protocol=protocol, runner=runner, schemas=schemas,
            )
            ratification = {"status": "valid", "receipt": receipt,
                            "file_sha256": args.ratification_receipt_sha256}
        except CpuTpError as exc:
            ratification = {"status": "invalid", "error": str(exc)}
            reasons.append("protocol_ratification_invalid")
    else:
        ratification = {"status": "absent"}
        reasons.append("protocol_not_human_ratified")

    unique_reasons = sorted(set(reasons))
    body = {
        "schema": PREFLIGHT_SCHEMA, "generated_at": generated_at,
        "mode": "validate_only", "claim_status": "OBSERVATION",
        "protocol": protocol, "ratification": ratification,
        "runner": runner, "schemas": schemas,
        "host": {
            "hostname": platform.node(), "kernel": platform.release(),
            "machine": platform.machine(), "uptime_seconds": uptime_seconds,
            "uptime_limit_seconds": UPTIME_LIMIT_SECONDS,
            "perf_event_paranoid": Path(
                "/proc/sys/kernel/perf_event_paranoid").read_text().strip(),
        },
        "topology": topology, "production": production, "model": model,
        "tools": tools,
        "region_lock": {"status": lock, "command": _compact_command(lock_result)},
        "n25": n25, "n25_handoff": n25_handoff,
        "stopping_rules": stopping_rules(args.seed),
        "execution_authorized": False,
        "status": "blocked" if unique_reasons else "ready_for_human_ratified_execution",
        "blockers": unique_reasons,
    }
    return self_hashed(body)


def panel_receipt(kind: str, source: Path, parsed: Mapping[str, Any]) -> dict[str, Any]:
    return self_hashed({
        "schema": PANEL_SCHEMA, "kind": kind,
        "source": file_identity(source), "parsed": dict(parsed),
        "claim_status": "OBSERVATION", "protocol_id": PROTOCOL_ID,
    })


def _common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--protocol-file", type=Path, required=True)
    parser.add_argument("--production-root", type=Path, default=DEFAULT_PRODUCTION_ROOT)
    parser.add_argument("--llama-server", type=Path, default=DEFAULT_LLAMA_SERVER)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--region-lock", type=Path, default=DEFAULT_REGION_LOCK)
    parser.add_argument("--n25-handoff", type=Path, default=DEFAULT_N25_HANDOFF)
    parser.add_argument("--n25-attestation", type=Path)
    parser.add_argument("--n25-attestation-sha256")
    parser.add_argument("--perf", type=Path)
    parser.add_argument("--uprof", type=Path)
    parser.add_argument("--ratification-receipt", type=Path)
    parser.add_argument("--ratification-receipt-sha256")
    parser.add_argument("--seed", type=int, default=CAMPAIGN_SEED)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-only")
    _common_paths(validate)
    validate.add_argument("--hash-model", action="store_true")
    validate.add_argument("--output", type=Path)

    perf = sub.add_parser("parse-perf")
    perf.add_argument("--input", type=Path, required=True)
    perf.add_argument("--panel", choices=tuple(PERF_PANELS), required=True)
    perf.add_argument("--output", type=Path)

    uprof = sub.add_parser("parse-uprof")
    uprof.add_argument("--input", type=Path, required=True)
    uprof.add_argument("--groups", nargs="+", choices=UPROF_GROUPS, required=True)
    uprof.add_argument("--output", type=Path)

    schedule = sub.add_parser("schedule")
    schedule.add_argument("--seed", type=int, default=CAMPAIGN_SEED)
    schedule.add_argument("--output", type=Path)

    execute = sub.add_parser("execute")
    _common_paths(execute)
    execute.add_argument("--hash-model", action="store_true")
    return parser.parse_args(argv)


def _emit(payload: Mapping[str, Any], output: Path | None) -> None:
    if output is not None:
        if output.exists() or output.is_symlink():
            raise CpuTpError(f"refusing to replace existing output: {output}")
        unsigned = dict(payload)
        if "receipt_sha256" in unsigned:
            unsigned.pop("receipt_sha256")
            write_receipt(output, unsigned)
        else:
            write_receipt(output, unsigned)
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "validate-only":
            payload = collect_preflight(args)
            _emit(payload, args.output)
            return 0 if payload["status"] != "blocked" else 2
        if args.command == "parse-perf":
            parsed = parse_perf_stat(args.input.read_text(encoding="utf-8"), PERF_PANELS[args.panel])
            _emit(panel_receipt(f"perf:{args.panel}", args.input, parsed), args.output)
            return 0
        if args.command == "parse-uprof":
            parsed = parse_uprof_pcm(args.input.read_text(encoding="utf-8"), args.groups)
            _emit(panel_receipt("uprof", args.input, parsed), args.output)
            return 0
        if args.command == "schedule":
            _emit(self_hashed(stopping_rules(args.seed)), args.output)
            return 0
        if args.command == "execute":
            payload = collect_preflight(args)
            if payload["ratification"]["status"] != "valid":
                raise CpuTpError(
                    "benchmark execution refused: exact human-ratified protocol hash is absent"
                )
            raise CpuTpError(
                "benchmark execution refused: this runner version is validate-only; "
                "a separately reviewed execution implementation is required"
            )
        raise AssertionError(args.command)
    except (OSError, CpuTpError) as exc:
        print(f"cpu_tp_phase01_runner: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
