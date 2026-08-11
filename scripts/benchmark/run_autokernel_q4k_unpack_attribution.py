#!/usr/bin/env python3
"""Differential gfx90a attribution for Q4_K superblock unpack in MMVQ.

The production kernel fuses weight loads, Q4_K scale/min unpack, dot products,
and reduction into one ``mul_mat_vec_q`` dispatch.  Dispatch-level PMCs cannot
time only the unpack instructions, so this producer deliberately does *not*
emit an inside-kernel wall-share estimate.  It instead runs three matched
production-shape cells (Q4_K, Q4_0, Q8_0): Q4_K-vs-Q4_0 is the closest
available superblock-metadata control and Q4_K-vs-Q8_0 retains the handoff's
quant-ladder comparison.

The runner creates test-backend-ops input files itself.  This exercises the
unchanged production implementation at arbitrary model-derived shapes without
adding a benchmark-only case to llama.cpp.  Every active row is correctness
checked, the full campaign holds the MI210 claim, and zero/absent SQ counter
transport is a hard failure rather than evidence of zero unpack work.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark import gguf_tensor_contract as G
from scripts.benchmark import run_autokernel_omniperf_fallback as O
from scripts.benchmark.autokernel_claimed_sampling import error_payload, stop_sampler_and_release
from scripts.benchmark.capture_autokernel_c4_profile import artifact_inventory, assert_source_identity
from scripts.benchmark.run_autokernel_gpu_factorial import sha256_file, write_json_atomic
from scripts.kernel_rnd.autokernel import storage
from scripts.kernel_rnd.autokernel.execution import device_sampler
from scripts.kernel_rnd.autokernel.resource import device_claim


SCHEMA = "epyc.autokernel.q4k_unpack_attribution.v1"
AUTHORITY = "diagnostic_only"
FROZEN_V9_COMMIT = "0db32c06e3e550065b78311a6031ef3dd2c4f27c"
PRODUCTION_SHAPE = (17408, 1, 5120)  # m,n,k; 128 same-name Qwen3.6-27B FFN tensors per arm.
GGML_OP_MUL_MAT = 29
GGML_TYPE_F32 = 0
QUANTS = {
    "q4_K": {"type_id": 12, "block_size": 256, "type_size": 144},
    "q4_0": {"type_id": 2, "block_size": 32, "type_size": 18},
    "q8_0": {"type_id": 8, "block_size": 32, "type_size": 34},
}
PRIMARY_COUNTERS = (
    "SQ_WAVES",
    "SQ_INSTS",
    "SQ_INSTS_VALU",
    "SQ_INSTS_VALU_INT32",
    "SQ_INSTS_VMEM_RD",
    "SQ_INSTS_SALU",
    "SQ_INSTS_BRANCH",
    "SQ_WAIT_ANY",
    "SQ_ACTIVE_INST_VALU",
    "SQ_ACTIVE_INST_VMEM",
    "TCC_REQ_sum",
    "TCC_HIT_sum",
    "TCC_MISS_sum",
    "TCC_EA_RDREQ_DRAM_sum",
)
LEGACY_DETERMINISTIC_SQ_COUNTERS = (
    "SQ_WAVES",
    *(counter for counter in PRIMARY_COUNTERS if counter.startswith("SQ_INSTS")),
)
ROCPROFV2_COUNTERS = (
    "SQ_WAVES",
    "SQ_INSTS_VALU",
    "SQ_INSTS_VALU_INT32",
)
ROCPROFV2_PMC_LINE = "pmc: " + " ".join(ROCPROFV2_COUNTERS)
ROCPROFV2_COUNTER_SEMANTICS = {
    "SQ_WAVES": "waves sent to SQs; per-SIMD emulated global counter",
    "SQ_INSTS_VALU": "VALU instructions issued; per-SIMD emulated counter",
    "SQ_INSTS_VALU_INT32": (
        "32-bit signed/unsigned integer VALU instructions issued; per-SIMD emulated counter"),
}
IDENTIFIABILITY = {
    "direct_hardware_counter_attribution": "differential_mechanism_only",
    "exact_inside_kernel_wall_share": None,
    "reason": (
        "gfx90a PMCs and device timestamps are dispatch-scoped; mul_mat_vec_q fuses "
        "unpack, dot-product, memory, and reduction, and Q8_0 uses a different nwarps setting"
    ),
    "closest_control": "Q4_K minus Q4_0 at identical m,n,k",
}


class CounterTransportError(RuntimeError):
    """The profiler ran, but its counter transport cannot support attribution."""


class MissingProfilerArtifactError(RuntimeError):
    """A profiler exited successfully without producing its declared CSV artifact."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def derive_ggml_op_value(header: Path, member: str) -> int:
    """Resolve one ``enum ggml_op`` member from the source-bound header."""
    text = header.read_text(encoding="utf-8")
    matches = list(re.finditer(r"\benum\s+ggml_op\s*\{(?P<body>.*?)\};", text, re.DOTALL))
    if len(matches) != 1:
        raise RuntimeError(f"expected one enum ggml_op in {header}, observed {len(matches)}")
    body = re.sub(r"/\*.*?\*/", "", matches[0].group("body"), flags=re.DOTALL)
    body = re.sub(r"//[^\n]*", "", body)
    value = -1
    for raw_entry in body.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        parsed = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)(?:\s*=\s*([0-9]+))?", entry)
        if parsed is None:
            raise RuntimeError(f"unsupported ggml_op enum entry in {header}: {entry!r}")
        name, explicit = parsed.groups()
        value = int(explicit) if explicit is not None else value + 1
        if name == member:
            return value
    raise RuntimeError(f"{member} is absent from enum ggml_op in {header}")


def _strides(type_size: int, block_size: int, ne: tuple[int, int, int, int]) -> tuple[int, ...]:
    if ne[0] % block_size:
        raise RuntimeError(f"k={ne[0]} is not divisible by block size {block_size}")
    nb0 = type_size
    nb1 = type_size * ne[0] // block_size
    nb2 = nb1 * ne[1]
    nb3 = nb2 * ne[2]
    return nb0, nb1, nb2, nb3


def test_file_line(quant: str, *, m: int, n: int, k: int) -> str:
    """Serialize one contiguous MUL_MAT generic-op row for test-backend-ops."""
    spec = QUANTS[quant]
    src0_ne = (k, m, 1, 1)
    src1_ne = (k, n, 1, 1)
    src0_nb = _strides(spec["type_size"], spec["block_size"], src0_ne)
    src1_nb = _strides(4, 1, src1_ne)
    output_ne = (m, n, 1, 1)
    op_params = (0,) * 16  # GGML_MAX_OP_PARAMS == 64 bytes at frozen v9.
    values = [GGML_OP_MUL_MAT, GGML_TYPE_F32, *output_ne, len(op_params), *op_params, 2]
    values.extend((spec["type_id"], *src0_ne, *src0_nb))
    values.extend((GGML_TYPE_F32, *src1_ne, *src1_nb))
    return " ".join(str(value) for value in values) + " -\n"


def write_test_file(path: Path, quant: str, *, m: int, n: int, k: int,
                    repetitions: int) -> dict[str, Any]:
    if repetitions < 1:
        raise RuntimeError("test-file repetitions must be positive")
    line = test_file_line(quant, m=m, n=n, k=k)
    path.write_text(line * repetitions, encoding="utf-8")
    return {"path": str(path), "sha256": sha256_file(path), "rows": repetitions,
            "quant": quant, "shape": {"m": m, "n": n, "k": k}}


def backend_command(binary: Path, test_file: Path, *, backend: str) -> tuple[str, ...]:
    return (str(binary), "test", "-o", "MUL_MAT", "-b", backend,
            "--test-file", str(test_file), "--output", "csv")


def omniperf_command(binary: Path, test_file: Path, output_dir: Path, *,
                     workload_name: str, backend: str,
                     args: argparse.Namespace) -> tuple[str, ...]:
    proxy = argparse.Namespace(**vars(args))
    proxy.workload_name = workload_name
    proxy.backend = backend
    # O.omniperf_command only needs these fields plus profiler paths.  Replace
    # its stock backend command with our arbitrary-shape test-file producer.
    prefix = (
        # Do not resolve this symlink: Python uses argv[0] to discover its
        # virtual environment.  Dereferencing it silently selects system Python.
        str(Path(args.omniperf_python).absolute()), "-c", O._LOCALE_COMPAT,
        str(Path(args.omniperf).resolve()), "profile", "-n", workload_name,
        "-p", str(output_dir), "-b", "SQ", "TCC", "--no-roof", "--",
    )
    return prefix + backend_command(binary, test_file, backend=backend)


def rocprofv2_command(binary: Path, test_file: Path, counter_file: Path,
                      raw_dir: Path, *, workload_name: str, backend: str,
                      args: argparse.Namespace) -> tuple[str, ...]:
    profiler = Path(args.profiler_prefix).resolve() / "bin/rocprofv2"
    return (
        str(profiler), "-i", str(counter_file), "--plugin", "file",
        "--plugin-version", "2", "-d", str(raw_dir), "-o", workload_name,
        *backend_command(binary, test_file, backend=backend),
    )


def parse_rocprofv2_counter_listing(text: str, *, arch_device: str = "gfx90a:0"
                                     ) -> dict[str, str]:
    """Return exact required-counter descriptions from rocprofv2's device list."""
    clean = re.sub(r"\x1b\[[0-9;]*m", "", text)
    found: dict[str, list[str]] = {counter: [] for counter in ROCPROFV2_COUNTERS}
    pattern = re.compile(
        rf"^\s*{re.escape(arch_device)}\s+:\s+([A-Za-z0-9_]+)\s+:\s+(.*?)\s*$")
    for line in clean.splitlines():
        match = pattern.match(line)
        if match and match.group(1) in found:
            found[match.group(1)].append(match.group(2))
    invalid = {counter: descriptions for counter, descriptions in found.items()
               if len(descriptions) != 1}
    if invalid:
        raise RuntimeError(
            f"rocprofv2 counter listing lacks an exact {arch_device} minimal set: {invalid}")
    return {counter: descriptions[0] for counter, descriptions in found.items()}


def validate_rocprofv2_counter_support(args: argparse.Namespace, *,
                                       env: dict[str, str], output_dir: Path
                                       ) -> dict[str, Any]:
    profiler = Path(args.profiler_prefix).resolve() / "bin/rocprofv2"
    if not profiler.is_file():
        raise RuntimeError(f"rocprofv2 is unavailable: {profiler}")
    command = (str(profiler), "--list-counters")
    rc, stdout, stderr, duration = O.run_owned(command, env=env, timeout_s=60.0)
    (output_dir / "rocprofv2-list-counters.stdout.txt").write_text(
        stdout, encoding="utf-8")
    (output_dir / "rocprofv2-list-counters.stderr.txt").write_text(
        stderr, encoding="utf-8")
    descriptions = parse_rocprofv2_counter_listing(stdout)
    # ROCm 6.2's wrapper returns 1 after successfully printing the complete
    # list.  Accept only that exact observed quirk: any stderr or other rc fails.
    if rc not in (0, 1) or (rc == 1 and stderr.strip()):
        raise RuntimeError(
            f"rocprofv2 --list-counters failed rc={rc}: {stderr[-2000:]!r}")
    return {
        "command": list(command), "returncode": rc, "duration_s": duration,
        "arch_device": "gfx90a:0", "counter_descriptions": descriptions,
        "counter_semantics": ROCPROFV2_COUNTER_SEMANTICS,
        "counter_file_line": ROCPROFV2_PMC_LINE,
        "single_pass_group": True,
        "profiler": str(profiler), "profiler_sha256": sha256_file(profiler),
    }


def select_rocprofv2_counter_csv(raw_dir: Path) -> Path:
    matches = []
    required = {"Kernel_Name", "Start_Timestamp", "End_Timestamp", *ROCPROFV2_COUNTERS}
    candidates = sorted(raw_dir.rglob("*.csv"))
    if not candidates:
        raise MissingProfilerArtifactError(
            f"rocprofv2 emitted no CSV artifact below {raw_dir}")
    for candidate in candidates:
        with candidate.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not required.issubset(set(reader.fieldnames or ())):
                continue
            if any("mul_mat_vec_q" in row.get("Kernel_Name", "") for row in reader):
                matches.append(candidate)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one rocprofv2 counter CSV below {raw_dir}, found {matches}")
    return matches[0]


def profile_cell(binary: Path, test_file: Path, arm_dir: Path, *,
                 workload_name: str, quant: str, args: argparse.Namespace,
                 env: dict[str, str], counter_file: Path | None) -> dict[str, Any]:
    """Run a bounded transport retry without retrying parsed counter failures."""
    attempts = []
    profile: dict[str, Any] | None = None
    for attempt_number in range(1, args.transport_attempts + 1):
        attempt_dir = arm_dir / f"attempt-{attempt_number:02d}"
        attempt_dir.mkdir(parents=True)
        attempt_workload = f"{workload_name}_a{attempt_number:02d}"
        if args.counter_transport == "rocprofv2":
            if counter_file is None:
                raise RuntimeError("rocprofv2 transport requires a counter file")
            raw_dir = attempt_dir / "raw"
            raw_dir.mkdir()
            command = rocprofv2_command(
                binary, test_file, counter_file, raw_dir,
                workload_name=attempt_workload, backend=args.backend, args=args)
            stdout_path = attempt_dir / "rocprofv2.stdout.txt"
            stderr_path = attempt_dir / "rocprofv2.stderr.txt"
        else:
            raw_dir = None
            command = omniperf_command(
                binary, test_file, attempt_dir,
                workload_name=attempt_workload, backend=args.backend, args=args)
            stdout_path = attempt_dir / "omniperf.stdout.txt"
            stderr_path = attempt_dir / "omniperf.stderr.txt"
        try:
            rc, stdout, stderr, duration = O.run_owned(
                command, env=env, timeout_s=args.profile_timeout_s)
        except Exception as exc:
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
            attempts.append({
                "attempt": attempt_number, "attempt_dir": str(attempt_dir),
                "command": list(command), "returncode": None, "duration_s": None,
                "stdout_path": str(stdout_path), "stderr_path": str(stderr_path),
                "result": "execution_exception", "retryable": False,
                "error": f"{type(exc).__name__}: {exc}",
            })
            break
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        attempt = {
            "attempt": attempt_number, "attempt_dir": str(attempt_dir),
            "command": list(command), "returncode": rc, "duration_s": duration,
            "stdout_path": str(stdout_path), "stderr_path": str(stderr_path),
        }
        if rc != 0:
            attempt.update({
                "result": "nonzero_profiler_exit", "retryable": True,
                "error": f"{args.counter_transport} exited {rc}",
            })
            attempts.append(attempt)
            continue
        try:
            if raw_dir is not None:
                pmc = select_rocprofv2_counter_csv(raw_dir)
                counter_fields = ROCPROFV2_COUNTERS
                deterministic = ROCPROFV2_COUNTERS
            else:
                pmc = attempt_dir / "pmc_perf.csv"
                if not pmc.is_file():
                    raise MissingProfilerArtifactError(
                        f"Omniperf emitted no counter artifact: {pmc}")
                counter_fields = PRIMARY_COUNTERS
                deterministic = LEGACY_DETERMINISTIC_SQ_COUNTERS
        except MissingProfilerArtifactError as exc:
            attempt.update({
                "result": "missing_profiler_artifact", "retryable": True,
                "error": str(exc),
            })
            attempts.append(attempt)
            continue
        except Exception as exc:
            attempt.update({
                "result": "artifact_selection_failure", "retryable": False,
                "error": f"{type(exc).__name__}: {exc}",
            })
            attempts.append(attempt)
            break
        try:
            profile = summarize_counter_table(
                pmc, quant=quant, expected_dispatches=args.active_repetitions,
                counter_fields=counter_fields,
                deterministic_counters=deterministic)
            attempt.update({
                "result": ("accepted_parsed_transport"
                           if profile["counter_transport_valid"]
                           else "accepted_parsed_counter_failure"),
                "retryable": False, "pmc_path": str(pmc),
                "pmc_sha256": sha256_file(pmc),
            })
            attempts.append(attempt)
            profile.update({
                "pmc_sha256": sha256_file(pmc), "pmc_path": str(pmc),
                "profiler_returncode": rc, "command": list(command),
                "duration_s": duration, "accepted_attempt": attempt_number,
            })
            break
        except Exception as exc:
            attempt.update({
                "result": "parsed_counter_failure", "retryable": False,
                "pmc_path": str(pmc), "pmc_sha256": sha256_file(pmc),
                "error": f"{type(exc).__name__}: {exc}",
            })
            attempts.append(attempt)
            break
    if profile is None:
        last_error = attempts[-1].get("error") if attempts else "no attempts executed"
        profile = {
            "quant": quant, "profiler_returncode": (
                attempts[-1].get("returncode") if attempts else None),
            "counter_transport_valid": False, "counter_per_wave": None,
            "claim_eligible": False, "accepted_attempt": None,
            "error": last_error,
        }
    profile["attempts"] = attempts
    profile["predeclared_attempt_limit"] = args.transport_attempts
    return profile


def validate_test_output(text: str, *, expected_rows: int) -> dict[str, Any]:
    rows = [row for row in csv.DictReader(text.splitlines())
            if row.get("op_name") == "MUL_MAT"]
    if len(rows) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} MUL_MAT rows, observed {len(rows)}")
    failures = [row for row in rows if row.get("supported") != "1"
                or row.get("hard_failure") == "1" or row.get("error_message")]
    if failures:
        raise RuntimeError(f"backend-op output contains {len(failures)} failure(s)")
    return {"rows": len(rows), "all_supported": True, "all_correct": True}


def _number(row: dict[str, str], field: str) -> float:
    value = row.get(field)
    if value in (None, ""):
        raise RuntimeError(f"counter table has no value for {field}")
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"counter {field} is not numeric: {value!r}") from exc


def summarize_counter_table(path: Path, *, quant: str,
                            expected_dispatches: int,
                            counter_fields: tuple[str, ...] = PRIMARY_COUNTERS,
                            deterministic_counters: tuple[str, ...] = (
                                LEGACY_DETERMINISTIC_SQ_COUNTERS)) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        raw = list(reader)
    required = {"Kernel_Name", "Start_Timestamp", "End_Timestamp", *counter_fields}
    missing = sorted(required - fields)
    if missing:
        raise RuntimeError(f"counter table is missing fields: {missing}")
    type_id = QUANTS[quant]["type_id"]
    targets = [row for row in raw if "mul_mat_vec_q" in row["Kernel_Name"]
               and f"(ggml_type){type_id}" in row["Kernel_Name"]]
    if len(targets) != expected_dispatches:
        raise RuntimeError(
            f"{quant} expected {expected_dispatches} exact MMVQ dispatches, observed {len(targets)}")

    values: dict[str, list[float]] = {field: [] for field in counter_fields}
    durations = []
    dispatch_rows = []
    for dispatch_index, row in enumerate(targets):
        start = _number(row, "Start_Timestamp")
        end = _number(row, "End_Timestamp")
        if end <= start:
            raise RuntimeError("counter table emitted a non-positive device duration")
        durations.append(end - start)
        dispatch_counters = {}
        for field in counter_fields:
            value = _number(row, field)
            values[field].append(value)
            dispatch_counters[field] = value
        dispatch_rows.append({
            "dispatch_index": dispatch_index,
            "kernel_name": row["Kernel_Name"],
            "grid_size": row.get("Grid_Size"),
            "workgroup_size": row.get("Workgroup_Size"),
            "start_timestamp": start, "end_timestamp": end,
            "device_duration_ns": end - start,
            "counters": dispatch_counters,
        })

    # These are the counters that make an unpack mechanism distinguishable.
    # TCC-only data can characterize traffic but cannot identify unpack work.
    transport_errors = []
    for field in deterministic_counters:
        invalid = [index for index, value in enumerate(values[field]) if value <= 0]
        if invalid:
            transport_errors.append({
                "counter": field, "invalid_dispatch_indices": invalid,
                "reason": "zero_or_negative",
            })

    medians = {field: statistics.median(series) for field, series in values.items()}
    transport_valid = not transport_errors
    per_wave = None
    if transport_valid:
        waves = medians["SQ_WAVES"]
        per_wave = {field: value / waves for field, value in medians.items()
                    if field.startswith("SQ_INSTS")}
    return {
        "quant": quant,
        "dispatches": len(targets),
        "kernel_name": targets[0]["Kernel_Name"],
        "grid_size": targets[0].get("Grid_Size"),
        "workgroup_size": targets[0].get("Workgroup_Size"),
        "device_duration_ns_median": statistics.median(durations),
        "counter_medians": medians,
        "raw_dispatch_evidence": dispatch_rows,
        "counter_transport_valid": transport_valid,
        "counter_transport_errors": transport_errors,
        "counter_per_wave": per_wave,
        "transport_contract": {
            "counter_fields": list(counter_fields),
            "deterministic_counters": list(deterministic_counters),
        },
        "claim_eligible": transport_valid,
    }


def transport_integrity_summary(blocks: list[dict[str, Any]], *,
                                expected_blocks: int,
                                deterministic_counters: tuple[str, ...]) -> dict[str, Any]:
    """Prove rocprof pass rows stayed aligned across the repeated campaign.

    Wave and executed-instruction counts are deterministic for an identical
    quant/shape/kernel dispatch.  Their tolerance is therefore exactly zero;
    accepting block-local positive values would let permuted multipass rows
    masquerade as a valid profile.
    """
    expected_ids = list(range(expected_blocks))
    by_id = {block["block"]: block for block in blocks}
    result: dict[str, Any] = {
        "expected_blocks": expected_ids,
        "observed_blocks": sorted(by_id),
        "tolerance": 0.0,
        "tolerance_reason": (
            "SQ_WAVES and executed SQ instruction counts are dispatch-deterministic "
            "for identical quant/shape/kernel cells"),
        "deterministic_counters": list(deterministic_counters),
        "quants": {},
    }
    for quant in QUANTS:
        missing_blocks = []
        block_errors = []
        profiles = []
        for block_id in expected_ids:
            arm = by_id.get(block_id, {}).get("arms", {}).get(quant)
            if arm is None:
                missing_blocks.append(block_id)
                continue
            if not arm.get("counter_transport_valid", False):
                block_errors.append({
                    "block": block_id, "error": arm.get("error"),
                    "counter_transport_errors": arm.get("counter_transport_errors", []),
                })
            profiles.append((block_id, arm))
        counters = {}
        for counter in deterministic_counters:
            block_medians = {
                str(block_id): arm.get("counter_medians", {}).get(counter)
                for block_id, arm in profiles
            }
            raw_by_block = {
                str(block_id): [row.get("counters", {}).get(counter)
                                for row in arm.get("raw_dispatch_evidence", [])]
                for block_id, arm in profiles
            }
            raw_values = [value for series in raw_by_block.values() for value in series]
            present_raw = [value for value in raw_values if value is not None]
            present_medians = [value for value in block_medians.values() if value is not None]
            complete = (
                len(profiles) == expected_blocks
                and all(raw_by_block.get(str(block_id)) for block_id in expected_ids)
                and all(value is not None for value in block_medians.values())
                and all(value is not None for value in raw_values)
            )
            positive = complete and all(value > 0 for value in raw_values)
            unique = sorted(set(present_raw))
            median_unique = sorted(set(present_medians))
            invariant = complete and len(unique) == 1
            counters[counter] = {
                "block_medians": block_medians,
                "raw_dispatch_counts_by_block": {
                    block_id: len(series) for block_id, series in raw_by_block.items()},
                "cross_block_median_unique_values": median_unique,
                "cross_block_drift": len(median_unique) > 1,
                "raw_unique_values": unique,
                "complete": complete, "positive": positive,
                "exactly_invariant": invariant,
                "valid": complete and positive and invariant,
            }
        quant_valid = (
            not missing_blocks and not block_errors
            and all(item["valid"] for item in counters.values())
        )
        result["quants"][quant] = {
            "valid": quant_valid, "missing_blocks": missing_blocks,
            "block_errors": block_errors, "counters": counters,
        }
    result["valid"] = (
        sorted(by_id) == expected_ids
        and all(item["valid"] for item in result["quants"].values())
    )
    return result


def paired_block_summary(blocks: list[dict[str, Any]], *,
                         expected_blocks: int | None = None,
                         deterministic_counters: tuple[str, ...] = (
                             LEGACY_DETERMINISTIC_SQ_COUNTERS)) -> dict[str, Any]:
    if not blocks:
        raise RuntimeError("cannot summarize an empty attribution campaign")
    if expected_blocks is None:
        expected_blocks = len(blocks)
    integrity = transport_integrity_summary(
        blocks, expected_blocks=expected_blocks,
        deterministic_counters=deterministic_counters)
    comparisons = {"q4_K_minus_q4_0": [], "q4_K_minus_q8_0": []}
    eligibility = []
    for block in blocks:
        arms = block["arms"]
        invalid_arms = [quant for quant in QUANTS
                        if not arms.get(quant, {}).get("counter_transport_valid", False)]
        eligibility.append({
            "block": block["block"], "eligible": integrity["valid"],
            "invalid_arms": invalid_arms,
            "campaign_transport_integrity_valid": integrity["valid"],
        })
        if not integrity["valid"]:
            continue
        for label, control in (("q4_K_minus_q4_0", "q4_0"),
                               ("q4_K_minus_q8_0", "q8_0")):
            q4k = arms["q4_K"]
            other = arms[control]
            comparisons[label].append({
                "block": block["block"],
                "device_duration_ns_delta": (
                    q4k["device_duration_ns_median"] - other["device_duration_ns_median"]),
                "int32_insts_per_wave_delta": (
                    q4k["counter_per_wave"]["SQ_INSTS_VALU_INT32"]
                    - other["counter_per_wave"]["SQ_INSTS_VALU_INT32"]),
                "valu_insts_per_wave_delta": (
                    q4k["counter_per_wave"]["SQ_INSTS_VALU"]
                    - other["counter_per_wave"]["SQ_INSTS_VALU"]),
            })
    return {
        "identifiability": IDENTIFIABILITY,
        "comparisons": comparisons,
        "comparison_eligibility": eligibility,
        "transport_integrity": integrity,
        "counter_transport_valid": integrity["valid"],
        "claim_eligible": integrity["valid"],
        "inside_unpack_wall_share": None,
    }


def shape_evidence(q4_model: Path, q8_model: Path, *, gguf_py: Path,
                   m: int, k: int) -> dict[str, Any]:
    contracts = {}
    for label, path in (("q4_K", q4_model), ("q8_0", q8_model)):
        if not path.is_file():
            raise RuntimeError(f"{label} shape-evidence model is unavailable: {path}")
        contracts[label] = G.read_contract(
            [path], gguf_py=gguf_py, tensor_patterns=(r"^blk\.[0-9]+\..*\.weight$",),
            metadata_patterns=(r"^general\.architecture$",), layer_start=None, layer_end=None)
    selected = {}
    tensor_types = {"q4_K": "Q4_K", "q8_0": "Q8_0"}
    for label, contract in contracts.items():
        names = sorted(tensor["name"] for tensor in contract["tensors"]
                       if tensor["tensor_type"] == tensor_types[label]
                       and tensor["shape"] == [k, m])
        if not names:
            raise RuntimeError(f"{label} model has no [{k},{m}] weight tensors")
        selected[label] = names
    common = sorted(set(selected["q4_K"]) & set(selected["q8_0"]))
    if len(common) < 16:
        raise RuntimeError(
            f"production-shape contract needs >=16 same-name tensors, observed {len(common)}")
    payload = json.dumps(common, sort_keys=True, separators=(",", ":")).encode()
    return {
        "shape": {"m": m, "n": 1, "k": k},
        "q4_model": str(q4_model), "q8_model": str(q8_model),
        "q4_model_size": q4_model.stat().st_size,
        "q8_model_size": q8_model.stat().st_size,
        "same_name_tensor_count": len(common),
        "same_name_tensor_names_sha256": hashlib.sha256(payload).hexdigest(),
        "first_tensor_names": common[:8],
        "authority": "GGUF metadata/tensor descriptors only; model weights are not executed",
    }


def cmake_build_identity(binary: Path, source_root: Path) -> dict[str, Any]:
    """Bind the instrument to a fresh HIP build of the asserted source tree."""
    build_root = binary.parent.parent
    cache = build_root / "CMakeCache.txt"
    if not cache.is_file():
        raise RuntimeError(f"binary has no adjacent CMakeCache.txt: {cache}")
    values: dict[str, str] = {}
    for line in cache.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(("//", "#")) or "=" not in line:
            continue
        key_and_type, value = line.split("=", 1)
        key = key_and_type.split(":", 1)[0]
        values[key] = value
    home = Path(values.get("CMAKE_HOME_DIRECTORY", "")).resolve()
    if home != source_root:
        raise RuntimeError(
            f"binary CMake source mismatch: expected {source_root}, observed {home}")
    required = {
        "CMAKE_BUILD_TYPE": "Release",
        "GGML_HIP": "ON",
        "GGML_HIP_MMQ_MFMA": "ON",
        "GGML_HIP_ROCWMMA_FATTN": "ON",
    }
    drift = {key: {"expected": expected, "observed": values.get(key)}
             for key, expected in required.items() if values.get(key) != expected}
    targets = {part for key in ("AMDGPU_TARGETS", "GPU_TARGETS")
               for part in re.split(r"[;,]", values.get(key, "")) if part}
    if "gfx90a" not in targets:
        drift["GPU_TARGETS"] = {"expected": "contains gfx90a", "observed": sorted(targets)}
    if drift:
        raise RuntimeError(f"binary CMake contract drifted: {drift}")

    completed = subprocess.run(
        ("git", "ls-files", "-z"), cwd=source_root, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30.0)
    tracked = [source_root / raw.decode() for raw in completed.stdout.split(b"\0") if raw]
    missing = [str(path) for path in tracked if not path.exists()]
    if missing:
        raise RuntimeError(f"source tree has missing tracked inputs: {missing[:8]}")
    newest_source = max((path.stat().st_mtime_ns for path in tracked), default=0)
    if binary.stat().st_mtime_ns < newest_source:
        raise RuntimeError("test-backend-ops predates one or more tracked source inputs")
    return {
        "build_root": str(build_root), "cmake_cache": str(cache),
        "cmake_cache_sha256": sha256_file(cache),
        "cmake_home_directory": str(home),
        "cmake_contract": {key: values.get(key) for key in required},
        "gpu_targets": sorted(targets), "tracked_source_files": len(tracked),
        "newest_tracked_source_mtime_ns": newest_source,
        "binary_mtime_ns": binary.stat().st_mtime_ns,
    }


def linkage_identity(binary: Path, *, env: dict[str, str],
                     minimum_mtime_ns: int) -> dict[str, Any]:
    completed = subprocess.run(("ldd", str(binary)), env=env, check=True, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30.0)
    libraries = []
    for line in completed.stdout.splitlines():
        if "libggml" not in line and "libllama" not in line:
            continue
        target = line.split("=>", 1)[1].strip().split(" ", 1)[0] if "=>" in line else ""
        if not target or not Path(target).is_relative_to(binary.parent):
            raise RuntimeError(f"binary resolves a llama/ggml DSO outside its build: {line.strip()}")
        if Path(target).stat().st_mtime_ns < minimum_mtime_ns:
            raise RuntimeError(f"linked build-local DSO predates tracked source inputs: {target}")
        libraries.append(target)
    if not libraries:
        raise RuntimeError("ldd exposed no build-local llama/ggml linkage")
    return {
        "resolved_libraries": [
            {"path": path, "sha256": sha256_file(Path(path)),
             "mtime_ns": Path(path).stat().st_mtime_ns}
            for path in libraries
        ],
        "ldd_stdout": completed.stdout,
    }


def source_identity(source_root: Path, expected_commit: str | None) -> dict[str, Any]:
    commit = assert_source_identity(source_root, expected_commit)
    mmvq = source_root / "ggml/src/ggml-cuda/mmvq.cu"
    vecdot = source_root / "ggml/src/ggml-cuda/vecdotq.cuh"
    header = source_root / "ggml/include/ggml.h"
    for path in (mmvq, vecdot, header):
        if not path.is_file():
            raise RuntimeError(f"source contract file is missing: {path}")
    derived_mul_mat = derive_ggml_op_value(header, "GGML_OP_MUL_MAT")
    if derived_mul_mat != GGML_OP_MUL_MAT:
        raise RuntimeError(
            "frozen-v9 generic-op serialization drifted: "
            f"GGML_OP_MUL_MAT={derived_mul_mat}, runner expects {GGML_OP_MUL_MAT}")
    bodies = mmvq.read_text(encoding="utf-8") + vecdot.read_text(encoding="utf-8")
    required = (
        "case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1;",
        "case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1;",
        "static __device__ __forceinline__ float vec_dot_q4_K_q8_1(",
        "static __global__ void mul_mat_vec_q(",
    )
    missing = [fragment for fragment in required if fragment not in bodies]
    if missing:
        raise RuntimeError(f"frozen-v9 MMVQ source contract drifted: {missing}")
    return {
        "source_root": str(source_root), "source_commit": commit,
        "mmvq_sha256": sha256_file(mmvq), "vecdotq_sha256": sha256_file(vecdot),
        "ggml_header_sha256": sha256_file(header),
        "ggml_op_mul_mat": derived_mul_mat,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    binary = Path(args.binary).resolve()
    source_root = Path(args.source_root).resolve()
    output_dir = Path(storage.assert_not_scratch(
        args.output_dir, what="Q4_K unpack attribution evidence directory"))
    if output_dir.exists():
        raise RuntimeError(f"attribution output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    started_at = utc_now()
    started = time.monotonic()
    captured_error: BaseException | None = None
    claim = None
    sampler = None
    opened = released = sampling_receipt = None
    teardown_errors: tuple[BaseException, ...] = ()
    identity = shape = None
    counter_support = None
    preflights: dict[str, Any] = {}
    blocks: list[dict[str, Any]] = []
    profile_failures: list[dict[str, Any]] = []
    profile_schedule = []
    for block in range(args.blocks):
        order = ("q4_K", "q4_0", "q8_0") if block % 2 == 0 else ("q8_0", "q4_0", "q4_K")
        for quant in order:
            profile_schedule.append({
                "block": block, "quant": quant,
                "attempt_limit": args.transport_attempts,
            })
    try:
        if not binary.is_file():
            raise RuntimeError(f"test-backend-ops binary is unavailable: {binary}")
        env = O.profiler_environment(binary, args)
        env["GGML_CUDA_Q8_PREFETCH"] = "0"
        identity = source_identity(source_root, args.source_commit)
        build = cmake_build_identity(binary, source_root)
        identity.update({"binary": str(binary), "binary_sha256": sha256_file(binary),
                         "build": build,
                         "linkage": linkage_identity(
                             binary, env=env,
                             minimum_mtime_ns=build["newest_tracked_source_mtime_ns"]),
                         "runner_sha256": sha256_file(Path(__file__).resolve())})
        shape = shape_evidence(
            Path(args.q4_model).resolve(), Path(args.q8_model).resolve(),
            gguf_py=Path(args.gguf_py).resolve(), m=args.op_m, k=args.op_k)

        counter_file = None
        if args.counter_transport == "rocprofv2":
            counter_file = output_dir / "rocprofv2.counters.txt"
            counter_file.write_text(ROCPROFV2_PMC_LINE + "\n", encoding="utf-8")

        test_files = {}
        for quant in QUANTS:
            test_files[quant] = write_test_file(
                output_dir / f"{quant}.test-ops.txt", quant,
                m=args.op_m, n=1, k=args.op_k, repetitions=args.active_repetitions)

        claim = device_claim.acquire_device_claim(
            "mi210_0", purpose="AutoKernel INF-37 Q4_K unpack differential attribution",
            campaign_id=args.campaign_id,
            journal=device_claim.ClaimJournal(args.claim_journal),
            holder_label="run_autokernel_q4k_unpack_attribution.py",
            timeout_s=args.claim_timeout_s,
            max_hold_s=(
                args.profile_timeout_s * (3 * args.blocks * args.transport_attempts + 1)
                + 300.0))
        opened = claim.receipt().to_dict()
        sampler = device_sampler.RocmSmiSampler(device_index=0, interval_s=0.250).start()
        if args.counter_transport == "rocprofv2":
            counter_support = validate_rocprofv2_counter_support(
                args, env=env, output_dir=output_dir)
        else:
            counter_support = {
                "transport": "omniperf-v1",
                "authority": "legacy_diagnostic_only",
                "warning": (
                    "rocprof-v1 multipass merge has demonstrated cross-block row permutation; "
                    "campaign-wide exact invariance remains mandatory"),
            }

        for quant in QUANTS:
            one = output_dir / f"{quant}.preflight.test-ops.txt"
            write_test_file(one, quant, m=args.op_m, n=1, k=args.op_k,
                            repetitions=args.preflight_repetitions)
            command = backend_command(binary, one, backend=args.backend)
            rc, stdout, stderr, duration = O.run_owned(
                command, env=env, timeout_s=args.preflight_timeout_s)
            (output_dir / f"{quant}.preflight.stdout.csv").write_text(stdout, encoding="utf-8")
            (output_dir / f"{quant}.preflight.stderr.txt").write_text(stderr, encoding="utf-8")
            if rc != 0:
                raise RuntimeError(f"{quant} correctness preflight exited {rc}")
            preflights[quant] = validate_test_output(
                stdout, expected_rows=args.preflight_repetitions)
            preflights[quant].update({"command": list(command), "duration_s": duration})

        for block in range(args.blocks):
            order = ("q4_K", "q4_0", "q8_0") if block % 2 == 0 else ("q8_0", "q4_0", "q4_K")
            record = {"block": block, "order": list(order), "arms": {}}
            for quant in order:
                arm_dir = output_dir / f"block-{block:02d}-{quant}"
                arm_dir.mkdir(parents=True)
                workload = f"inf37_q4k_unpack_b{block:02d}_{quant.replace('_', '')}"
                profile = profile_cell(
                    binary, Path(test_files[quant]["path"]), arm_dir,
                    workload_name=workload, quant=quant, args=args,
                    env=env, counter_file=counter_file)
                if not profile["counter_transport_valid"]:
                    profile_failures.append({
                        "block": block, "quant": quant,
                        "error": profile.get("error"),
                        "counter_transport_errors": profile.get("counter_transport_errors", []),
                    })
                record["arms"][quant] = profile
            blocks.append(record)
    except BaseException as exc:
        captured_error = exc
    finally:
        if claim is not None:
            sampling_receipt, released_receipt, teardown_errors = stop_sampler_and_release(
                sampler=sampler, claim=claim)
            released = released_receipt.to_dict() if released_receipt is not None else None
    if teardown_errors and captured_error is None:
        captured_error = teardown_errors[0]

    summary = None
    if blocks:
        try:
            deterministic = (ROCPROFV2_COUNTERS
                             if args.counter_transport == "rocprofv2"
                             else LEGACY_DETERMINISTIC_SQ_COUNTERS)
            summary = paired_block_summary(
                blocks, expected_blocks=args.blocks,
                deterministic_counters=deterministic)
        except BaseException as exc:
            if captured_error is None:
                captured_error = exc
    if captured_error is None and summary is not None and not summary["claim_eligible"]:
        captured_error = CounterTransportError(
            f"campaign-wide counter transport integrity failed "
            f"({len(profile_failures)} block-local arm failures); full raw matrix retained, "
            "no normalized comparisons emitted")
    payload = {
        "schema": SCHEMA,
        "status": "failed" if captured_error is not None else "passed",
        "authority": AUTHORITY,
        "campaign_id": args.campaign_id,
        "started_at": started_at, "ended_at": utc_now(),
        "duration_s": time.monotonic() - started,
        "identity": identity, "shape_evidence": shape,
        "workload": {"shape": {"m": args.op_m, "n": 1, "k": args.op_k},
                     "blocks": args.blocks,
                     "active_repetitions": args.active_repetitions,
                     "counter_transport": args.counter_transport,
                     "transport_attempts": args.transport_attempts,
                     "profile_schedule": profile_schedule,
                     "q8_prefetch": "forced_off",
                     "graphs": "disabled_by_profiler_environment"},
        "preflights": preflights, "blocks": blocks, "summary": summary,
        "profile_failures": profile_failures,
        "counter_support": counter_support,
        "identifiability": IDENTIFIABILITY,
        "belief_measurements": [],
        "device_claim_open": opened, "device_claim_released": released,
        "device_sampling": sampling_receipt.to_dict() if sampling_receipt is not None else None,
        "teardown_errors": error_payload(teardown_errors),
        "error": None if captured_error is None else {
            "type": type(captured_error).__name__, "message": str(captured_error)},
    }
    payload["artifacts"] = artifact_inventory(output_dir)
    write_json_atomic(output_dir / "receipt.json", payload)
    if captured_error is not None:
        raise RuntimeError(
            f"Q4_K unpack attribution failed; durable receipt: {output_dir / 'receipt.json'}") from captured_error
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-root", required=True)
    result.add_argument("--source-commit", default=FROZEN_V9_COMMIT)
    result.add_argument("--binary", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--campaign-id", default="inf37-q4k-unpack-v9-20260811")
    result.add_argument("--q4-model", default="/mnt/raid0/llm/models/Qwen_Qwen3.6-27B-Q4_K_M.gguf")
    result.add_argument("--q8-model", default="/mnt/raid0/llm/models/Qwen_Qwen3.6-27B-Q8_0.gguf")
    result.add_argument("--gguf-py", default="/mnt/raid0/llm/llama.cpp/gguf-py")
    result.add_argument("--op-m", type=int, default=PRODUCTION_SHAPE[0])
    result.add_argument("--op-k", type=int, default=PRODUCTION_SHAPE[2])
    result.add_argument("--blocks", type=int, default=4)
    result.add_argument("--preflight-repetitions", type=int, default=1)
    result.add_argument("--active-repetitions", type=int, default=5)
    result.add_argument("--backend", default="ROCm0")
    result.add_argument(
        "--counter-transport", choices=("rocprofv2", "omniperf-v1"),
        default="rocprofv2",
        help="rocprofv2 is the governed single-pass path; omniperf-v1 is legacy diagnostic only")
    result.add_argument(
        "--transport-attempts", type=int, default=2,
        help="predeclared per-cell attempt ceiling; retries only nonzero exit/missing artifact")
    result.add_argument("--profiler-root", default="/mnt/raid0/llm/tools/rocm-profilers-6.2")
    result.add_argument("--profiler-prefix", default="/mnt/raid0/llm/tools/rocm-profilers-6.2/opt/rocm-6.2.0")
    result.add_argument("--omniperf", default="/mnt/raid0/llm/tools/rocm-profilers-6.2/opt/rocm-6.2.0/libexec/omniperf/omniperf")
    result.add_argument("--omniperf-python", default="/mnt/raid0/llm/tools/omniperf-venv-6.2/bin/python")
    result.add_argument("--claim-journal", default="/mnt/raid0/llm/ak-claims/device.jsonl")
    result.add_argument("--claim-timeout-s", type=float, default=0.0)
    result.add_argument("--preflight-timeout-s", type=float, default=900.0)
    result.add_argument("--profile-timeout-s", type=float, default=1800.0)
    return result


def main() -> int:
    args = parser().parse_args()
    for field in ("op_m", "op_k", "blocks", "preflight_repetitions",
                  "active_repetitions", "transport_attempts"):
        if getattr(args, field) < 1:
            raise RuntimeError(f"--{field.replace('_', '-')} must be positive")
    payload = run(args)
    print(json.dumps({"receipt": str(Path(args.output_dir) / "receipt.json"),
                      "status": payload["status"],
                      "identifiability": payload["identifiability"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
