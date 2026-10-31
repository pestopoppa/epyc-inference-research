#!/usr/bin/env python3
"""Sealed/unseen C2/C6 and exact C3 provider panel for raw-HIP SiLU.

This is a task-local, no-promotion evaluator.  Candidate source is sealed before
the cryptographic suite seed exists.  The candidate child receives inputs but
never the independent host-double oracle, runs inside the AutoKernel sandbox,
and must overwrite two differently poisoned outputs deterministically.  C3 is
the exact SiLU expression compiled by Torch-Inductor for ROCm on the same tensor
and device, not Torch eager and not an operation-transferred GEMM library.
"""

from __future__ import annotations

import argparse
from array import array
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import difflib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import secrets
import shutil
import stat
import subprocess
import sys
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence

from . import arena_adapter
from . import hip_authoring_arm as hip
from ..evaluator import devices, statistics
from ..execution import reward_hack_scan, sandbox


SCHEMA = "epyc.autokernel.hip_decision_grade.v1"
PRODUCER_ID = "autokernel.controller.hip_decision_grade/v1"
AUTHORITY = "task_local_rank_no_release_or_promotion_authority"
CONTRIBUTION_FLOOR = 0.03
TIMING_BLOCKS = 20
REPETITIONS_PER_ARM = 30_000
MAX_CANDIDATES = 20
ALPHA_SELECTION = 1.0 / MAX_CANDIDATES
E_THRESHOLD = 1.0 / ALPHA_SELECTION
CONSTRUCTION_ID = "sign_martingale_predictable_lambda/v1"
CORRECTNESS_ATOL = 3.0e-6
CORRECTNESS_RTOL = 3.0e-6
DEFAULT_ARENA_ROOT = hip.DEFAULT_ARENA_ROOT
DEFAULT_TASK = hip.DEFAULT_TASK
DEFAULT_CLAIM_JOURNAL = hip.DEFAULT_CLAIM_JOURNAL
COUNTER_EXPECTATION_SCHEMA = "epyc.autokernel.gfx90a_counter_expectation.v1"
COUNTER_EVIDENCE_SCHEMA = "epyc.autokernel.gfx90a_counter_evidence.v1"
_MAX_COUNTER_CSV_BYTES = 64 * 1024 * 1024


class HipDecisionGradeError(RuntimeError):
    """The HIP hardening panel is malformed, incomplete, or failed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _read_counter_csv(path: Path) -> bytes:
    flags = (os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
             | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("counter profile raw CSV is not a regular file")
        if before.st_size < 0 or before.st_size > _MAX_COUNTER_CSV_BYTES:
            raise ValueError("counter profile raw CSV exceeds the bounded read size")
        chunks: list[bytes] = []
        size = 0
        while True:
            block = os.read(descriptor, min(1024 * 1024,
                                            _MAX_COUNTER_CSV_BYTES + 1 - size))
            if not block:
                break
            chunks.append(block)
            size += len(block)
            if size > _MAX_COUNTER_CSV_BYTES:
                raise ValueError("counter profile raw CSV exceeds the bounded read size")
        after = os.fstat(descriptor)
        linked_after = os.stat(path, follow_symlinks=False)
        identity = lambda value: (
            value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
            value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        if (identity(before) != identity(after)
                or identity(after) != identity(linked_after)
                or size != after.st_size):
            raise ValueError("counter profile raw CSV changed during bounded read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _counter_expectation(value: object) -> dict[str, str]:
    required = {"schema", "counter_name", "expected_direction", "unit",
                "candidate_kernel_name", "anchor_kernel_name", "normalization"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("counter expectation has unknown or missing fields")
    if value.get("schema") != COUNTER_EXPECTATION_SCHEMA:
        raise ValueError("counter expectation schema is invalid")
    result: dict[str, str] = {}
    for field in ("counter_name", "unit", "candidate_kernel_name", "anchor_kernel_name"):
        item = value.get(field)
        if not isinstance(item, str) or not item.strip() or "\0" in item:
            raise ValueError(f"counter expectation {field} is invalid")
        result[field] = item.strip()
    direction = value.get("expected_direction")
    if direction not in {"increase", "decrease"}:
        raise ValueError("counter expectation direction must be increase or decrease")
    result["expected_direction"] = str(direction)
    if value.get("normalization") != "sum_exact_dispatches":
        raise ValueError("counter expectation normalization is unsupported")
    result["normalization"] = "sum_exact_dispatches"
    return {"schema": COUNTER_EXPECTATION_SCHEMA, **result}


def _counter_profile_sum(profile: object, *, kernel_name: str, counter_name: str,
                         unit: str, workload_sha256: str,
                         normalization: str, campaign_id: str) -> dict[str, Any]:
    required = {"run_id", "campaign_id", "source_sha256", "workload_sha256", "csv_path",
                "csv_sha256", "kernel_name", "dispatch_ids", "counter_name",
                "unit", "normalization"}
    if not isinstance(profile, Mapping) or set(profile) != required:
        raise ValueError("counter profile has unknown or missing fields")
    for field in ("run_id", "campaign_id", "kernel_name", "counter_name", "unit",
                  "normalization"):
        if not isinstance(profile.get(field), str) or not profile[field].strip():
            raise ValueError(f"counter profile {field} is invalid")
    for field in ("source_sha256", "workload_sha256", "csv_sha256"):
        digest = profile.get(field)
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError(f"counter profile {field} is invalid")
    if (profile["kernel_name"] != kernel_name
            or profile["campaign_id"] != campaign_id
            or profile["counter_name"] != counter_name
            or profile["unit"] != unit
            or profile["normalization"] != normalization
            or profile["workload_sha256"] != workload_sha256):
        raise ValueError("counter profile identity, unit, workload, or normalization mismatch")
    dispatch_ids = profile.get("dispatch_ids")
    if (not isinstance(dispatch_ids, list) or not dispatch_ids
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 0
                   for item in dispatch_ids)
            or len(set(dispatch_ids)) != len(dispatch_ids)):
        raise ValueError("counter profile dispatch_ids must be unique nonnegative integers")
    path_value = profile.get("csv_path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("counter profile csv_path is invalid")
    path = Path(path_value)
    raw_bytes = _read_counter_csv(path)
    if hashlib.sha256(raw_bytes).hexdigest() != profile["csv_sha256"]:
        raise ValueError("counter profile raw CSV is missing, unsafe, or hash-mismatched")

    parsed: dict[int, Decimal] = {}
    try:
        raw_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("counter profile raw CSV is not UTF-8") from exc
    reader = csv.DictReader(line for line in io.StringIO(raw_text, newline="") if line.strip())
    if reader.fieldnames is None:
        raise ValueError("counter profile raw CSV has no header")
    header_names = reader.fieldnames
    if (any(not isinstance(name, str) or not name.strip() for name in header_names)
            or len({name.strip().casefold() for name in header_names}) != len(header_names)):
        raise ValueError("counter profile raw CSV has empty or ambiguous headers")
    folded = {name.strip().casefold(): name for name in header_names}
    dispatch_col = next((folded[key] for key in
                         ("dispatch_id", "dispatchid", "dispatch_id ")
                         if key in folded), None)
    kernel_col = next((folded[key] for key in ("kernel_name", "kernelname", "name")
                       if key in folded), None)
    if dispatch_col is None or kernel_col is None:
        raise ValueError("counter profile raw CSV lacks dispatch/kernel identity")
    long_format = "counter_name" in folded and "counter_value" in folded
    value_col = folded.get(counter_name.casefold()) if not long_format else None
    if not long_format and value_col is None:
        raise ValueError("counter profile raw CSV lacks the declared native counter")
    for ordinal, row in enumerate(reader, 2):
        if row.get(kernel_col) != kernel_name:
            continue
        try:
            dispatch = int(row[dispatch_col])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"counter CSV row {ordinal} has invalid dispatch id") from exc
        if long_format:
            if row.get(folded["counter_name"]) != counter_name:
                continue
            raw_value = row.get(folded["counter_value"])
        else:
            raw_value = row.get(value_col)
        if dispatch in parsed:
            raise ValueError(f"counter CSV repeats selected dispatch {dispatch}")
        try:
            numeric = Decimal(str(raw_value))
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise ValueError(f"counter CSV row {ordinal} has invalid counter value") from exc
        if not numeric.is_finite() or numeric < 0:
            raise ValueError(f"counter CSV row {ordinal} has non-finite or negative value")
        parsed[dispatch] = numeric
    if set(parsed) != set(dispatch_ids):
        raise ValueError("counter CSV dispatch set differs from the declared exact dispatch set")
    total = sum(parsed.values(), Decimal(0))
    return {
        "run_id": profile["run_id"], "campaign_id": campaign_id,
        "source_sha256": profile["source_sha256"],
        "workload_sha256": workload_sha256, "kernel_name": kernel_name,
        "csv_sha256": profile["csv_sha256"], "counter_name": counter_name,
        "unit": unit, "normalization": normalization,
        "dispatch_ids": sorted(dispatch_ids), "dispatch_count": len(dispatch_ids),
        "raw_dispatch_values": {str(key): str(parsed[key]) for key in sorted(parsed)},
        "normalized_value": str(total),
    }


def evaluate_counter_gate(*, expectation: Mapping[str, Any] | None,
                          evidence: Mapping[str, Any] | None,
                          candidate_source_sha256: str, task_id: str, campaign_id: str,
                          workload_sha256: str,
                          anchor_identity_sha256: str) -> dict[str, Any]:
    """Replay a predeclared gfx90a counter against hash-bound native PMC CSVs.

    Missing or malformed evidence is UNKNOWN.  No counter value is inferred from
    absence, and the reported totals are recomputed from the raw dispatch rows.
    The CSV/receipt are caller-supplied: this function does not launch or attest
    profiler capture, so this seam alone is not an end-to-end PMC gate or live
    controller adoption.
    """
    if expectation is None:
        return {"status": "NOT_REQUIRED", "required": False,
                "capture_boundary": "external_receipt_replay_only",
                "end_to_end_capture_proven": False,
                "speedup_claim_allowed": True, "reason": "no explicit counter declaration"}
    try:
        expected = _counter_expectation(expectation)
        if evidence is None:
            raise ValueError("native counter receipt is missing")
        required = {"schema", "task_id", "campaign_id", "candidate_source_sha256",
                    "anchor_identity_sha256", "workload_sha256", "expectation_sha256",
                    "counter_name", "unit", "normalization", "candidate", "anchor"}
        if not isinstance(evidence, Mapping) or set(evidence) != required:
            raise ValueError("native counter receipt has unknown or missing fields")
        if (evidence.get("schema") != COUNTER_EVIDENCE_SCHEMA
                or evidence.get("task_id") != task_id
                or evidence.get("campaign_id") != campaign_id
                or evidence.get("candidate_source_sha256") != candidate_source_sha256
                or evidence.get("anchor_identity_sha256") != anchor_identity_sha256
                or evidence.get("workload_sha256") != workload_sha256
                or evidence.get("expectation_sha256") != _canonical_sha256(expected)
                or evidence.get("counter_name") != expected["counter_name"]
                or evidence.get("unit") != expected["unit"]
                or evidence.get("normalization") != expected["normalization"]):
            raise ValueError("native counter receipt does not match the sealed pair")
        candidate = _counter_profile_sum(
            evidence["candidate"], kernel_name=expected["candidate_kernel_name"],
            counter_name=expected["counter_name"], unit=expected["unit"],
            workload_sha256=workload_sha256, normalization=expected["normalization"],
            campaign_id=campaign_id)
        anchor = _counter_profile_sum(
            evidence["anchor"], kernel_name=expected["anchor_kernel_name"],
            counter_name=expected["counter_name"], unit=expected["unit"],
            workload_sha256=workload_sha256, normalization=expected["normalization"],
            campaign_id=campaign_id)
        if candidate["run_id"] == anchor["run_id"]:
            raise ValueError("candidate and anchor profile run ids must be distinct")
        if (candidate["source_sha256"] != candidate_source_sha256
                or anchor["source_sha256"] != anchor_identity_sha256):
            raise ValueError("profile source identity does not match the sealed candidate/anchor")
        candidate_value = Decimal(candidate["normalized_value"])
        anchor_value = Decimal(anchor["normalized_value"])
        fired = (candidate_value > anchor_value if expected["expected_direction"] == "increase"
                 else candidate_value < anchor_value)
        return {
            "status": "PASS" if fired else "FAIL",
            "required": True, "speedup_claim_allowed": fired,
            "capture_boundary": "external_receipt_replay_only",
            "end_to_end_capture_proven": False,
            "expectation_sha256": _canonical_sha256(expected),
            "expected_direction": expected["expected_direction"],
            "counter_name": expected["counter_name"], "unit": expected["unit"],
            "normalization": expected["normalization"],
            "candidate": candidate, "anchor": anchor,
            "direction_fired": fired,
            "reason": None if fired else "native counter did not move in the predeclared direction",
        }
    except (OSError, ValueError, TypeError, KeyError, csv.Error) as exc:
        return {"status": "UNKNOWN", "required": True,
                "speedup_claim_allowed": False,
                "capture_boundary": "external_receipt_replay_only",
                "end_to_end_capture_proven": False, "reason": str(exc)}


def candidate_decision(*, correctness_passed: bool, integrity_clean: bool,
                       timing: Mapping[str, Any], counter_gate: Mapping[str, Any]
                       ) -> dict[str, bool]:
    counter_required = counter_gate.get("required") is True
    counter_allowed = counter_gate.get("speedup_claim_allowed") is True
    if not counter_required:
        # Preserve legacy no-declaration behavior.  Existing callers keep their
        # prior speedup observation and rankability rules until they opt in.
        speedup_claim_allowed = True
        candidate_beats = timing.get("candidate_beats_exact_provider") is True
        rankable = bool(integrity_clean and candidate_beats)
    else:
        speedup_claim_allowed = bool(
            correctness_passed and integrity_clean and counter_allowed)
        candidate_beats = bool(
            speedup_claim_allowed
            and timing.get("candidate_beats_exact_provider") is True)
        rankable = candidate_beats
    return {
        "speedup_claim_allowed": speedup_claim_allowed,
        "candidate_beats_exact_provider": candidate_beats,
        "rankable_against_exact_task_local_provider": rankable,
    }


def _self_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["receipt_sha256"] = _canonical_sha256(result)
    return result


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_f32(path: Path, values: Sequence[float]) -> None:
    encoded = array("f", values)
    if sys.byteorder != "little":
        encoded.byteswap()
    with path.open("wb") as handle:
        encoded.tofile(handle)


def _read_f32(path: Path) -> tuple[float, ...]:
    values = array("f")
    with path.open("rb") as handle:
        values.fromfile(handle, path.stat().st_size // values.itemsize)
    if sys.byteorder != "little":
        values.byteswap()
    return tuple(float(value) for value in values)


def _silu_host_double(value: float) -> float:
    if value >= 0.0:
        return value / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return value * exp_value / (1.0 + exp_value)


def _materialize(distribution: str, count: int, rng: random.Random) -> list[float]:
    if distribution == "baseline":
        return [rng.uniform(-12.0, 12.0) for _ in range(count)]
    if distribution == "alternating":
        return [((-1.0) ** index) * (0.001 if index % 4 < 2 else 40.0)
                for index in range(count)]
    if distribution == "sparse_outlier":
        values = [rng.uniform(-0.02, 0.02) for _ in range(count)]
        for index in range(0, count, max(count // 7, 1)):
            values[index] = 60.0 if (index // max(count // 7, 1)) % 2 == 0 else -60.0
        return values
    if distribution == "cancellation":
        values = []
        while len(values) < count:
            value = rng.uniform(0.05, 18.0)
            values.extend((value, -value))
        return values[:count]
    raise HipDecisionGradeError(f"unknown hostile distribution {distribution}")


def _create_suite(work: Path, suite_seed: str) -> tuple[dict[str, Any], dict[str, tuple[float, ...]]]:
    inputs = work / "inputs"
    outputs = work / "outputs"
    inputs.mkdir()
    outputs.mkdir()
    shapes = (255, 257, 511, 513, 1023, 1025)
    distributions = ("baseline", "alternating", "sparse_outlier", "cancellation")
    cases = []
    expected = {}
    for shape_index, count in enumerate(shapes):
        for distribution_index, distribution in enumerate(distributions):
            case_id = f"s{shape_index:02d}-{distribution_index:02d}"
            seed = statistics.derive_seed(suite_seed, count, distribution)
            values = _materialize(distribution, count, random.Random(seed))
            path = inputs / f"{case_id}.f32"
            _write_f32(path, values)
            rounded = _read_f32(path)
            expected[case_id] = tuple(_silu_host_double(value) for value in rounded)
            cases.append({
                "case_id": case_id,
                "shape": [count],
                "distribution": distribution,
                "input": path.relative_to(work).as_posix(),
                "input_sha256": _sha256(path),
            })
    timing_values = _materialize(
        "baseline", 1_048_579,
        random.Random(statistics.derive_seed(suite_seed, "timing")))
    timing_path = inputs / "timing.f32"
    _write_f32(timing_path, timing_values)
    specification = {
        "suite_id": "sealed_silu_boundary_host_double/v1",
        "suite_seed": suite_seed,
        "suite_seed_generated_after_candidate_seal": True,
        "cases": cases,
        "timing_input": timing_path.relative_to(work).as_posix(),
        "timing_input_shape": [len(timing_values)],
        "timing_input_sha256": _sha256(timing_path),
        "timing_blocks": TIMING_BLOCKS,
        "repetitions_per_arm": REPETITIONS_PER_ARM,
        "timing_order_seed": statistics.derive_seed(suite_seed, "order"),
        "contribution_floor": CONTRIBUTION_FLOOR,
        "max_candidates": MAX_CANDIDATES,
        "alpha_selection": ALPHA_SELECTION,
        "e_threshold": E_THRESHOLD,
        "e_process_construction_id": CONSTRUCTION_ID,
    }
    return specification, expected


def _scan_candidate(source: Path) -> dict[str, Any]:
    text = source.read_text(encoding="utf-8")
    diff = "".join(difflib.unified_diff(
        [], text.splitlines(keepends=True), fromfile="/dev/null",
        tofile="b/candidate.hip"))
    scan = reward_hack_scan.scan_unified_diff(diff)
    document = asdict(scan)
    findings = []
    for name, value in document.items():
        if name.endswith("_findings"):
            findings.extend(value)
    document["findings"] = findings
    document["clean"] = not findings
    return document


def _run_sandboxed(
    *, mode: str, source: Path, root: Path, work: Path, spec_path: Path | None,
    gpu_visible: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipts = root / "sandbox-receipts"
    logs = root / "sandbox-logs"
    receipts.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)
    output = work / f"{mode}-result.json"
    receipt_path = receipts / f"{mode}.json"
    temporary = work / "tmp"
    cache_home = work / "cache-home"
    temporary.mkdir(exist_ok=True)
    cache_home.mkdir(exist_ok=True)
    argv = [
        sys.executable, "-m",
        "scripts.kernel_rnd.autokernel.controller.hip_decision_grade_worker",
        "--mode", mode, "--source", str(source), "--work", str(work),
        "--output", str(output),
    ]
    if spec_path is not None:
        argv.extend(("--spec", str(spec_path)))
    policy = sandbox.SandboxPolicy(
        writable_root=str(work), token=f"hipdg{mode}",
        writable_device_paths=("/dev/kfd", "/dev/dri/renderD128")
        if gpu_visible else ())
    environment = dict(os.environ)
    environment["PATH"] = (
        f"{Path(sys.executable).resolve().parent}:"
        f"{Path(sys.executable).parent}:"
        f"{environment.get('PATH', '')}")
    environment.update({
        "PYTORCH_ROCM_ARCH": "gfx90a",
        "TORCH_EXTENSIONS_DIR": str(work / "extension-cache"),
        "TORCHINDUCTOR_CACHE_DIR": str(work / "torchinductor-cache"),
        "TMPDIR": str(temporary),
        "XDG_CACHE_HOME": str(cache_home),
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    if not gpu_visible:
        environment.update({
            "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": "",
            "CUDA_VISIBLE_DEVICES": "",
        })
    wrapped = policy.wrap(argv, receipt_path=str(receipt_path))
    with (logs.joinpath(f"{mode}.stdout").open("wb") as stdout,
          logs.joinpath(f"{mode}.stderr").open("wb") as stderr):
        process = subprocess.Popen(
            wrapped, cwd=Path(__file__).resolve().parents[4], env=environment,
            stdout=stdout, stderr=stderr)
        try:
            returncode = process.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            raise HipDecisionGradeError(f"sandboxed {mode} timed out")
    activation = sandbox.read_receipt(receipt_path)
    sandbox.verify_receipt(activation, policy=policy, pid=process.pid, argv=argv)
    teardown = sandbox.cleanup_cgroup(policy, process.pid)
    if returncode != 0:
        error = logs.joinpath(f"{mode}.stderr").read_text(errors="replace")[-4000:]
        raise HipDecisionGradeError(f"sandboxed {mode} failed ({returncode}): {error}")
    result = json.loads(output.read_text(encoding="utf-8"))
    return result, {"activation": activation, "teardown": teardown}


def _reduce_correctness(
    *, work: Path, specification: Mapping[str, Any], expected: Mapping[str, tuple[float, ...]],
    child: Mapping[str, Any],
) -> dict[str, Any]:
    by_id = {row["case_id"]: row for row in child["cases"]}
    rows = []
    for item in specification["cases"]:
        case_id = item["case_id"]
        row = by_id.get(case_id)
        if row is None:
            raise HipDecisionGradeError(f"candidate omitted sealed case {case_id}")
        first = _read_f32(work / row["output_a"])
        second = _read_f32(work / row["output_b"])
        oracle = expected[case_id]
        if len(first) != len(oracle) or len(second) != len(oracle):
            raise HipDecisionGradeError(f"candidate output length drifted for {case_id}")
        max_abs = max(abs(a - b) for a, b in zip(first, oracle))
        passed = (
            row["input_file_unchanged"] and row["device_input_unchanged"]
            and (work / row["output_a"]).read_bytes() == (work / row["output_b"]).read_bytes()
            and all(math.isfinite(value) for value in first)
            and all(abs(actual - wanted) <= CORRECTNESS_ATOL
                    + CORRECTNESS_RTOL * abs(wanted)
                    for actual, wanted in zip(first, oracle))
        )
        rows.append({"case_id": case_id, "passed": passed, "max_abs_error": max_abs})
    return {
        "oracle": "independent_host_double_stable_silu/v1",
        "atol": CORRECTNESS_ATOL,
        "rtol": CORRECTNESS_RTOL,
        "cases": rows,
        "passed": sum(row["passed"] for row in rows),
        "total": len(rows),
        "all_passed": all(row["passed"] for row in rows),
        "two_distinct_output_poisons": ["nan", -12345.25],
        "bitwise_repeatability_required": True,
    }


def _reduce_timing(child: Mapping[str, Any]) -> dict[str, Any]:
    blocks = child["blocks"]
    ranked_durations = tuple(
        duration
        for row in blocks
        for duration in (
            float(row["candidate_measured_duration_ns"]),
            float(row["anchor_measured_duration_ns"]),
        )
    )
    duration_checks = tuple(
        devices.GFX90A_RANKED_DURATION_ADMISSION.check(
            (duration,), device_id="ROCm0")
        for duration in ranked_durations)
    duration_all_passed = all(check.outcome == "PASS" for check in duration_checks)
    effects = [float(row["anchor_ns"]) / float(row["candidate_ns"]) - 1.0
               for row in blocks]
    speedups = [effect + 1.0 for effect in effects]
    e_run = statistics.run_e_process(
        effects, construction=statistics.select_construction(CONSTRUCTION_ID),
        hypothesis=statistics.HYPOTHESIS_IMPROVEMENT, margin=0.0,
        threshold=E_THRESHOLD)
    median_speedup = statistics.median(speedups)
    return {
        "provider": child["provider"],
        "blocks": blocks,
        "block_count": len(blocks),
        "repetitions_per_arm": child["repetitions_per_arm"],
        "ranked_duration_admission": {
            "all_arms_passed": duration_all_passed,
            "checks": [{"outcome": check.outcome,
                        "reasons": list(check.reasons)}
                       for check in duration_checks],
            "minimum_ns": devices.GFX90A_RANKED_DURATION_ADMISSION.min_window_ns,
            "evidence_ref": devices.GFX90A_RANKED_DURATION_ADMISSION.evidence_ref,
            "minimum_observed_ns": min(ranked_durations),
        },
        "median_speedup": median_speedup,
        "median_relative_effect": statistics.median(effects),
        "mad_relative_effect": statistics.mad(effects),
        "e_process": e_run.to_dict(),
        "threshold_derivation": {
            "max_candidates": MAX_CANDIDATES,
            "alpha_selection": ALPHA_SELECTION,
            "threshold": E_THRESHOLD,
            "formula": "alpha_selection=1/max_candidates; threshold=1/alpha_selection",
        },
        "contribution_floor": CONTRIBUTION_FLOOR,
        "candidate_beats_exact_provider": (
            duration_all_passed
            and median_speedup >= 1.0 + CONTRIBUTION_FLOOR and e_run.crossed),
    }


def run(
    *, candidate_source: str | Path, output_root: str | Path, campaign_id: str,
    arena_root: str | Path = DEFAULT_ARENA_ROOT, task_id: str = DEFAULT_TASK,
    claim_journal: str | Path = DEFAULT_CLAIM_JOURNAL,
    visible_device: str = "0", claim_timeout_s: float = 3600.0,
    counter_expectation: Mapping[str, Any] | None = None,
    counter_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    started = _utc_now()
    declared_counter_expectation = (
        _counter_expectation(counter_expectation)
        if counter_expectation is not None else None)
    if not re.fullmatch(r"[a-z][a-z0-9_.-]{2,95}", campaign_id):
        raise HipDecisionGradeError("campaign_id is not a safe governed identifier")
    root = Path(output_root).resolve()
    if root.exists():
        raise HipDecisionGradeError("output_root already exists; decision panels never resume")
    root.mkdir(parents=True)
    work = root / "candidate-work"
    work.mkdir()
    source = Path(candidate_source).resolve()
    if source.is_symlink() or not source.is_file() or source.suffix != ".hip":
        raise HipDecisionGradeError("candidate must be a regular .hip source")
    source_sha = _sha256(source)
    sealed_at = _utc_now()
    task = hip.audit_task(Path(arena_root), task_id)
    toolchain = hip.toolchain_identity()
    hardware = dict(arena_adapter.detect_gfx_arch())
    if hardware.get("architectures") != [hip.TARGET_GFX_ARCH]:
        raise HipDecisionGradeError("hardware did not resolve exactly gfx90a")
    static_scan = _scan_candidate(source)
    if not static_scan["clean"]:
        raise HipDecisionGradeError("candidate failed the executable C6 static scan")

    suite_seed = secrets.token_hex(32)
    specification, expected = _create_suite(work, suite_seed)
    spec_path = work / "sealed-suite.json"
    _atomic_json(spec_path, specification)
    counter_declaration = None
    if declared_counter_expectation is not None:
        counter_declaration = {
            "schema": COUNTER_EXPECTATION_SCHEMA,
            "declared_at": _utc_now(), "task_id": task_id,
            "campaign_id": campaign_id, "candidate_source_sha256": source_sha,
            "workload_sha256": specification["timing_input_sha256"],
            "expectation": declared_counter_expectation,
            "expectation_sha256": _canonical_sha256(declared_counter_expectation),
        }
        # Persist the direction before either native GPU measurement window opens.
        _atomic_json(root / "counter-expectation.json", counter_declaration)
    compile_result, compile_sandbox = _run_sandboxed(
        mode="compile", source=source, root=root, work=work,
        spec_path=None, gpu_visible=False)
    if not compile_result.get("extension_has_forward_out"):
        raise HipDecisionGradeError("compiled candidate lacks forward_out")

    correctness_child, correctness_window = hip._measurement_window(
        phase="sealed_correctness", task_id=task_id, campaign_id=campaign_id,
        output_root=root, claim_journal=Path(claim_journal),
        visible_device=visible_device, claim_timeout_s=claim_timeout_s,
        action=lambda: _run_sandboxed(
            mode="correctness", source=source, root=root, work=work,
            spec_path=spec_path, gpu_visible=True))
    correctness_payload, correctness_sandbox = correctness_child
    correctness = _reduce_correctness(
        work=work, specification=specification, expected=expected,
        child=correctness_payload)

    timing_child, timing_window = hip._measurement_window(
        phase="exact_provider_timing", task_id=task_id, campaign_id=campaign_id,
        output_root=root, claim_journal=Path(claim_journal),
        visible_device=visible_device, claim_timeout_s=claim_timeout_s,
        action=lambda: _run_sandboxed(
            mode="timing", source=source, root=root, work=work,
            spec_path=spec_path, gpu_visible=True))
    timing_payload, timing_sandbox = timing_child
    timing = _reduce_timing(timing_payload)
    anchor_identity_sha256 = _canonical_sha256(timing_payload.get("provider"))
    counter_gate = evaluate_counter_gate(
        expectation=declared_counter_expectation, evidence=counter_evidence,
        candidate_source_sha256=source_sha, task_id=task_id, campaign_id=campaign_id,
        workload_sha256=str(specification["timing_input_sha256"]),
        anchor_identity_sha256=anchor_identity_sha256)
    source_unchanged = _sha256(source) == source_sha
    integrity_clean = (
        source_unchanged and static_scan["clean"]
        and correctness["all_passed"]
        and compile_sandbox["teardown"]["verified_empty"]
        and correctness_sandbox["teardown"]["verified_empty"]
        and timing_sandbox["teardown"]["verified_empty"])
    decision = candidate_decision(
        correctness_passed=correctness["all_passed"], integrity_clean=integrity_clean,
        timing=timing, counter_gate=counter_gate)
    belief_measurements = [
        {"measurement_id": "hip_sealed_correctness_pass_rate",
         "metric": "autokernel_hip_sealed_correctness_pass_rate",
         "value": correctness["passed"] / correctness["total"], "unit": "fraction",
         "metric_direction": "higher_better", "category": "CANDIDATE",
         "claim": "Fraction of sealed hostile host-double SiLU cases passed",
         "reps": correctness["total"], "reps_basis": "sealed_hostile_cases"},
    ]
    if decision["speedup_claim_allowed"]:
        belief_measurements.append(
            {"measurement_id": "hip_exact_provider_speedup",
             "metric": "autokernel_hip_speedup_vs_exact_torch_rocm_compile",
             "value": timing["median_speedup"], "unit": "ratio",
             "metric_direction": "higher_better", "category": "CANDIDATE",
             "claim": "Median paired-block speedup over exact Torch-ROCm-compile SiLU",
             "reps": timing["block_count"], "reps_basis": "paired_randomized_blocks"})
    receipt = _self_hash({
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "campaign_id": campaign_id,
        "status": "complete" if integrity_clean else "failed_integrity",
        "started_at": started,
        "ended_at": _utc_now(),
        "producer": {"producer_id": PRODUCER_ID, "path": str(Path(__file__).resolve()),
                     "sha256": _sha256(Path(__file__).resolve())},
        "task": task.to_dict(),
        "hardware": hardware,
        "toolchain": toolchain,
        "candidate": {"source": str(source), "sha256": source_sha,
                      "sealed_at": sealed_at, "unchanged_at_terminal": source_unchanged},
        "sealed_suite": {**specification, "manifest_sha256": _sha256(spec_path),
                         "exact_shapes_disclosed_after_candidate_seal": True},
        "correctness": correctness,
        "integrity": {"static_scan": static_scan, "clean": integrity_clean,
                      "candidate_never_received_expected_outputs": True,
                      "sandbox": {"compile": compile_sandbox,
                                  "correctness": correctness_sandbox,
                                  "timing": timing_sandbox}},
        "timing": timing,
        "counter_declaration": counter_declaration,
        "counter_gate": counter_gate,
        "decision": {
            **decision,
            "release_or_promotion_authority": False,
            "experimental_llama_integration_required_before_any_release": True,
        },
        "measurement_windows": [correctness_window, timing_window],
        "belief_measurements": belief_measurements,
        "constraints": {"production_tree_touched": False,
                        "frozen_kernel_built": False,
                        "promotion_authority": False,
                        "counter_capture_performed": False,
                        "end_to_end_counter_gate_adopted": False,
                        "live_controller_activated": False,
                        "shared_rocm_mutated": False},
    })
    _atomic_json(root / "receipt.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-source", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--arena-root", default=str(DEFAULT_ARENA_ROOT))
    parser.add_argument("--task-id", default=DEFAULT_TASK)
    parser.add_argument("--claim-journal", default=str(DEFAULT_CLAIM_JOURNAL))
    parser.add_argument("--visible-device", default="0")
    parser.add_argument("--claim-timeout-seconds", type=float, default=3600.0)
    parser.add_argument(
        "--counter-expectation",
        help="JSON declaration of the pre-run native counter and expected direction")
    parser.add_argument(
        "--counter-evidence",
        help="JSON receipt binding raw native counter CSVs for candidate and anchor")
    args = parser.parse_args(argv)
    def load_json_object(path_value: str | None, label: str) -> Mapping[str, Any] | None:
        if path_value is None:
            return None
        value = json.loads(Path(path_value).read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise HipDecisionGradeError(f"{label} must be a JSON object")
        return value
    with hip._graceful_signals():
        receipt = run(
            candidate_source=args.candidate_source, output_root=args.output_root,
            campaign_id=args.campaign_id, arena_root=args.arena_root,
            task_id=args.task_id, claim_journal=args.claim_journal,
            visible_device=args.visible_device,
            claim_timeout_s=args.claim_timeout_seconds,
            counter_expectation=load_json_object(
                args.counter_expectation, "counter expectation"),
            counter_evidence=load_json_object(
                args.counter_evidence, "counter evidence"))
    speedup_allowed = receipt["decision"]["speedup_claim_allowed"]
    print(json.dumps({
        "status": receipt["status"], "campaign_id": receipt["campaign_id"],
        "receipt_sha256": receipt["receipt_sha256"],
        "rankable": receipt["decision"]["rankable_against_exact_task_local_provider"],
        "speedup_claim_allowed": speedup_allowed,
        "counter_gate_status": receipt["counter_gate"]["status"],
        "median_speedup": (receipt["timing"]["median_speedup"]
                           if speedup_allowed else None),
    }, sort_keys=True))
    return 0 if receipt["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
