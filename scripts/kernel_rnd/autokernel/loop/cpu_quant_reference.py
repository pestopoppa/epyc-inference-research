"""Independent output check for fixed CPU Q4_K/Q5_K/Q8_0 matmul probes.

The probe runs the candidate ggml graph and emits the *stored quantized bytes*.
This module decodes those bytes without ggml and computes scalar dot products.
It checks numerical output, not whether IQK or any other optimized dispatch fired.
The latter needs a separate runtime dispatch witness before a specialized path
claim can be made. This deliberately is not a generic precision policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping
import json
import math
import os
import re
import struct
import subprocess
import tempfile


MARKER = "AK_CPU_QUANT_REFERENCE_V1"
METRIC_MARKER = "AK_CPU_QUANT_METRIC_V1"
PROBE = Path(__file__).with_name("cpu_quant_reference_probe.cpp")
QUANTS = ("Q4_K", "Q5_K", "Q8_0")
OPS = ("MUL_MAT", "MUL_MAT_ID")
K, ROWS, TOKENS = 256, 40, 2
ROW_BYTES = {"Q4_K": 144, "Q5_K": 176, "Q8_0": 272}
# Also constrain the encoding against the fixed F32 input. Otherwise a broken
# quantizer that writes all zeroes could agree with its own all-zero matmul.
# Empirical fixed-fixture bounds, not a general quant accuracy policy. On the
# retained experimental anchor-gen-014 (2026-09-17), max |decoded-source|
# across all 40/80 rows was Q4_K 0.0438862, Q5_K 0.000228882,
# Q8_0 0.00344849. Rounded allowances here preserve roughly 2-3x headroom
# for Q4_K/Q8_0 and a small absolute allowance for Q5_K, while zeroed rows
# remain decisively wrong.
QUANT_ABS_TOL = {"Q4_K": 0.10, "Q5_K": 0.005, "Q8_0": 0.01}
# On that same anchor, max scalar-vs-graph |error| across both ops was
# Q4_K 0.00516272, Q5_K 0.00492007, Q8_0 0.000077963. 0.01 is <2.1x
# the worst observed absolute error. Relative allowance matters only for
# large outputs; near-zero outputs remain under the absolute bound.
ABS_TOL = 0.01
REL_TOL = 0.005


@dataclass(frozen=True)
class QuantResult:
    status: Literal["pass", "wrong", "unavailable"]
    reason: str = ""
    detail: str = ""
    # A graph result does not establish which optimized CPU implementation ran.
    path_verified: bool = False


def _activation(token: int, column: int) -> float:
    for slot in range(4):
        if column == 3 + token * 11 + slot * 64:
            return -0.5 if slot & 1 else 1.0
    return 0.0


def _source_weight(expert: int, row: int, column: int) -> float:
    return ((column * 13 + row * 7 + expert * 17) % 31 - 15) / 16.0


def _scale_min(scales: bytes, group: int) -> tuple[int, int]:
    if group < 4:
        return scales[group] & 63, scales[group + 4] & 63
    return ((scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4),
            (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4))


def _decode_row(quant: str, row: bytes) -> tuple[float, ...]:
    if len(row) != ROW_BYTES[quant]:
        raise ValueError("quant row byte length mismatch")
    if quant == "Q8_0":
        values = []
        for offset in range(0, len(row), 34):
            scale = struct.unpack_from("<e", row, offset)[0]
            values.extend(scale * value for value in
                          struct.unpack_from("<32b", row, offset + 2))
        return tuple(values)
    d, dmin = struct.unpack_from("<ee", row)
    scales = row[4:16]
    if quant == "Q4_K":
        high = None
        packed = row[16:]
    else:
        high = row[16:48]
        packed = row[48:]
    values = []
    for group64 in range(4):
        for half in range(2):
            group = group64 * 2 + half
            scale, minimum = _scale_min(scales, group)
            for column32 in range(32):
                packed_value = packed[group64 * 32 + column32]
                quant_value = (packed_value & 15) if half == 0 else (packed_value >> 4)
                if high is not None and high[column32] & (1 << group):
                    quant_value += 16
                values.append(d * scale * quant_value - dmin * minimum)
    return tuple(values)


def _reference(quant: str, op: str, rows: Mapping[tuple[int, int], bytes],
               token: int, row: int) -> float:
    expert = (1, 0)[token] if op == "MUL_MAT_ID" else 0
    weights = _decode_row(quant, rows[expert, row])
    return math.fsum(weight * _activation(token, column)
                     for column, weight in enumerate(weights))


def _parse_and_compare(output: str, quant: str, op: str) -> QuantResult:
    lines = output.splitlines()
    headers = [i for i, line in enumerate(lines) if line.startswith(MARKER + " ")]
    if len(headers) != 1:
        raise ValueError("probe marker missing or duplicated")
    parts = lines[headers[0]].split()
    if len(parts) != 7 or parts[1:6] != [quant, op, str(K), str(ROWS), str(TOKENS)]:
        raise ValueError("probe identity or shape mismatch")
    if int(parts[6]) != ROW_BYTES[quant]:
        raise ValueError("probe row-byte layout mismatch")
    experts = 2 if op == "MUL_MAT_ID" else 1
    rows: dict[tuple[int, int], bytes] = {}
    observed: dict[tuple[int, int], float] = {}
    for line in lines[headers[0] + 1:]:
        item = line.split()
        if len(item) != 4:
            raise ValueError("malformed probe line")
        if item[0] == "A":
            expert, row = int(item[1]), int(item[2])
            if not (0 <= expert < experts and 0 <= row < ROWS) or (expert, row) in rows:
                raise ValueError("out-of-range or duplicate quant row")
            if not re.fullmatch(r"[0-9a-f]+", item[3]):
                raise ValueError("invalid quant row hex")
            rows[expert, row] = bytes.fromhex(item[3])
        elif item[0] == "O":
            token, row = int(item[1]), int(item[2])
            if not (0 <= token < TOKENS and 0 <= row < ROWS) or (token, row) in observed:
                raise ValueError("out-of-range or duplicate output")
            value = float.fromhex(item[3])
            if not math.isfinite(value):
                raise ValueError("non-finite output")
            observed[token, row] = value
        else:
            raise ValueError("unknown probe line")
    if len(rows) != experts * ROWS or len(observed) != TOKENS * ROWS:
        raise ValueError("incomplete probe payload")
    max_quant_abs_error = 0.0
    for (expert, row), encoded in rows.items():
        for column, decoded in enumerate(_decode_row(quant, encoded)):
            source = _source_weight(expert, row, column)
            quant_abs_error = abs(decoded - source)
            if not math.isfinite(decoded) or quant_abs_error > QUANT_ABS_TOL[quant]:
                return QuantResult("wrong", f"{quant} quant encoding mismatch "
                                   f"expert={expert} row={row} column={column}",
                                   f"decoded={decoded:.9g}, source={source:.9g}, "
                                   f"abs_tol={QUANT_ABS_TOL[quant]}")
            max_quant_abs_error = max(max_quant_abs_error, quant_abs_error)
    max_output_abs_error = 0.0
    max_output_limit_fraction = 0.0
    for token in range(TOKENS):
        for row in range(ROWS):
            actual = observed[token, row]
            expected = _reference(quant, op, rows, token, row)
            output_abs_error = abs(actual - expected)
            if not math.isclose(actual, expected, abs_tol=ABS_TOL, rel_tol=REL_TOL):
                return QuantResult("wrong", f"{quant} {op} mismatch token={token} row={row}",
                                   f"actual={actual:.9g}, scalar={expected:.9g}, "
                                   f"abs_tol={ABS_TOL}, rel_tol={REL_TOL}")
            # This is the exact combined limit used by math.isclose above;
            # max-abs alone is telemetry, not a separate acceptance rule.
            limit = max(ABS_TOL, REL_TOL * max(abs(actual), abs(expected)))
            max_output_abs_error = max(max_output_abs_error, output_abs_error)
            max_output_limit_fraction = max(max_output_limit_fraction,
                                            output_abs_error / limit)
    metric = {"schema": "epyc.autokernel.cpu_quant_metric.v1", "quant": quant,
              "op": op, "outputs": TOKENS * ROWS,
              "max_quant_abs_error": max_quant_abs_error,
              "quant_abs_tol": QUANT_ABS_TOL[quant],
              "max_output_abs_error": max_output_abs_error,
              "max_output_limit_fraction": max_output_limit_fraction,
              "output_abs_tol": ABS_TOL, "output_rel_tol": REL_TOL}
    return QuantResult("pass", f"{quant} {op}: {TOKENS * ROWS} scalar outputs agree; "
                       "optimized dispatch not proven",
                       METRIC_MARKER + " " + json.dumps(metric, sort_keys=True))


def check_cpu_quant_suite(build_dir: Path, source_root: Path, *,
                          launch_env: Mapping[str, str] | None = None,
                          topology_prefix: tuple[str, ...] = (),
                          quants: tuple[str, ...] = QUANTS,
                          ops: tuple[str, ...] = OPS) -> QuantResult:
    """Compile once and run requested candidate-bound quant cases.

    Infrastructure/malformed-output failures are `unavailable`, never numerical
    `wrong`. The caller must separately establish dispatch-path engagement.
    """
    if not quants or not ops or any(q not in QUANTS for q in quants) or \
            any(op not in OPS for op in ops):
        raise ValueError("unsupported or empty quant/op selection")
    build_dir, source_root = Path(build_dir), Path(source_root)
    lib_dir = build_dir / "bin"
    needed = (lib_dir / "libggml.so", lib_dir / "libggml-base.so",
              lib_dir / "libggml-cpu.so", source_root / "ggml/include/ggml.h", PROBE)
    missing = [str(path) for path in needed if not path.is_file()]
    if missing:
        return QuantResult("unavailable", "candidate ggml library/header or probe missing",
                           ", ".join(missing))
    env = dict(os.environ if launch_env is None else launch_env)
    env["LD_LIBRARY_PATH"] = str(lib_dir) + (":" + env["LD_LIBRARY_PATH"]
                                         if env.get("LD_LIBRARY_PATH") else "")
    try:
        with tempfile.TemporaryDirectory(prefix="ak-cpu-quant-ref-") as temp:
            binary = Path(temp) / "quant-reference-probe"
            command = ["c++", "-std=c++17", "-O2", "-I", str(source_root / "ggml/include"),
                       str(PROBE), "-L", str(lib_dir), "-Wl,-rpath," + str(lib_dir),
                       "-lggml-cpu", "-lggml-base", "-lggml", "-o", str(binary)]
            built = subprocess.run(command, capture_output=True, text=True, timeout=120, env=env)
            if built.returncode:
                return QuantResult("unavailable", "CPU quant probe compile failed",
                                   built.stderr[-2000:])
            metrics = []
            for quant in quants:
                for op in ops:
                    run = subprocess.run([*topology_prefix, str(binary), quant, op],
                                         capture_output=True, text=True, timeout=120, env=env)
                    if run.returncode:
                        return QuantResult("unavailable", f"{quant} {op} probe did not complete",
                                           f"exit={run.returncode}; {run.stderr[-1800:]}")
                    try:
                        result = _parse_and_compare(run.stdout, quant, op)
                    except (ValueError, KeyError, OverflowError) as exc:
                        return QuantResult("unavailable", f"{quant} {op} probe output invalid",
                                           str(exc))
                    if result.status != "pass":
                        return result
                    metrics.append(result.detail)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return QuantResult("unavailable", "CPU quant probe infrastructure fault", str(exc))
    return QuantResult("pass", f"{len(quants) * len(ops)} quant/op cases passed; "
                       "optimized dispatch not proven", "\n".join(metrics))


__all__ = ["QuantResult", "check_cpu_quant_suite"]
