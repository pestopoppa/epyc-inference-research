"""Independent RMS_NORM check for the `cpu_norm_rowsplit` route (DS41 hc_mixes norm).

The probe (`cpu_norm_reference_probe.cpp`) runs the candidate's ggml graph on a pinned
8-thread team and prints the raw float32 output bits. This module regenerates the same
inputs without ggml and applies two independent verdicts to every output:

1. **Bit identity with HEAD's arithmetic.** HEAD's `ggml_compute_forward_rms_norm_f32`
   rounds each square to float, accumulates the squares left to right in a `ggml_float`
   (double), rounds `sum/ne00` to float, forms `1.0f/sqrtf(mean + eps)` in float, and
   writes `x * scale` (fused: `(x * scale) * w`). Python floats are IEEE doubles and
   `array('f', ...)` rounds to nearest float, so every step is emulated exactly: +, *, /
   and sqrt of floats evaluated in double and rounded once to float are correctly rounded
   (53 >= 2*24 + 2). The route promises a bit-exact split, so ANY differing bit is
   `wrong`. The emulation was checked bit-for-bit against the DS41 anchor build
   (anchor-gen-001, cafb59c3b) on every case.

2. **A float64 reference with an analytic bound.** y = x * w / sqrt(sum(x^2)/n + eps),
   with the squares exact in double (24-bit significands) and summed by `math.fsum`.
   HEAD-order float arithmetic is within 5.5 u of it (u = 2^-24): one rounding each for
   the square (relative, all terms positive), the double sum (n * 2^-53, negligible for
   n <= 2^16), the float mean, `mean + eps`, sqrtf (half the argument error plus one
   rounding), the reciprocal, `x * scale` and the fused `* w`. The bound used is
   16 u = 2^-20 relative (about 3x margin). This verdict is independent of the emulation.

What bit identity can and cannot see: the squares are positive, so the double sum's
relative error is at most n * 2^-53 ~ 2^-38 whatever the order, and rounding the mean to
float hides a reordered DOUBLE accumulation except near a rounding boundary. The
summation order is therefore held textually by the route's forbidden-added pattern;
this check catches what changes float bits: a float accumulator, partial sums that do not
cover the row, a different scale formula or product association, a skipped, duplicated
or misplaced column segment, a wrong weight row, and an in-place race (repeated runs).

Every case runs on 8 threads. Rows < 8 exercise the within-row split; the wide cases keep
the row split, the solo single-row path and the 3-D/4-D row walk covered.
"""
from __future__ import annotations

from array import array
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


MARKER = "AK_CPU_NORM_REFERENCE_V1"
METRIC_MARKER = "AK_CPU_NORM_METRIC_V1"
PROBE = Path(__file__).with_name("cpu_norm_reference_probe.cpp")
THREADS = 8
SEED = 20260927
VIEW_PAD = 13      # must equal the probe's VIEW_PAD
VIEW_OFFSET = 5    # must equal the probe's VIEW_OFFSET (floats)
U = 2.0 ** -24
REL_BOUND = 16 * U
MODES = ("plain", "inplace", "view", "fused", "fused_bcast")
_MASK = (1 << 64) - 1


@dataclass(frozen=True)
class NormCase:
    name: str
    mode: str
    ne: tuple[int, int, int, int]
    eps: float
    reps: int = 3

    @property
    def rows(self) -> int:
        return self.ne[1] * self.ne[2] * self.ne[3]


# Narrow (rows < THREADS) first: the DS41 hc_mixes shapes are [20480, 2..3].
CASES = (
    NormCase("hc_mixes_nt3", "plain", (20480, 3, 1, 1), 1e-6),
    NormCase("hc_mixes_nt2", "plain", (20480, 2, 1, 1), 1e-6),
    NormCase("long_single_row", "plain", (20480, 1, 1, 1), 1e-6),
    NormCase("odd_len_narrow", "plain", (1000, 7, 1, 1), 1e-6),
    NormCase("five_rows", "plain", (4096, 5, 1, 1), 0.5),
    NormCase("narrow_3d", "plain", (640, 2, 3, 1), 1e-6),
    # In-place: a split must not scale a segment while another thread still sums the row.
    NormCase("inplace_narrow", "inplace", (20480, 3, 1, 1), 1e-6, reps=16),
    NormCase("view_narrow", "view", (5000, 3, 1, 1), 1e-6),
    # RMS_NORM + MUL fused into ggml_compute_forward_rms_norm_mul_fused.
    NormCase("fused_bcast_narrow", "fused_bcast", (20480, 3, 1, 1), 1e-6),
    NormCase("fused_full_narrow", "fused", (1000, 7, 1, 1), 1e-6),
    # Wide (rows >= THREADS) and the single-row solo candidate (<= 4096 elements).
    NormCase("wide_rows", "plain", (256, 40, 1, 1), 1e-6),
    NormCase("wide_odd", "plain", (1025, 17, 1, 1), 1e-4),
    NormCase("wide_4d", "plain", (64, 5, 4, 3), 1e-6),
    NormCase("eight_rows_4d", "plain", (96, 2, 2, 2), 1e-6),
    NormCase("fused_bcast_wide", "fused_bcast", (256, 40, 1, 1), 1e-6),
    NormCase("solo_row", "plain", (2048, 1, 1, 1), 1e-6),
)


@dataclass(frozen=True)
class NormResult:
    status: Literal["pass", "wrong", "unavailable"]
    reason: str = ""
    detail: str = ""


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _mix(z: int) -> int:
    z = (z + 0x9E3779B97F4A7C15) & _MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
    return z ^ (z >> 31)


def _value(seed: int, index: int, exp_lo: int, exp_span: int) -> float:
    """Bit-identical to the probe's `value()`: exact 24-bit significand floats."""
    h = _mix((seed * 0x100000001B3 + index) & _MASK)
    mantissa = (h & 0x7FFFFF) | 0x800000
    exponent = (h >> 24) % exp_span - exp_lo
    magnitude = math.ldexp(mantissa, exponent - 23)
    return -magnitude if (h >> 40) & 1 else magnitude


def _fnv1a(data: bytes, h: int = 0xCBF29CE484222325) -> int:
    for byte in data:
        h = ((h ^ byte) * 0x100000001B3) & _MASK
    return h


def _inputs(case: NormCase, seed: int = SEED) -> tuple[list[list[float]], list[list[float]] | None, int]:
    """Rows of the normalized operand, weight rows aligned to them, and the input hash."""
    ne0, ne1, ne2, ne3 = case.ne
    src_ne0 = ne0 + VIEW_PAD if case.mode == "view" else ne0
    src = [_value(seed, i, 12, 25) for i in range(src_ne0 * case.rows)]
    offset = VIEW_OFFSET if case.mode == "view" else 0
    rows = [src[r * src_ne0 + offset:r * src_ne0 + offset + ne0] for r in range(case.rows)]
    weights = None
    w_flat: list[float] = []
    if case.mode in ("fused", "fused_bcast"):
        count = ne0 if case.mode == "fused_bcast" else ne0 * case.rows
        w_flat = [_value(seed + 1, i, 4, 9) for i in range(count)]
        weights = [w_flat[0:ne0] if case.mode == "fused_bcast" else
                   w_flat[r * ne0:(r + 1) * ne0] for r in range(case.rows)]
    digest = _fnv1a(array("f", src).tobytes())
    digest = _fnv1a(array("f", w_flat).tobytes(), digest)
    return rows, weights, digest


def head_semantics(row: list[float], eps: float, weight: list[float] | None = None) -> array:
    """HEAD's per-row arithmetic, bit for bit (see the module docstring)."""
    squares = array("f", [x * x for x in row])       # float x*x, rounded once
    total = 0.0                                      # ggml_float, left to right
    for square in squares:
        total += square
    eps = _f32(eps)
    mean = _f32(total / len(row))                    # (float)(sum/ne00)
    scale = _f32(1.0 / _f32(math.sqrt(_f32(mean + eps))))
    scaled = array("f", [x * scale for x in row])
    if weight is None:
        return scaled
    return array("f", [y * w for y, w in zip(scaled, weight)])


def float64_reference(row: list[float], eps: float,
                      weight: list[float] | None = None) -> list[float]:
    mean = math.fsum(x * x for x in row) / len(row)  # squares are exact in double
    inv = 1.0 / math.sqrt(mean + _f32(eps))
    if weight is None:
        return [x * inv for x in row]
    return [x * inv * w for x, w in zip(row, weight)]


def compare(case: NormCase, output: str, seed: int = SEED) -> NormResult:
    """Verdict for one probe run; malformed output raises ValueError (-> unavailable)."""
    lines = output.splitlines()
    headers = [i for i, line in enumerate(lines) if line.startswith(MARKER + " ")]
    if len(headers) != 1:
        raise ValueError("probe marker missing or duplicated")
    expected_header = (f"{MARKER} {case.mode} {' '.join(map(str, case.ne))} "
                       f"{_f32_bits(case.eps):08x} {THREADS} {case.reps} {seed}")
    if lines[headers[0]] != expected_header:
        raise ValueError("probe identity or shape mismatch")
    rows, weights, input_hash = _inputs(case, seed)
    digests: dict[int, int] = {}
    observed: dict[int, bytes] = {}
    seen_input = None
    for line in lines[headers[0] + 1:]:
        item = line.split()
        if item[:1] == ["I"] and len(item) == 2 and seen_input is None:
            seen_input = int(item[1], 16)
        elif item[:1] == ["D"] and len(item) == 3:
            rep = int(item[1])
            if not 0 <= rep < case.reps or rep in digests:
                raise ValueError("out-of-range or duplicate repetition digest")
            digests[rep] = int(item[2], 16)
        elif item[:1] == ["O"] and len(item) == 3:
            row = int(item[1])
            if not 0 <= row < case.rows or row in observed or \
                    not re.fullmatch(r"[0-9a-f]+", item[2]) or len(item[2]) != 8 * case.ne[0]:
                raise ValueError("out-of-range, duplicate or malformed output row")
            observed[row] = bytes.fromhex(item[2])
        else:
            raise ValueError("unknown probe line")
    if seen_input != input_hash:
        raise ValueError("probe inputs differ from the reference generator (fixture drift)")
    if len(digests) != case.reps or len(observed) != case.rows:
        raise ValueError("incomplete probe payload")
    expected_bytes = b""
    max_rel = 0.0
    for row in range(case.rows):
        weight = weights[row] if weights is not None else None
        expected = head_semantics(rows[row], case.eps, weight)
        expected_bytes += expected.tobytes()
        actual = array("f")
        actual.frombytes(observed[row])
        if actual.tobytes() != expected.tobytes():
            column = next(i for i, (a, e) in enumerate(zip(actual, expected))
                          if _f32_bits(a) != _f32_bits(e))
            differing = sum(_f32_bits(a) != _f32_bits(e) for a, e in zip(actual, expected))
            return NormResult("wrong", f"{case.name}: output is not bit-identical to HEAD's "
                              f"arithmetic (row={row} column={column}, {differing} of "
                              f"{case.ne[0]} columns differ)",
                              f"actual={actual[column]!r}, head={expected[column]!r}")
        reference = float64_reference(rows[row], case.eps, weight)
        for column, (value, ref) in enumerate(zip(actual, reference)):
            if not math.isfinite(value) or abs(value - ref) > REL_BOUND * abs(ref):
                return NormResult("wrong", f"{case.name}: float64 reference exceeded "
                                  f"(row={row} column={column})",
                                  f"actual={value!r}, float64={ref!r}, rel_bound={REL_BOUND}")
            if ref:
                max_rel = max(max_rel, abs(value - ref) / abs(ref))
    want = _fnv1a(expected_bytes)
    bad = sorted(rep for rep, digest in digests.items() if digest != want)
    if bad:
        return NormResult("wrong", f"{case.name}: repetitions {bad} of {case.reps} differ from "
                          "HEAD's bits (first repetition matched: nondeterministic, e.g. an "
                          "in-place race)")
    metric = {"schema": "epyc.autokernel.cpu_norm_metric.v1", "case": case.name,
              "mode": case.mode, "ne": list(case.ne), "threads": THREADS,
              "reps": case.reps, "outputs": case.rows * case.ne[0], "bit_identical": True,
              "max_rel_error_vs_float64": max_rel, "rel_bound": REL_BOUND}
    return NormResult("pass", f"{case.name}: bit-identical to HEAD arithmetic",
                      METRIC_MARKER + " " + json.dumps(metric, sort_keys=True))


def probe_argv(binary: Path, case: NormCase, seed: int = SEED) -> list[str]:
    return [str(binary), case.mode, *map(str, case.ne), f"{_f32_bits(case.eps):08x}",
            str(THREADS), str(case.reps), str(seed)]


def check_rms_norm_suite(build_dir: Path, source_root: Path, *,
                         launch_env: Mapping[str, str] | None = None,
                         topology_prefix: tuple[str, ...] = (),
                         cases: tuple[NormCase, ...] = CASES) -> NormResult:
    """Compile the probe once against the candidate and run every case.

    Infrastructure and malformed output are `unavailable`, never numerical `wrong`.
    This checks numbers only; the route witness separately proves the candidate DSO's
    rms_norm entries executed.
    """
    if not cases or any(case.mode not in MODES for case in cases):
        raise ValueError("unsupported or empty RMS_NORM case selection")
    build_dir, source_root = Path(build_dir), Path(source_root)
    lib_dir = build_dir / "bin"
    needed = (lib_dir / "libggml.so", lib_dir / "libggml-base.so",
              lib_dir / "libggml-cpu.so", source_root / "ggml/include/ggml.h", PROBE)
    missing = [str(path) for path in needed if not path.is_file()]
    if missing:
        return NormResult("unavailable", "candidate ggml library/header or probe missing",
                          ", ".join(missing))
    env = dict(os.environ if launch_env is None else launch_env)
    env["LD_LIBRARY_PATH"] = str(lib_dir) + (":" + env["LD_LIBRARY_PATH"]
                                         if env.get("LD_LIBRARY_PATH") else "")
    metrics = []
    try:
        with tempfile.TemporaryDirectory(prefix="ak-cpu-norm-ref-") as temp:
            binary = Path(temp) / "norm-reference-probe"
            command = ["c++", "-std=c++17", "-O2", "-I", str(source_root / "ggml/include"),
                       str(PROBE), "-L", str(lib_dir), "-Wl,-rpath," + str(lib_dir),
                       "-lggml-cpu", "-lggml-base", "-lggml", "-o", str(binary)]
            # The loop's toolchain env compiles; only the RUN uses the candidate launch env
            # (same split as cpu_quant_reference, DS41 run 10i).
            built = subprocess.run(command, capture_output=True, text=True, timeout=120,
                                   env=dict(os.environ))
            if built.returncode:
                return NormResult("unavailable", "CPU RMS_NORM probe compile failed",
                                  built.stderr[-2000:])
            for case in cases:
                run = subprocess.run([*topology_prefix, *probe_argv(binary, case)],
                                     capture_output=True, text=True, timeout=120, env=env)
                if run.returncode:
                    return NormResult("unavailable", f"{case.name} probe did not complete",
                                      f"exit={run.returncode}; {run.stderr[-1800:]}")
                try:
                    result = compare(case, run.stdout)
                except (ValueError, KeyError, OverflowError, StopIteration) as exc:
                    return NormResult("unavailable", f"{case.name} probe output invalid",
                                      str(exc))
                if result.status != "pass":
                    return result
                metrics.append(result.detail)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return NormResult("unavailable", "CPU RMS_NORM probe infrastructure fault", str(exc))
    return NormResult("pass", f"{len(cases)} RMS_NORM cases bit-identical to HEAD arithmetic "
                      f"on {THREADS} threads and within {REL_BOUND:.3g} of float64",
                      "\n".join(metrics))


__all__ = ["CASES", "NormCase", "NormResult", "check_rms_norm_suite", "compare",
           "float64_reference", "head_semantics"]
