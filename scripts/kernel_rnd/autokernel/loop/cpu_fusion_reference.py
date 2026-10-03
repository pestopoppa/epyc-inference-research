"""Independent check of the DS41 hc_mixes chain, RMS_NORM -> MUL_MAT(F16), for fusion edits.

`cpu_graph_sync` admits `ggml_cpu_try_fuse_ops` (and, since 2026-10-03, new static
helpers beside it), which is where Fable CPU seed 3 fuses `rms_norm -> hc_mixes` into
one split-K op. `use_ref` disables fusion and no native test-backend-ops case builds that
pair, so before this fixture a fusion of it was gated by nothing that ran it. The probe
(`cpu_fusion_reference_probe.cpp`) builds the exact graph `build_hc_mixes` emits
(reshape of a [K/4, 4, nt] tensor to [K, nt], `ggml_rms_norm` without weight, then
`ggml_mul_mat` with an F16 [K, M] weight) and runs it through the candidate's CPU
backend on a pinned team, so a candidate fusion executes exactly as in serving.

The verdict is a float64 reference computed without ggml:
    y[m, t] = sum_k W[k, m] x[k, t] / sqrt(mean_k x[k, t]^2 + eps)
and every output must satisfy |y_hat - y| <= 2^-10 * sum_k |W[k, m] x[k, t]| / rms_t.
HEAD's unfused path rounds the normed activations to F16 (2^-11 per term, random
signs) and accumulates in float: emulated on these inputs, even a fully sequential
float32 sum lands at 0.5-0.8% of this bound (measured 2026-10-03, 3.2e-5 / 1.8e-5 of
the absolute sum at K = 20480). The POSITIVE cases make every term positive, so the
bound is relative to |y| itself: a split-K that drops or doubles a thread's slice
(1/48 of K is about 2% of y; even 1/427 is about 0.2%), a
wrong or missing normalisation, a mis-strided weight or a transposed output fails by an
order of magnitude. The SIGNED cases keep cancellation honest. Repetitions must be
bit-identical to each other (a racy partial-sum reduction fails). Bit identity with HEAD
is not required: a split-K fusion changes the summation order by design (TOL class).
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


MARKER = "AK_CPU_FUSION_REFERENCE_V1"
METRIC_MARKER = "AK_CPU_FUSION_METRIC_V1"
PROBE = Path(__file__).with_name("cpu_fusion_reference_probe.cpp")
SEED = 20261003
EPS = 1e-6
BOUND = 2.0 ** -10
_MASK = (1 << 64) - 1


@dataclass(frozen=True)
class FusionCase:
    name: str
    mode: str          # "positive" | "signed"
    K: int
    nt: int
    M: int
    threads: int = 8
    reps: int = 3


# DS41 hc_mixes: hc_dim = 4 * 5120 = 20480, hc_mix_dim = 24, nt = 1..3 in serving.
CASES = (
    FusionCase("hc_mixes_nt3_pos", "positive", 20480, 3, 24),
    FusionCase("hc_mixes_nt2_pos", "positive", 20480, 2, 24),
    FusionCase("hc_mixes_nt1_pos", "positive", 20480, 1, 24),
    FusionCase("hc_mixes_nt3_signed", "signed", 20480, 3, 24, reps=4),
    # An odd team and a K that no power-of-two split divides evenly.
    FusionCase("odd_team_pos", "positive", 4100, 5, 24, threads=13),
    FusionCase("small_signed", "signed", 1024, 4, 7, threads=6),
    FusionCase("single_thread_pos", "positive", 2048, 2, 24, threads=1),
)


@dataclass(frozen=True)
class FusionResult:
    status: Literal["pass", "wrong", "unavailable"]
    reason: str = ""
    detail: str = ""


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _mix(z: int) -> int:
    z = (z + 0x9E3779B97F4A7C15) & _MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
    return z ^ (z >> 31)


def _value(seed: int, index: int, bits: int, exp_lo: int, exp_span: int,
           positive: bool) -> float:
    """Bit-identical to the probe's `value()`."""
    h = _mix((seed * 0x100000001B3 + index) & _MASK)
    top = 1 << (bits - 1)
    mantissa = (h & (top - 1)) | top
    exponent = (h >> 24) % exp_span - exp_lo
    magnitude = math.ldexp(mantissa, exponent - (bits - 1))
    return -magnitude if (not positive and (h >> 40) & 1) else magnitude


def _fp16_bits(value: float) -> int:
    return struct.unpack("<H", struct.pack("<e", value))[0]


def _fnv1a(data: bytes, h: int = 0xCBF29CE484222325) -> int:
    for byte in data:
        h = ((h ^ byte) * 0x100000001B3) & _MASK
    return h


def _inputs(case: FusionCase, seed: int = SEED):
    positive = case.mode == "positive"
    x = [_value(seed, i, 24, 12, 25, positive) for i in range(case.K * case.nt)]
    w = [_value(seed + 1, i, 11, 6, 8, positive) for i in range(case.K * case.M)]
    digest = _fnv1a(array("f", x).tobytes())
    digest = _fnv1a(struct.pack(f"<{len(w)}H", *(_fp16_bits(v) for v in w)), digest)
    return x, w, digest


def reference(case: FusionCase, x: list[float], w: list[float]):
    """[(y[m], abs_sum[m]) per m] per t, in float64 (fsum; every product is exact)."""
    eps = struct.unpack("<f", struct.pack("<f", EPS))[0]
    out = []
    for t in range(case.nt):
        col = x[t * case.K:(t + 1) * case.K]
        inv = 1.0 / math.sqrt(math.fsum(v * v for v in col) / case.K + eps)
        row = []
        for m in range(case.M):
            weights = w[m * case.K:(m + 1) * case.K]
            products = [a * b for a, b in zip(weights, col)]
            row.append((math.fsum(products) * inv, math.fsum(map(abs, products)) * inv))
        out.append(row)
    return out


def compare(case: FusionCase, output: str, seed: int = SEED) -> FusionResult:
    lines = output.splitlines()
    headers = [i for i, line in enumerate(lines) if line.startswith(MARKER + " ")]
    if len(headers) != 1:
        raise ValueError("probe marker missing or duplicated")
    expected_header = (f"{MARKER} {case.mode} {case.K} {case.nt} {case.M} "
                       f"{_f32_bits(EPS):08x} {case.threads} {case.reps} {seed}")
    if lines[headers[0]] != expected_header:
        raise ValueError("probe identity or shape mismatch")
    x, w, input_hash = _inputs(case, seed)
    digests, observed, seen_input = {}, {}, None
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
            t = int(item[1])
            if not 0 <= t < case.nt or t in observed or \
                    not re.fullmatch(r"[0-9a-f]+", item[2]) or len(item[2]) != 8 * case.M:
                raise ValueError("out-of-range, duplicate or malformed output row")
            observed[t] = bytes.fromhex(item[2])
        else:
            raise ValueError("unknown probe line")
    if seen_input != input_hash:
        raise ValueError("probe inputs differ from the reference generator (fixture drift)")
    if len(digests) != case.reps or len(observed) != case.nt:
        raise ValueError("incomplete probe payload")
    want = _fnv1a(b"".join(observed[t] for t in range(case.nt)))
    bad = sorted(rep for rep, digest in digests.items() if digest != want)
    if bad:
        return FusionResult("wrong", f"{case.name}: repetitions {bad} of {case.reps} differ "
                            "from the first repetition's bits (nondeterministic reduction)")
    worst = 0.0
    for t, row in enumerate(reference(case, x, w)):
        actual = array("f")
        actual.frombytes(observed[t])
        for m, ((ref, scale), value) in enumerate(zip(row, actual)):
            err = abs(value - ref)
            if not math.isfinite(value) or err > BOUND * scale:
                return FusionResult("wrong", f"{case.name}: output (t={t}, m={m}) is outside "
                                    f"2^-10 of the float64 reference",
                                    f"actual={value!r}, float64={ref!r}, abs_sum={scale!r}")
            if scale:
                worst = max(worst, err / scale)
    metric = {"schema": "epyc.autokernel.cpu_fusion_metric.v1", "case": case.name,
              "mode": case.mode, "K": case.K, "nt": case.nt, "M": case.M,
              "threads": case.threads, "reps": case.reps,
              "max_err_over_abs_sum": worst, "bound": BOUND}
    return FusionResult("pass", f"{case.name}: within 2^-10 of float64, deterministic",
                        METRIC_MARKER + " " + json.dumps(metric, sort_keys=True))


def probe_argv(binary: Path, case: FusionCase, seed: int = SEED) -> list[str]:
    return [str(binary), case.mode, str(case.K), str(case.nt), str(case.M),
            f"{_f32_bits(EPS):08x}", str(case.threads), str(case.reps), str(seed)]


def check_norm_mulmat_suite(build_dir: Path, source_root: Path, *,
                            launch_env: Mapping[str, str] | None = None,
                            topology_prefix: tuple[str, ...] = (),
                            cases: tuple[FusionCase, ...] = CASES) -> FusionResult:
    """Compile the probe once against the candidate and run every case.

    Infrastructure and malformed output are `unavailable`, never numerical `wrong`."""
    if not cases or any(case.mode not in ("positive", "signed") for case in cases):
        raise ValueError("unsupported or empty fusion case selection")
    build_dir, source_root = Path(build_dir), Path(source_root)
    lib_dir = build_dir / "bin"
    needed = (lib_dir / "libggml.so", lib_dir / "libggml-base.so",
              lib_dir / "libggml-cpu.so", source_root / "ggml/include/ggml.h", PROBE)
    missing = [str(path) for path in needed if not path.is_file()]
    if missing:
        return FusionResult("unavailable", "candidate ggml library/header or probe missing",
                            ", ".join(missing))
    env = dict(os.environ if launch_env is None else launch_env)
    env["LD_LIBRARY_PATH"] = str(lib_dir) + (":" + env["LD_LIBRARY_PATH"]
                                         if env.get("LD_LIBRARY_PATH") else "")
    metrics = []
    try:
        with tempfile.TemporaryDirectory(prefix="ak-cpu-fusion-ref-") as temp:
            binary = Path(temp) / "fusion-reference-probe"
            command = ["c++", "-std=c++17", "-O2", "-I", str(source_root / "ggml/include"),
                       str(PROBE), "-L", str(lib_dir), "-Wl,-rpath," + str(lib_dir),
                       "-lggml-cpu", "-lggml-base", "-lggml", "-o", str(binary)]
            built = subprocess.run(command, capture_output=True, text=True, timeout=120,
                                   env=dict(os.environ))
            if built.returncode:
                return FusionResult("unavailable", "CPU fusion probe compile failed",
                                    built.stderr[-2000:])
            for case in cases:
                run = subprocess.run([*topology_prefix, *probe_argv(binary, case)],
                                     capture_output=True, text=True, timeout=120, env=env)
                if run.returncode:
                    return FusionResult("unavailable", f"{case.name} probe did not complete",
                                        f"exit={run.returncode}; {run.stderr[-1800:]}")
                try:
                    result = compare(case, run.stdout)
                except (ValueError, KeyError, OverflowError, StopIteration) as exc:
                    return FusionResult("unavailable", f"{case.name} probe output invalid",
                                        str(exc))
                if result.status != "pass":
                    return result
                metrics.append(result.detail)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return FusionResult("unavailable", "CPU fusion probe infrastructure fault", str(exc))
    return FusionResult("pass", f"{len(cases)} RMS_NORM->MUL_MAT(F16) hc_mixes cases within "
                        "2^-10 of float64 and deterministic", "\n".join(metrics))


__all__ = ["CASES", "FusionCase", "FusionResult", "check_norm_mulmat_suite", "compare",
           "reference"]
