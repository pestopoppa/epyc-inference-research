"""Independent, exact F32 GDN fixture for the CPU AutoKernel lane.

This deliberately does not compare candidate CPU against ggml CPU. The candidate
ggml library executes a fixed C++ probe; a Python scalar recurrence supplies the
expected attention and state snapshots. All inputs and intermediates are small
dyadic values, D=64 gives an exact 1/sqrt(D)=1/8, and q/k are one-hot. Therefore
every expected F32 output is exactly representable: absolute tolerance is zero.
The fixture is a narrow GDN row-layout check, not a general precision policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping
import math
import os
import subprocess
import tempfile


D, H, T, K = 64, 2, 4, 2
Q_ROWS = (0, 15, 32, 63)
K_ROWS = (1, 16, 31, 48)
MARKER = "AK_GDN_REFERENCE_V1"
PROBE = Path(__file__).with_name("gdn_reference_probe.cpp")


@dataclass(frozen=True)
class GDNResult:
    status: Literal["pass", "wrong", "unavailable"]
    reason: str = ""
    detail: str = ""


def expected_output() -> tuple[float, ...]:
    """Scalar recurrence in the documented ggml packed-output order."""
    attention = [0.0] * (D * H * T)
    snapshots = [[0.0] * (D * D * H) for _ in range(K)]
    for h in range(H):
        # ggml stores S[i][j] as state[j][i].
        state = [[(i % 8 - 4)/128 + (j % 8 - 4)/1024 + h/64
                  for i in range(D)] for j in range(D)]
        for t in range(T):
            ki = (K_ROWS[t] + 3*h) % D
            qi = (Q_ROWS[t] + 3*h) % D
            for j in range(D):
                v = (j % 8 - 4)/16 + h/32 + t/64
                delta = (v - state[j][ki]) * 0.5
                state[j][ki] += delta
                attention[D*(h + H*t) + j] = state[j][qi] / 8
            slot = T - 1 - t
            if slot < K:
                for j in range(D):
                    for i in range(D):
                        snapshots[slot][i + D*(j + D*h)] = state[j][i]
    return tuple(attention + [v for slot in snapshots for v in slot])


def _parse_output(output: str) -> tuple[float, ...]:
    lines = output.splitlines()
    headers = [i for i, line in enumerate(lines) if line.startswith(MARKER + " ")]
    if len(headers) != 1:
        raise ValueError("probe marker missing or duplicated")
    index = headers[0]
    count = int(lines[index].split()[1])
    values = tuple(float.fromhex(value) for value in lines[index + 1:])
    if count != len(values) or count != len(expected_output()):
        raise ValueError(f"probe output length {len(values)} != expected {len(expected_output())}")
    if not all(math.isfinite(v) for v in values):
        raise ValueError("probe emitted a non-finite value")
    return values


def check_cpu_gdn(build_dir: Path, source_root: Path, *,
                  launch_env: Mapping[str, str] | None = None,
                  topology_prefix: tuple[str, ...] = ()) -> GDNResult:
    """Compile/run the candidate-bound probe; classify infrastructure separately.

    ``wrong`` is only returned after a successful probe with the exact output
    count. Missing candidate libraries, compile/run errors, and malformed output
    are ``unavailable`` rather than claims that the kernel is incorrect.
    """
    build_dir = Path(build_dir)
    source_root = Path(source_root)
    lib_dir = build_dir / "bin"
    needed = (lib_dir / "libggml.so", lib_dir / "libggml-base.so",
              lib_dir / "libggml-cpu.so")
    missing = [str(path) for path in needed if not path.is_file()]
    if missing:
        return GDNResult("unavailable", "candidate ggml libraries missing", ", ".join(missing))
    include_dir = source_root / "ggml" / "include"
    if not (include_dir / "ggml.h").is_file() or not PROBE.is_file():
        return GDNResult("unavailable", "GDN probe source or ggml headers missing")
    env = dict(os.environ if launch_env is None else launch_env)
    env["LD_LIBRARY_PATH"] = str(lib_dir) + (":" + env["LD_LIBRARY_PATH"]
                                             if env.get("LD_LIBRARY_PATH") else "")
    try:
        with tempfile.TemporaryDirectory(prefix="ak-gdn-ref-") as temp:
            binary = Path(temp) / "gdn-reference-probe"
            compile_argv = ["c++", "-std=c++17", "-O2", "-I", str(include_dir),
                            str(PROBE), "-L", str(lib_dir),
                            "-Wl,-rpath," + str(lib_dir),
                            "-lggml-cpu", "-lggml-base", "-lggml", "-o", str(binary)]
            built = subprocess.run(compile_argv, capture_output=True, text=True,
                                   timeout=120, env=env)
            if built.returncode:
                return GDNResult("unavailable", "GDN probe compile failed", built.stderr[-2000:])
            run = subprocess.run([*topology_prefix, str(binary)], capture_output=True,
                                 text=True, timeout=120, env=env)
            if run.returncode:
                return GDNResult("unavailable", "GDN probe did not complete",
                                 f"exit={run.returncode}; {run.stderr[-1800:]}")
            try:
                observed = _parse_output(run.stdout)
            except ValueError as exc:
                return GDNResult("unavailable", "GDN probe output invalid", str(exc))
    except (OSError, subprocess.TimeoutExpired) as exc:
        return GDNResult("unavailable", "GDN probe infrastructure fault", str(exc))

    expected = expected_output()
    for index, (actual, reference) in enumerate(zip(observed, expected)):
        if actual != reference:
            region = "attention" if index < D*H*T else "state snapshot"
            return GDNResult("wrong", f"GDN {region} mismatch at packed index {index}",
                             f"actual={actual:.9g}, expected={reference:.9g}, abs_tolerance=0")
    return GDNResult("pass", detail=f"{len(expected)} exact F32 outputs; abs_tolerance=0")


__all__ = ["GDNResult", "check_cpu_gdn", "expected_output"]
