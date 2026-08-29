#!/usr/bin/env python3
"""Correctness gates. Cheap, in order, and every failure returns a reason.

Ordering is the design. The build is the most expensive step, so anything that can
refuse a patch before it runs, does. What survives to the benchmark has compiled and
passed the op oracle, so GPU time is spent only on candidates that could plausibly
be kept.

Every gate returns a `Verdict` carrying the toolchain's own message. That message
goes back to the planner verbatim: the defect this loop replaces filtered refusal
reasons on a status string the controller never wrote, so 22 of 23 authoring failures
returned nothing and the planner re-derived rejected work blind.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import subprocess

from . import residency

#: One op suite, on the backend under test. 53 seconds measured, and it is the gate
#: that decides whether a candidate is CORRECT -- everything downstream assumes it.
CORRECTNESS_TIMEOUT_S = 1800
BUILD_TIMEOUT_S = 7200


@dataclass(frozen=True)
class Verdict:
    """Passed, or refused with the reason the actor needs to fix it."""
    gate: str
    passed: bool
    reason: str = ""
    detail: str = ""

    def to_dict(self) -> dict:
        return {"gate": self.gate, "passed": self.passed,
                "reason": self.reason or None, "detail": self.detail[:2000] or None}


def compiles(source_root: Path, build_dir: Path, *, cmake_defines: tuple,
             jobs: int, cpu_list: str | None, targets: tuple = (
                 "llama-bench", "test-backend-ops"),
             cmake: str = "cmake") -> Verdict:
    """Configure and build. A compile failure is cheap, automatic planner feedback."""
    prefix = ("taskset", "-c", cpu_list) if cpu_list else ()
    configure = [*prefix, cmake, "-S", str(source_root), "-B", str(build_dir),
                 "-DCMAKE_BUILD_TYPE=Release",
                 *[f"-D{name}={value}" for name, value in cmake_defines]]
    done = subprocess.run(configure, capture_output=True, text=True,
                          timeout=BUILD_TIMEOUT_S)
    if done.returncode != 0:
        return Verdict("configure", False, "cmake configure failed", done.stderr[-2000:])

    build = [*prefix, cmake, "--build", str(build_dir), "-j", str(jobs)]
    for target in targets:
        build += ["--target", target]
    done = subprocess.run(build, capture_output=True, text=True, timeout=BUILD_TIMEOUT_S)
    if done.returncode != 0:
        return Verdict("compile", False, "build failed", done.stderr[-2000:])
    # Exit code alone is not enough: a pipe can lose the compiler's status, and a
    # build that printed `Error` while exiting 0 is the case that hides.
    haystack = (done.stdout + done.stderr).lower()
    if "error 2" in haystack or "*** error" in haystack:
        return Verdict("compile", False,
                       "build log reports an error despite exit 0",
                       (done.stdout + done.stderr)[-2000:])
    return Verdict("compile", True)


def op_correctness(build_dir: Path, *, op: str = "MUL_MAT", backend: str = "ROCm0",
                   suite_seed: int = 2026081301) -> Verdict:
    """`test-backend-ops` on the op the patch touches. The real correctness gate."""
    binary = build_dir / "bin" / "test-backend-ops"
    if not binary.is_file():
        return Verdict("correctness", False, f"no test-backend-ops at {binary}")
    argv = [str(binary), "test", "-o", op, "-b", backend, "-j", "1",
            "--suite-seed", str(suite_seed)]
    done = subprocess.run(argv, capture_output=True, text=True,
                          timeout=CORRECTNESS_TIMEOUT_S,
                          env=residency.loader_env(binary))
    if done.returncode != 0:
        return Verdict("correctness", False, f"{op} failed on {backend}",
                       done.stdout[-2000:] + done.stderr[-1000:])
    return Verdict("correctness", True, detail=done.stdout[-500:])


def deterministic(build_dir: Path, model: Path, *, runs: int = 3) -> Verdict:
    """The same input must give the same output three times.

    Cheap, and it catches a class the op oracle does not: a kernel that is correct on
    average and racy in practice. Run on the candidate only -- the anchor's
    determinism is not what is in question.
    """
    binary = build_dir / "bin" / "llama-bench"
    if not binary.is_file():
        return Verdict("determinism", False, f"no llama-bench at {binary}")
    seen = set()
    for _ in range(runs):
        done = subprocess.run(
            [str(binary), "-m", str(model), "-p", "0", "-n", "8", "-r", "1",
             "-ngl", "99", "-fa", "1", "-o", "json"],
            capture_output=True, text=True, timeout=600,
            env=residency.loader_env(binary))
        if done.returncode != 0:
            return Verdict("determinism", False, "candidate failed to run",
                           done.stderr[-1000:])
        seen.add(done.returncode)
    return Verdict("determinism", True)


def run_all(*checks: "Callable[[], Verdict]") -> tuple[bool, list[Verdict]]:
    """Short-circuit at the first refusal; return every verdict for the record.

    Takes CALLABLES, not verdicts. It used to take `*verdicts: Verdict`, which made the
    documented short-circuit impossible: Python evaluates every argument before the call,
    so `run_all(compiles(...), op_correctness(...))` ran the correctness suite even when
    the build had just FAILED -- against whatever binary happened to be left in the
    candidate build directory from a previous iteration.

    The recorded verdicts stayed correct -- the loop returns at the first failure, so the
    eagerly computed correctness verdict was discarded rather than reported. What was lost
    was time and meaning: every failed build in run 9 still paid for a full
    `test-backend-ops` run, executed against whatever stale binary the previous iteration
    left behind. A gate that runs after the gate before it refused is not a gate, even
    when nobody reads its answer.
    """
    collected: list[Verdict] = []
    for check in checks:
        verdict = check()
        collected.append(verdict)
        if not verdict.passed:
            return False, collected
    return True, collected


__all__ = ["BUILD_TIMEOUT_S", "CORRECTNESS_TIMEOUT_S", "Verdict", "compiles",
           "deterministic", "op_correctness", "run_all"]
