"""Single source of truth for the EPYC 9655 canonical benchmark recipe.

This module exists to prevent recipe drift between the launcher (executor.py),
the preflight gate (preflight_canonical.py), and any future tooling. Both
modules import the constants and helpers below; a unit test (or the preflight
itself) verifies that the resulting subprocess invocation matches CANONICAL_*.

Recipe history:
- The 47-48 t/s Coder-30B-A3B Q4_K_M tg128 baseline documented in
  cpu-inference-optimization-index.md and 2026-04-28-cpu11-pgo/decision.md
  was measured under the wrapping defined here.
- During the 2026-05-02 benchmark debugging session it became clear that the
  launcher had drifted away from this recipe (missing taskset, mmap defaulted
  to ON, AOCC libomp resolved instead of clang-20). The recipe is now codified
  to prevent silent drift.

Memory references:
- feedback_canonical_baseline_protocol.md
- feedback_omp_env_stack_required.md
- feedback_host_throttle_check.md
"""

from __future__ import annotations

import os
from typing import Optional


# Process-level wrapping. taskset pins to all 96 logical cores (prevents thread
# migration off-socket). numactl --interleave=all stripes pages across all 4 NUMA
# nodes (mandatory for ≥30 t/s on EPYC 9655). Order matters: taskset BEFORE numactl.
CANONICAL_PREFIX: list[str] = ["taskset", "-c", "0-95", "numactl", "--interleave=all"]

# Bench-time flags applied to llama-bench / llama-server invocations.
# -mmp 0 (no-mmap) matters: mmap=ON pulls weights through file-cache first-touch,
# which on EPYC NUMA defeats the --interleave=all striping. Bulk read (no-mmap)
# loads weights into anonymous mmap pages that respect the numactl policy.
CANONICAL_BENCH_FLAGS_LLAMA_BENCH: list[str] = ["-t", "96", "-fa", "1", "-mmp", "0"]

# Subprocess env contributions. These are merged with os.environ.copy() in
# build_canonical_env() — the existing shell env is preserved unless explicitly
# overridden.
CANONICAL_OMP_ENV: dict[str, str] = {
    "OMP_PROC_BIND": "spread",
    "OMP_PLACES": "cores",
    "OMP_WAIT_POLICY": "active",
    "OMP_DYNAMIC": "false",
}

# clang-20's libomp directory. Prepended to LD_LIBRARY_PATH so the dynamic loader
# resolves the binary's libomp.so dependency to clang-20 even when AMD AOCC is
# also on disk. The build's CMakeCache.txt records OpenMP_omp_LIBRARY=
# /opt/AMD/aocc-compiler-5.0.0/lib/libomp.so (CMake found AOCC at configure time),
# but at runtime we override via LD_LIBRARY_PATH. AOCC libomp has different
# thread-pinning behavior and costs a few % throughput; clang-20 libomp is the
# documented-recipe choice.
LLVM20_LIBDIR: str = "/usr/lib/llvm-20/lib"


def build_canonical_env(extra_vars: Optional[dict] = None) -> dict[str, str]:
    """Build a subprocess environment with the canonical OMP stack + libomp override.

    Starts from os.environ.copy(), prepends LLVM20_LIBDIR to LD_LIBRARY_PATH (if
    not already there), merges CANONICAL_OMP_ENV, then applies any extra_vars on
    top. Caller-supplied extra_vars take precedence over CANONICAL_OMP_ENV — that
    is the contract: the canonical stack is the default, but a per-model override
    in the registry can override individual keys.
    """
    env = os.environ.copy()

    existing_ld = env.get("LD_LIBRARY_PATH", "")
    if LLVM20_LIBDIR not in existing_ld.split(":"):
        env["LD_LIBRARY_PATH"] = (
            f"{LLVM20_LIBDIR}:{existing_ld}" if existing_ld else LLVM20_LIBDIR
        )

    for k, v in CANONICAL_OMP_ENV.items():
        env[k] = v

    if extra_vars:
        for k, v in extra_vars.items():
            if v is not None:
                env[k] = str(v)

    return env


def apply_canonical_prefix(cmd: list[str]) -> list[str]:
    """Prepend the taskset+numactl wrapping to a command list.

    Idempotent: if the command already starts with the canonical prefix, returns
    cmd unchanged. This lets callers re-wrap defensively without double-wrapping.
    """
    if cmd[: len(CANONICAL_PREFIX)] == CANONICAL_PREFIX:
        return cmd
    return CANONICAL_PREFIX + cmd


# ---------------------------------------------------------------------------
# Verification helpers (used by preflight + can be used by tests)
# ---------------------------------------------------------------------------


class CanonicalRecipeViolation(AssertionError):
    """Raised when a constructed cmd/env doesn't match the canonical recipe."""


def assert_canonical_cmd(cmd: list[str]) -> None:
    """Raise CanonicalRecipeViolation if cmd doesn't start with the canonical prefix.

    Also asserts that --no-mmap (or -mmp 0) appears somewhere in cmd; without
    this flag, mmap=ON is the default and breaks NUMA interleave.
    """
    if cmd[: len(CANONICAL_PREFIX)] != CANONICAL_PREFIX:
        raise CanonicalRecipeViolation(
            f"cmd does not start with canonical prefix.\n"
            f"  expected: {CANONICAL_PREFIX}\n"
            f"  got:      {cmd[: len(CANONICAL_PREFIX)]}\n"
            f"Fix: route through canonical_recipe.apply_canonical_prefix()."
        )

    has_no_mmap = "--no-mmap" in cmd
    has_mmp_0 = any(
        cmd[i] == "-mmp" and i + 1 < len(cmd) and cmd[i + 1] == "0"
        for i in range(len(cmd))
    )
    if not (has_no_mmap or has_mmp_0):
        raise CanonicalRecipeViolation(
            "cmd is missing --no-mmap (or -mmp 0). mmap=ON breaks NUMA interleave\n"
            "on EPYC and produces sub-canonical decode throughput."
        )


def assert_canonical_env(env: dict[str, str]) -> None:
    """Raise CanonicalRecipeViolation if env doesn't carry the canonical OMP stack
    + libomp override.
    """
    missing = [k for k, v in CANONICAL_OMP_ENV.items() if env.get(k) != v]
    if missing:
        raise CanonicalRecipeViolation(
            f"env is missing or has wrong values for: {missing}\n"
            f"  expected: {CANONICAL_OMP_ENV}\n"
            f"Fix: route env through canonical_recipe.build_canonical_env() or merge "
            f"in CANONICAL_OMP_ENV explicitly."
        )

    ld = env.get("LD_LIBRARY_PATH", "")
    if LLVM20_LIBDIR not in ld.split(":"):
        raise CanonicalRecipeViolation(
            f"LD_LIBRARY_PATH does not include {LLVM20_LIBDIR}.\n"
            f"  got: {ld!r}\n"
            f"Without this, the binary's libomp.so dependency may resolve to AMD AOCC\n"
            f"libomp (different thread-pinning, ~5-10% throughput cost) or fail to load\n"
            f"entirely if AOCC isn't on the loader path."
        )


# ---------------------------------------------------------------------------
# Diagnostic constants (for preflight reporting)
# ---------------------------------------------------------------------------

# Tripwire reference: Qwen3-Coder-30B-A3B Q4_K_M is the canonical-recipe model.
# Decode-only tg128 r=2 takes ~6 seconds and serves as both a speed gate AND
# the load source for the freq-under-load gate (parallel sampling).
TRIPWIRE_MODEL_PATH: str = (
    "/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/"
    "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"
)
TRIPWIRE_TARGET_TPS: float = 45.0  # documented baseline 47-48; allow 5% margin
TRIPWIRE_TIMEOUT_S: int = 90  # generous; actual run is ~6 s

# Freq gate threshold: under load, expect ALL 96 cores boosting above 2.5 GHz.
# A few cores below threshold can be tolerated (background system processes
# briefly de-scheduling). Hard fail if more than 16 cores under threshold.
FREQ_BOOST_THRESHOLD_KHZ: int = 2_500_000
FREQ_BOOST_MIN_CORES: int = 80  # of 96
