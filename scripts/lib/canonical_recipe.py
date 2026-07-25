"""Single source of truth for the EPYC 9655 canonical benchmark recipe.

This module exists to prevent recipe drift between the launcher (executor.py),
the preflight gate (preflight_canonical.py), and any future tooling. Both
modules import the constants and helpers below; a unit test (or the preflight
itself) verifies that the resulting subprocess invocation matches CANONICAL_*.

Recipe history:
- The raw llama-bench tripwire for Coder-30B-A3B Q4_K_M tg128 is expected
  around 29 t/s under the wrapping defined here. The 47-48 t/s Coder-30B
  numbers in older docs are accelerated MoE/spec/lookup recipes, not this raw
  standalone llama-bench gate.
- During the 2026-05-02 benchmark debugging session it became clear that the
  launcher had drifted away from this recipe (missing taskset, mmap defaulted
  to ON, AOCC libomp resolved instead of clang-20). The recipe is now codified
  to prevent silent drift.
- During the 2026-05-28 post-reboot session SEVEN compounding drift bugs were
  caught in one bench run (wrong binary, wrong libomp, missing OMP_DYNAMIC=false,
  THP defrag reset, perf_event_paranoid reset, broken ik_llama bench binary with
  RUNPATH-vs-LD_LIBRARY_PATH issue). All seven could have been caught earlier
  by a single composite validator. Added validate_canonical_env() +
  validate_host_environment() + assert_binary_resolves_correctly() to surface
  the failures pre-flight, plus discover_canonical_bench_binary() to pick the
  right binary automatically. Use the wrapper script bench_canonical.sh as the
  ONLY sanctioned bench entry point; it composes these validators correctly.

Drift-traps this module catches at validate time (do not reconstruct from memory):
1. taskset/numactl prefix missing or out of order            → assert_canonical_cmd
2. --no-mmap / -mmp 0 missing                                → assert_canonical_cmd
3. OMP_PROC_BIND/PLACES/WAIT_POLICY/DYNAMIC wrong            → assert_canonical_env
4. LD_LIBRARY_PATH missing /usr/lib/llvm-20/lib (AOCC bug)   → assert_canonical_env
5. Binary resolves to wrong libllama/libggml via ldd         → assert_binary_resolves_correctly
6. THP enabled/defrag not 'always'                           → validate_host_environment
7. scaling_governor not 'performance'                        → validate_host_environment
8. numa_balancing not 0                                      → validate_host_environment
9. perf_event_paranoid > 1                                   → validate_host_environment

Memory references:
- feedback_canonical_baseline_protocol.md
- feedback_omp_env_stack_required.md
- feedback_host_throttle_check.md
- feedback_use_codified_recipes_not_memory.md  (the 2026-05-28 recurrence memo)
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
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
    # 2026-06-26 v6 cutover: the v6 binary has ik_llama's iqk AVX-512 GEMM
    # kernels compiled-in but RUNTIME-GATED by GGML_IQK. Without this var the
    # bench measures the v6-clean codepath (no iqk), not the production kernel.
    # Must be present in every role's launch env per the cutover grammar.
    "GGML_IQK": "1",
}

# V4 §Throughput-gate-specific env additions. NOT applied by the orchestrator
# stack universally — scoped here to V4 bench/runner paths only.
#
# Honest production-env reality (as of 2026-05-28, verified against
# epyc-orchestrator/scripts/server/stack_env.py):
#   KMP_BLOCKTIME=10  — applied at orchestrator_stack worker_pool launch (the
#                       ik_llama.cpp PR #1744 branch only), NOT inside
#                       stack_env.build_launch_env(). Documented in
#                       feedback_ik_llamacpp_omp_idle_spin.
#   GGML_NUMA_WEIGHTS=1 — DELIBERATELY EXCLUDED from the worker role per
#                       stack_env.py:62: "DEPRECATED per CPU21 P3 isolation
#                       (unstable, 19-22σ at warmed state)". Per
#                       project_cpu1_phase13_v1 it was +140% alone on Coder-30B
#                       Q4_K_M; later isolation showed warmed-state instability.
#
# Both ARE listed in §Throughput gate of the V4 port handoff because the V4
# bench is the antirez fork (mainstream llama.cpp lineage), not ik_llama —
# its codepath responds to GGML_NUMA_WEIGHTS without the warmed-state
# instability seen on ik_llama, and the throughput floor was specified
# assuming this env. Hence the V4-scoped naming.
#
# For non-V4 (gemma4 / Qwen3.6 / Coder-30B) bench, callers should NOT pass
# use_v4_gate_extras=True — the OMP-only baseline preserves comparability
# with the 47-48 t/s Coder-30B documented baseline AND with stack_env.py.
V4_GATE_EXTRA_ENV: dict[str, str] = {
    "KMP_BLOCKTIME": "10",
    "GGML_NUMA_WEIGHTS": "1",
}

# clang-20's libomp directory. Prepended to LD_LIBRARY_PATH so the dynamic loader
# resolves the binary's libomp.so dependency to clang-20 even when AMD AOCC is
# also on disk. The build's CMakeCache.txt records OpenMP_omp_LIBRARY=
# /opt/AMD/aocc-compiler-5.0.0/lib/libomp.so (CMake found AOCC at configure time),
# but at runtime we override via LD_LIBRARY_PATH. AOCC libomp has different
# thread-pinning behavior and costs a few % throughput; clang-20 libomp is the
# documented-recipe choice.
LLVM20_LIBDIR: str = "/usr/lib/llvm-20/lib"


def build_canonical_env(extra_vars: Optional[dict] = None,
                        use_v4_gate_extras: bool = False,
                        library_path: Optional[str] = None) -> dict[str, str]:
    """Build a subprocess environment with the canonical OMP stack + libomp override.

    Starts from os.environ.copy(), prepends LLVM20_LIBDIR to LD_LIBRARY_PATH (if
    not already there), merges CANONICAL_OMP_ENV, then applies any extra_vars on
    top. Caller-supplied extra_vars take precedence over CANONICAL_OMP_ENV — that
    is the contract: the canonical stack is the default, but a per-model override
    in the registry can override individual keys.

    Args:
        extra_vars: per-model overrides on top of the canonical stack.
        use_v4_gate_extras: if True, also merge V4_GATE_EXTRA_ENV
            (KMP_BLOCKTIME=10, GGML_NUMA_WEIGHTS=1). Required for the V4
            §Throughput gate. NOT applied for non-V4 callers — orchestrator
            stack_env.py deliberately excludes these for the worker role.
        library_path: explicit build-library directory to place first in
            LD_LIBRARY_PATH. Used by candidate A/B arms so their shared
            libraries cannot be shadowed by the production build directory.
    """
    env = os.environ.copy()

    existing_ld = env.get("LD_LIBRARY_PATH", "")
    ld_entries = [entry for entry in existing_ld.split(":") if entry]
    if library_path is not None:
        resolved_library_path = os.path.realpath(library_path)
        ld_entries = [
            entry
            for entry in ld_entries
            if os.path.realpath(entry) != resolved_library_path
        ]
        ld_entries.insert(0, resolved_library_path)
    if LLVM20_LIBDIR not in ld_entries:
        insert_at = 1 if library_path is not None else 0
        ld_entries.insert(insert_at, LLVM20_LIBDIR)
    env["LD_LIBRARY_PATH"] = ":".join(ld_entries)

    for k, v in CANONICAL_OMP_ENV.items():
        env[k] = v
    if use_v4_gate_extras:
        for k, v in V4_GATE_EXTRA_ENV.items():
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


def assert_canonical_env(env: dict[str, str],
                         require_v4_gate_extras: bool = False) -> None:
    """Raise CanonicalRecipeViolation if env doesn't carry the canonical OMP stack
    + libomp override (and optionally V4_GATE_EXTRA_ENV).

    Args:
        env: subprocess env dict to validate.
        require_v4_gate_extras: if True, also require V4_GATE_EXTRA_ENV (the V4
            §Throughput gate stack). Default False — non-V4 callers must NOT
            require these, since orchestrator stack_env.py deliberately
            excludes GGML_NUMA_WEIGHTS for the worker role and only applies
            KMP_BLOCKTIME via a separate worker_pool launch branch.
    """
    missing_omp = [k for k, v in CANONICAL_OMP_ENV.items() if env.get(k) != v]
    if missing_omp:
        raise CanonicalRecipeViolation(
            f"env is missing or has wrong values for OMP vars: {missing_omp}\n"
            f"  expected: {CANONICAL_OMP_ENV}\n"
            f"Fix: route env through canonical_recipe.build_canonical_env() or merge "
            f"in CANONICAL_OMP_ENV explicitly."
        )
    if require_v4_gate_extras:
        missing_v4 = [k for k, v in V4_GATE_EXTRA_ENV.items() if env.get(k) != v]
        if missing_v4:
            raise CanonicalRecipeViolation(
                f"env is missing or has wrong values for V4 gate-extra vars: "
                f"{missing_v4}\n"
                f"  expected: {V4_GATE_EXTRA_ENV}\n"
                f"These are required by §Throughput gate of the V4 port handoff.\n"
                f"Fix: build_canonical_env(use_v4_gate_extras=True)."
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
TRIPWIRE_TARGET_TPS: float = 28.0  # raw llama-bench baseline ~29; accelerated recipes are ~47
TRIPWIRE_TIMEOUT_S: int = 90  # generous; actual run is ~6 s

# Freq gate threshold: under load, expect ALL 96 cores boosting above 2.5 GHz.
# A few cores below threshold can be tolerated (background system processes
# briefly de-scheduling). Hard fail if more than 16 cores under threshold.
FREQ_BOOST_THRESHOLD_KHZ: int = 2_500_000
FREQ_BOOST_MIN_CORES: int = 80  # of 96


# ---------------------------------------------------------------------------
# Binary paths + expected library resolution (catches the 2026-05-28 RUNPATH bug)
# ---------------------------------------------------------------------------

# Production gemma4 MTP server uses ik_llama.cpp. For bench reproducibility we
# default to the same binary — DIFFERENT code paths in mainstream llama.cpp
# vs ik_llama produce ~16% different throughput on the same model.
IK_LLAMA_BENCH: str = "/mnt/raid0/llm/ik_llama.cpp/build/bin/llama-bench"
IK_LLAMA_SERVER: str = "/mnt/raid0/llm/ik_llama.cpp/build/bin/llama-server"

EXPECTED_LIBS_IK_LLAMA: list[str] = [
    "/mnt/raid0/llm/ik_llama.cpp/build/src/libllama.so",
    "/mnt/raid0/llm/ik_llama.cpp/build/ggml/src/libggml.so",
]

# Fallback: mainstream llama.cpp v5_clean build. ONLY use if ik_llama is broken.
# Bench numbers from v5_clean are NOT directly comparable to the production
# baseline; the codepaths differ (different AVX-512BW quant kernels).
V5_CLEAN_BENCH: str = "/mnt/raid0/llm/llama.cpp-experimental/build_v5_clean/bin/llama-bench"
EXPECTED_LIBS_V5_CLEAN: list[str] = [
    "/mnt/raid0/llm/llama.cpp/build/bin/libllama.so",
    "/mnt/raid0/llm/llama.cpp/build/bin/libggml.so",
]

# 2026-06-26 v6 cutover: production-consolidated-v6 is the canonical tree
# (/mnt/raid0/llm/llama.cpp), which becomes v6 at cutover: upstream framework +
# native MTP/NEXTN + our CPU kernels + ik_llama's iqk AVX-512 GEMM kernels gated
# at runtime by GGML_IQK=1. This single kernel REPLACES the two-kernel setup
# (v5 llama.cpp + a separate ik_llama for the gemma worker). ik_llama is fully
# deprecated; the IK_LLAMA_* constants above are retained ONLY as historical
# fallbacks. v6-iqk is the SELECTED canonical bench binary so the gate measures
# the production kernel (iqk-ON) — NOT v5-clean and NOT a stale ik_llama build.
V6_IQK_BENCH: str = "/mnt/raid0/llm/llama.cpp/build/bin/llama-bench"
V6_IQK_SERVER: str = "/mnt/raid0/llm/llama.cpp/build/bin/llama-server"
EXPECTED_LIBS_V6_IQK: list[str] = [
    "/mnt/raid0/llm/llama.cpp/build/bin/libggml-cpu.so.0.15.2",
]

# Strategy-B fork for DeepSeek-V4 (antirez/llama.cpp-deepseek-v4-flash). Cloned
# 2026-05-28 into /mnt/raid0/llm/llama.cpp-deepseek-v4 and built with the same
# canonical hardening as ik_llama (-Wl,--disable-new-dtags so DT_RPATH wins
# over LD_LIBRARY_PATH; clang-20 + LLVM-20 libomp + znver5 native).
#
# This binary is used ONLY for DeepSeek-V4 — it does NOT support gemma4 / Qwen3.6
# / other production-stack models. Bench numbers from this fork are comparable
# to other mainstream-llama.cpp-based forks (e.g. v5_clean) but NOT to ik_llama
# (different codepaths). See handoff deepseek-v4-flash-cpu-port.md Strategy D.
V4_FORK_BENCH: str = "/mnt/raid0/llm/llama.cpp-deepseek-v4/build/bin/llama-bench"
EXPECTED_LIBS_V4_FORK: list[str] = [
    "/mnt/raid0/llm/llama.cpp-deepseek-v4/build/bin/libllama.so",
    "/mnt/raid0/llm/llama.cpp-deepseek-v4/build/bin/libggml.so",
]


def _ldd_output(binary: str, env: Optional[dict[str, str]] = None) -> str:
    """Return ldd output for binary, raising a fail-closed recipe violation."""
    if not os.path.isfile(binary):
        raise CanonicalRecipeViolation(f"binary not found: {binary}")
    if not os.access(binary, os.X_OK):
        raise CanonicalRecipeViolation(f"binary is not executable: {binary}")

    try:
        return subprocess.check_output(
            ["ldd", binary],
            text=True,
            stderr=subprocess.STDOUT,
            env=env,
        )
    except (OSError, subprocess.CalledProcessError) as e:
        output = getattr(e, "output", "")
        returncode = getattr(e, "returncode", "unavailable")
        raise CanonicalRecipeViolation(
            f"ldd {binary} failed (exit {returncode}):\n{output}"
        ) from e


def assert_binary_resolves_correctly(
    binary: str,
    expected_libs: list[str],
    env: Optional[dict[str, str]] = None,
) -> None:
    """Run `ldd binary` and check that libllama/libggml resolve to expected paths.

    Catches the RUNPATH-vs-LD_LIBRARY_PATH drift that broke ik_llama llama-bench
    on 2026-05-28. The binary's DT_RUNPATH is overridden by a polluted
    LD_LIBRARY_PATH, causing it to resolve to mainstream llama.cpp's libllama.so
    (which has since dropped the llama_set_offload_policy symbol that ik_llama
    expects). The fix is to rebuild with `-Wl,--disable-new-dtags` so DT_RPATH
    is set instead of DT_RUNPATH (RPATH beats LD_LIBRARY_PATH).

    Raises CanonicalRecipeViolation with the actual resolution if drift is found.
    """
    out = _ldd_output(binary, env=env)

    for expected in expected_libs:
        lib_name = os.path.basename(expected)
        # Match either "libname => /path (0x...)" or "libname.0 => /path (0x...)"
        # (suffix variants happen with SONAME differences).
        # Strip any trailing version from the expected lib_name for matching.
        base_lib = lib_name.split(".so")[0]
        # First: fail CLOSED on "=> not found" — this is the broken-link case
        # that the previous skip-on-no-match would silently mask.
        not_found_pat = re.compile(
            rf"^\s*{re.escape(base_lib)}\.so(?:\.[0-9.]+)?\s*=>\s*not found\s*$",
            re.MULTILINE,
        )
        if not_found_pat.search(out):
            raise CanonicalRecipeViolation(
                f"{binary} has UNRESOLVED dependency on {lib_name}:\n"
                f"  ldd reports: => not found\n"
                f"  expected:    {expected}\n"
                f"\n"
                f"The binary lists this as a NEEDED dep but the loader cannot\n"
                f"resolve it. Either the lib was deleted/renamed since build\n"
                f"time or LD_LIBRARY_PATH points away from the build tree.\n"
                f"\n"
                f"Inspect with:\n"
                f"  readelf -d {binary} | grep NEEDED\n"
                f"  ldd {binary}"
            )
        pattern = re.compile(
            rf"^\s*{re.escape(base_lib)}\.so(?:\.[0-9.]+)?\s*=>\s*(\S+)\s+\(0x[0-9a-f]+\)\s*$",
            re.MULTILINE,
        )
        match = pattern.search(out)
        if not match:
            # Lib might not be a NEEDED dependency at all (e.g. statically linked).
            # That's not a violation; just means we can't check it. Skip.
            continue
        actual = match.group(1)
        # Expected and actual can differ by suffix (.so vs .so.0); compare prefix.
        expected_base = expected.rstrip("0123456789.")
        actual_base = actual.rstrip("0123456789.")
        if actual_base != expected_base:
            raise CanonicalRecipeViolation(
                f"{binary} resolves {lib_name} INCORRECTLY:\n"
                f"  expected base: {expected_base}\n"
                f"  actual:        {actual}\n"
                f"\n"
                f"This is the 2026-05-28 RUNPATH-vs-LD_LIBRARY_PATH bug. The\n"
                f"binary has DT_RUNPATH set (per `readelf -d {binary}`), but\n"
                f"RUNPATH loses to LD_LIBRARY_PATH at runtime. Fix: rebuild the\n"
                f"binary with `-Wl,--disable-new-dtags` so DT_RPATH is set\n"
                f"instead (RPATH beats LD_LIBRARY_PATH).\n"
                f"\n"
                f"Example fix for ik_llama:\n"
                f"  cd /mnt/raid0/llm/ik_llama.cpp/build\n"
                f"  cmake .. \\\n"
                f"      -DCMAKE_EXE_LINKER_FLAGS='-Wl,--disable-new-dtags' \\\n"
                f"      -DCMAKE_SHARED_LINKER_FLAGS='-Wl,--disable-new-dtags'\n"
                f"  cmake --build . -j 32"
            )


def _path_is_within(path: str, root: str) -> bool:
    """Return whether path resolves to root or one of its descendants."""
    resolved_path = os.path.realpath(path)
    resolved_root = os.path.realpath(root)
    try:
        return os.path.commonpath([resolved_path, resolved_root]) == resolved_root
    except ValueError:
        return False


def assert_explicit_bench_identity(
    binary: str,
    source_root: str,
    library_path: str,
    env: dict[str, str],
) -> None:
    """Validate an explicit candidate arm's source, binary, and shared libraries.

    All three paths are required for explicit A/B arms. The binary and library
    directory must belong to the selected Git worktree, that directory must be
    first in LD_LIBRARY_PATH, and every dynamic llama.cpp library must resolve
    from it. This prevents an experimental binary from silently loading
    production libraries through an ambient loader path or embedded RUNPATH.
    """
    resolved_binary = os.path.realpath(binary)
    resolved_source_root = os.path.realpath(source_root)
    resolved_library_path = os.path.realpath(library_path)

    if not os.path.isfile(resolved_binary):
        raise CanonicalRecipeViolation(f"binary not found: {resolved_binary}")
    if not os.access(resolved_binary, os.X_OK):
        raise CanonicalRecipeViolation(f"binary is not executable: {resolved_binary}")
    if not os.path.isdir(resolved_source_root):
        raise CanonicalRecipeViolation(
            f"source root is not a directory: {resolved_source_root}"
        )
    if not os.path.isdir(resolved_library_path):
        raise CanonicalRecipeViolation(
            f"library path is not a directory: {resolved_library_path}"
        )
    if not _path_is_within(resolved_binary, resolved_source_root):
        raise CanonicalRecipeViolation(
            f"binary is outside --source-root:\n"
            f"  binary:      {resolved_binary}\n"
            f"  source root: {resolved_source_root}"
        )
    if not _path_is_within(resolved_library_path, resolved_source_root):
        raise CanonicalRecipeViolation(
            f"--library-path is outside --source-root:\n"
            f"  library path: {resolved_library_path}\n"
            f"  source root:  {resolved_source_root}"
        )
    if os.path.dirname(resolved_binary) != resolved_library_path:
        raise CanonicalRecipeViolation(
            f"--library-path must be the selected binary's directory:\n"
            f"  binary directory: {os.path.dirname(resolved_binary)}\n"
            f"  library path:     {resolved_library_path}"
        )

    try:
        git_root = subprocess.check_output(
            ["git", "-C", resolved_source_root, "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as e:
        output = getattr(e, "output", "")
        raise CanonicalRecipeViolation(
            f"--source-root is not a readable Git worktree: {resolved_source_root}\n"
            f"{output}"
        ) from e
    if os.path.realpath(git_root) != resolved_source_root:
        raise CanonicalRecipeViolation(
            f"--source-root does not identify the worktree root:\n"
            f"  supplied: {resolved_source_root}\n"
            f"  git root: {os.path.realpath(git_root)}"
        )

    ld_entries = [
        os.path.realpath(entry)
        for entry in env.get("LD_LIBRARY_PATH", "").split(":")
        if entry
    ]
    if not ld_entries or ld_entries[0] != resolved_library_path:
        raise CanonicalRecipeViolation(
            f"--library-path must be first in LD_LIBRARY_PATH:\n"
            f"  expected first: {resolved_library_path}\n"
            f"  actual:         {env.get('LD_LIBRARY_PATH', '')}"
        )

    out = _ldd_output(resolved_binary, env=env)
    relevant_count = 0
    dependency_pattern = re.compile(
        r"^\s*((?:libllama|libggml)\S*\.so(?:\.[0-9.]+)?)\s*=>\s*(\S+)",
        re.MULTILINE,
    )
    for match in dependency_pattern.finditer(out):
        lib_name, actual = match.groups()
        relevant_count += 1
        if actual == "not":
            raise CanonicalRecipeViolation(
                f"{resolved_binary} has unresolved candidate dependency: {lib_name}"
            )
        if not _path_is_within(actual, resolved_library_path):
            raise CanonicalRecipeViolation(
                f"{resolved_binary} resolves {lib_name} outside --library-path:\n"
                f"  expected under: {resolved_library_path}\n"
                f"  actual:         {os.path.realpath(actual)}\n"
                f"Refusing to run a mixed candidate/production benchmark arm."
            )
    if relevant_count == 0:
        raise CanonicalRecipeViolation(
            f"ldd found no libllama/libggml dependencies for {resolved_binary}; "
            f"cannot verify candidate library identity"
        )


def discover_v4_fork_bench() -> tuple[str, list[str]]:
    """Return (V4_FORK_BENCH, EXPECTED_LIBS_V4_FORK) if available and correctly
    linked. Used ONLY for DeepSeek-V4 — this binary doesn't support other archs.

    Raises CanonicalRecipeViolation if the fork isn't built or has the same
    RUNPATH-vs-LD_LIBRARY_PATH issue that ik_llama had pre-2026-05-28.
    """
    if not os.path.isfile(V4_FORK_BENCH):
        raise FileNotFoundError(
            f"DeepSeek-V4 fork bench not found at {V4_FORK_BENCH}.\n"
            f"Build it first: cd /mnt/raid0/llm/llama.cpp-deepseek-v4/build && "
            f"cmake .. -DCMAKE_EXE_LINKER_FLAGS='-Wl,--disable-new-dtags' "
            f"-DCMAKE_SHARED_LINKER_FLAGS='-Wl,--disable-new-dtags' && "
            f"cmake --build . -j 32"
        )
    assert_binary_resolves_correctly(V4_FORK_BENCH, EXPECTED_LIBS_V4_FORK)
    return V4_FORK_BENCH, EXPECTED_LIBS_V4_FORK


def discover_canonical_bench_binary(
    prefer_v6: bool = True, prefer_ik_llama: bool = False
) -> tuple[str, list[str]]:
    """Return (binary_path, expected_libs_list), preferring the v6-iqk binary.

    Verifies via assert_binary_resolves_correctly that the chosen binary resolves
    to ITS OWN libggml(-cpu)/libllama, not someone else's. Candidate order:

      prefer_v6 (DEFAULT, 2026-06-26 cutover):
          v6-iqk -> ik_llama -> v5_clean
      prefer_ik_llama (legacy two-kernel behavior, retained as a fallback knob):
          ik_llama -> v6-iqk -> v5_clean
      neither (force v5_clean first):
          v5_clean -> v6-iqk -> ik_llama

    2026-06-26 v6 cutover: production runs ONE kernel — production-consolidated-v6
    with iqk gated by GGML_IQK=1. The canonical bench MUST measure that binary, so
    v6-iqk is the SELECTED candidate by default. ik_llama is deprecated and kept
    ONLY as a historical fallback; reproducibility now requires the v6 binary +
    GGML_IQK=1 (threaded in via build_canonical_env), not ik_llama.

    NOTE: this function does NOT consider V4_FORK_BENCH; that binary supports
    ONLY DeepSeek-V4 and is selected explicitly via discover_v4_fork_bench()
    or by passing the v4-fork flag through CLI / build_canonical_bench_command.
    """
    candidates: list[tuple[str, list[str]]] = []
    if prefer_ik_llama:
        # Legacy two-kernel preference. Retained as an explicit fallback knob;
        # v6-iqk still ranks above v5_clean.
        candidates = [
            (IK_LLAMA_BENCH, EXPECTED_LIBS_IK_LLAMA),
            (V6_IQK_BENCH, EXPECTED_LIBS_V6_IQK),
            (V5_CLEAN_BENCH, EXPECTED_LIBS_V5_CLEAN),
        ]
    elif prefer_v6:
        # 2026-06-26 v6 cutover: v6-iqk is the selected canonical candidate.
        candidates = [
            (V6_IQK_BENCH, EXPECTED_LIBS_V6_IQK),
            (IK_LLAMA_BENCH, EXPECTED_LIBS_IK_LLAMA),
            (V5_CLEAN_BENCH, EXPECTED_LIBS_V5_CLEAN),
        ]
    else:
        candidates = [
            (V5_CLEAN_BENCH, EXPECTED_LIBS_V5_CLEAN),
            (V6_IQK_BENCH, EXPECTED_LIBS_V6_IQK),
            (IK_LLAMA_BENCH, EXPECTED_LIBS_IK_LLAMA),
        ]

    last_err: Optional[Exception] = None
    for binary, expected_libs in candidates:
        if not os.path.isfile(binary):
            continue
        try:
            assert_binary_resolves_correctly(binary, expected_libs)
            return binary, expected_libs
        except CanonicalRecipeViolation as e:
            last_err = e
            print(
                f"WARN: {binary} failed linkage validation; trying next candidate.\n"
                f"  Reason: {e}",
                file=sys.stderr,
            )

    if last_err is not None:
        raise CanonicalRecipeViolation(
            f"No working llama-bench binary found. Last failure:\n{last_err}"
        )
    raise FileNotFoundError(
        f"No llama-bench binary found at any candidate path:\n"
        f"  {V6_IQK_BENCH}\n  {IK_LLAMA_BENCH}\n  {V5_CLEAN_BENCH}\n"
        f"Check that the v6 canonical tree is built (cmake --build "
        f"/mnt/raid0/llm/llama.cpp/build).\n"  # 2026-06-26 v6 cutover
    )


# ---------------------------------------------------------------------------
# Host-environment validation (catches 2026-05-28 post-reboot defaults reset)
# ---------------------------------------------------------------------------

# Required host state for canonical bench. apply_host_prerequisites() in
# orchestrator_stack.py applies all of these on stack start, but they reset
# to kernel defaults on reboot.
REQUIRED_THP_ENABLED: str = "always"
REQUIRED_THP_DEFRAG: str = "always"
REQUIRED_SCALING_GOVERNOR: str = "performance"
REQUIRED_NUMA_BALANCING: str = "0"
MAX_PERF_EVENT_PARANOID: int = 1  # user-mode HW events need ≤1


def _read_sysfs(path: str) -> Optional[str]:
    """Read a single-line sysfs/proc file; return None if missing or unreadable."""
    try:
        with open(path) as f:
            return f.read().strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _parse_thp_active(thp_value: str) -> Optional[str]:
    """Parse `[always] madvise never` and return the active mode ('always'),
    or None if unparseable.
    """
    m = re.search(r"\[(\w+)\]", thp_value)
    return m.group(1) if m else None


def validate_host_environment(skip_perf_paranoid: bool = False) -> None:
    """Check host-level prerequisites: THP, scaling_governor, numa_balancing,
    perf_event_paranoid.

    All of these reset to kernel defaults at boot. orchestrator_stack.py applies
    them via apply_host_prerequisites() on stack start; this helper catches the
    drift if bench is run before the stack is started.

    Raises CanonicalRecipeViolation listing every drift found, with the exact
    sysctl/sysfs fix for each.

    skip_perf_paranoid: pass True if you don't intend to use perf stat (it's only
    needed for the perf-wrapped bench path).
    """
    issues: list[str] = []

    # THP enabled
    thp_enabled = _read_sysfs("/sys/kernel/mm/transparent_hugepage/enabled")
    if thp_enabled is not None:
        active = _parse_thp_active(thp_enabled)
        if active != REQUIRED_THP_ENABLED:
            issues.append(
                f"transparent_hugepage/enabled active mode = {active!r}\n"
                f"  expected: {REQUIRED_THP_ENABLED!r}\n"
                f"  fix: echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled"
            )

    # THP defrag (this was the missed one on 2026-05-28 reboot)
    thp_defrag = _read_sysfs("/sys/kernel/mm/transparent_hugepage/defrag")
    if thp_defrag is not None:
        active = _parse_thp_active(thp_defrag)
        if active != REQUIRED_THP_DEFRAG:
            issues.append(
                f"transparent_hugepage/defrag active mode = {active!r}\n"
                f"  expected: {REQUIRED_THP_DEFRAG!r}\n"
                f"  fix: echo always | sudo tee /sys/kernel/mm/transparent_hugepage/defrag"
            )

    # CPU governor (sample cpu0)
    governor = _read_sysfs("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if governor is not None and governor != REQUIRED_SCALING_GOVERNOR:
        issues.append(
            f"cpu0 scaling_governor = {governor!r}\n"
            f"  expected: {REQUIRED_SCALING_GOVERNOR!r}\n"
            f"  fix: sudo cpupower frequency-set -g performance"
        )

    # numa_balancing (per feedback_numa_balancing_self_reset, this resets to 0
    # at boot which is what we want; only flag if non-zero)
    nb = _read_sysfs("/proc/sys/kernel/numa_balancing")
    if nb is not None and nb != REQUIRED_NUMA_BALANCING:
        issues.append(
            f"kernel.numa_balancing = {nb!r}\n"
            f"  expected: {REQUIRED_NUMA_BALANCING!r}\n"
            f"  fix: sudo sysctl -w kernel.numa_balancing=0\n"
            f"  see feedback_numa_balancing_self_reset memory."
        )

    # perf_event_paranoid (post-reboot default is 4; we need ≤1 for user-mode HW events)
    if not skip_perf_paranoid:
        pep = _read_sysfs("/proc/sys/kernel/perf_event_paranoid")
        if pep is not None:
            try:
                pep_int = int(pep)
                if pep_int > MAX_PERF_EVENT_PARANOID:
                    issues.append(
                        f"kernel.perf_event_paranoid = {pep_int}\n"
                        f"  expected: ≤{MAX_PERF_EVENT_PARANOID} (for user-mode HW events)\n"
                        f"  fix: sudo sysctl -w kernel.perf_event_paranoid={MAX_PERF_EVENT_PARANOID}"
                    )
            except ValueError:
                pass

    if issues:
        raise CanonicalRecipeViolation(
            "Host environment drift detected (kernel defaults reset on reboot;\n"
            "orchestrator_stack.py start applies the correct values via\n"
            "apply_host_prerequisites()):\n\n"
            + "\n\n".join(issues)
            + "\n\n"
            "Alternative: run `python3 /mnt/raid0/llm/epyc-orchestrator/scripts/server/"
            "orchestrator_stack.py start` to apply all prerequisites at once."
        )


# ---------------------------------------------------------------------------
# Composite validator — call BEFORE constructing any bench command
# ---------------------------------------------------------------------------


def validate_canonical_env(
    cmd: Optional[list[str]] = None,
    env: Optional[dict[str, str]] = None,
    binary: Optional[str] = None,
    expected_libs: Optional[list[str]] = None,
    check_host: bool = True,
    skip_perf_paranoid: bool = False,
    require_v4_gate_extras: bool = False,
) -> None:
    """All-in-one validation: command shape + env vars + binary linkage + host config.

    Call this BEFORE running any bench. Raises CanonicalRecipeViolation with a
    detailed error message identifying what drifted. None of the checks run the
    binary; they're all static / read-only.

    Args:
        cmd: command list to validate (must start with CANONICAL_PREFIX, must
            include --no-mmap or -mmp 0). Pass None to skip cmd check.
        env: environment dict to validate (must have CANONICAL_OMP_ENV + LLVM20
            on LD_LIBRARY_PATH). Pass None to skip env check.
        binary: path to llama-bench (or llama-server). Pass None to skip linkage
            check.
        expected_libs: expected libllama/libggml resolutions for the binary.
            Required if binary is provided.
        check_host: validate THP/governor/numa_balancing/perf_event_paranoid.
            Pass False to skip (default True).
        skip_perf_paranoid: pass True if you don't intend to use perf stat;
            skips the perf_event_paranoid check.
    """
    if cmd is not None:
        assert_canonical_cmd(cmd)
    if env is not None:
        assert_canonical_env(env, require_v4_gate_extras=require_v4_gate_extras)
    if binary is not None:
        if expected_libs is None:
            raise ValueError("expected_libs must be provided when binary is given")
        assert_binary_resolves_correctly(binary, expected_libs)
    if check_host:
        validate_host_environment(skip_perf_paranoid=skip_perf_paranoid)


# ---------------------------------------------------------------------------
# High-level bench-command constructor (the one canonical entry point)
# ---------------------------------------------------------------------------


def build_canonical_bench_command(
    model: str,
    n_prompt: int = 0,
    n_gen: int = 512,
    reps: int = 2,
    extra_flags: Optional[list[str]] = None,
    prefer_v6: bool = True,  # 2026-06-26 v6 cutover: select v6-iqk by default
    prefer_ik_llama: bool = False,
    use_v4_fork: bool = False,
    binary: Optional[str] = None,
    source_root: Optional[str] = None,
    library_path: Optional[str] = None,
    ggml_iqk: str = "1",
    ggml_iqk_q8_0: Optional[str] = None,
) -> tuple[str, list[str], dict[str, str]]:
    """Build (binary_path, cmd_list, env_dict) for the canonical llama-bench run.

    This is the ONLY blessed way to construct a bench command. Do not reconstruct
    from memory — drift bit this project at least 3 times (2026-05-02, 2026-05-28
    multiple), and the recipe has been codified specifically to prevent it.

    Args:
        model: path to GGUF model file
        n_prompt: prefill tokens (default 0 = decode-only bench)
        n_gen: generation tokens per rep (default 512)
        reps: repetitions (default 2)
        extra_flags: optional list of additional flags to pass to llama-bench
            (e.g. ['-ctk', 'q8_0', '-ctv', 'q8_0']). Do NOT include flags that
            are already in CANONICAL_BENCH_FLAGS_LLAMA_BENCH.
        prefer_v6: 2026-06-26 v6 cutover — prefer the v6-iqk llama-bench (the
            production-consolidated-v6 kernel, iqk gated by GGML_IQK=1 which
            build_canonical_env threads in). DEFAULT. Falls back to ik_llama
            then v5_clean. Ignored when use_v4_fork=True.
        prefer_ik_llama: legacy two-kernel preference (ik_llama first). Retained
            only as a fallback knob; ik_llama is deprecated post-cutover.
            Ignored when use_v4_fork=True.
        use_v4_fork: select the DeepSeek-V4 fork binary (V4_FORK_BENCH). The
            V4 fork is the ONLY way to run V4 GGUFs; it doesn't support other
            archs. See handoff deepseek-v4-flash-cpu-port.md Strategy D.
        binary: explicit llama-bench binary for a candidate A/B arm. Must be
            supplied together with source_root and library_path.
        source_root: Git worktree root owning an explicit binary.
        library_path: directory containing the explicit binary's llama.cpp
            shared libraries. It is placed first in LD_LIBRARY_PATH.
        ggml_iqk: runtime iqk gate, either "0" or "1" (default "1").
        ggml_iqk_q8_0: Q8_0 IQK sub-gate. Only "1" is accepted when supplied;
            omission leaves the sub-gate absent from the effective environment.

    Returns:
        binary: absolute path to llama-bench
        cmd: full command list ready for subprocess.run (includes
            taskset+numactl prefix + binary + canonical flags + user flags)
        env: subprocess environment ready for subprocess.run (canonical OMP +
            libomp LD_LIBRARY_PATH override + caller's os.environ)
    """
    if not os.path.isfile(model):
        raise FileNotFoundError(f"Model file not found: {model}")

    ggml_iqk = str(ggml_iqk)
    if ggml_iqk not in {"0", "1"}:
        raise CanonicalRecipeViolation(
            f"ggml_iqk must be '0' or '1', got {ggml_iqk!r}"
        )
    if ggml_iqk_q8_0 is not None and str(ggml_iqk_q8_0) != "1":
        raise CanonicalRecipeViolation(
            "ggml_iqk_q8_0 must be omitted or '1', got "
            f"{ggml_iqk_q8_0!r}"
        )

    explicit_values = (binary, source_root, library_path)
    if any(value is not None for value in explicit_values) and not all(
        value is not None for value in explicit_values
    ):
        raise CanonicalRecipeViolation(
            "--binary, --source-root, and --library-path must be supplied together"
        )
    explicit_arm = all(value is not None for value in explicit_values)
    if explicit_arm and (use_v4_fork or prefer_ik_llama):
        raise CanonicalRecipeViolation(
            "explicit binary identity options cannot be combined with "
            "--v4-fork or --ik-llama"
        )

    extra_vars = {"GGML_IQK": ggml_iqk}
    if ggml_iqk_q8_0 is not None:
        extra_vars["GGML_IQK_Q8_0"] = "1"
    env = build_canonical_env(
        extra_vars=extra_vars,
        use_v4_gate_extras=use_v4_fork,
        library_path=library_path,
    )

    if explicit_arm:
        assert binary is not None
        assert source_root is not None
        assert library_path is not None
        binary = os.path.realpath(binary)
        source_root = os.path.realpath(source_root)
        library_path = os.path.realpath(library_path)
        assert_explicit_bench_identity(binary, source_root, library_path, env)
    elif use_v4_fork:
        binary, _expected_libs = discover_v4_fork_bench()
    else:
        # 2026-06-26 v6 cutover: default selection is the v6-iqk candidate.
        binary, _expected_libs = discover_canonical_bench_binary(
            prefer_v6=prefer_v6, prefer_ik_llama=prefer_ik_llama
        )

    bench_args: list[str] = (
        list(CANONICAL_BENCH_FLAGS_LLAMA_BENCH)
        + ["-m", model, "-p", str(n_prompt), "-n", str(n_gen), "-r", str(reps), "-o", "md"]
    )
    if extra_flags:
        bench_args = bench_args + list(extra_flags)

    cmd = apply_canonical_prefix([binary, *bench_args])

    return binary, cmd, env


# ---------------------------------------------------------------------------
# CLI entry point for shell-script consumption
# ---------------------------------------------------------------------------


def _emit_bench_command_json(args) -> int:
    """Build the canonical bench command and emit it as JSON to stdout.

    The shell wrapper (bench_canonical.sh) consumes this and exec's the command
    with the right env. Keeping the construction logic in Python (single source
    of truth) and the orchestration in shell (so perf-stat wrapping is natural)
    gives clean separation.
    """
    import json

    try:
        # 2026-06-26 v6 cutover: default to the v6-iqk candidate. --ik-llama
        # restores the legacy two-kernel preference (ik_llama first).
        prefer_ik = getattr(args, "ik_llama", False)
        binary, cmd, env = build_canonical_bench_command(
            model=args.model,
            n_prompt=args.n_prompt,
            n_gen=args.n_gen,
            reps=args.reps,
            extra_flags=args.extra,
            prefer_v6=not prefer_ik,
            prefer_ik_llama=prefer_ik,
            use_v4_fork=args.v4_fork,
            binary=args.binary,
            source_root=args.source_root,
            library_path=args.library_path,
            ggml_iqk=args.ggml_iqk,
            ggml_iqk_q8_0=args.ggml_iqk_q8_0,
        )
    except (FileNotFoundError, CanonicalRecipeViolation) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if args.validate_host:
        try:
            validate_host_environment(skip_perf_paranoid=not args.with_perf)
        except CanonicalRecipeViolation as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2

    # Emit only the env variables WE set (don't dump entire os.environ).
    # V4-fork bench gets the V4 gate extras; other paths stay OMP-only to
    # match the orchestrator's non-V4 launch env (stack_env.py).
    emitted_env = {
        "LD_LIBRARY_PATH": env["LD_LIBRARY_PATH"],
        **{key: env[key] for key in CANONICAL_OMP_ENV},
    }
    if args.v4_fork:
        emitted_env.update(V4_GATE_EXTRA_ENV)
    if args.ggml_iqk_q8_0 is not None:
        emitted_env["GGML_IQK_Q8_0"] = "1"

    out = {
        "binary": binary,
        "cmd": cmd,
        "env": emitted_env,
        "source_root": os.path.realpath(args.source_root) if args.source_root else None,
        "library_path": (
            os.path.realpath(args.library_path) if args.library_path else None
        ),
        "ggml_iqk": args.ggml_iqk,
        "ggml_iqk_q8_0": args.ggml_iqk_q8_0,
    }
    print(json.dumps(out, indent=2))
    return 0


def _validate_only(args) -> int:
    """Run validate_canonical_env() against the host + discovered binary,
    without emitting any command. Useful as a pre-bench preflight check.
    """
    try:
        if args.v4_fork:
            binary, expected_libs = discover_v4_fork_bench()
        else:
            # 2026-06-26 v6 cutover: default to the v6-iqk candidate.
            prefer_ik = getattr(args, "ik_llama", False)
            binary, expected_libs = discover_canonical_bench_binary(
                prefer_v6=not prefer_ik, prefer_ik_llama=prefer_ik
            )
        validate_canonical_env(
            binary=binary,
            expected_libs=expected_libs,
            check_host=True,
            skip_perf_paranoid=not args.with_perf,
            require_v4_gate_extras=args.v4_fork,
        )
    except (FileNotFoundError, CanonicalRecipeViolation) as e:
        print(f"VALIDATION FAILED:\n{e}", file=sys.stderr)
        return 1

    print(f"OK: canonical recipe validated. Binary: {binary}")
    return 0


def _main() -> int:
    import argparse

    # Manually split off post-`--` argv before argparse sees it. argparse's
    # nargs=REMAINDER on a subparser arg silently rejects dash-prefixed extras
    # passed after `--` (it treats `--` as an unknown option). Splitting here
    # preserves the natural shell-wrapper convention:
    #   bench_canonical.sh -m MODEL -- -ctk q8_0 -ctv q8_0
    argv = sys.argv[1:]
    extra_args: Optional[list[str]] = None
    if "--" in argv:
        idx = argv.index("--")
        extra_args = argv[idx + 1:]
        argv = argv[:idx]

    p = argparse.ArgumentParser(
        prog="canonical_recipe",
        description="Single source of truth for EPYC 9655 canonical bench recipe.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # validate subcommand
    pv = sub.add_parser(
        "validate",
        help="Validate host + binary linkage. Exits non-zero on drift.",
    )
    pv.add_argument(
        # 2026-06-26 v6 cutover: default is the v6-iqk binary; --ik-llama opts
        # back into the deprecated legacy two-kernel preference (ik_llama first).
        "--ik-llama",
        action="store_true",
        help="Legacy: prefer the deprecated ik_llama binary (default: prefer v6-iqk).",
    )
    pv.add_argument(
        "--v4-fork",
        action="store_true",
        help="Validate the DeepSeek-V4 fork binary instead of v6-iqk/ik_llama/v5_clean.",
    )
    pv.add_argument(
        "--with-perf",
        action="store_true",
        help="Also validate perf_event_paranoid (needed for perf-wrapped bench).",
    )

    # emit-bench-command subcommand
    pe = sub.add_parser(
        "emit-bench-command",
        help="Emit canonical bench command as JSON (for shell wrapper consumption).",
    )
    pe.add_argument("--model", required=True, help="Path to GGUF model file.")
    pe.add_argument("--n-prompt", type=int, default=0)
    pe.add_argument("--n-gen", type=int, default=512)
    pe.add_argument("--reps", type=int, default=2)
    pe.add_argument(
        "--binary",
        help="Explicit llama-bench binary. Requires --source-root and --library-path.",
    )
    pe.add_argument(
        "--source-root",
        help="Git worktree root owning --binary. Requires all explicit identity options.",
    )
    pe.add_argument(
        "--library-path",
        help="Directory containing candidate llama.cpp libraries; pinned first in "
        "LD_LIBRARY_PATH. Requires all explicit identity options.",
    )
    pe.add_argument(
        "--ggml-iqk",
        choices=("0", "1"),
        default="1",
        help="Set the GGML_IQK runtime gate (default: 1).",
    )
    pe.add_argument(
        "--ggml-iqk-q8-0",
        choices=("1",),
        help="Explicitly enable GGML_IQK_Q8_0 for Q8_0 benchmark rows.",
    )
    # NOTE: --extra is handled by the pre-argparse split-on-`--` above.
    # Any args after `--` on the command line are captured into extra_args and
    # attached to args.extra below. Do not declare --extra as an argparse arg —
    # REMAINDER + subparsers + dash-prefixed extras is a known argparse failure.
    pe.add_argument(
        # 2026-06-26 v6 cutover: default is the v6-iqk binary; --ik-llama opts
        # back into the deprecated legacy two-kernel preference (ik_llama first).
        "--ik-llama",
        action="store_true",
        help="Legacy: prefer the deprecated ik_llama binary (default: prefer v6-iqk).",
    )
    pe.add_argument(
        "--v4-fork",
        action="store_true",
        help="Select the DeepSeek-V4 fork binary (V4_FORK_BENCH). Only valid for "
        "V4 GGUFs; this binary doesn't support other archs.",
    )
    pe.add_argument(
        "--no-validate-host",
        dest="validate_host",
        action="store_false",
        default=True,
        help="Skip host-environment validation (default: validate).",
    )
    pe.add_argument(
        "--with-perf",
        action="store_true",
        help="Wrap-with-perf intent — affects perf_event_paranoid check only.",
    )

    args = p.parse_args(argv)
    # Attach the post-`--` extras (parsed before argparse).
    args.extra = extra_args
    if args.cmd == "validate":
        return _validate_only(args)
    if args.cmd == "emit-bench-command":
        return _emit_bench_command_json(args)
    p.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(_main())
