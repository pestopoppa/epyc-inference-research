#!/usr/bin/env python3
"""EPYC canonical-recipe preflight gate.

Run before any benchmark sweep to confirm the host + binary + launcher are
all in canonical-throughput state. Exits 0 on PASS, 1 on FAIL. Records the
result to data/preflight/<timestamp>.json for trend tracking.

Five gates (in execution order):
  1. uptime          — warn if >2 days (multi-day uptime causes freq throttle)
  2. libomp          — fail if binary's libomp resolves to AMD AOCC
  3. canonical_cmd   — dry-run executor.start, assert taskset/numactl/--no-mmap
  4. tripwire_bench  — standalone llama-bench tg128 r=2 on Coder-30B-A3B Q4_K_M
  5. freq_under_load — sample /sys/cpufreq mid-tripwire, ≥80/96 cores >2.5 GHz

Gate 4 doubles as the load source for gate 5 — they run in parallel.

Usage:
    # auto-invoked by run_benchmark.py at sweep start (unless --skip-preflight)
    python3 scripts/preflight_canonical.py
    # standalone, ad-hoc check after kernel rebuild / reboot / executor change
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Allow `python3 scripts/preflight_canonical.py` to import from scripts/lib/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.canonical_recipe import (  # noqa: E402
    CANONICAL_PREFIX,
    FREQ_BOOST_MIN_CORES,
    FREQ_BOOST_THRESHOLD_KHZ,
    LLVM20_LIBDIR,
    TRIPWIRE_MODEL_PATH,
    TRIPWIRE_TARGET_TPS,
    TRIPWIRE_TIMEOUT_S,
    apply_canonical_prefix,
    assert_canonical_cmd,
    assert_canonical_env,
    build_canonical_env,
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str
    fix: Optional[str] = None
    metric: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Gate implementations
# ---------------------------------------------------------------------------


def gate_uptime(warn_days: float = 2.0) -> GateResult:
    """Warn (not fail) on multi-day uptime — freq throttle hysteresis builds up."""
    try:
        with open("/proc/uptime") as f:
            uptime_s = float(f.read().split()[0])
    except OSError as e:
        return GateResult(
            "uptime",
            False,
            f"could not read /proc/uptime: {e}",
            fix="confirm /proc is mounted (running on Linux EPYC host)",
        )

    days = uptime_s / 86400.0
    metric = {"uptime_s": uptime_s, "uptime_days": round(days, 2)}

    if days > warn_days:
        return GateResult(
            "uptime",
            True,  # warn-only, doesn't fail
            f"uptime {days:.1f} days — exceeds {warn_days}d warn threshold",
            fix="If gate 4/5 fail, REBOOT first: feedback_host_throttle_check.md",
            metric=metric,
        )
    return GateResult("uptime", True, f"uptime {days:.1f}d ≤ {warn_days}d", metric=metric)


def gate_libomp(binary_path: str) -> GateResult:
    """Fail if the binary's libomp.so resolves to AMD AOCC."""
    try:
        result = subprocess.run(
            ["ldd", binary_path],
            capture_output=True,
            text=True,
            timeout=5,
            env=build_canonical_env(),  # use canonical LD_LIBRARY_PATH for the resolution
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return GateResult(
            "libomp",
            False,
            f"ldd failed: {e}",
            fix=f"check binary exists at {binary_path}",
        )

    libomp_line = next(
        (line for line in result.stdout.splitlines() if "libomp" in line.lower()),
        None,
    )
    metric = {"binary": binary_path, "ldd_libomp": libomp_line}

    if libomp_line is None:
        # binary doesn't link libomp at all — unusual but not necessarily fatal
        return GateResult(
            "libomp",
            True,
            "binary does not link libomp (built without OpenMP?)",
            metric=metric,
        )

    if "/opt/AMD" in libomp_line or "aocc" in libomp_line.lower():
        return GateResult(
            "libomp",
            False,
            f"binary links AMD AOCC libomp: {libomp_line.strip()}",
            fix=(
                "Either (a) ensure LD_LIBRARY_PATH starts with "
                f"{LLVM20_LIBDIR} (executor.py / canonical_recipe.build_canonical_env handles this), "
                "or (b) rebuild binary with `LD_LIBRARY_PATH= cmake -B build ...` to "
                "stop CMake from picking up AOCC at configure time."
            ),
            metric=metric,
        )

    return GateResult(
        "libomp",
        True,
        f"libomp resolves to {libomp_line.strip()}",
        metric=metric,
    )


def gate_canonical_cmd(model_path: str = TRIPWIRE_MODEL_PATH) -> GateResult:
    """Dry-run executor.ServerManager.start for a canonical model role; verify
    constructed cmd starts with the canonical prefix and includes --no-mmap.
    """
    try:
        from lib.executor import ServerManager
        from lib.registry import load_registry
    except ImportError as e:
        return GateResult(
            "canonical_cmd",
            False,
            f"could not import executor/registry: {e}",
            fix="ensure scripts/lib/executor.py and scripts/lib/registry.py are intact",
        )

    registry = load_registry(
        str(Path(__file__).resolve().parent.parent / "orchestration" / "model_registry.yaml")
    )

    # Monkey-patch Popen so start() captures cmd + env without launching anything.
    captured: dict = {}

    class _DryPopen:
        def __init__(self, cmd, **kw):
            captured["cmd"] = cmd
            captured["env"] = kw.get("env", {})
            raise SystemExit("dry-run: cmd captured")

    orig_popen = subprocess.Popen
    subprocess.Popen = _DryPopen  # type: ignore[assignment]

    try:
        mgr = ServerManager(port=8099, threads=96, registry=registry)
        try:
            mgr.start(
                model_path=model_path,
                registry=registry,
                role=None,  # bypass role-driven max_context lookup
                use_chat_api=False,
                env_vars={
                    "OMP_PROC_BIND": "spread",
                    "OMP_PLACES": "cores",
                    "OMP_WAIT_POLICY": "active",
                    "OMP_DYNAMIC": "false",
                },
            )
        except SystemExit:
            pass  # expected
    finally:
        subprocess.Popen = orig_popen  # type: ignore[assignment]

    cmd = captured.get("cmd", [])
    env = captured.get("env", {})

    try:
        assert_canonical_cmd(cmd)
        assert_canonical_env(env)
    except AssertionError as e:
        return GateResult(
            "canonical_cmd",
            False,
            f"executor cmd/env violates canonical recipe:\n{e}",
            fix="route through canonical_recipe.apply_canonical_prefix() and build_canonical_env()",
            metric={"cmd_prefix": cmd[: len(CANONICAL_PREFIX)], "env_keys": sorted(env.keys())},
        )

    return GateResult(
        "canonical_cmd",
        True,
        f"executor cmd starts with {' '.join(CANONICAL_PREFIX)}, env carries OMP stack + LD_LIBRARY_PATH",
        metric={
            "cmd_prefix": cmd[: len(CANONICAL_PREFIX)],
            "no_mmap_in_cmd": "--no-mmap" in cmd,
        },
    )


def _sample_freq() -> dict:
    """Return per-CPU current freq in kHz, plus aggregate stats."""
    freqs: list[int] = []
    for cpu in range(96):
        path = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq"
        try:
            with open(path) as f:
                freqs.append(int(f.read().strip()))
        except (OSError, ValueError):
            freqs.append(0)
    boosting = sum(1 for f in freqs if f >= FREQ_BOOST_THRESHOLD_KHZ)
    return {
        "boosting_count": boosting,
        "min_khz": min(freqs) if freqs else 0,
        "max_khz": max(freqs) if freqs else 0,
        "mean_khz": sum(freqs) // len(freqs) if freqs else 0,
        "per_cpu_khz": freqs,
    }


def gates_tripwire_and_freq(
    binary_path: str, model_path: str
) -> tuple[GateResult, GateResult]:
    """Combined tripwire + freq gate (single llama-bench run powers both).

    Launches `llama-bench tg128 r=2` in a subprocess with the canonical wrapping;
    samples /sys/cpufreq while it's running; parses the t/s from stdout afterwards.
    """
    if not Path(model_path).exists():
        return (
            GateResult(
                "tripwire_bench",
                False,
                f"tripwire model missing: {model_path}",
                fix="download Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf or update TRIPWIRE_MODEL_PATH",
            ),
            GateResult(
                "freq_under_load",
                False,
                "skipped (tripwire could not start)",
            ),
        )
    if not Path(binary_path).exists():
        return (
            GateResult(
                "tripwire_bench",
                False,
                f"llama-bench binary missing: {binary_path}",
                fix="cmake --build build --target llama-bench",
            ),
            GateResult(
                "freq_under_load",
                False,
                "skipped (tripwire could not start)",
            ),
        )

    cmd = apply_canonical_prefix(
        [
            binary_path,
            "-m", model_path,
            "-t", "96",
            "-fa", "1",
            "-p", "0",
            "-n", "128",
            "-mmp", "0",
            "-r", "2",
        ]
    )
    env = build_canonical_env()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )

    # Sample freq mid-run (after warmup, before tripwire finishes). Two samples
    # at 4s and 8s in case run is short.
    freq_samples: list[dict] = []

    def sampler():
        time.sleep(4)
        if proc.poll() is None:
            freq_samples.append(_sample_freq())
        time.sleep(4)
        if proc.poll() is None:
            freq_samples.append(_sample_freq())

    sampler_thread = threading.Thread(target=sampler, daemon=True)
    sampler_thread.start()

    try:
        stdout, stderr = proc.communicate(timeout=TRIPWIRE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        return (
            GateResult(
                "tripwire_bench",
                False,
                f"llama-bench did not complete within {TRIPWIRE_TIMEOUT_S}s",
                fix="check binary loads model + freq is healthy (gate 5)",
            ),
            GateResult(
                "freq_under_load",
                False,
                "tripwire timed out before completion",
            ),
        )
    sampler_thread.join(timeout=1)

    # Parse llama-bench output: looks for `tg128` line with t/s value.
    tg_match = re.search(r"tg128\s*\|\s*([\d.]+)\s*±\s*([\d.]+)", stdout)
    if not tg_match:
        return (
            GateResult(
                "tripwire_bench",
                False,
                f"could not parse tg128 t/s from llama-bench output\n--- stdout ---\n{stdout[-500:]}\n--- stderr ---\n{stderr[-500:]}",
                fix="check binary version / model path / canonical args",
            ),
            GateResult(
                "freq_under_load",
                False,
                "skipped (tripwire output unparseable)",
            ),
        )

    tps = float(tg_match.group(1))
    sigma = float(tg_match.group(2))
    tripwire_passed = tps >= TRIPWIRE_TARGET_TPS

    tripwire_result = GateResult(
        "tripwire_bench",
        tripwire_passed,
        (
            f"Coder-30B-A3B Q4_K_M tg128 = {tps:.2f} ± {sigma:.2f} t/s "
            f"(target ≥{TRIPWIRE_TARGET_TPS:.1f})"
        ),
        fix=(
            None
            if tripwire_passed
            else (
                "Below canonical baseline. Order of investigation: (1) freq throttle (gate 5), "
                "(2) AOCC libomp (gate 2), (3) launcher drift (gate 3), (4) hardware regression."
            )
        ),
        metric={"tg128_tps": tps, "tg128_sigma": sigma, "target_tps": TRIPWIRE_TARGET_TPS},
    )

    # Freq gate
    if not freq_samples:
        freq_result = GateResult(
            "freq_under_load",
            False,
            "no freq samples captured (tripwire too short or sampler thread failed)",
            fix="run preflight again; if persists, sample manually via cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq during a long bench",
        )
    else:
        # Use the LAST sample (most-saturated)
        s = freq_samples[-1]
        boosting = s["boosting_count"]
        passed = boosting >= FREQ_BOOST_MIN_CORES
        freq_result = GateResult(
            "freq_under_load",
            passed,
            (
                f"{boosting}/96 cores ≥{FREQ_BOOST_THRESHOLD_KHZ // 1000} MHz under load "
                f"(min {s['min_khz'] // 1000}, max {s['max_khz'] // 1000}, mean {s['mean_khz'] // 1000} MHz)"
            ),
            fix=(
                None
                if passed
                else (
                    f"Only {boosting}/96 cores boosting. This is the multi-day-uptime hysteresis "
                    "(see feedback_host_throttle_check.md). Fix: REBOOT the host; verify post-reboot "
                    "with this preflight again."
                )
            ),
            metric={
                "samples": freq_samples,
                "threshold_khz": FREQ_BOOST_THRESHOLD_KHZ,
                "min_cores_required": FREQ_BOOST_MIN_CORES,
            },
        )

    return tripwire_result, freq_result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def _color(passed: bool, warn: bool = False) -> str:
    if not passed:
        return RED
    return YELLOW if warn else GREEN


def report(gates: list[GateResult]) -> bool:
    print("=== EPYC Canonical Preflight ===")
    all_passed = True
    for i, g in enumerate(gates, 1):
        # Treat the uptime gate as a warning if uptime > threshold (passed but with fix)
        is_warn = g.passed and g.fix is not None
        status = "WARN" if is_warn else ("PASS" if g.passed else "FAIL")
        col = _color(g.passed, is_warn)
        print(f"[{i}/{len(gates)}] {g.name:18s} {col}{status}{RESET}  {g.detail}")
        if g.fix and not g.passed:
            print(f"        {YELLOW}fix:{RESET} {g.fix}")
        if not g.passed:
            all_passed = False
    print()
    if all_passed:
        print(f"{GREEN}ALL GATES PASS{RESET} — safe to proceed with benchmark sweep.")
    else:
        print(f"{RED}PREFLIGHT FAILED{RESET} — refusing to proceed; address the fix(es) above.")
    return all_passed


def write_record(gates: list[GateResult], record_dir: Path) -> Path:
    record_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = record_dir / f"{ts}.json"
    payload = {
        "timestamp": ts,
        "all_passed": all(g.passed for g in gates),
        "gates": [asdict(g) for g in gates],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


DEFAULT_BINARY = "/mnt/raid0/llm/llama.cpp/build/bin/llama-bench"
DEFAULT_RECORD_DIR = (
    Path(__file__).resolve().parent.parent / "data" / "preflight"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        default=DEFAULT_BINARY,
        help=f"path to llama-bench binary (default: {DEFAULT_BINARY})",
    )
    parser.add_argument(
        "--model",
        default=TRIPWIRE_MODEL_PATH,
        help="path to tripwire model GGUF (default: Coder-30B-A3B Q4_K_M)",
    )
    parser.add_argument(
        "--record-dir",
        default=str(DEFAULT_RECORD_DIR),
        help="directory to write JSON record (default: data/preflight)",
    )
    parser.add_argument(
        "--skip-bench",
        action="store_true",
        help="skip the tripwire bench (and consequently the freq-under-load gate). Use ONLY for dry-run debugging.",
    )
    args = parser.parse_args()

    gates: list[GateResult] = []

    gates.append(gate_uptime())
    gates.append(gate_libomp(args.binary))
    gates.append(gate_canonical_cmd(args.model))

    if args.skip_bench:
        gates.append(GateResult("tripwire_bench", True, "SKIPPED (--skip-bench)"))
        gates.append(GateResult("freq_under_load", True, "SKIPPED (--skip-bench)"))
    else:
        tripwire, freq = gates_tripwire_and_freq(args.binary, args.model)
        gates.extend([tripwire, freq])

    all_passed = report(gates)
    record_path = write_record(gates, Path(args.record_dir))
    print(f"\nResult recorded: {record_path}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
