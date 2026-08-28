#!/usr/bin/env python3
"""Measure the screening instrument's noise floor by running it against ITSELF.

WHY THIS EXISTS
---------------
The loop nominated candidates against a 3% threshold without ever measuring what 3%
means on its own instrument. It could not: a mismatched estimator was injecting
+2.014pp on every run (fixed, research c3034ede), and the workload dispatched Q5_0
rather than production's Q4_K (fixed, research abcdf787).

An A/A run is the same binary against the same binary. Every non-zero effect it
reports is noise BY CONSTRUCTION, so the distribution of those effects is the floor
below which no candidate result means anything. Without it, "+2.5%" is a number with
no scale.

DESIGN, taken from scripts/benchmark/mmq_mfma_recheck.py -- the 154-line harness that
produced this project's actual decision-grade results:

  * arms ALTERNATE rather than running in blocks, so drift over the window hits both
    arms equally instead of loading onto whichever ran second;
  * one estimator on BOTH arms (median/median), matching the corrected producer;
  * the full sample vector is printed, not just the median, because the failure mode
    that produced a bogus +46.9% was BIMODALITY, which a median hides;
  * a max/min spread check flags a suspect arm;
  * GPU residency is PROVEN -- VRAM sampled DURING the run and the KFD process count
    -- because "I invoked the HIP build" is not evidence of a HIP run and ldd cannot
    supply it (llama.cpp dlopens libggml-hip.so).

It holds the mi210_0 flock for the whole window and never signals a process it did
not start.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import statistics as st
import subprocess
import sys
import threading
import time

DEVICE_LOCK = Path("/mnt/raid0/llm/tmp/gpu_device.mi210_0.lock")
VRAM_SYSFS = Path("/sys/class/drm/card2/device/mem_info_vram_used")
KFD_PROC = Path("/sys/class/kfd/kfd/proc")
CPU_LIST = "184-191"


def read_vram() -> int:
    try:
        return int(VRAM_SYSFS.read_text().strip())
    except (OSError, ValueError):
        return -1


def kfd_count() -> int:
    try:
        return len(list(KFD_PROC.iterdir()))
    except OSError:
        return -1


class ResidencySampler:
    """Sample VRAM and KFD DURING the run.

    A sample taken after the process exits proves nothing -- llama-bench frees its
    allocation on the way out, which is exactly why a post-hoc reading of 0% VRAM is
    the NORMAL result and not evidence of a CPU run.
    """

    def __init__(self, interval: float = 0.25) -> None:
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_vram = 0
        self.peak_kfd = 0
        self.samples = 0

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.peak_vram = max(self.peak_vram, read_vram())
            self.peak_kfd = max(self.peak_kfd, kfd_count())
            self.samples += 1
            self._stop.wait(self.interval)

    def __enter__(self) -> "ResidencySampler":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


def run_bench(binary: Path, model: Path, *, pp: int, tg: int,
              reps: int = 1) -> tuple[dict[str, float], dict[str, int]]:
    """One llama-bench invocation, with residency proven while it runs."""
    argv = ["taskset", "-c", CPU_LIST, "numactl", "--interleave=all",
            str(binary), "-m", str(model), "-p", str(pp), "-n", str(tg),
            "-r", str(reps), "-ngl", "99", "-fa", "1", "-o", "json"]
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{binary.parent}:/opt/rocm/lib"
    env.pop("HSA_OVERRIDE_GFX_VERSION", None)

    with ResidencySampler() as sampler:
        completed = subprocess.run(argv, capture_output=True, text=True,
                                   timeout=3600, env=env)
    if completed.returncode != 0:
        raise RuntimeError(f"llama-bench rc={completed.returncode}: "
                           f"{completed.stderr[-400:]}")
    try:
        rows = json.loads(completed.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"llama-bench emitted non-JSON: {completed.stdout[:300]}")

    metrics = {}
    for row in rows:
        key = f"pp{row['n_prompt']}" if row["n_prompt"] else f"tg{row['n_gen']}"
        metrics[key] = float(row["avg_ts"])
    residency = {"peak_vram_bytes": sampler.peak_vram,
                 "peak_kfd_processes": sampler.peak_kfd,
                 "residency_samples": sampler.samples}
    return metrics, residency


def spread_flag(samples: list[float]) -> str:
    if not samples or min(samples) <= 0:
        return ""
    ratio = max(samples) / min(samples)
    return f"  ** SUSPECT max/min={ratio:.2f}x (bimodal?)" if ratio > 1.3 else ""


def summarise(label: str, arm_a: list[float], arm_b: list[float]) -> dict:
    """Both arms are the same binary, so every effect here is noise."""
    median_a, median_b = st.median(arm_a), st.median(arm_b)
    effect = (median_b / median_a - 1.0) * 100.0
    paired = [(b / a - 1.0) * 100.0 for a, b in zip(arm_a, arm_b)]
    abs_paired = sorted(abs(value) for value in paired)
    p95 = abs_paired[max(0, int(round(0.95 * len(abs_paired))) - 1)] if abs_paired else 0.0

    print(f"\n  {label}")
    print(f"    arm A  median {median_a:10.3f}  n={len(arm_a)}  "
          f"{[round(v, 2) for v in sorted(arm_a)]}{spread_flag(arm_a)}")
    print(f"    arm B  median {median_b:10.3f}  n={len(arm_b)}  "
          f"{[round(v, 2) for v in sorted(arm_b)]}{spread_flag(arm_b)}")
    print(f"    median/median effect (should be ~0): {effect:+.3f}%")
    print(f"    per-pair |effect|: p95 = {p95:.3f}%   max = "
          f"{max(abs_paired) if abs_paired else 0.0:.3f}%")
    print(f"    NOISE FLOOR (p95 of |paired effect|): {p95:.3f}%")
    return {"surface": label, "median_a": median_a, "median_b": median_b,
            "median_over_median_effect_pct": effect,
            "paired_effects_pct": paired,
            "noise_floor_p95_pct": p95,
            "noise_floor_max_pct": max(abs_paired) if abs_paired else 0.0,
            "samples_a": arm_a, "samples_b": arm_b, "pairs": len(paired)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, required=True, help="llama-bench path")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=20,
                        help="alternating A/A pairs (default 20)")
    parser.add_argument("--reps", type=int, default=9,
                        help="llama-bench -r per invocation. Defaults to 9 to match "
                             "the loop's own --calls 9 shape, so the floor "
                             "characterises the instrument as the loop uses it.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if not args.binary.is_file():
        print(f"REFUSED: no llama-bench at {args.binary}", file=sys.stderr)
        return 2
    if not args.model.is_file():
        print(f"REFUSED: no model at {args.model}", file=sys.stderr)
        return 2

    # The claim is ACQUIRED, never observed. Held for the whole window.
    lock_handle = DEVICE_LOCK.open("a")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"REFUSED: mi210_0 is claimed by another holder ({DEVICE_LOCK})",
              file=sys.stderr)
        return 3

    started = time.time()
    baseline_vram = read_vram()
    print(f"claim acquired  baseline VRAM {baseline_vram / 2**30:.3f} GB  "
          f"KFD procs {kfd_count()}")
    print(f"binary {args.binary}\nmodel  {args.model}")

    surfaces = [("prefill pp512", 512, 0), ("decode tg128", 0, 128)]
    results, residency_proofs = [], []
    try:
        for label, pp, tg in surfaces:
            key = f"pp{pp}" if pp else f"tg{tg}"
            arm_a: list[float] = []
            arm_b: list[float] = []
            print(f"\n[{time.strftime('%H:%M:%S')}] {label} — {args.pairs} "
                  f"ALTERNATING A/A pairs")
            for index in range(args.pairs):
                for arm, sink in (("A", arm_a), ("B", arm_b)):
                    try:
                        metrics, residency = run_bench(args.binary, args.model,
                                                       pp=pp, tg=tg, reps=args.reps)
                        sink.append(metrics[key])
                        residency_proofs.append(residency)
                    except Exception as exc:                      # noqa: BLE001
                        print(f"    pair {index} arm {arm}: FAILED {exc}")
            if arm_a and arm_b:
                results.append(summarise(label, arm_a, arm_b))
    finally:
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
        lock_handle.close()

    resident = [proof for proof in residency_proofs if proof["peak_vram_bytes"] > 2**30]
    proof = {
        "invocations": len(residency_proofs),
        "invocations_with_vram_above_1GiB": len(resident),
        "peak_vram_bytes": max((p["peak_vram_bytes"] for p in residency_proofs),
                               default=0),
        "peak_kfd_processes": max((p["peak_kfd_processes"] for p in residency_proofs),
                                  default=0),
        "baseline_vram_bytes": baseline_vram,
    }
    print(f"\nGPU RESIDENCY: {proof['invocations_with_vram_above_1GiB']}/"
          f"{proof['invocations']} invocations sampled >1 GiB VRAM while running; "
          f"peak {proof['peak_vram_bytes'] / 2**30:.2f} GB, "
          f"peak KFD procs {proof['peak_kfd_processes']}")
    if not resident:
        print("  WARNING: no invocation was sampled resident. Treat every number "
              "above as unproven -- this may not have run on the GPU at all.")

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "epyc.autokernel.aa_noise_floor.v1",
        "authority": "instrument_characterisation_not_a_claim",
        "binary": str(args.binary), "model": str(args.model),
        "pairs": args.pairs, "reps_per_invocation": args.reps, "elapsed_s": round(time.time() - started, 1),
        "results": results, "gpu_residency": proof,
    }
    (args.out / "aa-noise-floor.json").write_text(json.dumps(payload, indent=2),
                                                  encoding="utf-8")
    print(f"\nwrote {args.out / 'aa-noise-floor.json'}  "
          f"({payload['elapsed_s']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
