#!/usr/bin/env python3
"""Build one arm, measure it, compare the pair. The whole decision arithmetic.

This is the part the old loop buried: 3,869 LOC of `gpu_source_evidence.py` produced
a full per-kernel table and the controller read one float out of it, while the actual
comparison was five lines with a mismatched estimator.

Two rules carry everything here:

  * **One statistic on BOTH arms.** The superseded rule centred on mean(anchor) and
    reported median(candidate effects) against it. Across all 25 historical screens
    that injected +2.014pp, flipped 10 signs, and took nominations from 3 to 7.
  * **Alternate across PROCESSES.** Running all of one arm then all of the other
    leaves between-process variance unsampled and loads window drift onto whichever
    arm ran second. The measured single-pair noise floor is p95 2.175% (prefill) and
    3.452% (decode). See MEASURED_FLOOR_PCT for what averaging actually buys, and
    note the ENFORCED floors in `run.py` sit deliberately above it.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import statistics as st
import subprocess
import time
from typing import Sequence

from . import residency

#: Host threads for the GPU lane. NOT 88-95 -- these are the SMT siblings the
#: production GPU recipe uses, and taking others contends with the CPU baseline.
CPU_LIST = "184-191"
#: p95 |median effect| over EVERY C(20,k) subset of the 20 measured A/A pairs
#: (`artifacts/autokernel-aa-noise-floor/aa-noise-floor.json`). Exhaustive, so it is
#: reproducible exactly rather than resampled -- an earlier hand-written table in
#: `program.md` quoted 0.753%/1.848% at k=5, which no method reproduces from this data.
#:
#: The floors ENFORCED in `run.py` (0.973% / 1.544%, sigma/sqrt(n)) sit ABOVE these on
#: both surfaces, deliberately: 20 pairs is a thin sample of a heavy tail, and a bar
#: below the instrument's resolution is the defect this rebuild exists to close. Never
#: lower an enforced floor beneath its row here.
MEASURED_FLOOR_PCT = {
    "pp512": {1: 2.175, 3: 1.432, 5: 0.479, 9: 0.168, 20: 0.029},
    "tg128": {1: 3.452, 3: 2.527, 5: 1.502, 9: 1.175, 20: 0.067},
}

#: Measured floor, 2026-08-28, n=20 alternating pairs. Below five pairs a single
#: observation on decode can exceed a 3% bar on noise alone (4 of 20 did).
MIN_PAIRS = 5
#: Alternating pairs run and thrown away before measurement begins.
WARMUP_PAIRS = 1


class BenchFailed(RuntimeError):
    """The benchmark did not produce a usable measurement."""


@dataclass(frozen=True)
class Arm:
    name: str
    binary: Path


@dataclass(frozen=True)
class Comparison:
    """A paired result. `effect` is the only number that decides anything."""
    surface: str
    anchor_samples: list[float]
    candidate_samples: list[float]
    effect: float
    estimator: str
    pairs: int
    noise_floor_pct: float | None
    residency: dict
    #: Wall seconds this comparison spent with work on the device. The utilization
    #: leg is meaningless without it, and utilization is the number that would have
    #: caught 1.4 hours of GPU held against 29.0 hours of compiling.
    device_seconds: float = 0.0
    #: Signed within-arm drift, anchor and candidate. A run whose arms are still
    #: warming is not measuring steady-state throughput.
    anchor_drift_pct: float = 0.0
    candidate_drift_pct: float = 0.0

    @property
    def drifting(self) -> bool:
        """True when either arm moved by more than the floor across the run."""
        if self.noise_floor_pct is None:
            return False
        return max(abs(self.anchor_drift_pct),
                   abs(self.candidate_drift_pct)) > self.noise_floor_pct

    @property
    def decisive(self) -> bool:
        """True only if the effect clears the instrument's own noise floor.

        A result inside the floor is not a small win; it is not a measurement of
        anything. The old loop had no floor and a 3% bar that sat below it.
        """
        if self.noise_floor_pct is None:
            return False
        if self.drifting:
            # An arm that is still warming is not resolving anything. Reporting this
            # as an effect is how a first-use cost becomes a kernel finding.
            return False
        return abs(self.effect * 100.0) > self.noise_floor_pct

    def to_dict(self) -> dict:
        return {
            "surface": self.surface, "effect": self.effect,
            "effect_pct": self.effect * 100.0, "estimator": self.estimator,
            "pairs": self.pairs, "noise_floor_pct": self.noise_floor_pct,
            "decisive": self.decisive, "device_seconds": self.device_seconds,
            "drifting": self.drifting,
            "anchor_drift_pct": self.anchor_drift_pct,
            "candidate_drift_pct": self.candidate_drift_pct,
            "anchor_samples": self.anchor_samples,
            "candidate_samples": self.candidate_samples,
            "residency": self.residency,
        }


def run_once(binary: Path, model: Path, *, pp: int, tg: int, reps: int = 9,
             timeout_s: int = 3600) -> tuple[float, dict]:
    """One llama-bench invocation with residency proven while it runs."""
    argv = ["taskset", "-c", CPU_LIST, "numactl", "--interleave=all",
            str(binary), "-m", str(model), "-p", str(pp), "-n", str(tg),
            "-r", str(reps), "-ngl", "99", "-fa", "1", "-o", "json"]
    with residency.Sampler() as sampler:
        done = subprocess.run(argv, capture_output=True, text=True,
                              timeout=timeout_s, env=residency.loader_env(binary))
    if done.returncode != 0:
        raise BenchFailed(f"llama-bench rc={done.returncode}: {done.stderr[-400:]}")
    try:
        rows = json.loads(done.stdout)
    except json.JSONDecodeError as exc:
        raise BenchFailed(f"llama-bench emitted non-JSON: {done.stdout[:200]}") from exc
    key = f"pp{pp}" if pp else f"tg{tg}"
    for row in rows:
        name = f"pp{row['n_prompt']}" if row["n_prompt"] else f"tg{row['n_gen']}"
        if name == key:
            return float(row["avg_ts"]), sampler.proof
    raise BenchFailed(f"llama-bench produced no {key} row")


def compare(anchor: Arm, candidate: Arm, model: Path, *, pp: int, tg: int,
            pairs: int = MIN_PAIRS, reps: int = 9,
            noise_floor_pct: float | None = None,
            warmup_pairs: int = WARMUP_PAIRS) -> Comparison:
    """Alternating paired A/B. The arms swap every pair, never run as two blocks.

    `warmup_pairs` are run and DISCARDED first. Without them the first measured pair
    carries each binary's first-use cost, and that cost is not symmetric: the force-MMQ
    probe's candidate was 4.3% slower on pair 1 than on pair 5 while the anchor was
    flat, which alone produced a decisive-looking -1.469%.
    """
    if pairs < 1:
        raise ValueError("compare needs at least one pair")
    for _ in range(max(0, warmup_pairs)):
        for arm in (anchor, candidate):
            run_once(arm.binary, model, pp=pp, tg=tg, reps=reps)
    anchor_samples: list[float] = []
    candidate_samples: list[float] = []
    proofs: list[dict] = []
    started = time.monotonic()
    for _ in range(pairs):
        for arm, sink in ((anchor, anchor_samples), (candidate, candidate_samples)):
            value, proof = run_once(arm.binary, model, pp=pp, tg=tg, reps=reps)
            sink.append(value)
            proofs.append(proof)

    if not anchor_samples or not candidate_samples:
        raise BenchFailed("comparison produced no samples")

    resident = [proof for proof in proofs if proof["resident"]]
    if len(resident) != len(proofs):
        # A run that cannot be shown resident is not a GPU result. Refuse rather
        # than report a number whose provenance is unknown.
        raise BenchFailed(
            f"only {len(resident)}/{len(proofs)} invocations were sampled resident "
            f"(>= {residency.RESIDENT_FLOOR_BYTES >> 30} GiB VRAM during the run); "
            f"this may not have executed on the GPU")

    centre = st.median(anchor_samples)
    if centre <= 0:
        raise BenchFailed("anchor median is not positive; a relative effect is undefined")
    return Comparison(
        surface=f"pp{pp}" if pp else f"tg{tg}",
        anchor_samples=anchor_samples, candidate_samples=candidate_samples,
        effect=(st.median(candidate_samples) / centre) - 1.0,
        estimator="median_over_median", pairs=pairs,
        noise_floor_pct=noise_floor_pct,
        residency={"invocations": len(proofs),
                   "resident": len(resident),
                   "peak_vram_bytes": max(p["peak_vram_bytes"] for p in proofs),
                   "peak_kfd_processes": max(p["peak_kfd_processes"] for p in proofs),
                   # Aggregated across every invocation in the comparison: if the
                   # governor moved the clock at any point, the effect is partly a
                   # measurement of DVFS rather than of the kernel.
                   "sclk_min_mhz": min((p.get("sclk_min_mhz") or 0) for p in proofs),
                   "sclk_max_mhz": max((p.get("sclk_max_mhz") or 0) for p in proofs),
                   "clock_stable": all(p.get("clock_stable") for p in proofs)},
        device_seconds=time.monotonic() - started,
        anchor_drift_pct=drift_pct(anchor_samples),
        candidate_drift_pct=drift_pct(candidate_samples),
    )


def drift_pct(samples: Sequence[float]) -> float:
    """Signed drift across a run: median(second half) vs median(first half), in %.

    `spread_is_suspect` catches BIMODALITY (a max/min ratio). It cannot catch a
    monotonic warm-up: the force-MMQ probe's candidate arm climbed +4.324% across five
    pairs while the anchor stayed flat, and max/min was only 1.043 -- far under the 1.3
    bar. The per-pair effect marched -4.491% -> -0.037%, so a headline of -1.469%
    described first-use cost, not throughput.
    """
    values = [value for value in samples if value > 0]
    if len(values) < 4:
        return 0.0
    half = len(values) // 2
    early, late = st.median(values[:half]), st.median(values[half:])
    centre = st.median(values)
    return 0.0 if centre <= 0 else (late - early) / centre * 100.0


def spread_is_suspect(samples: Sequence[float], ratio: float = 1.3) -> bool:
    """Bimodality check. A median hides the failure that produced a bogus +46.9%."""
    values = [value for value in samples if value > 0]
    return bool(values) and (max(values) / min(values)) > ratio


__all__ = ["Arm", "BenchFailed", "CPU_LIST", "Comparison", "MEASURED_FLOOR_PCT",
           "MIN_PAIRS", "WARMUP_PAIRS", "compare", "drift_pct", "run_once",
           "spread_is_suspect"]
