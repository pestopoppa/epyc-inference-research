"""CPU A/A calibration record: screen floor + serving floor, unit and host-state hash, n >= 24.

Both arms run the SAME store with the SAME env, so every between-arm difference is
instrument noise. The record carries, per workload surface, the paired floors at the screen
pair count and at the serving comparison pair count, each with a descriptive interval, from
the repo's own estimator (`serving._matched_summary`, which wraps `bench.bootstrap_floor`).

Rules (refusals, not warnings):
  * n >= 24 independent pairs for the gating surface (FLOOR-UNIT-1; serving.MATCHED_CALIBRATION_PAIRS);
  * the unit is `process` (every sample is its own server launch, R23-55);
  * a host-state hash over a declared minimal state is recorded, and a `quiet` label contradicted by
    the recorded foreign-load samples refuses (CPU timing scatter is +-1-2% quiet, ~+-20% under
    co-tenant load, so a floor is only meaningful with its contention state attached);
  * CPU numerics are deterministic: a correctness block, if given, records the exact-zero A/A KLD
    and its source row; it is evidence about numerics, never about timing.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import bench, serving, status
from .floor_units import UNIT_PROCESS

SCHEMA = "epyc.autokernel.cpu_aa_calibration.v1"
MIN_PAIRS = serving.MATCHED_CALIBRATION_PAIRS
SCREEN_PAIRS = bench.MIN_PAIRS
CONTENTION = ("quiet", "co-tenant")
QUIET_FOREIGN_BUSY_P95_MAX = 15.0
HOST_STATE_KEYS = ("store_digest", "cpu_list", "threads", "numa", "recipe_env_digest")


class CpuAACalibrationError(ValueError):
    """The A/A record is under-sampled, mislabelled, or missing a required identity."""


def _num(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise CpuAACalibrationError(f"{label} must be a finite positive number")
    return float(value)


def host_state_hash(state: Mapping[str, Any]) -> str:
    missing = [k for k in HOST_STATE_KEYS if state.get(k) in (None, "", [])]
    if missing:
        raise CpuAACalibrationError(f"host state lacks {missing}")
    raw = json.dumps({k: state[k] for k in sorted(state)}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def surface_floors(anchor: Sequence[float], candidate: Sequence[float], *, serving_pairs: int,
                   gating: bool) -> dict:
    if len(anchor) != len(candidate):
        raise CpuAACalibrationError("A/A arms have different sample counts")
    n = len(anchor)
    a = [_num(v, "anchor sample") for v in anchor]
    c = [_num(v, "candidate sample") for v in candidate]
    if gating and n < MIN_PAIRS:
        raise CpuAACalibrationError(f"gating surface has n={n} < {MIN_PAIRS} pairs (FLOOR-UNIT-1)")
    if n < 2:
        raise CpuAACalibrationError("a surface needs at least 2 pairs")
    if not 1 <= serving_pairs <= n:
        raise CpuAACalibrationError("serving comparison pair count must be within 1..n")
    screen = serving._matched_summary(a, c, min(SCREEN_PAIRS, n))
    serv = serving._matched_summary(a, c, serving_pairs)
    return {"n": n, "gating": gating, "anchor_samples": a, "candidate_samples": c,
            "screen_floor": {"pairs": min(SCREEN_PAIRS, n), "floor_pct": screen["floor_pct"],
                             "interval": screen["interval"]},
            "serving_floor": {"pairs": serving_pairs, "floor_pct": serv["floor_pct"],
                              "interval": serv["interval"]}}


def build_record(*, recipe: str, recipe_hash: str, store_digest: str, unit: str, host_state: Mapping,
                 contention_label: str, foreign_busy_p95: Sequence[float],
                 surfaces: Mapping[str, Mapping], serving_pairs: int,
                 determinism: Mapping | None = None, source_row: str) -> dict:
    """surfaces: {name: {"anchor": [...], "candidate": [...], "gating": bool}} (per-step ms or tok/s)."""
    if unit != UNIT_PROCESS:
        raise CpuAACalibrationError(f"unit must be {UNIT_PROCESS!r}: every sample is a separate server launch")
    if contention_label not in CONTENTION:
        raise CpuAACalibrationError(f"contention_label must be one of {CONTENTION}")
    if host_state.get("store_digest") != store_digest:
        raise CpuAACalibrationError("host state names a different store than the record")
    busy = [float(v) for v in foreign_busy_p95]
    if any(isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in busy):
        raise CpuAACalibrationError("foreign busy samples must be finite and non-negative")
    if not busy:
        raise CpuAACalibrationError("no foreign-load samples recorded")
    worst = max(busy)
    if contention_label == "quiet" and worst > QUIET_FOREIGN_BUSY_P95_MAX:
        raise CpuAACalibrationError(
            f"labelled quiet but a launch saw foreign busy p95 {worst:.1f}% > {QUIET_FOREIGN_BUSY_P95_MAX}%")
    if not surfaces or not any(s.get("gating") for s in surfaces.values()):
        raise CpuAACalibrationError("at least one gating surface is required")
    if not isinstance(source_row, str) or not source_row:
        raise CpuAACalibrationError("source_row is required")
    out = {name: surface_floors(s["anchor"], s["candidate"], serving_pairs=serving_pairs,
                                gating=bool(s.get("gating")))
           for name, s in sorted(surfaces.items())}
    record = {"schema": SCHEMA, "surface": "cpu", "recipe": recipe, "recipe_hash": recipe_hash,
              "store_digest": store_digest, "unit": unit, "host_state": dict(host_state),
              "host_state_hash": host_state_hash(host_state),
              "contention_label": contention_label, "foreign_busy_p95_max": round(worst, 3),
              "min_pairs": MIN_PAIRS, "surfaces": out, "source_row": source_row,
              "use": "arm-identical A/A; a floor is valid only at this contention label and host state"}
    if determinism is not None:
        if determinism.get("kld_aa") != 0 or not determinism.get("row"):
            raise CpuAACalibrationError("determinism block must state kld_aa == 0 and its source row")
        record["determinism"] = dict(determinism)
    return record


def write(store: Path | str, record: Mapping) -> Path:
    if record.get("schema") != SCHEMA or not record.get("host_state_hash"):
        raise CpuAACalibrationError("not a cpu A/A calibration record")
    key = serving.floor_key(str(record["recipe"]))
    name = f"cpu-aa-calibration.{key}.{record['contention_label']}.json"
    return status.write_json(Path(store), name, dict(record), prefix=".cpu-aa-")
