"""RVP-C6-4 physical speed-of-light admission for measured throughput.

The handoff states the bound in time form: elapsed time per delivered unit
cannot be smaller than ``max(compute_time, memory_time)``.  Throughput is the
inverse quantity, so its equivalent ceiling is
``min(peak_compute / flops_per_unit, peak_memory / bytes_per_unit)``.  Keeping
both forms on one immutable object prevents the common max/min direction error.

This module executes nothing and imports no benchmark implementation.  A
campaign must predeclare the per-shape work lower bounds and hardware peak
upper bounds with a source receipt.  A candidate-produced FLOP or byte count is
not an admissible constructor input. The envelope also binds the exact registered
recipe/model/parameter frame and delivered unit whose samples it can grade.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .. import schemas


BOUND_ID = "autokernel.physical-speed-of-light/v1"


class PhysicalBoundError(ValueError):
    """A physical envelope is malformed or cannot grade the observed metric."""


def measurement_frame_sha256(recipe_id: str, params: Mapping[str, Any]) -> str:
    """Digest the exact registered recipe inputs whose samples the bound grades."""
    if not isinstance(recipe_id, str) or not recipe_id.strip():
        raise PhysicalBoundError("recipe_id must be a non-empty string")
    if not isinstance(params, Mapping):
        raise PhysicalBoundError("measurement-frame params must be a mapping")
    return schemas.content_hash({"recipe_id": recipe_id, "params": dict(params)})


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhysicalBoundError(f"{name} must be a finite positive number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise PhysicalBoundError(f"{name} must be a finite positive number")
    return value


@dataclass(frozen=True)
class PhysicalEnvelope:
    """One predeclared work shape and its two independent physical ceilings.

    ``flops_per_unit`` and ``bytes_per_unit`` are conservative LOWER bounds on
    work that must be delivered. ``peak_*`` values are conservative UPPER
    bounds on the named hardware. Both choices bias toward permitting a result,
    so crossing the resulting ceiling is a strong wrong-work/wrong-timer signal.
    """

    shape_id: str
    delivered_unit: str
    flops_per_unit: float
    bytes_per_unit: float
    peak_compute_flops_s: float
    peak_memory_bytes_s: float
    measurement_frame_sha256: str
    work_derivation_ref: str
    hardware_peak_ref: str
    bound_id: str = BOUND_ID

    def __post_init__(self) -> None:
        for name in ("shape_id", "delivered_unit", "work_derivation_ref",
                     "hardware_peak_ref"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise PhysicalBoundError(f"{name} must be a non-empty string")
        if self.bound_id != BOUND_ID:
            raise PhysicalBoundError(f"bound_id must be {BOUND_ID!r}")
        if not isinstance(self.measurement_frame_sha256, str) or not re.fullmatch(
                r"[0-9a-f]{64}", self.measurement_frame_sha256):
            raise PhysicalBoundError(
                "measurement_frame_sha256 must be a lowercase 64-hex digest")
        for name in ("flops_per_unit", "bytes_per_unit", "peak_compute_flops_s",
                     "peak_memory_bytes_s"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PhysicalEnvelope":
        if not isinstance(payload, Mapping):
            raise PhysicalBoundError("physical envelope JSON must be an object")
        constructor_fields = {
            "bound_id", "shape_id", "delivered_unit", "flops_per_unit",
            "bytes_per_unit", "peak_compute_flops_s", "peak_memory_bytes_s",
            "measurement_frame_sha256", "work_derivation_ref", "hardware_peak_ref",
        }
        derived_fields = {
            "compute_time_floor_s", "memory_time_floor_s", "time_floor_s",
            "throughput_ceiling_units_s", "limiting_resource",
        }
        allowed = constructor_fields | derived_fields
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise PhysicalBoundError(f"physical envelope has unknown fields {unknown}")
        required = constructor_fields - {"bound_id"}
        missing = sorted(required - set(payload))
        if missing:
            raise PhysicalBoundError(f"physical envelope is missing {missing}")
        envelope = cls(**{key: value for key, value in payload.items()
                          if key in constructor_fields})
        expected = envelope.to_dict()
        for key in derived_fields & set(payload):
            if payload[key] != expected[key]:
                raise PhysicalBoundError(
                    f"physical envelope derived field {key!r} is {payload[key]!r}, "
                    f"but the declared work and peaks derive {expected[key]!r}")
        return envelope

    @property
    def compute_time_floor_s(self) -> float:
        return self.flops_per_unit / self.peak_compute_flops_s

    @property
    def memory_time_floor_s(self) -> float:
        return self.bytes_per_unit / self.peak_memory_bytes_s

    @property
    def time_floor_s(self) -> float:
        return max(self.compute_time_floor_s, self.memory_time_floor_s)

    @property
    def compute_ceiling_units_s(self) -> float:
        return self.peak_compute_flops_s / self.flops_per_unit

    @property
    def memory_ceiling_units_s(self) -> float:
        return self.peak_memory_bytes_s / self.bytes_per_unit

    @property
    def throughput_ceiling_units_s(self) -> float:
        return min(self.compute_ceiling_units_s, self.memory_ceiling_units_s)

    @property
    def limiting_resource(self) -> str:
        return "compute" if self.compute_time_floor_s >= self.memory_time_floor_s \
            else "memory"

    def check_throughput(self, samples: Sequence[float]) -> schemas.Check:
        """FAIL any sample above the physical ceiling; empty/unreadable is UNKNOWN."""
        if not isinstance(samples, (tuple, list)) or not samples:
            return schemas.Check(schemas.COULD_NOT_CHECK, (
                f"{self.bound_id}: no throughput samples were supplied for shape "
                f"{self.shape_id}; an absent speed vector is not below the ceiling",))
        parsed: list[float] = []
        for index, sample in enumerate(samples):
            try:
                value = _positive(sample, f"samples[{index}]")
            except PhysicalBoundError as exc:
                return schemas.Check(schemas.COULD_NOT_CHECK, (str(exc),))
            parsed.append(value)
        ceiling = self.throughput_ceiling_units_s
        # One ULP-scale comparison tolerance only. A percentage margin would
        # turn a physical impossibility screen into another tunable threshold.
        crossed = tuple((index, value) for index, value in enumerate(parsed)
                        if value > math.nextafter(ceiling, math.inf))
        if crossed:
            rendered = ", ".join(f"sample[{i}]={v:.9g}" for i, v in crossed)
            return schemas.Check(schemas.FAIL, (
                f"{self.bound_id}: {rendered} {self.delivered_unit}/s exceeds the "
                f"{ceiling:.9g} physical ceiling for {self.shape_id} "
                f"({self.limiting_resource}-limited). The time lower bound is "
                f"max({self.compute_time_floor_s:.9g}s compute, "
                f"{self.memory_time_floor_s:.9g}s memory)={self.time_floor_s:.9g}s; "
                "this run measured the wrong work, wrong unit, or wrong timer",))
        return schemas.Check(schemas.PASS, (
            f"all {len(parsed)} samples are at or below {ceiling:.9g} "
            f"{self.delivered_unit}/s for {self.shape_id}; limiting resource="
            f"{self.limiting_resource}; work={self.work_derivation_ref}; "
            f"peaks={self.hardware_peak_ref}",))

    def to_dict(self) -> dict:
        return {
            "bound_id": self.bound_id,
            "shape_id": self.shape_id,
            "delivered_unit": self.delivered_unit,
            "flops_per_unit": self.flops_per_unit,
            "bytes_per_unit": self.bytes_per_unit,
            "peak_compute_flops_s": self.peak_compute_flops_s,
            "peak_memory_bytes_s": self.peak_memory_bytes_s,
            "measurement_frame_sha256": self.measurement_frame_sha256,
            "compute_time_floor_s": self.compute_time_floor_s,
            "memory_time_floor_s": self.memory_time_floor_s,
            "time_floor_s": self.time_floor_s,
            "throughput_ceiling_units_s": self.throughput_ceiling_units_s,
            "limiting_resource": self.limiting_resource,
            "work_derivation_ref": self.work_derivation_ref,
            "hardware_peak_ref": self.hardware_peak_ref,
        }
