#!/usr/bin/env python3
"""Two-slot hardware context for the C4 prompt feed (INF48-LIVE-3: RVP-C4-8).

From intake-1221#record Fig 3: an ARCH-CONSTANTS slot and a SEPARATE
BLOCK-SCHEDULING-POLICY slot. The feed is curated and bounded on purpose: that
source's own 3-arm ablation found a ~10k-token unfiltered architecture dump no better
than no hardware context, and intake-1222#record measured a raw ~50-counter dump
WORSE than no profiling. So the schema has a closed key set and a rendered-size cap.

gfx90a values are first-party (`rocminfo` on this host, recorded in the RVP-C4-8 box):
104 CUs, 8 MB unified L2, 1 die, no XCDs, linear dispatch. Stating the ABSENCE of XCDs
is the point of the scheduling slot -- it suppresses an XCD-aware remap proposal that
is provably the identity map on a single-die part.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping

SCHEMA = "epyc.autokernel.c4_hardware_context.v1"
MAX_RENDERED_CHARS = 1200   # curated feed; a dump is refused, not truncated

ARCH_KEYS = ("gfx_target", "compute_units", "wavefront_size", "simds_per_cu",
             "lds_bytes_per_cu", "l2_bytes", "l2_topology", "dies",
             "matrix_cores", "hbm_bandwidth_gbps")
SCHED_KEYS = ("xcds", "dispatch_order", "workgroup_to_cu", "remap_advice")
SCHED_DISPATCH = ("linear", "round_robin_xcd")


class HardwareContextError(ValueError):
    pass


@dataclass(frozen=True)
class ArchConstants:
    gfx_target: str
    compute_units: int
    wavefront_size: int
    simds_per_cu: int
    lds_bytes_per_cu: int
    l2_bytes: int
    l2_topology: str
    dies: int
    matrix_cores: str
    hbm_bandwidth_gbps: float


@dataclass(frozen=True)
class SchedulingPolicy:
    xcds: int
    dispatch_order: str
    workgroup_to_cu: str
    remap_advice: str


@dataclass(frozen=True)
class HardwareContext:
    arch: ArchConstants
    scheduling: SchedulingPolicy
    source: str

    def __post_init__(self) -> None:
        a, s = self.arch, self.scheduling
        for name in ("compute_units", "wavefront_size", "simds_per_cu",
                     "lds_bytes_per_cu", "l2_bytes", "dies"):
            if not isinstance(getattr(a, name), int) or getattr(a, name) <= 0:
                raise HardwareContextError(f"arch.{name} must be a positive int")
        if s.xcds < 0:
            raise HardwareContextError("scheduling.xcds must be >= 0")
        if s.dispatch_order not in SCHED_DISPATCH:
            raise HardwareContextError(f"dispatch_order must be one of {SCHED_DISPATCH}")
        # An XCD remap is only meaningful with >1 XCD; on 0/1 it is the identity map.
        if s.xcds <= 1 and s.dispatch_order != "linear":
            raise HardwareContextError("a part with <=1 XCD dispatches linearly")
        if s.xcds > 1 and a.dies < 1:
            raise HardwareContextError("XCDs without a die count is incoherent")
        if not self.source.strip():
            raise HardwareContextError("source must name the first-party evidence")

    def as_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "arch_constants": asdict(self.arch),
                "block_scheduling_policy": asdict(self.scheduling),
                "source": self.source}

    def render(self) -> str:
        """Prompt text: two labelled slots, nothing else."""
        a, s = self.arch, self.scheduling
        text = "\n".join((
            "[arch constants]",
            f"{a.gfx_target}: {a.compute_units} CUs, wave{a.wavefront_size}, "
            f"{a.simds_per_cu} SIMD/CU, LDS {a.lds_bytes_per_cu // 1024} KiB/CU, "
            f"L2 {a.l2_bytes // (1024 * 1024)} MB {a.l2_topology}, {a.dies} die, "
            f"matrix: {a.matrix_cores}, HBM ~{a.hbm_bandwidth_gbps:g} GB/s",
            "[block scheduling policy]",
            f"XCDs: {s.xcds if s.xcds else 'none'}; dispatch: {s.dispatch_order}; "
            f"workgroup->CU: {s.workgroup_to_cu}",
            f"advice: {s.remap_advice}",
        ))
        if len(text) > MAX_RENDERED_CHARS:
            raise HardwareContextError("hardware context exceeds the curated cap")
        return text


def from_dict(payload: Mapping[str, Any]) -> HardwareContext:
    if payload.get("schema") != SCHEMA:
        raise HardwareContextError(f"schema must be {SCHEMA}")
    arch, sched = payload.get("arch_constants"), payload.get("block_scheduling_policy")
    if not isinstance(arch, Mapping) or not isinstance(sched, Mapping):
        raise HardwareContextError("both slots are required")
    if set(arch) != set(ARCH_KEYS):
        raise HardwareContextError(f"arch_constants keys must be exactly {ARCH_KEYS}")
    if set(sched) != set(SCHED_KEYS):
        raise HardwareContextError(f"block_scheduling_policy keys must be exactly {SCHED_KEYS}")
    return HardwareContext(ArchConstants(**arch), SchedulingPolicy(**sched),
                           str(payload.get("source", "")))


GFX90A_MI210 = HardwareContext(
    arch=ArchConstants(
        gfx_target="gfx90a (MI210, CDNA2)", compute_units=104, wavefront_size=64,
        simds_per_cu=4, lds_bytes_per_cu=64 * 1024, l2_bytes=8 * 1024 * 1024,
        l2_topology="unified", dies=1,
        matrix_cores="MFMA f64/f32/f16/bf16/i8; no fp8", hbm_bandwidth_gbps=1638.0),
    scheduling=SchedulingPolicy(
        xcds=0, dispatch_order="linear",
        workgroup_to_cu="hardware linear across all 104 CUs, one L2",
        remap_advice=("do not propose XCD-aware or chiplet workgroup remapping: "
                      "there are no XCDs, so any such remap is the identity map")),
    source=("rocminfo on this host (RVP-C4-8 box, rocm-kernel-authoring.md); "
            "intake-1221#record Fig 3 two-slot layout"),
)


if __name__ == "__main__":
    print(json.dumps(GFX90A_MI210.as_dict(), indent=1))
    print(GFX90A_MI210.render())
