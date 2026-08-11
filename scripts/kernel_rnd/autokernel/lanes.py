#!/usr/bin/env python3
"""Typed screening-lane registry and transfer calibration for AutoKernel.

This module launches nothing.  A lane is a declared resource/isolation shape,
not evidence that the shape predicts full-machine performance.  Only a measured
``RankCalibration`` can make a partial lane eligible for a change class, and
only a full lane can verify a performance claim.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from .execution import cpu_region_claim


CPU_REGION = "cpu_region"
GPU_DEVICE = "gpu_device"
CLAIM_TYPES = frozenset({CPU_REGION, GPU_DEVICE})

FULL = "full"
HALF = "half"
QUARTER = "quarter"
PARTITIONED = "partitioned"
DEVICE = "device"
MACHINE_SUBSETS = frozenset({FULL, HALF, QUARTER, PARTITIONED, DEVICE})


class LaneError(ValueError):
    """A lane declaration or calibration is unsafe or incomplete."""


@dataclass(frozen=True)
class LaneSpec:
    lane_id: str
    resource_type: str
    cost: float
    capacity: int
    proxy_for: str
    machine_subset: str
    cpu_list: str | None = None
    membind_nodes: tuple[int, ...] = ()
    use_mmap: bool = False
    verification_lane: bool = False
    aa_control_required: bool = True
    availability_ref: str | None = None

    def __post_init__(self) -> None:
        if not self.lane_id.strip() or not self.proxy_for.strip():
            raise LaneError("lane_id and proxy_for must be non-empty")
        if self.resource_type not in CLAIM_TYPES:
            raise LaneError(f"unknown resource_type {self.resource_type!r}")
        if self.machine_subset not in MACHINE_SUBSETS:
            raise LaneError(f"unknown machine_subset {self.machine_subset!r}")
        if isinstance(self.cost, bool) or not isinstance(self.cost, (int, float)) \
                or not math.isfinite(self.cost) or self.cost <= 0:
            raise LaneError("lane cost must be finite and positive")
        if isinstance(self.capacity, bool) or not isinstance(self.capacity, int) \
                or self.capacity <= 0:
            raise LaneError("lane capacity must be a positive integer")
        if self.resource_type == CPU_REGION:
            if not self.cpu_list:
                raise LaneError("a CPU lane requires an explicit cpu_list")
            cpu_region_claim.parse_cpu_list(self.cpu_list)
            if not self.membind_nodes:
                raise LaneError("a CPU lane requires explicit membind_nodes")
            if self.use_mmap:
                raise LaneError("a CPU lane must use --no-mmap for isolated placement")
            if any(isinstance(node, bool) or not isinstance(node, int) or node < 0
                   for node in self.membind_nodes):
                raise LaneError("membind_nodes must contain non-negative integers")
        elif self.machine_subset != DEVICE:
            raise LaneError("a GPU device lane must declare machine_subset='device'")
        if self.verification_lane and self.machine_subset not in {FULL, DEVICE}:
            raise LaneError("a partial-machine lane may screen but never verify")


@dataclass(frozen=True)
class RankCalibration:
    """Measured ordering fidelity from one cheap lane to a verification lane."""

    change_class: str
    screening_lane_id: str
    verification_lane_id: str
    candidate_ids: tuple[str, ...]
    screening_order: tuple[str, ...]
    verification_order: tuple[str, ...]
    rank_fidelity: float
    event_ids: tuple[str, ...]
    aa_control_passed: bool
    prediction_ref: str

    def __post_init__(self) -> None:
        if not self.change_class.strip() or not self.prediction_ref.strip():
            raise LaneError("calibration needs a change class and preregistered prediction")
        if len(self.candidate_ids) < 2 or len(set(self.candidate_ids)) != len(
                self.candidate_ids):
            raise LaneError("calibration needs at least two distinct candidates")
        expected = set(self.candidate_ids)
        if set(self.screening_order) != expected or set(self.verification_order) != expected:
            raise LaneError("both rank orders must contain the fixed candidate set exactly once")
        if len(self.screening_order) != len(expected) or len(self.verification_order) != len(
                expected):
            raise LaneError("rank orders must not contain duplicates")
        measured = spearman_rank_fidelity(self.screening_order, self.verification_order)
        if not math.isclose(self.rank_fidelity, measured, rel_tol=0, abs_tol=1e-12):
            raise LaneError(
                f"rank_fidelity {self.rank_fidelity} does not match recomputed {measured}")
        if not self.event_ids:
            raise LaneError("calibration must cite its measurement events")


@dataclass(frozen=True)
class ProfiledOp:
    op: str
    wall_share: float
    profile_event_id: str

    def __post_init__(self) -> None:
        if not self.op.strip() or not self.profile_event_id.strip():
            raise LaneError("profiled op needs an op and profile event")
        if isinstance(self.wall_share, bool) or not isinstance(
                self.wall_share, (int, float)) or not math.isfinite(self.wall_share) \
                or not 0 <= self.wall_share <= 1:
            raise LaneError("op wall_share must be finite in [0,1]")


@dataclass(frozen=True)
class OpScreeningAssignment:
    candidate_id: str
    op: str
    lane_id: str
    wave: int
    profile_event_id: str
    full_verification_required: bool = True


def spearman_rank_fidelity(first: Sequence[str], second: Sequence[str]) -> float:
    """Spearman rho for two complete, tie-free rankings of the same items."""
    if len(first) < 2 or len(first) != len(second) or set(first) != set(second):
        raise LaneError("rankings must contain the same two-or-more distinct items")
    if len(set(first)) != len(first) or len(set(second)) != len(second):
        raise LaneError("rankings must be tie-free permutations")
    right = {item: rank for rank, item in enumerate(second)}
    squared = sum((rank - right[item]) ** 2 for rank, item in enumerate(first))
    count = len(first)
    return 1.0 - (6.0 * squared) / (count * (count * count - 1))


def validate_concurrent_cpu_lanes(lanes: Sequence[LaneSpec]) -> None:
    """Refuse shared physical CPUs, mmap, or implicit binding.

    Several independent lanes may intentionally bind memory to the same NUMA
    node; the historical 48x4t sweep did exactly that. That shared bandwidth is
    why rank fidelity must be calibrated, not a reason to claim the CPU sets
    overlap.
    """
    cpu_lanes = [lane for lane in lanes if lane.resource_type == CPU_REGION]
    for lane in cpu_lanes:
        # Re-run the isolation checks here because this is the concurrency seam,
        # and callers may later construct LaneSpec-compatible objects.
        if lane.use_mmap or not lane.cpu_list or not lane.membind_nodes:
            raise LaneError(
                f"lane {lane.lane_id!r} needs --no-mmap, cpu_list and membind_nodes")
    needs_sibling_map = any(
        any(cpu > cpu_region_claim.MAX_PHYSICAL_CORE
            for cpu in cpu_region_claim.parse_cpu_list(lane.cpu_list))
        for lane in cpu_lanes)
    sibling_map = cpu_region_claim.read_sibling_map() if needs_sibling_map else None
    physical_by_lane = {
        lane.lane_id: cpu_region_claim.physical_cores(
            cpu_region_claim.parse_cpu_list(lane.cpu_list), sibling_map)
        for lane in cpu_lanes
    }
    for index, left in enumerate(cpu_lanes):
        for right in cpu_lanes[index + 1:]:
            overlap = physical_by_lane[left.lane_id].intersection(
                physical_by_lane[right.lane_id])
            if overlap:
                raise LaneError(
                    f"CPU lanes {left.lane_id!r} and {right.lane_id!r} overlap physical CPUs "
                    f"{sorted(overlap)}")


def select_screening_lane(
        registry: Mapping[str, LaneSpec], change_class: str,
        calibrations: Sequence[RankCalibration], *, minimum_rank_fidelity: float
) -> LaneSpec:
    """Choose the cheapest evidenced proxy; otherwise return full verification.

    A passing cross-lane A/A is required, but it never substitutes for a
    change-class-specific rank calibration.  No blanket partition haircut is
    represented because such a correction would assume class independence.
    """
    if not 0 <= minimum_rank_fidelity <= 1:
        raise LaneError("minimum_rank_fidelity must be in [0,1]")
    eligible: list[LaneSpec] = []
    for calibration in calibrations:
        if calibration.change_class != change_class \
                or calibration.rank_fidelity < minimum_rank_fidelity \
                or not calibration.aa_control_passed:
            continue
        lane = registry.get(calibration.screening_lane_id)
        verifier = registry.get(calibration.verification_lane_id)
        if lane is None or verifier is None or not verifier.verification_lane:
            continue
        if lane.proxy_for != verifier.lane_id:
            continue
        eligible.append(lane)
    if eligible:
        return min(eligible, key=lambda lane: (lane.cost, lane.lane_id))
    full = [lane for lane in registry.values()
            if lane.verification_lane and lane.resource_type == CPU_REGION]
    if not full:
        raise LaneError("registry has no full CPU verification lane")
    return min(full, key=lambda lane: (lane.cost, lane.lane_id))


def plan_op_screening(
        profile: Sequence[ProfiledOp], candidate_ids: Sequence[str],
        registry: Mapping[str, LaneSpec], calibrations: Sequence[RankCalibration], *,
        change_class: str, minimum_rank_fidelity: float
) -> tuple[OpScreeningAssignment, ...]:
    """Fan candidates over evidenced lanes for the single highest-share op.

    This is a plan only. It cannot profile, acquire a claim, or run
    ``test-backend-ops``. Candidates that outnumber eligible lanes are assigned
    to later waves; every row retains the mandatory full-instance verification.
    """
    if not profile:
        raise LaneError("op-level fan-out requires a measured profile")
    candidates = tuple(candidate_ids)
    if not candidates or any(not isinstance(item, str) or not item.strip()
                             for item in candidates):
        raise LaneError("candidate_ids must be non-empty strings")
    if len(set(candidates)) != len(candidates):
        raise LaneError("candidate_ids must be distinct")
    bottleneck = sorted(profile, key=lambda row: (-row.wall_share, row.op))[0]
    lanes: list[LaneSpec] = []
    for calibration in calibrations:
        lane = registry.get(calibration.screening_lane_id)
        verifier = registry.get(calibration.verification_lane_id)
        if calibration.change_class == change_class \
                and calibration.rank_fidelity >= minimum_rank_fidelity \
                and calibration.aa_control_passed and lane is not None \
                and verifier is not None and verifier.verification_lane \
                and lane.proxy_for == verifier.lane_id:
            lanes.append(lane)
    lanes = sorted({lane.lane_id: lane for lane in lanes}.values(),
                   key=lambda lane: (lane.cost, lane.lane_id))
    if not lanes:
        lanes = [select_screening_lane(
            registry, change_class, calibrations,
            minimum_rank_fidelity=minimum_rank_fidelity)]
    return tuple(OpScreeningAssignment(
        candidate, bottleneck.op, lanes[index % len(lanes)].lane_id,
        index // len(lanes), bottleneck.profile_event_id)
        for index, candidate in enumerate(candidates))


def may_gate_performance_claim(lane: LaneSpec) -> bool:
    """Partitioned lanes rank candidates; only full resources carry claims."""
    return lane.verification_lane and lane.machine_subset in {FULL, DEVICE}


def default_lane_registry() -> dict[str, LaneSpec]:
    """Current and historically exercised host lanes; no transfer is presumed.

    The 4x48t, 8x24t, 16x12t, 32x6t and 48x4t shapes are preserved by
    ``progress/2026-04/2026-04-24.md`` and its raw sweep artifacts. Those runs
    establish that the host can execute the fan-out, not that any shape ranks
    kernel changes like the full machine; ``select_screening_lane`` still
    requires new, change-class-specific calibration.
    """
    rows = {
        "cpu-full": LaneSpec("cpu-full", CPU_REGION, 4.0, 1, "cpu-full", FULL,
                             "0-95", (0, 1, 2, 3), False, True),
        "gpu-full": LaneSpec("gpu-full", GPU_DEVICE, 1.0, 1, "gpu-full", DEVICE,
                             verification_lane=True),
    }
    for index in range(4):
        start = index * 24
        rows[f"cpu-q{index}"] = LaneSpec(
            f"cpu-q{index}", CPU_REGION, 1.0, 1, "cpu-full", QUARTER,
            f"{start}-{start + 23}", (index,), False, False)
    rows.update(historical_split_lane_registry())
    return rows


def historical_split_lane_registry() -> dict[str, LaneSpec]:
    """Concrete lanes for every successfully exercised concurrent split depth."""
    source = "progress/2026-04/2026-04-24.md#concurrent-split-sweep"
    rows: dict[str, LaneSpec] = {}
    # (instances, logical threads, physical cores per instance). Each lane pins
    # both SMT siblings of its physical cores, matching the preserved scripts.
    for instances, threads, physical_count in (
            (4, 48, 24), (8, 24, 12), (16, 12, 6), (32, 6, 3), (48, 4, 2)):
        per_node = 24 // physical_count
        for index in range(instances):
            node = index // per_node
            within = index % per_node
            first = node * 24 + within * physical_count
            physical = range(first, first + physical_count)
            logical = tuple(physical) + tuple(cpu + 96 for cpu in physical)
            lane_id = f"cpu-{instances}x{threads}t-{index:02d}"
            rows[lane_id] = LaneSpec(
                # Cost is in quarter-machine physical-core equivalents: this
                # makes a two-core historical lane cheaper than a 24-core
                # quarter, without pretending its SMT threads are extra cores.
                lane_id, CPU_REGION, physical_count / 24.0, 1, "cpu-full", PARTITIONED,
                cpu_region_claim.render_cpu_list(logical), (node,), False, False,
                availability_ref=source)
    return rows
