#!/usr/bin/env python3
"""Validated hardware facts for AutoKernel planning; no probe or process path."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOFLINE_ROLE = "diagnostic_and_routing_only"
SPEC_BASIS = "datasheet_bandwidth"
ACHIEVABLE_BASIS = "measured_achievable_bandwidth"
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_POOLED_QUANT_NAMES = frozenset({"all", "any", "mixed", "pooled"})


class SubstrateFactError(ValueError):
    pass


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value) or value <= 0:
        raise SubstrateFactError(f"{name} must be a finite positive number")
    return float(value)


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise SubstrateFactError(f"{name} must be a lowercase identifier")
    return value


def _quant(value: Any) -> str:
    value = _identifier(value, "quantization")
    if value in _POOLED_QUANT_NAMES:
        raise SubstrateFactError(
            "roofline observations must name one exact quantization, never a pooled quant")
    return value


def _positive(row: Mapping[str, Any], key: str) -> float:
    return _finite_positive(row.get(key), key)


def _receipt(row: Mapping[str, Any]) -> str:
    value = row.get("measured_receipt")
    if not isinstance(value, str) or not value.strip():
        raise SubstrateFactError("every measured substrate fact needs a receipt")
    return value


@dataclass(frozen=True)
class SubstrateFacts:
    document: Mapping[str, Any]

    def __post_init__(self) -> None:
        doc = self.document
        if doc.get("schema") != "epyc.autokernel.substrate_facts.v1":
            raise SubstrateFactError("unexpected substrate-facts schema")
        if doc.get("hardware") != "AMD Instinct MI210 (gfx90a)":
            raise SubstrateFactError("hardware identity must name the MI210 gfx90a substrate")
        if doc.get("numa_node") != 1:
            raise SubstrateFactError("MI210 numa_node must be the sysfs-grounded node 1")
        facts = doc.get("facts")
        derived = doc.get("derived")
        if not isinstance(facts, Mapping) or not isinstance(derived, Mapping):
            raise SubstrateFactError("facts and derived must be mappings")
        compute = facts.get("compute_tflops")
        bandwidth = facts.get("memory_bandwidth_gbps")
        pcie = facts.get("pcie_gbps")
        if not all(isinstance(row, Mapping) for row in (compute, bandwidth, pcie)):
            raise SubstrateFactError("compute, bandwidth and PCIe facts are required")
        measured_compute = _positive(compute, "measured")
        spec_compute = _positive(compute, "datasheet")
        measured_bw = _positive(bandwidth, "measured")
        spec_bw = _positive(bandwidth, "datasheet")
        for row in (compute, bandwidth, pcie):
            _receipt(row)
        _positive(pcie, "h2d_measured")
        _positive(pcie, "d2h_measured")
        measured_ridge = _positive(derived, "ridge_flop_per_byte_measured_basis")
        spec_ridge = _positive(derived, "ridge_flop_per_byte_datasheet_basis")
        if not math.isclose(measured_ridge, measured_compute * 1000 / measured_bw,
                            rel_tol=5e-4):
            raise SubstrateFactError("measured-basis ridge does not rederive")
        if not math.isclose(spec_ridge, spec_compute * 1000 / spec_bw, rel_tol=5e-4):
            raise SubstrateFactError("datasheet-basis ridge does not rederive")
        crossover = derived.get("batch_crossover_measured_basis")
        if not isinstance(crossover, Mapping) or set(crossover) != {"q4_k", "q8_0", "bf16"}:
            raise SubstrateFactError("batch crossover must name q4_k, q8_0 and bf16")
        if not all(isinstance(value, int) and not isinstance(value, bool) and value > 0
                   for value in crossover.values()):
            raise SubstrateFactError("batch crossover values must be positive integers")


@dataclass(frozen=True)
class QuantRooflineObservation:
    """One local decode observation with both honest roofline denominators.

    ``bytes_per_token`` is quant-specific by construction.  Keeping it on the
    observation makes it impossible to pool Q4, Q8 and BF16 behind one inferred
    denominator.  The measured-bandwidth basis describes local headroom; the
    datasheet basis is the only basis permitted for a cross-vendor comparison.
    """

    hardware: str
    quantization: str
    workload_regime: str
    measured_tps: float
    bytes_per_token: float
    measured_bandwidth_gbps: float
    datasheet_bandwidth_gbps: float
    measurement_receipt: str

    def __post_init__(self) -> None:
        if not isinstance(self.hardware, str) or not self.hardware.strip():
            raise SubstrateFactError("hardware must be non-empty")
        object.__setattr__(self, "quantization", _quant(self.quantization))
        object.__setattr__(self, "workload_regime",
                           _identifier(self.workload_regime, "workload_regime"))
        for name in ("measured_tps", "bytes_per_token", "measured_bandwidth_gbps",
                     "datasheet_bandwidth_gbps"):
            object.__setattr__(self, name, _finite_positive(getattr(self, name), name))
        if not isinstance(self.measurement_receipt, str) \
                or not self.measurement_receipt.strip():
            raise SubstrateFactError("roofline observations require a measurement receipt")
        if self.achievable_utilization > 1.0 + 1e-9:
            raise SubstrateFactError(
                "measured throughput exceeds the achievable-bandwidth roof; "
                "bytes_per_token or the measurement basis is inconsistent")
        if self.spec_utilization > 1.0 + 1e-9:
            raise SubstrateFactError(
                "measured throughput exceeds the datasheet-bandwidth roof; "
                "bytes_per_token or the measurement basis is inconsistent")

    @property
    def achievable_roof_tps(self) -> float:
        return self.measured_bandwidth_gbps * 1e9 / self.bytes_per_token

    @property
    def spec_roof_tps(self) -> float:
        return self.datasheet_bandwidth_gbps * 1e9 / self.bytes_per_token

    @property
    def achievable_utilization(self) -> float:
        return self.measured_tps / self.achievable_roof_tps

    @property
    def spec_utilization(self) -> float:
        return self.measured_tps / self.spec_roof_tps


@dataclass(frozen=True)
class CudaRooflineAnchor:
    """A primary-source CUDA result that reports absolute spec-basis utilization."""

    quantization: str
    workload_regime: str
    utilization_spec_basis: float
    hardware: str
    implementation: str
    source_url: str
    source_scope: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "quantization", _quant(self.quantization))
        object.__setattr__(self, "workload_regime",
                           _identifier(self.workload_regime, "workload_regime"))
        value = _finite_positive(
            self.utilization_spec_basis, "utilization_spec_basis")
        if value > 1:
            raise SubstrateFactError("CUDA roofline utilization cannot exceed 1")
        object.__setattr__(self, "utilization_spec_basis", value)
        for name in ("hardware", "implementation", "source_url", "source_scope"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise SubstrateFactError(f"{name} must be non-empty")
        if not self.source_url.startswith("https://"):
            raise SubstrateFactError("CUDA anchor must cite an https primary source")


@dataclass(frozen=True)
class CudaAnchorGap:
    """Why a tempting CUDA result is not an absolute, exact-quant anchor."""

    quantization: str
    workload_regime: str
    reason: str
    source_url: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "quantization", _quant(self.quantization))
        object.__setattr__(self, "workload_regime",
                           _identifier(self.workload_regime, "workload_regime"))
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise SubstrateFactError("CUDA anchor gap needs a reason")
        if self.source_url and not self.source_url.startswith("https://"):
            raise SubstrateFactError("CUDA anchor-gap source must use https")


@dataclass(frozen=True)
class QuantRooflineComparison:
    """Per-quant local headroom plus an optional exact CUDA target.

    This is deliberately not a gate input.  A missing exact-quant anchor is an
    explicit ``COULD_NOT_CHECK`` cell, never an excuse to borrow BF16's 78% for
    Q4 or to treat Marlin's relative 3.87x speedup as absolute bandwidth use.
    """

    observation: QuantRooflineObservation
    cuda_anchor: CudaRooflineAnchor | None
    anchor_gap: str | None
    role: str = ROOFLINE_ROLE
    cross_vendor_basis: str = SPEC_BASIS
    local_headroom_basis: str = ACHIEVABLE_BASIS

    def __post_init__(self) -> None:
        if self.role != ROOFLINE_ROLE:
            raise SubstrateFactError("roofline comparison is diagnostic/routing only")
        if self.cross_vendor_basis != SPEC_BASIS:
            raise SubstrateFactError("cross-vendor roofline comparisons must be spec-basis")
        if self.local_headroom_basis != ACHIEVABLE_BASIS:
            raise SubstrateFactError("local headroom must use measured achievable bandwidth")
        if (self.cuda_anchor is None) == (self.anchor_gap is None):
            raise SubstrateFactError(
                "roofline comparison needs exactly one of cuda_anchor or anchor_gap")
        if self.cuda_anchor is not None:
            if self.cuda_anchor.quantization != self.observation.quantization:
                raise SubstrateFactError("CUDA anchor quantization does not exactly match the cell")
            if self.cuda_anchor.workload_regime != self.observation.workload_regime:
                raise SubstrateFactError("CUDA anchor workload regime does not match the cell")

    @property
    def anchor_status(self) -> str:
        return "PASS" if self.cuda_anchor is not None else "COULD_NOT_CHECK"

    @property
    def local_achievable_headroom(self) -> float:
        return 1.0 - self.observation.achievable_utilization

    @property
    def target_utilization_spec_basis(self) -> float | None:
        return None if self.cuda_anchor is None \
            else self.cuda_anchor.utilization_spec_basis

    @property
    def utilization_gap_to_cuda(self) -> float | None:
        target = self.target_utilization_spec_basis
        return None if target is None else target - self.observation.spec_utilization

    @property
    def target_tps_on_local_spec_roof(self) -> float | None:
        target = self.target_utilization_spec_basis
        return None if target is None else target * self.observation.spec_roof_tps


BF16_CUDA_ANCHOR = CudaRooflineAnchor(
    quantization="bf16",
    workload_regime="batch1_single_sequence_decode",
    utilization_spec_basis=0.78,
    hardware="NVIDIA H100",
    implementation="Hazy Research Llama-3.2-1B megakernel",
    source_url="https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles",
    source_scope=(
        "Primary source reports 78% memory-bandwidth use for batch-1, single-sequence "
        "Llama-3.2-1B BF16 decode on H100; it is not a quantized-kernel result."),
)

KNOWN_CUDA_ANCHOR_GAPS = (
    CudaAnchorGap(
        quantization="q4_k",
        workload_regime="batch1_single_sequence_decode",
        reason=(
            "Marlin reports an INT4-g128 relative speedup (3.87x ideal including scale "
            "overhead), not absolute spec-basis bandwidth utilization, and INT4-g128 is "
            "not GGUF Q4_K; neither denominator may be borrowed."),
        source_url="https://github.com/IST-DASLab/marlin"),
    CudaAnchorGap(
        quantization="q8_0",
        workload_regime="batch1_single_sequence_decode",
        reason=(
            "No primary source with exact Q8_0 semantics and absolute spec-basis CUDA "
            "bandwidth utilization is registered.")),
)


def compare_per_quant(
        observations: Sequence[QuantRooflineObservation],
        *,
        anchors: Sequence[CudaRooflineAnchor] = (BF16_CUDA_ANCHOR,),
        known_gaps: Sequence[CudaAnchorGap] = KNOWN_CUDA_ANCHOR_GAPS,
) -> tuple[QuantRooflineComparison, ...]:
    """Build an exact-quant surface; never pool or silently borrow an anchor."""
    anchor_map: dict[tuple[str, str], CudaRooflineAnchor] = {}
    for anchor in anchors:
        if not isinstance(anchor, CudaRooflineAnchor):
            raise TypeError("anchors must contain CudaRooflineAnchor objects")
        key = (anchor.quantization, anchor.workload_regime)
        if key in anchor_map:
            raise SubstrateFactError(f"duplicate CUDA anchor for {key}")
        anchor_map[key] = anchor
    gap_map: dict[tuple[str, str], CudaAnchorGap] = {}
    for gap in known_gaps:
        if not isinstance(gap, CudaAnchorGap):
            raise TypeError("known_gaps must contain CudaAnchorGap objects")
        key = (gap.quantization, gap.workload_regime)
        if key in gap_map:
            raise SubstrateFactError(f"duplicate CUDA anchor gap for {key}")
        gap_map[key] = gap

    seen: set[tuple[str, str]] = set()
    cells = []
    for observation in observations:
        if not isinstance(observation, QuantRooflineObservation):
            raise TypeError("observations must contain QuantRooflineObservation objects")
        key = (observation.quantization, observation.workload_regime)
        if key in seen:
            raise SubstrateFactError(f"duplicate local roofline cell for {key}")
        seen.add(key)
        anchor = anchor_map.get(key)
        if anchor is not None:
            cells.append(QuantRooflineComparison(
                observation=observation, cuda_anchor=anchor, anchor_gap=None))
            continue
        gap = gap_map.get(key)
        reason = gap.reason if gap is not None else (
            "no exact-quant, same-regime absolute spec-basis CUDA anchor is registered")
        cells.append(QuantRooflineComparison(
            observation=observation, cuda_anchor=None, anchor_gap=reason))
    return tuple(cells)


def load(path: str | Path | None = None) -> SubstrateFacts:
    source = Path(path) if path is not None else Path(__file__).with_name(
        "substrate_facts.json")
    return SubstrateFacts(json.loads(source.read_text(encoding="utf-8")))
