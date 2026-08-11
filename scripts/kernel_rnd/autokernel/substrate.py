#!/usr/bin/env python3
"""Validated hardware facts for AutoKernel planning; no probe or process path."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class SubstrateFactError(ValueError):
    pass


def _positive(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value) or value <= 0:
        raise SubstrateFactError(f"{key} must be a finite positive number")
    return float(value)


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


def load(path: str | Path | None = None) -> SubstrateFacts:
    source = Path(path) if path is not None else Path(__file__).with_name(
        "substrate_facts.json")
    return SubstrateFacts(json.loads(source.read_text(encoding="utf-8")))
