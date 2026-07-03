#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import server_np_sweep as sweep


def test_host_health_accepts_canonical_numa_balancing_disabled() -> None:
    warnings = sweep.host_health_warnings(
        {
            "uptime_seconds": 3600,
            "numa_balancing": "0",
            "existing_llama_processes": [],
        }
    )

    assert warnings == []


def test_host_health_flags_numa_balancing_enabled() -> None:
    warnings = sweep.host_health_warnings(
        {
            "uptime_seconds": 3600,
            "numa_balancing": "1",
            "existing_llama_processes": [],
        }
    )

    assert warnings == [
        "kernel.numa_balancing='1'; expected '0' for canonical NUMA-interleave CPU benchmarking"
    ]

