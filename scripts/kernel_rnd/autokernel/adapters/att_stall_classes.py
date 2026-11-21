"""ROCPROF-ATT-ADAPTER-1: rocprofv2 ATT decoded CSV -> amd/skills stall-class cycle shares.

ROCm 6.2 has no rocprofv3 thread trace; stall attribution on the MI210 goes through the rocprofv2
ATT decode (operator decision 2026-11-20, ROCM-PROFILER-DECISION option d). The decoded CSV columns
are ``Line, Instruction, Hitcount, Cycles, C++ Reference``. ``Cycles`` is the total ATT cycles
attributed to the instruction (this CSV has no separate stall/issue split, so a class share is a
share of attributed cycles, NOT a stall-only share).

Class mapping is a verbatim port of ``Instruction.stall_type`` in
amd/skills ``skills/flydsl-kernel-trace-analysis/scripts/hotspot_analyzer.py`` (same precedence:
s_waitcnt vmcnt -> VMEM-wait, lgkmcnt -> LDS/SMEM-wait, expcnt -> EXP-wait, bare waitcnt ->
waitcnt, then barrier, VMEM-load, VMEM-store, LDS, SMEM, MFMA/FMA, other).
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

REQUIRED_COLUMNS = ("Line", "Instruction", "Hitcount", "Cycles")
CLASSES = ("VMEM-load", "VMEM-wait", "LDS/SMEM-wait", "EXP-wait", "waitcnt", "barrier",
           "VMEM-store", "LDS", "SMEM", "MFMA/FMA", "other")


class AttCsvError(ValueError):
    """The ATT decoded CSV is malformed (typed refusal, never a silent zero)."""


def classify_instruction(asm: str) -> str:
    """hotspot_analyzer.py Instruction.stall_type, ported unchanged."""
    asm = asm.lower()
    if "s_waitcnt" in asm:
        if "vmcnt" in asm:
            return "VMEM-wait"
        if "lgkmcnt" in asm:
            return "LDS/SMEM-wait"
        if "expcnt" in asm:
            return "EXP-wait"
        return "waitcnt"
    if "s_barrier" in asm or "s_wait_idle" in asm:
        return "barrier"
    if "buffer_load" in asm or "global_load" in asm or "flat_load" in asm:
        return "VMEM-load"
    if "buffer_store" in asm or "global_store" in asm:
        return "VMEM-store"
    if "ds_read" in asm or "ds_write" in asm:
        return "LDS"
    if "s_load" in asm or "s_store" in asm:
        return "SMEM"
    if "v_mfma" in asm or "v_fma" in asm:
        return "MFMA/FMA"
    return "other"


def class_cycles(csv_path: str | Path) -> dict[str, int]:
    """Total attributed cycles per class (all CLASSES keys present, zeros kept)."""
    totals: dict[str, int] = defaultdict(int)
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or ())]
        if missing:
            raise AttCsvError(f"ATT CSV missing columns {missing}")
        for number, row in enumerate(reader, start=2):
            try:
                cycles = int(row["Cycles"])
            except (TypeError, ValueError):
                raise AttCsvError(f"line {number}: non-integer Cycles {row['Cycles']!r}") from None
            if cycles < 0:
                raise AttCsvError(f"line {number}: negative Cycles")
            totals[classify_instruction(row["Instruction"] or "")] += cycles
    return {name: totals.get(name, 0) for name in CLASSES}


def class_cycle_shares(csv_path: str | Path) -> dict[str, float]:
    """Per-class share (0..1) of the CSV's total attributed cycles.

    Raises AttCsvError when the total is zero: a decode with no cycles has no shares to report
    (an empty capture must not read as a clean all-zero profile).
    """
    cycles = class_cycles(csv_path)
    total = sum(cycles.values())
    if total == 0:
        raise AttCsvError("ATT CSV carries zero total cycles")
    return {name: value / total for name, value in cycles.items()}
