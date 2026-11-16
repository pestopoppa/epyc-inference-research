"""RVP-ATT-1: MFMA efficiency of a hot loop from a decoded gfx90a ATT thread trace.

Desk-check answer (2026-11-16, no GPU): YES on gfx90a / ROCm 6.2, but through
**rocprofv2 --plugin att**, not rocprofv3. The rocprofv3 shipped with
rocprofiler-sdk 6.2.0-66 has no thread-trace CLI option (its help lists only tracing,
``-i`` counter input and ``--kernel-names``; the ATT service exists only as the
``amd_detail/thread_trace*.h`` SDK headers). The rocprofv2 ATT path is proven on the
MI210: canary ``codex-att-r4e-20261109`` wrote nonempty raw ``.att`` streams, and the
decoder in ``/mnt/raid0/llm/tmp/att-decoded-r5-prep-20261028`` turned them into the
per-instruction CSV this module reads (``Line,Instruction,Hitcount,Cycles,C++ Reference``).

Metric (intake-1823): total MFMA cycles over the average hot-loop iteration, per SIMD.

* A loop is a backward branch (``s_branch``/``s_cbranch_*`` to a label defined above it).
* Iterations = the branch row's ``Hitcount`` (every traced wave-iteration executes it).
* Loop cycles per iteration = the summed ``Cycles`` of the body rows / iterations.
  Cycles are the decoder's per-instruction stall+issue cycles of the traced waves.
* MFMA cycles per iteration = sum over MFMA rows of passes*4*Hitcount / iterations,
  passes from ``MFMA_PASSES``.
* The hot loop is the one with the most MFMA cycles per iteration.

The trace covers the SIMD selected by SIMD_SELECT (a bitmask on gfx90a; 0x1 = SIMD0),
so the ratio is per traced SIMD. The loop-header row is kept in the denominator: on the
canary it carries 8680 of 20460 loop cycles, and dropping it gives a ratio of 1.39 (>1),
so those are steady-state stall cycles of the accumulator chain the decoder books on the
header MFMA, not a one-off entry latency.

Reference value (decoded canary ``mfma_peak``, 8 x v_mfma_f32_16x16x16f16 per
iteration, 64 wave-iterations, se0 and se4): 256 MFMA cycles over 319.7 loop cycles per
iteration = 0.80.

The cycle table is declared, not measured here. 16x16x16f16 = 8 passes is cited from
LLVM SISchedule.td:259 at rocm-6.2.0 (handoff RVP-ATT-1). The remaining entries follow
the same gfx90a scheduling classes (Write2PassMAI/Write4PassMAI/Write8PassMAI/
Write16PassMAI); an MFMA mnemonic not in the table is refused, never guessed.
"""
from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from typing import Any, Sequence

SCHEMA = "epyc.autokernel.att_mfma_efficiency.v1"
CYCLES_PER_PASS = 4

MFMA_PASSES: dict[str, int] = {
    # 2 passes (8 cycles)
    "v_mfma_f32_4x4x1f32": 2, "v_mfma_f32_4x4x4f16": 2, "v_mfma_i32_4x4x4i8": 2,
    "v_mfma_f32_4x4x2bf16": 2, "v_mfma_f32_4x4x4bf16_1k": 2,
    # 4 passes (16 cycles)
    "v_mfma_f64_4x4x4f64": 4,
    # 8 passes (32 cycles)
    "v_mfma_f32_16x16x1f32": 8, "v_mfma_f32_16x16x4f32": 8, "v_mfma_f32_16x16x4f16": 8,
    "v_mfma_f32_16x16x16f16": 8, "v_mfma_i32_16x16x4i8": 8, "v_mfma_i32_16x16x16i8": 8,
    "v_mfma_f32_16x16x2bf16": 8, "v_mfma_f32_16x16x8bf16": 8,
    "v_mfma_f32_16x16x4bf16_1k": 8, "v_mfma_f32_16x16x16bf16_1k": 8,
    "v_mfma_f64_16x16x4f64": 8,
    # 16 passes (64 cycles)
    "v_mfma_f32_32x32x1f32": 16, "v_mfma_f32_32x32x2f32": 16, "v_mfma_f32_32x32x4f16": 16,
    "v_mfma_f32_32x32x8f16": 16, "v_mfma_i32_32x32x4i8": 16, "v_mfma_i32_32x32x8i8": 16,
    "v_mfma_f32_32x32x2bf16": 16, "v_mfma_f32_32x32x4bf16": 16,
    "v_mfma_f32_32x32x4bf16_1k": 16, "v_mfma_f32_32x32x8bf16_1k": 16,
}

_BRANCH = re.compile(r"^s_(?:c)?branch(?:_\w+)?\s+(\.L\w+)")
_LABEL = re.compile(r"^(\.L\w+):$")


class AttError(ValueError):
    pass


def parse_rows(text: str) -> list[dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(text))
    need = {"Line", "Instruction", "Hitcount", "Cycles"}
    if not reader.fieldnames or need - set(reader.fieldnames):
        raise AttError(f"decoded ATT csv lacks {sorted(need - set(reader.fieldnames or []))}")
    return [{"line": int(r["Line"]), "inst": r["Instruction"].strip(),
             "hits": int(r["Hitcount"]), "cycles": int(r["Cycles"])} for r in reader]


def mnemonic(inst: str) -> str:
    return inst.split(None, 1)[0] if inst else ""


def mfma_passes(inst: str) -> int | None:
    m = mnemonic(inst)
    if not m.startswith("v_mfma"):
        return None
    if m not in MFMA_PASSES:
        raise AttError(f"MFMA {m!r} has no declared gfx90a pass count")
    return MFMA_PASSES[m]


def loops(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels = {}
    out = []
    for i, row in enumerate(rows):
        lab = _LABEL.match(row["inst"])
        if lab:
            labels[lab.group(1)] = i
            continue
        br = _BRANCH.match(row["inst"])
        if br and br.group(1) in labels and labels[br.group(1)] < i:
            start = labels[br.group(1)]
            body = [r for r in rows[start + 1:i + 1] if not _LABEL.match(r["inst"])]
            iters = row["hits"]
            if iters <= 0:
                continue
            mfma_cycles = 0
            n_mfma = 0
            for r in body:
                p = mfma_passes(r["inst"])
                if p is not None:
                    mfma_cycles += p * CYCLES_PER_PASS * r["hits"]
                    n_mfma += 1
            executed = [r for r in body if r["hits"]]
            total = sum(r["cycles"] for r in executed)
            out.append({
                "label": br.group(1), "first_line": rows[start]["line"],
                "last_line": row["line"], "iterations": iters, "mfma_static": n_mfma,
                "mfma_cycles_per_iter": mfma_cycles / iters,
                "loop_cycles_per_iter": total / iters,
                "efficiency": (mfma_cycles / total) if total else None,
            })
    return out


def analyze(text: str, *, source: str = "") -> dict[str, Any]:
    found = loops(parse_rows(text))
    with_mfma = [lp for lp in found if lp["mfma_static"]]
    hot = max(with_mfma, key=lambda lp: lp["mfma_cycles_per_iter"]) if with_mfma else None
    return {"schema": SCHEMA, "source": source, "loops": found, "hot_loop": hot,
            "mfma_efficiency": None if hot is None else hot["efficiency"],
            "status": "measured" if hot else "no_mfma_loop"}


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="MFMA efficiency of a decoded gfx90a ATT csv")
    ap.add_argument("csv", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    doc = analyze(args.csv.read_text(), source=str(args.csv))
    text = json.dumps(doc, indent=1, sort_keys=True)
    if args.json:
        args.json.write_text(text + "\n")
    print(text)
    return 0 if doc["status"] == "measured" else 1


if __name__ == "__main__":
    raise SystemExit(main())
