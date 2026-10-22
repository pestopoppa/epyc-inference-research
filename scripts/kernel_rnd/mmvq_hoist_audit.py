#!/usr/bin/env python3
"""Static weight-hoist audit of HIP ``mul_mat_vec_q`` on gfx90a (INF37-LB1; intake-2386/2393).

Question: in the compiled ``mul_mat_vec_q<type, ncols_dst>`` hot loop, is each weight word loaded
(and unpacked) once per row for ALL ``ncols_dst`` activation columns, or once per column?  The
Locals ``qmv_multi`` kernel loads and converts each weight once for all rows; if the llama.cpp
cells repeat the weight VMEM loads per column, a weight-hoisted multi-column MMVQ is a seed
(Q4_K MMVQ at 8 rows measures 9.4 % of the HBM roof vs 46.5 % at 1 row in the lb1 profile).

This tool NEVER builds and NEVER touches a GPU.  It reads an existing gfx90a library with
``gfx90a_isa_audit`` (same ELF/bundle reader, same ``llvm-objdump -d``), takes the hot loop of
each requested ``mul_mat_vec_q`` instance, and classifies every VMEM load in it as WEIGHT (its
address derives from kernarg ``vx``), ACTIVATION (``vy``) or OTHER by forward taint from the
kernel-argument loads ``s_load_* s[a:b], s[4:5], 0x0`` (vx = bytes 0..7, vy = bytes 8..15).
Carry registers (``vcc``, the carry-out SGPR pair of ``v_add_co`` / ``v_mad_u64``) carry no taint,
so two interleaved pointer chains cannot contaminate each other.

Counting basis.  One hot-loop iteration handles one weight block position for
``rows_per_thread`` rows x ``ncols_dst`` columns, so the headline numbers are

  w_loads_per_row   = weight VMEM loads per iteration / rows_per_thread
                      (FLAT in ncols_dst  => weights are hoisted out of the column loop;
                       growing ~linearly  => the weight load is repeated per column)
  w_loads_per_unit  = weight VMEM loads per iteration / (rows_per_thread * ncols_dst)
                      (the "per block per column" figure)

``rows_per_thread`` is taken from the v10 ``calc_rows_per_block`` GCN/CDNA2 table (mmvq.cu) and
cross-checked here against the v_dot count: dots / (rows_per_thread * ncols_dst) must be
constant across ncols_dst for one type, otherwise the row is flagged ``dot_unit_mismatch``.

    /usr/bin/python3 scripts/kernel_rnd/mmvq_hoist_audit.py \
        /mnt/raid0/llm/kernels/production/gpu/libggml-hip.so.0.16.0 \
        --json out.json --table out.txt
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gfx90a_isa_audit as isa  # noqa: E402

SCHEMA = "epyc.gfx90a.mmvq_hoist_audit.v1"
TYPES = {8: "Q8_0", 12: "Q4_K", 23: "IQ4_XS"}

_KERNEL_RE = re.compile(
    r"mul_mat_vec_qIL9ggml_type(\d+)ELi(\d+)ELb([01])ELb([01])ELi(\d+)ELb([01])ELb([01])ELb([01])E")

# Kernarg layout of mul_mat_vec_q (first two pointer args).
_KERNARG_LABELS = ((0, 8, "X"), (8, 16, "Y"))  # [lo, hi) byte ranges -> taint label


# ---------------------------------------------------------------- source-derived geometry


def rows_per_thread(type_id: int, ncols: int, small_k: bool, eight_warp: bool = False) -> int:
    """rows_per_cuda_block == rows_per_thread for the cells audited here (no halfwave rows).

    Mirrors v10 ``calc_rows_per_block`` for the GCN/CDNA2 table (mmvq.cu L916-L945).  The
    halfwave Q4_K cell (``small_k``, ncols 1, nwarps>=2) runs 2 rows per CTA but ONE row per
    thread, so it reports 1 here.
    """
    nwarps = 8 if (type_id == 8 and ncols == 1 and eight_warp) else (
        4 if (type_id == 8 and ncols == 1) else (2 if ncols <= 4 else 1))
    if type_id == 8 and ncols == 4 and not small_k:
        return 4
    if type_id == 8 and ncols == 1 and small_k:
        return 4
    if ncols == 1:
        if type_id == 12 and small_k and nwarps >= 2:
            return 1  # halfwave_rows: 2 rows/CTA, 1 row/thread
        return nwarps if small_k else 1
    return 2 if 2 <= ncols <= 8 else 1


# ---------------------------------------------------------------- operand parsing / taint


_REG = re.compile(r"([vs])(?:\[(\d+):(\d+)\]|(\d+))")
_CARRY_OPS = ("_co_", "mad_u64", "mad_i64")


def parse_regs(token: str) -> set[tuple[str, int]]:
    """Registers named in one operand token (``v[4:7]``, ``s9``).  vcc/exec/imm give nothing."""
    out: set[tuple[str, int]] = set()
    for kind, lo, hi, one in _REG.findall(token):
        if one:
            out.add((kind, int(one)))
        else:
            out.update((kind, i) for i in range(int(lo), int(hi) + 1))
    return out


def split_operands(args: str) -> list[str]:
    """Top-level comma split (brackets in ``v[4:7]`` hold no commas, but modifiers may)."""
    args = re.sub(r"\s+(offset|glc|slc|dlc|nt|sc0|sc1|op_sel_hi|op_sel|neg_lo|neg_hi|clamp|"
                  r"row_shr|row_ror|row_shl|quad_perm|row_mask|bank_mask|bound_ctrl|wave_\w+)[:\w\[\],.]*.*$",
                  "", args)
    return [t.strip() for t in args.split(",") if t.strip()]


def _is_vmem_load(op: str) -> bool:
    return op.startswith(("global_load", "flat_load", "buffer_load", "scratch_load"))


def _dest_and_srcs(ins) -> tuple[list[str], list[str], list[str]]:
    """(value destinations, carry destinations, source tokens) of one instruction."""
    op, ops = ins.op, split_operands(ins.args)
    if not ops:
        return [], [], []
    if op.startswith(("s_cbranch", "s_branch", "s_waitcnt", "s_barrier", "s_endpgm", "s_nop",
                      "s_setprio", "global_store", "flat_store", "buffer_store", "ds_write",
                      "ds_add", "s_store", "scratch_store", "s_dcache", "s_sleep")):
        return [], [], ops
    if _is_vmem_load(op) or op.startswith(("s_load", "s_buffer_load", "ds_read", "ds_bpermute",
                                           "ds_swizzle")):
        return [ops[0]], [], ops[1:]
    if op.startswith(("v_", "s_")) and any(c in op for c in _CARRY_OPS):
        carry = [ops[1]] if len(ops) > 1 else []
        srcs = ops[2:]
        # v_addc_co_*: the trailing carry-in (vcc / s pair) is a source too, but carries no taint
        return [ops[0]], carry, srcs
    return [ops[0]], [], ops[1:]


def classify_loads(insts: list, loop: tuple[int, int]) -> list[dict]:
    """Taint pass over ``insts`` (whole kernel, two sweeps for loop-carried values), then the
    VMEM loads of ``insts[loop[0]:loop[1]+1]`` labelled weight / activation / other."""
    label: dict[tuple[str, int], str] = {}

    def src_label(tokens: list[str]) -> str:
        got: set[str] = set()
        for tok in tokens:
            for reg in parse_regs(tok):
                got.update(label.get(reg, "").split("+") if reg in label else ())
        got.discard("")
        return "+".join(sorted(got))

    lo, hi = loop
    result: list[dict] = []
    for sweep in (0, 1):
        result = []
        for idx, ins in enumerate(insts):
            dests, carry, srcs = _dest_and_srcs(ins)
            if ins.op.startswith("s_load"):
                m = re.match(r"\s*s\[4:5\]\s*", srcs[0]) if srcs else None
                off = srcs[1] if len(srcs) > 1 else "0x0"
                if m and re.fullmatch(r"0x[0-9a-fA-F]+|\d+", off.strip()):
                    base = int(off, 0)
                    regs = sorted(parse_regs(dests[0]), key=lambda r: r[1])
                    for k, reg in enumerate(regs):
                        byte = base + 4 * k
                        lab = next((name for a, b, name in _KERNARG_LABELS if a <= byte < b), "")
                        if lab:
                            label[reg] = lab
                        else:
                            label.pop(reg, None)
                    continue
            if _is_vmem_load(ins.op):
                lab = src_label(srcs)
                if lo <= idx <= hi:
                    result.append({"addr": ins.addr, "op": ins.op, "label": lab or "other",
                                   "args": ins.args})
                for d in dests:
                    for reg in parse_regs(d):
                        label.pop(reg, None)  # loaded DATA carries no pointer taint
                continue
            lab = src_label(srcs)
            for d in dests:
                for reg in parse_regs(d):
                    if lab:
                        label[reg] = lab
                    else:
                        label.pop(reg, None)
            for c in carry:
                for reg in parse_regs(c):
                    label.pop(reg, None)
    return result


_DOT = ("v_dot4", "v_dot2", "v_sdot", "v_udot")
_UNPACK = ("v_and_b32", "v_lshrrev_b32", "v_lshlrev_b32", "v_bfe_", "v_perm_b32", "v_or_b32",
           "v_alignbit", "v_lshl_or", "v_and_or", "v_xad", "v_pk_", "v_cvt_f32_ubyte")


def hot_loop_report(insts: list, name: str) -> dict | None:
    loops = isa.find_loops(insts, name)
    quads = isa.private_resource_quads(insts)
    key, hot = isa.hot_loops(insts, loops, quads)
    if not hot:
        return None
    (lo, hi), _counts = hot[0]
    body = insts[lo:hi + 1]
    loads = classify_loads(insts, (lo, hi))
    by = {"X": [], "Y": [], "other": [], "mixed": []}
    for ld in loads:
        lab = ld["label"]
        by["X" if lab == "X" else "Y" if lab == "Y" else "mixed" if "+" in lab else "other"].append(ld)
    # Forward branches inside the loop make the static counts an UPPER bound per iteration
    # (both arms are counted).  The back-edge latch is the last branch; any other is internal.
    branches = [i for i in body if i.op.startswith(("s_cbranch", "s_branch"))]
    internal = max(0, len(branches) - 1)
    dots = sum(1 for i in body if i.op.startswith(_DOT))
    valu = sum(1 for i in body if i.op.startswith("v_"))
    unpack = sum(1 for i in body if i.op.startswith(_UNPACK))
    return {
        "loop_insts": len(body), "n_hot_loops": len(hot), "internal_branches": internal,
        "weight_loads": len(by["X"]), "act_loads": len(by["Y"]),
        "other_loads": len(by["other"]), "mixed_loads": len(by["mixed"]),
        "weight_load_ops": sorted(i["op"] for i in by["X"]),
        "act_load_ops": sorted(i["op"] for i in by["Y"]),
        "dot": dots, "valu": valu, "unpack_valu": unpack,
        "vmcnt0_waits": sum(1 for i in body if i.op == "s_waitcnt" and "vmcnt(0)" in i.args),
    }


# ---------------------------------------------------------------- library walk


@dataclasses.dataclass
class Cell:
    type_id: int
    ncols: int
    small_k: bool
    eight_warp: bool
    name: str


def wanted_cells(names, types) -> list[Cell]:
    cells = []
    for n in names:
        m = _KERNEL_RE.search(n)
        if not m or "mul_mat_vec_q_moe" in n:
            continue
        t, nc, fus, sk, fixed, gate, bias, eight = m.groups()
        if int(t) not in types or int(fixed) or fus != "0" or gate != "0" or bias != "0":
            continue
        cells.append(Cell(int(t), int(nc), sk == "1", eight == "1", n))
    return sorted(cells, key=lambda c: (c.type_id, c.small_k, c.eight_warp, c.ncols))


def audit_library(path: Path, types=tuple(TYPES)) -> dict:
    rows = []
    for idx, blob in enumerate(isa.load_code_objects(path)):
        with tempfile.NamedTemporaryFile(suffix=".o") as tmp:
            tmp.write(blob)
            tmp.flush()
            meta = isa.parse_notes_text(isa._run("llvm-readelf", "--notes", tmp.name))
            cells = wanted_cells(meta, set(types))
            if not cells:
                continue
            dis = isa.split_disassembly(isa._run("llvm-objdump", "-d", "--no-show-raw-insn",
                                                 tmp.name), {c.name for c in cells})
        for c in cells:
            rep = hot_loop_report(dis.get(c.name, []), c.name)
            if rep is None:
                continue
            r = rows_per_thread(c.type_id, c.ncols, c.small_k, c.eight_warp)
            unit = r * c.ncols
            rows.append({
                "code_object": f"co_{idx:03d}", "type": TYPES[c.type_id], "ncols": c.ncols,
                "small_k": c.small_k, "eight_warp": c.eight_warp, "rows_per_thread": r,
                "kernel": c.name, **rep,
                "w_loads_per_row": rep["weight_loads"] / r,
                "w_loads_per_unit": rep["weight_loads"] / unit,
                "act_loads_per_col": rep["act_loads"] / c.ncols,
                "dot_per_unit": rep["dot"] / unit,
                "valu_per_unit": rep["valu"] / unit,
                "unpack_valu_per_unit": rep["unpack_valu"] / unit,
            })
    flag_dot_unit(rows)
    return {"schema": SCHEMA, "source": str(path), "source_sha256": isa.file_sha256(path),
            "tool": "mmvq_hoist_audit", "rows": rows}


def flag_dot_unit(rows: list[dict]) -> None:
    """Self-consistency flags (a flagged row is reported but never feeds a verdict).

    * ``dot_unit_mismatch``: dots per (row, col) unit must be constant per (type, family) --
      otherwise ``rows_per_thread`` is wrong for that cell.
    * ``load_overcount``: activation loads per column must equal the ncols=1 value of the same
      (type, family).  A loop whose internal forward branches duplicate load arms (the Q4_K
      ncols=4 ``b4`` cell) counts both arms, so its static totals are only an upper bound.
    """
    groups: dict[tuple, set] = {}
    base_act: dict[tuple, float] = {}
    for r in rows:
        key = (r["type"], r["small_k"], r["eight_warp"])
        groups.setdefault(key, set()).add(round(r["dot_per_unit"], 3))
        if r["ncols"] == 1:
            base_act[key] = r["act_loads_per_col"]
    for r in rows:
        key = (r["type"], r["small_k"], r["eight_warp"])
        r["dot_unit_mismatch"] = len(groups[key]) > 1
        r["load_overcount"] = key in base_act and r["act_loads_per_col"] > base_act[key] + 1e-9


def verdict(rows: list[dict], type_name: str, small_k: bool = False) -> dict:
    """Hoisted iff weight loads per ROW are flat (within one load) over the clean ncols>=2 cells.

    ``repeated_ops`` names the load opcodes whose per-row count grows with ncols (for Q4_K the
    ushort d/dmin/scale loads), so a partial hoist is reported as such, not as a blanket yes/no.
    """
    sel = [r for r in rows if r["type"] == type_name and r["small_k"] == small_k and r["ncols"] >= 2
           and not r["eight_warp"] and not r["load_overcount"] and not r["dot_unit_mismatch"]]
    if not sel:
        return {"type": type_name, "small_k": small_k, "verdict": "no_data"}
    per_row = [r["w_loads_per_row"] for r in sel]
    ops: dict[str, list[float]] = {}
    for r in sel:
        for op in set(r["weight_load_ops"]):
            ops.setdefault(op, []).append(r["weight_load_ops"].count(op) / r["rows_per_thread"])
    repeated = sorted(op for op, v in ops.items() if max(v) - min(v) > 0.5)
    flat = max(per_row) - min(per_row) <= 1.0
    return {"type": type_name, "small_k": small_k,
            "ncols_cells": sorted(r["ncols"] for r in sel),
            "w_loads_per_row_min": min(per_row), "w_loads_per_row_max": max(per_row),
            "repeated_ops": repeated,
            "verdict": "hoisted" if flat else ("partly_repeated_per_column" if len(repeated) < len(ops)
                                               else "repeated_per_column")}


def format_table(doc: dict) -> str:
    head = (f"{'type':7} {'N':>2} {'sk':>2} {'8w':>2} {'R':>2} {'Wld':>4} {'Yld':>4} {'oth':>3} "
            f"{'W/row':>6} {'W/unit':>7} {'Y/col':>6} {'dot':>4} {'dot/u':>6} {'valu/u':>7} {'unp/u':>6} "
            f"{'vm0':>3} flags")
    lines = [head]
    for r in doc["rows"]:
        flags = " ".join(f for f in ("dot_unit_mismatch" if r.get("dot_unit_mismatch") else "",
                                     "load_overcount" if r.get("load_overcount") else "",
                                     f"branchy({r['internal_branches']})" if r["internal_branches"] else "") if f)
        lines.append(f"{r['type']:7} {r['ncols']:>2} {int(r['small_k']):>2} {int(r['eight_warp']):>2} "
                     f"{r['rows_per_thread']:>2} {r['weight_loads']:>4} {r['act_loads']:>4} "
                     f"{r['other_loads'] + r['mixed_loads']:>3} {r['w_loads_per_row']:>6.2f} "
                     f"{r['w_loads_per_unit']:>7.3f} {r['act_loads_per_col']:>6.2f} {r['dot']:>4} "
                     f"{r['dot_per_unit']:>6.2f} {r['valu_per_unit']:>7.2f} "
                     f"{r['unpack_valu_per_unit']:>6.2f} {r['vmcnt0_waits']:>3} {flags}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("binary", type=Path)
    ap.add_argument("--types", default="Q8_0,Q4_K,IQ4_XS")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--table", type=Path)
    a = ap.parse_args(argv)
    wanted = {k for k, v in TYPES.items() if v in a.types.split(",")}
    doc = audit_library(a.binary, tuple(wanted))
    doc["verdicts"] = [verdict(doc["rows"], TYPES[t], sk) for t in sorted(wanted) for sk in (False, True)]
    text = format_table(doc)
    sys.stdout.write(text)
    for v in doc["verdicts"]:
        sys.stdout.write(json.dumps(v) + "\n")
    if a.json:
        a.json.write_text(json.dumps(doc, indent=1, sort_keys=True))
    if a.table:
        a.table.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
