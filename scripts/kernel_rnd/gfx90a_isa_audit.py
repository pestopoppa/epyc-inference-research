#!/usr/bin/env python3
"""Static register/ISA audit for gfx90a HIP kernels (INF03-REGAUDIT-1).

Zero-GPU and read-only. It inspects EXISTING binaries only: a host library with a
``.hip_fatbin`` section (``libggml-hip.so``), a clang offload bundle, or a bare
gfx90a code object (``.hsaco`` / ``.o``). It never builds anything and never
touches a GPU.

Per kernel instance it emits one JSON row with:

* resource metadata from the AMDGPU ``.note`` (vgpr/agpr/sgpr counts, spill
  counts, private and LDS bytes, max flat workgroup size, wavefront size);
* kernel-descriptor fields (``accum_offset`` -> arch VGPRs, ``tg_split``,
  granulated VGPR allocation);
* derived gfx90a occupancy (waves per SIMD, workgroups per CU, and the limiter);
* ISA counts over the whole kernel and over the identified hot loop(s): private
  (scratch) loads/stores and spill reloads, ``v_accvgpr_*`` copies, ``s_waitcnt``
  drains, MFMA and ``v_dot`` counts, LDS traffic, and MFMA placement relative to
  ``s_barrier``.

``diff`` compares a candidate audit against a baseline audit and flags
regressions (new or larger spills, more hot-loop spill reloads or accvgpr
copies, AGPR use appearing, higher VGPR/AGPR counts, lower occupancy). It exits
non-zero on a FAIL finding, so it is the accept gate for any MMQ/FA
register-affecting change: no timing before it passes.

Usage::

    gfx90a_isa_audit.py audit LIB_OR_CO [--families mmq,mmf,fattn_mma,fattn_wmma]
                             [--match REGEX] [--json OUT.json] [--table OUT.txt]
    gfx90a_isa_audit.py table AUDIT.json [--families ...] [--sort spill]
    gfx90a_isa_audit.py diff BASELINE.json CANDIDATE.json [--fail-on fail|warn]

Heuristics, stated so nobody over-reads them:

* A *private access* is a ``scratch_*`` op, or a ``buffer_*`` op whose resource
  descriptor is the private-segment buffer (``s[0:3]`` at entry, or an SGPR quad
  copied from it). ``off`` addressing = fixed frame slot, ``offen`` = a
  dynamically indexed stack array.
* A *spill reload* is a private ``off``-addressed load in a kernel whose
  metadata reports ``vgpr_spill_count > 0``. With zero spills, private traffic
  is stack-object traffic and is reported as such, never as spills. Kernels
  with both spills and fixed-offset stack objects are reported ``mixed`` and the
  reload count is an upper bound.
* A *loop* is a backward branch whose target lies inside the same symbol. The
  hot loops are the minimal loops carrying the kernel's maximum loop MFMA count
  (``v_dot`` if the kernel has no MFMA). ``hot_*`` values are the maximum over
  the hot loops (worst case per iteration); ``n_hot_loops`` says how many.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

SCHEMA = "epyc.gfx90a.isa_audit.v1"
TOOL_ID = "epyc.gfx90a_isa_audit/v1"
LLVM_BIN = Path(os.environ.get("ROCM_LLVM_BIN", "/opt/rocm/llvm/bin"))
DEFAULT_FAMILIES = ("mmq", "mmf", "fattn_mma", "fattn_wmma")
ALL_FAMILIES = ("mmq", "mmq_fixup", "mmvq", "mmf", "fattn_mma", "fattn_wmma",
                "fattn_vec", "fattn_tile", "fattn_other", "other")

# gfx90a (CDNA2, MI210) per-SIMD / per-CU resources.
GFX90A = {
    "max_waves_per_simd": 8,
    "simds_per_cu": 4,
    "vgprs_per_simd_lane": 512,   # unified arch + acc file
    "vgpr_granule": 8,
    "sgprs_per_simd": 800,
    "sgpr_granule": 16,
    "lds_per_cu": 65536,
    "wave_size": 64,
}

GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS",
    17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
    23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4", 41: "Q1_0", 42: "Q2_0",
}

STUB_MAX_INSTS = 16
BUNDLE_MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"
COMPRESSED_MAGIC = b"CCOB"
ELF_MAGIC = b"\x7fELF"
EM_AMDGPU = 224


class AuditError(RuntimeError):
    pass


# --------------------------------------------------------------------------- ELF


@dataclasses.dataclass
class Section:
    name: str
    type: int
    addr: int
    offset: int
    size: int
    link: int
    entsize: int


def elf_sections(data: bytes) -> list[Section]:
    if data[:4] != ELF_MAGIC or data[4] != 2 or data[5] != 1:
        raise AuditError("not a little-endian ELF64 object")
    shoff = struct.unpack_from("<Q", data, 0x28)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    raw = []
    for i in range(shnum):
        name, stype, _flags, addr, off, size, link, _info, _align, entsize = struct.unpack_from(
            "<IIQQQQIIQQ", data, shoff + i * shentsize)
        raw.append((name, stype, addr, off, size, link, entsize))
    stroff = raw[shstrndx][3]
    out = []
    for name, stype, addr, off, size, link, entsize in raw:
        end = data.index(b"\0", stroff + name)
        out.append(Section(data[stroff + name:end].decode(), stype, addr, off, size, link, entsize))
    return out


def elf_machine(data: bytes) -> int:
    return struct.unpack_from("<H", data, 0x12)[0]


def section_bytes(data: bytes, sections: list[Section], name: str) -> bytes | None:
    for s in sections:
        if s.name == name:
            return data[s.offset:s.offset + s.size]
    return None


def elf_symbols(data: bytes, sections: list[Section]) -> list[tuple[str, int, int, int, int]]:
    """(name, value, size, info, shndx) for .symtab (fallback .dynsym)."""
    tab = next((s for s in sections if s.type == 2), None) or next(
        (s for s in sections if s.type == 11), None)
    if tab is None:
        return []
    strtab = sections[tab.link]
    out = []
    for i in range(tab.size // 24):
        name, info, _other, shndx, value, size = struct.unpack_from(
            "<IBBHQQ", data, tab.offset + i * 24)
        end = data.index(b"\0", strtab.offset + name)
        out.append((data[strtab.offset + name:end].decode(errors="replace"), value, size, info, shndx))
    return out


def parse_kernel_descriptor(kd: bytes) -> dict:
    """Decode the 64-byte amdhsa kernel descriptor (gfx90a layout)."""
    if len(kd) < 64:
        raise AuditError("short kernel descriptor")
    group, private, kernarg = struct.unpack_from("<III", kd, 0)
    rsrc3, rsrc1, rsrc2 = struct.unpack_from("<III", kd, 44)
    props = struct.unpack_from("<H", kd, 56)[0]
    accum_field = rsrc3 & 0x3F
    return {
        "kd_group_segment_bytes": group,
        "kd_private_segment_bytes": private,
        "kd_kernarg_bytes": kernarg,
        "accum_offset": (accum_field + 1) * 4,
        "tg_split": bool(rsrc3 & (1 << 16)),
        "vgpr_alloc_granulated": ((rsrc1 & 0x3F) + 1) * GFX90A["vgpr_granule"],
        "sgpr_alloc_granulated_field": (rsrc1 >> 6) & 0xF,
        "private_segment_enabled": bool(rsrc2 & 1),
        "kernel_code_properties": props,
    }


# ------------------------------------------------------------- code objects


def split_bundles(blob: bytes, target: str = "gfx90a") -> list[bytes]:
    """Every ``target`` code object inside a (possibly concatenated) offload bundle."""
    if COMPRESSED_MAGIC in blob[:4096] and BUNDLE_MAGIC not in blob:
        raise AuditError("compressed offload bundle (CCOB): unbundle with "
                         "clang-offload-bundler --unbundle first")
    out, pos = [], 0
    while True:
        i = blob.find(BUNDLE_MAGIC, pos)
        if i < 0:
            break
        p = i + len(BUNDLE_MAGIC)
        (num,) = struct.unpack_from("<Q", blob, p)
        p += 8
        for _ in range(num):
            off, size, tlen = struct.unpack_from("<QQQ", blob, p)
            p += 24
            triple = blob[p:p + tlen].decode(errors="replace")
            p += tlen
            if target in triple and size:
                out.append(blob[i + off:i + off + size])
        pos = i + len(BUNDLE_MAGIC)
    return out


def load_code_objects(path: Path) -> list[bytes]:
    data = path.read_bytes()
    if data[:len(BUNDLE_MAGIC)] == BUNDLE_MAGIC:
        return split_bundles(data)
    if data[:4] != ELF_MAGIC:
        raise AuditError(f"{path}: neither ELF nor offload bundle")
    secs = elf_sections(data)
    if elf_machine(data) == EM_AMDGPU:
        return [data]
    fat = section_bytes(data, secs, ".hip_fatbin")
    if fat is None:
        raise AuditError(f"{path}: host ELF without a .hip_fatbin section")
    return split_bundles(fat)


# ------------------------------------------------------------------ metadata

_META_KEYS = (".name", ".symbol", ".vgpr_count", ".agpr_count", ".sgpr_count",
              ".vgpr_spill_count", ".sgpr_spill_count", ".private_segment_fixed_size",
              ".group_segment_fixed_size", ".max_flat_workgroup_size", ".wavefront_size",
              ".reqd_workgroup_size", ".uses_dynamic_stack", ".kernarg_segment_size")


def parse_notes_text(text: str) -> dict[str, dict]:
    """Kernel metadata from ``llvm-readelf --notes`` output, keyed by symbol name."""
    i = text.find("amdhsa.kernels:")
    if i < 0:
        return {}
    j = text.find("\namdhsa.", i + 1)
    body = text[i:j if j > 0 else None]
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    doc = yaml.load(body, Loader=loader) or {}
    out = {}
    for k in doc.get("amdhsa.kernels") or []:
        m = {key: k.get(key) for key in _META_KEYS if key in k}
        out[k[".name"]] = m
    return out


def toolchain_from_comment(comment: bytes | None) -> dict:
    text = (comment or b"").replace(b"\0", b"\n").decode(errors="replace")
    clang = next((ln.strip() for ln in text.splitlines() if "clang version" in ln), "")
    m = re.search(r"clang version (\d+)", clang)
    major = int(m.group(1)) if m else None
    rocm = re.search(r"roc-(\d+\.\d+\.\d+)", clang)
    # ROCm 6.2 (LLVM 18) applies the MayNeedAGPRs rule: kernels whose
    # max_flat_workgroup_size is <=256 may be given AGPRs and pay v_accvgpr copies.
    # LLVM #159493 (LLVM 22 development cycle) changes that default.
    if major is None:
        regime = "unknown"
    elif major < 22:
        regime = "mayneedagprs_rule_pre_159493"
    else:
        regime = "llvm_159493_default"
    return {"toolchain_id": clang or "unknown", "llvm_major": major,
            "rocm_version": rocm.group(1) if rocm else None, "mfma_form_regime": regime}


# --------------------------------------------------------------- disassembly

_SYM_RE = re.compile(r"^([0-9a-f]+) <(.+)>:$")
_INS_RE = re.compile(r"^\s+([a-z_][\w.]*)\s*(.*?)\s*//\s*([0-9A-Fa-f]+):(.*)$")
_TGT_RE = re.compile(r"<(.+)\+0x([0-9a-f]+)>\s*$")
_SREG_PAIR = re.compile(r"s\[(\d+):(\d+)\]")


@dataclasses.dataclass
class Inst:
    addr: int
    op: str
    args: str
    target: tuple[str, int] | None = None


def split_disassembly(text: str, wanted: set[str] | None = None) -> dict[str, list[Inst]]:
    out: dict[str, list[Inst]] = {}
    cur = None
    for line in text.splitlines():
        m = _SYM_RE.match(line)
        if m:
            name = m.group(2)
            cur = out.setdefault(name, []) if wanted is None or name in wanted else None
            continue
        if cur is None:
            continue
        m = _INS_RE.match(line)
        if not m:
            continue
        op, args, addr, tail = m.groups()
        t = _TGT_RE.search(tail)
        cur.append(Inst(int(addr, 16), op, args, (t.group(1), int(t.group(2), 16)) if t else None))
    return out


def private_resource_quads(insts: list[Inst]) -> set[int]:
    """Base SGPR indices of quads that hold the private-segment buffer resource.

    At entry the private segment buffer is s[0:3]; the prologue commonly copies it
    to another quad with two ``s_mov_b64`` (lo pair, hi pair). Follow copies to a
    fixpoint.
    """
    movs = set()
    for ins in insts:
        if ins.op == "s_mov_b64":
            regs = _SREG_PAIR.findall(ins.args)
            if len(regs) == 2:
                movs.add((int(regs[0][0]), int(regs[1][0])))
    quads = {0}
    changed = True
    while changed:
        changed = False
        for dst, src in movs:
            if src in quads and dst not in quads and (dst + 2, src + 2) in movs:
                quads.add(dst)
                changed = True
    return quads


def classify_private(ins: Inst, quads: set[int]) -> tuple[str, str] | None:
    """('ld'|'st', 'off'|'offen'|'scratch') for a private-segment access, else None."""
    op = ins.op
    if op.startswith("scratch_load"):
        return "ld", "scratch"
    if op.startswith("scratch_store"):
        return "st", "scratch"
    if not (op.startswith("buffer_load") or op.startswith("buffer_store")):
        return None
    m = re.search(r"s\[(\d+):(\d+)\]", ins.args)
    if not m or int(m.group(1)) not in quads or int(m.group(2)) != int(m.group(1)) + 3:
        return None
    kind = "ld" if op.startswith("buffer_load") else "st"
    mode = "offen" if re.search(r"\boffen\b", ins.args) else "off"
    return kind, mode


def resolve_callees(insts: list[Inst], func_addrs: dict[int, str] | None) -> list[str]:
    """Names of direct callees: ``s_getpc_b64 s[a:b]; s_add_u32 sa, sa, imm; ...;
    s_swappc_b64 s[30:31], s[a:b]`` resolves to (getpc address + 4 + imm)."""
    out = []
    getpc: dict[int, tuple[int, int | None]] = {}
    for ins in insts:
        if ins.op == "s_getpc_b64":
            m = _SREG_PAIR.search(ins.args)
            if m:
                getpc[int(m.group(1))] = (ins.addr + 4, None)
        elif ins.op == "s_add_u32":
            m = re.match(r"s(\d+),\s*s(\d+),\s*(0x[0-9a-fA-F]+|-?\d+)", ins.args)
            if m and int(m.group(1)) == int(m.group(2)) and int(m.group(1)) in getpc:
                imm = int(m.group(3), 0)
                if imm >= 1 << 31:
                    imm -= 1 << 32
                pc, _ = getpc[int(m.group(1))]
                getpc[int(m.group(1))] = (pc, pc + imm)
        elif ins.op == "s_swappc_b64":
            regs = _SREG_PAIR.findall(ins.args)
            tgt = getpc.get(int(regs[1][0]), (None, None))[1] if len(regs) == 2 else None
            out.append((func_addrs or {}).get(tgt, "?") if tgt is not None else "?")
    return out


COUNTERS = ("insts", "mfma", "dot", "valu", "accvgpr_read", "accvgpr_write", "accvgpr_mov",
            "priv_ld", "priv_st", "priv_ld_off", "priv_st_off", "priv_ld_offen", "priv_st_offen",
            "global_ld", "global_st", "ds_read", "ds_write", "s_barrier",
            "waitcnt", "waitcnt_vmcnt0", "waitcnt_lgkmcnt0", "waitcnt_full", "calls")


def count_range(insts: list[Inst], quads: set[int]) -> dict[str, int]:
    c = dict.fromkeys(COUNTERS, 0)
    for ins in insts:
        op = ins.op
        c["insts"] += 1
        if op.startswith("v_mfma"):
            c["mfma"] += 1
        elif op.startswith("v_dot"):
            c["dot"] += 1
        if op.startswith("v_"):
            c["valu"] += 1
        if op == "v_accvgpr_read_b32":
            c["accvgpr_read"] += 1
        elif op == "v_accvgpr_write_b32":
            c["accvgpr_write"] += 1
        elif op == "v_accvgpr_mov_b32":
            c["accvgpr_mov"] += 1
        priv = classify_private(ins, quads)
        if priv:
            kind, mode = priv
            c["priv_" + kind] += 1
            c[f"priv_{kind}_{'offen' if mode == 'offen' else 'off'}"] += 1
        elif op.startswith(("global_load", "buffer_load", "flat_load")):
            c["global_ld"] += 1
        elif op.startswith(("global_store", "buffer_store", "flat_store")):
            c["global_st"] += 1
        if op.startswith("ds_read") or op.startswith("ds_load"):
            c["ds_read"] += 1
        elif op.startswith("ds_write") or op.startswith("ds_store"):
            c["ds_write"] += 1
        if op == "s_barrier":
            c["s_barrier"] += 1
        if op == "s_swappc_b64":
            c["calls"] += 1
        if op == "s_waitcnt":
            c["waitcnt"] += 1
            vm = "vmcnt(0)" in ins.args
            lg = "lgkmcnt(0)" in ins.args
            c["waitcnt_vmcnt0"] += vm
            c["waitcnt_lgkmcnt0"] += lg
            c["waitcnt_full"] += (vm and lg) or ins.args.strip() in ("0", "")
    return c


def find_loops(insts: list[Inst], symbol: str) -> list[tuple[int, int]]:
    """(header_idx, latch_idx) per loop header, merging back edges to one header."""
    if not insts:
        return []
    base = insts[0].addr
    index = {ins.addr: i for i, ins in enumerate(insts)}
    headers: dict[int, int] = {}
    for i, ins in enumerate(insts):
        if not (ins.op.startswith("s_cbranch") or ins.op == "s_branch") or ins.target is None:
            continue
        sym, off = ins.target
        if sym != symbol:
            continue
        tgt = base + off
        if tgt >= ins.addr:
            continue
        j = index.get(tgt)
        if j is None:
            j = next((k for k, x in enumerate(insts) if x.addr >= tgt), None)
        if j is not None:
            headers[j] = max(headers.get(j, i), i)
    return sorted(headers.items())


def mfma_barrier_segments(insts: list[Inst]) -> list[int]:
    """MFMA count per s_barrier-delimited phase of a loop body, treated cyclically."""
    segs, cur = [], 0
    for ins in insts:
        if ins.op == "s_barrier":
            segs.append(cur)
            cur = 0
        elif ins.op.startswith("v_mfma"):
            cur += 1
    if not segs:
        return [cur]
    segs[0] += cur  # the tail wraps into the first phase on the next iteration
    return segs


def hot_loops(insts: list[Inst], loops: list[tuple[int, int]], quads: set[int]):
    counted = [((lo, hi), count_range(insts[lo:hi + 1], quads)) for lo, hi in loops]
    if not counted:
        return "none", []
    key = "mfma" if max(c["mfma"] for _, c in counted) > 0 else (
        "dot" if max(c["dot"] for _, c in counted) > 0 else None)
    if key is None:
        return "none", []
    best = max(c[key] for _, c in counted)
    cands = [(r, c) for r, c in counted if c[key] == best]
    minimal = [(r, c) for r, c in cands
               if not any(o != r and r[0] <= o[0] and o[1] <= r[1] for o, _ in cands)]
    return key, minimal


# ------------------------------------------------------------------- family

_MMQ_RE = re.compile(r"^_ZL9mul_mat_qIL9ggml_type(\d+)ELi(\d+)ELb([01])E")


def classify_family(name: str) -> tuple[str, dict]:
    m = _MMQ_RE.match(name)
    if m:
        t, j, fb = (int(x) for x in m.groups())
        return "mmq", {"ggml_type": t, "ggml_type_name": GGML_TYPES.get(t, f"type{t}"),
                       "mmq_x": j, "need_check": bool(fb)}
    if "mul_mat_q_stream_k_fixup" in name:
        return "mmq_fixup", {}
    if "mul_mat_vec_q" in name:
        return "mmvq", {}
    m = re.match(r"_ZL\d+mul_mat_f(_ids)?I(.+?)((?:Li\d+E)+)", name)
    if m:
        ints = [int(x) for x in re.findall(r"Li(\d+)E", m.group(3))]
        return "mmf", {"ids": bool(m.group(1)), "T": m.group(2), "ints": ints}
    if "flash_attn_ext_vec" in name:
        return "fattn_vec", {}
    if "flash_attn_tile" in name:
        return "fattn_tile", {}
    m = re.search(r"flash_attn_ext_f16ILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)E(.)(.*?)EEv", name)
    if m:
        a, b, c, d, nxt, rest = m.groups()
        bools = [int(x) for x in re.findall(r"Lb([01])", nxt + rest)]
        # MMA FA: <DKQ, DV, ncols1, ncols2, bool logit_softcap, bool mla>  -> next is 'L' (bool literal)
        # rocWMMA FA: <D, ncols, nwarps, VKQ_stride, KQ_acc_t, bool> -> next is a type (f / 6__half)
        if nxt == "L":
            return "fattn_mma", {"dkq": int(a), "dv": int(b), "ncols1": int(c), "ncols2": int(d),
                                 "logit_softcap": bool(bools[0]) if bools else None,
                                 "mla": bool(bools[1]) if len(bools) > 1 else None}
        return "fattn_wmma", {"d": int(a), "ncols": int(b), "nwarps": int(c), "vkq_stride": int(d),
                              "kq_acc": "half" if (nxt + rest).startswith("6__half") else "float",
                              "logit_softcap": bool(bools[-1]) if bools else None}
    if "flash_attn" in name:
        return "fattn_other", {}
    return "other", {}


# ---------------------------------------------------------------- occupancy


def occupancy(vgpr_total: int, sgpr: int, lds: int, wg_size: int) -> dict:
    g = GFX90A
    alloc_v = max(g["vgpr_granule"], math.ceil(max(vgpr_total, 1) / g["vgpr_granule"]) * g["vgpr_granule"])
    waves_v = min(g["max_waves_per_simd"], g["vgprs_per_simd_lane"] // alloc_v)
    alloc_s = max(g["sgpr_granule"], math.ceil(max(sgpr, 1) / g["sgpr_granule"]) * g["sgpr_granule"])
    waves_s = min(g["max_waves_per_simd"], g["sgprs_per_simd"] // alloc_s)
    waves_per_wg = max(1, math.ceil(wg_size / g["wave_size"]))
    simds = g["simds_per_cu"]

    def wgs(per_simd_waves: int) -> int:
        # A workgroup of >= 4 waves spreads over all SIMDs and needs ceil(w/4) slots on
        # each; smaller workgroups pack several per CU against the CU-wide slot total.
        if waves_per_wg >= simds:
            return per_simd_waves // math.ceil(waves_per_wg / simds)
        return per_simd_waves * simds // waves_per_wg

    limits = {
        "vgpr": wgs(waves_v),
        "sgpr": wgs(waves_s),
        "waves": wgs(g["max_waves_per_simd"]),
        "lds": (g["lds_per_cu"] // lds) if lds else 10 ** 6,
    }
    wg_per_cu = min(limits.values())
    limiter = min(limits, key=lambda k: (limits[k], ["vgpr", "lds", "sgpr", "waves"].index(k)))
    waves = min(g["max_waves_per_simd"], wg_per_cu * waves_per_wg / g["simds_per_cu"])
    return {"vgpr_alloc": alloc_v, "waves_per_simd_vgpr_limit": waves_v,
            "waves_per_simd_sgpr_limit": waves_s, "waves_per_wg": waves_per_wg,
            "wg_per_cu": wg_per_cu, "occupancy_waves_per_simd": float(round(waves, 3)),
            "occupancy_limiter": limiter}


# ------------------------------------------------------------------- audit


def _run(tool: str, *args: str) -> str:
    exe = LLVM_BIN / tool
    res = subprocess.run([str(exe), *args], capture_output=True, text=True)
    if res.returncode != 0:
        raise AuditError(f"{tool} failed: {res.stderr.strip()[:400]}")
    return res.stdout


def analyze_kernel(name: str, insts: list[Inst], meta: dict, kd: dict | None,
                   func_addrs: dict[int, str] | None = None) -> dict:
    family, params = classify_family(name)
    vgpr = int(meta.get(".vgpr_count") or 0)
    agpr = int(meta.get(".agpr_count") or 0)
    sgpr = int(meta.get(".sgpr_count") or 0)
    vspill = int(meta.get(".vgpr_spill_count") or 0)
    sspill = int(meta.get(".sgpr_spill_count") or 0)
    priv = int(meta.get(".private_segment_fixed_size") or 0)
    lds = int(meta.get(".group_segment_fixed_size") or 0)
    reqd = meta.get(".reqd_workgroup_size")
    wg_max = int(meta.get(".max_flat_workgroup_size") or 0)
    wg = int(math.prod(reqd)) if reqd else wg_max
    row = {
        "kernel": name, "family": family, "params": params,
        "wg_max": wg_max, "reqd_workgroup_size": reqd, "wavefront_size": meta.get(".wavefront_size"),
        "vgpr_total": vgpr, "agpr": agpr, "sgpr": sgpr,
        "vgpr_spill": vspill, "sgpr_spill": sspill, "private_bytes": priv, "lds_bytes": lds,
        "uses_dynamic_stack": bool(meta.get(".uses_dynamic_stack")),
    }
    if kd:
        row["accum_offset"] = kd["accum_offset"]
        row["arch_vgpr"] = kd["accum_offset"] if agpr else vgpr
        row["tg_split"] = kd["tg_split"]
        row["vgpr_alloc_granulated"] = kd["vgpr_alloc_granulated"]
    else:
        row["accum_offset"] = None
        row["arch_vgpr"] = vgpr - agpr if agpr else vgpr
    row.update(occupancy(vgpr, sgpr, lds, wg))
    quads = private_resource_quads(insts)
    kern = count_range(insts, quads)
    row["kern"] = kern
    # Template instances the build compiles out (NO_DEVICE_CODE: unsupported MMQ tile
    # sizes, FA / mul_mat_f configs) keep a prologue plus one call to no_device_code();
    # they are flagged and left out of tables, summaries and the gate. ISA inside the
    # callees of a real kernel is not counted (``callees`` lists them).
    callees = resolve_callees(insts, func_addrs)
    row["callees"] = callees
    loopless = not find_loops(insts, name) and kern["mfma"] == 0
    row["stub"] = loopless and (
        (bool(callees) and all("no_device_code" in c for c in callees))
        or kern["insts"] <= STUB_MAX_INSTS)
    if vspill == 0:
        row["scratch_kind"] = "stack" if (kern["priv_ld"] or kern["priv_st"] or priv) else "none"
    else:
        stack_bytes = priv - 4 * vspill
        row["scratch_kind"] = "mixed" if (kern["priv_ld_offen"] or kern["priv_st_offen"]
                                          or stack_bytes > 4 * vspill) else "spill"
    loops = find_loops(insts, name)
    key, hot = hot_loops(insts, loops, quads)
    row["n_loops"] = len(loops)
    row["hot_loop_key"] = key
    row["n_hot_loops"] = len(hot)
    spill = vspill > 0
    if hot:
        agg = {c: max(h[c] for _, h in hot) for c in COUNTERS}
        row["hot"] = agg
        segs = [mfma_barrier_segments(insts[lo:hi + 1]) for (lo, hi), _ in hot]
        row["hot_mfma_barrier_segments"] = segs
        row["hot_mfma_phases"] = max(sum(1 for s in sg if s) for sg in segs)
        row["hot_spill_reloads"] = agg["priv_ld_off"] if spill else 0
        row["hot_spill_stores"] = agg["priv_st_off"] if spill else 0
        row["hot_accvgpr_copies"] = agg["accvgpr_read"] + agg["accvgpr_write"] + agg["accvgpr_mov"]
        row["hot_loops_detail"] = [
            {"range": [f"0x{insts[lo].addr:x}", f"0x{insts[hi].addr:x}"], "insts": h["insts"],
             "mfma": h["mfma"], "dot": h["dot"], "s_barrier": h["s_barrier"],
             "spill_reloads": h["priv_ld_off"] if spill else 0,
             "private_ld": h["priv_ld"], "private_st": h["priv_st"],
             "accvgpr_copies": h["accvgpr_read"] + h["accvgpr_write"] + h["accvgpr_mov"],
             "waitcnt_vmcnt0": h["waitcnt_vmcnt0"], "waitcnt_lgkmcnt0": h["waitcnt_lgkmcnt0"],
             "mfma_barrier_segments": mfma_barrier_segments(insts[lo:hi + 1])}
            for (lo, hi), h in hot]
    else:
        row["hot_loops_detail"] = []
        row["hot"] = None
        row["hot_mfma_barrier_segments"] = []
        row["hot_mfma_phases"] = 0
        row["hot_spill_reloads"] = 0
        row["hot_spill_stores"] = 0
        row["hot_accvgpr_copies"] = 0
    row["kern_spill_reloads"] = kern["priv_ld_off"] if spill else 0
    row["kern_accvgpr_copies"] = kern["accvgpr_read"] + kern["accvgpr_write"] + kern["accvgpr_mov"]
    return row


def audit_code_object(args: tuple[int, bytes, tuple[str, ...], str | None]) -> tuple[dict, list[dict]]:
    idx, blob, families, match = args
    rx = re.compile(match) if match else None
    secs = elf_sections(blob)
    chain = toolchain_from_comment(section_bytes(blob, secs, ".comment"))
    with tempfile.NamedTemporaryFile(prefix=f"co_{idx:03d}_", suffix=".gfx90a.o") as tmp:
        tmp.write(blob)
        tmp.flush()
        meta = parse_notes_text(_run("llvm-readelf", "--notes", tmp.name))
        wanted = {n for n in meta
                  if (families == ("all",) or classify_family(n)[0] in families)
                  and (rx is None or rx.search(n))}
        if not wanted:
            return {"index": idx, **chain}, []
        dis = split_disassembly(_run("llvm-objdump", "-d", "--no-show-raw-insn", tmp.name), wanted)
    syms = elf_symbols(blob, secs)
    kds = {}
    for name, value, size, _info, shndx in syms:
        if name.endswith(".kd") and name[:-3] in wanted and 0 < shndx < len(secs):
            s = secs[shndx]
            off = s.offset + (value - s.addr)
            kds[name[:-3]] = parse_kernel_descriptor(blob[off:off + 64])
    func_addrs = {value: name for name, value, _size, info, _shndx in syms if info & 0xF == 2}
    rows = []
    for name in sorted(wanted):
        row = analyze_kernel(name, dis.get(name, []), meta[name], kds.get(name), func_addrs)
        row["code_object"] = f"co_{idx:03d}"
        rows.append(row)
    info = {"index": idx, "sha256": hashlib.sha256(blob).hexdigest(), "size": len(blob),
            "n_kernels": len(meta), "n_audited": len(rows), **chain}
    return info, rows


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def run_audit(path: Path, families: tuple[str, ...], match: str | None, jobs: int,
              label: str | None = None, category: str | None = None,
              source_commit: str | None = None, flags_pragmas: str | None = None) -> dict:
    objs = load_code_objects(path)
    if not objs:
        raise AuditError(f"{path}: no gfx90a code objects found")
    work = [(i, b, families, match) for i, b in enumerate(objs)]
    infos, rows = [], []
    if jobs > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as ex:
            results = list(ex.map(audit_code_object, work))
    else:
        results = [audit_code_object(w) for w in work]
    for info, rs in results:
        infos.append(info)
        rows.extend(rs)
    binary_sha = file_sha256(path)
    chains = sorted({i.get("toolchain_id", "unknown") for i in infos if i.get("n_audited")})
    regimes = sorted({i.get("mfma_form_regime", "unknown") for i in infos if i.get("n_audited")})
    for r in rows:
        r["schema"] = SCHEMA
        r["binary_sha256"] = binary_sha
        r["gfx_target"] = "gfx90a"
        r["row_id"] = f"{binary_sha[:16]}:{r['code_object']}:{r['kernel']}"
        r["self_sha256"] = hashlib.sha256(canonical({k: v for k, v in r.items()
                                                     if k != "self_sha256"})).hexdigest()
    return {
        "schema": SCHEMA, "tool_id": TOOL_ID, "tool_sha256": file_sha256(Path(__file__)),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "label": label,
        # Claim identity for the belief-kernel write side (SC84/SC84a). Absent unless the
        # caller states them; a document without a category projects no claims.
        "category": category, "source_commit": source_commit, "flags_pragmas": flags_pragmas,
        "source": {"path": str(path), "resolved": str(path.resolve()), "sha256": binary_sha,
                   "size": path.stat().st_size, "n_code_objects": len(objs)},
        "toolchain": chains, "mfma_form_regime": regimes, "families": list(families),
        "match": match, "hardware_model": GFX90A,
        "occupancy_basis": "registers + static LDS only; dynamic (extern __shared__) LDS is "
                           "set at launch and invisible here, so occupancy is an upper bound",
        "code_objects": sorted(infos, key=lambda i: i["index"]),
        "n_rows": len(rows),
        "rows": sorted(rows, key=lambda r: (r["family"], r["kernel"], r["code_object"])),
    }


# -------------------------------------------------------------------- table

TABLE_COLS = [
    ("family", "family", 10), ("kernel", "kernel", 44), ("wg", "wg_max", 4),
    ("vgpr", "vgpr_total", 4), ("arch", "arch_vgpr", 4), ("agpr", "agpr", 4),
    ("sgpr", "sgpr", 4), ("vspl", "vgpr_spill", 4), ("priv", "private_bytes", 5),
    ("lds", "lds_bytes", 6), ("occ", "occupancy_waves_per_simd", 5),
    ("lim", "occupancy_limiter", 4), ("hot", "n_hot_loops", 3),
    ("h.rld", "hot_spill_reloads", 5), ("h.acc", "hot_accvgpr_copies", 5),
    ("h.mfma", ("hot", "mfma"), 6), ("h.bar", ("hot", "s_barrier"), 5),
    ("h.vm0", ("hot", "waitcnt_vmcnt0"), 5), ("h.lg0", ("hot", "waitcnt_lgkmcnt0"), 5),
    ("phases", "hot_mfma_phases", 6), ("k.acc", "kern_accvgpr_copies", 5),
]


def short_name(row: dict) -> str:
    p = row.get("params") or {}
    fam = row["family"]
    if fam == "mmq":
        return f"mmq {p['ggml_type_name']} J={p['mmq_x']} chk={int(p['need_check'])}"
    if fam == "fattn_mma":
        return (f"fa-mma DKQ={p['dkq']} DV={p['dv']} c1={p['ncols1']} c2={p['ncols2']}"
                f"{' sc' if p.get('logit_softcap') else ''}{' mla' if p.get('mla') else ''}")
    if fam == "fattn_wmma":
        return (f"fa-wmma D={p['d']} nc={p['ncols']} nw={p['nwarps']} {p.get('kq_acc', '')}"
                f"{' sc' if p.get('logit_softcap') else ''}")
    if fam == "mmf":
        return f"mmf{'-ids' if p.get('ids') else ''} {p.get('T')} {p.get('ints')}"
    return row["kernel"][:44]


def _get(row, key):
    if isinstance(key, tuple):
        v = row.get(key[0])
        return v.get(key[1]) if isinstance(v, dict) else None
    return row.get(key)


def format_table(rows: list[dict]) -> str:
    head = " ".join(h.rjust(w) if i > 1 else h.ljust(w) for i, (h, _, w) in enumerate(TABLE_COLS))
    out = [head, "-" * len(head)]
    for r in rows:
        cells = []
        for i, (_, key, w) in enumerate(TABLE_COLS):
            v = short_name(r) if key == "kernel" else _get(r, key)
            s = "-" if v is None else str(v)
            cells.append(s[:w].ljust(w) if i <= 1 else s.rjust(w))
        out.append(" ".join(cells))
    return "\n".join(out)


def select_rows(doc: dict, families: tuple[str, ...] | None, match: str | None,
                include_stubs: bool = False) -> list[dict]:
    rows = [r for r in doc["rows"] if include_stubs or not r.get("stub")]
    if families and families != ("all",):
        rows = [r for r in rows if r["family"] in families]
    if match:
        rx = re.compile(match)
        rows = [r for r in rows if rx.search(r["kernel"])]
    return rows


def dedupe(rows: list[dict]) -> list[dict]:
    """One row per kernel symbol (TU duplicates of a template instance are identical code)."""
    seen = {}
    for r in rows:
        seen.setdefault(r["kernel"], r)
    return list(seen.values())


SORTS = {
    "spill": lambda r: (-r["hot_spill_reloads"], -r["vgpr_spill"], r["kernel"]),
    "acc": lambda r: (-r["hot_accvgpr_copies"], -r["kern_accvgpr_copies"], r["kernel"]),
    "vgpr": lambda r: (-r["vgpr_total"], r["kernel"]),
    "name": lambda r: (r["family"], r["kernel"]),
}


def summarize(doc: dict) -> str:
    rows = dedupe(select_rows(doc, None, None))
    n_stub = len(dedupe([r for r in doc["rows"] if r.get("stub")]))
    fams: dict[str, list[dict]] = {}
    for r in rows:
        fams.setdefault(r["family"], []).append(r)
    lines = [f"source {doc['source']['path']} sha256 {doc['source']['sha256']}",
             f"toolchain {'; '.join(doc['toolchain'])}", f"mfma_form_regime {doc['mfma_form_regime']}",
             f"unique kernels {len(rows)} (+{n_stub} trap-only stubs excluded)",
             "", "family      n  n_agpr  n_vspill  n_hot_reload  n_hot_acc  max_vgpr  max_hot_reload"]
    for fam, rs in sorted(fams.items()):
        lines.append(f"{fam:<10} {len(rs):>3} {sum(r['agpr'] > 0 for r in rs):>7} "
                     f"{sum(r['vgpr_spill'] > 0 for r in rs):>9} "
                     f"{sum(r['hot_spill_reloads'] > 0 for r in rs):>13} "
                     f"{sum(r['hot_accvgpr_copies'] > 0 for r in rs):>10} "
                     f"{max(r['vgpr_total'] for r in rs):>9} "
                     f"{max(r['hot_spill_reloads'] for r in rs):>15}")
    return "\n".join(lines)


# --------------------------------------------------------------------- diff

FAIL, WARN, INFO = "FAIL", "WARN", "INFO"


def _worst(rows: list[dict]) -> dict[str, dict]:
    by: dict[str, dict] = {}
    for r in rows:
        cur = by.get(r["kernel"])
        if cur is None or (r["vgpr_spill"], r["hot_spill_reloads"], r["vgpr_total"]) > (
                cur["vgpr_spill"], cur["hot_spill_reloads"], cur["vgpr_total"]):
            by[r["kernel"]] = r
    return by


def compare_rows(b: dict, c: dict) -> list[tuple[str, str, str]]:
    """(severity, check, detail) findings for one kernel, candidate c vs baseline b."""
    out = []

    def chk(sev, name, bv, cv, worse):
        if worse(bv, cv):
            out.append((sev, name, f"{bv} -> {cv}"))

    gt = lambda bv, cv: (cv or 0) > (bv or 0)  # noqa: E731
    chk(FAIL, "vgpr_spill", b["vgpr_spill"], c["vgpr_spill"], gt)
    chk(WARN, "sgpr_spill", b["sgpr_spill"], c["sgpr_spill"], gt)  # lands in VGPR lanes first
    chk(FAIL, "hot_spill_reloads", b["hot_spill_reloads"], c["hot_spill_reloads"], gt)
    chk(FAIL, "hot_accvgpr_copies", b["hot_accvgpr_copies"], c["hot_accvgpr_copies"], gt)
    chk(FAIL, "occupancy_waves_per_simd", b["occupancy_waves_per_simd"],
        c["occupancy_waves_per_simd"], lambda bv, cv: cv < bv)
    chk(FAIL, "agpr_appears", b["agpr"], c["agpr"], lambda bv, cv: bv == 0 and cv > 0)
    chk(WARN, "agpr", b["agpr"], c["agpr"], lambda bv, cv: bv > 0 and cv > bv)
    chk(WARN, "vgpr_total", b["vgpr_total"], c["vgpr_total"], gt)
    chk(WARN, "private_bytes", b["private_bytes"], c["private_bytes"], gt)
    chk(WARN, "lds_bytes", b["lds_bytes"], c["lds_bytes"], gt)
    chk(WARN, "kern_accvgpr_copies", b["kern_accvgpr_copies"], c["kern_accvgpr_copies"], gt)
    hb, hc = b.get("hot") or {}, c.get("hot") or {}
    chk(WARN, "hot_waitcnt_vmcnt0", hb.get("waitcnt_vmcnt0"), hc.get("waitcnt_vmcnt0"), gt)
    chk(WARN, "hot_waitcnt_lgkmcnt0", hb.get("waitcnt_lgkmcnt0"), hc.get("waitcnt_lgkmcnt0"), gt)
    chk(WARN, "hot_loop_lost", b["n_hot_loops"], c["n_hot_loops"],
        lambda bv, cv: bv > 0 and cv == 0)
    better = []
    for key in ("vgpr_spill", "hot_spill_reloads", "hot_accvgpr_copies", "vgpr_total", "agpr"):
        if (c[key] or 0) < (b[key] or 0):
            better.append(f"{key} {b[key]} -> {c[key]}")
    if c["occupancy_waves_per_simd"] > b["occupancy_waves_per_simd"]:
        better.append(f"occupancy {b['occupancy_waves_per_simd']} -> {c['occupancy_waves_per_simd']}")
    if better:
        out.append((INFO, "improved", "; ".join(better)))
    return out


def run_diff(base: dict, cand: dict, families, match) -> dict:
    b = _worst(select_rows(base, families, match))
    c = _worst(select_rows(cand, families, match))
    findings = []
    for k in sorted(set(b) & set(c)):
        for sev, name, detail in compare_rows(b[k], c[k]):
            findings.append({"severity": sev, "kernel": k, "label": short_name(c[k]),
                             "check": name, "detail": detail})
    for k in sorted(set(b) - set(c)):
        findings.append({"severity": WARN, "kernel": k, "label": short_name(b[k]),
                         "check": "kernel_removed", "detail": "absent from candidate"})
    for k in sorted(set(c) - set(b)):
        r = c[k]
        sev = WARN if (r["vgpr_spill"] or r["hot_spill_reloads"] or r["hot_accvgpr_copies"]) else INFO
        findings.append({"severity": sev, "kernel": k, "label": short_name(r),
                         "check": "kernel_added",
                         "detail": f"vspill={r['vgpr_spill']} hot_reload={r['hot_spill_reloads']} "
                                   f"hot_acc={r['hot_accvgpr_copies']}"})
    counts = {s: sum(f["severity"] == s for f in findings) for s in (FAIL, WARN, INFO)}
    return {"baseline": base["source"], "candidate": cand["source"], "families": families,
            "match": match, "n_common": len(set(b) & set(c)), "counts": counts,
            "findings": findings}


def format_diff(d: dict) -> str:
    lines = [f"baseline  {d['baseline']['path']} {d['baseline']['sha256'][:16]}",
             f"candidate {d['candidate']['path']} {d['candidate']['sha256'][:16]}",
             f"common kernels {d['n_common']}  FAIL {d['counts']['FAIL']}  "
             f"WARN {d['counts']['WARN']}  INFO {d['counts']['INFO']}"]
    for f in d["findings"]:
        lines.append(f"{f['severity']:<4} {f['label'][:48]:<48} {f['check']:<26} {f['detail']}")
    return "\n".join(lines)


# ------------------------------------------------------ belief-kernel feed

CLAIM_METRICS = (  # SC84 base + SC84a widening; all lower-is-better counts
    ("vgpr_total", "registers"), ("arch_vgpr", "registers"), ("agpr", "registers"),
    ("sgpr", "registers"), ("vgpr_spill", "registers"), ("sgpr_spill", "registers"),
    ("private_bytes", "bytes"), ("hot_spill_reloads", "instructions_per_iteration"),
    ("hot_accvgpr_copies", "instructions_per_iteration"),
)


CLAIM_CATEGORIES = ("BASELINE", "CANDIDATE", "OPTIMUM")


def claim_projection(row: dict, doc: dict, *, attestation_path: str, attestation_sha256: str,
                     category: str | None = None, date: str | None = None) -> list[dict]:
    """ClaimTuple keyword sets for one audit row (OBSERVATION grade, protocol_id='').

    Pure projection for a root-side adapter: it never grades and never fills an absent
    element. A metric the row lacks is skipped, and a document that states no
    ``category`` (every audit written before the write-side hook) projects nothing.
    """
    category = category or doc.get("category")
    if category not in CLAIM_CATEGORIES or row.get("stub"):
        return []
    date = date or str(doc.get("created_utc", ""))[:10]
    if not date:
        return []
    out = []
    for metric, unit in CLAIM_METRICS:
        value = row.get(metric)
        if value is None:
            continue
        locator = [row["binary_sha256"], row["code_object"], row["kernel"], metric]
        out.append({
            "measurement_id": f"{row['row_id']}:{metric}",
            "metric": f"gfx90a_static.{metric}", "value": value, "unit": unit, "date": date,
            "category": category, "metric_direction": "lower_better", "protocol_id": "",
            "reps": 1, "reps_basis": "single static read of a frozen binary",
            "claim": f"{short_name(row)} ({row['kernel']}) has {metric}={value} in binary "
                     f"{row['binary_sha256'][:16]} ({'; '.join(doc['toolchain'])})",
            "attestation_path": attestation_path, "attestation_sha256": attestation_sha256,
            "attestation_locator": json.dumps(locator, separators=(",", ":")),
            "attestation_present": True, "source_kind": SCHEMA,
            "extra": {"schema": SCHEMA, "tool_id": doc["tool_id"], "tool_sha256": doc["tool_sha256"],
                      "row_id": row["row_id"], "self_sha256": row["self_sha256"],
                      "binary_sha256": row["binary_sha256"], "toolchain": doc["toolchain"],
                      "mfma_form_regime": doc["mfma_form_regime"],
                      "source_commit": doc.get("source_commit"),
                      "flags_pragmas": doc.get("flags_pragmas"), "family": row["family"],
                      "params": row["params"], "accum_offset": row.get("accum_offset"),
                      "wg_max": row["wg_max"], "hot": row.get("hot"),
                      "hot_mfma_barrier_segments": row.get("hot_mfma_barrier_segments"),
                      "promotion_authority": False, "production_authority": False},
        })
    return out


# ---------------------------------------------------------------------- CLI


def _families(s: str | None) -> tuple[str, ...]:
    if not s:
        return DEFAULT_FAMILIES
    fams = tuple(x.strip() for x in s.split(",") if x.strip())
    if fams == ("all",):
        return fams
    bad = [f for f in fams if f not in ALL_FAMILIES]
    if bad:
        raise SystemExit(f"unknown families {bad}; choose from {ALL_FAMILIES} or 'all'")
    return fams


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit", help="audit a HIP .so / offload bundle / code object")
    a.add_argument("binary", type=Path)
    a.add_argument("--families", default=None, help=f"comma list or 'all' (default {DEFAULT_FAMILIES})")
    a.add_argument("--match", default=None, help="regex on the mangled kernel name")
    a.add_argument("--jobs", type=int, default=len(os.sched_getaffinity(0)))
    a.add_argument("--label", default=None)
    a.add_argument("--category", choices=("BASELINE", "CANDIDATE", "OPTIMUM"), default=None,
                   help="claim category for the belief kernel (omit: the audit projects no claims)")
    a.add_argument("--source-commit", default=None, help="source commit the binary was built from")
    a.add_argument("--flags", default=None, help="compile flag/pragma set of this arm (free text)")
    a.add_argument("--json", type=Path, default=None, help="write the audit document here")
    a.add_argument("--table", type=Path, default=None, help="write the readable table here")
    a.add_argument("--sort", choices=sorted(SORTS), default="spill")
    t = sub.add_parser("table", help="render a saved audit as a table")
    t.add_argument("audit", type=Path)
    t.add_argument("--families", default="all")
    t.add_argument("--match", default=None)
    t.add_argument("--sort", choices=sorted(SORTS), default="spill")
    t.add_argument("--limit", type=int, default=0)
    d = sub.add_parser("diff", help="candidate vs baseline accept gate")
    d.add_argument("baseline", type=Path)
    d.add_argument("candidate", type=Path)
    d.add_argument("--families", default="all")
    d.add_argument("--match", default=None)
    d.add_argument("--fail-on", choices=("fail", "warn"), default="fail")
    d.add_argument("--json", type=Path, default=None)
    d.add_argument("--quiet-info", action="store_true", help="omit INFO findings from the text")
    args = ap.parse_args(argv)

    if args.cmd == "audit":
        doc = run_audit(args.binary, _families(args.families), args.match, max(1, args.jobs),
                        args.label, args.category, args.source_commit, args.flags)
        text = summarize(doc) + "\n\n" + format_table(
            sorted(dedupe(select_rows(doc, None, None)), key=SORTS[args.sort]))
        if args.json:
            args.json.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        if args.table:
            args.table.write_text(text + "\n")
        if not args.json and not args.table:
            print(text)
        else:
            print(summarize(doc))
        return 0
    if args.cmd == "table":
        doc = json.loads(args.audit.read_text())
        rows = sorted(dedupe(select_rows(doc, _families(args.families), args.match)), key=SORTS[args.sort])
        if args.limit:
            rows = rows[:args.limit]
        print(format_table(rows))
        return 0
    base = json.loads(args.baseline.read_text())
    cand = json.loads(args.candidate.read_text())
    res = run_diff(base, cand, _families(args.families), args.match)
    if args.json:
        args.json.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    shown = dict(res, findings=[f for f in res["findings"]
                                if not (args.quiet_info and f["severity"] == INFO)])
    print(format_diff(shown))
    bad = res["counts"][FAIL] + (res["counts"][WARN] if args.fail_on == "warn" else 0)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
