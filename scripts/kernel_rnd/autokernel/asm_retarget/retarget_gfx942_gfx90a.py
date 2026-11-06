#!/usr/bin/env python3
"""AK-ASM-T1: checked gfx942 code object -> gfx90a code object retarget (offline, CPU only).

Generalises the hand-built H5 pipeline (ak-asm-h5-fa-retarget-20261105/tools/retarget_fa.py) so that on the
H5 inputs it emits the SAME .s and therefore the same .co, and adds every rewrite the MI300 asm inventory
(§0) named:

  R1  MFMA renames (gfx940 spelling -> gfx90a spelling), pass counts per arch for the hazard model.
  R2  exact int8 K-split: v_mfma_i32_16x16x32_i8 / 32x32x16_i8 (K-doubled, 8 bytes/lane) ->
      two v_mfma_i32_16x16x16i8 / 32x32x8i8 on the lo and hi dword of A and B; the second MFMA
      chains on the first's D as SrcC (same register => 0 wait states, hardware interlocked).
      Refused when cbsz/abid/blgp modifiers are present (they change the lane->k mapping).
  R3  v_mov_b64 -> v_pk_mov_b32 D, S, S op_sel:[0,1] (VGPR/SGPR pair) or two v_mov_b32 (constant).
  R4  v_lshl_add_u64 D, S0, sh, S2 -> v_lshlrev_b64 T, sh, S0 ; v_add_co_u32 D.lo, C, T.lo, S2.lo ;
      v_addc_co_u32 D.hi, C, T.hi, S2.hi, C.  T = D when D does not overlap S2, else a VGPR pair that
      is dead at that point (liveness), else allocation growth. C = vcc when vcc is dead (liveness),
      else a dead SGPR pair. hasShift64HighRegBug (gfx90a) is avoided (shift VGPR idx%8==7 refused).
  R5  v_fmamk_f32 D, S0, K, S1 -> v_fmac_f32_e32 D, K, S0 (D==S1), or v_mov_b32 D,S1 + v_fmac (D!=S0),
      or v_mov_b32 T,K + v_fma_f32 D,S0,T,S1 with a dead VGPR T. All bit-exact (single fma rounding).
  R6  cache-policy bits: gfx940 sc0/sc1/nt occupy the gfx90a glc/scc/slc encoding bits -> renamed 1:1.
  H   gfx90a MFMA hazard audit + s_nop padding (CFG-aware, = H5), incl. the split-MFMA SrcC chain.
  KD  kernel descriptor rebuild: accum_offset / next_free_vgpr from the actual VGPR/AGPR use (unified
      register file: arch VGPRs [0,accum_offset), AGPRs above), next_free_sgpr <= 102, private-segment
      spelling, xnack mask, kernarg size from metadata; code object v5.

Liveness (R4/R5): backward dataflow over the CFG.  SGPR/VCC defs by SALU and by VALU-to-SGPR writes
(VOPC, carry-out, readlane) kill; VGPR defs never kill (a VALU write under partial EXEC does not
define inactive lanes), so a VGPR is free only after its last use on every path.  That is conservative,
never unsafe.  Kernels that use s_set_gpr_idx_on (indexed VGPR access) get no liveness-picked VGPRs.

Usage:
  retarget_gfx942_gfx90a.py one OUT_DIR AUDIT_DIR file.co [...]      # full pipeline, writes audits
  retarget_gfx942_gfx90a.py batch ROOT LIST OUT_DIR TABLE.tsv [jobs]   # every .co, pass/fail table
Exit status of `one` is non-zero on any failure.
"""
import collections
import json
import os
import re
import subprocess
import sys
import traceback

LLVM = "/opt/rocm/llvm/bin"


class Refuse(Exception):
    """A retarget that cannot be done mechanically. str() is the table reason."""


# ----------------------------------------------------------------------------- R1 renames
RENAMES = [
    (r"\bv_mfma_f32_32x32x8_bf16\b", "v_mfma_f32_32x32x8bf16_1k"),
    (r"\bv_mfma_f32_16x16x16_bf16\b", "v_mfma_f32_16x16x16bf16_1k"),
    (r"\bv_mfma_f32_32x32x4_2b_bf16\b", "v_mfma_f32_32x32x4bf16_1k"),
    (r"\bv_mfma_f32_16x16x4_4b_bf16\b", "v_mfma_f32_16x16x4bf16_1k"),
    (r"\bv_mfma_f32_4x4x4_16b_bf16\b", "v_mfma_f32_4x4x4bf16_1k"),
    (r"\bv_mfma_f32_32x32x8_f16\b", "v_mfma_f32_32x32x8f16"),
    (r"\bv_mfma_f32_16x16x16_f16\b", "v_mfma_f32_16x16x16f16"),
    (r"\bv_mfma_f32_32x32x4_2b_f16\b", "v_mfma_f32_32x32x4f16"),
    (r"\bv_mfma_f32_16x16x4_4b_f16\b", "v_mfma_f32_16x16x4f16"),
    (r"\bv_mfma_f32_4x4x4_16b_f16\b", "v_mfma_f32_4x4x4f16"),
    (r"\bv_mfma_f32_32x32x2_f32\b", "v_mfma_f32_32x32x2f32"),
    (r"\bv_mfma_f32_16x16x4_f32\b", "v_mfma_f32_16x16x4f32"),
    (r"\bv_mfma_f32_32x32x1_2b_f32\b", "v_mfma_f32_32x32x1f32"),
    (r"\bv_mfma_f32_16x16x1_4b_f32\b", "v_mfma_f32_16x16x1f32"),
    (r"\bv_mfma_f32_4x4x1_16b_f32\b", "v_mfma_f32_4x4x1f32"),
    (r"\bv_mfma_i32_32x32x4_2b_i8\b", "v_mfma_i32_32x32x4i8"),
    (r"\bv_mfma_i32_16x16x4_4b_i8\b", "v_mfma_i32_16x16x4i8"),
    (r"\bv_mfma_i32_4x4x4_16b_i8\b", "v_mfma_i32_4x4x4i8"),
]
I8SPLIT = {"v_mfma_i32_16x16x32_i8": "v_mfma_i32_16x16x16i8", "v_mfma_i32_32x32x16_i8": "v_mfma_i32_32x32x8i8"}
FP8_RE = re.compile(r"fp8|bf8|_f8|fp4|fp6|mxf|v_smfmac", re.I)


def mfma_passes(mn, arch):
    """Pass count from SISchedule.td (ref-llvm-src, rocm-6.2.x): gfx90a lines 258-260, gfx942 280-290."""
    m = re.match(r"v_(?:mfma|smfmac)_[fi]32_(\d+)x(\d+)x(\d+)", mn)
    if not m:
        raise Refuse(f"unknown MFMA {mn}")
    mm, k = int(m.group(1)), int(m.group(3))
    if arch == "gfx90a":
        return {4: 2, 16: 8, 32: 16}[mm]
    # gfx942 (XDL): 4x4 -> 2; 16x16xK (K>=8 non-legacy) -> 4, legacy 16x16x{1,4}{f,b,i} -> 8;
    # 32x32x{8,16} and 32x32x4 xf32 -> 8, legacy 32x32x{1,2,4}{f,b,i} -> 16.
    legacy = re.search(r"_\d+b_|x[124]_?[fbi]|x1f32|x2f32", mn) is not None
    if mm == 4:
        return 2
    if mm == 16:
        return 8 if (legacy and k <= 4) else 4
    return 16 if (legacy and k <= 4) else 8


# Required wait states per rule keyed by producer MFMA pass count (= H5 table; ROCm 6.2 GCNHazardRecognizer
# checkMAIHazards90A / checkMAIVALUHazards).
RULES = {
    "gfx90a": {"READ": {2: 5, 8: 11, 16: 19}, "WAW": {2: 5, 8: 11, 16: 19}, "WAR": {2: 1, 8: 7, 16: 15},
               "AB": {2: 5, 8: 11, 16: 19}, "COVL": {2: 2, 8: 8, 16: 16}},
    "gfx942": {"READ": {2: 5, 4: 7, 8: 11, 16: 19}, "WAW": {2: 5, 4: 7, 8: 11, 16: 19},
               "WAR": {2: 1, 4: 3, 8: 7, 16: 15}, "AB": {2: 5, 4: 7, 8: 11, 16: 19},
               "COVL": {2: 3, 4: 5, 8: 9, 16: 17}},
}
VALU_TO_MFMA = 2
VALU_EXEC_TO_MFMA = 4
M0_TO_LDSDMA = 1
VALU_SGPR_TO_VMEM = 5
STORE12_TO_VALU = 1
CAP = 19


def run(cmd, **kw):
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if p.returncode != 0:
        raise Refuse("tool failed: " + " ".join(os.path.basename(c) for c in cmd[:1]) + ": "
                     + (p.stderr or p.stdout)[:4000])
    return p


# ----------------------------------------------------------------------------- parsing
def disassemble(co, mcpu, kname=None):
    """kname: start at this symbol (drops section-alignment padding before the kernel)."""
    t = run([f"{LLVM}/llvm-objdump", "-d", f"--mcpu={mcpu}", co]).stdout
    items, kernel = [], None
    started = kname is None
    for line in t.splitlines():
        m = re.match(r"^([0-9a-f]{16}) <(.+)>:$", line)
        if m and not started and m.group(2) == kname:
            started = True
        if not started:
            continue
        if m:
            name = m.group(2)
            if kernel is None:
                kernel = name
                items.append(("label", "__entry"))
            else:
                items.append(("label", name))
            continue
        if not line.startswith("\t"):
            continue
        body, _, cmt = line.partition("//")
        body = body.strip()
        if not body:
            continue
        m = re.match(r"\s*([0-9A-F]+):\s*(.*)$", cmt)
        addr, enc = (int(m.group(1), 16), m.group(2).split()) if m else (None, [])
        items.append(("inst", body, addr, enc))
    # branches objdump left numeric (no symbol at the target): synthesize a label at addr+4+4*simm16
    want = {}
    for i, it in enumerate(items):
        if it[0] != "inst":
            continue
        m = re.match(r"(s_branch|s_cbranch_\w+)\s+(-?\d+)$", it[1])
        if m and it[2] is not None:
            simm = int(m.group(2))
            simm = simm - 65536 if simm >= 32768 else simm  # objdump prints simm16 unsigned
            tgt = it[2] + 4 + 4 * simm
            lab = f"L_{tgt:x}"
            want[tgt] = lab
            items[i] = ("inst", f"{m.group(1)} {lab}", it[2], it[3])
    if want:
        out, have = [], set()
        for it in items:
            if it[0] == "inst" and it[2] in want and it[2] not in have:
                out.append(("label", want[it[2]]))
                have.add(it[2])
            out.append(it)
        if set(want) - have:
            raise Refuse(f"branch target not at an instruction boundary: {sorted(set(want) - have)[:3]}")
        items = out
    return kernel, items


REG_RE = re.compile(r"^([va])(?:\[(\d+):(\d+)\]|(\d+))$")


def strip_mods(tok):
    tok = tok.strip()
    while tok and tok[0] in "-|!":
        tok = tok[1:]
    tok = tok.rstrip("|")
    m = re.match(r"^(?:neg|abs|sext)\((.*)\)$", tok)
    return m.group(1) if m else tok


def reg_of(tok):
    m = REG_RE.match(strip_mods(tok))
    if not m:
        return None
    if m.group(4) is not None:
        lo = hi = int(m.group(4))
    else:
        lo, hi = int(m.group(2)), int(m.group(3))
    return (m.group(1), lo, hi)


def sgprs_of(tok):
    tok = strip_mods(tok)
    m = re.match(r"^s\[(\d+):(\d+)\]$", tok)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    m = re.match(r"^s(\d+)$", tok)
    return [int(m.group(1))] if m else []


def split_ops(body):
    mn = body.split()[0]
    rest = body[len(mn):].strip()
    ops, mods = [], []
    if rest:
        for p in [p.strip() for p in rest.split(",")]:
            toks = p.split()
            if toks:
                ops.append(toks[0])
                mods.extend(toks[1:])
    return mn, ops, mods


def classify(body):
    """= H5 classify, plus v_mad_u64_u32 sdst."""
    mn, ops, mods = split_ops(body)
    d = {"mn": mn, "defs": [], "uses": [], "srcC": None, "srcAB": [], "writes_exec": False, "ws": 1,
         "writes_m0": False, "lds_dma": False, "valu_sgpr_defs": [], "vmem_sgpr_uses": [], "bigstore": None}
    regs = [reg_of(o) for o in ops]
    if mn == "s_nop":
        d["kind"] = "nop"
        d["ws"] = int(ops[0], 0) + 1
    elif mn == "s_endpgm":
        d["kind"] = "end"
    elif mn == "s_branch":
        d["kind"] = "branch"; d["target"] = ops[0]
    elif mn.startswith("s_cbranch"):
        d["kind"] = "cbranch"; d["target"] = ops[0]
    elif mn.startswith("s_"):
        d["kind"] = "salu"
        if ops and ops[0] == "m0" and not mn.startswith(("s_cmp", "s_bitcmp")):
            d["writes_m0"] = True
    elif mn.startswith("v_mfma") or mn.startswith("v_smfmac"):
        d["kind"] = "mfma"
        d["defs"] = [regs[0]] if regs[0] else []
        d["srcAB"] = [r for r in regs[1:3] if r]
        d["srcC"] = regs[3] if len(regs) > 3 else None
        d["uses"] = [r for r in regs[1:] if r]
    elif mn.startswith("v_"):
        d["kind"] = "valu"
        if mn in ("v_swap_b32", "v_swaprel_b32"):
            raise Refuse(f"unhandled two-def VALU: {body}")
        if ops:
            if ops[0] == "exec" or mn.startswith("v_cmpx"):
                d["writes_exec"] = True
            d["defs"] = [regs[0]] if regs[0] else []
            d["uses"] = [r for r in regs[1:] if r]
            d["valu_sgpr_defs"] = list(sgprs_of(ops[0]))
            if ("_co_" in mn or "div_scale" in mn or "mad_u64" in mn or "mad_i64" in mn) and len(ops) > 1:
                d["valu_sgpr_defs"] += sgprs_of(ops[1])
    elif mn.startswith(("ds_", "buffer_", "global_", "flat_", "scratch_")):
        d["kind"] = "mem"
        is_load_like = (mn.startswith(("buffer_load", "global_load", "flat_load", "scratch_load"))
                        or mn.startswith(("ds_read", "ds_permute", "ds_bpermute", "ds_swizzle"))
                        or "_rtn" in mn
                        or (re.search(r"atomic", mn) and "glc" in mods))
        if "lds" in mods:
            is_load_like = False
            d["lds_dma"] = True
        if not mn.startswith("ds_"):
            d["vmem_sgpr_uses"] = sorted({x for o in ops for x in sgprs_of(o)})
        if re.match(r"^(buffer_store_dwordx[34]|global_store_dwordx[34]|flat_store_dwordx[34]|"
                    r"buffer_store_b(96|128)|global_store_b(96|128))$", mn):
            d["bigstore"] = regs[0] if mn.startswith("buffer_") else regs[1]
        if is_load_like:
            d["defs"] = [regs[0]] if regs[0] else []
            d["uses"] = [r for r in regs[1:] if r]
        else:
            d["uses"] = [r for r in regs if r]
    else:
        raise Refuse(f"unclassified instruction: {body}")
    return d


def overlap(a, b):
    return a[0] == b[0] and not (a[2] < b[1] or a[1] > b[2])


# ----------------------------------------------------------------------------- CFG
def build_blocks(items):
    blocks, cur = [], None
    for idx, it in enumerate(items):
        if it[0] == "label":
            if cur is not None and (cur["insts"] or cur["labels"]):
                blocks.append(cur)
            cur = {"labels": [it[1]], "insts": []}
            continue
        if cur is None:
            cur = {"labels": [], "insts": []}
        cur["insts"].append(idx)
        if it[4]["kind"] in ("branch", "cbranch", "end"):
            blocks.append(cur)
            cur = {"labels": [], "insts": []}
    if cur is not None and (cur["insts"] or cur["labels"]):
        blocks.append(cur)
    blocks = [b for b in blocks if b["insts"] or b["labels"]]
    lab2blk = {l: bi for bi, b in enumerate(blocks) for l in b["labels"]}
    for bi, b in enumerate(blocks):
        last = items[b["insts"][-1]][4] if b["insts"] else None
        nxt = [bi + 1] if bi + 1 < len(blocks) else []
        if last is None:
            succ = nxt
        elif last["kind"] == "end":
            succ = []
        elif last["kind"] in ("branch", "cbranch"):
            if last["target"] not in lab2blk:
                raise Refuse(f"branch to unknown label {last['target']}")
            succ = [lab2blk[last["target"]]] + (nxt if last["kind"] == "cbranch" else [])
        else:
            succ = nxt
        b["succ"] = succ
    for b in blocks:
        b["pred"] = []
    for bi, b in enumerate(blocks):
        for s in b["succ"]:
            blocks[s]["pred"].append(bi)
    return blocks


# ----------------------------------------------------------------------------- hazard dataflow (= H5)
def empty_state():
    return {"mw": {}, "mc": {}, "vw": {}, "ex": None, "vs": {}, "m0": None, "st": {}}


def merge(states):
    out = empty_state()
    for s in states:
        if s is None:
            continue
        for key in ("mw", "mc"):
            for r, dd in s[key].items():
                o = out[key].setdefault(r, {})
                for k, v in dd.items():
                    o[k] = min(o.get(k, 99), v)
        for key in ("vw", "vs", "st"):
            for r, v in s[key].items():
                out[key][r] = min(out[key].get(r, 99), v)
        for key in ("ex", "m0"):
            if s[key] is not None:
                out[key] = s[key] if out[key] is None else min(out[key], s[key])
    return out


def regs_in(r):
    return [(r[0], i) for i in range(r[1], r[2] + 1)]


def check_and_step(state, d, arch):
    rules = RULES[arch]
    need, finds = 0, []

    def want(rule, req, dist, detail):
        nonlocal need
        if req - dist > 0:
            finds.append({"rule": rule, "required": req, "have": dist, "deficit": req - dist, "reg": detail})
            need = max(need, req - dist)

    k = d["kind"]
    if d["lds_dma"] and state["m0"] is not None:
        want("M0_LDSDMA", M0_TO_LDSDMA, state["m0"], "m0")
    for sg in d["vmem_sgpr_uses"]:
        if sg in state["vs"]:
            want("VALUSGPR_VMEM", VALU_SGPR_TO_VMEM, state["vs"][sg], f"s{sg}")
    if k == "valu":
        for df in d["defs"]:
            for rr in regs_in(df):
                if rr in state["st"]:
                    want("STORE12", STORE12_TO_VALU, state["st"][rr], f"{rr[0]}{rr[1]}")
    if k in ("valu", "mem"):
        for u in d["uses"]:
            for rr in regs_in(u):
                for (dst, p), dist in state["mw"].get(rr, {}).items():
                    want("READ", rules["READ"][p], dist, f"{rr[0]}{rr[1]}")
        for df in d["defs"]:
            for rr in regs_in(df):
                for (dst, p), dist in state["mw"].get(rr, {}).items():
                    want("WAW", rules["WAW"][p], dist, f"{rr[0]}{rr[1]}")
                for p, dist in state["mc"].get(rr, {}).items():
                    want("WAR", rules["WAR"][p], dist, f"{rr[0]}{rr[1]}")
    if k == "mfma":
        if state["ex"] is not None:
            want("EXEC", VALU_EXEC_TO_MFMA, state["ex"], "exec")
        for u in d["uses"]:
            for rr in regs_in(u):
                if rr in state["vw"]:
                    want("VALU2MFMA", VALU_TO_MFMA, state["vw"][rr], f"{rr[0]}{rr[1]}")
        for u in d["srcAB"]:
            for rr in regs_in(u):
                for (dst, p), dist in state["mw"].get(rr, {}).items():
                    want("AB", rules["AB"][p], dist, f"{rr[0]}{rr[1]}")
        c = d["srcC"]
        if c:
            for rr in regs_in(c):
                for (dst, p), dist in state["mw"].get(rr, {}).items():
                    if dst == c:
                        continue  # FullReg SrcC chain (incl. the K-split pair): 0 wait states on gfx90a
                    want("COVL", rules["COVL"][p], dist, f"{rr[0]}{rr[1]}")
    ws = d["ws"]
    for key in ("mw", "mc"):
        nm = {}
        for r, dd in state[key].items():
            nd = {kk: v + ws for kk, v in dd.items() if v + ws <= CAP}
            if nd:
                nm[r] = nd
        state[key] = nm
    state["vw"] = {r: v + ws for r, v in state["vw"].items() if v + ws <= CAP}
    if state["ex"] is not None:
        state["ex"] = state["ex"] + ws if state["ex"] + ws <= CAP else None
    state["vs"] = {r: v + ws for r, v in state["vs"].items() if v + ws <= VALU_SGPR_TO_VMEM}
    state["st"] = {r: v + ws for r, v in state["st"].items() if v + ws <= STORE12_TO_VALU}
    if state["m0"] is not None:
        state["m0"] = state["m0"] + ws if state["m0"] + ws <= M0_TO_LDSDMA else None
    if d["writes_m0"]:
        state["m0"] = 0
    for sg in d["valu_sgpr_defs"]:
        state["vs"][sg] = 0
    if d["bigstore"]:
        for rr in regs_in(d["bigstore"]):
            state["st"][rr] = 0
    if k == "mfma":
        p = mfma_passes(d["mn"], arch)
        dst = d["defs"][0]
        for rr in regs_in(dst):
            state["mw"].setdefault(rr, {})[(dst, p)] = 0
        if d["srcC"]:
            for rr in regs_in(d["srcC"]):
                state["mc"].setdefault(rr, {})[p] = 0
    elif k == "valu":
        for df in d["defs"]:
            for rr in regs_in(df):
                state["vw"][rr] = 0
        if d["writes_exec"]:
            state["ex"] = 0
    return need, finds


def _copy(st):
    return {"mw": {r: dict(v) for r, v in st["mw"].items()}, "mc": {r: dict(v) for r, v in st["mc"].items()},
            "vw": dict(st["vw"]), "ex": st["ex"], "vs": dict(st["vs"]), "m0": st["m0"], "st": dict(st["st"])}


def audit(items, arch):
    blocks = build_blocks(items)
    IN, OUT = [None] * len(blocks), [None] * len(blocks)
    changed, it = True, 0
    while changed:
        changed, it = False, it + 1
        for bi, b in enumerate(blocks):
            st = merge([OUT[p] for p in b["pred"]]) if b["pred"] else empty_state()
            IN[bi] = st
            s = _copy(st)
            for idx in b["insts"]:
                check_and_step(s, items[idx][4], arch)
            if OUT[bi] != s:
                OUT[bi] = s
                changed = True
        if it > 200:
            raise Refuse("hazard dataflow did not converge")
    deficits = []
    for bi, b in enumerate(blocks):
        s = _copy(IN[bi])
        for idx in b["insts"]:
            need, finds = check_and_step(s, items[idx][4], arch)
            if need > 0:
                deficits.append({"block": bi, "idx": idx, "need": need, "inst": items[idx][1],
                                 "multi_pred": len(b["pred"]) > 1, "first_in_block": idx == b["insts"][0],
                                 "findings": finds})
    return blocks, deficits, it


def rule_hist(defs):
    c = collections.Counter()
    for dfc in defs:
        for r in sorted({f["rule"] for f in dfc["findings"]}):
            c[r] += 1
    return dict(c)


def cf_in_window(items, idx, window=19):
    ev, ws, j = [], 0, idx - 1
    while j >= 0 and ws < window:
        it = items[j]
        if it[0] == "label":
            ev.append("label:" + it[1])
        else:
            if it[4]["kind"] in ("branch", "cbranch") or it[4]["mn"] == "s_barrier":
                ev.append(it[4]["mn"])
            ws += it[4]["ws"]
        j -= 1
    return ev


def nop_items(n, tag):
    out = []
    while n > 0:
        a = min(n, 8)
        n -= a
        txt = f"s_nop {a - 1}"
        out.append(("inst", txt, None, [], classify(txt), tag))
    return out


# ----------------------------------------------------------------------------- liveness (R4/R5)
SALU_NOKILL = ("s_cmp", "s_bitcmp", "s_store", "s_atomic", "s_cbranch", "s_branch", "s_waitcnt", "s_nop",
               "s_barrier", "s_setprio", "s_set_gpr_idx", "s_endpgm", "s_trap", "s_sleep", "s_setvskip",
               "s_addk", "s_mulk", "s_cmovk", "s_cmov", "s_setreg", "s_dcache", "s_icache", "s_sendmsg")


def reg_units(tok):
    """All register units a token names: ('v',i) ('a',i) ('s',i) ('vcc',0/1)."""
    t = strip_mods(tok)
    r = reg_of(t)
    if r:
        return [(r[0], i) for i in range(r[1], r[2] + 1)]
    s = sgprs_of(t)
    if s:
        return [("s", i) for i in s]
    if t == "vcc":
        return [("vcc", 0), ("vcc", 1)]
    if t == "vcc_lo":
        return [("vcc", 0)]
    if t == "vcc_hi":
        return [("vcc", 1)]
    return []


def def_use(body):
    """(kills, uses) as sets of register units. Uses over-approximate (every operand), kills only SGPR/VCC."""
    mn, ops, mods = split_ops(body)
    uses = set()
    for o in ops:
        uses.update(reg_units(o))
    if mn.startswith("s_cbranch_vcc") or (mn.endswith("_e32") and ("addc" in mn or "subb" in mn or "cndmask" in mn)):
        uses.update([("vcc", 0), ("vcc", 1)])
    kills = set()
    if mn.startswith("s_") and not mn.startswith(SALU_NOKILL) and ops:
        kills.update(u for u in reg_units(ops[0]) if u[0] in ("s", "vcc"))
    elif mn.startswith("v_") and ops and "_sdwa" not in mn and "_dpp" not in mn:
        if mn.startswith("v_cmp_") or mn.startswith("v_readlane") or mn.startswith("v_readfirstlane"):
            kills.update(u for u in reg_units(ops[0]) if u[0] in ("s", "vcc"))
        if ("_co_" in mn or "mad_u64" in mn or "mad_i64" in mn) and len(ops) > 1:
            kills.update(u for u in reg_units(ops[1]) if u[0] in ("s", "vcc"))
    # a register that is killed AND read by the same instruction stays live (read first)
    own_reads = set()
    for o in ops[1:]:
        own_reads.update(reg_units(o))
    if mn.startswith("v_") and ("_co_" in mn or "mad_u64" in mn) and len(ops) > 2:
        own_reads = set()
        for o in ops[2:]:
            own_reads.update(reg_units(o))
    if mn.endswith("_e32") and ("addc" in mn or "subb" in mn):
        own_reads.update([("vcc", 0), ("vcc", 1)])
    return kills, own_reads | (uses - kills)


def live_after(items):
    """For each inst index: set of register units live immediately AFTER it."""
    blocks = build_blocks(items)
    du = {idx: def_use(items[idx][1]) for b in blocks for idx in b["insts"]}
    LIN = [set() for _ in blocks]
    changed = True
    while changed:
        changed = False
        for bi in range(len(blocks) - 1, -1, -1):
            b = blocks[bi]
            live = set().union(*[LIN[s] for s in b["succ"]]) if b["succ"] else set()
            for idx in reversed(b["insts"]):
                k, u = du[idx]
                live = (live - k) | u
            if live != LIN[bi]:
                LIN[bi] = live
                changed = True
    after = {}
    for bi, b in enumerate(blocks):
        live = set().union(*[LIN[s] for s in b["succ"]]) if b["succ"] else set()
        for idx in reversed(b["insts"]):
            after[idx] = set(live)
            k, u = du[idx]
            live = (live - k) | u
    return after


# ----------------------------------------------------------------------------- R2..R6 rewrites
def cache_bits(body):
    # gfx940 sc0/sc1/nt are the gfx90a glc/scc/slc bits (same encoding positions).
    mn = body.split()[0]
    if not mn.startswith(("buffer_", "global_", "flat_", "scratch_", "s_load", "s_buffer", "s_store", "s_atomic")):
        return body
    body = re.sub(r"\bsc0\b", "glc", body)
    body = re.sub(r"\bsc1\b", "scc", body)
    return re.sub(r"\bnt\b", "slc", body)


def int_const(tok):
    try:
        v = int(tok, 0)
    except ValueError:
        return None
    return v


def halves(tok):
    """(lo, hi) spelling of a 64-bit operand (reg pair or integer constant). None if not expressible."""
    r = reg_of(tok)
    if r:
        if r[2] != r[1] + 1:
            return None
        return f"{r[0]}{r[1]}", f"{r[0]}{r[1] + 1}"
    s = sgprs_of(tok)
    if len(s) == 2:
        return f"s{s[0]}", f"s{s[1]}"
    v = int_const(tok)
    if v is not None and -16 <= v <= 64:
        return str(v), ("-1" if v < 0 else "0")
    return None


class Ctx:
    def __init__(self, items, has_gpr_idx, max_v, vlimit):
        self.items = items
        self.after = None
        self.has_gpr_idx = has_gpr_idx
        self.used_v = set()
        self.used_s = set()
        self.max_v = max_v
        self.vlimit = vlimit
        self.grown = []
        self.pool = set()
        self.scratch = None
        self.okd = {}
        self.preload = False
        self.max_a = max_reg([it[1] for it in items if it[0] == "inst"], "a")
        for it in items:
            if it[0] == "inst":
                for o in split_ops(it[1])[1]:
                    for u in reg_units(o):
                        (self.used_v if u[0] == "v" else self.used_s if u[0] == "s" else set()).add(u[1])

    def live(self, idx):
        if self.after is None:
            self.after = live_after(self.items)
        return self.after[idx]

    def free_vgpr(self, idx, n, exclude=()):
        live = self.live(idx)
        if not self.has_gpr_idx:
            for base in range(0, self.max_v + 1, n):
                regs = list(range(base, base + n))
                if base + n - 1 > self.max_v:
                    break
                if any(("v", r) in live or r in exclude for r in regs):
                    continue
                return base, "liveness"
        # Grown VGPRs are named by no original instruction, only by rewrite sequences; every sequence writes
        # its temps before reading them and leaves nothing in them, so they form a scratch pool any later site
        # may reuse (sequences never nest). The VGPR liveness (defs never kill) cannot see that by itself.
        for base in sorted(self.pool):
            if base % n == 0 and all(r in self.pool and r not in exclude for r in range(base, base + n)):
                return base, "scratch-reuse"
        base = ((self.max_v + 1 + n - 1) // n) * n
        if base + n <= self.vlimit:
            self.max_v = base + n - 1
            self.grown.append(base)
            self.pool.update(range(base, base + n))
            return base, "grown"
        return None, None

    def free_sgpr_pair(self, idx):
        live = self.live(idx)
        if ("vcc", 0) not in live and ("vcc", 1) not in live:
            return "vcc"
        for base in range(0, 102, 2):
            if ("s", base) not in live and ("s", base + 1) not in live:
                return f"s[{base}:{base + 1}]"
        return None


def rw_mov_b64(body):
    m = re.match(r"v_mov_b64(?:_e32|_e64)?\s+(\S+),\s*(\S+)$", body)
    if not m:
        raise Refuse(f"v_mov_b64 form not handled: {body}")
    d, s = m.group(1), m.group(2)
    if reg_of(s) or len(sgprs_of(s)) == 2:
        return [f"v_pk_mov_b32 {d}, {s}, {s} op_sel:[0,1]"]
    h, dh = halves(s), halves(d)
    if h and dh:
        return [f"v_mov_b32_e32 {dh[0]}, {h[0]}", f"v_mov_b32_e32 {dh[1]}, {h[1]}"]
    raise Refuse(f"v_mov_b64 literal source: {body}")


def rw_lshl_add_u64(body, idx, ctx):
    m = re.match(r"v_lshl_add_u64\s+(\S+),\s*(\S+),\s*(\S+),\s*(\S+)$", body)
    if not m:
        raise Refuse(f"v_lshl_add_u64 form not handled: {body}")
    D, S0, SH, S2 = m.groups()
    dr = reg_of(D)
    if not dr or halves(D) is None:
        raise Refuse(f"v_lshl_add_u64 dst: {body}")
    s2h, s0r = halves(S2), reg_of(S0)
    if s2h is None:
        raise Refuse(f"v_lshl_add_u64 S2 not reg pair / inline int: {body}")
    if reg_of(SH) and reg_of(SH)[1] % 8 == 7:
        raise Refuse(f"hasShift64HighRegBug shift operand: {body}")
    shv = int_const(SH)
    s2r = reg_of(S2)
    # T: where the shifted value goes
    if shv == 0:
        th = halves(S0)
        if th is None:
            raise Refuse(f"v_lshl_add_u64 S0: {body}")
        pre = []
        how = "shift0"
    else:
        if not (s2r and overlap(dr, s2r)):
            T, how = D, "T=D"
        else:
            base, how = ctx.free_vgpr(idx, 2)
            if base is None:
                raise Refuse(f"no free VGPR pair for v_lshl_add_u64: {body}")
            T = f"v[{base}:{base + 1}]"
        th = halves(T)
        pre = [f"v_lshlrev_b64 {T}, {SH}, {S0}"]
    C = ctx.free_sgpr_pair(idx)
    if C is None:
        raise Refuse(f"no dead VCC/SGPR pair for v_lshl_add_u64 carry: {body}")
    dh = halves(D)
    # gfx9 constant-bus limit is 1 per VALU: the carry-in SGPR pair already uses it in v_addc, so an SGPR
    # hi source goes through a separate v_add_u32 (hi = (T.hi + 0 + carry) + S2.hi mod 2^32, exact).
    iss = lambda x: x.startswith("s")
    if iss(th[0]) and iss(s2h[0]):
        raise Refuse(f"v_lshl_add_u64 two SGPR sources: {body}")
    seq = pre + [f"v_add_co_u32_e64 {dh[0]}, {C}, {th[0]}, {s2h[0]}"]
    a, b = th[1], s2h[1]
    if not iss(a) and not iss(b):
        seq.append(f"v_addc_co_u32_e64 {dh[1]}, {C}, {a}, {b}, {C}")
    elif iss(a) and iss(b):
        raise Refuse(f"v_lshl_add_u64 two SGPR hi sources: {body}")
    else:
        sg, vg = (a, b) if iss(a) else (b, a)
        seq.append(f"v_addc_co_u32_e64 {dh[1]}, {C}, {vg}, 0, {C}")
        seq.append(f"v_add_u32_e32 {dh[1]}, {sg}, {dh[1]}")
    # the lo write must not clobber an input of the hi add
    if dh[0] in (th[1], s2h[1]):
        raise Refuse(f"v_lshl_add_u64 lo/hi aliasing: {body}")
    return seq, f"{how},carry={C}"


def rw_fmamk(body, idx, ctx):
    m = re.match(r"v_fmamk_f32(?:_e32)?\s+(\S+),\s*(\S+),\s*(\S+),\s*(\S+)$", body)
    if not m:
        raise Refuse(f"v_fmamk_f32 form not handled: {body}")
    D, S0, K, S1 = m.groups()
    if D == S1 and D != S0:
        return [f"v_fmac_f32_e32 {D}, {K}, {S0}"], "fmac(D==S1)"
    if D != S0 and reg_of(S0):
        mov = [] if D == S1 else [f"v_mov_b32_e32 {D}, {S1}"]
        return mov + [f"v_fmac_f32_e32 {D}, {K}, {S0}"], "mov+fmac"
    k = int_const(K)
    base, how = ctx.free_vgpr(idx, 1)
    if base is None:
        raise Refuse(f"no free VGPR for v_fmamk_f32: {body}")
    return [f"v_mov_b32_e32 v{base}, {K}", f"v_fma_f32 {D}, {S0}, v{base}, {S1}"], f"mov+fma,{how}"


def rw_i8split(body):
    m = re.match(r"(v_mfma_i32_(?:16x16x32|32x32x16)_i8)\s+(\S+),\s*(\S+),\s*(\S+),\s*(\S+)(.*)$", body)
    if not m:
        raise Refuse(f"int8 MFMA form not handled: {body}")
    mn, d, a, b, c, mod = m.groups()
    if re.search(r"cbsz|abid|blgp", mod):
        raise Refuse(f"int8 split with cbsz/abid/blgp: {body}")
    A, B = reg_of(a), reg_of(b)
    if not A or not B or A[2] - A[1] != 1 or B[2] - B[1] != 1:
        raise Refuse(f"int8 split operands: {body}")
    new = I8SPLIT[mn]
    if overlap(reg_of(d), A) or overlap(reg_of(d), B):
        raise Refuse(f"int8 split D overlaps A/B: {body}")
    return [f"{new} {d}, {A[0]}{A[1]}, {B[0]}{B[1]}, {c}{mod}",
            f"{new} {d}, {A[0]}{A[1] + 1}, {B[0]}{B[1] + 1}, {d}{mod}"]


def _units_of_ops(ops):
    u = set()
    for o in ops:
        u.update(reg_units(o))
    return u


def _dead_sgpr_pairs(ctx, idx, k, exclude_units):
    """k distinct dead SGPR pairs (vcc first if dead). Liveness = live_after(site) | units the site reads."""
    live = ctx.live(idx) | exclude_units
    out = []
    if ("vcc", 0) not in live and ("vcc", 1) not in live:
        out.append("vcc")
    for base in range(0, 102, 2):
        if len(out) >= k:
            break
        if ("s", base) not in live and ("s", base + 1) not in live:
            out.append(f"s[{base}:{base + 1}]")
    return out[:k] if len(out) >= k else None


def _dead_vgprs(ctx, idx, n, exclude_units):
    """n consecutive VGPRs, base aligned to n (gfx90a 64-bit tuples must be even-aligned), dead after the
    site and not read by it. Liveness-proven; allocation growth into never-named VGPRs as fallback."""
    ex = {u[1] for u in exclude_units if u[0] == "v"}
    return ctx.free_vgpr(idx, n, exclude=ex)


DRAIN = "s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)"


def _borrow_vgprs(ctx, idx, n, exu):
    """Fallback when no VGPR is dead: borrow n consecutive (n-aligned) VGPRs the site does not read and park their
    values in n AGPRs that are dead across the site (liveness; AGPRs past the highest named one count as dead, the KD
    rebuild refuses if that overflows the unified 512-entry file). Saves/restores run under the ENTRY exec, and every
    temp write inside the sequence runs under a subset of it, so lanes outside the entry exec are never touched.
    Returns (base, pre, post, how) or None."""
    live = ctx.live(idx) | exu
    dead_a = [i for i in range(256) if ("a", i) not in live][:n]
    ex = {u[1] for u in exu if u[0] == "v"}
    if len(dead_a) < n:
        return _scratch_spill(ctx, n, ex, idx, exu)
    base = next((b for b in range(0, ctx.max_v + 1 - (n - 1), n) if all(r not in ex for r in range(b, b + n))), None)
    if base is None:
        return None
    pre = [f"v_accvgpr_write_b32 a{a}, v{base + j}" for j, a in enumerate(dead_a)]
    post = [f"v_accvgpr_read_b32 v{base + j}, a{a}" for j, a in enumerate(dead_a)]
    ctx.max_a = max(ctx.max_a, max(dead_a))
    return base, pre, post, "borrowed,park=" + ",".join(f"a{a}" for a in dead_a)


SCRATCH_SLOT_DWORDS = 4  # per-lane private segment: one dword per spilled VGPR (max 4 at any site)


def _scratch_reserve(ctx):
    """Reserve, once per kernel, an aligned SGPR quad + one SGPR that NO instruction of the kernel names: they receive the
    private segment buffer descriptor and the wave scratch offset in the entry prologue (see scratch_prologue)."""
    if ctx.scratch is not None:
        return ctx.scratch
    if ctx.preload:
        raise Refuse("scratch spill with kernarg preload prologue not supported")
    kd = ctx.okd
    if kd.get("enable_private_segment", "0") != "0" or kd.get("user_sgpr_private_segment_buffer", "0") != "0":
        raise Refuse("scratch spill: kernel already has a private segment")
    for k in ("user_sgpr_dispatch_ptr", "user_sgpr_queue_ptr", "user_sgpr_dispatch_id", "user_sgpr_flat_scratch_init",
              "user_sgpr_private_segment_size", "system_sgpr_workgroup_info"):
        if kd.get(k, "0") != "0":
            raise Refuse(f"scratch spill: SGPR layout with {k} not handled")
    free = [i for i in range(102) if i not in ctx.used_s]
    q = next((b for b in range(0, 99, 4) if all(r in free for r in range(b, b + 4))), None)
    if q is not None:
        store = list(range(q, q + 4))
    else:
        # no aligned never-named quad: keep the descriptor in 4 scattered never-named SGPRs and assemble it at each spill
        # site into an aligned quad that is dead there (liveness)
        if len(free) < 4:
            raise Refuse(f"scratch spill: only {len(free)} never-named SGPRs (need 4)")
        store = free[:4]
    # with no 5th never-named SGPR the wave scratch offset is folded into the descriptor base in the prologue
    # (s_add_u32/s_addc_u32 at entry, where SCC carries nothing) and the spills use soffset 0
    w = next((r for r in free if r not in store), None)
    nuser = 2 * int(kd.get("user_sgpr_kernarg_segment_ptr", "0"))
    nsys = sum(int(kd.get(f"system_sgpr_workgroup_id_{a}", "0")) for a in "xyz")
    ctx.used_s.update(store + ([w] if w is not None else []))
    ctx.scratch = {"q": q, "store": store, "w": w, "nuser_old": nuser, "nsys": nsys}
    return ctx.scratch


def _scratch_spill(ctx, n, ex, idx=None, exu=frozenset()):
    """Last fallback (no dead VGPR, no dead AGPR): spill n VGPRs the site does not read to the private segment
    (per-lane, swizzled by the runtime's scratch descriptor) around the sequence; restore after it under the entry exec."""
    sc = _scratch_reserve(ctx)
    base = next((b for b in range(0, ctx.max_v + 1 - (n - 1), n) if all(r not in ex for r in range(b, b + n))), None)
    if base is None:
        return None
    W = f"s{sc['w']}" if sc["w"] is not None else "0"
    asm = []
    ctx.spill_sgpr_units = set()
    if sc["q"] is not None:
        Q = f"s[{sc['q']}:{sc['q'] + 3}]"
    else:
        live = ctx.live(idx) | exu
        qa = next((b for b in range(0, 99, 4) if all(("s", r) not in live and r not in sc["store"] and r != sc["w"]
                                                      for r in range(b, b + 4))), None)
        if qa is None:
            raise Refuse("scratch spill: no dead aligned SGPR quad at the site to assemble the descriptor")
        Q = f"s[{qa}:{qa + 3}]"
        asm = [f"s_mov_b32 s{qa + j}, s{r}" for j, r in enumerate(sc["store"])]
        ctx.spill_sgpr_units = {("s", qa + j) for j in range(4)}
    pre = asm + [f"buffer_store_dword v{base + j}, off, {Q}, {W} offset:{4 * j}" for j in range(n)] + ["s_waitcnt vmcnt(0)"]
    post = [f"buffer_load_dword v{base + j}, off, {Q}, {W} offset:{4 * j}" for j in range(n)] + ["s_waitcnt vmcnt(0)"]
    return base, pre, post, f"scratch-spill,{Q},{W}"


def scratch_prologue(sc):
    """gfx90a with the private segment enabled puts the 4-SGPR private segment buffer FIRST, shifting every other user/system
    SGPR up by 4, and appends the wave scratch offset after the workgroup ids. Save both into the reserved SGPRs, then move
    the original layout back down (ascending copy, source index > destination) so the gfx942 code sees its own layout."""
    st, w, n = sc["store"], sc["w"], sc["nuser_old"] + sc["nsys"]
    if w is not None:
        seq = [f"s_mov_b32 s{r}, s{j}" for j, r in enumerate(st)] + [f"s_mov_b32 s{w}, s{n + 4}"]
    else:
        seq = [f"s_add_u32 s{st[0]}, s0, s{n + 4}", f"s_addc_u32 s{st[1]}, s1, 0", f"s_mov_b32 s{st[2]}, s2",
               f"s_mov_b32 s{st[3]}, s3"]
    seq += [f"s_mov_b32 s{i}, s{i + 4}" for i in range(n)]
    return seq


def _bf16_round(x, t, m):
    """x holds an f32; leaves bf16-RNE(x) in x[31:16] (NaN -> quiet NaN). t: VGPR temp, m: SGPR-pair temp.
    VOP2 forms carry the literals (gfx90a VOP3 has no literal)."""
    return [f"v_bfe_u32 {t}, {x}, 16, 1",
            f"v_add_u32_e32 {t}, 0x7fff, {t}",
            f"v_add_u32_e32 {t}, {x}, {t}",
            f"v_cmp_u_f32_e64 {m}, {x}, {x}",
            f"v_or_b32_e32 {x}, 0x400000, {x}",
            f"v_cndmask_b32_e64 {x}, {t}, {x}, {m}"]


def rw_pk_add_bf16(body, idx, ctx, n):
    """global_atomic_pk_add_bf16 (gfx940-only) -> exec-masked global_atomic_cmpswap retry loop.

    Per active lane: old = load(addr); loop { new.lo = rne(f32(old.lo)+f32(d.lo)); new.hi likewise;
    ret = cmpswap(addr, {new, old}); lanes with ret==old are done (dropped from exec); old = ret }.
    SCC is never written (exec is set from a VALU compare via s_mov), so SCC liveness is not needed.
    Temps: 4 VGPRs (pair N,N+1 even-aligned = cmpswap {src,cmp}; R = returned value / f32 temp; B = f32
    temp) and 2 SGPR pairs (exec save, compare mask), all dead after the site and not read by it.
    A full s_waitcnt drain first, so no in-flight load can later land in a 'dead' temp."""
    mn, ops, mods = split_ops(body)
    if any(m in ("glc", "sc0") for m in mods):
        raise Refuse(f"pk_add_bf16 returning form: {body}")
    if len(ops) != 3:
        raise Refuse(f"pk_add_bf16 form not handled: {body}")
    V, D, S = ops
    vr, dr = reg_of(V), reg_of(D)
    if not vr or vr[0] != "v" or not dr or dr[0] != "v" or dr[1] != dr[2]:
        raise Refuse(f"pk_add_bf16 operands (need VGPR addr/data): {body}")
    if S == "off" or len(sgprs_of(S)) != 2:
        raise Refuse(f"pk_add_bf16 without SGPR base: {body}")
    keep = " ".join(m for m in mods if m not in ("glc", "sc0"))
    keep = (" " + keep) if keep else ""
    ex = _units_of_ops(ops)
    base, how = _dead_vgprs(ctx, idx, 4, ex)
    pre, post = [], []
    if base is None:
        b = _borrow_vgprs(ctx, idx, 4, ex)
        if b is None:
            raise Refuse(f"pk_add_bf16: no dead VGPR quad and no 4 dead AGPRs to park a borrowed one: {body}")
        base, pre, post, how = b
        ex = ex | getattr(ctx, "spill_sgpr_units", set())
    pairs = _dead_sgpr_pairs(ctx, idx, 2, ex)
    if pairs is None:
        raise Refuse(f"pk_add_bf16: no two dead SGPR pairs: {body}")
    m, sv = pairs[0], pairs[1]
    N, C, R, B = f"v{base}", f"v{base + 1}", f"v{base + 2}", f"v{base + 3}"
    L = f".Lcas_{n}"
    seq = [DRAIN] + pre + [f"s_mov_b64 {sv}, exec", f"global_load_dword {C}, {V}, {S}{keep}", "s_waitcnt vmcnt(0)",
           ("label", L),
           f"v_lshlrev_b32_e32 {R}, 16, {C}", f"v_lshlrev_b32_e32 {B}, 16, {D}", f"v_add_f32_e32 {R}, {R}, {B}"]
    seq += _bf16_round(R, B, m)
    seq += [f"v_lshrrev_b32_e32 {N}, 16, {R}",
            f"v_and_b32_e32 {R}, 0xffff0000, {C}", f"v_and_b32_e32 {B}, 0xffff0000, {D}",
            f"v_add_f32_e32 {R}, {R}, {B}"]
    seq += _bf16_round(R, B, m)
    seq += [f"v_and_b32_e32 {R}, 0xffff0000, {R}", f"v_or_b32_e32 {N}, {R}, {N}",
            f"global_atomic_cmpswap {R}, {V}, v[{base}:{base + 1}], {S}{keep} glc", "s_waitcnt vmcnt(0)",
            f"v_cmp_ne_u32_e64 {m}, {R}, {C}", f"v_mov_b32_e32 {C}, {R}",
            f"s_mov_b64 exec, {m}", f"s_cbranch_execnz {L}", f"s_mov_b64 exec, {sv}"] + post
    return seq, f"cas,v{base}..v{base + 3}({how}),mask={m},save={sv}"


def rw_load_lds_dword(body, idx, ctx):
    """global_load_lds_dword (gfx940-only) -> global_load_dword + ds_write_b32 (synchronous).

    gfx940 semantics (LLVM llvm.amdgcn.global.load.lds: imm offset applies to both addresses):
    LDS_ADDR = M0 + inst_offset + 4*lane, global addr = vaddr + inst_offset. The requested
    buffer_load_dword ... lds needs a uniform base, but every site here has a 64-bit per-lane vaddr with
    saddr=off, so the LDS DMA is emulated: VMEM load into a dead VGPR, ds_write_b32 at
    (M0 & 0xffff) + 4*lane + offset, then lgkmcnt(0) so the kernel's own vmcnt-wait + s_barrier still orders
    the LDS data before readers in other waves."""
    mn, ops, mods = split_ops(body)
    if len(ops) != 2:
        raise Refuse(f"global_load_lds_dword form not handled: {body}")
    V, S = ops
    off, rest = 0, []
    for x in mods:
        if x.startswith("offset:"):
            off = int(x.split(":")[1], 0)
        else:
            rest.append(x)
    if off < 0 or off > 65535:
        raise Refuse(f"global_load_lds_dword offset not a ds offset: {body}")
    exu = _units_of_ops(ops)
    base, how = _dead_vgprs(ctx, idx, 2, exu)
    pre, post = [], []
    if base is None:
        b = _borrow_vgprs(ctx, idx, 2, exu)
        if b is None:
            raise Refuse(f"global_load_lds_dword: no dead VGPR pair and no two dead AGPRs: {body}")
        base, pre, post, how = b
    T, A = f"v{base}", f"v{base + 1}"
    o = f" offset:{off}" if off else ""
    r = (" " + " ".join(rest)) if rest else ""
    seq = [DRAIN] + pre + [f"v_mov_b32_e32 {T}, 0xffff", f"v_and_b32_e32 {T}, m0, {T}",
           f"v_mbcnt_lo_u32_b32 {A}, -1, 0", f"v_mbcnt_hi_u32_b32 {A}, -1, {A}",
           f"v_lshl_add_u32 {A}, {A}, 2, {T}",
           f"global_load_dword {T}, {V}, {S}{o}{r}", "s_waitcnt vmcnt(0)",
           f"ds_write_b32 {A}, {T}{o}", "s_waitcnt lgkmcnt(0)"] + post
    return seq, f"lds-emul,v{base}..v{base + 1}({how})"


# ----------------------------------------------------------------------------- KD + metadata
def preload_prologue(raw, kd):
    """gfx940 kernarg preload -> explicit s_load prologue (gfx90a has no preload).

    LLVM emits a 256-byte compat block at the kernel entry (s_trap 2 + s_nop 0 padding) that preload-capable
    firmware skips. The preloaded dwords live in the user SGPRs right after the other enabled user SGPRs.
    We replace the compat block with s_load_dword* of those dwords from the kernarg pointer + s_waitcnt, and
    pin .amdhsa_user_sgpr_count so the workgroup-id SGPRs stay where the code reads them."""
    n = int(kd.get("user_sgpr_kernarg_preload_length", "0"))
    if n == 0:
        return None
    off = int(kd.get("user_sgpr_kernarg_preload_offset", "0"))
    insts = [i for i, r in enumerate(raw) if r[0] == "inst"]
    first = raw[insts[0]]
    if first[1] != "s_trap 2":
        raise Refuse("kernarg preload without s_trap compat prologue")
    entry = first[2]
    pro = [i for i in insts if raw[i][2] < entry + 256]
    if any(raw[i][1] not in ("s_trap 2", "s_nop 0") for i in pro) or any(raw[j][0] == "label" for j in range(pro[0] + 1, pro[-1] + 1)):
        raise Refuse("kernarg preload compat block is not s_trap+s_nop")
    if kd.get("user_sgpr_kernarg_segment_ptr") != "1":
        raise Refuse("kernarg preload without kernarg segment ptr")
    base = 0
    for k, w in (("user_sgpr_private_segment_buffer", 4), ("user_sgpr_dispatch_ptr", 2), ("user_sgpr_queue_ptr", 2)):
        base += w * int(kd.get(k, "0"))
    kptr = base
    base += 2
    for k, w in (("user_sgpr_dispatch_id", 2), ("user_sgpr_flat_scratch_init", 2), ("user_sgpr_private_segment_size", 1)):
        base += w * int(kd.get(k, "0"))
    if base + n > 16:
        raise Refuse("preload exceeds 16 user SGPRs")
    loads, r, o, rem = [], base, off * 4, n
    while rem:
        for sz in (8, 4, 2, 1):
            if sz <= rem and r % min(sz, 4) == 0:
                break
        dst = f"s{r}" if sz == 1 else f"s[{r}:{r + sz - 1}]"
        mn = "s_load_dword" if sz == 1 else f"s_load_dwordx{sz}"
        loads.append(f"{mn} {dst}, s[{kptr}:{kptr + 1}], 0x{o:x}")
        r, o, rem = r + sz, o + 4 * sz, rem - sz
    loads.append("s_waitcnt lgkmcnt(0)")
    new = list(raw[:pro[0]]) + [("inst", l, None, []) for l in loads] + list(raw[pro[-1] + 1:])
    return {"raw": new, "user_sgpr_count": base + n, "loads": loads}


INV_BR = {"s_cbranch_scc0": "s_cbranch_scc1", "s_cbranch_scc1": "s_cbranch_scc0", "s_cbranch_vccz": "s_cbranch_vccnz",
          "s_cbranch_vccnz": "s_cbranch_vccz", "s_cbranch_execz": "s_cbranch_execnz", "s_cbranch_execnz": "s_cbranch_execz"}
SCC_READERS = ("s_cbranch_scc", "s_cselect", "s_addc", "s_subb", "s_cmov")
SCC_KEEPERS = ("s_mov", "s_movk", "s_nop", "s_waitcnt", "s_mul_i32", "s_load", "s_buffer_load", "s_barrier",
               "s_setprio", "s_cmov", "s_getpc", "s_sleep")


def relax_branch(items, i, n):
    """Out-of-simm16 branch -> (inverted short branch over) s_getpc / s_add / s_addc / s_setpc, as LLVM's
    branch relaxation does. Runs AFTER the hazard padding: it only adds SALU instructions, so every MFMA
    hazard distance can only grow. Needs an SGPR pair dead at the target and SCC dead at the target
    (s_add_u32 clobbers SCC)."""
    it = items[i]
    mn, ops, _ = split_ops(it[1])
    tgt = ops[0]
    ti = next((j for j, x in enumerate(items) if x[0] == "label" and x[1] == tgt), None)
    if ti is None:
        raise Refuse(f"far branch target {tgt} not found")
    for x in items[ti + 1:]:  # SCC must be written before it is read at the target
        if x[0] == "label":
            continue
        xm = x[1].split()[0]
        if xm.startswith(SCC_READERS):
            raise Refuse(f"far branch to {tgt}: SCC live at target")
        if xm.startswith("s_") and not xm.startswith(SCC_KEEPERS) and not xm.startswith(("s_cbranch", "s_branch")):
            break
        if xm.startswith(("s_cbranch", "s_branch", "s_endpgm", "s_setpc")):
            raise Refuse(f"far branch to {tgt}: cannot prove SCC dead")
    live = live_after([x for x in items])[i]
    nxt = next(j for j in range(ti + 1, len(items)) if items[j][0] == "inst")
    live = live | live_after(items)[nxt] | set().union(*[set(def_use(items[nxt][1])[1])])
    pair = next((b for b in range(0, 102, 2) if ("s", b) not in live and ("s", b + 1) not in live), None)
    if pair is None:
        raise Refuse(f"far branch to {tgt}: no dead SGPR pair")
    sp = f"s[{pair}:{pair + 1}]"
    pc, skip = f".Lfarpc_{n}", f".Lfarskip_{n}"
    tag = f"FARBRANCH {it[1]} via {sp}"
    seq = []
    if mn != "s_branch":
        if mn not in INV_BR:
            raise Refuse(f"far branch {mn} has no inverse")
        b = f"{INV_BR[mn]} {skip}"
        seq.append(("inst", b, it[2], it[3], classify(b), tag))
    body = [f"s_getpc_b64 {sp}", None, f"s_add_u32 s{pair}, s{pair}, ({tgt}-{pc})&4294967295",
            f"s_addc_u32 s{pair + 1}, s{pair + 1}, ({tgt}-{pc})>>32", f"s_setpc_b64 {sp}"]
    for j, b in enumerate(body):
        if b is None:
            seq.append(("label", pc))
            continue
        addr = it[2] if (mn == "s_branch" and j == 0) else None
        seq.append(("inst", b, addr, it[3] if addr else [], {"mn": b.split()[0], "kind": "salu", "ws": 1}, "FARBRANCH"))
    if mn != "s_branch":
        seq.append(("label", skip))
    return seq, {"branch": it[1], "pair": sp}


def original_metadata(co):
    t = run([f"{LLVM}/llvm-readelf", "--notes", co]).stdout
    m = re.search(r"AMDGPU Metadata:\n(.*?\n\.\.\.)", t, re.S)
    if not m:
        raise Refuse("no AMDGPU metadata note")
    lines = m.group(1).splitlines()
    lines[0] = lines[0].strip()
    return "\n".join(lines)


def original_kd(co):
    t = run([f"{LLVM}/llvm-objdump", "-D", "-j", ".rodata", "--mcpu=gfx942", co]).stdout
    return re.findall(r"^\t\.amdhsa_(\S+) (\S+)$", t, re.M)


def max_sgpr(texts):
    mx = -1
    for t in texts:
        for a, b in re.findall(r"\bs\[(\d+):(\d+)\]", t):
            mx = max(mx, int(b))
        for a in re.findall(r"\bs(\d+)\b", t):
            mx = max(mx, int(a))
    return mx


def max_reg(texts, f):
    mx = -1
    for t in texts:
        for a, b in re.findall(rf"\b{f}\[(\d+):(\d+)\]", t):
            mx = max(mx, int(b))
        for a in re.findall(rf"\b{f}(\d+)\b", t):
            mx = max(mx, int(a))
    return mx


def r4(x):
    return ((x + 3) // 4) * 4


def gfx90a_kd(orig_kd, sgpr_max, vmax, amax, meta_kernarg, user_sgpr_count=None):
    kd = collections.OrderedDict(orig_kd)
    changes = {}

    def setv(k, v):
        old = kd.get(k)
        if str(old) != str(v):
            changes[k] = {"gfx942": old, "gfx90a": str(v)}
        kd[k] = str(v)

    if amax < 0:
        # No AGPR: all VGPRs architectural. H5 rule: accum_offset = next_free_vgpr = VGPRs named (x4).
        # (keep the gfx942 total allocation: same occupancy as shipped; H5 used 256 = gfx942 next_free_vgpr)
        nv = max(4, r4(vmax + 1), min(256, int(kd.get("next_free_vgpr", 0))))
        if nv > 256:
            raise Refuse(f"needs {nv} arch VGPRs")
        setv("accum_offset", nv)
        setv("next_free_vgpr", nv)
    else:
        ao = int(kd["accum_offset"])
        if vmax + 1 > ao:
            ao = r4(vmax + 1)  # arch VGPRs past the gfx942 accum_offset: grow, AGPRs move up
        total = ao + r4(amax + 1)
        if total > 512 or ao > 256:
            raise Refuse(f"unified register file overflow: accum_offset {ao} + {r4(amax + 1)} AGPR > 512")
        setv("accum_offset", ao)
        setv("next_free_vgpr", max(total, int(kd["next_free_vgpr"])) if ao == int(dict(orig_kd)["accum_offset"]) else total)
    if meta_kernarg is not None:
        setv("kernarg_size", meta_kernarg)
    if sgpr_max >= 102:
        raise Refuse(f"code names s{sgpr_max} (gfx90a addressable SGPRs end at s101)")
    if int(kd.get("next_free_sgpr", 0)) > 102:
        setv("next_free_sgpr", 102)
    elif int(kd.get("next_free_sgpr", 0)) <= sgpr_max:
        setv("next_free_sgpr", sgpr_max + 1)
    if "enable_private_segment" in kd:
        v = kd.pop("enable_private_segment")
        kd["system_sgpr_private_segment_wavefront_offset"] = v
        kd["user_sgpr_private_segment_buffer"] = "0"
        kd["user_sgpr_flat_scratch_init"] = "0"
        changes["enable_private_segment"] = {"gfx942": v, "gfx90a":
            f"system_sgpr_private_segment_wavefront_offset {v}; user_sgpr_private_segment_buffer 0; user_sgpr_flat_scratch_init 0"}
        if v != "0" or int(kd.get("private_segment_fixed_size", 0)) != 0:
            raise Refuse("kernel uses scratch (gfx940 architected flat scratch has no gfx90a spelling here)")
    if "reserve_xnack_mask" in kd:
        changes["reserve_xnack_mask"] = {"gfx942": kd.pop("reserve_xnack_mask"), "gfx90a": "(derived from target id)"}
    for k in ("user_sgpr_kernarg_preload_length", "user_sgpr_kernarg_preload_offset"):
        if k in kd:
            if kd[k] != "0" and user_sgpr_count is None:
                raise Refuse("kernarg preload (gfx940-only) without the standard s_trap prologue")
            changes[k] = {"gfx942": kd.pop(k), "gfx90a": "(dropped; s_load prologue emitted)"}
    if user_sgpr_count is not None:
        # keep the workgroup-id SGPRs where the gfx940 code expects them (after the preloaded block)
        setv("user_sgpr_count", user_sgpr_count)
    return list(kd.items()), changes


def scratch_kd(kd, sc):
    if not sc:
        return kd
    d = collections.OrderedDict(kd)
    d["user_sgpr_private_segment_buffer"] = "1"
    d["system_sgpr_private_segment_wavefront_offset"] = "1"
    d["private_segment_fixed_size"] = str(4 * SCRATCH_SLOT_DWORDS)
    d["user_sgpr_count"] = str(sc["nuser_old"] + 4)
    return list(d.items())


def fix_metadata(y, nv, na):
    y = y.replace("amdhsa.version:\n  - 1\n  - 0", "amdhsa.version:\n  - 1\n  - 2")
    y = y.replace("amdhsa.version:\n  - 1\n  - 1", "amdhsa.version:\n  - 1\n  - 2")
    if ".agpr_count" not in y:
        y = y.replace("    .group_segment_fixed_size:", f"    .agpr_count:     {na}\n    .group_segment_fixed_size:", 1)
    else:
        y = re.sub(r"(\.agpr_count:\s+)\d+", lambda m: m.group(1) + str(na), y)
    if "amdhsa.target" not in y:
        y = y.replace("amdhsa.kernels:", "amdhsa.target:   amdgcn-amd-amdhsa--gfx90a\namdhsa.kernels:", 1)
    else:
        y = re.sub(r"amdhsa\.target:\s+\S+", "amdhsa.target:   amdgcn-amd-amdhsa--gfx90a", y)
    if nv is not None:
        y = re.sub(r"(\.vgpr_count:\s+)\d+", lambda m: m.group(1) + str(nv), y)
    return y


def emit_asm(kernel, items, kd, meta_yaml, path):
    L = ['\t.amdgcn_target "amdgcn-amd-amdhsa--gfx90a"', "\t.text", f"\t.globl {kernel}", "\t.p2align 8",
         f"\t.type {kernel},@function", f"{kernel}:"]
    for it in items:
        if it[0] == "label":
            if it[1] != "__entry":
                L.append(f"{it[1]}:")
        else:
            L.append(f"\t{it[1]}" + (f"\t; {it[5]}" if len(it) > 5 and it[5] else ""))
    L += [".Lfunc_end0:", f"\t.size {kernel}, .Lfunc_end0-{kernel}", "", "\t.rodata", "\t.p2align 6",
          f".amdhsa_kernel {kernel}"]
    L += [f"\t.amdhsa_{k} {v}" for k, v in kd]
    L += [".end_amdhsa_kernel", "", "\t.amdgpu_metadata", meta_yaml.rstrip(), "\t.end_amdgpu_metadata"]
    open(path, "w").write("\n".join(L) + "\n")


def norm_inst(s):
    return re.sub(r"\s+", " ", s.strip())


# ----------------------------------------------------------------------------- pipeline
def retarget(co, out_dir, audit_dir=None, base=None):
    """Returns report dict; raises Refuse(reason) on anything not mechanically retargetable."""
    base = base or os.path.splitext(os.path.basename(co))[0]
    knames = set(re.findall(r"(\S+)\.kd$", run([f"{LLVM}/llvm-readelf", "-s", "--wide", co]).stdout, re.M))
    nk = len(knames)
    if nk != 1:
        raise Refuse(f"{nk} kernel descriptors in one .co")
    kernel, raw = disassemble(co, "gfx942", next(iter(knames)))
    okd = original_kd(co)
    preload = preload_prologue(raw, dict(okd))
    if preload:
        raw = preload["raw"]
    txt = "\n".join(r[1] for r in raw if r[0] == "inst")
    fp8 = sorted({r[1].split()[0] for r in raw if r[0] == "inst" and FP8_RE.search(r[1].split()[0])})
    if fp8:
        raise Refuse("FP8: " + " ".join(fp8))
    agpr_use = bool(re.search(r"\ba\[\d+:\d+\]|\ba\d+\b|accvgpr", txt))
    has_gpr_idx = "s_set_gpr_idx_on" in txt

    # orig (gfx942 spelling, for the checker self-check)
    orig942 = []
    for r in raw:
        orig942.append(r if r[0] == "label" else ("inst", r[1], r[2], r[3], classify(r[1]), ""))

    # pass 1: simple per-instruction rewrites (no liveness needed)
    items, counts = [], collections.Counter()
    for r in raw:
        if r[0] == "label":
            items.append(r)
            continue
        body = r[1]
        nb = body
        for a, b in RENAMES:
            nb = re.sub(a, b, nb)
        if nb != body:
            counts["R1_rename"] += 1
        cb = cache_bits(nb)
        if cb != nb:
            counts["R6_cachebits"] += 1
        nb = cb
        mn = nb.split()[0]
        if mn in I8SPLIT:
            outs, tag = rw_i8split(nb), "I8SPLIT"
            counts["R2_i8split"] += 1
        elif mn.startswith("v_mov_b64"):
            outs, tag = rw_mov_b64(nb), "MOVB64"
            counts["R3_movb64"] += 1
        else:
            outs, tag = [nb], ""
        for j, o in enumerate(outs):
            items.append(("inst", o, r[2] if j == 0 else None, r[3] if j == 0 else [], classify(o),
                          (f"{tag} {body}" if tag and j == 0 else tag)))

    # pass 2: liveness-aware rewrites, one site at a time (liveness recomputed after each edit)
    vmax0 = max_reg([it[1] for it in items if it[0] == "inst"], "v")
    ctx = Ctx(items, has_gpr_idx, vmax0, 256)
    ctx.okd, ctx.preload = dict(okd), bool(preload)
    rewrites = []
    while True:
        site = next((i for i, it in enumerate(items) if it[0] == "inst"
                     and it[4]["mn"] in ("v_lshl_add_u64", "v_fmamk_f32", "v_fmamk_f32_e32",
                                         "global_atomic_pk_add_bf16", "global_load_lds_dword")), None)
        if site is None:
            break
        it = items[site]
        if it[4]["mn"] == "global_atomic_pk_add_bf16":
            seq, how = rw_pk_add_bf16(it[1], site, ctx, counts["R7_pkaddbf16_cas"])
            counts["R7_pkaddbf16_cas"] += 1
            tag = "PKADDBF16CAS"
        elif it[4]["mn"] == "global_load_lds_dword":
            seq, how = rw_load_lds_dword(it[1], site, ctx)
            counts["R8_loadlds_emul"] += 1
            tag = "LOADLDS"
        elif it[4]["mn"] == "v_lshl_add_u64":
            seq, how = rw_lshl_add_u64(it[1], site, ctx)
            counts["R4_lshl_add_u64"] += 1
            tag = "LSHLADD64"
        else:
            seq, how = rw_fmamk(it[1], site, ctx)
            counts["R5_fmamk"] += 1
            tag = "FMAMK"
        rewrites.append({"orig_addr": hex(it[2]) if it[2] is not None else None, "inst": it[1], "how": how,
                         "seq": [o if isinstance(o, str) else o[1] + ":" for o in seq]})
        # the original instruction's address/encoding ride on the FIRST emitted instruction (labels carry none)
        new, first = [], True
        for o in seq:
            if not isinstance(o, str):
                new.append(o)
                continue
            new.append(("inst", o, it[2] if first else None, it[3] if first else [], classify(o),
                        f"{tag} {it[1]} [{how}]" if first else tag))
            first = False
        items[site:site + 1] = new
        ctx.items = items
        ctx.after = None

    if ctx.scratch:
        ent = next(i for i, it in enumerate(items) if it[0] == "label" and it[1] == "__entry")
        items[ent + 1:ent + 1] = [("inst", o, None, [], classify(o), "SCRATCHPRO") for o in scratch_prologue(ctx.scratch)]
        counts["R9_scratch_prologue"] = 1

    # hazard: self-check gfx942 on the original, gfx90a before/after padding
    _, def942, it942 = audit(orig942, "gfx942")
    blocks, def90a_pre, _ = audit(items, "gfx90a")
    insertions, rounds = [], 0
    while True:
        rounds += 1
        _, defs, _ = audit(items, "gfx90a")
        if not defs:
            break
        seen, todo = set(), []
        for dfc in defs:
            if dfc["block"] in seen:
                continue
            seen.add(dfc["block"])
            todo.append(dfc)
        for dfc in sorted(todo, key=lambda x: -x["idx"]):
            idx = dfc["idx"]
            oa = items[idx][2]
            tag = f"HAZPAD gfx90a +{dfc['need']}ws before 0x{oa:x}" if oa is not None else f"HAZPAD gfx90a +{dfc['need']}ws"
            items[idx:idx] = nop_items(dfc["need"], tag)
            insertions.append({"round": rounds, "before_orig_addr": hex(oa) if oa else None,
                               "before_inst": dfc["inst"], "wait_states_added": dfc["need"],
                               "first_in_block_multi_pred": dfc["first_in_block"] and dfc["multi_pred"],
                               "findings": dfc["findings"]})
        if rounds > 500:
            raise Refuse("padding did not converge")
    _, post, _ = audit(items, "gfx90a")
    assert not post

    # KD + emit + assemble
    texts = [it[1] for it in items if it[0] == "inst"]
    smax = max_sgpr(texts)
    vmax, amax = max_reg(texts, "v"), max_reg(texts, "a")
    meta = original_metadata(co)
    mk = re.search(r"\.kernarg_segment_size:\s+(\d+)", meta)
    kd, kd_changes = gfx90a_kd(okd, smax, vmax, amax, mk.group(1) if mk else None,
                               preload["user_sgpr_count"] if preload else None)
    kd = scratch_kd(kd, ctx.scratch)
    kdd = dict(kd)
    meta = fix_metadata(meta, None, r4(amax + 1) if amax >= 0 else 0)
    if ctx.scratch:
        meta = re.sub(r"(\.private_segment_fixed_size:\s+)\d+", lambda m: m.group(1) + str(4 * SCRATCH_SLOT_DWORDS), meta)
    os.makedirs(out_dir, exist_ok=True)
    s_path = os.path.join(out_dir, f"{base}.gfx90a.s")
    o_path = os.path.join(out_dir, f"{base}.gfx90a.o")
    co_path = os.path.join(out_dir, f"{base}.gfx90a.co")
    relaxed = []
    for attempt in range(4):
        emit_asm(kernel, items, kd, meta, s_path)
        p = subprocess.run([f"{LLVM}/llvm-mc", "-triple", "amdgcn-amd-amdhsa", "-mcpu=gfx90a",
                            "--amdhsa-code-object-version=5", "-filetype=obj", "-o", o_path, s_path],
                           capture_output=True, text=True)
        far = [int(m.group(1)) for m in re.finditer(r":(\d+):\d+: error: branch size exceeds simm16", p.stderr)]
        if p.returncode == 0 or not far or len(far) != len(re.findall(r"error:", p.stderr)):
            break
        line2idx, ln = {}, 6
        for i, it in enumerate(items):
            if it[0] == "label" and it[1] == "__entry":
                continue
            ln += 1
            line2idx[ln] = i
        for ln_ in sorted(far, reverse=True):
            i = line2idx[ln_]
            seq, info = relax_branch(items, i, len(relaxed))
            relaxed.append(info)
            items[i:i + 1] = seq
        kd, _ = gfx90a_kd(okd, max_sgpr([it[1] for it in items if it[0] == "inst"]), vmax, amax,
                          mk.group(1) if mk else None, preload["user_sgpr_count"] if preload else None)
        kd = scratch_kd(kd, ctx.scratch)
    if p.returncode != 0:
        bad = collections.Counter()
        lines = open(s_path).read().splitlines()
        for m in re.finditer(r":(\d+):\d+: error: (.*)", p.stderr):
            ln = lines[int(m.group(1)) - 1].strip()
            bad[(ln.split()[0] if ln else "?") + " (" + m.group(2)[:60] + ")"] += 1
        raise Refuse("llvm-mc: " + "; ".join(f"{k} x{v}" for k, v in bad.most_common(6)))
    run([f"{LLVM}/ld.lld", "-shared", "-o", co_path, o_path])

    # verify
    _, newraw = disassemble(co_path, "gfx90a", kernel)
    new_insts = [r for r in newraw if r[0] == "inst"]
    src_insts = [it for it in items if it[0] == "inst"]
    if len(new_insts) < len(src_insts):
        raise Refuse("objdump round-trip lost instructions")
    rt_mismatch = [(a[1], b[1]) for a, b in zip(src_insts, new_insts) if norm_inst(a[1]) != norm_inst(b[1])]
    extra = [r[1] for r in new_insts[len(src_insts):]]
    if any(not e.startswith("s_code_end") and e != "s_nop 0" for e in extra) and extra:
        pass
    enc_diff, j = collections.Counter(), 0
    orig_insts = [r for r in raw if r[0] == "inst" and r[2] is not None]
    for it, nw in zip(src_insts, new_insts):
        if it[2] is None:
            continue
        o = orig_insts[j]; j += 1
        assert o[2] == it[2]
        if o[3] != nw[3]:
            enc_diff[it[1].split()[0]] += 1
    assert j == len(orig_insts)
    notes = run([f"{LLVM}/llvm-readelf", "--notes", co_path]).stdout
    hdr = run([f"{LLVM}/llvm-readelf", "-h", "-s", co_path]).stdout
    rep = {
        "input_co": co, "kernel": kernel, "agpr_use_in_code": agpr_use, "gpr_idx_mode": has_gpr_idx,
        "n_inst_orig": len(orig_insts), "n_inst_gfx90a": len(src_insts),
        "rewrite_counts": dict(counts), "liveness_rewrites": rewrites, "vgprs_grown_at": ctx.grown,
        "cfg": {"blocks": len(blocks), "edges": sum(len(b["succ"]) for b in blocks),
                "back_edges": [(bi, s) for bi, b in enumerate(blocks) for s in b["succ"] if s <= bi]},
        "checker_selfcheck_gfx942_rules_on_original": {
            "deficits": len(def942), "dataflow_iters": it942, "by_rule": rule_hist(def942),
            "mfma_rule_deficits_with_branch_or_barrier_in_window": sum(
                1 for x in def942 if any(f["rule"] in ("READ", "WAW", "WAR", "AB", "COVL") for f in x["findings"])
                and cf_in_window(orig942, x["idx"])),
            "mfma_rule_deficits_total": sum(
                1 for x in def942 if any(f["rule"] in ("READ", "WAW", "WAR", "AB", "COVL") for f in x["findings"]))},
        "gfx90a_deficits_by_rule_before_padding": rule_hist(def90a_pre),
        "gfx90a_deficits_before_padding": len(def90a_pre),
        "padding_rounds": rounds, "n_insertions": len(insertions), "insertions": insertions,
        "s_nop_wait_states_added": sum(i["wait_states_added"] for i in insertions),
        "gfx90a_deficits_after_padding": len(post), "kd_changes": kd_changes, "kd": kdd,
        "max_sgpr_named_in_code": smax, "max_vgpr": vmax, "max_agpr": amax,
        "n_roundtrip_mismatches": len(rt_mismatch), "roundtrip_text_mismatches": rt_mismatch[:20],
        "trailing_extra_insts": extra, "encoding_diff_vs_gfx942_by_mnemonic": dict(enc_diff),
        "metadata_target": re.findall(r"amdhsa\.target:\s+(\S+)", notes),
        "elf_flags": re.findall(r"Flags:\s+(.*)", hdr), "outputs": {"s": s_path, "co": co_path},
    }
    if audit_dir:
        os.makedirs(audit_dir, exist_ok=True)
        json.dump(rep, open(os.path.join(audit_dir, f"{base}.audit.json"), "w"), indent=1)
        open(os.path.join(audit_dir, f"{base}.gfx90a.dis.txt"), "w").write(
            "\n".join((r[1] if r[0] == "inst" else f"<{r[1]}>:") for r in newraw) + "\n")
        open(os.path.join(audit_dir, f"{base}.gfx90a.notes.txt"), "w").write(notes)
    return rep


def _batch_one(args):
    root, rel, out_dir = args
    base = rel.replace("/", "__")[:-3]
    try:
        rep = retarget(os.path.join(root, rel), out_dir, None, base)
        rc = rep["rewrite_counts"]
        note = ",".join(f"{k}={v}" for k, v in sorted(rc.items()))
        return (rel, "yes", f"pads={rep['n_insertions']}/{rep['s_nop_wait_states_added']}ws rt_mismatch={rep['n_roundtrip_mismatches']}"
                + (f" grownV={rep['vgprs_grown_at']}" if rep["vgprs_grown_at"] else "") + (" " + note if note else ""))
    except Refuse as e:
        return (rel, "no", str(e).replace("\n", " ")[:400])
    except Exception as e:  # noqa: BLE001 - table must record every file
        return (rel, "no", "INTERNAL " + repr(e)[:300] + " " + traceback.format_exc().splitlines()[-2].strip()[:100])


def main():
    mode = sys.argv[1]
    if mode == "one":
        out_dir, audit_dir, cos = sys.argv[2], sys.argv[3], sys.argv[4:]
        summary, rc = {}, 0
        for co in cos:
            try:
                rep = retarget(co, out_dir, audit_dir)
                summary[os.path.basename(co)] = {k: rep[k] for k in (
                    "rewrite_counts", "n_insertions", "s_nop_wait_states_added", "gfx90a_deficits_after_padding",
                    "n_roundtrip_mismatches", "encoding_diff_vs_gfx942_by_mnemonic", "kd_changes", "elf_flags")}
            except Refuse as e:
                summary[os.path.basename(co)] = {"REFUSED": str(e)}
                rc = 1
        print(json.dumps(summary, indent=1))
        sys.exit(rc)
    elif mode == "batch":
        from concurrent.futures import ProcessPoolExecutor
        root, lst, out_dir, table = sys.argv[2:6]
        jobs = int(sys.argv[6]) if len(sys.argv) > 6 else 8
        rels = [l.strip().lstrip("./") for l in open(lst) if l.strip()]
        rels = [r if r.endswith(".co") else r for r in rels]
        with ProcessPoolExecutor(jobs) as ex:
            res = list(ex.map(_batch_one, [(root, r, out_dir) for r in rels], chunksize=4))
        with open(table, "w") as f:
            f.write("file\tassembles\treason_or_notes\n")
            for r in res:
                f.write("\t".join(r) + "\n")
        print(collections.Counter(r[1] for r in res))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
