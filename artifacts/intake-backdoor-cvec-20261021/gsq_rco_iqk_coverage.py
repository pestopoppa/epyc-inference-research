#!/usr/bin/env python3
"""NEW-11: iqk coverage of ISTA-DASLab GSQ-RCO Q38FN builds (metadata only).

Reads ONLY the GGUF header (KV + tensor-info section, first few MB via HTTP
Range) of each shard-1 file; never touches tensor data.  Tensor byte sizes come
from dims x ggml block geometry and are cross-checked against offset deltas.
Classification mirrors ggml/src/ggml-cpu/iqk/iqk_dispatch.cpp @ ffc1bac82.
Usage: gsq_rco_iqk_coverage.py HEADER_BIN_DIR   (files hdr_<build>_s1.bin, hdr_<build>_s2.bin)
"""
import json, struct, sys
from collections import defaultdict

# id -> (name, block elems, bytes/block)   [ggml.h @ ffc1bac82]
T = {0:("F32",1,4),1:("F16",1,2),30:("BF16",1,2),2:("Q4_0",32,18),6:("Q5_0",32,22),8:("Q8_0",32,34),
     11:("Q3_K",256,110),12:("Q4_K",256,144),13:("Q5_K",256,176),14:("Q6_K",256,210),
     16:("IQ2_XXS",256,66),17:("IQ2_XS",256,74),22:("IQ2_S",256,82),18:("IQ3_XXS",256,98),
     21:("IQ3_S",256,110),20:("IQ4_NL",32,18),23:("IQ4_XS",256,136),29:("IQ1_M",256,56),42:("Q2_0",64,18)}
IQK = {"Q4_K","Q5_K","Q6_K","Q8_0","Q4_0","Q5_0","IQ2_XXS","IQ2_XS","IQ2_S","IQ3_XXS","IQ3_S","IQ4_XS"}

def cls(t, name, nd):
    if t in ("F32","F16","BF16"):
        return "float (not a quant matmul)"
    if t == "IQ3_XXS" and "_exps" in name:
        return "iqk decode only (MMID n_tokens==1); native for batched prefill"
    if t == "Q8_0":
        return "iqk (gated by iqk_q8_0_enabled)"
    if t in IQK:
        return "iqk"
    return "NATIVE (no iqk kernel)"

def parse(buf):
    p = 4
    def rd(fmt):
        nonlocal p
        v = struct.unpack_from(fmt, buf, p)[0]; p += struct.calcsize(fmt); return v
    def rs():
        nonlocal p
        n = rd("<Q"); s = buf[p:p+n].decode("utf-8","replace"); p += n; return s
    sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    def skip(t):
        nonlocal p
        if t in sz: p += sz[t]
        elif t == 8: rs()
        elif t == 9:
            et = rd("<I"); n = rd("<Q")
            if et in sz: p += sz[et]*n
            else:
                for _ in range(n): skip(et)
    assert buf[:4] == b"GGUF"
    rd("<I"); nt = rd("<Q"); nkv = rd("<Q")
    align = 32
    for _ in range(nkv):
        k = rs(); t = rd("<I")
        if k == "general.alignment": align = rd("<I")
        else: skip(t)
    ts = []
    for _ in range(nt):
        name = rs(); nd = rd("<I"); dims = [rd("<Q") for _ in range(nd)]
        ty = rd("<I"); off = rd("<Q"); ts.append((name, dims, ty, off))
    return ts, p, align

def sizes(ts, filesize=None):
    out = []
    for name, dims, ty, off in ts:
        n, blk, by = T[ty]
        ne = 1
        for d in dims: ne *= d
        assert ne % blk == 0, (name, dims, n)
        out.append((name, dims, n, ne // blk * by))
    return out

if __name__ == "__main__":
    d = sys.argv[1]
    res = {}
    for b in ("Q2_0", "IQ2_XS", "IQ3_XXS", "IQ3_S"):
        r = {}
        for sh in (("s1", "s2") if b == "Q2_0" else ("s1",)):
            buf = open(f"{d}/hdr_{b}_{sh}.bin", "rb").read()
            ts, _, _ = parse(buf)
            # offset-delta cross check (all but last tensor)
            so = sorted(ts, key=lambda x: x[3])
            sz = sizes(ts); bymap = {n: s for n, _, _, s in sz}
            bad = 0
            for i in range(len(so) - 1):
                delta = so[i+1][3] - so[i][3]
                if delta != bymap[so[i][0]] and delta - bymap[so[i][0]] >= 32: bad += 1
            agg = defaultdict(lambda: [0, 0]); fam = defaultdict(lambda: [0, 0]); cl = defaultdict(int)
            for name, dims, t, s in sz:
                agg[t][0] += 1; agg[t][1] += s
                kind = ("routed_exps" if "_exps" in name else "shared_exp" if "shexp" in name else
                        "per_layer_token_embd" if "per_layer_token_embd" in name else
                        "token_embd/output" if name in ("token_embd.weight", "output.weight") else
                        "attn/ssm/other")
                fam[kind][0] += 1; fam[kind][1] += s
                cl[cls(t, name, len(dims))] += s
            r[sh] = {"n_tensors": len(ts), "bytes": sum(s for *_, s in sz), "offset_mismatch": bad,
                     "by_type": {k: v for k, v in agg.items()}, "by_family": {k: v for k, v in fam.items()},
                     "by_class": dict(cl)}
            # routed-expert types detail
            rt = defaultdict(int)
            for name, dims, t, s in sz:
                if "_exps" in name: rt[t] += s
            r[sh]["routed_by_type"] = dict(rt)
        res[b] = r
    json.dump(res, open(f"{d}/coverage.json", "w"), indent=1)
    print("wrote coverage.json")
