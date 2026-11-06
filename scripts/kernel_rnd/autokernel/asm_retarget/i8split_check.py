#!/usr/bin/env python3
"""Retarget trial: renames + split K-doubled int8 MFMA into two gfx90a MFMAs (lo/hi dword; exact since the k
permutation is identical for A and B), v_mov_b64 -> v_pk_mov_b32, then llvm-mc gfx90a. Stdout summary only."""
import collections, re, subprocess, sys
LLVM="/opt/rocm/llvm/bin"
REN=[(r"\bv_mfma_f32_32x32x8_bf16\b","v_mfma_f32_32x32x8bf16_1k"),(r"\bv_mfma_f32_16x16x16_bf16\b","v_mfma_f32_16x16x16bf16_1k"),
     (r"\bv_mfma_f32_32x32x8_f16\b","v_mfma_f32_32x32x8f16"),(r"\bv_mfma_f32_16x16x16_f16\b","v_mfma_f32_16x16x16f16")]
def cb(s):
    s=re.sub(r"\bsc0 sc1\b","glc",s); s=re.sub(r"\bsc0\b","glc",s); s=re.sub(r"\bsc1\b","",s); return re.sub(r"\bnt\b","slc",s)
def rng(r):
    m=re.match(r"([va])\[(\d+):(\d+)\]",r); return (m.group(1),int(m.group(2)),int(m.group(3))) if m else None
def split(l):
    m=re.match(r"(v_mfma_i32_(16x16x32|32x32x16)_i8)\s+(\S+),\s*(\S+),\s*(\S+),\s*(\S+)(.*)",l)
    if not m: return [l]
    new="v_mfma_i32_16x16x16i8" if m.group(2)=="16x16x32" else "v_mfma_i32_32x32x8i8"
    d,a,b,c,mod=m.group(3),m.group(4),m.group(5),m.group(6),m.group(7)
    A,B=rng(a),rng(b)
    if not A or not B: return ["UNSPLITTABLE "+l]
    lo=lambda R,o:f"{R[0]}{R[1]+o}"
    return [f"{new} {d}, {lo(A,0)}, {lo(B,0)}, {c}{mod}", f"{new} {d}, {lo(A,1)}, {lo(B,1)}, {d}{mod}"]
def movb64(l):
    m=re.match(r"v_mov_b64(?:_e32)?\s+(\S+),\s*(\S+)$",l)
    return f"v_pk_mov_b32 {m.group(1)}, {m.group(2)}, {m.group(2)} op_sel:[0,1]" if m else l
for f in sys.argv[1:]:
    t=subprocess.run([f"{LLVM}/llvm-objdump","-d","--mcpu=gfx942",f],capture_output=True,text=True).stdout
    lines=[x.split("//")[0].strip() for x in t.splitlines() if x.startswith("\t")]; lines=[x for x in lines if x]
    out=[]; ns=0
    for l in lines:
        for a,b in REN: l=re.sub(a,b,l)
        l=movb64(cb(l)); s=split(l); ns+=len(s)>1; out+=s
    p=subprocess.run([f"{LLVM}/llvm-mc","-triple=amdgcn-amd-amdhsa","-mcpu=gfx90a","-filetype=null"],input="\n".join(out)+"\n",capture_output=True,text=True)
    errs=re.findall(r"<stdin>:(\d+):\d+: error: (.*)",p.stderr); by=collections.Counter()
    for ln,msg in errs: by[(out[int(ln)-1].split()[0],msg[:50])]+=1
    print(f"== {f}: {len(lines)} lines, {ns} int8 MFMA split, {len(errs)} still fail")
    for (mn,msg),n in by.most_common(): print(f"   {n:5d} {mn:30s} {msg}")
