#!/usr/bin/env python3
"""Refresh mandatory, small real-tile fixtures from the pinned local donor captures.
The generated fixtures are committed; tests never download or skip absent data.
Independent unpacking uses Python big integers, not C++ gather or index tables.
"""
import hashlib
import json
from pathlib import Path
import struct
import numpy as np

OUT = Path(__file__).parent / 'fixtures'

def codebook(cb):
    s = np.arange(65536, dtype=np.uint64)
    if cb == 'mul1':
        p = (s * 0x83dcd12d).astype(np.uint32)
        b = sum((p >> shift) & 255 for shift in (0,8,16,24))
        h = (b+0x6400).astype(np.uint16).view(np.float16).astype(np.float32)
        return (h*np.array([0x1eee],dtype=np.uint16).view(np.float16)[0] +
                np.array([0xc931],dtype=np.uint16).view(np.float16)[0]).astype(np.float16)
    p = ((s * 0xcbac1fed).astype(np.uint32) & 0x8fff8fff) ^ 0x3b603b60
    return ((p & 65535).astype(np.uint16).view(np.float16) +
            (p >> 16).astype(np.uint16).view(np.float16)).astype(np.float16)

def decode(words, bits, lut):
    raw = np.empty((128,128), np.float32)
    for kt in range(8):
        for nt in range(8):
            tile=words[kt,nt].reshape(-1,2)[:,::-1].reshape(-1)
            bitstring=0
            for word in tile: bitstring=(bitstring<<16)|int(word)
            doubled=(bitstring<<(256*bits))|bitstring
            for i in range(256):
                index=(doubled>>(256*bits-(i+1)*bits))&65535
                t,j=divmod(i,8)
                row=(t%4)*2+(j&1)+8*((j>>1)&1)
                col=t//4+8*(j>>2)
                raw[kt*16+row,nt*16+col]=lut[index]
    return raw

def left_hadamard(h, a):
    # Explicit ascending FP32 FMA: exact FP64 intermediate then one FP32 round.
    out=np.zeros_like(a)
    for j in range(128):
        out=(h[:,j,None].astype(np.float64)*a[j,None,:].astype(np.float64)+out.astype(np.float64)).astype(np.float32)
    return out

def main():
    OUT.mkdir(exist_ok=True)
    manifest={'schema':'exl3-cpu-real-slices-v1','scope':'real 128x128 tile submatrices; no full-model inference','fixtures':[]}
    for cb in ('mul1','mcg'):
        codebook(cb).astype('<f2').tofile(OUT/f'{cb}_lut.f16')
    specs=[]
    donor=Path('/mnt/raid0/llm/tmp/inf70/agents/x1')
    for bits, folder in ((3,'data3'),(4,'data')):
        specs.append((f'mul1_k{bits}',bits,'mul1',donor/folder/'L3_gate_proj',40,
                      'turboderp/Qwen3.8-Flash-Next-exl3',f'{bits}.05bpw_h{bits+2}_ng{bits+2}',None))
    base=Path('/workspace/tmp/intake-jev-exl3/exp-exl3/repo1/engines/tools/dione-evidence/payloads')
    specs.append(('mcg_k4',4,'mcg',base/'model.language_model.layers.3.mlp.experts.0.gate_proj.rank0',32,
                  '0xSero/GLM-5.3-Flash-EXL3-Q4','99cccdf0e8741715662c383828a9ea601990c125','.'))
    H=np.array([[1]],np.float32)
    while H.shape[0]<128:H=np.block([[H,H],[H,-H]])
    H*=np.float32(1/np.sqrt(128))
    for name,bits,cb,base,nt,repo,rev,sep in specs:
        sep=sep or '_'
        trellis=Path(str(base)+sep+'trellis.bin')
        # Read the first eight input tile rows; retain eight output tiles each.
        with trellis.open('rb') as f: slab=f.read(8*nt*16*bits*2)
        words=np.frombuffer(slab,dtype='<u2').reshape(8,nt,16*bits)[:,:8].copy()
        suh=np.fromfile(str(base)+sep+'suh.bin',dtype='<f2',count=128).astype(np.float32)
        svh=np.fromfile(str(base)+sep+'svh.bin',dtype='<f2',count=128).astype(np.float32)
        raw=decode(words,bits,codebook(cb))
        w=left_hadamard(H,raw).astype(np.float16).astype(np.float32)
        w=(w*suh[:,None]).astype(np.float16).astype(np.float32)
        w=left_hadamard(H,w.T).T.astype(np.float16).astype(np.float32)
        w=(w*svh[None,:]).astype(np.float16).astype(np.float32)
        path=OUT/f'{name}.bin'
        with path.open('wb') as f:
            f.write(struct.pack('<4I',0x334c5845,bits,int(cb=='mcg'),128))
            for a,dtype in ((words,'<u2'),(suh,'<f4'),(svh,'<f4'),(raw,'<f4'),(w,'<f4')):f.write(a.astype(dtype).tobytes())
        if cb=='mul1': rev=json.loads((OUT/f'{name}.provenance.json').read_text())['revision']
        manifest['fixtures'].append({'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
             'source':str(trellis),'model':repo,'revision_or_branch':rev,'source_prefix_sha256':hashlib.sha256(slab).hexdigest(),
             'slice':'input tiles [0:8], output tiles [0:8], scales [0:128]','source_output_tiles':nt,'prefix_bytes':len(slab)})
    manifest['files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob('*.bin'))}
    manifest['files'].update({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob('*.f16'))})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
